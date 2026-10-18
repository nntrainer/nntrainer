# Gemma4 on the HTP: the untied lm_head on the NPU

Companion to `gemma4-htp-user-guide.md` and `gemma4-htp-code-guide.md`.
This note records what the two `[HTP] lm_head:` commits changed, why, what
was measured and what is still open.

Updated 2026-10-18: the native TL2/affine pack directly registers its
prepacked affine head slices and needs no `.lmhead_wh` conversion cache
(section 7). Greedy ID-only decode skips unused vocabulary-sized FP32
copies; sampling and other logits consumers retain their output path.
The fixture timings below are historical. Current whole-model throughput,
memory changes and validation limits are in the
[TL2 guide](gemma4-htp-tl2-guide.md#current-validation-coverage).

## 1. The head before

The Gemma4 26B-A4B pack ships a tied head: `htp_tie_word_embeddings` reads
the Q4_0 embedding table (vocab 262144 x hidden 2816) as the head weight.
It ran split:

- **decode**: the resident DSP list's `LM_HEAD` op held the table as
  Q4M1 slices (16 of 16384 rows, 396 MiB attached in the arena) and
  computed the argmax on the DSP;
- **prefill**: the last row's logits came from the CPU (`lmhead_engine:
  cpu`), through a second, blocked copy of the table (396 MiB of host
  memory) built at load.

An untied head of the same Q4_0 type (`tie_word_embeddings: false`,
`lm_head` layer) went the same way: Q4M1 slices for decode, the CPU for the
prefill row, the host weight kept for it.

## 2. The head after: a QS4CX head as WH slices

A delivered pack can now carry the untied head quantized like the FC
projections -- `lmhead_dtype: QS4CX`, on disk N rows of K/2 int4 nibbles
(`q + 8`, even k in the low nibble) followed by N f32 per-column scales --
and the whole head then runs on the DSP:

| step | what happens | where |
|---|---|---|
| load, `Gemma4HtpMoECausalLM::load_weight` | the head's codes and scales are handed to the backend as the list's last weight (`add_decode_graph_qs4cx`) | `gemma4_htp_moe_causallm.cpp` |
| load, `bindQ4m1` | the head is cut into column slices, each converted (`htp_qs4cx_from_packed`: nibbles to K-major int8 plus column sums, `whPack`: the HMX 32 x 32 tile layout), placed in a 256 MiB arena chunk and registered as a u8i4 weight -- the path every QS4CX FC slice already takes | `htp_compute_ops.cpp` `bindLmHeadWh` |
| decode | the `LM_HEAD` op carries `HTP_GRAPH_FEED_WH` and its slices as parts; the DSP runs the FC's M = 1 call (`hexkl_mm_u8i4_fc_m1_run`) into the logits, then the soft-cap and the argmax as before | `hexkl_graph.c` `graph_op_lm_head` |
| prefill, last row | `LmHeadLayer` calls `lm_head_shared_fp32`, which runs the same handles through the new skel entry `lm_head_wh_f32` (the same kernel, outside the token driver) | `lm_head.cpp`, `nntr_hvx_mm_u8i4.c` |
| load, after the bind | the host copy of the head is released with the FC set (`keep_host` false): nothing on the CPU reads it any more | `htp_compute_ops.cpp` `dropHostFc` |

The prefill row and the decode tokens therefore see one head: the same
bytes through the same kernel, so the first generated token and the ones
after it are computed identically.

A tied head, or an untied Q4_0 head, is untouched and runs as in section 1.
The untied head's final norm and logit soft-cap stay where they were (the
`output_norm` layer and the `logit_softcapping` layer); the `LM_HEAD` op's
soft-cap is 0 for an untied head and the argmax it picks is unchanged by
the cap.

### 2.1 Slicing and the wire change

A registered u8i4 handle has to fit VTCM, because the heap registration
bakes its tiles there and the arena registration keeps the same bound
(`hexkl_weight_u8i4_check`). The head is 352 MiB of WH bytes, so it is
sliced:

- a slice holds at most 6 MiB of WH bytes (4448 columns at K = 2816; an
  8 MiB VTCM is assumed, the v79 / v81 phones);
- the slice count is bounded by `HTP_GRAPH_MAX_PARTS`, raised from 32 to
  64 for this vocabulary (64 slices of 4096 columns); the bind throws when
  a head would need more.

Raising the part limit widens the op record (`HTP_GRAPH_OP_WORDS` 80 to
144), so `HTP_GRAPH_VERSION` is now 3: a skel built before this refuses the
new list with `HTP_GRAPH_E_VERSION` instead of misreading it, and the
graph host check pins the new size.

### 2.2 The cache file

Converting the head costs 783 ms per load on the S26 Ultra. After the
first conversion the slices are written to `<model file>.lmhead_wh` beside
the model (a `.tmp` renamed at the end, so a half-written file never
matches), and every later load preads each slice straight into its arena
room:

```
header  (32 bytes): "NNTRLMWH", u32 version = 1, u32 K, u32 N,
                    u32 cols (the slice width), u64 hash of the N scales
per slice, in column order:
        WH bytes (K/32 x cols/32 tiles of 512 bytes),
        f32 scales [cols], i32 column sums [cols]
```

The match is the header: K, N, the slice width and the scales' hash. Codes
requantized to the same scales would pass it, so delete the file by hand
after re-quantizing the head. A directory that cannot be written just skips
the cache (one line on stderr) and converts every load.

## 3. Measured

Galaxy S26 Ultra `R3CY70LV96T` (SM-S948U, v81, 11 GB RAM), the untied 2-bit
pack (`QS2CX_WH` experts, `QS4CX` FC, `moe_cache_experts 16`), 48 generated
tokens, one run per cell, the same binary for both heads.

**The head under test is a fixture, not delivered weights**: the pack's own
Q4_0 embedding, dequantized and requantized per vocabulary row to QS4CX,
appended in place of the pack's Q4_0 head. It exercises the path; it says
nothing about accuracy (section 5).

| prompt | shipped Q4_0 untied head: prefill row on the CPU, decode on Q4M1 slices | QS4CX head: prefill row and decode on the NPU WH slices |
|---|---|---|
| 1028 tokens, ctx 1280 | prefill 1706 ms (603 TPS), decode 13.09 tok/s, fc+dense_ffn+lm_head 20.4 ms/token | prefill 1676 ms (613 TPS), decode 13.41 tok/s, 19.0 ms/token |
| 1986 tokens, ctx 2304 | prefill 3033 ms (655 TPS), decode 11.60 tok/s, 21.1 ms/token | prefill 3021 ms (657 TPS), decode 13.20 tok/s, 19.2 ms/token |

The decode gap at 1986 tokens is run-to-run MoE variance (1.23 against
0.98 ms per MoE op), not the head; the head's own share is the 1.2 to
1.9 ms/token drop in `fc+dense_ffn+lm_head`. The prefill gains about 30 ms:
the CPU head of the last row.

Load, the head alone: converted and placed 783 ms; with the cache written
in the same load 852 ms; read from the cache 124 ms.

Both heads produce a coherent summary of the sample prompt; their first 11
generated ids agree and then diverge, as two quantizations of the same head
do.

## 4. Running it on a phone

The documented layout (`GEMMA2_GPU_RUN.md`): binaries in
`/data/local/tmp/nntrainer/causallm/`, packs in
`/data/local/tmp/nntrainer/causallm/models/<name>/`.

```sh
# build (stub, skel, ARM) and stage the binaries; move them to the layout
source tools/htp/env.sh   # or your SDK environment
HEX_ARCH=v81 HEXKL_SDK_VER=6.4.0.2 ANDROID_SERIAL=<serial> \
  bash Applications/CausalLM/build_android.sh --htp --install
adb shell 'C=/data/local/tmp/nntrainer/causallm; mkdir -p $C/models;
  cp -a /data/local/tmp/Quick.AI/*.so /data/local/tmp/Quick.AI/nntr_causallm $C/'

# the pack: config.json with tie_word_embeddings false, nntr_config.json
# with lmhead_dtype QS4CX (plus the htp_chat preset keys)
adb push <pack dir> /data/local/tmp/nntrainer/causallm/models/<name>

adb shell
cd /data/local/tmp/nntrainer/causallm
export LD_LIBRARY_PATH=$PWD ADSP_LIBRARY_PATH=$PWD NNTR_NUM_THREADS=8
NNTR_GENERATED_IDS=/sdcard/ids.txt ./nntr_causallm models/<name>
```

Lines to read on stderr:

- `[HTP] lm_head on the NPU: 262144 x 2816 QS4CX as 64 WH slices of 4096
  ... placed in N ms` -- the head is bound to the DSP (absent: the head is
  Q4_0 or tied and runs as in section 1);
- `[HTP] lm_head WH cache <file>: written` / `read` / `stale, rebuilt`;
- `prefill: <tokens> tokens, <ms> ms, <TPS>` and `generation: ...`;
- `fc+dense_ffn+lm_head ms/token` in the per-token breakdown.

## 5. Open

- **Accuracy with the delivered weights.** The NPU head is the same kernel
  and numeric path as every FC in the model (u8 row quantization, int4
  weights, the dequant epilogue); what changes the logits is the head's
  own quantization (per-column int4 instead of Q4_0's per-32-block
  scales), which the weight producer decides. The gate is a token / PPL
  comparison of the delivered QS4CX head against a reference, on the real
  pack; the fixture above cannot give it.
- **Pre-baked affine WH head: implemented.** The native pack's
  `QS4CX_WH_AFFINE` head uses slice registration through `registerFcWh`,
  including offsets, as described in section 7. The QS4CX conversion/cache
  path remains available for its own layout.
- **VTCM size.** The 6 MiB slice bound assumes the 8 MiB VTCM of v79 /
  v81; a smaller VTCM wants the session's size in `bindLmHeadWh`.

## 6. Files

| file | change |
|---|---|
| `nntrainer/tensor/htp_backend/htp_graph_desc.h` | `FEED_WH` allowed on `LM_HEAD`; `HTP_GRAPH_MAX_PARTS` 64; `HTP_GRAPH_VERSION` 3 |
| `nntrainer/tensor/htp_backend/hmx/hexkl_graph.c` | `LM_HEAD` on WH handles: the init check and the op; the logits allocated whatever the feed |
| `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.h` | `HEXKL_FC_M1_MAX_PARTS` 64 |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` | `bindLmHeadWh` (slices, arena, cache), `invokeLmHeadWh`, `SharedLmHead::wh`, `set_lm_head_wh_cache` |
| `nntrainer/tensor/cpu_backend/compute_ops.h` | `set_lm_head_wh_cache` |
| `Applications/CausalLM/models/gemma4/gemma4_htp_moe_causallm.cpp` | a QS4CX head accepted for the `LM_HEAD` op; the cache named beside the model |
| `test/htp/nntr_hvx.idl`, `test/htp/nntr_hvx_mm_u8i4.c` | skel entry `lm_head_wh_f32` |
| `test/htp/host/graph_host_check.c` | the WH `LM_HEAD` case; wire v3 |
| `docs/gemma4-htp-user-guide.md`, `docs/gemma4-htp-code-guide.md` | the keys and the path |

## 7. Unified native TL2 and affine WH pack

The original-preserving ternary pack uses `TL2CX_WH` experts and
`QS4CX_WH_AFFINE` FC / untied head records in one model file. An affine
record stores WH int4 codes, channel scales, code sums and integer channel
offsets: `w = scale * (q + offset)`. BF16 rounding reconstructs the
original checkpoint values; no symmetric int4 re-quantization is needed.

The head bind gathers each vocabulary slice directly from the model's WH
tile rows with `registerFcWh`, including its offsets. It uses the same
64-part v3 graph ABI and the same `lm_head_wh_f32` entry described above.
Both prefill's last row and decode therefore use those registered slices.
This pack needs no `.lmhead_wh` conversion cache and no expert sidecar.
Prefill expands native TL2 expert panels to WH int4 in DSP scratch;
decode reads native TL2. The expert palette is {-1, 0, 1}, without a
7-fold scale adjustment. Rebuild both ARM client and DSP skel for the
extended affine weight registration and the new LM head RPC entry.
