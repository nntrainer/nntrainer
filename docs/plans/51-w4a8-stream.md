# 51 — w4a8 weight stream on HVX: 4-bit tiles with group-128 fp16 scales, ABI v5

Issue: dlwlzzero/nntrainer#51 (`prio:p1`; decision (a) of #49, 2026-09-18; supervisor re-base of
2026-09-18 on the #25 measurement). Contract: `docs/plans/0000-agent-system-and-env.md`.
Branch: `hvx/51-w4a8-stream` from `hvx_impl` (`2c880173`: #25 prefetch merged, #25 and #53 folded).
Line numbers below are `hvx_impl @ 2c880173`.

**Three PRs and one handoff.** PR 1 is the format (checkpoint producer, image, x86 reference, ABI
v5, validator) and produces the W4 PPL band with no kernel; PR 2 is the tiled W4A8 kernel
(q/k/v/o/gate/up, lm_head, EMBED); PR 3 is the W4A16 `down` kernel, after which the one device
handoff runs. Every PR leaves the shipping `tiled32` (W8) path bit-identical (the W4 path is a
header layout id plus a per-op flag, never a replacement), which is what lets PR 1 and PR 2 merge on
simulator gates alone.

## 1. Goal and gate

The user's goal policy (HEXAGON_BENCHMARK.md, 2026-09-18): HVX goals are read **relative to the CPU
baseline** (#60). This plan therefore gates on bytes, Mcyc and GB/s on a named unit, and the PPL band;
the GENIEX_LLAMACPP q4_0 cells (70.3 @512, 27.5 @4096) stay the row's *reference*, not a pass rule —
the supervisor's arithmetic shows 70.3 also needs #58 (ATTN ≥ 3× down) and the 4-bit stream alone
lands near 47–50 tok/s at 512 on `R3CY10WM83Y`'s clock.

Every device number is DSP Mcycles per decode step (`E2E decode … median_pcycles`), FARF per-kind
Mcyc from `tools/hexagon/summ_farf_prof.py`, pass 2 of a reversed double pass, one unit, one session
(HEXAGON.md §7 rule 9 (a)–(e)).

| # | Acceptance (issue + supervisor comment) | Pass means (measurable) |
|---|---|---|
| G0 | image bytes | `nntr_hexpack` prints `HEXPACK weights=307959808` for `weight_layout=w4g128` (all weight tensors 4-bit) and `350738432` for `w4g128_down8`; the W8 image is byte-identical to today (`qwen3_full.hexw` md5 `5abf61bef086368559423daa3bea9a99`), proof that PR 1 moved no W8 byte. The skel prints `stream bytes/step=307304448` (w4g128) / `350083072` (w4g128_down8) under `HTP_MM_STREAM_ONLY` — 0.516 / 0.587 of today's 595,984,384 |
| G1 | stream-phase Mcyc ≤ ≈ 0.55 × the W8 control | handoff §4 step 9, 512, pass 2, same unit and session: `mm8`+`mm16`+`lg` of **B on the w4g128 image ≤ 0.55 ×** the same sum of A (post-#25 `hvx_impl`, W8 image; 33.6 Mcyc in #25 B → B ≤ 18.5). 0.55–0.60 = partial (adopt, record which path missed from the per-path split); > 0.60 = fail. Expected from #25's per-path rates: `mm8` 181.7 MB at 37.0 GB/s ≈ 9.0 Mcyc, `lg` 80.2 MB at 32.5 ≈ 4.5, `mm16` 45.4 MB of nibble rows read directly ≈ 4.3–5.0 → 17.8–18.5 Mcyc = 0.53–0.55 |
| G1' | the W4 stream ceiling and whether compute surfaces | C (B + `HTP_MM_STREAM_ONLY`) on the w4g128 image, 512 pass 2: per-path GB/s = bytes ÷ (Mcyc ÷ `pcycles_per_us`); **`mm8`(B) − `mm8`(C) ≤ 1.0 Mcyc and `lg`(B) − `lg`(C) ≤ 0.5 Mcyc** = the nibble widening stays hidden behind the DMA at half the bytes; a larger gap is recorded as the compute share and becomes the first item of the follow-up (MM_TB for W4, §5) |
| G2 | W4 PPL band | (a) x86, PR 1: `hexagon_ref_run --eval` on `t512_23.i32` for the w4g128 image ≤ **1.10 ×** the W8 x86 figure of the same build (container clang 41.2365 / 161, workstation gcc 41.5466 / 164; §5.1 — quote the build); the number becomes the **W4 x86 reference** in HEXAGON.md §5.1. (b) device: B `--eval` PPL on `t512_23.i32` within **±2 %** of the W4 x86 reference and top-1 within ±6 (today's DSP-vs-x86 band is −1.73 … +0.21 %, §8.2); at 1024 / 4096 against the x86 W4 figures on `t1024.i32` / `t4096.i32` produced in PR 1 (same bounds). A > 1.10 x86 band is a stop: comment on the issue, `needs-user` (the loss the user accepts is not in the code) |
| G3 | the saving reaches the step | decode Mcyc/step(B, w4g128) ≤ Mcyc/step(A) − 0.8 × (stream(A) − stream(B)) at 512, pass 2 — the DMA saving may not be eaten by scale reads or the epilogue; `attn` and `rest` of B within ± 1 Mcyc of A (nothing moved into another kind) |
| G4 | prefill not sacrificed | prefill (m=128) DSP Mcyc sum of B ≤ 1.15 × A at 512 (the nibble unpack is amortised over `MM_TB` tokens per weight vector; if it fails, the recorded gap sizes the W4 `MM_TB` sweep in the follow-up) |
| G5 | simulator | rung 2/3 per §4: `profile acc` STAT **bit-identical** to the v79 record (`0.0659682/92.7791`, the W8 model), 13/13; the new W4 cases inside the bounds of §4 and, on v79, the single-group cases `0/0` (the integer path proof, §3) |
| G6 | goal rows (recorded, not gated) | B on w4g128 at 512 / 1024 / 4096: tok/s, Mcyc, PPL / top-1 → the `nntrainer w4a8` rows of HEXAGON_BENCHMARK.md; the 70.3 / 27.5 cells stay the GENIEX reference; the CPU-baseline multiple is filled when #60 lands |

Environment check inside the handoff: A (W8 image) at 512, pass 2, within ± 5 % of the unit's
latest §8.2 cell (`R3CY10WM83Y`: 54.6 Mcyc, #25 B; if #58 has merged, its post-#58 cell); outside that
band the session is void.

## 2. Where it lives

**ABI: `NNTR_HTP_ABI_VERSION` 4 → 5** (`nntrainer/tensor/hexagon/htp/nntr_htp_common.h:18-19`), once, in
PR 1. Consumers that move with it: `lower_qwen3` (flags, cursor), `pack_weights`, `nntr_hexpack`
(`--layout`), `.hexcfg` (`weight_layout` strings), `ref_ops.c` / `ref_graph_forward` (flag dispatch),
`sim_model` (layout switch), the sim `graph` / `profile` tests, `test_oplist_header.c`, HEXAGON.md
§1.4 / §2. Not moving: the op *sequence* (451 ops, same kinds, same indices), so `find_divergence.py`,
`summ_farf_prof.py`, `next_mm[]`, the FARF fields and `htp_op_table` are untouched; `MATMUL_LOGITS`
keeps `m = 1`.

| File:line | Today | Change (PR) |
|---|---|---|
| `nntr_htp_common.h:18-19, :21-33` | v4, `WEIGHT_LAYOUT_TILED32 1`, 4 KB tiles | v5; `NNTR_HTP_WEIGHT_LAYOUT_W4G128 2u` (every weight tensor 4-bit), `NNTR_HTP_WEIGHT_LAYOUT_W4G128_DOWN8 3u` (`down` stays int8 row-major); `NNTR_HTP_FLAG_W4 0x2u` (op flag on EMBED / MATMUL_W8A8 / MATMUL_LOGITS / MATMUL_W8A16: "weights are w4g128"); `NNTR_HTP_W4_GROUP 128u` (= `NNTR_HTP_TILE_K`, fixed), `NNTR_HTP_W4_TILE_BYTES 2048u` (PR 1) |
| `nntr_htp_common.h:103-122` (`nntr_htp_tile_off`, `nntr_htp_repack_tiled32`) | int8 index + repack | siblings: `nntr_htp_w4_strip_bytes(K)`, `nntr_htp_w4_tile_off(n, k, K)` (byte) + `nntr_htp_w4_nibble_shift(k)` (0 / 4), `nntr_htp_w4_scale_off(n, k, K)`, `nntr_htp_repack_w4(dst, codes, scales, N, K)` from the checkpoint's row-major nibbles; `nntr_htp_w4_row_bytes(K) = K/2`, `nntr_htp_w4_row_scale_bytes(K) = K/128*2` for the row kind (PR 1) |
| `nntr_htp_common.h:210-277` (`nntr_htp_op_extent`) | in1 = `n*k`, in2 = `n*4` | with `FLAG_W4`: EMBED / W8A8 / LOGITS in1 = `(n/32) * strip_bytes(k)` (EMBED: `vocab/32`), in2 **unused** (`used` drops bit 2: scales are inline in the strip); W8A16 in1 = `n * k/2`, in2 = `n * (k/128) * 2` fp16 (PR 1) |
| `nntr_htp_common.h:294-408` (`nntr_htp_oplist_validate`, `:309-312` header, `:333-359` per-kind) | layout must be 1 | layout ∈ {1, 2, 3}; `FLAG_W4` legal only on the four weight-reading kinds; **consistency rule** (rc 5): for each such op, `FLAG_W4` must be set iff `layout == 2 || (layout == 3 && kind != W8A16)`; the existing `k % 128`, `n % 32`, `vocab % 32`, `k ≤ 16384` rules apply unchanged (PR 1) |
| `nntrainer/tensor/hexagon/host/graph_lowering.h:72-76, :82-96, :102-110` | `HexModelConfig`, int8 + fp32-scale pointers, offsets | `HexModelConfig.weight_layout` (default TILED32; `kQwen3_0_6b` unchanged); `HexLayerWeights` / `HexModelWeights` gain the W4 views (`const uint8_t *wq4` codes `[N][K/2]`, `const uint16_t *wq4_s` fp16 `[N][K/128]`, … , `embed4` / `embed4_s`); `HexWeightOffsets` keeps one offset per tensor (`wq_s` etc. are 0 / unused for tiled W4 tensors, `down_s` holds the fp16 group scales for W4 `down`, `embed_scale` unused) (PR 1) |
| `graph_lowering.cpp:57-61, :65-106` (`write_tiled`, `pack_weights`) | tiled32 repack, `down` memcpy | `write_w4_tiled` (`nntr_htp_repack_w4`), `write_w4_rows` (codes memcpy + fp16 scales memcpy) chosen per `cfg.weight_layout` (PR 1) |
| `Applications/CausalLM/hexagon/qwen3_lowering.cpp:66-94` (cursor), `:121-131`, `:149-184`, `:231-242`, `:264-287`, `:301-311`, `:335-347`, `:350-363` (header) | int8 sizes, `weight_layout = TILED32` | cursor sizes from the layout (strip bytes for tiled W4 tensors, `K/2 + K/128*2` for W4 `down`, no `embed_scale` / `*_s` under W4); `op.flags |= FLAG_W4` per the consistency rule; `header.weight_layout = cfg.weight_layout` (PR 1) |
| `Applications/CausalLM/hexagon/hex_image.cpp:24-28, :60-69` | writes / requires `tiled32` | writes the layout string (`tiled32` / `w4g128` / `w4g128_down8`), reads any of the three into `cfg.weight_layout`, still rejects a missing key (PR 1) |
| `Applications/CausalLM/hexagon/hex_pack.cpp:44-58, :64-78` | `--layers --max-seq --max-chunk` | `--layout tiled32|w4g128|w4g128_down8` (default: the checkpoint's format, see next row); prints the layout with `HEXPACK weights=…` (PR 1) |
| `Applications/CausalLM/hexagon/qwen3_w8cx_bin.{h,cpp}` (`:53-62` `expected_size`, `:87-110` cursor) | one header-less W8_CX format, size-checked | the reader recognises three sizes (W8_CX 598,230,528; W4G128; W4G128_DOWN8 — the producer prints them) and exposes `layout()`; W4 tensors are read as codes `[N][K/2]` + fp16 scales `[N][K/128]`; `HexagonBackend::create` (`hexagon_backend.cpp:26-39`) sets `cfg.weight_layout = bin.layout()` before `lower_qwen3`, so the app runs either `.bin` with no new option (PR 1) |
| `tools/hexagon/make_w8cx_bin.py:73-81, :95-101, :125-132, :137-153` | `quant_w8cx` | `--bits 4 [--down-bits 8]`: `quant_w4g128(w)` per row per 128-group: `amax = max|w|`, `d = fp16(amax / 7)` (rounded to fp16 **first**, the stored scale is the dequant multiplier), `q = clip(lround_half_away(w / d), -7, 7)`, stored as unsigned nibble `q + 8` (1..15; 0 = −8 is never produced), byte `b` of row `n` = `k = 2b` low nibble, `k = 2b + 1` high; scales fp16 `[N][K/128]` after the codes; `--down-bits 8` writes `down` as today (int8 + fp32); the expected-size formula and the `ok:` line follow (PR 1) |
| `test/hexagon/sim/ref_ops.{c,h}` (`:52-63` tiled dot, `:65-100` matmuls, `:213-224` embed, `:253-307` dispatch) | int8 refs | `ref_dot_w4_tiled(w, n, xq, K, g)` (int32 per group: `Σ (u − 8) · x`), `ref_matmul_w4a8` / `ref_matmul_w4logits` (`f = 0; for g: p = (float)dot_g * s[n][g]; f = f + p;` then `y = (fp16)(f * sx)` — two statements, `-ffp-contract=off` on `ref_ops.c` in `build_host_x86.sh:41-46` and `build_sim_test.sh`), `ref_matmul_w4a16` (exact int64 per group, same fp32 chain), `ref_embed_w4`; `ref_graph_forward_upto` dispatches on `d->flags & FLAG_W4` (PR 1) |
| `test/hexagon/hexagon_ref_run.cpp:42-44, :146-156` | kind names, `--list-ops` | `--list-ops` prints `flags=` (PR 1); `--eval` unchanged |
| `test/hexagon/test_lowering.cpp:635-646, :743-745`, `test_oplist_header.c:78-107`, `test_w8cx_bin.cpp` | v4 asserts, tiled32 bijection | v5; W4 offset examples (`w4_tile_off(0,0,256)=0`, `(0,4,256)` = same byte shift 4, `(0,8,256)=128`, `(1,0,256)=4`, `(0,128,256)=2048`, `(32,0,256)=strip_bytes(256)=4096+128`), `repack_w4` bijection over `(n, k)` → `(byte, nibble)`, scale block position; lowering checks for both W4 layouts (sizes above, flags, no `embed_scale`); `test_w8cx_bin` on the W4 `.bin` (scales finite, codes in 1..15) (PR 1) |
| `test/hexagon/sim/sim_model.{h,c}:37-62, :114-145, :197-255` | one layout | `sim_model_cfg.weight_layout`; plan / fill / emit per layout (W4 fill: random nibble bytes, fp16 group scales in `[1.6e-3, 8e-3]` so `|w|` matches the int8 fill's magnitude); `QWEN3_2L` stays TILED32, `QWEN3_2L_W4` added (PR 1: plan + negatives; PR 2: forward) |
| `test/hexagon/sim/test_graph.c:120-128` | layout-0 negative | negatives: layout 2 with a flag-less W8A8 op, layout 1 with a flagged op, layout 3 with a flagged W8A16 op → rc 5 (PR 1); a `graph_w4_*` forward block (PR 2) |
| `nntrainer/tensor/hexagon/htp/htp_graph.c:116-123` (`k_max`, `stream_bytes`), `:124-127` (`xq`) | `stream_bytes += n*k` | `stream_bytes += e.in1` from `nntr_htp_op_extent` (right for both layouts, G0); new scratch `xq_gsum` int32 `[max_chunk][k_max/128]` (PR 2); **PR 1 / PR 2 only**: refuse a list with a `FLAG_W4` `MATMUL_W8A16` op (rc 5, "no W4A16 kernel yet") right after `nntr_htp_oplist_validate` — the shared validator accepts it so `test_lowering` and the x86 tools can already produce and evaluate the all-W4 image; PR 3 deletes the line |
| `htp_ops.h:41-71` (`htp_exec_ctx`) | `xq`, `xq_scale` | `int32_t *xq_gsum` (`[max_chunk][k_max/128]`: `8 × Σ_{k∈g} xq[t][k]`, the group correction of §3) (PR 2) |
| `hvx/hvx-quant.h:49-104` (`quant_row`, pack at `:94-101`) | per-token int8 / int16 | when `i8`, after packing each 128-byte group: `Q6_Vw_vrmpy_VbVb(hq_packed, splat 0x01010101)` → 32 lane sums → 5 `vror`+`vadd` → `xq_gsum[t][g] = 8 × lane0` (PR 2) |
| `ops/hvx-matmul.c:54-59` (`mm_job`), `:64-71`, `:82-108` (`mm_tile`), `:113-130` (`mm_tiles`), `:138-144`, `:152-165` (`mm_slab`), `:189-214` (`mm_pf_kick`), `:224-286` (`mm_worker_vtcm`), `:295-321` (`mm16_stream_only`), `:326-349` (`mm_worker`) | strip = `32*k` bytes, `rows_per_buf = (half/k) & ~31`, DMA rows of `k` bytes | a **geometry** `struct mm_geom { uint32_t strip; bool w4; }` from `(kind, flags, k)` — W8 `strip = 32*k`, W4 `strip = nntr_htp_w4_strip_bytes(k)`; every `n0*k` / `rows*k` / `(jn-n0)*k` becomes `(rows/32)*strip`; `mm_slab` computes `rows_per_buf = ((half / strip) * 32)`; `mm_dma_push` pushes `rows/32` DMA rows of `strip` bytes (one 2-D descriptor, `strip ≤ 50,688 < 65,535`); `mm_pf_kick` takes the *next* op's geometry; `mm_tile` gains the W4 arm of §3; `mm_worker` passes the flag; `mm16_stream_only` gets the row byte length (`k` or `k/2`) (PR 2) |
| `hvx-matmul.c:355-373` (`quant_job` / `quant_worker`) | | passes `xq_gsum` (PR 2) |
| `ops/hvx-embed.c:20-43` | `nntr_htp_tile_off` gather, `scale[row]` | W4 arm: byte `w4_tile_off`, nibble `w4_nibble_shift`, `u − 8`, scale from `w4_scale_off(row, k)` (fp16 → float) (PR 2) |
| `hvx-matmul.c:388-399` (`MM16_R` / `MM16_TB`), `:401-445` (`mm16_block`), `:452-482` (`mm16_worker`), `:484-492` | int8 rows, one `Ww_vmpyacc` chain per row, fold, `× swv` | W4A16 arm of §3: nibble rows of `k/2` bytes, per 128-B chunk two groups, per-group `Ww_vmpy` + `Ww_vmpyacc` into a fresh pair, `lo+hi` → `Vsf_equals_Vw` → `× splat(s[n][g])` → qf32 add into the row's fp32 lane accumulator; the fold tree unchanged; the per-channel `swv` multiply skipped for W4; `MM16_R == 4` pin kept (PR 3) |
| `nntrainer/tensor/hexagon/htp/executor.c:132-139` | stream-bytes print | unchanged (reads `g->stream_bytes`) |
| `test/hexagon/sim/test_matmul.c:21-116, :118-146` | W8A8 / W8A16 cases | `run_case(m, k, n, kind)` gains a W4 flag: repack the row-major `(codes, scales)` with `nntr_htp_repack_w4`; cases `w4a8` m ∈ {1, 7, 128} × k ∈ {128, 1024} × n ∈ {64, 256} plus `(1, 2048, 256)`; the k = 128 rows carry the single-group `0/0` expectation on v79; `ref_dot_w4_tiled` vs a row-major `Σ (u−8)·x` on rows {0, 1, 31, 32, n−1} (PR 2); `w4a16` m ∈ {1, 8}, k ∈ {3072}, n ∈ {256, 100}, and `(2, 256, 6)` (PR 3); the ctx allocates `xq_gsum` |
| `test/hexagon/sim/test_matmul_dma.c:34-80, :109-140` | W8 X → Y chain | a W4 X (k=1024, n=3072) → Y (k=2048, n=1024) chain at 4 MB / 256 KB / 64 KB: `memcmp` vs the DDR path, prefetch hits = workers, mismatch drain, and a **mixed** chain W8 X → W4 Y (the prefetch geometry of the *next* op) (PR 2) |
| `test/hexagon/sim/test_logits.c:20-93`, `test_embed.c:20-120` | W8 | a W4 block each (same shapes; `logits_w4` 5e-3 / 1e-2; `embed_w4` and `embed_w4_rowmajor` 1e-3 / 1e-4) (PR 2) |
| `test/hexagon/sim/test_profile.c:118-124`, `tools/hexagon/run_sim_test.sh` | `acc|prefill0|prefill512|decode512` | scenario `acc_w4` (`QWEN3_2L_W4`, the same 8-token check, 0.1 / 0.1; printed as `profile_prefill_acc_w4 STAT`); `acc` stays the W8 gate (PR 2) |
| `tools/hexagon/build_skel.sh`, `build_sim_test.sh` | | untouched except the `-ffp-contract=off` for `ref_ops.c` (sim build) |

## 3. Design

**Format (PR 1).** One 4-bit format, `w4g128`: symmetric, 15 levels (`q ∈ [−7, 7]`, stored `u = q + 8`),
one fp16 scale per (output row, 128-k group) — the group is exactly one tiled32 k-tile, so the
kernel's existing "one strip of vectors per k-tile" loop is the group loop. A tiled W4 tensor `[N][K]`
is `N/32` **strips** of `strip_bytes(K) = k_tiles × 2048 + align128(k_tiles × 64)`:

```
strip(nt) at nt * strip_bytes; inside: k_tiles W4 tiles of 2048 B, then the scale block
W4 tile(kt)  : 16 vectors v[i], i = 0..15 (128 B each)
v[i] byte [4r + j]: low nibble  = u[nt*32 + r][kt*128 + 8i + j]       (j = 0..3)
                    high nibble = u[nt*32 + r][kt*128 + 8i + 4 + j]
scale block  : fp16 s[kt][r] at k_tiles*2048 + kt*64 + 2r   (64 B per k-tile, padded to 128)
```

so `vand(v, 0x0F…)` is the tiled32 int8 vector of k-group `2i` and `vand(vlsr_uh(v, 4), 0x0F0F…)` is
k-group `2i + 1` — the W8 kernel's vectors, two per load. Inline scales travel through the DMA ring
with their tile (no DDR touch in the hot loop). The row kind (`down`, `MATMUL_W8A16`) keeps row-major
codes `[N][K/2]` with **byte `b` of a 128-B chunk `c` = `k = 256c + b` (low) and `k = 256c + 128 + b`
(high)**, so each unpacked byte vector is one whole 128-k group in the lane order `mm16_block` already
uses, and a separate fp16 scale tensor `[N][K/128]` (`in2`). qwen3-0.6b: strips 16,896 B (k=1024) and
33,792 B (k=2048), image **307,959,808 B** (`w4g128`) / **350,738,432 B** (`w4g128_down8`), stream
307,304,448 / 350,083,072 B per step. The checkpoint producer quantises from the bf16 source
(never from the int8 W8_CX bytes); the repack is a pure permutation shared by host, DSP references and
tests through `nntr_htp_common.h`, as `nntr_htp_tile_off` is today.

**Tiled kernel (PR 2).** Per k-tile `kt`, per token `u`: a fresh int32 accumulator
`acc[u] = Σ_i vrmpyacc_VwVubVb(acc[u], lo_i, splat(xw[u][32kt + 2i])) + vrmpyacc(…, hi_i, splat(xw[u][32kt + 2i + 1]))`
(16 loads, 16 `vand`, 16 `vlsr`+`vand`, 32 `vrmpyacc` for 2048 B — the W8 loop is 32 loads + 32
`vrmpyacc` for 4096 B). The nibbles are **unsigned** in the MAC (`Q6_Vw_vrmpyacc_VwVubVb`, ub × b) and
the offset is removed once per (token, group) with the exact identity
`Σ_k (u−8)·x = Σ_k u·x − 8·Σ_k x`: `acc[u] −= splat(xq_gsum[u][kt])` (one `vsub` instead of 32 sign
fix-ups; `xq_gsum` is produced by `quant_worker` from the packed group bytes). Then the group scale in
the §7 rule 5 order: `f = Vsf_equals_Vw(acc)`, `f = sf(qf32 vmpy(f, s_kt))`,
`facc[u] = sf(qf32 vadd(facc[u], f))` — one qf32 op per add, never chained (rule 1); after the last
k-tile `facc[u] = sf(qf32 vmpy(facc[u], splat sx[u]))` and the same fp16 / fp32 store as today.
`s_kt` (32 fp16 → 32 fp32) comes from the strip's scale block through `hvx_vec_f16_to_f32`, one load
per two k-tiles. Registers at `MM_TB = 4`: 4 `acc` + 4 `facc` + `v`/`lo`/`hi` + 2 scale vectors + masks.
EMBED keeps its scalar gather (once per forward) with the W4 index. `MATMUL_LOGITS` is the same kernel
with `y_is_f32`, on the tied W4 table. The DMA ring, chunking, cross-op prefetch and `HTP_MM_STREAM_ONLY`
are unchanged in structure — only the strip geometry is parameterised, so `mm_pf_kick` must use the
next op's geometry (the `test_matmul_dma` mixed chain is the proof).

*Bit-exactness rule for the 4-bit path (x86 ref vs sim).* The integer part — unpack, `vrmpy`, the
group correction — is exact by construction; the group sum is an fp32 chain that the W8 path did not
have. So: (i) every **single-group** case (`k = 128`) has the W8 epilogue's numerics and its v79 STAT
must be `0/0`, like `matmul_w8a8_*` today (rule 5: the v79 qf32 → sf conversion is RNE); (ii)
multi-group cases keep the unit-test bound 2e-3 / 1e-3 (`logits` 5e-3 / 1e-2) with the STAT recorded —
expected `0/0` on v79 as well, and any non-zero STAT must be identical between the DDR and the DMA path
(`memcmp`) and on both worker counts, otherwise it is an ordering bug, not rounding; (iii) the
reference fixes the order (`p = (float)dot_g * s_g; f = f + p;` in `g` order, `-ffp-contract=off`),
and the kernel reproduces it lane by lane; (iv) `graph_w4_*` and `profile_prefill_acc_w4` STATs are
recorded in §8.3 as the v79 W4 record and are bit-identity gates from then on; the W8 STATs never move.

**Row kernel (PR 3).** `mm16_block` per 128-B chunk: `lo = vand(v, 0x0F)`, `hi = vand(vlsr_uh(v,4),
0x0F0F)`, `−8` via `Q6_Vb_vsub_VbVb` (signed, [−8, 7]; the 2-op correction is not worth a per-lane
insert here), `Q6_Wh_vunpack_Vb` → int16 as today; per group a **fresh** pair `Q6_Ww_vmpy_VhVh` /
`Q6_Ww_vmpyacc_WwVhVh` (64 + 64 k), `lo + hi` (int32, exact: 2 products ≤ 8 × 32767 each),
`Vsf_equals_Vw`, `× splat(s[n][g])` (scalar fp16 → fp32 → splat, 4 rows per block share the token),
qf32 add into the row's fp32 lane vector; after the row the existing shuffle tree folds four rows,
skips the `swv` multiply for W4, applies `sx[t]` and narrows. 8 vector ops per (row, token, group)
against 2 today; at m=1 the kernel is latency-bound (8.59 Mcyc for 7.23 of DMA wait in #25), at m=128
it is ≈ 4 Mcyc per 128-token chunk, inside G4. Numerics: the reference is the exact int64 group dot
and the fp32 chain; the kernel's lane-wise fp32 partials and the fold differ in order as they do today
(rule 8), bound 2e-3 / 1e-3.

**Why `down` goes 4-bit at all.** `down` is 88.1 of 596 MB (14.8 %). With `down` int8 the stream is
0.587 of today's bytes and the 0.55 gate cannot be met on bytes alone even at equal GB/s (§1 arithmetic:
≈ 8.6 + 4.5 + 9.0 / 33.6 = 0.66); with W4 rows it is 0.516 and the estimate lands at 0.53–0.55. The
`w4g128_down8` layout exists so PR 2 is runnable and measurable on its own and so the handoff reads
`down`'s share on the device with one skel and two images, and it is the fallback if G2 (a) fails only
for W4 `down`.

**Rejected: per-channel fp32 scale + integer per-group sub-scale (Q4_K-style: `w = sw[n] · m[n][g] · q`, `m ≤ 255`).**
It keeps the whole dot in int32 (`Σ_g m_g · dot_g` fits for `k ≤ 3072`) and the W8 epilogue bit-exact,
but an 8-bit sub-scale is ±0.2 % on the loudest group and ±5 % on a group at 1/25 of the row's max, an
accuracy cost without a precedent PPL band at 0.6B; the row kernel could not use it (`w' ≤ 889`
overflows the `lo+hi` int32 lane at `k = 3072`); and the issue fixed fp16 group scales (9.3 MB) with
q4_0 as the precedent. It stays the fallback if the device shows the qf32 group chain misbehaving (the
handoff's three-way PPL column is what would show it).

**Rejected: new op kinds (`MATMUL_W4A8`, …) instead of a flag.** Same DSP code paths, but every kind
switch (`htp_op_table`, `kKindName`, the FARF `mm8`/`lg`/`mm16` fields, `next_mm[]`, `summ_*`,
`find_divergence.py`'s per-layer kind list, the sim `KIND_NAME`) would move; the flag keeps the
instrument that reads G1 unchanged and the op indices identical between the W8 and W4 images.

## 4. Steps

All commands through `tools/docker/run.sh` from the implementer's worktree, `HEX_ARCH=v79`. Simulator
budget (gates skill): the kind's test per step, `profile acc` (+ `acc_w4` from PR 2) and the 13 tests
**once** before each PR.

### PR 1 — `[hexagon] w4g128 weight format: producer, image, x86 reference, ABI v5` (host + validator)

1. **Branch.** `git checkout -b hvx/51-w4a8-stream hvx_impl`; record the W8 baselines from this tree
   before touching anything: `nntr_hexpack` → `qwen3_full.hexw` md5 `5abf61be…`, `hexagon_ref_run
   --eval` on `t512_23.i32` (container clang: expect 41.2365 / 161), `LOWER_TEST PASS`,
   `W8CX_BIN_TEST PASS`, oplist header `PASS`.
2. **Header (commit 1, `[htp]`).** `nntr_htp_common.h`: v5, layout ids, flag, W4 index / strip / scale
   helpers, `repack_w4`, extent arms, validator rules. `test_oplist_header.c`: v5 asserts, W4 offset
   examples, `repack_w4` bijection. Gate: `gcc -Wall -Werror … test_oplist_header.c -lm && /tmp/t` → PASS.
3. **Producer (commit 2, `[tools]`).** `make_w8cx_bin.py --bits 4 [--down-bits 8]`; run it twice on
   the HF snapshot (`/model`): print and record sizes + md5s of the two W4 `.bin` files (they join
   §2.4 / §5.1 and the wizard's md5 table of #32 as optional artifacts).
4. **Reader, lowering, packer, image (commit 3, `[hexagon]`).** `qwen3_w8cx_bin`, `graph_lowering`,
   `qwen3_lowering`, `hex_image`, `hex_pack --layout`, `hexagon_backend` layout pick-up,
   `test_lowering`, `test_w8cx_bin`. Gate (rung 1): `LOWER_TEST PASS` (both W4 layouts asserted:
   sizes 307,959,808 / 350,738,432, flags per rule, validator accepts), `W8CX_BIN_TEST PASS` on all
   three `.bin`s, `nntr_hexpack /model/<w8cx>.bin … ` → `.hexw` md5 **unchanged** (`5abf61be…`),
   `nntr_hexpack /model/<w4>.bin /work/…/qwen3_w4` → `HEXPACK weights=307959808`, and
   `--layout w4g128_down8` from the down-8 `.bin` → `350738432`.
5. **x86 reference (commit 4, `[test]`).** `ref_ops.{c,h}` W4 refs + flag dispatch,
   `-ffp-contract=off` on `ref_ops.c` in both build scripts, `hexagon_ref_run --list-ops flags=`.
   Gate (rung 1, the **G2 (a) band**): `hexagon_ref_run qwen3_w4 --tokens t512_23.i32 --eval` and
   the same for `qwen3_w4d8`; record PPL / top-1 for both next to the W8 figure of the same build;
   pass = w4g128 ≤ 1.10 × W8. Also run `t1024.i32` and `t4096.i32` (`--max-seq 4224` images; the
   x86 4096 eval is long — run it in the background, it is the 4096 reference cell of the handoff)
   and a `--steps 16` greedy generation on the local prompt (sanity: coherent ids). Paste all
   numbers into the PR body and HEXAGON.md §5.1. If w4g128 fails 1.10 but w4g128_down8 passes:
   comment on the issue (the gate re-bases to 0.62), set `needs-user`, continue with PR 2 on the
   down-8 layout. If both fail: stop, `needs-user`.
6. **Sim model + negatives (commit 5, `[test]`).** `sim_model` layout switch (plan, fill, emit),
   `QWEN3_2L_W4`, the three `test_graph.c` validator negatives, the `htp_graph_init_ex` "no W4A16
   kernel yet" refusal. Gate: `run_sim_test.sh graph` → PASS, `graph_prefill 0.0218946/6.88818`,
   `graph_decode 0.0197323/23.2374`, `graph_prefill_2workers` bit-identical (the W8 model is
   untouched; only the version constant changed).
7. **Pre-PR gates.** `clang-format-14` on changed `.c/.cpp/.h`; rung 2 `profile acc` → PASS, STAT
   bit-identical; rung 3 13/13 with every STAT equal to the #25 v79 record; rung 4 skel + harnesses
   build (md5s in the PR; the skel md5 changes: the validator moved). No handoff (nothing to
   measure). PR into `hvx_impl`, `state:review`; the PR body carries the W4 PPL table and the
   statement "W8 image byte-identical, W8 STATs bit-identical".

### PR 2 — `[htp] W4A8 tiled kernel: nibble strips for MATMUL_W8A8 / MATMUL_LOGITS / EMBED under FLAG_W4`

Branch continues (or a second branch `hvx/51-w4a8-kernel` off PR 1's head if PR 1 is still in
review). **Rebase point for #58**: if #58 has merged, rebase once — the overlap is `htp_ops.h`
(struct fields) and `htp_graph.c` (scratch allocation), two adjacent hunks.

8. **Tests first (commit 1, `[test]`).** `test_matmul` W4 cases (fail on the old kernel: expected),
   `test_logits` / `test_embed` W4 blocks, the `test_matmul_dma` W4 and mixed chains, `xq_gsum`
   allocation in the hand-built contexts. Gate: the files compile (`build_sim_test.sh`); the W8 cases
   still PASS.
9. **Quantiser group sums (commit 2, `[htp]`).** `xq_gsum` scratch (`htp_ops.h`, `htp_graph.c`),
   `quant_row` group reduction. Gate: `run_sim_test.sh quant` PASS, `quant_generic 0/65536` and
   `quant16_generic 184/25600` unchanged (the sums are a side output), plus a `quant_gsum` check
   against a scalar sum in `test_quant.c`.
10. **Geometry + W4 `mm_tile` (commit 3, `[htp]`).** `mm_geom`, the W4 arm, DMA push in strips,
    `mm_pf_kick` on the next op's geometry, `mm16_stream_only` row length, `stream_bytes` from the
    extent table, EMBED arm. Gate: `matmul` PASS (W8 STATs bit-identical; W4 k=128 cases `0/0`;
    others in bound, STATs recorded), `matmul_dma` PASS (`prefetch hits=6 workers=6` on the W4 chain
    and on the mixed chain, `memcmp` identical at 4 MB / 256 KB / 64 KB), `logits` PASS, `embed` PASS.
11. **Graph + profile (commit 4, `[test]`).** `graph_w4_prefill` / `graph_w4_decode` /
    `graph_w4_prefill_2workers` blocks (bounds 3e-2 / 5e-2 as the W8 blocks), `profile acc_w4`. Gate:
    `graph` PASS with the W8 STATs bit-identical and the W4 STATs recorded; `profile acc_w4` PASS,
    STAT recorded (the v79 W4 record for §8.3).
12. **Pre-PR gates.** clang-format; review against the §7 list (qf-format ops only — the new
    instructions on silicon are `Q6_Vw_vrmpyacc_VwVubVb`, `Q6_Vuh_vlsr_VuhR` and the per-k-tile qf32
    add: state them in the PR as rule 5 did for `Q6_Vsf_equals_Vw`; 128-B alignment of every strip
    and scale block; `wp_size` from the pool); rung 2 `profile acc` bit-identical **and**
    `profile acc_w4`; rung 3 13/13; rung 4 skel + harnesses, md5s. PR into `hvx_impl`,
    `state:review`. The PR states that the device verdict comes with PR 3's handoff and why merging is
    safe (W8 path bit-identical; W4 only through a layout id the shipping images do not carry).

### PR 3 — `[htp] W4A16 down_proj: nibble rows with per-group fp32 lane accumulation` + the handoff

13. **Kernel (commit 1, `[htp]`).** `mm16_block` W4 arm, `mm16_worker` row length, the `w4a16` sim
    cases, delete the `htp_graph_init_ex` refusal and its `test_graph` negative. Gate: `matmul` PASS
    (`w8a16` STATs bit-identical, `w4a16` in bound), `graph` PASS with `QWEN3_2L_W4` now on the
    all-W4 layout (STATs recorded).
14. **Pre-handoff gates.** clang-format; §7 review; rung 2 `profile acc` bit-identical + `acc_w4`;
    rung 3 13/13; rung 4: **A** from a clean checkout of the `hvx_impl` HEAD the branch is based on,
    **B** = this branch, **C** = `-DHTP_MM_STREAM_ONLY` — all three `-DHTP_PROF_FARF` (the FARF cost
    cancels); `build_host_test.sh`; images `qwen3_w4` (`--layout w4g128`), `qwen3_w4d8`, `qwen3_w4_4k`
    (`--max-seq 4224`); md5 of everything (`qwen3_full` / `qwen3_full4k` / the token files are already
    on the device, #25 table).
15. **Handoff — the device measurement is unavoidable (G1, G1', G2 (b), G3, G4, G6).**
    `docs/measurements/51-w4a8-stream.md` (`hexagon-handoff` template), one session ≈ 25 min, unit
    named, `summ_farf_prof.py` over every log; battery °C and charge state before the 4096 order (rule
    9(e)):

    | run | skel / image | reads | est. |
    |---|---|---|---|
    | 512 speed, pass 1 then pass 2 reversed | A/W8, B/W4, B/W4d8, C/W4 — then C/W4, B/W4d8, B/W4, A/W8 | env check (A), **G1** (B/W4 vs A), G1' (B vs C per path), G3, G4 (prefill Mcyc), `down`'s share (B/W4 vs B/W4d8: expect `mm16` −3 … −4 Mcyc) | 8 × ~0.5 min |
    | `--eval` 512 | B/W4, B/W4d8 (`t512_23.i32`) | **G2 (b)** against the x86 W4 figures of PR 1 | 2 × ~1 min |
    | 1024 | B/W4 speed + `--eval` | goal row | ~2 min |
    | 4096, one order after ≥ 10 min idle | A/W8 4k, B/W4 4k speed, then B/W4 4k `--eval` | goal row as an A/B inside one order; 4096 accuracy | ~2.5 + 2.5 + 6 min |

    Result table columns as #25 (variant, image, pass, skel md5 from the log, ctx, prefill tok/s,
    decode host ms, decode tok/s, **DSP Mcyc/step**, `pcycles_per_us`, FARF `mm8 / mm16 / lg / attn /
    rest`, log stamp) plus for C the per-path GB/s (`mm8` 181,665,792 B, `lg` 80,222,208 B, `mm16`
    45,416,448 B); the accuracy table with the x86 W4 reference cells and the ±2 % / ±6 bounds; the
    pass-rule table with G1 / G1' / G2 (b) / G3 / G4 written out. Set `state:needs-measurement`, stop.
16. **After `state:measured`.** Docs per §6; PR into `hvx_impl` (`state:review`). Verdict paths: G1
    pass → the w4g128 image is the shipping format for the w4a8 rows (the W8 rows stay the W8A8
    record); G1 partial → adopt, and the per-path split names the next issue (`mm16` above 5 Mcyc →
    #59 re-targeted to W4 rows; `lg`/`mm8` compute surfacing → a W4 `MM_TB` sweep in #57's session);
    G2 (b) fail → the ⑮ procedure (1-layer W4 image, `--dump-op` on the first W4 matmul, x86 W4 ref)
    before anything else, no merge of the format as shipping.

## 5. Risks

* **Bandwidth is invisible on the simulator** (gap rule 1): the 0.55 verdict, the per-path GB/s and
  whether the nibble compute surfaces are device-only; B vs C in one session is what separates
  "DMA-bound at half the bytes" from "compute now shows". The strip / chunk geometry is the same DMA
  descriptor shape as today (2-D, 16-bit fields, strips ≤ 50,688 B), so no new DMA behaviour is
  introduced; the `matmul_dma` W4 and mixed chains prove the prefetch handover geometry on the
  simulator.
* **qf32 numerics of the group chain** (gap rule 2): the W8 path never accumulated in fp32 across
  k-tiles; the W4 path adds `k/128` qf32 adds per output. Rule 1's 2026-08 failure was exactly
  chained `Vqf32_vadd_Vqf32Vqf32`; the plan uses `Vqf32_vadd_VsfVsf` + `Vsf_equals_Vqf32` per add and
  #35 showed IEEE = qf helpers digit for digit on this part with toolchain 19.0.04. The three-way
  column (x86 W4 ref / v79 sim W4 STATs / device `--eval`) is the record; a device deviation beyond
  ±2 % is a kernel bug and goes through ⑮ with the 1-layer W4 image, and the Q4_K-style integer
  sub-scale of §3 is the fallback format.
* **The PPL loss is a user judgement**: the plan sets 1.10 × W8 as the x86 stop line and reports the
  measured number in PR 1 before any kernel work; the user can lower it at PR-1 review. The reference
  figure depends on the `hexagon_ref_run` build (§5.1: clang vs gcc 0.75 % apart) — every W4 cell
  names its build.
* **Two images of two sizes, one skel**: the environment-gap rule is met by the `.hexcfg` layout string
  (a W8 skel that meets a W4 image fails init with rc 3 (version) — there is no way to run a v5 list
  on a v4 skel — and a W4 list on the PR-2 skel with a W4 `down` fails rc 5 by the coverage check);
  `push_if_changed` compares md5 (ledger ⑪); every result row repeats the md5 the harness printed and
  the `E2E init ok weights=` size (307,959,808 / 350,738,432 / 598,623,744 tells the images apart).
* **Prefill regression**: the unpack is amortised over `MM_TB` tokens per weight vector, but the
  per-k-tile epilogue (7 ops per token) and the row kernel's 8-op groups are new MAC-phase work at
  m = 128. G4 bounds it at +15 %; the fix if it fails is a W4-specific `MM_TB` (register budget
  allows 8 at 2 accumulators per token) measured in #57's prefill session, not a format change.
* **Register pressure / spills in `mm_tile`**: 8 accumulators + 3 unpack + 2 scale vectors at
  `MM_TB = 4`; if hexagon-clang spills, the `matmul` pcycles (relative signal) and a look at the
  `.s` show it before the device does. Keep `tb` a compile-time constant as today.
* **4096 cross-session offset (#53)**: the 4096 row is an A/B inside one order (rule 9(e)); no 4096
  cell is compared with another sitting.
* **#58 in flight**: two adjacent hunks (`htp_ops.h`, `htp_graph.c`); rebase once at the PR-2 branch
  point. If #58's device session has not happened when the PR-3 handoff is written, A is still a
  fair control (both A and B carry the same ATTN), and the goal rows are simply pre-#58.
* **Stale x86 artifacts**: `build_x86_hexagon/` is rebuilt at step 1 so the W8 baselines are
  `2c880173`'s, and the W8 `.hexw` md5 is checked after every PR-1 commit that touches the packer.

## 6. Overlap and ordering with #58, #59, #57, #60

* **#58 (decode attention, `state:in-progress` now).** Disjoint kernels (`hvx-attn.c` vs
  `hvx-matmul.c`); shared `htp_ops.h` / `htp_graph.c` hunks. PR 1 of this plan can run in parallel
  with #58 (host, validator, references); PR 2 branches or rebases after #58 merges. The tok/s goal
  cells need both; the handoff's goal rows are read whichever order the two land in, and the
  benchmark's w4a8 row names which ATTN it ran with.
* **#59 (`down` through the VTCM ring, p2).** Its object becomes the W4A16 row after PR 3 (rows of
  1,536 B + a 48-B scale row; the same `MM16_R`-multiple chunking, half the bytes). Order: PR 3 first,
  then #59 rebases onto the W4 row kernel and its gate re-bases from `mm16 ≤ 6.0` to the W4 rows'
  DMA wait read from C in this handoff. If G1 is partial because of `mm16`, #59 is the named next step.
* **#57 (`MM16_R` generalisation, p2).** Same function as PR 3 (`mm16_block`). Order: PR 3 first
  (p1); #57 then generalises the fold on the per-group fp32 lane accumulator, and its sweep session
  can carry a W4 `MM_TB` variant if G4 or G1' asked for one.
* **#60 (CPU baseline).** Independent; the w4a8 row's multiple is written once #60 fills.
* **#26 (prefill attention), ⑮ bisection**: untouched; the 1-layer W4 image is a new tool for ⑮ if
  G2 (b) fails.

## 7. Docs to update

* HEXAGON.md §1.4 (`:146-222`): "ABI v5" — layout ids 2 / 3, `FLAG_W4`, the consistency rule, extents
  of the W4 kinds, v5 rejects v4 lists; §2.1 (`:238-287`): the strip / W4 tile / scale-block layout,
  the row-kind nibble order, `nntr_htp_w4_*`, the two image sizes (the "still exactly 598,623,744"
  sentence gains the W4 rows); §2.2 (`:289-339`): chunking in strips, `xq_gsum`; §2.4 (`:367-389`):
  `make_w8cx_bin.py --bits 4 [--down-bits 8]`, the two `.bin` sizes / md5s, `nntr_hexpack --layout`,
  `.hexcfg` strings; §3 (`:391-453`): helper names; §5.1: the W4 x86 reference lines (both builds,
  512 / 1024 / 4096) and the G2 band; §5.2: the new cases, STATs and `profile acc_w4`; §7: rule 5
  addendum (the per-k-tile qf32 group add and the unsigned `vrmpy` + correction identity), a new rule
  only if the device deviates; §8.2: a "#51" block (A / B / B' / C table, per-path W4 GB/s, the
  `down` share, verdicts, unit, md5s); §8.3: the v79 W4 STATs and the unchanged W8 ones; §8.1: the W4
  row of the accuracy table; §9: rewrite the "after #58" sentence with what #51 measured and what is
  left (#59 on W4 rows, W4 `MM_TB`).
* HEXAGON_BENCHMARK.md: the `nntrainer | w4a8` row split into 512 / 1024 / 4096 rows (unit, handoff,
  Mcyc, PPL, which ATTN), the stage-1 w4a8 goal cell (bytes 0.516, measured stream Mcyc and GB/s, the
  CPU-baseline multiple pending #60, 70.3 / 27.5 as the GENIEX reference), the W8A8 rows unchanged,
  a log line per PR and one for the measurement.
* `docs/superpowers/specs/hexagon-hvx-optimization/07-follow-ups.md`: new entry ⑯ (w4a8 stream: format,
  numbers, what is left); ⑤ note (the row-major `down` now has a W4 form; tiling `down` still open);
  ⑬ note (#59 re-targeted to W4 rows).
* `docs/backend_guide/hexagon-guide/02-architecture.html` (guide writer): the W4 strip picture next to
  the tiled32 one.
* `tools/docker/setup_wizard.sh` (#32's md5 line): the W4 `.bin` md5s as optional lines — only if the
  wizard is asked to produce W4 checkpoints; otherwise a sentence in §2.4 that the W4 `.bin` is built
  on demand.

## 8. Out of scope

HMX (u8i4 is the HMX MAC format; stage 2). Activation quantisation changes. Tiling `down` (ledger ⑤).
Streaming `down` rows through the ring (#59). Asymmetric / min-max 4-bit (q4_1) and super-block
sub-scales (Q4_K): the fallback of §3 only if the fp16 group chain fails on silicon. Group sizes other
than 128 (the group = k-tile equality is the design).
