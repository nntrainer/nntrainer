# Hexagon cDSP backend — observation log

Chronological record of bring-up on **R3CX9078DNH** (Galaxy S25 / SM-S936U,
Snapdragon 8 Elite / SM8750, HTP **v79**), Qwen3-0.6B.

Background concepts are in
[`HEXAGON_NPU_PRIMER.md`](HEXAGON_NPU_PRIMER.md). This file is the "what
happened, what it did to the numbers, and why" record.

Two repos are involved:
- **nntrainer**, branch `hexagon-cdsp-bridge`
- **ggml-hexagon** (llama.cpp fork), branch `self-build` — contains
  `ggml/src/ggml-hexagon/nntr-htp-bridge.cpp`, the C-ABI bridge

> Supersedes `docs/nntrainer-htp-bridge-status.md` in the ggml-hexagon repo,
> which predates any hardware run and contains conclusions since disproven
> (noted in §2).

---

## Headline result

| config | prefill | decode | total (469-tok prompt) |
|---|---|---|---|
| CPU only, 4 threads | 512.0 | 90.3 | 2765 ms |
| NPU everything (first working version) | 137.4 | **1.37** | 93,790 ms |
| NPU everything (after pooling rpcmem) | 428.6 | 25.2 | 5,294 ms |
| **hybrid: prefill NPU, decode CPU** | **611.5** | **68–93** | **2644 ms** |

Final state: enabling `NNTR_USE_HEXAGON_CDSP=1` is never slower than CPU, and
gives **1.19× prefill / ~4% end-to-end** on long prompts.

**Decode was never made competitive on the NPU and cannot be** — see §7. The
hybrid's decode number is the CPU path; the DSP is idle during generation.

---

## 1. Starting state

Committed before this effort, none of it ever built for Android or run on
hardware:

- `HexagonContext` + `HexagonComputeOps` (572 lines, ~500 of them hand-forwarding
  unaccelerated ops to `get_cpu_ops()`)
- `hexagon_repack.cpp` — the q4x4x2 repack
- `nntr-htp-bridge.cpp` — stateless, allocated all rpcmem per call
- CausalLM Qwen3 FC layers tagged `engine=cdsp` behind `NNTR_USE_HEXAGON_CDSP`

The status doc reported "all 3 cases failed, something about Q4_0 unrecognized"
and hypothesised the DSP was returning `HTP_STATUS_NO_SUPPORT`.

---

## 2. Static analysis findings (before touching any code)

| # | finding | consequence |
|---|---|---|
| 1 | `ISA::HEXAGON` unreachable — `quantize.cpp:113` `isa_str_map` had only DEFAULT/X86/ARM | the q4x4x2 repack was **dead code**; the device held ARM `q4_0x4` weights while the bridge assumed q4x4x2 |
| 2 | `neuralnet.cpp:890` tagged HEXAGON as `nntr_q4_0_isa = "arm"` via `default:` | a q4x4x2 file would claim to be ARM; also nothing ever *reads* that key |
| 3 | `float_tensor.cpp:1000` gated accel on hardcoded `M > 1` | decode never reached the NPU |
| 4 | `matmul-ops.c:4762` — HMX needs `M >= 32` (`m_hmx = M & ~31`) | below 32 rows the DSP only runs HVX; NPU can't win there |
| 5 | `layer_node.cpp:952-968` stamps a layer's `ct_data` onto its **inputs**, which are the previous layer's outputs | a cdsp FC hijacks its CPU neighbour's tensors — the real cause of the earlier `sgemv_fp32` "not implemented" crash, which the 500 forwarding lines were treating as a symptom |
| 6 | bridge allocated/freed 3 rpcmem buffers **per call** + full weight memcpy | ~300 MB/token of memcpy, plus mapping churn |
| 7 | `buf->buft = nullptr` | `add_op` → `dump_op_exec` → `ggml_backend_buft_name` → `GGML_ASSERT(buft)`. **Aborts under `GGML_HEXAGON_VERBOSE=1`.** The prior audit explicitly cleared this, having checked only `add_tensor`, not the dump path inside `add_op` |
| 8 | weight buffer tagged `USAGE_COMPUTE` | DSP `hex_l2flush`es it per op; >128 KB escalates to **full D-cache flush-invalidate**, 196×/token |
| 9 | `HTP_OP_MAX_BUFS` / `HTP_MAX_MMAPS` both 16 | one rpcmem buffer per weight can't scale past 16 of 196 weights — must pool |
| 10 | DSP's `src1_spad` activation cache is **per batch** (`matmul-ops.c:4687`) | one-op-per-flush means the shared QKV activation is re-quantized 3× |

Also corrected an earlier mistaken claim that nntrainer's CPU Q4_0 GEMM was
single-threaded: `ggml_interface_omp.cpp` fans out over rows via
`tm.getComputeThreadCount()`, including GEMV. The NPU is competing with 4–8 ARM
cores, not one.

**Superseded status-doc conclusions:** the `NO_SUPPORT` hypothesis was wrong;
`buft = nullptr` does crash; and the claim that a prior "Phase 0" run had proven
the DSP path (so no skel rebuild was needed) was unsupported — the device had no
`libggml-htp-v79.so` or `libggml-hexagon.so` at all.

---

## 3. Fixes 1–2: the bridge, before any run

Both one-liners in `nntr-htp-bridge.cpp`, both from findings 7 and 8.

- `buf->buft = nullptr` → `&sess->repack_buffer_type` (weight) /
  `&sess->buffer_type` (staging). **Unblocks `GGML_HEXAGON_VERBOSE=1`**, which is
  the only way to read the DSP's response status. Diagnosis was blocked on this.
- weight buffer `USAGE_COMPUTE` → **`USAGE_WEIGHTS`**. Removes a full DSP D-cache
  flush per GEMM.

---

## 4. Build bring-up

Toolchains found on the server: NDK r26d (`/storage_data/snap/anup/`), Hexagon
SDK **6.4.0.2** with Tools 19.0.04 (`/storage_data/raunak/`).

| step | outcome |
|---|---|
| `libggml-hexagon.so` + `libggml-htp-v79.so` via preset `arm64-android-snapdragon-release` | worked first try |
| nntrainer Android with `-Denable-hexagon-cdsp=true` | **FAILED** |

The failure: `hexagon_repack.cpp:54` uses `std::to_string` but only included
`<stdexcept>`. glibc pulls `<string>` in transitively; **the NDK's libc++ does
not.** So this had never once compiled for the target — which is precisely why
the option had only ever been "verified by code inspection." Adding
`#include <string>` produced the **first successful cdsp-enabled Android build.**

Also needed: `git submodule update --init subprojects/iniparser` (all submodules
were unfetched).

---

## 5. First hardware runs — `nntr_htp_bridge_check`

| run | result |
|---|---|
| 1 | `dlopen(libggml-hexagon.so)` failed — `libggml-base.so` not pushed |
| 2 | **DSP session came up.** `HTP0 new session ... libggml-htp-v79.so`. But all 3 cases FAIL with `cpu=1713599`, `mean_abs_err=nan` |
| 3 | **ALL CASES PASSED** |

**Run 2 was the turning point.** The session log proved unsigned-PD, skel load,
`enqueue_op`/`flush`, and the DSP round trip all worked — the single largest
unknown, resolved. And reading the numbers the right way round settled the rest:
activations and weights are `U(-1,1)` with K=512, so a correct dot product is
O(√512) ≈ ±20. `hexagon = -16.16` was plausible; `cpu = 1713599` was not.

**The CPU reference was the broken side.** The tool repacked weights to q4x4x2 for
the Hexagon path but handed the CPU reference raw `block_q4_0`;
`__ggml_q4_0_4x8_q8_0_GEMM` expects 4-row-interleaved `q4_0x4`, so it read four
rows' nibbles out of one row's storage. No error, just 1.7e6 and NaNs, which then
swamped the comparison and blamed the DSP.

**This was the entire "all 3 cases failed / Q4_0 unrecognized" mystery.** A broken
test, not a DSP or layout problem. Fix: give the CPU path its own
`repack_q4_0(ISA::DEFAULT)`.

Passing values, unchanged through every later refactor (a useful regression
signal):

```
M=64  N=8    K=512   max_abs_err 0.114649  mean 0.027970
M=64  N=1024 K=1024  max_abs_err 0.187000  mean 0.031859
M=32  N=1024 K=1024  max_abs_err 0.187000  mean 0.031982
```

Sub-1% relative, consistent with the DSP quantizing activations to `q8x4x2` where
the CPU uses `q8_0`. Coverage is better than it looks: case 1 fails HMX's
`ne01 % 32` gate and exercises HVX; cases 2–3 take the HMX path.

A separate run with `GGML_HEXAGON_VERBOSE=1` confirmed the `buft` fix — it no
longer aborts.

---

## 6. First end-to-end CausalLM run

Prerequisites that had to be built or fixed:

- `ISA::HEXAGON` made reachable — `quantize.cpp` `isa_str_map`, the missing
  `fallback.cpp` case, and the `neuralnet.cpp` metadata case (findings 1–2)
- `gemm_q4_0_accel_min_rows()` added, Hexagon → **1**, so decode reached the NPU.
  Necessary at this stage: with a q4x4x2 `.bin` there is *no* valid CPU fallback,
  so `M > 1` would have emitted garbage for every token.
  Deliberately **not** just deleting the `M > 1` test — `ClComputeOps` also
  returns `true` from `supports_gemm_q4_0_accel_fp32()`, so a blanket removal
  would have silently rerouted OpenCL decode too.
- `libtokenizers_android_c.a` — required rustup (installed user-local,
  Rust 1.97.1) + `aarch64-linux-android` target. A colleague's prebuilt aarch64
  archive was tried first and rejected: it lacked five symbols CausalLM calls
  (`tokenizers_encode_batch`, `tokenizers_free_encode_results`,
  `tokenizers_get_vocab_size`, `tokenizers_id_to_token`, `tokenizers_token_to_id`)
- quantized **on device** (`nntr_quantize` ships in the Android build):
  `--fc_dtype Q4_0 --embd_dtype Q6_K --isa HEXAGON` → 2273 MB → **358 MB**

Two unrelated upstream bugs hit along the way:
- `nntr_quantize` writes its invocation-relative path into `tokenizer_file`
  instead of the bare filename
- `nntrainer_causallm` doubles the model path unless given an absolute one

**Result — coherent text generated end to end:**

```
prefill: 18 tokens, 932 ms, 19.31 TPS
generation: 128 tokens, 93114 ms, 1.37 TPS      ← 727 ms/token
total: 94051 ms          peak 651 MB
```

CPU baseline on the same model (ARM-quantized): prefill 236.8, decode **92.9**,
total 1456 ms. So: correct, and ~65× slower than CPU.

**The diagnostic that mattered:** prefill took 932 ms for 18 tokens but **655 ms
for 90 tokens** — more arithmetic, less wall time. 196 GEMMs × fixed overhead,
independent of M. 727 ms/token ÷ 196 = **3.7 ms per GEMM** against 0.055 ms on
CPU, while the actual DSP math for a 1024×1024 Q4_0 GEMV is microseconds. So
~100% of it was host-side bookkeeping.

---

## 7. Upper bound — what is even achievable

Before optimising further, measured llama.cpp's **own** Hexagon backend with
`llama-bench` on the same device and an official `Qwen3-0.6B-Q4_0.gguf`. This is
the mature reference: full graph scheduler, op batching, weights resident in
rpcmem from load.

| test | CPU | NPU (HTP0, ngl 99) | NPU vs CPU |
|---|---|---|---|
| prefill pp90 | 721.8 | 1014.6 | **1.41× faster** |
| prefill pp512 | 571.1 | **2045.9** | **3.58× faster** |
| decode tg128 | **158.9** | 34.6 | **4.6× slower** |

**This reframed the whole effort.**

- **Decode cannot be won.** 4.6× slower in the reference implementation. Decode
  is GEMV, i.e. bandwidth-bound; the DSP has no bandwidth advantage, adds a
  FastRPC round trip, and can't engage HMX below M=32. Not an implementation gap.
- **Prefill is where the NPU belongs**, and its advantage *grows* with prompt
  length — CPU prefill falls (721.8 → 571.1, attention is O(n²)) while NPU
  prefill rises (1014.6 → 2045.9, HMX scales with M).

Incidental: llama.cpp's **CPU** kernels also beat nntrainer's — 158.9 vs 92.9
decode, 721.8 vs 573.2 prefill. A separate, larger opportunity.

---

## 8. Pooling rpcmem — the 18× fix

Everything the bridge was doing per call was hoisted out of the loop. Not clever,
just correctly scoped.

| | before | after |
|---|---|---|
| rpcmem alloc + map | 3× **per GEMM** | 3 total **per run** |
| DSP-side `HAP_mmap` | 3× per GEMM (fresh fd ⇒ `reuse_buf` never hit) | once per region |
| weight memcpy | every GEMM (~300 MB/token) | once per weight |
| `ggml_init`/`ggml_free` | every GEMM | never (tensors hand-built) |

Design: **chunked pinned weight arenas** (128 MiB × ≤12), bump-allocated,
hash-keyed on the caller's weight pointer; plus **one non-pinned growable staging
region** for activation + output. Chunked because of finding 9 (16-mapping cap);
pooling also suits the wire format, where `htp_tensor.data` is an offset from a
buffer base.

### Two bugs this introduced, both instructive

**(a) Pinned buffers must never be freed.** Staging initially grew by
destroy-and-recreate *while pinned*. `shared_buffer::unmap()` skips
`htp_iface_munmap` for pinned buffers, so the DSP kept its `fd → VA` entry; the
kernel reused the fd; the DSP resolved a stale translation. Symptom:
`buffer mapping failed ... error 0x0000001a`, then a segfault. Fix: arenas pinned
(never released), staging **not** pinned.

**(b) Never own rpcmem in a static destructor.** Once the state owned regions, its
destructor ran from `__cxa_finalize` at `exit()` and called `unmap()` through
function pointers `dlsym`'d out of `libcdsprpc.so` — **already unloaded by then**.
Dangling jump, SIGSEGV. Worse, crashing inside `__cxa_finalize` pre-empted stdio
flush, so a run's entire buffered output vanished and it merely *looked* like the
test printed nothing. Fix: deliberately leaked singleton (the session was already
never deleted).

Both were caught by `nntr_htp_bridge_check`, not by the model — the cheap test
paid for itself.

**Result:**

| | prefill(18) | prefill(90) | decode | total(90) |
|---|---|---|---|---|
| before | 19.3 | 137.4 | 1.37 | 93,790 ms |
| after | **89.1** | **428.6** | **25.2** | **5,294 ms** |

Decode **18.4×**, prefill 3.1–4.6×, end-to-end 17.7×. Per-GEMM cost 3.3 ms →
0.20 ms. Output identical token-for-token and check-tool error values unchanged,
so pure overhead removal.

Still 3.5× slower than CPU on decode — consistent with §7 saying that's a losing
fight.

---

## 9. The hybrid — prefill NPU, decode CPU

Acting on §7. Two coupled changes:

1. `min_rows` 1 → **32** (the DSP's own HMX gate)
2. The `q4_0x4 → plain q4_0 → q4x4x2` conversion moved **into
   `HexagonComputeOps`**, done once per weight at first use

Why (2) is required by (1): if decode runs on the CPU, the CPU kernels need
ARM-layout weights present. So the `.bin` stays ARM, and the q4x4x2 copy exists
only in the bridge's rpcmem arena — derived at load. Both layouts resident, and
the arena memory was already being paid for.

This also **removed the need for `--isa HEXAGON` models entirely** and satisfied
the original "use the weights bin file as it is" goal.

Bridge ABI split to support it: `nntr_htp_bridge_upload_weight_q4x4x2(key, bytes,
N, K)` + `nntr_htp_bridge_gemm_q4_0(key, act, out, M, N, K)`. Decoupling `key`
from the byte pointer matters — the q4x4x2 bytes come from a small reused scratch
whose address is meaningless, while the cacheable identity is the long-lived ARM
weight pointer. Folding them together would have forced a permanent q4x4x2 copy
per weight (~358 MB) purely to have a stable key.

**Result (90-token prompt):** decode **88.8** (was 25.2 — i.e. back to the CPU's
88.2), prefill 416.7. But total 1661 ms vs CPU's 1610 ms — *slightly worse*,
because at 90 rows NPU prefill still loses.

---

## 10. Setting the threshold from measurement

32 is where the *hardware* switches kernels. It is not where *we* start winning,
because prefill is still dispatch-bound (one flush per GEMM ⇒ wall time nearly
independent of M).

Swept prefill throughput, NPU/CPU ratio:

| tokens | 79 | 157 | 196 | 235 | 274 | 313 | 391 |
|---|---|---|---|---|---|---|---|
| ratio | 0.64 | 0.86 | 0.93 | **1.14** | 1.06 | **1.19** | **1.19** |

Crossover ≈ **215 rows**. So `min_rows = 32` was making short prompts up to
**1.6× slower than plain CPU** — a real regression, introduced by trusting the
hardware constant instead of measuring.

Set **256** (safely past crossover given noise) + `NNTR_HEXAGON_MIN_ROWS` env
override, clamped ≥32 so it can't be used to push decode onto the DSP by
accident.

**Final verification:**

| | prefill CPU | prefill cdsp | decode CPU | decode cdsp | total CPU | total cdsp |
|---|---|---|---|---|---|---|
| 22 tok | 301.4 | 293.3 | 90.3 | 93.4 | 1492 ms | **1448 ms** |
| 469 tok | 512.0 | **611.5** | 69.4 | 68.3 | 2765 ms | **2644 ms** |

Short prompts stay on CPU and are indistinguishable from it; long prompts get
1.19× prefill and ~4% end-to-end. **Enabling cdsp is now never a loss.**

---

## 11. Commits

**nntrainer** (`hexagon-cdsp-bridge`):

| commit | what |
|---|---|
| `0a371ce6` | `<string>` — the Android build break |
| `a9ea61af` | make `ISA::HEXAGON` reachable (quantize / fallback / metadata) |
| `105965a2` | backend-declared `gemm_q4_0_accel_min_rows()` |
| `75b877e6` | bridge-check CPU reference layout fix |
| `75d29975` | hybrid split + in-process layout conversion |
| `3a93f796` | threshold from measurement (256) + env override |

**ggml-hexagon** (`self-build`):

| commit | what |
|---|---|
| `72af5fb2a` | pooled rpcmem, null `buft`, weight `usage` |
| `bc7f55295` | split weight upload from the GEMM |

---

## 12. What's left

Prefill sits at **~31% of the reference** (611 vs 2046 t/s at ~500 tokens). All
the remaining headroom is there, in order of value:

1. **Op batching** via `gemm_q4_0_batch_fp32` (`float_tensor.cpp:771`). Prefill
   wall time being nearly flat in M is the signature of 196 blocking round trips.
   Collapsing Q/K/V and gate/up gives ~112 dispatches *and* lets the DSP's
   per-batch `src1_spad` cache skip re-quantizing the shared activation 3× per
   group. Also what would let `min_rows` come back down toward 32.
2. **Eliminate the activation copy** — ~1.9 MB per prefill GEMM now. Route
   nntrainer's `MemoryPool` through an rpcmem `MemAllocator`
   (`network_graph.h:105` `setComputeBackend` already exists for this; QNN uses
   it at `neuralnet.cpp:252`). Note `reuse_inference_tensor_pool_` must also be
   set for cdsp or pool addresses churn every token.
3. **Not worth doing:** NPU decode. See §7.

Known outstanding issues not yet addressed:

- `ct_data` aliasing (finding 5) — still papered over by ~500 lines of
  forwarding in `HexagonComputeOps`. Real fix is at `layer_node.cpp:952`.
- `CpuComputeOps` is `.cpp`-private (`cpu_ops_table.cpp:24`), forcing every
  backend to hand-forward ~80 methods. Hoisting it to a header would delete those
  500 lines *and* fix `ClComputeOps`, which overrides only 12 and would throw on
  any real `engine=gpu` layer today.
- `nntr_q4_0_isa` metadata is still write-only; nothing validates weight layout
  at load, so a layout mismatch remains silent.
