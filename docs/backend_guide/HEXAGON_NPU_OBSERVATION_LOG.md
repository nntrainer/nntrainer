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

## Current configuration

Per instruction, **everything including decode runs on the DSP**
(`gemm_q4_0_accel_min_rows() == 1`). This is a coverage choice, not the
throughput-optimal setting. The threshold is a runtime knob
(`NNTR_HEXAGON_MIN_ROWS`) and needs no requantization to change — see §9, §10,
§13.

## Decode progression (Qwen3-0.6B, 4 threads, all matmuls on DSP)

| state | decode t/s | what changed |
|---|---|---|
| first working version | **1.37** | per-call rpcmem alloc/map/free + weight memcpy |
| pooled rpcmem | 25.2 | §8 — hoist all of that out of the loop |
| `OPPOLL` default | **30.3** | §14 — busy-poll instead of sleeping per op |
| CPU reference | 90.5 | for comparison |

## Prefill

| config | prefill t/s (469 tok) |
|---|---|
| CPU, 4 threads | 512.0 |
| DSP | **611.5** (1.19×) |

**Decode on the DSP is ~3× slower than CPU and that is not fully closable** —
see §7 and §12. Prefill on the DSP wins past ~215 tokens.

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
  `&sess->buffer_type` (staging). Intended to unblock `GGML_HEXAGON_VERBOSE=1`,
  the only way to read the DSP's response status.
  **Caveat discovered much later (§14): `GGML_HEXAGON_VERBOSE` was itself dead on
  the bridge path at this point**, because the bridge never calls
  `ggml_hexagon_init()`. So this fix was correct and necessary, but the claim
  made at the time that a verbose run "confirmed" it was unfounded — the dump
  path had never actually been exercised.
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

(Later commits are listed in the sections that describe them — §13 for the
`min_rows` change, §14 for the env-var resolver and the OPPOLL default.)

---

## 12. Framework comparison — why our ratios trail ggml-hexagon

Measured with `llama-bench` (ggml-hexagon) and CausalLM (nntrainer) on the same
device, same model family, 4 threads, at matching M. ggml-hexagon runs
`-ngl 99`, i.e. the **whole graph** on the DSP.

| M (prompt tok) | ggml CPU | ggml NPU | ggml ratio | nntr CPU | nntr NPU | nntr ratio |
|---|---|---|---|---|---|---|
| 22 | 732.9 | 219.3 | 0.30 | 301.4 | 93.2 | 0.31 |
| 79 | 678.7 | 864.1 | 1.27 | 544.8 | 349.6 | 0.64 |
| 157 | 700.0 | 1419.6 | 2.03 | 646.1 | 554.8 | 0.86 |
| 196 | — | — | — | 662.2 | 612.5 | 0.93 |
| 235 | 664.2 | 1590.5 | 2.39 | 605.7 | 691.2 | 1.14 |
| 274 | — | — | — | 614.4 | 652.4 | 1.06 |
| 313 | 644.0 | 1593.7 | 2.47 | 592.8 | **706.5** | 1.19 |
| 391 | 554.0 | 1754.8 | 3.17 | 526.2 | 628.6 | 1.19 |
| 469 | 535.5 | 1872.9 | **3.50** | 512.0 | 611.5 | 1.19 |
| decode (128) | 156.2 | 34.6 | **0.22** | ~90.5 | ~25.4 | **0.28** |

Two premise corrections worth recording:

- **Our decode *ratio* is better, not worse** (0.28 vs 0.22). It is our
  *absolute* decode that is lower (25.4 vs 34.6). Partly because our CPU
  baseline is weaker, which flatters the ratio.
- **ggml also loses at small M** — 0.30 at M=22, essentially identical to our
  0.31. Their crossover is between 22 and 79 tokens; ours is ~215.

### The curve shape is the real finding

```
ggml NPU:  219 -> 864 -> 1420 -> 1590 -> 1594 -> 1755 -> 1873   keeps climbing
ours NPU:   93 -> 350 ->  555 ->  691 ->  707 ->  629 ->  612   peaks ~313, DECLINES
```

Dispatch overhead amortises *better* at larger M, so it cannot explain a curve
that turns down. Something grows superlinearly with M — and that is **attention,
which is on our CPU and is O(n^2)**. As the prompt grows it takes over total
prefill time no matter how fast the matmuls get. ggml offloads attention too
(`htp/flash-attn-ops.c`, `hmx-flash-attn-ops.c`), so their curve keeps scaling.

### Reasons, ranked

1. **Scope of offload.** `-ngl 99` moves attention, softmax, RoPE, RMSNorm and
   elementwise to the DSP; we move only the Q4_0 FC matmuls. So per layer we
   ping-pong DSP -> CPU -> DSP, their intermediates never leave the DSP, and the
   part of prefill the CPU is *worst* at is never offloaded. Produces the
   declining curve.
2. **Dispatch granularity.** 196 blocking round trips per forward pass vs a
   handful for a whole graph. They also keep up to 16 batches in flight
   (`opt_opqueue`); we block on every op.
3. **Data residency + activation requantization.** Per GEMM we memcpy the
   activation in and the result out. And the DSP's `src1_spad` cache is
   per-batch (`matmul-ops.c:4687`), so with one op per batch it re-quantizes the
   shared Q/K/V activation 3x and gate/up 2x.

Separately, **our CPU is also ~1.7x slower** (90.5 vs 156.2 decode) — different
kernels and graph overhead, nothing to do with the NPU.

Comparability caveats: our model is Q4_0 FC + **Q6_K** embedding/LM-head while
the GGUF is essentially all Q4_0; and llama-bench `pp` is pure prompt-processing
throughput where nntrainer's `prefill` is the whole prefill phase. Treat
single-digit-percent gaps as noise; the 3x prefill gap and the curve shape are
far too large to be artifacts.

---

## 13. Decode moved back onto the DSP (by instruction)

`min_rows` 256 -> **1**, and the env-override clamp lowered from `>= 32` to
`>= 1` — that clamp had been added specifically to stop decode reaching the DSP,
so it was actively blocking this request.

Verified all three settings on device, no rebuild between them:

| `NNTR_HEXAGON_MIN_ROWS` | prefill t/s | decode t/s | effect |
|---|---|---|---|
| **1 (default)** | 93.2 | **25.7** | everything on DSP |
| 32 | 314.3 | 93.6 | decode on CPU |
| 256 | 293.3 | 93.1 | decode on CPU |

Because the hybrid work (§9) moved the layout conversion in-process, weights
arrive as ARM `q4_0x4` regardless and the q4x4x2 copy is derived on first use.
So the threshold is a pure runtime knob — both paths stay correct at any value,
with the same `.bin`. Had `--isa HEXAGON` models been kept, this comparison
would have needed two separate weight files.

---

## 14. Every GGML_HEXAGON_* env var was dead on our path

`ggml_hexagon_init()` (`ggml-hexagon.cpp:3975`) parses **all** of them, and it is
called only from `ggml_backend_hexagon_reg()` (`:4093`). The bridge never touches
the backend registry, so `VERBOSE`, `PROFILE`, `OPPOLL`, `NHVX`, `USE_HMX`,
`OPBATCH`, `OPQUEUE` were all silently inert. Exactly the same trap as
`opt_arch`, which the bridge already had to resolve by hand for this reason.

**This invalidated an earlier claim in this log's history:** "VERBOSE=1 runs
clean, confirming the buft fix" was wrong — verbose was never actually enabled,
so the dump path that would trip `GGML_ASSERT(buft)` was never exercised. The
fix (§3) is still correct and still necessary; it simply had not been proven at
runtime at that point.

Fixed by adding `nntr_htp_bridge_resolve_opts()`, parsing the subset meaningful
without a registry. Order matters: `nhvx`/`use_hmx`/`vmem` feed
`htp_iface_start()` and `profile` gates `htp_iface_profiler()`, so all of it must
run *before* the session is constructed.

### What that unlocked: the first real per-op breakdown

`GGML_HEXAGON_PROFILE=1` prints `batch-dur-usec` (host-visible round trip) vs
`htp-ops-usec` (DSP compute) per batch. Over a full CausalLM run, 25,088 decode
ops:

| phase | ops | round trip | DSP compute | overhead | DSP share |
|---|---|---|---|---|---|
| prefill (M=6) | 196 | 180.7 us | 92.0 us | 88.7 us | 50.9% |
| **decode (M=1)** | 25,088 | **108.8 us** | **27.4 us** | **81.4 us** | **25.2%** |

Decode DSP compute is remarkably consistent (p50 27 us, p90 37, p99 40). So
**75% of decode time in the bridge is dispatch overhead, not DSP work** — the
DSP finishes each GEMV in 27 us and we spend 109 us getting it there and back.

Note first-touch ops are outliers (1674 us) — arena allocation, weight upload and
the DSP-side `HAP_mmap` all land on the first GEMM of each weight.

### OPPOLL: +20% for one line

`opt_oppoll` makes `dspqueue_read` busy-poll (timeout 0) instead of sleeping on
`DSPQUEUE_TIMEOUT`. ggml-hexagon defaults it off, correctly for its own usage:
it submits a whole graph then waits a long time, where burning a core is
wasteful. Our pattern is the opposite — one op per flush, calling thread idle
until the result lands — so sleeping adds a scheduler wakeup to every op.

| config | decode t/s |
|---|---|
| baseline | 25.27 |
| **OPPOLL=1** | **30.28** (+20%) |
| OPBATCH=4 OPQUEUE=2 | 25.65 (noise) |
| OPPOLL=1 + OPBATCH=4 OPQUEUE=2 | 30.05 |

Now the bridge default. Interesting detail: the *measured* round trip only moved
108.8 -> 102.3 us, so most of the 20% came from **outside** the profiled window
— it is the wakeup latency, not the queue read.

Ring sizing does nothing. DSP thread count is already optimal at the default:

| NHVX (with OPPOLL=1) | decode t/s |
|---|---|
| 0 = all (default) | **30.28** |
| 4 | 30.05 |
| 2 | 26.92 |
| 1 | 21.38 |

### Where decode time goes now (34.4 ms/token at 30.3 t/s)

| | ms/token | note |
|---|---|---|
| measured DSP round trips | 20.1 | 196 ops x 102 us |
| — of which actual DSP compute | 5.2 | 196 ops x 26.5 us |
| everything else | 14.3 | nntrainer CPU work + bridge host-side work |

That last row is the next target and does **not** need op batching. It is
bridge-side per-call host work sitting *outside* the profiled window: the
activation/output memcpys, rebuilding three tensor descriptors, and
`add_tensor`/`add_op`'s dedup `unordered_multimap` lookups plus the 64-byte
`op_params` copy — all per op. For reference CPU-only decode is 11.1 ms/token
*in total*, so nntrainer's own non-matmul work cannot be more than a fraction of
that 14.3 ms.

---

## 15. Next steps for single-prompt performance (no op batching)

Ranked by expected value, given the §14 breakdown:

1. **Cut bridge host-side per-call work (~14.3 ms/token bucket).** Bypass the
   `opbatch` machinery for the single-op case and write the descriptor block
   directly — it exists to dedup and pack many ops, and we always have exactly
   three tensors and one op. Avoids two multimap lookups plus the pack/reset
   cycle per GEMM.
2. **Eliminate the activation and output memcpys.** Route nntrainer's
   `MemoryPool` through an rpcmem `MemAllocator`
   (`network_graph.h:105 setComputeBackend` already exists; QNN uses it at
   `neuralnet.cpp:252`). Also needs `reuse_inference_tensor_pool_` set for cdsp,
   or pool addresses churn every token and invalidate the arena keys.
3. **The 75.8 us in-window round trip** is FastRPC/dspqueue IPC latency. OPPOLL
   already took the easy part; the rest is close to a hardware floor for
   one-op-per-submission.
4. **Floor check.** Even at zero bridge overhead, decode is bounded by streaming
   every weight from DDR per token. 27 us/op x 196 = 5.2 ms/token = ~190 t/s
   ceiling, so the DSP is not intrinsically incapable at decode — the cost is
   entirely in getting work to it one op at a time.

---

## 16. Longer-term / structural

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

---

---

## 17. The §15 plan was wrong — measured, not guessed

§15 proposed two decode optimisations on the assumption that the 14.3 ms/token
sitting outside `GGML_HEXAGON_PROFILE`'s window was bridge host-side work.
Added `NNTR_HTP_BRIDGE_PROF=1` to split our own per-call cost before building
either. Over 25,000 decode ops (us/op):

| phase | us/op |
|---|---|
| weight cache lookup | 0.0 |
| staging alloc + activation memcpy **in** | 0.1 |
| descriptor build + `enqueue_op` (all the opbatch machinery) | **0.4** |
| **`flush()` — the DSP round trip** | **125.3** |
| result memcpy **out** | 0.2 |
| total | 126.1 |

**Both planned fixes are worth ~0.4 us and ~0.3 us respectively.** Bypassing
`opbatch` to write the descriptor block directly would save 0.4 us on a 126 us
operation. Eliminating the memcpys would save 0.3 us — obvious in hindsight,
since a decode activation is 1 x 1024 x 4 = 4 KB.

The 14.3 ms/token is therefore almost entirely **nntrainer's own CPU work**
(attention, norms, Q6_K LM head, sampling), not ours: 126.1 us x 196 ops =
24.7 ms/token in the bridge, against 33 ms/token measured, leaving ~8.3 ms —
consistent with CPU-only decode being 11.1 ms/token in total.

Lesson for the log: the profiled window boundary was doing the arguing, not the
data. Two days of work avoided by twenty lines of timing.

### Everything else is already at its optimum

| lever | result |
|---|---|
| DSP clock | already pinned — `htp/main.c` sets DCVS v3 to `HAP_DCVS_VCORNER_MAX`, `sleep_disable` |
| `NHVX` (DSP threads) | default (all) 30.28 > 4: 30.05 > 2: 26.92 > 1: 21.38 |
| ring sizing (`OPBATCH`/`OPQUEUE`) | inert |
| FastRPC QoS mode | `RPC_POLL_QOS` +1% (30.27 -> 30.62); applies to sync RPC, our path is dspqueue |
| `OPPOLL` | already taken, §14, +20% |

Incidental bug found and fixed: `ggml_hexagon_session::allocate()` assigned only
`.enable` on its `remote_rpc_control_latency`, leaving `.latency` as stack
garbage. `remote.h` documents it as required for every mode but
`RPC_DISABLE_QOS`.

### So decode is at its floor for one-op-per-submission

Per-op cost decomposes as ~27 us DSP compute + ~98 us IPC round trip, and the
IPC part is a driver/hardware floor. **The only remaining lever is fewer round
trips, which is exactly op batching.** Projecting from the measured constants
(dispatch ~98 us per submission, compute 196 x 27 us = 5.3 ms/token, nntrainer
CPU 8.3 ms/token):

| submissions/token | bridge ms | total ms | decode t/s |
|---|---|---|---|
| 196 (today, one op per flush) | 24.7 | 33.0 | **30.4** |
| 112 (fuse Q/K/V and gate/up) | 16.3 | 24.6 | ~41 |
| 28 (fuse whole layer) | 8.0 | 16.3 | ~61 |
| 1 (whole forward pass) | 5.4 | 13.7 | ~73 |
| CPU reference | — | 11.1 | 90.5 |

Note even perfect batching lands at ~73 t/s, still short of CPU — consistent
with §7 and §12. But it moves decode from 30 to 41-61 t/s, which is the whole
remaining opportunity on the single-prompt path.

**This puts the current constraints in conflict:** "maximise single-prompt
performance" and "no op batching" cannot both hold any further. Everything
reachable without batching has been taken.

---

## 18. Multi-batch as a decode lever - tested, and a batch-handling bug found along the way

Question: since HMX needs M >= 32, could running N independent sequences in
parallel (batch_size = N) reach that threshold during *decode* without
touching prefill at all - M = N rows per decode GEMM, one row per sequence?

### ggml-hexagon: `llama-batched-bench -npl`

Directly answers this - N parallel sequences, aggregate decode t/s across all
of them. Qwen3-0.6B Q4_0, 4 threads:

| B (=M at decode) | CPU aggregate tg t/s | NPU aggregate tg t/s | NPU/CPU |
|---|---|---|---|
| 1 | 122.0 | 12.9 | 0.11 |
| 4 | 329.3 | 14.9 | 0.05 |
| 8 | 309.0 | 15.5 | 0.05 |
| 16 | 383.9 | 15.7 | 0.04 |
| 32 | 425.9 | 15.8 | 0.04 |
| 64 | 423.7 | 16.0 | 0.04 |

**Batching does not help NPU decode, at all**, despite crossing M=32 at B=32.
CPU aggregate throughput scales 3.5x with batch; NPU aggregate throughput moves
1.24x. The ratio gets *worse* as B grows, not better. Load log explains why:

```
sched_reserve: layer 0 is assigned to device HTP0 but the Flash Attention
tensor is assigned to device CPU (usually due to missing support)
Flash Attention was auto, set to disabled
```

Attention stays on the CPU even with `-ngl 99`. As B grows, CPU-side
attention/KV-cache work scales with it and dominates the step time regardless
of how fast HMX makes the FC matmuls - the same "attention isn't offloaded"
mechanism as SS12, now showing up as a batch-scaling ceiling instead of a
prompt-length one. NPU prefill throughput *also* falls with B (462 -> 230
t/s), likely VTCM/cache pressure from more concurrent KV state.

**Conclusion: multi-batch is not a free path to HMX-accelerated decode on this
device**, in ggml-hexagon's own mature backend. No reason to expect nntrainer's
bridge to fare better - it offloads a strict subset of what ggml-hexagon does.

### nntrainer: found and fixed a real pre-existing bug, then hit a second one

Before this could even be tested, found: `FloatTensor::dotQnK` (and
`dotQInteger`, `dotQs4cx`, the batched-weights `dot()`, and `HalfTensor`'s FP16
twin) computed `M = getDim().height()`, never reading `batch()` or `channel()`.
The standard FP32 path (`dotFloat`, via `TensorBase::calculateFlattenDot`)
already uses the correct convention -
`M = batch()*channel()*height()` for NCHW. So any activation with
`batch()*channel() > 1` through one of these quantized paths would compute
only the first slot and leave the rest of the output stale - on CPU and NPU
alike. Predates this project; never exercised because CausalLM has always run
`batch_size: 1`.

Fixed at all five call sites, mirroring `calculateFlattenDot`'s convention.
Verified two ways:

- `unittest_nntrainer_cpu_backend` (host x86): 12/12 unchanged - expected,
  since those tests call the kernels directly and never exercise `Tensor::
  dot()`'s M computation, so they could not have caught this.
- A new standalone tool (`verify_batch_fix.cpp`, not in the test suite, same
  spirit as `tools/nntr_htp_bridge_check.cpp`): builds a real Q4_0-weight
  `Tensor`, replicates one activation row across `batch=5` slots, sentinel-fills
  the output with NaN, calls the real `Tensor::dot()` -> `dotQnK` path. All 5
  rows came back identical to a batch=1 reference and NaN-free. The sentinel
  matters: before the fix this would have failed decisively (rows 1-4 still
  NaN), not just numerically.

Then tried to actually measure it: rebuilt for Android, set `batch_size: 32`,
ran on device. **Segfault, on plain CPU, before Hexagon or even the FC layer is
reached:**

```
#00 RunLayerContext::updateTensor
#01 MHACoreLayer::setBatch          (Applications/CausalLM/layers/mha_core.cpp:1502)
#02 LayerNode::setBatch
#03 NetworkGraph::setBatchSize
#04 NeuralNetwork::initialize
```

Fault address `0x7fffffff8` is almost exactly `0xFFFFFFFF * 8` - the signature
of an uninitialized/sentinel tensor index used before assignment. This is
inside CausalLM's own attention/KV-cache layer, a different subsystem from the
tensor fix above and unrelated to Hexagon. It means `batch_size > 1` has
apparently never been exercised end-to-end in this app before.

**Decision: stop here, do not chase the MHACoreLayer bug.** The ggml-hexagon
result already answers the original question - batching is not a useful lever
for decode on this device, because attention is CPU-bound and scales with
batch regardless of accelerator. The `dotQnK` fix stands on its own
correctness merits (a latent bug, now fixed and verified) rather than as a
prerequisite for a decode-throughput win that the reference implementation
shows will not materialize. Logged as a known, out-of-scope issue if
multi-sequence batching is ever wanted for a different reason.

---

## 19. Correction: what ggml-hexagon actually puts on CPU vs DSP

SS12 and SS18 left an ambiguous impression - that ggml-hexagon generally leaves
attention on the CPU. Checked directly with `GGML_SCHED_DEBUG=2 -v` (llama.cpp's
own per-node device-assignment dump; requires `-v` because llama-bench raises
its own log threshold to DEBUG only then - GGML_SCHED_DEBUG alone is not
enough, the message is emitted at GGML_LOG_LEVEL_DEBUG and the CLI's default
callback filters it out otherwise).

### Single-sequence (npl=1 - the basis for every pp/tg number in this log)

| op | count | device |
|---|---|---|
| MUL_MAT (all 7 FC per layer) | 2744 | HTP0 |
| RMS_NORM / MUL (norm weight) | 1582 / 1582 | HTP0 |
| SET_ROWS (KV cache write) | 784 | HTP0 |
| ROPE | 784 | HTP0 |
| ADD (residual) | 784 | HTP0 |
| SWIGLU | 392 | HTP0 |
| **FLASH_ATTN** | 392 | **HTP0** |
| GET_ROWS | 28 | HTP0 |
| GET_ROWS (token embedding, 1/call) | 14 | CPU |
| MUL_MAT (LM head, 1/call) | 14 | CPU |

Confirmed directly (`node #25 (FLASH_ATTN): __fattn__-0 [ HTP0 ]`) and by the
load log (`Flash Attention was auto, set to enabled`). **Correction to SS12/
SS18: attention is NOT generally left on the CPU in ggml-hexagon - only in the
batched case (see below).** Practically the entire graph runs on the DSP for a
single sequence; only two things stay on CPU per forward pass:

1. The very first token-embedding `GET_ROWS` - tiny, negligible.
2. The LM-head `MUL_MAT` (vocab x hidden) - and this one has a precise, known
   cause: `ggml_hexagon_supported_mul_mat`'s `nrows(src0) > 16*1024` VTCM-size
   guard (comment: "typically the lm-head which would be too large for VTCM").
   Qwen3's ~152K vocab is far past that. **nntrainer's own bridge mirrors this
   identical guard** (`nntr-htp-bridge.cpp`'s `N > 16*1024` check), so our LM
   head stays on CPU for the same hardware reason, not a gap in our design.

### Multi-sequence (npl>1, SS18's batched test) - attention IS pinned to CPU, and here is exactly why

`ggml_hexagon_supported_flash_attn_ext` (`ggml-hexagon.cpp:2578`):
```cpp
if (dst->ne[3] != 1) { return false; }
```
`ne[3]` is the sequence-batch dimension. The DSP's flash-attention kernel only
supports one sequence per call, by explicit design, not a general capability
gap. Confirmed directly: rerunning `llama-batched-bench -npl 4` with the same
`-v` trace reproduces the mismatch warning verbatim
(`sched_reserve: layer 0 is assigned to device HTP0 but the Flash Attention
tensor is assigned to device CPU`), and llama_context::sched_reserve
(src/llama-context.cpp:479) is where a single such mismatch anywhere in the
reserved graph disables FA **globally** for the whole run (`cparams.
flash_attn = false`), not just for the offending layer.

So SS18's explanation stands, precisely scoped: multi-sequence batching pins
attention to the CPU because of this one `ne[3] == 1` gate, and that (not a
general "attention isn't offloaded") is why NPU decode throughput does not
scale with batch size even past the HMX M>=32 threshold.

### HMX, restated precisely

Confirmed to engage for ggml-hexagon exactly as documented for our own bridge
(SS9/SS10) - `m_hmx = M & ~31`, needs `N % 32 == 0` and `K % 256 == 0` for
quantized weights. This is a DSP-side, not framework-side, property: it
applies identically whether the matmul was dispatched by ggml-hexagon's
scheduler or by our one-op-at-a-time bridge. Prefill's large M engages it in
both; decode's M=1 does not, in both.

---

## 20. Checked zhouwg's self-build-jz rewrite - no new lever, but independent corroboration

User pointed at https://github.com/zhouwg/ggml-hexagon/discussions/18 and the
`self-build-jz` branch (already present as a remote-tracking branch on our
fork, commit `56507c2bb`) asking whether it explains why our performance is
slow. Checked the actual code via a git worktree, not just the discussion.

**The discussion thread oversold what's shipped.** Three specific claims
checked against `56507c2bb`, all either wrong or unconfirmed:

1. **"JZ's dispatch avoids dspqueue overhead, faster."** His IDL
   (`kernels/ggmlop.idl:44-49`) defines a real batch call
   (`dsp_execute_batch`), but it is **not wired up** - the hot path calls
   `ggmlop_dsp_execute_task` once per op (`ggml-hexagon.cpp:5829`), same
   granularity as the official dspqueue path, just a different IPC primitive
   (raw synchronous FastRPC vs. a ring buffer). His own code says so:
   `:6698` "TODO: offload cgraph or multiple op via ggmlop_dsp_execute_batch."
2. **"JZ offloads the LM head to the DSP, ~214MB session-resident weight."**
   Zero hits for any VTCM-size guard, any `lm_head`/`output.weight`
   special-casing, anywhere in his ~11K lines of DSP-side code. Not found -
   reported as unconfirmed, not as shipped.
3. **"dsp_cache_mode / ion_sync_mode give fine-grained coherency control."**
   Neither name exists anywhere in the tree. The only cache-related call
   (`HAP_compute_res_attr_set_cache_mode(&attr, 1)`, `kernels/entry.c:172`)
   is identical to what the official backend already does
   (`htp/main.c:325`, same value). No software flush mechanism at all beyond
   an `l2fetch` prefetch hint, which is not a coherency primitive. Our own
   USAGE_WEIGHTS/USAGE_COMPUTE distinction has no counterpart here.

**One claim from the discussion turned out to be independently verifiable and
correct, from the primary source itself** (`self-build-jz`'s own README.md,
~line 617): JZ states plainly that PP (prefill) in the official backend - the
one we have been benchmarking against all session as "the reference" - is
**faster** than his own rewrite's PP, and asks for help fixing it. So the
branch we compared ourselves against throughout SS7/SS12 is already the
faster one on the metric where we have the largest gap (SS12: ~31% of
reference). An earlier auto-summary of the discussion thread claimed the
opposite (JZ 1.58x faster on PP) - that summary was wrong, likely conflating
which side's numbers were which; do not repeat it.

**Why this matters more than "no new trick found":** JZ tried swapping the
IPC transport for the exact same one-op-per-round-trip pattern we use, and it
did not move prefill. That is independent corroboration - from a completely
separate implementation, arrived at independently - of SS17's conclusion:
the bottleneck was never *which* IPC primitive carries a dispatch, it is *how
many round trips per forward pass*. His own unshipped TODO
(`ggmlop_dsp_execute_batch`) targets exactly what our queued
`gemm_q4_0_batch_fp32` work targets. Nobody, including a second independent
implementation of this backend, has shipped the fix yet.

Structural note for anyone revisiting this: `self-build-jz` is a genuine
from-scratch reimplementation (`ggml-hexagon.cpp` 7445 lines +
`kernels/ggml-dsp.c` 7366 lines + `kernels/mulmat.c` 3786 lines, separate IDL,
separate DSP-side skel), not a patch on top of `htp/`. The one architectural
idea in it that is real - a single ION pool with offset addressing instead of
per-buffer allocation - is something we already arrived at independently in
the SS8 pooled-arena rewrite.

---

## 21. The decisive measurement: even at near-zero dispatch overhead, decode still loses

Prompted by "if ggml-hexagon delegates the whole graph unlike our per-op
bridge, does it actually avoid the round-trip problem - and if the NPU is
slower, why build this at all?" Answered both with one direct measurement
rather than argument: `GGML_HEXAGON_PROFILE=1 -v` on real llama-bench decode
(Qwen3-0.6B, tg-only, `-p 0 -n 32`, HTP0, `-ngl 99`):

```
n-ops 535   batch-dur-usec ~18716   htp-ops-usec ~17759   (avg over 15 tokens)
```

**535 ops in ONE FastRPC/dspqueue round trip** - essentially the entire
per-token transformer body (28 layers x ~19 ops), matching SS19's
device-assignment trace exactly (everything except the tiny embedding lookup
and the LM head runs on HTP0). Overhead per op: `(18716-17759)/535 = 1.79us`.
Ours, one op per flush: ~98us (SS17) - **55x worse, purely from dispatch
granularity**, exactly as expected from "1 round trip vs. 196."

**And it still loses to CPU.** DSP compute alone for that 535-op batch is
17.76ms/token -> a **56.3 tok/s ceiling from compute time alone, with zero
dispatch overhead assumed**. Measured CPU is 158.9 tok/s. So even the
*already-optimal* dispatch case - which is not a hypothetical, it is what is
actually shipping - is ~2.8x too slow at the arithmetic itself. There is also
a further ~10.2ms/token gap between the 18.7ms DSP round trip and the
measured 28.9ms/token total (1e6/34.58), consistent with SS19's CPU-side
LM-head + embedding + sampling + host bookkeeping cost.

**This is the number that separates "dispatch-fixable" from "hardware-fixed"
for us specifically.** Our 196 round trips x ~98us =~19.2ms/token of pure
dispatch waste is comparable in size to the reference implementation's
*entire* DSP compute time for the whole model body - so batching (SS17's
queued gemm_q4_0_batch_fp32 work) is a real, large win for us (30 -> ~73
tok/s projected). But the destination of that work is "behind CPU for a
physical reason" (56 tok/s HVX-only compute ceiling vs. CPU's 158.9), not
"ahead of CPU" - matching almost exactly where the reference implementation
already sits (34.6 vs 158.9, SS7).

**Why build an NPU backend at all, then, if decode loses:** it is not
uniformly slower. Prefill is 1.4-3.6x faster and the advantage grows with
prompt length (SS7), because HMX is compute-density hardware that pays off
when there is enough arithmetic per byte moved to keep a systolic array fed.
Decode is the opposite: one token at a time means streaming the *entire*
weight matrix from DRAM to produce *one* output row - bandwidth-bound, not
compute-bound - and the Hexagon DSP shares the same DRAM controller as the
ARM cores, so it has no bandwidth advantage to exploit regardless of how
polished the dispatch is. That is a hardware property, not a software gap,
and it is exactly why the sensible design - prefill on NPU, decode on CPU -
is the one both this project's hybrid (SS9/SS13) and the physical evidence
above independently arrive at, rather than "NPU for everything."

---

## 22. Correction to S18: it is not "attention falls to CPU," it is a 20x DSP-side compute blowup

User's question, precisely stated: since batched decode at M=32 and single-sequence
prefill at M=32 present the same row count to the FC matmuls, shouldn't they
compute the same way? Tested directly rather than re-asserting S18's looser
explanation ("attention stays on CPU, that pulls the aggregate down").

`GGML_HEXAGON_PROFILE=1 -v`, same device, same model:

| config | n-ops | DSP compute (htp-ops-usec) |
|---|---|---|
| single-sequence prefill, M=32 | 535 | **96,600us** |
| batched decode, npl=32 (M=32 via 32 sequences x 1 token) | 647 | **1,941,000us** |

**20x slower on the DSP itself**, not a CPU-fallback effect - op count only grew
~20% (535 -> 647), so a proportional per-op slowdown cannot explain a 20x time
increase; a small number of ops must be enormously more expensive in the batched
config. (Sanity check: 32 seqs x 8 tg tokens / (8 x 1.941s) = 16.5 tok/s, matching
S18's directly-measured npl=32 aggregate of 15.8 tok/s.)

**Why "same M" does not mean "same shape."** FC/MLP layers are shared-weight
matmuls: 32 rows through the *same* weight matrix costs the same whether those
rows are 32 positions of one prompt or 32 independent decode steps. Attention is
not shared-weight. Prefill's M=32 is 32 positions of *one* sequence attending to
one common, causal-masked, growing KV cache - exactly the shape the fused
flash-attention kernel is built for. Batched decode's M=32 is 32 *independent*
sequences, each with its own separate KV history - there is no shared structure
to fuse over, which is precisely why `ggml_hexagon_supported_flash_attn_ext`'s
`dst->ne[3] != 1` gate (S19) rejects it. The decomposed fallback, even for the
ops that still land on the DSP, is dramatically more expensive per row for 32
independent small attentions than one fused kernel over 32 shared positions.

**Correction to S18: M>=32 is necessary for HMX to help but not sufficient - the
op also needs "more rows" to mean "more of the same shared work."** FC layers
qualify unconditionally. Attention only qualifies when the rows share one KV
history, which single-sequence prefill has and multi-sequence batched decode
does not. This is a sharper, quantified replacement for S18's explanation, not
a reversal of its conclusion (batching still does not help decode on this
device) - the mechanism is a 20x DSP-side compute cost, not a CPU handoff.

---

## 23. Clean side-by-side matrix: prefill / single decode / batched decode, both frameworks

Requested explicitly: one consistent pass, valid (not tiny) prefill length,
single vs. batched decode measured separately, both ggml-hexagon and
nntrainer. Rebuilt `libggml-hexagon.so` fresh from the current tree first and
pushed the identical binary (md5 `a38bb97489c441aedc4d544776e21c03`) to all
three device targets (`llamabench`, `htpbridge`,
`nntrainer/causallm`) - eliminates any doubt about whether earlier llama-bench
runs used a pre- or post- S17 QoS-fix binary (the uninitialized
`remote_rpc_control_latency.latency` fix is in shared `allocate()` code, so it
silently affected every caller, llama-bench included, from the moment it was
committed).

**Branch check (requested): neither `self-build` nor `self-build-jz` has moved.**
`origin/self-build` is exactly the commit this whole log has treated as "the
reference" since S7 - we are 4 commits ahead locally, unpushed, nothing new
landed upstream. `self-build-jz` is the identical `56507c2bb` already analyzed
in S20. Nothing new to account for from either branch.

### ggml-hexagon reference (llama-bench / llama-batched-bench, Qwen3-0.6B-Q4_0.gguf, 4 threads)

| test | CPU | NPU (HTP0) | ratio |
|---|---|---|---|
| Prefill, 512 tok, single sequence | 576.5 | **2044.3** | 3.55x faster |
| Decode, single sequence, 128 tok | 155.7 | 34.6 | 0.22x (4.5x slower) |
| Decode, batched (npl=32), aggregate | 445.1 | 16.1 | 0.036x (27.6x slower) |

### nntrainer (CausalLM hybrid bridge, ARM-layout .bin, 4 threads)

| test | CPU | cdsp | ratio |
|---|---|---|---|
| Prefill, 547 tok, single sequence | 495.9 | **588.8** | 1.19x faster |
| Decode, single sequence, 128 tok (`min_rows=1`, current default - all on DSP) | 68.0 | 26.8 | 0.39x (2.5x slower) |
| Decode, single sequence, 128 tok (`min_rows=256`, hybrid - decode on CPU) | 68.0 | 67.7 | ~1.0x (this *is* the CPU path) |
| Decode, batched | - | **blocked** | `MHACoreLayer::setBatch` crash (S18) - pre-existing, unfixed, left alone per direction at the time |

### Reading the two matrices side by side

- **Prefill**: both win on the DSP; ggml-hexagon's margin is far larger
  (3.55x vs 1.19x) because it offloads the *whole graph* (S19), while our
  bridge only offloads the FC matmuls - everything else (attention, norms,
  RoPE) still round-trips through nntrainer's CPU every layer.
- **Single decode**: both lose to CPU, but proportionally we lose less
  (2.5x vs 4.5x) - consistent with S17/S21: the decode floor is a hardware
  property (HVX-only, bandwidth-bound GEMV) neither implementation escapes,
  and the reference's superior dispatch efficiency (S21: 1.79us/op vs our
  ~98us/op) does not buy it back on decode either, because the floor is
  compute/bandwidth, not dispatch.
- **Batched decode**: catastrophic in the reference (27.6x slower) for the
  S22 reason - attention loses fused-kernel eligibility once sequences are
  independent, a 20x DSP-side compute cost, not a CPU handoff. Cannot be
  measured for nntrainer at all - genuinely missing data point, not a small
  gap, since `batch_size > 1` crashes before any forward pass runs.

---

## 24. The core structural difference, and a leveled menu of what to do about it

Asked explicitly: how does ggml-hexagon's whole-graph offload actually work
mechanically, and what does that tell us about improving nntrainer's design
(multi-batch set aside for this). Investigated with a fork grounded in exact
code citations (self-build branch), not inference from timing alone.

### Three things ggml-hexagon has that nntrainer's bridge structurally cannot, today

1. **A graph object seen all at once -> fusion + one flush.**
   `ggml_backend_hexagon_graph_compute` (`:3302-3345`) enqueues every
   HTP0-assigned node in the whole graph, then calls `flush()` **exactly
   once**, after the loop - not per split, not per node. Before that,
   `ggml_backend_hexagon_graph_optimize` (`:3403-3461`) walks the graph and
   **merges** consecutive compatible nodes (RMS_NORM+MUL -> one `htp_opnode`
   with `.fused` populated); `add_op` (`:1980-2000`) writes **one** wire-level
   op for the whole fused chain. This is real op-merging, distinct from the
   already-known `stackable()`/`same_input()` reordering (SS4/SS8), which only
   regroups independent MUL_MATs for `src1_spad` quant-cache reuse.
2. **A generic graph allocator gives it stable buffers for free.**
   `ggml-alloc.c`'s `ggml_gallocr_alloc_graph` skips reallocation and reuses
   the *same addresses* across calls when the graph shape is unchanged
   (`ggml_gallocr_needs_realloc`). Decode's shape never changes token to
   token, so `rpcmem_alloc2`/`fastrpc_mmap` for activations run once at
   `sched_reserve`, never again. **This is the exact problem SS8's pooled-arena
   rewrite solved by hand** - ggml solves it generically, for every backend,
   via its own allocator; we had to solve it ourselves becaus