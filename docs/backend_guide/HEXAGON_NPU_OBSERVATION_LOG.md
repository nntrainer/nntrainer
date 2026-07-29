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
