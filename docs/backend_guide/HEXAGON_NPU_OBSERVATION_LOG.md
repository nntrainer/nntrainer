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
   via its own allocator; we had to solve it ourselves because we run each op through a separate FastRPC round trip with no shared
   scheduler underneath.

---

## 25. Q4_0 batching implementation attempt (Q/K/V, gate/up) — INCOMPLETE, real correctness bug found, session paused here

Following from §24's menu, attempted the first concrete item: collapse Q/K/V
(3 independent Q4_0 matmuls sharing one activation) and gate/up (2 matmuls
sharing one activation) into a single dispatch each, cutting per-token round
trips from ~196 toward ~112.

### What was built (all present in the working tree, uncommitted, HEAD is still 7e972c62)

- **`ggml-hexagon` repo**: `nntr_htp_bridge_gemm_q4_0_batch` in
  `ggml/src/ggml-hexagon/nntr-htp-bridge.cpp` — takes N (key, weight) pairs +
  1 shared activation, enqueues N ops, flushes once. Compiles clean for
  Android; symbol confirmed present in the rebuilt `libggml-hexagon.so`.
- **`nntrainer` repo**:
  - `nntrainer/hexagon/hexagon_compute_ops.cpp`: `HexagonComputeOps::gemm_q4_0_batch_fp32`
    override + `supports_gemm_q4_0_batch_fp32() -> true`, dlsym'ing the new
    bridge symbol, reusing the existing per-weight upload/scratch-buffer path.
  - `nntrainer/tensor/float_tensor.cpp`: `FloatTensor::dot(vector<Tensor*>, ...)`'s
    batch-dispatch gate changed from a hardcoded `M > 1` to
    `M >= o->gemm_q4_0_accel_min_rows()`, so Hexagon (min_rows=1) can batch at
    decode (M=1) too, matching the single-weight path's existing threshold logic.
  - **New core layers** `nntrainer/layers/qkv_layer.{h,cpp}` and
    `gate_up_layer.{h,cpp}` — LayerImpl subclasses with N independent weights
    sharing one input activation, N declared outputs via
    `context.setOutputDimensions`, dispatching through
    `Tensor::dot(vector<Tensor*> weights, vector<Tensor*> outputs)` (the
    pre-existing but until-now-never-exercised-in-this-codebase batched-GEMM
    entry point). Registered in both `app_context.cpp` (cpu) and
    `hexagon_context.cpp` (cdsp) — each `Context` has its own factory map, so
    both registrations are required for `engine=cdsp` to resolve.
  - Wired into `Applications/CausalLM/models/qwen3/qwen3_causallm.cpp`
    (`createAttention`: 3 separate `fully_connected` wq/wk/wv → one
    `qkv_layer`, outputs consumed via the experimental symbolic
    `ml::train::Tensor::output(idx)` API) and
    `Applications/CausalLM/models/transformer.cpp` (`createMlp`: ffn_up/ffn_gate
    → one `gate_up_layer`, same `.output(idx)` pattern).
  - Old dormant, broken app-level `Applications/CausalLM/layers/qkv_layer.{h,cpp}`
    (pre-existing in the repo, `forwarding()` was a no-op `return;`, unused by
    anything) deleted; references removed from that dir's meson.build and
    `Applications/CausalLM/jni/Android.mk`.

Everything above **compiles and links cleanly** — host x86 (`enable-fp16=false`,
to dodge an unrelated pre-existing FP16 KV-cache build gap in
`mha_core.cpp`/`deberta_attention_layer.cpp`, not touched this session) and
Android arm64 (`enable-hexagon-cdsp=true`), both `libnntrainer.so` and the
`nntr_causallm` executable.

### The real problem: wrong output, confirmed independent of Hexagon/Q4_0

On-device run (Qwen3-0.6B, R3CX9078DNH) produces garbage token output
(`&!(2*/.(((3(&!*,&)...`) instead of coherent text, for both:
- The Q4_0 quantized model (`nntr_qwen3_0.6b_q40_hexagon.bin`) - the actual
  target of this work.
- The **plain FP32 model** (`nntr_qwen3_0.6b_fp32.bin`) - which never touches
  Q4_0, Hexagon, or the batch dispatch path at all (FP32 activations route
  through `FloatTensor::dot`'s `input_dtype != Q4_0/QINT4` branch, which just
  loops calling the ordinary single-weight `dot()` per output - untouched,
  pre-existing code).

**Decisive bisection**: `git stash`'d every uncommitted change back to true
HEAD (7e972c62 - the exact commit the earlier phases of this investigation
had already confirmed produces coherent output), rebuilt, re-ran on-device.
**Same garbage output.** This is wrong on its face (HEAD was working before),
so something about *how the device/model state was exercised in this later
session* is suspect independent of my code - except then restoring the full
batching diff and rebuilding reproduced the FP32 garbage too, and reverting
*only* the model-wiring files (qwen3_causallm.cpp/transformer.cpp, keeping
hexagon_compute_ops.cpp/float_tensor.cpp changes) while still Q4_0 testing
also gave garbage. Net conclusion: **the bug tracks the QKVLayer/GateUpLayer +
wiring change specifically**, not the bridge/HexagonComputeOps changes, and
not a device/environment fluke - but the exact mechanism was not found before
this session was paused.

### Investigated and ruled out (with citations, so this isn't repeated)

- Weight tensor dimension construction in both new layers' `finalize()` -
  byte-for-byte identical pattern to `fc_layer.cpp`'s (`nntrainer/layers/fc_layer.cpp:83-92,117-123`).
  q_unit/k_unit/v_unit and up_unit/gate_unit formulas are unchanged copies of
  the original single-FC-layer `withKey("unit", ...)` expressions.
- Weight *loading order/offset* mechanism - NOT a simple sequential stream
  read as first assumed. `NeuralNetwork::load` (`neuralnet.cpp:944-1112`) does
  two passes: (1) `neuralnet.cpp:957-988` walks `model_graph` node order, then
  per-node `getRunContext().getWeights()` order, assigning each weight tensor
  its own `file_offset` via `Tensor::setFileOffset`; (2) actual reads happen
  **in parallel across up to 8 threads** (`neuralnet.cpp:1019-1112`), each
  weight independently seeking to its own precomputed `file_offset`
  (`TensorBase::read`, `tensor_base.cpp:69-99`: `start_offset ==
  SIZE_MAX` sentinel → use the tensor's own stored `file_offset`). Traced this
  fully and confirmed it is positionally equivalent to the old 3-separate-FC
  layout, *provided* graph node order and per-node weight request order both
  match - which they appear to.
- The generic multi-output connection-resolution path (`Tensor::output(idx)`
  symbolic API → `"producer(idx)"` connection strings → `LayerNode::setOutputConnection`
  → `NetworkGraph::setOutputConnections`) - traced in full by a forked
  sub-investigation (api/ccapi/src/tensor_api_graph.cpp:369-433,
  network_graph.cpp:195-207, layer_node.cpp:292-298, network_graph.cpp:1218-1239).
  Confirmed **fully generic over any layer declaring N outputs via
  `setOutputDimensions`** - no SplitLayer/MultiOutLayer-specific logic in this
  path. `MultiOutLayer::type` special-casing that does exist
  (`network_graph.cpp:705,756`) is for automatic fan-out insertion (one tensor
  feeding 2+ consumers), unrelated to N-distinct-outputs-from-one-layer.
- `RunLayerContext::getOutput(idx)` / `getWeight(idx)` - trivial vector
  indexing (`layer_context.cpp:246-252`), nothing special-cased.

### Not yet ruled out / where to pick this up

- **`QKVLayer`/`GateUpLayer::updateTensorsByInputDimensions`** - both new
  layers override this (added by me, unclear if actually necessary).
  `FullyConnectedLayer` **does not override it at all**, and the base-class
  default (`layer_devel.h:288-290`) **throws** "not currently supported" -
  meaning ordinary FC layers in this same CausalLM graph are never asked to
  update by input dimensions this way. My override is unconditional
  (`Updim.height(input_dimensions[0].height())` etc. with no guard), called
  from `network_graph.cpp:341`. This is the most suspicious untested piece of
  new code and the natural next thing to check - either by tracing exactly
  when/why `network_graph.cpp:341` calls this per-layer-type, or by
  temporarily stubbing the override to throw like the base class (matching
  FC's behavior) and seeing whether that either fixes output or surfaces a
  clearer error about why it's needed.
- A minimal host-only reproduction (no Android/device round trip needed,
  since the FP32 garbage confirms this is CPU-side and backend-independent):
  build a tiny 2-3 layer model using `qkv_layer` via the **string-based**
  `"input_layers=name(idx)"` connection syntax (as in every existing
  multi-output test, e.g. `test/unittest/models/unittest_models_multiout.cpp`)
  instead of the symbolic `Tensor::output()` API CausalLM uses, with known
  deterministic weights, and diff against 3 separate `fully_connected` calls.
  This isolates: is the bug in `QKVLayer` itself, or specific to the
  experimental symbolic-Tensor wiring path (which the forked sub-investigation
  traced as theoretically correct but did not verify empirically at runtime).
  The existing golden-test framework (`models_golden_test.cpp`) needs
  pre-generated golden files and was judged too heavy for a quick check; a
  raw standalone program using `nntrainer::NeuralNetwork` + `makeGraph`
  directly (`test/nntrainer_test_util.cpp`'s `makeGraph` helper) is cheaper.

### Session cost note

This debugging session (repo-internal investigation across two Bash-tool
false starts on `jni/libs` vs `Applications/CausalLM/libs` build-output paths,
several full Android rebuild+push+run cycles, and one `fork` sub-agent that
alone consumed ~666k tokens tracing the connection-resolution code) ran far
more expensive than expected for the amount of forward progress made. Flagged
explicitly by the user mid-session ("it used 5 dollars just for the above
10k tokens"). Session paused here at the user's request, with this log entry
written specifically so the next session does not need to re-derive any of
the above from scratch.

### Repo state at pause

All changes uncommitted (`git status` on `hexagon-cdsp-bridge`, 18 commits
ahead of `origin/hexagon-cdsp-bridge`, HEAD 7e972c62):
modified `Applications/CausalLM/jni/Android.mk`,
`Applications/CausalLM/layers/meson.build`, `Applications/CausalLM/meson.build`,
`Applications/CausalLM/models/qwen3/qwen3_causallm.cpp`,
`Applications/CausalLM/models/transformer.cpp`, `nntrainer/app_context.cpp`,
`nntrainer/hexagon/hexagon_compute_ops.cpp`, `nntrainer/hexagon/hexagon_context.cpp`,
`nntrainer/layers/meson.build`, `nntrainer/tensor/float_tensor.cpp`;
deleted `Applications/CausalLM/layers/qkv_layer.{h,cpp}`; new (untracked)
`nntrainer/layers/{qkv_layer,gate_up_layer}.{h,cpp}`. On-device
(`/data/local/tmp/nntrainer/causallm/`) currently has the **batched** build
pushed (garbage output); `nntr_config.json` on-device currently points at the
**fp32** backup (`nntr_config_fp32_backup.json` was copied over
`nntr_config.json` for the last test) - restore from `nntr_config_q40_backup.json`
before resuming Q4_0 work.

---

## 26. The garbage output bug, root-caused and fixed: a stale `.bin` file layout, not a QKVLayer bug

Follow-up session, working from the stash left at the end of §25. Resumed by
popping the stash (`git stash apply --index` - partial failure restoring
`qkv_layer.{h,cpp}` since they already matched in the working tree as loose
untracked files from a prior inspection; everything else applied cleanly,
`gate_up_layer.{h,cpp}` included) and picking up the one unresolved lead:
"does QKVLayer's `updateTensorsByInputDimensions` override, or the symbolic
`Tensor::output(idx)` wiring, explain the garbage?"

### Both suspects cleared before writing any test

Static read first, since it's free:

- `updateTensorsByInputDimensions` is dead code on this path. Its only
  caller is `NetworkGraph::resetInputDimension` (`network_graph.cpp:341`),
  whose only call site anywhere in the tree is
  `Applications/CausalLM/models/causal_lm.cpp:506` -
  `// model->resetInputDimension(input_dims);` - commented out, with the
  adjacent note `///@note contains possible bug`. §25's "most suspicious
  untested piece of new code" can never execute in the real binary.
- `FloatTensor::dot(vector<Tensor*>, ...)`'s FP32 branch
  (`float_tensor.cpp:785-789`) is a plain loop calling the ordinary
  single-weight `dot()` once per output - algebraically identical to three
  separate `fully_connected` calls. Not a candidate for a computation bug.

### Two independent isolation tests, both PASS

Built two small standalone programs (no gtest, no golden files - same spirit
as `tools/nntr_htp_bridge_check.cpp`), comparing `QKVLayer` against 3 separate
`fully_connected` layers with identical injected weights and identical input:

1. `tools/verify_qkv_batch.cpp` - plain `nntrainer::NeuralNetwork` +
   string-based `input_layers=name(idx)` wiring (the multiout-test style),
   equal Q/K/V units. **Exact zero diff.**
2. `tools/verify_qkv_batch_symbolic.cpp` - `ml::train::Model` +
   `LayerHandle`/`Tensor::output(idx)`, the *exact* symbolic API
   `qwen3_causallm.cpp` uses, with **asymmetric** GQA-shaped units
   (Q=8, K=4, V=4) and `weight_initializer=ones` matching the real call site.
   **Exact zero diff again.**

Between them: both wiring mechanisms CausalLM could plausibly use, both
equal and asymmetric unit configurations. `QKVLayer`'s `finalize()` /
`forwarding()` / weight-request logic is not the bug, full stop.

(Side notes picked up along the way, both dead ends but worth recording so
they aren't re-checked: `QKVLayer::finalize()`'s `Initializer::NONE` -
`Weight`'s constructor throws on that value - never actually reaches it, since
`requestWeight()` only stores a `WeightSpec`, and whatever later materializes
a real `Weight` object from it evidently substitutes a real initializer, since
`nnB.initialize()` never throws. And `weight_dim.width(k_unit)` mutating the
same `TensorDim` object used to request Q's weight moments earlier does not
retroactively corrupt Q's spec - confirmed `TensorSpecV2::dim` is stored by
value, not reference.)

### Reproducing on the real model - and finding it needs only QKVLayer

Rather than going back to the device, built `Applications/CausalLM/nntr_causallm`
for host x86 (`meson setup build-verify -Denable-fp16=false
-Denable-transformer=true`) and ran the real Qwen3-0.6B FP32 model locally -
**~2 minutes per run, no adb round trip.** (One local-only snag: the
checked-in `nntr_config.json`'s `tokenizer_file` is a hardcoded
`/data/local/tmp/...` Android path - `main.cpp`'s `resolveNntrConfigPath`
only joins a config path with the model directory when it's *relative*;
an absolute path, even one for the wrong device, is used as-is. Worked
around locally at the time by pointing a scratch copy at the real
`tokenizer.json`; fixed for real in the "Two lingering bugs" section below.)

With the batched wiring, real weights, real 28-layer model: reproduced the
exact garbage-output symptom from §25, on host, in FP32, no Hexagon involved.

Bisected by reverting just the model-wiring files
(`git checkout HEAD -- .../qwen3_causallm.cpp .../transformer.cpp`, i.e. back
to 3 separate FC layers) with everything else (the `QKVLayer`/`GateUpLayer`
classes, hexagon changes) left in place: **coherent output.** So the bug
lives specifically in swapping the model's wiring to use the batched layers,
not anywhere else in the stash.

Then bisected `QKVLayer` against `GateUpLayer` by restoring *only*
`qwen3_causallm.cpp` (`git show 900231a8:.../qwen3_causallm.cpp`, the actual
stash commit's own tree - not `9d6b1027`, the index parent, which doesn't
carry this modification) while leaving `transformer.cpp` at HEAD (gate/up
still 2 separate FC layers): **identical garbage, byte-for-byte the same
degenerate token stream.** `GateUpLayer` is not required to reproduce this at
all - the bug is `QKVLayer` alone.

### Root cause: per-node weight-offset walking assumes the old node order

Added throwaway instrumentation (`fprintf` dumps in `QKVLayer::forwarding`/
`FullyConnectedLayer::forwarding` and in `NeuralNetwork::load`'s file-offset
loop, all removed once the finding was confirmed - not left behind as a
build flag) and compared the two configurations layer 0, weight by weight:

| | 3-separate-FC (coherent) | QKVLayer (garbage) | delta |
|---|---|---|---|
| Q | offset 622333952 | 622333952 | 0 |
| **K** | **630723072** | 630722560 | **-512 B** |
| **V** | **634917888** | 634916864 | **-1024 B** |
| attention_out | 639112192 | 639112192 | 0 (recovers) |

512 and 1024 bytes are exactly one and two `gamma` vectors
(`q_norm`/`k_norm`, 128 floats each in this config). Mechanism:
`NeuralNetwork::load` (`neuralnet.cpp:957-985`) assigns each weight a file
offset by walking `model_graph` in **node order** and summing
`getMemoryBytes()` sequentially - it has no notion of tensor identity beyond
that walk. In the original topology, `q_norm` is a real graph node sitting
*between* `wq` and `wk` (`q = wq(query); q_normed = q_norm(q); k =
wk(key); ...`), and `k_norm` sits between `wk` and `wv` the same way - so the
`.bin` on disk has `[[Q][q_norm.gamma][K][k_norm.gamma][V]]`. Batching Q/K/V
into one `QKVLayer` node makes the node-order walk visit all three
contiguously, with `q_norm`/`k_norm` pushed to *after* - the loader now
assumes `[[Q][K][V]]` contiguous, so K's assumed start is short by
`q_norm.gamma`'s 512 bytes and V's by both gammas' 1024 bytes. Q is
unaffected (first weight, nothing precedes it to shift). Everything from
`attention_out` on recovers, because total bytes per layer are conserved
regardless of internal ordering - only the reordered span itself
(K, V) actually reads wrong bytes.

This explains every earlier observation: Q exactly correct at layer 0 (the
isolation tests were right - the math is fine), K and V measurably wrong but
not simply swapped (a fixed small byte shift, not a random corruption), and
the corruption compounding through every later layer (broken K/V corrupt
attention's output, which corrupts the residual stream, which is layer 1's
input - so by layer 1 even the *correct* Q weight is reading already-polluted
activations).

Not a bug in `QKVLayer`'s code, and nothing to do with Hexagon, Q4_0, or the
DSP bridge. It is a structural incompatibility between "batch N weights into
one graph node" and "the `.bin`'s byte layout is implicitly defined by
walking the *old* per-node graph order" - true of *any* layer batching that
changes relative node order against interleaved siblings (norms, in this
case), regardless of backend.

### The fix: regenerate the `.bin` from the original checkpoint, don't patch the loader

Two real options existed: teach the loader to use stable per-tensor-name
offsets (a safetensors-shaped fix, since that format already stores explicit
offsets in its header rather than relying on graph-walk order), or produce a
`.bin` whose byte layout matches the *new* node order. Took the second,
narrower option since it requires no loader change.

`Applications/CausalLM/res/qwen3/regen_qkv_bin.py` (new, checked in) re-reads
the original HF checkpoint - safetensors, read directly via `struct`+`json`+
`numpy` with a manual BF16->FP32 bit-shift (`u16.astype(uint32) << 16`), no
`torch`/`transformers`/`safetensors` dependency, and no re-download needed
since the checkpoint was already present locally
(`/storage_data/snap/ramees/SR_2026/Qwen3-0.6B`). Handles both single-file
(`model.safetensors`) and sharded (`model.safetensors.index.json`) checkpoints,
and reads `num_hidden_layers`/`tie_word_embeddings` from the checkpoint's own
`config.json` rather than hardcoding them. Emits attention weights as
`[input_layernorm, Q, K, V, q_norm, k_norm, o_proj]` - Q/K/V now genuinely
contiguous - and leaves `up_proj, gate_proj, down_proj` in their original
order, since that span was never interleaved with a norm and `GateUpLayer`
was never actually affected by this bug.

Regenerated `Applications/CausalLM/res/qwen3/qwen3-0.6b/nntr_qwen3_0.6b_fp32.bin`
in place (2,384,199,680 bytes - down from the checked-in file's 3,006,529,536;
see "Two lingering bugs" below for why the old file was larger). Verified
end to end, straight from the checked-in path with no scratch workarounds:
coherent output, matching the 3-separate-FC reference's reasoning pattern
almost verbatim.

Only `qwen3-0.6b` was regenerated this session. `qwen3-4b`'s checkpoint is
also present locally (sharded, `/storage_data/snap/sumon/
GAI-Deployment-Toolkit-v3.0.1_qwen3-4b-v0.1/Qwen3-4B`) and the script
supports it, but the actual conversion wasn't run. `qwen3-30b-a3b` (MoE) does
not use `QKVLayer` at all (`grep` across `models/qwen3_moe/`,
`models/qwen3_slim_moe/`, `models/qwen3_cached_slim_moe/` for
`qkv_layer`/`QKVLayer` - no hits) - unaffected by this bug, nothing to
regenerate there.

### Two lingering bugs, found but out of scope for this fix

- **The pre-existing local `.bin` was 622,329,856 bytes (exactly one
  embedding table) larger than nntrainer's own weight list needs.** (Note on
  "checked-in": `Applications/CausalLM/res/qwen3/*` is gitignored, so neither
  the `.bin` nor `qwen3-0.6b/nntr_config.json` is version-controlled - they
  are local working-copy artifacts. Only the *script* that regenerates them
  is committable, which is why `regen_qkv_bin.py` is the actual deliverable
  here and needed a `.gitignore` negation to be trackable at all.) Confirmed
  via the same offset-dump instrumentation: the model's last weight
  (`output_norm:gamma`) ends at exactly 2,384,199,680 - the new file's full
  size - regardless of how much trailing data follows in the file. Harmless
  in practice (the loader simply never reads past what its own `weights_spec`
  requires), but means the old file silently carried an unused ~622 MB
  tail - most likely a leftover `lm_head.weight` from tied-embedding
  handling that changed between whatever produced the checked-in file and
  the current `weight_converter.py` (which explicitly skips `lm_head` when
  `tie_word_embeddings` is true, per its own comment). Not investigated
  further; the new file simply omits it, correctly.
- **`nntr_config.json`'s `tokenizer_file` was a hardcoded
  `/data/local/tmp/nntrainer/causallm/models/qwen3-0.6b/tokenizer.json`
  Android path**, silently broken for any host run. Fixed to the bare
  `"tokenizer.json"`, which `resolveNntrConfigPath` (`main.cpp:76-86`)
  correctly joins with whatever model directory is actually passed in -
  works on host and device unchanged. Root cause not chased further, but
  matches §6's already-noted `nntr_quantize` bug ("writes its
  invocation-relative path into `tokenizer_file` instead of the bare
  filename") - plausibly the same code path, now doubly confirmed as worth
  fixing at the source rather than patching every generated config by hand.

---

## 27. Prefill acceleration: quantizer fix, zero-copy activations, and honest dispatch accounting

Continuation of §26, same session. With the `.bin` layout bug fixed, the goal
became "get real prefill numbers out of the batching work, then push prefill
further" - explicitly *not* decode (§7/§21 already established the decode
floor is a hardware property).

### A second bug: the quantizer never knew the batched layer names

The FP32 model worked after §26, but the **Q4_0** build - the one that
actually matters on device - produced the same class of garbage. Not the
§26 bug (that only concerned the FP32 `.bin`'s byte layout), so a second,
independent fault.

Found by size, before any device run: quantizing the batched topology emitted
**743 MB** where the unbatched one emitted **358 MB**. `buildLayerDtypeMap`
(`Applications/CausalLM/quantize.cpp:~477`) is a hardcoded
layer-name -> dtype table, keyed on `_wq` / `_wk` / `_wv` / `_ffn_up` /
`_ffn_gate` / etc. It has no notion of layer *class*, so `_wqkv` and
`_ffn_gateup` simply matched nothing and fell through to unquantized FP32 -
while the runtime graph still expected Q4_0. A dtype mismatch, not a data
mismatch, which is why the corruption looked similar but had a completely
different cause.

Fixed by adding `dtype_map[prefix + "_wqkv"]` and
`dtype_map[prefix + "_ffn_gateup"]` alongside the existing entries. Output
size returned to 358 MB (byte-identical accounting to the unbatched model)
and Q4_0 inference went coherent. Worth noting the structural smell: any new
batched/fused layer silently degrades to FP32 until someone remembers to add
a string here. A class-based or "unknown FC-shaped layer defaults to
fc_dtype" rule would fail loudly instead - not attempted, logged as a real
design weakness.

### Zero-copy activations, via nntrainer's own MemAllocator seam

§15/§16's item 2, finally built. §17 had measured the activation memcpy at
~0.1 us/op for *decode* (4 KB activations - negligible, as it said), but
prefill's activations are ~300x larger, so the same copy is worth revisiting
there and only there.

Implementation, all ARM-side (no Hexagon SDK, no DSP-kernel rebuild):

- **`nntrainer/hexagon/hexagon_rpc_allocator.{h,cpp}`** (new) - a
  `MemAllocator` subclass that serves allocations from `rpcmem_alloc`.
  Deliberately independent of `nntrainer/qnn/jni/rpc_mem.h`'s `RpcMem` and
  `QNNRpcManager`: those compile only into the optional QNN module (gated on
  the QNN SDK being installed), whereas `HexagonContext` is core. Uses the
  same `dlopen("libcdsprpc.so")` pattern `hexagon_compute_ops.cpp` already
  uses for its own bridge loader.
- **`HexagonContext::initialize()`** installs it via `setMemAllocator`.
- **Bridge side** (`ggml-hexagon`): new exported
  `nntr_htp_bridge_register_activation_pool(base, size)`, plus
  `ggml_hexagon_shared_buffer::wrap_external()` /
  `nntr_htp_bridge_region::init_external()` to `rpcmem_to_fd` + `fastrpc_mmap`
  a pointer *someone else* allocated (with an `owns_base` flag so `free()`
  unmaps without `rpcmem_free`-ing memory it does not own). Both
  `gemm_q4_0` and `gemm_q4_0_batch` then check each activation/output pointer
  against the registered pools (`nntr_htp_bridge_find_ext_pool`) and map it
  in place, skipping the memcpy.
- **Strictly additive by construction**: the check is per-pointer, and any
  pointer *not* in a registered pool takes the original staging+memcpy path.
  A build without the allocator wired up, or a CPU-heap activation, behaves
  exactly as before. Registration failure logs and continues rather than
  throwing, for the same reason.

Only the *activation* pool moves to rpcmem, not weights - weights already
live in the bridge's own pinned DSP arenas via `ensure_uploaded`, so routing
them here too would double-allocate. Same "tensor-only" split as the QNN
block it mirrors (`neuralnet.cpp`, `setComputeBackend("", "qnn")`).

### A timing bug in the above, worth recording because the failure was silent

First build ran clean, produced correct output, and changed nothing -
the allocator was never invoked. Cause: the `has_cdsp_engine` check was
placed in `NeuralNetwork::compile()`, but `node->getComputeEngineType()`
reads the `compute_engine` *member*, which `LayerNode::finalize()`
(`layer_node.cpp:637-639`) is what populates from the `engine=cdsp`
property - and `finalize()` does not run until `NetworkGraph::initialize()`
(via `finalizeContext`, `network_graph.cpp:~1212`), called from the separate,
later `NeuralNetwork::initialize()`. So the check always saw the
pre-finalize default and silently did nothing.

QNN's equivalent check *does* work in `compile()`, which is what made this
easy to get wrong by copying it: QNN identifies itself by
`node->getType() == "qnn_graph"` - an intrinsic layer type available
immediately - whereas an `engine=cdsp` FC/QKVLayer is an ordinary layer
distinguished only by a property that finalize() has to copy first.

Fixed by moving the cdsp check into `NeuralNetwork::initialize()`, after
`model_graph.initialize()` but still before `allocateWeights()`, so
`setComputeBackend()`'s "must be called before allocateTensors()/
allocateWeights()" contract still holds. Verified by instrumentation (since
removed): `has_cdsp_engine=1` -> allocator called -> `registered external
activation pool base 0x... size 50331648`.

### Numbers (S25 / SM-S936U, Qwen3-0.6B Q4_0 + Q6_K embd, 4 threads, 308-token prompt)

Three runs per config, prefill TPS:

| config | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| CPU | 663.8 | 588.9 | 570.4 | **588.9** |
| cdsp, QKV batching + zero-copy | 877.5 | 821.3 | 791.8 | **821.3** |

So **~1.39x CPU** on prefill, from ~1.19-1.20x before this session (§10/§23).
Attributing the two steps (single runs, so treat as indicative not tight):
batching alone measured ~703-740 TPS, zero-copy took it to ~791-877 - i.e.
roughly a 15% incremental gain from zero-copy, matching the ~10-15% estimated
up front from activation-copy volume.

Decode unchanged and left on CPU (~67-69 TPS either way, since
`NNTR_HEXAGON_MIN_ROWS=256` keeps M=1 off the DSP) - the deliberate hybrid
config from §9/§13.

Run-to-run spread is real and worth noting for anyone comparing: CPU prefill
varied 570-664 (16%) and cdsp 792-877 (11%) across three back-to-back runs,
presumably thermal/DVFS. Single-run comparisons at this granularity are not
trustworthy; the medians above are.

### Correction: gate/up batching is NOT in these numbers

Stated incorrectly mid-session and corrected here. `GateUpLayer` exists,
is registered in both `app_context.cpp` and `hexagon_context.cpp`, and now
has its quantizer entry - but its **model wiring was never restored** after
the §26 QKV-vs-GateUp bisection reverted `transformer.cpp` to HEAD.
`createMlp` still builds `ffn_up` and `ffn_gate` as two separate
`fully_connected` layers.

Actual FC dispatches per layer in the measured binary:
`wqkv`(1) + `attention_out`(1) + `ffn_up`(1) + `ffn_gate`(1) + `ffn_down`(1)
= **5**, i.e. **140 per forward pass**, not the 112 that fully-wired
FC-group batching would give. (`ffn_down` has no batching partner - nothing
shares its activation - so 4/layer = 112 is the floor for this approach.)
Wiring gate/up is therefore still-unclaimed prefill headroom with the code
already written.

### Why the remaining gap to ggml-hexagon is not a batching problem

For calibration, since "batching gave only 1.2-1.4x while ggml-hexagon is
3.5x" is the obvious question: the two levers are not the same lever.
Dispatch-count reduction helps *decode* (dispatch-bound: ~98 us IPC vs ~27 us
compute per op, §17) far more than *prefill*, which past the ~215-token
crossover (§10) is already compute-bound - HMX scales with M regardless of
how many submissions carry the work. The reference's prefill margin comes
from **scope of offload**, not dispatch efficiency: it runs attention,
RMSNorm, RoPE and elementwise on the DSP too (§19), so its CPU never
re-enters the loop mid-layer, while ours round-trips DSP->CPU->DSP every
layer for everything that is not an FC matmul. §24's leveled menu remains
the correct roadmap; this session completed level 1 (partially - see the
gate/up correction) and level 2.

### Known-unresolved: decode-on-DSP via the batched path

With `NNTR_HEXAGON_MIN_ROWS=1` (everything on DSP, §13's default) the
batched path produces *coherent but degenerate* output - repetition loops,
not the byte-garbage of §26. Isolated to the batched M=1 dispatch
specifically: hybrid mode (decode on CPU, prefill batched on DSP) is clean,
and decode-on-DSP through the older single-weight `gemm_q4_0` path was clean
in §13.

Ruled out on the ARM side: `op_batch->reset()` runs after every `flush_batch`
(`ggml-hexagon.cpp:2105`), so there is no stale-descriptor carryover between
calls despite the bridge reusing fixed `t_weights_batch[]`/`t_outs_batch[]`
slots. Remaining hypothesis, unverified: `op_matmul_hvx` sizes its VTCM
scratchpads (`src0_spad`/`src1_spad`/`dst_spad`) per-op from that op's own N
(`htp_mminit_spad`), and a batch mixes N=2048 (Q) with N=1024 (K/V) in one
flush - so differently-sized ops sharing one flush is the place to look.
Confirming it needs Hexagon SDK cross-compilation and DSP-side debug builds,
and decode-on-DSP is a throughput regression regardless (§7/§21), so it was
deliberately not chased. **Do not use `NNTR_HEXAGON_MIN_ROWS=1` with the
batched layers until this is resolved.**

### Direction from here: prefill only

Per direction, decode work is dropped as a goal. Next prefill steps, in
order:

1. **Wire gate/up** (`transformer.cpp` `createMlp`) - code already written,
   takes 140 -> 112 dispatches/token. Smallest remaining effort.
2. **Offload norm/RoPE/elementwise via `ComputeOps`** - §24 confirmed both
   the seam and the DSP-side kernels already exist. This is the first lever
   that attacks prefill's actual bottleneck (per-layer CPU round trips)
   rather than dispatch count, and is where the remaining multiple lives.
3. **Attention + DSP-resident KV cache** - the only route to the reference's
   round-trip count, but revisits the project's founding premise (§24).

### S27 addendum: gate/up wired - dispatch count confirmed down 20%, prefill unmoved

Wired `GateUpLayer` into `transformer.cpp`'s `createMlp` (the step S27 flagged
as still-unclaimed). No `.bin` change needed, unlike QKVLayer: node order goes
`[ffn_norm.gamma][up][gate][down]` either way, because nothing is interleaved
between up and gate the way `q_norm`/`k_norm` are between Q/K/V.

Correct - output is token-identical to the pre-wiring run.

**Round-trip reduction confirmed directly** rather than inferred from wall
time, via `GGML_HEXAGON_PROFILE=1`'s per-batch `n-ops`, counted over one
prefill forward pass:

| flush size | count | ops |
|---|---|---|
| 3 ops | 28 | Q/K/V batched |
| 2 ops | 28 | up/gate batched |
| 1 op | 56 | attention_out + ffn_down |

= **112 flushes carrying 196 GEMMs**, i.e. exactly the predicted 4 dispatches
per layer, down from 140. (196 also cross-checks against the long-known
"196 FC GEMMs per token" figure from S6.)

**And prefill did not move: median 821.3 TPS both before and after** (5 runs
after: 922.2 / 853.2 / 821.3 / 797.9 / 795.9). A 20% cut in round trips
bought nothing measurable.

That is not a disappointment, it is the cleanest confirmation yet of what
S10/S12 argued and S27 predicted: **past the ~215-token crossover prefill is
compute-bound, so dispatch count is simply not its lever.** Because the op
count was verified directly, this conclusion needs no thermal control - the
mechanism provably changed while the outcome did not.

Worth noting for anyone re-measuring: the 5 runs above drift monotonically
downward (922 -> 796, ~14%) purely from thermals over ~40s of back-to-back
runs. Comparing medians of separately-taken batches is unreliable at this
granularity; either interleave configs or, better, measure the mechanism
directly as done here.

Kept anyway despite the null prefill result: it is correct, free, and a 20%
round-trip cut is real work removed - it would matter if decode is ever
revisited (and it lets the DSP's per-batch `src1_spad` cache skip
re-quantizing the shared activation, per S2's finding 10). It also carries
the same batched-M=1 caveat as QKVLayer, so it stays behind the hybrid split.

**Dispatch-count reduction for prefill is now exhausted** (`ffn_down` and
`attention_out` have no activation-sharing partner, so 4/layer is the floor
for this approach). The remaining prefill headroom is entirely in scope of
offload - norm/RoPE/elementwise, then attention - per S24's menu.

---

## 28. Where prefill time actually goes: attention is 57% of it

Measured rather than argued, in response to "how is ggml-hexagon much faster".
Two decompositions, both on the same device in the same thermal window, with
the **identical** `libggml-hexagon.so` pushed to both `llamabench/` and
`nntrainer/causallm/` (S23's lesson).

### Cross-framework readings, 308-token prompt, 4 threads

| | CPU | NPU | ratio |
|---|---|---|---|
| ggml-hexagon (`llama-bench -r 3`, mean±sd) | 586.5 ± 31.7 | **1636.6 ± 40.6** | **2.79x** |
| nntrainer (interleaved pairs) | 341-629 | 542-830 | **~1.4x** (1.15-1.61) |

Methodology notes that matter more than the absolutes here:

- `llama-bench` does internal repetitions and reports mean±sd (tight, 2-5%);
  our harness is single-shot per invocation. Our absolutes swung **35%**
  within one 5-run batch purely from throttling, so cross-batch
  median-vs-median is not a valid comparison at this granularity. The
  **ratios** (each framework's own CPU baseline taken adjacent to its NPU
  run) and the breakdowns below are the trustworthy parts.
- Interleaving CPU/cdsp runs pairwise revealed something useful on its own:
  **the cdsp/CPU ratio *rises* as the device heats** (1.32 -> 1.58 across four
  pairs, as absolutes fell 629->341 CPU / 830->542 cdsp). The CPU throttles
  harder than the DSP, so NPU offload is worth *more* on a hot device - i.e.
  in realistic sustained use, not less.
- Our CPU baselines are broadly comparable to llama.cpp's (586 vs 341-629),
  so the gap is in the NPU path, not in a weaker CPU reference.

### Our prefill, split DSP vs CPU (bridge profiler, GGML_HEXAGON_PROFILE=1)

Summed over the 112 prefill flushes of one 308-token forward pass:

```
prefill wall time      : 353 ms
  DSP path (196 GEMMs) :  58.4 ms  (17%)
    - DSP compute      :  49.1 ms
    - dispatch overhead:   9.4 ms   <- only 2.7% of total prefill
  everything else      : 294.6 ms  (83%)
```

This retroactively explains S27's null gate/up result exactly: dispatch
overhead is 2.7% of prefill, so cutting round trips 20% could not possibly
have shown up. It also bounds the whole matmul-side effort - making the Q4_0
GEMMs *infinitely fast* would only reach 1.20x.

### Our prefill, split per layer type (temporary instrumentation, since removed)

Wall time per layer type across the first forward pass. nntrainer's built-in
`PROFILE_TIME` hooks in `NetworkGraph::incremental_forwarding` already key on
`ln->getType()` and would give this, but wiring them up needs
`-Denable-profile=true` on *both* the core and the app plus a
`GenericProfileListener` subscription that CausalLM does not have (SimpleFC/
Resnet/MNIST do). A throwaway `std::chrono` loop in the same place was
cheaper, and let the diagnostic ride in `libnntrainer.so` alone - so it could
be pushed as a single library with no app rebuild.

| layer | ms | % | runs on |
|---|---|---|---|
| **mha_core** (attention + RoPE + KV cache) | **190.2** | **56.7%** | CPU |
| fully_connected (attention_out + ffn_down, n=56) | 33.2 | 9.9% | DSP |
| gate_up_layer | 32.3 | 9.6% | DSP |
| qkv_layer | 28.4 | 8.5% | DSP |
| swiglu | 17.6 | 5.3% | CPU |
| tie_word_embeddings (embedding + LM head, n=2) | 15.9 | 4.7% | CPU |
| reshaped_rms_norm (q_norm/k_norm, n=56) | 8.5 | 2.5% | CPU |
| addition (residuals, n=56) | 4.5 | 1.3% | CPU |
| rms_norm | 4.4 | 1.3% | CPU |
| multiout / input | ~0 | 0% | - |

(The DSP-dispatching layers total 93.9 ms of *layer* wall time against 58.4 ms
of measured DSP round trip - the ~35 ms difference is nntrainer-side per-layer
overhead around the dispatch: tensor setup, the `getSharedDataTensor`
windowing in `incremental_forwarding`, etc. Worth a look eventually, but it is
10% of prefill, not the story.)

### This kills the plan S27 closed with

S27 ended by recommending "offload norm/RoPE/elementwise via ComputeOps" as
the next lever. **That was wrong, and the numbers say so plainly.** Two
independent reasons:

1. **The cheap ops are only 10.4% combined** (swiglu 5.3 + reshaped_rms_norm
   2.5 + addition 1.3 + rms_norm 1.3). Offloading *all* of them perfectly caps
   out at 1.12x.
2. **Per-op offload would make it worse, not better.** A round trip is ~100 us
   (S17/S21); an RMSNorm over 308x1024 fp32 is arithmetically trivial and
   cheaper than that on CPU. Dispatching each one separately would take us
   from 112 flushes to ~280 and *add* time. The gain was only ever in removing
   the CPU handoff, which per-op offload does not do - it needs contiguous
   runs of ops kept resident on the DSP, which is what ggml-hexagon's
   single-flush graph offload plus RMS_NORM+MUL fusion (S24) actually
   achieves, and which our per-tensor-op `ComputeOps` seam cannot express.

Also corrected: RoPE is **not** independently offloadable. There is no
standalone rope layer - it lives inside `mha_core.cpp` (45 references) together
with attention and the KV cache. It comes bundled with the attention work.

### So attention is the entire remaining opportunity

Amdahl, from the 335.2 ms measured pass:

| change | prefill | vs now |
|---|---|---|
| all cheap ops -> 0 (unachievable, see above) | 300.2 ms | 1.12x |
| mha_core halved | 240.1 ms | 1.40x |
| mha_core -> 0 | 145.0 ms | **2.31x** (~2120 TPS, past ggml's 1637) |

`tie_word_embeddings`'s 15.9 ms is *not* addressable: the LM head exceeds the
DSP's VTCM guard (`N > 16*1024`), which our bridge mirrors from
ggml-hexagon's own `ggml_hexagon_supported_mul_mat` - they leave it on CPU for
the identical hardware reason (S19).

Which means the answer to "how is ggml-hexagon 2.8x where we are 1.4x" is
fully quantified and singular: **they run attention on the DSP as a fused
FLASH_ATTN kernel; we run it on the CPU, and that one op is over half our
prefill.** Not dispatch count, not kernels, not memory layout - all of those
are now measured and bounded well under what attention alone costs.

The next step is therefore the one S24 flagged as a founding-premise decision,
with no cheaper intermediate step left standing: offload attention, which
brings RoPE and a DSP-resident KV cache with it.

---

## 29. Attention decomposed, and the DSP already has every kernel we need

Follow-up to S28, which established `mha_core` as 57% of prefill. Two
questions had to be answered before choosing an offload strategy: what inside
attention actually costs, and what the DSP can already do.

### Confirmed first: ggml-hexagon really does run attention on the DSP

Re-verified directly rather than trusting the log, because S18 and S19 say
different-sounding things and the distinction decides whether "offload
attention" is even the right target. `GGML_SCHED_DEBUG=2 ... -v`:

```
node # 25 (FLASH_ATTN): __fattn__-0 (8K) [ HTP0 ] use=1,c=1:
    Qcur-0 (view) (permu (8K) [ HTP0 ]
    cache_k_l0 (view) (p (512K) [ HTP0 ]
    cache_v_l0 (view) (p (512K) [ HTP0 ]
```

Every layer's FLASH_ATTN is on HTP0, **and so are `cache_k_l*`/`cache_v_l*`** -
the KV cache is DSP-resident, as S24 said. S18's "attention stays on CPU"
applies **only to multi-sequence batched decode** (`npl>1`), where
`ggml_hexagon_supported_flash_attn_ext`'s `dst->ne[3] != 1` gate rejects it;
S19 scoped this correctly. For single-sequence - every pp/tg number in this
log - their attention is on the DSP and ours is not.

### Attention's internal phases (temporary instrumentation, since removed)

Accumulated across all 28 layers of one 308-token prefill pass, inside
`MHACoreLayer::one_batch_incremental_forwarding` (the non-`sink_step`
overload - the other is gpt-oss only):

| phase | ms | % of attention | % of prefill (349 ms) |
|---|---|---|---|
| RoPE + KV-cache write (3x `apply_rotary_emb_tensor_v2`) | 50.1 | 26.0% | 14.4% |
| **Q.K^T** (`compute_kcaches`) | **68.0** | **35.4%** | 19.5% |
| softmax, causal-masked (`softmax_triangle`) | 10.4 | 5.4% | 3.0% |
| **scores.V** (`compute_fp16vcache_transposed`) | **63.7** | **33.2%** | 18.3% |
| total | 192.2 | | 55.1% |

Cross-validates S28's 190.2 ms layer-level figure from a completely separate
measurement, which is reassuring for both.

**The two matmuls are 131.7 ms = 68.5% of attention and 38% of all prefill.**
Softmax is only 10.4 ms - so the *fusion* in "flash attention" is worth far
less here than the *matmuls*. That matters: it means a plain-GEMM offload
captures most of the win without needing flash-attention semantics (mask
handling, online softmax, tiling) to be reproduced exactly.

### The DSP skel already implements everything we run on CPU

Checked `htp/htp-ops.h`'s opcode enum. The skel we are *already loading*
(`libggml-htp-v79.so`) implements:

`HTP_OP_MUL_MAT`, `HTP_OP_FLASH_ATTN_EXT`, `HTP_OP_ROPE`, `HTP_OP_SOFTMAX`,
`HTP_OP_RMS_NORM`, `HTP_OP_RMS_NORM_MUL` (fused), `HTP_OP_GLU_SWIGLU`,
`HTP_OP_ADD`, `HTP_OP_SET_ROWS` (KV-cache write), `HTP_OP_SCALE`,
`HTP_OP_CPY`, `HTP_OP_GET_ROWS`, ...

i.e. **a DSP kernel exists for every single op we currently run on CPU.** We
have never dispatched any of them - our bridge only ever emits
`HTP_OP_MUL_MAT` with Q4_0 weights. So all remaining work is bridge + nntrainer
wiring, **not DSP kernel development**, which is a much better position than
S24 implied.

And `MUL_MAT` is not quantized-only: `ggml_hexagon_supported_mul_mat`
(`ggml-hexagon.cpp:~2715`) accepts `src0` of `GGML_TYPE_F16` or
`GGML_TYPE_F32` (with `dst` F32, `src1` F32/F16), subject to
`ggml_nrows(src1) <= 1024` - our 308 prefill rows are fine. So the attention
matmuls need no new kernel and no new opcode.

### Shared prerequisite for any attention offload: the KV cache must be DSP-visible

Both candidate paths need the DSP to read K and V. Today the KV cache is host
memory owned by CausalLM's `KVCacheManager` and bound into the graph as
external tensors, so it is invisible to the DSP. Two options: memcpy K/V into
rpcmem per layer per call (~1.26 MB per layer per tensor at 308 tokens, ~35 MB
per forward pass - measurable but not fatal), or allocate the cache in rpcmem
once and register it with the bridge.

The second is clearly right and the machinery already exists: S27's
`nntr_htp_bridge_register_activation_pool` plus `wrap_external`/`init_external`
were built exactly for "someone else allocated this rpcmem, please map it".
The cache is allocated once and lives for the session, which is the ideal case
for a pinned registered pool.

### Two paths, and why the smaller one is likely right first

| | addressable | needs |
|---|---|---|
| **A. offload Q.K^T and scores.V as F16/F32 MUL_MAT** | 131.7 ms (38% of prefill) | KV cache in rpcmem; a bridge entry point for non-Q4_0 GEMM; per-head dispatch |
| **B. offload all of attention as FLASH_ATTN_EXT** | 192.2 ms (55% of prefill) | all of the above, plus reproducing ggml's exact Q/K/V/mask tensor layout and semantics, and matching KV layout |

A captures 69% of what B does for materially less risk, and is a strict
stepping stone - it forces the KV-cache-in-rpcmem work and a non-Q4_0 GEMM
path, both of which B also needs. RoPE (50.1 ms) can then be taken separately
via the existing `HTP_OP_ROPE`, again without touching B's semantics.

One sizing note for A: attention is per-head - 16 heads of
`[308,128] x [128,308]` per layer. `NNTR_HTP_MAX_BATCH` is currently 8, so
16-head dispatch either raises that cap or issues two batches per matmul.
