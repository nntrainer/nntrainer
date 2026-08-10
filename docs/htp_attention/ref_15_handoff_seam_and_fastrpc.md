# 15 — Handoff: the ComputeOps seam and the FastRPC problem

State of the work as of branch `claude/hexkl-mha-hmx-optimization-6ycsx0`,
2026-08-04. Supersedes [12](12_handoff.md) as the current handoff document.
Read [14](14_mha_fp16_micro_verified.md) first for the verified attention
numbers, then [13](13_fc_micro_dma_and_mha_plan.md) §3a for the FC ones.

**Start here if you are picking this up.** §8 has the exact next task.

---

## 1. One-paragraph summary

Both `mha_core` attention matmuls and the FC matmul are now measured, on
device, running 1.7-65× faster on hand-written micro-API HMX kernels than the
alternatives they'd replace ([13](13_fc_micro_dma_and_mha_plan.md),
[14](14_mha_fp16_micro_verified.md)). **None of it is callable from
production.** Every one of those numbers was measured inside a standalone DSP
program with no ARM↔DSP call boundary in the timed region, and there is no
mechanism today for `nntrainer` on the ARM side to invoke any of those
kernels. Building that mechanism is the remaining work, and it turns out
someone else has already built most of it: **upstream PR #4236** (open, by
dlwlzzero) ships a working custom FastRPC skel plus the HVX
quantize/dequantize kernels this branch's documents kept excluding as "the
HVX owner's scope." The next task is not to build a second skel — it is to
**port this branch's verified optimisations onto #4236's interface**, which
also requires migrating FC dispatch off the SDKL macro API because a measured
hardware constraint makes macro and micro mutually exclusive in one process.

---

## 2. The one measurement that constrains the whole architecture

**Once a process calls HexKL's micro API, it can never open an SDKL macro
session again.** Measured with
`test/unittest/jni_htp/hexagon/hexkl_hmx_handoff_probe.c` (this session,
reproducible across 3 fresh runs):

| direction | result |
| :-- | :-- |
| macro `initialize`+`unlock_hmx` → `hexkl_micro_hw_init` | **works**, 556-559 µs, every time |
| `hexkl_micro_hw_init` → then `hexkl_macro_initialize` | **fails permanently**, `0x80000401` (`AEE_EFAILED`) |

The failing direction never recovers: 12-attempt exponential backoff (~204 ms)
and a separate flat 2000 ms wait both failed. `hexkl_micro.h` has no teardown
paired with `hw_init`, so there is nothing to release. A control confirmed
macro can cleanly re-open its own session when no micro call is involved, so
this is not general flakiness. Simultaneous-lock failure was already known;
what is new is that it is **irreversible, and asymmetric**.

**Consequence.** Production FC dispatch (`hexkl_mm.cpp`) uses the macro API
(`sdkl_npu_mm_f32f16_f32`, `sdkl_npu_mm_u8i8_i32`). The verified attention
kernels and PR #4236 both use the micro API. A real forward pass needs both.
So "keep two skels and hand HMX ownership back and forth per layer" is not
slow — it is **impossible**. FC must migrate to the micro API. This is not a
preference; it is the only configuration that runs.

---

## 3. The FastRPC cost, measured for the first time

Everything in 13 and 14 was measured via `run_main_on_hexagon`, which is
itself a FastRPC call — but exactly **one** call ("upload this program and run
its `main()`"), with all of the DMA and tile work happening inside, ARM
uninvolved. Those benches also fed themselves synthetic data generated
on-DSP; no real ARM-side tensor ever crossed the boundary.

`test/unittest/jni_htp/hexagon/skel_probe/` (this session, uncommitted)
built the first real custom skel on this branch and measured the crossing.
N=50, 5 warm-up discarded, `sample` shape (M=64,N=128,K=128, u8i4),
correctness verified exact against an ARM-side scalar reference:

| | µs |
| :-- | --: |
| ARM wall-clock per call (full round trip) | ~494 |
| DSP HMX matmul (`HAP_perf`) | ~254 |
| DSP weight bake, re-paid every call | ~72 |
| **net FastRPC transport overhead** | **~166-170** |

That is **~1.7× SDKL's own production per-op figure of ~97 µs**
([13](13_fc_micro_dma_and_mha_plan.md) §2). Read it as an **upper bound, not
a verdict**: this probe was a Debug build, marshalled through plain `malloc`
rather than `rpcmem`/ION zero-copy, and re-baked the weight on every call.
All three are fixable and all three inflate the number.

**Why it matters more the faster the kernel gets.** ~170 µs is a fixed
per-call toll. Behind a 56 µs FC matmul it triples the cost; behind a whole
layer's attention (239 µs, doc 14) it is a 70% adder that still leaves a win;
behind a per-kv_head call (×8) it would erase the win entirely. **Call
granularity is now a first-class design variable** — bigger per-call work
units, and `dspqueue` batching if many calls remain unavoidable. Doc 13 §2
assumed "a persistent skel with `dspqueue` batching, so no per-op RPC" without
ever building one; this is the first actual number for it.

---

## 4. What upstream PR #4236 already provides — do not rebuild it

Open PR, author dlwlzzero, title "[htp] complete A8W4 DSP quantization
pipeline with HMX matmul and activation quant". Not merged as of this writing;
`upstream/main` is 10 commits ahead of this branch and already contains the
reviewed/merged form of #4162 (the HTP skeleton).

| file | what it is |
| :-- | :-- |
| `test/htp/nntr_hvx.idl` | **a working custom FastRPC interface** (`remote_handle64`), endpoint `mm_u8i4_from_f32` |
| `test/htp/build.sh` | `qaic` → stub/skel → `libnntr_hvx_skel.so`; links `libhexkl_micro.a` |
| `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4.c` | VTCM planning, one-shot weight bake, HMX u8i4 matmul loop — **in the production tree** |
| `nntrainer/tensor/htp_backend/hvx/hvx_quant_u8.c` | f32 → per-row asymmetric u8 dynamic quant, writing **straight into HMX AH layout** |
| `nntrainer/tensor/htp_backend/hvx/hvx_dequant_i32.c` | int32 accumulator → f32, with zero-point correction |
| `nntrainer/tensor/htp_backend/hvx/hvx_convert.h` | int32↔f32 vector conversion (HVX has no such instruction) |
| `test/unittest/unittest_hvx_mm_u8i4.cpp` | 476-line accuracy suite, S1-S4 per shape |

The dequant formula documents the whole A8W4 scheme:
`out[m][n] = (acc[m][n] - act_zp[m]*colsum_w[n]) * act_scale[m] * w_scale[n] + bias[n]`
— per-row dynamic activation scale/zp, per-column weight scale, `colsum_w` as
the zero-point correction.

### Three of this branch's findings independently confirmed there

- **VTCM is ~8 MB on S25 Ultra V79** ("~13× the design's 598 KB peak
  footprint"). Matches `hexkl_micro_hw_init` in both of this branch's benches.
  [08](08_attention_hmx_design.md) §10's 18 MiB (from
  `sdkl_npu_get_hw_info`) is the outlier; budget against 8 MB.
- **Baking every weight tile up front beats HexKL's example**, which converts
  inside the innermost loop — same finding as this branch's
  `load_weight_resident`.
- **`copy_submatrix` is a debug helper**: #4236 avoids it because "its own
  header marks [it] as debug-only." This branch measured *why* — extraction-
  style micro functions cost ~26 µs/tile from a VTCM source, ~400× a
  VTCM→VTCM DMA of the same bytes ([14](14_mha_fp16_micro_verified.md) §5.3).

### But it is an accuracy harness, and says so

The IDL comment reads "Accuracy harness: the whole flow, quantization and
dequantization on the DSP. Intermediate buffers come back so each stage is
checkable." Four things make it unfit as a performance path, each of which
this branch has already measured the fix for:

1. **`hmx_bringup()` runs on every invoke** — `hexkl_micro_hw_init` plus
   `hmx_lock`/`unlock` per call. Belongs in an `open()`/`close()` pair, once
   per model.
2. **`bake_weights` runs on every invoke.** This is precisely the cost
   [11](11_decode_time_budget.md) measured as 39.3 ms of a 102 ms token, and
   [14](14_mha_fp16_micro_verified.md) §5.5 measured the fix as 32× (full
   re-bake 123.7 µs/head vs incremental 3.35 µs/head).
3. **Four debug `rout` buffers** (`act_u8_ah`, `act_scale`, `act_zp`,
   `acc_i32`) marshalled back every call — `m_pad*K + m_pad*N*4` bytes of
   pure overhead, on top of §3's already-significant transport cost.
4. **No user-DMA, no cross-matmul prefetch, no dmlink ring.** Output goes
   through scalar `copy_32b_to_submatrix`.

### One constraint to re-check before u8i4 attention

`hvx_convert.h`'s magic-number int32→f32 identity is documented exact only for
`|x| ≤ 2^22 = 4,194,304`, with accumulators bounded by `255*8*K` — 2,088,960
at K=1024, so ~2× margin. Attention `P·V` has `K = kv_len`; at kv=2048 that
bound is 4,177,920, essentially touching the limit. Note that the K3 dequant
path actually uses `Q6_Vsf_equals_Vw` (a real numeric conversion, per that
header's own comment), so the bound may not bind there — **verify which path
applies before assuming u8i4 attention is safe at long context.**

---

## 5. What this branch has that must be carried forward

All committed; the benches remain runnable reference implementations. **The
optimisations are here, not in `skel_probe/`** — that directory deliberately
used the simple `matmul_resident` path because proving the FastRPC mechanism,
not speed, was its job.

`test/unittest/jni_htp/hexagon/hexkl_micro_fc_bench.c` (commit `e031d03bc`,
u8i4 + u8i8):

| component | function |
| :-- | :-- |
| VTCM tiling plan | `plan_vtcm` |
| one-shot weight bake, reused | `load_weight_resident` |
| raw user-DMA | `hexdma_start/wait/2d_ddr_to_vtcm/2d_vtcm_to_ddr/_async/poll_done` |
| dmlink ring (async chaining) | `dmlink_`, `ring_init_lite`, `ring_push2d`, `ring_drain` |
| **cross-matmul weight prefetch** | `run_pipelined_layer` — **the 1.7-2× itself** |
| pipelined output DMA | `matmul_resident_dma` |

`test/unittest/jni_htp/hexagon/hexkl_micro_mha_bench.c` (commit `5620855f8`,
fp16 attention):

| component | function |
| :-- | :-- |
| **layer-wide contiguous DMA** | `matmul_layer_dma_f16` — the 53 GB/s path |
| VTCM→VTCM tile repack | `push_band_repack_vtcm` |
| **incremental append-time bake** | `prebake_pv_weight`, `restage_pv_weight`, `bench_incremental_bake_pv` |
| Q·Kᵀ transposed bake | `prebake_qk_weight_transposed` |
| ring/DMA, fp16 variants | same set as above |

**Learn from #4236 in return:** its quant kernel writes activations *directly*
into AH layout, which is better than this branch's two-step
`copy_submatrix_to_f16` + `rm_to_ah_f16`. Adopt that.

---

## 6. Build knowledge paid for in this session

Both were silent failures that cost real time:

- **The SDK's `make android` hardcodes an NDK path** (`tools/android-ndk-r25c`)
  that does not exist in this install, plus a `clang_rt.builtins` path baked to
  clang 14.0.7. Override with `ANDROID_ROOT_DIR` and `STATIC_LIB_PATH=`; the
  real values are discoverable in `build/make.d.ext/android/*.min`.
- **`rules.min` splices a link target's `_LD_FLAGS` before its object files**,
  so a plain archive path for `libhexkl_micro.a` links as an unresolved
  dependency (`nm` shows `U`, not `T`). Wrap it in
  `--whole-archive`/`--no-whole-archive`. Also: per-source compile flags
  (`_INCDIRS`/`_CC_FLAGS`) are keyed by the **source file's basename**, not the
  link target's name.
- **On this device, unsigned-PD FastRPC needed no testsig at all** — verified by
  removing it and re-running successfully 3×. If a device/security config does
  need one: read `/sys/devices/soc0/serial_number` (decimal), convert to hex,
  then `echo y | python3 $HEXAGON_SDK/tools/elfsigner/elfsigner.py -t 0x<hex>
  -o <outdir>`, and push `testsig-0x<hex>.so` into the same
  `ADSP_LIBRARY_PATH` directory as the skel.
- **Running #4236's `build.sh` here needs overrides** — its defaults are
  `HEXKL_ROOT=$HOME/Downloads/hexkl_addon` and `HEXKL_SDK_VER=6.4.0.1`:
  ```bash
  source ~/workspace/Hexagon_SDK/6.4.0.2/setup_sdk_env.source
  HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon HEXKL_SDK_VER=6.4.0.2 \
    ./test/htp/build.sh
  ```
  (v79 `libhexkl_micro.a` exists under `6.4.0.0/6.4.0.1/6.4.0.2/6.5.0.0/6.5.0.1/6.6.0.0`.)

---

## 7. Reproducing every number in 13, 14 and this document

```bash
./test/unittest/jni_htp/hexagon/run_micro_fc_bench.sh     # u8i4 vs u8i8 FC, doc 13 §3a
./test/unittest/jni_htp/hexagon/run_micro_mha_bench.sh    # fp16 attention DSP, doc 14 §3
./test/unittest/jni_htp/hexagon/run_mha_cpu_bench.sh      # ARM CPU bars, doc 14 §3
./test/unittest/jni_htp/hexagon/run_hmx_handoff_probe.sh  # §2 (uncommitted)
./test/unittest/jni_htp/hexagon/run_skel_probe.sh         # §3 (uncommitted)
```

All run clean on a plain invocation with no env overrides, on a Galaxy S25
Ultra V79 (`R3CY10WM83Y`). Every one reports through self-describing marker
lines (`FC_FIELD`/`MHA_FIELD`/`HANDOFF_FIELD`/`SKEL_FIELD`
`field=... value=...`) rather than positional parsing — a report script
silently dropping shapes and printing a stale column happened twice in this
session before that convention was adopted.

---

## 8. The next task, in order

1. **Reconcile with #4236 and upstream.** This branch is 10 commits behind
   `upstream/main`, which already carries the merged form of #4162 — and that
   merge dropped the kernel-wiring parts (`HtpComputeOps`, `supports_shgemm*`,
   the u8i8 BLAS virtuals) that this branch then built 38 commits on top of.
   Rebasing is a real task, not a formality. Decide it before writing new
   production code, or it gets rewritten twice. Also decide what to do with
   `e1f5bb480` — a commit this branch carries that device A/B showed crashes
   and regresses (memory `htp-decode-wh-cache-fix-broken`); carrying a known
   broken commit through a rebase makes future `bisect` useless.
2. **Retire `skel_probe/` in favour of #4236's `test/htp/`** — it is a
   duplicate skel. Keep only §3's measurement methodology, re-aimed at
   `nntr_hvx.idl`. §6 already records the build knowledge, so the files
   themselves are safe to drop.
3. **Turn #4236's harness into a performance path.** §4's four items, using
   §5's verified components: `open()`/`close()` holding `hw_init` + HMX lock;
   a persistent weight cache; drop the debug `rout` buffers; add user-DMA
   output and cross-matmul prefetch. Then re-measure §3's per-op cost on
   *that*.
4. **Migrate FC dispatch off SDKL macro** (`hexkl_mm.cpp`'s
   `sdkl_npu_mm_f32f16_f32` / `sdkl_npu_mm_u8i8_i32`) onto the same skel. §2
   makes this mandatory, not optional. #4236's DSP-side quant/dequant means
   the seam signature can be **f32 in → f32 out**, which is simpler than what
   `hexkl_mm.cpp` does today (host-side quantisation).
5. **Add attention endpoints to the same IDL** (`attn_qk_f16`, `attn_pv_f16`),
   porting §5's fp16 kernels.
6. **Then the `mha_core` seam**: two `ComputeOps` virtuals plus
   `supports_*()` gates, replacing the direct `nntrainer::compute_kcaches` /
   `compute_fp16vcache_transposed` free-function calls at
   `mha_core.cpp:577` / `:1381`, behind a shape gate. A shape-gate threshold
   still needs measuring — doc 14's kv=512 P·V margin is already down to 2.5×,
   so there is a kv_len below which CPU wins.
7. **Re-verify doc 14's numbers with FastRPC included.** Until then, none of
   13's or 14's figures are production predictions.

Still open, unrelated to the above ordering:

- **u8i4/u8i8 attention accuracy is entirely unmeasured.** The fp16
  accumulation risk was retired ([14](14_mha_fp16_micro_verified.md) §4), but
  quantising Q/K/V/P to 4 or 8 bits is a different failure mode and has never
  been tested. Note that the HMX u8i4/u8i8 ops fix **activation as `uint8`**
  and only the *weight* slot as i4/i8, so the real question is per-operand:
  Q·Kᵀ's weight slot (`Qᵀ`) and P·V's weight slot (`V`) independently.
  Evidence worth weighing: QNN's shipping Gauss deployment quantises its KV
  cache to **int8, not int4** (verified by reading
  `~/workspace/Quick.AI`'s `gauss*_qnn.cpp` runtime dtype assertions).
- **The KV cache is hardcoded fp16/uint16** (`mha_core.cpp:233-245`) and does
  not follow `fc_layer_dtype`. Any u8i4 attention path needs a KV cache
  quantisation scheme designed from scratch — including how per-entry scales
  are maintained as the cache grows, which the static-weight FC recipe does
  not answer.
- **`hexkl_micro_fc_bench.c`'s SDKL-macro correctness check is unreliable** —
  flips OK/WRONG for the same shape across reruns, from a missing
  `qurt_mem_cache_clean()` around `hexkl_macro_mm_*` (the header requires
  FLUSH on inputs, FLUSH_INVALIDATE on output). Timing numbers unaffected.
- **The softmax `[gqa, kv]` layout ask** ([14](14_mha_fp16_micro_verified.md)
  §6) — optional, worth ~7 µs/head, not blocking.
