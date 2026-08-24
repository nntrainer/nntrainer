# NPU Prefill Debugging Session — Complete Log, State, and Roadmap

**Date:** 2026-08-18
**Device used for all on-device verification:** Samsung S24 (Snapdragon 8 Gen 3, Hexagon HTP v79), serial `R3CX9078DNH`.
**Purpose of this doc:** everything done this session, every bug found and fixed, every
benchmark number actually measured (with exact env vars), and a concrete roadmap for
whoever continues this. Treat every number/claim in `docs/backend_guide/` dated before
today with suspicion — most of it was measured against stale binaries or incomplete env
var combinations (see "Why every earlier benchmark doc is unreliable" below).

---

## 0. TL;DR for the next agent

- **GEMM-on-NPU never worked before today** — `libnntrainer.so` was built with
  `enable-hexagon-cdsp=false`. Nobody's "NPU vs CPU" comparison this whole project ever
  actually had a GEMM running on HMX. Fixed today: rebuild with
  `-Denable-hexagon-cdsp=true -Denable-fp16=true` (exact recipe in §4).
- **Two separate silent "stale build" bugs** meant most on-device tests this project ran
  against old binaries regardless of source changes: ndk-build's install step, and the
  Hexagon `ExternalProject`'s install step. Both bypassed with manual `cp` from the
  `obj/`/sub-build output to the final `libs/`/install location — see §4.
- **Four real, previously-unknown correctness bugs found and fixed** (mutex self-deadlock,
  KV-cache stale-read, a diagnostic-function self-deadlock, and a deferred-copy-onto-
  freed-memory crash) — all four were blocking the batch-mode/full-pipeline combination
  from ever running at all. Full pipeline (GEMM + flash_attn + fused_ffn + RMSNorm/RoPE/
  ADD + batching) now runs correctly and reproducibly. See §2.
- **Verified, reproducible result**: flash_attn + fused_ffn + GEMM (all pre-existing,
  before this project started) alone gets **~2.64× over CPU** (576ms vs 1520ms prefill,
  650 tokens). Adding this session's new work (RMSNorm/RoPE/ADD-on-NPU + batching
  infrastructure) lands at **~2.60×** (582-587ms) — i.e. **no measurable net improvement**
  over the pre-existing baseline, confirmed via real round-trip counts (§3.3): legacy-only
  needs 112 real FastRPC round trips, everything needs 225, and the extra 113 cost only
  ~12ms total (~0.1ms each) — small added cost roughly cancels small saved CPU time for
  tiny ops. See §3 for full numbers and §5 for the "why" and the concrete next steps.
- **The single highest-leverage next step is not more op-porting — it's making the
  `LayerNode` sync guard layer-aware instead of universal.** See §5.2.

---

## 1. Why every benchmark doc dated before today should not be trusted

Two independent, silent build/deploy bugs mean most historical numbers in this
`docs/backend_guide/` directory were measured against code that didn't match what was
actually in the source tree at the time:

1. **`ndk-build`'s install step silently no-op'd.** `libcausallm_core.so`,
   `libcausallm_api.so`, `libccapi-nntrainer.so`, `libnntrainer.so`, and
   `nntrainer_causallm` in `Applications/CausalLM/jni/libs/arm64-v8a/` would report
   "Install: X => libs/arm64-v8a/X" in the build log while leaving the *actual* file on
   disk untouched — verified by direct `md5sum`/`cmp` against the freshly-linked copy in
   `obj/local/arm64-v8a/`. The workaround (§4.2) is to always `cmp`/force-`cp` from
   `obj/local/arm64-v8a/<name>` to `jni/libs/arm64-v8a/<name>` after every build, never
   trust the ndk-build log alone.
2. **The Hexagon DSP skel's `ExternalProject_Add` install step has the identical bug.**
   `ggml-hexagon/build-hexagon-android/ggml/src/ggml-hexagon/libggml-htp-v79.so` (the
   "installed" copy `ninja` considers up to date) can silently lag behind
   `.../htp-v79-prefix/src/htp-v79-build/libggml-htp-v79.so` (the actual freshly-built
   sub-project output). Same manual-`cp` workaround.

Additionally: **`NNTR_HEXAGON_DISABLE` does not exist anywhere in the source code.**
Every "CPU-only" comparison row in `bench_sweep.sh`, `bench_sweep2.sh`, and
`QWEN3_NPU_PREFILL_RESULTS_2026-08-18.md` that sets this variable was not actually
CPU-only — it did nothing, and RMSNorm/RoPE/ADD's own gate
(`NNTR_HEXAGON_NO_ELEM_OPS`, opt-*out*, on by default) still fired regardless. The real,
verified CPU-only baseline measured this session (§3.1) required
`NNTR_HEXAGON_NO_ELEM_OPS=1 NNTR_HEXAGON_NO_BATCH=1` with `NNTR_USE_HEXAGON_CDSP` and
`NNTR_HEXAGON_FLASH_ATTN`/`NNTR_HEXAGON_FUSED_FFN` all left **unset** — confirmed by the
complete absence of any `ggml-hex: Loading driver...` line in that run's log.

---

## 2. Bugs found and fixed this session (chronological)

All fixes are in the working tree now (uncommitted — see §4 for exact file list). All
are in `/home/anirudh/ggml-hexagon` (the bridge/DSP repo) unless stated otherwise.

### 2.1 Mutex self-deadlock (12 sites, `nntr-htp-bridge.cpp`)
Every per-op bridge function (`gemm_q4_0`, `gemm_q4_0_batch`, `flash_attn`,
`ffn_swiglu`, `sgemm_fp32`, `sgemm_batch_fp32`, `fused_fc_forward`, `rms_norm`, `rope`,
`add`, `register_activation_pool`, `upload_weight_q4x4x2`) did an unconditional
`std::lock_guard<std::mutex> lock(state.mtx)` at entry. `begin_batch()` holds that same
non-recursive mutex raw for the whole batch scope. Any op call made *inside* a
begin_batch/end_batch scope would self-deadlock. **Fix**: `std::unique_lock<std::mutex>
lock(state.mtx, std::defer_lock); if (!state.batch_mode) lock.lock();` at all 12 sites.

### 2.2 KV-cache stale-read (`Applications/CausalLM/layers/mha_core.cpp`, nntrainer repo)
`rope_dsp(...)` only *enqueues* the rotation when a batch is open — it does not wait for
the DSP to execute it. The very next line copied `key_step`'s raw memory into the KV
cache synchronously, reading pre-rotation/stale data. Same hazard existed for the
V-cache append, the Q-RoPE CPU-fallback path, and the CPU attention fallback path in the
`2 ≤ step_size < 160` gap (where RoPE-DSP fires but flash_attn's threshold hasn't been
met yet). **Fix**: a `flush_if_batch_active()` helper (backed by a new
`nntr_htp_bridge_is_batch_active()`/`nntr_htp_bridge_flush()` combo, later replaced by
the single `nntr_htp_bridge_flush_if_batch_active()` — see §2.5) called at five points in
`mha_core.cpp` immediately before each of these reads.

### 2.3 `dump_pool_stats()` self-deadlock
Same class of bug as §2.1, introduced *after* §2.1's fix while adding the zero-copy
diagnostic — `nntr_htp_bridge_dump_pool_stats()` used a fresh unconditional
`std::lock_guard`. Manifested as a hang immediately after the last transformer layer
finished (right where `causal_lm.cpp` calls `dump_pool_stats()` before `end_batch()`).
Fixed with the same conditional-lock pattern.

### 2.4 Deferred-copy-onto-freed-memory crash (the big one — SIGSEGV, 8 call sites)
Root cause, found via crash backtrace + a targeted `pending_copies` pointer dump:
non-zero-copy op outputs (i.e. outputs not living in a registered rpcmem pool) were
staged into a shared, reused staging buffer, with the copy-back to the caller's real
destination **deferred** via a `pending_copies` list when batch mode was active. Two
independent ways this corrupts memory:
- The staging region itself is grown/reset (`nntr_htp_bridge_get_staging()`) by a
  *later* op needing more space, freeing memory an earlier op's still-undrained
  `pending_copies` entry pointed into.
- **The actual crash trigger**: the *destination* tensor (not the staging source) can be
  reused/freed by nntrainer's own tensor memory pool before the deferred copy ever runs
  — this is the mirror image of §2.2 (stale read); here it's a stale *write target*.
  Confirmed via crash backtrace: `memmove` inside `nntr_htp_bridge_flush_if_batch_active`,
  called from a *downstream* layer's `LayerNode::incremental_forwarding`, writing to an
  address (`0xb400...`-range) that was already invalid by the time the copy ran. Exact
  reproduction: `flash_attn`'s FP16 output (not pool-backed in this run) staged and
  deferred; by the time a later layer's guard call tried to drain it, the destination was
  gone.

**Fix (two parts):**
1. `nntr_htp_bridge_get_staging()` now drains (`flush + copy back`) any outstanding
   `pending_copies` *before* handing out memory in either the "reuse same region" path
   or the "grow/reset" path — a new `nntr_htp_bridge_drain_staging_users()` helper.
2. **The real fix**: removed the entire `pending_copies`-deferral mechanism. Every bridge
   function now flushes immediately and copies back immediately whenever its output
   isn't zero-copy, regardless of batch mode — only genuinely zero-copy outputs (DSP
   writes directly into the caller's final tensor, nothing to copy) are safe to defer,
   and those already require no action either way. Applied to all 8 call sites that used
   to push `pending_copies` entries: `gemm_q4_0`, `gemm_q4_0_batch`, `flash_attn`,
   `ffn_swiglu`, `sgemm_fp32`, `rms_norm`, `rope`, `add` (plus the unused
   `nntr_htp_bridge_finish` helper, fixed for consistency though it has no callers).

### 2.5 Diagnostic infrastructure added (kept in the tree, useful for future work)
- `nntr_htp_bridge_flush_if_batch_active()` — the systemic per-layer sync-guard's
  backing function, wired into `nntrainer/layers/layer_node.cpp`'s
  `LayerNode::forwarding()`/`incremental_forwarding()` (this specific wiring predates
  this session — someone implemented it from an earlier recommendation; this session
  found and fixed the bugs in what it calls).
- `nntr_htp_bridge_get_flush_count()`/`reset_flush_count()` — counts every *call* to a
  flush path, including cheap no-ops where nothing was queued. **Misleading on its own**
  — see §2.6.
- `nntr_htp_bridge_get_real_flush_count()` — added today specifically to fix the
  misleading metric above; only increments when `op_batch` was actually non-empty at
  flush time, i.e. counts genuine FastRPC round trips. **Use this one, not
  `get_flush_count()`, for any future performance reasoning.**
- `nntr_htp_bridge_dump_pool_stats()` — logs zero-copy pool hit/miss counts per
  call-site label (`gemm_q4_0:act`, `rms_norm:in`, `rms_norm:gamma`, `add:a`, etc.).
  Wired into `causal_lm.cpp` to fire unconditionally (works in every env-var
  configuration, not just batch mode).
- `[LAYER_FLUSH]` trace in `layer_node.cpp` — logs which layer name triggered each real
  flush, with per-layer call counts. This is what proved the "hang" in §2.3 was not an
  infinite loop (it was making clean, non-repeating progress through all 28 layers, then
  crashing/hanging on the very next thing after the last layer).

### 2.6 The flush-count metric was itself misleading — found and fixed today
`nntr_htp_bridge_get_flush_count()` increments on every call to
`flush_if_batch_active()`, including ones where `flush_batch()` internally no-ops
because nothing was enqueued since the last real flush. Since the `LayerNode` guard
fires before *every* layer regardless of whether anything needs syncing, this metric is
dominated by the fixed number of layers in the graph (~509 for Qwen3-0.6B's ~18
layer-nodes/block × 28 blocks), **not** by how many ops actually dispatch to NPU — it
was identical (509) whether or not RMSNorm/RoPE/ADD were even active. Added
`real_flush_count` (§2.5) to separate "guard fired" from "guard's flush actually talked
to the DSP" — this is what made the final root-cause analysis in §5 possible.

---

## 3. Verified benchmark data (all runs today, on `R3CX9078DNH`, 650-token prompt)

Every row below is directly reproduced from an actual `adb shell` run this session, not
inferred. Exact env vars given so any row can be reproduced exactly.

### 3.1 True CPU-only baseline
```
NNTR_NUM_THREADS=4
(no NNTR_USE_HEXAGON_CDSP, no NNTR_HEXAGON_FLASH_ATTN, no NNTR_HEXAGON_FUSED_FFN)
NNTR_HEXAGON_NO_ELEM_OPS=1
NNTR_HEXAGON_NO_BATCH=1
```
Confirmed zero `ggml-hex:` driver-load lines in the log (genuinely never touches the DSP).
**Result: prefill 1520ms, 427.6 TPS, peak memory 728MB.**

### 3.2 Legacy-only (everything that existed *before* this session's work)
```
NNTR_USE_HEXAGON_CDSP=1
NNTR_HEXAGON_FLASH_ATTN=1
NNTR_HEXAGON_FUSED_FFN=1
NNTR_HEXAGON_NO_ELEM_OPS=1   # RMSNorm/RoPE/ADD stay on CPU
NNTR_HEXAGON_NO_BATCH=1       # (also tested with batching on, see 3.3)
```
**Result: prefill 575-580ms, ~1120-1130 TPS. Speedup: ~2.64× over CPU.**
This is QKV-batching (pre-existing `qkv_layer`) + flash_attn + fused_ffn + Q4_0 GEMM —
all built before this session started. **This is the number to beat.**

### 3.3 Everything (legacy + this session's RMSNorm/RoPE/ADD-on-NPU + batching)
```
NNTR_USE_HEXAGON_CDSP=1
NNTR_HEXAGON_FLASH_ATTN=1
NNTR_HEXAGON_FUSED_FFN=1
(no NO_ELEM_OPS, no NO_BATCH - everything on, batching on)
```
**Result: prefill 582-587ms, ~1107-1117 TPS. Speedup: ~2.60× over CPU.**
Reproducible across 3 separate runs after all bugs in §2 were fixed. **Statistically
indistinguishable from §3.2** (within ~1-2% noise), despite doing strictly more work.

### 3.4 Real FastRPC round-trip counts (the decisive diagnostic, using §2.6's fix)
| Config | Real round trips | `get_flush_count()` (misleading) | Prefill |
|---|---|---|---|
| Legacy-only + batching on | **112** | 509 | 575-580ms |
| Everything + batching on | **225** | 509 | 582-587ms |

112 = exactly 4 real dispatches/block × 28 blocks (QKV-batch, O-proj, flash_attn,
fused_ffn) — matches hand-derivation exactly. The extra 113 round trips from adding
RMSNorm/RoPE/ADD cost only ~12ms total wall-clock (~0.1ms/round-trip average) — far
cheaper than the ~1.6-3ms/flush figure quoted in older docs, which is why the net effect
is a wash rather than a clear regression: small added round-trip cost roughly cancels
small saved CPU compute time for ops this cheap. See §5 for the full explanation.

### 3.5 Isolated tests along the way (for anyone re-deriving the above)
- GEMM-only, no flash_attn, no fused_ffn, no elem-ops, no batch:
  **983ms, 661.2 TPS** (confirms GEMM-on-NPU alone works and gives real speedup once
  `enable-hexagon-cdsp` is actually on).
- flash_attn alone (+ GEMM), no fused_ffn, no elem-ops, no batch:
  **623ms, 1043.3 TPS, 2.44×** — attention was always the single biggest lever; the
  original week-old "1.5×" narrative almost certainly never had this active either (was
  probably measuring something narrower).
- `NNTR_USE_HEXAGON_CDSP=1` alone, without a `libnntrainer.so` built with
  `-Denable-hexagon-cdsp=true`: **fatal error, "[Engine] cdsp Context is not
  registered"** — this confirms the flag gap in §0/§4 concretely; it is not a soft
  runtime toggle, the symbol must be compiled in.

---

## 4. Exact build/deploy recipe (needed because of §1's two silent-staleness bugs)

### 4.1 Rebuild the ARM-side Hexagon bridge (`libggml-hexagon.so`)
```bash
cd /home/anirudh/ggml-hexagon
ninja -C build-hexagon-android libggml-hexagon.so
# Verify it's actually fresh (this target does NOT have the staleness bug,
# ninja builds straight into bin/ - but always double check with strings/md5sum
# after any edit, trust nothing in this codebase by default):
strings build-hexagon-android/bin/libggml-hexagon.so | grep -c "<a string unique to your edit>"
```
This target already builds directly into `build-hexagon-android/bin/`, no separate
install step — it does not have the §1 staleness bug itself. (The DSP-*skel*
`libggml-htp-v79.so`, a different, separate build via `ExternalProject_Add`, DOES have
it — see below, only relevant if you touch `htp/*.c`, which this session did not.)

### 4.2 Rebuild nntrainer core with the flags that actually matter
```bash
cd /home/anirudh/nntrainer
rm -rf build-cdsp
mkdir -p build-cdsp/ml-api-inference
cp -r builddir/ml-api-inference/include build-cdsp/ml-api-inference/
cp -r builddir/ml-api-inference/lib build-cdsp/ml-api-inference/
# (pre-seeding ml-api-inference avoids a network fetch to an S3 bucket that
# 404s in this environment - prepare_ml-api.sh skips downloading if
# ${TARGET}/include already exists)

meson setup build-cdsp --cross-file android-aarch64.ini \
  -Denable-transformer=true -Dplatform=android \
  -Denable-tflite-interpreter=false -Denable-tflite-backbone=false \
  -Denable-test=false \
  -Denable-hexagon-cdsp=true \
  -Denable-fp16=true
  # BOTH flags are required together. enable-hexagon-cdsp alone links but the
  # resulting .so is missing every _FP16/half-typed symbol CausalLM's layers
  # need (mha_core.cpp, swiglu.cpp, rms_norm.cpp all use FP16 codepaths),
  # causing "cannot locate symbol ... referenced by libcausallm_core.so" at
  # runtime load. enable-fp16 defaults to FALSE in meson_options.txt - it is
  # not implied by platform=android or by the cross-file's -march=...+fp16.

cd build-cdsp/jni
ANDROID_NDK=/opt/android-ndk-r26d /opt/android-ndk-r26d/ndk-build -j$(nproc) nntrainer ccapi-nntrainer
# Building the "nntrainer"/"ccapi-nntrainer" module targets by name avoids
# ndk-build also trying (and failing on unrelated, pre-existing issues) to
# build every other module in this Android.mk.

# THE STALENESS BUG: verify and force-copy before trusting the output.
cd /home/anirudh/nntrainer
cmp build-cdsp/obj/local/arm64-v8a/libnntrainer.so \
    build-cdsp/libs/arm64-v8a/libnntrainer.so && echo SAME || echo DIFFERENT
# If DIFFERENT (it has been, every single time this session), the obj/ copy
# is the correct one - copy it into the location Applications/CausalLM's
# Android.mk expects as a "Prebuilt":
cp build-cdsp/obj/local/arm64-v8a/libnntrainer.so \
   builddir/android_build_result/lib/arm64-v8a/libnntrainer.so
```

### 4.3 Rebuild CausalLM against the fresh prebuilt
```bash
cd /home/anirudh/nntrainer/Applications/CausalLM/jni
ANDROID_NDK=/opt/android-ndk-r26d /opt/android-ndk-r26d/ndk-build -j$(nproc) \
  causallm_core causallm_api ccapi-nntrainer nntrainer_causallm nntrainer
# (naming modules explicitly again avoids the pre-existing unittest targets,
# which fail on a missing gtest/gtest.h include unrelated to anything here)

# AGAIN check for staleness before trusting anything:
cd /home/anirudh/nntrainer/Applications/CausalLM
for f in libcausallm_core.so libcausallm_api.so libccapi-nntrainer.so libnntrainer.so nntrainer_causallm; do
  cmp -s "obj/local/arm64-v8a/$f" "jni/libs/arm64-v8a/$f" && echo "$f: same" || cp -v "obj/local/arm64-v8a/$f" "jni/libs/arm64-v8a/$f"
done
```

### 4.4 Push and run
```bash
DEV=R3CX9078DNH
REMOTE=/data/local/tmp/nntrainer/causallm
adb -s $DEV push /home/anirudh/ggml-hexagon/build-hexagon-android/bin/libggml-hexagon.so $REMOTE/
adb -s $DEV push Applications/CausalLM/jni/libs/arm64-v8a/{libcausallm_core.so,libcausallm_api.so,libccapi-nntrainer.so,libnntrainer.so,nntrainer_causallm} $REMOTE/
adb -s $DEV shell "chmod 755 $REMOTE/*"

adb -s $DEV shell "cd $REMOTE && \
  export LD_LIBRARY_PATH=$REMOTE:\$LD_LIBRARY_PATH NNTR_NUM_THREADS=4 \
         NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_FLASH_ATTN=1 NNTR_HEXAGON_FUSED_FFN=1 && \
  timeout 150 ./nntrainer_causallm models/qwen3-0.6b > run.log 2>&1; echo EXIT_\$?"
adb -s $DEV shell "cat $REMOTE/run.log" | grep -E "^prefill|^generation|^total|^peak|flush|pool_stats"
```

### 4.5 If you ever touch `ggml-hexagon/ggml/src/ggml-hexagon/htp/*.c` (DSP-side kernels)
This session did not, but if a future one does, the DSP skel also has the §1 staleness
bug:
```bash
cd /home/anirudh/ggml-hexagon/build-hexagon-android/ggml/src/ggml-hexagon/htp-v79-prefix/src/htp-v79-build
ninja   # rebuilds the sub-project directly, bypassing the stale top-level "install" step
cp libggml-htp-v79.so ../../../libggml-htp-v79.so   # force the "installed" copy fresh
# then push that path's libggml-htp-v79.so to the device
```

### 4.6 Environment reference table
| Variable | Effect | Default |
|---|---|---|
| `NNTR_USE_HEXAGON_CDSP` | opt-in, enables `engine=cdsp` on FC/QKV/gate-up layers (GEMM-on-NPU) | off |
| `NNTR_HEXAGON_FLASH_ATTN` | opt-in, enables flash_attn bridge dispatch in `mha_core.cpp` | off |
| `NNTR_HEXAGON_FUSED_FFN` | opt-in, selects `fused_ffn` layer over separate gate_up+swiglu+down at graph-construction time | off |
| `NNTR_HEXAGON_NO_ELEM_OPS` | opt-**out**, disables RMSNorm/RoPE/ADD NPU dispatch | **on by default** (inconsistent with the others - worth fixing for consistency) |
| `NNTR_HEXAGON_NO_BATCH` | opt-out, disables `begin_batch`/`end_batch` wrapping | on by default (batching is default-on) |
| `NNTR_HEXAGON_DISABLE` | **does not exist in the code at all** | n/a |
| `NNTR_HEXAGON_FLASH_ATTN_MIN_TOKENS` | overrides the 160-token flash_attn threshold | 160 |
| `NNTR_HEXAGON_FLASH_ATTN_VERBOSE` | verbose gate-decision logging | off |

---

## 5. Why the new work (RMSNorm/RoPE/ADD-on-NPU + batching) adds nothing measurable

Full reasoning, grounded in §3.4's real round-trip counts:

### 5.1 The round-trip count is fixed by graph structure, not by which ops target NPU
The `LayerNode` guard (`layer_node.cpp:nntr_hexagon_flush_if_batch_active`) fires before
*every* layer unconditionally — `get_flush_count()` was 509 in both the legacy-only and
everything configurations, exactly matching Qwen3's fixed ~18-layer-nodes-per-block × 28
structure, regardless of whether RMSNorm/RoPE/ADD dispatch to NPU or stay on CPU.

### 5.2 The guard doesn't distinguish "needs a real sync" from "could safely chain"
This is the actual, root, fixable reason. A GEMM whose input is simply a pointer to the
*previous* NPU op's output (pure DSP-to-DSP handoff, correctness guaranteed by the DSP's
own FIFO execution order — no CPU visibility needed at all) still gets a flush forced
before it, purely because the guard is universal, not because that specific layer needs
one. This is why legacy-only *already* needs 112 real round trips (4/block) even though,
in principle, several of those 4 could theoretically chain into fewer real round trips if
the guard only fired where genuinely necessary (see §6.1 for the concrete fix).

### 5.3 The economics: tiny ops can't pay for even a small round trip
RMSNorm/RoPE/ADD run on HVX (vector unit), not HMX (systolic array) — confirmed via
direct kernel dispatch code reads (`hvx_fast_rms_norm_mul_f32`, `hvx_rope_neox_f32_aa`,
`hvx_add_f32_aaa`, all in `ggml-hexagon/ggml/src/ggml-hexagon/htp/{unary,rope,binary}-ops.c`).
HVX has no compute-density advantage over ARM NEON for elementwise work, and pays real
DMA/VTCM staging costs a CPU loop doesn't. Measured: the extra 113 real round trips this
session's work adds cost only ~12ms total (~0.1ms each) — genuinely cheap, much cheaper
than the ~1.6-3ms/flush figure from earlier (2026-08-13-dated) profiling docs, which
suggests the batching/pinning infrastructure built this session is working as intended.
But there was correspondingly little CPU compute time to save in the first place (these
are O(hidden_dim) loops, not O(hidden_dim × intermediate_dim) matmuls) — so the two small
numbers (added round-trip cost, saved CPU time) land within noise of each other.

**This is not a failure of implementation quality. It's the correct economic outcome:
NPU offload pays off for compute-heavy ops (matmuls, attention) and doesn't for
memory-bound elementwise ops, no matter how well you batch them**, unless the round-trip
cost can be driven to genuinely zero (see §6).

---

## 6. Concrete roadmap to "the whole transformer block runs on NPU"

Ranked by leverage/effort. None of this was implemented this session — it's the
handoff plan.

### 6.1 Make the sync guard layer-aware (highest leverage, do this first)
Currently: `LayerNode::forwarding()`/`incremental_forwarding()`
(`nntrainer/layers/layer_node.cpp`) calls `nntr_hexagon_flush_if_batch_active()`
unconditionally before every layer. Change to: only call it before a layer that is
*not* itself going to enqueue-and-defer (i.e., a layer whose own `forwarding()` will
touch tensor data on the CPU directly — the CPU-fallback branches, any layer not yet
NPU-dispatched, and the final logits read). This requires either (a) a per-layer
property/flag set at graph-construction time indicating "this layer's forward touches
CPU memory," checked by the guard, or (b) inverting the model so *layers themselves*
call a "I'm about to read real data" sync point immediately before they do so (this is
what `mha_core.cpp`'s five explicit `flush_if_batch_active()` calls from §2.2 already do
manually) and the blanket `LayerNode`-level guard is removed/weakened to a defensive
fallback rather than the primary mechanism. Option (b) is more surgical and lower-risk
(no need to correctly classify every layer type up front) but requires auditing every
custom layer for CPU reads, same class of audit as §2.2/§2.4 found bugs in.

### 6.2 Port the KV-cache append onto the DSP
Currently a CPU `Tensor::copyData()` in `mha_core.cpp`. Either have the RoPE kernel
write its rotated output directly into the cache's rpcmem-backed slot (eliminating the
separate copy entirely), or add a dedicated DSP-side copy op (`HTP_OP_CPY` already
exists in `ggml-hexagon`'s op catalog, per `htp/htp-ops.h`).

### 6.3 Port causal mask generation, or use an implicit-causal flash_attn mode
`build_causal_mask()` in `mha_core.cpp` is a CPU loop. Check whether
`hmx_flash_attn_ext` (`ggml-hexagon/ggml/src/ggml-hexagon/htp/hmx-flash-attn-ops.c`)
supports computing the causal mask internally from a flag rather than requiring an
explicit precomputed mask array — if so, this requirement disappears rather than needing
porting.

### 6.4 Route the weight pool through rpcmem
`nntrainer/models/neuralnet.cpp`'s `setComputeBackend("", "cdsp")` only routes the
*activation* tensor pool to `HexagonRpcAllocator`, by explicit design ("weights stay on
CPU... not DSP-registered... routing weights to rpcmem too would needlessly exhaust the
scarce CMA pool"). This is a real constraint (rpcmem/CMA is limited), but for a QLoRA/
small-model context the actual weight volume needing rpcmem (LoRA adapters, RMSNorm
gammas) is small — worth reassessing whether the original constraint still holds for
this use case.

### 6.5 Port embedding lookup
`HTP_OP_GET_ROWS` exists and requires an F32 table (Qwen3-0.6B's tied embedding *is*
F32) and I32/I64 indices (nntrainer currently stores token IDs as FP32-encoded floats,
needs a cast). Previously deprioritized as low-value (cheap gather, not worth a round
trip on its own) — reconsider only in the context of "zero CPU touch" as a goal in
itself, not as a standalone perf win.

### 6.6 What remains even after 6.1-6.5 (cannot be eliminated, and that's fine)
- The C++ interpreter loop driving `NetworkGraph` — CPU instruction execution, not data
  touching. Compatible with "the whole block's math runs on NPU."
- Reading the final logits to sample the next token, and tokenization — inherently
  host-side, tiny, irreducible.
- One-time session/weight setup — amortized, irrelevant to steady-state throughput.

With 6.1-6.5 done, the *only* remaining CPU-visible reads for a full prefill should be
the final logits — i.e., genuinely "one flush for the whole prefill," matching
ggml-hexagon's own native inference model exactly.

---

## 7. Known gaps / unverified items for whoever continues this

1. **Correctness beyond "same first token."** No full-sequence logit/perplexity
   comparison between CPU and NPU paths has been done this session — only that the first
   generated token (`&`) matched across configs in the original (pre-session)
   `QWEN3_NPU_PREFILL_RESULTS_2026-08-18.md` doc, before any of today's bug fixes.
   Re-verify this once §6 work lands.
2. **The `mha_core.cpp`-style stale-read/stale-write hazard (§2.2/§2.4) was only audited
   in `mha_core.cpp` and the 8 bridge functions.** `reshaped_rms_norm.cpp` (q_norm/k_norm)
   sits directly downstream of Q/K projection GEMMs and has never been audited for the
   same class of bug — it's currently CPU-only (not NPU-dispatched) so may not be exposed
   yet, but will need auditing the moment it's ported.
3. **`NNTR_HEXAGON_NO_ELEM_OPS`'s default (on) is inconsistent** with every other gate in
   this codebase (`NNTR_USE_HEXAGON_CDSP`, `NNTR_HEXAGON_FLASH_ATTN`,
   `NNTR_HEXAGON_FUSED_FFN` are all opt-*in*, default off). Worth fixing for consistency
   and to avoid the exact "silently active without realizing it" confusion this session
   hit repeatedly.
4. **Decode (single-token generation) was not touched or optimized this session** — all
   work and all numbers in this doc are prefill-only, by design (established very early
   this project that NPU loses for M=1 GEMV-scale decode; this remains correct and
   untouched).
5. **The training path (nntrainer-lora repo) was not touched at all this session** — all
   work here is in the inference-oriented `nntrainer` + `ggml-hexagon` repos. Everything
   in this document needs re-verification/re-porting if applied to the LoRA training
   codebase, which has its own separate, CPU-only compute path (see the much earlier
   `LORA_NPU_FEASIBILITY_STUDY.md`, itself written before any of this session's findings
   and likely needing revision in light of them).

---

## 8. File-level change list (uncommitted, in the working tree now)

### `/home/anirudh/ggml-hexagon` (bridge + DSP-side host code)
- `ggml/src/ggml-hexagon/nntr-htp-bridge.cpp` — all of §2.1, §2.3, §2.4, §2.5, §2.6.
  This is where almost all of this session's changes live.
- `ggml/src/ggml-hexagon/ggml-hexagon.cpp` — added tracing prints inside
  `ggml_hexagon_session::flush_pending()`/`flush_batch()` (iteration counters, op_pending
  before/after logging) used to diagnose §2.4. Safe to leave in (loud `GGML_LOG_ERROR`
  calls, low-frequency) or strip if considered too noisy for production.

### `/home/anirudh/nntrainer` (nntrainer core + CausalLM app)
- `Applications/CausalLM/layers/mha_core.cpp` — §2.2's five `flush_if_batch_active()`
  call sites, plus the `is_prefill` gating fix for RoPE-DSP from an earlier session
  segment (predates today but is load-bearing for correctness).
- `Applications/CausalLM/layers/rms_norm.cpp` — prefill gating fix (predates today).
- `Applications/CausalLM/models/causal_lm.cpp` — `begin_batch`/`end_batch` wiring,
  `[NPU_BATCH]` diagnostics, `dump_pool_stats` wiring, and today's `real_flush_count`
  wiring (§2.6).
- `nntrainer/layers/addition_layer.cpp` — ADD's DSP dispatch moved into
  `incremental_forwarding()` (predates today, was previously dead code in `forwarding()`).
- `nntrainer/layers/layer_node.cpp` — the systemic `LayerNode` sync-guard
  (`nntr_hexagon_flush_if_batch_active`), `[LAYER_FLUSH]` per-layer tracing.
- `meson.build`, `Applications/CausalLM/jni/Android.mk` — minor build-system fixes
  (`-fexceptions -frtti` override, `subdir('Applications')` for android app builds) from
  earlier in this project, unrelated to today's specific bugs but required for the
  build recipe in §4 to work at all.

No files were changed in `/home/anirudh/nntrainer-lora` this session.

Signed-off-by: Anirudh <anirudh1023@gmail.com>
