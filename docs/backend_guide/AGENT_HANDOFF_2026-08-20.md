# Agent handoff — Hexagon/NPU regression investigation, 2026-08-20

Context dump for switching between agents/sessions on this exact thread of
work. Device: `R3CX9078DNH` (Samsung S24 Ultra-class, Snapdragon 8 Elite,
HTP v79). Model: Qwen3-0.6B, Q4_0 FC weights, `model_tensor_type:
"Q4_0-FP32"`. Everything below is against the **uncommitted working tree**
(nothing in this investigation is committed to git).

## What was asked, in order

1. Build the current uncommitted tree for Android, deploy to
   `R3CX9078DNH`, benchmark, document the build process. → done, see
   `BUILD_OBSERVATIONS_2026-08-20.md`.
2. Investigate a measured NPU regression (uncommitted tree was 8-10× slower
   than CPU) and answer: is the whole transformer block on NPU? → root
   cause found and fixed (see below).
3. Follow-up thread (this doc): why does the *fixed* build (141 round-trips)
   not beat the naive no-batching `HEAD` (196 round-trips)? What FP16
   workarounds exist, does ggml-hexagon need the same ones? How many
   flushes now, and why?

## Fixes already applied and deployed (uncommitted, on disk right now)

1. `Applications/CausalLM/models/transformer.cpp`:
   `createKVCachePlaceholders()`'s two `"input"` layers and `createMlp()`'s
   `fused_ffn` layer now wrapped in `withHexagonEngine(...)` (previously
   untagged / deliberately un-tagged).
2. `nntrainer/hexagon/hexagon_context.cpp`: registered `InputLayer` under
   `HexagonContext` (required for fix #1 to not throw "Key is not found").
3. `nntrainer/compiler/multiout_realizer.cpp`: re-enabled engine-propagation
   onto auto-generated multiout nodes, but fixed the check itself —
   original code read `getComputeEngineType()` (populated only by
   `LayerNode::finalize()`, which runs *after* realizers), switched to
   `getProperty("engine")` (populated at construction time, visible to
   realizers).
4. `Applications/meson.build`: gated non-CausalLM example apps out of the
   `platform == 'android'` build (unrelated pre-existing breakage, see
   `BUILD_OBSERVATIONS_2026-08-20.md` §5).
5. `Applications/CausalLM/models/gpt_oss/gpt_oss_moe_layer.h`,
   `Applications/CausalLM/models/lfm2/lfm2_causallm.h`: reverted two
   unrelated syntax bugs (bad find/replace artifacts) back to `HEAD`-
   identical, unrelated to Hexagon work, needed to get any build to
   compile at all.

All currently deployed to the device (`/data/local/tmp/nntrainer/causallm/`)
and reflected in the working tree.

## Why the regression happened (confirmed root cause)

`layer_node.cpp` has a per-layer sync-guard
(`nntr_hexagon_flush_if_batch_active`, itself new/uncommitted): before
running any layer whose `compute_engine != CDSP`, it force-flushes the
pending NPU batch, so a CPU layer never reads stale pre-NPU data. It
correctly skips the flush for `engine=cdsp`-tagged layers. Three layer
types in the per-block loop were untagged or had their tag *deliberately
disabled mid-debugging* (two "TEMP DEBUG" comments citing an unrelated,
undocumented "2x prefill regression" investigation with no written record
anywhere) — KV-cache placeholders, the fused-FFN layer, and the residual
multiout nodes. Fixing all three (fixes #1-3 above) cut the LayerNode-guard
flushes from ~143 to 4 (only genuine graph-boundary ops), and the real
FastRPC round-trip count from 252 to 141, and NPU went from 8-10× slower
than CPU back to 2.1-2.7× faster.

## Open question: why is 141-round-trip "fixed" still slower than 196-round-trip `HEAD`?

- `HEAD` (commit `c7d30ec7`) has **zero** batching machinery at all — no
  `begin_batch`/`end_batch` in `causal_lm.cpp`, no flush-guard in
  `layer_node.cpp`, no `is_batch_active` in `mha_core.cpp`. Confirmed by
  direct `git show HEAD:...` diff — all of it (556 new lines in
  `mha_core.cpp` alone) is uncommitted. `HEAD` does plain per-op
  dispatch: enqueue, flush, wait, repeat. Measured: **196 real round-trips,
  819 ms** at 909 tokens.
- Fixed uncommitted tree: **141 real round-trips, 968-978 ms** at 909
  tokens. Slightly *slower* than `HEAD`'s dumber approach.
- I originally (wrongly) attributed the gap to (a) mandatory CPU-side K-cache
  RoPE rotation with no FP16 DSP kernel, and (b) later, a hard 16-buffer
  cap (`HTP_OP_MAX_BUFS`/`HTP_MAX_MMAPS`) in ggml-hexagon forcing automatic
  sub-flushes. **Both retracted, checked and disproven:**
  - RoPE is *not* CPU-side. `mha_core.cpp` has `try_dsp_fp16_rope()` — a
    3-step DSP cast-chain (F16→F32 cpy, F32 RoPE, F32→F16 cpy), explicitly
    "re-enabled" per its own comment. Log evidence: `pool_stats
    rope:inout: 56 hit(s), 0 miss(es)`, zero "falling back to CPU" lines
    anywhere for RoPE.
  - **Tested directly, both sides.** All measurements up through the first
    141-round-trip report were against the **16**-buffer
    `libggml-hexagon.so`. The user then bumped `HTP_OP_MAX_BUFS`/
    `HTP_MAX_MMAPS` to 64 (`htp-ctx.h`/`htp-ops.h`, edited ~18:52-53),
    rebuilt, and pushed the new `.so` to the device (confirmed by mtime,
    ~21:39, after the 141 report). Re-ran the identical 909-token prefill
    against that new 64-buffer `.so`: **still 145 flushes / 141 real
    round-trips, unchanged.** So this is a real, controlled before/after —
    not a source-reading correction — and it conclusively rules out the
    buffer cap as the cause. Raising 16→64 bought nothing.
- **Current true state: cause of the 141-vs-ideal-~1 gap is unconfirmed.**
  Every op-level bridge call (gemm_q4_0, rms_norm, rope, cpy, add) shows
  clean DSP hits with zero CPU fallback in `pool_stats`. The 4
  LayerNode-guard flushes are accounted for. The remaining ~137 are real
  (`op_pending=1` → `flush_pending` → wait, confirmed in raw log), but I
  have not yet traced *why* they're not deferred by the open
  `begin_batch()`/`end_batch()` scope.
- Attempted to get an exact per-phase breakdown via
  `NNTR_HTP_BRIDGE_PROF=1` (a real, existing hook in
  `nntr-htp-bridge.cpp:425-452` — reports weight/stage/desc/flush/out µs
  per op). **Ran it once, the `"prof over N ops"` report line never
  printed** — the hook likely only fires on a clean-shutdown/specific
  code path this app doesn't reach during a single-shot prefill run. Not
  yet root-caused; deprioritized per explicit user instruction to keep
  token spend minimal.

**Next concrete step, if resumed:** find why `report()` in
`nntr_htp_bridge_prof` never fires (grep `nntr-htp-bridge.cpp` for where
`.report()` is called — likely gated on a counter/exit path), fix or force
it, and get the actual weight/stage/desc/flush/out µs split. That's the
fastest path to a confirmed cause for the 137 flushes.

## RESOLVED: root cause of the 137 (FP16) and 25-29 (FP32) residual round-trips

Two independent, now-fixed causes — the "next concrete step" above (the
profiling hook) was never needed; both were found by cross-referencing
`nntr_htp_bridge_find_ext_pool` (the actual hit/miss check) against
`pool_stats`, not by profiling.

### Fix A: FP16 — Q/K/V/O staging tensors reallocated every layer

`mha_core.cpp:849` block created a **brand-new** `nntrainer::Tensor(dim,
true)` for `Q_step`/`K_step`/`V_step`/`O_step` on every one of the 28
layers, every prefill. `nntr_htp_bridge_cpy`'s hit/miss check
(`nntr_htp_bridge_find_ext_pool`) is **not** "have I seen this pointer
before" — it only checks the bridge's own registered-pool table. A fresh,
never-registered allocation misses on every single touch, permanently,
even if you later reuse the same address (confirmed by instrumenting:
reused the same pointer across all 28 layers and *still* got 112 misses,
because the pointer was never registered).

**Fix, in `mha_core.cpp`:** added `get_reusable_fp16_scratch()` — a
function-static scratch `Tensor` per role (Q/K/V/O), shared across all 28
`MHACoreLayer` instances (both `forwarding()` and `incremental_forwarding()`
have their own static set), grown-once, and — the actually necessary part —
**registered once via `nntr_htp_bridge_register_activation_pool`** the same
way `RopeScratchRpcMem` already does for its own scratch buffer. Result:
`cpy:dst/src` pool stats went from `140 hit/112 miss` to `56 hit/0 miss`.
**141 round-trips → 1.** Prefill 960ms → 446ms at 909 tokens. FP16 NPU now
essentially matches FP32 NPU.

### Fix B: FP32 — KV-cache stored as `UINT16`, invisible to the dtype gate

`try_dsp_cache_copy`'s dtype gate (`to_flag()`) only recognizes `FP32`
(→0) and, if `ENABLE_FP16`, `FP16` (→1) — anything else returns -1 and the
function bails to the CPU fallback. The non-`ENABLE_FP16` build's KV-cache
was `UINT16` (`transformer.cpp`'s `createKVCachePlaceholders()` and
`causal_lm.cpp`'s `allocateAndBindKVCache()`, both had the same
`#ifdef ENABLE_FP16 ... FP16 ... #else ... UINT16` pattern) — `UINT16`
never matches either case, so **every** K/V-cache append fell through to
`flush_if_batch_active()` + CPU rotate-and-write, once per layer. That's
the 28-29 round-trips measured on the FP32 build.

**Fix:** changed both sites' `#else` branch from `UINT16` to `FP32` (must
match between the two - one declares the placeholder layer's dtype, the
other allocates and binds the real backing buffer). `FP32` matches what
the GEMM output already is on this build, so the *existing* FP32 case in
`to_flag()`/`try_dsp_cache_copy` just works - zero new DSP code. Verified
safe: `KVCacheManager::allocate()` sizes generically via
`TensorDim::getDataTypeSize()`, no hardcoded byte-width assumption found.
Tradeoff: 2x KV-cache memory (4 bytes/elem vs 2) in the FP32 build only -
acceptable for this model/context length, not evaluated for longer
contexts.

Result: `61 flushes/29 round-trips → 5 flushes/1 round-trip`. **Bonus,
unexpected:** CPU also got 30-40% faster on the FP32 build (820ms vs
1174ms @300tok, 2952ms vs 4897ms @900tok) - the old `UINT16` encoding
wasn't a free reinterpretation, it needed a real quantize/dequantize
conversion in `apply_rotary_emb_tensor_v2`'s CPU path too, which genuine
FP32 doesn't need.

### Final state, both builds fixed, 1 round-trip each

| Tokens | FP32 CPU | FP32 NPU | FP16 CPU | FP16 NPU |
|---|---|---|---|---|
| 392 | 820ms | 233ms | 760ms | 209ms |
| 779 | 2529ms | 366ms | 1962ms | 396ms |
| 909 | 2952ms | 464ms | 2332ms | 446ms |

FP32 NPU and FP16 NPU are now within ~5-8% of each other at every length,
alternating which is ahead - confirms there was never a hidden 2x waiting
behind the round-trip count; once both are near-ideal, the dominant cost
(the Q4_0 GEMM, which is FP32-in/FP32-out on **both** builds — see below)
doesn't care which build you're on.

### Checked: does the FP16 CPU-speedup pattern also apply on NPU? No, and here's precisely why

FC/GEMM layers (`wq`/`wk`/`wv`/`wo`/`gate_up`/`down`/`lm_head`) are FP32
activation in **both** builds - verified directly, no `DataType::FP16`
override anywhere in `transformer.cpp`/`lm_head.cpp`, they all use
`context.getActivationDataType()` which resolves to FP32
(`model_tensor_type: "Q4_0-FP32"`). `ENABLE_FP16` only ever touches: the
Q/K/V/O staging cast, the RoPE cast-chain, flash-attention's internal
K/V/mask (always F16 natively regardless of the flag), and KV-cache
storage. On CPU, FP16 is strictly less work end-to-end (native FP16 ops,
no cast needed anywhere) - a clean 2x-ish win. On the DSP, FP16 requires
*extra* dispatches (the RoPE cast-chain: F16→F32 cpy, F32 rotate, F32→F16
cpy, 3 ops instead of 1) to route around ggml-hexagon's F32-only RoPE
kernel - confirmed the same gap exists in the `zhouwg/ggml-hexagon` fork
too (`kernels/rope-ops.c`: `switch (src[0]->type) { case HTP_TYPE_F32:
...; default: return HTP_STATUS_NO_SUPPORT; }` - no F16 case, different
reimplementation, same limitation). Those extra ops cost real DSP
dispatch time even fully batched (batching removes the FastRPC round-trip
wait, not the DSP's own per-kernel launch+DMA overhead), which is why FP16
NPU doesn't show the same advantage FP16 CPU does - the savings are being
spent paying for the workaround instead of showing up as speedup.

### KV-cache memory savings from FP16, quantified

Only the KV-cache saves real memory (staging buffers are now a single
small reused scratch, same either way). This model
(`num_key_value_heads=8, head_dim=128, num_hidden_layers=28,
max_seq_len=2048` from the bench config):
`2 × 28 × 2048 × 8 × 128 = 117,440,512 elements` → **~224MB (FP16) vs
~448MB (FP32)**. Scales linearly with context length - at this model's
actual `max_position_embeddings` (40960) that's ~4.5GB vs ~9GB. This is
the real, standing argument for FP16 despite the now-negligible NPU speed
difference - it's a memory decision, not a speed one.

### lm_head / tie_word_embeddings: not on NPU, checked why

`causal_lm.cpp:254`: `createLayer(lmhead_type, lmhead_prop)` - no
`withHexagonEngine()` at all (`lmhead_type` is `tie_word_embeddings` for
this model, since `tie_word_embeddings: true`). Two separate issues:
- **Missing tag** - trivial fix (same pattern as everything else this
  session), but it's a one-time boundary op, not per-layer, so the win is
  a fraction of a millisecond at most.
- **The real blocker:** lm_head weight is `Q6_K`-quantized
  (`"lmhead_dtype": "Q6_K"`). Grepped `ggml-hexagon.cpp` and every
  `htp/*.c` kernel for `Q6_K` - **zero hits, ARM bridge side or DSP kernel
  side.** `HexagonComputeOps::gemm_q6_K_fp32` forwards straight to CPU
  unconditionally; there's no `supports_gemm_q6_K_accel_fp32()` to flip.
  This GEMM is wide (`hidden_size(1024) × vocab_size(151936)` - wider than
  any transformer-block FC layer), so accelerating it would be a real win,
  but needs a brand-new Hexagon DSP kernel (Q6_K dequant + matmul) - out of
  ARM-side reach, same category as the FP32 cache fix would have needed if
  a genuine UINT16 kernel had been chosen instead of the FP32-dtype
  workaround.
- **Pragmatic zero-new-kernel alternative, not yet done:** requantize
  lm_head to Q4_0 via `nntr_quantize` (already in this build) to reuse the
  existing, proven `gemm_q4_0_accel_fp32` path. Real accuracy tradeoff -
  Q6_K is normally chosen for output heads specifically because they're
  the most quantization-sensitive layer (directly determines logits). Not
  attempted; would want a perplexity/output-token-drift check against the
  Q6_K reference before treating this as safe.

## FP16 workarounds — full picture (verified, not guessed)

**Why FP16 is even in play:** the model's GEMM output is genuinely FP32
(`model_tensor_type: "Q4_0-FP32"`). `mha_core.cpp:849`, gated on
`#if ENABLE_FP16 && defined(__ANDROID__)`, explicitly downcasts Q/K/V/O to
FP16 right before attention — this is **this app's own build-time choice**
(presumably memory/bandwidth), not something the model or DSP requires.
Building with `-Denable-fp16=false` removes this cast entirely (Q/K/V/O
stay FP32 throughout) — but note the KV-cache *storage* dtype is a
separate, independently-hardcoded choice: `FP16` under `ENABLE_FP16`, else
`UINT16` (**not** FP32) — don't assume disabling the flag makes everything
uniformly FP32.

**nntrainer-side FP16 workarounds** (all "try DSP first, CPU only if the
bridge call itself fails" — verified via pool_stats, none are hard CPU
fallbacks in our runs):
1. Q/K/V/O FP32→FP16 staging cast (`mha_core.cpp:849`) — DSP `cpy`.
2. `try_dsp_fp16_rope` cast-chain (`mha_core.cpp:370-421`) — 3 DSP ops,
   works around the ggml-hexagon RoPE gap below.
3. K/V-cache append copy (`try_dsp_cache_copy`) — DSP `cpy`.
4. `rms_norm.cpp:285-317` — DSP first, `rms_norm_wrt_width_fp16_intrinsic`
   (CPU) only if the bridge call fails.

**ggml-hexagon-side FP16 support** — checked directly in
`ggml-hexagon.cpp`/`nntr-htp-bridge.cpp`, more nuanced than "same gaps":
- GEMM (`nntr_htp_bridge_gemm_q4_0_fp16`) — genuine native FP16 kernel.
- Flash-attention — genuine native FP16: K/V and mask are *always* F16
  internally, explicit `q_is_fp16`/`out_is_fp16` flags.
- **RoPE is the one real gap**, confirmed directly:
  `ggml-hexagon.cpp:3122-3124`: `if (src0->type != GGML_TYPE_F32) return
  false; // FIXME: add support for GGML_TYPE_F16 for src0`. This is an
  open upstream TODO, not nntrainer-specific. It's *why* llama.cpp's own
  FP16 models still work fine on this backend: llama.cpp's graphs keep
  Q/K in F32 through the RoPE op and only cast to F16 afterward (for cache
  storage) — the missing F16 kernel is never exercised in typical usage.
  nntrainer's app downcasts to F16 *before* RoPE, so it's the one place
  that actually needs (and built, via `try_dsp_fp16_rope`) a workaround.

## Corrections made mid-investigation (for the record, don't repeat these)

- Wrongly claimed `HEAD` gets "1 round-trip" without checking — it doesn't
  have the batching feature at all; corrected to 196 round-trips, verified.
- Wrongly claimed K-cache RoPE was a mandatory, unavoidable CPU fallback —
  it isn't; `try_dsp_fp16_rope` dispatches it to the DSP successfully.
- Wrongly attributed the 141-round-trip residual to the 16-buffer HTP cap —
  that cap was already raised to 64 before any measurement in this session.

## Files touched this session (uncommitted, currently on disk and deployed)

- `Applications/CausalLM/models/transformer.cpp` (engine tags; KV-cache
  `UINT16`→`FP32` dtype fix)
- `Applications/CausalLM/models/causal_lm.cpp` (KV-cache `UINT16`→`FP32`
  dtype fix, must match transformer.cpp)
- `Applications/CausalLM/layers/mha_core.cpp` (`get_reusable_fp16_scratch()`
  - shared, registered Q/K/V/O scratch tensors, both `forwarding()` and
  `incremental_forwarding()`)
- `nntrainer/hexagon/hexagon_context.cpp`
- `nntrainer/compiler/multiout_realizer.cpp`
- `Applications/meson.build`
- `Applications/CausalLM/models/gpt_oss/gpt_oss_moe_layer.h` (reverted to HEAD)
- `Applications/CausalLM/models/lfm2/lfm2_causallm.h` (reverted to HEAD)
- Full build/deploy narrative, environment gotchas (cwd-dependent DSP
  session open, missing `ml-api` download workaround, etc.):
  `docs/backend_guide/BUILD_OBSERVATIONS_2026-08-20.md`

## Build/deploy commands that actually work (for the next agent)

```bash
# 1. Rebuild nntrainer core for Android (fast incremental once configured):
ninja -C /home/anirudh/nntrainer/builddir install

# 2. Rebuild CausalLM app via ndk-build (NOT meson — meson's own
#    CausalLM targets are broken-by-design on platform=android, see
#    BUILD_OBSERVATIONS_2026-08-20.md §5-6):
cd /home/anirudh/nntrainer/Applications/CausalLM/jni
export ANDROID_NDK=/home/anirudh/android-ndk-r26d
export NNTRAINER_ROOT=/home/anirudh/nntrainer
rm -rf libs obj
ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj \
  APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk \
  causallm_core nntrainer_causallm nntr_quantize nntr_safetensors_info \
  -j $(nproc)

# 3. Deploy (artifacts land in obj/local/arm64-v8a/, not always libs/):
DEV=R3CX9078DNH
DEST=/data/local/tmp/nntrainer/causallm
for f in libcausallm_core.so nntrainer_causallm nntr_quantize nntr_safetensors_info; do
  adb -s $DEV push jni/obj/local/arm64-v8a/$f $DEST/
done
adb -s $DEV push jni/obj/local/arm64-v8a/libnntrainer.so $DEST/
adb -s $DEV shell "chmod 755 $DEST/nntrainer_causallm $DEST/nntr_quantize $DEST/nntr_safetensors_info"

# 4. Run — MUST cd into $DEST first (the DSP skel .so is resolved via a
#    bare "file:///libggml-htp-v79.so" URI relative to cwd on this
#    device/driver, NOT into the model dir):
MODEL_DIR=$DEST/models/qwen3-0.6b
adb -s $DEV shell "cp $MODEL_DIR/nntr_config_bench_900.json $MODEL_DIR/nntr_config.json"
adb -s $DEV shell "export LD_LIBRARY_PATH=$DEST:\$LD_LIBRARY_PATH; export NNTR_NUM_THREADS=4; \
  export NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_FLASH_ATTN=1 NNTR_HEXAGON_FUSED_FFN=1; \
  cd $DEST && ./nntrainer_causallm $MODEL_DIR"
```

Useful env vars discovered/confirmed this session:
- `NNTR_HTP_BRIDGE_PROF=1` — per-phase µs profiling in `nntr-htp-bridge.cpp`
  (currently not firing its report — see "Next concrete step" above).
- `NNTR_HTP_BRIDGE_POOL_DEBUG=1` — pool hit/miss tracing (this is what
  produced the `pool_stats ...` lines used throughout this doc).
- `GGML_HEXAGON_PROFILE=1` — DSP-side batch-duration-only profiling
  (narrower than `NNTR_HTP_BRIDGE_PROF`, doesn't cover host-side cost).
