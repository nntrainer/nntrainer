# NPU Prefill Validation — 2026-08-19: GAP A / GAP B Closed

## Summary

Implemented and validated the two gaps identified in the 2026-08-18 review:

- **GAP B (dispatch unification)**: `rms_norm`, `addition`, and `mha_core` are now
  tagged `engine=cdsp` (like the FC/QKV/GateUp GEMM layers) instead of relying on
  ad hoc, disconnected `getenv()`/`dlopen()` checks in each file. `compute_engine`
  is now the single source of truth for both (a) `LayerNode`'s flush-guard skip
  and (b) whether a layer's own `forwarding()` attempts DSP dispatch.
- **GAP A (zero-copy for elementwise ops)**: confirmed via
  `NNTR_HTP_BRIDGE_POOL_DEBUG=1` tracing that the "0% pool hit rate" reported on
  2026-08-18 was a stale/mismeasured result (that run predates several
  correctness fixes from later the same day). With a corrected measurement:
  **ADD and RMSNorm's activation operands (in/out) already hit the zero-copy
  pool at 100%** — the only genuine, reproducible miss is `rms_norm:gamma`
  (a weight tensor, deliberately excluded from the graph's rpcmem-backed
  activation pool). A fix was implemented and found to have a real device-level
  scaling limit (see "What's still open" below) — currently reverted to the
  safe staging-memcpy fallback for gamma specifically, everything else fixed.

Net result: **prefill dropped from 1655ms (CPU-only) to ~550ms (NPU hybrid) —
a ~3x speedup**, versus the 2.7% speedup reported on 2026-08-18. Most of this
gap between "2.7%" and "3x" was not GAP A/B themselves — it was two build/infra
bugs uncovered along the way (see below) that meant the 2026-08-18 numbers were
never actually exercising the code paths this session fixed.

## What was implemented

### GAP B: unified dispatch

1. `nntrainer/layers/layer_context.h`/`.cpp`: `RunLayerContext` gained a
   `compute_engine` field + `getComputeEngineType()`/`setComputeEngineType()`
   accessors, set from `LayerNode::compute_engine` in
   `LayerNode::configureRunContext()` (`layer_node.cpp`). This mirrors the
   `InitLayerContext` accessor of the same name that already existed for
   graph-construction-time use; `RunLayerContext` had no equivalent for a
   layer's own `forwarding()` to query at runtime.
2. `nntrainer/hexagon/hexagon_context.cpp`: registered `AdditionLayer` under
   the `"cdsp"` context (same pattern as the existing `QKVLayer`/`GateUpLayer`
   registration) — core-to-core, no layering concerns.
3. `Applications/CausalLM/models/transformer.cpp`:
   `Transformer::registerCustomLayers()` now also registers
   `causallm::RMSNormLayer` and `causallm::MHACoreLayer` under `"cdsp"`,
   probed at runtime (`try`/`catch` around `getRegisteredContext("cdsp")`) so
   builds without `-Denable-hexagon-cdsp=true` are unaffected. `withHexagonEngine()`
   now wraps the `rms_norm`/`addition`/`mha_core` construction call sites in
   `createTransformerDecoderBlock()`/`createAttention()`, gated on the existing
   `NNTR_USE_HEXAGON_CDSP` check, exactly like the FC/QKV/GateUp layers.
4. `rms_norm.cpp`, `addition_layer.cpp`, `mha_core.cpp`: the primary DSP-dispatch
   gate in each is now `context.getComputeEngineType() == CDSP` (or, for
   `mha_core.cpp`'s `one_batch_incremental_forwarding()` helper — which has no
   `RunLayerContext` of its own — a cached `is_cdsp_engine` member set once in
   `finalize()`). `NNTR_HEXAGON_NO_ELEM_OPS`/`NNTR_HEXAGON_FLASH_ATTN` remain as
   secondary AND'd override knobs for benchmarking, not the primary decision.

### GAP A: zero-copy verification + gamma fix (partial)

- Confirmed via `NNTR_HTP_BRIDGE_POOL_DEBUG=1` that `add:a/b/out`,
  `gemm_q4_0:act/out`, and `rms_norm:in/out` all hit the pool at 100%.
  `rms_norm:gamma` misses 100% of the time — expected, since gamma is a weight
  tensor and weights are deliberately excluded from the rpcmem-backed
  activation pool (CMA budget concern for the large GEMM weight matrices).
- Implemented a per-layer persistent rpcmem copy of gamma
  (`RMSNormLayer::getOrCreateGammaRpcmem()`, `rms_norm.cpp`/`.h`), following
  the same allocate-once-register-once pattern `KVCacheManager` uses for the
  K/V cache. **This is currently disabled** — see below.

## Follow-up fix (same day, after initial write-up): `mha_core` wasn't actually tagged for Qwen3

The initial GAP B fix wrapped `mha_core`'s construction in `withHexagonEngine()`
inside `Transformer::createAttention()` (`Applications/CausalLM/models/transformer.cpp`)
— the **base-class** implementation. Qwen3 doesn't use it: `Qwen3Transformer`
overrides `createAttention()` in `Applications/CausalLM/models/qwen3/qwen3_causallm.cpp`
with its own `mha_core` construction (QKV-batched, q_norm/k_norm reshaped-norm),
which was never touched by the original fix. Confirmed via on-device tracing
(`LayerNode::setProperty()` dump): the actual properties reaching `mha_core`
for Qwen3 never included `engine=cdsp` at all, so `compute_engine` stayed at
its `CPU` default and the LayerNode-level flush guard fired before every
attention call (`layer0_attention` through `layer27_attention`), and
`is_cdsp_engine` (used to gate RoPE dispatch) was always false — despite the
class itself being correctly registered under the `cdsp` context and instances
constructing successfully (no "Key is not found" error, which is what made
this easy to miss). Fixed by adding `withHexagonEngine()` to the `mha_core`
construction in `qwen3_causallm.cpp` too. Re-verified on-device: `layerN_attention`
no longer appears in the flush trace at all.

Note this same pattern (a model-specific `createAttention()` override
constructing `mha_core` without `withHexagonEngine()`) likely also affects
`qwen2_causallm.cpp`, `gpt_oss_causallm.cpp`, `lfm2_causallm.cpp`,
`gemma3_causallm.cpp`, `gemma4_causallm.cpp`, `gpt_oss_cached_slim_causallm.cpp`,
and `timm_vit_transformer.cpp` (all have their own `"mha_core"` construction
call sites, per `grep -rn "\"mha_core\"" Applications/CausalLM/models/`) — not
verified/fixed this session, since only Qwen3 was in scope, but worth checking
before assuming any of those models get attention dispatch today.

RoPE dispatch (`nntr_htp_bridge_rope`) still never fires even with this fix —
`rms_norm:in`/`out` and `add:*` hit the zero-copy pool at 100% but no
`rope:inout` line ever appears in `pool_stats`. Not root-caused this session;
plausibly the Q/K tensors reaching `one_batch_incremental_forwarding` are a
dtype the FP32-only DSP RoPE kernel gate rejects (the gate checks
`query_step.getDataType() == FP32`) even though RMSNorm's tensors on the same
model do pass that check — worth instrumenting directly rather than assuming.

## KV-cache DSP copy (follow-up, same day)

Wired up `nntr_htp_bridge_cpy` (previously dead code, F32-only) to actually
dispatch KV-cache append to the DSP. Two real findings changed the plan
mid-implementation:

1. **The DSP kernel didn't need new code.** `htp/cpy-ops.c`'s `op_cpy()`
   already has same-shape F32↔F16 and F16↔F16 conversion kernels
   (`cpy_thread_f16_f32_sameshape`, etc., using the existing HVX
   `hvx_copy_f16_f32_uu`/`hvx_copy_uu` primitives) — the host-side bridge
   function just never exposed a dtype parameter to select them. Extended
   `nntr_htp_bridge_cpy(const void *src, void *dst, unsigned n_elems, int
   src_is_fp16, int dst_is_fp16)` to pick the right kernel via the tensor
   descriptor's `type` field, mirroring exactly how `nntr_htp_bridge_flash_attn`
   already selects `out_is_fp16`. No DSP skel rebuild was needed — only
   `libggml-hexagon.so` (host-side bridge) changed.
2. **Q/K/V activations and the KV cache are both FP16 on this model**, not
   FP32 as the original (dead) code's comment assumed. This matters because:
   - **V-cache append**: no rotation needed, purely a copy — now dispatches to
     the DSP with zero host-side flush. Confirmed via `NNTR_HTP_BRIDGE_POOL_DEBUG=1`:
     `cpy:dst` hits the KV-cache's registered rpcmem pool 28/28 times (100%).
   - **K-cache append**: still can't move to the DSP. K needs an actual RoPE
     *rotation* (not just a copy) before caching, and the DSP RoPE kernel
     (`htp/rope-ops.c`) is F32-only — no FP16 case exists there. Since this
     model's K tensor is FP16, `nntr_htp_bridge_rope` never dispatches for K,
     so the CPU fallback (which computes the real rotation) always runs and
     always needs its flush. Extending RoPE to FP16 would need real new DSP
     kernel work (a genuinely different, larger task from what was asked) -
     not done this session.

Net effect on this model: correct and functional, but **not a measurable
speedup** — `cpy:src` (V's source, a transient GEMM-output view) isn't itself
pool-registered, so the DSP call still pays a staging memcpy in, and real
round-trips went from 140 to 168 (+28, one per block) with prefill latency
unchanged within noise (~1%, 790ms → 799ms on the 909-token benchmark). Left
in place since it's harmless and correctly falls back when unavailable — it's
the right infrastructure for the day RoPE gets FP16 support or a model with
FP32 attention activations runs through this path.

## K/Q RoPE via cast-rotate-cast (tried, measured, reverted)

Since the DSP RoPE kernel is F32-only but this model's Q/K are FP16, tried
chaining three DSP ops that already exist (no new kernel code needed) instead
of building a new FP16 RoPE kernel:
1. `cpy` (F16→F32): cast the FP16 activation into a scratch F32 rpcmem buffer
2. `rope` (F32, existing): rotate the scratch buffer in place
3. `cpy` (F32→F16): cast the result into its destination (KV cache for K,
   back into the activation tensor in-place for Q)

All three enqueue into the same batch with no flush needed between or before
them (same FIFO-chaining guarantee `nntr_htp_bridge_ffn_swiglu` already
relies on). The scratch buffer is allocated and registered **once** (not per
layer - registering many small pools was what hung the DSP in the earlier
gamma-rpcmem attempt, see above) and reused across every block's Q and K
calls.

**Verified correct**: `NNTR_HTP_BRIDGE_POOL_DEBUG=1` showed `rope:inout`
hitting 56/56 (28 blocks × Q+K), no fallback warnings, same first-generated-
token as before.

**Measured as a real regression, not a speedup**: 3 runs at 909 tokens
averaged ~910ms vs a 799ms baseline without it (+14%). Root cause: the real
FastRPC round-trip count did **not** drop when this replaced the CPU
fallback's `flush_if_batch_active()` calls - those flushes were mostly
finding nothing pending to flush at that exact point anyway (cheap no-ops,
not real round trips), so removing them saved ~nothing, while the 3 extra
DSP ops per rotation cost real HVX cycles the tiny CPU RoPE computation
(head_dim=128, a handful of heads) never needed. **Reverted** (the function
now returns `false` immediately, with the full implementation left in place
and documented) - confirmed prefill back to the 778-806ms baseline after
reverting.

## q_norm/k_norm ported to DSP — a real, confirmed win

Ported `ReshapedRMSNormLayer` (the RMSNorm applied inside attention to Q/K,
distinct from the residual-stream `attention_norm`/`ffn_norm` which already
dispatched) to the DSP, reusing the same `nntr_htp_bridge_rms_norm` bridge as
`rms_norm.cpp`, plus registering it under the `cdsp` context and tagging its
construction in `qwen3_causallm.cpp` (its own `createAttention()` override -
same lesson as `mha_core`'s earlier miss, this layer is also only built in
the Qwen3-specific file, not the base `Transformer::createAttention()`).

Initially assumed q_norm/k_norm's input would be FP16 (matching mha_core's
Q/K) and built the same cast-rotate-cast-style chain as the (reverted) FP16
RoPE attempt. On-device tracing showed this was wrong: **q_norm/k_norm's
actual in/out/gamma tensors are FP32** - the FP16 downcast mha_core.cpp sees
happens later, at the connection between q_norm's output and mha_core's
input, not inside q_norm/k_norm itself. So the direct dispatch path (no
cast needed, mirrors `rms_norm.cpp` exactly) is what actually fires. The
FP16 cast-chain path is kept in the code as an untested fallback in case a
future config's q_norm/k_norm really does see FP16 data.

**Verified correct**: `rms_norm:in`/`rms_norm:out` pool_stats jumped from
57 hits (attention_norm + ffn_norm + output_norm) to 113 (+56 = 28 blocks ×
q_norm+k_norm), zero fallback warnings, same first-generated-token.

**Measured as a real, reproducible improvement** (unlike the RoPE attempt):
3 runs at 909 tokens averaged ~764ms (752/749/791) vs the ~794ms baseline
(778/797/806) from before this change - a consistent ~4% reduction. Real
FastRPC round-trip count stayed at 168 (unchanged) even though real work
moved to the DSP and CPU-forced flushes were removed - reinforcing the same
lesson as the RoPE experiment that round-trip *count* doesn't reliably
predict wall-clock impact in either direction; only measuring directly
settles it.

## What's still open

- **Gamma zero-copy is implemented but disabled.** On-device testing found
  that registering one small (4KB) rpcmem pool per `RMSNormLayer` instance
  works for the first ~12-13 layers, then permanently hangs the DSP/FastRPC
  driver on the next `dspqueue_write`/`flush_pending` — reproduced twice, at
  a consistent point (pool addresses/sizes are consistent with a fixed
  FastRPC buffer-registration/mmap slot limit being exhausted, not a bug in
  the copy/registration sequence itself). Qwen3-0.6B has 56 `RMSNormLayer`
  instances, well past that point. The fix is a **shared rpcmem arena for all
  gamma vectors** — one `rpcmem_alloc` + one `register_activation_pool` call
  for the whole model, each layer copying its gamma into its own offset
  (the same pattern already used for the graph's activation `tensor_pool` and
  for `KVCacheManager`'s K/V cache) — not one registration per layer. That
  needs coordination across `RMSNormLayer` instances (a shared allocator keyed
  by model/session, sized once all layers are known), which is a bigger change
  than this session's scope. The code is left in place
  (`RMSNormLayer::getOrCreateGammaRpcmem()`, `GammaRpcMem` class) with an early
  `return nullptr;` and a comment explaining why, so the next session doesn't
  have to rediscover this.
- RoPE dispatch (`nntr_htp_bridge_rope`) never fires in this config — no
  `rope:inout` pool_stats line appears at all, meaning `use_rope` is false (or
  some other gate) for this model/build. Not investigated this session; worth
  checking before assuming the RoPE dispatch path (which does get the
  compute_engine-based gate fix in this diff) is actually exercised anywhere.
- Real FastRPC round-trip count is 140 this run — higher than the
  113 previously reported (which, per the bugs below, wasn't a reliable
  number). Now that RMSNorm/ADD/MHACore genuinely dispatch, this is a real
  baseline to optimize down from, not yet investigated further.

## Two infra bugs found (not present in prior docs)

Both of these made every "NPU vs CPU" comparison and every diagnostic run
earlier in this session look broken or contradictory, before being found:

1. **`RunLayerContext`'s new field was placed mid-struct instead of appended**,
   changing the byte offset of every member after it
   (`weights`/`inputs`/`outputs`/`tensors`). Combined with bug #2 below (a stale
   header snapshot in a different build tree), this produced a real ODR
   violation across `libnntrainer.so` and `libcausallm_core.so` and crashed
   with `SIGSEGV` in `RunLayerContext::weightHasGradient()` during
   `NetworkGraph::initialize()` — every time, reproducibly. Fixed by moving the
   new field to the end of the member list.
2. **`Applications/CausalLM`'s own ndk-build silently fails to refresh
   `jni/libs/arm64-v8a/*.so` from `obj/local/arm64-v8a/*.so`** — the exact
   "silent staleness" class of bug the 2026-08-18 session log already warned
   about for `nntrainer`'s own build, but this time on the **consuming app's**
   side. Several "fixed" test runs during this session were actually still
   running a `libnntrainer.so` from *before* the very first fix, because
   `adb push` was faithfully pushing a stale `jni/libs` copy while the real,
   freshly-built `.so` sat untouched in `obj/local`. Confirmed via `md5sum`;
   worked around by always force-`cp`ing from `obj/local` to `jni/libs` before
   pushing (the same pattern the nntrainer-core build recipe already uses —
   this app-level build needs the identical treatment every time).
3. Also had to hand-populate `builddir/android_build_result/` (headers via
   `meson-info/intro-installed.json`'s source→dest manifest, libs via direct
   copy from `build-cdsp/obj/local/arm64-v8a/`) because `builddir` itself is
   configured as a plain native build (not a real Android cross-build) and a
   full `ninja install` there fails on unrelated pre-existing `-Werror`/gtest
   issues across many unrelated `Applications/*` targets. This is a real gap
   in the documented build recipe worth fixing properly (either repair
   `builddir`'s cross-compilation config, or document the manifest-based
   workaround) so the next session doesn't have to rediscover it.

## Numbers (S24, `R3CX9078DNH`, 650-token prompt, `qwen3-0.6b`)

| Mode | Prefill (ms) | Prefill TPS | Real FastRPC round-trips |
|---|---|---|---|
| CPU-only (no `NNTR_USE_HEXAGON_CDSP`) | 1655 | 393 | 0 |
| NPU hybrid (this session's fixes) | 526 / 580 (2 runs) | ~1180 avg | 140 |

Both modes generate the same first token (`&`) — not a rigorous correctness
check (no full-sequence logit/perplexity comparison was done), but consistent
with the bar prior docs in this series have used.

## Build/verify recipe notes (additions to the 2026-08-18 recipe)

- `meson configure build-cdsp -Dwerror=false` is needed if you ever need
  `ninja -C build-cdsp install` to regenerate `android_build_result` from
  scratch — several unrelated `Applications/*` targets have pre-existing
  `-Wunused-private-field` issues that are otherwise fatal under this
  project's default `werror=true`.
- `mkdir -p build-cdsp/ml-api-inference/include` (empty is fine) before that
  install step, or `prepare_ml-api.sh` will attempt a network fetch that fails
  in this environment (same class of issue the 2026-08-18 doc already noted
  for `build-cdsp/ml-api-inference/lib`).
- Always force-copy `Applications/CausalLM/obj/local/arm64-v8a/*.so` over
  `jni/libs/arm64-v8a/*.so` before pushing (bug #2 above) — `cmp -s ... || cp`
  each of the 5 files, every time, not just when something "seems" stale.
