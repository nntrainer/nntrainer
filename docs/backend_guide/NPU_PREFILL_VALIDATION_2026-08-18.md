# Validation of "Why You Can't (Yet) Put Everything on NPU" Analysis

**Date:** 2026-08-18
**Validated against:** Source code in `nntrainer/`, `ggml-hexagon/`, and `NPU_PREFILL_SESSION_LOG_2026-08-18.md`
**Purpose:** Independent code-level verification of every claim in the analysis document.

---

## Summary

The analysis is **overwhelmingly accurate**. Every major claim is confirmed by the source code. There are a few minor imprecisions (noted below) that do not affect the conclusions. The three-layer problem decomposition and the three approaches are all structurally correct.

---

## "What's NOT a blocker" — Validation

### 1. DSP kernels exist for every op ✅ VALIDATED

**Claim:** `HTP_OP_RMS_NORM_MUL`, `HTP_OP_ROPE`, `HTP_OP_FLASH_ATTN_EXT`, `HTP_OP_MUL_MAT`, `HTP_OP_ADD`, `HTP_OP_GET_ROWS`, `HTP_OP_SILU` — all exist in `htp/*.c`.

**Finding:** Confirmed in `ggml-hexagon/ggml/src/ggml-hexagon/htp/htp-ops.h`:
- `HTP_OP_RMS_NORM_MUL` (line 61) ✅
- `HTP_OP_ROPE` (line 77) ✅
- `HTP_OP_FLASH_ATTN_EXT` (line 78) ✅
- `HTP_OP_MUL_MAT` (line 58) ✅
- `HTP_OP_ADD` (line 55) ✅
- `HTP_OP_GET_ROWS` (line 79) ✅
- `HTP_OP_UNARY_SILU` (line 63) ✅ — **minor naming inaccuracy**: the enum is `HTP_OP_UNARY_SILU`, not `HTP_OP_SILU`. Also `HTP_OP_GLU_SWIGLU` (line 70) exists for the fused SwiGLU path.
- `HTP_OP_CPY` (line 81) also exists, relevant to the KV-cache porting roadmap (§6.2 of the session log).

**Verdict:** No missing kernel. The claim is correct.

### 2. rpcmem for all tensors ✅ VALIDATED (with caveat)

**Claim:** `HexagonRpcAllocator` already exists for GEMM activations. Extending it to all tensors is mechanical.

**Finding:** Confirmed in `nntrainer/hexagon/hexagon_rpc_allocator.cpp` and `.h`:
- `HexagonRpcAllocator` class exists, inherits `MemAllocator`, allocates via `rpcmem_alloc2`.
- On every `alloc()`, it calls `nntr_htp_bridge_register_activation_pool()` to register the pool with the bridge for zero-copy dispatch.
- The bridge (`nntr-htp-bridge.cpp` lines 756-789) stores these as `ext_act_pools` and uses `find_ext_pool()` to check if an activation pointer is in a registered pool — if so, it's mapped in-place (zero-copy).

**Caveat (not reflected in the analysis):** The session log §6.4 explicitly notes that weights are **deliberately NOT** routed through rpcmem: *"weights stay on CPU... not DSP-registered... routing weights to rpcmem too would needlessly exhaust the scarce CMA pool."* The `neuralnet.cpp` `setComputeBackend("", "cdsp")` only routes the **activation** tensor pool to `HexagonRpcAllocator`. So "extending to all tensors is mechanical" understates a real CMA budget constraint — it's mechanical for activations, but weights are a design tradeoff, not a mechanical extension.

### 3. Enqueue/flush API ✅ VALIDATED

**Claim:** `begin_batch()`/`end_batch()` already exists.

**Finding:** Confirmed in `nntr-htp-bridge.cpp`:
- `nntr_htp_bridge_begin_batch()` (line 2900): locks `state.mtx`, sets `batch_mode = true`, clears `pending_copies`, ensures session.
- `nntr_htp_bridge_end_batch()` (line 2938): flushes all pending ops, copies back deferred outputs, logs flush count, resets `batch_mode`, unlocks `mtx`.
- `nntr_htp_bridge_flush()` (line 2915): mid-batch flush without ending the batch scope.
- `nntr_htp_bridge_flush_if_batch_active()` (line 3061): the systemic sync-guard — flushes if and only if `batch_mode` is true.

Wired into `causal_lm.cpp` via dlsym (confirmed by search results showing `begin_batch`/`end_batch` resolution and call sites).

---

## "What IS the blocker" — Validation

### Layer 1: The sync guard fires before every layer ✅ VALIDATED

**Claim:** `LayerNode::forwarding()` calls `nntr_hexagon_flush_if_batch_active()` unconditionally before every layer. 509 flushes for Qwen3's 28-layer × ~18-nodes-per-block structure.

**Finding:** Confirmed in `nntrainer/layers/layer_node.cpp`:
- Lines 795-816: `nntr_hexagon_flush_if_batch_active()` is a static function that lazily dlopens `libggml-hexagon.so` and dlsyms `nntr_htp_bridge_flush_if_batch_active`.
- Line 825: Called unconditionally at the start of `LayerNode::forwarding()`.
- Line 869: Called unconditionally at the start of `LayerNode::incremental_forwarding()`.
- The `[LAYER_FLUSH]` trace (lines 807-814) logs which layer triggered each real flush.

**On the "509" number:** The session log §2.6 and §5.1 clarify that 509 is the `get_flush_count()` metric — which counts every *call* to the flush path, including no-ops where nothing was queued. The *real* FastRPC round-trip count (`real_flush_count`) was 112 (legacy-only) or 225 (everything-on). The analysis text uses "509 flushes" which is the guard-fire count, not the real round-trip count. This is a **conflation** — but the fix described (making the guard layer-aware) is correct regardless, because even the 112 real round-trips are inflated by the guard firing at layer boundaries where no CPU read is needed.

**On "0 mid-pass flushes for a pure NPU forward pass":** This is **aspirational, not fully accurate**. Even with Approach A, `mha_core.cpp` has 5 explicit `flush_if_batch_active()` calls per block (lines 961, 975, 987, 1034, 1098) for KV-cache management. These would still fire unless KV-cache append is also ported to the DSP (§6.2 of the session log). So the real number with Approach A alone is ~5×28 = 140 flushes from mha_core, not 0. The "0 mid-pass flushes" claim only holds if *both* Approach A *and* the KV-cache porting (§6.2) are done.

### Layer 2: CPU-only glue between NPU ops ✅ VALIDATED

**Claim:** KV-cache management, embedding lookup, sampling/argmax, and the layer loop itself are CPU-side control flow, not compute kernels.

**Finding:**

1. **KV-cache management** (`mha_core.cpp`): Confirmed. Lines 917-997 show `getSharedDataTensor()` for cache slicing, `apply_rotary_emb_tensor_v2()` for cache append (a CPU copy with optional rotation), and `copyData()` for V-cache append. These are host-side memory management operations — tensor slicing, pointer arithmetic, and memcpy. There is no `HTP_OP_KV_CACHE_APPEND` because it's bookkeeping, not compute. ✅

2. **Embedding lookup** (`tie_word_embedding.cpp`): Confirmed. Lines 200-298 show `incremental_forwarding_embedding()` doing `getSharedDataTensor()` (pointer arithmetic) + `copyData()` / `dequantize_row_q4_0()` (gather + dequant). It's a C++ object with tensor views, not a raw `HTP_OP_GET_ROWS` dispatch. ✅

3. **Sampling/argmax**: Confirmed via search results. `causal_lm.cpp` reads logits as `float*` and calls `applyTKP()` (in `llm_util.hpp`) for top-k/top-p sampling. `argmax` is used when `do_sample == false`. These are CPU logic. ✅

4. **The layer loop**: Confirmed. `NetworkGraph::forwarding()` (in `network_graph.h`) iterates `LayerNode` objects and calls virtual `forwarding()`. The graph executor has no visibility into what ops a layer will execute. ✅

### Layer 3: No op-level IR ✅ VALIDATED

**Claim:** nntrainer's `NetworkGraph` is a list of `LayerNode` objects with opaque `forwarding()` virtual methods. No op-level view. The graph executor doesn't know what ops a layer will execute until it runs.

**Finding:** Confirmed in `nntrainer/graph/network_graph.h`:
- `NetworkGraph` holds a `GraphCore graph` of `LayerNode` objects (line 605).
- `forwarding()` iterates over sorted `LayerNode`s and calls their `forwarding()` (lines 234-240).
- There is no op-level representation — no flat list of primitive ops, no lowering step, no op-level enqueue.
- The bridge file's own header comment (line 7) explicitly states: *"nntrainer has no ggml_cgraph - it calls in per-op from its own FullyConnectedLayer"*.

This is the structural root cause. ggml-hexagon achieves "one flush for the whole graph" because `ggml_cgraph` is a flat op list enqueued in one batch. nntrainer has no equivalent. ✅

---

## "The path to everything on NPU" — Validation

### Approach A: Smart sync guard ✅ VALIDATED as feasible

**Claim:** Tag each `LayerNode` as NPU-bound or CPU-bound at compile time. Sync guard only flushes at NPU→CPU boundaries. 509→~5-10 flushes.

**Finding:** The codebase already has the infrastructure for this:
- `LayerNode` already has a `compute_engine` field (line 316: `setComputeEngine()`) and reads it from the `ComputeEngine` property (line 642-644).
- The sync guard (`nntr_hexagon_flush_if_batch_active`) is a single call site that could be conditioned on `compute_engine`.
- The session log §6.1 describes this exact approach and calls it "highest leverage, do this first."

**Nuance:** The session log §5.3 found that even with perfect batching, elementwise ops (RMSNorm/RoPE/ADD) on HVX don't beat CPU NEON — the extra 113 real round-trips they add cost only ~12ms but save roughly the same. So Approach A alone won't dramatically change prefill time for those ops. The real win is for GEMM/attention chains that currently get unnecessary flushes between them. The "509→10" reduction is real for the *guard fire count*, but the *wall-clock* impact is modest (the session measured 575ms→582ms, i.e. a wash) because the guard's no-op flushes are cheap (~0.1ms each). The analysis's claim that "this alone gets you most of the benefit" is **optimistic** — the session log's measured data shows the benefit is near zero for the ops already ported, because the real round-trip cost is already low.

### Approach B: Trace-and-replay ✅ VALIDATED as plausible

**Claim:** Run the forward pass once, record the sequence of NPU ops, replay on subsequent passes.

**Finding:** The bridge already has the enqueue/flush infrastructure. A trace-and-replay would need to:
1. Record the sequence of `enqueue_op()` calls (the `htp_opnode` structs) during a first pass.
2. On subsequent passes, replay the recorded op sequence as one batch.
3. This requires shapes to be fixed (true for inference, tricky for training).

This is feasible with the existing infrastructure. The bridge's `htp_opbatch_req` struct (in `htp-ops.h` lines 194-204) already supports batched op submission. ✅

### Approach C: Op-level IR ✅ VALIDATED as the structural fix

**Claim:** Rewrite layers to emit explicit op nodes. `NetworkGraph::compile()` lowers `LayerNode`s into primitive ops. One flush for the entire forward pass.

**Finding:** This is accurately described. The current `NetworkGraph` has no op-level representation. Implementing this would require:
1. An op-level IR (like `ggml_cgraph` / `htp_op_desc`).
2. Each layer's `forwarding()` would need to emit ops into a builder instead of doing math directly.
3. `compile()` would lower `LayerNode`s into primitive ops.
4. The executor would enqueue all ops at once.

This is the "real fix" but is multi-week effort. The analysis correctly identifies this as what ggml does natively. ✅

---

## Inaccuracies and Nuances

### 1. "509 flushes" conflation
The analysis uses "509 flushes" as the problem statement, but the session log distinguishes between:
- `get_flush_count()` = 509 (guard fires, including no-ops) — this is what the analysis quotes.
- `real_flush_count` = 112 (legacy) or 225 (everything) — actual FastRPC round trips.

The fix (Approach A) targets the 509 guard fires, but the *wall-clock* cost of those is only ~12ms total (§3.4). The real performance bottleneck is not the guard's no-op flushes — it's that the guard *prevents chaining* of real ops that could otherwise share a single flush. The analysis's framing is directionally correct but the magnitude of benefit is overstated.

### 2. "Extending rpcmem to all tensors is mechanical"
Weights are deliberately NOT in rpcmem (CMA budget constraint, §6.4). This is a design tradeoff, not a mechanical extension. For QLoRA/small models the weight volume in rpcmem would be small (LoRA adapters, RMSNorm gammas), but it's not zero-effort.

### 3. "0 mid-pass flushes for a pure NPU forward pass"
This requires both Approach A *and* porting KV-cache management to the DSP (§6.2). With Approach A alone, `mha_core.cpp`'s 5 explicit `flush_if_batch_active()` calls per block still fire. The real number is ~140 flushes (5×28), not 0.

### 4. "HTP_OP_SILU" naming
The actual enum is `HTP_OP_UNARY_SILU` (line 63 of `htp-ops.h`). Minor.

### 5. "This alone gets you most of the benefit"
The session log's measured data (§3.2 vs §3.3: 575ms vs 582ms) shows that adding RMSNorm/RoPE/ADD-on-NPU + batching produced **no measurable net improvement** over the pre-existing baseline. The analysis's claim that Approach A "gets you most of the benefit" is not supported by the session's measured data. The benefit of Approach A is in *enabling future chaining* (for Approach B/C), not in direct wall-clock improvement with the current op set.

---

## Bottom Line

The analysis is **structurally correct and well-reasoned**. Every claim about what exists (DSP kernels, rpcmem, batch API, sync guard, CPU glue, no op-level IR) is confirmed by the source code. The three-layer decomposition is accurate. The three approaches are correctly described and feasible.

The main overstatement is in the expected benefit of Approach A. The session log's measured data shows that the guard's no-op flushes are cheap (~0.1ms each), so eliminating them alone won't dramatically change prefill time. The real value of Approach A is that it *unblocks* Approach B/C by making it possible to chain real ops without unnecessary CPU round-trips between them. The analysis should frame Approach A as an *enabler* rather than a *standalone win*.

The remaining CPU touches (KV-cache management, sampling, the layer loop) are correctly identified as control flow that stays on CPU — cheap and infrequent, not the bottleneck. The session log §6.6 confirms this: *"With 6.1-6.5 done, the only remaining CPU-visible reads for a full prefill should be the final logits."*

**For the goal of running entire prefill on NPU end-to-end efficiently**, the recommended path is:
1. **Approach A** (smart sync guard) — do this first, it's the prerequisite.
2. **Port KV-cache append** (§6.2) — eliminates the 5/block mha_core flushes.
3. **Port embedding lookup** (§6.5) — eliminates the CPU gather.
4. **Approach B** (trace-and-replay) — once A + KV-cache + embedding are done, trace the op sequence and replay as one batch. This is the practical path to "one flush for the whole prefill."
5. **Approach C** (op-level IR) — the long-term structural fix, only if B's shape-fixed limitation is a problem.

Signed-off-by: Cline SR <noreply@anthropic.com>
