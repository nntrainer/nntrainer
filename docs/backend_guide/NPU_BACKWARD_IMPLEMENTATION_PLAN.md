# NPU Backward Pass Implementation Plan

**Date:** 2026-08-19
**Goal:** Enable transformer block backward pass on NPU (Hexagon HMX/HVX) for LoRA training with frozen base weights.

---

## 1. Scope and Assumptions

### What This Plan Covers
- Backward pass for a Qwen3-style transformer block on NPU
- LoRA training: base weights frozen, only LoRA adapters trainable
- Prefill mode (batch > 1 tokens) — decode backward is out of scope (M=1 GEMMs stay on CPU)
- All tensors in rpcmem (assumes all-rpcmem work is done or in progress)

### What This Plan Does NOT Cover
- Full fine-tuning (all weights trainable) — extends naturally but more memory
- Decode-step backward (M=1, GEMVs — CPU is faster)
- Optimizer step (Adam/SGD) — stays on CPU, operates on LoRA gradients only

### Key Insight
For frozen-weight backward, each FC layer needs only `dX = dY · W^T` (activation gradient propagation). No `dW` computation for frozen layers. Only LoRA layers need `dW`. Attention has no weights at all — its backward is pure activation gradients (dQ, dK, dV).

---

## 2. Current State: What Already Works

| Component | Forward on NPU? | Backward on NPU? | Bridge function |
|-----------|-----------------|-------------------|-----------------|
| FC GEMM (q4_0) | ✅ | ❌ (uses CPU sgemm) | `nntr_htp_bridge_gemm_q4_0` |
| FC GEMM (fp32) | N/A | ⚠️ Bridge exists, untested | `nntr_htp_bridge_sgemm_fp32` |
| Batched GEMM (fp32) | N/A | ⚠️ Bridge exists, untested | `nntr_htp_bridge_sgemm_batch_fp32` |
| Flash attention | ✅ | ❌ | `nntr_htp_bridge_flash_attn` |
| RoPE | ✅ | ❌ (but same kernel works) | `nntr_htp_bridge_rope` |
| RMSNorm | ✅ | ❌ | `nntr_htp_bridge_rms_norm` |
| Residual ADD | ✅ | ❌ | `nntr_htp_bridge_add` |
| Fused FFN | ✅ | ❌ | `nntr_htp_bridge_ffn_swiglu` |
| KV-cache copy | ❌ (CPU) | N/A | `nntr_htp_bridge_cpy` (unused) |
| Enqueue/flush | ✅ | ✅ (reusable) | `begin_batch`/`end_batch`/`flush` |

### What's Already Wired
- `sgemm_fp32` in `hexagon_compute_ops.cpp` already dispatches to `nntr_htp_bridge_sgemm_fp32` with TransA/TransB support
- The FC layer's `calcDerivative` calls `dot_deriv_wrt_1` → `sgemm_fp32(TransB=true)`
- The FC layer's `calcGradient` calls `dot_deriv_wrt_2` → `sgemm_fp32(TransA=true)`
- The smart sync guard in `layer_node.cpp` already skips flush before CDSP layers
- Batch mode (`begin_batch`/`end_batch`) is wired in `causal_lm.cpp`

### What's Missing
- No backward dispatch in `mha_core.cpp` (attention backward is CPU-only)
- No backward dispatch in `rms_norm.cpp` (RMSNorm backward is CPU-only)
- No backward dispatch in `addition_layer.cpp` (ADD backward is CPU-only)
- No backward dispatch in FFN/SwiGLU (backward is CPU-only)
- `sgemm_fp32` bridge is loaded but **untested on-device** — transpose correctness unverified
- No activation checkpointing infrastructure

---

## 3. Implementation Phases

### Phase 1: GEMM Backward Validation (1-2 days)

**Goal:** Verify that `sgemm_fp32` with TransA/TransB produces correct results on-device.

**Why first:** Every backward path depends on transposed GEMMs. If the bridge has a transpose bug, nothing works. The `NPU_WHOLE_GRAPH_DELEGATION_STUDY` noted a "transpose-logic bug" — must resolve this first.

**Tasks:**
1. Write a unit test (`test/htp/test_sgemm_backward.cpp`) that:
   - Runs `sgemm_fp32(A, B, C, M, N, K, transA=0, transB=0)` on NPU
   - Runs the same on CPU
   - Compares results (max abs error < 1e-5)
   - Repeats with `transA=1`, `transB=1`, `transA=1 transB=1`
   - Tests with training-relevant shapes: M=650, N=1024, K=1024 (QKV backward)
   - Tests with M=650, N=3072, K=1024 (FFN backward)

2. If transpose is broken:
   - Debug in `nntr-htp-bridge.cpp:1773` (`nntr_htp_bridge_sgemm_fp32`)
   - The bridge swaps src0/src1 to compensate for the matmul_2d kernel's transposed output
   - May need to physically transpose A/B before enqueuing (as the code comment suggests)

3. Test `sgemm_batch_fp32` with 3-matrix batch (for LoRA backward)

**Success criteria:** All 4 transpose combinations match CPU reference within 1e-5.

---

### Phase 2: FC Layer Backward on NPU (2-3 days)

**Goal:** Make `FullyConnectedLayer::calcDerivative` and `calcGradient` dispatch to NPU.

**Why second:** FC is the simplest backward (one GEMM each) and the biggest compute. If this works, the rest is composition.

**Tasks:**

1. **Verify `calcDerivative` dispatches to NPU:**
   - `calcDerivative` calls `ret_.dot_deriv_wrt_1(weight, derivative_, false, false)`
   - This calls `sgemm_fp32(TransB=true)` under the hood
   - With `compute_engine == CDSP`, the HexagonComputeOps should dispatch to the bridge
   - Add a log in `hexagon_compute_ops.cpp:sgemm_fp32` to confirm dispatch

2. **Verify `calcGradient` dispatches to NPU:**
   - `calcGradient` calls `input_.dot_deriv_wrt_2(djdw, derivative_, false, false)`
   - This calls `sgemm_fp32(TransA=true)` under the hood
   - For frozen weights, `calcGradient` is skipped (LoRA mode) — verify this

3. **Add batch mode for backward:**
   - In `causal_lm.cpp`, wrap the backward pass in `begin_batch()`/`end_batch()`
   - The smart sync guard already skips flush before CDSP layers
   - One flush per backward pass (or per block, if checkpointing)

4. **Test:** Run LoRA training on Qwen3-0.6B, compare loss curve with CPU-only backward.

**Code changes:**
- `causal_lm.cpp`: Add `begin_batch()`/`end_batch()` around backward pass
- `hexagon_compute_ops.cpp`: Add diagnostic logging to `sgemm_fp32` for backward dispatches
- No new bridge functions needed

**Success criteria:** FC backward runs on NPU, loss matches CPU within tolerance.

---

### Phase 3: Attention Backward on NPU (3-5 days)

**Goal:** Compute dQ, dK, dV on NPU using decomposed GEMMs + CPU softmax backward.

**Why third:** Attention is the most complex backward. Decompose into GEMMs (reuse Phase 1) + element-wise (CPU).

**Tasks:**

1. **Save forward intermediates:**
   - During forward, save Q, K, V, and attention probs (or recompute from Q, K)
   - Store in rpcmem (they're already there if all-rpcmem is done)
   - If memory-constrained, save only Q, K, V and recompute probs during backward

2. **Implement attention backward in `mha_core.cpp`:**
   ```cpp
   void MHACoreLayer::backward_incremental_forwarding(
     RunLayerContext &context, unsigned int from, unsigned int to) {
     
     // Get saved forward tensors
     Tensor &Q = context.getTensor(q_idx);      // [M, n_heads_Q, head_dim]
     Tensor &K = context.getTensor(k_idx);      // [M, n_heads_KV, head_dim]
     Tensor &V = context.getTensor(v_idx);      // [M, n_heads_KV, head_dim]
     Tensor &dY = context.getIncomingDerivative(0);  // [M, n_heads_Q, head_dim]
     
     // 1. dV = probs^T · dY  (per head)
     //    For each head: dV[h] = probs[h]^T · dY[h]
     //    probs[h] is [M, M], dY[h] is [M, d], dV[h] is [M, d]
     //    → sgemm_fp32(probs, dY, dV, TransA=true)
     
     // 2. dprobs = dY · V^T  (per head)
     //    → sgemm_fp32(dY, V, dprobs, TransB=true)
     
     // 3. dscores = softmax_backward(dprobs, probs)  (CPU)
     //    dscores = probs * (dprobs - sum(dprobs * probs, axis=-1, keepdim))
     
     // 4. dQ = dscores · K  (per head)
     //    → sgemm_fp32(dscores, K, dQ)
     
     // 5. dK = dscores^T · Q  (per head)
     //    → sgemm_fp32(dscores, Q, dK, TransA=true)
     
     // 6. RoPE backward: dQ = rotate(dQ, pos), dK = rotate(dK, pos)
     //    → nntr_htp_bridge_rope(dQ, positions, ...)
     //    → nntr_htp_bridge_rope(dK, positions, ...)
   }
   ```

3. **Softmax backward on CPU:**
   ```cpp
   // For each head h:
   // dprobs[h] is [M, M], probs[h] is [M, M]
   // dscores[h] = probs[h] * (dprobs[h] - sum(dprobs[h] * probs[h], axis=1, keepdim))
   for (int h = 0; h < n_heads; h++) {
     // Extract head slice
     // Compute on CPU (element-wise, cheap)
   }
   ```

4. **Handle GQA (grouped query attention):**
   - Qwen3 has 16 query heads, 8 KV heads (GQA factor 2)
   - dK and dV are shared across 2 query heads each
   - Accumulate: `dK[kv_head] += dK_from_q_head_0 + dK_from_q_head_1`

5. **Test:** Compare dQ, dK, dV with CPU reference (max abs error < 1e-4).

**Code changes:**
- `mha_core.cpp`: Add `backward_incremental_forwarding` method
- `mha_core.h`: Declare the method, add tensor indices for saved Q/K/V
- `causal_lm.cpp`: Call backward during training step

**Success criteria:** Attention backward matches CPU reference, runs on NPU for GEMMs.

---

### Phase 4: RMSNorm and SwiGLU Backward (2-3 days)

**Goal:** Backward for element-wise ops. CPU-first, NPU if beneficial.

**Tasks:**

1. **RMSNorm backward (CPU):**
   ```cpp
   void RMSNormLayer::calcDerivative(RunLayerContext &context) {
     // Forward: y = x / rms(x) * gamma
     // Backward: dx = gamma * (1/rms) * (dy - x * sum(x * dy * gamma) / (rms^2 * W))
     //
     // Steps:
     // 1. rms = sqrt(sum(x^2, axis=-1) / W + eps)  — recompute from saved x
     // 2. x_dy_gamma = x * dy * gamma  — element-wise
     // 3. correction = sum(x_dy_gamma, axis=-1) / (rms^2 * W)  — reduction
     // 4. dx = (dy - x * correction) * gamma / rms  — element-wise
     //
     // All element-wise, memory-bound. CPU is fine.
   }
   ```

2. **SwiGLU backward (CPU for element-wise, NPU for GEMMs):**
   ```cpp
   // Forward: y = down(silu(gate(x)) * up(x))
   // Backward:
   // 1. d_down = dY  (pass through)
   // 2. d_inter = dY · W_down^T  → sgemm_fp32(TransB=true)  [NPU]
   // 3. d_silu_gate = d_inter * up_val * silu'(gate_val)  [CPU]
   // 4. d_up = d_inter * silu(gate_val)  [CPU]
   // 5. dX = d_silu_gate · W_gate^T + d_up · W_up^T  → 2× sgemm_fp32(TransB=true)  [NPU]
   ```

3. **Residual ADD backward (trivial):**
   ```cpp
   void AdditionLayer::calcDerivative(RunLayerContext &context) {
     // dA = dY, dB = dY — just copy the incoming derivative to both outputs
     for (int i = 0; i < context.getNumInputs(); i++) {
       context.getOutgoingDerivative(i).copy(
         context.getIncomingDerivative(SINGLE_INOUT_IDX));
     }
   }
   ```
   Already implemented in nntrainer. No changes needed.

**Code changes:**
- `rms_norm.cpp`: Implement `calcDerivative` (currently throws "not supported")
- `reshaped_rms_norm.cpp`: Implement `calcDerivative` (currently throws)
- FFN/SwiGLU: Implement backward in `mha_core.cpp` or a new FFN layer

**Success criteria:** Element-wise backward matches CPU reference.

---

### Phase 5: LoRA Backward on NPU (2-3 days)

**Goal:** Compute LoRA weight gradients (djdA, djdB) on NPU.

**Tasks:**

1. **LoRA backward GEMMs:**
   ```cpp
   // For each LoRA layer:
   // djdB = loraTmp^T · dY          → sgemm_fp32(TransA=true)  [NPU]
   // djdtmp = dY · loraB^T           → sgemm_fp32(TransB=true)  [NPU]
   // djdA = X^T · djdtmp             → sgemm_fp32(TransA=true)  [NPU]
   // dX = djdtmp · loraA^T           → sgemm_fp32(TransB=true)  [NPU]
   ```

2. **Batch the 4 LoRA GEMMs:**
   - Use `sgemm_batch_fp32` to submit all 4 in one dispatch
   - Or enqueue them individually in batch mode (1 flush)

3. **Gradient accumulation:**
   - LoRA gradients accumulate across micro-batches
   - The `isGradientFirstAccess` flag controls zero vs. accumulate
   - The bridge's `sgemm_fp32` uses `beta=0` (overwrite) — for accumulation, need `beta=1`
   - **Issue:** The bridge only supports `alpha=1, beta=0`. For gradient accumulation, either:
     - (a) Add `beta` parameter to the bridge (small change)
     - (b) Accumulate on CPU after NPU computes the gradient (extra copy)
     - (c) Use `sgemm_fp32` with beta=0, then `HTP_OP_ADD` to accumulate

4. **Test:** Compare LoRA gradients with CPU reference.

**Code changes:**
- `fc_layer.cpp`: Verify `calcGradient` LoRA path dispatches to NPU
- `nntr-htp-bridge.cpp`: Add `beta` support to `sgemm_fp32` (if option (a))
- Or: `hexagon_compute_ops.cpp`: Handle accumulation in the C++ layer

**Success criteria:** LoRA gradients match CPU reference, accumulate correctly across micro-batches.

---

### Phase 6: Activation Checkpointing (3-5 days)

**Goal:** Reduce memory by discarding forward activations and recomputing during backward.

**Why last:** Only needed when memory is tight. For Qwen3-0.6B at 650 tokens, all activations fit in rpcmem (~70 MB). Checkpointing is needed for larger models or longer sequences.

**Tasks:**

1. **Add `TensorLifespan::CHECKPOINTED`:**
   - In `tensor_wrap_specs.h`, add new lifespan value
   - Tensors with this lifespan are freed after forward, recomputed during backward

2. **Mark checkpoint boundaries:**
   - In `causal_lm.cpp`, mark every Nth layer as a checkpoint boundary
   - Store only the checkpoint layer's input activation
   - Discard intermediate layers' activations

3. **Recompute during backward:**
   - When backward reaches a non-checkpointed layer, re-run its forward
   - The forward re-executes on NPU (enqueue + flush, ~0.1ms per layer)
   - With all-rpcmem, re-execution is cheap (no data movement)

4. **Test:** Verify gradients match non-checkpointed version.

**Code changes:**
- `tensor_wrap_specs.h`: Add `CHECKPOINTED` lifespan
- `tensor_pool.cpp`: Handle checkpointed tensor lifecycle
- `network_graph.cpp`: Add recompute logic in backward pass
- `causal_lm.cpp`: Mark checkpoint boundaries

**Success criteria:** Memory reduced by ~N× with N checkpoint intervals, gradients unchanged.

---

## 4. Dependency Graph

```
Phase 1 (GEMM validation)
    │
    ├── Phase 2 (FC backward) ──────────────┐
    │                                       │
    ├── Phase 3 (Attention backward) ───────┤
    │                                       │
    └── Phase 5 (LoRA backward) ────────────┤
                                            │
    Phase 4 (Element-wise backward) ────────┤
                                            │
                                    Phase 6 (Checkpointing)
```

- Phases 2, 3, 4, 5 can proceed in parallel after Phase 1
- Phase 6 depends on all others being functional

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `sgemm_fp32` transpose bug | High | Blocks everything | Phase 1 validates first; if broken, fix in bridge |
| Attention backward correctness | Medium | Wrong gradients | Compare with CPU reference per-head |
| Gradient accumulation (beta≠0) | Medium | Wrong LoRA gradients | Add beta to bridge, or accumulate on CPU |
| rpcmem exhaustion | Low (0.6B) | OOM crash | Phase 6 (checkpointing) for larger models |
| Sync guard during backward | Low | Stale data | Guard already works for forward; backward uses same path |
| Softmax backward on CPU | Low | Correctness | Standard formula, well-tested |
| GQA gradient accumulation | Medium | Wrong dK, dV | Accumulate across query heads sharing a KV head |

---

## 6. Testing Strategy

### Unit Tests
1. `test_sgemm_backward.cpp` — Transpose correctness for all 4 combinations
2. `test_attention_backward.cpp` — dQ, dK, dV vs CPU reference
3. `test_rmsnorm_backward.cpp` — dx vs CPU reference
4. `test_lora_backward.cpp` — djdA, djdB vs CPU reference

### Integration Test
- Run LoRA training on Qwen3-0.6B with NPU backward
- Compare loss curve with CPU-only backward
- Loss should match within 1e-3 relative error
- Training should converge to same loss

### Performance Benchmark
- Measure backward time: NPU vs CPU
- Expected: NPU backward ~2-3× faster (GEMMs on HMX)
- Measure total training step: forward + backward + optimizer
- Expected: NPU training ~1.5-2× faster end-to-end

---

## 7. Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: GEMM validation | 1-2 days | None |
| Phase 2: FC backward | 2-3 days | Phase 1 |
| Phase 3: Attention backward | 3-5 days | Phase 1 |
| Phase 4: Element-wise backward | 2-3 days | None (parallel) |
| Phase 5: LoRA backward | 2-3 days | Phase 1 |
| Phase 6: Checkpointing | 3-5 days | Phases 2-5 |
| **Total** | **13-21 days** | |

Critical path: Phase 1 → Phase 2 → Phase 3 → Phase 6 = ~9-15 days

---

## 8. What This Plan Does NOT Require

- **No new DSP kernels** — all backward ops reuse existing HTP ops (MUL_MAT, ROPE, ADD, etc.)
- **No new bridge functions** — `sgemm_fp32`, `rope`, `add` already exist with the right signatures
- **No changes to the enqueue/flush mechanism** — batch mode works the same for backward
- **No changes to the sync guard** — it already skips flush before CDSP layers
- **No changes to rpcmem allocation** — backward uses the same tensor pools as forward

The entire plan is: (1) validate the existing GEMM bridge works for transposed cases, (2) wire up the backward call paths in the layer code, (3) compose attention backward from existing GEMMs + CPU softmax. The infrastructure is already there.

---

Signed-off-by: Cline SR <noreply@anthropic.com>
