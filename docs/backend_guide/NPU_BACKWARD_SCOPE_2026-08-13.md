# NPU Backward Scope: Only What's Already on NPU

**Date:** 2026-08-13  
**Scope:** Add NPU backward support only for layers that currently dispatch to NPU in forward.  
**CPU-only layers (RMSNorm, SwiGLU, embedding, etc.) are out of scope for now.**

---

## Layers Currently on NPU (Forward) → Their Backward Status

### 1. `fully_connected` (Q/K/V/O/Down/LM-Head projections)

| | Forward | Backward |
|---|---------|----------|
| NPU GEMM | ✅ Q4_0 GEMM via `engine=cdsp` | ✅ **Already works** — FP32 GEMM via `engine=cdsp` |
| What happens | `Y = X · W` on DSP | `dX = dY · W^T` and `dW = X^T · dY` both dispatch to `sgemm_fp32` on DSP |
| Code path | `HexagonComputeOps::gemm_q4_0` | `HexagonComputeOps::sgemm_fp32` (transA/transB flags) |
| Status | Done | **DONE — no work needed** |

**Proof:** MNIST training benchmarks show FC backward GEMMs dispatching to NPU with 2-4× speedup and 100% accuracy.

---

### 2. `gate_up_layer` (Gate + Up FFN projections, batched)

| | Forward | Backward |
|---|---------|----------|
| NPU GEMM | ✅ Two Q4_0 GEMMs batched in 1 flush via `engine=cdsp` | ✅ **IMPLEMENTED** — `calcDerivative` and `calcGradient` now dispatch GEMMs to NPU |
| What happens | `gate = X · W_gate` and `up = X · W_up` in one DSP flush | `dX = dY_gate · W_gate^T + dY_up · W_up^T` (beta=1.0 accumulation) and `dW_gate = X^T · dY_gate`, `dW_up = X^T · dY_up` — all GEMMs → NPU via `dot_deriv_wrt_1/2` → `sgemm_fp32` |
| What's needed | — | ✅ Done — `supportBackwarding()=true`, `calcDerivative()` uses `dot_deriv_wrt_1` with beta=1.0 for accumulation, `calcGradient()` uses `dot_deriv_wrt_2` with gradient-first-access check |
| Effort | — | **DONE** (committed 2026-08-13) |


**Key point:** Once `gate_up_layer` has `supportBackwarding()=true` and calls the standard FC backward logic, the GEMMs automatically go to NPU. No new bridge code needed.

---

### 3. `mha_core` (Multi-Head Attention — flash attention path)

| | Forward | Backward |
|---|---------|----------|
| NPU GEMM | ✅ `nntr_htp_bridge_flash_attn` — fused QK^T + softmax + AV on DSP | ❌ **Not implemented** — `calcDerivative()` and `calcGradient()` are empty stubs |
| What happens | `A = softmax(Q·K^T / √d) · V` — all on DSP in one fused kernel | Needs: `dQ = dA · V^T`, `dK = dA^T · Q`, `dV = Q · dA^T` (3 GEMMs → NPU) + softmax backward + RoPE backward (CPU) |
| What's needed | — | Implement backward. The 3 GEMMs call `sgemm_fp32` or `sgemm_batch_fp32` → auto-dispatch to NPU. Softmax bw and RoPE bw are CPU element-wise. |
| Effort | — | **~1-2 weeks** (200 lines: 3 GEMM calls + softmax backward + RoPE backward + cache gradient handling) |

**What goes to NPU in backward:**
- `dQ = dA · V^T` → `sgemm_fp32` → NPU
- `dK = dA^T · Q` → `sgemm_fp32` → NPU  
- `dV = Q · dA^T` → `sgemm_fp32` → NPU
- (Optionally batch all 3 into one `sgemm_batch_fp32` flush)

**What stays CPU:**
- Softmax backward: `dA = (dOutput - sum(dOutput * A, axis=-1)) * A` — element-wise
- RoPE backward: inverse rotation — element-wise

---

### 4. `fused_ffn` (Fused FFN with SwiGLU — alternative path)

| | Forward | Backward |
|---|---------|----------|
| NPU GEMM | ✅ `nntr_htp_bridge_ffn_swiglu` — gate+up+SwiGLU+down all fused in 1 DSP flush | ❌ **Not implemented** — `calcDerivative()` and `calcGradient()` are empty |
| What happens | `gate = X·W_gate`, `up = X·W_up`, `act = SwiGLU(gate, up)`, `out = act · W_down` — all on DSP | Needs: backward through all 3 GEMMs → NPU, plus SwiGLU backward → CPU |
| What's needed | — | Implement backward. 3 GEMMs call `sgemm_fp32` → NPU. SwiGLU backward is CPU element-wise. |
| Effort | — | **~2-3 days** (3 GEMM calls + SwiGLU backward) |

**Note:** Qwen3 uses `gate_up_layer` + `swiglu` + `fully_connected` (separate layers), not `fused_ffn`. So this is only needed if you switch to the fused path. **Lower priority for Qwen3.**

---

## Summary: What to Implement (NPU backward only)

| # | Layer | Forward NPU | Backward NPU | Effort | How |
|---|-------|-------------|-------------|--------|-----|
| 1 | `fully_connected` | ✅ Done | ✅ **Already done** | 0 | Auto-dispatches via `engine=cdsp` |
| 2 | `gate_up_layer` | ✅ Done | ✅ **DONE** | Done | `calcDerivative`/`calcGradient` implemented — GEMMs auto-dispatch to NPU |
| 3 | `mha_core` | ✅ Done | ❌ **Need to add** | ~1-2 weeks | Implement backward — 3 GEMMs → `sgemm_fp32` → NPU, softmax/RoPE bw → CPU |
| 4 | `fused_ffn` | ✅ Done | ❌ **Need to add** | ~2-3 days | Implement backward — 3 GEMMs → NPU, SwiGLU bw → CPU (lower priority for Qwen3) |

**Remaining effort for item 3 (Qwen3 path): ~1-2 weeks**


### What you DON'T need to implement (out of scope):

| Layer | Why it's out of scope |
|-------|----------------------|
| `rms_norm` / `reshaped_rms_norm` | CPU-only (element-wise), backward not on NPU |
| `swiglu` | CPU-only (element-wise), backward not on NPU |
| `embedding_layer` / `tie_word_embeddings` | CPU-only (lookup), backward not on NPU |
| `addition` (residual) | CPU-only (element-wise), already has pass-through backward |
| `lm_head` | CPU-only (or could use FC), backward not on NPU |

**Note:** These CPU layers still need *some* backward implementation (even if stub/identity) for the training graph to execute. But the NPU GEMM backward work is only items 2 and 3 above.

---

## The GEMMs That Will Run on NPU in Backward (per transformer block)

```
BACKWARD (28 blocks × these GEMMs = 756 NPU GEMMs per training step)

  1. dX_down  = dY · W_down^T     → sgemm_fp32 → NPU
  2. dW_down  = X^T · dY           → sgemm_fp32 → NPU
  3. dX_gate  = dY · W_gate^T      → sgemm_fp32 → NPU  (batched with dX_up)
  4. dX_up    = dY · W_up^T        → sgemm_fp32 → NPU  (batched with dX_gate)
  5. dW_gate  = X^T · dY_gate      → sgemm_fp32 → NPU  (batched with dW_up)
  6. dW_up    = X^T · dY_up        → sgemm_fp32 → NPU  (batched with dW_gate)
  7. dX_wo    = dY · W_o^T         → sgemm_fp32 → NPU
  8. dW_wo    = X^T · dY           → sgemm_fp32 → NPU
  9. dQ       = dA · V^T            → sgemm_fp32 → NPU  (attention)
  10. dK      = dA^T · Q            → sgemm_fp32 → NPU  (attention)
  11. dV      = Q · dA^T            → sgemm_fp32 → NPU  (attention)
  12. dX_wq   = dY · W_q^T          → sgemm_fp32 → NPU
  13. dW_wq   = X^T · dY            → sgemm_fp32 → NPU
  14. dX_wk   = dY · W_k^T          → sgemm_fp32 → NPU
  15. dW_wk   = X^T · dY            → sgemm_fp32 → NPU
  16. dX_wv   = dY · W_v^T          → sgemm_fp32 → NPU
  17. dW_wv   = X^T · dY            → sgemm_fp32 → NPU
  18. dX_lmhead = dY · W_lm^T       → sgemm_fp32 → NPU  (only at output layer)
  19. dW_lmhead = X^T · dY          → sgemm_fp32 → NPU  (only at output layer)

  With batching (gate+up together, attention dQ/dK/dV together):
  → ~10-12 flushes per block instead of 19 individual GEMMs
  → 28 blocks × ~12 flushes = ~336 NPU flushes per training step
```

**Plus the forward GEMMs (already working):** ~9-11 GEMMs/block × 28 = ~280 NPU GEMMs

**Total NPU GEMMs per training step: ~600+** (forward + backward + weight gradients)

---

## What the Bridge Already Supports for Backward

| Bridge Function | Purpose | Status |
|---|---|---|
| `nntr_htp_bridge_sgemm_fp32` | Single FP32 GEMM (dX, dW, dQ, dK, dV) | ✅ Working |
| `nntr_htp_bridge_sgemm_batch_fp32` | Batched FP32 GEMMs (fuse dX+dW pairs) | ✅ Working |
| `nntr_htp_bridge_gemm_q4_0` | Q4_0 GEMM (forward inference) | ✅ Working |
| `nntr_htp_bridge_flash_attn` | Fused flash attention (forward only) | ✅ Working |
| Flash attention backward | Fused attn backward on DSP | ❌ Not implemented (optional — can use separate GEMMs instead) |

**The bridge already has everything needed for NPU backward.** The `sgemm_fp32` and `sgemm_batch_fp32` functions handle all transpose combinations (transA, transB) needed for dX = dY·W^T and dW = X^T·dY. The work is just in the layer `calcDerivative`/`calcGradient` implementations.

---

## Action Plan (NPU backward only)

### Step 1: `gate_up_layer` backward (~1 day)
```cpp
// In gate_up_layer.cpp:
// 1. Set supportBackwarding() = true
// 2. calcDerivative(): dX = dY_gate · W_gate^T + dY_up · W_up^T
//    → Calls FC calcDerivative for both, results summed
//    → GEMMs auto-dispatch to NPU via engine=cdsp
// 3. calcGradient(): dW_gate = X^T · dY_gate, dW_up = X^T · dY_up
//    → Calls FC calcGradient for both
//    → GEMMs auto-dispatch to NPU via engine=cdsp
```

### Step 2: `mha_core` backward (~1-2 weeks)
```cpp
// In mha_core.cpp:
// 1. calcDerivative(): 
//    a. softmax_backward(dA, attn_weights) → CPU
//    b. dQ = dA · V^T  → sgemm_fp32 → NPU
//    c. dK = dA^T · Q  → sgemm_fp32 → NPU
//    d. dV = Q · dA^T  → sgemm_fp32 → NPU
//    e. RoPE_backward → CPU
//    f. Concatenate dQ/dK/dV heads → pass to Q/K/V proj backward
//
// 2. calcGradient():
//    → No weights in mha_core itself (weights are in Q/K/V/O proj layers)
//    → But needs to handle KV cache gradients if training with cache
```

### Step 3: (Optional) `fused_ffn` backward (~2-3 days)
Only needed if using the fused FFN path instead of gate_up + swiglu + down.

### Step 4: Fix `transB=1` bridge bug (~2 days)
The `sgemm_fp32` with `transB=1` has high rel_err when called through the
nntrainer framework path. This must be fixed before framework-level training
works correctly. The standalone benchmark path is correct.
