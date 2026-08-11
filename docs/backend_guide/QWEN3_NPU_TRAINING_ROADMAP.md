# Training Actual Qwen3-0.6B on NPU — Gap Analysis & Roadmap

## Current State (What Works)

### Inference (Q4_0 quantized) — WORKING
- Qwen3-0.6B generates text end-to-end via CausalLM on device
- Q4_0 GEMMs dispatch to cDSP via `nntr_htp_bridge_gemm_q4_0`
- Batched Q4_0 GEMMs (`gemm_q4_0_batch_fp32`) fuse Q/K/V and gate/up
- Weights are uploaded once to persistent rpcmem arenas
- Flash attention bridge (`nntr_htp_bridge_flash_attn`) exists
- FFN SwiGLU bridge (`nntr_htp_bridge_ffn_swiglu`) exists

### Training (FP32) — WORKING for simple FC networks
- `nntr_htp_bridge_sgemm_fp32` dispatches individual FP32 GEMMs to cDSP
- `nntr_htp_bridge_sgemm_batch_fp32` batches independent GEMMs into 1 flush
- `HexagonComputeOps::sgemm_fp32` auto-dispatches (with CPU fallback)
- Verified on MNIST FC networks at Qwen3-scale dimensions (1024×3072)
- 4.9× speedup over naive CPU at Qwen3 scale

## What's Missing for Real Qwen3 Training

Qwen3-0.6B has: 28 transformer blocks, hidden=1024, intermediate=3072, 16 heads,
head_dim=64, vocab=151936. Each block has:

```
┌─ RMSNorm ─────────────────────────────────────┐
│  x_norm = x / sqrt(mean(x²) + eps) * gamma    │  ← CPU (element-wise)
└───────────────────────────────────────────────┘
┌─ Attention ──────────────────────────────────┐
│  Q = x_norm @ W_q   [B,seq,1024]              │  ← NPU GEMM ✅
│  K = x_norm @ W_k   [B,seq,128]  (GQA 16→4)  │  ← NPU GEMM ✅
│  V = x_norm @ W_v   [B,seq,128]              │  ← NPU GEMM ✅
│  RoPE(Q, K)                                 │  ← CPU (element-wise) ⚠️
│  scores = Q @ K^T * scale  [B,heads,seq,seq] │  ← NPU GEMM ✅ (batched BMM)
│  scores = softmax(scores + mask)              │  ← CPU (element-wise) ⚠️
│  O = scores @ V            [B,seq,1024]       │  ← NPU GEMM ✅ (batched BMM)
│  out = O @ W_o            [B,seq,1024]        │  ← NPU GEMM ✅
└───────────────────────────────────────────────┘
┌─ RMSNorm ─────────────────────────────────────┐
│  (same as above)                               │  ← CPU
└───────────────────────────────────────────────┘
┌─ FFN (SwiGLU) ────────────────────────────────┐
│  gate = x_norm @ W_gate  [B,seq,3072]          │  ← NPU GEMM ✅
│  up   = x_norm @ W_up    [B,seq,3072]          │  ← NPU GEMM ✅ (batched)
│  act  = SiLU(gate) * up                        │  ← CPU (element-wise) ⚠️
│  down = act @ W_down     [B,seq,1024]          │  ← NPU GEMM ✅
└───────────────────────────────────────────────┘
┌─ Residual ────────────────────────────────────┐
│  x = x + attn_out + ffn_out                    │  ← CPU (element-wise)
└───────────────────────────────────────────────┘
```

### Gap 1: Training Mode in CausalLM Layers (BLOCKER)

**Problem:** All CausalLM custom layers only implement `incremental_forwarding()`
(inference). None implement `backwarding()`. The nntrainer framework's training
loop calls `backwarding()` on each layer, which will throw "not supported".

| Layer | Forward (inference) | Backward (training) | Status |
|-------|--------------------|--------------------|--------|
| RMSNormLayer | ✅ incremental_forwarding | ❌ | NYI |
| MHACoreLayer | ✅ incremental_forwarding | ❌ | NYI |
| SwiGLULayer | ✅ incremental_forwarding | ❌ | NYI |
| FusedFFNLayer | ✅ forwarding | ❌ | NYI |
| EmbeddingLayer | ✅ forwarding | ❌ | NYI |
| TieWordEmbedding | ✅ incremental_forwarding | ❌ | NYI |
| LmHeadLayer | ✅ incremental_forwarding | ❌ | NYI |

**Fix:** Implement `backwarding()` for each layer. This is the biggest piece of
work. Each layer needs:
- Forward: save activations (already done for inference)
- Backward: compute gradients w.r.t. input and weights
- The GEMMs in backward (dX = dY·W^T, dW = X^T·dY) will automatically dispatch
  to NPU via `HexagonComputeOps::sgemm_fp32`

**Effort:** ~2-3 weeks. The backward math is standard; the framework integration
is the bulk (gradient tensor management, RunLayerContext wiring).

### Gap 2: FP32 Weight Storage (BLOCKER)

**Problem:** Inference uses Q4_0 quantized weights. Training requires FP32
weights (you can't backpropagate through Q4_0 dequantization easily). The model
loader loads Q4_0 `.bin` files; training needs FP32 `.safetensors`.

**Fix:** 
- Load model in FP32 (safetensors format is already supported by the loader)
- Set `engine=cdsp` on FC layers so GEMMs dispatch to NPU via `sgemm_fp32`
- The Q4_0 weight upload path (`nntr_htp_bridge_upload_weight_q4x4x2`) is
  irrelevant for training — `sgemm_fp32` stages both A and B as FP32 per call

**Effort:** ~2-3 days. Config change + verify safetensors FP32 loading works
with cdsp engine.

### Gap 3: Attention Backward (BLOCKER)

**Problem:** Attention backward (dQ, dK, dV, dO gradients) is complex:
- `dQ = (dO @ V^T) * scale` → GEMM (NPU can do)
- `dK = Q^T @ dO * scale` → GEMM (NPU can do)
- `dV = O_scores^T @ dO` → GEMM (NPU can do)
- `dO_scores = softmax_backward(dO, scores)` → element-wise (CPU)
- `dO = dO_scores @ V` → GEMM (NPU can do)
- RoPE backward → element-wise (CPU)

The bridge has `nntr_htp_bridge_flash_attn` for **forward only**. No backward
flash attention kernel exists.

**Fix:** Implement attention backward using individual `sgemm_fp32` calls for
the GEMM parts, CPU for softmax/RoPE backward. The flash-attn forward kernel
is NOT needed for training — standard attention with explicit Q@K^T and
softmax@V is fine (and easier to differentiate).

**Effort:** ~1 week. Standard transformer attention backward, well-documented.

### Gap 4: Memory Budget (RISK)

**Problem:** FP32 Qwen3-0.6B training needs:
- Weights: 28 blocks × 7 matrices × ~1-3M params × 4 bytes ≈ 600 MB
- Activations: 28 blocks × seq_len × 1024 × 4 bytes (forward cache for backward)
- Gradients: same as weights ≈ 600 MB
- Adam state: 2× weights ≈ 1.2 GB
- **Total: ~2.5 GB** for seq_len=128

Galaxy S24 Ultra has ~12 GB RAM, but Android apps typically get ~2-4 GB.
Training might OOM at longer sequences.

**Mitigations:**
- Gradient checkpointing (recompute forward in backward) — halves activation memory
- LoRA/QLoRA (train only adapter weights) — 10-50× less memory
- Mixed precision (FP16 activations, FP32 master weights) — halves activation memory

### Gap 5: Forward Fusion Bug (OPTIMIZATION)

**Problem:** `fused_fc_forward` (3 GEMMs + 2 ReLUs in 1 DSP flush) has an
op-chaining race condition. Without it, forward does 3 individual flushes.

**Impact:** Not a blocker — individual `sgemm_fp32` calls work correctly. But
fixing this would reduce flushes from 6→4 per transformer block per direction,
~30% faster training.

**Fix:** The DSP op graph needs dependency edges between ops in the same batch.
Currently `sgemm_batch_fp32` assumes all GEMMs are independent. For fused
forward, need to add `HTP_OP_RELU` support to the bridge and wire op-to-op
dependencies in the enqueue path.

**Effort:** ~1 week.

### Gap 6: Element-wise Ops on DSP (OPTIMIZATION)

**Problem:** RMSNorm, RoPE, softmax, SwiGLU, residual add all run on CPU.
For each, there's a CPU↔DSP handoff before and after.

**Impact:** For Qwen3 scale, the GEMMs dominate (~95% of FLOPs), so element-wise
ops on CPU are not a major bottleneck. But each handoff costs ~600 µs of
FastRPC overhead. With ~12 element-wise ops per block × 28 blocks = 336 extra
handoffs per step.

**Fix:** Implement `HTP_OP_UNARY` dispatch in the bridge for:
- `HTP_OP_RMS_NORM` — bridge calls `enqueue_op` with norm descriptor
- `HTP_OP_ROPE` — rotation on DSP
- `HTP_OP_SOFTMAX` — along last dim
- `HTP_OP_SWIGLU` — fused SiLU + multiply
- `HTP_OP_ADD` — residual

The DSP's `unary-ops.c` already has some of these for inference.

**Effort:** ~2 weeks. Not a blocker but would significantly reduce handoff count.

## Implementation Roadmap

### Phase 1: Get Training Working (CPU GEMMs) — 2-3 weeks
1. Load Qwen3-0.6B in FP32 (safetensors)
2. Implement `backwarding()` for all CausalLM custom layers:
   - RMSNormLayer (simple: dgamma, dx)
   - SwiGLULayer (simple: SiLU backward + element-wise)
   - FusedFFNLayer (3 GEMMs backward = 6 GEMMs, auto-dispatches to CPU)
   - MHACoreLayer (attention backward — most complex)
   - EmbeddingLayer / TieWordEmbedding (embedding lookup backward)
   - LmHeadLayer (GEMM backward)
3. Verify training converges on CPU (sanity check with tiny dataset)
4. Add `loss = CrossEntropy` after LmHead

### Phase 2: NPU Dispatch — 1 week
1. Set `engine=cdsp` on all FC layers in the model config
2. `HexagonComputeOps::sgemm_fp32` auto-dispatches all training GEMMs to NPU
3. Element-wise ops stay on CPU (auto-forwarded via `get_cpu_ops()`)
4. Verify NPU training matches CPU training (same loss curve)
5. Benchmark NPU vs CPU training speed

### Phase 3: Optimization — 2-3 weeks
1. Fix `fused_fc_forward` op-chaining bug (forward fusion)
2. Implement `sgemm_batch_fp32` in `HexagonComputeOps` for backward fusion
   (currently only the standalone bench uses it; nntrainer's FC layer calls
   individual `sgemm_fp32`)
3. Add DSP unary ops for RMSNorm/softmax/SwiGLU (reduce handoffs)
4. Gradient checkpointing for memory

### Phase 4: Fine-tuning — 1 week
1. LoRA adapter layers (train only rank-8/16 adapters)
2. QLoRA: Q4_0 frozen weights + FP32 LoRA adapters
3. This reduces memory from ~2.5 GB to ~200 MB

## Expected Performance (Phase 2, NPU training)

Qwen3-0.6B, 28 layers, batch=1, seq=128, FP32:

| Component | GEMMs/block | Flushes/block | Est. time/block |
|-----------|-------------|---------------|-----------------|
| Q/K/V/O proj fwd | 4 (3 batched + 1) | 2 | ~2 ms |
| QK^T + softmax + AV fwd | 2 BMM | 2 | ~1 ms |
| Gate/Up/Down fwd | 3 (2 batched + 1) | 2 | ~3 ms |
| **Forward total/block** | 9 | 6 | **~6 ms** |
| Backward (2× forward GEMMs) | 18 | ~12 (batched) | **~12 ms** |
| **Total/block** | 27 | 18 | **~18 ms** |
| **Total/step (28 blocks)** | 756 | 504 | **~504 ms** |

With forward fusion (Phase 3): ~350 ms/step.
With LoRA (Phase 4): ~200 ms/step (fewer trainable GEMMs).

## Architecture: How NPU Training Would Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    nntrainer model->train()                       │
│                                                                   │
│  for each transformer block (×28):                                │
│    ┌─ RMSNorm ─────────────────────────────────────────────────┐  │
│    │  CPU: x_norm = rms_norm(x, gamma)                         │  │
│    └──────────────────────────────────────────────────────────┘  │
│    ┌─ Attention ───────────────────────────────────────────────┐  │
│    │  NPU: Q = x_norm @ W_q    (sgemm_fp32 → cDSP)            │  │
│    │  NPU: K = x_norm @ W_k    (sgemm_fp32 → cDSP)            │  │
│    │  NPU: V = x_norm @ W_v    (sgemm_fp32 → cDSP)            │  │
│    │  CPU: RoPE(Q, K)                                          │  │
│    │  NPU: scores = Q @ K^T    (sgemm_fp32 → cDSP)            │  │
│    │  CPU: softmax(scores + mask)                              │  │
│    │  NPU: O = scores @ V      (sgemm_fp32 → cDSP)            │  │
│    │  NPU: out = O @ W_o       (sgemm_fp32 → cDSP)            │  │
│    └──────────────────────────────────────────────────────────┘  │
│    ┌─ RMSNorm + Residual ─────────────────────────────────────┐  │
│    │  CPU: x = x + attn_out; x_norm = rms_norm(x, gamma)      │  │
│    └──────────────────────────────────────────────────────────┘  │
│    ┌─ FFN ─────────────────────────────────────────────────────┐  │
│    │  NPU: gate = x_norm @ W_gate (sgemm_fp32 → cDSP)         │  │
│    │  NPU: up   = x_norm @ W_up   (sgemm_fp32 → cDSP)         │  │
│    │  CPU: act = SiLU(gate) * up                               │  │
│    │  NPU: down = act @ W_down    (sgemm_fp32 → cDSP)         │  │
│    └──────────────────────────────────────────────────────────┘  │
│    ┌─ Residual ────────────────────────────────────────────────┐  │
│    │  CPU: x = x + ffn_out                                     │  │
│    └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Backward: same structure in reverse, all GEMMs → NPU            │
│  Adam: CPU                                                       │
└──────────────────────────────────────────────────────────────────┘
         │ sgemm_fp32 calls
         ▼
┌──────────────────────────────────────────────────────────────────┐
│              libggml-hexagon.so (bridge)                          │
│                                                                   │
│  nntr_htp_bridge_sgemm_fp32()      ← per-GEMM dispatch            │
│  nntr_htp_bridge_sgemm_batch_fp32() ← batched dispatch            │
│                                                                   │
│  1. stage A/B into rpcmem                                       │
│  2. enqueue_op(HTP_OP_MUL_MAT)                                  │
│  3. flush(true) → FastRPC to cDSP                               │
│  4. copy result from rpcmem                                     │
└──────────────────────────────────────────────────────────────────┘
         │ FastRPC
         ▼
┌──────────────────────────────────────────────────────────────────┐
│              Hexagon cDSP (HTP v75)                               │
│                                                                   │
│  HMX Matrix Engine (FP32 MUL_MAT)                                │
│  ~504 GEMM dispatches per training step                           │
└──────────────────────────────────────────────────────────────────┘
```

## Summary

| Gap | Severity | Effort | Blocks Training? |
|-----|----------|--------|-----------------|
| Layer backwarding () | BLOCKER | 2-3 weeks | Yes |
| FP32 weight loading | BLOCKER | 2-3 days | Yes |
| Attention backward | BLOCKER | 1 week | Yes |
| Memory budget | RISK | 1 week (checkpointing) | Maybe (OOM) |
| Forward fusion bug | OPTIMIZATION | 1 week | No (30% slower) |
| Element-wise on DSP | OPTIMIZATION | 2 weeks | No (more handoffs) |

**Bottom line:** The NPU GEMM infrastructure is ready — `sgemm_fp32` and
`sgemm_batch_fp32` work correctly and deliver 5× speedup at Qwen3 scale. The
missing piece is **layer-level backwarding implementations** in the CausalLM
code. Once those are written (standard transformer math), all GEMMs in both
forward and backward will automatically dispatch to the NPU through
`HexagonComputeOps::sgemm_fp32`. No bridge changes needed for Phase 2.
