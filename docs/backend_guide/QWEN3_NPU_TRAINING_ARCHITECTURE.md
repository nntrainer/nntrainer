# Qwen3-0.6B NPU Training — Architectural Plan

## 1. Current Inference Architecture (What's Fused, What's on NPU)

### 1.1 Per-Decoder-Layer Operation Breakdown

Each of Qwen3-0.6B's 28 decoder layers performs:

```
Input h
  │
  ├─ RMSNorm (attn_norm)          [CPU]  — elementwise FP32
  ├─ Q/K/V Projection             [NPU]  — 3× FC with engine=cdsp → sgemm_fp32/gemm_q4_0
  ├─ RoPE                         [CPU]  — elementwise rotation
  ├─ Attention (QK^T, softmax, AV) [NPU for prefill≥160 tokens, CPU otherwise]
  │                                       flash_attn bridge → hmx_flash_attn_ext
  ├─ O Projection                  [NPU]  — FC with engine=cdsp
  ├─ Residual Add                 [CPU]  — elementwise
  ├─ RMSNorm (ffn_norm)           [CPU]  — elementwise FP32
  ├─ FFN (gate+up+SwiGLU+down)   [NPU if NNTR_HEXAGON_FUSED_FFN=1, else NPU per-FC]
  │     ├─ Non-fused: gate_up_layer [NPU] + swiglu [CPU] + ffn_down [NPU]
  │     └─ Fused: fused_ffn [NPU] — single FastRPC call: 3 GEMMs + SwiGLU
  └─ Residual Add                 [CPU]  — elementwise
```

### 1.2 NPU vs CPU Summary (Inference)

| Operation | Where | Mechanism | Notes |
|-----------|-------|-----------|-------|
| Q/K/V/O projection GEMMs | **NPU** | `engine=cdsp` on FC layers → `HexagonComputeOps::sgemm_fp32` or `gemm_q4_0` | Q4_0 weights → HMX 4-bit units; FP32 weights → cDSP FP32 units |
| Flash attention (prefill) | **NPU** | `nntr_htp_bridge_flash_attn` → `hmx_flash_attn_ext` | Only when step_size ≥ 160 tokens and head_dim % 64 == 0 |
| Attention (decode) | **CPU** | `compute_kcaches` + `softmax_triangle` + `compute_vcache_transposed` | Single-token decode: O(seq) not worth FastRPC round-trip |
| RoPE | **CPU** | `apply_rotary_emb_tensor_v2` | Elementwise, too small for NPU |
| RMSNorm | **CPU** | Custom layer | Elementwise |
| SwiGLU | **CPU** | Custom layer | Elementwise (sigmoid × gate) |
| Residual add | **CPU** | `addition` layer | Elementwise |
| Embedding lookup | **CPU** | `embedding_layer` / `tie_word_embedding` | Gather, not GEMM |
| LM head (logits) | **CPU** | FC layer without engine=cdsp | Large vocab=151936, but only 1 row at decode time |
| Fused FFN (optional) | **NPU** | `nntr_htp_bridge_ffn_swiglu` | Single FastRPC for 3 GEMMs + SwiGLU; env: `NNTR_HEXAGON_FUSED_FFN=1` |

### 1.3 GEMM Count Per Layer (Inference)

| Path | GEMMs on NPU | GEMMs on CPU | Total FastRPC calls |
|------|-------------|-------------|-------------------|
| Prefill (flash_attn + fused_ffn) | 4 FC (Q,K,V,O) + 1 flash_attn + 1 fused_ffn | 0 | 6 |
| Prefill (flash_attn, no fused_ffn) | 4 FC + 1 flash_attn + 3 FC (gate,up,down) | 0 | 8 |
| Decode (no flash_attn, no fused_ffn) | 4 FC + 3 FC | attention (CPU) | 7 |
| Decode (fused_ffn) | 4 FC + 1 fused_ffn | attention (CPU) | 5 |

---

## 2. Training Architecture Plan (Forward + Backward)

### 2.1 Key Difference from Inference

Training requires **backward pass** (calcDerivative + calcGradient). The nntrainer-lora repo has a modified `fc_layer.cpp` that adds LoRA adapters and implements `calcDerivative`/`calcGradient` for LoRA. The current nntrainer repo (this one) has `supportBackwarding() = false` on `FusedFFNLayer` and no training path for `mha_core`.

### 2.2 Per-Step Training GEMM Breakdown

For one decoder layer, one training step (forward + backward):

#### Forward Pass (same as inference, but all activations cached for backward):

| # | Operation | Where | GEMM Shape (M×N×K) | Notes |
|---|-----------|-------|---------------------|-------|
| 1 | Q projection | NPU | seq×(n_heads×head_dim)×dim | FC with engine=cdsp |
| 2 | K projection | NPU | seq×(n_kv_heads×head_dim)×dim | FC with engine=cdsp |
| 3 | V projection | NPU | seq×(n_kv_heads×head_dim)×dim | FC with engine=cdsp |
| 4 | Attention (QK^T) | NPU (prefill) / CPU | seq×seq×head_dim | flash_attn or CPU |
| 5 | Attention (AV) | NPU (prefill) / CPU | seq×head_dim×seq | Part of flash_attn |
| 6 | O projection | NPU | seq×dim×(n_heads×head_dim) | FC with engine=cdsp |
| 7 | FFN gate | NPU | seq×inter_dim×dim | FC or fused_ffn |
| 8 | FFN up | NPU | seq×inter_dim×dim | FC or fused_ffn |
| 9 | FFN down | NPU | seq×dim×inter_dim | FC or fused_ffn |

#### Backward Pass (LoRA — base weights frozen, only LoRA A/B gradients):

| # | Operation | Where | GEMM Shape | Notes |
|---|-----------|-------|------------|-------|
| 10 | dX for O proj | NPU | dim×seq×dim | `dot_deriv_wrt_1(W^T, dY)` — dequantize W to FP32, add LoRA contrib |
| 11 | dW for O loraB | NPU | inter×seq×rank | `dot_deriv_wrt_2(loraTmp, dY)` — gradient of loraB |
| 12 | dX for O loraTmp | NPU | rank×seq×inter | `dot_deriv_wrt_1(loraB, dY)` — propagate through loraB |
| 13 | dW for O loraA | NPU | dim×seq×rank | `dot_deriv_wrt_2(input, dTmp)` — gradient of loraA |
| 14 | Attention backward | **CPU** | — | No NPU backward for flash_attn yet |
| 15 | dX for V proj | NPU | dim×seq×dim | Same pattern as #10-13 |
| 16-18 | dW/dX for V LoRA | NPU | | Same pattern as #11-13 |
| 19 | dX for K proj | NPU | | Same pattern |
| 20-22 | dW/dX for K LoRA | NPU | | Same pattern |
| 23 | dX for Q proj | NPU | | Same pattern |
| 24-26 | dW/dX for Q LoRA | NPU | | Same pattern |
| 27 | dX for FFN down | NPU | | Same pattern |
| 28-30 | dW/dX for FFN down LoRA | NPU | | Same pattern |
| 31 | dX for FFN up | NPU | | Same pattern |
| 32-34 | dW/dX for FFN up LoRA | NPU | | Same pattern |
| 35 | dX for FFN gate | NPU | | Same pattern |
| 36-38 | dW/dX for FFN gate LoRA | NPU | | Same pattern |

**Total per layer per training step: ~38 GEMMs** (vs 9 for MNIST)

### 2.3 NPU vs CPU Summary (Training)

| Operation | Forward | Backward | Notes |
|-----------|---------|----------|-------|
| Q/K/V/O projection | **NPU** (sgemm_fp32) | **NPU** (sgemm_fp32 for dX; LoRA grad GEMMs) | Base weight dequantized to FP32, LoRA contrib added |
| Attention | **NPU** (flash_attn, prefill) / **CPU** (decode) | **CPU** (NYI on NPU) | Flash attention backward not implemented on cDSP |
| FFN (gate/up/down) | **NPU** (sgemm_fp32 or fused) | **NPU** (sgemm_fp32 for dX; LoRA grad GEMMs) | Same as Q/K/V/O pattern |
| RoPE | **CPU** | **CPU** | Elementwise, trivial |
| RMSNorm | **CPU** | **CPU** | Elementwise |
| SwiGLU | **CPU** (or in fused_ffn) | **CPU** | Elementwise backward through sigmoid |
| Residual add | **CPU** | **CPU** | Elementwise |
| Embedding | **CPU** | **CPU** | Gather, not GEMM |
| LM head (logits + loss) | **CPU** | **CPU** | Large vocab, but only 1 row for training |
| Softmax + CE loss | **CPU** | **CPU** | Scalar operations |
| Adam optimizer | **CPU** | — | Elementwise update of LoRA A/B |

---

## 3. Migration Plan: calcDerivative/calcGradient from nntrainer-lora → this repo

### 3.1 What nntrainer-lora Has That This Repo Doesn't

The nntrainer-lora repo (`/home/anirudh/nntrainer-lora/nntrainer`) has a modified `fc_layer.cpp` with:

1. **LoRA weight registration** in `finalize()`: loraA [in_dim×rank], loraB [rank×unit], loraTmp, loraOut tensors
2. **LoRA forward** in `forwarding()`: `hidden = input.dot(W) + (input.dot(loraA)).dot(loraB) * scaling`
3. **LoRA calcDerivative**: dequantize base W to FP32, compute `W_eff = W + loraA.dot(loraB) * scaling`, then `dX = dY.dot(W_eff^T)`
4. **LoRA calcGradient**: 3 GEMMs — `dW_loraB = loraTmp^T.dot(dY)`, `dW_loraTmp = dY.dot(loraB^T)`, `dW_loraA = input^T.dot(dTmp)`
5. **QAT support**: fake-quantize loraA/loraB with straight-through estimator
6. **Training script**: `train_qwen3_lora.cpp` with dataset generator, thermal monitoring, epoch callbacks

### 3.2 Migration Steps

#### Step 1: Port fc_layer.cpp LoRA changes (1-2 days)

Copy the LoRA-specific code from nntrainer-lora's `fc_layer.cpp` into this repo's `fc_layer.cpp`:

- `LoraRank`, `LoraAlpha`, `LoraQAT`, `LoraWeightQ4` properties
- LoRA weight/tensor registration in `finalize()`
- LoRA forward in `forwarding()` and `incremental_forwarding()`
- LoRA `calcDerivative()` — dequantize + LoRA contrib
- LoRA `calcGradient()` — 3 GEMMs for loraA/loraB gradients
- QAT fake-quantize support

**Key concern:** The `engine=cdsp` FC layers currently dispatch to NPU via `HexagonComputeOps`. The LoRA forward adds 2 extra GEMMs (input·loraA, loraTmp·loraB) per FC layer. These are small (rank=8) so should stay on CPU. Only the base weight GEMM goes to NPU.

**Decision:** LoRA GEMMs (rank=8) stay on CPU. Only base weight GEMMs (frozen, dequantized to FP32) dispatch to NPU via `sgemm_fp32`. This matches the MNIST pattern where all 9 GEMMs go to NPU.

#### Step 2: Port training infrastructure (1 day)

- Copy `lora_train.h` / `lora_train.cpp` (TrainingDataGenerator)
- Copy `train_qwen3_lora.cpp` (CLI driver)
- Add `Qwen3CausalLM` class (extends Transformer with `initializeForTraining`, `train`, `save_weight_lora`)
- Add `lora_train` target to `Applications/CausalLM/meson.build`

#### Step 3: Enable training mode in Transformer (1-2 days)

- Change `model->compile(x, y, ml::train::ExecutionMode::INFERENCE)` to support `ExecutionMode::TRAIN`
- Add loss layer (cross-entropy) after LM head
- Add optimizer (AdamW) for LoRA parameters only
- Ensure `supportBackwarding()` returns true for FC layers with LoRA

#### Step 4: NPU dispatch for backward GEMMs (2-3 days)

The backward pass has 2 types of GEMMs:

**a) dX GEMMs** (`calcDerivative`): `dX = dY · W_eff^T` where `W_eff = W_dequantized + loraA·loraB·scaling`
- These are large GEMMs (seq×dim×dim) → dispatch to NPU via `sgemm_fp32`
- Need to modify `HexagonComputeOps` to handle the `dot_deriv_wrt_1` path
- Or: compute `W_eff` on CPU, then call `sgemm_fp32` directly

**b) LoRA gradient GEMMs** (`calcGradient`): `dW_loraB = loraTmp^T · dY`, `dW_loraTmp = dY · loraB^T`, `dW_loraA = input^T · dTmp`
- These are small GEMMs (rank=8 in one dimension) → keep on CPU
- Not worth the FastRPC round-trip overhead

#### Step 5: Attention backward (future, not blocking)

- Flash attention backward on cDSP is NYI
- For now, attention backward runs on CPU
- This is acceptable because attention is O(seq²) and seq is small (128-256) for training

### 3.3 GEMM Count Summary (Training, per layer, per step)

| Phase | NPU GEMMs | CPU GEMMs | CPU elementwise |
|-------|-----------|-----------|-----------------|
| Forward | 7 (Q,K,V,O,gate,up,down) | 0 | RoPE, RMSNorm, SwiGLU, residual |
| Backward dX | 7 (dX for each FC) | 0 | attention backward, SwiGLU backward |
| Backward LoRA grad | 0 (too small) | 21 (3 per FC × 7 FCs) | — |
| **Total** | **14** | **21** | — |

With 28 layers: **392 NPU GEMMs + 588 CPU GEMMs** per training step.

The 21 CPU LoRA-gradient GEMMs per layer are tiny (rank=8), so they complete in <10µs each on ARM. The 14 NPU GEMMs per layer are the real compute, each taking ~240-710µs (dominated by FastRPC flush).

---

## 4. Memory Budget (Qwen3-0.6B, seq=128, LoRA rank=8)

| Component | Size | Notes |
|-----------|------|-------|
| Base weights (Q4_0) | 350 MB | Frozen, on flash |
| LoRA adapters (FP32) | 7 layers × 28 × (dim×8 + 8×unit) × 4B ≈ 14 MB | Trainable |
| Forward activations | 28 × 128 × 1024 × 4B × ~6 ≈ 88 MB | Cached for backward |
| LoRA tmp tensors | 28 × 128 × 8 × 4B × 7 ≈ 0.8 MB | loraTmp per FC |
| Logits + loss | 1 × 151936 × 4B ≈ 0.6 MB | Single token |
| Optimizer state (Adam) | 14 MB × 2 (m, v) ≈ 28 MB | LoRA params only |
| **Total** | **~480 MB** | Fits in 12 GB RAM easily |

At seq=512: activations grow to ~350 MB, total ~750 MB. Still fits.

---

## 5. Implementation Priority

1. **Port fc_layer.cpp LoRA changes** (1-2 days) — enables LoRA forward + backward
2. **Port training infrastructure** (1 day) — train_qwen3_lora.cpp, lora_train.cpp
3. **Enable training mode in Transformer** (1-2 days) — compile with TRAIN mode, add loss + optimizer
4. **NPU dispatch for backward dX GEMMs** (2-3 days) — sgemm_fp32 for calcDerivative
5. **Test on device** (1 day) — verify convergence, measure timing
6. **Attention backward on NPU** (future) — flash_attn backward kernel on cDSP
