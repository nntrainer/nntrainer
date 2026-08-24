# Project Status: NPU Training for Qwen3-0.6B — Latest Update

**Date:** 2026-08-13 (evening session)
**Scope:** Incremental update from `PROJECT_STATUS_2026-08-13.md`

---

## What Changed Since the Previous Status

### 1. All CausalLM Layer Backward Passes — ✅ IMPLEMENTED

All custom CausalLM layers now have `supportBackwarding() = true` and
implemented `calcDerivative` / `calcGradient`:

| Layer | `supportBackwarding()` | `calcDerivative` | `calcGradient` | Status |
|-------|------------------------|------------------|----------------|--------|
| `fully_connected` (FC) | ✅ true | ✅ | ✅ (with LoRA) | **Done** |
| `gate_up_layer` | ✅ true | ✅ | ✅ | **Done** |
| `addition` (residual) | ✅ true | ✅ Pass-through | N/A | **Done** |
| `rms_norm` | ✅ true | ✅ | ✅ | **Done** |
| `swiglu` | ✅ true | ✅ | ✅ | **Done** |
| `mha_core` (attention) | ✅ true | ✅ (with RoPE bw) | ✅ (no weights) | **Done** |
| `embedding_layer` | ✅ true | ✅ | ✅ | **Done** |
| `tie_word_embedding` | ✅ true | ✅ | ✅ | **Done** |
| `lm_head` (FC) | ✅ true | ✅ | ✅ | **Done** |

### 2. Training Data Pipeline — ✅ FIXED

The training data generator now implements proper next-token prediction:
- Input: tokens `[t0, t1, ..., t_{n-1}]`
- Target: shifted tokens `[t1, t2, ..., t_n]`
- Cross-entropy loss over all positions

### 3. `transB=1` Bridge Bug — ✅ FIXED

The `nntr_htp_bridge_sgemm_fp32` transpose logic for `transB=1` (used in
`dot_deriv_wrt_2` / `calcGradient`) has been corrected.

### 4. Cross-Entropy Loss Compilation — ✅ FIXED

**Problem:** `model->compile(x, y, TRAIN)` failed silently because
`addLossLayer("cross")` requires the last layer to be an `ActivationLayer`
with softmax. The training model's last layer was `fully_connected` (lm_head).

**Fix:** Added a softmax activation layer after lm_head in
`constructTrainingModel()`.

### 5. Qwen3 Training Now Runs — ✅ VERIFIED

**Build:** `ninja -C build` succeeds with `-Denable-debug=true`
**Run:** `train_qwen3_lora` compiles the model in TRAIN mode and executes
training iterations.

Log evidence (`logs/log_nntrainer_20260813201544.out`):
- All 28 transformer blocks allocated with forward + gradient tensors
- `cross_softmax0` layer created (softmax + CE fusion working)
- `[NNTrainer] Starts training. Current epoch: 1. Total epochs: 1.`
- `train for iteration` → `# 1 / 1` → training iteration completed
- ~94 seconds per training step (28 layers, seq_len=64, batch=1)

### 6. Build System — ✅ `train_qwen3_lora` in Meson

The `train_qwen3_lora` binary is built via meson and links against
`libcausallm.so` and all custom layer shared libraries.

---

## Current State Summary

| Workstream | Previous Status | Current Status |
|------------|----------------|----------------|
| Qwen3 Inference (hybrid NPU) | ✅ Working | ✅ Working (unchanged) |
| MNIST Training on NPU | ✅ Working | ✅ Working (unchanged) |
| Qwen3 Training on NPU | 🔨 Blocked (no backwarding, compile error) | ✅ **Compiles & runs training iterations** |

---

## What Remains

### Immediate (verify training quality)
1. **Check if loss is decreasing** — run multiple epochs and verify loss
   trend. The previous flat loss (0.939) was due to missing backwarding;
   now that all backward passes are implemented, loss should decrease.
2. **Verify gradient correctness** — compare NPU vs CPU loss curves with
   same seed to ensure bridge GEMMs produce correct gradients.
3. **LoRA adapter saving/loading** — verify the trained adapter can be
   saved and reloaded for inference.

### Short-term (optimization)
4. **Profile training step** — identify bottlenecks (CPU element-wise ops
   vs NPU GEMM dispatch overhead).
5. **Reduce FastRPC overhead** — batch more GEMMs per flush.
6. **Memory optimization** — gradient checkpointing for larger seq_len.

### Medium-term (features)
7. **Q4_0 QAT LoRA training** — train with quantized base weights.
8. **Flash attention backward** on DSP (currently attention backward runs
   on CPU with NPU GEMMs for dQ/dK/dV).
9. **Fused FFN backward** — 3 GEMMs + SwiGLU backward in 1 DSP flush.

---

## Key Files Modified This Session

- `Applications/CausalLM/models/transformer.cpp` — Added softmax activation
  layer after lm_head in `constructTrainingModel()` to enable cross-entropy
  loss fusion.
- `Applications/CausalLM/layers/rms_norm.cpp` — Implemented `forwarding()`,
  `calcDerivative()`, `calcGradient()`, set `supportBackwarding() = true`.
- `Applications/CausalLM/layers/rms_norm.h` — Added `calcGradient()` declaration,
  set `supportBackwarding() = true`.
- `Applications/CausalLM/layers/swiglu.cpp` — Implemented `calcDerivative()`.
- `Applications/CausalLM/layers/embedding_layer.cpp` — Implemented
  `calcDerivative()` (zero), `calcGradient()` (scatter-add), set
  `supportBackwarding() = true`.
- `Applications/CausalLM/layers/embedding_layer.h` — Set
  `supportBackwarding() = true`.
- `Applications/CausalLM/layers/tie_word_embedding.cpp` — Implemented
  `calcDerivative()` (embedding + lm_head modes), `calcGradient()` (scatter-add
  + GEMM), set `supportBackwarding() = true`.
- `Applications/CausalLM/layers/tie_word_embedding.h` — Set
  `supportBackwarding() = true`.
- `Applications/CausalLM/layers/lm_head.cpp` — Implemented `calcDerivative()`
  (dX = dy·W^T), `calcGradient()` (dW = X^T·dy), set `supportBackwarding() = true`.
- `Applications/CausalLM/layers/lm_head.h` — Set `supportBackwarding() = true`.
- `Applications/CausalLM/layers/mha_core.cpp` — Implemented full attention
  backward (dQ, dK, dV, softmax backward, RoPE inverse), `calcGradient()` (no
  weights).
- `Applications/CausalLM/lora_train.cpp` — Fixed training data pipeline
  (next-token prediction), forced FP32-FP32 model_tensor_type for training.
- `Applications/CausalLM/meson.build` — Added `train_qwen3_lora` executable.
- `../ggml-hexagon/ggml/src/ggml-hexagon/nntr-htp-bridge.cpp` — Fixed
  `transB=1` staging allocation bug.

## Build & Run Commands

```bash
# Build (with debug logging)
meson configure build -Denable-debug=true
ninja -C build

# Run Qwen3 LoRA training
LD_LIBRARY_PATH=build/Applications/CausalLM/layers:build/nntrainer:$LD_LIBRARY_PATH \
  ./build/Applications/CausalLM/train_qwen3_lora \
  /path/to/qwen3-0.6b \
  /path/to/prompt.txt \
  --epochs 1 --seq_len 64 --lr 0.0001 --lora_rank 8
```
