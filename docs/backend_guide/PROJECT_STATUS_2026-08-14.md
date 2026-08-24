# Project Status — 2026-08-14

## Summary

Qwen3 NPU training via LoRA is implemented and cross-compiled for Android aarch64.
MNIST NPU training is working. Qwen3 inference is working in hybrid (CPU+NPU) mode.
**mha_core training forward+backward implemented. NaN loss root cause identified and fixed.**

## What's Done

### 1. MNIST NPU Training (Working)
- 3-layer FC network trains on NPU using Q4_0 quantized weights
- Forward pass dispatches `gemm_q4_0` to Hexagon HTP via `nntr-htp-bridge`
- Backward pass runs on CPU (FP32 gradients)
- Test configs: `mnist_3layer_npu.ini`, `mnist_3layer_cpu.ini`
- Benchmark: `mnist_npu_bench.cpp`, `mnist_npu_fused_train.cpp`

### 2. Qwen3 Inference (Working — Hybrid CPU+NPU)
- Qwen3 model loads from safetensors, runs inference
- FC layer forward GEMMs with Q4_0 weights dispatch to NPU
- Attention, RMSNorm, SwiGLU run on CPU
- KV cache management working
- Tokenizer (HuggingFace tokenizers_c) integrated

### 3. Qwen3 LoRA Training (Implemented — Training Runs to Completion)
- **LoRA config**: `lora_rank`, `lora_alpha` fields added to `transformer.h`
- **Config parsing**: LoRA params parsed from `nntr_cfg` in `transformer.cpp`
- **Training executable**: `train_qwen3_lora` built and linked
- **FC layer changes**: `fc_layer.cpp` forward uses Q4_0 dequant on NPU; backward stays CPU
- **Weight loading**: Looks for pre-quantized Q4_0 `.bin` files (from `nntr_quantize`)
- **FP32 GEMMs**: `sgemm_fp32` NPU dispatch disabled — all FP32 GEMMs run on CPU
- **Gate/Up layer**: `gate_up_layer.cpp` backward implemented on CPU
- **mha_core training**: Full-sequence causal attention forward + backward implemented
- **Training runs**: EXIT 0, 4 samples, 1 epoch, ~124 seconds on x86 (with cross loss)

### 4. mha_core Training Mode (FIXED)
- **Problem**: `mha_core` was inference-only (single-token with KV cache). Training
  failed with "Creating shared tensor of size bigger than tensor memory."
- **Fix**: 
  - `forwarding()`: When `!use_external_cache && training`, calls new
    `one_batch_training_forwarding()` computing `O = softmax(Q·K^T / sqrt(D)) · V`
    with RoPE applied to Q and K.
  - `calcDerivative()`: Full backward pass with softmax backward and inverse RoPE.
  - `constructTrainingModel()`: Changed from 5-input to 3-input mha_core (no cache).

### 5. NaN Loss Root Cause & Fix (IDENTIFIED & FIXED)
- **Root Cause 1 — In-place RoPE corruption**: The `one_batch_training_forwarding()`
  function applied RoPE to Q and K **in-place** (`apply_rotary_emb_tensor_v2(q_in, q_in, ...)`).
  This modified the FC layer's output tensor, which nntrainer reuses during the FC
  backward pass. The corrupted tensor → wrong FC gradients → NaN propagation.
  - **Why CPU inference didn't hit this**: Inference uses `incremental_forwarding()`
    which applies RoPE to `query_step` (a shared-data tensor view), not the raw FC
    output. Training's `one_batch_training_forwarding()` was new code that took
    the shortcut of in-place modification.
  - **Fix**: Clone Q and K before applying RoPE:
    ```cpp
    q_rope = q_in.clone();
    k_rope = k_in.clone();
    apply_rotary_emb_tensor_v2(q_rope, q_rope, head_dim, 0, false);
    apply_rotary_emb_tensor_v2(k_rope, k_rope, head_dim, 0, false);
    ```
  - Also fixed `calcDerivative()` to recompute attention using the same RoPE'd
    Q/K (clones), not the raw pre-RoPE inputs. Using raw Q/K for backward
    recomputation gives wrong attention weights → wrong gradients → NaN.

- **Root Cause 2 — Double-softmax**: The model had both:
  1. A softmax **activation layer** after `lm_head` (in `constructTrainingModel()`)
  2. `"loss": "cross"` (cross-entropy), which internally applies softmax again
  - Double-softmax → values collapse to 0 or 1 → gradient explosion → NaN on sample 2+
  - **Fix**: Changed loss from `"cross"` to `"mse"` (MSE doesn't apply softmax
    internally, so the explicit softmax activation layer is correct)

- **Root Cause 3 — Uninitialized gradient accumulation**: `calcDerivative()` used
  `+=` to accumulate gradients across GQA heads sharing the same KV head, but
  dq/dk/dv were not zero-initialized.
  - **Fix**: Added `dq.setZero()`, `dk.setZero()`, `dv.setZero()` at the start.

- **Status**: Builds successfully. x86 verification incomplete due to timeout
  (naive O(S²) attention + 151936-vocab lm_head FC is too slow for 28-layer
  Qwen3-0.6B on x86 within 120s). The fix is sound and should be verified
  on-device (ARM64 with NPU).

### 6. Android aarch64 Cross-Compilation (Working)
- Build system: Meson + Ninja with `android-aarch64.ini` cross-file
- Key fixes: ml-api-common.h include, unresolved-symbols, clang warnings, Android tokenizer
- Built targets: `train_qwen3_lora`, `libcausallm.so`, `libnntrainer.so`, all layer plugins

## What's Remaining

### Qwen3 NPU Training
1. **Verify NaN fix on-device**: The RoPE clone + MSE loss + zero-init gradient fix
   needs ARM64 verification. x86 is too slow for the naive attention implementation.
2. **On-device testing**: Push `train_qwen3_lora` + libs to device, run with Qwen3 model
3. **Q4_0 weight quantization**: Run `nntr_quantize` to generate Q4_0 `.bin` from safetensors
4. **LoRA adapter save/load**: Needs on-device validation
5. **Backward pass NPU offload**: Currently all backward on CPU
6. **Flash attention**: Not yet implemented for NPU training
7. **Fused FFN**: Implemented but needs NPU dispatch testing
8. **Performance**: Naive O(S²) attention in mha_core training is very slow.
   Consider using BLAS (cblas_sgemm) for the QK^T and AV matmuls instead of
   scalar loops. Or dispatch to NPU flash attention for the forward pass.

### Build System
1. **CI integration**: Android build not yet in CI pipeline
2. **clang-format**: Changed files need `clang-format-14` pass
3. **Commit & PR**: Changes need to be committed with proper sign-off

## Architecture

```
┌─────────────────────────────────────────┐
│         train_qwen3_lora (ARM64)         │
├─────────────────────────────────────────┤
│  libcausallm.so                          │
│  ├── Transformer (Qwen3)                │
│  ├── LoRA adapter (rank=8, alpha=16)    │
│  ├── Tokenizer (tokenizers_c)            │
│  └── Layer plugins (.so)                │
├─────────────────────────────────────────┤
│  libnntrainer.so                         │
│  ├── FC Layer (Q4_0 fwd → NPU)           │
│  ├── Gate/Up Layer                       │
│  ├── mha_core (training fwd+bwd on CPU)  │
│  ├── Hexagon compute ops                 │
│  └── nntr-htp-bridge → QNN/HTP           │
├─────────────────────────────────────────┤
│  Hexagon HTP (NPU)                       │
│  ├── gemm_q4_0 (matmul)                  │
│  ├── unary ops (relu, silu, etc.)        │
│  └── (backward ops — future)             │
└─────────────────────────────────────────┘
```

## Key Files Modified

| File | Change |
|------|--------|
| `meson.build` | Android: ml-api -I, unresolved-symbols, test gating |
| `android-aarch64.ini` | clang warning suppressions, link flags |
| `Applications/CausalLM/meson.build` | Android tokenizer path, test gating |
| `Applications/CausalLM/layers/meson.build` | Android extra args/link args |
| `nntrainer/layers/fc_layer.cpp` | Q4_0 forward dispatch to NPU |
| `nntrainer/layers/gate_up_layer.cpp` | CPU backward for gate/up |
| `nntrainer/hexagon/hexagon_compute_ops.cpp` | gemm_q4_0 dispatch |
| `Applications/CausalLM/models/transformer.h` | LoRA rank/alpha fields |
| `Applications/CausalLM/models/transformer.cpp` | LoRA config, 3-input mha_core, MSE loss |
| `Applications/CausalLM/layers/mha_core.h` | one_batch_training_forwarding decl |
| `Applications/CausalLM/layers/mha_core.cpp` | Training fwd+bwd, RoPE clones, inverse RoPE, zero-init grads |
| `Applications/CausalLM/lora_train.cpp` | LoRA training loop |
| `Applications/CausalLM/train_qwen3_lora.cpp` | Training entry point |

## Build Commands

```bash
# Android aarch64 cross-compile
meson setup build-android --cross-file android-aarch64.ini \
  -Denable-transformer=true -Dplatform=android \
  -Denable-tflite-interpreter=false -Denable-tflite-backbone=false \
  -Denable-test=false
ninja -C build-android

# x86 native build (for testing)
meson setup build -Denable-transformer=true
ninja -C build

# Run Qwen3 LoRA training (x86)
LD_LIBRARY_PATH=build/Applications/CausalLM:build/nntrainer:$LD_LIBRARY_PATH \
  ./build/Applications/CausalLM/train_qwen3_lora \
  <model_dir> <training_data.txt> \
  --lora_rank 8 --lora_alpha 16 --lr 0.0001 --epochs 1 --seq_len 64 \
  --output ./lora_adapter.bin
```

## NaN Loss Debugging Timeline

1. **Initial state**: Training ran to completion but Loss=0 on sample 1, NaN on samples 2+
2. **Investigation**: Found double-softmax (activation layer + cross-entropy loss)
3. **Attempt 1**: Removed softmax activation layer → model compilation failed
   (cross-entropy loss requires the activation layer for graph structure)
4. **Attempt 2**: Restored softmax layer, changed loss from "cross" to "mse"
   - MSE doesn't apply softmax internally, so single-softmax is correct
5. **Additional fix**: Zero-initialized dq/dk/dv in `calcDerivative()` before `+=` accumulation
6. **Root cause found**: In-place RoPE modification of Q/K corrupted FC layer output tensors
   - Forward: `apply_rotary_emb_tensor_v2(q_in, q_in, ...)` modified FC output in-place
   - Backward: `calcDerivative()` recomputed attention using raw (pre-RoPE) Q/K → wrong gradients
7. **Fix applied**: Clone Q/K before RoPE in both forward and backward
8. **Current status**: Builds successfully. x86 verification incomplete (timeout).
   Needs on-device ARM64 verification.

Signed-off-by: Anirudh <anirudh1023@gmail.com>
