# NPU Training Plan: From Prefill Acceleration to On-Device Training

**Date:** 2026-08-05  
**Goal:** Extend the Hexagon cDSP acceleration infrastructure (built for inference/prefill) to support **training** on the NPU, starting with a simple 3-layer MNIST model.

---

## 1. What Exists Today (Prefill Acceleration Summary)

### 1.1 Hexagon cDSP Backend (`nntrainer/hexagon/`)

| Component | File | What it does |
|---|---|---|
| **HexagonContext** | `hexagon_context.cpp` | Registers FC/QKV/GateUp layers under `engine=cdsp`. Sets `HexagonComputeOps` + `HexagonRpcAllocator` on the context. |
| **HexagonComputeOps** | `hexagon_compute_ops.cpp` | Overrides only `gemm_q4_0_accel_fp32` and `gemm_q4_0_batch_fp32` → dispatches Q4_0 GEMMs to DSP via `libggml-hexagon.so`. **All other ops (including all backward/gradient ops) forward to CPU.** |
| **HexagonRpcAllocator** | `hexagon_rpc_allocator.cpp` | Allocates activation tensors in rpcmem (zero-copy DSP access). Registers pools with the bridge. |
| **hexagon_repack** | `hexagon_repack.cpp` | Converts ARM q4_0x4 → DSP q4x4x2 layout at weight load time. |

### 1.2 Bridge Functions (in `libggml-hexagon.so`)

| Bridge function | Purpose | Status |
|---|---|---|
| `nntr_htp_bridge_upload_weight_q4x4x2` | Upload Q4_0 weight to DSP rpcmem arena | ✅ Working |
| `nntr_htp_bridge_gemm_q4_0` | Single Q4_0 GEMM on DSP | ✅ Working |
| `nntr_htp_bridge_gemm_q4_0_batch` | Batched Q4_0 GEMMs (Q/K/V, gate/up) in one flush | ✅ Working |
| `nntr_htp_bridge_flash_attn` | Fused flash attention (Q·K^T + softmax + ·V) on HMX | ✅ Working |
| `nntr_htp_bridge_ffn_swiglu` | Fused FFN (3 GEMMs + SwiGLU) in one flush | ✅ Working |

### 1.3 CausalLM Integration

- **`withHexagonEngine()`** (`llm_util.hpp`): Appends `engine=cdsp` to FC layer properties when `NNTR_USE_HEXAGON_CDSP` is set.
- **Flash attention** (`mha_core.cpp`): dlopen/dlsym bridge, token-threshold gate, causal mask, CPU fallback.
- **Fused FFN** (`fused_ffn_layer.cpp`): Custom layer with bridge dispatch, CPU fallback.
- **Transformer** (`transformer.cpp`): Conditionally tags Q/K/V/O/FFN layers with `engine=cdsp`.

### 1.4 Performance Results (Qwen3-0.6B, Galaxy S25)

| Variant | Prefill TPS | Speedup vs CPU |
|---|---|---|
| CPU (4 threads) | 640 | 1.0× |
| NPU (CDSP, Q4_0 GEMM only) | 906 | 1.42× |
| NPU + Flash Attention | 1136 | 1.78× |
| **NPU + Flash Attn + Fused FFN** | **1233** | **1.93×** |

---

## 2. The Training Gap

### 2.1 What's Missing for Training

The entire NPU acceleration is **inference-only**. Specifically:

1. **No backward GEMM on DSP**: `HexagonComputeOps` only overrides forward Q4_0 GEMM. `calcDerivative()` and `calcGradient()` in FC layers use `Tensor::dot()` which dispatches to CPU `sgemm_fp32`. The DSP bridge has no FP32 GEMM function.

2. **Inference-only layers**: `QKVLayer`, `GateUpLayer`, and `FusedFFNLayer` all throw or return empty for `calcDerivative`/`calcGradient`. `FusedFFNLayer::supportBackwarding()` returns `false`.

3. **Q4_0 weight format**: The DSP bridge works with **quantized** Q4_0 weights. Training requires FP32 weights for gradient computation and weight updates. The quantized format is fundamentally incompatible with gradient-based optimization.

4. **No optimizer on DSP**: The Adam/SGD optimizer runs entirely on CPU.

5. **No training loop in CausalLM**: The CausalLM application is inference-only (forward + generate). No `NeuralNetwork::train()` path is exercised.

### 2.2 What DOES Work for Training

- **`FullyConnectedLayer`** (`fc_layer.cpp`): Full training support — `forwarding()`, `calcDerivative()`, `calcGradient()`. Uses FP32 weights. The `engine=cdsp` tag is already supported (HexagonContext registers FC layers), but the DSP path only activates for Q4_0 weights (the `dotQnK` path in `float_tensor.cpp`).
- **`NeuralNetwork::train()`** (`neuralnet.cpp`): Full training loop — `forwarding(training=true)` → `backwarding()` → optimizer `applyGradient()`.
- **MNIST config** (`test/test_models/models/mnist.ini`): Complete training config with conv/pool/FC layers, Adam optimizer, cross-entropy loss.
- **`HexagonComputeOps`**: Already forwards all non-Q4_0 ops to CPU, so an FP32 FC layer tagged `engine=cdsp` would just run on CPU today (no crash, no acceleration).

---

## 3. Strategy: NPU-Accelerated Training

### 3.1 Core Insight

The DSP's HMX systolic array excels at **quantized GEMM** (Q4_0 × FP32). For training, we need:
- **Forward**: `Y = W × X` (weight × activation)
- **Backward (grad w.r.t. input)**: `dX = W^T × dY`
- **Backward (grad w.r.t. weight)**: `dW = dY × X^T`
- **Weight update**: `W -= lr × dW` (element-wise, on CPU)

The forward GEMM can already go to DSP if weights are Q4_0. But training needs FP32 weights for updates. **The key question is: can we do FP32 GEMM on the DSP?**

### 3.2 Three Approaches (ranked by pragmatism)

#### Approach A: FP32 SGEMM Bridge (Recommended Starting Point)

Add a new bridge function `nntr_htp_bridge_sgemm_fp32` that dispatches FP32 matrix multiply to the DSP. The DSP's HVX/HMX can do FP32 GEMM (slower than Q4_0, but still parallelized across HTP threads).

**Why this is the best starting point:**
- Clean separation: forward and backward GEMMs both go to DSP
- No quantization complexity — weights stay FP32 throughout
- `HexagonComputeOps` already has the pattern: override `sgemm_fp32` to dispatch to bridge, fall back to CPU on error
- The 3-layer MNIST model is pure FC layers, so GEMM is the only compute-intensive op
- Directly answers "can we train on NPU?"

**What needs to be built:**
1. `nntr_htp_bridge_sgemm_fp32()` in `libggml-hexagon.so` (new bridge function)
2. Override `sgemm_fp32()` in `HexagonComputeOps` to dispatch to bridge
3. A 3-layer MNIST model config with `engine=cdsp` on FC layers
4. Training driver that calls `NeuralNetwork::train()`

#### Approach B: Quantization-Aware Training (QAT)

Keep weights in Q4_0 for forward pass (leveraging existing fast DSP GEMM), compute gradients in FP32, update weights in FP32 master copy, re-quantize.

**Pros:** Forward pass is already accelerated (1233 TPS demonstrated)
**Cons:** Very complex — needs FP32 master weights, re-quantization after every step, backward Q4_0 GEMM bridge function, gradient straight-through estimator for quantization. This is a research project, not a first step.

#### Approach C: Hybrid (Forward on DSP, Backward on CPU)

Forward pass uses Q4_0 GEMM on DSP (already works). Backward pass uses CPU FP32 GEMM. Weight update on CPU.

**Pros:** Minimal new code — forward already works
**Cons:** Needs dual weight representation (Q4_0 for forward, FP32 for backward/update). The weight synchronization overhead may negate the forward speedup. Also, the forward output is FP32 (from Q4_0 dequant GEMM), so the backward pass can use FP32 gradients — but the weight gradient `dW = dY × X^T` needs the original FP32 input, not the Q4_0 weight.

---

## 4. Implementation Plan: 3-Layer MNIST on NPU (Approach A)

### 4.1 Target Model

```
Input (1×28×28 = 784)
  → Flatten
  → FC(784 → 128) + ReLU     [engine=cdsp]
  → FC(128 → 64) + ReLU      [engine=cdsp]
  → FC(64 → 10) + Softmax    [engine=cdsp]
  → Cross-Entropy Loss
```

All FC layers tagged `engine=cdsp`, weights in FP32, GEMMs dispatched to DSP.

### 4.2 Phase 1: FP32 SGEMM Bridge Function

**File:** `ggml-hexagon/ggml/src/ggml-hexagon/nntr-htp-bridge.cpp` (external repo)

Add a new C-ABI function:

```c
extern "C" __attribute__((visibility("default")))
int nntr_htp_bridge_sgemm_fp32(
    const float *A,          // M×K, row-major
    const float *B,          // K×N, row-major
    float *C,                // M×N, row-major
    unsigned int M,
    unsigned int N,
    unsigned int K,
    int transA,              // 0=no transpose, 1=transpose A
    int transB               // 0=no transpose, 1=transpose B
);
```

**Implementation approach:**
1. Stage A, B, C in rpcmem (or use zero-copy if already in rpcmem via HexagonRpcAllocator)
2. Build `ggml_tensor` descriptors for A, B, C (all FP32)
3. Set `C.op = GGML_OP_MUL_MAT`, `C.src[0] = A`, `C.src[1] = B`
4. Enqueue `HTP_OP_MUL_MAT` opnode
5. `flush(true)` — single FastRPC round trip
6. Copy result back if not zero-copy

**Key consideration:** The DSP's HMX array is optimized for quantized GEMM. FP32 GEMM will use HVX (vector) path, which is still parallelized across HTP threads but slower per-op than Q4_0. For MNIST-sized matrices (784×128, 128×64, 64×10), the GEMMs are small — the DSP may not be faster than CPU for these sizes. **This is expected for a proof-of-concept; the goal is to validate the training pipeline, not to beat CPU on MNIST.**

### 4.3 Phase 2: Override `sgemm_fp32` in HexagonComputeOps

**File:** `nntrainer/hexagon/hexagon_compute_ops.cpp`

```cpp
// Add to BridgeApi struct:
using nntr_htp_bridge_sgemm_fn = int (*)(const float *, const float *,
                                          float *, unsigned int,
                                          unsigned int, unsigned int,
                                          int, int);
nntr_htp_bridge_sgemm_fn sgemm = nullptr;

// In get_bridge_api():
a.sgemm = reinterpret_cast<nntr_htp_bridge_sgemm_fn>(
  sym("nntr_htp_bridge_sgemm_fp32"));

// Override sgemm_fp32 in HexagonComputeOps:
void sgemm_fp32(const unsigned int TStorageOrder, bool TransA, bool TransB,
                const unsigned int M, const unsigned int N,
                const unsigned int K, const float alpha, const float *A,
                const unsigned int lda, const float *B,
                const unsigned int ldb, const float beta, float *C,
                const unsigned int ldc) override {
  // For now, only dispatch simple cases (alpha=1, beta=0, no padding)
  if (alpha == 1.0f && beta == 0.0f &&
      lda == K && ldb == N && ldc == N) {
    const BridgeApi *api = get_locked_bridge_api();
    int rc = api->sgemm(A, B, C, M, N, K, TransA, TransB);
    if (rc == 0) return;
    ml_logw("HexagonComputeOps::sgemm_fp32: bridge failed (rc=%d), "
            "falling back to CPU", rc);
  }
  // Fallback to CPU
  cpu_->sgemm_fp32(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda,
                   B, ldb, beta, C, ldc);
}
```

**Important:** The `sgemm_fp32` override must handle the cases used by:
- `FullyConnectedLayer::forwarding()` → `input_.dot(weight, hidden_, false, false)` → `sgemm_fp32` with M=batch, N=unit, K=input_dim
- `FullyConnectedLayer::calcDerivative()` → `ret_.dot_deriv_wrt_1(weight, derivative_, false, false)` → `sgemm_fp32` with transposed weight
- `FullyConnectedLayer::calcGradient()` → `input_.dot_deriv_wrt_2(djdw, derivative_, false, false)` → `sgemm_fp32` with transposed input

### 4.4 Phase 3: MNIST Training Config

**File:** `test/test_models/models/mnist_3layer_npu.ini` (new)

```ini
[Model]
Type = NeuralNetwork
Epochs = 100
Loss = cross
Save_Path = "mnist_3layer_npu.bin"
batch_size = 64

[LearningRateScheduler]
type = constant
Learning_rate = 1e-3

[Optimizer]
Type = adam
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7

[inputlayer]
Type = input
Input_Shape = 1:28:28

[flatten]
Type = flatten
input_layers = inputlayer

[fc1]
Type = fully_connected
input_layers = flatten
Unit = 128
weight_initializer = xavier_uniform
bias_initializer = zeros
Activation = relu
engine = cdsp

[fc2]
Type = fully_connected
input_layers = fc1
Unit = 64
weight_initializer = xavier_uniform
bias_initializer = zeros
Activation = relu
engine = cdsp

[outputlayer]
Type = fully_connected
input_layers = fc2
Unit = 10
weight_initializer = xavier_uniform
bias_initializer = zeros
Activation = softmax
engine = cdsp
```

### 4.5 Phase 4: Training Driver

**File:** `test/mnist_npu_train.cpp` (new)

A minimal training driver that:
1. Creates a `NeuralNetwork` from the INI config
2. Loads MNIST training data (using existing `DataBuffer` or raw MNIST files)
3. Calls `model->train()` with training/validation data
4. Prints loss and accuracy per epoch

```cpp
// Pseudocode
auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NETWORK);
model->load("mnist_3layer_npu.ini");
model->compile();
model->initialize();
model->readTrainingData(train_data, train_labels);
model->readValData(val_data, val_labels);
model->train();
```

### 4.6 Phase 5: Build & Test on Device

```bash
# Build nntrainer with Hexagon cDSP support
meson setup build -Denable-hexagon-cdsp=true -Denable-transformer=true
ninja -C build

# Deploy to device
adb push build/libnntrainer.so /data/local/tmp/nntrainer/
adb push libggml-hexagon.so /data/local/tmp/nntrainer/
adb push mnist_3layer_npu.ini /data/local/tmp/nntrainer/
adb push mnist_npu_train /data/local/tmp/nntrainer/

# Run training
adb shell "cd /data/local/tmp/nntrainer && \
  export LD_LIBRARY_PATH=/system/lib64:. && \
  export NNTR_USE_HEXAGON_CDSP=1 && \
  ./mnist_npu_train mnist_3layer_npu.ini"
```

---

## 5. Expected Challenges & Mitigations

| Challenge | Mitigation |
|---|---|
| FP32 GEMM on DSP may be slower than CPU for small matrices | Start with batch_size=64+ to amortize FastRPC overhead. Profile to find crossover point. |
| `sgemm_fp32` signature has alpha/beta/lda/ldb/ldc — bridge may not support all cases | Only dispatch to DSP for the simple case (alpha=1, beta=0, no padding). Fall back to CPU otherwise. |
| Backward GEMM uses transposed weight — need `transA`/`transB` support | Bridge function accepts transA/transB flags. DSP kernel handles transpose via descriptor strides. |
| Activation tensors must be in rpcmem for zero-copy | Already handled by `HexagonRpcAllocator` — it's set as the context's memory allocator. |
| Conv2D layers don't have DSP acceleration | 3-layer MNIST uses only FC layers (no conv). If conv is needed later, it stays on CPU. |
| Weight gradients need accumulation across mini-batch | `calcGradient` already handles this via `isGradientFirstAccess` check. The GEMM itself is the same. |
| No FP32 GEMM kernel exists in the DSP skel | The `HTP_OP_MUL_MAT` opcode already supports FP32 tensors. The kernel dispatches to HVX FP32 path. May need to verify this works. |

---

## 6. Success Criteria

1. **Correctness:** Training loss decreases over epochs, validation accuracy > 90% on MNIST
2. **NPU utilization:** `nntr_htp_bridge_sgemm_fp32` is called for every FC forward and backward GEMM (verified via call counter or logging)
3. **No CPU fallback:** Bridge function returns 0 (success) for all GEMM calls in the training loop
4. **End-to-end:** `NeuralNetwork::train()` completes without errors with `engine=cdsp` layers

---

## 7. MNIST-First vs Direct Qwen Training: Complexity Analysis

### 7.1 Why MNIST First (Strongly Recommended)

The 3-layer MNIST model uses **only standard `FullyConnectedLayer`** layers, which already have complete training support (`forwarding`, `calcDerivative`, `calcGradient` all implemented). The only new work is the FP32 GEMM bridge function. This makes MNIST the ideal first target.

### 7.2 Qwen3 Training: Layer-by-Layer Backwarding Status

Training Qwen3-0.6B end-to-end requires **every layer** in the forward graph to support backwarding. Here is the current status of every CausalLM custom layer used in the Qwen3 transformer:

| Layer | Used in Qwen3? | `supportBackwarding()` | `calcDerivative` | `calcGradient` | Training Status |
|---|---|---|---|---|---|
| `fully_connected` (FC) | ✅ Q/K/V/O/FFN-down | ✅ true | ✅ Implemented | ✅ Implemented | **Ready** |
| `reshaped_rms_norm` | ✅ Q/K norm, attention/FFN norm | ❌ false | ❌ Throws | ❌ Throws | **Not implemented** |
| `mha_core` | ✅ Attention | ✅ true | ⚠️ Empty `{}` | ⚠️ Empty `{}` | **Stub only** |
| `fused_ffn` | ✅ FFN (when enabled) | ❌ false | ⚠️ Empty `{}` | ⚠️ Empty `{}` | **Not implemented** |
| `gate_up_layer` | ✅ FFN (when not fused) | ❌ false | ❌ Throws | ❌ Throws | **Not implemented** |
| `swiglu` | ✅ FFN activation | ✅ true | ⚠️ Commented out | ❌ Not tested | **Partial/stub** |
| `embedding_layer` | ✅ Token embedding | ❌ false | ❌ Throws | ⚠️ Empty `{}` | **Not implemented** |
| `tie_word_embedding` | ✅ LM head | ❌ false | ❌ Throws | ⚠️ Empty `{}` | **Not implemented** |
| `custom_multiply` | ✅ Residual adds | ✅ Conditional | ✅ Implemented | — | **Ready** |
| `rms_reverse_norm` | ✅ (some models) | ✅ true | ❌ "Not implemented" | ❌ Throws | **Not implemented** |
| `per_layer_slice` | ✅ KV cache slicing | ❌ false | ❌ Throws | ❌ Throws | **Not implemented** |
| `logit_softcapping` | ✅ (some models) | ❌ false | ⚠️ Implemented | — | **Partial** |

### 7.3 What It Would Take to Train Qwen3 End-to-End

To train Qwen3-0.6B, you would need to implement backwarding for **at minimum 7 layers** that currently don't support it:

1. **`reshaped_rms_norm`** — RMSNorm backward: `dX = (W/RMS) * (dY - mean(dY*W*X) * X / RMS^2 - mean(dY*W) * W/RMS)`. Moderate complexity. Also needs gradient w.r.t. weight `dW = sum(dY * X / RMS)`.

2. **`mha_core`** — Attention backward: `dQ = (dS * V) * K^T`, `dK = Q^T * (dS * V)`, `dV = S^T * dS`, where `dS = softmax_backward(dY*V)`. This is the **hardest** — attention backward is notoriously complex, especially with RoPE, GQA, and KV cache. The existing flash attention bridge is forward-only. Would need a backward attention bridge or CPU implementation.

3. **`fused_ffn` / `gate_up_layer`** — FFN backward: 3 transposed GEMMs (already doable via `sgemm_fp32` bridge) + SwiGLU backward (derivative of SiLU). Moderate if using standard FC layers instead of fused.

4. **`swiglu`** — SwiGLU backward: `d(gate) = dY * up * sigmoid'(gate)`, `d(up) = dY * gate`. Low complexity, but currently stubbed.

5. **`embedding_layer`** — Embedding backward: scatter-add gradients to embedding matrix. Low complexity.

6. **`tie_word_embedding`** — LM head backward: same as embedding backward + weight tying. Low-moderate complexity.

7. **`per_layer_slice`** — Slice backward: just gather/scatter. Low complexity.

**Estimated effort:** 2-4 weeks of engineering work for a single developer, assuming the GEMM bridge already exists. The attention backward alone is ~1 week.

### 7.4 Additional Qwen Training Challenges Beyond Layer Backwarding

| Challenge | Complexity | Notes |
|---|---|---|
| **Weight format** | High | Qwen weights are Q4_0 quantized. Training needs FP32. Must dequantize → train → (optionally) re-quantize. Or load FP32 weights from safetensors. |
| **Memory** | High | Qwen3-0.6B has ~600M params. FP32 weights = 2.4GB. Gradients = another 2.4GB. Optimizer states (Adam m,v) = 4.8GB. Total ~9.6GB. Galaxy S25 has ~12GB RAM. Very tight. |
| **Attention backward on DSP** | Very High | No backward flash attention kernel exists. Would need to implement `dQ/dK/dV` bridge function or run attention backward on CPU. |
| **Training data pipeline** | Medium | CausalLM has no training data loader. Need to add dataset loading, batching, label generation for causal LM training. |
| **Gradient checkpointing** | Medium | 28 layers × full activations = too much memory. Need gradient checkpointing (recompute forward during backward). Not implemented. |
| **Mixed precision** | Medium | FP32 training of 0.6B model is slow. FP16 mixed precision would help but needs FP16 GEMM bridge + loss scaling. |

### 7.5 Recommended Path: MNIST → Qwen

```
Step 1: MNIST 3-layer FC (this plan)
  - Validates FP32 GEMM bridge on DSP
  - Validates training loop with engine=cdsp
  - Validates optimizer + gradient flow
  - Effort: ~1 week

Step 2: Add backwarding to simple CausalLM layers
  - RMSNorm backward
  - SwiGLU backward  
  - Embedding backward
  - Use standard FC layers (not fused_ffn/gate_up) for FFN
  - Effort: ~1 week

Step 3: Attention backward
  - Implement mha_core calcDerivative/calcGradient
  - Start with CPU backward (no DSP)
  - Effort: ~1-2 weeks

Step 4: Qwen3 fine-tuning (LoRA)
  - Use LoRA adapters (FC layer already supports this!)
  - Only train LoRA params → much less memory
  - Forward on DSP (existing), backward on CPU initially
  - Effort: ~1 week (after steps 2-3)

Step 5: Full Qwen3 training on DSP
  - Move backward GEMMs to DSP
  - Attention backward on DSP
  - Effort: ~2+ weeks
```

**Bottom line:** Direct Qwen training is 5-10× more work than MNIST. MNIST first is the right call — it validates the core NPU training pipeline (GEMM bridge + training loop + optimizer) with minimal complexity, then each piece can be extended incrementally toward Qwen.

---

## 8. Future Extensions (After MNIST Works)

### 8.1 FP16 Training

- Use FP16 activations + FP32 master weights (mixed precision)
- Override `sgemm_fp16` in `HexagonComputeOps` to dispatch to DSP
- FP16 GEMM on HMX is significantly faster than FP32

### 8.2 Larger Models

- Once the training pipeline works on MNIST, scale to transformer fine-tuning
- The flash attention bridge already exists — could be reused for training forward pass
- Need backward attention kernel (or use CPU backward + DSP forward)

### 8.3 Gradient Accumulation on DSP

- Keep weight gradients in rpcmem, accumulate across micro-batches
- Single flush for forward + backward + gradient accumulation
- Approaches ggml-hexagon's single-flush graph execution

### 8.4 QAT (Quantization-Aware Training)

- Forward: Q4_0 GEMM on DSP (existing, fast)
- Backward: FP32 GEMM on DSP (new bridge function)
- Weight update: CPU, with straight-through estimator for quantization
- Re-quantize weights after each step

---

## 9. File Summary


| File | Action | Description |
|---|---|---|
| `ggml-hexagon/.../nntr-htp-bridge.cpp` | Modify | Add `nntr_htp_bridge_sgemm_fp32()` |
| `nntrainer/hexagon/hexagon_compute_ops.cpp` | Modify | Override `sgemm_fp32()` to dispatch to bridge |
| `test/test_models/models/mnist_3layer_npu.ini` | Create | 3-layer FC MNIST config with `engine=cdsp` |
| `test/mnist_npu_train.cpp` | Create | Training driver |
| `test/meson.build` | Modify | Add training driver target |

---

## 10. Key Architecture Diagram


```
                    Training Loop (NeuralNetwork::train)
                    ┌─────────────────────────────────┐
                    │  for each epoch:                │
                    │    for each batch:               │
                    │      forwarding(training=true)   │
                    │      backwarding()                │
                    │      optimizer.applyGradient()   │
                    │    validate()                    │
                    └─────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   FC Layer (engine=cdsp)        │
                    │   ┌─────────────┐                │
                    │   │ forwarding() │ → sgemm_fp32() │──→ DSP bridge
                    │   └─────────────┘                │
                    │   ┌─────────────┐                │
                    │   │calcDerivative│ → sgemm_fp32()│──→ DSP bridge
                    │   └─────────────┘                │
                    │   ┌─────────────┐                │
                    │   │calcGradient()│ → sgemm_fp32()│──→ DSP bridge
                    │   └─────────────┘                │
                    └──────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Optimizer (CPU)                 │
                    │  W -= lr × dW                    │
                    └──────────────────────────────────┘
```

The DSP handles all three GEMMs (forward, backward-input, backward-weight). The optimizer runs on CPU (element-wise ops, not worth offloading). Weight updates happen in FP32 on CPU, then weights are re-staged to rpcmem for the next forward pass.
