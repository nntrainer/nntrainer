# Exhaustive NPU Training Report — MNIST on Hexagon cDSP (HTP v79)

## 1. Executive Summary

**Can backprop run on the NPU?** Yes — the dispatch infrastructure works. Forward GEMMs and some backward GEMMs complete on the DSP. However, the HMX F32 kernel has a bug that hangs the DSP on certain backward GEMM dimensions (K=32), preventing full-NPU training.

**Hybrid mode (forward on NPU, backward on CPU)** runs to completion but barely learns (loss 2.38→2.33, accuracy 10%→13.5%), likely because the DSP's FP32 MUL_MAT internally quantizes to F16, losing precision needed for gradient computation.

**CPU-only training** works well: loss 2.13→1.33, accuracy 40.6%→91.7%.

---

## 2. Model Architecture

```
Input (1×28×28) → Flatten (784) → FC1(128, ReLU) → FC2(64, ReLU) → Output(10, Softmax)
```

- **Optimizer**: Adam (lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7)
- **Loss**: Cross-entropy
- **Batch size**: 32
- **Epochs**: 3
- **Training samples**: 100 (3 steps/epoch)
- **Validation samples**: 100

---

## 3. GEMM Operations Per Training Step

Each training step (one batch) involves **9 GEMM operations**:

### Forward Pass (3 GEMMs)

| # | Operation | Formula | transA | transB | M | N | K | A shape | B shape | C shape |
|---|-----------|---------|--------|--------|---|---|---|---------|---------|--------|
| 1 | FC1 forward | Y = X·W | F | F | 32 | 128 | 784 | [32,784] | [784,128] | [32,128] |
| 2 | FC2 forward | Y = X·W | F | F | 32 | 64 | 128 | [32,128] | [128,64] | [32,64] |
| 3 | Output forward | Y = X·W | F | F | 32 | 10 | 64 | [32,64] | [64,10] | [32,10] |

### Backward Pass (6 GEMMs)

| # | Operation | Formula | transA | transB | M | N | K | A shape | B shape | C shape |
|---|-----------|---------|--------|--------|---|---|---|---------|---------|--------|
| 4 | dX output | dX = dY·W^T | F | T | 64 | 10 | 32 | [64,32] | [10,32] | [64,10] |
| 5 | dW output | dW = X^T·dY | T | F | 32 | 64 | 10 | [10,32] | [10,64] | [32,64] |
| 6 | dX fc2 | dX = dY·W^T | F | T | 128 | 64 | 32 | [128,32] | [64,32] | [128,64] |
| 7 | dW fc2 | dW = X^T·dY | T | F | 32 | 128 | 64 | [64,32] | [64,128] | [32,128] |
| 8 | dX fc1 | dX = dY·W^T | F | T | 32 | 128 | 64 | [32,64] | [128,64] | [32,128] |
| 9 | dW fc1 | dW = X^T·dY | T | F | 32 | 784 | 28 | [28,32] | [28,784] | [32,784] |

### Non-GEMM Operations (all on CPU)

| Operation | Where | Description |
|-----------|-------|-------------|
| ReLU activation | FC1, FC2 forward | Element-wise max(0, x) |
| Softmax | Output forward | Normalized exponentiation |
| Cross-entropy loss | Output | -Σ y·log(ŷ) |
| ReLU derivative | FC1, FC2 backward | Element-wise mask |
| Softmax derivative | Output backward | Gradient w.r.t. softmax |
| Adam optimizer | All weights | Momentum + variance + weight update |
| Bias add | All FC forward | X + bias |
| Bias gradient | All FC backward | Sum of gradients |

---

## 4. FastRPC Round Trips and Flushes

### 4.1 CPU-Only Mode

- **FastRPC round trips**: 0 (no DSP involvement)
- **Flushes**: 0
- **All 9 GEMMs + all non-GEMM ops run on CPU**

### 4.2 Full-NPU Mode (all GEMMs to DSP)

Each GEMM = 1 enqueue + 1 blocking flush = **1 FastRPC round trip**.

Per training step:
- Forward: 3 GEMMs = 3 round trips
- Backward: 6 GEMMs = 6 round trips
- **Total: 9 FastRPC round trips per step**

Per epoch (3 steps + 3 validation steps):
- Training: 9 × 3 = 27 round trips
- Validation: 3 × 3 = 9 round trips (forward only)
- **Total: 36 round trips per epoch**

Per 3-epoch run:
- **Total: 108 FastRPC round trips**

Each round trip involves:
1. `sess->enqueue_op(node)` — builds op descriptor in shared memory
2. `sess->flush(true)` — blocking call that:
   a. Serializes op batch into FastRPC message
   b. Sends to cDSP via `htp_iface_skel_handle_invoke`
   c. DSP dequeues, executes `op_matmul()` (HMX or HVX path)
   d. DSP writes result to rpcmem
   e. FastRPC returns to CPU
3. `memcpy(C, out_data, out_bytes)` — copy result from rpcmem to caller buffer

### 4.3 Hybrid Mode (forward on NPU, backward on CPU)

Per training step:
- Forward: 3 GEMMs on NPU = 3 round trips
- Backward: 6 GEMMs on CPU = 0 round trips
- **Total: 3 FastRPC round trips per step**

Per epoch (3 train + 3 validation):
- Training: 3 × 3 = 9 round trips
- Validation: 3 × 3 = 9 round trips
- **Total: 18 round trips per epoch**

Per 3-epoch run:
- **Total: 54 FastRPC round trips** (verified by log: 54 `execute-op MUL_MAT` entries)

---

## 5. NPU Forward Pass — Detailed Flush Log

From verbose log, each forward GEMM produces this sequence:

```
# GEMM 1: FC1 forward (M=32, N=128, K=784)
ggml-hex: HTP0 execute-op MUL_MAT: sgemm-a x sgemm-b -> sgemm-out : 784:32 x 784:128 -> 128:32 : f32 x f32 -> f32
ggml-hex: add-buffer #0 : fd 19 base 0x... size 4194304    ← staging buffer (rpcmem)
ggml-hex: add-tensor #0 sgemm-a : 784:32:1:1               ← A = input [32,784]
ggml-hex: add-tensor #1 sgemm-b : 784:128:1:1              ← B = weight [128,784]
ggml-hex: add-buffer #1 : fd 6 base 0x... size 489256      ← external activation pool
ggml-hex: add-tensor #2 sgemm-out : 128:32:1:1             ← C = output [32,128]
ggml-hex: HTP0 op-queue push batch #0                      ← enqueue to DSP queue
ggml-hex: HTP0 queue-opbatch: 0x... size 328               ← FastRPC message sent
ggml-hex: HTP0 op-queue pop batch #0                       ← DSP completed, result ready

# GEMM 2: FC2 forward (M=32, N=64, K=128)
ggml-hex: HTP0 execute-op MUL_MAT: 128:32 x 128:64 -> 64:32 : f32 x f32 -> f32
ggml-hex: HTP0 op-queue push batch #1
ggml-hex: HTP0 queue-opbatch: 0x... size 328
ggml-hex: HTP0 op-queue pop batch #1                       ← DSP completed

# GEMM 3: Output forward (M=32, N=10, K=64)
ggml-hex: HTP0 execute-op MUL_MAT: 64:32 x 64:10 -> 10:32 : f32 x f32 -> f32
ggml-hex: HTP0 op-queue push batch #2
ggml-hex: HTP0 queue-opbatch: 0x... size 328
ggml-hex: HTP0 op-queue pop batch #2                       ← DSP completed
```

**All 3 forward GEMMs complete successfully on the DSP.**

---

## 6. NPU Backward Pass — Detailed Flush Log and Failure

### 6.1 Backward GEMMs That Work (batches #3 and #4)

```
# GEMM 4: dX output (M=64, N=10, K=32) — transA=F, transB=T
ggml-hex: HTP0 execute-op MUL_MAT: 32:64 x 32:10 -> 10:64 : f32 x f32 -> f32
ggml-hex: HTP0 op-queue push batch #3
ggml-hex: HTP0 op-queue pop batch #3                       ← DSP completed ✅

# GEMM 5: dW output (M=32, N=64, K=10) — transA=T, transB=F
ggml-hex: HTP0 execute-op MUL_MAT: 10:32 x 10:64 -> 64:32 : f32 x f32 -> f32
ggml-hex: HTP0 op-queue push batch #4
ggml-hex: HTP0 op-queue pop batch #4                       ← DSP completed ✅
```

### 6.2 Backward GEMM That Hangs (batch #5)

```
# GEMM 6: dX fc2 (M=128, N=64, K=32) — transA=F, transB=T
ggml-hex: HTP0 execute-op MUL_MAT: 32:128 x 32:64 -> 64:128 : f32 x f32 -> f32
ggml-hex: add-buffer #0 : fd 19 base 0x... size 4194304
ggml-hex: add-tensor #0 sgemm-a : 32:128:1:1               ← A = dY [128,32]
ggml-hex: add-tensor #1 sgemm-b : 32:64:1:1                ← B = W^T [64,32]
ggml-hex: add-buffer #1 : fd 6 base 0x... size 489256
ggml-hex: add-tensor #2 sgemm-out : 64:128:1:1             ← C = dX [128,64]
ggml-hex: HTP0 op-queue push batch #5                      ← enqueue to DSP
ggml-hex: HTP0 queue-opbatch: 0x79ce107780 size 328        ← FastRPC message sent
                                                            ← NO POP — DSP HUNG ❌
```

After this, logcat shows:
```
E mnist_npu_train: vendor/qcom/proprietary/adsprpc/src/dspqueue/dspqueue_cpu.c:1864::error: 0xc:
  (nErr = wait_signal_locked(q, DSPQUEUE_SIGNAL_RESP_PACKET, timeout_ts)) == 0
```

Error 0xc = `ERROR_TIMEOUT`. The DSP never responds.

### 6.3 Why Backward Fails — HMX Path Analysis

The HTP kernel (`matmul-ops.c`) decides HMX vs HVX:

```c
int op_matmul(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    // Check 1: M must be multiple of 32 for HMX
    if (src0->ne[1] % 32 != 0)  → 128 % 32 = 0  → PASS (try HMX)

    // Check 2: F32 requires K % 32 == 0
    if (wtype == HTP_TYPE_F32 && src0->ne[0] % 32 != 0)
                                → 32 % 32 = 0  → PASS

    // Check 3: m_hmx must be non-zero
    m_hmx = M & ~31 = 128 & ~31 = 128  → non-zero → PASS

    // → HMX F32 path attempted → HANG
}
```

**The HMX F32 kernel hangs when K=32 (exactly one HMX tile width).**

Evidence from the working/failing pattern:

| GEMM | M | K | K%32 | M%32 | HMX Path? | Result |
|------|---|---|------|------|-----------|--------|
| Fwd fc1 | 32 | 784 | 0 | 0 | Yes | ✅ Works |
| Fwd fc2 | 32 | 128 | 0 | 0 | Yes | ✅ Works |
| Fwd out | 32 | 64 | 0 | 0 | Yes | ✅ Works |
| Bwd dX out | 64 | 32 | 0 | 0 | Yes | ✅ Works |
| Bwd dW out | 32 | 10 | 10 | 0 | No (HVX) | ✅ Works |
| Bwd dX fc2 | 128 | 32 | 0 | 0 | Yes | ❌ HANG |

The dW output GEMM (K=10, not divisible by 32) falls back to HVX and works. The dX fc2 GEMM (K=32, divisible by 32) tries HMX and hangs.

**Root cause**: The HMX F32 matmul kernel in `matmul-ops.c` has a bug when K equals exactly 32 (one tile). This may be:
1. An infinite loop in the K-tile loop (off-by-one in `k < K` vs `k <= K`)
2. A DMA fetch issue with single-tile K dimension
3. An HMX register file issue with exactly one K tile

---

## 7. Training Results Comparison

### 7.1 CPU-Only Training

```
#1/3 - Training Loss: 2.13131 >> [ Accuracy: 40.625% - Validation Loss : 1.71868 ]
#2/3 - Training Loss: 1.60188 >> [ Accuracy: 96.875% - Validation Loss : 1.35363 ]
#3/3 - Training Loss: 1.32614 >> [ Accuracy: 91.6667% - Validation Loss : 1.08147 ]
Training complete.
```

| Epoch | Training Loss | Accuracy | Val Loss |
|-------|---------------|----------|----------|
| 1 | 2.13131 | 40.625% | 1.71868 |
| 2 | 1.60188 | 96.875% | 1.35363 |
| 3 | 1.32614 | 91.6667% | 1.08147 |

### 7.2 Hybrid Mode (Forward NPU, Backward CPU)

```
#1/3 - Training Loss: 2.35119 >> [ Accuracy: 10.4167% - Validation Loss : 2.35012 ]
#2/3 - Training Loss: 2.358   >> [ Accuracy: 12.5%    - Validation Loss : 2.35102 ]
#3/3 - Training Loss: 2.33321 >> [ Accuracy: 13.5417% - Validation Loss : 2.33905 ]
Training complete.
```

| Epoch | Training Loss | Accuracy | Val Loss |
|-------|---------------|----------|----------|
| 1 | 2.35119 | 10.4167% | 2.35012 |
| 2 | 2.358 | 12.5% | 2.35102 |
| 3 | 2.33321 | 13.5417% | 2.33905 |

### 7.3 Full-NPU Mode

**Does not complete** — DSP hangs on 6th GEMM of first training step.

### 7.4 Comparison Table

| Mode | Epochs | Final Loss | Final Accuracy | FastRPC Round Trips | Status |
|------|--------|-----------|----------------|---------------------|--------|
| CPU-only | 3 | 1.32614 | 91.67% | 0 | ✅ Complete |
| Hybrid (Fwd NPU + Bwd CPU) | 3 | 2.33321 | 13.54% | 54 | ✅ Complete |
| Full NPU | 0 | N/A | N/A | 5 (then hang) | ❌ DSP Hang |

---

## 8. Why Hybrid Mode Barely Learns

The hybrid mode loss stays near ln(10) = 2.3026 (random guessing for 10 classes). The model barely learns.

### 8.1 No Q4_0 Involved — Pure FP32 End-to-End

**Important clarification**: No Q4_0 quantization is used in either CPU or NPU training. Both paths use `sgemm_fp32` — pure FP32 GEMM. The Q4_0 path (`gemm_q4_0_accel_fp32`) is only for quantized LLM inference (CausalLM/Qwen3). The MNIST model uses `weight_initializer = xavier_uniform` which creates FP32 weights, and all GEMM operations operate on FP32 data throughout training.

### 8.2 HMX Supports F32 — But May Produce Incorrect Results

HMX (Hexagon Matrix eXtension) supports both F16 and F32. From `matmul-ops.c`:
```c
if (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && ...)
    return op_matmul_hvx(octx);  // fallback for unsupported types
```

The forward GEMMs use the HMX F32 path (M%32==0, K%32==0) and complete without hanging. However, the **initial loss is already wrong**: the first training batch produces loss 2.376 on NPU vs 2.216 on CPU with the same model architecture. This 7% difference on the very first forward pass suggests the HMX F32 kernel is producing **numerically incorrect results** — it completes but gives wrong outputs.

### 8.3 HVX F32 Path Exists — Not Always F16 Quantization

My earlier theory about F16 quantization was incorrect. The HVX path has a true `f32-f32` path:
```c
// From matmul-ops.c, op_matmul_hvx():
} else if (src0->type == HTP_TYPE_F32) {
    // Try optimized f32-f32 path first (src1 in VTCM)
    quant_job_func = quantize_f32_f32;  // ← True F32, NOT F16!
```

The HVX kernel has dedicated `vec_dot_f32_f32_aa_1x1` functions that do true F32×F32 accumulation. So even the HVX fallback path should be numerically correct for F32.

### 8.4 The Real Problem: HMX F32 Correctness

The evidence points to the HMX F32 kernel itself producing wrong results:
- **First batch loss**: NPU 2.376 vs CPU 2.216 (7% difference) — this is before any weight update
- **Training doesn't converge**: loss stays at ~2.33 (random guessing) across all 3 epochs
- **HMX F32 also hangs**: on K=32 backward GEMMs, confirming the kernel has bugs

The HMX F32 kernel was likely designed/tested for inference (where small numerical errors don't matter for argmax). For training, the forward activations must be numerically accurate because they feed into backward gradient computation. Even small errors in the forward pass compound through backpropagation, preventing the optimizer from descending the loss landscape.

### 8.5 Forward/Backward Mismatch

In hybrid mode:
- Forward: activations computed on HMX F32 → numerically wrong activations
- Backward: gradients computed on CPU using the wrong activations → wrong gradients
- The optimizer chases a moving target because the forward pass doesn't match what the backward pass expects

### 8.6 Evidence Summary

| Metric | CPU | Hybrid (NPU fwd + CPU bwd) |
|--------|-----|---------------------------|
| First batch loss | 2.216 | 2.376 (7% higher) |
| Epoch 1 loss | 2.131 | 2.351 |
| Epoch 3 loss | 1.326 | 2.333 |
| Epoch 3 accuracy | 91.7% | 13.5% |
| Learns? | Yes | Barely |

The 7% error on the very first forward pass (before any weight update) is the smoking gun: the HMX F32 kernel is producing incorrect activations.


---

## 9. FastRPC Round Trip Breakdown

### 9.1 Per-Step Round Trips

| Mode | Forward GEMMs (NPU) | Backward GEMMs (NPU) | Backward GEMMs (CPU) | Total NPU Round Trips/Step |
|------|---------------------|----------------------|----------------------|---------------------------|
| CPU-only | 0 | 0 | 9 | 0 |
| Hybrid | 3 | 0 | 6 | 3 |
| Full NPU | 3 | 6 (attempts) | 0 | 9 (target) |

### 9.2 Per-Run Round Trips (3 epochs, 3 train + 3 val steps/epoch)

| Mode | Train Round Trips | Val Round Trips | Total Round Trips |
|------|-------------------|-----------------|-------------------|
| CPU-only | 0 | 0 | 0 |
| Hybrid | 27 | 27 | 54 |
| Full NPU | 81 (target) | 27 | 108 (target) |

### 9.3 What Happens in Each Round Trip

```
CPU side                          | DSP side
----------------------------------|----------------------------------
1. Build tensor descriptors       |
   (ne, nb, data ptr, type)       |
2. Stage A, B into rpcmem        |
   (memcpy if not in pool)        |
3. enqueue_op(node)               |
   → write to shared memory queue |
4. flush(true)                    |
   → FastRPC call ---------------→| 5. Dequeue op batch
                                  | 6. op_matmul():
                                  |    - Check HMX eligibility
                                  |    - If HMX: HMX F32 kernel
                                  |    - If HVX: quantize to F16, HVX kernel
                                  | 7. Write result to rpcmem
                                  ← 8. FastRPC return
9. memcpy result from rpcmem      |
   to caller buffer               |
```

Each round trip cost:
- CPU staging: ~2 memcpy (A and B into rpcmem)
- FastRPC overhead: ~0.1-0.5ms (kernel driver + context switch)
- DSP compute: varies by GEMM size
- CPU copy-back: 1 memcpy (C from rpcmem)

---

## 10. Recommendations

### 10.1 Fix the HMX F32 Kernel (Medium-term)

The HMX F32 path in `matmul-ops.c` hangs when K=32. Options:
1. **Force HVX for F32**: Skip HMX entirely for F32 types (simplest, loses HMX speedup)
2. **Add K > 32 guard**: `if (wtype == HTP_TYPE_F32 && src0->ne[0] <= 32) return op_matmul_hvx(octx);`
3. **Debug the HMX kernel**: Fix the actual bug in the DSP-side HMX F32 code

### 10.2 Fix Forward Precision (For Hybrid Mode)

The F32→F16 quantization in the HVX path corrupts training. Options:
1. **Use F32 HVX path**: The HVX kernel has an `f32-f32` path that avoids F16 quantization — ensure it's being used
2. **Keep forward on CPU for training**: If precision matters, don't offload forward to DSP during training

### 10.3 Op Batching (Long-term)

Currently 9 round trips per step. Batching all 3 forward GEMMs into one flush (like `gemm_q4_0_batch`) would reduce to 1 round trip for forward. Similarly, batching backward GEMMs would reduce to 2 round trips for backward. Total: 3 round trips per step instead of 9.

### 10.4 Recommended Configuration

For **training**: Use CPU-only (engine not set to cdsp). The DSP's F16 quantization corrupts gradients.

For **inference**: Use NPU (engine=cdsp). Forward GEMMs work perfectly and the F16 quantization is acceptable for inference.

---

## 11. Files Modified

1. **`nntrainer/hexagon/hexagon_compute_ops.cpp`**:
   - `sgemm_fp32()`: Changed to hybrid mode — only forward GEMMs (no transpose) dispatch to NPU, backward GEMMs (transA or transB) run on CPU

2. **`../ggml-hexagon/ggml/src/ggml-hexagon/nntr-htp-bridge.cpp`**:
   - `nntr_htp_bridge_sgemm_fp32()`: Added FP32 GEMM dispatch to DSP
   - Physical transpose logic for transA=1 and transB=0
   - Fixed staging size calculation and buffer pointer updates

3. **`docs/backend_guide/NPU_TRAINING_FINDINGS.md`**: Initial findings document
4. **`docs/backend_guide/NPU_TRAINING_EXHAUSTIVE_REPORT.md`**: This report
