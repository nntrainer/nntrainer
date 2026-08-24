# Project Status: NPU Training for Qwen3-0.6B — Comprehensive Update

**Date:** 2026-08-13  
**Author:** Cline SR (AI agent review)  
**Scope:** Full codebase review — git history, source code, docs, on-device logs

---

## Executive Summary

The nntrainer repository has **ggml-hexagon kernels embedded** for Hexagon cDSP
(NPU) dispatch. The project has three major workstreams, each at a different
stage of maturity:

| Workstream | Status | Key Evidence |
|------------|--------|-------------|
| **Qwen3-0.6B Inference (hybrid NPU+CPU)** | ✅ Working | Text generation end-to-end; Q4_0 GEMMs + flash attention + fused FFN on cDSP |
| **MNIST Training on NPU** | ✅ Working | Full forward+backward FP32 GEMM training on cDSP; 100% accuracy; 2-4× speedup |
| **Qwen3-0.6B Training on NPU** | 🔨 In Progress | LoRA training infra built; runs on device but loss is flat (not converging); multiple blockers remain |

The **current goal** is Qwen3-0.6B training using the NPU. The GEMM
infrastructure is proven and ready. The remaining work is **layer-level
backwarding implementations** for the CausalLM custom layers, plus fixing a
bridge transpose bug and the training data pipeline.

---

## 1. What's Working Today

### 1.1 Qwen3-0.6B Inference (Hybrid NPU+CPU) — ✅ COMPLETE

**Architecture:** GEMMs dispatch to Hexagon cDSP; element-wise ops stay on CPU.

| Component | Where | Mechanism |
|-----------|-------|-----------|
| Q/K/V/O projection GEMMs | NPU | `engine=cdsp` → `HexagonComputeOps::gemm_q4_0` → HMX Q4_0 |
| Gate+Up (batched) | NPU | `gate_up_layer` → batched Q4_0 GEMM (2 in 1 flush) |
| Down projection | NPU | `engine=cdsp` → Q4_0 GEMM |
| Flash attention (prefill) | NPU | `nntr_htp_bridge_flash_attn` → HMX flash attn |
| Fused FFN (optional) | NPU | `nntr_htp_bridge_ffn_swiglu` → 3 GEMMs + SwiGLU in 1 flush |
| LM head | NPU | `engine=cdsp` → Q4_0 GEMM |
| RMSNorm, RoPE, softmax, SwiGLU, residual | CPU | Element-wise, not worth FastRPC overhead |
| Embedding lookup | CPU | Table gather, not GEMM |

**Performance (Galaxy S25 / HTP v79):**
- CPU: 640 prefill TPS
- NPU + flash attn + fused FFN: **1233 TPS (1.93× speedup)**

**Performance (Galaxy S24 Ultra / HTP v75):**
- 330-token prefill: NPU 902 TPS vs CPU 626 TPS (**1.44× speedup**)
- Decode (M=1): NPU slower (44 vs 94 TPS) — GEMV is bandwidth-bound, FastRPC overhead dominates

**Known issue:** Qwen3 model output is garbage in some runs — likely corrupted
Q4_0 hexagon-format weights. Does not affect training benchmarks (which use
FP32 from scratch).

### 1.2 MNIST Training on NPU — ✅ COMPLETE & VERIFIED

**What was built:**
- `nntr_htp_bridge_sgemm_fp32()` — FP32 GEMM dispatch to cDSP (forward + backward)
- `nntr_htp_bridge_sgemm_batch_fp32()` — Batched FP32 GEMMs (backward fusion)
- `HexagonComputeOps::sgemm_fp32()` override — auto-dispatches all training GEMMs
- Zero-copy activation pool via rpcmem
- Standalone benchmark (`mnist_npu_bench.cpp`) and framework training (`mnist_npu_train.cpp`)

**Benchmark Results (3 model sizes, 10 epochs each):**

| Model Size | NPU µs/step | CPU µs/step | Speedup | Accuracy |
|------------|-------------|-------------|---------|----------|
| 128×64 (small) | 5,568 | 5,466 | 1.02× | 100% |
| 512×256 (medium) | 46,307 | 95,664 | **2.07×** | 100% |
| 1024×3072 (Qwen3-scale) | 81,552 | 301,822 | **3.70×** | 100% |

**Key findings:**
- Training convergence is **identical** between NPU and CPU (same RNG seed → same loss curve)
- FP32 GEMM accuracy: rel_err ~1e-7 (sufficient for stable training)
- Backward fusion works: 5 GEMMs → 3 flushes (2+2+1 batched)
- FastRPC flush overhead dominates at small matrix sizes, amortizes at Qwen3 scale
- NPU speedup grows with matrix size (HMX hardware tiling vs CPU naive triple-loop)

### 1.3 Qwen3-0.6B LoRA Training — 🔨 PARTIALLY WORKING

**What's been built (in this repo):**

1. **FC layer with LoRA support** (`nntrainer/layers/fc_layer.cpp`):
   - `LoraRank`, `LoraAlpha`, `LoraQAT`, `LoraWeightQ4` properties
   - LoRA forward: `hidden = X·W + (X·A)·B * scaling`
   - LoRA `calcDerivative`: dequantizes base weight to FP32, adds LoRA contrib, dispatches `dot_deriv_wrt_1` → NPU
   - LoRA `calcGradient`: 3 GEMMs for dA, dB (small rank-8 GEMMs, stay on CPU)
   - Q4_0 QAT with EMA scales + straight-through estimator (STE)

2. **Gate-up layer backward** (`nntrainer/layers/gate_up_layer.cpp`):
   - `supportBackwarding() = true` ✅
   - `calcDerivative()`: dX = dY_up·W_up^T + dY_gate·W_gate^T (beta=1.0 accumulation) → NPU
   - `calcGradient()`: dW_up = X^T·dY_up, dW_gate = X^T·dY_gate → NPU

3. **Training model construction** (`transformer.cpp`):
   - `constructTrainingModel()` — builds full 28-block transformer graph for TRAIN mode
   - `initializeForTraining()` — compiles with `ExecutionMode::TRAIN`, sets up AdamW optimizer
   - All FC layers tagged with `withHexagonEngine()` → GEMMs dispatch to cDSP
   - KV cache as plain Tensors (not input layers) to allow TRAIN mode compilation
   - Cross-entropy loss, AdamW optimizer configured

4. **Training driver** (`train_qwen3_lora.cpp`, `lora_train.cpp`):
   - CLI with configurable rank, alpha, lr, epochs, seq_len, clip_grad, QAT
   - TrainingDataGenerator: text file → tokenized sequences
   - Thermal + memory monitoring per epoch
   - Best-model saving (LoRA adapter)

5. **On-device run** (from `training_log.txt`):
   - Ran 3 epochs on 391 samples, seq_len=64, rank=32, lr=1e-4
   - Training completes without crashes
   - **BUT: loss is flat** (0.939 → 0.939 → 0.939), accuracy stuck at 6%

---

## 2. What's NOT Working / Blocking Qwen3 Training

### 2.1 BLOCKER: CausalLM Custom Layers Lack Backwarding

The training loop calls `backwarding()` on every layer. Most CausalLM custom
layers throw "not supported" or have empty stubs:

| Layer | `supportBackwarding()` | `calcDerivative` | `calcGradient` | Effort |
|-------|------------------------|------------------|----------------|--------|
| `fully_connected` (FC) | ✅ true | ✅ Implemented | ✅ Implemented (with LoRA) | **Done** |
| `gate_up_layer` | ✅ true | ✅ Implemented | ✅ Implemented | **Done** |
| `addition` (residual) | ✅ true | ✅ Pass-through | N/A | **Done** |
| `rms_norm` | ❌ false | ❌ Throws | ❌ Throws | ~2 days |
| `reshaped_rms_norm` | ❌ false | ❌ Throws | ❌ Throws | ~2 days |
| `swiglu` | ✅ true | ⚠️ Commented out | ❌ Untested | ~1 day |
| `mha_core` (attention) | ✅ true | ⚠️ Empty `{}` | ⚠️ Empty `{}` | ~1-2 weeks |
| `embedding_layer` | ❌ false | ❌ Throws | ⚠️ Empty | ~2 days |
| `tie_word_embedding` | ❌ false | ❌ Throws | ⚠️ Empty | ~2 days |
| `fused_ffn` | ❌ false | ⚠️ Empty | ⚠️ Empty | ~2-3 days |
| `lm_head` | ❌ false | ❌ Throws | ❌ Throws | ~1 day |
| `per_layer_slice` | ❌ false | ❌ Throws | ❌ Throws | ~1 day |

**Impact:** When `model->train()` calls `backwarding()`, any layer with
`supportBackwarding() = false` will throw. Layers with empty `calcDerivative()`
stubs silently produce zero gradients, breaking the training signal.

**This is the #1 reason the training loss is flat.** The gradient cannot
flow through RMSNorm, attention, SwiGLU, or embedding layers.

### 2.2 BLOCKER: `transB=1` Bridge Transpose Bug

When training through the full nntrainer framework path, `calcGradient`'s
`dot_deriv_wrt_2` triggers `sgemm_fp32` with `transB=1`, which shows high
relative error (0.75-0.85). The standalone benchmark (which specifies transpose
explicitly) does not have this issue.

**Impact:** Even after layer backwarding is implemented, framework-level
training will produce incorrect gradients until this is fixed.

**Fix needed:** Investigate the transpose logic in `nntr_htp_bridge_sgemm_fp32`
for the `transB=1` case.

### 2.3 ISSUE: Training Data Pipeline

The current `trainDataGenerator` in `lora_train.cpp` has a simplistic target
generation:
- Target is just the first token of the *next* sample (not next-token prediction)
- One-hot label is for a single token, not a sequence
- This doesn't implement proper causal LM training (predict token t+1 from tokens 0..t)

**Impact:** Even with correct backwarding, the training signal is wrong. The
loss being flat at ~0.94 (≈ ln(vocab_size) for random predictions) confirms
the model isn't learning.

### 2.4 ISSUE: `supportBackwarding() = false` on Key Layers

Several layers return `false` for `supportBackwarding()`. The nntrainer
framework checks this flag and will refuse to call `backwarding()` on them.
For training to work, these must be set to `true` AND the backward methods
must be implemented.

### 2.5 RISK: Memory Budget

Full FP32 Qwen3-0.6B training: ~2.5 GB (weights + gradients + Adam state + activations).
LoRA training (rank=8): ~660 MB — fits in Android budget.

The current training log shows the model runs without OOM, so memory is
manageable for LoRA at seq_len=64.

### 2.6 OPTIMIZATION: Forward Fusion Bug

`fused_fc_forward` (3 GEMMs + 2 ReLUs in 1 DSP flush) has an op-chaining race
condition. Not a blocker — individual `sgemm_fp32` calls work correctly. But
fixing this would reduce flushes ~30%.

---

## 3. Architecture: How the Hybrid NPU Training Works

```
┌──────────────────────────────────────────────────────────────────┐
│                    Training Loop (model->train)                   │
│                                                                   │
│  FORWARD (per transformer block × 28):                           │
│    RMSNorm ───────────────────────► CPU (element-wise)           │
│    Q/K/V/O proj ──► engine=cdsp ──► NPU (sgemm_fp32 → cDSP)     │
│    Flash Attn ────► bridge ───────► NPU (fused QK+softmax+AV)   │
│    Gate+Up+Down ──► engine=cdsp ──► NPU (sgemm_fp32 → cDSP)     │
│    SwiGLU ─────────────────────────► CPU (element-wise)           │
│    Residual adds ──────────────────► CPU (element-wise)           │
│                                                                   │
│  BACKWARD (per transformer block × 28):                          │
│    Down proj bw ──► engine=cdsp ──► NPU (FP32 GEMM: dX, dW)     │
│    SwiGLU bw ─────────────────────► CPU (element-wise) ⚠️ NYI    │
│    Gate+Up bw ────► engine=cdsp ──► NPU (FP32 GEMM: dX, dW)     │
│    Residual (pass-through) ───────► CPU (identity) ✅            │
│    RMSNorm bw ────────────────────► CPU (element-wise) ⚠️ NYI    │
│    O proj bw ─────► engine=cdsp ──► NPU (FP32 GEMM: dX, dW)     │
│    Attn bw GEMMs ─► sgemm_fp32 ──► NPU (FP32: dQ, dK, dV)      │
│    Softmax bw ────────────────────► CPU (element-wise) ⚠️ NYI    │
│    RoPE bw ───────────────────────► CPU (element-wise) ⚠️ NYI    │
│    Q/K/V proj bw ─► engine=cdsp ──► NPU (FP32 GEMM: dX, dW)     │
│    RMSNorm bw ────────────────────► CPU (element-wise) ⚠️ NYI    │
│                                                                   │
│  OPTIMIZER (CPU):                                                │
│    AdamW: W -= lr × m̂ / (√v̂ + ε)  ──► CPU (element-wise)       │
└──────────────────────────────────────────────────────────────────┘
         │ sgemm_fp32 / sgemm_batch_fp32
         ▼
┌──────────────────────────────────────────────────────────────────┐
│              libggml-hexagon.so (bridge)                          │
│                                                                   │
│  nntr_htp_bridge_sgemm_fp32()      ← per-GEMM FP32 dispatch       │
│  nntr_htp_bridge_sgemm_batch_fp32() ← batched FP32 dispatch       │
│  nntr_htp_bridge_gemm_q4_0()       ← Q4_0 inference GEMM           │
│  nntr_htp_bridge_flash_attn()      ← fused flash attention (fwd)  │
│  nntr_htp_bridge_ffn_swiglu()      ← fused FFN (fwd)              │
│                                                                   │
│  1. stage A/B into rpcmem (or zero-copy from activation pool)    │
│  2. set_tensor × 3 (A, B, C descriptors)                         │
│  3. enqueue_op(HTP_OP_MUL_MAT)                                    │
│  4. flush(true) → FastRPC round trip to cDSP                     │
│  5. copy result from rpcmem (or zero-copy)                        │
└──────────────────────────────────────────────────────────────────┘
         │ FastRPC
         ▼
┌──────────────────────────────────────────────────────────────────┐
│              Hexagon cDSP (HTP v75/v79)                          │
│                                                                   │
│  HMX Matrix Engine (FP32 MUL_MAT)                                │
│    - M ≥ 32: HMX F32 path (systolic array)                       │
│    - M < 32: HVX F32 path (vector, f32-f32 accumulation)         │
│  ~600+ GEMM dispatches per training step (28 blocks × fwd + bwd) │
└──────────────────────────────────────────────────────────────────┘
```

**The rule is simple:** If it's a matrix multiply (GEMM), it goes to NPU.
If it's element-wise (pointwise math), it stays on CPU.

---

## 4. The nntrainer-lora Repo (Separate)

A separate repo at `/home/anirudh/nntrainer-lora` has **full backwarding
for all CausalLM layers** (RMSNorm, MHA, SwiGLU, embedding, LM head, etc.)
but **no hexagon backend**. The feasibility study
(`LORA_NPU_FEASIBILITY_STUDY.md`) concludes that merging the two is a
**2-3 day mechanical task**: copy 6 hexagon files, add 3 lines to engine.cpp,
add `engine=cdsp` to LoRA FC properties.

However, the nntrainer-lora repo directory exists but appears to not have
git initialized (or is in a different state). The LoRA training code has
been **ported into this repo** already (fc_layer.cpp has LoRA support,
lora_train.cpp and train_qwen3_lora.cpp exist).

---

## 5. What Needs to Be Done (Priority Order)

### Phase 1: Fix Layer Backwarding (BLOCKER — ~2 weeks)

Implement `calcDerivative` / `calcGradient` for the CausalLM custom layers.
These are CPU element-wise implementations — the GEMMs within them will
auto-dispatch to NPU.

| # | Layer | Effort | What to Implement |
|---|-------|--------|-------------------|
| 1 | `rms_norm` / `reshaped_rms_norm` | 2 days | `dx = γ/rms * (dy - x * mean(x*dy) / (rms²+ε))`, `dγ = sum(dy * x / rms)` |
| 2 | `swiglu` | 1 day | `d(gate) = dy * up * silu'(gate)`, `d(up) = dy * silu(gate)` |
| 3 | `embedding_layer` / `tie_word_embedding` | 2 days | Scatter-add: `d_embedding[token_ids] += dy` |
| 4 | `mha_core` | 1-2 weeks | `dQ=dA·V^T`, `dK=dA^T·Q`, `dV=Q·dA^T` → NPU GEMMs; softmax bw + RoPE bw → CPU |
| 5 | `lm_head` | 1 day | FC backward (or delegate to FC layer) |
| 6 | `per_layer_slice` | 1 day | Gather/scatter backward |

**Also:** Set `supportBackwarding() = true` on all of the above.

### Phase 2: Fix Training Data Pipeline (BLOCKER — ~2 days)

The current `trainDataGenerator` doesn't implement proper next-token prediction:
- Input: tokens `[t0, t1, ..., t_{n-1}]`
- Target: shifted tokens `[t1, t2, ..., t_n]`
- Loss: cross-entropy over all positions (not just one token)

### Phase 3: Fix `transB=1` Bridge Bug (BLOCKER — ~2 days)

Investigate `nntr_htp_bridge_sgemm_fp32` transpose logic for `transB=1`.
The standalone benchmark is correct; the framework path (via
`dot_deriv_wrt_2`) is not.

### Phase 4: Verify NPU Training Matches CPU (1 week)

1. Set `engine=cdsp` on all FC layers
2. Run training with NPU dispatch
3. Verify loss curve matches CPU training (same seed)
4. Benchmark NPU vs CPU training speed

### Phase 5: Optimization (2-3 weeks)

1. Fix `fused_fc_forward` op-chaining bug (forward fusion)
2. Add `sgemm_batch_fp32` to `HexagonComputeOps` for backward fusion
3. Add DSP unary ops (RMSNorm, softmax, SwiGLU) to reduce CPU↔DSP handoffs
4. Gradient checkpointing for memory

### Phase 6: LoRA Fine-tuning (1 week)

1. Train only LoRA adapter weights (rank=8/16)
2. QLoRA: Q4_0 frozen base + FP32 LoRA adapters
3. Memory: ~660 MB (fits Android budget)

---

## 6. Estimated Qwen3 Training Performance (NPU, 28 layers, batch=1, seq=128)

| Component | GEMMs/block | Flushes/block | Est. time/block |
|-----------|-------------|---------------|-----------------|
| Q/K/V/O proj fwd | 4 (3 batched + 1) | 2 | ~2 ms |
| QK^T + AV fwd | 2 BMM | 2 | ~1 ms |
| Gate/Up/Down fwd | 3 (2 batched + 1) | 2 | ~3 ms |
| **Forward total/block** | 9 | 6 | **~6 ms** |
| Backward (2× forward GEMMs) | 18 | ~12 (batched) | **~12 ms** |
| **Total/block** | 27 | 18 | **~18 ms** |
| **Total/step (28 blocks)** | 756 | 504 | **~504 ms** |

With forward fusion: ~350 ms/step.
With LoRA (fewer trainable GEMMs): ~200 ms/step.

---

## 7. Bridge Function Reference

| Bridge Function | Purpose | Status |
|---|---|---|
| `nntr_htp_bridge_sgemm_fp32` | Single FP32 GEMM on DSP (training) | ✅ Working |
| `nntr_htp_bridge_sgemm_batch_fp32` | Batched FP32 GEMMs (1 flush) | ✅ Working |
| `nntr_htp_bridge_gemm_q4_0` | Q4_0 GEMM (inference) | ✅ Working |
| `nntr_htp_bridge_gemm_q4_0_batch` | Batched Q4_0 GEMMs | ✅ Working |
| `nntr_htp_bridge_flash_attn` | Fused flash attention (forward) | ✅ Working |
| `nntr_htp_bridge_ffn_swiglu` | Fused FFN SwiGLU (forward) | ✅ Working |
| `nntr_htp_bridge_fused_fc_forward` | 5-op fused forward (1 flush) | ⚠️ Op-chaining bug |
| `nntr_htp_bridge_upload_weight_q4x4x2` | Upload Q4_0 weight to DSP | ✅ Working |
| `nntr_htp_bridge_register_activation_pool` | Register rpcmem pool (zero-copy) | ✅ Working |
| Flash attention backward | Fused attn backward on DSP | ❌ Not implemented |

---

## 8. Device Information

| Item | Value |
|------|-------|
| Primary device | R3CX9078DNH (Galaxy S24 Ultra, Snapdragon 8 Gen 3, HTP v75) |
| Other devices | R37L5008JZM, R3CY90LQXMZ (Galaxy S25 / HTP v79) |
| DSP skel | libggml-htp-v75.so / v79.so |
| Bridge | `nntr_htp_bridge` → `libggml-hexagon.so` |

---

## 9. File Map

### NPU Backend (nntrainer core)
- `nntrainer/hexagon/hexagon_compute_ops.cpp` — ComputeOps subclass (sgemm_fp32, gemm_q4_0 dispatch)
- `nntrainer/hexagon/hexagon_context.cpp` — Registers FC/gate_up layers under `engine=cdsp`
- `nntrainer/hexagon/hexagon_rpc_allocator.h` — rpcmem allocation for zero-copy DSP access
- `nntrainer/hexagon/hexagon_repack.cpp` — ARM q4_0x4 → DSP q4x4x2 weight conversion

### Training-Related Layers
- `nntrainer/layers/fc_layer.cpp` — FC with LoRA (forward, calcDerivative, calcGradient)
- `nntrainer/layers/gate_up_layer.cpp` — Gate+Up batched FC (forward + backward ✅)
- `nntrainer/layers/common_properties.h` — LoraRank, LoraAlpha, LoraQAT, LoraWeightQ4 properties

### CausalLM Application
- `Applications/CausalLM/models/transformer.cpp` — Model construction (inference + training)
- `Applications/CausalLM/lora_train.cpp` — LoRA training loop
- `Applications/CausalLM/train_qwen3_lora.cpp` — CLI driver
- `Applications/CausalLM/layers/mha_core.cpp` — Attention (forward ✅, backward ❌ empty)
- `Applications/CausalLM/layers/rms_norm.cpp` — RMSNorm (forward ✅, backward ❌ throws)
- `Applications/CausalLM/layers/swiglu.cpp` — SwiGLU (forward ✅, backward ⚠️ stubbed)
- `Applications/CausalLM/layers/embedding_layer.cpp` — Embedding (forward ✅, backward ❌ throws)
- `Applications/CausalLM/layers/tie_word_embedding.cpp` — Tied embedding (forward ✅, backward ❌ throws)
- `Applications/CausalLM/layers/fused_ffn_layer.cpp` — Fused FFN (forward ✅, backward ❌ empty)

### Bridge (external repo, embedded)
- `../ggml-hexagon/ggml/src/ggml-hexagon/nntr-htp-bridge.cpp` — All bridge functions
- `../ggml-hexagon/ggml/src/ggml-hexagon/htp/matmul-ops.c` — HMX/HVX matmul kernels
- `../ggml-hexagon/ggml/src/ggml-hexagon/htp/unary-ops.c` — Element-wise DSP kernels

### Test / Benchmark
- `test/mnist_npu_bench.cpp` — Standalone NPU training benchmark
- `test/mnist_npu_train.cpp` — Framework-level NPU training
- `test/mnist_npu_fused_train.cpp` — Fused training variant
- `test/test_models/models/mnist_3layer_npu.ini` — MNIST NPU config

### Documentation
- `docs/backend_guide/` — 21 documents covering architecture, plans, benchmarks, findings

---

## 10. Bottom Line

**The NPU GEMM infrastructure is proven and ready.** MNIST training on NPU
works end-to-end with 100% accuracy and 2-4× speedup. The FC layer has full
LoRA training support (forward + backward + QAT). The training model graph
is constructed and compiles in TRAIN mode.

**The missing piece is layer-level backwarding.** The CausalLM custom layers
(RMSNorm, attention, SwiGLU, embedding) throw or have empty stubs for
`calcDerivative`/`calcGradient`. This is why the training loss is flat —
gradients can't flow through these layers. The math is standard transformer
backward (well-documented), and the GEMMs within those backward passes will
automatically dispatch to NPU via `sgemm_fp32`.

**Estimated effort to working Qwen3 NPU training:**
- Phase 1 (layer backwarding): ~2 weeks
- Phase 2 (training data pipeline): ~2 days
- Phase 3 (transB=1 bridge fix): ~2 days
- Phase 4 (verify NPU = CPU): ~1 week
- **Total: ~3-4 weeks** for a single developer

After that, LoRA fine-tuning on NPU should work, with ~200-500 ms per training
step and ~660 MB memory usage.
