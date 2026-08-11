# NPU vs CPU Training Benchmark — 2026-08-10

## Device: R3CX9078DNH (Galaxy S24 Ultra, Snapdragon 8 Gen 3, HTP v75)

## Summary

NPU training (Hexagon cDSP via `nntr_htp_bridge`) is **2.0× faster** than CPU
at MNIST scale (128×64) and **4.9× faster** at Qwen3-0.6B scale (1024×3072).
All 4 scenarios achieve 100% test accuracy with identical weight init.

## Fusion Strategy

**Forward (3 GEMMs, 3 flushes):** Each GEMM must wait for ReLU output from the
previous layer (ReLU runs on CPU). So forward GEMMs cannot be batched across
ReLU — 3 individual `sgemm_fp32` calls.

**Backward (5 GEMMs, 3 flushes):** Sequential dependency chain
`dH2 → dY2(relu) → dH1 → dY1(relu) → dW1` forces 3 phases:
- Phase 1: `{dW3, dH2}` batched into 1 flush
- ReLU backward on CPU
- Phase 2: `{dW2, dH1}` batched into 1 flush
- ReLU backward on CPU
- Phase 3: `{dW1}` single flush

**Total: 6 flushes/step** (down from 8 unfused). Further reduction requires
ReLU on DSP (the `fused_fc_forward` op-chaining path, currently buggy).

## Results Table

| Model | Mode | Test Acc | Fwd µs/step | Bwd µs/step | Total µs/step | Inference µs/step |
|-------|------|----------|-------------|-------------|---------------|-------------------|
| 784→128→64→10 | NPU | 100% | 3,836 | 1,714 | **5,550** | 588 |
| 784→128→64→10 | CPU | 100% | 5,742 | 5,261 | **11,003** | 3,935 |
| 784→1024→3072→10 | NPU | 100% | 70,999 | 32,689 | **103,687** | 8,603 |
| 784→1024→3072→10 | CPU | 100% | 255,309 | 253,914 | **509,222** | 260,994 |

## Speedup Analysis

| Model | NPU µs/step | CPU µs/step | **Speedup** | GEMM FLOPs/step |
|-------|-------------|-------------|-------------|-----------------|
| 128×64 | 5,550 | 11,003 | **2.0×** | ~3.1M |
| 1024×3072 | 103,687 | 509,222 | **4.9×** | ~405M |

The speedup grows with matrix size because:
- **CPU** (naive triple-loop, no OpenBLAS): O(M×N×K) with no tiling → poor
  cache utilization, scales linearly with FLOPs
- **NPU** (Hexagon HTP v75): HW matrix accelerator with tiling + DMA → near-
  linear scaling but FastRPC flush overhead (~600 µs) becomes amortized at
  larger sizes

At Qwen3 scale (1024×3072), the NPU's per-GEMM compute time dominates the
FastRPC flush overhead, yielding ~5× speedup. At MNIST scale (128×64), flush
overhead is a larger fraction of total time, so speedup is only ~2×.

## Per-GEMM Latency Breakdown (Qwen3 scale, NPU)

| GEMM | M | N | K | Phase | Estimated µs |
|------|---|---|---|-------|-------------|
| FC1 fwd | 32 | 1024 | 784 | individual | ~22,000 |
| FC2 fwd | 32 | 3072 | 1024 | individual | ~44,000 |
| Out fwd | 32 | 10 | 3072 | individual | ~5,000 |
| dW3 + dH2 | 3072×10 + 32×3072 | — | — | batched (2 GEMMs, 1 flush) | ~16,000 |
| dW2 + dH1 | 1024×3072 + 32×1024 | — | — | batched (2 GEMMs, 1 flush) | ~14,000 |
| dW1 | 784 | 1024 | 32 | individual | ~2,700 |

## Training Convergence (identical for NPU and CPU — same RNG seed)

### MNIST-scale (128×64, 10 epochs)

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 2.483 | 22.9% |
| 2 | 1.697 | 51.0% |
| 3 | 1.309 | 85.4% |
| 4 | 0.979 | 100% |
| 10 | 0.135 | 100% |

### Qwen3-scale (1024×3072, 5 epochs)

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 2.271 | 29.2% |
| 2 | 0.649 | 76.0% |
| 3 | 0.143 | 97.9% |
| 4 | 0.040 | 100% |
| 5 | 0.011 | 100% |

## Key Takeaways

1. **Backward fusion works** — 3-phase batching (2+2+1) correctly respects the
   dH2→dY2→dH1→dY1 dependency chain while reducing 5 flushes to 3.

2. **NPU is 2-5× faster than naive CPU** at GEMM workloads, with the speedup
   increasing with matrix size as FastRPC overhead becomes amortized.

3. **Training convergence is identical** between NPU and CPU modes — the FP32
   GEMM accuracy (rel_err ~1e-7) is sufficient for training stability.

4. **Forward fusion still requires individual flushes** because ReLU between
   layers runs on CPU. Fusing forward GEMMs + ReLU into a single DSP flush
   (the `fused_fc_forward` path) would reduce to 2 flushes/step total.

5. **At Qwen3 scale, NPU step time is ~104 ms** (103,687 µs). For a 28-layer
   Qwen3-0.6B, one forward+backward pass (one transformer block) would take
   ~104 ms × 1 block = 104 ms. Full 28-layer training step ≈ 2.9 seconds.
   With forward fusion (3→1 flush), this could drop to ~2.1 seconds.
