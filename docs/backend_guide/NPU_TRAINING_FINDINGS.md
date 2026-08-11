# NPU Training Findings — FP32 GEMM on Hexagon cDSP

## Summary

**FP32 training GEMMs cannot run on the NPU (Hexagon cDSP / HTP v79).** Both
available F32 matmul paths on the DSP produce incorrect results:

| Path    | Relative Error | Hangs on K=32 | Notes                          |
|---------|---------------|---------------|--------------------------------|
| HMX F32 | ~7%            | Yes           | Wrong results + deadlock       |
| HVX F32 | ~100%          | No            | Completely uncorrelated output |

**Decision:** `sgemm_fp32` in `HexagonComputeOps` falls back to CPU. Only Q4_0
inference GEMMs go to the NPU. Training (forward + backward) runs on CPU.

## Evidence

### CPU baseline (works)
```
#1/3 - Training Loss: 2.13131 >> [ Accuracy: 40.625% ]
#2/3 - Training Loss: 1.60188 >> [ Accuracy: 96.875% ]
#3/3 - Training Loss: 1.32614 >> [ Accuracy: 91.6667% ]
```

### NPU full-training (HVX F32 path — broken)
```
#1/3 - Training Loss: 2.32903 >> [ Accuracy: 10.4167% ]
#2/3 - Training Loss: 2.32486 >> [ Accuracy: 10.4167% ]
#3/3 - Training Loss: 2.31898 >> [ Accuracy: 10.4167% ]
```

### Per-GEMM correctness check (NPU vs CPU reference)
Added a debug check in `nntr_htp_bridge_sgemm_fp32` that computes a CPU
reference and compares. **Every single GEMM has ~100% relative error:**

```
sgemm_fp32 debug[0]: M=32 N=128 K=784 tA=0 tB=0 max_err=3.16 max_val=1.84 rel_err=1.72
sgemm_fp32 debug[1]: M=32 N=64  K=128 tA=0 tB=0 max_err=2.03 max_val=2.03 rel_err=1.00
sgemm_fp32 debug[2]: M=32 N=10  K=64  tA=0 tB=0 max_err=1.50 max_val=1.50 rel_err=1.00
sgemm_fp32 debug[3]: M=64 N=10  K=32  tA=1 tB=0 max_err=1.18 max_val=1.18 rel_err=1.00
sgemm_fp32 debug[4]: M=32 N=64  K=10  tA=0 tB=1 max_err=1.64 max_val=1.65 rel_err=1.00
sgemm_fp32 debug[5]: M=128 N=64 K=32  tA=1 tB=0 max_err=2.18 max_val=2.18 rel_err=1.00
sgemm_fp32 debug[6]: M=32 N=128 K=64  tA=0 tB=1 max_err=0.46 max_val=0.45 rel_err=1.00
sgemm_fp32 debug[7]: M=784 N=128 K=32 tA=1 tB=0 max_err=1.15 max_val=1.09 rel_err=1.05
```

The NPU output is completely uncorrelated with the correct result — `rel_err=1.0`
means the error equals the signal magnitude.

## Root Cause Analysis

### HMX F32 path (matmul-ops.c)
- The HMX (Hexagon Matrix Extension) F32 kernel produces ~7% relative error
  on small matrices and **hangs indefinitely** when K=32 (the backward pass
  dimension for the last FC layer).
- HMX is designed for quantized types (Q4_0, Q8_0). The F32 path was added but
  has correctness bugs in the HMX tile configuration.

### HVX F32 path (matmul-ops.c)
- The `vec_dot_f32_f32_aa_1x1` kernel itself has correct arithmetic (verified
  by code review — simple FMA loop).
- The **dispatch infrastructure** in `op_matmul_hvx` does not properly handle
  F32 inputs. The function was written for quantized types (Q4_0, Q8_0) and
  the F32 path was bolted on without testing.
- The `quantize_f32_f32` function copies F32 data as-is (no quantization), but
  the thread work distribution, DMA setup, or scratchpad layout in
  `op_matmul_hvx` is wrong for F32 — producing completely garbage output.

### Bridge transpose logic (nntr-htp-bridge.cpp)
- The bridge's transpose logic is **correct** — verified by the CPU reference
  check matching the bridge's intended computation. The bug is in the DSP
  kernel, not the bridge.

## Current State

### What works on NPU
- **Q4_0 inference GEMM** (`gemm_q4_0_accel_fp32`): verified correct, used by
  CausalLM for Qwen3-0.6B text generation.
- **Q4_0 batched GEMM** (`gemm_q4_0_batch_fp32`): Q/K/V and gate/up fusion.
- **Flash attention** (`nntr_htp_bridge_flash_attn`): F16 attention on cDSP.
- **Fused FFN SwiGLU** (`nntr_htp_bridge_ffn_swiglu`): 5-op fusion on cDSP.

### What does NOT work on NPU
- **FP32 SGEMM** (`sgemm_fp32`): both HMX and HVX paths produce wrong results.
  Falls back to CPU.

### Training architecture
```
Forward pass:  CPU (sgemm_fp32 → OpenBLAS)
Backward pass: CPU (sgemm_fp32 → OpenBLAS)
Inference:     NPU (gemm_q4_0 → HMX Q4_0 kernel, verified correct)
```

## Files Changed

1. `nntrainer/hexagon/hexagon_compute_ops.cpp` — `sgemm_fp32` falls back to CPU
   with explanatory comment. The `nntr_htp_bridge_sgemm_fp32` symbol is still
   dlsym'd (for future use) but not called.

2. `../ggml-hexagon/ggml/src/ggml-hexagon/nntr-htp-bridge.cpp` — debug
   correctness check added (first 20 calls compare NPU vs CPU reference).

3. `../ggml-hexagon/ggml/src/ggml-hexagon/htp/matmul-ops.c` — HVX F32 path
   forced (bypasses HMX F32 hang), but the HVX dispatch is still broken.

## Next Steps (if NPU training is needed)

1. **Fix `op_matmul_hvx` for F32**: The dispatch infrastructure needs to be
   audited. The `quantize_f32_f32` function copies data correctly, but the
   thread work distribution (`nrows_per_thread`), DMA setup (`hex_l2fetch`
   parameters), and scratchpad layout in `op_matmul_hvx` may assume quantized
   block sizes.

2. **Alternative: FP16 training**: The DSP has a verified-correct FP16 HMX
   path. If nntrainer's FP16 mode is enabled, `sgemm_fp16` could be dispatched
   to the NPU instead. This would require adding a `nntr_htp_bridge_sgemm_fp16`
   bridge function.

3. **Alternative: Mixed precision**: Keep forward on NPU (Q4_0 inference is
   correct), do backward on CPU. This was tried but the Q4_0 forward pass
   introduces ~1% quantization error per layer, which accumulates through
   backprop and prevents learning.
