# Qwen3 NPU Prefill Results — 2026-08-18

## Device
- **Model**: Samsung S24, Snapdragon 8 Gen 3
- **Serial**: R3CX9078DNH
- **NPU**: Hexagon HTP v79

## Model
- Qwen3-0.6B, Q4_0 FC weights + Q6_K embedding/lm_head
- 28 transformer layers, 16 attention heads, 8 KV heads, head_dim=128
- Quantized binary: `nntr_qwen3_0.6b_q40_hexagon.bin` (375MB)

## NPU Offload Coverage (Prefill)

All major prefill ops are now dispatched to NPU:

| Op | NPU Bridge Function | HTP Op | Status |
|----|-------------------|--------|--------|
| FC GEMM (Q4_0) | `nntr_htp_bridge_gemm_q4_0` | `HTP_OP_GEMM_Q4_0` | ✅ Zero-copy (rpcmem) |
| Flash Attention | `nntr_htp_bridge_flash_attn` | `HTP_OP_FLASH_ATTN` | ✅ Zero-copy |
| RMSNorm + Scale | `nntr_htp_bridge_rms_norm` | `HTP_OP_RMS_NORM_MUL` | ⚠️ Staging memcpy |
| RoPE (Q, K) | `nntr_htp_bridge_rope` | `HTP_OP_ROPE` | ⚠️ Staging memcpy |
| Residual ADD | `nntr_htp_bridge_add` | `HTP_OP_ADD` | ⚠️ Staging memcpy |
| SwiGLU / Fused FFN | `nntr_htp_bridge_ffn_swiglu` | `HTP_OP_FFN_SWIGLU` | ⚠️ Staging memcpy |
| Batch enqueue/flush | `begin_batch`/`flush`/`end_batch` | — | ✅ Single FastRPC round-trip |

## Benchmark Results (650-token prompt)

| Mode | Prefill (ms) | Prefill TPS | Gen TPS | Total (ms) | Peak Mem (KB) |
|------|-------------|-------------|---------|-----------|---------------|
| CPU-only (`NNTR_HEXAGON_DISABLE=1`) | 1502 | 432.8 | 62.5 | 1521 | 759332 |
| **Hybrid batch (default)** | **1570** | **414.0** | **58.8** | **1593** | **708832** |
| Hybrid no-batch (`NNTR_HEXAGON_NO_BATCH=1`) | 1562 | 416.1 | 58.8 | 1583 | 741680 |
| Hybrid no-elem-ops (`NNTR_HEXAGON_NO_ELEM_OPS=1`) | 1604 | 405.2 | 55.6 | 1625 | 732220 |

## Analysis

### What works
- **Full prefill pipeline runs on NPU** — all ops (GEMM, flash_attn, RMSNorm, RoPE, ADD, SwiGLU) dispatch to Hexagon HTP
- **Batch mode collapses ~196 FastRPC round-trips into 1** — `begin_batch()`/`end_batch()` wrapping works correctly
- **Memory savings**: NPU mode uses ~50MB less peak memory (708MB vs 759MB CPU)
- **Correctness**: Output token (`&`) matches across all modes

### Bottleneck: staging memcpy for element-wise ops
At 650 tokens, NPU is ~4.5% slower than CPU (1570ms vs 1502ms). Root cause:
1. **Only GEMM activations are zero-copy** (in rpcmem) — element-wise ops (RMSNorm, RoPE, ADD) use CPU heap memory
2. Each element-wise op pays 2 staging memcpys (host→rpcmem, rpcmem→host)
3. At 650 tokens, each activation is ~2.5MB; ~364 ops × 2 memcpys × 2.5MB = ~1.8GB memory traffic

### Path to NPU speedup
The fix (per PROJECT_STATUS_2026-08-17 analysis) is to route ALL tensor memory through rpcmem, matching ggml-hexagon's native design where every tensor is zero-copy. This would eliminate all staging memcpys and let the NPU's higher compute throughput dominate.

## Commands Used

```bash
# Default hybrid (batch mode, all ops on NPU)
adb -s R3CX9078DNH shell 'cd /data/local/tmp/nntrainer/causallm && \
  export LD_LIBRARY_PATH=/data/local/tmp/nntrainer/causallm:$LD_LIBRARY_PATH && \
  export NNTR_NUM_THREADS=4 && \
  ./nntrainer_causallm models/qwen3-0.6b'

# CPU-only
adb -s R3CX9078DNH shell 'cd /data/local/tmp/nntrainer/causallm && \
  export LD_LIBRARY_PATH=/data/local/tmp/nntrainer/causallm:$LD_LIBRARY_PATH && \
  export NNTR_NUM_THREADS=4 && \
  export NNTR_HEXAGON_DISABLE=1 && \
  ./nntrainer_causallm models/qwen3-0.6b'

# NPU GEMM+flash_attn only (element-wise on CPU)
adb -s R3CX9078DNH shell 'cd /data/local/tmp/nntrainer/causallm && \
  export LD_LIBRARY_PATH=/data/local/tmp/nntrainer/causallm:$LD_LIBRARY_PATH && \
  export NNTR_NUM_THREADS=4 && \
  export NNTR_HEXAGON_NO_ELEM_OPS=1 && \
  ./nntrainer_causallm models/qwen3-0.6b'
```

Signed-off-by: Anirudh <anirudh1023@gmail.com>
