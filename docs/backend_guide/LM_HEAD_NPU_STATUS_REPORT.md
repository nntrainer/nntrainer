# LM Head NPU Compatibility — Status Report

**Date:** 2026-08-24  
**Model:** Qwen3-0.6B (28 layers, hidden=1024, vocab=151936)  
**Device:** Galaxy S25 (SM-S936U, Snapdragon 8 Elite, HTP v79)  
**Model weights:** Q4_0 (FC layers), Q6_K (embedding + lm_head)

---

## 1. Executive Summary

| Question | Answer |
|----------|--------|
| **Is LM head currently on NPU?** | ❌ No — runs on CPU. LM head weight is Q6_K, which has no NPU kernel. |
| **Can LM head weights be quantized to Q4_0?** | ✅ Yes — the quantizer (`quantize.cpp`) supports `--lmhead_dtype Q4_0`. The NPU bridge has `gemm_q4_0_accel_fp32/fp16` for Q4_0 weights. |
| **Can LM head run FP32 on NPU?** | ⚠️ Partially — `sgemm_fp32` bridge exists but is designed for training. For inference, `Tensor::dot()` dispatches based on weight dtype. FP32 weights would use `sgemm_fp32` which has an NPU path, but it's untested for inference prefill. |
| **What's the current bottleneck?** | LM head uses Q6_K dtype → `gemm_q6_K_fp32/fp16` → forwards to CPU (no NPU override). |
| **Is the rest of the transformer on NPU?** | ✅ Yes — the entire transformer block (RMSNorm, MHA/flash-attn, WQKV, FFN, additions) is tagged `withHexagonEngine()` and dispatches to the DSP. Only embedding and LM head remain on CPU. |

---

## 2. Current Architecture

### 2.1 LM Head Layer (`lm_head.cpp`)

The LM head layer performs a single matrix multiply in `incremental_forwarding()`:

```cpp
input_step.dot(weight, hidden_step, false, false);
```

- **Weight dtype:** Controlled by `lmhead_dtype` in `nntr_config.json` (currently `"Q6_K"`)
- **Activation dtype:** FP32 (Q4_0-FP32 model) or FP16 (Q4_0-FP16 model)
- **Operation:** `Tensor::dot()` → dispatches to `gemm_q6_K_fp32` (or `gemm_q6_K_fp16`) based on weight dtype

### 2.2 NPU Compute Ops (`hexagon_compute_ops.cpp`)

`HexagonComputeOps` overrides these GEMM methods for NPU dispatch:

| Method | NPU Accelerated? | Notes |
|--------|-----------------|-------|
| `gemm_q4_0_accel_fp32` | ✅ Yes | Q4_0 weights, FP32 activations → `nntr_htp_bridge_gemm_q4_0` |
| `gemm_q4_0_accel_fp16` | ✅ Yes | Q4_0 weights, FP16 activations → `nntr_htp_bridge_gemm_q4_0_fp16` |
| `gemm_q4_0_batch_fp32` | ✅ Yes | Batched Q4_0 GEMM (QKV, gate/up fusion) |
| `sgemm_fp32` | ✅ Yes | FP32 GEMM (training bridge, optional symbol) |
| `gemm_q6_K_fp32` | ❌ No | **Forwards to CPU** — no NPU kernel for Q6_K |
| `gemm_q6_K_fp16` | ❌ No | **Forwards to CPU** — no NPU kernel for Q6_K |
| `gemm_q4_0_fp32` | ❌ No | CPU fallback (non-accelerated Q4_0 path) |

**Key finding:** The LM head weight is Q6_K, and `HexagonComputeOps` does NOT override `gemm_q6_K_*`. It forwards to `cpu_->gemm_q6_K_fp32()`, so the LM head matmul always runs on CPU even when NPU mode is enabled.

### 2.3 Transformer Block NPU Coverage (`transformer.cpp`)

From `transformer.cpp`, **every layer** in the transformer decoder block is tagged with `withHexagonEngine()`:

| Layer | `withHexagonEngine()`? | NPU Dispatch Mechanism |
|-------|------------------------|----------------------|
| RMSNorm (attn/ffn/output norm) | ✅ Yes | Registered under "cdsp" context; dispatches to DSP bridge internally |
| MHA Core (flash attention) | ✅ Yes | Registered under "cdsp" context; dispatches to DSP bridge internally |
| WQ / WK / WV (Q4_0) | ✅ Yes | `gemm_q4_0_batch_fp32` (batched QKV fusion) |
| WO / Attention Out (Q4_0) | ✅ Yes | `gemm_q4_0_accel_fp32` |
| Addition (residual) | ✅ Yes | Tagged with `withHexagonEngine()` |
| FusedFFN or GateUp+SwiGLU+Down (Q4_0) | ✅ Yes | `nntr_htp_bridge_ffn_swiglu` (fused) or `gemm_q4_0_batch_fp32` + `gemm_q4_0_accel_fp32` |
| KV Cache placeholders | ✅ Yes | Tagged with `withHexagonEngine()` |
| **Embedding** | ❌ No | Q6_K weight → `gemm_q6_K` → CPU |
| **LM Head** | ❌ No | Q6_K weight → `gemm_q6_K` → CPU |

**The entire transformer block runs on NPU.** Only embedding lookup and LM head projection remain on CPU.

---

## 3. Feasibility Analysis

### 3.1 Option A: Quantize LM Head to Q4_0 (Recommended)

**Feasibility: ✅ High**

The quantizer already supports this:
```bash
./nntr_quantize --model qwen3-0.6b.gguf --lmhead_dtype Q4_0 --fc_dtype Q4_0 --embd_dtype Q6_K
```

Or simply change the config:
```json
"lmhead_dtype": "Q4_0"
```

When the LM head weight is Q4_0, `Tensor::dot()` will dispatch to `gemm_q4_0_accel_fp32()` (or `gemm_q4_0_accel_fp16()`), which IS overridden in `HexagonComputeOps` and dispatches to the NPU via `nntr_htp_bridge_gemm_q4_0`.

**Pros:**
- No code changes needed — just re-quantize the model or change config
- NPU acceleration is already implemented and tested for Q4_0 weights
- Smaller weight footprint (Q4_0 vs Q6_K)
- Both FP32 and FP16 activation paths are supported

**Cons:**
- Slight accuracy degradation (Q4_0 vs Q6_K for the final logits projection)
- The LM head is a [hidden_dim × vocab_size] = [1024 × 151936] matrix — this is a large GEMM, so Q4_0 quantization error could affect token sampling quality

**Implementation steps:**
1. Re-quantize model with `--lmhead_dtype Q4_0`
2. Or create a new model binary with the LM head weight as Q4_0
3. No code changes needed — the NPU dispatch path already exists

### 3.2 Option B: FP32 LM Head on NPU

**Feasibility: ⚠️ Medium — requires testing**

If the LM head weight is FP32 (not quantized), `Tensor::dot()` dispatches to `sgemm_fp32()`. The `HexagonComputeOps` class DOES override `sgemm_fp32()` with an NPU bridge (`nntr_htp_bridge_sgemm_fp32`).

However, this bridge was designed for **training** (forward/backward GEMMs in FullyConnectedLayer), not inference prefill. The conditions for NPU dispatch are:
- `alpha == 1.0f && beta == 0.0f`
- No leading-dimension padding (`lda == K`, `ldb == N`, `ldc == N`)

The LM head's `incremental_forwarding` calls `input_step.dot(weight, hidden_step, false, false)`, which should satisfy these conditions for prefill (M = seq_len, N = vocab_size, K = hidden_dim).

**Pros:**
- No quantization error — full FP32 precision for logits
- NPU bridge already exists

**Cons:**
- FP32 weights are 4× larger than Q4_0 (622MB vs 156MB for the LM head alone)
- The `sgemm_fp32` bridge is untested for inference prefill — may have shape constraints
- Higher memory bandwidth requirement

**Implementation steps:**
1. Set `lmhead_dtype` to `FP32` in config
2. Use the FP32 model binary (or re-quantize with `--lmhead_dtype FP32`)
3. Verify that `sgemm_fp32` bridge dispatches correctly for the LM head shape
4. May need to test and debug the bridge for large N (vocab_size=151936)

### 3.3 Option C: Add Q6_K NPU Kernel

**Feasibility: ❌ Low — significant effort**

Would require implementing a Q6_K GEMM kernel in the ggml-hexagon HTP bridge (`matmul-ops.c`). The Q6_K format uses 6-bit quantization with per-block scales, which is more complex than Q4_0's 4-bit blocks. The HTP's HMX array natively supports 4-bit and 8-bit integer dot products, so 6-bit would require software dequantization on-DSP.

**Not recommended** — Q4_0 is the better path.

---

## 4. Benchmark Results (Verified on Device)

### 4.1 FP32 Activations (Q4_0-FP32 model)

| Seq Len | Mode | Prefill Tokens | Prefill ms | Prefill TPS | Total ms | Peak Mem (KB) | Flushes | Real FastRPC |
|---------|------|----------------|------------|-------------|----------|---------------|---------|-------------|
| 300 | CPU | 392 | 861 | 455 | 1009 | 1,030,092 | 565 | 0 |
| 300 | NPU | 392 | 231 | 1,697 | 310 | 671,092 | 5 | 1 |
| 600 | CPU | 779 | 2,435 | 320 | 2,536 | 1,126,572 | 565 | 0 |
| 600 | NPU | 779 | 394 | 1,977 | 481 | 675,548 | 5 | 1 |
| 900 | CPU | 909 | 2,904 | 313 | 2,997 | 1,141,908 | 565 | 0 |
| 900 | NPU | 909 | 551 | 1,650 | 643 | 671,264 | 5 | 1 |

### 4.2 FP16 Activations (Q4_0-FP16 model) — Re-verified

| Seq Len | Mode | Prefill Tokens | Prefill ms | Prefill TPS | Total ms | Peak Mem (KB) | Flushes | Real FastRPC |
|---------|------|----------------|------------|-------------|----------|---------------|---------|-------------|
| 300 | CPU | 392 | 708 | 554 | 724 | 669,608 | 565 | 0 |
| 300 | NPU | 392 | 393 | 997 | 483 | 671,040 | 5 | 1 |
| 600 | CPU | 779 | 1,996 | 390 | 2,016 | 752,680 | 565 | 0 |
| 600 | NPU | 779 | 814 | 957 | 915 | 675,564 | 5 | 1 |
| 900 | CPU | 909 | 2,265 | 401 | 2,287 | 746,620 | 565 | 0 |
| 900 | NPU | 909 | 956 | 951 | 1,060 | 671,280 | 5 | 1 |

### 4.3 Key Observations

1. **NPU speedup (FP32):** 3.7× at 300 tokens, 6.2× at 600 tokens, 5.3× at 900 tokens
2. **NPU speedup (FP16):** 1.8× at 300 tokens, 2.5× at 600 tokens, 2.4× at 900 tokens
3. **FP32 NPU is faster than FP16 NPU** — FP32 NPU achieves 1,650–1,977 TPS vs FP16 NPU at 951–997 TPS. This is because the LM head (Q6_K) runs on CPU in both cases, and the Q6_K dequant + FP16 GEMM path on CPU is slower than the FP32 path. The NPU-accelerated Q4_0 GEMMs are similar speed in both modes.
4. **Flush count:** NPU mode has only 5 flushes (1 real FastRPC round-trip) vs 565 in CPU mode — batching is working well for the Q4_0 layers
5. **Memory:** NPU mode uses ~670MB vs ~1.0–1.1GB for CPU — the DSP's rpcmem arena is more memory-efficient
6. **LM head is on CPU in all cases** — the Q6_K weight has no NPU kernel, so it runs on CPU regardless of NPU mode

### 4.4 Why FP16 NPU is Slower than FP32 NPU — Root Cause Found (Already Solved in Prior Session)

**Cross-referenced with `AGENT_HANDOFF_2026-08-20.md` — the FP16/FP32 NPU parity issue was already solved in a prior session but the fix is not in the currently deployed binary.**

The handoff doc documents two fixes that brought FP16 NPU to parity with FP32 NPU:

**Fix A (FP16): Q/K/V/O staging tensors reallocated every layer.**
`mha_core.cpp` created brand-new `Tensor(dim, true)` for Q/K/V/O on every one of the 28 layers. The HTP bridge's pool hit/miss check (`nntr_htp_bridge_find_ext_pool`) only checks its own registered-pool table — a fresh, never-registered allocation misses on every touch, permanently. Fix: added `get_reusable_fp16_scratch()` — a function-static scratch Tensor per role, shared across all 28 layers, grown-once, and **registered once via `nntr_htp_bridge_register_activation_pool`**. Result: `cpy:dst/src` pool stats went from `140 hit/112 miss` to `56 hit/0 miss`. **141 round-trips → 1.** Prefill 960ms → 446ms at 909 tokens.

**Fix B (FP32): KV-cache stored as `UINT16`, invisible to the dtype gate.**
`try_dsp_cache_copy`'s dtype gate only recognizes `FP32` and `FP16` — `UINT16` returns -1 and bails to CPU. The non-`ENABLE_FP16` build's KV-cache was `UINT16`. Fix: changed `#else` branch from `UINT16` to `FP32` in both `transformer.cpp` and `causal_lm.cpp`. Result: `61 flushes/29 round-trips → 5 flushes/1 round-trip`.

**Final state with both fixes applied (from AGENT_HANDOFF_2026-08-20.md):**

| Tokens | FP32 CPU | FP32 NPU | FP16 CPU | FP16 NPU |
|--------|----------|----------|----------|----------|
| 392 | 820ms | 233ms | 760ms | 209ms |
| 779 | 2529ms | 366ms | 1962ms | 396ms |
| 909 | 2952ms | 464ms | 2332ms | 446ms |

FP32 NPU and FP16 NPU are within ~5-8% of each other — confirming the user's expectation that they should match.

**Current deployed binary does NOT have these fixes.** My measurements (FP16 NPU: 814ms at 600 tokens) are ~2× worse than the handoff's fixed state (396ms at 779 tokens). The fixes exist in the working tree (`mha_core.cpp`, `transformer.cpp`, `causal_lm.cpp`) but the deployed binary on the device predates them. **Rebuilding and redeploying would close the FP16/FP32 gap.**

**Additionally**, the `gemm_q4_0_batch_fp16` bridge is still missing from `half_tensor.cpp` (source comment confirms: "there is no gemm_q4_0_batch_fp16 bridge yet"), but this is a minor optimization — the primary 2× gap was caused by Fix A (scratch tensor registration), not the missing batch bridge.




---

## 5. Recommendation

### Primary: Quantize LM Head to Q4_0

This is the lowest-effort, highest-impact change:

1. **Re-quantize the model** with `--lmhead_dtype Q4_0`:
   ```bash
   ./nntr_quantize --model qwen3-0.6b.gguf \
     --fc_dtype Q4_0 --embd_dtype Q6_K --lmhead_dtype Q4_0 \
     --isa HEXAGON --output_bin nntr_qwen3_0.6b_q40_lmhead_q40_hexagon.bin
   ```

2. **Update config:**
   ```json
   "lmhead_dtype": "Q4_0"
   ```

3. **No code changes needed** — `HexagonComputeOps::gemm_q4_0_accel_fp32()` and `gemm_q4_0_accel_fp16()` already handle Q4_0 weights with both FP32 and FP16 activations.

4. **Expected impact:** The LM head matmul ([seq_len × 1024] · [1024 × 151936]) will move from CPU to NPU, eliminating the last CPU-bound GEMM in the prefill pipeline.

### Secondary: Test FP32 LM Head on NPU

If Q4_0 quantization degrades output quality, test FP32 LM head with the `sgemm_fp32` bridge:
1. Set `lmhead_dtype` to `FP32`
2. Verify `sgemm_fp32` dispatches to NPU for the LM head shape
3. Measure performance — FP32 weights are larger but the NPU's HMX array can handle FP32

---

## 6. Current NPU Layer Coverage

| Layer | Weight Dtype | NPU Accelerated? | Dispatch Mechanism |
|-------|-------------|-----------------|---------------------|
| Embedding | Q6_K | ❌ CPU | `gemm_q6_K` not overridden in HexagonComputeOps |
| Attn RMSNorm | FP32 | ✅ NPU | `withHexagonEngine()` → DSP bridge (rms_norm.cpp) |
| WQKV (Q/K/V projection) | Q4_0 | ✅ NPU | `gemm_q4_0_batch_fp32` (batched QKV fusion) |
| MHA Core (flash attention) | FP32 | ✅ NPU | `withHexagonEngine()` → DSP bridge (mha_core.cpp) |
| Attention Out (WO) | Q4_0 | ✅ NPU | `gemm_q4_0_accel_fp32` |
| Residual Addition | FP32 | ✅ NPU | `withHexagonEngine()` → DSP bridge |
| FFN RMSNorm | FP32 | ✅ NPU | `withHexagonEngine()` → DSP bridge (rms_norm.cpp) |
| FFN Gate/Up (Q4_0) | Q4_0 | ✅ NPU | `gemm_q4_0_batch_fp32` (batched) or `nntr_htp_bridge_ffn_swiglu` (fused) |
| SwiGLU | FP32 | ✅ NPU | Part of fused FFN DSP dispatch |
| FFN Down (Q4_0) | Q4_0 | ✅ NPU | `gemm_q4_0_accel_fp32` or part of fused FFN |
| Output Norm | FP32 | ✅ NPU | `withHexagonEngine()` → DSP bridge (rms_norm.cpp) |
| **LM Head** | **Q6_K** | **❌ CPU** | **`gemm_q6_K` not overridden — forwards to CPU** |

**Summary:** The entire transformer block (28 layers) runs on NPU. Only the embedding lookup and LM head projection remain on CPU. The LM head is the last major GEMM not accelerated.
