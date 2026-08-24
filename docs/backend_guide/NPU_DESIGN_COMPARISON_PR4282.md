# NPU Design Comparison: Local Fork vs PR #4282 (Hvx impl)

**Date:** 2026-08-24  
**PR:** [nntrainer/nntrainer#4282](https://github.com/nntrainer/nntrainer/pull/4282) — "Hvx impl" by dlwlzzero  
**Local fork:** Anirudh1023/nntrainer (ggml-hexagon bridge approach)

---

## 1. Executive Summary

The two implementations take **fundamentally different architectural approaches** to NPU acceleration on Hexagon cDSP:

| Aspect | Local Fork (ggml-hexagon bridge) | PR #4282 (Hvx impl) |
|--------|----------------------------------|---------------------|
| **Dispatch granularity** | Per-op FastRPC (batched via begin/end) | Per-graph (one RPC per token) |
| **DSP codebase** | ggml-hexagon (external repo, `nntr-htp-bridge.cpp`) | Native HVX kernels in nntrainer tree |
| **Weight format** | Q4_0 / Q6_K (gguf-style block quant) | int8 W8A8 (per-token dynamic quant) |
| **Activation format** | FP32 or FP16 | FP16 (with int8 dynamic quant for matmul) |
| **Buffer management** | rpcmem pool with registration + cache-copy | dma-buf fd handoff, persistent HAP_mmap |
| **LM head** | ❌ CPU (Q6_K, no NPU kernel) | ✅ NPU (`MATMUL_LOGITS` op, int8) |
| **Embedding** | ❌ CPU (Q6_K) | ✅ NPU (`EMBED` op, int8 gather + dequant) |
| **Build integration** | ndk-build with prebuilt .so | Meson `enable-hexagon` option + QAIC codegen |
| **Maturity** | Running on device, benchmarked | Infrastructure + sim tests, device end-to-end NYI |

---

## 2. Architecture Comparison

### 2.1 Local Fork: Per-Op FastRPC with ggml-hexagon Bridge

```
nntrainer (C++)
  └─ HexagonComputeOps (overrides GEMM methods)
       └─ dlsym("nntr_htp_bridge_gemm_q4_0") → FastRPC to DSP
            └─ ggml-hexagon skel (external repo)
                 └─ HMX matmul kernels (Q4_0/Q6_K dequant + GEMM)
```

**How it works:**
- `HexagonComputeOps` overrides `gemm_q4_0_accel_fp32/fp16`, `gemm_q4_0_batch_fp32`, `sgemm_fp32` — each makes a FastRPC call to the DSP
- `begin_batch` / `end_batch` markers let the DSP batch multiple GEMMs into one round-trip
- rpcmem pool registration (`nntr_htp_bridge_register_activation_pool`) avoids per-call buffer allocation
- `try_dsp_cache_copy` gates KV-cache copies by dtype (FP32/FP16 only)
- Flash attention, fused FFN, RoPE, RMSNorm each have their own DSP bridge functions

**Strengths:**
- ✅ **Running on device** — benchmarked at 735ms (FP16) / 394ms (FP32) for 600-token prefill
- ✅ **Q4_0 weight quantization** — 4-bit block format, well-tested in llama.cpp ecosystem
- ✅ **FP16 activation support** — full pipeline including KV-cache
- ✅ **Batched dispatch** — 1 real FastRPC round-trip per prefill (via begin/end batching)
- ✅ **No SDK dependency at build time** — prebuilt .so, ndk-build only

**Weaknesses:**
- ❌ **LM head on CPU** — Q6_K has no NPU kernel, `gemm_q6_K` forwards to CPU
- ❌ **Embedding on CPU** — same Q6_K issue
- ❌ **External dependency** — ggml-hexagon is a separate repo, not in nntrainer tree
- ❌ **Per-op dispatch model** — even with batching, the host still makes N separate dlsym/FastRPC calls
- ❌ **No graph-level optimization** — no op fusion, no persistent buffer mapping across ops
- ❌ **Q4_0 only for NPU** — Q6_K, Q8_0, FP32 weights all fall back to CPU

### 2.2 PR #4282: Graph-Level Dispatch with Native HVX Kernels

```
nntrainer (C++)
  └─ HexagonRunner (host)
       └─ init(oplist, weights_fd) → HAP_mmap persistent
       └─ forward(token_ids) → logits (one RPC per call)
            └─ DSP: htp_graph executor
                 └─ 8 HVX op kernels (QuRT worker pool)
```

**How it works:**
- Host lowers the model to a flat op-list (64-byte descriptors) + packed weight buffer
- `init()` hands off weights/KV/activation buffers as dma-buf fds → DSP maps them with `HAP_mmap` once
- `forward()` carries only `token_ids` in and `logits` out — **one RPC per token**
- DSP-side `htp_graph` validates the op-list once, then dispatches ops sequentially through a QuRT worker pool
- 8 op kinds cover the full qwen3 decoder: EMBED, RMSNORM, MATMUL_W8A8, ROPE, ATTN, SILU_MUL, ADD, MATMUL_LOGITS

**Strengths:**
- ✅ **LM head on NPU** — `MATMUL_LOGITS` op handles last-token int8 matmul → fp32 logits
- ✅ **Embedding on NPU** — `EMBED` op does int8 row gather + dequant → fp16
- ✅ **Full graph offload** — entire decoder layer on DSP, one RPC per token
- ✅ **Persistent buffer mapping** — weights/KV/activations mapped once at init, zero-copy
- ✅ **Native to nntrainer** — all code in-tree, no external repo dependency
- ✅ **W8A8 dynamic quantization** — per-token int8 quant for matmul, better accuracy than static Q4_0
- ✅ **Proper build integration** — meson `enable-hexagon` option, QAIC codegen, sim tests
- ✅ **VTCM + user-DMA** — on-chip memory for hot kernels, DMA for weight streaming
- ✅ **QuRT worker pool** — one worker per HVX unit, barrier-synchronized
- ✅ **Comprehensive test suite** — sim golden tests for every op, on-device round-trip test

**Weaknesses:**
- ❌ **Not yet running end-to-end on device** — "device end-to-end wiring is planned work" (section 7)
- ❌ **No prefill support** — designed for autoregressive decode (M=1), `max_chunk` suggests chunked prefill is planned but not implemented
- ❌ **Requires Hexagon SDK** — proprietary, build-time dependency
- ❌ **int8 W8A8 only** — no Q4_0/Q6_K support, requires re-quantization from gguf formats
- ❌ **No FP32 weight path** — all matmuls are int8×int8, no FP32 fallback for sensitive layers

---

## 3. LM Head NPU Compatibility — Direct Comparison

| Feature | Local Fork | PR #4282 |
|---------|-----------|----------|
| **LM head on NPU?** | ❌ No (Q6_K → CPU) | ✅ Yes (`MATMUL_LOGITS` op) |
| **Weight format** | Q6_K (6-bit block quant) | int8 (W8A8 dynamic quant) |
| **Output dtype** | FP32/FP16 | FP32 logits |
| **Implementation effort** | Zero code changes (just re-quantize to Q4_0) | Built into the graph executor |
| **Accuracy** | Q4_0: 4-bit (lossy), Q6_K: 6-bit (better) | int8 W8A8 (best, dynamic per-token) |

**Key insight:** PR #4282 solves the LM head NPU problem by design — `MATMUL_LOGITS` is a first-class op. The local fork would need Q4_0 re-quantization (already supported) or a new Q6_K NPU kernel (not implemented).

---

## 4. Quantization Format Comparison

| Format | Local Fork | PR #4282 |
|--------|-----------|----------|
| **Weight quant** | Q4_0 (4-bit, 32-element blocks) | int8 (W8A8, per-channel scales) |
| **Activation quant** | None (FP32/FP16) | int8 (per-token dynamic quant) |
| **LM head** | Q6_K (6-bit, on CPU) | int8 (on NPU via MATMUL_LOGITS) |
| **Embedding** | Q6_K (on CPU) | int8 (on NPU via EMBED) |
| **Dequant location** | DSP-side (in HMX kernel) | DSP-side (in HVX kernel) |
| **Accuracy** | Q4_0: moderate, Q6_K: good | W8A8: best (8-bit weights + 8-bit activations) |

**Trade-off:** Q4_0 gives 4-bit weight compression (smaller model file) but requires dequant on every GEMM. W8A8 uses 8-bit weights (larger) but the int8×int8 GEMM is natively supported by HMX/HVX without dequant overhead.

---

## 5. Dispatch Model Comparison

### Local Fork: Per-Op with Batching
```
Host: begin_batch()
      gemm_q4_0(WQ)  → FastRPC
      gemm_q4_0(WK)  → FastRPC
      gemm_q4_0(WV)  → FastRPC
      flash_attn()   → FastRPC
      gemm_q4_0(WO)  → FastRPC
      ffn_swiglu()   → FastRPC
      end_batch()    → 1 real round-trip
```
- N dlsym calls per layer, but DSP batches them into 1 round-trip
- Each op is a separate FastRPC method call (even if batched)
- Buffer registration reduces allocation overhead but doesn't eliminate marshalling

### PR #4282: Graph-Level
```
Host: init(oplist, weights_fd)  → one-time setup
      forward(token_ids)         → one RPC, DSP runs entire graph
```
- 1 RPC per token (decode) or per chunk (prefill, planned)
- DSP executes the entire op-list sequentially with no host involvement
- Buffers are persistently mapped — zero marshalling per call

**Advantage of graph-level:** For autoregressive decode (M=1), the per-op overhead dominates. PR #4282 eliminates it entirely. For prefill (M>1), the per-op model is less penalized because GEMMs are large enough to amortize RPC overhead.

---

## 6. Maturity & Integration

| Aspect | Local Fork | PR #4282 |
|--------|-----------|----------|
| **On-device benchmarks** | ✅ Yes (735ms FP16, 394ms FP32 at 600 tokens) | ❌ Not yet (sim tests only) |
| **Prefill support** | ✅ Yes (full seq_len prefill) | ❌ Planned (max_chunk in wire format) |
| **Decode support** | ⚠️ Partial (per-token, not optimized) | ✅ Designed for decode (1 RPC/token) |
| **Unit tests** | ⚠️ Integration tests only | ✅ Comprehensive sim golden tests |
| **Build system** | ndk-build + prebuilt | Meson `enable-hexagon` + QAIC |
| **SDK dependency** | None (prebuilt .so) | Hexagon SDK 6.3.0+ required |
| **Code location** | External repo (ggml-hexagon) + nntrainer/hexagon/ | All in nntrainer/tensor/hexagon/ |
| **CI ready** | ❌ No (external dependency) | ⚠️ Partial (sim tests, no device tests in CI) |

---

## 7. Recommendation

### For LM Head NPU Compatibility (immediate need)

**Local fork approach (fastest path):**
1. Re-quantize LM head to Q4_0: `--lmhead_dtype Q4_0`
2. No code changes — `gemm_q4_0_accel_fp32/fp16` already exists
3. LM head moves from CPU to NPU automatically
4. Risk: Q4_0 accuracy for logits projection (vocab=151936 is large)

**PR #4282 approach (better long-term):**
1. `MATMUL_LOGITS` is already designed and has a sim test
2. int8 W8A8 gives better accuracy than Q4_0
3. But: device end-to-end is NYI, no prefill support yet

### For long-term architecture

PR #4282 is the **better architecture**:
- Graph-level dispatch eliminates per-op overhead
- Native in-tree code, no external dependency
- Full NPU coverage (including LM head + embedding)
- W8A8 quantization is more accurate than Q4_0
- Proper build integration with meson + QAIC
- Comprehensive test suite (sim golden tests)

**However**, the local fork is **what works today**:
- Running on device with real benchmarks
- Full prefill support (not just decode)
- FP16 activation pipeline
- No SDK dependency for builds

### Suggested path forward

1. **Short-term:** Use the local fork with Q4_0 LM head re-quantization to get LM head on NPU immediately
2. **Medium-term:** Contribute the local fork's prefill + FP16 + batching experience to PR #4282's graph-level architecture
3. **Long-term:** Migrate to PR #4282's design once it has device end-to-end + prefill support

The two approaches are **complementary**, not competing:
- Local fork: prefill-optimized, per-op dispatch, Q4_0 weights, running today
- PR #4282: decode-optimized, graph-level dispatch, W8A8 weights, proper architecture

A hybrid approach could use the local fork's ggml-hexagon bridge for prefill (large GEMMs where per-op overhead is negligible) and PR #4282's graph executor for decode (small GEMMs where per-op overhead dominates).
