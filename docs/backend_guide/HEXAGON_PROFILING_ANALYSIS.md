# Hexagon cDSP Profiling & Prefill Analysis

**Date:** 2026-08-03  
**Device:** Galaxy S25 (SM-S936U / R3CX9078DNH), Snapdragon 8 Elite, HTP v79  
**Model:** Qwen3-0.6B Q4_0  

---

## 1. What runs on NPU vs CPU during prefill (nntrainer)

Based on the observation log (§28) and code analysis, here is the exact split:

### On DSP (NPU) — 17% of prefill wall time
| Layer | ms | % of prefill | What it does |
|---|---|---|---|
| qkv_layer | 28.4 | 8.5% | Q/K/V batched Q4_0 GEMM (1 flush) |
| gate_up_layer | 32.3 | 9.6% | gate/up batched Q4_0 GEMM (1 flush) |
| fully_connected (attention_out + ffn_down) | 33.2 | 9.9% | Single Q4_0 GEMMs (2 flushes) |

**Total DSP path:** 58.4 ms (17% of 353 ms prefill)  
- DSP compute: 49.1 ms  
- Dispatch overhead: 9.4 ms (only 2.7% of total prefill)

### On CPU — 83% of prefill wall time
| Layer | ms | % of prefill | What it does |
|---|---|---|---|
| **mha_core (attention + RoPE + KV cache)** | **190.2** | **56.7%** | Q·K^T, softmax, scores·V, RoPE, KV write |
| swiglu | 17.6 | 5.3% | SwiGLU activation |
| tie_word_embeddings | 15.9 | 4.7% | Embedding lookup + LM head (Q6_K) |
| reshaped_rms_norm (q_norm/k_norm) | 8.5 | 2.5% | RMSNorm for Q/K |
| addition (residuals) | 4.5 | 1.3% | Residual adds |
| rms_norm | 4.4 | 1.3% | RMSNorm for attention/FFN |
| nntrainer per-layer overhead | ~35 | 9.9% | Tensor setup, getSharedDataTensor windowing |

### Key takeaway
**Attention is 57% of prefill and runs entirely on CPU.** The DSP only handles Q4_0 FC matmuls. Every layer, the data round-trips: DSP→CPU (for attention/norms/RoPE)→DSP (for next matmul).

---

## 2. FastRPC call batching — current state

### What's batched (§27)
- **Q/K/V:** 3 GEMMs → 1 flush (QKVLayer + `gemm_q4_0_batch_fp32`)
- **gate/up:** 2 GEMMs → 1 flush (GateUpLayer + `gemm_q4_0_batch_fp32`)

### What's NOT batched
- `attention_out` FC: 1 flush (no sharing partner)
- `ffn_down` FC: 1 flush (no sharing partner)

**Per layer:** 4 flushes carrying 7 GEMMs = **112 flushes per forward pass** (down from 196 unbatched).

### Why batching didn't help prefill (§27)
Dispatch overhead is only **2.7%** of prefill (9.4 ms out of 353 ms). Cutting round trips 20% (140→112) saved ~1.9 ms — unmeasurable against thermal noise. **Prefill is compute-bound, not dispatch-bound, past the ~215-token crossover.**

### Where batching matters: decode (§17)
Decode IS dispatch-bound: 196 ops × ~98 µs IPC = 19.2 ms/token of pure dispatch waste. Batching projections:

| Submissions/token | Bridge ms | Total ms | Decode t/s |
|---|---|---|---|
| 196 (today) | 24.7 | 33.0 | 30.4 |
| 112 (Q/K/V + gate/up) | 16.3 | 24.6 | ~41 |
| 28 (whole layer) | 8.0 | 16.3 | ~61 |
| 1 (whole forward pass) | 5.4 | 13.7 | ~73 |

But decode on DSP is a losing battle regardless (§7/§21): even at zero dispatch overhead, DSP compute ceiling is 56 t/s vs CPU's 158.9 t/s.

---

## 3. Why ggml-hexagon's prefill is 3.5× faster

### Measured comparison (this session, same device)
| Framework | CPU pp512 | NPU pp512 | NPU/CPU |
|---|---|---|---|
| **ggml-hexagon** (llama-bench) | 563.8 | **2083.3** | **3.69×** |
| **nntrainer** (CausalLM, §27) | ~590 | ~821 | **1.39×** |

### Three structural advantages ggml-hexagon has

#### A. Whole-graph offload (§19, §24)
ggml-hexagon runs **everything** on the DSP for single-sequence:
- All 7 FC MUL_MATs per layer → HTP0
- RMS_NORM, MUL (norm weight) → HTP0
- ROPE → HTP0
- **FLASH_ATTN** → HTP0 (fused kernel)
- SWIGLU, ADD (residual) → HTP0
- SET_ROWS (KV cache write) → HTP0

Only 2 ops stay on CPU: token embedding GET_ROWS (tiny) and LM-head MUL_MAT (VTCM size guard, N > 16K).

**nntrainer only offloads Q4_0 FC matmuls.** Everything else round-trips through CPU every layer.

#### B. Single-flush graph execution (§21, §24)
`ggml_backend_hexagon_graph_compute` (ggml-hexagon.cpp:3346-3395):
1. Walks the entire `ggml_cgraph`
2. Fuses RMS_NORM+MUL → one `HTP_OP_RMS_NORM_MUL` op
3. Enqueues ALL nodes
4. Calls `flush()` **exactly once** after the loop

**535 ops in ONE FastRPC round trip** (§21), with 1.79 µs overhead per op.

nntrainer's bridge: 112 flushes × ~100 µs each = 11.2 ms of dispatch overhead per forward pass. ggml-hexagon: ~1 ms.

#### C. DSP-resident KV cache + fused flash attention (§29, §30)
ggml-hexagon's `FLASH_ATTN_EXT` kernel:
- Reads Q/K/V from DSP-resident rpcmem
- Computes Q·K^T + softmax + scores·V in one fused kernel
- Uses HMX systolic array (head_dim=128 satisfies the fast path)
- No intermediate scores tensor, no CPU round trip

nntrainer's attention (CPU):
- Q·K^T: 68.0 ms (35.4% of attention)
- scores·V: 63.7 ms (33.2% of attention)
- RoPE + KV write: 50.1 ms (26.0%)
- softmax: 10.4 ms (5.4%)
- **Total: 192.2 ms = 55% of prefill**

---

## 4. FastRPC call reduction opportunities

### Already done
- [x] Pooled rpcmem arenas (§8): 18× decode improvement
- [x] OPPOLL busy-poll (§14): +20% decode
- [x] Q/K/V + gate/up batching (§25-27): 196→112 flushes
- [x] Zero-copy activations via HexagonRpcAllocator (§27)
- [x] KV cache in rpcmem (§30)

### Remaining opportunities (ranked by impact)

#### 1. Offload attention as FLASH_ATTN_EXT (§30) — **55% of prefill**
- DSP skel already implements `HTP_OP_FLASH_ATTN_EXT`
- KV cache already in rpcmem (§30)
- KV cache is already F16 (2 bytes/element) — satisfies the kernel's dtype requirement
- Tensor layouts map as pure stride permutations, no copies needed
- head_dim=128 hits the HMX fast path
- Needs: bridge entry point for 5 strided descriptors + op_params + causal mask
- **Projected: 1.39× → ~2.3× prefill** (attention→0 would give 2120 TPS, past ggml's 2083)

#### 2. Offload RMS_NORM + MUL as fused HTP_OP_RMS_NORM_MUL — **2.5% of prefill**
- DSP kernel exists
- But per-op offload would ADD round trips (§28 proved this)
- Only valuable if done as part of a larger contiguous DSP-resident op chain
- **Standalone impact: negligible (1.12× cap)**

#### 3. Offload RoPE via HTP_OP_ROPE — **14.4% of prefill**
- DSP kernel exists
- But RoPE lives inside mha_core.cpp (45 references), not independently offloadable
- Comes for free with FLASH_ATTN_EXT offload

#### 4. Offload SWIGLU via HTP_OP_GLU_SWIGLU — **5.3% of prefill**
- DSP kernel exists
- Only valuable in a contiguous op chain, not standalone

#### 5. Multi-op flush (beyond Q/K/V batching) — **decode only**
- Fuse entire layer (QKV + attention + norms + SWIGLU + FC) into one flush
- Would take decode from 112→28 flushes → ~61 t/s
- But decode on DSP loses to CPU regardless (§21)

### What's NOT worth doing
- **Further dispatch reduction for prefill**: dispatch is 2.7% of prefill (§28). Even infinite-fast dispatch only reaches 1.20×.
- **NPU decode**: hardware floor is 56 t/s (HVX-only, bandwidth-bound) vs CPU's 158.9 t/s (§21).
- **FP16 activations**: tested, no gain on this device (§30). The KV cache is already F16 regardless.

---

## 5. Profiling results (this session)

### llama-bench (ggml-hexagon reference)
```
| test  | CPU (ngl=0)  | NPU (ngl=99) | NPU/CPU |
|-------|-------------|-------------|---------|
| pp512 | 563.8 t/s   | 2083.3 t/s  | 3.69×   |
| tg128 | 157.4 t/s   | 34.5 t/s    | 0.22×   |
```

### nntrainer CausalLM (Q4_0, on-device)
```
| config              | prefill (18 tok) | decode (128 tok) |
|---------------------|-----------------|-----------------|
| MIN_ROWS=1 (all DSP)| 264.7 TPS       | 94.5 TPS        |
| MIN_ROWS=256 (hybrid)| 243.2 TPS      | 93.7 TPS        |
| FP32 (CPU only)     | 50.0 TPS        | 17.9 TPS        |
```

**Note:** The Q4_0 `.bin` on device has the §26 layout bug (old node order, pre-regeneration), producing garbage output. The FP32 `.bin` also produces garbage (same root cause). The timing numbers are still valid for profiling purposes since the computation path is exercised regardless of output correctness.

### Bridge profiler (§17, from observation log)
Per-op breakdown for decode (M=1, 25,088 ops):
| Phase | µs/op |
|---|---|
| Weight cache lookup | 0.0 |
| Staging alloc + activation memcpy in | 0.1 |
| Descriptor build + enqueue_op | 0.4 |
| **flush() — DSP round trip** | **125.3** |
| Result memcpy out | 0.2 |
| **Total** | **126.1** |

75% of decode time is dispatch overhead, not DSP compute.

### Prefill decomposition (§28, from observation log)
```
prefill wall time:      353 ms
  DSP path (196 GEMMs):   58.4 ms  (17%)
    - DSP compute:        49.1 ms
    - dispatch overhead:   9.4 ms  (2.7%)
  everything else:       294.6 ms  (83%)
    - attention:         190.2 ms  (57%)
    - other CPU ops:     104.4 ms  (26%)
```

---

## 6. Architecture comparison

```
ggml-hexagon (single flush, whole graph on DSP):
  ┌─────────────────────────────────────────────────┐
  │  enqueue ALL ops → flush() ONCE → wait          │
  │  535 ops, 1.79 µs overhead/op                  │
  │  attention = fused FLASH_ATTN_EXT on HMX        │
  │  KV cache DSP-resident, never touches CPU       │
  └─────────────────────────────────────────────────┘

nntrainer (per-op bridge, FC matmuls only on DSP):
  ┌──────┐     ┌──────┐     ┌──────────┐     ┌──────┐
  │ QKV  │────▶│ attn │────▶│ attn_out │────▶│ gate │
  │ DSP  │     │ CPU  │     │ DSP      │     │ DSP  │
  │1 flush│    │57%   │     │1 flush   │     │1 fl. │
  └──────┘     └──────┘     └──────────┘     └──────┘
       ↑                          ↑               ↑
       │ 112 flushes/forward pass │               │
       │ Each ~100µs IPC round trip               │
```

---

## 7. Recommended next steps

### Immediate (highest impact, code already exists)
1. **Implement `nntr_htp_bridge_flash_attn`** — the bridge entry point for `HTP_OP_FLASH_ATTN_EXT`
   - KV cache already in rpcmem (§30)
   - KV cache already F16 (satisfies kernel requirement)
   - Tensor layouts are stride permutations (§30)
   - `verify_flash_attn.cpp` test tool already written
   - **Expected: 1.39× → ~2.3× prefill**

### Medium term
2. **Contiguous op chains on DSP** — instead of per-op flush, enqueue multiple op types (RMS_NORM_MUL, ROPE, FLASH_ATTN_EXT, MUL_MAT, SWIGLU, ADD) before a single flush
   - This is what ggml-hexagon's `graph_compute` does
   - Requires a "begin batch / enqueue / flush" API in the bridge
   - Would eliminate the DSP→CPU→DSP round trip per layer

### Long term (structural)
3. **Graph-level offload** — give the bridge a view of multiple ops at once
   - The `ComputeOps` seam is per-tensor-op; it can't express "keep these ops resident"
   - Needs either a graph-level API or a "session flush" that defers flush until explicitly requested
   - This is the fundamental architectural gap (§24)

### Not recommended
- Further dispatch count reduction for prefill (dispatch is 2.7%)
- NPU decode (hardware floor is 56 t/s vs CPU 158.9 t/s)
- FP16 activations (no gain on this device)

---

## 8. Full Variant Benchmark (2026-08-05)

**Model:** Qwen3-0.6B Q4_0, 318-token prompt, 128 tokens generated  
**Device:** Galaxy S25 (SM-S936U), Snapdragon 8 Elite, HTP v79

### Run 1: DSP bridge decode (weights [N,K], DSP for both prefill+decode)

| Variant | Prefill TPS | Decode TPS | Total ms |
|---------|------------|-----------|---------|
| CPU (4 threads) | 626 | 77.4 | 2164 |
| NPU (CDSP) | 916 | 76.9 | 2021 |
| NPU + Flash Attn | 1161 | 78.0 | 1921 |
| NPU + Fused FFN | 978 | 65.7 | 2279 |
| **NPU + Flash Attn + Fused FFN** | **1237** | 64.9 | 2234 |

### Run 2: CPU decode with [K,N] weights (transpose at load time, CPU for decode)

| Variant | Prefill TPS | Decode TPS | Total ms |
|---------|------------|-----------|---------|
| CPU (4 threads) | 646 | 79.0 | 2115 |
| NPU (CDSP) | 933 | 77.7 | 1998 |
| NPU + Flash Attn | 1156 | 77.4 | 1936 |
| NPU + Fused FFN | 506 | 72.9 | 2389 |
| NPU + Flash Attn + Fused FFN | 505 | 75.7 | 2325 |

### Key findings

1. **Best prefill:** NPU + Flash Attn + Fused FFN (DSP bridge) at **1237 TPS**
2. **Best decode:** NPU + Flash Attn at **78–79 TPS** (consistent across runs)
3. **Best overall:** NPU + Flash Attn at **~1930 ms**
4. **Fused FFN with DSP bridge decode (Run 1):** prefill 978→1237 TPS (great),
   but decode drops to 65 TPS (3 individual DSP calls vs 2 batched for non-fused)
5. **Fused FFN with CPU decode (Run 2):** decode recovers to 73–76 TPS (close to
   non-fused 77 TPS), but prefill drops to 505 TPS because CPU dot() is used
   instead of the DSP bridge (bridge can't work with [K,N] weight layout)

### Decode regression root cause (Run 1)

The existing FC layers use `gemm_q4_0_batch_fp32` to batch all layers' GEMMs into
one FastRPC dispatch (2 calls per layer: gate_up batched + ffn_down). The fused FFN
layer's DSP bridge (`nntr_htp_bridge_ffn_swiglu`) does all 3 GEMMs + SwiGLU in one
call for prefill, but for decode (M=1) it still uses 3 individual `gemm_q4_0` calls
(84 FastRPC calls/token vs 2 batched for non-fused).

### CPU decode fix (Run 2)

To enable CPU decode, weights are stored as `[K,N]` (matching GateUpLayer) instead
of `[N,K]`. The `read()` method transposes Q4_0 weights at load time:
dequantize → transpose FP32 → re-quantize. This allows `dot(weight, false, false)`
with no transpose, which `dotQnK` supports. Decode runs on CPU at ~73–76 TPS.
However, the DSP fused bridge can't work with `[K,N]` weights (it expects `[N,K]`),
so prefill also runs on CPU at 505 TPS.

### Run 3: Dequantize-on-the-fly for decode (weights [N,K], FP32 dot for decode)

| Variant | Prefill TPS | Decode TPS | Total ms |
|---------|------------|-----------|---------|
| NPU + FlashAttn + FusedFFN | 129 | 30.2 | 6705 |

This approach keeps `[N,K]` weights and dequantizes Q4_0→FP32 at decode time,
using FP32 `dot(transpose=true)`. It's **extremely slow** (30 TPS decode) because
FP32 GEMM with transpose is 2.5x slower than Q4_0 `dot(false, false)`. Not viable.

### Final implementation: [N,K] weights with DSP bridge for prefill+decode

The shipped implementation keeps the original `[N,K]` weight layout (matching
the .bin file) and uses the DSP fused bridge for both prefill and decode.

**Final benchmark numbers (latest code, 2026-08-05):**

| Variant | Prefill TPS | Decode TPS | Total ms |
|---------|------------|-----------|---------|
| CPU (4 threads) | 640 | 77.3 | 2154 |
| NPU (CDSP) | 906 | 77.6 | 2011 |
| NPU + Flash Attn | 1136 | 77.4 | 1940 |
| NPU + Fused FFN | 961 | 67.9 | 2226 |
| **NPU + Flash Attn + Fused FFN** | **1233** | 66.3 | 2196 |

**Key takeaways:**
- **Best prefill:** NPU + Flash Attn + Fused FFN at **1233 TPS** (1.93x over CPU)
- **Best decode:** NPU + Flash Attn at **77.4 TPS** (flash attn doesn't affect decode)
- **Best overall:** NPU + Flash Attn at **1940 ms**
- **Fused FFN prefill gain:** 1136→1233 TPS (+8.5%) when combined with flash attn
- **Fused FFN decode cost:** 77.4→66.3 TPS (−14%) due to 3 individual DSP calls
  per token instead of 2 batched calls

The decode is slower than non-fused (66.3 vs 77.4 TPS) because the fused bridge
makes 3 individual `gemm_q4_0` calls per token instead of the 2 batched calls
that non-fused FC layers use. However, the prefill gain (1233 vs 1136 TPS) and
the architectural simplicity (single layer, no read() override) make this the
preferred implementation.

The CPU decode alternatives were explored but rejected:
- `[K,N]` + read() transpose: 75.7 TPS decode but only 507 TPS prefill (no DSP)
- Dequantize-on-fly: 30 TPS decode (FP32 GEMM too slow)





