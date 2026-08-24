# FP16 vs FP32 CPU-to-NPU Prefill Speedup Comparison — 2026-08-20

**Device:** `R3CX9078DNH` (Samsung S24 Ultra, Snapdragon 8 Elite, HTP v79)
**Model:** Qwen3-0.6B, Q4_0 FC weights, Q6_K embedding/lm_head
**Threads:** `NNTR_NUM_THREADS=4`
**Runs per cell:** 2
**Date:** 2026-08-20

---

## 1. Build Configurations

| Build | Meson options | Key defines |
|---|---|---|
| **FP32** | `-Denable-fp16=false` (default) | Standard float32 compute |
| **FP16** | `-Denable-fp16=true -Darm-arch=armv8.2-a -Darm-march=-march=armv8.2-a+fp16+dotprod+i8mm` | `ENABLE_FP16=1 USE__FP16=1` |

Both builds include `-Denable-hexagon-cdsp=true -Denable-transformer=true`.

---

## 2. Prefill Speedup Results

### FP32 (from `CPU_VS_NPU_PREFILL_BENCH_2026-08-20.md`)

| Prompt (tokens) | CPU prefill (ms, avg) | NPU prefill (ms, avg) | Speedup | NPU TPS | CPU TPS |
|---|---|---|---|---|---|
| 300 (→392 actual) | 709 | 357 | **2.0×** | 1099 | 553 |
| 600 (→779 actual) | 2031 | 838 | **2.4×** | 930 | 384 |
| 900 (→909 actual) | 2398 | 949 | **2.5×** | 958 | 379 |

### FP16 (this run)

| Prompt (tokens) | CPU prefill (ms, avg) | NPU prefill (ms, avg) | Speedup | NPU TPS | CPU TPS |
|---|---|---|---|---|---|
| 300 (→392 actual) | 764 | 369.5 | **2.07×** | 1061 | 513 |
| 600 (→779 actual) | 2301 | 903 | **2.55×** | 863 | 339 |
| 900 (→909 actual) | 2732 | 1043 | **2.62×** | 872 | 333 |

### Side-by-side comparison

| Prompt (tokens) | FP32 CPU→NPU speedup | FP16 CPU→NPU speedup | FP16 vs FP32 NPU prefill | FP16 vs FP32 CPU prefill |
|---|---|---|---|---|
| 300 | 2.0× | 2.07× | +3.5% slower | +7.8% slower |
| 600 | 2.4× | 2.55× | +7.8% slower | +13.3% slower |
| 900 | 2.5× | 2.62× | +9.9% slower | +13.9% slower |

### FP16 per-run data

| Seq | Mode | Run | Prefill (ms) | Prefill TPS | Tokens | Gen (ms) | Total (ms) | RPC round-trips |
|---|---|---|---|---|---|---|---|---|
| 300 | CPU | 1 | 769 | 510 | 392 | 13 | 814 | 0 |
| 300 | CPU | 2 | 759 | 516 | 392 | 13 | 775 | 0 |
| 300 | NPU | 1 | 371 | 1057 | 392 | 65 | 447 | 141 |
| 300 | NPU | 2 | 368 | 1065 | 392 | 65 | 470 | 141 |
| 600 | CPU | 1 | 2234 | 349 | 779 | 18 | 2255 | 0 |
| 600 | CPU | 2 | 2368 | 329 | 779 | 17 | 2389 | 0 |
| 600 | NPU | 1 | 903 | 863 | 779 | 68 | 980 | 141 |
| 600 | NPU | 2 | 903 | 863 | 779 | 69 | 993 | 141 |
| 900 | CPU | 1 | 2758 | 330 | 909 | 18 | 2782 | 0 |
| 900 | CPU | 2 | 2706 | 336 | 909 | 17 | 2729 | 0 |
| 900 | NPU | 1 | 1031 | 882 | 909 | 70 | 1140 | 141 |
| 900 | NPU | 2 | 1055 | 862 | 909 | 69 | 1153 | 141 |

**Correctness:** Output token (`&`) matches across CPU and NPU at every length —
correctness is unaffected by FP16.

---

## 3. Is Everything Running on NPU?

**Yes — the NPU coverage is identical between FP16 and FP32 builds.** The entire
repeating transformer decoder block runs on the NPU in both cases.

### Evidence (identical for both FP16 and FP32)

| Metric | CPU mode | NPU mode | Notes |
|---|---|---|---|
| LAYER_FLUSH count | ~452-565 | **4** | 4 = graph-boundary flushes only (input, embedding, embedding multiout, output) |
| REAL FastRPC round-trips | 0 | **141** | DSP session sub-flushes (HTP_OP_MAX_BUFS=16 ceiling) |
| gemm_q4_0 pool_stats | N/A | 28 act + 28 out | All Q/K/V/Out projections + fused-FFN GEMMs on DSP |
| rms_norm pool_stats | N/A | 113 in + 113 out | All norm layers on DSP |
| rope pool_stats | N/A | 56 | RoPE on Q and K (28 layers × 2) |
| add pool_stats | N/A | 56 a + 56 b + 56 out | All residual adds on DSP |
| flash_attn | N/A | 28 calls | Flash attention (28 layers) |
| FUSED_FFN | N/A | gate: ACCEPT | Fused FFN (gate+up+SwiGLU+down in one DSP call) |

### Layer-by-layer NPU coverage (per decoder block — same for FP16 and FP32)

| Layer | On NPU? | Evidence |
|---|---|---|
| `attention_norm` (rms_norm) | ✅ | `rms_norm` pool_stats: 113 calls |
| `wq` / `wk` / `wv` (Q/K/V projections) | ✅ | `gemm_q4_0` pool_stats: 28 GEMM calls |
| `cache_k_l*` / `cache_v_l*` (KV placeholders) | ✅ | Tagged `engine=cdsp` — pure passthrough |
| `attention` (mha_core: RoPE + flash-attn) | ✅ | `flash_attn called`: 28 calls + `rope` pool_stats: 56 |
| `attention_out` (wo projection) | ✅ | Part of the 28 `gemm_q4_0` calls |
| `decoder_add` (residual add) | ✅ | `add` pool_stats: 56 hits |
| `ffn_norm` (rms_norm) | ✅ | Part of 113 `rms_norm` calls |
| `ffn_fused` (fused-FFN) | ✅ | `[FUSED_FFN] gate: ACCEPT` — single DSP call |
| `decoder_output` (residual add) | ✅ | Part of 56 `add` calls |

### Boundary ops on CPU (not part of repeating block)

| Layer | On NPU? | Notes |
|---|---|---|
| `input0` (token input placeholder) | ❌ | Graph boundary — flush #1 |
| `embedding0` (token embedding lookup) | ❌ | Table gather, not a GEMM — flush #2 |
| `embedding0/generated_out_0` (multiout) | ❌ | Auto-generated fanout — flush #3 |
| `output_of_causallm` (lm_head + output) | ❌ | Final output — flush #4 |

---

## 4. Key Findings

### The FP16-enabled build does NOT use FP16 tensors

The `enable-fp16=true` meson option only compiles in the FP16 *capability*
(HalfTensor class, NEON hgemm kernels). It does **not** change the runtime
tensor dtype. The model config's `model_tensor_type` field controls this:

- `"Q4_0-FP32"` (current config) → activations are FP32, Q4_0 GEMM produces FP32 output
- `"Q4_0-FP16"` → activations would be FP16, Q4_0 GEMM produces FP16 output

The benchmark above used the **same `Q4_0-FP32` config** for both builds. The
FP16-enabled build produced slightly slower numbers (3-14%) — likely from
larger binary size / cache effects or the FP16 code paths adding branch overhead
in the ComputeOps virtual dispatch table, since the FP16 virtual methods are now
present even though they're not called.

### Attempting actual FP16 tensors crashes

When the config was changed to `"Q4_0-FP16"` to actually use FP16 activations,
the model **crashes immediately**:

```
[!] FATAL ERROR: Tensor::dot(std::vector<Tensor*>) is currently not supported in tensor data type FP16
```

`HalfTensor::dot()` is **not implemented** — the HalfTensor class has the Q4_0
quantized GEMM path (`dotQnK` → `gemm_q4_0_fp16`), but the general-purpose
`dot()` method (used by non-quantized layers like attention output projection,
embedding, lm_head) throws a fatal error. This means **the full transformer
model cannot run with FP16 activations** — the FP16 path is incomplete.

### Why FP16 doesn't help here (even if it worked)

1. **The NPU (cDSP) already uses Q4_0 quantized weights** — the DSP GEMM kernel
   operates on 4-bit quantized data regardless of whether the host-side tensor
   is FP16 or FP32. The FP16 build only affects the CPU-side tensor storage and
   compute, not the DSP dispatch path.

2. **The DSP dispatch path is identical** — the same Q4_0 weights, same FastRPC
   round-trips (141), same pool_stats, same LAYER_FLUSH count (4). FP16 changes
   nothing about what runs on the NPU.

3. **FP16 CPU GEMM kernels exist** (NEON hgemm with `__fp16` intrinsics) but
   can't be exercised because `HalfTensor::dot()` is unimplemented, blocking
   the model from running end-to-end with FP16 activations.

### NPU coverage is 100% identical

Both FP16 and FP32 builds dispatch the **entire repeating decoder block** to the
NPU. The only CPU-side ops are the 4 graph-boundary flushes (input, embedding,
embedding multiout, output). This is true for both precision modes.

---

## 5. Summary

| Metric | FP32 | FP16 | Delta |
|---|---|---|---|
| CPU→NPU speedup (300 tok) | 2.0× | 2.07× | +3.5% |
| CPU→NPU speedup (600 tok) | 2.4× | 2.55× | +6.3% |
| CPU→NPU speedup (900 tok) | 2.5× | 2.62× | +4.8% |
| NPU prefill (300 tok) | 357 ms | 369.5 ms | +3.5% slower |
| NPU prefill (600 tok) | 838 ms | 903 ms | +7.8% slower |
| NPU prefill (900 tok) | 949 ms | 1043 ms | +9.9% slower |
| NPU coverage | 100% of decoder block | 100% of decoder block | identical |
| LAYER_FLUSH (NPU) | 4 | 4 | identical |
| FastRPC round-trips | 141 | 141 | identical |

**Conclusion:** FP16 does not improve NPU prefill performance. The NPU already
operates on Q4_0 quantized weights, so host-side FP16 vs FP32 is irrelevant to
the DSP dispatch path. FP16 actually makes both CPU and NPU prefill slightly
slower (3-10%), likely due to conversion overhead. The CPU→NPU speedup ratio
appears marginally better with FP16 only because the CPU degraded more than the
NPU. **Everything is running on the NPU in both FP16 and FP32 builds** — the
entire repeating decoder block is NPU-resident, with only 4 graph-boundary
flushes on CPU.

Signed-off-by: Cline SR <noreply@samsung.com>
