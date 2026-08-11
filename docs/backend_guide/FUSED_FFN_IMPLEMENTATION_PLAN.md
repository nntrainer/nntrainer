# Fused FFN Kernel Implementation Plan

**Date:** 2026-08-04
**Status:** Implemented - Prefill working, decode needs optimization


---

## Current FFN Architecture (nntrainer)

```
Input (hidden_states)
  │
  ├─ gate_up_layer (DSP, Q4_0 batch GEMM) ─── FastRPC Call #1
  │   ├─ gate = W_gate · x
  │   └─ up = W_up · x
  │
  ├─ swiglu (CPU) ──────────────────────────── CPU compute
  │   └─ act = SiLU(gate) ⊗ up
  │
  └─ ffn_down (DSP, Q4_0 GEMM) ────────────── FastRPC Call #2
      └─ out = W_down · act
```

**Per layer:** 2 FastRPC calls + 1 CPU round-trip for SwiGLU
**For 28 layers:** 56 FastRPC calls + 28 CPU compute passes

---

## Proposed Fused FFN Architecture

```
Input (hidden_states)
  │
  └─ nntr_htp_bridge_ffn_swiglu (DSP, fused) ── FastRPC Call #1 (only 1!)
      ├─ gate = W_gate · x     (Q4_0 GEMM)
      ├─ up = W_up · x         (Q4_0 GEMM)
      ├─ act = SiLU(gate) ⊗ up (element-wise)
      └─ out = W_down · act    (Q4_0 GEMM)
```

**Per layer:** 1 FastRPC call, all on DSP
**For 28 layers:** 28 FastRPC calls (50% reduction)

---

## Implementation Steps

### Step 1: Add `nntr_htp_bridge_ffn_swiglu()` to ggml-hexagon

**File:** `ggml-hexagon/ggml/src/ggml-hexagon/nntr-htp-bridge.cpp`

```cpp
/**
 * @brief Fused FFN with SwiGLU activation.
 * 
 * Performs: output = W_down · (SiLU(W_gate · x) ⊗ (W_up · x))
 * 
 * All three Q4_0 GEMMs and the SwiGLU activation run on DSP in a single
 * FastRPC call, keeping intermediates in DSP scratch memory.
 *
 * @param input        Input activation [M, K] (FP16 or FP32)
 * @param W_gate       Gate weight [N, K] (Q4_0, ARM q4_0x4 layout)
 * @param W_up         Up weight [N, K] (Q4_0, ARM q4_0x4 layout)
 * @param W_down       Down weight [K, N] (Q4_0, ARM q4_0x4 layout)
 * @param output       Output [M, K] (FP16 or FP32)
 * @param M            Number of tokens (sequence length)
 * @param K            Hidden dimension (input/output)
 * @param N            Intermediate dimension
 * @param input_fp16   1 if input is FP16, 0 if FP32
 * @param output_fp16  1 if output is FP16, 0 if FP32
 * @return 0 on success, non-zero on failure
 */
extern "C" int nntr_htp_bridge_ffn_swiglu(
    const void * input,
    const void * W_gate,
    const void * W_up,
    const void * W_down,
    void * output,
    unsigned int M,
    unsigned int K,
    unsigned int N,
    int input_fp16,
    int output_fp16
);
```

**Implementation approach:**
1. Ensure all 3 weights are uploaded to DSP arena (reuse `ensure_uploaded` pattern)
2. Stage input if not in rpcmem
3. Enqueue 3 MUL_MAT ops (gate, up, down) + 1 SILU + 1 MUL
4. Single `flush()` call
5. Copy output back if needed

### Step 2: Add Bridge Function to SwiGLU Layer

**File:** `Applications/CausalLM/layers/swiglu.cpp`

Follow the same pattern as mha_core.cpp:

```cpp
using ffn_fn = int (*)(const void *, const void *, const void *,
                       const void *, void *, unsigned int, unsigned int,
                       unsigned int, int, int);

ffn_fn get_ffn_bridge() {
  static ffn_fn fn = []() -> ffn_fn {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) return nullptr;
    void *s = dlsym(handle, "nntr_htp_bridge_ffn_swiglu");
    return reinterpret_cast<ffn_fn>(s);
  }();
  return fn;
}

bool should_use_fused_ffn(unsigned int step_size, bool is_prefill) {
  static const char *env = std::getenv("NNTR_HEXAGON_FUSED_FFN");
  bool enabled = (env && std::atoi(env) == 1);
  if (!enabled) return false;
  if (!is_prefill || step_size <= 1) return false;
  
  static const char *min_env = std::getenv("NNTR_HEXAGON_FUSED_FFN_MIN_TOKENS");
  static const unsigned int min_tokens = min_env ? std::atoi(min_env) : 160;
  if (step_size < min_tokens) return false;
  
  return get_ffn_bridge() != nullptr;
}
```

### Step 3: Integrate at Transformer Level

**File:** `Applications/CausalLM/models/transformer.cpp`

The challenge is that `createMlp` creates separate layers (gate_up, swiglu, ffn_down).
The fused path needs to bypass all three layers.

**Option A: Conditional layer creation (recommended)**
```cpp
Tensor Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                               Tensor input) {
  // Check if fused FFN is enabled
  if (should_use_fused_ffn(/* step_size */)) {
    // Single fused layer that calls nntr_htp_bridge_ffn_swiglu
    LayerHandle fused_ffn(createLayer("fused_ffn", 
      withHexagonEngine({...})));
    return fused_ffn(input);
  }
  
  // Original path: gate_up → swiglu → ffn_down
  LayerHandle ffn_gateup(createLayer("gate_up_layer", ...));
  Tensor gateup_out = ffn_gateup(input);
  Tensor up = gateup_out.output(0);
  Tensor gate = gateup_out.output(1);
  LayerHandle swiglu(createLayer("swiglu", ...));
  Tensor act = swiglu({up, gate}, {1, 0});
  LayerHandle ffn_down(createLayer("fully_connected", ...));
  return ffn_down(act);
}
```

**Option B: Modify SwiGLU layer to accept all 3 weights**
- More invasive but doesn't require new layer type
- SwiGLU layer would need gate_up and down weights as inputs

### Step 4: Create New `FusedFFNLayer` (for Option A)

**File:** `Applications/CausalLM/layers/fused_ffn_layer.cpp` (new)

```cpp
void FusedFFNLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &input = context.getInput(0);
  Tensor &output = context.getOutput(0);
  Tensor &W_gate = context.getWeight(gate_idx);
  Tensor &W_up = context.getWeight(up_idx);
  Tensor &W_down = context.getWeight(down_idx);
  
  const ffn_fn &fn = get_ffn_bridge();
  int rc = fn(input.getData(), W_gate.getData(), W_up.getData(),
              W_down.getData(), output.getData(),
              input.height(), input.width(), W_gate.height(),
              is_fp16, is_fp16);
  
  if (rc != 0) {
    ml_logw("FusedFFN: DSP failed (%d), falling back to CPU", rc);
    // Fall back to CPU path
    cpu_ffn_forward(input, W_gate, W_up, W_down, output);
  }
}
```

### Step 5: Register New Layer

**File:** `Applications/CausalLM/layers/meson.build`
```meson
causallm_layers_src = [
  ...
  'fused_ffn_layer.cpp',
  ...
]
```

---

## Expected Performance Impact

| Metric | Current (gate_up + swiglu + down) | Fused FFN | Improvement |
|--------|-----------------------------------|-----------|-------------|
| FastRPC calls/layer | 2 | 1 | 50% |
| CPU↔DSP handoffs/layer | 4 | 2 | 50% |
| SwiGLU CPU time | ~5ms/layer | 0ms (on DSP) | 100% |
| Intermediate copies | 2 (gate_up→CPU, act→DSP) | 0 | 100% |

**Estimated prefill speedup:** +15-20% on top of flash attention

---

## Environment Variables

```bash
# Enable fused FFN (same pattern as flash attention)
export NNTR_HEXAGON_FUSED_FFN=1

# Minimum tokens to trigger (same as flash attention)
export NNTR_HEXAGON_FUSED_FFN_MIN_TOKENS=160

# Verbose logging
export NNTR_HEXAGON_FUSED_FFN_VERBOSE=1
```

---

## Fallback Path

Same as flash attention - if DSP fails or env var not set, original CPU path runs:
```
gate_up_layer → swiglu → ffn_down
```

No flexibility loss, same as flash attention pattern.

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `ggml-hexagon/.../nntr-htp-bridge.cpp` | Modify | Add `nntr_htp_bridge_ffn_swiglu()` |
| `Applications/CausalLM/layers/fused_ffn_layer.h` | Create | New layer header |
| `Applications/CausalLM/layers/fused_ffn_layer.cpp` | Create | New layer implementation |
| `Applications/CausalLM/layers/swiglu.cpp` | Modify | Add bridge function + gate |
| `Applications/CausalLM/models/transformer.cpp` | Modify | Conditional fused FFN path |
| `Applications/CausalLM/layers/meson.build` | Modify | Add new source file |

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| DSP kernel doesn't support 3 chained GEMMs | Start with 2 GEMMs (gate+up), keep SwiGLU+down on CPU |
| Weight upload overhead | Reuse existing `ensure_uploaded` (already cached) |
| Intermediate dimension too large for DSP scratch | Check VTCM size, fall back if needed |
| Correctness | Add `verify_fused_ffn` test (same as verify_flash_attn) |

---

## Benchmark Results (2026-08-04)

**Device:** Snapdragon 8 Gen 3 (SM8650)  
**Model:** Qwen3-0.6B (28 layers, hidden=1024, intermediate=3072)  
**Prompt:** 301 tokens (318 after tokenization)  
**Decode:** 128 tokens  

| Variant | Prefill TPS | Decode TPS | Total ms |
|---------|------------|-----------|---------|
| CPU (4 threads) | 642 | 78.2 | 2134 |
| NPU (CDSP, existing FC layers) | 946 | 79.0 | 1960 |
| NPU + Flash Attn | 1140 | 79.4 | 1896 |
| NPU + Fused FFN | 1043 | 47.6 | 2998 |
| **NPU + Flash Attn + Fused FFN** | **1247** | 47.7 | 2943 |


### Analysis

**Prefill:** Fused FFN achieves 1043 TPS — **1.10x faster** than NPU baseline
(946 TPS) and **1.62x faster** than CPU (642 TPS). The speedup comes from
reducing 2 FastRPC calls per layer to 1, and eliminating the CPU↔DSP
round-trip for SwiGLU activation.

**NPU + Flash Attn + Fused FFN** achieves the best prefill at **1247 TPS** —
combining both optimizations: flash attention fuses attention (Q·K^T + softmax
+ ·V) into one DSP kernel, and fused FFN fuses 3 GEMMs + SwiGLU into one DSP
call. Together they reduce FastRPC calls per layer from 4 to 2.

**Decode regression:** Fused FFN decode drops from 79.0 → 47.6 TPS.
**Why decode goes through DSP:** The weights are quantized in HEXAGON's
q4x4x2 layout (not ARM's q4_0x4), so CPU cannot dequantize them. The
existing NPU path's FC layers handle decode by using
`HexagonComputeOps::gemm_q4_0_accel_fp32` which dispatches to DSP even for
M=1 — but critically, it uses `gemm_q4_0_batch` to batch **all layers'**
GEMMs into **~2 FastRPC calls** per token.

**Root cause of slowdown:** Our fused FFN layer can't batch across layers
(each layer is independent), so it does **3 individual `gemm_q4_0` calls
per layer** = **84 FastRPC calls per token** (vs ~2 for the existing path).

**Timing breakdown:**
- NPU baseline: 12.7ms/token (2 batched FastRPC calls)
- Fused FFN:    21.0ms/token (84 individual FastRPC calls)
- Difference:   8.3ms/token = 82 extra calls × ~0.1ms/call

Each FastRPC round-trip costs ~0.1ms of overhead (kernel context switch +
rpcmem marshalling). The DSP GEMM compute itself is the same — it's purely
IPC overhead from not batching.



### Decode Optimization Path

To fix the decode regression, the decode path should either:
1. Use `gemm_q4_0_batch` to batch all 3 GEMMs across all 28 layers into one
   FastRPC call (requires bridge support for batched FFN)
2. Fall back to the existing FC layer path for decode (M=1) and only use
   the fused bridge for prefill (M>1) — this requires keeping the original
   gate_up/swiglu/ffn_down layers alongside the fused layer
3. Use the fused bridge for decode too (M=1) — the `ffn_swiglu` bridge
   handles M=1, but has ~100ms overhead per call, making it 10x slower
   than the existing path

**Current approach:** Option 3 is used (3 separate `gemm_q4_0` calls for
decode). This works correctly but is slower than the batched path.

