# Implementation Plan: Offload Attention to DSP via `nntr_htp_bridge_flash_attn`

**Goal:** Replace the 3-step CPU attention (compute_kcaches → softmax_triangle → compute_fp16vcache_transposed) with a single `HTP_OP_FLASH_ATTN_EXT` dispatch through the bridge, eliminating 57% of prefill wall time.

**Expected result:** Prefill from ~821 TPS (1.39×) to ~2000+ TPS (~2.5×+), approaching ggml-hexagon's 2032 TPS.

---

## Current State (as of this profiling session)

### Profiling results (live, on R3CX9078DNH / S25, 4 threads)

**ggml-hexagon reference (llama-bench, Qwen3-0.6B-Q4_0.gguf, 3 repeats):**

| test | CPU (ngl=0) | NPU (ngl=99) | NPU/CPU |
|---|---|---|---|
| pp512 | 565.72 ± 14.19 | **2032.38 ± 41.53** | 3.59× |
| tg128 | 155.27 ± 1.07 | 34.48 ± 0.10 | 0.22× |

**nntrainer CausalLM (Q4_0, on-device, 18-token prefill):**

| config | prefill TPS | decode TPS |
|---|---|---|
| MIN_ROWS=256 (hybrid, decode on CPU) | 214.3 | 94.3 |

### verify_flash_attn results (on-device, all PASS)

| test mode | FP32 Q/out | FP16 Q/out | max_abs_err |
|---|---|---|---|
| small (2 heads, 3 tokens) | PASS | PASS | 0.000833 |
| full (16 heads, 308 tokens) | PASS | PASS | 0.000833 |
| chunked (16 heads, 18 tokens, cache_from=1) | PASS | PASS | 0.000833 |

### What exists already

| Component | Location | Status |
|---|---|---|
| DSP kernel `op_flash_attn_ext` | `ggml-hexagon/htp/flash-attn-ops.c` | ✅ Working (HVX path + HMX fast path for head_dim%64==0) |
| HMX flash attention | `ggml-hexagon/htp/hmx-flash-attn-ops.c` | ✅ Working (head_dim=128 hits this path) |
| `HTP_OP_FLASH_ATTN_EXT` opcode | `htp-ops.h:75` | ✅ Defined |
| `nntr_htp_bridge_flash_attn` C-ABI | on-device `libggml-hexagon.so` | ✅ **Implemented and passing** (source not in current tree — built from uncommitted changes) |
| `verify_flash_attn.cpp` test tool | `tools/verify_flash_attn.cpp` | ✅ Written, all 3 modes pass |
| KV cache in rpcmem | `hexagon_rpc_allocator.cpp` | ✅ Already registered via `register_activation_pool` |
| mha_core call site | `Applications/CausalLM/layers/mha_core.cpp:774-781` | ❌ **NOT YET WIRED** — still uses 3 CPU calls |


### The verify_flash_attn.cpp expected signature
```c
int nntr_htp_bridge_flash_attn(
    const void *q,           // F32 or F16, [n_tokens, n_head, head_dim]
    const void *k,           // F16, [n_kv, n_head_kv, head_dim] (rpcmem)
    const void *v,           // F16, [n_kv, n_head_kv, head_dim] (rpcmem)
    const void *mask,         // F16, [n_tokens, n_kv] (causal mask)
    void *out,                // F32 or F16, [n_tokens, n_head, head_dim]
    unsigned int n_tokens,    // step_size (prefill tokens)
    unsigned int n_head,      // num_heads_Q (e.g. 16)
    unsigned int n_head_kv,   // num_heads_KV (e.g. 8)
    unsigned int head_dim,    // 128
    unsigned int n_kv,        // cache_to (total KV length)
    float scale,              // 1/sqrt(head_dim)
    int q_is_fp16,            // 0=F32, 1=F16
    int out_is_fp16           // 0=F32, 1=F16
);
```

### The mha_core call site to replace (lines 774-781)
```cpp
// CURRENT: 3 CPU calls, ~190ms for 18-token prefill
compute_kcaches(query_step, b_cached_key, out_, cache_from,
                cache_to - cache_from, num_heads_Q, gqa_size, head_dim);
softmax_triangle(out_, step_size, num_heads_Q, cache_from);
compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                              cache_from, num_heads_KV, gqa_size, head_dim, cache_to);

// TARGET: 1 DSP dispatch, ~0ms (fused on HMX)
nntr_htp_bridge_flash_attn(query_step.getData(), b_cached_key.getData(),
                           b_cached_value.getData(), mask, attention_output_step.getData(),
                           step_size, num_heads_Q, num_heads_KV, head_dim, cache_to,
                           1.0f/sqrtf(head_dim), q_is_fp16, out_is_fp16);
```

---

## Phase 1: Implement `nntr_htp_bridge_flash_attn` in the bridge

**File:** `../ggml-hexagon/ggml/src/ggml-hexagon/nntr-htp-bridge.cpp`

### 1.1 Add the C-ABI declaration
```c
extern "C" __attribute__((visibility("default")))
int nntr_htp_bridge_flash_attn(const void *q, const void *k, const void *v,
                                const void *mask, void *out,
                                unsigned int n_tokens, unsigned int n_head,
                                unsigned int n_head_kv, unsigned int head_dim,
                                unsigned int n_kv, float scale,
                                int q_is_fp16, int out_is_fp16);
```

### 1.2 Implementation approach
Follow the exact same pattern as `nntr_htp_bridge_gemm_q4_0`:
1. Lock state mutex
2. Ensure session
3. Find ext_act_pool for K/V (already in rpcmem) — zero-copy
4. Stage Q and output (may or may not be in rpcmem)
5. Build 5 `ggml_tensor` descriptors:
   - `t_q`: type = q_is_fp16 ? F16 : F32, ne = [head_dim, n_tokens, n_head, 1]
   - `t_k`: type = F16, ne = [head_dim, n_kv, n_head_kv, 1]
   - `t_v`: type = F16, ne = [head_dim, n_kv, n_head_kv, 1]  (or [head_dim, n_kv, n_head_kv, 1] — need to match DSP kernel's expected layout)
   - `t_mask`: type = F16, ne = [n_kv, n_tokens, 1, 1]
   - `t_out`: type = out_is_fp16 ? F16 : F32, ne = [head_dim, n_tokens, n_head, 1]
6. Set `t_out.op = GGML_OP_FLASH_ATTN_EXT`
7. Set `t_out.src[0..3] = {t_q, t_k, t_v, t_mask}`
8. Set `t_out.op_params`: scale (float at offset 0), max_bias=0.0f (offset 1), logit_softcap=0.0f (offset 2)
9. Build `htp_opnode` with `opcode = HTP_OP_FLASH_ATTN_EXT`
10. `sess->enqueue_op(node)`
11. `sess->flush(true)`
12. Copy output from staging if not zero-copy

### 1.3 Key tensor layout considerations

The DSP kernel (`flash_attn_ext_f16_thread`) reads:
- Q: `q->ne[0]` = head_dim, `q->ne[1]` = n_tokens, `q->ne[2]` = n_head
- K: `k->ne[0]` = head_dim, `k->ne[1]` = n_kv, `k->ne[2]` = n_head_kv
- V: `v->ne[0]` = head_dim, `v->ne[1]` = n_kv, `v->ne[2]` = n_head_kv
- mask: `mask->ne[0]` = n_kv, accessed as `mask->nb[1]` per query row
- dst: `dst->ne[0]` = head_dim, `dst->ne[1]` = n_head, `dst->ne[2]` = n_tokens

The GQA head mapping (h → h/gqa_size) is handled inside the DSP kernel via `broadcast_rk2`/`broadcast_rv2` fastdiv values, computed from `q->ne[2]/k->ne[2]`.

**Critical:** The KV cache in nntrainer is laid out as `[max_seq_len, n_head_kv * head_dim]` (row-major, one row per timestep). The DSP kernel expects `k->ne = [head_dim, n_kv, n_head_kv, 1]` with `nb[1]` = stride between KV positions. This maps directly if we set:
- `ne[0]` = head_dim, `ne[1]` = n_kv, `ne[2]` = n_head_kv
- `nb[0]` = 2 (F16), `nb[1]` = head_dim * 2, `nb[2]` = n_kv * head_dim * 2

This is a **stride permutation** of the cache layout — no copy needed, just correct `ne`/`nb` values.

### 1.4 Mask construction
The causal mask is `[n_tokens, n_kv]` F16, where `mask[i][j] = 0` if `j < cache_from + i + 1` else `-INF` (0xFC00). This is built on CPU (small: n_tokens * n_kv * 2 bytes) and staged into rpcmem.

### 1.5 op_params
The DSP kernel reads 3 floats from `op_params`:
```c
memcpy(&scale,         (float *) octx->op_params + 0, sizeof(float));
memcpy(&max_bias,      (float *) octx->op_params + 1, sizeof(float));
memcpy(&logit_softcap, (float *) octx->op_params + 2, sizeof(float));
```
We set: `scale = 1/sqrt(head_dim)`, `max_bias = 0.0f`, `logit_softcap = 0.0f`.

---

## Phase 2: Wire into mha_core.cpp

**File:** `Applications/CausalLM/layers/mha_core.cpp`

### 2.1 Add a dlopen/dlsym bridge loader (same pattern as `hexagon_compute_ops.cpp`)

```cpp
using nntr_htp_bridge_flash_attn_fn = int (*)(const void *, const void *, const void *,
                                              const void *, void *, unsigned int,
                                              unsigned int, unsigned int, unsigned int,
                                              unsigned int, float, int, int);
```

### 2.2 Gate the flash_attn path
Add a config flag (env var `NNTR_HEXAGON_FLASH_ATTN=1` or a model config property). Only enable when:
- `is_prefill && step_size > 1` (prefill path only — decode M=1 stays on CPU)
- `head_dim == 128` (HMX fast path)
- KV cache is F16 (already the case)
- Bridge is available (dlopen succeeded)

### 2.3 Replace the 3 CPU calls in `one_batch_incremental_forwarding`

In both overloads (lines 774-781 and 856-862):

```cpp
if (use_hexagon_flash_attn && is_prefill && step_size > 1) {
    // Build causal mask: [step_size, cache_to] F16
    // mask[i][j] = 0 if j < cache_from + i + 1 else -INF
    // ...
    
    // Single DSP dispatch
    int rc = flash_attn_fn(query_step.getData(), b_cached_key.getData(),
                           b_cached_value.getData(), mask_data,
                           attention_output_step.getData(),
                           step_size, num_heads_Q, num_heads_KV, head_dim,
                           cache_to, 1.0f/sqrtf((float)head_dim),
                           q_is_fp16, out_is_fp16);
    if (rc != 0) {
        // Fall back to CPU path
        compute_kcaches(...);
        softmax_triangle(...);
        compute_fp16vcache_transposed(...);
    }
} else {
    // Original CPU path
    compute_kcaches(...);
    softmax_triangle(...);
    compute_fp16vcache_transposed(...);
}
```

### 2.4 RoPE handling
RoPE is currently applied to Q and K **before** the attention calls (lines 725-748). The flash_attn kernel does NOT apply RoPE — it expects pre-rotated Q/K. So the existing RoPE application stays as-is; only the Q·K^T + softmax + scores·V part moves to DSP.

---

## Phase 3: Test with verify_flash_attn

### 3.1 Build and push
```bash
# Cross-compile (or build on device)
${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ \
  --target=aarch64-linux-android30 -std=c++17 -O2 \
  -o verify_flash_attn tools/verify_flash_attn.cpp -ldl
adb push verify_flash_attn /data/local/tmp/nntrainer/causallm/
```

### 3.2 Run
```bash
adb shell "cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=. ./verify_flash_attn"
adb shell "cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=. ./verify_flash_attn full"
adb shell "cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=. ./verify_flash_attn chunked"
```

**Pass criteria:** `max_abs_err <= 0.05` for both FP32 and FP16 Q/out paths.

---

## Phase 4: Profile and measure

### 4.1 Run CausalLM with flash_attn enabled
```bash
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/nntrainer/causallm:\$LD_LIBRARY_PATH && \
  export NNTR_NUM_THREADS=4 && export NNTR_HEXAGON_MIN_ROWS=1 && \
  export NNTR_HEXAGON_FLASH_ATTN=1 && export GGML_HEXAGON_PROFILE=1 && \
  cd /data/local/tmp/nntrainer/causallm && ./nntrainer_causallm models/qwen3-0.6b"
```

### 4.2 Expected results
| Metric | Before (CPU attention) | After (DSP flash_attn) | Target |
|---|---|---|---|
| Prefill TPS (18 tok) | ~265 | ~600+ | — |
| Prefill TPS (512 tok) | ~821 | ~2000+ | ~2083 (ggml-hexagon) |
| Attention time/layer | ~6.8 ms | ~0.5 ms | — |
| Total prefill (512 tok) | ~624 ms | ~256 ms | ~246 ms (ggml-hexagon) |

---

## Phase 5: Medium-term — Contiguous op chains (future)

After flash_attn is working, the next step is to batch multiple op types into a single flush:
1. `HTP_OP_RMS_NORM_MUL` (input norm)
2. `HTP_OP_MUL_MAT` (QKV projection)
3. `HTP_OP_FLASH_ATTN_EXT` (attention)
4. `HTP_OP_MUL_MAT` (attention output projection)
5. `HTP_OP_RMS_NORM_MUL` (post-attention norm)
6. `HTP_OP_MUL_MAT` (gate/up)
7. `HTP_OP_GLU_SWIGLU` (activation)
8. `HTP_OP_MUL_MAT` (down projection)
9. `HTP_OP_ADD` (residual)

All in one `flush()` — this is what ggml-hexagon does. Requires a "begin_batch / enqueue / flush" API in the bridge.

---

## Risk assessment

| Risk | Mitigation |
|---|---|
| Tensor layout mismatch (nntrainer vs ggml) | verify_flash_attn tests both small and full-scale shapes |
| HMX path not taken (head_dim not %64) | head_dim=128 for Qwen3-0.6B, always hits HMX |
| KV cache not in rpcmem | Already confirmed (§30) — KVCacheManager uses HexagonRpcAllocator |
| Mask format wrong | verify_flash_attn builds the same causal mask as mha_core |
| Output correctness | verify_flash_attn compares against CPU reference (max_abs_err ≤ 0.05) |
| Decode regression | Flash_attn only enabled for prefill (step_size > 1); decode stays on CPU |
| Fall-back safety | If bridge returns error, fall back to CPU path |
