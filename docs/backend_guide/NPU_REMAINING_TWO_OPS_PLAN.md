# Plan: Fix the Last 2 CPU Ops in the Transformer Block

**Date:** 2026-08-19
**Goal:** Eliminate the last 2 CPU touch points that force FastRPC flushes.

---

## The 2 Remaining CPU Ops

### Op 1: K-cache RoPE Rotation (CPU)

**Location:** `mha_core.cpp`, `try_dsp_fp16_rope()` at line ~379 (short-circuits with `return false`)

**Why it's on CPU:** The DSP RoPE kernel (`htp/rope-ops.c`) is F32-only. K activations are FP16 on this model. The cast-chain (F16→F32→rotate→F32→F16) was tried and measured as 14% regression — 3 extra HVX dispatches cost more than the tiny CPU RoPE saves.

**Why it forces a flush:** CPU RoPE reads the K GEMM output (enqueued on DSP, not yet executed). Must flush before reading.

**Impact:** 28 flushes per prefill (1 per block × 28 blocks).

### Op 2: KV-cache Input Node Copies (CPU)

**Location:** `mha_core.cpp`, lines 869-871 and 1034-1035 (prefill path) and lines 884 (output copy)

```cpp
Q_step.copyData(query_step);   // line 869 / 1034
K_step.copyData(key_step);     // line 870 / 1035
V_step.copyData(value_step);   // line 871
output_step.copyData(O_step);  // line 884
```

**Why it's on CPU:** These are `Tensor::copyData()` calls — host-side memcpy. They exist because of `#ifdef ENABLE_FP16`: when FP16 is enabled, the code creates local FP16 tensors (`Q_step`, `K_step`, `V_step`, `O_step`) and copies the projected Q/K/V into them before passing to `one_batch_incremental_forwarding()`. When FP16 is disabled (`#else` path), the original tensors are passed directly — no copy.

**Why it forces a flush:** `copyData` reads the source tensor (Q/K/V GEMM output, pending on DSP). Must flush before reading.

**Impact:** ~56-84 flushes per prefill (3-4 copies × 2 code paths × 28 blocks, minus overlaps).

---

## Fix for Op 1: K-cache RoPE — Add FP16 Path to DSP RoPE Kernel

### Approach

Add FP16 support to the existing `htp/rope-ops.c` DSP kernel. The HVX has native FP16 support (`v16u16` pairs, `Hx_vec_fp16` intrinsics). The kernel currently loads `float`, rotates, stores `float`. Adding a FP16 path means: load `_FP16`, convert to F32 for the rotation math (or do F16 math directly), store `_FP16`.

This eliminates the cast chain entirely — one DSP dispatch instead of three, no scratch buffer, no F16↔F32 conversion.

### Why This Won't Have the Same Regression as the Cast Chain

The cast chain was 3 ops: `cpy(F16→F32)` + `rope(F32)` + `cpy(F32→F16)`. Each op is a separate enqueue with its own DMA and HVX dispatch overhead.

A native FP16 RoPE kernel is 1 op: `rope_fp16(F16→F16)`. One enqueue, one DMA in, one DMA out, one HVX execute. The rotation math is the same (sin/cos lookup + multiply-add), just with F16 load/store.

The 14% regression was from 3× the dispatch overhead. 1× dispatch has 1/3 the overhead. For head_dim=128, n_heads=16, n_tokens=650: the rotation is 650×16×128 = 1.3M elements. At HVX throughput (~10 GFLOP/s for F16), that's ~0.13ms. One dispatch overhead is ~0.1ms. Total: ~0.23ms per block. The CPU version is ~0.01ms compute + 0.1ms flush = 0.11ms. So DSP is ~2× slower per block.

**But:** the DSP version eliminates the flush. 28 flushes × 0.1ms = 2.8ms saved. DSP adds 28 × 0.23ms = 6.4ms. Net: +3.6ms. Still a regression.

**Hmm.** The math says native FP16 RoPE on DSP is still slower than CPU for this model size. The rotation is too small to amortize the dispatch.

### Alternative Approach: Fuse RoPE into the K-cache Copy

Instead of a separate RoPE dispatch, fuse the rotation into the K-cache append copy. The DSP already does a `cpy` for V-cache append. Add a `rope_cpy` kernel that: reads K (F16), applies rotation, writes to cache (F16). One dispatch, does both rotation and copy.

This eliminates:
- The CPU RoPE (0.01ms compute + 0.1ms flush = 0.11ms/block)
- The K-cache copy (if it was CPU) or chains after the V-cache cpy (if DSP)

The fused kernel reads K from the GEMM output (in rpcmem), rotates in VTCM, writes to the cache (in rpcmem). One DMA in, one DMA out, one HVX execute. Same cost as the V-cache cpy (~0.1ms) plus the rotation (~0.13ms) = ~0.23ms.

vs CPU: 0.01ms (RoPE) + 0.1ms (flush) + 0.01ms (copy) = 0.12ms.

**Still slower.** The rotation is just too small for this model.

### The Honest Answer for Op 1

For Qwen3-0.6B (head_dim=128, 16 heads), RoPE is too small to benefit from DSP dispatch. The CPU version is faster. The flush it causes (0.1ms × 28 = 2.8ms) is the cost of keeping it on CPU.

**The only way to eliminate this flush without regression is to fuse RoPE into a kernel that's already dispatching.** Specifically: fuse RoPE into the flash attention kernel. The flash attention kernel already reads Q and K from DDR → VTCM. If it applies RoPE inside the kernel (before computing QK^T), there's no separate RoPE dispatch at all.

This is exactly what ggml-hexagon does — `hmx_flash_attn_ext` applies RoPE internally.

**Plan for Op 1:**
1. **Short term (no new kernel):** Accept the 28 flushes. The 2.8ms cost is 0.35% of 800ms prefill. Not worth the regression risk.
2. **Medium term (modify flash_attn kernel):** Add a `apply_rope` flag to `nntr_htp_bridge_flash_attn`. When set, the kernel applies RoPE to Q and K inside VTCM before computing attention. This eliminates both the RoPE dispatch AND the flush. Zero extra cost — the rotation happens while data is in VTCM for attention anyway.
3. **Long term (if needed):** Write a standalone FP16 RoPE kernel for models with larger head_dim (e.g., 256) where the rotation is big enough to amortize dispatch.

### Concrete Steps for Op 1 (Medium Term: Fuse into Flash Attention)

**File:** `ggml-hexagon/ggml/src/ggml-hexagon/htp/flash-attn-ops.c` (or equivalent)

1. Add `apply_rope` parameter to the flash attention kernel signature:
   ```c
   // Current:
   int nntr_htp_bridge_flash_attn(const void *q, const void *k, const void *v,
                                   void *out, ...);
   
   // New:
   int nntr_htp_bridge_flash_attn(const void *q, const void *k, const void *v,
                                   void *out, ...,
                                   int apply_rope,           // new
                                   const int32_t *positions,   // new
                                   float theta);              // new
   ```

2. Inside the kernel, after loading Q and K tiles into VTCM:
   ```c
   if (apply_rope) {
     // Apply RoPE to Q tile in VTCM
     for (int i = 0; i < tile_rows; i++) {
       int pos = positions[row_offset + i];
       for (int d = 0; d < head_dim; d += 2) {
         float angle = pos * inv_freq[d];
         float c = cosf(angle), s = sinf(angle);
         float q0 = q_tile[i][d], q1 = q_tile[i][d+1];
         q_tile[i][d]     = q0 * c - q1 * s;
         q_tile[i][d+1]   = q0 * s + q1 * c;
         // Same for K tile
       }
     }
   }
   ```

3. In `mha_core.cpp`, when calling flash attention, pass `apply_rope=true` and the positions. Remove the separate RoPE dispatch for Q and K.

4. **Eliminates:** 2 RoPE dispatches (Q + K) + 2 flushes per block = 56 flushes.

**Risk:** Medium. Modifying the flash attention kernel is delicate — the VTCM tiling must accommodate the RoPE step. But the rotation is a simple element-wise op that fits between the DMA load and the GEMM compute.

**Testing:** Compare attention output with and without fused RoPE. Max abs error should be < 1e-3 (FP16 precision).

---

## Fix for Op 2: KV-cache Input Node Copies — Use DSP cpy

### Approach

Replace `Tensor::copyData()` with `try_dsp_cache_copy()` for the Q/K/V/O copies. The `try_dsp_cache_copy` function already exists and works (it's used for V-cache append). It dispatches `HTP_OP_CPY` which handles FP16→FP16 copies.

### The 4 Copies

| Copy | Source | Destination | Can use DSP cpy? |
|------|--------|-------------|-------------------|
| `Q_step.copyData(query_step)` | query_step (QKV GEMM output) | Q_step (local FP16) | ✅ — same dtype, contiguous |
| `K_step.copyData(key_step)` | key_step (QKV GEMM output) | K_step (local FP16) | ✅ |
| `V_step.copyData(value_step)` | value_step (QKV GEMM output) | V_step (local FP16) | ✅ |
| `output_step.copyData(O_step)` | O_step (attention output) | output_step (layer output) | ✅ |

All 4 are FP16→FP16 contiguous copies. `try_dsp_cache_copy` handles this.

### The Problem: Why These Copies Exist

Looking at the code more carefully:

```cpp
#ifdef ENABLE_FP16
  // Create local FP16 tensors with specific dimensions
  nntrainer::Tensor Q_step = nntrainer::Tensor(Q_step_dim, true);
  nntrainer::Tensor K_step = nntrainer::Tensor(K_step_dim, true);
  nntrainer::Tensor V_step = nntrainer::Tensor(V_step_dim, true);
  nntrainer::Tensor O_step = nntrainer::Tensor(O_step_dim, true);

  Q_step.copyData(query_step);  // Copy projected Q into local tensor
  K_step.copyData(key_step);
  V_step.copyData(value_step);
  
  one_batch_incremental_forwarding(..., Q_step, K_step, V_step, O_step, ...);
  
  output_step.copyData(O_step);  // Copy attention output back
#else
  // No copy — pass original tensors directly
  one_batch_incremental_forwarding(..., query_step, key_step, value_step, output_step, ...);
#endif
```

The copies exist because `ENABLE_FP16` changes the dtype of the local tensors. The original `query_step`/`key_step`/`value_step` might be FP32 (from the GEMM), and the local tensors are FP16. The `copyData` does a dtype conversion.

**Wait** — but the QKV GEMM output is already FP16 on this model (the bridge does q4_0 weight × FP16 activation → FP16 output). So the copy is FP16→FP16, same dtype. The copy is just a layout reshape, not a dtype conversion.

If the dtypes match, the copy is unnecessary — we can pass the original tensors directly, like the `#else` path does.

### The Fix: Eliminate Unnecessary Copies

**Step 1: Check if the copies are actually needed.**

If `query_step`, `key_step`, `value_step` are already FP16 with the right dimensions, the `#ifdef ENABLE_FP16` path is creating unnecessary copies. The fix is to pass the original tensors directly, same as the `#else` path.

**Step 2: If copies are needed (different layout/dims), use DSP cpy.**

If the copies serve a real purpose (e.g., reshaping from [batch, n_heads, seq, head_dim] to [batch, seq, n_heads, head_dim]), replace `copyData` with `try_dsp_cache_copy`:

```cpp
// Before:
Q_step.copyData(query_step);

// After:
if (!try_dsp_cache_copy(is_cdsp_engine, query_step, Q_step)) {
  Q_step.copyData(query_step);  // CPU fallback
}
```

This dispatches the copy as a DSP op in the same batch — no flush needed. The DSP executes the copy in FIFO order after the QKV GEMM.

### Concrete Steps for Op 2

**Step 1: Determine if copies are necessary**

Check if `query_step` and `Q_step` have the same dims and dtype. If yes, eliminate the copy entirely:

```cpp
// In mha_core.cpp, replace the #ifdef ENABLE_FP16 path:
// Instead of creating Q_step/K_step/V_step/O_step and copying,
// just pass the original tensors (like the #else path does).
// 
// This requires verifying that one_batch_incremental_forwarding
// can accept the original tensor shapes.
```

**Step 2: If copies are needed, wire DSP cpy**

```cpp
// Replace:
Q_step.copyData(query_step);
K_step.copyData(key_step);
V_step.copyData(value_step);

// With:
if (is_cdsp_engine) {
  bool q_ok = try_dsp_cache_copy(true, query_step, Q_step);
  bool k_ok = try_dsp_cache_copy(true, key_step, K_step);
  bool v_ok = try_dsp_cache_copy(true, value_step, V_step);
  if (!q_ok) Q_step.copyData(query_step);
  if (!k_ok) K_step.copyData(key_step);
  if (!v_ok) V_step.copyData(value_step);
} else {
  Q_step.copyData(query_step);
  K_step.copyData(key_step);
  V_step.copyData(value_step);
}
```

And for the output copy:
```cpp
// Replace:
output_step.copyData(O_step);

// With:
if (!try_dsp_cache_copy(is_cdsp_engine, O_step, output_step)) {
  output_step.copyData(O_step);
}
```

**Step 3: Remove the flush before the copies**

The copies are currently on CPU, so there's an implicit flush (the CPU reads the GEMM output). With DSP cpy, the copy is enqueued in the same batch — no flush. The DSP executes GEMM first, then cpy, in FIFO order.

**Step 4: Verify correctness**

The DSP cpy must produce identical results to `Tensor::copyData`. For FP16→FP16 contiguous copies, this is guaranteed (it's a memcpy). For layout-changing copies, verify the reshape is correct.

**Risk:** Low for FP16→FP16 contiguous copies. Medium if the copy involves a reshape (the DSP cpy kernel assumes contiguous source and destination).

---

## Summary

| Op | Fix | New DSP kernel? | Flushes eliminated | Risk |
|----|-----|-----------------|-------------------|------|
| K-cache RoPE | Fuse RoPE into flash_attn kernel | Modify existing kernel | 28-56 | Medium |
| KV-cache input copies | Use `try_dsp_cache_copy` or eliminate copies | No | 56-84 | Low |

### Priority

1. **Op 2 first** (KV-cache input copies) — lower risk, bigger flush reduction, no new DSP code
2. **Op 1 second** (K-cache RoPE) — requires modifying flash_attn kernel, higher risk, smaller flush reduction

### Expected Outcome

After both fixes:
- Op 2 eliminates 56-84 flushes → ~6-8ms saved
- Op 1 eliminates 28-56 flushes → ~3-6ms saved
- Total: ~9-14ms saved, ~60-140 flushes eliminated
- Remaining: ~5-10 boundary flushes (embedding, LM-head, sampling)
- **Near-zero flushes achieved**

### What's NOT in This Plan

- All-rpcmem (separate work, eliminates staging memcpy cost)
- Op-level IR (separate work, eliminates host-side per-op overhead)
- Backward pass (separate plan: `NPU_BACKWARD_IMPLEMENTATION_PLAN.md`)

---

Signed-off-by: Cline SR <noreply@anthropic.com>
