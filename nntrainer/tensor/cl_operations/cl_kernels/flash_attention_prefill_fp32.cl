// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Tiwari <anup.tiwari@samsung.com>
 *
 * @file    flash_attention_prefill_fp32.cl
 * @date    23 April 2026
 * @brief   Tiled GEMM prefill kernel for flash attention (FP32) with GQA grouping
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Anup Tiwari <anup.tiwari@samsung.com>
 *
 * Tiled GEMM approach with online softmax for prefill attention.
 * Each work-group processes NCOLS1 Q tokens against tiles of K/V.
 * Online softmax avoids the two-pass approach, reading K only once.
 * Cooperative loading and KQ computation across work-items.
 * Phase 4: GQA grouping — multiple Q heads sharing the same KV head are
 * processed in a single work-group, reusing K/V tiles.
 * Phase 5: Adaptive cols_per_block — ncols1 is a runtime parameter, allowing
 * the dispatch code to select optimal Q tokens per work-group based on seqlen_q.
 * NCOLS1 remains as the compile-time maximum for local memory allocation.
 * float2 vectorized loads for 2× memory bandwidth on global and local access.
 *
 * Round 2 Optimization - Phase A: Parallel Online Softmax
 * - Replaces single-writer softmax with parallel reduction across all work-items
 * - Each work-item processes a subset of KV rows for max/exp computation
 * - Uses work_group_reduce_max and work_group_reduce_add for parallel reduction
 * - Eliminates serialization bottleneck in softmax computation
 *
 * Local memory layout (NCOLS1=4, NCOLS2=2, NBATCH_FA=16, HEAD_DIM=128):
 *   Q_tile:   4*2 x 128 x 4 = 4 KB   (NCOLS1*NCOLS2 rows)
 *   K_tile:  16   x 128 x 4 = 8 KB   (shared across Q heads)
 *   V_tile:  16   x 128 x 4 = 8 KB   (shared across Q heads)
 *   KQ_tile:  4*2 x  16 x 4 = 512 B  (NCOLS1*NCOLS2 rows)
 *   VKQ:      4*2 x 128 x 4 = 4 KB
 *   l_max_val:    4*2 x 4 = 32 B
 *   l_exp_sum:    4*2 x 4 = 32 B
 *   l_correction: 4*2 x 4 = 32 B
 *   Total: ~24.6 KB (fits in typical 32KB local memory)
 *
 * When NCOLS2=1, this kernel behaves identically to the pre-Phase-4 version.
 */

#define SOFTMAX_MIN -1e30f

// Configuration constants — NCOLS1 is the MAXIMUM (compile-time) for local memory allocation
// The actual ncols1 used at runtime may be less (passed as kernel argument)
#ifndef NCOLS1
#define NCOLS1 4
#endif

#ifndef NCOLS2
#define NCOLS2 2
#endif

#ifndef NBATCH_FA
#define NBATCH_FA 16
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

// Ensure HEAD_DIM is even for float2 vectorized access
#if (HEAD_DIM % 2) != 0
#error "HEAD_DIM must be even for float2 vectorized access"
#endif

#define HEAD_DIM2 (HEAD_DIM / 2)
#define NQ (NCOLS1 * NCOLS2)  // Total Q rows per work-group (maximum)

// Round 2 Phase A: Helper function for parallel max reduction
// Uses iterative halving until all work-items have the same max value
float work_group_reduce_max_fp32(float value) {
  // Use OpenCL 2.0 work_group_reduce_max if available, otherwise manual reduction
  // Manual parallel reduction using local memory
  __local float l_max_tmp[128];  // Support up to 128 work-items
  const int local_id = get_local_id(0);
  const int local_size = get_local_size(0);
  
  l_max_tmp[local_id] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // Parallel reduction - each iteration halves the number of active work-items
  for (int stride = local_size / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      l_max_tmp[local_id] = fmax(l_max_tmp[local_id], l_max_tmp[local_id + stride]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  return l_max_tmp[0];
}

// Round 2 Phase A: Helper function for parallel sum reduction
float work_group_reduce_add_fp32(float value) {
  __local float l_sum_tmp[128];  // Support up to 128 work-items
  const int local_id = get_local_id(0);
  const int local_size = get_local_size(0);
  
  l_sum_tmp[local_id] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // Parallel reduction - each iteration halves the number of active work-items
  for (int stride = local_size / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      l_sum_tmp[local_id] += l_sum_tmp[local_id + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  return l_sum_tmp[0];
}

__kernel void flash_attention_prefill_fp32(
    __global const float *query,
    __global const float *key,
    __global const float *value,
    __global float *output,
    const int seqlen_q,
    const int seqlen_k,
    const int head_dim,
    const int num_heads_q,
    const int num_heads_kv,
    const int batch,
    const float scale,
    const int ncols2,          // Number of Q heads per KV head (runtime, <= NCOLS2)
    const int head_group_offset, // First Q head index in this KV group
    const int ncols1_runtime   // Actual ncols1 for this dispatch (runtime, <= NCOLS1)
) {

  // Local memory tiles — declared inside kernel function for OpenCL compliance
  __local float Q_tile[NQ][HEAD_DIM];           // Q tile: NQ x HEAD_DIM
  __local float K_tile[NBATCH_FA][HEAD_DIM];     // K tile: NBATCH_FA x HEAD_DIM — shared!
  __local float V_tile[NBATCH_FA][HEAD_DIM];     // V tile: NBATCH_FA x HEAD_DIM — shared!
  __local float KQ_tile[NQ][NBATCH_FA];          // KQ dot products: NQ x NBATCH_FA
  __local float VKQ[NQ][HEAD_DIM];               // VKQ accumulator: NQ x HEAD_DIM
  __local float l_max_val[NQ];                   // Running max per Q row
  __local float l_exp_sum[NQ];                   // Running exp_sum per Q row
  __local float l_correction[NQ];                // Correction factor for VKQ rescaling

  const int group_id = get_group_id(0);
  const int local_id = get_local_id(0);
  const int local_size = get_local_size(0);

  // Use runtime ncols1 (adaptive cols_per_block from Phase 5)
  const int ncols1 = ncols1_runtime;

  // Total number of work-groups: batch * num_heads_kv * ceil(seqlen_q / ncols1)
  // Note: we iterate over KV heads, not Q heads — each work-group handles ncols2 Q heads
  const int num_q_groups = (seqlen_q + ncols1 - 1) / ncols1;
  const int total_groups = batch * num_heads_kv * num_q_groups;

  if (group_id >= total_groups) return;

  // Decode group_id into batch, kv_head, and q_group indices
  const int q_group = group_id % num_q_groups;
  const int kv_head_batch = group_id / num_q_groups;
  const int kv_head_id = kv_head_batch % num_heads_kv;
  const int batch_id = kv_head_batch / num_heads_kv;

  // Number of valid Q heads in this group (may be less than ncols2 at boundary)
  const int gqa_ratio = num_heads_q / num_heads_kv;
  const int first_q_head = kv_head_id * gqa_ratio + head_group_offset;
  const int valid_ncols2 = min(ncols2, num_heads_q - first_q_head);
  // Also ensure we don't exceed gqa_ratio
  const int actual_ncols2 = min(valid_ncols2, gqa_ratio - head_group_offset);

  if (actual_ncols2 <= 0) return;

  // Starting Q row for this work-group (same for all Q heads in the group)
  const int q_start = q_group * ncols1;

  // Number of valid Q rows in this group (may be less than ncols1 at boundary)
  const int valid_ncols1 = min(seqlen_q - q_start, ncols1);

  // Calculate base offsets for KV (shared across all Q heads in this group)
  const int kv_batch_offset = batch_id * num_heads_kv * seqlen_k * head_dim;
  const int kv_head_offset = kv_batch_offset + kv_head_id * seqlen_k * head_dim;

  // Initialize online softmax state in local memory
  for (int i = local_id; i < NQ; i += local_size) {
    l_max_val[i] = SOFTMAX_MIN;
    l_exp_sum[i] = 0.0f;
    l_correction[i] = 1.0f;
  }

  // Initialize VKQ accumulator in local memory
  for (int i = 0; i < NQ; i++) {
    for (int d = local_id; d < HEAD_DIM; d += local_size) {
      VKQ[i][d] = 0.0f;
    }
  }

  // Load Q tile cooperatively using float2 vectorized loads
  // Q_tile layout: rows [h*NCOLS1 .. (h+1)*NCOLS1-1] are for Q head h
  for (int h = 0; h < actual_ncols2; h++) {
    const int q_head_id = first_q_head + h;
    const int query_batch_offset = batch_id * num_heads_q * seqlen_q * head_dim;
    const int query_head_offset = query_batch_offset + q_head_id * seqlen_q * head_dim;

    for (int i = 0; i < valid_ncols1; i++) {
      const int q_row_offset = query_head_offset + (q_start + i) * head_dim;
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        const float2 val = *((__global const float2*)(query + q_row_offset + d2 * 2));
        *((__local float2*)&Q_tile[h * NCOLS1 + i][d2 * 2]) = val;
      }
    }
    // Zero out unused Q rows (when valid_ncols1 < NCOLS1)
    for (int i = valid_ncols1; i < NCOLS1; i++) {
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        *((__local float2*)&Q_tile[h * NCOLS1 + i][d2 * 2]) = (float2)(0.0f, 0.0f);
      }
    }
  }
  // Zero out unused Q head slots (when actual_ncols2 < NCOLS2)
  for (int h = actual_ncols2; h < NCOLS2; h++) {
    for (int i = 0; i < NCOLS1; i++) {
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        *((__local float2*)&Q_tile[h * NCOLS1 + i][d2 * 2]) = (float2)(0.0f, 0.0f);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Process KV in tiles of NBATCH_FA
  for (int kv_start = 0; kv_start < seqlen_k; kv_start += NBATCH_FA) {
    const int nrows_kv = min(seqlen_k - kv_start, NBATCH_FA);

    // Load K tile cooperatively using float2 vectorized loads (ONCE per KV head!)
    for (int j = 0; j < nrows_kv; j++) {
      const int k_row_offset = kv_head_offset + (kv_start + j) * head_dim;
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        const float2 val = *((__global const float2*)(key + k_row_offset + d2 * 2));
        *((__local float2*)&K_tile[j][d2 * 2]) = val;
      }
    }
    // Zero out unused K rows
    for (int j = nrows_kv; j < NBATCH_FA; j++) {
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        *((__local float2*)&K_tile[j][d2 * 2]) = (float2)(0.0f, 0.0f);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute KQ tile: KQ_tile[qi][j] = dot(Q_tile[qi], K_tile[j]) * scale
    // Phase 5: Use float2 vectorized access for Q and K tiles
    // Phase 4: Process all NQ Q rows (multiple heads) reusing the same K tile
    for (int qi = 0; qi < NQ; qi++) {
      for (int j = local_id; j < NBATCH_FA; j += local_size) {
        float kq_acc = 0.0f;
        for (int d2 = 0; d2 < HEAD_DIM2; d2++) {
          const float2 q_val = *((__local float2*)&Q_tile[qi][d2 * 2]);
          const float2 k_val = *((__local float2*)&K_tile[j][d2 * 2]);
          kq_acc += q_val.x * k_val.x;
          kq_acc += q_val.y * k_val.y;
        }
        KQ_tile[qi][j] = kq_acc * scale;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Round 2 Phase A: Parallel Online Softmax
    // All work-items cooperatively compute softmax for each Q row
    // Each work-item handles a subset of KV rows, then parallel reduction combines results
    for (int h = 0; h < actual_ncols2; h++) {
      for (int i = 0; i < valid_ncols1; i++) {
        const int qi = h * NCOLS1 + i;
        
        // Phase A.1: Parallel max reduction across work-items
        // Each work-item finds max in its assigned subset of KV rows
        float local_max = SOFTMAX_MIN;
        for (int j = local_id; j < nrows_kv; j += local_size) {
          local_max = fmax(local_max, KQ_tile[qi][j]);
        }
        // Parallel reduction to find global max across all work-items
        float new_max = work_group_reduce_max_fp32(local_max);
        
        // Combine with previous running max
        float prev_max = l_max_val[qi];
        new_max = fmax(new_max, prev_max);
        
        // Phase A.2: Compute correction factor and update state
        float correction = 1.0f;
        if (new_max != prev_max) {
          correction = exp(prev_max - new_max);
        }
        
        // Phase A.3: Parallel exp computation and sum reduction
        // Each work-item computes exp for its subset and accumulates local sum
        float local_exp_sum = 0.0f;
        for (int j = local_id; j < nrows_kv; j += local_size) {
          float exp_val = exp(KQ_tile[qi][j] - new_max);
          KQ_tile[qi][j] = exp_val;  // Store exp value for VKQ computation
          local_exp_sum += exp_val;
        }
        // Parallel reduction to get total exp sum
        float tile_exp_sum = work_group_reduce_add_fp32(local_exp_sum);
        
        // Update global state (single writer to avoid race condition)
        if (local_id == 0) {
          l_correction[qi] = correction;
          l_exp_sum[qi] = l_exp_sum[qi] * correction + tile_exp_sum;
          l_max_val[qi] = new_max;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Phase A.4: Apply correction to VKQ accumulator (parallel across work-items)
        for (int d = local_id; d < HEAD_DIM; d += local_size) {
          VKQ[qi][d] *= correction;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Load V tile cooperatively using float2 vectorized loads (ONCE per KV head!)
    for (int j = 0; j < nrows_kv; j++) {
      const int v_row_offset = kv_head_offset + (kv_start + j) * head_dim;
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        const float2 val = *((__global const float2*)(value + v_row_offset + d2 * 2));
        *((__local float2*)&V_tile[j][d2 * 2]) = val;
      }
    }
    // Zero out unused V rows
    for (int j = nrows_kv; j < NBATCH_FA; j++) {
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        *((__local float2*)&V_tile[j][d2 * 2]) = (float2)(0.0f, 0.0f);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Accumulate VKQ: VKQ[qi][d] += sum_j KQ_tile[qi][j] * V_tile[j][d]
    // Phase 5: Use float2 vectorized access for V tile reads
    // Phase 4: Process all NQ Q rows reusing the same V tile
    for (int h = 0; h < actual_ncols2; h++) {
      for (int i = 0; i < valid_ncols1; i++) {
        const int qi = h * NCOLS1 + i;
        for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
          float2 vkq_acc = (float2)(0.0f, 0.0f);
          for (int j = 0; j < nrows_kv; j++) {
            const float kq_exp = KQ_tile[qi][j];
            const float2 v_val = *((__local float2*)&V_tile[j][d2 * 2]);
            vkq_acc.x += kq_exp * v_val.x;
            vkq_acc.y += kq_exp * v_val.y;
          }
          VKQ[qi][d2 * 2] += vkq_acc.x;
          VKQ[qi][d2 * 2 + 1] += vkq_acc.y;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Final normalization: output = VKQ / exp_sum
  // Write results to global memory using float2 vectorized stores
  for (int h = 0; h < actual_ncols2; h++) {
    const int q_head_id = first_q_head + h;
    const int query_batch_offset = batch_id * num_heads_q * seqlen_q * head_dim;
    const int output_head_offset = query_batch_offset + q_head_id * seqlen_q * head_dim;

    for (int i = 0; i < valid_ncols1; i++) {
      const int qi = h * NCOLS1 + i;
      const int out_row_offset = output_head_offset + (q_start + i) * head_dim;
      const float inv_exp_sum = 1.0f / l_exp_sum[qi];
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        const float2 out_val = (float2)(
          VKQ[qi][d2 * 2] * inv_exp_sum,
          VKQ[qi][d2 * 2 + 1] * inv_exp_sum
        );
        *((__global float2*)(output + out_row_offset + d2 * 2)) = out_val;
      }
    }
  }
}