// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Tiwari <anup.tiwari@samsung.com>
 *
 * @file    flash_attention_prefill_fp32_adreno.cl
 * @date    26 April 2026
 * @brief   Tiled GEMM prefill kernel for flash attention (FP32) optimized for Adreno GPU
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Anup Tiwari <anup.tiwari@samsung.com>
 *
 * Adreno-optimized variant of the FP32 prefill kernel with:
 * - Parallel online softmax using sub-group reductions
 * - cl_khr_subgroups / cl_qcom_subgroups extension for hardware-accelerated reductions
 * - Larger NBATCH_FA (32) leveraging Adreno's 64+ KB local memory
 * - float4 vectorized loads for 4× memory bandwidth
 * - Eliminated unnecessary zero-filling of unused K/V rows
 * - KQ computation limited to valid rows only (nrows_kv instead of NBATCH_FA)
 *
 * Local memory layout (NCOLS1=4, NCOLS2=1, NBATCH_FA=32, HEAD_DIM=128):
 *   Q_tile:    4 x 128 x 4 = 2 KB    (float)
 *   K_tile:   32 x 128 x 4 = 16 KB   (float — shared across Q heads)
 *   V_tile:   32 x 128 x 4 = 16 KB   (float — shared across Q heads)
 *   KQ_tile:   4 x  32 x 4 = 512 B   (float)
 *   VKQ:       4 x 128 x 4 = 2 KB     (float accumulator)
 *   l_max_val:     4 x 4 = 16 B
 *   l_exp_sum:     4 x 4 = 16 B
 *   l_correction:  4 x 4 = 16 B
 *   Reduction buffers: 256 x 4 = 1 KB (l_max_tmp + l_sum_tmp)
 *   Total: ~37.5 KB (fits in 64 KB Adreno local memory)
 *
 * When NCOLS2=1, this kernel behaves identically to the non-GQA version.
 */

// Enable sub-group extension if available (for hardware-accelerated reductions)
// Adreno 6xx uses cl_qcom_subgroups, Adreno 7xx+/8xx uses cl_khr_subgroups
// Both provide sub_group_reduce_max/sub_group_reduce_add with identical APIs
#ifdef CL_QCOM_SUBGROUPS
#pragma OPENCL EXTENSION cl_qcom_subgroups : enable
#define USE_SUBGROUPS 1
#elif defined(CL_KHR_SUBGROUPS)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define USE_SUBGROUPS 1
#endif

#define SOFTMAX_MIN -1e30f

// Configuration constants — NCOLS1 is the MAXIMUM (compile-time) for local memory allocation
// The actual ncols1 used at runtime may be less (passed as kernel argument)
#ifndef NCOLS1
#define NCOLS1 4
#endif

#ifndef NCOLS2
#define NCOLS2 1
#endif

// Adreno has more local memory (64+ KB), so we can use larger NBATCH_FA
// NBATCH_FA=32 tested optimal on Adreno
#ifndef NBATCH_FA
#define NBATCH_FA 32
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

// Ensure HEAD_DIM is multiple of 4 for float4 vectorized access
#if (HEAD_DIM % 4) != 0
#error "HEAD_DIM must be multiple of 4 for float4 vectorized access on Adreno"
#endif

#define HEAD_DIM4 (HEAD_DIM / 4)
#define NQ (NCOLS1 * NCOLS2)  // Total Q rows per work-group (maximum)

// --- Parallel reduction helper functions ---
// Note: __local arrays must be declared at kernel function scope in OpenCL.
// These functions take __local pointers to reduction buffers allocated in the kernel.

#ifdef USE_SUBGROUPS
// Sub-group accelerated max reduction for Adreno GPUs
float reduce_max_impl(float value, __local float *l_max_tmp) {
  // Step 1: Reduce within each sub-group (hardware-accelerated, no barriers needed)
  float sg_max = sub_group_reduce_max(value);

  // Step 2: First lane in each sub-group writes result to local memory
  if (get_sub_group_local_id() == 0) {
    l_max_tmp[get_sub_group_id()] = sg_max;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Step 3: First sub-group reduces across sub-group results
  const int num_sg = get_num_sub_groups();
  if (get_sub_group_id() == 0) {
    float cross_sg = (get_sub_group_local_id() < num_sg)
                         ? l_max_tmp[get_sub_group_local_id()]
                         : SOFTMAX_MIN;
    cross_sg = sub_group_reduce_max(cross_sg);
    if (get_local_id(0) == 0) {
      l_max_tmp[0] = cross_sg;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  return l_max_tmp[0];
}

// Sub-group accelerated sum reduction for Adreno GPUs
float reduce_add_impl(float value, __local float *l_sum_tmp) {
  // Step 1: Reduce within each sub-group (hardware-accelerated, no barriers needed)
  float sg_sum = sub_group_reduce_add(value);

  // Step 2: First lane in each sub-group writes result to local memory
  if (get_sub_group_local_id() == 0) {
    l_sum_tmp[get_sub_group_id()] = sg_sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Step 3: First sub-group reduces across sub-group results
  const int num_sg = get_num_sub_groups();
  if (get_sub_group_id() == 0) {
    float cross_sg = (get_sub_group_local_id() < num_sg)
                         ? l_sum_tmp[get_sub_group_local_id()]
                         : 0.0f;
    cross_sg = sub_group_reduce_add(cross_sg);
    if (get_local_id(0) == 0) {
      l_sum_tmp[0] = cross_sg;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  return l_sum_tmp[0];
}

#else
// Manual parallel max reduction using local memory (fallback for non-Adreno)
float reduce_max_impl(float value, __local float *l_max_tmp) {
  const int lid = get_local_id(0);
  const int lsize = get_local_size(0);

  l_max_tmp[lid] = value;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = lsize / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      l_max_tmp[lid] = fmax(l_max_tmp[lid], l_max_tmp[lid + stride]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  return l_max_tmp[0];
}

// Manual parallel sum reduction using local memory (fallback for non-Adreno)
float reduce_add_impl(float value, __local float *l_sum_tmp) {
  const int lid = get_local_id(0);
  const int lsize = get_local_size(0);

  l_sum_tmp[lid] = value;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = lsize / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      l_sum_tmp[lid] += l_sum_tmp[lid + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  return l_sum_tmp[0];
}
#endif

__kernel void flash_attention_prefill_fp32_adreno(
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
    const int ncols1_runtime    // Actual ncols1 for this dispatch (runtime, <= NCOLS1)
) {

  // Local memory tiles — declared inside kernel function for OpenCL compliance
  __local float Q_tile[NQ][HEAD_DIM];             // Q tile: NQ x HEAD_DIM
  __local float K_tile[NBATCH_FA][HEAD_DIM];      // K tile: NBATCH_FA x HEAD_DIM — shared!
  __local float V_tile[NBATCH_FA][HEAD_DIM];      // V tile: NBATCH_FA x HEAD_DIM — shared!
  __local float KQ_tile[NQ][NBATCH_FA];           // KQ dot products: NQ x NBATCH_FA
  __local float VKQ[NQ][HEAD_DIM];                // VKQ accumulator: NQ x HEAD_DIM
  __local float l_max_val[NQ];                   // Running max per Q row
  __local float l_exp_sum[NQ];                   // Running exp_sum per Q row
  __local float l_correction[NQ];                // Correction factor

  // Reduction buffers for parallel softmax (must be at kernel scope for OpenCL compliance)
  __local float l_max_tmp[128];  // For reduce_max_impl — supports up to 128 work-items
  __local float l_sum_tmp[128];  // For reduce_add_impl — supports up to 128 work-items

  const int group_id = get_group_id(0);
  const int local_id = get_local_id(0);
  const int local_size = get_local_size(0);

  // Use runtime ncols1 (adaptive cols_per_block from Phase 5)
  const int ncols1 = ncols1_runtime;

  // Total number of work-groups: batch * num_heads_kv * ceil(seqlen_q / ncols1)
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
  const int actual_ncols2 = min(valid_ncols2, gqa_ratio - head_group_offset);

  if (actual_ncols2 <= 0) return;

  // Starting Q row for this work-group
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

  // Initialize VKQ accumulator
  for (int i = 0; i < NQ; i++) {
    for (int d = local_id; d < HEAD_DIM; d += local_size) {
      VKQ[i][d] = 0.0f;
    }
  }

  // Load Q tile cooperatively using float4 vectorized loads (4× bandwidth on Adreno)
  for (int h = 0; h < actual_ncols2; h++) {
    const int q_head_id = first_q_head + h;
    const int query_batch_offset = batch_id * num_heads_q * seqlen_q * head_dim;
    const int query_head_offset = query_batch_offset + q_head_id * seqlen_q * head_dim;

    for (int i = 0; i < valid_ncols1; i++) {
      const int q_row_offset = query_head_offset + (q_start + i) * head_dim;
      for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        const float4 val = *((__global const float4*)(query + q_row_offset + d4 * 4));
        *((__local float4*)&Q_tile[h * NCOLS1 + i][d4 * 4]) = val;
      }
    }
    // Zero out unused Q rows
    for (int i = valid_ncols1; i < NCOLS1; i++) {
      for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        *((__local float4*)&Q_tile[h * NCOLS1 + i][d4 * 4]) = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
  }
  // Zero out unused Q head slots
  for (int h = actual_ncols2; h < NCOLS2; h++) {
    for (int i = 0; i < NCOLS1; i++) {
      for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        *((__local float4*)&Q_tile[h * NCOLS1 + i][d4 * 4]) = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Process KV in tiles of NBATCH_FA
  for (int kv_start = 0; kv_start < seqlen_k; kv_start += NBATCH_FA) {
    const int nrows_kv = min(seqlen_k - kv_start, NBATCH_FA);

    // Load K tile cooperatively using float4 vectorized loads (ONCE per KV head!)
    // No need to zero unused rows — all computation loops use nrows_kv
    for (int j = 0; j < nrows_kv; j++) {
      const int k_row_offset = kv_head_offset + (kv_start + j) * head_dim;
      for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        const float4 val = *((__global const float4*)(key + k_row_offset + d4 * 4));
        *((__local float4*)&K_tile[j][d4 * 4]) = val;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute KQ tile: KQ_tile[qi][j] = dot(Q_tile[qi], K_tile[j]) * scale
    // Only compute for valid KV rows (nrows_kv) — unused rows not zero-filled
    for (int qi = 0; qi < NQ; qi++) {
      for (int j = local_id; j < nrows_kv; j += local_size) {
        float4 kq_acc4 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        for (int d4 = 0; d4 < HEAD_DIM4; d4++) {
          const float4 q_val = *((__local float4*)&Q_tile[qi][d4 * 4]);
          const float4 k_val = *((__local float4*)&K_tile[j][d4 * 4]);
          kq_acc4 += q_val * k_val;
        }
        float kq_acc = kq_acc4.x + kq_acc4.y + kq_acc4.z + kq_acc4.w;
        KQ_tile[qi][j] = kq_acc * scale;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel Online Softmax — all work-items cooperate
    for (int h = 0; h < actual_ncols2; h++) {
      for (int i = 0; i < valid_ncols1; i++) {
        const int qi = h * NCOLS1 + i;

        // Phase A.1: Parallel max reduction across work-items
        float local_max = SOFTMAX_MIN;
        for (int j = local_id; j < nrows_kv; j += local_size) {
          local_max = fmax(local_max, KQ_tile[qi][j]);
        }
        float new_max = reduce_max_impl(local_max, l_max_tmp);

        // Combine with previous running max
        float prev_max = l_max_val[qi];
        new_max = fmax(new_max, prev_max);

        // Phase A.2: Compute correction factor
        float correction = 1.0f;
        if (new_max != prev_max) {
          correction = exp(prev_max - new_max);
        }

        // Phase A.3: Parallel exp computation and sum reduction
        float local_exp_sum = 0.0f;
        for (int j = local_id; j < nrows_kv; j += local_size) {
          float exp_val = exp(KQ_tile[qi][j] - new_max);
          KQ_tile[qi][j] = exp_val;  // Store exp value for VKQ computation
          local_exp_sum += exp_val;
        }
        float tile_exp_sum = reduce_add_impl(local_exp_sum, l_sum_tmp);

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

    // Load V tile cooperatively using float4 vectorized loads (ONCE per KV head!)
    // No need to zero unused rows — VKQ accumulation loop uses nrows_kv
    for (int j = 0; j < nrows_kv; j++) {
      const int v_row_offset = kv_head_offset + (kv_start + j) * head_dim;
      for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        const float4 val = *((__global const float4*)(value + v_row_offset + d4 * 4));
        *((__local float4*)&V_tile[j][d4 * 4]) = val;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Accumulate VKQ: VKQ[qi][d] += sum_j KQ_tile[qi][j] * V_tile[j][d]
    // Use float4 for V_tile reads
    for (int h = 0; h < actual_ncols2; h++) {
      for (int i = 0; i < valid_ncols1; i++) {
        const int qi = h * NCOLS1 + i;
        for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
          float4 vkq_acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
          for (int j = 0; j < nrows_kv; j++) {
            const float kq_exp = KQ_tile[qi][j];
            const float4 v_val = *((__local float4*)&V_tile[j][d4 * 4]);
            vkq_acc.x += kq_exp * v_val.x;
            vkq_acc.y += kq_exp * v_val.y;
            vkq_acc.z += kq_exp * v_val.z;
            vkq_acc.w += kq_exp * v_val.w;
          }
          VKQ[qi][d4 * 4]     += vkq_acc.x;
          VKQ[qi][d4 * 4 + 1] += vkq_acc.y;
          VKQ[qi][d4 * 4 + 2] += vkq_acc.z;
          VKQ[qi][d4 * 4 + 3] += vkq_acc.w;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Final normalization: output = VKQ / exp_sum
  // Write results to global memory using float4 vectorized stores
  for (int h = 0; h < actual_ncols2; h++) {
    const int q_head_id = first_q_head + h;
    const int query_batch_offset = batch_id * num_heads_q * seqlen_q * head_dim;
    const int output_head_offset = query_batch_offset + q_head_id * seqlen_q * head_dim;

    for (int i = 0; i < valid_ncols1; i++) {
      const int qi = h * NCOLS1 + i;
      const int out_row_offset = output_head_offset + (q_start + i) * head_dim;
      const float inv_exp_sum = 1.0f / l_exp_sum[qi];
      for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        const float4 out_val = (float4)(
          VKQ[qi][d4 * 4]     * inv_exp_sum,
          VKQ[qi][d4 * 4 + 1] * inv_exp_sum,
          VKQ[qi][d4 * 4 + 2] * inv_exp_sum,
          VKQ[qi][d4 * 4 + 3] * inv_exp_sum
        );
        *((__global float4*)(output + out_row_offset + d4 * 4)) = out_val;
      }
    }
  }
}