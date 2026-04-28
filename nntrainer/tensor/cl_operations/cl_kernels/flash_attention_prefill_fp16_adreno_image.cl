// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Tiwari <anup.tiwari@samsung.com>
 *
 * @file    flash_attention_prefill_fp16_adreno_image.cl
 * @date    28 April 2026
 * @brief   Tiled GEMM prefill kernel using image objects for texture cache optimization
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Anup Tiwari <anup.tiwari@samsung.com>
 *
 * L4-2 Optimization: Image Objects for Texture Cache
 * 
 * This kernel uses OpenCL image objects (__read_only image2d_t) for K and V access,
 * which routes reads through the Adreno Texture Processor (TP) cache instead of L2 cache.
 * This provides:
 * - Separate cache path: TP cache (16-32 KB) is separate from L2 cache
 * - Hardware-accelerated format conversion: read_imageh() returns half4 natively
 * - Better cache locality for K/V data which is read multiple times
 *
 * Architecture: Cooperative processing (same as v1 kernel)
 * - All 128 work-items cooperate on NQ=4 Q rows
 * - Parallel softmax with sub-group reductions
 * - K/V tiles loaded via texture reads, Q via buffer (loaded once, reused)
 *
 * Image Layout:
 * - K_img: Width = HEAD_DIM/4 (32), Height = batch * num_heads_kv * seqlen_k
 * - V_img: Width = HEAD_DIM/4 (32), Height = batch * num_heads_kv * seqlen_k
 * - Q: Still uses buffer (loaded once at start, not worth image overhead)
 * - Output: Buffer (write-once, no cache benefit from image)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Enable sub-group extension if available
#ifdef CL_QCOM_SUBGROUPS
#pragma OPENCL EXTENSION cl_qcom_subgroups : enable
#define USE_SUBGROUPS 1
#elif defined(CL_KHR_SUBGROUPS)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define USE_SUBGROUPS 1
#endif

#define SOFTMAX_MIN -1e30f

// Configuration constants
#ifndef NCOLS1
#define NCOLS1 4
#endif

#ifndef NCOLS2
#define NCOLS2 1
#endif

#ifndef NBATCH_FA
#define NBATCH_FA 32
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#if (HEAD_DIM % 4) != 0
#error "HEAD_DIM must be multiple of 4 for half4 vectorized access"
#endif

#define HEAD_DIM4 (HEAD_DIM / 4)
#define HEAD_DIM2 (HEAD_DIM / 2)
#define NQ (NCOLS1 * NCOLS2)

// --- Parallel reduction helper functions (same as v1 kernel) ---

#ifdef USE_SUBGROUPS
float reduce_max_impl(float value, __local float *l_max_tmp) {
  float sg_max = sub_group_reduce_max(value);
  if (get_sub_group_local_id() == 0) {
    l_max_tmp[get_sub_group_id()] = sg_max;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

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

float reduce_add_impl(float value, __local float *l_sum_tmp) {
  float sg_sum = sub_group_reduce_add(value);
  if (get_sub_group_local_id() == 0) {
    l_sum_tmp[get_sub_group_id()] = sg_sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

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
// Manual parallel reduction fallback
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

__attribute__((work_group_size_hint(128, 1, 1)))
__kernel void flash_attention_prefill_fp16_adreno_image(
    __global const half *query,           // Q as buffer (loaded once)
    __read_only image2d_t K_img,          // K as image (texture cache)
    __read_only image2d_t V_img,          // V as image (texture cache)
    __global half *output,                // Output as buffer
    const int seqlen_q,
    const int seqlen_k,
    const int head_dim,
    const int num_heads_q,
    const int num_heads_kv,
    const int batch,
    const float scale,
    const int ncols2,
    const int head_group_offset,
    const int ncols1_runtime
) {
  // Sampler for image reads: non-normalized coordinates, clamp to edge, nearest filtering
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  // Local memory tiles
  __local half Q_tile[NQ][HEAD_DIM];
  __local half K_tile[NBATCH_FA][HEAD_DIM];
  __local half V_tile[NBATCH_FA][HEAD_DIM];
  __local float KQ_tile[NQ][NBATCH_FA];
  __local float VKQ[NQ][HEAD_DIM];
  __local float l_max_val[NQ];
  __local float l_exp_sum[NQ];
  __local float l_correction[NQ];
  __local float l_max_tmp[128];
  __local float l_sum_tmp[128];

  const int group_id = get_group_id(0);
  const int local_id = get_local_id(0);
  const int local_size = get_local_size(0);

  const int ncols1 = ncols1_runtime;
  const int num_q_groups = (seqlen_q + ncols1 - 1) / ncols1;
  const int total_groups = batch * num_heads_kv * num_q_groups;

  if (group_id >= total_groups) return;

  // Decode group_id
  const int q_group = group_id % num_q_groups;
  const int kv_head_batch = group_id / num_q_groups;
  const int kv_head_id = kv_head_batch % num_heads_kv;
  const int batch_id = kv_head_batch / num_heads_kv;

  // GQA parameters
  const int gqa_ratio = num_heads_q / num_heads_kv;
  const int first_q_head = kv_head_id * gqa_ratio + head_group_offset;
  const int valid_ncols2 = min(ncols2, num_heads_q - first_q_head);
  const int actual_ncols2 = min(valid_ncols2, gqa_ratio - head_group_offset);

  if (actual_ncols2 <= 0) return;

  const int q_start = q_group * ncols1;
  const int valid_ncols1 = min(seqlen_q - q_start, ncols1);

  // Initialize softmax state
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

  // Load Q tile from buffer (loaded once, reused across all KV tiles)
  for (int h = 0; h < actual_ncols2; h++) {
    const int q_head_id = first_q_head + h;
    const int query_batch_offset = batch_id * num_heads_q * seqlen_q * head_dim;
    const int query_head_offset = query_batch_offset + q_head_id * seqlen_q * head_dim;

    for (int i = 0; i < valid_ncols1; i++) {
      const int q_row_offset = query_head_offset + (q_start + i) * head_dim;
      for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        const half4 val = *((__global const half4*)(query + q_row_offset + d4 * 4));
        *((__local half4*)&Q_tile[h * NCOLS1 + i][d4 * 4]) = val;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Image base Y coordinate for this batch and KV head
  const int kv_img_base_y = (batch_id * num_heads_kv + kv_head_id) * seqlen_k;

  // Process KV in tiles
  for (int kv_start = 0; kv_start < seqlen_k; kv_start += NBATCH_FA) {
    const int nrows_kv = min(seqlen_k - kv_start, NBATCH_FA);

    // Load K tile from image (texture cache path)
    // Each work-item loads multiple elements across the HEAD_DIM dimension
    for (int j = 0; j < nrows_kv; j++) {
      const int kv_idx = kv_start + j;
      const int img_y = kv_img_base_y + kv_idx;
      
      for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        // read_imageh returns half4, x coordinate is the half4 index
        const half4 val = read_imageh(K_img, smp, (int2)(d4, img_y));
        *((__local half4*)&K_tile[j][d4 * 4]) = val;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Load V tile from image (texture cache path)
    for (int j = 0; j < nrows_kv; j++) {
      const int kv_idx = kv_start + j;
      const int img_y = kv_img_base_y + kv_idx;
      
      for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        const half4 val = read_imageh(V_img, smp, (int2)(d4, img_y));
        *((__local half4*)&V_tile[j][d4 * 4]) = val;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute KQ tile
    for (int h = 0; h < actual_ncols2; h++) {
      for (int i = 0; i < valid_ncols1; i++) {
        const int qi = h * NCOLS1 + i;
        for (int j = local_id; j < nrows_kv; j += local_size) {
          half4 kq_acc4 = (half4)(0.0h, 0.0h, 0.0h, 0.0h);
          for (int d4 = 0; d4 < HEAD_DIM4; d4++) {
            const half4 q_val = *((__local half4*)&Q_tile[qi][d4 * 4]);
            const half4 k_val = *((__local half4*)&K_tile[j][d4 * 4]);
            kq_acc4 += q_val * k_val;
          }
          float kq_acc = (float)kq_acc4.s0 + (float)kq_acc4.s1 +
                         (float)kq_acc4.s2 + (float)kq_acc4.s3;
          KQ_tile[qi][j] = kq_acc * scale;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel Online Softmax
    for (int h = 0; h < actual_ncols2; h++) {
      for (int i = 0; i < valid_ncols1; i++) {
        const int qi = h * NCOLS1 + i;

        // Max reduction
        float local_max = SOFTMAX_MIN;
        for (int j = local_id; j < nrows_kv; j += local_size) {
          local_max = fmax(local_max, KQ_tile[qi][j]);
        }
        float new_max = reduce_max_impl(local_max, l_max_tmp);

        float prev_max = l_max_val[qi];
        new_max = fmax(new_max, prev_max);

        // Correction factor
        float correction = 1.0f;
        if (new_max != prev_max) {
          correction = exp(prev_max - new_max);
        }

        // Exp and sum reduction
        float local_exp_sum = 0.0f;
        for (int j = local_id; j < nrows_kv; j += local_size) {
          float exp_val = native_exp(KQ_tile[qi][j] - new_max);
          KQ_tile[qi][j] = exp_val;
          local_exp_sum += exp_val;
        }
        float tile_exp_sum = reduce_add_impl(local_exp_sum, l_sum_tmp);

        if (local_id == 0) {
          l_correction[qi] = correction;
          l_exp_sum[qi] = l_exp_sum[qi] * correction + tile_exp_sum;
          l_max_val[qi] = new_max;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Apply correction to VKQ
        for (int d = local_id; d < HEAD_DIM; d += local_size) {
          VKQ[qi][d] *= correction;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Accumulate VKQ
    for (int h = 0; h < actual_ncols2; h++) {
      for (int i = 0; i < valid_ncols1; i++) {
        const int qi = h * NCOLS1 + i;
        for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
          float4 vkq_acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
          for (int j = 0; j < nrows_kv; j++) {
            const float kq_exp = KQ_tile[qi][j];
            const half4 v_val = *((__local half4*)&V_tile[j][d4 * 4]);
            const float4 v_f = convert_float4(v_val);
            vkq_acc += kq_exp * v_f;
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

  // Final normalization and output
  for (int h = 0; h < actual_ncols2; h++) {
    const int q_head_id = first_q_head + h;
    const int query_batch_offset = batch_id * num_heads_q * seqlen_q * head_dim;
    const int output_head_offset = query_batch_offset + q_head_id * seqlen_q * head_dim;

    for (int i = 0; i < valid_ncols1; i++) {
      const int qi = h * NCOLS1 + i;
      const int out_row_offset = output_head_offset + (q_start + i) * head_dim;
      const float inv_exp_sum = 1.0f / l_exp_sum[qi];
      for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        const half4 out_val = (half4)(
          (half)(VKQ[qi][d4 * 4]     * inv_exp_sum),
          (half)(VKQ[qi][d4 * 4 + 1] * inv_exp_sum),
          (half)(VKQ[qi][d4 * 4 + 2] * inv_exp_sum),
          (half)(VKQ[qi][d4 * 4 + 3] * inv_exp_sum)
        );
        *((__global half4*)(output + out_row_offset + d4 * 4)) = out_val;
      }
    }
  }
}