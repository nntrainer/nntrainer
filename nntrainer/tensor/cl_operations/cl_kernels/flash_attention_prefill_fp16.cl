// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Tiwari <anup.tiwari@samsung.com>
 *
 * @file    flash_attention_prefill_fp16.cl
 * @date    24 April 2026
 * @brief   Tiled GEMM prefill kernel for flash attention (FP16)
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Anup Tiwari <anup.tiwari@samsung.com>
 *
 * Tiled GEMM approach with online softmax for prefill attention (FP16).
 * Phase 3: Uses half for storage (Q/K/V tiles in local memory) but float
 * for all accumulation (KQ dot products, VKQ, softmax state).
 * half2 vectorized loads for 2× memory bandwidth on global and local access.
 *
 * Local memory layout (NCOLS1=4, NBATCH_FA=16, HEAD_DIM=128):
 *   Q_tile:   4 x 128 x 2 = 1 KB
 *   K_tile:  16 x 128 x 2 = 4 KB
 *   V_tile:  16 x 128 x 2 = 4 KB
 *   KQ_tile:  4 x  16 x 2 = 128 B  (half — intermediate, overwritten by exp)
 *   VKQ:      4 x 128 x 4 = 2 KB  (float accumulator)
 *   l_max_val:    4 x 4 = 16 B
 *   l_exp_sum:    4 x 4 = 16 B
 *   l_correction: 4 x 4 = 16 B
 *   Total: ~11.2 KB (fits easily in 32KB local memory)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SOFTMAX_MIN -1e30f

// Configuration constants
#ifndef NCOLS1
#define NCOLS1 4
#endif

#ifndef NBATCH_FA
#define NBATCH_FA 16
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

// Ensure HEAD_DIM is even for half2 vectorized access
#if (HEAD_DIM % 2) != 0
#error "HEAD_DIM must be even for half2 vectorized access"
#endif

#define HEAD_DIM2 (HEAD_DIM / 2)

__kernel void flash_attention_prefill_fp16(
    __global const half *query,
    __global const half *key,
    __global const half *value,
    __global half *output,
    const int seqlen_q,
    const int seqlen_k,
    const int head_dim,
    const int num_heads_q,
    const int num_heads_kv,
    const int batch,
    const float scale) {

  // Local memory tiles — declared inside kernel function for OpenCL compliance
  // Q/K/V stored as half for compact storage; KQ_tile stored as half (intermediate)
  __local half Q_tile[NCOLS1][HEAD_DIM];        // Q tile: NCOLS1 x HEAD_DIM (half)
  __local half K_tile[NBATCH_FA][HEAD_DIM];      // K tile: NBATCH_FA x HEAD_DIM (half)
  __local half V_tile[NBATCH_FA][HEAD_DIM];      // V tile: NBATCH_FA x HEAD_DIM (half)
  __local half KQ_tile[NCOLS1][NBATCH_FA];       // KQ dot products: NCOLS1 x NBATCH_FA (half)
  // VKQ accumulator in FP32 — critical for numerical accuracy
  __local float VKQ[NCOLS1][HEAD_DIM];           // VKQ accumulator: NCOLS1 x HEAD_DIM (float!)
  __local float l_max_val[NCOLS1];               // Running max per Q row (float)
  __local float l_exp_sum[NCOLS1];               // Running exp_sum per Q row (float)
  __local float l_correction[NCOLS1];            // Correction factor (float)

  const int group_id = get_group_id(0);
  const int local_id = get_local_id(0);
  const int local_size = get_local_size(0);

  // Total number of work-groups: batch * num_heads_q * ceil(seqlen_q / NCOLS1)
  const int num_q_groups = (seqlen_q + NCOLS1 - 1) / NCOLS1;
  const int total_groups = batch * num_heads_q * num_q_groups;

  if (group_id >= total_groups) return;

  // Decode group_id into batch, head, and q_group indices
  const int q_group = group_id % num_q_groups;
  const int head_batch = group_id / num_q_groups;
  const int head_id = head_batch % num_heads_q;
  const int batch_id = head_batch / num_heads_q;

  // Map query head to KV head (GQA)
  const int kv_head_id = head_id * num_heads_kv / num_heads_q;

  // Starting Q row for this work-group
  const int q_start = q_group * NCOLS1;

  // Number of valid Q rows in this group (may be less than NCOLS1 at boundary)
  const int ncols1 = min(seqlen_q - q_start, NCOLS1);

  // Calculate base offsets
  const int query_batch_offset = batch_id * num_heads_q * seqlen_q * head_dim;
  const int query_head_offset = query_batch_offset + head_id * seqlen_q * head_dim;

  const int kv_batch_offset = batch_id * num_heads_kv * seqlen_k * head_dim;
  const int kv_head_offset = kv_batch_offset + kv_head_id * seqlen_k * head_dim;

  const int output_head_offset = query_head_offset;

  // Initialize online softmax state in local memory
  if (local_id < NCOLS1) {
    l_max_val[local_id] = SOFTMAX_MIN;
    l_exp_sum[local_id] = 0.0f;
    l_correction[local_id] = 1.0f;
  }

  // Initialize VKQ accumulator in FP32
  for (int i = 0; i < NCOLS1; i++) {
    for (int d = local_id; d < HEAD_DIM; d += local_size) {
      VKQ[i][d] = 0.0f;
    }
  }

  // Load Q tile cooperatively using half2 vectorized loads
  for (int i = 0; i < ncols1; i++) {
    const int q_row_offset = query_head_offset + (q_start + i) * head_dim;
    for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
      const half2 val = *((__global const half2*)(query + q_row_offset + d2 * 2));
      *((__local half2*)&Q_tile[i][d2 * 2]) = val;
    }
  }
  // Zero out unused Q rows (when ncols1 < NCOLS1)
  for (int i = ncols1; i < NCOLS1; i++) {
    for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
      *((__local half2*)&Q_tile[i][d2 * 2]) = (half2)(0.0h, 0.0h);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Process KV in tiles of NBATCH_FA
  for (int kv_start = 0; kv_start < seqlen_k; kv_start += NBATCH_FA) {
    const int nrows_kv = min(seqlen_k - kv_start, NBATCH_FA);

    // Load K tile cooperatively using half2 vectorized loads
    for (int j = 0; j < nrows_kv; j++) {
      const int k_row_offset = kv_head_offset + (kv_start + j) * head_dim;
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        const half2 val = *((__global const half2*)(key + k_row_offset + d2 * 2));
        *((__local half2*)&K_tile[j][d2 * 2]) = val;
      }
    }
    // Zero out unused K rows
    for (int j = nrows_kv; j < NBATCH_FA; j++) {
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        *((__local half2*)&K_tile[j][d2 * 2]) = (half2)(0.0h, 0.0h);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute KQ tile: KQ_tile[i][j] = dot(Q_tile[i], K_tile[j]) * scale
    // Phase 3: Accumulate in FP32 for numerical accuracy
    // Use half2 vectorized loads from local memory for 2× bandwidth
    for (int i = 0; i < NCOLS1; i++) {
      for (int j = local_id; j < NBATCH_FA; j += local_size) {
        float kq_acc = 0.0f;  // FP32 accumulation!
        for (int d2 = 0; d2 < HEAD_DIM2; d2++) {
          const half2 q_val = *((__local half2*)&Q_tile[i][d2 * 2]);
          const half2 k_val = *((__local half2*)&K_tile[j][d2 * 2]);
          // FMA in FP32 from half inputs
          kq_acc += (float)q_val.x * (float)k_val.x;
          kq_acc += (float)q_val.y * (float)k_val.y;
        }
        KQ_tile[i][j] = (half)(kq_acc * scale);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax — Phase 1: work-item 0 computes softmax for all Q rows
    if (local_id == 0) {
      for (int i = 0; i < ncols1; i++) {
        // Find new max for this Q row across the current K tile
        float new_max = l_max_val[i];
        for (int j = 0; j < nrows_kv; j++) {
          new_max = fmax(new_max, (float)KQ_tile[i][j]);
        }

        // Compute correction factor for rescaling VKQ
        if (new_max != l_max_val[i]) {
          l_correction[i] = exp(l_max_val[i] - new_max);
          l_exp_sum[i] *= l_correction[i];
          l_max_val[i] = new_max;
        } else {
          l_correction[i] = 1.0f;
        }

        // Compute exp(KQ - max) and accumulate exp_sum
        // Replace KQ_tile values with exp values for VKQ accumulation
        for (int j = 0; j < nrows_kv; j++) {
          const float exp_val = exp((float)KQ_tile[i][j] - l_max_val[i]);
          KQ_tile[i][j] = (half)exp_val;
          l_exp_sum[i] += exp_val;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax — Phase 2: all work-items cooperatively rescale VKQ (FP32)
    for (int i = 0; i < ncols1; i++) {
      for (int d = local_id; d < HEAD_DIM; d += local_size) {
        VKQ[i][d] *= l_correction[i];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Load V tile cooperatively using half2 vectorized loads
    for (int j = 0; j < nrows_kv; j++) {
      const int v_row_offset = kv_head_offset + (kv_start + j) * head_dim;
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        const half2 val = *((__global const half2*)(value + v_row_offset + d2 * 2));
        *((__local half2*)&V_tile[j][d2 * 2]) = val;
      }
    }
    // Zero out unused V rows
    for (int j = nrows_kv; j < NBATCH_FA; j++) {
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        *((__local half2*)&V_tile[j][d2 * 2]) = (half2)(0.0h, 0.0h);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Accumulate VKQ: VKQ[i][d] += sum_j KQ_tile[i][j] * V_tile[j][d]
    // Phase 3: Accumulate in FP32; use half2 for V_tile reads
    for (int i = 0; i < ncols1; i++) {
      for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
        float2 vkq_acc = (float2)(0.0f, 0.0f);
        for (int j = 0; j < nrows_kv; j++) {
          const float kq_exp = (float)KQ_tile[i][j];
          const half2 v_val = *((__local half2*)&V_tile[j][d2 * 2]);
          vkq_acc.x += kq_exp * (float)v_val.x;
          vkq_acc.y += kq_exp * (float)v_val.y;
        }
        VKQ[i][d2 * 2] += vkq_acc.x;
        VKQ[i][d2 * 2 + 1] += vkq_acc.y;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Final normalization: output = VKQ / exp_sum
  // Write results to global memory using half2 vectorized stores
  for (int i = 0; i < ncols1; i++) {
    const int out_row_offset = output_head_offset + (q_start + i) * head_dim;
    const float inv_exp_sum = 1.0f / l_exp_sum[i];
    for (int d2 = local_id; d2 < HEAD_DIM2; d2 += local_size) {
      const half2 out_val = (half2)(
        (half)(VKQ[i][d2 * 2] * inv_exp_sum),
        (half)(VKQ[i][d2 * 2 + 1] * inv_exp_sum)
      );
      *((__global half2*)(output + out_row_offset + d2 * 2)) = out_val;
    }
  }
}