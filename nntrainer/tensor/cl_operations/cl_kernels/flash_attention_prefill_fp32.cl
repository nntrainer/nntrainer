// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Tiwari <anup.tiwari@samsung.com>
 *
 * @file    flash_attention_prefill_fp32.cl
 * @date    23 April 2026
 * @brief   Tiled GEMM prefill kernel for flash attention (FP32)
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Anup Tiwari <anup.tiwari@samsung.com>
 *
 * Tiled GEMM approach with online softmax for prefill attention.
 * Each work-group processes NCOLS1 Q tokens against tiles of K/V.
 * Online softmax avoids the two-pass approach, reading K only once.
 * Cooperative loading and KQ computation across work-items.
 *
 * Local memory layout (NCOLS1=4, NBATCH_FA=16, HEAD_DIM=128):
 *   Q_tile:       4 x 128 x 4 = 2 KB
 *   K_tile:      16 x 128 x 4 = 8 KB
 *   V_tile:      16 x 128 x 4 = 8 KB
 *   KQ_tile:      4 x  16 x 4 = 256 B
 *   VKQ:          4 x 128 x 4 = 2 KB
 *   l_max_val:    4 x 4 = 16 B
 *   l_exp_sum:    4 x 4 = 16 B
 *   l_correction: 4 x 4 = 16 B
 *   Total: ~20.3 KB (fits in typical 32KB local memory)
 */

#define SOFTMAX_MIN -1e30f

// Configuration constants
// NCOLS1: Number of Q rows (tokens) processed per work-group
// NBATCH_FA: Number of K/V rows processed per tile (KV tile size)
// HEAD_DIM: Dimension of each attention head

#ifndef NCOLS1
#define NCOLS1 4
#endif

#ifndef NBATCH_FA
#define NBATCH_FA 16
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

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
    const float scale) {

  // Local memory tiles — declared inside kernel function for OpenCL compliance
  __local float Q_tile[NCOLS1][HEAD_DIM];        // Q tile: NCOLS1 x HEAD_DIM
  __local float K_tile[NBATCH_FA][HEAD_DIM];      // K tile: NBATCH_FA x HEAD_DIM
  __local float V_tile[NBATCH_FA][HEAD_DIM];      // V tile: NBATCH_FA x HEAD_DIM
  __local float KQ_tile[NCOLS1][NBATCH_FA];       // KQ dot products: NCOLS1 x NBATCH_FA
  __local float VKQ[NCOLS1][HEAD_DIM];            // VKQ accumulator: NCOLS1 x HEAD_DIM
  __local float l_max_val[NCOLS1];                // Running max per Q row
  __local float l_exp_sum[NCOLS1];                // Running exp_sum per Q row
  __local float l_correction[NCOLS1];             // Correction factor for VKQ rescaling

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

  // Initialize VKQ accumulator in local memory
  for (int i = 0; i < NCOLS1; i++) {
    for (int d = local_id; d < HEAD_DIM; d += local_size) {
      VKQ[i][d] = 0.0f;
    }
  }

  // Load Q tile cooperatively
  for (int i = 0; i < ncols1; i++) {
    const int q_row_offset = query_head_offset + (q_start + i) * head_dim;
    for (int d = local_id; d < head_dim; d += local_size) {
      Q_tile[i][d] = query[q_row_offset + d];
    }
  }
  // Zero out unused Q rows (when ncols1 < NCOLS1)
  for (int i = ncols1; i < NCOLS1; i++) {
    for (int d = local_id; d < head_dim; d += local_size) {
      Q_tile[i][d] = 0.0f;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Process KV in tiles of NBATCH_FA
  for (int kv_start = 0; kv_start < seqlen_k; kv_start += NBATCH_FA) {
    const int nrows_kv = min(seqlen_k - kv_start, NBATCH_FA);

    // Load K tile cooperatively
    for (int j = 0; j < nrows_kv; j++) {
      const int k_row_offset = kv_head_offset + (kv_start + j) * head_dim;
      for (int d = local_id; d < head_dim; d += local_size) {
        K_tile[j][d] = key[k_row_offset + d];
      }
    }
    // Zero out unused K rows
    for (int j = nrows_kv; j < NBATCH_FA; j++) {
      for (int d = local_id; d < head_dim; d += local_size) {
        K_tile[j][d] = 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute KQ tile: KQ_tile[i][j] = dot(Q_tile[i], K_tile[j]) * scale
    // Each work-item computes complete dot products for a subset of (i,j) pairs
    for (int i = 0; i < NCOLS1; i++) {
      for (int j = local_id; j < NBATCH_FA; j += local_size) {
        float kq_acc = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          kq_acc += Q_tile[i][d] * K_tile[j][d];
        }
        KQ_tile[i][j] = kq_acc * scale;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax — Phase 1: work-item 0 computes softmax for all Q rows
    // Note: Causal masking is not applied here to match the existing CPU reference
    // implementation. To enable causal masking, add: if (kv_start + j > q_start + i)
    // set KQ_tile[i][j] = SOFTMAX_MIN before finding new_max.
    if (local_id == 0) {
      for (int i = 0; i < ncols1; i++) {
        // Find new max for this Q row across the current K tile
        float new_max = l_max_val[i];
        for (int j = 0; j < nrows_kv; j++) {
          new_max = fmax(new_max, KQ_tile[i][j]);
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
          const float exp_val = exp(KQ_tile[i][j] - l_max_val[i]);
          KQ_tile[i][j] = exp_val;
          l_exp_sum[i] += exp_val;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax — Phase 2: all work-items cooperatively rescale VKQ
    // Each work-item handles a subset of the HEAD_DIM dimension
    for (int i = 0; i < ncols1; i++) {
      for (int d = local_id; d < HEAD_DIM; d += local_size) {
        VKQ[i][d] *= l_correction[i];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Load V tile cooperatively
    for (int j = 0; j < nrows_kv; j++) {
      const int v_row_offset = kv_head_offset + (kv_start + j) * head_dim;
      for (int d = local_id; d < head_dim; d += local_size) {
        V_tile[j][d] = value[v_row_offset + d];
      }
    }
    // Zero out unused V rows
    for (int j = nrows_kv; j < NBATCH_FA; j++) {
      for (int d = local_id; d < head_dim; d += local_size) {
        V_tile[j][d] = 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Accumulate VKQ: VKQ[i][d] += sum_j KQ_tile[i][j] * V_tile[j][d]
    // Distribute across work-items by d dimension
    for (int i = 0; i < ncols1; i++) {
      for (int d = local_id; d < head_dim; d += local_size) {
        float vkq_acc = 0.0f;
        for (int j = 0; j < nrows_kv; j++) {
          vkq_acc += KQ_tile[i][j] * V_tile[j][d];
        }
        VKQ[i][d] += vkq_acc;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Final normalization: output = VKQ / exp_sum
  // Write results to global memory
  for (int i = 0; i < ncols1; i++) {
    const int out_row_offset = output_head_offset + (q_start + i) * head_dim;
    for (int d = local_id; d < HEAD_DIM; d += local_size) {
      output[out_row_offset + d] = VKQ[i][d] / l_exp_sum[i];
    }
  }
}