// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Tiwari <anup.tiwari@samsung.com>
 *
 * @file    flash_attention_decode_fp16_adreno.cl
 * @date    28 April 2026
 * @brief   Decode-optimized Flash Attention kernel for FP16 on Adreno GPU
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Anup Tiwari <anup.tiwari@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * Phase 6 + Adreno optimizations: Decode-specific kernel with split-KV approach
 * 
 * Adreno-optimized variant with:
 * - Sub-group reductions (cl_qcom_subgroups/cl_khr_subgroups) for hardware-accelerated reductions
 * - half4 vectorized loads for 4× memory bandwidth
 * - Work-group size 128 (Adreno optimal)
 * - Score caching to eliminate redundant Q·K computation
 * - native_exp() for faster exp computation
 * 
 * Design:
 * - When seqlen_q == 1 (decode), split KV sequence across multiple work-groups
 * - Each work-group computes partial (max_val, exp_sum, VKQ_partial) for its KV chunk
 * - Partials are written to a temporary global buffer
 * - A reduction phase combines partials using log-sum-exp trick
 * - KV_max skip: Skip tiles where kv_start >= kv_max (useful for batched sequences)
 * - FP16 storage with FP32 accumulation for numerical stability
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Enable sub-group extension if available (for hardware-accelerated reductions)
// Adreno 6xx uses cl_qcom_subgroups, Adreno 7xx+/8xx uses cl_khr_subgroups
#ifdef CL_QCOM_SUBGROUPS
#pragma OPENCL EXTENSION cl_qcom_subgroups : enable
#define USE_SUBGROUPS 1
#elif defined(CL_KHR_SUBGROUPS)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define USE_SUBGROUPS 1
#endif

#define SOFTMAX_MIN -1e30f  // Use float for softmax state (FP32 accumulation)

// Local memory size for Q cache (head_dim typically 128)
#define HEAD_DIM 128
#define HEAD_DIM4 (HEAD_DIM / 4)

// KV rows processed per work-group (tile size)
#define KV_TILE_SIZE 64

// --- Sub-group reduction helper functions ---

#ifdef USE_SUBGROUPS
// Sub-group accelerated max reduction for Adreno GPUs
float reduce_max_impl_decode(float value, __local float *l_max_tmp) {
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
float reduce_add_impl_decode(float value, __local float *l_sum_tmp) {
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
#endif

/**
 * @brief Decode kernel: Each work-group processes a chunk of KV sequence
 * 
 * Work-group mapping:
 * - global_work_size = batch * num_heads_q * num_kv_tiles * work_group_size
 * - Each work-group computes partial attention for one (batch, head) pair
 * - num_kv_tiles = ceil(seqlen_k / KV_TILE_SIZE)
 * 
 * Adreno optimizations:
 * - Work-group size 128 (better occupancy than 64)
 * - half4 vectorized loads for Q, K, V
 * - Sub-group reductions instead of manual barrier-based reductions
 * - native_exp() for faster exp computation
 */
__attribute__((work_group_size_hint(128, 1, 1)))
__kernel void flash_attention_decode_fp16_adreno(
    __global const half *query,          // [batch, num_heads_q, 1, head_dim]
    __global const half *key,            // [batch, num_heads_kv, seqlen_k, head_dim]
    __global const half *value,          // [batch, num_heads_kv, seqlen_k, head_dim]
    __global half *output,               // [batch, num_heads_q, 1, head_dim]
    __global float *partials_max,        // [batch, num_heads_q, num_kv_tiles] - FP32
    __global float *partials_sum,        // [batch, num_heads_q, num_kv_tiles] - FP32
    __global float *partials_vkq,        // [batch, num_heads_q, num_kv_tiles, head_dim] - FP32
    const int seqlen_q,                  // Must be 1 for decode
    const int seqlen_k,                  // KV sequence length
    const int head_dim,                  // Dimension of each head
    const int num_heads_q,               // Number of query heads
    const int num_heads_kv,              // Number of key/value heads
    const int batch,                     // Batch size
    const float scale,                   // Scaling factor (1/sqrt(head_dim))
    const int kv_max,                    // Valid KV positions (for variable length batches)
    const int num_kv_tiles               // Number of KV tiles
) {
    // Local memory for Q cache (half4 aligned)
    __local half4 local_q4[HEAD_DIM4];  // 32 x 8 = 256 bytes
    
    // Local memory for partial VKQ accumulation (FP32 for precision)
    __local float local_vkq[HEAD_DIM];  // 128 x 4 = 512 bytes
    
    // Local memory for score caching (eliminates redundant Q·K computation)
    __local float local_scores[KV_TILE_SIZE];  // 64 x 4 = 256 bytes
    
    // Reduction buffers (sized for max 128 work-items / 32 sub-group size = 4 sub-groups)
    __local float l_max_tmp[4];
    __local float l_sum_tmp[4];
    
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);
    
    // Calculate indices
    // Work-groups are laid out as: [batch][num_heads_q][num_kv_tiles]
    const int total_groups = batch * num_heads_q * num_kv_tiles;
    
    if (global_id >= total_groups * local_size) return;
    
    const int group_id = global_id / local_size;
    const int tile_id = group_id % num_kv_tiles;
    const int head_group_id = group_id / num_kv_tiles;
    const int head_id = head_group_id % num_heads_q;
    const int batch_id = head_group_id / num_heads_q;
    
    // For GQA, map query head to corresponding key/value head
    const int kv_head_id = head_id * num_heads_kv / num_heads_q;
    
    // KV tile range for this work-group
    const int kv_start = tile_id * KV_TILE_SIZE;
    const int kv_end = min(kv_start + KV_TILE_SIZE, min(seqlen_k, kv_max));
    
    // Early exit if this tile is outside valid KV range
    if (kv_start >= kv_max || kv_start >= seqlen_k) {
        // Write identity values for reduction (max = -inf, sum = 0, vkq = 0)
        if (local_id == 0) {
            const int partial_offset = batch_id * num_heads_q * num_kv_tiles + 
                                       head_id * num_kv_tiles + tile_id;
            partials_max[partial_offset] = SOFTMAX_MIN;
            partials_sum[partial_offset] = 0.0f;
        }
        for (int d = local_id; d < head_dim; d += local_size) {
            const int vkq_offset = batch_id * num_heads_q * num_kv_tiles * head_dim +
                                   head_id * num_kv_tiles * head_dim +
                                   tile_id * head_dim + d;
            partials_vkq[vkq_offset] = 0.0f;
        }
        return;
    }
    
    // Calculate offsets
    const int query_offset = batch_id * num_heads_q * head_dim + head_id * head_dim;
    const int kv_head_offset = batch_id * num_heads_kv * seqlen_k * head_dim +
                               kv_head_id * seqlen_k * head_dim;
    
    // Load Q into local memory using half4 vectorized loads (all work-items cooperate)
    for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        local_q4[d4] = *((__global const half4*)(query + query_offset + d4 * 4));
    }
    
    // Initialize local VKQ to zero
    for (int d = local_id; d < head_dim; d += local_size) {
        local_vkq[d] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute attention for this KV tile
    // Each work-item processes a subset of KV rows, then we reduce
    float thread_max = SOFTMAX_MIN;
    float thread_sum = 0.0f;
    
    // Process KV rows assigned to this work-item
    // Cache scores in local memory to eliminate redundant Q·K computation
    for (int k = kv_start + local_id; k < kv_end; k += local_size) {
        const int idx = k - kv_start;
        const int k_offset = kv_head_offset + k * head_dim;
        
        // Compute Q·K for this KV row using FP32 accumulation with half4 loads
        float score = 0.0f;
        
        // Use half4 vectorized loads for bandwidth
        for (int d4 = 0; d4 < HEAD_DIM4; d4++) {
            half4 q_val = local_q4[d4];
            half4 k_val = *((__global const half4*)(key + k_offset + d4 * 4));
            // FP32 accumulation for precision (dot product)
            score += (float)q_val.s0 * (float)k_val.s0 +
                     (float)q_val.s1 * (float)k_val.s1 +
                     (float)q_val.s2 * (float)k_val.s2 +
                     (float)q_val.s3 * (float)k_val.s3;
        }
        score *= scale;
        
        // Cache score for reuse in second pass
        local_scores[idx] = score;
        
        // Track max for this thread
        thread_max = fmax(thread_max, score);
    }
    
    // Reduce max across work-group using sub-groups or manual reduction
    const float tile_max = 
#ifdef USE_SUBGROUPS
        reduce_max_impl_decode(thread_max, l_max_tmp);
#else
    {
        // Manual parallel reduction fallback
        __local float local_max[128];
        local_max[local_id] = thread_max;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int stride = local_size / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
                local_max[local_id] = fmax(local_max[local_id], local_max[local_id + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        local_max[0];
    }
#endif
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Second pass: use cached scores (no Q·K recomputation, no K re-read)
    for (int k = kv_start + local_id; k < kv_end; k += local_size) {
        const int idx = k - kv_start;
        const int v_offset = kv_head_offset + k * head_dim;
        
        // Use cached score and native_exp for faster exp computation
        const float exp_score = native_exp(local_scores[idx] - tile_max);
        thread_sum += exp_score;
        
        // Accumulate weighted V into local VKQ (FP32 accumulation)
        // Use half4 vectorized loads for bandwidth
        for (int d4 = 0; d4 < HEAD_DIM4; d4++) {
            half4 v_val = *((__global const half4*)(value + v_offset + d4 * 4));
            local_vkq[d4 * 4]     += exp_score * (float)v_val.s0;
            local_vkq[d4 * 4 + 1] += exp_score * (float)v_val.s1;
            local_vkq[d4 * 4 + 2] += exp_score * (float)v_val.s2;
            local_vkq[d4 * 4 + 3] += exp_score * (float)v_val.s3;
        }
    }
    
    // Reduce sum across work-group
    const float tile_sum = 
#ifdef USE_SUBGROUPS
        reduce_add_impl_decode(thread_sum, l_sum_tmp);
#else
    {
        // Manual parallel reduction fallback
        __local float local_sum[128];
        local_sum[local_id] = thread_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int stride = local_size / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
                local_sum[local_id] += local_sum[local_id + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        local_sum[0];
    }
#endif
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Write partials to global memory
    if (local_id == 0) {
        const int partial_offset = batch_id * num_heads_q * num_kv_tiles + 
                                   head_id * num_kv_tiles + tile_id;
        partials_max[partial_offset] = tile_max;
        partials_sum[partial_offset] = tile_sum;
    }
    
    // Write partial VKQ to global memory (FP32)
    for (int d = local_id; d < head_dim; d += local_size) {
        const int vkq_offset = batch_id * num_heads_q * num_kv_tiles * head_dim +
                               head_id * num_kv_tiles * head_dim +
                               tile_id * head_dim + d;
        partials_vkq[vkq_offset] = local_vkq[d];
    }
}

/**
 * @brief Reduction kernel: Combine partials using log-sum-exp trick
 * 
 * This kernel is launched after the decode kernel to combine all partial results.
 * Each work-group processes one (batch, head) pair.
 * 
 * Algorithm (log-sum-exp):
 * 1. Find global max across all tiles
 * 2. Compute correction factor for each tile: exp(tile_max - global_max)
 * 3. Scale each tile's sum and VKQ by correction factor
 * 4. Sum across all tiles
 * 5. Normalize and write to output
 * 
 * Adreno optimizations:
 * - Sub-group reductions
 * - half4 vectorized stores for output
 */
__attribute__((work_group_size_hint(128, 1, 1)))
__kernel void flash_attention_decode_reduce_fp16_adreno(
    __global const float *partials_max,   // [batch, num_heads_q, num_kv_tiles] - FP32
    __global const float *partials_sum,   // [batch, num_heads_q, num_kv_tiles] - FP32
    __global const float *partials_vkq,   // [batch, num_heads_q, num_kv_tiles, head_dim] - FP32
    __global half *output,                // [batch, num_heads_q, 1, head_dim] - FP16 output
    const int num_kv_tiles,
    const int head_dim,
    const int num_heads_q
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);
    
    // Each work-group handles one (batch, head) pair
    const int group_id = get_group_id(0);
    const int head_id = group_id % num_heads_q;
    const int batch_id = group_id / num_heads_q;
    
    // Local memory for reduction
    __local float l_max_tmp[4];
    __local float l_sum_tmp[4];
    __local float local_vkq[HEAD_DIM];  // head_dim
    
    // Initialize local VKQ
    for (int d = local_id; d < head_dim; d += local_size) {
        local_vkq[d] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 1: Find global max across all tiles
    float thread_max = SOFTMAX_MIN;
    for (int t = local_id; t < num_kv_tiles; t += local_size) {
        const int offset = batch_id * num_heads_q * num_kv_tiles + 
                          head_id * num_kv_tiles + t;
        thread_max = fmax(thread_max, partials_max[offset]);
    }
    
    const float global_max = 
#ifdef USE_SUBGROUPS
        reduce_max_impl_decode(thread_max, l_max_tmp);
#else
    {
        __local float local_max[128];
        local_max[local_id] = thread_max;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int stride = local_size / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
                local_max[local_id] = fmax(local_max[local_id], local_max[local_id + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        local_max[0];
    }
#endif
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 2: Compute weighted sum using log-sum-exp correction
    float thread_sum = 0.0f;
    for (int t = local_id; t < num_kv_tiles; t += local_size) {
        const int offset = batch_id * num_heads_q * num_kv_tiles + 
                          head_id * num_kv_tiles + t;
        const float tile_max = partials_max[offset];
        const float tile_sum = partials_sum[offset];
        
        // Correction factor: exp(tile_max - global_max)
        // Use native_exp for faster computation (Adreno optimization)
        const float correction = native_exp(tile_max - global_max);
        const float corrected_sum = tile_sum * correction;
        thread_sum += corrected_sum;
        
        // Accumulate corrected VKQ - parallelize head_dim across work-items
        for (int d = local_id; d < head_dim; d += local_size) {
            const int vkq_offset = batch_id * num_heads_q * num_kv_tiles * head_dim +
                                   head_id * num_kv_tiles * head_dim +
                                   t * head_dim + d;
            local_vkq[d] += correction * partials_vkq[vkq_offset];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce sum across work-group
    const float global_sum = 
#ifdef USE_SUBGROUPS
        reduce_add_impl_decode(thread_sum, l_sum_tmp);
#else
    {
        __local float local_sum[128];
        local_sum[local_id] = thread_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int stride = local_size / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
                local_sum[local_id] += local_sum[local_id + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        local_sum[0];
    }
#endif
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 3: Normalize and write output (convert to FP16 at final output)
    const float inv_sum = 1.0f / global_sum;
    const int output_offset = batch_id * num_heads_q * head_dim + head_id * head_dim;
    
    // Use half4 vectorized stores for bandwidth
    for (int d4 = local_id; d4 < HEAD_DIM4; d4 += local_size) {
        half4 out_val;
        out_val.s0 = (half)(local_vkq[d4 * 4] * inv_sum);
        out_val.s1 = (half)(local_vkq[d4 * 4 + 1] * inv_sum);
        out_val.s2 = (half)(local_vkq[d4 * 4 + 2] * inv_sum);
        out_val.s3 = (half)(local_vkq[d4 * 4 + 3] * inv_sum);
        *((__global half4*)(output + output_offset + d4 * 4)) = out_val;
    }
}