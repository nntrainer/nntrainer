// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Tiwari <anup.tiwari@samsung.com>
 *
 * @file    flash_attention_decode_fp32.cl
 * @date    25 April 2026
 * @brief   Decode-optimized Flash Attention kernel for FP32
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Anup Tiwari <anup.tiwari@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * Phase 6: Decode-specific kernel with split-KV approach
 * 
 * Design:
 * - When seqlen_q == 1 (decode), split KV sequence across multiple work-groups
 * - Each work-group computes partial (max_val, exp_sum, VKQ_partial) for its KV chunk
 * - Partials are written to a temporary global buffer
 * - A reduction phase (CPU or separate kernel) combines partials using log-sum-exp trick
 * - KV_max skip: Skip tiles where kv_start >= kv_max (useful for batched sequences)
 */

#define SOFTMAX_MIN -1e30f

// Local memory size for Q cache (head_dim typically 128)
#define LOCAL_SIZE 256

// KV rows processed per work-group (tile size)
#define KV_TILE_SIZE 64

/**
 * @brief Decode kernel: Each work-group processes a chunk of KV sequence
 * 
 * Work-group mapping:
 * - global_work_size = batch * num_heads_q * num_kv_tiles
 * - Each work-group computes partial attention for one (batch, head) pair
 * - num_kv_tiles = ceil(seqlen_k / KV_TILE_SIZE)
 * 
 * Partial result structure (written to partials buffer):
 * - partial_max[batch, head, tile] = max attention score in this tile
 * - partial_sum[batch, head, tile] = sum of exp(scores - max) in this tile
 * - partial_vkq[batch, head, tile, d] = weighted sum of values for this tile
 */
__kernel void flash_attention_decode_fp32(
    __global const float *query,         // [batch, num_heads_q, 1, head_dim]
    __global const float *key,           // [batch, num_heads_kv, seqlen_k, head_dim]
    __global const float *value,         // [batch, num_heads_kv, seqlen_k, head_dim]
    __global float *output,              // [batch, num_heads_q, 1, head_dim]
    __global float *partials_max,        // [batch, num_heads_q, num_kv_tiles]
    __global float *partials_sum,        // [batch, num_heads_q, num_kv_tiles]
    __global float *partials_vkq,        // [batch, num_heads_q, num_kv_tiles, head_dim]
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
    // Local memory for Q cache
    __local float local_q[LOCAL_SIZE];
    
    // Local memory for partial VKQ accumulation (reduced at end)
    __local float local_vkq[LOCAL_SIZE];
    
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
    
    // Load Q into local memory (all work-items cooperate)
    for (int d = local_id; d < head_dim; d += local_size) {
        local_q[d] = query[query_offset + d];
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
    
    // Local memory for thread-level partial VKQ (avoid race conditions)
    // Each thread accumulates its own VKQ, then we reduce at the end
    // Note: For head_dim <= LOCAL_SIZE, each thread handles 1 or more dimensions
    // We'll accumulate VKQ directly in local_vkq with proper synchronization
    
    // Process KV rows assigned to this work-item
    for (int k = kv_start + local_id; k < kv_end; k += local_size) {
        const int k_offset = kv_head_offset + k * head_dim;
        
        // Compute Q·K for this KV row
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += local_q[d] * key[k_offset + d];
        }
        score *= scale;
        
        // Track max for this thread
        thread_max = fmax(thread_max, score);
    }
    
    // Reduce max across work-group
    // Use parallel reduction in local memory
    __local float local_max[64];  // Assuming local_size <= 64
    if (local_id < 64) {
        local_max[local_id] = SOFTMAX_MIN;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id < 64) {
        local_max[local_id] = thread_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel reduction for max
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_max[local_id] = fmax(local_max[local_id], local_max[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    const float tile_max = local_max[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Second pass: compute exp(scores - tile_max) and accumulate
    for (int k = kv_start + local_id; k < kv_end; k += local_size) {
        const int k_offset = kv_head_offset + k * head_dim;
        const int v_offset = k_offset;  // Same layout for K and V
        
        // Compute Q·K for this KV row
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += local_q[d] * key[k_offset + d];
        }
        score *= scale;
        
        // Compute exp(score - tile_max)
        const float exp_score = exp(score - tile_max);
        thread_sum += exp_score;
        
        // Accumulate weighted V into local VKQ
        // Each thread adds its contribution (atomic not needed - each thread writes different k)
        for (int d = 0; d < head_dim; d++) {
            local_vkq[d] += exp_score * value[v_offset + d];
        }
    }
    
    // Reduce sum across work-group
    __local float local_sum[64];
    if (local_id < 64) {
        local_sum[local_id] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id < 64) {
        local_sum[local_id] = thread_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel reduction for sum
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_sum[local_id] += local_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    const float tile_sum = local_sum[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Write partials to global memory
    if (local_id == 0) {
        const int partial_offset = batch_id * num_heads_q * num_kv_tiles + 
                                   head_id * num_kv_tiles + tile_id;
        partials_max[partial_offset] = tile_max;
        partials_sum[partial_offset] = tile_sum;
    }
    
    // Write partial VKQ to global memory
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
 */
__kernel void flash_attention_decode_reduce_fp32(
    __global const float *partials_max,   // [batch, num_heads_q, num_kv_tiles]
    __global const float *partials_sum,   // [batch, num_heads_q, num_kv_tiles]
    __global const float *partials_vkq,   // [batch, num_heads_q, num_kv_tiles, head_dim]
    __global float *output,               // [batch, num_heads_q, 1, head_dim]
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
    __local float local_max[64];
    __local float local_sum[64];
    __local float local_vkq[256];  // head_dim
    
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
    
    if (local_id < 64) {
        local_max[local_id] = thread_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel reduction for max
    for (int stride = min(32, local_size / 2); stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_max[local_id] = fmax(local_max[local_id], local_max[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    const float global_max = local_max[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 2: Compute weighted sum using log-sum-exp correction
    float thread_sum = 0.0f;
    for (int t = local_id; t < num_kv_tiles; t += local_size) {
        const int offset = batch_id * num_heads_q * num_kv_tiles + 
                          head_id * num_kv_tiles + t;
        const float tile_max = partials_max[offset];
        const float tile_sum = partials_sum[offset];
        
        // Correction factor: exp(tile_max - global_max)
        const float correction = exp(tile_max - global_max);
        const float corrected_sum = tile_sum * correction;
        thread_sum += corrected_sum;
        
        // Accumulate corrected VKQ
        for (int d = 0; d < head_dim; d++) {
            const int vkq_offset = batch_id * num_heads_q * num_kv_tiles * head_dim +
                                   head_id * num_kv_tiles * head_dim +
                                   t * head_dim + d;
            local_vkq[d] += correction * partials_vkq[vkq_offset];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce sum across work-group
    if (local_id < 64) {
        local_sum[local_id] = thread_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int stride = min(32, local_size / 2); stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_sum[local_id] += local_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    const float global_sum = local_sum[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 3: Normalize and write output
    const float inv_sum = 1.0f / global_sum;
    const int output_offset = batch_id * num_heads_q * head_dim + head_id * head_dim;
    
    for (int d = local_id; d < head_dim; d += local_size) {
        output[output_offset + d] = local_vkq[d] * inv_sum;
    }
}