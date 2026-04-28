// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Tiwari <anup.tiwari@samsung.com>
 *
 * @file    flash_attention_prefill_fp16_adreno_v2.cl
 * @date    28 April 2026
 * @brief   FlashAttention-v2 style prefill kernel (FP16) optimized for Adreno GPU
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Anup Tiwari <anup.tiwari@samsung.com>
 *
 * L4-1: Per-Work-Item Q Row Processing (FlashAttention-v2 Architecture)
 *
 * Key differences from flash_attention_prefill_fp16_adreno.cl:
 * - Each work-item processes ONE Q row independently (vs cooperative NQ=4)
 * - BLOCK_SIZE_M=64 work-items per work-group (vs 128)
 * - Thread-local online softmax (no barriers inside KV loop!)
 * - No KQ_tile (scores computed and consumed immediately)
 * - VKQ accumulator in registers (not local memory)
 * - half8 vectorization with dot() intrinsic for QK computation
 * - Only ONE barrier per KV tile (after K/V load)
 *
 * Local memory comparison:
 * | Component     | Current (Coop) | L4-1 (Per-WI) |
 * |---------------|----------------|---------------|
 * | Q_tile        | 1 KB (local)   | 0 (registers) |
 * | K_tile        | 8 KB (local)   | 8 KB (local)  |
 * | V_tile        | 8 KB (local)   | 8 KB (local)  |
 * | KQ_tile       | 512 B (local)  | 0 (registers) |
 * | VKQ           | 2 KB (local)   | 0 (registers) |
 * | Reduction bufs| 1 KB (local)   | 0 (no reduce) |
 * | Total local   | ~20.5 KB       | ~16 KB        |
 * | Barriers/KV   | 8+             | 1             |
 *
 * Expected improvement: 30-50% prefill (from barrier elimination)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define BLOCK_SIZE_M 64   // Q rows per work-group (1 WI per Q row)
#define BLOCK_SIZE_N 32   // KV rows per tile
#define HEAD_DIM 128
#define VEC_SIZE 8         // half8 vectorization
#define VEC_LANES (HEAD_DIM / VEC_SIZE)  // 128/8 = 16

#define SOFTMAX_MIN -1e30f

__attribute__((work_group_size_hint(64, 1, 1)))
__kernel void flash_attention_prefill_fp16_adreno_v2(
    __global const half* restrict Q,    // [Batch, Num_Heads_Q, Seq_Q, Head_Dim]
    __global const half* restrict K,    // [Batch, Num_Heads_KV, Seq_KV, Head_Dim]
    __global const half* restrict V,    // [Batch, Num_Heads_KV, Seq_KV, Head_Dim]
    __global half* restrict Output,     // [Batch, Num_Heads_Q, Seq_Q, Head_Dim]
    const float scale,
    const int seq_len_q,
    const int seq_len_kv,
    const int num_heads_q,
    const int num_heads_kv,
    const int q_per_kv     // GQA ratio (num_heads_q / num_heads_kv)
) {
    // Indices
    const int head_id = get_global_id(1);      // Current Q head
    const int batch_id = get_global_id(2);
    const int m_block_idx = get_group_id(0);   // Which tile of Q we are processing
    const int tid = get_local_id(0);
    
    // GQA: Map Q head to KV head
    const int kv_head_id = head_id / q_per_kv;
    
    // Local memory for KV tiles (Shared across work-items in a work-group)
    // Using half8 for 8-byte alignment and dot() intrinsic support
    __local half8 local_k[BLOCK_SIZE_N][VEC_LANES];  // 32 x 16 x 16 = 8 KB
    __local half8 local_v[BLOCK_SIZE_N][VEC_LANES];  // 32 x 16 x 16 = 8 KB
    
    // Thread-local accumulators for Online Softmax (in registers, NOT local memory!)
    float row_m = SOFTMAX_MIN;  // Max score for this Q row
    float row_l = 0.0f;         // Sum of exps for this Q row
    
    // VKQ accumulator in FP32 for numerical precision
    // Using float4 for vectorized accumulation (4 x 16 = 64 floats = 256 B in registers)
    float4 acc[VEC_LANES * 2];  // 32 x float4 = 128 floats = 512 B in registers
    for (int i = 0; i < VEC_LANES * 2; i++) {
        acc[i] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    // Load Q into registers (each WI loads its own Q row)
    int q_local_idx = tid;
    int q_idx = m_block_idx * BLOCK_SIZE_M + q_local_idx;
    
    // Q in registers: half8[16] = 256 B per work-item
    half8 q_vec[VEC_LANES];
    
    if (q_idx < seq_len_q) {
        const int q_offset = ((batch_id * num_heads_q + head_id) * seq_len_q + q_idx) * HEAD_DIM;
        for (int d = 0; d < VEC_LANES; d++) {
            q_vec[d] = *((__global const half8*)(Q + q_offset + d * VEC_SIZE));
        }
    }
    
    // Outer Loop: Iterate over KV blocks (FlashAttention-v2 strategy)
    const int num_kv_tiles = (seq_len_kv + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    
    for (int n_tile = 0; n_tile < num_kv_tiles; n_tile++) {
        // 1. Load K and V tiles into Local Memory (cooperative load)
        // Only BLOCK_SIZE_N work-items participate in loading
        if (tid < BLOCK_SIZE_N) {
            int kv_idx = n_tile * BLOCK_SIZE_N + tid;
            if (kv_idx < seq_len_kv) {
                const int kv_offset = ((batch_id * num_heads_kv + kv_head_id) * seq_len_kv + kv_idx) * HEAD_DIM;
                for (int d = 0; d < VEC_LANES; d++) {
                    local_k[tid][d] = *((__global const half8*)(K + kv_offset + d * VEC_SIZE));
                    local_v[tid][d] = *((__global const half8*)(V + kv_offset + d * VEC_SIZE));
                }
            } else {
                // Zero-pad for boundary tiles
                for (int d = 0; d < VEC_LANES; d++) {
                    local_k[tid][d] = (half8)(0.0h);
                    local_v[tid][d] = (half8)(0.0h);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // ONLY barrier in the KV loop!
        
        // 2. Compute Attention for this tile (per-WI, NO barriers!)
        if (q_idx < seq_len_q) {
            const int nrows_kv = min(BLOCK_SIZE_N, seq_len_kv - n_tile * BLOCK_SIZE_N);
            
            for (int j = 0; j < nrows_kv; j++) {
                // Dot product QK^T using half8 dot() intrinsic
                // dot(half4, half4) computes 4-element dot product in one instruction on Adreno
                float score = 0.0f;
                for (int d = 0; d < VEC_LANES; d++) {
                    const half8 k_vec = local_k[j][d];
                    // dot() on half4 computes 4 multiplies + 3 adds in ONE instruction
                    score += (float)dot(q_vec[d].lo, k_vec.lo);  // dot4 on first 4 elements
                    score += (float)dot(q_vec[d].hi, k_vec.hi);  // dot4 on last 4 elements
                }
                score *= scale;
                
                // Online Softmax Update (thread-local, no barriers!)
                // This is the FlashAttention-v2 online softmax algorithm
                const float old_m = row_m;
                row_m = max(row_m, score);
                
                // Compute exp and scaling factor
                // Use native_exp for speed (5-10x faster than exp on Adreno)
                const float exp_score = native_exp(score - row_m);
                const float p_scale = native_exp(old_m - row_m);
                
                // Update running sum
                row_l = row_l * p_scale + exp_score;
                
                // Fused VKQ accumulation (thread-local, no barriers!)
                // acc = p_scale * acc + exp_score * V[j]
                for (int d = 0; d < VEC_LANES; d++) {
                    const half8 v_vec = local_v[j][d];
                    
                    // Convert half8 to float4 pairs for FP32 accumulation
                    const float4 v_lo = convert_float4(v_vec.lo);
                    const float4 v_hi = convert_float4(v_vec.hi);
                    
                    // FP32 vector FMA (critical for precision!)
                    acc[d * 2]     = acc[d * 2]     * p_scale + exp_score * v_lo;
                    acc[d * 2 + 1] = acc[d * 2 + 1] * p_scale + exp_score * v_hi;
                }
            }
        }
        // NO barrier here! Each WI proceeds independently
    }
    
    // 3. Finalize and Store
    if (q_idx < seq_len_q) {
        const float inv_l = 1.0f / row_l;
        const int out_offset = ((batch_id * num_heads_q + head_id) * seq_len_q + q_idx) * HEAD_DIM;
        
        // Write output using half8 stores for bandwidth
        for (int d = 0; d < VEC_LANES; d++) {
            // Convert FP32 accumulators back to half8
            const half4 out_lo = convert_half4(acc[d * 2]     * inv_l);
            const half4 out_hi = convert_half4(acc[d * 2 + 1] * inv_l);
            const half8 out_val = (half8)(out_lo.s0, out_lo.s1, out_lo.s2, out_lo.s3,
                                          out_hi.s0, out_hi.s1, out_hi.s2, out_hi.s3);
            *((__global half8*)(Output + out_offset + d * VEC_SIZE)) = out_val;
        }
    }
}