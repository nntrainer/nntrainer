// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_attention.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA flash-style attention core for the gemma4 mha (FP32-safe).
 *
 * O[h,i,:] = sum_j softmax_j( softcap * tanh( (Q[h,i]·K[hkv,j]) /
 * (softcap*sqrt(d)) ) ) * V[hkv,j] with a causal (j<=i_abs) + sliding-window (j
 * > i_abs - window) mask and GQA (hkv = h / gqa). One block per (query head h,
 * query row i); online (flash) softmax in FP32. Q/K/V/O are per-head contiguous
 * [head, rows, head_dim]. softcap<=0 disables the tanh soft-cap. This is the
 * O(M^2) compute that dominates prefill on the host path; RoPE / KV-cache fill
 * stay outside.
 */

#ifndef __CUDA_ATTENTION_H__
#define __CUDA_ATTENTION_H__

namespace nntrainer::cuda {

/**
 * @param Q         [num_heads, q_rows, head_dim] FP32 (device-accessible)
 * @param K         [num_kv_heads, kv_len, head_dim] FP32
 * @param V         [num_kv_heads, kv_len, head_dim] FP32
 * @param O         [num_heads, q_rows, head_dim] FP32 output
 * @param num_heads query heads
 * @param num_kv_heads kv heads (GQA: gqa = num_heads/num_kv_heads)
 * @param q_rows    number of query rows (M)
 * @param kv_len    number of cached keys/values (S); query row i has absolute
 *                  position q_pos0 + i and attends keys [0, q_pos0+i]
 * @param q_pos0    absolute position of query row 0 (cache offset)
 * @param head_dim  per-head dim (256 sliding / 512 full)
 * @param window    sliding-window size (use a huge value for full attention)
 * @param softcap   attn logit soft-cap (30.0 for gemma4; <=0 disables)
 * @return true on success
 */
bool cuda_attention_core_fp32(const float *Q, const float *K, const float *V,
                              float *O, int num_heads, int num_kv_heads,
                              int q_rows, int kv_len, int q_pos0, int head_dim,
                              int window, float softcap);

/**
 * @brief Drop-in GPU replacement for the host gemm_attention: reads the
 *        head-interleaved fp16 query step [N_q, num_heads_Q*head_dim] and the
 *        fp16 KV cache [N_kv, num_heads_KV*head_dim] directly (de-interleave +
 *        fp16->fp32 happen inside the kernel), runs the flash core in FP32, and
 *        writes the interleaved fp16 output [N_q, num_heads_Q*head_dim].
 * Matches gemm_attention's math: scale 1/sqrt(d), causal + sliding mask with
 * the cache_from offset, GQA; softcap<=0 (gemm_attention applies none).
 *
 * @param q_fp16   query step, fp16 bits (device-accessible)
 * @param k_fp16   KV cache key, fp16 bits
 * @param v_fp16   KV cache value, fp16 bits
 * @param o_fp16   output, fp16 bits (written)
 * @param num_heads_Q / num_heads_KV  GQA heads
 * @param N_q / N_kv  query rows / cached keys
 * @param cache_from  absolute position of query row 0
 * @param head_dim, window, softcap
 */
bool cuda_attention_interleaved_fp16(const unsigned short *q_fp16,
                                     const unsigned short *k_fp16,
                                     const unsigned short *v_fp16,
                                     unsigned short *o_fp16, int num_heads_Q,
                                     int num_heads_KV, int N_q, int N_kv,
                                     int cache_from, int head_dim, int window,
                                     float softcap);

/**
 * @brief Pre-grow the split-KV decode scratch buffers to the model's max decode
 *        capacity at load, so the M=1 flash-decode path never cudaMalloc/Frees
 *        during a CUDA-graph stream capture (which would invalidate the graph).
 * @param max_seq_len max KV length (=> max chunks at NNTR_CUDA_FLASH_DECODE
 * size)
 * @param max_hq      max number of query heads across attention layers
 * @param max_head_dim max head_dim across layers (gemma4: global 512 vs 256)
 */
bool cuda_attention_splitkv_prewarm(int max_seq_len, int max_hq,
                                    int max_head_dim);

} // namespace nntrainer::cuda

#endif // __CUDA_ATTENTION_H__
