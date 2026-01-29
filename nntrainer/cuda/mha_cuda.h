/**
 * @file mha_cuda.h
 * @brief CUDA implementation for Multi-Head Attention operations
 * @date 2026-01-29
 * @author Samsung R&D Institute
 *
 * This file contains the CUDA implementation for Multi-Head Attention (MHA)
 * operations, including Q*K^T computation, softmax with triangular masking, and
 * attention*value multiplication. These implementations are optimized for GPU
 * execution.
 */

#ifndef __CUSTOM_MHA_CORE_V2_CUDA_H__
#define __CUSTOM_MHA_CORE_V2_CUDA_H__

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace custom {

void compute_kcaches_prefill_cuda(
  const float *query, const uint16_t *key_cache, float *output,
  unsigned int sequence_len, unsigned int num_heads_kv, unsigned int num_head,
  unsigned int group_size, unsigned int head_dim, cudaStream_t stream = 0);

void softmax_triangle_prefill_cuda(float *attention_scores, int seq_len,
                                   int num_heads, cudaStream_t stream = 0);

void compute_attention_value_mul_prefill_cuda(
  const float *attention_weights, const uint16_t *value_cache, float *output,
  int seq_len, int num_heads_kv, int num_q_heads_per_kv, int head_dim,
  cudaStream_t stream = 0);

void run_attention_sequence_prefill_cuda(
  const float *query, const uint16_t *key_cache, const uint16_t *value_cache,
  float *output, unsigned int seq_len, unsigned int num_heads,
  unsigned int group_size, unsigned int head_dim, float *d_attn_scores,
  cudaStream_t stream = 0);

} // namespace custom

#endif // __CUSTOM_MHA_CORE_V2_CUDA_H__
