/**
 * @file mha_cuda.cu
 * @brief CUDA implementation for Multi-Head Attention operations
 * @date 2026-01-29
 * @author Samsung R&D Institute
 *
 * This file contains the CUDA implementation for Multi-Head Attention (MHA)
 * operations, including Q*K^T computation, softmax with triangular masking, and
 * attention*value multiplication. These implementations are optimized for GPU
 * execution.
 */

#include "mha_cuda.h"
#include <cmath>
#include <cstdio>

namespace custom {

__global__ void compute_kcaches_prefill_kernel(const float *query,
                                               const __half *key_cache,
                                               float *output, int seq_len,
                                               int num_heads, int num_heads_kv,
                                               int group_size, int head_dim) {

  // Map linear index to triangular position (i, j, h)
  // Total threads: attn_len * num_heads, where attn_len = seq_len * (seq_len +
  // 1) / 2
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t attn_len = (size_t)seq_len * (seq_len + 1) / 2;
  size_t total_threads = attn_len * num_heads;

  if (idx >= total_threads)
    return;

  // Extract head index
  int h = idx % num_heads;
  size_t flat_pos = idx / num_heads; // Position in triangular matrix

  // Convert flat triangular position to (i, j)
  // For triangular indexing: flat_pos = i*(i+1)/2 + j
  // Solve for i: i = floor((-1 + sqrt(1 + 8*flat_pos)) / 2)
  int i = (int)((sqrtf(1.0f + 8.0f * flat_pos) - 1.0f) / 2.0f);
  size_t row_start = (size_t)i * (i + 1) / 2;
  int j = flat_pos - row_start;

  int h_kv = h / group_size;

  // Q ptr: [i, h, :] -> i * num_heads * head_dim + h * head_dim
  const float *q_ptr = query + (i * num_heads + h) * head_dim;

  // K ptr: [j, h_kv, :] -> j * num_heads_kv * head_dim + h_kv * head_dim
  const __half *k_ptr = key_cache + (j * num_heads_kv + h_kv) * head_dim;

  float sum = 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    float q = q_ptr[d];
    float k = __half2float(k_ptr[d]);
    sum += q * k;
  }

  float scale = 1.0f / sqrtf((float)head_dim);
  sum *= scale;

  // Output index (packed triangular)
  size_t out_idx = flat_pos * num_heads + h;

  output[out_idx] = sum;
}

void compute_kcaches_prefill_cuda(const float *query, const uint16_t *key_cache,
                                  float *output, unsigned int sequence_len,
                                  unsigned int num_heads_kv,
                                  unsigned int num_head,
                                  unsigned int group_size,
                                  unsigned int head_dim, cudaStream_t stream) {

  // Only launch threads for valid triangular elements
  size_t attn_len = (size_t)sequence_len * (sequence_len + 1) / 2;
  size_t total_threads = attn_len * num_head;
  int blockSize = 256;
  int gridDim = (total_threads + blockSize - 1) / blockSize;

  compute_kcaches_prefill_kernel<<<gridDim, blockSize, 0, stream>>>(
    query, reinterpret_cast<const __half *>(key_cache), output, sequence_len,
    num_head, num_heads_kv, group_size, head_dim);
}

__global__ void softmax_triangle_prefill_kernel(float *attention_scores,
                                                int seq_len, int num_heads) {

  // One thread per (i, h) pair
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= seq_len * num_heads)
    return;

  int h = idx % num_heads;
  int i = idx / num_heads;

  // Row i, head h. Length is i + 1.
  // Start index for this row in packed buffer
  size_t out_start_row = (size_t)i * (i + 1) / 2;
  float *row_ptr = attention_scores + out_start_row * num_heads + h;
  int stride = num_heads; // Elements are spaced by num_heads

  // 1. Max
  float max_val = -INFINITY;
  for (int j = 0; j <= i; ++j) {
    float val = row_ptr[j * stride];
    if (val > max_val)
      max_val = val;
  }

  // 2. Exp & Sum
  float sum = 0.0f;
  for (int j = 0; j <= i; ++j) {
    float val = exp(row_ptr[j * stride] - max_val);
    row_ptr[j * stride] = val;
    sum += val;
  }

  // 3. Normalize
  for (int j = 0; j <= i; ++j) {
    row_ptr[j * stride] /= sum;
  }
}

void softmax_triangle_prefill_cuda(float *attention_scores, int seq_len,
                                   int num_heads, cudaStream_t stream) {

  int total_threads = seq_len * num_heads;
  int blockSize = 512;
  int gridDim = (total_threads + blockSize - 1) / blockSize;

  softmax_triangle_prefill_kernel<<<gridDim, blockSize, 0, stream>>>(
    attention_scores, seq_len, num_heads);
}

__global__ void compute_attention_value_mul_prefill_kernel(
  const float *attention_weights, const __half *value_cache, float *output,
  int seq_len, int num_heads, int num_heads_kv, int group_size, int head_dim) {

  // Output: [i, h, d]
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = (size_t)seq_len * num_heads * head_dim;
  if (idx >= total_elements)
    return;

  int d = idx % head_dim;
  size_t tmp = idx / head_dim;
  int h = tmp % num_heads;
  int i = tmp / num_heads;

  int h_kv = h / group_size;

  float sum = 0.0f;

  // Weighted sum over j (0..i)
  size_t attn_start_row = (size_t)i * (i + 1) / 2;
  const float *attn_ptr = attention_weights + attn_start_row * num_heads + h;
  int attn_stride = num_heads;

  // Value cache: [j, h_kv, d]
  // Stride for j: num_heads_kv * head_dim
  const __half *v_ptr_base = value_cache + h_kv * head_dim + d;
  int v_stride = num_heads_kv * head_dim;

  for (int j = 0; j <= i; ++j) {
    float weight = attn_ptr[j * attn_stride];
    float val = __half2float(v_ptr_base[j * v_stride]);
    sum += weight * val;
  }

  output[idx] = sum;
}

void compute_attention_value_mul_prefill_cuda(
  const float *attention_weights, const uint16_t *value_cache, float *output,
  int seq_len, int num_heads_kv, int num_q_heads_per_kv, int head_dim,
  cudaStream_t stream) {

  int num_heads = num_heads_kv * num_q_heads_per_kv;
  size_t total_elements = (size_t)seq_len * num_heads * head_dim;
  int blockSize = 512;
  int gridDim = (total_elements + blockSize - 1) / blockSize;

  compute_attention_value_mul_prefill_kernel<<<gridDim, blockSize, 0, stream>>>(
    attention_weights, reinterpret_cast<const __half *>(value_cache), output,
    seq_len, num_heads, num_heads_kv, num_q_heads_per_kv, head_dim);
}

/**
 * @brief Run the full attention sequence (Q*K^T -> Softmax -> A*V) for prefill
 * mode on CUDA.
 *
 * This function orchestrates the complete attention computation pipeline:
 * 1. Compute Q * K^T (attention scores)
 * 2. Apply softmax with triangular masking
 * 3. Compute attention-weighted sum of values (A * V)
 *
 * @param query Query tensor on device [seq_len, num_heads, head_dim]
 * @param key_cache Key cache tensor on device [seq_len, num_heads_kv, head_dim]
 * (FP16)
 * @param value_cache Value cache tensor on device [seq_len, num_heads_kv,
 * head_dim] (FP16)
 * @param output Output tensor on device [seq_len, num_heads, head_dim]
 * @param seq_len Sequence length
 * @param num_heads Total number of query heads
 * @param group_size Number of query heads per KV head (for Grouped Query
 * Attention)
 * @param head_dim Dimension of each attention head
 * @param d_attn_scores Pre-allocated device buffer for attention scores
 * [attn_len * num_heads] where attn_len = seq_len * (seq_len + 1) / 2
 * (triangular packed)
 * @param stream CUDA stream for asynchronous execution (default: 0)
 */
void run_attention_sequence_prefill_cuda(
  const float *query, const uint16_t *key_cache, const uint16_t *value_cache,
  float *output, unsigned int seq_len, unsigned int num_heads,
  unsigned int group_size, unsigned int head_dim, float *d_attn_scores,
  cudaStream_t stream) {

  int num_heads_kv = num_heads / group_size;
  int num_q_heads_per_kv = group_size;

  // 1. Q * K^T
  compute_kcaches_prefill_cuda(query, key_cache, d_attn_scores, seq_len,
                               num_heads_kv, num_heads, group_size, head_dim,
                               stream);

  // 2. Softmax
  softmax_triangle_prefill_cuda(d_attn_scores, seq_len, num_heads, stream);

  // 3. A * V
  compute_attention_value_mul_prefill_cuda(
    d_attn_scores, value_cache, output, seq_len, num_heads_kv,
    num_q_heads_per_kv, head_dim, stream);
}

} // namespace custom
