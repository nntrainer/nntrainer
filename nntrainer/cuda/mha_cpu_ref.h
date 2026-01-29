/**
 * @file mha_cpu_ref.h
 * @brief CPU reference implementation for Multi-Head Attention operations
 * @date 2026-01-29
 * @author Samsung R&D Institute
 *
 * This file contains the CPU reference implementation for Multi-Head Attention
 * (MHA) operations, including Q*K^T computation, softmax with triangular
 * masking, and attention*value multiplication. These implementations are used
 * for testing and validation of CUDA implementations.
 */

#ifndef __REFERENCE_COMPUTE_H__
#define __REFERENCE_COMPUTE_H__

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace reference {

// Helper to convert FP16 (uint16_t) to FP32 (float)
// Simplified version for test reference
inline float fp16_to_fp32(uint16_t value) {
  uint32_t sign = (value & 0x8000) << 16;
  int32_t exponent = ((value >> 10) & 0x1F);
  uint32_t mantissa = (value & 0x3FF);

  if (exponent == 0) {
    if (mantissa == 0) {
      uint32_t result = sign;
      return *reinterpret_cast<float *>(&result);
    }
    // Denormalized number
    exponent = 1;
    while ((mantissa & 0x400) == 0) {
      mantissa <<= 1;
      exponent--;
    }
    mantissa &= 0x3FF;
  } else if (exponent == 31) {
    uint32_t result = sign | 0x7F800000 | (mantissa << 13);
    return *reinterpret_cast<float *>(&result);
  }

  exponent = exponent - 15 + 127;
  uint32_t result = sign | (exponent << 23) | (mantissa << 13);
  return *reinterpret_cast<float *>(&result);
}

/**
 * @brief Reference implementation for single-token Q * K^T computation
 *
 * Computes attention scores for ONE token's queries against the entire key
 * cache. This is used per-token: Query * Key_Cache^T, scaled by
 * 1/sqrt(head_dim)
 *
 * Note: This computes attention scores for a SINGLE query token (or step),
 * not the entire query matrix. The 'query' parameter contains queries for
 * one token across all heads.
 *
 * @param query Single-token query tensor [num_heads_kv * num_q_heads_per_kv *
 * head_dim] Shape: [num_heads_KV, num_q_heads_per_kv, head_dim] Contains
 * queries for ONE token across all heads
 * @param key_cache Key cache tensor [seq_len * num_heads_kv * head_dim]
 *                  Shape: [seq_len, num_heads_KV, head_dim]
 *                  Contains keys for ALL previous tokens
 * @param output Output attention scores [seq_len * num_heads_kv *
 * num_q_heads_per_kv] Shape: [seq_len, num_heads_KV, num_q_heads_per_kv]
 * @param seq_len Sequence length (number of key/value tokens to attend to)
 * @param num_heads_kv Number of Key/Value heads
 * @param head_dim Dimension of each attention head
 * @param num_q_heads_per_kv Number of Query heads per KV head (GQA group size)
 * @param tile_size Tile size for optimization (unused in reference
 * implementation)
 * @param process_all Whether to process all positions (unused in reference
 * implementation)
 */
template <typename KeyType>
void computeTokenQK(const float *query, const KeyType *key_cache, float *output,
                    int seq_len, int num_heads_kv, int head_dim,
                    int num_q_heads_per_kv, int tile_size, bool process_all);

/**
 * @brief Reference implementation for multi-token Q * K^T computation
 *
 * Computes Q * K^T for multiple tokens (entire sequence) with triangular
 * attention pattern. This matches the behavior of
 * CustomMHACoreV2Layer::compute_kcaches.
 *
 * @param query Query data pointer in FP32
 *              - Prefill (from=0): [sequence_len, num_head, head_dim]
 *              - Incremental (from>0): [1, num_head, head_dim] (single token)
 * @param key_cache Key cache data pointer [max_seq_len, num_heads_kv, head_dim]
 * in FP16
 * @param output Output attention scores (Q * K^T / sqrt(head_dim))
 *               - Prefill (from=0): [(sequence_len*(sequence_len+1)/2),
 * num_head] Packed triangular format: token i has i+1 scores
 *               - Incremental (from>0): [from+1, num_head]
 *                 Single row: token at 'from' attending to tokens 0..from
 * @param from Starting position (0 for prefill, >0 for incremental)
 * @param sequence_len Sequence length to process
 * @param num_head Total number of query heads
 * @param group_size Number of query heads per KV head (GQA group size)
 * @param head_dim Dimension of each attention head
 */
void compute_kcaches(const float *query, const uint16_t *key_cache,
                     float *output, unsigned int from, size_t sequence_len,
                     unsigned int num_head, unsigned int group_size,
                     unsigned int head_dim);

void compute_kcaches_prefill(const float *query, const uint16_t *key_cache,
                             float *output, size_t sequence_len,
                             unsigned int num_heads_kv, unsigned int num_head,
                             unsigned int group_size, unsigned int head_dim);
void compute_kcaches_incremental(const float *query, const uint16_t *key_cache,
                                 float *output, unsigned int from,
                                 unsigned int num_heads_kv,
                                 unsigned int group_size,
                                 unsigned int head_dim);

/**
 * @brief Reference implementation for softmax with triangular mask (Prefill
 * mode)
 *
 * Applies softmax to the packed triangular attention scores for prefill mode.
 * Each token i attends to tokens 0..i in a triangular pattern.
 *
 * @param attention_scores Input/Output attention scores
 * [(seq_len*(seq_len+1)/2), num_heads] Packed triangular format: token i has
 * i+1 scores
 * @param seq_len Sequence length
 * @param num_heads Number of heads
 */
void softmax_triangle_prefill(float *attention_scores, int seq_len,
                              int num_heads);

/**
 * @brief Reference implementation for softmax (Incremental mode)
 *
 * Applies softmax to a single row of attention scores for incremental mode.
 * Token at position 'from' attends to all previous tokens 0..from.
 *
 * @param attention_scores Input/Output attention scores [from+1, num_heads]
 *                         Single row: token 'from' attending to tokens 0..from
 * @param from Current token position
 * @param num_heads Number of heads
 */
void softmax_triangle_incremental(float *attention_scores, unsigned int from,
                                  int num_heads);

/**
 * @brief Reference implementation for softmax with triangular mask
 *
 * Applies softmax to the packed triangular attention scores.
 * Dispatches to prefill or incremental implementation based on 'from'
 * parameter.
 *
 * @param attention_scores Input/Output attention scores
 *                         - Prefill (from=0): [(seq_len*(seq_len+1)/2),
 * num_heads]
 *                         - Incremental (from>0): [from+1, num_heads]
 * @param seq_len Sequence length
 * @param num_heads Number of heads
 * @param from Start position (0 for prefill/triangular)
 */
void softmax_triangle(float *attention_scores, int seq_len, int num_heads,
                      unsigned int from);

void compute_attention_value_mul_prefill(const float *attention_weights,
                                         const uint16_t *value_cache,
                                         float *output, int seq_len,
                                         int num_heads_kv,
                                         int num_q_heads_per_kv, int head_dim);

void compute_attention_value_mul_incremental(
  const float *attention_weights, const uint16_t *value_cache, float *output,
  int seq_len, int num_heads_kv, int num_q_heads_per_kv, int head_dim);

/**
 * @brief Reference implementation for Attention * Value multiplication
 *
 * Computes: Attention_Weights * Value_Cache
 * This is the second matrix multiplication in attention: softmax(Q*K^T) * V
 *
 * @param attention_weights Attention weights after softmax [variable_size,
 * num_heads_kv, num_q_heads_per_kv] Size depends on triangular attention
 * pattern
 * @param value_cache Value cache in FP16 format [seq_len, num_heads_kv,
 * head_dim]
 * @param output Output tensor [seq_len, num_heads_kv, num_q_heads_per_kv,
 * head_dim]
 * @param seq_len Sequence length
 * @param num_heads_kv Number of Key/Value heads
 * @param num_q_heads_per_kv Number of Query heads per KV head (GQA group size)
 * @param head_dim Dimension of each attention head
 * @param process_all If true, process all sequence positions; if false, only
 * last position
 */
void compute_attention_value_mul(const float *attention_weights,
                                 const uint16_t *value_cache, float *output,
                                 int seq_len, int num_heads_kv,
                                 int num_q_heads_per_kv, int head_dim,
                                 bool process_all);

/**
 * @brief Reference implementation for full attention sequence
 *
 * Runs the full pipeline: Q*K^T -> Softmax -> A*V
 *
 * @param query Query tensor [q_len, num_heads, head_dim]
 * @param key_cache Key cache tensor [seq_len, num_heads_kv, head_dim] (FP16)
 * @param value_cache Value cache tensor [seq_len, num_heads_kv, head_dim]
 * (FP16)
 * @param output Output tensor [q_len, num_heads, head_dim]
 * @param from Start position (0 for prefill)
 * @param to End position (sequence length)
 * @param num_heads Total number of query heads
 * @param group_size GQA group size
 * @param head_dim Head dimension
 */
void run_attention_sequence(const float *query, const uint16_t *key_cache,
                            const uint16_t *value_cache, float *output,
                            unsigned int from, unsigned int to, int num_heads,
                            int group_size, int head_dim);

} // namespace reference

#endif // __REFERENCE_COMPUTE_H__
