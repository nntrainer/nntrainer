/**
 * @file mha_cpu_ref.cpp
 * @brief CPU reference implementation for Multi-Head Attention operations
 * @date 2026-01-29
 * @author Samsung R&D Institute
 *
 * This file contains the CPU reference implementation for Multi-Head Attention
 * (MHA) operations, including Q*K^T computation, softmax with triangular
 * masking, and attention*value multiplication. These implementations are used
 * for testing and validation of CUDA implementations.
 */

#include "mha_cpu_ref.h"

namespace reference {

// Template function definition for computeTokenQK
template <typename KeyType>
void computeTokenQK(const float *query, const KeyType *key_cache, float *output,
                    int seq_len, int num_heads_kv, int head_dim,
                    int num_q_heads_per_kv, int tile_size, bool process_all) {

  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int r = 0; r < seq_len; ++r) {        // For each sequence step
    for (int n = 0; n < num_heads_kv; ++n) { // For each KV head
      for (int g = 0; g < num_q_heads_per_kv;
           ++g) { // For each Q head in the group

        float sum = 0.0f;

        for (int k = 0; k < head_dim; ++k) { // Dot product over head_dim
          // Query index: [n, g, k]
          float q_val =
            query[n * num_q_heads_per_kv * head_dim + g * head_dim + k];

          // Key cache index: [r, n, k]
          KeyType key_raw =
            key_cache[r * num_heads_kv * head_dim + n * head_dim + k];

          float key_val;
          if constexpr (std::is_same<KeyType, uint16_t>::value) {
            key_val = fp16_to_fp32(key_raw);
          } else {
            key_val = static_cast<float>(key_raw);
          }

          sum += q_val * key_val;
        }

        // Output index: [r, n, g]
        output[r * num_heads_kv * num_q_heads_per_kv + n * num_q_heads_per_kv +
               g] = sum * scale;
      }
    }
  }
}

// Explicit template instantiations
template void computeTokenQK<float>(const float *query, const float *key_cache,
                                    float *output, int seq_len,
                                    int num_heads_kv, int head_dim,
                                    int num_q_heads_per_kv, int tile_size,
                                    bool process_all);

template void computeTokenQK<uint16_t>(const float *query,
                                       const uint16_t *key_cache, float *output,
                                       int seq_len, int num_heads_kv,
                                       int head_dim, int num_q_heads_per_kv,
                                       int tile_size, bool process_all);

/**
 * @brief Helper function to compute Attention * Value for a single query token.
 *
 * Calculates the weighted sum of value vectors based on attention weights.
 * Iterates through all heads and the head dimension.
 *
 * @param token_idx The index of the current query token in the sequence
 * (0-based). Determines the range of key/value tokens to attend to (0 to
 * token_idx).
 * @param attn_weights_offset The starting offset in the attention_weights array
 * for this token's scores.
 * @param output_row_idx The row index in the output tensor where the result
 * will be stored. (e.g., 'i' for prefill, '0' for incremental).
 * @param attention_weights Pointer to the attention weights buffer.
 * @param value_cache Pointer to the value cache buffer (FP16).
 * @param output Pointer to the output buffer.
 * @param num_heads_kv Number of Key/Value heads.
 * @param num_q_heads_per_kv Number of Query heads per KV head (Grouped Query
 * Attention).
 * @param head_dim Dimension of each attention head.
 */
static void compute_single_token_attention_value(
  int token_idx, int attn_weights_offset, int output_row_idx,
  const float *attention_weights, const uint16_t *value_cache, float *output,
  int num_heads_kv, int num_q_heads_per_kv, int head_dim) {
  for (int n = 0; n < num_heads_kv; ++n) {
    for (int h = 0; h < num_q_heads_per_kv; ++h) {
      for (int d = 0; d < head_dim; ++d) {
        float sum = 0.0f;

        // Sum over all previous positions (triangular attention)
        for (int j = 0; j <= token_idx; ++j) {
          // Get attention weight
          int attn_idx = attn_weights_offset +
                         j * num_heads_kv * num_q_heads_per_kv +
                         n * num_q_heads_per_kv + h;
          float attn_weight = attention_weights[attn_idx];

          // Get value from cache
          int vcache_idx = j * num_heads_kv * head_dim + n * head_dim + d;
          float value = fp16_to_fp32(value_cache[vcache_idx]);

          sum += attn_weight * value;
        }

        // Output index
        int out_idx =
          ((output_row_idx * num_heads_kv + n) * num_q_heads_per_kv + h) *
            head_dim +
          d;
        output[out_idx] = sum;
      }
    }
  }
}

void compute_attention_value_mul_prefill(const float *attention_weights,
                                         const uint16_t *value_cache,
                                         float *output, int seq_len,
                                         int num_heads_kv,
                                         int num_q_heads_per_kv, int head_dim) {
  // Calculate triangular index for attention weights
  auto calc_attn_index = [](int i) -> int { return (i * (i + 1)) / 2; };

  for (int i = 0; i < seq_len; ++i) {
    // Calculate starting index in attention_weights for this sequence position
    int attn_row_start = calc_attn_index(i) * num_heads_kv * num_q_heads_per_kv;
    compute_single_token_attention_value(
      i, attn_row_start, i, attention_weights, value_cache, output,
      num_heads_kv, num_q_heads_per_kv, head_dim);
  }
}

void compute_attention_value_mul_incremental(
  const float *attention_weights, const uint16_t *value_cache, float *output,
  int seq_len, int num_heads_kv, int num_q_heads_per_kv, int head_dim) {
  int i = seq_len - 1;
  compute_single_token_attention_value(i, 0, 0, attention_weights, value_cache,
                                       output, num_heads_kv, num_q_heads_per_kv,
                                       head_dim);
}

void compute_attention_value_mul(const float *attention_weights,
                                 const uint16_t *value_cache, float *output,
                                 int seq_len, int num_heads_kv,
                                 int num_q_heads_per_kv, int head_dim,
                                 bool process_all) {
  if (process_all) {
    compute_attention_value_mul_prefill(attention_weights, value_cache, output,
                                        seq_len, num_heads_kv,
                                        num_q_heads_per_kv, head_dim);
  } else {
    compute_attention_value_mul_incremental(attention_weights, value_cache,
                                            output, seq_len, num_heads_kv,
                                            num_q_heads_per_kv, head_dim);
  }
}

void run_attention_sequence_prefill(const float *query,
                                    const uint16_t *key_cache,
                                    const uint16_t *value_cache, float *output,
                                    unsigned int to, int num_heads,
                                    int group_size, int head_dim) {
  int num_heads_kv = num_heads / group_size;
  size_t attn_len = (size_t)to * (to + 1) / 2;
  std::vector<float> attn_scores(attn_len * num_heads);

  compute_kcaches_prefill(query, key_cache, attn_scores.data(), to,
                          num_heads_kv, num_heads, group_size, head_dim);
  softmax_triangle_prefill(attn_scores.data(), to, num_heads);
  compute_attention_value_mul_prefill(attn_scores.data(), value_cache, output,
                                      to, num_heads_kv, group_size, head_dim);
}

void run_attention_sequence_incremental(const float *query,
                                        const uint16_t *key_cache,
                                        const uint16_t *value_cache,
                                        float *output, unsigned int from,
                                        int num_heads, int group_size,
                                        int head_dim) {
  int num_heads_kv = num_heads / group_size;
  size_t attn_len = from + 1;
  std::vector<float> attn_scores(attn_len * num_heads);

  compute_kcaches_incremental(query, key_cache, attn_scores.data(), from,
                              num_heads_kv, group_size, head_dim);
  softmax_triangle_incremental(attn_scores.data(), from, num_heads);
  // Note: compute_attention_value_mul_incremental takes 'seq_len' which
  // corresponds to 'from + 1' (total tokens processed so far) Inside it uses
  // 'seq_len - 1' to get the current token index, which will be 'from'.
  compute_attention_value_mul_incremental(attn_scores.data(), value_cache,
                                          output, from + 1, num_heads_kv,
                                          group_size, head_dim);
}

void run_attention_sequence(const float *query, const uint16_t *key_cache,
                            const uint16_t *value_cache, float *output,
                            unsigned int from, unsigned int to, int num_heads,
                            int group_size, int head_dim) {
  if (from == 0) {
    run_attention_sequence_prefill(query, key_cache, value_cache, output, to,
                                   num_heads, group_size, head_dim);
  } else {
    run_attention_sequence_incremental(query, key_cache, value_cache, output,
                                       from, num_heads, group_size, head_dim);
  }
}

void compute_kcaches_prefill(const float *query, const uint16_t *key_cache,
                             float *output, size_t sequence_len,
                             unsigned int num_heads_kv, unsigned int num_head,
                             unsigned int group_size, unsigned int head_dim) {
  // Prefill case: process all tokens in sequence with triangular attention
  for (int i = 0; i < sequence_len; ++i) {
    // Get query for this token: [num_head, head_dim]
    const float *query_ptr = query + i * num_head * head_dim;

    // Number of keys to attend to (triangular attention)
    int row_to_compute = i + 1;

    // Calculate output offset for this token (triangular packed format)
    size_t out_start_row = (i * (i + 1)) / 2;
    float *output_ptr = output + out_start_row * num_head;

    // Call single-token reference implementation
    computeTokenQK<uint16_t>(query_ptr, key_cache, output_ptr, row_to_compute,
                             num_heads_kv, head_dim, group_size, 16, true);
  }
}

void compute_kcaches_incremental(const float *query, const uint16_t *key_cache,
                                 float *output, unsigned int from,
                                 unsigned int num_heads_kv,
                                 unsigned int group_size,
                                 unsigned int head_dim) {
  // Incremental case: process single token at position 'from'
  // Compute attention scores for token at 'from' against all previous tokens
  // (0..from)
  const float *query_ptr = query; // Single token query
  int row_to_compute = from + 1;  // Attend to tokens 0..from (inclusive)

  computeTokenQK<uint16_t>(query_ptr, key_cache, output, row_to_compute,
                           num_heads_kv, head_dim, group_size, 16, true);
}

void compute_kcaches(const float *query, const uint16_t *key_cache,
                     float *output, unsigned int from, size_t sequence_len,
                     unsigned int num_head, unsigned int group_size,
                     unsigned int head_dim) {
  const int num_heads_kv = num_head / group_size;

  if (from) {
    compute_kcaches_incremental(query, key_cache, output, from, num_heads_kv,
                                group_size, head_dim);
  } else {
    compute_kcaches_prefill(query, key_cache, output, sequence_len,
                            num_heads_kv, num_head, group_size, head_dim);
  }
}

void softmax_triangle_prefill(float *attention_scores, int seq_len,
                              int num_heads) {
  // Triangular packed case (Prefill)
  for (int i = 0; i < seq_len; ++i) {
    // Calculate start index for this token's attention scores
    // For token i, we have i+1 scores (0..i)
    int start_row = (i * (i + 1)) / 2;
    int len = i + 1;

    for (int h = 0; h < num_heads; ++h) {
      // Find max for numerical stability
      float max_val = -INFINITY;
      for (int j = 0; j < len; ++j) {
        int idx = (start_row + j) * num_heads + h;
        if (attention_scores[idx] > max_val) {
          max_val = attention_scores[idx];
        }
      }

      // Compute exp and sum
      float sum = 0.0f;
      for (int j = 0; j < len; ++j) {
        int idx = (start_row + j) * num_heads + h;
        float val = std::exp(attention_scores[idx] - max_val);
        attention_scores[idx] = val; // Store exp temporarily
        sum += val;
      }

      // Normalize
      for (int j = 0; j < len; ++j) {
        int idx = (start_row + j) * num_heads + h;
        attention_scores[idx] /= sum;
      }
    }
  }
}

void softmax_triangle_incremental(float *attention_scores, unsigned int from,
                                  int num_heads) {
  // Incremental case: single row of attention scores
  // Token at position 'from' attends to all previous tokens 0..from
  int len = from + 1;

  for (int h = 0; h < num_heads; ++h) {
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int j = 0; j < len; ++j) {
      int idx = j * num_heads + h;
      if (attention_scores[idx] > max_val) {
        max_val = attention_scores[idx];
      }
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int j = 0; j < len; ++j) {
      int idx = j * num_heads + h;
      float val = std::exp(attention_scores[idx] - max_val);
      attention_scores[idx] = val;
      sum += val;
    }

    // Normalize
    for (int j = 0; j < len; ++j) {
      int idx = j * num_heads + h;
      attention_scores[idx] /= sum;
    }
  }
}

void softmax_triangle(float *attention_scores, int seq_len, int num_heads,
                      unsigned int from) {
  if (from == 0) {
    softmax_triangle_prefill(attention_scores, seq_len, num_heads);
  } else {
    softmax_triangle_incremental(attention_scores, from, num_heads);
  }
}

} // namespace reference
