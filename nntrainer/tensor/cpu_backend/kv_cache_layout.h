// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   kv_cache_layout.h
 * @date   02 November 2026
 * @brief  The KV cache row layouts and the int8 element the CPU
 *         kv_cache_write() and attention() ops share
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * @details Moved from MHACoreLayer (mha_core.cpp) as they were: the K-OHWI
 * scatter / gather of its NNTR_KV_OHWI cache layout and the symmetric
 * per-(row, head) int8 quantization of its NNTR_KV_INT8 cache. Internal to
 * cpu_backend; not installed.
 */

#ifndef __KV_CACHE_LAYOUT_H__
#define __KV_CACHE_LAYOUT_H__

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <fp16.h>

namespace nntrainer {

/**
 * @brief Scatter a step's K (concat layout [step_size, num_heads_kv *
 *        head_dim]) into the OHWI cache buffer at positions
 *        [cache_pos, cache_pos + step_size).
 * @param cache_base the start of element [0,0,0,0] of the cache (the raw
 *        Tensor data pointer); the cache's per-batch element count is
 *        max_seq_len * num_heads_kv * head_dim
 */
static inline void scatter_k_concat_to_ohwi_fp16(
  const uint16_t *src,  // [step_size, num_heads_kv * head_dim]
  uint16_t *cache_base, // raw cache buffer
  unsigned int batch, unsigned int cache_pos, unsigned int step_size,
  unsigned int num_heads_kv, unsigned int head_dim, unsigned int max_seq_len) {
  const size_t HD = (size_t)num_heads_kv * head_dim;
  const size_t per_head = (size_t)max_seq_len * head_dim;
  const size_t batch_off = (size_t)batch * num_heads_kv * per_head;
  for (unsigned int h = 0; h < num_heads_kv; ++h) {
    uint16_t *dst_head = cache_base + batch_off + (size_t)h * per_head;
    const uint16_t *src_head = src + (size_t)h * head_dim;
    for (unsigned int t = 0; t < step_size; ++t) {
      std::memcpy(dst_head + (size_t)(cache_pos + t) * head_dim,
                  src_head + (size_t)t * HD,
                  (size_t)head_dim * sizeof(uint16_t));
    }
  }
}

/**
 * @brief Gather an OHWI K cache slice [0, cache_to) into concat layout
 *        [cache_to, num_heads_kv * head_dim].
 * @param dst a buffer of cache_to * num_heads_kv * head_dim elements
 */
static inline void gather_k_ohwi_to_concat_fp16(
  const uint16_t *cache_base, // raw cache buffer
  uint16_t *dst,              // [cache_to, num_heads_kv * head_dim]
  unsigned int batch, unsigned int cache_to, unsigned int num_heads_kv,
  unsigned int head_dim, unsigned int max_seq_len) {
  const size_t HD = (size_t)num_heads_kv * head_dim;
  const size_t per_head = (size_t)max_seq_len * head_dim;
  const size_t batch_off = (size_t)batch * num_heads_kv * per_head;
  for (unsigned int h = 0; h < num_heads_kv; ++h) {
    const uint16_t *src_head = cache_base + batch_off + (size_t)h * per_head;
    uint16_t *dst_head = dst + (size_t)h * head_dim;
    for (unsigned int t = 0; t < cache_to; ++t) {
      std::memcpy(dst_head + (size_t)t * HD, src_head + (size_t)t * head_dim,
                  (size_t)head_dim * sizeof(uint16_t));
    }
  }
}

/**
 * @brief Quantize a [step_size, num_heads, head_dim] fp16 source (row-major,
 *        row stride num_heads * head_dim) into an int8 destination plus a
 *        per-(token, head) fp16 scale. Scale = amax / 127, dequantized as
 *        int8_value * scale. Symmetric (no zero point), so the matmul stays
 *        an int8 dot product plus an fp16 scale-out multiply.
 */
static inline void quantize_kv_fp16_to_int8_per_row(
  const uint16_t *src_fp16, // [step_size, num_heads * head_dim]
  int8_t *dst_int8,         // [step_size, num_heads * head_dim]
  uint16_t *dst_scale_fp16, // [step_size, num_heads]
  unsigned int step_size, unsigned int num_heads, unsigned int head_dim) {
  for (unsigned int s = 0; s < step_size; ++s) {
    for (unsigned int h = 0; h < num_heads; ++h) {
      const uint16_t *row =
        src_fp16 + (size_t)s * num_heads * head_dim + (size_t)h * head_dim;
      float amax = 0.0f;
      for (unsigned int d = 0; d < head_dim; ++d) {
        float v = std::fabs(nntrainer::compute_fp16_to_fp32(row[d]));
        if (v > amax)
          amax = v;
      }
      const float scale = amax / 127.0f;
      const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
      dst_scale_fp16[s * num_heads + h] =
        nntrainer::compute_fp32_to_fp16(scale);
      int8_t *out =
        dst_int8 + (size_t)s * num_heads * head_dim + (size_t)h * head_dim;
      for (unsigned int d = 0; d < head_dim; ++d) {
        int q = (int)std::round(nntrainer::compute_fp16_to_fp32(row[d]) * inv);
        if (q < -127)
          q = -127;
        if (q > 127)
          q = 127;
        out[d] = (int8_t)q;
      }
    }
  }
}

} // namespace nntrainer

#endif /* __KV_CACHE_LAYOUT_H__ */
