// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   turboquant_utils.h
 * @date   28 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  TurboQuant KV cache compression utilities.
 *
 *         Norm normalization + Hadamard rotation + Lloyd-Max codebook
 *         (paper Algorithm 1: MSE-optimal, 4-bit per coordinate)
 *
 * Packing layout (per byte):
 *   Lower nibble (bits 0-3): element[2i]   → [data(4)]
 *   Upper nibble (bits 4-7): element[2i+1] → [data(4)]
 */

#ifndef __TURBOQUANT_UTILS_H__
#define __TURBOQUANT_UTILS_H__
#ifdef __cplusplus

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nntrainer {

/***********************************************************************
 * Hadamard transform and rotation utilities
 ***********************************************************************/

/**
 * @brief In-place Walsh-Hadamard Transform (WHT).
 *        Normalizes by 1/sqrt(n) so the transform is orthogonal.
 *        Fast Walsh-Hadmard Transform (FWHT)
 */
inline void hadamard_transform(float *x, int n) {
  for (int len = 1; len < n; len <<= 1) {
    for (int i = 0; i < n; i += len << 1) {
      for (int j = 0; j < len; ++j) {
        float u = x[i + j];
        float v = x[i + j + len];
        x[i + j] = u + v;
        x[i + j + len] = u - v;
      }
    }
  }
  float inv_sqrt_n = 1.0f / std::sqrt((float)n);
  for (int i = 0; i < n; ++i)
    x[i] *= inv_sqrt_n;
}

/**
 * @brief Generate deterministic random sign vector (±1) via LCG.
 * */
inline void generate_random_signs(float *signs, int n,
                                  uint32_t seed = 0x5EED1234) {
  uint32_t state = seed;
  for (int i = 0; i < n; ++i) {
    state = state * 1664525u + 1013904223u; // LCG
    signs[i] = (state & 0x80000000u) ? -1.0f : 1.0f;
  }
}

/**
 * @brief Forward rotation: R = (1/sqrt(n)) * H * D
 *        Forward: R∙x  = (1/sqrt(n))∙H∙(D∙x)
 * */
inline void apply_rotation(const float *input, float *output,
                           const float *signs, int n) {
  for (int i = 0; i < n; ++i)
    output[i] = input[i] * signs[i];
  hadamard_transform(output, n);
}

/**
 * @brief Inverse rotation: R^T = D * (1/sqrt(n)) * H
 * */
inline void apply_inverse_rotation(float *data, const float *signs, int n) {
  hadamard_transform(data, n);
  for (int i = 0; i < n; ++i)
    data[i] *= signs[i];
}

/***********************************************************************
 * Lloyd-Max optimal codebooks (precomputed for Beta distribution)
 ***********************************************************************/

struct LloydMaxCodebook {
  float centroids[16];
  float boundaries[15];
};

/** d=64, 4-bit (16 levels), Beta(31.5, 31.5) on [-1,1] */
static constexpr LloydMaxCodebook CODEBOOK_D64 = {
  {-0.33079493f, -0.25291211f, -0.19885445f, -0.15492391f, -0.11648534f,
   -0.08131068f, -0.04808909f, -0.01591879f, 0.01591879f, 0.04808909f,
   0.08131068f, 0.11648534f, 0.15492391f, 0.19885445f, 0.25291211f,
   0.33079493f},
  {-0.29185352f, -0.22588328f, -0.17688918f, -0.13570463f, -0.09889801f,
   -0.06469988f, -0.03200394f, 0.0f, 0.03200394f, 0.06469988f, 0.09889801f,
   0.13570463f, 0.17688918f, 0.22588328f, 0.29185352f}};

/** d=128, 4-bit (16 levels), Beta(63.5, 63.5) on [-1,1] */
static constexpr LloydMaxCodebook CODEBOOK_D128 = {
  {-0.23766275f, -0.18083472f, -0.14180392f, -0.11028715f, -0.08282740f,
   -0.05777148f, -0.03415105f, -0.01130232f, 0.01130232f, 0.03415105f,
   0.05777148f, 0.08282740f, 0.11028715f, 0.14180392f, 0.18083472f,
   0.23766275f},
  {-0.20924874f, -0.16131932f, -0.12604554f, -0.09655728f, -0.07029944f,
   -0.04596127f, -0.02272669f, 0.0f, 0.02272669f, 0.04596127f, 0.07029944f,
   0.09655728f, 0.12604554f, 0.16131932f, 0.20924874f}};

/**
 * @brief  Return the Lloyd-Max codebook for the requested head dimension.
 *         Only the precomputed codebooks (head_dim == 64 or 128) are valid;
 *         any other head_dim is explicitly rejected to avoid silently using a
 *         mismatched quantizer that would degrade attention quality.
 */
inline const LloydMaxCodebook &get_codebook(int head_dim) {
  if (head_dim == 64)
    return CODEBOOK_D64;
  if (head_dim == 128)
    return CODEBOOK_D128;
  throw std::invalid_argument("TurboQuant get_codebook: unsupported head_dim=" +
                              std::to_string(head_dim) +
                              " (only 64 and 128 are supported)");
}

/** Lloyd-Max quantize: boundary search for optimal bin index (4-bit). */
inline uint8_t lloydmax_quantize(float val, const LloydMaxCodebook &cb) {
  int idx = 0;
  for (int i = 0; i < 15; ++i) {
    if (val > cb.boundaries[i])
      idx = i + 1;
  }
  return (uint8_t)idx;
}

/***********************************************************************
 * Norm + Rotation + Lloyd-Max (paper Algorithm 1)
 ***********************************************************************/

/**
 * @brief Full TurboQuant quantize pipeline (paper Algorithm 1):
 *        1. Compute norm, normalize to unit vector
 *        2. Apply Hadamard rotation
 *        3. Lloyd-Max quantize each coordinate (4-bit, 16 levels)
 *        4. Pack into nibbles (2 elements per byte)
 */
inline void turboquant_quantize_head(const float *input, int head_dim,
                                     uint8_t *out_packed, float *out_norm,
                                     const float *rot_signs,
                                     const LloydMaxCodebook &cb) {
  // Step 1: caculcate norm and save
  float norm_sq = 0.0f;
  for (int i = 0; i < head_dim; ++i)
    norm_sq += input[i] * input[i];
  float norm = std::sqrt(norm_sq);
  *out_norm = norm;

  // Step 2: normalize to the unit vector + random sign + Hadamard
  std::vector<float> rotated(head_dim);
  float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
  for (int i = 0; i < head_dim; ++i)
    rotated[i] = input[i] * inv_norm * rot_signs[i];
  hadamard_transform(rotated.data(), head_dim);

  // Step 3: 4-bit quantize each coordinate (lloydmax-quantize)
  // Step 4: packing to 1 byte with 2 elements
  for (int d = 0; d < head_dim; d += 2) {
    uint8_t q0 = lloydmax_quantize(rotated[d], cb);
    uint8_t q1 = 8; // midpoint default (near zero centroid)
    if (d + 1 < head_dim)
      q1 = lloydmax_quantize(rotated[d + 1], cb);

    out_packed[d / 2] = (q1 << 4) | q0;
  }
}

/**
 * @brief Dequantize one head: centroid lookup + inverse rotation + norm
 * rescale.
 */
inline void turboquant_dequantize_head(const uint8_t *packed, float norm,
                                       int head_dim, float *output,
                                       const float *rot_signs,
                                       const LloydMaxCodebook &cb) {
  // Step 1: Unpack + centroid lookup
  for (int d = 0; d < head_dim; d += 2) {
    uint8_t byte = packed[d / 2];
    uint8_t q0 = byte & 0x0F;
    uint8_t q1 = (byte >> 4) & 0x0F;
    output[d] = cb.centroids[q0];
    if (d + 1 < head_dim)
      output[d + 1] = cb.centroids[q1];
  }
  // Step 2: Inverse rotation
  hadamard_transform(output, head_dim);
  for (int i = 0; i < head_dim; ++i)
    output[i] *= rot_signs[i] * norm;
}

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TURBOQUANT_UTILS_H__ */
