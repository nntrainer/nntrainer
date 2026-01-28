// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   rope_reference.cpp
 * @date   27 January 2025
 * @brief  Reference implementation for RoPE (Rotary Positional Embedding)
 * @see    https://github.com/nnstreamer/nntrainer
 * @author [Your Name] <[Your Email]>
 * @bug    No known bugs except for NYI items
 *
 */

#include "rope_reference.h"
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>
#include <tensor_dim.h>

#ifdef _WIN32
#define COMPUTE_FP16_TO_FP32(x)                                                \
  _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define COMPUTE_FP32_TO_FP16(x)                                                \
  _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#else
#define COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
#endif

namespace nntrainer {

void rotary_embedding_avx2_ref(void *output, unsigned int width,
                               unsigned int dim, unsigned int half_,
                               float *inout, const float *cos_,
                               const float *sin_, bool only_convert_to_fp16) {

  using OutputType = ml::train::TensorDim::DataType;
  OutputType out_type = OutputType::FP32;

#ifdef __AVX2__
  if (output != nullptr)
    out_type = OutputType::UINT16;

  for (unsigned int w = 0; w < width; w += dim) {
    unsigned int k = 0;

    for (; k + 7 < half_; k += 8) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;

      __m256 a = _mm256_loadu_ps(&inout[i0]);
      __m256 b = _mm256_loadu_ps(&inout[i1]);

      if (only_convert_to_fp16) {
        if (out_type == OutputType::UINT16) {
          __m128i a_fp16 = _mm256_cvtps_ph(a, 0);
          __m128i b_fp16 = _mm256_cvtps_ph(b, 0);

          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i0),
            a_fp16);
          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i1),
            b_fp16);
        }

      } else {
        __m256 cos_v = _mm256_loadu_ps(&cos_[k]);
        __m256 sin_v = _mm256_loadu_ps(&sin_[k]);

        __m256 out0 =
          _mm256_sub_ps(_mm256_mul_ps(a, cos_v), _mm256_mul_ps(b, sin_v));
        __m256 out1 =
          _mm256_add_ps(_mm256_mul_ps(a, sin_v), _mm256_mul_ps(b, cos_v));

        if (out_type == OutputType::UINT16) {
          __m128i out0_fp16 = _mm256_cvtps_ph(out0, 0);
          __m128i out1_fp16 = _mm256_cvtps_ph(out1, 0);

          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i0),
            out0_fp16);
          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i1),
            out1_fp16);

        } else if (out_type == OutputType::FP32) {
          _mm256_storeu_ps(&inout[i0], out0);
          _mm256_storeu_ps(&inout[i1], out1);
        }
      }
    }

    for (; k < half_; ++k) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;
      // assert(i1 < width && "Scalar i1 overflow!");
      float a = inout[i0];
      float b = inout[i1];

      if (only_convert_to_fp16) {
        static_cast<uint16_t *>(output)[i0] = COMPUTE_FP32_TO_FP16(a);
        static_cast<uint16_t *>(output)[i1] = COMPUTE_FP32_TO_FP16(b);

      } else {

        float c = cos_[k];
        float s = sin_[k];

        float out0 = a * c - b * s;
        float out1 = a * s + b * c;

        if (out_type == OutputType::UINT16) {
          static_cast<uint16_t *>(output)[i0] = COMPUTE_FP32_TO_FP16(out0);
          static_cast<uint16_t *>(output)[i1] = COMPUTE_FP32_TO_FP16(out1);
        } else if (out_type == OutputType::FP32) {
          inout[i0] = out0;
          inout[i1] = out1;
        }
      }
    }
  }
#else
  throw std::runtime_error(
    "RoPE reference implementation requires AVX2 support.");
#endif
}

} // namespace nntrainer

namespace nntrainer {

void rotary_embedding_ref(void *output, unsigned int width, unsigned int dim,
                          unsigned int half_, float *inout, const float *cos_,
                          const float *sin_, bool only_convert_to_fp16) {

  using OutputType = ml::train::TensorDim::DataType;
  OutputType out_type = OutputType::FP32;

  if (output != nullptr)
    out_type = OutputType::UINT16;

  for (unsigned int w = 0; w < width; w += dim) {
    for (unsigned int k = 0; k < half_; ++k) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;

      float a = inout[i0];
      float b = inout[i1];

      if (only_convert_to_fp16) {
        static_cast<uint16_t *>(output)[i0] = COMPUTE_FP32_TO_FP16(a);
        static_cast<uint16_t *>(output)[i1] = COMPUTE_FP32_TO_FP16(b);

      } else {

        float c = cos_[k];
        float s = sin_[k];

        float out0 = a * c - b * s;
        float out1 = a * s + b * c;

        if (out_type == OutputType::UINT16) {
          static_cast<uint16_t *>(output)[i0] = COMPUTE_FP32_TO_FP16(out0);
          static_cast<uint16_t *>(output)[i1] = COMPUTE_FP32_TO_FP16(out1);
        } else if (out_type == OutputType::FP32) {
          inout[i0] = out0;
          inout[i1] = out1;
        }
      }
    }
  }
}

} // namespace nntrainer
