// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   x86_qsi4cxp.cpp
 * @brief  AVX2 GEMV/GEMM kernel for channel-wise int4 (qsi4cxp kxn)
 *         weights with fp32 activation and fp32 output. Reads the
 *         Int4QTensor canonical layout directly — no transpose, no
 *         repack, no extra memory.
 *
 * Layout consumed (KleidiAI kxn / Int4QTensor canonical):
 *   packed_data: K rows × ceil(N/2) bytes per row, row-major.
 *     byte[k * row_stride + n/2]:
 *       low  nibble (bits 0-3) = stored value for even n_idx
 *       high nibble (bits 4-7) = stored value for odd  n_idx
 *     stored value = real_int4 + 8  (offset-binary, zero_point=8)
 *
 *   scales: N fp32 values, one per output column (per-channel).
 *
 * Computation:
 *   output[m][n] = scale[n] * sum_k( activation[m][k] * (nibble[k,n] - 8) )
 *
 * @see nntrainer/tensor/int4_tensor.h for canonical layout docs.
 */

#include "x86_qsi4cxp.h"

#include <cstdint>
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace nntrainer {

void gemv_qsi4cxp_kxn_fp32_avx2(unsigned int N, unsigned int K,
                                 const float *activation,
                                 const uint8_t *packed_data,
                                 const float *scales, float *output) {
  const unsigned int row_stride = (N + 1) / 2;

#ifdef __AVX2__
  // Process 32 output columns per iteration (16 bytes of packed data
  // = 32 nibbles). Each iteration walks all K rows, accumulating
  // the dot product in four __m256 registers (8 floats each = 32).
  unsigned int n_block = 0;
  for (; n_block + 32 <= N; n_block += 32) {
    __m256 acc0 = _mm256_setzero_ps(); // columns n_block+0..7
    __m256 acc1 = _mm256_setzero_ps(); // columns n_block+8..15
    __m256 acc2 = _mm256_setzero_ps(); // columns n_block+16..23
    __m256 acc3 = _mm256_setzero_ps(); // columns n_block+24..31

    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    const __m128i offset_8 = _mm_set1_epi8(8);

    for (unsigned int k = 0; k < K; ++k) {
      // Load 16 bytes = 32 nibbles from row k, columns n_block..n_block+31
      const uint8_t *src = packed_data + k * row_stride + n_block / 2;
      __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));

      // Unpack nibbles:
      //   low  = even-n values (bits 0-3 of each byte)
      //   high = odd-n  values (bits 4-7 of each byte)
      __m128i low = _mm_and_si128(packed, nibble_mask);
      __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);

      // Interleave to restore column order:
      //   unpacklo(low, high) = [n0,n1,n2,n3,...,n15]
      //   unpackhi(low, high) = [n16,n17,...,n31]
      __m128i cols_lo = _mm_unpacklo_epi8(low, high); // columns 0-15
      __m128i cols_hi = _mm_unpackhi_epi8(low, high); // columns 16-31

      // Subtract offset 8 (unsigned -> signed): int4_real = stored - 8
      cols_lo = _mm_sub_epi8(cols_lo, offset_8);
      cols_hi = _mm_sub_epi8(cols_hi, offset_8);

      // Broadcast activation[k] to all 8 lanes
      __m256 vact = _mm256_set1_ps(activation[k]);

      // Columns 0-7: sign-extend int8 -> int32 -> fp32, FMA
      __m256i i32_0 = _mm256_cvtepi8_epi32(cols_lo);
      acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_0), vact, acc0);

      // Columns 8-15
      __m256i i32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(cols_lo, 8));
      acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_1), vact, acc1);

      // Columns 16-23
      __m256i i32_2 = _mm256_cvtepi8_epi32(cols_hi);
      acc2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_2), vact, acc2);

      // Columns 24-31
      __m256i i32_3 = _mm256_cvtepi8_epi32(_mm_srli_si128(cols_hi, 8));
      acc3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_3), vact, acc3);
    }

    // Multiply accumulated dot products by per-column scales
    __m256 s0 = _mm256_loadu_ps(scales + n_block);
    __m256 s1 = _mm256_loadu_ps(scales + n_block + 8);
    __m256 s2 = _mm256_loadu_ps(scales + n_block + 16);
    __m256 s3 = _mm256_loadu_ps(scales + n_block + 24);

    _mm256_storeu_ps(output + n_block,      _mm256_mul_ps(acc0, s0));
    _mm256_storeu_ps(output + n_block + 8,  _mm256_mul_ps(acc1, s1));
    _mm256_storeu_ps(output + n_block + 16, _mm256_mul_ps(acc2, s2));
    _mm256_storeu_ps(output + n_block + 24, _mm256_mul_ps(acc3, s3));
  }

  // Scalar tail for remaining columns (N not divisible by 32)
  for (unsigned int n = n_block; n < N; ++n) {
    float acc = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      uint8_t byte = packed_data[k * row_stride + n / 2];
      int8_t val = (n % 2 == 0) ? (int8_t)((byte & 0x0F) - 8)
                                 : (int8_t)(((byte >> 4) & 0x0F) - 8);
      acc += activation[k] * val;
    }
    output[n] = acc * scales[n];
  }

#else
  // Scalar fallback (no AVX2)
  for (unsigned int n = 0; n < N; ++n) {
    float acc = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      uint8_t byte = packed_data[k * row_stride + n / 2];
      int8_t val = (n % 2 == 0) ? (int8_t)((byte & 0x0F) - 8)
                                 : (int8_t)(((byte >> 4) & 0x0F) - 8);
      acc += activation[k] * val;
    }
    output[n] = acc * scales[n];
  }
#endif
}

void gemm_qsi4cxp_kxn_fp32(unsigned int M, unsigned int N, unsigned int K,
                            const float *activation,
                            const uint8_t *packed_data,
                            const float *scales, float *output) {
  // GEMV fast path (M=1): direct AVX2 kernel
  if (M == 1) {
    gemv_qsi4cxp_kxn_fp32_avx2(N, K, activation, packed_data, scales, output);
    return;
  }

  // GEMM (M>1): row-by-row GEMV. Each activation row is independent
  // so this is trivially parallelizable (future: thread pool).
  for (unsigned int m = 0; m < M; ++m) {
    gemv_qsi4cxp_kxn_fp32_avx2(N, K, activation + m * K, packed_data, scales,
                                output + m * N);
  }
}

} // namespace nntrainer
