// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Portions of this file are derived from llama.cpp
 * (https://github.com/ggml-org/llama.cpp), licensed under the MIT License.
 * Copyright (c) Contributors to llama.cpp
 *
 * Modified by Sungsik Kong, 2025: Adapted for CPU backend integration
 *
 * @file   nntr_ggml_impl_fallback.cpp
 * @date   9 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback (scalar C) implementations of GGML functions
 *
 * This file provides portable C implementations that work on any platform.
 * For optimized SIMD implementations, use architecture-specific files:
 * - nntr_ggml_impl_neon.cpp for ARM NEON (armv8.2-a)
 * - nntr_ggml_impl_sve.cpp for ARM SVE (armv9.2-a)
 * - nntr_ggml_impl_avx.cpp for x86_64 AVX/AVX2
 */

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <math.h>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>

#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_common.h>
#include <nntr_ggml_impl_utils.h>

//============================================================================
// Helper functions for block packing
//============================================================================

static block_q4_0x4 nntr_make_block_q4_0x4(block_q4_0 *in,
                                           unsigned int blck_size_interleave) {
  block_q4_0x4 out;

  for (int i = 0; i < 4; i++) {
    out.d[i] = in[i].d;
  }

  const int end = Q4_0 * 2 / blck_size_interleave;

  if (blck_size_interleave == 8) {
    const uint64_t xor_mask = 0x8888888888888888ULL;
    for (int i = 0; i < end; ++i) {
      int src_id = i % 4;
      int src_offset = (i / 4) * blck_size_interleave;
      int dst_offset = i * blck_size_interleave;

      uint64_t elems;
      // Using memcpy to avoid unaligned memory accesses
      memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
      elems ^= xor_mask;
      memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }
  } else if (blck_size_interleave == 4) {
    const uint32_t xor_mask = 0x88888888;
    for (int i = 0; i < end; ++i) {
      int src_id = i % 4;
      int src_offset = (i / 4) * blck_size_interleave;
      int dst_offset = i * blck_size_interleave;

      uint32_t elems;
      memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
      elems ^= xor_mask;
      memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
    }
  } else {
    assert(false);
  }

  return out;
}

static block_q4_0x8 nntr_make_block_q4_0x8(block_q4_0 *in,
                                           unsigned int blck_size_interleave) {
  block_q4_0x8 out;

  for (int i = 0; i < 8; i++) {
    out.d[i] = in[i].d;
  }

  const int end = QK_0<4>() * 4 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int src_id = i % 8;
    int src_offset = (i / 8) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
  }

  return out;
}

static block_q4_Kx8 make_block_q4_Kx8(block_q4_K *in,
                                      unsigned int blck_size_interleave) {
  block_q4_Kx8 out;
  // Delta(scale) and dmin values of the eight Q4_K structures are copied onto
  // the output interleaved structure
  for (int i = 0; i < 8; i++) {
    out.d[i] = in[i].data.data.d;
  }

  for (int i = 0; i < 8; i++) {
    out.dmin[i] = in[i].data.data.dmin;
  }

  const int end = QK_K * 4 / blck_size_interleave;

  // Interleave Q4_K quants by taking 8 bytes at a time
  for (int i = 0; i < end; ++i) {
    int src_id = i % 8;
    int src_offset = (i / 8) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
    memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
  }

  // The below logic is designed so as to unpack and rearrange scales and mins
  // values in Q4_K Currently the Q4_K structure has 8 scales and 8 mins packed
  // in 12 bytes ( 6 bits for each value) The output Q4_Kx8 structure has 96
  // bytes Every 12 byte is packed such that it contains scales and mins for
  // corresponding sub blocks from Q4_K structure For eg - First 12 bytes
  // contains 8 scales and 8 mins - each of first sub block from different Q4_K
  // structures
  uint8_t s[8], m[8];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      s[j] = in[j].scales[i] & 63;
      m[j] = in[j].scales[i + 4] & 63;
    }

    out.scales[i * 12] = (s[0] & 63) + ((s[4] & 48) << 2);
    out.scales[i * 12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
    out.scales[i * 12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
    out.scales[i * 12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
    out.scales[i * 12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
    out.scales[i * 12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
    out.scales[i * 12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
    out.scales[i * 12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
    out.scales[i * 12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
    out.scales[i * 12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
    out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
    out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
      m[j] =
        ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
    }

    out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
    out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
    out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
    out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
    out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
    out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
    out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
    out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
    out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
    out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
    out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
    out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  return out;
}

//============================================================================
// GEMV (General Matrix-Vector Multiplication) - Q4_0 4x8
//============================================================================

void nntr_gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  float sumf[4];
  int sumi;

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++)
      sumf[j] = 0.0;
    for (int l = 0; l < nb; l++) {
      for (int k = 0; k < (qk / (2 * blocklen)); k++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i]
                       << 4);
            const int v1 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i] &
                       0xF0);
            sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) +
                     (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >>
                    4;
          }
          sumf[j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                     nntr_compute_fp16_to_fp32(a_ptr[l].d);
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++)
      s[x * ncols_interleaved + j] = sumf[j];
  }
}

//============================================================================
// GEMM (General Matrix-Matrix Multiplication) - Q4_0 4x8
//============================================================================

void nntr_gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  float sumf[4][4];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          sumf[m][j] = 0.0;
      }
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i]
                           << 4);
                const int v1 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i] &
                           0xF0);
                sumi +=
                  ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                   (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i +
                                     qk / 2 * 4])) >>
                  4;
              }
              sumf[m][j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                            nntr_compute_fp16_to_fp32(a_ptr[l].d[m]);
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
      }
    }
  }
}

//============================================================================
// GEMM/GEMV - Q4_0 8x8 (NYI in fallback - requires SIMD for performance)
//============================================================================

void nntr_gemm_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for 8x8 GEMM
  // For armv7l, the 4x8 kernels should be used instead
  throw std::runtime_error("NYI: nntr_gemm_q4_0_8x8_q8_0 fallback - use 4x8 "
                           "kernels for armv7l");
}

void nntr_gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for 8x8 GEMV
  throw std::runtime_error("NYI: nntr_gemv_q4_0_8x8_q8_0 fallback - use 4x8 "
                           "kernels for armv7l");
}

//============================================================================
// GEMM/GEMV - Q4_K 8x8 (NYI in fallback)
//============================================================================

void nntr_gemm_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for q4_K GEMM
  throw std::runtime_error("NYI: nntr_gemm_q4_K_8x8_q8_K fallback");
}

void nntr_gemv_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for q4_K GEMV
  throw std::runtime_error("NYI: nntr_gemv_q4_K_8x8_q8_K fallback");
}

//============================================================================
// Quantization helper functions (matrix packing)
//============================================================================

void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  // NYI: Fallback quantization - requires row-by-row implementation
  throw std::runtime_error("NYI: nntr_quantize_mat_q8_0_4x8 fallback");
}

void nntr_quantize_mat_q8_K_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  // NYI: Fallback quantization
  throw std::runtime_error("NYI: nntr_quantize_mat_q8_K_4x8 fallback");
}

//============================================================================
// Vector dot product - Q6_K
//============================================================================

void nntr_vec_dot_q6_K_q8_K(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx, size_t bx,
                            const void *__restrict vy, size_t by, int nrc) {
  assert(n % QK_K == 0);
  assert(nrc == 1);

  const block_q6_K *__restrict x = (const block_q6_K *)vx;
  const block_q8_K *__restrict y = (const block_q8_K *)vy;

  const int nb = n / QK_K;

  float sumf = 0.0f;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * nntr_compute_fp16_to_fp32(x[i].d);
    const uint8_t *__restrict q4 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict q8 = y[i].qs;
    const int8_t *__restrict scales = x[i].scales;

    int32_t sum = 0;

    for (int j = 0; j < QK_K / 128; ++j) {
      int32_t sum_block = 0;
      for (int l = 0; l < 32; ++l) {
        // Decode 6-bit quantized values
        const int q4_0 = q4[l] & 0xF;
        const int q4_1 = q4[l] >> 4;
        const int q4_2 = q4[l + 32] & 0xF;
        const int q4_3 = q4[l + 32] >> 4;

        const int qh_0 = (qh[l] & 0x03) << 4;
        const int qh_1 = (qh[l] & 0x0C) << 2;
        const int qh_2 = (qh[l] & 0x30);
        const int qh_3 = (qh[l] & 0xC0) >> 2;

        const int q6_0 = (q4_0 | qh_0) - 32;
        const int q6_1 = (q4_1 | qh_1) - 32;
        const int q6_2 = (q4_2 | qh_2) - 32;
        const int q6_3 = (q4_3 | qh_3) - 32;

        sum_block += q6_0 * q8[l] * scales[0] + q6_1 * q8[l + 32] * scales[1] +
                     q6_2 * q8[l + 64] * scales[2] +
                     q6_3 * q8[l + 96] * scales[3];
      }
      sum += sum_block;
      q4 += 64;
      qh += 32;
      q8 += 128;
      scales += 4;
    }
    sumf += d * sum;
  }
  *s = sumf;
}

//============================================================================
// Repack functions - Q4_0 to interleaved formats
//============================================================================

int nntr_repack_q4_0_to_q4_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 4 || interleave_block == 8);
  constexpr int nrows_interleaved = 4;

  block_q4_0x4 *dst_ = (block_q4_0x4 *)dst;
  const block_q4_0 *src = (const block_q4_0 *)data;
  block_q4_0 dst_tmp[4];
  int nblocks = k / Q4_0;

  assert(data_size == nrow * nblocks * sizeof(block_q4_0));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = nntr_make_block_q4_0x4(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

int nntr_repack_q4_0_to_q4_0_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 8);
  constexpr size_t nrows_interleaved = 8;

  block_q4_0x8 *dst_ = (block_q4_0x8 *)dst;
  const block_q4_0 *src = (const block_q4_0 *)data;
  block_q4_0 dst_tmp[8];
  int nblocks = k / QK_0<4>();

  assert(data_size == nrow * nblocks * sizeof(block_q4_0));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = nntr_make_block_q4_0x8(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

int nntr_repack_q4_K_to_q4_K_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 8);
  constexpr size_t nrows_interleaved = 8;

  block_q4_Kx8 *dst_ = (block_q4_Kx8 *)dst;
  const block_q4_K *src = (const block_q4_K *)data;
  block_q4_K dst_tmp[8];
  int nblocks = k / QK_K;

  assert(data_size == nrow * nblocks * sizeof(block_q4_K));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = make_block_q4_Kx8(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}
