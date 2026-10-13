// SPDX-License-Identifier: Apache-2.0
/**
 * @file	avx2_tq2cx_impl.cpp
 * @date	02 October 2026
 * @brief	AVX2 tile kernel for the TQ2CX (ternary) batch GEMM.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs
 *
 * Eight output channels per vector, one per float lane, so every lane walks
 * the input taps in ascending order exactly like the scalar reference (see
 * tq2cx_kernels.h). For one 32-tap unit the pos / neg words of the eight rows
 * are gathered once; tap b is then turned into a per-lane weight
 * w in {-1, 0, +1} with two shifts, a blend and an or (the low bit of the
 * running (pos | neg) word selects 1.0f, the low bit of the running neg word,
 * shifted into the sign position, makes it negative), and accumulated with
 * acc = fma(x[k], w, acc), which is bit-identical to the reference add /
 * subtract. The decoded weight vector is shared by up to MR activation rows
 * and NG groups of eight channels run as independent dependency chains.
 *
 * Stored layout: the gather reads the unit of eight consecutive rows. Row-major
 * puts them a row (units * 8 bytes) apart; the tiled layout puts eight rows of
 * one tile (rows n0 .. n0 + 7 with n0 % 8 == 0 and R % 8 == 0) one block row
 * (w * 8 bytes, w = the block's width) apart, so only the gather base and
 * stride change. Values and the per-lane summation order do not.
 */

#include <algorithm>
#include <cstring>
#include <immintrin.h>

#include <avx2_tq2cx_impl.h>

namespace nntrainer::avx2 {

namespace {

/**
 * @brief MR rows x (NG * 8) output channels starting at row m0 / channel n0
 */
template <int MR, int NG>
inline void tq2cx_block(const float *A, unsigned int lda, unsigned int m0,
                        const uint32_t *masks, size_t units, unsigned int K,
                        const float *scales, unsigned int n0, float *C,
                        unsigned int ldc, const Tq2cxLayout &lay) {
  const __m256i iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  // uint32 between two consecutive rows' copies of unit u
  auto row_stride = [&](size_t u) -> int {
    if (!lay.tiled())
      return static_cast<int>(units * 2);
    const size_t b = u / lay.V;
    return static_cast<int>(2 * std::min<size_t>(lay.V, units - b * lay.V));
  };
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 zero = _mm256_setzero_ps();

  __m256 acc[MR][NG];
  for (int m = 0; m < MR; ++m)
    for (int g = 0; g < NG; ++g)
      acc[m][g] = _mm256_setzero_ps();

  const float *xrow[MR];
  for (int m = 0; m < MR; ++m)
    xrow[m] = A + static_cast<size_t>(m0 + m) * lda;

  for (size_t u = 0; u < units; ++u) {
    __m256i nz[NG], ng[NG];
    const __m256i idx =
      _mm256_mullo_epi32(_mm256_set1_epi32(row_stride(u)), iota);
    for (int g = 0; g < NG; ++g) {
      const uint32_t *base =
        masks +
        2 * tq2cxUnitIndex(lay, static_cast<size_t>(n0) + 8 * g, u, units);
      const __m256i p =
        _mm256_i32gather_epi32(reinterpret_cast<const int *>(base), idx, 4);
      const __m256i q =
        _mm256_i32gather_epi32(reinterpret_cast<const int *>(base + 1), idx, 4);
      nz[g] = _mm256_or_si256(p, q);
      ng[g] = q;
    }

    const size_t kb = u * 32;
    const size_t lim = std::min<size_t>(32, K - kb);
    for (size_t b = 0; b < lim; ++b) {
      __m256 w[NG];
      for (int g = 0; g < NG; ++g) {
        const __m256 sel = _mm256_castsi256_ps(_mm256_slli_epi32(nz[g], 31));
        const __m256 sgn = _mm256_castsi256_ps(_mm256_slli_epi32(ng[g], 31));
        w[g] = _mm256_or_ps(_mm256_blendv_ps(zero, one, sel), sgn);
        nz[g] = _mm256_srli_epi32(nz[g], 1);
        ng[g] = _mm256_srli_epi32(ng[g], 1);
      }
      for (int m = 0; m < MR; ++m) {
        const __m256 x = _mm256_broadcast_ss(xrow[m] + kb + b);
        for (int g = 0; g < NG; ++g)
          acc[m][g] = _mm256_fmadd_ps(x, w[g], acc[m][g]);
      }
    }
  }

  for (int g = 0; g < NG; ++g) {
    const __m256 s = _mm256_loadu_ps(scales + n0 + 8 * g);
    for (int m = 0; m < MR; ++m)
      _mm256_storeu_ps(C + static_cast<size_t>(m0 + m) * ldc + n0 + 8 * g,
                       _mm256_mul_ps(acc[m][g], s));
  }
}

/**
 * @brief all output channel groups of [n0, n1) for MR rows at m0
 * @return first channel not handled (the < 8 tail)
 */
template <int MR>
inline unsigned int
tq2cx_rows(const float *A, unsigned int lda, unsigned int m0,
           const uint32_t *masks, size_t units, unsigned int K,
           const float *scales, unsigned int n0, unsigned int n1, float *C,
           unsigned int ldc, const Tq2cxLayout &lay) {
  // keep about eight independent accumulator chains in flight
  constexpr int NG_MAX = MR >= 4 ? 2 : 4;
  unsigned int n = n0;
  for (; n + 8 * NG_MAX <= n1; n += 8 * NG_MAX)
    tq2cx_block<MR, NG_MAX>(A, lda, m0, masks, units, K, scales, n, C, ldc,
                            lay);
  if (NG_MAX > 2)
    for (; n + 16 <= n1; n += 16)
      tq2cx_block<MR, 2>(A, lda, m0, masks, units, K, scales, n, C, ldc, lay);
  for (; n + 8 <= n1; n += 8)
    tq2cx_block<MR, 1>(A, lda, m0, masks, units, K, scales, n, C, ldc, lay);
  return n;
}

} // namespace

void tq2cx_tile(const float *A, unsigned int lda, unsigned int m0,
                unsigned int m1, const void *W, unsigned int N, unsigned int K,
                unsigned int n0, unsigned int n1, float *C, unsigned int ldc,
                const Tq2cxLayout &lay) {
  const size_t units = (K + 31) / 32;
  const uint32_t *masks = static_cast<const uint32_t *>(W);
  // the scale tail starts after the (padded) masks, 4-byte aligned
  const float *scales = reinterpret_cast<const float *>(
    static_cast<const uint8_t *>(W) + tq2cxMaskBytes(lay, N, K));
  // a tiled layout keeps eight rows of one gather inside one tile only from
  // an 8-aligned first row
  if (lay.tiled() && (n0 % 8 != 0 || lay.R % 8 != 0)) {
    tq2cx_tile_ref(A, lda, m0, m1, W, N, K, n0, n1, C, ldc, lay);
    return;
  }

  unsigned int n_tail = n0;
  unsigned int m = m0;
  for (; m + 4 <= m1; m += 4)
    n_tail =
      tq2cx_rows<4>(A, lda, m, masks, units, K, scales, n0, n1, C, ldc, lay);
  for (; m < m1; ++m)
    n_tail =
      tq2cx_rows<1>(A, lda, m, masks, units, K, scales, n0, n1, C, ldc, lay);

  if (n_tail < n1)
    tq2cx_tile_ref(A, lda, m0, m1, W, N, K, n_tail, n1, C, ldc, lay);
}

} // namespace nntrainer::avx2
