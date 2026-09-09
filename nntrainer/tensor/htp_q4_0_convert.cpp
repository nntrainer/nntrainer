// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_q4_0_convert.cpp
 * @date   04 Sep 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <htp_q4_0_convert.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <q4_0_utils.h>

namespace nntrainer {

namespace {
/**
 * @brief Tile edge for the transposed writes below.
 *
 * q_w4_i8 wants [K, N] (row-major by input channel -- what hexkl's WH bake
 * consumes), but every source this file reads from is naturally [N, K]
 * (row-major by output channel -- what the per-channel scale/colsum need).
 * A full transpose has no loop order where both the read and the write are
 * contiguous; done as a single (n outer, k inner) pass the write side
 * touches a fresh N-stride-apart cache line on nearly every store -- device
 * measured at 25.4 ms/weight (~4.6 ns/value), almost entirely cache-miss
 * cost, not arithmetic (docs/htp_attention -- 41's follow-on measurement,
 * registration section). 64x64 keeps one tile's int8 buffer (4 KiB) well
 * inside a typical 32-64 KiB L1: fill it in the cache-friendly read order,
 * then drain it transposed -- the only strided access left touches this
 * tiny, already-resident buffer instead of the full K*N destination.
 */
constexpr uint32_t TRANSPOSE_TILE = 64;
} // namespace

void htp_qs4cx_from_q4_0x4(const void *q4_0x4_repacked, uint32_t K, uint32_t N,
                           int8_t *q_w4_i8, float *w_scale, int32_t *colsum_w) {
  // Q4_0Utils::dequantizeQ4_0x4's own convention: N rows of K values each,
  // row-major -- one row per output channel, which is exactly the axis
  // qs4cx's per-channel scale needs, so no transpose is needed before the
  // per-row min/max pass below.
  std::vector<float> w_f32(static_cast<size_t>(N) * K);
  Q4_0Utils::dequantizeQ4_0x4(q4_0x4_repacked, static_cast<int>(N),
                              static_cast<int>(K), w_f32.data());

  // Pass 1: per-row min/max -> scale. Already contiguous in k for each n,
  // so this pass was never the cost -- only pass 2's write was.
  std::vector<float> scale(N);
  for (uint32_t n = 0; n < N; ++n) {
    const float *row = w_f32.data() + static_cast<size_t>(n) * K;
    float min0 = row[0];
    float max0 = row[0];
    for (uint32_t k = 1; k < K; ++k) {
      min0 = std::min(min0, row[k]);
      max0 = std::max(max0, row[k]);
    }
    const float rmin = std::min(0.0f, min0);
    const float rmax = std::max(0.0f, max0);
    scale[n] = (rmin == rmax) ? 1.0f : 15.0f / (rmax - rmin);
    w_scale[n] = 1.0f / scale[n];
    colsum_w[n] = 0;
  }

  // Pass 2: quantize + transposed write, tiled (see TRANSPOSE_TILE's doc).
  int8_t tile[TRANSPOSE_TILE][TRANSPOSE_TILE];
  for (uint32_t n0 = 0; n0 < N; n0 += TRANSPOSE_TILE) {
    const uint32_t n1 = std::min(n0 + TRANSPOSE_TILE, N);
    for (uint32_t k0 = 0; k0 < K; k0 += TRANSPOSE_TILE) {
      const uint32_t k1 = std::min(k0 + TRANSPOSE_TILE, K);

      for (uint32_t n = n0; n < n1; ++n) {
        const float *row = w_f32.data() + static_cast<size_t>(n) * K;
        const float s = scale[n];
        int32_t sum = 0;
        for (uint32_t k = k0; k < k1; ++k) {
          int32_t q = static_cast<int32_t>(std::round(row[k] * s));
          q = std::max(-8, std::min(7, q));
          tile[n - n0][k - k0] = static_cast<int8_t>(q);
          sum += q;
        }
        colsum_w[n] += sum;
      }

      for (uint32_t k = k0; k < k1; ++k) {
        int8_t *dst_row = q_w4_i8 + static_cast<size_t>(k) * N + n0;
        for (uint32_t n = n0; n < n1; ++n) {
          dst_row[n - n0] = tile[n - n0][k - k0];
        }
      }
    }
  }
}

void htp_qs4cx_from_packed(const void *qs4cx_packed, const float *qs4cx_scales,
                           uint32_t K, uint32_t N, int8_t *q_w4_i8,
                           float *w_scale, int32_t *colsum_w) {
  const uint8_t *packed = static_cast<const uint8_t *>(qs4cx_packed);
  const size_t stride = (static_cast<size_t>(K) + 1) / 2;

  // The scale passes through untouched (already 1/scale in both
  // conventions -- see this function's docstring); colsum_w accumulates
  // per k-tile below.
  for (uint32_t n = 0; n < N; ++n) {
    w_scale[n] = qs4cx_scales[n];
    colsum_w[n] = 0;
  }

  int8_t tile[TRANSPOSE_TILE][TRANSPOSE_TILE];
  for (uint32_t n0 = 0; n0 < N; n0 += TRANSPOSE_TILE) {
    const uint32_t n1 = std::min(n0 + TRANSPOSE_TILE, N);
    for (uint32_t k0 = 0; k0 < K; k0 += TRANSPOSE_TILE) {
      const uint32_t k1 = std::min(k0 + TRANSPOSE_TILE, K);

      for (uint32_t n = n0; n < n1; ++n) {
        const uint8_t *row = packed + static_cast<size_t>(n) * stride;
        int32_t sum = 0;
        for (uint32_t k = k0; k < k1; ++k) {
          // quant_qs4cx_f32 packs the even k into the low nibble and the
          // odd k into the high one, storing q + 8 so the nibble stays
          // unsigned.
          const uint8_t byte = row[k >> 1];
          const uint8_t nibble = (k & 1u) ? (byte >> 4) : (byte & 0x0Fu);
          const int8_t q =
            static_cast<int8_t>(static_cast<int32_t>(nibble) - 8);
          tile[n - n0][k - k0] = q;
          sum += q;
        }
        colsum_w[n] += sum;
      }

      for (uint32_t k = k0; k < k1; ++k) {
        int8_t *dst_row = q_w4_i8 + static_cast<size_t>(k) * N + n0;
        for (uint32_t n = n0; n < n1; ++n) {
          dst_row[n - n0] = tile[n - n0][k - k0];
        }
      }
    }
  }
}

} // namespace nntrainer
