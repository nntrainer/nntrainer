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

void htp_qs4cx_from_q4_0x4(const void *q4_0x4_repacked, uint32_t K, uint32_t N,
                           int8_t *q_w4_i8, float *w_scale, int32_t *colsum_w) {
  // Q4_0Utils::dequantizeQ4_0x4's own convention: N rows of K values each,
  // row-major -- one row per output channel, which is exactly the axis
  // qs4cx's per-channel scale needs, so no transpose is needed before the
  // per-row min/max pass below.
  std::vector<float> w_f32(static_cast<size_t>(N) * K);
  Q4_0Utils::dequantizeQ4_0x4(q4_0x4_repacked, static_cast<int>(N),
                              static_cast<int>(K), w_f32.data());

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
    const float scale = (rmin == rmax) ? 1.0f : 15.0f / (rmax - rmin);

    int32_t sum = 0;
    for (uint32_t k = 0; k < K; ++k) {
      int32_t q = static_cast<int32_t>(std::round(row[k] * scale));
      q = std::max(-8, std::min(7, q));
      q_w4_i8[static_cast<size_t>(k) * N + n] = static_cast<int8_t>(q);
      sum += q;
    }
    w_scale[n] = 1.0f / scale;
    colsum_w[n] = sum;
  }
}

} // namespace nntrainer
