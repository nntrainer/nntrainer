// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_act_quant.cpp
 * @date   08 Sep 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <htp_act_quant.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nntrainer {

namespace {

/**
 * @brief Round-to-nearest-even, without a libm call.
 *
 * Same trick as hvx_convert.h's hvx_sf_to_w_rne on the DSP side: adding a
 * magic constant (1.5 * 2^23) forces the FPU to round @a f to integer
 * precision using whatever the current rounding mode is (round-to-nearest-
 * even, same assumption this file's own callers and quant_row_params_one
 * already make), then subtracting the magic bit pattern back out recovers
 * the integer. Device-measured motivation: a plain std::nearbyint call
 * here cost 265-301us over a 55-row prefill activation (this function's
 * whole point is removing a DSP-side cost smaller than that), almost
 * entirely the libm call; this is 1.37-1.4x faster and bit-identical
 * (checked against std::nearbyint over both production shapes).
 *
 * Valid for |f| well under 2^22 -- true here by a wide margin: f is either
 * -rmin/scale (at most 255, by construction: rmax-rmin <= 255*scale) or
 * x*inv_scale for an activation value, neither of which approaches that
 * bound before the caller's own clamp to [0,255].
 */
inline int32_t round_nearest_even(float f) {
  constexpr float MAGIC = 12582912.0f; // 1.5 * 2^23
  const float biased = f + MAGIC;
  int32_t bits;
  std::memcpy(&bits, &biased, sizeof(bits));
  return bits - 0x4B400000; // MAGIC's own bit pattern, as an int
}

} // namespace

void htp_quant_pack_u8_ah(const float *x, uint32_t M, uint32_t K,
                          uint8_t *out_ah, float *act_scale, int32_t *act_zp) {
  const uint32_t m_pad = htp_act_m_pad(M);
  const uint32_t n_ktiles = K / HTP_ACT_TILE_INNER;

  // Padding rows (M..m_pad) get scale 1 / zp 0 / zeroed bytes, matching
  // hvx_quant_rows_u8_params + hvx_quant_pack_u8_ah's combined output --
  // the DSP-side dequant never reads them (m_valid bounds every readout),
  // so only "does not crash", not "is meaningful", is required here.
  std::memset(out_ah, 0, static_cast<size_t>(m_pad) * K);
  for (uint32_t m = 0; m < m_pad; ++m) {
    act_scale[m] = 1.0f;
    act_zp[m] = 0;
  }

  for (uint32_t m = 0; m < M; ++m) {
    const float *row = x + static_cast<size_t>(m) * K;

    // K1: per-row scale/zp, 0 folded into the range -- quant_row_params_one's
    // exact formula.
    float min0 = row[0];
    float max0 = row[0];
    for (uint32_t k = 1; k < K; ++k) {
      min0 = std::min(min0, row[k]);
      max0 = std::max(max0, row[k]);
    }
    const float rmin = std::min(0.0f, min0);
    const float rmax = std::max(0.0f, max0);
    if (rmin == rmax) {
      continue; // leaves scale 1, zp 0, bytes 0 -- same early return
    }
    const float scale = (rmax - rmin) / 255.0f;
    // round-to-nearest-even, matching hvx_sf_to_w_rne and
    // quant_row_params_one's own nearbyintf call for this same zp -- see
    // round_nearest_even's doc for why this isn't nearbyintf itself.
    int32_t zp = round_nearest_even(-rmin / scale);
    zp = std::max(0, std::min(255, zp));
    act_scale[m] = scale;
    act_zp[m] = zp;

    // K2: quantize + AH-tile pack. hvx_quant_pack_u8_ah rounds x*inv_scale
    // to the nearest even integer, THEN adds zp, THEN clamps to [0,255] --
    // that order, not "round(x*inv_scale+zp)" (the two agree numerically
    // since zp is already an integer, but this keeps the arithmetic a
    // literal restatement rather than a derived equivalent).
    const float inv_scale = 1.0f / scale;
    const uint32_t rb = m / HTP_ACT_TILE_ROW;
    const uint32_t r = m % HTP_ACT_TILE_ROW;
    for (uint32_t kt = 0; kt < n_ktiles; ++kt) {
      uint8_t *dst =
        out_ah +
        (static_cast<size_t>(rb) * n_ktiles + kt) * HTP_ACT_TILE_BYTES +
        static_cast<size_t>(r) * HTP_ACT_TILE_INNER;
      for (uint32_t c = 0; c < HTP_ACT_TILE_INNER; ++c) {
        const uint32_t k = kt * HTP_ACT_TILE_INNER + c;
        int32_t q = round_nearest_even(row[k] * inv_scale);
        q += zp;
        q = std::max(0, std::min(255, q));
        dst[c] = static_cast<uint8_t>(q);
      }
    }
  }
}

} // namespace nntrainer
