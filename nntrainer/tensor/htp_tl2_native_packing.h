// SPDX-License-Identifier: Apache-2.0
/** @file htp_tl2_native_packing.h
 * @brief Symmetric reference and native TL2 preparation for QS2 experts.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __NNTRAINER_HTP_TL2_NATIVE_PACKING_H__
#define __NNTRAINER_HTP_TL2_NATIVE_PACKING_H__
#include "htp_backend/htp_tl2_opts.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nntrainer {
/** Preserve pure {-1,0,1} palettes and their original scales. Other int4
 * palettes retain the legacy symmetric {-7,0,7} approximation. Native TL2
 * stores signs; the epilogue restores the magnitude recorded in the palette.
 * packed: QS2 file record [codes][palette][scales][float column sums].
 * native may be null for the symmetric QS2 reference. */
inline void prepareSymmetricTl2(uint8_t *packed, uint32_t K, uint32_t N,
                                std::vector<uint8_t> *native) {
  if (!packed || !K || !N || K % 32u || N % 32u || K > 8192u || N > 8192u)
    throw std::invalid_argument("symmetric TL2: invalid WH dimensions");
  const size_t codes = static_cast<size_t>(K) * N / 4u;
  int8_t pal[4];
  std::memcpy(pal, packed + codes, 4u);
  bool unit = true;
  for (const auto value : pal)
    unit = unit && value >= -1 && value <= 1;
  for (auto &value : pal) {
    if (value < -8 || value > 7)
      throw std::invalid_argument("symmetric TL2: invalid palette");
    if (!unit)
      value = value < 0 ? -7 : value > 0 ? 7 : 0;
  }
  std::vector<int32_t> sums(N, 0);
  for (uint32_t kt = 0; kt < K / 32u; ++kt)
    for (uint32_t nt = 0; nt < N / 32u; ++nt) {
      const uint8_t *tile =
        packed + (static_cast<size_t>(kt) * (N / 32u) + nt) * 256u;
      for (uint32_t j = 0; j < 256u; ++j) {
        const unsigned v = tile[j];
        sums[nt * 32u + (j % 128u) / 4u] += pal[v & 3u] + pal[(v >> 2u) & 3u] +
                                            pal[(v >> 4u) & 3u] + pal[v >> 6u];
      }
    }
  std::memcpy(packed + codes, pal, 4u);
  for (uint32_t n = 0; n < N; ++n) {
    const float value = static_cast<float>(sums[n]);
    std::memcpy(packed + codes + 4u + 4u * (N + n), &value, 4u);
  }
  if (!native)
    return;
  native->assign(static_cast<size_t>(htp_tl2_native_bytes(K, N)), 0u);
  const uint32_t blocks = (K + 23u) / 24u;
  for (uint32_t p = 0; p < (N + 127u) / 128u; ++p)
    for (uint32_t kb = 0; kb < blocks; ++kb) {
      uint8_t *dst =
        native->data() + (static_cast<size_t>(p) * blocks + kb) * 640u;
      for (uint32_t j = 0; j < 128u; ++j) {
        const uint32_t n = p * 128u + (j % 4u) * 32u + j / 4u;
        for (uint32_t g = 0; g < 8u; ++g) {
          int value = 0;
          for (uint32_t t = 0; t < 3u; ++t) {
            const uint32_t k = kb * 24u + 3u * g + t;
            int w = 0;
            if (k < K && n < N) {
              const uint32_t r = k % 32u;
              const size_t at =
                (static_cast<size_t>(k / 32u) * (N / 32u) + n / 32u) * 256u +
                (r / 16u) * 128u + (n % 32u) * 4u + r % 4u;
              const int level =
                pal[(packed[at] >> (2u * ((r / 4u) % 4u))) & 3u];
              w = level < 0 ? -1 : level > 0 ? 1 : 0;
            }
            value = value * 3 + w;
          }
          const unsigned magnitude = value < 0 ? -value : value;
          dst[(g / 2u) * 128u + j] |= magnitude << (4u * (g % 2u));
          dst[512u + j] |= (value < 0) << g;
        }
      }
    }
}
} // namespace nntrainer
#endif
