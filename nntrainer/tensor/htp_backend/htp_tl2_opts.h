// SPDX-License-Identifier: Apache-2.0
/** @file htp_tl2_opts.h
 * @brief Opt-in symmetric TL2 decode with a QS2 prefill companion.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __NNTRAINER_HTP_TL2_OPTS_H__
#define __NNTRAINER_HTP_TL2_OPTS_H__
#include <stdint.h>
/* Bit 1 is independent of Conv output-chunk options at bits 30..31. */
#define HTP_TL2_FLAG 0x2u
/** Miss protocol format: QS2 arena span with native TL2 codes. */
#define HTP_TL2_WEIGHT_BITS 0x102u
/** Native-only codes followed immediately by scales and integer sums. */
#define HTP_TL2_COMPACT_WEIGHT_BITS 0x202u

/** Magnitude shared by all nonzero palette entries; zero means unsupported.
 * Pure ternary uses one, and legacy symmetric int4 artifacts use seven. */
static inline uint32_t htp_tl2_palette_magnitude(const int8_t palette[4]) {
  uint32_t magnitude = 0u;
  for (uint32_t j = 0; j < 4u; ++j) {
    const int value = palette[j];
    const uint32_t level = (uint32_t)(value < 0 ? -value : value);
    if (!level)
      continue;
    if ((level != 1u && level != 7u) || (magnitude && magnitude != level))
      return 0u;
    magnitude = level;
  }
  return magnitude ? magnitude : 1u;
}

/** Arena suffix after QS2 codes, scales, sums and palette. */
static inline uint64_t htp_tl2_native_offset(uint32_t K, uint32_t N) {
  return ((uint64_t)K * N / 4u + 8u * N + 4u + 127u) & ~(uint64_t)127u;
}
static inline uint64_t htp_tl2_native_bytes(uint32_t K, uint32_t N) {
  return ((uint64_t)K + 23u) / 24u * (((uint64_t)N + 127u) / 128u) * 640u;
}
#endif
