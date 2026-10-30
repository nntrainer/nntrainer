// SPDX-License-Identifier: Apache-2.0
/** @file htp_qs2_lut_layout.h
 * @brief Optional offline decode image after the existing compact WH arena.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __HTP_QS2_LUT_LAYOUT_H__
#define __HTP_QS2_LUT_LAYOUT_H__
#include <stdint.h>

/** Preserve codes, scales, sums and palette for every existing HMX path.
 * The appended image is [K/32][N/128][8][128] bytes. Within each tile it
 * uses hvx_i2_lut_pack_wh's nibble pairing and output permutation.
 * All offset arithmetic is 64-bit until the arena bound check. */
static inline uint64_t htp_qs2_lut_offset(uint32_t K, uint32_t N) {
  return ((uint64_t)K * N / 4u + (uint64_t)N * 8u + 4u + 511u) & ~UINT64_C(511);
}
#endif
