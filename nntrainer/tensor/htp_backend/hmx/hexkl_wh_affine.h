// SPDX-License-Identifier: Apache-2.0
/**
 * @file hexkl_wh_affine.h
 * @brief Exact affine WH activation sums and accumulator correction.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#ifndef NNTR_HEXKL_WH_AFFINE_H
#define NNTR_HEXKL_WH_AFFINE_H

#include <stddef.h>
#include <stdint.h>

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

/** Sum the *same* quantized activation consumed by HMX/HVX, with its zero
 * point removed. AH is a block of 64 rows, K/32 tiles of 2048 bytes.
 * For W=s*(q+offset), add offset*sum(u-zp) to the signed-int4 accumulator
 * before ordinary dequantization. No second activation quantizer is used. */
static inline void hexkl_wh_affine_sums(const uint8_t *ah, uint32_t rows,
                                        uint32_t K, const int32_t *zp,
                                        int32_t *sums) {
#if defined(__hexagon__)
  for (uint32_t r = 0; r < rows; r += 4u) {
    HVX_Vector sum = Q6_V_vzero();
    for (uint32_t kt = 0; kt < K / 32u; ++kt) {
      const HVX_Vector a =
        *(const HVX_UVector *)(ah + (size_t)kt * 2048u + r * 32u);
      sum = Q6_Vuw_vrmpyacc_VuwVubRub(sum, a, 0x01010101);
    }
    /** Each AH vector contains four rows. Reduce its eight word sums per
     * row; only lanes 0, 8, 16 and 24 are used, so no row boundary wraps. */
    sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, 4));
    sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, 8));
    sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, 16));
    int32_t lane[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lane = sum;
    for (uint32_t j = 0; j < 4u && r + j < rows; ++j)
      sums[r + j] = lane[j * 8u] - (int32_t)K * zp[r + j];
  }
#else
  for (uint32_t r = 0; r < rows; ++r) {
    int32_t sum = -(int32_t)K * zp[r];
    for (uint32_t kt = 0; kt < K / 32u; ++kt)
      for (uint32_t k = 0; k < 32u; ++k)
        sum += ah[(size_t)kt * 2048u + r * 32u + k];
    sums[r] = sum;
  }
#endif
}

static inline void hexkl_wh_affine_acc(int32_t *acc, uint32_t stride,
                                       uint32_t rows, uint32_t cols,
                                       const int32_t *offset,
                                       const int32_t *sums) {
  if (!offset)
    return;
  uint32_t c = 0;
#if defined(__hexagon__)
  for (; c + 32u <= cols; c += 32u) {
    const HVX_Vector off = *(const HVX_UVector *)(offset + c);
    for (uint32_t r = 0; r < rows; ++r) {
      const HVX_Vector sum = Q6_V_vsplat_R(sums[r]);
      HVX_UVector *dst = (HVX_UVector *)(acc + (size_t)r * stride + c);
      /** Registration validates offsets in [-15, 15], so the signed low
       * halfword is exact; the row sum remains a full 32-bit integer. */
      *dst = Q6_Vw_vmpyieacc_VwVwVh(*dst, sum, off);
    }
  }
#endif
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t j = c; j < cols; ++j)
      acc[(size_t)r * stride + j] += sums[r] * offset[j];
}

#endif
