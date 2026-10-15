// SPDX-License-Identifier: Apache-2.0
#ifndef NNTR_HEXKL_WH_AFFINE_H
#define NNTR_HEXKL_WH_AFFINE_H

#include <stddef.h>
#include <stdint.h>

/** Sum the *same* quantized activation consumed by HMX/HVX, with its zero
 * point removed. AH is a block of 64 rows, K/32 tiles of 2048 bytes.
 * For W=s*(q+offset), add offset*sum(u-zp) to the signed-int4 accumulator
 * before ordinary dequantization. No second activation quantizer is used. */
static inline void hexkl_wh_affine_sums(const uint8_t *ah, uint32_t rows,
                                        uint32_t K, const int32_t *zp,
                                        int32_t *sums) {
  for (uint32_t r = 0; r < rows; ++r) {
    int32_t sum = -(int32_t)K * zp[r];
    for (uint32_t kt = 0; kt < K / 32u; ++kt)
      for (uint32_t k = 0; k < 32u; ++k)
        sum += ah[(size_t)kt * 2048u + r * 32u + k];
    sums[r] = sum;
  }
}

static inline void hexkl_wh_affine_acc(int32_t *acc, uint32_t stride,
                                       uint32_t rows, uint32_t cols,
                                       const int32_t *offset,
                                       const int32_t *sums) {
  if (offset)
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < cols; ++c)
        acc[(size_t)r * stride + c] += sums[r] * offset[c];
}

#endif
