// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_gemm_u8i4_wh.c
 * @date   16 Sep 2026
 * @brief  HVX u8 x i4 matmul over HMX-layout tiles, bit-identical to HMX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Why this exists: the HMX unit computes a 64-row block whatever the row
 * count, so an expert's tail block of 10-20 rows costs the full 252 us
 * (doc 47 section 14, O1). Those rows are cheap on HVX, and the HVX is
 * idle for most of the HMX issue, so the tail rides on the worker pool's
 * background lane and the HMX skips the block. What makes it exact rather
 * than approximately equal is that both units do integer sums of the same
 * products; what makes it cheap is the WH layout (see the header).
 */

#include "hvx_gemm_u8i4_wh.h"

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define WH_TILE_BYTES 512u
#define AH_TILE_BYTES 2048u
#define AH_ROW_BYTES 32u

/**
 * @brief Prefetches the n-tile column's k_tiles tiles into L2: 512 bytes
 *        every n_col*512, k_tiles times -- one 2D l2fetch, so the loads
 *        below find the weight in L2 instead of paying DDR latency 64
 *        times over. A hint only; it changes no result.
 */
static inline void prefetch_col(const uint8_t *col, uint32_t stride,
                                uint32_t k_tiles) {
#if defined(__hexagon__)
  /* Rtt: [47:32] stride, [31:16] width, [15:0] height. Both strides this
     is called with (112*512, 64*512) fit 16 bits. */
  const uint64_t cfg = ((uint64_t)stride << 32) |
                       ((uint64_t)WH_TILE_BYTES << 16) | (uint64_t)k_tiles;
  Q6_l2fetch_AP((void *)col, cfg);
#else
  (void)col;
  (void)stride;
  (void)k_tiles;
#endif
}

/**
 * @brief The four rows [r0, r0+4) of one column, all k-tiles, accumulators
 *        in registers.
 *
 * Per weight quarter-tile vector v (rows 8g..8g+7 of the tile), the low
 * nibbles are rows 8g..8g+3 and the high nibbles rows 8g+4..8g+7, each as a
 * signed 4-bit value. Shifting the low nibble up by 4 and masking gives a
 * signed byte equal to 16*w; masking the high nibble in place gives 16*w
 * for the other four rows. The activation words that go with them are
 * bytes 8g..8g+3 and 8g+4..8g+7 of the row, splatted. vrmpy's u8 x i8 dot
 * over those four bytes into lane c is then 16 times the true partial sum
 * for column c, exact in int32 (|sum| < 255*8*32*16*k_tiles), and one
 * arithmetic shift at the end undoes the 16.
 */
static inline void gemm_rows4(const uint8_t *act_ah, uint32_t r0, uint32_t rows,
                              uint32_t k_tiles, const uint8_t *col,
                              uint32_t stride, int32_t *out) {
  const HVX_Vector mask = Q6_V_vsplat_R((int)0xF0F0F0F0u);
  HVX_Vector acc0 = Q6_V_vzero(), acc1 = Q6_V_vzero(), acc2 = Q6_V_vzero(),
             acc3 = Q6_V_vzero();
  /* Rows past the count read row r0's bytes and are never stored: the
     block's AH tile has 64 rows, so the reads are in range. */
  const uint32_t r1 = (rows > 1u) ? r0 + 1u : r0;
  const uint32_t r2 = (rows > 2u) ? r0 + 2u : r0;
  const uint32_t r3 = (rows > 3u) ? r0 + 3u : r0;

  for (uint32_t kt = 0; kt < k_tiles; ++kt) {
    const uint8_t *tile = col + (size_t)kt * stride;
    const uint8_t *arow = act_ah + (size_t)kt * AH_TILE_BYTES;
    const uint32_t *a0 = (const uint32_t *)(arow + r0 * AH_ROW_BYTES);
    const uint32_t *a1 = (const uint32_t *)(arow + r1 * AH_ROW_BYTES);
    const uint32_t *a2 = (const uint32_t *)(arow + r2 * AH_ROW_BYTES);
    const uint32_t *a3 = (const uint32_t *)(arow + r3 * AH_ROW_BYTES);
    for (uint32_t g = 0; g < 4u; ++g) {
      const HVX_Vector v = *(const HVX_UVector *)(tile + g * 128u);
      const HVX_Vector wlo = Q6_V_vand_VV(Q6_Vh_vasl_VhR(v, 4), mask);
      const HVX_Vector whi = Q6_V_vand_VV(v, mask);
      acc0 = Q6_Vw_vrmpyacc_VwVubVb(acc0, Q6_V_vsplat_R((int)a0[2u * g]), wlo);
      acc0 =
        Q6_Vw_vrmpyacc_VwVubVb(acc0, Q6_V_vsplat_R((int)a0[2u * g + 1u]), whi);
      acc1 = Q6_Vw_vrmpyacc_VwVubVb(acc1, Q6_V_vsplat_R((int)a1[2u * g]), wlo);
      acc1 =
        Q6_Vw_vrmpyacc_VwVubVb(acc1, Q6_V_vsplat_R((int)a1[2u * g + 1u]), whi);
      acc2 = Q6_Vw_vrmpyacc_VwVubVb(acc2, Q6_V_vsplat_R((int)a2[2u * g]), wlo);
      acc2 =
        Q6_Vw_vrmpyacc_VwVubVb(acc2, Q6_V_vsplat_R((int)a2[2u * g + 1u]), whi);
      acc3 = Q6_Vw_vrmpyacc_VwVubVb(acc3, Q6_V_vsplat_R((int)a3[2u * g]), wlo);
      acc3 =
        Q6_Vw_vrmpyacc_VwVubVb(acc3, Q6_V_vsplat_R((int)a3[2u * g + 1u]), whi);
    }
  }
  *(HVX_UVector *)(out + (size_t)r0 * 32u) = Q6_Vw_vasr_VwR(acc0, 4);
  if (rows > 1u) {
    *(HVX_UVector *)(out + (size_t)(r0 + 1u) * 32u) = Q6_Vw_vasr_VwR(acc1, 4);
  }
  if (rows > 2u) {
    *(HVX_UVector *)(out + (size_t)(r0 + 2u) * 32u) = Q6_Vw_vasr_VwR(acc2, 4);
  }
  if (rows > 3u) {
    *(HVX_UVector *)(out + (size_t)(r0 + 3u) * 32u) = Q6_Vw_vasr_VwR(acc3, 4);
  }
}

void hvx_gemm_u8i4_wh_col(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                          const uint8_t *wh, uint32_t n_col, uint32_t nt,
                          int32_t *out) {
  const uint32_t stride = n_col * WH_TILE_BYTES;
  const uint8_t *col = wh + (size_t)nt * WH_TILE_BYTES;
  if (m > HVX_GEMM_U8I4_MAX_ROWS) {
    m = HVX_GEMM_U8I4_MAX_ROWS;
  }
  prefetch_col(col, stride, k_tiles);
  for (uint32_t r0 = 0; r0 < m; r0 += 4u) {
    gemm_rows4(act_ah, r0, m - r0, k_tiles, col, stride, out);
  }
}
