// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_mm_u8i4.h
 * @date   03 Aug 2026
 * @brief  VTCM layout and HMX orchestration for the A8W4 matmul
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HEXKL_MM_U8I4_H__
#define __NNTRAINER_HEXKL_MM_U8I4_H__

#include <stdint.h>

/**
 * @brief Where each region lives inside the VTCM arena.
 *
 * Byte offsets from @a vtcm_base, not pointers, because that is what the
 * HexKL micro API takes.
 */
typedef struct {
  uint8_t *vtcm_base;  /**< base of the VTCM arena from hexkl_micro_hw_init */
  uint32_t vtcm_size;  /**< total arena size in bytes */
  uint32_t w_base;     /**< all baked weight tiles, (K/32)*(N/32)*512 bytes */
  uint32_t w_size;     /**< byte span of the weight region */
  uint32_t act_base;   /**< activation AH tiles, m_pad*K bytes, 2048-aligned */
  uint32_t act_size;   /**< byte span of the activation region */
  uint32_t result_off; /**< 8192-byte accumulator readout, 2048-aligned */
  uint32_t config_off; /**< HMX config region, 256-aligned */
} hexkl_mm_u8i4_layout;

/**
 * @brief Computes the VTCM partitioning and checks it against the arena.
 *
 * Pure arithmetic apart from reading hexkl_micro_hmx_config_size(), so it
 * is the one piece of this file that could be unit tested on the host.
 *
 * @return AEE_SUCCESS, AEE_EBADPARM on a shape violation, or
 *         AEE_ENOMEMORY when the partitioning does not fit.
 */
int hexkl_mm_u8i4_plan(uint8_t *vtcm_base, uint32_t vtcm_size, uint32_t m_pad,
                       uint32_t k, uint32_t n, hexkl_mm_u8i4_layout *out);

/**
 * @brief Converts every weight tile to WH layout once, into VTCM.
 *
 * HexKL's example calls rm_to_wh_i4 from the innermost matmul loop, which
 * serializes the layout conversion against the multiply. Hoisting it here
 * means the matmul reads the tiles in place with no copy at all.
 *
 * @param[in] w_i4_rm weights as int4 values in int8 containers, range
 *                    [-8, 7], K rows by N columns, row-major
 */
int hexkl_mm_u8i4_bake_weights(const hexkl_mm_u8i4_layout *L,
                               const int8_t *w_i4_rm, uint32_t k, uint32_t n);

/**
 * @brief Runs the HMX matmul over the baked weights and staged activation.
 *
 * Requires hexkl_micro_hmx_setup_acc_read_int32() to have been called for
 * L->config_off, the HMX lock to be held, and the activation to already be
 * in AH layout at L->act_base.
 *
 * @param[out] acc_i32 m_pad by n int32 accumulators, row-major, in DDR
 */
int hexkl_mm_u8i4_run(const hexkl_mm_u8i4_layout *L, uint32_t m_pad, uint32_t k,
                      uint32_t n, int32_t *acc_i32);

#endif /* __NNTRAINER_HEXKL_MM_U8I4_H__ */
