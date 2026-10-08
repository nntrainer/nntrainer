// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_scalar.h
 * @date   27 Sep 2026
 * @brief  Scalar stand-ins for the HMX tile and the intrinsic-heavy HVX
 *         kernels (u8 quantizer, int32 dequant, u8i4 GEMV, SwiGLU)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * One definition of the arithmetic the host checks and the in-process
 * HTP build (meson -Dhtp-inproc=true, #84) both run in place of the HMX
 * unit and of hvx_quant_u8.c / hvx_dequant_i32.c / hvx_gemm_u8i4_wh.c /
 * hvx_swiglu_f32.c. What this file is not: the device's arithmetic. The
 * HMX and the HVX GEMV compute the same int32 sums (that is checked on
 * the device and, for the GEMV, on libnative), but the u8 activation
 * quantizer's rounding and the int32 -> f32 epilogue order have no scalar
 * specification yet, so those two are this file's formulas, and a bit
 * gate built on them is a regression gate until their _det exists. The
 * SwiGLU is swiglu_det.h, the spec the HVX kernel matches bit for bit.
 *
 * Pulled out of moe_layer_host_check.c unchanged, except that the SwiGLU
 * now runs the spec rather than libm's expf (so a dump is the same bytes
 * on every host) and the pooled workers honour their (n_threads, i) slice
 * (the real pool runs them concurrently).
 */

#ifndef __NNTRAINER_HVX_SCALAR_STANDIN_H__
#define __NNTRAINER_HVX_SCALAR_STANDIN_H__

#include <stddef.h>
#include <stdint.h>

/** @brief One int4 value of a WH weight tile (htp_wh_layout.h's byte
 *  order: byte (k/8)*128 + c*4 + k%4, low nibble for k%8 < 4). The HMX
 *  stand-in, the GEMV stand-in and the checks' references all read a tile
 *  through this one function, so the three cannot disagree on the layout.
 */
int wh_value(const uint8_t *tile, uint32_t k, uint32_t c);

/** @brief The GEMV's sum: out[r][c] over the k_tiles tiles of column nt,
 *  the row-stride-32 tile hvx_gemm_u8i4_wh.h promises. */
void hvx_scalar_gemv(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                     const uint8_t *wh, uint32_t n_col, uint32_t nt,
                     int32_t *out);

/** @brief [plan 229] hvx_scalar_gemv over QS2CX_WH codes (256-byte tiles)
 *  read through @a table (hvx_expand_i2i4_table): the sum the u8i2 GEMV
 *  must equal. */
void hvx_scalar_gemv_i2(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                        const uint8_t *wh, uint32_t n_col, uint32_t nt,
                        const uint8_t *table, int32_t *out);

/**
 * @brief A check's instrumentation of the GEMV entry points. NULL (the
 *        default) runs the plain sum; moe_layer_host_check.c installs its
 *        prefetch ring / lead audit and its feed scoreboard here.
 */
typedef struct {
  /** hvx_gemm_u8i4_wh_prefetch, whole. The plain stand-in is a no-op. */
  void (*prefetch)(const uint8_t *wh, uint32_t n_col, uint32_t nt,
                   uint32_t n_tiles, uint32_t k_tiles);
  /** hvx_gemm_u8i4_wh_col (nopf = 0) and _col_nopf (nopf = 1), whole:
   *  the hook computes @a out itself, normally through hvx_scalar_gemv. */
  void (*gemv)(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
               const uint8_t *wh, uint32_t n_col, uint32_t nt, uint32_t rows1,
               int nopf, int32_t *out);
  /** [plan 229] The same for hvx_gemm_u8i2_wh_col / _col_nopf (and each
   *  column of _cols2_nopf), with the expansion table: the hook computes
   *  @a out itself, normally through hvx_scalar_gemv_i2. */
  void (*gemv2)(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                const uint8_t *wh, uint32_t n_col, uint32_t nt, uint32_t rows1,
                int nopf, const uint8_t *table, int32_t *out);
  /** A buffer the stand-ins below read (@a write 0) or write (1): the
   *  GEMV's activation block, the quantizer's rows and row params, the
   *  pack's output, the dequant's row params and output, the SwiGLU's
   *  output. moe_layer_host_check.c's dataflow scoreboard (#185). */
  void (*buf)(const void *p, size_t bytes, int write);
} hvx_scalar_hooks;

extern hvx_scalar_hooks hvx_scalar_hook;

#endif /* __NNTRAINER_HVX_SCALAR_STANDIN_H__ */
