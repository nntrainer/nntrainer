// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_q4_gemv_f32.h
 * @date   29 Sep 2026
 * @brief  The Android CPU's M=1 Q4_0 GEMV on the DSP, bit for bit
 *         (q4_gemv_cpu_det.h), over the Q4M1 weight layout (#132 PR 2)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Each output column is the CPU's chain acc = fma(isum_b, s_b, acc) over
 * the 32-blocks in order. Three ways to run it, all equal to the spec:
 *
 *  HVX_Q4M1_NATIVE / HVX_Q4M1_INTRIN: 32 columns per vector. The integer
 *    dot is vrmpy; the product isum * d_w * d_a is split exactly into two
 *    f32 (P1 + P2), and RN(acc + P1 + P2) is Boldo-Melquiond's emulated
 *    FMA: Fast2Sum, TwoSum, a round-to-odd add, one final RN add. NATIVE
 *    issues the IEEE vadd / vsub / vmpy .sf instructions (-mhvx-ieee-fp);
 *    INTRIN the Q6_Vsf_* intrinsics, which the compiler lowers to a qf32
 *    op and a conversion (the host emulation's form; the silicon fallback).
 *  HVX_Q4M1_SFFMA: HVX makes isum and s (both exact) for all blocks of a
 *    group, then the scalar core runs the chains with sffma, @a cif
 *    columns in flight (8, 16 or 32 independent chains).
 *
 * No DSP state: the caller owns the weight (128-byte aligned, the Q4M1
 * bytes), the prepared activation and the scratch.
 */

#ifndef __NNTRAINER_HVX_Q4_GEMV_F32_H__
#define __NNTRAINER_HVX_Q4_GEMV_F32_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Kernel variants (the fc_q4m1_f32 entry's variant, low byte). */
enum {
  HVX_Q4M1_NATIVE = 0,
  HVX_Q4M1_INTRIN = 1,
  HVX_Q4M1_SFFMA = 2,
};

/** @brief One quantized activation row, from hvx_q4m1_prep. Arrays of K
 *         (q) and K / 32 (the rest) elements, owned by the caller. */
typedef struct {
  int8_t *q;   /**< the Q8_0 quants, 4-byte aligned */
  int32_t *s8; /**< per block: -8 * sum of its quants */
  int32_t *ma; /**< per block: d_a's integer mantissa, 0..2047 */
  int32_t *ea; /**< per block: max(e_a, 1) - 50 + 127, the P2 exponent part */
  float *df;   /**< per block: d_a as f32 (exact) */
  uint16_t *d; /**< per block: d_a, the f16 the CPU stores */
} hvx_q4m1_act;

/** @brief q8_0_quant_cpu_det of x (K % 64 == 0) plus the per-block terms
 *         the kernels read. Scalar, exact integer helpers. */
void hvx_q4m1_prep(const float *x, uint32_t K, hvx_q4m1_act *a);

/**
 * @brief @a ngroups consecutive groups of 32 columns: y[32 i + l] for group
 *        i of @a w (Q4M1, 128-byte aligned, K / 64 units of 1152 bytes
 *        per group).
 *
 * @param variant HVX_Q4M1_*
 * @param cif     columns in flight for HVX_Q4M1_SFFMA: 8, 16 or 32
 * @param fs      HVX_Q4M1_SFFMA scratch: 2 K floats, 128-byte aligned
 */
void hvx_q4m1_gemv_groups(const uint8_t *w, uint32_t K, uint32_t ngroups,
                          const hvx_q4m1_act *a, float *y, uint32_t variant,
                          uint32_t cif, float *fs);

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HVX_Q4_GEMV_F32_H__ */
