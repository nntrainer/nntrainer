// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_attn_m1_probe.h
 * @date   29 Sep 2026
 * @brief  Op codes and result words of the debug-only attn_m1_probe entry
 *         shared by the skel (nntr_hvx_attn_m1_probe.c)
 *         and the device gtest (HvxAttnM1Probe.*)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Operations 0--8 run hvx_attn_m1_hf.h on n fp16 lanes, one thread:
 * a, b, c and y all n long, n a multiple of 64, y[i] = f(a[i], b[i], c[i]).
 * Historical synthetic cost operations 16--26 return AEE_EUNSUPPORTED.
 * Free of Hexagon headers so the ARM gtest can include it.
 */

#ifndef __NNTR_ATTN_M1_PROBE_H__
#define __NNTR_ATTN_M1_PROBE_H__

/* semantics: y = ... */
#define ATTN_M1_PROBE_QFMA 0u   /**< hvx_hf_fma(c, a, b) */
#define ATTN_M1_PROBE_ADD 1u    /**< a + b (hf) */
#define ATTN_M1_PROBE_SUB 2u    /**< a - b */
#define ATTN_M1_PROBE_MUL 3u    /**< a * b */
#define ATTN_M1_PROBE_MAX 4u    /**< max(a, b) */
#define ATTN_M1_PROBE_EIGHTH 5u /**< a * 0.125 */
#define ATTN_M1_PROBE_ZPLUS 6u  /**< 0 + a */
#define ATTN_M1_PROBE_EXP16 7u  /**< hvx_hf_exp16(a), a <= 0 */
#define ATTN_M1_PROBE_DIV16 8u  /**< hvx_hf_div16(a, l = b), b in [1, 2048] */
#define ATTN_M1_PROBE_N_SEM 9u

/* prof words (uint32); pairs are lo, hi of a 64-bit count */
#define ATTN_M1_PROBE_W_LANES 0u /**< one thread */
#define ATTN_M1_PROBE_W_REPS 1u  /**< one pass */
#define ATTN_M1_PROBE_W_WALL 2u  /**< elapsed pcycles (2 words) */
/* Preserve the original profile buffer size; words 4--11 are reserved. */
#define ATTN_M1_PROBE_WORDS 12u

#endif /* __NNTR_ATTN_M1_PROBE_H__ */
