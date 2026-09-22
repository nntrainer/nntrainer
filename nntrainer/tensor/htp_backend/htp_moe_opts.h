// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_moe_opts.h
 * @date   22 Sep 2026
 * @brief  The MoE layer call's option resolver and its profile-row residue,
 *         as pure functions a host check can run (#101, #102)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * C99 and header-only so test/htp/host/moe_opts_host_check.c compiles it
 * with gcc -std=c99 like the other host checks, and htp_compute_ops.cpp
 * includes the same definitions.
 */
#ifndef __HTP_MOE_OPTS_H__
#define __HTP_MOE_OPTS_H__

#include <stdint.h>
#include <stdlib.h>

/** @brief hexkl_mm_u8i4_moe.h's HEXKL_MOE_FLAG_M1_GEMV restated for the ARM
 *  side (the DSP header does not compile there): the moe_set_opts bit that
 *  lets a call of at most 4 rows take the HVX GEMV path. */
#define HTP_MOE_FLAG_M1_GEMV 1u

/**
 * @brief moe_set_opts flags for the value of NNTR_MOE_HTP_M1_GEMV.
 * @param env getenv's result, NULL when the variable is unset
 * @return HTP_MOE_FLAG_M1_GEMV when unset (the default since #101) or set to
 *         a non-zero number; 0 for "0" (the opt-out)
 */
static inline uint32_t htp_moe_opts_flags(const char *env) {
  if (env == NULL)
    return HTP_MOE_FLAG_M1_GEMV;
  return atoi(env) != 0 ? HTP_MOE_FLAG_M1_GEMV : 0u;
}

/**
 * @brief The profile row's rest: the DSP clock minus every named stage.
 * @param dsp per-call DSP time
 * @param named_sum_wo_swiglu per-call sum of every named stage but swiglu
 * @param swiglu per-call swiglu slot
 * @param gemv_row non-zero when any call in the row took the M=1 GEMV path
 * @return the residue
 *
 * On the GEMV path the swiglu slot is the sum of every worker lane's wall
 * time inside the two GEMV stages (hexkl_mm_u8i4_moe.c moe_tail_probe_add),
 * not a stage on the caller's clock, so it is left out of the subtraction;
 * swiglu / mm then reads as the number of lanes busy over mm (#102).
 */
static inline double htp_moe_row_rest_us(double dsp, double named_sum_wo_swiglu,
                                         double swiglu, int gemv_row) {
  return dsp - named_sum_wo_swiglu - (gemv_row ? 0.0 : swiglu);
}

#endif /* __HTP_MOE_OPTS_H__ */
