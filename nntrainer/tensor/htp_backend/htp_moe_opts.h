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

/** @brief hexkl_mm_u8i4_moe.h's two tune bits and their fields, restated
 *  for the ARM side (#113). Bit 7 says the DSP takes the l2fetch lead from
 *  bits [15:8], in units of HTP_MOE_GEMV_LEAD_KB_UNIT KB, instead of from
 *  its build default; bit 6 says the same of the row loop in bit 16. They
 *  are separate so that naming one knob leaves the other at the build's
 *  value rather than silently resetting it to zero. */
#define HTP_MOE_FLAG_GEMV_LEAD_SET 0x80u
#define HTP_MOE_FLAG_GEMV_ROWS1_SET 0x40u
#define HTP_MOE_GEMV_LEAD_SHIFT 8u
#define HTP_MOE_GEMV_LEAD_BITS 0xFFu
#define HTP_MOE_GEMV_LEAD_KB_UNIT 64u
#define HTP_MOE_GEMV_LEAD_MAX_UNITS 127u
#define HTP_MOE_FLAG_GEMV_ROWS1 0x10000u

/**
 * @brief The lead field for a requested lead in KB: rounded to the nearest
 *        multiple of HTP_MOE_GEMV_LEAD_KB_UNIT and clamped to
 *        HTP_MOE_GEMV_LEAD_MAX_UNITS.
 *
 * The clamp is the l2fetch width field's 16 bits: a box is at most 127
 * tiles of 512 B (hvx_gemm_u8i4_wh.h), and at the decode shape a stage-A
 * unit is exactly 64 KB, so one field unit is one unit of weight there.
 * A negative or unparsable value reads as 0, which is "every column
 * fetches itself".
 */
static inline uint32_t htp_moe_gemv_lead_units(const char *env) {
  long kb;
  unsigned long u;
  if (env == NULL)
    return 0u;
  kb = atol(env);
  if (kb <= 0)
    return 0u;
  u = ((unsigned long)kb + HTP_MOE_GEMV_LEAD_KB_UNIT / 2u) /
      HTP_MOE_GEMV_LEAD_KB_UNIT;
  if (u > HTP_MOE_GEMV_LEAD_MAX_UNITS)
    u = HTP_MOE_GEMV_LEAD_MAX_UNITS;
  return (uint32_t)u;
}

/**
 * @brief moe_set_opts flags for NNTR_MOE_HTP_M1_GEMV and, since #113, the
 *        two GEMV tuning variables.
 * @param env       getenv("NNTR_MOE_HTP_M1_GEMV"), NULL when unset
 * @param lead_env  getenv("NNTR_MOE_HTP_GEMV_LEAD_KB"), NULL when unset
 * @param rows1_env getenv("NNTR_MOE_HTP_GEMV_ROWS1"), NULL when unset
 * @return HTP_MOE_FLAG_M1_GEMV when @a env is unset (the default since
 *         #101) or set to a non-zero number, 0 for "0" (the opt-out); plus
 *         a tune bit and its field for each tuning variable that is set
 *
 * Each knob is overridden only by its own variable: setting the lead
 * leaves the row loop at the DSP's build default and vice versa, so a
 * sitting that rebuilds the skel with a winning pair cannot have one half
 * of it silently reset to zero by an unrelated export. With neither set
 * the word is exactly what it was before #113.
 */
static inline uint32_t htp_moe_opts_flags(const char *env, const char *lead_env,
                                          const char *rows1_env) {
  uint32_t flags = 0u;
  if (env == NULL || atoi(env) != 0)
    flags = HTP_MOE_FLAG_M1_GEMV;
  if (lead_env != NULL)
    flags |= HTP_MOE_FLAG_GEMV_LEAD_SET |
             (htp_moe_gemv_lead_units(lead_env) << HTP_MOE_GEMV_LEAD_SHIFT);
  if (rows1_env != NULL) {
    flags |= HTP_MOE_FLAG_GEMV_ROWS1_SET;
    if (atoi(rows1_env) != 0)
      flags |= HTP_MOE_FLAG_GEMV_ROWS1;
  }
  return flags;
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
