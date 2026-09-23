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

/** @brief hexkl_mm_u8i4_moe.h's third tune bit and its field (#117): bit 5
 *  says the DSP takes the weight feed from bit 17 (1 = each expert's
 *  weights staged into VTCM by DMA, 0 = read from the arena) instead of
 *  from its build default. Unlike the pair above this one is sent only
 *  when NNTR_MOE_HTP_GEMV_FEED is set, so an unset run keeps the skel's
 *  build default and the banner reads feed=default (0x103c1 stays A's
 *  word); FEED=1 on top of D192 prints applied=0x303e1, FEED=0 0x103e1. */
#define HTP_MOE_FLAG_GEMV_FEED_SET 0x20u
#define HTP_MOE_FLAG_GEMV_FEED 0x20000u

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

/** @brief The (loop, lead) pair an unset run asks for: #113's D192, the
 *  same values as hexkl_mm_u8i4_moe.h's HVX_GEMV_M1_ROWS1 and
 *  HVX_GEMV_PF_LEAD_KB. Restated here because the DSP header does not
 *  compile on the ARM side; keep the two in step. Sent explicitly rather
 *  than left to the skel's default so that every log's echo names its
 *  cell (LEDGER rule 21): an unset run prints applied=0x103c1, and
 *  NNTR_MOE_HTP_GEMV_LEAD_KB=0 NNTR_MOE_HTP_GEMV_ROWS1=0 (the old
 *  four-row, no-lead cell) prints applied=0xc1. */
#define HTP_MOE_GEMV_LEAD_KB_DEFAULT 192u
#define HTP_MOE_GEMV_ROWS1_DEFAULT 1u

/**
 * @brief moe_set_opts flags for NNTR_MOE_HTP_M1_GEMV and, since #113, the
 *        GEMV tuning variables.
 * @param env       getenv("NNTR_MOE_HTP_M1_GEMV"), NULL when unset
 * @param lead_env  getenv("NNTR_MOE_HTP_GEMV_LEAD_KB"), NULL when unset
 * @param rows1_env getenv("NNTR_MOE_HTP_GEMV_ROWS1"), NULL when unset
 * @param feed_env  getenv("NNTR_MOE_HTP_GEMV_FEED"), NULL when unset (#117)
 * @return HTP_MOE_FLAG_M1_GEMV when @a env is unset (the default since
 *         #101) or set to a non-zero number, 0 for "0" (the opt-out); plus
 *         both (loop, lead) tune bits, each field from its own variable
 *         when set and from the *_DEFAULT above when not; plus the feed
 *         tune bit and its field only when @a feed_env is set
 *
 * Each knob is overridden only by its own variable: setting the lead
 * leaves the row loop at the default and vice versa, so naming one knob
 * cannot silently reset the other to zero.
 */
static inline uint32_t htp_moe_opts_flags(const char *env, const char *lead_env,
                                          const char *rows1_env,
                                          const char *feed_env) {
  uint32_t flags = HTP_MOE_FLAG_GEMV_LEAD_SET | HTP_MOE_FLAG_GEMV_ROWS1_SET;
  if (env == NULL || atoi(env) != 0)
    flags |= HTP_MOE_FLAG_M1_GEMV;
  flags |= (lead_env != NULL
              ? htp_moe_gemv_lead_units(lead_env)
              : HTP_MOE_GEMV_LEAD_KB_DEFAULT / HTP_MOE_GEMV_LEAD_KB_UNIT)
           << HTP_MOE_GEMV_LEAD_SHIFT;
  if (rows1_env != NULL ? atoi(rows1_env) != 0
                        : HTP_MOE_GEMV_ROWS1_DEFAULT != 0u)
    flags |= HTP_MOE_FLAG_GEMV_ROWS1;
  if (feed_env != NULL) {
    flags |= HTP_MOE_FLAG_GEMV_FEED_SET;
    if (atoi(feed_env) != 0)
      flags |= HTP_MOE_FLAG_GEMV_FEED;
  }
  return flags;
}

/** @brief The banner's name for the feed cell a flags word asks for:
 *  "default" (the skel's build value), "vtcm" or "arena". */
static inline const char *htp_moe_opts_feed_name(uint32_t flags) {
  if ((flags & HTP_MOE_FLAG_GEMV_FEED_SET) == 0u)
    return "default";
  return (flags & HTP_MOE_FLAG_GEMV_FEED) != 0u ? "vtcm" : "arena";
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
