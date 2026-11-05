// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_moe_opts.h
 * @date   22 Sep 2026
 * @brief  The MoE layer call's option resolver and its profile-row residue,
 *         as pure functions shared with host checks
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
#include <string.h>

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

/** @brief Weight feed override: bit 5 enables bit 17, where 1 selects DMA
 *  staging into VTCM and 0 selects direct arena reads. Always sent, like
 *  the lead and row-loop overrides above; matches hexkl_mm_u8i4_moe.h. */
#define HTP_MOE_FLAG_GEMV_FEED_SET 0x20u
#define HTP_MOE_FLAG_GEMV_FEED 0x20000u

/** @brief Bit 18 sets src_bypass on arena expert weight descriptors for
 *  both GEMV and HMX, matching hexkl_mm_u8i4_moe.h. Enabled by default;
 *  The runtime can explicitly disable it. */
#define HTP_MOE_FLAG_DMA_BYPASS 0x40000u

/** @brief Source-cache bypass for expert weight DMA. */
static inline uint32_t htp_moe_opts_dma_bypass(unsigned enabled) {
  return enabled ? HTP_MOE_FLAG_DMA_BYPASS : 0u;
}

/** @brief Bits [23:22] encode N - 1 DMA queues for the M=1 weight feed,
 *  matching hexkl_mm_u8i4_moe.h. */
#define HTP_MOE_DMA_Q_SHIFT 22u
#define HTP_MOE_DMA_Q_BITS 3u

/** @brief Encode a validated queue count; reject values outside 1..4. */
static inline int htp_moe_opts_dma_queues(unsigned n, uint32_t *bits) {
  if (n < 1u || n > 4u)
    return -1;
  *bits = (n - 1u) << HTP_MOE_DMA_Q_SHIFT;
  return 0;
}

/** @brief N, the queue count a flags word names: its field + 1. */
static inline uint32_t htp_moe_opts_dma_q(uint32_t flags) {
  return ((flags >> HTP_MOE_DMA_Q_SHIFT) & HTP_MOE_DMA_Q_BITS) + 1u;
}

/** @brief Bit 19 selects gelu_tanh(gate) * up in GEMV and HMX, matching
 *  HEXKL_MOE_FLAG_GELU_TANH. Prefill supplies the activation per call;
 *  resident MOE and DENSE_FFN ops read the session's set_moe_geglu value. */
#define HTP_MOE_FLAG_GEGLU 0x80000u

/** Opt-in online rotations, matching hexkl_mm_u8i4_moe.h. These rotate
 * activations only: ordinary checkpoints are useful for timing, but their
 * outputs are not valid until weights have the matching offline rotation.
 * The app defaults to 2-bit experts; the shared FFN stays unchanged. */
#define HTP_MOE_FLAG_HAD_UP 0x04000000u
#define HTP_MOE_FLAG_HAD_DOWN 0x08000000u
#define HTP_MOE_FLAG_HAD_SCALAR 0x10000000u
#define HTP_MOE_FLAG_HAD_HMX 0x20000000u
#define HTP_MOE_FLAG_HAD_2BIT_ONLY 0x40000000u
#define HTP_MOE_FLAG_HAD_PALEY_I 0x80000000u
#define HTP_MOE_FLAG_HAD_MASK 0xfc000000u

/** moe.hadamard: unset/off, up, down, both, both_scalar,
 * up_hmx, down_hmx, both_hmx, both_paley. HMX uses full-width FP16 on prefill;
 * calls of at most four tokens and HVX expert tails keep FP32.
 * Invalid values leave the output untouched and fail instead of silently
 * benchmarking the baseline. */
static inline int htp_moe_opts_hadamard(const char *env, uint32_t *bits) {
  if (env == NULL || strcmp(env, "off") == 0)
    *bits = 0u;
  else if (strcmp(env, "up") == 0)
    *bits = HTP_MOE_FLAG_HAD_UP;
  else if (strcmp(env, "down") == 0)
    *bits = HTP_MOE_FLAG_HAD_DOWN;
  else if (strcmp(env, "both") == 0)
    *bits = HTP_MOE_FLAG_HAD_UP | HTP_MOE_FLAG_HAD_DOWN;
  else if (strcmp(env, "both_paley") == 0)
    *bits =
      HTP_MOE_FLAG_HAD_UP | HTP_MOE_FLAG_HAD_DOWN | HTP_MOE_FLAG_HAD_PALEY_I;
  else if (strcmp(env, "both_scalar") == 0)
    *bits =
      HTP_MOE_FLAG_HAD_UP | HTP_MOE_FLAG_HAD_DOWN | HTP_MOE_FLAG_HAD_SCALAR;
  else if (strcmp(env, "up_hmx") == 0)
    *bits = HTP_MOE_FLAG_HAD_UP | HTP_MOE_FLAG_HAD_HMX;
  else if (strcmp(env, "down_hmx") == 0)
    *bits = HTP_MOE_FLAG_HAD_DOWN | HTP_MOE_FLAG_HAD_HMX;
  else if (strcmp(env, "both_hmx") == 0)
    *bits = HTP_MOE_FLAG_HAD_UP | HTP_MOE_FLAG_HAD_DOWN | HTP_MOE_FLAG_HAD_HMX;
  else
    return -1;
  return 0;
}

/** moe.hadamard_scope: default/2bit rotates only 2-bit pairs.
 * all is for kernel experiments; every active width must then be supported.
 * The Gemma shared FFN uses unsupported 1056-wide chunks. */
static inline int htp_moe_opts_hadamard_scope(const char *env, uint32_t *bits) {
  if (env == NULL || strcmp(env, "2bit") == 0)
    *bits = HTP_MOE_FLAG_HAD_2BIT_ONLY;
  else if (strcmp(env, "all") == 0)
    *bits = 0u;
  else
    return -1;
  return 0;
}

/** @brief The bits of moe_set_opts' echo that must equal what was sent.
 *  With the GEMV on, all of them. With it off, the tune bits mean nothing
 *  on the HMX loop, so only bit 0 -- plus the bypass bit when it was asked
 *  for, because the HMX loop's weight DMA honours it (#158): a skel that
 *  predates it must fail the run, not measure the default. Likewise the
 *  GeGLU bit: a skel without it would run SwiGLU on a Gemma expert. */
static inline uint32_t htp_moe_opts_must_match(uint32_t flags) {
  if ((flags & HTP_MOE_FLAG_M1_GEMV) != 0u)
    return ~0u;
  return HTP_MOE_FLAG_M1_GEMV |
         (flags & (HTP_MOE_FLAG_DMA_BYPASS | HTP_MOE_FLAG_GEGLU |
                   HTP_MOE_FLAG_HAD_MASK));
}

/** @brief Round the lead to 64 KiB units, bounded by the DMA field. */
static inline uint32_t htp_moe_gemv_lead_units(unsigned kb) {
  const uint64_t units =
    ((uint64_t)kb + HTP_MOE_GEMV_LEAD_KB_UNIT / 2u) / HTP_MOE_GEMV_LEAD_KB_UNIT;
  return units > HTP_MOE_GEMV_LEAD_MAX_UNITS ? HTP_MOE_GEMV_LEAD_MAX_UNITS
                                             : (uint32_t)units;
}

/** @brief Encode explicit GEMV settings for the DSP session. */
static inline uint32_t htp_moe_opts_flags(unsigned enabled, unsigned lead_kb,
                                          unsigned rows1, unsigned feed) {
  uint32_t flags = HTP_MOE_FLAG_GEMV_LEAD_SET | HTP_MOE_FLAG_GEMV_ROWS1_SET |
                   HTP_MOE_FLAG_GEMV_FEED_SET;
  if (enabled)
    flags |= HTP_MOE_FLAG_M1_GEMV;
  flags |= htp_moe_gemv_lead_units(lead_kb) << HTP_MOE_GEMV_LEAD_SHIFT;
  if (rows1)
    flags |= HTP_MOE_FLAG_GEMV_ROWS1;
  if (feed)
    flags |= HTP_MOE_FLAG_GEMV_FEED;
  return flags;
}

/** @brief The banner's name for the feed cell a flags word asks for:
 *  "vtcm" or "arena". */
static inline const char *htp_moe_opts_feed_name(uint32_t flags) {
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
