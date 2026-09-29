// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_attn_m1_probe.h
 * @date   29 Sep 2026
 * @brief  Op codes and result words of the debug-only attn_m1_probe entry
 *         (plan 170 step 1): shared by the skel (nntr_hvx_attn_m1_probe.c)
 *         and the device gtest (HvxAttnM1Probe.*)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * SEMANTICS ops (< 16) run hvx_attn_m1_hf.h on n fp16 lanes, one thread:
 * a, b, c and y all n long, n a multiple of 64, y[i] = f(a[i], b[i], c[i]).
 * COST ops (16 ..) run a kernel-shaped loop on synthetic, L2-resident data
 * (the FETCH pair and the COLD ops: a cold slab) on `lanes` pool lanes,
 * `reps` pool runs,
 * each timed; a, b, c and y are empty and prof holds the words below.
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

/* cost: one pool run per rep, per lane (ATTN_M1_PROBE_L positions) */
#define ATTN_M1_PROBE_FMA16_SF 16u  /**< today's score loop: hvx_fma16_sf */
#define ATTN_M1_PROBE_SCORES1 17u   /**< hf scores, tiled Kt, one q head */
#define ATTN_M1_PROBE_SCORES2 18u   /**< the same, two q heads per Kt load */
#define ATTN_M1_PROBE_PV4 19u       /**< hf PV, four q heads per V row */
#define ATTN_M1_PROBE_FETCH 20u     /**< stream a cold slab, no l2fetch */
#define ATTN_M1_PROBE_FETCH_L2F 21u /**< the same with an l2fetch lead */
/* #170 round 2: the kernel's leads on cold data (a slab per lane, evicted
   between reps) and the splat source, each against its warm op above */
#define ATTN_M1_PROBE_PV4_COLD 22u         /**< PV4 over cold V rows, no lead */
#define ATTN_M1_PROBE_PV4_COLD_L2F 23u     /**< the same, 32 KiB window lead */
#define ATTN_M1_PROBE_SCORES1_COLD 24u     /**< SCORES1 over cold tiles */
#define ATTN_M1_PROBE_SCORES1_COLD_L2F 25u /**< the same, next-tile lead */
#define ATTN_M1_PROBE_SCORES1_SPLAT 26u /**< SCORES1, q splat per scalar load  \
                                         */
#define ATTN_M1_PROBE_COST_END 27u

/** @brief Positions per lane and rep of the compute cost ops. */
#define ATTN_M1_PROBE_L 1024u
/** @brief The FETCH slab: 3 MiB of fp16, the KV of one layer at L 1536,
 *         split evenly over the lanes. */
#define ATTN_M1_PROBE_SLAB_BYTES (3u << 20)
/** @brief The cold ops' slab per lane: ATTN_M1_PROBE_L V rows, or
 *         ATTN_M1_PROBE_L / 64 Kt tiles (128 KiB). */
#define ATTN_M1_PROBE_COLD_BYTES (128u << 10)
#define ATTN_M1_PROBE_MAX_LANES 8u
#define ATTN_M1_PROBE_MAX_REPS 10000u

/* prof words (uint32); pairs are lo, hi of a 64-bit count */
#define ATTN_M1_PROBE_W_LANES 0u /**< lanes that ran (the pool's n) */
#define ATTN_M1_PROBE_W_REPS 1u
#define ATTN_M1_PROBE_W_WALL 2u     /**< pcycles of the pool runs (2 words) */
#define ATTN_M1_PROBE_W_QT 4u       /**< qtimer ticks of the same (2 words) */
#define ATTN_M1_PROBE_W_BUSY_MAX 6u /**< max over lanes of its pcycles */
#define ATTN_M1_PROBE_W_BUSY_SUM 7u /**< sum over lanes (2 words) */
#define ATTN_M1_PROBE_W_FMA64 9u    /**< 64-lane FMAs per lane and rep */
#define ATTN_M1_PROBE_W_BYTES 10u   /**< bytes read per lane and rep */
#define ATTN_M1_PROBE_W_SINK 11u    /**< the loops' results folded (anti-DCE) */
#define ATTN_M1_PROBE_WORDS 12u

#endif /* __NNTR_ATTN_M1_PROBE_H__ */
