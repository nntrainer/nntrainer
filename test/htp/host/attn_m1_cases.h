// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   attn_m1_cases.h
 * @date   29 Sep 2026
 * @brief  The m=1 attention test inputs shared by the host check and the
 *         two device gtests: fixed-kind rows in the fp16 domain and the
 *         midpoint-adversarial rows that make the fused FMA observable
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * One file so test/htp/host/attn_m1_host_check.c (C, hvx_emu),
 * unittest_hvx_attn.cpp (HvxAttnM1.*, the DSP) and
 * unittest_nntrainer_cpu_backend_fp16.cpp (AttnM1F16Det.*, the ARM CPU)
 * feed the same rows to their comparisons.
 *
 * WHY ADVERSARIAL ROWS. fma16(c, a, b) differs from rounding c + a*b to
 * f32 first only when that f32 sum sits exactly on an fp16 midpoint and is
 * inexact: 42 in 10^7 random triples. Random rows therefore cannot tell a
 * fused FMA from a double-rounded one. The two constructions below hit the
 * case on purpose:
 *  - SCORES. q head n_q - 1 is 1 at d = 0 and A_MINUS at d = 8, zero
 *    elsewhere; q head n_q - 2 is 1 at d = 1 and A_PLUS at d = 9. At every
 *    position p >= 3 of their kv head, k[0] / k[1] = c (fp16, odd / even
 *    mantissa, |c| in [0.5, 4)) and k[8] / k[9] = b with a*b = h * (1 -
 *    25 * 2^-20) (inside the midpoint c +- h, h = half the fp16 ulp at c)
 *    or h * (1 + 2^-15) (just outside it). Accumulator lane 0 / 1 then
 *    computes fma16(c, a, b) on exactly such a sum: every position is a
 *    case (the other lanes stay 0, so the tree passes it through).
 *  - PV. q head 3 is "soft" (|q| <= 1/16), so its probabilities are
 *    O(1/L) with a full fp16 mantissa. The spec gives them (V does not
 *    enter the softmax); then, per position j in 3 .. 3 + ADV_PV_POS and
 *    per d, v of its kv head is replaced by one that puts fma16(o, p_j, v)
 *    on a midpoint for the chain's o at that point. The candidates are
 *    found at o in [1, 2) (only o's binade and parity matter there) and
 *    scaled to o's binade; each is re-checked at the real o, and v stays
 *    random where none fits. A uniform p = rne16(1/L) would not do: at
 *    L = 64 or 1024 it is a power of two and p * v is exact.
 *
 * DOMAIN. Everything stays inside attn_m1_det.h's fp16 domain: random
 * values in [-4, 4], the "large" kind +-16 (a score of at most 16 * 16 * 64
 * = 16384 < 65504), the "subnormal" kind fp16 subnormals.
 */

#ifndef __NNTRAINER_ATTN_M1_CASES_H__
#define __NNTRAINER_ATTN_M1_CASES_H__

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "attn_m1_det.h"

/** @brief a*b = h * (1 - 25 * 2^-20) with b = B_MINUS * 2^(E-5): 1029 * 1019 */
#define AMC_A_MINUS (1029.0f / 65536.0f)
#define AMC_B_MINUS (1019.0f / 1024.0f)
/** @brief a*b = h * (1 + 2^-15) with b = B_PLUS * 2^(E-5): 1056 * 993 */
#define AMC_A_PLUS (1056.0f / 65536.0f)
#define AMC_B_PLUS (993.0f / 1024.0f)
/** @brief PV positions (from 3) that may take an adversarial v. */
#define AMC_ADV_PV_POS 61u

/** @brief The generator: an LCG, so C and C++ see the same stream. */
typedef struct {
  uint32_t s;
} amc_rng;

static inline uint32_t amc_next(amc_rng *r) {
  r->s = r->s * 1664525u + 1013904223u;
  return r->s;
}

/** @brief A float uniform in [lo, hi), not rounded. */
static inline float amc_frand(amc_rng *r, float lo, float hi) {
  return lo + (hi - lo) * ((float)(amc_next(r) >> 8) / 16777216.0f);
}

/**
 * @brief One row of @a n values of a kind: 1 zeros; 2 fp16 subnormals
 *        (+-k * 2^-24, k = 1..1023); 3 large, +-16; else random in [-4, 4].
 *        Not rounded to fp16: the kernel and the CPU round on entry.
 */
static inline void amc_fill_row(amc_rng *r, float *x, uint32_t n, int kind) {
  for (uint32_t i = 0; i < n; ++i) {
    switch (kind) {
    case 1:
      x[i] = 0.0f;
      break;
    case 2:
      x[i] = ldexpf((float)(1u + amc_next(r) % 1023u), -24) *
             ((i & 1u) ? -1.0f : 1.0f);
      break;
    case 3:
      x[i] = amc_frand(r, -16.0f, 16.0f);
      break;
    default:
      x[i] = amc_frand(r, -4.0f, 4.0f);
    }
  }
}

/** @brief The soft q head of the PV construction. */
#define AMC_PV_HEAD 3u

/** @brief q [n_q][hd]: heads 0..2 the fixed kinds (zero, subnormal, large),
 *         head 3 soft (|q| <= 1/16, the PV construction's), the last two
 *         the adversarial score heads when n_q >= 6. */
static inline void amc_fill_q(amc_rng *r, float *q, uint32_t n_q, uint32_t hd) {
  for (uint32_t h = 0; h < n_q; ++h) {
    amc_fill_row(r, q + (size_t)h * hd, hd, (h < 3u) ? (int)h + 1 : 0);
    if (h == AMC_PV_HEAD) {
      for (uint32_t d = 0; d < hd; ++d) {
        q[(size_t)h * hd + d] *= 1.0f / 64.0f;
      }
    }
  }
  if (n_q >= 6u) {
    float *qm = q + (size_t)(n_q - 1u) * hd, *qp = q + (size_t)(n_q - 2u) * hd;
    memset(qm, 0, hd * sizeof(float));
    memset(qp, 0, hd * sizeof(float));
    qm[0] = 1.0f;
    qm[8] = AMC_A_MINUS;
    qp[1] = 1.0f;
    qp[9] = AMC_A_PLUS;
  }
}

/** @brief L rows of k and v, [L][n_kv][hd]: position 0 zero k and v, 1
 *         subnormal k and v, 2 a large k; then the adversarial score rows
 *         (see the file comment) when n_q >= 6. */
static inline void amc_fill_kv(amc_rng *r, float *k, float *v, uint32_t L,
                               uint32_t n_kv, uint32_t gqa, uint32_t hd) {
  const uint32_t row = n_kv * hd, n_q = n_kv * gqa;
  for (uint32_t p = 0; p < L; ++p) {
    const int kind = (p == 0u) ? 1 : (p == 1u) ? 2 : 0;
    amc_fill_row(r, k + (size_t)p * row, row, (p == 2u) ? 3 : kind);
    amc_fill_row(r, v + (size_t)p * row, row, kind);
  }
  if (n_q < 6u) {
    return;
  }
  /* The kv heads of q heads n_q - 1 (lane 0) and n_q - 2 (lane 1). */
  for (uint32_t lane = 0; lane < 2u; ++lane) {
    const uint32_t h = (n_q - 1u - lane) / gqa;
    for (uint32_t p = 3; p < L; ++p) {
      float *kr = k + (size_t)p * row + (size_t)h * hd;
      const int e = (int)(amc_next(r) % 3u) - 1;      /* c in [2^e, 2^(e+1)) */
      uint32_t mant = 1u + amc_next(r) % 1023u;       /* 1..1023, not 0 */
      mant = lane == 0u ? (mant | 1u) : (mant & ~1u); /* odd / even */
      if (mant == 0u) {
        mant = 2u;
      }
      const float c = ldexpf(1.0f + (float)mant / 1024.0f, e) *
                      ((amc_next(r) & 1u) ? -1.0f : 1.0f);
      const float b = ldexpf(lane == 0u ? AMC_B_MINUS : AMC_B_PLUS, e - 5) *
                      ((amc_next(r) & 1u) ? -1.0f : 1.0f);
      kr[lane] = c;
      kr[8u + lane] = b;
    }
  }
}

/** @brief 1 when fma16(c, a, b) differs from rounding c + a*b to f32 first
 *         (the case the adversarial rows are for). */
static inline int amc_is_midpoint_case(float c, float a, float b) {
  const float fused = attn_m1_det_fma16(c, a, b);
  const float twice = attn_m1_det_rne16(attn_m1_det_add(
    c, attn_m1_det_mul(attn_m1_det_rne16(a), attn_m1_det_rne16(b))));
  return attn_m1_det_bits(fused) != attn_m1_det_bits(twice);
}

/**
 * @brief Rewrites v of the soft q head's kv head so that head meets
 *        midpoint cases in its PV chain (the file comment's PV rows).
 *        q and k must be final; the spec supplies the probabilities.
 * @return the number of (position, d) cases planted
 */
static inline uint32_t amc_plant_pv(const float *q, const float *k, float *v,
                                    uint32_t L, uint32_t n_kv, uint32_t gqa,
                                    uint32_t hd) {
  enum { MAX_HITS = 16 };
  const uint32_t n_q = n_kv * gqa, row = n_kv * hd;
  uint32_t planted = 0;
  if (n_q <= AMC_PV_HEAD || L < 4u || hd > 128u) {
    return 0u;
  }
  const uint32_t h = AMC_PV_HEAD / gqa;
  const uint32_t last = L < 3u + AMC_ADV_PV_POS ? L : 3u + AMC_ADV_PV_POS;
  float *kt = (float *)calloc((size_t)hd * L, sizeof(float));
  float *vz = (float *)calloc((size_t)L * hd, sizeof(float));
  float *pr = (float *)malloc((size_t)L * sizeof(float));
  float *hits = (float *)malloc((size_t)last * MAX_HITS * sizeof(float));
  uint32_t *n_hits = (uint32_t *)calloc(last, sizeof(uint32_t));
  float out[128];
  if (!kt || !vz || !pr || !hits || !n_hits) {
    free(kt), free(vz), free(pr), free(hits), free(n_hits);
    return 0u;
  }
  for (uint32_t p = 0; p < L; ++p) {
    for (uint32_t d = 0; d < hd; ++d) {
      kt[(size_t)d * L + p] =
        attn_m1_det_rne16(k[(size_t)p * row + h * hd + d]);
    }
  }
  attn_m1_det_head(q + (size_t)AMC_PV_HEAD * hd, kt, vz, hd, L, L, 0.125f, pr,
                   out, NULL, NULL);
  /* Candidates per position at o = 1.5 and at the odd 1.5 + 2^-10, with
     |p v| small enough to keep o's binade. */
  for (uint32_t j = 3; j < last; ++j) {
    for (int ev = -14; ev <= 4 && n_hits[j] < MAX_HITS; ++ev) {
      for (uint32_t m = 0; m < 1024u && n_hits[j] < MAX_HITS; m += 3u) {
        for (int sg = 0; sg < 4 && n_hits[j] < MAX_HITS; ++sg) {
          const float cand =
            ldexpf(1.0f + (float)m / 1024.0f, ev) * ((sg & 1) ? -1.0f : 1.0f);
          const float o = (sg & 2) ? 1.5f + 1.0f / 1024.0f : 1.5f;
          if (fabsf(pr[j] * cand) < 0.0625f &&
              amc_is_midpoint_case(o, pr[j], cand)) {
            hits[j * MAX_HITS + n_hits[j]++] = cand;
          }
        }
      }
    }
  }
  for (uint32_t d = 0; d < hd; ++d) {
    float o = 0.0f;
    for (uint32_t j = 0; j < L; ++j) {
      float *vj = v + (size_t)j * row + h * hd + d;
      if (j >= 3u && j < last && o != 0.0f) {
        int eo = 0;
        (void)frexpf(o, &eo); /* |o| in [2^(eo-1), 2^eo) */
        for (uint32_t t = 0; t < n_hits[j]; ++t) {
          const float cand =
            ldexpf(hits[j * MAX_HITS + (t + d) % n_hits[j]], eo - 1) *
            (o < 0.0f ? -1.0f : 1.0f);
          if (attn_m1_det_rne16(cand) == cand &&
              amc_is_midpoint_case(o, pr[j], cand)) {
            *vj = cand;
            ++planted;
            break;
          }
        }
      }
      o = attn_m1_det_fma16(o, pr[j], attn_m1_det_rne16(*vj));
    }
  }
  free(kt), free(vz), free(pr), free(hits), free(n_hits);
  return planted;
}

#endif /* __NNTRAINER_ATTN_M1_CASES_H__ */
