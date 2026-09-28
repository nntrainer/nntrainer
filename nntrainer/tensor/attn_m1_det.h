// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   attn_m1_det.h
 * @date   27 Sep 2026
 * @brief  The scalar specification of decode attention at m=1 over an f32
 *         KV cache: scores, exact max, exp_det, lane-tree sum, recip_det,
 *         PV -- the sequence hvx_attn_m1_f32.c reproduces bit for bit
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * WHY THIS EXISTS
 *
 * The attention output feeds the o-projection's u8 activation quantizer on
 * the decode path (doc 45 section 3.3), so two implementations that are
 * merely accurate will eventually round one element to different u8
 * levels and the token stream forks -- swiglu_det.h tells that story with
 * device numbers. As for the M=1 small ops (m1_ops_det.h), the cure is ONE
 * sequence of IEEE-754 f32 operations, written down here, that the HVX
 * kernel (htp_backend/hvx/hvx_attn_m1_f32.c) reproduces bit for bit and a
 * later ARM twin can compile as well.
 *
 * THE CONTRACT (plan 81 section 3.2)
 *
 * Every step is one f32 multiply, add, subtract or max, rounded to nearest
 * even on its own, subnormals kept (LEDGER rule 24). No fused multiply-add,
 * no qf32, no divide, no libm exp. exp_det and recip_det are
 * swiglu_det_exp / swiglu_det_recip, the normative scalar twins of
 * hvx_exp_det_sf / hvx_recip_det_sf. The ORDER is part of the contract.
 *
 * Per call (L = pos + 1 positions in the cache, q head hq over kv head
 * h = hq / gqa, Kt[head_dim][max_seq] and V[max_seq][head_dim] per kv head):
 *
 *   1. append   Kt[d][pos] = k[d];  V[pos][d] = v[d]            (copies)
 *   2. scores   acc = 0; for d ascending: acc = acc + (q[d] * Kt[d][p])
 *               s[p] = acc * scale                    (one lane per p)
 *   3. max      m = max over p < L of s[p]     (exact, order-free on finite
 *               values: the HVX vmax over blocks + rotate tree is the spec)
 *   4. exp      e[p] = exp_det(s[p] - m)       (argument <= 0, e[argmax] = 1)
 *   5. sum      acc[j] = 0; for block b ascending: acc[j] = acc[j] + e[32b+j]
 *               (positions >= L add nothing), then the pair tree
 *               (j, j+16), (j, j+8), (j, j+4), (j, j+2), (j, j+1), lane 0:
 *               what vror 64/32/16/8/4 bytes + vadd leaves in every lane
 *   6. recip    r = recip_det(l)               (l >= 1 since e[argmax] = 1)
 *   7. PV       o[d] = 0; for p ascending: o[d] = o[d] + (e[p] * V[p][d])
 *               out[d] = o[d] * r
 *
 * DOMAIN. Scores stay finite for |q|, |k| below about 1e17; a residual row
 * past that is already broken. exp_det's argument is <= 0 by construction,
 * so its clamp (LEDGER section 4's exp argument rule, inherited from
 * swiglu_det.h: [-88, 85]) fires only on the far tail, where the true
 * weight is below f32's smallest normal anyway. recip_det's domain
 * (d <= 1.6e38) holds because 1 <= l <= L <= max_seq. "fp32 inside, narrow
 * once": nothing here narrows; the cache is f32 (plan 81 section 3.1).
 *
 * TWO PROPERTIES WORTH KNOWING. (a) A score is never -0: acc starts at +0
 * and +0 + (-0) = +0, so the max's sign-of-zero ambiguity never arises and
 * m is well defined bit for bit. (b) recip_det(1.0f) is 0x3F7FFFFF, one
 * ulp below 1 (swiglu_det.h documents the fixed point), so at L = 1 the
 * output is v * (1 - 2^-24), not v itself: bit identity between the two
 * machines, not agreement with a divide, is the property.
 */

#ifndef __NNTRAINER_ATTN_M1_DET_H__
#define __NNTRAINER_ATTN_M1_DET_H__

#include <stdint.h>

#include "swiglu_det.h"

/** @brief f32 lanes in one 128-byte HVX vector: the sum tree's width and
 *         the position block the score loop computes at once. */
#define ATTN_M1_DET_LANES 32u

/** @brief One f32 operation, forced to round on its own (see swiglu_det.h
 *         for why a volatile store and not -ffp-contract). */
static inline float attn_m1_det_mul(float a, float b) {
  volatile float r = a * b;
  return r;
}
static inline float attn_m1_det_add(float a, float b) {
  volatile float r = a + b;
  return r;
}
static inline float attn_m1_det_sub(float a, float b) {
  volatile float r = a - b;
  return r;
}

/** @brief Step 1: one kv head's row at @a pos. Kt is [head_dim][max_seq],
 *         V is [max_seq][head_dim]. */
static inline void attn_m1_det_append(float *Kt, float *V, uint32_t head_dim,
                                      uint32_t max_seq, uint32_t pos,
                                      const float *k, const float *v) {
  for (uint32_t d = 0; d < head_dim; ++d) {
    Kt[(size_t)d * max_seq + pos] = k[d];
    V[(size_t)pos * head_dim + d] = v[d];
  }
}

/** @brief Step 2: the scaled score of one q head at position @a p. */
static inline float attn_m1_det_score(const float *q, const float *Kt,
                                      uint32_t head_dim, uint32_t max_seq,
                                      uint32_t p, float scale) {
  float acc = 0.0f;
  for (uint32_t d = 0; d < head_dim; ++d) {
    acc =
      attn_m1_det_add(acc, attn_m1_det_mul(q[d], Kt[(size_t)d * max_seq + p]));
  }
  return attn_m1_det_mul(acc, scale);
}

/** @brief Step 5: the sum of e[0..L) as the HVX reduces it: 32 lane
 *         accumulators over the position blocks, then the pairwise tree.
 *         IEEE add is commutative bit for bit, so which operand of a pair
 *         comes first does not matter. */
static inline float attn_m1_det_sum(const float *e, uint32_t L) {
  float acc[ATTN_M1_DET_LANES];
  for (uint32_t j = 0; j < ATTN_M1_DET_LANES; ++j) {
    acc[j] = 0.0f;
  }
  for (uint32_t p = 0; p < L; ++p) {
    acc[p % ATTN_M1_DET_LANES] =
      attn_m1_det_add(acc[p % ATTN_M1_DET_LANES], e[p]);
  }
  for (uint32_t s = ATTN_M1_DET_LANES / 2u; s >= 1u; s >>= 1) {
    for (uint32_t j = 0; j < s; ++j) {
      acc[j] = attn_m1_det_add(acc[j], acc[j + s]);
    }
  }
  return acc[0];
}

/**
 * @brief Steps 2-7 for one q head against one kv head's cache.
 *
 * @param q        [head_dim], this q head (post-RoPE)
 * @param Kt, V    the kv head's cache: Kt [head_dim][max_seq], V
 *                 [max_seq][head_dim]
 * @param L        positions in the cache, 1..max_seq
 * @param e        scratch of at least L floats (the probabilities)
 * @param out      [head_dim]
 * @param m_out, l_out  the max and the sum (the localising intermediates
 *                 the IDL's stats report); may be NULL
 */
static inline void attn_m1_det_head(const float *q, const float *Kt,
                                    const float *V, uint32_t head_dim,
                                    uint32_t max_seq, uint32_t L, float scale,
                                    float *e, float *out, float *m_out,
                                    float *l_out) {
  for (uint32_t p = 0; p < L; ++p) {
    e[p] = attn_m1_det_score(q, Kt, head_dim, max_seq, p, scale);
  }
  float m = e[0];
  for (uint32_t p = 1; p < L; ++p) {
    if (e[p] > m) {
      m = e[p];
    }
  }
  for (uint32_t p = 0; p < L; ++p) {
    e[p] = swiglu_det_exp(attn_m1_det_sub(e[p], m));
  }
  const float l = attn_m1_det_sum(e, L);
  const float r = swiglu_det_recip(l);
  for (uint32_t d = 0; d < head_dim; ++d) {
    float o = 0.0f;
    for (uint32_t p = 0; p < L; ++p) {
      o =
        attn_m1_det_add(o, attn_m1_det_mul(e[p], V[(size_t)p * head_dim + d]));
    }
    out[d] = attn_m1_det_mul(o, r);
  }
  if (m_out) {
    *m_out = m;
  }
  if (l_out) {
    *l_out = l;
  }
}

/**
 * @brief One layer's decode attention for every q head, the cache already
 *        holding L positions (step 1 done by the caller).
 *
 * @param q        [n_kv * gqa][head_dim]
 * @param Kt_layer [n_kv][head_dim][max_seq]
 * @param V_layer  [n_kv][max_seq][head_dim]
 * @param e        scratch of at least L floats
 * @param out      [n_kv * gqa][head_dim]
 * @param stats    2 * n_kv * gqa floats, (m, l) per q head; may be NULL
 */
static inline void attn_m1_det_forward(const float *q, const float *Kt_layer,
                                       const float *V_layer, uint32_t n_kv,
                                       uint32_t gqa, uint32_t head_dim,
                                       uint32_t max_seq, uint32_t L,
                                       float scale, float *e, float *out,
                                       float *stats) {
  for (uint32_t h = 0; h < n_kv; ++h) {
    const float *Kt = Kt_layer + (size_t)h * head_dim * max_seq;
    const float *V = V_layer + (size_t)h * max_seq * head_dim;
    for (uint32_t g = 0; g < gqa; ++g) {
      const uint32_t hq = h * gqa + g;
      attn_m1_det_head(q + (size_t)hq * head_dim, Kt, V, head_dim, max_seq, L,
                       scale, e, out + (size_t)hq * head_dim,
                       stats ? stats + 2u * hq : (float *)0,
                       stats ? stats + 2u * hq + 1u : (float *)0);
    }
  }
}

#endif /* __NNTRAINER_ATTN_M1_DET_H__ */
