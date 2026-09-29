// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   attn_m1_det.h
 * @date   27 Sep 2026
 * @brief  The scalar specification of decode attention at m=1: the Android
 *         fp16 CPU attention (mha_core.cpp's ENABLE_FP16 branch) written as
 *         f32 operations, which hvx_attn_m1_f32.c reproduces bit for bit
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * WHY THIS EXISTS
 *
 * The attention output feeds the o-projection's u8 activation quantizer on
 * the decode path (doc 45 section 3.3), so two implementations that are
 * merely accurate will eventually round one element to different u8
 * levels and the token stream forks. The resident path must also match
 * the CPU it replaces: on Android that CPU runs attention in fp16 from end
 * to end (q/k/v, RoPE, scores, softmax, PV), and #152 measured that an f32
 * DSP attention lands 41-52 dB from it while only the whole fp16 sequence,
 * fused multiply-adds included, closes the gap. So this header is the
 * Android CPU's sequence (plan 152 section 3.1), each fp16 operation
 * written as one IEEE f32 operation followed by rne16, and the HVX kernel
 * (htp_backend/hvx/hvx_attn_m1_f32.c) reproduces it bit for bit.
 *
 * THE CONTRACT (plan 152 section 3.2)
 *
 * Every value is an fp16 value held in an f32. Every step is one IEEE f32
 * add, subtract, multiply or divide (round to nearest even, stored through
 * a volatile so no compiler contracts it) followed by rne16, or an integer
 * operation on the bits. No hf instruction, no qf32.
 *
 *   rne16(x)     round to the fp16 grid, ties to even, fp16 subnormals
 *                kept: (x + C) - C with C = 1.5 * 2^(max(E, -14) + 13),
 *                E = x's exponent, then x's sign restored (so a value that
 *                rounds to zero keeps its sign, as the CPU's fcvt does).
 *                For an f32 op on fp16 operands, rounding to f32 first and
 *                then to fp16 equals rounding once: 24 >= 2 * 11 + 2.
 *   fma16(c,a,b) RN16(c + a*b), rounded ONCE (vfmaq_f16 / fmla .8h). a*b
 *                is exact in f32 (22 bits, >= 2^-48). s = c + a*b in f32
 *                and TwoSum's exact error err; if err != 0, s is replaced
 *                by its round-to-odd neighbour (truncate toward zero, then
 *                set the last bit), which makes the later rne16 equal to
 *                the single rounding (round-to-odd at 24 >= 11 + 2 bits).
 *   exp16(d)     rne16(exp_ps(d)): exp_ps is neon_mathfun.hxx's, whose
 *                fx = x * LOG2EF + 0.5 compiles to one fused fmla -- here
 *                (float)((double)x * LOG2EF + 0.5), exact in double for
 *                every fp16 x and for the clamp value -- then its separate
 *                f32 steps in its order, and the (int) truncation.
 *
 * Per call (L = pos + 1 positions, q head hq over kv head h = hq / gqa,
 * Kt[head_dim][max_seq] and V[max_seq][head_dim] per kv head, head_dim a
 * multiple of 8 -- 64 on the kernel):
 *
 *   0. round    q16 = rne16(q), k16 = rne16(k), v16 = rne16(v) (copyData)
 *   1. append   Kt[d][pos] = k16[d];  V[pos][d] = v16[d]
 *   2. scores   acc[l] = 0 for l < 8; for d ascending:
 *                 acc[d % 8] = fma16(acc[d % 8], q16[d], Kt[d][p])
 *               t = rne16(rne16(rne16(acc0 + acc1) + rne16(acc2 + acc3))
 *                         + rne16(rne16(acc4 + acc5) + rne16(acc6 + acc7)))
 *               (the vpaddq_f16 tree), t = 0 + t (the CPU's `sum += t`),
 *               s[p] = rne16(t * scale)   (scale 0.125 = the CPU's / 8)
 *   3. max      m = max over p < L of s[p], then m = m + 0 (sign of zero
 *               canonical; it cannot change any s - m)
 *   4. exp      e[p] = exp16(rne16(s[p] - m))
 *   5. sum      l = 0; for p ascending: l = rne16(l + e[p])  (sequential)
 *   6. divide   e[p] = rne16(e[p] / l)                     (fdiv .8h)
 *   7. PV       o[d] = 0; for p ascending: o[d] = fma16(o[d], e[p],
 *               V[p][d]);  out[d] = o[d]
 *
 * The CPU's functions, in the order mha_core.cpp calls them:
 * compute_rotary_emb_value(__fp16) (m1_ops_det.h's m1_rope64_det),
 * compute_kcaches(_FP16) (step 2), softmax_row_inplace(_FP16) (3-6),
 * compute_fp16vcache_transposed (7). AttnM1F16Det.* in
 * unittest_nntrainer_cpu_backend_fp16.cpp compares this header with those
 * exported functions on the device.
 *
 * WHICH CPU. This is the CPU path of an Android build with enable-fp16
 * (tools/package_android.sh's default, and every set measured so far).
 * A build with -Denable-fp16=false, or the host, runs mha_core's f32
 * attention instead; against that CPU this spec is as far off as the f32
 * one was from the fp16 CPU (plan 152's table, the other way round), and
 * the host E2E's resident attention lines read against it (#152's
 * fwd-hd64 39.18 -> 37.17 dB).
 *
 * DOMAIN. Every fp16 value finite: |q|, |k|, |v| < 65520 and no score, sum
 * or output past 65504 (the CPU's own softmax turns an overflow into NaN,
 * so nothing outside this domain is worth matching). rne16 does not
 * saturate to inf; past 65520 it returns 65536. d = s - m <= 0 by
 * construction; exp16 is 1 at d = 0, so 1 <= l <= L and the divide never
 * sees 0.
 *
 * NOT IN THIS HEADER: how the kernel gets there. It reads exp16 from a
 * table (attn_m1_det_exp_table below, 18.5 k entries; the host check
 * compares it with exp16 at every fp16 d <= 0) and divides with recip_det
 * plus an exact midpoint check (hvx_attn_m1_f32.c); both are proved equal
 * to the lines above, not approximations of them.
 */

#ifndef __NNTRAINER_ATTN_M1_DET_H__
#define __NNTRAINER_ATTN_M1_DET_H__

#include <stdint.h>
#include <string.h>

/** @brief f32 lanes in one 128-byte HVX vector: the position block the
 *         kernel's score loop computes at once. */
#define ATTN_M1_DET_LANES 32u
/** @brief The CPU's score accumulators: one float16x8_t over d = 8 blk + l. */
#define ATTN_M1_DET_ACC 8u

/*
 * THE PHASE WORDS (#146) -- a measurement channel, not arithmetic. A
 * caller that passes 2 * n_q + ATTN_M1_PROF_WORDS stats floats to the
 * attn_m1_forward entry receives the (m, l) pairs followed by these uint32
 * words, bit-copied (memcpy, so no 2^24 float bound). They live here
 * because this is the one header the kernel, the skel entry, the host
 * check and the device gtest all include. Pcycles are the core-wide
 * counter, so the caller's and the units' readings share one clock.
 *   APPEND     pcycles of the k/v append (caller)
 *   POOL       pcycles from just before the pool run to its return (caller)
 *   LANES      units that ran concurrently: min(units, workers + 1)
 *   SCORES     sum over units of the score loops (incl. the running vmax)
 *   SOFTMAX    sum over units of max tree + exp16 + sum + divide
 *   PV         sum over units of PV
 *   BUSY_MAX   max over units of (end - start)
 *   START_MAX  max over units of (start - POOL's t0): dispatch skew
 *   CALL_QT    19.2 MHz qtimer ticks of the whole forward (dsp_us = /19.2)
 * Pool dispatch + merge = POOL - BUSY_MAX; lane imbalance = BUSY_MAX /
 * ((SCORES + SOFTMAX + PV) / LANES).
 */
#define ATTN_M1_PROF_APPEND 0u
#define ATTN_M1_PROF_POOL 1u
#define ATTN_M1_PROF_LANES 2u
#define ATTN_M1_PROF_SCORES 3u
#define ATTN_M1_PROF_SOFTMAX 4u
#define ATTN_M1_PROF_PV 5u
#define ATTN_M1_PROF_BUSY_MAX 6u
#define ATTN_M1_PROF_START_MAX 7u
#define ATTN_M1_PROF_CALL_QT 8u
#define ATTN_M1_PROF_WORDS 9u

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
static inline float attn_m1_det_div(float a, float b) {
  volatile float r = a / b;
  return r;
}

static inline uint32_t attn_m1_det_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return u;
}
static inline float attn_m1_det_float(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

/** @brief The exponent field of 2^-14, fp16's smallest normal. */
#define ATTN_M1_DET_EF_MIN16 0x38800000u
/** @brief Added to an exponent field: 13 more in the exponent and the 0.5
 *         mantissa bit, i.e. C = 1.5 * 2^(E + 13), whose f32 ulp is 2^(E-10),
 *         the fp16 ulp at E. */
#define ATTN_M1_DET_RNE16_C 0x06C00000u

/** @brief x rounded to the fp16 grid, ties to even (the contract's rne16;
 *         hvx_rne16_sf in hvx_convert.h is the same operations per lane). */
static inline float attn_m1_det_rne16(float x) {
  const uint32_t u = attn_m1_det_bits(x);
  uint32_t ef = u & 0x7F800000u;
  if (ef < ATTN_M1_DET_EF_MIN16) {
    ef = ATTN_M1_DET_EF_MIN16;
  }
  const float c = attn_m1_det_float(ef + ATTN_M1_DET_RNE16_C);
  const float r = attn_m1_det_sub(attn_m1_det_add(x, c), c);
  return attn_m1_det_float(attn_m1_det_bits(r) | (u & 0x80000000u));
}

/** @brief RN16(c + a * b) rounded once, for fp16 values c, a, b. */
static inline float attn_m1_det_fma16(float c, float a, float b) {
  const float p = attn_m1_det_mul(a, b); /* exact */
  const float s = attn_m1_det_add(c, p);
  const float bv = attn_m1_det_sub(s, c);
  const float err = attn_m1_det_add(attn_m1_det_sub(c, attn_m1_det_sub(s, bv)),
                                    attn_m1_det_sub(p, bv));
  const uint32_t e = attn_m1_det_bits(err);
  uint32_t u = attn_m1_det_bits(s);
  if (e != 0u) {
    /* Round to odd: one step toward zero when the exact sum is smaller in
       magnitude than s (err's sign differs), then the last bit set. err is
       never -0 (x - x is +0), and s is never 0 when err is not. */
    if ((e ^ u) & 0x80000000u) {
      u -= 1u;
    }
    u |= 1u;
  }
  return attn_m1_det_rne16(attn_m1_det_float(u));
}

/** @brief neon_mathfun.hxx's constants, with the conversions its source
 *         applies: double literals narrowed to float (vdupq_n_f32 of a
 *         double, a static const float array), float literals as is. */
#define ATTN_M1_DET_EXP_HI 88.3762626647949f
#define ATTN_M1_DET_EXP_LO -88.3762626647949f
#define ATTN_M1_DET_LOG2EF ((float)1.44269504088896341)
#define ATTN_M1_DET_EXP_C1 ((float)0.693359375)
#define ATTN_M1_DET_EXP_C2 ((float)-2.12194440e-4)

/** @brief exp_ps for one lane, bit-identical to the NEON function as the
 *         NDK compiles it (one fused fmla for fx, every other step
 *         separate; objdump-checked per plan 152 step 6). */
static inline float attn_m1_det_exp_ps(float x) {
  static const float p[6] = {(float)1.9875691500E-4, (float)1.3981999507E-3,
                             (float)8.3334519073E-3, (float)4.1665795894E-2,
                             (float)1.6666665459E-1, (float)5.0000001201E-1};
  x = (x < ATTN_M1_DET_EXP_HI) ? x : ATTN_M1_DET_EXP_HI;
  x = (x > ATTN_M1_DET_EXP_LO) ? x : ATTN_M1_DET_EXP_LO;
  /* The fused x * LOG2EF + 0.5, rounded once: exact in double, then one
     rounding to f32. volatile keeps a host from evaluating it wider. */
  volatile double fxd = (double)x * (double)ATTN_M1_DET_LOG2EF + 0.5;
  float fx = (float)fxd;
  const float tmp = (float)(int32_t)fx; /* fcvtzs + scvtf */
  fx = attn_m1_det_sub(tmp, (tmp > fx) ? 1.0f : 0.0f);
  const float t1 = attn_m1_det_mul(fx, ATTN_M1_DET_EXP_C1);
  float z = attn_m1_det_mul(fx, ATTN_M1_DET_EXP_C2);
  x = attn_m1_det_sub(x, t1);
  x = attn_m1_det_sub(x, z);
  float y = attn_m1_det_mul(p[0], x);
  z = attn_m1_det_mul(x, x);
  y = attn_m1_det_add(y, p[1]);
  y = attn_m1_det_mul(y, x);
  y = attn_m1_det_add(y, p[2]);
  y = attn_m1_det_mul(y, x);
  y = attn_m1_det_add(y, p[3]);
  y = attn_m1_det_mul(y, x);
  y = attn_m1_det_add(y, p[4]);
  y = attn_m1_det_mul(y, x);
  y = attn_m1_det_add(y, p[5]);
  y = attn_m1_det_mul(y, z);
  y = attn_m1_det_add(y, x);
  y = attn_m1_det_add(y, 1.0f);
  const uint32_t mm = (uint32_t)((int32_t)fx + 0x7F) << 23;
  return attn_m1_det_mul(y, attn_m1_det_float(mm));
}

/** @brief The contract's exp16: fp16(exp_ps(d)). */
static inline float attn_m1_det_exp16(float d) {
  return attn_m1_det_rne16(attn_m1_det_exp_ps(d));
}

/*
 * THE EXP TABLE (the kernel's form of exp16 for d <= 0). Index 0 holds
 * exp16(0) and serves every |d| < 2^-14 (fp16 zero and subnormals, where
 * exp16 is 1.0 too); index i >= 1 holds the fp16 magnitude whose f32 bits
 * are (i + BIAS) << 13, up to 17.5 at LAST; LAST + 1 is 0.0, which serves
 * every d below -17.5 (exp16 is 0 there: e^-17.5 is under half of fp16's
 * smallest subnormal). The host check proves table == exp16 at every fp16
 * d <= 0, the sentinel and the subnormal claim included.
 */
#define ATTN_M1_DET_EXP_BIAS 0x1C3FFu /* (bits(2^-14) >> 13) - 1 */
#define ATTN_M1_DET_EXP_LAST 0x4861u  /* (bits(17.5) >> 13) - BIAS */
#define ATTN_M1_DET_EXP_N (ATTN_M1_DET_EXP_LAST + 2u)

/** @brief The table index of an fp16 value d <= 0. */
static inline uint32_t attn_m1_det_exp_index(float d) {
  const int32_t i =
    (int32_t)((attn_m1_det_bits(d) & 0x7FFFFFFFu) >> 13) - ATTN_M1_DET_EXP_BIAS;
  return i < 0                               ? 0u
         : i > (int32_t)ATTN_M1_DET_EXP_LAST ? ATTN_M1_DET_EXP_LAST + 1u
                                             : (uint32_t)i;
}

/** @brief Fills ATTN_M1_DET_EXP_N floats. */
static inline void attn_m1_det_exp_table(float *tab) {
  tab[0] = attn_m1_det_exp16(0.0f);
  for (uint32_t i = 1; i <= ATTN_M1_DET_EXP_LAST; ++i) {
    tab[i] = attn_m1_det_exp16(
      attn_m1_det_float(0x80000000u | ((i + ATTN_M1_DET_EXP_BIAS) << 13)));
  }
  tab[ATTN_M1_DET_EXP_LAST + 1u] = 0.0f;
}

/** @brief Steps 0-1: one kv head's row at @a pos. Kt is [head_dim][max_seq],
 *         V is [max_seq][head_dim]. */
static inline void attn_m1_det_append(float *Kt, float *V, uint32_t head_dim,
                                      uint32_t max_seq, uint32_t pos,
                                      const float *k, const float *v) {
  for (uint32_t d = 0; d < head_dim; ++d) {
    Kt[(size_t)d * max_seq + pos] = attn_m1_det_rne16(k[d]);
    V[(size_t)pos * head_dim + d] = attn_m1_det_rne16(v[d]);
  }
}

/** @brief Step 2: the scaled score of one q head (already rne16'd) at
 *         position @a p. */
static inline float attn_m1_det_score(const float *q16, const float *Kt,
                                      uint32_t head_dim, uint32_t max_seq,
                                      uint32_t p, float scale) {
  float a[ATTN_M1_DET_ACC] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t d = 0; d < head_dim; ++d) {
    a[d % ATTN_M1_DET_ACC] = attn_m1_det_fma16(a[d % ATTN_M1_DET_ACC], q16[d],
                                               Kt[(size_t)d * max_seq + p]);
  }
  const float s01 = attn_m1_det_rne16(attn_m1_det_add(a[0], a[1]));
  const float s23 = attn_m1_det_rne16(attn_m1_det_add(a[2], a[3]));
  const float s45 = attn_m1_det_rne16(attn_m1_det_add(a[4], a[5]));
  const float s67 = attn_m1_det_rne16(attn_m1_det_add(a[6], a[7]));
  const float s03 = attn_m1_det_rne16(attn_m1_det_add(s01, s23));
  const float s47 = attn_m1_det_rne16(attn_m1_det_add(s45, s67));
  const float t =
    attn_m1_det_add(0.0f, attn_m1_det_rne16(attn_m1_det_add(s03, s47)));
  return attn_m1_det_rne16(attn_m1_det_mul(t, scale));
}

/**
 * @brief Steps 3-6 in place: scores s[0..L) (fp16 values) become the
 *        probabilities (softmax_row_inplace(_FP16) for one head).
 * @param m_out, l_out  the max and the sum; may be NULL
 */
static inline void attn_m1_det_softmax(float *s, uint32_t L, float *m_out,
                                       float *l_out) {
  float m = s[0];
  for (uint32_t p = 1; p < L; ++p) {
    if (s[p] > m) {
      m = s[p];
    }
  }
  m = attn_m1_det_add(m, 0.0f);
  float l = 0.0f;
  for (uint32_t p = 0; p < L; ++p) {
    s[p] = attn_m1_det_exp16(attn_m1_det_rne16(attn_m1_det_sub(s[p], m)));
    l = attn_m1_det_rne16(attn_m1_det_add(l, s[p]));
  }
  for (uint32_t p = 0; p < L; ++p) {
    s[p] = attn_m1_det_rne16(attn_m1_det_div(s[p], l));
  }
  if (m_out) {
    *m_out = m;
  }
  if (l_out) {
    *l_out = l;
  }
}

/**
 * @brief Steps 2-7 for one q head against one kv head's cache.
 *
 * @param q        [head_dim], this q head (post-RoPE), head_dim <= 128
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
  float q16[128];
  for (uint32_t d = 0; d < head_dim; ++d) {
    q16[d] = attn_m1_det_rne16(q[d]);
  }
  for (uint32_t p = 0; p < L; ++p) {
    e[p] = attn_m1_det_score(q16, Kt, head_dim, max_seq, p, scale);
  }
  attn_m1_det_softmax(e, L, m_out, l_out);
  for (uint32_t d = 0; d < head_dim; ++d) {
    float o = 0.0f;
    for (uint32_t p = 0; p < L; ++p) {
      o = attn_m1_det_fma16(o, e[p], V[(size_t)p * head_dim + d]);
    }
    out[d] = o;
  }
}

/**
 * @brief One layer's decode attention for every q head, the cache already
 *        holding L positions (steps 0-1 done by the caller).
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
