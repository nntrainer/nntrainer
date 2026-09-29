// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   m1_ops_det.h
 * @date   27 Sep 2026
 * @brief  The scalar specification of the M=1 small ops: RMSNorm (whole row
 *         and per head), RoPE at head_dim 64, causal conv1d L=3 + gate, and
 *         the MoE router's logits + sigmoid + top-k (#132)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * WHY THIS EXISTS
 *
 * Every one of these ops feeds a quantizer on the decode path (the next
 * FC's u8 activation quant, doc 45 section 3.3), so two implementations
 * that are merely accurate will eventually round one element to different
 * u8 levels and the token stream forks -- swiglu_det.h tells that story
 * with device numbers. The cure is the same: ONE sequence of IEEE-754 f32
 * operations, written down here, that the HVX kernel
 * (htp_backend/hvx/hvx_m1_ops_f32.c) reproduces bit for bit and that a
 * later ARM twin can compile as well. This header is that sequence.
 *
 * THE CONTRACT
 *
 * Every step is one f32 multiply, add or subtract, rounded to nearest even
 * on its own, with subnormals kept (LEDGER rule 24: HVX does not flush, so
 * neither may the host). No qf32 (v75 and v79 disagree on qf32 -> sf), no
 * libm. Two exceptions, both in the RMSNorm row scale and both exact by
 * IEEE's definition (#164): one fused multiply-add per square (fmaf; the
 * Android CPU's fmla, the DSP's sffma), and a square root and a
 * reciprocal computed in integers, each correctly rounded (no libm sqrt,
 * no division). The ORDER is part of the contract: reassociating the
 * reduction, splitting the fused step or fusing any other changes the
 * bits. Each scalar operation below stores through a volatile so no
 * compiler can contract or reassociate it.
 *
 *   rmsnorm_det(x[chunk], gamma[chunk], eps), per chunk (2048 for the
 *   hidden norm, 64 for a q/k head) -- the Android CPU's
 *   neon::rms_norm_wrt_width_fp32_intrinsic + multiply_i(gamma), read off
 *   the shipped libnntrainer.so's aarch64 disassembly (plan 164 section 0):
 *     acc[j] = 0,  acc[j] = fma(x[16 i + j], x[16 i + j], acc[j]),
 *              j = 0 .. 15              (four float32x4 fmla accumulators)
 *     h[k]   = (acc[4k] + acc[4k+1]) + (acc[4k+2] + acc[4k+3])   (faddp x2)
 *     s      = ((h[0] + h[1]) + h[2]) + h[3]
 *     d      = s * (1/chunk) + eps  (1/chunk exact: power of two, so this
 *                                     is the CPU's s / chunk bit for bit)
 *     r      = RN(1 / RN(sqrt(d)))  (fsqrt then fdiv: two roundings, NOT
 *                                     the correctly rounded 1/sqrt)
 *     y[i]   = (x[i] * r) * gamma[i]
 *   rope64_det(x[64], cs[64]),  cs = cos[0..31] | sin[0..31], i < 32 --
 *   in fp16, the Android CPU's compute_rotary_emb_value(__fp16) (#152;
 *   rne16 is attn_m1_det.h's, the operands its copyData / (_FP16) casts):
 *     a = rne16(x[i]); b = rne16(x[i + 32]); c = rne16(cs[i]);
 *     s = rne16(cs[i + 32])
 *     x[i]      = rne16(rne16(a*c) - rne16(b*s))
 *     x[i + 32] = rne16(rne16(a*s) + rne16(b*c))
 *   (fmul / fsub / fadd .8h, each rounded; no fmla in that loop)
 *
 *   conv_gate_m1_det(abc[3C], state[2C], w[3C], out[C]):
 *     g   = a*c                                (the in_proj split a | b | c)
 *     y   = (w0*g + w1*s1) + w2*s0             (hvx_conv_gate_f32's order;
 *                                               state = x_{t-2} | x_{t-1})
 *     out = b*y
 *     state <- s1 | g
 *
 *   router_topk_det(x[K], W[K][E], bias[E], top_k), E <= 32, K % 4 == 0
 *   (#132; LFM2's buildExpertAssignments, lfm2_moe_layer.cpp):
 *     a_j      = 0,  a_j = a_j + x[k] * W[k][e]  for k = j (mod 4), in k
 *                order                  (four lane accumulators on HVX)
 *     logit[e] = (a_0 + a_1) + (a_2 + a_3)
 *     sig[e]   = recip_det(1 + exp_det(0 - logit[e]))   (swiglu_det.h)
 *     score[e] = sig[e] + bias[e]
 *     sel[r]   = argmax of score over the unchosen, the LOWEST index on a
 *                tie, r = 0 .. top_k-1  (the CPU comparator's total order)
 *     wsum     = 0, wsum = wsum + sig[sel[r]] in selection order
 *     inv      = recip_det(wsum + 1e-6)
 *     weight[r] = (sig[sel[r]] * inv) * 1.0  (NORM_TOPK_PROB, its epsilon
 *                and ROUTED_SCALING_FACTOR)
 *     ponytail: LFM2's router constants are hard-coded here; a router with
 *     another scale or no normalization needs them in the op record.
 *
 * DOMAIN (the analogue of LEDGER section 4's SiLU/exp argument clamp). The
 * router's exp is exp_det, clamped to [-88, 85] (swiglu_det.h), so its
 * recip_det sees 1 + e in [1, 1 + e^85] and wsum + 1e-6 in [1e-6, 4]: both
 * inside recip_det's seed range. The norms contain no exp: d >= eps, a
 * positive normal (the graph validator requires it), keeps sqrt_rn and
 * recip_rn on normal inputs with normal results; a sum of squares that
 * overflows gives d = +inf and r = +0, the CPU's IEEE answer. "fp32
 * inside, narrow once": nothing here narrows -- every op reads and writes
 * f32, and the u8 narrowing is the next FC's quantizer.
 *
 * ACCURACY, for the record. The RMSNorm scale is no longer this file's
 * choice: it is the CPU's, 16 fused chains and two correctly rounded
 * steps, within about 1.5 ulp of the exact 1/sqrt. Until #164 this was a
 * 32-lane tree sum and a three-step Newton rsqrt, within 1-3 ulp of the
 * CPU's r on about half of the decode rows -- enough to flip the rare
 * element at an fp16 boundary downstream of the q/k norm.
 */

#ifndef __NNTRAINER_M1_OPS_DET_H__
#define __NNTRAINER_M1_OPS_DET_H__

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "attn_m1_det.h"
#include "swiglu_det.h"

/** @brief f32 lanes in one 128-byte HVX vector; the reduction's width. */
#define M1_DET_LANES 32u
/** @brief RoPE head dimension: one vector per half head. */
#define M1_DET_HEAD_DIM 64u
/** @brief The CPU RMSNorm's independent fused chains: 4 x float32x4. */
#define M1_DET_NORM_CHAINS 16u

/** @brief One f32 operation, forced to round on its own (see swiglu_det.h
 *         for why a volatile store and not -ffp-contract). */
static inline float m1_det_mul(float a, float b) {
  volatile float r = a * b;
  return r;
}
static inline float m1_det_add(float a, float b) {
  volatile float r = a + b;
  return r;
}
static inline float m1_det_sub(float a, float b) {
  volatile float r = a - b;
  return r;
}
/** @brief a * b + c with ONE rounding (IEEE fusedMultiplyAdd). */
static inline float m1_det_fma(float a, float b, float c) {
  volatile float r = fmaf(a, b, c);
  return r;
}

static inline uint32_t m1_det_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return u;
}
static inline float m1_det_float(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

/**
 * @brief RN(sqrt(d)) in integers, the CPU's fsqrt. DOMAIN: d a positive
 *        normal or +inf (NaN passes through).
 *
 * d = m 2^ee; M = m << s in [2^48, 2^50) with ee - s even, so
 * sqrt(d) = sqrt(M) 2^((ee-s)/2). q = floor(sqrt(M)) has 25 bits, and
 * (q + 1) >> 1 rounds to nearest: M is not an odd square (it has at least
 * 25 trailing zero bits), so the root is never a tie.
 */
static inline float m1_sqrt_rn_det(float d) {
  const uint32_t u = m1_det_bits(d);
  if (u >= 0x7f800000u) {
    return d;
  }
  const int ee = (int)(u >> 23) - 150;
  const uint64_t m = (u & 0x7fffffu) | 0x800000u;
  const int s = ((ee - 25) & 1) ? 26 : 25;
  uint64_t rem = m << s, q = 0, bit = 1ull << 48;
  while (bit) {
    if (rem >= q + bit) {
      rem -= q + bit;
      q = (q >> 1) + bit;
    } else {
      q >>= 1;
    }
    bit >>= 2;
  }
  uint32_t r = (uint32_t)((q + 1u) >> 1);
  int E = (ee - s) / 2 + 1;
  if (r == 1u << 24) {
    r >>= 1;
    ++E;
  }
  return m1_det_float(((uint32_t)(E + 150) << 23) | (r & 0x7fffffu));
}

/**
 * @brief RN(1/q) in integers, the CPU's fdiv 1.0f / q. DOMAIN: q a positive
 *        normal whose reciprocal is normal (q < 2^126), or +inf (-> +0).
 *
 * Exact for a power of two; else Q = floor(2^48 / mq) has 25 bits and
 * (Q + 1) >> 1 rounds to nearest (1/mq is not a dyadic rational, so no
 * tie). On the DSP the 64-bit divide is __hexagon_udivdi3.
 */
static inline float m1_recip_rn_det(float q) {
  const uint32_t u = m1_det_bits(q);
  if (u >= 0x7f800000u) {
    return u == 0x7f800000u ? 0.0f : q;
  }
  const int Eq = (int)(u >> 23) - 150;
  const uint64_t mq = (u & 0x7fffffu) | 0x800000u;
  if (mq == 0x800000u) {
    return m1_det_float((uint32_t)(-Eq - 23 + 127) << 23);
  }
  uint32_t r = (uint32_t)(((1ull << 48) / mq + 1u) >> 1);
  int E = -47 - Eq;
  if (r == 1u << 24) {
    r >>= 1;
    ++E;
  }
  return m1_det_float(((uint32_t)(E + 150) << 23) | (r & 0x7fffffu));
}

/** @brief s from the 16 chains: two faddp per float32x4, then the four
 *         in order -- ((h0 + h1) + h2) + h3. */
static inline float m1_norm_reduce_det(const float *acc) {
  float h[4];
  for (uint32_t k = 0; k < 4u; ++k) {
    h[k] = m1_det_add(m1_det_add(acc[4u * k], acc[4u * k + 1u]),
                      m1_det_add(acc[4u * k + 2u], acc[4u * k + 3u]));
  }
  return m1_det_add(m1_det_add(m1_det_add(h[0], h[1]), h[2]), h[3]);
}

/** @brief The CPU's sum of squares of @a chunk floats (chunk % 16 == 0):
 *         16 fused chains, then m1_norm_reduce_det. */
static inline float m1_sumsq_cpu_det(const float *x, uint32_t chunk) {
  float acc[M1_DET_NORM_CHAINS];
  for (uint32_t j = 0; j < M1_DET_NORM_CHAINS; ++j) {
    acc[j] = 0.0f;
  }
  for (uint32_t i = 0; i < chunk; i += M1_DET_NORM_CHAINS) {
    for (uint32_t j = 0; j < M1_DET_NORM_CHAINS; ++j) {
      acc[j] = m1_det_fma(x[i + j], x[i + j], acc[j]);
    }
  }
  return m1_norm_reduce_det(acc);
}

/** @brief r from the sum of squares: RN(1 / RN(sqrt(s / chunk + eps))). */
static inline float m1_norm_scale_det(float s, uint32_t chunk, float eps) {
  const float d = m1_det_add(m1_det_mul(s, 1.0f / (float)chunk), eps);
  return m1_recip_rn_det(m1_sqrt_rn_det(d));
}

/**
 * @brief RMSNorm over one chunk: y = (x * r) * gamma, r the CPU's scale.
 *
 * @param chunk  a power of two and a multiple of 32
 * @return r, the row scale (what the IDL's row_scale reports)
 */
static inline float m1_rmsnorm_chunk_det(const float *x, const float *gamma,
                                         float *y, uint32_t chunk, float eps) {
  const float r = m1_norm_scale_det(m1_sumsq_cpu_det(x, chunk), chunk, eps);
  for (uint32_t i = 0; i < chunk; ++i) {
    y[i] = m1_det_mul(m1_det_mul(x[i], r), gamma[i]);
  }
  return r;
}

/**
 * @brief RMSNorm over n floats in chunks of @a chunk, one shared gamma of
 *        @a chunk floats (chunk == n is the hidden norm; chunk == 64 with
 *        n = heads * 64 is reshaped_rms_norm's per-head q/k norm).
 *
 * @param row_scale  n / chunk floats, r per chunk; may be NULL
 */
static inline void m1_rmsnorm_det(const float *x, const float *gamma, float *y,
                                  uint32_t n, uint32_t chunk, float eps,
                                  float *row_scale) {
  for (uint32_t c = 0; c * chunk < n; ++c) {
    const float r =
      m1_rmsnorm_chunk_det(x + c * chunk, gamma, y + c * chunk, chunk, eps);
    if (row_scale) {
      row_scale[c] = r;
    }
  }
}

/** @brief RoPE on one head of 64 in fp16, in place. cs = cos[32] |
 *         sin[32] in f32 (the table the host uploads); the output is fp16
 *         values in f32. */
static inline void m1_rope64_det(float *x, const float *cs) {
  for (uint32_t i = 0; i < M1_DET_HEAD_DIM / 2u; ++i) {
    const float a = attn_m1_det_rne16(x[i]);
    const float b = attn_m1_det_rne16(x[i + 32]);
    const float c = attn_m1_det_rne16(cs[i]);
    const float s = attn_m1_det_rne16(cs[i + 32]);
    x[i] = attn_m1_det_rne16(m1_det_sub(attn_m1_det_rne16(m1_det_mul(a, c)),
                                        attn_m1_det_rne16(m1_det_mul(b, s))));
    x[i + 32] =
      attn_m1_det_rne16(m1_det_add(attn_m1_det_rne16(m1_det_mul(a, s)),
                                   attn_m1_det_rne16(m1_det_mul(b, c))));
  }
}

/**
 * @brief Causal depthwise conv1d (L=3) + gate for one token.
 *
 * @param abc    [3C]: the in_proj split a | b | c
 * @param state  [2C]: x_{t-2} | x_{t-1} in, x_{t-1} | (a*c) out
 * @param w      [3C]: w0 (current), w1 (t-1), w2 (t-2)
 * @param out    [C]:  b * conv1d(a*c)
 */
static inline void m1_conv_gate_det(const float *abc, float *state,
                                    const float *w, float *out, uint32_t C) {
  const float *a = abc, *b = abc + C, *c = abc + 2u * C;
  const float *w0 = w, *w1 = w + C, *w2 = w + 2u * C;
  float *s0 = state, *s1 = state + C;
  for (uint32_t j = 0; j < C; ++j) {
    const float g = m1_det_mul(a[j], c[j]);
    float y = m1_det_add(m1_det_mul(w0[j], g), m1_det_mul(w1[j], s1[j]));
    y = m1_det_add(y, m1_det_mul(w2[j], s0[j]));
    out[j] = m1_det_mul(b[j], y);
    s0[j] = s1[j];
    s1[j] = g;
  }
}

/** @brief Router width limit: one HVX vector of f32 lanes per weight row. */
#define M1_DET_ROUTER_MAX_E M1_DET_LANES

/**
 * @brief The MoE router of one token: logits, sigmoid, biased top-k and the
 *        normalized routing weights (LFM2's buildExpertAssignments).
 *
 * @param x       K floats, the ffn-normed row
 * @param w       K x E floats, row-major [K][E] (the gate weight's layout)
 * @param bias    E floats, added for the selection only
 * @param K       a multiple of 4
 * @param E       1..32
 * @param top_k   1..E
 * @param logits  E floats out
 * @param sel     top_k expert indices out, in selection order
 * @param weight  top_k routing weights out, in selection order
 */
static inline void m1_router_topk_det(const float *x, const float *w,
                                      const float *bias, uint32_t K, uint32_t E,
                                      uint32_t top_k, float *logits,
                                      uint32_t *sel, float *weight) {
  float sig[M1_DET_ROUTER_MAX_E], score[M1_DET_ROUTER_MAX_E];
  uint32_t taken = 0u, e, r, k;
  float wsum = 0.0f, inv;
  for (e = 0; e < E; ++e) {
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (k = 0; k < K; ++k) {
      a[k & 3u] = m1_det_add(a[k & 3u], m1_det_mul(x[k], w[(size_t)k * E + e]));
    }
    logits[e] = m1_det_add(m1_det_add(a[0], a[1]), m1_det_add(a[2], a[3]));
    sig[e] = swiglu_det_recip(
      m1_det_add(1.0f, swiglu_det_exp(m1_det_sub(0.0f, logits[e]))));
    score[e] = m1_det_add(sig[e], bias[e]);
  }
  for (r = 0; r < top_k; ++r) {
    uint32_t best = E;
    for (e = 0; e < E; ++e) {
      if (!(taken & (1u << e)) && (best == E || score[e] > score[best])) {
        best = e;
      }
    }
    taken |= 1u << best;
    sel[r] = best;
    wsum = m1_det_add(wsum, sig[best]);
  }
  inv = swiglu_det_recip(m1_det_add(wsum, 1e-6f));
  for (r = 0; r < top_k; ++r) {
    weight[r] = m1_det_mul(m1_det_mul(sig[sel[r]], inv), 1.0f);
  }
}

#endif /* __NNTRAINER_M1_OPS_DET_H__ */
