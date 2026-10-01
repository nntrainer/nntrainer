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
 * libm. Two exceptions, both exact by IEEE's definition: the fused
 * multiply-add (fmaf; the Android CPU's fmla, the DSP's sffma) where the
 * CPU fuses, and square roots, reciprocals and divisions computed in
 * integers, each correctly rounded (no libm sqrt, no division) -- in the
 * RMSNorm row scale (#164) and in the CPU-order router and SwiGLU (#132
 * PR 2, through q4_gemv_cpu_det.h's helpers). The ORDER is part of the
 * contract: reassociating the reduction, splitting the fused step or fusing any
 * other changes the bits. Each scalar operation below stores through a volatile
 * so no compiler can contract or reassociate it.
 *
 *   rmsnorm_det(x[chunk], gamma[chunk], eps), per chunk (2048 for the
 *   hidden norm, 64 for a q/k head; [plan 201 S4] any multiple of 32, e.g.
 *   Gemma's 2816, and gamma NULL for y = x * r) -- the Android CPU's
 *   neon::rms_norm_wrt_width_fp32_intrinsic + multiply_i(gamma), read off
 *   the shipped libnntrainer.so's aarch64 disassembly (plan 164 section 0):
 *     acc[j] = 0,  acc[j] = fma(x[16 i + j], x[16 i + j], acc[j]),
 *              j = 0 .. 15              (four float32x4 fmla accumulators)
 *     h[k]   = (acc[4k] + acc[4k+1]) + (acc[4k+2] + acc[4k+3])   (faddp x2)
 *     s      = ((h[0] + h[1]) + h[2]) + h[3]
 *     d      = s / chunk + eps      (the CPU's fdiv: a multiply by the
 *                                     exact 1/chunk for a power of two,
 *                                     else cpu_det_div_rn)
 *     r      = RN(1 / RN(sqrt(d)))  (fsqrt then fdiv: two roundings, NOT
 *                                     the correctly rounded 1/sqrt)
 *     y[i]   = (x[i] * r) * gamma[i]
 *   rope64_det(x[64], cs[64]),  cs = cos[0..31] | sin[0..31], i < 32 --
 *   ([plan 201 S4] rope_det(x[hd], hd, cs[hd]) is the same with 32 -> hd / 2)
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
 *     y   = fma(w2, s0, fma(w1, s1, w0*g))     (the Android CPU's decode:
 *                                               neon causal_depthwise_
 *                                               conv1d_k3_decode's vmulq +
 *                                               two vfmaq; state =
 *                                               x_{t-2} | x_{t-1}. #132 E5c:
 *                                               the unfused (w0*g + w1*s1)
 *                                               + w2*s0 it replaced left the
 *                                               CPU on the device after 5-25
 *                                               tokens)
 *     out = b*y
 *     state <- s1 | g
 *
 *   router_cpu_det(x[K], W[K][E], bias[E], top_k), E <= 32 (#132 PR 2;
 *   the Android CPU's router, read off the shipped binaries: OpenBLAS
 *   sgemv_n and buildExpertAssignments in libcausallm_core.so). Unlike
 *   the ops above it is the CPU's order, so it takes q4_gemv_cpu_det.h's
 *   fused step and exact integer division:
 *     logit[e] = 0, logit[e] = fma(x[k], W[k][e], logit[e]), k = 0..K-1
 *     sig[e]   = RN(1 / (expf(-logit[e]) + 1))   (expf: bionic's, ported
 *                                                  as m1_expf_bionic_det)
 *     score[e] = sig[e] + bias[e]
 *     sel[r]   = argmax of score over the unchosen, the LOWEST index on a
 *                tie, r = 0 .. top_k-1  (the CPU comparator's total order)
 *     wsum     = (sig[sel1] + sig[sel3] + ..) + (sig[sel0] + sig[sel2] + ..)
 *                (two accumulators over odd / even r from 0, -ffast-math's
 *                unroll; an odd top_k adds its last one after)
 *     inv      = RN(1 / (wsum + 1e-6))
 *     weight[r] = sig[sel[r]] * inv  (NORM_TOPK_PROB, its epsilon;
 *                ROUTED_SCALING_FACTOR 1.0 is folded away in the binary)
 *     ponytail: LFM2's router constants are hard-coded here; a router with
 *     another scale or no normalization needs them in the op record.
 *
 *   router_softmax_det(x[K], W[K][E], pes[E], top_k), E <= 128 ([plan 201
 *   S4] Gemma 4, #4296's Gemma4MoELayer; x is its input already normed and
 *   scaled, see m1_router_input_scale_det): unfused chains, softmax with
 *   swiglu_det.h's exp / recip, top_k with the lowest index on a tie,
 *   renormalised over the chosen and times the per-expert scale (the
 *   function's comment has the order).
 *
 *   rmsnorm_n1 ([plan 201 S4] N1, opt-in, not the CPU's bits): the sum of
 *   squares in integers after a per-row power of two (its comment).
 *
 *   swiglu_cpu_det(y[n], z[n]): neon::swiglu with neon_mathfun's exp_ps as
 *   the binary computes it (one fused step, the rest separate; plan 132
 *   section 0.2): out = RN(y / (exp_ps(-y) + 1)) * z. The MoE's own
 *   swiglu_det.h stays: that path's reference is the HTP, not the CPU.
 *
 *   argmax_first(x[n]): std::max_element, the first maximum wins.
 *
 * DOMAIN (the analogue of LEDGER section 4's SiLU/exp argument clamp). The
 * CPU-order router and SwiGLU take any finite input, as the CPU does.
 * The norms contain no exp: d >= eps, a positive normal (the graph
 * validator requires it), keeps sqrt_rn and recip_rn on normal inputs
 * with normal results; a sum of squares that overflows gives d = +inf and
 * r = +0, the CPU's IEEE answer. "fp32
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
#include "q4_gemv_cpu_det.h"
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

/** @brief r from the sum of squares: RN(1 / RN(sqrt(s / chunk + eps))).
 *         s / chunk is the CPU's fdiv: for a power-of-two chunk the
 *         multiply by its exact reciprocal (the same bits, and the code
 *         before plan 201 S4), else the integer correctly rounded divide
 *         (Gemma's hidden 2816). */
static inline float m1_norm_scale_det(float s, uint32_t chunk, float eps) {
  const float mean = (chunk & (chunk - 1u)) == 0u
                       ? m1_det_mul(s, 1.0f / (float)chunk)
                       : cpu_det_div_rn(s, (float)chunk);
  return m1_recip_rn_det(m1_sqrt_rn_det(m1_det_add(mean, eps)));
}

/** @brief y = (x * r) * gamma, or x * r with no gamma (Gemma's v norm). */
static inline void m1_norm_apply_det(const float *x, const float *gamma,
                                     float *y, uint32_t chunk, float r) {
  for (uint32_t i = 0; i < chunk; ++i) {
    y[i] =
      gamma ? m1_det_mul(m1_det_mul(x[i], r), gamma[i]) : m1_det_mul(x[i], r);
  }
}

/**
 * @brief RMSNorm over one chunk: y = (x * r) * gamma, r the CPU's scale.
 *
 * @param gamma  chunk floats, or NULL for y = x * r
 * @param chunk  a multiple of 32 (the CPU's 16 chains cover it, no tail)
 * @return r, the row scale (what the IDL's row_scale reports)
 */
static inline float m1_rmsnorm_chunk_det(const float *x, const float *gamma,
                                         float *y, uint32_t chunk, float eps) {
  const float r = m1_norm_scale_det(m1_sumsq_cpu_det(x, chunk), chunk, eps);
  m1_norm_apply_det(x, gamma, y, chunk, r);
  return r;
}

/* ---- [plan 201 S4] N1: the sum of squares in integers (opt-in) ---------- */

/** @brief N1's fixed point: |q| <= 2^M1_N1_BITS (2^22, the largest the
 *         1.5 2^23 rounding identity takes). The HVX kernel splits q =
 *         a 2^11 + b and sums a^2, a b and b^2 (each < 2^22 + 1) on u32
 *         lanes, M1_N1_BLOCK_VECS vectors before a lane can overflow. */
#define M1_N1_BITS 22
#define M1_N1_BLOCK_VECS 512u

/** @brief k with |x| 2^k < 2^M1_N1_BITS for every x whose |bits| <= amax
 *         (the row's largest |bits|): 13 - E for amax in [2^E, 2^(E+1)),
 *         at most 127 so 2^k is a normal f32 (a row below 2^-114 keeps
 *         fewer bits; its s is far under any eps). */
static inline int m1_n1_shift_det(uint32_t amax) {
  const int k = (M1_N1_BITS - 1) - ((int)(amax >> 23) - 127);
  return k > 127 ? 127 : k;
}

/** @brief 2^e as f32 bits for e in [-149, 127] (subnormal below -126). */
static inline float m1_n1_pow2_det(int e) {
  return m1_det_float(e >= -126 ? (uint32_t)(e + 127) << 23
                                : 1u << (uint32_t)(e + 149));
}

/** @brief q = RNE(|x| 2^k) in integers: |x| 2^k is exact (or below 2^-126,
 *         where q is 0 either way), and adding 1.5 2^23 rounds it to an
 *         integer in the low mantissa bits (hvx_sf_to_w_rne's identity). */
static inline uint32_t m1_n1_q_det(float x, float p) {
  const float v = m1_det_mul(m1_det_float(m1_det_bits(x) & 0x7fffffffu), p);
  return m1_det_bits(m1_det_add(v, 12582912.0f)) - 0x4B400000u;
}

/** @brief RN(S) for S < 2^64, in integers (no u64 -> f32 instruction to
 *         trust): the top 24 bits, round to nearest even on the rest. */
static inline float m1_n1_u64_to_f32_det(uint64_t S) {
  int e = 0;
  if (S == 0u) {
    return 0.0f;
  }
  while ((S >> e) >= (1ull << 24)) {
    ++e;
  }
  uint64_t m = S >> e;
  if (e > 0) {
    const uint64_t rest = S & ((1ull << e) - 1u), half = 1ull << (e - 1);
    m += rest > half || (rest == half && (m & 1u));
    if (m == 1ull << 24) {
      m >>= 1;
      ++e;
    }
  }
  /* m < 2^24 exactly, e <= 40: (float)m is exact, the scale a power of 2 */
  return m1_det_mul((float)(uint32_t)m, m1_n1_pow2_det(e));
}

/** @brief N1's row scale from the exact integer sum S of q^2 at shift k:
 *         s = RN(S) 2^-k 2^-k (two exact scalings while s stays normal),
 *         then the CPU form of m1_norm_scale_det. Shared by the spec and
 *         the HVX kernel; only S is the kernel's own. */
static inline float m1_n1_norm_scale_det(uint64_t S, int k, uint32_t chunk,
                                         float eps) {
  const float p = m1_n1_pow2_det(-k);
  const float s = m1_det_mul(m1_det_mul(m1_n1_u64_to_f32_det(S), p), p);
  return m1_norm_scale_det(s, chunk, eps);
}

/**
 * @brief N1 RMSNorm over one chunk (plan 201 section 3.7): the per-row
 *        power of two k from the largest |x|, q = RNE(|x| 2^k) <= 2^22,
 *        S = sum q^2 exactly in integers -- any lane count, any order, the
 *        same S -- then r from S with the spec's integer sqrt / recip, and
 *        y = (x * r) * gamma in f32. Not the f32 norm's bits: each |x| keeps
 *        22 bits relative to the row's largest. DOMAIN: finite x.
 */
static inline float m1_rmsnorm_n1_chunk_det(const float *x, const float *gamma,
                                            float *y, uint32_t chunk,
                                            float eps) {
  uint32_t amax = 0u;
  uint64_t S = 0u;
  for (uint32_t i = 0; i < chunk; ++i) {
    const uint32_t a = m1_det_bits(x[i]) & 0x7fffffffu;
    amax = a > amax ? a : amax;
  }
  const int k = m1_n1_shift_det(amax);
  const float p = m1_n1_pow2_det(k);
  for (uint32_t i = 0; i < chunk; ++i) {
    const uint64_t q = m1_n1_q_det(x[i], p);
    S += q * q;
  }
  const float r = m1_n1_norm_scale_det(S, k, chunk, eps);
  m1_norm_apply_det(x, gamma, y, chunk, r);
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

/**
 * @brief RoPE on one head of @a hd (even) in fp16, in place: pairs (i, i +
 *        hd / 2), the CPU's compute_rotary_emb_value(__fp16) with half_ =
 *        hd / 2. cs = cos[hd / 2] | sin[hd / 2] in f32 (the table the host
 *        uploads); the output is fp16 values in f32.
 *
 * [plan 201 S4] The rotary variant is the table's, not this function's:
 * mha_core's precompute_freqs builds the angles -- "default" theta_i =
 * theta^(-2i / hd), "proportional" the same for i < partial * hd / 2 and
 * 0 above (cos 1, sin 0: those pairs pass through the same operations),
 * per layer theta -- so Gemma 4's partial rotary factor and per-layer
 * theta reach the DSP as the per-op table, computed by the CPU's own code.
 */
static inline void m1_rope_det(float *x, uint32_t hd, const float *cs) {
  const uint32_t h = hd / 2u;
  for (uint32_t i = 0; i < h; ++i) {
    const float a = attn_m1_det_rne16(x[i]);
    const float b = attn_m1_det_rne16(x[i + h]);
    const float c = attn_m1_det_rne16(cs[i]);
    const float s = attn_m1_det_rne16(cs[i + h]);
    x[i] = attn_m1_det_rne16(m1_det_sub(attn_m1_det_rne16(m1_det_mul(a, c)),
                                        attn_m1_det_rne16(m1_det_mul(b, s))));
    x[i + h] =
      attn_m1_det_rne16(m1_det_add(attn_m1_det_rne16(m1_det_mul(a, s)),
                                   attn_m1_det_rne16(m1_det_mul(b, c))));
  }
}

/** @brief m1_rope_det at head_dim 64: cs = cos[32] | sin[32]. */
static inline void m1_rope64_det(float *x, const float *cs) {
  m1_rope_det(x, M1_DET_HEAD_DIM, cs);
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
    float y = m1_det_mul(w0[j], g);
    y = m1_det_fma(w1[j], s1[j], y);
    y = m1_det_fma(w2[j], s0[j], y);
    out[j] = m1_det_mul(b[j], y);
    s0[j] = s1[j];
    s1[j] = g;
  }
}

/** @brief Router width limit: one HVX vector of f32 lanes per weight row. */
#define M1_DET_ROUTER_MAX_E M1_DET_LANES

/**
 * @brief expf as the phone's bionic libm computes it: the Arm
 *        optimized-routines algorithm (32-entry 2^(i/32) table, a cubic in
 *        double, one final rounding to f32). G0 (ExpfBionic.*) pins this
 *        port to the device's libm over all 2^32 inputs; the host check
 *        pins it to the same algorithm in glibc.
 *
 * The contraction question is settled: fused and unfused polynomial
 * steps, and round-half-even (the host's shift) versus round-half-away
 * (aarch64's frinta) for k, give the same f32 on every input, so the port
 * uses plain double operations -- the DSP has no double fma to call.
 */
static inline float m1_expf_bionic_det(float x) {
  static const uint64_t tab[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full,
    0x3fef9301d0125b51ull, 0x3fef72b83c7d517bull, 0x3fef54873168b9aaull,
    0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull, 0x3fef06fe0a31b715ull,
    0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull,
    0x3feea47eb03a5585ull, 0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull,
    0x3feea11473eb0187ull, 0x3feea589994cce13ull, 0x3feeace5422aa0dbull,
    0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull,
    0x3fef3720dcef9069ull, 0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full,
    0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull};
  const double inv_ln2_n = 0x1.71547652b82fep+0 * 32.0;
  const double c0 = 0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0;
  const double c1 = 0x1.ebfce50fac4f3p-3 / 32.0 / 32.0;
  const double c2 = 0x1.62e42ff0c52d6p-1 / 32.0;
  const double shift = 0x1.8p+52;
  const uint32_t ux = cpu_det_bits(x), abstop = (ux >> 20) & 0x7ffu;
  if (abstop >= 0x42bu) { /* |x| >= 88 or NaN */
    if (ux == 0xff800000u) {
      return 0.0f;
    }
    if (abstop >= 0x7f8u) {
      return cpu_det_add(x, x);
    }
    if (x > 0x1.62e42ep6f) {
      return cpu_det_float(0x7f800000u);
    }
    if (x < -0x1.9fe368p6f) {
      return 0.0f;
    }
  }
  volatile double z = inv_ln2_n * (double)x;
  volatile double kd = z + shift;
  uint64_t ki, t;
  memcpy(&ki, (const void *)&kd, sizeof(ki));
  kd = kd - shift;
  volatile double r = z - kd;
  t = tab[ki % 32u] + (ki << 47);
  double s;
  memcpy(&s, &t, sizeof(s));
  volatile double p = c0 * r;
  p = p + c1;
  volatile double r2 = r * r;
  volatile double y = c2 * r;
  y = y + 1.0;
  volatile double q = p * r2;
  y = q + y;
  y = y * s;
  return (float)y;
}

/** @brief One expert's sigmoid and selection score, as
 *         buildExpertAssignments computes them: sig = RN(1 / (expf(-l) +
 *         1)) with bionic expf, score = sig + bias. */
static inline void m1_router_cpu_sigmoid(float logit, float bias, float *sig,
                                         float *score) {
  const float ex =
    m1_expf_bionic_det(m1_det_float(m1_det_bits(logit) ^ 0x80000000u));
  *sig = cpu_det_div_rn(1.0f, m1_det_add(ex, 1.0f));
  *score = m1_det_add(*sig, bias);
}

/**
 * @brief The router after its sigmoids: biased top-k (the lowest index on
 *        a tie), the -ffast-math weight sum, a true divide, the weights.
 *        Shared by the spec and the DSP kernel.
 *
 * @param sig     E sigmoids (1..32), m1_router_cpu_sigmoid's
 * @param score   E selection scores, the same
 * @param top_k   1..E
 * @param sel     top_k expert indices out, in selection order
 * @param weight  top_k routing weights out, in selection order
 */
static inline void m1_router_cpu_pick(const float *sig, const float *score,
                                      uint32_t E, uint32_t top_k, uint32_t *sel,
                                      float *weight) {
  uint32_t taken = 0u, e, r;
  float a0 = 0.0f, a1 = 0.0f, wsum, inv;
  for (r = 0; r < top_k; ++r) {
    uint32_t best = E;
    for (e = 0; e < E; ++e) {
      if (!(taken & (1u << e)) && (best == E || score[e] > score[best])) {
        best = e;
      }
    }
    taken |= 1u << best;
    sel[r] = best;
  }
  for (r = 0; r + 1u < top_k; r += 2u) {
    a0 = m1_det_add(sig[sel[r]], a0);
    a1 = m1_det_add(sig[sel[r + 1u]], a1);
  }
  wsum = m1_det_add(a1, a0);
  if (top_k & 1u) {
    wsum = m1_det_add(sig[sel[top_k - 1u]], wsum);
  }
  inv = cpu_det_div_rn(1.0f, m1_det_add(wsum, 1e-6f));
  for (r = 0; r < top_k; ++r) {
    weight[r] = m1_det_mul(sig[sel[r]], inv);
  }
}

/**
 * @brief The Android CPU's MoE router of one token (the doc above).
 *
 * @param x       K floats, the ffn-normed row
 * @param w       K x E floats, row-major [K][E] (the gate weight's layout)
 * @param bias    E floats, added for the selection only
 * @param E       1..32
 * @param top_k   1..E
 * @param logits  E floats out
 * @param sel     top_k expert indices out, in selection order
 * @param weight  top_k routing weights out, in selection order
 */
static inline void m1_router_cpu_det(const float *x, const float *w,
                                     const float *bias, uint32_t K, uint32_t E,
                                     uint32_t top_k, float *logits,
                                     uint32_t *sel, float *weight) {
  float sig[M1_DET_ROUTER_MAX_E], score[M1_DET_ROUTER_MAX_E];
  for (uint32_t e = 0; e < E; ++e) {
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
      acc = cpu_det_fma(x[k], w[(size_t)k * E + e], acc);
    }
    logits[e] = acc;
    m1_router_cpu_sigmoid(acc, bias[e], &sig[e], &score[e]);
  }
  m1_router_cpu_pick(sig, score, E, top_k, sel, weight);
}

/* ---- [plan 201 S4] Gemma 4's softmax router ---------------------------- */

/** @brief The softmax router's widest expert set (Gemma 4: 128). */
#define M1_DET_ROUTER_SM_MAX_E 128u

/** @brief g[f] = RN(router_scale[f] * RN(1 / RN(sqrt(H)))): #4296's
 *         per-feature factor `router_scale[feature] * hidden_scale`, with
 *         hidden_scale = 1.0f / std::sqrt((float)H), each step correctly
 *         rounded as on the CPU. The norm's (x * r) * g is then the CPU's
 *         rms_norm_wrt_width followed by `row[f] *= g[f]`, bit for bit. */
static inline void m1_router_input_scale_det(const float *router_scale,
                                             uint32_t H, float *g) {
  const float hs = m1_recip_rn_det(m1_sqrt_rn_det((float)H));
  for (uint32_t f = 0; f < H; ++f) {
    g[f] = m1_det_mul(router_scale[f], hs);
  }
}

/**
 * @brief The softmax router after its logits (shared by the spec and the
 *        DSP kernel): m = max logit; e = exp_det(l - m); p = e * recip_det(
 *        sum e, in expert order); top_k by p, the LOWEST index on a tie;
 *        t = sum of the chosen p in selection order; weight = (p *
 *        recip_det(t)) * per_expert_scale. #4296 divides where this takes
 *        recip_det, and its topK is an unstable std::partial_sort, so an
 *        exact tie may go either way there (D2: not bit-preserving).
 *
 * @param E      1..M1_DET_ROUTER_SM_MAX_E; logits finite
 * @param top_k  1..E
 */
static inline void m1_router_softmax_pick(const float *logits, const float *pes,
                                          uint32_t E, uint32_t top_k,
                                          uint32_t *sel, float *weight) {
  float p[M1_DET_ROUTER_SM_MAX_E], m = logits[0], s = 0.0f, t = 0.0f, inv;
  uint8_t taken[M1_DET_ROUTER_SM_MAX_E];
  uint32_t e, r;
  for (e = 1; e < E; ++e) {
    m = logits[e] > m ? logits[e] : m;
  }
  for (e = 0; e < E; ++e) {
    p[e] = swiglu_det_exp(m1_det_sub(logits[e], m));
    s = m1_det_add(s, p[e]);
    taken[e] = 0u;
  }
  inv = swiglu_det_recip(s);
  for (e = 0; e < E; ++e) {
    p[e] = m1_det_mul(p[e], inv);
  }
  for (r = 0; r < top_k; ++r) {
    uint32_t best = E;
    for (e = 0; e < E; ++e) {
      if (!taken[e] && (best == E || p[e] > p[best])) {
        best = e;
      }
    }
    taken[best] = 1u;
    sel[r] = best;
    t = m1_det_add(t, p[best]);
  }
  inv = swiglu_det_recip(t);
  for (r = 0; r < top_k; ++r) {
    weight[r] = m1_det_mul(m1_det_mul(p[sel[r]], inv), pes[sel[r]]);
  }
}

/**
 * @brief Gemma 4's router of one token after its input norm (plan 201 S4;
 *        Gemma4MoELayer::forwardTensors in nntrainer/nntrainer#4296): the
 *        logits as E unfused chains over k in order, acc = RN(acc +
 *        RN(x[k] * W[k][e])) -- 32 experts a vector on HVX -- then
 *        m1_router_softmax_pick. The whole router is m1_rmsnorm_det(h,
 *        g, x, H, H, eps) with g from m1_router_input_scale_det, then this.
 *
 * @param x       K floats: the normed, scaled router input
 * @param w       K x E floats, row-major [K][E]
 * @param pes     E floats, the per-expert scale
 * @param E       1..M1_DET_ROUTER_SM_MAX_E
 */
static inline void m1_router_softmax_det(const float *x, const float *w,
                                         const float *pes, uint32_t K,
                                         uint32_t E, uint32_t top_k,
                                         float *logits, uint32_t *sel,
                                         float *weight) {
  for (uint32_t e = 0; e < E; ++e) {
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
      acc = m1_det_add(acc, m1_det_mul(x[k], w[(size_t)k * E + e]));
    }
    logits[e] = acc;
  }
  m1_router_softmax_pick(logits, pes, E, top_k, sel, weight);
}

/** @brief neon_mathfun's exp_ps for one lane, as the shipped binary runs
 *         it: fmin / fmax clamp, ONE fma (fx), a truncating floor, the
 *         cephes polynomial in separate multiplies and adds. */
static inline float m1_exp_ps_cpu_det(float x) {
  const float hi = 88.3762626647949f, lo = -88.3762626647949f;
  x = x < hi ? x : hi;
  x = x > lo ? x : lo;
  float fx = cpu_det_fma(x, 1.44269504088896341f, 0.5f);
  const float t = (float)(int32_t)fx; /* fcvtzs + scvtf: |fx| < 129 */
  fx = m1_det_sub(t, t > fx ? 1.0f : 0.0f);
  x = m1_det_sub(x, m1_det_mul(fx, 0.693359375f));
  x = m1_det_sub(x, m1_det_mul(fx, -2.12194440e-4f));
  const float z = m1_det_mul(x, x);
  float y = m1_det_mul(1.9875691500E-4f, x);
  y = m1_det_add(y, 1.3981999507E-3f);
  y = m1_det_mul(y, x);
  y = m1_det_add(y, 8.3334519073E-3f);
  y = m1_det_mul(y, x);
  y = m1_det_add(y, 4.1665795894E-2f);
  y = m1_det_mul(y, x);
  y = m1_det_add(y, 1.6666665459E-1f);
  y = m1_det_mul(y, x);
  y = m1_det_add(y, 5.0000001201E-1f);
  y = m1_det_mul(y, z);
  y = m1_det_add(y, x);
  y = m1_det_add(y, 1.0f);
  const uint32_t pow2n = (uint32_t)((int32_t)fx + 127) << 23;
  return m1_det_mul(y, m1_det_float(pow2n));
}

/** @brief neon::swiglu: out[i] = RN(y[i] / (exp_ps(-y[i]) + 1)) * z[i]
 *         (n % 4 == 0: the scalar tail uses std::exp, not exp_ps). */
static inline void m1_swiglu_cpu_det(const float *y, const float *z, float *out,
                                     uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    const float e =
      m1_exp_ps_cpu_det(m1_det_float(m1_det_bits(y[i]) ^ 0x80000000u));
    out[i] = m1_det_mul(cpu_det_div_rn(y[i], m1_det_add(e, 1.0f)), z[i]);
  }
}

/** @brief std::max_element: the index of the first maximum. */
static inline uint32_t m1_argmax_first(const float *x, uint32_t n) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < n; ++i) {
    if (x[best] < x[i]) {
      best = i;
    }
  }
  return best;
}

#endif /* __NNTRAINER_M1_OPS_DET_H__ */
