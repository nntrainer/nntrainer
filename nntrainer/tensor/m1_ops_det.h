// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   m1_ops_det.h
 * @date   27 Sep 2026
 * @brief  The scalar specification of the M=1 small ops: RMSNorm (whole row
 *         and per head), RoPE at head_dim 64, causal conv1d L=3 + gate
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
 * neither may the host). No fused multiply-add, no qf32 (v75 and v79
 * disagree on qf32 -> sf), no divide, no sqrt, no libm. The ORDER is part
 * of the contract: reassociating the reduction or contracting a
 * multiply-add changes the bits. Each scalar operation below stores
 * through a volatile so no compiler can contract or reassociate it.
 *
 *   rmsnorm_det(x[chunk], gamma[chunk], eps), per chunk (2048 for the
 *   hidden norm, 64 for a q/k head):
 *     acc[j]   = 0,  acc[j] = acc[j] + x[32 i + j] * x[32 i + j]  (32 lanes)
 *     sum      = pairwise tree over the lanes: (j, j+16), (j, j+8), (j, j+4),
 *                (j, j+2), (j, j+1) -- what vror + vadd leaves in every lane
 *     d        = sum * (1/chunk) + eps        (1/chunk exact: power of two)
 *     r        = rsqrt_det(d)
 *     y[i]     = (x[i] * r) * gamma[i]        (the CPU's order: scale, then
 *                                               multiply_i(gamma))
 *   rsqrt_det(d):
 *     y = bitcast_f32(0x5F3759DF - (bitcast_u32(d) >> 1))
 *     h = d * 0.5f
 *     three times: t = h*y; t = t*y; t = 1.5f - t; y = y*t
 *
 *   rope64_det(x[64], cs[64]),  cs = cos[0..31] | sin[0..31], i < 32:
 *     a = x[i]; b = x[i + 32]
 *     x[i]      = (a*c) - (b*s)
 *     x[i + 32] = (a*s) + (b*c)                (neon_impl.cpp's formula)
 *
 *   conv_gate_m1_det(abc[3C], state[2C], w[3C], out[C]):
 *     g   = a*c                                (the in_proj split a | b | c)
 *     y   = (w0*g + w1*s1) + w2*s0             (hvx_conv_gate_f32's order;
 *                                               state = x_{t-2} | x_{t-1})
 *     out = b*y
 *     state <- s1 | g
 *
 * DOMAIN (the analogue of LEDGER section 4's SiLU/exp argument clamp). No
 * op here contains an exp, so no argument clamp applies; what bounds the
 * inputs instead is the rsqrt seed. d >= eps = 1e-5 keeps the seed in the
 * normal range from below; from above, the sum of chunk squares stays
 * finite while |x| < sqrt(FLT_MAX / chunk), about 4e17 at chunk = 2048
 * and 2.3e18 at 64, and a residual row past that is already broken. "fp32
 * inside, narrow once": nothing here narrows -- every op reads and writes
 * f32, and the u8 narrowing is the next FC's quantizer.
 *
 * ACCURACY, for the record. Three Newton-Raphson steps take the magic
 * seed's ~3.4 % to below f32's last bit; the 32-lane tree sum is more
 * accurate than a sequential one, not less. The fixed point is not always
 * the correctly rounded 1/sqrt, and it does not have to be: bit identity
 * between the implementations, not agreement with libm, is the property.
 */

#ifndef __NNTRAINER_M1_OPS_DET_H__
#define __NNTRAINER_M1_OPS_DET_H__

#include <stdint.h>
#include <string.h>

/** @brief f32 lanes in one 128-byte HVX vector; the reduction's width. */
#define M1_DET_LANES 32u
/** @brief RoPE head dimension: one vector per half head. */
#define M1_DET_HEAD_DIM 64u
/** @brief The fast inverse square root seed (Lomont / Quake III). */
#define M1_DET_RSQRT_SEED 0x5F3759DFu

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

/** @brief 1/sqrt(d), the normative scalar form of hvx_rsqrt_det_sf.
 *         DOMAIN: d a positive normal (the eps floor guarantees it). */
static inline float m1_rsqrt_det(float d) {
  float y = m1_det_float(M1_DET_RSQRT_SEED - (m1_det_bits(d) >> 1));
  const float h = m1_det_mul(d, 0.5f);
  for (int it = 0; it < 3; ++it) {
    float t = m1_det_mul(h, y);
    t = m1_det_mul(t, y);
    t = m1_det_sub(1.5f, t);
    y = m1_det_mul(y, t);
  }
  return y;
}

/**
 * @brief The 32-lane sum of squares of @a chunk floats, reduced as the
 *        HVX does it: lane accumulation, then the pairwise tree.
 *
 * IEEE add is commutative bit for bit, so which of a pair's two operands
 * comes first does not matter, and neither does the direction vror
 * rotates: every lane ends with the same bits.
 */
static inline float m1_sumsq_det(const float *x, uint32_t chunk) {
  float acc[M1_DET_LANES];
  for (uint32_t j = 0; j < M1_DET_LANES; ++j) {
    acc[j] = 0.0f;
  }
  for (uint32_t i = 0; i < chunk; i += M1_DET_LANES) {
    for (uint32_t j = 0; j < M1_DET_LANES; ++j) {
      acc[j] = m1_det_add(acc[j], m1_det_mul(x[i + j], x[i + j]));
    }
  }
  for (uint32_t s = M1_DET_LANES / 2u; s >= 1u; s >>= 1) {
    for (uint32_t j = 0; j < s; ++j) {
      acc[j] = m1_det_add(acc[j], acc[j + s]);
    }
  }
  return acc[0];
}

/**
 * @brief RMSNorm over one chunk: y = (x * rsqrt(mean(x^2) + eps)) * gamma.
 *
 * @param chunk  a power of two and a multiple of 32
 * @return r, the row scale (what the IDL's row_scale reports)
 */
static inline float m1_rmsnorm_chunk_det(const float *x, const float *gamma,
                                         float *y, uint32_t chunk, float eps) {
  const float sum = m1_sumsq_det(x, chunk);
  const float d = m1_det_add(m1_det_mul(sum, 1.0f / (float)chunk), eps);
  const float r = m1_rsqrt_det(d);
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

/** @brief RoPE on one head of 64, in place. cs = cos[32] | sin[32]. */
static inline void m1_rope64_det(float *x, const float *cs) {
  for (uint32_t i = 0; i < M1_DET_HEAD_DIM / 2u; ++i) {
    const float a = x[i], b = x[i + 32], c = cs[i], s = cs[i + 32];
    x[i] = m1_det_sub(m1_det_mul(a, c), m1_det_mul(b, s));
    x[i + 32] = m1_det_add(m1_det_mul(a, s), m1_det_mul(b, c));
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

#endif /* __NNTRAINER_M1_OPS_DET_H__ */
