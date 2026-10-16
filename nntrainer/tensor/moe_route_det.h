// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   moe_route_det.h
 * @date   03 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Deterministic MoE router (rms_norm, scale, projection, softmax,
 *         top-k, renormalisation, per-expert scale) whose every operation is
 *         an IEEE add / multiply / fma / exact bit operation in a fixed
 *         order, so the host code here and the OpenCL router kernels
 *         (tq2moe_route_norm / _logits / _select) produce the same bits:
 *         no BLAS, no libm exp / sqrt, no driver-dependent division.
 *
 * Order (the device kernels follow it term for term):
 *  - sum of squares: 32 fma lanes (lane l takes i = l, l + 32, ...; the
 *    8-wide tail goes to lanes 0..7), lanes folded as
 *    ((v0+v4)+(v1+v5))+((v2+v6)+(v3+v7)) per 8, then ((h0+h1)+h2)+h3;
 *  - mean = sumsq / H, scale = rsqrt(mean + eps) by three Newton steps from
 *    a fixed bit-trick seed; y = ((x * scale) * norm) * (rscale * hscale);
 *  - logits: per 32-wide segment of the hidden axis an fma chain over the
 *    segment, the segment sums added in order;
 *  - softmax: max, exp(l - max) by a fixed range reduction and polynomial,
 *    sum in expert order, p = ex / sum;
 *  - top-k by rank (larger first, ties to the lower expert id), sum of the
 *    k values in rank order, w = p / s * per_expert_scale.
 * Every division is the correctly rounded quotient built from fma (the same
 * construction as the KV snap's), so it does not depend on the device's
 * division accuracy either.
 */

#ifndef __MOE_ROUTE_DET_H__
#define __MOE_ROUTE_DET_H__
#ifdef __cplusplus

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nntrainer {

// Strict IEEE evaluation whatever the including unit's flags (an
// application may build with -ffast-math): no contraction, no
// reassociation, no reciprocal or approximate-function substitution.
#if defined(__clang__)
#define MOE_DET_NO_CONTRACT                                                    \
  _Pragma("float_control(precise, on)") _Pragma("clang fp contract(off)")
#else
#define MOE_DET_NO_CONTRACT
#endif

/** width of one projection segment (RT_SEG of the device kernel) */
constexpr unsigned int kMoeRouteSeg = 32;

#if defined(__GNUC__) && !defined(__clang__)
#define MOE_DET_ATTR                                                           \
  __attribute__((optimize("no-fast-math", "fp-contract=off")))
#else
#define MOE_DET_ATTR
#endif

/** correctly rounded x / s from fma (Markstein; -0 / s stays -0) */
MOE_DET_ATTR inline float moe_det_div(const float x, const float s) {
  MOE_DET_NO_CONTRACT
  float y = 1.0f / s;
  float e = std::fma(-s, y, 1.0f);
  y = std::fma(e, y, y);
  e = std::fma(-s, y, 1.0f);
  y = std::fma(e, y, y);
  const float q = x * y;
  const float r = std::fma(-s, q, x);
  return r == 0.0f ? q : std::fma(r, y, q);
}

/** 1 / sqrt(a), a > 0: three Newton steps from a fixed seed */
MOE_DET_ATTR inline float moe_det_rsqrt(const float a) {
  MOE_DET_NO_CONTRACT
  uint32_t i;
  std::memcpy(&i, &a, 4);
  i = 0x5f375a86u - (i >> 1);
  float y;
  std::memcpy(&y, &i, 4);
  const float h = 0.5f * a;
  for (int it = 0; it < 3; ++it) {
    const float hy = h * y;
    const float r = std::fma(-hy, y, 1.5f);
    y = y * r;
  }
  return y;
}

/** exp(x) for x <= 0: fixed range reduction and degree-6 polynomial */
MOE_DET_ATTR inline float moe_det_exp(const float x) {
  MOE_DET_NO_CONTRACT
  if (x < -87.0f)
    return 0.0f;
  const float t = x * 1.44269504088896341f;
  const float n = std::nearbyint(t);
  float r = std::fma(-n, 0.693145751953125f, x);
  r = std::fma(-n, 1.428606765330187e-06f, r);
  float p = 1.3888889e-3f;
  p = std::fma(p, r, 8.3333333e-3f);
  p = std::fma(p, r, 4.1666667e-2f);
  p = std::fma(p, r, 1.6666667e-1f);
  p = std::fma(p, r, 0.5f);
  p = std::fma(p, r, 1.0f);
  p = std::fma(p, r, 1.0f);
  return std::ldexp(p, (int)n);
}

/**
 * @brief Route one row.
 * @param x      [H] router input row (fp32)
 * @param norm   [H] router norm weight, or null
 * @param rscale [H] router scale
 * @param hscale hidden^-1/2 (passed in so both sides use the same value)
 * @param W      [H][E] projection
 * @param pes    [E] per-expert scale
 * @param ids    [K] out: selected experts in rank order (larger first)
 * @param w      [K] out: their routing weights
 * @param probs  [E] out (optional): the softmax probabilities
 */
MOE_DET_ATTR inline void
moe_route_det_row(const float *x, unsigned int H, const float *norm,
                  const float *rscale, float hscale, const float *W,
                  const float *pes, unsigned int E, unsigned int K, float eps,
                  uint32_t *ids, float *w, float *probs = nullptr) {
  MOE_DET_NO_CONTRACT
  float acc[32] = {0.0f};
  const unsigned int n32 = H / 32 * 32;
  for (unsigned int l = 0; l < 32; ++l)
    for (unsigned int i = l; i < n32; i += 32)
      acc[l] = std::fma(x[i], x[i], acc[l]);
  for (unsigned int l = 0; l < 8; ++l)
    for (unsigned int i = n32 + l; i + 8 - l <= H; i += 8)
      acc[l] = std::fma(x[i], x[i], acc[l]);
  float hsum[4];
  for (int j = 0; j < 4; ++j) {
    const float *v = acc + 8 * j;
    hsum[j] = ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]));
  }
  const float sumsq = ((hsum[0] + hsum[1]) + hsum[2]) + hsum[3];
  const float mean = moe_det_div(sumsq, (float)H);
  const float scale = moe_det_rsqrt(mean + eps);
  std::vector<float> y(H);
  for (unsigned int i = 0; i < H; ++i) {
    float v = x[i] * scale;
    if (norm != nullptr)
      v = v * norm[i];
    y[i] = v * (rscale[i] * hscale);
  }
  std::vector<float> l(E), ex(E), p(E), a(E);
  const unsigned int nseg = (H + kMoeRouteSeg - 1) / kMoeRouteSeg;
  // Loops ordered expert-innermost: every expert keeps its own fma chain in
  // the same order, and the inner loop over contiguous W rows vectorises
  // without changing any lane's arithmetic.
  for (unsigned int s = 0; s < nseg; ++s) {
    const unsigned int f0 = s * kMoeRouteSeg;
    const unsigned int f1 = std::min(H, f0 + kMoeRouteSeg);
    for (unsigned int e = 0; e < E; ++e)
      a[e] = 0.0f;
    for (unsigned int f = f0; f < f1; ++f) {
      const float yf = y[f];
      const float *Wr = W + (size_t)f * E;
      for (unsigned int e = 0; e < E; ++e)
        a[e] = std::fma(yf, Wr[e], a[e]);
    }
    if (s == 0)
      for (unsigned int e = 0; e < E; ++e)
        l[e] = a[e];
    else
      for (unsigned int e = 0; e < E; ++e)
        l[e] = l[e] + a[e];
  }
  float mx = l[0];
  for (unsigned int e = 1; e < E; ++e)
    mx = std::fmax(mx, l[e]);
  float sum = 0.0f;
  for (unsigned int e = 0; e < E; ++e) {
    ex[e] = moe_det_exp(l[e] - mx);
    sum = sum + ex[e];
  }
  for (unsigned int e = 0; e < E; ++e)
    p[e] = moe_det_div(ex[e], sum);
  if (probs != nullptr)
    std::memcpy(probs, p.data(), E * sizeof(float));
  // rank: larger first, ties to the lower id
  for (unsigned int e = 0; e < E; ++e) {
    unsigned int rank = 0;
    for (unsigned int i = 0; i < E; ++i)
      rank += (p[i] > p[e] || (p[i] == p[e] && i < e)) ? 1u : 0u;
    if (rank < K) {
      ids[rank] = e;
      w[rank] = p[e];
    }
  }
  float s = 0.0f;
  for (unsigned int j = 0; j < K; ++j)
    s = s + w[j];
  for (unsigned int j = 0; j < K; ++j)
    w[j] = moe_det_div(w[j], s) * pes[ids[j]];
}

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MOE_ROUTE_DET_H__ */
