// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_m1_ops_f32.c
 * @date   27 Sep 2026
 * @brief  The M=1 small ops on HVX, bit-identical to m1_ops_det.h
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Every arithmetic step is a Vsf multiply, add or subtract on whole
 * vectors, with one exception: the RMSNorm row scale r (#164) is the
 * Android CPU's, computed on the scalar core -- 16 sffma chains held in
 * registers (IEEE fused multiply-add, the CPU's fmla), m1_ops_det.h's
 * reduction order, and its integer sqrt / reciprocal, each correctly
 * rounded. HVX has no sf FMA, and an emulated one costs more than the
 * whole kernel (plan 164 section 3.2); r is one scalar per chunk, splat
 * once for the (x * r) * gamma loop.
 * RoPE is the other exception to "f32 inside": it is the Android CPU's
 * fp16 RoPE (#152), every step rounded with hvx_rne16_sf, so the resident
 * attention sees the CPU's q and k bit for bit.
 * Rule 24: no flush-to-zero anywhere, no qf32 (v75/v79 differ); the
 * norms' domain is m1_ops_det.h's (d >= eps, normal). The router is the
 * third exception (#132 PR 2): the Android CPU's order, E fused sffma
 * chains on the scalar core and the spec's own sigmoid and selection
 * (m1_router_cpu_sigmoid / _pick). Gemma's softmax router (plan 201 S4) is
 * plain Vsf chains, 32 experts a vector, and the spec's own scalar
 * softmax / top-k / renormalise (m1_router_softmax_pick).
 */

#include "hvx_m1_ops_f32.h"

#include <stdlib.h>
#include <string.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_conv_gate_f32.h"
#include "hvx_convert.h"
#include "hvx_swiglu_det.h"
#include "m1_ops_det.h"

#define LANES 32u

/** @brief The CPU's row scale r of one chunk (m1_norm_scale_det of
 *         m1_sumsq_cpu_det): the 16 chains are sffma (Q6_R_sfmpyacc_RR,
 *         one rounding, what fmaf is), in registers; the reduction and the
 *         integer sqrt / reciprocal are the spec's own functions. */
static inline float hvx_rmsnorm_scale(const float *x, uint32_t chunk,
                                      float eps) {
  float acc[M1_DET_NORM_CHAINS];
  for (uint32_t j = 0; j < M1_DET_NORM_CHAINS; ++j) {
    acc[j] = 0.0f;
  }
  for (uint32_t i = 0; i < chunk; i += M1_DET_NORM_CHAINS) {
    for (uint32_t j = 0; j < M1_DET_NORM_CHAINS; ++j) {
      acc[j] = Q6_R_sfmpyacc_RR(acc[j], x[i + j], x[i + j]);
    }
  }
  return m1_norm_scale_det(m1_norm_reduce_det(acc), chunk, eps);
}

/** @brief [plan 201 S4] N1's row scale: the largest |bits|, then q =
 *         a 2^11 + b on 32 word lanes and the integer sums of a^2, a b and
 *         b^2 (each term <= 2^22, so M1_N1_BLOCK_VECS of them fit a u32
 *         lane), each block's lanes added into u64s: S = 2^22 A + 2^12 B +
 *         C = sum q^2 exactly, so any lane count or order gives
 *         m1_rmsnorm_n1_chunk_det's S. The rest is the spec's own scalar
 *         m1_n1_norm_scale_det. */
static inline float hvx_rmsnorm_n1_scale(const float *x, uint32_t chunk,
                                         float eps) {
  const HVX_UVector *vx = (const HVX_UVector *)x;
  const uint32_t nvec = chunk / LANES;
  const HVX_Vector mag = Q6_V_vsplat_R(0x7FFFFFFF);
  const HVX_Vector magic = Q6_V_vsplat_R(0x4B400000);
  const HVX_Vector lo11 = Q6_V_vsplat_R(0x7FF);
  HVX_Vector vmax = Q6_V_vzero();
  uint32_t amax = 0u;
  uint64_t A = 0u, B = 0u, C = 0u;
  for (uint32_t i = 0; i < nvec; ++i) {
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vand_VV(vx[i], mag));
  }
  const uint32_t *mw = (const uint32_t *)&vmax;
  for (uint32_t l = 0; l < LANES; ++l) {
    amax = mw[l] > amax ? mw[l] : amax;
  }
  const int k = m1_n1_shift_det(amax);
  const HVX_Vector p = hvx_splat_sf(m1_n1_pow2_det(k));
  for (uint32_t b0 = 0; b0 < nvec; b0 += M1_N1_BLOCK_VECS) {
    const uint32_t e =
      b0 + M1_N1_BLOCK_VECS < nvec ? b0 + M1_N1_BLOCK_VECS : nvec;
    HVX_Vector va = Q6_V_vzero(), vb = Q6_V_vzero(), vc = Q6_V_vzero();
    for (uint32_t i = b0; i < e; ++i) {
      const HVX_Vector v = Q6_Vsf_vmpy_VsfVsf(Q6_V_vand_VV(vx[i], mag), p);
      const HVX_Vector q = Q6_Vw_vsub_VwVw(Q6_Vsf_vadd_VsfVsf(v, magic), magic);
      /* a, b <= 2^11 sit in the low halfword: w * w.uh is exact */
      const HVX_Vector a = Q6_Vuw_vlsr_VuwR(q, 11);
      const HVX_Vector lo = Q6_V_vand_VV(q, lo11);
      va = Q6_Vw_vadd_VwVw(va, Q6_Vw_vmpyie_VwVuh(a, a));
      vb = Q6_Vw_vadd_VwVw(vb, Q6_Vw_vmpyie_VwVuh(a, lo));
      vc = Q6_Vw_vadd_VwVw(vc, Q6_Vw_vmpyie_VwVuh(lo, lo));
    }
    const uint32_t *aw = (const uint32_t *)&va, *bw = (const uint32_t *)&vb,
                   *cw = (const uint32_t *)&vc;
    for (uint32_t l = 0; l < LANES; ++l) {
      A += aw[l];
      B += bw[l];
      C += cw[l];
    }
  }
  return m1_n1_norm_scale_det((A << 22) + (B << 12) + C, k, chunk, eps);
}

static inline void rmsnorm_apply(const float *x, const float *gamma, float *y,
                                 uint32_t n, float scale) {
  const HVX_UVector *vx = (const HVX_UVector *)x;
  const HVX_UVector *vg = (const HVX_UVector *)gamma;
  HVX_UVector *vy = (HVX_UVector *)y;
  const HVX_Vector r = hvx_splat_sf(scale);
  const uint32_t nvec = n / LANES;
  if (gamma) {
    for (uint32_t i = 0; i < nvec; ++i) {
      vy[i] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vx[i], r), vg[i]);
    }
  } else {
    for (uint32_t i = 0; i < nvec; ++i) {
      vy[i] = Q6_Vsf_vmpy_VsfVsf(vx[i], r);
    }
  }
}

void hvx_rmsnorm_apply_f32(const float *x, const float *gamma, float *y,
                           uint32_t n, float scale) {
  if (!x || !y || n % LANES != 0u) {
    return;
  }
  rmsnorm_apply(x, gamma, y, n, scale);
}

static void rmsnorm_rows(const float *x, const float *gamma, float *y,
                         uint32_t n, uint32_t chunk, float eps,
                         float *row_scale_out, int n1) {
  if (!x || !y || chunk == 0u || chunk % LANES != 0u || n % chunk != 0u) {
    return;
  }
  for (uint32_t c = 0; c < n / chunk; ++c) {
    const float rs = n1
                       ? hvx_rmsnorm_n1_scale(x + (size_t)c * chunk, chunk, eps)
                       : hvx_rmsnorm_scale(x + (size_t)c * chunk, chunk, eps);
    if (row_scale_out) {
      row_scale_out[c] = rs;
    }
    rmsnorm_apply(x + (size_t)c * chunk, gamma, y + (size_t)c * chunk, chunk,
                  rs);
  }
}

void hvx_rmsnorm_f32(const float *x, const float *gamma, float *y, uint32_t n,
                     uint32_t chunk, float eps, float *row_scale_out) {
  rmsnorm_rows(x, gamma, y, n, chunk, eps, row_scale_out, 0);
}

typedef struct {
  const float *x, *gamma;
  float *y;
  uint32_t n_q, n_kv, head_dim;
  float eps;
  int norm_v, k_eq_v;
} qkv_rmsnorm_ctx;

/** @brief One query head or one KV pair per job. A pair reads V before
 *  normalizing K, so K==V is safe even in place. Round-robin assignment
 *  balances Gemma's 16 query heads and 8 KV pairs across eight lanes. */
static void qkv_rmsnorm_lane(uint32_t n_threads, uint32_t i, void *v) {
  const qkv_rmsnorm_ctx *c = (const qkv_rmsnorm_ctx *)v;
  const uint32_t hd = c->head_dim;
  for (uint32_t h = i; h < c->n_q + c->n_kv; h += n_threads) {
    const size_t offset = (size_t)h * hd;
    if (h < c->n_q) {
      rmsnorm_rows(c->x + offset, c->gamma, c->y + offset, hd, hd, c->eps, NULL,
                   0);
      continue;
    }
    const float *v_in = c->x + offset + (c->k_eq_v ? 0u : (size_t)c->n_kv * hd);
    float *v_out = c->y + offset + (size_t)c->n_kv * hd;
    if (c->norm_v) {
      rmsnorm_rows(v_in, NULL, v_out, hd, hd, c->eps, NULL, 0);
    } else if (v_in != v_out) {
      memcpy(v_out, v_in, (size_t)hd * sizeof(float));
    }
    rmsnorm_rows(c->x + offset, c->gamma + hd, c->y + offset, hd, hd, c->eps,
                 NULL, 0);
  }
}

void hvx_qkv_rmsnorm_f32(const float *x, const float *gamma, float *y,
                         uint32_t n_q, uint32_t n_kv, uint32_t head_dim,
                         float eps, int norm_v, int k_eq_v,
                         hvx_worker_pool *pool) {
  if (!x || !gamma || !y || head_dim == 0u || head_dim % LANES != 0u) {
    return;
  }
  qkv_rmsnorm_ctx c = {x, gamma, y, n_q, n_kv, head_dim, eps, norm_v, k_eq_v};
  const uint32_t units = n_q + n_kv;
  /** A few narrow heads do not amortize the pool's fork/join. Gemma's
     sixteen query heads have 256 or 512 elements each. */
  if (pool == NULL || units < 8u || head_dim < 256u) {
    qkv_rmsnorm_lane(1u, 0u, &c);
    return;
  }
  hvx_worker_pool_run(pool, qkv_rmsnorm_lane, &c, units);
}

void hvx_rmsnorm_n1_f32(const float *x, const float *gamma, float *y,
                        uint32_t n, uint32_t chunk, float eps,
                        float *row_scale_out) {
  rmsnorm_rows(x, gamma, y, n, chunk, eps, row_scale_out, 1);
}

/** @brief 32 lanes of one head in fp16 (m1_rope_det): a | b halves
 *         rounded, then rne16(rne16(a*c) - rne16(b*s)) and
 *         rne16(rne16(a*s) + rne16(b*c)), the CPU's fmul / fsub / fadd .8h.
 *         @a h is the half head in floats: b is a + h. */
static inline void hvx_rope_lanes(float *x, uint32_t h, HVX_Vector c,
                                  HVX_Vector s) {
  HVX_UVector *va = (HVX_UVector *)x, *vb = (HVX_UVector *)(x + h);
  const HVX_Vector a = hvx_rne16_sf(*va), b = hvx_rne16_sf(*vb);
  *va =
    hvx_rne16_sf(Q6_Vsf_vsub_VsfVsf(hvx_rne16_sf(Q6_Vsf_vmpy_VsfVsf(a, c)),
                                    hvx_rne16_sf(Q6_Vsf_vmpy_VsfVsf(b, s))));
  *vb =
    hvx_rne16_sf(Q6_Vsf_vadd_VsfVsf(hvx_rne16_sf(Q6_Vsf_vmpy_VsfVsf(a, s)),
                                    hvx_rne16_sf(Q6_Vsf_vmpy_VsfVsf(b, c))));
}

void hvx_rope_f32(float *q, uint32_t n_q, float *k, uint32_t n_k,
                  const float *cs, uint32_t head_dim) {
  const uint32_t h = head_dim / 2u;
  if (!cs || (n_q && !q) || (n_k && !k) || head_dim == 0u || h % LANES != 0u) {
    return;
  }
  /** One 32-lane chunk of the half head at a time: the table vectors are
     loaded once per chunk for every head (the CPU's table is (_FP16) of
     the same f32 values). */
  for (uint32_t j = 0; j < h; j += LANES) {
    const HVX_Vector c = hvx_rne16_sf(*(const HVX_UVector *)(cs + j));
    const HVX_Vector s = hvx_rne16_sf(*(const HVX_UVector *)(cs + h + j));
    for (uint32_t i = 0; i < n_q; ++i) {
      hvx_rope_lanes(q + (size_t)i * head_dim + j, h, c, s);
    }
    for (uint32_t i = 0; i < n_k; ++i) {
      hvx_rope_lanes(k + (size_t)i * head_dim + j, h, c, s);
    }
  }
}

void hvx_rope64_f32(float *q, uint32_t n_q, float *k, uint32_t n_k,
                    const float *cs) {
  hvx_rope_f32(q, n_q, k, n_k, cs, 2u * LANES);
}

static void conv_gate_m1_range(const float *abc, float *state3,
                               const float *conv_w, float *out, uint32_t C,
                               uint32_t lo, uint32_t hi) {
  const HVX_UVector *va = (const HVX_UVector *)abc;
  const HVX_UVector *vb = (const HVX_UVector *)(abc + C);
  const HVX_UVector *vc = (const HVX_UVector *)(abc + 2u * C);
  const HVX_UVector *vw0 = (const HVX_UVector *)conv_w;
  HVX_UVector *vg = (HVX_UVector *)(state3 + 2u * C);
  HVX_UVector *vo = (HVX_UVector *)out;
  const float *w1 = conv_w + C, *w2 = conv_w + 2u * C;
  const float *s0 = state3, *s1 = state3 + C;

  /** [#132 E5c] The Android CPU's decode order (m1_conv_gate_det): g = a*c
     into row 2, y = w0*g (both one rounding each, as the vector multiply),
     then the two fused taps y = fma(w1, s1, y), y = fma(w2, s0, y) --
     HVX has no fused f32 multiply-add, so they are the scalar sffma, 2C of
     them -- and out = b*y. The unfused prefill kernel (hvx_conv_gate_f32)
     it used before is not the CPU's order. */
  for (uint32_t i = lo / LANES; i < hi / LANES; ++i) {
    vg[i] = Q6_Vsf_vmpy_VsfVsf(va[i], vc[i]);
    vo[i] = Q6_Vsf_vmpy_VsfVsf(vw0[i], vg[i]);
  }
  /** the vector stores above and the scalar loads below alias out / state3
     through different types: keep the compiler from reordering them */
  __asm__ volatile("" ::: "memory");
  for (uint32_t j = lo; j < hi; ++j) {
    float y = Q6_R_sfmpyacc_RR(out[j], w1[j], s1[j]);
    out[j] = Q6_R_sfmpyacc_RR(y, w2[j], s0[j]);
  }
  __asm__ volatile("" ::: "memory");
  for (uint32_t i = lo / LANES; i < hi / LANES; ++i) {
    vo[i] = Q6_Vsf_vmpy_VsfVsf(vb[i], vo[i]);
  }

  /** Every channel owns its three state entries, so workers can shift
     their disjoint column ranges after the same two fused taps. */
  const size_t bytes = (size_t)(hi - lo) * sizeof(float);
  memcpy(state3 + lo, state3 + C + lo, bytes);
  memcpy(state3 + C + lo, state3 + 2u * C + lo, bytes);
}

void hvx_conv_gate_m1_f32(const float *abc, float *state3, const float *conv_w,
                          float *out, uint32_t C) {
  if (!abc || !state3 || !conv_w || !out || C == 0u || C % LANES != 0u) {
    return;
  }
  conv_gate_m1_range(abc, state3, conv_w, out, C, 0u, C);
}

typedef struct {
  const float *abc, *conv_w;
  float *state3, *out;
  uint32_t C;
} conv_gate_m1_ctx;

static void conv_gate_m1_lane(uint32_t n_threads, uint32_t i, void *v) {
  const conv_gate_m1_ctx *c = (const conv_gate_m1_ctx *)v;
  const uint32_t nvec = c->C / LANES;
  const uint32_t lo = (uint32_t)((uint64_t)nvec * i / n_threads) * LANES;
  const uint32_t hi = (uint32_t)((uint64_t)nvec * (i + 1u) / n_threads) * LANES;
  conv_gate_m1_range(c->abc, c->state3, c->conv_w, c->out, c->C, lo, hi);
}

void hvx_conv_gate_m1_f32_parallel(const float *abc, float *state3,
                                   const float *conv_w, float *out, uint32_t C,
                                   hvx_worker_pool *pool) {
  if (!abc || !state3 || !conv_w || !out || C == 0u || C % LANES != 0u) {
    return;
  }
  conv_gate_m1_ctx c = {abc, conv_w, state3, out, C};
  hvx_worker_pool_run(pool, conv_gate_m1_lane, &c, C / LANES);
}

/** @brief l2fetch of the next weight block: 128 rows of 128 B (16 KiB) at
 *         a time, a hint only -- the bits do not depend on it. */
#define ROUTER_PF_ROWS 128u

/** @brief Router chains run at once: ROUTER_CHAINS accumulators stay in
 *  registers across the k loop (the first sitting's 32-entry array spilled
 *  to memory, 427784 pcycles/op against the vector router's 34100). */
#ifndef ROUTER_CHAINS
#define ROUTER_CHAINS 8u
#endif
/* a lane writes acc[e0 .. e0 + ROUTER_CHAINS) of 32 and loads row pairs */
_Static_assert(32u % ROUTER_CHAINS == 0u && ROUTER_CHAINS % 2u == 0u,
               "ROUTER_CHAINS must divide 32 and be even");
/** @brief L1 prefetch distance of the chain loop, in rows (a hint: the
 *  bits do not depend on it). ISS, one 8-chain group at K = 2048: 47.7k
 *  pcycles without it, 15.7k at 4 / 8 / 16 rows, 25.0k at 32. */
#ifndef ROUTER_L1_AHEAD
#define ROUTER_L1_AHEAD 8u
#endif

/** @brief Experts e0 .. e0 + ROUTER_CHAINS - 1 of the w32 rows: one fused
 *         chain each over k in order (the CPU's sgemv_n fmadd). */
/** @brief One k of the chains: acc[j] = fma(x[k], w32[k][e0 + j], acc[j]).
 *  w32 is at least 8-byte aligned (malloc's, the graph's memalign), a row
 *  128 bytes, e0 a multiple of 8 floats: paired loads. */
#define ROUTER_STEP(k)                                                         \
  do {                                                                         \
    const float xk_ = x[(k)];                                                  \
    const float *row_ = (const float *)__builtin_assume_aligned(               \
      w32 + (size_t)(k)*stride + e0, 8);                                       \
    for (uint32_t j_ = 0; j_ < ROUTER_CHAINS; ++j_) {                          \
      acc[j_] = Q6_R_sfmpyacc_RR(acc[j_], xk_, row_[j_]);                      \
    }                                                                          \
  } while (0)

static inline __attribute__((always_inline)) void
router_chains(const float *x, const float *w32, uint32_t K, uint32_t stride,
              uint32_t e0, float *out) {
  float acc[ROUTER_CHAINS];
  for (uint32_t j = 0; j < ROUTER_CHAINS; ++j) {
    acc[j] = 0.0f;
  }
  for (uint32_t k0 = 0; k0 < K; k0 += ROUTER_PF_ROWS) {
    const uint32_t k1 = k0 + ROUTER_PF_ROWS < K ? k0 + ROUTER_PF_ROWS : K;
#if defined(__hexagon__)
    if (e0 == 0u && k1 < K) {
      /* the next 16 KiB; Rtt: [47:32] stride, [31:16] width, [15:0] height */
      Q6_l2fetch_AP((void *)(w32 + (size_t)k1 * stride),
                    ((uint64_t)(stride * 4u) << 32) |
                      ((uint64_t)(stride * 4u) << 16) |
                      (uint64_t)ROUTER_PF_ROWS);
    }
#endif
    /** rows up to kpf prefetch the row ROUTER_L1_AHEAD ahead into L1
       (each k reads a new line, and the loop was bound by that miss:
       #132 S3, 65528 pcycles/op); the last ROUTER_L1_AHEAD rows do not,
       so the hint never leaves the buffer and the loop has no test */
    const uint32_t kpf =
      K > ROUTER_L1_AHEAD
        ? (k1 < K - ROUTER_L1_AHEAD ? k1 : K - ROUTER_L1_AHEAD)
        : k0;
    uint32_t k = k0;
    for (; k < kpf; ++k) {
#if defined(__hexagon__)
      Q6_dcfetch_A((void *)(w32 + (size_t)(k + ROUTER_L1_AHEAD) * stride + e0));
#endif
      ROUTER_STEP(k);
    }
    for (; k < k1; ++k) {
      ROUTER_STEP(k);
    }
  }
  for (uint32_t j = 0; j < ROUTER_CHAINS; ++j) {
    out[j] = acc[j];
  }
}

typedef struct {
  const float *x, *w32, *bias;
  uint32_t K, E;
  float *acc, *sig, *score;
} router_ctx;

/** @brief Pool lane i: chain groups i, i + n, ... of ROUTER_CHAINS, then
 *         the sigmoids of the experts it owns (the expf port and the
 *         integer divide are the costly scalar part, so they spread too). */
static void router_lane(uint32_t n, uint32_t i, void *v) {
  const router_ctx *c = (const router_ctx *)v;
  for (uint32_t e0 = i * ROUTER_CHAINS; e0 < c->E; e0 += n * ROUTER_CHAINS) {
    router_chains(c->x, c->w32, c->K, (c->E + 31u) / 32u * 32u, e0,
                  c->acc + e0);
    for (uint32_t e = e0; e < e0 + ROUTER_CHAINS && e < c->E; ++e) {
      m1_router_cpu_sigmoid(c->acc[e], c->bias[e], &c->sig[e], &c->score[e]);
    }
  }
}

void hvx_router_topk_f32(const float *x, const float *w32, const float *bias,
                         uint32_t K, uint32_t E, uint32_t top_k, float *logits,
                         uint32_t *sel, float *weight, hvx_worker_pool *pool) {
  float acc[M1_DET_ROUTER_MAX_E], sig[M1_DET_ROUTER_MAX_E],
    score[M1_DET_ROUTER_MAX_E];
  if (!x || !w32 || !bias || !logits || !sel || !weight || K == 0u || E == 0u ||
      E > M1_DET_ROUTER_MAX_E || top_k == 0u || top_k > E) {
    return;
  }
  /** [#132 PR 2] the CPU's sgemv_n: one fused chain per expert in k order
     (m1_router_cpu_det), ROUTER_CHAINS of them per pool lane on the scalar
     cores; lanes >= E are the padding's zero columns, computed and dropped.
     The bits do not depend on the lane count. */
  router_ctx c = {x, w32, bias, K, E, acc, sig, score};
  hvx_worker_pool_run(pool, router_lane, &c,
                      (E + ROUTER_CHAINS - 1u) / ROUTER_CHAINS);
  memcpy(logits, acc, (size_t)E * sizeof(float));
  m1_router_cpu_pick(sig, score, E, top_k, sel, weight);
}

typedef struct {
  const float *x, *scale, *bias;
  const int32_t *offset;
  const uint8_t *w;
  uint32_t K, E;
  float *acc, *sig, *score;
} router_wh_ctx;

/** @brief Decode eight packed expert columns together, retaining one fused
 * k-ordered FP32 chain per expert. No unpacked weight copy is retained. */
static void router_wh_lane(uint32_t n, uint32_t lane, void *v) {
  const router_wh_ctx *c = (const router_wh_ctx *)v;
  for (uint32_t e0 = lane * ROUTER_CHAINS; e0 < c->E; e0 += n * ROUTER_CHAINS) {
    float acc[ROUTER_CHAINS] = {0};
    for (uint32_t k = 0; k < c->K; ++k) {
      const uint32_t r = k % 32u;
      const size_t tile = ((size_t)(k / 32u) * (c->E / 32u) + e0 / 32u) * 512u;
      const uint8_t *packed =
        c->w + tile + (r / 8u) * 128u + (e0 % 32u) * 4u + r % 4u;
      const float x = c->x[k];
#pragma clang loop unroll(full)
      for (uint32_t j = 0; j < ROUTER_CHAINS; ++j) {
        int q = (packed[j * 4u] >> (4u * ((r / 4u) % 2u))) & 15u;
        q = q >= 8 ? q - 16 : q;
        acc[j] = Q6_R_sfmpyacc_RR(acc[j], x, (float)(q + c->offset[e0 + j]));
      }
    }
    for (uint32_t j = 0; j < ROUTER_CHAINS; ++j) {
      const uint32_t e = e0 + j;
      c->acc[e] = acc[j] * c->scale[e];
      m1_router_cpu_sigmoid(c->acc[e], c->bias[e], &c->sig[e], &c->score[e]);
    }
  }
}

void hvx_router_topk_wh(const float *x, const uint8_t *w, const float *bias,
                        uint32_t K, uint32_t E, uint32_t top_k, float *logits,
                        uint32_t *sel, float *weight, hvx_worker_pool *pool) {
  float acc[M1_DET_ROUTER_MAX_E], sig[M1_DET_ROUTER_MAX_E],
    score[M1_DET_ROUTER_MAX_E];
  if (!x || !w || !bias || !logits || !sel || !weight || !K || K % 32u || !E ||
      E % 32u || E > M1_DET_ROUTER_MAX_E || !top_k || top_k > E)
    return;
  const float *scale = (const float *)(w + (size_t)K * E / 2u);
  const int32_t *offset = (const int32_t *)(scale + 2u * E);
  router_wh_ctx c = {x, scale, bias, offset, w, K, E, acc, sig, score};
  hvx_worker_pool_run(pool, router_wh_lane, &c, E / ROUTER_CHAINS);
  memcpy(logits, acc, (size_t)E * sizeof(float));
  m1_router_cpu_pick(sig, score, E, top_k, sel, weight);
}

/* ---- [plan 201 S4] Gemma 4's softmax router ----------------------------- */

typedef struct {
  const float *x, *w;
  uint32_t K, Ep;
  float *acc;
} router_sm_ctx;

/** @brief [#261 lever 1] l2fetch rows of one 32-expert block's column:
 *         `rows` lines of 128 B at a stride of Ep floats. Without it every
 *         k of the chain below reads a new line and waits for DDR (device:
 *         281 879 pcycles/op at 2816 x 128, 100 a step). A hint only: the
 *         chain and its bits do not depend on it; HVX loads bypass L1, so
 *         there is no dcfetch. ISS v79, one thread, cold 2816 x 128: 298 k
 *         pcycles without, 153 k with it (64 rows of lead 148 k, 256 166 k;
 *         ROUTER_PF_ROWS keeps the longer lead for the device's DDR). */
static inline void router_sm_l2fetch(const float *p, uint32_t Ep,
                                     uint32_t rows) {
#if defined(__hexagon__)
  /* Rtt: [47:32] stride, [31:16] width, [15:0] height */
  Q6_l2fetch_AP((void *)p, ((uint64_t)(Ep * 4u) << 32) |
                             ((uint64_t)(LANES * 4u) << 16) | (uint64_t)rows);
#else
  (void)p;
  (void)Ep;
  (void)rows;
#endif
}

/** @brief Pool lane i: 32-expert blocks i, i + n, ... -- one Vsf chain
 *         each over k in order, acc = acc + x[k] * W[k][block] (two
 *         roundings, m1_router_softmax_det's order). */
static void router_sm_lane(uint32_t n, uint32_t i, void *v) {
  const router_sm_ctx *c = (const router_sm_ctx *)v;
  /** [D3] block-major weight (hvx_router_sm_pack_f32): block b is one
     contiguous K x 128 B run, so the l2fetch is a plain stream of lines
     and the lead is 2 x ROUTER_PF_ROWS. */
  for (uint32_t b = i; b < c->Ep / LANES; b += n) {
    const float *col = c->w + (size_t)b * c->K * LANES;
    HVX_Vector acc = Q6_V_vzero();
    for (uint32_t k0 = 0; k0 < c->K; k0 += ROUTER_PF_ROWS) {
      const uint32_t k1 =
        k0 + ROUTER_PF_ROWS < c->K ? k0 + ROUTER_PF_ROWS : c->K;
      if (k0 == 0u) {
        router_sm_l2fetch(
          col, LANES, c->K < 2u * ROUTER_PF_ROWS ? c->K : 2u * ROUTER_PF_ROWS);
      } else if (k1 < c->K) {
        router_sm_l2fetch(col + (size_t)k1 * LANES, LANES,
                          c->K - k1 < ROUTER_PF_ROWS ? c->K - k1
                                                     : ROUTER_PF_ROWS);
      }
      for (uint32_t k = k0; k < k1; ++k) {
        acc = Q6_Vsf_vadd_VsfVsf(
          acc,
          Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(c->x[k]),
                             *(const HVX_UVector *)(col + (size_t)k * LANES)));
      }
    }
    *(HVX_UVector *)(c->acc + (size_t)b * LANES) = acc;
  }
}

void hvx_router_sm_pack_f32(float *dst, const float *w, uint32_t K,
                            uint32_t E) {
  const uint32_t Ep = (E + LANES - 1u) / LANES * LANES;
  for (uint32_t b = 0; b < Ep / LANES; ++b) {
    for (uint32_t k = 0; k < K; ++k) {
      float *d = dst + ((size_t)b * K + k) * LANES;
      for (uint32_t j = 0; j < LANES; ++j) {
        const uint32_t e = b * LANES + j;
        d[j] = e < E ? w[(size_t)k * E + e] : 0.0f;
      }
    }
  }
}

void hvx_router_softmax_topk_f32(const float *x, const float *wp,
                                 const float *pes, uint32_t K, uint32_t E,
                                 uint32_t top_k, float *logits, uint32_t *sel,
                                 float *weight, hvx_worker_pool *pool) {
  float acc[M1_DET_ROUTER_SM_MAX_E];
  const uint32_t Ep = (E + LANES - 1u) / LANES * LANES;
  if (!x || !wp || !pes || !logits || !sel || !weight || K == 0u || E == 0u ||
      E > M1_DET_ROUTER_SM_MAX_E || top_k == 0u || top_k > E) {
    return;
  }
  /** wp is block-major (hvx_router_sm_pack_f32). ponytail: f32 weights,
     1.4 MiB a layer at Gemma's 2816 x 128 read once a token (43 MiB over
     30 layers); f16 / bf16 storage halves it when the router shows in the
     stage timers. */
  router_sm_ctx c = {x, wp, K, Ep, acc};
  hvx_worker_pool_run(pool, router_sm_lane, &c, Ep / LANES);
  memcpy(logits, acc, (size_t)E * sizeof(float));
  m1_router_softmax_pick(acc, pes, E, top_k, sel, weight);
}

/** Storage only: every arithmetic step below remains FP32. Enabled on the
   measured v81 target; other targets retain FP32 unless opted in. */
#ifndef NNTR_ROUTER_BF16_IF_EXACT
#if defined(__HEXAGON_ARCH__) && __HEXAGON_ARCH__ == 81
#define NNTR_ROUTER_BF16_IF_EXACT 1
#else
#define NNTR_ROUTER_BF16_IF_EXACT 0
#endif
#endif

_Static_assert(sizeof(hvx_router_sm_weights) == 128u,
               "router weight payload must remain vector aligned");

hvx_router_sm_weights *hvx_router_sm_weights_create(const float *w, uint32_t K,
                                                    uint32_t E) {
  if (!w || K == 0u || E == 0u || E > M1_DET_ROUTER_SM_MAX_E ||
      K == UINT32_MAX) {
    return NULL;
  }
  const uint32_t Ep = (E + LANES - 1u) / LANES * LANES;
  const uint32_t Kp = (K + 1u) & ~1u;
  const uint64_t count = (uint64_t)K * E;
  if (count > SIZE_MAX / sizeof(float)) {
    return NULL;
  }
  uint32_t bf16 = 0u;
#if NNTR_ROUTER_BF16_IF_EXACT
  /** Check raw bits in vectors: neither eligibility nor packing performs
     a floating-point conversion. Avoid a scalar scan of every weight on
     the first decode token, when all router matrices are registered. */
  const HVX_Vector low = Q6_V_vsplat_R(0xFFFF);
  const HVX_Vector exponent = Q6_V_vsplat_R(0x7F800000);
  const HVX_Vector zero = Q6_V_vzero();
  HVX_Vector bad = zero;
  size_t i = 0;
  for (; i + LANES <= (size_t)count; i += LANES) {
    const HVX_Vector bits = *(const HVX_UVector *)(w + i);
    const HVX_VectorPred nonfinite =
      Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(bits, exponent), exponent);
    bad = Q6_V_vor_VV(bad, Q6_V_vand_VV(bits, low));
    bad = Q6_V_vor_VV(bad, Q6_V_vmux_QVV(nonfinite, low, zero));
  }
  for (uint32_t rot = 64u; rot >= 4u; rot >>= 1u)
    bad = Q6_V_vor_VV(bad, Q6_V_vror_VR(bad, (int)rot));
  bf16 = Q6_R_vextract_VR(bad, 0) == 0;
  for (; bf16 && i < (size_t)count; ++i) {
    uint32_t bits;
    memcpy(&bits, w + i, sizeof(bits));
    if ((bits & 0xFFFFu) != 0u || (bits & 0x7F800000u) == 0x7F800000u)
      bf16 = 0u;
  }
#endif
  const uint64_t bytes = sizeof(hvx_router_sm_weights) +
                         (uint64_t)(bf16 ? Kp : K) * Ep * (bf16 ? 2u : 4u);
  if (bytes > SIZE_MAX) {
    return NULL;
  }
  hvx_router_sm_weights *dst =
    (hvx_router_sm_weights *)memalign(128, (size_t)bytes);
  if (!dst) {
    return NULL;
  }
  memset(dst, 0, sizeof(*dst));
  dst->K = K;
  dst->E = E;
  dst->Ep = Ep;
  dst->Kp = Kp;
  dst->bf16 = bf16;
  if (!bf16) {
    hvx_router_sm_pack_f32((float *)(dst + 1), w, K, E);
    return dst;
  }
  uint16_t *data = (uint16_t *)(dst + 1);
  for (uint32_t b = 0; b < Ep / LANES; ++b) {
    if (E - b * LANES >= LANES) {
      for (uint32_t k = 0; k < Kp; k += 2u) {
        const HVX_Vector row0 =
          *(const HVX_UVector *)(w + (size_t)k * E + b * LANES);
        const HVX_Vector row1 =
          k + 1u < K
            ? *(const HVX_UVector *)(w + (size_t)(k + 1u) * E + b * LANES)
            : Q6_V_vzero();
        *(HVX_Vector *)(data + ((size_t)b * Kp + k) * LANES) =
          Q6_Vh_vpacke_VwVw(Q6_Vuw_vlsr_VuwR(row1, 16),
                            Q6_Vuw_vlsr_VuwR(row0, 16));
      }
      continue;
    }
    for (uint32_t k = 0; k < Kp; ++k) {
      for (uint32_t j = 0; j < LANES; ++j) {
        const uint32_t e = b * LANES + j;
        uint32_t bits = 0u;
        if (k < K && e < E) {
          memcpy(&bits, w + (size_t)k * E + e, sizeof(bits));
        }
        data[((size_t)b * Kp + k) * LANES + j] = (uint16_t)(bits >> 16);
      }
    }
  }
  return dst;
}

typedef struct {
  const float *x;
  const hvx_router_sm_weights *w;
  float *acc;
} router_sm_bf16_ctx;

/** @brief Two adjacent K rows share one aligned 128-byte BF16 load.
 *  Interleaving zero halfwords reconstructs the original FP32 bits. Each expert
 *  retains its k-ascending, separately rounded multiply/add chain. */
static void router_sm_bf16_lane(uint32_t n, uint32_t i, void *v) {
  const router_sm_bf16_ctx *c = (const router_sm_bf16_ctx *)v;
  const hvx_router_sm_weights *w = c->w;
  const uint16_t *data = (const uint16_t *)(w + 1);
  const uint32_t block_rows = 2u * ROUTER_PF_ROWS;
  HVX_Vector identity = Q6_V_vsplat_R(-1);
#if defined(__hexagon__)
  /** Make this integer identity opaque once, outside the hot loop. Its
     AND below requires FP32 bits between the pair's two terms, so the
     compiler cannot keep that intermediate in qf32. */
  __asm__("" : "+v"(identity));
#endif
  for (uint32_t b = i; b < w->Ep / LANES; b += n) {
    const uint16_t *col = data + (size_t)b * w->Kp * LANES;
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t initial_rows =
      w->Kp < 2u * block_rows ? w->Kp : 2u * block_rows;
    router_sm_l2fetch((const float *)col, LANES, initial_rows / 2u);
    for (uint32_t k0 = 0; k0 < w->K; k0 += block_rows) {
      const uint32_t k1 = k0 + block_rows < w->K ? k0 + block_rows : w->K;
      if (k0 != 0u && k1 < w->K) {
        const uint32_t rows = w->Kp - k1 < block_rows ? w->Kp - k1 : block_rows;
        router_sm_l2fetch((const float *)(col + (size_t)k1 * LANES), LANES,
                          rows / 2u);
      }
      const uint32_t pairs = (k1 - k0) / 2u;
      for (uint32_t p = 0; p < pairs; ++p) {
        const uint32_t k = k0 + 2u * p;
        const HVX_VectorPair pair = Q6_W_vshuff_VVR(
          *(const HVX_Vector *)(col + (size_t)k * LANES), Q6_V_vzero(), -2);
        acc = Q6_Vsf_vadd_VsfVsf(
          acc, Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(c->x[k]), Q6_V_lo_W(pair)));
        acc = Q6_V_vand_VV(acc, identity);
        HVX_Vector product =
          Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(c->x[k + 1u]), Q6_V_hi_W(pair));
        product = Q6_V_vand_VV(product, identity);
        acc = Q6_Vsf_vadd_VsfVsf(acc, product);
      }
      const uint32_t k = k0 + 2u * pairs;
      if (k < k1) {
        const HVX_VectorPair pair = Q6_W_vshuff_VVR(
          *(const HVX_Vector *)(col + (size_t)k * LANES), Q6_V_vzero(), -2);
        acc = Q6_Vsf_vadd_VsfVsf(
          acc, Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(c->x[k]), Q6_V_lo_W(pair)));
      }
    }
    *(HVX_UVector *)(c->acc + (size_t)b * LANES) = acc;
  }
}

void hvx_router_softmax_topk_owned_f32(const float *x,
                                       const hvx_router_sm_weights *w,
                                       const float *pes, uint32_t top_k,
                                       float *logits, uint32_t *sel,
                                       float *weight, hvx_worker_pool *pool) {
  if (!w) {
    return;
  }
  if (!w->bf16) {
    hvx_router_softmax_topk_f32(x, (const float *)(w + 1), pes, w->K, w->E,
                                top_k, logits, sel, weight, pool);
    return;
  }
  if (!x || !pes || !logits || !sel || !weight || top_k == 0u || top_k > w->E) {
    return;
  }
  float acc[M1_DET_ROUTER_SM_MAX_E];
  router_sm_bf16_ctx c = {x, w, acc};
  hvx_worker_pool_run(pool, router_sm_bf16_lane, &c, w->Ep / LANES);
  memcpy(logits, acc, (size_t)w->E * sizeof(float));
  m1_router_softmax_pick(acc, pes, w->E, top_k, sel, weight);
}

/* ---- [#132 Part B E5f] SwiGLU over the pool, argmax in one vector pass -- */

typedef struct {
  const float *y, *z;
  float *out;
  uint32_t n;
} swiglu_cpu_ctx;

static void swiglu_cpu_lane(uint32_t n_threads, uint32_t i, void *v) {
  const swiglu_cpu_ctx *c = (const swiglu_cpu_ctx *)v;
  const uint32_t lo = (uint32_t)((uint64_t)c->n * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->n * (i + 1u) / n_threads);
  for (uint32_t k = lo; k < hi; ++k) {
    const float e =
      m1_exp_ps_cpu_det(m1_det_float(m1_det_bits(c->y[k]) ^ 0x80000000u));
    /* one IEEE RN divide (sfrecipa / sffixup), as cpu_det_div_rn rounds */
    volatile float q = c->y[k] / m1_det_add(e, 1.0f);
    c->out[k] = m1_det_mul(q, c->z[k]);
  }
}

void hvx_swiglu_cpu_f32(const float *y, const float *z, float *out, uint32_t n,
                        hvx_worker_pool *pool) {
  swiglu_cpu_ctx c = {y, z, out, n};
  /* 1024 elements a lane at least: a fork / join is a few microseconds */
  const uint32_t units = n / 1024u ? n / 1024u : 1u;
  hvx_worker_pool_run(pool, swiglu_cpu_lane, &c, units);
}

uint32_t hvx_argmax_first_f32(const float *x, uint32_t n) {
  if (n == 0u || n % LANES != 0u) {
    return m1_argmax_first(x, n);
  }
  const HVX_Vector negz = Q6_V_vsplat_R((int32_t)0x80000000);
  const HVX_Vector mag = Q6_V_vsplat_R(0x7FFFFFFF);
  const HVX_Vector inf = Q6_V_vsplat_R(0x7F800000);
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector ones = Q6_V_vsplat_R(-1);
  HVX_Vector best = Q6_V_vsplat_R((int32_t)0x80000000); /* below every key */
  HVX_Vector blk = zero, nan = zero;
  const HVX_UVector *xv = (const HVX_UVector *)x;
  for (uint32_t b = 0; b < n / LANES; ++b) {
    HVX_Vector v = xv[b];
    /* NaN: |bits| > +inf */
    nan =
      Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(v, mag), inf), ones, nan);
    /* -0 compares equal to +0 */
    v = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(v, negz), zero, v);
    /* order-preserving key: negatives flip their magnitude bits */
    const HVX_Vector key =
      Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(zero, v), Q6_V_vxor_VV(v, mag), v);
    const HVX_VectorPred gt = Q6_Q_vcmp_gt_VwVw(key, best); /* strictly */
    best = Q6_V_vmux_QVV(gt, key, best);
    blk = Q6_V_vmux_QVV(gt, Q6_V_vsplat_R((int32_t)b), blk);
  }
  const int32_t *bw = (const int32_t *)&best, *kw = (const int32_t *)&blk,
                *nw = (const int32_t *)&nan;
  int32_t m = bw[0];
  for (uint32_t l = 0; l < LANES; ++l) {
    if (nw[l]) {
      return m1_argmax_first(x, n);
    }
    if (bw[l] > m) {
      m = bw[l];
    }
  }
  uint32_t idx = n;
  for (uint32_t l = 0; l < LANES; ++l) {
    const uint32_t at = (uint32_t)kw[l] * LANES + l;
    if (bw[l] == m && at < idx) {
      idx = at;
    }
  }
  return idx;
}

/* ---- [plan 201 S4] Gemma 4: the dense GeGLU, the logit soft-cap, x * s -- */

void hvx_geglu_f32(const float *gate, const float *up, float *out, uint32_t n) {
  uint32_t i = 0;
  for (; i + LANES <= n; i += LANES) {
    *(HVX_UVector *)(out + i) = hvx_geglu_det_sf(
      *(const HVX_UVector *)(gate + i), *(const HVX_UVector *)(up + i));
  }
  for (; i < n; ++i) {
    out[i] = geglu_det_one(gate[i], up[i]);
  }
}

typedef struct {
  float *x;
  uint32_t n;
  float inv, cap;
} softcap_ctx;

/* m1_softcap_one_det, operation for operation, on a lane's vectors */
static void softcap_lane(uint32_t n_threads, uint32_t i, void *v) {
  const softcap_ctx *c = (const softcap_ctx *)v;
  const uint32_t nvec = c->n / LANES;
  const uint32_t lo = (uint32_t)((uint64_t)nvec * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)nvec * (i + 1u) / n_threads);
  const HVX_Vector inv = hvx_splat_sf(c->inv), cap = hvx_splat_sf(c->cap);
  const HVX_Vector one = hvx_splat_sf(1.0f), two = hvx_splat_sf(2.0f);
  const HVX_Vector m2 = hvx_splat_sf(-2.0f);
  HVX_UVector *xv = (HVX_UVector *)c->x;
  for (uint32_t b = lo; b < hi; ++b) {
    const HVX_Vector a = Q6_Vsf_vmpy_VsfVsf(xv[b], inv);
    const HVX_Vector e = hvx_exp_det_sf(Q6_Vsf_vmpy_VsfVsf(m2, a));
    const HVX_Vector s = hvx_recip_det_sf(Q6_Vsf_vadd_VsfVsf(one, e));
    xv[b] = Q6_Vsf_vmpy_VsfVsf(
      Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(two, s), one), cap);
  }
}

void hvx_softcap_m1_f32(float *x, uint32_t n, float cap,
                        hvx_worker_pool *pool) {
  /* one IEEE RN divide, cpu_det_div_rn's result (hvx_swiglu_cpu_f32) */
  volatile float inv = 1.0f / cap;
  softcap_ctx c = {x, n, inv, cap};
  const uint32_t nvec = n / LANES;
  /* 256 vectors a lane at least: a fork / join is a few microseconds */
  hvx_worker_pool_run(pool, softcap_lane, &c, nvec / 256u ? nvec / 256u : 1u);
  for (uint32_t i = nvec * LANES; i < n; ++i) {
    x[i] = m1_softcap_one_det(x[i], c.inv, cap);
  }
}

void hvx_mul_scalar_f32(float *x, float s, uint32_t n) {
  const HVX_Vector sv = hvx_splat_sf(s);
  uint32_t i = 0;
  for (; i + LANES <= n; i += LANES) {
    *(HVX_UVector *)(x + i) =
      Q6_Vsf_vmpy_VsfVsf(*(const HVX_UVector *)(x + i), sv);
  }
  for (; i < n; ++i) {
    x[i] = m1_det_mul(x[i], s);
  }
}
