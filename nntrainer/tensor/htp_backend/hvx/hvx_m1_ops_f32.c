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

#include <string.h>

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

static void rmsnorm_rows(const float *x, const float *gamma, float *y,
                         uint32_t n, uint32_t chunk, float eps,
                         float *row_scale_out, int n1) {
  if (!x || !y || chunk == 0u || chunk % LANES != 0u || n % chunk != 0u) {
    return;
  }
  const uint32_t nvec = chunk / LANES;

  for (uint32_t c = 0; c < n / chunk; ++c) {
    const HVX_UVector *vx = (const HVX_UVector *)(x + (size_t)c * chunk);
    const HVX_UVector *vg = (const HVX_UVector *)gamma;
    HVX_UVector *vy = (HVX_UVector *)(y + (size_t)c * chunk);

    const float rs = n1
                       ? hvx_rmsnorm_n1_scale(x + (size_t)c * chunk, chunk, eps)
                       : hvx_rmsnorm_scale(x + (size_t)c * chunk, chunk, eps);
    if (row_scale_out) {
      row_scale_out[c] = rs;
    }
    const HVX_Vector r = hvx_splat_sf(rs);
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
}

void hvx_rmsnorm_f32(const float *x, const float *gamma, float *y, uint32_t n,
                     uint32_t chunk, float eps, float *row_scale_out) {
  rmsnorm_rows(x, gamma, y, n, chunk, eps, row_scale_out, 0);
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
  /* One 32-lane chunk of the half head at a time: the table vectors are
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

void hvx_conv_gate_m1_f32(const float *abc, float *state3, const float *conv_w,
                          float *out, uint32_t C) {
  if (!abc || !state3 || !conv_w || !out || C == 0u || C % LANES != 0u) {
    return;
  }
  const HVX_UVector *va = (const HVX_UVector *)abc;
  const HVX_UVector *vb = (const HVX_UVector *)(abc + C);
  const HVX_UVector *vc = (const HVX_UVector *)(abc + 2u * C);
  const HVX_UVector *vw0 = (const HVX_UVector *)conv_w;
  HVX_UVector *vg = (HVX_UVector *)(state3 + 2u * C);
  HVX_UVector *vo = (HVX_UVector *)out;
  const float *w1 = conv_w + C, *w2 = conv_w + 2u * C;
  const float *s0 = state3, *s1 = state3 + C;
  const uint32_t nvec = C / LANES;

  /* [#132 E5c] The Android CPU's decode order (m1_conv_gate_det): g = a*c
     into row 2, y = w0*g (both one rounding each, as the vector multiply),
     then the two fused taps y = fma(w1, s1, y), y = fma(w2, s0, y) --
     HVX has no fused f32 multiply-add, so they are the scalar sffma, 2C of
     them -- and out = b*y. The unfused prefill kernel (hvx_conv_gate_f32)
     it used before is not the CPU's order. */
  for (uint32_t i = 0; i < nvec; ++i) {
    vg[i] = Q6_Vsf_vmpy_VsfVsf(va[i], vc[i]);
    vo[i] = Q6_Vsf_vmpy_VsfVsf(vw0[i], vg[i]);
  }
  /* the vector stores above and the scalar loads below alias out / state3
     through different types: keep the compiler from reordering them */
  __asm__ volatile("" ::: "memory");
  for (uint32_t j = 0; j < C; ++j) {
    float y = Q6_R_sfmpyacc_RR(out[j], w1[j], s1[j]);
    out[j] = Q6_R_sfmpyacc_RR(y, w2[j], s0[j]);
  }
  __asm__ volatile("" ::: "memory");
  for (uint32_t i = 0; i < nvec; ++i) {
    vo[i] = Q6_Vsf_vmpy_VsfVsf(vb[i], vo[i]);
  }

  /* state <- x_{t-1} | g. ponytail: two 8 KiB copies per token at C = 2048
     (x 18 conv layers, roughly 1-2 us); a ring of three row pointers
     removes them if #85's per-op pcycles ever show it. */
  memcpy(state3, state3 + C, (size_t)C * sizeof(float));
  memcpy(state3 + C, state3 + 2u * C, (size_t)C * sizeof(float));
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
      w32 + (size_t)(k)*LANES + e0, 8);                                        \
    for (uint32_t j_ = 0; j_ < ROUTER_CHAINS; ++j_) {                          \
      acc[j_] = Q6_R_sfmpyacc_RR(acc[j_], xk_, row_[j_]);                      \
    }                                                                          \
  } while (0)

static inline __attribute__((always_inline)) void
router_chains(const float *x, const float *w32, uint32_t K, uint32_t e0,
              float *out) {
  float acc[ROUTER_CHAINS];
  for (uint32_t j = 0; j < ROUTER_CHAINS; ++j) {
    acc[j] = 0.0f;
  }
  for (uint32_t k0 = 0; k0 < K; k0 += ROUTER_PF_ROWS) {
    const uint32_t k1 = k0 + ROUTER_PF_ROWS < K ? k0 + ROUTER_PF_ROWS : K;
#if defined(__hexagon__)
    if (e0 == 0u && k1 < K) {
      /* the next 16 KiB; Rtt: [47:32] stride, [31:16] width, [15:0] height */
      Q6_l2fetch_AP((void *)(w32 + (size_t)k1 * LANES),
                    ((uint64_t)(LANES * 4u) << 32) |
                      ((uint64_t)(LANES * 4u) << 16) |
                      (uint64_t)ROUTER_PF_ROWS);
    }
#endif
    /* rows up to kpf prefetch the row ROUTER_L1_AHEAD ahead into L1
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
      Q6_dcfetch_A((void *)(w32 + (size_t)(k + ROUTER_L1_AHEAD) * LANES + e0));
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
    router_chains(c->x, c->w32, c->K, e0, c->acc + e0);
    for (uint32_t e = e0; e < e0 + ROUTER_CHAINS && e < c->E; ++e) {
      m1_router_cpu_sigmoid(c->acc[e], c->bias[e], &c->sig[e], &c->score[e]);
    }
  }
}

void hvx_router_topk_f32(const float *x, const float *w32, const float *bias,
                         uint32_t K, uint32_t E, uint32_t top_k, float *logits,
                         uint32_t *sel, float *weight, hvx_worker_pool *pool) {
  float acc[LANES], sig[LANES], score[LANES];
  if (!x || !w32 || !bias || !logits || !sel || !weight || K == 0u || E == 0u ||
      E > LANES || top_k == 0u || top_k > E) {
    return;
  }
  /* [#132 PR 2] the CPU's sgemv_n: one fused chain per expert in k order
     (m1_router_cpu_det), ROUTER_CHAINS of them per pool lane on the scalar
     cores; lanes >= E are the padding's zero columns, computed and dropped.
     The bits do not depend on the lane count. */
  router_ctx c = {x, w32, bias, K, E, acc, sig, score};
  hvx_worker_pool_run(pool, router_lane, &c,
                      (E + ROUTER_CHAINS - 1u) / ROUTER_CHAINS);
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
  for (uint32_t b = i; b < c->Ep / LANES; b += n) {
    const float *col = c->w + (size_t)b * LANES;
    HVX_Vector acc = Q6_V_vzero();
    for (uint32_t k0 = 0; k0 < c->K; k0 += ROUTER_PF_ROWS) {
      const uint32_t k1 =
        k0 + ROUTER_PF_ROWS < c->K ? k0 + ROUTER_PF_ROWS : c->K;
      if (k0 == 0u) {
        router_sm_l2fetch(col, c->Ep, k1);
      }
      if (k1 < c->K) {
        router_sm_l2fetch(col + (size_t)k1 * c->Ep, c->Ep,
                          c->K - k1 < ROUTER_PF_ROWS ? c->K - k1
                                                     : ROUTER_PF_ROWS);
      }
      for (uint32_t k = k0; k < k1; ++k) {
        acc = Q6_Vsf_vadd_VsfVsf(
          acc,
          Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(c->x[k]),
                             *(const HVX_UVector *)(col + (size_t)k * c->Ep)));
      }
    }
    *(HVX_UVector *)(c->acc + (size_t)b * LANES) = acc;
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
  /* ponytail: f32 weights, 1.4 MiB a layer at Gemma's 2816 x 128 read once
     a token (43 MiB over 30 layers); f16 / bf16 storage halves it when the
     router shows in the stage timers. */
  router_sm_ctx c = {x, wp, K, Ep, acc};
  hvx_worker_pool_run(pool, router_sm_lane, &c, Ep / LANES);
  memcpy(logits, acc, (size_t)E * sizeof(float));
  m1_router_softmax_pick(acc, pes, E, top_k, sel, weight);
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

void hvx_softcap_f32(float *x, uint32_t n, float cap, hvx_worker_pool *pool) {
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
