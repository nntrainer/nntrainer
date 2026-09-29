// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_attn_m1_f32.c
 * @date   27 Sep 2026
 * @brief  Decode attention at m=1 on HVX over the session's f32 KV cache,
 *         bit-identical to attn_m1_det.h
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Per unit (a kv head and a pair of its q heads, or the whole group when
 * gqa is odd), the spec's steps in its order on fp16 values held in f32:
 *   q16     the unit's q rows rounded (hvx_rne16_sf) into a local block
 *   scores  one 32-position block at a time, per q head: eight
 *           accumulators acc[l] over d = 8 blk + l, acc[l] = fma16(acc[l],
 *           splat(q16[d]), Kt[d][block]) -- the CPU's float16x8_t lanes
 *           as eight vectors whose lanes are positions -- then the
 *           vpaddq tree, 0 + t, * scale, each rounded; stored to the q
 *           head's scratch row, the block's running vmax over live lanes
 *   softmax m = rotate-tree max + 0; per block the exp-table index of
 *           rne16(s - m) (vector); then on the scalar core, per position
 *           in order, e = table[index] and l = rne16(l + e) -- the CPU's
 *           sequential fp16 sum, about 1 k scalar steps per q head; then
 *           per block e / l as div16 (vector: recip_det and an exact
 *           midpoint check, below)
 *   PV      for p ascending, o[g][vec] = fma16(o[g][vec], splat(e[g][p]),
 *           V[p][vec]); out = o
 * fma16 is attn_m1_det_fma16 per lane: the exact product, TwoSum, the
 * round-to-odd fix-up in word ops, then hvx_rne16_sf -- about 20 vector
 * ops where the f32 kernel had 2 (plan 152 section 3.3's cost).
 * Rule 24: plain Vsf add / sub / mul and word ops, no qf32, no hf, no
 * flush-to-zero; after the entry rounding every sf operand is a normal f32
 * or zero (fp16 values are >= 2^-24, their products >= 2^-48). The one
 * exception is the caller's raw q / k / v entering hvx_rne16_sf, which may
 * be f32 subnormals: C absorbs them, so the result is the same +-0
 * whether the hardware flushes that input or not. The masked lanes of the last
 * block are filled with -FLT_MAX for the max rather than -inf, and zeroed
 * before the divide.
 *
 * ponytail: the q splats are re-formed per block (a scalar load and a
 * splat per d); a per-head splat table would trade 32 KiB of writes for
 * them, worth it only if the phase words say the score loop is not
 * compute-bound (it is: fma16 dominates).
 */

#include "hvx_attn_m1_f32.h"

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "attn_m1_det.h"
#include "hvx_convert.h"
#include "hvx_swiglu_det.h"

/** @brief f32 lanes per HVX vector: one position block. */
#define LANES 32u
/** @brief Vector length in bytes, the cache alignment. */
#define VLEN 128u
/** @brief The mask fill for the max: the most negative finite f32. */
#define NEG_FLT_MAX_BITS 0xFF7FFFFFu
/** @brief Upper bound on q heads per kv head (the q16 block and the PV
 *         accumulators: 2 * gqa vectors). */
#define MAX_GQA 8u
/** @brief The one head_dim: the CPU order's eight accumulators over d and
 *         the resident RoPE's head size (#152). */
#define HD 64u
/** @brief Vectors per head row. */
#define HD_VEC (HD / LANES)

hvx_attn_m1_ctx *hvx_attn_m1_create(uint32_t n_layers, uint32_t n_kv,
                                    uint32_t gqa, uint32_t head_dim,
                                    uint32_t max_seq, hvx_worker_pool *pool,
                                    int *err) {
  int rc = AEE_SUCCESS;
  hvx_attn_m1_ctx *ctx = NULL;
  if (n_layers == 0u || n_kv == 0u || gqa == 0u || gqa > MAX_GQA ||
      head_dim != HD || max_seq == 0u || max_seq % LANES != 0u) {
    rc = AEE_EINVALIDFORMAT;
    goto out;
  }
  /* size_t is 32 bits on the DSP: bound the cache before multiplying. */
  const uint64_t cache_floats = (uint64_t)n_layers * n_kv * head_dim * max_seq;
  const uint64_t scratch_floats = (uint64_t)n_kv * gqa * max_seq;
  if (cache_floats * sizeof(float) >= (1ull << 31) ||
      scratch_floats * sizeof(float) >= (1ull << 31)) {
    rc = AEE_EINVALIDFORMAT;
    goto out;
  }
  ctx = (hvx_attn_m1_ctx *)calloc(1, sizeof(*ctx));
  if (!ctx) {
    rc = AEE_ENOMEMORY;
    goto out;
  }
  ctx->n_layers = n_layers;
  ctx->n_kv = n_kv;
  ctx->gqa = gqa;
  ctx->head_dim = head_dim;
  ctx->max_seq = max_seq;
  ctx->cache_floats = (size_t)cache_floats;
  ctx->pool = pool;
  ctx->kv_len = (uint32_t *)calloc(n_layers, sizeof(uint32_t));
  ctx->kt = (float *)memalign(VLEN, (size_t)cache_floats * sizeof(float));
  ctx->v = (float *)memalign(VLEN, (size_t)cache_floats * sizeof(float));
  ctx->scratch =
    (float *)memalign(VLEN, (size_t)scratch_floats * sizeof(float));
  ctx->exp_tab = (float *)malloc(ATTN_M1_DET_EXP_N * sizeof(float));
  if (!ctx->kv_len || !ctx->kt || !ctx->v || !ctx->scratch || !ctx->exp_tab) {
    rc = AEE_ENOMEMORY;
    hvx_attn_m1_free(ctx);
    ctx = NULL;
    goto out;
  }
  /* Finite zeros in every lane the last block can read past the context.
     The scratch is fully written before it is read. */
  memset(ctx->kt, 0, (size_t)cache_floats * sizeof(float));
  memset(ctx->v, 0, (size_t)cache_floats * sizeof(float));
  /* exp16 at every fp16 d in [-17.5, 0], once per cache (18.5 k scalar
     exp_ps, a few ms; the double helpers are __hexagon_* imports). */
  attn_m1_det_exp_table(ctx->exp_tab);
out:
  if (err) {
    *err = rc;
  }
  return ctx;
}

void hvx_attn_m1_free(hvx_attn_m1_ctx *ctx) {
  if (!ctx) {
    return;
  }
  free(ctx->exp_tab);
  free(ctx->scratch);
  free(ctx->v);
  free(ctx->kt);
  free(ctx->kv_len);
  free(ctx);
}

/** @brief Kt of one (layer, kv head): [head_dim][max_seq]. */
static inline float *kt_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                             uint32_t h) {
  return ctx->kt +
         ((size_t)layer * ctx->n_kv + h) * ctx->head_dim * ctx->max_seq;
}

/** @brief V of one (layer, kv head): [max_seq][head_dim]. */
static inline float *v_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                            uint32_t h) {
  return ctx->v +
         ((size_t)layer * ctx->n_kv + h) * ctx->max_seq * ctx->head_dim;
}

/** @brief Spec steps 0-1 for one kv head: the rows rounded to fp16, the
 *         transposed scatter into Kt (scalar) and the row into V (V's row
 *         is vector-aligned; k and v are the caller's, so HVX_UVector). A
 *         seed row that is already fp16 (the CPU's cache) is unchanged. */
static inline void append_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                               uint32_t h, uint32_t pos, const float *k,
                               const float *v) {
  float *kt = kt_head(ctx, layer, h);
  for (uint32_t d = 0; d < HD; ++d) {
    kt[(size_t)d * ctx->max_seq + pos] = attn_m1_det_rne16(k[d]);
  }
  HVX_Vector *vd = (HVX_Vector *)(v_head(ctx, layer, h) + (size_t)pos * HD);
  const HVX_UVector *vs = (const HVX_UVector *)v;
  for (uint32_t i = 0; i < HD_VEC; ++i) {
    vd[i] = hvx_rne16_sf(vs[i]);
  }
}

int hvx_attn_m1_kv_append(hvx_attn_m1_ctx *ctx, uint32_t layer,
                          uint32_t kv_from, uint32_t n_rows,
                          const float *k_rows, const float *v_rows) {
  if (!ctx) {
    return AEE_EBADSTATE;
  }
  if (layer >= ctx->n_layers || kv_from > ctx->max_seq ||
      n_rows > ctx->max_seq - kv_from || (n_rows && (!k_rows || !v_rows))) {
    return AEE_EINVALIDFORMAT;
  }
  if (kv_from > ctx->kv_len[layer]) {
    return AEE_EBADSTATE;
  }
  const size_t row = (size_t)ctx->n_kv * ctx->head_dim;
  for (uint32_t r = 0; r < n_rows; ++r) {
    for (uint32_t h = 0; h < ctx->n_kv; ++h) {
      append_head(ctx, layer, h, kv_from + r,
                  k_rows + r * row + (size_t)h * ctx->head_dim,
                  v_rows + r * row + (size_t)h * ctx->head_dim);
    }
  }
  ctx->kv_len[layer] = kv_from + n_rows;
  return AEE_SUCCESS;
}

/** @brief One unit's phase pcycles (#146): each unit writes only its own
 *         slot, the caller reduces them after the pool run, so no atomics. */
typedef struct {
  uint64_t start, end;
  uint32_t scores, softmax, pv;
  uint32_t lanes; /**< the pool's n, as the unit saw it */
} prof_slot;

/** @brief Lanes the phase words record. ponytail: a pool of more than 15
 *         workers would leave its lanes >= 16 out of the sums (the device
 *         has 6 HVX contexts, the host check 8); widen the array then. */
#define PROF_SLOTS 16u

typedef struct {
  const hvx_attn_m1_ctx *ctx;
  uint32_t layer;
  uint32_t L;
  HVX_Vector vscale;
  const float *q;
  float *out;
  float *stats;
  prof_slot *slots; /**< NULL unless the phase words were requested */
  uint32_t units;   /**< attn_units(ctx) */
} forward_job;

/** @brief Adds the pcycles since *t to *acc and moves *t to now. */
static inline void prof_mark(uint32_t *acc, uint64_t *t) {
  const uint64_t now = HAP_perf_get_pcycles();
  *acc += (uint32_t)(now - *t);
  *t = now;
}

/** @brief Every lane equal to the max of the input's lanes: rotate by
 *         half the remaining width and combine, five steps. Each step is
 *         periodic in the rotation, so the last one leaves every lane with
 *         the same bits. */
static inline HVX_Vector reduce_max_sf(HVX_Vector v) {
  for (uint32_t rot = VLEN / 2u; rot >= 4u; rot >>= 1) {
    v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vror_VR(v, (int)rot));
  }
  return v;
}

/**
 * @brief attn_m1_det_rne16 for the scalar sum, without its volatiles.
 *        The skel is built without -ffast-math, so the compiler neither
 *        reassociates (x + c) - c nor contracts (there is no multiply);
 *        the volatiles in the spec are for the host and ARM builds, and
 *        here they would put three store-load round trips on the one
 *        dependent chain of the softmax (#152). HvxAttnM1.* compares the
 *        result with the spec on silicon.
 */
static inline float rne16_scalar(float x) {
  uint32_t u, rb;
  memcpy(&u, &x, sizeof(u));
  uint32_t ef = u & 0x7F800000u;
  ef = ef < ATTN_M1_DET_EF_MIN16 ? ATTN_M1_DET_EF_MIN16 : ef;
  const uint32_t cb = ef + ATTN_M1_DET_RNE16_C;
  float c;
  memcpy(&c, &cb, sizeof(c));
  float r = (x + c) - c;
  memcpy(&rb, &r, sizeof(rb));
  rb |= u & 0x80000000u;
  memcpy(&r, &rb, sizeof(r));
  return r;
}

/**
 * @brief q heads g0 .. g0 + ng - 1 of kv head @a h; their probabilities go
 *        to the scratch rows of those q heads. @a ps (NULL: no timestamps)
 *        accumulates the phase pcycles.
 *
 * Always inlined so attn_unit can call it with a literal ng: then the g
 * loops unroll and the accumulators live in registers (plan 146 section
 * 3.2). Same operations, same order per lane, whichever call site.
 */
static inline __attribute__((always_inline)) void
attn_body(const forward_job *job, uint32_t h, uint32_t g0, prof_slot *ps,
          const uint32_t ng) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const uint32_t ms = ctx->max_seq, hq0 = h * ctx->gqa + g0;
  const uint32_t L = job->L, nblk = (L + LANES - 1u) / LANES;
  const uint32_t n_live = L - (nblk - 1u) * LANES; /* lanes of the last */
  /* The last block needs a mask only when it is partial; a full one skips
     the mux, so nothing rests on vsetq2's behaviour at 128 bytes. */
  const int partial = n_live < LANES;
  const HVX_VectorPred live = Q6_Q_vsetq2_R((int)(n_live * sizeof(float)));
  const float *kt = kt_head(ctx, job->layer, h);
  const float *vv = v_head(ctx, job->layer, h);
  float *e = ctx->scratch + (size_t)hq0 * ms;
  const HVX_Vector neg_max = Q6_V_vsplat_R((int32_t)NEG_FLT_MAX_BITS);
  HVX_Vector vmax[MAX_GQA];
  union {
    HVX_Vector v[MAX_GQA * HD_VEC];
    float f[MAX_GQA * HD];
  } q16;
  uint64_t t = ps ? HAP_perf_get_pcycles() : 0u;

  /* Step 0 for q: the unit's rows rounded to fp16 once. */
  const HVX_UVector *qs = (const HVX_UVector *)(job->q + (size_t)hq0 * HD);
  for (uint32_t i = 0; i < ng * HD_VEC; ++i) {
    q16.v[i] = hvx_rne16_sf(qs[i]);
  }

  /* Scores. Per block and q head, the CPU's eight lanes over d = 8 blk + l
     as eight vectors of positions; the vpaddq tree; 0 + t; * scale. */
  for (uint32_t g = 0; g < ng; ++g) {
    vmax[g] = neg_max;
  }
  for (uint32_t b = 0; b < nblk; ++b) {
    for (uint32_t g = 0; g < ng; ++g) {
      const float *qg = q16.f + (size_t)g * HD;
      HVX_Vector acc[ATTN_M1_DET_ACC];
      for (uint32_t l = 0; l < ATTN_M1_DET_ACC; ++l) {
        acc[l] = Q6_V_vzero();
      }
      for (uint32_t d = 0; d < HD; ++d) {
        const HVX_Vector kv =
          *(const HVX_Vector *)(kt + (size_t)d * ms + b * LANES);
        acc[d % ATTN_M1_DET_ACC] =
          hvx_fma16_sf(acc[d % ATTN_M1_DET_ACC], hvx_splat_sf(qg[d]), kv);
      }
      const HVX_Vector s03 = hvx_rne16_sf(
        Q6_Vsf_vadd_VsfVsf(hvx_rne16_sf(Q6_Vsf_vadd_VsfVsf(acc[0], acc[1])),
                           hvx_rne16_sf(Q6_Vsf_vadd_VsfVsf(acc[2], acc[3]))));
      const HVX_Vector s47 = hvx_rne16_sf(
        Q6_Vsf_vadd_VsfVsf(hvx_rne16_sf(Q6_Vsf_vadd_VsfVsf(acc[4], acc[5])),
                           hvx_rne16_sf(Q6_Vsf_vadd_VsfVsf(acc[6], acc[7]))));
      const HVX_Vector tt = Q6_Vsf_vadd_VsfVsf(
        Q6_V_vzero(), hvx_rne16_sf(Q6_Vsf_vadd_VsfVsf(s03, s47)));
      const HVX_Vector s = hvx_rne16_sf(Q6_Vsf_vmpy_VsfVsf(tt, job->vscale));
      *(HVX_Vector *)(e + (size_t)g * ms + b * LANES) = s;
      vmax[g] = Q6_Vsf_vmax_VsfVsf(vmax[g], (partial && b + 1u == nblk)
                                              ? Q6_V_vmux_QVV(live, s, neg_max)
                                              : s);
    }
  }

  if (ps) {
    prof_mark(&ps->scores, &t);
  }

  /* Softmax: m; the exp-table index of rne16(s - m) per lane; the table
     read and the sequential fp16 sum on the scalar core; the divide. */
  const HVX_Vector abs_mask = Q6_V_vsplat_R(0x7FFFFFFF);
  const HVX_Vector bias = Q6_V_vsplat_R((int)ATTN_M1_DET_EXP_BIAS);
  const HVX_Vector idx_max = Q6_V_vsplat_R((int)ATTN_M1_DET_EXP_LAST + 1);
  HVX_Vector m[MAX_GQA];
  for (uint32_t g = 0; g < ng; ++g) {
    m[g] = Q6_Vsf_vadd_VsfVsf(reduce_max_sf(vmax[g]), Q6_V_vzero());
    HVX_Vector *eg = (HVX_Vector *)(e + (size_t)g * ms);
    for (uint32_t b = 0; b < nblk; ++b) {
      const HVX_Vector d = hvx_rne16_sf(Q6_Vsf_vsub_VsfVsf(eg[b], m[g]));
      HVX_Vector ix =
        Q6_Vw_vsub_VwVw(Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(d, abs_mask), 13), bias);
      ix = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(ix, Q6_V_vzero()), idx_max);
      eg[b] = ix;
    }
  }
  /* The sums: one dependent scalar chain per q head, the unit's heads
     interleaved so the core overlaps them. */
  float l[MAX_GQA];
  for (uint32_t g = 0; g < ng; ++g) {
    l[g] = 0.0f;
  }
  for (uint32_t p = 0; p < L; ++p) {
    for (uint32_t g = 0; g < ng; ++g) {
      float *ep = e + (size_t)g * ms + p;
      uint32_t ix;
      memcpy(&ix, ep, sizeof(ix));
      const float ev = ctx->exp_tab[ix];
      *ep = ev;
      l[g] = rne16_scalar(l[g] + ev);
    }
  }
  for (uint32_t g = 0; g < ng; ++g) {
    HVX_Vector *eg = (HVX_Vector *)(e + (size_t)g * ms);
    const HVX_Vector lv = hvx_splat_sf(l[g]);
    const HVX_Vector r = hvx_recip_det_sf(lv);
    for (uint32_t b = 0; b < nblk; ++b) {
      HVX_Vector ev = eg[b];
      if (partial && b + 1u == nblk) {
        ev = Q6_V_vmux_QVV(live, ev, Q6_V_vzero());
      }
      eg[b] = hvx_div16_sf(ev, lv, r);
    }
    if (job->stats) {
      const uint32_t hq = hq0 + g;
      const int32_t mb = Q6_R_vextract_VR(m[g], 0);
      memcpy(job->stats + 2u * hq, &mb, sizeof(float));
      memcpy(job->stats + 2u * hq + 1u, &l[g], sizeof(float));
    }
  }

  if (ps) {
    prof_mark(&ps->softmax, &t);
  }

  /* PV: o[g][vec] = fma16(o[g][vec], e[g][p], V[p][vec]), p ascending. */
  HVX_Vector o[MAX_GQA][HD_VEC];
  for (uint32_t g = 0; g < ng; ++g) {
    for (uint32_t i = 0; i < HD_VEC; ++i) {
      o[g][i] = Q6_V_vzero();
    }
  }
  for (uint32_t p = 0; p < L; ++p) {
    /* The V row once per p, not once per g. */
    const HVX_Vector *vp = (const HVX_Vector *)(vv + (size_t)p * HD);
    HVX_Vector vr[HD_VEC];
    for (uint32_t i = 0; i < HD_VEC; ++i) {
      vr[i] = vp[i];
    }
    for (uint32_t g = 0; g < ng; ++g) {
      const HVX_Vector ep = hvx_splat_sf(e[(size_t)g * ms + p]);
      for (uint32_t i = 0; i < HD_VEC; ++i) {
        o[g][i] = hvx_fma16_sf(o[g][i], ep, vr[i]);
      }
    }
  }
  for (uint32_t g = 0; g < ng; ++g) {
    HVX_UVector *vo = (HVX_UVector *)(job->out + ((size_t)hq0 + g) * HD);
    for (uint32_t i = 0; i < HD_VEC; ++i) {
      vo[i] = o[g][i];
    }
  }
  if (ps) {
    prof_mark(&ps->pv, &t);
  }
}

/** @brief Units per call: one per (kv head, q-head pair) when gqa is even,
 *         else one per kv head. */
static inline uint32_t attn_units(const hvx_attn_m1_ctx *ctx) {
  return ctx->gqa % 2u == 0u ? ctx->n_kv * (ctx->gqa / 2u) : ctx->n_kv;
}

/** @brief Unit @a u: q heads 2j, 2j + 1 of kv head u / (gqa/2) (j = u %
 *         (gqa/2)); an odd gqa takes a whole kv head on the runtime-ng
 *         copy. The two units of one kv head are adjacent indices, so they
 *         run at the same time and share its Kt / V slab in L2. */
static void attn_unit(const forward_job *job, uint32_t u, prof_slot *ps) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  if (ctx->gqa % 2u != 0u) {
    attn_body(job, u, 0u, ps, ctx->gqa);
    return;
  }
  const uint32_t pairs = ctx->gqa / 2u;
  attn_body(job, u / pairs, 2u * (u % pairs), ps, 2u);
}

/** @brief Pool lane i of n: units i, i + n, ... */
static void forward_unit(uint32_t n, uint32_t i, void *arg) {
  const forward_job *job = (const forward_job *)arg;
  prof_slot *ps = (job->slots && i < PROF_SLOTS) ? &job->slots[i] : NULL;
  if (ps) {
    ps->lanes = n;
    ps->start = HAP_perf_get_pcycles();
  }
  for (uint32_t u = i; u < job->units; u += n) {
    attn_unit(job, u, ps);
  }
  if (ps) {
    ps->end = HAP_perf_get_pcycles();
  }
}

int hvx_attn_m1_forward(hvx_attn_m1_ctx *ctx, uint32_t layer, uint32_t pos,
                        float scale, const float *q, const float *k,
                        const float *v, float *out, float *stats) {
  return hvx_attn_m1_forward_prof(ctx, layer, pos, scale, q, k, v, out, stats,
                                  NULL);
}

int hvx_attn_m1_forward_prof(hvx_attn_m1_ctx *ctx, uint32_t layer, uint32_t pos,
                             float scale, const float *q, const float *k,
                             const float *v, float *out, float *stats,
                             uint32_t *prof) {
  const uint64_t qt0 = prof ? HAP_perf_get_qtimer_count() : 0u;
  if (!ctx) {
    return AEE_EBADSTATE;
  }
  if (layer >= ctx->n_layers || pos >= ctx->max_seq || !q || !k || !v || !out) {
    return AEE_EINVALIDFORMAT;
  }
  if (pos > ctx->kv_len[layer]) {
    return AEE_EBADSTATE;
  }
  prof_slot slots[PROF_SLOTS];
  uint64_t t0 = prof ? HAP_perf_get_pcycles() : 0u;
  for (uint32_t h = 0; h < ctx->n_kv; ++h) {
    append_head(ctx, layer, h, pos, k + (size_t)h * HD, v + (size_t)h * HD);
  }
  ctx->kv_len[layer] = pos + 1u;

  forward_job job;
  job.ctx = ctx;
  job.layer = layer;
  job.L = pos + 1u;
  job.vscale = hvx_splat_sf(scale);
  job.q = q;
  job.out = out;
  job.stats = stats;
  job.slots = prof ? slots : NULL;
  job.units = attn_units(ctx);
  if (prof) {
    memset(slots, 0, sizeof(slots));
    memset(prof, 0, ATTN_M1_PROF_WORDS * sizeof(uint32_t));
    prof_mark(&prof[ATTN_M1_PROF_APPEND], &t0);
  }
  hvx_worker_pool_run(ctx->pool, forward_unit, &job, job.units);
  if (prof) {
    const uint64_t pool_t0 = t0;
    prof_mark(&prof[ATTN_M1_PROF_POOL], &t0);
    const uint32_t lanes = slots[0].lanes;
    prof[ATTN_M1_PROF_LANES] = lanes;
    for (uint32_t i = 0; i < lanes && i < PROF_SLOTS; ++i) {
      const uint32_t busy = (uint32_t)(slots[i].end - slots[i].start);
      const uint32_t skew = (uint32_t)(slots[i].start - pool_t0);
      prof[ATTN_M1_PROF_SCORES] += slots[i].scores;
      prof[ATTN_M1_PROF_SOFTMAX] += slots[i].softmax;
      prof[ATTN_M1_PROF_PV] += slots[i].pv;
      if (busy > prof[ATTN_M1_PROF_BUSY_MAX]) {
        prof[ATTN_M1_PROF_BUSY_MAX] = busy;
      }
      if (skew > prof[ATTN_M1_PROF_START_MAX]) {
        prof[ATTN_M1_PROF_START_MAX] = skew;
      }
    }
    prof[ATTN_M1_PROF_CALL_QT] = (uint32_t)(HAP_perf_get_qtimer_count() - qt0);
  }
  return AEE_SUCCESS;
}
