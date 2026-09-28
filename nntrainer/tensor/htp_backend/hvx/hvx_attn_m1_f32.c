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
 * Per kv head, three passes over the cache in the spec's order:
 *   scores  one 32-position block at a time: for d ascending, acc[g] =
 *           acc[g] + splat(q[g][d]) * Kt[d][block] for the gqa q heads g
 *           that share this kv head, then s = acc * scale (stored to the
 *           lane's scratch; the block's running vmax takes the live lanes)
 *   softmax m = rotate-tree max; e = exp_det(s - m), masked lanes -> 0,
 *           summed lane-wise over blocks, then the rotate tree; r =
 *           recip_det(l). Both trees leave every lane equal (the rotation
 *           by half the remaining width makes each step periodic), so the
 *           reduced vectors are used whole -- no scalar FPU anywhere.
 *   PV      for p ascending, o[g][vec] = o[g][vec] + splat(e[g][p]) *
 *           V[p][vec]; out = o * r
 * Rule 24: plain Vsf, no qf32, no flush-to-zero. The masked lanes of the
 * last block are filled with -FLT_MAX for the max rather than -inf, so no
 * lane ever holds an infinity (the inf encodings are the open case of
 * rule 24 and the domain keeps every live score far from FLT_MAX).
 *
 * ponytail: the q splats are re-formed per block (4 scalar loads and
 * splats per d); a per-head splat table in scratch would trade 32 KiB of
 * writes for them, worth it only if #85's per-op pcycles say the score
 * loop is not DDR-bound (plan 81 section 0 expects it to be).
 */

#include "hvx_attn_m1_f32.h"

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_swiglu_det.h"

/** @brief f32 lanes per HVX vector: one position block. */
#define LANES 32u
/** @brief Vector length in bytes, the cache alignment. */
#define VLEN 128u
/** @brief The mask fill for the max: the most negative finite f32. */
#define NEG_FLT_MAX_BITS 0xFF7FFFFFu
/** @brief Upper bound on q heads per kv head the register file holds:
 *         gqa score accumulators plus gqa * head_dim/32 PV accumulators
 *         stay under the 32 vector registers at 4 x 2. */
#define MAX_GQA 8u
/** @brief Upper bound on head_dim / 32 for the PV accumulator array. */
#define MAX_HD_VEC 4u

hvx_attn_m1_ctx *hvx_attn_m1_create(uint32_t n_layers, uint32_t n_kv,
                                    uint32_t gqa, uint32_t head_dim,
                                    uint32_t max_seq, hvx_worker_pool *pool,
                                    int *err) {
  int rc = AEE_SUCCESS;
  hvx_attn_m1_ctx *ctx = NULL;
  if (n_layers == 0u || n_kv == 0u || gqa == 0u || gqa > MAX_GQA ||
      head_dim == 0u || head_dim % LANES != 0u ||
      head_dim / LANES > MAX_HD_VEC || max_seq == 0u || max_seq % LANES != 0u) {
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
  if (!ctx->kv_len || !ctx->kt || !ctx->v || !ctx->scratch) {
    rc = AEE_ENOMEMORY;
    hvx_attn_m1_free(ctx);
    ctx = NULL;
    goto out;
  }
  /* Finite zeros in every lane the last block can read past the context.
     The scratch is fully written before it is read. */
  memset(ctx->kt, 0, (size_t)cache_floats * sizeof(float));
  memset(ctx->v, 0, (size_t)cache_floats * sizeof(float));
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

/** @brief Spec step 1 for one kv head: the transposed scatter into Kt and
 *         the row copy into V (V's row is vector-aligned; k and v are the
 *         caller's, so scalar and HVX_UVector). */
static inline void append_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                               uint32_t h, uint32_t pos, const float *k,
                               const float *v) {
  float *kt = kt_head(ctx, layer, h);
  for (uint32_t d = 0; d < ctx->head_dim; ++d) {
    kt[(size_t)d * ctx->max_seq + pos] = k[d];
  }
  HVX_Vector *vd =
    (HVX_Vector *)(v_head(ctx, layer, h) + (size_t)pos * ctx->head_dim);
  const HVX_UVector *vs = (const HVX_UVector *)v;
  for (uint32_t i = 0; i < ctx->head_dim / LANES; ++i) {
    vd[i] = vs[i];
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

typedef struct {
  const hvx_attn_m1_ctx *ctx;
  uint32_t layer;
  uint32_t L;
  HVX_Vector vscale;
  const float *q;
  float *out;
  float *stats;
} forward_job;

/** @brief Every lane equal to the max / sum of the input's lanes: rotate
 *         by half the remaining width and combine, five steps. Each step
 *         is periodic in the rotation, so the last one leaves every lane
 *         with the same bits (attn_m1_det.h step 5's tree, lane 0). */
static inline HVX_Vector reduce_max_sf(HVX_Vector v) {
  for (uint32_t rot = VLEN / 2u; rot >= 4u; rot >>= 1) {
    v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vror_VR(v, (int)rot));
  }
  return v;
}
static inline HVX_Vector reduce_sum_sf(HVX_Vector v) {
  for (uint32_t rot = VLEN / 2u; rot >= 4u; rot >>= 1) {
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, (int)rot));
  }
  return v;
}

/** @brief One kv head, written by the unit that owns scratch lane @a lane. */
static void attn_head(const forward_job *job, uint32_t h, uint32_t lane) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const uint32_t gqa = ctx->gqa, hd = ctx->head_dim, ms = ctx->max_seq;
  const uint32_t L = job->L, nblk = (L + LANES - 1u) / LANES;
  const uint32_t n_live = L - (nblk - 1u) * LANES; /* lanes of the last */
  /* The last block needs a mask only when it is partial; a full one skips
     the mux, so nothing rests on vsetq2's behaviour at 128 bytes. */
  const int partial = n_live < LANES;
  const HVX_VectorPred live = Q6_Q_vsetq2_R((int)(n_live * sizeof(float)));
  const float *kt = kt_head(ctx, job->layer, h);
  const float *vv = v_head(ctx, job->layer, h);
  const float *q = job->q + (size_t)h * gqa * hd;
  float *e = ctx->scratch + (size_t)lane * gqa * ms;
  const HVX_Vector neg_max = Q6_V_vsplat_R((int32_t)NEG_FLT_MAX_BITS);
  HVX_Vector vmax[MAX_GQA], acc[MAX_GQA];

  /* Scores: s[g][p] = (sum_d q[g][d] * Kt[d][p], d ascending) * scale. */
  for (uint32_t g = 0; g < gqa; ++g) {
    vmax[g] = neg_max;
  }
  for (uint32_t b = 0; b < nblk; ++b) {
    for (uint32_t g = 0; g < gqa; ++g) {
      acc[g] = Q6_V_vzero();
    }
    for (uint32_t d = 0; d < hd; ++d) {
      const HVX_Vector kv =
        *(const HVX_Vector *)(kt + (size_t)d * ms + b * LANES);
      for (uint32_t g = 0; g < gqa; ++g) {
        acc[g] = Q6_Vsf_vadd_VsfVsf(
          acc[g], Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(q[(size_t)g * hd + d]), kv));
      }
    }
    for (uint32_t g = 0; g < gqa; ++g) {
      const HVX_Vector s = Q6_Vsf_vmpy_VsfVsf(acc[g], job->vscale);
      *(HVX_Vector *)(e + (size_t)g * ms + b * LANES) = s;
      vmax[g] = Q6_Vsf_vmax_VsfVsf(vmax[g], (partial && b + 1u == nblk)
                                              ? Q6_V_vmux_QVV(live, s, neg_max)
                                              : s);
    }
  }

  /* Softmax: m, e = exp_det(s - m) with the dead lanes zeroed, l, r. */
  HVX_Vector r[MAX_GQA];
  for (uint32_t g = 0; g < gqa; ++g) {
    const HVX_Vector m = reduce_max_sf(vmax[g]);
    HVX_Vector *eg = (HVX_Vector *)(e + (size_t)g * ms);
    HVX_Vector sum = Q6_V_vzero();
    for (uint32_t b = 0; b < nblk; ++b) {
      HVX_Vector ev = hvx_exp_det_sf(Q6_Vsf_vsub_VsfVsf(eg[b], m));
      if (partial && b + 1u == nblk) {
        ev = Q6_V_vmux_QVV(live, ev, Q6_V_vzero());
      }
      eg[b] = ev;
      sum = Q6_Vsf_vadd_VsfVsf(sum, ev);
    }
    const HVX_Vector l = reduce_sum_sf(sum);
    r[g] = hvx_recip_det_sf(l);
    if (job->stats) {
      const uint32_t hq = h * gqa + g;
      const int32_t mb = Q6_R_vextract_VR(m, 0), lb = Q6_R_vextract_VR(l, 0);
      memcpy(job->stats + 2u * hq, &mb, sizeof(float));
      memcpy(job->stats + 2u * hq + 1u, &lb, sizeof(float));
    }
  }

  /* PV: o[g][vec] = sum_p e[g][p] * V[p][vec], p ascending; out = o * r. */
  const uint32_t nvec = hd / LANES;
  HVX_Vector o[MAX_GQA][MAX_HD_VEC];
  for (uint32_t g = 0; g < gqa; ++g) {
    for (uint32_t i = 0; i < nvec; ++i) {
      o[g][i] = Q6_V_vzero();
    }
  }
  for (uint32_t p = 0; p < L; ++p) {
    const HVX_Vector *vp = (const HVX_Vector *)(vv + (size_t)p * hd);
    for (uint32_t g = 0; g < gqa; ++g) {
      const HVX_Vector ep = hvx_splat_sf(e[(size_t)g * ms + p]);
      for (uint32_t i = 0; i < nvec; ++i) {
        o[g][i] = Q6_Vsf_vadd_VsfVsf(o[g][i], Q6_Vsf_vmpy_VsfVsf(ep, vp[i]));
      }
    }
  }
  for (uint32_t g = 0; g < gqa; ++g) {
    HVX_UVector *vo = (HVX_UVector *)(job->out + ((size_t)h * gqa + g) * hd);
    for (uint32_t i = 0; i < nvec; ++i) {
      vo[i] = Q6_Vsf_vmpy_VsfVsf(o[g][i], r[g]);
    }
  }
}

/** @brief Pool unit i of n: kv heads i, i + n, ..., scratch lane i. */
static void forward_unit(uint32_t n, uint32_t i, void *arg) {
  const forward_job *job = (const forward_job *)arg;
  for (uint32_t h = i; h < job->ctx->n_kv; h += n) {
    attn_head(job, h, i);
  }
}

int hvx_attn_m1_forward(hvx_attn_m1_ctx *ctx, uint32_t layer, uint32_t pos,
                        float scale, const float *q, const float *k,
                        const float *v, float *out, float *stats) {
  if (!ctx) {
    return AEE_EBADSTATE;
  }
  if (layer >= ctx->n_layers || pos >= ctx->max_seq || !q || !k || !v || !out) {
    return AEE_EINVALIDFORMAT;
  }
  if (pos > ctx->kv_len[layer]) {
    return AEE_EBADSTATE;
  }
  for (uint32_t h = 0; h < ctx->n_kv; ++h) {
    append_head(ctx, layer, h, pos, k + (size_t)h * ctx->head_dim,
                v + (size_t)h * ctx->head_dim);
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
  hvx_worker_pool_run(ctx->pool, forward_unit, &job, ctx->n_kv);
  return AEE_SUCCESS;
}
