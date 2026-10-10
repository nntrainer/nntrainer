// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_attn_m1_f32.c
 * @date   27 Sep 2026
 * @brief  Decode attention at m=1 on HVX over the session's fp16 KV cache,
 *         in fp16 lanes, bit-identical to attn_m1_det.h
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plan 170 section 3. Every fp16 step of the spec is one hf op or one
 * hvx_attn_m1_hf.h primitive (its fused FMA was bit-exact on silicon
 * against the spec on real, adversarial and zero / sign triples: S1, G1).
 * One call rounds k / v / q as vectors (the v row into V, the k row kept
 * for P1, q as one zipped row per head), then runs three pool runs with
 * two short serial steps between them (the lane plan from S1's cost sweep:
 * the one-head score shape and the four-head PV shape scale to 6 lanes,
 * the two-head score shape does not):
 *
 *   P1 scores   unit = (kv head, 64-position Kt tile): the last tile first
 *               takes the new k column (spec step 1); for each q head of
 *               the kv head, eight accumulators over d = 8 blk + l (the
 *               CPU's float16x8_t lanes, positions in the vector lanes;
 *               q[d] a vlut16 splat of the zipped row, round 3), the
 *               vpaddq tree, 0 + t, * scale -> S[hq][tile]. The tile
 *               (8 KiB) is read from DDR once, behind an l2fetch the lane
 *               issued one unit earlier, and from L2 for the other q heads.
 *   caller      m per q head: hf max over its tiles (masked past L), a
 *               rotate-tree to one lane, + 0.
 *   P2 exp      same units: e = exp16(hf(s - m)) (masked lanes 0) back to
 *               S, and into ET[p][hq], one 128-byte row per position with
 *               the q heads in lanes 0 .. n_q - 1, by a 4-head transpose
 *               and masked vector stores. Since
 *               round 3 exp16 is the spec's checked table (37 KiB in the
 *               ctx) at the spec's index, three integer ops per vector,
 *               gathered by scalar loads for the unit's heads at once.
 *   caller      l: the CPU's sequential fp16 sum, all q heads at once: one
 *               hf add per position on the ET rows (lanes past n_q are 0).
 *   P3 PV       lane i of n takes a range of q-head chains (whole pairs
 *               for an even gqa) and walks it by kv head in groups of 4 / 2
 *               / 1: divide their rows by l (hvx_hf_div16), then o[g] =
 *               fma(o[g], splat(p[g][p]), V[p]) for p ascending, the chains
 *               of a group sharing each V row load, the V rows l2fetched
 *               two PV_BLOCKs ahead; out = o widened to f32 as vectors.
 *
 * [plan 201 S4] Gemma 4: a head_dim of 256 or 512 is head_dim / 64 chunks
 * of the same vectors (P1 runs its chains on through the chunks' zipped q
 * rows, P3 one walk per PV_CG(ng) chunks, #261), and a window starts every
 * run at the tile of the first visible position, its lanes before that masked
 * like the lanes past L (max, exp), the sum and PV chains from that position.
 *
 * The phase words keep their meaning (attn_m1_det.h): APPEND = the
 * caller's rounding and stores, SCORES = P1 (with the k column merge),
 * SOFTMAX = P2 + the two caller steps + the divides, PV = P3's chains,
 * summed over lanes; POOL spans P1 .. P3; BUSY_MAX is the busiest lane over
 * the three runs; LANES and START_MAX are P1's. Round 3 splits SOFTMAX:
 * EXP and ET bracket P2's two halves per unit, MAX and SUM the caller's
 * steps, DIV the divides (every bracket behind if (ps) / if (prof), so a
 * NULL prof takes no timestamp).
 */

#include "hvx_attn_m1_f32.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "attn_m1_det.h"
#include "hvx_attn_m1_hf.h"
#include "hvx_convert.h"
#include "hvx_swiglu_det.h"

/** @brief Positions per Kt tile = hf lanes per vector. */
#define TILE 64u
/** @brief Vector length in bytes, the alignment of every buffer. */
#define VLEN 128u
/** @brief hf lanes per vector: one 64-wide chunk of a head row. head_dim
 *         is a multiple of it (#152: 64; plan 201 S4: up to 512, Gemma 4's
 *         256 and 512), head_dim / CH chunks a row. */
#define CH 64u
/** @brief Upper bound on q heads per kv head. */
#define MAX_GQA 8u
/** @brief q heads per call: the lanes of the sum vector. */
#define MAX_NQ 64u
/** @brief q heads per PV unit, the chains that share a V row load. */
#define PV_GROUP 4u
/** @brief PV chains (head x 64-wide chunk) one V walk keeps in registers. */
#define PV_ACC 8u
/** @brief 64-wide chunks per V walk for ng heads. */
#define PV_CG(ng) (PV_ACC / (ng))
/** @brief fp16 bits of -65504, the mask fill for the max. */
#define HF_NEG_MAX 0xFBFF

/* exp16 by the checked table (plan 170 round 3 section 3.1), the same bits
   as hvx_hf_exp16. S4 (cold pos 1023, 6 lanes, lane-summed): the table's
   scalar gather read 387-389 k pcycles where hvx_hf_exp16 read 272 k (the
   v79 ISS: 124 k and 271 k -- it prices the compute, not six threads'
   gathers). The G6 gates were read with the table on, so it stays the
   default; 0 compiles hvx_hf_exp16 back, round 4's first cell, before
   vgather from VTCM (plan section 3.5). */
#ifndef ATTN_M1_EXP_TAB
#define ATTN_M1_EXP_TAB 1 /**< P2: exp16 by the checked table */
#endif

/** @brief Tiles of a max_seq. */
static inline uint32_t tiles_of(uint32_t max_seq) {
  return (max_seq + TILE - 1u) / TILE;
}

/** @brief fp16 bits (finite) to f32, exact. */
static inline float hf_float(uint16_t h) {
  const uint32_t e = (h >> 10) & 31u, m = h & 1023u;
  const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  if (e == 0u) {
    const float f = (float)m * (1.0f / 16777216.0f);
    return attn_m1_det_float(attn_m1_det_bits(f) | s);
  }
  return attn_m1_det_float(s | ((e + 112u) << 23) | (m << 13));
}

hvx_attn_m1_ctx *hvx_attn_m1_create(uint32_t n_layers, uint32_t n_kv,
                                    uint32_t gqa, uint32_t head_dim,
                                    uint32_t max_seq, hvx_worker_pool *pool,
                                    int *err) {
  int rc = AEE_SUCCESS;
  hvx_attn_m1_ctx *ctx = NULL;
  const uint32_t q8 = (max_seq & HVX_ATTN_M1_Q8) != 0u;     /* [#282 D] */
  const uint32_t lazy = (max_seq & HVX_ATTN_M1_LAZY) != 0u; /* [#289] */
  max_seq &= ~(HVX_ATTN_M1_Q8 | HVX_ATTN_M1_LAZY);
  if (n_layers == 0u || n_kv == 0u || gqa == 0u || gqa > MAX_GQA ||
      n_kv * gqa > MAX_NQ || head_dim == 0u || head_dim % CH != 0u ||
      head_dim > ATTN_M1_DET_MAX_HD || max_seq == 0u || max_seq % 32u != 0u) {
    rc = AEE_EINVALIDFORMAT;
    goto out;
  }
  /* size_t is 32 bits on the DSP: bound the cache before multiplying. */
  const uint64_t seq = (uint64_t)tiles_of(max_seq) * TILE;
  const uint64_t halves = (uint64_t)n_layers * n_kv * head_dim * seq;
  if (halves * sizeof(uint16_t) >= (1ull << 31)) {
    rc = AEE_EINVALIDFORMAT;
    goto out;
  }
  ctx = (hvx_attn_m1_ctx *)calloc(1, sizeof(*ctx));
  if (!ctx) {
    rc = AEE_ENOMEMORY;
    goto out;
  }
  const uint32_t n_q = n_kv * gqa;
  ctx->n_layers = n_layers;
  ctx->n_kv = n_kv;
  ctx->gqa = gqa;
  ctx->head_dim = head_dim;
  ctx->max_seq = max_seq;
  ctx->seq = (uint32_t)seq;
  ctx->cache_halves = (size_t)halves;
  ctx->pool = pool;
  ctx->kv_len = (uint32_t *)calloc(n_layers, sizeof(uint32_t));
  const size_t per_layer =
    (size_t)(halves / n_layers) * (q8 ? 1u : sizeof(uint16_t));
  ctx->q8 = q8;
  if (q8) {
    ctx->sk =
      (float *)calloc((size_t)n_layers * n_kv * head_dim, sizeof(float));
    ctx->sv =
      (float *)calloc((size_t)n_layers * n_kv * head_dim, sizeof(float));
    ctx->sv_inv =
      (float *)calloc((size_t)n_layers * n_kv * head_dim, sizeof(float));
    ctx->sk_inv =
      (float *)calloc((size_t)n_layers * n_kv * head_dim, sizeof(float));
    ctx->scaled = (uint8_t *)calloc(n_layers, 1);
    if (!ctx->sk || !ctx->sv || !ctx->sv_inv || !ctx->sk_inv || !ctx->scaled) {
      rc = AEE_ENOMEMORY;
    }
  }
  ctx->kt = (uint16_t **)calloc(n_layers, sizeof(uint16_t *));
  ctx->v = (uint16_t **)calloc(n_layers, sizeof(uint16_t *));
  ctx->layer_bytes = per_layer;
  for (uint32_t l = 0;
       !lazy && rc == AEE_SUCCESS && ctx->kt && ctx->v && l < n_layers; ++l) {
    ctx->kt[l] = (uint16_t *)memalign(VLEN, per_layer);
    ctx->v[l] = (uint16_t *)memalign(VLEN, per_layer);
    if (!ctx->kt[l] || !ctx->v[l]) {
      rc = AEE_ENOMEMORY;
      break;
    }
    /* Finite zeros wherever a tile is read past the context */
    memset(ctx->kt[l], 0, per_layer);
    memset(ctx->v[l], 0, per_layer);
  }
  ctx->s = (uint16_t *)memalign(VLEN, (size_t)n_q * seq * sizeof(uint16_t));
  ctx->et = (uint16_t *)memalign(VLEN, (size_t)seq * VLEN);
  ctx->qs = (uint16_t *)memalign(VLEN, (size_t)n_q * (head_dim / CH) * VLEN);
  ctx->kr =
    (uint16_t *)memalign(VLEN, (size_t)n_kv * head_dim * sizeof(uint16_t));
  ctx->exp_tab = (uint16_t *)malloc(ATTN_M1_DET_EXP_N * sizeof(uint16_t));
  float *tmp = (float *)malloc(ATTN_M1_DET_EXP_N * sizeof(float));
  if (rc != AEE_SUCCESS || !ctx->kv_len || !ctx->kt || !ctx->v || !ctx->s ||
      !ctx->et || !ctx->qs || !ctx->kr || !ctx->exp_tab || !tmp) {
    free(tmp);
    rc = AEE_ENOMEMORY;
    hvx_attn_m1_free(ctx);
    ctx = NULL;
    goto out;
  }
  hvx_hf_exp16_fill(ctx->exp_tab, tmp);
  free(tmp);
  /* The lanes of ET past n_q stay 0 for good (the cache was zeroed above). */
  memset(ctx->et, 0, (size_t)seq * VLEN);
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
  free(ctx->kr);
  free(ctx->qs);
  free(ctx->et);
  free(ctx->s);
  for (uint32_t l = 0; l < ctx->n_layers; ++l) {
    if (ctx->kt) {
      free(ctx->kt[l]);
    }
    if (ctx->v) {
      free(ctx->v[l]);
    }
  }
  free(ctx->v);
  free(ctx->kt);
  free(ctx->sk);
  free(ctx->sv);
  free(ctx->sv_inv);
  free(ctx->sk_inv);
  free(ctx->scaled);
  free(ctx->kv_len);
  free(ctx);
}

/** @brief [#289] A lazy layer's kt / v, allocated and zeroed on first use;
 *  a no-op for an allocated one. */
static int layer_alloc(hvx_attn_m1_ctx *ctx, uint32_t l) {
  if (ctx->kt[l] && ctx->v[l]) {
    return AEE_SUCCESS;
  }
  ctx->kt[l] = (uint16_t *)memalign(VLEN, ctx->layer_bytes);
  ctx->v[l] = (uint16_t *)memalign(VLEN, ctx->layer_bytes);
  if (!ctx->kt[l] || !ctx->v[l]) {
    free(ctx->kt[l]);
    free(ctx->v[l]);
    ctx->kt[l] = NULL;
    ctx->v[l] = NULL;
    return AEE_ENOMEMORY;
  }
  memset(ctx->kt[l], 0, ctx->layer_bytes);
  memset(ctx->v[l], 0, ctx->layer_bytes);
  return AEE_SUCCESS;
}

/** @brief Kt of one (layer, kv head): [tiles][head_dim][TILE] fp16. */
static inline uint16_t *kt_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                                uint32_t h) {
  return ctx->kt[layer] + (size_t)h * ctx->head_dim * ctx->seq;
}

/** @brief V of one (layer, kv head): [seq][head_dim] fp16. */
static inline uint16_t *v_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                               uint32_t h) {
  return ctx->v[layer] + (size_t)h * ctx->seq * ctx->head_dim;
}

/** @brief [#282 D] The int8 Kt of one (layer, kv head): [tiles][hd][64]. */
static inline int8_t *kt8_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                               uint32_t h) {
  return (int8_t *)ctx->kt[layer] + (size_t)h * ctx->head_dim * ctx->seq;
}

/** @brief [#282 D] The int8 V of one (layer, kv head): [seq][hd]. */
static inline int8_t *v8_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                              uint32_t h) {
  return (int8_t *)ctx->v[layer] + (size_t)h * ctx->seq * ctx->head_dim;
}

/** @brief [#282 D] x * inv rounded half away from zero, clamped to
 *         [-127, 127] (no libm call: the seed quantizes a prompt's rows of
 *         every layer at the first token, and rintf was ~10 s of it). */
static inline int8_t q8_of(float x, float inv) {
  float r = x * inv;
  r = r > 127.0f ? 127.0f : (r < -127.0f ? -127.0f : r);
  return (int8_t)(int32_t)(r + (r >= 0.0f ? 0.5f : -0.5f));
}

/** [#289] A seed over the pool: the amax by 32-dim slices of the row, the
 *  rows by (kv head, 64-row chunk). Each lane writes only its own dims or
 *  positions, with the per-element code of the one-thread loops. */
typedef struct {
  hvx_attn_m1_ctx *ctx;
  uint32_t layer, kv_from, n_rows, units;
  const float *k_rows, *v_rows;
} append_job;

#define APPEND_ROWS 64u

static void run_amax(uint32_t n, uint32_t i, void *arg) {
  const append_job *job = (const append_job *)arg;
  const size_t row = (size_t)job->ctx->n_kv * job->ctx->head_dim;
  float *sk = job->ctx->sk + (size_t)job->layer * row;
  float *sv = job->ctx->sv + (size_t)job->layer * row;
  for (uint32_t u = i; u < job->units; u += n) {
    const size_t lo = (size_t)u * 32u, hi = lo + 32u < row ? lo + 32u : row;
    for (uint32_t r = 0; r < job->n_rows; ++r) {
      const float *kr = job->k_rows + (size_t)r * row;
      const float *vr = job->v_rows + (size_t)r * row;
      for (size_t x = lo; x < hi; ++x) {
        const float kk = fabsf(kr[x]), vv = fabsf(vr[x]);
        sk[x] = kk > sk[x] ? kk : sk[x];
        sv[x] = vv > sv[x] ? vv : sv[x];
      }
    }
  }
}

/** @brief [#282 D] The layer's scales from @a n_rows rows [n][n_kv][hd]
 *         (amax x @a margin / 127; 1 for an all-zero head or dim). */
static void q8_set_scales(hvx_attn_m1_ctx *ctx, uint32_t layer, uint32_t n_rows,
                          const float *k_rows, const float *v_rows,
                          float margin) {
  const uint32_t hd = ctx->head_dim, nkv = ctx->n_kv;
  const size_t n = (size_t)nkv * hd;
  float *sk = ctx->sk + (size_t)layer * n;
  float *sv = ctx->sv + (size_t)layer * n;
  /* amax per (kv head, dim), rows outer so every read is contiguous */
  memset(sk, 0, n * sizeof(float));
  memset(sv, 0, n * sizeof(float));
  {
    append_job job = {ctx,    layer, 0u, n_rows, (uint32_t)((n + 31u) / 32u),
                      k_rows, v_rows};
    hvx_worker_pool_run(ctx->pool, run_amax, &job, job.units);
  }
  float *ski = ctx->sk_inv + (size_t)layer * n;
  float *svi = ctx->sv_inv + (size_t)layer * n;
  for (size_t i = 0; i < n; ++i) {
    sk[i] = sk[i] > 0.0f ? sk[i] * margin / 127.0f : 1.0f;
    sv[i] = sv[i] > 0.0f ? sv[i] * margin / 127.0f : 1.0f;
    ski[i] = 1.0f / sk[i];
    svi[i] = 1.0f / sv[i];
  }
  ctx->scaled[layer] = 1u;
}

/** @brief [#282 D] One position of one (layer, kv head) into the int8
 *         cache: k into its tile's column, v as the position's row. */
static inline void append_head_q8(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                                  uint32_t h, uint32_t pos, const float *k,
                                  const float *v) {
  const uint32_t hd = ctx->head_dim;
  int8_t *kt =
    kt8_head(ctx, layer, h) + (size_t)(pos / TILE) * hd * TILE + pos % TILE;
  int8_t *vr = v8_head(ctx, layer, h) + (size_t)pos * hd;
  const float *ski = ctx->sk_inv + ((size_t)layer * ctx->n_kv + h) * hd;
  const float *svi = ctx->sv_inv + ((size_t)layer * ctx->n_kv + h) * hd;
  for (uint32_t d = 0; d < hd; ++d) {
    kt[(size_t)d * TILE] = q8_of(k[d], ski[d]);
    vr[d] = q8_of(v[d], svi[d]);
  }
}

/** @brief Spec steps 0-1 for one kv head of the prefill seed: k and v
 *         rounded to fp16 (a seed row that is already fp16 is unchanged), k
 *         into its tile's column, v as the position's row. forward's own
 *         append is vector (hvx_attn_m1_forward_prof, p1_unit's merge). */
static inline void append_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                               uint32_t h, uint32_t pos, const float *k,
                               const float *v) {
  const uint32_t hd = ctx->head_dim;
  uint16_t *kt =
    kt_head(ctx, layer, h) + (size_t)(pos / TILE) * hd * TILE + pos % TILE;
  uint16_t *vr = v_head(ctx, layer, h) + (size_t)pos * hd;
  for (uint32_t d = 0; d < hd; ++d) {
    kt[(size_t)d * TILE] = hvx_hf_bits_rne(k[d]);
    vr[d] = hvx_hf_bits_rne(v[d]);
  }
}

static void run_append(uint32_t n, uint32_t i, void *arg) {
  const append_job *job = (const append_job *)arg;
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const size_t row = (size_t)ctx->n_kv * ctx->head_dim;
  for (uint32_t u = i; u < job->units; u += n) {
    const uint32_t h = u % ctx->n_kv, r0 = (u / ctx->n_kv) * APPEND_ROWS;
    const uint32_t r1 =
      r0 + APPEND_ROWS < job->n_rows ? r0 + APPEND_ROWS : job->n_rows;
    for (uint32_t r = r0; r < r1; ++r) {
      (ctx->q8 ? append_head_q8 : append_head)(
        ctx, job->layer, h, job->kv_from + r,
        job->k_rows + r * row + (size_t)h * ctx->head_dim,
        job->v_rows + r * row + (size_t)h * ctx->head_dim);
    }
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
  if (layer_alloc(ctx, layer) != AEE_SUCCESS) {
    return AEE_ENOMEMORY;
  }
  if (ctx->q8 && n_rows != 0u && !ctx->scaled[layer]) { /* [#282 D] */
    q8_set_scales(ctx, layer, n_rows, k_rows, v_rows, HVX_ATTN_M1_Q8_MARGIN);
  }
  {
    append_job job = {ctx,
                      layer,
                      kv_from,
                      n_rows,
                      ctx->n_kv * ((n_rows + APPEND_ROWS - 1u) / APPEND_ROWS),
                      k_rows,
                      v_rows};
    hvx_worker_pool_run(ctx->pool, run_append, &job, job.units);
  }
  ctx->kv_len[layer] = kv_from + n_rows;
  return AEE_SUCCESS;
}

/** [#289] seed_q8 over the pool: units are (kv head, 64-row chunk) */
typedef struct {
  hvx_attn_m1_ctx *ctx;
  uint32_t layer, n_rows, units;
  const uint8_t *k8, *v8;
  size_t head_stride, row_stride;
  const float *k_rows; /**< non-NULL: K quantized from these instead */
} seed_job;

static void run_seed_q8(uint32_t n, uint32_t i, void *arg) {
  const seed_job *job = (const seed_job *)arg;
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const uint32_t hd = ctx->head_dim;
  const float *ski_l = ctx->sk_inv + (size_t)job->layer * ctx->n_kv * hd;
  for (uint32_t u = i; u < job->units; u += n) {
    const uint32_t h = u % ctx->n_kv, r0 = (u / ctx->n_kv) * APPEND_ROWS;
    const uint32_t r1 =
      r0 + APPEND_ROWS < job->n_rows ? r0 + APPEND_ROWS : job->n_rows;
    for (uint32_t r = r0; r < r1; ++r) {
      const uint8_t *ks = job->k8 + h * job->head_stride + r * job->row_stride;
      const uint8_t *vs = job->v8 + h * job->head_stride + r * job->row_stride;
      int8_t *kt = kt8_head(ctx, job->layer, h) +
                   (size_t)(r / TILE) * hd * TILE + r % TILE;
      int8_t *vr = v8_head(ctx, job->layer, h) + (size_t)r * hd;
      if (job->k_rows) { /* append_head_q8's K: the same bytes */
        const float *k = job->k_rows + ((size_t)r * ctx->n_kv + h) * hd;
        const float *ski = ski_l + (size_t)h * hd;
        for (uint32_t d = 0; d < hd; ++d) {
          kt[(size_t)d * TILE] = q8_of(k[d], ski[d]);
        }
      } else {
        for (uint32_t d = 0; d < hd; ++d) {
          kt[(size_t)d * TILE] = (int8_t)(ks[d] ^ 0x80u);
        }
      }
      for (uint32_t d = 0; d < hd; ++d) {
        vr[d] = (int8_t)(vs[d] ^ 0x80u);
      }
    }
  }
}

int hvx_attn_m1_seed_q8(hvx_attn_m1_ctx *ctx, uint32_t layer, uint32_t n_rows,
                        const uint8_t *k8, const uint8_t *v8,
                        size_t head_stride, size_t row_stride, const float *s_k,
                        const float *s_v, const float *s_v_inv,
                        const float *k_rows) {
  if (!ctx || !ctx->q8 || layer >= ctx->n_layers || ctx->kv_len[layer] != 0u ||
      !k8 || !v8 || !s_k || !s_v || !s_v_inv) {
    return AEE_EBADSTATE;
  }
  if (n_rows == 0u || n_rows > ctx->max_seq) {
    return AEE_EINVALIDFORMAT;
  }
  if (layer_alloc(ctx, layer) != AEE_SUCCESS) {
    return AEE_ENOMEMORY;
  }
  const uint32_t hd = ctx->head_dim;
  const size_t base = (size_t)layer * ctx->n_kv * hd;
  if (k_rows) {
    /* K's scales as kv_append makes them (the amax pass reads K for both
       halves; sv is overwritten below) */
    q8_set_scales(ctx, layer, n_rows, k_rows, k_rows, HVX_ATTN_M1_Q8_MARGIN);
  }
  for (uint32_t h = 0; h < ctx->n_kv; ++h) {
    for (uint32_t d = 0; d < hd; ++d) {
      const size_t x = base + (size_t)h * hd + d;
      if (!k_rows) {
        ctx->sk[x] = s_k[h];
        ctx->sk_inv[x] = 1.0f / s_k[h]; /* the prefill append's inverse */
      }
      ctx->sv[x] = s_v[(size_t)h * hd + d];
      ctx->sv_inv[x] = s_v_inv[(size_t)h * hd + d];
    }
  }
  ctx->scaled[layer] = 1u;
  seed_job job = {
    ctx,         layer,
    n_rows,      ctx->n_kv * ((n_rows + APPEND_ROWS - 1u) / APPEND_ROWS),
    k8,          v8,
    head_stride, row_stride,
    k_rows};
  hvx_worker_pool_run(ctx->pool, run_seed_q8, &job, job.units);
  ctx->kv_len[layer] = n_rows;
  return AEE_SUCCESS;
}

/** @brief One lane's phase pcycles (#146), summed over the three runs:
 *         each lane writes only its own slot. */
typedef struct {
  uint64_t start; /**< P1's start */
  uint64_t busy;  /**< sum of the runs' (end - start) */
  uint32_t scores, softmax, pv;
  uint32_t exp, et, div; /**< SOFTMAX's pieces (round 3) */
  uint32_t lanes;        /**< P1's n */
} prof_slot;

/** @brief Lanes the phase words record. ponytail: a pool of more than 15
 *         workers would leave its lanes >= 16 out of the sums (the device
 *         has 6 HVX contexts, the host check 8); widen the array then. */
#define PROF_SLOTS 16u

/* The l2fetch leads (plan 170 round 2 sections 3.1-3.3). A lead is a hint
   with no effect on any value; each is compiled out with -D<name>=0 if its
   probe cell reads slower than no lead (LEDGER rule 31). A lane keeps at
   most three boxes queued (the hardware stalls the thread on a fourth).
   S4 (round 3): the P1 lead is worth 649 k -> 387 k of cold scores at pos
   1023; the caller's ET lead read within 3 % of none and was removed. */
#ifndef ATTN_M1_P1_LEAD
#define ATTN_M1_P1_LEAD 1 /**< P1: the lane's next Kt tile */
#endif
#ifndef ATTN_M1_PV_LEAD
#define ATTN_M1_PV_LEAD 1 /**< P3: the V rows two 16 KiB blocks ahead */
#endif
/** @brief V rows per PV fetch block (16 KiB), S1's FETCH_L2F shape. */
#define PV_BLOCK 128u

/** @brief l2fetch of @a rows 128-byte rows at @a p (Rtt: [47:32] stride,
 *         [31:16] width, [15:0] height; the height clamped to 16 bits). */
static inline void l2fetch_rows(const void *p, uint32_t rows) {
  Q6_l2fetch_AP((void *)p, (128ull << 32) | (128ull << 16) |
                             (rows < 0xFFFFu ? rows : 0xFFFFu));
}

/** @brief l2fetch of @a rows rows of @a width bytes, @a width apart (a V
 *         row is head_dim / 64 vectors). */
static inline void l2fetch_box(const void *p, uint32_t width, uint32_t rows) {
  Q6_l2fetch_AP((void *)p, ((uint64_t)width << 32) | ((uint64_t)width << 16) |
                             (rows < 0xFFFFu ? rows : 0xFFFFu));
}

typedef struct {
  const hvx_attn_m1_ctx *ctx;
  uint32_t layer, L, ntl; /**< positions, and their tiles */
  uint32_t n_live;        /**< live lanes of the last tile, 1..64 */
  uint32_t lo, tlo, ntt;  /**< [plan 201 S4] the first visible position (the
                               window's), its tile, tiles tlo .. ntl - 1 */
  HVX_Vector scale;       /**< hf splat */
  uint16_t m[MAX_NQ];     /**< hf max per q head, + 0 (P2) */
  float l[MAX_NQ];        /**< the sums (P3) */
  float *out;
  prof_slot *slots; /**< NULL unless the phase words were requested */
  uint32_t units;
} forward_job;

/** @brief The Kt tile of P1 / P2 unit u: kv head u / ntt, tile tlo + u %
 *         ntt (with no window tlo = 0 and ntt = ntl). */
static inline HVX_Vector *unit_tile(const forward_job *job, uint32_t u) {
  if (job->ctx->q8) { /* [#282 D] the int8 tile: half the bytes */
    return (HVX_Vector *)(kt8_head(job->ctx, job->layer, u / job->ntt) +
                          (size_t)(job->tlo + u % job->ntt) *
                            job->ctx->head_dim * TILE);
  }
  return (HVX_Vector *)(kt_head(job->ctx, job->layer, u / job->ntt) +
                        (size_t)(job->tlo + u % job->ntt) * job->ctx->head_dim *
                          TILE);
}

/** @brief The visible lanes of tile t, [a, b): the window's first position
 *         in tile tlo, the last position in tile ntl - 1. @return 1 when
 *         that is not the whole tile, with *q its predicate. */
static inline int tile_live(const forward_job *job, uint32_t t,
                            HVX_VectorPred *q) {
  const uint32_t a = t == job->tlo ? job->lo % TILE : 0u;
  const uint32_t b = t + 1u == job->ntl ? job->n_live : TILE;
  if (a == 0u && b == TILE) {
    return 0;
  }
  *q = Q6_Q_vsetq2_R((int)(2u * b));
  if (a != 0u) {
    *q = Q6_Q_and_QQn(*q, Q6_Q_vsetq2_R((int)(2u * a)));
  }
  return 1;
}

/** @brief P1 unit u: kv head u / ntl, tile u % ntl, every q head of it. The
 *         last tile first takes the new position's k column (spec step 1),
 *         from the row the caller rounded. */
static void p1_unit(const forward_job *job, uint32_t u) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const uint32_t h = u / job->ntt, t = job->tlo + u % job->ntt;
  const uint32_t hd = ctx->head_dim, nch = hd / CH;
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  HVX_Vector *kt = unit_tile(job, u);
  if (t + 1u == job->ntl) { /* lane pos % 64 = n_live - 1 of each d row */
    const uint32_t j = job->n_live - 1u;
    HVX_VectorPred lane = Q6_Q_vsetq2_R((int)(2u * j + 2u));
    if (j) {
      lane = Q6_Q_and_QQn(lane, Q6_Q_vsetq2_R((int)(2u * j)));
    }
    const uint16_t *kr = ctx->kr + (size_t)h * hd;
    for (uint32_t d = 0; d < hd; ++d) {
      kt[d] = Q6_V_vmux_QVV(lane, Q6_Vh_vsplat_R(kr[d]), kt[d]);
    }
  }
  const HVX_Vector b2 = Q6_Vb_vsplat_R(2), b8 = Q6_Vb_vsplat_R(8);
  for (uint32_t g = 0; g < ctx->gqa; ++g) {
    const uint32_t hq = h * ctx->gqa + g;
    const HVX_Vector *qrows = (const HVX_Vector *)ctx->qs + (size_t)hq * nch;
    /* acc[l] sees only d = 8 k + l, so the eight chains run as two passes
       of four (the same operations per chain; eight chains at once spill
       on hexagon-clang 19). q[d], q[d + 1] are one vlut16 of the caller's
       zipped row with index bytes (d, d + 1), q[d + 2], q[d + 3] one more
       at + 2, the next step + 8. [plan 201 S4] A head of more than 64 is
       head_dim / 64 zipped rows: the chains run on through each in turn
       (64 is a multiple of 8, so d % 8 is the chunk's own d % 8). */
    HVX_Vector a0, a1, a2, a3, a4, a5, a6, a7;
    for (uint32_t half = 0; half < 2u; ++half) {
      HVX_Vector c0 = Q6_V_vzero(), c1 = c0, c2 = c0, c3 = c0;
      for (uint32_t c = 0; c < nch; ++c) {
        const HVX_Vector *kp = kt + (size_t)c * CH + 4u * half;
        const HVX_Vector qrow = qrows[c];
        HVX_Vector i01 =
          Q6_Vh_vsplat_R((int)((4u * half) | (4u * half + 1u) << 8));
#if defined(__hexagon__)
#pragma unroll 1
#endif
        for (uint32_t d = 0; d < CH; d += ATTN_M1_DET_ACC) {
          const int rt = (int)((d + 4u * half) >> 4); /* the same for all 4 */
          const HVX_VectorPair q01 = hvx_hf_splat2_lut(i01, qrow, rt);
          const HVX_VectorPair q23 =
            hvx_hf_splat2_lut(Q6_Vb_vadd_VbVb(i01, b2), qrow, rt);
          c0 = hvx_hf_fma(c0, Q6_V_lo_W(q01), kp[d], one);
          c1 = hvx_hf_fma(c1, Q6_V_hi_W(q01), kp[d + 1u], one);
          c2 = hvx_hf_fma(c2, Q6_V_lo_W(q23), kp[d + 2u], one);
          c3 = hvx_hf_fma(c3, Q6_V_hi_W(q23), kp[d + 3u], one);
          i01 = Q6_Vb_vadd_VbVb(i01, b8);
        }
      }
      if (half == 0u) {
        a0 = c0, a1 = c1, a2 = c2, a3 = c3;
      } else {
        a4 = c0, a5 = c1, a6 = c2, a7 = c3;
      }
    }
    const HVX_Vector acc[ATTN_M1_DET_ACC] = {a0, a1, a2, a3, a4, a5, a6, a7};
    *(HVX_Vector *)(ctx->s + (size_t)hq * ctx->seq + (size_t)t * TILE) =
      hvx_hf_score(acc, job->scale);
  }
}

/**
 * @brief P2 unit u: e = exp16(s - m) of its tile for every q head, masked
 *        past L, back into S and into the ET rows.
 *
 * ET row p holds the q heads in lanes. The unit's heads are adjacent lanes
 * hq0 .. hq0 + 3 of every row of its tile, so four e vectors (positions in
 * lanes) are transposed in groups of four heads: a halfword and then a
 * word vshuff make 8-byte tuples (e0[i], e1[i], e2[i], e3[i]), 16 per
 * vector; a vror puts tuple i at byte 2 * hq0 and one masked store writes
 * those 2 * (heads) bytes of row i. Moves only: every lane of ET gets the
 * bits the scalar scatter wrote, and the lanes past n_q are never in a mask.
 */
static void p2_unit(const forward_job *job, uint32_t u, prof_slot *ps) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const uint32_t h = u / job->ntt, t = job->tlo + u % job->ntt;
  const uint32_t live = t + 1u == job->ntl ? job->n_live : TILE;
  /* the lanes past L, and [plan 201 S4] before the window's first */
  HVX_VectorPred mask = Q6_Q_vsetq2_R((int)(live * 2u));
  const int partial = tile_live(job, t, &mask);
  HVX_Vector e[MAX_GQA];
  for (uint32_t g = 0; g < MAX_GQA; ++g) {
    e[g] = Q6_V_vzero();
  }
  const uint64_t t0 = ps ? HAP_perf_get_pcycles() : 0u;
#if ATTN_M1_EXP_TAB
  /* The unit's index vectors are stored first and gathered in one scalar
     loop, then reloaded: the HVX-store -> scalar-load hazard is paid once
     per unit, not per head (the v79 ISS: 225 vs 325 pcycles per vector). */
  union {
    HVX_Vector v[MAX_GQA];
    uint16_t h[MAX_GQA * TILE];
  } ix, ev;
  for (uint32_t g = 0; g < ctx->gqa; ++g) {
    const uint32_t hq = h * ctx->gqa + g;
    const HVX_Vector *row =
      (const HVX_Vector *)(ctx->s + (size_t)hq * ctx->seq + (size_t)t * TILE);
    ix.v[g] =
      hvx_hf_exp16_idx(Q6_Vhf_vsub_VhfVhf(*row, Q6_Vh_vsplat_R(job->m[hq])));
  }
  const uint16_t *restrict tab = ctx->exp_tab;
  const uint32_t n_ix = ctx->gqa * TILE;
#if defined(__hexagon__)
#pragma unroll 8
#endif
  for (uint32_t i = 0; i < n_ix; ++i) {
    ev.h[i] = tab[ix.h[i]];
  }
  for (uint32_t g = 0; g < ctx->gqa; ++g) {
    const uint32_t hq = h * ctx->gqa + g;
    HVX_Vector *row =
      (HVX_Vector *)(ctx->s + (size_t)hq * ctx->seq + (size_t)t * TILE);
    e[g] = partial ? Q6_V_vmux_QVV(mask, ev.v[g], Q6_V_vzero()) : ev.v[g];
    *row = e[g];
  }
#else
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  for (uint32_t g = 0; g < ctx->gqa; ++g) {
    const uint32_t hq = h * ctx->gqa + g;
    HVX_Vector *row =
      (HVX_Vector *)(ctx->s + (size_t)hq * ctx->seq + (size_t)t * TILE);
    const HVX_Vector d = Q6_Vhf_vsub_VhfVhf(*row, Q6_Vh_vsplat_R(job->m[hq]));
    e[g] = hvx_hf_exp16(d, one);
    if (partial) {
      e[g] = Q6_V_vmux_QVV(mask, e[g], Q6_V_vzero());
    }
    *row = e[g];
  }
#endif
  const uint64_t t1 = ps ? HAP_perf_get_pcycles() : 0u;
  HVX_Vector *et = (HVX_Vector *)ctx->et + (size_t)t * TILE;
  for (uint32_t g0 = 0; g0 < ctx->gqa; g0 += 4u) {
    const uint32_t hq0 = h * ctx->gqa + g0;
    const uint32_t cnt = ctx->gqa - g0 < 4u ? ctx->gqa - g0 : 4u;
    const HVX_VectorPair w01 = Q6_W_vshuff_VVR(e[g0 + 1u], e[g0], -2);
    const HVX_VectorPair w23 = Q6_W_vshuff_VVR(e[g0 + 3u], e[g0 + 2u], -2);
    const HVX_VectorPair lo =
      Q6_W_vshuff_VVR(Q6_V_lo_W(w23), Q6_V_lo_W(w01), -4);
    const HVX_VectorPair hi =
      Q6_W_vshuff_VVR(Q6_V_hi_W(w23), Q6_V_hi_W(w01), -4);
    const HVX_Vector tup[4] = {Q6_V_lo_W(lo), Q6_V_hi_W(lo), Q6_V_lo_W(hi),
                               Q6_V_hi_W(hi)};
    HVX_VectorPred lanes = Q6_Q_vsetq2_R((int)(2u * (hq0 + cnt)));
    if (hq0) {
      lanes = Q6_Q_and_QQn(lanes, Q6_Q_vsetq2_R((int)(2u * hq0)));
    }
    for (uint32_t b = 0; b < 4u && 16u * b < live; ++b) {
      const HVX_Vector tv = tup[b];
      const uint32_t n = live - 16u * b < 16u ? live - 16u * b : 16u;
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t rot = (8u * i - 2u * hq0) & (VLEN - 1u);
        Q6_vmem_QRIV(lanes, et + 16u * b + i, Q6_V_vror_VR(tv, (int)rot));
      }
    }
  }
  if (ps) {
    ps->exp += (uint32_t)(t1 - t0);
    ps->et += (uint32_t)(HAP_perf_get_pcycles() - t1);
  }
}

/** @brief One walk of P3 over the window's V rows for q heads hq0 .. hq0 +
 *         ng - 1 of kv head h (rows @a vr) and the 64-wide chunks c0 .. c0 +
 *         cg - 1 (ng, cg literal, ng x cg <= PV_ACC chains in registers):
 *         p ascending from the window's first in blocks of PV_BLOCK, the
 *         row's cg vectors loaded once for the ng heads, o[g][c] = fma(o,
 *         splat(p[g][p]), V[p][c]). Each (head, d) chain is the spec's
 *         sequence in p order, so the walk order over chunks moves no bit.
 */
static inline __attribute__((always_inline)) void
pv_walk(const forward_job *job, const HVX_Vector *vr,
        const uint16_t *p[PV_GROUP], uint32_t hq0, uint32_t c0,
        const uint32_t ng, const uint32_t cg) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  const uint32_t L = job->L, lo = job->lo;
  const uint32_t hd = ctx->head_dim, nch = hd / CH;
  const HVX_Vector *vc = vr + c0;
#if ATTN_M1_PV_LEAD
  l2fetch_box(vr + (size_t)lo * nch, nch * VLEN,
              L - lo < 2u * PV_BLOCK ? L - lo : 2u * PV_BLOCK);
#endif
  HVX_Vector o[PV_ACC];
  for (uint32_t i = 0; i < ng * cg; ++i) {
    o[i] = Q6_V_vzero();
  }
  for (uint32_t q0 = lo; q0 < L; q0 += PV_BLOCK) {
#if ATTN_M1_PV_LEAD
    if (q0 + 2u * PV_BLOCK < L) {
      const uint32_t rest = L - q0 - 2u * PV_BLOCK;
      l2fetch_box(vr + (size_t)(q0 + 2u * PV_BLOCK) * nch, nch * VLEN,
                  rest < PV_BLOCK ? rest : PV_BLOCK);
    }
#endif
    const HVX_Vector *vb = vc + (size_t)q0 * nch;
    const uint16_t *b[PV_GROUP];
    for (uint32_t g = 0; g < ng; ++g) {
      b[g] = p[g] + q0;
    }
    const uint32_t nb = L - q0 < PV_BLOCK ? L - q0 : PV_BLOCK;
    /* No software pipelining: with eight chains hexagon-clang 19's
       pipeliner carries a product across the back edge as sf and back
       (vadd(sf, 0)), the sequence S1 never measured on silicon; with it
       off no pv_walk_* position loop has an .sf operand (objdump). */
#pragma clang loop pipeline(disable)
    for (uint32_t q = 0; q < nb; ++q) {
      const HVX_Vector *vq = vb + (size_t)q * nch;
      HVX_Vector sp[PV_GROUP];
      for (uint32_t g = 0; g < ng; ++g) {
        sp[g] = Q6_Vh_vsplat_R(b[g][q]);
      }
      for (uint32_t c = 0; c < cg; ++c) {
        const HVX_Vector v = vq[c];
        for (uint32_t g = 0; g < ng; ++g) {
          o[g * cg + c] = hvx_hf_fma(o[g * cg + c], sp[g], v, one);
        }
      }
    }
  }
  for (uint32_t g = 0; g < ng; ++g) {
    for (uint32_t c = 0; c < cg; ++c) {
      hvx_hf_store_sf(job->out + (size_t)(hq0 + g) * hd + (size_t)(c0 + c) * CH,
                      o[g * cg + c]);
    }
  }
}

/* One function per (ng, cg), each walk's chains in their own register
   allocation: inlined together into run_p3, hexagon-clang 19 spilled the
   eight-chain loops' accumulators to the stack (bit-safe, but which loop
   moved with unrelated edits). */
#define PV_WALK_FN(NG, CG)                                                     \
  static __attribute__((noinline)) void pv_walk_##NG##_##CG(                   \
    const forward_job *job, const HVX_Vector *vr, const uint16_t *p[PV_GROUP], \
    uint32_t hq0, uint32_t c0) {                                               \
    pv_walk(job, vr, p, hq0, c0, NG##u, CG##u);                                \
  }
PV_WALK_FN(4, 2)
PV_WALK_FN(4, 1)
PV_WALK_FN(2, 4)
PV_WALK_FN(2, 1)
PV_WALK_FN(1, 1)

/** @brief P3 for q heads g0 .. g0 + ng - 1 of kv head h (ng literal at the
 *         call sites, so the chains live in registers): the divides, then
 *         PV by pv_walk, nch / cg walks over the V rows. */
static inline __attribute__((always_inline)) void
pv_group(const forward_job *job, uint32_t h, uint32_t g0, const uint32_t ng,
         prof_slot *ps, uint64_t *t) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  const uint32_t hq0 = h * ctx->gqa + g0, nch = ctx->head_dim / CH;
  const HVX_Vector *vr = (const HVX_Vector *)v_head(ctx, job->layer, h);
  const uint16_t *p[PV_GROUP];
  for (uint32_t g = 0; g < ng; ++g) {
    uint16_t *row = ctx->s + (size_t)(hq0 + g) * ctx->seq;
    const HVX_Vector lv = hvx_splat_sf(job->l[hq0 + g]);
    const HVX_Vector rv = hvx_recip_det_sf(lv);
    const HVX_VectorPair lw = Q6_W_vcombine_VV(lv, lv);
    const HVX_VectorPair rw = Q6_W_vcombine_VV(rv, rv);
    for (uint32_t b = job->tlo; b < job->ntl; ++b) {
      HVX_Vector *e = (HVX_Vector *)(row + (size_t)b * TILE);
      *e = hvx_hf_div16(*e, lw, rw, one);
    }
    p[g] = row;
  }
  if (ps) {
    const uint64_t now = HAP_perf_get_pcycles();
    ps->softmax += (uint32_t)(now - *t);
    ps->div += (uint32_t)(now - *t);
    *t = now;
  }
  /* [#261] Every chain of a walk takes the V row's vectors of its cg
     chunks from one load of the row, so a V row of nch chunks is walked
     nch / cg times instead of nch (the 261 sitting: PV at 64-71 lane
     pcycles per FMA on silicon, 11.4 on the ISS). Gemma 4's sliding shape
     (gqa 2, hd 256) walks once, the full one (gqa 8 in groups of 4, hd
     512) four times, LFM's hd 64 once as before. ponytail: one head (odd
     gqa only, no host case beyond hd 64) and an nch that PV_CG(ng) does
     not divide (hd 128, 192, ...) walk per chunk as before; a pv_walk_1_8
     and a halving cg, each with a host shape, when such a model comes. */
  const int whole = ng > 1u && nch % PV_CG(ng) == 0u;
  const uint32_t cg = whole ? PV_CG(ng) : 1u;
  for (uint32_t c0 = 0; c0 < nch; c0 += cg) {
    if (ng == 4u) {
      (whole ? pv_walk_4_2 : pv_walk_4_1)(job, vr, p, hq0, c0);
    } else if (ng == 2u) {
      (whole ? pv_walk_2_4 : pv_walk_2_1)(job, vr, p, hq0, c0);
    } else {
      pv_walk_1_1(job, vr, p, hq0, c0);
    }
  }
  if (ps) {
    const uint64_t now = HAP_perf_get_pcycles();
    ps->pv += (uint32_t)(now - *t);
    *t = now;
  }
}

/** @brief [#282 D] p1_unit over the int8 cache: the new k column is in the
 *         tile already (forward wrote it), q carries K's scale. Rows d,
 *         d + 1 of a tile are one vector: unpacked to halfwords and
 *         converted to hf, then the fp16 path's chains. */
static void p1_unit_q8(const forward_job *job, uint32_t u) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const uint32_t h = u / job->ntt, t = job->tlo + u % job->ntt;
  const uint32_t hd = ctx->head_dim, nch = hd / CH;
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  const HVX_Vector *tv =
    (const HVX_Vector *)(kt8_head(ctx, job->layer, h) + (size_t)t * hd * TILE);
  const HVX_Vector b2 = Q6_Vb_vsplat_R(2), b8 = Q6_Vb_vsplat_R(8);
  for (uint32_t g = 0; g < ctx->gqa; ++g) {
    const uint32_t hq = h * ctx->gqa + g;
    const HVX_Vector *qrows = (const HVX_Vector *)ctx->qs + (size_t)hq * nch;
    HVX_Vector a0, a1, a2, a3, a4, a5, a6, a7;
    for (uint32_t half = 0; half < 2u; ++half) {
      HVX_Vector c0 = Q6_V_vzero(), c1 = c0, c2 = c0, c3 = c0;
      for (uint32_t c = 0; c < nch; ++c) {
        const HVX_Vector *kp = tv + ((size_t)c * CH + 4u * half) / 2u;
        const HVX_Vector qrow = qrows[c];
        HVX_Vector i01 =
          Q6_Vh_vsplat_R((int)((4u * half) | (4u * half + 1u) << 8));
#if defined(__hexagon__)
#pragma unroll 1
#endif
        for (uint32_t d = 0; d < CH; d += ATTN_M1_DET_ACC) {
          const int rt = (int)((d + 4u * half) >> 4);
          const HVX_VectorPair q01 = hvx_hf_splat2_lut(i01, qrow, rt);
          const HVX_VectorPair q23 =
            hvx_hf_splat2_lut(Q6_Vb_vadd_VbVb(i01, b2), qrow, rt);
          const HVX_VectorPair k01 = Q6_Wh_vunpack_Vb(kp[d / 2u]);
          const HVX_VectorPair k23 = Q6_Wh_vunpack_Vb(kp[d / 2u + 1u]);
          c0 = hvx_hf_fma(c0, Q6_V_lo_W(q01), Q6_Vhf_equals_Vh(Q6_V_lo_W(k01)),
                          one);
          c1 = hvx_hf_fma(c1, Q6_V_hi_W(q01), Q6_Vhf_equals_Vh(Q6_V_hi_W(k01)),
                          one);
          c2 = hvx_hf_fma(c2, Q6_V_lo_W(q23), Q6_Vhf_equals_Vh(Q6_V_lo_W(k23)),
                          one);
          c3 = hvx_hf_fma(c3, Q6_V_hi_W(q23), Q6_Vhf_equals_Vh(Q6_V_hi_W(k23)),
                          one);
          i01 = Q6_Vb_vadd_VbVb(i01, b8);
        }
      }
      if (half == 0u) {
        a0 = c0, a1 = c1, a2 = c2, a3 = c3;
      } else {
        a4 = c0, a5 = c1, a6 = c2, a7 = c3;
      }
    }
    const HVX_Vector acc[ATTN_M1_DET_ACC] = {a0, a1, a2, a3, a4, a5, a6, a7};
    *(HVX_Vector *)(ctx->s + (size_t)hq * ctx->seq + (size_t)t * TILE) =
      hvx_hf_score(acc, job->scale);
  }
}

/** @brief [#282 E] Two q heads' PV over a 256-wide int8 V row in one walk:
 *         the row's two vectors unpacked to four hf chunks, eight chains.
 *         noinline and no pipelining, as the fp16 pv_walk_* shapes. */
static __attribute__((noinline)) void
pv_walk_q8_2x2(const forward_job *job, const int8_t *vr,
               const uint16_t *const *p, uint32_t hq0, const float *sv) {
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  const uint32_t L = job->L, lo = job->lo, hd = 256u;
  HVX_Vector a0 = Q6_V_vzero(), a1 = a0, a2 = a0, a3 = a0;
  HVX_Vector b0 = a0, b1 = a0, b2 = a0, b3 = a0;
#if ATTN_M1_PV_LEAD
  l2fetch_box(vr + (size_t)lo * hd, hd,
              L - lo < 2u * PV_BLOCK ? L - lo : 2u * PV_BLOCK);
#endif
  for (uint32_t q0 = lo; q0 < L; q0 += PV_BLOCK) {
#if ATTN_M1_PV_LEAD
    if (q0 + 2u * PV_BLOCK < L) {
      const uint32_t rest = L - q0 - 2u * PV_BLOCK;
      l2fetch_box(vr + (size_t)(q0 + 2u * PV_BLOCK) * hd, hd,
                  rest < PV_BLOCK ? rest : PV_BLOCK);
    }
#endif
    const uint16_t *pa = p[0] + q0, *pb = p[1] + q0;
    const uint32_t nb = L - q0 < PV_BLOCK ? L - q0 : PV_BLOCK;
#if defined(__hexagon__)
#pragma clang loop pipeline(disable)
#endif
    for (uint32_t q = 0; q < nb; ++q) {
      const HVX_Vector *row = (const HVX_Vector *)(vr + (size_t)(q0 + q) * hd);
      const HVX_VectorPair w0 = Q6_Wh_vunpack_Vb(row[0]);
      const HVX_VectorPair w1 = Q6_Wh_vunpack_Vb(row[1]);
      const HVX_Vector v0 = Q6_Vhf_equals_Vh(Q6_V_lo_W(w0));
      const HVX_Vector v1 = Q6_Vhf_equals_Vh(Q6_V_hi_W(w0));
      const HVX_Vector v2 = Q6_Vhf_equals_Vh(Q6_V_lo_W(w1));
      const HVX_Vector v3 = Q6_Vhf_equals_Vh(Q6_V_hi_W(w1));
      const HVX_Vector sa = Q6_Vh_vsplat_R(pa[q]), sb = Q6_Vh_vsplat_R(pb[q]);
      a0 = hvx_hf_fma(a0, sa, v0, one);
      a1 = hvx_hf_fma(a1, sa, v1, one);
      a2 = hvx_hf_fma(a2, sa, v2, one);
      a3 = hvx_hf_fma(a3, sa, v3, one);
      b0 = hvx_hf_fma(b0, sb, v0, one);
      b1 = hvx_hf_fma(b1, sb, v1, one);
      b2 = hvx_hf_fma(b2, sb, v2, one);
      b3 = hvx_hf_fma(b3, sb, v3, one);
    }
  }
  const HVX_Vector o[2][4] = {{a0, a1, a2, a3}, {b0, b1, b2, b3}};
  for (uint32_t g = 0; g < 2u; ++g) {
    float *dst = job->out + (size_t)(hq0 + g) * hd;
    for (uint32_t c = 0; c < 4u; ++c) {
      hvx_hf_store_sf(dst + (size_t)c * CH, o[g][c]);
    }
    for (uint32_t d = 0; d < hd; ++d) {
      dst[d] *= sv[d];
    }
  }
}

/** @brief [#282 D] pv_group over the int8 cache: two 64-wide chunks a
 *         pass (one V vector is 128 dims of one position), V's per-dim
 *         scale applied to the f32 output after the chains. */
static inline __attribute__((always_inline)) void
pv_group_q8(const forward_job *job, uint32_t h, uint32_t g0, const uint32_t ng,
            prof_slot *ps, uint64_t *t) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  const uint32_t hq0 = h * ctx->gqa + g0, L = job->L, lo = job->lo;
  const uint32_t hd = ctx->head_dim, nv = hd / VLEN;
  const int8_t *vr = v8_head(ctx, job->layer, h);
  const float *sv = ctx->sv + ((size_t)job->layer * ctx->n_kv + h) * hd;
  const uint16_t *p[PV_GROUP];
  for (uint32_t g = 0; g < ng; ++g) {
    uint16_t *row = ctx->s + (size_t)(hq0 + g) * ctx->seq;
    const HVX_Vector lv = hvx_splat_sf(job->l[hq0 + g]);
    const HVX_Vector rv = hvx_recip_det_sf(lv);
    const HVX_VectorPair lw = Q6_W_vcombine_VV(lv, lv);
    const HVX_VectorPair rw = Q6_W_vcombine_VV(rv, rv);
    for (uint32_t b = job->tlo; b < job->ntl; ++b) {
      HVX_Vector *e = (HVX_Vector *)(row + (size_t)b * TILE);
      *e = hvx_hf_div16(*e, lw, rw, one);
    }
    p[g] = row;
  }
  if (ps) {
    const uint64_t now = HAP_perf_get_pcycles();
    ps->softmax += (uint32_t)(now - *t);
    ps->div += (uint32_t)(now - *t);
    *t = now;
  }
  /* [#282 E] the PV single walk (760bcaf30) for the int8 cache: two q
     heads x a 256-wide row (Gemma's sliding layers) are 8 chains, one walk
     over the V rows instead of two */
  if (ng == 2u && nv == 2u) {
    pv_walk_q8_2x2(job, vr, p, hq0, sv);
    if (ps) {
      const uint64_t now = HAP_perf_get_pcycles();
      ps->pv += (uint32_t)(now - *t);
      *t = now;
    }
    return;
  }
  for (uint32_t c = 0; c < nv; ++c) { /* dims [128 c, 128 c + 128) */
#if ATTN_M1_PV_LEAD
    l2fetch_box(vr + (size_t)lo * hd, hd,
                L - lo < 2u * PV_BLOCK ? L - lo : 2u * PV_BLOCK);
#endif
    HVX_Vector o0 = Q6_V_vzero(), o1 = o0, o2 = o0, o3 = o0;
    HVX_Vector r0 = o0, r1 = o0, r2 = o0, r3 = o0;
    for (uint32_t q0 = lo; q0 < L; q0 += PV_BLOCK) {
#if ATTN_M1_PV_LEAD
      if (q0 + 2u * PV_BLOCK < L) {
        const uint32_t rest = L - q0 - 2u * PV_BLOCK;
        l2fetch_box(vr + (size_t)(q0 + 2u * PV_BLOCK) * hd, hd,
                    rest < PV_BLOCK ? rest : PV_BLOCK);
      }
#endif
      const uint16_t *b0 = p[0] + q0, *b1 = p[ng > 1u ? 1u : 0u] + q0,
                     *b2 = p[ng > 2u ? 2u : 0u] + q0,
                     *b3 = p[ng > 3u ? 3u : 0u] + q0;
      const uint32_t nb = L - q0 < PV_BLOCK ? L - q0 : PV_BLOCK;
      for (uint32_t q = 0; q < nb; ++q) {
        const HVX_VectorPair w = Q6_Wh_vunpack_Vb(
          *(const HVX_Vector *)(vr + (size_t)(q0 + q) * hd + (size_t)c * VLEN));
        const HVX_Vector va = Q6_Vhf_equals_Vh(Q6_V_lo_W(w));
        const HVX_Vector vb = Q6_Vhf_equals_Vh(Q6_V_hi_W(w));
        const HVX_Vector s0 = Q6_Vh_vsplat_R(b0[q]);
        o0 = hvx_hf_fma(o0, s0, va, one);
        r0 = hvx_hf_fma(r0, s0, vb, one);
        if (ng > 1u) {
          const HVX_Vector s1 = Q6_Vh_vsplat_R(b1[q]);
          o1 = hvx_hf_fma(o1, s1, va, one);
          r1 = hvx_hf_fma(r1, s1, vb, one);
        }
        if (ng > 2u) {
          const HVX_Vector s2 = Q6_Vh_vsplat_R(b2[q]);
          o2 = hvx_hf_fma(o2, s2, va, one);
          r2 = hvx_hf_fma(r2, s2, vb, one);
        }
        if (ng > 3u) {
          const HVX_Vector s3 = Q6_Vh_vsplat_R(b3[q]);
          o3 = hvx_hf_fma(o3, s3, va, one);
          r3 = hvx_hf_fma(r3, s3, vb, one);
        }
      }
    }
    const HVX_Vector o[PV_GROUP] = {o0, o1, o2, o3};
    const HVX_Vector r[PV_GROUP] = {r0, r1, r2, r3};
    for (uint32_t g = 0; g < ng; ++g) {
      float *dst = job->out + (size_t)(hq0 + g) * hd + (size_t)c * VLEN;
      hvx_hf_store_sf(dst, o[g]);
      hvx_hf_store_sf(dst + CH, r[g]);
      for (uint32_t d = 0; d < VLEN; ++d) {
        dst[d] *= sv[(size_t)c * VLEN + d];
      }
    }
  }
  if (ps) {
    const uint64_t now = HAP_perf_get_pcycles();
    ps->pv += (uint32_t)(now - *t);
    *t = now;
  }
}

/** @brief The P1 / P2 runs' pool lane i of n: units i, i + n, ... */
static void run_p1(uint32_t n, uint32_t i, void *arg) {
  const forward_job *job = (const forward_job *)arg;
  prof_slot *ps = (job->slots && i < PROF_SLOTS) ? &job->slots[i] : NULL;
  const uint64_t t0 = ps ? HAP_perf_get_pcycles() : 0u;
  for (uint32_t u = i; u < job->units; u += n) {
#if ATTN_M1_P1_LEAD
    if (u + n < job->units) {
      l2fetch_rows(unit_tile(job, u + n),
                   job->ctx->q8 ? job->ctx->head_dim / 2u : job->ctx->head_dim);
    }
#endif
    if (job->ctx->q8) {
      p1_unit_q8(job, u);
    } else {
      p1_unit(job, u);
    }
  }
  if (ps) {
    const uint32_t dt = (uint32_t)(HAP_perf_get_pcycles() - t0);
    ps->start = t0;
    ps->lanes = n;
    ps->scores += dt;
    ps->busy += dt;
  }
}

static void run_p2(uint32_t n, uint32_t i, void *arg) {
  const forward_job *job = (const forward_job *)arg;
  prof_slot *ps = (job->slots && i < PROF_SLOTS) ? &job->slots[i] : NULL;
  const uint64_t t0 = ps ? HAP_perf_get_pcycles() : 0u;
  for (uint32_t u = i; u < job->units; u += n) {
    p2_unit(job, u, ps);
  }
  if (ps) {
    const uint32_t dt = (uint32_t)(HAP_perf_get_pcycles() - t0);
    ps->softmax += dt;
    ps->busy += dt;
  }
}

/**
 * @brief P3's lane i of n: a range of q-head chains, in whole pairs when
 *        gqa is even (unit = 2 chains, else 1): units [i u / n, (i + 1) u
 *        / n) of u = n_q / unit, walked by kv head in groups of 4, 2 or 1.
 *        Every chain is its own arithmetic, so any split gives the same
 *        bytes. At LFM2.5 on 6 lanes the lanes run 4 or 6 chains (1.125x
 *        the mean) in groups of 4 and 2 only.
 *
 * No group of 3: hexagon-clang 19 compiles pv_group's ng = 3 chain loop
 * with one product moved through sf and back to qf32 (vadd(sf, 0)), a
 * sequence S1 never measured; the 4, 2 and 1 loops are round 1's shape.
 */
static void run_p3(uint32_t n, uint32_t i, void *arg) {
  const forward_job *job = (const forward_job *)arg;
  const uint32_t gqa = job->ctx->gqa, n_q = job->ctx->n_kv * gqa;
  const uint32_t unit = gqa % 2u == 0u ? 2u : 1u, n_units = n_q / unit;
  prof_slot *ps = (job->slots && i < PROF_SLOTS) ? &job->slots[i] : NULL;
  uint64_t t = ps ? HAP_perf_get_pcycles() : 0u;
  const uint64_t t0 = t;
  const uint32_t end = (i + 1u) * n_units / n * unit;
  for (uint32_t hq = i * n_units / n * unit; hq < end;) {
    const uint32_t h = hq / gqa, g0 = hq % gqa, head_end = (h + 1u) * gqa;
    const uint32_t left = (end < head_end ? end : head_end) - hq;
    if (left >= 4u) {
      if (job->ctx->q8) {
        pv_group_q8(job, h, g0, 4u, ps, &t);
      } else {
        pv_group(job, h, g0, 4u, ps, &t);
      }
      hq += 4u;
    } else if (left >= 2u) {
      if (job->ctx->q8) {
        pv_group_q8(job, h, g0, 2u, ps, &t);
      } else {
        pv_group(job, h, g0, 2u, ps, &t);
      }
      hq += 2u;
    } else {
      if (job->ctx->q8) {
        pv_group_q8(job, h, g0, 1u, ps, &t);
      } else {
        pv_group(job, h, g0, 1u, ps, &t);
      }
      hq += 1u;
    }
  }
  if (ps) {
    ps->busy += HAP_perf_get_pcycles() - t0;
  }
}

int hvx_attn_m1_forward(hvx_attn_m1_ctx *ctx, uint32_t layer, uint32_t pos,
                        uint32_t window, float scale, const float *q,
                        const float *k, const float *v, float *out,
                        float *stats) {
  return hvx_attn_m1_forward_prof(ctx, layer, pos, window, scale, q, k, v, out,
                                  stats, NULL);
}

int hvx_attn_m1_forward_prof(hvx_attn_m1_ctx *ctx, uint32_t layer, uint32_t pos,
                             uint32_t window, float scale, const float *q,
                             const float *k, const float *v, float *out,
                             float *stats, uint32_t *prof) {
  const uint64_t qt0 = prof ? HAP_perf_get_qtimer_count() : 0u;
  if (!ctx) {
    return AEE_EBADSTATE;
  }
  /* scale must be an fp16 value: the spec's rne16(t * scale) is then the
     one hf multiply (the model's is 0.125). */
  if (layer >= ctx->n_layers || pos >= ctx->max_seq || !q || !k || !v || !out ||
      !(attn_m1_det_rne16(scale) == scale) ||
      !(scale <= 65504.0f && scale >= -65504.0f)) {
    return AEE_EINVALIDFORMAT;
  }
  if (pos > ctx->kv_len[layer]) {
    return AEE_EBADSTATE;
  }
  if (layer_alloc(ctx, layer) != AEE_SUCCESS) {
    return AEE_ENOMEMORY;
  }
  prof_slot slots[PROF_SLOTS];
  uint64_t t0 = prof ? HAP_perf_get_pcycles() : 0u;
  /* Steps 0-1, as vectors (hvx_hf_round_row is hvx_hf_bits_rne per lane):
     v into its row; k kept for P1, which writes the column into the tile
     it reads anyway; q as one zipped row per head, which P1 splats by
     vlut16. */
  const uint32_t hd = ctx->head_dim;
  const uint32_t n_q = ctx->n_kv * ctx->gqa, L = pos + 1u;
  HVX_Vector *qs = (HVX_Vector *)ctx->qs;
  if (ctx->q8) { /* [#282 D] */
    if (!ctx->scaled[layer]) {
      /* ponytail: no prefill seed (a one-token prompt) -- the scales come
         from this row with 4x headroom; a seed is the normal case */
      q8_set_scales(ctx, layer, 1u, k, v, 4.0f);
    }
    for (uint32_t h = 0; h < ctx->n_kv; ++h) {
      append_head_q8(ctx, layer, h, pos, k + (size_t)h * hd,
                     v + (size_t)h * hd);
    }
    float tmp[CH] __attribute__((aligned(128)));
    for (uint32_t i = 0; i < n_q * hd; i += CH) { /* q x K's scale */
      const float *s =
        ctx->sk + ((size_t)layer * ctx->n_kv + i / hd / ctx->gqa) * hd + i % hd;
      for (uint32_t j = 0; j < CH; ++j) {
        tmp[j] = q[i + j] * s[j];
      }
      qs[i / CH] = hvx_hf_round_row_lut(tmp);
    }
  } else {
    for (uint32_t i = 0; i < ctx->n_kv * hd; i += CH) { /* kv head i / hd */
      *(HVX_Vector *)(ctx->kr + i) = hvx_hf_round_row(k + i);
      *(HVX_Vector *)(v_head(ctx, layer, i / hd) + (size_t)pos * hd + i % hd) =
        hvx_hf_round_row(v + i);
    }
    for (uint32_t i = 0; i < n_q * hd; i += CH) { /* q head i / hd's rows */
      qs[i / CH] = hvx_hf_round_row_lut(q + i);
    }
  }
  ctx->kv_len[layer] = pos + 1u;
  forward_job job;
  job.ctx = ctx;
  job.layer = layer;
  job.L = L;
  job.ntl = (L + TILE - 1u) / TILE;
  job.n_live = L - (job.ntl - 1u) * TILE;
  job.lo = attn_m1_det_lo(L, window);
  job.tlo = job.lo / TILE;
  job.ntt = job.ntl - job.tlo;
  job.scale = Q6_Vh_vsplat_R(hvx_hf_bits_rne(scale));
  job.out = out;
  job.slots = prof ? slots : NULL;
  uint32_t max_pc = 0, sum_pc = 0; /* the caller's steps, into SOFTMAX */
  if (prof) {
    memset(slots, 0, sizeof(slots));
    memset(prof, 0, ATTN_M1_PROF_WORDS * sizeof(uint32_t));
    const uint64_t now = HAP_perf_get_pcycles();
    prof[ATTN_M1_PROF_APPEND] = (uint32_t)(now - t0);
    t0 = now;
  }

  job.units = ctx->n_kv * job.ntt;
  hvx_worker_pool_run(ctx->pool, run_p1, &job, job.units);

  /* m per q head: the hf max over its tiles, the last masked, then the
     others ascending ([plan 201 S4] from the window's tile, its lanes
     before the window masked); + 0. */
  uint64_t ts = prof ? HAP_perf_get_pcycles() : 0u;
  const HVX_Vector neg = Q6_Vh_vsplat_R(HF_NEG_MAX);
  HVX_VectorPred last_q = Q6_Q_vsetq2_R((int)(job.n_live * 2u)), first_q;
  const int last_partial = tile_live(&job, job.ntl - 1u, &last_q);
  const int first_partial =
    job.tlo + 1u < job.ntl && tile_live(&job, job.tlo, &first_q);
  for (uint32_t hq = 0; hq < n_q; ++hq) {
    const HVX_Vector *row =
      (const HVX_Vector *)(ctx->s + (size_t)hq * ctx->seq);
    HVX_Vector mx = last_partial ? Q6_V_vmux_QVV(last_q, row[job.ntl - 1u], neg)
                                 : row[job.ntl - 1u];
    for (uint32_t b = job.tlo; b + 1u < job.ntl; ++b) {
      mx = Q6_Vhf_vmax_VhfVhf(mx, (b == job.tlo && first_partial)
                                    ? Q6_V_vmux_QVV(first_q, row[b], neg)
                                    : row[b]);
    }
    for (uint32_t rot = VLEN / 2u; rot >= 2u; rot >>= 1) {
      mx = Q6_Vhf_vmax_VhfVhf(mx, Q6_V_vror_VR(mx, (int)rot));
    }
    const uint16_t m = (uint16_t)Q6_R_vextract_VR(mx, 0);
    job.m[hq] = m == 0x8000u ? 0u : m; /* m + 0: only -0 changes */
  }
  if (prof) {
    max_pc = (uint32_t)(HAP_perf_get_pcycles() - ts);
  }

  hvx_worker_pool_run(ctx->pool, run_p2, &job, job.units);

  /* l: the sequential fp16 sum of every q head at once, one hf add per
     position; lanes past n_q add zeros. */
  ts = prof ? HAP_perf_get_pcycles() : 0u;
  union {
    HVX_Vector v;
    uint16_t h[TILE];
  } lsum;
  lsum.v = Q6_V_vzero();
  const HVX_Vector *et = (const HVX_Vector *)ctx->et;
  for (uint32_t p = job.lo; p < L; ++p) {
    lsum.v = Q6_Vhf_vadd_VhfVhf(lsum.v, et[p]);
  }
  for (uint32_t hq = 0; hq < n_q; ++hq) {
    job.l[hq] = hf_float(lsum.h[hq]);
  }
  if (prof) {
    sum_pc = (uint32_t)(HAP_perf_get_pcycles() - ts);
  }

  hvx_worker_pool_run(ctx->pool, run_p3, &job, n_q);

  if (stats) {
    for (uint32_t hq = 0; hq < n_q; ++hq) {
      stats[2u * hq] = hf_float(job.m[hq]);
      stats[2u * hq + 1u] = job.l[hq];
    }
  }
  if (prof) {
    prof[ATTN_M1_PROF_POOL] = (uint32_t)(HAP_perf_get_pcycles() - t0);
    const uint32_t lanes = slots[0].lanes;
    prof[ATTN_M1_PROF_LANES] = lanes;
    prof[ATTN_M1_PROF_MAX] = max_pc;
    prof[ATTN_M1_PROF_SUM] = sum_pc;
    prof[ATTN_M1_PROF_SOFTMAX] = max_pc + sum_pc;
    for (uint32_t i = 0; i < PROF_SLOTS; ++i) {
      prof[ATTN_M1_PROF_SCORES] += slots[i].scores;
      prof[ATTN_M1_PROF_SOFTMAX] += slots[i].softmax;
      prof[ATTN_M1_PROF_PV] += slots[i].pv;
      prof[ATTN_M1_PROF_EXP] += slots[i].exp;
      prof[ATTN_M1_PROF_ET] += slots[i].et;
      prof[ATTN_M1_PROF_DIV] += slots[i].div;
      if (slots[i].busy > prof[ATTN_M1_PROF_BUSY_MAX]) {
        prof[ATTN_M1_PROF_BUSY_MAX] = (uint32_t)slots[i].busy;
      }
      if (i < lanes && slots[i].start - t0 > prof[ATTN_M1_PROF_START_MAX]) {
        prof[ATTN_M1_PROF_START_MAX] = (uint32_t)(slots[i].start - t0);
      }
    }
    prof[ATTN_M1_PROF_CALL_QT] = (uint32_t)(HAP_perf_get_qtimer_count() - qt0);
  }
  return AEE_SUCCESS;
}
