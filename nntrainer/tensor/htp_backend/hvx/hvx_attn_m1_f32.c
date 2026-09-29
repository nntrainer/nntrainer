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
 *               and masked vector stores (the caller l2fetched ET). Since
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
 *               two 16 KiB blocks ahead; out = o widened to f32 as vectors.
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
/** @brief The one head_dim (#152): one hf vector per head row. */
#define HD 64u
/** @brief Upper bound on q heads per kv head. */
#define MAX_GQA 8u
/** @brief q heads per call: the lanes of the sum vector. */
#define MAX_NQ 64u
/** @brief q heads per PV unit, the chains that share a V row load. */
#define PV_GROUP 4u
/** @brief fp16 bits of -65504, the mask fill for the max. */
#define HF_NEG_MAX 0xFBFF

/* Round 3's two lookups (plan 170 round 3 sections 3.1-3.2), each the same
   bits as round 2's form; 0 compiles round 2's form back for the S4 A/B
   skel (r2w) and goes in the fold. */
#ifndef ATTN_M1_EXP_TAB
#define ATTN_M1_EXP_TAB 1 /**< P2: exp16 by the checked table */
#endif
#ifndef ATTN_M1_Q_LUT
#define ATTN_M1_Q_LUT 1 /**< P1: q[d] by vlut16 from one zipped row */
#endif
/** @brief q bytes per q head: one zipped row, or round 2's 64 splats. */
#define QS_BYTES (ATTN_M1_Q_LUT ? VLEN : HD * VLEN)

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
  if (n_layers == 0u || n_kv == 0u || gqa == 0u || gqa > MAX_GQA ||
      n_kv * gqa > MAX_NQ || head_dim != HD || max_seq == 0u ||
      max_seq % 32u != 0u) {
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
  ctx->kt = (uint16_t *)memalign(VLEN, (size_t)halves * sizeof(uint16_t));
  ctx->v = (uint16_t *)memalign(VLEN, (size_t)halves * sizeof(uint16_t));
  ctx->s = (uint16_t *)memalign(VLEN, (size_t)n_q * seq * sizeof(uint16_t));
  ctx->et = (uint16_t *)memalign(VLEN, (size_t)seq * VLEN);
  ctx->qs = (uint16_t *)memalign(VLEN, (size_t)n_q * QS_BYTES);
  ctx->kr = (uint16_t *)memalign(VLEN, (size_t)n_kv * HD * sizeof(uint16_t));
  ctx->exp_tab = (uint16_t *)malloc(ATTN_M1_DET_EXP_N * sizeof(uint16_t));
  float *tmp = (float *)malloc(ATTN_M1_DET_EXP_N * sizeof(float));
  if (!ctx->kv_len || !ctx->kt || !ctx->v || !ctx->s || !ctx->et || !ctx->qs ||
      !ctx->kr || !ctx->exp_tab || !tmp) {
    free(tmp);
    rc = AEE_ENOMEMORY;
    hvx_attn_m1_free(ctx);
    ctx = NULL;
    goto out;
  }
  hvx_hf_exp16_fill(ctx->exp_tab, tmp);
  free(tmp);
  /* Finite zeros wherever a tile or a sum row is read past the context;
     the lanes of ET past n_q stay 0 for good. */
  memset(ctx->kt, 0, (size_t)halves * sizeof(uint16_t));
  memset(ctx->v, 0, (size_t)halves * sizeof(uint16_t));
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
  free(ctx->v);
  free(ctx->kt);
  free(ctx->kv_len);
  free(ctx);
}

/** @brief Kt of one (layer, kv head): [tiles][head_dim][TILE] fp16. */
static inline uint16_t *kt_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                                uint32_t h) {
  return ctx->kt + ((size_t)layer * ctx->n_kv + h) * HD * ctx->seq;
}

/** @brief V of one (layer, kv head): [seq][head_dim] fp16. */
static inline uint16_t *v_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                               uint32_t h) {
  return ctx->v + ((size_t)layer * ctx->n_kv + h) * ctx->seq * HD;
}

/** @brief Spec steps 0-1 for one kv head of the prefill seed: k and v
 *         rounded to fp16 (a seed row that is already fp16 is unchanged), k
 *         into its tile's column, v as the position's row. forward's own
 *         append is vector (hvx_attn_m1_forward_prof, p1_unit's merge). */
static inline void append_head(const hvx_attn_m1_ctx *ctx, uint32_t layer,
                               uint32_t h, uint32_t pos, const float *k,
                               const float *v) {
  uint16_t *kt =
    kt_head(ctx, layer, h) + (size_t)(pos / TILE) * HD * TILE + pos % TILE;
  uint16_t *vr = v_head(ctx, layer, h) + (size_t)pos * HD;
  for (uint32_t d = 0; d < HD; ++d) {
    kt[(size_t)d * TILE] = hvx_hf_bits_rne(k[d]);
    vr[d] = hvx_hf_bits_rne(v[d]);
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
   most three boxes queued (the hardware stalls the thread on a fourth). */
#ifndef ATTN_M1_P1_LEAD
#define ATTN_M1_P1_LEAD 1 /**< P1: the lane's next Kt tile */
#endif
#ifndef ATTN_M1_PV_LEAD
#define ATTN_M1_PV_LEAD 1 /**< P3: the V rows two 16 KiB blocks ahead */
#endif
#ifndef ATTN_M1_ET_LEAD
#define ATTN_M1_ET_LEAD 1 /**< the caller: the ET rows P2 will write */
#endif
/** @brief V rows per PV fetch block (16 KiB), S1's FETCH_L2F shape. */
#define PV_BLOCK 128u

/** @brief l2fetch of @a rows 128-byte rows at @a p (Rtt: [47:32] stride,
 *         [31:16] width, [15:0] height; the height clamped to 16 bits). */
static inline void l2fetch_rows(const void *p, uint32_t rows) {
  Q6_l2fetch_AP((void *)p, (128ull << 32) | (128ull << 16) |
                             (rows < 0xFFFFu ? rows : 0xFFFFu));
}

typedef struct {
  const hvx_attn_m1_ctx *ctx;
  uint32_t layer, L, ntl; /**< positions, and their tiles */
  uint32_t n_live;        /**< live lanes of the last tile, 1..64 */
  HVX_Vector scale;       /**< hf splat */
  uint16_t m[MAX_NQ];     /**< hf max per q head, + 0 (P2) */
  float l[MAX_NQ];        /**< the sums (P3) */
  float *out;
  prof_slot *slots; /**< NULL unless the phase words were requested */
  uint32_t units;
} forward_job;

/** @brief The Kt tile of P1 / P2 unit u. */
static inline HVX_Vector *unit_tile(const forward_job *job, uint32_t u) {
  return (HVX_Vector *)(kt_head(job->ctx, job->layer, u / job->ntl) +
                        (size_t)(u % job->ntl) * HD * TILE);
}

/** @brief P1 unit u: kv head u / ntl, tile u % ntl, every q head of it. The
 *         last tile first takes the new position's k column (spec step 1),
 *         from the row the caller rounded. */
static void p1_unit(const forward_job *job, uint32_t u) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const uint32_t h = u / job->ntl, t = u % job->ntl;
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  HVX_Vector *kt = unit_tile(job, u);
  if (t + 1u == job->ntl) { /* lane pos % 64 = n_live - 1 of each d row */
    const uint32_t j = job->n_live - 1u;
    HVX_VectorPred lane = Q6_Q_vsetq2_R((int)(2u * j + 2u));
    if (j) {
      lane = Q6_Q_and_QQn(lane, Q6_Q_vsetq2_R((int)(2u * j)));
    }
    const uint16_t *kr = ctx->kr + (size_t)h * HD;
    for (uint32_t d = 0; d < HD; ++d) {
      kt[d] = Q6_V_vmux_QVV(lane, Q6_Vh_vsplat_R(kr[d]), kt[d]);
    }
  }
#if ATTN_M1_Q_LUT
  const HVX_Vector b2 = Q6_Vb_vsplat_R(2), b8 = Q6_Vb_vsplat_R(8);
#endif
  for (uint32_t g = 0; g < ctx->gqa; ++g) {
    const uint32_t hq = h * ctx->gqa + g;
#if ATTN_M1_Q_LUT
    const HVX_Vector qrow = ((const HVX_Vector *)ctx->qs)[hq];
#else
    const HVX_Vector *qs = (const HVX_Vector *)ctx->qs + (size_t)hq * HD;
#endif
    /* acc[l] sees only d = 8 k + l, so the eight chains run as two passes
       of four (the same operations per chain; eight chains at once spill
       on hexagon-clang 19). q[d], q[d + 1] are one vlut16 of the caller's
       zipped row with index bytes (d, d + 1), q[d + 2], q[d + 3] one more
       at + 2, the next step + 8 (round 2: the caller's splat vectors). */
    HVX_Vector a0, a1, a2, a3, a4, a5, a6, a7;
    for (uint32_t half = 0; half < 2u; ++half) {
      const HVX_Vector *kp = kt + 4u * half;
      HVX_Vector c0 = Q6_V_vzero(), c1 = c0, c2 = c0, c3 = c0;
#if ATTN_M1_Q_LUT
      HVX_Vector i01 =
        Q6_Vh_vsplat_R((int)((4u * half) | (4u * half + 1u) << 8));
#if defined(__hexagon__)
#pragma unroll 1
#endif
      for (uint32_t d = 0; d < HD; d += ATTN_M1_DET_ACC) {
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
#else
      const HVX_Vector *qp = qs + 4u * half;
#if defined(__hexagon__)
#pragma unroll 1
#endif
      for (uint32_t d = 0; d < HD; d += ATTN_M1_DET_ACC) {
        c0 = hvx_hf_fma(c0, qp[d], kp[d], one);
        c1 = hvx_hf_fma(c1, qp[d + 1u], kp[d + 1u], one);
        c2 = hvx_hf_fma(c2, qp[d + 2u], kp[d + 2u], one);
        c3 = hvx_hf_fma(c3, qp[d + 3u], kp[d + 3u], one);
      }
#endif
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
  const uint32_t h = u / job->ntl, t = u % job->ntl;
  const uint32_t live = t + 1u == job->ntl ? job->n_live : TILE;
  const HVX_VectorPred mask = Q6_Q_vsetq2_R((int)(live * 2u));
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
    e[g] = live < TILE ? Q6_V_vmux_QVV(mask, ev.v[g], Q6_V_vzero()) : ev.v[g];
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
    if (live < TILE) {
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

/** @brief P3 for q heads g0 .. g0 + ng - 1 of kv head h (ng literal at the
 *         call sites, so the chains live in registers): the divides, then
 *         PV with the V row loaded once per position, the V rows fetched
 *         two 16 KiB blocks ahead of the chain. */
static inline __attribute__((always_inline)) void
pv_group(const forward_job *job, uint32_t h, uint32_t g0, const uint32_t ng,
         prof_slot *ps, uint64_t *t) {
  const hvx_attn_m1_ctx *ctx = job->ctx;
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  const uint32_t hq0 = h * ctx->gqa + g0, L = job->L;
  const HVX_Vector *vr = (const HVX_Vector *)v_head(ctx, job->layer, h);
#if ATTN_M1_PV_LEAD
  l2fetch_rows(vr, L < 2u * PV_BLOCK ? L : 2u * PV_BLOCK);
#endif
  const uint16_t *p[PV_GROUP];
  for (uint32_t g = 0; g < ng; ++g) {
    uint16_t *row = ctx->s + (size_t)(hq0 + g) * ctx->seq;
    const HVX_Vector lv = hvx_splat_sf(job->l[hq0 + g]);
    const HVX_Vector rv = hvx_recip_det_sf(lv);
    const HVX_VectorPair lw = Q6_W_vcombine_VV(lv, lv);
    const HVX_VectorPair rw = Q6_W_vcombine_VV(rv, rv);
    for (uint32_t b = 0; b < job->ntl; ++b) {
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
  HVX_Vector o0 = Q6_V_vzero(), o1 = o0, o2 = o0, o3 = o0;
  /* p ascending in blocks of PV_BLOCK; the chain loop is round 1's. */
  for (uint32_t q0 = 0; q0 < L; q0 += PV_BLOCK) {
#if ATTN_M1_PV_LEAD
    if (q0 + 2u * PV_BLOCK < L) {
      const uint32_t rest = L - q0 - 2u * PV_BLOCK;
      l2fetch_rows(vr + q0 + 2u * PV_BLOCK, rest < PV_BLOCK ? rest : PV_BLOCK);
    }
#endif
    const HVX_Vector *vb = vr + q0;
    const uint16_t *b0 = p[0] + q0, *b1 = p[ng > 1u ? 1u : 0u] + q0,
                   *b2 = p[ng > 2u ? 2u : 0u] + q0,
                   *b3 = p[ng > 3u ? 3u : 0u] + q0;
    const uint32_t nb = L - q0 < PV_BLOCK ? L - q0 : PV_BLOCK;
    for (uint32_t q = 0; q < nb; ++q) {
      const HVX_Vector v = vb[q];
      o0 = hvx_hf_fma(o0, Q6_Vh_vsplat_R(b0[q]), v, one);
      if (ng > 1u) {
        o1 = hvx_hf_fma(o1, Q6_Vh_vsplat_R(b1[q]), v, one);
      }
      if (ng > 2u) {
        o2 = hvx_hf_fma(o2, Q6_Vh_vsplat_R(b2[q]), v, one);
      }
      if (ng > 3u) {
        o3 = hvx_hf_fma(o3, Q6_Vh_vsplat_R(b3[q]), v, one);
      }
    }
  }
  const HVX_Vector o[PV_GROUP] = {o0, o1, o2, o3};
  for (uint32_t g = 0; g < ng; ++g) {
    hvx_hf_store_sf(job->out + (size_t)(hq0 + g) * HD, o[g]);
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
      l2fetch_rows(unit_tile(job, u + n), HD);
    }
#endif
    p1_unit(job, u);
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
      pv_group(job, h, g0, 4u, ps, &t);
      hq += 4u;
    } else if (left >= 2u) {
      pv_group(job, h, g0, 2u, ps, &t);
      hq += 2u;
    } else {
      pv_group(job, h, g0, 1u, ps, &t);
      hq += 1u;
    }
  }
  if (ps) {
    ps->busy += HAP_perf_get_pcycles() - t0;
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
  prof_slot slots[PROF_SLOTS];
  uint64_t t0 = prof ? HAP_perf_get_pcycles() : 0u;
  /* Steps 0-1, as vectors (hvx_hf_round_row is hvx_hf_bits_rne per lane):
     v into its row; k kept for P1, which writes the column into the tile
     it reads anyway; q as one zipped row per head, which P1 splats by
     vlut16 (round 2: one splat vector per value). */
  for (uint32_t h = 0; h < ctx->n_kv; ++h) {
    *(HVX_Vector *)(ctx->kr + (size_t)h * HD) =
      hvx_hf_round_row(k + (size_t)h * HD);
    *(HVX_Vector *)(v_head(ctx, layer, h) + (size_t)pos * HD) =
      hvx_hf_round_row(v + (size_t)h * HD);
  }
  ctx->kv_len[layer] = pos + 1u;

  const uint32_t n_q = ctx->n_kv * ctx->gqa, L = pos + 1u;
  HVX_Vector *qs = (HVX_Vector *)ctx->qs;
  for (uint32_t hq = 0; hq < n_q; ++hq) {
#if ATTN_M1_Q_LUT
    qs[hq] = hvx_hf_round_row_lut(q + (size_t)hq * HD);
#else
    union {
      HVX_Vector v;
      uint16_t h[TILE];
    } qv;
    qv.v = hvx_hf_round_row(q + (size_t)hq * HD);
    for (uint32_t d = 0; d < HD; ++d) {
      *qs++ = Q6_Vh_vsplat_R(qv.h[d]);
    }
#endif
  }
#if ATTN_M1_ET_LEAD
  l2fetch_rows(ctx->et, L);
#endif
  forward_job job;
  job.ctx = ctx;
  job.layer = layer;
  job.L = L;
  job.ntl = (L + TILE - 1u) / TILE;
  job.n_live = L - (job.ntl - 1u) * TILE;
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

  job.units = ctx->n_kv * job.ntl;
  hvx_worker_pool_run(ctx->pool, run_p1, &job, job.units);

  /* m per q head: the hf max over its tiles, the last masked; + 0. */
  uint64_t ts = prof ? HAP_perf_get_pcycles() : 0u;
  const HVX_Vector neg = Q6_Vh_vsplat_R(HF_NEG_MAX);
  const HVX_VectorPred live = Q6_Q_vsetq2_R((int)(job.n_live * 2u));
  for (uint32_t hq = 0; hq < n_q; ++hq) {
    const HVX_Vector *row =
      (const HVX_Vector *)(ctx->s + (size_t)hq * ctx->seq);
    HVX_Vector mx = job.n_live < TILE
                      ? Q6_V_vmux_QVV(live, row[job.ntl - 1u], neg)
                      : row[job.ntl - 1u];
    for (uint32_t b = 0; b + 1u < job.ntl; ++b) {
      mx = Q6_Vhf_vmax_VhfVhf(mx, row[b]);
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
  for (uint32_t p = 0; p < L; ++p) {
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
