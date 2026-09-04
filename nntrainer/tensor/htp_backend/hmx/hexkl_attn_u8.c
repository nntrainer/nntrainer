// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file hexkl_attn_u8.c
 * @date 07 Aug 2026
 * @brief KV block registry and the S = Q.Kt half of the fused attention path
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs except for NYI items
 *
 * hexkl_attn_u8.h for the contract. The host-side executable
 * specification this must agree with is mha_htp_host_forward's PHASE A in
 * test/unittest/mha_htp_host_model.cpp. */

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hexkl_attn_u8.h"
#include "hexkl_kv_quant.h"
#include "hexkl_mm_opts.h"
#include "hexkl_probe.h"
#include "hvx_convert.h"
#include "hvx_softmax_blocked_f32.h"
#include "hvx_worker_pool.h"

/** @brief Bytes one fp16 KV row occupies across all heads. */
static uint32_t row_stride(const hexkl_attn_u8_ctx *ctx) {
  return ctx->nch * ctx->head_dim;
}

/**
 * @brief Microsecond clock, read only when a caller asked for the breakdown.
 *
 * HAP_perf_get_qtimer_count is the DSP's own free-running counter, so this
 * measures DSP-internal time and excludes the FastRPC round trip which is
 * the point: the host side already times the whole call, and the difference
 * between the two is the transport. */
static inline uint64_t now_us(void) {
  return HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count);
}

#define TICK(dst)                                                              \
  do {                                                                         \
    if (stage_us) {                                                            \
      (dst) = now_us;                                                          \
    }                                                                          \
  } while (0)

#define ACCUM(slot, t0, t1)                                                    \
  do {                                                                         \
    if (stage_us) {                                                            \
      stage_us[slot] += (uint32_t)((t1) - (t0));                               \
    }                                                                          \
  } while (0)

uint32_t hexkl_attn_u8_n_blocks(const hexkl_attn_u8_ctx *ctx) {
  return (ctx->kv_len + ctx->T - 1u) / ctx->T;
}

int hexkl_attn_u8_ctx_init(hexkl_attn_u8_ctx *ctx, uint8_t *vtcm_base,
                           uint32_t vtcm_size, uint32_t config_off,
                           uint32_t nch, uint32_t gqa, uint32_t head_dim,
                           uint32_t max_kv, uint32_t T, uint32_t M_band,
                           hexkl_w_width w_k, hexkl_w_width w_v, void *table_k,
                           void *table_v) {
  uint32_t i, n_slots, biggest;

  if (ctx == NULL || nch == 0u || nch > HEXKL_ATTN_MAX_HEADS || gqa == 0u ||
      head_dim == 0u || max_kv == 0u || T == 0u) {
    return AEE_EBADPARM;
  }
  /** hexkl_mm_u8iX_plan enforces K % 32 == 0 and N % 32 == 0; for S those are
   * head_dim and T. Check here rather than letting the first run fail. */
  if ((T % 32u) != 0u || (head_dim % 32u) != 0u) {
    return AEE_EBADPARM;
  }
  /** Property 5: a band's rows must never span more than one extra block once
   * masking restricts them. Rejecting it here is the
   * only place it can be caught violating it misbehaves silently until
   * block skipping lands, one task later. */
  if (M_band == 0u || M_band > T) {
    return AEE_EBADPARM;
  }

  memset(ctx, 0, sizeof(*ctx));
  ctx->vtcm_base = vtcm_base;
  ctx->vtcm_size = vtcm_size;
  ctx->config_off = config_off;
  ctx->nch = nch;
  ctx->gqa = gqa;
  ctx->head_dim = head_dim;
  ctx->max_kv = max_kv;
  ctx->T = T;
  ctx->M_band = M_band;
  ctx->max_blocks = (max_kv + T - 1u) / T;
  ctx->kv_len = 0u;

  if (ctx->max_blocks > HEXKL_ATTN_MAX_BLOCKS) {
    return AEE_EBADPARM;
  }
  if (hexkl_attn_ops_init(&ctx->ops_k, w_k, table_k) != 0 ||
      hexkl_attn_ops_init(&ctx->ops_v, w_v, table_v) != 0) {
    return AEE_EBADPARM;
  }

  n_slots = nch * ctx->max_blocks;
  /** Kt is [head_dim][T], V is [T][head_dim] same element count, so one
   * scratch buffer serves both. The scale/colsum vectors differ in length
   * (T vs head_dim); size for the larger. */
  biggest = (T > head_dim) ? T : head_dim;

  ctx->k_shadow =
    (uint16_t *)malloc((size_t)max_kv * row_stride(ctx) * sizeof(uint16_t));
  ctx->v_shadow =
    (uint16_t *)malloc((size_t)max_kv * row_stride(ctx) * sizeof(uint16_t));
  ctx->h_kt = (uint32_t *)malloc((size_t)n_slots * sizeof(uint32_t));
  ctx->h_v = (uint32_t *)malloc((size_t)n_slots * sizeof(uint32_t));
  ctx->q_rm = (int8_t *)malloc((size_t)T * head_dim);
  ctx->q_scale = (float *)malloc((size_t)biggest * sizeof(float));
  ctx->q_colsum = (int32_t *)malloc((size_t)biggest * sizeof(int32_t));
  ctx->q_bias = (float *)calloc((size_t)biggest, sizeof(float));

  ctx->s_band =
    (float *)malloc((size_t)ctx->max_blocks * M_band * T * sizeof(float));
  ctx->o_band = (float *)malloc((size_t)M_band * head_dim * sizeof(float));
  ctx->l_row = (float *)malloc((size_t)M_band * sizeof(float));
  ctx->sink_row = (float *)malloc((size_t)M_band * sizeof(float));
  ctx->q_gather = (float *)malloc((size_t)M_band * head_dim * sizeof(float));
  ctx->begin = (uint32_t *)malloc((size_t)M_band * sizeof(uint32_t));
  ctx->end = (uint32_t *)malloc((size_t)M_band * sizeof(uint32_t));
  ctx->seg = (float **)malloc((size_t)ctx->max_blocks * sizeof(float *));

  {
    const uint32_t m_pad64 = ((M_band + 63u) / 64u) * 64u;
    ctx->bm = (float *)malloc((size_t)ctx->max_blocks * M_band * sizeof(float));
    ctx->p_scale = (float *)malloc((size_t)m_pad64 * sizeof(float));
    /** Permanently zero: p >= 0 pins rmin at 0, so the scan's zp is always
     * round(-0/s) = 0 including its synthetic (1, 0) for padding rows. */
    ctx->p_zp = (int32_t *)calloc((size_t)m_pad64, sizeof(int32_t));
  }

  if (!ctx->k_shadow || !ctx->v_shadow || !ctx->h_kt || !ctx->h_v ||
      !ctx->q_rm || !ctx->q_scale || !ctx->q_colsum || !ctx->q_bias ||
      !ctx->s_band || !ctx->o_band || !ctx->l_row || !ctx->sink_row ||
      !ctx->q_gather || !ctx->begin || !ctx->end || !ctx->seg || !ctx->bm ||
      !ctx->p_scale || !ctx->p_zp) {
    hexkl_attn_u8_ctx_fini(ctx);
    return AEE_ENOMEMORY;
  }

  /** Zero-fill the shadows: a tail block's unused rows are quantized along with
   * the real ones, and garbage there is a silent wrong answer that the mask
   * alone will not save. */
  memset(ctx->k_shadow, 0, (size_t)max_kv * row_stride(ctx) * sizeof(uint16_t));
  memset(ctx->v_shadow, 0, (size_t)max_kv * row_stride(ctx) * sizeof(uint16_t));
  for (i = 0; i < n_slots; ++i) {
    ctx->h_kt[i] = HEXKL_ATTN_NO_HANDLE;
    ctx->h_v[i] = HEXKL_ATTN_NO_HANDLE;
  }
  return AEE_SUCCESS;
}

void hexkl_attn_u8_ctx_fini(hexkl_attn_u8_ctx *ctx) {
  if (ctx == NULL) {
    return;
  }
  if (ctx->h_kt != NULL || ctx->h_v != NULL) {
    const uint32_t n_slots = ctx->nch * ctx->max_blocks;
    uint32_t i;
    for (i = 0; i < n_slots; ++i) {
      if (ctx->h_kt != NULL && ctx->h_kt[i] != HEXKL_ATTN_NO_HANDLE) {
        (void)ctx->ops_k.rel(ctx->ops_k.table, ctx->h_kt[i]);
      }
      if (ctx->h_v != NULL && ctx->h_v[i] != HEXKL_ATTN_NO_HANDLE) {
        (void)ctx->ops_v.rel(ctx->ops_v.table, ctx->h_v[i]);
      }
    }
  }
  free(ctx->k_shadow);
  free(ctx->v_shadow);
  free(ctx->h_kt);
  free(ctx->h_v);
  free(ctx->q_rm);
  free(ctx->q_scale);
  free(ctx->q_colsum);
  free(ctx->q_bias);
  free(ctx->s_band);
  free(ctx->o_band);
  free(ctx->l_row);
  free(ctx->sink_row);
  free(ctx->q_gather);
  free(ctx->begin);
  free(ctx->end);
  free(ctx->seg);
  free(ctx->bm);
  free(ctx->p_scale);
  free(ctx->p_zp);
  memset(ctx, 0, sizeof(*ctx));
}

/** @brief (Re-)registers block @a blk of head @a head, both operands. */
static int register_block(hexkl_attn_u8_ctx *ctx, uint32_t head, uint32_t blk) {
  const uint32_t slot = head * ctx->max_blocks + blk;
  const uint32_t base = blk * ctx->T;
  const uint32_t valid =
    (ctx->kv_len > base)
      ? ((ctx->kv_len - base < ctx->T) ? ctx->kv_len - base : ctx->T)
      : 0u;
  const size_t off = (size_t)base * row_stride(ctx);
  int rc;

  /** Release before re-registering: the slot's scales change when a new token
   * lands in it, and leaking the old handle would exhaust the registry after
   * T appends. */
  if (ctx->h_kt[slot] != HEXKL_ATTN_NO_HANDLE) {
    (void)ctx->ops_k.rel(ctx->ops_k.table, ctx->h_kt[slot]);
    ctx->h_kt[slot] = HEXKL_ATTN_NO_HANDLE;
  }
  if (ctx->h_v[slot] != HEXKL_ATTN_NO_HANDLE) {
    (void)ctx->ops_v.rel(ctx->ops_v.table, ctx->h_v[slot]);
    ctx->h_v[slot] = HEXKL_ATTN_NO_HANDLE;
  }
  if (valid == 0u) {
    return AEE_SUCCESS;
  }

  /* Kt block: logical [head_dim][T], N = kv position, K = head_dim. */
  hexkl_kvq_pack_kt_block(ctx->k_shadow + off, valid, ctx->T, ctx->head_dim,
                          ctx->nch, head, ctx->ops_k.width, ctx->q_rm,
                          ctx->q_scale, ctx->q_colsum);
  rc = ctx->ops_k.reg(ctx->ops_k.table, ctx->vtcm_base, ctx->vtcm_size,
                      ctx->head_dim, ctx->T, ctx->q_rm, ctx->q_scale,
                      ctx->q_colsum, ctx->q_bias, &ctx->h_kt[slot]);
  if (rc != AEE_SUCCESS) {
    return rc;
  }

  /* V block: logical [T][head_dim], N = head_dim column, K = T. */
  hexkl_kvq_pack_v_block(ctx->v_shadow + off, valid, ctx->T, ctx->head_dim,
                         ctx->nch, head, ctx->ops_v.width, ctx->q_rm,
                         ctx->q_scale, ctx->q_colsum);
  return ctx->ops_v.reg(ctx->ops_v.table, ctx->vtcm_base, ctx->vtcm_size,
                        ctx->T, ctx->head_dim, ctx->q_rm, ctx->q_scale,
                        ctx->q_colsum, ctx->q_bias, &ctx->h_v[slot]);
}

int hexkl_attn_u8_kv_append(hexkl_attn_u8_ctx *ctx, uint32_t kv_from,
                            uint32_t n_rows, const uint16_t *k_rows_f16,
                            const uint16_t *v_rows_f16) {
  uint32_t first_blk, last_blk, head, blk;
  size_t off, bytes;

  if (ctx == NULL || k_rows_f16 == NULL || v_rows_f16 == NULL) {
    return AEE_EBADPARM;
  }
  if (n_rows == 0u || kv_from + n_rows > ctx->max_kv) {
    return AEE_EBADPARM;
  }

  off = (size_t)kv_from * row_stride(ctx);
  bytes = (size_t)n_rows * row_stride(ctx) * sizeof(uint16_t);
  memcpy(ctx->k_shadow + off, k_rows_f16, bytes);
  memcpy(ctx->v_shadow + off, v_rows_f16, bytes);

  if (kv_from + n_rows > ctx->kv_len) {
    ctx->kv_len = kv_from + n_rows;
  }

  /** Only the blocks the new rows landed in change. Everything before
   * first_blk keeps the handles it already holds which is what makes the
   * append cost independent of kv_len. */
  first_blk = kv_from / ctx->T;
  last_blk = (ctx->kv_len - 1u) / ctx->T;
  for (head = 0; head < ctx->nch; ++head) {
    for (blk = first_blk; blk <= last_blk; ++blk) {
      const int rc = register_block(ctx, head, blk);
      if (rc != AEE_SUCCESS) {
        return rc;
      }
    }
  }
  return AEE_SUCCESS;
}

int hexkl_attn_u8_scores(hexkl_attn_u8_ctx *ctx, uint32_t head, uint32_t M,
                         const float *q_band, float *out_s) {
  uint32_t handles[HEXKL_ATTN_MAX_BLOCKS];
  uint32_t n_blocks, blk;

  if (ctx == NULL || q_band == NULL || out_s == NULL || head >= ctx->nch ||
      M == 0u) {
    return AEE_EBADPARM;
  }
  n_blocks = hexkl_attn_u8_n_blocks(ctx);
  if (n_blocks == 0u) {
    return AEE_EBADPARM;
  }

  for (blk = 0; blk < n_blocks; ++blk) {
    const uint32_t h = ctx->h_kt[head * ctx->max_blocks + blk];
    if (h == HEXKL_ATTN_NO_HANDLE) {
      return AEE_EBADSTATE;
    }
    handles[blk] = h;
  }

  /** ONE call. layer_run quantizes q_band once, shares it across every handle,
   * and prefetches block j+1's weight while j computes. out_s comes back
   * block-major, which is the layout PHASE B walks directly. */
  {
    const hexkl_mm_opts opts = {.pool = (hvx_worker_pool *)ctx->pool};
    return ctx->ops_k.run(ctx->ops_k.table, ctx->vtcm_base, ctx->vtcm_size,
                          ctx->config_off, M, ctx->head_dim, handles, n_blocks,
                          q_band, out_s, &opts);
  }
}

/** @brief One PHASE B job: the band, ready for a row split. */
typedef struct {
  float *const *seg;
  uint32_t n_seg, T, M;
  float scale;
  const uint32_t *begin;
  const uint32_t *end;
  const float *sink;
  float *l_row;
  float *bm; /**< [n_seg][M] stored maxima; each worker writes its own rows */
} attn_softmax_job;

/**
 * @brief hvx_worker_pool_func running rows [i*per, min((i+1)*per, M)).
 *
 * Rows are independent and each worker writes only its own l_row entries
 * (hvx_softmax_blocked_f32's contract), so the split is bitwise-invisible:
 * every row is computed by exactly the arithmetic the inline call would use. */
static void attn_softmax_worker(uint32_t n_threads, uint32_t i, void *arg) {
  const attn_softmax_job *j = (const attn_softmax_job *)arg;
  const uint32_t per = (j->M + n_threads - 1u) / n_threads;
  const uint32_t m0 = i * per;
  uint32_t m1 = m0 + per;

  if (m1 > j->M) {
    m1 = j->M;
  }
  if (m0 >= m1) {
    return;
  }
  hvx_softmax_blocked_f32(j->seg, j->n_seg, j->T, m0, m1, j->M, j->scale,
                          j->begin, j->end, j->sink, j->l_row, j->bm);
}

/** @brief Bands with fewer rows run PHASE B inline: a fork/join costs more
 * than a decode band's whole softmax (~0.8 us). */
#define ATTN_SOFTMAX_POOL_MIN_ROWS 16u

int hexkl_attn_u8_forward(hexkl_attn_u8_ctx *ctx, uint32_t kv_from,
                          uint32_t n_query, float scale, int is_causal,
                          uint32_t window, const float *sink, const float *q,
                          float *out, float *dbg_p, float *dbg_l,
                          uint32_t *stage_us) {
  uint32_t nHq, M, n_blocks, n, b0, m, j, d;
  uint64_t t0 = 0, t1 = 0, t_call0 = 0;
  int rc;

  if (ctx == NULL || q == NULL || out == NULL || n_query == 0u) {
    return AEE_EBADPARM;
  }
  if (kv_from + n_query != ctx->kv_len) {
    /** kv_len is kv_from + n_query by construction: the step rows must already
     * have been appended, so query row i sits at absolute position
     * kv_from + i. Catch the mismatch here rather than producing a plausible
     * wrong answer. */
    return AEE_EBADSTATE;
  }

  if (stage_us) {
    for (n = 0; n < (uint32_t)HEXKL_ATTN_N_STAGES; ++n) {
      stage_us[n] = 0u;
    }
  }
  hexkl_probe_reset(stage_us != NULL);
  TICK(t_call0);

  nHq = ctx->nch * ctx->gqa;
  M = n_query * ctx->gqa; /* GQA row fold, per kv_head */
  n_blocks = hexkl_attn_u8_n_blocks(ctx);

  for (n = 0; n < ctx->nch; ++n) {
    for (b0 = 0; b0 < M; b0 += ctx->M_band) {
      const uint32_t mb = (M - b0 < ctx->M_band) ? (M - b0) : ctx->M_band;

      /* ---- PHASE A: scores, streaming the K blocks ---------------------- */
      TICK(t0);
      for (m = 0; m < mb; ++m) {
        const uint32_t i = (b0 + m) / ctx->gqa;
        const uint32_t g = (b0 + m) % ctx->gqa;
        memcpy(ctx->q_gather + (size_t)m * ctx->head_dim,
               q + ((size_t)i * nHq + n * ctx->gqa + g) * ctx->head_dim,
               (size_t)ctx->head_dim * sizeof(float));
      }
      TICK(t1);
      ACCUM(HEXKL_ATTN_T_GATHER, t0, t1);

      TICK(t0);
      rc = hexkl_attn_u8_scores(ctx, n, mb, ctx->q_gather, ctx->s_band);
      TICK(t1);
      ACCUM(HEXKL_ATTN_T_QK, t0, t1);
      if (stage_us) {
        stage_us[HEXKL_ATTN_T_CALLS] += 1u;
      }
      if (rc != AEE_SUCCESS) {
        return rc;
      }

      /* ---- PHASE B: one masked softmax pass over the whole band --------- */
      for (m = 0; m < mb; ++m) {
        const uint32_t i = (b0 + m) / ctx->gqa;
        const uint32_t g = (b0 + m) % ctx->gqa;
        const uint32_t p = kv_from + i;
        const uint32_t e = is_causal ? (p + 1u) : ctx->kv_len;
        ctx->end[m] = e;
        ctx->begin[m] = (window != 0u && window < e) ? (e - window) : 0u;
        ctx->sink_row[m] = (sink != NULL) ? sink[n * ctx->gqa + g] : 0.0f;
      }
      for (j = 0; j < n_blocks; ++j) {
        ctx->seg[j] = ctx->s_band + (size_t)j * mb * ctx->T;
      }
      TICK(t0);
      if (ctx->pool != NULL && mb >= ATTN_SOFTMAX_POOL_MIN_ROWS) {
        attn_softmax_job job = {
          ctx->seg,   n_blocks,
          ctx->T,     mb,
          scale,      ctx->begin,
          ctx->end,   (sink != NULL) ? ctx->sink_row : NULL,
          ctx->l_row, ctx->bm};
        hvx_worker_pool_run(ctx->pool, attn_softmax_worker, &job, mb);
      } else {
        hvx_softmax_blocked_f32(
          ctx->seg, n_blocks, ctx->T, 0u, mb, mb, scale, ctx->begin, ctx->end,
          (sink != NULL) ? ctx->sink_row : NULL, ctx->l_row, ctx->bm);
      }
      TICK(t1);
      ACCUM(HEXKL_ATTN_T_SOFTMAX, t0, t1);

      /* ---- PHASE C: output, streaming the V blocks ---------------------- */
      memset(ctx->o_band, 0, (size_t)mb * ctx->head_dim * sizeof(float));
      for (j = 0; j < n_blocks; ++j) {
        const uint32_t hv = ctx->h_v[n * ctx->max_blocks + j];
        if (hv == HEXKL_ATTN_NO_HANDLE) {
          return AEE_EBADSTATE;
        }
        /** P is still quantized PER BLOCK (property 2) what changed is who
         * computes the params. PHASE B already took the running max of every
         * value it stored into this block row, so scale is bm/255 with rmin
         * pinned at 0 by p >= 0 exactly what layer_run's scan would
         * return (hvx_softmax_blocked_f32.h). bm == 0 is the scan's
         * degenerate all-zero branch, (1, 0), and padding rows get its
         * synthetic (1, 0) too. */
        {
          const uint32_t m_pad64 = ((mb + 63u) / 64u) * 64u;
          for (m = 0; m < mb; ++m) {
            const float bm = ctx->bm[(size_t)j * mb + m];
            ctx->p_scale[m] = (bm > 0.0f) ? (bm / 255.0f) : 1.0f;
          }
          for (m = mb; m < m_pad64; ++m) {
            ctx->p_scale[m] = 1.0f;
          }
        }

        /** The f32 accumulation across blocks (property 3) now happens INSIDE
         * the dequant: each block still carries its own constants, and each
         * element still receives exactly one add per block only the staged
         * o_part buffer and its add loop are gone. */
        TICK(t0);
        {
          const hexkl_mm_opts opts = {.pool = (hvx_worker_pool *)ctx->pool,
                                      .act_scale = ctx->p_scale,
                                      .act_zp = ctx->p_zp,
                                      .accumulate = 1};
          rc = ctx->ops_v.run(ctx->ops_v.table, ctx->vtcm_base, ctx->vtcm_size,
                              ctx->config_off, mb, ctx->T, &hv, 1u, ctx->seg[j],
                              ctx->o_band, &opts);
        }
        TICK(t1);
        ACCUM(HEXKL_ATTN_T_PV, t0, t1);
        if (stage_us) {
          stage_us[HEXKL_ATTN_T_CALLS] += 1u;
        }
        if (rc != AEE_SUCCESS) {
          return rc;
        }
      }

      /** The 1/l normalization happens ONCE, here, after every block has been
       * accumulated (property 4) not as a third softmax pass. */
      TICK(t0);
      for (m = 0; m < mb; ++m) {
        const uint32_t i = (b0 + m) / ctx->gqa;
        const uint32_t g = (b0 + m) % ctx->gqa;
        const float inv_l =
          (ctx->l_row[m] > 0.0f) ? (1.0f / ctx->l_row[m]) : 0.0f;
        float *dst = out + ((size_t)i * nHq + n * ctx->gqa + g) * ctx->head_dim;
        /** head_dim % 32 == 0 is enforced at init, so whole vectors and no
         * tail. One multiply per element, same op the scalar loop did; the
         * S band gate holds bit-identical through these same Vsf intrinsics,
         * so the vectorization is not a numerics change. */
        const HVX_UVector *vsrc =
          (const HVX_UVector *)(ctx->o_band + (size_t)m * ctx->head_dim);
        HVX_UVector *vdst = (HVX_UVector *)dst;
        const HVX_Vector vinv = hvx_splat_sf(inv_l);
        for (d = 0; d < ctx->head_dim / 32u; ++d) {
          vdst[d] = Q6_Vsf_vmpy_VsfVsf(vsrc[d], vinv);
        }
      }

      TICK(t1);
      ACCUM(HEXKL_ATTN_T_GATHER, t0, t1);

      if (dbg_p != NULL) {
        memcpy(dbg_p, ctx->s_band,
               (size_t)n_blocks * mb * ctx->T * sizeof(float));
      }
      if (dbg_l != NULL) {
        memcpy(dbg_l, ctx->l_row, (size_t)mb * sizeof(float));
      }
    }
  }
  TICK(t1);
  ACCUM(HEXKL_ATTN_T_TOTAL, t_call0, t1);
  if (stage_us) {
    /** The probes accumulated inside every layer_run this call made; hand
     * them back beside the stage totals they subdivide. */
    stage_us[HEXKL_ATTN_T_ACC_READ] =
      (uint32_t)hexkl_probe_us[HEXKL_PROBE_ACC_READ];
    stage_us[HEXKL_ATTN_T_ACC_COPY] =
      (uint32_t)hexkl_probe_us[HEXKL_PROBE_ACC_COPY];
    stage_us[HEXKL_ATTN_T_DEQUANT] =
      (uint32_t)hexkl_probe_us[HEXKL_PROBE_DEQUANT];
    stage_us[HEXKL_ATTN_T_QUANT] = (uint32_t)hexkl_probe_us[HEXKL_PROBE_QUANT];
    stage_us[HEXKL_ATTN_T_DRAIN] = (uint32_t)hexkl_probe_us[HEXKL_PROBE_DRAIN];
    stage_us[HEXKL_ATTN_T_ACC_STRIDE] =
      (uint32_t)hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE];
  }
  return AEE_SUCCESS;
}
