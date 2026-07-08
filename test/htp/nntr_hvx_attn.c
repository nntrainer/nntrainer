// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   nntr_hvx_attn.c
 * @date   07 Aug 2026
 * @brief  DSP-side entries for the attention KV registry and the S band
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * attn_scores_debug is scaffolding: it exists so PHASE A can be gated against
 * mha_htp_host_forward before the fused forward that will call it in-process
 * exists, and it goes away with the rest of the debug surface before the PR
 * (MHA_HTP_PLAN.md §9.5).
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>
#include <string.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "hexkl_attn_u8.h"

/** @brief Attention layers live at once. One per mha_core layer; a 28-layer
 *         model would register 28, and this leaves room without making the
 *         table itself large. */
#define NNTR_HVX_MAX_ATTN 32u

static hexkl_attn_u8_ctx g_attn[NNTR_HVX_MAX_ATTN];
static int g_attn_in_use[NNTR_HVX_MAX_ATTN];

static hexkl_attn_u8_ctx *attn_lookup(uint32 h) {
  if (h >= NNTR_HVX_MAX_ATTN || !g_attn_in_use[h]) {
    return NULL;
  }
  return &g_attn[h];
}

int nntr_hvx_attn_register(remote_handle64 handle, uint32 nch, uint32 gqa,
                           uint32 head_dim, uint32 max_kv, uint32 T,
                           uint32 M_band, uint32 w_k, uint32 w_v, uint32 *h) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  uint32 i;
  int rc;

  if (!s || !h) {
    return AEE_EBADPARM;
  }
  if ((w_k != HEXKL_W_I4 && w_k != HEXKL_W_I8) ||
      (w_v != HEXKL_W_I4 && w_v != HEXKL_W_I8)) {
    FARF(ERROR, "attn_register: width must be 4 or 8 (w_k=%u w_v=%u)",
         (unsigned)w_k, (unsigned)w_v);
    return AEE_EBADPARM;
  }

  for (i = 0; i < NNTR_HVX_MAX_ATTN; ++i) {
    if (!g_attn_in_use[i]) {
      break;
    }
  }
  if (i == NNTR_HVX_MAX_ATTN) {
    return AEE_ENOMEMORY;
  }

  /** Kt and V draw from the session's registry for their own width, which is
   * what lets the two operands differ at no structural cost (§0.2). */
  rc = hexkl_attn_u8_ctx_init(
    &g_attn[i], s->vtcm_base, s->vtcm_size, s->config_off, nch, gqa, head_dim,
    max_kv, T, M_band, (hexkl_w_width)w_k, (hexkl_w_width)w_v,
    (w_k == HEXKL_W_I4) ? (void *)&s->weights_u8i4 : (void *)&s->weights_u8i8,
    (w_v == HEXKL_W_I4) ? (void *)&s->weights_u8i4 : (void *)&s->weights_u8i8);
  if (rc != AEE_SUCCESS) {
    return rc;
  }

  /** Not owned by the ctx: the session's pool outlives every attention
   * handle, and close() destroys it after everything is released. */
  g_attn[i].pool = s->quant_pool;

  g_attn_in_use[i] = 1;
  *h = i;
  return AEE_SUCCESS;
}

int nntr_hvx_attn_release(remote_handle64 handle, uint32 h) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  hexkl_attn_u8_ctx *ctx;

  if (!s) {
    return AEE_EBADPARM;
  }
  ctx = attn_lookup(h);
  if (!ctx) {
    return AEE_EBADPARM;
  }

  hexkl_attn_u8_ctx_fini(ctx);
  g_attn_in_use[h] = 0;
  return AEE_SUCCESS;
}

int nntr_hvx_attn_kv_append(remote_handle64 handle, uint32 h, uint32 kv_from,
                            uint32 n_rows, const uint16 *k_rows, int k_rowsLen,
                            const uint16 *v_rows, int v_rowsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  hexkl_attn_u8_ctx *ctx;
  uint32 want;

  if (!s) {
    return AEE_EBADPARM;
  }
  ctx = attn_lookup(h);
  if (!ctx) {
    return AEE_EBADPARM;
  }

  want = n_rows * ctx->nch * ctx->head_dim;
  if ((uint32)k_rowsLen != want || (uint32)v_rowsLen != want) {
    FARF(ERROR, "attn_kv_append: bad row length (%d/%d, expected %u)",
         k_rowsLen, v_rowsLen, (unsigned)want);
    return AEE_EBADPARM;
  }

  return hexkl_attn_u8_kv_append(ctx, kv_from, n_rows, k_rows, v_rows);
}

int nntr_hvx_attn_scores_debug(remote_handle64 handle, uint32 h, uint32 head,
                               uint32 M, const float *q_band, int q_bandLen,
                               float *out_s, int out_sLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  hexkl_attn_u8_ctx *ctx;
  uint32 n_blocks;

  if (!s) {
    return AEE_EBADPARM;
  }
  ctx = attn_lookup(h);
  if (!ctx) {
    return AEE_EBADPARM;
  }

  n_blocks = hexkl_attn_u8_n_blocks(ctx);
  if ((uint32)q_bandLen != M * ctx->head_dim ||
      (uint32)out_sLen != n_blocks * M * ctx->T) {
    FARF(ERROR, "attn_scores_debug: bad lengths (q=%d want %u, s=%d want %u)",
         q_bandLen, (unsigned)(M * ctx->head_dim), out_sLen,
         (unsigned)(n_blocks * M * ctx->T));
    return AEE_EBADPARM;
  }

  return hexkl_attn_u8_scores(ctx, head, M, q_band, out_s);
}

int nntr_hvx_attn_forward(remote_handle64 handle, uint32 h, uint32 kv_from,
                          uint32 n_query, float scale, uint32 is_causal,
                          uint32 window, const float *sink, int sinkLen,
                          const float *q, int qLen, float *out, int outLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  hexkl_attn_u8_ctx *ctx;
  uint32 nHq, want;

  if (!s) {
    return AEE_EBADPARM;
  }
  ctx = attn_lookup(h);
  if (!ctx) {
    return AEE_EBADPARM;
  }

  nHq = ctx->nch * ctx->gqa;
  want = n_query * nHq * ctx->head_dim;
  if ((uint32)qLen != want || (uint32)outLen != want) {
    FARF(ERROR, "attn_forward: bad q/out length (%d/%d, expected %u)", qLen,
         outLen, (unsigned)want);
    return AEE_EBADPARM;
  }
  if (sinkLen != 0 && (uint32)sinkLen != nHq) {
    FARF(ERROR, "attn_forward: sink must be empty or nHq=%u", (unsigned)nHq);
    return AEE_EBADPARM;
  }

  return hexkl_attn_u8_forward(ctx, kv_from, n_query, scale, (int)is_causal,
                               window, sinkLen ? sink : NULL, q, out, NULL,
                               NULL, NULL);
}

int nntr_hvx_attn_forward_timed(remote_handle64 handle, uint32 h,
                                uint32 kv_from, uint32 n_query, float scale,
                                uint32 is_causal, uint32 window,
                                const float *sink, int sinkLen, const float *q,
                                int qLen, float *attn_out, int attn_outLen,
                                uint32 *stage_us, int stage_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  hexkl_attn_u8_ctx *ctx;
  uint32 nHq, want;

  if (!s) {
    return AEE_EBADPARM;
  }
  ctx = attn_lookup(h);
  if (!ctx) {
    return AEE_EBADPARM;
  }
  if (stage_usLen != HEXKL_ATTN_N_STAGES) {
    FARF(ERROR, "attn_forward_timed: stage_us must have %d entries, got %d",
         (int)HEXKL_ATTN_N_STAGES, stage_usLen);
    return AEE_EBADPARM;
  }

  nHq = ctx->nch * ctx->gqa;
  want = n_query * nHq * ctx->head_dim;
  if ((uint32)qLen != want || (uint32)attn_outLen != want) {
    return AEE_EBADPARM;
  }
  if (sinkLen != 0 && (uint32)sinkLen != nHq) {
    return AEE_EBADPARM;
  }

  return hexkl_attn_u8_forward(ctx, kv_from, n_query, scale, (int)is_causal,
                               window, sinkLen ? sink : NULL, q, attn_out, NULL,
                               NULL, stage_us);
}
