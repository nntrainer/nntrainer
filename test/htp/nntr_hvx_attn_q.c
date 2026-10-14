// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   nntr_hvx_attn_q.c
 * @date   23 Sep 2026
 * @brief  FastRPC entries for the quantized (A8W8 / A8W4) KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <remote.h>

#include "hexkl_acc_tile.h"
#include "hexkl_attn_q.h"
#include "hexkl_attn_q2.h"
#include "hexkl_cvt.h"
#include "hexkl_kv_q.h"
#include "hexkl_micro.h"
#include "hvx_attn_decode_q.h"
#include "hvx_convert.h"
#include "hvx_softmax_q.h"
#include "hvx_worker_pool.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

/** @brief stats_us slots of attn_q_prefill. */
enum {
  QSTAT_QPREP = 0,
  QSTAT_DMA,
  QSTAT_QK,
  QSTAT_DEQUANT,
  QSTAT_SOFTMAX,
  QSTAT_PQUANT,
  QSTAT_PV,
  QSTAT_OUPD,
  QSTAT_STORE,
  QSTAT_N_BLOCKS,
  QSTAT_US_TOTAL,
  QSTAT_KCYCLES, /**< processor kilocycles of the call */
  QSTAT_COUNT
};

/** @brief One uint8 activation tile (64x32), also its alignment. */
#define ACT_TILE_BYTES HEXKL_HMX_ACTIVATION_ALIGNMENT
/** @brief Largest head_dim / 32. */
#define MAX_DT 16u

/**
 * @brief VTCM offsets the probe uses: activation tiles, weight tiles copied
 *        from the registry, one accumulator readout tile. All far below the
 *        session's config region.
 */
enum {
  OFF_ACT_S = 0, /**< MAX_DT u8 tiles */
  OFF_ACT_P = OFF_ACT_S + MAX_DT * ACT_TILE_BYTES,
  OFF_W_KT = OFF_ACT_P + ACT_TILE_BYTES, /**< MAX_DT x 1024 B */
  OFF_W_V = OFF_W_KT + MAX_DT * 1024u,
  OFF_ACC = OFF_W_V + MAX_DT * 1024u,              /**< 8 KiB, 2048-aligned */
  OFF_CVT_BLK0 = OFF_ACC + HEXKL_ATTN_Q_ACC_BYTES, /**< two 1 KiB blocks */
  OFF_CVT_BLK1 = OFF_CVT_BLK0 + HEXKL_CVT_BLOCK_BYTES,
  OFF_PLANE0 = OFF_CVT_BLK0 + ACT_TILE_BYTES, /**< 2 KiB each */
  OFF_PLANE1 = OFF_PLANE0 + HEXKL_CVT_PLANE_BYTES,
  OFF_S16 = OFF_PLANE1 + HEXKL_CVT_PLANE_BYTES, /**< 4 KiB int16 tile */
  /** The softmax probe: up to 128 score tiles, their P' tiles, corr, row
   * sums and the routine's scratch, all well below the config region. */
  OFF_SM_S = 0x20000,                    /**< 128 x 4 KiB */
  OFF_SM_P = OFF_SM_S + 128u * 4096u,    /**< 128 x 2 KiB */
  OFF_SM_CORR = OFF_SM_P + 128u * 2048u, /**< 129 x 64 B */
  OFF_SM_RS = OFF_SM_CORR + 0x4000,      /**< 2 KiB */
  OFF_SM_SCR = OFF_SM_RS + 0x1000,       /**< HVX_SOFTMAX_Q_SCRATCH_BYTES */
  OFF_SM_PH = OFF_SM_SCR + 0x8000,       /**< 128 x 2 KiB, the high bytes */
};

int nntr_hvx_kv_register_q(remote_handle64 handle, uint32 kind, uint32 max_rows,
                           uint32 n_head_kv, uint32 head_dim,
                           uint32 *kv_handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !kv_handle) {
    return AEE_EBADPARM;
  }
  return hexkl_kv_q_register(&s->kv_q, (hexkl_kv_q_kind)kind, max_rows,
                             n_head_kv, head_dim, kv_handle);
}

int nntr_hvx_kv_set_fixed_scales_q(remote_handle64 handle, uint32 kv_handle,
                                   const float *s_k, int s_kLen,
                                   const float *s_v, int s_vLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv || (uint32_t)s_kLen != kv->n_head_kv ||
      (uint32_t)s_vLen != kv->n_head_kv * kv->head_dim) {
    FARF(ERROR, "kv_set_fixed_scales_q: bad lengths");
    return AEE_EBADPARM;
  }
  return hexkl_kv_q_set_fixed_scales(&s->kv_q, kv_handle, s_k, s_v);
}

int nntr_hvx_kv_release_q(remote_handle64 handle, uint32 kv_handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  return hexkl_kv_q_release(&s->kv_q, kv_handle);
}

int nntr_hvx_kv_append_q(remote_handle64 handle, uint32 kv_handle, uint32 row0,
                         const uint16 *k_rows, int k_rowsLen,
                         const uint16 *v_rows, int v_rowsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv) {
    return AEE_EBADPARM;
  }
  const uint32_t stride = kv->n_head_kv * kv->head_dim;
  if (k_rowsLen != v_rowsLen || k_rowsLen <= 0 ||
      ((uint32_t)k_rowsLen % stride) != 0) {
    FARF(ERROR, "kv_append_q: bad lengths");
    return AEE_EBADPARM;
  }
  return hexkl_kv_q_append(&s->kv_q, kv_handle, row0,
                           (uint32_t)k_rowsLen / stride, k_rows, v_rows,
                           s->vtcm_base, NULL);
}

int nntr_hvx_kv_dump_q(remote_handle64 handle, uint32 kv_handle, uint32 row0,
                       uint32 n_rows, int8 *k_q, int k_qLen, int8 *v_q,
                       int v_qLen, float *s_k, int s_kLen, int32 *colsum_k,
                       int colsum_kLen, float *s_v, int s_vLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv) {
    return AEE_EBADPARM;
  }
  const uint64_t values = (uint64_t)n_rows * kv->n_head_kv * kv->head_dim;
  const uint64_t heads = (uint64_t)n_rows * kv->n_head_kv;
  if ((uint64_t)k_qLen != values || (uint64_t)v_qLen != values ||
      (uint64_t)s_kLen != heads || (uint64_t)colsum_kLen != heads ||
      (uint64_t)s_vLen != heads) {
    FARF(ERROR, "kv_dump_q: bad lengths");
    return AEE_EBADPARM;
  }
  return hexkl_kv_q_dump(kv, row0, n_rows, k_q, v_q, s_k, colsum_k, s_v);
}

int nntr_hvx_probe_acc_i32_layout(remote_handle64 handle, uint32 *layout,
                                  int layoutLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || layoutLen < 3) {
    return AEE_EBADPARM;
  }
  const hexkl_acc_layout *L = hexkl_acc_layout_get(s->vtcm_base, OFF_ACC);
  layout[0] = (uint32)L->usable;
  layout[1] = L->base;
  layout[2] = L->row_stride;
  return AEE_SUCCESS;
}

/** @brief The 64x32 int32 accumulator at OFF_ACC -> row-major @a out with
 *         @a out_stride elements per row, through the probed layout or the
 *         vendor copy when the probe found nothing usable. */
static int read_acc_tile(nntr_hvx_session *s, int32_t *out,
                         uint32_t out_stride) {
  const hexkl_acc_layout *L = hexkl_acc_layout_get(s->vtcm_base, OFF_ACC);
  int rc = hexkl_micro_hmx_acc_read_int32(s->vtcm_base, s->config_off, OFF_ACC);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  if (L->usable) {
    const int32_t *tile = (const int32_t *)(s->vtcm_base + OFF_ACC);
    for (uint32_t r = 0; r < HEXKL_ACC_TILE_ROWS; ++r) {
      for (uint32_t c = 0; c < HEXKL_ACC_TILE_COLS; ++c) {
        out[(size_t)r * out_stride + c] = hexkl_acc_tile_at(L, tile, r, c);
      }
    }
    return AEE_SUCCESS;
  }
  // One tile into a 64 x out_stride matrix at tile (0, 0).
  return hexkl_micro_hmx_copy_32b_to_submatrix(
    s->vtcm_base, OFF_ACC, out, 0u, 0u, HEXKL_ACC_TILE_ROWS, out_stride);
}

int nntr_hvx_probe_kv_q_mm(remote_handle64 handle, uint32 kv_handle, uint32 n,
                           uint32 c, const uint8 *act_s, int act_sLen,
                           const uint8 *act_p, int act_pLen, int32 *s_i32,
                           int s_i32Len, int32 *o_i32, int o_i32Len) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv || n >= kv->n_head_kv || c >= kv->n_col_tiles) {
    return AEE_EBADPARM;
  }
  const uint32_t hd = kv->head_dim;
  const uint32_t dt = kv->n_dot_tiles;
  if ((uint32_t)act_sLen != 64u * hd || (uint32_t)act_pLen != 64u * 32u ||
      (uint32_t)s_i32Len != 64u * 32u || (uint32_t)o_i32Len != 64u * hd) {
    FARF(ERROR, "probe_kv_q_mm: bad lengths");
    return AEE_EBADPARM;
  }
  uint8_t *vb = s->vtcm_base;

  // Activation tiles: flat row-major 64x32 u8, tile d = columns 32d..32d+31.
  for (uint32_t d = 0; d < dt; ++d) {
    uint8_t *tile = vb + OFF_ACT_S + d * ACT_TILE_BYTES;
    for (uint32_t r = 0; r < 64u; ++r) {
      memcpy(tile + r * 32u, act_s + (size_t)r * hd + 32u * d, 32u);
    }
  }
  memcpy(vb + OFF_ACT_P, act_p, 64u * 32u);
  // Weight tiles straight from the registry.
  for (uint32_t d = 0; d < dt; ++d) {
    memcpy(vb + OFF_W_KT + d * 1024u, kv->kt + hexkl_kv_q_tile_off(kv, n, c, d),
           kv->tile_bytes);
    memcpy(vb + OFF_W_V + d * 1024u, kv->v + hexkl_kv_q_tile_off(kv, n, c, d),
           kv->tile_bytes);
  }

  // S = act_s . K^T: reduction over the dot tiles into one accumulator.
  hexkl_micro_hmx_acc_clear_int32();
  for (uint32_t d = 0; d < dt; ++d) {
    int rc = kv->kind == HEXKL_KV_Q4
               ? hexkl_micro_hmx_mm_u8i4(vb, OFF_ACT_S + d * ACT_TILE_BYTES,
                                         OFF_W_KT + d * 1024u)
               : hexkl_micro_hmx_mm_u8i8(vb, OFF_ACT_S + d * ACT_TILE_BYTES,
                                         OFF_W_KT + d * 1024u);
    if (rc != AEE_SUCCESS) {
      return rc;
    }
  }
  int rc = read_acc_tile(s, s_i32, 32u);
  if (rc != AEE_SUCCESS) {
    return rc;
  }

  // O = act_p . V: one accumulator per output dot tile.
  for (uint32_t d = 0; d < dt; ++d) {
    hexkl_micro_hmx_acc_clear_int32();
    rc = kv->kind == HEXKL_KV_Q4
           ? hexkl_micro_hmx_mm_u8i4(vb, OFF_ACT_P, OFF_W_V + d * 1024u)
           : hexkl_micro_hmx_mm_u8i8(vb, OFF_ACT_P, OFF_W_V + d * 1024u);
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    rc = read_acc_tile(s, o_i32 + 32u * d, hd);
    if (rc != AEE_SUCCESS) {
      return rc;
    }
  }
  return AEE_SUCCESS;
}

int nntr_hvx_probe_cvt(remote_handle64 handle, uint32 kv_handle, uint32 n,
                       uint32 c, const uint8 *act_s, int act_sLen,
                       uint32 scale0_hf, uint32 scale1_hf, uint32 bias_q,
                       uint32 n_rep, uint8 *plane0, int plane0Len,
                       uint8 *plane1, int plane1Len, int16 *s16, int s16Len,
                       int32 *acc_i32, int acc_i32Len, uint32 *stats,
                       int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv || n >= kv->n_head_kv || c >= kv->n_col_tiles ||
      kv->kind != HEXKL_KV_Q8 || n_rep == 0) {
    return AEE_EBADPARM;
  }
  const uint32_t hd = kv->head_dim;
  const uint32_t dt = kv->n_dot_tiles;
  if ((uint32_t)act_sLen != 64u * hd ||
      (uint32_t)plane0Len != HEXKL_CVT_PLANE_BYTES ||
      (uint32_t)plane1Len != HEXKL_CVT_PLANE_BYTES ||
      (uint32_t)s16Len != 64u * 32u || (uint32_t)acc_i32Len != 64u * 32u ||
      statsLen < 2) {
    FARF(ERROR, "probe_cvt: bad lengths");
    return AEE_EBADPARM;
  }
  uint8_t *vb = s->vtcm_base;

  for (uint32_t d = 0; d < dt; ++d) {
    uint8_t *tile = vb + OFF_ACT_S + d * ACT_TILE_BYTES;
    for (uint32_t r = 0; r < 64u; ++r) {
      memcpy(tile + r * 32u, act_s + (size_t)r * hd + 32u * d, 32u);
    }
    memcpy(vb + OFF_W_KT + d * 1024u, kv->kt + hexkl_kv_q_tile_off(kv, n, c, d),
           kv->tile_bytes);
  }
  uint16_t sc[32], bq[32];
  for (uint32_t i = 0; i < 32u; ++i) {
    sc[i] = (uint16_t)scale0_hf;
    bq[i] = (uint16_t)bias_q;
  }
  hexkl_cvt_block_set(vb + OFF_CVT_BLK0, sc, bq);
  for (uint32_t i = 0; i < 32u; ++i) {
    sc[i] = (uint16_t)scale1_hf;
  }
  hexkl_cvt_block_set(vb + OFF_CVT_BLK1, sc, bq);

  hexkl_micro_hmx_acc_clear_int32();
  for (uint32_t d = 0; d < dt; ++d) {
    int rc = hexkl_micro_hmx_mm_u8i8(vb, OFF_ACT_S + d * ACT_TILE_BYTES,
                                     OFF_W_KT + d * 1024u);
    if (rc != AEE_SUCCESS) {
      return rc;
    }
  }
  // The same pass n_rep times: the accumulator is retained, so every pass
  // writes the same bytes and the loop measures the pass alone.
  const uint64_t c0 = HAP_perf_get_pcycles();
  for (uint32_t i = 0; i < n_rep; ++i) {
    hexkl_cvt_issue(vb + OFF_CVT_BLK0, vb + OFF_PLANE0);
  }
  const uint64_t c1 = HAP_perf_get_pcycles();
  // The same again, but each pass is followed by a vector load of what it
  // wrote, so the loop runs at the rate the data actually arrives.
  HVX_Vector sink = Q6_V_vzero();
  const uint64_t c4 = HAP_perf_get_pcycles();
  for (uint32_t i = 0; i < n_rep; ++i) {
    hexkl_cvt_issue(vb + OFF_CVT_BLK0, vb + OFF_PLANE0);
    sink = Q6_V_vor_VV(sink, *(const HVX_Vector *)(vb + OFF_PLANE0));
  }
  const uint64_t c5 = HAP_perf_get_pcycles();
  *(HVX_Vector *)(vb + OFF_PLANE1) = sink;
  hexkl_cvt_issue(vb + OFF_CVT_BLK1, vb + OFF_PLANE1);
  memcpy(plane0, vb + OFF_PLANE0, HEXKL_CVT_PLANE_BYTES);
  memcpy(plane1, vb + OFF_PLANE1, HEXKL_CVT_PLANE_BYTES);
  hexkl_cvt_zip_i16(vb + OFF_PLANE0, vb + OFF_PLANE1,
                    (int16_t *)(vb + OFF_S16));
  memcpy(s16, vb + OFF_S16, 2u * HEXKL_CVT_PLANE_BYTES);

  const uint64_t c2 = HAP_perf_get_pcycles();
  int rc = read_acc_tile(s, acc_i32, 32u);
  const uint64_t c3 = HAP_perf_get_pcycles();
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  stats[0] = (uint32)((c1 - c0) / n_rep);
  stats[1] = (uint32)(c3 - c2);
  if (statsLen > 2) {
    stats[2] = (uint32)((c5 - c4) / n_rep);
  }
  return AEE_SUCCESS;
}

int nntr_hvx_attn_q2_prefill(remote_handle64 handle, uint32 kv_handle,
                             uint32 n_q, uint32 cache_from, uint32 cache_to,
                             uint32 n_head_q, uint32 window,
                             const uint16 *q_u16, int q_u16Len,
                             const float *q_enc, int q_encLen,
                             const float *out_enc, int out_encLen,
                             uint16 *out_u16, int out_u16Len, uint32 *stats_us,
                             int stats_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv) {
    return AEE_EBADPARM;
  }
  const uint64_t need = (uint64_t)n_q * n_head_q * kv->head_dim;
  if ((uint64_t)q_u16Len != need || (uint64_t)out_u16Len != need ||
      stats_usLen < 9 || (uint32)q_encLen != 2u * n_head_q ||
      (uint32)out_encLen != 2u * n_head_q) {
    FARF(ERROR, "attn_q2_prefill: bad lengths");
    return AEE_EBADPARM;
  }
  hexkl_attn_f16_shape shape = {n_q,           cache_from,   cache_to, n_head_q,
                                kv->n_head_kv, kv->head_dim, window,   0.0f};
  hexkl_attn_q2_io io;
  memset(&io, 0, sizeof(io));
  io.q = q_u16;
  io.q_stride = n_head_q * kv->head_dim;
  io.q_enc = q_enc;
  io.out = out_u16;
  io.out_stride = n_head_q * kv->head_dim;
  io.out_enc = out_enc;
  io.kv = kv;
  hexkl_attn_q2_stats st;
  const int rc = hexkl_attn_q2_prefill(s->vtcm_base, s->config_off, &shape, &io,
                                       s->quant_pool, &st);
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "attn_q2_prefill: kernel failed: 0x%08x", rc);
    return rc;
  }
  stats_us[0] = (uint32)st.us_qprep;
  stats_us[1] = (uint32)st.us_dma;
  stats_us[2] = (uint32)st.us_qk;
  stats_us[3] = (uint32)st.us_softmax;
  stats_us[4] = (uint32)st.us_pv;
  stats_us[5] = (uint32)st.us_epi;
  stats_us[6] = (uint32)st.us_total;
  stats_us[7] = st.n_blocks;
  stats_us[8] = (uint32)(st.pcycles / 1000u);
  if (stats_usLen > 9) {
    stats_us[9] = (uint32)st.us_wait;
  }
  if (stats_usLen > 11) {
    stats_us[10] = (uint32)st.us_head;
    stats_us[11] = (uint32)st.us_submit;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_probe_hvx_rate(remote_handle64 handle, uint32 n, uint32 *stats,
                            int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || n == 0 || statsLen < 12) {
    return AEE_EBADPARM;
  }
  uint32_t out[12];
  hvx_softmax_q_rate(n, s->vtcm_base + OFF_SM_SCR, s->vtcm_base + OFF_SM_S,
                     out);
  memcpy(stats, out, sizeof(out));
  return AEE_SUCCESS;
}

/** Both q2 entries: Q either u16 (q_u16) or f32 (q_f32 + q_inv, the pool
   converts), the lengths in elements. */
static int attn_q2_step_impl(
  remote_handle64 handle, uint32 kv_handle, uint32 row0, const uint16 *k_rows,
  int k_rowsLen, const uint16 *v_rows, int v_rowsLen, uint32 n_q,
  uint32 cache_from, uint32 cache_to, uint32 n_head_q, uint32 window,
  const uint16 *q_u16, const float *q_f32, const float *q_inv, int qLen,
  const float *q_enc, int q_encLen, const float *out_enc, int out_encLen,
  uint16 *out_u16, int outLen, uint32 *stats_us, int stats_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv || cache_to > kv->max_rows || stats_usLen < 8 ||
      (uint32)q_encLen != 2u * n_head_q ||
      (uint32)out_encLen != 2u * n_head_q) {
    FARF(ERROR, "attn_q2_step: bad handle, cache_to, encodings or stats");
    return AEE_EBADPARM;
  }
  hexkl_kv_q_append_stats ast;
  memset(&ast, 0, sizeof(ast));
  const uint64_t t0 = HAP_perf_get_time_us();
  const uint32_t stride = kv->n_head_kv * kv->head_dim;
  if (k_rowsLen != v_rowsLen || k_rowsLen < 0 ||
      ((uint32_t)k_rowsLen % stride) != 0) {
    FARF(ERROR, "attn_q2_step: bad row lengths");
    return AEE_EBADPARM;
  }
  if (k_rowsLen > 0) {
    const int rc = hexkl_kv_q_append_pool(
      &s->kv_q, kv_handle, row0, (uint32_t)k_rowsLen / stride, k_rows, v_rows,
      s->vtcm_base, s->quant_pool, &ast);
    if (rc != AEE_SUCCESS) {
      FARF(ERROR, "attn_q2_step: append failed: 0x%08x", rc);
      return rc;
    }
  }
  const uint64_t t1 = HAP_perf_get_time_us();

  hexkl_attn_f16_shape shape = {n_q,           cache_from,   cache_to, n_head_q,
                                kv->n_head_kv, kv->head_dim, window,   0.0f};
  const uint64_t q_elems = (uint64_t)n_q * n_head_q * kv->head_dim;
  if ((uint64_t)qLen != q_elems || (uint64_t)outLen != q_elems) {
    FARF(ERROR, "attn_q2_step: bad lengths");
    return AEE_EBADPARM;
  }
  /** [P7] the f32 entry's Q rows go to the kernel as they are and are
     quantized inside its Q staging (hexkl_attn_q2_io.q_f32); the DDR u16
     pass this entry used to run (stats[2]) is gone. */
  hexkl_attn_q2_io io;
  memset(&io, 0, sizeof(io));
  io.q = q_u16;
  io.q_f32 = q_f32;
  io.q_inv = q_inv;
  io.q_stride = n_head_q * kv->head_dim;
  io.q_enc = q_enc;
  io.out = out_u16;
  io.out_stride = n_head_q * kv->head_dim;
  io.out_enc = out_enc;
  io.kv = kv;
  hexkl_attn_q2_stats st;
  const int res = hexkl_attn_q2_prefill(s->vtcm_base, s->config_off, &shape,
                                        &io, s->quant_pool, &st);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "attn_q2_step: kernel failed: 0x%08x", res);
    return res;
  }
  const uint64_t t2 = HAP_perf_get_time_us();
  stats_us[0] = (uint32)(t1 - t0);
  stats_us[1] = (uint32)(t2 - t1);
  stats_us[2] = 0u;
  stats_us[3] = (uint32)(t2 - t0);
  stats_us[4] = ast.quant_us;
  stats_us[5] = ast.stage_us;
  stats_us[6] = ast.bake_us;
  stats_us[7] = (uint32)((uint32_t)k_rowsLen / stride);
  if (stats_usLen >= 12) {
    stats_us[8] = (uint32)st.us_qk;
    stats_us[9] = (uint32)st.us_softmax;
    stats_us[10] = (uint32)st.us_pv;
    stats_us[11] = (uint32)st.us_wait;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_attn_q2_step(remote_handle64 handle, uint32 kv_handle, uint32 row0,
                          const uint16 *k_rows, int k_rowsLen,
                          const uint16 *v_rows, int v_rowsLen, uint32 n_q,
                          uint32 cache_from, uint32 cache_to, uint32 n_head_q,
                          uint32 window, const uint16 *q_u16, int q_u16Len,
                          const float *q_enc, int q_encLen,
                          const float *out_enc, int out_encLen, uint16 *out_u16,
                          int out_u16Len, uint32 *stats_us, int stats_usLen) {
  if (q_u16Len != out_u16Len) {
    return AEE_EBADPARM;
  }
  return attn_q2_step_impl(
    handle, kv_handle, row0, k_rows, k_rowsLen, v_rows, v_rowsLen, n_q,
    cache_from, cache_to, n_head_q, window, q_u16, NULL, NULL, q_u16Len, q_enc,
    q_encLen, out_enc, out_encLen, out_u16, out_u16Len, stats_us, stats_usLen);
}

int nntr_hvx_attn_q2_step_fq(
  remote_handle64 handle, uint32 kv_handle, uint32 row0, const uint16 *k_rows,
  int k_rowsLen, const uint16 *v_rows, int v_rowsLen, uint32 n_q,
  uint32 cache_from, uint32 cache_to, uint32 n_head_q, uint32 window,
  const float *q_f32, int q_f32Len, const float *q_inv, int q_invLen,
  const float *q_enc, int q_encLen, const float *out_enc, int out_encLen,
  uint16 *out_u16, int out_u16Len, uint32 *stats_us, int stats_usLen) {
  if ((uint32)q_invLen != n_head_q || q_f32Len != out_u16Len) {
    FARF(ERROR, "attn_q2_step_fq: bad q_inv or lengths");
    return AEE_EBADPARM;
  }
  return attn_q2_step_impl(
    handle, kv_handle, row0, k_rows, k_rowsLen, v_rows, v_rowsLen, n_q,
    cache_from, cache_to, n_head_q, window, NULL, q_f32, q_inv, q_f32Len, q_enc,
    q_encLen, out_enc, out_encLen, out_u16, out_u16Len, stats_us, stats_usLen);
}

/* ---- [P2] the layer's calibration on the DSP ---------------------------- */

#define Q2C_MAX_LANES 16u

typedef struct {
  const uint16_t *k, *v;
  uint32_t n_rows, stride, n_kv, hd;
  const float *q;
  uint32_t n_q, n_hq, q_stride;
  uint16_t *vmax;   /**< [lane][n_kv * hd] |V| max as fp16 bits */
  uint16_t *kmax;   /**< [lane][n_kv] */
  float *qlo, *qhi; /**< [lane][n_hq] */
} q2c_ctx;

/** |x| of fp16 bits is the bits with the sign cleared, and for finite
   values their magnitude order is the unsigned order of those bits: the
   maxima are taken as u16 and converted once, exactly, at the end. The
   row split across lanes changes nothing: max is order-free. */
static void q2c_worker(uint32_t n_threads, uint32_t i, void *v) {
  const q2c_ctx *c = (const q2c_ctx *)v;
  const HVX_Vector mask = Q6_Vh_vsplat_R(0x7fff);
  uint16_t *vmax = c->vmax + (size_t)i * c->n_kv * c->hd;
  uint16_t *kmax = c->kmax + (size_t)i * c->n_kv;
  float *qlo = c->qlo + (size_t)i * c->n_hq,
        *qhi = c->qhi + (size_t)i * c->n_hq;
  {
    const uint32_t lo = (uint32_t)(((uint64_t)c->n_rows * i) / n_threads);
    const uint32_t hi =
      (uint32_t)(((uint64_t)c->n_rows * (i + 1u)) / n_threads);
    const uint32_t nvec = c->hd / 64u;
    memset(vmax, 0, (size_t)c->n_kv * c->hd * sizeof(uint16_t));
    for (uint32_t n = 0; n < c->n_kv; ++n) {
      HVX_Vector acc = Q6_V_vzero();
      for (uint32_t r = lo; r < hi; ++r) {
        const HVX_UVector *kr =
          (const HVX_UVector *)(c->k + (size_t)r * c->stride + n * c->hd);
        const HVX_UVector *vr =
          (const HVX_UVector *)(c->v + (size_t)r * c->stride + n * c->hd);
        HVX_UVector *vm = (HVX_UVector *)(vmax + n * c->hd);
        for (uint32_t j = 0; j < nvec; ++j) {
          acc = Q6_Vuh_vmax_VuhVuh(acc, Q6_V_vand_VV(kr[j], mask));
          vm[j] = Q6_Vuh_vmax_VuhVuh(vm[j], Q6_V_vand_VV(vr[j], mask));
        }
      }
      uint16_t lanes[64] __attribute__((aligned(128)));
      *(HVX_Vector *)lanes = acc;
      uint16_t m = 0;
      for (uint32_t j = 0; j < 64u; ++j)
        m = lanes[j] > m ? lanes[j] : m;
      kmax[n] = m;
    }
  }
  {
    const uint32_t lo = (uint32_t)(((uint64_t)c->n_q * i) / n_threads);
    const uint32_t hi = (uint32_t)(((uint64_t)c->n_q * (i + 1u)) / n_threads);
    const uint32_t nvec = c->hd / 32u;
    for (uint32_t h = 0; h < c->n_hq; ++h) {
      HVX_Vector vlo = hvx_splat_sf(1e30f), vhi = hvx_splat_sf(-1e30f);
      for (uint32_t r = lo; r < hi; ++r) {
        const HVX_UVector *qr =
          (const HVX_UVector *)(c->q + (size_t)r * c->q_stride + h * c->hd);
        for (uint32_t j = 0; j < nvec; ++j) {
          vlo = Q6_Vsf_vmin_VsfVsf(vlo, qr[j]);
          vhi = Q6_Vsf_vmax_VsfVsf(vhi, qr[j]);
        }
      }
      float l[32] __attribute__((aligned(128)));
      float u[32] __attribute__((aligned(128)));
      *(HVX_Vector *)l = vlo;
      *(HVX_Vector *)u = vhi;
      float flo = 1e30f, fhi = -1e30f;
      for (uint32_t j = 0; j < 32u; ++j) {
        flo = l[j] < flo ? l[j] : flo;
        fhi = u[j] > fhi ? u[j] : fhi;
      }
      qlo[h] = flo;
      qhi[h] = fhi;
    }
  }
}

/** std::round: halves away from zero; the DSP image has no roundf. Below
   2^23 trunc then the exact fraction compared with 0.5; above, integral. */
static inline float q2c_roundf(float x) {
  const float a = x < 0.0f ? -x : x;
  if (a >= 8388608.0f) {
    return x;
  }
  float r = (float)(int32_t)a; /* trunc; a - r is exact */
  if (a - r >= 0.5f) {
    r += 1.0f;
  }
  return x < 0.0f ? -r : r;
}

int nntr_hvx_attn_q2_step_fqc(remote_handle64 handle, uint32 kv_handle,
                              uint32 row0, const uint16 *k_rows, int k_rowsLen,
                              const uint16 *v_rows, int v_rowsLen, uint32 n_q,
                              uint32 cache_from, uint32 cache_to,
                              uint32 n_head_q, uint32 window,
                              const float *q_f32, int q_f32Len, float *calib,
                              int calibLen, uint16 *out_u16, int out_u16Len,
                              uint32 *stats_us, int stats_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv || n_head_q == 0 || kv->n_head_kv == 0 ||
      n_head_q % kv->n_head_kv != 0 || kv->head_dim % 64u != 0 ||
      q_f32Len != out_u16Len || k_rowsLen != v_rowsLen ||
      (uint32)calibLen !=
        kv->n_head_kv + kv->n_head_kv * kv->head_dim + 4u * n_head_q) {
    FARF(ERROR, "attn_q2_step_fqc: bad handle or lengths");
    return AEE_EBADPARM;
  }
  const uint32_t n_kv = kv->n_head_kv, hd = kv->head_dim;
  const uint32_t stride = n_kv * hd;
  const uint32_t n_rows = (uint32_t)k_rowsLen / stride;
  const uint32_t lanes = hvx_worker_pool_workers(s->quant_pool) + 1u;
  if (lanes > Q2C_MAX_LANES || (uint32_t)k_rowsLen != n_rows * stride) {
    return AEE_EBADPARM;
  }
  const uint64_t tc0 = HAP_perf_get_time_us();
  uint16_t *vmax =
    (uint16_t *)malloc((size_t)lanes * stride * sizeof(uint16_t));
  uint16_t *kmax = (uint16_t *)malloc((size_t)lanes * n_kv * sizeof(uint16_t));
  float *qlo = (float *)malloc((size_t)lanes * n_head_q * sizeof(float) * 2u);
  float *s_k = calib, *s_v = calib + n_kv;
  float *q_enc = s_v + stride, *out_enc = q_enc + 2u * n_head_q;
  float q_inv[64];
  if (!vmax || !kmax || !qlo || n_head_q > 64u) {
    free(vmax);
    free(kmax);
    free(qlo);
    return AEE_ENOMEMORY;
  }
  q2c_ctx c = {k_rows,   v_rows,
               n_rows,   stride,
               n_kv,     hd,
               q_f32,    n_q,
               n_head_q, n_head_q * hd,
               vmax,     kmax,
               qlo,      qlo + (size_t)lanes * n_head_q};
  hvx_worker_pool_run(s->quant_pool, q2c_worker, &c,
                      n_rows > n_q ? n_rows : n_q);
  /** the lanes merged, the maxima to f32 (exact), the scales as
     mha_core::calibrate_q2_scales derives them, in its order of operations */
  for (uint32_t n = 0; n < n_kv; ++n) {
    uint16_t m = 0;
    for (uint32_t l = 0; l < lanes; ++l)
      m = kmax[(size_t)l * n_kv + n] > m ? kmax[(size_t)l * n_kv + n] : m;
    s_k[n] = hexkl_kv_q_hf_to_f32(m);
  }
  for (uint32_t e = 0; e < stride; ++e) {
    uint16_t m = 0;
    for (uint32_t l = 0; l < lanes; ++l)
      m = vmax[(size_t)l * stride + e] > m ? vmax[(size_t)l * stride + e] : m;
    s_v[e] = hexkl_kv_q_hf_to_f32(m);
  }
  const uint32_t G = n_head_q / n_kv;
  for (uint32_t h = 0; h < n_head_q; ++h) {
    float lo = 1e30f, hi = -1e30f;
    for (uint32_t l = 0; l < lanes; ++l) {
      const float a = c.qlo[(size_t)l * n_head_q + h];
      const float b = c.qhi[(size_t)l * n_head_q + h];
      lo = a < lo ? a : lo;
      hi = b > hi ? b : hi;
    }
    const float range = (hi - lo) > 1e-6f ? (hi - lo) : 1e-6f;
    q_enc[2u * h] = range / 65535.0f;
    q_enc[2u * h + 1u] = q2c_roundf(-lo / q_enc[2u * h]);
    float vm = 0.0f;
    for (uint32_t d = 0; d < hd; ++d) {
      const float x = s_v[(size_t)(h / G) * hd + d];
      vm = x > vm ? x : vm;
    }
    const float two = 2.0f * vm;
    out_enc[2u * h] = (two > 1e-6f ? two : 1e-6f) / 65535.0f;
    out_enc[2u * h + 1u] = 32768.0f;
    q_inv[h] = 1.0f / q_enc[2u * h];
  }
  for (uint32_t n = 0; n < n_kv; ++n)
    s_k[n] = s_k[n] > 0.0f ? s_k[n] / 127.0f : 1.0f;
  for (uint32_t e = 0; e < stride; ++e)
    s_v[e] = s_v[e] > 0.0f ? s_v[e] / 127.0f : 1.0f;
  free(vmax);
  free(kmax);
  free(qlo);
  int rc = hexkl_kv_q_set_fixed_scales(&s->kv_q, kv_handle, s_k, s_v);
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "attn_q2_step_fqc: fixed scales: 0x%08x", rc);
    return rc;
  }
  const uint32_t calib_us = (uint32_t)(HAP_perf_get_time_us() - tc0);
  rc = attn_q2_step_impl(handle, kv_handle, row0, k_rows, k_rowsLen, v_rows,
                         v_rowsLen, n_q, cache_from, cache_to, n_head_q, window,
                         NULL, q_f32, q_inv, q_f32Len, q_enc,
                         (int)(2u * n_head_q), out_enc, (int)(2u * n_head_q),
                         out_u16, out_u16Len, stats_us, stats_usLen);
  if (rc == AEE_SUCCESS && stats_usLen > 2) {
    stats_us[2] = calib_us; /* the slot the DDR Q pass used to report */
  }
  return rc;
}

int nntr_hvx_probe_wh_i8_pos(remote_handle64 handle, uint16 *pos, int posLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || posLen != 1024) {
    return AEE_EBADPARM;
  }
  const uint16_t *p = hexkl_kv_q_wh_i8_pos(s->vtcm_base);
  if (!p) {
    return AEE_EUNSUPPORTED;
  }
  memcpy(pos, p, 1024u * sizeof(uint16_t));
  return AEE_SUCCESS;
}

int nntr_hvx_probe_softmax_q(remote_handle64 handle, uint32 n_col_tiles,
                             uint32 col0, uint32 n_cols, uint32 row0,
                             uint32 n_rows, uint32 window, uint32 rho_q15,
                             uint32 frac_bits, const int16 *s_tiles,
                             int s_tilesLen, const int16 *corr, int corrLen,
                             uint8 *p_tiles, int p_tilesLen, int32 *rowsum,
                             int rowsumLen, uint32 *stats, int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || n_col_tiles == 0 || n_col_tiles > 128u || frac_bits == 0 ||
      frac_bits > 8 || rho_q15 > 32767u) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)s_tilesLen != n_col_tiles * HVX_SOFTMAX_Q_TILE ||
      (uint32_t)corrLen != (n_col_tiles + 1u) * 32u ||
      (uint32_t)p_tilesLen != n_col_tiles * HVX_SOFTMAX_Q_TILE ||
      (uint32_t)rowsumLen != HVX_SOFTMAX_Q_ROWSUM_WORDS || statsLen < 1) {
    FARF(ERROR, "probe_softmax_q: bad lengths");
    return AEE_EBADPARM;
  }
  uint8_t *vb = s->vtcm_base;
  memcpy(vb + OFF_SM_S, s_tiles, (size_t)s_tilesLen * 2u);
  memcpy(vb + OFF_SM_CORR, corr, (size_t)corrLen * 2u);
  hvx_softmax_q_block b;
  b.n_col_tiles = n_col_tiles;
  b.col0 = col0;
  b.n_cols = n_cols;
  b.row0 = row0;
  b.n_rows = n_rows;
  b.window = window;
  b.rho_q15 = (uint16_t)rho_q15;
  b.frac_bits = (uint8_t)frac_bits;
  const uint64_t c0 = HAP_perf_get_pcycles();
  hvx_softmax_q(&b, (const int16_t *)(vb + OFF_SM_S),
                (const int16_t *)(vb + OFF_SM_CORR), vb + OFF_SM_P,
                (int32_t *)(vb + OFF_SM_RS), vb + OFF_SM_SCR);
  const uint64_t c1 = HAP_perf_get_pcycles();
  memcpy(p_tiles, vb + OFF_SM_P, (size_t)p_tilesLen);
  memcpy(rowsum, vb + OFF_SM_RS, (size_t)rowsumLen * 4u);
  stats[0] = (uint32)(c1 - c0);
  for (int i = 1; i < statsLen && i < 5; ++i) {
    stats[i] = 0;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_probe_softmax_q16(remote_handle64 handle, uint32 n_col_tiles,
                               uint32 col0, uint32 n_cols, uint32 row0,
                               uint32 n_rows, uint32 window, uint32 rho_q15,
                               uint32 frac_bits, const int16 *s_tiles,
                               int s_tilesLen, const int16 *corr, int corrLen,
                               uint8 *p_lo, int p_loLen, uint8 *p_hi,
                               int p_hiLen, int32 *rowsum, int rowsumLen,
                               uint32 *stats, int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || n_col_tiles == 0 || n_col_tiles > 128u || frac_bits == 0 ||
      frac_bits > 9 || rho_q15 > 32767u) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)s_tilesLen != n_col_tiles * HVX_SOFTMAX_Q_TILE ||
      (uint32_t)corrLen != (n_col_tiles + 1u) * 32u ||
      (uint32_t)p_loLen != n_col_tiles * HVX_SOFTMAX_Q_TILE ||
      (uint32_t)p_hiLen != n_col_tiles * HVX_SOFTMAX_Q_TILE ||
      (uint32_t)rowsumLen != HVX_SOFTMAX_Q_ROWSUM_WORDS || statsLen < 1) {
    FARF(ERROR, "probe_softmax_q16: bad lengths");
    return AEE_EBADPARM;
  }
  uint8_t *vb = s->vtcm_base;
  memcpy(vb + OFF_SM_S, s_tiles, (size_t)s_tilesLen * 2u);
  memcpy(vb + OFF_SM_CORR, corr, (size_t)corrLen * 2u);
  hvx_softmax_q_block b;
  b.n_col_tiles = n_col_tiles;
  b.col0 = col0;
  b.n_cols = n_cols;
  b.row0 = row0;
  b.n_rows = n_rows;
  b.window = window;
  b.rho_q15 = (uint16_t)rho_q15;
  b.frac_bits = (uint8_t)frac_bits;
  const uint64_t c0 = HAP_perf_get_pcycles();
  hvx_softmax_q16(&b, (const int16_t *)(vb + OFF_SM_S),
                  (const int16_t *)(vb + OFF_SM_CORR), vb + OFF_SM_P,
                  vb + OFF_SM_PH, (int32_t *)(vb + OFF_SM_RS), vb + OFF_SM_SCR);
  const uint64_t c1 = HAP_perf_get_pcycles();
  memcpy(p_lo, vb + OFF_SM_P, (size_t)p_loLen);
  memcpy(p_hi, vb + OFF_SM_PH, (size_t)p_hiLen);
  memcpy(rowsum, vb + OFF_SM_RS, (size_t)rowsumLen * 4u);
  stats[0] = (uint32)(c1 - c0);
  for (int i = 1; i < statsLen && i < 5; ++i) {
    stats[i] = 0;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_attn_q_prefill(remote_handle64 handle, uint32 kv_handle,
                            uint32 n_q, uint32 cache_from, uint32 cache_to,
                            uint32 n_head_q, uint32 window, uint32 br,
                            uint32 bc, float softcap, const float *q_f32,
                            int q_f32Len, const float *sinks, int sinksLen,
                            float *out_f32, int out_f32Len, uint32 *stats_us,
                            int stats_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv || cache_to > kv->max_rows) {
    FARF(ERROR, "attn_q_prefill: bad handle or cache_to");
    return AEE_EBADPARM;
  }
  hexkl_attn_f16_shape shape = {n_q,           cache_from,   cache_to, n_head_q,
                                kv->n_head_kv, kv->head_dim, window,   softcap};
  hexkl_attn_f16_tiling tiling = {br, bc, 0, 0};
  if (br == 0 || bc == 0) {
    hexkl_attn_f16_choose_tiling(&shape, &tiling);
  }
  if (hexkl_attn_q_tiling_init(&shape, &tiling) != HEXKL_ATTN_OK) {
    FARF(ERROR, "attn_q_prefill: bad shape/tiling");
    return AEE_EBADPARM;
  }
  const uint64_t q_elems = (uint64_t)n_q * n_head_q * kv->head_dim;
  if ((uint64_t)q_f32Len != q_elems || (uint64_t)out_f32Len != q_elems ||
      stats_usLen < QSTAT_COUNT ||
      (sinksLen != 0 && (uint32)sinksLen != n_head_q)) {
    FARF(ERROR, "attn_q_prefill: bad lengths");
    return AEE_EBADPARM;
  }
  hexkl_attn_q_io io;
  memset(&io, 0, sizeof(io));
  io.q = q_f32;
  io.q_stride = n_head_q * kv->head_dim;
  io.sinks = sinksLen ? sinks : NULL;
  io.out = out_f32;
  io.out_stride = n_head_q * kv->head_dim;
  io.kv = kv;
  hexkl_attn_q_stats st;
  int res = hexkl_attn_q_prefill(s->vtcm_base, s->config_off, &shape, &tiling,
                                 &io, s->quant_pool, &st);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "attn_q_prefill: kernel failed: 0x%08x", res);
    return res;
  }
  stats_us[QSTAT_QPREP] = (uint32)st.us_qprep;
  stats_us[QSTAT_DMA] = (uint32)st.us_dma;
  stats_us[QSTAT_QK] = (uint32)st.us_qk;
  stats_us[QSTAT_DEQUANT] = (uint32)st.us_dequant;
  stats_us[QSTAT_SOFTMAX] = (uint32)st.us_softmax;
  stats_us[QSTAT_PQUANT] = (uint32)st.us_pquant;
  stats_us[QSTAT_PV] = (uint32)st.us_pv;
  stats_us[QSTAT_OUPD] = (uint32)st.us_oupd;
  stats_us[QSTAT_STORE] = (uint32)st.us_store;
  stats_us[QSTAT_N_BLOCKS] = st.n_blocks;
  stats_us[QSTAT_US_TOTAL] = (uint32)st.us_total;
  stats_us[QSTAT_KCYCLES] = (uint32)(st.pcycles / 1000u);
  return AEE_SUCCESS;
}

int nntr_hvx_attn_q_decode(remote_handle64 handle, uint32 kv_handle, uint32 n_q,
                           uint32 cache_from, uint32 cache_to, uint32 n_head_q,
                           uint32 window, float softcap, const float *q_f32,
                           int q_f32Len, const float *sinks, int sinksLen,
                           float *out_f32, int out_f32Len, uint32 *stats_us,
                           int stats_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv || cache_to > kv->max_rows) {
    FARF(ERROR, "attn_q_decode: bad handle or cache_to");
    return AEE_EBADPARM;
  }
  hexkl_attn_f16_shape shape = {n_q,           cache_from,   cache_to, n_head_q,
                                kv->n_head_kv, kv->head_dim, window,   softcap};
  const uint64_t q_elems = (uint64_t)n_q * n_head_q * kv->head_dim;
  if ((uint64_t)q_f32Len != q_elems || (uint64_t)out_f32Len != q_elems ||
      stats_usLen < 2 || (sinksLen != 0 && (uint32)sinksLen != n_head_q)) {
    FARF(ERROR, "attn_q_decode: bad lengths");
    return AEE_EBADPARM;
  }
  hexkl_attn_q_io io;
  memset(&io, 0, sizeof(io));
  io.q = q_f32;
  io.q_stride = n_head_q * kv->head_dim;
  io.sinks = sinksLen ? sinks : NULL;
  io.out = out_f32;
  io.out_stride = n_head_q * kv->head_dim;
  io.kv = kv;
  uint64_t us = 0;
  const uint64_t c0 = HAP_perf_get_pcycles();
  int res = hvx_attn_decode_q(&shape, &io, s->quant_pool, &us);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "attn_q_decode: kernel failed: 0x%08x", res);
    return res;
  }
  stats_us[0] = (uint32)us;
  stats_us[1] = (uint32)((HAP_perf_get_pcycles() - c0) / 1000u);
  return AEE_SUCCESS;
}

int nntr_hvx_attn_q_step(remote_handle64 handle, uint32 kv_handle, uint32 row0,
                         const uint16 *k_rows, int k_rowsLen,
                         const uint16 *v_rows, int v_rowsLen, uint32 n_q,
                         uint32 cache_from, uint32 cache_to, uint32 n_head_q,
                         uint32 window, float softcap, const float *q_f32,
                         int q_f32Len, const float *sinks, int sinksLen,
                         float *out_f32, int out_f32Len, uint32 *stats_us,
                         int stats_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!kv || cache_to > kv->max_rows || stats_usLen < 8) {
    FARF(ERROR, "attn_q_step: bad handle, cache_to or stats");
    return AEE_EBADPARM;
  }
  hexkl_kv_q_append_stats ast;
  memset(&ast, 0, sizeof(ast));
  const uint64_t t0 = HAP_perf_get_time_us();
  const uint32_t stride = kv->n_head_kv * kv->head_dim;
  if (k_rowsLen != v_rowsLen || k_rowsLen < 0 ||
      ((uint32_t)k_rowsLen % stride) != 0) {
    FARF(ERROR, "attn_q_step: bad row lengths");
    return AEE_EBADPARM;
  }
  if (k_rowsLen > 0) {
    const int rc =
      hexkl_kv_q_append(&s->kv_q, kv_handle, row0, (uint32_t)k_rowsLen / stride,
                        k_rows, v_rows, s->vtcm_base, &ast);
    if (rc != AEE_SUCCESS) {
      FARF(ERROR, "attn_q_step: append failed: 0x%08x", rc);
      return rc;
    }
  }
  const uint64_t t1 = HAP_perf_get_time_us();

  hexkl_attn_f16_shape shape = {n_q,           cache_from,   cache_to, n_head_q,
                                kv->n_head_kv, kv->head_dim, window,   softcap};
  const uint64_t q_elems = (uint64_t)n_q * n_head_q * kv->head_dim;
  if ((uint64_t)q_f32Len != q_elems || (uint64_t)out_f32Len != q_elems ||
      (sinksLen != 0 && (uint32)sinksLen != n_head_q)) {
    FARF(ERROR, "attn_q_step: bad lengths");
    return AEE_EBADPARM;
  }
  hexkl_attn_q_io io;
  memset(&io, 0, sizeof(io));
  io.q = q_f32;
  io.q_stride = n_head_q * kv->head_dim;
  io.sinks = sinksLen ? sinks : NULL;
  io.out = out_f32;
  io.out_stride = n_head_q * kv->head_dim;
  io.kv = kv;

  int res;
  const int decode = n_q < 5u && kv->head_dim <= 128u;
  if (decode) {
    res = hvx_attn_decode_q(&shape, &io, s->quant_pool, NULL);
  } else {
    hexkl_attn_f16_tiling tiling = {0, 0, 0, 0};
    hexkl_attn_f16_choose_tiling(&shape, &tiling);
    if (hexkl_attn_q_tiling_init(&shape, &tiling) != HEXKL_ATTN_OK) {
      return AEE_EBADPARM;
    }
    res = hexkl_attn_q_prefill(s->vtcm_base, s->config_off, &shape, &tiling,
                               &io, s->quant_pool, NULL);
  }
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "attn_q_step: kernel failed: 0x%08x", res);
    return res;
  }
  const uint64_t t2 = HAP_perf_get_time_us();
  stats_us[0] = (uint32)(t1 - t0);
  stats_us[1] = (uint32)(t2 - t1);
  stats_us[2] = decode ? 0u : 1u;
  stats_us[3] = (uint32)(t2 - t0);
  stats_us[4] = ast.quant_us;
  stats_us[5] = ast.stage_us;
  stats_us[6] = ast.bake_us;
  stats_us[7] = (uint32)((uint32_t)k_rowsLen / stride);
  return AEE_SUCCESS;
}
