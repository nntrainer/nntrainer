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

#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <remote.h>

#include "hexkl_acc_tile.h"
#include "hexkl_attn_q.h"
#include "hexkl_attn_q2.h"
#include "hexkl_cvt.h"
#include "hexkl_kv_q.h"
#include "hexkl_micro.h"
#include "hvx_attn_decode_q.h"
#include "hvx_softmax_q.h"
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

int nntr_hvx_attn_q2_step(remote_handle64 handle, uint32 kv_handle, uint32 row0,
                          const uint16 *k_rows, int k_rowsLen,
                          const uint16 *v_rows, int v_rowsLen, uint32 n_q,
                          uint32 cache_from, uint32 cache_to, uint32 n_head_q,
                          uint32 window, const uint16 *q_u16, int q_u16Len,
                          const float *q_enc, int q_encLen,
                          const float *out_enc, int out_encLen, uint16 *out_u16,
                          int out_u16Len, uint32 *stats_us, int stats_usLen) {
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
    const int rc =
      hexkl_kv_q_append(&s->kv_q, kv_handle, row0, (uint32_t)k_rowsLen / stride,
                        k_rows, v_rows, s->vtcm_base, &ast);
    if (rc != AEE_SUCCESS) {
      FARF(ERROR, "attn_q2_step: append failed: 0x%08x", rc);
      return rc;
    }
  }
  const uint64_t t1 = HAP_perf_get_time_us();

  hexkl_attn_f16_shape shape = {n_q,           cache_from,   cache_to, n_head_q,
                                kv->n_head_kv, kv->head_dim, window,   0.0f};
  const uint64_t q_elems = (uint64_t)n_q * n_head_q * kv->head_dim;
  if ((uint64_t)q_u16Len != q_elems || (uint64_t)out_u16Len != q_elems) {
    FARF(ERROR, "attn_q2_step: bad lengths");
    return AEE_EBADPARM;
  }
  hexkl_attn_q2_io io;
  io.q = q_u16;
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
