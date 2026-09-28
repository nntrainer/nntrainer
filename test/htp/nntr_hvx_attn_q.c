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
#include <remote.h>

#include "hexkl_acc_tile.h"
#include "hexkl_attn_q.h"
#include "hexkl_kv_q.h"
#include "hexkl_micro.h"
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
  QSTAT_COUNT
};

/** @brief One uint8 activation tile (64x32), also its alignment. */
#define ACT_TILE_BYTES HEXKL_HMX_ACTIVATION_ALIGNMENT
/** @brief Largest head_dim / 32. */
#define MAX_DT 8u

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
  OFF_ACC = OFF_W_V + MAX_DT * 1024u, /**< 8 KiB, 2048-aligned */
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
                           s->vtcm_base);
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
      (uint64_t)s_vLen != heads * kv->n_dot_tiles) {
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
  return AEE_SUCCESS;
}
