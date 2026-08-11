// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_mm_u8i4_dma.c
 * @date   06 Aug 2026
 * @brief  Persistent u8i4 weight registry and the cross-matmul DMA layer path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hexkl_mm_u8i4_dma.h"

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>

#include "hexkl_dma_ring.h"
#include "hexkl_hmx_geom.h"
#include "hexkl_micro.h"
#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"

#define ROUND_UP_U32(v, a) HEXKL_ROUND_UP_U32(v, a)
#define WEIGHT_TILE_BYTES_U8I4 HEXKL_WEIGHT_TILE_BYTES_I4
#define ACC_TILE_BYTES HEXKL_ACC_TILE_BYTES
/** @brief Largest single DMA row HexKL/HVX will move correctly; rows above
 *         this size have a documented hardware bug (doc13 §3a, §5). */
#define MAX_DMA_ROW_BYTES 262144u

static uint32_t dma_row_size_dividing(uint32_t total_bytes) {
  uint32_t rs = MAX_DMA_ROW_BYTES;
  while (rs > 1 && (total_bytes % rs) != 0) {
    rs >>= 1;
  }
  return rs;
}

int hexkl_weight_u8i4_register(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                               uint32_t vtcm_size, uint32_t K, uint32_t N,
                               const int8_t *w_i4_rm, const float *w_scale,
                               const int32_t *colsum_w, const float *bias,
                               uint32_t *out_handle) {
  if (!tbl || !vtcm_base || !w_i4_rm || !w_scale || !colsum_w || !bias ||
      !out_handle || K == 0 || N == 0) {
    return AEE_EBADPARM;
  }
  if ((K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0 ||
      (N % HEXKL_HMX_INT8_BLOCK_N_COL) != 0) {
    return AEE_EBADPARM;
  }

  uint32_t slot = HEXKL_MM_U8I4_MAX_WEIGHTS;
  for (uint32_t i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i) {
    if (!tbl->slots[i].in_use) {
      slot = i;
      break;
    }
  }
  if (slot == HEXKL_MM_U8I4_MAX_WEIGHTS) {
    return AEE_ENOMEMORY;
  }

  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n_tiles = N / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t wh_bytes = k_tiles * n_tiles * WEIGHT_TILE_BYTES_U8I4;
  if (wh_bytes > vtcm_size) {
    return AEE_ENOMEMORY; // weight alone does not fit the VTCM scratch arena
  }

  // Bake every tile into VTCM (borrowed as scratch -- caller holds the HMX
  // lock and has no other VTCM use in flight), then copy the baked bytes
  // out to DSP heap memory where they stay resident across calls. A plain
  // copy, not the DMA ring: registration happens once per weight at model
  // load, off the per-token hot path, so its cost is not what this file
  // exists to optimise.
  for (uint32_t kt = 0; kt < k_tiles; ++kt) {
    for (uint32_t nt = 0; nt < n_tiles; ++nt) {
      const uint32_t off = (kt * n_tiles + nt) * WEIGHT_TILE_BYTES_U8I4;
      int res = hexkl_micro_hmx_rm_to_wh_i4(vtcm_base, off, w_i4_rm, kt, nt, N);
      if (res != AEE_SUCCESS) {
        return res;
      }
    }
  }

  hexkl_weight_u8i4 *h = &tbl->slots[slot];
  h->wh_bytes = (uint8_t *)malloc(wh_bytes);
  h->w_scale = (float *)malloc(sizeof(float) * N);
  h->colsum_w = (int32_t *)malloc(sizeof(int32_t) * N);
  h->bias = (float *)malloc(sizeof(float) * N);
  if (!h->wh_bytes || !h->w_scale || !h->colsum_w || !h->bias) {
    free(h->wh_bytes);
    free(h->w_scale);
    free(h->colsum_w);
    free(h->bias);
    memset(h, 0, sizeof(*h));
    return AEE_ENOMEMORY;
  }
  memcpy(h->wh_bytes, vtcm_base, wh_bytes);
  memcpy(h->w_scale, w_scale, sizeof(float) * N);
  memcpy(h->colsum_w, colsum_w, sizeof(int32_t) * N);
  memcpy(h->bias, bias, sizeof(float) * N);
  h->K = K;
  h->N = N;
  h->in_use = 1;

  *out_handle = slot;
  return AEE_SUCCESS;
}

int hexkl_weight_u8i4_release(hexkl_weight_u8i4_table *tbl, uint32_t handle) {
  if (!tbl || handle >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
      !tbl->slots[handle].in_use) {
    return AEE_EBADPARM;
  }
  hexkl_weight_u8i4 *h = &tbl->slots[handle];
  free(h->wh_bytes);
  free(h->w_scale);
  free(h->colsum_w);
  free(h->bias);
  memset(h, 0, sizeof(*h));
  return AEE_SUCCESS;
}

int hexkl_mm_u8i4_layer_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                            uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                            uint32_t K, const uint32_t *handles,
                            uint32_t n_handles, const float *act_f32,
                            float *out_cat) {
  if (!tbl || !vtcm_base || !handles || n_handles == 0 || !act_f32 ||
      !out_cat || M == 0 || K == 0) {
    return AEE_EBADPARM;
  }

  const uint32_t m_pad = ROUND_UP_U32(M, HEXKL_HMX_INT8_BLOCK_N_ROW);
  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n_rblocks = m_pad / HEXKL_HMX_INT8_BLOCK_N_ROW;
  if ((K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0) {
    return AEE_EBADPARM;
  }

  // Validate every handle up front -- K must match, and this is also where
  // the widest weight (for the double-buffer size) and the total output
  // width (for out_cat bounds) come from.
  uint32_t n_tiles_max = 0, n_max = 0;
  for (uint32_t i = 0; i < n_handles; ++i) {
    if (handles[i] >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
        !tbl->slots[handles[i]].in_use) {
      return AEE_EBADPARM;
    }
    const hexkl_weight_u8i4 *h = &tbl->slots[handles[i]];
    if (h->K != K) {
      return AEE_EBADPARM;
    }
    const uint32_t nt = h->N / HEXKL_HMX_INT8_BLOCK_N_COL;
    if (nt > n_tiles_max) {
      n_tiles_max = nt;
    }
    if (h->N > n_max) {
      n_max = h->N;
    }
  }

  // VTCM layout: activation (all row-bands, shared across every handle) |
  // weight double-buffer (sized for the widest handle) | one result tile.
  const uint32_t act_bytes =
    n_rblocks * k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t act_off = 0;
  const uint32_t wb_max = k_tiles * n_tiles_max * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t wbuf[2] = {
    ROUND_UP_U32(act_off + act_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT),
    ROUND_UP_U32(act_off + act_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT) + wb_max,
  };
  const uint32_t result_off =
    ROUND_UP_U32(wbuf[1] + wb_max, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  if (result_off + ACC_TILE_BYTES > config_off) {
    return AEE_ENOMEMORY; // double-buffered widest weight does not fit VTCM
  }
  if (result_off + ACC_TILE_BYTES > vtcm_size) {
    return AEE_ENOMEMORY;
  }

  // K1/K2: quantize the shared activation once, straight into its AH tiles
  // in VTCM -- no separate layout pass, matching hvx_quant_pack_u8_ah's
  // contract.
  float *act_scale = (float *)malloc(sizeof(float) * m_pad);
  int32_t *act_zp = (int32_t *)malloc(sizeof(int32_t) * m_pad);
  int32_t *acc_scratch =
    (int32_t *)malloc(sizeof(int32_t) * (size_t)m_pad * n_max);
  if (!act_scale || !act_zp || !acc_scratch) {
    free(act_scale);
    free(act_zp);
    free(acc_scratch);
    return AEE_ENOMEMORY;
  }
  hvx_quant_rows_u8_params(act_f32, M, m_pad, K, act_scale, act_zp);
  hvx_quant_pack_u8_ah(act_f32, M, m_pad, K, act_scale, act_zp,
                       vtcm_base + act_off);

  int rc = AEE_SUCCESS;
  size_t out_off = 0;
  hexkl_dma_ring_reset();

  // Load handle 0's weight before the loop starts -- there is nothing to
  // prefetch it behind yet, so this one transfer is blocking.
  {
    const hexkl_weight_u8i4 *h0 = &tbl->slots[handles[0]];
    const uint32_t nt0 = h0->N / HEXKL_HMX_INT8_BLOCK_N_COL;
    const uint32_t wb0 = k_tiles * nt0 * WEIGHT_TILE_BYTES_U8I4;
    const uint32_t rs0 = dma_row_size_dividing(wb0);
    hexkl_dma_ring_push2d(vtcm_base + wbuf[0], h0->wh_bytes, rs0, rs0, rs0,
                          wb0 / rs0, /*src_vtcm=*/0, /*dst_vtcm=*/1);
    hexkl_dma_ring_drain();
  }

  for (uint32_t i = 0; i < n_handles; ++i) {
    const hexkl_weight_u8i4 *h = &tbl->slots[handles[i]];
    const uint32_t nt_n = h->N / HEXKL_HMX_INT8_BLOCK_N_COL;
    const uint32_t wcur = wbuf[i & 1u];

    if (i + 1 < n_handles) {
      // Cross-matmul prefetch (doc13 §3a): while handle i computes below,
      // stream handle i+1's weight into the OTHER buffer in the
      // background. One transfer, chunked under the >512KB-row DMA bug's
      // threshold, not one per tile.
      const hexkl_weight_u8i4 *hn = &tbl->slots[handles[i + 1]];
      const uint32_t nt_next = hn->N / HEXKL_HMX_INT8_BLOCK_N_COL;
      const uint32_t wb_next = k_tiles * nt_next * WEIGHT_TILE_BYTES_U8I4;
      const uint32_t rs = dma_row_size_dividing(wb_next);
      hexkl_dma_ring_push2d(vtcm_base + wbuf[(i + 1) & 1u], hn->wh_bytes, rs,
                            rs, rs, wb_next / rs, /*src_vtcm=*/0,
                            /*dst_vtcm=*/1);
    }

    for (uint32_t rb = 0; rb < n_rblocks; ++rb) {
      for (uint32_t nt = 0; nt < nt_n; ++nt) {
        hexkl_micro_hmx_acc_clear_int32();
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
          const uint32_t act_tile_off =
            act_off + (rb * k_tiles + kt) * HEXKL_HMX_ACTIVATION_ALIGNMENT;
          const uint32_t w_tile_off =
            wcur + (kt * nt_n + nt) * WEIGHT_TILE_BYTES_U8I4;
          rc = hexkl_micro_hmx_mm_u8i4(vtcm_base, act_tile_off, w_tile_off);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off, result_off);
        if (rc != AEE_SUCCESS) {
          goto out;
        }
        rc = hexkl_micro_hmx_copy_32b_to_submatrix(
          vtcm_base, result_off, acc_scratch, rb, nt, m_pad, h->N);
        if (rc != AEE_SUCCESS) {
          goto out;
        }
      }
    }

    // Block until handle i+1's weight has fully landed before moving on to
    // it -- matches the measured bench's "next weight fully in wnxt before
    // matmul i+1" invariant. Dequant below does not touch the DMA engine,
    // so a later pass could move it ahead of this drain to overlap with
    // the prefetch; left as-is for correctness first.
    hexkl_dma_ring_drain();

    hvx_dequant_i32_to_f32(acc_scratch, M, m_pad, h->N, act_scale, act_zp,
                           h->colsum_w, h->w_scale, h->bias, out_cat + out_off);
    out_off += (size_t)M * h->N;
  }

out:
  free(act_scale);
  free(act_zp);
  free(acc_scratch);
  return rc;
}
