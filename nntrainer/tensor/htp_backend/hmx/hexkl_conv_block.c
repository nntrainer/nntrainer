// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_conv_block.c
 * @date   22 Sep 2026
 * @brief  An LFM2 conv block -- in_proj, gate, conv1d, gate, out_proj -- in
 *         one call
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The header says what and why; this file is the two loops. Everything
 * about buffer reuse follows hexkl_mm_u8i4_moe.c and its rules apply:
 * two accumulator staging buffers alternate under the HMX issue, the
 * pool runs one batch behind, and every wait sits where the dependency
 * is. The one new hazard is g: phase 1's epilogues write it row-block by
 * row-block and phase 2 reads it with a two-row look-back, so phase 2
 * starts only after phase 1's last epilogue is retired.
 */

#include "hexkl_conv_block.h"

#include <stdatomic.h>
#include <string.h>

#include <AEEStdErr.h>

#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_micro.h"
#include "hexkl_probe.h"
#include "hvx_conv_gate_f32.h"
#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"

#define ROUND_UP_U32(v, a) ((((v) + ((a) - 1)) / (a)) * (a))
#define ROUND_UP_SZ(v, a) ((((v) + ((a) - 1)) / (a)) * (a))
#define CB_SCRATCH_ALIGN 128u

/** @brief Kept in sync with hexkl_mm_u8i4_dma.c's copies by inspection. */
#define WEIGHT_TILE_BYTES_U8I4 512u
#define ACC_TILE_BYTES 8192u

/** @brief Bounded chunk-index arrays; a shape needing more is refused. */
#define CB_MAX_CHUNKS 16u

/* ---- phase 2's background stage: z *= conv1d(g), then z -> mid (u8) ----
 *
 * One block's gate and requantization are ~100 us of HVX work that phase
 * 2 used to run synchronously between the b matmul and the out_proj
 * matmul, with the HMX idle for all of it: 0.75 ms of a 4.35 ms call
 * (doc 51 section 2.8). They are background-lane jobs now, submitted as
 * soon as the block's z is dequantized and retired only where out_proj
 * needs mid -- one block later -- so they run under the next block's b
 * matmul. The lane runs jobs in order, so the requant job cannot start
 * before every gate unit is done. */
#define CB_GATE_UNIT_ROWS 16u

typedef struct {
  float *z;            /**< this block's [64 x C] f32, VTCM */
  const float *g;      /**< the whole conv input, heap */
  const float *conv_w; /**< [3 x C], VTCM */
  uint8_t *mid;        /**< this block's AH tiles, VTCM */
  float *rq_scale;     /**< this block's row params, 64 each */
  int32_t *rq_zp;
  uint32_t mb, m_blk, C;
} cb_stage_ctx;

/** @brief Worker time inside the stage units, summed across workers and
 *         filed under SWIGLU -- the column is 0 for this kernel otherwise.
 *         Hidden work, not wall time: what it says is how much of the HMX
 *         shadow the stage consumes. Atomic because units run
 *         concurrently; HEXKL_PROBE_ADD is not. */
static inline void cb_stage_probe_add(uint64_t t0) {
  if (hexkl_probe_on) {
    atomic_fetch_add_explicit(
      (_Atomic uint64_t *)&hexkl_probe_us[HEXKL_PROBE_SWIGLU],
      hexkl_probe_now() - t0, memory_order_relaxed);
  }
}

static void cb_gate_unit(uint32_t n_units, uint32_t u, void *v) {
  (void)n_units;
  const cb_stage_ctx *c = (const cb_stage_ctx *)v;
  const uint32_t r0 = u * CB_GATE_UNIT_ROWS;
  if (r0 >= c->m_blk) {
    return;
  }
  const uint32_t n =
    (c->m_blk - r0 < CB_GATE_UNIT_ROWS) ? (c->m_blk - r0) : CB_GATE_UNIT_ROWS;
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  hvx_conv_gate_f32(c->z + (size_t)r0 * c->C, c->C, c->g, c->C, c->mb + r0, n,
                    c->C, c->conv_w, NULL);
  cb_stage_probe_add(t0);
}

/** @brief The one requant unit: the same two calls the synchronous path
 *         made, on the calling worker (the pool cannot be used from inside
 *         a unit). Rows [m_blk, m4) are padding the pack takes whole,
 *         zeroed so the bytes are deterministic. */
static void cb_requant_unit(uint32_t n_units, uint32_t u, void *v) {
  (void)n_units;
  (void)u;
  const cb_stage_ctx *c = (const cb_stage_ctx *)v;
  const uint32_t m4 = ROUND_UP_U32(c->m_blk, 4u);
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  if (m4 > c->m_blk) {
    memset(c->z + (size_t)c->m_blk * c->C, 0,
           sizeof(float) * (size_t)(m4 - c->m_blk) * c->C);
  }
  hvx_quant_rows_u8_params(c->z, c->m_blk, HEXKL_HMX_INT8_BLOCK_N_ROW, c->C,
                           c->rq_scale, c->rq_zp, NULL);
  hvx_quant_pack_u8_ah_rows(c->z, NULL, 0u, m4, c->C, c->rq_scale, c->rq_zp,
                            c->mid);
  cb_stage_probe_add(t0);
}

/** @brief Units of a block's gate job. */
#define CB_GATE_UNITS                                                          \
  ((HEXKL_HMX_INT8_BLOCK_N_ROW + CB_GATE_UNIT_ROWS - 1u) / CB_GATE_UNIT_ROWS)

int hexkl_conv_block_layout_get(uint32_t K, uint32_t C, uint32_t N_out,
                                uint32_t arena_bytes,
                                hexkl_conv_block_layout *out) {
  if (!out || K == 0u || C == 0u || N_out == 0u) {
    return AEE_EBADPARM;
  }
  if ((K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0u ||
      (C % HEXKL_HMX_INT8_BLOCK_N_COL) != 0u ||
      (C % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0u ||
      (N_out % HEXKL_HMX_INT8_BLOCK_N_COL) != 0u) {
    return AEE_EBADPARM;
  }
  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t c_ntiles = C / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t c_ktiles = C / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t o_ntiles = N_out / HEXKL_HMX_INT8_BLOCK_N_COL;

  const uint32_t act_bytes = k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t w_in_bytes = k_tiles * c_ntiles * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t w_out_bytes = c_ktiles * o_ntiles * WEIGHT_TILE_BYTES_U8I4;
  /* Slot A holds W_a then W_b (both K x C); slot B holds W_c then W_out. */
  const uint32_t slot_b_bytes =
    w_in_bytes > w_out_bytes ? w_in_bytes : w_out_bytes;
  const uint32_t z_bytes = HEXKL_HMX_INT8_BLOCK_N_ROW * C * 4u;
  const uint32_t mid_bytes = c_ktiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t conv_w_bytes = 3u * C * 4u;

  hexkl_conv_block_layout L;
  L.act_off = 0u;
  L.w_a_off =
    ROUND_UP_U32(L.act_off + act_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.w_b_off =
    ROUND_UP_U32(L.w_a_off + w_in_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.z_off =
    ROUND_UP_U32(L.w_b_off + slot_b_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  /* z and mid twice: phase 2 pipelines blocks (block b's gate and
     requant run on the pool's background lane under block b+1's b matmul
     and block b's out_proj), so each has a live reader while the other
     is being written. */
  L.mid_off =
    ROUND_UP_U32(L.z_off + 2u * z_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.conv_w_off =
    ROUND_UP_U32(L.mid_off + 2u * mid_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.result_off =
    ROUND_UP_U32(L.conv_w_off + conv_w_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  {
    /* Same cap and pairing rule as hexkl_mm_u8i4_moe_layout: 32 tiles a
       staging buffer, even, at least one pair. */
    uint32_t want = 2u * c_ntiles;
    uint32_t fits = 0u;
    if (arena_bytes > L.result_off) {
      fits = (arena_bytes - L.result_off) / (2u * ACC_TILE_BYTES);
    }
    if (want > 32u) {
      want = 32u;
    }
    L.acc_tiles = (fits < want) ? fits : want;
    L.acc_tiles &= ~1u;
    if (L.acc_tiles < 2u) {
      return AEE_ENOMEMORY;
    }
  }
  L.total = L.result_off + 2u * L.acc_tiles * ACC_TILE_BYTES;
  if (L.total > arena_bytes) {
    return AEE_ENOMEMORY;
  }
  *out = L;
  return AEE_SUCCESS;
}

/** @brief A registered weight of the expected shape, or NULL. */
static const hexkl_weight_u8i4 *cb_weight(const hexkl_weight_u8i4_table *tbl,
                                          uint32_t h, uint32_t K, uint32_t N) {
  if (h >= HEXKL_MM_U8I4_MAX_WEIGHTS) {
    return NULL;
  }
  const hexkl_weight_u8i4 *w = &tbl->slots[h];
  return (w->in_use && w->K == K && w->N == N) ? w : NULL;
}

/**
 * @brief Pushes a weight in n-tile chunks of @a cn into @a dst_off.
 * @return chunks pushed; idx_out gets each chunk's ring index
 */
static uint32_t cb_push_chunks(uint8_t *vtcm_base, uint32_t dst_off,
                               const hexkl_weight_u8i4 *h, uint32_t k_tiles,
                               uint32_t n_col, uint32_t cn, uint32_t *idx_out) {
  uint32_t n = 0u;
  for (uint32_t nt0 = 0; nt0 < n_col; nt0 += cn) {
    const uint32_t c = (n_col - nt0 < cn) ? (n_col - nt0) : cn;
    idx_out[n++] = hexkl_moe_push_weight_chunk(vtcm_base, dst_off, h, k_tiles,
                                               n_col, nt0, c);
  }
  return n;
}

int hexkl_conv_block_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                         uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                         uint32_t K, uint32_t C, uint32_t N_out, uint32_t h_a,
                         uint32_t h_b, uint32_t h_c, uint32_t h_out,
                         const float *conv_w, const float *act_f32,
                         float *out_f32, float *state_f32,
                         hvx_worker_pool *pool, hexkl_moe_scratch *scratch) {
  if (!tbl || !vtcm_base || !conv_w || !act_f32 || !out_f32 || !state_f32 ||
      !scratch || M == 0u) {
    return AEE_EBADPARM;
  }
  const uint32_t arena = vtcm_size < config_off ? vtcm_size : config_off;
  hexkl_conv_block_layout L;
  int rc = hexkl_conv_block_layout_get(K, C, N_out, arena, &L);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  const hexkl_weight_u8i4 *wa = cb_weight(tbl, h_a, K, C);
  const hexkl_weight_u8i4 *wb = cb_weight(tbl, h_b, K, C);
  const hexkl_weight_u8i4 *wc = cb_weight(tbl, h_c, K, C);
  const hexkl_weight_u8i4 *wo = cb_weight(tbl, h_out, C, N_out);
  if (!wa || !wb || !wc || !wo) {
    return AEE_EBADPARM;
  }
  const hexkl_acc_layout *acc = hexkl_acc_layout_get(vtcm_base, L.result_off);
  if (!acc->usable) {
    return AEE_EUNSUPPORTED;
  }
  hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE] = acc->row_stride;

  const uint32_t BR = HEXKL_HMX_INT8_BLOCK_N_ROW;
  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t c_ntiles = C / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t c_ktiles = C / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t o_ntiles = N_out / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t half = L.acc_tiles / 2u; /* a/c pairs per batch */
  const uint32_t m_pad = ROUND_UP_U32(M, BR);
  if ((c_ntiles + half - 1u) / half > CB_MAX_CHUNKS ||
      (c_ntiles + L.acc_tiles - 1u) / L.acc_tiles > CB_MAX_CHUNKS ||
      (o_ntiles + L.acc_tiles - 1u) / L.acc_tiles > CB_MAX_CHUNKS) {
    return AEE_EUNSUPPORTED;
  }

  /* Heap scratch, session-lifetime (hexkl_moe_scratch): the cached copy of
     the activation -- m_pad rows, the padding rows zero, since the pack
     reads whole 64-row blocks -- its per-row quant params, its AH tiles,
     g, the cached output, and the pack's done bytes. */
  const size_t sz_mpad_f32 = sizeof(float) * m_pad;
  const size_t sz_act_c = sizeof(float) * (size_t)m_pad * K;
  const size_t sz_act_ah = (size_t)m_pad * K;
  const size_t sz_g = sizeof(float) * (size_t)M * C;
  const size_t sz_out_c = sizeof(float) * (size_t)M * N_out;
  const size_t sz_pack_done = m_pad / HEXKL_MOE_PACK_UNIT_ROWS + 1u;
  const size_t sz_rq = sizeof(float) * BR;
  const size_t need = 2u * ROUND_UP_SZ(sz_mpad_f32, CB_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_act_c, CB_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_act_ah, CB_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_g, CB_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_out_c, CB_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_pack_done, CB_SCRATCH_ALIGN) +
                      4u * ROUND_UP_SZ(sz_rq, CB_SCRATCH_ALIGN);
  uint64_t p0 = 0;
  HEXKL_PROBE_T0(p0);
  rc = hexkl_moe_scratch_reserve(scratch, need);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ALLOC, p0);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  uint8_t *cur = scratch->base;
  float *scale_all = (float *)hexkl_moe_carve(&cur, sz_mpad_f32);
  int32_t *zp_all = (int32_t *)hexkl_moe_carve(&cur, sz_mpad_f32);
  float *act_c = (float *)hexkl_moe_carve(&cur, sz_act_c);
  uint8_t *act_ah = (uint8_t *)hexkl_moe_carve(&cur, sz_act_ah);
  float *g = (float *)hexkl_moe_carve(&cur, sz_g);
  float *out_c = (float *)hexkl_moe_carve(&cur, sz_out_c);
  uint8_t *pack_done = (uint8_t *)hexkl_moe_carve(&cur, sz_pack_done);
  float *rq_scale[2], *zbuf[2];
  int32_t *rq_zp[2];
  uint8_t *midbuf[2];
  for (uint32_t i = 0; i < 2u; ++i) {
    rq_scale[i] = (float *)hexkl_moe_carve(&cur, sz_rq);
    rq_zp[i] = (int32_t *)hexkl_moe_carve(&cur, sz_rq);
    zbuf[i] = (float *)(vtcm_base + L.z_off + i * BR * C * 4u);
    midbuf[i] =
      vtcm_base + L.mid_off + i * c_ktiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  }
  const float *const conv_w_v = (const float *)(vtcm_base + L.conv_w_off);

  hvx_bg_job pack_job;
  hexkl_moe_pack_ctx pack;
  int pack_submitted = 0;
  hvx_dq_mul_job mul_job[2];
  hvx_dq_tiles_job dq_job[2];
  /* Phase 2's background stage, two blocks in flight: jobs and their done
     bytes live here because a job outlives its submit until the wait
     that retires it. */
  cb_stage_ctx stage_ctx[2];
  hvx_bg_job gate_job[2], rq_job[2];
  uint8_t gate_done[2][CB_GATE_UNITS], rq_done[2][1];
  int stage_submitted[2] = {0, 0};
  uint32_t idx_ac[CB_MAX_CHUNKS], idx_b[CB_MAX_CHUNKS], idx_o[CB_MAX_CHUNKS];
  uint32_t act_idx;
  uint64_t mm_t0 = 0, mm_acc0 = 0, mm_dq0 = 0;

  /* In: the FastRPC buffers are uncached (see hexkl_mm_u8i4_moe.c on why
     each is touched exactly once). conv_w goes to VTCM: phase 2 reads it
     once per row. */
  HEXKL_PROBE_T0(p0);
  hexkl_dma_ring_reset();
  hexkl_moe_dma_copy(act_c, act_f32, sizeof(float) * (size_t)M * K, 0, 0);
  hexkl_moe_dma_copy(vtcm_base + L.conv_w_off, conv_w, 3u * C * sizeof(float),
                     0, 1);
  if (m_pad > M) {
    memset(act_c + (size_t)M * K, 0, sizeof(float) * (size_t)(m_pad - M) * K);
  }
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);

  /* Phase 1's weights go out before the quantization so the scan hides
     them: W_a and W_c in paired chunks -- chunk ci is a's n-tile columns
     [ci*half, +half) and c's opposite them, one index each (the c push's,
     the ring retires in order). */
  uint32_t n_ac = 0u;
  for (uint32_t g0 = 0; g0 < c_ntiles; g0 += half) {
    const uint32_t cn = (c_ntiles - g0 < half) ? (c_ntiles - g0) : half;
    (void)hexkl_moe_push_weight_chunk(vtcm_base, L.w_a_off, wa, k_tiles,
                                      c_ntiles, g0, cn);
    idx_ac[n_ac++] = hexkl_moe_push_weight_chunk(vtcm_base, L.w_b_off, wc,
                                                 k_tiles, c_ntiles, g0, cn);
  }

  HEXKL_PROBE_T0(p0);
  hvx_quant_rows_u8_params(act_c, M, m_pad, K, scale_all, zp_all, pool);
  pack.act_c = act_c;
  pack.slot_row = NULL;
  pack.slot_scale = scale_all;
  pack.slot_zp = zp_all;
  pack.act_ah = act_ah;
  pack.K = K;
  pack_job.func = hexkl_moe_pack_bg_worker;
  pack_job.ctx = &pack;
  pack_job.n_units = m_pad / HEXKL_MOE_PACK_UNIT_ROWS;
  pack_job.done = pack_done;
  hvx_worker_pool_submit_bg(pool, &pack_job);
  pack_submitted = 1;
  hvx_worker_pool_wait_bg(pool, &pack_job, HEXKL_MOE_PACK_UNITS_THROUGH(0u));
  HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
  act_idx =
    hexkl_moe_push_act_block(vtcm_base, L.act_off, act_ah, 0u, K, k_tiles);

  /* ---- phase 1: g = dq(x . W_a) * dq(x . W_c), block by block ---------- */
  for (uint32_t mb = 0; mb < M; mb += BR) {
    const uint32_t m_blk = (M - mb < BR) ? (M - mb) : BR;
    HEXKL_PROBE_COUNT(HEXKL_PROBE_BLOCKS, 1);
    HEXKL_PROBE_T0(p0);
    hexkl_dma_ring_wait(act_idx);
    HEXKL_PROBE_ADD(HEXKL_PROBE_GATHER, p0);

    for (uint32_t g0 = 0, ci = 0; g0 < c_ntiles; g0 += half, ++ci) {
      const uint32_t np = (c_ntiles - g0 < half) ? (c_ntiles - g0) : half;
      const uint32_t stage_off =
        L.result_off + (ci & 1u) * L.acc_tiles * ACC_TILE_BYTES;
      if (mb == 0u) {
        HEXKL_PROBE_T0(p0);
        hexkl_dma_ring_wait(idx_ac[ci]);
        HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);
        if (ci == 0u) {
          hexkl_probe_us[HEXKL_PROBE_DMA_FIRST] =
            hexkl_probe_us[HEXKL_PROBE_DRAIN];
          HEXKL_PROBE_COUNT(HEXKL_PROBE_DMA_FIRST_KB,
                            (2u * np * k_tiles * WEIGHT_TILE_BYTES_U8I4) >> 10);
        }
      }
      /* Slots [0, np) get a's tiles, [np, 2np) c's, at the same columns. */
      HEXKL_MOE_MM_BEGIN();
      for (uint32_t j = 0; j < 2u * np; ++j) {
        const uint32_t col = g0 + ((j < np) ? j : (j - np));
        const uint32_t w_off = (j < np) ? L.w_a_off : L.w_b_off;
        hexkl_micro_hmx_acc_clear_int32();
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
          rc = hexkl_micro_hmx_mm_u8i4(
            vtcm_base, L.act_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
            w_off + (kt * c_ntiles + col) * WEIGHT_TILE_BYTES_U8I4);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        HEXKL_PROBE_T0(p0);
        rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off,
                                            stage_off + j * ACC_TILE_BYTES);
        HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
        if (rc != AEE_SUCCESS) {
          goto out;
        }
      }
      HEXKL_MOE_MM_END();

      /* Retire the epilogue that read this staging buffer's partner
         (batch ci-1), then launch this batch's straight into g. */
      HEXKL_PROBE_T0(p0);
      hvx_worker_pool_wait(pool);
      HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
      {
        hvx_dq_mul_job *jb = &mul_job[ci & 1u];
        jb->tiles_base =
          (const uint8_t *)((const int32_t *)(vtcm_base + stage_off) +
                            acc->base);
        jb->tile_stride = ACC_TILE_BYTES;
        jb->n_pairs = np;
        jb->c0 = g0 * HEXKL_HMX_INT8_BLOCK_N_COL;
        jb->row_stride = acc->row_stride;
        jb->m_count = m_blk;
        jb->act_scale = scale_all + mb;
        jb->act_zp = zp_all + mb;
        jb->colsum_a = wa->colsum_w;
        jb->w_scale_a = wa->w_scale;
        jb->bias_a = wa->bias;
        jb->colsum_b = wc->colsum_w;
        jb->w_scale_b = wc->w_scale;
        jb->bias_b = wc->bias;
        jb->dst = g + (size_t)mb * C;
        jb->dst_stride = C;
        hvx_worker_pool_submit(pool, hvx_dq_mul_worker, jb, np);
      }
    }
    /* The activation slot is free once the last acc_read returned; the
       next block rides under this block's last epilogue. */
    if (mb + BR < M) {
      HEXKL_PROBE_T0(p0);
      hvx_worker_pool_wait_bg(pool, &pack_job,
                              HEXKL_MOE_PACK_UNITS_THROUGH(mb + BR));
      HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
      act_idx = hexkl_moe_push_act_block(vtcm_base, L.act_off, act_ah, mb + BR,
                                         K, k_tiles);
    }
  }
  /* g complete: phase 2 looks two rows back across blocks. */
  HEXKL_PROBE_T0(p0);
  hvx_worker_pool_wait(pool);
  HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);

  /* The conv state the CPU decode path continues from: g's last two rows
     (x_{t-2}, x_{t-1} in causal_conv1d_layer's order), zero-padded when M
     is shorter. 16 KB, so a plain copy into the uncached buffer. */
  if (M >= 2u) {
    memcpy(state_f32, g + (size_t)(M - 2u) * C, 2u * C * sizeof(float));
  } else {
    memset(state_f32, 0, C * sizeof(float));
    memcpy(state_f32 + C, g, C * sizeof(float));
  }

  /* ---- phase 2: out = (dq(x . W_b) * conv1d(g)) . W_out ---------------- */
  /* Both slots are dead (every HMX read of them returned). Block 0's
     activation first, then W_b into A and W_out into B, chunked so the
     first batch of each starts before the rest has landed. */
  act_idx =
    hexkl_moe_push_act_block(vtcm_base, L.act_off, act_ah, 0u, K, k_tiles);
  const uint32_t n_b = cb_push_chunks(vtcm_base, L.w_a_off, wb, k_tiles,
                                      c_ntiles, L.acc_tiles, idx_b);
  const uint32_t n_o = cb_push_chunks(vtcm_base, L.w_b_off, wo, c_ktiles,
                                      o_ntiles, L.acc_tiles, idx_o);
  (void)n_b;
  (void)n_o;

  /* The pipeline, one block ahead: block n's b matmul and dequant into
     z[n&1], then its gate and requant submitted to the background lane;
     block n-1's out_proj from mid[(n-1)&1] issued while they run. Every
     buffer has one writer and one reader a block apart:
       z[i]    written by the dequant of block n, read by its gate and
               requant, both retired before block n+2's dequant writes it
       mid[i]  written by block n's requant, read by block n's out_proj
       rq[i]   written by block n's requant, read by block n's out_proj
               epilogues, retired before block n+2's requant */
  for (uint32_t mb = 0, n = 0; mb < M + BR; mb += BR, ++n) {
    const uint32_t cur = n & 1u;
    if (mb < M) {
      const uint32_t m_blk = (M - mb < BR) ? (M - mb) : BR;
      HEXKL_PROBE_T0(p0);
      hexkl_dma_ring_wait(act_idx);
      HEXKL_PROBE_ADD(HEXKL_PROBE_GATHER, p0);

      /* --- b: dequantized into z[cur] --- */
      for (uint32_t nt0 = 0, ci = 0; nt0 < c_ntiles; nt0 += L.acc_tiles, ++ci) {
        const uint32_t nb =
          (c_ntiles - nt0 < L.acc_tiles) ? (c_ntiles - nt0) : L.acc_tiles;
        const uint32_t stage_off =
          L.result_off + (ci & 1u) * L.acc_tiles * ACC_TILE_BYTES;
        if (mb == 0u) {
          HEXKL_PROBE_T0(p0);
          hexkl_dma_ring_wait(idx_b[ci]);
          HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN_DN, p0);
        }
        HEXKL_MOE_MM_BEGIN();
        for (uint32_t j = 0; j < nb; ++j) {
          hexkl_micro_hmx_acc_clear_int32();
          for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            rc = hexkl_micro_hmx_mm_u8i4(
              vtcm_base, L.act_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
              L.w_a_off + (kt * c_ntiles + nt0 + j) * WEIGHT_TILE_BYTES_U8I4);
            if (rc != AEE_SUCCESS) {
              goto out;
            }
          }
          HEXKL_PROBE_T0(p0);
          rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off,
                                              stage_off + j * ACC_TILE_BYTES);
          HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        HEXKL_MOE_MM_END();
        HEXKL_PROBE_T0(p0);
        hvx_worker_pool_wait(pool);
        HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
        {
          hvx_dq_tiles_job *jb = &dq_job[ci & 1u];
          jb->tiles_base =
            (const uint8_t *)((const int32_t *)(vtcm_base + stage_off) +
                              acc->base);
          jb->tile_stride = ACC_TILE_BYTES;
          jb->nt0 = nt0;
          jb->row_stride = acc->row_stride;
          jb->m_count = m_blk;
          jb->act_scale = scale_all + mb;
          jb->act_zp = zp_all + mb;
          jb->colsum_w = wb->colsum_w;
          jb->w_scale = wb->w_scale;
          jb->bias = wb->bias;
          jb->dst_a = zbuf[cur];
          jb->dst_b = NULL;
          jb->split = C;
          jb->dst_stride = C;
          jb->n_tiles = nb;
          hvx_worker_pool_submit(pool, hvx_dq_tiles_worker, jb, nb);
        }
      }
      HEXKL_PROBE_T0(p0);
      hvx_worker_pool_wait(pool);
      HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);

      /* The activation slot is dead until the next b matmul; the next
         block's DMA rides under this block's stage and the previous
         block's out_proj. */
      if (mb + BR < M) {
        act_idx = hexkl_moe_push_act_block(vtcm_base, L.act_off, act_ah,
                                           mb + BR, K, k_tiles);
      }

      /* This block's gate and requant, to the background lane. */
      {
        cb_stage_ctx *c = &stage_ctx[cur];
        c->z = zbuf[cur];
        c->g = g;
        c->conv_w = conv_w_v;
        c->mid = midbuf[cur];
        c->rq_scale = rq_scale[cur];
        c->rq_zp = rq_zp[cur];
        c->mb = mb;
        c->m_blk = m_blk;
        c->C = C;
        gate_job[cur].func = cb_gate_unit;
        gate_job[cur].ctx = c;
        gate_job[cur].n_units = CB_GATE_UNITS;
        gate_job[cur].done = gate_done[cur];
        rq_job[cur].func = cb_requant_unit;
        rq_job[cur].ctx = c;
        rq_job[cur].n_units = 1u;
        rq_job[cur].done = rq_done[cur];
        hvx_worker_pool_submit_bg(pool, &gate_job[cur]);
        hvx_worker_pool_submit_bg(pool, &rq_job[cur]);
        stage_submitted[cur] = 1;
      }
    }

    /* --- out_proj of the PREVIOUS block, from mid[prev] --- */
    if (mb >= BR) {
      const uint32_t pmb = mb - BR;
      const uint32_t prev = cur ^ 1u;
      const uint32_t pm_blk = (M - pmb < BR) ? (M - pmb) : BR;
      /* Its requant is usually long done; what this wait reads is the
         exposure, filed under REQUANT as the synchronous path's was. */
      HEXKL_PROBE_T0(p0);
      hvx_worker_pool_wait_bg(pool, &rq_job[prev], UINT32_MAX);
      HEXKL_PROBE_ADD(HEXKL_PROBE_REQUANT, p0);
      for (uint32_t nt0 = 0, ci = 0; nt0 < o_ntiles; nt0 += L.acc_tiles, ++ci) {
        const uint32_t nb =
          (o_ntiles - nt0 < L.acc_tiles) ? (o_ntiles - nt0) : L.acc_tiles;
        const uint32_t stage_off =
          L.result_off + (ci & 1u) * L.acc_tiles * ACC_TILE_BYTES;
        if (pmb == 0u) {
          HEXKL_PROBE_T0(p0);
          hexkl_dma_ring_wait(idx_o[ci]);
          HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN_DN, p0);
        }
        HEXKL_MOE_MM_BEGIN();
        for (uint32_t j = 0; j < nb; ++j) {
          hexkl_micro_hmx_acc_clear_int32();
          for (uint32_t kt = 0; kt < c_ktiles; ++kt) {
            rc = hexkl_micro_hmx_mm_u8i4(vtcm_base,
                                         (uint32_t)(midbuf[prev] - vtcm_base) +
                                           kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
                                         L.w_b_off + (kt * o_ntiles + nt0 + j) *
                                                       WEIGHT_TILE_BYTES_U8I4);
            if (rc != AEE_SUCCESS) {
              goto out;
            }
          }
          HEXKL_PROBE_T0(p0);
          rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off,
                                              stage_off + j * ACC_TILE_BYTES);
          HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        HEXKL_MOE_MM_END();
        HEXKL_PROBE_T0(p0);
        hvx_worker_pool_wait(pool);
        HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
        {
          hvx_dq_tiles_job *jb = &dq_job[ci & 1u];
          jb->tiles_base =
            (const uint8_t *)((const int32_t *)(vtcm_base + stage_off) +
                              acc->base);
          jb->tile_stride = ACC_TILE_BYTES;
          jb->nt0 = nt0;
          jb->row_stride = acc->row_stride;
          jb->m_count = pm_blk;
          jb->act_scale = rq_scale[prev];
          jb->act_zp = rq_zp[prev];
          jb->colsum_w = wo->colsum_w;
          jb->w_scale = wo->w_scale;
          jb->bias = wo->bias;
          jb->dst_a = out_c + (size_t)pmb * N_out;
          jb->dst_b = NULL;
          jb->split = N_out;
          jb->dst_stride = N_out;
          jb->n_tiles = nb;
          hvx_worker_pool_submit(pool, hvx_dq_tiles_worker, jb, nb);
        }
      }
      /* The last epilogue reads rq[prev], which block n+1's requant
         rewrites; retire it here. */
      HEXKL_PROBE_T0(p0);
      hvx_worker_pool_wait(pool);
      HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
    }
  }

  HEXKL_PROBE_T0(p0);
  hexkl_moe_dma_copy(out_f32, out_c, sizeof(float) * (size_t)M * N_out, 0, 0);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);

out:
  /* An error path may leave jobs in flight over VTCM this call owns. */
  hvx_worker_pool_wait(pool);
  if (pack_submitted) {
    hvx_worker_pool_wait_bg(pool, &pack_job, UINT32_MAX);
  }
  for (uint32_t i = 0; i < 2u; ++i) {
    if (stage_submitted[i]) {
      hvx_worker_pool_wait_bg(pool, &rq_job[i], UINT32_MAX);
    }
  }
  return rc;
}
