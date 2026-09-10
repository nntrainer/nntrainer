// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_quant_u8.c
 * @date   03 Aug 2026
 * @brief  Per-row asymmetric uint8 dynamic activation quantization
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_quant_u8.h"

/** @brief HMX activation tile geometry, mirrored from hexkl_micro.h. */
#define TILE_ROW 64u
#define TILE_INNER 32u
#define ACT_TILE_BYTES 2048u
/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128u
/** @brief f32 lanes per HVX vector. */
#define FLANES (VLEN / 4u)

/** @brief One row of K1: scale[m]/zp[m] from x's m-th row. Independent of
 *         every other row, which is what makes splitting this by row
 *         range across worker threads safe. */
static void quant_row_params_one(const float *x, uint32_t m, uint32_t k,
                                 float *scale, int32_t *zp) {
  const float *row = x + (size_t)m * k;
  const uint32_t n_vec = k / FLANES;
  float min0 = row[0];
  float max0 = row[0];

  if (n_vec > 0) {
    // FastRPC buffers carry no vector alignment guarantee, so the
    // unaligned vector type is what keeps this from faulting.
    const HVX_UVector *vrow = (const HVX_UVector *)row;
    HVX_Vector vmin = vrow[0];
    HVX_Vector vmax = vrow[0];
    for (uint32_t i = 1; i < n_vec; ++i) {
      vmin = Q6_Vsf_vmin_VsfVsf(vmin, vrow[i]);
      vmax = Q6_Vsf_vmax_VsfVsf(vmax, vrow[i]);
    }
    // Fold 32 lanes down to 1 by rotating half the remaining width each
    // step: 64, 32, 16, 8, 4 bytes.
    for (uint32_t rot = VLEN / 2u; rot >= 4u; rot >>= 1) {
      vmin = Q6_Vsf_vmin_VsfVsf(vmin, Q6_V_vror_VR(vmin, (int)rot));
      vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_V_vror_VR(vmax, (int)rot));
    }
    float lane_min[FLANES];
    float lane_max[FLANES];
    *(HVX_UVector *)lane_min = vmin;
    *(HVX_UVector *)lane_max = vmax;
    min0 = lane_min[0];
    max0 = lane_max[0];
  }

  for (uint32_t i = n_vec * FLANES; i < k; ++i) {
    if (row[i] < min0) {
      min0 = row[i];
    }
    if (row[i] > max0) {
      max0 = row[i];
    }
  }
  const float rmin = min0 < 0.0f ? min0 : 0.0f;
  const float rmax = max0 > 0.0f ? max0 : 0.0f;
  if (rmin == rmax) {
    return; /* leaves scale 1, zp 0 */
  }
  const float s = (rmax - rmin) / 255.0f;
  /** nearbyintf, not roundf: the vectorized path rounds to nearest even
     and the host reference is written to match. */
  int32_t z = (int32_t)nearbyintf(-rmin / s);
  if (z < 0) {
    z = 0;
  }
  if (z > 255) {
    z = 255;
  }
  scale[m] = s;
  zp[m] = z;
}

typedef struct {
  const float *x;
  uint32_t m_valid, k;
  float *scale;
  int32_t *zp;
} quant_rows_ctx;

static void quant_rows_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  quant_rows_ctx *ctx = (quant_rows_ctx *)ctx_;
  const uint32_t lo = (uint32_t)((uint64_t)ctx->m_valid * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)ctx->m_valid * (i + 1) / n_threads);
  for (uint32_t m = lo; m < hi; ++m) {
    quant_row_params_one(ctx->x, m, ctx->k, ctx->scale, ctx->zp);
  }
}

void hvx_quant_rows_u8_params(const float *x, uint32_t m_valid, uint32_t m_pad,
                              uint32_t k, float *scale, int32_t *zp,
                              hvx_worker_pool *pool) {
  for (uint32_t m = 0; m < m_pad; ++m) {
    scale[m] = 1.0f;
    zp[m] = 0;
  }

  quant_rows_ctx ctx = {x, m_valid, k, scale, zp};
  hvx_worker_pool_run(pool, quant_rows_worker, &ctx, m_valid);
}

/**
 * @brief One row's worth of K2, unvectorized -- used only for the 0-3 rows
 *        left over when m_valid is not a multiple of 4.
 */
static void quant_pack_row_scalar(const float *row, uint32_t n_ktiles,
                                  HVX_Vector vinv, HVX_Vector vz,
                                  uint8_t *dst_tile0, uint32_t tile_stride) {
  const HVX_Vector vlo = Q6_V_vzero();
  const HVX_Vector vhi = Q6_V_vsplat_R(255);
  for (uint32_t kt = 0; kt < n_ktiles; ++kt) {
    const HVX_UVector *vin = (const HVX_UVector *)(row + kt * TILE_INNER);
    HVX_Vector vq = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(vin[0], vinv));
    vq = Q6_Vw_vadd_VwVw(vq, vz);
    vq = Q6_Vw_vmax_VwVw(vq, vlo);
    vq = Q6_Vw_vmin_VwVw(vq, vhi);

    // Only the low 32 lanes carry data, so stage the vector and copy the
    // 32 bytes we want. A packed store would need three quarters of the
    // vector to be live.
    int32_t lane[FLANES];
    *(HVX_UVector *)lane = vq;
    uint8_t *dst = dst_tile0 + (size_t)kt * tile_stride;
    for (uint32_t c = 0; c < TILE_INNER; ++c) {
      dst[c] = (uint8_t)lane[c];
    }
  }
}

typedef struct {
  const float *x;
  uint32_t m_vec_end, k, n_ktiles;
  const HVX_UVector *vinv; /**< m_vec_end entries, precomputed once --
                               malloc'd, so unaligned: see the two heap
                               HVX_UVector arrays below. */
  const HVX_UVector *vz;   /**< m_vec_end entries, precomputed once */
  uint8_t *out_ah;
} quant_pack_ctx;

/**
 * @brief One k-tile's worth of K2, every row-group in it.
 *
 * A k-tile's destination bytes ((rb*n_ktiles+kt)*ACT_TILE_BYTES + r0*32,
 * for every rb/r0) never overlap another k-tile's, so splitting by k-tile
 * (not by row) is what lets workers write without touching each other's
 * memory -- and it keeps every row-group's four rows on one worker, which
 * is what the vectorized store below needs.
 */
static void quant_pack_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  quant_pack_ctx *ctx = (quant_pack_ctx *)ctx_;
  const uint32_t lo = (uint32_t)((uint64_t)ctx->n_ktiles * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)ctx->n_ktiles * (i + 1) / n_threads);

  for (uint32_t kt = lo; kt < hi; ++kt) {
    for (uint32_t m = 0; m < ctx->m_vec_end; m += 4u) {
      const uint32_t rb = m / TILE_ROW;
      const uint32_t r0 = m % TILE_ROW; // multiple of 4; group never wraps rb
      const HVX_UVector *vin0 =
        (const HVX_UVector *)(ctx->x + (size_t)(m + 0) * ctx->k +
                              kt * TILE_INNER);
      const HVX_UVector *vin1 =
        (const HVX_UVector *)(ctx->x + (size_t)(m + 1) * ctx->k +
                              kt * TILE_INNER);
      const HVX_UVector *vin2 =
        (const HVX_UVector *)(ctx->x + (size_t)(m + 2) * ctx->k +
                              kt * TILE_INNER);
      const HVX_UVector *vin3 =
        (const HVX_UVector *)(ctx->x + (size_t)(m + 3) * ctx->k +
                              kt * TILE_INNER);

      const HVX_Vector vq0 = Q6_Vw_vadd_VwVw(
        hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(vin0[0], ctx->vinv[m + 0])),
        ctx->vz[m + 0]);
      const HVX_Vector vq1 = Q6_Vw_vadd_VwVw(
        hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(vin1[0], ctx->vinv[m + 1])),
        ctx->vz[m + 1]);
      const HVX_Vector vq2 = Q6_Vw_vadd_VwVw(
        hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(vin2[0], ctx->vinv[m + 2])),
        ctx->vz[m + 2]);
      const HVX_Vector vq3 = Q6_Vw_vadd_VwVw(
        hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(vin3[0], ctx->vinv[m + 3])),
        ctx->vz[m + 3]);

      // vpack(Vu, Vv):sat places Vv's (narrowed, saturated) elements in the
      // low half of the result and Vu's in the high half, in order --
      // same convention as vpacke/vpacko (V79 HVX Programmer Reference
      // Manual, "Pack"). Vv first each time keeps row order intact:
      // vh01 = [row0(32), row1(32)], vh23 = [row2(32), row3(32)],
      // vb = [row0, row1, row2, row3] (128 bytes).
      const HVX_Vector vh01 = Q6_Vh_vpack_VwVw_sat(vq1, vq0);
      const HVX_Vector vh23 = Q6_Vh_vpack_VwVw_sat(vq3, vq2);
      const HVX_Vector vb = Q6_Vub_vpack_VhVh_sat(vh23, vh01);

      uint8_t *dst = ctx->out_ah +
                     (size_t)(rb * ctx->n_ktiles + kt) * ACT_TILE_BYTES +
                     r0 * TILE_INNER;
      *(HVX_UVector *)dst = vb;
    }
  }
}

int hvx_quant_pack_u8_ah(const float *x, uint32_t m_valid, uint32_t m_pad,
                         uint32_t k, const float *scale, const int32_t *zp,
                         uint8_t *out_ah, hvx_worker_pool *pool) {
  const uint32_t n_ktiles = k / TILE_INNER;

  memset(out_ah, 0, (size_t)m_pad * k);

  // Four rows at a time: TILE_ROW (64) is a multiple of 4, so a group of 4
  // consecutive rows never straddles a row-block boundary, and the group's
  // 4*32=128 destination bytes are exactly one HVX vector -- one store
  // replaces the 32 scalar byte stores (per row, per k-tile) the row-at-a-
  // time version needed, measured at 92% of hvx_quant_pack_u8_ah's own time
  // relative to a full layer call (P2_QUANT_AND_WORKER_POOL.md §1).
  //
  // Q6_Vub_vpack_VhVh_sat's saturation to [0,255] is byte-identical to the
  // Vw_vmax/Vw_vmin clamp it replaces for every input, not just the common
  // case: [0,255] sits inside int16's range, so the first narrowing pack
  // (int32->int16) never saturates a value this clamp would have kept, and
  // any int32 value the clamp WOULD have clipped saturates the same way at
  // the int16->uint8 stage (V79 HVX Programmer Reference Manual, "Pack").
  const uint32_t m_vec_end = (m_valid / 4u) * 4u;
  if (m_vec_end > 0) {
    // Splitting the work below by k-tile means every k-tile's worker needs
    // every row's vinv/vz -- computed here, once, instead of once per
    // (k-tile, row) the way splitting by k-tile would otherwise force.
    // HVX_UVector, not HVX_Vector: malloc gives no 128-byte alignment
    // guarantee, and HVX_Vector's aligned load/store faults the moment it
    // isn't (found on-device -- this crashed the DSP process the first
    // time M was big enough to take this path).
    HVX_UVector *vinv = (HVX_UVector *)malloc(sizeof(HVX_UVector) * m_vec_end);
    HVX_UVector *vz = (HVX_UVector *)malloc(sizeof(HVX_UVector) * m_vec_end);
    if (!vinv || !vz) {
      free(vinv);
      free(vz);
      return AEE_ENOMEMORY;
    }
    for (uint32_t m = 0; m < m_vec_end; ++m) {
      vinv[m] = hvx_splat_sf(1.0f / scale[m]);
      vz[m] = Q6_V_vsplat_R(zp[m]);
    }

    quant_pack_ctx ctx = {x, m_vec_end, k, n_ktiles, vinv, vz, out_ah};
    hvx_worker_pool_run(pool, quant_pack_worker, &ctx, n_ktiles);

    free(vinv);
    free(vz);
  }

  // Tail: 0-3 rows that didn't fit a group of 4 (e.g. every row when
  // m_valid == 1, the decode case). Unvectorized, same as before.
  for (uint32_t m = m_vec_end; m < m_valid; ++m) {
    const uint32_t rb = m / TILE_ROW;
    const uint32_t r = m % TILE_ROW;
    uint8_t *dst_tile0 =
      out_ah + (size_t)rb * n_ktiles * ACT_TILE_BYTES + r * TILE_INNER;
    quant_pack_row_scalar(x + (size_t)m * k, n_ktiles,
                          hvx_splat_sf(1.0f / scale[m]), Q6_V_vsplat_R(zp[m]),
                          dst_tile0, ACT_TILE_BYTES);
  }
  return AEE_SUCCESS;
}
