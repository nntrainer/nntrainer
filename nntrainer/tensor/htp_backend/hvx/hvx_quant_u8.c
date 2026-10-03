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
#include <string.h>

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

void hvx_quant_rows_u8_params(const float *x, uint32_t m_valid, uint32_t m_pad,
                              uint32_t k, float *scale, int32_t *zp) {
  for (uint32_t m = 0; m < m_pad; ++m) {
    scale[m] = 1.0f;
    zp[m] = 0;
  }

  for (uint32_t m = 0; m < m_valid; ++m) {
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
      continue; /* leaves scale 1, zp 0 */
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
}

void hvx_quant_pack_u8_ah(const float *x, uint32_t m_valid, uint32_t m_pad,
                          uint32_t k, const float *scale, const int32_t *zp,
                          uint8_t *out_ah) {
  const uint32_t n_ktiles = k / TILE_INNER;

  memset(out_ah, 0, (size_t)m_pad * k);

  for (uint32_t m = 0; m < m_valid; ++m) {
    const float *row = x + (size_t)m * k;
    const float inv_s = 1.0f / scale[m];
    const int32_t z = zp[m];
    const uint32_t rb = m / TILE_ROW;
    const uint32_t r = m % TILE_ROW;

    const HVX_Vector vinv = hvx_splat_sf(inv_s);
    const HVX_Vector vz = Q6_V_vsplat_R(z);
    const HVX_Vector vlo = Q6_V_vzero();
    const HVX_Vector vhi = Q6_V_vsplat_R(255);

    for (uint32_t kt = 0; kt < n_ktiles; ++kt) {
      const HVX_UVector *vin = (const HVX_UVector *)(row + kt * TILE_INNER);
      // 32 f32 lanes is exactly one vector, and TILE_INNER is 32.
      HVX_Vector vq = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(vin[0], vinv));
      vq = Q6_Vw_vadd_VwVw(vq, vz);
      vq = Q6_Vw_vmax_VwVw(vq, vlo);
      vq = Q6_Vw_vmin_VwVw(vq, vhi);

      // Only the low 32 lanes carry data, so stage the vector and copy the
      // 32 bytes we want. A packed store would need three quarters of the
      // vector to be live.
      int32_t lane[FLANES];
      *(HVX_UVector *)lane = vq;
      uint8_t *dst =
        out_ah + (size_t)(rb * n_ktiles + kt) * ACT_TILE_BYTES + r * TILE_INNER;
      for (uint32_t c = 0; c < TILE_INNER; ++c) {
        dst[c] = (uint8_t)lane[c];
      }
    }
  }
}
