// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_dequant_i32.c
 * @date   03 Aug 2026
 * @brief  int32 accumulator to f32 dequantization for the A8W4 path
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_dequant_i32.h"

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128u
/** @brief f32 or int32 lanes per HVX vector. */
#define LANES (VLEN / 4u)

void hvx_dequant_i32_to_f32(const int32_t *acc, uint32_t m_valid,
                            uint32_t m_pad, uint32_t n, const float *act_scale,
                            const int32_t *act_zp, const int32_t *colsum_w,
                            const float *w_scale, const float *bias,
                            float *out) {
  (void)m_pad;

  for (uint32_t m = 0; m < m_valid; ++m) {
    const int32_t *arow = acc + (size_t)m * n;
    float *orow = out + (size_t)m * n;
    const float s = act_scale[m];
    const int32_t z = act_zp[m];

    const uint32_t n_vec = n / LANES;
    const HVX_Vector vs = hvx_splat_sf(s);
    const HVX_Vector vzf = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(z));

    const HVX_UVector *vacc = (const HVX_UVector *)arow;
    const HVX_UVector *vcs = (const HVX_UVector *)colsum_w;
    const HVX_UVector *vws = (const HVX_UVector *)w_scale;
    const HVX_UVector *vb = (const HVX_UVector *)bias;
    HVX_UVector *vout = (HVX_UVector *)orow;

    for (uint32_t v = 0; v < n_vec; ++v) {
      const HVX_Vector af = Q6_Vsf_equals_Vw(vacc[v]);
      const HVX_Vector csf = Q6_Vsf_equals_Vw(vcs[v]);
      const HVX_Vector corrected =
        Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vzf, csf));
      const HVX_Vector scaled =
        Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(corrected, vs), vws[v]);
      vout[v] = Q6_Vsf_vadd_VsfVsf(scaled, vb[v]);
    }

    for (uint32_t j = n_vec * LANES; j < n; ++j) {
      const int32_t corrected = arow[j] - z * colsum_w[j];
      orow[j] = (float)corrected * s * w_scale[j] + bias[j];
    }
  }
}

/** @brief One tile row is exactly one HVX vector, which is what kills the
 *         tail handling. HEXKL_HMX_INT8_BLOCK_N_COL is 32. */
_Static_assert(LANES == 32u, "tile kernel assumes 32 columns per HVX vector");

void hvx_dequant_tile_i32_to_f32(const int32_t *tile, uint32_t n_rows,
                                 const float *act_scale, const int32_t *act_zp,
                                 const int32_t *colsum_w, const float *w_scale,
                                 const float *bias, float *out,
                                 uint32_t out_stride) {
  // Every row of the tile shares these, so they load once. The DDR-side
  // buffers come from FastRPC, whose base alignment is not part of the
  // contract, hence the unaligned loads.
  const HVX_Vector csf = Q6_Vsf_equals_Vw(*(const HVX_UVector *)colsum_w);
  const HVX_Vector vws = *(const HVX_UVector *)w_scale;
  const HVX_Vector vb = *(const HVX_UVector *)bias;

  // VTCM, and unshuf_off inherits result_off's 2048-byte alignment.
  const HVX_Vector *vtile = (const HVX_Vector *)tile;

  for (uint32_t r = 0; r < n_rows; ++r) {
    const HVX_Vector vs = hvx_splat_sf(act_scale[r]);
    const HVX_Vector vzf = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(act_zp[r]));

    const HVX_Vector af = Q6_Vsf_equals_Vw(vtile[r]);
    const HVX_Vector corrected =
      Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vzf, csf));
    const HVX_Vector scaled =
      Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(corrected, vs), vws);
    *(HVX_UVector *)(out + (size_t)r * out_stride) =
      Q6_Vsf_vadd_VsfVsf(scaled, vb);
  }
}
