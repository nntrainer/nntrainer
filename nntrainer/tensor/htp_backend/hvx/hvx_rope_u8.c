// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_rope_u8.c
 * @brief  HVX NeoX RoPE with a uint8 activation contract.
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs.
 */

#include <stdint.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_rope_u8.h"

#define ROPE_LANES 32u

/**
 * The input bytes are staged into unaligned HVX vectors before arithmetic.
 * This keeps the FastRPC alignment contract honest while using HVX for the
 * complete rotate/requantize arithmetic. The scalar staging is a known
 * ceiling see the ponytail note on hvx_rope_u8_rows in the header.
 *
 * cos and sin arrive as uint8 and are dequantized here with their own
 * per-tensor (scale, zero-point). vinv = s_in / s_out with cos/sin
 * already real-valued, no 32768 factor remains (the q15 path divided it
 * out in inv_out). */
static void rope_block(const uint8_t *row, uint8_t *out, uint32_t offset,
                       uint32_t half, const uint8_t *cos_u8,
                       const uint8_t *sin_u8, float cos_scale, int32_t cos_zp,
                       float sin_scale, int32_t sin_zp, int32_t zp, float vinv,
                       int32_t zp_out, uint32_t *n_saturated) {
  float a[ROPE_LANES], b[ROPE_LANES], c[ROPE_LANES], s[ROPE_LANES];
  int32_t q0[ROPE_LANES], q1[ROPE_LANES];
  const uint32_t n = (half - offset < ROPE_LANES) ? half - offset : ROPE_LANES;

  for (uint32_t i = 0; i < n; ++i) {
    a[i] = (float)((int32_t)row[offset + i] - zp);
    b[i] = (float)((int32_t)row[offset + half + i] - zp);
    c[i] = ((float)cos_u8[offset + i] - (float)cos_zp) * cos_scale;
    s[i] = ((float)sin_u8[offset + i] - (float)sin_zp) * sin_scale;
  }
  for (uint32_t i = n; i < ROPE_LANES; ++i) {
    a[i] = b[i] = c[i] = s[i] = 0.0f;
  }

  const HVX_Vector va = *(const HVX_UVector *)a;
  const HVX_Vector vb = *(const HVX_UVector *)b;
  const HVX_Vector vc = *(const HVX_UVector *)c;
  const HVX_Vector vs = *(const HVX_UVector *)s;
  const HVX_Vector vac = Q6_Vqf32_vmpy_VsfVsf(va, vc);
  const HVX_Vector vbs = Q6_Vqf32_vmpy_VsfVsf(vb, vs);
  const HVX_Vector vas = Q6_Vqf32_vmpy_VsfVsf(va, vs);
  const HVX_Vector vbc = Q6_Vqf32_vmpy_VsfVsf(vb, vc);
  const HVX_Vector vt0 =
    Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vac, vbs));
  const HVX_Vector vt1 =
    Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vas, vbc));
  const HVX_Vector vscale = hvx_splat_sf(vinv);
  const HVX_Vector vq0 = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(vt0, vscale));
  const HVX_Vector vq1 = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(vt1, vscale));
  *(HVX_UVector *)q0 = vq0;
  *(HVX_UVector *)q1 = vq1;

  for (uint32_t i = 0; i < n; ++i) {
    const int32_t r0 = q0[i] + zp_out;
    const int32_t r1 = q1[i] + zp_out;
    if (r0 < 0 || r0 > 255) {
      ++*n_saturated;
    }
    if (r1 < 0 || r1 > 255) {
      ++*n_saturated;
    }
    out[offset + i] = (uint8_t)(r0 < 0 ? 0 : r0 > 255 ? 255 : r0);
    out[offset + half + i] = (uint8_t)(r1 < 0 ? 0 : r1 > 255 ? 255 : r1);
  }
}

void hvx_rope_u8_rows(const uint8_t *x, uint8_t *y, uint32_t m_first,
                      uint32_t m_last, uint32_t width, uint32_t dim,
                      const float *s_in, uint32_t s_in_stride,
                      const int32_t *zp_in, uint32_t zp_in_stride,
                      const uint8_t *cos_u8, const uint8_t *sin_u8,
                      float cos_scale, int32_t cos_zp, float sin_scale,
                      int32_t sin_zp, float s_out, int32_t zp_out,
                      uint32_t *n_saturated) {
  const uint32_t half = dim / 2u;
  const float inv_out = 1.0f / s_out;
  /** Whole dim-sized segments only: the last partial segment, if any, is
  left to the single post-loop memcpy below rather than being written
  from inside the w-loop, which is what made the previous version's
  tail copy index past the row (out + w + dim at the final w). */
  const uint32_t rotated = width - (width % dim);

  for (uint32_t m = m_first; m < m_last; ++m) {
    const uint8_t *row = x + (size_t)m * width;
    uint8_t *out = y + (size_t)m * width;
    const uint8_t *cq = cos_u8 + (size_t)m * half;
    const uint8_t *sq = sin_u8 + (size_t)m * half;
    const int32_t zp = zp_in[(size_t)m * zp_in_stride];
    const float vinv = s_in[(size_t)m * s_in_stride] * inv_out;

    for (uint32_t w = 0; w < rotated; w += dim) {
      for (uint32_t offset = 0; offset < half; offset += ROPE_LANES) {
        rope_block(row + w, out + w, offset, half, cq, sq, cos_scale, cos_zp,
                   sin_scale, sin_zp, zp, vinv, zp_out, n_saturated);
      }
    }
    if (out != row && rotated < width) {
      memcpy(out + rotated, row + rotated, width - rotated);
    }
  }
}
