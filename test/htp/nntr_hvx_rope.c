// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   nntr_hvx_rope.c
 * @brief  FastRPC entry point for the HVX uint8 RoPE kernel.
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs.
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <math.h>
#include <remote.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_rope_u8.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

static uint8_t rope_real_to_u8(float value, float scale, int32_t zp) {
  int32_t v = (int32_t)nearbyintf(value / scale) + zp;
  return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

static void rope_cache_free(nntr_hvx_session *s) {
  free(s->rope_cos_u8);
  free(s->rope_sin_u8);
  free(s->rope_thetas);
  s->rope_cos_u8 = NULL;
  s->rope_sin_u8 = NULL;
  s->rope_thetas = NULL;
  s->rope_cache_positions = 0u;
  s->rope_cache_half = 0u;
  s->rope_cache_attention_scaling = 0.0f;
  s->rope_cache_cos_scale = 0.0f;
  s->rope_cache_cos_zp = 0;
  s->rope_cache_sin_scale = 0.0f;
  s->rope_cache_sin_zp = 0;
}

int nntr_hvx_rope_cache_clear(remote_handle64 handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  rope_cache_free(s);
  return AEE_SUCCESS;
}

int nntr_hvx_rope_cache_init(remote_handle64 handle, uint32 n_positions,
                             const float *thetas, int thetasLen,
                             float attention_scaling, float cos_scale,
                             int32 cos_zp, float sin_scale, int32 sin_zp,
                             uint32 *generation) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !generation || n_positions == 0u || n_positions > 4096u ||
      thetasLen <= 0 || (uint32)thetasLen > UINT32_MAX / n_positions ||
      cos_scale <= 0.0f || sin_scale <= 0.0f) {
    FARF(ERROR, "rope_cache_init: bad shape or scale");
    return AEE_EBADPARM;
  }
  const uint32 half = (uint32)thetasLen;

  if (s->rope_cos_u8 && s->rope_cache_positions == n_positions &&
      s->rope_cache_half == half &&
      s->rope_cache_attention_scaling == attention_scaling &&
      s->rope_cache_cos_scale == cos_scale && s->rope_cache_cos_zp == cos_zp &&
      s->rope_cache_sin_scale == sin_scale && s->rope_cache_sin_zp == sin_zp &&
      memcmp(s->rope_thetas, thetas, (size_t)half * sizeof(float)) == 0) {
    *generation = s->rope_cache_generation;
    return AEE_SUCCESS;
  }

  const size_t count = (size_t)n_positions * half;
  uint8_t *cos_u8 = (uint8_t *)malloc(count * sizeof(uint8_t));
  uint8_t *sin_u8 = (uint8_t *)malloc(count * sizeof(uint8_t));
  float *thetas_copy = (float *)malloc((size_t)half * sizeof(float));
  if (!cos_u8 || !sin_u8 || !thetas_copy) {
    free(cos_u8);
    free(sin_u8);
    free(thetas_copy);
    return AEE_ENOMEMORY;
  }
  memcpy(thetas_copy, thetas, (size_t)half * sizeof(float));

  for (uint32 pos = 0u; pos < n_positions; ++pos) {
    for (uint32 k = 0u; k < half; ++k) {
      const float angle = (float)pos * thetas[k];
      const size_t i = (size_t)pos * half + k;
      cos_u8[i] =
        rope_real_to_u8(cosf(angle) * attention_scaling, cos_scale, cos_zp);
      sin_u8[i] =
        rope_real_to_u8(sinf(angle) * attention_scaling, sin_scale, sin_zp);
    }
  }
  rope_cache_free(s);
  s->rope_cos_u8 = cos_u8;
  s->rope_sin_u8 = sin_u8;
  s->rope_thetas = thetas_copy;
  s->rope_cache_positions = n_positions;
  s->rope_cache_half = half;
  s->rope_cache_attention_scaling = attention_scaling;
  s->rope_cache_cos_scale = cos_scale;
  s->rope_cache_cos_zp = cos_zp;
  s->rope_cache_sin_scale = sin_scale;
  s->rope_cache_sin_zp = sin_zp;
  ++s->rope_cache_generation;
  if (s->rope_cache_generation == 0u) {
    s->rope_cache_generation = 1u;
  }
  *generation = s->rope_cache_generation;
  return AEE_SUCCESS;
}

int nntr_hvx_rope_u8(remote_handle64 handle, uint32 n_rows, uint32 width,
                     uint32 dim, const uint8 *x, int xLen, const float *s_in,
                     int s_inLen, const int32 *zp_in, int zp_inLen,
                     const uint8 *cos_u8, int cos_u8Len, const uint8 *sin_u8,
                     int sin_u8Len, float cos_scale, int32 cos_zp,
                     float sin_scale, int32 sin_zp, float s_out, int32 zp_out,
                     uint8 *y, int yLen, uint32 *n_saturated) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (n_rows == 0u || n_rows > 4096u || width == 0u || dim == 0u ||
      (dim & 1u) != 0u || dim > width || s_out <= 0.0f || n_saturated == NULL ||
      width > UINT32_MAX / n_rows || cos_scale <= 0.0f || sin_scale <= 0.0f) {
    FARF(ERROR, "rope_u8: bad shape or scale");
    return AEE_EBADPARM;
  }
  const uint32 expected_x = n_rows * width;
  const uint32 half = dim / 2u;
  if (half > UINT32_MAX / n_rows) {
    FARF(ERROR, "rope_u8: table length overflow");
    return AEE_EBADPARM;
  }
  const uint32 expected_table = n_rows * half;
  if ((uint32)xLen != expected_x || (uint32)yLen != expected_x ||
      ((uint32)s_inLen != 1u && (uint32)s_inLen != n_rows) ||
      ((uint32)zp_inLen != 1u && (uint32)zp_inLen != n_rows) ||
      (uint32)cos_u8Len != expected_table ||
      (uint32)sin_u8Len != expected_table) {
    FARF(ERROR, "rope_u8: bad buffer lengths");
    return AEE_EBADPARM;
  }

  *n_saturated = 0u;
  hvx_rope_u8_rows(x, y, 0u, n_rows, width, dim, s_in,
                   (uint32)s_inLen == 1u ? 0u : 1u, zp_in,
                   (uint32)zp_inLen == 1u ? 0u : 1u, cos_u8, sin_u8, cos_scale,
                   cos_zp, sin_scale, sin_zp, s_out, zp_out, n_saturated);
  return AEE_SUCCESS;
}

int nntr_hvx_rope_u8_cached(remote_handle64 handle, uint32 n_rows, uint32 width,
                            uint32 dim, uint32 position_start, const uint8 *x,
                            int xLen, const float *s_in, int s_inLen,
                            const int32 *zp_in, int zp_inLen, float s_out,
                            int32 zp_out, uint8 *y, int yLen,
                            uint32 *n_saturated) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !s->rope_cos_u8 || n_rows == 0u || width == 0u || dim == 0u ||
      (dim & 1u) != 0u || dim > width || s_out <= 0.0f ||
      position_start > s->rope_cache_positions ||
      n_rows > s->rope_cache_positions - position_start) {
    return AEE_EBADPARM;
  }
  const uint32 half = dim / 2u;
  if (width > UINT32_MAX / n_rows || s->rope_cache_half != half ||
      !n_saturated) {
    return AEE_EBADPARM;
  }
  const uint32 expected = n_rows * width;
  if ((uint32)xLen != expected || (uint32)yLen != expected ||
      ((uint32)s_inLen != 1u && (uint32)s_inLen != n_rows) ||
      ((uint32)zp_inLen != 1u && (uint32)zp_inLen != n_rows)) {
    return AEE_EBADPARM;
  }
  *n_saturated = 0u;
  hvx_rope_u8_rows(
    x, y, 0u, n_rows, width, dim, s_in, (uint32)s_inLen == 1u ? 0u : 1u, zp_in,
    (uint32)zp_inLen == 1u ? 0u : 1u,
    s->rope_cos_u8 + (size_t)position_start * half,
    s->rope_sin_u8 + (size_t)position_start * half, s->rope_cache_cos_scale,
    s->rope_cache_cos_zp, s->rope_cache_sin_scale, s->rope_cache_sin_zp, s_out,
    zp_out, n_saturated);
  return AEE_SUCCESS;
}
