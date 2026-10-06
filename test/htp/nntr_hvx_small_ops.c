// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_small_ops.c
 * @date   27 Sep 2026
 * @brief  DSP-side test entries for the M=1 small ops (hvx_m1_ops_f32.c):
 *         RMSNorm / per-head norm, RoPE at head_dim 64, conv1d + gate, the
 *         MoE router (#132)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Each entry validates the FastRPC lengths the kernel cannot see, then
 * runs the kernel on the caller's buffers: no session state, heap or VTCM.
 * A bad shape is AEE_EINVALIDFORMAT; AEE_EBADPARM stays the stale-skel
 * symptom (rule 3), so the gtest can tell the two apart.
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "hvx_m1_ops_f32.h"
#include "hvx_rmsnorm_rows_f32.h"

/** @brief f32 lanes per HVX vector. */
#define LANES 32u
/** @brief RoPE head dimension: two vectors per head. */
#define HEAD_DIM 64u

int nntr_hvx_rmsnorm_det_f32(remote_handle64 handle, uint32 chunk, float eps,
                             const float *x, int xLen, const float *gamma,
                             int gammaLen, float *y, int yLen, float *row_scale,
                             int rsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (xLen <= 0 || chunk == 0u || chunk % LANES != 0u ||
      (chunk & (chunk - 1u)) != 0u || (uint32)xLen % chunk != 0u ||
      (uint32)gammaLen != chunk || yLen != xLen ||
      (uint32)rsLen != (uint32)xLen / chunk) {
    FARF(ERROR,
         "rmsnorm_det_f32: bad shape (chunk=%u x=%d gamma=%d y=%d rs=%d)",
         (unsigned)chunk, xLen, gammaLen, yLen, rsLen);
    return AEE_EINVALIDFORMAT;
  }
  hvx_rmsnorm_f32(x, gamma, y, (uint32_t)xLen, chunk, eps, row_scale);
  return AEE_SUCCESS;
}

int nntr_hvx_rmsnorm_add_f32(remote_handle64 handle, uint32 M, uint32 n,
                             float eps, float scale, const float *gamma,
                             int gammaLen, const float *x, int xLen, float *out,
                             int outLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const uint64_t rows = (uint64_t)M * n;
  if (M == 0u || n == 0u || n % LANES != 0u || rows > 0x7FFFFFFFu ||
      (uint64_t)outLen != rows ||
      ((uint64_t)xLen != rows && (uint64_t)xLen != 2u * rows) ||
      (gammaLen != 0 && (uint32)gammaLen != n)) {
    FARF(ERROR, "rmsnorm_add_f32: bad shape (M=%u n=%u gamma=%d x=%d out=%d)",
         (unsigned)M, (unsigned)n, gammaLen, xLen, outLen);
    return AEE_EINVALIDFORMAT;
  }
  const float *x2 = (uint64_t)xLen == 2u * rows ? x + rows : NULL;
  if (hvx_rmsnorm_add_f32(out, x, x2, M, n, gammaLen ? gamma : NULL, eps, scale,
                          s->quant_pool) != 0) {
    return AEE_EINVALIDFORMAT;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_rope64_det_f32(remote_handle64 handle, uint32 n_q, const float *cs,
                            int csLen, const float *qk, int qkLen, float *y,
                            int yLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if ((uint32)csLen != HEAD_DIM || qkLen <= 0 ||
      (uint32)qkLen % HEAD_DIM != 0u || yLen != qkLen ||
      n_q > (uint32)qkLen / HEAD_DIM) {
    FARF(ERROR, "rope64_det_f32: bad shape (n_q=%u cs=%d qk=%d y=%d)",
         (unsigned)n_q, csLen, qkLen, yLen);
    return AEE_EINVALIDFORMAT;
  }
  memcpy(y, qk, (size_t)qkLen * sizeof(float));
  const uint32_t n_k = (uint32_t)qkLen / HEAD_DIM - n_q;
  hvx_rope64_f32(y, n_q, y + (size_t)n_q * HEAD_DIM, n_k, cs);
  return AEE_SUCCESS;
}

int nntr_hvx_conv_gate_m1_f32(remote_handle64 handle, const float *abc,
                              int abcLen, const float *conv_w, int wLen,
                              const float *state_in, int sLen, float *y,
                              int yLen, float *state_out, int soLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const uint32_t C = (yLen > 0) ? (uint32_t)yLen : 0u;
  if (C == 0u || C % LANES != 0u || (uint32)abcLen != 3u * C ||
      (uint32)wLen != 3u * C || (uint32)sLen != 2u * C ||
      (uint32)soLen != 3u * C) {
    FARF(ERROR,
         "conv_gate_m1_f32: bad shape (abc=%d w=%d state=%d y=%d "
         "state_out=%d)",
         abcLen, wLen, sLen, yLen, soLen);
    return AEE_EINVALIDFORMAT;
  }
  memcpy(state_out, state_in, (size_t)sLen * sizeof(float));
  hvx_conv_gate_m1_f32(abc, state_out, conv_w, y, C);
  return AEE_SUCCESS;
}

int nntr_hvx_router_topk_det_f32(remote_handle64 handle, uint32 top_k,
                                 const float *x, int xLen, const float *w,
                                 int wLen, const float *bias, int biasLen,
                                 float *logits, int logitsLen, uint32 *sel,
                                 int selLen, float *weight, int weightLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const uint32_t K = (xLen > 0) ? (uint32_t)xLen : 0u;
  const uint32_t E = (biasLen > 0) ? (uint32_t)biasLen : 0u;
  if (K == 0u || K % 4u != 0u || E == 0u || E > LANES || top_k == 0u ||
      top_k > E || (uint32)wLen != K * E || (uint32)logitsLen != E ||
      (uint32)selLen != top_k || (uint32)weightLen != top_k) {
    FARF(ERROR,
         "router_topk_det_f32: bad shape (top_k=%u x=%d w=%d bias=%d "
         "logits=%d sel=%d weight=%d)",
         (unsigned)top_k, xLen, wLen, biasLen, logitsLen, selLen, weightLen);
    return AEE_EINVALIDFORMAT;
  }
  /* the kernel reads one 32-lane vector per weight row */
  float *w32 = (float *)calloc((size_t)K * LANES, sizeof(float));
  if (w32 == NULL) {
    return AEE_ENOMEMORY;
  }
  for (uint32_t k = 0; k < K; ++k) {
    memcpy(w32 + (size_t)k * LANES, w + (size_t)k * E, E * sizeof(float));
  }
  hvx_router_topk_f32(x, w32, bias, K, E, top_k, logits, (uint32_t *)sel,
                      weight, s->quant_pool);
  free(w32);
  return AEE_SUCCESS;
}
