// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_kv_q.c
 * @date   23 Sep 2026
 * @brief  DSP-resident int8 / int4 KV cache with per-token scales
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hexkl_kv_q.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __hexagon__
#include "hexkl_micro.h"
#include <AEEStdErr.h>
#else
// Host build (the unit test): the codes this file returns, with the SDK's
// values (AEEStdErr.h: AEE_EOFFSET + 0x002 / 0x00E / 0x001).
#define AEE_SUCCESS 0
#define AEE_EFAILED 0x80000401
#define AEE_ENOMEMORY 0x80000402
#define AEE_EBADPARM 0x8000040E
#endif

void hexkl_kv_q_quant_k_row(const float *x, uint32_t hd, int32_t qmax,
                            int8_t *q, float *scale, int32_t *colsum) {
  float amax = 0.0f;
  for (uint32_t d = 0; d < hd; ++d) {
    const float a = fabsf(x[d]);
    if (a > amax) {
      amax = a;
    }
  }
  const float fq = (float)qmax;
  int32_t sum = 0;
  if (amax == 0.0f) {
    memset(q, 0, hd);
    *scale = 1.0f;
    *colsum = 0;
    return;
  }
  const float inv = fq / amax;
  for (uint32_t d = 0; d < hd; ++d) {
    float r = rintf(x[d] * inv);
    if (r > fq) {
      r = fq;
    } else if (r < -fq) {
      r = -fq;
    }
    q[d] = (int8_t)r;
    sum += (int32_t)r;
  }
  *scale = amax / fq;
  *colsum = sum;
}

void hexkl_kv_q_quant_v_row(const float *x, uint32_t hd, int32_t qmax,
                            int8_t *q, float *scales) {
  const float fq = (float)qmax;
  for (uint32_t g = 0; g < hd / 32u; ++g) {
    const float *xg = x + 32u * g;
    int8_t *qg = q + 32u * g;
    float amax = 0.0f;
    for (uint32_t d = 0; d < 32u; ++d) {
      const float a = fabsf(xg[d]);
      if (a > amax) {
        amax = a;
      }
    }
    if (amax == 0.0f) {
      memset(qg, 0, 32u);
      scales[g] = 1.0f;
      continue;
    }
    const float inv = fq / amax;
    for (uint32_t d = 0; d < 32u; ++d) {
      float r = rintf(xg[d] * inv);
      if (r > fq) {
        r = fq;
      } else if (r < -fq) {
        r = -fq;
      }
      qg[d] = (int8_t)r;
    }
    scales[g] = amax / fq;
  }
}

static void free_slot(hexkl_kv_q *kv) {
  free(kv->kt4);
  free(kv->v4);
  free(kv->s_k);
  free(kv->colsum_k);
  free(kv->s_v);
  free(kv->kt);
  free(kv->v);
  free(kv->stage_kt);
  free(kv->stage_v);
  memset(kv, 0, sizeof(*kv));
}

int hexkl_kv_q_register(hexkl_kv_q_table *tbl, hexkl_kv_q_kind kind,
                        uint32_t max_rows, uint32_t n_head_kv,
                        uint32_t head_dim, uint32_t *out_handle) {
  if (!tbl || !out_handle || max_rows == 0 || n_head_kv == 0 || head_dim == 0 ||
      (head_dim % 32u) != 0 || head_dim > 256u ||
      (kind != HEXKL_KV_Q8 && kind != HEXKL_KV_Q4)) {
    return AEE_EBADPARM;
  }
  uint32_t slot = HEXKL_KV_Q_MAX;
  for (uint32_t i = 0; i < HEXKL_KV_Q_MAX; ++i) {
    if (!tbl->slots[i].in_use) {
      slot = i;
      break;
    }
  }
  if (slot == HEXKL_KV_Q_MAX) {
    return AEE_ENOMEMORY;
  }

  hexkl_kv_q *kv = &tbl->slots[slot];
  memset(kv, 0, sizeof(*kv));
  kv->kind = kind;
  kv->qmax = hexkl_kv_q_qmax(kind);
  kv->tile_bytes = hexkl_kv_q_tile_bytes(kind);
  kv->max_rows = ((max_rows + 31u) / 32u) * 32u;
  kv->n_head_kv = n_head_kv;
  kv->head_dim = head_dim;
  kv->n_col_tiles = kv->max_rows / 32u;
  kv->n_dot_tiles = head_dim / 32u;

  const size_t values = (size_t)n_head_kv * kv->max_rows * head_dim;
  const size_t rows = (size_t)n_head_kv * kv->max_rows;
  const size_t n_tiles = rows / 32u * kv->n_dot_tiles;
  // calloc throughout: rows past cache_to read as zero with scale 1, so a
  // block that runs past the cache multiplies finite zeros, never garbage.
  // Offset-binary masters: an unwritten row must read as 0, i.e. 128.
  kv->kt4 = (uint8_t *)malloc(values);
  kv->v4 = (uint8_t *)malloc(values);
  kv->s_k = (float *)calloc(rows, sizeof(float));
  kv->colsum_k = (int32_t *)calloc(rows, sizeof(int32_t));
  kv->s_v =
    (float *)calloc(rows * kv->n_dot_tiles + HEXKL_KV_Q_SV_PAD, sizeof(float));
  kv->kt = (uint8_t *)calloc(n_tiles, kv->tile_bytes);
  kv->v = (uint8_t *)calloc(n_tiles, kv->tile_bytes);
  kv->stage_kt = (int8_t *)calloc((size_t)head_dim * 32u, 1u);
  kv->stage_v = (int8_t *)calloc((size_t)head_dim * 32u, 1u);
  if (!kv->kt4 || !kv->v4 || !kv->s_k || !kv->colsum_k || !kv->s_v || !kv->kt ||
      !kv->v || !kv->stage_kt || !kv->stage_v) {
    free_slot(kv);
    return AEE_ENOMEMORY;
  }
  memset(kv->kt4, HEXKL_KV_Q_BIAS, values);
  memset(kv->v4, HEXKL_KV_Q_BIAS, values);
  for (size_t i = 0; i < rows; ++i) {
    kv->s_k[i] = 1.0f;
  }
  for (size_t i = 0; i < rows * kv->n_dot_tiles + HEXKL_KV_Q_SV_PAD; ++i) {
    kv->s_v[i] = 1.0f;
  }
  kv->in_use = 1;
  *out_handle = slot;
  return AEE_SUCCESS;
}

int hexkl_kv_q_release(hexkl_kv_q_table *tbl, uint32_t handle) {
  if (!tbl || handle >= HEXKL_KV_Q_MAX || !tbl->slots[handle].in_use) {
    return AEE_EBADPARM;
  }
  free_slot(&tbl->slots[handle]);
  return AEE_SUCCESS;
}

const hexkl_kv_q *hexkl_kv_q_get(const hexkl_kv_q_table *tbl, uint32_t handle) {
  if (!tbl || handle >= HEXKL_KV_Q_MAX || !tbl->slots[handle].in_use) {
    return NULL;
  }
  return &tbl->slots[handle];
}

void hexkl_kv_q_stage(hexkl_kv_q *kv, uint32_t n, uint32_t c) {
  const uint32_t hd = kv->head_dim;
  const uint32_t row0 = 32u * c;
  for (uint32_t rr = 0; rr < 32u; ++rr) {
    const uint32_t row = row0 + rr;
    for (uint32_t d = 0; d < hd; ++d) {
      kv->stage_kt[(size_t)d * 32u + rr] =
        (int8_t)((int)kv->kt4[hexkl_kv_q_kt4_index(kv, n, row, d)] -
                 HEXKL_KV_Q_BIAS);
      kv->stage_v[(size_t)rr * hd + d] =
        (int8_t)((int)kv->v4[hexkl_kv_q_v4_index(kv, n, row, d)] -
                 HEXKL_KV_Q_BIAS);
    }
  }
}

/**
 * @brief Re-bakes the K^T and V tiles of column tile c of head n from the
 *        staging through a VTCM scratch tile. DSP only: needs HexKL.
 */
static int bake_col_tile(hexkl_kv_q *kv, uint32_t n, uint32_t c,
                         uint8_t *vtcm_base) {
#ifdef __hexagon__
  hexkl_kv_q_stage(kv, n, c);
  const uint32_t tb = kv->tile_bytes;
  for (uint32_t d = 0; d < kv->n_dot_tiles; ++d) {
    int rc;
    // K^T is [head_dim][32]: reduction rows are dims, columns cache rows;
    // tile (d, 0) of a 32-column matrix.
    if (kv->kind == HEXKL_KV_Q4) {
      rc = hexkl_micro_hmx_rm_to_wh_i4(vtcm_base, 0u, kv->stage_kt, d, 0u, 32u);
    } else {
      rc = hexkl_micro_hmx_rm_to_wh_i8(vtcm_base, 0u, kv->stage_kt, d, 0u, 32u);
    }
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    memcpy(kv->kt + hexkl_kv_q_tile_off(kv, n, c, d), vtcm_base, tb);
    // V is [32][head_dim]: reduction rows are cache rows; tile (0, d).
    if (kv->kind == HEXKL_KV_Q4) {
      rc = hexkl_micro_hmx_rm_to_wh_i4(vtcm_base, tb, kv->stage_v, 0u, d,
                                       kv->head_dim);
    } else {
      rc = hexkl_micro_hmx_rm_to_wh_i8(vtcm_base, tb, kv->stage_v, 0u, d,
                                       kv->head_dim);
    }
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    memcpy(kv->v + hexkl_kv_q_tile_off(kv, n, c, d), vtcm_base + tb, tb);
  }
  return AEE_SUCCESS;
#else
  (void)kv;
  (void)n;
  (void)c;
  (void)vtcm_base;
  return AEE_EFAILED;
#endif
}

int hexkl_kv_q_append(hexkl_kv_q_table *tbl, uint32_t handle, uint32_t row0,
                      uint32_t n_rows, const uint16_t *k_rows,
                      const uint16_t *v_rows, uint8_t *vtcm_base) {
  hexkl_kv_q *kv = (hexkl_kv_q *)hexkl_kv_q_get(tbl, handle);
  if (!kv || !k_rows || !v_rows || n_rows == 0 ||
      row0 + n_rows > kv->max_rows || row0 + n_rows < row0) {
    return AEE_EBADPARM;
  }
  const uint32_t hd = kv->head_dim;
  const uint32_t stride = kv->n_head_kv * hd;
  float x[256];
  int8_t q[256];
  float sv[8];

  for (uint32_t r = 0; r < n_rows; ++r) {
    const uint32_t row = row0 + r;
    const uint16_t *krow = k_rows + (size_t)r * stride;
    const uint16_t *vrow = v_rows + (size_t)r * stride;
    for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
      for (uint32_t d = 0; d < hd; ++d) {
        x[d] = hexkl_kv_q_hf_to_f32(krow[n * hd + d]);
      }
      float sk;
      int32_t cs;
      hexkl_kv_q_quant_k_row(x, hd, kv->qmax, q, &sk, &cs);
      kv->s_k[hexkl_kv_q_sk_index(kv, n, row)] = sk;
      kv->colsum_k[hexkl_kv_q_sk_index(kv, n, row)] = cs;
      for (uint32_t d = 0; d < hd; ++d) {
        kv->kt4[hexkl_kv_q_kt4_index(kv, n, row, d)] =
          (uint8_t)(q[d] + HEXKL_KV_Q_BIAS);
      }

      for (uint32_t d = 0; d < hd; ++d) {
        x[d] = hexkl_kv_q_hf_to_f32(vrow[n * hd + d]);
      }
      hexkl_kv_q_quant_v_row(x, hd, kv->qmax, q, sv);
      for (uint32_t g = 0; g < kv->n_dot_tiles; ++g) {
        kv->s_v[hexkl_kv_q_sv_index(kv, n, row, g)] = sv[g];
      }
      for (uint32_t d = 0; d < hd; ++d) {
        kv->v4[hexkl_kv_q_v4_index(kv, n, row, d)] =
          (uint8_t)(q[d] + HEXKL_KV_Q_BIAS);
      }
    }
  }

  if (!vtcm_base) {
    return AEE_SUCCESS;
  }
  const uint32_t c_lo = row0 / 32u;
  const uint32_t c_hi = (row0 + n_rows - 1u) / 32u;
  for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
    for (uint32_t c = c_lo; c <= c_hi; ++c) {
      const int rc = bake_col_tile(kv, n, c, vtcm_base);
      if (rc != AEE_SUCCESS) {
        return rc;
      }
    }
  }
  return AEE_SUCCESS;
}

int hexkl_kv_q_dump(const hexkl_kv_q *kv, uint32_t row0, uint32_t n_rows,
                    int8_t *k_q, int8_t *v_q, float *s_k, int32_t *colsum_k,
                    float *s_v) {
  if (!kv || n_rows == 0 || row0 + n_rows > kv->max_rows ||
      row0 + n_rows < row0) {
    return AEE_EBADPARM;
  }
  const uint32_t hd = kv->head_dim;
  const uint32_t stride = kv->n_head_kv * hd;
  for (uint32_t r = 0; r < n_rows; ++r) {
    const uint32_t row = row0 + r;
    for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
      if (k_q) {
        for (uint32_t d = 0; d < hd; ++d) {
          k_q[(size_t)r * stride + n * hd + d] =
            (int8_t)((int)kv->kt4[hexkl_kv_q_kt4_index(kv, n, row, d)] -
                     HEXKL_KV_Q_BIAS);
        }
      }
      if (v_q) {
        for (uint32_t d = 0; d < hd; ++d) {
          v_q[(size_t)r * stride + n * hd + d] =
            (int8_t)((int)kv->v4[hexkl_kv_q_v4_index(kv, n, row, d)] -
                     HEXKL_KV_Q_BIAS);
        }
      }
      if (s_k) {
        s_k[(size_t)r * kv->n_head_kv + n] =
          kv->s_k[hexkl_kv_q_sk_index(kv, n, row)];
      }
      if (colsum_k) {
        colsum_k[(size_t)r * kv->n_head_kv + n] =
          kv->colsum_k[hexkl_kv_q_sk_index(kv, n, row)];
      }
      if (s_v) {
        for (uint32_t g = 0; g < kv->n_dot_tiles; ++g) {
          s_v[((size_t)r * kv->n_head_kv + n) * kv->n_dot_tiles + g] =
            kv->s_v[hexkl_kv_q_sv_index(kv, n, row, g)];
        }
      }
    }
  }
  return AEE_SUCCESS;
}
