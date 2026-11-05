// SPDX-License-Identifier: Apache-2.0
/**
 * @file attn_qprep_native_check.c
 * @brief Compare the production attention Q preparation with separate passes.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hvx_quant_u8.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned params_calls, pack_calls, fused_calls;
static void checked_params(const float *x, uint32_t valid, uint32_t pad,
                           uint32_t k, float *scale, int32_t *zp,
                           hvx_worker_pool *pool) {
  ++params_calls;
  hvx_quant_rows_u8_params(x, valid, pad, k, scale, zp, pool);
}
static int checked_pack(const float *x, uint32_t valid, uint32_t pad,
                        uint32_t k, const float *scale, const int32_t *zp,
                        uint8_t *out, hvx_worker_pool *pool) {
  ++pack_calls;
  return hvx_quant_pack_u8_ah(x, valid, pad, k, scale, zp, out, pool);
}
static int checked_fused(const float *x, const uint16_t *x16, const float *enc,
                         uint32_t hd, uint32_t valid, uint32_t pad, uint32_t k,
                         const float *gamma, float eps, float *scratch,
                         float *scale, int32_t *zp, uint8_t *out,
                         hvx_worker_pool *pool) {
  ++fused_calls;
  return hvx_quant_params_pack_u8_ah(x, x16, enc, hd, valid, pad, k, gamma, eps,
                                     scratch, scale, zp, out, pool);
}

// Generated from the candidate's actual context/address helpers/Q-prep body.
// The helper calls above run the production HVX code on SDK libnative.
#include "attn_qprep_under_test.h"

typedef struct {
  uint32_t nq, from, qh, kvh, hd, br, chunk, window;
  float softcap;
  int fixed, kind, fused;
} Case;

static void *aligned_bytes(size_t bytes) {
  void *p = NULL;
  assert(posix_memalign(&p, 128u, (bytes + 127u) & ~(size_t)127u) == 0);
  assert(p);
  return p;
}

static float value_at(uint32_t row, uint32_t col, unsigned pattern) {
  switch (pattern) {
  case 0:
    return 0.0f;
  case 1:
    return -0.0f;
  case 2:
    return 3.5f;
  case 3:
    return -4.75f;
  case 4:
    return col % 64u == 0u   ? -127.0f
           : col % 64u == 1u ? 128.0f
                             : (float)((int)(col % 255u) - 127) + 0.5f;
  default: {
    if (row % 11u == 0u)
      return (row & 1u) ? -0.0f : 0.0f;
    uint32_t x = (row + 1u) * 1664525u + col * 1013904223u;
    float f = (float)((int)(x % 65536u) - 32768) / 1024.0f;
    return row % 3u == 0u ? f * 0x1p20f : row % 3u == 1u ? f * 0x1p-20f : f;
  }
  }
}

static void check_case(Case tc, uint32_t head, uint32_t qb0, uint32_t chunk,
                       unsigned pattern, hvx_worker_pool *pool) {
  hexkl_attn_f16_shape shape = {tc.nq,  tc.from, tc.from + tc.nq, tc.qh,
                                tc.kvh, tc.hd,   tc.window,       tc.softcap};
  hexkl_attn_f16_tiling tile = {tc.br, 256u, 0u, 0u};
  assert(hexkl_attn_q_tiling_init(&shape, &tile) == HEXKL_ATTN_OK);
  hexkl_kv_q kv;
  memset(&kv, 0, sizeof(kv));
  kv.kind = (hexkl_kv_q_kind)tc.kind;
  kv.fixed = tc.fixed;
  kv.n_head_kv = tc.kvh;
  kv.head_dim = tc.hd;
  kv.tile_bytes = kv.kind == HEXKL_KV_Q8 ? 1024u : 512u;
  const uint32_t rows = chunk * tile.g_br;
  const uint32_t stride = tc.qh * tc.hd + 3u;
  const size_t q_bytes = (size_t)tc.nq * stride * sizeof(float);
  float *q_alloc = aligned_bytes(q_bytes + 128u);
  float *q =
    q_alloc + 1; // Unaligned source rows are valid at the RPC boundary.
  for (uint32_t r = 0; r < tc.nq; ++r)
    for (uint32_t k = 0; k < stride; ++k)
      q[(size_t)r * stride + k] = value_at(r, k, pattern);
  float *q_saved = malloc(q_bytes);
  assert(q_saved);
  memcpy(q_saved, q, q_bytes);
  hexkl_attn_q_io io = {q, stride, NULL, NULL, stride, &kv};
  attn_ctx c;
  memset(&c, 0, sizeof(c));
  c.s = &shape;
  c.t = &tile;
  c.kv = &kv;
  c.io = &io;
  c.pool = pool;
  c.scale = 1.4426950408889634f / sqrtf((float)tc.hd);
  assert(hexkl_attn_q_plan(&shape, &tile, 8u << 20, kv.tile_bytes, chunk,
                           &c.L) == HEXKL_ATTN_OK);
  c.vb = aligned_bytes(c.L.total + 128u);
  memset(c.vb, 0xa5, c.L.total + 128u);
  float *scale_all = aligned_bytes((rows + 64u) * sizeof(float));
  int32_t *zp_all = aligned_bytes((rows + 64u) * sizeof(int32_t));
  c.q_scale = scale_all + 32u;
  c.q_zp = zp_all + 32u;
  for (uint32_t r = 0; r < rows + 64u; ++r) {
    scale_all[r] = -37.0f;
    zp_all[r] = -7654321;
  }
  const size_t elems = (size_t)rows * tc.hd;
  float *gathered = aligned_bytes(elems * sizeof(float));
  uint8_t *reference = aligned_bytes(elems);
  float *scale = malloc(rows * sizeof(float));
  int32_t *zp = malloc(rows * sizeof(int32_t));
  assert(scale && zp);
  // Scalar gather spec, with an independent direct (query, head) mapping.
  for (uint32_t qi = 0; qi < chunk; ++qi)
    for (uint32_t r = 0; r < tile.g_br; ++r) {
      const uint32_t query = (qb0 + qi) * tc.br + r / tile.g;
      const uint32_t h = head * tile.g + r % tile.g;
      for (uint32_t k = 0; k < tc.hd; ++k)
        gathered[((size_t)qi * tile.g_br + r) * tc.hd + k] =
          r < tile.g * tc.br && query < tc.nq
            ? q[(size_t)query * stride + (size_t)h * tc.hd + k]
            : 0.0f;
    }
  hvx_quant_rows_u8_params(gathered, rows, rows, tc.hd, scale, zp, pool);
  assert(hvx_quant_pack_u8_ah(gathered, rows, rows, tc.hd, scale, zp, reference,
                              pool) == 0);
  for (uint32_t r = 0; r < rows; ++r)
    scale[r] *= c.scale;
  params_calls = pack_calls = fused_calls = 0;
  assert(phase_qprep(&c, head, qb0, chunk) == 0);
  assert(fused_calls == (unsigned)tc.fused);
  assert(params_calls == (unsigned)!tc.fused &&
         pack_calls == (unsigned)!tc.fused);
  assert(!memcmp(reference, c.vb + c.L.q_ah, elems));
  assert(!memcmp(gathered, c.vb + c.L.qx_f32, elems * sizeof(float)));
  assert(!memcmp(scale, c.q_scale, rows * sizeof(float)));
  assert(!memcmp(zp, c.q_zp, rows * sizeof(int32_t)));
  assert(!memcmp(q, q_saved, q_bytes));
  for (uint32_t r = 0; r < 32u; ++r) {
    assert(scale_all[r] == -37.0f && scale_all[rows + 32u + r] == -37.0f);
    assert(zp_all[r] == -7654321 && zp_all[rows + 32u + r] == -7654321);
  }
  for (size_t i = 0; i < c.L.total + 128u; ++i)
    if (!(i >= c.L.q_ah && i < c.L.q_ah + elems) &&
        !(i >= c.L.qx_f32 && i < c.L.qx_f32 + elems * sizeof(float)))
      assert(c.vb[i] == 0xa5);
  free(zp);
  free(scale);
  free(reference);
  free(gathered);
  free(zp_all);
  free(scale_all);
  free(c.vb);
  free(q_saved);
  free(q_alloc);
}

int main(void) {
  // Full production chunks, narrow-plan fallback, and a valid padded gather.
  const Case cases[] = {
    {128, 0, 32, 8, 64, 128, 1, 0, 0, 0, 0, 1},
    {1024, 0, 32, 8, 64, 128, 8, 0, 0, 0, 0, 1},
    {2048, 0, 32, 8, 64, 128, 9, 0, 0, 0, 0, 1},
    {1024, 0, 32, 8, 64, 16, 64, 0, 0, 0, 0, 1},
    {128, 0, 32, 8, 64, 96, 2, 0, 0, 0, 0, 1},
    // Every guard falls back to the original separate quantizer and pack.
    {127, 0, 32, 8, 64, 16, 8, 0, 0, 0, 0, 0},
    {128, 64, 32, 8, 64, 128, 1, 0, 0, 0, 0, 0},
    {128, 0, 32, 8, 64, 128, 1, 64, 0, 0, 0, 0},
    {128, 0, 32, 8, 64, 128, 1, 0, 5, 0, 0, 0},
    {128, 0, 32, 8, 64, 128, 1, 0, 0, 1, 0, 0},
    {128, 0, 32, 8, 64, 128, 1, 0, 0, 0, 1, 0},
    {128, 0, 16, 8, 64, 128, 1, 0, 0, 0, 0, 0},
    {128, 0, 32, 4, 64, 64, 2, 0, 0, 0, 0, 0},
    {128, 0, 32, 8, 128, 128, 1, 0, 0, 0, 0, 0}};
  unsigned checked = 0;
  hvx_worker_pool *pool = hvx_worker_pool_create(3u);
  assert(pool);
  for (unsigned mode = 0; mode < 2u; ++mode)
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
      const Case tc = cases[ci];
      const uint32_t nqb = (tc.nq + tc.br - 1u) / tc.br;
      for (uint32_t qb = 0; qb < nqb; qb += tc.chunk) {
        const uint32_t chunk = nqb - qb < tc.chunk ? nqb - qb : tc.chunk;
        for (unsigned pattern = 0; pattern < 6u; ++pattern)
          for (unsigned last = 0; last < 2u; ++last) {
            check_case(tc, last ? tc.kvh - 1u : 0u, qb, chunk, pattern,
                       mode ? pool : NULL);
            ++checked;
          }
      }
    }
  hvx_worker_pool_destroy(pool);
  printf("LFM Q-prep exact bytes/scales/zp/gather/guards: %u cases PASS\n",
         checked);
  return 0;
}
