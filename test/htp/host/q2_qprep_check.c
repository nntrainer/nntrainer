// SPDX-License-Identifier: Apache-2.0
/**
 * @file   q2_qprep_check.c
 * @brief  hexkl_attn_q2's stage_qprep_kd (extracted by
 *         q2c_workers_extract.py) on hvx_emu: the f32 Q it quantizes in
 *         VTCM, with K's scale per dim folded in (fk_ratio), against the
 *         u16 path fed with the host's quantization of the same rows
 *         (attention_quant_utils.h quant_u16_f32 with one inv per element: x *
 * inv[d] + (zp + 0.5), clamped, truncated). The u16 path moves bytes only, so
 * the two runs' tiles must be equal byte for byte, whatever the tile layout.
 * Rows past n_rows and a head past the first KV head's group are covered.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "HAP_perf.h"
#include "hexkl_attn_q2.h"
#include "hvx_convert.h"

/* the stub has the qtimer only; stage times are not checked */
static inline uint64_t HAP_perf_get_time_us(void) { return 0; }

#include "q2_qprep.inc"

static uint32_t g_rng = 0x9e37u;
static uint32_t rnd(void) {
  g_rng = g_rng * 1664525u + 1013904223u;
  return g_rng >> 8;
}

/** quant_u16_f32 (inv per element), its two roundings */
static uint16_t host_q(float x, float inv, float zp) {
  volatile float m = x * inv;
  float v = m + (zp + 0.5f);
  v = v < 0.0f ? 0.0f : v > 65535.0f ? 65535.0f : v;
  return (uint16_t)v;
}

static int check_case(uint32_t hd, uint32_t n_kv, uint32_t n_hq, uint32_t h,
                      uint32_t n_rows) {
  const uint32_t dt = hd / 32u, G = n_hq / n_kv, q_stride = n_hq * hd;
  const size_t tiles = 3u * 2u * dt * HEXKL_ATTN_Q2_TILE;
  const size_t stage = 2u * HEXKL_ATTN_Q2_ROWS * hd * 4u;
  uint8_t *vb = aligned_alloc(128, tiles + stage);
  uint8_t *ref = malloc(2u * dt * HEXKL_ATTN_Q2_TILE);
  float *ratio = aligned_alloc(128, (size_t)n_kv * hd * 4u);
  float *q_enc = calloc(2u * n_hq, 4u), *q_inv = calloc(n_hq, 4u);
  float *x = calloc((size_t)HEXKL_ATTN_Q2_ROWS * q_stride, 4u);
  uint16_t *x16 = calloc((size_t)HEXKL_ATTN_Q2_ROWS * q_stride, 2u);
  for (uint32_t i = 0; i < n_kv * hd; ++i)
    ratio[i] = (float)(1u + rnd() % 1000u) / 1000.0f;
  for (uint32_t k = 0; k < n_hq; ++k) {
    q_enc[2u * k] = 0.0004f * (float)(1u + rnd() % 50u);
    q_enc[2u * k + 1u] = (float)(20000u + rnd() % 20000u);
    q_inv[k] = 1.0f / q_enc[2u * k];
  }
  /* most codes inside, a few past both ends (the clamp) */
  for (size_t i = 0; i < (size_t)HEXKL_ATTN_Q2_ROWS * q_stride; ++i)
    x[i] = ((float)(int)(rnd() % 40001u) - 20000.0f) * 0.0011f;
  for (uint32_t r = 0; r < HEXKL_ATTN_Q2_ROWS; ++r)
    for (uint32_t d = 0; d < hd; ++d) {
      const size_t at = (size_t)r * q_stride + h * hd + d;
      x16[at] =
        host_q(x[at], q_inv[h] * ratio[(h / G) * hd + d], q_enc[2u * h + 1u]);
    }
  hexkl_kv_q kv;
  memset(&kv, 0, sizeof kv);
  kv.fk_ratio = ratio;
  kv.k_per_dim = 1;
  hexkl_attn_q2_io io;
  memset(&io, 0, sizeof io);
  io.q_stride = q_stride;
  io.q_enc = q_enc;
  io.q_inv = q_inv;
  io.kv = &kv;
  item_t it;
  memset(&it, 0, sizeof it);
  it.h = h;
  it.n = h / G;
  it.n_rows = n_rows;
  ctx c;
  memset(&c, 0, sizeof c);
  c.vb = vb;
  c.io = &io;
  c.hd = hd;
  c.dt = dt;
  c.G = G;
  c.items = &it;
  c.n_items = 1;
  c.L.q_ah = 0;
  c.L.q_u16 = (uint32_t)tiles;
  int bad = 0;
  for (int pass = 0; pass < 2; ++pass) {
    /* pass 0: the u16 rows; pass 1: the f32 rows, folded inside */
    io.q = pass ? NULL : x16;
    io.q_f32 = pass ? x : NULL;
    const size_t esz = pass ? 4u : 2u;
    uint8_t *st = q_stage(&c, 0);
    for (uint32_t r = 0; r < n_rows; ++r)
      memcpy(st + (size_t)r * hd * esz,
             (const uint8_t *)(pass ? (const void *)x : (const void *)x16) +
               ((size_t)r * q_stride + h * hd) * esz,
             hd * esz);
    memset(vb, 0xa5, tiles);
    stage_qprep_kd(&c, 0, pass);
    if (!pass)
      memcpy(ref, q_tiles_lo(&c, 0), 2u * dt * HEXKL_ATTN_Q2_TILE);
    else
      bad = memcmp(ref, q_tiles_lo(&c, 0), 2u * dt * HEXKL_ATTN_Q2_TILE) != 0;
  }
  printf("q2 qprep kd hd=%u n_kv=%u n_hq=%u h=%u rows=%u: %s\n", hd, n_kv, n_hq,
         h, n_rows, bad ? "MISMATCH" : "same");
  free(vb);
  free(ref);
  free(ratio);
  free(q_enc);
  free(q_inv);
  free(x);
  free(x16);
  return bad;
}

int main(void) {
  int bad = 0;
  /* Gemma 4's sliding (8 x 256) and full (2 x 512) heads, a short block,
     a head of the second group, a small head (hd 128) */
  bad += check_case(256u, 8u, 16u, 0u, 64u);
  bad += check_case(256u, 8u, 16u, 5u, 17u);
  bad += check_case(512u, 2u, 16u, 9u, 64u);
  bad += check_case(512u, 2u, 16u, 15u, 3u);
  bad += check_case(128u, 1u, 2u, 1u, 40u);
  printf("%s\n", bad ? "Q2 QPREP KD FAILED" : "Q2 QPREP KD OK");
  return bad != 0;
}
