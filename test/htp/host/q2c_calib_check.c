// SPDX-License-Identifier: Apache-2.0
/**
 * @file   q2c_calib_check.c
 * @brief  The q2 attention calibration's lane workers (nntr_hvx_attn_q.c
 *         q2c_worker, q2c_worker_kd, q2c_worker_kd_q, extracted by
 *         q2c_workers_extract.py) on hvx_emu against a scalar reference,
 *         lane by lane: per (head, dim) |K| and |V| maxima as sign-cleared
 *         fp16 bits (q2c_worker_kd), their per-head reductions and Q's
 *         per-head range (q2c_worker), and the per-head range of q * ratio
 *         (q2c_worker_kd_q), which must equal the per-dim extremes times
 *         the ratio that it replaced, bit for bit. Lanes past the rows (a
 *         prompt shorter than the pool) are run as the pool runs them: n =
 *         min(units, lanes).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "q2c_workers.inc"

static uint32_t g_rng = 0x2c11u;
static uint32_t rnd(void) {
  g_rng = g_rng * 1664525u + 1013904223u;
  return g_rng >> 8;
}

/** a finite fp16 of either sign */
static uint16_t rnd_hf(void) {
  const uint32_t r = rnd();
  return (uint16_t)(((r & 1u) << 15) | ((r >> 1) % 31u) << 10 |
                    ((r >> 6) & 0x3ffu));
}

static int check_case(uint32_t n_kv, uint32_t hd, uint32_t n_hq,
                      uint32_t n_rows, uint32_t n_q, uint32_t lanes) {
  const uint32_t stride = n_kv * hd, q_stride = n_hq * hd;
  uint16_t *k = malloc((size_t)n_rows * stride * 2u + 128u);
  uint16_t *v = malloc((size_t)n_rows * stride * 2u + 128u);
  float *q = malloc((size_t)n_q * q_stride * 4u + 128u);
  for (size_t i = 0; i < (size_t)n_rows * stride; ++i) {
    k[i] = rnd_hf();
    v[i] = rnd_hf();
  }
  for (size_t i = 0; i < (size_t)n_q * q_stride; ++i)
    q[i] = ((float)(int)(rnd() % 20001u) - 10000.0f) * 0.00037f;
  uint16_t *vmax = calloc((size_t)lanes * stride, 2u);
  uint16_t *kmax = calloc((size_t)lanes * stride, 2u);
  float *qlo = calloc((size_t)lanes * n_hq * 2u, 4u);
  float *ratio = malloc((size_t)stride * 4u);
  for (uint32_t e = 0; e < stride; ++e)
    ratio[e] = 0.001f * (float)(1u + rnd() % 997u) / 0.997f;
  uint16_t *vmax0 = calloc((size_t)lanes * stride, 2u);
  uint16_t *kmax0 = calloc((size_t)lanes * n_kv, 2u);
  float *qlo0 = calloc((size_t)lanes * n_hq * 2u, 4u);
  q2c_ctx kd = {
    k,    v,    n_rows,   stride, n_kv, hd,  q,
    n_q,  n_hq, q_stride, vmax,   kmax, qlo, qlo + (size_t)lanes * n_hq,
    ratio};
  q2c_ctx hd0 = {
    k,   v,    n_rows,   stride, n_kv,  hd,   q,
    n_q, n_hq, q_stride, vmax0,  kmax0, qlo0, qlo0 + (size_t)lanes * n_hq,
    NULL};
  const uint32_t units = n_rows > n_q ? n_rows : n_q;
  const uint32_t n = units < lanes ? units : lanes;
  for (uint32_t i = 0; i < n; ++i) {
    q2c_worker_kd(n, i, &kd);
    q2c_worker(n, i, &hd0);
  }
  for (uint32_t i = 0; i < n; ++i)
    q2c_worker_kd_q(n, i, &kd);
  int bad = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t r0 = (uint32_t)(((uint64_t)n_rows * i) / n);
    const uint32_t r1 = (uint32_t)(((uint64_t)n_rows * (i + 1u)) / n);
    for (uint32_t h = 0; h < n_kv; ++h) {
      uint16_t khead = 0;
      for (uint32_t d = 0; d < hd; ++d) {
        uint16_t km = 0, vm = 0;
        for (uint32_t r = r0; r < r1; ++r) {
          const size_t at = (size_t)r * stride + h * hd + d;
          const uint16_t a = k[at] & 0x7fffu, b = v[at] & 0x7fffu;
          km = a > km ? a : km;
          vm = b > vm ? b : vm;
        }
        const size_t e = (size_t)i * stride + h * hd + d;
        bad += kmax[e] != km || vmax[e] != vm || vmax0[e] != vm;
        khead = km > khead ? km : khead;
      }
      bad += kmax0[(size_t)i * n_kv + h] != khead;
    }
    const uint32_t q0 = (uint32_t)(((uint64_t)n_q * i) / n);
    const uint32_t q1 = (uint32_t)(((uint64_t)n_q * (i + 1u)) / n);
    const uint32_t G = n_hq / n_kv;
    for (uint32_t h = 0; h < n_hq; ++h) {
      float hlo = 1e30f, hhi = -1e30f; /* q2c_worker: q */
      float flo = 1e30f, fhi = -1e30f; /* q2c_worker_kd_q: q * ratio */
      float plo = 1e30f, phi = -1e30f; /* what it replaced: min (q) * ratio */
      for (uint32_t d = 0; d < hd; ++d) {
        const float rd = ratio[(h / G) * hd + d];
        float lo = 1e30f, hi = -1e30f;
        for (uint32_t r = q0; r < q1; ++r) {
          const float x = q[(size_t)r * q_stride + h * hd + d];
          volatile float xr = x * rd;
          lo = x < lo ? x : lo;
          hi = x > hi ? x : hi;
          flo = xr < flo ? xr : flo;
          fhi = xr > fhi ? xr : fhi;
        }
        volatile float a = lo * rd, b = hi * rd;
        plo = a < plo ? a : plo;
        phi = b > phi ? b : phi;
        hlo = lo < hlo ? lo : hlo;
        hhi = hi > hhi ? hi : hhi;
      }
      const size_t e = (size_t)i * n_hq + h;
      bad +=
        memcmp(&hd0.qlo[e], &hlo, 4) != 0 || memcmp(&hd0.qhi[e], &hhi, 4) != 0;
      bad += q1 > q0 && (memcmp(&kd.qlo[e], &flo, 4) != 0 ||
                         memcmp(&kd.qhi[e], &fhi, 4) != 0);
      bad +=
        q1 > q0 && (memcmp(&flo, &plo, 4) != 0 || memcmp(&fhi, &phi, 4) != 0);
    }
  }
  printf("q2c workers n_kv=%u hd=%u n_hq=%u rows=%u q=%u lanes=%u ran=%u: "
         "%d bad\n",
         n_kv, hd, n_hq, n_rows, n_q, lanes, n, bad);
  free(k);
  free(v);
  free(q);
  free(vmax);
  free(kmax);
  free(qlo);
  free(ratio);
  free(vmax0);
  free(kmax0);
  free(qlo0);
  return bad;
}

int main(void) {
  int bad = 0;
  /* Gemma 4's sliding (8 x 256) and full (2 x 512, two Q blocks) shapes;
     rows below, at and past the 8 lanes; a pool of 3 */
  bad += check_case(8u, 256u, 16u, 130u, 130u, 8u);
  bad += check_case(2u, 512u, 16u, 70u, 70u, 8u);
  bad += check_case(8u, 256u, 16u, 5u, 5u, 8u);
  bad += check_case(2u, 512u, 16u, 200u, 1u, 3u);
  bad += check_case(2u, 64u, 4u, 33u, 33u, 8u);
  printf("%s\n", bad ? "Q2C CALIB WORKERS FAILED" : "Q2C CALIB WORKERS OK");
  return bad != 0;
}
