// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   m1_ops_host_check.c
 * @date   27 Sep 2026
 * @brief  Host check: the real HVX M=1 small-op kernels, compiled against
 *         hvx_emu/, are bit-identical to m1_ops_det.h and within tolerance
 *         of a double reference
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Two halves (plan 82 section 1):
 *  1. BIT IDENTITY. hvx_m1_ops_f32.c and hvx_conv_gate_f32.c -- the skel's
 *     own sources, not stand-ins -- run on the lane-by-lane emulation and
 *     are memcmp'd against the scalar spec: RMSNorm n = 2048, the per-head
 *     q/k norm (32 x 64, 8 x 64), RoPE at positions 0 / 1 / 511 / 1023 /
 *     4095 (0 is the identity), and the conv1d + gate as an M=1 chain of 8
 *     tokens from a zero state and from the state a 7-row prefill-shape
 *     call leaves, each chain against ONE prefill-shape hvx_conv_gate_f32
 *     call over the same rows. What this proves: loop bounds, lane and
 *     head indexing, the reduction tree, operation order, the zero state
 *     and the state shift. What it rests on: one Vsf op = one IEEE op,
 *     device-confirmed for add/sub/mul (rule 24), re-checked by HvxM1Ops.*.
 *  2. TOLERANCE. The spec against a plain fp32 reference (straight C,
 *     sqrtf, CPU order; printed) and a double reference (asserted):
 *     RMSNorm <= 4 ulp of |y|, RoPE <= 2^-22 (|a| + |b|), conv
 *     <= 2^-21 |b| sum|w_i g_i|, each with a floor of two subnormal quanta
 *     so the all-subnormal row (kept, not flushed: rule 24) is judged on
 *     what f32 can hold.
 *
 * Inputs: an LCG from a fixed seed, plus three fixed rows / heads: all
 * zeros, all subnormal (+-1e-39) and large (|x| ~ 1e4).
 *
 * The router (#132) has no tolerance half: its output is a selection. It
 * is memcmp'd against the spec (logits, selection order, weights) at
 * three shapes on random, exact-tie and 1-ulp near-tie rows; then the
 * spec is set against a copy of the CPU's buildExpertAssignments (expf, a
 * true divide, a sequential dot, the total-order comparator) on 100000
 * LFM2.5-shaped rows, and every selection that differs must sit at a CPU
 * 4th/5th score gap below 1e-4 -- the near-tie class, never a logic or
 * tie-rule difference.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_conv_gate_f32.h"
#include "hvx_m1_ops_f32.h"
#include "m1_ops_det.h"

/* The pool's NULL path, in place of the pthread pool: hvx_worker_pool_run
   with no workers calls func(1, 0, ctx) once on the caller. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  (void)pool;
  if (n_units) {
    func(1u, 0u, ctx);
  }
}

static int g_fail = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL: " __VA_ARGS__);                                            \
      printf("\n");                                                            \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

/* ---- inputs ------------------------------------------------------------ */

static uint32_t g_seed = 0x82820001u;
static float frand(float lo, float hi) {
  g_seed = g_seed * 1664525u + 1013904223u;
  return lo + (hi - lo) * ((float)(g_seed >> 8) / 16777216.0f);
}
static void fill_rand(float *x, uint32_t n, float lo, float hi) {
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = frand(lo, hi);
  }
}
/* Row kind 1: zeros; 2: subnormal +-1e-39; 3: large; else random. */
static void fill_row(float *x, uint32_t n, int kind) {
  for (uint32_t i = 0; i < n; ++i) {
    switch (kind) {
    case 1:
      x[i] = 0.0f;
      break;
    case 2:
      x[i] = (i & 1u) ? -1e-39f : 1e-39f;
      break;
    case 3:
      x[i] = frand(-1e4f, 1e4f);
      break;
    default:
      x[i] = frand(-3.0f, 3.0f);
    }
  }
}

/* ---- tolerance helpers ------------------------------------------------- */

/** @brief Two subnormal quanta: the absolute floor under every relative
 *         bound, since a subnormal result cannot be closer than that. */
#define SUBNORMAL_FLOOR (2.0 * 1.4012984643248171e-45)

/** @brief |a - b| in ulps of |b| (b's f32 spacing; the subnormal quantum
 *         at and below the smallest normal). */
static double ulp_dist(float a, double b) {
  const float bf = (float)b;
  float ulp = nextafterf(fabsf(bf), INFINITY) - fabsf(bf);
  if (ulp < 1.4012984643248171e-45f) {
    ulp = 1.4012984643248171e-45f;
  }
  return fabs((double)a - b) / (double)ulp;
}

/* ---- RMSNorm / q-k norm ------------------------------------------------ */

static void ref_rmsnorm_f32(const float *x, const float *gamma, float *y,
                            uint32_t chunk, float eps) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < chunk; ++i) {
    sum += x[i] * x[i];
  }
  const float r = 1.0f / sqrtf(sum / (float)chunk + eps);
  for (uint32_t i = 0; i < chunk; ++i) {
    y[i] = x[i] * r * gamma[i];
  }
}
static void ref_rmsnorm_f64(const float *x, const float *gamma, double *y,
                            uint32_t chunk, float eps) {
  double sum = 0.0;
  for (uint32_t i = 0; i < chunk; ++i) {
    sum += (double)x[i] * (double)x[i];
  }
  const double r = 1.0 / sqrt(sum / (double)chunk + (double)eps);
  for (uint32_t i = 0; i < chunk; ++i) {
    y[i] = (double)x[i] * r * (double)gamma[i];
  }
}

/** @brief One norm case: n floats in chunks; heads 0..2 of the row are the
 *         fixed kinds when there is more than one chunk, else the whole
 *         row cycles through the kinds. */
static void check_rmsnorm(const char *name, uint32_t n, uint32_t chunk) {
  const float eps = 1e-5f;
  const uint32_t nchunk = n / chunk;
  float *x = malloc(n * sizeof(float));
  float *gamma = malloc(chunk * sizeof(float));
  float *y_hvx = malloc(n * sizeof(float));
  float *y_det = malloc(n * sizeof(float));
  float *y_f32 = malloc(n * sizeof(float));
  double *y_f64 = malloc(n * sizeof(double));
  float *rs_hvx = malloc(nchunk * sizeof(float));
  float *rs_det = malloc(nchunk * sizeof(float));
  fill_rand(gamma, chunk, 0.5f, 1.5f);

  uint32_t bad = 0;
  double max_ulp32 = 0.0, max_ulp64 = 0.0;
  const int n_kinds = (nchunk > 1u) ? 1 : 4;
  for (int kind = 0; kind < n_kinds; ++kind) {
    if (nchunk > 1u) {
      for (uint32_t c = 0; c < nchunk; ++c) {
        fill_row(x + c * chunk, chunk, (c < 3u) ? (int)c + 1 : 0);
      }
    } else {
      fill_row(x, n, kind);
    }
    memset(y_hvx, 0xA5, n * sizeof(float));
    memset(rs_hvx, 0xA5, nchunk * sizeof(float));
    hvx_rmsnorm_f32(x, gamma, y_hvx, n, chunk, eps, rs_hvx);
    m1_rmsnorm_det(x, gamma, y_det, n, chunk, eps, rs_det);
    for (uint32_t i = 0; i < n; ++i) {
      bad += memcmp(&y_hvx[i], &y_det[i], sizeof(float)) ? 1u : 0u;
    }
    for (uint32_t c = 0; c < nchunk; ++c) {
      bad += memcmp(&rs_hvx[c], &rs_det[c], sizeof(float)) ? 1u : 0u;
    }
    for (uint32_t c = 0; c < nchunk; ++c) {
      ref_rmsnorm_f32(x + c * chunk, gamma, y_f32 + c * chunk, chunk, eps);
      ref_rmsnorm_f64(x + c * chunk, gamma, y_f64 + c * chunk, chunk, eps);
    }
    for (uint32_t i = 0; i < n; ++i) {
      const double u32 = ulp_dist(y_det[i], (double)y_f32[i]);
      const double u64 = ulp_dist(y_det[i], y_f64[i]);
      if (u32 > max_ulp32) {
        max_ulp32 = u32;
      }
      if (u64 > max_ulp64) {
        max_ulp64 = u64;
      }
    }
  }
  printf("M1 OPS %s n=%u chunk=%u bad=%u max_ulp(fp32)=%.2f "
         "max_ulp(double)=%.2f\n",
         name, n, chunk, bad, max_ulp32, max_ulp64);
  CHECK(bad == 0u, "%s: HVX differs from m1_ops_det", name);
  CHECK(max_ulp64 <= 4.0, "%s: %.2f ulp from the double reference", name,
        max_ulp64);

  free(x);
  free(gamma);
  free(y_hvx);
  free(y_det);
  free(y_f32);
  free(y_f64);
  free(rs_hvx);
  free(rs_det);
}

/* ---- RoPE -------------------------------------------------------------- */

/** @brief The cos | sin row for one position: the CPU's inv_freq formula,
 *         theta = rope_theta of config.json (5e6), in double then rounded
 *         once. The DSP never computes this; it indexes an uploaded table
 *         (plan 82 section 3.4). */
static void rope_cs(float *cs, uint32_t pos, double theta) {
  for (uint32_t i = 0; i < 32u; ++i) {
    const double inv_freq = pow(theta, -(2.0 * (double)i) / 64.0);
    const double ang = (double)pos * inv_freq;
    cs[i] = (float)cos(ang);
    cs[32u + i] = (float)sin(ang);
  }
}

static void check_rope(void) {
  enum { NQ = 32, NK = 8, N = (NQ + NK) * 64 };
  static const uint32_t positions[] = {0u, 1u, 511u, 1023u, 4095u};
  float *x = malloc(N * sizeof(float));
  float *y_hvx = malloc(N * sizeof(float));
  float *y_det = malloc(N * sizeof(float));
  float cs[64];

  for (int h = 0; h < NQ + NK; ++h) {
    fill_row(x + h * 64, 64, (h < 3) ? h + 1 : 0);
  }

  uint32_t bad = 0, bad_identity = 0;
  double max_rel32 = 0.0, max_rel64 = 0.0;
  for (size_t p = 0; p < sizeof(positions) / sizeof(positions[0]); ++p) {
    rope_cs(cs, positions[p], 5e6);
    memcpy(y_hvx, x, N * sizeof(float));
    memcpy(y_det, x, N * sizeof(float));
    hvx_rope64_f32(y_hvx, NQ, y_hvx + NQ * 64, NK, cs);
    for (int h = 0; h < NQ + NK; ++h) {
      m1_rope64_det(y_det + h * 64, cs);
    }
    uint32_t bad_p = 0;
    for (int i = 0; i < N; ++i) {
      bad_p += memcmp(&y_hvx[i], &y_det[i], sizeof(float)) ? 1u : 0u;
    }
    if (positions[p] == 0u) {
      for (int i = 0; i < N; ++i) {
        bad_identity += memcmp(&y_det[i], &x[i], sizeof(float)) ? 1u : 0u;
      }
    }
    for (int h = 0; h < NQ + NK; ++h) {
      for (int i = 0; i < 32; ++i) {
        const float a = x[h * 64 + i], b = x[h * 64 + 32 + i];
        const float c = cs[i], s = cs[32 + i];
        /* Plain fp32 (compiled -ffp-contract=off) and double. */
        const float r0_32 = a * c - b * s, r1_32 = a * s + b * c;
        const double r0 = (double)a * c - (double)b * s;
        const double r1 = (double)a * s + (double)b * c;
        double tol = ldexp(fabs(a) + fabs(b), -22);
        if (tol < SUBNORMAL_FLOOR) {
          tol = SUBNORMAL_FLOOR;
        }
        const double e0 = fabs((double)y_det[h * 64 + i] - r0) / tol;
        const double e1 = fabs((double)y_det[h * 64 + 32 + i] - r1) / tol;
        const double f0 = fabs((double)y_det[h * 64 + i] - r0_32) / tol;
        const double f1 = fabs((double)y_det[h * 64 + 32 + i] - r1_32) / tol;
        if (e0 > max_rel64) {
          max_rel64 = e0;
        }
        if (e1 > max_rel64) {
          max_rel64 = e1;
        }
        if (f0 > max_rel32) {
          max_rel32 = f0;
        }
        if (f1 > max_rel32) {
          max_rel32 = f1;
        }
      }
    }
    printf("M1 OPS rope64 pos=%u heads=%d+%d bad=%u\n", positions[p], NQ, NK,
           bad_p);
    bad += bad_p;
  }
  printf("M1 OPS rope64 identity_at_pos0 bad=%u err/tol(fp32)=%.3f "
         "err/tol(double)=%.3f (tol = 2^-22 (|a|+|b|))\n",
         bad_identity, max_rel32, max_rel64);
  CHECK(bad == 0u, "rope64: HVX differs from m1_ops_det");
  CHECK(bad_identity == 0u, "rope64: position 0 is not the identity");
  CHECK(max_rel64 <= 1.0, "rope64: %.3f of the tolerance", max_rel64);

  free(x);
  free(y_hvx);
  free(y_det);
}

/* ---- conv1d + gate ----------------------------------------------------- */

enum { CONV_C = 2048, CONV_ROWS = 15, CONV_CHAIN = 8 };

/** @brief g rows for the prefill-shape call: a*c per row, one f32 multiply
 *         (m1_det_mul == Vsf vmpy). */
static void conv_g_rows(const float *abc, float *g, uint32_t rows) {
  for (uint32_t r = 0; r < rows; ++r) {
    const float *a = abc + (size_t)r * 3u * CONV_C;
    const float *c = a + 2u * CONV_C;
    for (uint32_t j = 0; j < CONV_C; ++j) {
      g[(size_t)r * CONV_C + j] = m1_det_mul(a[j], c[j]);
    }
  }
}

/** @brief Chain of CONV_CHAIN M=1 calls from row @a first, HVX and spec,
 *         starting from the same two-row state; returns the bad count of
 *         outputs and states. The HVX outputs land in @a out_hvx. */
static uint32_t conv_chain(const float *abc, const float *w,
                           const float *state2, uint32_t first,
                           float *out_hvx) {
  float *state3 = malloc(3u * CONV_C * sizeof(float));
  float *state_det = malloc(2u * CONV_C * sizeof(float));
  float *out_det = malloc(CONV_C * sizeof(float));
  memcpy(state3, state2, 2u * CONV_C * sizeof(float));
  memset(state3 + 2u * CONV_C, 0xA5, CONV_C * sizeof(float));
  memcpy(state_det, state2, 2u * CONV_C * sizeof(float));

  uint32_t bad = 0;
  for (uint32_t t = 0; t < CONV_CHAIN; ++t) {
    const float *row = abc + (size_t)(first + t) * 3u * CONV_C;
    float *o = out_hvx + (size_t)t * CONV_C;
    memset(o, 0xA5, CONV_C * sizeof(float));
    hvx_conv_gate_m1_f32(row, state3, w, o, CONV_C);
    m1_conv_gate_det(row, state_det, w, out_det, CONV_C);
    for (uint32_t j = 0; j < CONV_C; ++j) {
      bad += memcmp(&o[j], &out_det[j], sizeof(float)) ? 1u : 0u;
    }
    for (uint32_t j = 0; j < 2u * CONV_C; ++j) {
      bad += memcmp(&state3[j], &state_det[j], sizeof(float)) ? 1u : 0u;
    }
  }
  free(state3);
  free(state_det);
  free(out_det);
  return bad;
}

static void check_conv(void) {
  float *abc = malloc((size_t)CONV_ROWS * 3u * CONV_C * sizeof(float));
  float *w = malloc(3u * CONV_C * sizeof(float));
  float *g = malloc((size_t)CONV_ROWS * CONV_C * sizeof(float));
  float *z = malloc((size_t)CONV_ROWS * CONV_C * sizeof(float));
  float *out = malloc((size_t)CONV_CHAIN * CONV_C * sizeof(float));
  float *state2 = calloc(2u * CONV_C, sizeof(float));

  fill_rand(w, 3u * CONV_C, -1.0f, 1.0f);
  for (uint32_t r = 0; r < CONV_ROWS; ++r) {
    /* Rows 1..3 of the first chain and 8..10 of the second are the fixed
       kinds; the fixed row's a | b | c all take the kind. */
    const int kind = (r >= 1u && r <= 3u)    ? (int)r
                     : (r >= 8u && r <= 10u) ? (int)r - 7
                                             : 0;
    fill_row(abc + (size_t)r * 3u * CONV_C, 3u * CONV_C, kind);
  }
  conv_g_rows(abc, g, CONV_ROWS);

  /* (1) M=1 chain from a zero state vs the spec, then vs one prefill-shape
     call over rows 0..7 (t0 = 0). */
  uint32_t bad_spec = conv_chain(abc, w, state2, 0u, out);
  for (uint32_t r = 0; r < CONV_CHAIN; ++r) {
    memcpy(z + (size_t)r * CONV_C, abc + (size_t)r * 3u * CONV_C + CONV_C,
           CONV_C * sizeof(float));
  }
  hvx_conv_gate_f32(z, CONV_C, g, CONV_C, 0u, CONV_CHAIN, CONV_C, w, NULL);
  uint32_t bad_prefill = 0;
  for (uint32_t i = 0; i < (uint32_t)CONV_CHAIN * CONV_C; ++i) {
    bad_prefill += memcmp(&z[i], &out[i], sizeof(float)) ? 1u : 0u;
  }
  printf("M1 OPS conv_gate_m1 C=%d chain=%d from zero state: bad(spec)=%u "
         "bad(prefill t0=0 m=%d)=%u\n",
         CONV_C, CONV_CHAIN, bad_spec, CONV_CHAIN, bad_prefill);
  CHECK(bad_spec == 0u, "conv_gate_m1: HVX differs from m1_ops_det");
  CHECK(bad_prefill == 0u, "conv_gate_m1: chain differs from the prefill");

  /* (2) From the state a 7-row prefill-shape call leaves (rows 5, 6 of g),
     the chain over rows 7..14 vs one call with t0 = 7, m = 8. */
  memcpy(state2, g + 5u * CONV_C, 2u * CONV_C * sizeof(float));
  bad_spec = conv_chain(abc, w, state2, 7u, out);
  for (uint32_t r = 0; r < CONV_CHAIN; ++r) {
    memcpy(z + (size_t)r * CONV_C,
           abc + (size_t)(7u + r) * 3u * CONV_C + CONV_C,
           CONV_C * sizeof(float));
  }
  hvx_conv_gate_f32(z, CONV_C, g, CONV_C, 7u, CONV_CHAIN, CONV_C, w, NULL);
  bad_prefill = 0;
  for (uint32_t i = 0; i < (uint32_t)CONV_CHAIN * CONV_C; ++i) {
    bad_prefill += memcmp(&z[i], &out[i], sizeof(float)) ? 1u : 0u;
  }
  printf("M1 OPS conv_gate_m1 C=%d chain=%d from a 7-row prefill state: "
         "bad(spec)=%u bad(prefill t0=7 m=%d)=%u\n",
         CONV_C, CONV_CHAIN, bad_spec, CONV_CHAIN, bad_prefill);
  CHECK(bad_spec == 0u, "conv_gate_m1 (prefill state): differs from spec");
  CHECK(bad_prefill == 0u,
        "conv_gate_m1 (prefill state): chain differs from the prefill");

  /* (3) Tolerance of the spec, rows 7..14, against plain fp32 in the CPU
     order and against double. */
  double max_rel32 = 0.0, max_rel64 = 0.0;
  for (uint32_t r = 0; r < CONV_CHAIN; ++r) {
    const uint32_t t = 7u + r;
    const float *b = abc + (size_t)t * 3u * CONV_C + CONV_C;
    for (uint32_t j = 0; j < CONV_C; ++j) {
      const float g0 = g[(size_t)t * CONV_C + j];
      const float g1 = g[(size_t)(t - 1u) * CONV_C + j];
      const float g2 = g[(size_t)(t - 2u) * CONV_C + j];
      const float w0 = w[j], w1 = w[CONV_C + j], w2 = w[2u * CONV_C + j];
      const float y32 = w0 * g0 + w1 * g1 + w2 * g2;
      const float o32 = b[j] * y32;
      const double y64 = (double)w0 * g0 + (double)w1 * g1 + (double)w2 * g2;
      const double o64 = (double)b[j] * y64;
      double tol = ldexp(fabs((double)b[j]) *
                           (fabs((double)w0 * g0) + fabs((double)w1 * g1) +
                            fabs((double)w2 * g2)),
                         -21);
      if (tol < SUBNORMAL_FLOOR) {
        tol = SUBNORMAL_FLOOR;
      }
      const float o = out[(size_t)r * CONV_C + j];
      const double e64 = fabs((double)o - o64) / tol;
      const double e32 = fabs((double)o - (double)o32) / tol;
      if (e64 > max_rel64) {
        max_rel64 = e64;
      }
      if (e32 > max_rel32) {
        max_rel32 = e32;
      }
    }
  }
  printf("M1 OPS conv_gate_m1 err/tol(fp32)=%.3f err/tol(double)=%.3f "
         "(tol = 2^-21 |b| sum|w_i g_i|)\n",
         max_rel32, max_rel64);
  CHECK(max_rel64 <= 1.0, "conv_gate_m1: %.3f of the tolerance", max_rel64);

  free(abc);
  free(w);
  free(g);
  free(z);
  free(out);
  free(state2);
}

/* ---- router (#132) ----------------------------------------------------- */

enum { ROUTER_MAX_K = 2048, ROUTER_MAX_E = 32 };

/** @brief A router case's inputs: W [K][E] and its 32-column copy. Row
 *         kind 0 random; 1 exact tie (experts e1 < e2 share a column and a
 *         bias, both biased above the rest after three sure winners, so
 *         they meet at the last selected place); 2 the same with e2's bias
 *         one ulp above e1's. */
static void router_inputs(float *x, float *w, float *w32, float *bias,
                          uint32_t K, uint32_t E, uint32_t top_k, int kind) {
  fill_rand(x, K, -2.0f, 2.0f);
  fill_rand(w, K * E, -0.05f, 0.05f);
  fill_rand(bias, E, -0.01f, 0.01f);
  if (kind != 0) {
    const uint32_t e1 = (g_seed >> 8) % (E - 1u);
    const uint32_t e2 = e1 + 1u + (g_seed >> 16) % (E - 1u - e1);
    for (uint32_t k = 0; k < K; ++k) {
      w[(size_t)k * E + e2] = w[(size_t)k * E + e1];
    }
    for (uint32_t e = 0, n = 0; e < E && n + 1u < top_k; ++e) {
      if (e != e1 && e != e2) {
        bias[e] = 2.0f;
        ++n;
      }
    }
    bias[e1] = 1.0f;
    bias[e2] = kind == 1 ? 1.0f : nextafterf(1.0f, 2.0f);
  }
  for (uint32_t k = 0; k < K; ++k) {
    for (uint32_t e = 0; e < ROUTER_MAX_E; ++e) {
      w32[(size_t)k * ROUTER_MAX_E + e] = e < E ? w[(size_t)k * E + e] : 0.0f;
    }
  }
}

static void check_router_kernel(void) {
  static const uint32_t shapes[3][3] = {
    {2048u, 32u, 4u}, {128u, 4u, 2u}, {64u, 4u, 2u}};
  float *x = malloc(ROUTER_MAX_K * sizeof(float));
  float *w = malloc((size_t)ROUTER_MAX_K * ROUTER_MAX_E * sizeof(float));
  float *w32 = malloc((size_t)ROUTER_MAX_K * ROUTER_MAX_E * sizeof(float));
  float bias[ROUTER_MAX_E], lg_h[ROUTER_MAX_E], lg_d[ROUTER_MAX_E];
  float wt_h[ROUTER_MAX_E], wt_d[ROUTER_MAX_E];
  uint32_t sel_h[ROUTER_MAX_E], sel_d[ROUTER_MAX_E];
  for (int s = 0; s < 3; ++s) {
    const uint32_t K = shapes[s][0], E = shapes[s][1], top_k = shapes[s][2];
    uint32_t bad = 0, ties_low = 0, rows = 0;
    for (int kind = 0; kind < 3; ++kind) {
      for (int rep = 0; rep < 8; ++rep, ++rows) {
        router_inputs(x, w, w32, bias, K, E, top_k, kind);
        memset(lg_h, 0xA5, sizeof(lg_h));
        memset(wt_h, 0xA5, sizeof(wt_h));
        memset(sel_h, 0xA5, sizeof(sel_h));
        hvx_router_topk_f32(x, w32, bias, K, E, top_k, lg_h, sel_h, wt_h);
        m1_router_topk_det(x, w, bias, K, E, top_k, lg_d, sel_d, wt_d);
        bad += memcmp(lg_h, lg_d, E * sizeof(float)) != 0;
        bad += memcmp(sel_h, sel_d, top_k * sizeof(uint32_t)) != 0;
        bad += memcmp(wt_h, wt_d, top_k * sizeof(float)) != 0;
        if (kind == 1) {
          /* the tied pair's lower index is the last one selected */
          uint32_t lo = E, hi = 0;
          for (uint32_t e = 0; e < E; ++e) {
            if (bias[e] == 1.0f) {
              lo = e < lo ? e : lo;
              hi = e > hi ? e : hi;
            }
          }
          ties_low += sel_d[top_k - 1u] == lo && lo < hi;
        }
      }
    }
    printf("ROUTER TOPK K=%u E=%u top_k=%u rows=%u (random, exact tie, "
           "1-ulp near tie) bad=%u tie_to_lowest=%u/8\n",
           K, E, top_k, rows, bad, ties_low);
    CHECK(bad == 0u, "router K=%u: HVX differs from m1_router_topk_det", K);
    CHECK(ties_low == 8u,
          "router K=%u: an exact tie did not go to the lower "
          "index",
          K);
  }
  free(x);
  free(w);
  free(w32);
}

/** @brief buildExpertAssignments (lfm2_moe_layer.cpp) in C: a sequential
 *         dot for the logits (the CPU's BLAS has its own order), expf, a
 *         true divide, and std::partial_sort under the total order score
 *         descending, index ascending -- a full sort gives the same first
 *         top_k. Also returns the 4th / 5th score gap. */
typedef struct {
  float s;
  int e;
} scored_t;
static int scored_cmp(const void *pa, const void *pb) {
  const scored_t *a = pa, *b = pb;
  if (a->s != b->s) {
    return a->s > b->s ? -1 : 1;
  }
  return a->e - b->e;
}
static float router_cpu(const float *x, const float *w, const float *bias,
                        uint32_t K, uint32_t E, uint32_t top_k, uint32_t *sel,
                        float *logits) {
  scored_t sc[ROUTER_MAX_E];
  for (uint32_t e = 0; e < E; ++e) {
    float l = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
      l += x[k] * w[(size_t)k * E + e];
    }
    logits[e] = l;
    const float s = 1.0f / (1.0f + expf(-l));
    sc[e].s = s + bias[e];
    sc[e].e = (int)e;
  }
  qsort(sc, E, sizeof(sc[0]), scored_cmp);
  for (uint32_t r = 0; r < top_k; ++r) {
    sel[r] = (uint32_t)sc[r].e;
  }
  return sc[top_k - 1u].s - sc[top_k].s;
}

static uint32_t sel_mask(const uint32_t *sel, uint32_t top_k) {
  uint32_t m = 0;
  for (uint32_t r = 0; r < top_k; ++r) {
    m |= 1u << sel[r];
  }
  return m;
}

static void check_router_vs_cpu(void) {
  enum { ROWS = 100000, TIE_ROWS = 1000 };
  const uint32_t K = 2048u, E = 32u, top_k = 4u;
  float *x = malloc(K * sizeof(float));
  float *w = malloc((size_t)K * E * sizeof(float));
  float *w32 = malloc((size_t)K * ROUTER_MAX_E * sizeof(float));
  float bias[ROUTER_MAX_E], lg[ROUTER_MAX_E], lg_c[ROUTER_MAX_E];
  float wt[ROUTER_MAX_E];
  uint32_t sel_d[ROUTER_MAX_E], sel_c[ROUTER_MAX_E];
  uint32_t flips = 0, tie_flips = 0;
  float max_gap = 0.0f, max_dlogit = 0.0f;
  /* one weight matrix and bias (a layer), a fresh activation per row */
  router_inputs(x, w, w32, bias, K, E, top_k, 0);
  for (uint32_t row = 0; row < ROWS; ++row) {
    fill_rand(x, K, -2.0f, 2.0f);
    m1_router_topk_det(x, w, bias, K, E, top_k, lg, sel_d, wt);
    const float gap = router_cpu(x, w, bias, K, E, top_k, sel_c, lg_c);
    for (uint32_t e = 0; e < E; ++e) {
      const float d = fabsf(lg[e] - lg_c[e]);
      max_dlogit = d > max_dlogit ? d : max_dlogit;
    }
    if (sel_mask(sel_d, top_k) != sel_mask(sel_c, top_k)) {
      ++flips;
      max_gap = gap > max_gap ? gap : max_gap;
    }
  }
  for (uint32_t row = 0; row < TIE_ROWS; ++row) {
    router_inputs(x, w, w32, bias, K, E, top_k, 1);
    m1_router_topk_det(x, w, bias, K, E, top_k, lg, sel_d, wt);
    router_cpu(x, w, bias, K, E, top_k, sel_c, lg_c);
    tie_flips += sel_mask(sel_d, top_k) != sel_mask(sel_c, top_k);
  }
  /* max_dlogit is the noise scale a flip needs a gap below */
  printf("ROUTER SPEC vs CPU rows=%d set_flips=%u max_gap_at_flip=%.3g "
         "max_logit_diff=%.3g exact_tie_rows=%d tie_flips=%u\n",
         ROWS, flips, (double)max_gap, (double)max_dlogit, TIE_ROWS, tie_flips);
  CHECK(max_gap < 1e-4f, "router spec vs CPU: a flip at a score gap %.3g",
        (double)max_gap);
  CHECK(tie_flips == 0u, "router spec vs CPU: %u exact ties split", tie_flips);
  free(x);
  free(w);
  free(w32);
}

int main(void) {
  check_rmsnorm("rmsnorm", 2048u, 2048u);
  check_rmsnorm("qk_norm_q", 32u * 64u, 64u);
  check_rmsnorm("qk_norm_k", 8u * 64u, 64u);
  check_rope();
  check_conv();
  check_router_kernel();
  check_router_vs_cpu();
  if (g_fail) {
    printf("M1 OPS CHECK FAILED\n");
    return 1;
  }
  printf("ROUTER TOPK BIT-IDENTICAL\n");
  printf("M1 OPS BIT-IDENTICAL\n");
  return 0;
}
