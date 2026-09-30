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
 *     RMSNorm <= 4 ulp of |y|, RoPE (fp16 since #152) <= 2^-8 (|a| +
 *     |b|), conv <= 2^-21 |b| sum|w_i g_i|, each with a floor of two
 *     subnormal quanta (2^-22 for the fp16 RoPE)
 *     so the all-subnormal row (kept, not flushed: rule 24) is judged on
 *     what f32 can hold.
 *
 * The norms' row scale (#164) has a third half: the spec against an
 * independent model of the Android CPU's RMSNorm written from its aarch64
 * disassembly (fmaf chains, sqrtf, 1.0f /), on the fixed kinds and 20 000
 * random rows over magnitudes 2^-20 .. 2^20 at chunk 2048 and 64, plus four
 * mutants that must differ from it (NORM CPU-ORDER); the integer sqrt and
 * reciprocal against sqrtf and 1.0f / on every 251st positive normal and
 * +-4 ulp around every power of two (SQRT/RECIP RN; M1_NORM_EXHAUSTIVE=1
 * for every positive normal, about a minute); and a replay mode,
 *   m1_ops_host_check --replay <gamma_rms.f32> <gamma_qk.f32> <norm.bin>...
 * that runs the kernel on dev/norm-shadow dumps with the true gammas and
 * memcmp's it against the CPU's dumped output.
 *
 * Inputs: an LCG from a fixed seed, plus three fixed rows / heads: all
 * zeros, all subnormal (+-1e-39) and large (|x| ~ 1e4).
 *
 * The router (#132; the Android CPU's order since #132 PR 2) has no
 * tolerance half: it is memcmp'd kernel against spec (logits, selection
 * order, weights) at three shapes on random, exact-tie and 1-ulp near-tie
 * rows; then the spec against an independent copy of the CPU's router
 * (fmaf chains, glibc expf, true divides, the -ffast-math weight sum) bit
 * for bit on LFM2.5-shaped rows, with three mutants that must be caught.
 * The bionic expf port is pinned to glibc's same algorithm, the SwiGLU
 * spec to an independent copy of neon::swiglu (three mutants), and
 * argmax_first to std::max_element's tie rule.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_conv_gate_f32.h"
#include "hvx_m1_ops_f32.h"
#include "m1_ops_det.h"

/* The pool, run in place of the pthread pool: min(n_units, 3) lanes called
   one after another on the caller -- lanes are independent by the pool's
   contract, and three lanes over the router's four chain groups give lane
   0 two groups, so the lane start and step (#132 PR 2) are exercised. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  const uint32_t n = n_units < 3u ? n_units : 3u;
  (void)pool;
  for (uint32_t i = 0; i < n; ++i) {
    func(n, i, ctx);
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

/* ---- RMSNorm row scale vs the Android CPU (#164) ----------------------- */

/* Written from the aarch64 disassembly of the shipped libnntrainer.so's
   neon::rms_norm_wrt_width_fp32_intrinsic (plan 164 section 0), NOT from
   m1_ops_det.h: four float32x4 fmla accumulators, faddp twice per vector,
   the four in order, fdiv by W, fadd eps, fsqrt, fdiv 1 / q. Mutant 0 is
   that; 1-4 are what the check must tell apart from it. */
enum {
  NORM_CPU = 0,
  NORM_MUT_TREE32, /* the pre-#164 spec's 32-lane non-fused tree sum */
  NORM_MUT_NOFMA,  /* the 16 chains as a multiply, then an add */
  NORM_MUT_RSQRT,  /* the pre-#164 magic-seed Newton rsqrt for fsqrt+fdiv */
  NORM_MUT_ORDER,  /* h0 + (h1 + (h2 + h3)) */
  NORM_N_MUT = 4
};
static float norm_rsqrt_newton(float d) {
  volatile float y = m1_det_float(0x5F3759DFu - (m1_det_bits(d) >> 1));
  volatile float h = d * 0.5f;
  for (int it = 0; it < 3; ++it) {
    volatile float t = h * y;
    t = t * y;
    t = 1.5f - t;
    y = y * t;
  }
  return y;
}
static float norm_cpu_r(const float *x, uint32_t W, float eps, int mut) {
  volatile float s;
  if (mut == NORM_MUT_TREE32) {
    volatile float a[32] = {0};
    for (uint32_t i = 0; i < W; i += 32u) {
      for (uint32_t j = 0; j < 32u; ++j) {
        volatile float p = x[i + j] * x[i + j];
        a[j] = a[j] + p;
      }
    }
    for (uint32_t st = 16u; st >= 1u; st >>= 1) {
      for (uint32_t j = 0; j < st; ++j) {
        a[j] = a[j] + a[j + st];
      }
    }
    s = a[0];
  } else {
    volatile float acc[4][4] = {{0}};
    for (uint32_t i = 0; i < W; i += 16u) {
      for (int v = 0; v < 4; ++v) {
        for (int l = 0; l < 4; ++l) {
          const float e = x[i + 4u * v + l];
          if (mut == NORM_MUT_NOFMA) {
            volatile float p = e * e;
            acc[v][l] = acc[v][l] + p;
          } else {
            acc[v][l] = fmaf(e, e, acc[v][l]);
          }
        }
      }
    }
    volatile float h[4];
    for (int v = 0; v < 4; ++v) {
      volatile float p = acc[v][0] + acc[v][1], q = acc[v][2] + acc[v][3];
      h[v] = p + q;
    }
    if (mut == NORM_MUT_ORDER) {
      volatile float t = h[2] + h[3];
      t = h[1] + t;
      s = h[0] + t;
    } else {
      s = h[0] + h[1];
      s = s + h[2];
      s = s + h[3];
    }
  }
  volatile float mean = s / (float)W;
  volatile float d = eps + mean;
  if (mut == NORM_MUT_RSQRT) {
    return norm_rsqrt_newton(d);
  }
  volatile float q = sqrtf(d);
  volatile float r = 1.0f / q;
  return r;
}

/** @brief One row of the sweep: the spec's r and y against the CPU model's,
 *         and each mutant's r against the model's (a difference is a catch). */
static uint32_t norm_row(const float *x, const float *gamma, float *y,
                         uint32_t W, float eps, uint32_t *caught) {
  const float eps_ = eps;
  const float r_cpu = norm_cpu_r(x, W, eps_, NORM_CPU);
  const float r_det = m1_rmsnorm_chunk_det(x, gamma, y, W, eps_);
  uint32_t bad = memcmp(&r_cpu, &r_det, sizeof(float)) != 0;
  for (uint32_t i = 0; i < W; ++i) {
    volatile float t = x[i] * r_cpu;
    const float yc = t * gamma[i];
    bad += memcmp(&yc, &y[i], sizeof(float)) != 0;
  }
  for (int m = 1; m <= NORM_N_MUT; ++m) {
    const float r_m = norm_cpu_r(x, W, eps_, m);
    caught[m - 1] += memcmp(&r_m, &r_cpu, sizeof(float)) != 0;
  }
  return bad;
}

static void check_norm_cpu_order(void) {
  enum { ROWS = 20000 };
  static const uint32_t widths[2] = {2048u, 64u};
  const float eps = 1e-5f;
  float *x = malloc(2048u * sizeof(float));
  float *g = malloc(2048u * sizeof(float));
  float *y = malloc(2048u * sizeof(float));
  uint32_t caught[NORM_N_MUT] = {0, 0, 0, 0}, bad = 0, rows = 0;
  fill_rand(g, 2048u, 0.5f, 1.5f);
  for (int w = 0; w < 2; ++w) {
    const uint32_t W = widths[w];
    for (int kind = 1; kind <= 3; ++kind, ++rows) {
      fill_row(x, W, kind);
      bad += norm_row(x, g, y, W, eps, caught);
    }
    /* 2048: 2 000 rows; 64: 18 000 -- one magnitude 2^e, e in [-20, 20],
       per row, elements +-[0.5, 1) 2^e */
    const uint32_t n = w == 0 ? ROWS / 10 : ROWS - ROWS / 10;
    for (uint32_t k = 0; k < n; ++k, ++rows) {
      const int e = (int)(k % 41u) - 20;
      for (uint32_t i = 0; i < W; ++i) {
        const float m = frand(0.5f, 1.0f);
        x[i] = ldexpf((g_seed >> 31) ? -m : m, e);
      }
      bad += norm_row(x, g, y, W, eps, caught);
    }
  }
  uint32_t n_caught = 0;
  for (int m = 0; m < NORM_N_MUT; ++m) {
    n_caught += caught[m] != 0u;
  }
  printf("M1 OPS NORM spec vs CPU model rows=%u bad=%u; mutant r differs on "
         "tree32=%u nofma=%u rsqrt=%u order=%u rows\n",
         rows, bad, caught[0], caught[1], caught[2], caught[3]);
  CHECK(bad == 0u, "norm: m1_ops_det differs from the CPU model");
  CHECK(n_caught == NORM_N_MUT, "norm: a mutant went unseen");
  if (bad == 0u && n_caught == NORM_N_MUT) {
    printf("M1 OPS NORM CPU-ORDER OK mutants=%u/%d\n", n_caught, NORM_N_MUT);
  }
  free(x);
  free(g);
  free(y);
}

/** @brief sqrt_rn and recip_rn against sqrtf and 1.0f / for one input;
 *         recip only where 1/d is normal (d < 2^126). */
static void rn_one(uint32_t u, uint64_t *n, uint64_t *bad_s, uint64_t *bad_r) {
  const float d = m1_det_float(u);
  volatile float s = sqrtf(d);
  const float sd = m1_sqrt_rn_det(d);
  if (memcmp(&sd, (const void *)&s, sizeof(float))) {
    if ((*bad_s)++ < 5u) {
      printf("sqrt_rn bad at %08x\n", u);
    }
  }
  if (u < 0x7e800000u) {
    volatile float r = 1.0f / d;
    const float rd = m1_recip_rn_det(d);
    if (memcmp(&rd, (const void *)&r, sizeof(float))) {
      if ((*bad_r)++ < 5u) {
        printf("recip_rn bad at %08x\n", u);
      }
    }
  }
  ++*n;
}

static void check_sqrt_recip_rn(void) {
  uint64_t n = 0, bad_s = 0, bad_r = 0;
  const char *ex = getenv("M1_NORM_EXHAUSTIVE");
  const int exhaustive = ex && ex[0] == '1';
  const uint32_t stride = exhaustive ? 1u : 251u;
  for (uint32_t u = 0x00800000u; u < 0x7f800000u; u += stride) {
    rn_one(u, &n, &bad_s, &bad_r);
  }
  /* +-4 ulp around every power of two: the exponent edges, where the
     mantissa wraps and the 2^24 rounding carry happens */
  for (uint32_t e = 1u; e < 255u; ++e) {
    for (int k = -4; k <= 4; ++k) {
      const int64_t u = (int64_t)(e << 23) + k;
      if (u >= 0x00800000 && u < 0x7f800000) {
        rn_one((uint32_t)u, &n, &bad_s, &bad_r);
      }
    }
  }
  /* specials: +inf -> sqrt +inf, recip +0 */
  const float inf = m1_det_float(0x7f800000u);
  const int inf_ok = m1_det_bits(m1_sqrt_rn_det(inf)) == 0x7f800000u &&
                     m1_det_bits(m1_recip_rn_det(inf)) == 0u;
  printf("M1 OPS SQRT/RECIP RN %s inputs=%llu sqrt_bad=%llu recip_bad=%llu "
         "inf_ok=%d\n",
         exhaustive ? "EXHAUSTIVE" : "every 251st + exponent edges",
         (unsigned long long)n, (unsigned long long)bad_s,
         (unsigned long long)bad_r, inf_ok);
  CHECK(bad_s == 0u && bad_r == 0u && inf_ok, "sqrt_rn / recip_rn");
  if (bad_s == 0u && bad_r == 0u && inf_ok) {
    printf("M1 OPS SQRT/RECIP RN OK\n");
  }
}

/* ---- replay of dev/norm-shadow dumps ----------------------------------- */

static float *read_all(const char *path, size_t n_floats) {
  FILE *f = fopen(path, "rb");
  float *b = malloc(n_floats * sizeof(float));
  const size_t got = f ? fread(b, sizeof(float), n_floats, f) : 0u;
  if (f) {
    fclose(f);
  }
  if (got != n_floats) {
    printf("REPLAY: %s: %zu of %zu floats\n", path, got, n_floats);
    free(b);
    return NULL;
  }
  return b;
}

/**
 * @brief Runs hvx_rmsnorm_f32 (on hvx_emu) on every record of each dump and
 *        memcmp's it with the CPU's output in the record.
 *
 * Record (dev/norm-shadow): u32 tag, pos, n_in, n_out; f32 in[n_in],
 * cpu[n_out], other[n_out]. Tag 0 (RMSNORM on HTP) and 2 (on CPU): one
 * 2048 row, gamma = the call's, in order operator_norm, ffn_norm per layer
 * then embedding_norm (49 per step). Tag 1: in = q | k | v, cpu = the
 * CPU's normed q | k, 64 per head, gamma = q | k of the step's next
 * attention layer (6 per step). gamma_rms.f32: 49 x 2048; gamma_qk.f32:
 * 6 x (q 64 | k 64); both from hf/model.safetensors, bf16 -> f32.
 */
static int replay(int argc, char **argv) {
  enum { W = 2048, N_RMS = 49, N_QK = 6, HD = 64 };
  const float eps = 1e-5f;
  float *grms = read_all(argv[0], (size_t)N_RMS * W);
  float *gqk = read_all(argv[1], (size_t)N_QK * 2 * HD);
  if (!grms || !gqk) {
    return 2;
  }
  unsigned long ok[3] = {0, 0, 0}, n[3] = {0, 0, 0};
  float *in = malloc(8192u * sizeof(float)), *cpu = malloc(8192u * 4u);
  float *other = malloc(8192u * 4u), *y = malloc(8192u * 4u);
  for (int fi = 2; fi < argc; ++fi) {
    FILE *f = fopen(argv[fi], "rb");
    if (!f) {
      printf("REPLAY: cannot open %s\n", argv[fi]);
      return 2;
    }
    unsigned long fok[3] = {0, 0, 0}, fn[3] = {0, 0, 0};
    uint32_t h[4], k_rms = 0, k_qk = 0;
    while (fread(h, sizeof(uint32_t), 4, f) == 4) {
      if (h[2] > 8192u || h[3] > 8192u || h[0] > 2u ||
          fread(in, 4, h[2], f) != h[2] || fread(cpu, 4, h[3], f) != h[3] ||
          fread(other, 4, h[3], f) != h[3]) {
        printf("REPLAY: %s: bad record\n", argv[fi]);
        return 2;
      }
      if (h[0] != 1u) {
        if (h[2] != W || h[3] != W) {
          printf("REPLAY: %s: RMSNORM record of %u\n", argv[fi], h[2]);
          return 2;
        }
        hvx_rmsnorm_f32(in, grms + (size_t)(k_rms++ % N_RMS) * W, y, W, W, eps,
                        NULL);
        ++fn[h[0]];
        fok[h[0]] += !memcmp(y, cpu, W * sizeof(float));
      } else {
        const uint32_t wk = h[3] <= h[2] ? h[2] - h[3] : 0u;
        const uint32_t wq = wk <= h[3] ? h[3] - wk : 0u;
        if (wk == 0u || wq == 0u || wq % HD != 0u || wk % HD != 0u) {
          printf("REPLAY: %s: QK_NORM record of %u / %u\n", argv[fi], h[2],
                 h[3]);
          return 2;
        }
        const float *gq = gqk + (size_t)(k_qk++ % N_QK) * 2u * HD;
        hvx_rmsnorm_f32(in, gq, y, wq, HD, eps, NULL);
        hvx_rmsnorm_f32(in + wq, gq + HD, y + wq, wk, HD, eps, NULL);
        for (uint32_t hd = 0; hd < h[3] / HD; ++hd) {
          ++fn[1];
          fok[1] += !memcmp(y + hd * HD, cpu + hd * HD, HD * sizeof(float));
        }
      }
    }
    fclose(f);
    printf("REPLAY %s rms(tag0)=%lu/%lu rms_cpu(tag2)=%lu/%lu "
           "qk_heads=%lu/%lu\n",
           argv[fi], fok[0], fn[0], fok[2], fn[2], fok[1], fn[1]);
    for (int t = 0; t < 3; ++t) {
      ok[t] += fok[t];
      n[t] += fn[t];
    }
  }
  printf("REPLAY rms=%lu/%lu qk_heads=%lu/%lu (rms_cpu=%lu/%lu)\n", ok[0], n[0],
         ok[1], n[1], ok[2], n[2]);
  free(in);
  free(cpu);
  free(other);
  free(y);
  free(grms);
  free(gqk);
  return ok[0] == n[0] && ok[1] == n[1] && ok[2] == n[2] ? 0 : 1;
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
  double max_rel = 0.0;
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
      /* cos 1, sin 0: the fp16 input itself (by value: a -0 input and a
         negative b give +0, as the CPU's fsub does). */
      for (int i = 0; i < N; ++i) {
        bad_identity += y_det[i] != attn_m1_det_rne16(x[i]);
      }
    }
    for (int h = 0; h < NQ + NK; ++h) {
      for (int i = 0; i < 32; ++i) {
        const float a = x[h * 64 + i], b = x[h * 64 + 32 + i];
        const float c = cs[i], s = cs[32 + i];
        const double r0 = (double)a * c - (double)b * s;
        const double r1 = (double)a * s + (double)b * c;
        /* fp16: the operands and three steps rounded, 2^-11 each. */
        double tol = ldexp(fabs(a) + fabs(b), -8);
        if (tol < ldexp(1.0, -22)) {
          tol = ldexp(1.0, -22);
        }
        const double e0 = fabs((double)y_det[h * 64 + i] - r0) / tol;
        const double e1 = fabs((double)y_det[h * 64 + 32 + i] - r1) / tol;
        max_rel = e0 > max_rel ? e0 : max_rel;
        max_rel = e1 > max_rel ? e1 : max_rel;
      }
    }
    printf("M1 OPS rope64 pos=%u heads=%d+%d bad=%u\n", positions[p], NQ, NK,
           bad_p);
    bad += bad_p;
  }
  printf("M1 OPS rope64 (fp16, #152) identity_at_pos0 bad=%u "
         "err/tol(double)=%.3f (tol = 2^-8 (|a|+|b|), floor 2^-22)\n",
         bad_identity, max_rel);
  CHECK(bad == 0u, "rope64: HVX differs from m1_ops_det");
  CHECK(bad_identity == 0u, "rope64: position 0 is not the identity");
  CHECK(max_rel <= 1.0, "rope64: %.3f of the tolerance", max_rel);

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

/** @brief [#132 E5c] The Android CPU's M=1 conv + gate written out on
 *  its own (not the spec): g = a*c, y = fmaf(w2, s0, fmaf(w1, s1, w0*g)),
 *  out = b*y, over the chain from row @a first; the bad count against the
 *  HVX outputs @a out_hvx of conv_chain. */
static uint32_t conv_cpu_model(const float *abc, const float *w,
                               const float *state2, uint32_t first,
                               const float *out_hvx) {
  float *s0 = malloc(CONV_C * sizeof(float));
  float *s1 = malloc(CONV_C * sizeof(float));
  memcpy(s0, state2, CONV_C * sizeof(float));
  memcpy(s1, state2 + CONV_C, CONV_C * sizeof(float));
  uint32_t bad = 0;
  for (uint32_t t = 0; t < CONV_CHAIN; ++t) {
    const float *a = abc + (size_t)(first + t) * 3u * CONV_C;
    const float *b = a + CONV_C, *c = a + 2u * CONV_C;
    for (uint32_t j = 0; j < CONV_C; ++j) {
      volatile float g = a[j] * c[j];
      volatile float y0 = w[j] * g;
      volatile float y1 = fmaf(w[CONV_C + j], s1[j], y0);
      volatile float y2 = fmaf(w[2u * CONV_C + j], s0[j], y1);
      volatile float o = b[j] * y2;
      const float of = o;
      bad +=
        memcmp(&of, &out_hvx[(size_t)t * CONV_C + j], sizeof(float)) ? 1u : 0u;
      s0[j] = s1[j];
      s1[j] = g;
    }
  }
  free(s0);
  free(s1);
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

  /* (1) M=1 chain from a zero state vs the spec, and the spec vs an
     independent model of the Android CPU's decode (neon
     causal_depthwise_conv1d_k3_decode: vmulq, then two vfmaq -- fmaf
     here -- then the gate). [#132 E5c] No longer against the prefill
     kernel: hvx_conv_gate_f32 is unfused, which is not the CPU's order. */
  uint32_t bad_spec = conv_chain(abc, w, state2, 0u, out);
  uint32_t bad_cpu = conv_cpu_model(abc, w, state2, 0u, out);
  printf("M1 OPS conv_gate_m1 C=%d chain=%d from zero state: bad(spec)=%u "
         "bad(cpu fma model)=%u\n",
         CONV_C, CONV_CHAIN, bad_spec, bad_cpu);
  CHECK(bad_spec == 0u, "conv_gate_m1: HVX differs from m1_ops_det");
  CHECK(bad_cpu == 0u, "conv_gate_m1: differs from the CPU's fused order");

  /* (2) From the state a 7-row prefill leaves (rows 5, 6 of g). */
  memcpy(state2, g + 5u * CONV_C, 2u * CONV_C * sizeof(float));
  bad_spec = conv_chain(abc, w, state2, 7u, out);
  bad_cpu = conv_cpu_model(abc, w, state2, 7u, out);
  printf("M1 OPS conv_gate_m1 C=%d chain=%d from a 7-row prefill state: "
         "bad(spec)=%u bad(cpu fma model)=%u\n",
         CONV_C, CONV_CHAIN, bad_spec, bad_cpu);
  CHECK(bad_spec == 0u, "conv_gate_m1 (prefill state): differs from spec");
  CHECK(bad_cpu == 0u,
        "conv_gate_m1 (prefill state): differs from the CPU's fused order");
  (void)z;

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
        hvx_router_topk_f32(x, w32, bias, K, E, top_k, lg_h, sel_h, wt_h, NULL);
        m1_router_cpu_det(x, w, bias, K, E, top_k, lg_d, sel_d, wt_d);
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
    CHECK(bad == 0u, "router K=%u: HVX differs from m1_router_cpu_det", K);
    CHECK(ties_low == 8u,
          "router K=%u: an exact tie did not go to the lower "
          "index",
          K);
  }
  free(x);
  free(w);
  free(w32);
}

/**
 * @brief The Android CPU's router written independently from its
 *        disassembly (plan 132 section 0.2), with the host's own fmaf, a
 *        true divide and glibc expf, plus one mutant per rejected order:
 *        mut 1 the merged spec's four partial sums, 2 an unfused chain, 3
 *        the wsum in selection order ((s0 + s1) + s2) + s3.
 */
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
static void router_ref(const float *x, const float *w, const float *bias,
                       uint32_t K, uint32_t E, uint32_t top_k, int mut,
                       float *logits, uint32_t *sel, float *weight) {
  scored_t sc[ROUTER_MAX_E];
  float sig[ROUTER_MAX_E];
  for (uint32_t e = 0; e < E; ++e) {
    volatile float l = 0.0f;
    if (mut == 1) {
      volatile float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (uint32_t k = 0; k < K; ++k) {
        volatile float p = x[k] * w[(size_t)k * E + e];
        a[k & 3u] = a[k & 3u] + p;
      }
      volatile float h0 = a[0] + a[1], h1 = a[2] + a[3];
      l = h0 + h1;
    } else {
      for (uint32_t k = 0; k < K; ++k) {
        if (mut == 2) {
          volatile float p = x[k] * w[(size_t)k * E + e];
          l = l + p;
        } else {
          l = fmaf(x[k], w[(size_t)k * E + e], l);
        }
      }
    }
    logits[e] = l;
    volatile float d = expf(-l) + 1.0f;
    volatile float sg = 1.0f / d;
    sig[e] = sg;
    volatile float scv = sg + bias[e];
    sc[e].s = scv;
    sc[e].e = (int)e;
  }
  qsort(sc, E, sizeof(sc[0]), scored_cmp);
  volatile float wsum;
  if (mut == 3) {
    wsum = 0.0f;
    for (uint32_t r = 0; r < top_k; ++r) {
      wsum = wsum + sig[sc[r].e];
    }
  } else {
    volatile float a0 = 0.0f, a1 = 0.0f;
    uint32_t r = 0;
    for (; r + 1u < top_k; r += 2u) {
      a0 = sig[sc[r].e] + a0;
      a1 = sig[sc[r + 1u].e] + a1;
    }
    wsum = a1 + a0;
    if (r < top_k) {
      wsum = sig[sc[r].e] + wsum;
    }
  }
  volatile float den = wsum + 1e-6f;
  volatile float inv = 1.0f / den;
  for (uint32_t r = 0; r < top_k; ++r) {
    sel[r] = (uint32_t)sc[r].e;
    volatile float wt = sig[sc[r].e] * inv;
    weight[r] = wt;
  }
}

/** @brief The spec against the independent CPU reference on LFM2.5-shaped
 *         rows, bit for bit (logits, selection, weights); every mutant
 *         must be caught on at least one row. */
static void check_router_vs_cpu(void) {
  enum { ROWS = 5000, TIE_ROWS = 500, MUTS = 3 };
  const uint32_t K = 2048u, E = 32u, top_k = 4u;
  float *x = malloc(K * sizeof(float));
  float *w = malloc((size_t)K * E * sizeof(float));
  float *w32 = malloc((size_t)K * ROUTER_MAX_E * sizeof(float));
  float bias[ROUTER_MAX_E], lg[ROUTER_MAX_E], lg_c[ROUTER_MAX_E];
  float wt[ROUTER_MAX_E], wt_c[ROUTER_MAX_E];
  uint32_t sel_d[ROUTER_MAX_E], sel_c[ROUTER_MAX_E];
  uint32_t bad = 0, caught[MUTS + 1] = {0};
  router_inputs(x, w, w32, bias, K, E, top_k, 0);
  for (uint32_t row = 0; row < ROWS + TIE_ROWS; ++row) {
    if (row < ROWS) {
      /* magnitudes 2^-20 .. 2^20 across rows: the chain's rounding at
         every scale */
      const float m = ldexpf(1.0f, (int)(row % 41u) - 20);
      fill_rand(x, K, -2.0f * m, 2.0f * m);
    } else {
      router_inputs(x, w, w32, bias, K, E, top_k, 1);
    }
    m1_router_cpu_det(x, w, bias, K, E, top_k, lg, sel_d, wt);
    for (int mut = 0; mut <= MUTS; ++mut) {
      router_ref(x, w, bias, K, E, top_k, mut, lg_c, sel_c, wt_c);
      const int diff = memcmp(lg, lg_c, E * sizeof(float)) != 0 ||
                       memcmp(sel_d, sel_c, top_k * sizeof(uint32_t)) != 0 ||
                       memcmp(wt, wt_c, top_k * sizeof(float)) != 0;
      if (mut == 0) {
        bad += diff;
      } else {
        caught[mut] += diff;
      }
    }
  }
  printf("ROUTER CPU-ORDER rows=%d (+%d exact ties) bad=%u mutants caught: "
         "4-partial=%u unfused=%u wsum-in-order=%u\n",
         ROWS, TIE_ROWS, bad, caught[1], caught[2], caught[3]);
  CHECK(bad == 0u, "router spec differs from the CPU reference on %u rows",
        bad);
  for (int mut = 1; mut <= MUTS; ++mut) {
    CHECK(caught[mut] > 0u, "router mutant %d not caught", mut);
  }
  free(x);
  free(w);
  free(w32);
}

/* ---- bionic expf port, swiglu, argmax (#132 PR 2) ----------------------- */

/** @brief The port against the host's glibc expf, which is the same Arm
 *         optimized-routines algorithm plus a correction on two inputs the
 *         algorithm misrounds (found by the full 2^32 sweep, 2026-09-29).
 *         Every 61st bit pattern here, and those two by name. The device
 *         gtest ExpfBionic.* compares all 2^32 against bionic itself. */
static void check_expf_port(void) {
  static const uint32_t glibc_fixed[2] = {0x4202422fu, 0xc27c65d9u};
  uint64_t n = 0, bad = 0;
  for (uint64_t u = 0; u < (1ull << 32); u += 61u) {
    float x;
    const uint32_t b = (uint32_t)u;
    memcpy(&x, &b, sizeof(x));
    if (x != x || b == glibc_fixed[0] || b == glibc_fixed[1]) {
      continue;
    }
    const float p = m1_expf_bionic_det(x), g = expf(x);
    ++n;
    bad += memcmp(&p, &g, sizeof(p)) != 0;
  }
  uint32_t fixed_ulp = 0;
  for (int i = 0; i < 2; ++i) {
    float x;
    memcpy(&x, &glibc_fixed[i], sizeof(x));
    const float p = m1_expf_bionic_det(x), g = expf(x);
    fixed_ulp += (uint32_t)abs((int32_t)(m1_det_bits(p) - m1_det_bits(g)));
  }
  printf("EXPF PORT vs glibc inputs=%llu bad=%llu glibc-corrected pair "
         "ulp=%u (0 or 2: one each way if this glibc corrects them)\n",
         (unsigned long long)n, (unsigned long long)bad, fixed_ulp);
  CHECK(bad == 0u, "expf port differs from glibc on %llu inputs",
        (unsigned long long)bad);
  CHECK(fixed_ulp <= 2u, "expf port: the glibc pair is %u ulp off", fixed_ulp);
}

/** @brief neon::swiglu as the binary runs it, written independently with
 *         plain float operators (this file is built with
 *         -ffp-contract=off) and fmaf for the one fused step; mut 1 fuses
 *         every polynomial step, 2 is libm expf (the scalar tail's
 *         std::exp), 3 multiplies by the reciprocal instead of dividing.
 *         Splitting the fused fx step is no mutant: after the floor it is
 *         bit-equivalent (0 of 1017966 inputs within 2000 ulps of every
 *         floor boundary differ, checked 2026-09-29). */
static float swiglu_ref(float y, float z, int mut) {
  volatile float x = -y;
  x = fminf(x, 88.3762626647949f);
  x = fmaxf(x, -88.3762626647949f);
  volatile float fx = fmaf(x, 1.44269504088896341f, 0.5f);
  volatile float t = (float)(int32_t)fx;
  fx = (t > fx) ? t - 1.0f : t - 0.0f;
  volatile float a = fx * 0.693359375f;
  x = x - a;
  volatile float bq = fx * -2.12194440e-4f;
  x = x - bq;
  volatile float zz = x * x, p;
  static const float c[6] = {1.9875691500E-4f, 1.3981999507E-3f,
                             8.3334519073E-3f, 4.1665795894E-2f,
                             1.6666665459E-1f, 5.0000001201E-1f};
  p = c[0];
  for (int i = 1; i < 6; ++i) {
    if (mut == 1) {
      p = fmaf(p, x, c[i]);
    } else {
      volatile float m = p * x;
      p = m + c[i];
    }
  }
  p = p * zz;
  p = p + x;
  p = p + 1.0f;
  uint32_t pb = (uint32_t)((int32_t)fx + 127) << 23;
  float pow2n;
  memcpy(&pow2n, &pb, sizeof(pow2n));
  volatile float e = p * pow2n;
  if (mut == 2) {
    e = expf(-y);
  }
  volatile float den = e + 1.0f;
  volatile float q;
  if (mut == 3) {
    volatile float r = 1.0f / den;
    q = y * r;
  } else {
    q = y / den;
  }
  volatile float out = q * z;
  return out;
}

static void check_swiglu_argmax(void) {
  enum { N = 7168, ROWS = 64, MUTS = 3 };
  float *y = malloc(N * sizeof(float)), *z = malloc(N * sizeof(float));
  float *o = malloc(N * sizeof(float));
  uint32_t bad = 0, caught[MUTS + 1] = {0};
  for (int row = 0; row < ROWS; ++row) {
    /* the dense FFN's gate range plus the clamp and far tails */
    const float span = row < 16 ? 8.0f : row < 48 ? 100.0f : 1e4f;
    fill_rand(y, N, -span, span);
    fill_rand(z, N, -4.0f, 4.0f);
    if (row == 1) {
      for (uint32_t i = 0; i < N; ++i) {
        y[i] = (i & 1u) ? -1e-39f : 1e-39f;
      }
    }
    if (row == 2) {
      /* -y a few ulps either side of every floor boundary of fx,
         (k - 0.5) / log2(e) */
      for (uint32_t i = 0; i < N; ++i) {
        const int k = (int)(i / 56u) - 127;
        const float x0 = (float)(((double)k - 0.5) / 1.4426950408889634);
        y[i] = -m1_det_float(m1_det_bits(x0) + (uint32_t)(i % 56u) - 28u);
      }
    }
    m1_swiglu_cpu_det(y, z, o, N);
    for (uint32_t i = 0; i < N; ++i) {
      for (int mut = 0; mut <= MUTS; ++mut) {
        const float r = swiglu_ref(y[i], z[i], mut);
        const int diff = memcmp(&r, &o[i], sizeof(r)) != 0;
        if (mut == 0) {
          bad += diff;
        } else {
          caught[mut] += diff;
        }
      }
    }
  }
  printf("SWIGLU CPU-ORDER elems=%d bad=%u mutants caught: fused-poly=%u "
         "libm-expf=%u recip-mul=%u\n",
         N * ROWS, bad, caught[1], caught[2], caught[3]);
  CHECK(bad == 0u, "swiglu spec differs from the CPU reference");
  for (int mut = 1; mut <= MUTS; ++mut) {
    CHECK(caught[mut] > 0u, "swiglu mutant %d not caught", mut);
  }
  /* argmax: first of equal maxima; a later larger one wins */
  float v[8] = {1.0f, 3.0f, 2.0f, 3.0f, -1.0f, 3.0f, 0.0f, 2.5f};
  const uint32_t i0 = m1_argmax_first(v, 8);
  v[6] = 4.0f;
  const uint32_t i1 = m1_argmax_first(v, 8);
  printf("ARGMAX first-of-ties=%u later-max=%u\n", i0, i1);
  CHECK(i0 == 1u && i1 == 6u, "argmax_first");
  free(y);
  free(z);
  free(o);
}

/* ---- [#132 Part B E5f] pooled SwiGLU (hardware divide), vector argmax -- */

/** @brief hvx_swiglu_cpu_f32 on the stand-in's 3 lanes against
 * m1_swiglu_cpu_det (its integer division) over random rows of three spans and
 * the exp_ps clamp's edges; hvx_argmax_first_f32 against m1_argmax_first on
 * random rows with planted ties, -0 / +0 maxima, -inf rows and a NaN (the spec
 *  path). The host's divide is IEEE RN, so this holds the kernel's order;
 *  the DSP's divide is checked on the device (HvxFcQ4.ScalarDivide). */
static void check_swiglu_argmax_kernels(void) {
  enum { N = 7168, V = 4096 };
  hvx_worker_pool *pool = NULL; /* the run stand-in above: 3 lanes */
  float *y = malloc(N * sizeof(float)), *z = malloc(N * sizeof(float));
  float *o = malloc(N * sizeof(float)), *r = malloc(N * sizeof(float));
  uint32_t bad_s = 0, bad_a = 0;
  for (int t = 0; t < 12; ++t) {
    const float span = t < 4 ? 8.0f : t < 8 ? 100.0f : 1e4f;
    fill_rand(y, N, -span, span);
    fill_rand(z, N, -4.0f, 4.0f);
    if (t == 11) {
      for (uint32_t i = 0; i < N; ++i)
        y[i] = (i % 3 == 0) ? 88.37626f : (i % 3 == 1) ? -88.37627f : -0.0f;
    }
    hvx_swiglu_cpu_f32(y, z, o, N, pool);
    m1_swiglu_cpu_det(y, z, r, N);
    bad_s += memcmp(o, r, N * sizeof(float)) != 0;
  }
  float *x = malloc(V * sizeof(float));
  for (int t = 0; t < 40; ++t) {
    fill_rand(x, V, -30.0f, 30.0f);
    if (t % 4 == 1) { /* a tie: the first maximum must win */
      const uint32_t a = (uint32_t)(t * 97) % V, b = (a + 1000u) % V;
      x[a] = x[b] = 40.0f;
    } else if (t % 4 == 2) { /* +0 and -0 as the maximum */
      for (uint32_t i = 0; i < V; ++i)
        x[i] = -1.0f - (float)(i % 7);
      x[(uint32_t)(t * 31) % V] = -0.0f;
      x[(uint32_t)(t * 31 + 500) % V] = 0.0f;
    } else if (t % 4 == 3) {
      for (uint32_t i = 0; i < V; ++i)
        x[i] = -INFINITY;
      if (t % 8 == 7)
        x[(uint32_t)t % V] = NAN; /* the spec path */
    }
    bad_a += hvx_argmax_first_f32(x, V) != m1_argmax_first(x, V);
  }
  printf("M1 OPS swiglu_cpu pooled bad_rows=%u/12  argmax_first bad=%u/40\n",
         bad_s, bad_a);
  CHECK(bad_s == 0u, "hvx_swiglu_cpu_f32 differs from m1_swiglu_cpu_det");
  CHECK(bad_a == 0u, "hvx_argmax_first_f32 differs from m1_argmax_first");
  free(y);
  free(z);
  free(o);
  free(r);
  free(x);
}

int main(int argc, char **argv) {
  if (argc >= 5 && !strcmp(argv[1], "--replay")) {
    return replay(argc - 2, argv + 2);
  }
  check_rmsnorm("rmsnorm", 2048u, 2048u);
  check_rmsnorm("qk_norm_q", 32u * 64u, 64u);
  check_rmsnorm("qk_norm_k", 8u * 64u, 64u);
  check_norm_cpu_order();
  check_sqrt_recip_rn();
  check_rope();
  check_conv();
  check_swiglu_argmax_kernels();
  check_router_kernel();
  check_router_vs_cpu();
  check_expf_port();
  check_swiglu_argmax();
  if (g_fail) {
    printf("M1 OPS CHECK FAILED\n");
    return 1;
  }
  printf("ROUTER TOPK BIT-IDENTICAL\n");
  printf("M1 OPS BIT-IDENTICAL\n");
  return 0;
}
