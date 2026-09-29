// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   q4_gemv_host_check.c
 * @date   29 Sep 2026
 * @brief  Host check (#132 PR 2): the CPU-order Q4_0 FC spec
 *         (q4_gemv_cpu_det.h) against an independent reference of the
 *         Android CPU, with mutants; then the REAL DSP kernel
 *         hvx_q4_gemv_f32.c on hvx_emu/ against the spec, bit for bit
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * 1. HELPERS. The integer division, f32 <-> f16 and fcvtns against the
 *    host's own IEEE operations (x86 divss, _Float16 casts, lrintf), on
 *    special values, subnormals and random bit patterns.
 * 2. SPEC == CPU. The reference is written from the disassembly with the
 *    host's '/', fmaf and _Float16 (plan 132 section 0.1); the spec must
 *    equal it on the five FC shapes, activations 2^-20 .. 2^20 per block,
 *    zero and tiny blocks, negative and subnormal d_w. Each mutant (one
 *    per rejected order) must differ somewhere: Q4 GEMV CPU-ORDER OK
 *    mutants=N/N. Near ties are counted: steps whose exact sum is a
 *    midpoint of two floats, and round-to-odd corrections of the kernel's
 *    algorithm (a scalar model of it), so the pass says what it covered.
 * 3. KERNEL == SPEC. The vector kernel over the same rows, and the vector
 *    quantizer hvx_q4m1_prep over 3000 more (every row kind, blocks around
 *    its 2^-100 scalar fallback, f16-overflowing d excluded):
 *    Q4 GEMV BIT-IDENTICAL, Q8 QUANT HVX BIT-IDENTICAL. What this rests
 *    on: one Vsf op = one IEEE op (silicon: the 2026-09-29 G1) and the
 *    DSP's scalar divide = IEEE RN, which HvxFcQ4.ScalarDivide sweeps on
 *    silicon.
 * 4. LAYOUT. q4_0_from_q4_0x4 inverts the ARM repack (the ggml packer,
 *    re-typed here) bit for bit.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_q4_gemv_f32.h"
#include "q4_gemv_cases.h"
#include "q4_gemv_cpu_det.h"

static int g_fail = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL: " __VA_ARGS__);                                            \
      printf("\n");                                                            \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

static uint32_t fbits(float f) { return cpu_det_bits(f); }

/* ---- 1. helpers --------------------------------------------------------- */

static float rand_float_any(void) {
  /* every exponent, both signs, often subnormal / special */
  const uint32_t r = rnd();
  switch (r & 7u) {
  case 0:
    return cpu_det_float(rnd() & 0x807fffffu); /* subnormal */
  case 1:
    return cpu_det_float((rnd() & 0x80000000u) | (0x7f800000u));
  default:
    return cpu_det_float(rnd());
  }
}

static void check_helpers(void) {
  uint32_t n = 0, bad_div = 0, bad_h = 0, bad_hf = 0, bad_cvt = 0;
  for (uint32_t i = 0; i < 4000000u; ++i) {
    const float a = rand_float_any(), b = rand_float_any();
    volatile float q = a / b;
    const float s = cpu_det_div_rn(a, b);
    ++n;
    if (q != q) {
      bad_div += s == s; /* any NaN is fine */
    } else {
      bad_div += fbits(q) != fbits(s);
    }
    volatile _Float16 h = (_Float16)a;
    uint16_t hb;
    memcpy(&hb, (const void *)&h, sizeof(hb));
    const uint16_t sh = cpu_det_f32_to_f16(a);
    if (a == a) {
      bad_h += hb != sh;
    }
    if (fabsf(a) < 2.1e9f && a == a) {
      bad_cvt += (long)cpu_det_fcvtns(a) != lrintf(a);
    }
  }
  for (uint32_t u = 0; u < 65536u; ++u) {
    const uint16_t hb = (uint16_t)u;
    _Float16 h;
    memcpy(&h, &hb, sizeof(h));
    const float f = (float)h, g = cpu_det_f16_to_f32(hb);
    if (f == f) {
      bad_hf += fbits(f) != fbits(g);
    }
  }
  bad_cvt += cpu_det_fcvtns(3e9f) != INT32_MAX;
  bad_cvt += cpu_det_fcvtns(-3e9f) != INT32_MIN;
  bad_cvt += cpu_det_fcvtns(cpu_det_float(CPU_DET_NAN)) != 0;
  bad_cvt += cpu_det_fcvtns(2.5f) != 2 || cpu_det_fcvtns(-3.5f) != -4;
  printf("Q4 HELPERS pairs=%u div_rn bad=%u f32->f16 bad=%u f16->f32 (all "
         "65536) bad=%u fcvtns bad=%u\n",
         n, bad_div, bad_h, bad_hf, bad_cvt);
  CHECK(bad_div + bad_h + bad_hf + bad_cvt == 0u, "exact helpers");
}

/* ---- 2. the independent CPU reference and its mutants ------------------- */

enum {
  MUT_NONE = 0,
  MUT_UNFUSED,    /* acc = acc + isum * s */
  MUT_PARTIAL4,   /* four chains over b mod 4, summed at the end */
  MUT_RECIP127,   /* d = amax * (1/127) */
  MUT_DIV_D,      /* q = x / d instead of x * (1/d) */
  MUT_ROUND_AWAY, /* q = roundf (ties away), not fcvtns */
  MUT_F32_DA,     /* s from the unrounded f32 d_a, not its f16 */
  MUT_COUNT
};

static float h2f(uint16_t h) {
  _Float16 f;
  memcpy(&f, &h, sizeof(f));
  return (float)f;
}

/** @brief The CPU FC from the disassembly: nntr_quantize_row_q8_0 then
 *  nntr_gemv_q4_0_4x8_q8_0's chain, host operators only. */
static void cpu_fc_ref(const float *x, const uint8_t *w, uint32_t K, uint32_t N,
                       float *y, int mut) {
  const uint32_t nb = K / Q4_CPU_QK;
  int8_t *q = malloc(K);
  uint16_t *dh = malloc(nb * sizeof(uint16_t));
  float *df = malloc(nb * sizeof(float));
  for (uint32_t b = 0; b < nb; ++b) {
    float amax = 0.0f;
    for (uint32_t j = 0; j < Q4_CPU_QK; ++j) {
      amax = fmaxf(amax, fabsf(x[b * Q4_CPU_QK + j]));
    }
    volatile float d =
      mut == MUT_RECIP127 ? amax * (1.0f / 127.0f) : amax / 127.0f;
    volatile float id = d != 0.0f ? 1.0f / d : 0.0f;
    for (uint32_t j = 0; j < Q4_CPU_QK; ++j) {
      volatile float v = mut == MUT_DIV_D
                           ? (d != 0.0f ? x[b * Q4_CPU_QK + j] / d : 0.0f)
                           : x[b * Q4_CPU_QK + j] * id;
      const long r = mut == MUT_ROUND_AWAY ? lroundf(v) : lrintf(v);
      q[b * Q4_CPU_QK + j] = (int8_t)(uint8_t)(r & 0xff);
    }
    volatile _Float16 h = (_Float16)d;
    memcpy(&dh[b], (const void *)&h, sizeof(uint16_t));
    df[b] = d;
  }
  for (uint32_t n = 0; n < N; ++n) {
    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    volatile float acc = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
      const uint8_t *blk = w + ((size_t)n * nb + b) * Q4_CPU_BLOCK_BYTES;
      int32_t isum = 0;
      for (uint32_t j = 0; j < 16u; ++j) {
        isum += ((int)(blk[2 + j] & 15) - 8) * q[b * 32u + j];
        isum += ((int)(blk[2 + j] >> 4) - 8) * q[b * 32u + 16u + j];
      }
      const float da = mut == MUT_F32_DA ? df[b] : h2f(dh[b]);
      volatile float s = da * h2f((uint16_t)(blk[0] | (blk[1] << 8)));
      if (mut == MUT_UNFUSED) {
        volatile float p = (float)isum * s;
        acc = acc + p;
      } else if (mut == MUT_PARTIAL4) {
        acc4[b & 3u] = fmaf((float)isum, s, acc4[b & 3u]);
      } else {
        acc = fmaf((float)isum, s, acc);
      }
    }
    if (mut == MUT_PARTIAL4) {
      volatile float h0 = acc4[0] + acc4[1], h1 = acc4[2] + acc4[3];
      acc = h0 + h1;
    }
    y[n] = acc;
  }
  free(q);
  free(dh);
  free(df);
}

/** @brief Near-tie coverage of one column's chain (the spec's inputs):
 *  steps whose exact acc + isum s is a midpoint of two floats, and steps
 *  where the kernel's round-to-odd step moves w. A scalar model of
 *  hvx_q4_gemv_f32.c's algorithm, for counting only. */
static void tie_stats(const uint8_t *w, const int8_t *q, const uint16_t *da,
                      uint32_t K, uint32_t n, uint32_t *ties,
                      uint32_t *ro_moves, uint32_t *to_zero) {
  const uint32_t nb = K / Q4_CPU_QK;
  float acc = 0.0f;
  for (uint32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w + ((size_t)n * nb + b) * Q4_CPU_BLOCK_BYTES;
    const int32_t isum = q4_cpu_block_isum(blk + 2, q + b * Q4_CPU_QK);
    const float s = cpu_det_mul(cpu_det_f16_to_f32(da[b]),
                                cpu_det_f16_to_f32(q4_cpu_block_d(blk)));
    /* the exact sum in long double when it fits (64-bit mantissa) */
    const long double ex = (long double)acc + (long double)isum * s;
    const float rn = (float)ex;
    if (ex == (long double)acc + (long double)isum * s && rn != 0.0f) {
      const float lo = ex < (long double)rn ? nextafterf(rn, -INFINITY) : rn;
      const float hi = ex < (long double)rn ? rn : nextafterf(rn, INFINITY);
      *ties += ((long double)lo + (long double)hi) / 2.0L == ex && lo != hi;
    }
    /* model: P = isum * s exact in double (37 bits), split into uh + ul */
    const double P = (double)isum * (double)s;
    const float uh = (float)P, ul = (float)(P - (double)uh);
    volatile float th = acc + uh;
    volatile float bb = th - acc, ab = th - bb;
    volatile float tl = (acc - ab) + (uh - bb);
    volatile float wv = tl + ul;
    volatile float b2 = wv - tl, a2 = wv - b2;
    volatile float er = (tl - a2) + (ul - b2);
    *ro_moves += er != 0.0f && !(fbits(wv) & 1u);
    const float was = acc;
    acc = cpu_det_fma((float)isum, s, acc);
    *to_zero += was != 0.0f && acc == 0.0f;
  }
}

/* ---- 4. the ARM repack, re-typed from nntr_make_block_q4_0x4 ------------ */

static void repack_q4_0x4(const uint8_t *w, uint32_t K, uint32_t N,
                          uint8_t *x4) {
  const uint32_t nb = K / Q4_CPU_QK;
  for (uint32_t n4 = 0; n4 < N / 4u; ++n4) {
    for (uint32_t b = 0; b < nb; ++b) {
      uint8_t *out = x4 + ((size_t)n4 * nb + b) * 72u;
      for (uint32_t i = 0; i < 4u; ++i) {
        const uint8_t *in =
          w + ((size_t)(4u * n4 + i) * nb + b) * Q4_CPU_BLOCK_BYTES;
        out[2u * i] = in[0];
        out[2u * i + 1u] = in[1];
      }
      for (uint32_t i = 0; i < 8u; ++i) { /* Q4_0 * 2 / 8 = 8 chunks */
        const uint32_t src_id = i % 4u, src_off = (i / 4u) * 8u;
        const uint8_t *in =
          w + ((size_t)(4u * n4 + src_id) * nb + b) * Q4_CPU_BLOCK_BYTES;
        for (uint32_t k = 0; k < 8u; ++k) {
          out[8u + i * 8u + k] = (uint8_t)(in[2 + src_off + k] ^ 0x88u);
        }
      }
    }
  }
}

/* ---- the shapes ---------------------------------------------------------- */

static void check_shape(uint32_t K, uint32_t N, uint32_t *mut_caught,
                        uint32_t *total_bad_spec, uint32_t *total_bad_kernel,
                        uint32_t *ties, uint32_t *ro_moves, uint32_t *to_zero) {
  const uint32_t nb = K / Q4_CPU_QK;
  const size_t wbytes = (size_t)N * nb * Q4_CPU_BLOCK_BYTES;
  uint8_t *w = malloc(wbytes), *w4 = malloc(wbytes), *wc = malloc(wbytes);
  uint8_t *m1 = aligned_alloc(128, q4m1_bytes(K, N));
  float *x = malloc(K * sizeof(float));
  float *ys = malloc(N * sizeof(float)), *yr = malloc(N * sizeof(float));
  float *yk = malloc(N * sizeof(float));
  int8_t *q = aligned_alloc(128, K);
  uint16_t *da = malloc(nb * sizeof(uint16_t));
  hvx_q4m1_act act;
  act.q = q;
  act.d = da;
  act.s8 = malloc(nb * sizeof(int32_t));
  act.ma = malloc(nb * sizeof(int32_t));
  act.ea = malloc(nb * sizeof(int32_t));
  act.df = malloc(nb * sizeof(float));

  make_weights(w, K, N);
  repack_q4_0x4(w, K, N, w4);
  q4_0_from_q4_0x4(w4, K, N, wc);
  const int layout_ok = memcmp(w, wc, wbytes) == 0;
  CHECK(layout_ok, "K=%u N=%u: q4_0_from_q4_0x4 does not invert the repack", K,
        N);
  q4m1_from_q4_0(w, K, N, m1);

  const int rows = 4;
  uint32_t bad_spec = 0, bad_k = 0;
  for (int row = 0; row < rows; ++row) {
    make_row(x, K, row, row);
    q8_0_quant_cpu_det(x, K, q, da);
    q4_gemv_cpu_det(w, q, da, K, N, ys);
    cpu_fc_ref(x, w, K, N, yr, MUT_NONE);
    for (uint32_t n = 0; n < N; ++n) {
      bad_spec += fbits(ys[n]) != fbits(yr[n]);
    }
    for (int mut = 1; mut < MUT_COUNT; ++mut) {
      cpu_fc_ref(x, w, K, N, yr, mut);
      uint32_t d = 0;
      for (uint32_t n = 0; n < N; ++n) {
        d += fbits(ys[n]) != fbits(yr[n]);
      }
      mut_caught[mut] += d;
    }
    for (uint32_t n = 0; n < N; ++n) {
      if (n % 7u == 0u || n % 32u == 4u) {
        tie_stats(w, q, da, K, n, ties, ro_moves, to_zero);
      }
    }
    hvx_q4m1_prep(x, K, &act);
    CHECK(memcmp(act.q, q, K) == 0 && memcmp(act.d, da, nb * 2u) == 0,
          "hvx_q4m1_prep differs from q8_0_quant_cpu_det");
    memset(yk, 0xA5, N * sizeof(float));
    hvx_q4m1_gemv_groups(m1, K, N / Q4M1_GROUP, &act, yk);
    for (uint32_t n = 0; n < N; ++n) {
      bad_k += fbits(yk[n]) != fbits(ys[n]);
    }
  }
  printf("Q4 GEMV K=%u N=%u rows=%d spec-vs-cpu bad=%u kernel bad=%u "
         "layout=%s\n",
         K, N, rows, bad_spec, bad_k, layout_ok ? "ok" : "BAD");
  *total_bad_spec += bad_spec;
  *total_bad_kernel += bad_k;
  free(w);
  free(w4);
  free(wc);
  free(m1);
  free(x);
  free(ys);
  free(yr);
  free(yk);
  free(q);
  free(da);
  free(act.s8);
  free(act.ma);
  free(act.ea);
  free(act.df);
}

/** @brief hvx_q4m1_prep == q8_0_quant_cpu_det on 3000 rows of K = 2048:
 *  make_row's four kinds, and rows whose blocks sit at 2^-100 times 2^-2
 *  .. 2^2 (both sides of the scalar fallback). */
static void check_quant(void) {
  enum { K = 2048, NB = K / 32 };
  static float x[K];
  static int8_t q[K], qs[K];
  static uint16_t d[NB], ds[NB];
  static int32_t s8[NB], ma[NB], ea[NB];
  static float df[NB];
  hvx_q4m1_act act = {q, s8, ma, ea, df, d};
  uint32_t bad = 0, rows = 0;
  for (int r = 0; r < 3000; ++r, ++rows) {
    if (r < 2800) {
      make_row(x, K, r % 4, r);
    } else {
      for (uint32_t i = 0; i < K; ++i) {
        x[i] = ldexpf(frand(-1.0f, 1.0f), -100 + (int)((i / 32u) % 5u) - 2);
      }
    }
    hvx_q4m1_prep(x, K, &act);
    q8_0_quant_cpu_det(x, K, qs, ds);
    bad += memcmp(q, qs, K) != 0 || memcmp(d, ds, sizeof(d)) != 0;
  }
  printf("Q8 QUANT HVX rows=%u K=%d bad_rows=%u\n", rows, K, bad);
  CHECK(bad == 0u, "hvx_q4m1_prep differs from q8_0_quant_cpu_det");
}

int main(void) {
  static const uint32_t shapes[5][2] = {{2048u, 6144u},
                                        {2048u, 2048u},
                                        {2048u, 512u},
                                        {7168u, 2048u},
                                        {2048u, 7168u}};
  static const char *mut_name[MUT_COUNT] = {
    "", "unfused", "partial4", "amax*(1/127)", "x/d", "round-away", "f32-d_a"};
  uint32_t caught[MUT_COUNT] = {0}, bad_spec = 0, bad_kernel = 0, ties = 0,
           ro = 0, to_zero = 0;
  check_helpers();
  check_quant();
  for (int s = 0; s < 5; ++s) {
    check_shape(shapes[s][0], shapes[s][1], caught, &bad_spec, &bad_kernel,
                &ties, &ro, &to_zero);
  }
  uint32_t n_caught = 0;
  printf("Q4 GEMV mutants (columns differing from the spec):");
  for (int m = 1; m < MUT_COUNT; ++m) {
    printf(" %s=%u", mut_name[m], caught[m]);
    n_caught += caught[m] > 0u;
    CHECK(caught[m] > 0u, "mutant %s not caught", mut_name[m]);
  }
  printf("\nQ4 GEMV near ties: exact-midpoint steps=%u round-to-odd "
         "moves=%u cancellations-to-zero=%u\n",
         ties, ro, to_zero);
  CHECK(ties > 0u && ro > 0u && to_zero > 0u, "no near-tie coverage");
  CHECK(bad_spec == 0u, "spec differs from the CPU reference");
  CHECK(bad_kernel == 0u, "kernel differs from the spec");
  if (g_fail) {
    printf("Q4 GEMV CHECK FAILED\n");
    return 1;
  }
  printf("Q4 GEMV CPU-ORDER OK mutants=%u/%d\n", n_caught, MUT_COUNT - 1);
  printf("Q8 QUANT HVX BIT-IDENTICAL\n");
  printf("Q4 GEMV BIT-IDENTICAL\n");
  return 0;
}
