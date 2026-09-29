// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   attn_m1_host_check.c
 * @date   27 Sep 2026
 * @brief  Host check: attn_m1_det.h equals an independent model of the
 *         Android fp16 CPU attention, and the real HVX decode-attention
 *         kernel, compiled against hvx_emu/ and the real worker pool on
 *         pthreads, is bit-identical to attn_m1_det.h at every worker count
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Three parts (plan 152 section 4 steps 1-2):
 *  1. SPEC == CPU ORDER. A second implementation of the Android CPU's
 *     attention, written from neon_impl_fp16.cpp / neon_mathfun.hxx and
 *     sharing no arithmetic with the spec: _Float16 conversions for every
 *     non-fused fp16 op (the compiler's rounding, not rne16), the fused
 *     vfmaq_f16 as a double sum with an explicit sticky (round to odd at
 *     53 bits, then one conversion), exp_ps with fmaf for its fused step,
 *     and the divide in double. The spec must equal it bit for bit on
 *     RoPE rows, on exp at every fp16 d <= 0 (and the kernel's table on
 *     the same d), and on whole attention calls at L = 1, 2, 63, 64, 65,
 *     512, 513, 1024 with the adversarial rows of attn_m1_cases.h. Then
 *     five mutants of the reference -- no round-to-odd, non-fused fp16,
 *     a 32-lane sum, an unfused fx, an f32 RoPE -- must each differ
 *     somewhere, which proves the data can see each rounding point
 *     (ATTN M1 F16 CPU-ORDER OK mutants=5/5). What this cannot prove: that
 *     the phone's CPU is this model. AttnM1F16Det.* does that on device.
 *  2. PRIMITIVES == SPEC. hvx_convert.h's hvx_rne16_sf on a sweep of f32
 *     values, hvx_fma16_sf on random and constructed midpoint triples, and
 *     hvx_div16_sf on every fp16 quotient e / l (l in [1, 2048], e in
 *     [0, l]) whose recip_det product is off by one or which is an exact
 *     tie, plus a sample: the cases attention-level data almost never
 *     reaches (a divide whose correction is removed passes every
 *     attention call below). Since #170 also hvx_attn_m1_hf.h, the fp16-lane
 *     primitives of the fast kernel (ATTN M1 HF PRIM OK): the one-rounding
 *     FMA on adversarial, zero / sign and random triples, the hf ops over
 *     every finite fp16, the score tree, exp16 at every fp16 d <= 0 and
 *     the divide on the sweep's hard quotients; since round 2 the vector
 *     rounding hvx_hf_round_row at every f32 the scalar row walks and the
 *     vector widening hvx_hf_store_sf at every finite fp16.
 *  3. KERNEL == SPEC. hvx_attn_m1_f32.c -- the skel's own source -- on the
 *     lane-by-lane emulation with the pthread worker pool at 0, 3 and 7
 *     workers, memcmp'd against the spec for L = 1, 63, 64, 65, 512, 513,
 *     1024, 1536 at the shapes (n_kv, gqa) = (8, 4) (LFM2.5), (1, 2) (the
 *     hd64 fixture), (2, 3) (odd gqa: single-chain P3 ranges, partial ET
 *     groups) and (1, 8) (two ET transpose groups per kv head), head_dim
 *     64, at max_seq 2048 (the model's). Each forward is repeated with
 *     the phase words requested (ATTN M1 PHASES OK). Structural cases: an
 *     append chain equals one bulk append; at L = 1 the output is rne16(v)
 *     exactly; a division tie (p = 2^-25 between 0 and 2^-24) rounds to
 *     even; the error codes. What it rests on: one Vsf op = one IEEE op
 *     (rule 24) and integer ops exact, re-checked on the device by
 *     HvxAttnM1.*.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>

#include "attn_m1_cases.h"
#include "attn_m1_det.h"
#include "hvx_attn_m1_f32.h"
#include "hvx_attn_m1_hf.h"
#include "hvx_convert.h"
#include "hvx_swiglu_det.h"
#include "hvx_worker_pool.h"
#include "m1_ops_det.h"

static int g_fail = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL: " __VA_ARGS__);                                            \
      printf("\n");                                                            \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

static uint32_t count_bad(const float *a, const float *b, size_t n) {
  uint32_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    bad += memcmp(&a[i], &b[i], sizeof(float)) ? 1u : 0u;
  }
  return bad;
}

/* ==== 1. the independent CPU-order reference ============================ */

typedef _Float16 h16;

/** @brief The reference's deliberate faults; MUT_NONE is the CPU. */
enum {
  MUT_NONE,
  MUT_NO_RTO,      /* fused FMA rounded to f32 first, then to fp16 */
  MUT_HF_NONFUSED, /* fp16 multiply, then fp16 add */
  MUT_SUM32,       /* the old spec's 32 lane sums + pairwise tree */
  MUT_UNFUSED_FX,  /* exp_ps's fx = x * LOG2EF + 0.5 in two roundings */
  MUT_ROPE_F32,    /* RoPE in f32, rounded to fp16 once */
  MUT_N
};
static const char *const MUT_NAME[MUT_N] = {
  "none", "no_rto", "hf_nonfused", "sum32", "unfused_fx", "rope_f32"};
static int g_mut = MUT_NONE;

/** @brief One fp16 op the way the CPU's fadd / fsub / fmul / fdiv .8h do
 *         it: exact operands, one rounding (through f32 first, which is
 *         harmless for these four: 24 >= 2 * 11 + 2). */
static h16 r_add(h16 a, h16 b) { return (h16)((float)a + (float)b); }
static h16 r_sub(h16 a, h16 b) { return (h16)((float)a - (float)b); }
static h16 r_mul(h16 a, h16 b) { return (h16)((float)a * (float)b); }

/** @brief vfmaq_f16: c + a*b rounded once. The double sum is rounded to
 *         odd (a sticky bit when it was inexact) so the one conversion to
 *         fp16 is the single rounding. */
static h16 r_fma(h16 c, h16 a, h16 b) {
  if (g_mut == MUT_HF_NONFUSED) {
    return r_add(c, r_mul(a, b));
  }
  const double p = (double)a * (double)b; /* exact: 22 bits */
  if (g_mut == MUT_NO_RTO) {
    return (h16)(float)((double)c + p);
  }
  double s = (double)c + p;
  const double bv = s - (double)c;
  const double err = ((double)c - (s - bv)) + (p - bv);
  if (err != 0.0) {
    uint64_t w;
    memcpy(&w, &s, sizeof(w));
    if ((w & 1u) == 0u) {
      w += ((err < 0.0) == (s < 0.0)) ? 1u : (uint64_t)-1;
      memcpy(&s, &w, sizeof(w));
    }
  }
  return (h16)s;
}

/** @brief neon_mathfun.hxx's exp_ps on one lane; fmaf is the fused vmlaq. */
static float r_exp_ps(float x) {
  static const float c[6] = {(float)1.9875691500E-4, (float)1.3981999507E-3,
                             (float)8.3334519073E-3, (float)4.1665795894E-2,
                             (float)1.6666665459E-1, (float)5.0000001201E-1};
  const float log2ef = (float)1.44269504088896341;
  volatile float t;
  x = fminf(x, 88.3762626647949f);
  x = fmaxf(x, -88.3762626647949f);
  float fx;
  if (g_mut == MUT_UNFUSED_FX) {
    t = x * log2ef;
    fx = t + 0.5f;
  } else {
    fx = fmaf(x, log2ef, 0.5f);
  }
  float tmp = (float)(int32_t)fx;
  fx = tmp - ((tmp > fx) ? 1.0f : 0.0f);
  t = fx * (float)0.693359375;
  float z = fx * (float)-2.12194440e-4;
  x = x - t;
  x = x - z;
  float y = c[0] * x;
  z = x * x;
  y = y + c[1];
  y = y * x;
  y = y + c[2];
  y = y * x;
  y = y + c[3];
  y = y * x;
  y = y + c[4];
  y = y * x;
  y = y + c[5];
  y = y * z;
  y = y + x;
  y = y + 1.0f;
  const int32_t mm = ((int32_t)fx + 0x7F) * (1 << 23);
  float pow2n;
  memcpy(&pow2n, &mm, sizeof(pow2n));
  return y * pow2n;
}

/** @brief compute_rotary_emb_value(__fp16) on one 64-wide head, in place. */
static void r_rope64(h16 *x, const float *cs) {
  for (int i = 0; i < 32; ++i) {
    const h16 a = x[i], b = x[i + 32];
    const h16 c = (h16)cs[i], s = (h16)cs[32 + i];
    if (g_mut == MUT_ROPE_F32) {
      x[i] = (h16)((float)a * (float)c - (float)b * (float)s);
      x[i + 32] = (h16)((float)a * (float)s + (float)b * (float)c);
    } else {
      x[i] = r_sub(r_mul(a, c), r_mul(b, s));
      x[i + 32] = r_add(r_mul(a, s), r_mul(b, c));
    }
  }
}

/**
 * @brief compute_kcaches(_FP16) + softmax_row_inplace(_FP16) +
 *        compute_fp16vcache_transposed for one layer at one position.
 *        k16 / v16 are the fp16 cache rows [L][n_kv][hd]; out [n_q][hd]
 *        widened to f32; l_out the fp16 sums.
 */
static void r_attention(const h16 *q16, const h16 *k16, const h16 *v16,
                        uint32_t n_kv, uint32_t gqa, uint32_t hd, uint32_t L,
                        float *out, float *l_out) {
  h16 *pr = malloc((size_t)L * sizeof(h16));
  for (uint32_t hq = 0; hq < n_kv * gqa; ++hq) {
    const uint32_t h = hq / gqa;
    const h16 *q = q16 + (size_t)hq * hd;
    for (uint32_t p = 0; p < L; ++p) {
      const h16 *k = k16 + ((size_t)p * n_kv + h) * hd;
      h16 acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (uint32_t d = 0; d < hd; ++d) {
        acc[d % 8] = r_fma(acc[d % 8], q[d], k[d]);
      }
      const h16 t = r_add(r_add(r_add(acc[0], acc[1]), r_add(acc[2], acc[3])),
                          r_add(r_add(acc[4], acc[5]), r_add(acc[6], acc[7])));
      const h16 sum = (h16)(0.0f + (float)t);
      pr[p] = (h16)((float)sum / 8.0f); /* sum / sqrt(64.f) */
    }
    h16 m = pr[0];
    for (uint32_t p = 1; p < L; ++p) {
      m = pr[p] > m ? pr[p] : m;
    }
    h16 l = 0;
    h16 lanes[32] = {0};
    for (uint32_t p = 0; p < L; ++p) {
      pr[p] = (h16)r_exp_ps((float)r_sub(pr[p], m));
      if (g_mut == MUT_SUM32) {
        lanes[p % 32] = r_add(lanes[p % 32], pr[p]);
      } else {
        l = r_add(l, pr[p]);
      }
    }
    if (g_mut == MUT_SUM32) {
      for (int s = 16; s >= 1; s >>= 1) {
        for (int j = 0; j < s; ++j) {
          lanes[j] = r_add(lanes[j], lanes[j + s]);
        }
      }
      l = lanes[0];
    }
    for (uint32_t p = 0; p < L; ++p) {
      pr[p] = (h16)((double)pr[p] / (double)l);
    }
    for (uint32_t d = 0; d < hd; ++d) {
      h16 o = 0;
      for (uint32_t p = 0; p < L; ++p) {
        o = r_fma(o, pr[p], v16[((size_t)p * n_kv + h) * hd + d]);
      }
      out[(size_t)hq * hd + d] = (float)o;
    }
    l_out[hq] = (float)l;
  }
  free(pr);
}

/* ---- the spec side of the same calls ----------------------------------- */

/** @brief The spec's cache for one layer from L rows, then its forward. */
static void spec_forward(const float *q, const float *k, const float *v,
                         uint32_t n_kv, uint32_t gqa, uint32_t hd, uint32_t L,
                         uint32_t max_seq, float *out, float *stats) {
  float *kt = calloc((size_t)n_kv * hd * max_seq, sizeof(float));
  float *vv = calloc((size_t)n_kv * max_seq * hd, sizeof(float));
  float *e = malloc((size_t)L * sizeof(float));
  for (uint32_t p = 0; p < L; ++p) {
    for (uint32_t h = 0; h < n_kv; ++h) {
      attn_m1_det_append(kt + (size_t)h * hd * max_seq,
                         vv + (size_t)h * max_seq * hd, hd, max_seq, p,
                         k + ((size_t)p * n_kv + h) * hd,
                         v + ((size_t)p * n_kv + h) * hd);
    }
  }
  attn_m1_det_forward(q, kt, vv, n_kv, gqa, hd, max_seq, L, 0.125f, e, out,
                      stats);
  free(kt);
  free(vv);
  free(e);
}

/** @brief The cos | sin row of one position (the CPU's formula in double,
 *         rounded once to f32, as mha_core's table holds it). */
static void rope_cs(float *cs, uint32_t pos) {
  for (uint32_t i = 0; i < 32u; ++i) {
    const double ang = (double)pos * pow(5e6, -(2.0 * (double)i) / 64.0);
    cs[i] = (float)cos(ang);
    cs[32u + i] = (float)sin(ang);
  }
}

/** @brief exp at every fp16 d <= 0 (-0 .. -inf): spec exp16 vs the
 *         reference, and the kernel's table vs exp16. */
static uint32_t cmp_exp(uint32_t *bad_table) {
  float *tab = malloc(ATTN_M1_DET_EXP_N * sizeof(float));
  attn_m1_det_exp_table(tab);
  uint32_t bad = 0;
  *bad_table = 0;
  for (uint32_t b = 0x8000u; b <= 0xFC00u; ++b) {
    const uint16_t hb = (uint16_t)b;
    h16 dh;
    memcpy(&dh, &hb, sizeof(dh));
    const float d = (float)dh;
    const float ref = (float)(h16)r_exp_ps(d);
    const float spec = attn_m1_det_exp16(d);
    bad += count_bad(&spec, &ref, 1);
    *bad_table += count_bad(&tab[attn_m1_det_exp_index(d)], &spec, 1);
  }
  free(tab);
  return bad;
}

/** @brief RoPE at five positions on 32 q + 8 k heads of every kind. */
static uint32_t cmp_rope(void) {
  static const uint32_t positions[] = {0u, 1u, 511u, 1023u, 4095u};
  enum { NH = 40 };
  float x[NH * 64], cs[64];
  h16 xr[64];
  amc_rng r = {0x15200001u};
  uint32_t bad = 0;
  for (int h = 0; h < NH; ++h) {
    amc_fill_row(&r, x + h * 64, 64, (h < 3) ? h + 1 : 0);
  }
  for (size_t i = 0; i < sizeof(positions) / sizeof(positions[0]); ++i) {
    rope_cs(cs, positions[i]);
    for (int h = 0; h < NH; ++h) {
      float y[64];
      memcpy(y, x + h * 64, sizeof(y));
      m1_rope64_det(y, cs);
      for (int d = 0; d < 64; ++d) {
        xr[d] = (h16)x[h * 64 + d];
      }
      r_rope64(xr, cs);
      for (int d = 0; d < 64; ++d) {
        const float ref = (float)xr[d];
        bad += count_bad(&y[d], &ref, 1);
      }
    }
  }
  return bad;
}

/** @brief One attention call at the LFM2.5 shape and length L. */
static uint32_t cmp_attention(uint32_t L, uint32_t *planted, int verbose) {
  enum { KV = 8, GQA = 4, HD = 64, NQ = KV * GQA };
  const size_t row = (size_t)KV * HD;
  amc_rng r = {0x15200100u + L};
  float *q = malloc((size_t)NQ * HD * sizeof(float));
  float *k = malloc((size_t)L * row * sizeof(float));
  float *v = malloc((size_t)L * row * sizeof(float));
  float *out = malloc((size_t)NQ * HD * sizeof(float));
  float *ref = malloc((size_t)NQ * HD * sizeof(float));
  float stats[2 * NQ], lref[NQ], lspec[NQ];
  h16 *q16 = malloc((size_t)NQ * HD * sizeof(h16));
  h16 *k16 = malloc((size_t)L * row * sizeof(h16));
  h16 *v16 = malloc((size_t)L * row * sizeof(h16));
  amc_fill_q(&r, q, NQ, HD);
  amc_fill_kv(&r, k, v, L, KV, GQA, HD);
  *planted = amc_plant_pv(q, k, v, L, KV, GQA, HD);
  spec_forward(q, k, v, KV, GQA, HD, L, L < 32u ? 32u : L, out, stats);
  for (size_t i = 0; i < (size_t)NQ * HD; ++i) {
    q16[i] = (h16)q[i];
  }
  for (size_t i = 0; i < (size_t)L * row; ++i) {
    k16[i] = (h16)k[i];
    v16[i] = (h16)v[i];
  }
  r_attention(q16, k16, v16, KV, GQA, HD, L, ref, lref);
  for (int hq = 0; hq < NQ; ++hq) {
    lspec[hq] = stats[2 * hq + 1];
  }
  const uint32_t bad = count_bad(out, ref, (size_t)NQ * HD);
  const uint32_t bad_l = count_bad(lspec, lref, NQ);
  if (verbose) {
    printf("ATTN M1 F16 CPU-ORDER L=%u out bad=%u of %d, l bad=%u, pv "
           "midpoint cases planted=%u, score cases=%u\n",
           L, bad, NQ * HD, bad_l, *planted, L > 3u ? 2u * (L - 3u) : 0u);
  }
  free(q);
  free(k);
  free(v);
  free(out);
  free(ref);
  free(q16);
  free(k16);
  free(v16);
  return bad + bad_l;
}

/** @brief Part 1: every comparison under the current g_mut. */
static uint32_t cpu_order_suite(int verbose) {
  static const uint32_t lengths[] = {1u, 2u, 63u, 64u, 65u, 512u, 513u, 1024u};
  uint32_t bad_table = 0, planted = 0, total = 0;
  const uint32_t bad_exp = cmp_exp(&bad_table);
  const uint32_t bad_rope = cmp_rope();
  if (verbose) {
    printf("ATTN M1 F16 exp16 == exp_ps twin at every fp16 d <= 0 (31745): "
           "bad=%u; kernel table == exp16: bad=%u\n",
           bad_exp, bad_table);
    printf("ATTN M1 F16 rope64 == fp16 RoPE, 40 heads x 5 positions: bad=%u\n",
           bad_rope);
    CHECK(bad_table == 0u, "the exp table differs from exp16");
  }
  total = bad_exp + bad_rope;
  for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    total += cmp_attention(lengths[i], &planted, verbose);
  }
  return total;
}

/**
 * @brief The spec against the reference, then each mutant against the
 *        spec. unfused_fx is an EQUIVALENT mutant on this domain, not a
 *        missed one: exp_ps uses fx only through floor(fx), and no fp16 d
 *        puts x * LOG2EF + 0.5 close enough to an integer for the extra
 *        rounding to move that floor. The sweep over all 31745 fp16 d <= 0
 *        in the suite is the proof, so it must differ NOWHERE; if a change
 *        ever makes it differ, the fused step matters and it is caught.
 */
static void check_cpu_order(void) {
  g_mut = MUT_NONE;
  const uint32_t bad = cpu_order_suite(1);
  CHECK(bad == 0u, "the spec differs from the CPU-order reference (%u)", bad);
  int caught = 0, equivalent = 0;
  for (int m = MUT_NONE + 1; m < MUT_N; ++m) {
    g_mut = m;
    const uint32_t diff = cpu_order_suite(0);
    const int equiv = (m == MUT_UNFUSED_FX && diff == 0u);
    printf("ATTN M1 F16 mutant %-11s differs from the spec at %u values%s\n",
           MUT_NAME[m], diff,
           equiv  ? " (equivalent on fp16 d: exhaustive)"
           : diff ? ""
                  : "  <-- NOT CAUGHT");
    caught += diff != 0u;
    equivalent += equiv;
  }
  g_mut = MUT_NONE;
  CHECK(caught + equivalent == MUT_N - 1, "only %d of %d mutants caught",
        caught, MUT_N - 1 - equivalent);
  if (bad == 0u && caught + equivalent == MUT_N - 1) {
    printf("ATTN M1 F16 CPU-ORDER OK mutants=%d/%d equivalent=%d\n", caught,
           MUT_N - 1 - equivalent, equivalent);
  }
}

/* ==== 2. the kernel's fp16 primitives against the spec ================= */

/** @brief hvx_rne16_sf over a sweep of f32 magnitudes (every 61st pattern
 *         below 65520, both signs, plus +-0 and fp16 midpoints +- 1 ulp). */
static void check_prim_rne16(void) {
  uint32_t bad = 0, n = 0;
  HVX_Vector x, y;
  int lane = 0;
  for (uint64_t u = 0; u < 0x477FF000ull; u += 61u) {
    for (int sg = 0; sg < 2; ++sg) {
      uint32_t w = (uint32_t)u | (sg ? 0x80000000u : 0u);
      if ((u & 0xFFFu) == 0u) { /* an fp16 midpoint's neighbourhood */
        w = ((uint32_t)u & ~0x1FFFu) | 0x1000u | (sg ? 0x80000000u : 0u);
      }
      x.w[lane++] = (int32_t)w;
      if (lane == 32) {
        y = hvx_rne16_sf(x);
        for (int i = 0; i < 32; ++i) {
          const float ref = attn_m1_det_rne16(hvx_emu_f(x.w[i]));
          bad += count_bad(&ref, (const float *)&y.w[i], 1);
        }
        n += 32;
        lane = 0;
      }
    }
  }
  printf("ATTN M1 PRIM rne16: %u lanes, bad=%u\n", n, bad);
  CHECK(bad == 0u, "hvx_rne16_sf differs from attn_m1_det_rne16");
}

/** @brief hvx_fma16_sf on random fp16 triples and on constructed midpoint
 *         cases (c + a*b on an fp16 midpoint after f32 rounding). */
static void check_prim_fma16(void) {
  amc_rng r = {0x15200200u};
  uint32_t bad = 0, n = 0, cases = 0;
  for (int it = 0; it < 200000; ++it) {
    HVX_Vector c, a, b;
    for (int i = 0; i < 32; ++i) {
      float fc, fa, fb;
      if (i & 1) { /* a constructed case: the file comment of attn_m1_cases.h */
        const int e = (int)(amc_next(&r) % 3u) - 1;
        const int plus = (int)(amc_next(&r) & 1u);
        uint32_t mant = 1u + amc_next(&r) % 1023u;
        mant = plus ? ((mant & ~1u) ? (mant & ~1u) : 2u) : (mant | 1u);
        fc = ldexpf(1.0f + (float)mant / 1024.0f, e) *
             ((amc_next(&r) & 1u) ? -1.0f : 1.0f);
        fa = plus ? AMC_A_PLUS : AMC_A_MINUS;
        fb = ldexpf(plus ? AMC_B_PLUS : AMC_B_MINUS, e - 5) *
             ((amc_next(&r) & 1u) ? -1.0f : 1.0f);
      } else {
        fc = attn_m1_det_rne16(amc_frand(&r, -8.0f, 8.0f));
        fa = attn_m1_det_rne16(amc_frand(&r, -8.0f, 8.0f));
        fb = attn_m1_det_rne16(amc_frand(&r, -1.0f, 1.0f));
      }
      c.w[i] = hvx_emu_w(fc);
      a.w[i] = hvx_emu_w(fa);
      b.w[i] = hvx_emu_w(fb);
      cases += (i & 1) ? (uint32_t)amc_is_midpoint_case(fc, fa, fb) : 0u;
    }
    const HVX_Vector y = hvx_fma16_sf(c, a, b);
    for (int i = 0; i < 32; ++i) {
      const float ref = attn_m1_det_fma16(hvx_emu_f(c.w[i]), hvx_emu_f(a.w[i]),
                                          hvx_emu_f(b.w[i]));
      bad += count_bad(&ref, (const float *)&y.w[i], 1);
    }
    n += 32;
  }
  printf("ATTN M1 PRIM fma16: %u lanes (%u midpoint cases), bad=%u\n", n, cases,
         bad);
  CHECK(cases == n / 2u, "a constructed fma16 case is not a midpoint case");
  CHECK(bad == 0u, "hvx_fma16_sf differs from attn_m1_det_fma16");
}

/** @brief The hard quotients of the sweep below (c0 off by one or a tie),
 *         which check_hf_prim runs again through the hf divide. */
#define N_DIV_HARD_MAX 40000u
static float g_div_hard_e[N_DIV_HARD_MAX], g_div_hard_l[N_DIV_HARD_MAX];
static uint32_t g_div_hard = 0;

/**
 * @brief hvx_div16_sf against rne16(e / l) over every fp16 l in [1, 2048]
 *        and every fp16 e in [0, l]: 173 M quotients. Every 32-lane batch
 *        that holds a hard lane -- rne16(e * r) off by one ulp, or e / l an
 *        exact fp16 midpoint (a tie) -- runs through the vector code, and
 *        every 64th batch besides; the rest are only classified.
 */
static void check_prim_div16(void) {
  uint64_t total = 0, run = 0, off = 0, ties = 0;
  uint32_t bad = 0;
  for (uint32_t lb = 0x3C00u; lb <= 0x6800u; ++lb) {
    const uint16_t lh16 = (uint16_t)lb;
    h16 lh;
    memcpy(&lh, &lh16, sizeof(lh));
    const float l = (float)lh;
    const HVX_Vector lv = hvx_splat_sf(l), rv = hvx_recip_det_sf(lv);
    const float r = hvx_emu_f(rv.w[0]);
    HVX_Vector e;
    float want[32];
    int lane = 0, hard = 0;
    uint32_t batch = 0;
    for (uint32_t eb = 0; eb <= 0x7C00u; ++eb) {
      const uint16_t eh16 = (uint16_t)eb;
      h16 eh;
      memcpy(&eh, &eh16, sizeof(eh));
      const float ef = (float)eh;
      const int last = eb == 0x7C00u;
      if (ef <= l && !last) {
        want[lane] = attn_m1_det_rne16(attn_m1_det_div(ef, l));
        const float c0 = attn_m1_det_rne16(attn_m1_det_mul(ef, r));
        /* A tie: e / l exactly on an fp16 midpoint -- 1 + 12 bits with the
           last set above 2^-14, an odd multiple of 2^-25 below. */
        const double q = (double)ef / (double)l;
        const float qf = (float)q;
        const int tie =
          (double)qf == q && attn_m1_det_rne16(qf) != qf &&
          (qf >= ldexpf(1.0f, -14) ? (attn_m1_det_bits(qf) & 0x1FFFu) == 0x1000u
                                   : fmod(q * 33554432.0, 2.0) == 1.0);
        off += c0 != want[lane];
        ties += (uint64_t)tie;
        hard |= (c0 != want[lane]) | tie;
        if (((c0 != want[lane]) | tie) && g_div_hard < N_DIV_HARD_MAX) {
          g_div_hard_e[g_div_hard] = ef;
          g_div_hard_l[g_div_hard++] = l;
        }
        e.w[lane++] = hvx_emu_w(ef);
        ++total;
      }
      if (lane == 32 || (last && lane > 0)) {
        if (hard || batch % 64u == 0u) {
          for (int i = lane; i < 32; ++i) { /* pad a short last batch */
            e.w[i] = 0;
            want[i] = 0.0f;
          }
          const HVX_Vector y = hvx_div16_sf(e, lv, rv);
          for (int i = 0; i < 32; ++i) {
            bad += count_bad(&want[i], (const float *)&y.w[i], 1);
          }
          ++run;
        }
        lane = 0;
        hard = 0;
        ++batch;
      }
    }
  }
  printf("ATTN M1 PRIM div16: %llu quotients (%llu with rne16(e*r) off by "
         "one, %llu ties), %llu batches run, bad=%u\n",
         (unsigned long long)total, (unsigned long long)off,
         (unsigned long long)ties, (unsigned long long)run, bad);
  CHECK(off > 0u && ties > 0u, "the division sweep found no hard case");
  CHECK(bad == 0u, "hvx_div16_sf differs from rne16(e / l)");
}

/* ==== 2b. the hf primitives (#170) against the spec ===================== */

static HVX_Vector hf_load(const uint16_t *x) {
  HVX_Vector v;
  for (int i = 0; i < 64; ++i) {
    hvx_emu_set_h(&v, i, x[i]);
  }
  return v;
}

/** @brief hvx_hf_fma on 64 triples against attn_m1_det_fma16, bitwise;
 *         out-of-domain lanes (|result| > 65504) skipped. Adds the lanes
 *         run to *n and the double-rounding hazards among them to *hz. */
static uint32_t hf_fma_batch(const uint16_t *c, const uint16_t *a,
                             const uint16_t *b, uint32_t *n, uint32_t *hz) {
  const HVX_Vector y =
    hvx_hf_fma(hf_load(c), hf_load(a), hf_load(b), Q6_Vh_vsplat_R(HVX_HF_ONE));
  uint32_t bad = 0;
  for (int i = 0; i < 64; ++i) {
    const float fc = amc_h2f(c[i]), fa = amc_h2f(a[i]), fb = amc_h2f(b[i]);
    const float ref = attn_m1_det_fma16(fc, fa, fb);
    if (fabsf(ref) > 65504.0f) {
      continue;
    }
    ++*n;
    *hz += (uint32_t)amc_is_midpoint_case(fc, fa, fb);
    bad += hvx_emu_h(&y, i) != amc_f2h(ref);
  }
  return bad;
}

/**
 * @brief hvx_attn_m1_hf.h against the spec (plan 170 step 1, G2's ATTN M1
 *        HF PRIM): the one-rounding FMA on the two attn_m1_cases.h
 *        families and random fp16 triples (the hazards among them are
 *        counted, so a family that stopped reaching the double-rounding
 *        case fails); the hf ops the tree and softmax use over every
 *        finite fp16 a x 4 random b; the score tree; exp16 at every fp16
 *        d <= 0; the divide on the hard quotients of check_prim_div16.
 *        What the emulation assumes (hvx_emu's qf32 note) is S1's to test.
 */
static void check_hf_prim(void) {
  amc_rng r = {0x17000001u};
  uint16_t c[64], a[64], b[64];
  uint32_t n_adv = 0, hz_adv = 0, bad_adv = 0, n_zs = 0, hz_zs = 0, bad_zs = 0;
  uint32_t n_rnd = 0, hz_rnd = 0, bad_rnd = 0;
  for (uint32_t it = 0; it < 1200u; ++it) {
    for (uint32_t i = 0; i < 64u; ++i) {
      amc_fma_adversarial(&r, it * 64u + i, &c[i], &a[i], &b[i]);
    }
    bad_adv += hf_fma_batch(c, a, b, &n_adv, &hz_adv);
  }
  for (uint32_t it = 0; it < 400u; ++it) {
    for (uint32_t i = 0; i < 64u; ++i) {
      amc_fma_zero_sign(&r, &c[i], &a[i], &b[i]);
    }
    bad_zs += hf_fma_batch(c, a, b, &n_zs, &hz_zs);
  }
  for (uint32_t it = 0; it < 16000u; ++it) {
    for (uint32_t i = 0; i < 64u; ++i) {
      c[i] = amc_f2h(attn_m1_det_rne16(amc_frand(&r, -8.0f, 8.0f)));
      a[i] = amc_f2h(attn_m1_det_rne16(amc_frand(&r, -8.0f, 8.0f)));
      b[i] = amc_f2h(attn_m1_det_rne16(amc_frand(&r, -1.0f, 1.0f)));
    }
    bad_rnd += hf_fma_batch(c, a, b, &n_rnd, &hz_rnd);
  }
  printf("ATTN M1 HF PRIM qfma: adversarial n=%u hazards=%u bad=%u, "
         "zero_sign n=%u bad=%u, random n=%u hazards=%u bad=%u\n",
         n_adv, hz_adv, bad_adv, n_zs, bad_zs, n_rnd, hz_rnd, bad_rnd);
  CHECK(hz_adv > 0u, "the adversarial family reaches no hazard");
  CHECK(bad_adv + bad_zs + bad_rnd == 0u, "hvx_hf_fma differs from fma16");

  /* hf add / sub / mul / max, * 0.125 and 0 + x: every finite fp16 a (both
     signs) against 4 random b (a quarter of them +-0). */
  uint32_t n_ops = 0, bad_op[6] = {0, 0, 0, 0, 0, 0};
  const HVX_Vector eighth = Q6_Vh_vsplat_R(0x3000), zero = Q6_V_vzero();
  for (uint32_t base = 0; base < 2u * 0x7C00u; base += 64u) {
    for (uint32_t i = 0; i < 64u; ++i) {
      const uint32_t x = base + i;
      a[i] = (uint16_t)(((x & 1u) << 15) | (x >> 1));
    }
    const HVX_Vector va = hf_load(a);
    for (int rep = 0; rep < 4; ++rep) {
      for (uint32_t i = 0; i < 64u; ++i) {
        const uint32_t k = amc_next(&r);
        b[i] = (uint16_t)((k & 0x8000u) |
                          ((k & 3u) == 0u ? 0u : (k >> 2) % 0x7C00u));
      }
      const HVX_Vector vb = hf_load(b);
      const HVX_Vector y[6] = {
        Q6_Vhf_vadd_VhfVhf(va, vb),     Q6_Vhf_vsub_VhfVhf(va, vb),
        Q6_Vhf_vmpy_VhfVhf(va, vb),     Q6_Vhf_vmax_VhfVhf(va, vb),
        Q6_Vhf_vmpy_VhfVhf(va, eighth), Q6_Vhf_vadd_VhfVhf(zero, va)};
      for (uint32_t i = 0; i < 64u; ++i) {
        const float x = amc_h2f(a[i]), z = amc_h2f(b[i]);
        const float ref[6] = {attn_m1_det_rne16(attn_m1_det_add(x, z)),
                              attn_m1_det_rne16(attn_m1_det_sub(x, z)),
                              attn_m1_det_rne16(attn_m1_det_mul(x, z)),
                              x > z ? x : z,
                              attn_m1_det_rne16(attn_m1_det_mul(x, 0.125f)),
                              attn_m1_det_add(0.0f, x)};
        for (int o = 0; o < 6; ++o) {
          if (fabsf(ref[o]) > 65504.0f) {
            continue;
          }
          const uint16_t got = hvx_emu_h(&y[o], (int)i);
          /* max: by value (the spec's max is canonicalised by + 0) */
          bad_op[o] += o == 3 ? amc_h2f(got) != ref[o] : got != amc_f2h(ref[o]);
        }
        ++n_ops;
      }
    }
  }
  printf("ATTN M1 HF PRIM hf ops, every finite fp16 x 4 random: n=%u bad "
         "add=%u sub=%u mul=%u max=%u mul0.125=%u zero_plus=%u\n",
         n_ops, bad_op[0], bad_op[1], bad_op[2], bad_op[3], bad_op[4],
         bad_op[5]);
  CHECK(bad_op[0] + bad_op[1] + bad_op[2] + bad_op[3] + bad_op[4] + bad_op[5] ==
          0u,
        "an hf op differs from rne16(f32 op)");

  /* hvx_hf_bits_rne and the kernel's vector form hvx_hf_round_row (#170
     round 2) == fp16 bits of rne16 at every f32 below 65520 (the positive
     half exhaustively, the negative half every 61st pattern), 64 values
     per vector call. */
  uint64_t n_cvt = 0;
  uint32_t bad_cvt = 0, bad_vcvt = 0, nb = 0;
  float xb[64];
  uint16_t want_b[64];
  for (uint32_t a = 0; a < 0x477FF000u; ++a) {
    for (uint32_t sg = 0; sg < (a % 61u == 0u ? 2u : 1u); ++sg) {
      const float x = attn_m1_det_float(a | (sg << 31));
      const uint16_t want = amc_f2h(attn_m1_det_rne16(x));
      bad_cvt += hvx_hf_bits_rne(x) != want;
      ++n_cvt;
      xb[nb] = x;
      want_b[nb++] = want;
      if (nb == 64u || a + 1u == 0x477FF000u) {
        const HVX_Vector y = hvx_hf_round_row(xb);
        for (uint32_t i = 0; i < nb; ++i) {
          bad_vcvt += hvx_emu_h(&y, (int)i) != want_b[i];
        }
        nb = 0;
      }
    }
  }
  /* ... and back: hvx_hf_store_sf == the f32 of every finite fp16 (the
     output widening; -0 and the subnormals included). */
  uint32_t n_wid = 0, bad_wid = 0;
  for (uint32_t base = 0; base < 0x10000u; base += 64u) {
    float o[64];
    for (uint32_t i = 0; i < 64u; ++i) {
      a[i] = (uint16_t)(base + i);
    }
    hvx_hf_store_sf(o, hf_load(a));
    for (uint32_t i = 0; i < 64u; ++i) {
      if ((a[i] & 0x7C00u) != 0x7C00u) {
        const float want = amc_h2f(a[i]);
        bad_wid += memcmp(&o[i], &want, sizeof(want)) != 0;
        ++n_wid;
      }
    }
  }
  printf("ATTN M1 HF PRIM f32 -> fp16 bits: %llu values, bad=%u, vector "
         "bad=%u; fp16 -> f32 vector: %u values, bad=%u\n",
         (unsigned long long)n_cvt, bad_cvt, bad_vcvt, n_wid, bad_wid);
  CHECK(bad_cvt == 0u, "hvx_hf_bits_rne differs from rne16");
  CHECK(bad_vcvt == 0u, "hvx_hf_round_row differs from rne16");
  CHECK(n_wid == 63488u && bad_wid == 0u,
        "hvx_hf_store_sf differs from the fp16 value (%u of %u)", bad_wid,
        n_wid);

  /* The score tree on random fp16 accumulators. */
  uint32_t bad_tree = 0;
  for (uint32_t it = 0; it < 4000u; ++it) {
    HVX_Vector acc[ATTN_M1_DET_ACC];
    uint16_t h[ATTN_M1_DET_ACC][64];
    for (uint32_t l = 0; l < ATTN_M1_DET_ACC; ++l) {
      for (uint32_t i = 0; i < 64u; ++i) {
        h[l][i] = amc_f2h(attn_m1_det_rne16(amc_frand(&r, -64.0f, 64.0f)));
      }
      acc[l] = hf_load(h[l]);
    }
    const HVX_Vector y = hvx_hf_score(acc, eighth);
    for (uint32_t i = 0; i < 64u; ++i) {
      float f[ATTN_M1_DET_ACC];
      for (uint32_t l = 0; l < ATTN_M1_DET_ACC; ++l) {
        f[l] = amc_h2f(h[l][i]);
      }
      const float s03 = attn_m1_det_rne16(
        attn_m1_det_add(attn_m1_det_rne16(attn_m1_det_add(f[0], f[1])),
                        attn_m1_det_rne16(attn_m1_det_add(f[2], f[3]))));
      const float s47 = attn_m1_det_rne16(
        attn_m1_det_add(attn_m1_det_rne16(attn_m1_det_add(f[4], f[5])),
                        attn_m1_det_rne16(attn_m1_det_add(f[6], f[7]))));
      const float t =
        attn_m1_det_add(0.0f, attn_m1_det_rne16(attn_m1_det_add(s03, s47)));
      bad_tree += hvx_emu_h(&y, (int)i) !=
                  amc_f2h(attn_m1_det_rne16(attn_m1_det_mul(t, 0.125f)));
    }
  }

  /* exp16 at every fp16 d <= 0: +0 and 0x8000 .. 0xFBFF. */
  uint32_t n_exp = 0, bad_exp = 0;
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  for (uint32_t base = 0x7FFFu; base < 0xFC00u; base += 64u) {
    uint32_t live = 0;
    for (uint32_t i = 0; i < 64u; ++i) {
      const uint32_t x = base + i;
      a[i] = x == 0x7FFFu ? 0u : x < 0xFC00u ? (uint16_t)x : 0x8000u;
      live += x < 0xFC00u;
    }
    const HVX_Vector y = hvx_hf_exp16(hf_load(a), one);
    for (uint32_t i = 0; i < live; ++i) {
      bad_exp +=
        hvx_emu_h(&y, (int)i) != amc_f2h(attn_m1_det_exp16(amc_h2f(a[i])));
      ++n_exp;
    }
  }

  /* The divide on check_prim_div16's hard quotients, l per lane. */
  uint32_t bad_div = 0;
  for (uint32_t base = 0; base < g_div_hard; base += 64u) {
    float want[64];
    for (uint32_t i = 0; i < 64u; ++i) {
      const uint32_t j = base + i < g_div_hard ? base + i : base;
      a[i] = amc_f2h(g_div_hard_e[j]);
      b[i] = amc_f2h(g_div_hard_l[j]);
      want[i] =
        attn_m1_det_rne16(attn_m1_det_div(g_div_hard_e[j], g_div_hard_l[j]));
    }
    const HVX_VectorPair l = hvx_hf_widen(hf_load(b), one);
    const HVX_VectorPair rc = Q6_W_vcombine_VV(hvx_recip_det_sf(Q6_V_hi_W(l)),
                                               hvx_recip_det_sf(Q6_V_lo_W(l)));
    const HVX_Vector y = hvx_hf_div16(hf_load(a), l, rc, one);
    for (uint32_t i = 0; i < 64u; ++i) {
      bad_div += hvx_emu_h(&y, (int)i) != amc_f2h(want[i]);
    }
  }
  printf("ATTN M1 HF PRIM score tree n=%u bad=%u; exp16 at every fp16 d <= 0 "
         "(%u) bad=%u; div16 on %u hard quotients bad=%u\n",
         4000u * 64u, bad_tree, n_exp, bad_exp, g_div_hard, bad_div);
  CHECK(n_exp == 31745u, "exp16 swept %u values", n_exp);
  CHECK(g_div_hard > 0u && g_div_hard < N_DIV_HARD_MAX,
        "div16 hard quotients: %u", g_div_hard);
  CHECK(bad_tree + bad_exp + bad_div == 0u,
        "the hf tree, exp16 or divide differs from the spec");
  if (bad_adv + bad_zs + bad_rnd + bad_tree + bad_exp + bad_div + bad_cvt +
          bad_vcvt + bad_wid ==
        0u &&
      bad_op[0] + bad_op[1] + bad_op[2] + bad_op[3] + bad_op[4] + bad_op[5] ==
        0u &&
      hz_adv > 0u && n_exp == 31745u) {
    printf("ATTN M1 HF PRIM OK\n");
  }
}

/* ==== 3. the kernel against the spec ==================================== */

enum { N_LAYERS = 2, LAYER = 1 };
/** @brief LFM2.5's shape, SHAPES[0]: the fixed-size cases run at it. */
enum { LFM_KV = 8, LFM_Q = 32, HD = 64 };
/** @brief The shape under test, set per row of SHAPES by set_shape(). */
static uint32_t N_KV, GQA, N_Q;
static amc_rng g_rng;
/** @brief (n_kv, gqa): LFM2.5 first (the q-head-pair units), the hd64
 *         fixture's, an odd gqa (one unit per kv head) and gqa 8 (two PV
 *         groups and two ET transpose groups per kv head). */
static const uint32_t SHAPES[4][2] = {{8, 4}, {1, 2}, {2, 3}, {1, 8}};

static void set_shape(const uint32_t *shape) {
  g_rng.s = 0x81810001u; /* every shape sees the same input stream */
  N_KV = shape[0];
  GQA = shape[1];
  N_Q = N_KV * GQA;
}
static const float SCALE = 0.125f;
static const uint32_t POOLS[3] = {0u, 3u, 7u};

static hvx_attn_m1_ctx *make_ctx(uint32_t max_seq, hvx_worker_pool *pool) {
  int err = -1;
  hvx_attn_m1_ctx *ctx =
    hvx_attn_m1_create(N_LAYERS, N_KV, GQA, HD, max_seq, pool, &err);
  CHECK(ctx && err == AEE_SUCCESS, "create(max_seq=%u): err=%d", max_seq, err);
  return ctx;
}

static int g_prof_fail = 0;

/**
 * @brief The phase words (#146): the same forward again (a rewind to L-1
 *        and the same row) with the words requested must give the same out
 *        and stats bytes, every pcycle word must have been taken (the host
 *        stub's counter is monotonic, so a bracket that ran reads > 0),
 *        LANES must be min(units, workers + 1) and POOL >= BUSY_MAX. The
 *        qtimer stub reads 0, so CALL_QT is a device-only word.
 * @return the number of failed conditions
 */
static uint32_t check_phase_words(hvx_attn_m1_ctx *ctx, uint32_t L,
                                  uint32_t workers, const float *q,
                                  const float *k, const float *v,
                                  const float *out_ref,
                                  const float *stats_ref) {
  float *out = malloc((size_t)N_Q * HD * sizeof(float));
  float *stats = malloc(2u * N_Q * sizeof(float));
  uint32_t w[ATTN_M1_PROF_WORDS];
  memset(w, 0xA5, sizeof(w));
  const int rc =
    hvx_attn_m1_forward_prof(ctx, LAYER, L - 1u, SCALE, q, k, v, out, stats, w);
  /* LANES is the score run's: one unit per (kv head, 64-position tile). */
  const uint32_t units = N_KV * ((L + 63u) / 64u);
  const uint32_t lanes =
    workers == 0u ? 1u : (units < workers + 1u ? units : workers + 1u);
  uint32_t bad = (rc != AEE_SUCCESS);
  bad += count_bad(out, out_ref, (size_t)N_Q * HD) != 0u;
  bad += count_bad(stats, stats_ref, 2u * N_Q) != 0u;
  bad += w[ATTN_M1_PROF_LANES] != lanes;
  static const uint32_t taken[] = {ATTN_M1_PROF_APPEND,   ATTN_M1_PROF_POOL,
                                   ATTN_M1_PROF_SCORES,   ATTN_M1_PROF_SOFTMAX,
                                   ATTN_M1_PROF_PV,       ATTN_M1_PROF_BUSY_MAX,
                                   ATTN_M1_PROF_START_MAX};
  for (size_t i = 0; i < sizeof(taken) / sizeof(taken[0]); ++i) {
    bad += w[taken[i]] == 0u;
  }
  bad += w[ATTN_M1_PROF_POOL] < w[ATTN_M1_PROF_BUSY_MAX];
  if (bad) {
    printf("  phase words L=%u workers=%u rc=%d:", L, workers, rc);
    for (uint32_t i = 0; i < ATTN_M1_PROF_WORDS; ++i) {
      printf(" %u", w[i]);
    }
    printf(" (lanes expected %u)\n", lanes);
    g_prof_fail = 1;
  }
  free(out);
  free(stats);
  return bad;
}

/**
 * @brief One length: kv_append of L-1 rows, forward of the last, at each
 *        worker count; byte-equal across counts and against the spec.
 */
static void check_length(uint32_t L, uint32_t max_seq,
                         hvx_worker_pool *const pools[3]) {
  uint32_t prof_bad = 0;
  float *q = malloc((size_t)N_Q * HD * sizeof(float));
  float *k = malloc((size_t)L * N_KV * HD * sizeof(float));
  float *v = malloc((size_t)L * N_KV * HD * sizeof(float));
  float *out[3], *stats[3];
  float *out_det = malloc((size_t)N_Q * HD * sizeof(float));
  float *stats_det = malloc(2u * N_Q * sizeof(float));
  amc_fill_q(&g_rng, q, N_Q, HD);
  amc_fill_kv(&g_rng, k, v, L, N_KV, GQA, HD);
  const uint32_t planted = amc_plant_pv(q, k, v, L, N_KV, GQA, HD);

  for (int p = 0; p < 3; ++p) {
    out[p] = malloc((size_t)N_Q * HD * sizeof(float));
    stats[p] = malloc(2u * N_Q * sizeof(float));
    memset(out[p], 0xA5, (size_t)N_Q * HD * sizeof(float));
    memset(stats[p], 0xA5, 2u * N_Q * sizeof(float));
    hvx_attn_m1_ctx *ctx = make_ctx(max_seq, pools[p]);
    const size_t last = (size_t)(L - 1u) * N_KV * HD;
    int rc = hvx_attn_m1_kv_append(ctx, LAYER, 0u, L - 1u, k, v);
    CHECK(rc == AEE_SUCCESS, "L=%u kv_append rc=%d", L, rc);
    rc = hvx_attn_m1_forward(ctx, LAYER, L - 1u, SCALE, q, k + last, v + last,
                             out[p], stats[p]);
    CHECK(rc == AEE_SUCCESS, "L=%u forward rc=%d", L, rc);
    CHECK(ctx->kv_len[LAYER] == L, "L=%u kv_len=%u", L, ctx->kv_len[LAYER]);
    prof_bad += check_phase_words(ctx, L, POOLS[p], q, k + last, v + last,
                                  out[p], stats[p]);
    hvx_attn_m1_free(ctx);
  }
  spec_forward(q, k, v, N_KV, GQA, HD, L, max_seq, out_det, stats_det);

  uint32_t pool_bad = 0;
  for (int p = 1; p < 3; ++p) {
    pool_bad += count_bad(out[p], out[0], (size_t)N_Q * HD);
    pool_bad += count_bad(stats[p], stats[0], 2u * N_Q);
  }
  const uint32_t bad_out = count_bad(out[0], out_det, (size_t)N_Q * HD);
  const uint32_t bad_stats = count_bad(stats[0], stats_det, 2u * N_Q);

  printf("ATTN M1 shape=(%u,%u,%d) L=%u max_seq=%u workers={0,3,7} bad=%u "
         "bad_stats=%u pool_bad=%u prof_bad=%u pv_cases=%u\n",
         N_KV, GQA, HD, L, max_seq, bad_out, bad_stats, pool_bad, prof_bad,
         planted);
  if (bad_out || bad_stats) {
    for (uint32_t hq = 0; hq < N_Q; ++hq) {
      if (count_bad(out[0] + hq * HD, out_det + hq * HD, HD) ||
          count_bad(stats[0] + 2 * hq, stats_det + 2 * hq, 2u)) {
        printf("  first divergence: head %u hvx (m, l) = (%a, %a) spec "
               "(%a, %a)\n",
               hq, stats[0][2 * hq], stats[0][2 * hq + 1], stats_det[2 * hq],
               stats_det[2 * hq + 1]);
        break;
      }
    }
  }
  CHECK(bad_out == 0u && bad_stats == 0u, "L=%u: HVX differs from the spec", L);
  CHECK(pool_bad == 0u, "L=%u: worker counts disagree", L);
  CHECK(prof_bad == 0u, "L=%u: the phase words are wrong or change the output",
        L);

  for (int p = 0; p < 3; ++p) {
    free(out[p]);
    free(stats[p]);
  }
  free(q);
  free(k);
  free(v);
  free(out_det);
  free(stats_det);
}

/** @brief L forward calls from an empty cache vs one kv_append of L-1 rows
 *         and a forward of the last: caches and outputs byte-equal. */
static void check_append_chain(uint32_t L, hvx_worker_pool *pool) {
  const uint32_t max_seq = 1024u;
  float *q = malloc((size_t)N_Q * HD * sizeof(float));
  float *k = malloc((size_t)L * N_KV * HD * sizeof(float));
  float *v = malloc((size_t)L * N_KV * HD * sizeof(float));
  float *out_a = malloc((size_t)N_Q * HD * sizeof(float));
  float *out_b = malloc((size_t)N_Q * HD * sizeof(float));
  amc_fill_q(&g_rng, q, N_Q, HD);
  amc_fill_kv(&g_rng, k, v, L, N_KV, GQA, HD);
  hvx_attn_m1_ctx *a = make_ctx(max_seq, pool), *b = make_ctx(max_seq, pool);
  /* The chain touches layer 0 too, so an unwritten layer 0 in b would show
     up in the cache compare if the layer offset were wrong. */
  for (uint32_t p = 0; p < L; ++p) {
    const size_t row = (size_t)p * N_KV * HD;
    int rc =
      hvx_attn_m1_forward(a, LAYER, p, SCALE, q, k + row, v + row, out_a, NULL);
    CHECK(rc == AEE_SUCCESS, "chain pos=%u rc=%d", p, rc);
  }
  const size_t last = (size_t)(L - 1u) * N_KV * HD;
  int rc = hvx_attn_m1_kv_append(b, LAYER, 0u, L - 1u, k, v);
  CHECK(rc == AEE_SUCCESS, "bulk kv_append rc=%d", rc);
  rc = hvx_attn_m1_forward(b, LAYER, L - 1u, SCALE, q, k + last, v + last,
                           out_b, NULL);
  CHECK(rc == AEE_SUCCESS, "bulk forward rc=%d", rc);
  const int kt_eq =
    memcmp(a->kt, b->kt, a->cache_halves * sizeof(uint16_t)) == 0;
  const int v_eq = memcmp(a->v, b->v, a->cache_halves * sizeof(uint16_t)) == 0;
  const uint32_t bad = count_bad(out_a, out_b, (size_t)N_Q * HD);
  printf("ATTN M1 append-chain L=%u vs bulk: Kt_equal=%d V_equal=%d "
         "out_bad=%u kv_len=%u/%u\n",
         L, kt_eq, v_eq, bad, a->kv_len[LAYER], b->kv_len[LAYER]);
  CHECK(kt_eq && v_eq, "append chain leaves a different cache");
  CHECK(bad == 0u, "append chain's last output differs from bulk");
  CHECK(a->kv_len[LAYER] == L && b->kv_len[LAYER] == L, "kv_len after chain");
  hvx_attn_m1_free(a);
  hvx_attn_m1_free(b);
  free(q);
  free(k);
  free(v);
  free(out_a);
  free(out_b);
}

/** @brief L = 1: e = [1], l = 1, p = 1, out = fma16(0, 1, v16) = rne16(v)
 *         exactly, for every q head. */
static void check_identity(hvx_worker_pool *pool) {
  float q[LFM_Q * HD], k[LFM_KV * HD], v[LFM_KV * HD], out[LFM_Q * HD],
    stats[2 * LFM_Q];
  amc_fill_q(&g_rng, q, N_Q, HD);
  amc_fill_row(&g_rng, k, N_KV * HD, 0);
  amc_fill_row(&g_rng, v, N_KV * HD, 0);
  hvx_attn_m1_ctx *ctx = make_ctx(1024u, pool);
  const int rc = hvx_attn_m1_forward(ctx, 0u, 0u, SCALE, q, k, v, out, stats);
  CHECK(rc == AEE_SUCCESS, "L=1 forward rc=%d", rc);
  uint32_t bad = 0, bad_l = 0;
  for (uint32_t hq = 0; hq < N_Q; ++hq) {
    for (uint32_t d = 0; d < HD; ++d) {
      const float ref = attn_m1_det_rne16(v[(hq / GQA) * HD + d]);
      bad += count_bad(&out[hq * HD + d], &ref, 1);
    }
    bad_l += (stats[2 * hq + 1] == 1.0f) ? 0u : 1u;
  }
  printf("ATTN M1 L=1: out == rne16(v) bad=%u, l == 1.0f bad=%u\n", bad, bad_l);
  CHECK(bad == 0u, "L=1: out is not rne16(v)");
  CHECK(bad_l == 0u, "L=1: the sum is not exactly 1.0f");
  hvx_attn_m1_free(ctx);
}

/** @brief A division tie: scores (0, 0, -17) give e = (1, 1, 2^-24), l = 2,
 *         and p = 2^-25 exactly half-way between 0 and 2^-24, which rounds
 *         to even (0). Every q head sees it; kernel and spec must agree. */
static void check_division_tie(hvx_worker_pool *pool) {
  float q[LFM_Q * HD], k[3 * LFM_KV * HD], v[3 * LFM_KV * HD], out[LFM_Q * HD],
    stats[2 * LFM_Q], out_det[LFM_Q * HD], stats_det[2 * LFM_Q];
  memset(q, 0, sizeof(q));
  memset(k, 0, sizeof(k));
  for (uint32_t hq = 0; hq < N_Q; ++hq) {
    q[hq * HD] = 1.0f;
  }
  for (uint32_t h = 0; h < N_KV; ++h) {
    k[(2u * N_KV + h) * HD] = -136.0f; /* s = -136 / 8 = -17 */
  }
  amc_fill_row(&g_rng, v, 3u * N_KV * HD, 0);
  hvx_attn_m1_ctx *ctx = make_ctx(1024u, pool);
  int rc = hvx_attn_m1_kv_append(ctx, 0u, 0u, 2u, k, v);
  rc |= hvx_attn_m1_forward(ctx, 0u, 2u, SCALE, q, k + 2u * N_KV * HD,
                            v + 2u * N_KV * HD, out, stats);
  CHECK(rc == AEE_SUCCESS, "division tie: rc=%d", rc);
  spec_forward(q, k, v, N_KV, GQA, HD, 3u, 1024u, out_det, stats_det);
  const float e = attn_m1_det_exp16(-17.0f), l = stats_det[1];
  const float p = attn_m1_det_rne16(attn_m1_det_div(e, l));
  const uint32_t bad = count_bad(out, out_det, (size_t)N_Q * HD) +
                       count_bad(stats, stats_det, 2u * N_Q);
  printf("ATTN M1 division tie: e=%a l=%a p=%a (tie to even: 0) bad=%u\n", e, l,
         p, bad);
  CHECK(e == ldexpf(1.0f, -24) && l == 2.0f && p == 0.0f,
        "the tie case is not a tie");
  CHECK(bad == 0u, "division tie: HVX differs from the spec");
  hvx_attn_m1_free(ctx);
}

/** @brief The error codes and the rewind rule. */
static void check_errors(hvx_worker_pool *pool) {
  int err = 0;
  hvx_attn_m1_ctx *bad = hvx_attn_m1_create(1, N_KV, GQA, HD, 100u, pool, &err);
  CHECK(!bad && err == AEE_EINVALIDFORMAT, "max_seq 100: ctx=%p err=%d",
        (void *)bad, err);
  /* head_dim is 64 only (#152: the CPU order's 8 accumulators and the
     RoPE's head size); 32, 48 and 128 register-time errors. */
  static const uint32_t bad_hd[] = {32u, 48u, 128u};
  for (size_t i = 0; i < sizeof(bad_hd) / sizeof(bad_hd[0]); ++i) {
    bad = hvx_attn_m1_create(1, N_KV, GQA, bad_hd[i], 1024u, pool, &err);
    CHECK(!bad && err == AEE_EINVALIDFORMAT, "head_dim %u: ctx=%p err=%d",
          bad_hd[i], (void *)bad, err);
  }
  /* The register-file bounds are register-time errors, not forward-time. */
  bad = hvx_attn_m1_create(1, N_KV, 16u, HD, 1024u, pool, &err);
  CHECK(!bad && err == AEE_EINVALIDFORMAT, "gqa 16: ctx=%p err=%d", (void *)bad,
        err);

  float q[LFM_Q * HD], k[LFM_KV * HD], v[LFM_KV * HD], out[LFM_Q * HD];
  amc_fill_q(&g_rng, q, N_Q, HD);
  amc_fill_row(&g_rng, k, N_KV * HD, 0);
  amc_fill_row(&g_rng, v, N_KV * HD, 0);
  hvx_attn_m1_ctx *ctx = make_ctx(64u, pool);
  int rc = hvx_attn_m1_forward(ctx, 0u, 64u, SCALE, q, k, v, out, NULL);
  CHECK(rc == AEE_EINVALIDFORMAT, "pos == max_seq: rc=%d", rc);
  rc = hvx_attn_m1_forward(ctx, N_LAYERS, 0u, SCALE, q, k, v, out, NULL);
  CHECK(rc == AEE_EINVALIDFORMAT, "layer out of range: rc=%d", rc);
  rc = hvx_attn_m1_forward(ctx, 0u, 1u, SCALE, q, k, v, out, NULL);
  CHECK(rc == AEE_EBADSTATE, "hole at pos 1 of an empty layer: rc=%d", rc);
  rc = hvx_attn_m1_kv_append(ctx, 0u, 1u, 1u, k, v);
  CHECK(rc == AEE_EBADSTATE, "kv_append hole: rc=%d", rc);
  rc = hvx_attn_m1_kv_append(ctx, 0u, 0u, 65u, k, v);
  CHECK(rc == AEE_EINVALIDFORMAT, "kv_append past max_seq: rc=%d", rc);
  rc = hvx_attn_m1_forward(NULL, 0u, 0u, SCALE, q, k, v, out, NULL);
  CHECK(rc == AEE_EBADSTATE, "NULL ctx: rc=%d", rc);
  /* #170: the scale is one hf multiply, so it must be an fp16 value */
  rc = hvx_attn_m1_forward(ctx, 0u, 0u, 0.1f, q, k, v, out, NULL);
  CHECK(rc == AEE_EINVALIDFORMAT, "scale 0.1 (not fp16): rc=%d", rc);
  for (uint32_t p = 0; p < 5u; ++p) {
    rc = hvx_attn_m1_forward(ctx, 0u, p, SCALE, q, k, v, out, NULL);
    CHECK(rc == AEE_SUCCESS, "pos %u rc=%d", p, rc);
  }
  rc = hvx_attn_m1_forward(ctx, 0u, 2u, SCALE, q, k, v, out, NULL);
  CHECK(rc == AEE_SUCCESS && ctx->kv_len[0] == 3u, "rewind to 2: rc=%d len=%u",
        rc, ctx->kv_len[0]);
  CHECK(ctx->kv_len[1] == 0u, "layer 1 touched: len=%u", ctx->kv_len[1]);
  printf("ATTN M1 error codes and rewind OK\n");
  hvx_attn_m1_free(ctx);
}

int main(void) {
  check_cpu_order();
  check_prim_rne16();
  check_prim_fma16();
  check_prim_div16();
  check_hf_prim();

  hvx_worker_pool *pools[3];
  for (int p = 0; p < 3; ++p) {
    pools[p] = POOLS[p] ? hvx_worker_pool_create(POOLS[p]) : NULL;
    CHECK(POOLS[p] == 0u || pools[p], "pool of %u workers", POOLS[p]);
  }

  static const uint32_t lengths[] = {1u,   63u,  64u,   65u,
                                     512u, 513u, 1024u, 1536u};
  for (size_t s = sizeof(SHAPES) / sizeof(SHAPES[0]); s-- > 0;) {
    set_shape(SHAPES[s]); /* LFM2.5 last: the cases below use it */
    for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
      check_length(lengths[i], 2048u, pools);
    }
  }
  check_append_chain(65u, pools[1]);
  check_identity(pools[2]);
  check_division_tie(pools[1]);
  check_errors(pools[1]);

  for (int p = 0; p < 3; ++p) {
    hvx_worker_pool_destroy(pools[p]);
  }
  if (!g_prof_fail) {
    printf("ATTN M1 PHASES OK\n");
  }
  if (g_fail) {
    printf("ATTN M1 CHECK FAILED\n");
    return 1;
  }
  printf("ATTN M1 BIT-IDENTICAL\n");
  return 0;
}
