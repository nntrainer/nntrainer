// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   attn_m1_host_check.c
 * @date   27 Sep 2026
 * @brief  Host check: the real HVX decode-attention kernel, compiled
 *         against hvx_emu/ and the real worker pool on pthreads, is
 *         bit-identical to attn_m1_det.h at every worker count and within
 *         tolerance of a double reference
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Two halves (plan 81 section 1):
 *  1. BIT IDENTITY. hvx_attn_m1_f32.c -- the skel's own source, not a
 *     stand-in -- runs on the lane-by-lane emulation with the pthread
 *     worker pool (stub/qurt.h) at 0, 3 and 7 workers, and is memcmp'd
 *     against the scalar spec for L = 1, 63, 64, 65, 512, 1024 at three
 *     shapes (n_kv, gqa, head_dim) -- LFM2.5's (8, 4, 64), the hd64
 *     fixture's (1, 2, 64) and (2, 3, 32), which only the generic path
 *     serves (#146) -- at max_seq 1024, and LFM2.5 once at 2048. The three
 *     worker counts must agree byte for byte: the head split is
 *     deterministic by construction and this proves it. Each forward is
 *     repeated with the phase words requested (#146): the same bytes out,
 *     every word taken (ATTN M1 PHASES OK).
 *     Two structural cases: an append chain of L forward calls leaves the
 *     cache byte-equal to one kv_append of L rows and the same last
 *     output; at L = 1 the output is v * recip_det(1.0f) bit for bit
 *     (recip_det(1.0f) is 0x3F7FFFFF, one ulp below 1 -- swiglu_det.h --
 *     so the identity holds to that ulp, not exactly; plan 81 section 3.2
 *     expected exactly 1, which the number below corrects). Plus the
 *     error codes: bad shape, position past max_seq, a hole, a rewind.
 *     What this proves: the cache layout and indexing, the block and tail
 *     masks, the reduction trees, the operation order, the head split.
 *     What it rests on: one Vsf op = one IEEE op (rule 24), re-checked on
 *     the device by HvxAttnM1.*.
 *  2. TOLERANCE. The spec against plain softmax(q.Kt * scale).V in double:
 *     max_abs_err / max|V| per length, asserted <= 2^-13 (a sanity bound;
 *     the bit identity is the contract, rule 25).
 *
 * Inputs: an LCG from a fixed seed, q/k/v in [-4, 4], plus fixed rows:
 * q head 0 all zero, head 1 all subnormal (+-1e-39, kept: rule 24), head 2
 * large (|q| ~ 1e3, so its softmax hits exp_det's clamp on almost every
 * position); cache position 0 zero k and v, position 1 subnormal k and v,
 * position 2 large k (a one-hot or a clamped-out position, by sign).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>

#include "attn_m1_det.h"
#include "hvx_attn_m1_f32.h"
#include "hvx_worker_pool.h"

static int g_fail = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL: " __VA_ARGS__);                                            \
      printf("\n");                                                            \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

enum { N_LAYERS = 2, LAYER = 1 };
/** @brief LFM2.5's shape, SHAPES[0]: the fixed-size cases run at it. */
enum { LFM_KV = 8, LFM_Q = 32, LFM_HD = 64 };
static uint32_t g_seed;
/** @brief The shape under test, set per row of SHAPES by set_shape(). */
static int N_KV, GQA, HD, N_Q;
/** @brief (n_kv, gqa, head_dim): LFM2.5 first (the specialised path, and
 *         the shape of the append-chain, identity and error cases), the
 *         hd64 fixture's, and one the kernel serves on its generic path. */
static const int SHAPES[3][3] = {{8, 4, 64}, {1, 2, 64}, {2, 3, 32}};

static void set_shape(const int *shape) {
  g_seed = 0x81810001u; /* every shape sees the same input stream */
  N_KV = shape[0];
  GQA = shape[1];
  HD = shape[2];
  N_Q = N_KV * GQA;
}
static const float SCALE = 0.125f;
static const uint32_t POOLS[3] = {0u, 3u, 7u};

/* ---- inputs ------------------------------------------------------------ */

static float frand(float lo, float hi) {
  g_seed = g_seed * 1664525u + 1013904223u;
  return lo + (hi - lo) * ((float)(g_seed >> 8) / 16777216.0f);
}
/* Kind 1: zeros; 2: subnormal +-1e-39; 3: large; else random [-4, 4]. */
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
      x[i] = frand(-1e3f, 1e3f);
      break;
    default:
      x[i] = frand(-4.0f, 4.0f);
    }
  }
}

/** @brief q [N_Q][HD] with heads 0..2 the fixed kinds. */
static void fill_q(float *q) {
  for (int h = 0; h < N_Q; ++h) {
    fill_row(q + h * HD, HD, (h < 3) ? h + 1 : 0);
  }
}

/** @brief L rows of k and v, [L][N_KV][HD]: positions 0 and 1 the zero and
 *         subnormal kinds in both, position 2 a large k with a random v. */
static void fill_kv(float *k, float *v, uint32_t L) {
  for (uint32_t p = 0; p < L; ++p) {
    const int kind = (p == 0u) ? 1 : (p == 1u) ? 2 : 0;
    fill_row(k + (size_t)p * N_KV * HD, N_KV * HD, (p == 2u) ? 3 : kind);
    fill_row(v + (size_t)p * N_KV * HD, N_KV * HD, kind);
  }
}

/* ---- the spec and the double reference --------------------------------- */

/** @brief The spec's cache for one layer from L rows, then its forward. */
static void spec_forward(const float *q, const float *k, const float *v,
                         uint32_t L, uint32_t max_seq, float *out,
                         float *stats) {
  float *kt = calloc((size_t)N_KV * HD * max_seq, sizeof(float));
  float *vv = calloc((size_t)N_KV * max_seq * HD, sizeof(float));
  float *e = malloc((size_t)L * sizeof(float));
  for (uint32_t p = 0; p < L; ++p) {
    for (uint32_t h = 0; h < (uint32_t)N_KV; ++h) {
      attn_m1_det_append(kt + (size_t)h * HD * max_seq,
                         vv + (size_t)h * max_seq * HD, HD, max_seq, p,
                         k + ((size_t)p * N_KV + h) * HD,
                         v + ((size_t)p * N_KV + h) * HD);
    }
  }
  attn_m1_det_forward(q, kt, vv, N_KV, GQA, HD, max_seq, L, SCALE, e, out,
                      stats);
  free(kt);
  free(vv);
  free(e);
}

/** @brief max_abs_err / max|V| of @a out against softmax(q.Kt*scale).V in
 *         double, over every q head. */
static double double_ref_err(const float *q, const float *k, const float *v,
                             uint32_t L, const float *out) {
  double *s = malloc((size_t)L * sizeof(double));
  double worst = 0.0;
  for (int hq = 0; hq < N_Q; ++hq) {
    const int h = hq / GQA;
    double m = -INFINITY, den = 0.0;
    for (uint32_t p = 0; p < L; ++p) {
      double acc = 0.0;
      for (int d = 0; d < HD; ++d) {
        acc +=
          (double)q[hq * HD + d] * (double)k[((size_t)p * N_KV + h) * HD + d];
        den = fmax(den, fabs((double)v[((size_t)p * N_KV + h) * HD + d]));
      }
      s[p] = acc * (double)SCALE;
      m = fmax(m, s[p]);
    }
    double l = 0.0;
    for (uint32_t p = 0; p < L; ++p) {
      s[p] = exp(s[p] - m);
      l += s[p];
    }
    for (int d = 0; d < HD; ++d) {
      double o = 0.0;
      for (uint32_t p = 0; p < L; ++p) {
        o += s[p] * (double)v[((size_t)p * N_KV + h) * HD + d];
      }
      o /= l;
      const double err = fabs((double)out[hq * HD + d] - o);
      if (den > 0.0 && err / den > worst) {
        worst = err / den;
      }
    }
  }
  free(s);
  return worst;
}

/* ---- the kernel ---------------------------------------------------------- */

static hvx_attn_m1_ctx *make_ctx(uint32_t max_seq, hvx_worker_pool *pool) {
  int err = -1;
  hvx_attn_m1_ctx *ctx =
    hvx_attn_m1_create(N_LAYERS, N_KV, GQA, HD, max_seq, pool, &err);
  CHECK(ctx && err == AEE_SUCCESS, "create(max_seq=%u): err=%d", max_seq, err);
  return ctx;
}

static uint32_t count_bad(const float *a, const float *b, size_t n) {
  uint32_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    bad += memcmp(&a[i], &b[i], sizeof(float)) ? 1u : 0u;
  }
  return bad;
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
  /* One unit per (kv head, q-head pair), or per kv head at an odd gqa. */
  const uint32_t units = (uint32_t)(GQA % 2 == 0 ? N_KV * GQA / 2 : N_KV);
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
  fill_q(q);
  fill_kv(k, v, L);

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
  spec_forward(q, k, v, L, max_seq, out_det, stats_det);

  uint32_t pool_bad = 0;
  for (int p = 1; p < 3; ++p) {
    pool_bad += count_bad(out[p], out[0], (size_t)N_Q * HD);
    pool_bad += count_bad(stats[p], stats[0], 2u * N_Q);
  }
  const uint32_t bad_out = count_bad(out[0], out_det, (size_t)N_Q * HD);
  const uint32_t bad_stats = count_bad(stats[0], stats_det, 2u * N_Q);
  const double err = double_ref_err(q, k, v, L, out_det);

  printf("ATTN M1 shape=(%d,%d,%d) L=%u max_seq=%u workers={0,3,7} bad=%u "
         "bad_stats=%u pool_bad=%u prof_bad=%u err/max|V|(double)=%.3e\n",
         N_KV, GQA, HD, L, max_seq, bad_out, bad_stats, pool_bad, prof_bad,
         err);
  if (bad_out || bad_stats) {
    for (int hq = 0; hq < N_Q; ++hq) {
      if (count_bad(out[0] + hq * HD, out_det + hq * HD, HD) ||
          count_bad(stats[0] + 2 * hq, stats_det + 2 * hq, 2u)) {
        printf("  first divergence: head %d hvx (m, l) = (%a, %a) spec "
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
  CHECK(err <= ldexp(1.0, -13),
        "L=%u: %.3e of max|V| from the double "
        "reference (bound 2^-13)",
        L, err);

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
  fill_q(q);
  fill_kv(k, v, L);
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
  const int kt_eq = memcmp(a->kt, b->kt, a->cache_floats * sizeof(float)) == 0;
  const int v_eq = memcmp(a->v, b->v, a->cache_floats * sizeof(float)) == 0;
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

/** @brief L = 1: e = [1.0], l = 1.0, out = v * recip_det(1.0f) per head. */
static void check_identity(hvx_worker_pool *pool) {
  float q[LFM_Q * LFM_HD], k[LFM_KV * LFM_HD], v[LFM_KV * LFM_HD],
    out[LFM_Q * LFM_HD], stats[2 * LFM_Q];
  float ref[LFM_Q * LFM_HD] = {0.0f}; /* gcc cannot see N_Q == LFM_Q */
  fill_q(q);
  fill_row(k, N_KV * HD, 0);
  fill_row(v, N_KV * HD, 0);
  hvx_attn_m1_ctx *ctx = make_ctx(1024u, pool);
  const int rc = hvx_attn_m1_forward(ctx, 0u, 0u, SCALE, q, k, v, out, stats);
  CHECK(rc == AEE_SUCCESS, "L=1 forward rc=%d", rc);
  const float r1 = swiglu_det_recip(1.0f);
  uint32_t r1_bits;
  memcpy(&r1_bits, &r1, sizeof(r1_bits));
  for (int hq = 0; hq < N_Q; ++hq) {
    for (int d = 0; d < HD; ++d) {
      ref[hq * HD + d] = attn_m1_det_mul(v[(hq / GQA) * HD + d], r1);
    }
  }
  const uint32_t bad = count_bad(out, ref, (size_t)N_Q * HD);
  uint32_t bad_exact = 0, bad_l = 0;
  for (int hq = 0; hq < N_Q; ++hq) {
    bad_exact += count_bad(out + hq * HD, v + (hq / GQA) * HD, HD);
    bad_l += (stats[2 * hq + 1] == 1.0f) ? 0u : 1u;
  }
  printf("ATTN M1 L=1: out == v * recip_det(1.0f) bad=%u, l == 1.0f bad=%u, "
         "recip_det(1.0f) = 0x%08X (out == v exactly: bad=%u)\n",
         bad, bad_l, r1_bits, bad_exact);
  CHECK(bad == 0u, "L=1: out is not v * recip_det(1.0f)");
  CHECK(bad_l == 0u, "L=1: the sum is not exactly 1.0f");
  hvx_attn_m1_free(ctx);
}

/** @brief The error codes and the rewind rule. */
static void check_errors(hvx_worker_pool *pool) {
  int err = 0;
  hvx_attn_m1_ctx *bad = hvx_attn_m1_create(1, N_KV, GQA, HD, 100u, pool, &err);
  CHECK(!bad && err == AEE_EINVALIDFORMAT, "max_seq 100: ctx=%p err=%d",
        (void *)bad, err);
  bad = hvx_attn_m1_create(1, N_KV, GQA, 48u, 1024u, pool, &err);
  CHECK(!bad && err == AEE_EINVALIDFORMAT, "head_dim 48: ctx=%p err=%d",
        (void *)bad, err);
  /* The register-file bounds are register-time errors, not forward-time. */
  bad = hvx_attn_m1_create(1, N_KV, 16u, HD, 1024u, pool, &err);
  CHECK(!bad && err == AEE_EINVALIDFORMAT, "gqa 16: ctx=%p err=%d", (void *)bad,
        err);
  bad = hvx_attn_m1_create(1, N_KV, GQA, 256u, 1024u, pool, &err);
  CHECK(!bad && err == AEE_EINVALIDFORMAT, "head_dim 256: ctx=%p err=%d",
        (void *)bad, err);

  float q[LFM_Q * LFM_HD], k[LFM_KV * LFM_HD], v[LFM_KV * LFM_HD],
    out[LFM_Q * LFM_HD];
  fill_q(q);
  fill_row(k, N_KV * HD, 0);
  fill_row(v, N_KV * HD, 0);
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
  hvx_worker_pool *pools[3];
  for (int p = 0; p < 3; ++p) {
    pools[p] = POOLS[p] ? hvx_worker_pool_create(POOLS[p]) : NULL;
    CHECK(POOLS[p] == 0u || pools[p], "pool of %u workers", POOLS[p]);
  }

  static const uint32_t lengths[] = {1u, 63u, 64u, 65u, 512u, 1024u};
  for (size_t s = sizeof(SHAPES) / sizeof(SHAPES[0]); s-- > 0;) {
    set_shape(SHAPES[s]); /* LFM2.5 last: the cases below use it */
    for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
      check_length(lengths[i], 1024u, pools);
    }
  }
  check_length(1024u, 2048u, pools);
  check_append_chain(65u, pools[1]);
  check_identity(pools[2]);
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
