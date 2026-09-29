// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_attn.c
 * @date	19 August 2026
 * @brief	Hexagon-sim test for the fused ATTN kernel: KV append plus
 *		causal SDPA with GQA, prefill then decode reusing the cache,
 *		and the m=1 long-context cases of the position-split decode
 *		(issue #58) at model shape on the default and a 5-worker pool
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <HAP_perf.h>

#include "htp_ops.h"
#include "ref_ops.h"
#include "sim_test_util.h"

#define LAYER 1u /* non-zero KV base offset on purpose (small shape) */

/** One test shape: the model dims the kernel reads from the op-list header
 * plus the layer the ATTN op addresses. */
struct attn_shape {
  uint32_t n_layers, n_heads, n_kv, hd, max_seq, layer;
};

static const struct attn_shape SHAPE_SMALL = {2u, 4u, 2u, 128u, 64u, LAYER};
/** qwen3-0.6b head shape, one layer, max_seq 17 lane blocks: 1088 keeps the
 * KV cache at 4.5 MB on the simulator while pos 1024 / 1087 exercise a
 * tail lane block of 1 and a full last block. */
static const struct attn_shape SHAPE_LONG = {1u, 16u, 8u, 128u, 1088u, 0u};

static uint32_t align128(uint32_t n) { return (n + 127u) & ~127u; }

static size_t kv_halves(const struct attn_shape *s) {
  return (size_t)s->n_layers * s->n_kv * s->max_seq * s->hd;
}

/** The kernel caches K transposed ([hd][max_seq] per layer/head), the
 * reference keeps rows; V is row-major on both sides. */
static int kv_equal(const struct attn_shape *s, const uint16_t *got,
                    const uint16_t *ref) {
  const size_t half = kv_halves(s);
  for (size_t lh = 0; lh < (size_t)s->n_layers * s->n_kv; ++lh)
    for (uint32_t p = 0; p < s->max_seq; ++p)
      for (uint32_t i = 0; i < s->hd; ++i)
        if (got[lh * s->max_seq * s->hd + (size_t)i * s->max_seq + p] !=
            ref[lh * s->max_seq * s->hd + (size_t)p * s->hd + i])
          return 0;
  return !memcmp(got + half, ref + half, half * sizeof(uint16_t));
}

/** Pre-fill every layer/head/position of both caches with the same random
 * fp16 (row-major for the reference, K transposed for the kernel) so a
 * decode step at pos > 0 attends to a populated context without a prefill
 * of that length. */
static void kv_prefill(const struct attn_shape *s, __fp16 *kv, __fp16 *kv_ref) {
  const size_t half = kv_halves(s);
  for (size_t lh = 0; lh < (size_t)s->n_layers * s->n_kv; ++lh) {
    __fp16 *kh = kv + lh * s->max_seq * s->hd;
    __fp16 *kr = kv_ref + lh * s->max_seq * s->hd;
    for (uint32_t p = 0; p < s->max_seq; ++p)
      for (uint32_t i = 0; i < s->hd; ++i) {
        const __fp16 kval = (__fp16)frand(), vval = (__fp16)frand();
        kr[(size_t)p * s->hd + i] = kval;
        kh[(size_t)i * s->max_seq + p] = kval;
        kr[half + (size_t)p * s->hd + i] = vval;
        kh[half + (size_t)p * s->hd + i] = vval;
      }
  }
}

static int run_step(const struct attn_shape *s, struct htp_exec_ctx *c,
                    __fp16 *kv_ref, uint32_t m, uint32_t pos, float scale,
                    const char *tag, uint64_t *pcycles) {
  const uint32_t n_heads = s->n_heads, n_kv = s->n_kv, hd = s->hd;
  uint32_t off_q = 0;
  uint32_t off_k =
    align128(off_q + m * n_heads * hd * (uint32_t)sizeof(__fp16));
  uint32_t off_v = align128(off_k + m * n_kv * hd * (uint32_t)sizeof(__fp16));
  uint32_t off_y = align128(off_v + m * n_kv * hd * (uint32_t)sizeof(__fp16));
  __fp16 *q = (__fp16 *)(c->buf[NNTR_HTP_BUF_ACT] + off_q);
  __fp16 *k = (__fp16 *)(c->buf[NNTR_HTP_BUF_ACT] + off_k);
  __fp16 *v = (__fp16 *)(c->buf[NNTR_HTP_BUF_ACT] + off_v);
  __fp16 *y = (__fp16 *)(c->buf[NNTR_HTP_BUF_ACT] + off_y);

  for (uint32_t i = 0; i < m * n_heads * hd; ++i)
    q[i] = (__fp16)frand();
  for (uint32_t i = 0; i < m * n_kv * hd; ++i) {
    k[i] = (__fp16)frand();
    v[i] = (__fp16)frand();
  }

  uint32_t param0;
  memcpy(&param0, &scale, sizeof(param0));

  struct nntr_htp_op_desc d;
  memset(&d, 0, sizeof(d));
  d.kind = NNTR_HTP_OP_ATTN;
  d.layer = s->layer;
  d.m = m;
  d.in0.buf = NNTR_HTP_BUF_ACT;
  d.in0.offset = off_q;
  d.in1.buf = NNTR_HTP_BUF_ACT;
  d.in1.offset = off_k;
  d.in2.buf = NNTR_HTP_BUF_ACT;
  d.in2.offset = off_v;
  d.out.buf = NNTR_HTP_BUF_ACT;
  d.out.offset = off_y;
  d.param0 = param0;

  c->pos = pos;
  const uint64_t t0 = HAP_perf_get_pcycles();
  hvx_op_attn(c, &d);
  if (pcycles)
    *pcycles = HAP_perf_get_pcycles() - t0;

  __fp16 *y_ref = malloc((size_t)m * n_heads * hd * sizeof(__fp16));
  ref_attn(q, k, v, kv_ref, y_ref, m, pos, s->layer, s->n_layers, n_heads, n_kv,
           hd, s->max_seq, scale);

  float *ref_f = malloc((size_t)m * n_heads * hd * sizeof(float));
  float *got_f = malloc((size_t)m * n_heads * hd * sizeof(float));
  for (uint32_t i = 0; i < m * n_heads * hd; ++i) {
    ref_f[i] = (float)y_ref[i];
    got_f[i] = (float)y[i];
  }
  int rc = cmp_f(tag, ref_f, got_f, m * n_heads * hd, 2e-2f, 5e-3f);

  /* Appends are pure fp16 copies on both sides: bit-exact compare. */
  if (!rc && !kv_equal(s, (const uint16_t *)c->buf[NNTR_HTP_BUF_KV],
                       (const uint16_t *)kv_ref)) {
    printf("SIM_TEST %s FAIL kv cache mismatch\n", tag);
    rc = 1;
  }

  free(ref_f);
  free(got_f);
  free(y_ref);
  return rc;
}

/** Buffers and context for one shape; `m_max` rows of activations. */
struct attn_env {
  uint8_t *act, *kv;
  __fp16 *kv_ref;
  struct nntr_htp_oplist_header cfg;
  struct htp_exec_ctx c;
};

static void env_init(struct attn_env *e, const struct attn_shape *s,
                     uint32_t m_max, int n_workers) {
  const size_t kv_bytes = 2u * kv_halves(s) * sizeof(__fp16);
  const uint32_t act_total = align128(m_max * (2u * s->n_heads + 2u * s->n_kv) *
                                        s->hd * (uint32_t)sizeof(__fp16) +
                                      3u * 128u);
  memset(e, 0, sizeof(*e));
  e->act = memalign(128, act_total);
  e->kv = memalign(128, kv_bytes);
  e->kv_ref = malloc(kv_bytes);
  memset(e->kv, 0, kv_bytes);
  memset(e->kv_ref, 0, kv_bytes);

  e->cfg.n_layers = s->n_layers;
  e->cfg.n_heads = s->n_heads;
  e->cfg.n_kv_heads = s->n_kv;
  e->cfg.head_dim = s->hd;
  e->cfg.max_seq = s->max_seq;

  e->c.buf[NNTR_HTP_BUF_ACT] = e->act;
  e->c.buf_size[NNTR_HTP_BUF_ACT] = act_total;
  e->c.buf[NNTR_HTP_BUF_KV] = e->kv;
  e->c.buf_size[NNTR_HTP_BUF_KV] = (uint32_t)kv_bytes;
  e->c.cfg = &e->cfg;
  e->c.pool = wp_create(n_workers);
  e->c.attn_scratch =
    memalign(128, htp_attn_scratch_bytes(wp_size(e->c.pool), &e->cfg));
}

static void env_free(struct attn_env *e) {
  free(e->c.attn_scratch);
  wp_destroy(e->c.pool);
  free(e->kv_ref);
  free(e->kv);
  free(e->act);
}

/** The three m=1 long-context cases (issue #58 plan §2) on one pool: pos
 * 1024 (L = 1025, tail lane block of one position), 1087 (L = 1088, no
 * tail) and 100 (L = 101, one full block plus a 37-lane block). The
 * kernel's split (nb blocks per kv head) follows the worker count, so the
 * same cases on a 5-worker pool exercise the remainder rotation. */
static int run_long(int n_workers, uint64_t *pc1024) {
  const struct attn_shape *s = &SHAPE_LONG;
  const float scale = 1.0f / sqrtf((float)s->hd);
  static const uint32_t POS[3] = {1024u, 1087u, 100u};
  struct attn_env e;
  char tag[64], suffix[16] = "";
  int rc = 0;

  env_init(&e, s, 1u, n_workers);
  /** wp_create clamps to the HVX unit count, so the tag carries the pool
   * the cases actually ran on (v79: _w5; a 4-unit part would say _w4). */
  if (n_workers > 0)
    snprintf(suffix, sizeof(suffix), "_w%d", wp_size(e.c.pool));
  kv_prefill(s, (__fp16 *)e.kv, e.kv_ref);
  for (int i = 0; i < 3 && !rc; ++i) {
    snprintf(tag, sizeof(tag), "attn_decode_p%u%s", (unsigned)POS[i], suffix);
    rc = run_step(s, &e.c, e.kv_ref, 1u, POS[i], scale, tag,
                  i == 0 ? pc1024 : NULL);
  }
  env_free(&e);
  return rc;
}

int test_attn(void) {
  const struct attn_shape *s = &SHAPE_SMALL;
  const float scale = 1.0f / sqrtf((float)s->hd);
  struct attn_env e;
  uint64_t pc1024 = 0;

  env_init(&e, s, 8u, 0);
  int rc = run_step(s, &e.c, e.kv_ref, 8, 0, scale, "attn_prefill", NULL);
  if (!rc)
    rc = run_step(s, &e.c, e.kv_ref, 1, 8, scale, "attn_decode", NULL);
  env_free(&e);

  if (!rc)
    rc = run_long(0, &pc1024);
  if (!rc)
    rc = run_long(5, NULL);

  if (rc)
    return 1;

  /** Relative signal only (the simulator charges no DDR latency): the
   * per-op pcycles of the pos-1024 decode on the default pool. */
  printf("SIM_TEST attn_decode_p1024 STAT pcycles=%llu\n",
         (unsigned long long)pc1024);
  printf("SIM_TEST attn PASS\n");
  return 0;
}
