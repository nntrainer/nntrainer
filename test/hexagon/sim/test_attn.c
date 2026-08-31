// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_attn.c
 * @date	19 August 2026
 * @brief	Hexagon-sim test for the fused ATTN kernel: KV append plus
 *		causal SDPA with GQA, prefill then decode reusing the cache
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "htp_ops.h"
#include "ref_ops.h"
#include "sim_test_util.h"

#define N_LAYERS 2u
#define N_HEADS 4u
#define N_KV_HEADS 2u
#define HEAD_DIM 128u
#define MAX_SEQ 64u
#define LAYER 1u /* non-zero KV base offset on purpose */

static uint32_t align128(uint32_t n) { return (n + 127u) & ~127u; }

/* The kernel caches K transposed ([hd][max_seq] per layer/head), the
 * reference keeps rows; V is row-major on both sides. */
static int kv_equal(const uint16_t *got, const uint16_t *ref) {
  const size_t half = (size_t)N_LAYERS * N_KV_HEADS * MAX_SEQ * HEAD_DIM;
  for (size_t lh = 0; lh < N_LAYERS * N_KV_HEADS; ++lh)
    for (uint32_t p = 0; p < MAX_SEQ; ++p)
      for (uint32_t i = 0; i < HEAD_DIM; ++i)
        if (got[lh * MAX_SEQ * HEAD_DIM + (size_t)i * MAX_SEQ + p] !=
            ref[lh * MAX_SEQ * HEAD_DIM + (size_t)p * HEAD_DIM + i])
          return 0;
  return !memcmp(got + half, ref + half, half * sizeof(uint16_t));
}

static int run_step(struct htp_exec_ctx *c, __fp16 *kv_ref, uint32_t kv_halves,
                    uint32_t m, uint32_t pos, float scale, const char *tag) {
  uint32_t off_q = 0;
  uint32_t off_k =
    align128(off_q + m * N_HEADS * HEAD_DIM * (uint32_t)sizeof(__fp16));
  uint32_t off_v =
    align128(off_k + m * N_KV_HEADS * HEAD_DIM * (uint32_t)sizeof(__fp16));
  uint32_t off_y =
    align128(off_v + m * N_KV_HEADS * HEAD_DIM * (uint32_t)sizeof(__fp16));
  __fp16 *q = (__fp16 *)(c->buf[NNTR_HTP_BUF_ACT] + off_q);
  __fp16 *k = (__fp16 *)(c->buf[NNTR_HTP_BUF_ACT] + off_k);
  __fp16 *v = (__fp16 *)(c->buf[NNTR_HTP_BUF_ACT] + off_v);
  __fp16 *y = (__fp16 *)(c->buf[NNTR_HTP_BUF_ACT] + off_y);

  for (uint32_t i = 0; i < m * N_HEADS * HEAD_DIM; ++i)
    q[i] = (__fp16)frand();
  for (uint32_t i = 0; i < m * N_KV_HEADS * HEAD_DIM; ++i) {
    k[i] = (__fp16)frand();
    v[i] = (__fp16)frand();
  }

  uint32_t param0;
  memcpy(&param0, &scale, sizeof(param0));

  struct nntr_htp_op_desc d;
  memset(&d, 0, sizeof(d));
  d.kind = NNTR_HTP_OP_ATTN;
  d.layer = LAYER;
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
  hvx_op_attn(c, &d);

  __fp16 *y_ref = malloc((size_t)m * N_HEADS * HEAD_DIM * sizeof(__fp16));
  ref_attn(q, k, v, kv_ref, y_ref, m, pos, LAYER, N_LAYERS, N_HEADS,
           N_KV_HEADS, HEAD_DIM, MAX_SEQ, scale);

  float *ref_f = malloc((size_t)m * N_HEADS * HEAD_DIM * sizeof(float));
  float *got_f = malloc((size_t)m * N_HEADS * HEAD_DIM * sizeof(float));
  for (uint32_t i = 0; i < m * N_HEADS * HEAD_DIM; ++i) {
    ref_f[i] = (float)y_ref[i];
    got_f[i] = (float)y[i];
  }
  int rc = cmp_f(tag, ref_f, got_f, m * N_HEADS * HEAD_DIM, 2e-2f, 5e-3f);

  /* Appends are pure fp16 copies on both sides: bit-exact compare. */
  if (!rc && !kv_equal((const uint16_t *)c->buf[NNTR_HTP_BUF_KV],
                       (const uint16_t *)kv_ref)) {
    printf("SIM_TEST %s FAIL kv cache mismatch\n", tag);
    rc = 1;
  }

  free(ref_f);
  free(got_f);
  free(y_ref);
  return rc;
}

int test_attn(void) {
  const uint32_t kv_halves = 2u * N_LAYERS * N_KV_HEADS * MAX_SEQ * HEAD_DIM;
  const float scale = 1.0f / sqrtf((float)HEAD_DIM);
  const uint32_t act_total = align128(
    8u * (2u * N_HEADS + 2u * N_KV_HEADS) * HEAD_DIM *
      (uint32_t)sizeof(__fp16) +
    3u * 128u);

  uint8_t *act = memalign(128, act_total);
  uint8_t *kv = memalign(128, kv_halves * sizeof(__fp16));
  __fp16 *kv_ref = malloc(kv_halves * sizeof(__fp16));
  memset(kv, 0, kv_halves * sizeof(__fp16));
  memset(kv_ref, 0, kv_halves * sizeof(__fp16));

  struct nntr_htp_oplist_header cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.n_layers = N_LAYERS;
  cfg.n_heads = N_HEADS;
  cfg.n_kv_heads = N_KV_HEADS;
  cfg.head_dim = HEAD_DIM;
  cfg.max_seq = MAX_SEQ;

  struct htp_exec_ctx c;
  memset(&c, 0, sizeof(c));
  c.buf[NNTR_HTP_BUF_ACT] = act;
  c.buf_size[NNTR_HTP_BUF_ACT] = act_total;
  c.buf[NNTR_HTP_BUF_KV] = kv;
  c.buf_size[NNTR_HTP_BUF_KV] = kv_halves * sizeof(__fp16);
  c.cfg = &cfg;
  c.pool = wp_create(0);
  /* [n_workers][max_seq] fp32 scores, +128B pad: hvx_exp_f32's tail path
   * reads one whole unaligned vector starting at the last elements. */
  c.attn_scratch =
    memalign(128, (size_t)wp_size(c.pool) * MAX_SEQ * sizeof(float) + 128u);

  int rc = run_step(&c, kv_ref, kv_halves, 8, 0, scale, "attn_prefill");
  if (!rc)
    rc = run_step(&c, kv_ref, kv_halves, 1, 8, scale, "attn_decode");

  free(c.attn_scratch);
  wp_destroy(c.pool);
  free(kv_ref);
  free(kv);
  free(act);

  if (rc)
    return 1;

  printf("SIM_TEST attn PASS\n");
  return 0;
}
