// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_embed.c
 * @date	19 August 2026
 * @brief	Hexagon-sim test for the EMBED gather-dequant kernel
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "htp_ops.h"
#include "ref_ops.h"
#include "sim_test_util.h"

static uint32_t align128(uint32_t n) { return (n + 127u) & ~127u; }

int test_embed(void) {
  const uint32_t vocab = 512, k = 256, m = 8;

  /* WEIGHTS: int8 table then fp32 scales. */
  uint32_t off_w = 0;
  uint32_t off_s = align128(off_w + vocab * k);
  uint32_t w_total = align128(off_s + vocab * (uint32_t)sizeof(float));
  uint8_t *wbuf = memalign(128, w_total);
  int8_t *w = (int8_t *)(wbuf + off_w);
  float *scale = (float *)(wbuf + off_s);

  for (uint32_t i = 0; i < vocab * k; ++i)
    w[i] = (int8_t)(frand() * 127.0f);
  for (uint32_t i = 0; i < vocab; ++i)
    scale[i] = 0.001f + 0.02f * (frand() * 0.5f + 0.5f);

  /* TOKENS: boundary rows plus PRNG ids. */
  int32_t *tokens = memalign(128, align128(m * (uint32_t)sizeof(int32_t)));
  tokens[0] = 0;
  tokens[1] = 1;
  tokens[2] = 255;
  tokens[3] = 511;
  for (uint32_t t = 4; t < m; ++t)
    tokens[t] = (int32_t)((frand() * 0.5f + 0.5f) * (float)(vocab - 1));

  /* ACT: fp16 output. */
  const uint32_t count = m * k;
  uint32_t y_total = align128(count * (uint32_t)sizeof(__fp16));
  uint8_t *act = memalign(128, y_total);
  __fp16 *y = (__fp16 *)act;

  struct nntr_htp_oplist_header cfg;
  memset(&cfg, 0, sizeof(cfg));

  struct htp_exec_ctx c;
  memset(&c, 0, sizeof(c));
  c.buf[NNTR_HTP_BUF_WEIGHTS] = wbuf;
  c.buf_size[NNTR_HTP_BUF_WEIGHTS] = w_total;
  c.buf[NNTR_HTP_BUF_TOKENS] = (uint8_t *)tokens;
  c.buf_size[NNTR_HTP_BUF_TOKENS] = align128(m * (uint32_t)sizeof(int32_t));
  c.buf[NNTR_HTP_BUF_ACT] = act;
  c.buf_size[NNTR_HTP_BUF_ACT] = y_total;
  c.cfg = &cfg;
  c.n_tokens = m;
  c.pool = wp_create(0);

  struct nntr_htp_op_desc d;
  memset(&d, 0, sizeof(d));
  d.kind = NNTR_HTP_OP_EMBED;
  d.m = 0; /* runtime n_tokens */
  d.k = k;
  d.in0.buf = NNTR_HTP_BUF_TOKENS;
  d.in0.offset = 0;
  d.in1.buf = NNTR_HTP_BUF_WEIGHTS;
  d.in1.offset = off_w;
  d.in2.buf = NNTR_HTP_BUF_WEIGHTS;
  d.in2.offset = off_s;
  d.out.buf = NNTR_HTP_BUF_ACT;
  d.out.offset = 0;

  hvx_op_embed(&c, &d);

  __fp16 *y_ref = malloc((size_t)count * sizeof(__fp16));
  float *ref_f = malloc((size_t)count * sizeof(float));
  float *got_f = malloc((size_t)count * sizeof(float));

  ref_embed(tokens, w, scale, y_ref, m, k);
  for (uint32_t i = 0; i < count; ++i) {
    ref_f[i] = (float)y_ref[i];
    got_f[i] = (float)y[i];
  }
  int rc = cmp_f("embed", ref_f, got_f, count, 1e-3f, 1e-4f);

  free(ref_f);
  free(got_f);
  free(y_ref);
  wp_destroy(c.pool);
  free(act);
  free(tokens);
  free(wbuf);
  if (rc)
    return 1;

  printf("SIM_TEST embed PASS\n");
  return 0;
}
