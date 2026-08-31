// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_logits.c
 * @date	19 August 2026
 * @brief	Hexagon-sim test for the MATMUL_LOGITS kernel (last-token fp32 logits)
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

int test_logits(void) {
  const uint32_t M = 4, K = 1024, N = 2048;

  uint32_t off_x = 0;
  uint32_t off_w = align128(off_x + M * K * (uint32_t)sizeof(__fp16));
  uint32_t off_sw = align128(off_w + N * K);
  uint32_t total = align128(off_sw + N * (uint32_t)sizeof(float));

  uint8_t *act = memalign(128, total);
  __fp16 *x = (__fp16 *)(act + off_x);
  int8_t *w = (int8_t *)(act + off_w);
  float *sw = (float *)(act + off_sw);
  float *logits = memalign(128, N * sizeof(float));

  /* Garbage (extreme, non-NaN) in the first M-1 rows: the kernel must only
   * read the last token row, so these would blow up the result if touched. */
  for (uint32_t t = 0; t + 1 < M; ++t)
    for (uint32_t i = 0; i < K; ++i)
      x[(size_t)t * K + i] = (__fp16)((i & 1) ? 30000.f : -30000.f);
  for (uint32_t i = 0; i < K; ++i)
    x[(size_t)(M - 1) * K + i] = (__fp16)frand();
  for (uint32_t i = 0; i < N * K; ++i)
    w[i] = (int8_t)(frand() * 127.f);
  for (uint32_t j = 0; j < N; ++j)
    sw[j] = 0.001f + 0.019f * (frand() * 0.5f + 0.5f);

  struct htp_exec_ctx c;
  memset(&c, 0, sizeof(c));
  c.buf[NNTR_HTP_BUF_ACT] = act;
  c.buf_size[NNTR_HTP_BUF_ACT] = total;
  c.buf[NNTR_HTP_BUF_LOGITS] = (uint8_t *)logits;
  c.buf_size[NNTR_HTP_BUF_LOGITS] = N * (uint32_t)sizeof(float);
  c.n_tokens = M;
  c.pool = wp_create(0);
  c.xq = memalign(128, (size_t)K);
  c.xq_scale = malloc(sizeof(float));

  struct nntr_htp_op_desc d;
  memset(&d, 0, sizeof(d));
  d.kind = NNTR_HTP_OP_MATMUL_LOGITS;
  d.m = 1;
  d.k = K;
  d.n = N;
  d.in0.buf = NNTR_HTP_BUF_ACT;
  d.in0.offset = off_x;
  d.in1.buf = NNTR_HTP_BUF_ACT;
  d.in1.offset = off_w;
  d.in2.buf = NNTR_HTP_BUF_ACT;
  d.in2.offset = off_sw;
  d.out.buf = NNTR_HTP_BUF_LOGITS;
  d.out.offset = 0;

  hvx_op_matmul_logits(&c, &d);

  float *ref = malloc((size_t)N * sizeof(float));
  ref_matmul_logits(x + (size_t)(M - 1) * K, w, sw, ref, K, N);

  int rc = cmp_f("logits", ref, logits, N, 5e-3f, 1e-2f);

  free(ref);
  free(c.xq);
  free(c.xq_scale);
  wp_destroy(c.pool);
  free(logits);
  free(act);
  if (rc)
    return 1;

  printf("SIM_TEST logits PASS\n");
  return 0;
}
