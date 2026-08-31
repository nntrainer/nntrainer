// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_matmul.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for the MATMUL_W8A8 and MATMUL_W8A16 kernels
 *		(DDR direct-read path)
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

static int run_case(uint32_t m, uint32_t k, uint32_t n, int a16) {
  uint32_t off_x = 0;
  uint32_t off_w = align128(off_x + m * k * (uint32_t)sizeof(__fp16));
  uint32_t off_sw = align128(off_w + n * k);
  uint32_t off_y = align128(off_sw + n * (uint32_t)sizeof(float));
  uint32_t total = align128(off_y + m * n * (uint32_t)sizeof(__fp16));

  uint8_t *act = memalign(128, total);
  __fp16 *x = (__fp16 *)(act + off_x);
  int8_t *w = (int8_t *)(act + off_w);
  float *sw = (float *)(act + off_sw);
  __fp16 *y = (__fp16 *)(act + off_y);

  for (uint32_t i = 0; i < m * k; ++i)
    x[i] = (__fp16)frand();
  for (uint32_t i = 0; i < n * k; ++i)
    w[i] = (int8_t)(frand() * 127.f);
  for (uint32_t j = 0; j < n; ++j)
    sw[j] = 0.001f + 0.019f * (frand() * 0.5f + 0.5f);

  struct htp_exec_ctx c;
  memset(&c, 0, sizeof(c));
  c.buf[NNTR_HTP_BUF_ACT] = act;
  c.buf_size[NNTR_HTP_BUF_ACT] = total;
  c.pool = wp_create(0);
  c.xq = memalign(128, (size_t)m * k);
  c.xq_scale = malloc((size_t)m * sizeof(float));

  struct nntr_htp_op_desc d;
  memset(&d, 0, sizeof(d));
  d.kind = a16 ? NNTR_HTP_OP_MATMUL_W8A16 : NNTR_HTP_OP_MATMUL_W8A8;
  d.m = m;
  d.k = k;
  d.n = n;
  d.in0.buf = NNTR_HTP_BUF_ACT;
  d.in0.offset = off_x;
  d.in1.buf = NNTR_HTP_BUF_ACT;
  d.in1.offset = off_w;
  d.in2.buf = NNTR_HTP_BUF_ACT;
  d.in2.offset = off_sw;
  d.out.buf = NNTR_HTP_BUF_ACT;
  d.out.offset = off_y;

  __fp16 *y_ref = malloc((size_t)m * n * sizeof(__fp16));
  if (a16) {
    hvx_op_matmul_w8a16(&c, &d);
    ref_matmul_w8a16(x, w, sw, y_ref, m, k, n);
  } else {
    hvx_op_matmul_w8a8(&c, &d);
    ref_matmul_w8a8(x, w, sw, y_ref, m, k, n);
  }

  float *ref_f = malloc((size_t)m * n * sizeof(float));
  float *got_f = malloc((size_t)m * n * sizeof(float));
  for (uint32_t i = 0; i < m * n; ++i) {
    ref_f[i] = (float)y_ref[i];
    got_f[i] = (float)y[i];
  }

  char tag[32];
  snprintf(tag, sizeof(tag), "matmul_%s_m%u", a16 ? "w8a16" : "w8a8",
           (unsigned)m);
  int rc = cmp_f(tag, ref_f, got_f, m * n, 2e-3f, 1e-3f);

  free(ref_f);
  free(got_f);
  free(y_ref);
  free(c.xq);
  free(c.xq_scale);
  wp_destroy(c.pool);
  free(act);
  return rc;
}

int test_matmul(void) {
  if (run_case(1, 1024, 256, 0))
    return 1;
  if (run_case(8, 1024, 256, 0))
    return 1;
  if (run_case(1, 3072, 256, 1))
    return 1;
  if (run_case(8, 3072, 256, 1))
    return 1;

  printf("SIM_TEST matmul PASS\n");
  return 0;
}
