// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_eltwise.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for the ADD and SILU_MUL kernels
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

int test_eltwise(void) {
  const uint32_t m = 4, n = 3072;
  const uint32_t count = m * n;

  uint32_t off_a = 0;
  uint32_t off_b = align128(off_a + count * (uint32_t)sizeof(__fp16));
  uint32_t off_y = align128(off_b + count * (uint32_t)sizeof(__fp16));
  uint32_t total = align128(off_y + count * (uint32_t)sizeof(__fp16));

  uint8_t *act = memalign(128, total);
  __fp16 *a = (__fp16 *)(act + off_a);
  __fp16 *b = (__fp16 *)(act + off_b);
  __fp16 *y = (__fp16 *)(act + off_y);

  for (uint32_t i = 0; i < count; ++i) {
    a[i] = (__fp16)(frand() * 8.0f);
    b[i] = (__fp16)(frand() * 8.0f);
  }

  struct nntr_htp_oplist_header cfg;
  memset(&cfg, 0, sizeof(cfg));

  struct htp_exec_ctx c;
  memset(&c, 0, sizeof(c));
  c.buf[NNTR_HTP_BUF_ACT] = act;
  c.buf_size[NNTR_HTP_BUF_ACT] = total;
  c.cfg = &cfg;
  c.pool = wp_create(0);

  struct nntr_htp_op_desc d_add;
  memset(&d_add, 0, sizeof(d_add));
  d_add.kind = NNTR_HTP_OP_ADD;
  d_add.m = m;
  d_add.n = n;
  d_add.in0.buf = NNTR_HTP_BUF_ACT;
  d_add.in0.offset = off_a;
  d_add.in1.buf = NNTR_HTP_BUF_ACT;
  d_add.in1.offset = off_b;
  d_add.out.buf = NNTR_HTP_BUF_ACT;
  d_add.out.offset = off_y;

  hvx_op_add(&c, &d_add);

  __fp16 *y_ref = malloc((size_t)count * sizeof(__fp16));
  float *ref_f = malloc((size_t)count * sizeof(float));
  float *got_f = malloc((size_t)count * sizeof(float));

  ref_add(a, b, y_ref, count);
  for (uint32_t i = 0; i < count; ++i) {
    ref_f[i] = (float)y_ref[i];
    got_f[i] = (float)y[i];
  }
  int rc = cmp_f("add", ref_f, got_f, count, 1e-3f, 1e-4f);

  struct nntr_htp_op_desc d_silu;
  memset(&d_silu, 0, sizeof(d_silu));
  d_silu.kind = NNTR_HTP_OP_SILU_MUL;
  d_silu.m = m;
  d_silu.n = n;
  d_silu.in0.buf = NNTR_HTP_BUF_ACT;
  d_silu.in0.offset = off_a;
  d_silu.in1.buf = NNTR_HTP_BUF_ACT;
  d_silu.in1.offset = off_b;
  d_silu.out.buf = NNTR_HTP_BUF_ACT;
  d_silu.out.offset = off_y;

  if (!rc) {
    hvx_op_silu_mul(&c, &d_silu);

    ref_silu_mul(a, b, y_ref, count);
    for (uint32_t i = 0; i < count; ++i) {
      ref_f[i] = (float)y_ref[i];
      got_f[i] = (float)y[i];
    }
    rc = cmp_f("silu_mul", ref_f, got_f, count, 2e-2f, 5e-3f);
  }

  free(ref_f);
  free(got_f);
  free(y_ref);
  wp_destroy(c.pool);
  free(act);
  if (rc)
    return 1;

  printf("SIM_TEST eltwise PASS\n");
  return 0;
}
