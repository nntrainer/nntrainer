// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_rmsnorm.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for the RMSNORM kernel, whole-row and
 *		PER_HEAD (QK-Norm) modes
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

static int run_case(uint32_t m, uint32_t n, uint32_t gamma_len, uint32_t flags,
                    float eps, const char *tag) {
  uint32_t off_x = 0;
  uint32_t off_g = align128(off_x + m * n * (uint32_t)sizeof(__fp16));
  uint32_t off_y = align128(off_g + gamma_len * (uint32_t)sizeof(__fp16));
  uint32_t total = align128(off_y + m * n * (uint32_t)sizeof(__fp16));

  uint8_t *act = memalign(128, total);
  __fp16 *x = (__fp16 *)(act + off_x);
  __fp16 *gamma = (__fp16 *)(act + off_g);
  __fp16 *y = (__fp16 *)(act + off_y);

  for (uint32_t i = 0; i < m * n; ++i)
    x[i] = (__fp16)frand();
  for (uint32_t i = 0; i < gamma_len; ++i)
    gamma[i] = (__fp16)(1.0f + 0.5f * frand());

  uint32_t param0;
  memcpy(&param0, &eps, sizeof(param0));

  struct nntr_htp_oplist_header cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.head_dim = gamma_len;

  struct htp_exec_ctx c;
  memset(&c, 0, sizeof(c));
  c.buf[NNTR_HTP_BUF_ACT] = act;
  c.buf_size[NNTR_HTP_BUF_ACT] = total;
  c.cfg = &cfg;
  c.pool = wp_create(0);

  struct nntr_htp_op_desc d;
  memset(&d, 0, sizeof(d));
  d.kind = NNTR_HTP_OP_RMSNORM;
  d.flags = flags;
  d.m = m;
  d.n = n;
  d.in0.buf = NNTR_HTP_BUF_ACT;
  d.in0.offset = off_x;
  d.in1.buf = NNTR_HTP_BUF_ACT;
  d.in1.offset = off_g;
  d.out.buf = NNTR_HTP_BUF_ACT;
  d.out.offset = off_y;
  d.param0 = param0;

  hvx_op_rmsnorm(&c, &d);

  __fp16 *y_ref = malloc((size_t)m * n * sizeof(__fp16));
  ref_rmsnorm(x, gamma, y_ref, m, n, gamma_len, eps);

  float *ref_f = malloc((size_t)m * n * sizeof(float));
  float *got_f = malloc((size_t)m * n * sizeof(float));
  for (uint32_t i = 0; i < m * n; ++i) {
    ref_f[i] = (float)y_ref[i];
    got_f[i] = (float)y[i];
  }

  int rc = cmp_f(tag, ref_f, got_f, m * n, 5e-3f, 2e-3f);

  free(ref_f);
  free(got_f);
  free(y_ref);
  wp_destroy(c.pool);
  free(act);
  return rc;
}

int test_rmsnorm(void) {
  if (run_case(4, 1024, 1024, 0, 1e-6f, "rmsnorm_general"))
    return 1;
  if (run_case(4, 512, 128, NNTR_HTP_FLAG_PER_HEAD, 1e-6f, "rmsnorm_per_head"))
    return 1;

  printf("SIM_TEST rmsnorm PASS\n");
  return 0;
}
