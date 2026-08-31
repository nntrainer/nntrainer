// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_matmul_dma.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for the MATMUL_W8A8 VTCM/DMA streaming path:
 *		checks it against the scalar reference and against the Task 6
 *		DDR direct-read path for bit-exact fp16 output.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <HAP_compute_res.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "htp_ops.h"
#include "ref_ops.h"
#include "sim_test_util.h"

static uint32_t align128(uint32_t n) { return (n + 127u) & ~127u; }

int test_matmul_dma(void) {
  const uint32_t m = 8, k = 1024, n = 3072;

  uint32_t off_x = 0;
  uint32_t off_w = align128(off_x + m * k * (uint32_t)sizeof(__fp16));
  uint32_t off_sw = align128(off_w + n * k);
  uint32_t off_y_ddr = align128(off_sw + n * (uint32_t)sizeof(float));
  uint32_t off_y_dma = align128(off_y_ddr + m * n * (uint32_t)sizeof(__fp16));
  uint32_t total = align128(off_y_dma + m * n * (uint32_t)sizeof(__fp16));

  uint8_t *act = memalign(128, total);
  __fp16 *x = (__fp16 *)(act + off_x);
  int8_t *w = (int8_t *)(act + off_w);
  float *sw = (float *)(act + off_sw);
  __fp16 *y_ddr = (__fp16 *)(act + off_y_ddr);
  __fp16 *y_dma = (__fp16 *)(act + off_y_dma);

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
  d.kind = NNTR_HTP_OP_MATMUL_W8A8;
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

  /* DDR direct-read path (Task 6). */
  d.out.offset = off_y_ddr;
  hvx_op_matmul_w8a8(&c, &d);

  /* VTCM/DMA streaming path (this task). */
  compute_res_attr_t rattr;
  HAP_compute_res_attr_init(&rattr);
  HAP_compute_res_attr_set_vtcm_param(&rattr, 4 * 1024 * 1024, 1);
  unsigned ctx_id = HAP_compute_res_acquire(&rattr, 10000 /*us*/);
  if (ctx_id == 0) {
    printf("SIM_TEST matmul_dma vtcm acquire fail\n");
    free(c.xq);
    free(c.xq_scale);
    wp_destroy(c.pool);
    free(act);
    return 1;
  }
  void *vtcm = HAP_compute_res_attr_get_vtcm_ptr(&rattr);
  if (!vtcm) {
    printf("SIM_TEST matmul_dma vtcm acquire fail\n");
    HAP_compute_res_release(ctx_id);
    free(c.xq);
    free(c.xq_scale);
    wp_destroy(c.pool);
    free(act);
    return 1;
  }
  c.vtcm = (uint8_t *)vtcm;
  c.vtcm_size = 4 * 1024 * 1024;
  d.out.offset = off_y_dma;
  hvx_op_matmul_w8a8(&c, &d);
  HAP_compute_res_release(ctx_id);

  int rc = 0;

  /* (a) DMA path vs scalar reference. */
  __fp16 *y_ref = malloc((size_t)m * n * sizeof(__fp16));
  ref_matmul_w8a8(x, w, sw, y_ref, m, k, n);
  float *ref_f = malloc((size_t)m * n * sizeof(float));
  float *got_f = malloc((size_t)m * n * sizeof(float));
  for (uint32_t i = 0; i < m * n; ++i) {
    ref_f[i] = (float)y_ref[i];
    got_f[i] = (float)y_dma[i];
  }
  if (cmp_f("matmul_dma_ref", ref_f, got_f, m * n, 2e-3f, 1e-3f))
    rc = 1;
  free(ref_f);
  free(got_f);
  free(y_ref);

  /* (b) DMA path vs DDR path: must be bit-identical. */
  size_t nbytes = (size_t)m * n * sizeof(__fp16);
  int diff = memcmp(y_ddr, y_dma, nbytes);
  if (diff) {
    const uint8_t *a = (const uint8_t *)y_ddr, *b = (const uint8_t *)y_dma;
    size_t i;
    for (i = 0; i < nbytes; ++i)
      if (a[i] != b[i])
        break;
    printf("SIM_TEST matmul_dma FAIL bit-mismatch at byte %u\n", (unsigned)i);
    rc = 1;
  }

  free(c.xq);
  free(c.xq_scale);
  wp_destroy(c.pool);
  free(act);

  if (rc)
    return 1;

  printf("SIM_TEST matmul_dma PASS\n");
  return 0;
}
