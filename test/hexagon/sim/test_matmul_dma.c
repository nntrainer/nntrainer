// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_matmul_dma.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for the MATMUL_W8A8 VTCM/DMA streaming path:
 *		checks it against the scalar reference and against the DDR
 *		direct-read path for bit-exact fp16 output at three VTCM sizes
 *		(4 MB, 256 KB, 64 KB fallback), and since issue #25 the
 *		graph-lifetime queue with the cross-op weight prefetch: two
 *		consecutive ops X (k=1024, n=3072) and Y (k=2048, n=1024) run
 *		with ctx.next_mm set, so X kicks Y's chunk 0 at its tail and Y
 *		must find it (hit counted per worker); a prefetch that the next
 *		op does not match (X again) or that meets the DDR fallback
 *		(64 KB) is drained, never used; every run stays bit-identical
 *		to the DDR path.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <HAP_compute_res.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "htp_graph.h"
#include "htp_ops.h"
#include "ref_ops.h"
#include "sim_test_util.h"

static uint32_t align128(uint32_t n) { return (n + 127u) & ~127u; }

/** One W8A8 op over ACT-buffer offsets: x fp16[m][k], w tiled32 [n][k],
 * sw fp32[n], y fp16[m][n] (a DDR-path copy and a DMA-path copy). */
struct mm_case {
  uint32_t m, k, n;
  uint32_t off_x, off_w, off_sw, off_y_ddr, off_y_dma;
  struct nntr_htp_op_desc d;
};

static uint32_t mm_case_layout(struct mm_case *t, uint32_t m, uint32_t k,
                               uint32_t n, uint32_t off) {
  t->m = m;
  t->k = k;
  t->n = n;
  t->off_x = off;
  t->off_w = align128(t->off_x + m * k * (uint32_t)sizeof(__fp16));
  t->off_sw = align128(t->off_w + n * k);
  t->off_y_ddr = align128(t->off_sw + n * (uint32_t)sizeof(float));
  t->off_y_dma = align128(t->off_y_ddr + m * n * (uint32_t)sizeof(__fp16));
  return align128(t->off_y_dma + m * n * (uint32_t)sizeof(__fp16));
}

static void mm_case_fill(struct mm_case *t, uint8_t *act) {
  __fp16 *x = (__fp16 *)(act + t->off_x);
  int8_t *w = (int8_t *)(act + t->off_w);
  float *sw = (float *)(act + t->off_sw);
  for (uint32_t i = 0; i < t->m * t->k; ++i)
    x[i] = (__fp16)frand();
  int8_t *wrm = malloc((size_t)t->n * t->k);
  for (uint32_t i = 0; i < t->n * t->k; ++i)
    wrm[i] = (int8_t)(frand() * 127.f);
  nntr_htp_repack_tiled32((uint8_t *)w, (const uint8_t *)wrm, t->n, t->k);
  free(wrm);
  for (uint32_t j = 0; j < t->n; ++j)
    sw[j] = 0.001f + 0.019f * (frand() * 0.5f + 0.5f);

  memset(&t->d, 0, sizeof(t->d));
  t->d.kind = NNTR_HTP_OP_MATMUL_W8A8;
  t->d.m = t->m;
  t->d.k = t->k;
  t->d.n = t->n;
  t->d.in0.buf = NNTR_HTP_BUF_ACT;
  t->d.in0.offset = t->off_x;
  t->d.in1.buf = NNTR_HTP_BUF_ACT;
  t->d.in1.offset = t->off_w;
  t->d.in2.buf = NNTR_HTP_BUF_ACT;
  t->d.in2.offset = t->off_sw;
  t->d.out.buf = NNTR_HTP_BUF_ACT;
  t->d.out.offset = t->off_y_dma;
}

/** Run one op on the DMA path and compare with its DDR-path output. */
static int mm_case_run(struct htp_exec_ctx *c, struct mm_case *t,
                       const struct nntr_htp_op_desc *next, const char *tag) {
  uint8_t *act = c->buf[NNTR_HTP_BUF_ACT];
  const size_t nbytes = (size_t)t->m * t->n * sizeof(__fp16);
  memset(act + t->off_y_dma, 0, nbytes);
  c->next_mm = next;
  hvx_op_matmul_w8a8(c, &t->d);
  c->next_mm = NULL;
  if (memcmp(act + t->off_y_ddr, act + t->off_y_dma, nbytes)) {
    printf("SIM_TEST matmul_dma FAIL %s k=%u differs from the DDR path\n", tag,
           (unsigned)t->k);
    return 1;
  }
  return 0;
}

static int pf_all_clear(const struct htp_exec_ctx *c, int nw, const char *tag) {
  for (int w = 0; w < nw; ++w)
    if (c->pf[w].desc) {
      printf("SIM_TEST matmul_dma FAIL %s worker %d still has a prefetch\n",
             tag, w);
      return 1;
    }
  return 0;
}

int test_matmul_dma(void) {
  const uint32_t m = 8;
  struct mm_case X, Y;
  uint32_t total = mm_case_layout(&X, m, 1024u, 3072u, 0u);
  total = mm_case_layout(&Y, m, 2048u, 1024u, total);

  uint8_t *act = memalign(128, total);
  mm_case_fill(&X, act);
  mm_case_fill(&Y, act);

  struct htp_exec_ctx c;
  memset(&c, 0, sizeof(c));
  c.buf[NNTR_HTP_BUF_ACT] = act;
  c.buf_size[NNTR_HTP_BUF_ACT] = total;
  c.pool = wp_create(0);
  c.xq = memalign(128, (size_t)m * Y.k);
  c.xq_scale = malloc((size_t)m * sizeof(float));
  const int nw = wp_size(c.pool);

  /* DDR direct-read path (c.vtcm == NULL) for both ops. */
  X.d.out.offset = X.off_y_ddr;
  hvx_op_matmul_w8a8(&c, &X.d);
  X.d.out.offset = X.off_y_dma;
  Y.d.out.offset = Y.off_y_ddr;
  hvx_op_matmul_w8a8(&c, &Y.d);
  Y.d.out.offset = Y.off_y_dma;

  compute_res_attr_t rattr;
  HAP_compute_res_attr_init(&rattr);
  HAP_compute_res_attr_set_vtcm_param(&rattr, 4 * 1024 * 1024, 1);
  unsigned ctx_id = HAP_compute_res_acquire(&rattr, 10000 /*us*/);
  void *vtcm = ctx_id ? HAP_compute_res_attr_get_vtcm_ptr(&rattr) : NULL;
  if (!vtcm) {
    printf("SIM_TEST matmul_dma vtcm acquire fail\n");
    if (ctx_id)
      HAP_compute_res_release(ctx_id);
    free(c.xq);
    free(c.xq_scale);
    wp_destroy(c.pool);
    free(act);
    return 1;
  }
  c.vtcm = (uint8_t *)vtcm;
  c.vtcm_size = 4u << 20;
  /* The same per-worker graph-lifetime queues htp_graph_init_ex allocates. */
  if (htp_graph_dma_init(&c, nw)) {
    printf("SIM_TEST matmul_dma dma init fail\n");
    HAP_compute_res_release(ctx_id);
    free(c.xq);
    free(c.xq_scale);
    wp_destroy(c.pool);
    free(act);
    return 1;
  }

  __fp16 *y_ref = malloc((size_t)m * X.n * sizeof(__fp16));
  ref_matmul_w8a8((const __fp16 *)(act + X.off_x),
                  (const int8_t *)(act + X.off_w),
                  (const float *)(act + X.off_sw), y_ref, m, X.k, X.n);
  float *ref_f = malloc((size_t)m * X.n * sizeof(float));
  float *got_f = malloc((size_t)m * X.n * sizeof(float));
  for (uint32_t i = 0; i < m * X.n; ++i)
    ref_f[i] = (float)y_ref[i];

  /** (d) VTCM/DMA streaming path at three slab sizes: 4 MB (what htp_graph
   * acquires), 256 KB (with 4 workers a half-slab is exactly one 32-row
   * tile at k=1024, so X pipelines 24 single-tile chunks while Y at k=2048
   * takes the DDR fallback; with 6 workers both fall back) and 64 KB (both
   * fall back). Each X → Y pair runs with the prefetch on; X must match the
   * scalar reference and both must be bit-identical to their DDR run. At
   * 4 MB every worker must have found its Y chunk 0 in flight: (a). */
  const uint32_t sizes[3] = {4u << 20, 256u << 10, 64u << 10};
  int rc = 0;
  for (uint32_t s = 0; s < 3u && !rc; ++s) {
    char tag[32];
    c.vtcm_size = sizes[s];
    for (int w = 0; w < nw; ++w)
      c.pf[w].hits = 0u;
    rc = mm_case_run(&c, &X, &Y.d, "sweep X");
    if (!rc)
      rc = mm_case_run(&c, &Y, NULL, "sweep Y");
    for (uint32_t i = 0; i < m * X.n; ++i)
      got_f[i] = (float)((const __fp16 *)(act + X.off_y_dma))[i];
    snprintf(tag, sizeof(tag), "matmul_dma_ref_%uk",
             (unsigned)(sizes[s] >> 10));
    if (cmp_f(tag, ref_f, got_f, m * X.n, 2e-3f, 1e-3f))
      rc = 1;
    if (!rc && s == 0u) {
      uint32_t hits = 0;
      for (int w = 0; w < nw; ++w)
        hits += c.pf[w].hits;
      printf("SIM_TEST matmul_dma prefetch hits=%u workers=%d\n",
             (unsigned)hits, nw);
#ifndef HTP_MM_NO_PREFETCH
      for (int w = 0; w < nw; ++w)
        if (c.pf[w].hits != 1u) {
          printf("SIM_TEST matmul_dma FAIL worker %d prefetch hits %u != 1\n",
                 w, (unsigned)c.pf[w].hits);
          rc = 1;
        }
#endif
    }
    if (!rc)
      rc = pf_all_clear(&c, nw, "sweep");
  }

  /** (b) Mismatch: X prefetches Y's chunk 0, then X runs again instead of
   * Y - the record is for another op, so it is drained and X kicks its
   * own chunk 0; no hit, output still bit-identical. */
  if (!rc) {
    c.vtcm_size = sizes[0];
    for (int w = 0; w < nw; ++w)
      c.pf[w].hits = 0u;
    rc = mm_case_run(&c, &X, &Y.d, "mismatch X1");
    if (!rc)
      rc = mm_case_run(&c, &X, NULL, "mismatch X2");
    for (int w = 0; w < nw && !rc; ++w)
      if (c.pf[w].hits != 0u) {
        printf("SIM_TEST matmul_dma FAIL mismatch counted as a hit\n");
        rc = 1;
      }
    if (!rc)
      rc = pf_all_clear(&c, nw, "mismatch");
  }

  /** (c) DDR fallback with a prefetch pending: X at 4 MB prefetches Y, then
   * the slab shrinks to 64 KB (a unit-test artefact standing in for any
   * geometry change) so Y takes the DDR path, which must drain the pending
   * descriptor and leave no record behind. */
  if (!rc) {
    c.vtcm_size = sizes[0];
    rc = mm_case_run(&c, &X, &Y.d, "fallback X");
    c.vtcm_size = sizes[2];
    if (!rc)
      rc = mm_case_run(&c, &Y, NULL, "fallback Y");
    if (!rc)
      rc = pf_all_clear(&c, nw, "fallback");
  }

  htp_graph_dma_flush(&c);
  HAP_compute_res_release(ctx_id);
  htp_graph_dma_destroy(&c);

  free(ref_f);
  free(got_f);
  free(y_ref);
  free(c.xq);
  free(c.xq_scale);
  wp_destroy(c.pool);
  free(act);
  if (rc)
    return 1;

  printf("SIM_TEST matmul_dma PASS\n");
  return 0;
}
