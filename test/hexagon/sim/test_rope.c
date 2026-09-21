// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_rope.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for the ROPE kernel: in-place q/k rotation
 *		against the precomputed cos/sin table
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

int test_rope(void) {
  const uint32_t n_heads = 4, n_kv_heads = 2, max_seq = 64;
  const uint32_t m = 8, pos = 5;

  uint32_t off_tbl = 0;
  uint32_t off_q =
    align128(off_tbl + max_seq * 128u * (uint32_t)sizeof(__fp16));
  uint32_t off_k =
    align128(off_q + m * n_heads * 128u * (uint32_t)sizeof(__fp16));
  uint32_t total =
    align128(off_k + m * n_kv_heads * 128u * (uint32_t)sizeof(__fp16));

  uint8_t *act = memalign(128, total);
  __fp16 *table = (__fp16 *)(act + off_tbl);
  __fp16 *q = (__fp16 *)(act + off_q);
  __fp16 *k = (__fp16 *)(act + off_k);

  ref_rope_table_fill(table, max_seq, 1e6f);
  for (uint32_t i = 0; i < m * n_heads * 128u; ++i)
    q[i] = (__fp16)frand();
  for (uint32_t i = 0; i < m * n_kv_heads * 128u; ++i)
    k[i] = (__fp16)frand();

  __fp16 *q_ref = malloc((size_t)m * n_heads * 128u * sizeof(__fp16));
  __fp16 *k_ref = malloc((size_t)m * n_kv_heads * 128u * sizeof(__fp16));
  memcpy(q_ref, q, (size_t)m * n_heads * 128u * sizeof(__fp16));
  memcpy(k_ref, k, (size_t)m * n_kv_heads * 128u * sizeof(__fp16));
  ref_rope(q_ref, table, m, n_heads, pos);
  ref_rope(k_ref, table, m, n_kv_heads, pos);

  struct nntr_htp_oplist_header cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.n_heads = n_heads;
  cfg.n_kv_heads = n_kv_heads;
  cfg.max_seq = max_seq;

  struct htp_exec_ctx c;
  memset(&c, 0, sizeof(c));
  c.buf[NNTR_HTP_BUF_ACT] = act;
  c.buf_size[NNTR_HTP_BUF_ACT] = total;
  c.cfg = &cfg;
  c.pos = pos;
  c.pool = wp_create(0);

  struct nntr_htp_op_desc d;
  memset(&d, 0, sizeof(d));
  d.kind = NNTR_HTP_OP_ROPE;
  d.m = m;
  d.in0.buf = NNTR_HTP_BUF_ACT;
  d.in0.offset = off_q;
  d.in1.buf = NNTR_HTP_BUF_ACT;
  d.in1.offset = off_k;
  d.in2.buf = NNTR_HTP_BUF_ACT;
  d.in2.offset = off_tbl;

  hvx_op_rope(&c, &d);

  float *ref_f = malloc((size_t)m * n_heads * 128u * sizeof(float));
  float *got_f = malloc((size_t)m * n_heads * 128u * sizeof(float));
  for (uint32_t i = 0; i < m * n_heads * 128u; ++i) {
    ref_f[i] = (float)q_ref[i];
    got_f[i] = (float)q[i];
  }
  int rc = cmp_f("rope_q", ref_f, got_f, m * n_heads * 128u, 5e-3f, 2e-3f);

  for (uint32_t i = 0; i < m * n_kv_heads * 128u; ++i) {
    ref_f[i] = (float)k_ref[i];
    got_f[i] = (float)k[i];
  }
  if (!rc)
    rc = cmp_f("rope_k", ref_f, got_f, m * n_kv_heads * 128u, 5e-3f, 2e-3f);

  free(ref_f);
  free(got_f);
  free(q_ref);
  free(k_ref);
  wp_destroy(c.pool);
  free(act);

  if (rc)
    return 1;

  printf("SIM_TEST rope PASS\n");
  return 0;
}
