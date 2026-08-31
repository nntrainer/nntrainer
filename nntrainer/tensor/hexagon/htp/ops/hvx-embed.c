// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-embed.c
 * @date	19 August 2026
 * @brief	EMBED kernel: gather int8 rows by token id, dequant to fp16
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "htp_ops.h"

struct embed_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
};

/* Scalar on purpose: memory-bound gather of at most max_chunk rows x hidden
 * elements per call; HVX brings nothing here. */
static void embed_worker(void *arg, int wid, int nw) {
  struct embed_job *j = arg;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t k = d->k;
  const int32_t *tokens = (const int32_t *)htp_ref_ptr(j->c, d->in0);
  const int8_t *w = (const int8_t *)htp_ref_ptr(j->c, d->in1);
  const float *scale = (const float *)htp_ref_ptr(j->c, d->in2);
  __fp16 *y = (__fp16 *)htp_ref_ptr(j->c, d->out);

  uint32_t t0 = (uint32_t)(((uint64_t)j->m * wid) / nw);
  uint32_t t1 = (uint32_t)(((uint64_t)j->m * (wid + 1)) / nw);

  for (uint32_t t = t0; t < t1; ++t) {
    uint32_t row = (uint32_t)tokens[t];
    const int8_t *wrow = w + (size_t)row * k;
    float s = scale[row];
    __fp16 *yrow = y + (size_t)t * k;
    for (uint32_t i = 0; i < k; ++i)
      yrow[i] = (__fp16)((float)wrow[i] * s);
  }
}

void hvx_op_embed(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  struct embed_job j = {c, d, htp_m(c, d)};
  wp_run(c->pool, embed_worker, &j);
}
