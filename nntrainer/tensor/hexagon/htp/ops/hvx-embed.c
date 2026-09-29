// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-embed.c
 * @date	19 August 2026
 * @brief	EMBED kernel: gather tiled32 int8 rows by token id, dequant to fp16
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

/** Scalar on purpose: memory-bound gather of at most max_chunk rows x hidden
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
    float s = scale[row];
    __fp16 *yrow = y + (size_t)t * k;
    /* tiled32 table: 4 consecutive k share one lane (nntr_htp_tile_off). */
    for (uint32_t i = 0; i < k; i += 4u) {
      const int8_t *w4 = w + nntr_htp_tile_off(row, i, k);
      for (uint32_t b = 0; b < 4u; ++b)
        yrow[i + b] = (__fp16)((float)w4[b] * s);
    }
  }
}

void hvx_op_embed(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  struct embed_job j = {c, d, htp_m(c, d)};
  wp_run(c->pool, embed_worker, &j);
}
