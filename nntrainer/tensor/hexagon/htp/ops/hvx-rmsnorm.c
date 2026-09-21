// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-rmsnorm.c
 * @date	18 August 2026
 * @brief	RMSNORM kernel: whole-row normalization, or PER_HEAD QK-Norm
 *		(head_dim-sized chunks sharing one gamma vector)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <math.h>
#include <string.h>

#include "htp_ops.h"
#include "hvx-f16-math.h"

struct rmsnorm_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m, chunk;
};

static void rmsnorm_worker(void *arg, int wid, int nw) {
  struct rmsnorm_job *j = arg;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t n = d->n, chunk = j->chunk;
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(j->c, d->in0);
  const __fp16 *gamma = (const __fp16 *)htp_ref_ptr(j->c, d->in1);
  __fp16 *y = (__fp16 *)htp_ref_ptr(j->c, d->out);
  float eps;
  memcpy(&eps, &d->param0, sizeof(eps));

  uint32_t t0 = (uint32_t)(((uint64_t)j->m * wid) / nw);
  uint32_t t1 = (uint32_t)(((uint64_t)j->m * (wid + 1)) / nw);

  for (uint32_t t = t0; t < t1; ++t) {
    const __fp16 *xrow = x + (size_t)t * n;
    __fp16 *yrow = y + (size_t)t * n;
    for (uint32_t c0 = 0; c0 < n; c0 += chunk) {
      float sumsq = hvx_sumsq_fp16(xrow + c0, chunk);
      float r = 1.0f / sqrtf(sumsq / (float)chunk + eps);
      hvx_scale_mul_fp16(xrow + c0, gamma, r, yrow + c0, chunk);
    }
  }
}

void hvx_op_rmsnorm(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  const uint32_t m = htp_m(c, d);
  const uint32_t chunk =
    (d->flags & NNTR_HTP_FLAG_PER_HEAD) ? c->cfg->head_dim : d->n;
  struct rmsnorm_job j = {c, d, m, chunk};
  wp_run(c->pool, rmsnorm_worker, &j);
}
