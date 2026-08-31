// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-rope.c
 * @date	18 August 2026
 * @brief	ROPE kernel: in-place rotate-half rotation of q/k using a
 *		precomputed cos/sin table, one HVX vector pair per (token, head)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "htp_ops.h"
#include "hvx-base.h"

struct rope_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
};

/* Rotate one head_dim=128 slice in place: x0 = lanes [0,64), x1 = [64,128).
 * out0 = x0*cos - x1*sin; out1 = x1*cos + x0*sin. cos/sin are the two
 * halves of the table row for this token's position. */
static inline void rope_rotate(__fp16 *x, const __fp16 *row) {
  HVX_Vector x0 = hvx_vmem(x);
  HVX_Vector x1 = hvx_vmem(x + VLEN_FP16);
  HVX_Vector cs = hvx_vmem(row);
  HVX_Vector sn = hvx_vmem(row + VLEN_FP16);

  /* Products in qf32 (exact for hf inputs), combine as IEEE sf, narrow to
   * hf once. The qf16 version (Vqf16_vmpy + Vqf16_vadd, Vhf_equals_Vqf16)
   * rounds worse than truncation on ~40% of values (v75 sim probe) and
   * cost ~1% PPL on qwen3-0.6b. */
  HVX_VectorPair x0c = Q6_Wqf32_vmpy_VhfVhf(x0, cs);
  HVX_VectorPair x1s = Q6_Wqf32_vmpy_VhfVhf(x1, sn);
  HVX_VectorPair x1c = Q6_Wqf32_vmpy_VhfVhf(x1, cs);
  HVX_VectorPair x0s = Q6_Wqf32_vmpy_VhfVhf(x0, sn);

  HVX_Vector o0l = Q6_Vqf32_vsub_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(x0c)),
                                        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(x1s)));
  HVX_Vector o0h = Q6_Vqf32_vsub_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(x0c)),
                                        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(x1s)));
  HVX_Vector o1l = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(x1c)),
                                        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(x0s)));
  HVX_Vector o1h = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(x1c)),
                                        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(x0s)));

  hvx_vmem(x) = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(o0h, o0l));
  hvx_vmem(x + VLEN_FP16) = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(o1h, o1l));
}

static void rope_worker(void *arg, int wid, int nw) {
  struct rope_job *j = arg;
  const struct nntr_htp_op_desc *d = j->d;
  struct htp_exec_ctx *c = j->c;
  const uint32_t m = j->m;
  const uint32_t n_heads = c->cfg->n_heads;
  const uint32_t n_kv_heads = c->cfg->n_kv_heads;
  const uint32_t pos = c->pos;

  __fp16 *q = (__fp16 *)htp_ref_ptr(c, d->in0);
  __fp16 *k = (__fp16 *)htp_ref_ptr(c, d->in1);
  const __fp16 *table = (const __fp16 *)htp_ref_ptr(c, d->in2);

  /* Flat index space: [0, m*n_heads) is q, [m*n_heads, m*(n_heads+n_kv_heads))
   * is k. Split evenly across workers. */
  const uint64_t q_units = (uint64_t)m * n_heads;
  const uint64_t total = q_units + (uint64_t)m * n_kv_heads;
  const uint64_t u0 = (total * (uint64_t)wid) / nw;
  const uint64_t u1 = (total * (uint64_t)(wid + 1)) / nw;

  for (uint64_t u = u0; u < u1; ++u) {
    __fp16 *x;
    uint32_t t;
    if (u < q_units) {
      t = (uint32_t)(u / n_heads);
      uint32_t h = (uint32_t)(u % n_heads);
      x = q + ((size_t)t * n_heads + h) * 128;
    } else {
      uint64_t v = u - q_units;
      t = (uint32_t)(v / n_kv_heads);
      uint32_t h = (uint32_t)(v % n_kv_heads);
      x = k + ((size_t)t * n_kv_heads + h) * 128;
    }
    const __fp16 *row = table + (size_t)(pos + t) * 128;
    rope_rotate(x, row);
  }
}

void hvx_op_rope(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  const uint32_t m = htp_m(c, d);
  struct rope_job j = {c, d, m};
  wp_run(c->pool, rope_worker, &j);
}
