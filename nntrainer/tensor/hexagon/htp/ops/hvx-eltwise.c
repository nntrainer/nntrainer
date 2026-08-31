// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-eltwise.c
 * @date	18 August 2026
 * @brief	ADD and SILU_MUL kernels: per-row, 64-half HVX strips
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "htp_ops.h"
#include "hvx-base.h"
#include "hvx-exp.h"
#include "hvx-inverse.h"

struct eltwise_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
};

static inline HVX_Vector hvx_add_f16_via_f32(HVX_Vector a, HVX_Vector b) {
  const HVX_Vector one = hvx_vec_splat_f16(1.0f);
  HVX_VectorPair pa = Q6_Wqf32_vmpy_VhfVhf(a, one);
  HVX_VectorPair pb = Q6_Wqf32_vmpy_VhfVhf(b, one);
  HVX_Vector lo = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(pa)),
                                       Q6_Vsf_equals_Vqf32(Q6_V_lo_W(pb)));
  HVX_Vector hi = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(pa)),
                                       Q6_Vsf_equals_Vqf32(Q6_V_hi_W(pb)));
  return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
}

static void add_worker(void *arg, int wid, int nw) {
  struct eltwise_job *j = arg;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t n = d->n;
  const __fp16 *a = (const __fp16 *)htp_ref_ptr(j->c, d->in0);
  const __fp16 *b = (const __fp16 *)htp_ref_ptr(j->c, d->in1);
  __fp16 *y = (__fp16 *)htp_ref_ptr(j->c, d->out);

  uint32_t t0 = (uint32_t)(((uint64_t)j->m * wid) / nw);
  uint32_t t1 = (uint32_t)(((uint64_t)j->m * (wid + 1)) / nw);

  for (uint32_t t = t0; t < t1; ++t) {
    const __fp16 *arow = a + (size_t)t * n;
    const __fp16 *brow = b + (size_t)t * n;
    __fp16 *yrow = y + (size_t)t * n;
    for (uint32_t i = 0; i < n; i += VLEN_FP16) {
      HVX_Vector av = hvx_vmem(arow + i);
      HVX_Vector bv = hvx_vmem(brow + i);
      /* Add in fp32 through qf-format ops only: widen both operands with
       * Wqf32_vmpy_VhfVhf(x, 1.0), add as sf, narrow with Vhf_equals_Wqf32
       * (same lane interleave both ways). Q6_Vqf16_vadd_VhfVhf quantizes
       * near-cancelling residual adds to 2^-8 steps, and the IEEE
       * Q6_Vhf_vadd_VhfVhf returned all zeros on 8 Elite silicon (fine on
       * the v75/v79 simulators), so neither is used. */
      hvx_vmem(yrow + i) = hvx_add_f16_via_f32(av, bv);
    }
  }
}

void hvx_op_add(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  struct eltwise_job j = {c, d, htp_m(c, d)};
  wp_run(c->pool, add_worker, &j);
}

/* silu(g) = g / (1 + exp(-g)); y = silu(g) * up. Strip = 64 halves,
 * widened to two 32-lane fp32 vectors (hvx_vec_f16_to_f32) so the vendor
 * exp/inverse register primitives (hvx-exp.h, hvx-inverse.h) can run in
 * fp32, multiplied by up in fp32, then narrowed back (hvx_vec_f32_to_f16).
 * No scratch buffers. */
static inline HVX_Vector silu_f32(HVX_Vector g) {
  static const float kMaxExp = 88.7228f;
  const HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
  const HVX_Vector inf = hvx_vec_splat_f32(INFINITY);
  const HVX_Vector one = hvx_vec_splat_f32(1.0f);

  HVX_Vector neg_g = hvx_vec_neg_f32(g);
  /* Clamp -g to 80 so exp stays finite (5.5e34): a large negative g (seen
   * in qwen3 layer 27, |g| > 250) otherwise yields exp -> inf and the
   * inverse approximation turns 1/(1+inf) into NaN instead of 0. With the
   * clamp silu(g) underflows to 0, matching the scalar g/(1+expf(-g)). */
  static const float kMaxArg = 80.0f;
  const HVX_Vector max_arg = hvx_vec_splat_f32(kMaxArg);
  neg_g = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(neg_g, max_arg), max_arg, neg_g);
  HVX_Vector e = hvx_vec_exp_f32_guard(neg_g, max_exp, inf);
  HVX_Vector denom = hvx_vec_add_f32_f32(one, e);
  HVX_Vector inv = hvx_vec_inverse_f32(denom);
  return hvx_vec_mul_f32_f32(g, inv);
}

static void silu_mul_worker(void *arg, int wid, int nw) {
  struct eltwise_job *j = arg;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t n = d->n;
  const __fp16 *g = (const __fp16 *)htp_ref_ptr(j->c, d->in0);
  const __fp16 *u = (const __fp16 *)htp_ref_ptr(j->c, d->in1);
  __fp16 *y = (__fp16 *)htp_ref_ptr(j->c, d->out);

  uint32_t t0 = (uint32_t)(((uint64_t)j->m * wid) / nw);
  uint32_t t1 = (uint32_t)(((uint64_t)j->m * (wid + 1)) / nw);

  for (uint32_t t = t0; t < t1; ++t) {
    const __fp16 *grow = g + (size_t)t * n;
    const __fp16 *urow = u + (size_t)t * n;
    __fp16 *yrow = y + (size_t)t * n;
    for (uint32_t i = 0; i < n; i += VLEN_FP16) {
      HVX_Vector gv = hvx_vmem(grow + i);
      HVX_Vector uv = hvx_vmem(urow + i);

      /* silu and the product both in fp32; one fp16 rounding at the end
       * (same contract as the scalar reference). */
      HVX_VectorPair g32 = hvx_vec_f16_to_f32(gv);
      HVX_VectorPair u32 = hvx_vec_f16_to_f32(uv);
      HVX_Vector p_lo = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_VsfVsf(silu_f32(Q6_V_lo_W(g32)), Q6_V_lo_W(u32)));
      HVX_Vector p_hi = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_VsfVsf(silu_f32(Q6_V_hi_W(g32)), Q6_V_hi_W(u32)));
      hvx_vmem(yrow + i) = hvx_vec_f32_to_f16(p_lo, p_hi);
    }
  }
}

void hvx_op_silu_mul(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  struct eltwise_job j = {c, d, htp_m(c, d)};
  wp_run(c->pool, silu_mul_worker, &j);
}
