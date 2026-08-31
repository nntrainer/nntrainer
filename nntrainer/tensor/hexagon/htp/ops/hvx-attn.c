// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-attn.c
 * @date	19 August 2026
 * @brief	Fused ATTN kernel: KV-cache append, causal SDPA and GQA.
 *		Workers split by kv head. K is cached transposed
 *		([head_dim][max_seq] per layer/head) so one vector of 64
 *		positions accumulates q[d]*K[d][p] per head dim; scores are
 *		fp32 in per-worker scratch, softmax uses the borrowed HVX exp,
 *		and the output is accumulated in IEEE fp32 vector pairs (qf16
 *		adds lose precision under cancellation; chained qf32 adds break
 *		on v79).
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <math.h>
#include <string.h>

#include "htp_ops.h"
#include "hvx-exp.h"
#include "hvx-f16-math.h"

struct attn_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
};

static void attn_worker(void *arg, int wid, int nw) {
  struct attn_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const struct nntr_htp_oplist_header *cfg = c->cfg;
  const uint32_t hd = cfg->head_dim; /* 128 by design: 2 hf vectors/row */
  const uint32_t n_heads = cfg->n_heads, n_kv = cfg->n_kv_heads;
  const uint32_t max_seq = cfg->max_seq, group = n_heads / n_kv;
  const uint32_t m = j->m, pos = c->pos;
  const __fp16 *q = (const __fp16 *)htp_ref_ptr(c, d->in0);
  const __fp16 *kin = (const __fp16 *)htp_ref_ptr(c, d->in1);
  const __fp16 *vin = (const __fp16 *)htp_ref_ptr(c, d->in2);
  __fp16 *out = (__fp16 *)htp_ref_ptr(c, d->out);
  __fp16 *kv = (__fp16 *)c->buf[NNTR_HTP_BUF_KV];
  const size_t v_off = (size_t)cfg->n_layers * n_kv * max_seq * hd;
  float *scores = c->attn_scratch + (size_t)wid * max_seq;
  /* After exp, the fp32 scores are narrowed in place into the first half
   * of the same scratch (block k reads bytes [256k, 256k+256) and writes
   * [128k, 128k+128): never ahead of a pending read). */
  uint16_t *s16 = (uint16_t *)scores;
  float scale;
  memcpy(&scale, &d->param0, sizeof(scale));

  uint32_t h0 = (uint32_t)(((uint64_t)n_kv * wid) / nw);
  uint32_t h1 = (uint32_t)(((uint64_t)n_kv * (wid + 1)) / nw);

  for (uint32_t h = h0; h < h1; ++h) {
    /* K^T: kh[i * max_seq + p] = K[p][i]. V: vh[p * hd + i]. */
    __fp16 *kh = kv + ((size_t)d->layer * n_kv + h) * max_seq * hd;
    __fp16 *vh = kh + v_off;

    /* 1) KV append: rows [pos, pos+m) of this kv head. */
    for (uint32_t t = 0; t < m; ++t) {
      const __fp16 *krow = kin + ((size_t)t * n_kv + h) * hd;
      for (uint32_t i = 0; i < hd; ++i)
        kh[(size_t)i * max_seq + pos + t] = krow[i];
      memcpy(vh + (size_t)(pos + t) * hd, vin + ((size_t)t * n_kv + h) * hd,
             hd * sizeof(__fp16));
    }

    /* 2) SDPA for every q head in this kv head's GQA group. */
    for (uint32_t g = 0; g < group; ++g) {
      const uint32_t hq = h * group + g;
      for (uint32_t t = 0; t < m; ++t) {
        const uint16_t *qrow =
          (const uint16_t *)(q + ((size_t)t * n_heads + hq) * hd);
        const uint32_t L = pos + t + 1; /* causal: attend to [0, pos+t] */

        /* scores[p] = q . K[p] for 64 positions per vector pair: 128
         * widening mpyacc of (K^T row d, splat q[d]). Lanes beyond L
         * (up to the 64-multiple; max_seq % 64 == 0 keeps the loads
         * aligned and inside the cache) are computed and ignored. The
         * widened pair holds even positions in lo and odd in hi, so
         * vshuff by 4 bytes restores position order before the store. */
        for (uint32_t p0 = 0; p0 < L; p0 += VLEN_FP16) {
          HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
          for (uint32_t i = 0; i < hd; ++i)
            acc = hvx_vec_mpyacc_f32_f16(
              acc, hvx_vmem(kh + (size_t)i * max_seq + p0),
              Q6_Vh_vsplat_R(qrow[i]));
          HVX_VectorPair s =
            Q6_W_vshuff_VVR(Q6_V_hi_W(acc), Q6_V_lo_W(acc), -4);
          hvx_vmem(scores + p0) = Q6_V_lo_W(s);
          hvx_vmem(scores + p0 + VLEN_FP32) = Q6_V_hi_W(s);
        }

        float mx = -INFINITY;
        for (uint32_t p = 0; p < L; ++p) {
          scores[p] *= scale;
          if (scores[p] > mx)
            mx = scores[p];
        }
        for (uint32_t p = 0; p < L; ++p)
          scores[p] -= mx;
        hvx_exp_f32((uint8_t *)scores, (const uint8_t *)scores, (int)L, false);
        float sum = 0.f;
        for (uint32_t p = 0; p < L; ++p)
          sum += scores[p];
        const float inv = 1.0f / sum;
        for (uint32_t p0 = 0; p0 < L; p0 += VLEN_FP16)
          hvx_vmem(s16 + p0) = hvx_vec_f32_to_f16(
            hvx_vmem(scores + p0), hvx_vmem(scores + p0 + VLEN_FP32));

        /* out[t,hq] = sum_p scores[p] * V[h][p], accumulated as two IEEE
         * fp32 vector pairs (V row = 2 hf vectors) through
         * hvx_vec_mpyacc_f32_f16. Chained qf32 adds are not used: they
         * produced inf on v79 (see hvx_dot_fp16), and qf16 adds lose
         * precision under cancellation. */
        HVX_VectorPair a0 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        HVX_VectorPair a1 = a0;
        for (uint32_t p = 0; p < L; ++p) {
          const __fp16 *vrow = vh + (size_t)p * hd;
          HVX_Vector pv = Q6_Vh_vsplat_R(s16[p]);
          a0 = hvx_vec_mpyacc_f32_f16(a0, hvx_vmem(vrow), pv);
          a1 = hvx_vec_mpyacc_f32_f16(a1, hvx_vmem(vrow + VLEN_FP16), pv);
        }

        /* 1/sum in fp32, then narrow through hvx_vec_f32_to_f16_shuff
         * (keeps the vmpy lane interleave: even lanes from lo, odd from hi). */
        __fp16 *orow = out + ((size_t)t * n_heads + hq) * hd;
        const HVX_Vector iv = hvx_vec_splat_f32(inv);
        HVX_Vector s0l =
          Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(a0), iv));
        HVX_Vector s0h =
          Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(a0), iv));
        HVX_Vector s1l =
          Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(a1), iv));
        HVX_Vector s1h =
          Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(a1), iv));
        hvx_vmem(orow) = hvx_vec_f32_to_f16_shuff(s0l, s0h);
        hvx_vmem(orow + VLEN_FP16) = hvx_vec_f32_to_f16_shuff(s1l, s1h);
      }
    }
  }
}

/* ponytail: workers split by kv head (8 on qwen3) and every query row
 * re-streams K^T; split by (q head, row block) and reuse each K^T load
 * across rows if prefill ATTN still dominates after this. */
void hvx_op_attn(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  struct attn_job j = {c, d, htp_m(c, d)};
  wp_run(c->pool, attn_worker, &j);
}
