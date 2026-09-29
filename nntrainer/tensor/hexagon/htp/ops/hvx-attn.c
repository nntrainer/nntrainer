// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-attn.c
 * @date	19 August 2026
 * @brief	Fused ATTN kernel: KV-cache append, causal SDPA and GQA.
 *		Jobs are (kv head, position block): at decode (m == 1) each kv
 *		head's context is cut into nb lane-block ranges so the 8 heads
 *		balance over the 6 workers, and the block partials (max, sum,
 *		unnormalised output) are merged in a second pass (flash
 *		decoding, issue #58); at prefill nb == 1. One K^T stream and one
 *		V stream per job serve both q heads of the GQA pair. K is cached
 *		transposed ([head_dim][max_seq] per layer/head) so one vector of
 *		64 positions accumulates q[d]*K[d][p] per head dim; scores are
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

/** Position blocks per kv head at m == 1: 0 = auto (n_workers / gcd(n_kv,
 * n_workers), so every worker draws the same number of jobs), 1 = no split
 * (the GQA fusion alone; handoff variant D). Never more than the lane
 * blocks of the context or the workers of the pool. */
#ifndef HTP_ATTN_NB
#define HTP_ATTN_NB 0u
#endif

/** 1: l2fetch the next lane block of K^T (one 2-D descriptor: the 128 lines
 * the score loop touches) during the score pass and the next 16 KB V block
 * during the PV pass (handoff variant C). Values are unchanged. */
#ifndef HTP_ATTN_L2FETCH
#define HTP_ATTN_L2FETCH 0
#endif

/** q heads of one kv head served by one K^T / V stream: two accumulator
 * sets, so a GQA group is walked in pairs (an odd group's last pair
 * computes its head twice and discards the copy). */
#define ATTN_PAIR 2u

struct attn_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
  uint32_t nb;  /**< position blocks per kv head (1 = whole context) */
  uint32_t nlb; /**< lane blocks of the m == 1 context, ceil(L / 64) */
};

static uint32_t gcd_u32(uint32_t a, uint32_t b) {
  while (b) {
    const uint32_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

/** Lane blocks [lb0, lb1) of block b of head h: nlb / nb each, the nlb % nb
 * remainder blocks going to (b + h) % nb < nlb % nb — rotated by head so
 * a worker that always draws the same b (nw % nb == 0) does not always
 * draw the long block. */
static void attn_block_range(uint32_t nlb, uint32_t nb, uint32_t h, uint32_t b,
                             uint32_t *lb0, uint32_t *lb1) {
  const uint32_t base = nlb / nb, rem = nlb % nb;
  uint32_t start = 0;
  for (uint32_t k = 0; k < b; ++k)
    start += base + ((k + h) % nb < rem ? 1u : 0u);
  *lb0 = start;
  *lb1 = start + base + ((b + h) % nb < rem ? 1u : 0u);
}

/** Per-worker scratch: group score rows of max_seq fp32 each. */
static inline float *attn_rows(struct htp_exec_ctx *c, int wid) {
  const struct nntr_htp_oplist_header *cfg = c->cfg;
  return c->attn_scratch +
         (size_t)wid * (cfg->n_heads / cfg->n_kv_heads) * cfg->max_seq;
}

/** Decode partials (htp_attn_scratch_bytes): o [n_heads][nw][hd] fp32 then
 * (m, l) [n_heads][nw][2] fp32. */
static inline float *attn_part_o(struct htp_exec_ctx *c, int nw, uint32_t hq,
                                 uint32_t b) {
  const struct nntr_htp_oplist_header *cfg = c->cfg;
  float *base =
    c->attn_scratch + htp_attn_scratch_rows_bytes(nw, cfg) / sizeof(float);
  return base + ((size_t)hq * nw + b) * cfg->head_dim;
}

static inline float *attn_part_ml(struct htp_exec_ctx *c, int nw, uint32_t hq,
                                  uint32_t b) {
  const struct nntr_htp_oplist_header *cfg = c->cfg;
  float *base =
    c->attn_scratch + htp_attn_scratch_rows_bytes(nw, cfg) / sizeof(float);
  return base + (size_t)cfg->n_heads * nw * cfg->head_dim +
         ((size_t)hq * nw + b) * 2u;
}

/** Scale four fp32 accumulator vectors by inv and narrow them to two fp16
 * rows through hvx_vec_f32_to_f16_shuff (keeps the vmpy lane interleave:
 * even lanes from lo, odd from hi). */
static inline void attn_store_row(__fp16 *orow, HVX_VectorPair a0,
                                  HVX_VectorPair a1, float inv) {
  const HVX_Vector iv = hvx_vec_splat_f32(inv);
  HVX_Vector s0l = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(a0), iv));
  HVX_Vector s0h = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(a0), iv));
  HVX_Vector s1l = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(a1), iv));
  HVX_Vector s1h = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(a1), iv));
  hvx_vmem(orow) = hvx_vec_f32_to_f16_shuff(s0l, s0h);
  hvx_vmem(orow + VLEN_FP16) = hvx_vec_f32_to_f16_shuff(s1l, s1h);
}

/**
 * @brief SDPA of one query row pair over positions [p_lo, p_hi) of one kv
 *        head. scores rows hold fp32 scores at row-local index p - p_lo;
 *        after exp they are narrowed in place into the first half of the
 *        same row (block k reads bytes [256k, 256k+256) and writes
 *        [128k, 128k+128): never ahead of a pending read, for any prefix of
 *        lane blocks processed in order).
 * @param whole non-zero: the range is the whole context — divide by the
 *        sum and store fp16 into orow0/orow1 (the pre-#58 numerics, bit
 *        for bit); zero: write the block partials o (unnormalised fp32,
 *        vmpy lane interleave), m (block max) and l (block sum) into
 *        po0/po1 and pml0/pml1.
 */
static void attn_sdpa_pair(const __fp16 *kh, const __fp16 *vh, uint32_t max_seq,
                           uint32_t hd, const uint16_t *qrow0,
                           const uint16_t *qrow1, int pair_valid, uint32_t p_lo,
                           uint32_t p_hi, float scale, float *row0, float *row1,
                           int whole, __fp16 *orow0, __fp16 *orow1, float *po0,
                           float *po1, float *pml0, float *pml1) {
  const uint32_t len = p_hi - p_lo;
  float *rows[ATTN_PAIR] = {row0, row1};
  uint16_t *s16[ATTN_PAIR] = {(uint16_t *)row0, (uint16_t *)row1};

  /** scores[p] = q . K[p] for 64 positions per vector pair: 128 widening
   * mpyacc of (K^T row d, splat q[d]) per q head, one K^T load for both.
   * Lanes beyond p_hi (up to the 64-multiple; max_seq % 64 == 0 keeps the
   * loads aligned and inside the cache) are computed and ignored. The
   * widened pair holds even positions in lo and odd in hi, so vshuff by 4
   * bytes restores position order before the store. */
#if HTP_ATTN_L2FETCH
  hex_l2fetch(kh + p_lo, VLEN, max_seq * sizeof(__fp16), hd);
#endif
  for (uint32_t p0 = p_lo; p0 < p_hi; p0 += VLEN_FP16) {
#if HTP_ATTN_L2FETCH
    if (p0 + VLEN_FP16 < p_hi)
      hex_l2fetch(kh + p0 + VLEN_FP16, VLEN, max_seq * sizeof(__fp16), hd);
#endif
    HVX_VectorPair acc0 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
    HVX_VectorPair acc1 = acc0;
    const __fp16 *kcol = kh + p0;
    for (uint32_t i = 0; i < hd; ++i) {
      const HVX_Vector kv = hvx_vmem(kcol + (size_t)i * max_seq);
      acc0 = hvx_vec_mpyacc_f32_f16(acc0, kv, Q6_Vh_vsplat_R(qrow0[i]));
      acc1 = hvx_vec_mpyacc_f32_f16(acc1, kv, Q6_Vh_vsplat_R(qrow1[i]));
    }
    HVX_VectorPair s0 = Q6_W_vshuff_VVR(Q6_V_hi_W(acc0), Q6_V_lo_W(acc0), -4);
    hvx_vmem(row0 + (p0 - p_lo)) = Q6_V_lo_W(s0);
    hvx_vmem(row0 + (p0 - p_lo) + VLEN_FP32) = Q6_V_hi_W(s0);
    if (pair_valid) {
      HVX_VectorPair s1 = Q6_W_vshuff_VVR(Q6_V_hi_W(acc1), Q6_V_lo_W(acc1), -4);
      hvx_vmem(row1 + (p0 - p_lo)) = Q6_V_lo_W(s1);
      hvx_vmem(row1 + (p0 - p_lo) + VLEN_FP32) = Q6_V_hi_W(s1);
    }
  }

#if HTP_ATTN_L2FETCH
  hex_l2fetch_block(vh + (size_t)p_lo * hd, VLEN_FP16 * hd * sizeof(__fp16));
#endif
  float mb[ATTN_PAIR] = {-INFINITY, -INFINITY}, lb[ATTN_PAIR] = {0.f, 0.f};
  for (uint32_t g = 0; g < (pair_valid ? ATTN_PAIR : 1u); ++g) {
    float *sc = rows[g];
    float mx = -INFINITY;
    for (uint32_t p = 0; p < len; ++p) {
      sc[p] *= scale;
      if (sc[p] > mx)
        mx = sc[p];
    }
    for (uint32_t p = 0; p < len; ++p)
      sc[p] -= mx;
    hvx_exp_f32((uint8_t *)sc, (const uint8_t *)sc, (int)len, false);
    float sum = 0.f;
    for (uint32_t p = 0; p < len; ++p)
      sum += sc[p];
    for (uint32_t p0 = 0; p0 < len; p0 += VLEN_FP16)
      hvx_vmem(s16[g] + p0) =
        hvx_vec_f32_to_f16(hvx_vmem(sc + p0), hvx_vmem(sc + p0 + VLEN_FP32));
    mb[g] = mx;
    lb[g] = sum;
  }

  /** out[hq] = sum_p scores[p] * V[h][p], accumulated as two IEEE fp32
   * vector pairs per q head (V row = 2 hf vectors) through
   * hvx_vec_mpyacc_f32_f16; one V load serves both heads. Chained qf32
   * adds are not used: they produced inf on v79 (see hvx_dot_fp16), and
   * qf16 adds lose precision under cancellation. */
  HVX_VectorPair a00 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
  HVX_VectorPair a01 = a00, a10 = a00, a11 = a00;
  const uint16_t *pr0 = s16[0], *pr1 = s16[pair_valid ? 1u : 0u];
  for (uint32_t p0 = 0; p0 < len; p0 += VLEN_FP16) {
#if HTP_ATTN_L2FETCH
    if (p0 + VLEN_FP16 < len)
      hex_l2fetch_block(vh + (size_t)(p_lo + p0 + VLEN_FP16) * hd,
                        VLEN_FP16 * hd * sizeof(__fp16));
#endif
    const uint32_t p1 = p0 + VLEN_FP16 < len ? p0 + VLEN_FP16 : len;
    for (uint32_t p = p0; p < p1; ++p) {
      const __fp16 *vrow = vh + (size_t)(p_lo + p) * hd;
      const HVX_Vector v0 = hvx_vmem(vrow), v1 = hvx_vmem(vrow + VLEN_FP16);
      const HVX_Vector pv0 = Q6_Vh_vsplat_R(pr0[p]);
      const HVX_Vector pv1 = Q6_Vh_vsplat_R(pr1[p]);
      a00 = hvx_vec_mpyacc_f32_f16(a00, v0, pv0);
      a01 = hvx_vec_mpyacc_f32_f16(a01, v1, pv0);
      a10 = hvx_vec_mpyacc_f32_f16(a10, v0, pv1);
      a11 = hvx_vec_mpyacc_f32_f16(a11, v1, pv1);
    }
  }

  if (whole) {
    attn_store_row(orow0, a00, a01, 1.0f / lb[0]);
    if (pair_valid)
      attn_store_row(orow1, a10, a11, 1.0f / lb[1]);
    return;
  }
  hvx_vmem(po0) = Q6_V_lo_W(a00);
  hvx_vmem(po0 + VLEN_FP32) = Q6_V_hi_W(a00);
  hvx_vmem(po0 + 2u * VLEN_FP32) = Q6_V_lo_W(a01);
  hvx_vmem(po0 + 3u * VLEN_FP32) = Q6_V_hi_W(a01);
  pml0[0] = mb[0];
  pml0[1] = lb[0];
  if (pair_valid) {
    hvx_vmem(po1) = Q6_V_lo_W(a10);
    hvx_vmem(po1 + VLEN_FP32) = Q6_V_hi_W(a10);
    hvx_vmem(po1 + 2u * VLEN_FP32) = Q6_V_lo_W(a11);
    hvx_vmem(po1 + 3u * VLEN_FP32) = Q6_V_hi_W(a11);
    pml1[0] = mb[1];
    pml1[1] = lb[1];
  }
}

static void attn_sdpa_worker(void *arg, int wid, int nw) {
  struct attn_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const struct nntr_htp_oplist_header *cfg = c->cfg;
  const uint32_t hd = cfg->head_dim; /* 128 by design: 2 hf vectors/row */
  const uint32_t n_heads = cfg->n_heads, n_kv = cfg->n_kv_heads;
  const uint32_t max_seq = cfg->max_seq, group = n_heads / n_kv;
  const uint32_t m = j->m, pos = c->pos, nb = j->nb;
  const __fp16 *q = (const __fp16 *)htp_ref_ptr(c, d->in0);
  const __fp16 *kin = (const __fp16 *)htp_ref_ptr(c, d->in1);
  const __fp16 *vin = (const __fp16 *)htp_ref_ptr(c, d->in2);
  __fp16 *out = (__fp16 *)htp_ref_ptr(c, d->out);
  __fp16 *kv = (__fp16 *)c->buf[NNTR_HTP_BUF_KV];
  const size_t v_off = (size_t)cfg->n_layers * n_kv * max_seq * hd;
  float *rows = attn_rows(c, wid);
  float scale;
  memcpy(&scale, &d->param0, sizeof(scale));

  for (uint32_t jb = (uint32_t)wid; jb < n_kv * nb; jb += (uint32_t)nw) {
    const uint32_t h = jb / nb, b = jb % nb;
    /* K^T: kh[i * max_seq + p] = K[p][i]. V: vh[p * hd + i]. */
    __fp16 *kh = kv + ((size_t)d->layer * n_kv + h) * max_seq * hd;
    __fp16 *vh = kh + v_off;
    uint32_t p_lo = 0, p_hi = 0;
    if (nb > 1) {
      uint32_t lb0, lb1;
      attn_block_range(j->nlb, nb, h, b, &lb0, &lb1);
      p_lo = lb0 * VLEN_FP16;
      p_hi = lb1 * VLEN_FP16 < pos + 1u ? lb1 * VLEN_FP16 : pos + 1u;
    }

    /* 1) KV append: rows [pos, pos+m) of this kv head. With nb > 1 (m ==
     * 1) only the block that holds pos, its sole reader, appends it. */
    if (nb == 1 || (pos >= p_lo && pos < p_hi)) {
      for (uint32_t t = 0; t < m; ++t) {
        const __fp16 *krow = kin + ((size_t)t * n_kv + h) * hd;
        for (uint32_t i = 0; i < hd; ++i)
          kh[(size_t)i * max_seq + pos + t] = krow[i];
        memcpy(vh + (size_t)(pos + t) * hd, vin + ((size_t)t * n_kv + h) * hd,
               hd * sizeof(__fp16));
      }
    }

    /* 2) SDPA for the q heads of this kv head's GQA group, two per K^T /
     * V stream. */
    for (uint32_t g = 0; g < group; g += ATTN_PAIR) {
      const int pair_valid = g + 1u < group;
      const uint32_t hq0 = h * group + g, hq1 = pair_valid ? hq0 + 1u : hq0;
      for (uint32_t t = 0; t < m; ++t) {
        const uint16_t *qrow0 =
          (const uint16_t *)(q + ((size_t)t * n_heads + hq0) * hd);
        const uint16_t *qrow1 =
          (const uint16_t *)(q + ((size_t)t * n_heads + hq1) * hd);
        if (nb == 1) {
          /* causal: attend to [0, pos+t] */
          attn_sdpa_pair(kh, vh, max_seq, hd, qrow0, qrow1, pair_valid, 0,
                         pos + t + 1u, scale, rows + (size_t)g * max_seq,
                         rows + (size_t)(g + 1u) * max_seq, 1,
                         out + ((size_t)t * n_heads + hq0) * hd,
                         out + ((size_t)t * n_heads + hq1) * hd, NULL, NULL,
                         NULL, NULL);
        } else {
          attn_sdpa_pair(
            kh, vh, max_seq, hd, qrow0, qrow1, pair_valid, p_lo, p_hi, scale,
            rows + (size_t)g * max_seq, rows + (size_t)(g + 1u) * max_seq, 0,
            NULL, NULL, attn_part_o(c, nw, hq0, b), attn_part_o(c, nw, hq1, b),
            attn_part_ml(c, nw, hq0, b), attn_part_ml(c, nw, hq1, b));
        }
      }
    }
  }
}

/**
 * @brief Merge the nb block partials of every q head (one head per job):
 *        M = max_b m_b, w_b = exp(m_b - M) through the same HVX exp the
 *        block softmax used (on this worker's score row), l = sum w_b l_b,
 *        o = sum w_b o_b with one qf32 op per add (never chained), then
 *        1/l and the fp16 narrowing of the nb == 1 path.
 */
static void attn_merge_worker(void *arg, int wid, int nw) {
  struct attn_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_oplist_header *cfg = c->cfg;
  const uint32_t hd = cfg->head_dim, n_heads = cfg->n_heads, nb = j->nb;
  __fp16 *out = (__fp16 *)htp_ref_ptr(c, j->d->out);
  float *w = attn_rows(c, wid);

  for (uint32_t hq = (uint32_t)wid; hq < n_heads; hq += (uint32_t)nw) {
    float M = -INFINITY;
    for (uint32_t b = 0; b < nb; ++b) {
      const float mb = attn_part_ml(c, nw, hq, b)[0];
      if (mb > M)
        M = mb;
    }
    for (uint32_t b = 0; b < nb; ++b)
      w[b] = attn_part_ml(c, nw, hq, b)[0] - M;
    hvx_exp_f32((uint8_t *)w, (const uint8_t *)w, (int)nb, false);

    float l = 0.f;
    HVX_Vector o0 = Q6_V_vzero(), o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t b = 0; b < nb; ++b) {
      const float *ob = attn_part_o(c, nw, hq, b);
      const HVX_Vector wv = hvx_vec_splat_f32(w[b]);
      l += w[b] * attn_part_ml(c, nw, hq, b)[1];
      o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(
        o0, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(ob), wv))));
      o1 = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_VsfVsf(o1, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(
                                   hvx_vmem(ob + VLEN_FP32), wv))));
      o2 = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_VsfVsf(o2, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(
                                   hvx_vmem(ob + 2u * VLEN_FP32), wv))));
      o3 = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_VsfVsf(o3, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(
                                   hvx_vmem(ob + 3u * VLEN_FP32), wv))));
    }
    attn_store_row(out + (size_t)hq * hd, Q6_W_vcombine_VV(o1, o0),
                   Q6_W_vcombine_VV(o3, o2), 1.0f / l);
  }
}

/** Follow-up: prefill (m > 1) keeps one job per kv head and re-streams K^T
 * for every query row; the (q head, row block) split that reuses each K^T
 * load across rows and the vector softmax are issue #26. */
void hvx_op_attn(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  const struct nntr_htp_oplist_header *cfg = c->cfg;
  const uint32_t nw = (uint32_t)wp_size(c->pool);
  struct attn_job j = {c, d, htp_m(c, d), 1u, 0u};
  if (j.m == 1u) {
    j.nlb = (c->pos + 1u + VLEN_FP16 - 1u) / VLEN_FP16;
    j.nb =
      HTP_ATTN_NB ? (uint32_t)HTP_ATTN_NB : nw / gcd_u32(cfg->n_kv_heads, nw);
    if (j.nb > nw)
      j.nb = nw;
    if (j.nb > j.nlb)
      j.nb = j.nlb;
    if (j.nb < 1u)
      j.nb = 1u;
  }
  wp_run(c->pool, attn_sdpa_worker, &j);
  if (j.nb > 1u)
    wp_run(c->pool, attn_merge_worker, &j);
}
