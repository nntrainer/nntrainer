// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-matmul.c
 * @date	18 August 2026
 * @brief	MATMUL_W8A8 kernel: fp16 activation x int8 weight. DDR
 *		direct-read path plus a VTCM/DMA double-buffered streaming
 *		path (used when c->vtcm != NULL); both produce bit-identical
 *		fp16 output since the streaming path only moves the same
 *		bytes through VTCM before the identical dot/scale math.
 *		Also MATMUL_LOGITS: quantizes only the last token row and
 *		emits fp32 logits through the same DDR dot/scale path.
 *		MATMUL_W8A16 keeps the activation in fp16 (no per-token int8
 *		quantization) and accumulates in qf32; used for down_proj,
 *		whose SwiGLU input is too outlier-heavy for per-token int8.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdlib.h>
#include <string.h>

#include "dma-queue.h"
#include "htp_ops.h"
#include "hvx-f16-math.h"
#include "hvx-quant.h"

/* Ring depth for the per-worker DMA queue: only ever one chunk in flight
 * (kick c+1, wait c), but the ring needs room for 2 outstanding slots so
 * the push for c+1 does not collide with the not-yet-popped descriptor
 * for c. Must be a power of two (dma_queue_init rounds up regardless). */
#define MM_DMA_QUEUE_CAP 4

struct mm_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
};

/* Task 6 compute: dot + scale + store for output columns [n0, n1), using
 * whatever xq/xrow the caller hands in (DDR pointers, or VTCM copies -
 * same bytes either way, so the result is bit-identical). y_is_f32 selects
 * the store type: fp16 for W8A8, fp32 for MATMUL_LOGITS. */
static void mm_compute_range(const int8_t *w, uint32_t w_base, const float *sw,
                             const int8_t *xq, const float *xq_scale, void *y,
                             bool y_is_f32, uint32_t m, uint32_t k, uint32_t n,
                             uint32_t n0, uint32_t n1) {
  for (uint32_t jn = n0; jn < n1; ++jn) {
    const int8_t *wrow = w + (size_t)(jn - w_base) * k;
    for (uint32_t t = 0; t < m; ++t) {
      int32_t acc = hvx_dot_i8(wrow, xq + (size_t)t * k, k);
      float v = (float)acc * sw[jn] * xq_scale[t];
      if (y_is_f32)
        ((float *)y)[(size_t)t * n + jn] = v;
      else
        ((__fp16 *)y)[(size_t)t * n + jn] = (__fp16)v;
    }
  }
}

/* VTCM/DMA streaming path for this worker's N-slab. Returns false if the
 * slab is too small to hold the Xq copy plus at least one double-buffered
 * weight row, in which case the caller falls back to the DDR path. */
static bool mm_worker_vtcm(struct htp_exec_ctx *c, const int8_t *w,
                           const float *sw, __fp16 *y, uint32_t m, uint32_t k,
                           uint32_t n, uint32_t n0, uint32_t n1, int wid,
                           int nw) {
#ifdef HTP_MM_NO_VTCM
  return false; /* measurement-only: forces the direct DDR read path */
#endif
  /* Round the per-worker slab size down to a multiple of 128 first, so
   * every worker's slab base (and therefore buf[0]/buf[1]) stays 128B
   * aligned - hvx_dot_i8 does raw HVX_Vector loads and requires it. */
  size_t slab_sz = (c->vtcm_size / (uint32_t)nw) & ~(size_t)127;
  uint8_t *slab = c->vtcm + (size_t)wid * slab_sz;
  size_t xq_bytes = (size_t)m * k;

  if (xq_bytes + HEX_L2_LINE_SIZE + 2 * k > slab_sz)
    return false; /* not even one double-buffered row fits */

  size_t half = (slab_sz - xq_bytes - HEX_L2_LINE_SIZE) / 2;
  uint32_t rows_per_buf = (uint32_t)(half / k);
  if (rows_per_buf < 1)
    return false;

  int8_t *xq_local = (int8_t *)slab;
  memcpy(xq_local, c->xq, xq_bytes);
  uint8_t *buf[2];
  buf[0] = slab + xq_bytes + HEX_L2_LINE_SIZE;
  buf[1] = buf[0] + (size_t)rows_per_buf * k;

  void *qmem =
    memalign(dma_queue_alignof(), dma_queue_sizeof(MM_DMA_QUEUE_CAP));
  if (!qmem)
    return false;
  dma_queue_t q = dma_queue_init(qmem, MM_DMA_QUEUE_CAP, (uintptr_t)c->vtcm,
                                 c->vtcm_size, NULL);

  uint32_t total_rows = n1 - n0;
  uint32_t n_chunks = (total_rows + rows_per_buf - 1) / rows_per_buf;

  /* Kick the first chunk before entering the pipeline. */
  uint32_t rows0 = rows_per_buf < total_rows ? rows_per_buf : total_rows;
  dma_queue_push_ddr_to_vtcm(q, dma_make_ptr(buf[0], w + (size_t)n0 * k), k, k,
                             rows0);

  for (uint32_t ci = 0; ci < n_chunks; ++ci) {
    uint32_t row0 = ci * rows_per_buf;
    uint32_t rows = rows_per_buf;
    if (row0 + rows > total_rows)
      rows = total_rows - row0;

    if (ci + 1 < n_chunks) {
      uint32_t next_row0 = (ci + 1) * rows_per_buf;
      uint32_t next_rows = rows_per_buf;
      if (next_row0 + next_rows > total_rows)
        next_rows = total_rows - next_row0;
      dma_queue_push_ddr_to_vtcm(
        q, dma_make_ptr(buf[(ci + 1) & 1], w + (size_t)(n0 + next_row0) * k), k,
        k, next_rows);
    }

    dma_queue_pop(q); /* wait for this chunk's DMA (kicked one iteration ago) */

    mm_compute_range((const int8_t *)buf[ci & 1], n0 + row0, sw, xq_local,
                     c->xq_scale, y, false, m, k, n, n0 + row0,
                     n0 + row0 + rows);
  }

  dma_queue_free(q);
  free(qmem);
  return true;
}

static void mm_worker(void *arg, int wid, int nw) {
  struct mm_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t k = d->k, n = d->n;
  const int8_t *w = (const int8_t *)htp_ref_ptr(c, d->in1);
  const float *sw = (const float *)htp_ref_ptr(c, d->in2);
  __fp16 *y = (__fp16 *)htp_ref_ptr(c, d->out);
  /* Per-worker N-slab [n0,n1) - outer N, inner M (isomorphic to the VTCM
   * tile-reuse layout). */
  uint32_t n0 = (uint32_t)(((uint64_t)n * wid) / nw);
  uint32_t n1 = (uint32_t)(((uint64_t)n * (wid + 1)) / nw);

  if (c->vtcm && mm_worker_vtcm(c, w, sw, y, j->m, k, n, n0, n1, wid, nw))
    return;

  mm_compute_range(w, 0, sw, c->xq, c->xq_scale, y, false, j->m, k, n, n0, n1);
}

/* MATMUL_LOGITS worker: m=1, fp32 output, DDR direct-read only (a single
 * activation row makes VTCM streaming pointless). Same N-slab split and the
 * same mm_compute_range/hvx_dot_i8 path as the W8A8 DDR case. */
static void mm_logits_worker(void *arg, int wid, int nw) {
  struct mm_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t k = d->k, n = d->n;
  const int8_t *w = (const int8_t *)htp_ref_ptr(c, d->in1);
  const float *sw = (const float *)htp_ref_ptr(c, d->in2);
  float *y = (float *)htp_ref_ptr(c, d->out);
  uint32_t n0 = (uint32_t)(((uint64_t)n * wid) / nw);
  uint32_t n1 = (uint32_t)(((uint64_t)n * (wid + 1)) / nw);

  mm_compute_range(w, 0, sw, c->xq, c->xq_scale, y, true, 1, k, n, n0, n1);
}

/* fp16 x . int8 w dot in fp32: each 128B of w is sign-extended to two
 * int16 vectors, converted to hf (exact for |w| <= 127) and multiply-
 * accumulated with the matching 64-half x vectors in IEEE sf (see
 * hvx_dot_fp16 for why not qf32). k%128==0, w/x 128B aligned (validator +
 * lowering guarantee both). */
static inline float hvx_dot_fp16_i8(const __fp16 *x, const int8_t *w,
                                    uint32_t k) {
  HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
  for (uint32_t i = 0; i < k; i += 128u) {
    HVX_VectorPair wh = Q6_Wh_vunpack_Vb(hvx_vmem(w + i));
    acc = hvx_vec_mpyacc_f32_f16(acc, hvx_vmem(x + i),
                                 Q6_Vhf_equals_Vh(Q6_V_lo_W(wh)));
    acc = hvx_vec_mpyacc_f32_f16(acc, hvx_vmem(x + i + 64u),
                                 Q6_Vhf_equals_Vh(Q6_V_hi_W(wh)));
  }
  return hvx_sum_sf_pair(acc);
}

/* MATMUL_W8A16 worker: same N-slab split, DDR direct read.
 * ponytail: no VTCM/DMA streaming for this kind yet - add by generalizing
 * mm_worker_vtcm if the M5 measurements show down_proj prefill needs it. */
static void mm_w8a16_worker(void *arg, int wid, int nw) {
  struct mm_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t k = d->k, n = d->n, m = j->m;
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  const int8_t *w = (const int8_t *)htp_ref_ptr(c, d->in1);
  const float *sw = (const float *)htp_ref_ptr(c, d->in2);
  __fp16 *y = (__fp16 *)htp_ref_ptr(c, d->out);
  uint32_t n0 = (uint32_t)(((uint64_t)n * wid) / nw);
  uint32_t n1 = (uint32_t)(((uint64_t)n * (wid + 1)) / nw);

  for (uint32_t jn = n0; jn < n1; ++jn) {
    const int8_t *wrow = w + (size_t)jn * k;
    for (uint32_t t = 0; t < m; ++t)
      y[(size_t)t * n + jn] =
        (__fp16)(hvx_dot_fp16_i8(x + (size_t)t * k, wrow, k) * sw[jn]);
  }
}

void hvx_op_matmul_w8a16(struct htp_exec_ctx *c,
                         const struct nntr_htp_op_desc *d) {
  struct mm_job j = {c, d, htp_m(c, d)};
  wp_run(c->pool, mm_w8a16_worker, &j);
}

void hvx_op_matmul_w8a8(struct htp_exec_ctx *c,
                        const struct nntr_htp_op_desc *d) {
  const uint32_t m = htp_m(c, d), k = d->k;
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  for (uint32_t t = 0; t < m; ++t) /* one quant pass on op entry */
    c->xq_scale[t] =
      htp_quant_row_fp16(x + (size_t)t * k, c->xq + (size_t)t * k, k);
  struct mm_job j = {c, d, m};
  wp_run(c->pool, mm_worker, &j);
}

void hvx_op_matmul_logits(struct htp_exec_ctx *c,
                          const struct nntr_htp_op_desc *d) {
  const uint32_t k = d->k;
  /* in0 is the full X fp16[n_tokens][k]; only the last token row feeds the
   * logits. The desc carries m=1, so the row offset comes from the runtime
   * chunk size c->n_tokens, not from htp_m(). */
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  const __fp16 *x_last = x + (size_t)(c->n_tokens - 1) * k;
  c->xq_scale[0] = htp_quant_row_fp16(x_last, c->xq, k);
  struct mm_job j = {c, d, 1};
  wp_run(c->pool, mm_logits_worker, &j);
}
