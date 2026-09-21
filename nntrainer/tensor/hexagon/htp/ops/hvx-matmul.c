// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-matmul.c
 * @date	18 August 2026
 * @brief	MATMUL_W8A8 / MATMUL_LOGITS: int8 x int8 vrmpy kernel over the
 *		tiled32 WEIGHTS layout. One n-tile (32 output rows) is a
 *		contiguous strip of k/4 vectors; vector i holds bytes
 *		k = 4i..4i+3 of all 32 rows, so one vrmpyacc against a 4-byte
 *		activation splat advances 32 dot products with no horizontal
 *		reduction and no vector->scalar traffic. DDR direct-read path
 *		plus a VTCM/DMA double-buffered streaming path (c->vtcm !=
 *		NULL); both run the same math on the same bytes, so their
 *		output is bit-identical. MATMUL_LOGITS is the same kernel with
 *		m=1 and fp32 output.
 *		MATMUL_W8A16 (down_proj) quantizes the activation per token to
 *		int16 instead (its SwiGLU input is too outlier-heavy for int8),
 *		keeps the row-major weights and accumulates 64 int32 lanes per
 *		(row, token) with Ww_vmpyacc; 4 rows x 2 tokens per block and one
 *		shuffle tree reduces the four rows together.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdlib.h>

#include "dma-queue.h"
#include "htp_ops.h"
#include "hvx-quant.h"

/** Tokens per weight-vector load: one vload feeds MM_TB vrmpyacc. Guarded
 * so a device sweep can build a variant with HEX_EXTRA_CFLAGS=-DMM_TB=8u
 * (HEXAGON.md section 8.2, issue #25); inert at m=1 (decode). */
#ifndef MM_TB
#define MM_TB 4u
#endif

/** Measurement-only knobs (HEXAGON.md section 5.3), all off by default:
 * HTP_MM_NO_VTCM forces the direct DDR read path; HTP_MM_CHUNK_ROWS=<n>
 * caps the DMA chunk at n rows (0 = as many whole tiles as fit in half the
 * slab, the shipping behaviour; ledger (9)); HTP_MM_NO_PREFETCH compiles
 * the cross-op prefetch out (mm_pf_kick); HTP_MM_STREAM_ONLY keeps every
 * weight DMA (chunking, prefetch, and a DMA read of the W8A16 rows that
 * are otherwise read directly) but skips the multiply - the step time of
 * that build is the weight stream plus the non-matmul ops, the ceiling the
 * W8A8 decode goal is stated against (HEXAGON_BENCHMARK.md). Its outputs
 * are garbage by construction. */
#ifndef HTP_MM_CHUNK_ROWS
#define HTP_MM_CHUNK_ROWS 0u
#endif
typedef char
  htp_mm_chunk_rows_check[(HTP_MM_CHUNK_ROWS % NNTR_HTP_TILE_ROWS) == 0u ? 1
                                                                         : -1];

struct mm_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
  bool y_is_f32; /* MATMUL_LOGITS writes fp32, W8A8 fp16 */
};

/** Worker N-range in whole tiles: n0/n1 are multiples of 32 (the validator
 * guarantees n % 32 == 0), so rows [n0, n1) of a tiled tensor are the
 * contiguous bytes [n0 * k, n1 * k). */
static void mm_tile_range(uint32_t n, int wid, int nw, uint32_t *n0,
                          uint32_t *n1) {
  const uint32_t nt = n / NNTR_HTP_TILE_ROWS;
  *n0 = (uint32_t)(((uint64_t)nt * (uint32_t)wid) / (uint32_t)nw) *
        NNTR_HTP_TILE_ROWS;
  *n1 = (uint32_t)(((uint64_t)nt * (uint32_t)(wid + 1)) / (uint32_t)nw) *
        NNTR_HTP_TILE_ROWS;
}

/** One n-tile x tb tokens. wv is the tile's strip: k/4 vectors, since the
 * tiled32 offset kt*4096 + g*128 of k = kt*128 + 4g is 128 * (k/4). swv
 * holds the tile's 32 fp32 scales; xq, sx and y point at the first of the
 * tb token rows. int32 lanes are exact whatever the order, so the sums are
 * bit-exact against the scalar reference; the scale product keeps the
 * reference order ((float)acc * sw) * sx, in qf32 (HEXAGON.md section 7:
 * qf-format ops only), and narrows to fp16 once (Vhf_equals_Wqf32, RNE
 * except exact ties (qf32 carries an implicit half-LSB)). Always inlined
 * with a constant tb so the accumulators stay in registers. */
static inline __attribute__((always_inline)) void
mm_tile(const HVX_Vector *wv, HVX_Vector swv, const int8_t *xq, const float *sx,
        uint8_t *y, size_t y_stride, bool y_is_f32, uint32_t k, uint32_t tb) {
  HVX_Vector acc[MM_TB];
  for (uint32_t u = 0; u < tb; ++u)
    acc[u] = Q6_V_vzero();
  for (uint32_t i = 0; i < k / 4u; ++i) {
    const HVX_Vector w4 = wv[i];
    for (uint32_t u = 0; u < tb; ++u) {
      const uint32_t *xw = (const uint32_t *)(const void *)(xq + (size_t)u * k);
      acc[u] = Q6_Vw_vrmpyacc_VwVbVb(acc[u], w4, Q6_V_vsplat_R(xw[i]));
    }
  }
  for (uint32_t u = 0; u < tb; ++u) {
    HVX_Vector f = Q6_Vsf_equals_Vw(acc[u]);
    f = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(f, swv));
    f = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(f, hvx_vec_splat_f32(sx[u])));
    uint8_t *yu = y + u * y_stride;
    /** Masked stores: 32 floats = 128 B, 32 halves = 64 B. The fp16 y is
     * 64 B-aligned (n % 32 == 0), but the fp32 logits buffer comes from the
     * caller with no alignment guarantee, so neither store may assume one. */
    if (y_is_f32)
      hvx_vec_store_u(yu, 128u, f);
    else
      hvx_vec_store_u(yu, 64u, hvx_vec_f32_to_f16(f, f));
  }
}

/** Output rows [n0, n1) (whole tiles) x tokens [0, m). w_n0 is the strip of
 * row n0 - the DDR tensor at n0*k, or a DMA'd VTCM slab that starts on that
 * tile; either way strip s of the range sits at w_n0 + s*32*k. */
static void mm_tiles(const int8_t *w_n0, const float *sw, const int8_t *xq,
                     const float *sx, uint8_t *y, bool y_is_f32, uint32_t m,
                     uint32_t k, uint32_t n, uint32_t n0, uint32_t n1) {
  const size_t esz = y_is_f32 ? 4u : 2u, y_stride = (size_t)n * esz;
  for (uint32_t jn = n0; jn < n1; jn += NNTR_HTP_TILE_ROWS) {
    const HVX_Vector *wv =
      (const HVX_Vector *)(const void *)(w_n0 + (size_t)(jn - n0) * k);
    const HVX_Vector swv = hvx_vmem(sw + jn);
    uint8_t *yj = y + (size_t)jn * esz;
    uint32_t t = 0;
    for (; t + MM_TB <= m; t += MM_TB)
      mm_tile(wv, swv, xq + (size_t)t * k, sx + t, yj + t * y_stride, y_stride,
              y_is_f32, k, MM_TB);
    for (; t < m; ++t)
      mm_tile(wv, swv, xq + (size_t)t * k, sx + t, yj + t * y_stride, y_stride,
              y_is_f32, k, 1u);
  }
}

/** Push one weight chunk (rows x k bytes, contiguous in both DDR and VTCM)
 * onto the worker's queue. Under the ring invariants (at most the in-op
 * "chunk c+1" kick plus one cross-op prefetch outstanding,
 * HTP_MM_DMA_QUEUE_CAP) the push cannot fail; if it ever does, drain the ring
 * and retry rather than silently skip the chunk (the compute would read stale
 * VTCM). */
static void mm_dma_push(dma_queue_t q, uint8_t *dst, const int8_t *src,
                        uint32_t k, uint32_t rows) {
  if (!dma_queue_push_ddr_to_vtcm(q, dma_make_ptr(dst, src), k, k, rows)) {
    dma_queue_flush(q);
    (void)dma_queue_push_ddr_to_vtcm(q, dma_make_ptr(dst, src), k, k, rows);
  }
}

/** Worker slab geometry for an op with row length k: the per-worker slab
 * (VTCM / nw, rounded down to 128 B) is split into two fixed halves, each a
 * DMA target of whole 32-row tiles. The halves are op-independent on purpose
 * (buf[1] is not buf[0] + rows*k) so that a chunk of the *next* op can be
 * kicked into the half the current op's last chunk does not occupy. Returns
 * false when a half cannot hold one tile: the caller uses the DDR path. */
static bool mm_slab(const struct htp_exec_ctx *c, uint32_t k, int wid, int nw,
                    uint8_t *buf[2], uint32_t *rows_per_buf) {
  const size_t slab_sz = (c->vtcm_size / (uint32_t)nw) & ~(size_t)127;
  const size_t half = (slab_sz / 2u) & ~(size_t)127;
  uint32_t rows = (uint32_t)(half / k) & ~(NNTR_HTP_TILE_ROWS - 1u);
  if (HTP_MM_CHUNK_ROWS != 0u && rows > HTP_MM_CHUNK_ROWS)
    rows = HTP_MM_CHUNK_ROWS;
  if (rows < NNTR_HTP_TILE_ROWS)
    return false;
  buf[0] = c->vtcm + (size_t)wid * slab_sz;
  buf[1] = buf[0] + half;
  *rows_per_buf = rows;
  return true;
}

/** Drop this worker's pending cross-op prefetch: the op it was kicked for
 * is not the one running (a partial forward_upto run, a re-init, a unit
 * test), or the running op takes the DDR path. Waits for the descriptor
 * rather than abandoning it - VTCM must not be written by a forgotten DMA. */
static void mm_pf_drop(struct htp_exec_ctx *c, int wid) {
  if (c->pf && c->pf[wid].desc) {
    dma_queue_flush(c->dmaq[wid]);
    c->pf[wid].desc = NULL;
  }
}

/** Tail of a streamed op (issue #25, ledger (1)): once the last chunk's DMA
 * has landed, the other half-slab is idle for the rest of this op and for
 * every op until the next tiled matmul, so this worker's chunk 0 of that
 * matmul (c->next_mm: same (wid, nw) split, same slab geometry for its k)
 * is kicked into it now and recorded in c->pf[wid]. The DDR keeps
 * streaming while ATTN / RMSNORM / eltwise run on these same worker
 * threads, and the next mm_worker_vtcm starts with the chunk in flight.
 * Skipped when the next op would take the DDR path for this worker (a half
 * cannot hold a tile of its k, or an empty tile range): the fallback must
 * never find a descriptor pending. Compiled out by HTP_MM_NO_PREFETCH (the
 * A/B control of the device handoff). */
static void mm_pf_kick(struct htp_exec_ctx *c, uint8_t *free_buf, int wid,
                       int nw) {
#ifdef HTP_MM_NO_PREFETCH
  (void)c;
  (void)free_buf;
  (void)wid;
  (void)nw;
#else
  const struct nntr_htp_op_desc *nx = c->next_mm;
  uint8_t *nbuf[2];
  uint32_t rows_per_buf, n0, n1, rows;
  if (!nx || nx->n % NNTR_HTP_TILE_ROWS != 0u ||
      !mm_slab(c, nx->k, wid, nw, nbuf, &rows_per_buf))
    return;
  mm_tile_range(nx->n, wid, nw, &n0, &n1);
  if (n0 == n1)
    return;
  rows = rows_per_buf < n1 - n0 ? rows_per_buf : n1 - n0;
  mm_dma_push(c->dmaq[wid], free_buf,
              (const int8_t *)htp_ref_ptr(c, nx->in1) + (size_t)n0 * nx->k,
              nx->k, rows);
  c->pf[wid].desc = nx;
  c->pf[wid].buf = free_buf;
  c->pf[wid].rows = rows;
#endif
}

/** VTCM/DMA streaming path for this worker's N-slab. Returns false if the
 * slab cannot hold two 32-row tiles, in which case the caller falls back to
 * the DDR path. Activations are not copied: the kernel reads c->xq through
 * the cache (4-byte scalar loads gain nothing from VTCM). The queue is the
 * worker's graph-lifetime one (c->dmaq[wid], htp_graph_dma_init); on entry
 * a prefetch record for this very op (pointer identity on the graph's op
 * array) means chunk 0 is already in flight in pf.buf, anything else
 * pending is drained first. */
static bool mm_worker_vtcm(struct htp_exec_ctx *c,
                           const struct nntr_htp_op_desc *d, const int8_t *w,
                           const float *sw, uint8_t *y, bool y_is_f32,
                           uint32_t m, uint32_t k, uint32_t n, uint32_t n0,
                           uint32_t n1, int wid, int nw) {
#ifdef HTP_MM_NO_VTCM
  return false; /* measurement-only: forces the direct DDR read path */
#endif
  uint8_t *buf[2];
  uint32_t rows_per_buf;
  if (!c->dmaq || !mm_slab(c, k, wid, nw, buf, &rows_per_buf))
    return false;
  dma_queue_t q = c->dmaq[wid];
  struct htp_mm_prefetch *pf = &c->pf[wid];

  uint32_t total_rows = n1 - n0;
  uint32_t n_chunks = (total_rows + rows_per_buf - 1) / rows_per_buf;
  uint32_t rows0 = rows_per_buf < total_rows ? rows_per_buf : total_rows;

  /** Chunk 0: already in flight if the previous matmul prefetched it for
   * us (then the pipeline starts from that half); otherwise drain whatever
   * is pending and kick it here. */
  uint32_t start = 0;
  if (pf->desc == d && pf->rows == rows0 &&
      (pf->buf == buf[0] || pf->buf == buf[1])) {
    start = pf->buf == buf[1] ? 1u : 0u;
    pf->hits += 1u;
  } else {
    if (pf->desc)
      dma_queue_flush(q);
    mm_dma_push(q, buf[0], w + (size_t)n0 * k, k, rows0);
  }
  pf->desc = NULL;

  for (uint32_t ci = 0; ci < n_chunks; ++ci) {
    uint32_t row0 = ci * rows_per_buf;
    uint32_t rows = rows_per_buf;
    uint8_t *cur = buf[(start + ci) & 1], *other = buf[(start + ci + 1) & 1];
    if (row0 + rows > total_rows)
      rows = total_rows - row0;

    if (ci + 1 < n_chunks) {
      uint32_t next_row0 = (ci + 1) * rows_per_buf;
      uint32_t next_rows = rows_per_buf;
      if (next_row0 + next_rows > total_rows)
        next_rows = total_rows - next_row0;
      mm_dma_push(q, other, w + (size_t)(n0 + next_row0) * k, k, next_rows);
    }

    dma_queue_pop(q); /* wait for this chunk's DMA (kicked one iteration ago) */

    if (ci + 1 == n_chunks)
      mm_pf_kick(c, other, wid, nw); /* the other half is free from here */

#ifdef HTP_MM_STREAM_ONLY
    (void)cur; /* measurement-only: the bytes landed, nothing is computed */
#else
    mm_tiles((const int8_t *)cur, sw, c->xq, c->xq_scale, y, y_is_f32, m, k, n,
             n0 + row0, n0 + row0 + rows);
#endif
  }
  return true;
}

#ifdef HTP_MM_STREAM_ONLY
/** Measurement-only stand-in for the W8A16 row loop: DMA rows [n0, n1)
 * (row-major, k bytes each) into the worker's two half-slabs, double
 * buffered, and compute nothing. A prefetch pending from the previous
 * tiled matmul is drained first so the ring's pop order stays the pipeline's
 * (the following matmul then kicks its own chunk 0: this build loses the
 * up -> q handover across down, one chunk latency per layer). */
static void mm16_stream_only(struct htp_exec_ctx *c, const int8_t *w,
                             uint32_t k, uint32_t n0, uint32_t n1, int wid,
                             int nw) {
  if (!c->vtcm || !c->dmaq)
    return;
  mm_pf_drop(c, wid);
  const size_t slab_sz = (c->vtcm_size / (uint32_t)nw) & ~(size_t)127;
  const size_t half = (slab_sz / 2u) & ~(size_t)127;
  const uint32_t rows_per = (uint32_t)(half / k);
  if (rows_per == 0u || n1 <= n0)
    return;
  uint8_t *buf[2];
  buf[0] = c->vtcm + (size_t)wid * slab_sz;
  buf[1] = buf[0] + half;
  dma_queue_t q = c->dmaq[wid];
  const uint32_t total = n1 - n0, n_pieces = (total + rows_per - 1u) / rows_per;
  mm_dma_push(q, buf[0], w + (size_t)n0 * k, k,
              rows_per < total ? rows_per : total);
  for (uint32_t i = 0; i < n_pieces; ++i) {
    if (i + 1u < n_pieces) {
      const uint32_t r0 = (i + 1u) * rows_per;
      mm_dma_push(q, buf[(i + 1u) & 1u], w + (size_t)(n0 + r0) * k, k,
                  r0 + rows_per <= total ? rows_per : total - r0);
    }
    dma_queue_pop(q);
  }
}
#endif

/** W8A8 and LOGITS worker: per-worker N-slab [n0, n1) of whole tiles, VTCM
 * streaming when it fits, DDR direct read otherwise. */
static void mm_worker(void *arg, int wid, int nw) {
  struct mm_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t k = d->k, n = d->n;
  const int8_t *w = (const int8_t *)htp_ref_ptr(c, d->in1);
  const float *sw = (const float *)htp_ref_ptr(c, d->in2);
  uint8_t *y = htp_ref_ptr(c, d->out);
  uint32_t n0, n1;
  mm_tile_range(n, wid, nw, &n0, &n1);
  if (n0 == n1) {
    mm_pf_drop(c, wid);
    return;
  }
  if (c->vtcm &&
      mm_worker_vtcm(c, d, w, sw, y, j->y_is_f32, j->m, k, n, n0, n1, wid, nw))
    return;
  /** DDR path: a prefetch pending for this worker (kicked for another op
   * or with another slab size) would otherwise meet the next matmul as a
   * stale record. */
  mm_pf_drop(c, wid);
  mm_tiles(w + (size_t)n0 * k, sw, c->xq, c->xq_scale, y, j->y_is_f32, j->m, k,
           n, n0, n1);
}

/** Per-token quantization spread over the pool: rows [m*wid/nw, m*(wid+1)/nw).
 * Dispatches the HVX routine htp_quant_row_fp16 over the pool, each row
 * quantized by exactly one worker; also used for the single LOGITS row so
 * that no HVX code ever runs on the RPC thread. */
struct quant_job {
  const __fp16 *x;
  void *xq;
  float *sx;
  uint32_t m, k;
  bool i16; /* W8A16 quantizes to int16, W8A8/LOGITS to int8 */
};

static void quant_worker(void *arg, int wid, int nw) {
  struct quant_job *q = arg;
  uint32_t t0 = (uint32_t)(((uint64_t)q->m * wid) / nw);
  uint32_t t1 = (uint32_t)(((uint64_t)q->m * (wid + 1)) / nw);
  for (uint32_t t = t0; t < t1; ++t)
    q->sx[t] =
      q->i16 ? htp_quant_row_fp16_i16(q->x + (size_t)t * q->k,
                                      (int16_t *)q->xq + (size_t)t * q->k, q->k)
             : htp_quant_row_fp16(q->x + (size_t)t * q->k,
                                  (int8_t *)q->xq + (size_t)t * q->k, q->k);
}

/** W8A16 block: MM16_R rows x tb tokens. w points at row jn (row-major, k
 * bytes per row); rows < MM16_R means the last valid row is repeated so the
 * block shape never changes and only 2*rows bytes are stored. Each 128 B of a
 * row is unpacked once to two int16 vectors and multiplied lane-wise against
 * the token's two int16 vectors with Ww_vmpyacc (64 int32 lanes; the widened
 * pair order does not matter because every lane is summed at the end).
 * |x| <= 32767, |w| <= 127: a lane holds k/64 products, lo+hi 2*k/64, so
 * int32 is exact for k <= 16384 (the validator enforces it).
 * Epilogue: lo+hi (int32) converted to sf, then a vshuff tree folds the four
 * rows into one vector (lane 4j+r = row r partial) with qf32 adds; three
 * vror folds in qf32 -> lanes 0..3 hold the four dots; scale in the
 * reference order ((float)dot * sw[n]) * sx[t], narrow with
 * hvx_vec_f32_to_f16 and store 2*rows bytes (HEXAGON.md section 7, rule 8). */
#ifndef MM16_R
#define MM16_R 4u
#endif
#ifndef MM16_TB
#define MM16_TB 2u
#endif
/** The epilogue below reduces exactly four rows (rows (0,1) and (2,3)
 * interleaved at 4 B, then at 8 B, three vror folds, lanes 0..3 stored):
 * another MM16_R would compile and silently produce wrong rows, so it is
 * pinned until that tree is generalised (a kernel change with its own
 * simulator gate per value). MM16_TB is free. */
typedef char mm16_r_check[MM16_R == 4u ? 1 : -1];

static inline __attribute__((always_inline)) void
mm16_block(const int8_t *w, uint32_t k, uint32_t rows, const int16_t *xq,
           const float *sx, const float *sw, __fp16 *y, uint32_t n,
           uint32_t tb) {
  const int8_t *wr[MM16_R];
  HVX_VectorPair acc[MM16_R][MM16_TB];
  for (uint32_t r = 0; r < MM16_R; ++r) {
    wr[r] = w + (size_t)(r < rows ? r : rows - 1u) * k;
    for (uint32_t u = 0; u < tb; ++u)
      acc[r][u] = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
  }
  for (uint32_t i = 0; i < k; i += 128u) {
    for (uint32_t r = 0; r < MM16_R; ++r) {
      const HVX_VectorPair wh = Q6_Wh_vunpack_Vb(hvx_vmem(wr[r] + i));
      for (uint32_t u = 0; u < tb; ++u) {
        const int16_t *xu = xq + (size_t)u * k + i;
        acc[r][u] =
          Q6_Ww_vmpyacc_WwVhVh(acc[r][u], Q6_V_lo_W(wh), hvx_vmem(xu));
        acc[r][u] =
          Q6_Ww_vmpyacc_WwVhVh(acc[r][u], Q6_V_hi_W(wh), hvx_vmem(xu + 64));
      }
    }
  }
  const HVX_Vector swv = hvx_vmemu(sw); /* lanes 0..3 = sw[jn..jn+3] */
  for (uint32_t u = 0; u < tb; ++u) {
    HVX_Vector s[MM16_R];
    for (uint32_t r = 0; r < MM16_R; ++r)
      s[r] = Q6_Vsf_equals_Vw(
        Q6_Vw_vadd_VwVw(Q6_V_lo_W(acc[r][u]), Q6_V_hi_W(acc[r][u])));
    /** rows (0,1) and (2,3) interleaved at 4 B, then the two at 8 B: lane
     * 4j + r holds a partial of row r. */
    HVX_VectorPair p01 = Q6_W_vshuff_VVR(s[1], s[0], -4);
    HVX_VectorPair p23 = Q6_W_vshuff_VVR(s[3], s[2], -4);
    HVX_Vector e = hvx_vec_add_f32_f32(Q6_V_lo_W(p01), Q6_V_hi_W(p01));
    HVX_Vector f = hvx_vec_add_f32_f32(Q6_V_lo_W(p23), Q6_V_hi_W(p23));
    HVX_VectorPair pef = Q6_W_vshuff_VVR(f, e, -8);
    HVX_Vector g = hvx_vec_add_f32_f32(Q6_V_lo_W(pef), Q6_V_hi_W(pef));
    g = hvx_vec_add_f32_f32(g, Q6_V_vror_VR(g, 64));
    g = hvx_vec_add_f32_f32(g, Q6_V_vror_VR(g, 32));
    g = hvx_vec_add_f32_f32(g, Q6_V_vror_VR(g, 16));
    g = hvx_vec_mul_f32_f32(g, swv);
    g = hvx_vec_mul_f32_f32(g, hvx_vec_splat_f32(sx[u]));
    hvx_vec_store_u(y + (size_t)u * n, 2u * rows, hvx_vec_f32_to_f16(g, g));
  }
}

/** MATMUL_W8A16 worker: rows [n*wid/nw, n*(wid+1)/nw) in 4-row blocks, tokens
 * in 2-token blocks with a 1-token tail; DDR direct read.
 * Follow-up: no VTCM/DMA streaming for this kind - the simulator cannot show
 * its value (no DDR model); decide on the device with an A/B like
 * HTP_MM_NO_VTCM (spec 07 (13)). */
static void mm16_worker(void *arg, int wid, int nw) {
  struct mm_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t k = d->k, n = d->n, m = j->m;
  const int8_t *w = (const int8_t *)htp_ref_ptr(c, d->in1);
  const float *sw = (const float *)htp_ref_ptr(c, d->in2);
  __fp16 *y = (__fp16 *)htp_ref_ptr(c, d->out);
  const int16_t *xq = (const int16_t *)(const void *)c->xq;
  uint32_t n0 = (uint32_t)(((uint64_t)n * wid) / nw);
  uint32_t n1 = (uint32_t)(((uint64_t)n * (wid + 1)) / nw);

#ifdef HTP_MM_STREAM_ONLY
  (void)sw;
  (void)y;
  (void)xq;
  (void)m;
  mm16_stream_only(c, w, k, n0, n1, wid, nw);
  return;
#endif
  for (uint32_t jn = n0; jn < n1; jn += MM16_R) {
    const uint32_t rows = n1 - jn < MM16_R ? n1 - jn : MM16_R;
    uint32_t t = 0;
    for (; t + MM16_TB <= m; t += MM16_TB)
      mm16_block(w + (size_t)jn * k, k, rows, xq + (size_t)t * k,
                 c->xq_scale + t, sw + jn, y + (size_t)t * n + jn, n, MM16_TB);
    for (; t < m; ++t)
      mm16_block(w + (size_t)jn * k, k, rows, xq + (size_t)t * k,
                 c->xq_scale + t, sw + jn, y + (size_t)t * n + jn, n, 1u);
  }
}

void hvx_op_matmul_w8a16(struct htp_exec_ctx *c,
                         const struct nntr_htp_op_desc *d) {
  const uint32_t m = htp_m(c, d), k = d->k;
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  struct quant_job q = {x, c->xq, c->xq_scale, m, k, true};
  wp_run(c->pool, quant_worker, &q);
  struct mm_job j = {c, d, m, false};
  wp_run(c->pool, mm16_worker, &j);
}

void hvx_op_matmul_w8a8(struct htp_exec_ctx *c,
                        const struct nntr_htp_op_desc *d) {
  /** validator guarantees this for op-lists; unit tests build descriptors
   * by hand */
  if (d->n % NNTR_HTP_TILE_ROWS != 0u)
    return;
  const uint32_t m = htp_m(c, d), k = d->k;
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  struct quant_job q = {x, c->xq, c->xq_scale, m, k, false};
  wp_run(c->pool, quant_worker, &q);
  struct mm_job j = {c, d, m, false};
  wp_run(c->pool, mm_worker, &j);
}

void hvx_op_matmul_logits(struct htp_exec_ctx *c,
                          const struct nntr_htp_op_desc *d) {
  /** validator guarantees this for op-lists; unit tests build descriptors
   * by hand */
  if (d->n % NNTR_HTP_TILE_ROWS != 0u)
    return;
  const uint32_t k = d->k;
  /** in0 is the full X fp16[n_tokens][k]; only the last token row feeds the
   * logits. The desc carries m=1, so the row offset comes from the runtime
   * chunk size c->n_tokens, not from htp_m(). Quantization of that single
   * row runs on the worker pool via quant_worker, same as W8A8, so no HVX
   * code ever executes on the RPC/caller thread. */
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  const __fp16 *x_last = x + (size_t)(c->n_tokens - 1) * k;
  struct quant_job q = {x_last, c->xq, c->xq_scale, 1u, k, false};
  wp_run(c->pool, quant_worker, &q);
  struct mm_job j = {c, d, 1, true};
  wp_run(c->pool, mm_worker, &j);
}
