// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_f16.c
 * @date   14 Sep 2026
 * @brief  fp16 flash attention on HMX over nntrainer's fp16 KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * FA-2 online softmax, block by block over the cache, with the running-max
 * rescale of the partial output done on HMX as a diagonal matmul:
 *
 *   S      = Q.K^T                      (Q scaled by log2e/sqrt(hd) up front)
 *   S      = c.tanh(S/c)                (optional softcap, c = cap*log2e)
 *   m'     = max(m, rowmax S)           (masked lanes are -inf first)
 *   P      = 2^(S - m'),  l' = l*2^(m-m') + rowsum P
 *   O'     = diag(2^(m-m')).O + P.V     (one HMX accumulation, read once)
 *   out    = O / l                       (in the HVX store, f32)
 *
 * A sink logit enters as the initial state m = sink*log2e, l = 1: it is one
 * more term of the denominator with no value behind it.
 *
 * S, P and O never leave VTCM. K and V arrive by DMA either as raw cache
 * rows that HVX turns into tiles, or -- with a resident tile cache
 * (hexkl_kv_tiles_f16) -- as ready tiles, one contiguous DMA of 2 KiB rows
 * per block and no tile work. Causality and the sliding window are
 * geometry (blocks not visited, lanes masked), never a mask tensor.
 *
 * Pipeline (llama.cpp's shape): with block k's scores in S[k&1], the pool
 * workers run the softmax of block k while the calling thread lands block
 * k+1 (DMA issued before the softmax started), builds its tiles, and runs
 * its Q.K^T on HMX into S[(k+1)&1]. Then the caller waits, builds the D
 * tiles from the workers' rescale factors, and runs block k's P.V on HMX.
 * Every unit of work writes only its own rows or tiles, so the overlap is
 * bitwise invisible. See docs/backend_guide/htp_backend/
 * 20_hmx_flash_attention_plan.md.
 */

#include "hexkl_attn_f16.h"

#include <math.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hexkl_dma_ring.h"
#include "hexkl_micro.h"
#include "hvx_attn_softmax_f16.h"
#include "hvx_tile_f16.h"

#define TB HEXKL_ATTN_TILE_BYTES
#define LOG2E 1.4426950408889634f
/** @brief Row cap for the per-block rescale scratch; 8 row tiles is far
 *         above any G*Br llama.cpp's chooser lands on. */
#define MAX_G_BR 256u
/** @brief Fewer (col, d) tile pairs than this build inline in the prologue. */
#define TILE_POOL_MIN_UNITS 4u

static int map_plan_rc(int rc) {
  switch (rc) {
  case HEXKL_ATTN_OK:
    return AEE_SUCCESS;
  case HEXKL_ATTN_ENOMEM:
    return AEE_ENOMEMORY;
  default:
    return AEE_EBADPARM;
  }
}

/** @brief Everything one call needs, so the phase functions stay short. */
typedef struct {
  uint8_t *vb;
  const hexkl_attn_f16_shape *s;
  const hexkl_attn_f16_tiling *t;
  const hexkl_attn_f16_io *io;
  hvx_worker_pool *pool;
  hexkl_attn_f16_layout L;
  int resident;            /**< io->kt_tiles != NULL */
  float scale;             /**< log2e / sqrt(hd), folded into Q */
  uint16_t *m_hf;          /**< [g_br] running max, fp16 bits, in VTCM */
  float *l_f32;            /**< [g_br] running sum, in VTCM */
  uint16_t a_hf[MAX_G_BR]; /**< this block's 2^(m-m') per row */
  HVX_Vector col_idx;      /**< {0,0,1,1,...,31,31} */
  HVX_Vector cap_hf;       /**< softcap*log2e, every lane; unused if 0 */
  HVX_Vector inv_cap_hf;   /**< 2*log2e/(softcap*log2e) */
  int softcap_on;
  /** per-block parameters handed to the pool workers; frozen while a
      softmax job is in flight */
  uint32_t w_qb, w_kb, w_buf, w_sbuf;
  int w_masked;
  hexkl_attn_f16_stats *st;
} attn_ctx;

/* --- tile addressing ------------------------------------------------- */

static inline uint32_t off_of(const attn_ctx *c, const void *p) {
  return (uint32_t)((const uint8_t *)p - c->vb);
}
static inline HVX_Vector *q_tile(const attn_ctx *c, uint32_t i, uint32_t d) {
  return (HVX_Vector *)(c->vb + c->L.q_ah + (i * c->L.n_dot_tiles + d) * TB);
}
/** K^T and V tiles are both (col, d) ordered: a block's tiles are then one
 *  contiguous run, which is what lets the resident path land a block with
 *  a single DMA. */
static inline HVX_Vector *kt_tile(const attn_ctx *c, uint32_t buf, uint32_t d,
                                  uint32_t col) {
  return (HVX_Vector *)(c->vb + c->L.kt_wh[buf] +
                        (col * c->L.n_dot_tiles + d) * TB);
}
static inline HVX_Vector *v_tile(const attn_ctx *c, uint32_t buf, uint32_t col,
                                 uint32_t d) {
  return (HVX_Vector *)(c->vb + c->L.v_wh[buf] +
                        (col * c->L.n_dot_tiles + d) * TB);
}
static inline HVX_Vector *s_tile(const attn_ctx *c, uint32_t sbuf, uint32_t i,
                                 uint32_t col) {
  return (HVX_Vector *)(c->vb + c->L.s_ah[sbuf] +
                        (i * c->L.n_col_tiles + col) * TB);
}
static inline HVX_Vector *d_tile(const attn_ctx *c, uint32_t i) {
  return (HVX_Vector *)(c->vb + c->L.d_ah + i * TB);
}
static inline HVX_Vector *o_tile(const attn_ctx *c, uint32_t buf, uint32_t i,
                                 uint32_t d) {
  return (HVX_Vector *)(c->vb + c->L.o_wh[buf] +
                        (i * c->L.n_dot_tiles + d) * TB);
}
static inline uint16_t *k_land(const attn_ctx *c, uint32_t buf) {
  return (uint16_t *)(c->vb + c->L.k_land[buf]);
}
static inline uint16_t *v_land(const attn_ctx *c, uint32_t buf) {
  return (uint16_t *)(c->vb + c->L.v_land[buf]);
}

static inline uint64_t now_us(void) { return HAP_perf_get_time_us(); }

/** @brief [lo, hi) slice of @a n units for worker @a i of @a n_threads. */
static inline void unit_range(uint32_t n, uint32_t n_threads, uint32_t i,
                              uint32_t *lo, uint32_t *hi) {
  *lo = (uint32_t)((uint64_t)n * i / n_threads);
  *hi = (uint32_t)((uint64_t)n * (i + 1) / n_threads);
}

/* --- phases ---------------------------------------------------------- */

/**
 * @brief Q rows of (kv head n, query block qb) -> scaled fp16 AH tiles, and
 *        the per-row softmax state reset.
 *
 * Tile row r is query row qb*br + r/G, head n*G + r%G; rows past the real
 * ones are zero, which makes their scores 0, their probabilities uniform
 * and harmless, and they are skipped at the store.
 */
static void phase_qprep(attn_ctx *c, uint32_t n, uint32_t qb) {
  const uint32_t hd = c->s->head_dim;
  const float *rows[32];
  for (uint32_t i = 0; i < c->L.n_row_tiles; ++i) {
    for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
      for (uint32_t rr = 0; rr < 32u; ++rr) {
        uint32_t q, g;
        if (hexkl_attn_f16_tile_row_to_qg(c->s, c->t, qb, 32u * i + rr, &q,
                                          &g)) {
          rows[rr] = c->io->q + (size_t)q * c->io->q_stride +
                     (size_t)(n * c->t->g + g) * hd + 32u * d;
        } else {
          rows[rr] = NULL;
        }
      }
      hvx_tile_f16_rows_f32_gather_to_tile(q_tile(c, i, d), rows, c->scale);
    }
  }

  // Softmax state: nothing seen (m = -inf, l = 0), or with a sink, the
  // sink already seen: m = sink*log2e and its own term 2^(m-m) = 1 in l.
  for (uint32_t r = 0; r < c->t->g_br; ++r) {
    uint32_t q, g;
    if (c->io->sinks &&
        hexkl_attn_f16_tile_row_to_qg(c->s, c->t, qb, r, &q, &g)) {
      c->m_hf[r] = hvx_attn_f32_to_hf(c->io->sinks[n * c->t->g + g] * LOG2E);
      c->l_f32[r] = 1.0f;
    } else {
      c->m_hf[r] = (uint16_t)HVX_ATTN_HF_NEG_INF;
      c->l_f32[r] = 0.0f;
    }
  }
}

/**
 * @brief Number of the registry's column tiles that block kb actually
 *        covers (the last block may run past max_rows).
 */
static inline uint32_t resident_cols(const attn_ctx *c, uint32_t kb) {
  const uint32_t ct = c->L.n_col_tiles;
  const uint32_t c0 = kb * ct;
  if (c0 >= c->io->kv_tile_cols) {
    return 0;
  }
  const uint32_t left = c->io->kv_tile_cols - c0;
  return left < ct ? left : ct;
}

/**
 * @brief Queues the DMA of cache block kb of kv head n. Raw path: rows
 *        into the landing buffer @a buf. Resident path: ready tiles
 *        straight into the K^T / V tile buffers @a buf. Does not wait.
 */
static void dma_issue(attn_ctx *c, uint32_t n, uint32_t kb, uint32_t buf) {
  const uint32_t hd = c->s->head_dim;
  const uint32_t bc = c->t->bc;
  const uint32_t k0 = kb * bc;

  if (c->resident) {
    const uint32_t ncol = resident_cols(c, kb);
    if (ncol == 0) {
      return;
    }
    const uint32_t dt = c->L.n_dot_tiles;
    const size_t src =
      ((size_t)n * c->io->kv_tile_cols + (size_t)kb * c->L.n_col_tiles) * dt *
      TB;
    // 2 KiB rows: the DMA regime the prior branch measured at ~5x the
    // bandwidth of the 256 B rows the raw path moves.
    hexkl_dma_ring_push2d(kt_tile(c, buf, 0, 0), c->io->kt_tiles + src, TB, TB,
                          TB, ncol * dt, /*src_vtcm=*/0, /*dst_vtcm=*/1);
    hexkl_dma_ring_push2d(v_tile(c, buf, 0, 0), c->io->v_tiles + src, TB, TB,
                          TB, ncol * dt, /*src_vtcm=*/0, /*dst_vtcm=*/1);
    return;
  }

  uint32_t nrows = c->s->cache_to - k0;
  if (nrows > bc) {
    nrows = bc;
  }
  const size_t src_off = (size_t)k0 * c->io->kv_stride + (size_t)n * hd;
  const uint32_t row_bytes = hd * 2u;
  const uint32_t src_stride = c->io->kv_stride * 2u;

  hexkl_dma_ring_push2d(k_land(c, buf), c->io->k + src_off, row_bytes,
                        src_stride, row_bytes, nrows, /*src_vtcm=*/0,
                        /*dst_vtcm=*/1);
  hexkl_dma_ring_push2d(v_land(c, buf), c->io->v + src_off, row_bytes,
                        src_stride, row_bytes, nrows, /*src_vtcm=*/0,
                        /*dst_vtcm=*/1);
}

/**
 * @brief Waits for the queued DMA and zeroes whatever the block covers
 *        beyond the cache, so nothing non-finite reaches HMX (a masked P
 *        lane is exactly 0, but 0 * NaN is still NaN). Raw path: rows past
 *        cache_to. Resident path: tiles past the registry's last column.
 */
static void dma_land(attn_ctx *c, uint32_t kb, uint32_t buf) {
  const uint32_t hd = c->s->head_dim;
  const uint32_t bc = c->t->bc;
  const uint32_t k0 = kb * bc;
  hexkl_dma_ring_drain();

  if (c->resident) {
    const uint32_t ncol = resident_cols(c, kb);
    const uint32_t ct = c->L.n_col_tiles;
    if (ncol < ct) {
      const size_t bytes = (size_t)(ct - ncol) * c->L.n_dot_tiles * TB;
      memset(kt_tile(c, buf, 0, ncol), 0, bytes);
      memset(v_tile(c, buf, ncol, 0), 0, bytes);
    }
    return;
  }

  uint32_t nrows = c->s->cache_to - k0;
  if (nrows > bc) {
    nrows = bc;
  }
  if (nrows < bc) {
    const uint32_t row_bytes = hd * 2u;
    memset(k_land(c, buf) + (size_t)nrows * hd, 0,
           (size_t)(bc - nrows) * row_bytes);
    memset(v_land(c, buf) + (size_t)nrows * hd, 0,
           (size_t)(bc - nrows) * row_bytes);
  }
}

/** @brief One (col, d) pair of landing rows -> its K^T tile and V tile. */
static void tile_one(attn_ctx *c, uint32_t buf, uint32_t col, uint32_t d) {
  const uint32_t hd = c->s->head_dim;
  const size_t src = (size_t)(32u * col) * hd + 32u * d;
  hvx_tile_f16_rows_to_tile_transposed(kt_tile(c, buf, d, col),
                                       k_land(c, buf) + src, hd);
  hvx_tile_f16_rows_to_tile(v_tile(c, buf, col, d), v_land(c, buf) + src, hd);
}

static void tile_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  attn_ctx *c = (attn_ctx *)ctx_;
  const uint32_t n = c->L.n_col_tiles * c->L.n_dot_tiles;
  uint32_t lo, hi;
  unit_range(n, n_threads, i, &lo, &hi);
  for (uint32_t u = lo; u < hi; ++u) {
    tile_one(c, c->w_buf, u / c->L.n_dot_tiles, u % c->L.n_dot_tiles);
  }
}

/** @brief Landing rows -> K^T and V tiles, across the pool (prologue only:
 *         inside the pipeline the pool is busy with the softmax). No-op on
 *         the resident path, whose tiles arrive ready. */
static void tile_block_pool(attn_ctx *c, uint32_t buf) {
  if (c->resident) {
    return;
  }
  c->w_buf = buf;
  const uint32_t n = c->L.n_col_tiles * c->L.n_dot_tiles;
  hvx_worker_pool_run(n >= TILE_POOL_MIN_UNITS ? c->pool : NULL, tile_worker, c,
                      n);
}

/** @brief Same, on the calling thread only. */
static void tile_block_inline(attn_ctx *c, uint32_t buf) {
  if (c->resident) {
    return;
  }
  for (uint32_t col = 0; col < c->L.n_col_tiles; ++col) {
    for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
      tile_one(c, buf, col, d);
    }
  }
}

/** @brief S[sbuf] = Q.K^T for the block whose tiles are in @a buf. */
static int phase_qk(attn_ctx *c, uint32_t buf, uint32_t sbuf) {
  for (uint32_t i = 0; i < c->L.n_row_tiles; ++i) {
    for (uint32_t col = 0; col < c->L.n_col_tiles; ++col) {
      hexkl_micro_hmx_acc_clear_f16();
      for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
        int rc = hexkl_micro_hmx_mm_f16(c->vb, off_of(c, q_tile(c, i, d)),
                                        off_of(c, kt_tile(c, buf, d, col)));
        if (rc != AEE_SUCCESS) {
          return rc;
        }
      }
      int rc = hexkl_micro_hmx_acc_read_f16(c->vb, c->L.cfg_f16,
                                            off_of(c, s_tile(c, sbuf, i, col)));
      if (rc != AEE_SUCCESS) {
        return rc;
      }
    }
  }
  return AEE_SUCCESS;
}

/**
 * @brief Online softmax for one row pair (tile vector v of row tile i)
 *        across this block's column tiles, in place -> P, plus the row
 *        state and this block's rescale factor for the two rows.
 *
 * Pass 1 applies the softcap, masks, and finds the row max; pass 2
 * exponentiates and sums.
 *
 * Rows that have seen nothing visible yet keep m = -inf. Their exponent
 * base has to be a finite stand-in (0 here) so -inf - m does not become
 * NaN; every lane of such a row is -inf and exponentiates to exactly 0,
 * and its rescale 2^(m - 0) is 0 against an O that is still 0. Once a
 * visible lane appears m becomes finite and the ordinary path takes over.
 * This happens whenever a query block straddles a cache-block boundary --
 * the first rows of the block see nothing in the last cache block.
 */
static void softmax_row_pair(attn_ctx *c, uint32_t i, uint32_t v) {
  const HVX_Vector neg_inf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_NEG_INF);
  const HVX_Vector one_hf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_ONE);
  const HVX_Vector zero = Q6_V_vzero();
  const uint32_t ct = c->L.n_col_tiles;
  const uint32_t sbuf = c->w_sbuf;
  const uint32_t k0 = c->w_kb * c->t->bc;
  const int masked = c->w_masked;
  const uint32_t r0 = 32u * i + 2u * v;
  const uint32_t r1 = r0 + 1u;

  HVX_Vector lo_pair = zero, hi_pair = zero;
  if (masked) {
    // Pad rows get [0, 65535): everything visible, nothing to mask.
    uint32_t lo[2] = {0, 0}, hi[2] = {0xFFFFu, 0xFFFFu};
    for (uint32_t p = 0; p < 2u; ++p) {
      uint32_t q, g;
      if (hexkl_attn_f16_tile_row_to_qg(c->s, c->t, c->w_qb, r0 + p, &q, &g)) {
        hexkl_attn_f16_row_range(c->s, q, &lo[p], &hi[p]);
      }
    }
    lo_pair = hvx_attn_splat_pair_hf(lo[0], lo[1]);
    hi_pair = hvx_attn_splat_pair_hf(hi[0], hi[1]);
  }

  // Pass 1: softcap and mask in place, row max.
  HVX_Vector mx = neg_inf;
  for (uint32_t col = 0; col < ct; ++col) {
    HVX_Vector *sp = s_tile(c, sbuf, i, col) + v;
    HVX_Vector sv = *sp;
    int dirty = 0;
    if (c->softcap_on) {
      sv = hvx_attn_softcap_hf(sv, c->cap_hf, c->inv_cap_hf);
      dirty = 1;
    }
    if (masked) {
      const HVX_VectorPred vis =
        hvx_attn_visible(c->col_idx, k0 + 32u * col, lo_pair, hi_pair);
      sv = Q6_V_vmux_QVV(vis, sv, neg_inf);
      dirty = 1;
    }
    if (dirty) {
      *sp = sv;
    }
    mx = Q6_Vhf_vmax_VhfVhf(mx, sv);
  }
  mx = hvx_attn_pairmax_hf(mx);

  const HVX_Vector m_old = hvx_attn_splat_pair_hf(c->m_hf[r0], c->m_hf[r1]);
  const HVX_Vector m_new = Q6_Vhf_vmax_VhfVhf(m_old, mx);
  const HVX_VectorPred unseen = Q6_Q_vcmp_eq_VhVh(m_new, neg_inf);
  const HVX_Vector m_use = Q6_V_vmux_QVV(unseen, zero, m_new);
  const HVX_Vector a = hvx_attn_exp2_hf(Q6_Vhf_vsub_VhfVhf(m_old, m_use));

  const uint32_t mw = hvx_attn_word0(m_new);
  c->m_hf[r0] = (uint16_t)(mw & 0xFFFFu);
  c->m_hf[r1] = (uint16_t)(mw >> 16);
  const uint32_t aw = hvx_attn_word0(a);
  c->a_hf[r0] = (uint16_t)(aw & 0xFFFFu);
  c->a_hf[r1] = (uint16_t)(aw >> 16);

  // Pass 2: P = 2^(S - m'), row sums by parity through the widening
  // multiply (even lanes -> low half -> row r0).
  HVX_Vector sum0 = zero, sum1 = zero;
  for (uint32_t col = 0; col < ct; ++col) {
    HVX_Vector *sp = s_tile(c, sbuf, i, col) + v;
    const HVX_Vector p = hvx_attn_exp2_hf(Q6_Vhf_vsub_VhfVhf(*sp, m_use));
    *sp = p;
    const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(p, one_hf);
    sum0 = Q6_Vqf32_vadd_Vqf32Vqf32(sum0, Q6_V_lo_W(w));
    sum1 = Q6_Vqf32_vadd_Vqf32Vqf32(sum1, Q6_V_hi_W(w));
  }
  const float add0 =
    hvx_attn_lane0_f32(hvx_attn_sum32_sf(Q6_Vsf_equals_Vqf32(sum0)));
  const float add1 =
    hvx_attn_lane0_f32(hvx_attn_sum32_sf(Q6_Vsf_equals_Vqf32(sum1)));
  c->l_f32[r0] = c->l_f32[r0] * hvx_attn_hf_to_f32(c->a_hf[r0]) + add0;
  c->l_f32[r1] = c->l_f32[r1] * hvx_attn_hf_to_f32(c->a_hf[r1]) + add1;
}

static void softmax_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  attn_ctx *c = (attn_ctx *)ctx_;
  const uint32_t n = c->L.n_row_tiles * HVX_TILE_F16_VECS;
  uint32_t lo, hi;
  unit_range(n, n_threads, i, &lo, &hi);
  for (uint32_t u = lo; u < hi; ++u) {
    softmax_row_pair(c, u / HVX_TILE_F16_VECS, u % HVX_TILE_F16_VECS);
  }
}

/**
 * @brief Starts the softmax of block kb (scores in S[sbuf]) on the pool
 *        workers. Returns what hvx_worker_pool_submit returned: 1 if it is
 *        running asynchronously, 0 if it already completed inline.
 */
static int softmax_start(attn_ctx *c, uint32_t qb, uint32_t kb, uint32_t sbuf,
                         int masked) {
  c->w_qb = qb;
  c->w_kb = kb;
  c->w_sbuf = sbuf;
  c->w_masked = masked;
  const uint32_t n = c->L.n_row_tiles * HVX_TILE_F16_VECS;
  return hvx_worker_pool_submit(c->pool, softmax_worker, c, n);
}

/** @brief Joins the softmax and builds the D tiles from its rescale factors. */
static void softmax_finish(attn_ctx *c) {
  hvx_worker_pool_wait(c->pool);
  for (uint32_t i = 0; i < c->L.n_row_tiles; ++i) {
    hvx_tile_f16_diag(d_tile(c, i), c->a_hf + 32u * i);
  }
}

/**
 * @brief O_new = D.O_prev + P.V, one accumulator pass per output tile.
 *
 * The D term is skipped on the first visited block, where O_prev does not
 * exist yet -- equivalent to O_prev = 0 without having to zero a buffer.
 */
static int phase_pv(attn_ctx *c, uint32_t buf, uint32_t sbuf, uint32_t o_prev,
                    uint32_t o_cur, int first) {
  for (uint32_t i = 0; i < c->L.n_row_tiles; ++i) {
    for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
      hexkl_micro_hmx_acc_clear_f16();
      int rc;
      if (!first) {
        rc = hexkl_micro_hmx_mm_f16(c->vb, off_of(c, d_tile(c, i)),
                                    off_of(c, o_tile(c, o_prev, i, d)));
        if (rc != AEE_SUCCESS) {
          return rc;
        }
      }
      for (uint32_t col = 0; col < c->L.n_col_tiles; ++col) {
        rc = hexkl_micro_hmx_mm_f16(c->vb, off_of(c, s_tile(c, sbuf, i, col)),
                                    off_of(c, v_tile(c, buf, col, d)));
        if (rc != AEE_SUCCESS) {
          return rc;
        }
      }
      rc = hexkl_micro_hmx_acc_read_f16(c->vb, c->L.cfg_f16,
                                        off_of(c, o_tile(c, o_cur, i, d)));
      if (rc != AEE_SUCCESS) {
        return rc;
      }
    }
  }
  return AEE_SUCCESS;
}

/**
 * @brief out rows = O / l, f32.
 *
 * The widening fp16 -> qf32 multiply de-interleaves a tile vector into its
 * even row (low half) and odd row (high half) in column order, so each
 * row is one 32-lane f32 vector: scale by 1/l and store to its DDR row.
 */
static void phase_store(attn_ctx *c, uint32_t n, uint32_t qb, uint32_t o_buf) {
  const HVX_Vector one_hf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_ONE);
  const uint32_t hd = c->s->head_dim;
  for (uint32_t i = 0; i < c->L.n_row_tiles; ++i) {
    for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
      const HVX_Vector *tile = o_tile(c, o_buf, i, d);
      for (uint32_t v = 0; v < HVX_TILE_F16_VECS; ++v) {
        const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(tile[v], one_hf);
        for (uint32_t p = 0; p < 2u; ++p) {
          const uint32_t r = 32u * i + 2u * v + p;
          uint32_t q, g;
          if (!hexkl_attn_f16_tile_row_to_qg(c->s, c->t, qb, r, &q, &g)) {
            continue;
          }
          const float inv_l = 1.0f / c->l_f32[r];
          int bits;
          memcpy(&bits, &inv_l, sizeof(bits));
          const HVX_Vector row =
            Q6_Vsf_equals_Vqf32(p ? Q6_V_hi_W(w) : Q6_V_lo_W(w));
          float *dst = c->io->out + (size_t)q * c->io->out_stride +
                       (size_t)(n * c->t->g + g) * hd + 32u * d;
          hvx_tile_store_u(dst, Q6_Vsf_vmpy_VsfVsf(row, Q6_V_vsplat_R(bits)));
        }
      }
    }
  }
}

/* --- the block pipeline for one (kv head, query block) ---------------- */

static int run_q_block(attn_ctx *c, uint32_t n, uint32_t qb) {
  hexkl_attn_f16_stats *st = c->st;
  uint32_t kb_lo, kb_hi;
  hexkl_attn_f16_kv_block_range(c->s, c->t, qb, &kb_lo, &kb_hi);
  if (kb_lo >= kb_hi) {
    return AEE_SUCCESS; // cannot happen: every row sees at least itself
  }

  // Prologue: block kb_lo landed, tiled (pool is idle, so use it) and
  // multiplied, so the loop can start with its scores ready.
  uint64_t t0 = now_us();
  dma_issue(c, n, kb_lo, kb_lo & 1u);
  dma_land(c, kb_lo, kb_lo & 1u);
  uint64_t t1 = now_us();
  st->us_dma += t1 - t0;
  tile_block_pool(c, kb_lo & 1u);
  t0 = now_us();
  st->us_tile += t0 - t1;
  int rc = phase_qk(c, kb_lo & 1u, 0);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  t1 = now_us();
  st->us_qk += t1 - t0;

  uint32_t o_prev = 0;
  int first = 1;
  for (uint32_t kb = kb_lo; kb < kb_hi; ++kb) {
    const uint32_t buf = kb & 1u;
    const uint32_t sbuf = (kb - kb_lo) & 1u;
    const uint32_t o_cur = first ? 0u : (o_prev ^ 1u);
    const int has_next = (kb + 1u < kb_hi);

    // 1. Next block's DMA goes out first so it lands under everything below.
    t0 = now_us();
    if (has_next) {
      dma_issue(c, n, kb + 1u, buf ^ 1u);
    }
    // 2. This block's softmax on the workers.
    const int masked = !hexkl_attn_f16_kv_block_unmasked(c->s, c->t, qb, kb);
    softmax_start(c, qb, kb, sbuf, masked);
    t1 = now_us();
    st->us_softmax += t1 - t0; // inline cost when the pool was unavailable

    // 3. Meanwhile, on this thread: land, tile and multiply the next block.
    if (has_next) {
      dma_land(c, kb + 1u, buf ^ 1u);
      uint64_t t2 = now_us();
      st->us_dma += t2 - t1;
      tile_block_inline(c, buf ^ 1u);
      t1 = now_us();
      st->us_tile += t1 - t2;
      rc = phase_qk(c, buf ^ 1u, sbuf ^ 1u);
      if (rc != AEE_SUCCESS) {
        hvx_worker_pool_wait(c->pool);
        return rc;
      }
      t2 = now_us();
      st->us_qk += t2 - t1;
      t1 = t2;
    }

    // 4. Join the softmax (exposed time only), build D, run this block's
    //    P.V.
    softmax_finish(c);
    t0 = now_us();
    st->us_softmax += t0 - t1;

    rc = phase_pv(c, buf, sbuf, o_prev, o_cur, first);
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    st->us_pv += now_us() - t0;

    o_prev = o_cur;
    first = 0;
    st->n_blocks++;
  }

  t0 = now_us();
  phase_store(c, n, qb, o_prev);
  st->us_store += now_us() - t0;
  return AEE_SUCCESS;
}

/* --- entry ----------------------------------------------------------- */

int hexkl_attn_f16_prefill(uint8_t *vtcm_base, uint32_t arena_top,
                           uint32_t hmx_fp16_rate,
                           const hexkl_attn_f16_shape *s,
                           const hexkl_attn_f16_tiling *t,
                           const hexkl_attn_f16_io *io, hvx_worker_pool *pool,
                           hexkl_attn_f16_stats *st) {
  if (!vtcm_base || !s || !t || !io || !io->q || !io->out) {
    return AEE_EBADPARM;
  }
  const int resident = io->kt_tiles != NULL;
  if (resident ? (!io->v_tiles || io->kv_tile_cols * 32u < s->cache_to)
               : (!io->k || !io->v)) {
    return AEE_EBADPARM;
  }
  if (hmx_fp16_rate == 0) {
    return AEE_EUNSUPPORTED;
  }
  // The lane mask compares 16-bit cache positions.
  if (s->cache_to > 0xFFFFu || t->g_br > MAX_G_BR || s->softcap < 0.0f) {
    return AEE_EBADPARM;
  }

  attn_ctx c;
  memset(&c, 0, sizeof(c));
  c.vb = vtcm_base;
  c.s = s;
  c.t = t;
  c.io = io;
  c.pool = pool;
  c.resident = resident;
  int rc = map_plan_rc(
    hexkl_attn_f16_plan(s, t, arena_top, hexkl_micro_hmx_config_size(), &c.L));
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  c.scale = LOG2E / sqrtf((float)s->head_dim);
  c.m_hf = (uint16_t *)(vtcm_base + c.L.ml);
  c.l_f32 = (float *)(vtcm_base + c.L.ml + t->g_br * 4u);
  c.col_idx = hvx_attn_col_index_pairs();
  if (s->softcap > 0.0f) {
    // Scores carry log2e already, so the cap does too: c.tanh(S/c) with
    // c = softcap*log2e equals log2e * softcap*tanh(s/softcap).
    // hvx_attn_softcap_hf wants k = 2*log2e/c for its exponent.
    const float cap = s->softcap * LOG2E;
    c.cap_hf = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(cap));
    c.inv_cap_hf = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(2.0f * LOG2E / cap));
    c.softcap_on = 1;
  }

  rc = hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, c.L.cfg_f16);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  hexkl_dma_ring_reset();

  hexkl_attn_f16_stats local;
  memset(&local, 0, sizeof(local));
  c.st = st ? st : &local;
  memset(c.st, 0, sizeof(*c.st));

  const uint32_t n_qb = hexkl_attn_f16_n_q_blocks(s, t);
  for (uint32_t n = 0; n < s->n_head_kv; ++n) {
    for (uint32_t qb = 0; qb < n_qb; ++qb) {
      const uint64_t t0 = now_us();
      phase_qprep(&c, n, qb);
      c.st->us_qprep += now_us() - t0;
      rc = run_q_block(&c, n, qb);
      if (rc != AEE_SUCCESS) {
        return rc;
      }
    }
  }
  return AEE_SUCCESS;
}
