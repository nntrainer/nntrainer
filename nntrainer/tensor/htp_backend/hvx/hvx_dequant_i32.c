// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_dequant_i32.c
 * @date   03 Aug 2026
 * @brief  int32 accumulator to f32 dequantization for the A8W4 path
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hexkl_acc_tile.h" /* HEXKL_ACC_TILE_COLS, for the width assert */
#include "hvx_convert.h"
#include "hvx_dequant_i32.h"
#include "hvx_swiglu_det.h"

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128u
/** @brief f32 or int32 lanes per HVX vector. */
#define LANES (VLEN / 4u)

void hvx_dequant_i32_to_f32(const int32_t *acc, uint32_t m_valid,
                            uint32_t m_pad, uint32_t n, const float *act_scale,
                            const int32_t *act_zp, const int32_t *colsum_w,
                            const float *w_scale, const float *bias, float *out,
                            int accumulate) {
  (void)m_pad;

  for (uint32_t m = 0; m < m_valid; ++m) {
    const int32_t *arow = acc + (size_t)m * n;
    float *orow = out + (size_t)m * n;
    const float s = act_scale[m];
    const int32_t z = act_zp[m];

    const uint32_t n_vec = n / LANES;
    const HVX_Vector vs = hvx_splat_sf(s);
    const HVX_Vector vzf = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(z));

    const HVX_UVector *vacc = (const HVX_UVector *)arow;
    const HVX_UVector *vcs = (const HVX_UVector *)colsum_w;
    const HVX_UVector *vws = (const HVX_UVector *)w_scale;
    const HVX_UVector *vb = (const HVX_UVector *)bias;
    HVX_UVector *vout = (HVX_UVector *)orow;

    for (uint32_t v = 0; v < n_vec; ++v) {
      const HVX_Vector af = Q6_Vsf_equals_Vw(vacc[v]);
      const HVX_Vector csf = Q6_Vsf_equals_Vw(vcs[v]);
      const HVX_Vector corrected =
        Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vzf, csf));
      const HVX_Vector scaled =
        Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(corrected, vs), vws[v]);
      const HVX_Vector r = Q6_Vsf_vadd_VsfVsf(scaled, vb[v]);
      vout[v] = accumulate ? Q6_Vsf_vadd_VsfVsf(vout[v], r) : r;
    }

    for (uint32_t j = n_vec * LANES; j < n; ++j) {
      const int32_t corrected = arow[j] - z * colsum_w[j];
      const float r = (float)corrected * s * w_scale[j] + bias[j];
      orow[j] = accumulate ? (orow[j] + r) : r;
    }
  }
}

/**
 * @brief One row's dequantized vector.
 *
 * Written as a macro rather than a loop body so four of them can sit in one
 * iteration with four INDEPENDENT dependency chains. The chain per row is
 * load -> convert -> multiply -> subtract -> multiply -> multiply -> add,
 * about six dependent HVX ops; at one row in flight the unit stalls on
 * latency the whole way and the measured cost was ~36 cycles a row against
 * roughly ten ops of actual work. Rows are independent, so interleaving
 * them fills those slots.
 *
 * The arithmetic is byte for byte what the single-row loop did -- same
 * intrinsics, same order, same operands. Only the scheduling changes, which
 * is why the bit-exact gates stay a valid check on this.
 */
#define DQ_TILE_ROW(i)                                                         \
  const HVX_Vector af##i = Q6_Vsf_equals_Vw(                                   \
    ((const HVX_UVector *)(tile + (size_t)(m + (i)) * row_stride))[0]);        \
  const HVX_Vector vs##i = hvx_splat_sf(act_scale[m + (i)]);                   \
  const HVX_Vector vz##i = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(act_zp[m + (i)]));   \
  const HVX_Vector r##i = Q6_Vsf_vadd_VsfVsf(                                  \
    Q6_Vsf_vmpy_VsfVsf(                                                        \
      Q6_Vsf_vmpy_VsfVsf(                                                      \
        Q6_Vsf_vsub_VsfVsf(af##i, Q6_Vsf_vmpy_VsfVsf(vz##i, csf)), vs##i),     \
      vw),                                                                     \
    vbias)

/** @brief Stores (or accumulates) row @a i of the unrolled group. */
#define DQ_TILE_STORE(i)                                                       \
  do {                                                                         \
    HVX_UVector *vo = (HVX_UVector *)(out + (size_t)(m + (i)) * out_stride);   \
    vo[0] = accumulate ? Q6_Vsf_vadd_VsfVsf(vo[0], r##i) : r##i;               \
  } while (0)

void hvx_dequant_acc_tile_to_f32(const int32_t *tile, uint32_t row_stride,
                                 uint32_t m_count, const float *act_scale,
                                 const int32_t *act_zp, const int32_t *colsum_w,
                                 const float *w_scale, const float *bias,
                                 float *out, uint32_t out_stride,
                                 int accumulate) {
  /** One vector per tile row, so the loop above's n_vec/tail split collapses.
     If a future part changes the tile width this stops being true, and the
     build should stop with it rather than silently emit 32 of n columns. */
  _Static_assert(HEXKL_ACC_TILE_COLS == LANES,
                 "an accumulator tile row must be exactly one HVX vector");

  /** Loop-invariant: the column tile is fixed for this call, so all three are
     the same values the row loop would reload and recompute every row. */
  const HVX_Vector csf = Q6_Vsf_equals_Vw(((const HVX_UVector *)colsum_w)[0]);
  const HVX_Vector vw = ((const HVX_UVector *)w_scale)[0];
  const HVX_Vector vbias = ((const HVX_UVector *)bias)[0];

  uint32_t m = 0;
  for (; m + 4u <= m_count; m += 4u) {
    DQ_TILE_ROW(0);
    DQ_TILE_ROW(1);
    DQ_TILE_ROW(2);
    DQ_TILE_ROW(3);
    DQ_TILE_STORE(0);
    DQ_TILE_STORE(1);
    DQ_TILE_STORE(2);
    DQ_TILE_STORE(3);
  }
  for (; m < m_count; ++m) {
    DQ_TILE_ROW(0);
    DQ_TILE_STORE(0);
  }
}

void hvx_dequant_acc_tiles_scatter_rows_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_tiles,
  uint32_t row_stride, uint32_t m0, uint32_t m1, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, float *out, uint32_t out_stride, const uint32_t *rows,
  const float *weights) {
  /* One output row's n_tiles vectors, [47:32] stride, [31:16] width, [15:0]
     height. A hint: it moves no bits. Three rows ahead, the most a lane
     can queue (the hardware stalls the thread on a fourth). */
  const uint64_t box = ((uint64_t)(n_tiles * 128u) << 32) |
                       ((uint64_t)(n_tiles * 128u) << 16) | 1u;
  for (uint32_t m = m0; m < m1 && m < m0 + 3u; ++m)
    Q6_l2fetch_AP((void *)(out + (size_t)rows[m] * out_stride), box);
  for (uint32_t m = m0; m < m1; ++m) {
    if (m + 3u < m1)
      Q6_l2fetch_AP((void *)(out + (size_t)rows[m + 3u] * out_stride), box);
    const HVX_Vector vs = hvx_splat_sf(act_scale[m]);
    const HVX_Vector vz = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(act_zp[m]));
    const HVX_Vector vwt = hvx_splat_sf(weights[m]);
    HVX_UVector *vo = (HVX_UVector *)(out + (size_t)rows[m] * out_stride);
    for (uint32_t j = 0; j < n_tiles; ++j) {
      const int32_t *tile =
        (const int32_t *)(tiles_base + (size_t)j * tile_stride);
      const HVX_Vector af = Q6_Vsf_equals_Vw(
        ((const HVX_UVector *)(tile + (size_t)m * row_stride))[0]);
      const HVX_Vector csf =
        Q6_Vsf_equals_Vw(((const HVX_UVector *)colsum_w)[j]);
      const HVX_Vector r = Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_vmpy_VsfVsf(
          Q6_Vsf_vmpy_VsfVsf(
            Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vz, csf)), vs),
          ((const HVX_UVector *)w_scale)[j]),
        ((const HVX_UVector *)bias)[j]);
      vo[j] = Q6_Vsf_vadd_VsfVsf(vo[j], Q6_Vsf_vmpy_VsfVsf(r, vwt));
    }
  }
}

#undef DQ_TILE_ROW
#undef DQ_TILE_STORE

/* ---- pooled dequant over a run of staged tiles --------------------------- */

void hvx_dq_tiles_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  const hvx_dq_tiles_job *c = (const hvx_dq_tiles_job *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->n_tiles * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->n_tiles * (i + 1) / n_threads);

  for (uint32_t j = lo; j < hi; ++j) {
    const uint32_t c0 = (c->nt0 + j) * HEXKL_ACC_TILE_COLS;
    const int32_t *tile =
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
    /* Which half of a gate_up result this tile belongs to. A caller with one
       destination passes a split past the last column, so this always takes
       dst_a and never reads dst_b. */
    float *out =
      (c0 < c->split) ? (c->dst_a + c0) : (c->dst_b + (c0 - c->split));
    hvx_dequant_acc_tile_to_f32(tile, c->row_stride, c->m_count, c->act_scale,
                                c->act_zp, c->colsum_w + c0, c->w_scale + c0,
                                c->bias + c0, out, c->dst_stride,
                                /*accumulate=*/0);
  }
}

/** @brief DQ_TILE_ROW's operations, in its order, on one row's int32 (as
 *         f32) of output tile j: (a - zp cs) s ws + b. */
static inline HVX_Vector dq_formula(HVX_Vector af, HVX_Vector vs, HVX_Vector vz,
                                    HVX_Vector cs, HVX_Vector ws,
                                    HVX_Vector bs) {
  return Q6_Vsf_vadd_VsfVsf(
    Q6_Vsf_vmpy_VsfVsf(
      Q6_Vsf_vmpy_VsfVsf(
        Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vz, Q6_Vsf_equals_Vw(cs))),
        vs),
      ws),
    bs);
}

/** @brief Stores one output vector: aligned when @a al (the rows worker's
 *         destination check), else unaligned. On v79 the aligned store
 *         takes the rows worker from 1.59 to 1.28 us a 64 x 32 tile; most
 *         of the rest is the f32 rows going to DDR. */
static inline void dq_store(float *p, HVX_Vector v, int al) {
  if (al) {
    *(HVX_Vector *)p = v;
  } else {
    *(HVX_UVector *)p = v;
  }
}

/** @brief True when every row the job writes starts on a vector boundary. */
static inline int dq_dst_aligned(const hvx_dq_tiles_job *c) {
  return ((uintptr_t)(c->dst_a + c->nt0 * HEXKL_ACC_TILE_COLS) & 127u) == 0u &&
         (c->dst_stride % HEXKL_ACC_TILE_COLS) == 0u;
}

/** @brief Rows 4g..4g+3 of a tile staged as three byte planes
 *         (hvx_dq_tiles_job.planes24) as four vectors of 32 int32: one
 *         aligned vector of each plane holds the four rows, b0 + 256 b1 by
 *         a byte shuffle, sext(b2) by an unpack, the halfwords to words by
 *         a halfword shuffle -- hexkl_attn_q2.c's planes_to_f32. */
static inline void planes_rows4(const uint8_t *tile, uint32_t g,
                                HVX_Vector out[4]) {
  const uint8_t *p = tile + (size_t)g * 128u;
  const HVX_VectorPair h01 = Q6_W_vshuff_VVR(
    *(const HVX_Vector *)(p + HVX_DQ_PLANE_BYTES), *(const HVX_Vector *)p, -1);
  const HVX_VectorPair h2 =
    Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(p + 2u * HVX_DQ_PLANE_BYTES));
  const HVX_VectorPair w01 = Q6_W_vshuff_VVR(Q6_V_lo_W(h2), Q6_V_lo_W(h01), -2);
  const HVX_VectorPair w23 = Q6_W_vshuff_VVR(Q6_V_hi_W(h2), Q6_V_hi_W(h01), -2);
  out[0] = Q6_V_lo_W(w01);
  out[1] = Q6_V_hi_W(w01);
  out[2] = Q6_V_lo_W(w23);
  out[3] = Q6_V_hi_W(w23);
}

/** @brief hvx_dq_tiles_rows_worker over byte-plane tiles: the rows in
 *         groups of four, the same floats per row. */
static void dq_rows_planes(const hvx_dq_rows_job *jb, uint32_t n_threads,
                           uint32_t i) {
  const hvx_dq_tiles_job *c = &jb->t;
  const uint32_t n_g = (c->m_count + 3u) / 4u;
  const uint32_t g_lo = (uint32_t)((uint64_t)n_g * i / n_threads);
  const uint32_t g_hi = (uint32_t)((uint64_t)n_g * (i + 1) / n_threads);
  const uint32_t c0 = c->nt0 * HEXKL_ACC_TILE_COLS;
  const HVX_UVector *cs = (const HVX_UVector *)(c->colsum_w + c0);
  const HVX_UVector *ws = (const HVX_UVector *)(c->w_scale + c0);
  const HVX_UVector *bs = (const HVX_UVector *)(c->bias + c0);
  const HVX_Vector v256 = hvx_splat_sf(256.0f);
  const int al = dq_dst_aligned(c);

  for (uint32_t g = g_lo; g < g_hi; ++g) {
    const uint32_t m0 = 4u * g;
    const uint32_t rows = c->m_count - m0 < 4u ? c->m_count - m0 : 4u;
    HVX_Vector vs[4], vz[4];
    for (uint32_t r = 0; r < rows; ++r) {
      vs[r] = hvx_splat_sf(c->act_scale[m0 + r]);
      vz[r] = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->act_zp[m0 + r]));
    }
    for (uint32_t j = 0; j < c->n_tiles; ++j) {
      const uint8_t *tile = c->tiles_base + (size_t)j * c->tile_stride;
      HVX_Vector a[4], b[4];
      planes_rows4(tile, g, a);
      if (c->lo_off) {
        planes_rows4(tile + c->lo_off, g, b);
      }
      for (uint32_t r = 0; r < rows; ++r) {
        HVX_Vector af = Q6_Vsf_equals_Vw(a[r]);
        if (c->lo_off) { /* 256 hi + lo, in f32 */
          af = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(af, v256),
                                  Q6_Vsf_equals_Vw(b[r]));
        }
        dq_store(c->dst_a + (size_t)(m0 + r) * c->dst_stride + c0 + 32u * j,
                 dq_formula(af, vs[r], vz[r], cs[j], ws[j], bs[j]), al);
      }
    }
    if (jb->row_done) {
      for (uint32_t r = 0; r < rows; ++r) {
        jb->row_done(jb->ctx, c->dst_a + (size_t)(m0 + r) * c->dst_stride + c0,
                     m0 + r);
      }
    }
  }
}

void hvx_dq_tiles_rows_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_dq_rows_job *jb = (const hvx_dq_rows_job *)vjob;
  const hvx_dq_tiles_job *c = &jb->t;
  if (c->planes24) {
    dq_rows_planes(jb, n_threads, i);
    return;
  }
  const uint32_t lo = (uint32_t)((uint64_t)c->m_count * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->m_count * (i + 1) / n_threads);
  const uint32_t c0 = c->nt0 * HEXKL_ACC_TILE_COLS;
  const HVX_UVector *cs = (const HVX_UVector *)(c->colsum_w + c0);
  const HVX_UVector *ws = (const HVX_UVector *)(c->w_scale + c0);
  const HVX_UVector *bs = (const HVX_UVector *)(c->bias + c0);
  const HVX_Vector v256 = hvx_splat_sf(256.0f);
  const int al = dq_dst_aligned(c);

  for (uint32_t m = lo; m < hi; ++m) {
    const HVX_Vector vs = hvx_splat_sf(c->act_scale[m]);
    const HVX_Vector vz = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->act_zp[m]));
    float *seg = c->dst_a + (size_t)m * c->dst_stride + c0;
    for (uint32_t j = 0; j < c->n_tiles; ++j) {
      const int32_t *tile =
        (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
      HVX_Vector af = Q6_Vsf_equals_Vw(
        ((const HVX_UVector *)(tile + (size_t)m * c->row_stride))[0]);
      if (c->lo_off) { /* 256 hi + lo: past int32 at K >~ 1024, so in f32 */
        const int32_t *tlo =
          (const int32_t *)((const uint8_t *)tile + c->lo_off);
        af = Q6_Vsf_vadd_VsfVsf(
          Q6_Vsf_vmpy_VsfVsf(af, v256),
          Q6_Vsf_equals_Vw(
            ((const HVX_UVector *)(tlo + (size_t)m * c->row_stride))[0]));
      }
      dq_store(seg + 32u * j, dq_formula(af, vs, vz, cs[j], ws[j], bs[j]), al);
    }
    if (jb->row_done) {
      jb->row_done(jb->ctx, seg, m);
    }
  }
}

void hvx_dequant_acc_tiles_to_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_tiles,
  uint32_t nt0, uint32_t row_stride, uint32_t m_count, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, float *dst_a, float *dst_b, uint32_t split,
  uint32_t dst_stride, hvx_worker_pool *pool) {
  if (!tiles_base || n_tiles == 0u || m_count == 0u) {
    return;
  }
  hvx_dq_tiles_job c = {tiles_base, tile_stride, nt0,      row_stride, m_count,
                        act_scale,  act_zp,      colsum_w, w_scale,    bias,
                        dst_a,      dst_b,       split,    dst_stride, n_tiles,
                        0u,         0u};
  hvx_worker_pool_run(pool, hvx_dq_tiles_worker, &c, n_tiles);
}

/* ---- fused dequant + SwiGLU over gate/up tile pairs ---------------------- */

/**
 * @brief One row of one tile, dequantized -- DQ_TILE_ROW's operations in
 *        the same order, as a function so two of them can feed SwiGLU.
 */
static inline HVX_Vector dq_row_sf(const int32_t *row, float act_scale,
                                   int32_t act_zp, HVX_Vector csf,
                                   HVX_Vector vw, HVX_Vector vbias) {
  const HVX_Vector af = Q6_Vsf_equals_Vw(((const HVX_UVector *)row)[0]);
  const HVX_Vector vs = hvx_splat_sf(act_scale);
  const HVX_Vector vz = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(act_zp));
  return Q6_Vsf_vadd_VsfVsf(
    Q6_Vsf_vmpy_VsfVsf(
      Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vz, csf)),
                         vs),
      vw),
    vbias);
}

void hvx_dq_swiglu_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  const hvx_dq_swiglu_job *c = (const hvx_dq_swiglu_job *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->n_pairs * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->n_pairs * (i + 1) / n_threads);

  for (uint32_t j = lo; j < hi; ++j) {
    const uint32_t cg = (c->g0 + j) * HEXKL_ACC_TILE_COLS; /* gate column */
    const uint32_t cu = c->inter + cg;                     /* its up column */
    const int32_t *gt =
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
    const int32_t *ut =
      (const int32_t *)(c->tiles_base +
                        (size_t)(c->n_pairs + j) * c->tile_stride);
    /* Loop-invariant per tile, exactly as hvx_dequant_acc_tile_to_f32
       hoists them. */
    const HVX_Vector csg =
      Q6_Vsf_equals_Vw(((const HVX_UVector *)(c->colsum_w + cg))[0]);
    const HVX_Vector vwg = ((const HVX_UVector *)(c->w_scale + cg))[0];
    const HVX_Vector vbg = ((const HVX_UVector *)(c->bias + cg))[0];
    const HVX_Vector csu =
      Q6_Vsf_equals_Vw(((const HVX_UVector *)(c->colsum_w + cu))[0]);
    const HVX_Vector vwu = ((const HVX_UVector *)(c->w_scale + cu))[0];
    const HVX_Vector vbu = ((const HVX_UVector *)(c->bias + cu))[0];

    /* Four rows an iteration: the GLU is ~50 dependent HVX ops a row, so
       one row in flight stalls on latency the whole way (DQ_TILE_ROW's
       reason); four independent chains in one block let them interleave.
       Each row's operations are unchanged. */
#define DQ_GLU_ROW(i)                                                          \
  const HVX_Vector g##i =                                                      \
    dq_row_sf(gt + (size_t)(m + (i)) * c->row_stride, c->act_scale[m + (i)],   \
              c->act_zp[m + (i)], csg, vwg, vbg);                              \
  const HVX_Vector u##i =                                                      \
    dq_row_sf(ut + (size_t)(m + (i)) * c->row_stride, c->act_scale[m + (i)],   \
              c->act_zp[m + (i)], csu, vwu, vbu)
#define DQ_GLU_STORE(i, fn)                                                    \
  ((HVX_UVector *)(c->dst + (size_t)(m + (i)) * c->dst_stride + cg))[0] =      \
    fn(g##i, u##i)
    uint32_t m = 0;
    if (c->act == HVX_GLU_GELU_TANH) {
      for (; m + 4u <= c->m_count; m += 4u) {
        DQ_GLU_ROW(0);
        DQ_GLU_ROW(1);
        DQ_GLU_ROW(2);
        DQ_GLU_ROW(3);
        DQ_GLU_STORE(0, hvx_geglu_det_sf);
        DQ_GLU_STORE(1, hvx_geglu_det_sf);
        DQ_GLU_STORE(2, hvx_geglu_det_sf);
        DQ_GLU_STORE(3, hvx_geglu_det_sf);
      }
      for (; m < c->m_count; ++m) {
        DQ_GLU_ROW(0);
        DQ_GLU_STORE(0, hvx_geglu_det_sf);
      }
    } else {
      for (; m + 4u <= c->m_count; m += 4u) {
        DQ_GLU_ROW(0);
        DQ_GLU_ROW(1);
        DQ_GLU_ROW(2);
        DQ_GLU_ROW(3);
        DQ_GLU_STORE(0, hvx_swiglu_det_sf);
        DQ_GLU_STORE(1, hvx_swiglu_det_sf);
        DQ_GLU_STORE(2, hvx_swiglu_det_sf);
        DQ_GLU_STORE(3, hvx_swiglu_det_sf);
      }
      for (; m < c->m_count; ++m) {
        DQ_GLU_ROW(0);
        DQ_GLU_STORE(0, hvx_swiglu_det_sf);
      }
    }
#undef DQ_GLU_ROW
#undef DQ_GLU_STORE
  }
}

void hvx_dequant_swiglu_acc_tiles_to_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_pairs,
  uint32_t g0, uint32_t row_stride, uint32_t m_count, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, uint32_t inter, uint32_t act, float *dst,
  uint32_t dst_stride, hvx_worker_pool *pool) {
  if (!tiles_base || n_pairs == 0u || m_count == 0u) {
    return;
  }
  hvx_dq_swiglu_job c = {tiles_base, tile_stride, n_pairs,   g0,
                         row_stride, m_count,     act_scale, act_zp,
                         colsum_w,   w_scale,     bias,      inter,
                         dst,        dst_stride,  act};
  hvx_worker_pool_run(pool, hvx_dq_swiglu_worker, &c, n_pairs);
}

/* ---- fused dequant + product over tile pairs of two weights -------------- */

void hvx_dq_mul_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  const hvx_dq_mul_job *c = (const hvx_dq_mul_job *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->n_pairs * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->n_pairs * (i + 1) / n_threads);

  for (uint32_t j = lo; j < hi; ++j) {
    const uint32_t col = c->c0 + j * HEXKL_ACC_TILE_COLS;
    const int32_t *at =
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
    const int32_t *bt =
      (const int32_t *)(c->tiles_base +
                        (size_t)(c->n_pairs + j) * c->tile_stride);
    const HVX_Vector csa =
      Q6_Vsf_equals_Vw(((const HVX_UVector *)(c->colsum_a + col))[0]);
    const HVX_Vector vwa = ((const HVX_UVector *)(c->w_scale_a + col))[0];
    const HVX_Vector vba = ((const HVX_UVector *)(c->bias_a + col))[0];
    const HVX_Vector csb =
      Q6_Vsf_equals_Vw(((const HVX_UVector *)(c->colsum_b + col))[0]);
    const HVX_Vector vwb = ((const HVX_UVector *)(c->w_scale_b + col))[0];
    const HVX_Vector vbb = ((const HVX_UVector *)(c->bias_b + col))[0];

    for (uint32_t m = 0; m < c->m_count; ++m) {
      const HVX_Vector a =
        dq_row_sf(at + (size_t)m * c->row_stride, c->act_scale[m], c->act_zp[m],
                  csa, vwa, vba);
      const HVX_Vector b =
        dq_row_sf(bt + (size_t)m * c->row_stride, c->act_scale[m], c->act_zp[m],
                  csb, vwb, vbb);
      ((HVX_UVector *)(c->dst + (size_t)m * c->dst_stride + col))[0] =
        Q6_Vsf_vmpy_VsfVsf(a, b);
    }
  }
}
