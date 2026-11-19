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

#include <math.h>
#include <stddef.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hexkl_acc_tile.h" /* HEXKL_ACC_TILE_COLS, for the width assert */
#include "hvx_convert.h"
#include "hvx_dequant_i32.h"
#include "hvx_rmsnorm_rows_f32.h"
#include "hvx_rope_rows_f32.h"
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

/** @brief dq_store, or (@a acc) the add into what @a p holds. */
static inline void dq_put(float *p, HVX_Vector v, int al, uint32_t acc) {
  if (acc) {
    v = Q6_Vsf_vadd_VsfVsf(*(const HVX_UVector *)p, v);
  }
  dq_store(p, v, al);
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

/** @brief hvx_dq_rows_job's per-head passes on one written row, @a seg its
 *         first column and @a m its index. */
static void dq_row_post(const hvx_dq_rows_job *jb, float *seg, uint32_t m) {
  const uint32_t n = jb->t.n_tiles * HEXKL_ACC_TILE_COLS;
  if (jb->post_chunk != 0u) {
    hvx_rmsnorm_row_chunks_f32(seg, seg, jb->post_gamma, n, jb->post_chunk,
                               jb->post_eps);
  }
  if (jb->rope_cs) {
    hvx_rope_row_f32(seg, n, jb->rope_hd,
                     jb->rope_cs + (size_t)m * 2u * jb->rope_hd);
  }
}

/** @brief dq_rows_planes's rows m0 .. m0 + rows - 1 (fewer than four, the
 *         job's last group), one row at a time. */
static void dq_planes_tail(const hvx_dq_rows_job *jb, uint32_t g,
                           uint32_t rows) {
  const hvx_dq_tiles_job *c = &jb->t;
  const uint32_t m0 = 4u * g, c0 = c->nt0 * HEXKL_ACC_TILE_COLS;
  const HVX_UVector *cs = (const HVX_UVector *)(c->colsum_w + c0);
  const HVX_UVector *ws = (const HVX_UVector *)(c->w_scale + c0);
  const HVX_UVector *bs = (const HVX_UVector *)(c->bias + c0);
  const int al = dq_dst_aligned(c);
  for (uint32_t j = 0; j < c->n_tiles; ++j) {
    const uint8_t *tile = c->tiles_base + (size_t)j * c->tile_stride;
    HVX_Vector a[4], b[4];
    planes_rows4(tile, g, a);
    if (c->lo_off) {
      planes_rows4(tile + c->lo_off, g, b);
    }
    for (uint32_t r = 0; r < rows; ++r) {
      const HVX_Vector vs = hvx_splat_sf(c->act_scale[m0 + r]);
      const HVX_Vector vz = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->act_zp[m0 + r]));
      HVX_Vector af = Q6_Vsf_equals_Vw(a[r]);
      if (c->lo_off) { /* 256 hi + lo, in f32 */
        af = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(af, hvx_splat_sf(256.0f)),
                                Q6_Vsf_equals_Vw(b[r]));
      }
      dq_put(c->dst_a + (size_t)(m0 + r) * c->dst_stride + c0 + 32u * j,
             dq_formula(af, vs, vz, cs[j], ws[j], bs[j]), al, c->accumulate);
    }
  }
  for (uint32_t r = 0; r < rows; ++r) {
    dq_row_post(jb, c->dst_a + (size_t)(m0 + r) * c->dst_stride + c0, m0 + r);
  }
}

/** @brief hvx_rmsnorm_rows_f32 norm_rs's lanes_sum of its qf32 per-lane
 *         sums @a q: the same conversion, five rotate-and-adds and lane 0. */
static inline float dq_lanes_sum(HVX_Vector q) {
  HVX_Vector v = Q6_Vsf_equals_Vqf32(q);
  for (int rot = 64; rot >= 4; rot >>= 1) {
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, rot));
  }
  const int32_t bits = Q6_R_vextract_VR(v, 0);
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

/** @brief norm_rs's 1 / sqrt(ss / chunk + eps), splat. */
static inline HVX_Vector dq_rs(float ss, uint32_t chunk, float eps) {
  return hvx_splat_sf(1.0f / sqrtf(ss / (float)chunk + eps));
}

/** @brief dq_store's load. */
static inline HVX_Vector dq_load(const float *p, int al) {
  return al ? *(const HVX_Vector *)p : *(const HVX_UVector *)p;
}

/** @brief One head of the 4-row group at @a d (rows @a ld apart), as
 *         dq_row_post leaves it: x rs gamma (norm_chunk's order), then
 *         (@a cs, row 0's table row) per pair of tiles a half head apart
 *         a c - b s and a s + b c (hvx_rope_row_f32's). Each step's four
 *         rows are loaded before any is stored, so they overlap. */
static inline __attribute__((always_inline)) void
dq_head_post4(float *restrict d, uint32_t ld, uint32_t ct,
              const float *restrict gamma, const float *restrict cs,
              uint32_t hd, HVX_Vector r0, HVX_Vector r1, HVX_Vector r2,
              HVX_Vector r3, int al) {
  const HVX_UVector *g = (const HVX_UVector *)gamma;
  if (!cs) {
    for (uint32_t k = 0; k < ct; ++k) {
      float *p = d + 32u * k;
      const HVX_Vector gk = g[k];
      const HVX_Vector x0 = dq_load(p, al), x1 = dq_load(p + ld, al);
      const HVX_Vector x2 = dq_load(p + 2u * ld, al);
      const HVX_Vector x3 = dq_load(p + 3u * ld, al);
      dq_store(p, Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x0, r0), gk), al);
      dq_store(p + ld, Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x1, r1), gk), al);
      dq_store(p + 2u * ld, Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x2, r2), gk),
               al);
      dq_store(p + 3u * ld, Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x3, r3), gk),
               al);
    }
    return;
  }
  const uint32_t half = hd / 2u;
#define DQ_ROPE_ROW(R)                                                         \
  const HVX_Vector a##R = Q6_Vsf_vmpy_VsfVsf(                                  \
    Q6_Vsf_vmpy_VsfVsf(dq_load(pa + R * ld, al), r##R), ga);                   \
  const HVX_Vector b##R = Q6_Vsf_vmpy_VsfVsf(                                  \
    Q6_Vsf_vmpy_VsfVsf(dq_load(pa + R * ld + half, al), r##R), gb);            \
  const HVX_Vector c##R = *(const HVX_UVector *)(cr + R * 2u * hd);            \
  const HVX_Vector s##R = *(const HVX_UVector *)(cr + R * 2u * hd + hd)
#define DQ_ROPE_PUT(R)                                                         \
  dq_store(pa + R * ld,                                                        \
           Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a##R, c##R),                  \
                              Q6_Vsf_vmpy_VsfVsf(b##R, s##R)),                 \
           al);                                                                \
  dq_store(pa + R * ld + half,                                                 \
           Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a##R, s##R),                  \
                              Q6_Vsf_vmpy_VsfVsf(b##R, c##R)),                 \
           al)
  for (uint32_t k = 0; k < ct / 2u; ++k) {
    float *pa = d + 32u * k;
    const float *cr = cs + 32u * k;
    const HVX_Vector ga = g[k], gb = g[k + ct / 2u];
    DQ_ROPE_ROW(0);
    DQ_ROPE_ROW(1);
    DQ_ROPE_ROW(2);
    DQ_ROPE_ROW(3);
    DQ_ROPE_PUT(0);
    DQ_ROPE_PUT(1);
    DQ_ROPE_PUT(2);
    DQ_ROPE_PUT(3);
  }
#undef DQ_ROPE_ROW
#undef DQ_ROPE_PUT
}

/** @brief dq_rows_planes's groups [g_lo, g_hi), the a16 low sweep
 *         (@a has_lo), the add (@a acc) and the per-head passes in the
 *         tile loop (@a fused) fixed per instance. The four rows are named
 *         values, the pointers restrict and the tile loop has no branch:
 *         with arrays indexed by the group's row count, the flags read per
 *         row and a store per row behind a row check, every row was one
 *         dependent chain, the column factors reloaded per row -- 92 cycles
 *         a vector on v79, 3x what the stores take. Fused, the tile loop
 *         runs one head at a time and sums the squares on the way (named
 *         qf32 sums: in an array hexagon-clang once carried one as sf, doc
 *         62 section 5.1), then dq_head_post4 rereads the head from the
 *         cache it was just written to. The floats are dq_row_post's to the
 *         bit: the squares of the stored values summed in the same order,
 *         norm_rs's scalar tail, the products in the same order. */
static inline __attribute__((always_inline)) void
dq_planes_groups(const hvx_dq_rows_job *jb, uint32_t g_lo, uint32_t g_hi,
                 const uint32_t has_lo, const uint32_t acc,
                 const uint32_t fused) {
  const hvx_dq_tiles_job *c = &jb->t;
  const uint32_t c0 = c->nt0 * HEXKL_ACC_TILE_COLS, ld = c->dst_stride;
  const uint32_t n_tiles = c->n_tiles, ts = c->tile_stride, lo = c->lo_off;
  const uint8_t *restrict tiles = c->tiles_base;
  const HVX_UVector *restrict cs = (const HVX_UVector *)(c->colsum_w + c0);
  const HVX_UVector *restrict ws = (const HVX_UVector *)(c->w_scale + c0);
  const HVX_UVector *restrict bs = (const HVX_UVector *)(c->bias + c0);
  float *restrict dst = c->dst_a + c0;
  const int al = dq_dst_aligned(c);
  const HVX_Vector v256 = hvx_splat_sf(256.0f);
  const uint32_t pc = jb->post_chunk;
  const uint32_t ct = fused ? pc / HEXKL_ACC_TILE_COLS : n_tiles;
  const float eps = jb->post_eps;
  for (uint32_t g = g_lo; g < g_hi; ++g) {
    const uint32_t m0 = 4u * g;
    if (c->m_count - m0 < 4u) {
      dq_planes_tail(jb, g, c->m_count - m0);
      continue;
    }
    const HVX_Vector vs0 = hvx_splat_sf(c->act_scale[m0]);
    const HVX_Vector vs1 = hvx_splat_sf(c->act_scale[m0 + 1u]);
    const HVX_Vector vs2 = hvx_splat_sf(c->act_scale[m0 + 2u]);
    const HVX_Vector vs3 = hvx_splat_sf(c->act_scale[m0 + 3u]);
    const HVX_Vector vz0 = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->act_zp[m0]));
    const HVX_Vector vz1 = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->act_zp[m0 + 1u]));
    const HVX_Vector vz2 = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->act_zp[m0 + 2u]));
    const HVX_Vector vz3 = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->act_zp[m0 + 3u]));
    float *row = dst + (size_t)m0 * ld;
    for (uint32_t h0 = 0; h0 < n_tiles; h0 += ct) {
      HVX_Vector q0 = Q6_V_vzero(), q1 = Q6_V_vzero();
      HVX_Vector q2 = Q6_V_vzero(), q3 = Q6_V_vzero();
      for (uint32_t j = h0; j < h0 + ct; ++j) {
        const uint8_t *tile = tiles + (size_t)j * ts;
        float *d = row + 32u * j;
        HVX_Vector a[4];
        planes_rows4(tile, g, a);
        HVX_Vector f0 = Q6_Vsf_equals_Vw(a[0]), f1 = Q6_Vsf_equals_Vw(a[1]);
        HVX_Vector f2 = Q6_Vsf_equals_Vw(a[2]), f3 = Q6_Vsf_equals_Vw(a[3]);
        if (has_lo) { /* 256 hi + lo, in f32 */
          HVX_Vector b[4];
          planes_rows4(tile + lo, g, b);
          f0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(f0, v256),
                                  Q6_Vsf_equals_Vw(b[0]));
          f1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(f1, v256),
                                  Q6_Vsf_equals_Vw(b[1]));
          f2 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(f2, v256),
                                  Q6_Vsf_equals_Vw(b[2]));
          f3 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(f3, v256),
                                  Q6_Vsf_equals_Vw(b[3]));
        }
        const HVX_Vector cj = cs[j], wj = ws[j], bj = bs[j];
        const HVX_Vector y0 = dq_formula(f0, vs0, vz0, cj, wj, bj);
        const HVX_Vector y1 = dq_formula(f1, vs1, vz1, cj, wj, bj);
        const HVX_Vector y2 = dq_formula(f2, vs2, vz2, cj, wj, bj);
        const HVX_Vector y3 = dq_formula(f3, vs3, vz3, cj, wj, bj);
        if (fused) { /* norm_rs's per-lane sums */
          q0 = Q6_Vqf32_vadd_Vqf32Vqf32(q0, Q6_Vqf32_vmpy_VsfVsf(y0, y0));
          q1 = Q6_Vqf32_vadd_Vqf32Vqf32(q1, Q6_Vqf32_vmpy_VsfVsf(y1, y1));
          q2 = Q6_Vqf32_vadd_Vqf32Vqf32(q2, Q6_Vqf32_vmpy_VsfVsf(y2, y2));
          q3 = Q6_Vqf32_vadd_Vqf32Vqf32(q3, Q6_Vqf32_vmpy_VsfVsf(y3, y3));
        }
        dq_put(d, y0, al, acc);
        dq_put(d + ld, y1, al, acc);
        dq_put(d + 2u * ld, y2, al, acc);
        dq_put(d + 3u * ld, y3, al, acc);
      }
      if (fused) {
        /** All four sums out of the vector registers before the first
         *  sqrtf call: a qf32 value held across a call came back from its
         *  spill read as sf, rows 1-3 of every group garbage. */
        const float s0 = dq_lanes_sum(q0), s1 = dq_lanes_sum(q1);
        const float s2 = dq_lanes_sum(q2), s3 = dq_lanes_sum(q3);
        dq_head_post4(row + 32u * h0, ld, ct, jb->post_gamma,
                      jb->rope_cs ? jb->rope_cs + (size_t)m0 * 2u * jb->rope_hd
                                  : NULL,
                      jb->rope_hd, dq_rs(s0, pc, eps), dq_rs(s1, pc, eps),
                      dq_rs(s2, pc, eps), dq_rs(s3, pc, eps), al);
      }
    }
    if (!fused && (jb->post_chunk != 0u || jb->rope_cs)) {
      for (uint32_t r = 0; r < 4u; ++r) {
        dq_row_post(jb, row + (size_t)r * ld, m0 + r);
      }
    }
  }
}

/** @brief hvx_dq_tiles_rows_worker over byte-plane tiles: the rows in
 *         groups of four, the same floats per row. The per-head passes go
 *         in the tile loop when they are what the FC call asks for (a norm
 *         with its gamma, the rotation's head the norm's piece, whole
 *         pieces, no add); otherwise dq_row_post after each group. */
static void dq_rows_planes(const hvx_dq_rows_job *jb, uint32_t n_threads,
                           uint32_t i) {
  const hvx_dq_tiles_job *c = &jb->t;
  const uint32_t n_g = (c->m_count + 3u) / 4u;
  const uint32_t g_lo = (uint32_t)((uint64_t)n_g * i / n_threads);
  const uint32_t g_hi = (uint32_t)((uint64_t)n_g * (i + 1) / n_threads);
  const uint32_t pc = jb->post_chunk;
  const int fused = pc != 0u && pc % (2u * HEXKL_ACC_TILE_COLS) == 0u &&
                    jb->post_gamma && !c->accumulate &&
                    (c->n_tiles * HEXKL_ACC_TILE_COLS) % pc == 0u &&
                    (!jb->rope_cs || jb->rope_hd == pc);
  if (fused) {
    if (c->lo_off) {
      dq_planes_groups(jb, g_lo, g_hi, 1u, 0u, 1u);
    } else {
      dq_planes_groups(jb, g_lo, g_hi, 0u, 0u, 1u);
    }
    return;
  }
  switch ((c->lo_off != 0u) * 2u + (c->accumulate != 0u)) {
  case 0:
    dq_planes_groups(jb, g_lo, g_hi, 0u, 0u, 0u);
    break;
  case 1:
    dq_planes_groups(jb, g_lo, g_hi, 0u, 1u, 0u);
    break;
  case 2:
    dq_planes_groups(jb, g_lo, g_hi, 1u, 0u, 0u);
    break;
  default:
    dq_planes_groups(jb, g_lo, g_hi, 1u, 1u, 0u);
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
      dq_put(seg + 32u * j, dq_formula(af, vs, vz, cs[j], ws[j], bs[j]), al,
             c->accumulate);
    }
    dq_row_post(jb, seg, m);
  }
}

/** @brief Rows 4g..4g+3 of byte-plane tile @a tile as dq_rows_planes makes
 *         them: f32, the a16 low sweep (lo_off) combined. */
static inline void dq_rows4_f32(const hvx_dq_tiles_job *c, const uint8_t *tile,
                                uint32_t g, const HVX_Vector cs,
                                const HVX_Vector ws, const HVX_Vector bs,
                                HVX_Vector out[4]) {
  HVX_Vector a[4], b[4];
  planes_rows4(tile, g, a);
  if (c->lo_off) {
    planes_rows4(tile + c->lo_off, g, b);
  }
  for (uint32_t r = 0; r < 4u; ++r) {
    HVX_Vector af = Q6_Vsf_equals_Vw(a[r]);
    if (c->lo_off) {
      af = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(af, hvx_splat_sf(256.0f)),
                              Q6_Vsf_equals_Vw(b[r]));
    }
    out[r] = dq_formula(af, hvx_splat_sf(c->act_scale[4u * g + r]),
                        Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->act_zp[4u * g + r])),
                        cs, ws, bs);
  }
}

void hvx_gelu_tab_build(float *tab) {
  HVX_Vector *t = (HVX_Vector *)tab,
             *s = (HVX_Vector *)(tab + HVX_GELU_TAB_STRIDE);
  float x0[32];
  for (uint32_t k = 0; k < 32u; ++k) {
    x0[k] = (float)((int32_t)k - 1536) * (1.0f / 256.0f);
  }
  /* k / 256 - 6 is exact in f32 here, and so is each step's sum; memcpy,
     not a float store under a vector load, which strict aliasing lets the
     compiler reorder */
  HVX_Vector x;
  memcpy(&x, x0, sizeof x);
  HVX_Vector cur = hvx_geglu_det_sf(x, hvx_splat_sf(1.0f));
  /* each slope from registers, the next vector's first node by valign: an
     unaligned reload of the just-stored nodes read lane 31 wrong on the S25
     (the emulator's was right; why, not pinned down) */
  for (uint32_t v = 0; v < HVX_GELU_TAB_STRIDE / 32u; ++v) {
    x = Q6_Vsf_vadd_VsfVsf(x, hvx_splat_sf(32.0f / 256.0f));
    const HVX_Vector nxt = hvx_geglu_det_sf(x, hvx_splat_sf(1.0f));
    t[v] = cur;
    s[v] = Q6_Vsf_vsub_VsfVsf(Q6_V_valign_VVR(nxt, cur, 4), cur);
    cur = nxt;
  }
}

/** @brief gelu(g) * u of @a nr rows (4 or 8) by hvx_gelu_tab_build's
 *         tables: node k = rne(x - 1/2) <= x = 256 g + 1536 (clamped to
 *         [0, 3072], so k + f lands on node 3072 with f = 0 at most),
 *         T[k] + S[k] f, both gathered from VTCM, and g past 6. Every
 *         gather is issued before the first landing read. */
static inline __attribute__((always_inline)) void
gelu_tab_rows(const float *tab, HVX_Vector *land, const HVX_Vector *g,
              const HVX_Vector *u, HVX_Vector *y, const uint32_t nr) {
  const uint32_t mu = 4u * HVX_GELU_TAB_NODES - 1u;
  HVX_Vector f[8];
  for (uint32_t r = 0; r < nr; ++r) {
    HVX_Vector x = Q6_Vsf_vadd_VsfVsf(
      Q6_Vsf_vmpy_VsfVsf(g[r], hvx_splat_sf(256.0f)), hvx_splat_sf(1536.0f));
    x = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(x, Q6_V_vzero()),
                           hvx_splat_sf(3072.0f));
    const HVX_Vector k =
      hvx_sf_to_w_rne(Q6_Vsf_vsub_VsfVsf(x, hvx_splat_sf(0.5f)));
    f[r] = Q6_Vsf_vsub_VsfVsf(x, Q6_Vsf_equals_Vw(k));
    const HVX_Vector off = Q6_Vw_vasl_VwR(k, 2);
    Q6_vgather_ARMVw(land + 2u * r, (uintptr_t)tab, mu, off);
    Q6_vgather_ARMVw(land + 2u * r + 1u, (uintptr_t)(tab + HVX_GELU_TAB_STRIDE),
                     mu, off);
  }
  for (uint32_t r = 0; r < nr; ++r) {
    const HVX_Vector t = Q6_Vsf_vadd_VsfVsf(
      land[2u * r], Q6_Vsf_vmpy_VsfVsf(land[2u * r + 1u], f[r]));
    /* g > 6 on the bits: 6.0f is positive, so a negative g compares less */
    y[r] = Q6_Vsf_vmpy_VsfVsf(
      Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(g[r], Q6_V_vsplat_R(0x40c00000)), g[r],
                    t),
      u[r]);
  }
}

/** @brief Unit (g, j)'s four gate rows into @a gv and up rows into @a uv. */
static inline __attribute__((always_inline)) void
glu16_in(const hvx_dq_tiles_job *c, uint32_t inter, uint32_t g, uint32_t j,
         HVX_Vector *gv, HVX_Vector *uv) {
  const uint32_t cg = (c->nt0 + j) * HEXKL_ACC_TILE_COLS;
  const uint32_t cu = inter + cg;
  dq_rows4_f32(c, c->tiles_base + (size_t)j * c->tile_stride, g,
               *(const HVX_UVector *)(c->colsum_w + cg),
               *(const HVX_UVector *)(c->w_scale + cg),
               *(const HVX_UVector *)(c->bias + cg), gv);
  dq_rows4_f32(c, c->tiles_base + (size_t)(c->n_tiles + j) * c->tile_stride, g,
               *(const HVX_UVector *)(c->colsum_w + cu),
               *(const HVX_UVector *)(c->w_scale + cu),
               *(const HVX_UVector *)(c->bias + cu), uv);
}

/** @brief Four GLU rows of group @a g quantized and stored at byte @a d of
 *         both planes. */
static inline __attribute__((always_inline)) void
glu16_out(const HVX_Vector *y, uint32_t g, uint32_t m_count, HVX_Vector vinv,
          HVX_Vector vzp, uint8_t *plo, uint8_t *phi, size_t d) {
  HVX_Vector q[4];
  for (uint32_t r = 0; r < 4u; ++r) {
    q[r] =
      4u * g + r < m_count
        ? Q6_Vw_vadd_VwVw(hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(y[r], vinv)), vzp)
        : Q6_V_vzero();
  }
  /* hvx_quant_u8.c's store16_group4: rows 0-1 low half, 2-3 high */
  const HVX_Vector h01 = Q6_Vuh_vpack_VwVw_sat(q[1], q[0]);
  const HVX_Vector h23 = Q6_Vuh_vpack_VwVw_sat(q[3], q[2]);
  *(HVX_UVector *)(plo + d) = Q6_Vb_vpacke_VhVh(h23, h01);
  *(HVX_UVector *)(phi + d) = Q6_Vb_vpacko_VhVh(h23, h01);
}

void hvx_dq_glu16_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_dq_glu16_job *jb = (const hvx_dq_glu16_job *)vjob;
  /* a copy: the byte stores below may alias *jb, and through it the
     compiler reloaded every field a unit */
  const hvx_dq_tiles_job t = jb->t;
  const hvx_dq_tiles_job *c = &t;
  const uint32_t np = c->n_tiles;
  const uint32_t n = (c->m_count + 3u) / 4u * np;
  const uint32_t lo = (uint32_t)((uint64_t)n * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)n * (i + 1) / n_threads);
  if (lo >= hi) {
    return;
  }
  /* hvx_quant_u8.c's q16_run: rne(x / s) + zp */
  const HVX_Vector vinv = hvx_splat_sf(1.0f / jb->scale);
  const HVX_Vector vzp = Q6_V_vsplat_R(jb->zp);
  const uint32_t inter = jb->inter, act = jb->act;
  const float *const tab = jb->tab;
  HVX_Vector *const land =
    tab ? (HVX_Vector *)(jb->land + (size_t)i * HVX_GELU_TAB_LANE_BYTES) : NULL;
  /* AH tiles of 64 rows x 32 columns, 2 KiB each */
  uint8_t *const plo = jb->lo + (size_t)(jb->col / 32u) * 2048u;
  uint8_t *const phi = jb->hi + (size_t)(jb->col / 32u) * 2048u;
  /** (group, pair) units, a group's pairs in a run, the rows' own pad rows
      computed and then dropped. With the table, two pairs of a group at a
      time: eight independent GLU chains and 16 gathers in flight. On the
      S25, a 16-pair batch of 64 rows on five workers, gates within +-2:
      14.9 us a pair at a time, 11.2 two, 20.3 four (spilled). */
  uint32_t g = lo / np, j = lo % np;
  for (uint32_t u = lo; u < hi;) {
    const uint32_t two = tab != NULL && j + 1u < np && u + 1u < hi;
    HVX_Vector gv[8], uv[8], y[8];
    glu16_in(c, inter, g, j, gv, uv);
    if (two) {
      glu16_in(c, inter, g, j + 1u, gv + 4, uv + 4);
      gelu_tab_rows(tab, land, gv, uv, y, 8u);
    } else if (tab) {
      gelu_tab_rows(tab, land, gv, uv, y, 4u);
    } else {
      for (uint32_t r = 0; r < 4u; ++r) {
        y[r] = act == HVX_GLU_GELU_TANH ? hvx_geglu_det_sf(gv[r], uv[r])
                                        : hvx_swiglu_det_sf(gv[r], uv[r]);
      }
    }
    const size_t d = (size_t)j * 2048u + (size_t)g * 128u;
    glu16_out(y, g, c->m_count, vinv, vzp, plo, phi, d);
    if (two) {
      glu16_out(y + 4, g, c->m_count, vinv, vzp, plo, phi, d + 2048u);
    }
    u += 1u + two;
    j += 1u + two;
    if (j == np) {
      j = 0;
      ++g;
    }
  }
}

void hvx_dequant_acc16_tile_to_f32(
  const int32_t *hi, const int32_t *lo, uint32_t row_stride, uint32_t m_count,
  const float *act_scale, const int32_t *act_zp, const int32_t *colsum_w,
  const float *w_scale, const float *bias, float *out, uint32_t out_stride,
  int accumulate) {
  const HVX_Vector cs = ((const HVX_UVector *)colsum_w)[0];
  const HVX_Vector ws = ((const HVX_UVector *)w_scale)[0];
  const HVX_Vector bs = ((const HVX_UVector *)bias)[0];
  const HVX_Vector v256 = hvx_splat_sf(256.0f);
  for (uint32_t m = 0; m < m_count; ++m) {
    const HVX_Vector af = Q6_Vsf_vadd_VsfVsf(
      Q6_Vsf_vmpy_VsfVsf(
        Q6_Vsf_equals_Vw(
          ((const HVX_UVector *)(hi + (size_t)m * row_stride))[0]),
        v256),
      Q6_Vsf_equals_Vw(
        ((const HVX_UVector *)(lo + (size_t)m * row_stride))[0]));
    dq_put(out + (size_t)m * out_stride,
           dq_formula(af, hvx_splat_sf(act_scale[m]),
                      Q6_Vsf_equals_Vw(Q6_V_vsplat_R(act_zp[m])), cs, ws, bs),
           0, (uint32_t)accumulate);
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
                        0u,         0u,          0u};
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
