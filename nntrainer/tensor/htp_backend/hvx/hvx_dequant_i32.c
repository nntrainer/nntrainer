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

/** Four contiguous rows of one signed24 tile, reconstructed in registers. */
typedef struct {
  HVX_Vector v0, v1, v2, v3;
} i24_rows4;

static inline i24_rows4 i24_unpack4(HVX_Vector lo, HVX_Vector mid,
                                    HVX_Vector hi) {
  const HVX_VectorPair low16 = Q6_W_vshuff_VVR(mid, lo, -1);
  const HVX_VectorPair high16 = Q6_Wh_vunpack_Vb(hi);
  const HVX_VectorPair a =
    Q6_W_vshuff_VVR(Q6_V_lo_W(high16), Q6_V_lo_W(low16), -2);
  const HVX_VectorPair b =
    Q6_W_vshuff_VVR(Q6_V_hi_W(high16), Q6_V_hi_W(low16), -2);
  const i24_rows4 result = {Q6_V_lo_W(a), Q6_V_hi_W(a), Q6_V_lo_W(b),
                            Q6_V_hi_W(b)};
  return result;
}

/** m is a multiple of four, so the three loads are aligned. */
static inline i24_rows4 i24_load4(const uint8_t *tile, uint32_t m) {
  const uint8_t *p = tile + m * 32u;
  return i24_unpack4(*(const HVX_Vector *)p, *(const HVX_Vector *)(p + 2048u),
                     *(const HVX_Vector *)(p + 4096u));
}

/** A partial worker range or the last 1..3 rows. Unaligned loads may read
 * 96 bytes past a plane, still inside the tile's original 8 KiB ring slot.
 * Only the first 32 lanes are used; unused adjacent rows cannot affect them. */
static inline HVX_Vector i24_load_row(const uint8_t *tile, uint32_t m) {
  const uint8_t *p = tile + m * 32u;
  return i24_unpack4(*(const HVX_UVector *)p, *(const HVX_UVector *)(p + 2048u),
                     *(const HVX_UVector *)(p + 4096u))
    .v0;
}

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

void hvx_dequant_acc_3tiles_m1_f32(const int32_t *tiles, float act_scale,
                                   int32_t act_zp, const int32_t *colsum_w,
                                   const float *w_scale, const float *bias,
                                   float *out) {
  const HVX_Vector vs = hvx_splat_sf(act_scale);
  const HVX_Vector vz = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(act_zp));
  const HVX_UVector *va = (const HVX_UVector *)tiles;
  const HVX_UVector *vc = (const HVX_UVector *)colsum_w;
  const HVX_UVector *vw = (const HVX_UVector *)w_scale;
  const HVX_UVector *vb = (const HVX_UVector *)bias;
  HVX_UVector *vo = (HVX_UVector *)out;

  /** Prefill interleaves independent rows; M=1 instead interleaves the
     three neighboring output tiles. Keep each tile's exact conversion,
     subtract, two multiplies and add from hvx_dequant_acc_tile_to_f32. */
  const HVX_Vector c0 = Q6_Vsf_vsub_VsfVsf(
    Q6_Vsf_equals_Vw(va[0]), Q6_Vsf_vmpy_VsfVsf(vz, Q6_Vsf_equals_Vw(vc[0])));
  const HVX_Vector c1 = Q6_Vsf_vsub_VsfVsf(
    Q6_Vsf_equals_Vw(va[1]), Q6_Vsf_vmpy_VsfVsf(vz, Q6_Vsf_equals_Vw(vc[1])));
  const HVX_Vector c2 = Q6_Vsf_vsub_VsfVsf(
    Q6_Vsf_equals_Vw(va[2]), Q6_Vsf_vmpy_VsfVsf(vz, Q6_Vsf_equals_Vw(vc[2])));
  const HVX_Vector s0 = Q6_Vsf_vmpy_VsfVsf(c0, vs);
  const HVX_Vector s1 = Q6_Vsf_vmpy_VsfVsf(c1, vs);
  const HVX_Vector s2 = Q6_Vsf_vmpy_VsfVsf(c2, vs);
  const HVX_Vector w0 = Q6_Vsf_vmpy_VsfVsf(s0, vw[0]);
  const HVX_Vector w1 = Q6_Vsf_vmpy_VsfVsf(s1, vw[1]);
  const HVX_Vector w2 = Q6_Vsf_vmpy_VsfVsf(s2, vw[2]);
  vo[0] = Q6_Vsf_vadd_VsfVsf(w0, vb[0]);
  vo[1] = Q6_Vsf_vadd_VsfVsf(w1, vb[1]);
  vo[2] = Q6_Vsf_vadd_VsfVsf(w2, vb[2]);
}

void hvx_dequant_acc_4tiles_m1_f32(const int32_t *tiles, float act_scale,
                                   int32_t act_zp, const int32_t *colsum_w,
                                   const float *w_scale, const float *bias,
                                   float *out) {
  const HVX_Vector vs = hvx_splat_sf(act_scale);
  const HVX_Vector vz = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(act_zp));
  const HVX_UVector *va = (const HVX_UVector *)tiles;
  const HVX_UVector *vc = (const HVX_UVector *)colsum_w;
  const HVX_UVector *vw = (const HVX_UVector *)w_scale;
  const HVX_UVector *vb = (const HVX_UVector *)bias;
  HVX_UVector *vo = (HVX_UVector *)out;
  /** Preserve each column's conversion, subtract, multiply, multiply, add
   * order while scheduling four independent dependency chains together. */
  const HVX_Vector c0 = Q6_Vsf_vsub_VsfVsf(
    Q6_Vsf_equals_Vw(va[0]), Q6_Vsf_vmpy_VsfVsf(vz, Q6_Vsf_equals_Vw(vc[0])));
  const HVX_Vector c1 = Q6_Vsf_vsub_VsfVsf(
    Q6_Vsf_equals_Vw(va[1]), Q6_Vsf_vmpy_VsfVsf(vz, Q6_Vsf_equals_Vw(vc[1])));
  const HVX_Vector c2 = Q6_Vsf_vsub_VsfVsf(
    Q6_Vsf_equals_Vw(va[2]), Q6_Vsf_vmpy_VsfVsf(vz, Q6_Vsf_equals_Vw(vc[2])));
  const HVX_Vector c3 = Q6_Vsf_vsub_VsfVsf(
    Q6_Vsf_equals_Vw(va[3]), Q6_Vsf_vmpy_VsfVsf(vz, Q6_Vsf_equals_Vw(vc[3])));
  const HVX_Vector s0 = Q6_Vsf_vmpy_VsfVsf(c0, vs);
  const HVX_Vector s1 = Q6_Vsf_vmpy_VsfVsf(c1, vs);
  const HVX_Vector s2 = Q6_Vsf_vmpy_VsfVsf(c2, vs);
  const HVX_Vector s3 = Q6_Vsf_vmpy_VsfVsf(c3, vs);
  const HVX_Vector w0 = Q6_Vsf_vmpy_VsfVsf(s0, vw[0]);
  const HVX_Vector w1 = Q6_Vsf_vmpy_VsfVsf(s1, vw[1]);
  const HVX_Vector w2 = Q6_Vsf_vmpy_VsfVsf(s2, vw[2]);
  const HVX_Vector w3 = Q6_Vsf_vmpy_VsfVsf(s3, vw[3]);
  vo[0] = Q6_Vsf_vadd_VsfVsf(w0, vb[0]);
  vo[1] = Q6_Vsf_vadd_VsfVsf(w1, vb[1]);
  vo[2] = Q6_Vsf_vadd_VsfVsf(w2, vb[2]);
  vo[3] = Q6_Vsf_vadd_VsfVsf(w3, vb[3]);
}

static inline void dq_acc_tiles_scatter_rows(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_tiles,
  uint32_t row_stride, uint32_t m0, uint32_t m1, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, float *out, uint32_t out_stride, const uint32_t *rows,
  const float *weights, uint8_t *seen, const int32_t *offset,
  const int32_t *sums) {
  /** One output row's n_tiles vectors, [47:32] stride, [31:16] width, [15:0]
     height. A hint: it moves no bits. Three rows ahead, the most a lane
     can queue (the hardware stalls the thread on a fourth). */
  const uint64_t box = ((uint64_t)(n_tiles * 128u) << 32) |
                       ((uint64_t)(n_tiles * 128u) << 16) | 1u;
  for (uint32_t m = m0; m < m1 && m < m0 + 3u; ++m)
    if (!seen || seen[rows[m]])
      Q6_l2fetch_AP((void *)(out + (size_t)rows[m] * out_stride), box);
  for (uint32_t m = m0; m < m1; ++m) {
    if (m + 3u < m1 && (!seen || seen[rows[m + 3u]]))
      Q6_l2fetch_AP((void *)(out + (size_t)rows[m + 3u] * out_stride), box);
    /* a row nobody has written yet: zeros, not its bytes */
    const int first = seen && !seen[rows[m]];
    if (first)
      seen[rows[m]] = 1u;
    const HVX_Vector vs = hvx_splat_sf(act_scale[m]);
    const HVX_Vector vz = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(act_zp[m]));
    const HVX_Vector vwt = hvx_splat_sf(weights[m]);
    const HVX_Vector sum = offset ? Q6_V_vsplat_R(sums[m]) : Q6_V_vzero();
    HVX_UVector *vo = (HVX_UVector *)(out + (size_t)rows[m] * out_stride);
    for (uint32_t j = 0; j < n_tiles; ++j) {
      const int32_t *tile =
        (const int32_t *)(tiles_base + (size_t)j * tile_stride);
      HVX_Vector acc =
        ((const HVX_UVector *)(tile + (size_t)m * row_stride))[0];
      if (offset)
        acc =
          Q6_Vw_vmpyieacc_VwVwVh(acc, sum, ((const HVX_UVector *)offset)[j]);
      const HVX_Vector af = Q6_Vsf_equals_Vw(acc);
      const HVX_Vector csf =
        Q6_Vsf_equals_Vw(((const HVX_UVector *)colsum_w)[j]);
      const HVX_Vector r = Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_vmpy_VsfVsf(
          Q6_Vsf_vmpy_VsfVsf(
            Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vz, csf)), vs),
          ((const HVX_UVector *)w_scale)[j]),
        ((const HVX_UVector *)bias)[j]);
      const HVX_Vector prev = first ? Q6_V_vzero() : vo[j];
      vo[j] = Q6_Vsf_vadd_VsfVsf(prev, Q6_Vsf_vmpy_VsfVsf(r, vwt));
    }
  }
}

void hvx_dequant_acc_tiles_scatter_rows_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_tiles,
  uint32_t row_stride, uint32_t m0, uint32_t m1, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, float *out, uint32_t out_stride, const uint32_t *rows,
  const float *weights, uint8_t *seen) {
  dq_acc_tiles_scatter_rows(tiles_base, tile_stride, n_tiles, row_stride, m0,
                            m1, act_scale, act_zp, colsum_w, w_scale, bias, out,
                            out_stride, rows, weights, seen, NULL, NULL);
}

void hvx_dequant_acc_tiles_scatter_rows_affine_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_tiles,
  uint32_t row_stride, uint32_t m0, uint32_t m1, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, float *out, uint32_t out_stride, const uint32_t *rows,
  const float *weights, uint8_t *seen, const int32_t *offset,
  const int32_t *sums) {
  dq_acc_tiles_scatter_rows(tiles_base, tile_stride, n_tiles, row_stride, m0,
                            m1, act_scale, act_zp, colsum_w, w_scale, bias, out,
                            out_stride, rows, weights, seen, offset, sums);
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
    /** Which half of a gate_up result this tile belongs to. A caller with one
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

static inline void dq_tiles_rows(uint32_t n_threads, uint32_t i,
                                 const hvx_dq_rows_job *jb,
                                 const int32_t *offset, const int32_t *sums) {
  const hvx_dq_tiles_job *c = &jb->t;
  const uint32_t lo = (uint32_t)((uint64_t)c->m_count * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->m_count * (i + 1) / n_threads);
  const uint32_t c0 = c->nt0 * HEXKL_ACC_TILE_COLS;
  const HVX_UVector *cs = (const HVX_UVector *)(c->colsum_w + c0);
  const HVX_UVector *ws = (const HVX_UVector *)(c->w_scale + c0);
  const HVX_UVector *bs = (const HVX_UVector *)(c->bias + c0);
  const HVX_UVector *off = offset ? (const HVX_UVector *)(offset + c0) : NULL;

  uint32_t m = lo;
  // Four independent rows share each column's weight metadata. Integer
  // affine correction and every floating-point operation retain their order.
  for (; m + 4u <= hi; m += 4u) {
    HVX_Vector scale[4], zp[4], sum[4];
    HVX_UVector *dst[4];
    for (uint32_t p = 0; p < 4u; ++p) {
      scale[p] = hvx_splat_sf(c->act_scale[m + p]);
      zp[p] = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->act_zp[m + p]));
      sum[p] = offset ? Q6_V_vsplat_R(sums[m + p]) : Q6_V_vzero();
      dst[p] = (HVX_UVector *)(c->dst_a + (size_t)(m + p) * c->dst_stride + c0);
    }
    for (uint32_t j = 0; j < c->n_tiles; ++j) {
      const int32_t *tile =
        (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
      const HVX_Vector colsum = Q6_Vsf_equals_Vw(cs[j]);
      const HVX_Vector weight = ws[j], bias = bs[j];
      const HVX_Vector correction = offset ? off[j] : Q6_V_vzero();
      for (uint32_t p = 0; p < 4u; ++p) {
        HVX_Vector acc =
          ((const HVX_UVector *)(tile + (size_t)(m + p) * c->row_stride))[0];
        if (offset)
          acc = Q6_Vw_vmpyieacc_VwVwVh(acc, sum[p], correction);
        const HVX_Vector af = Q6_Vsf_equals_Vw(acc);
        dst[p][j] = Q6_Vsf_vadd_VsfVsf(
          Q6_Vsf_vmpy_VsfVsf(
            Q6_Vsf_vmpy_VsfVsf(
              Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(zp[p], colsum)),
              scale[p]),
            weight),
          bias);
      }
    }
    if (jb->row_done)
      for (uint32_t p = 0; p < 4u; ++p)
        jb->row_done(jb->ctx, (float *)dst[p], m + p);
  }
  for (; m < hi; ++m) {
    const HVX_Vector vs = hvx_splat_sf(c->act_scale[m]);
    const HVX_Vector vz = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->act_zp[m]));
    const HVX_Vector sum = offset ? Q6_V_vsplat_R(sums[m]) : Q6_V_vzero();
    float *seg = c->dst_a + (size_t)m * c->dst_stride + c0;
    HVX_UVector *vo = (HVX_UVector *)seg;
    for (uint32_t j = 0; j < c->n_tiles; ++j) {
      const int32_t *tile =
        (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
      HVX_Vector acc =
        ((const HVX_UVector *)(tile + (size_t)m * c->row_stride))[0];
      /** The offset is a signed low halfword, while the sum and accumulator
       * remain int32. Correct before conversion: a float correction would
       * round differently when the integer result exceeds 2^24. */
      if (offset)
        acc = Q6_Vw_vmpyieacc_VwVwVh(acc, sum, off[j]);
      const HVX_Vector af = Q6_Vsf_equals_Vw(acc);
      /* DQ_TILE_ROW's operations, in its order */
      vo[j] = Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_vmpy_VsfVsf(
          Q6_Vsf_vmpy_VsfVsf(
            Q6_Vsf_vsub_VsfVsf(af,
                               Q6_Vsf_vmpy_VsfVsf(vz, Q6_Vsf_equals_Vw(cs[j]))),
            vs),
          ws[j]),
        bs[j]);
    }
    if (jb->row_done) {
      jb->row_done(jb->ctx, seg, m);
    }
  }
}

void hvx_dq_tiles_rows_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  dq_tiles_rows(n_threads, i, (const hvx_dq_rows_job *)vjob, NULL, NULL);
}

void hvx_dq_tiles_rows_affine(uint32_t n_threads, uint32_t i,
                              const hvx_dq_rows_job *job, const int32_t *offset,
                              const int32_t *sums) {
  dq_tiles_rows(n_threads, i, job, offset, sums);
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
                        dst_a,      dst_b,       split,    dst_stride, n_tiles};
  hvx_worker_pool_run(pool, hvx_dq_tiles_worker, &c, n_tiles);
}

/* ---- fused dequant + SwiGLU over gate/up tile pairs ---------------------- */

/**
 * @brief One row of one tile, dequantized -- DQ_TILE_ROW's operations in
 *        the same order, as a function so two of them can feed SwiGLU.
 */
static inline HVX_Vector dq_acc_sf(HVX_Vector acc, float act_scale,
                                   int32_t act_zp, HVX_Vector csf,
                                   HVX_Vector vw, HVX_Vector vbias) {
  const HVX_Vector af = Q6_Vsf_equals_Vw(acc);
  const HVX_Vector vs = hvx_splat_sf(act_scale);
  const HVX_Vector vz = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(act_zp));
  return Q6_Vsf_vadd_VsfVsf(
    Q6_Vsf_vmpy_VsfVsf(
      Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vz, csf)),
                         vs),
      vw),
    vbias);
}

static inline HVX_Vector dq_row_sf(const int32_t *row, float act_scale,
                                   int32_t act_zp, HVX_Vector csf,
                                   HVX_Vector vw, HVX_Vector vbias) {
  return dq_acc_sf(((const HVX_UVector *)row)[0], act_scale, act_zp, csf, vw,
                   vbias);
}

/** Exact zero-point correction for signed24 u8 accumulators. The caller
 * establishes hexkl_acc_i24_range and zero points in [0, 255], so acc and
 * zp * colsum are integers in [-2^23, 2^23 - 1]. Their difference fits the
 * f32 exact-integer range [-2^24, 2^24]. The original float multiply and
 * subtract therefore have no rounding to preserve here. Keep the scale,
 * bias, activation and scatter operations in their original order. */
static inline HVX_Vector dq_acc_i24_sf(HVX_Vector acc, float scale, int32_t zp,
                                       HVX_Vector colsum, HVX_Vector ws,
                                       HVX_Vector bias) {
  const HVX_Vector corrected =
    Q6_Vw_vmpyieacc_VwVwVh(acc, colsum, Q6_V_vsplat_R(-zp));
  const HVX_Vector af = Q6_Vsf_equals_Vw(corrected);
  return Q6_Vsf_vadd_VsfVsf(
    Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(af, hvx_splat_sf(scale)), ws), bias);
}

void hvx_dequant_i24_tiles_scatter_rows_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_tiles,
  uint32_t m0, uint32_t m1, const float *act_scale, const int32_t *act_zp,
  const int32_t *colsum_w, const float *w_scale, const float *bias, float *out,
  uint32_t out_stride, const uint32_t *rows, const float *weights,
  uint8_t *seen) {
  const uint64_t box = ((uint64_t)(n_tiles * 128u) << 32) |
                       ((uint64_t)(n_tiles * 128u) << 16) | 1u;
  uint32_t m = m0;
  while (m < m1) {
    if (m % 4u || m1 - m < 4u) {
      const int first = seen && !seen[rows[m]];
      if (first)
        seen[rows[m]] = 1u;
      if (!first)
        Q6_l2fetch_AP((void *)(out + (size_t)rows[m] * out_stride), box);
      const HVX_Vector wt = hvx_splat_sf(weights[m]);
      HVX_UVector *vo = (HVX_UVector *)(out + (size_t)rows[m] * out_stride);
      for (uint32_t j = 0; j < n_tiles; ++j) {
        const HVX_Vector acc =
          i24_load_row(tiles_base + (size_t)j * tile_stride, m);
        const HVX_Vector r = dq_acc_i24_sf(
          acc, act_scale[m], act_zp[m], ((const HVX_UVector *)colsum_w)[j],
          ((const HVX_UVector *)w_scale)[j], ((const HVX_UVector *)bias)[j]);
        const HVX_Vector prev = first ? Q6_V_vzero() : vo[j];
        vo[j] = Q6_Vsf_vadd_VsfVsf(prev, Q6_Vsf_vmpy_VsfVsf(r, wt));
      }
      ++m;
      continue;
    }
#define I24_SCATTER_PREP(i)                                                    \
  const int first##i = seen && !seen[rows[m + (i)]];                           \
  if (first##i)                                                                \
    seen[rows[m + (i)]] = 1u;                                                  \
  HVX_UVector *vo##i =                                                         \
    (HVX_UVector *)(out + (size_t)rows[m + (i)] * out_stride);                 \
  if (!first##i)                                                               \
    Q6_l2fetch_AP(vo##i, box);                                                 \
  const HVX_Vector vs##i = hvx_splat_sf(act_scale[m + (i)]);                   \
  const HVX_Vector vz##i = Q6_V_vsplat_R(-act_zp[m + (i)]);                    \
  const HVX_Vector wt##i = hvx_splat_sf(weights[m + (i)])
    I24_SCATTER_PREP(0);
    I24_SCATTER_PREP(1);
    I24_SCATTER_PREP(2);
    I24_SCATTER_PREP(3);
    for (uint32_t j = 0; j < n_tiles; ++j) {
      const i24_rows4 acc = i24_load4(tiles_base + (size_t)j * tile_stride, m);
      const HVX_Vector csw = ((const HVX_UVector *)colsum_w)[j];
      const HVX_Vector ws = ((const HVX_UVector *)w_scale)[j];
      const HVX_Vector bs = ((const HVX_UVector *)bias)[j];
#define I24_SCATTER_STORE(i)                                                   \
  do {                                                                         \
    const HVX_Vector af =                                                      \
      Q6_Vsf_equals_Vw(Q6_Vw_vmpyieacc_VwVwVh(acc.v##i, csw, vz##i));          \
    const HVX_Vector r = Q6_Vsf_vadd_VsfVsf(                                   \
      Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(af, vs##i), ws), bs);              \
    const HVX_Vector prev = first##i ? Q6_V_vzero() : vo##i[j];                \
    vo##i[j] = Q6_Vsf_vadd_VsfVsf(prev, Q6_Vsf_vmpy_VsfVsf(r, wt##i));         \
  } while (0)
      I24_SCATTER_STORE(0);
      I24_SCATTER_STORE(1);
      I24_SCATTER_STORE(2);
      I24_SCATTER_STORE(3);
    }
    m += 4u;
#undef I24_SCATTER_PREP
#undef I24_SCATTER_STORE
  }
}

/** Specialize the legacy entry point at NULL offset: its hot loops must
 * not inherit affine branches for every native TL2 expert tile. */
static inline __attribute__((always_inline)) void
dq_swiglu_span(const hvx_dq_swiglu_job *c, const int32_t *offset,
               const int32_t *sums, int planar, uint32_t up_pairs, uint32_t lo,
               uint32_t hi) {

  for (uint32_t j = lo; j < hi; ++j) {
    const uint32_t cg = (c->g0 + j) * HEXKL_ACC_TILE_COLS; /* gate column */
    const uint32_t cu = c->inter + cg;                     /* its up column */
    const int32_t *gt =
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
    const int32_t *ut =
      (const int32_t *)(c->tiles_base +
                        (size_t)(up_pairs + j) * c->tile_stride);
    /** Loop-invariant per tile, exactly as hvx_dequant_acc_tile_to_f32
       hoists them. */
    const HVX_Vector csg =
      planar ? ((const HVX_UVector *)(c->colsum_w + cg))[0]
             : Q6_Vsf_equals_Vw(((const HVX_UVector *)(c->colsum_w + cg))[0]);
    const HVX_Vector vwg = ((const HVX_UVector *)(c->w_scale + cg))[0];
    const HVX_Vector vbg = ((const HVX_UVector *)(c->bias + cg))[0];
    const HVX_Vector csu =
      planar ? ((const HVX_UVector *)(c->colsum_w + cu))[0]
             : Q6_Vsf_equals_Vw(((const HVX_UVector *)(c->colsum_w + cu))[0]);
    const HVX_Vector vwu = ((const HVX_UVector *)(c->w_scale + cu))[0];
    const HVX_Vector vbu = ((const HVX_UVector *)(c->bias + cu))[0];
    const HVX_Vector offg =
      offset ? ((const HVX_UVector *)(offset + cg))[0] : Q6_V_vzero();
    const HVX_Vector offu =
      offset ? ((const HVX_UVector *)(offset + cu))[0] : Q6_V_vzero();

    /** Four rows an iteration: the GLU is ~50 dependent HVX ops a row, so
       one row in flight stalls on latency the whole way (DQ_TILE_ROW's
       reason); four independent chains in one block let them interleave.
       Each row's operations are unchanged. */
#define DQ_GLU_ROW(i)                                                          \
  HVX_Vector ga##i =                                                           \
    planar                                                                     \
      ? g24.v##i                                                               \
      : ((const HVX_UVector *)(gt + (size_t)(m + (i)) * c->row_stride))[0];    \
  HVX_Vector ua##i =                                                           \
    planar                                                                     \
      ? u24.v##i                                                               \
      : ((const HVX_UVector *)(ut + (size_t)(m + (i)) * c->row_stride))[0];    \
  if (offset) {                                                                \
    const HVX_Vector sum##i = Q6_V_vsplat_R(sums[m + (i)]);                    \
    ga##i = Q6_Vw_vmpyieacc_VwVwVh(ga##i, sum##i, offg);                       \
    ua##i = Q6_Vw_vmpyieacc_VwVwVh(ua##i, sum##i, offu);                       \
  }                                                                            \
  const HVX_Vector g##i = (planar ? dq_acc_i24_sf : dq_acc_sf)(                \
    ga##i, c->act_scale[m + (i)], c->act_zp[m + (i)], csg, vwg, vbg);          \
  const HVX_Vector u##i = (planar ? dq_acc_i24_sf : dq_acc_sf)(                \
    ua##i, c->act_scale[m + (i)], c->act_zp[m + (i)], csu, vwu, vbu)
#define DQ_GLU_STORE(i, fn)                                                    \
  ((HVX_UVector *)(c->dst + (size_t)(m + (i)) * c->dst_stride + cg))[0] =      \
    fn(g##i, u##i)
#define DQ_GLU_PREP4()                                                         \
  i24_rows4 g24, u24;                                                          \
  if (planar) {                                                                \
    g24 = i24_load4((const uint8_t *)gt, m);                                   \
    u24 = i24_load4((const uint8_t *)ut, m);                                   \
  }
#define DQ_GLU_PREP1()                                                         \
  i24_rows4 g24, u24;                                                          \
  if (planar) {                                                                \
    g24.v0 = i24_load_row((const uint8_t *)gt, m);                             \
    u24.v0 = i24_load_row((const uint8_t *)ut, m);                             \
  }
    uint32_t m = 0;
    if (c->act == HVX_GLU_GELU_TANH) {
      for (; m + 4u <= c->m_count; m += 4u) {
        DQ_GLU_PREP4();
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
        DQ_GLU_PREP1();
        DQ_GLU_ROW(0);
        DQ_GLU_STORE(0, hvx_geglu_det_sf);
      }
    } else {
      for (; m + 4u <= c->m_count; m += 4u) {
        DQ_GLU_PREP4();
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
        DQ_GLU_PREP1();
        DQ_GLU_ROW(0);
        DQ_GLU_STORE(0, hvx_swiglu_det_sf);
      }
    }
#undef DQ_GLU_ROW
#undef DQ_GLU_STORE
#undef DQ_GLU_PREP4
#undef DQ_GLU_PREP1
  }
}

static inline __attribute__((always_inline)) void
dq_swiglu(uint32_t n_threads, uint32_t i, const hvx_dq_swiglu_job *c,
          const int32_t *offset, const int32_t *sums, int planar) {
  const uint32_t lo = (uint32_t)((uint64_t)c->n_pairs * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->n_pairs * (i + 1u) / n_threads);
  dq_swiglu_span(c, offset, sums, planar, c->n_pairs, lo, hi);
}

void hvx_dq_swiglu_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  dq_swiglu(n_threads, i, (const hvx_dq_swiglu_job *)vctx, NULL, NULL, 0);
}

void hvx_dq_swiglu_affine(uint32_t n_threads, uint32_t i,
                          const hvx_dq_swiglu_job *job, const int32_t *offset,
                          const int32_t *sums) {
  dq_swiglu(n_threads, i, job, offset, sums, 0);
}

/** Keep the arithmetic kernel shared by full-tile and remainder-row spans. */
static __attribute__((noinline)) void
dq_swiglu_i24_span(const hvx_dq_swiglu_job *job, uint32_t up_pairs, uint32_t lo,
                   uint32_t hi) {
  dq_swiglu_span(job, NULL, NULL, 1, up_pairs, lo, hi);
}

void hvx_dq_swiglu_i24(uint32_t n_threads, uint32_t i,
                       const hvx_dq_swiglu_job *job) {
  const uint32_t lo = (uint32_t)((uint64_t)job->n_pairs * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)job->n_pairs * (i + 1u) / n_threads);
  dq_swiglu_i24_span(job, job->n_pairs, lo, hi);
}

void hvx_dq_swiglu_i24_balanced(uint32_t n_threads, uint32_t i,
                                const hvx_dq_swiglu_job *job) {
  const uint32_t whole = job->n_pairs / n_threads;
  const uint32_t full_pairs = whole * n_threads;
  if (n_threads <= 1u || job->m_count < 32u || job->m_count > 64u ||
      whole == 0u || full_pairs == job->n_pairs) {
    const uint32_t lo = (uint32_t)((uint64_t)job->n_pairs * i / n_threads);
    const uint32_t hi =
      (uint32_t)((uint64_t)job->n_pairs * (i + 1u) / n_threads);
    dq_swiglu_i24_span(job, job->n_pairs, lo, hi);
    return;
  }
  /** Full pairs divide evenly. Only the remainder pairs share their rows;
   * each four-row group still loads three complete aligned byte planes.
   * For 16 pairs and 7 workers, a full 64-row job's longest slice drops from
   * 192 row vectors to 152, without repeating any output or operation. */
  dq_swiglu_i24_span(job, job->n_pairs, whole * i, whole * (i + 1u));
  const uint32_t groups = (job->m_count + 3u) / 4u;
  const uint32_t m0 = 4u * (uint32_t)((uint64_t)groups * i / n_threads);
  uint32_t m1 = 4u * (uint32_t)((uint64_t)groups * (i + 1u) / n_threads);
  if (m1 > job->m_count)
    m1 = job->m_count;
  if (m0 >= m1)
    return;
  hvx_dq_swiglu_job part = *job;
  part.tiles_base += (size_t)full_pairs * job->tile_stride + m0 * 32u;
  part.g0 += full_pairs;
  part.m_count = m1 - m0;
  part.act_scale += m0;
  part.act_zp += m0;
  part.dst += (size_t)m0 * part.dst_stride;
  dq_swiglu_i24_span(&part, job->n_pairs, 0u, job->n_pairs - full_pairs);
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
