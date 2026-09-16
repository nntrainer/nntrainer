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
    float *out = (c0 < c->split) ? (c->dst_a + c0)
                                 : (c->dst_b + (c0 - c->split));
    hvx_dequant_acc_tile_to_f32(tile, c->row_stride, c->m_count,
                                c->act_scale, c->act_zp, c->colsum_w + c0,
                                c->w_scale + c0, c->bias + c0, out,
                                c->dst_stride, /*accumulate=*/0);
  }
}

void hvx_dequant_acc_tiles_to_f32(const uint8_t *tiles_base,
                                  uint32_t tile_stride, uint32_t n_tiles,
                                  uint32_t nt0, uint32_t row_stride,
                                  uint32_t m_count, const float *act_scale,
                                  const int32_t *act_zp,
                                  const int32_t *colsum_w,
                                  const float *w_scale, const float *bias,
                                  float *dst_a, float *dst_b, uint32_t split,
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

/** @brief One lane out of a vector that holds the same reduction in
 *         every lane after the rotate-and-fold below. */
static inline float dq_lane0(HVX_Vector v) {
  float lanes[32];
  *(HVX_UVector *)lanes = v;
  return lanes[0];
}

void hvx_dq_swiglu_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  const hvx_dq_swiglu_job *c = (const hvx_dq_swiglu_job *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->n_pairs * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->n_pairs * (i + 1) / n_threads);
  /* Row extrema across this worker's pairs, one vector pair per row, on
     the stack: 16 KB, kept in memory across the pair loop so the per-pair
     tile constants stay hoisted. Zero is a valid start because the
     parameters fold 0 in anyway (hvx_quant_row_params_from_minmax). */
  const int track = (c->rmin != NULL) && (i < HVX_DQ_SWIGLU_MAX_PARTS);
  HVX_Vector vmn[HEXKL_ACC_TILE_ROWS], vmx[HEXKL_ACC_TILE_ROWS];
  if (track) {
    for (uint32_t m = 0; m < c->m_count; ++m) {
      vmn[m] = Q6_V_vzero();
      vmx[m] = Q6_V_vzero();
    }
  }

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

    for (uint32_t m = 0; m < c->m_count; ++m) {
      const HVX_Vector g =
        dq_row_sf(gt + (size_t)m * c->row_stride, c->act_scale[m],
                  c->act_zp[m], csg, vwg, vbg);
      const HVX_Vector u =
        dq_row_sf(ut + (size_t)m * c->row_stride, c->act_scale[m],
                  c->act_zp[m], csu, vwu, vbu);
      const HVX_Vector y = hvx_swiglu_det_sf(g, u);
      ((HVX_UVector *)(c->dst + (size_t)m * c->dst_stride + cg))[0] = y;
      if (track) {
        vmn[m] = Q6_Vsf_vmin_VsfVsf(vmn[m], y);
        vmx[m] = Q6_Vsf_vmax_VsfVsf(vmx[m], y);
      }
    }
  }
  if (track) {
    /* Fold 32 lanes to 1 the way hvx_quant_u8.c's scan does: 64, 32, 16,
       8, 4 bytes. min/max are exact, so the order is immaterial. */
    for (uint32_t m = 0; m < c->m_count; ++m) {
      HVX_Vector mn = vmn[m], mx = vmx[m];
      for (uint32_t rot = 64u; rot >= 4u; rot >>= 1) {
        mn = Q6_Vsf_vmin_VsfVsf(mn, Q6_V_vror_VR(mn, (int)rot));
        mx = Q6_Vsf_vmax_VsfVsf(mx, Q6_V_vror_VR(mx, (int)rot));
      }
      c->rmin[i * HEXKL_ACC_TILE_ROWS + m] = dq_lane0(mn);
      c->rmax[i * HEXKL_ACC_TILE_ROWS + m] = dq_lane0(mx);
    }
  }
}

void hvx_dequant_swiglu_acc_tiles_to_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_pairs,
  uint32_t g0, uint32_t row_stride, uint32_t m_count, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, uint32_t inter, float *dst, uint32_t dst_stride,
  hvx_worker_pool *pool) {
  if (!tiles_base || n_pairs == 0u || m_count == 0u) {
    return;
  }
  hvx_dq_swiglu_job c = {tiles_base, tile_stride, n_pairs,   g0,
                         row_stride, m_count,     act_scale, act_zp,
                         colsum_w,   w_scale,     bias,      inter,
                         dst,        dst_stride,  NULL,      NULL};
  hvx_worker_pool_run(pool, hvx_dq_swiglu_worker, &c, n_pairs);
}
