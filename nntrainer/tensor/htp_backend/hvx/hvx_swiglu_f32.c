// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_swiglu_f32.c
 * @date   08 Sep 2026
 * @brief  In-place SwiGLU over f32 rows for the fused MoE FFN path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * out = silu(gate) * up = gate / (1 + exp(-gate)) * up, row by row. The
 * fused layer (hexkl_mm_u8i4_fused_run) keeps gate_up's dequantized output
 * in VTCM and runs this on it before requantizing for the down matmul --
 * the intermediate never touches DDR.
 *
 * The reference (avx2_impl.cpp's swiglu) divides; HVX has no f32 divide, so
 * sigmoid comes from exp plus a magic-seed + Newton-Raphson reciprocal.
 * Both exp (hvx_exp_sf) and the reciprocal hold ~1e-6 relative error, and
 * the result feeds a uint8 requantization two stages later, so the fused
 * path is gated on SNR against the unfused path (doc 43 §[L2]), not on
 * bit equality.
 */

#include "hvx_swiglu_f32.h"

#include <math.h>
#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_exp_f32.h"

/** @brief f32 lanes per HVX vector at 128B. Kept in sync with
 *         hvx_dequant_i32.c's LANES by inspection, the same file-scoped
 *         constant practice as hexkl_mm_u8i4_dma.c's tile byte counts. */
#define LANES 32u

/**
 * @brief qf32 reciprocal: 1/a, magic-seed estimate + 3 Newton-Raphson
 *        steps, all in the qf32 extended format.
 *
 * The seed 0x7EF311C2 - bits(a) is the reciprocal counterpart of the
 * fast inverse square root constant; it lands within ~6% for every normal
 * input, and each NR step y <- y*(2 - a*y) doubles the correct bits, so
 * three steps end past f32's 24-bit mantissa. The whole chain stays in
 * qf32 -- the same reason hvx_exp_sf runs its Horner recurrence there.
 *
 * @param[in] aq  qf32 lanes of a; must be nonzero and finite
 * @return qf32 lanes of 1/a
 */
static inline HVX_Vector hvx_recip_qf32(HVX_Vector aq) {
  const HVX_Vector zero = Q6_V_vzero();
  /** Vsf/Vw handoff is a free reinterpretation (hvx_convert.h's note). */
  const HVX_Vector seed =
    Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C2u), Q6_Vsf_equals_Vqf32(aq));
  const HVX_Vector two = Q6_Vqf32_vadd_VsfVsf(hvx_splat_sf(2.0f), zero);
  HVX_Vector y = Q6_Vqf32_vadd_VsfVsf(seed, zero);
  for (int it = 0; it < 3; ++it) {
    y = Q6_Vqf32_vmpy_Vqf32Vqf32(
      y, Q6_Vqf32_vsub_Vqf32Vqf32(two, Q6_Vqf32_vmpy_Vqf32Vqf32(aq, y)));
  }
  return y;
}

/** @brief Swiglus one row in place: gate[j] = silu(gate[j]) * up[j]. */
static void hvx_swiglu_row_f32(float *gate, const float *up, uint32_t n_out) {
  const HVX_Vector zero = Q6_V_vzero();
  /** hvx_exp_sf is undefined above 88.7 and has no overflow clamp; -gate
   *  past 88 only happens for very negative gates, where sigmoid is 1e-38
   *  or smaller either way. Clamping keeps exp in its documented domain
   *  and the product at "effectively zero", where the reference lands too
   *  (its exp overflows to inf and 1/(1+inf) == 0). */
  const HVX_Vector exp_top = hvx_splat_sf(88.0f);

  uint32_t j = 0;
  for (; j + LANES <= n_out; j += LANES) {
    const HVX_Vector g = ((const HVX_UVector *)(gate + j))[0];
    const HVX_Vector u = ((const HVX_UVector *)(up + j))[0];
    HVX_Vector t = Q6_Vsf_vsub_VsfVsf(zero, g); /* -g */
    t = Q6_Vsf_vmin_VsfVsf(t, exp_top);
    const HVX_Vector e = hvx_exp_sf(t); /* exp(-g), Vsf */
    const HVX_Vector sig =
      hvx_recip_qf32(Q6_Vqf32_vadd_VsfVsf(hvx_splat_sf(1.0f), e));
    const HVX_Vector out = Q6_Vqf32_vmpy_Vqf32Vqf32(
      Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_VsfVsf(g, zero), sig),
      Q6_Vqf32_vadd_VsfVsf(u, zero));
    ((HVX_UVector *)(gate + j))[0] = Q6_Vsf_equals_Vqf32(out);
  }
  for (; j < n_out; ++j) {
    const float g = gate[j];
    /* expf overflow -> inf -> sigmoid 0, underflow -> 0 -> sigmoid 1: the
       IEEE edges land where the reference does. */
    gate[j] = g / (1.0f + expf(-g)) * up[j];
  }
}

/** @brief hvx_worker_pool_func body: rows are independent, so the pool
 *         splits by row range -- the same contiguous lo/hi split the bake
 *         and quant workers use. */
typedef struct {
  float *gate;
  const float *up;
  uint32_t rows; /**< m_valid, == the n_units the pool run was given */
  uint32_t n_out;
} hvx_swiglu_ctx;

static void hvx_swiglu_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  hvx_swiglu_ctx *c = (hvx_swiglu_ctx *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->rows * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->rows * (i + 1) / n_threads);
  for (uint32_t r = lo; r < hi; ++r) {
    hvx_swiglu_row_f32(c->gate + (size_t)r * c->n_out,
                       c->up + (size_t)r * c->n_out, c->n_out);
  }
}

void hvx_swiglu_inplace_f32(float *gate, const float *up, uint32_t m_valid,
                            uint32_t n_out, hvx_worker_pool *pool) {
  if (!gate || !up || m_valid == 0 || n_out == 0) {
    return;
  }
  hvx_swiglu_ctx c = {gate, up, m_valid, n_out};
  hvx_worker_pool_run(pool, hvx_swiglu_worker, &c, m_valid);
}
