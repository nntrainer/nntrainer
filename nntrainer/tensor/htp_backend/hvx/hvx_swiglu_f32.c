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
 * [A1] This runs hvx_swiglu_det.h's specification, not an approximation
 * chosen for accuracy. The host side of the MoE path runs swiglu_det.h,
 * which is the same specification operation for operation, so the two
 * produce the same bits.
 *
 * That is the whole point and it is not about precision. The result feeds a
 * uint8 requantization two stages later, and two independently-rounded
 * pipelines -- however accurate each one is -- eventually land an element
 * on opposite sides of a quantization boundary. Measured on this model
 * (doc 43 §7): one element of 1792 flips on 5 of 32 expert calls, and the
 * down matmul spreads it across every output column. The earlier version
 * here used hvx_exp_sf plus a qf32 Newton-Raphson reciprocal, both well
 * inside their ~1e-6 spec, and that was exactly the problem.
 *
 * So the gate on this file is bit equality (HvxSwigluDet.MatchesScalarBitExact
 * and SwigluDetNeon.MatchesScalar), not SNR. Do not "improve" the arithmetic
 * below without changing swiglu_det.h in the same commit.
 */

#include "hvx_swiglu_f32.h"

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_swiglu_det.h"

#include "swiglu_det.h"

/** @brief f32 lanes per HVX vector at 128B. Kept in sync with
 *         hvx_dequant_i32.c's LANES by inspection, the same file-scoped
 *         constant practice as hexkl_mm_u8i4_dma.c's tile byte counts. */
#define LANES 32u

/** @brief Swiglus one row in place: gate[j] = silu(gate[j]) * up[j]. */
static void hvx_swiglu_row_f32(float *gate, const float *up, uint32_t n_out) {
  uint32_t j = 0;
  for (; j + LANES <= n_out; j += LANES) {
    const HVX_Vector g = ((const HVX_UVector *)(gate + j))[0];
    const HVX_Vector u = ((const HVX_UVector *)(up + j))[0];
    ((HVX_UVector *)(gate + j))[0] = hvx_swiglu_det_sf(g, u);
  }
  /* The tail is the same specification in scalar form -- the shared header,
     not expf. A tail computed by a different approximation would reintroduce
     exactly the divergence this path exists to remove, on whatever columns
     happen to fall in it. (This model's n_out is 1792 = 56*32, so the tail
     is empty here; that is not a reason to leave it wrong.) */
  for (; j < n_out; ++j) {
    gate[j] = swiglu_det_one(gate[j], up[j]);
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
