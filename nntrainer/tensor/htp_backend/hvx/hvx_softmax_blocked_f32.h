// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file hvx_softmax_blocked_f32.h
 * @date 07 Aug 2026
 * @brief Blocked masked softmax on HVX for the fused attention kernel
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs except for NYI items */

#ifndef __NNTRAINER_HVX_SOFTMAX_BLOCKED_F32_H__
#define __NNTRAINER_HVX_SOFTMAX_BLOCKED_F32_H__

#include <stdint.h>

/**
 * @brief Intersects a row's logical valid range [b, e) with block @a j.
 *
 * The result is a CONTIGUOUS local slice [lo, hi) of the block, which is the
 * whole reason the kernel's vector loops keep the dense kernel's shape: the
 * mask never has to become a per-lane predicate. lo == hi means the block
 * lies entirely outside this row's range and is skipped.
 *
 * This is the off-by-one trap in the whole block design window T-1, T and
 * T+1 all land differently here so it lives in the header, free of any HVX
 * dependency, and is tested on the host against the reference's own
 * position-in-range predicate rather than only through the device kernel. */
static inline void hvx_softmax_seg_range(uint32_t j, uint32_t T, uint32_t b,
                                         uint32_t e, uint32_t *lo,
                                         uint32_t *hi) {
  const uint32_t base = j * T;
  uint32_t l = (b > base) ? (b - base) : 0u;
  uint32_t h = (e > base) ? (e - base) : 0u;

  if (l > T) {
    l = T;
  }
  if (h > T) {
    h = T;
  }
  *lo = l;
  *hi = (h > l) ? h : l;
}

/**
 * @brief PHASE B of the fused attention loop: one masked softmax pass over a
 * block-major score band.
 *
 * This is hvx_softmax_rows_f32 adapted to what flash attention on this part
 * actually needs, and the four differences are each load-bearing:
 *
 * 1. BLOCK-MAJOR input. hexkl_mm_u8iX_layer_run returns one contiguous M x T
 * block per weight handle in call order, so row m of block j lives at
 * seg[j] + m*T. The softmax axis runs ACROSS segments. Nothing is
 * repacked into a flat [M][kv] matrix the block-major layout IS the KV
 * tiling, and a repack would cost a VTCM->VTCM pass per band.
 *
 * 2. PER-ROW valid range [begin[m], end[m]) on the logical kv axis, so causal
 * masking, the sliding window and the attention sink are all expressed as
 * one mechanism. Inside a segment the range is contiguous, which is what
 * lets the vector loop stay the same shape as the dense kernel's.
 *
 * 3. MASKED LANES ARE EXCLUDED FROM THE MAXIMUM, not merely from the sum.
 * hvx_exp_sf is UNDEFINED above +88.7 and has no overflow clamp; a masked
 * lane holds whatever the padded weight tile left in the accumulator. If
 * it skips the max pass but still reaches exp, the result is undefined * not a
 * small numerical error a tolerance would catch. Running both passes over the
 * SAME lane set is what makes x - max <= 0 true by construction.
 *
 * 4. UNNORMALIZED output plus l. There is no third pass: seg[] keeps
 * exp(s*scale - rowmax), whose row maximum is exactly 1.0f, and l_out[m]
 * carries the denominator. P.V then accumulates over the whole kv_len
 * before a single 1/l multiply at the very end. Normalizing here would
 * leave the u8 quantization of P with a row maximum of 1/l instead of 1,
 * wasting codes which matters at i4, where there are only 16 of them.
 *
 * The sink is deliberately NOT in the maximum: leaving it out is what keeps
 * max(seg[][m][]) exactly 1.0f. It is one scalar exp per row, evaluated
 * through the same hvx_exp_sf as every lane so the two cannot drift.
 *
 * The [m_first, m_last) row range exists so this drops onto
 * hvx_worker_pool_run unchanged same convention as hvx_softmax_rows_f32
 * and hvx_dequant_i32_to_f32. Rows are independent.
 *
 * IN PLACE: every seg[j] is overwritten. Positions outside a row's valid
 * range are set to exactly 0.0f, so they contribute nothing to P.V and no
 * separate mask has to survive into PHASE C.
 *
 * @param[in,out] seg n_seg pointers, each M*T floats, block-major
 * @param[in] n_seg number of KV blocks in the band
 * @param[in] T kv positions per block
 * @param[in] m_first first row of this worker's range
 * @param[in] m_last one past the last row of this worker's range
 * @param[in] M total rows in the band (row stride within a segment)
 * @param[in] scale folded into the scores before the max; 1/sqrt(head_dim)
 * @param[in] begin M entries, first valid kv position of each row
 * @param[in] end M entries, one past the last valid kv position
 * @param[in] sink NULL, or M entries; enters l_out only, never seg
 * @param[out] l_out M entries; only [m_first, m_last) are written
 * @param[out] bm_out NULL, or n_seg * M entries at bm_out[j*M + m]: the
 * maximum of the STORED values of row m within block j,
 * floored at 0. This equals exactly, not
 * approximately the rmax hvx_quant_rows_u8_params
 * would find scanning that block row afterwards, because
 * every stored value is either one of the exps this pass
 * just took the running max of, or a literal 0.0f, and
 * f32 max is order-independent. With rmin pinned at 0 by
 * p >= 0, PHASE C can hand layer_run
 * (bm/255, 0) or (1, 0) when bm is 0 and skip the
 * re-scan of P entirely, bit-identically. Only
 * [m_first, m_last) rows are written. */
void hvx_softmax_blocked_f32(float *const *seg, uint32_t n_seg, uint32_t T,
                             uint32_t m_first, uint32_t m_last, uint32_t M,
                             float scale, const uint32_t *begin,
                             const uint32_t *end, const float *sink,
                             float *l_out, float *bm_out);

#endif /* __NNTRAINER_HVX_SOFTMAX_BLOCKED_F32_H__ */
