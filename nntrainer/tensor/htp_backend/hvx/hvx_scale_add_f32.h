// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_scale_add_f32.h
 * @date   10 Sep 2026
 * @brief  dst += src * scale over a row, on HVX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_SCALE_ADD_F32_H__
#define __NNTRAINER_HVX_SCALE_ADD_F32_H__

#include <stdint.h>

/**
 * @brief dst[i] += src[i] * scale, for i in [0, n).
 *
 * The MoE layer's routing multiply and scatter-add. It was a scalar loop
 * and measured 108 ms/layer against the 1.9 ms hvx_swiglu_inplace_f32
 * spends on a comparable element count while also computing an exponential
 * and a reciprocal -- about 30 cycles an element against well under one.
 * The DSP's scalar float unit is why HVX exists; a per-element float loop
 * on it is not a small inefficiency.
 *
 * The multiply and the add are separate operations and stay separate: HVX
 * has no f32 fused multiply-add, and neither does the ARM path this has to
 * agree with, which applies the routing weight with multiply_i and then
 * accumulates with add_i. A fused version would round once where both of
 * them round twice.
 *
 * @param[in,out] dst  n floats, accumulated into. Any alignment.
 * @param[in]     src  n floats. Any alignment.
 */
void hvx_scale_add_rows_f32(float *dst, const float *src, float scale,
                            uint32_t n);

#endif /* __NNTRAINER_HVX_SCALE_ADD_F32_H__ */
