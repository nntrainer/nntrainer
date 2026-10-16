// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_conv_gate_f32.h
 * @date   22 Sep 2026
 * @brief  Causal depthwise conv1d (L=3) folded into the conv block's gate
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The LFM2 conv block's middle (doc 51 section 2.1): y = conv1d(a * c),
 * z = b * y. With a * c already in g and b already dequantized into z,
 * this is one pass: z[r] *= w0 * g[t] + w1 * g[t-1] + w2 * g[t-2] with
 * t = t0 + r, rows before 0 taken as zero -- what
 * causal_depthwise_conv1d_k3 computes on the CPU, minus its FMA: HVX has
 * no f32 fused multiply-add, so the sum is two multiplies, an add, a
 * multiply and an add, in that order, and the host stand-in keeps the
 * same order (the ARM side never sees these values, it sees the block's
 * output, so there is no bit-identity contract with the CPU kernel to
 * keep, only one between this and its host check).
 */

#ifndef __NNTRAINER_HVX_CONV_GATE_F32_H__
#define __NNTRAINER_HVX_CONV_GATE_F32_H__

#include <stdint.h>

#include "hvx_worker_pool.h"

/**
 * @brief z[r][:] *= conv1d(g)[t0 + r][:] for r in [0, m_count).
 *
 * @param z        [m_count x C] f32 at @a z_stride floats a row (VTCM)
 * @param g        the conv input from row 0, @a g_stride floats a row;
 *                 rows t0 - 2 .. t0 + m_count - 1 are read where >= 0
 * @param t0       the row of g that z's row 0 corresponds to
 * @param C        columns, a multiple of 32
 * @param conv_w   [3 x C] f32: w0 (current row), w1 (t-1), w2 (t-2)
 * @param pool     rows are split across it; NULL runs on the caller
 */
void hvx_conv_gate_f32(float *z, uint32_t z_stride, const float *g,
                       uint32_t g_stride, uint32_t t0, uint32_t m_count,
                       uint32_t C, const float *conv_w, hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_CONV_GATE_F32_H__ */
