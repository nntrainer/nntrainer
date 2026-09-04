// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file hvx_softmax_f32.h
 * @date 05 Aug 2026
 * @brief Row-wise f32 softmax on HVX
 * @see https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug No known bugs except for NYI items */

#ifndef __NNTRAINER_HVX_SOFTMAX_F32_H__
#define __NNTRAINER_HVX_SOFTMAX_F32_H__

#include <stdint.h>

/**
 * @brief y[m][*] = softmax(x[m][*] * scale) for rows [m_first, m_last).
 *
 * Rows are dense with stride @a k and are independent of each other, so
 * this knows nothing about threads: calling it with (0, m) is the whole
 * matrix, and a worker pool can hand each thread its own row range. The
 * range shape mirrors hvx_dequant_i32_to_f32.
 *
 * @a y may alias @a x.
 *
 * @param[in] x m_last*k floats, row-major. No alignment requirement
 * @param[out] y same shape as @a x; may be the same pointer
 * @param[in] m_first first row to process
 * @param[in] m_last one past the last row
 * @param[in] k row length, at least 1
 * @param[in] scale folded into x before the softmax */
void hvx_softmax_rows_f32(const float *x, float *y, uint32_t m_first,
                          uint32_t m_last, uint32_t k, float scale);

#endif /* __NNTRAINER_HVX_SOFTMAX_F32_H__ */
