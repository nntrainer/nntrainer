// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gelu_cl_op.h
 * @date   28 July 2026
 * @brief  OpenCL GELU whole-op kernel dispatch (elementwise activation).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details out[i] = gelu(in[i]). It lives here rather than in a Layer so the
 * backend-neutral ActivationLayer covers the GPU too: ClComputeOps::activation
 * forwards here for the gelu and tanh_gelu modes and throws for the rest.
 * Same two-symbol shape as the other whole-ops in this directory.
 */

#ifndef __GELU_CL_OP_H__
#define __GELU_CL_OP_H__

#include <cl_context.h>

namespace nntrainer {

class Tensor;

/**
 * @brief Register the OpenCL GELU kernels (gelu_cl, gelu_cl_fp16). Called once
 *        from ClContext::add_default_object().
 * @param cl_context the OpenCL context to register on
 * @return true when every kernel was registered
 */
bool registerGeluClKernels(ClContext &cl_context);

/**
 * @brief out = gelu(in) over rows [row_offset, row_offset + active_rows).
 * @param in input tensor
 * @param out output tensor
 * @param mode 0 = erf-based exact GELU (ACT_GELU), 1 = tanh approximation
 *             (ACT_TANH_GELU). The ActivationType to mode mapping lives in
 *             ClComputeOps::activation, not in a Layer -- which mode a
 *             backend can serve is a backend concern.
 * @param active_rows number of rows to process
 * @param row_offset first row to process
 */
void gelu_cl_op(const Tensor &in, Tensor &out, int mode,
                unsigned int active_rows, unsigned int row_offset);

} // namespace nntrainer

#endif // __GELU_CL_OP_H__
