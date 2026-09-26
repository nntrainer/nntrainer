// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   geglu_cl_op.h
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL GeGLU whole-op kernel dispatch (gelu_tanh(gate) * up).
 *
 * @details The two symbols below are the shape every OpenCL whole-op in this
 * directory has: a one-time kernel registration called from
 * ClContext::add_default_object(), and the op itself, which
 * ClComputeOps::geglu forwards to. No Layer is involved -- the layer that
 * consumes this op is backend-neutral and reaches it through the tensor's
 * ComputeOps table.
 */

#ifndef __GEGLU_CL_OP_H__
#define __GEGLU_CL_OP_H__

#include <cl_context.h>

namespace nntrainer {

class Tensor;

/**
 * @brief Register the OpenCL GeGLU kernels. Called once from
 *        ClContext::add_default_object().
 * @param cl_context the OpenCL context to register on
 * @return true when every kernel was registered
 */
bool registerGeGLUClKernels(ClContext &cl_context);

/**
 * @brief out = gelu_tanh(gate) * up over rows [row_offset, row_offset +
 * active_rows).
 * @param in1 gate operand
 * @param in2 second operand
 * @param out result
 * @param active_rows number of rows to process
 * @param row_offset first row to process
 */
void geglu_cl_op(const Tensor &in1, const Tensor &in2, Tensor &out,
                 unsigned int active_rows, unsigned int row_offset);

} // namespace nntrainer

#endif // __GEGLU_CL_OP_H__
