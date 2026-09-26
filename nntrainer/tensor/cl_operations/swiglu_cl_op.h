// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   swiglu_cl_op.h
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL SwiGLU whole-op kernel dispatch (silu(gate) * up).
 *
 * @details The two symbols below are the shape every OpenCL whole-op in this
 * directory has: a one-time kernel registration called from
 * ClContext::add_default_object(), and the op itself, which
 * ClComputeOps::swiglu forwards to. No Layer is involved -- the layer that
 * consumes this op is backend-neutral and reaches it through the tensor's
 * ComputeOps table.
 */

#ifndef __SWIGLU_CL_OP_H__
#define __SWIGLU_CL_OP_H__

#include <cl_context.h>

namespace nntrainer {

class Tensor;

/**
 * @brief Register the OpenCL SwiGLU kernels. Called once from
 *        ClContext::add_default_object().
 * @param cl_context the OpenCL context to register on
 * @return true when every kernel was registered
 */
bool registerSwiGLUClKernels(ClContext &cl_context);

/**
 * @brief Raw SwiGLU kernel dispatch, exposed for the OpenCL kernel
 *        micro-benchmarks. The layer path goes through swiglu_cl_op() below.
 * @param in1 gate operand
 * @param in2 second operand
 * @param out result
 * @param dim1 row count
 * @param dim2 row width
 * @param svm bind the pointers directly rather than staging them
 */
void swiglu_cl(const float *in1, const float *in2, float *out,
               unsigned int dim1, unsigned int dim2, bool svm);

/**
 * @brief out = silu(gate) * up over rows [row_offset, row_offset +
 * active_rows).
 * @param in1 gate operand
 * @param in2 second operand
 * @param out result
 * @param active_rows number of rows to process
 * @param row_offset first row to process
 */
void swiglu_cl_op(const Tensor &in1, const Tensor &in2, Tensor &out,
                  unsigned int active_rows, unsigned int row_offset);

} // namespace nntrainer

#endif // __SWIGLU_CL_OP_H__
