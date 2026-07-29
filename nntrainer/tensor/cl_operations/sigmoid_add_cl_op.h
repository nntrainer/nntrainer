// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   sigmoid_add_cl_op.h
 * @date   29 June 2026
 * @brief  OpenCL SigmoidAdd whole-op kernel dispatch (silu(gate) * up).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details Relocated verbatim from SigmoidAddLayerCl::sigmoid_addProcess so the
 * SigmoidAdd layer collapses into a single backend-neutral Layer that
 * dispatches through ComputeOps (ClComputeOps::sigmoid_add forwards here). The
 * CL kernels (sigmoid_add_cl / sigmoid_add_cl_fp16) are registered once at
 * ClContext init via registerSigmoidAddClKernels(). The residency logic
 * (cl_mem/SVM binding, the all-cl_mem decode live-row path) is unchanged.
 */

#ifndef __SIGMOID_ADD_CL_OP_H__
#define __SIGMOID_ADD_CL_OP_H__

#include <cl_context.h>

namespace nntrainer {

class Tensor;

/**
 * @brief Register the OpenCL SigmoidAdd kernels (sigmoid_add_cl,
 * sigmoid_add_cl_fp16). Called once from ClContext::add_default_object().
 */
bool registerSigmoidAddClKernels(ClContext &cl_context);

/**
 * @brief out = silu(in1) * in2 over rows [row_offset, row_offset +
 * active_rows). Binds the planner-chosen residency (cl_mem sub-buffer or SVM)
 * of each tensor directly. ClComputeOps::sigmoid_add forwards here.
 */
void sigmoid_add_cl_op(const Tensor &in1, const Tensor &in2, Tensor &out,
                       unsigned int active_rows, unsigned int row_offset);

/**
 * @brief Raw-pointer fp32 sigmoid_add kernel dispatch (gate, up -> out). The
 * layer path uses sigmoid_add_cl_op() above; this lower-level entry is kept
 * exposed for the OpenCL kernel micro-benchmarks in test/unittest.
 */
void sigmoid_add_cl(float *gate, float *up, float *out, unsigned int dim1,
                    unsigned int dim2, bool svm = true);

} // namespace nntrainer

#endif // __SIGMOID_ADD_CL_OP_H__
