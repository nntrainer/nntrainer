// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   sigmoid_glu_cl_op.h
 * @date   29 June 2026
 * @brief  OpenCL SigmoidGlu whole-op kernel dispatch (silu(gate) * up).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details Relocated verbatim from SigmoidGluLayerCl::sigmoid_gluProcess so the
 * SigmoidGlu layer collapses into a single backend-neutral Layer that
 * dispatches through ComputeOps (ClComputeOps::sigmoid_glu forwards here). The
 * CL kernels (sigmoid_glu_cl / sigmoid_glu_cl_fp16) are registered once at
 * ClContext init via registerSigmoidGluClKernels(). The residency logic
 * (cl_mem/SVM binding, the all-cl_mem decode live-row path) is unchanged.
 */

#ifndef __SIGMOID_GLU_CL_OP_H__
#define __SIGMOID_GLU_CL_OP_H__

#include <cl_context.h>

namespace nntrainer {

class Tensor;

/**
 * @brief Register the OpenCL SigmoidGlu kernels (sigmoid_glu_cl,
 * sigmoid_glu_cl_fp16). Called once from ClContext::add_default_object().
 */
bool registerSigmoidGluClKernels(ClContext &cl_context);

/**
 * @brief out = silu(in1) * in2 over rows [row_offset, row_offset +
 * active_rows). Binds the planner-chosen residency (cl_mem sub-buffer or SVM)
 * of each tensor directly. ClComputeOps::sigmoid_glu forwards here.
 */
void sigmoid_glu_cl_op(const Tensor &in1, const Tensor &in2, Tensor &out,
                       unsigned int active_rows, unsigned int row_offset);

/**
 * @brief Raw-pointer fp32 sigmoid_glu kernel dispatch (gate, up -> out). The
 * layer path uses sigmoid_glu_cl_op() above; this lower-level entry is kept
 * exposed for the OpenCL kernel micro-benchmarks in test/unittest.
 */
void sigmoid_glu_cl(float *gate, float *up, float *out, unsigned int dim1,
                    unsigned int dim2, bool svm = true);

} // namespace nntrainer

#endif // __SIGMOID_GLU_CL_OP_H__
