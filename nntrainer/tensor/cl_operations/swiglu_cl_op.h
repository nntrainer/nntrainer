// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   swiglu_cl_op.h
 * @date   29 June 2026
 * @brief  OpenCL SwiGLU whole-op kernel dispatch (silu(gate) * up).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details Relocated verbatim from SwiGLULayerCl::swigluProcess so the SwiGLU
 * layer collapses into a single backend-neutral Layer that dispatches through
 * ComputeOps (ClComputeOps::swiglu forwards here). The CL kernels (swiglu_cl /
 * swiglu_cl_fp16) are registered once at ClContext init via
 * registerSwiGLUClKernels(). The residency logic (cl_mem/SVM binding, the
 * all-cl_mem decode live-row path) is unchanged. [T7]
 */

#ifndef __SWIGLU_CL_OP_H__
#define __SWIGLU_CL_OP_H__

#include <cl_context.h>

namespace nntrainer {

class Tensor;

/**
 * @brief Register the OpenCL SwiGLU kernels (swiglu_cl, swiglu_cl_fp16).
 *        Called once from ClContext::add_default_object().
 */
bool registerSwiGLUClKernels(ClContext &cl_context);

/**
 * @brief out = silu(in1) * in2 over rows [row_offset, row_offset +
 * active_rows). Binds the planner-chosen residency (cl_mem sub-buffer or SVM)
 * of each tensor directly. ClComputeOps::swiglu forwards here.
 */
void swiglu_cl_op(const Tensor &in1, const Tensor &in2, Tensor &out,
                  unsigned int active_rows, unsigned int row_offset);

/**
 * @brief Raw-pointer fp32 swiglu kernel dispatch (gate, up -> out). The layer
 *        path uses swiglu_cl_op() above; this lower-level entry is kept exposed
 *        for the OpenCL kernel micro-benchmarks in test/unittest.
 */
void swiglu_cl(float *gate, float *up, float *out, unsigned int dim1,
               unsigned int dim2, bool svm = true);

} // namespace nntrainer

#endif // __SWIGLU_CL_OP_H__
