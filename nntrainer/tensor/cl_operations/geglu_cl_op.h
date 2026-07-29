// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   geglu_cl_op.h
 * @date   29 June 2026
 * @brief  OpenCL GeGLU whole-op kernel dispatch (gelu_tanh(gate) * up).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details Relocated verbatim from GeGLULayerCl::gegluProcess so the GeGLU
 * layer collapses into a single backend-neutral Layer that dispatches through
 * ComputeOps (ClComputeOps::geglu forwards here). The CL kernels (geglu_cl /
 * geglu_cl_fp16) are registered once at ClContext init via
 * registerGeGLUClKernels(). The residency logic (cl_mem/SVM binding, the
 * resident-act overlay, the all-cl_mem row_off decode path) is unchanged.
 */

#ifndef __GEGLU_CL_OP_H__
#define __GEGLU_CL_OP_H__

#include <cl_context.h>

namespace nntrainer {

class Tensor;

/**
 * @brief Register the OpenCL GeGLU kernels (geglu_cl, geglu_cl_fp16).
 *        Called once from ClContext::add_default_object().
 */
bool registerGeGLUClKernels(ClContext &cl_context);

/**
 * @brief out = gelu_tanh(in1) * in2 over rows [row_offset, row_offset +
 *        active_rows). Binds the planner-chosen residency (cl_mem sub-buffer
 *        or SVM) of each tensor directly. ClComputeOps::geglu forwards here.
 */
void geglu_cl_op(const Tensor &in1, const Tensor &in2, Tensor &out,
                 unsigned int active_rows, unsigned int row_offset);

} // namespace nntrainer

#endif // __GEGLU_CL_OP_H__
