// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   layernorm_cl_op.h
 * @date   28 July 2026
 * @brief  OpenCL LayerNorm whole-op kernel dispatch.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details out = (x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta, per row
 * over the last (width) axis. It lives here rather than in blas_kernels.cpp,
 * which is the BLAS/GEMM file, and rather than in a Layer: the backend-neutral
 * LayerNormalizationLayer dispatches through ComputeOps, and
 * ClComputeOps::layer_norm forwards here. Same two-symbol shape as the other
 * whole-ops in this directory -- the kernels are registered once at ClContext
 * init via registerLayerNormClKernels().
 */

#ifndef __LAYERNORM_CL_OP_H__
#define __LAYERNORM_CL_OP_H__

#include <cl_context.h>

namespace nntrainer {

class Tensor;

/**
 * @brief Register the OpenCL LayerNorm kernels (layernorm_cl,
 *        layernorm_cl_fp16). Called once from ClContext::add_default_object().
 */
bool registerLayerNormClKernels(ClContext &cl_context);

/**
 * @brief out = (in - mean) * rsqrt(var + epsilon) * gamma + beta over rows
 *        [row_offset, row_offset + active_rows), mean/variance over width.
 *        gamma/beta are [1,1,1,width] and must share the activation dtype (the
 *        kernels are single-dtype); a mismatch throws rather than silently
 *        falling back to a host loop -- a tensor on this context may live in
 *        device memory, where a host loop is not merely slower but wrong.
 * @param in input tensor
 * @param out output tensor
 * @param gamma per-width scale
 * @param beta per-width shift
 * @param epsilon variance epsilon
 * @param active_rows number of rows to process
 * @param row_offset first row to process
 */
void layernorm_cl_op(const Tensor &in, Tensor &out, const Tensor &gamma,
                     const Tensor &beta, float epsilon,
                     unsigned int active_rows, unsigned int row_offset);

} // namespace nntrainer

#endif // __LAYERNORM_CL_OP_H__
