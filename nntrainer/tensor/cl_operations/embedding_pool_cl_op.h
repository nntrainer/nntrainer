// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding_pool_cl_op.h
 * @date   28 July 2026
 * @brief  OpenCL whole-op dispatch for the sentence-embedding pooling /
 *         normalize tail (ComputeOps::mean_rows / l2_normalize_rows).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __EMBEDDING_POOL_CL_OP_H__
#define __EMBEDDING_POOL_CL_OP_H__

#include <cl_context.h>

namespace nntrainer {

class Tensor;

/**
 * @brief Register the embedding pooling/normalize kernels on the CL context.
 * @param[in] cl_context the OpenCL context to register into
 * @return true on success
 */
bool registerEmbeddingPoolClKernels(ClContext &cl_context);

/**
 * @brief Mean over rows on the GPU: out[i] = mean over the live rows of in.
 *        FP32 only (the embedding tail runs on the FP32 pooled output).
 * @param[in] in source tensor, rows laid out along height()
 * @param[out] out destination tensor, one row of width()
 * @param[in] active_rows number of rows to reduce
 * @param[in] row_offset first row to reduce
 */
void mean_rows_cl_op(const Tensor &in, Tensor &out, unsigned int active_rows,
                     unsigned int row_offset);

/**
 * @brief Row-wise L2 normalize along the last dim on the GPU:
 *        out[r, :] = in[r, :] / max(||in[r, :]||_2, epsilon). FP32 only.
 * @param[in] in source tensor
 * @param[out] out destination tensor (may alias @p in)
 * @param[in] epsilon floor applied to the norm
 */
void l2_normalize_rows_cl_op(const Tensor &in, Tensor &out, float epsilon);

} // namespace nntrainer

#endif /* __EMBEDDING_POOL_CL_OP_H__ */
