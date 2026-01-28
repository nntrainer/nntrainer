// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   rope.h
 * @date   27 January 2025
 * @brief  RoPE (Rotary Positional Embedding) CUDA implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jung, dk11.jung@samsung.com
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __ROPE_H__
#define __ROPE_H__

#include <cuda_runtime.h>

namespace nntrainer {

/**
 * @brief Compute rotary embedding value CUDA implementation
 *
 * This function applies Rotary Positional Embedding (RoPE) to a tensor using
 * CUDA. The input tensor is expected to have the last dimension equal to `dim`.
 * It iterates through the tensor in chunks of `dim`, applying rotation to
 * paired elements (i, i + dim/2) within each chunk.
 *
 * @param output Output buffer (device pointer)
 * @param width Total number of elements in the tensor (Batch * Seq * Heads *
 * Dim). Must be a multiple of dim.
 * @param dim The dimension size of a single head (the size of the inner-most
 * dimension).
 * @param half_ Half of the dimension (dim / 2)
 * @param inout Input and output buffer (in-place if output is nullptr) (device
 * pointer)
 * @param cos_ Cosine frequency values (device pointer) (size: half_)
 * @param sin_ Sine frequency values (device pointer) (size: half_)
 * @param only_convert_to_fp16 If true, only convert to FP16 without applying
 * RoPE
 * @param stream CUDA stream (optional, default 0)
 */
void rotary_embedding_cuda(void *output, unsigned int width, unsigned int dim,
                           unsigned int half_, float *inout, const float *cos_,
                           const float *sin_, bool only_convert_to_fp16,
                           cudaStream_t stream = 0);

} // namespace nntrainer

#endif // __ROPE_H__
