// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   rope_reference.h
 * @date   27 January 2025
 * @brief  Reference implementation for RoPE (Rotary Positional Embedding)
 * @see    https://github.com/nnstreamer/nntrainer
 * @author [Your Name] <[Your Email]>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __ROPE_REFERENCE_H__
#define __ROPE_REFERENCE_H__

namespace nntrainer {

/**
 * @brief Compute rotary embedding value reference implementation
 *
 * This function applies Rotary Positional Embedding (RoPE) to a tensor.
 * The input tensor is expected to have the last dimension equal to `dim`.
 * It iterates through the tensor in chunks of `dim`, applying rotation to
 * paired elements (i, i + dim/2) within each chunk.
 *
 * @param output Output buffer (can be nullptr for in-place operation)
 * @param width Total number of elements in the tensor (Batch * Seq * Heads *
 * Dim). Must be a multiple of dim.
 * @param dim The dimension size of a single head (the size of the inner-most
 * dimension).
 * @param half_ Half of the dimension (dim / 2)
 * @param inout Input and output buffer (in-place if output is nullptr)
 * @param cos_ Cosine frequency values (size: half_)
 * @param sin_ Sine frequency values (size: half_)
 * @param only_convert_to_fp16 If true, only convert to FP16 without applying
 * RoPE
 */
void rotary_embedding_avx2_ref(void *output, unsigned int width,
                               unsigned int dim, unsigned int half_,
                               float *inout, const float *cos_,
                               const float *sin_, bool only_convert_to_fp16);

/**
 * @brief Compute rotary embedding value reference implementation (scalar
 * version)
 *
 * This function applies Rotary Positional Embedding (RoPE) to a tensor.
 * The input tensor is expected to have the last dimension equal to `dim`.
 * It iterates through the tensor in chunks of `dim`, applying rotation to
 * paired elements (i, i + dim/2) within each chunk.
 *
 * @param output Output buffer (can be nullptr for in-place operation)
 * @param width Total number of elements in the tensor (Batch * Seq * Heads *
 * Dim). Must be a multiple of dim.
 * @param dim The dimension size of a single head (the size of the inner-most
 * dimension).
 * @param half_ Half of the dimension (dim / 2)
 * @param inout Input and output buffer (in-place if output is nullptr)
 * @param cos_ Cosine frequency values (size: half_)
 * @param sin_ Sine frequency values (size: half_)
 * @param only_convert_to_fp16 If true, only convert to FP16 without applying
 * RoPE
 */
void rotary_embedding_ref(void *output, unsigned int width, unsigned int dim,
                          unsigned int half_, float *inout, const float *cos_,
                          const float *sin_, bool only_convert_to_fp16);

} // namespace nntrainer

#endif // __ROPE_REFERENCE_H__
