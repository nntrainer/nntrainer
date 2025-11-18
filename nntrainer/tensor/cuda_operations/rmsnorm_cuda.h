// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	rmsnorm_cuda.h
 * @date	14 Nov 2025
 * @brief	Common blas CUDA kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Samsung Electronics Co., Ltd.
 * @bug		No known bugs except for NYI items
 *
 */

#pragma once

namespace nntrainer {

/**
 * @brief rmsnorm each row of the tensor
 * @param[in] input float * for input
 * @param[in] gamma float * for gamma multiplier for each row
 * @param[in] result float * for result
 * @param[in] epsilon epsilon to add to each row sum to prevent division by zero
 * @param[in] height height of the tensor
 * @param[in] width width of the tensor
 */
void rmsnorm_cuda(const float *input, const float *gamma, float *result,
                  const float epsilon, unsigned int height, unsigned int width);

/**
 * @brief     sscal value element by element immediately
 * @param[in] X float * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cuda(float *X, const unsigned int N, const float alpha);

} // namespace nntrainer
