// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	addition_cuda.h
 * @date	20 Nov 2025
 * @brief	Common blas CUDA kernels for addition
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Samsung Electronics Co., Ltd.
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __ADDITION_CUDA_H__
#define __ADDITION_CUDA_H__

namespace nntrainer {

/**
 * @brief     addition : sum of all input vectors
 * @param[in] input float * for input
 * @param[in] res float * for result/output
 * @param[in] size_input number of elements in input vector
 * @param[in] size_res number of elements in result vector
 */
void addition_cuda(const float *input, float *res, unsigned int size_input,
                   unsigned int size_res);

} // namespace nntrainer

#endif /* __ADDITION_CUDA_H__ */
