// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_interface.h
 * @date	5 June 2024
 * @brief	Interface for blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNEL_INTERFACE_H__
#define __BLAS_KERNEL_INTERFACE_H__

#include <string>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
Tensor dotCl(Tensor const &input, Tensor const &m, bool trans = false,
             bool trans_m = false);

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] result Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
void dotCl(Tensor const &input, Tensor const &m, Tensor &result,
           bool trans = false, bool trans_m = false);

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] result Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
void dotBatchedCl(Tensor const &input, Tensor const &m, Tensor &result,
                  bool trans = false, bool trans_m = false);

/**
 * @brief Multiply value element by element immediately
 * @param[in] input Tensor
 * @param[in] value multiplier
 * @param[in] RunLayerContext reference
 */
void multiplyCl(Tensor &input, float const &value);

/**
 * @brief Process data and dimensions for add operation
 * @param[in] result Tensor
 * @param[in] input Tensor
 */
void add_i_cl(Tensor &result, Tensor const &input);

/**
 * @brief Process data and dimensions for transpose operation
 * @param[in] direction string
 * @param[in] input Tensor
 * @param[in] result Tensor
 */
void transposeCl(const std::string &direction, Tensor const &in,
                 Tensor &result);

/**
 * @brief Copy data from one tensor to another
 *
 * @param input Tensor
 * @param result Tensor
 * @throws exception::not_supported no OpenCL kernel backs this entry point
 */
void copyCl(const Tensor &input, Tensor &result);

/**
 * @brief nrm2 computation : Euclidean norm
 * @param input Tensor
 * @return Euclidean norm
 * @note This function is used to compute the Euclidean norm of a vector.
 * @throws exception::not_supported no OpenCL kernel backs this entry point
 */
float nrm2Cl(const Tensor &input);

/**
 * @brief Absolute sum computation
 *
 * @param input Tensor
 * @return float absolute sum of the elements
 * @throws exception::not_supported no OpenCL kernel backs this entry point
 */
float asumCl(const Tensor &input);

/**
 * @brief Absolute max computation
 *
 * @param input Tensor
 * @return int index of the maximum absolute value
 * @note Not necessarily the first if there are multiple maximums.
 * @throws exception::not_supported no OpenCL kernel backs this entry point
 */
int amaxCl(const Tensor &input);

/**
 * @brief Absolute min computation
 *
 * @param input Tensor
 * @return int index of the minimum absolute value
 * @note Not necessarily the first if there are multiple minimums.
 * @throws exception::not_supported no OpenCL kernel backs this entry point
 */
int aminCl(const Tensor &input);

} // namespace nntrainer
#endif /* __BLAS_KERNEL_INTERFACE_H__ */
