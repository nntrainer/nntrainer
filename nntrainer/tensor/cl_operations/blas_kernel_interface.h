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
 */
void copyCl(const Tensor &input, Tensor &result);

/**
 * @brief nrm2 computation : Euclidean norm
 * @param input Tensor
 * @return Euclidean norm
 * @note This function is used to compute the Euclidean norm of a vector.
 */
float nrm2Cl(const Tensor &input);

/**
 * @brief Absolute sum computation
 *
 * @param input Tensor
 * @return float absolute sum of the elements
 */
float asumCl(const Tensor &input);

/**
 * @brief Absolute max computation
 *
 * @param input Tensor
 * @return int index of the maximum absolute value
 * @note Not necessarily the first if there are multiple maximums.
 */
int amaxCl(const Tensor &input);

/**
 * @brief Absolute min computation
 *
 * @param input Tensor
 * @return int index of the minimum absolute value
 * @note Not necessarily the first if there are multiple minimums.
 */
int aminCl(const Tensor &input);

/**
 * @brief v8c GPU path entry point — paper 8/4/4 (arXiv:2505.00232): int8
 *        activation × channel-wise QINT4 weight GEMM. Default-on for the GPU
 *        FC dispatch; NNTR_FC_INT8_GPU=0 disables. Caller falls back to the
 *        generic host path on false.
 * @param[in] input fp32 or fp16 activation tensor [M, K]
 * @param[in] weight channel-wise QINT4 (QS4CX) weight tensor [K, N]
 * @param[out] output fp32 or fp16 tensor [M, N] (preallocated)
 * @return true if the v8c path executed; false if not applicable
 *         (env disabled, weight not int4, shape misaligned).
 */
bool dotCl_v8c(const Tensor &input, const Tensor &weight, Tensor &output);

/**
 * @brief Eagerly build the v8c GPU weight entry (nibble permute + upload +
 *        image view) for a freshly READ int4 FC weight, so the first prefill
 *        does not pay the lazy per-weight build. Called by the CL FC layer
 *        after the base read. Returns false (no-op) off the v8c path (env
 *        unset / non-int4 / unsupported shape); the lazy build in dotCl_v8c
 *        still covers those.
 */
bool dotCl_v8c_prebuild_weight(const Tensor &weight);

/**
 * @brief Eager v8c weight build whose nibbles come from @p src_nibbles rather
 *        than from the tensor's own storage.
 *
 * @details The zero-copy weight load: the loader hands the weight file's
 * mapping straight to the device build, so the plain payload is never copied
 * into the tensor at all. The tensor is still the cache identity (its address
 * is the key) and still owns the per-channel scales, which the caller has
 * read; only the nibbles are read from @p src_nibbles.
 *
 * On success the tensor's payload holds nothing meaningful and must not be
 * read by a host consumer -- the same contract the post-build DROP_PLAIN
 * release establishes, which is why this path is tied to that lever. On
 * failure nothing has changed and the caller must read the payload.
 *
 * @param[in] weight the QS4CX weight tensor (identity + scales)
 * @param[in] src_nibbles N*ceil(K/2) plain nibble bytes for this weight
 * @return true when the device backing was built from @p src_nibbles
 */
bool dotCl_v8c_prebuild_weight_from(const Tensor &weight,
                                    const void *src_nibbles);

} // namespace nntrainer
#endif /* __BLAS_KERNEL_INTERFACE_H__ */
