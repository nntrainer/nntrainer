// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 *
 * @file   fallback_kleidiai.h
 * @date   15 September 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @brief  Modified computational backend components of kleidiai. Portions of
 * this file are derived from Arm Limited code licensed under the Apache
 * License, Version 2.0, with modifications
 * @bug    No known bugs except for NYI items
 * @note   Licensed under the Apache License, Version 2.0 (the "License");
 *         you may not use this file except in compliance with the License.
 *         You may obtain a copy of the License at
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * @modifications
 *   - [2025-09-15] Integrated and adapted Arm-provided code into
 *     nntrainer CPU backend
 *
 */

#include <cstdint>
#include <stddef.h>

/**
 * @brief rhs transpose indicator
 *
 */
enum class rhs_format {
  nxk,
  kxn,
};

/**
 * @brief qs4cx quantization of (n*k) matrix. Typically a weight quantization,
 * and generally regard the weight is already transposed, and quantize it as it
 * is. qs4cx refers to quantized symmetric 4-bit quantization of channelwise x
 * groups.
 *
 * @param n N length of the matrix
 * @param k K length of the matrix
 * @param format whether rhs matrix is transpoed or not
 * @param rhs_f32 matrix data before quantization to load
 * @param rhs_qs4cx matrix data after quantization to store
 * @param rhs_scales_f32  matrix quant scale after quantization to store
 */
void quant_qs4cx_f32(size_t n, size_t k, rhs_format format,
                     const float *rhs_f32, uint8_t *rhs_qs4cx,
                     float *rhs_scales_f32);
/**
 * @brief qa8dx quantization of (m*k) matrix. Typically a runtime activation
 * quantization. qa8dx refer to quantized asymmetric 8bit per-dimension
 * quantization
 *
 * @param m M length of the matrix
 * @param k K length of the matrix
 * @param lhs_f32 matrix data before quantization to load
 * @param lhs_qa8dx matrix data after quantization to store
 */
void ref_quant_qa8dx_f32(size_t m, size_t k, const float *lhs_f32,
                         int8_t *lhs_qa8dx);

/**
 * @brief GEMM of qai8dxp runtime-quantized activation and offline qs4cx
 * quantized weight
 *
 * @param m M length of the matrix
 * @param n N length of the matrix
 * @param k K length of the matrix
 * @param format whether rhs matrix is transpoed or not
 * @param lhs_qa8dx activation
 * @param rhs_qs4cx weight
 * @param rhs_scales_f32 weight scale factor
 * @param dst_f32 dst matrix
 * @param scalar_min lower bound to clamp
 * @param scalar_max upper bound to clamp
 */
void ref_matmul_f32_qa8dx_qs4cx(size_t m, size_t n, size_t k, rhs_format format,
                                const int8_t *lhs_qa8dx,
                                const uint8_t *rhs_qs4cx,
                                const float *rhs_scales_f32, float *dst_f32,
                                float scalar_min, float scalar_max);

/**
 * @brief qs4c32 quantization of (n*k) matrix with block size 32.
 * qsi4c32p refers to quantized symmetric 4-bit quantization with block size 32.
 *
 * @param n N length of the matrix
 * @param k K length of the matrix (must be divisible by bl)
 * @param bl block length (typically 32)
 * @param rhs_f32 matrix data before quantization to load
 * @param rhs_qs4c32 matrix data after quantization to store
 */
void quant_qs4c32_f32(size_t n, size_t k, size_t bl, const float *rhs_f32,
                      uint8_t *rhs_qs4c32);

/**
 * @brief qs8d32 quantization of (m*k) matrix with block size 32.
 * qsi8d32p refers to quantized symmetric 8-bit quantization with block size 32.
 *
 * @param n N length of the matrix
 * @param k K length of the matrix (must be divisible by bl)
 * @param bl block length (typically 32)
 * @param rhs_f32 matrix data before quantization to load
 * @param rhs_qs8c32 matrix data after quantization to store
 */
void ref_quant_qs8d32_f32(size_t n, size_t k, size_t bl, const float *rhs_f32,
                          uint8_t *rhs_qs8c32);

