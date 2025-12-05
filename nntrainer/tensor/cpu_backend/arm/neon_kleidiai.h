// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   neon_kleidiai.h
 * @date   15 September 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong
 *
 * @brief  Modified computational backend components of
 * kleidiai. Portions of this file are derived from Arm
 * Limited code licensed under the Apache License, Version 2.0, with
 * modifications
 *
 * @note   Licensed under the Apache License, Version 2.0 (the "License");
 *         you may not use this file except in compliance with the License.
 *         You may obtain a copy of the License at
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * @modifications
 *   - [2025-09-15] Integrated and adapted Arm-provided code into
 *     nntrainer CPU backend
 *
 * @bug    No known bugs except for NYI items
 */

#include <cstdint>
#include <stddef.h>

/**
 * @brief get size of memory to allocate for rhs weight packing of qsi4cxp to
 * qs4cxs1s0
 *
 * @param n row length if not transposed
 * @param k col length if not transposed
 * @return size_t size of memory to allocate
 */
size_t nntr_kai_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                                      uint32_t idx_variant,
                                                      bool transB);

/**
 * @brief rhs matrix packing for qsi4cxp format
 *
 * @param n row length if not transposed
 * @param k col length if not transposed
 * @param rhs_packed_mtx_qs4cx dst* to store results
 * @param rhs_native_mtx_qs4cx input matrix data
 * @param rhs_scales_f32 input qparam data
 * @param transB rather the matrix is transposed or not
 */
void nntr_kai_qsi4cxp_qs4cxs1s0_rhs_pack(size_t n, size_t k,
                                         void *rhs_packed_mtx_qs4cx,
                                         void *rhs_native_mtx_qs4cx,
                                         void *rhs_scales_f32,
                                         uint32_t idx_variant, bool transB);
/**
 * @brief run qai8dxp_qsi4cxp GEMM with runtime weight packing
 *
 * @param m M for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param lhs_native_mtx_f32 activation
 * @param rhs_native_mtx_qs4cx qs4cx quantized weight matrix data
 * @param rhs_scales_f32 qs4cx quantized weight matrix scale data
 * @param dst_act_mtx_f32 dst data
 * @param transB rather the weight matrix is transposed or not
 * @param lower_bound clipping param
 * @param upper_bound clipping param
 */
uint32_t nntr_kai_gemm_qai8dxp_qsi4cxp_rtp(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_act_mtx_f32,
  bool transB, float lower_bound, float upper_bound);
/**
 * @brief run qai8dxp_qsi4cxp GEMM with offline weight packing
 *
 * @param m M for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param lhs_native_mtx_f32 activation
 * @param rhs_packed_mtx_qs4cx qs4cx quantized weight, packed already
 * @param dst_act_mtx_f32 dst data
 * @param transB rather the weight matrix is transposed or not
 * @param lower_bound clipping param
 * @param upper_bound clipping param
 */
void nntr_kai_gemm_qai8dxp_qsi4cxp_olp(size_t m, size_t n, size_t k,
                                       void *lhs_native_mtx_f32,
                                       void *rhs_packed_mtx_qs4cx,
                                       float *dst_act_mtx_f32,
                                       uint32_t idx_variant, bool transB,
                                       float lower_bound, float upper_bound);

/**
 * @brief get size of memory to allocate for rhs weight packing of qsi8d32p to
 * qsi4c32p
 *
 * @param n row length if not transposed
 * @param k col length if not transposed
 * @return size_t size of memory to allocate
 */
size_t nntr_kai_get_rhs_packed_size_qsi8d32p_qsi4c32p(size_t n, size_t k,
                                                      uint32_t idx_variant,
                                                      bool transB);

/**
 * @brief rhs matrix packing for qsi8d32p format
 *
 * @param n row length if not transposed
 * @param k col length if not transposed
 * @param rhs_packed_mtx_qs4cx dst* to store results
 * @param rhs_native_mtx_qs4cx input matrix data
 * @param rhs_scales_f32 input qparam data
 * @param transB rather the matrix is transposed or not
 */
void nntr_kai_qsi8d32p_qsi4c32p_rhs_pack(size_t n, size_t k,
                                         void *rhs_packed_mtx_qs4cx,
                                         void *rhs_native_mtx_qs4cx,
                                         void *rhs_scales_f32,
                                         uint32_t idx_variant, bool transB);

/**
 * @brief run qsi8d32p_qsi4c32p GEMM with runtime weight packing
 *
 * @param m M for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param lhs_native_mtx_f32 activation
 * @param rhs_native_mtx_qs4cx qs4cx quantized weight matrix data
 * @param rhs_scales_f32 qs4cx quantized weight matrix scale data
 * @param dst_act_mtx_f32 dst data
 * @param transB rather the weight matrix is transposed or not
 * @param lower_bound clipping param
 * @param upper_bound clipping param
 */
uint32_t nntr_kai_gemm_qsi8d32p_qsi4c32p_rtp(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_act_mtx_f32,
  bool transB, float lower_bound, float upper_bound);

/**
 * @brief run qsi8d32p_qsi4c32p GEMM with offline weight packing
 *
 * @param m M for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param lhs_native_mtx_f32 activation
 * @param rhs_packed_mtx_qs4cx qs4cx quantized weight, packed already
 * @param dst_act_mtx_f32 dst data
 * @param transB rather the weight matrix is transposed or not
 * @param lower_bound clipping param
 * @param upper_bound clipping param
 */
void nntr_kai_gemm_qsi8d32p_qsi4c32p_olp(size_t m, size_t n, size_t k,
                                         void *lhs_native_mtx_f32,
                                         void *rhs_packed_mtx_qs4cx,
                                         float *dst_act_mtx_f32,
                                         uint32_t idx_variant, bool transB,
                                         float lower_bound, float upper_bound);
