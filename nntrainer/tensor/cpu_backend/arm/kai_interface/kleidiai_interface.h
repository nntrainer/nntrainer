// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   kleidiai_interface.h
 * @date   15 September 2025
 * @see    https://github.com/nntrainer/nntrainer
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

#include <cfloat>
#include <cstdint>
#include <string>

namespace nntrainer {
/**
 * matmul_clamp_f32_qai8dxp_qsi4cxp
 */

/**
 * @brief get name of ukernel
 *
 * @param[in] idx_variant index of ukernel
 * @return std::string name of ukernel
 */
std::string __kai_get_num_ukernel_name_qai8dxp_qsi4cxp(size_t idx_variant);

/**
 * @brief get number of ukernels
 *
 * @return size_t number of ukernels
 */
size_t __kai_get_num_ukernel_variants_qai8dxp_qsi4cxp();

/**
 * @brief get size of memory to allocate for packed rhs from nxk qs4cxs1s0 to
 * qsi4cxp
 * Note that nxk is the format of quantized rhs, not the shape of rhs
 *
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] idx_variant index of ukernel to use
 * @param[in] is_nxk true if rhs is in nxk format
 * @return size_t size of memory to allocate
 */
size_t __kai_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                                   size_t idx_variant,
                                                   bool is_nxk);

/**
 * @brief rhs matrix packing from nxk qs4cxs1s0 to qsi4cxp
 * Note that rhs_qs4cx must be stored with UINT4 shape (zero point = 8)
 *
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[out] rhs_packed_mtx_qs4cx packed rhs
 * @param[in] rhs_native_mtx_qs4cx quantized matrix data
 * @param[in] rhs_scales_f32 qparam data
 * @param[in] idx_variant index of ukernel to use
 * @param[in] is_nxk true if rhs is in nxk format
 */
void __kai_rhs_pack_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                      void *rhs_packed_mtx_qs4cx,
                                      void *rhs_native_mtx_qs4cx,
                                      void *rhs_scales_f32, size_t idx_variant,
                                      bool is_nxk);

/**
 * @brief run qai8dxp_qsi4cxp GEMM with online rhs packing
 *
 * @param[in] m M for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] lhs_native_mtx_f32 matrix data
 * @param[in] rhs_native_mtx_qs4cx quantized matrix data
 * @param[in] rhs_scales_f32 qparam data
 * @param[out] dst_act_mtx_f32 output data
 * @param[in] idx_variant index of ukernel to use
 * @param[in] is_nxk true if rhs is in nxk format
 * @param[in] lower_bound clipping param
 * @param[in] upper_bound clipping param
 */
void __kai_gemm_qai8dxp_qsi4cxp_rhs_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_act_mtx_f32,
  size_t idx_variant, bool is_nxk, float lower_bound = -FLT_MAX,
  float upper_bound = FLT_MAX);

/**
 * @brief run qai8dxp_qsi4cxp GEMM with offline packed rhs
 *
 * @param[in] m M for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] lhs_native_mtx_f32 matrix data
 * @param[in] rhs_packed_mtx_qs4cx quantized matrix data, packed already
 * @param[out] dst_act_mtx_f32 output data
 * @param[in] idx_variant index of ukernel to use
 * @param[in] lower_bound clipping param
 * @param[in] upper_bound clipping param
 */
void __kai_gemm_qai8dxp_qsi4cxp(size_t m, size_t n, size_t k,
                                void *lhs_native_mtx_f32,
                                void *rhs_packed_mtx_qs4cx,
                                float *dst_act_mtx_f32, size_t idx_variant,
                                float lower_bound = -FLT_MAX,
                                float upper_bound = FLT_MAX);

#ifdef ENABLE_FP16
/**
 * matmul_clamp_f16_qai8dxp_qsi4cxp
 */

/**
 * @brief get name of ukernel
 *
 * @param[in] idx_variant index of ukernel
 * @return std::string name of ukernel
 */
std::string __kai_get_num_ukernel_name_f16_qai8dxp_qsi4cxp(size_t idx_variant);

/**
 * @brief get number of ukernels
 *
 * @return size_t number of ukernels
 */
size_t __kai_get_num_ukernel_variants_f16_qai8dxp_qsi4cxp();

/**
 * @brief get size of memory to allocate for packed rhs from nxk qs4cxs1s0 to
 * qsi4cxp
 * Note that nxk is the format of quantized rhs, not the shape of rhs
 *
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] idx_variant index of ukernel to use
 * @param[in] is_nxk true if rhs is in nxk format
 * @return size_t size of memory to allocate
 */
size_t __kai_get_rhs_packed_size_f16_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                                       size_t idx_variant,
                                                       bool is_nxk);

/**
 * @brief rhs matrix packing from nxk qs4cxs1s0 to qsi4cxp
 * Note that rhs_qs4cx must be stored with UINT4 shape (zero point = 8)
 *
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[out] rhs_packed_mtx_qs4cx packed rhs
 * @param[in] rhs_native_mtx_qs4cx quantized matrix data
 * @param[in] rhs_scales_f32 qparam data
 * @param[in] idx_variant index of ukernel to use
 * @param[in] is_nxk true if rhs is in nxk format
 */
void __kai_rhs_pack_f16_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                          void *rhs_packed_mtx_qs4cx,
                                          void *rhs_native_mtx_qs4cx,
                                          void *rhs_scales_f32,
                                          size_t idx_variant, bool is_nxk);

/**
 * @brief run f16_qai8dxp_qsi4cxp GEMM with online rhs packing
 *
 * @param[in] m M for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] lhs_native_mtx_f16 matrix data (F16)
 * @param[in] rhs_native_mtx_qs4cx quantized matrix data
 * @param[in] rhs_scales_f32 qparam data
 * @param[out] dst_act_mtx_f16 output data (F16)
 * @param[in] idx_variant index of ukernel to use
 * @param[in] is_nxk true if rhs is in nxk format
 * @param[in] lower_bound clipping param
 * @param[in] upper_bound clipping param
 */
void __kai_gemm_f16_qai8dxp_qsi4cxp_rhs_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f16,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, void *dst_act_mtx_f16,
  size_t idx_variant, bool is_nxk, float lower_bound = -FLT_MAX,
  float upper_bound = FLT_MAX);

/**
 * @brief run f16_qai8dxp_qsi4cxp GEMM with offline packed rhs
 *
 * @param[in] m M for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] lhs_native_mtx_f16 matrix data (F16)
 * @param[in] rhs_packed_mtx_qs4cx quantized matrix data, packed already
 * @param[out] dst_act_mtx_f16 output data (F16)
 * @param[in] idx_variant index of ukernel to use
 * @param[in] lower_bound clipping param
 * @param[in] upper_bound clipping param
 */
void __kai_gemm_f16_qai8dxp_qsi4cxp(size_t m, size_t n, size_t k,
                                    void *lhs_native_mtx_f16,
                                    void *rhs_packed_mtx_qs4cx,
                                    void *dst_act_mtx_f16, size_t idx_variant,
                                    float lower_bound = -FLT_MAX,
                                    float upper_bound = FLT_MAX);
#endif // ENABLE_FP16
/**
 * matmul_clamp_f32_qai8dxp_qsi8cxp
 */

/**
 * @brief get name of ukernel
 *
 * @param[in] idx_variant index of ukernel
 * @return std::string name of ukernel
 */
std::string __kai_get_num_ukernel_name_qai8dxp_qsi8cxp(size_t idx_variant);

/**
 * @brief get number of ukernels
 *
 * @return size_t number of ukernels
 */
size_t __kai_get_num_ukernel_variants_qai8dxp_qsi8cxp();

/**
 * @brief get size of memory to allocate for packed rhs from nxk qsi8cx to
 * qsi8cxp
 * Note that nxk is the format of quantized rhs, not the shape of rhs
 *
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] idx_variant index of ukernel to use
 * @return size_t size of memory to allocate
 */
size_t __kai_get_rhs_packed_size_qsi8cxp_qsi8cx(size_t n, size_t k,
                                                size_t idx_variant);

/**
 * @brief rhs matrix packing from nxk qsi8cx to qsi8cxp
 *
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[out] rhs_packed_mtx_qs8cx packed rhs
 * @param[in] rhs_native_mtx_qs8cx quantized matrix data (int8, nxk)
 * @param[in] rhs_scales_f32 qparam data
 * @param[in] idx_variant index of ukernel to use
 */
void __kai_rhs_pack_qsi8cxp_qsi8cx(size_t n, size_t k,
                                   void *rhs_packed_mtx_qs8cx,
                                   void *rhs_native_mtx_qs8cx,
                                   void *rhs_scales_f32, size_t idx_variant);

/**
 * @brief run qai8dxp_qsi8cxp GEMM with online rhs packing
 *
 * @param[in] m M for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] lhs_native_mtx_f32 matrix data
 * @param[in] rhs_native_mtx_qs8cx quantized matrix data (int8, nxk)
 * @param[in] rhs_scales_f32 qparam data
 * @param[out] dst_act_mtx_f32 output data
 * @param[in] idx_variant index of ukernel to use
 * @param[in] lower_bound clipping param
 * @param[in] upper_bound clipping param
 */
void __kai_gemm_qai8dxp_qsi8cxp_rhs_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs8cx, void *rhs_scales_f32, float *dst_act_mtx_f32,
  size_t idx_variant, float lower_bound = -FLT_MAX,
  float upper_bound = FLT_MAX);

/**
 * @brief run qai8dxp_qsi8cxp GEMM with offline packed rhs
 *
 * @param[in] m M for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] n N for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] k K for (M, K) * (K, N) = (M, N) in noTrans GEMM
 * @param[in] lhs_native_mtx_f32 matrix data
 * @param[in] rhs_packed_mtx_qs8cx quantized matrix data, packed already
 * @param[out] dst_act_mtx_f32 output data
 * @param[in] idx_variant index of ukernel to use
 * @param[in] lower_bound clipping param
 * @param[in] upper_bound clipping param
 */
void __kai_gemm_qai8dxp_qsi8cxp(size_t m, size_t n, size_t k,
                                void *lhs_native_mtx_f32,
                                void *rhs_packed_mtx_qs8cx,
                                float *dst_act_mtx_f32, size_t idx_variant,
                                float lower_bound = -FLT_MAX,
                                float upper_bound = FLT_MAX);

} // namespace nntrainer
