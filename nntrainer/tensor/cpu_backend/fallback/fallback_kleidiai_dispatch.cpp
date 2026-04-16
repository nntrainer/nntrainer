// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   fallback_kleidiai_dispatch.cpp
 * @brief  Thin wrappers for qsi4cxp functions when ENABLE_FP16 is OFF.
 *         When FP16 is ON, fallback_fp16.cpp / fallback_internal_fp16.cpp
 *         provide the same functions. This file ensures the qsi4cxp
 *         path is always linkable regardless of the FP16 build flag.
 *
 *         The actual computation lives in fallback_kleidiai.cpp (pure C
 *         reference: quant_qs4cx_f32, ref_matmul_f32_qa8dx_qs4cx, etc.)
 *         which is ALWAYS compiled.
 */

#ifndef ENABLE_FP16
// Only needed when FP16 is OFF — avoids double-definition with
// fallback_internal_fp16.cpp and fallback_fp16.cpp.

#include <fallback_internal.h>
#include <fallback_kleidiai.h>
#include <fallback.h>

namespace nntrainer {

void __fallback_nntr_quant_qs4cx_f32(size_t n, size_t k,
                                     void *rhs_native_mtx_f32,
                                     void *rhs_native_mtx_qs4cx,
                                     void *rhs_scales_f32, bool transB) {
  rhs_format format = rhs_format::nxk;
  if (!transB) {
    format = rhs_format::kxn;
  }
  quant_qs4cx_f32(n, k, format, (const float *)rhs_native_mtx_f32,
                  (uint8_t *)rhs_native_mtx_qs4cx, (float *)rhs_scales_f32);
}

template <>
uint32_t __fallback_nntr_gemm_qai8dxp_qsi4cxp_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_mtx_f32,
  bool transB, float lower_bound, float upper_bound) {

  rhs_format format = rhs_format::nxk;
  if (!transB) {
    format = rhs_format::kxn;
  }

  uint8_t *lhs_ref_mtx_qa8dx =
    new uint8_t[m * (k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t))];

  ref_quant_qa8dx_f32(m, k, (const float *)lhs_native_mtx_f32,
                      (int8_t *)lhs_ref_mtx_qa8dx);

  ref_matmul_f32_qa8dx_qs4cx(m, n, k, format, (const int8_t *)lhs_ref_mtx_qa8dx,
                             (const uint8_t *)rhs_native_mtx_qs4cx,
                             (const float *)rhs_scales_f32,
                             (float *)dst_mtx_f32, lower_bound, upper_bound);

  delete[] lhs_ref_mtx_qa8dx;
  return 1;
}

// Public thin wrappers
void nntr_quant_qs4cx_f32(size_t n, size_t k, void *rhs_native_mtx_f32,
                          void *rhs_native_mtx_qs4cx, void *rhs_scales_f32,
                          bool transB) {
  __fallback_nntr_quant_qs4cx_f32(n, k, rhs_native_mtx_f32,
                                  rhs_native_mtx_qs4cx, rhs_scales_f32, transB);
}

template <>
uint32_t nntr_gemm_qai8dxp_qsi4cxp_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_mtx_f32,
  bool transB, float lower_bound, float upper_bound) {
  return __fallback_nntr_gemm_qai8dxp_qsi4cxp_unpacked(
    m, n, k, lhs_native_mtx_f32, rhs_native_mtx_qs4cx, rhs_scales_f32,
    dst_mtx_f32, transB, lower_bound, upper_bound);
}

} // namespace nntrainer

#endif // !ENABLE_FP16
