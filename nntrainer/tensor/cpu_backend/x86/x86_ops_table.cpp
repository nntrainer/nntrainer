// SPDX-License-Identifier: Apache-2.0
/**
 * @file   x86_ops_table.cpp
 * @brief  x86 CPU backend ComputeOps table.
 *
 * Each wrapper function adapts the ARM backend implementation to the
 * ComputeOps interface. This follows the same pattern as cl_compute_ops.cpp
 * (GPU) and any future NPU backend — all vendors use the same pattern:
 *
 *   static void VENDOR_OPNAME(...) { vendor_internal_impl(...); }
 *   .op_field = VENDOR_OPNAME,
 */
#include <x86_compute_backend.h>
#include <x86_qsi4cxp.h>
#include <compute_ops.h>

namespace nntrainer {
namespace x86_backend {

// ── FP32 BLAS ────────────────────────────────────────────────
static void sgemm_fp32(const unsigned int o, bool tA, bool tB,
                        const unsigned int M, const unsigned int N,
                        const unsigned int K, const float a, const float *A,
                        const unsigned int lda, const float *B,
                        const unsigned int ldb, const float b, float *C,
                        const unsigned int ldc) {
  nntrainer::sgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
}
static void sgemv_fp32(const unsigned int o, bool tA, const unsigned int M,
                        const unsigned int N, const float a, const float *A,
                        const unsigned int lda, const float *X,
                        const unsigned int iX, const float b, float *Y,
                        const unsigned int iY) {
  nntrainer::sgemv(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
}
static float sdot_fp32(const unsigned int N, const float *X,
                        const unsigned int iX, const float *Y,
                        const unsigned int iY) {
  return nntrainer::sdot(N, X, iX, Y, iY);
}
static void saxpy_fp32(const unsigned int N, const float a, const float *X,
                        const unsigned int iX, float *Y,
                        const unsigned int iY) {
  nntrainer::saxpy(N, a, X, iX, Y, iY);
}
static void scopy_fp32(const unsigned int N, const float *X,
                        const unsigned int iX, float *Y,
                        const unsigned int iY) {
  nntrainer::scopy(N, X, iX, Y, iY);
}
static void sscal_fp32(const unsigned int N, const float a, float *X,
                        const unsigned int iX) {
  nntrainer::sscal(N, a, X, iX);
}
static float snrm2_fp32(const unsigned int N, const float *X,
                         const unsigned int iX) {
  return nntrainer::snrm2(N, X, iX);
}
static unsigned int isamax_fp32(const unsigned int N, const float *X,
                                 const unsigned int iX) {
  return nntrainer::isamax(N, X, iX);
}

// ── FP32 Element-wise ────────────────────────────────────────
static void ele_mul_fp32(const unsigned int N, const float *X, const float *Y,
                          float *Z, float a, float b, unsigned int is,
                          unsigned int os) {
  nntrainer::ele_mul(N, X, Y, Z, a, b, is, os);
}
static void ele_add_fp32(const unsigned int N, const float *X, const float *Y,
                          float *Z, float a, float b, unsigned int is,
                          unsigned int os) {
  nntrainer::ele_add(N, X, Y, Z, a, b, is, os);
}
static void ele_sub_fp32(const unsigned int N, const float *X, const float *Y,
                          float *Z, float a, float b, unsigned int is,
                          unsigned int os) {
  nntrainer::ele_sub(N, X, Y, Z, a, b, is, os);
}
static void ele_div_fp32(const unsigned int N, const float *X, const float *Y,
                          float *Z, float a, float b, unsigned int is,
                          unsigned int os) {
  nntrainer::ele_div(N, X, Y, Z, a, b, is, os);
}

// ── FP32 Activation / Special ────────────────────────────────
static void swiglu_fp32(const unsigned int N, float *X, float *Y, float *Z) {
  nntrainer::swiglu(N, X, Y, Z);
}
static void swiglu_alpha_fp32(const unsigned int N, float *X, float *Y,
                               float *Z, float alpha) {
  nntrainer::swiglu(N, X, Y, Z, alpha);
}
static void tanh_gelu_fp32(const unsigned int N, const float *X, float *Y) {
  nntrainer::tanh_gelu(N, X, Y);
}
static void gelu_v2_fp32(const unsigned int N, const float *X, float *Y) {
  nntrainer::gelu_v2(N, X, Y);
}
static void tanh_gelu_v2_fp32(const unsigned int N, const float *X, float *Y) {
  nntrainer::tanh_gelu_v2(N, X, Y);
}
static void tanh_gelu_mul_fp32(const unsigned int N, float *X, float *Y,
                                float *Z) {
  nntrainer::tanh_gelu_mul(N, X, Y, Z);
}
static void tanh_gelu_v2_mul_fp32(const unsigned int N, float *X, float *Y,
                                   float *Z) {
  nntrainer::tanh_gelu_v2_mul(N, X, Y, Z);
}
static float max_val_fp32(const unsigned int N, float *X) {
  return nntrainer::max_val(N, X);
}
static void softmax_fp32(const unsigned int N, float *X, float *Y) {
  nntrainer::softmax(N, X, Y);
}
static bool is_valid_fp32(const unsigned int N, const float *X) {
  return nntrainer::is_valid(N, X);
}

// ── FP32 Matrix / Copy ──────────────────────────────────────
static void transpose_matrix_fp32(const unsigned int M, const unsigned int N,
                                   const float *s, unsigned int lds,
                                   float *d, unsigned int ldd) {
  nntrainer::transpose_matrix(M, N, s, lds, d, ldd);
}

// ── Quantized GEMM ──────────────────────────────────────────
static void gemm_q4_0_fp32(const unsigned int M, const unsigned int N,
                             const unsigned int K, const float *A,
                             const unsigned int lda, const void *B,
                             const unsigned int ldb, float *C,
                             const unsigned int ldc) {
  nntrainer::gemm_q4_0(M, N, K, A, lda, B, ldb, C, ldc);
}
static void gemm_q4_K_fp32(const unsigned int M, const unsigned int N,
                             const unsigned int K, const float *A,
                             const unsigned int lda, const void *B,
                             const unsigned int ldb, float *C,
                             const unsigned int ldc) {
  nntrainer::gemm_q4_K(M, N, K, A, lda, B, ldb, C, ldc);
}
static void gemm_q6_K_fp32(const unsigned int M, const unsigned int N,
                             const unsigned int K, const float *A,
                             const unsigned int lda, const void *B,
                             const unsigned int ldb, float *C,
                             const unsigned int ldc) {
  nntrainer::gemm_q6_K(M, N, K, A, lda, B, ldb, C, ldc);
}

// ── Quantization / Utility ──────────────────────────────────
static size_t quantize_q4_0_impl(const float *src, void *dst, int64_t nrow,
                                  int64_t n_per_row, const float *qw) {
  return nntrainer::quantize_q4_0(src, dst, nrow, n_per_row, qw);
}
static void dequantize_row_q4_0_impl(const void *x, float *y, int64_t k) {
  nntrainer::dequantize_row_q4_0(x, y, k);
}
static void repack_q4_0_impl(void *dst, void *src, size_t ds,
                              const unsigned int M, const unsigned int N) {
  nntrainer::repack_q4_0(dst, src, ds, M, N);
}
static void clamp_fp32(const float *in, float *out, size_t len, float lb,
                        float ub) {
  nntrainer::clamp(in, out, len, lb, ub);
}
static void scopy_int8_to_fp32_u(const unsigned int N, const uint8_t *X,
                                  const unsigned int iX, float *Y,
                                  const unsigned int iY) {
  nntrainer::scopy_int8_to_float32(N, X, iX, Y, iY);
}
static void scopy_int8_to_fp32_s(const unsigned int N, const int8_t *X,
                                  const unsigned int iX, float *Y,
                                  const unsigned int iY) {
  nntrainer::scopy_int8_to_float32(N, X, iX, Y, iY);
}

} // namespace x86_backend

// ═══════════════════════════════════════════════════════════════
// ARM ComputeOps Table
// ═══════════════════════════════════════════════════════════════
static ComputeOps x86_ops = {
  .sgemm_fp32 = x86_backend::sgemm_fp32,
  .sgemv_fp32 = x86_backend::sgemv_fp32,
  .sdot_fp32 = x86_backend::sdot_fp32,
  .saxpy_fp32 = x86_backend::saxpy_fp32,
  .scopy_fp32 = x86_backend::scopy_fp32,
  .sscal_fp32 = x86_backend::sscal_fp32,
  .snrm2_fp32 = x86_backend::snrm2_fp32,
  .isamax_fp32 = x86_backend::isamax_fp32,
  .ele_mul_fp32 = x86_backend::ele_mul_fp32,
  .ele_add_fp32 = x86_backend::ele_add_fp32,
  .ele_sub_fp32 = x86_backend::ele_sub_fp32,
  .ele_div_fp32 = x86_backend::ele_div_fp32,
  .swiglu_fp32 = x86_backend::swiglu_fp32,
  .swiglu_alpha_fp32 = x86_backend::swiglu_alpha_fp32,
  .tanh_gelu_fp32 = x86_backend::tanh_gelu_fp32,
  .gelu_v2_fp32 = x86_backend::gelu_v2_fp32,
  .tanh_gelu_v2_fp32 = x86_backend::tanh_gelu_v2_fp32,
  .tanh_gelu_mul_fp32 = x86_backend::tanh_gelu_mul_fp32,
  .tanh_gelu_v2_mul_fp32 = x86_backend::tanh_gelu_v2_mul_fp32,
  .max_val_fp32 = x86_backend::max_val_fp32,
  .softmax_fp32 = x86_backend::softmax_fp32,
  .is_valid_fp32 = x86_backend::is_valid_fp32,
  .transpose_matrix_fp32 = x86_backend::transpose_matrix_fp32,
  .scopy_u8 = nntrainer::scopy,
  .scopy_s8 = nntrainer::scopy,
  .scopy_int4_to_float32 = nntrainer::scopy_int4_to_float32,
  .copy_s16_fp32 = nntrainer::copy_s16_fp32,
  .copy_u16_fp32 = nntrainer::copy_u16_fp32,
  .copy_fp32_u32 = nntrainer::copy_fp32_u32,
  .copy_fp32_u16 = nntrainer::copy_fp32_u16,
  .copy_fp32_u8 = nntrainer::copy_fp32_u8,
  .copy_fp32_s16 = nntrainer::copy_fp32_s16,
  .copy_fp32_s8 = nntrainer::copy_fp32_s8,
  .gemm_q4_0_fp32 = x86_backend::gemm_q4_0_fp32,
  .gemm_q4_K_fp32 = x86_backend::gemm_q4_K_fp32,
  .gemm_q6_K_fp32 = x86_backend::gemm_q6_K_fp32,
  .unpack_q4_0 = nntrainer::unpack_q4_0,
  .unpack_q4_0x8_transpose16 = nntrainer::unpack_q4_0x8_transpose16,
  .quantize_q4_0 = x86_backend::quantize_q4_0_impl,
  .dequantize_row_q4_0 = x86_backend::dequantize_row_q4_0_impl,
  .repack_q4_0 = x86_backend::repack_q4_0_impl,
  .clamp_fp32 = x86_backend::clamp_fp32,
  .scopy_int8_to_fp32_u = x86_backend::scopy_int8_to_fp32_u,
  .scopy_int8_to_fp32_s = x86_backend::scopy_int8_to_fp32_s,
  .gemm_q4_0_batch_fp32 = nullptr,
  .gemm_q4_0_accel_fp32 = nullptr,
  .gemv_int4_batch_fp32 = nullptr,
  .gemm_int4_batch_fp32 = nullptr,
  .gemv_int4_accel_fp32 = nullptr,
  .sgemm_int4_accel_fp32 = nullptr,
  .gemm_qsi4cxp_fp32 = [](unsigned int M, unsigned int N, unsigned int K,
                           const float *A, const void *B, const void *S,
                           float *C) {
    nntrainer::gemm_qsi4cxp_kxn_fp32(M, N, K, A,
                                      static_cast<const uint8_t *>(B),
                                      static_cast<const float *>(S), C);
  },
#ifdef ENABLE_FP16
  // FP16 ops use nntrainer:: directly (same pattern applies when
  // FP16-specific wrappers are needed)
  .sgemm_fp16 = nntrainer::sgemm,
  .sgemv_fp16 = nntrainer::sgemv,
  .sdot_fp16 = nntrainer::sdot,
  .saxpy_fp16 = nntrainer::saxpy,
  .scopy_fp16 =
    static_cast<void (*)(const unsigned int, const _FP16 *, const unsigned int,
                         _FP16 *, const unsigned int)>(nntrainer::scopy),
  .scopy_fp32_to_fp16 =
    static_cast<void (*)(const unsigned int, const float *, const unsigned int,
                         _FP16 *, const unsigned int)>(nntrainer::scopy),
  .scopy_fp16_to_fp32 =
    static_cast<void (*)(const unsigned int, const _FP16 *, const unsigned int,
                         float *, const unsigned int)>(nntrainer::scopy),
  .sscal_fp16 = nntrainer::sscal,
  .snrm2_fp16 = nntrainer::snrm2,
  .isamax_fp16 = nntrainer::isamax,
  .ele_mul_fp16 = nntrainer::ele_mul,
  .ele_add_fp16 = nntrainer::ele_add,
  .ele_sub_fp16 = nntrainer::ele_sub,
  .ele_div_fp16 = nntrainer::ele_div,
  .swiglu_fp16 = nntrainer::swiglu,
  .max_val_fp16 = nntrainer::max_val,
  .softmax_fp16 = nntrainer::softmax,
  .is_valid_fp16 = nntrainer::is_valid,
  .inv_sqrt_inplace_fp16 = nntrainer::inv_sqrt_inplace,
  .transpose_matrix_fp16 = nntrainer::transpose_matrix,
  .scopy_int4_to_float16 = nntrainer::scopy_int4_to_float16,
  .scopy_int8_to_float16_u =
    static_cast<void (*)(const unsigned int, const uint8_t *,
                         const unsigned int, _FP16 *, const unsigned int)>(
      nntrainer::scopy_int8_to_float16),
  .scopy_int8_to_float16_s =
    static_cast<void (*)(const unsigned int, const int8_t *,
                         const unsigned int, _FP16 *, const unsigned int)>(
      nntrainer::scopy_int8_to_float16),
  .shgemm = nntrainer::shgemm,
  .shgemv = nntrainer::shgemv,
  .hsgemm = nntrainer::hsgemm,
  .hsgemv = nntrainer::hsgemv,
  .gemm_q4_0_fp16 = nntrainer::gemm_q4_0<_FP16>,
  .gemm_q6_K_fp16 = nntrainer::gemm_q6_K<_FP16>,
  .compute_rotary_embedding_value = nntrainer::compute_rotary_embedding_value,
#endif
};

ComputeOps *get_x86_ops() { return &x86_ops; }

} // namespace nntrainer
