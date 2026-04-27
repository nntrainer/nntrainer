// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cpu_ops_table.cpp
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Unified CPU backend ComputeOps subclass.
 *
 * Single concrete ComputeOps subclass for ALL CPU targets (ARM /
 * x86 / fallback). The nntrainer::sgemm etc. functions are arch-
 * specialized — each arch_compute_backend.cpp defines its own body
 * — so a single forwarding wrapper is enough; build-time arch
 * dispatch picks the right symbol at link time.
 */

#include <compute_ops.h>
#include <cpu_backend.h>

namespace nntrainer {

class CpuComputeOps : public ComputeOps {
public:
  // FP32 BLAS
  void sgemm_fp32(unsigned int o, bool tA, bool tB, unsigned int M,
                  unsigned int N, unsigned int K, float a, const float *A,
                  unsigned int lda, const float *B, unsigned int ldb, float b,
                  float *C, unsigned int ldc) override {
    nntrainer::sgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void sgemv_fp32(unsigned int o, bool tA, unsigned int M, unsigned int N,
                  float a, const float *A, unsigned int lda, const float *X,
                  unsigned int iX, float b, float *Y,
                  unsigned int iY) override {
    nntrainer::sgemv(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }
  float sdot_fp32(unsigned int N, const float *X, unsigned int iX,
                  const float *Y, unsigned int iY) override {
    return nntrainer::sdot(N, X, iX, Y, iY);
  }
  void saxpy_fp32(unsigned int N, float a, const float *X, unsigned int iX,
                  float *Y, unsigned int iY) override {
    nntrainer::saxpy(N, a, X, iX, Y, iY);
  }
  void scopy_fp32(unsigned int N, const float *X, unsigned int iX, float *Y,
                  unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void sscal_fp32(unsigned int N, float a, float *X, unsigned int iX) override {
    nntrainer::sscal(N, a, X, iX);
  }
  float snrm2_fp32(unsigned int N, const float *X, unsigned int iX) override {
    return nntrainer::snrm2(N, X, iX);
  }
  unsigned int isamax_fp32(unsigned int N, const float *X,
                           unsigned int iX) override {
    return nntrainer::isamax(N, X, iX);
  }

  // FP32 Element-wise
  void ele_mul_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_mul(N, X, Y, Z, a, b, is, os);
  }
  void ele_add_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_add(N, X, Y, Z, a, b, is, os);
  }
  void ele_sub_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_sub(N, X, Y, Z, a, b, is, os);
  }
  void ele_div_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_div(N, X, Y, Z, a, b, is, os);
  }

  // FP32 Activation / Special
  void swiglu_fp32(unsigned int N, float *X, float *Y, float *Z) override {
    nntrainer::swiglu(N, X, Y, Z);
  }
  void swiglu_alpha_fp32(unsigned int N, float *X, float *Y, float *Z,
                         float alpha) override {
    nntrainer::swiglu(N, X, Y, Z, alpha);
  }
  void tanh_gelu_fp32(unsigned int N, const float *X, float *Y) override {
    nntrainer::tanh_gelu(N, X, Y);
  }
  void gelu_v2_fp32(unsigned int N, const float *X, float *Y) override {
    nntrainer::gelu_v2(N, X, Y);
  }
  void tanh_gelu_v2_fp32(unsigned int N, const float *X, float *Y) override {
    nntrainer::tanh_gelu_v2(N, X, Y);
  }
  void tanh_gelu_mul_fp32(unsigned int N, float *X, float *Y,
                          float *Z) override {
    nntrainer::tanh_gelu_mul(N, X, Y, Z);
  }
  void tanh_gelu_v2_mul_fp32(unsigned int N, float *X, float *Y,
                             float *Z) override {
    nntrainer::tanh_gelu_v2_mul(N, X, Y, Z);
  }
  float max_val_fp32(unsigned int N, float *X) override {
    return nntrainer::max_val(N, X);
  }
  void softmax_fp32(unsigned int N, float *X, float *Y) override {
    nntrainer::softmax(N, X, Y);
  }
  bool is_valid_fp32(unsigned int N, const float *X) override {
    return nntrainer::is_valid(N, X);
  }

  // FP32 Matrix
  void transpose_matrix_fp32(unsigned int M, unsigned int N, const float *s,
                             unsigned int lds, float *d,
                             unsigned int ldd) override {
    nntrainer::transpose_matrix(M, N, s, lds, d, ldd);
  }

  // FP32 Data conversion / Copy
  void scopy_u8(unsigned int N, const uint8_t *X, unsigned int iX, uint8_t *Y,
                unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void scopy_s8(unsigned int N, const int8_t *X, unsigned int iX, int8_t *Y,
                unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void scopy_int4_to_float32(unsigned int N, const uint8_t *X, unsigned int iX,
                             float *Y, unsigned int iY) override {
    nntrainer::scopy_int4_to_float32(N, X, iX, Y, iY);
  }
  void copy_s16_fp32(unsigned int N, const int16_t *X, float *Y) override {
    nntrainer::copy_s16_fp32(N, X, Y);
  }
  void copy_u16_fp32(unsigned int N, const uint16_t *X, float *Y) override {
    nntrainer::copy_u16_fp32(N, X, Y);
  }
  void copy_fp32_u32(unsigned int N, const float *X, uint32_t *Y) override {
    nntrainer::copy_fp32_u32(N, X, Y);
  }
  void copy_fp32_u16(unsigned int N, const float *X, uint16_t *Y) override {
    nntrainer::copy_fp32_u16(N, X, Y);
  }
  void copy_fp32_u8(unsigned int N, const float *X, uint8_t *Y) override {
    nntrainer::copy_fp32_u8(N, X, Y);
  }
  void copy_fp32_s16(unsigned int N, const float *X, int16_t *Y) override {
    nntrainer::copy_fp32_s16(N, X, Y);
  }
  void copy_fp32_s8(unsigned int N, const float *X, int8_t *Y) override {
    nntrainer::copy_fp32_s8(N, X, Y);
  }

  // Quantized GEMM
  void gemm_q4_0_fp32(unsigned int M, unsigned int N, unsigned int K,
                      const float *A, unsigned int lda, const void *B,
                      unsigned int ldb, float *C, unsigned int ldc) override {
    nntrainer::gemm_q4_0(M, N, K, A, lda, B, ldb, C, ldc);
  }
  void gemm_q4_K_fp32(unsigned int M, unsigned int N, unsigned int K,
                      const float *A, unsigned int lda, const void *B,
                      unsigned int ldb, float *C, unsigned int ldc) override {
    nntrainer::gemm_q4_K(M, N, K, A, lda, B, ldb, C, ldc);
  }
  void gemm_q6_K_fp32(unsigned int M, unsigned int N, unsigned int K,
                      const float *A, unsigned int lda, const void *B,
                      unsigned int ldb, float *C, unsigned int ldc) override {
    nntrainer::gemm_q6_K(M, N, K, A, lda, B, ldb, C, ldc);
  }

  // Quantization / Utility
  void unpack_q4_0(const void *in, void *out, size_t ds, unsigned int M,
                   unsigned int N) override {
    nntrainer::unpack_q4_0(in, out, ds, M, N);
  }
  void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                                 uint16_t *qs_out, int N, int K) override {
    nntrainer::unpack_q4_0x8_transpose16(src, d_out, qs_out, N, K);
  }
  size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                       int64_t n_per_row, const float *qw) override {
    return nntrainer::quantize_q4_0(src, dst, nrow, n_per_row, qw);
  }
  void dequantize_row_q4_0(const void *x, float *y, int64_t k) override {
    nntrainer::dequantize_row_q4_0(x, y, k);
  }
  void repack_q4_0(void *dst, void *src, size_t ds, unsigned int M,
                   unsigned int N) override {
    nntrainer::repack_q4_0(dst, src, ds, M, N);
  }

  void clamp_fp32(const float *in, float *out, size_t len, float lb,
                  float ub) override {
    nntrainer::clamp(in, out, len, lb, ub);
  }

  void scopy_int8_to_fp32_u(unsigned int N, const uint8_t *X, unsigned int iX,
                            float *Y, unsigned int iY) override {
    nntrainer::scopy_int8_to_float32(N, X, iX, Y, iY);
  }
  void scopy_int8_to_fp32_s(unsigned int N, const int8_t *X, unsigned int iX,
                            float *Y, unsigned int iY) override {
    nntrainer::scopy_int8_to_float32(N, X, iX, Y, iY);
  }

  // Accelerator-only ops: CPU backends do not implement these — base
  // class defaults (throw + supports_*() = false) are correct.

#ifdef ENABLE_FP16
  void sgemm_fp16(unsigned int o, bool tA, bool tB, unsigned int M,
                  unsigned int N, unsigned int K, float a, const _FP16 *A,
                  unsigned int lda, const _FP16 *B, unsigned int ldb, float b,
                  _FP16 *C, unsigned int ldc) override {
    nntrainer::sgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void sgemv_fp16(unsigned int o, bool tA, unsigned int M, unsigned int N,
                  float a, const _FP16 *A, unsigned int lda, const _FP16 *X,
                  unsigned int iX, float b, _FP16 *Y,
                  unsigned int iY) override {
    nntrainer::sgemv(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }
  _FP16 sdot_fp16(unsigned int N, const _FP16 *X, unsigned int iX,
                  const _FP16 *Y, unsigned int iY) override {
    return nntrainer::sdot(N, X, iX, Y, iY);
  }
  void saxpy_fp16(unsigned int N, float a, const _FP16 *X, unsigned int iX,
                  _FP16 *Y, unsigned int iY) override {
    nntrainer::saxpy(N, a, X, iX, Y, iY);
  }
  void scopy_fp16(unsigned int N, const _FP16 *X, unsigned int iX, _FP16 *Y,
                  unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void scopy_fp32_to_fp16(unsigned int N, const float *X, unsigned int iX,
                          _FP16 *Y, unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void scopy_fp16_to_fp32(unsigned int N, const _FP16 *X, unsigned int iX,
                          float *Y, unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void sscal_fp16(unsigned int N, float a, _FP16 *X, unsigned int iX) override {
    nntrainer::sscal(N, a, X, iX);
  }
  _FP16 snrm2_fp16(unsigned int N, const _FP16 *X, unsigned int iX) override {
    return nntrainer::snrm2(N, X, iX);
  }
  unsigned int isamax_fp16(unsigned int N, const _FP16 *X,
                           unsigned int iX) override {
    return nntrainer::isamax(N, X, iX);
  }

  void ele_mul_fp16(unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_mul(N, X, Y, Z, a, b, is, os);
  }
  void ele_add_fp16(unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_add(N, X, Y, Z, a, b, is, os);
  }
  void ele_sub_fp16(unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_sub(N, X, Y, Z, a, b, is, os);
  }
  void ele_div_fp16(unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_div(N, X, Y, Z, a, b, is, os);
  }

  void swiglu_fp16(unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) override {
    nntrainer::swiglu(N, X, Y, Z);
  }
  _FP16 max_val_fp16(unsigned int N, _FP16 *X) override {
    return nntrainer::max_val(N, X);
  }
  void softmax_fp16(unsigned int N, _FP16 *X, _FP16 *Y) override {
    nntrainer::softmax(N, X, Y);
  }
  bool is_valid_fp16(unsigned int N, const _FP16 *X) override {
    return nntrainer::is_valid(N, X);
  }
  void inv_sqrt_inplace_fp16(unsigned int N, _FP16 *X) override {
    nntrainer::inv_sqrt_inplace(N, X);
  }

  void transpose_matrix_fp16(unsigned int M, unsigned int N, const _FP16 *s,
                             unsigned int lds, _FP16 *d,
                             unsigned int ldd) override {
    nntrainer::transpose_matrix(M, N, s, lds, d, ldd);
  }

  void scopy_int4_to_float16(unsigned int N, const uint8_t *X, unsigned int iX,
                             _FP16 *Y, unsigned int iY) override {
    nntrainer::scopy_int4_to_float16(N, X, iX, Y, iY);
  }
  void scopy_int8_to_float16_u(unsigned int N, const uint8_t *X,
                               unsigned int iX, _FP16 *Y,
                               unsigned int iY) override {
    nntrainer::scopy_int8_to_float16(N, X, iX, Y, iY);
  }
  void scopy_int8_to_float16_s(unsigned int N, const int8_t *X, unsigned int iX,
                               _FP16 *Y, unsigned int iY) override {
    nntrainer::scopy_int8_to_float16(N, X, iX, Y, iY);
  }

  void shgemm(unsigned int o, bool tA, bool tB, unsigned int M, unsigned int N,
              unsigned int K, float a, const float *A, unsigned int lda,
              const _FP16 *B, unsigned int ldb, float b, float *C,
              unsigned int ldc) override {
    nntrainer::shgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void shgemv(unsigned int o, bool tA, unsigned int M, unsigned int N, float a,
              const float *A, unsigned int lda, const _FP16 *X, unsigned int iX,
              float b, float *Y, unsigned int iY) override {
    nntrainer::shgemv(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }
  void hsgemm(unsigned int o, bool tA, bool tB, unsigned int M, unsigned int N,
              unsigned int K, float a, const _FP16 *A, unsigned int lda,
              const float *B, unsigned int ldb, float b, float *C,
              unsigned int ldc) override {
    nntrainer::hsgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void hsgemv(unsigned int o, bool tA, unsigned int M, unsigned int N, float a,
              const _FP16 *A, unsigned int lda, const float *X, unsigned int iX,
              float b, float *Y, unsigned int iY) override {
    nntrainer::hsgemv(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }

  void gemm_q4_0_fp16(unsigned int M, unsigned int N, unsigned int K,
                      const _FP16 *A, unsigned int lda, const void *B,
                      unsigned int ldb, _FP16 *C, unsigned int ldc) override {
    nntrainer::gemm_q4_0<_FP16>(M, N, K, A, lda, B, ldb, C, ldc);
  }
  void gemm_q6_K_fp16(unsigned int M, unsigned int N, unsigned int K,
                      const _FP16 *A, unsigned int lda, const void *B,
                      unsigned int ldb, _FP16 *C, unsigned int ldc) override {
    nntrainer::gemm_q6_K<_FP16>(M, N, K, A, lda, B, ldb, C, ldc);
  }

  void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                      unsigned int w, _FP16 *in, _FP16 *out,
                                      float *cos_, float *sin_) override {
    nntrainer::compute_rotary_embedding_value(dim, half_, w, in, out, cos_,
                                              sin_);
  }
#endif // ENABLE_FP16
};

ComputeOps *get_cpu_ops() {
  static CpuComputeOps instance;
  return &instance;
}

} // namespace nntrainer
