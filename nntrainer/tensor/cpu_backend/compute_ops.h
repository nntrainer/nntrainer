// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 NNtrainer contributors
 *
 * @file   compute_ops.h
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 * @brief  ComputeOps function pointer table for backend-agnostic dispatch
 *
 * Each Context (CPU/GPU/NPU) provides its own ComputeOps table.
 * Tensor operations call through this table instead of global functions,
 * enabling runtime dispatch to the correct backend (ARM NEON, x86 AVX,
 * OpenCL, CUDA, QNN/HMX, etc.) without #ifdef.
 */

#ifndef __COMPUTE_OPS_H__
#define __COMPUTE_OPS_H__
#ifdef __cplusplus

#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef ENABLE_FP16
#include <tensor_dim.h>
#endif

namespace nntrainer {

/**
 * @brief Function pointer table for compute operations.
 *
 * All function pointers follow the BLAS-standard parameter ordering.
 * FP32 and FP16 variants are stored as separate pointers since C++ function
 * overloading is not available for function pointers.
 *
 * Nullable pointers (set to nullptr) indicate the backend does not support
 * that operation; callers should fall back to a CPU implementation.
 */
struct ComputeOps {
  // =========================================================================
  // FP32 BLAS
  // =========================================================================
  void (*sgemm_fp32)(const unsigned int TStorageOrder, bool TransA, bool TransB,
                     const unsigned int M, const unsigned int N,
                     const unsigned int K, const float alpha, const float *A,
                     const unsigned int lda, const float *B,
                     const unsigned int ldb, const float beta, float *C,
                     const unsigned int ldc);

  void (*sgemv_fp32)(const unsigned int TStorageOrder, bool TransA,
                     const unsigned int M, const unsigned int N,
                     const float alpha, const float *A, const unsigned int lda,
                     const float *X, const unsigned int incX, const float beta,
                     float *Y, const unsigned int incY);

  float (*sdot_fp32)(const unsigned int N, const float *X,
                     const unsigned int incX, const float *Y,
                     const unsigned int incY);

  void (*saxpy_fp32)(const unsigned int N, const float alpha, const float *X,
                     const unsigned int incX, float *Y,
                     const unsigned int incY);

  void (*scopy_fp32)(const unsigned int N, const float *X,
                     const unsigned int incX, float *Y,
                     const unsigned int incY);

  void (*sscal_fp32)(const unsigned int N, const float alpha, float *X,
                     const unsigned int incX);

  float (*snrm2_fp32)(const unsigned int N, const float *X,
                      const unsigned int incX);

  unsigned int (*isamax_fp32)(const unsigned int N, const float *X,
                              const unsigned int incX);

  // =========================================================================
  // FP32 Element-wise
  // =========================================================================
  void (*ele_mul_fp32)(const unsigned int N, const float *X, const float *Y,
                       float *Z, float alpha, float beta,
                       unsigned int i_stride, unsigned int o_stride);

  void (*ele_add_fp32)(const unsigned int N, const float *X, const float *Y,
                       float *Z, float alpha, float beta,
                       unsigned int i_stride, unsigned int o_stride);

  void (*ele_sub_fp32)(const unsigned int N, const float *X, const float *Y,
                       float *Z, float alpha, float beta,
                       unsigned int i_stride, unsigned int o_stride);

  void (*ele_div_fp32)(const unsigned int N, const float *X, const float *Y,
                       float *Z, float alpha, float beta,
                       unsigned int i_stride, unsigned int o_stride);

  // =========================================================================
  // FP32 Activation / Special
  // =========================================================================
  void (*swiglu_fp32)(const unsigned int N, float *X, float *Y, float *Z);
  void (*swiglu_alpha_fp32)(const unsigned int N, float *X, float *Y, float *Z,
                            float alpha);
  void (*tanh_gelu_fp32)(const unsigned int N, const float *X, float *Y);
  void (*gelu_v2_fp32)(const unsigned int N, const float *X, float *Y);
  void (*tanh_gelu_v2_fp32)(const unsigned int N, const float *X, float *Y);
  void (*tanh_gelu_mul_fp32)(const unsigned int N, float *X, float *Y,
                             float *Z);
  void (*tanh_gelu_v2_mul_fp32)(const unsigned int N, float *X, float *Y,
                                float *Z);
  float (*max_val_fp32)(const unsigned int N, float *X);
  void (*softmax_fp32)(const unsigned int N, float *X, float *Y);
  bool (*is_valid_fp32)(const unsigned int N, const float *X);

  // =========================================================================
  // FP32 Matrix ops
  // =========================================================================
  void (*transpose_matrix_fp32)(const unsigned int M, const unsigned int N,
                                const float *src, unsigned int ld_src,
                                float *dst, unsigned int ld_dst);

  // =========================================================================
  // FP32 Data conversion / Copy
  // =========================================================================
  void (*scopy_u8)(const unsigned int N, const uint8_t *X,
                   const unsigned int incX, uint8_t *Y,
                   const unsigned int incY);
  void (*scopy_s8)(const unsigned int N, const int8_t *X,
                   const unsigned int incX, int8_t *Y,
                   const unsigned int incY);
  void (*scopy_int4_to_float32)(const unsigned int N, const uint8_t *X,
                                const unsigned int incX, float *Y,
                                const unsigned int incY);
  void (*copy_s16_fp32)(const unsigned int N, const int16_t *X, float *Y);
  void (*copy_u16_fp32)(const unsigned int N, const uint16_t *X, float *Y);
  void (*copy_fp32_u32)(const unsigned int N, const float *X, uint32_t *Y);
  void (*copy_fp32_u16)(const unsigned int N, const float *X, uint16_t *Y);
  void (*copy_fp32_u8)(const unsigned int N, const float *X, uint8_t *Y);
  void (*copy_fp32_s16)(const unsigned int N, const float *X, int16_t *Y);
  void (*copy_fp32_s8)(const unsigned int N, const float *X, int8_t *Y);

  // =========================================================================
  // Quantized GEMM (GGUF format)
  // =========================================================================
  void (*gemm_q4_0_fp32)(const unsigned int M, const unsigned int N,
                         const unsigned int K, const float *A,
                         const unsigned int lda, const void *B,
                         const unsigned int ldb, float *C,
                         const unsigned int ldc);

  void (*gemm_q4_K_fp32)(const unsigned int M, const unsigned int N,
                         const unsigned int K, const float *A,
                         const unsigned int lda, const void *B,
                         const unsigned int ldb, float *C,
                         const unsigned int ldc);

  void (*gemm_q6_K_fp32)(const unsigned int M, const unsigned int N,
                         const unsigned int K, const float *A,
                         const unsigned int lda, const void *B,
                         const unsigned int ldb, float *C,
                         const unsigned int ldc);

  // =========================================================================
  // Quantized weight packing / quantization
  // =========================================================================
  void (*unpack_q4_0)(const void *in_q4_0x, void *out_q4_0, size_t data_size,
                      const unsigned int M, const unsigned int N);

  void (*unpack_q4_0x8_transpose16)(const void *src, uint16_t *d_out,
                                    uint16_t *qs_out, int N, int K);

  size_t (*quantize_q4_0)(const float *src, void *dst, int64_t nrow,
                          int64_t n_per_row, const float *quant_weights);

  void (*dequantize_row_q4_0)(const void *x, float *y, int64_t k);

  void (*repack_q4_0)(void *dst, void *src, size_t data_size,
                      const unsigned int M, const unsigned int N);

  // =========================================================================
  // Clamp
  // =========================================================================
  void (*clamp_fp32)(const float *input, float *output, size_t length,
                     float lower_bound, float upper_bound);

  // =========================================================================
  // Data conversion (int8 → FP32)
  // =========================================================================
  void (*scopy_int8_to_fp32_u)(const unsigned int N, const uint8_t *X,
                               const unsigned int incX, float *Y,
                               const unsigned int incY);
  void (*scopy_int8_to_fp32_s)(const unsigned int N, const int8_t *X,
                               const unsigned int incX, float *Y,
                               const unsigned int incY);

  // =========================================================================
  // GPU-accelerated quantized ops (nullable — nullptr on CPU)
  // These function pointers have GPU-specific signatures.
  // When non-null, they provide batch/async acceleration for quantized GEMM.
  // =========================================================================

  /** @brief Batch async Q4_0 GEMM (e.g., OpenCL gemm_q4_0_async_cl) */
  void (*gemm_q4_0_batch_fp32)(std::vector<void *> matAdata, float *matBdata,
                               std::vector<float *> matCdata, unsigned int M,
                               std::vector<unsigned int> N, unsigned int K);

  /** @brief Single Q4_0 GEMM on accelerator (e.g., OpenCL gemm_q4_0_cl) */
  void (*gemm_q4_0_accel_fp32)(void *matAdata, float *matBdata,
                               float *matCdata, unsigned int M, unsigned int N,
                               unsigned int K);

  /** @brief Batch async INT4 GEMV (e.g., OpenCL gemv_int4_async_cl) */
  void (*gemv_int4_batch_fp32)(std::vector<void *> weights,
                               std::vector<uint16_t *> scales, float *input,
                               std::vector<float *> outputs, unsigned int K,
                               std::vector<unsigned int> Ns,
                               unsigned int group_size);

  /** @brief Batch async INT4 GEMM (e.g., OpenCL gemm_int4_async_cl) */
  void (*gemm_int4_batch_fp32)(float *input, std::vector<void *> weights,
                               std::vector<uint16_t *> scales,
                               std::vector<float *> matCdata, unsigned int M,
                               std::vector<unsigned int> Ns, unsigned int K,
                               unsigned int group_size);

  /** @brief Single INT4 GEMV on accelerator */
  void (*gemv_int4_accel_fp32)(char *weight, uint16_t *scale, float *input,
                               float *output, unsigned int K, unsigned int N,
                               unsigned int group_size);

  /** @brief Single INT4 SGEMM on accelerator */
  void (*sgemm_int4_accel_fp32)(float *input, char *weight, uint16_t *scale,
                                float *output, unsigned int M, unsigned int N,
                                unsigned int K, unsigned int group_size);

  // =========================================================================
  // Channel-wise INT4 GEMM (qsi4cxp kxn format, per-column fp32 scales)
  // =========================================================================
  // This slot consumes Int4QTensor's canonical layout directly:
  //   packed_data = K rows × ceil(N/2) bytes (kxn nibble order)
  //   scales      = N fp32 per-output-column scales
  // No transpose or repack needed.
  //
  // Filled by:
  //   ARM  → KleidiAI qsi4cxp_unpacked wrapper (or fallback C ref)
  //   x86  → AVX2 gemm_qsi4cxp_kxn_fp32 (x86_qsi4cxp.cpp)
  //   GPU  → nullptr (int4 CL kernel uses the SVM/accel path above)
  //
  // Nullable: when nullptr, FloatTensor::dotQInteger throws for QINT4
  // on non-SVM CPU path.
  void (*gemm_qsi4cxp_fp32)(unsigned int M, unsigned int N, unsigned int K,
                             const float *activation,
                             const void *packed_data,
                             const void *scales, float *output);

#ifdef ENABLE_FP16
  // =========================================================================
  // FP16 BLAS
  // =========================================================================
  void (*sgemm_fp16)(const unsigned int TStorageOrder, bool TransA,
                     bool TransB, const unsigned int M, const unsigned int N,
                     const unsigned int K, const float alpha, const _FP16 *A,
                     const unsigned int lda, const _FP16 *B,
                     const unsigned int ldb, const float beta, _FP16 *C,
                     const unsigned int ldc);

  void (*sgemv_fp16)(const unsigned int TStorageOrder, bool TransA,
                     const unsigned int M, const unsigned int N,
                     const float alpha, const _FP16 *A, const unsigned int lda,
                     const _FP16 *X, const unsigned int incX, const float beta,
                     _FP16 *Y, const unsigned int incY);

  _FP16 (*sdot_fp16)(const unsigned int N, const _FP16 *X,
                     const unsigned int incX, const _FP16 *Y,
                     const unsigned int incY);

  void (*saxpy_fp16)(const unsigned int N, const float alpha, const _FP16 *X,
                     const unsigned int incX, _FP16 *Y,
                     const unsigned int incY);

  void (*scopy_fp16)(const unsigned int N, const _FP16 *X,
                     const unsigned int incX, _FP16 *Y,
                     const unsigned int incY);

  void (*scopy_fp32_to_fp16)(const unsigned int N, const float *X,
                             const unsigned int incX, _FP16 *Y,
                             const unsigned int incY);

  void (*scopy_fp16_to_fp32)(const unsigned int N, const _FP16 *X,
                             const unsigned int incX, float *Y,
                             const unsigned int incY);

  void (*sscal_fp16)(const unsigned int N, const float alpha, _FP16 *X,
                     const unsigned int incX);

  _FP16 (*snrm2_fp16)(const unsigned int N, const _FP16 *X,
                      const unsigned int incX);

  unsigned int (*isamax_fp16)(const unsigned int N, const _FP16 *X,
                              const unsigned int incX);

  // =========================================================================
  // FP16 Element-wise
  // =========================================================================
  void (*ele_mul_fp16)(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                       _FP16 *Z, float alpha, float beta,
                       unsigned int i_stride, unsigned int o_stride);

  void (*ele_add_fp16)(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                       _FP16 *Z, float alpha, float beta,
                       unsigned int i_stride, unsigned int o_stride);

  void (*ele_sub_fp16)(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                       _FP16 *Z, float alpha, float beta,
                       unsigned int i_stride, unsigned int o_stride);

  void (*ele_div_fp16)(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                       _FP16 *Z, float alpha, float beta,
                       unsigned int i_stride, unsigned int o_stride);

  // =========================================================================
  // FP16 Activation / Special
  // =========================================================================
  void (*swiglu_fp16)(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z);
  _FP16 (*max_val_fp16)(const unsigned int N, _FP16 *X);
  void (*softmax_fp16)(const unsigned int N, _FP16 *X, _FP16 *Y);
  bool (*is_valid_fp16)(const unsigned int N, const _FP16 *X);
  void (*inv_sqrt_inplace_fp16)(const unsigned int N, _FP16 *X);

  // =========================================================================
  // FP16 Matrix ops
  // =========================================================================
  void (*transpose_matrix_fp16)(const unsigned int M, const unsigned int N,
                                const _FP16 *src, unsigned int ld_src,
                                _FP16 *dst, unsigned int ld_dst);

  // =========================================================================
  // FP16 Data conversion
  // =========================================================================
  void (*scopy_int4_to_float16)(const unsigned int N, const uint8_t *X,
                                const unsigned int incX, _FP16 *Y,
                                const unsigned int incY);

  void (*scopy_int8_to_float16_u)(const unsigned int N, const uint8_t *X,
                                  const unsigned int incX, _FP16 *Y,
                                  const unsigned int incY);

  void (*scopy_int8_to_float16_s)(const unsigned int N, const int8_t *X,
                                  const unsigned int incX, _FP16 *Y,
                                  const unsigned int incY);

  // =========================================================================
  // Mixed precision BLAS (F32 * F16 = F32, F16 * F32 = F32)
  // =========================================================================
  void (*shgemm)(const unsigned int TStorageOrder, bool TransA, bool TransB,
                 const unsigned int M, const unsigned int N,
                 const unsigned int K, const float alpha, const float *A,
                 const unsigned int lda, const _FP16 *B,
                 const unsigned int ldb, const float beta, float *C,
                 const unsigned int ldc);

  void (*shgemv)(const unsigned int TStorageOrder, bool TransA,
                 const unsigned int M, const unsigned int N, const float alpha,
                 const float *A, const unsigned int lda, const _FP16 *X,
                 const unsigned int incX, const float beta, float *Y,
                 const unsigned int incY);

  void (*hsgemm)(const unsigned int TStorageOrder, bool TransA, bool TransB,
                 const unsigned int M, const unsigned int N,
                 const unsigned int K, const float alpha, const _FP16 *A,
                 const unsigned int lda, const float *B,
                 const unsigned int ldb, const float beta, float *C,
                 const unsigned int ldc);

  void (*hsgemv)(const unsigned int TStorageOrder, bool TransA,
                 const unsigned int M, const unsigned int N, const float alpha,
                 const _FP16 *A, const unsigned int lda, const float *X,
                 const unsigned int incX, const float beta, float *Y,
                 const unsigned int incY);

  // =========================================================================
  // Quantized GEMM (FP16 variants)
  // =========================================================================
  void (*gemm_q4_0_fp16)(const unsigned int M, const unsigned int N,
                         const unsigned int K, const _FP16 *A,
                         const unsigned int lda, const void *B,
                         const unsigned int ldb, _FP16 *C,
                         const unsigned int ldc);

  void (*gemm_q6_K_fp16)(const unsigned int M, const unsigned int N,
                         const unsigned int K, const _FP16 *A,
                         const unsigned int lda, const void *B,
                         const unsigned int ldb, _FP16 *C,
                         const unsigned int ldc);

  // =========================================================================
  // Rotary embedding
  // =========================================================================
  void (*compute_rotary_embedding_value)(unsigned int dim, unsigned int half_,
                                        unsigned int w, _FP16 *in, _FP16 *out,
                                        float *cos_, float *sin_);
#endif // ENABLE_FP16
};

/**
 * @brief Global compute ops pointer.
 *
 * Set once during init_backend() and never changed afterwards.
 * When a Context-specific ops table is available (via ContextData),
 * that takes precedence over this global pointer.
 */
extern ComputeOps *g_compute_ops;

/**
 * @brief Ensure the global compute ops table is initialized.
 */
void ensureComputeOps();

/**
 * @brief Get the active compute ops table with lazy initialization.
 *
 * If g_compute_ops is nullptr (init_backend() not yet called),
 * automatically initializes it. Safe to call from any context.
 */
inline ComputeOps *getComputeOps() {
  if (__builtin_expect(g_compute_ops == nullptr, 0))
    ensureComputeOps();
  return g_compute_ops;
}

/**
 * @brief Initialize the CPU compute backend.
 *
 * Sets up architecture-specific resources (e.g., GGML, OpenBLAS threads)
 * and sets g_compute_ops to the appropriate ops table for the current
 * CPU architecture. Called once by Engine::add_default_object().
 */
void init_backend();

/**
 * @brief Backend-specific ops table getters.
 * Each backend defines its own getter in its ops_table.cpp file.
 */
ComputeOps *get_arm_ops();
ComputeOps *get_x86_ops();
ComputeOps *get_fallback_ops();

} // namespace nntrainer

#endif // __cplusplus
#endif // __COMPUTE_OPS_H__
