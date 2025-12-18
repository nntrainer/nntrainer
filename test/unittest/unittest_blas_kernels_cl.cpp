// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	unittest_blas_kernels_cl.cpp
 * @date	6 June 2024
 * @brief	Test setup for blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cstring>
#include <gtest/gtest.h>
#include <utility>

#include "fallback_internal.h"
#include "int4_utils.h"
#include "nntrainer_test_util.h"
#include "q4_0_utils.h"
#include "swiglu_cl.h"
#include "tensor_dim.h"
#include "timer.h"
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <cpu_backend.h>
#include <fp16.h>
#include <layer_context.h>
#include <tensor.h>

#include "unittest_util.h"
using namespace nntrainer;

// -----
// Functions
// -----

#define Q4_0 32

static std::pair<float, float>
dotCL_sgemv_test_func(const int batch, const int channel, const int height,
                      const int width, const int height_b, const int width_b,
                      const bool transA, const bool transB, const float alpha,
                      const int MOD) {
  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  return {mseError, static_cast<float>(cosSim)};
}

#ifdef ENABLE_FP16
static std::pair<float, float> dotCL_sgemv_fp16_test_func(
  const int batch, const int channel, const int height, const int width,
  const int height_b, const int width_b, const bool transA, const bool transB,
  const float alpha, const int MOD) {
  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp16, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp16, B_fp16, transA, transB);
  nntrainer::Tensor C_fp16 = A_fp16.dot(B_fp16, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  return {mseError, static_cast<float>(cosSim)};
}

static std::pair<float, float>
dot_gemm_fp16(const int batch, const int channel, const int height,
              const int width, const int height_b, const int width_b,
              const bool transA, const bool transB, const float alpha,
              const int MOD) {
  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);

  nntrainer::Tensor C = dotCl(A, B, transA, transB);
  nntrainer::Tensor C_fp16 = A.dot(B, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  return {mseError, static_cast<float>(cosSim)};
}
#endif

auto debug_print_data = [](const float *const data, const unsigned int len,
                           const uint32_t count = 5) {
  std::cout << "[";
  for (unsigned int i = 0; i < count; ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << "][";
  for (unsigned int i = len - count; i < len; ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << "]";
};

static inline void printMatrixF(const char *name, float *data, int Y, int X) {
  printf("%s :\n", name);
  for (int y = 0; y < Y; y++) {
    printf("[");
    for (int x = 0; x < X; x++) {
      std::cout << data[y * X + x] << " ";
    }
    printf("]\n");
  }
}

/**
 * @brief Helper function to print data
 *
 * @param data
 * @param size
 * @param count
 */
template <typename T = float>
static void debug_print_beg_end(const T *const data, const unsigned int size,
                                const uint32_t count = 5) {
  std::cout << "[";
  for (unsigned int i = 0; i < count; ++i) {
    std::cout << std::fixed << std::setprecision(3) << data[i] << " ";
  }
  std::cout << "][";
  for (unsigned int i = size - count; i < size; ++i) {
    std::cout << std::fixed << std::setprecision(3) << data[i] << " ";
  }
  std::cout << "]" << std::endl;
};

static void run_dequantization_test_(const uint32_t K, const uint32_t N) {
  const float epsilon = 0.01f;

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  std::vector<float> weight_fp32 =
    generate_random_vector<float>(N * K, -2.0f, 2.0f);

  // Dequantization Q4_0
  if (K % Q4_0 == 0 && N % 8 == 0) {
    size_t q4_data_size = K * N / Q4_0 * sizeof(block_q4_0);
    std::vector<uint8_t> q4_weight(q4_data_size);
    std::vector<uint8_t> q4_weight_repack(q4_data_size);
    nntrainer::quantize_q4_0(weight_fp32.data(), q4_weight.data(), N, K,
                             nullptr);
    nntrainer::repack_q4_0(q4_weight_repack.data(), q4_weight.data(),
                           q4_data_size, N, K);

    std::vector<float> dequantized_weights_q4(N * K);
    Q4_0Utils::dequantizeQ4_0x8(q4_weight_repack.data(), N, K,
                                dequantized_weights_q4.data());

    float mse_dequantized_q4 =
      mse<float>(weight_fp32.data(), dequantized_weights_q4.data(), N * K);

    std::cout << "MSE dequantized Q4_0: " << std::setprecision(10)
              << mse_dequantized_q4 << std::endl;

    EXPECT_IN_RANGE(mse_dequantized_q4, 0, epsilon);
  }

  // Dequantization INT4
  int scale_group_size = 32;
  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> quantized_scales;
  Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, scale_group_size,
                               quantized_weights, quantized_scales);

  std::vector<float> dequantized_weights_int4;
  Int4Utils::dequantizePacked(quantized_weights, quantized_scales, N, K,
                              scale_group_size, dequantized_weights_int4);

  // Dequantize QINT4 by row
  std::vector<float> dequantized_weights_int4_row(N * K, 0);
  for (int row_idx = 0; row_idx < N; ++row_idx) {
    Int4Utils::dequantizePackedRow(
      quantized_weights.data(), quantized_scales.data(), N, K, scale_group_size,
      row_idx, dequantized_weights_int4_row.data() + (K * row_idx));
  }

  float mse_dequantized_int4 =
    mse<float>(weight_fp32.data(), dequantized_weights_int4.data(), N * K);

  float mse_dequantized_int4_row =
    mse<float>(dequantized_weights_int4_row.data(), weight_fp32.data(), N * K);

  std::cout << "MSE dequantized INT4: " << std::setprecision(10)
            << mse_dequantized_int4 << std::endl;
  std::cout << "MSE dequantized INT4 by row: " << std::setprecision(10)
            << mse_dequantized_int4_row << std::endl;

  EXPECT_IN_RANGE(mse_dequantized_int4, 0, epsilon);
  EXPECT_IN_RANGE(mse_dequantized_int4_row, 0, epsilon);

  // This must be equal
  EXPECT_FLOAT_EQ(mse_dequantized_int4, mse_dequantized_int4_row);
}

#define DECLARE_dequantization_test_K_N(K, N)                                  \
  TEST(nntrainer_blas_kernel, dequantization_test_##K##_##N) {                 \
    run_dequantization_test_(K, N);                                            \
  }

DECLARE_dequantization_test_K_N(8192, 3072);
DECLARE_dequantization_test_K_N(3072, 8192);

DECLARE_dequantization_test_K_N(8188, 3068);
DECLARE_dequantization_test_K_N(3068, 8188);

DECLARE_dequantization_test_K_N(144, 168);

static void run_int4_gemv_test_(const uint32_t K, const uint32_t N,
                                int scale_group_size, bool use_ones = false,
                                bool print = false) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const int INT4_BLOCK_N_SIZE = 32;
  uint32_t alignN = align(N, INT4_BLOCK_N_SIZE);
  uint32_t alignK = align(K, scale_group_size);

  static constexpr uint32_t run_count = 200;

  // Allocate & initialize group-wise int4 data
  char *weight_ptr = (char *)allocateSVM(alignK * alignN / 2);
  uint16_t *scale_ptr = (uint16_t *)allocateSVM(ceilDiv(K, scale_group_size) *
                                                alignN * sizeof(uint16_t));
  uint16_t *input_ptr = (uint16_t *)allocateSVM(K * sizeof(uint16_t));
  uint16_t *output_ptr = (uint16_t *)allocateSVM(alignN * sizeof(uint16_t));

  std::vector<float> weight_fp32;
  std::vector<float> input_fp32;
  if (use_ones) {
    float ones_ratio = 0.1f;
    if (N * K > 1000000) { // For large input/output decrease ones_ratio to
                           // decrese results error
      ones_ratio = 0.01f;
    }
    weight_fp32 = generate_01_vector(N * K, ones_ratio);
    input_fp32 = generate_vector(K, 1.0f, 1.0f);
  } else {
    // weight_fp32 = generate_vector(N * K, -2.0f, 2.0f);
    // input_fp32 = generate_vector(K, -2.0f, 2.0f);
    weight_fp32 = generate_random_vector<float>(N * K, -1.0f, 1.0f);
    input_fp32 = generate_random_vector<float>(K, -1.0f, 1.0f);
  }

  // Print input weights ones
  if (print && use_ones) {
    for (int x = 0; x < K; x++) {
      for (int y = 0; y < N; y++) {
        if (y % 10 == 0) {
          printf("| ");
        }
        if (weight_fp32[y * K + x] > 0.1) {
          printf("1 ");
        } else {
          printf("0 ");
        }
      }
      printf("\n");
    }
  }

  // Reference SGEMV
  std::vector<float> reference_output_fp32(N);
  nntrainer::sgemv(0, false, N, K, 1.0f, weight_fp32.data(), K,
                   input_fp32.data(), 1, 0.0f, reference_output_fp32.data(), 1);

  // Reference Q4_0 GENV
  std::vector<float> q4_output_fp32(N);
  std::vector<float> q4_weight(N * K);
  std::vector<float> q4_weight_repack(N * K);
  if (K % 32 == 0 && N % 8 == 0) {
    nntrainer::quantize_q4_0(weight_fp32.data(), q4_weight.data(), N, K,
                             nullptr);
    nntrainer::repack_q4_0(q4_weight_repack.data(), q4_weight.data(),
                           K * N / 32 * 18, N, K);
    nntrainer::gemm_q4_0(1, N, K, input_fp32.data(), K, q4_weight_repack.data(),
                         N, q4_output_fp32.data(), N);
  }

  // GPU INT4 GEMV - MAIN TEST
  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> quantized_scales;
  Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, scale_group_size,
                               quantized_weights, quantized_scales);

  for (unsigned int i = 0; i < K; ++i) {
    input_ptr[i] = compute_fp32_to_fp16((input_fp32.data())[i]);
  }

  unsigned int scales_cnt = quantized_scales.size();
  std::vector<float> scales_fp32(ceilDiv(K, scale_group_size) * alignN);

  for (unsigned int i = 0; i < ceilDiv(K, scale_group_size) * alignN; ++i) {
    scale_ptr[i] = quantized_scales[i];
    scales_fp32[i] = compute_fp16_to_fp32(quantized_scales[i]);
  }

  for (unsigned int i = 0; i < alignN * alignK / 2; ++i) {
    weight_ptr[i] = quantized_weights[i];
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemv_int4_cl(weight_ptr, scale_ptr, input_ptr, output_ptr, K, N,
                            scale_group_size);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::vector<float> output_fp32(N, 0.0f);
  for (unsigned int i = 0; i < N; ++i) {
    output_fp32[i] = compute_fp16_to_fp32(output_ptr[i]);
  }

  std::cout << "INT4 GEMV : K:" << K << " x N:" << N << std::endl;
  std::cout << " - time : GPU = " << gpu_dt.count() / (run_count * 1.0f)
            << " ms" << std::endl;

  if (print) {
    if (use_ones) {
      printMatrixI("REF ", reference_output_fp32.data(), 1, N);
      printMatrixI("INT4", output_fp32.data(), 1, N);
    } else {
      std::cout << " - fp32 :   ";
      debug_print_beg_end(reference_output_fp32.data(), N, 16);
      if (K % 32 == 0 && N % 8 == 0) {
        std::cout << " - q4_0 :   ";
        debug_print_beg_end(q4_output_fp32.data(), N, 16);
      }
      std::cout << " - int4 :   ";
      debug_print_beg_end(output_fp32.data(), N, 16);
    }
  }

  if (K % 32 == 0 && N % 8 == 0) {
    float mse_q4_0_err =
      mse<float>(reference_output_fp32.data(), q4_output_fp32.data(), N);
    std::cout << "MSE q4_0: " << mse_q4_0_err << std::endl;
  }

  float mse_int4_err =
    mse<float>(reference_output_fp32.data(), output_fp32.data(), N);
  std::cout << "MSE int4: " << mse_int4_err << std::endl;

  if (use_ones) {
    EXPECT_IN_RANGE(mse_int4_err, 0.0f, 0.0001f);
  }

  freeSVM(weight_ptr);
  freeSVM(scale_ptr);
  freeSVM(input_ptr);
  freeSVM(output_ptr);
}

#define DECLARE_int4_gemv_test_K_N(K, N, G)                                    \
  TEST(nntrainer_blas_kernel, int4_gemv_test_##K##_##N##_Group##G) {           \
    run_int4_gemv_test_(K, N, G);                                              \
  }

#define DECLARE_int4_gemv_test_ones_K_N(K, N, G, print)                        \
  TEST(nntrainer_blas_kernel, int4_gemv_test_ones_##K##_##N##_Group##G) {      \
    run_int4_gemv_test_(K, N, G, true, print);                                 \
  }

DECLARE_int4_gemv_test_K_N(3072, 256, 32);
DECLARE_int4_gemv_test_K_N(3072, 8192, 32);
DECLARE_int4_gemv_test_K_N(8192, 3072, 32);
DECLARE_int4_gemv_test_K_N(3072, 3072, 32);

DECLARE_int4_gemv_test_K_N(3072, 256, 128);
DECLARE_int4_gemv_test_K_N(3072, 8192, 128);
DECLARE_int4_gemv_test_K_N(8192, 3072, 128);
DECLARE_int4_gemv_test_K_N(3072, 3072, 128);

DECLARE_int4_gemv_test_K_N(105920, 3072, 32);
DECLARE_int4_gemv_test_K_N(105900, 3072, 32);

DECLARE_int4_gemv_test_ones_K_N(105920, 3072, 32, false);
DECLARE_int4_gemv_test_ones_K_N(105900, 3072, 32, false);

DECLARE_int4_gemv_test_ones_K_N(64, 64, 32, false);

DECLARE_int4_gemv_test_ones_K_N(36, 40, 32, true);
DECLARE_int4_gemv_test_ones_K_N(28, 40, 32, false);
DECLARE_int4_gemv_test_ones_K_N(32, 40, 32, false);
DECLARE_int4_gemv_test_ones_K_N(144, 168, 32, false);

TEST(nntrainer_blas_kernel, tensor_dot_qint4) {
  const int batch = 1;
  const int channel = 1;
  const int height = 68;
  const int width = 3072;

  const int height_2 = 3072;
  const int width_2 = 256;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_qint4 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  /// Tensor A, B
  nntrainer::TensorPool pool;
  auto A_fp32_pool = pool.request(
    "A", nntrainer::TensorDim(batch, channel, height, width, t_type_nchw_fp32),
    {0}, nntrainer::TensorLifespan::MAX_LIFESPAN);

  /// FP32 weight (1 x 1 x 3072 x 256)
  auto B_fp32_pool = pool.request(
    "B",
    nntrainer::TensorDim(batch, channel, height_2, width_2, t_type_nchw_fp32),
    {0}, nntrainer::TensorLifespan::MAX_LIFESPAN);

  /// Tensor C, D
  auto C_fp32_pool = pool.request(
    "C", nntrainer::TensorDim(batch, channel, height, width, t_type_nchw_fp32),
    {0}, nntrainer::TensorLifespan::MAX_LIFESPAN);

  /// QINT4 weight  (1 x 1 x 3072 x 256)
  auto D_fp32_pool = pool.request(
    "D",
    nntrainer::TensorDim(batch, channel, height_2, width_2, t_type_nchw_qint4),
    {0}, nntrainer::TensorLifespan::MAX_LIFESPAN);

  /// FP32 outputs E, F
  auto E_fp32_pool = pool.request(
    "E",
    nntrainer::TensorDim(batch, channel, height, width_2, t_type_nchw_fp32),
    {0}, nntrainer::TensorLifespan::MAX_LIFESPAN);
  auto F_fp32_pool = pool.request(
    "F",
    nntrainer::TensorDim(batch, channel, height, width_2, t_type_nchw_fp32),
    {0}, nntrainer::TensorLifespan::MAX_LIFESPAN);

  pool.finalize(nntrainer::BasicPlanner(), 0, 4);
  pool.allocate();

  auto A_fp32 = *A_fp32_pool;
  auto B_fp32 = *B_fp32_pool;
  auto C_fp32 = *C_fp32_pool;
  auto D_fp32 = *D_fp32_pool;
  auto E_fp32 = *E_fp32_pool;
  auto F_fp32 = *F_fp32_pool;

  const int M = height;  // 68
  const int K = width;   // 3072
  const int N = width_2; // 256

  std::vector<float> weight_fp32 = generate_random_vector<float, false>(N * K);
  std::vector<float> input = generate_random_vector<float, false>(M * K);

  // Initialize A, C (fp32) input
  for (unsigned int i = 0; i < K * M; ++i) {
    A_fp32.getData()[i] = input[i];
    C_fp32.getData()[i] = input[i];
  }

  // Initialize B (fp32)
  for (unsigned int i = 0; i < K * N; ++i) {
    B_fp32.getData()[i] = weight_fp32[i];
  }

  nntrainer::Tensor tmp(1, 1, 256, 3072, t_type_nchw_fp32);

  /// Initialize D (QINT4)
  /// @note Must Transpose B before quantization
  B_fp32.transpose("0:2:1", tmp);

  std::vector<float> weight_fp32_T(N * K);
  for (unsigned int i = 0; i < K * N; ++i) {
    weight_fp32_T[i] = tmp.getData()[i];
  }

  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> quantized_scales;
  Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, 32, quantized_weights,
                               quantized_scales);

  int8_t *qdata = D_fp32.getData<int8_t>();
  uint16_t *scales = D_fp32.getScale<uint16_t>();

  for (unsigned int i = 0; i < N * K / 2; ++i) {
    qdata[i] = quantized_weights[i];
  }
  for (unsigned int i = 0; i < K * N / 32; ++i) {
    scales[i] = quantized_scales[i];
  }

  /// Perform dot mul
  A_fp32.dot(B_fp32, E_fp32);
  C_fp32.dot(D_fp32, F_fp32);

  // Compare output
  debug_print_beg_end(E_fp32.getData(), E_fp32.size());
  debug_print_beg_end(F_fp32.getData(), F_fp32.size());
}

static void run_int4_sgemv_test_(const uint32_t K, const uint32_t N,
                                 const int scale_group_size) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 200;

  // Allocate & initialize data
  char *weight_ptr = (char *)allocateSVM(K * N / 2);
  uint16_t *scale_ptr =
    (uint16_t *)allocateSVM(K * N / scale_group_size * sizeof(uint16_t));
  float *input_ptr = (float *)allocateSVM(K * sizeof(float));
  float *output_ptr = (float *)allocateSVM(N * sizeof(float));

  std::vector<float> weight_fp32 =
    generate_random_vector<float, false>(N * K, -2.0f, 2.0f);
  std::vector<float> input =
    generate_random_vector<float, false>(K, -2.0f, 2.0f);

  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> quantized_scales;
  Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, scale_group_size,
                               quantized_weights, quantized_scales);

  for (unsigned int i = 0; i < K; ++i) {
    input_ptr[i] = (input.data())[i];
  }

  for (unsigned int i = 0; i < K * N / scale_group_size; ++i) {
    scale_ptr[i] = quantized_scales[i];
  }

  for (unsigned int i = 0; i < N * K / 2; ++i) {
    weight_ptr[i] = quantized_weights[i];
  }

  // GPU INT4 GEMV
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemv_int4_cl(weight_ptr, scale_ptr, input_ptr, output_ptr, K, N,
                            scale_group_size);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::cout << "INT4 GEMV : " << K << " x " << N << std::endl;
  std::cout << " - time : GPU = " << gpu_dt.count() / (run_count * 1.0f)
            << " ms" << std::endl;

  std::cout << " - sample : [";
  for (unsigned int i = 0; i < 5; ++i) {
    std::cout << output_ptr[i] << " ";
  }
  std::cout << "][";
  for (unsigned int i = N - 5; i < N; ++i) {
    std::cout << output_ptr[i] << " ";
  }
  std::cout << "]" << std::endl;

  freeSVM(weight_ptr);
  freeSVM(scale_ptr);
  freeSVM(input_ptr);
  freeSVM(output_ptr);
}

#define DECLARE_int4_sgemv_test_K_N(K, N, G)                                   \
  TEST(nntrainer_blas_kernel, int4_sgemv_test_##K##_##N##_##G) {               \
    run_int4_sgemv_test_(K, N, G);                                             \
  }

DECLARE_int4_sgemv_test_K_N(3072, 256, 32);
DECLARE_int4_sgemv_test_K_N(3072, 3072, 128);

TEST(nntrainer_blas_kernel, int4_gemv_async_test) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 200;
  static constexpr int scale_group_size = 32;

  const int M = 1;
  const int K = 3072;
  const int N0 = 3072, N1 = 256;

  // Initialize Activation
  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  uint16_t *input = (uint16_t *)allocateSVM(M * K * sizeof(uint16_t));
  blas_cc->command_queue_inst_.enqueueSVMMap(input, M * K * sizeof(uint16_t),
                                             false);
  for (unsigned int i = 0; i < M * K; ++i) {
    input[i] = compute_fp32_to_fp16(activation[i]);
  }

  // Initialize Weight
  std::vector<float> weight0 = generate_random_vector<float, true>(N0 * K);
  std::vector<float> weight1 = generate_random_vector<float, true>(N1 * K);
  std::vector<float> weight2 = generate_random_vector<float, true>(N1 * K);
  size_t data_size_n0 = N0 * K * sizeof(uint16_t);
  size_t data_size_n1 = N1 * K * sizeof(uint16_t);

  // weight 0 (3072 x 3072)

  void *ws0 = allocateSVM(data_size_n0 / scale_group_size);
  void *wq0 = allocateSVM(data_size_n0 / 2);
  blas_cc->command_queue_inst_.enqueueSVMMap(wq0, data_size_n0 / 2, false);

  std::vector<uint8_t> quantized_weights0;
  std::vector<uint16_t> quantized_scales0;
  Int4Utils::quantizeAndRepack(weight0.data(), N0, K, scale_group_size,
                               quantized_weights0, quantized_scales0);

  for (unsigned int i = 0; i < K * N0 / scale_group_size; ++i) {
    ((uint16_t *)ws0)[i] = quantized_scales0[i];
  }

  for (unsigned int i = 0; i < N0 * K / 2; ++i) {
    ((int8_t *)wq0)[i] = quantized_weights0[i];
  }

  // weight 1 (3072 x 256)
  void *ws1 = allocateSVM(data_size_n1 / scale_group_size);
  void *wq1 = allocateSVM(data_size_n1 / 2);
  blas_cc->command_queue_inst_.enqueueSVMMap(wq1, data_size_n1 / 2, false);

  std::vector<uint8_t> quantized_weights1;
  std::vector<uint16_t> quantized_scales1;
  Int4Utils::quantizeAndRepack(weight1.data(), N1, K, scale_group_size,
                               quantized_weights1, quantized_scales1);

  for (unsigned int i = 0; i < K * N1 / scale_group_size; ++i) {
    ((uint16_t *)ws1)[i] = quantized_scales1[i];
  }

  for (unsigned int i = 0; i < N1 * K / 2; ++i) {
    ((int8_t *)wq1)[i] = quantized_weights1[i];
  }

  // weight 2 (3072 x 256)
  void *ws2 = allocateSVM(data_size_n1 / scale_group_size);
  void *wq2 = allocateSVM(data_size_n1 / 2);
  blas_cc->command_queue_inst_.enqueueSVMMap(wq2, data_size_n1 / 2, false);

  std::vector<uint8_t> quantized_weights2;
  std::vector<uint16_t> quantized_scales2;
  Int4Utils::quantizeAndRepack(weight2.data(), N1, K, scale_group_size,
                               quantized_weights2, quantized_scales2);

  for (unsigned int i = 0; i < K * N1 / scale_group_size; ++i) {
    ((uint16_t *)ws2)[i] = quantized_scales2[i];
  }

  for (unsigned int i = 0; i < N1 * K / 2; ++i) {
    ((int8_t *)wq2)[i] = quantized_weights2[i];
  }

  // Initialize Output data
  uint16_t *out0 = (uint16_t *)allocateSVM(M * N0 * sizeof(uint16_t));
  uint16_t *out1 = (uint16_t *)allocateSVM(M * N1 * sizeof(uint16_t));
  uint16_t *out2 = (uint16_t *)allocateSVM(M * N1 * sizeof(uint16_t));

  // In-order kernel execution
  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemv_int4_cl((char *)wq0, (uint16_t *)ws0, input, out0, K, N0,
                            scale_group_size);
    nntrainer::gemv_int4_cl((char *)wq1, (uint16_t *)ws1, input, out1, K, N1,
                            scale_group_size);
    nntrainer::gemv_int4_cl((char *)wq2, (uint16_t *)ws2, input, out2, K, N1,
                            scale_group_size);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  uint16_t *async_out0 = (uint16_t *)allocateSVM(M * N0 * sizeof(uint16_t));
  uint16_t *async_out1 = (uint16_t *)allocateSVM(M * N1 * sizeof(uint16_t));
  uint16_t *async_out2 = (uint16_t *)allocateSVM(M * N1 * sizeof(uint16_t));

  std::vector<void *> weight_vec = {wq0, wq1, wq2};
  std::vector<uint16_t *> scale_vec = {(uint16_t *)ws0, (uint16_t *)ws1,
                                       (uint16_t *)ws2};
  std::vector<uint16_t *> out_vec = {async_out0, async_out1, async_out2};
  std::vector<unsigned int> n_vec = {N0, N1, N1};

  // Async
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemv_int4_async_cl(weight_vec, scale_vec, input, out_vec, K,
                                  n_vec, scale_group_size);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::cout << "INT4 GEMV : " << M << " x " << K << " x " << N1 << std::endl;
  std::cout << " - time : Orig = " << dt.count() / (run_count * 1.0f) << " ms"
            << std::endl;

  std::cout << " - out0 : [";
  for (unsigned int i = 0; i < 5; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(out0[i]) << " ";
  }
  std::cout << "][";
  for (unsigned int i = N0 - 5; i < N0; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(out0[i]) << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << " - out1 : [";
  for (unsigned int i = 0; i < 5; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(out1[i]) << " ";
  }
  std::cout << "][";
  for (unsigned int i = N1 - 5; i < N1; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(out1[i]) << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << " - out2 : [";
  for (unsigned int i = 0; i < 5; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(out2[i]) << " ";
  }
  std::cout << "][";
  for (unsigned int i = N1 - 5; i < N1; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(out2[i]) << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << " - time : Async = " << gpu_dt.count() / (run_count * 1.0f)
            << " ms" << std::endl;

  std::cout << " - out0 : [";
  for (unsigned int i = 0; i < 5; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(async_out0[i]) << " ";
  }
  std::cout << "][";
  for (unsigned int i = N0 - 5; i < N0; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(async_out0[i]) << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << " - out1 : [";
  for (unsigned int i = 0; i < 5; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(async_out1[i]) << " ";
  }
  std::cout << "][";
  for (unsigned int i = N1 - 5; i < N1; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(async_out1[i]) << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << " - out2 : [";
  for (unsigned int i = 0; i < 5; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(async_out2[i]) << " ";
  }
  std::cout << "][";
  for (unsigned int i = N1 - 5; i < N1; ++i) {
    std::cout << std::fixed << std::setprecision(3)
              << compute_fp16_to_fp32(async_out2[i]) << " ";
  }
  std::cout << "]" << std::endl;

  // Free allocated SVM
  freeSVM(input);
  freeSVM(ws0);
  freeSVM(wq0);
  freeSVM(ws1);
  freeSVM(wq1);
  freeSVM(ws2);
  freeSVM(wq2);
  freeSVM(out0);
  freeSVM(out1);
  freeSVM(out2);
  freeSVM(async_out0);
  freeSVM(async_out1);
  freeSVM(async_out2);
}

static void run_int4_gemm_test_(const uint32_t M, const uint32_t K,
                                const uint32_t N, const int scale_group_size,
                                bool use_ones = false, bool print = false) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const int INT4_BLOCK_N_SIZE = 32;
  uint32_t alignN = align(N, INT4_BLOCK_N_SIZE);
  uint32_t alignK = align(K, scale_group_size);

  static constexpr uint32_t run_count = 200;

  uint32_t input_size = M * alignK;

  std::vector<float> input;
  std::vector<float> weight_fp32;
  if (use_ones) {
    float ones_ratio = 0.1f;
    if (M * N * K > 1000000) { // For large input/output decrease ones_ratio to
                               // decrese results error
      ones_ratio = 0.01f;
    }
    // input = generate_vector(input_size, 1.0f, 1.0f);
    input = generate_01_vector(input_size, ones_ratio);
    weight_fp32 = generate_01_vector(N * K, ones_ratio);
  } else {
    input = generate_random_vector<float, false>(input_size, -1.0, 1.0);
    weight_fp32 = generate_random_vector<float, false>(N * K, -1.0, 1.0);
    // input = generate_vector(input_size, -2.0f, 2.0f);
    // weight_fp32 = generate_vector(N * K, -2.0f, 2.0f);
  }

  // Print input weights ones
  if (print && use_ones) {
    for (int x = 0; x < K; x++) {
      for (int y = 0; y < N; y++) {
        if (y % 10 == 0) {
          printf("| ");
        }
        if (weight_fp32[y * K + x] > 0.1) {
          printf("1 ");
        } else {
          printf("0 ");
        }
      }
      printf("\n");
    }
  }

  // Reference SGEMM
  std::vector<float> ref_dst(M * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, input.data(), K,
                   weight_fp32.data(), K, 0.F, ref_dst.data(), N);

  // Reference Q4_0 GEMM
  if (K % Q4_0 == 0 && N % 8 == 0) {
    size_t q4_data_size = K * N / Q4_0 * sizeof(block_q4_0);
    std::vector<float> q4_output_fp32(M * N);
    std::vector<uint8_t> q4_weight(q4_data_size);
    std::vector<uint8_t> q4_weight_repack(q4_data_size);
    nntrainer::quantize_q4_0(weight_fp32.data(), q4_weight.data(), N, K,
                             nullptr);
    nntrainer::repack_q4_0(q4_weight_repack.data(), q4_weight.data(),
                           q4_data_size, N, K);
    nntrainer::gemm_q4_0(M, N, K, input.data(), K, q4_weight_repack.data(), N,
                         q4_output_fp32.data(), N);
    float mse_q4 = mse<float>(ref_dst.data(), q4_output_fp32.data(), M * N);
    std::cout << "MSE Q4_0: " << std::setprecision(10) << mse_q4 << std::endl;
  }

  // Int4 GEMM - THE MAIN TEST
  uint16_t *input_ptr = (uint16_t *)allocateSVM(input_size * sizeof(uint16_t));
  int8_t *weight_ptr = (int8_t *)allocateSVM(alignK * alignN / 2);
  uint16_t *scale_ptr = (uint16_t *)allocateSVM(ceilDiv(K, scale_group_size) *
                                                alignN * sizeof(uint16_t));
  uint16_t *output_ptr = (uint16_t *)allocateSVM(M * alignN * sizeof(uint16_t));

  blas_cc->command_queue_inst_.enqueueSVMMap(
    input_ptr, input_size * sizeof(uint16_t), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(weight_ptr, alignK * alignN / 2,
                                             false);
  blas_cc->command_queue_inst_.enqueueSVMMap(
    scale_ptr, ceilDiv(K, scale_group_size) * alignN * sizeof(uint16_t), false);

  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> quantized_scales;
  Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, scale_group_size,
                               quantized_weights, quantized_scales);

  for (unsigned int i = 0; i < input_size; ++i) {
    input_ptr[i] = compute_fp32_to_fp16((input.data())[i]);
  }

  for (unsigned int i = 0; i < ceilDiv(K, scale_group_size) * alignN; ++i) {
    scale_ptr[i] = quantized_scales[i];
  }

  for (unsigned int i = 0; i < alignN * align(K, scale_group_size) / 2; ++i) {
    weight_ptr[i] = quantized_weights[i];
  }

  blas_cc->command_queue_inst_.enqueueSVMUnmap(input_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(weight_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(scale_ptr);
  // GPU INT4 GEMM
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::openvino_gemm_cl(input_ptr, weight_ptr, scale_ptr, output_ptr, M,
                                N, K, scale_group_size);
  }

  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt =
    std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

  std::vector<float> output_fp32(M * N);
  for (unsigned int i = 0; i < M * N; ++i) {
    output_fp32[i] = compute_fp16_to_fp32(output_ptr[i]);
  }

  uint32_t first_zero_index = UINT32_MAX;
  uint32_t first_nonzero_index = UINT32_MAX;
  int zeros = 0;
  int non_zeros = 0;
  int nans = 0;

  for (uint32_t i = 0; i < M * N; ++i) {
    if (compute_fp16_to_fp32(output_ptr[i]) == 0) {
      zeros++;
      if (first_zero_index == UINT32_MAX) {
        first_zero_index = i;
      }
    } else {
      non_zeros++;
      if (first_nonzero_index == UINT32_MAX) {
        first_nonzero_index = i;
      }
    }

    if (std::isnan(compute_fp16_to_fp32(output_ptr[i]))) {
      nans++;
    }
  }

  auto debug_print_beg_end = [M, K, N](const uint16_t *const data,
                                       const uint32_t count = 6) {
    std::cout << "[";
    for (unsigned int i = 0; i < count; ++i) {
      std::cout << compute_fp16_to_fp32(data[i]) << " ";
    }
    std::cout << "][";
    for (unsigned int i = M * N - count; i < M * N; ++i) {
      std::cout << compute_fp16_to_fp32(data[i]) << " ";
    }
    std::cout << "]";
  };

  auto debug_print_beg_end_float = [M, K, N](const float *const data,
                                             const uint32_t count = 6) {
    std::cout << "[";
    for (unsigned int i = 0; i < count; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << "][";
    for (unsigned int i = M * N - count; i < M * N; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << "]";
  };

  std::cout << "INT4 GEMM : M:" << M << " x K:" << K << " x N:" << N
            << std::endl;
  std::cout << " - time : GPU = " << gpu_dt / (run_count * 1.0f) << " ms"
            << std::endl;

  std::vector<float> diff(M * N);
  float maxDiff = 0;
  for (int i = 0; i < M * N; i++) {
    diff[i] = ref_dst[i] - output_fp32[i];
    if (abs(diff[i]) > maxDiff) {
      maxDiff = diff[i];
    }
  }

  if (print) {
    if (use_ones) {
      printMatrixI("REF ", ref_dst.data(), M, N);
      printMatrixI("INT4", output_fp32.data(), M, N);
      printMatrixI("DIFF", diff.data(), M, N);
    } else {
      std::cout << " - sample: REF = ";
      debug_print_beg_end_float(ref_dst.data(), 16);
      std::cout << std::endl;
      std::cout << " - sample : GPU = ";
      debug_print_beg_end(output_ptr, 16);
      std::cout << std::endl;

      std::cout << " - zeros : " << zeros << " / " << M * N << " [ "
                << zeros * 100.0f / float(M * N) << " %] - first at [ "
                << first_zero_index << " ]" << std::endl;
      std::cout << " - non zeros : " << non_zeros << " / " << M * N << " [ "
                << non_zeros * 100.0f / float(M * N) << " %] - first at [ "
                << first_nonzero_index << " ]" << std::endl;
      std::cout << " - nans : " << nans << " / " << M * N << " [ "
                << nans * 100.0f / float(M * N) << " %]" << std::endl;
    }
  }
  printf("maxDiff:%f\n", maxDiff);

  float mse_int4_err = mse<float>(ref_dst.data(), output_fp32.data(), M * N);
  std::cout << "MSE int4: " << mse_int4_err << std::endl;

  if (use_ones) {
    EXPECT_IN_RANGE(mse_int4_err, 0.0f, 0.0001f);
  }

  freeSVM(weight_ptr);
  freeSVM(scale_ptr);
  freeSVM(input_ptr);
  freeSVM(output_ptr);
}

#define DECLARE_int4_gemm_test_M_K_N(M, K, N, G)                               \
  TEST(nntrainer_blas_kernel, int4_gemm_test_##M##_##K##_##N##_Group##G) {     \
    run_int4_gemm_test_(M, K, N, G);                                           \
  }

#define DECLARE_int4_gemm_test_ones_M_K_N(M, K, N, G, print)                   \
  TEST(nntrainer_blas_kernel,                                                  \
       int4_gemm_test_ones_##M##_##K##_##N##_Group##G) {                       \
    run_int4_gemm_test_(M, K, N, G, true, print);                              \
  }

DECLARE_int4_gemm_test_M_K_N(28, 3072, 256, 32);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 8192, 32);
DECLARE_int4_gemm_test_M_K_N(28, 8192, 3072, 32);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 3072, 32);
DECLARE_int4_gemm_test_M_K_N(28, 32, 3072, 32);

DECLARE_int4_gemm_test_M_K_N(28, 3072, 256, 128);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 8192, 128);
DECLARE_int4_gemm_test_M_K_N(28, 8192, 3072, 128);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 3072, 128);

DECLARE_int4_gemm_test_M_K_N(4, 3060, 3072, 32);
DECLARE_int4_gemm_test_M_K_N(4, 3072, 3072, 32);

DECLARE_int4_gemm_test_ones_M_K_N(3, 105900, 3072, 32, false);
DECLARE_int4_gemm_test_ones_M_K_N(3, 105920, 3072, 32, false);

DECLARE_int4_gemm_test_ones_M_K_N(4, 64, 64, 32, false);

DECLARE_int4_gemm_test_ones_M_K_N(4, 36, 40, 32, true);
DECLARE_int4_gemm_test_ones_M_K_N(4, 28, 40, 32, false);
DECLARE_int4_gemm_test_ones_M_K_N(10, 32, 40, 32, false);
DECLARE_int4_gemm_test_ones_M_K_N(10, 144, 168, 32, false);

TEST(nntrainer_blas_kernel, int4_gemm_async_test) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 200;
  static constexpr int scale_group_size = 32;

  const int M = 68;
  const int K = 3072;
  const int N0 = 3072, N1 = 256;

  // Initialize Activation
  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  float *activations_f32_ptr = (float *)allocateSVM(M * K * sizeof(float));
  blas_cc->command_queue_inst_.enqueueSVMMap(activations_f32_ptr,
                                             M * K * sizeof(float), false);
  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  // Initialize Weight
  std::vector<float> weight0 = generate_random_vector<float, true>(N0 * K);
  std::vector<float> weight1 = generate_random_vector<float, true>(N1 * K);
  std::vector<float> weight2 = generate_random_vector<float, true>(N1 * K);
  size_t data_size_n0 = N0 * K * sizeof(uint16_t);
  size_t data_size_n1 = N1 * K * sizeof(uint16_t);

  // weight 0 (3072 x 3072)
  void *ws0 = allocateSVM(data_size_n0 / scale_group_size);
  void *wq0 = allocateSVM(data_size_n0 / 2);
  blas_cc->command_queue_inst_.enqueueSVMMap(wq0, data_size_n0 / 2, false);

  std::vector<uint8_t> quantized_weights0;
  std::vector<uint16_t> quantized_scales0;
  Int4Utils::quantizeAndRepack(weight0.data(), N0, K, scale_group_size,
                               quantized_weights0, quantized_scales0);

  for (unsigned int i = 0; i < K * N0 / scale_group_size; ++i) {
    ((uint16_t *)ws0)[i] = quantized_scales0[i];
  }

  for (unsigned int i = 0; i < N0 * K / 2; ++i) {
    ((int8_t *)wq0)[i] = quantized_weights0[i];
  }

  // weight 1 (3072 x 256)
  void *ws1 = allocateSVM(data_size_n1 / scale_group_size);
  void *wq1 = allocateSVM(data_size_n1 / 2);
  blas_cc->command_queue_inst_.enqueueSVMMap(wq1, data_size_n1 / 2, false);

  std::vector<uint8_t> quantized_weights1;
  std::vector<uint16_t> quantized_scales1;
  Int4Utils::quantizeAndRepack(weight1.data(), N1, K, scale_group_size,
                               quantized_weights1, quantized_scales1);

  for (unsigned int i = 0; i < K * N1 / scale_group_size; ++i) {
    ((uint16_t *)ws1)[i] = quantized_scales1[i];
  }

  for (unsigned int i = 0; i < N1 * K / 2; ++i) {
    ((int8_t *)wq1)[i] = quantized_weights1[i];
  }

  // weight 2 (3072 x 256)
  void *ws2 = allocateSVM(data_size_n1 / scale_group_size);
  void *wq2 = allocateSVM(data_size_n1 / 2);
  blas_cc->command_queue_inst_.enqueueSVMMap(wq2, data_size_n1 / 2, false);

  std::vector<uint8_t> quantized_weights2;
  std::vector<uint16_t> quantized_scales2;
  Int4Utils::quantizeAndRepack(weight2.data(), N1, K, scale_group_size,
                               quantized_weights2, quantized_scales2);

  for (unsigned int i = 0; i < K * N1 / scale_group_size; ++i) {
    ((uint16_t *)ws2)[i] = quantized_scales2[i];
  }

  for (unsigned int i = 0; i < N1 * K / 2; ++i) {
    ((int8_t *)wq2)[i] = quantized_weights2[i];
  }

  // Initialize Output data
  float *out0 = (float *)allocateSVM(M * N0 * sizeof(float));
  float *out1 = (float *)allocateSVM(M * N1 * sizeof(float));
  float *out2 = (float *)allocateSVM(M * N1 * sizeof(float));

  // In-order kernel execution
  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::openvino_sgemm_cl(activations_f32_ptr, (char *)wq0,
                                 (uint16_t *)ws0, out0, M, N0, K,
                                 scale_group_size);
    nntrainer::openvino_sgemm_cl(activations_f32_ptr, (char *)wq1,
                                 (uint16_t *)ws1, out1, M, N1, K,
                                 scale_group_size);
    nntrainer::openvino_sgemm_cl(activations_f32_ptr, (char *)wq2,
                                 (uint16_t *)ws2, out2, M, N1, K,
                                 scale_group_size);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  float *async_out0 = (float *)allocateSVM(M * N0 * sizeof(float));
  float *async_out1 = (float *)allocateSVM(M * N1 * sizeof(float));
  float *async_out2 = (float *)allocateSVM(M * N1 * sizeof(float));

  std::vector<void *> weight_vec = {wq0, wq1, wq2};
  std::vector<uint16_t *> scale_vec = {(uint16_t *)ws0, (uint16_t *)ws1,
                                       (uint16_t *)ws2};
  std::vector<float *> out_vec = {async_out0, async_out1, async_out2};
  std::vector<unsigned int> n_vec = {N0, N1, N1};

  // Async
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::openvino_gemm_async_cl(activations_f32_ptr, weight_vec,
                                      scale_vec, out_vec, M, n_vec, K,
                                      scale_group_size);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::cout << "Q4_0 GEMM : " << M << " x " << K << " x " << N1 << std::endl;
  std::cout << " - time : Orig = " << dt.count() / (run_count * 1.0f) << " ms"
            << std::endl;
  std::cout << " - time : Async = " << gpu_dt.count() / (run_count * 1.0f)
            << " ms" << std::endl;

  debug_print_beg_end(out0, M * N0);
  debug_print_beg_end(out1, M * N1);
  debug_print_beg_end(out2, M * N1);

  debug_print_beg_end(async_out0, M * N0);
  debug_print_beg_end(async_out1, M * N1);
  debug_print_beg_end(async_out2, M * N1);

  // Free allocated SVM
  freeSVM(activations_f32_ptr);
  freeSVM(ws0);
  freeSVM(wq0);
  freeSVM(ws1);
  freeSVM(wq1);
  freeSVM(ws2);
  freeSVM(wq2);
  freeSVM(out0);
  freeSVM(out1);
  freeSVM(out2);
  freeSVM(async_out0);
  freeSVM(async_out1);
  freeSVM(async_out2);
}

TEST(blas_kernels, dotCL_sgemv_M_1_1) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_M_1_2) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_1) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 768;
  const int width_b = 1;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_2) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 2048;
  const int width_b = 1;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_n) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  EXPECT_THROW(dotCl(A_fp32, B_fp32, transA, transB), std::runtime_error);
}

TEST(blas_kernels, dotCL_sgemv_N_1_M_1_1) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 1;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  auto gen_data = [](nntrainer::Tensor x) {
    auto ptr = x.getData();
    for (unsigned int i = 0; i < x.size(); ++i) {
      ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  };

  gen_data(A_fp32), gen_data(B_fp32);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float err = mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-5 * width;

  EXPECT_IN_RANGE(err, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_M_1_2) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 1;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-5 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_noTrans) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transB) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_transA) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transAB) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = true;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, addition_i) {
  const int batch = 1;
  const int channel = 1;
  const int height = 64;
  const int width = 3072;

  const int batch_b = 1;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch_b, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor C_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor D_fp32(batch_b, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(B_fp32, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);
  GEN_TEST_INPUT(C_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(D_fp32, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);

  A_fp32.add_i(B_fp32);
  add_i_cl(C_fp32, D_fp32);

  float mseError =
    mse<float>(A_fp32.getData<float>(), C_fp32.getData<float>(), A_fp32.size());

  double cosSim = cosine_similarity<float>(
    A_fp32.getData<float>(), C_fp32.getData<float>(), A_fp32.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, l2norm) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 768;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  GEN_TEST_INPUT(B_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  float gpu_result = nrm2Cl(A_fp32);
  float cpu_result = B_fp32.l2norm();

  EXPECT_FLOAT_EQ(gpu_result, cpu_result);
}

TEST(blas_kernels, absolute_sum) {
  const int batch = 1;
  const int channel = 1;
  const int height = 32;
  const int width = 32;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  float cpu_result = 0.0f;
  for (size_t i = 0; i < A_fp32.size(); ++i) {
    cpu_result += fabs(A_fp32.getData<float>()[i]);
  }

  float gpu_result = asumCl(A_fp32);

  EXPECT_FLOAT_EQ(cpu_result, gpu_result);
}

TEST(blas_kernels, rmsnorm_fp32) {
  const int batch = 1;
  const int channel = 1;
  const int height = 67;
  const int width = 3072;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor in_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width, t_type_nchw_fp32);
  nntrainer::Tensor out_cl_fp32(batch, channel, height, width,
                                t_type_nchw_fp32);
  nntrainer::Tensor out_ref_fp32(batch, channel, height, width,
                                 t_type_nchw_fp32);

  /// Initialize CPU input data
  GEN_TEST_INPUT(in_fp32, ((i * (batch * height * channel) +
                            j * (batch * height) + k * (width) + l + 1) %
                           MOD) *
                            alpha);
  for (int l = 0; l < width; ++l) {
    float val = ((l + 1) % MOD) * alpha;
    gamma_fp32.setValue(0, 0, 0, l, val);
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  /// Initialize GPU input/ouput data
  void *in_fp32_svm = allocateSVM(in_fp32.size() * sizeof(float));
  void *gamma_fp32_svm = allocateSVM(gamma_fp32.size() * sizeof(float));
  void *out_fp32_svm = allocateSVM(out_cl_fp32.size() * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(
    in_fp32_svm, in_fp32.size() * sizeof(float), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(
    gamma_fp32_svm, gamma_fp32.size() * sizeof(float), false);

  std::memcpy(in_fp32_svm, in_fp32.getData<float>(),
              in_fp32.size() * sizeof(float));
  std::memcpy(gamma_fp32_svm, gamma_fp32.getData<float>(),
              gamma_fp32.size() * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMUnmap(in_fp32_svm);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(gamma_fp32_svm);

  static constexpr uint32_t run_count = 500;
  static constexpr float kEpsilon = 0.001f;

  Timer timer1{};
  for (unsigned int i = 0; i < run_count; ++i) {
    rmsnorm_cl((float *)in_fp32_svm, (float *)gamma_fp32_svm,
               (float *)out_fp32_svm, kEpsilon,
               in_fp32.batch() * in_fp32.channel() * in_fp32.height(),
               in_fp32.width(), true);
  }
  auto t2_cl = timer1.GetElapsedMilliseconds();

  Timer timer2{};
  for (unsigned int i = 0; i < run_count; ++i) {
    std::function<float(float)> f = [](float x) { return 1 / std::sqrt(x); };
    auto t = in_fp32.multiply(in_fp32).average(3).add(kEpsilon);
    t.apply_i(f);
    in_fp32.multiply(t, out_ref_fp32);
    out_ref_fp32.multiply_i(gamma_fp32);
  }
  auto t2_ref = timer2.GetElapsedMilliseconds();

  std::cout << "RMSNorm time : GPU = " << t2_cl / (run_count * 1.0f) << " ms"
            << std::endl;

  std::cout << "RMSNorm time : CPU = " << t2_ref / (run_count * 1.0f) << " ms"
            << std::endl;

  blas_cc->command_queue_inst_.enqueueSVMMap(
    out_fp32_svm, out_cl_fp32.size() * sizeof(float), false);

  float mseError = mse<float>((float *)out_fp32_svm,
                              out_ref_fp32.getData<float>(), height * width);

  double cosSim = cosine_similarity<float>(
    (float *)out_fp32_svm, out_ref_fp32.getData<float>(), height * width);

  const float epsilon = 1e-3 * width;

  freeSVM(out_fp32_svm);
  freeSVM(in_fp32_svm);
  freeSVM(gamma_fp32_svm);

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, swiglu_layer_fp32_67_3072) {
  const int batch = 1;
  const int channel = 1;
  const int height = 67;
  const int width = 3072;

  const int dim = width * height;

  const int batch_b = 1;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  /// Initialize CPU input/output
  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch_b, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor out_ref_fp32(batch, channel, height, width,
                                 t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(B_fp32, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);

  /// Initialize GPU input/output
  void *gpu_in1 = allocateSVM(dim * sizeof(float));
  void *gpu_in2 = allocateSVM(dim * sizeof(float));
  void *gpu_dst = allocateSVM(dim * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(gpu_in1, dim * sizeof(float),
                                             false);
  blas_cc->command_queue_inst_.enqueueSVMMap(gpu_in2, dim * sizeof(float),
                                             false);

  std::memcpy(gpu_in1, A_fp32.getData(), dim * sizeof(float));
  std::memcpy(gpu_in2, B_fp32.getData(), dim * sizeof(float));

  static constexpr uint32_t run_count = 500;

  SwiGLULayerCl layer;

  auto t1_cl = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    layer.swiglu_cl((float *)gpu_in1, (float *)gpu_in2, (float *)gpu_dst, width,
                    height, true);
  }
  auto t2_cl = std::chrono::high_resolution_clock::now();

  auto t1_ref = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::swiglu(height * width, out_ref_fp32.getData(), A_fp32.getData(),
                      B_fp32.getData());
  }
  auto t2_ref = std::chrono::high_resolution_clock::now();

  auto dt_cl =
    std::chrono::duration_cast<std::chrono::microseconds>(t2_cl - t1_cl);
  auto dt_ref =
    std::chrono::duration_cast<std::chrono::microseconds>(t2_ref - t1_ref);

  std::cout << "Swiglu time : GPU = " << dt_cl.count() / (run_count * 1000.0f)
            << " ms" << std::endl;

  std::cout << "Swiglu time : CPU = " << dt_ref.count() / (run_count * 1000.0f)
            << " ms" << std::endl;

  float mseError =
    mse<float>((float *)gpu_dst, out_ref_fp32.getData<float>(), height * width);

  double cosSim = cosine_similarity<float>(
    (float *)gpu_dst, out_ref_fp32.getData<float>(), height * width);

  const float epsilon = 1e-3 * width;

  freeSVM(gpu_in1);
  freeSVM(gpu_in2);
  freeSVM(gpu_dst);

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

#ifdef ENABLE_FP16

TEST(blas_kernels, dotCL_sgemv_M_1_1_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_M_1_2_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_1_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 768;
  const int width_b = 1;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_2_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 2048;
  const int width_b = 1;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_n_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp16, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  EXPECT_THROW(dotCl(A_fp16, B_fp16, transA, transB), std::runtime_error);
}

TEST(blas_kernels, sgemv_3072_20120_noTrans) {
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 3072;

  int height_b = 3072;
  int width_b = 20120;

  bool transA = false;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, multiply_i) {
  const int batch = 1;
  const int channel = 1;
  const int height = 2;
  const int width = 11;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  // fp16
  multiplyCl(input, 0.1);

  // fp32
  multiplyCl(input_fp32, 0.1);

  float mseError = mse<_FP16>(input.getData<_FP16>(),
                              input_fp32.getData<float>(), input.size());

  double cosSim = cosine_similarity<_FP16>(
    input.getData<_FP16>(), input_fp32.getData<float>(), input.size());

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_noTrans_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transB_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_transA_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transAB_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = true;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, addition_i_fp16) {
  const int batch = 12;
  const int channel = 1;
  const int height = 26;
  const int width = 26;

  const int batch_b = 1;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch_b, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor C_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor D_fp16(batch_b, channel, height, width, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(B_fp16, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);
  GEN_TEST_INPUT(C_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(D_fp16, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);

  A_fp16.add_i(B_fp16);
  add_i_cl(C_fp16, D_fp16);

  float mseError =
    mse<_FP16>(A_fp16.getData<_FP16>(), C_fp16.getData<_FP16>(), A_fp16.size());

  double cosSim = cosine_similarity<_FP16>(
    A_fp16.getData<_FP16>(), C_fp16.getData<_FP16>(), A_fp16.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}
#endif

static void run_q_6_K_test(const uint32_t M, const uint32_t K,
                           const uint32_t N) {
  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 10;

  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  std::vector<float> weight = generate_random_vector<float, false>(N * K);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> cpu_q6_dst(M * N, 0.0f);

  const auto data_size = 210 * N * (K / 256);
  std::vector<char> q6_weight = std::vector<char>(data_size);

  // Generate result from SGEMM
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  // Initialize data
  void *gpu_q6_dst = allocateSVM(M * N * sizeof(float));

  void *q6_weight_ptr = allocateSVM(data_size);

  blas_cc->command_queue_inst_.enqueueSVMMap(q6_weight_ptr, data_size, false);

  float *weights_f32_ptr = weight.data();

  float *activations_f32_ptr = (float *)allocateSVM(M * K * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(activations_f32_ptr,
                                             M * K * sizeof(float), false);

  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  /// Quantize weight data
  nntrainer::quantize_q6_K(weights_f32_ptr, q6_weight_ptr, N, K, nullptr);

  // CPU Q6_K GEMV
  Timer timer1{};
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q6_K(M, N, K, activations_f32_ptr, K, q6_weight_ptr, N,
                         cpu_q6_dst.data(), N);
  }
  auto t2 = timer1.GetElapsedMilliseconds();

  // GPU Q6_K GEMV
  Timer timer2{};
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::sgemv_q6_k_cl(q6_weight_ptr, activations_f32_ptr,
                             (float *)gpu_q6_dst, K, N);
  }
  auto t4 = timer2.GetElapsedMilliseconds();

  // Compute raports
  {
    uint32_t first_zero_index = UINT32_MAX;
    int zeros = 0;
    int nans = 0;

    for (uint32_t i = 0; i < M * N; ++i) {
      if (((float *)gpu_q6_dst)[i] == 0) {
        zeros++;
        if (first_zero_index == UINT32_MAX) {
          first_zero_index = i;
        }
      }

      if (std::isnan(((float *)gpu_q6_dst)[i])) {
        nans++;
      }
    }

    std::cout << "Q6_K GEMV : " << M << " x " << K << " x " << N << std::endl;
    std::cout << " - time : CPU = " << t2 / (run_count * 1.0f) << " ms"
              << std::endl;
    std::cout << " - time : GPU = " << t4 / (run_count * 1.0f) << " ms"
              << std::endl;
    std::cout << " - sample : CPU = ";
    debug_print_beg_end(cpu_q6_dst.data(), M * N);
    std::cout << " - sample : GPU = ";
    debug_print_beg_end((float *)gpu_q6_dst, M * N);
    std::cout << " - zeros : " << zeros << " / " << M * N << " [ "
              << zeros * 100.0f / float(M * N) << " %] - first at [ "
              << first_zero_index << " ]" << std::endl;
    std::cout << " - nans : " << nans << " / " << M * N << " [ "
              << nans * 100.0f / float(M * N) << " %]" << std::endl;

    freeSVM(gpu_q6_dst);
    freeSVM(q6_weight_ptr);
    freeSVM(activations_f32_ptr);
  }
}

#define DECLARE_q_6_K_test_M_K_N(M, K, N)                                      \
  TEST(nntrainer_cpu_backend_standalone, q_6_K_test_##M##_##K##_##N) {         \
    run_q_6_K_test(M, K, N);                                                   \
  }

DECLARE_q_6_K_test_M_K_N(1, 3072, 105900);

static void run_q4_0_test(const uint32_t M, const uint32_t K,
                          const uint32_t N) {
  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 200;

  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  std::vector<float> weight = generate_random_vector<float, false>(N * K);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> cpu_q4_dst(M * N, 0.0f);

  struct block_q4_0 {
    uint16_t d[1];
    uint8_t qs[16];
  };

  int64_t block_size = 32;
  int64_t q4_0_tile_size = sizeof(block_q4_0);
  int64_t num_blocks = (K * N) / block_size;
  size_t data_size = q4_0_tile_size * num_blocks;

  // Generate result from SGEMM
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  // Initialize data
  void *gpu_q4_dst = allocateSVM(M * N * sizeof(float));

  void *q4_weight_ptr = allocateSVM(data_size);
  void *q4_weight_repack_ptr = allocateSVM(data_size);

  blas_cc->command_queue_inst_.enqueueSVMMap(q4_weight_ptr, data_size, false);

  float *weights_f32_ptr = weight.data();

  float *activations_f32_ptr = (float *)allocateSVM(M * K * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(activations_f32_ptr,
                                             M * K * sizeof(float), false);

  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  /// Quantize weight data
  nntrainer::quantize_q4_0(weights_f32_ptr, q4_weight_ptr, N, K, nullptr);
  nntrainer::repack_q4_0(q4_weight_repack_ptr, q4_weight_ptr, data_size, N, K);

  // CPU Q4_0 GEMM

  Timer timer1{};
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0(M, N, K, activations_f32_ptr, K, q4_weight_repack_ptr,
                         N, cpu_q4_dst.data(), N);
  }
  auto t2 = timer1.GetElapsedMilliseconds();

  // GPU Q4_0 GEMM
  Timer timer2{};
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_cl(q4_weight_repack_ptr, activations_f32_ptr,
                            (float *)gpu_q4_dst, M, N, K);
  }
  auto t4 = timer2.GetElapsedMilliseconds();

  // Compute raports
  {
    uint32_t first_zero_index = UINT32_MAX;
    int zeros = 0;
    int nans = 0;

    for (uint32_t i = 0; i < M * N; ++i) {
      if (((float *)gpu_q4_dst)[i] == 0) {
        zeros++;
        if (first_zero_index == UINT32_MAX) {
          first_zero_index = i;
        }
      }

      if (std::isnan(((float *)gpu_q4_dst)[i])) {
        nans++;
      }
    }

    const auto data_size_mb = data_size / (1024 * 1024.0f);

    std::cout << "Q4_0 GEMM : " << M << " x " << K << " x " << N << std::endl;
    std::cout << " - time : CPU = " << t2 / (run_count * 1.0f) << " ms"
              << std::endl;
    std::cout << " - time : GPU = " << t4 / (run_count * 1.0f) << " ms"
              << std::endl;
    std::cout << " - sample : REF = ";
    debug_print_beg_end(ref_dst.data(), M * N);
    std::cout << " - sample : CPU = ";
    debug_print_beg_end(cpu_q4_dst.data(), M * N);
    std::cout << " - sample : GPU = ";
    debug_print_beg_end((float *)gpu_q4_dst, M * N);
    std::cout << " - zeros : " << zeros << " / " << M * N << " [ "
              << zeros * 100.0f / float(M * N) << " %] - first at [ "
              << first_zero_index << " ]" << std::endl;
    std::cout << " - nans : " << nans << " / " << M * N << " [ "
              << nans * 100.0f / float(M * N) << " %]" << std::endl;

    freeSVM(gpu_q4_dst);
    freeSVM(q4_weight_ptr);
    freeSVM(q4_weight_repack_ptr);
    freeSVM(activations_f32_ptr);
  }
}

#define DECLARE_q4_0_test_M_K_N(M, K, N)                                       \
  TEST(nntrainer_blas_kernel, q4_0_test_##M##_##K##_##N) {                     \
    run_q4_0_test(M, K, N);                                                    \
  }

DECLARE_q4_0_test_M_K_N(68, 3072, 256);
DECLARE_q4_0_test_M_K_N(68, 3072, 8192);
DECLARE_q4_0_test_M_K_N(68, 8192, 3072);
DECLARE_q4_0_test_M_K_N(68, 3072, 3072);

DECLARE_q4_0_test_M_K_N(28, 3072, 256);
DECLARE_q4_0_test_M_K_N(28, 3072, 8192);
DECLARE_q4_0_test_M_K_N(28, 8192, 3072);
DECLARE_q4_0_test_M_K_N(28, 3072, 3072);

TEST(nntrainer_blas_kernel, q4_0_async_test) {

  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 200;

  const int M = 68;
  const int K = 3072;
  const int N0 = 3072, N1 = 256;

  // Initialize Activation
  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  float *activations_f32_ptr = (float *)allocateSVM(M * K * sizeof(float));
  blas_cc->command_queue_inst_.enqueueSVMMap(activations_f32_ptr,
                                             M * K * sizeof(float), false);
  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  // Initialize Weight
  struct block_q4_0 {
    uint16_t d[1];
    uint8_t qs[16];
  };
  int64_t block_size = 32;
  int64_t q4_0_tile_size = sizeof(block_q4_0);
  size_t data_size_n0 = q4_0_tile_size * (K * N0) / block_size;
  size_t data_size_n1 = q4_0_tile_size * (K * N1) / block_size;

  std::vector<float> weight0 = generate_random_vector<float, true>(N0 * K);
  std::vector<float> weight1 = generate_random_vector<float, true>(N1 * K);
  std::vector<float> weight2 = generate_random_vector<float, true>(N1 * K);

  // weight 0 (3072 x 3072)
  void *w0 = allocateSVM(data_size_n0);
  void *wq0 = allocateSVM(data_size_n0);
  blas_cc->command_queue_inst_.enqueueSVMMap(w0, data_size_n0, false);
  float *weights_f32_ptr = weight0.data();
  nntrainer::quantize_q4_0(weights_f32_ptr, w0, N0, K, nullptr);
  nntrainer::repack_q4_0(wq0, w0, data_size_n0, N0, K);

  // weight 1 (3072 x 256)
  void *w1 = allocateSVM(data_size_n1);
  void *wq1 = allocateSVM(data_size_n1);
  blas_cc->command_queue_inst_.enqueueSVMMap(w1, data_size_n1, false);
  weights_f32_ptr = weight1.data();
  nntrainer::quantize_q4_0(weights_f32_ptr, w1, N1, K, nullptr);
  nntrainer::repack_q4_0(wq1, w1, data_size_n1, N1, K);

  // weight 2 (3072 x 256)
  void *w2 = allocateSVM(data_size_n1);
  void *wq2 = allocateSVM(data_size_n1);
  blas_cc->command_queue_inst_.enqueueSVMMap(w2, data_size_n1, false);
  weights_f32_ptr = weight2.data();
  nntrainer::quantize_q4_0(weights_f32_ptr, w2, N1, K, nullptr);
  nntrainer::repack_q4_0(wq2, w2, data_size_n1, N1, K);

  // Initialize Output data
  float *out0 = (float *)allocateSVM(M * N0 * sizeof(float));
  float *out1 = (float *)allocateSVM(M * N1 * sizeof(float));
  float *out2 = (float *)allocateSVM(M * N1 * sizeof(float));

  // In-order kernel execution
  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_cl(wq0, activations_f32_ptr, (float *)out0, M, N0, K);
    nntrainer::gemm_q4_0_cl(wq1, activations_f32_ptr, (float *)out1, M, N1, K);
    nntrainer::gemm_q4_0_cl(wq2, activations_f32_ptr, (float *)out2, M, N1, K);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  float *async_out0 = (float *)allocateSVM(M * N0 * sizeof(float));
  float *async_out1 = (float *)allocateSVM(M * N1 * sizeof(float));
  float *async_out2 = (float *)allocateSVM(M * N1 * sizeof(float));

  std::vector<void *> weight_vec = {wq0, wq1, wq2};
  std::vector<float *> out_vec = {async_out0, async_out1, async_out2};
  std::vector<unsigned int> n_vec = {N0, N1, N1};

  // Async
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_async_cl(weight_vec, activations_f32_ptr, out_vec, M,
                                  n_vec, K);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::cout << "Q4_0 GEMM : " << M << " x " << K << " x " << N1 << std::endl;
  std::cout << " - time : Orig = " << dt.count() / (run_count * 1.0f) << " ms"
            << std::endl;
  std::cout << " - time : Async = " << gpu_dt.count() / (run_count * 1.0f)
            << " ms" << std::endl;

  // Free allocated SVM
  freeSVM(activations_f32_ptr);
  freeSVM(w0);
  freeSVM(wq0);
  freeSVM(w1);
  freeSVM(wq1);
  freeSVM(w2);
  freeSVM(wq2);
  freeSVM(out0);
  freeSVM(out1);
  freeSVM(out2);
  freeSVM(async_out0);
  freeSVM(async_out1);
  freeSVM(async_out2);
}

TEST(nntrainer_blas_kernel, q4_0_async_test2) {

  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 200;

  const int M = 68;
  const int K = 3072;
  const int N0 = 8192;

  // Initialize Activation
  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  float *activations_f32_ptr = (float *)allocateSVM(M * K * sizeof(float));
  blas_cc->command_queue_inst_.enqueueSVMMap(activations_f32_ptr,
                                             M * K * sizeof(float), false);
  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  // Initialize Weight
  struct block_q4_0 {
    uint16_t d[1];
    uint8_t qs[16];
  };
  int64_t block_size = 32;
  int64_t q4_0_tile_size = sizeof(block_q4_0);
  size_t data_size_n0 = q4_0_tile_size * (K * N0) / block_size;

  std::vector<float> weight0 = generate_random_vector<float, true>(N0 * K);
  std::vector<float> weight1 = generate_random_vector<float, true>(N0 * K);

  // weight 0 (3072 x 8192)
  void *w0 = allocateSVM(data_size_n0);
  void *wq0 = allocateSVM(data_size_n0);
  blas_cc->command_queue_inst_.enqueueSVMMap(w0, data_size_n0, false);
  float *weights_f32_ptr = weight0.data();
  nntrainer::quantize_q4_0(weights_f32_ptr, w0, N0, K, nullptr);
  nntrainer::repack_q4_0(wq0, w0, data_size_n0, N0, K);

  // weight 1 (3072 x 8192)
  void *w1 = allocateSVM(data_size_n0);
  void *wq1 = allocateSVM(data_size_n0);
  blas_cc->command_queue_inst_.enqueueSVMMap(w1, data_size_n0, false);
  weights_f32_ptr = weight1.data();
  nntrainer::quantize_q4_0(weights_f32_ptr, w1, N0, K, nullptr);
  nntrainer::repack_q4_0(wq1, w1, data_size_n0, N0, K);

  // Initialize Output data
  float *out0 = (float *)allocateSVM(M * N0 * sizeof(float));
  float *out1 = (float *)allocateSVM(M * N0 * sizeof(float));

  // CPU
  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_cl(wq0, activations_f32_ptr, (float *)out0, M, N0, K);
    nntrainer::gemm_q4_0_cl(wq1, activations_f32_ptr, (float *)out1, M, N0, K);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  float *async_out0 = (float *)allocateSVM(M * N0 * sizeof(float));
  float *async_out1 = (float *)allocateSVM(M * N0 * sizeof(float));

  std::vector<void *> weight_vec = {wq0, wq1};
  std::vector<float *> out_vec = {async_out0, async_out1};
  std::vector<unsigned int> n_vec = {N0, N0};

  // Async
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_async_cl(weight_vec, activations_f32_ptr, out_vec, M,
                                  n_vec, K);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::cout << "Q4_0 GEMM : " << M << " x " << K << " x " << N0 << std::endl;
  std::cout << " - time : Orig = " << dt.count() / (run_count * 1.0f) << " ms"
            << std::endl;
  std::cout << " - time : Async = " << gpu_dt.count() / (run_count * 1.0f)
            << " ms" << std::endl;

  // Free allocated SVM
  freeSVM(activations_f32_ptr);
  freeSVM(w0);
  freeSVM(wq0);
  freeSVM(w1);
  freeSVM(wq1);
  freeSVM(out0);
  freeSVM(out1);
  freeSVM(async_out0);
  freeSVM(async_out1);
}

/**
 * @note Disable due to an error.
 *
 * ld.lld: error: undefined symbol:
nntrainer::SwiGLULayerCl::swiglu_cl_fp16(_Float16*, _Float16*, _Float16*,
unsigned int, unsigned int, bool)

* ld.lld: error: undefined symbol: nntrainer::swiglu(unsigned int, _Float16*,
_Float16*, _Float16*)
 *
 */
// #ifdef ENABLE_FP16
// TEST(blas_kernels, swiglu_layer_fp16) {
//   const int batch = 1;
//   const int channel = 1;
//   const int height = 64;
//   const int width = 3072;

//   const int batch_b = 1;

//   const float alpha = 1e-1;
//   const int MOD = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B_fp16(batch_b, channel, height, width,
//   t_type_nchw_fp16); nntrainer::Tensor out_cl_fp16(batch_b, channel, height,
//   width,
//                                 t_type_nchw_fp16);
//   nntrainer::Tensor out_ref_fp16(batch, channel, height, width,
//                                  t_type_nchw_fp16);

//   GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_C(B_fp16, ((i * (batch_b * height * channel) +
//                              j * (batch_b * height) + k * (width) + l + 1) %
//                             MOD) *
//                              alpha);

//   static constexpr uint32_t run_count = 500;

//   SwiGLULayerCl layer;

//   Timer timer1{};
//   for (unsigned int i = 0; i < run_count; ++i) {
//     layer.swiglu_cl_fp16(A_fp16.getData<_FP16>(), B_fp16.getData<_FP16>(),
//                          out_cl_fp16.getData<_FP16>(), width, height);
//   }
//   auto t2_cl = timer1.GetElapsedMilliseconds();

//   Timer timer2{};
//   for (unsigned int i = 0; i < run_count; ++i) {
//     nntrainer::swiglu(height * width, out_ref_fp16.getData<_FP16>(),
//                       A_fp16.getData<_FP16>(), B_fp16.getData<_FP16>());
//   }
//   auto t2_ref = timer2.GetElapsedMilliseconds();

//   std::cout << "Swiglu time : GPU = " << t2_cl / (run_count * 1.0f) << " ms"
//             << std::endl;

//   std::cout << "Swiglu time : CPU = " << t2_ref / (run_count * 1.0f) << " ms"
//             << std::endl;

//   float mseError = mse<_FP16>(out_cl_fp16.getData<_FP16>(),
//                               out_ref_fp16.getData<_FP16>(), height * width);

//   double cosSim =
//     cosine_similarity<_FP16>(out_cl_fp16.getData<_FP16>(),
//                              out_ref_fp16.getData<_FP16>(), height * width);

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseError, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
// }
// #endif

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
