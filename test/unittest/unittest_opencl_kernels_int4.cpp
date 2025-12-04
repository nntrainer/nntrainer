// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file	unittest_opencl_kernels_int4.cpp
 * @date	3 December 2025
 * @brief	Test setup for int4 OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cstring>
#include <gtest/gtest.h>
#include <utility>

#include "fallback_internal.h"
#include "int4_utils.h"
#include "nntrainer_test_util.h"
#include "q4_0_utils.h"
#include "tensor_dim.h"
#include "timer.h"
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <cpu_backend.h>
#include <fp16.h>
#include <tensor.h>

using namespace nntrainer;
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

  EXPECT_IN_RANGE(mse_dequantized_int4, 0, epsilon);
  EXPECT_IN_RANGE(mse_dequantized_int4_row, 0, epsilon);

  // This must be equal
  EXPECT_FLOAT_EQ(mse_dequantized_int4, mse_dequantized_int4_row);
}

#define DECLARE_dequantization_test_K_N(K, N)                                  \
  TEST(nntrainer_opencl_kernels_int4, dequantization_test_##K##_##N) {         \
    run_dequantization_test_(K, N);                                            \
  }

DECLARE_dequantization_test_K_N(8192, 3072);
DECLARE_dequantization_test_K_N(3072, 8192);
DECLARE_dequantization_test_K_N(8188, 3068);
DECLARE_dequantization_test_K_N(3068, 8188);
DECLARE_dequantization_test_K_N(144, 168);

static void run_int4_gemv_test_(const uint32_t K, const uint32_t N,
                                int scale_group_size) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const int INT4_BLOCK_N_SIZE = 32;
  uint32_t alignN = align(N, INT4_BLOCK_N_SIZE);
  uint32_t alignK = align(K, scale_group_size);

  // Allocate & initialize group-wise int4 data
  char *weight_ptr = (char *)allocateSVM(alignK * alignN / 2);
  uint16_t *scale_ptr = (uint16_t *)allocateSVM(ceilDiv(K, scale_group_size) *
                                                alignN * sizeof(uint16_t));
  uint16_t *input_ptr = (uint16_t *)allocateSVM(K * sizeof(uint16_t));
  uint16_t *output_ptr = (uint16_t *)allocateSVM(alignN * sizeof(uint16_t));

  std::vector<float> weight_fp32 =
    generate_random_vector<float>(N * K, -1.0f, 1.0f);
  std::vector<float> input_fp32 = generate_random_vector<float>(K, -1.0f, 1.0f);

  // Reference SGEMV
  std::vector<float> reference_output_fp32(N);
  nntrainer::sgemv(0, false, N, K, 1.0f, weight_fp32.data(), K,
                   input_fp32.data(), 1, 0.0f, reference_output_fp32.data(), 1);

  // Reference Q4_0 GEMM
  std::vector<float> ref_dst(N, 0.0f);
  float mse_q4 = 0.0f;
  if (K % Q4_0 == 0 && N % 8 == 0) {
    size_t q4_data_size = K * N / Q4_0 * sizeof(block_q4_0);
    std::vector<float> q4_output_fp32(N);
    std::vector<uint8_t> q4_weight(q4_data_size);
    std::vector<uint8_t> q4_weight_repack(q4_data_size);
    nntrainer::quantize_q4_0(weight_fp32.data(), q4_weight.data(), N, K,
                             nullptr);
    nntrainer::repack_q4_0(q4_weight_repack.data(), q4_weight.data(),
                           q4_data_size, N, K);
    nntrainer::gemm_q4_0(1, N, K, input_fp32.data(), K, q4_weight_repack.data(),
                         N, q4_output_fp32.data(), N);
    mse_q4 = mse<float>(ref_dst.data(), q4_output_fp32.data(), N);
  }

  // GPU INT4 GEMV - MAIN TEST
  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> quantized_scales;
  Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, scale_group_size,
                               quantized_weights, quantized_scales);

  for (unsigned int i = 0; i < K; ++i) {
    input_ptr[i] = compute_fp32_to_fp16(input_fp32[i]);
  }

  for (unsigned int i = 0; i < ceilDiv(K, scale_group_size) * alignN; ++i) {
    scale_ptr[i] = quantized_scales[i];
  }

  for (unsigned int i = 0; i < alignN * alignK / 2; ++i) {
    weight_ptr[i] = quantized_weights[i];
  }

  // Warmup & Calibration
  unsigned int run_count = 5;
  auto t_w1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemv_int4_cl(weight_ptr, scale_ptr, input_ptr, output_ptr, K, N,
                            scale_group_size);
  }
  auto t_w2 = std::chrono::high_resolution_clock::now();
  double avg_time =
    std::chrono::duration<double>(t_w2 - t_w1).count() / run_count;

  if (avg_time > 0) {
    run_count = std::max(1u, (unsigned int)(0.5 / avg_time));
  } else {
    run_count = 100;
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

  float mse_int4_err =
    mse<float>(reference_output_fp32.data(), output_fp32.data(), N);

  // INT4 quantization is lossy, expect some error but should be reasonable
  if (mse_q4 > 0.0f) {
    // If we have a Q4 reference, the error should be comparable
    EXPECT_IN_RANGE(mse_int4_err, 0.0f, mse_q4 * 2.0f);
  } else {
    EXPECT_IN_RANGE(mse_int4_err, 0.0f, 1e-5 * K * N);
  }

  freeSVM(weight_ptr);
  freeSVM(scale_ptr);
  freeSVM(input_ptr);
  freeSVM(output_ptr);
}

#define DECLARE_int4_gemv_test_K_N(K, N, G)                                    \
  TEST(nntrainer_opencl_kernels_int4, int4_gemv_test_##K##_##N##_Group##G) {   \
    run_int4_gemv_test_(K, N, G);                                              \
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

TEST(nntrainer_opencl_kernels_int4, int4_gemv_async_test) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

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

  // Warmup & Calibration
  unsigned int run_count = 5;
  auto t_w1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemv_int4_cl((char *)wq0, (uint16_t *)ws0, input, out0, K, N0,
                            scale_group_size);
    nntrainer::gemv_int4_cl((char *)wq1, (uint16_t *)ws1, input, out1, K, N1,
                            scale_group_size);
    nntrainer::gemv_int4_cl((char *)wq2, (uint16_t *)ws2, input, out2, K, N1,
                            scale_group_size);
  }
  auto t_w2 = std::chrono::high_resolution_clock::now();
  double avg_time =
    std::chrono::duration<double>(t_w2 - t_w1).count() / run_count;

  if (avg_time > 0) {
    run_count = std::max(1u, (unsigned int)(0.5 / avg_time));
  } else {
    run_count = 100;
  }

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
  // Warmup & Calibration (Async)
  unsigned int run_count_async = 5;
  auto t_w3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count_async; ++i) {
    nntrainer::gemv_int4_async_cl(weight_vec, scale_vec, input, out_vec, K,
                                  n_vec, scale_group_size);
  }
  auto t_w4 = std::chrono::high_resolution_clock::now();
  double avg_time_async =
    std::chrono::duration<double>(t_w4 - t_w3).count() / run_count_async;

  if (avg_time_async > 0) {
    run_count = std::max(1u, (unsigned int)(0.5 / avg_time_async));
  } else {
    run_count = 100;
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemv_int4_async_cl(weight_vec, scale_vec, input, out_vec, K,
                                  n_vec, scale_group_size);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::cout << "INT4 GEMV Async: " << M << " x " << K << " x " << N1
            << std::endl;
  std::cout << " - Sync  time: " << dt.count() / (run_count * 1.0f) << " ms"
            << std::endl;
  std::cout << " - Async time: " << gpu_dt.count() / (run_count * 1.0f) << " ms"
            << std::endl;

  // Verify sync and async produce identical results
  for (unsigned int i = 0; i < M * N0; ++i) {
    EXPECT_FLOAT_EQ(compute_fp16_to_fp32(out0[i]),
                    compute_fp16_to_fp32(async_out0[i]));
  }
  for (unsigned int i = 0; i < M * N1; ++i) {
    EXPECT_FLOAT_EQ(compute_fp16_to_fp32(out1[i]),
                    compute_fp16_to_fp32(async_out1[i]));
    EXPECT_FLOAT_EQ(compute_fp16_to_fp32(out2[i]),
                    compute_fp16_to_fp32(async_out2[i]));
  }

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
                                const uint32_t N, const int scale_group_size) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const int INT4_BLOCK_N_SIZE = 32;
  uint32_t alignN = align(N, INT4_BLOCK_N_SIZE);
  uint32_t alignK = align(K, scale_group_size);

  uint32_t input_size = M * alignK;

  std::vector<float> input =
    generate_random_vector<float, false>(input_size, -1.0, 1.0);
  std::vector<float> weight_fp32 =
    generate_random_vector<float, false>(N * K, -1.0, 1.0);

  // Reference SGEMM
  std::vector<float> ref_dst(M * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, input.data(), K,
                   weight_fp32.data(), K, 0.F, ref_dst.data(), N);

  // Reference Q4_0 GEMM
  float mse_q4 = 0.0f;
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
    mse_q4 = mse<float>(ref_dst.data(), q4_output_fp32.data(), M * N);
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
  // Warmup & Calibration
  unsigned int run_count = 5;
  auto t_w1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::openvino_gemm_cl(input_ptr, weight_ptr, scale_ptr, output_ptr, M,
                                N, K, scale_group_size);
  }
  auto t_w2 = std::chrono::high_resolution_clock::now();
  double avg_time =
    std::chrono::duration<double>(t_w2 - t_w1).count() / run_count;

  if (avg_time > 0) {
    run_count = std::max(1u, (unsigned int)(0.5 / avg_time));
  } else {
    run_count = 100;
  }

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

  std::cout << "INT4 GEMM : M:" << M << " x K:" << K << " x N:" << N
            << std::endl;
  std::cout << " - time : GPU = " << gpu_dt / (run_count * 1.0f) << " ms"
            << std::endl;

  float mse_int4_err = mse<float>(ref_dst.data(), output_fp32.data(), M * N);

  // INT4 quantization is lossy, expect some error but should be reasonable
  if (mse_q4 > 0.0f) {
    // If we have a Q4 reference, the error should be comparable
    EXPECT_IN_RANGE(mse_int4_err, 0.0f, mse_q4 * 2.0f);
  } else {
    EXPECT_IN_RANGE(mse_int4_err, 0.0f, 1e-5 * M * K * N);
  }

  freeSVM(weight_ptr);
  freeSVM(scale_ptr);
  freeSVM(input_ptr);
  freeSVM(output_ptr);
}

#define DECLARE_int4_gemm_test_M_K_N(M, K, N, G)                               \
  TEST(nntrainer_opencl_kernels_int4,                                          \
       int4_gemm_test_##M##_##K##_##N##_Group##G) {                            \
    run_int4_gemm_test_(M, K, N, G);                                           \
  }

DECLARE_int4_gemm_test_M_K_N(28, 3072, 256, 32);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 8192, 32);
DECLARE_int4_gemm_test_M_K_N(28, 8192, 3072, 32);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 3072, 32);

DECLARE_int4_gemm_test_M_K_N(28, 3072, 256, 128);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 8192, 128);
DECLARE_int4_gemm_test_M_K_N(28, 8192, 3072, 128);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 3072, 128);

DECLARE_int4_gemm_test_M_K_N(4, 3060, 3072, 32);
DECLARE_int4_gemm_test_M_K_N(4, 3072, 3072, 32);

TEST(nntrainer_opencl_kernels_int4, int4_gemm_async_test) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

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

  // Warmup & Calibration
  unsigned int run_count = 5;
  auto t_w1 = std::chrono::high_resolution_clock::now();
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
  auto t_w2 = std::chrono::high_resolution_clock::now();
  double avg_time =
    std::chrono::duration<double>(t_w2 - t_w1).count() / run_count;

  if (avg_time > 0) {
    run_count = std::max(1u, (unsigned int)(0.5 / avg_time));
  } else {
    run_count = 100;
  }

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

  std::cout << "INT4 GEMM Async: " << M << " x " << K << " x " << N1
            << std::endl;
  std::cout << " - Sync  time: " << dt.count() / (run_count * 1.0f) << " ms"
            << std::endl;
  std::cout << " - Async time: " << gpu_dt.count() / (run_count * 1.0f) << " ms"
            << std::endl;

  // Verify sync and async produce identical results
  for (unsigned int i = 0; i < M * N0; ++i) {
    EXPECT_FLOAT_EQ(out0[i], async_out0[i]);
  }
  for (unsigned int i = 0; i < M * N1; ++i) {
    EXPECT_FLOAT_EQ(out1[i], async_out1[i]);
    EXPECT_FLOAT_EQ(out2[i], async_out2[i]);
  }

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
