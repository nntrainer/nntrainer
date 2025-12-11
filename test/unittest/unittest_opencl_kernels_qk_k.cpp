// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	unittest_opencl_kernels_qk_k.cpp
 * @date	6 June 2024
 * @brief	Test setup for Q4_0 and Q6_K OpenCL kernels
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

using namespace nntrainer;

static void run_q_6_K_test(const uint32_t M, const uint32_t K,
                           const uint32_t N) {
  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

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

  // Warmup & Calibration
  unsigned int run_count = 5;
  auto t_w1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q6_K(M, N, K, activations_f32_ptr, K, q6_weight_ptr, N,
                         cpu_q6_dst.data(), N);
  }
  auto t_w2 = std::chrono::high_resolution_clock::now();
  double avg_time =
    std::chrono::duration<double>(t_w2 - t_w1).count() / run_count;

  if (avg_time > 0) {
    run_count = std::max(1u, (unsigned int)(0.5 / avg_time));
  } else {
    run_count = 100;
  }

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

  std::cout << "Q6_K GEMV: " << M << " x " << K << " x " << N << std::endl;
  std::cout << " - CPU time: " << t2 / (run_count * 1.0f) << " ms" << std::endl;
  std::cout << " - GPU time: " << t4 / (run_count * 1.0f) << " ms" << std::endl;

  float mse_err = mse<float>(cpu_q6_dst.data(), (float *)gpu_q6_dst, M * N);
  std::cout << " - MSE: " << mse_err << std::endl;

  // Q6_K quantization has better precision than Q4_0
  EXPECT_IN_RANGE(mse_err, 0.0f, 0.05f);

  freeSVM(gpu_q6_dst);
  freeSVM(q6_weight_ptr);
  freeSVM(activations_f32_ptr);
}

#define DECLARE_q_6_K_test_M_K_N(M, K, N)                                      \
  TEST(nntrainer_opencl_kernels_qk_k, q_6_K_test_##M##_##K##_##N) {            \
    run_q_6_K_test(M, K, N);                                                   \
  }

DECLARE_q_6_K_test_M_K_N(1, 3072, 105900);

static void run_q4_0_test(const uint32_t M, const uint32_t K,
                          const uint32_t N) {
  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.01f, 0.01f);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> cpu_q4_dst(M * N, 0.0f);

  const auto data_size = K * N / Q4_0 * sizeof(block_q4_0);

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

  // Warmup & Calibration
  unsigned int run_count = 5;
  auto t_w1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0(M, N, K, activations_f32_ptr, K, q4_weight_repack_ptr,
                         N, cpu_q4_dst.data(), N);
  }
  auto t_w2 = std::chrono::high_resolution_clock::now();
  double avg_time =
    std::chrono::duration<double>(t_w2 - t_w1).count() / run_count;

  if (avg_time > 0) {
    run_count = std::max(1u, (unsigned int)(0.5 / avg_time));
  } else {
    run_count = 100;
  }

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

  std::cout << "Q4_0 GEMM: " << M << " x " << K << " x " << N << std::endl;
  std::cout << " - CPU time: " << t2 / (run_count * 1.0f) << " ms" << std::endl;
  std::cout << " - GPU time: " << t4 / (run_count * 1.0f) << " ms" << std::endl;

  float mse_err = mse<float>(cpu_q4_dst.data(), (float *)gpu_q4_dst, M * N);

  // Q4_0 quantization is lossy, expect some error
  EXPECT_IN_RANGE(mse_err, 0.0f, 0.1f);

  freeSVM(gpu_q4_dst);
  freeSVM(q4_weight_ptr);
  freeSVM(q4_weight_repack_ptr);
  freeSVM(activations_f32_ptr);
}

#define DECLARE_q4_0_test_M_K_N(M, K, N)                                       \
  TEST(nntrainer_opencl_kernels_qk_k, q4_0_test_##M##_##K##_##N) {             \
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

TEST(nntrainer_opencl_kernels_qk_k, q4_0_async_test) {

  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

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

  // Warmup & Calibration
  unsigned int run_count = 5;
  auto t_w1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_cl(wq0, activations_f32_ptr, (float *)out0, M, N0, K);
    nntrainer::gemm_q4_0_cl(wq1, activations_f32_ptr, (float *)out1, M, N1, K);
    nntrainer::gemm_q4_0_cl(wq2, activations_f32_ptr, (float *)out2, M, N1, K);
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

TEST(nntrainer_opencl_kernels_qk_k, q4_0_async_test2) {

  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

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

  // Warmup & Calibration
  unsigned int run_count = 5;
  auto t_w1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_cl(wq0, activations_f32_ptr, (float *)out0, M, N0, K);
    nntrainer::gemm_q4_0_cl(wq1, activations_f32_ptr, (float *)out1, M, N0, K);
  }
  auto t_w2 = std::chrono::high_resolution_clock::now();
  double avg_time =
    std::chrono::duration<double>(t_w2 - t_w1).count() / run_count;

  if (avg_time > 0) {
    run_count = std::max(1u, (unsigned int)(0.5 / avg_time));
  } else {
    run_count = 100;
  }

  // Sync
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

  // Verify sync and async produce identical results
  for (unsigned int i = 0; i < M * N0; ++i) {
    EXPECT_FLOAT_EQ(out0[i], async_out0[i]);
    EXPECT_FLOAT_EQ(out1[i], async_out1[i]);
  }

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
