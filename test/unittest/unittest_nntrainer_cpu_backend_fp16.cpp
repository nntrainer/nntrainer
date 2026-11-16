// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_cpu_backend_fp16.cpp
 * @date	03 April 2025
 * @brief	This is unittest for cpu_backend standalone
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nntrainer_test_util.h"
#include <cfloat>
#include <cpu_backend.h>
#include <fallback_internal.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

#include <chrono>
#include <iostream>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  double err_sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(static_cast<float>(
                                      src[i * N + j] - src2[i * N + j])));
      err_sum += std::abs(static_cast<float>(src[i * N + j] - src2[i * N + j]));
    }
  }
  // std::cout << "err_sum : " << err_sum << std::endl;
  return max_diff;
}

#define QK4_0 32
/**
 * @brief q4_0 block
 *
 */
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0_testonly;

/**
 * @brief q8_K block
 *
 */
typedef struct {
  float d;                 // delta
  int8_t qs[256];          // quants
  int16_t bsums[256 / 16]; // sum of quants in groups of 16
} block_q8_K_testonly;

#define QK_K 256
typedef struct {
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  uint16_t d;               // super-block scale
} block_q6_K_testonly;

template <typename T = float>
float compute_mse(const uint32_t M, const uint32_t N, std::vector<T> &ref_dst,
                  std::vector<T> &dst, bool print = false) {
  auto mean_squared_error = mse<T, T>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity<T, T>(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff<T>(ref_dst.data(), dst.data(), M, N);

  auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
  auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);
  if (print) {
    std::cout << "[INFO]            MSE: " << mean_squared_error
              << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
              << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  }
  return mean_squared_error;
}

float test_gemm_q4_0_fp16(const uint32_t M, const uint32_t K, const uint32_t N,
                          const float *weights, const _FP16 *activations,
                          std::vector<_FP16> &ref_dst, bool print = false) {
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t q4_0_block_size = 32;
  int64_t q4_0_num_blocks = (K * N) / q4_0_block_size;
  size_t q4_0_data_size = q4_0_type_size * N / q4_0_block_size;
  q4_0_data_size *= K;
  std::vector<char> q4_0_offline_qWeight = std::vector<char>(q4_0_data_size);

  char *q4_0_offline_qWeight_ptr = (char *)q4_0_offline_qWeight.data();
  nntrainer::quantize_q4_0(weights, (void *)q4_0_offline_qWeight_ptr, N, K,
                           nullptr);

  std::vector<char> q4_0_repacked_qWeight = std::vector<char>(q4_0_data_size);
  nntrainer::repack_q4_0(q4_0_repacked_qWeight.data(), q4_0_offline_qWeight_ptr,
                         q4_0_data_size, N, K);
  std::vector<_FP16> dst(M * N);
  auto t1 = high_resolution_clock::now();
  nntrainer::gemm_q4_0<_FP16>(M, N, K, activations, K,
                              (void *)q4_0_repacked_qWeight.data(), N,
                              dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_0: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  auto mean_squared_error = compute_mse<_FP16>(M, N, ref_dst, dst, print);

  return mean_squared_error;
}

float test_gemm_q6_K_fp16(const uint32_t M, const uint32_t K, const uint32_t N,
                          const float *weights, const _FP16 *activations,
                          std::vector<_FP16> &ref_dst, bool print = false) {
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * N / q6_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q6_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  std::vector<_FP16> dst(M * N);
  auto t1 = high_resolution_clock::now();
  nntrainer::gemm_q6_K<_FP16>(M, N, K, activations, K,
                              (void *)offline_qWeight_ptr, N, dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q6_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  auto mean_squared_error = compute_mse<_FP16>(M, N, ref_dst, dst, print);

  return mean_squared_error;
}

void run_quant_test_fp16(const uint32_t M, const uint32_t K, const uint32_t N,
                         float &q4_0_mse, float &q6_K_mse, bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "[INFO] Quantization Test (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
  }
  ///@note A(M, K) * W.T(N, K) = (M, N)
  ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

  ///@note q4_K GEMM is a Row-Major, transB GEMM
  std::vector<_FP16> activation = generate_random_vector<_FP16>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<_FP16> weight_fp16(N * K);
  nntrainer::scopy(N * K, weight.data(), 1, weight_fp16.data(), 1);
  std::vector<_FP16> ref_dst(M * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < 20; ++tc) {
    nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                     weight_fp16.data(), K, 0.F, ref_dst.data(), N);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm :    " << dt.count() / 20 << " ns "
              << dt.count() / 20 / 1'000 << " us "
              << dt.count() / 20 / 1'000'000 << " ms " << std::endl;
  }
  q4_0_mse = test_gemm_q4_0_fp16(M, K, N, weight.data(), activation.data(),
                                 ref_dst, print);
  q6_K_mse = test_gemm_q6_K_fp16(M, K, N, weight.data(), activation.data(),
                                 ref_dst, print);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_256x1024x512) {
  const unsigned int M = 256;
  const unsigned int K = 1024;
  const unsigned int N = 512;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_457x3072x3072) {
  const unsigned int M = 457;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_458x3072x3072) {
  const unsigned int M = 458;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_459x3072x3072) {
  const unsigned int M = 459;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x3072x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x768x1024) {
  const unsigned int M = 1;
  const unsigned int K = 768;
  const unsigned int N = 1024;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x3072) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

static void run_trigonometric_values_test(const unsigned int N,
                                          bool print = false) {
  const int TEST_CNT = 20;
  nanoseconds ref_mul_time = (nanoseconds)0;
  nanoseconds mul_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<_FP16> X = generate_random_vector<_FP16, false>(N);
    std::vector<float> X_ref = generate_random_vector<float, false>(N);
    std::vector<_FP16> Y = generate_random_vector<_FP16, false>(N);
    std::vector<float> Y_ref = generate_random_vector<float, false>(N);

    std::vector<_FP16> X2 = generate_random_vector<_FP16, false>(N);
    std::vector<float> X2_ref = generate_random_vector<float, false>(N);
    std::vector<_FP16> Y2 = generate_random_vector<_FP16, false>(N);
    std::vector<float> Y2_ref = generate_random_vector<float, false>(N);
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::sine(N, X_ref.data(), Y_ref.data());
      nntrainer::cosine(N, X2_ref.data(), Y2_ref.data());
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_mul_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::sine(N, X.data(), Y.data());
      nntrainer::cosine(N, X2.data(), Y2.data());
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        mul_time += dt;
      }
    }

    auto mean_squared_error = mse<float, _FP16>(Y_ref.data(), Y.data(), N);
    auto cos_sim = cosine_similarity<float, _FP16>(Y2_ref.data(), Y2.data(), N);

    ASSERT_LE(mean_squared_error, 1e-3);
    ASSERT_GE(cos_sim, 0.99);
  }

  if (print) {
    std::cout << "[INFO] trigonometric_values: TEST CNT: " << TEST_CNT
              << ", N: " << N
              << ", Average ref_time: " << ref_mul_time.count() / TEST_CNT
              << " ns, Average test_time: " << mul_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x512_CMP) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 512;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_768x768x768_CMP) {
  const unsigned int M = 768;
  const unsigned int K = 768;
  const unsigned int N = 768;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_512x768x2048_CMP) {
  const unsigned int M = 512;
  const unsigned int K = 768;
  const unsigned int N = 2048;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_3072x512x512_CMP) {
  const unsigned int M = 3072;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, trigonometric_values_test) {

  const unsigned int N = 3072;
  run_trigonometric_values_test(N);
}

int main(int argc, char **argv) {
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
