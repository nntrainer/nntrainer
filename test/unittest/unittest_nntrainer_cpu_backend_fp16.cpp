// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_cpu_backend_fp16.cpp
 * @date	03 April 2025
 * @brief	This is unittest for cpu_backend standalone
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "int4_utils.h"
#include "kleidiai_interface.h"
#include "nntrainer_test_util.h"
#include <cfloat>
#include <cpu_backend.h>
#include <fallback_internal.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <tuple>
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

std::tuple<float, uint32_t> test_gemm_qai8dxp_qsi4cxp_unpacked(
  const uint32_t M, const uint32_t K, const uint32_t N, const float *weights,
  const float *activations, std::vector<float> &ref_dst, bool transB = true,
  bool print = false) {
  // Step1. Set qai8dxp_qsi4cxp quant test components
  const size_t lhs_ref_size_qa8dx =
    static_cast<size_t>(M) * (K + sizeof(int32_t) + sizeof(float));
  const size_t rhs_native_size_qs4cx =
    transB
      ? static_cast<size_t>(N) * (((K + 2 - 1) / 2) * 2 / 2) * sizeof(uint8_t)
      : static_cast<size_t>(K) * (((N + 2 - 1) / 2) * 2 / 2) * sizeof(uint8_t);
  const size_t rhs_scales_size_f32 =
    transB ? N * sizeof(float) : K * sizeof(float);

  uint8_t *rhs_native_mtx_qs4cx = new uint8_t[rhs_native_size_qs4cx];
  uint8_t *rhs_scales_f32 = new uint8_t[rhs_scales_size_f32];

  // Step2. 4-bit Weight quantization, for qs4cx format, with fp32 scale
  nntrainer::nntr_quant_qs4cx_f32(N, K, (void *)weights,
                                  (void *)rhs_native_mtx_qs4cx, rhs_scales_f32,
                                  transB);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(static_cast<size_t>(M) * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  uint32_t opt_kernel_variant_idx =
    nntrainer::nntr_gemm_qai8dxp_qsi4cxp_unpacked(
      M, N, K, (void *)activations, (void *)rhs_native_mtx_qs4cx,
      (void *)rhs_scales_f32, dst.data(), transB);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] test_gemm_qai8dxp_qsi4cxp_unpacked: " << dt.count()
              << " ns " << dt.count() / 1'000 << " us "
              << dt.count() / 1'000'000 << " ms " << std::endl;
  }

  // Step4. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);

  delete[] rhs_native_mtx_qs4cx;
  delete[] rhs_scales_f32;

  return {mean_squared_error, opt_kernel_variant_idx};
}

static uint32_t
run_qai8dxp_qsi4cxp_test_unpacked(const uint32_t M, const uint32_t K,
                                  const uint32_t N, float &qai8dxp_qsi4cxp_mse,
                                  bool transB = true, bool print = false) {
  if (print) {
    std::cout << "[INFO] qai8dxp_qsi4cxp Test (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
  }
  ///@note A(M, K) * W.T(N, K) = (M, N)
  ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

  ///@note q4_K GEMM is a Row-Major, transB GEMM
  std::vector<float> activation =
    generate_random_vector<float>(static_cast<std::size_t>(M) * K);
  std::vector<float> weight =
    generate_random_vector<float>(static_cast<std::size_t>(N) * K);
  std::vector<float> ref_dst(static_cast<std::size_t>(M) * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm :    " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }
  const auto [mse, opt_kernel_variant_idx] = test_gemm_qai8dxp_qsi4cxp_unpacked(
    M, K, N, weight.data(), activation.data(), ref_dst, transB, print);

  qai8dxp_qsi4cxp_mse = mse;

  return opt_kernel_variant_idx;
}

float test_gemm_qai8dxp_qsi4cxp_packed(const uint32_t M, const uint32_t K,
                                       const uint32_t N, const float *weights,
                                       const float *activations,
                                       std::vector<float> &ref_dst,
                                       uint32_t opt_kernel_idx,
                                       bool transB = true, bool print = false) {
  // Step1. Set qai8dxp_qsi4cxp quant test components
  const size_t lhs_ref_size_qa8dx =
    static_cast<size_t>(M) * (K + sizeof(int32_t) + sizeof(float));
  const size_t rhs_native_size_qs4cx =
    transB
      ? static_cast<size_t>(N) * (((K + 2 - 1) / 2) * 2 / 2) * sizeof(uint8_t)
      : static_cast<size_t>(K) * (((N + 2 - 1) / 2) * 2 / 2) * sizeof(uint8_t);
  const size_t rhs_scales_size_f32 =
    transB ? N * sizeof(float) : K * sizeof(float);

  uint8_t *rhs_native_mtx_qs4cx = new uint8_t[rhs_native_size_qs4cx];
  uint8_t *rhs_scales_f32 = new uint8_t[rhs_scales_size_f32];

  // Step2. 4-bit Weight quantization, for qs4cx format, with fp32 scale
  nntrainer::nntr_quant_qs4cx_f32(N, K, (void *)weights,
                                  (void *)rhs_native_mtx_qs4cx, rhs_scales_f32,
                                  transB);
  // Step3. Offline weight packing
  size_t packed_weight_size =
    nntrainer::nntr_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(N, K, opt_kernel_idx,
                                                          transB);
  uint8_t *packed_weight = new uint8_t[packed_weight_size];

  nntrainer::nntr_qsi4cxp_qs4cxs1s0_rhs_pack(
    N, K, packed_weight, rhs_native_mtx_qs4cx, rhs_scales_f32, opt_kernel_idx,
    transB);

  // Step4. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(static_cast<size_t>(M) * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::nntr_gemm_qai8dxp_qsi4cxp_packed(M, N, K, (void *)activations,
                                              (void *)packed_weight, dst.data(),
                                              opt_kernel_idx, transB);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] test_gemm_qai8dxp_qsi4cxp_packed: " << dt.count()
              << " ns " << dt.count() / 1'000 << " us "
              << dt.count() / 1'000'000 << " ms " << std::endl;
  }

  // Step5. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);

  delete[] rhs_native_mtx_qs4cx;
  delete[] rhs_scales_f32;
  delete[] packed_weight;

  return mean_squared_error;
}

void run_qai8dxp_qsi4cxp_test_packed(const uint32_t M, const uint32_t K,
                                     const uint32_t N,
                                     float &qai8dxp_qsi4cxp_mse,
                                     uint32_t opt_kernel_idx,
                                     bool transB = true, bool print = false) {
  if (print) {
    std::cout << "[INFO] run_qai8dxp_qsi4cxp_test_packed Test (M:" << M
              << ", K:" << K << ", N:" << N
              << ") with opt_kernel_idx : " << opt_kernel_idx << std::endl;
  }
  ///@note A(M, K) * W.T(N, K) = (M, N)
  ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

  ///@note q4_K GEMM is a Row-Major, transB GEMM
  std::vector<float> activation =
    generate_random_vector<float>(static_cast<std::size_t>(M) * K);
  std::vector<float> weight =
    generate_random_vector<float>(static_cast<std::size_t>(N) * K);
  std::vector<float> ref_dst(static_cast<std::size_t>(M) * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm :    " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }
  qai8dxp_qsi4cxp_mse =
    test_gemm_qai8dxp_qsi4cxp_packed(M, K, N, weight.data(), activation.data(),
                                     ref_dst, opt_kernel_idx, transB, print);
}

template <typename T = uint32_t>
std::pair<T, size_t> most_frequent(const std::vector<T> &data) {
  // Range is fixed 0–7, so use a small fixed array for counting
  std::array<size_t, 8> counts{};
  counts.fill(0);

  for (T v : data) {
    counts[v]++;
  }

  T most_value = 0;
  size_t most_count = 0;
  for (uint32_t i = 0; i < counts.size(); ++i) {
    if (counts[i] > most_count) {
      most_count = counts[i];
      most_value = i;
    }
  }

  return {most_value, most_count};
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

TEST(nntrainer_cpu_backend_standalone, qai8dxp_qsi4cxp_1x3072x512_CMP) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 512;
  float qai8dxp_qsi4cxp_mse;
  float qai8dxp_qsi4cxp_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qai8dxp_qsi4cxp_test_unpacked(
      M, K, N, qai8dxp_qsi4cxp_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qai8dxp_qsi4cxp_test_packed(M, K, N, qai8dxp_qsi4cxp_mse_packed,
                                  opt_idx_variant, true, false);
  ASSERT_LE(qai8dxp_qsi4cxp_mse, eps * M * K * N);
  ASSERT_LE(qai8dxp_qsi4cxp_mse_packed, eps * M * K * N);
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

TEST(nntrainer_cpu_backend_standalone, qai8dxp_qsi4cxp_768x768x768_CMP) {
  const unsigned int M = 768;
  const unsigned int K = 768;
  const unsigned int N = 768;
  float qai8dxp_qsi4cxp_mse;
  float qai8dxp_qsi4cxp_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qai8dxp_qsi4cxp_test_unpacked(
      M, K, N, qai8dxp_qsi4cxp_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qai8dxp_qsi4cxp_test_packed(M, K, N, qai8dxp_qsi4cxp_mse_packed,
                                  opt_idx_variant, true, false);
  ASSERT_LE(qai8dxp_qsi4cxp_mse, eps * M * K * N);
  ASSERT_LE(qai8dxp_qsi4cxp_mse_packed, eps * M * K * N);
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

TEST(nntrainer_cpu_backend_standalone, qai8dxp_qsi4cxp_512x768x2048_CMP) {
  const unsigned int M = 512;
  const unsigned int K = 768;
  const unsigned int N = 2048;
  float qai8dxp_qsi4cxp_mse;
  float qai8dxp_qsi4cxp_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qai8dxp_qsi4cxp_test_unpacked(
      M, K, N, qai8dxp_qsi4cxp_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qai8dxp_qsi4cxp_test_packed(M, K, N, qai8dxp_qsi4cxp_mse_packed,
                                  opt_idx_variant, true, false);
  ASSERT_LE(qai8dxp_qsi4cxp_mse, eps * M * K * N);
  ASSERT_LE(qai8dxp_qsi4cxp_mse_packed, eps * M * K * N);
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

TEST(nntrainer_cpu_backend_standalone, qai8dxp_qsi4cxp_3072x512x512_CMP) {
  const unsigned int M = 3072;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float qai8dxp_qsi4cxp_mse;
  float qai8dxp_qsi4cxp_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qai8dxp_qsi4cxp_test_unpacked(
      M, K, N, qai8dxp_qsi4cxp_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qai8dxp_qsi4cxp_test_packed(M, K, N, qai8dxp_qsi4cxp_mse_packed,
                                  opt_idx_variant, true, false);
  ASSERT_LE(qai8dxp_qsi4cxp_mse, eps * M * K * N);
  ASSERT_LE(qai8dxp_qsi4cxp_mse_packed, eps * M * K * N);
}

std::tuple<float, uint32_t> test_gemm_qsi8d32p_qsi4c32p_unpacked(
  const uint32_t M, const uint32_t K, const uint32_t N, const float *weights,
  const float *activations, std::vector<float> &ref_dst, bool transB = true,
  bool print = false) {
  // Step1. Set qsi8d32p_qsi4c32p quant test components
  // For qs4c32 format with block size 32:
  // - Each block has: sizeof(uint16_t) (scale as fp16) + bl/2 bytes (4-bit
  // packed data)
  // - Number of blocks per row: K / bl
  const size_t bl = 32; // block length
  const size_t num_blocks_per_row = K / bl;
  const size_t bytes_per_block =
    sizeof(uint16_t) + bl / 2; // fp16 scale + packed 4-bit data
  const size_t rhs_native_size_qs4c32 =
    static_cast<size_t>(N) * num_blocks_per_row * bytes_per_block;

  uint8_t *rhs_native_mtx_qs4c32 = new uint8_t[rhs_native_size_qs4c32];
  std::memset(rhs_native_mtx_qs4c32, 0, rhs_native_size_qs4c32);

  // Step2. 4-bit Weight quantization with block size 32 (qsi4c32p format)
  nntrainer::nntr_quant_qs4c32_f32(N, K, bl, (void *)weights,
                                   (void *)rhs_native_mtx_qs4c32);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(static_cast<size_t>(M) * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  uint32_t opt_kernel_variant_idx =
    nntrainer::nntr_gemm_qsi8d32p_qsi4c32p_unpacked(
      M, N, K, (void *)activations, (void *)rhs_native_mtx_qs4c32, nullptr,
      dst.data(), transB); // scales are embedded in qs4c32 format
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] test_gemm_qsi8d32p_qsi4c32p_unpacked: " << dt.count()
              << " ns " << dt.count() / 1'000 << " us "
              << dt.count() / 1'000'000 << " ms " << std::endl;
  }

  // Step4. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);

  delete[] rhs_native_mtx_qs4c32;

  return {mean_squared_error, opt_kernel_variant_idx};
}

static uint32_t run_qsi8d32p_qsi4c32p_test_unpacked(
  const uint32_t M, const uint32_t K, const uint32_t N,
  float &qsi8d32p_qsi4c32p_mse, bool transB = true, bool print = false) {
  if (print) {
    std::cout << "[INFO] qsi8d32p_qsi4c32p Test (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
  }

  std::vector<float> activation =
    generate_random_vector<float>(static_cast<std::size_t>(M) * K);
  std::vector<float> weight =
    generate_random_vector<float>(static_cast<std::size_t>(N) * K);
  std::vector<float> ref_dst(static_cast<std::size_t>(M) * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm :    " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }
  const auto [mse, opt_kernel_variant_idx] =
    test_gemm_qsi8d32p_qsi4c32p_unpacked(
      M, K, N, weight.data(), activation.data(), ref_dst, transB, print);

  qsi8d32p_qsi4c32p_mse = mse;

  return opt_kernel_variant_idx;
}

float test_gemm_qsi8d32p_qsi4c32p_packed(
  const uint32_t M, const uint32_t K, const uint32_t N, const float *weights,
  const float *activations, std::vector<float> &ref_dst,
  uint32_t opt_kernel_idx, bool transB = true, bool print = false) {
  // Step1. Set qsi8d32p_qsi4c32p quant test components using qs4c32 format
  // For qs4c32 format with block size 32:
  // - Each block has: sizeof(uint16_t) (scale as fp16) + bl/2 bytes (4-bit
  // packed data)
  // - Number of blocks per row: K / bl
  const size_t bl = 32; // block length
  const size_t num_blocks_per_row = K / bl;
  const size_t bytes_per_block =
    sizeof(uint16_t) + bl / 2; // fp16 scale + packed 4-bit data
  const size_t rhs_native_size_qs4c32 =
    static_cast<size_t>(N) * num_blocks_per_row * bytes_per_block;

  uint8_t *rhs_native_mtx_qs4c32 = new uint8_t[rhs_native_size_qs4c32];
  std::memset(rhs_native_mtx_qs4c32, 0, rhs_native_size_qs4c32);

  // Step2. 4-bit Weight quantization with block size 32 (qsi4c32 format with
  // embedded fp16 scales)
  nntrainer::nntr_quant_qs4c32_f32(N, K, bl, (void *)weights,
                                   (void *)rhs_native_mtx_qs4c32);

  // Step3. Offline weight packing
  size_t packed_weight_size =
    nntrainer::nntr_get_rhs_packed_size_qsi8d32p_qsi4c32p(N, K, opt_kernel_idx,
                                                          transB);
  uint8_t *packed_weight = new uint8_t[packed_weight_size];

  nntrainer::nntr_qsi8d32p_qsi4c32p_rhs_pack(
    N, K, packed_weight, rhs_native_mtx_qs4c32, nullptr, opt_kernel_idx,
    transB); // scales are embedded in qs4c32 format

  // Step4. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(static_cast<size_t>(M) * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::nntr_gemm_qsi8d32p_qsi4c32p_packed(
    M, N, K, (void *)activations, (void *)packed_weight, dst.data(),
    opt_kernel_idx, transB);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] test_gemm_qsi8d32p_qsi4c32p_packed: " << dt.count()
              << " ns " << dt.count() / 1'000 << " us "
              << dt.count() / 1'000'000 << " ms " << std::endl;
  }

  // Step5. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);

  delete[] rhs_native_mtx_qs4c32;
  delete[] packed_weight;

  return mean_squared_error;
}

void run_qsi8d32p_qsi4c32p_test_packed(const uint32_t M, const uint32_t K,
                                       const uint32_t N,
                                       float &qsi8d32p_qsi4c32p_mse,
                                       uint32_t opt_kernel_idx,
                                       bool transB = true, bool print = false) {
  if (print) {
    std::cout << "[INFO] run_qsi8d32p_qsi4c32p_test_packed Test (M:" << M
              << ", K:" << K << ", N:" << N
              << ") with opt_kernel_idx : " << opt_kernel_idx << std::endl;
  }

  std::vector<float> activation =
    generate_random_vector<float>(static_cast<std::size_t>(M) * K);
  std::vector<float> weight =
    generate_random_vector<float>(static_cast<std::size_t>(N) * K);
  std::vector<float> ref_dst(static_cast<std::size_t>(M) * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm :    " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }
  qsi8d32p_qsi4c32p_mse = test_gemm_qsi8d32p_qsi4c32p_packed(
    M, K, N, weight.data(), activation.data(), ref_dst, opt_kernel_idx, transB,
    print);
}

TEST(nntrainer_cpu_backend_standalone, qsi8d32p_qsi4c32p_1x3072x512_CMP) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 512;
  float qsi8d32p_qsi4c32p_mse;
  float qsi8d32p_qsi4c32p_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qsi8d32p_qsi4c32p_test_unpacked(
      M, K, N, qsi8d32p_qsi4c32p_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qsi8d32p_qsi4c32p_test_packed(M, K, N, qsi8d32p_qsi4c32p_mse_packed,
                                    opt_idx_variant, true, false);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse, eps * M * K * N);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse_packed, eps * M * K * N);
}

TEST(nntrainer_cpu_backend_standalone, qsi8d32p_qsi4c32p_768x768x768_CMP) {
  const unsigned int M = 768;
  const unsigned int K = 768;
  const unsigned int N = 768;
  float qsi8d32p_qsi4c32p_mse;
  float qsi8d32p_qsi4c32p_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qsi8d32p_qsi4c32p_test_unpacked(
      M, K, N, qsi8d32p_qsi4c32p_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qsi8d32p_qsi4c32p_test_packed(M, K, N, qsi8d32p_qsi4c32p_mse_packed,
                                    opt_idx_variant, true, false);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse, eps * M * K * N);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse_packed, eps * M * K * N);
}

TEST(nntrainer_cpu_backend_standalone, qsi8d32p_qsi4c32p_512x768x2048_CMP) {
  const unsigned int M = 512;
  const unsigned int K = 768;
  const unsigned int N = 2048;
  float qsi8d32p_qsi4c32p_mse;
  float qsi8d32p_qsi4c32p_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qsi8d32p_qsi4c32p_test_unpacked(
      M, K, N, qsi8d32p_qsi4c32p_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qsi8d32p_qsi4c32p_test_packed(M, K, N, qsi8d32p_qsi4c32p_mse_packed,
                                    opt_idx_variant, true, false);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse, eps * M * K * N);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse_packed, eps * M * K * N);
}

TEST(nntrainer_cpu_backend_standalone, qsi8d32p_qsi4c32p_3072x512x512_CMP) {
  const unsigned int M = 3072;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float qsi8d32p_qsi4c32p_mse;
  float qsi8d32p_qsi4c32p_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qsi8d32p_qsi4c32p_test_unpacked(
      M, K, N, qsi8d32p_qsi4c32p_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qsi8d32p_qsi4c32p_test_packed(M, K, N, qsi8d32p_qsi4c32p_mse_packed,
                                    opt_idx_variant, true, false);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse, eps * M * K * N);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse_packed, eps * M * K * N);
}

TEST(nntrainer_cpu_backend_standalone, qsi8d32p_qsi4c32p_3072x102x1024_CMP) {
  const unsigned int M = 3072;
  const unsigned int K = 1024;
  const unsigned int N = 1024;
  float qsi8d32p_qsi4c32p_mse;
  float qsi8d32p_qsi4c32p_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qsi8d32p_qsi4c32p_test_unpacked(
      M, K, N, qsi8d32p_qsi4c32p_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qsi8d32p_qsi4c32p_test_packed(M, K, N, qsi8d32p_qsi4c32p_mse_packed,
                                    opt_idx_variant, true, false);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse, eps * M * K * N);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse_packed, eps * M * K * N);
}

/**
 * @brief Test helper function for osv32_isv2 to qsi4c32p transform
 *
 * Tests the lossless transformation from OpenVINO osv32_isv2 format to
 * KleidiAI qsi4c32p packed format by:
 * 1. Generating random FP32 weights
 * 2. Quantizing to osv32_isv2 format using Int4Utils
 * 3. Transforming to qsi4c32p using nntr_kai_repack_osv32_to_qsi4c32p
 * 4. Running GEMM with packed weights
 * 5. Comparing against FP32 reference GEMM
 */
static void run_transform_osv32_to_qsi4c32p_test(const uint32_t K,
                                                 const uint32_t N,
                                                 uint32_t kernel_idx = 3,
                                                 bool print = false) {
  const uint32_t M = 3072;      // Batch size for GEMM test
  const size_t group_size = 32; // Fixed group size

  // Step 1: Generate random FP32 weights
  std::vector<float> weight_fp32 =
    generate_random_vector<float>(N * K, -1.0f, 1.0f);

  // Step 2: Quantize to osv32_isv2 format
  std::vector<uint8_t> osv32_weights;
  std::vector<uint16_t> osv32_scales;
  nntrainer::Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, group_size,
                                          osv32_weights, osv32_scales);

  // Step 3: Transform osv32_isv2 -> qsi4c32p packed
  size_t packed_size = 0;
  size_t expected_packed_size =
    nntr_kai_get_rhs_packed_size_qsi8d32p_qsi4c32p(N, K, kernel_idx, true);
  std::vector<uint8_t> qsi4c32p_packed(expected_packed_size);

  auto t0 = high_resolution_clock::now();
  nntr_kai_repack_osv32_to_qsi4c32p(N, K, osv32_weights.data(),
                                    osv32_scales.data(), qsi4c32p_packed.data(),
                                    packed_size, kernel_idx, true);
  auto t1 = high_resolution_clock::now();
  auto transform_time = duration_cast<microseconds>(t1 - t0);

  if (print) {
    std::cout << "[INFO] Transform time: " << transform_time.count() << " us"
              << std::endl;
    std::cout << "[INFO] Packed size: " << packed_size << " bytes" << std::endl;
  }

  // Step 4: Generate random FP32 activations
  std::vector<float> activations =
    generate_random_vector<float>(M * K, -1.0f, 1.0f);

  // Step 5: Run FP32 reference GEMM
  std::vector<float> ref_dst(M * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.0f, activations.data(), K,
                   weight_fp32.data(), K, 0.0f, ref_dst.data(), N);

  // Step 6: Run GEMM with transformed qsi4c32p weights
  std::vector<float> qsi4c32p_dst(M * N, 0.0f);
  nntrainer::nntr_gemm_qsi8d32p_qsi4c32p_packed(
    M, N, K, (void *)activations.data(), (void *)qsi4c32p_packed.data(),
    qsi4c32p_dst.data(), kernel_idx, true);

  // Step 7: Compute MSE and cosine similarity
  float mean_squared_error = compute_mse(M, N, ref_dst, qsi4c32p_dst, print);
  float cos_sim =
    cosine_similarity<float, float>(ref_dst.data(), qsi4c32p_dst.data(), M * N);

  if (print) {
    std::cout << "[INFO] MSE: " << mean_squared_error
              << ", Cosine Sim: " << cos_sim << std::endl;
  }

  // Step 8: Assert quality metrics
  // For 4-bit quantization, expect some quantization noise
  const float mse_threshold = 0.6f;      // Allow quantization noise
  const float cos_sim_threshold = 0.99f; // High similarity expected

  EXPECT_LE(mean_squared_error, mse_threshold);
  EXPECT_GE(cos_sim, cos_sim_threshold);
}

#define DECLARE_transform_osv32_to_qsi4c32p_test(K, N)                         \
  TEST(nntrainer_cpu_backend_standalone,                                       \
       transform_osv32_to_qsi4c32p_K##K##_N##N) {                              \
    run_transform_osv32_to_qsi4c32p_test(K, N, 3, true);                       \
  }

// Test cases with various K and N dimensions
DECLARE_transform_osv32_to_qsi4c32p_test(128, 64);
DECLARE_transform_osv32_to_qsi4c32p_test(256, 128);
DECLARE_transform_osv32_to_qsi4c32p_test(512, 256);
DECLARE_transform_osv32_to_qsi4c32p_test(512, 512);
DECLARE_transform_osv32_to_qsi4c32p_test(1024, 512);
DECLARE_transform_osv32_to_qsi4c32p_test(1024, 1024);

TEST(nntrainer_cpu_backend_standalone, trigonometric_values_test) {

  const unsigned int N = 3072;
  run_trigonometric_values_test(N);
}

/**
 * @brief Benchmark comparison of three GEMM implementations
 *
 * Compares latency of:
 * - nntr_gemm_qsi8d32p_qsi4c32p_packed (KleidiAI with block size 32)
 * - nntr_gemm_qai8dxp_qsi4cxp_packed (KleidiAI with dynamic block)
 * - gemm_q4_0<float> (GGML-style Q4_0 GEMM)
 */
void run_gemm_benchmark_comparison(const uint32_t M, const uint32_t K,
                                   const uint32_t N,
                                   const uint32_t warmup_iters = 3,
                                   const uint32_t test_iters = 5,
                                   bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "\n=========================================" << std::endl;
    std::cout << "[BENCHMARK] GEMM Latency Comparison (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
    std::cout << "=========================================\n" << std::endl;
  }

  // Generate random data
  std::vector<float> activation =
    generate_random_vector<float>(static_cast<std::size_t>(M) * K);
  std::vector<float> weight =
    generate_random_vector<float>(static_cast<std::size_t>(N) * K);

  // ============================================================
  // Setup 1: qsi8d32p_qsi4c32p (KleidiAI with block size 32)
  // ============================================================
  const size_t bl = 32;
  const size_t num_blocks_per_row = K / bl;
  const size_t bytes_per_block = sizeof(uint16_t) + bl / 2;
  const size_t rhs_native_size_qs4c32 =
    static_cast<size_t>(N) * num_blocks_per_row * bytes_per_block;

  std::vector<uint8_t> rhs_native_mtx_qs4c32(rhs_native_size_qs4c32, 0);
  nntrainer::nntr_quant_qs4c32_f32(N, K, bl, (void *)weight.data(),
                                   (void *)rhs_native_mtx_qs4c32.data());

  // Get optimal kernel index for qsi8d32p_qsi4c32p by running unpacked version
  float dummy_mse;
  std::vector<float> ref_dst(static_cast<std::size_t>(M) * N);
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  const auto [mse_qsi8d32p, opt_idx_qsi8d32p] =
    test_gemm_qsi8d32p_qsi4c32p_unpacked(
      M, K, N, weight.data(), activation.data(), ref_dst, true, false);
  if (print) {
    std::cout << "[INFO] qsi8d32p_qsi4c32p optimal kernel index: "
              << opt_idx_qsi8d32p << std::endl;
  }

  // Pack weights for qsi8d32p_qsi4c32p
  size_t packed_weight_size_qsi8d32p =
    nntrainer::nntr_get_rhs_packed_size_qsi8d32p_qsi4c32p(
      N, K, opt_idx_qsi8d32p, true);
  std::vector<uint8_t> packed_weight_qsi8d32p(packed_weight_size_qsi8d32p);
  nntrainer::nntr_qsi8d32p_qsi4c32p_rhs_pack(
    N, K, packed_weight_qsi8d32p.data(), rhs_native_mtx_qs4c32.data(), nullptr,
    opt_idx_qsi8d32p, true);

  // ============================================================
  // Setup 2: qai8dxp_qsi4cxp (KleidiAI with dynamic block)
  // ============================================================
  const size_t rhs_native_size_qs4cx =
    static_cast<size_t>(N) * (((K + 2 - 1) / 2) * 2 / 2) * sizeof(uint8_t);
  const size_t rhs_scales_size_f32 = N * sizeof(float);

  std::vector<uint8_t> rhs_native_mtx_qs4cx(rhs_native_size_qs4cx);
  std::vector<uint8_t> rhs_scales_f32(rhs_scales_size_f32);
  nntrainer::nntr_quant_qs4cx_f32(N, K, (void *)weight.data(),
                                  (void *)rhs_native_mtx_qs4cx.data(),
                                  rhs_scales_f32.data(), true);

  // Get optimal kernel index for qai8dxp_qsi4cxp by running unpacked version
  const auto [mse_qai8dxp, opt_idx_qai8dxp] =
    test_gemm_qai8dxp_qsi4cxp_unpacked(M, K, N, weight.data(),
                                       activation.data(), ref_dst, true, false);
  if (print) {
    std::cout << "[INFO] qai8dxp_qsi4cxp optimal kernel index: "
              << opt_idx_qai8dxp << std::endl;
  }

  // Pack weights for qai8dxp_qsi4cxp
  size_t packed_weight_size_qai8dxp =
    nntrainer::nntr_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(N, K, opt_idx_qai8dxp,
                                                          true);
  std::vector<uint8_t> packed_weight_qai8dxp(packed_weight_size_qai8dxp);
  nntrainer::nntr_qsi4cxp_qs4cxs1s0_rhs_pack(
    N, K, packed_weight_qai8dxp.data(), rhs_native_mtx_qs4cx.data(),
    rhs_scales_f32.data(), opt_idx_qai8dxp, true);

  // ============================================================
  // Setup 3: gemm_q4_0<float> (GGML-style Q4_0)
  // ============================================================
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t q4_0_block_size = 32;
  size_t q4_0_data_size = q4_0_type_size * N / q4_0_block_size;
  q4_0_data_size *= K;
  std::vector<char> q4_0_offline_qWeight(q4_0_data_size);
  nntrainer::quantize_q4_0(weight.data(), (void *)q4_0_offline_qWeight.data(),
                           N, K, nullptr);

  std::vector<char> q4_0_repacked_qWeight(q4_0_data_size);
  nntrainer::repack_q4_0(q4_0_repacked_qWeight.data(),
                         q4_0_offline_qWeight.data(), q4_0_data_size, N, K);

  // Output buffers
  std::vector<float> dst_qsi8d32p(static_cast<size_t>(M) * N);
  std::vector<float> dst_qai8dxp(static_cast<size_t>(M) * N);
  std::vector<float> dst_q4_0(static_cast<size_t>(M) * N);

  // ============================================================
  // Warm-up runs
  // ============================================================
  if (print) {
    std::cout << "[INFO] Warm-up (" << warmup_iters << " iterations)..."
              << std::endl;
  }
  for (uint32_t i = 0; i < warmup_iters; ++i) {
    nntrainer::nntr_gemm_qsi8d32p_qsi4c32p_packed(
      M, N, K, (void *)activation.data(), (void *)packed_weight_qsi8d32p.data(),
      dst_qsi8d32p.data(), opt_idx_qsi8d32p, true);

    nntrainer::nntr_gemm_qai8dxp_qsi4cxp_packed(
      M, N, K, (void *)activation.data(), (void *)packed_weight_qai8dxp.data(),
      dst_qai8dxp.data(), opt_idx_qai8dxp, true);

    nntrainer::gemm_q4_0<float>(M, N, K, activation.data(), K,
                                (void *)q4_0_repacked_qWeight.data(), N,
                                dst_q4_0.data(), N);
  }

  // ============================================================
  // Benchmark: qsi8d32p_qsi4c32p_packed
  // ============================================================
  nanoseconds total_time_qsi8d32p = nanoseconds(0);
  for (uint32_t i = 0; i < test_iters; ++i) {
    auto t1 = high_resolution_clock::now();
    nntrainer::nntr_gemm_qsi8d32p_qsi4c32p_packed(
      M, N, K, (void *)activation.data(), (void *)packed_weight_qsi8d32p.data(),
      dst_qsi8d32p.data(), opt_idx_qsi8d32p, true);
    auto t2 = high_resolution_clock::now();
    total_time_qsi8d32p += duration_cast<nanoseconds>(t2 - t1);
  }

  // ============================================================
  // Benchmark: qai8dxp_qsi4cxp_packed
  // ============================================================
  nanoseconds total_time_qai8dxp = nanoseconds(0);
  for (uint32_t i = 0; i < test_iters; ++i) {
    auto t1 = high_resolution_clock::now();
    nntrainer::nntr_gemm_qai8dxp_qsi4cxp_packed(
      M, N, K, (void *)activation.data(), (void *)packed_weight_qai8dxp.data(),
      dst_qai8dxp.data(), opt_idx_qai8dxp, true);
    auto t2 = high_resolution_clock::now();
    total_time_qai8dxp += duration_cast<nanoseconds>(t2 - t1);
  }

  // ============================================================
  // Benchmark: gemm_q4_0<float>
  // ============================================================
  nanoseconds total_time_q4_0 = nanoseconds(0);
  for (uint32_t i = 0; i < test_iters; ++i) {
    auto t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_0<float>(M, N, K, activation.data(), K,
                                (void *)q4_0_repacked_qWeight.data(), N,
                                dst_q4_0.data(), N);
    auto t2 = high_resolution_clock::now();
    total_time_q4_0 += duration_cast<nanoseconds>(t2 - t1);
  }

  // ============================================================
  // Print results
  // ============================================================
  auto avg_ns_qsi8d32p = total_time_qsi8d32p.count() / test_iters;
  auto avg_ns_qai8dxp = total_time_qai8dxp.count() / test_iters;
  auto avg_ns_q4_0 = total_time_q4_0.count() / test_iters;

  if (print) {
    std::cout << "\n-----------------------------------------" << std::endl;
    std::cout << "[RESULT] Average latency over " << test_iters
              << " iterations:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "  qsi8d32p_qsi4c32p_packed: " << avg_ns_qsi8d32p << " ns ("
              << avg_ns_qsi8d32p / 1'000 << " us, "
              << avg_ns_qsi8d32p / 1'000'000 << " ms)" << std::endl;
    std::cout << "  qai8dxp_qsi4cxp_packed:   " << avg_ns_qai8dxp << " ns ("
              << avg_ns_qai8dxp / 1'000 << " us, " << avg_ns_qai8dxp / 1'000'000
              << " ms)" << std::endl;
    std::cout << "  gemm_q4_0<float>:         " << avg_ns_q4_0 << " ns ("
              << avg_ns_q4_0 / 1'000 << " us, " << avg_ns_q4_0 / 1'000'000
              << " ms)" << std::endl;
    std::cout << "-----------------------------------------\n" << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, gemm_benchmark_comparison_32x1024x4096) {
  run_gemm_benchmark_comparison(32, 1024, 4096);
}

TEST(nntrainer_cpu_backend_standalone, gemm_benchmark_comparison_1x3072x512) {
  run_gemm_benchmark_comparison(1, 3072, 512);
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
