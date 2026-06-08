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
#ifdef __ANDROID__
// KleidiAI interface is only wired in via Android.mk; on Linux/meson the
// header (and its qsi8d32p/qsi4c32p backend) is unavailable.
#include "kleidiai_interface.h"
#endif
#include "nntrainer_test_util.h"
#if defined(ENABLE_TEST) && (defined(__x86_64__) || defined(_M_X64))
#include <hgemm_common.h>
#include <hgemm_pack.h>
#include <hgemm_test.h>
#define X86_HGEMM_WORKSPACE_STATS_AVAILABLE 1
#endif
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cpu_backend.h>
#include <fallback_internal.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
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
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight_fp16.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm :    " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
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

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_457x1024x1024) {
  const unsigned int M = 457;
  const unsigned int K = 1024;
  const unsigned int N = 1024;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_458x1024x1024) {
  const unsigned int M = 458;
  const unsigned int K = 1024;
  const unsigned int N = 1024;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_459x1024x1024) {
  const unsigned int M = 459;
  const unsigned int K = 1024;
  const unsigned int N = 1024;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x1024x1024) {
  const unsigned int M = 1024;
  const unsigned int K = 1024;
  const unsigned int N = 1024;
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

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x1024x1024) {
  const unsigned int M = 1;
  const unsigned int K = 1024;
  const unsigned int N = 1024;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

#ifdef __ANDROID__
// nntrainer::sine / cosine have no _FP16 instantiation in the x86 backend
// (FP32-only). The helper and TEST below stay Android-only until the x86
// backend grows an FP16 sine/cosine path.
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

#ifdef __ANDROID__
// The qsi8d32p_qsi4c32p path is only exposed on Android via KleidiAI; the
// x86 cpu_backend does not provide nntr_quant_qs4c32_f32 /
// nntr_*_qsi8d32p_qsi4c32p_*, so these helpers and TESTs are excluded on
// Linux/meson builds.
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
#endif // __ANDROID__ (end of qsi8d32p_qsi4c32p region)

TEST(nntrainer_cpu_backend_standalone, trigonometric_values_test) {

  const unsigned int N = 3072;
  run_trigonometric_values_test(N);
}
#endif // __ANDROID__ (end of trigonometric_values_test region)

#ifdef __ANDROID__
// gemm_benchmark_comparison mixes the qai8dxp path (x86-available) with the
// qsi8d32p path (KleidiAI-only); both are needed for the three-way comparison,
// so the whole benchmark is gated to Android.
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
#endif // __ANDROID__ (end of gemm_benchmark_comparison region)

/// FP16 sgemm path: exercises the x86 cache-blocked GEMM.
/// Reference is a local FP32 GEMM loop to avoid using the optimized backend as
/// the oracle for this backend test.
static void run_sgemm_fp16_hgemm_test(unsigned int M, unsigned int N,
                                      unsigned int K, bool TransA = false,
                                      bool TransB = false, float alpha = 1.0F,
                                      float beta = 0.0F,
                                      unsigned int lda_extra = 0,
                                      unsigned int ldb_extra = 0,
                                      unsigned int ldc_extra = 0) {
  nntrainer::init_backend();

  const unsigned int lda = std::max(1u, (TransA ? M : K) + lda_extra);
  const unsigned int ldb = std::max(1u, (TransB ? K : N) + ldb_extra);
  const unsigned int ldc = std::max(1u, N + ldc_extra);
  const unsigned int a_rows = TransA ? K : M;
  const unsigned int b_rows = TransB ? N : K;
  const std::size_t a_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(a_rows) * lda);
  const std::size_t b_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(b_rows) * ldb);
  const std::size_t c_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(M) * ldc);

  auto A_fp16 = generate_random_vector<_FP16>(a_size, -0.25F, 0.25F);
  auto B_fp16 = generate_random_vector<_FP16>(b_size, -0.25F, 0.25F);
  auto C_fp16 = generate_random_vector<_FP16>(c_size, -0.25F, 0.25F);
  auto C_before = C_fp16;

  std::vector<float> C_fp32_ref(c_size);
  for (std::size_t i = 0; i < c_size; ++i) {
    C_fp32_ref[i] = static_cast<float>(C_before[i]);
  }

  if (M != 0 && N != 0) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N; ++n) {
        float acc = 0.0F;
        for (unsigned int k = 0; k < K; ++k) {
          const float a = static_cast<float>(TransA ? A_fp16[k * lda + m]
                                                    : A_fp16[m * lda + k]);
          const float b = static_cast<float>(TransB ? B_fp16[n * ldb + k]
                                                    : B_fp16[k * ldb + n]);
          acc += a * b;
        }
        const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
        C_fp32_ref[idx] = alpha * acc + beta * C_fp32_ref[idx];
      }
    }
  }

  // System under test: FP16 sgemm, routed to x86::hgemm_fp16 by the x86
  // backend dispatcher for row-major inputs.
  nntrainer::sgemm(0, TransA, TransB, M, N, K, alpha, A_fp16.data(), lda,
                   B_fp16.data(), ldb, beta, C_fp16.data(), ldc);

  if (M == 0 || N == 0) {
    for (std::size_t i = 0; i < c_size; ++i) {
      EXPECT_EQ(static_cast<float>(C_fp16[i]), static_cast<float>(C_before[i]))
        << "zero-dimension GEMM touched C at i=" << i << " M=" << M
        << " N=" << N << " K=" << K << " TransA=" << TransA
        << " TransB=" << TransB;
    }
    return;
  }

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
      const float got = static_cast<float>(C_fp16[idx]);
      const float ref = C_fp32_ref[idx];
      const float abs_diff = std::abs(got - ref);
      const float rel_diff = abs_diff / std::max(1.0F, std::abs(ref));
      EXPECT_TRUE(abs_diff <= 1e-2F || rel_diff <= 1e-2F)
        << "mismatch at m=" << m << " n=" << n << " M=" << M << " N=" << N
        << " K=" << K << " TransA=" << TransA << " TransB=" << TransB
        << " alpha=" << alpha << " beta=" << beta << " lda=" << lda
        << " ldb=" << ldb << " ldc=" << ldc << " got=" << got << " ref=" << ref
        << " abs_diff=" << abs_diff << " rel_diff=" << rel_diff;
    }

    for (unsigned int n = N; n < ldc; ++n) {
      const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
      EXPECT_EQ(static_cast<float>(C_fp16[idx]),
                static_cast<float>(C_before[idx]))
        << "C padding was modified at m=" << m << " n=" << n << " ldc=" << ldc;
    }
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_aligned_12x32x32) {
  run_sgemm_fp16_hgemm_test(12, 32, 32);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_aligned_256x512x128) {
  run_sgemm_fp16_hgemm_test(256, 512, 128);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_aligned_64x64x64) {
  run_sgemm_fp16_hgemm_test(64, 64, 64);
}

// Large shapes that exceed the 8M-FLOP parallel threshold and force the blocked
// path's panel subdivision: square (N-subdivision) and tall-thin
// (M-subdivision). Run the whole binary under NNTR_NUM_THREADS>1 to exercise
// the multi-threaded panel grid.
TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_parallel_square_512x512x512) {
  run_sgemm_fp16_hgemm_test(512, 512, 512);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_parallel_tall_thin_1024x96x256) {
  run_sgemm_fp16_hgemm_test(1024, 96, 256);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_parallel_wide_96x1024x256) {
  run_sgemm_fp16_hgemm_test(96, 1024, 256);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_parallel_square_alpha_beta_640x384x320) {
  run_sgemm_fp16_hgemm_test(640, 384, 320, false, false, 0.75F, 0.5F);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_parallel_transAB_384x320x512) {
  run_sgemm_fp16_hgemm_test(384, 320, 512, true, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_unaligned_7x17x33) {
  // Exercises both M-edge (7 = 6 + 1) and N-edge (17 = 16 + 1) cleanup.
  run_sgemm_fp16_hgemm_test(7, 17, 33);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_unaligned_13x33x65) {
  run_sgemm_fp16_hgemm_test(13, 33, 65);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_alpha_beta_13x33x65) {
  run_sgemm_fp16_hgemm_test(13, 33, 65, false, false, -0.75F, 0.25F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_transA_unaligned_13x33x65) {
  run_sgemm_fp16_hgemm_test(13, 33, 65, true, false, 0.5F, -0.125F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_transB_unaligned_13x33x65) {
  run_sgemm_fp16_hgemm_test(13, 33, 65, false, true, 1.25F, 0.5F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_transAB_unaligned_13x33x65) {
  run_sgemm_fp16_hgemm_test(13, 33, 65, true, true, -1.0F, 0.125F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_alpha_beta_boundary_cases) {
  struct Case {
    float alpha;
    float beta;
  };

  const std::vector<Case> cases = {
    {0.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {2.5F, -1.0F}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("alpha=" + std::to_string(tc.alpha) +
                 " beta=" + std::to_string(tc.beta));
    run_sgemm_fp16_hgemm_test(13, 33, 65, false, false, tc.alpha, tc.beta);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_zero_dimension_cases) {
  run_sgemm_fp16_hgemm_test(0, 17, 33, false, false, 1.0F, 0.0F);
  run_sgemm_fp16_hgemm_test(7, 0, 33, false, false, 1.0F, 0.0F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_zero_k_beta_cases) {
  struct Case {
    float beta;
  };

  const std::vector<Case> cases = {{0.0F}, {1.0F}, {-0.5F}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("K=0 beta=" + std::to_string(tc.beta));
    run_sgemm_fp16_hgemm_test(7, 17, 0, false, false, 1.0F, tc.beta);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_prime_size_all_transposes) {
  struct Case {
    bool trans_a;
    bool trans_b;
  };

  const std::vector<Case> cases = {
    {false, false}, {false, true}, {true, false}, {true, true}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("TransA=" + std::to_string(tc.trans_a) +
                 " TransB=" + std::to_string(tc.trans_b));
    run_sgemm_fp16_hgemm_test(97, 53, 71, tc.trans_a, tc.trans_b, 0.75F,
                              -0.25F);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_small_tile_cases) {
  run_sgemm_fp16_hgemm_test(1, 1, 1, false, false, 1.0F, 0.0F);
  run_sgemm_fp16_hgemm_test(5, 15, 7, false, false, 1.25F, -0.5F);
  run_sgemm_fp16_hgemm_test(5, 15, 7, true, true, -0.75F, 1.0F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_row_fast_path_cases) {
  run_sgemm_fp16_hgemm_test(1, 257, 129, false, false, 0.75F, -0.25F, 3, 5, 7);
  run_sgemm_fp16_hgemm_test(1, 257, 129, true, false, -0.5F, 0.125F, 2, 4, 3);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_small_k_notrans_large_padded) {
  run_sgemm_fp16_hgemm_test(129, 257, 4, false, false, -0.75F, 0.25F, 3, 5, 7);
  run_sgemm_fp16_hgemm_test(129, 257, 4, true, false, 0.5F, -0.125F, 2, 6, 3);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_skinny_direct_cases) {
  run_sgemm_fp16_hgemm_test(129, 1, 67, false, false, 1.25F, -0.5F, 5, 3, 2);
  run_sgemm_fp16_hgemm_test(97, 2, 71, true, true, -0.75F, 0.25F, 4, 6, 3);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_skinny_fast_path_all_transposes) {
  struct Case {
    unsigned int n;
    bool trans_a;
    bool trans_b;
  };

  const std::vector<Case> cases = {
    {1, false, false}, {1, false, true}, {1, true, false}, {1, true, true},
    {2, false, false}, {2, false, true}, {2, true, false}, {2, true, true}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("N=" + std::to_string(tc.n) +
                 " TransA=" + std::to_string(tc.trans_a) +
                 " TransB=" + std::to_string(tc.trans_b));
    run_sgemm_fp16_hgemm_test(131, tc.n, 67, tc.trans_a, tc.trans_b, 0.75F,
                              -0.25F, 3, 5, 7);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_row_transB_fast_path_cases) {
  run_sgemm_fp16_hgemm_test(1, 257, 129, false, true, 0.75F, -0.25F, 3, 5, 7);
  run_sgemm_fp16_hgemm_test(1, 257, 129, true, true, -0.5F, 0.125F, 2, 4, 3);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_padded_lda_ldb_ldc_all_transposes) {
  struct Case {
    bool trans_a;
    bool trans_b;
    float alpha;
    float beta;
    unsigned int lda_extra;
    unsigned int ldb_extra;
    unsigned int ldc_extra;
  };

  const std::vector<Case> cases = {
    {false, false, 1.0F, 0.0F, 4, 7, 5},
    {false, true, -0.5F, 0.125F, 6, 2, 1},
    {true, false, 0.75F, -0.25F, 3, 8, 4},
    {true, true, -1.25F, 0.5F, 5, 3, 6},
  };

  for (const auto &tc : cases) {
    SCOPED_TRACE("TransA=" + std::to_string(tc.trans_a) +
                 " TransB=" + std::to_string(tc.trans_b));
    run_sgemm_fp16_hgemm_test(13, 33, 65, tc.trans_a, tc.trans_b, tc.alpha,
                              tc.beta, tc.lda_extra, tc.ldb_extra,
                              tc.ldc_extra);
  }
}

#ifdef X86_HGEMM_WORKSPACE_STATS_AVAILABLE
TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_reuse_warmed_shape) {
  run_sgemm_fp16_hgemm_test(64, 64, 64);

  nntrainer::avx2::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(64, 64, 64);
  auto same_stats =
    nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(same_stats.total_realloc_count, 0u);

  nntrainer::avx2::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(7, 17, 33);
  auto smaller_stats =
    nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(smaller_stats.total_realloc_count, 0u);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_uses_c32_panel_and_packed_panels) {
  nntrainer::avx2::internal::testing::clear_hgemm_workspace();

  const auto &block = nntrainer::avx2::internal::get_hgemm_block_sizes();
  const unsigned int M = block.m + 1;
  const unsigned int N = block.n + 1;
  const unsigned int K = 5;

  run_sgemm_fp16_hgemm_test(M, N, K, false, false, 0.75F, 0.25F);
  const auto stats =
    nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();

  const auto round_up_to = [](unsigned int value, unsigned int tile) {
    return ((value + tile - 1) / tile) * tile;
  };

  const unsigned int panel_m = std::min<unsigned int>(M, block.m);
  const unsigned int panel_n = std::min<unsigned int>(N, block.n);
  const unsigned int k_block = std::min<unsigned int>(K, block.k);
  const unsigned int tile_m =
    std::min<unsigned int>(panel_m, std::min(block.m, block.c_m));

  const std::size_t expected_c32_capacity =
    static_cast<std::size_t>(round_up_to(panel_m, X86_HGEMM_MR)) *
    round_up_to(panel_n, X86_HGEMM_NR);
  const std::size_t full_c32_capacity =
    static_cast<std::size_t>(round_up_to(M, X86_HGEMM_MR)) *
    round_up_to(N, X86_HGEMM_NR);
  const std::size_t expected_pack_a_capacity =
    static_cast<std::size_t>(round_up_to(tile_m, X86_HGEMM_MR)) * k_block;
  const std::size_t expected_pack_b_capacity =
    static_cast<std::size_t>(k_block) * round_up_to(panel_n, X86_HGEMM_NR);

  EXPECT_EQ(stats.c32_capacity, expected_c32_capacity);
  EXPECT_LT(stats.c32_capacity, full_c32_capacity);
  EXPECT_EQ(stats.pack_a_capacity, expected_pack_a_capacity);
  EXPECT_EQ(stats.pack_b_capacity, expected_pack_b_capacity);
  EXPECT_EQ(stats.scratch_capacity, 0u);
  EXPECT_EQ(stats.total_capacity_bytes,
            (stats.c32_capacity + stats.pack_a_capacity +
             stats.pack_b_capacity + stats.scratch_capacity) *
              sizeof(float));
  EXPECT_EQ(stats.total_realloc_count,
            stats.c32_realloc_count + stats.pack_a_realloc_count +
              stats.pack_b_realloc_count + stats.scratch_realloc_count);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_row_fast_path_uses_scratch_only) {
  nntrainer::avx2::internal::testing::clear_hgemm_workspace();

  const unsigned int N = 257;
  run_sgemm_fp16_hgemm_test(1, N, 129, false, false, 0.75F, -0.25F, 3, 5, 7);
  auto stats = nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();

  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, N);
  EXPECT_EQ(stats.c32_realloc_count, 0u);
  EXPECT_EQ(stats.pack_a_realloc_count, 0u);
  EXPECT_EQ(stats.pack_b_realloc_count, 0u);
  EXPECT_EQ(stats.scratch_realloc_count, 1u);
  EXPECT_EQ(stats.total_realloc_count, 1u);
  EXPECT_EQ(stats.total_capacity_bytes, stats.scratch_capacity * sizeof(float));

  nntrainer::avx2::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(1, N, 129, true, false, -0.5F, 0.125F, 2, 4, 3);
  stats = nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();

  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, N);
  EXPECT_EQ(stats.scratch_realloc_count, 0u);
  EXPECT_EQ(stats.total_realloc_count, 0u);
  EXPECT_EQ(stats.total_capacity_bytes, stats.scratch_capacity * sizeof(float));
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_row_transB_fast_path_uses_scratch_only) {
  nntrainer::avx2::internal::testing::clear_hgemm_workspace();

  const unsigned int K = 129;
  run_sgemm_fp16_hgemm_test(1, 257, K, false, true, 0.75F, -0.25F, 3, 5, 7);
  auto stats = nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();

  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, K);
  EXPECT_EQ(stats.c32_realloc_count, 0u);
  EXPECT_EQ(stats.pack_a_realloc_count, 0u);
  EXPECT_EQ(stats.pack_b_realloc_count, 0u);
  EXPECT_EQ(stats.scratch_realloc_count, 1u);
  EXPECT_EQ(stats.total_realloc_count, 1u);
  EXPECT_EQ(stats.total_capacity_bytes, stats.scratch_capacity * sizeof(float));

  nntrainer::avx2::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(1, 257, K, true, true, -0.5F, 0.125F, 2, 4, 3);
  stats = nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();

  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, K);
  EXPECT_EQ(stats.scratch_realloc_count, 0u);
  EXPECT_EQ(stats.total_realloc_count, 0u);
  EXPECT_EQ(stats.total_capacity_bytes, stats.scratch_capacity * sizeof(float));
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_no_allocation_for_fast_small_shapes) {
  nntrainer::avx2::internal::testing::clear_hgemm_workspace();
  run_sgemm_fp16_hgemm_test(129, 257, 4, false, false, -0.75F, 0.25F, 3, 5, 7);
  auto stats = nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, 0u);
  EXPECT_EQ(stats.total_realloc_count, 0u);

  nntrainer::avx2::internal::testing::clear_hgemm_workspace();
  run_sgemm_fp16_hgemm_test(131, 2, 67, true, true, 0.75F, -0.25F, 3, 5, 7);
  stats = nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, 0u);
  EXPECT_EQ(stats.total_realloc_count, 0u);
}

template <typename SrcT> static void run_packing_B_N16_trans_test() {
  const unsigned int K = 13;
  const unsigned int N = X86_HGEMM_NR;
  const unsigned int stride = K + 5;
  auto src = generate_random_vector<SrcT>(static_cast<std::size_t>(N) * stride,
                                          -1.0F, 1.0F);
  std::vector<float> dst(static_cast<std::size_t>(K) * X86_HGEMM_NR, -123.0F);

  nntrainer::avx2::internal::packing_B_N16_trans(K, N, src.data(), stride,
                                                 dst.data());

  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int n = 0; n < N; ++n) {
      EXPECT_FLOAT_EQ(
        dst[static_cast<std::size_t>(k) * X86_HGEMM_NR + n],
        static_cast<float>(src[static_cast<std::size_t>(n) * stride + k]))
        << "mismatch at k=" << k << " n=" << n;
    }
  }
}

TEST(nntrainer_cpu_backend_standalone, hgemm_pack_B_N16_trans_fp16) {
  run_packing_B_N16_trans_test<_FP16>();
}

TEST(nntrainer_cpu_backend_standalone, hgemm_pack_B_N16_trans_float) {
  run_packing_B_N16_trans_test<float>();
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_no_allocation_for_degenerate_paths) {
  nntrainer::avx2::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(13, 33, 65, false, false, 0.0F, 0.5F);
  auto alpha_zero_stats =
    nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(alpha_zero_stats.total_realloc_count, 0u);

  nntrainer::avx2::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(13, 33, 0, false, false, 1.0F, -0.5F);
  auto k_zero_stats =
    nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(k_zero_stats.total_realloc_count, 0u);

  nntrainer::avx2::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(0, 33, 65, false, false, 1.0F, 0.0F);
  auto m_zero_stats =
    nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(m_zero_stats.total_realloc_count, 0u);

  nntrainer::avx2::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(13, 0, 65, false, false, 1.0F, 0.0F);
  auto n_zero_stats =
    nntrainer::avx2::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(n_zero_stats.total_realloc_count, 0u);
}
#endif

/// FP16 sgemv path: exercises the x86 AVX2 hgemv kernel.
/// Reference is a local FP32 GEMV loop to avoid using the optimized backend
/// as the oracle for this backend test.
static void run_sgemv_fp16_hgemv_test(unsigned int M, unsigned int N,
                                      bool TransA = false, float alpha = 1.0F,
                                      float beta = 0.0F,
                                      unsigned int lda_extra = 0,
                                      unsigned int incX = 1,
                                      unsigned int incY = 1) {
  nntrainer::init_backend();

  const unsigned int lda = std::max(1u, N + lda_extra);
  const unsigned int lenX = TransA ? M : N;
  const unsigned int lenY = TransA ? N : M;

  const std::size_t a_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(M) * lda);
  const std::size_t x_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenX) * std::max(incX, 1u));
  const std::size_t y_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenY) * std::max(incY, 1u));

  auto A_fp16 = generate_random_vector<_FP16>(a_size, -0.25F, 0.25F);
  auto X_fp16 = generate_random_vector<_FP16>(x_size, -0.25F, 0.25F);
  auto Y_fp16 = generate_random_vector<_FP16>(y_size, -0.25F, 0.25F);
  auto Y_before = Y_fp16;

  std::vector<float> Y_fp32_ref(y_size);
  for (std::size_t i = 0; i < y_size; ++i) {
    Y_fp32_ref[i] = static_cast<float>(Y_before[i]);
  }

  if (M != 0 && N != 0) {
    if (!TransA) {
      for (unsigned int i = 0; i < M; ++i) {
        float acc = 0.0F;
        for (unsigned int j = 0; j < N; ++j) {
          acc += static_cast<float>(A_fp16[i * lda + j]) *
                 static_cast<float>(X_fp16[j * incX]);
        }
        const std::size_t y_idx = static_cast<std::size_t>(i) * incY;
        Y_fp32_ref[y_idx] = alpha * acc + beta * Y_fp32_ref[y_idx];
      }
    } else {
      for (unsigned int j = 0; j < N; ++j) {
        float acc = 0.0F;
        for (unsigned int i = 0; i < M; ++i) {
          acc += static_cast<float>(A_fp16[i * lda + j]) *
                 static_cast<float>(X_fp16[i * incX]);
        }
        const std::size_t y_idx = static_cast<std::size_t>(j) * incY;
        Y_fp32_ref[y_idx] = alpha * acc + beta * Y_fp32_ref[y_idx];
      }
    }
  }

  // System under test: FP16 sgemv, routed to x86::hgemv by the x86 backend
  // dispatcher for row-major inputs.
  nntrainer::sgemv(0, TransA, M, N, alpha, A_fp16.data(), lda, X_fp16.data(),
                   incX, beta, Y_fp16.data(), incY);

  if (M == 0 || N == 0) {
    for (std::size_t i = 0; i < y_size; ++i) {
      EXPECT_EQ(static_cast<float>(Y_fp16[i]), static_cast<float>(Y_before[i]))
        << "zero-dimension GEMV touched Y at i=" << i << " M=" << M
        << " N=" << N << " TransA=" << TransA;
    }
    return;
  }

  for (unsigned int k = 0; k < lenY; ++k) {
    const std::size_t y_idx = static_cast<std::size_t>(k) * incY;
    const float got = static_cast<float>(Y_fp16[y_idx]);
    const float ref = Y_fp32_ref[y_idx];
    const float abs_diff = std::abs(got - ref);
    const float rel_diff = abs_diff / std::max(1.0F, std::abs(ref));
    EXPECT_TRUE(abs_diff <= 1e-2F || rel_diff <= 1e-2F)
      << "mismatch at k=" << k << " M=" << M << " N=" << N
      << " TransA=" << TransA << " alpha=" << alpha << " beta=" << beta
      << " lda=" << lda << " incX=" << incX << " incY=" << incY
      << " got=" << got << " ref=" << ref << " abs_diff=" << abs_diff
      << " rel_diff=" << rel_diff;
  }

  // Y gap slots (only present when incY > 1) must be untouched.
  if (incY > 1) {
    for (unsigned int k = 0; k < lenY; ++k) {
      const std::size_t base = static_cast<std::size_t>(k) * incY;
      for (unsigned int g = 1; g < incY; ++g) {
        const std::size_t gap_idx = base + g;
        if (gap_idx >= y_size) {
          break;
        }
        EXPECT_EQ(static_cast<float>(Y_fp16[gap_idx]),
                  static_cast<float>(Y_before[gap_idx]))
          << "Y gap was modified at idx=" << gap_idx << " k=" << k << " g=" << g
          << " incY=" << incY;
      }
    }
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_aligned_8x64) {
  // N multiple of 16: 16-element main loop only, no tails.
  run_sgemv_fp16_hgemv_test(8, 64);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_aligned_16x128) {
  run_sgemv_fp16_hgemv_test(16, 128);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_n8_tail_5x24) {
  // N=24 = 16 + 8: exercises the 8-element tail after the 16-element loop.
  run_sgemv_fp16_hgemv_test(5, 24);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_scalar_tail_7x31) {
  // N=31 = 16 + 8 + 7: exercises 16-loop, 8-tail, then scalar tail of 7.
  run_sgemv_fp16_hgemv_test(7, 31);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemv_fp16_noTrans_scalar_tail_only_5x9) {
  // N=9: skips 16-loop and 8-tail, scalar tail of 9.
  run_sgemv_fp16_hgemv_test(5, 9);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_transA_aligned_16x64) {
  run_sgemv_fp16_hgemv_test(16, 64, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_transA_unaligned_13x33) {
  // M=13 (AXPY row sweep tail), N=33 (8-element tail + scalar tail in
  // the FP32-scratch axpy update).
  run_sgemv_fp16_hgemv_test(13, 33, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_alpha_beta_13x33) {
  run_sgemv_fp16_hgemv_test(13, 33, false, -0.75F, 0.25F);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_transA_alpha_beta_13x33) {
  run_sgemv_fp16_hgemv_test(13, 33, true, 0.5F, -0.125F);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_lda_noTrans_7x17) {
  // lda = N + 7: row-padded A.
  run_sgemv_fp16_hgemv_test(7, 17, false, 1.0F, 0.0F, 7);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_lda_transA_7x17) {
  run_sgemv_fp16_hgemv_test(7, 17, true, 1.0F, 0.0F, 7);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_incX_noTrans_7x17) {
  // incX=2 forces the non-contiguous X path.
  run_sgemv_fp16_hgemv_test(7, 17, false, 1.0F, 0.0F, 0, 2, 1);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_incY_noTrans_7x17) {
  // incY=3 exercises Y-gap preservation in the writeback path.
  run_sgemv_fp16_hgemv_test(7, 17, false, 1.0F, 0.0F, 0, 1, 3);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemv_fp16_strided_lda_incX_incY_transA_7x17) {
  // All three strides at once on the TransA path (non-contiguous Y32 init,
  // non-contiguous incX read, non-contiguous incY writeback).
  run_sgemv_fp16_hgemv_test(7, 17, true, 0.5F, -0.25F, 3, 2, 4);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_alpha_beta_boundary_cases) {
  struct Case {
    float alpha;
    float beta;
  };

  const std::vector<Case> cases = {
    {0.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {2.5F, -1.0F}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("alpha=" + std::to_string(tc.alpha) +
                 " beta=" + std::to_string(tc.beta));
    run_sgemv_fp16_hgemv_test(13, 33, false, tc.alpha, tc.beta);
    run_sgemv_fp16_hgemv_test(13, 33, true, tc.alpha, tc.beta);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_zero_dimension_cases) {
  run_sgemv_fp16_hgemv_test(0, 17, false, 1.0F, 0.0F);
  run_sgemv_fp16_hgemv_test(7, 0, false, 1.0F, 0.0F);
  run_sgemv_fp16_hgemv_test(0, 17, true, 1.0F, 0.0F);
  run_sgemv_fp16_hgemv_test(7, 0, true, 1.0F, 0.0F);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_single_row_or_col) {
  run_sgemv_fp16_hgemv_test(1, 33, false);
  run_sgemv_fp16_hgemv_test(33, 1, false);
  run_sgemv_fp16_hgemv_test(1, 33, true);
  run_sgemv_fp16_hgemv_test(33, 1, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_beta_zero_does_not_read_Y) {
  // BLAS rule: when beta == 0, Y must not be read. Seed Y with NaN; if the
  // kernel reads it, NaN propagates into the result.
  nntrainer::init_backend();

  const _FP16 nan_v = static_cast<_FP16>(std::nan(""));

  auto check_path = [&](bool TransA) {
    const unsigned int M = 7;
    const unsigned int N = 17;
    const unsigned int lenY = TransA ? N : M;

    auto A = generate_random_vector<_FP16>(M * N, -0.25F, 0.25F);
    auto X = generate_random_vector<_FP16>(TransA ? M : N, -0.25F, 0.25F);
    std::vector<_FP16> Y(lenY, nan_v);

    nntrainer::sgemv(0, TransA, M, N, 1.0F, A.data(), N, X.data(), 1, 0.0F,
                     Y.data(), 1);

    for (unsigned int k = 0; k < lenY; ++k) {
      const float yv = static_cast<float>(Y[k]);
      EXPECT_FALSE(std::isnan(yv)) << "beta=0 path read NaN-seeded Y at k=" << k
                                   << " TransA=" << TransA << " value=" << yv;
    }
  };

  check_path(false);
  check_path(true);
}

/// Mixed-precision GEMM path (shgemm: A=FP32,B=FP16 / hsgemm: A=FP16,B=FP32),
/// FP32 output. Reference accumulates in double so the only approximation
/// under test is the kernel's FP32 blocked summation; both sides read the
/// identical generated operands, so operand rounding is not a discrepancy.
template <typename AType, typename BType>
static void run_mixed_sgemm_test(unsigned int M, unsigned int N, unsigned int K,
                                 bool TransA = false, bool TransB = false,
                                 float alpha = 1.0F, float beta = 0.0F,
                                 unsigned int lda_extra = 0,
                                 unsigned int ldb_extra = 0,
                                 unsigned int ldc_extra = 0) {
  static_assert(std::is_same_v<AType, float> != std::is_same_v<BType, float>,
                "exactly one operand must be FP32 for mixed-precision GEMM");
  nntrainer::init_backend();

  const unsigned int lda = std::max(1u, (TransA ? M : K) + lda_extra);
  const unsigned int ldb = std::max(1u, (TransB ? K : N) + ldb_extra);
  const unsigned int ldc = std::max(1u, N + ldc_extra);
  const unsigned int a_rows = TransA ? K : M;
  const unsigned int b_rows = TransB ? N : K;
  const std::size_t a_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(a_rows) * lda);
  const std::size_t b_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(b_rows) * ldb);
  const std::size_t c_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(M) * ldc);

  auto A = generate_random_vector<AType>(a_size, -0.25F, 0.25F);
  auto B = generate_random_vector<BType>(b_size, -0.25F, 0.25F);
  auto C = generate_random_vector<float>(c_size, -0.25F, 0.25F);
  auto C_before = C;

  std::vector<float> C_ref(c_size);
  for (std::size_t i = 0; i < c_size; ++i) {
    C_ref[i] = C_before[i];
  }

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (unsigned int k = 0; k < K; ++k) {
        const double a = static_cast<double>(
          static_cast<float>(TransA ? A[k * lda + m] : A[m * lda + k]));
        const double b = static_cast<double>(
          static_cast<float>(TransB ? B[n * ldb + k] : B[k * ldb + n]));
        acc += a * b;
      }
      const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
      C_ref[idx] = static_cast<float>(static_cast<double>(alpha) * acc +
                                      static_cast<double>(beta) * C_ref[idx]);
    }
  }

  if constexpr (std::is_same_v<AType, float>) {
    nntrainer::shgemm(0, TransA, TransB, M, N, K, alpha, A.data(), lda,
                      B.data(), ldb, beta, C.data(), ldc);
  } else {
    nntrainer::hsgemm(0, TransA, TransB, M, N, K, alpha, A.data(), lda,
                      B.data(), ldb, beta, C.data(), ldc);
  }

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
      const float got = C[idx];
      const float ref = C_ref[idx];
      const float abs_diff = std::abs(got - ref);
      const float rel_diff = abs_diff / std::max(1.0F, std::abs(ref));
      EXPECT_TRUE(abs_diff <= 1e-3F || rel_diff <= 1e-3F)
        << "mismatch at m=" << m << " n=" << n << " M=" << M << " N=" << N
        << " K=" << K << " TransA=" << TransA << " TransB=" << TransB
        << " alpha=" << alpha << " beta=" << beta << " got=" << got
        << " ref=" << ref << " abs_diff=" << abs_diff
        << " rel_diff=" << rel_diff;
    }
    for (unsigned int n = N; n < ldc; ++n) {
      const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
      EXPECT_FLOAT_EQ(C[idx], C_before[idx])
        << "C padding was modified at m=" << m << " n=" << n << " ldc=" << ldc;
    }
  }
}

TEST(nntrainer_cpu_backend_standalone, shgemm_fp32xfp16_noTrans_64x64x64) {
  run_mixed_sgemm_test<float, _FP16>(64, 64, 64);
}
TEST(nntrainer_cpu_backend_standalone, shgemm_fp32xfp16_noTrans_256x512x128) {
  run_mixed_sgemm_test<float, _FP16>(256, 512, 128);
}
TEST(nntrainer_cpu_backend_standalone, shgemm_fp32xfp16_unaligned_13x33x65) {
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65);
}
TEST(nntrainer_cpu_backend_standalone,
     shgemm_fp32xfp16_all_transposes_13x33x65) {
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65, true, false);
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65, false, true);
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65, true, true);
}
TEST(nntrainer_cpu_backend_standalone, shgemm_fp32xfp16_alpha_beta_padded) {
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65, false, false, 0.5F, 0.25F, 4,
                                     7, 5);
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65, true, true, -1.25F, 0.5F, 5, 3,
                                     6);
}
TEST(nntrainer_cpu_backend_standalone, hsgemm_fp16xfp32_noTrans_64x64x64) {
  run_mixed_sgemm_test<_FP16, float>(64, 64, 64);
}
TEST(nntrainer_cpu_backend_standalone, hsgemm_fp16xfp32_noTrans_256x512x128) {
  run_mixed_sgemm_test<_FP16, float>(256, 512, 128);
}
TEST(nntrainer_cpu_backend_standalone, hsgemm_fp16xfp32_unaligned_13x33x65) {
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65);
}
TEST(nntrainer_cpu_backend_standalone,
     hsgemm_fp16xfp32_all_transposes_13x33x65) {
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65, true, false);
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65, false, true);
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65, true, true);
}
TEST(nntrainer_cpu_backend_standalone, hsgemm_fp16xfp32_alpha_beta_padded) {
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65, false, false, 0.5F, 0.25F, 4,
                                     7, 5);
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65, true, true, -1.25F, 0.5F, 5, 3,
                                     6);
}

/// Mixed-precision GEMV path (shgemv: A=FP32,X=FP16 / hsgemv: A=FP16,X=FP32),
/// FP32 output. Reference accumulates in double; see run_mixed_sgemm_test.
template <typename MatT, typename VecT>
static void run_mixed_sgemv_test(unsigned int M, unsigned int N,
                                 bool TransA = false, float alpha = 1.0F,
                                 float beta = 0.0F, unsigned int lda_extra = 0,
                                 unsigned int incX = 1, unsigned int incY = 1) {
  static_assert(std::is_same_v<MatT, float> != std::is_same_v<VecT, float>,
                "exactly one operand must be FP32 for mixed-precision GEMV");
  nntrainer::init_backend();

  const unsigned int lda = std::max(1u, N + lda_extra);
  const unsigned int lenX = TransA ? M : N;
  const unsigned int lenY = TransA ? N : M;

  const std::size_t a_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(M) * lda);
  const std::size_t x_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenX) * std::max(incX, 1u));
  const std::size_t y_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenY) * std::max(incY, 1u));

  auto A = generate_random_vector<MatT>(a_size, -0.25F, 0.25F);
  auto X = generate_random_vector<VecT>(x_size, -0.25F, 0.25F);
  auto Y = generate_random_vector<float>(y_size, -0.25F, 0.25F);
  auto Y_before = Y;

  std::vector<float> Y_ref(y_size);
  for (std::size_t i = 0; i < y_size; ++i) {
    Y_ref[i] = Y_before[i];
  }

  if (!TransA) {
    for (unsigned int i = 0; i < M; ++i) {
      double acc = 0.0;
      for (unsigned int j = 0; j < N; ++j) {
        acc += static_cast<double>(static_cast<float>(A[i * lda + j])) *
               static_cast<double>(static_cast<float>(X[j * incX]));
      }
      const std::size_t y_idx = static_cast<std::size_t>(i) * incY;
      Y_ref[y_idx] =
        static_cast<float>(static_cast<double>(alpha) * acc +
                           static_cast<double>(beta) * Y_ref[y_idx]);
    }
  } else {
    for (unsigned int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (unsigned int i = 0; i < M; ++i) {
        acc += static_cast<double>(static_cast<float>(A[i * lda + j])) *
               static_cast<double>(static_cast<float>(X[i * incX]));
      }
      const std::size_t y_idx = static_cast<std::size_t>(j) * incY;
      Y_ref[y_idx] =
        static_cast<float>(static_cast<double>(alpha) * acc +
                           static_cast<double>(beta) * Y_ref[y_idx]);
    }
  }

  if constexpr (std::is_same_v<MatT, float>) {
    nntrainer::shgemv(0, TransA, M, N, alpha, A.data(), lda, X.data(), incX,
                      beta, Y.data(), incY);
  } else {
    nntrainer::hsgemv(0, TransA, M, N, alpha, A.data(), lda, X.data(), incX,
                      beta, Y.data(), incY);
  }

  for (unsigned int k = 0; k < lenY; ++k) {
    const std::size_t y_idx = static_cast<std::size_t>(k) * incY;
    const float got = Y[y_idx];
    const float ref = Y_ref[y_idx];
    const float abs_diff = std::abs(got - ref);
    const float rel_diff = abs_diff / std::max(1.0F, std::abs(ref));
    EXPECT_TRUE(abs_diff <= 1e-3F || rel_diff <= 1e-3F)
      << "mismatch at k=" << k << " M=" << M << " N=" << N
      << " TransA=" << TransA << " alpha=" << alpha << " beta=" << beta
      << " incX=" << incX << " incY=" << incY << " got=" << got
      << " ref=" << ref << " abs_diff=" << abs_diff << " rel_diff=" << rel_diff;
  }
}

TEST(nntrainer_cpu_backend_standalone, shgemv_fp32xfp16_noTrans_16x128) {
  run_mixed_sgemv_test<float, _FP16>(16, 128);
}
TEST(nntrainer_cpu_backend_standalone,
     shgemv_fp32xfp16_transA_unaligned_13x33) {
  run_mixed_sgemv_test<float, _FP16>(13, 33, true);
}
TEST(nntrainer_cpu_backend_standalone, shgemv_fp32xfp16_alpha_beta_13x33) {
  run_mixed_sgemv_test<float, _FP16>(13, 33, false, 0.5F, 0.25F);
  run_mixed_sgemv_test<float, _FP16>(13, 33, true, 0.5F, 0.25F);
}
TEST(nntrainer_cpu_backend_standalone, hsgemv_fp16xfp32_noTrans_16x128) {
  run_mixed_sgemv_test<_FP16, float>(16, 128);
}
TEST(nntrainer_cpu_backend_standalone,
     hsgemv_fp16xfp32_transA_unaligned_13x33) {
  run_mixed_sgemv_test<_FP16, float>(13, 33, true);
}
TEST(nntrainer_cpu_backend_standalone,
     hsgemv_fp16xfp32_strided_incX_incY_7x17) {
  run_mixed_sgemv_test<_FP16, float>(7, 17, false, 1.0F, 0.0F, 3, 2, 2);
  run_mixed_sgemv_test<float, _FP16>(7, 17, true, 0.75F, 0.5F, 0, 2, 2);
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
