// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_cpu_backend.cpp
 * @date	03 April 2025
 * @brief	This is unittest for cpu_backend standalone
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "int4_utils.h"
#include "nntrainer_test_util.h"
#include "q4_0_utils.h"
#include <cpu_backend.h>
#include <fallback_internal.h>
#include <fp16.h>
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

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

template <typename T, bool random_init = false>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  auto init_val = random_init ? rd() : 42;
  std::mt19937 gen(init_val);
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

#if defined(ENABLE_FP16) || defined(__AVX2__)
static inline std::vector<uint16_t>
convert_vector_f32_to_f16_as_uint16(std::vector<float> f32_vec) {
  std::vector<uint16_t> vec(f32_vec.size());
  for (size_t i = 0; i < f32_vec.size(); i++) {
    vec[i] = nntrainer::compute_fp32_to_fp16(f32_vec[i]);
  }
  return vec;
}

static inline std::vector<float>
convert_vector_f16_as_uint16_to_f32(std::vector<uint16_t> uint16_vec) {
  std::vector<float> vec(uint16_vec.size());
  for (size_t i = 0; i < uint16_vec.size(); i++) {
    vec[i] = nntrainer::compute_fp16_to_fp32(uint16_vec[i]);
  }
  return vec;
}
#endif

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  double err_sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(src[i * N + j] - src2[i * N + j]));
      err_sum += std::abs(src[i * N + j] - src2[i * N + j]);
    }
  }
  // std::cout << "err_sum : " << err_sum << std::endl;
  return max_diff;
}

#define QK4_0 32
#define Q4_0 32
/**
 * @brief q4_0 block
 *
 */
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0_testonly;
/**
 * @brief q4_K block
 *
 */
typedef struct {
  union {
    struct {
      int16_t d;    // super-block scale for quantized scales
      int16_t dmin; // super-block scale for quantized mins
    };
    uint32_t dm;
  };
  uint8_t scales[12];  // scales and mins, quantized with 6 bits
  uint8_t qs[256 / 2]; // 4--bit quants
} block_q4_K_testonly;
/**
 * @brief q8_K block
 *
 */
typedef struct {
  float d;                 // delta
  int8_t qs[256];          // quants
  int16_t bsums[256 / 16]; // sum of quants in groups of 16
} block_q8_K_testonly;
/**
 * @brief q4_Kx8 block
 *
 */
struct block_q4_Kx8_testonly {
  int16_t d[8];       // super-block scale for quantized scales
  int16_t dmin[8];    // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};

#define QK_K 256
typedef struct {
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  uint16_t d;               // super-block scale
} block_q6_K_testonly;

/**
 * @brief Elementwise-addition unittest : Vanilla example for formulating a TC
 * in unittest_nntrainer_cpu_backend.cpp
 *
 */
TEST(nntrainer_cpu_backend_standalone, ele_add) {
  const unsigned int TEST_SIZE = 100;
  float alpha = 1.F;
  float beta = 0.F;
  unsigned int i_stride = 1;
  unsigned int o_stride = 1;

  std::vector<float> lhs = generate_random_vector<float>(TEST_SIZE);
  std::vector<float> rhs = generate_random_vector<float>(TEST_SIZE);
  std::vector<float> dst(TEST_SIZE);

  const float *lhs_ptr = (const float *)lhs.data();
  const float *rhs_ptr = (const float *)rhs.data();
  float *dst_ptr = (float *)dst.data();

  nntrainer::ele_add(TEST_SIZE, lhs_ptr, rhs_ptr, dst_ptr, alpha, beta,
                     i_stride, o_stride);

  for (unsigned int i = 0; i < TEST_SIZE; ++i) {
    EXPECT_EQ(dst[i], lhs[i] + rhs[i]);
  }
}

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

TEST(nntrainer_cpu_backend_standalone, q4_K_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q4_K(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q4_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);
}

TEST(nntrainer_cpu_backend_standalone, q6_K_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * ne0 / q6_k_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q6_K(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q6_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);
}

TEST(nntrainer_cpu_backend_standalone, q4_0_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q4_0_block_size = QK4_0;
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t num_blocks = (K * N) / q4_0_block_size;
  size_t data_size = num_blocks * q4_0_type_size;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q4_0(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q4_0(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);
}

float test_gemm_q4_0(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // needed to initialize f16 tables

  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t q4_0_block_size = 32;
  int64_t q4_0_num_blocks = (K * N) / q4_0_block_size;
  size_t q4_0_data_size = q4_0_type_size * N / q4_0_block_size;
  q4_0_data_size *= K;
  std::vector<char> q4_0_offline_qWeight = std::vector<char>(q4_0_data_size);

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  char *q4_0_offline_qWeight_ptr = (char *)q4_0_offline_qWeight.data();
  nntrainer::quantize_q4_0(weights, (void *)q4_0_offline_qWeight_ptr, N, K,
                           nullptr);

  // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the
  // model weights. It's a one-time operation)
  std::vector<char> q4_0_repacked_qWeight = std::vector<char>(q4_0_data_size);
  nntrainer::repack_q4_0(q4_0_repacked_qWeight.data(), q4_0_offline_qWeight_ptr,
                         q4_0_data_size, N, K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_0: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

float test_gemm_q4_K(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * N / q4_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  nntrainer::quantize_q4_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the
  // model weights. It's a one-time operation)
  std::vector<char> repacked_qWeight = std::vector<char>(data_size);
  nntrainer::repack_q4_K(repacked_qWeight.data(), offline_qWeight_ptr,
                         data_size, N, K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
                       N, dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

float test_gemm_q6_K(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * N / q6_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  nntrainer::quantize_q6_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  // Step2. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q6_K(M, N, K, activations, K, (void *)offline_qWeight_ptr, N,
                       dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q6_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

static void run_quant_test(const uint32_t M, const uint32_t K, const uint32_t N,
                           float &q4_0_mse, float &q4_k_mse, float &q6_k_mse,
                           bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "[INFO] Quantization Test (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
  }
  ///@note A(M, K) * W.T(N, K) = (M, N)
  ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

  ///@note q4_K GEMM is a Row-Major, transB GEMM
  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> ref_dst(M * N);

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
  q4_0_mse =
    test_gemm_q4_0(M, K, N, weight.data(), activation.data(), ref_dst, print);
  q4_k_mse =
    test_gemm_q4_K(M, K, N, weight.data(), activation.data(), ref_dst, print);
  q6_k_mse =
    test_gemm_q6_K(M, K, N, weight.data(), activation.data(), ref_dst, print);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_457x512x512) {
  const unsigned int M = 457;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_458x512x512) {
  const unsigned int M = 458;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_459x512x512) {
  const unsigned int M = 459;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_512x512x512) {
  const unsigned int M = 512;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x512x512) {
  const unsigned int M = 1;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

static void run_vec_dot_test(const uint32_t K, bool print = false) {
  const int TEST_CNT = 20;
  nanoseconds ref_time = (nanoseconds)0;
  nanoseconds q6_k_time = (nanoseconds)0;
  float ref_result, result;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> activation = generate_random_vector<float, false>(K);
    std::vector<float> weight = generate_random_vector<float, false>(K);

    {
      // GROUND TRUTH sdot for reference
      auto t1 = high_resolution_clock::now();
      ref_result = nntrainer::sdot(K, weight.data(), 1, activation.data(), 1);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_time += dt;
      }
    }

    // Quantization of weights
    int64_t num_blocks = K / 256;
    size_t q6_k_data_size = num_blocks * sizeof(block_q6_K_testonly);
    std::vector<char> q6_K_weight = std::vector<char>(q6_k_data_size);
    nntrainer::quantize_row_q6_K(weight.data(), q6_K_weight.data(), K);

    // Quantization of activations
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    int q8_K_activation_size = sizeof(block_q8_K_testonly) * blocks_per_row;
    std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
    nntrainer::quantize_row_q8_K(activation.data(), v_q8_activation.data(), K);

    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      result =
        nntrainer::dot_q6_K_q8_K(K, q6_K_weight.data(), v_q8_activation.data());
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        q6_k_time += dt;
      }
    }
    EXPECT_NEAR(result, ref_result, 0.25 * K / 256);
  }
  if (print) {
    std::cout << "[INFO] dot_q6_K_q8_K: TEST CNT: " << TEST_CNT << ", K: " << K
              << ", Average ref_time: " << ref_time.count() / TEST_CNT
              << " ns, Average q6_k_time: " << q6_k_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_1024) {
  const uint32_t K = 1024;
  run_vec_dot_test(K);
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_2560) {
  const uint32_t K = 2560;
  run_vec_dot_test(K);
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_10240) {
  const uint32_t K = 10240;
  run_vec_dot_test(K);
}

static void run_ele_mul_test(const unsigned int N, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride,
                             bool print = false) {
  // Z = X ⊙ alpha * Y + beta * Z
  const int TEST_CNT = 20;
  nanoseconds ref_mul_time = (nanoseconds)0;
  nanoseconds mul_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> X =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Y = generate_random_vector<float, false>(
      std::max<size_t>(1, (size_t)N * i_stride));
    std::vector<float> Z =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Z_ref = Z;
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::__fallback_ele_mul(N, X.data(), Y.data(), Z_ref.data(), alpha,
                                    beta, i_stride, o_stride);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_mul_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::ele_mul(N, X.data(), Y.data(), Z.data(), alpha, beta, i_stride,
                         o_stride);
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        mul_time += dt;
      }
    }

    auto mean_squared_error = compute_mse(1, N, Z_ref, Z, false);
    ASSERT_LE(mean_squared_error, 0.00001f);
  }

  if (print) {
    std::cout << "[INFO] ele_mul: TEST CNT: " << TEST_CNT << ", N: " << N
              << ", i_stride: " << i_stride
              << ", Average ref_time: " << ref_mul_time.count() / TEST_CNT
              << " ns, Average mul_time: " << mul_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, ele_mul_3072_istr_0) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 0;
  const unsigned int o_stride = 1;
  run_ele_mul_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_mul_3072_istr_1) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 1;
  const unsigned int o_stride = 1;
  run_ele_mul_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_mul_3072_istr_16_ostr_16) {
  const unsigned int N = 3072;
  const float alpha = 3.f;
  const float beta = 2.f;
  const unsigned int i_stride = 16;
  const unsigned int o_stride = 16;
  run_ele_mul_test(N, alpha, beta, i_stride, o_stride);
}

static void run_ele_add_test(const unsigned int N, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride,
                             bool print = false) {
  // Z = X ⊙ alpha * Y + beta * Z
  const int TEST_CNT = 20;
  nanoseconds ref_add_time = (nanoseconds)0;
  nanoseconds add_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> X =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Y = generate_random_vector<float, false>(
      std::max<size_t>(1, (size_t)N * i_stride));
    std::vector<float> Z =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Z_ref = Z;
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::__fallback_ele_add(N, X.data(), Y.data(), Z_ref.data(), alpha,
                                    beta, i_stride, o_stride);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_add_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::ele_add(N, X.data(), Y.data(), Z.data(), alpha, beta, i_stride,
                         o_stride);
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        add_time += dt;
      }
    }
    auto mean_squared_error = compute_mse(1, N, Z_ref, Z, false);
    ASSERT_LE(mean_squared_error, 0.00001f);
  }
  if (print) {
    std::cout << "[INFO] ele_add: TEST CNT: " << TEST_CNT << ", N: " << N
              << ", i_stride: " << i_stride
              << ", Average ref_time: " << ref_add_time.count() / TEST_CNT
              << " ns, Average add_time: " << add_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, ele_add_3072_istr_0) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 0;
  const unsigned int o_stride = 1;
  run_ele_add_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_add_3072_istr_1) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 1;
  const unsigned int o_stride = 1;
  run_ele_add_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_add_3072_istr_16_ostrid_16) {
  const unsigned int N = 3072;
  const float alpha = 3.f;
  const float beta = 2.f;
  const unsigned int i_stride = 16;
  const unsigned int o_stride = 16;
  run_ele_add_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, softmax_row_inplace) {
  size_t start_row = 0;
  size_t end_row = 3;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  std::vector<float> qk_out = {
    -2.509198f, 5.930860f,  9.014286f,  -6.331305f, 4.639878f,  5.593820f,
    1.973170f,  1.937003f,  -6.879627f, -1.083345f, -6.880110f, -8.000502f,
    -8.838327f, -0.815022f, 7.323523f,  -3.325828f, 2.022300f,  -7.142663f,
    4.161452f,  3.017770f,  -9.588310f, -8.871769f, 9.398197f,  4.439975f,
    6.648853f,  8.771055f,  -5.753218f, -9.984425f, -6.363501f, 9.844232f};
  std::vector<float> ref_out = {
    0.986697f, 0.999999f, 0.405184f, 0.000021f, 0.043301f, 0.040031f,
    0.487615f, 0.999879f, 0.000016f, 0.000018f, 0.012472f, 0.000001f,
    0.000000f, 0.005194f, 0.633859f, 0.000005f, 0.512170f, 0.000114f,
    0.999957f, 0.001083f, 0.000831f, 0.000000f, 0.594816f, 0.994785f,
    0.322840f, 0.959963f, 0.000215f, 0.000007f, 0.000027f, 0.998899f};

  nntrainer::softmax_row_inplace(qk_out.data(), start_row, end_row, num_heads);

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, softmax_row) {
  size_t start_row = 0;
  size_t end_row = 3;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  std::vector<float> qk_out = {
    -2.509198f, 5.930860f,  9.014286f,  -6.331305f, 4.639878f,  5.593820f,
    1.973170f,  1.937003f,  -6.879627f, -1.083345f, -6.880110f, -8.000502f,
    -8.838327f, -0.815022f, 7.323523f,  -3.325828f, 2.022300f,  -7.142663f,
    4.161452f,  3.017770f,  -9.588310f, -8.871769f, 9.398197f,  4.439975f,
    6.648853f,  8.771055f,  -5.753218f, -9.984425f, -6.363501f, 9.844232f};
  std::vector<float> ref_out = {
    0.986697f, 0.999999f, 0.405184f, 0.000021f, 0.043301f, 0.040031f,
    0.487615f, 0.999879f, 0.000016f, 0.000018f, 0.012472f, 0.000001f,
    0.000000f, 0.005194f, 0.633859f, 0.000005f, 0.512170f, 0.000114f,
    0.999957f, 0.001083f, 0.000831f, 0.000000f, 0.594816f, 0.994785f,
    0.322840f, 0.959963f, 0.000215f, 0.000007f, 0.000027f, 0.998899f};

  nntrainer::softmax_row(qk_out.data(), start_row, end_row, num_heads);

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0001f);
  }
}

#if defined(ENABLE_FP16) || defined(__AVX2__)
TEST(nntrainer_cpu_backend_standalone, compute_kcaches) {
  int num_rows = 1;
  int N = 2;
  int head_dim = 10;
  int group_size = 4;
  int tile_size = 16;
  size_t in_size = N * group_size * head_dim;
  size_t kcache_size = num_rows * N * head_dim;
  size_t output_size = num_rows * N * group_size;
  std::vector<float> in = {
    -2.509198f, 5.930860f,  9.014286f,  -6.331305f, 4.639878f,  5.593820f,
    1.973170f,  1.937003f,  -6.879627f, -1.083345f, -6.880110f, -8.000502f,
    -8.838327f, -0.815022f, 7.323523f,  -3.325828f, 2.022300f,  -7.142663f,
    4.161452f,  3.017770f,  -9.588310f, -8.871769f, 9.398197f,  4.439975f,
    6.648853f,  8.771055f,  -5.753218f, -9.984425f, -6.363501f, 9.844232f,
    -6.331910f, 2.349631f,  -3.915155f, 2.233063f,  0.495129f,  -9.858674f,
    -1.361099f, -9.538752f, -4.175417f, 0.495493f,  2.237058f,  -2.002780f,
    -7.210123f, -9.066687f, -4.157107f, 9.475111f,  -2.672763f, -5.344573f,
    -0.878600f, -8.187871f, 5.703520f,  2.367721f,  -6.006524f, -2.350760f,
    0.284688f,  9.664618f,  1.848291f,  -0.664742f, -9.070992f, 7.198808f,
    2.150897f,  3.606151f,  -6.589518f, -0.990015f, -8.698968f, -9.734701f,
    8.977711f,  8.844034f,  9.312640f,  1.265764f,  6.167947f,  -2.291670f,
    -3.907725f, -9.680675f, -8.046557f, -5.382123f, 3.684660f,  -5.179491f,
    -1.196950f, 3.665271f};
  std::vector<uint16_t> kcache = {3751, 7967, 9507, 1842, 7322, 7799, 5990,
                                  5972, 1568, 4463, 1568, 1008, 590,  4597,
                                  8663, 3343, 6015, 1437, 7083, 6512};
  std::vector<float> ref_out = {0.089252f,  -0.072949f, 0.058948f,  -0.045583f,
                                -0.025812f, -0.002068f, -0.014971f, -0.028027f};
  std::vector<float> output(output_size);

  nntrainer::compute_kcaches<uint16_t>(in.data(), kcache.data(), output.data(),
                                       num_rows, N, head_dim, group_size,
                                       tile_size);

  for (size_t i = 0; i < output_size; i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_out_null) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;
  bool only_convert_to_fp16 = false;

  std::vector<float> inout = {
    -0.250920f, 0.593086f,  0.901429f,  -0.633130f, 0.463988f,  0.559382f,
    0.197317f,  0.193700f,  -0.687963f, -0.108334f, -0.688011f, -0.800050f,
    -0.883833f, -0.081502f, 0.732352f,  -0.332583f, 0.202230f,  -0.714266f,
    0.416145f,  0.301777f,  -0.958831f, -0.887177f, 0.939820f,  0.443998f,
    0.664885f,  0.877105f,  -0.575322f, -0.998442f, -0.636350f, 0.984423f,
    -0.633191f, 0.234963f,  -0.391515f, 0.223306f,  0.049513f,  -0.985867f,
    -0.136110f, -0.953875f, -0.417542f, 0.049549f};
  std::vector<float> cos_ = {-0.508061f, 0.998221f,  -0.733869f, 0.164825f,
                             0.297530f,  -0.591437f, -0.165907f, -0.999895f,
                             -0.163179f, 0.249787};
  std::vector<float> sin_ = {0.809128f,  -0.390829f, -0.112627f, 0.377992f,
                             -0.994569f, -0.641599f, -0.927266f, -0.401338f,
                             0.244335f,  -0.414953f};
  std::vector<float> ref_out = {
    0.684172f,  0.279348f,  -0.761074f, -0.073549f, 0.866425f,  -0.544224f,
    0.154785f,  -0.480342f, 0.010582f,  0.098163f,  0.146526f,  -1.030422f,
    0.547092f,  -0.252752f, -0.243571f, -0.162197f, -0.216517f, 0.636452f,
    -0.236000f, 0.120334f,  0.999478f,  -0.793769f, -0.733800f, -0.011226f,
    0.247067f,  -1.151284f, -0.030760f, 0.615511f,  0.205859f,  0.266457f,
    -0.454117f, 0.581280f,  0.181472f,  0.204634f,  -0.646542f, 0.020328f,
    0.556058f,  1.354487f,  -0.087349f, -0.396113f};

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(), nullptr,
                                      cos_.data(), sin_.data(),
                                      only_convert_to_fp16);

  for (size_t i = 0; i < inout.size(); i++) {
    EXPECT_NEAR(ref_out[i], inout[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_out_uint16) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;
  bool only_convert_to_fp16 = false;

  std::vector<float> inout = {
    -0.250920f, 0.593086f,  0.901429f,  -0.633130f, 0.463988f,  0.559382f,
    0.197317f,  0.193700f,  -0.687963f, -0.108334f, -0.688011f, -0.800050f,
    -0.883833f, -0.081502f, 0.732352f,  -0.332583f, 0.202230f,  -0.714266f,
    0.416145f,  0.301777f,  -0.958831f, -0.887177f, 0.939820f,  0.443998f,
    0.664885f,  0.877105f,  -0.575322f, -0.998442f, -0.636350f, 0.984423f,
    -0.633191f, 0.234963f,  -0.391515f, 0.223306f,  0.049513f,  -0.985867f,
    -0.136110f, -0.953875f, -0.417542f, 0.049549f};
  std::vector<float> cos_ = {-0.508061f, 0.998221f,  -0.733869f, 0.164825f,
                             0.297530f,  -0.591437f, -0.165907f, -0.999895f,
                             -0.163179f, 0.249787f};
  std::vector<float> sin_ = {0.809128f,  -0.390829f, -0.112627f, 0.377992f,
                             -0.994569f, -0.641599f, -0.927266f, -0.401338f,
                             0.244335f,  -0.414953f};
  std::vector<float> ref_out = {
    0.684172f,  0.279348f,  -0.761074f, -0.073549f, 0.866425f,  -0.544224f,
    0.154785f,  -0.480342f, 0.010582f,  0.098163f,  0.146526f,  -1.030422f,
    0.547092f,  -0.252752f, -0.243571f, -0.162197f, -0.216517f, 0.636452f,
    -0.236000f, 0.120334f,  0.999478f,  -0.793769f, -0.733800f, -0.011226f,
    0.247067f,  -1.151284f, -0.030760f, 0.615511f,  0.205859f,  0.266457f,
    -0.454117f, 0.581280f,  0.181472f,  0.204634f,  -0.646542f, 0.020328f,
    0.556058f,  1.354487f,  -0.087349f, -0.396113f};
  std::vector<uint16_t> output(width);

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(),
                                      output.data(), cos_.data(), sin_.data(),
                                      only_convert_to_fp16);

  std::vector<float> f32_output = convert_vector_f16_as_uint16_to_f32(output);
  for (size_t i = 0; i < f32_output.size(); i++) {
    EXPECT_NEAR(ref_out[i], f32_output[i], 0.0003f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_fp16vcache_fp32_transposed) {
  int row_num = 1;
  int num_cache_head = 2;
  int gqa_size = 2;
  int head_dim = 9;
  int max_iter = 2;
  size_t in_size = max_iter * max_iter * num_cache_head * gqa_size;
  size_t vcache_size = max_iter * num_cache_head * head_dim;
  size_t output_size = num_cache_head * gqa_size * head_dim;

  std::vector<float> in = {0.092400f,  -0.414656f, -0.271581f, -0.300455f,
                           0.317950f,  -0.176271f, 0.138504f,  -0.361754f,
                           -0.024215f, -0.968064f, 0.360932f,  0.384471f};
  std::vector<float> f32_vcache = {
    -0.250920f, 0.593086f,  0.901429f,  -0.633130f, 0.463988f,  0.559382f,
    0.197317f,  0.193700f,  -0.687963f, -0.108334f, -0.688011f, -0.800050f,
    -0.883833f, -0.081502f, 0.732352f,  -0.332583f, 0.202230f,  -0.714266f,
    0.416145f,  0.301777f,  -0.958831f, -0.887177f, 0.939820f,  0.443998f,
    0.664885f,  0.877105f,  -0.575322f, -0.998442f, -0.636350f, 0.984423f,
    -0.633191f, 0.234963f,  -0.391515f, 0.223306f,  0.049513f,  -0.985867f};
  std::vector<uint16_t> vcache =
    convert_vector_f32_to_f16_as_uint16(f32_vcache);
  std::vector<float> ref_out = {
    0.109160f,  0.150761f,  -0.221623f, -0.340605f, 0.341716f,  0.192903f,
    0.229677f,  0.296728f,  -0.246453f, 0.030694f,  -0.299190f, -0.204716f,
    0.418990f,  -0.358029f, -0.310309f, -0.199024f, -0.234911f, 0.386668f,
    -0.108879f, 0.098724f,  0.353685f,  0.152306f,  0.054675f,  -0.253151f,
    0.121229f,  -0.048077f, 0.057462f,  0.393775f,  0.436868f,  -0.115650f,
    0.494638f,  -0.060525f, -0.078396f, 0.019140f,  -0.078680f, 0.571263f};
  std::vector<float> output(output_size);

  nntrainer::compute_fp16vcache_fp32_transposed(
    row_num, in.data(), vcache.data(), output.data(), num_cache_head, gqa_size,
    head_dim);

  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0001f);
  }
}
#endif

#if defined(ENABLE_FP16) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
TEST(nntrainer_cpu_backend_standalone, softmax_row_inplace_fp16) {
  size_t start_row = 0;
  size_t end_row = 3;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  // auto qk_out = generate_random_vector<float>(qk_out_size, -10, 10);
  std::vector<__fp16> qk_out = {
    -2.509198, 5.930860,  9.014286,  -6.331305, 4.639878,  5.593820,
    1.973170,  1.937003,  -6.879627, -1.083345, -6.880110, -8.000502,
    -8.838327, -0.815022, 7.323523,  -3.325828, 2.022300,  -7.142663,
    4.161452,  3.017770,  -9.588310, -8.871769, 9.398197,  4.439975,
    6.648853,  8.771055,  -5.753218, -9.984425, -6.363501, 9.844232};
  std::vector<__fp16> ref_out = {
    0.986697, 0.999999, 0.405184, 0.000021, 0.043301, 0.040031,
    0.487615, 0.999879, 0.000016, 0.000018, 0.012472, 0.000001,
    0.000000, 0.005194, 0.633859, 0.000005, 0.512170, 0.000114,
    0.999957, 0.001083, 0.000831, 0.000000, 0.594816, 0.994785,
    0.322840, 0.959963, 0.000215, 0.000007, 0.000027, 0.998899};

  nntrainer::softmax_row_inplace(qk_out.data(), start_row, end_row, num_heads);

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0005f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_kcaches_fp16) {
  int num_rows = 1;
  int num_cache_head = 2;
  int head_dim = 10;
  int group_size = 4;
  int tile_size = 16;
  int tile_offset = 0;
  size_t in_size = num_cache_head * group_size * head_dim;
  size_t kcache_size = num_rows * num_cache_head * head_dim;
  size_t output_size = num_rows * num_cache_head * group_size;
  std::vector<__fp16> in = {
    -2.509198, 5.930860,  9.014286,  -6.331305, 4.639878,  5.593820,  1.973170,
    1.937003,  -6.879627, -1.083345, -6.880110, -8.000502, -8.838327, -0.815022,
    7.323523,  -3.325828, 2.022300,  -7.142663, 4.161452,  3.017770,  -9.588310,
    -8.871769, 9.398197,  4.439975,  6.648853,  8.771055,  -5.753218, -9.984425,
    -6.363501, 9.844232,  -6.331910, 2.349631,  -3.915155, 2.233063,  0.495129,
    -9.858674, -1.361099, -9.538752, -4.175417, 0.495493,  2.237058,  -2.002780,
    -7.210123, -9.066687, -4.157107, 9.475111,  -2.672763, -5.344573, -0.878600,
    -8.187871, 5.703520,  2.367721,  -6.006524, -2.350760, 0.284688,  9.664618,
    1.848291,  -0.664742, -9.070992, 7.198808,  2.150897,  3.606151,  -6.589518,
    -0.990015, -8.698968, -9.734701, 8.977711,  8.844034,  9.312640,  1.265764,
    6.167947,  -2.291670, -3.907725, -9.680675, -8.046557, -5.382123, 3.684660,
    -5.179491, -1.196950, 3.665271};
  std::vector<__fp16> kcache = {
    0.000406, 0.006954, 0.020065, 0.000110, 0.004494, 0.006313, 0.001806,
    0.001789, 0.000093, 0.000663, 0.000093, 0.000060, 0.000035, 0.000727,
    0.011406, 0.000309, 0.001830, 0.000086, 0.003744, 0.002655};
  std::vector<__fp16> ref_out = {0.089252,  -0.072949, 0.058948,  -0.045583,
                                 -0.025812, -0.002068, -0.014971, -0.028027};
  std::vector<__fp16> output(output_size);
  for (int n = 0; n < num_cache_head; ++n) {
    const __fp16 *in_ptr = in.data() + n * group_size * head_dim;
    const __fp16 *kcache_ptr = kcache.data() + n * head_dim;
    __fp16 *out_ptr = output.data() + n * group_size;
    nntrainer::compute_kcaches(in_ptr, kcache_ptr, out_ptr, num_rows,
                               num_cache_head, head_dim, group_size,
                               tile_offset, tile_size);
  }

  for (size_t i = 0; i < output_size; i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_out_null_fp16) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;

  std::vector<__fp16> inout = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,  0.197317,
    0.193700,  -0.687963, -0.108334, -0.688011, -0.800050, -0.883833, -0.081502,
    0.732352,  -0.332583, 0.202230,  -0.714266, 0.416145,  0.301777,  -0.958831,
    -0.887177, 0.939820,  0.443998,  0.664885,  0.877105,  -0.575322, -0.998442,
    -0.636350, 0.984423,  -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,
    -0.985867, -0.136110, -0.953875, -0.417542, 0.049549};
  std::vector<__fp16> cos_ = {-0.508061, 0.998221,  -0.733869, 0.164825,
                              0.297530,  -0.591437, -0.165907, -0.999895,
                              -0.163179, 0.249787};
  std::vector<__fp16> sin_ = {0.809128,  -0.390829, -0.112627, 0.377992,
                              -0.994569, -0.641599, -0.927266, -0.401338,
                              0.244335,  -0.414953};
  std::vector<__fp16> ref_out = {
    0.684172,  0.279348,  -0.761074, -0.073549, 0.866425,  -0.544224, 0.154785,
    -0.480342, 0.010582,  0.098163,  0.146526,  -1.030422, 0.547092,  -0.252752,
    -0.243571, -0.162197, -0.216517, 0.636452,  -0.236000, 0.120334,  0.999478,
    -0.793769, -0.733800, -0.011226, 0.247067,  -1.151284, -0.030760, 0.615511,
    0.205859,  0.266457,  -0.454117, 0.581280,  0.181472,  0.204634,  -0.646542,
    0.020328,  0.556058,  1.354487,  -0.087349, -0.396113};

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(), nullptr,
                                      cos_.data(), sin_.data());

  for (size_t i = 0; i < inout.size(); i++) {
    EXPECT_NEAR(ref_out[i], inout[i], 0.001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_fp16) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;

  std::vector<__fp16> inout = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,  0.197317,
    0.193700,  -0.687963, -0.108334, -0.688011, -0.800050, -0.883833, -0.081502,
    0.732352,  -0.332583, 0.202230,  -0.714266, 0.416145,  0.301777,  -0.958831,
    -0.887177, 0.939820,  0.443998,  0.664885,  0.877105,  -0.575322, -0.998442,
    -0.636350, 0.984423,  -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,
    -0.985867, -0.136110, -0.953875, -0.417542, 0.049549};
  std::vector<__fp16> cos_ = {-0.508061, 0.998221,  -0.733869, 0.164825,
                              0.297530,  -0.591437, -0.165907, -0.999895,
                              -0.163179, 0.249787};
  std::vector<__fp16> sin_ = {0.809128,  -0.390829, -0.112627, 0.377992,
                              -0.994569, -0.641599, -0.927266, -0.401338,
                              0.244335,  -0.414953};
  std::vector<__fp16> ref_out = {
    0.684172,  0.279348,  -0.761074, -0.073549, 0.866425,  -0.544224, 0.154785,
    -0.480342, 0.010582,  0.098163,  0.146526,  -1.030422, 0.547092,  -0.252752,
    -0.243571, -0.162197, -0.216517, 0.636452,  -0.236000, 0.120334,  0.999478,
    -0.793769, -0.733800, -0.011226, 0.247067,  -1.151284, -0.030760, 0.615511,
    0.205859,  0.266457,  -0.454117, 0.581280,  0.181472,  0.204634,  -0.646542,
    0.020328,  0.556058,  1.354487,  -0.087349, -0.396113};
  std::vector<__fp16> output(width);

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(),
                                      output.data(), cos_.data(), sin_.data());

  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_fp16vcache_transposed_fp16) {
  int row_num = 1;
  int num_cache_head = 2;
  int gqa_size = 2;
  int head_dim = 9;
  int max_iter = 2;
  int chunk_size = 9;
  size_t in_size = max_iter * max_iter * num_cache_head * gqa_size;
  size_t vcache_size = max_iter * num_cache_head * head_dim;
  size_t output_size = num_cache_head * gqa_size * head_dim;

  std::vector<__fp16> in = {0.092400,  -0.414656, -0.271581, -0.300455,
                            0.317950,  -0.176271, 0.138504,  -0.361754,
                            -0.024215, -0.968064, 0.360932,  0.384471};
  std::vector<__fp16> vcache = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,
    0.197317,  0.193700,  -0.687963, -0.108334, -0.688011, -0.800050,
    -0.883833, -0.081502, 0.732352,  -0.332583, 0.202230,  -0.714266,
    0.416145,  0.301777,  -0.958831, -0.887177, 0.939820,  0.443998,
    0.664885,  0.877105,  -0.575322, -0.998442, -0.636350, 0.984423,
    -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,  -0.985867};
  std::vector<__fp16> ref_out = {
    0.109160,  0.150761,  -0.221623, -0.340605, 0.341716,  0.192903,
    0.229677,  0.296728,  -0.246453, 0.030694,  -0.299190, -0.204716,
    0.418990,  -0.358029, -0.310309, -0.199024, -0.234911, 0.386668,
    -0.108879, 0.098724,  0.353685,  0.152306,  0.054675,  -0.253151,
    0.121229,  -0.048077, 0.057462,  0.393775,  0.436868,  -0.115650,
    0.494638,  -0.060525, -0.078396, 0.019140,  -0.078680, 0.571263};
  std::vector<__fp16> output(output_size);

  for (int n = 0; n < num_cache_head; ++n) {
    int chunk_size = head_dim;
    const _FP16 *in_ptr = in.data() + n * gqa_size;
    const _FP16 *vcache_ptr = vcache.data() + n * head_dim;
    _FP16 *out_ptr = output.data() + n * gqa_size * head_dim;
    nntrainer::compute_fp16vcache_transposed(row_num, in_ptr, vcache_ptr,
                                             out_ptr, num_cache_head, gqa_size,
                                             head_dim, chunk_size);
  }

  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0004f);
  }
}
#endif

static void run_clamp_test(const unsigned int N, float lower_bound,
                           float upper_bound, bool print = false) {
  const int TEST_CNT = 20;
  nanoseconds ref_mul_time = (nanoseconds)0;
  nanoseconds mul_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> X = generate_random_vector<float, false>(N, -10, 10);
    std::vector<float> Y = generate_random_vector<float, false>(N);
    std::vector<float> Y_ref = generate_random_vector<float, false>(N);
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::__fallback_clamp(X.data(), Y_ref.data(), N, lower_bound,
                                  upper_bound);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_mul_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::clamp(X.data(), Y.data(), N, lower_bound, upper_bound);
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        mul_time += dt;
      }
    }

    auto mean_squared_error = compute_mse(1, N, Y_ref, Y, print);
    ASSERT_LE(mean_squared_error, 0.00001f);
  }

  if (print) {
    std::cout << "[INFO] clamp: TEST CNT: " << TEST_CNT << ", N: " << N
              << ", Average ref_time: " << ref_mul_time.count() / TEST_CNT
              << " ns, Average mul_time: " << mul_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, clamp_3072_0_1) {
  const unsigned int N = 3072;
  float lower_bound = 0.F;
  float upper_bound = 1.F;
  run_clamp_test(N, lower_bound, upper_bound, false);
}

static inline void printMatrixI(const char *name, float *data, int Y, int X) {
  printf("%s :\n", name);
  for (int y = 0; y < Y; y++) {
    // printf("[");
    for (int x = 0; x < X; x++) {
      if (x % 10 == 0) {
        printf("| ");
      }
      std::cout << (int)(0.5f + data[y * X + x]) << " ";
    }
    printf("\n");
  }
}

static inline std::vector<float> generate_01_vector(const size_t size,
                                                    const float ones_ratio) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, (float)size);
  if (ones_ratio >= 1.0) {
    std::vector<float> vec(size, 1.0f);
    return vec;
  } else {
    std::vector<float> vec(size, 0.0f);
    size_t ones_cnt = (size_t)(size * ones_ratio);
    for (size_t i = 0; i < ones_cnt; i++) {
      int pos = static_cast<int>(dist(gen));
      vec[pos] = 1.0f;
    }
    return vec;
  }
}

static void run_transform_int4_test_(const uint32_t K, const uint32_t N,
                                     const int scale_group_size,
                                     bool use_ones = false) {
  const size_t q4_data_size = K * N / Q4_0 * sizeof(block_q4_0);
  std::vector<float> weight_fp32;
  if (use_ones) {
    float ones_ratio = 0.1f;
    weight_fp32 = generate_01_vector(N * K, ones_ratio);
  } else {
    weight_fp32 = generate_random_vector<float, false>(N * K, -1.0, 1.0);
  }

  bool print = false;
  if (print && use_ones) {
    printMatrixI("weight_fp32", weight_fp32.data(), N, K);
  }

  std::vector<uint8_t> osv32_weights;
  std::vector<uint16_t> osv32_scales;
  nntrainer::Int4Utils::quantizeAndRepack(
    weight_fp32.data(), N, K, scale_group_size, osv32_weights, osv32_scales);

  // MAIN TEST - direct transform Int4 data (osv32_isv2) ---> Q4_0x
  std::vector<uint8_t> dst_q4_0x(q4_data_size);
  auto t0 = std::chrono::high_resolution_clock::now();
  nntrainer::transform_q4_0x_from_int4(N, K, osv32_weights.data(),
                                       osv32_scales.data(), scale_group_size,
                                       dst_q4_0x.data());
  auto t1 = std::chrono::high_resolution_clock::now();
  auto exec_time =
    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  std::cout << "Time: " << (double)exec_time.count() / 1000 << " ms"
            << std::endl;

  // Check MSE quantized values
  std::vector<uint8_t> unpacked_weights_q4(q4_data_size);
  nntrainer::unpack_q4_0(dst_q4_0x.data(), unpacked_weights_q4.data(),
                         q4_data_size, N, K);
  std::vector<float> dequantized_weights_q4(N * K);
  nntrainer::dequantize_row_q4_0(unpacked_weights_q4.data(),
                                 dequantized_weights_q4.data(), N * K);
  if (print && use_ones) {
    printMatrixI("dequantized_weights_q4 I", dequantized_weights_q4.data(), N,
                 K);
  }
  float mse_direct_transform_q4 =
    mse<float>(weight_fp32.data(), dequantized_weights_q4.data(), N * K);
  std::cout << "MSE direct transform Q4_0: " << std::setprecision(10)
            << mse_direct_transform_q4 << std::endl;
  // MAIN TEST - END

  // Reference solution - Int4 data (osv32_isv2) --> FP32 --> quantization to
  // Q4_0x8 For checking difference of accuracy
  std::vector<float> dequant_weight_fp32(N * K);
  nntrainer::Int4Utils::dequantizePacked(osv32_weights, osv32_scales, N, K,
                                         scale_group_size, dequant_weight_fp32);
  std::vector<uint8_t> tmp_q4_weight(q4_data_size);
  nntrainer::quantize_q4_0(dequant_weight_fp32.data(), tmp_q4_weight.data(), N,
                           K, nullptr);
  std::vector<uint8_t> ref_q4_0x8(q4_data_size);
  nntrainer::repack_q4_0(ref_q4_0x8.data(), tmp_q4_weight.data(), q4_data_size,
                         N, K);
  // Check MSE quantized values
  std::vector<uint8_t> unpacked_ref_weights_q4(q4_data_size);
  nntrainer::unpack_q4_0(ref_q4_0x8.data(), unpacked_ref_weights_q4.data(),
                         q4_data_size, N, K);
  std::vector<float> dequantized_ref_weights_q4(N * K);
  nntrainer::dequantize_row_q4_0(unpacked_ref_weights_q4.data(),
                                 dequantized_ref_weights_q4.data(), N * K);
  float mse_fp32_transform_q4 =
    mse<float>(weight_fp32.data(), dequantized_ref_weights_q4.data(), N * K);
  std::cout << "MSE FP32 transform Q4_0:   " << std::setprecision(10)
            << mse_fp32_transform_q4 << std::endl;
  // Reference solution - END

  float mse_dequant_int4_vs_trans_q4 = mse<float>(
    dequant_weight_fp32.data(), dequantized_weights_q4.data(), N * K);
  std::cout << "MSE dequant Int4 vs direct transform Q4_0: "
            << std::setprecision(10) << mse_dequant_int4_vs_trans_q4
            << std::endl;

  // This is a proof that the transformQ4_0x_FromInt4() is lossless computation.
  const float epsilon = 0.00001f;
  EXPECT_IN_RANGE(mse_dequant_int4_vs_trans_q4, 0, epsilon);

  // Other MSEs
  const float epsilon_direct = (use_ones) ? 0.00001f : 0.002f;
  EXPECT_IN_RANGE(mse_direct_transform_q4, 0, epsilon_direct);
  const float epsilon_fp32 = (use_ones) ? 0.00001f : 0.004f;
  EXPECT_IN_RANGE(mse_fp32_transform_q4, 0, epsilon_fp32);
  if (!use_ones) {
    EXPECT_LE(mse_direct_transform_q4, mse_fp32_transform_q4);
  }

  // Additional test with using GEMMs
  const uint32_t M = 4;
  const uint32_t input_size = M * K;
  const int INT4_BLOCK_N_SIZE = 32;
  uint32_t alignN = nntrainer::align(N, INT4_BLOCK_N_SIZE);
  std::vector<float> input =
    generate_random_vector<float, false>(input_size, -1.0, 1.0);

  // SGEMM - reference
  std::vector<float> ref_output_fp32(M * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, input.data(), K,
                   weight_fp32.data(), K, 0.F, ref_output_fp32.data(), N);

  // GEMM INT4->transform->Q4
  std::vector<float> int4_trans_q4_output_fp32(M * N);
  nntrainer::gemm_q4_0(M, N, K, input.data(), K, dst_q4_0x.data(), N,
                       int4_trans_q4_output_fp32.data(), N);
  float mse_int4_trans_q4 =
    mse<float>(ref_output_fp32.data(), int4_trans_q4_output_fp32.data(), M * N);
  std::cout << "MSE GEMM INT4->transform->Q4:            "
            << std::setprecision(10) << mse_int4_trans_q4 << std::endl;

  // GEMM INT4->dequant->FP32->quant->Q4
  std::vector<float> int4_fp32_q4_output_fp32(M * N);
  nntrainer::gemm_q4_0(M, N, K, input.data(), K, ref_q4_0x8.data(), N,
                       int4_fp32_q4_output_fp32.data(), N);
  float mse_int4_fp32_q4 =
    mse<float>(ref_output_fp32.data(), int4_fp32_q4_output_fp32.data(), M * N);
  std::cout << "MSE GEMM INT4->dequant->FP32->quant->Q4: "
            << std::setprecision(10) << mse_int4_fp32_q4 << std::endl;
}

#define DECLARE_transform_int4_test_K_N(K, N, G)                               \
  TEST(nntrainer_blas_kernel,                                                  \
       _transform_int4_test_K##K##_N##N##_Group##G##_RandomOnes) {             \
    run_transform_int4_test_(K, N, G, true);                                   \
  }                                                                            \
  TEST(nntrainer_blas_kernel,                                                  \
       _transform_int4_test_K##K##_N##N##_Group##G##_RandomFloat) {            \
    run_transform_int4_test_(K, N, G, false);                                  \
  }

DECLARE_transform_int4_test_K_N(128, 40, 32);
DECLARE_transform_int4_test_K_N(128, 40, 64);
DECLARE_transform_int4_test_K_N(256, 40, 128);
DECLARE_transform_int4_test_K_N(32, 40, 32);
DECLARE_transform_int4_test_K_N(32, 48, 32);
DECLARE_transform_int4_test_K_N(64, 40, 32);
DECLARE_transform_int4_test_K_N(320, 640, 32);
DECLARE_transform_int4_test_K_N(1024, 640, 32);
DECLARE_transform_int4_test_K_N(1024, 648, 32);
DECLARE_transform_int4_test_K_N(1024, 648, 64);
DECLARE_transform_int4_test_K_N(1024, 648, 128);
DECLARE_transform_int4_test_K_N(3072, 8192, 32);

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
