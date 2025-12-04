// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_cuda_gemm_q40.cpp
 * @date   25 Nov 2025
 * @brief  Unit test for CUDA Q4_0 quantized matrix multiplication
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>
#include <nntrainer_test_util.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>

#include <cpu_backend.h>

// Include the function under test
#include "ggml_mmq.h"

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(error) << std::endl;                     \
      FAIL();                                                                  \
    }                                                                          \
  } while (0)

/**
 * @brief Compute reference matrix multiplication C = A * B (FP32)
 */
void matmul_fp32_ref(const float* A, const float* B, float* C,
                     int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

/**
 * @brief Compute Mean Squared Error between two arrays
 */
float compute_mse(const float* a, const float* b, int size) {
  double sum = 0.0;
  for (int i = 0; i < size; ++i) {
    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += diff * diff;
  }
  return static_cast<float>(sum / size);
}

/**
 * @brief Test for ggml_cuda_op_mul_mat_q40 function
 */
TEST(nntrainer_CUDA, gemm_q40_basic) {
  // Matrix dimensions
  const int M = 128;  // Rows of A
  const int K = 256;  // Cols of A, Rows of B (must be multiple of 32 for Q4_0)
  const int N = 64;   // Cols of B

  // Allocate host memory for FP32 matrices
  std::vector<float> A_fp32(M * K);
  std::vector<float> B_fp32(K * N);
  std::vector<float> C_ref(M * N);
  std::vector<float> C_result(M * N);

  // Initialize random number generator
  std::mt19937 gen(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Initialize matrices A and B with random values
  for (int i = 0; i < M * K; ++i) {
    A_fp32[i] = dist(gen);
  }
  for (int i = 0; i < K * N; ++i) {
    B_fp32[i] = dist(gen);
  }

  // Step 1: Compute reference result C_ref = A * B (FP32)
  matmul_fp32_ref(A_fp32.data(), B_fp32.data(), C_ref.data(), M, K, N);

  // Step 2: Quantize A to Q4_0 format
  const int q4_0_block_size = 36;  // sizeof(block_q4_0) = 2 + 16 = 18 bytes per 32 elements
  const int num_blocks_A = (M * K) / 32;
  std::vector<uint8_t> A_q40(num_blocks_A * q4_0_block_size);
  
  nntrainer::quantize_q4_0(A_fp32.data(), A_q40.data(), M, K, nullptr);

  // Step 3: Quantize B to Q8_1 format (using quantize_row_q8_1_ref)
  const int q8_1_block_size = 36;  // sizeof(block_q8_1) = 4 + 32 = 36 bytes per 32 elements
  const int num_blocks_B = (K * N) / 32;
  std::vector<uint8_t> B_q81(num_blocks_B * q8_1_block_size);
  
  // Quantize B row by row
  for (int row = 0; row < K; ++row) {
    quantize_row_q8_1_ref(
      B_fp32.data() + row * N,
      reinterpret_cast<block_q8_1*>(B_q81.data()) + row * (N / 32),
      N
    );
  }

  // Allocate device memory
  char* d_A_q40 = nullptr;
  char* d_B_q81 = nullptr;
  float* d_C = nullptr;

  CUDA_CHECK(cudaMalloc(&d_A_q40, A_q40.size()));
  CUDA_CHECK(cudaMalloc(&d_B_q81, B_q81.size()));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  // Copy quantized data to device
  CUDA_CHECK(cudaMemcpy(d_A_q40, A_q40.data(), A_q40.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B_q81, B_q81.data(), B_q81.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Step 4: Call ggml_cuda_op_mul_mat_q40
  ggml_cuda_op_mul_mat_q40(
    d_A_q40,        // src0_dd_i: quantized matrix A (Q4_0)
    nullptr,        // src1_ddf_i: unused
    d_B_q81,        // src1_ddq_i: quantized matrix B (Q8_1)
    d_C,            // dst_dd_i: output matrix C (FP32)
    0,              // row_low
    M,              // row_high
    N,              // src1_ncols
    N,              // src1_padded_row_size
    K,              // ne00: K dimension
    K,              // ne10: K dimension
    N,              // ne11: N dimension
    N,              // ne0: leading dimension of output
    stream
  );

  // Synchronize stream
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(C_result.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Step 5: Compute MSE between C_ref and C_result
  float mse = compute_mse(C_ref.data(), C_result.data(), M * N);

  std::cout << "Matrix dimensions: M=" << M << ", K=" << K << ", N=" << N << std::endl;
  std::cout << "MSE between FP32 and Q4_0×Q8_1: " << mse << std::endl;

  // Q4_0 quantization introduces error, so we use a relaxed threshold
  // Typical MSE for 4-bit quantization is in the range of 1e-3 to 1e-2
  const float mse_threshold = 1e-2f;
  EXPECT_IN_RANGE(mse, 0.0f, mse_threshold);

  // Cleanup
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_A_q40));
  CUDA_CHECK(cudaFree(d_B_q81));
  CUDA_CHECK(cudaFree(d_C));
}

/**
 * @brief Test with different matrix sizes
 */
TEST(nntrainer_CUDA, gemm_q40_various_sizes) {
  struct TestCase {
    int M, K, N;
    float max_mse;
  };

  std::vector<TestCase> test_cases = {
    {32, 64, 32, 1e-2f},
    {64, 128, 64, 1e-2f},
    {128, 256, 128, 1e-2f},
  };

  for (const auto& tc : test_cases) {
    std::cout << "\nTesting M=" << tc.M << ", K=" << tc.K << ", N=" << tc.N << std::endl;

    // Allocate and initialize matrices
    std::vector<float> A_fp32(tc.M * tc.K);
    std::vector<float> B_fp32(tc.K * tc.N);
    std::vector<float> C_ref(tc.M * tc.N);
    std::vector<float> C_result(tc.M * tc.N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& val : A_fp32) val = dist(gen);
    for (auto& val : B_fp32) val = dist(gen);

    // Compute reference
    matmul_fp32_ref(A_fp32.data(), B_fp32.data(), C_ref.data(), tc.M, tc.K, tc.N);

    // Quantize
    const int num_blocks_A = (tc.M * tc.K) / 32;
    const int num_blocks_B = (tc.K * tc.N) / 32;
    std::vector<uint8_t> A_q40(num_blocks_A * 36);
    std::vector<uint8_t> B_q81(num_blocks_B * 36);

    nntrainer::quantize_q4_0(A_fp32.data(), A_q40.data(), tc.M, tc.K, nullptr);
    
    for (int row = 0; row < tc.K; ++row) {
      quantize_row_q8_1_ref(
        B_fp32.data() + row * tc.N,
        reinterpret_cast<block_q8_1*>(B_q81.data()) + row * (tc.N / 32),
        tc.N
      );
    }

    // Allocate device memory
    char* d_A_q40 = nullptr;
    char* d_B_q81 = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A_q40, A_q40.size()));
    CUDA_CHECK(cudaMalloc(&d_B_q81, B_q81.size()));
    CUDA_CHECK(cudaMalloc(&d_C, tc.M * tc.N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A_q40, A_q40.data(), A_q40.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_q81, B_q81.data(), B_q81.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, tc.M * tc.N * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Execute
    ggml_cuda_op_mul_mat_q40(
      d_A_q40, nullptr, d_B_q81, d_C,
      0, tc.M, tc.N, tc.N,
      tc.K, tc.K, tc.N, tc.N,
      stream
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(C_result.data(), d_C, tc.M * tc.N * sizeof(float), cudaMemcpyDeviceToHost));

    // Validate
    float mse = compute_mse(C_ref.data(), C_result.data(), tc.M * tc.N);
    std::cout << "  MSE: " << mse << std::endl;
    EXPECT_IN_RANGE(mse, 0.0f, tc.max_mse);

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A_q40));
    CUDA_CHECK(cudaFree(d_B_q81));
    CUDA_CHECK(cudaFree(d_C));
  }
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
