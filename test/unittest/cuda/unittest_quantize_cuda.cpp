// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_quantize_cuda.cpp
 * @brief  Unit test for CUDA quantization functions
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 *
 */

#include "quantize_cuda.h"
#include "unittest_util.h"
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

// Q8_1 Block Structure definition for verification (should match CUDA)
#define QK8_1 32
struct block_q8_1 {
  half d;
  half s;
  int8_t qs[QK8_1];
};

// Test parameters: (M, K, group_size)
struct QuantizeInt8PadParams {
  unsigned int M;
  unsigned int K;
  unsigned int group_size;
};

class QuantizeInt8PadTest
  : public ::testing::TestWithParam<QuantizeInt8PadParams> {};

TEST_P(QuantizeInt8PadTest, CompareWithCPU) {
  auto params = GetParam();
  const unsigned int M = params.M;
  const unsigned int K = params.K;
  const unsigned int group_size = params.group_size;

  // Calculate sizes
  const unsigned int align_k = ((K + group_size - 1) / group_size) * group_size;
  const unsigned int groups_in_row = align_k / group_size;
  const unsigned int total_groups = M * groups_in_row;

  std::vector<float> input = nntrainer::generate_random_vector<float>(M * K);
  std::vector<int8_t> expected_output(M * align_k);
  std::vector<uint16_t> expected_scales(total_groups * 2);

  std::vector<int8_t> actual_output(M * align_k);
  std::vector<half> actual_scales(total_groups * 2);

  // CPU timing
  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; ++i) {
    nntrainer::cpu_quantize_input_int8_pad(input.data(), expected_output.data(),
                                           expected_scales.data(), M, K,
                                           group_size);
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  float cpu_time_ms =
    std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count() /
    10.0f;

  // CUDA implementation
  float *d_input;
  int8_t *d_output;
  half *d_scales;

  cudaMalloc(&d_input, M * K * sizeof(float));
  cudaMalloc(&d_output, M * align_k * sizeof(int8_t));
  cudaMalloc(&d_scales, total_groups * 2 * sizeof(half));

  cudaMemcpy(d_input, input.data(), M * K * sizeof(float),
             cudaMemcpyHostToDevice);

  // CUDA timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float cuda_time_ms = 0;
  for (int i = 0; i < 10; ++i) {
    cudaEventRecord(start);
    quantize_input_int8_pad_cuda(d_input, d_output, d_scales, M, K, group_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float iter_time = 0;
    cudaEventElapsedTime(&iter_time, start, stop);
    cuda_time_ms += iter_time;
  }
  cuda_time_ms /= 10.0f;

  cudaMemcpy(actual_output.data(), d_output, M * align_k * sizeof(int8_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(actual_scales.data(), d_scales, total_groups * 2 * sizeof(half),
             cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_scales);

  // Print timing comparison
  float speedup = cpu_time_ms / cuda_time_ms;
  printf("QuantizeInputInt8Pad [M=%u, K=%u, group=%u]: CPU=%.4f ms, CUDA=%.4f "
         "ms, Speedup=%.2fx\n",
         M, K, group_size, cpu_time_ms, cuda_time_ms, speedup);

  // Verification
  for (size_t i = 0; i < expected_output.size(); ++i) {
    EXPECT_NEAR(expected_output[i], actual_output[i], 1);
  }

  for (size_t i = 0; i < total_groups; ++i) {
    float expected_scale_val = ((half *)expected_scales.data())[i * 2];
    float actual_scale_val = (float)actual_scales[i * 2];
    EXPECT_NEAR(expected_scale_val, actual_scale_val, 1e-3);
  }
}

INSTANTIATE_TEST_SUITE_P(
  VariousSizes, QuantizeInt8PadTest,
  ::testing::Values(QuantizeInt8PadParams{512, 512, 32},
                    QuantizeInt8PadParams{1024, 1024, 32},
                    QuantizeInt8PadParams{3072, 3072, 32},
                    QuantizeInt8PadParams{3072, 8192, 32}));

// Test parameters for Q8_1: (k)
struct QuantizeQ8_1Params {
  int64_t k;
};

class QuantizeQ8_1Test : public ::testing::TestWithParam<QuantizeQ8_1Params> {};

TEST_P(QuantizeQ8_1Test, CompareWithCPU) {
  auto params = GetParam();
  const int64_t k = params.k;
  const int64_t num_blocks = k / 32;

  std::vector<float> input =
    nntrainer::generate_random_vector<float>(k, -10.0f, 10.0f);

  std::vector<uint8_t> expected_output(num_blocks * sizeof(block_q8_1));
  std::vector<uint8_t> actual_output(num_blocks * sizeof(block_q8_1));

  // CPU timing
  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; ++i) {
    quantize_row_q8_1_host(input.data(), expected_output.data(), k);
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  float cpu_time_ms =
    std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count() /
    10.0f;

  // CUDA implementation
  float *d_input;
  void *d_output;

  cudaMalloc(&d_input, k * sizeof(float));
  cudaMalloc(&d_output, num_blocks * sizeof(block_q8_1));

  cudaMemcpy(d_input, input.data(), k * sizeof(float), cudaMemcpyHostToDevice);

  // CUDA timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float cuda_time_ms = 0;
  for (int i = 0; i < 10; ++i) {
    cudaEventRecord(start);
    quantize_activation_q8_1_cuda(d_input, d_output, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float iter_time = 0;
    cudaEventElapsedTime(&iter_time, start, stop);
    cuda_time_ms += iter_time;
  }
  cuda_time_ms /= 10.0f;

  cudaMemcpy(actual_output.data(), d_output, num_blocks * sizeof(block_q8_1),
             cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);

  // Print timing comparison
  float speedup = cpu_time_ms / cuda_time_ms;
  printf("QuantizeActivationQ8_1 [k=%lld]: CPU=%.4f ms, CUDA=%.4f ms, "
         "Speedup=%.2fx\n",
         k, cpu_time_ms, cuda_time_ms, speedup);

  // Verification
  block_q8_1 *cpu_blocks = (block_q8_1 *)expected_output.data();
  block_q8_1 *gpu_blocks = (block_q8_1 *)actual_output.data();

  for (int64_t i = 0; i < num_blocks; ++i) {
    float cpu_d = (float)cpu_blocks[i].d;
    float gpu_d = (float)gpu_blocks[i].d;
    float cpu_s = (float)cpu_blocks[i].s;
    float gpu_s = (float)gpu_blocks[i].s;

    // d: scale factor stored in FP16, use relative error tolerance
    // FP16 has ~3 decimal digits of precision, so 0.1% relative error is
    // reasonable
    float d_threshold = std::max(1e-3f, std::abs(cpu_d) * 0.001f);
    EXPECT_NEAR(cpu_d, gpu_d, d_threshold) << "Block " << i << " d mismatch";

    // s: sum of 32 quantized values multiplied by d
    // Each quantized value can have ±0.5 rounding error
    // Worst case accumulated error: 32 * 0.5 * d = 16 * d
    // Use 5% relative error or absolute error based on d, whichever is larger
    float s_threshold =
      std::max(std::abs(cpu_d) * 16.0f, std::abs(cpu_s) * 0.05f);
    EXPECT_NEAR(cpu_s, gpu_s, s_threshold)
      << "Block " << i << " s mismatch (cpu_d=" << cpu_d << ")";

    for (int j = 0; j < 32; ++j) {
      EXPECT_EQ(cpu_blocks[i].qs[j], gpu_blocks[i].qs[j])
        << "Block " << i << " qs[" << j << "] mismatch";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(VariousSizes, QuantizeQ8_1Test,
                         ::testing::Values(QuantizeQ8_1Params{256 * 256},
                                           QuantizeQ8_1Params{1024 * 1024},
                                           QuantizeQ8_1Params{3072 * 3072},
                                           QuantizeQ8_1Params{8192 * 8192}));

/**
 * @brief Main function for running tests
 */
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
