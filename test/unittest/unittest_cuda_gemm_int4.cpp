// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_cuda_gemm_int4.cpp
 * @brief  Unit test for CUDA int4 GEMM operations
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "gemm_int4_cpu.h"
#include "gemm_int4_cuda.h"
#include "int4_utils.h"
#include "unittest_util.h" // For generate_random_vector, etc.

using namespace nntrainer;

// Helper for ceil division
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

static void run_cuda_int4_gemm_packed_block_test_(const uint32_t M,
                                                  const uint32_t K,
                                                  const uint32_t N,
                                                  const int scale_group_size) {
  // 1. Prepare Data
  std::vector<float> input = generate_random_vector<float>(M * K, -1.0f, 1.0f);
  std::vector<float> weights_fp32 =
    generate_random_vector<float>(K * N, -0.5f, 0.5f);
  std::vector<float> output_cuda(M * N);
  std::vector<float> output_ref(M * N);

  // 2. Quantize Weights using Int4Utils
  // Note: quantizeAndRepack expects weights in N x K layout (N rows, K cols)
  // But our weights_fp32 is K x N (K rows, N cols)
  // We need to transpose weights to N x K before quantization
  std::vector<float> weights_NxK(N * K);
  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int n = 0; n < N; ++n) {
      weights_NxK[n * K + k] = weights_fp32[k * N + n];
    }
  }

  std::vector<uint8_t> weights_int4;
  std::vector<uint16_t> scales_fp16;

  Int4Utils::quantizeAndRepack(weights_NxK.data(), N, K, scale_group_size,
                               weights_int4, scales_fp16);

  // 3. Quantize Input using cpu_quantize_input_int8_pad
  unsigned int alignK = ALIGN(K, scale_group_size);
  unsigned int groups_in_row = alignK / scale_group_size;
  std::vector<int8_t> input_quantized(M * alignK);
  std::vector<uint16_t> input_scales(M * groups_in_row * 2);

  cpu_quantize_input_int8_pad(input.data(), input_quantized.data(),
                              input_scales.data(), M, K, scale_group_size);

  // 4. Run CUDA Kernel
  float *d_output;
  int8_t *d_quantized_input;
  uint8_t *d_weights;
  uint16_t *d_scales;
  uint16_t *d_input_scales;

  // Allocate device memory
  size_t input_bytes = input_quantized.size() * sizeof(int8_t);
  size_t weights_bytes = weights_int4.size() * sizeof(uint8_t);
  size_t scales_bytes = scales_fp16.size() * sizeof(uint16_t);
  size_t input_scales_bytes = input_scales.size() * sizeof(uint16_t);
  size_t output_bytes = output_cuda.size() * sizeof(float);

  cudaMalloc(&d_quantized_input, input_bytes);
  cudaMalloc(&d_weights, weights_bytes);
  cudaMalloc(&d_scales, scales_bytes);
  cudaMalloc(&d_input_scales, input_scales_bytes);
  cudaMalloc(&d_output, output_bytes);

  // Copy data to device
  cudaMemcpy(d_quantized_input, input_quantized.data(), input_bytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights, weights_int4.data(), weights_bytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_scales, scales_fp16.data(), scales_bytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_scales, input_scales.data(), input_scales_bytes,
             cudaMemcpyHostToDevice);

  // Warmup run
  gemm_int4_cuda_packed_block(d_quantized_input, d_weights, d_scales,
                              d_input_scales, d_output, M, N, K,
                              scale_group_size);

  // Benchmark runs
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < 9; ++i) {
    gemm_int4_cuda_packed_block(d_quantized_input, d_weights, d_scales,
                                d_input_scales, d_output, M, N, K,
                                scale_group_size);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Average execution time (over 9 runs): " << milliseconds / 9.0f
            << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy output back
  cudaMemcpy(output_cuda.data(), d_output, output_bytes,
             cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_quantized_input);
  cudaFree(d_weights);
  cudaFree(d_scales);
  cudaFree(d_input_scales);
  cudaFree(d_output);

  // 5. Run Reference (FP32 GEMM)
  // Use FP32 reference as requested
  gemm_fp32_ref(input.data(), weights_fp32.data(), output_ref.data(), M, N, K);

  // 6. Compare
  float mse = 0.0f;
  float max_diff = 0.0f;
  for (size_t i = 0; i < output_cuda.size(); ++i) {
    float diff = output_cuda[i] - output_ref[i];
    mse += diff * diff;
    if (std::abs(diff) > max_diff) {
      max_diff = std::abs(diff);
    }
  }
  mse /= output_cuda.size();

  std::cout << "[Packed Block] M: " << M << " K: " << K << " N: " << N
            << " Group: " << scale_group_size << " MSE: " << mse
            << " MaxDiff: " << max_diff << std::endl;

  // Threshold calculation (same as CPU test)
  float input_min = -1.0f, input_max = 1.0f;
  float weight_min = -0.5f, weight_max = 0.5f;
  float input_range = input_max - input_min;
  float weight_range = weight_max - weight_min;
  float delta_input = input_range / 255.0f;
  float delta_weight = weight_range / 15.0f;
  float var_err_input = (delta_input * delta_input) / 12.0f;
  float var_err_weight = (delta_weight * delta_weight) / 12.0f;
  float mean_sq_input =
    (input_min * input_min + input_min * input_max + input_max * input_max) /
    3.0f;
  float mean_sq_weight = (weight_min * weight_min + weight_min * weight_max +
                          weight_max * weight_max) /
                         3.0f;
  float var_err_product =
    mean_sq_input * var_err_weight + mean_sq_weight * var_err_input;
  float expected_mse = K * var_err_product;
  float threshold = expected_mse * 2.0f;

  std::cout << "Expected MSE: " << expected_mse << " Threshold: " << threshold
            << std::endl;
  EXPECT_LT(mse, threshold);
}

#define DECLARE_CUDA_INT4_GEMM_PACKED_BLOCK_TEST(M, K, N, G)                   \
  TEST(gemm_int4_cuda, packed_block_##M##_##K##_##N##_Group##G) {              \
    run_cuda_int4_gemm_packed_block_test_(M, K, N, G);                         \
  }

DECLARE_CUDA_INT4_GEMM_PACKED_BLOCK_TEST(28, 3072, 256, 32);
DECLARE_CUDA_INT4_GEMM_PACKED_BLOCK_TEST(28, 3072, 3072, 32);
DECLARE_CUDA_INT4_GEMM_PACKED_BLOCK_TEST(28, 3072, 8192, 32);
DECLARE_CUDA_INT4_GEMM_PACKED_BLOCK_TEST(28, 8192, 3072, 32);

static void
run_cuda_int4_gemm_packed_block_16_test_(const uint32_t M, const uint32_t K,
                                         const uint32_t N,
                                         const int scale_group_size) {
  // 1. Prepare Data
  std::vector<float> input = generate_random_vector<float>(M * K, -1.0f, 1.0f);
  std::vector<float> weights_fp32 =
    generate_random_vector<float>(K * N, -0.5f, 0.5f);
  std::vector<float> output_cuda(M * N);
  std::vector<float> output_ref(M * N);

  // 2. Quantize Weights using Int4Utils
  // Note: quantizeAndRepack expects weights in N x K layout (N rows, K cols)
  // But our weights_fp32 is K x N (K rows, N cols)
  // We need to transpose weights to N x K before quantization
  std::vector<float> weights_NxK(N * K);
  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int n = 0; n < N; ++n) {
      weights_NxK[n * K + k] = weights_fp32[k * N + n];
    }
  }

  std::vector<uint8_t> weights_int4;
  std::vector<uint16_t> scales_fp16;

  Int4Utils::quantizeAndRepack(weights_NxK.data(), N, K, scale_group_size,
                               weights_int4, scales_fp16);

  // 3. Quantize Input using cpu_quantize_input_int8_pad
  unsigned int alignK = ALIGN(K, scale_group_size);
  unsigned int groups_in_row = alignK / scale_group_size;
  std::vector<int8_t> input_quantized(M * alignK);
  std::vector<uint16_t> input_scales(M * groups_in_row * 2);

  cpu_quantize_input_int8_pad(input.data(), input_quantized.data(),
                              input_scales.data(), M, K, scale_group_size);

  // 4. Run CUDA Kernel
  float *d_output;
  int8_t *d_quantized_input;
  uint8_t *d_weights;
  uint16_t *d_scales;
  uint16_t *d_input_scales;

  // Allocate device memory
  size_t input_bytes = input_quantized.size() * sizeof(int8_t);
  size_t weights_bytes = weights_int4.size() * sizeof(uint8_t);
  size_t scales_bytes = scales_fp16.size() * sizeof(uint16_t);
  size_t input_scales_bytes = input_scales.size() * sizeof(uint16_t);
  size_t output_bytes = output_cuda.size() * sizeof(float);

  cudaMalloc(&d_quantized_input, input_bytes);
  cudaMalloc(&d_weights, weights_bytes);
  cudaMalloc(&d_scales, scales_bytes);
  cudaMalloc(&d_input_scales, input_scales_bytes);
  cudaMalloc(&d_output, output_bytes);

  // Copy data to device
  cudaMemcpy(d_quantized_input, input_quantized.data(), input_bytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights, weights_int4.data(), weights_bytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_scales, scales_fp16.data(), scales_bytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_scales, input_scales.data(), input_scales_bytes,
             cudaMemcpyHostToDevice);

  // Warmup run
  gemm_int4_cuda_packed_block_16(d_quantized_input, d_weights, d_scales,
                                 d_input_scales, d_output, M, N, K,
                                 scale_group_size);

  // Benchmark runs
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < 9; ++i) {
    gemm_int4_cuda_packed_block_16(d_quantized_input, d_weights, d_scales,
                                   d_input_scales, d_output, M, N, K,
                                   scale_group_size);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Average execution time (over 9 runs): " << milliseconds / 9.0f
            << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy output back
  cudaMemcpy(output_cuda.data(), d_output, output_bytes,
             cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_quantized_input);
  cudaFree(d_weights);
  cudaFree(d_scales);
  cudaFree(d_input_scales);
  cudaFree(d_output);

  // 5. Run Reference (FP32 GEMM)
  // Use FP32 reference as requested
  gemm_fp32_ref(input.data(), weights_fp32.data(), output_ref.data(), M, N, K);

  // 6. Compare
  float mse = 0.0f;
  float max_diff = 0.0f;
  for (size_t i = 0; i < output_cuda.size(); ++i) {
    float diff = output_cuda[i] - output_ref[i];
    mse += diff * diff;
    if (std::abs(diff) > max_diff) {
      max_diff = std::abs(diff);
    }
  }
  mse /= output_cuda.size();

  std::cout << "[Packed Block 16x16] M: " << M << " K: " << K << " N: " << N
            << " Group: " << scale_group_size << " MSE: " << mse
            << " MaxDiff: " << max_diff << std::endl;

  // Threshold calculation (same as CPU test)
  float input_min = -1.0f, input_max = 1.0f;
  float weight_min = -0.5f, weight_max = 0.5f;
  float input_range = input_max - input_min;
  float weight_range = weight_max - weight_min;
  float delta_input = input_range / 255.0f;
  float delta_weight = weight_range / 15.0f;
  float var_err_input = (delta_input * delta_input) / 12.0f;
  float var_err_weight = (delta_weight * delta_weight) / 12.0f;
  float mean_sq_input =
    (input_min * input_min + input_min * input_max + input_max * input_max) /
    3.0f;
  float mean_sq_weight = (weight_min * weight_min + weight_min * weight_max +
                          weight_max * weight_max) /
                         3.0f;
  float var_err_product =
    mean_sq_input * var_err_weight + mean_sq_weight * var_err_input;
  float expected_mse = K * var_err_product;
  float threshold = expected_mse * 2.0f;

  std::cout << "Expected MSE: " << expected_mse << " Threshold: " << threshold
            << std::endl;
  EXPECT_LT(mse, threshold);
}

#define DECLARE_CUDA_INT4_GEMM_PACKED_BLOCK_16_TEST(M, K, N, G)                \
  TEST(gemm_int4_cuda, packed_block_16_##M##_##K##_##N##_Group##G) {           \
    run_cuda_int4_gemm_packed_block_16_test_(M, K, N, G);                      \
  }

DECLARE_CUDA_INT4_GEMM_PACKED_BLOCK_16_TEST(28, 3072, 256, 32);
DECLARE_CUDA_INT4_GEMM_PACKED_BLOCK_16_TEST(28, 3072, 3072, 32);
DECLARE_CUDA_INT4_GEMM_PACKED_BLOCK_16_TEST(28, 3072, 8192, 32);
DECLARE_CUDA_INT4_GEMM_PACKED_BLOCK_16_TEST(28, 8192, 3072, 32);

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
