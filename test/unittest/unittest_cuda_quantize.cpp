// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_cuda_quantize.cpp
 * @date   27 Nov 2025
 * @brief  Unit test for Q8_1 quantization operations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include "ggml_quantize_cpu.h"
#include "ggml_dequantize_cpu.h"
#include "ggml_cuda_common.h"
#include "ggml_quantize_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "   \
                << cudaGetErrorString(error) << std::endl;                     \
      FAIL();                                                                  \
    }                                                                          \
  } while (0)

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

/**
 * @brief Compute Mean Squared Error between two arrays
 */
static float compute_mse(const float* a, const float* b, int64_t size) {
  double sum = 0.0;
  for (int64_t i = 0; i < size; ++i) {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return static_cast<float>(sum / size);
}

/**
 * @brief Test Q8_1 quantization and dequantization round-trip
 */
TEST(nntrainer_CUDA_Quantize, q8_1_roundtrip_basic) {
  const int64_t size = 1024; // Must be multiple of 32
  
  // 1. Generate random FP32 array
  std::vector<float> original(size);
  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
  
  for (int64_t i = 0; i < size; ++i) {
    original[i] = dis(gen);
  }
  
  // 2. Quantize to Q8_1
  const int64_t num_blocks = size / QK8_1;
  std::vector<block_q8_1> quantized(num_blocks);
  quantize_row_q8_1_host(original.data(), quantized.data(), size);
  
  // 3. Dequantize back to FP32
  std::vector<float> dequantized(size);
  dequantize_row_q8_1_host(quantized.data(), dequantized.data(), size);
  
  // 4. Compute MSE and check it's within acceptable range
  float mse = compute_mse(original.data(), dequantized.data(), size);
  
  // Q8_1 uses 8-bit quantization, so we expect some loss
  // MSE should be small but non-zero due to quantization
  EXPECT_IN_RANGE(mse, 0.0f, 0.1f); // Adjust threshold based on expected precision
  
  std::cout << "Q8_1 Round-trip MSE: " << mse << std::endl;
}

/**
 * @brief Test Q8_1 with various sizes
 */
TEST(nntrainer_CUDA_Quantize, q8_1_roundtrip_various_sizes) {
  std::vector<int64_t> test_sizes = {32, 64, 128, 256, 512, 2048, 4096};
  
  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
  
  for (int64_t size : test_sizes) {
    std::vector<float> original(size);
    for (int64_t i = 0; i < size; ++i) {
      original[i] = dis(gen);
    }
    
    const int64_t num_blocks = size / QK8_1;
    std::vector<block_q8_1> quantized(num_blocks);
    quantize_row_q8_1_host(original.data(), quantized.data(), size);
    
    std::vector<float> dequantized(size);
    dequantize_row_q8_1_host(quantized.data(), dequantized.data(), size);
    
    float mse = compute_mse(original.data(), dequantized.data(), size);
    
    EXPECT_IN_RANGE(mse, 0.0f, 0.1f);
    
    std::cout << "Size " << size << " - MSE: " << mse << std::endl;
  }
}

/**
 * @brief Test Q8_1 with edge cases (zeros, small values, large values)
 */
TEST(nntrainer_CUDA_Quantize, q8_1_edge_cases) {
  const int64_t size = 128;
  
  // Test with all zeros
  {
    std::vector<float> original(size, 0.0f);
    const int64_t num_blocks = size / QK8_1;
    std::vector<block_q8_1> quantized(num_blocks);
    quantize_row_q8_1_host(original.data(), quantized.data(), size);
    
    std::vector<float> dequantized(size);
    dequantize_row_q8_1_host(quantized.data(), dequantized.data(), size);
    
    float mse = compute_mse(original.data(), dequantized.data(), size);
    EXPECT_FLOAT_EQ(mse, 0.0f);
  }
  
  // Test with very small values
  {
    std::vector<float> original(size);
    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dis(-0.01f, 0.01f);
    for (int64_t i = 0; i < size; ++i) {
      original[i] = dis(gen);
    }
    
    const int64_t num_blocks = size / QK8_1;
    std::vector<block_q8_1> quantized(num_blocks);
    quantize_row_q8_1_host(original.data(), quantized.data(), size);
    
    std::vector<float> dequantized(size);
    dequantize_row_q8_1_host(quantized.data(), dequantized.data(), size);
    
    float mse = compute_mse(original.data(), dequantized.data(), size);
    EXPECT_IN_RANGE(mse, 0.0f, 0.001f);
  }
  
  // Test with large values
  {
    std::vector<float> original(size);
    std::mt19937 gen(789);
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    for (int64_t i = 0; i < size; ++i) {
      original[i] = dis(gen);
    }
    
    const int64_t num_blocks = size / QK8_1;
    std::vector<block_q8_1> quantized(num_blocks);
    quantize_row_q8_1_host(original.data(), quantized.data(), size);
    
    std::vector<float> dequantized(size);
    dequantize_row_q8_1_host(quantized.data(), dequantized.data(), size);
    
    float mse = compute_mse(original.data(), dequantized.data(), size);
    EXPECT_IN_RANGE(mse, 0.0f, 10.0f); // Higher threshold for larger values
  }
}

/**
 * @brief Test CUDA Q8_1 quantization vs Host implementation
 */
TEST(nntrainer_CUDA_Quantize, q8_1_cuda_vs_host) {
  const int64_t size = 1024;
  
  // Generate random FP32 array
  std::vector<float> input_host(size);
  std::mt19937 gen(999);
  std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
  
  for (int64_t i = 0; i < size; ++i) {
    input_host[i] = dis(gen);
  }
  
  const int64_t num_blocks = size / QK8_1;
  
  // Host quantization
  std::vector<block_q8_1> quantized_host(num_blocks);
  quantize_row_q8_1_host(input_host.data(), quantized_host.data(), size);
  
  // CUDA quantization
  float *d_input = nullptr;
  block_q8_1 *d_quantized = nullptr;
  
  CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_quantized, num_blocks * sizeof(block_q8_1)));
  
  CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  
  quantize_row_q8_1_cuda(d_input, d_quantized, size, stream);
  
  CUDA_CHECK(cudaStreamSynchronize(stream));
  
  std::vector<block_q8_1> quantized_cuda(num_blocks);
  CUDA_CHECK(cudaMemcpy(quantized_cuda.data(), d_quantized, num_blocks * sizeof(block_q8_1), cudaMemcpyDeviceToHost));
  
  // Compare results
  int mismatches = 0;
  for (int64_t i = 0; i < num_blocks; ++i) {
    // Compare quantized values
    for (int j = 0; j < QK8_1; ++j) {
      if (quantized_host[i].qs[j] != quantized_cuda[i].qs[j]) {
        mismatches++;
      }
    }
    
    // Compare scale factors (d) - allow small FP16 differences
    uint16_t d_host = quantized_host[i].GGML_COMMON_AGGR_S.d;
    uint16_t d_cuda = quantized_cuda[i].GGML_COMMON_AGGR_S.d;
    if (d_host != d_cuda) {
      // Allow 1 ULP difference for FP16
      int diff = std::abs(static_cast<int>(d_host) - static_cast<int>(d_cuda));
      if (diff > 1) {
        mismatches++;
      }
    }
  }
  
  // Cleanup
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_quantized));
  
  // Expect very few or no mismatches (allowing for minor FP16 rounding differences)
  EXPECT_LE(mismatches, num_blocks * QK8_1 * 0.01); // Allow up to 1% mismatch
  
  std::cout << "CUDA vs Host mismatches: " << mismatches << " out of " << (num_blocks * (QK8_1 + 1)) << std::endl;
}

/**
 * @brief Performance benchmark for CUDA Q8_1 quantization
 */
TEST(nntrainer_CUDA_Quantize, q8_1_cuda_performance) {
  const int64_t size = 3072 * 1024;
  const int num_iterations = 10;
  
  // Generate random FP32 array
  std::vector<float> input_host(size);
  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
  
  for (int64_t i = 0; i < size; ++i) {
    input_host[i] = dis(gen);
  }
  
  const int64_t num_blocks = size / QK8_1;
  
  // Allocate device memory
  float *d_input = nullptr;
  block_q8_1 *d_quantized = nullptr;
  
  CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_quantized, num_blocks * sizeof(block_q8_1)));
  
  CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  
  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  std::vector<float> elapsed_times;
  elapsed_times.reserve(num_iterations - 1);
  
  for (int iter = 0; iter < num_iterations; ++iter) {
    if (iter == 0) {
      // Warm-up iteration (not measured)
      quantize_row_q8_1_cuda(d_input, d_quantized, size, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      // Measured iterations
      CUDA_CHECK(cudaEventRecord(start, stream));
      
      quantize_row_q8_1_cuda(d_input, d_quantized, size, stream);
      
      CUDA_CHECK(cudaEventRecord(stop, stream));
      CUDA_CHECK(cudaEventSynchronize(stop));
      
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
      elapsed_times.push_back(elapsed_ms);
    }
  }
  
  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_quantized));
  
  // Calculate statistics
  float total_time = 0.0f;
  float min_time = elapsed_times[0];
  float max_time = elapsed_times[0];
  
  for (float t : elapsed_times) {
    total_time += t;
    min_time = std::min(min_time, t);
    max_time = std::max(max_time, t);
  }
  
  float avg_time = total_time / elapsed_times.size();
  
  std::cout << "CUDA Q8_1 Quantization Performance (size=" << size << "):" << std::endl;
  std::cout << "  Average time: " << avg_time << " ms" << std::endl;
  std::cout << "  Min time:     " << min_time << " ms" << std::endl;
  std::cout << "  Max time:     " << max_time << " ms" << std::endl;
  std::cout << "  Throughput:   " << (size * sizeof(float) / (avg_time * 1e6)) << " GB/s" << std::endl;
  
  // Sanity check: time should be reasonable (not zero, not too large)
  EXPECT_GT(avg_time, 0.0f);
  EXPECT_LT(avg_time, 1000.0f); // Should complete in less than 1 second
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

