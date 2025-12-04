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

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#if defined(ENABLE_OPENCL)
#include "blas_kernels.h"
#include "cl_context.h"
#endif
#include "engine.h"
#include "fp16.h"
#include "ggml_cuda_common.h"
#include "ggml_dequantize_cpu.h"
#include "ggml_quantize_cpu.h"
#include "ggml_quantize_cuda.h"
#include "unittest_util.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
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
static float compute_mse(const float *a, const float *b, int64_t size) {
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
  EXPECT_IN_RANGE(mse, 0.0f,
                  0.1f); // Adjust threshold based on expected precision

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

  CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), size * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  quantize_row_q8_1_cuda(d_input, d_quantized, size, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<block_q8_1> quantized_cuda(num_blocks);
  CUDA_CHECK(cudaMemcpy(quantized_cuda.data(), d_quantized,
                        num_blocks * sizeof(block_q8_1),
                        cudaMemcpyDeviceToHost));

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

  // Expect very few or no mismatches (allowing for minor FP16 rounding
  // differences)
  EXPECT_LE(mismatches, num_blocks * QK8_1 * 0.01); // Allow up to 1% mismatch

  std::cout << "CUDA vs Host mismatches: " << mismatches << " out of "
            << (num_blocks * (QK8_1 + 1)) << std::endl;
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

  CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), size * sizeof(float),
                        cudaMemcpyHostToDevice));

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

  std::cout << "CUDA Q8_1 Quantization Performance (size=" << size
            << "):" << std::endl;
  std::cout << "  Average time: " << avg_time << " ms" << std::endl;
  std::cout << "  Min time:     " << min_time << " ms" << std::endl;
  std::cout << "  Max time:     " << max_time << " ms" << std::endl;
  std::cout << "  Throughput:   " << (size * sizeof(float) / (avg_time * 1e6))
            << " GB/s" << std::endl;

  // Sanity check: time should be reasonable (not zero, not too large)
  EXPECT_GT(avg_time, 0.0f);
  EXPECT_LT(avg_time, 1000.0f); // Should complete in less than 1 second
}

static void
run_int4_quantize_input_test_cuda_(const unsigned int M, const unsigned int K,
                                   const unsigned int quantization_group_size) {
  // Generate random FP32 input
  std::vector<float> input_host(M * K);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-2.0f, 2.0f);

  for (unsigned int i = 0; i < M * K; ++i) {
    input_host[i] = dis(gen);
  }

  // Convert to FP16 for GPU
  std::vector<uint16_t> input_fp16(M * K);
  for (unsigned int i = 0; i < M * K; ++i) {
    input_fp16[i] = nntrainer::compute_fp32_to_fp16(input_host[i]);
  }

  const unsigned int align_k =
    ((K + quantization_group_size - 1) / quantization_group_size) *
    quantization_group_size;
  const unsigned int groups_in_row = align_k / quantization_group_size;
  const unsigned int total_groups = M * groups_in_row;

  // CPU quantization
  std::vector<int8_t> ref_quantized_input(M * K);
  std::vector<uint16_t> ref_scales(total_groups * 2);
  nntrainer::cpu_quantize_input_int8_pad(
    input_host.data(), ref_quantized_input.data(), ref_scales.data(), M, K,
    quantization_group_size);

  // Allocate device memory
  void *d_input = nullptr;
  void *d_quantized_input = nullptr;
  void *d_scales = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_quantized_input, M * K * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_scales, total_groups * 2 * sizeof(uint16_t)));

  CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // CUDA quantization - 10 times to measure performance
  std::vector<float> cuda_elapsed_times;
  cuda_elapsed_times.reserve(9); // Exclude first iteration

  for (int iter = 0; iter < 10; ++iter) {
    if (iter == 0) {
      // First iteration - no timing
      quantize_input_int8_pad_cuda(d_input, d_quantized_input, d_scales, M, K,
                                   quantization_group_size, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      // Measured iterations
      CUDA_CHECK(cudaEventRecord(start, stream));

      quantize_input_int8_pad_cuda(d_input, d_quantized_input, d_scales, M, K,
                                   quantization_group_size, stream);

      CUDA_CHECK(cudaEventRecord(stop, stream));
      CUDA_CHECK(cudaEventSynchronize(stop));

      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
      cuda_elapsed_times.push_back(elapsed_ms);

      // Copy results back for verification
      std::vector<int8_t> cuda_quantized_input_iter(M * K);
      std::vector<uint16_t> cuda_scales_iter(total_groups * 2);

      CUDA_CHECK(cudaMemcpy(cuda_quantized_input_iter.data(), d_quantized_input,
                            M * K * sizeof(int8_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(cuda_scales_iter.data(), d_scales,
                            total_groups * 2 * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));

      // Verify results for this iteration
      int mismatch_count_iter = 0;
      for (unsigned int i = 0; i < M * K; ++i) {
        if (cuda_quantized_input_iter[i] != ref_quantized_input[i]) {
          mismatch_count_iter++;
        }
      }

      float mismatch_ratio_iter = (float)mismatch_count_iter / (M * K);
      EXPECT_LE(mismatch_ratio_iter, 0.01f); // Allow up to 1% mismatch

      // Compare scales (MSE) for this iteration
      float mse_scales_iter = 0.0f;
      for (unsigned int i = 0; i < total_groups; ++i) {
        float cuda_scale =
          nntrainer::compute_fp16_to_fp32(cuda_scales_iter[i * 2]);
        float ref_scale = nntrainer::compute_fp16_to_fp32(ref_scales[i * 2]);
        mse_scales_iter += (cuda_scale - ref_scale) * (cuda_scale - ref_scale);
      }
      mse_scales_iter /= total_groups;
      EXPECT_LE(mse_scales_iter, 1e-5f);
    }
  }

  // Calculate average CUDA time
  float total_cuda_time = 0.0f;
  for (float t : cuda_elapsed_times) {
    total_cuda_time += t;
  }
  float avg_cuda_time = total_cuda_time / cuda_elapsed_times.size();

  // Calculate throughput (GB/s)
  float data_size_gb = (M * K * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
  float throughput = data_size_gb / (avg_cuda_time / 1000.0f); // GB/s

  std::cout << "CUDA INT4 Quantization Average Time (M=" << M << ", K=" << K
            << "): " << avg_cuda_time << " ms" << std::endl;
  std::cout << "CUDA INT4 Quantization Throughput (M=" << M << ", K=" << K
            << "): " << throughput << " GB/s" << std::endl;

  // Copy final results back
  std::vector<int8_t> cuda_quantized_input(M * K);
  std::vector<uint16_t> cuda_scales(total_groups * 2);

  CUDA_CHECK(cudaMemcpy(cuda_quantized_input.data(), d_quantized_input,
                        M * K * sizeof(int8_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cuda_scales.data(), d_scales,
                        total_groups * 2 * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));

  // Compare quantized data
  int mismatch_count = 0;
  for (unsigned int i = 0; i < M * K; ++i) {
    if (cuda_quantized_input[i] != ref_quantized_input[i]) {
      mismatch_count++;
    }
  }

  float mismatch_ratio = (float)mismatch_count / (M * K);
  std::cout << "INT4 quantization mismatch count (" << M << "x" << K
            << "): " << mismatch_count << " (" << mismatch_ratio * 100.0f
            << "%)" << std::endl;

  // Compare scales (MSE)
  float mse_scales = 0.0f;
  for (unsigned int i = 0; i < total_groups; ++i) {
    float cuda_scale = nntrainer::compute_fp16_to_fp32(cuda_scales[i * 2]);
    float ref_scale = nntrainer::compute_fp16_to_fp32(ref_scales[i * 2]);
    mse_scales += (cuda_scale - ref_scale) * (cuda_scale - ref_scale);
  }
  mse_scales /= total_groups;
  std::cout << "Scales MSE: " << mse_scales << std::endl;
  // Cleanup
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_quantized_input));
  CUDA_CHECK(cudaFree(d_scales));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Assertions
  EXPECT_LE(mismatch_ratio, 0.01f); // Allow up to 1% mismatch
  EXPECT_LE(mse_scales, 1e-5f);
}

#define DECLARE_int4_quantize_input_test_cuda_M_K_G(M, K, G)                   \
  TEST(nntrainer_CUDA_Quantize, int4_quantize_input_test_##M##_##K##_##G) {    \
    run_int4_quantize_input_test_cuda_(M, K, G);                               \
  }

DECLARE_int4_quantize_input_test_cuda_M_K_G(32, 3072, 32);
DECLARE_int4_quantize_input_test_cuda_M_K_G(63, 3072, 32);
DECLARE_int4_quantize_input_test_cuda_M_K_G(128, 3072, 32);
DECLARE_int4_quantize_input_test_cuda_M_K_G(256, 3072, 32);
DECLARE_int4_quantize_input_test_cuda_M_K_G(512, 3072, 32);
DECLARE_int4_quantize_input_test_cuda_M_K_G(1024, 3072, 32);

/**
 * @brief Performance benchmark for CUDA INT4 quantization
 */
TEST(nntrainer_CUDA_Quantize, int4_quantize_input_pad_cuda_performance) {
  const unsigned int M = 128;
  const unsigned int K = 3072;
  const unsigned int quantization_group_size = 32;
  const int num_iterations = 10;

  // Generate random FP32 input
  std::vector<float> input_host(M * K);
  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> dis(-2.0f, 2.0f);

  for (unsigned int i = 0; i < M * K; ++i) {
    input_host[i] = dis(gen);
  }

  // Convert to FP16 for GPU
  std::vector<uint16_t> input_fp16(M * K);
  for (unsigned int i = 0; i < M * K; ++i) {
    input_fp16[i] = nntrainer::compute_fp32_to_fp16(input_host[i]);
  }

  const unsigned int align_k =
    ((K + quantization_group_size - 1) / quantization_group_size) *
    quantization_group_size;
  const unsigned int groups_in_row = align_k / quantization_group_size;
  const unsigned int total_groups = M * groups_in_row;

  // Allocate device memory
  void *d_input = nullptr;
  void *d_quantized_input = nullptr;
  void *d_scales = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_quantized_input, M * K * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_scales, total_groups * 2 * sizeof(uint16_t)));

  CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice));

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
      quantize_input_int8_pad_cuda(d_input, d_quantized_input, d_scales, M, K,
                                   quantization_group_size, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      // Measured iterations
      CUDA_CHECK(cudaEventRecord(start, stream));

      quantize_input_int8_pad_cuda(d_input, d_quantized_input, d_scales, M, K,
                                   quantization_group_size, stream);

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
  CUDA_CHECK(cudaFree(d_quantized_input));
  CUDA_CHECK(cudaFree(d_scales));

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

  std::cout << "CUDA INT4 Quantization Performance (M=" << M << ", K=" << K
            << "):" << std::endl;
  std::cout << "  Average time: " << avg_time << " ms" << std::endl;
  std::cout << "  Min time:     " << min_time << " ms" << std::endl;
  std::cout << "  Max time:     " << max_time << " ms" << std::endl;
  std::cout << "  Throughput:   "
            << (M * K * sizeof(uint16_t) / (avg_time * 1e6)) << " GB/s"
            << std::endl;

  // Sanity check
  EXPECT_GT(avg_time, 0.0f);
  EXPECT_LT(avg_time, 100.0f);
}

static void
run_cuda_vs_openvino_quantize_test(const unsigned int M, const unsigned int K,
                                   const unsigned int quantization_group_size) {
#ifdef ENABLE_OPENCL
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  if (!blas_cc) {
    GTEST_SKIP() << "OpenCL context not available";
    return;
  }

  // Allocate OpenCL memory (SVM)
  // Input is FP16 (uint16_t)
  size_t input_size_bytes = M * K * sizeof(uint16_t);
  uint16_t *input_cl = (uint16_t *)nntrainer::allocateSVM(input_size_bytes);

  // Output is int8
  size_t output_size_bytes = M * K * sizeof(int8_t);
  int8_t *quantized_cl = (int8_t *)nntrainer::allocateSVM(output_size_bytes);

  const unsigned int align_k =
    ((K + quantization_group_size - 1) / quantization_group_size) *
    quantization_group_size;
  const unsigned int groups_in_row = align_k / quantization_group_size;
  const unsigned int total_groups = M * groups_in_row;

  size_t scales_size_bytes = total_groups * 2 * sizeof(uint16_t);
  uint16_t *scales_cl = (uint16_t *)nntrainer::allocateSVM(scales_size_bytes);

  // Generate random input (FP32)
  std::vector<float> input_host(M * K);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
  for (auto &v : input_host)
    v = dis(gen);

  // Convert to FP16 for OpenCL input
  for (unsigned int i = 0; i < M * K; ++i) {
    input_cl[i] = nntrainer::compute_fp32_to_fp16(input_host[i]);
  }

  // Run OpenCL quantization
  nntrainer::openvino_quantize_input_int4_pad(input_cl, quantized_cl, scales_cl,
                                              M, K, quantization_group_size);

  // Run CUDA quantization
  // Allocate CUDA memory
  void *d_input = nullptr;
  void *d_quantized = nullptr;
  void *d_scales = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, M * K * sizeof(float))); // Input is FP16
  CUDA_CHECK(cudaMalloc(&d_quantized, output_size_bytes));
  CUDA_CHECK(cudaMalloc(&d_scales, scales_size_bytes));

  // Copy input to CUDA
  CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  quantize_input_int8_pad_cuda(d_input, d_quantized, d_scales, M, K,
                               quantization_group_size, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy results back
  std::vector<int8_t> cuda_quantized(M * K);
  std::vector<uint16_t> cuda_scales(total_groups * 2);

  CUDA_CHECK(cudaMemcpy(cuda_quantized.data(), d_quantized, output_size_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cuda_scales.data(), d_scales, scales_size_bytes,
                        cudaMemcpyDeviceToHost));

  // Compare
  int mismatch_count = 0;
  for (unsigned int i = 0; i < M * K; ++i) {
    if (cuda_quantized[i] != quantized_cl[i]) {
      mismatch_count++;
    }
  }

  float mismatch_ratio = (float)mismatch_count / (M * K);
  std::cout << "Mismatch count: " << mismatch_count << " ("
            << mismatch_ratio * 100.0f << "%)" << std::endl;
  EXPECT_LE(mismatch_ratio, 0.01f);

  float mse_scales = 0.0f;
  for (unsigned int i = 0; i < total_groups; ++i) {
    float val = nntrainer::compute_fp16_to_fp32(cuda_scales[i * 2]);
    float ref = nntrainer::compute_fp16_to_fp32(scales_cl[i * 2]);
    mse_scales += (val - ref) * (val - ref);
  }
  mse_scales /= total_groups;
  std::cout << "Scales MSE: " << mse_scales << std::endl;
  EXPECT_LE(mse_scales, 1e-5f);

  // Cleanup
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_quantized));
  CUDA_CHECK(cudaFree(d_scales));

  nntrainer::freeSVM(input_cl);
  nntrainer::freeSVM(quantized_cl);
  nntrainer::freeSVM(scales_cl);

#else
  GTEST_SKIP() << "OpenCL not enabled";
#endif
}

#define DECLARE_CUDA_VS_OPENVINO_TEST(M, K, G)                                 \
  TEST(nntrainer_CUDA_Quantize_OpenVINO_Ref, test_##M##_##K##_##G) {           \
    run_cuda_vs_openvino_quantize_test(M, K, G);                               \
  }

DECLARE_CUDA_VS_OPENVINO_TEST(32, 3072, 32);
DECLARE_CUDA_VS_OPENVINO_TEST(128, 3072, 32);
DECLARE_CUDA_VS_OPENVINO_TEST(1024, 3072, 32);

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
