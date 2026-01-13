// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_int4_utils_cuda.cpp
 * @date	19 December 2025
 * @brief	Unit test for INT4 CUDA utilities
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Daekyoung Jung <dk11.jung@samsung.com>
 * @bug		No known bugs
 */

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <dequantize_cuda.h>
#include <gtest/gtest.h>
#include <int4_utils.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>


using namespace nntrainer;

/**
 * @brief Parameter structure for Int4UtilsCuda Test
 */
struct Int4UtilsCudaParams {
  size_t vocab_size;
  size_t seq_len;
  size_t embedding_dim;
};

/**
 * @brief Parameterized test fixture
 */
class Int4UtilsCudaTest : public testing::TestWithParam<Int4UtilsCudaParams> {};

/**
 * @brief Parameterized Test for dequantize_rows_cuda
 */
TEST_P(Int4UtilsCudaTest, DequantizeRowsBench) {
  auto param = GetParam();
  const size_t rows = param.vocab_size;
  const size_t cols = param.embedding_dim;
  const size_t group_size = 32;
  const size_t seq_len = param.seq_len;

  std::cout << "[Test Param] VocabSize: " << rows << ", SeqLen: " << seq_len
            << ", EmbeddingDim: " << cols << std::endl;

  // 1. Generate random weights
  std::vector<float> host_weights(rows * cols);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &w : host_weights)
    w = dist(gen);

  // 2. Quantize using host Int4Utils
  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> scales;
  Int4Utils::quantizeAndRepack(host_weights.data(), rows, cols, group_size,
                               quantized_weights, scales);

  // 3. Prepare GPU memory
  uint8_t *d_weights;
  uint16_t *d_scales;
  ASSERT_EQ(cudaMalloc(&d_weights, quantized_weights.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_scales, scales.size() * sizeof(uint16_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_weights, quantized_weights.data(),
                       quantized_weights.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_scales, scales.data(),
                       scales.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  // 4. Prepare indices for dequantization
  std::vector<float> host_indices(seq_len);
  std::uniform_int_distribution<size_t> row_dist(0, rows - 1);
  for (size_t i = 0; i < seq_len; ++i) {
    host_indices[i] = static_cast<float>(row_dist(gen));
  }

  float *d_indices;
  ASSERT_EQ(cudaMalloc(&d_indices, host_indices.size() * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_indices, host_indices.data(),
                       host_indices.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  // 5. Output buffer on GPU
  float *d_output;
  ASSERT_EQ(cudaMalloc(&d_output, host_indices.size() * cols * sizeof(float)),
            cudaSuccess);

  // 6. Performance Benchmark
  const int iterations = 10;

  // 6.1 CUDA Performance
  cudaEvent_t start, stop;
  ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

  float cuda_total_ms = 0.0f;
  for (int it = 0; it < iterations; ++it) {
    ASSERT_EQ(cudaEventRecord(start), cudaSuccess);
    Int4UtilsCuda::dequantize_rows_cuda(d_weights, d_scales, rows, cols,
                                        group_size, d_indices,
                                        host_indices.size(), d_output);
    ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

    if (it > 0) { // Skip warm-up
      float ms = 0;
      cudaEventElapsedTime(&ms, start, stop);
      cuda_total_ms += ms;
    }
  }
  float cuda_avg_ms = cuda_total_ms / (iterations - 1);

  // 6.2 CPU Performance
  double cpu_total_ms = 0.0;
  std::vector<float> cpu_ref_buffer(host_indices.size() * cols);

  for (int it = 0; it < iterations; ++it) {
    auto begin = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < host_indices.size(); ++i) {
      size_t row_idx = static_cast<size_t>(host_indices[i]);
      Int4Utils::dequantizePackedRow(quantized_weights.data(), scales.data(),
                                     rows, cols, group_size, row_idx,
                                     cpu_ref_buffer.data() + i * cols);
    }
    auto end = std::chrono::high_resolution_clock::now();

    if (it > 0) { // Skip warm-up
      cpu_total_ms +=
        std::chrono::duration<double, std::milli>(end - begin).count();
    }
  }
  double cpu_avg_ms = cpu_total_ms / (iterations - 1);

  // 6.3 Performance Summary Output
  std::cout << "\n================ Performance Summary ================"
            << std::endl;
  std::cout << "Vocab: " << rows << ", Indices: " << host_indices.size()
            << ", Cols: " << cols << std::endl;
  std::cout << "Iterations: " << iterations << " (Warm-up: 1)" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Average Latency (CUDA): " << cuda_avg_ms << " ms" << std::endl;
  std::cout << "Average Latency (CPU) : " << cpu_avg_ms << " ms" << std::endl;
  std::cout << "Speedup: " << (cpu_avg_ms / cuda_avg_ms) << "x" << std::endl;
  std::cout << "=====================================================\n"
            << std::endl;

  // 7. Verify results
  std::vector<float> cuda_results(host_indices.size() * cols);
  ASSERT_EQ(cudaMemcpy(cuda_results.data(), d_output,
                       cuda_results.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (size_t i = 0; i < std::min(host_indices.size(), (size_t)3); ++i) {
    size_t row_idx = static_cast<size_t>(host_indices[i]);
    std::vector<float> ref_row(cols);
    Int4Utils::dequantizePackedRow(quantized_weights.data(), scales.data(),
                                   rows, cols, group_size, row_idx,
                                   ref_row.data());

    std::cout << "[Token Index " << row_idx << "]" << std::endl;
    std::cout << "  CUDA: ";
    for (size_t j = 0; j < 5 && j < cols; ++j)
      std::cout << cuda_results[i * cols + j] << " ";
    std::cout << "\n  REF : ";
    for (size_t j = 0; j < 5 && j < cols; ++j)
      std::cout << ref_row[j] << " ";
    std::cout << std::endl;

    for (size_t j = 0; j < cols; ++j) {
      EXPECT_NEAR(cuda_results[i * cols + j], ref_row[j], 1e-4);
    }
  }

  // Cleanup
  cudaFree(d_weights);
  cudaFree(d_scales);
  cudaFree(d_indices);
  cudaFree(d_output);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

INSTANTIATE_TEST_SUITE_P(Int4UtilsCudaTests, Int4UtilsCudaTest,
                         testing::Values(Int4UtilsCudaParams{64, 5, 3072},
                                         Int4UtilsCudaParams{32000, 10, 3072},
                                         Int4UtilsCudaParams{105900, 1, 3072},
                                         Int4UtilsCudaParams{105900, 512,
                                                             3072}));

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
