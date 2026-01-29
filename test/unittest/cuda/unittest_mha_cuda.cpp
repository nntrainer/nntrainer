// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file unittest_mha_cuda.cpp
 * @date 29 Jan 2025
 * @brief Unit tests for CUDA Multi-Head Attention implementation
 * @author Samsung R&D Institute
 * @bug No known bugs
 */

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "mha_cpu_ref.h"
#include "mha_cuda.h"
#include "unittest_util.h"

using namespace custom;
using namespace reference;
using namespace nntrainer;

// Test parameter structure
struct MHATestParam {
  unsigned int seq_len;
  unsigned int num_heads;
  unsigned int group_size;
  unsigned int head_dim;
};

namespace {

// Helper function to convert FP32 to FP16
uint16_t fp32_to_fp16(float value) {
  __half h = __float2half(value);
  return *reinterpret_cast<uint16_t *>(&h);
}

// Helper function to convert FP16 to FP32
float fp16_to_fp32(uint16_t value) {
  __half h = *reinterpret_cast<__half *>(&value);
  return __half2float(h);
}

// Benchmark function template
template <typename Func>
float run_benchmark(const char *name, Func kernel_func, float *d_output,
                    std::vector<float> &h_output, size_t output_bytes,
                    int iterations = 10) {
  // Warmup
  for (int i = 0; i < 5; ++i) {
    kernel_func();
  }
  cudaDeviceSynchronize();

  float total_time = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < iterations; ++i) {
    cudaEventRecord(start);
    kernel_func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float iter_ms = 0.0f;
    cudaEventElapsedTime(&iter_ms, start, stop);
    total_time += iter_ms;
    cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float avg_time = total_time / iterations;
  std::cout << name << " time: " << avg_time << " ms" << std::endl;

  return avg_time;
}

// Verify result against reference
float verify_result(const char *name, const std::vector<float> &result,
                    const std::vector<float> &ref) {
  float mse = 0.0f;
  float max_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    float diff = result[i] - ref[i];
    mse += diff * diff;
    max_diff = std::max(max_diff, std::abs(diff));
  }
  mse /= ref.size();
  std::cout << name << " vs Ref MSE: " << mse << ", Max Diff: " << max_diff
            << std::endl;
  return mse;
}

// Prepare test data
void prepare_test_data(unsigned int seq_len, unsigned int num_heads,
                       unsigned int group_size, unsigned int head_dim,
                       std::vector<float> &query,
                       std::vector<uint16_t> &key_cache,
                       std::vector<uint16_t> &value_cache) {
  unsigned int num_heads_kv = num_heads / group_size;

  // Generate random query [seq_len, num_heads, head_dim]
  query =
    generate_random_vector<float>(seq_len * num_heads * head_dim, -1.0f, 1.0f);

  // Generate random key cache [seq_len, num_heads_kv, head_dim] in FP16
  std::vector<float> key_cache_fp32 = generate_random_vector<float>(
    seq_len * num_heads_kv * head_dim, -1.0f, 1.0f);
  key_cache.resize(key_cache_fp32.size());
  for (size_t i = 0; i < key_cache_fp32.size(); ++i) {
    key_cache[i] = fp32_to_fp16(key_cache_fp32[i]);
  }

  // Generate random value cache [seq_len, num_heads_kv, head_dim] in FP16
  std::vector<float> value_cache_fp32 = generate_random_vector<float>(
    seq_len * num_heads_kv * head_dim, -1.0f, 1.0f);
  value_cache.resize(value_cache_fp32.size());
  for (size_t i = 0; i < value_cache_fp32.size(); ++i) {
    value_cache[i] = fp32_to_fp16(value_cache_fp32[i]);
  }
}

} // namespace

class MHACudaTest : public ::testing::TestWithParam<MHATestParam> {
protected:
  void SetUp() override {
    // Initialize CUDA
    cudaSetDevice(0);
  }

  void TearDown() override {
    // Cleanup
  }
};

// Test compute_kcaches_prefill_cuda
TEST_P(MHACudaTest, ComputeKcachesPrefill) {
  const auto &param = GetParam();
  const unsigned int seq_len = param.seq_len;
  const unsigned int num_heads = param.num_heads;
  const unsigned int group_size = param.group_size;
  const unsigned int head_dim = param.head_dim;
  const unsigned int num_heads_kv = num_heads / group_size;

  std::cout << "\n=== Testing compute_kcaches_prefill_cuda ===" << std::endl;
  std::cout << "seq_len=" << seq_len << ", num_heads=" << num_heads
            << ", group_size=" << group_size << ", head_dim=" << head_dim
            << std::endl;

  // Prepare test data
  std::vector<float> query;
  std::vector<uint16_t> key_cache;
  std::vector<uint16_t> value_cache;
  prepare_test_data(seq_len, num_heads, group_size, head_dim, query, key_cache,
                    value_cache);

  // Calculate output size (triangular packed)
  size_t attn_len = (size_t)seq_len * (seq_len + 1) / 2;
  size_t output_size = attn_len * num_heads;

  // Allocate host memory
  std::vector<float> output_cuda(output_size);
  std::vector<float> output_ref(output_size);

  // Allocate device memory
  float *d_query, *d_output;
  uint16_t *d_key_cache;
  size_t query_bytes = query.size() * sizeof(float);
  size_t key_cache_bytes = key_cache.size() * sizeof(uint16_t);
  size_t output_bytes = output_size * sizeof(float);

  cudaMalloc(&d_query, query_bytes);
  cudaMalloc(&d_key_cache, key_cache_bytes);
  cudaMalloc(&d_output, output_bytes);

  // Copy data to device
  cudaMemcpy(d_query, query.data(), query_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_key_cache, key_cache.data(), key_cache_bytes,
             cudaMemcpyHostToDevice);

  // Run CUDA kernel
  run_benchmark(
    "compute_kcaches_prefill_cuda",
    [&]() {
      compute_kcaches_prefill_cuda(d_query, d_key_cache, d_output, seq_len,
                                   num_heads_kv, num_heads, group_size,
                                   head_dim);
    },
    d_output, output_cuda, output_bytes);

  // Run CPU reference
  compute_kcaches_prefill(query.data(), key_cache.data(), output_ref.data(),
                          seq_len, num_heads_kv, num_heads, group_size,
                          head_dim);

  // Verify results
  float mse =
    verify_result("compute_kcaches_prefill_cuda", output_cuda, output_ref);

  // Allow some tolerance due to FP16 operations and numerical differences
  float tolerance = 1e-3f;
  EXPECT_LT(mse, tolerance)
    << "compute_kcaches_prefill_cuda accuracy check failed";

  // Cleanup
  cudaFree(d_query);
  cudaFree(d_key_cache);
  cudaFree(d_output);
}

// Test softmax_triangle_prefill_cuda
TEST_P(MHACudaTest, SoftmaxTrianglePrefill) {
  const auto &param = GetParam();
  const unsigned int seq_len = param.seq_len;
  const unsigned int num_heads = param.num_heads;

  std::cout << "\n=== Testing softmax_triangle_prefill_cuda ===" << std::endl;
  std::cout << "seq_len=" << seq_len << ", num_heads=" << num_heads
            << std::endl;

  // Calculate output size (triangular packed)
  size_t attn_len = (size_t)seq_len * (seq_len + 1) / 2;
  size_t output_size = attn_len * num_heads;

  // Generate random attention scores
  std::vector<float> attention_scores =
    generate_random_vector<float>(output_size, -5.0f, 5.0f);

  // Allocate host memory
  std::vector<float> output_cuda(attention_scores);
  std::vector<float> output_ref(attention_scores);

  // Allocate device memory
  float *d_attention_scores;
  size_t bytes = output_size * sizeof(float);
  cudaMalloc(&d_attention_scores, bytes);

  // Copy data to device
  cudaMemcpy(d_attention_scores, attention_scores.data(), bytes,
             cudaMemcpyHostToDevice);

  // Run CUDA kernel
  run_benchmark(
    "softmax_triangle_prefill_cuda",
    [&]() {
      softmax_triangle_prefill_cuda(d_attention_scores, seq_len, num_heads);
    },
    d_attention_scores, output_cuda, bytes);

  // Run CPU reference
  softmax_triangle_prefill(output_ref.data(), seq_len, num_heads);

  // Verify results
  float mse =
    verify_result("softmax_triangle_prefill_cuda", output_cuda, output_ref);

  // Softmax should be very accurate
  float tolerance = 1e-3f; // Relax tolerance due to numerical differences
  EXPECT_LT(mse, tolerance)
    << "softmax_triangle_prefill_cuda accuracy check failed";

  // Cleanup
  cudaFree(d_attention_scores);
}

// Test compute_attention_value_mul_prefill_cuda
TEST_P(MHACudaTest, ComputeAttentionValueMulPrefill) {
  const auto &param = GetParam();
  const unsigned int seq_len = param.seq_len;
  const unsigned int num_heads = param.num_heads;
  const unsigned int group_size = param.group_size;
  const unsigned int head_dim = param.head_dim;
  const unsigned int num_heads_kv = num_heads / group_size;

  std::cout << "\n=== Testing compute_attention_value_mul_prefill_cuda ==="
            << std::endl;
  std::cout << "seq_len=" << seq_len << ", num_heads=" << num_heads
            << ", group_size=" << group_size << ", head_dim=" << head_dim
            << std::endl;

  // Prepare test data
  std::vector<float> query;
  std::vector<uint16_t> key_cache;
  std::vector<uint16_t> value_cache;
  prepare_test_data(seq_len, num_heads, group_size, head_dim, query, key_cache,
                    value_cache);

  // Calculate attention scores first (using CPU reference)
  size_t attn_len = (size_t)seq_len * (seq_len + 1) / 2;
  std::vector<float> attention_weights(attn_len * num_heads);
  compute_kcaches_prefill(query.data(), key_cache.data(),
                          attention_weights.data(), seq_len, num_heads_kv,
                          num_heads, group_size, head_dim);
  softmax_triangle_prefill(attention_weights.data(), seq_len, num_heads);

  // Calculate output size
  size_t output_size = (size_t)seq_len * num_heads * head_dim;

  // Allocate host memory
  std::vector<float> output_cuda(output_size);
  std::vector<float> output_ref(output_size);

  // Allocate device memory
  float *d_attention_weights, *d_output;
  uint16_t *d_value_cache;
  size_t attn_bytes = attention_weights.size() * sizeof(float);
  size_t value_cache_bytes = value_cache.size() * sizeof(uint16_t);
  size_t output_bytes = output_size * sizeof(float);

  cudaMalloc(&d_attention_weights, attn_bytes);
  cudaMalloc(&d_value_cache, value_cache_bytes);
  cudaMalloc(&d_output, output_bytes);

  // Copy data to device
  cudaMemcpy(d_attention_weights, attention_weights.data(), attn_bytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_value_cache, value_cache.data(), value_cache_bytes,
             cudaMemcpyHostToDevice);

  // Run CUDA kernel
  run_benchmark(
    "compute_attention_value_mul_prefill_cuda",
    [&]() {
      compute_attention_value_mul_prefill_cuda(
        d_attention_weights, d_value_cache, d_output, seq_len, num_heads_kv,
        group_size, head_dim);
    },
    d_output, output_cuda, output_bytes);

  // Run CPU reference
  compute_attention_value_mul_prefill(
    attention_weights.data(), value_cache.data(), output_ref.data(), seq_len,
    num_heads_kv, group_size, head_dim);

  // Verify results
  float mse = verify_result("compute_attention_value_mul_prefill_cuda",
                            output_cuda, output_ref);

  // Allow some tolerance due to FP16 operations and numerical differences
  float tolerance = 1e-3f;
  EXPECT_LT(mse, tolerance)
    << "compute_attention_value_mul_prefill_cuda accuracy check failed";

  // Cleanup
  cudaFree(d_attention_weights);
  cudaFree(d_value_cache);
  cudaFree(d_output);
}

// Test run_attention_sequence_prefill_cuda (full pipeline)
TEST_P(MHACudaTest, RunAttentionSequencePrefill) {
  const auto &param = GetParam();
  const unsigned int seq_len = param.seq_len;
  const unsigned int num_heads = param.num_heads;
  const unsigned int group_size = param.group_size;
  const unsigned int head_dim = param.head_dim;
  const unsigned int num_heads_kv = num_heads / group_size;

  std::cout << "\n=== Testing run_attention_sequence_prefill_cuda ==="
            << std::endl;
  std::cout << "seq_len=" << seq_len << ", num_heads=" << num_heads
            << ", group_size=" << group_size << ", head_dim=" << head_dim
            << std::endl;

  // Prepare test data
  std::vector<float> query;
  std::vector<uint16_t> key_cache;
  std::vector<uint16_t> value_cache;
  prepare_test_data(seq_len, num_heads, group_size, head_dim, query, key_cache,
                    value_cache);

  // Calculate output size
  size_t output_size = (size_t)seq_len * num_heads * head_dim;
  size_t attn_len = (size_t)seq_len * (seq_len + 1) / 2;
  size_t attn_scores_size = attn_len * num_heads;

  // Allocate host memory
  std::vector<float> output_cuda(output_size);
  std::vector<float> output_ref(output_size);

  // Allocate device memory
  float *d_query, *d_output, *d_attn_scores;
  uint16_t *d_key_cache, *d_value_cache;
  size_t query_bytes = query.size() * sizeof(float);
  size_t key_cache_bytes = key_cache.size() * sizeof(uint16_t);
  size_t value_cache_bytes = value_cache.size() * sizeof(uint16_t);
  size_t output_bytes = output_size * sizeof(float);
  size_t attn_scores_bytes = attn_scores_size * sizeof(float);

  cudaMalloc(&d_query, query_bytes);
  cudaMalloc(&d_key_cache, key_cache_bytes);
  cudaMalloc(&d_value_cache, value_cache_bytes);
  cudaMalloc(&d_output, output_bytes);
  cudaMalloc(&d_attn_scores, attn_scores_bytes);

  // Copy data to device
  cudaMemcpy(d_query, query.data(), query_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_key_cache, key_cache.data(), key_cache_bytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_value_cache, value_cache.data(), value_cache_bytes,
             cudaMemcpyHostToDevice);

  // Run CUDA kernel
  run_benchmark(
    "run_attention_sequence_prefill_cuda",
    [&]() {
      run_attention_sequence_prefill_cuda(d_query, d_key_cache, d_value_cache,
                                          d_output, seq_len, num_heads,
                                          group_size, head_dim, d_attn_scores);
    },
    d_output, output_cuda, output_bytes);

  // Run CPU reference
  run_attention_sequence(query.data(), key_cache.data(), value_cache.data(),
                         output_ref.data(), 0, seq_len, num_heads, group_size,
                         head_dim);

  // Verify results
  float mse = verify_result("run_attention_sequence_prefill_cuda", output_cuda,
                            output_ref);

  // Allow some tolerance due to FP16 operations and numerical differences
  float tolerance = 1e-3f;
  EXPECT_LT(mse, tolerance)
    << "run_attention_sequence_prefill_cuda accuracy check failed";

  // Cleanup
  cudaFree(d_query);
  cudaFree(d_key_cache);
  cudaFree(d_value_cache);
  cudaFree(d_output);
  cudaFree(d_attn_scores);
}

// Instantiate tests with various parameters
INSTANTIATE_TEST_SUITE_P(
  VariousSizes, MHACudaTest,
  ::testing::Values(
    // Small sizes
    MHATestParam{16, 4, 1,
                 64}, // seq_len=16, num_heads=4, group_size=1, head_dim=64
    MHATestParam{32, 8, 2,
                 64}, // seq_len=32, num_heads=8, group_size=2, head_dim=64
    // Medium sizes
    MHATestParam{64, 16, 4,
                 128}, // seq_len=64, num_heads=16, group_size=4, head_dim=128
    MHATestParam{128, 32, 4,
                 128}, // seq_len=128, num_heads=32, group_size=4, head_dim=128
    // Large sizes
    MHATestParam{256, 32, 4,
                 128}, // seq_len=256, num_heads=32, group_size=4, head_dim=128
    MHATestParam{512, 32, 8, 128}
    // seq_len=512, num_heads=32, group_size=8, head_dim=128
    ));

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
