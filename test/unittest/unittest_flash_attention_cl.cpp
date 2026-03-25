// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	unittest_flash_attention_cl.cpp
 * @date	24 March 2026
 * @brief	Test setup for flash attention OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Anup Dhakal <anup.dhakal@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <fstream>
#include <gtest/gtest.h>
#include <type_traits>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <flash_attention.h>
#include <cl_context.h>
#include <layer_context.h>
#include <tensor.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

using namespace nntrainer;

/**
 * @brief Test flash attention kernel implementation for GQA
 */
static void run_flash_attention_test(const int seqlen_q, const int seqlen_k,
                                        const int head_dim, const int num_heads_q,
                                        const int num_heads_kv, const int batch);

/**
 * @brief Test case for GQA configuration: head_dim=128, q_len=512, kv_len=512, n_heads_q=16, n_heads_kv=8, batch=1
 */
TEST(nntrainer_opencl_kernels_flash_attention, flash_attention_test_gqa_512_512_128_16_8) {
  // For GQA, we need to adjust the test to handle different numbers of query and key/value heads
  const int seqlen_q = 512;
  const int seqlen_k = 512;
  const int head_dim = 128;
  const int num_heads_q = 16;
  const int num_heads_kv = 8;
  const int batch = 1;
  
  // Run GQA test with the above configuration
  run_flash_attention_test(seqlen_q, seqlen_k, head_dim, num_heads_q, num_heads_kv, batch);
}

/**
 * @brief Test case for GQA configuration: head_dim=128, q_len=1, kv_len=512, n_heads_q=16, n_heads_kv=8, batch=1
 */
TEST(nntrainer_opencl_kernels_flash_attention, flash_attention_test_gqa_1_512_128_16_8) {
  // For GQA, we need to adjust the test to handle different numbers of query and key/value heads
  const int seqlen_q = 1;
  const int seqlen_k = 512;
  const int head_dim = 128;
  const int num_heads_q = 16;
  const int num_heads_kv = 8;
  const int batch = 1;
  
  // Run GQA test with the above configuration
  run_flash_attention_test(seqlen_q, seqlen_k, head_dim, num_heads_q, num_heads_kv, batch);
}

/**
 * @brief Test flash attention kernel implementation for GQA
 */
static void run_flash_attention_test(const int seqlen_q, const int seqlen_k,
                                        const int head_dim, const int num_heads_q,
                                        const int num_heads_kv, const int batch) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const float alpha = 1e-3f;
  const int MOD = 10;

  // Allocate host memory for GQA
  // Query: batch * num_heads_q * seqlen_q * head_dim
  // Key/Value: batch * num_heads_kv * seqlen_k * head_dim
  // Output: batch * num_heads_q * seqlen_q * head_dim
  std::vector<float> query_h(batch * num_heads_q * seqlen_q * head_dim);
  std::vector<float> key_h(batch * num_heads_kv * seqlen_k * head_dim);
  std::vector<float> value_h(batch * num_heads_kv * seqlen_k * head_dim);
  std::vector<float> output_h_ref(batch * num_heads_q * seqlen_q * head_dim, 0.0f);
  std::vector<float> output_h_gpu(batch * num_heads_q * seqlen_q * head_dim, 0.0f);

  // Initialize input data
  for (size_t i = 0; i < query_h.size(); ++i) {
    query_h[i] = ((static_cast<float>(i % MOD) - MOD/2.0f) * alpha);
  }
  for (size_t i = 0; i < key_h.size(); ++i) {
    key_h[i] = ((static_cast<float>(i % MOD) - MOD/2.0f) * alpha);
  }
  for (size_t i = 0; i < value_h.size(); ++i) {
    value_h[i] = ((static_cast<float>(i % MOD) - MOD/2.0f) * alpha);
  }

  // Compute reference output for GQA
  auto cpu_t1 = std::chrono::high_resolution_clock::now();
  flash_attention_cpu(query_h.data(), key_h.data(), value_h.data(),
                                output_h_ref.data(), seqlen_q, seqlen_k, head_dim,
                                num_heads_q, num_heads_kv, batch, scale);
  auto cpu_t2 = std::chrono::high_resolution_clock::now();
  auto cpu_dt = std::chrono::duration_cast<std::chrono::microseconds>(cpu_t2 - cpu_t1).count();

  // Allocate GPU memory
  float *query_d = (float *)allocateSVM(batch * num_heads_q * seqlen_q * head_dim * sizeof(float));
  float *key_d = (float *)allocateSVM(batch * num_heads_kv * seqlen_k * head_dim * sizeof(float));
  float *value_d = (float *)allocateSVM(batch * num_heads_kv * seqlen_k * head_dim * sizeof(float));
  float *output_d = (float *)allocateSVM(batch * num_heads_q * seqlen_q * head_dim * sizeof(float));

  // Copy data to GPU
  blas_cc->command_queue_inst_.enqueueSVMMap(query_d, batch * num_heads_q * seqlen_q * head_dim * sizeof(float), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(key_d, batch * num_heads_kv * seqlen_k * head_dim * sizeof(float), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(value_d, batch * num_heads_kv * seqlen_k * head_dim * sizeof(float), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(output_d, batch * num_heads_q * seqlen_q * head_dim * sizeof(float), false);

  for (size_t i = 0; i < query_h.size(); ++i) {
    query_d[i] = query_h[i];
  }
  for (size_t i = 0; i < key_h.size(); ++i) {
    key_d[i] = key_h[i];
  }
  for (size_t i = 0; i < value_h.size(); ++i) {
    value_d[i] = value_h[i];
  }

  blas_cc->command_queue_inst_.enqueueSVMUnmap(query_d);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(key_d);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(value_d);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(output_d);

  // Run standard GPU kernel with GQA support
  auto t1 = std::chrono::high_resolution_clock::now();
  nntrainer::flash_attention_fp32_cl(query_d, key_d, value_d, output_d, 
                                 seqlen_q, seqlen_k, head_dim, num_heads_q, num_heads_kv, batch, scale);
  
  auto t2 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  std::cout << "Flash Attention GQA Standard GPU time: " << gpu_dt << " μs" << std::endl;
  std::cout << "Flash Attention GQA CPU time: " << cpu_dt << " μs" << std::endl;
  std::cout << "GQA Standard Speedup: " << (float)cpu_dt / (float)gpu_dt << "x" << std::endl;

  // Copy result back to host
  blas_cc->command_queue_inst_.enqueueSVMMap(output_d, batch * num_heads_q * seqlen_q * head_dim * sizeof(float), true);
  for (size_t i = 0; i < output_h_gpu.size(); ++i) {
    output_h_gpu[i] = output_d[i];
  }
  blas_cc->command_queue_inst_.enqueueSVMUnmap(output_d);

  // Compare results
  float mse_error = mse<float>(output_h_ref.data(), output_h_gpu.data(), output_h_ref.size());
  double cos_sim = cosine_similarity<float>(output_h_ref.data(), output_h_gpu.data(), output_h_ref.size());

  const float epsilon = 1e-5f;
  EXPECT_IN_RANGE(mse_error, 0, epsilon);
  EXPECT_IN_RANGE((float)cos_sim, 0.99, 1);

  std::cout << "GQA MSE Error: " << mse_error << std::endl;
  std::cout << "GQA Cosine Similarity: " << cos_sim << std::endl;
  
  std::cout << "GQA Configuration: seqlen_q=" << seqlen_q << ", seqlen_k=" << seqlen_k 
            << ", head_dim=" << head_dim << ", num_heads_q=" << num_heads_q 
            << ", num_heads_kv=" << num_heads_kv << ", batch=" << batch << std::endl;

  // Cleanup
  freeSVM(query_d);
  freeSVM(key_d);
  freeSVM(value_d);
  freeSVM(output_d);
}

/**
 * @brief Main function for the test
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