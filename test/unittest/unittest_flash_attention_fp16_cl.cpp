// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	unittest_flash_attention_fp16_cl.cpp
 * @date	24 March 2026
 * @brief	Test setup for flash attention FP16 OpenCL kernels
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
 * @brief Test flash attention kernel implementation for GQA with FP16
 */
static void run_flash_attention_fp16_test(const int seqlen_q, const int seqlen_k,
                                        const int head_dim, const int num_heads_q,
                                        const int num_heads_kv, const int batch);

/**
 * @brief Test case for GQA configuration: head_dim=128, q_len=512, kv_len=512, n_heads_q=16, n_heads_kv=8, batch=1
 */
TEST(nntrainer_opencl_kernels_flash_attention, flash_attention_test_fp16_gqa_512_512_128_16_8) {
  const int seqlen_q = 512;
  const int seqlen_k = 512;
  const int head_dim = 128;
  const int num_heads_q = 16;
  const int num_heads_kv = 8;
  const int batch = 1;
  
  run_flash_attention_fp16_test(seqlen_q, seqlen_k, head_dim, num_heads_q, num_heads_kv, batch);
}

/**
 * @brief Test case for GQA configuration: head_dim=128, q_len=1, kv_len=512, n_heads_q=16, n_heads_kv=8, batch=1
 */
TEST(nntrainer_opencl_kernels_flash_attention, flash_attention_test_fp16_gqa_1_512_128_16_8) {
  const int seqlen_q = 1;
  const int seqlen_k = 512;
  const int head_dim = 128;
  const int num_heads_q = 16;
  const int num_heads_kv = 8;
  const int batch = 1;
  
  run_flash_attention_fp16_test(seqlen_q, seqlen_k, head_dim, num_heads_q, num_heads_kv, batch);
}

/**
 * @brief Test flash attention kernel implementation for GQA with FP16
 */
static void run_flash_attention_fp16_test(const int seqlen_q, const int seqlen_k,
                                        const int head_dim, const int num_heads_q,
                                        const int num_heads_kv, const int batch) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const float alpha = 1e-3f;
  const int MOD = 10;

  // Allocate host memory for GQA
  std::vector<_FP16> query_h(batch * num_heads_q * seqlen_q * head_dim);
  std::vector<_FP16> key_h(batch * num_heads_kv * seqlen_k * head_dim);
  std::vector<_FP16> value_h(batch * num_heads_kv * seqlen_k * head_dim);
  std::vector<float> output_h_ref(batch * num_heads_q * seqlen_q * head_dim, 0.0f);
  std::vector<_FP16> output_h_gpu(batch * num_heads_q * seqlen_q * head_dim, 0.0f);

  // Initialize input data
  for (size_t i = 0; i < query_h.size(); ++i) {
    query_h[i] = static_cast<_FP16>(((static_cast<float>(i % MOD) - MOD/2.0f) * alpha));
  }
  for (size_t i = 0; i < key_h.size(); ++i) {
    key_h[i] = static_cast<_FP16>(((static_cast<float>(i % MOD) - MOD/2.0f) * alpha));
  }
  for (size_t i = 0; i < value_h.size(); ++i) {
    value_h[i] = static_cast<_FP16>(((static_cast<float>(i % MOD) - MOD/2.0f) * alpha));
  }

  // Compute reference output for GQA using FP32
  std::vector<float> query_fp32(query_h.size());
  std::vector<float> key_fp32(key_h.size());
  std::vector<float> value_fp32(value_h.size());
  
  for (size_t i = 0; i < query_h.size(); ++i) {
    query_fp32[i] = static_cast<float>(query_h[i]);
  }
  for (size_t i = 0; i < key_h.size(); ++i) {
    key_fp32[i] = static_cast<float>(key_h[i]);
  }
  for (size_t i = 0; i < value_h.size(); ++i) {
    value_fp32[i] = static_cast<float>(value_h[i]);
  }
  
  // Run CPU computation once to get single run time
  auto cpu_t1 = std::chrono::high_resolution_clock::now();
  flash_attention_cpu(query_fp32.data(), key_fp32.data(), value_fp32.data(),
                                output_h_ref.data(), seqlen_q, seqlen_k, head_dim,
                                num_heads_q, num_heads_kv, batch, scale);
  auto cpu_t2 = std::chrono::high_resolution_clock::now();
  auto single_cpu_dt = std::chrono::duration_cast<std::chrono::microseconds>(cpu_t2 - cpu_t1).count();
  
  // Run CPU computation 100 times for average timing
  const int cpu_iterations = 100;
  long long total_cpu_time = 0;
  
  // Warmup run
  std::vector<float> output_h_ref_warmup(batch * num_heads_q * seqlen_q * head_dim, 0.0f);
  flash_attention_cpu(query_fp32.data(), key_fp32.data(), value_fp32.data(),
                                output_h_ref_warmup.data(), seqlen_q, seqlen_k, head_dim,
                                num_heads_q, num_heads_kv, batch, scale);
  
  // Actual timing runs
  for (int i = 0; i < cpu_iterations; ++i) {
    std::vector<float> output_h_ref_timed(batch * num_heads_q * seqlen_q * head_dim, 0.0f);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    flash_attention_cpu(query_fp32.data(), key_fp32.data(), value_fp32.data(),
                                  output_h_ref_timed.data(), seqlen_q, seqlen_k, head_dim,
                                  num_heads_q, num_heads_kv, batch, scale);
    auto cpu_t2 = std::chrono::high_resolution_clock::now();
    auto cpu_dt = std::chrono::duration_cast<std::chrono::microseconds>(cpu_t2 - cpu_t1).count();
    total_cpu_time += cpu_dt;
  }
  
  auto avg_cpu_dt = total_cpu_time / cpu_iterations;

  // Allocate GPU memory
  _FP16 *query_d = (_FP16 *)allocateSVM(batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16));
  _FP16 *key_d = (_FP16 *)allocateSVM(batch * num_heads_kv * seqlen_k * head_dim * sizeof(_FP16));
  _FP16 *value_d = (_FP16 *)allocateSVM(batch * num_heads_kv * seqlen_k * head_dim * sizeof(_FP16));
  _FP16 *output_d = (_FP16 *)allocateSVM(batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16));

  // Copy data to GPU
  blas_cc->command_queue_inst_.enqueueSVMMap(query_d, batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(key_d, batch * num_heads_kv * seqlen_k * head_dim * sizeof(_FP16), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(value_d, batch * num_heads_kv * seqlen_k * head_dim * sizeof(_FP16), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(output_d, batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16), false);

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

  // Run FP16 GPU kernel with GQA support for timing
  auto t1 = std::chrono::high_resolution_clock::now();
  nntrainer::flash_attention_fp16_cl(query_d, key_d, value_d, output_d, 
                                 seqlen_q, seqlen_k, head_dim, num_heads_q, num_heads_kv, batch, scale);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto single_gpu_dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  
  // Then run 100 times more for average timing
  const int num_iterations = 100;
  long long total_gpu_time = 0;
  
  // Warmup run
  nntrainer::flash_attention_fp16_cl(query_d, key_d, value_d, output_d, 
                                 seqlen_q, seqlen_k, head_dim, num_heads_q, num_heads_kv, batch, scale);
  
  // Actual timing runs
  for (int i = 0; i < num_iterations; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();
    nntrainer::flash_attention_fp16_cl(query_d, key_d, value_d, output_d, 
                                   seqlen_q, seqlen_k, head_dim, num_heads_q, num_heads_kv, batch, scale);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto gpu_dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    total_gpu_time += gpu_dt;
  }
  
  auto avg_gpu_dt = total_gpu_time / num_iterations;

  std::cout << "Flash Attention GQA FP16 CPU time (single run): " << single_cpu_dt / 1000 << " ms" << std::endl;
  std::cout << "Flash Attention GQA FP16 CPU time (100 runs average): " << avg_cpu_dt / 1000 << " ms" << std::endl;
  std::cout << "Flash Attention GQA FP16 GPU time (single run): " << single_gpu_dt / 1000 << " ms" << std::endl;
  std::cout << "Flash Attention GQA FP16 GPU time (100 runs average): " << avg_gpu_dt / 1000 << " ms" << std::endl;
  std::cout << "GQA FP16 Speedup (single run): " << (float)single_cpu_dt / (float)single_gpu_dt << "x" << std::endl;
  std::cout << "GQA FP16 Speedup (average): " << (float)avg_cpu_dt / (float)avg_gpu_dt << "x" << std::endl;

  // Copy result back to host
  blas_cc->command_queue_inst_.enqueueSVMMap(output_d, batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16), true);
  for (size_t i = 0; i < output_h_gpu.size(); ++i) {
    output_h_gpu[i] = output_d[i];
  }
  blas_cc->command_queue_inst_.enqueueSVMUnmap(output_d);

  // Compare results (convert FP16 output to FP32 for comparison)
  std::vector<float> output_h_gpu_fp32(output_h_gpu.size());
  for (size_t i = 0; i < output_h_gpu.size(); ++i) {
    output_h_gpu_fp32[i] = static_cast<float>(output_h_gpu[i]);
  }
  
  float mse_error = mse<float>(output_h_ref.data(), output_h_gpu_fp32.data(), output_h_ref.size());
  double cos_sim = cosine_similarity<float>(output_h_ref.data(), output_h_gpu_fp32.data(), output_h_ref.size());

  const float epsilon = 1e-2f; // Looser epsilon for FP16
  EXPECT_IN_RANGE(mse_error, 0, epsilon);
  EXPECT_IN_RANGE((float)cos_sim, 0.95, 1); // Looser cosine similarity for FP16

  std::cout << "GQA FP16 MSE Error: " << mse_error << std::endl;
  std::cout << "GQA FP16 Cosine Similarity: " << cos_sim << std::endl;
  
  std::cout << "GQA FP16 Configuration: seqlen_q=" << seqlen_q << ", seqlen_k=" << seqlen_k 
            << ", head_dim=" << head_dim << ", num_heads_q=" << num_heads_q 
            << ", num_heads_kv=" << num_heads_kv << ", batch=" << batch << std::endl;

  // Cleanup
  freeSVM(query_d);
  freeSVM(key_d);
  freeSVM(value_d);
  freeSVM(output_d);
}

/**
 * @brief Test L4-2 image kernel (texture cache optimization) for GQA with FP16
 */
TEST(nntrainer_opencl_kernels_flash_attention, flash_attention_test_fp16_image_kernel_l4_2) {
  const int seqlen_q = 512;
  const int seqlen_k = 512;
  const int head_dim = 128;
  const int num_heads_q = 16;
  const int num_heads_kv = 8;
  const int batch = 1;
  
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const float alpha = 1e-3f;
  const int MOD = 10;

  // Allocate host memory for GQA
  std::vector<_FP16> query_h(batch * num_heads_q * seqlen_q * head_dim);
  std::vector<_FP16> key_h(batch * num_heads_kv * seqlen_k * head_dim);
  std::vector<_FP16> value_h(batch * num_heads_kv * seqlen_k * head_dim);
  std::vector<float> output_h_ref(batch * num_heads_q * seqlen_q * head_dim, 0.0f);
  std::vector<_FP16> output_h_v1(batch * num_heads_q * seqlen_q * head_dim, 0.0f);
  std::vector<_FP16> output_h_image(batch * num_heads_q * seqlen_q * head_dim, 0.0f);

  // Initialize input data
  for (size_t i = 0; i < query_h.size(); ++i) {
    query_h[i] = static_cast<_FP16>(((static_cast<float>(i % MOD) - MOD/2.0f) * alpha));
  }
  for (size_t i = 0; i < key_h.size(); ++i) {
    key_h[i] = static_cast<_FP16>(((static_cast<float>(i % MOD) - MOD/2.0f) * alpha));
  }
  for (size_t i = 0; i < value_h.size(); ++i) {
    value_h[i] = static_cast<_FP16>(((static_cast<float>(i % MOD) - MOD/2.0f) * alpha));
  }

  // Compute reference output using CPU
  std::vector<float> query_fp32(query_h.size());
  std::vector<float> key_fp32(key_h.size());
  std::vector<float> value_fp32(value_h.size());
  
  for (size_t i = 0; i < query_h.size(); ++i) {
    query_fp32[i] = static_cast<float>(query_h[i]);
  }
  for (size_t i = 0; i < key_h.size(); ++i) {
    key_fp32[i] = static_cast<float>(key_h[i]);
  }
  for (size_t i = 0; i < value_h.size(); ++i) {
    value_fp32[i] = static_cast<float>(value_h[i]);
  }
  
  // Run CPU computation for reference
  flash_attention_cpu(query_fp32.data(), key_fp32.data(), value_fp32.data(),
                      output_h_ref.data(), seqlen_q, seqlen_k, head_dim,
                      num_heads_q, num_heads_kv, batch, scale);

  // Allocate GPU memory
  _FP16 *query_d = (_FP16 *)allocateSVM(batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16));
  _FP16 *key_d = (_FP16 *)allocateSVM(batch * num_heads_kv * seqlen_k * head_dim * sizeof(_FP16));
  _FP16 *value_d = (_FP16 *)allocateSVM(batch * num_heads_kv * seqlen_k * head_dim * sizeof(_FP16));
  _FP16 *output_d = (_FP16 *)allocateSVM(batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16));

  // Copy data to GPU
  blas_cc->command_queue_inst_.enqueueSVMMap(query_d, batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(key_d, batch * num_heads_kv * seqlen_k * head_dim * sizeof(_FP16), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(value_d, batch * num_heads_kv * seqlen_k * head_dim * sizeof(_FP16), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(output_d, batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16), false);

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

  // Warmup and timing for v1 kernel (buffer-based)
  nntrainer::flash_attention_prefill_fp16_adreno_cl(query_d, key_d, value_d, output_d, 
                                                    seqlen_q, seqlen_k, head_dim, 
                                                    num_heads_q, num_heads_kv, batch, scale);
  
  const int num_iterations = 100;
  long long total_v1_time = 0;
  
  for (int i = 0; i < num_iterations; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();
    nntrainer::flash_attention_prefill_fp16_adreno_cl(query_d, key_d, value_d, output_d, 
                                                      seqlen_q, seqlen_k, head_dim, 
                                                      num_heads_q, num_heads_kv, batch, scale);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    total_v1_time += dt;
  }
  
  // Copy v1 result
  blas_cc->command_queue_inst_.enqueueSVMMap(output_d, batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16), true);
  for (size_t i = 0; i < output_h_v1.size(); ++i) {
    output_h_v1[i] = output_d[i];
  }
  blas_cc->command_queue_inst_.enqueueSVMUnmap(output_d);

  // Warmup and timing for L4-2 image kernel (texture cache)
  nntrainer::flash_attention_prefill_fp16_adreno_image_cl(query_d, key_d, value_d, output_d, 
                                                          seqlen_q, seqlen_k, head_dim, 
                                                          num_heads_q, num_heads_kv, batch, scale);
  
  long long total_image_time = 0;
  
  for (int i = 0; i < num_iterations; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();
    nntrainer::flash_attention_prefill_fp16_adreno_image_cl(query_d, key_d, value_d, output_d, 
                                                            seqlen_q, seqlen_k, head_dim, 
                                                            num_heads_q, num_heads_kv, batch, scale);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    total_image_time += dt;
  }
  
  // Copy image kernel result
  blas_cc->command_queue_inst_.enqueueSVMMap(output_d, batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16), true);
  for (size_t i = 0; i < output_h_image.size(); ++i) {
    output_h_image[i] = output_d[i];
  }
  blas_cc->command_queue_inst_.enqueueSVMUnmap(output_d);

  // Calculate average times
  auto avg_v1_time = total_v1_time / num_iterations;
  auto avg_image_time = total_image_time / num_iterations;

  // Compare results
  std::vector<float> output_h_v1_fp32(output_h_v1.size());
  std::vector<float> output_h_image_fp32(output_h_image.size());
  for (size_t i = 0; i < output_h_v1.size(); ++i) {
    output_h_v1_fp32[i] = static_cast<float>(output_h_v1[i]);
    output_h_image_fp32[i] = static_cast<float>(output_h_image[i]);
  }
  
  float mse_v1 = mse<float>(output_h_ref.data(), output_h_v1_fp32.data(), output_h_ref.size());
  float mse_image = mse<float>(output_h_ref.data(), output_h_image_fp32.data(), output_h_ref.size());
  double cos_sim_v1 = cosine_similarity<float>(output_h_ref.data(), output_h_v1_fp32.data(), output_h_ref.size());
  double cos_sim_image = cosine_similarity<float>(output_h_ref.data(), output_h_image_fp32.data(), output_h_ref.size());

  std::cout << "\n===== L4-2 Image Kernel Benchmark =====" << std::endl;
  std::cout << "v1 Kernel (buffer):    " << avg_v1_time / 1000 << " ms avg (" << num_iterations << " runs)" << std::endl;
  std::cout << "L4-2 Kernel (image):   " << avg_image_time / 1000 << " ms avg (" << num_iterations << " runs)" << std::endl;
  
  float speedup = (float)avg_v1_time / (float)avg_image_time;
  if (speedup >= 1.0f) {
    std::cout << "L4-2 Speedup:          " << speedup << "x FASTER" << std::endl;
  } else {
    std::cout << "L4-2 Speedup:          " << (1.0f/speedup) << "x SLOWER (REGRESSION)" << std::endl;
  }
  
  std::cout << "\nv1 MSE:                 " << mse_v1 << std::endl;
  std::cout << "L4-2 MSE:               " << mse_image << std::endl;
  std::cout << "v1 Cosine Sim:          " << cos_sim_v1 << std::endl;
  std::cout << "L4-2 Cosine Sim:        " << cos_sim_image << std::endl;
  std::cout << "=========================================" << std::endl;

  // Verify correctness
  const float epsilon = 1e-2f;
  EXPECT_IN_RANGE(mse_image, 0, epsilon);
  EXPECT_IN_RANGE((float)cos_sim_image, 0.95, 1);

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
  std::cout << "Inside Main fp16\n";
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
} // ENABLE_FP16