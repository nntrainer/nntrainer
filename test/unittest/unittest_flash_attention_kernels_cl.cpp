// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Kumar Tiwari(anup.tiwari@samsung.com)
 *
 * @file	unittest_flash_attention_kernels_cl.cpp
 * @date	23 March 2026
 * @brief	Test setup for flash attention OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Anup Kumar Tiwari(anup.tiwari@samsung.com)
 * @bug		No known bugs except for NYI items
 */

#include <fstream>
#include <gtest/gtest.h>
#include <type_traits>
#include <chrono>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <cl_context.h>
#include <layer_context.h>
#include <tensor.h>

#include <flash_attention_kernel.h>
#include <cl_context.h>
#include <engine.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

using namespace nntrainer;

// Helper function to compute flash attention on CPU for verification
template <typename T>
void flash_attention_cpu(const T *query, const T *key, const T *value,
                         T *output, const T *attention_mask, int batch_size,
                         int num_heads, int seq_len, int head_dim, T scale) {
  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < num_heads; h++) {
      for (int i = 0; i < seq_len; i++) {
        // Compute attention scores
        std::vector<T> scores(seq_len, 0);
        T max_score = -std::numeric_limits<T>::infinity();
        
        for (int j = 0; j < seq_len; j++) {
          T sum = 0;
          for (int d = 0; d < head_dim; d++) {
            int q_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
            int k_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
            sum += query[q_idx] * key[k_idx];
          }
          scores[j] = sum * scale;
          
          // Apply mask if provided
          if (attention_mask) {
            int mask_idx = (b * seq_len + i) * seq_len + j;
            scores[j] += attention_mask[mask_idx];
          }
          
          max_score = std::max(max_score, scores[j]);
        }
        
        // Compute softmax
        T exp_sum = 0;
        std::vector<T> softmax_scores(seq_len);
        for (int j = 0; j < seq_len; j++) {
          // Handle type-specific exp calculation
          if constexpr (std::is_same_v<T, _FP16>) {
            softmax_scores[j] = static_cast<T>(std::exp(static_cast<float>(scores[j] - max_score)));
          } else {
            softmax_scores[j] = std::exp(static_cast<float>(scores[j] - max_score));
          }
          exp_sum += softmax_scores[j];
        }
        
        // Normalize softmax scores
        for (int j = 0; j < seq_len; j++) {
          softmax_scores[j] /= exp_sum;
        }
        
        // Compute weighted sum
        for (int d = 0; d < head_dim; d++) {
          T sum = 0;
          for (int j = 0; j < seq_len; j++) {
            int v_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
            sum += softmax_scores[j] * value[v_idx];
          }
          int out_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
          output[out_idx] = sum;
        }
      }
    }
  }
}

// Helper function to time execution
template<typename Func>
double time_execution(Func&& func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return duration.count() / 1000.0; // Return milliseconds
}

// Test case structure
struct FlashAttentionTestCase {
  int batch_size;
  int num_heads;
  int seq_len;
  int head_dim;
  float scale;
  std::string name;
};

// Generate test cases
std::vector<FlashAttentionTestCase> generate_test_cases() {
  return {
    {1, 1, 8, 16, 0.1f, "Small_Single_Batch"},
    {1, 4, 16, 32, 0.05f, "Small_Multi_Head"},
    {2, 2, 32, 64, 0.025f, "Medium_Batch"},
    {1, 8, 64, 64, 0.0125f, "Large_Head_Count"},
    {4, 4, 32, 128, 0.008f, "Large_Head_Dim"},
    {2, 6, 128, 32, 0.01f, "Long_Sequence"},
    {8, 2, 16, 256, 0.005f, "Very_Large_Head_Dim"},
    {1, 12, 256, 16, 0.002f, "Very_Long_Sequence"},
    {16, 1, 8, 64, 0.01f, "Large_Batch"},
    {1, 1, 512, 8, 0.001f, "Extreme_Length"}
  };
}

// Test flash attention FP32
TEST(flash_attention_kernels, flash_attention_kernel_FP32) {
  auto test_cases = generate_test_cases();
  
  for (const auto& test_case : test_cases) {
    int batch = test_case.batch_size;
    int num_heads = test_case.num_heads;
    int seq_len = test_case.seq_len;
    int head_dim = test_case.head_dim;
    float scale = test_case.scale;
    
    nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
      nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};
    
    // Create tensors
    nntrainer::Tensor query(batch, num_heads, seq_len, head_dim, t_type_nchw_fp32);
    nntrainer::Tensor key(batch, num_heads, seq_len, head_dim, t_type_nchw_fp32);
    nntrainer::Tensor value(batch, num_heads, seq_len, head_dim, t_type_nchw_fp32);
    nntrainer::Tensor output_gpu(batch, num_heads, seq_len, head_dim, t_type_nchw_fp32);
    nntrainer::Tensor output_cpu(batch, num_heads, seq_len, head_dim, t_type_nchw_fp32);
    nntrainer::Tensor attention_mask(batch, 1, seq_len, seq_len, t_type_nchw_fp32);
    
    // Initialize with random data
    const float alpha = 1e-2;
    const int MOD = 100;
    
    query.setRandUniform(0, 1);
    key.setRandUniform(0, 1);
    value.setRandUniform(0, 1);
    attention_mask.setRandUniform(0, 1);
    
    // Copy data for CPU computation
    output_cpu.copy(output_gpu);
    
    // Time GPU execution
    double gpu_time = time_execution([&]() {
      flash_attention_cl(query, key, value, output_gpu, &attention_mask,
                         batch, num_heads, seq_len, head_dim, scale);
    });
    
    // Time CPU execution
    double cpu_time = time_execution([&]() {
      flash_attention_cpu<float>(query.getData<float>(), key.getData<float>(),
                                 value.getData<float>(), output_cpu.getData<float>(),
                                 attention_mask.getData<float>(), batch, num_heads,
                                 seq_len, head_dim, scale);
    });
    
    // Compare results
    float mseError = mse<float>(output_gpu.getData<float>(), output_cpu.getData<float>(), 
                                output_gpu.size());
    
    double cosSim = cosine_similarity<float>(output_gpu.getData<float>(), 
                                             output_cpu.getData<float>(), 
                                             output_gpu.size());
    
    const float epsilon = 1e-3f;
    
    EXPECT_IN_RANGE(mseError, 0, epsilon) 
      << "Test case: " << test_case.name << " MSE error too high";
    EXPECT_IN_RANGE((float)cosSim, 0.99f, 1.0f) 
      << "Test case: " << test_case.name << " Cosine similarity too low";
    
    // Print performance comparison
    std::cout << "Test case: " << test_case.name 
              << " | GPU Time: " << gpu_time << " ms"
              << " | CPU Time: " << cpu_time << " ms"
              << " | Speedup: " << (cpu_time / gpu_time) << "x"
              << " | MSE: " << mseError 
              << " | CosSim: " << cosSim << std::endl;
  }
}

#ifdef ENABLE_FP16

// Test flash attention FP16
TEST(flash_attention_kernels, flash_attention_kernel_FP16) {
  auto test_cases = generate_test_cases();
  
  for (const auto& test_case : test_cases) {
    int batch = test_case.batch_size;
    int num_heads = test_case.num_heads;
    int seq_len = test_case.seq_len;
    int head_dim = test_case.head_dim;
    float scale = test_case.scale;
    
    nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
      nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};
    
    // Create tensors
    nntrainer::Tensor query(batch, num_heads, seq_len, head_dim, t_type_nchw_fp16);
    nntrainer::Tensor key(batch, num_heads, seq_len, head_dim, t_type_nchw_fp16);
    nntrainer::Tensor value(batch, num_heads, seq_len, head_dim, t_type_nchw_fp16);
    nntrainer::Tensor output_gpu(batch, num_heads, seq_len, head_dim, t_type_nchw_fp16);
    nntrainer::Tensor output_cpu(batch, num_heads, seq_len, head_dim, t_type_nchw_fp16);
    nntrainer::Tensor attention_mask(batch, 1, seq_len, seq_len, t_type_nchw_fp16);
    
    // Initialize with random data
    const float alpha = 1e-2;
    const int MOD = 100;
    
    query.setRandUniform(0, 1);
    key.setRandUniform(0, 1);
    value.setRandUniform(0, 1);
    attention_mask.setRandUniform(0, 1);
    
    // Copy data for CPU computation
    output_cpu.copy(output_gpu);
    
    // Time GPU execution
    double gpu_time = time_execution([&]() {
      flash_attention_cl_fp16(query, key, value, output_gpu, &attention_mask,
                              batch, num_heads, seq_len, head_dim, scale);
    });
    
    // Time CPU execution
    double cpu_time = time_execution([&]() {
      flash_attention_cpu<_FP16>(query.getData<_FP16>(), key.getData<_FP16>(),
                                 value.getData<_FP16>(), output_cpu.getData<_FP16>(),
                                 attention_mask.getData<_FP16>(), batch, num_heads,
                                 seq_len, head_dim, static_cast<_FP16>(scale));
    });
    
    // Compare results
    float mseError = mse<_FP16>(output_gpu.getData<_FP16>(), output_cpu.getData<_FP16>(), 
                                output_gpu.size());
    
    double cosSim = cosine_similarity<_FP16>(output_gpu.getData<_FP16>(), 
                                             output_cpu.getData<_FP16>(), 
                                             output_gpu.size());
    
    const float epsilon = 1e-2f; // Slightly higher epsilon for FP16
    
    EXPECT_IN_RANGE(mseError, 0, epsilon) 
      << "Test case: " << test_case.name << " MSE error too high";
    EXPECT_IN_RANGE((float)cosSim, 0.95f, 1.0f) 
      << "Test case: " << test_case.name << " Cosine similarity too low";
    
    // Print performance comparison
    std::cout << "Test case (FP16): " << test_case.name 
              << " | GPU Time: " << gpu_time << " ms"
              << " | CPU Time: " << cpu_time << " ms"
              << " | Speedup: " << (cpu_time / gpu_time) << "x"
              << " | MSE: " << mseError 
              << " | CosSim: " << cosSim << std::endl;
  }
}

#endif

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
