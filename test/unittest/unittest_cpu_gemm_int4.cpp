// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_cpu_gemm_int4.cpp
 * @brief  Unit test for CPU int4 GEMM operations
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "gemm_int4_cpu.h"
#include "int4_utils.h"
#include "unittest_util.h" // For generate_random_vector, etc.

using namespace nntrainer;

// Helper for ceil division
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

static void run_cpu_int4_gemm_test_(const uint32_t M, const uint32_t K,
                                    const uint32_t N,
                                    const int scale_group_size) {
  // 1. Prepare Data
  float input_min = -1.0f, input_max = 1.0f;
  float weight_min = -0.5f, weight_max = 0.5f;

  std::vector<float> input =
    generate_random_vector<float>(M * K, input_min, input_max);
  std::vector<float> weights_fp32 =
    generate_random_vector<float>(K * N, weight_min, weight_max);
  std::vector<float> output_cpu(M * N);
  std::vector<float> output_ref(M * N);

  // 2. Quantize Weights using Int4Utils
  // weights_fp32 is K x N. Transpose to N x K for quantization.
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

  // 4. Run CPU INT4 GEMM
  gemm_int4_cpu(input_quantized.data(), weights_int4.data(), scales_fp16.data(),
                input_scales.data(), output_cpu.data(), M, N, K,
                scale_group_size);

  // 5. Run Reference FP32 GEMM
  gemm_fp32_ref(input.data(), weights_fp32.data(), output_ref.data(), M, N, K);

  // 6. Compare
  // Note: INT4 quantization introduces error.
  // We check if the MSE is within a reasonable range.
  float mse = 0.0f;
  float max_diff = 0.0f;
  for (size_t i = 0; i < output_cpu.size(); ++i) {
    float diff = output_cpu[i] - output_ref[i];
    mse += diff * diff;
    if (std::abs(diff) > max_diff) {
      max_diff = std::abs(diff);
    }
  }
  mse /= output_cpu.size();

  std::cout << "M: " << M << " K: " << K << " N: " << N
            << " Group: " << scale_group_size << " MSE: " << mse
            << " MaxDiff: " << max_diff << std::endl;

  // Threshold calculation based on quantization error variance
  // 1. Quantization Step Sizes
  float input_range = input_max - input_min;
  float weight_range = weight_max - weight_min;
  float delta_input = input_range / 255.0f;  // 8-bit
  float delta_weight = weight_range / 15.0f; // 4-bit

  // 2. Error Variances (Uniform distribution assumption)
  float var_err_input = (delta_input * delta_input) / 12.0f;
  float var_err_weight = (delta_weight * delta_weight) / 12.0f;

  // 3. Signal Mean Squared Values (Uniform distribution assumption)
  float mean_sq_input =
    (input_min * input_min + input_min * input_max + input_max * input_max) /
    3.0f;
  float mean_sq_weight = (weight_min * weight_min + weight_min * weight_max +
                          weight_max * weight_max) /
                         3.0f;

  // 4. Product Error Variance
  // Var(xy) approx E[x^2]Var(y) + E[y^2]Var(x)
  float var_err_product =
    mean_sq_input * var_err_weight + mean_sq_weight * var_err_input;

  // 5. Accumulated Error Variance (Sum of K products)
  float expected_mse = K * var_err_product;

  // 6. Threshold with safety margin (e.g., 2.0x for outliers/distribution
  // mismatch)
  float threshold = expected_mse * 2.0f;

  std::cout << "Expected MSE: " << expected_mse << " Threshold: " << threshold
            << std::endl;
  EXPECT_LT(mse, threshold);
}

#define DECLARE_CPU_INT4_GEMM_TEST(M, K, N, G)                                 \
  TEST(gemm_int4_cpu, test_##M##_##K##_##N##_Group##G) {                       \
    run_cpu_int4_gemm_test_(M, K, N, G);                                       \
  }

DECLARE_CPU_INT4_GEMM_TEST(28, 3072, 256, 32);
DECLARE_CPU_INT4_GEMM_TEST(28, 3072, 8192, 32);
DECLARE_CPU_INT4_GEMM_TEST(28, 8192, 3072, 32);
DECLARE_CPU_INT4_GEMM_TEST(28, 3072, 3072, 32);

// DECLARE_CPU_INT4_GEMM_TEST(28, 3072, 256, 128);
// DECLARE_CPU_INT4_GEMM_TEST(28, 3072, 8192, 128);
// DECLARE_CPU_INT4_GEMM_TEST(28, 8192, 3072, 128);
// DECLARE_CPU_INT4_GEMM_TEST(28, 3072, 3072, 128);

DECLARE_CPU_INT4_GEMM_TEST(28, 32, 3072, 32);
DECLARE_CPU_INT4_GEMM_TEST(28, 64, 3072, 32);
DECLARE_CPU_INT4_GEMM_TEST(28, 128, 3072, 32);
DECLARE_CPU_INT4_GEMM_TEST(28, 256, 3072, 32);

// DECLARE_CPU_INT4_GEMM_TEST(28, 3000, 3072, 128);
// DECLARE_CPU_INT4_GEMM_TEST(28, 2000, 3072, 128);

// DECLARE_CPU_INT4_GEMM_TEST(4, 3060, 3072, 32);
// DECLARE_CPU_INT4_GEMM_TEST(4, 3072, 3072, 32);

#include "fp16.h"

// Helper to dequantize input
// input_quantized: int8_t
// input_scales: uint16_t (fp16)
// M, K, group_size
static void dequantize_input(const int8_t *input_quantized,
                             const uint16_t *input_scales,
                             float *input_dequantized, unsigned int M,
                             unsigned int K, unsigned int group_size) {
  unsigned int alignK = ALIGN(K, group_size);
  unsigned int groups_in_row = alignK / group_size;

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int k = 0; k < K; ++k) {
      unsigned int group_id_in_row = k / group_size;
      unsigned int global_group_id = m * groups_in_row + group_id_in_row;
      unsigned int offset_in_group = k % group_size;
      unsigned int input_idx = global_group_id * group_size + offset_in_group;

      int8_t q_val = input_quantized[input_idx];
      float scale = compute_fp16_to_fp32(input_scales[global_group_id * 2]);

      input_dequantized[m * K + k] = static_cast<float>(q_val) * scale;
    }
  }
}

static void run_quantization_gemm_test_(const uint32_t M, const uint32_t K,
                                        const uint32_t N,
                                        const int scale_group_size) {
  // 1. Prepare Data
  float input_min = -1.0f, input_max = 1.0f;
  float weight_min = -0.5f, weight_max = 0.5f;

  std::vector<float> input =
    generate_random_vector<float>(M * K, input_min, input_max);
  std::vector<float> weights_fp32 =
    generate_random_vector<float>(K * N, weight_min, weight_max);
  std::vector<float> output_dequantized(M * N);
  std::vector<float> output_ref(M * N);

  // 2. Quantize Weights
  // weights_fp32 is K x N (Row-Major).
  // Int4Utils::quantizeAndRepack expects N x K (N rows, K columns), where each
  // row is quantized together. So we must transpose weights_fp32 to N x K.
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

  // 3. Quantize Input
  unsigned int alignK = ALIGN(K, scale_group_size);
  unsigned int groups_in_row = alignK / scale_group_size;
  std::vector<int8_t> input_quantized(M * alignK);
  std::vector<uint16_t> input_scales(M * groups_in_row * 2);
  cpu_quantize_input_int8_pad(input.data(), input_quantized.data(),
                              input_scales.data(), M, K, scale_group_size);

  // 4. Dequantize Weights back to FP32
  std::vector<float> weights_dequantized_NxK;
  Int4Utils::dequantizePacked(weights_int4, scales_fp16, N, K, scale_group_size,
                              weights_dequantized_NxK);

  // Transpose weights_dequantized (N x K) -> (K x N) for GEMM reference
  std::vector<float> weights_dequantized_KxN(K * N);
  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = 0; k < K; ++k) {
      weights_dequantized_KxN[k * N + n] = weights_dequantized_NxK[n * K + k];
    }
  }

  // 5. Dequantize Input back to FP32
  std::vector<float> input_dequantized(M * K);
  dequantize_input(input_quantized.data(), input_scales.data(),
                   input_dequantized.data(), M, K, scale_group_size);

  // 6. Run Reference FP32 GEMM on Dequantized Data
  gemm_fp32_ref(input_dequantized.data(), weights_dequantized_KxN.data(),
                output_dequantized.data(), M, N, K);

  // 7. Run Reference FP32 GEMM on Original Data
  gemm_fp32_ref(input.data(), weights_fp32.data(), output_ref.data(), M, N, K);

  // 8. Compare
  float mse = 0.0f;
  float max_diff = 0.0f;
  for (size_t i = 0; i < output_dequantized.size(); ++i) {
    float diff = output_dequantized[i] - output_ref[i];
    mse += diff * diff;
    if (std::abs(diff) > max_diff) {
      max_diff = std::abs(diff);
    }
  }
  mse /= output_dequantized.size();

  std::cout << "[Quantization Error] M: " << M << " K: " << K << " N: " << N
            << " Group: " << scale_group_size << " MSE: " << mse
            << " MaxDiff: " << max_diff << std::endl;

  // This MSE represents the pure quantization loss.
  // If this is high, then INT4 is just lossy.
  // If this is low, but the GEMM test fails, then GEMM implementation is wrong.
  // Threshold calculation based on quantization error variance
  // 1. Quantization Step Sizes
  float input_range = input_max - input_min;
  float weight_range = weight_max - weight_min;
  float delta_input = input_range / 255.0f;  // 8-bit
  float delta_weight = weight_range / 15.0f; // 4-bit

  // 2. Error Variances (Uniform distribution assumption)
  float var_err_input = (delta_input * delta_input) / 12.0f;
  float var_err_weight = (delta_weight * delta_weight) / 12.0f;

  // 3. Signal Mean Squared Values (Uniform distribution assumption)
  float mean_sq_input =
    (input_min * input_min + input_min * input_max + input_max * input_max) /
    3.0f;
  float mean_sq_weight = (weight_min * weight_min + weight_min * weight_max +
                          weight_max * weight_max) /
                         3.0f;

  // 4. Product Error Variance
  // Var(xy) approx E[x^2]Var(y) + E[y^2]Var(x)
  float var_err_product =
    mean_sq_input * var_err_weight + mean_sq_weight * var_err_input;

  // 5. Accumulated Error Variance (Sum of K products)
  float expected_mse = K * var_err_product;

  // 6. Threshold with safety margin (e.g., 2.0x for outliers/distribution
  // mismatch)
  float threshold = expected_mse * 2.0f;

  std::cout << "Expected MSE: " << expected_mse << " Threshold: " << threshold
            << std::endl;
  EXPECT_LT(mse, threshold);
}

#define DECLARE_QUANTIZATION_TEST(M, K, N, G)                                  \
  TEST(gemm_int4_cpu, quantization_##M##_##K##_##N##_Group##G) {               \
    run_quantization_gemm_test_(M, K, N, G);                                   \
  }

DECLARE_QUANTIZATION_TEST(28, 32, 3072, 32);
DECLARE_QUANTIZATION_TEST(28, 64, 3072, 32);
DECLARE_QUANTIZATION_TEST(28, 128, 3072, 32);
DECLARE_QUANTIZATION_TEST(28, 256, 3072, 32);

static void run_cpu_int4_gemm_packed_block_test_(const uint32_t M,
                                                 const uint32_t K,
                                                 const uint32_t N,
                                                 const int scale_group_size) {
  // 1. Prepare Data
  float input_min = -1.0f, input_max = 1.0f;
  float weight_min = -0.5f, weight_max = 0.5f;

  std::vector<float> input =
    generate_random_vector<float>(M * K, input_min, input_max);
  std::vector<float> weights_fp32 =
    generate_random_vector<float>(K * N, weight_min, weight_max);
  std::vector<float> output_cpu(M * N);
  std::vector<float> output_ref(M * N);

  // 2. Quantize Weights
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

  // 3. Quantize Input
  unsigned int alignK = ALIGN(K, scale_group_size);
  unsigned int groups_in_row = alignK / scale_group_size;
  std::vector<int8_t> input_quantized(M * alignK);
  std::vector<uint16_t> input_scales(M * groups_in_row * 2);

  cpu_quantize_input_int8_pad(input.data(), input_quantized.data(),
                              input_scales.data(), M, K, scale_group_size);

  // 4. Run Optimized CPU INT4 GEMM
  gemm_int4_cpu_packed_block(input_quantized.data(), weights_int4.data(),
                             scales_fp16.data(), input_scales.data(),
                             output_cpu.data(), M, N, K, scale_group_size);

  // 5. Run Reference FP32 GEMM
  gemm_fp32_ref(input.data(), weights_fp32.data(), output_ref.data(), M, N, K);

  // 6. Compare
  float mse = 0.0f;
  float max_diff = 0.0f;
  for (size_t i = 0; i < output_cpu.size(); ++i) {
    float diff = output_cpu[i] - output_ref[i];
    mse += diff * diff;
    if (std::abs(diff) > max_diff) {
      max_diff = std::abs(diff);
    }
  }
  mse /= output_cpu.size();

  std::cout << "[Packed Block] M: " << M << " K: " << K << " N: " << N
            << " Group: " << scale_group_size << " MSE: " << mse
            << " MaxDiff: " << max_diff << std::endl;

  // Threshold calculation
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

#define DECLARE_CPU_INT4_GEMM_PACKED_BLOCK_TEST(M, K, N, G)                    \
  TEST(gemm_int4_cpu, packed_block_##M##_##K##_##N##_Group##G) {               \
    run_cpu_int4_gemm_packed_block_test_(M, K, N, G);                          \
  }

DECLARE_CPU_INT4_GEMM_PACKED_BLOCK_TEST(28, 32, 3072, 32);
DECLARE_CPU_INT4_GEMM_PACKED_BLOCK_TEST(28, 64, 3072, 32);
DECLARE_CPU_INT4_GEMM_PACKED_BLOCK_TEST(28, 128, 3072, 32);
DECLARE_CPU_INT4_GEMM_PACKED_BLOCK_TEST(28, 256, 3072, 32);

DECLARE_CPU_INT4_GEMM_PACKED_BLOCK_TEST(28, 3072, 3072, 32);
DECLARE_CPU_INT4_GEMM_PACKED_BLOCK_TEST(28, 3072, 8192, 32);
DECLARE_CPU_INT4_GEMM_PACKED_BLOCK_TEST(28, 8192, 3072, 32);

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
