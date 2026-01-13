// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file unittest_gemm_fp32_qint4_comparison.cpp
 * @date 23 Dec 2024
 * @brief Performance comparison test between gemm_fp32_qint4_cuda and
 * gemm_fp32_qint4_cuda_v2
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

#include "custom_dot_wrapper_cuda.h"
#include "gemm_int4_cuda.h"
#include "quantize_cuda.h"
#include "unittest_util.h"
#include <int4_utils.h>

#define ALIGN(a, b) (((a) + (b) - 1) / (b) * (b))

using namespace nntrainer;

// Test parameter structure
struct GemmComparisonTestParam {
  unsigned int M;
  unsigned int K;
  unsigned int N;
};

namespace {
template <typename Func>
float run_benchmark(const char *name, Func kernel_func, float *d_output,
                    std::vector<float> &h_output, size_t output_bytes,
                    float *d_input_fp32, uint8_t *d_weights, uint16_t *d_scales,
                    unsigned int M, unsigned int N, unsigned int K,
                    unsigned int quantization_group_size, int iterations = 10) {

  // 1. Run Baseline (V1)
  unsigned int alignK = ALIGN(K, quantization_group_size);
  unsigned int groups_in_row = alignK / quantization_group_size;
  size_t input_quantized_bytes = M * alignK * sizeof(int8_t);
  size_t input_scales_bytes = M * groups_in_row * 2 * sizeof(uint16_t);

  int8_t *d_quantized_input_temp;
  uint16_t *d_input_scales_temp;
  float *d_output_baseline;

  cudaMalloc(&d_quantized_input_temp, input_quantized_bytes);
  cudaMalloc(&d_input_scales_temp, input_scales_bytes);
  cudaMalloc(&d_output_baseline, output_bytes);

  // Baseline Warmup
  custom::gemm_a32_w4_default_cuda(
    d_input_fp32, reinterpret_cast<const char *>(d_weights), d_scales, nullptr,
    d_quantized_input_temp, d_input_scales_temp, d_output_baseline, M, N, K,
    quantization_group_size);
  cudaDeviceSynchronize();

  // Baseline Benchmark
  float baseline_time = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < iterations; ++i) {
    cudaEventRecord(start);
    custom::gemm_a32_w4_default_cuda(
      d_input_fp32, reinterpret_cast<const char *>(d_weights), d_scales,
      nullptr, d_quantized_input_temp, d_input_scales_temp, d_output_baseline,
      M, N, K, quantization_group_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float iter_ms = 0.0f;
    cudaEventElapsedTime(&iter_ms, start, stop);
    baseline_time += iter_ms;
  }
  float avg_baseline = baseline_time / iterations;

  cudaFree(d_quantized_input_temp);
  cudaFree(d_input_scales_temp);
  cudaFree(d_output_baseline); // 버퍼 해제

  // 2. Run Target Kernel
  // Warmup (loop to wake up GPU clock)
  for (int i = 0; i < 50; ++i) {
    kernel_func();
  }
  cudaDeviceSynchronize();

  float total_time = 0.0f;
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

  std::cout << name << " time: " << avg_time << " ms";
  if (avg_time < avg_baseline) {
    std::cout << " (Speedup: " << avg_baseline / avg_time << "x vs V1 "
              << avg_baseline << " ms)" << std::endl;
  } else {
    std::cout << " (Slowdown: " << avg_time / avg_baseline << "x vs V1 "
              << avg_baseline << " ms)" << std::endl;
  }

  return avg_time;
}

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
} // namespace

// Helper function to prepare test data
void prepare_test_data(unsigned int M, unsigned int K, unsigned int N,
                       unsigned int quantization_group_size,
                       std::vector<float> &input,
                       std::vector<float> &weights_fp32,
                       std::vector<uint8_t> &weights_int4,
                       std::vector<uint16_t> &scales_fp16) {
  input = generate_random_vector<float>(M * K, -1.0f, 1.0f);
  weights_fp32 = generate_random_vector<float>(K * N, -0.5f, 0.5f);

  std::vector<float> weights_NxK(N * K);
  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int n = 0; n < N; ++n) {
      weights_NxK[n * K + k] = weights_fp32[k * N + n];
    }
  }

  Int4Utils::quantizeAndRepack(weights_NxK.data(), N, K,
                               quantization_group_size, weights_int4,
                               scales_fp16);
}

class GemmFp32Qint4ComparisonTest
  : public ::testing::TestWithParam<GemmComparisonTestParam> {
protected:
  template <typename KernelFunc>
  void RunGemmTest(const char *kernel_name, KernelFunc kernel_call) {
    const auto &param = GetParam();
    const unsigned int M = param.M;
    const unsigned int K = param.K;
    const unsigned int N = param.N;
    const unsigned int quantization_group_size = 32;

    std::cout << "\n=== " << kernel_name << " Testing M=" << M << ", K=" << K
              << ", N=" << N << " (Group=" << quantization_group_size
              << ") ===" << std::endl;

    std::vector<float> input, weights_fp32;
    std::vector<uint8_t> weights_int4;
    std::vector<uint16_t> scales_fp16;
    prepare_test_data(M, K, N, quantization_group_size, input, weights_fp32,
                      weights_int4, scales_fp16);

    std::vector<float> output(M * N);
    std::vector<float> output_ref(M * N);

    float *d_input_fp32, *d_output;
    uint8_t *d_weights;
    uint16_t *d_scales;
    int8_t *d_quantized_input_temp;
    uint16_t *d_input_scales_temp;

    size_t input_bytes = input.size() * sizeof(float);
    size_t weights_bytes = weights_int4.size() * sizeof(uint8_t);
    size_t scales_bytes = scales_fp16.size() * sizeof(uint16_t);
    size_t output_bytes = output.size() * sizeof(float);

    unsigned int alignK = ALIGN(K, quantization_group_size);
    unsigned int groups_in_row = alignK / quantization_group_size;
    size_t input_quantized_bytes = M * alignK * sizeof(int8_t);
    size_t input_scales_bytes = M * groups_in_row * 2 * sizeof(uint16_t);

    cudaMalloc(&d_input_fp32, input_bytes);
    cudaMalloc(&d_weights, weights_bytes);
    cudaMalloc(&d_scales, scales_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMalloc(&d_quantized_input_temp, input_quantized_bytes);
    cudaMalloc(&d_input_scales_temp, input_scales_bytes);

    cudaMemcpy(d_input_fp32, input.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_int4.data(), weights_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales_fp16.data(), scales_bytes,
               cudaMemcpyHostToDevice);

    run_benchmark(
      kernel_name,
      [&]() {
        kernel_call(d_input_fp32, d_weights, d_scales, d_quantized_input_temp,
                    d_input_scales_temp, d_output, M, N, K,
                    quantization_group_size);
      },
      d_output, output, output_bytes, d_input_fp32, d_weights, d_scales, M, N,
      K, quantization_group_size);

    cudaFree(d_input_fp32);
    cudaFree(d_weights);
    cudaFree(d_scales);
    cudaFree(d_output);
    cudaFree(d_quantized_input_temp);
    cudaFree(d_input_scales_temp);

    gemm_fp32_ref(input.data(), weights_fp32.data(), output_ref.data(), M, N,
                  K);
    float mse = verify_result(kernel_name, output, output_ref);

    float expected_mse = K * (1.0f / 12.0f) *
                         ((2.0f / 255.0f) * (2.0f / 255.0f) * (1.0f / 3.0f) +
                          (1.0f / 15.0f) * (1.0f / 15.0f) * (1.0f / 3.0f));
    float threshold = expected_mse * 2.0f;
    EXPECT_LT(mse, threshold) << kernel_name << " accuracy check failed";
  }
};

// V1: INT8 Quantization
TEST_P(GemmFp32Qint4ComparisonTest, V1_INT8_Quant) {
  RunGemmTest("V1 (INT8 Quant)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_default_cuda(
                  d_in, reinterpret_cast<const char *>(d_w), d_s, nullptr,
                  d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V2: No INT8 Quantization
TEST_P(GemmFp32Qint4ComparisonTest, V2_No_INT8_Quant) {
  RunGemmTest("V2 (No INT8 Quant)", [&](float *d_in, uint8_t *d_w,
                                        uint16_t *d_s, int8_t * /*d_q_in*/,
                                        uint16_t * /*d_q_s*/, float *d_out,
                                        int M, int N, int K, int grp) {
    custom::gemm_a32_w4_b16x16_naive_cuda(d_in, d_w, d_s, d_out, M, N, K, grp);
  });
}

// V3: Shared Mem Tiled
TEST_P(GemmFp32Qint4ComparisonTest, V3_Shared_Mem_Tiled) {
  RunGemmTest("V3 (Shared Mem Tiled)", [&](float *d_in, uint8_t *d_w,
                                           uint16_t *d_s, int8_t * /*d_q_in*/,
                                           uint16_t * /*d_q_s*/, float *d_out,
                                           int M, int N, int K, int grp) {
    custom::gemm_a32_w4_b32x32_s32x32_cuda(d_in, d_w, d_s, d_out, M, N, K, grp);
  });
}

// V4: Shared Memory with Dequantization
TEST_P(GemmFp32Qint4ComparisonTest, V4_Shared_Mem_Dequant) {
  RunGemmTest("V4 (Shared Mem Dequant)", [&](float *d_in, uint8_t *d_w,
                                             uint16_t *d_s, int8_t * /*d_q_in*/,
                                             uint16_t * /*d_q_s*/, float *d_out,
                                             int M, int N, int K, int grp) {
    custom::gemm_a32_w4_b32x32_s32x32_dequant_cuda(d_in, d_w, d_s, d_out, M, N,
                                                   K, grp);
  });
}

// V5: V1_2 Kernel
TEST_P(GemmFp32Qint4ComparisonTest, V5_V1_2_Kernel) {
  RunGemmTest("V5 (V1_2 Kernel)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b32x32_pre_dequant_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V7: 16x16 threads
TEST_P(GemmFp32Qint4ComparisonTest, V7_16x16_Threads) {
  RunGemmTest("V7 (16x16 threads)", [&](float *d_in, uint8_t *d_w,
                                        uint16_t *d_s, int8_t * /*d_q_in*/,
                                        uint16_t * /*d_q_s*/, float *d_out,
                                        int M, int N, int K, int grp) {
    custom::gemm_a32_w4_b16x16_s32x32_cuda(d_in, d_w, d_s, d_out, M, N, K, grp);
  });
}

// V7_1: Packed Block 16x16
TEST_P(GemmFp32Qint4ComparisonTest, V7_1_Packed_Block_16x16) {
  RunGemmTest("V7_1 (Packed Block 16x16)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b16x16_cuda(d_in, d_w, d_s, d_q_in, d_q_s,
                                                d_out, M, N, K, grp);
              });
}

// V8: WMMA Tensor Core
TEST_P(GemmFp32Qint4ComparisonTest, V8_WMMA_Tensor_Core) {
  RunGemmTest("V8 (WMMA Tensor Core)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b32x32_wmma_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V9: WMMA Packed Block 16x16 (Output 32x32)
TEST_P(GemmFp32Qint4ComparisonTest, V9_WMMA_B16x16_S32x32) {
  RunGemmTest("V9 (WMMA Packed Block 16x16 S32x32)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b16x16_s32x32_wmma_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V10: WMMA Packed Block 16x16
TEST_P(GemmFp32Qint4ComparisonTest, V10_WMMA_B16x16_S16x16) {
  RunGemmTest("V10 (WMMA Packed Block 16x16)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b16x16_wmma_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V11: WMMA Packed Block 16x16 (Output 64x64)
TEST_P(GemmFp32Qint4ComparisonTest, V11_WMMA_B16x16_S64x64) {
  RunGemmTest("V11 (WMMA Block 16x16 S64x64)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b16x16_s64x64_wmma_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V12: WMMA Packed Block 16x16 S32x32 (Vectorized Load)
TEST_P(GemmFp32Qint4ComparisonTest, V12_WMMA_B16x16_S32x32_VL) {
  RunGemmTest("V12 (WMMA Block 16x16 S32x32 VL)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b16x16_s32x32_wmma_vl_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V13: WMMA Packed Block 16x16 S32x32 (Async Copy)
TEST_P(GemmFp32Qint4ComparisonTest, V13_WMMA_B16x16_S32x32_CPASYNC) {
  RunGemmTest("V13 (WMMA Block 16x16 S32x32 Async)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b16x16_s32x32_wmma_cpasync_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V14: WMMA Packed Block 8x16 S32x32 (Vectorized Load)
TEST_P(GemmFp32Qint4ComparisonTest, V14_WMMA_B8x16_S32x32_VL) {
  RunGemmTest("V14 (WMMA Block 8x16 S32x32 VL)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b8x16_s32x32_wmma_vl_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V15: WMMA Packed Block 8x16 S32x32 (Async Copy)
TEST_P(GemmFp32Qint4ComparisonTest, V15_WMMA_B8x16_S32x32_CPASYNC) {
  RunGemmTest("V15 (WMMA Block 8x16 S32x32 Async)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b8x16_s32x32_wmma_cpasync_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V16: WMMA Packed Block 16x16 S32x32 Split-K Async
TEST_P(GemmFp32Qint4ComparisonTest, V16_WMMA_B16x16_S32x32_SplitK_CPASYNC) {
  RunGemmTest("V16 (WMMA 16x16 S32x32 SplitK Async)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b16x16_s32x32_wmma_cpasync_splitk_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

// V17: WMMA Packed Block 8x16 S32x32 (Async Copy Loop Unroll Double Buffer)
TEST_P(GemmFp32Qint4ComparisonTest, V17_WMMA_B8x16_S32x32_CPASYNC_LU) {
  RunGemmTest("V17 (WMMA Block 8x16 S32x32 Async LU)",
              [&](float *d_in, uint8_t *d_w, uint16_t *d_s, int8_t *d_q_in,
                  uint16_t *d_q_s, float *d_out, int M, int N, int K, int grp) {
                custom::gemm_a32_w4_b8x16_s32x32_wmma_cpasync_lu_cuda(
                  d_in, d_w, d_s, d_q_in, d_q_s, d_out, M, N, K, grp);
              });
}

INSTANTIATE_TEST_SUITE_P(
  VariousSizes, GemmFp32Qint4ComparisonTest,
  ::testing::Values(GemmComparisonTestParam{32, 32, 32},
                    // Small sizes
                    GemmComparisonTestParam{4, 128, 128},
                    // Medium sizes
                    GemmComparisonTestParam{28, 3072, 256},
                    GemmComparisonTestParam{28, 3072, 3072},
                    // Large sizes
                    GemmComparisonTestParam{28, 3072, 8192},
                    GemmComparisonTestParam{28, 8192, 3072}));

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
