// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	unittest_quantize_cl.cpp
 * @date	28 November 2024
 * @brief	Test setup for quantization OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cstring>
#include <gtest/gtest.h>
#include <utility>

#include "fallback_internal.h"
#include "int4_utils.h"
#include "nntrainer_test_util.h"
#include "q4_0_utils.h"
#include "swiglu_cl.h"
#include "tensor_dim.h"
#include "timer.h"
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <cpu_backend.h>
#include <fp16.h>
#include <layer_context.h>
#include <tensor.h>
#include "unittest_util.h"

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

using namespace nntrainer;

// Helper for Round to Nearest Even (RTE)

static int8_t round_half_to_even(float x) {
  float r = roundf(x);
  float d = r - x;
  if (fabsf(d) != 0.5f) {
    return (int8_t)r;
  }
  // If exactly halfway, round to even
  int ir = (int)r;
  return (int8_t)((ir % 2 == 0) ? ir : ir - (ir > 0 ? 1 : -1));
}

// CPU version of openvino_quantize_input_int4_pad for reference in tests
static void cpu_openvino_quantize_input_int4_pad(float *input, int8_t *quantized_input, uint16_t *scales,
                                                 unsigned int M, unsigned int K, unsigned int quantization_group_size) {
  int alignK = (K + quantization_group_size - 1) / quantization_group_size * quantization_group_size;
  int groups_in_row = alignK / quantization_group_size;
  
  for (int group_id = 0; group_id < M * groups_in_row; ++group_id) {
    int row_id = group_id / groups_in_row;
    int group_id_in_row = group_id % groups_in_row;
    int input_offset = (row_id * K) + (group_id_in_row * quantization_group_size);
    int output_offset = group_id * quantization_group_size;
    int max_quantize_block = quantization_group_size / 4;
    int quantize_block;
    
    if (group_id_in_row == groups_in_row - 1) {
      quantize_block = (quantization_group_size - (alignK - K)) / 4;
    } else {
      quantize_block = quantization_group_size / 4;
    }
    
    // Find maximum absolute value in the block
    float max_value = 0.0f;
    for (int i = 0; i < quantize_block; ++i) {
      for (int j = 0; j < 4; ++j) {
        int idx = input_offset + (i * 4) + j;
        // Simulate half precision for input
        float val = idx < row_id * K + K ? compute_fp16_to_fp32(compute_fp32_to_fp16(input[idx])) : 0.0f;
        float abs_val = fabsf(val);
        max_value = fmaxf(max_value, abs_val);
      }
    }
    // Simulate half precision for max_value
    max_value = compute_fp16_to_fp32(compute_fp32_to_fp16(max_value));
    // Simulate half precision for epsilon 0.001h
    float epsilon = compute_fp16_to_fp32(compute_fp32_to_fp16(0.001f));
    max_value = fmaxf(max_value, epsilon);
    
    // Calculate quantization scale
    float quan_scale = max_value / 127.0f;
    
    // Quantize the data
    for (int i = 0; i < quantize_block; ++i) {
      for (int j = 0; j < 4; ++j) {
        int input_idx = input_offset + (i * 4) + j;
        int output_idx = output_offset + (i * 4) + j;
        // Simulate half precision for input
        float val = (input_idx < row_id * K + K) ? compute_fp16_to_fp32(compute_fp32_to_fp16(input[input_idx])) : 0.0f;
        float quantized_val = val / quan_scale;
        // Round to nearest even (RTE)
        int8_t rounded_val = round_half_to_even(quantized_val);
        quantized_input[output_idx] = rounded_val;
      }
    }
    
    // Pad with zeros if necessary
    for (int i = quantize_block * 4; i < max_quantize_block * 4; ++i) {
      int output_idx = output_offset + i;
      quantized_input[output_idx] = 0;
    }
    
    // Store the scale
    // Kernel writes to group_id * 2 (interleaved with activation sum)
    scales[group_id * 2] = compute_fp32_to_fp16(quan_scale);
    scales[group_id * 2 + 1] = 0; // Placeholder for activation sum
  }
}

static void run_int4_quantize_input_test_(const uint32_t M, const uint32_t K,
                                          const int scale_group_size) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 200;

  // Allocate & initialize data
  // Input for kernel is half (uint16_t)
  uint16_t *input_ptr = (uint16_t *)allocateSVM(M * K * sizeof(uint16_t));
  int8_t *quantized_input_ptr = (int8_t *)allocateSVM(M * K / 2);
  // Scales size is doubled for interleaved storage
  uint16_t *scales_ptr =
    (uint16_t *)allocateSVM(M * K / scale_group_size * sizeof(uint16_t) * 2);

  std::vector<float> input =
    generate_random_vector<float, false>(M * K, -2.0f, 2.0f);

  for (unsigned int i = 0; i < M * K; ++i) {
    input_ptr[i] = compute_fp32_to_fp16(input[i]);
  }

  // CPU quantization for reference
  std::vector<int8_t> ref_quantized_input(M * K);
  // Ref scales size is doubled
  std::vector<uint16_t> ref_scales(M * K / scale_group_size * 2);
  cpu_openvino_quantize_input_int4_pad(input.data(), ref_quantized_input.data(), ref_scales.data(), M, K, scale_group_size);

  // GPU INT4 input quantization
  auto t3 = std::chrono::high_resolution_clock::now();
  nntrainer::openvino_quantize_input_int4_pad(
    input_ptr, quantized_input_ptr, scales_ptr, M, K, scale_group_size);
  clFinish(blas_cc->command_queue_inst_.GetCommandQueue());
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::cout << "INT4 input quantization : " << M << " x " << K << std::endl;
  std::cout << " - time : GPU = " << gpu_dt.count()
            << " ms" << std::endl;

  // Compare results
  bool quantized_data_match = true;
  bool scales_match = true;

  int mismatch_count = 0;
  for (unsigned int i = 0; i < M * K / 2; ++i) {
    if (quantized_input_ptr[i] != ref_quantized_input[i]) {
      mismatch_count++;
    }
  }
  
  float mismatch_ratio = (float)mismatch_count / (M * K / 2);
  if (mismatch_ratio > 0.01f) {
    quantized_data_match = false;
  }
  std::cout << " - quantized data mismatch count: " << mismatch_count << " (" << mismatch_ratio * 100.0f << "%)" << std::endl;

  float mse_scales = 0.0f;
  for (unsigned int i = 0; i < M * K / scale_group_size; ++i) {
    float val = compute_fp16_to_fp32(scales_ptr[i * 2]);
    float ref = compute_fp16_to_fp32(ref_scales[i * 2]);
    mse_scales += (val - ref) * (val - ref);
  }
  mse_scales /= (M * K / scale_group_size);
  
  if (mse_scales > 1e-5f) {
    scales_match = false;
  }
  std::cout << " - scales MSE: " << mse_scales << std::endl;

  EXPECT_TRUE(quantized_data_match);
  EXPECT_TRUE(scales_match);

  std::cout << " - quantized data match: " << (quantized_data_match ? "YES" : "NO") << std::endl;
  std::cout << " - scales match: " << (scales_match ? "YES" : "NO") << std::endl;

  freeSVM(input_ptr);
  freeSVM(quantized_input_ptr);
  freeSVM(scales_ptr);
}

#define DECLARE_int4_quantize_input_test_M_K_G(M, K, G)                        \
  TEST(nntrainer_blas_kernel, int4_quantize_input_test_##M##_##K##_##G) {      \
    run_int4_quantize_input_test_(M, K, G);                                    \
  }

DECLARE_int4_quantize_input_test_M_K_G(32, 3072, 32);
DECLARE_int4_quantize_input_test_M_K_G(128, 3072, 32);
DECLARE_int4_quantize_input_test_M_K_G(256, 3072, 32);
DECLARE_int4_quantize_input_test_M_K_G(512, 3072, 32);
DECLARE_int4_quantize_input_test_M_K_G(1024, 3072, 32);

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
