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
#include "unittest_util.h"
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <cpu_backend.h>
#include <fp16.h>
#include <layer_context.h>
#include <tensor.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

using namespace nntrainer;

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
  cpu_quantize_input_int4_pad(input.data(), ref_quantized_input.data(),
                              ref_scales.data(), M, K, scale_group_size);

  // GPU INT4 input quantization
  auto t3 = std::chrono::high_resolution_clock::now();
  nntrainer::openvino_quantize_input_int4_pad(
    input_ptr, quantized_input_ptr, scales_ptr, M, K, scale_group_size);
  clFinish(blas_cc->command_queue_inst_.GetCommandQueue());
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::cout << "INT4 input quantization : " << M << " x " << K << std::endl;
  std::cout << " - time : GPU = " << gpu_dt.count() << " ms" << std::endl;

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
  std::cout << " - quantized data mismatch count: " << mismatch_count << " ("
            << mismatch_ratio * 100.0f << "%)" << std::endl;

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

  std::cout << " - quantized data match: "
            << (quantized_data_match ? "YES" : "NO") << std::endl;
  std::cout << " - scales match: " << (scales_match ? "YES" : "NO")
            << std::endl;

  freeSVM(input_ptr);
  freeSVM(quantized_input_ptr);
  freeSVM(scales_ptr);
}

#define DECLARE_int4_quantize_input_test_M_K_G(M, K, G)                        \
  TEST(nntrainer_blas_kernel, int4_quantize_input_test_##M##_##K##_##G) {      \
    run_int4_quantize_input_test_(M, K, G);                                    \
  }

DECLARE_int4_quantize_input_test_M_K_G(32, 3072, 32);
DECLARE_int4_quantize_input_test_M_K_G(63, 3072, 32);
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
