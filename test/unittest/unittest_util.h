// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_util.h
 * @brief  Shared utility functions for unit tests
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef NNTRAINER_UNITTEST_UTIL_H
#define NNTRAINER_UNITTEST_UTIL_H

#include <cstddef>
#include <random>
#include <vector>

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

namespace nntrainer {

// Generate a random vector of given size and range.
// The template type T is expected to be convertible from float.
// This function is used across many unit tests.

template <typename T, bool random_init = false>
std::vector<T> generate_random_vector(size_t size, float min_val = -1.F,
                                      float max_val = 1.F) {
  std::random_device rd;
  auto init_val = random_init ? rd() : 42;
  std::mt19937 gen(init_val);
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}
#if defined(ENABLE_OPENCL)
// Allocate SVM memory using the OpenCL context.
void *allocateSVM(size_t size_bytes);

// Release SVM memory.
void freeSVM(void *ptr);
#endif
// Helper for Round to Nearest Even (RTE)
int8_t round_half_to_even(float x);

// CPU reference implementation for INT4 quantization
void cpu_quantize_input_int8_pad(float *input, int8_t *quantized_input,
                                 uint16_t *scales, unsigned int M,
                                 unsigned int K,
                                 unsigned int quantization_group_size);

void printMatrixI(const char *name, float *data, int Y, int X);

std::vector<float> generate_vector(const size_t size, float min_val,
                                   float max_val);

std::vector<float> generate_01_vector(const size_t size,
                                      const float ones_ratio);

// Standard FP32 GEMM for Reference
// C = A * B
// A: M x K
// B: K x N
// C: M x N
void gemm_fp32_ref(const float *input, const float *weights, float *output,
                   unsigned int M, unsigned int N, unsigned int K);

} // namespace nntrainer

#endif // NNTRAINER_UNITTEST_UTIL_H
