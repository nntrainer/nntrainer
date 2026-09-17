// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_interface.cpp
 * @date   13 August 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use ggml lib from cpu_backend
 */

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <ggml_interface.h>
#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_utils.h>
#include <string>
#include <thread>
#include <vector>

namespace nntrainer {

void __ggml_init() { nntr_ggml_init(); }

size_t __ggml_quantize_q4_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q6_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q8_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q8_0(src, dst, nrow, n_per_row, quant_weights);
}

void __ggml_quantize_row_q6_K(const float *src, void *dst, int64_t k) {
  __ggml_quantize_q6_K(src, dst, 1, k, nullptr);
}

template <>
void __ggml_quantize_row_q8_K(const float *src, void *dst, int64_t k) {
  nntr_quantize_row_q8_K(src, dst, k);
}

void __ggml_dequantize_row_q4_0(const void *x_raw, float *y, int64_t k) {
  nntr_dequantize_row_q4_0(x_raw, y, k);
}

void __ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  nntr_dequantize_row_q4_K(x_raw, y, k);
}

void __ggml_dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  nntr_dequantize_row_q6_K(x, y, k);
}

template <>
void __ggml_dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  nntr_dequantize_row_q8_K(x, y, k);
}

float __ggml_vec_dot_q6_K_q8_K(const unsigned int K,
                               const void *__restrict v_q6_K,
                               const void *__restrict v_q8_K) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ggml_vec_dot_q6_K_q8_K
  nntr_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_K, by, nrc);
  return result;
}

float __ggml_vec_dot_q6_K_f32(const unsigned int K, const void *v_q6_K,
                              const float *f) {
  // Quantization of activations
  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int q8_K_activation_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
  __ggml_quantize_row_q8_K(f, v_q8_activation.data(), K);

  return __ggml_vec_dot_q6_K_q8_K(K, v_q6_K, v_q8_activation.data());
}

float __ggml_vec_dot_q6_K(const unsigned int K, const void *__restrict v_q6_K,
                          const float *__restrict activation) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ggml_vec_dot_q6_K_q8_K

  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int q8_K_activation_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
  __ggml_quantize_row_q8_K(activation, v_q8_activation.data(), K);

  nntr_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_activation.data(), by,
                         nrc);
  return result;
}

size_t __ggml_q4_0_gemv_activation_size(const unsigned int K) {
  assert(K % QK4_0 == 0);
  return sizeof(block_q8_0) * (K / QK8_0);
}

void __ggml_quantize_q4_0_gemv_activation(const unsigned int K, const float *A,
                                          void *quantized_A) {
  assert(K % QK4_0 == 0);
  nntr_quantize_row_q8_0(A, quantized_A, K);
}

void __ggml_gemv_q4_0_rowwise_range(const unsigned int row_begin,
                                    const unsigned int row_end,
                                    const unsigned int K,
                                    const void *quantized_A, const void *B,
                                    float *C) {
  assert(K % QK4_0 == 0);
  if (row_begin >= row_end)
    return;

  const size_t blocks_per_row = K / QK4_0;
  const size_t weight_row_size = sizeof(block_q4_0) * blocks_per_row;
  const char *weight_data = static_cast<const char *>(B);
  for (size_t row = row_begin; row < row_end; ++row) {
    nntr_vec_dot_q4_0_q8_0(K, C + row, weight_data + row * weight_row_size,
                           quantized_A);
  }
}

void __ggml_repack_q4_0_to_q4_0_4(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  nntr_repack_q4_0_to_q4_0_4_bl(dst, 8, src, data_size, M, N);
}

void __ggml_repack_q4_0_to_q4_0_8(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  nntr_repack_q4_0_to_q4_0_8_bl(dst, 8, src, data_size, M, N);
}

void __ggml_repack_q8_0_to_q8_0_4x4(void *dst, void *src, size_t data_size,
                                    const unsigned int M,
                                    const unsigned int N) {
  nntr_repack_q8_0_to_q8_0_4_bl(dst, 4, src, data_size, M, N);
}

void __ggml_repack_q8_0_to_q8_0_4x8(void *dst, void *src, size_t data_size,
                                    const unsigned int M,
                                    const unsigned int N) {
  nntr_repack_q8_0_to_q8_0_4_bl(dst, 8, src, data_size, M, N);
}

void __ggml_repack_q4_K_to_q4_K_8(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  nntr_repack_q4_K_to_q4_K_8_bl(dst, 8, src, data_size, M, N);
}

} // namespace nntrainer
