// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   custom_dot_wrapper_cuda.cpp
 * @date   23 December 2024
 * @brief  CUDA wrapper implementation for custom dot operations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Daekyoung Jung <dk11.jung@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "custom_dot_wrapper_cuda.h"
#include "gemm_int4_cuda.h"
#include "quantize_cuda.h"
#include <float_tensor.h>
#include <int4_tensor.h>
#include <stdexcept>

namespace custom {

// GEMV INT4 CUDA - TODO
void gemv_int4_cuda(const char *mdata, const unsigned short *scales,
                    const float *data, float *rdata, unsigned int K,
                    unsigned int N, unsigned int quantization_group_size) {
  // TODO: Implement GEMV kernel for M=1 case
  // This should perform: rdata[N] = data[K] * mdata[K x N]
  // where mdata is int4 quantized with scales
}

void gemm_a32_w4_default_cuda(const float *data, const char *mdata,
                              const unsigned short *scales,
                              const unsigned short *input_scales,
                              void *d_quantized_input,
                              void *d_input_scales_temp, float *rdata,
                              unsigned int M, unsigned int N, unsigned int K,
                              unsigned int quantization_group_size) {

  // Cast buffers to appropriate types
  int8_t *d_quantized_input_typed = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_input_scales_temp_typed =
    static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize input on GPU
  quantize_input_int8_pad_cuda(data, d_quantized_input_typed,
                               d_input_scales_temp_typed, M, K,
                               quantization_group_size, 0);

  // Call V14 kernel (best performance)
  gemm_a8_w4_b8x16_s32x32_wmma_vl(d_quantized_input_typed, mdata, scales,
                                  d_input_scales_temp_typed, rdata, M, N, K,
                                  quantization_group_size);
}

void gemm_a32_w4_b16x16_naive_cuda(const float *input, const void *weights,
                                   const void *scales, float *output,
                                   unsigned int M, unsigned int N,
                                   unsigned int K,
                                   unsigned int quantization_group_size) {
  // Call kernel launcher function (defined in gemm_int4_cuda.cu)
  gemm_a32_w4_b16x16_naive(input, weights, scales, output, M, N, K,
                           quantization_group_size);
}

void gemm_a32_w4_b32x32_s32x32_cuda(const float *input, const void *weights,
                                    const void *scales, float *output,
                                    unsigned int M, unsigned int N,
                                    unsigned int K,
                                    unsigned int quantization_group_size) {
  // Call kernel launcher function (defined in gemm_int4_cuda.cu)
  gemm_a32_w4_b32x32_s32x32(input, weights, scales, output, M, N, K,
                            quantization_group_size);
}

void gemm_a32_w4_b32x32_s32x32_dequant_cuda(
  const float *input, const void *weights, const void *scales, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {
  // Call kernel launcher function (defined in gemm_int4_cuda.cu)
  gemm_a32_w4_b32x32_s32x32_dequant(input, weights, scales, output, M, N, K,
                                    quantization_group_size);
}

void gemm_a32_w4_b32x32_pre_dequant_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call gemm_a8_w4_b32x32_pre_dequant (V1_2 kernel)
  gemm_a8_w4_b32x32_pre_dequant(d_quantized, weights, scales, d_in_scales,
                                output, M, N, K, quantization_group_size);
}

void gemm_a32_w4_b16x16_s32x32_cuda(const float *input, const void *weights,
                                    const void *scales, float *output,
                                    unsigned int M, unsigned int N,
                                    unsigned int K,
                                    unsigned int quantization_group_size) {
  // Call kernel launcher function (defined in gemm_int4_cuda.cu)
  gemm_a32_w4_b16x16_s32x32(input, weights, scales, output, M, N, K,
                            quantization_group_size);
}

void gemm_a32_w4_b32x32_wmma_cuda(const float *input, const void *weights,
                                  const void *scales, void *d_quantized_input,
                                  void *d_input_scales_temp, float *output,
                                  unsigned int M, unsigned int N,
                                  unsigned int K,
                                  unsigned int quantization_group_size) {
  // 1. Quantize FP32 input to INT8
  quantize_input_int8_pad_cuda(input, static_cast<int8_t *>(d_quantized_input),
                               static_cast<uint16_t *>(d_input_scales_temp), M,
                               K, quantization_group_size);

  // 2. Call WMMA kernel
  gemm_a8_w4_b32x32_wmma(d_quantized_input, weights, scales,
                         d_input_scales_temp, output, M, N, K,
                         quantization_group_size);
}

void gemm_a32_w4_b16x16_cuda(const float *input, const void *weights,
                             const void *scales, void *d_quantized_input,
                             void *d_input_scales_temp, float *output,
                             unsigned int M, unsigned int N, unsigned int K,
                             unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call kernel
  gemm_a8_w4_b16x16(d_quantized, weights, scales, d_in_scales, output, M, N, K,
                    quantization_group_size);
}

void gemm_a32_w4_b16x16_s32x32_wmma_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call kernel wrapper
  gemm_a8_w4_b16x16_s32x32_wmma(d_quantized, weights, scales, d_in_scales,
                                output, M, N, K, quantization_group_size);
}

void gemm_a32_w4_b16x16_s64x64_wmma_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call kernel wrapper
  gemm_a8_w4_b16x16_s64x64_wmma(d_quantized, weights, scales, d_in_scales,
                                output, M, N, K, quantization_group_size);
}

void gemm_a32_w4_b16x16_s32x32_wmma_vl_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call kernel wrapper
  gemm_a8_w4_b16x16_s32x32_wmma_vl(d_quantized, weights, scales, d_in_scales,
                                   output, M, N, K, quantization_group_size);
}

void gemm_a32_w4_b16x16_s32x32_wmma_cpasync_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call kernel wrapper
  gemm_a8_w4_b16x16_s32x32_wmma_cpasync(d_quantized, weights, scales,
                                        d_in_scales, output, M, N, K,
                                        quantization_group_size);
}

void gemm_a32_w4_b8x16_s32x32_wmma_vl_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call kernel wrapper
  gemm_a8_w4_b8x16_s32x32_wmma_vl(d_quantized, weights, scales, d_in_scales,
                                  output, M, N, K, quantization_group_size);
}

void gemm_a32_w4_b8x16_s32x32_wmma_cpasync_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call kernel wrapper
  gemm_a8_w4_b8x16_s32x32_wmma_cpasync(d_quantized, weights, scales,
                                       d_in_scales, output, M, N, K,
                                       quantization_group_size);
}

void gemm_a32_w4_b16x16_s32x32_wmma_cpasync_splitk_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Zero-initialize output buffer for Split-K atomicAdd
  cudaMemset(output, 0, M * N * sizeof(float));

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call kernel wrapper
  gemm_a8_w4_b16x16_s32x32_wmma_cpasync_splitk(d_quantized, weights, scales,
                                               d_in_scales, output, M, N, K,
                                               quantization_group_size);
}

void gemm_a32_w4_b8x16_s32x32_wmma_cpasync_lu_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call kernel wrapper
  gemm_a8_w4_b8x16_s32x32_wmma_cpasync_lu(d_quantized, weights, scales,
                                          d_in_scales, output, M, N, K,
                                          quantization_group_size);
}

void gemm_a32_w4_b16x16_wmma_cuda(const float *input, const void *weights,
                                  const void *scales, void *d_quantized_input,
                                  void *d_input_scales_temp, float *output,
                                  unsigned int M, unsigned int N,
                                  unsigned int K,
                                  unsigned int quantization_group_size) {
  // Cast buffers to appropriate types
  int8_t *d_quantized = static_cast<int8_t *>(d_quantized_input);
  uint16_t *d_in_scales = static_cast<uint16_t *>(d_input_scales_temp);

  // Quantize FP32 input to INT8 with scales
  quantize_input_int8_pad_cuda(input, d_quantized, d_in_scales, M, K,
                               quantization_group_size, 0);

  // Call kernel wrapper
  gemm_a8_w4_b16x16_wmma(d_quantized, weights, scales, d_in_scales, output, M,
                         N, K, quantization_group_size);
}

void custom_dot_cuda(nntrainer::Tensor &output, nntrainer::Tensor weight,
                     nntrainer::Tensor input, unsigned int from,
                     unsigned int to) {

  // 1. Check if input is FloatTensor
  if (input.getDataType() != nntrainer::TensorDim::DataType::FP32) {
    throw std::invalid_argument(
      "custom_dot_cuda: input must be FloatTensor (FP32)");
  }

  // 2. Check if weight is QINT4
  if (weight.getDataType() != nntrainer::TensorDim::DataType::QINT4) {
    throw std::invalid_argument(
      "custom_dot_cuda: weight must be QINT4 quantized");
  }

  // 3. Get dimensions
  // Assuming input is [M, K] and weight is [K, N]
  unsigned int M = input.getDim().height();
  unsigned int K = input.getDim().width();
  unsigned int N = output.getDim().width();

  // 4. Get data pointers
  float *data = input.getData<float>();
  char *mdata = weight.getData<char>();
  float *rdata = output.getData<float>();

  // 5. Get scale data for QINT4
  // QINT4 tensors have scale information
  unsigned short *scales = weight.getScale<unsigned short>();

  // 6. Get quantization group size
  unsigned int quantization_group_size = nntrainer::Int4QTensor::getGroupSize();

  // 7. Dispatch based on M (similar to dotQInteger with ENABLE_OPENCL)
  if (M == 1) {
    // Vector-Matrix multiplication (GEMV)
    gemv_int4_cuda(mdata, scales, data, rdata, K, N, quantization_group_size);
  } else {
    // Matrix-Matrix multiplication (GEMM)
    // TODO: Implement proper buffer management for quantization
    // For now, passing nullptr for temporary buffers (this will cause issues)
    // Need to allocate d_quantized_input and d_input_scales_temp buffers
    gemm_a32_w4_default_cuda(data, mdata, scales, nullptr, nullptr,
                             nullptr, // TODO: allocate and pass buffers
                             rdata, M, N, K, quantization_group_size);
  }
}

} // namespace custom
