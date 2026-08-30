// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels.cpp
 * @date	14 May 2024
 * @brief	Common blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "blas_kernels_templates.h"
#include <cl_kernels/cl_kernels.h>

#include "cl_tensor_view.h"
#include "util_func.h"
#include "v8c_pack_cache.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fp16.h>
#include <opencl_loader.h>
#include <thread>

namespace nntrainer {

void gemv_int4_async_cl(std::vector<void *> weights,
                        std::vector<uint16_t *> scales, uint16_t *input,
                        std::vector<uint16_t *> outputs, unsigned int K,
                        std::vector<unsigned int> Ns,
                        unsigned int quantization_group_size) {
  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  const bool scale_row_major = false;

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_kernel, "fully_connected_gpu_int4_gemv");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for fully_connected_gpu_int4_gemv");
    return;
  }

  const int work_group_size[3] = {16, 1, 16};

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    int arg = 0;
    int N = Ns[i];
    const auto N_GROUP_SIZE = 32; // due to input data format
    const unsigned int alignN = align(N, N_GROUP_SIZE);
    void *weight = weights[i];
    uint16_t *scale = scales[i];
    uint16_t *output = outputs[i];
    result = kernel_ptr->SetKernelSVMArguments(arg++, input);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 0 for fully_connected_gpu_int4_gemv");

    kernel_ptr->SetKernelSVMArguments(arg++, scale);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 1 for fully_connected_gpu_int4_gemv");

    result = kernel_ptr->SetKernelSVMArguments(arg++, output);

    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 2 for fully_connected_gpu_int4_gemv");

    result = kernel_ptr->SetKernelSVMArguments(arg++, weight);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 3 for fully_connected_gpu_int4_gemv");

    result = kernel_ptr->SetKernelArguments(arg++, &K, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 4 for fully_connected_gpu_int4_gemv");

    result = kernel_ptr->SetKernelArguments(arg++, &N, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 5 for fully_connected_gpu_int4_gemv");

    int q_group_size = quantization_group_size;
    result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 6 for fully_connected_gpu_int4_gemv");

    int row_major = scale_row_major;
    result = kernel_ptr->SetKernelArguments(arg++, &row_major, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 7 for fully_connected_gpu_int4_gemv");

    const int work_groups_count[3] = {(int)(alignN / 2), 1, 16};
    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size);
    if (!result) {
      throw std::runtime_error(
        "Failed to dispatch kernel for fully_connected_gpu_int4_gemv");
      return;
    }
  }

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    blas_cc->command_queue_inst_.enqueueSVMMap(outputs[i],
                                               Ns[i] * sizeof(uint16_t), true);
  }
  if (!result) {
    throw std::runtime_error(
      "Failed to read output data for fully_connected_gpu_int4_gemv");
    return;
  }
}

void gemv_int4_cl(char *weight, uint16_t *scale, uint16_t *input,
                  uint16_t *output, unsigned int K, unsigned int N,
                  unsigned int quantization_group_size) {
  const auto N_GROUP_SIZE = 32; // due to input data format
  const unsigned int alignN = align(N, N_GROUP_SIZE);

  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  const bool scale_row_major = false;

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_kernel, "fully_connected_gpu_int4_gemv");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for fully_connected_gpu_int4_gemv");
    return;
  }

  int arg = 0;

  result = kernel_ptr->SetKernelSVMArguments(arg++, input);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 0 for fully_connected_gpu_int4_gemv");

  kernel_ptr->SetKernelSVMArguments(arg++, scale);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 1 for fully_connected_gpu_int4_gemv");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);

  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 for fully_connected_gpu_int4_gemv");

  result = kernel_ptr->SetKernelSVMArguments(arg++, weight);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 for fully_connected_gpu_int4_gemv");

  result = kernel_ptr->SetKernelArguments(arg++, &K, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 4 for fully_connected_gpu_int4_gemv");

  result = kernel_ptr->SetKernelArguments(arg++, &N, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 5 for fully_connected_gpu_int4_gemv");

  int q_group_size = quantization_group_size;
  result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 6 for fully_connected_gpu_int4_gemv");

  int row_major = scale_row_major;
  result = kernel_ptr->SetKernelArguments(arg++, &row_major, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 7 for fully_connected_gpu_int4_gemv");

  const int work_groups_count[3] = {(int)(alignN / 2), 1, 16};
  const int work_group_size[3] = {16, 1, 16};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    throw std::runtime_error(
      "Failed to dispatch kernel for fully_connected_gpu_int4_gemv");
    return;
  }

  /// @todo synchronize when only needed
  blas_cc->command_queue_inst_.enqueueSVMMap(output, N * sizeof(uint16_t),
                                             true);
  if (!result) {
    throw std::runtime_error(
      "Failed to read output data for fully_connected_gpu_int4_gemv");
    return;
  }
}

void gemv_int4_async_cl(std::vector<void *> weights,
                        std::vector<uint16_t *> scales, float *input,
                        std::vector<float *> outputs, unsigned int K,
                        std::vector<unsigned int> Ns,
                        unsigned int quantization_group_size) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  // copy fp32 input to fp16
  copy_fp32_u16(K, input, (uint16_t *)clbuffInstance.getSVMInput());
  std::vector<uint16_t *> output_vec;

  for (int i = 0; i < Ns.size(); ++i) {
    output_vec.push_back((uint16_t *)clbuffInstance.getSVMOutput(i));
  }

  gemv_int4_async_cl(weights, scales, (uint16_t *)clbuffInstance.getSVMInput(),
                     output_vec, K, Ns, quantization_group_size);

  for (int i = 0; i < Ns.size(); ++i) {
    copy_u16_fp32(Ns[i], (uint16_t *)clbuffInstance.getSVMOutput(i),
                  outputs[i]);
  }
}

void gemv_int4_cl(char *weight, uint16_t *scale, float *input, float *output,
                  unsigned int K, unsigned int N,
                  unsigned int quantization_group_size) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  // copy fp32 input to fp16
  copy_fp32_u16(K, input, (uint16_t *)clbuffInstance.getSVMInput());

  // perform int4 matmul
  gemv_int4_cl(weight, scale, (uint16_t *)clbuffInstance.getSVMInput(),
               (uint16_t *)clbuffInstance.getSVMOutput(), K, N,
               quantization_group_size);

  // copy fp16 output to fp32
  copy_u16_fp32(N, (uint16_t *)clbuffInstance.getSVMOutput(), output);
}

void gemm_q4_0_async_cl(std::vector<void *> matAdata, float *matBdata,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> Ns, unsigned int K) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  int padding = 0;
  if (M % 8 > 0) {
    padding = 8 - (M % 8);
  }

  int padded_M = M + padding;

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    q4_0_ab_bi_8x4_kernel, "kernel_mul_mat_Ab_Bi_8x4");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for kernel_mul_mat_Ab_Bi_8x4");
    return;
  }

  bool result = false;

  /// @note Transpose fp32 input. This can only be done once
  transpose_32_16(matBdata, M, K);

  const int work_group_size[3] = {1, 128, 1};

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    int N = Ns[i];
    void *mdata = matAdata[i];
    float *rdata = matCdata[i];

    unpack_q4_0x8_transpose16(mdata, (uint16_t *)clbuffInstance.getSVMScale(i),
                              (uint16_t *)clbuffInstance.getSVMQuant(i), N, K);

    int arg = 0;

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant(i));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 0 for kernel_mul_mat_Ab_Bi_8x4");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale(i));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 1 for kernel_mul_mat_Ab_Bi_8x4");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMInput());
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 2 for kernel_mul_mat_Ab_Bi_8x4");

    result = kernel_ptr->SetKernelSVMArguments(arg++, rdata);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 3 for kernel_mul_mat_Ab_Bi_8x4");

    result = kernel_ptr->SetKernelArguments(arg++, &N, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 4 for kernel_mul_mat_Ab_Bi_8x4");

    result = kernel_ptr->SetKernelArguments(arg++, &padded_M, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 5 for kernel_mul_mat_Ab_Bi_8x4");

    result = kernel_ptr->SetKernelArguments(arg++, &K, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 6 for kernel_mul_mat_Ab_Bi_8x4");

    result = kernel_ptr->SetKernelArguments(arg++, &M, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 7 for kernel_mul_mat_Ab_Bi_8x4");
    const int work_groups_count[3] = {(int)ceil(M / 8.0f), (int)N / 4, 1};

    // Perform Matrix Multiplication
    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size);
    if (!result) {
      throw std::runtime_error(
        "Failed to dispatch kernel for kernel_mul_mat_Ab_Bi_8x4");
    }
  }

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    blas_cc->command_queue_inst_.enqueueSVMMap(matCdata[i],
                                               M * Ns[i] * sizeof(float), true);
  }
}

void gemm_q4_0_cl(void *matAdata, float *matBdata, float *matCdata,
                  unsigned int M, unsigned int N, unsigned int K) {
  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t q_size_bytes = N * (K / 2);
  size_t d_size_bytes = N * (K / 32) * 2;

  // 1. Preprocess matrix A
  // 1.1 Unpack the Q4_0x8 matrix A to make a struct of array (src_q, src_d)
  // 1.2 Perform 2D 16-bit transpose src_q, src_d
  unpack_q4_0x8_transpose16(matAdata, (uint16_t *)clbuffInstance.getSVMScale(),
                            (uint16_t *)clbuffInstance.getSVMQuant(), N, K);

  // 2. Preprocess matrix B: Transpose the Matrix B and convert to FP16
  /// @note mat mul will compute 8 elements at once, padding
  // will be added if M is not multiple of 8.
  transpose_32_16(matBdata, M, K);

  int padding = 0;
  if (M % 8 > 0) {
    padding = 8 - (M % 8);
  }

  int padded_M = M + padding;

  // 3. Perform Matrix Multiplication
  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    q4_0_ab_bi_8x4_kernel, "kernel_mul_mat_Ab_Bi_8x4");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for kernel_mul_mat_Ab_Bi_8x4");
    return;
  }

  int arg = 0;

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 0 for kernel_mul_mat_Ab_Bi_8x4");

  kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 1 for kernel_mul_mat_Ab_Bi_8x4");

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMInput());

  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 for kernel_mul_mat_Ab_Bi_8x4");

  result = kernel_ptr->SetKernelSVMArguments(arg++, matCdata);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 for kernel_mul_mat_Ab_Bi_8x4");

  result = kernel_ptr->SetKernelArguments(arg++, &N, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 4 for kernel_mul_mat_Ab_Bi_8x4");

  result = kernel_ptr->SetKernelArguments(arg++, &padded_M, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 5 for kernel_mul_mat_Ab_Bi_8x4");

  result = kernel_ptr->SetKernelArguments(arg++, &K, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 6 for kernel_mul_mat_Ab_Bi_8x4");

  result = kernel_ptr->SetKernelArguments(arg++, &M, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 7 for kernel_mul_mat_Ab_Bi_8x4");

  const int work_groups_count[3] = {(int)ceil(M / 8.0f), (int)N / 4, 1};
  const int work_group_size[3] = {1, 128, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    throw std::runtime_error(
      "Failed to dispatch kernel for kernel_mul_mat_Ab_Bi_8x4");
    return;
  }

  /// @todo synchronize when only needed
  blas_cc->command_queue_inst_.enqueueSVMMap(matCdata, M * N * sizeof(float),
                                             true);
  if (!result) {
    throw std::runtime_error(
      "Failed to read output data for kernel_mul_mat_Ab_Bi_8x4");
    return;
  }
}

void gemm_int4_async_cl(float *input, std::vector<void *> weights,
                        std::vector<uint16_t *> scales,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> Ns, unsigned int K,
                        unsigned int quantization_group_size) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  bool result = false;

  // copy fp32 input to fp16
  copy_fp32_u16(M * K, input, (uint16_t *)clbuffInstance.getSVMInput());

  std::vector<cl_event> quantize_event(1);
  {
    int alignK = align(K, quantization_group_size);

    ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
      int4_quantize_input_kernel, "quantize_input_int4_pad");
    if (!kernel_ptr) {
      throw std::runtime_error("Failed to get kernel_ptr for quantize_input");
      return;
    }

    int arg = 0;

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMInput());
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 0 for "
                               "quantize_input");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 1 for "
                               "quantize_input");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 2 for "
                               "quantize_input");

    int size_n = Ns[0];
    int size_k = K;
    int q_group_size = quantization_group_size;

    result = kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 3 for "
                               "quantize_input");

    result = kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 4 for "
                               "quantize_input");

    result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 5 for "
                               "quantize_input");

    std::array<size_t, 3> global_work_size = {
      (M * alignK) / quantization_group_size, 1, 1};

    blas_cc->command_queue_inst_.enqueueKernel(
      kernel_ptr->GetKernel(), global_work_size.size(), global_work_size.data(),
      nullptr, 0, nullptr, &quantize_event.front());
  }

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    int N = Ns[i];
    const auto N_GROUP_SIZE = 32; // due to input data format
    const unsigned int alignN = align(N, N_GROUP_SIZE);

    const bool scale_row_major = false;

    ClContext::SharedPtrClKernel kernel_ptr =
      blas_cc->registerClKernel(gemm_int4_kernel, "fc_bf_tiled_kernel_default");
    if (!kernel_ptr) {
      throw std::runtime_error(
        "Failed to get kernel_ptr for fc_bf_tiled_kernel_default");
      return;
    }

    int arg = 0;
    int size_n = N;
    int size_k = K;
    int q_group_size = quantization_group_size;
    int row_major = scale_row_major;

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMInput());

    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 0 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelSVMArguments(arg++, scales[i]);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 1 for fc_bf_tiled_kernel_default");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMOutput(i));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 2 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelSVMArguments(arg++, weights[i]);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 3 for fc_bf_tiled_kernel_default");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 4 for fc_bf_tiled_kernel_default");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 5 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelArguments(arg++, &M, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 6 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 7 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 8 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 9 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelArguments(arg++, &row_major, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 10 for fc_bf_tiled_kernel_default");

    const int work_groups_count[3] = {(int)(alignN / 2),
                                      (int)(align(ceilDiv(M, 8), 8)), 1};
    const int work_group_size[3] = {16, 8, 1};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size, nullptr, quantize_event);
    if (!result) {
      throw std::runtime_error(
        "Failed to dispatch kernel for fc_bf_tiled_kernel_default");
      return;
    }
  }

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    blas_cc->command_queue_inst_.enqueueSVMMap(
      clbuffInstance.getSVMOutput(i), M * Ns[i] * sizeof(uint16_t), true);

    // copy fp16 output to fp32
    copy_u16_fp32(M * Ns[i], (uint16_t *)clbuffInstance.getSVMOutput(i),
                  matCdata[i]);
  }
}

///  @note remove this when fp16 is enabled on Windows
void sgemm_int4_cl(float *input, char *weight, uint16_t *scale, float *output,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int quantization_group_size) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  // copy fp32 input to fp16
  copy_fp32_u16(M * K, input, (uint16_t *)clbuffInstance.getSVMInput());

  // perform int4 matmul
  gemm_int4_cl(clbuffInstance.getSVMInput(), weight, scale,
               clbuffInstance.getSVMOutput(), M, N, K, quantization_group_size);

  // copy fp16 output to fp32
  copy_u16_fp32(M * N, (uint16_t *)clbuffInstance.getSVMOutput(), output);
}

void gemm_int4_cl(void *input, void *weights, void *scales, void *output,
                  unsigned int M, unsigned int N, unsigned int K,
                  unsigned int quantization_group_size) {
  int alignK = align(K, quantization_group_size);
  const auto N_GROUP_SIZE = 32; // due to input data format
  int alignN = align(N, N_GROUP_SIZE);

  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();
  const bool scale_row_major = false;

  std::vector<cl_event> quantize_event(1);
  {
    ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
      int4_quantize_input_kernel, "quantize_input_int4_pad");
    if (!kernel_ptr) {
      throw std::runtime_error("Failed to get kernel_ptr for quantize_input");
      return;
    }

    int arg = 0;
    int size_n = N;
    int size_k = K;
    int q_group_size = quantization_group_size;

    result = kernel_ptr->SetKernelSVMArguments(arg++, input);

    if (!result)
      throw std::runtime_error("Failed to set kernel argument 0 for "
                               "quantize_input");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 1 for "
                               "quantize_input");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 2 for "
                               "quantize_input");

    result = kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 3 for "
                               "quantize_input");

    result = kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 4 for "
                               "quantize_input");

    result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 5 for "
                               "quantize_input");

    std::array<size_t, 3> global_work_size = {
      (M * alignK) / quantization_group_size, 1, 1};

    blas_cc->command_queue_inst_.enqueueKernel(
      kernel_ptr->GetKernel(), global_work_size.size(), global_work_size.data(),
      nullptr, 0, nullptr, &quantize_event.front());
  }

  // 3. Perform Matrix Multiplication
  ClContext::SharedPtrClKernel kernel_ptr =
    blas_cc->registerClKernel(gemm_int4_kernel, "fc_bf_tiled_kernel_default");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for fc_bf_tiled_kernel_default");
    return;
  }

  int arg = 0;
  int size_n = N;
  int size_k = K;
  int q_group_size = quantization_group_size;
  int row_major = scale_row_major;

  result = kernel_ptr->SetKernelSVMArguments(arg++, input);

  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 0 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelSVMArguments(arg++, scales);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 1 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelSVMArguments(arg++, weights);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 for fc_bf_tiled_kernel_default");

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 4 for fc_bf_tiled_kernel_default");

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 5 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelArguments(arg++, &M, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 6 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 7 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 8 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 9 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelArguments(arg++, &row_major, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 10 for fc_bf_tiled_kernel_default");

  const int work_groups_count[3] = {(int)(alignN / 2),
                                    (int)(align(ceilDiv(M, 8), 8)), 1};
  const int work_group_size[3] = {16, 8, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size, nullptr, quantize_event);
  if (!result) {
    throw std::runtime_error(
      "Failed to dispatch kernel for fc_bf_tiled_kernel_default");
    return;
  }

  /// @todo synchronize when only needed
  blas_cc->command_queue_inst_.enqueueSVMMap(output, M * N * sizeof(uint16_t),
                                             true);
  if (!result) {
    throw std::runtime_error(
      "Failed to read output data for fc_bf_tiled_kernel_default");
    return;
  }
}

void sgemv_q6_k_cl(void *matAdata, float *vecXdata, float *vecYdata,
                   unsigned int M, unsigned int N) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_q6_k_sgemv_ptr;

  kernel_q6_k_sgemv_ptr =
    blas_cc->registerClKernel(q6_k_sgemv_kernel, "kernel_mul_mv_q6_K_f32");

  if (!kernel_q6_k_sgemv_ptr) {
    ml_loge("Failed to register kernel_q6_k_sgemv_ptr");
    return;
  }

  const size_t q6k_bytes = 210 * M * N / 256;

  result = blas_cc->command_queue_inst_.enqueueSVMUnmap(matAdata);
  if (!result) {
    ml_loge("Failed to write data to input buffer A for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = blas_cc->command_queue_inst_.enqueueSVMUnmap(vecXdata);
  if (!result) {
    ml_loge("Failed to write data to input buffer B for kernel_q6_k_sgemv_ptr");
    return;
  }

  int ne00 = M; // number of rows in matrix X
  int ne01 = N; // number of columns in matrix X
  int ne02 = 1; // number of channels in matrix X
  int ne10 = M; // number of rows in vector A
  int ne11 = 1; // number of columns in vector A
  int ne12 = 1; // number of channels in vector A
  int ne13 = 1; // number of channels in vector A (Need to check)
  int ne0 = N;  // number of rows in output vector Y
  int ne1 = 1;  // number of columns in output vector Y

  int r2 = 1; // number of batches in vector A
  int r3 = 1; // number of batches in matrix X

  int nth0 = 2;
  int nth1 = 16;

  cl_ulong offset0 = 0;
  cl_ulong offset1 = 0;
  cl_ulong offsetd = 0;

  result = kernel_q6_k_sgemv_ptr->SetKernelSVMArguments(0, matAdata);

  if (!result) {
    ml_loge("Failed to set kernel argument 0 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(1, &offset0, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 1 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelSVMArguments(2, vecXdata);

  if (!result) {
    ml_loge("Failed to set kernel argument 2 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(3, &offset1, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 3 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelSVMArguments(4, vecYdata);

  if (!result) {
    ml_loge("Failed to set kernel argument 4 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(5, &offsetd, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 5 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(6, &ne00, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 6 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(7, &ne01, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 7 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(8, &ne02, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 8 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(9, &ne10, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 9 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(10, &ne12, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 10 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(11, &ne0, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 11 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(12, &ne1, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 12 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(13, &r2, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 13 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(14, &r3, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 14 for kernel_q6_k_sgemv_ptr");
    return;
  }

#define N_SIMDWIDTH 16
#define N_SIMDGROUP 2

  const int work_groups_count[3] = {((ne0 + N_SIMDGROUP - 1) / N_SIMDGROUP) *
                                      (N_SIMDGROUP * N_SIMDWIDTH),
                                    ne1, 1};
  /// @todo: create a group size by device & input
  const int work_group_size[3] = {32, 1, 1};

  result = opencl::CommandQueueManager::Global().DispatchCommand(
    kernel_q6_k_sgemv_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel q6_k_sgemv");
    return;
  }

  result = blas_cc->command_queue_inst_.enqueueSVMMap(vecYdata,
                                                      N * sizeof(float), true);

  if (!result) {
    ml_loge(
      "Failed to read data from the output buffer for kernel_q6_k_sgemv_ptr");

    return;
  }
}

void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda, bool out_svm) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_sgemv_ptr;

  if (TransA) {
    kernel_sgemv_ptr = blas_cc->registerClKernel(sgemv_kernel, "sgemv_cl");
  } else {
    kernel_sgemv_ptr =
      blas_cc->registerClKernel(sgemv_no_trans_kernel, "sgemv_cl_noTrans");
  }

  if (!kernel_sgemv_ptr) {
    return;
  }

  sgemv_cl_internal<float>(kernel_sgemv_ptr, matAdata, vecXdata, vecYdata, dim1,
                           dim2, lda, out_svm);
}

float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_dot_ptr =
    blas_cc->registerClKernel(dot_kernel, "dot_cl");
  if (!kernel_dot_ptr) {
    return {};
  }

  return dot_cl_internal<float>(kernel_dot_ptr, vecAdata, vecXdata, dim1);
}

void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc,
              bool out_svm) {
  std::string kernel_func_;
  std::string sgemm_cl_kernel_;

  if (!TransA && !TransB) {
    kernel_func_ = "sgemm_cl_noTrans";
    sgemm_cl_kernel_ = sgemm_no_trans_kernel;
  } else if (TransA && !TransB) {
    kernel_func_ = "sgemm_cl_transA";
    sgemm_cl_kernel_ = sgemm_trans_a_kernel;
  } else if (!TransA && TransB) {
    kernel_func_ = "sgemm_cl_transB";
    sgemm_cl_kernel_ = sgemm_trans_b_kernel;
  } else {
    kernel_func_ = "sgemm_cl_transAB";
    sgemm_cl_kernel_ = sgemm_trans_ab_kernel;
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_sgemm_ptr =
    blas_cc->registerClKernel(sgemm_cl_kernel_, kernel_func_);
  if (!kernel_sgemm_ptr) {
    return;
  }

  sgemm_cl_internal<float>(kernel_sgemm_ptr, TransA, TransB, A, B, C, M, N, K,
                           lda, ldb, ldc, out_svm);
}

void addition_cl(const float *input, float *res, unsigned int size_input,
                 unsigned int size_res) {
  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_addition_ptr =
    blas_cc->registerClKernel(addition_kernel, "addition_cl");
  if (!kernel_addition_ptr) {
    return;
  }

  addition_cl_internal<float>(kernel_addition_ptr, input, res, size_input,
                              size_res);
}

void rmsnorm_cl(const float *input, const float *gamma, float *result,
                const float epsilon, unsigned int height, unsigned int width,
                bool use_svm) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_rmsnorm_ptr =
    blas_cc->registerClKernel(rmsnorm_kernel, "rmsnorm_cl");
  if (!kernel_rmsnorm_ptr) {
    return;
  }

  rmsnorm_cl_internal<float>(kernel_rmsnorm_ptr, input, gamma, result, epsilon,
                             height, width, use_svm);
}

void sscal_cl(float *X, const unsigned int N, const float alpha) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr =
    blas_cc->registerClKernel(sscal_kernel, "sscal_cl");

  if (!kernel_ptr) {
    return;
  }

  sscal_cl_internal<float>(kernel_ptr, X, N, alpha);
}

void transpose_cl_axis(const float *in, float *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_transpose_ptr;
  switch (axis) {
  case 0:
    kernel_transpose_ptr =
      blas_cc->registerClKernel(transpose_axis_0_kernel, "transpose_cl_axis0");
    break;
  case 1:
    kernel_transpose_ptr =
      blas_cc->registerClKernel(transpose_axis_1_kernel, "transpose_cl_axis1");
    break;
  case 2:
    kernel_transpose_ptr =
      blas_cc->registerClKernel(transpose_axis_2_kernel, "transpose_cl_axis2");
    break;
  default:
    throw std::invalid_argument("failed to register CL kernel");
    break;
  }
  if (!kernel_transpose_ptr) {
    return;
  }

  transpose_cl_axis_internal<float>(kernel_transpose_ptr, in, res,
                                    input_batch_size, input_channels,
                                    input_height, input_width, axis);
}

void flatten_block_q4_0_cl(const void *src, void *dst_q, void *dst_d,
                           unsigned int num_blocks) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    convert_block_q4_0_kernel, "kernel_convert_block_q4_0_noshuffle");
  if (!kernel_ptr) {
    ml_loge("Failed to register kernel_ptr for flatten_block_q4_0_cl");
    return;
  }

  int argIdx = 0;

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, src);
  if (!result) {
    ml_loge("Failed to set kernel argument 0 for flatten_block_q4_0_cl");
    return;
  }

  result =
    kernel_ptr->SetKernelSVMArguments(argIdx++, clbuffInstance.getSVMQuant());
  if (!result) {
    ml_loge("Failed to set kernel argument 1 for flatten_block_q4_0_cl");
    return;
  }

  result =
    kernel_ptr->SetKernelSVMArguments(argIdx++, clbuffInstance.getSVMScale());
  if (!result) {
    ml_loge("Failed to set kernel argument 2 for flatten_block_q4_0_cl");
    return;
  }

  const int work_groups_count[3] = {(int)num_blocks, 1, 1};
  const int work_group_size[3] = {64, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel for flatten_block_q4_0_cl");
    return;
  }
}

void restore_block_q4_0_cl(const void *src_q, const void *src_d, void *dst,
                           unsigned int num_blocks) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    convert_block_q4_0_kernel, "kernel_restore_block_q4_0");
  if (!kernel_ptr) {
    ml_loge("Failed to register kernel_ptr for restore_block_q4_0_cl");
    return;
  }

  int argIdx = 0;

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, src_q);
  if (!result) {
    ml_loge("Failed to set kernel argument 0 for restore_block_q4_0_cl");
    return;
  }

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, src_d);
  if (!result) {
    ml_loge("Failed to set kernel argument 1 for restore_block_q4_0_cl");
    return;
  }

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, dst);
  if (!result) {
    ml_loge("Failed to set kernel argument 2 for restore_block_q4_0_cl");
    return;
  }

  const int work_groups_count[3] = {(int)num_blocks, 1, 1};
  const int work_group_size[3] = {1, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel for restore_block_q4_0_cl");
    return;
  }
}

void transpose_32_16(float *data, int M, int K) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    transpose_32bit_16bit_kernel, "kernel_transpose_32_16");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for kernel_transpose_32_16");
    return;
  }

  int extra_elements = M % 8;
  int padding = 0;
  if (extra_elements > 0) {
    padding = 8 - extra_elements;
  }

  int width = K / 4;
  int height = M / 4;
  if (height == 0) {
    height = 1;
  }
  int padded_height = (M + padding) / 4;

  int arg = 0;
  bool result = false;

  result = kernel_ptr->SetKernelSVMArguments(arg++, data);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 0 for kernel_transpose_32_16");

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMInput());

  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 1 for kernel_transpose_32_16");

  result = kernel_ptr->SetKernelArguments(arg++, &height, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 for kernel_transpose_32_16");

  result = kernel_ptr->SetKernelArguments(arg++, &width, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 for kernel_transpose_32_16");

  result = kernel_ptr->SetKernelArguments(arg++, &padded_height, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 4 for kernel_transpose_32_16");

  const int work_groups_count[3] = {width, padded_height, 1};
  const int work_group_size[3] = {1, 16, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel for kernel_transpose_32_16");
    return;
  }
}

/** @todo Enable transpose_16 with proper fix.
void transpose_16(void *input, void *output, int width, int height,
                  int size_bytes, bool isQuant) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  ClContext::SharedPtrClKernel kernel_ptr =
    blas_cc->registerClKernel(transpose_16bit_kernel,
    "kernel_transpose_16");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for kernel_transpose_16");
    return;
  }

  int arg = 0;
  bool result = false;

  if (isQuant) {
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuantT());
  } else {
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScaleT());
  }

  result = kernel_ptr->SetKernelArguments(arg++, &height, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 for kernel_transpose_16");

  result = kernel_ptr->SetKernelArguments(arg++, &width, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 for kernel_transpose_16");

  const int work_groups_count[3] = {width, height, 1};
  const int work_group_size[3] = {4, 16, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel for kernel_transpose_16");
    return;
  }
}
*/

// =============================================================================
// v8c (paper 8/4/4) host wrappers — channel-wise QINT4 weight + int8 activation
// dot_4x8packed_su_int with bias-subtraction trick (validated near the dp4a
// device peak on Adreno).
// All inputs are plain cl_mem (image2d_from_buffer or buffer), no SVM.
// =============================================================================

/**
 * @brief Device caps for v8c dispatch (paper §3.4 device specialization),
 * queried once. Used to cap the LWS work-group product to the device's max
 * — so the tuned 4×16 sweet spot is honored on Adreno 830 (max 1024) yet
 * auto-reduced on a device with a smaller max work-group instead of
 * silently dropping to NULL.
 */
static const tv::DeviceImageCaps &v8c_device_caps() {
  static const tv::DeviceImageCaps caps = []() {
    auto *cc =
      static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
    cl_device_id dev = cc ? cc->context_inst_.GetDeviceId() : nullptr;
    return tv::queryDeviceImageCaps(dev);
  }();
  return caps;
}

/**
 * @brief Shared v8c LWS policy: preferred 4×16 (env NNTR_V8C_LWS overrides),
 * capped to the device max work-group size and required to divide gws.
 * @return whether a valid LWS was chosen; out is filled when true.
 */
static bool v8c_pick_lws(size_t gws_x, size_t gws_y,
                         std::array<size_t, 2> &out) {
  static const std::array<size_t, 2> pref = []() {
    const char *e = std::getenv("NNTR_V8C_LWS");
    size_t lx = 4, ly = 16; // swept sweet spot @ M=1024 (WG=64), 8.9× vs NULL.
    if (e) {
      long a = 0, b = 0;
      if (std::sscanf(e, "%ld,%ld", &a, &b) == 2 && a > 0 && b > 0) {
        lx = (size_t)a;
        ly = (size_t)b;
      }
    }
    return std::array<size_t, 2>{lx, ly};
  }();
  size_t ox = 0, oy = 0;
  bool ok = tv::select2dLws(gws_x, gws_y, pref[0], pref[1], v8c_device_caps(),
                            &ox, &oy);
  out = {ox, oy};
  return ok;
}

/**
 * @brief Device-specialization gate (paper §3.4). Selects the BUFFER-LOAD
 * v8c GEMM kernels (no sampled-image reads) for runtimes that advertise
 * cl_khr_image2d_from_buffer but cannot compile integer-coordinate
 * read_imageui (e.g. Intel NEO); the image path is the Adreno default. Now
 * caps-derived: NNTR_V8C_BUF still overrides, but with the flag unset the
 * choice comes from DeviceCaps::image_v8c (set from vendor_id at ClContext
 * init), so the buffer path no longer needs the mandatory flag on Intel.
 * Queried once per process.
 */
bool v8c_use_buffer_path() {
  static const bool use_buf = []() {
    const char *e = std::getenv("NNTR_V8C_BUF");
    if (e)
      return std::atoi(e) != 0; // explicit override (set wins)
    return !ClContext::Global().caps().image_v8c; // Intel ⇒ buffer
  }();
  return use_buf;
}

// Compile options for the buffer-load v8c program. -DV8C_BUFFER_ONLY excludes
// the image-sampling kernel bodies; -cl-std=CL3.0 exposes the core OpenCL C
// 3.0 dot_4x8packed_* builtins (Intel NEO does not declare them under the
// default CL1.2 std, and the legacy cl_khr_integer_dot_product #pragma is
// ignored by its front-end). All v8c registration sites on the buffer path
// must pass the IDENTICAL string so the same cached program is reused.
static const char *kV8cBufCompileOpts = "-DV8C_BUFFER_ONLY -cl-std=CL3.0";

/**
 * @brief [Adreno pitch fix] int4-weight row stride in BYTES. On the ADRENO
 * image path (caps.image_v8c, i.e. non-Intel) the weight backing rows are
 * padded up to a 64-byte multiple so image2d-from-buffer creation
 * satisfies CL_DEVICE_IMAGE_PITCH_ALIGNMENT (an unaligned K/2, e.g.
 * a K=192 projection -> 96 B, otherwise fails clCreateImage and
 * mis-routes the FC to the lm-head GEMV). Intel NEO (buffer path) has no
 * image and keeps the tight K/2 stride UNCHANGED -- padding it breaks the
 * 2D-block weight reads of the matrix-engine kernel stacked on this path.
 * The kernel K-loop still uses K/32 texels; only the ROW stride grows, and
 * the padding bytes are zero. Keep this in sync with
 * make_v8c_weight_backing_from_qs4cx and the image row_pitch in
 * blas_kernel_interface.cpp.
 */
static inline size_t v8c_wrow_bytes(unsigned int K) {
  return ClContext::Global().caps().image_v8c ? (((size_t)K / 2 + 63) / 64) * 64
                                              : (size_t)K / 2;
}
/** @brief v8c weight row stride in 16-byte texels (see v8c_wrow_bytes). */
static inline unsigned int v8c_wrow_texels(unsigned int K) {
  return (unsigned int)(v8c_wrow_bytes(K) / 16);
}

void gemm_int8_v8c_cl(cl_mem act_image, cl_mem weight_image, cl_mem scale_act,
                      cl_mem scale_wgt, cl_mem row_sum_act, cl_mem zp_act,
                      cl_mem row_sum_w_int4, cl_mem output_fp16, unsigned int M,
                      unsigned int N, unsigned int K, unsigned int M_valid) {
  if (M_valid == 0)
    M_valid = M; // legacy: store every (padded) row
  // [M=2..4 row-gap fix] The m1/coop GEMV kernels compute ONLY row 0. That is
  // correct for the M=1 decode (M_pad=4, M_valid=1) they were built for, but
  // a REAL 2-4-row call must compute every valid row: a short (e.g. 5-token)
  // prompt prefills at M=4, and the old "M_pad <= 4 means the real input had 1
  // valid row" assumption (below) silently left rows 1..M-1 of EVERY FC output
  // as stale garbage (KV cache poisoned -> deterministic fluent-but-off-topic
  // text; longer prompts prefill at M>4 and never hit this).
  // Route by the REAL row count (M_valid): single-row -> GEMV/m1 (fast decode
  // path unchanged); multi-row -> the TM=4 tiled kernel, which takes the
  // M_valid store guard. (When !direct_out the caller passes M_valid=M_pad;
  // the TM=4 kernel then also computes the zero-padded rows into scratch --
  // harmless, consumers read only the real rows.)
  const bool use_m1 = (M_valid == 1) && M <= 4;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  // For the M=1 (M_pad=4) decode case the default TM=4 kernel burns ~4×
  // the work needed (3 zero-padded rows). Dispatch a TM=1 variant
  // instead. (HISTORICAL BUG, fixed above: the caller passes M_pad here, and
  // this used to assume M_pad <= 4 implies "1 valid row" -- false for a
  // 4-row prefill. Routing now keys on M_valid, not M_pad.)
  // Buffer path (NNTR_V8C_BUF=1): act_image/weight_image carry the raw cl_mem
  // buffers; widths derived from K (int4 weight texel = 32 K, act texel = 16
  // K).
  const bool use_buf = v8c_use_buffer_path();
  // NNTR_FC_BUF=1 (probe): route ONLY the FC GEMMs through the buffer-load
  // kernels while the rest of the pipeline (attention images, copts) stays
  // on the image path. The image-capable program always contains the buffer
  // kernels too, so the compile opts stay "" — this also sidesteps the
  // on-disk .cl.bin cache collision (its filename ignores copts). Callers
  // must pass buffer handles for act/weight when setting this.
  static const bool fc_buf_probe = []() {
    const char *e = getenv("NNTR_FC_BUF");
    return e && atoi(e) != 0;
  }();
  const bool buf_kernel = use_buf || fc_buf_probe;
  const char *kname =
    buf_kernel
      ? (use_m1 ? "v8c_gemm_int8_int4_m1_buf" : "v8c_gemm_int8_int4_buf")
      : (use_m1 ? "v8c_gemm_int8_int4_m1" : "v8c_gemm_int8_int4");
  // Buffer path compiles the program with -DV8C_BUFFER_ONLY so the
  // image-sampling kernel bodies are excluded (Intel NEO can't compile them).
  std::string copts = use_buf ? kV8cBufCompileOpts : "";
  // NNTR_V8C_KCLOCK: in-kernel cl_khr_kernel_clock measurement build (off by
  // default). "1" = time the full K-loop; "2" = also drop the dp4a (fetch-only
  // floor) to isolate fetch-stall from compute. Measurement-only; the printf is
  // emitted by one work-item per dispatch.
  // NNTR_V8C_KCLOCK loop-decomposition modes (in-kernel clock_read_device):
  //   1 full | 2 no-dp4a (fetch+unpack) | 3 no-weight-fetch | 4 no-act-fetch |
  //   5 compute-only (no fetch). Diff of modes isolates weight- vs act-fetch
  //   vs pure compute. Synthetic runtime values suppress a stream w/o DCE.
  // Env-gated diagnostic/experimental compile flags — process-constant, so read
  // ONCE (static), not per dispatch. NNTR_V8C_KCLOCK {1..5} (in-kernel clock
  // decomposition), NNTR_V8C_PREFETCH=1 (1-ahead weight prefetch),
  // NNTR_V8C_MFAST =1 (M-fast dispatch order). All bit-identical / default-off.
  static const std::string v8c_env_copts = []() {
    std::string s;
    if (const char *kc = getenv("NNTR_V8C_KCLOCK")) {
      const char m = kc[0];
      if (m >= '1' && m <= '5')
        s += " -DV8C_KCLOCK";
      if (m == '2')
        s += " -DV8C_KCLOCK_NOCOMPUTE";
      if (m == '3' || m == '5')
        s += " -DV8C_KCLOCK_NOWFETCH";
      if (m == '4' || m == '5')
        s += " -DV8C_KCLOCK_NOAFETCH";
    }
    if (const char *pf = getenv("NNTR_V8C_PREFETCH"))
      if (pf[0] == '1')
        s += " -DV8C_PREFETCH";
    if (const char *mf = getenv("NNTR_V8C_MFAST"))
      if (mf[0] == '1')
        s += " -DV8C_MFAST";
    return s;
  }();
  static const bool v8c_mfast = []() {
    const char *mf = getenv("NNTR_V8C_MFAST");
    return mf && mf[0] == '1';
  }();
  copts += v8c_env_copts;
  // ML Drift reaudit (decode, 2026-06-12): cooperative M=1 GEMV. The m1
  // kernels' total parallelism is only N/8 work-items (at N=2304 that is
  // ~4.5 waves device-wide) = latency-bound at 12-22 GB/s effective, while
  // the decode FC stream is ~977 MB/token — the whole decode budget. The
  // coop kernel splits K 8-way per column under a 64-WI workgroup
  // (parallelism x64, LDS tree reduce) and reads the images' BACKING
  // buffers via uint4 vloads (clGetImageInfo(CL_IMAGE_BUFFER) — the
  // images are image2d-from-buffer views, so no extra copies exist). Output is
  // BIT-IDENTICAL to the m1 kernels: int32 dp4a accumulation is
  // order-independent and the per-column float epilogue is unchanged.
  // NNTR_GEMV_COOP=0 restores the m1 dispatch.
  static const bool gemv_coop = []() {
    const char *e = getenv("NNTR_GEMV_COOP");
    return !e || atoi(e) != 0;
  }();
  // K cap: the coop kernel stages the act row in a 768-uint4 LDS array
  // (12 KB). 12288 admits Gemma4's double-wide-MLP FFN-down (K=12288), which
  // previously exceeded the old 10240 cap and fell to the ~15x slower m1 GEMM
  // at decode (measured 2389us vs 158us/call).
  // Coop GEMV also serves the Intel buffer path (NNTR_V8C_BUF). The kernel
  // (v8c_gemv_int8_int4_coop) lives outside the V8C_BUFFER_ONLY guard and reads
  // plain uint4 buffers with the SAME byte layout the m1_buf kernel uses
  // (W_wgt = K/32 weight row stride; act row 0 = K/16 uint4). On the buffer
  // path act_image/weight_image ARE the raw cl_mem buffers, so feed them
  // directly; Adreno (image path) still extracts the image2d-from-buffer
  // backing via clGetImageInfo. This lifts Intel decode FC off the
  // latency-bound m1_buf GEMV (only N/8 work-items) onto the 64-WI K-split
  // coop kernel.
  if (use_m1 && gemv_coop && (N % 8) == 0 && (K % 32) == 0 && K <= 12288) {
    cl_mem wbuf = nullptr, abuf = nullptr;
    if (buf_kernel) {
      wbuf = weight_image; // raw cl_mem on the buffer path
      abuf = act_image;
    } else {
      opencl::clGetImageInfo(weight_image, CL_IMAGE_BUFFER, sizeof(cl_mem),
                             &wbuf, nullptr);
      opencl::clGetImageInfo(act_image, CL_IMAGE_BUFFER, sizeof(cl_mem), &abuf,
                             nullptr);
    }
    if (wbuf != nullptr && abuf != nullptr) {
      ClContext::SharedPtrClKernel ck = blas_cc->registerClKernel(
        int8_int4_gemm_v8c_kernel, "v8c_gemv_int8_int4_coop", copts);
      if (ck) {
        int Ni = (int)N, Ki = (int)K, Ww = (int)v8c_wrow_texels(K);
        int a = 0;
        const bool ok =
          ck->SetKernelArguments(a++, &abuf, sizeof(cl_mem)) &&
          ck->SetKernelArguments(a++, &wbuf, sizeof(cl_mem)) &&
          ck->SetKernelArguments(a++, &scale_act, sizeof(cl_mem)) &&
          ck->SetKernelArguments(a++, &scale_wgt, sizeof(cl_mem)) &&
          ck->SetKernelArguments(a++, &row_sum_act, sizeof(cl_mem)) &&
          ck->SetKernelArguments(a++, &zp_act, sizeof(cl_mem)) &&
          ck->SetKernelArguments(a++, &row_sum_w_int4, sizeof(cl_mem)) &&
          ck->SetKernelArguments(a++, &output_fp16, sizeof(cl_mem)) &&
          ck->SetKernelArguments(a++, &Ni, sizeof(int)) &&
          ck->SetKernelArguments(a++, &Ki, sizeof(int)) &&
          ck->SetKernelArguments(a++, &Ww, sizeof(int));
        if (ok) {
          std::array<size_t, 3> gws = {(size_t)(N / 8) * 64, 1, 1};
          std::array<size_t, 3> lws = {64, 1, 1};
          blas_cc->command_queue_inst_.enqueueKernel(
            ck->GetKernel(), 1, gws.data(), lws.data(), 0, nullptr, nullptr);
          return;
        }
      }
    }
    // Fall through to the m1 dispatch on any setup failure.
  }
  // --- NNTR_FC_XMX: Intel Xe2 (Lunar Lake) XMX/DPAS prefill GEMM ------------
  // Drop-in for the M>4 buffer-path v8c GEMM using int8 DPAS (systolic array).
  // Raw-nibble + i8_u8 MMA -> acc_raw, then the SAME bias-correction + scale
  // epilogue as the dp4a kernel (byte-identical output). Requires the buffer
  // path (raw cl_mem) + the Intel matrix-multiply / 2d_block_io extensions; on
  // a non-XMX device registerClKernel fails -> kx null -> fall through to dp4a.
  // M is padded up to MT*8*SG_M (=32) by the grid; M_valid guards the stores
  // and the A/weight 2D-block reads clamp OOB rows to 0 (surface height = M).
  // FIX 1 (XMX/DPAS capability gate): NNTR_FC_XMX=0 force-disables (explicit
  // override always wins). Otherwise (=1 or unset) XMX is only actually used
  // when the device has a real matrix engine (caps().dpas, i.e.
  // cl_intel_subgroup_matrix_multiply_accumulate) — cl_intel_subgroups alone
  // (the old gate) is advertised by every Intel GPU since Gen9, including
  // non-DPAS Xe-LPG "Arc" iGPUs, and registering gemm_xmx_i4 there let IGC
  // silently emulate the DPAS builtin in software (~4.9 TPS observed).
  // NNTR_FC_XMX_FORCE=1 (value-checked, not presence-checked) bypasses the
  // capability requirement for debugging/benchmarking a non-DPAS device
  // against the DPAS kernel.
  static const bool xmx_fc = []() {
    const char *e = getenv("NNTR_FC_XMX");
    const bool requested = e ? (atoi(e) != 0) : true; // unset = default-on
    if (e && atoi(e) == 0)
      return false; // NNTR_FC_XMX=0 force-disables regardless of caps
    const char *force = getenv("NNTR_FC_XMX_FORCE");
    if (force && std::string(force) == "1") {
      // Debug escape hatch: skip the capability check. Say so loudly -- on a
      // device with no matrix engine this asks the driver to emulate the DPAS
      // builtin, which is far slower than the fallback it replaces.
      ml_logw("[XMX] NNTR_FC_XMX_FORCE=1: dispatching the DPAS kernel without "
              "checking for a matrix engine. Benchmarking only.");
      return requested;
    }
    return requested && ClContext::Global().caps().dpas;
  }();
  // One-shot warning (stderr + ml_logw): XMX was requested/defaulted but this
  // device has no DPAS matrix engine, so the honest dp4a fallback runs
  // instead (~1.8x slower than XMX, not a correctness issue).
  static bool xmx_no_dpas_warned = false;
  if (!xmx_no_dpas_warned) {
    xmx_no_dpas_warned = true;
    const char *e = getenv("NNTR_FC_XMX");
    const bool requested = e ? (atoi(e) != 0) : true;
    const char *force = getenv("NNTR_FC_XMX_FORCE");
    const bool forced = force && std::string(force) == "1";
    if (requested && !forced && !ClContext::Global().caps().dpas) {
      ml_logw("[XMX] Intel GPU \"%s\" lacks the XMX/DPAS matrix engine "
              "(no cl_intel_subgroup_matrix_multiply_accumulate) — using the "
              "dp4a GEMM fallback (~1.8x slower than XMX, not a correctness "
              "issue). If an NVIDIA GPU is present, backend=cuda will be "
              "faster. Set NNTR_FC_XMX_FORCE=1 to force XMX for debugging.",
              ClContext::Global().caps().device_name.c_str());
    }
  }
  // One-shot diagnostic (stderr): why XMX/DPAS is or is not selected. On
  // Windows the Intel driver may fail to compile gemm_xmx_i4 (matrix-MAD /
  // 2d_block_io / 256-GRF), silently falling back to the ~40%-slower dp4a path.
  static bool xmx_gate_logged = false;
  if (!xmx_gate_logged) {
    xmx_gate_logged = true;
    const bool gate = xmx_fc && M > 4 && buf_kernel && (N % 64) == 0 &&
                      (K % 64) == 0 && K >= 128 &&
                      v8c_wrow_bytes(K) == (size_t)K / 2;
    // Quiet by default (SDK surface): always report the surprising case
    // (XMX requested but gated OFF = silent perf loss), the rest only
    // under NNTR_GPU_VERBOSE.
    if ((xmx_fc && !gate) || std::getenv("NNTR_GPU_VERBOSE"))
      fprintf(stderr,
              "[XMX] xmx_fc=%d dpas=%d buf_kernel=%d M=%u N=%u K=%u -> "
              "gate=%d%s\n",
              (int)xmx_fc, (int)ClContext::Global().caps().dpas,
              (int)buf_kernel, M, N, K, (int)gate,
              gate ? "" : " (XMX skipped -> dp4a)");
  }
  // K >= 128: the weight read is issued as a 2D block read with
  // base_width = base_pitch = K/2 bytes, and the block-io extension leaves
  // behaviour undefined below a 64-byte width. K % 64 == 0 alone admits
  // K == 64 -> 32 bytes. At K >= 128, K/2 >= 64 clears the width minimum and
  // K % 64 == 0 already makes K/2 a multiple of 32, clearing the pitch rule.
  // Narrower shapes fall through to dp4a: correct, just slower.
  // The kernel hardcodes the weight surface's width and pitch as K/2 rather
  // than reading the stride argument it is passed, so it is only correct while
  // the row stride IS K/2. That holds on the buffer path today, but buf_kernel
  // can also be set by the probe override while the row stride is the padded
  // image-path one, so the invariant is checked here instead of assumed.
  if (xmx_fc && M > 4 && buf_kernel && (N % 64) == 0 && (K % 64) == 0 &&
      K >= 128 && v8c_wrow_bytes(K) == (size_t)K / 2) {
    // ONE fixed tile, deliberately -- not a per-shape heuristic. A standalone
    // sweep does favour a different (NT, SG_M) per projection shape (SG_M=8
    // for the deep down-projections, SG_M=1 for the narrow models), but those
    // wins do not survive the whole model: SG_M=8 collapses occupancy in-model
    // by up to a third. NT=2 with SG_M=4 measured uniformly best in-model
    // across every validated LLM (prefill +3% to +50% against the microbench's
    // NT=4/SG_M=1), so it is the robust default and the only shape-dependent
    // logic below is the repair of an override N cannot divide.
    // Each (MT, NT, SG_M) is a distinct -D option string, so registerClKernel
    // caches a separate compiled program per tile (key = name + options).
    int xmx_mt = 4, xmx_nt = 2, xmx_sgm = 4;
    // Validate the overrides. These are documented tuning knobs, so a typo is
    // a realistic input, and an unvalidated one is not a bad tile but a
    // crash: 0 (which is also what a non-numeric string parses to) reaches
    // `N % (nt * 16)` and `M % (MT * 8 * SG_M)` as a division by zero, and a
    // negative value reaches the local size as a huge size_t. Reject anything
    // outside the range the kernel supports and keep the default.
    auto tile_override = [](const char *var, int lo, int hi, int dflt) {
      const char *e = getenv(var);
      if (!e || !*e)
        return dflt;
      char *end = nullptr;
      const long v = std::strtol(e, &end, 10);
      if (end == e || *end != '\0' || v < lo || v > hi) {
        ml_logw("[XMX] ignoring %s=\"%s\": expected an integer in [%d, %d]",
                var, e, lo, hi);
        return dflt;
      }
      return (int)v;
    };
    xmx_nt = tile_override("NNTR_XMX_NT", 1, 8, xmx_nt);
    xmx_sgm = tile_override("NNTR_XMX_SGM", 1, 16, xmx_sgm);
    if (N % ((unsigned)xmx_nt * 16) != 0)
      xmx_nt = 4; // N%64==0 guaranteed by the dispatch gate
    char xmx_co[176];
    snprintf(
      xmx_co, sizeof(xmx_co),
      "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread -DMT=%d -DNT=%d -DSG_M=%d",
      xmx_mt, xmx_nt, xmx_sgm);
    ClContext::SharedPtrClKernel kx = blas_cc->registerClKernel(
      int8_int8_gemm_xmx_kernel, "gemm_xmx_i4", std::string(xmx_co));
    static bool xmx_reg_logged = false;
    if (!xmx_reg_logged) {
      xmx_reg_logged = true;
      // FAILED always (silent perf loss otherwise); OK only when verbose.
      if (!kx || std::getenv("NNTR_GPU_VERBOSE"))
        fprintf(stderr, "[XMX] gemm_xmx_i4 registerClKernel -> %s\n",
                kx ? "OK (DPAS/XMX engaged)"
                   : "FAILED -> silent fallback to dp4a (slower)");
    }
    if (kx) {
      int Mi = (int)M, Ni = (int)N, Ki = (int)K, Wa = (int)(K / 16),
          Ww = (int)v8c_wrow_texels(K), Mv = (int)M_valid;
      int a = 0;
      const bool ok =
        kx->SetKernelArguments(a++, &act_image, sizeof(cl_mem)) &&
        kx->SetKernelArguments(a++, &weight_image, sizeof(cl_mem)) &&
        kx->SetKernelArguments(a++, &scale_act, sizeof(cl_mem)) &&
        kx->SetKernelArguments(a++, &scale_wgt, sizeof(cl_mem)) &&
        kx->SetKernelArguments(a++, &row_sum_act, sizeof(cl_mem)) &&
        kx->SetKernelArguments(a++, &zp_act, sizeof(cl_mem)) &&
        kx->SetKernelArguments(a++, &row_sum_w_int4, sizeof(cl_mem)) &&
        kx->SetKernelArguments(a++, &output_fp16, sizeof(cl_mem)) &&
        kx->SetKernelArguments(a++, &Mi, sizeof(int)) &&
        kx->SetKernelArguments(a++, &Ni, sizeof(int)) &&
        kx->SetKernelArguments(a++, &Ki, sizeof(int)) &&
        kx->SetKernelArguments(a++, &Wa, sizeof(int)) &&
        kx->SetKernelArguments(a++, &Ww, sizeof(int)) &&
        kx->SetKernelArguments(a++, &Mv, sizeof(int));
      if (ok) {
        const size_t mpwg = (size_t)xmx_mt * 8 * xmx_sgm; // rows per workgroup
        const size_t Mpad = (((size_t)M + mpwg - 1) / mpwg) * mpwg;
        const size_t npsz = (size_t)xmx_nt * 16; // cols per subgroup
        std::array<size_t, 3> gws = {(size_t)(N / npsz) * 16,
                                     Mpad / ((size_t)xmx_mt * 8), 1};
        std::array<size_t, 3> lws = {16, (size_t)xmx_sgm, 1};
        blas_cc->command_queue_inst_.enqueueKernel(
          kx->GetKernel(), 2, gws.data(), lws.data(), 0, nullptr, nullptr);
        return;
      }
    }
    // fall through to dp4a on any failure
  }
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(int8_int4_gemm_v8c_kernel, kname, copts);
  if (!kp)
    throw std::runtime_error(
      std::string("v8c_gemm: registerClKernel failed: ") + kname);

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &act_image, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 0 (act_image)");
  if (!kp->SetKernelArguments(arg++, &weight_image, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 1 (weight_image)");
  if (!kp->SetKernelArguments(arg++, &scale_act, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 2 (scale_act)");
  if (!kp->SetKernelArguments(arg++, &scale_wgt, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 3 (scale_wgt)");
  if (!kp->SetKernelArguments(arg++, &row_sum_act, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 4 (row_sum_act)");
  if (!kp->SetKernelArguments(arg++, &zp_act, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 5 (zp_act)");
  if (!kp->SetKernelArguments(arg++, &row_sum_w_int4, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 6 (row_sum_w_int4)");
  if (!kp->SetKernelArguments(arg++, &output_fp16, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 7 (output_fp16)");
  int Mi = (int)M, Ni = (int)N, Ki = (int)K;
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)))
    throw std::runtime_error("v8c gemm arg 8 (M)");
  if (!kp->SetKernelArguments(arg++, &Ni, sizeof(int)))
    throw std::runtime_error("v8c gemm arg 9 (N)");
  if (!kp->SetKernelArguments(arg++, &Ki, sizeof(int)))
    throw std::runtime_error("v8c gemm arg 10 (K)");
  if (buf_kernel) {
    int W_act = (int)(K / 16), W_wgt = (int)v8c_wrow_texels(K);
    if (!kp->SetKernelArguments(arg++, &W_act, sizeof(int)))
      throw std::runtime_error("v8c gemm arg 11 (W_act)");
    if (!kp->SetKernelArguments(arg++, &W_wgt, sizeof(int)))
      throw std::runtime_error("v8c gemm arg 12 (W_wgt)");
  }
  // TM=4 kernels take the trailing M_valid store guard; the m1 (single-row)
  // variants only ever write row 0 and keep their legacy signature.
  if (!use_m1) {
    int Mv = (int)M_valid;
    if (!kp->SetKernelArguments(arg++, &Mv, sizeof(int)))
      throw std::runtime_error("v8c gemm arg (M_valid)");
  }

  // V8C_TM=4, V8C_TN=8 (defaults in the .cl). global = (N/TN, M/TM); kernel
  // requires M%4=0, N%8=0, K%32=0. The historic 4×16 LWS is only valid when
  // gws[1] = M/TM is a multiple of 16, i.e. M is a multiple of 64. The
  // QINT4 caller pads M up to V8C_TM (=4) for the M=1 decode / M=18 prefill
  // cases, so M/TM can be as small as 1 or 5 and never satisfies that
  // constraint. Pass NULL lws so the runtime picks a compatible
  // workgroup shape — the kernel has no in-flight cross-thread sync, so
  // any LWS that divides gws works. (OpenCL 3.0 on Adreno 830 allows
  // non-uniform groups, but staying uniform with a runtime-chosen LWS is
  // the conservative win.)
  if (use_m1) {
    // M=1 (TM=1) GEMV-style dispatch: 1-D grid over output channels.
    constexpr size_t TN_M1 = 8;
    std::array<size_t, 3> gws = {(size_t)N / TN_M1, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                               nullptr, 0, nullptr, nullptr);
  } else {
    constexpr size_t TM = 4, TN = 8;
    // NNTR_V8C_MFAST: swap so M/TM is the fast-varying (dim0) axis; the kernel
    // (-DV8C_MFAST) reads m0 from gid0, n0 from gid1 to match. Weight-reuse
    // dispatch order. Default keeps {N/TN, M/TM}.
    std::array<size_t, 3> gws =
      v8c_mfast ? std::array<size_t, 3>{(size_t)M / TM, (size_t)N / TN, 1}
                : std::array<size_t, 3>{(size_t)N / TN, (size_t)M / TM, 1};
    // CL-event profiling showed the historic NULL lws (driver-chosen
    // workgroup) lands the GEMM at ~14% of dp4a peak in-forward, vs 87%
    // in the standalone microbench which used a tuned LWS. Pick a device-
    // specialized LWS (4×16 sweet spot, capped to the device max work-group
    // size and required to divide gws); fall back to NULL when none fits
    // (small-M prefill). NNTR_V8C_LWS="lx,ly" overrides.
    std::array<size_t, 2> picked{};
    const bool lws_ok = v8c_pick_lws(gws[0], gws[1], picked);
    std::array<size_t, 3> lws = {picked[0], picked[1], 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                               lws_ok ? lws.data() : nullptr, 0,
                                               nullptr, nullptr);
  }
}

void gemm_int8_v8c_v_ohwi_cl(cl_mem act_image, cl_mem weight_image,
                             cl_mem scale_act, cl_mem scale_wgt,
                             cl_mem row_sum_act, cl_mem zp_act,
                             cl_mem row_sum_w_int4, cl_mem v_ohwi,
                             unsigned int M_pad, unsigned int N, unsigned int K,
                             unsigned int head_dim, unsigned int S_max,
                             unsigned int position, unsigned int M_real) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (M_pad == 0 || N == 0 || K == 0 || head_dim == 0 || S_max == 0)
    throw std::runtime_error("gemm_int8_v8c_v_ohwi: zero dim");
  constexpr unsigned int V8C_TM = 4, V8C_TN = 8;
  if (M_pad % V8C_TM != 0 || N % V8C_TN != 0 || K % 32 != 0)
    throw std::runtime_error("gemm_int8_v8c_v_ohwi: M/N/K not aligned");
  if (head_dim % V8C_TN != 0)
    throw std::runtime_error("gemm_int8_v8c_v_ohwi: head_dim % TN != 0");
  if (N % head_dim != 0)
    throw std::runtime_error("gemm_int8_v8c_v_ohwi: N % head_dim != 0");
  if (M_real > M_pad)
    M_real = M_pad;

  const bool use_buf = v8c_use_buffer_path();
  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    int8_int4_gemm_v8c_kernel,
    use_buf ? "v8c_gemm_int8_int4_v_ohwi_buf" : "v8c_gemm_int8_int4_v_ohwi",
    use_buf ? kV8cBufCompileOpts : "");
  if (!kp)
    throw std::runtime_error("v8c_v_ohwi: registerClKernel failed");

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &act_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &weight_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_wgt, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &zp_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_w_int4, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &v_ohwi, sizeof(cl_mem)))
    throw std::runtime_error("v8c_v_ohwi: cl_mem arg failed");
  int Mi = (int)M_pad, Ni = (int)N, Ki = (int)K;
  int di = (int)head_dim, Si = (int)S_max, pi = (int)position;
  int Mr = (int)M_real;
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ni, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ki, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &di, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Si, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &pi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Mr, sizeof(int)))
    throw std::runtime_error("v8c_v_ohwi: int arg failed");
  if (use_buf) {
    int W_act = (int)(K / 16), W_wgt = (int)v8c_wrow_texels(K);
    if (!kp->SetKernelArguments(arg++, &W_act, sizeof(int)) ||
        !kp->SetKernelArguments(arg++, &W_wgt, sizeof(int)))
      throw std::runtime_error("v8c_v_ohwi: width arg failed");
  }
  std::array<size_t, 3> gws = {(size_t)N / V8C_TN, (size_t)M_pad / V8C_TM, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                             nullptr, 0, nullptr, nullptr);
}

static void quantize_act_v8c_cl_impl(cl_mem act_in, cl_mem out_int8,
                                     cl_mem out_scale, cl_mem out_zp,
                                     cl_mem out_row_sum, unsigned int M,
                                     unsigned int K, const char *kernel_name,
                                     bool parallel) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  // The act-quant kernels are image-free, but they live in the same program
  // string as the image GEMMs. On the buffer path we must build that program
  // with -DV8C_BUFFER_ONLY too, otherwise this (option-less) build compiles
  // the image kernels and fails on Intel NEO. Matches the GEMM wrappers so the
  // same cached program is reused.
  const std::string copts = v8c_use_buffer_path() ? kV8cBufCompileOpts : "";
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(int8_int4_gemm_v8c_kernel, kernel_name, copts);
  if (!kp)
    throw std::runtime_error(
      std::string("v8c quant: registerClKernel failed: ") + kernel_name);
  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &act_in, sizeof(cl_mem)))
    throw std::runtime_error("v8c quant arg 0");
  if (!kp->SetKernelArguments(arg++, &out_int8, sizeof(cl_mem)))
    throw std::runtime_error("v8c quant arg 1");
  if (!kp->SetKernelArguments(arg++, &out_scale, sizeof(cl_mem)))
    throw std::runtime_error("v8c quant arg 2");
  if (!kp->SetKernelArguments(arg++, &out_zp, sizeof(cl_mem)))
    throw std::runtime_error("v8c quant arg 3 (out_zp)");
  if (!kp->SetKernelArguments(arg++, &out_row_sum, sizeof(cl_mem)))
    throw std::runtime_error("v8c quant arg 4");
  int Mi = (int)M, Ki = (int)K;
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)))
    throw std::runtime_error("v8c quant arg 5");
  if (!kp->SetKernelArguments(arg++, &Ki, sizeof(int)))
    throw std::runtime_error("v8c quant arg 6");

  if (parallel) {
    // _par variant: one workgroup (LWS=64) per row. gws = M * 64.
    constexpr size_t LWS = 64;
    std::array<size_t, 3> gws = {(size_t)M * LWS, 1, 1};
    std::array<size_t, 3> lws = {LWS, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  } else {
    std::array<size_t, 3> gws = {(size_t)M, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                               nullptr, 0, nullptr, nullptr);
  }
}

void quantize_act_v8c_fp16_cl(cl_mem act_fp16, cl_mem out_int8,
                              cl_mem out_scale, cl_mem out_zp,
                              cl_mem out_row_sum, unsigned int M,
                              unsigned int K) {
  // The _par variant requires K % LWS == 0 (LWS=64). Qwen3 hidden dims
  // (1024/2048/3072/etc.) all satisfy this; smaller K paths fall back.
  const bool can_par = (K % 64 == 0);
  quantize_act_v8c_cl_impl(
    act_fp16, out_int8, out_scale, out_zp, out_row_sum, M, K,
    can_par ? "v8c_act_quant_f16_par" : "v8c_act_quant_f16", can_par);
}

void quantize_act_v8c_fp32_cl(cl_mem act_fp32, cl_mem out_int8,
                              cl_mem out_scale, cl_mem out_zp,
                              cl_mem out_row_sum, unsigned int M,
                              unsigned int K) {
  const bool can_par = (K % 64 == 0);
  quantize_act_v8c_cl_impl(
    act_fp32, out_int8, out_scale, out_zp, out_row_sum, M, K,
    can_par ? "v8c_act_quant_f32_par" : "v8c_act_quant_f32", can_par);
}

} // namespace nntrainer

// Use Int4Utils for osv32 row dequant; keep include inside the helper file
// (TU-local) to avoid leaking the dependency through the public header.
#include "../int4_utils.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace nntrainer {

std::unique_ptr<tv::TensorBacking>
make_v8c_weight_backing(const uint8_t *osv32_packed,
                        const uint16_t *fp16_scales, unsigned int group_size,
                        unsigned int N, unsigned int K, cl_mem *out_scale_buf) {
  if (K % 32 != 0)
    throw std::invalid_argument(
      "make_v8c_weight_backing: K must be a multiple of 32");

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext(); // OpenCL context handle

  // 1) Walk osv32 row by row, dequantize → re-quantize per-channel (paper §4.2)
  //    Pack into row-major v8c layout: per row N, K/2 bytes; per K-block of 32
  //    one 16-byte texel; byte offset = (k_outer/32)*16 + c*4 + b within row.
  //    Byte content: low_nib = (q[k_outer+c*8+b] + 8) & 0xF,
  //                  high_nib = (q[k_outer+c*8+b+4] + 8) & 0xF.
  const size_t row_bytes = (size_t)K / 2;
  std::vector<uint8_t> packed((size_t)N * row_bytes);
  std::vector<float> per_channel_scale(N);
  std::vector<float> deq_row((size_t)K);

  // dequantizePackedRow takes non-const pointers; cast safely (it reads only).
  uint8_t *osv_nonconst = const_cast<uint8_t *>(osv32_packed);
  uint16_t *scales_nonconst = const_cast<uint16_t *>(fp16_scales);

  for (unsigned int n = 0; n < N; ++n) {
    // (a) Dequantize this row from osv32 + per-group scales → fp32 [K].
    Int4Utils::dequantizePackedRow(osv_nonconst, scales_nonconst,
                                   /** rows_count */ N, /** cols_count */ K,
                                   /** group_size */ group_size,
                                   /** row_index */ n, deq_row.data());
    // (b) Compute per-channel scale = max|.| / 7
    float amax = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      float a = std::fabs(deq_row[k]);
      if (a > amax)
        amax = a;
    }
    float s = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
    per_channel_scale[n] = s;
    float inv_s = 1.0f / s;
    // (c) Re-quantize to int4 in [-7,7], offset-encode, pack to v8c layout
    uint8_t *out_row = packed.data() + (size_t)n * row_bytes;
    for (unsigned int k_outer = 0; k_outer < K; k_outer += 32) {
      for (int c = 0; c < 4; ++c) {
        for (int b = 0; b < 4; ++b) {
          unsigned int kLo = k_outer + (unsigned int)(c * 8 + b);
          unsigned int kHi = kLo + 4;
          int qL = (int)std::lrint(deq_row[kLo] * inv_s);
          int qH = (int)std::lrint(deq_row[kHi] * inv_s);
          qL = qL < -7 ? -7 : (qL > 7 ? 7 : qL);
          qH = qH < -7 ? -7 : (qH > 7 ? 7 : qH);
          uint8_t lo = (uint8_t)((qL + 8) & 0xF);
          uint8_t hi = (uint8_t)((qH + 8) & 0xF);
          size_t off = (size_t)(k_outer / 32) * 16 + (size_t)(c * 4 + b);
          out_row[off] = (uint8_t)((hi << 4) | lo);
        }
      }
    }
  }

  // 2) Upload packed weight to a new cl_mem buffer, wrap in TensorBacking.
  cl_int err = CL_SUCCESS;
  cl_mem w_buf =
    opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           packed.size(), packed.data(), &err);
  if (err != CL_SUCCESS || !w_buf)
    throw std::runtime_error("make_v8c_weight_backing: clCreateBuffer (weight) "
                             "failed: " +
                             std::to_string(err));
  auto backing = std::make_unique<tv::TensorBacking>(
    ctx, w_buf, tv::Encoding::INT4_OFFSET, tv::Layout::ROW_MAJOR, packed.size(),
    /** owned */ true);

  // 3) Upload per-channel scale buffer.
  cl_mem sb =
    opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * N, per_channel_scale.data(), &err);
  if (err != CL_SUCCESS || !sb)
    throw std::runtime_error("make_v8c_weight_backing: clCreateBuffer (scale) "
                             "failed: " +
                             std::to_string(err));
  *out_scale_buf = sb;
  return backing;
}

// Build the SAME v8c GEMM backing from an upstream QS4CX
// plain weight instead of our KAI Section-A. QS4CX on-disk = plain row-major
// nibbles (N rows of (K+1)/2 bytes; even k = low nibble, odd k = high nibble;
// stored uint4 = int4 + 8, NO XOR 0x88) + per-output-channel fp32 scale
// (range/15 dequant multiplier). Because the v8c GEMM only ever consumes the
// prebuilt backing + scale_buf + row_sum_buf, swapping the load-time decode
// source (plain index-by-K vs Section-A super-row inverse) leaves the GEMM, the
// scale buffer, and the row-sum byte-identical => zero GEMM/perf change. The
// only semantic differences vs the KAI path are: no XOR undo, and the scale is
// already fp32 (no fp16->fp32 promotion).

// NNTR_V8C_HOSTPTR: allocate the v8c weight backing as CL_MEM_ALLOC_HOST_PTR
// (host-visible, GPU reads it in place) instead of a device clCreateBuffer.
// MEASURED WORSE and therefore default OFF: on Xe3/Windows it does not remove a
// copy, it ADDS one -- peak working set 3797 -> 5231 MB (+1434, i.e. a second
// full weight image) with prefill 1588 -> 1461 TPS. So a plain device
// clCreateBuffer is already the single copy; keep it. Kept as an A/B lever (and
// a warning) for the next driver/platform where the zero-copy assumption might
// actually hold.
static bool v8c_hostptr_on() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_V8C_HOSTPTR");
    return e != nullptr && e[0] != '0';
  }();
  return on;
}

// Submit-and-go weight uploads. The
// blocking per-chunk write made every load worker pay a submit->wait round
// trip per chunk even though the in-order queue already sequences all later
// GEMMs after the writes — correctness needs NO barrier, only the staging
// buffer's lifetime. Non-blocking writes hand their staging to this
// registry; entries are freed once their event completes. Bounded: pushing
// past the cap first drains everything queued so far.
namespace {

/**
 * @brief One in-flight non-blocking weight upload: its completion event
 *        plus the staging buffer that must outlive the transfer.
 */
struct V8cPendingUpload {
  cl_event event = nullptr;
  std::vector<uint8_t> staging;
};

std::mutex &v8c_pending_mtx() {
  static std::mutex m;
  return m;
}
std::vector<V8cPendingUpload> &v8c_pending_list() {
  static std::vector<V8cPendingUpload> l;
  return l;
}
std::atomic<size_t> v8c_pending_bytes{0};
constexpr size_t V8C_PENDING_CAP_BYTES = 256u << 20;

bool v8c_upload_async_on() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_V8C_UPLOAD_ASYNC");
    return !(e && e[0] == '0'); // default ON; =0 restores blocking writes
  }();
  return on;
}

void v8c_wait_and_free(std::vector<V8cPendingUpload> &batch) {
  for (auto &p : batch) {
    if (p.event) {
      opencl::clWaitForEvents(1, &p.event);
      opencl::clReleaseEvent(p.event);
    }
  }
  batch.clear();
}

void v8c_push_pending(cl_event ev, std::vector<uint8_t> &&staging) {
  std::vector<V8cPendingUpload> overflow;
  {
    std::lock_guard<std::mutex> lock(v8c_pending_mtx());
    auto &list = v8c_pending_list();
    const size_t bytes = v8c_pending_bytes.load(std::memory_order_relaxed);
    if (bytes + staging.size() > V8C_PENDING_CAP_BYTES) {
      overflow.swap(list);
      v8c_pending_bytes.store(0, std::memory_order_relaxed);
    }
    v8c_pending_bytes.fetch_add(staging.size(), std::memory_order_relaxed);
    V8cPendingUpload p;
    p.event = ev;
    p.staging = std::move(staging);
    list.push_back(std::move(p));
  }
  v8c_wait_and_free(overflow); // waits happen outside the lock
}

} // namespace

void v8c_flush_pending_uploads() {
  if (v8c_pending_bytes.load(std::memory_order_relaxed) == 0)
    return;
  std::vector<V8cPendingUpload> batch;
  {
    std::lock_guard<std::mutex> lock(v8c_pending_mtx());
    batch.swap(v8c_pending_list());
    v8c_pending_bytes.store(0, std::memory_order_relaxed);
  }
  v8c_wait_and_free(batch);
}

std::unique_ptr<tv::TensorBacking> make_v8c_weight_backing_from_qs4cx(
  const uint8_t *plain_nibbles, const float *fp32_scales, unsigned int N,
  unsigned int K, cl_mem *out_scale_buf, cl_mem *out_row_sum_w_int4_buf,
  const char *cache_name) {
  if (K % 32 != 0)
    throw std::invalid_argument(
      "make_v8c_weight_backing_from_qs4cx: K must be a multiple of 32");
  // Every dispatch site rejects N % 8 before reaching a builder, because the
  // GEMM stores eight output channels per work-item. Stating the weaker rule
  // here made two different rules for one constraint; state the real one.
  if (N % 8 != 0)
    throw std::invalid_argument(
      "make_v8c_weight_backing_from_qs4cx: N must be a multiple of 8");

  const size_t plain_row_bytes = ((size_t)K + 1) / 2; // QS4CX nibble stride
  const size_t k_blocks = K / 32;
  // [Adreno pitch fix] image2d-from-buffer requires the row pitch to be a
  // multiple of the device CL_DEVICE_IMAGE_PITCH_ALIGNMENT; on Adreno an
  // unaligned pitch (e.g. a K=192 projection -> K/2=96 B) fails
  // clCreateImage, forcing a wrong fallback (the lm-head GEMV) that corrupts a
  // normal per-layer FC. Pad each weight row up to a 64-byte multiple so the
  // image always builds and the FC stays on the correct v8c GPU path. The
  // padding bytes are zero and never read (the kernel K-loop uses K/32 texels;
  // only the ROW STRIDE grows -- W_wgt is set to this padded texel stride).
  // Adreno-only (v8c_wrow_bytes gates on caps.image_v8c); Intel keeps K/2.
  const size_t v8c_row_bytes = v8c_wrow_bytes(K);
  const size_t total_bytes = (size_t)N * v8c_row_bytes;

  // Read the offset-encoded (int4+8) nibble for input index k of a plain row.
  auto plain_nib = [](const uint8_t *row, size_t k) -> uint8_t {
    const uint8_t byte = row[k >> 1];
    return (k & 1) ? (uint8_t)((byte >> 4) & 0x0F) : (uint8_t)(byte & 0x0F);
  };

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();

  cl_int err = CL_SUCCESS;
  // [peak-chunk] The old path built a FULL host mirror then COPY_HOST_PTR'd
  // it -- for the untied lm_head (N=262144) that is a 336MB peak-only
  // transient, the single biggest load-time WS spike left after DROP_PLAIN.
  // Build into an empty buffer via bounded chunks instead: permute + row-sum
  // fold into ONE pass over each chunk, and the host staging stays at
  // CHUNK_BYTES regardless of N. Byte-identical output (same permute, same
  // zeroed padding, same row-sum math).
  //
  // [NNTR_V8C_HOSTPTR, default OFF -- measured worse, see v8c_hostptr_on()] The
  // idea was that a device clCreateBuffer might carry a host shadow on WDDM and
  // that CL_MEM_ALLOC_HOST_PTR (fill through one map, no staging) would
  // collapse it to a single zero-copy allocation. It does the opposite: +1434MB
  // peak. The plain device buffer below is already the single copy.
  const bool hostptr = v8c_hostptr_on();
  cl_mem_flags wflags = CL_MEM_READ_ONLY;
  if (hostptr)
    wflags |= CL_MEM_ALLOC_HOST_PTR;
  cl_mem w_buf =
    opencl::clCreateBuffer(ctx, wflags, total_bytes, nullptr, &err);
  if (err != CL_SUCCESS || !w_buf)
    throw std::runtime_error("make_v8c_weight_backing_from_qs4cx: "
                             "clCreateBuffer (weight) failed: " +
                             std::to_string(err));

  std::vector<int32_t> row_sum_w_int4(N, 0);
  const size_t real_row_bytes = (size_t)K / 2;
  cl_command_queue cq = blas_cc->command_queue_inst_.GetCommandQueue();

  // Cache hit: the permute and the row-sum fold are a deterministic pure
  // function of the plain nibbles, so a validated record from a previous run
  // is the same bytes this pass would produce. Upload it straight from the
  // pack mapping -- no staging copy, no permute, no row-sum pass -- and drop
  // the file pages per chunk, so the transient residency of the pack stays at
  // one chunk instead of the whole payload.
  // Fingerprint of the SOURCE nibbles, so a weight file replaced in place
  // with one of the same size and timestamp cannot serve a stale pack. It is
  // sampled, so it costs a fixed ~192 KB of hashing whether the lookup hits
  // or misses.
  const uint64_t src_fnv =
    (cache_name != nullptr)
      ? v8c_pack::source_fingerprint(plain_nibbles, (size_t)N * plain_row_bytes)
      : 0;
  bool from_cache = false;
  if (!hostptr && cache_name != nullptr) {
    v8c_pack::Hit hit;
    if (v8c_pack::lookup(cache_name, N, K, v8c_row_bytes, total_bytes, src_fnv,
                         hit)) {
      constexpr size_t UP_CHUNK = 64u << 20;
      cl_int werr = CL_SUCCESS;
      for (size_t off = 0; off < total_bytes && werr == CL_SUCCESS;
           off += UP_CHUNK) {
        const size_t len = std::min(UP_CHUNK, total_bytes - off);
        werr = opencl::clEnqueueWriteBuffer(
          cq, w_buf, CL_TRUE, off, len, hit.payload + off, 0, nullptr, nullptr);
        v8c_pack::Hit consumed;
        consumed.payload = hit.payload + off;
        consumed.payload_len = len;
        v8c_pack::payload_consumed(consumed);
      }
      if (werr == CL_SUCCESS) {
        std::memcpy(row_sum_w_int4.data(), hit.rowsum,
                    (size_t)N * sizeof(int32_t));
        from_cache = true;
      }
      // An upload error simply falls through to the derive below.
    }
  }

  // Cache miss: tee each packed chunk to the pack's temp file as it is
  // derived. The writes go to disjoint offsets, so loader workers deriving
  // different weights stay independent, and the guard drops the record if
  // anything below throws, so a half-derived weight is never indexed.
  struct PackRecGuard {
    v8c_pack::RecordWriter *rw = nullptr;
    ~PackRecGuard() {
      if (rw)
        v8c_pack::abort_record(rw);
    }
  } pack_rec;
  if (!from_cache && !hostptr && cache_name != nullptr)
    pack_rec.rw = v8c_pack::begin_record(cache_name, N, K, v8c_row_bytes,
                                         total_bytes, src_fnv);

  // hostptr: map the whole buffer once and build straight into it (no staging).
  uint8_t *map_ptr = nullptr;
  if (hostptr) {
    map_ptr = static_cast<uint8_t *>(opencl::clEnqueueMapBuffer(
      cq, w_buf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, total_bytes, 0,
      nullptr, nullptr, &err));
    if (err != CL_SUCCESS || !map_ptr) {
      opencl::clReleaseMemObject(w_buf);
      throw std::runtime_error(
        "make_v8c_weight_backing_from_qs4cx: clEnqueueMapBuffer failed: " +
        std::to_string(err));
    }
    std::memset(map_ptr, 0, total_bytes); // padding stays 0
  }

  constexpr size_t CHUNK_BYTES = 16u << 20;
  // One pass over the whole N when mapped; bounded chunks otherwise.
  const size_t chunk_rows =
    hostptr ? (size_t)N : std::max<size_t>(1, CHUNK_BYTES / v8c_row_bytes);

  // Pack rows [n0, n0+nrows) into dst and fold the per-channel int4 row
  // sums.
  auto pack_rows = [&](size_t n0, size_t nrows, uint8_t *dst) {
    for (size_t r = 0; r < nrows; ++r) {
      const size_t n = n0 + r;
      const uint8_t *plain_row = plain_nibbles + n * plain_row_bytes;
      uint8_t *v8c_row = dst + r * v8c_row_bytes;
      // v8c K-block byte order (matches the KAI builder): byte(c*4+b) =
      // (qH<<4)|qL with qL = K(kblk*32+c*8+b), qH = K(kblk*32+c*8+b+4).
      for (size_t kblk = 0; kblk < k_blocks; ++kblk) {
        uint8_t *v8c_kblk = v8c_row + kblk * 16;
        const size_t kbase = kblk * 32;
        for (size_t c = 0; c < 4; ++c) {
          for (size_t b = 0; b < 4; ++b) {
            const uint8_t qL = plain_nib(plain_row, kbase + c * 8 + b);
            const uint8_t qH = plain_nib(plain_row, kbase + c * 8 + b + 4);
            v8c_kblk[c * 4 + b] = (uint8_t)((qH << 4) | qL);
          }
        }
      }
      // Per-channel int4 row sum (asymmetric-act zero-point correction),
      // folded into the same pass -- sum ONLY the real K/2 bytes, not the
      // Adreno pitch padding (a 0x00 pad byte would decode as two -8s).
      int32_t s = 0;
      for (size_t off = 0; off < real_row_bytes; ++off) {
        const uint8_t byte = v8c_row[off];
        s += ((int)(byte & 0x0Fu) - 8) + ((int)((byte >> 4) & 0x0Fu) - 8);
      }
      row_sum_w_int4[n] = s;
    }
  };

  // For the big weights (in practice an untied output projection,
  // N=262144 -> 336MB) the single-threaded permute is the longest pole of
  // model init even once the per-weight builds run concurrently. Pack
  // independent chunks from a small crew: writes to disjoint offsets are
  // thread-safe on the shared queue and the row-sum rows are disjoint too.
  // Small weights keep the serial path (thread spin-up would dominate).
  //
  // What this parallelizes is the CPU permute, not the upload: the crew's
  // writes are blocking on one in-order queue, so they still serialize
  // behind each other. The permute is the pole, so that is where the win is.
  //
  // Bound, stated honestly: this function is itself called concurrently by
  // the loader's weight workers, so the crew below is per CALL. The peak is
  // therefore (concurrent builds) x n_workers x CHUNK_BYTES, capped by the
  // global budget below rather than by the per-call limit alone.
  constexpr size_t PAR_THRESHOLD_BYTES = 64u << 20;
  const size_t n_chunks = ((size_t)N + chunk_rows - 1) / chunk_rows;
  if (from_cache) {
    // payload and row sums already came from the pack mapping above
  } else if (!hostptr && total_bytes >= PAR_THRESHOLD_BYTES && n_chunks > 1) {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0)
      hw = 4;
    // Global budget over every concurrent build, so N loader workers cannot
    // each spawn a full crew and oversubscribe the machine (and its staging)
    // by a factor of N. A build that finds the budget spent packs serially.
    static std::atomic<size_t> crew_budget{std::max<size_t>(hw, 4)};
    size_t n_workers = std::min(std::min((size_t)hw, n_chunks), (size_t)4);
    size_t taken = 0;
    for (size_t want = n_workers; want > 0; --want) {
      size_t avail = crew_budget.load(std::memory_order_relaxed);
      if (avail == 0)
        break;
      const size_t give = std::min(avail, want);
      if (crew_budget.compare_exchange_weak(avail, avail - give,
                                            std::memory_order_relaxed)) {
        taken = give;
        break;
      }
    }
    // At least one worker either way: an exhausted budget must degrade to a
    // serial pack of this weight, never to no pack at all.
    n_workers = std::max<size_t>(taken, 1);
    struct CrewBudget {
      size_t n;
      std::atomic<size_t> *b;
      ~CrewBudget() { b->fetch_add(n, std::memory_order_relaxed); }
    } crew_budget_guard{taken, &crew_budget};
    std::atomic<size_t> next_chunk{0};
    std::atomic<cl_int> chunk_err{CL_SUCCESS};
    std::vector<std::thread> crew;
    crew.reserve(n_workers);
    for (size_t t = 0; t < n_workers; ++t) {
      crew.emplace_back([&]() {
        std::vector<uint8_t> staging(chunk_rows * v8c_row_bytes);
        for (;;) {
          const size_t ci = next_chunk.fetch_add(1);
          if (ci >= n_chunks || chunk_err.load() != CL_SUCCESS)
            break;
          const size_t n0 = ci * chunk_rows;
          const size_t nrows = std::min(chunk_rows, (size_t)N - n0);
          std::memset(staging.data(), 0, nrows * v8c_row_bytes);
          pack_rows(n0, nrows, staging.data());
          v8c_pack::record_write(pack_rec.rw, n0 * v8c_row_bytes,
                                 staging.data(), nrows * v8c_row_bytes);
          const cl_int werr = opencl::clEnqueueWriteBuffer(
            cq, w_buf, CL_TRUE, n0 * v8c_row_bytes, nrows * v8c_row_bytes,
            staging.data(), 0, nullptr, nullptr);
          if (werr != CL_SUCCESS)
            chunk_err.store(werr);
        }
      });
    }
    for (auto &th : crew)
      th.join();
    if (chunk_err.load() != CL_SUCCESS) {
      opencl::clReleaseMemObject(w_buf);
      throw std::runtime_error(
        "make_v8c_weight_backing_from_qs4cx: chunk write failed: " +
        std::to_string(chunk_err.load()));
    }
  } else {
    // Default: pack into a per-chunk staging vector,
    // enqueue a NON-blocking write and hand the staging to the pending
    // registry (freed on event completion). This removes the per-chunk
    // submit->wait round trip from every load worker; the in-order queue
    // sequences later GEMMs after the writes. NNTR_V8C_UPLOAD_ASYNC=0
    // restores the blocking path (A/B lever).
    const bool upload_async = !hostptr && v8c_upload_async_on();
    std::vector<uint8_t> packed;
    if (!hostptr && !upload_async)
      packed.assign(chunk_rows * v8c_row_bytes, 0);
    for (size_t n0 = 0; n0 < N; n0 += chunk_rows) {
      const size_t nrows = std::min(chunk_rows, (size_t)N - n0);
      std::vector<uint8_t> chunk_staging;
      uint8_t *dst;
      if (hostptr) {
        dst = map_ptr + n0 * v8c_row_bytes;
      } else if (upload_async) {
        chunk_staging.assign(nrows * v8c_row_bytes, 0); // padding stays 0
        dst = chunk_staging.data();
      } else {
        dst = packed.data();
        std::memset(dst, 0, nrows * v8c_row_bytes); // padding stays 0
      }
      pack_rows(n0, nrows, dst);
      v8c_pack::record_write(pack_rec.rw, n0 * v8c_row_bytes, dst,
                             nrows * v8c_row_bytes);
      if (!hostptr) {
        if (upload_async) {
          cl_event ev = nullptr;
          const cl_int werr = opencl::clEnqueueWriteBuffer(
            cq, w_buf, CL_FALSE, n0 * v8c_row_bytes, nrows * v8c_row_bytes,
            chunk_staging.data(), 0, nullptr, &ev);
          if (werr != CL_SUCCESS) {
            opencl::clReleaseMemObject(w_buf);
            throw std::runtime_error(
              "make_v8c_weight_backing_from_qs4cx: chunk write failed: " +
              std::to_string(werr));
          }
          v8c_push_pending(ev, std::move(chunk_staging));
        } else {
          const cl_int werr = opencl::clEnqueueWriteBuffer(
            cq, w_buf, CL_TRUE, n0 * v8c_row_bytes, nrows * v8c_row_bytes,
            packed.data(), 0, nullptr, nullptr);
          if (werr != CL_SUCCESS) {
            opencl::clReleaseMemObject(w_buf);
            throw std::runtime_error(
              "make_v8c_weight_backing_from_qs4cx: chunk write failed: " +
              std::to_string(werr));
          }
        }
      }
    }
  }
  if (hostptr) {
    const cl_int uerr =
      opencl::clEnqueueUnmapMemObject(cq, w_buf, map_ptr, 0, nullptr, nullptr);
    if (uerr != CL_SUCCESS) {
      opencl::clReleaseMemObject(w_buf);
      throw std::runtime_error(
        "make_v8c_weight_backing_from_qs4cx: clEnqueueUnmapMemObject failed: " +
        std::to_string(uerr));
    }
    opencl::clFinish(
      cq); // the unmap must land before the GEMM binds this weight
  }

  // Derive finished: append the row sums, checksum the record and index it.
  // A no-op when the guard holds no writer; commit_record owns the handle.
  if (pack_rec.rw) {
    v8c_pack::commit_record(pack_rec.rw, row_sum_w_int4.data(), N);
    pack_rec.rw = nullptr;
  }

  auto backing = std::make_unique<tv::TensorBacking>(
    ctx, w_buf, tv::Encoding::INT4_OFFSET, tv::Layout::ROW_MAJOR, total_bytes,
    /** owned */ true);

  // Per-channel scale: QS4CX stores fp32 directly (no fp16->fp32 promotion).
  cl_mem sb =
    opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * N, (void *)fp32_scales, &err);
  if (err != CL_SUCCESS || !sb)
    throw std::runtime_error("make_v8c_weight_backing_from_qs4cx: "
                             "clCreateBuffer (scale) failed: " +
                             std::to_string(err));
  // Per-channel int4 row sum: computed in the chunk loop above.
  cl_mem rsw_buf =
    opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(int32_t) * N, row_sum_w_int4.data(), &err);
  if (err != CL_SUCCESS || !rsw_buf) {
    // The scale buffer is this function's until both out-parameters are set:
    // the caller's handler cannot release what it was never handed, so a
    // throw between the two allocations would leak it.
    opencl::clReleaseMemObject(sb);
    throw std::runtime_error("make_v8c_weight_backing_from_qs4cx: "
                             "clCreateBuffer (row_sum) failed: " +
                             std::to_string(err));
  }
  *out_scale_buf = sb;
  *out_row_sum_w_int4_buf = rsw_buf;

  return backing;
}

/**
 * @brief 8/4/4 paper attention path: int8(act) × int8(weight) channel-wise
 * GEMM.
 */
void gemm_int8_int8_v8c_cl(cl_mem act_image, cl_mem weight_image,
                           cl_mem scale_act, cl_mem scale_wgt,
                           cl_mem row_sum_act, cl_mem zp_act, cl_mem row_sum_w,
                           cl_mem output_fp16, unsigned int M, unsigned int N,
                           unsigned int K) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  // Buffer path (NNTR_V8C_BUF=1): int8 weight texel = 16 K, act texel = 16 K.
  const bool use_buf = v8c_use_buffer_path();
  const char *kname =
    use_buf
      ? ((M <= 4) ? "v8c_gemm_int8_int8_m1_buf" : "v8c_gemm_int8_int8_buf")
      : ((M <= 4) ? "v8c_gemm_int8_int8_m1" : "v8c_gemm_int8_int8");
  const std::string copts = use_buf ? kV8cBufCompileOpts : "";
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(int8_int4_gemm_v8c_kernel, kname, copts);
  if (!kp)
    throw std::runtime_error(
      std::string("v8c_gemm_int8_int8: registerClKernel failed: ") + kname);

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &act_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &weight_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_wgt, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &zp_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_w, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &output_fp16, sizeof(cl_mem)))
    throw std::runtime_error("v8c_gemm_int8_int8: cl_mem arg failed");
  int Mi = (int)M, Ni = (int)N, Ki = (int)K;
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ni, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ki, sizeof(int)))
    throw std::runtime_error("v8c_gemm_int8_int8: int arg failed");
  if (use_buf) {
    int W_act = (int)(K / 16), W_wgt = (int)(K / 16);
    if (!kp->SetKernelArguments(arg++, &W_act, sizeof(int)) ||
        !kp->SetKernelArguments(arg++, &W_wgt, sizeof(int)))
      throw std::runtime_error("v8c_gemm_int8_int8: width arg failed");
  }

  if (M <= 4) {
    constexpr size_t TN_M1 = 8;
    std::array<size_t, 3> gws = {(size_t)N / TN_M1, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                               nullptr, 0, nullptr, nullptr);
  } else {
    constexpr size_t TM = 4, TN = 8;
    std::array<size_t, 3> gws = {(size_t)N / TN, (size_t)M / TM, 1};
    // Device-specialized LWS (shared with the int4 path): 4×16 sweet spot
    // capped to device max work-group size + must divide gws, else NULL.
    std::array<size_t, 2> picked{};
    const bool lws_ok = v8c_pick_lws(gws[0], gws[1], picked);
    std::array<size_t, 3> lws = {picked[0], picked[1], 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                               lws_ok ? lws.data() : nullptr, 0,
                                               nullptr, nullptr);
  }
}

void gemm_int8_int8_v8c_v_ohwi_cl(
  cl_mem act_image, cl_mem weight_image, cl_mem scale_act, cl_mem scale_wgt,
  cl_mem row_sum_act, cl_mem zp_act, cl_mem row_sum_w, cl_mem v_ohwi,
  unsigned int M_pad, unsigned int N, unsigned int K, unsigned int head_dim,
  unsigned int S_max, unsigned int position, unsigned int M_real) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (M_pad == 0 || N == 0 || K == 0 || head_dim == 0 || S_max == 0)
    throw std::runtime_error("gemm_int8_int8_v8c_v_ohwi: zero dim");
  constexpr unsigned int V8C_TM = 4, V8C_TN = 8;
  if (M_pad % V8C_TM != 0 || N % V8C_TN != 0 || K % 32 != 0)
    throw std::runtime_error("gemm_int8_int8_v8c_v_ohwi: M/N/K not aligned");
  if (head_dim % V8C_TN != 0 || N % head_dim != 0)
    throw std::runtime_error("gemm_int8_int8_v8c_v_ohwi: head_dim constraint");
  if (M_real > M_pad)
    M_real = M_pad;

  const bool use_buf = v8c_use_buffer_path();
  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    int8_int4_gemm_v8c_kernel,
    use_buf ? "v8c_gemm_int8_int8_v_ohwi_buf" : "v8c_gemm_int8_int8_v_ohwi",
    use_buf ? kV8cBufCompileOpts : "");
  if (!kp)
    throw std::runtime_error("v8c_int8_v_ohwi: registerClKernel failed");

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &act_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &weight_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_wgt, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &zp_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_w, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &v_ohwi, sizeof(cl_mem)))
    throw std::runtime_error("v8c_int8_v_ohwi: cl_mem arg failed");
  int Mi = (int)M_pad, Ni = (int)N, Ki = (int)K;
  int di = (int)head_dim, Si = (int)S_max, pi = (int)position, Mr = (int)M_real;
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ni, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ki, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &di, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Si, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &pi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Mr, sizeof(int)))
    throw std::runtime_error("v8c_int8_v_ohwi: int arg failed");
  if (use_buf) {
    int W_act = (int)(K / 16), W_wgt = (int)(K / 16);
    if (!kp->SetKernelArguments(arg++, &W_act, sizeof(int)) ||
        !kp->SetKernelArguments(arg++, &W_wgt, sizeof(int)))
      throw std::runtime_error("v8c_int8_v_ohwi: width arg failed");
  }
  std::array<size_t, 3> gws = {(size_t)N / V8C_TN, (size_t)M_pad / V8C_TM, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                             nullptr, 0, nullptr, nullptr);
}

std::unique_ptr<tv::TensorBacking> make_v8c_int8_weight_backing(
  const int8_t *int8_weights, const uint16_t *fp16_scales, unsigned int N,
  unsigned int K, cl_mem *out_scale_buf, cl_mem *out_row_sum_w_buf) {
  if (K % 16 != 0)
    throw std::invalid_argument("make_v8c_int8_weight_backing: K%16!=0");
  if (N % 8 != 0)
    throw std::invalid_argument("make_v8c_int8_weight_backing: N%8!=0");

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  const size_t nbytes = (size_t)N * K; // 1 byte/int8 weight, plain row-major

  cl_int err = CL_SUCCESS;
  cl_mem w_buf =
    opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nbytes,
                           const_cast<int8_t *>(int8_weights), &err);
  if (err != CL_SUCCESS || !w_buf)
    throw std::runtime_error(
      "make_v8c_int8_weight_backing: clCreateBuffer (weight) failed: " +
      std::to_string(err));
  auto backing = std::make_unique<tv::TensorBacking>(
    ctx, w_buf, tv::Encoding::INT8, tv::Layout::ROW_MAJOR, nbytes,
    /** owned */ true);

  std::vector<float> per_channel_scale(N);
  for (unsigned int n = 0; n < N; ++n)
    per_channel_scale[n] = compute_fp16_to_fp32(fp16_scales[n]);
  cl_mem sb =
    opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * N, per_channel_scale.data(), &err);
  if (err != CL_SUCCESS || !sb)
    throw std::runtime_error(
      "make_v8c_int8_weight_backing: clCreateBuffer (scale) failed");
  *out_scale_buf = sb;

  std::vector<int32_t> row_sum_w(N, 0);
  for (unsigned int n = 0; n < N; ++n) {
    const int8_t *row = int8_weights + (size_t)n * K;
    int32_t s = 0;
    for (unsigned int k = 0; k < K; ++k)
      s += (int)row[k];
    row_sum_w[n] = s;
  }
  cl_mem rb =
    opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(int32_t) * N, row_sum_w.data(), &err);
  if (err != CL_SUCCESS || !rb)
    throw std::runtime_error("make_v8c_int8_weight_backing: "
                             "clCreateBuffer (row_sum_w) failed");
  *out_row_sum_w_buf = rb;

  return backing;
}

// Decode lm_head GEMV on a Q6_K weight — the gpu_native q6k_gemv_lmhead
// kernel verbatim (ML Drift reaudit #1; see blas_kernels.h doc). One 64-WI
// workgroup per vocab row; each WI owns one (block-lane, n-half, l-quad)
// unit with direct per-field uchar4 loads. Replicates
// dequantize_row_q6_K_impl exactly; only the fp32 summation order differs

static const std::string lmhead_q6k_gemv_kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void q6k_gemv_lmhead(__global const uchar *W,   // [V rows][H/256 blocks][210 B]
                              __constant float *x,        // [H] post-norm hidden
                              __global float *logits,     // [V]
                              const int V, const int H) {
  const int row = get_group_id(0);
  const int t = get_local_id(0); // 0..63
  const int nb = H >> 8;         // Q6_K blocks per row
  __global const uchar *rb = W + (size_t)row * (size_t)(nb * 210);
  const int bl = t >> 4;         // block lane 0..3
  const int u = t & 15;          // unit within block
  const int nh = u >> 3;         // n-half: 0 -> elems [0,128), 1 -> [128,256)
  const int q = u & 7;           // l-quad: l = 4q .. 4q+3
  float sum = 0.0f;
  for (int s = 0; s < nb; s += 4) {
    const int bi = s + bl;
    if (bi < nb) {
      __global const uchar *blk = rb + bi * 210;
      const uchar4 qlo = vload4(0, blk + (nh << 6) + (q << 2));
      const uchar4 qhi = vload4(0, blk + (nh << 6) + 32 + (q << 2));
      const uchar4 qh4 = vload4(0, blk + 128 + (nh << 5) + (q << 2));
      const float d = vload_half(0, (__global const half *)(blk + 208));
      const int is = q >> 2;     // (4q)/16 == (4q+3)/16 for q in 0..7
      const int sbase = 192 + (nh << 3);
      const float s0 = d * (float)((__global const char *)blk)[sbase + is];
      const float s2 = d * (float)((__global const char *)blk)[sbase + is + 2];
      const float s4 = d * (float)((__global const char *)blk)[sbase + is + 4];
      const float s6 = d * (float)((__global const char *)blk)[sbase + is + 6];
    const int yb = (bi << 8) + (nh << 7) + (q << 2);
    float4 a1, a2, a3, a4;
    a1.x = (float)((int)((qlo.x & 0xF) | (((qh4.x >> 0) & 3) << 4)) - 32);
    a1.y = (float)((int)((qlo.y & 0xF) | (((qh4.y >> 0) & 3) << 4)) - 32);
    a1.z = (float)((int)((qlo.z & 0xF) | (((qh4.z >> 0) & 3) << 4)) - 32);
    a1.w = (float)((int)((qlo.w & 0xF) | (((qh4.w >> 0) & 3) << 4)) - 32);
    a2.x = (float)((int)((qhi.x & 0xF) | (((qh4.x >> 2) & 3) << 4)) - 32);
    a2.y = (float)((int)((qhi.y & 0xF) | (((qh4.y >> 2) & 3) << 4)) - 32);
    a2.z = (float)((int)((qhi.z & 0xF) | (((qh4.z >> 2) & 3) << 4)) - 32);
    a2.w = (float)((int)((qhi.w & 0xF) | (((qh4.w >> 2) & 3) << 4)) - 32);
    a3.x = (float)((int)((qlo.x >> 4) | (((qh4.x >> 4) & 3) << 4)) - 32);
    a3.y = (float)((int)((qlo.y >> 4) | (((qh4.y >> 4) & 3) << 4)) - 32);
    a3.z = (float)((int)((qlo.z >> 4) | (((qh4.z >> 4) & 3) << 4)) - 32);
    a3.w = (float)((int)((qlo.w >> 4) | (((qh4.w >> 4) & 3) << 4)) - 32);
    a4.x = (float)((int)((qhi.x >> 4) | (((qh4.x >> 6) & 3) << 4)) - 32);
    a4.y = (float)((int)((qhi.y >> 4) | (((qh4.y >> 6) & 3) << 4)) - 32);
    a4.z = (float)((int)((qhi.z >> 4) | (((qh4.z >> 6) & 3) << 4)) - 32);
    a4.w = (float)((int)((qhi.w >> 4) | (((qh4.w >> 6) & 3) << 4)) - 32);
    const float4 x1 = (float4)(x[yb], x[yb + 1], x[yb + 2], x[yb + 3]);
    const float4 x2 = (float4)(x[yb + 32], x[yb + 33], x[yb + 34], x[yb + 35]);
    const float4 x3 = (float4)(x[yb + 64], x[yb + 65], x[yb + 66], x[yb + 67]);
    const float4 x4 = (float4)(x[yb + 96], x[yb + 97], x[yb + 98], x[yb + 99]);
    sum += s0 * dot(a1, x1) + s2 * dot(a2, x2) + s4 * dot(a3, x3) +
           s6 * dot(a4, x4);
    }
  }
  __local float red[64];
  red[t] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = 32; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) logits[row] = red[0];
}
)CL";

bool lmhead_gemv_q6_k_cl(const void *w_q6k_host, const float *act_f32_host,
                         float *logits_f32_host, unsigned int vocab,
                         unsigned int hidden) {
  if (hidden == 0 || (hidden % 256) != 0 || vocab == 0)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc)
    return false;
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  if (!ctx || !q)
    return false;

  // Per-weight device residency (the Q6_K table never changes after load):
  // weight + act + logits buffers keyed by the weight host pointer.
  struct LmheadEntry {
    cl_mem w = nullptr;
    cl_mem x = nullptr;
    cl_mem out = nullptr;
  };
  static std::unordered_map<const void *, LmheadEntry> cache;
  LmheadEntry &e = cache[w_q6k_host];
  const size_t nb = hidden / 256;
  const size_t w_bytes = (size_t)vocab * nb * 210;
  cl_int err = CL_SUCCESS;
  if (e.w == nullptr) {
    e.w = opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 w_bytes, const_cast<void *>(w_q6k_host), &err);
    if (err != CL_SUCCESS || !e.w) {
      std::fprintf(stderr, "[lmhead-q6k] weight clCreateBuffer(%zu B) err=%d\n",
                   w_bytes, err);
      e.w = nullptr;
      return false;
    }
    e.x = opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(float) * hidden,
                                 nullptr, &err);
    e.out = opencl::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * vocab, nullptr, &err);
    if (!e.x || !e.out) {
      std::fprintf(stderr, "[lmhead-q6k] act/out clCreateBuffer err=%d\n", err);
      if (e.w)
        opencl::clReleaseMemObject(e.w);
      if (e.x)
        opencl::clReleaseMemObject(e.x);
      if (e.out)
        opencl::clReleaseMemObject(e.out);
      cache.erase(w_q6k_host);
      return false;
    }
  }

  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(lmhead_q6k_gemv_kernel, "q6k_gemv_lmhead");
  if (!kp) {
    static int logged = 0;
    if (!logged++)
      std::fprintf(stderr, "[lmhead-q6k] registerClKernel failed\n");
    return false;
  }

  if (opencl::clEnqueueWriteBuffer(q, e.x, CL_FALSE, 0, sizeof(float) * hidden,
                                   act_f32_host, 0, nullptr,
                                   nullptr) != CL_SUCCESS) {
    std::fprintf(stderr, "[lmhead-q6k] act write failed\n");
    return false;
  }

  int Vi = (int)vocab, Hi = (int)hidden;
  int a = 0;
  if (!(kp->SetKernelArguments(a++, &e.w, sizeof(cl_mem)) &&
        kp->SetKernelArguments(a++, &e.x, sizeof(cl_mem)) &&
        kp->SetKernelArguments(a++, &e.out, sizeof(cl_mem)) &&
        kp->SetKernelArguments(a++, &Vi, sizeof(int)) &&
        kp->SetKernelArguments(a++, &Hi, sizeof(int))))
    return false;

  std::array<size_t, 3> gws = {(size_t)vocab * 64, 1, 1};
  std::array<size_t, 3> lws = {64, 1, 1};
  static const bool split_tprof = std::getenv("NNTR_LMHEAD_TPROF") != nullptr;
  const auto t_pre_kernel = std::chrono::steady_clock::now();
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);

  // NNTR_LMHEAD_TPROF: split the kernel vs readback cost (adds a clFinish, so
  // measurement only) to decide whether to optimize the GEMV kernel or the
  // host-readback path.
  std::chrono::steady_clock::time_point t_post_kernel;
  if (split_tprof) {
    opencl::clFinish(q);
    t_post_kernel = std::chrono::steady_clock::now();
  }

  // Blocking readback = the decode-step GPU->host boundary (one per token).
  const auto t_pre_read = std::chrono::steady_clock::now();
  if (opencl::clEnqueueReadBuffer(q, e.out, CL_TRUE, 0, sizeof(float) * vocab,
                                  logits_f32_host, 0, nullptr,
                                  nullptr) != CL_SUCCESS) {
    std::fprintf(stderr, "[lmhead-q6k] logits read failed\n");
    return false;
  }
  static int announced = 0;
  if (announced < 6) {
    ++announced;
    const auto t_end = std::chrono::steady_clock::now();
    if (split_tprof)
      std::fprintf(
        stderr,
        "[lmhead-q6k] call#%d V=%u H=%u kernel=%.2f read=%.2f total=%.2f ms\n",
        announced, vocab, hidden,
        std::chrono::duration<double, std::milli>(t_post_kernel - t_pre_kernel)
          .count(),
        std::chrono::duration<double, std::milli>(t_end - t_post_kernel)
          .count(),
        std::chrono::duration<double, std::milli>(t_end - t_pre_kernel)
          .count());
    else
      std::fprintf(
        stderr, "[lmhead-q6k] call#%d V=%u H=%u drain+gemv+read=%.2f ms\n",
        announced, vocab, hidden,
        std::chrono::duration<double, std::milli>(t_end - t_pre_read).count());
  }
  return true;
}

// Decode lm_head GEMV on a QINT4 (v8c row-major) weight buffer. For the untied
// int4 lm_head N=vocab=262144 exceeds the image2d height cap (~16384) so
// dotCl_v8c's image GEMM cannot run; this reads the already-built v8c row-major
// nibble buffer directly. v8c layout
// (make_v8c_weight_backing_from_kai_section_a): row n is K/2 contiguous bytes;
// within a row each 32-K block is 16 bytes at kblk*16; within a 16-byte block
// the byte at (c*4+b) (c,b in 0..3) holds two offset-encoded nibbles -- low =
// K(c*8+b), high = K(c*8+b+4), value = nibble-8. Activation stays fp16,
// accumulated in fp32 (no int8 act quant -> best argmax fidelity, matching
// q6k/fp32w lm_head). 64-WI workgroup per row, LDS-tree reduce; the coalesced
// row read (768 B/row) streams ~0.6x the q6k table bytes with a far simpler
// unpack.
static const std::string lmhead_int4_v8c_kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void lmhead_int4_v8c_gemv(__global const uchar *W,    // [N][K/2] nibbles
                                   __global const half *x,      // [K] fp16 act
                                   __global const float *scale, // [N] fp32
                                   __global half *logits,       // [N]
                                   const int N, const int K) {
  const int row = get_group_id(0);
  const int t = get_local_id(0); // 0..63
  const int kblocks = K >> 5;    // K/32
  __global const uchar *wr = W + (size_t)row * (size_t)(K >> 1);
  float sum = 0.0f;
  for (int kb = t; kb < kblocks; kb += 64) {
    __global const uchar *blk = wr + (kb << 4); // 16 bytes
    __global const half *xb = x + (kb << 5);    // K base for this block
    // Vectorized: each 4-byte group c (uchar4) decodes to 8 nibbles whose K
    // indices are contiguous -- lo nibbles -> K=[c*8 .. c*8+3], hi nibbles ->
    // K=[c*8+4 .. c*8+7] -- so the activation is one contiguous half8 and the
    // whole group is two dot4s.
    for (int c = 0; c < 4; ++c) {
      const uchar4 by = vload4(0, blk + (c << 2));
      const float4 lo = convert_float4(convert_int4(by & (uchar4)0x0F) - 8);
      const float4 hi = convert_float4(convert_int4(by >> (uchar4)4) - 8);
      const float8 a = convert_float8(vload8(0, xb + (c << 3)));
      sum += dot(a.lo, lo) + dot(a.hi, hi);
    }
  }
  __local float red[64];
  red[t] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = 32; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) logits[row] = (half)(red[0] * scale[row]);
}
)CL";

bool lmhead_int4_v8c_gemv_cl(void *w_buf_clmem, void *scale_buf_clmem,
                             void *act, bool act_is_clmem, void *logits_host,
                             bool out_fp16, unsigned int N, unsigned int K) {
  if (K == 0 || (K % 32) != 0 || N == 0 || !w_buf_clmem || !scale_buf_clmem ||
      !act || !logits_host)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc)
    return false;
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  if (!ctx || !q)
    return false;

  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(lmhead_int4_v8c_kernel, "lmhead_int4_v8c_gemv");
  if (!kp) {
    static int logged = 0;
    if (!logged++)
      std::fprintf(stderr, "[lmhead-int4] registerClKernel failed\n");
    return false;
  }

  // Cached device logits buffer (fp16, [N]); the lm_head N never changes.
  static cl_mem out_buf = nullptr;
  static size_t out_cap = 0;
  const size_t out_bytes = sizeof(uint16_t) * (size_t)N;
  if (out_buf == nullptr || out_cap < out_bytes) {
    if (out_buf)
      opencl::clReleaseMemObject(out_buf);
    cl_int e = CL_SUCCESS;
    out_buf =
      opencl::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_bytes, nullptr, &e);
    if (e != CL_SUCCESS || !out_buf) {
      out_buf = nullptr;
      out_cap = 0;
      return false;
    }
    out_cap = out_bytes;
  }

  cl_mem w_buf = static_cast<cl_mem>(w_buf_clmem);
  cl_mem scale_buf = static_cast<cl_mem>(scale_buf_clmem);
  int Ni = (int)N, Ki = (int)K;
  bool ok = kp->SetKernelArguments(0, &w_buf, sizeof(cl_mem));
  if (act_is_clmem) {
    cl_mem a = static_cast<cl_mem>(act);
    ok = ok && kp->SetKernelArguments(1, &a, sizeof(cl_mem));
  } else {
    ok = ok && kp->SetKernelSVMArguments(1, act);
  }
  ok = ok && kp->SetKernelArguments(2, &scale_buf, sizeof(cl_mem)) &&
       kp->SetKernelArguments(3, &out_buf, sizeof(cl_mem)) &&
       kp->SetKernelArguments(4, &Ni, sizeof(int)) &&
       kp->SetKernelArguments(5, &Ki, sizeof(int));
  if (!ok)
    return false;

  const int work_groups_count[3] = {(int)N * 64, 1, 1};
  const int work_group_size[3] = {64, 1, 1};
  static const bool tprof = std::getenv("NNTR_LMHEAD_TPROF") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, work_groups_count,
                                                    work_group_size))
    return false;
  std::chrono::steady_clock::time_point t1;
  if (tprof) {
    opencl::clFinish(q);
    t1 = std::chrono::steady_clock::now();
  }

  // Blocking readback to the host output (consumed by the host argmax/sampler).
  if (out_fp16) {
    if (opencl::clEnqueueReadBuffer(q, out_buf, CL_TRUE, 0, out_bytes,
                                    logits_host, 0, nullptr,
                                    nullptr) != CL_SUCCESS)
      return false;
  } else {
    std::vector<uint16_t> y_host(N);
    if (opencl::clEnqueueReadBuffer(q, out_buf, CL_TRUE, 0, out_bytes,
                                    y_host.data(), 0, nullptr,
                                    nullptr) != CL_SUCCESS)
      return false;
    float *o = static_cast<float *>(logits_host);
    for (unsigned int i = 0; i < N; ++i) {
      // fp16 -> fp32 (matches v8c_h2f / the q6k host conversion).
      const uint16_t h = y_host[i];
      const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
      uint32_t ex = (h >> 10) & 0x1fu, m = h & 0x3ffu, bits;
      if (ex == 0) {
        if (m == 0)
          bits = s;
        else {
          ex = 1;
          while ((m & 0x400u) == 0) {
            m <<= 1;
            ex--;
          }
          m &= 0x3ffu;
          bits = s | ((ex + 112) << 23) | (m << 13);
        }
      } else if (ex == 0x1f) {
        bits = s | 0x7f800000u | (m << 13);
      } else {
        bits = s | ((ex + 112) << 23) | (m << 13);
      }
      std::memcpy(&o[i], &bits, 4);
    }
  }
  if (tprof) {
    const auto t2 = std::chrono::steady_clock::now();
    static int announced = 0;
    if (announced < 6) {
      ++announced;
      std::fprintf(
        stderr,
        "[lmhead-int4] call#%d N=%u K=%u kernel=%.2f read+cvt=%.2f ms\n",
        announced, N, K,
        std::chrono::duration<double, std::milli>(t1 - t0).count(),
        std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
  }
  return true;
}

// High-precision lm_head GEMV on an UNQUANTIZED FP32 weight. The Q6_K lm_head
// (q6k_gemv_lmhead above) loses ~1.66 logit on the first-token argmax (the
// <think> vs garbage decision on Qwen3 thinking models => garbage "noise
// prefix"). This reads the full-precision FP32 embed weight directly — fp32 W ×
// fp16 act, fp32 accumulate — matching the HF reference that ranks the correct
// token at 0. Same 64-WI-per-row + LDS-tree-reduce shape as the Q6_K kernel.
static const std::string lmhead_fp32w_gemv_kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void lmhead_gemv_fp32w(__global const float *W,  // [V*H] row-major
                                __global const half *x,    // [H] fp16 act
                                __global float *logits,    // [V]
                                const int V, const int H) {
  const int row = get_group_id(0);
  const int t = get_local_id(0); // 0..63
  __global const float *rb = W + (size_t)row * (size_t)H;
  float sum = 0.0f;
  for (int k = t; k < H; k += 64)
    sum += rb[k] * (float)x[k];
  __local float red[64];
  red[t] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = 32; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) logits[row] = red[0];
}
)CL";

bool lmhead_gemv_fp32w_cl(const void *w_fp32_host, const void *act_fp16_host,
                          float *logits_f32_host, unsigned int vocab,
                          unsigned int hidden) {
  if (hidden == 0 || vocab == 0)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc)
    return false;
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  if (!ctx || !q)
    return false;

  // Per-weight device residency: the embed/lm_head table never changes after
  // load, so cache the device weight buffer (+ act/out scratch) keyed by the
  // weight host pointer.
  struct LmheadFp32Entry {
    cl_mem w = nullptr;
    cl_mem x = nullptr;
    cl_mem out = nullptr;
  };
  static std::unordered_map<const void *, LmheadFp32Entry> cache;
  LmheadFp32Entry &e = cache[w_fp32_host];
  const size_t w_bytes = (size_t)vocab * (size_t)hidden * sizeof(float);
  cl_int err = CL_SUCCESS;
  if (e.w == nullptr) {
    e.w =
      opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             w_bytes, const_cast<void *>(w_fp32_host), &err);
    if (err != CL_SUCCESS || !e.w) {
      std::fprintf(stderr,
                   "[lmhead-fp32w] weight clCreateBuffer(%zu B) err=%d\n",
                   w_bytes, err);
      e.w = nullptr;
      return false;
    }
    e.x = opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                 sizeof(uint16_t) * hidden, nullptr, &err);
    e.out = opencl::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * vocab, nullptr, &err);
    if (!e.x || !e.out) {
      std::fprintf(stderr, "[lmhead-fp32w] act/out clCreateBuffer err=%d\n",
                   err);
      if (e.w)
        opencl::clReleaseMemObject(e.w);
      if (e.x)
        opencl::clReleaseMemObject(e.x);
      if (e.out)
        opencl::clReleaseMemObject(e.out);
      cache.erase(w_fp32_host);
      return false;
    }
  }

  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(lmhead_fp32w_gemv_kernel, "lmhead_gemv_fp32w");
  if (!kp) {
    static int logged = 0;
    if (!logged++)
      std::fprintf(stderr, "[lmhead-fp32w] registerClKernel failed\n");
    return false;
  }

  // act_fp16_host is `hidden` IEEE-binary16 values (byte-identical to OpenCL
  // half); upload as raw uint16 bytes.
  if (opencl::clEnqueueWriteBuffer(q, e.x, CL_FALSE, 0,
                                   sizeof(uint16_t) * hidden, act_fp16_host, 0,
                                   nullptr, nullptr) != CL_SUCCESS) {
    std::fprintf(stderr, "[lmhead-fp32w] act write failed\n");
    return false;
  }

  int Vi = (int)vocab, Hi = (int)hidden;
  int a = 0;
  if (!(kp->SetKernelArguments(a++, &e.w, sizeof(cl_mem)) &&
        kp->SetKernelArguments(a++, &e.x, sizeof(cl_mem)) &&
        kp->SetKernelArguments(a++, &e.out, sizeof(cl_mem)) &&
        kp->SetKernelArguments(a++, &Vi, sizeof(int)) &&
        kp->SetKernelArguments(a++, &Hi, sizeof(int))))
    return false;

  std::array<size_t, 3> gws = {(size_t)vocab * 64, 1, 1};
  std::array<size_t, 3> lws = {64, 1, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);

  if (opencl::clEnqueueReadBuffer(q, e.out, CL_TRUE, 0, sizeof(float) * vocab,
                                  logits_f32_host, 0, nullptr,
                                  nullptr) != CL_SUCCESS) {
    std::fprintf(stderr, "[lmhead-fp32w] logits read failed\n");
    return false;
  }
  static int announced = 0;
  if (announced < 3) {
    ++announced;
    std::fprintf(stderr,
                 "[lmhead-fp32w] call#%d V=%u H=%u (fp32 weight GEMV)\n",
                 announced, vocab, hidden);
  }
  return true;
}

void v8c_collect_lazy_program_tasks(ClContext &cc,
                                    std::vector<std::function<void()>> &out) {
  // Deadlock: this runs inside ClContext bring-up, so nothing here -- neither
  // this function nor the tasks it produces -- may reach ClContext::Global().
  // v8c_use_buffer_path() does, and calling it from here re-enters the
  // context's one-time initialization and waits on itself. Derive the same
  // decision from the same inputs, using the caps of the context being
  // brought up.
  const bool buf_path = [&cc]() {
    if (const char *e = std::getenv("NNTR_V8C_BUF"))
      return std::atoi(e) != 0;  // explicit override (set wins)
    return !cc.caps().image_v8c; // Intel => buffer
  }();

  // The v8c GEMM/activation-quantization program, with the options its own
  // dispatch passes on this device. It is the largest source in the set, so
  // it is collected first.
  const std::string v8c_copts = buf_path ? kV8cBufCompileOpts : "";
  out.push_back([&cc, v8c_copts]() {
    cc.registerClKernel(int8_int4_gemm_v8c_kernel, "v8c_act_quant_f16_par",
                        v8c_copts);
  });

  // The untied lm_head int4 GEMV program. It is otherwise built on the first
  // decode step -- past the first token, but still a stall a user sees.
  out.push_back([&cc]() {
    cc.registerClKernel(lmhead_int4_v8c_kernel, "lmhead_int4_v8c_gemv");
  });
}

void cl_queue_finish() {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (blas_cc)
    opencl::clFinish(blas_cc->command_queue_inst_.GetCommandQueue());
}

void cl_svm_unmap_force(void *ptr) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (blas_cc && ptr)
    blas_cc->command_queue_inst_.enqueueSVMUnmap(ptr);
}

} // namespace nntrainer
