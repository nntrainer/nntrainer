// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels.cpp
 * @date	14 May 2024
 * @brief	Common blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "blas_kernels_templates.h"
#include <cl_kernels/cl_kernels.h>

#include "util_func.h"
#include <fp16.h>

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
  std::string compile_options =
    " -D SIZE_QUANTIZATION_GROUP=" + std::to_string(quantization_group_size) +
    " -D SCALE_ROW_MAJOR=" + std::to_string(scale_row_major);

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_kernel, "fully_connected_gpu_int4_gemv", compile_options);
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
  std::string compile_options =
    " -D SIZE_QUANTIZATION_GROUP=" + std::to_string(quantization_group_size) +
    " -D SCALE_ROW_MAJOR=" + std::to_string(scale_row_major);

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_kernel, "fully_connected_gpu_int4_gemv", compile_options);
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

void openvino_gemm_async_cl(float *input, std::vector<void *> weights,
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
    std::string compile_options =
      " -D SIZE_N=" + std::to_string(Ns[0]) +
      " -D SIZE_K=" + std::to_string(K) +
      " -D SIZE_QUANTIZATION_GROUP=" + std::to_string(quantization_group_size);

    ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
      int4_quantize_input_kernel, "quantize_input_int4_pad", compile_options);
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
    std::string compile_options =
      " -D SIZE_N=" + std::to_string(N) + " -D SIZE_K=" + std::to_string(K) +
      " -D SIZE_QUANTIZATION_GROUP=" + std::to_string(quantization_group_size) +
      " -D SCALE_ROW_MAJOR=" + std::to_string(scale_row_major);

    ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
      openvino_gemm_kernel, "fc_bf_tiled_kernel_default", compile_options);
    if (!kernel_ptr) {
      throw std::runtime_error(
        "Failed to get kernel_ptr for fc_bf_tiled_kernel_default");
      return;
    }

    int arg = 0;

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
void openvino_sgemm_cl(float *input, char *weight, uint16_t *scale,
                       float *output, unsigned int M, unsigned int N,
                       unsigned int K, unsigned int quantization_group_size) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  // copy fp32 input to fp16
  copy_fp32_u16(M * K, input, (uint16_t *)clbuffInstance.getSVMInput());

  // perform int4 matmul
  openvino_gemm_cl(clbuffInstance.getSVMInput(), weight, scale,
                   clbuffInstance.getSVMOutput(), M, N, K,
                   quantization_group_size);

  // copy fp16 output to fp32
  copy_u16_fp32(M * N, (uint16_t *)clbuffInstance.getSVMOutput(), output);
}

void openvino_gemm_cl(void *input, void *weights, void *scales, void *output,
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
  std::string compile_options =
    " -D SIZE_N=" + std::to_string(N) + " -D SIZE_K=" + std::to_string(K) +
    " -D SIZE_QUANTIZATION_GROUP=" + std::to_string(quantization_group_size) +
    " -D SCALE_ROW_MAJOR=" + std::to_string(scale_row_major);

  std::vector<cl_event> quantize_event(1);
  {
    ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
      int4_quantize_input_kernel, "quantize_input_int4_pad", compile_options);
    if (!kernel_ptr) {
      throw std::runtime_error("Failed to get kernel_ptr for quantize_input");
      return;
    }

    int arg = 0;

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

    std::array<size_t, 3> global_work_size = {
      (M * alignK) / quantization_group_size, 1, 1};

    blas_cc->command_queue_inst_.enqueueKernel(
      kernel_ptr->GetKernel(), global_work_size.size(), global_work_size.data(),
      nullptr, 0, nullptr, &quantize_event.front());
  }

  // 3. Perform Matrix Multiplication
  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    openvino_gemm_kernel, "fc_bf_tiled_kernel_default", compile_options);
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for fc_bf_tiled_kernel_default");
    return;
  }

  int arg = 0;

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
              unsigned int lda) {
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
                           dim2, lda);
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
              unsigned int lda, unsigned int ldb, unsigned int ldc) {
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
                           lda, ldb, ldc);
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
void openvino_quantize_input_int4_pad(void *input, void *quantized_input, void *scales,
                                      unsigned int M, unsigned int K,
                                      unsigned int quantization_group_size) {
  int alignK = align(K, quantization_group_size);

  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();
  const bool scale_row_major = false;
  std::string compile_options =
    " -D SIZE_N=" + std::to_string(M) + " -D SIZE_K=" + std::to_string(K) +
    " -D SIZE_QUANTIZATION_GROUP=" + std::to_string(quantization_group_size) +
    " -D SCALE_ROW_MAJOR=" + std::to_string(scale_row_major);

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_quantize_input_kernel, "quantize_input_int4_pad", compile_options);
  if (!kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for quantize_input");
    return;
  }

  int arg = 0;

  result = kernel_ptr->SetKernelSVMArguments(arg++, input);

  if (!result)
    throw std::runtime_error("Failed to set kernel argument 0 for "
                             "quantize_input");

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, quantized_input);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 1 for "
                             "quantize_input");

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, scales);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 2 for "
                             "quantize_input");

  std::array<size_t, 3> global_work_size = {
    (M * alignK) / quantization_group_size, 1, 1};

  blas_cc->command_queue_inst_.enqueueKernel(
    kernel_ptr->GetKernel(), global_work_size.size(), global_work_size.data(),
    nullptr, 0, nullptr, nullptr);
}

} // namespace nntrainer
