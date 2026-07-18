// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 *
 * @file	blas_kernels_templates.hpp
 * @date	07 July 2025
 * @brief	Common blas OpenCL kernels (common templates used by
 * blas_kernels_fp16.cpp and blas_kernels.cpp)
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @author	Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNELS_TEMPLATES_H__
#define __BLAS_KERNELS_TEMPLATES_H__

#include <blas_kernels.h>

namespace nntrainer {
template <typename T = float>
inline static void sgemv_cl_internal(ClContext::SharedPtrClKernel kernel,
                                     const T *matAdata, const T *vecXdata,
                                     T *vecYdata, unsigned int dim1,
                                     unsigned int dim2, unsigned int lda) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t dim1_size = sizeof(T) * dim1;
  size_t dim2_size = sizeof(T) * dim2;
  size_t dim1_dim2_size = sizeof(T) * dim1 * dim2;

  result = clbuffInstance.getInBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim1_dim2_size, matAdata);
  if (!result) {
    return;
  }

  result = clbuffInstance.getInBufferB()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim2_size, vecXdata);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim1_size, vecYdata);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, clbuffInstance.getInBufferB(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(2, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(3, &dim2, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(4, &lda, sizeof(int));
  if (!result) {
    return;
  }

  const int work_groups_count[3] = {(int)dim1, 1, 1};
  const int work_group_size[3] = {1, 1, 1};

  result = opencl::CommandQueueManager::Global().DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, dim1_size, vecYdata);
  if (!result) {
    return;
  }
}

template <typename T = float>
T dot_cl_internal(ClContext::SharedPtrClKernel kernel, const T *vecAdata,
                  const T *vecXdata, unsigned int dim1) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  T cl_ret = 0;

  do {
    size_t dim1_size = sizeof(T) * dim1;

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecXdata);
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(1, clbuffInstance.getInBufferB(),
                                        sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(3, clbuffInstance.getOutBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {1, 1, 1};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, sizeof(T), &cl_ret);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}

template <typename T = float>
inline static void
sgemm_cl_internal(ClContext::SharedPtrClKernel kernel, bool TransA, bool TransB,
                  const T *A, const T *B, T *C, unsigned int M, unsigned int N,
                  unsigned int K, unsigned int lda, unsigned int ldb,
                  unsigned int ldc) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  // sizes will be same for transpose
  size_t m_k_size = M * K * sizeof(T);
  size_t k_n_size = K * N * sizeof(T);
  size_t m_n_size = M * N * sizeof(T);

  result = clbuffInstance.getInBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, m_k_size, A);
  if (!result) {
    return;
  }

  result = clbuffInstance.getInBufferB()->WriteDataRegion(
    blas_cc->command_queue_inst_, k_n_size, B);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, m_n_size, C);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, clbuffInstance.getInBufferB(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(2, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(3, &M, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(4, &N, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(5, &K, sizeof(int));
  if (!result) {
    return;
  }

  const int tiled_size = 16;
  const int work_groups_count[3] = {
    (int)((N + tiled_size - 1) / tiled_size) * tiled_size,
    (int)((M + tiled_size - 1) / tiled_size) * tiled_size, 1}; // test-value

  const int work_group_size[3] = {tiled_size, tiled_size, 1}; // test-value

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, m_n_size, C);
  if (!result) {
    return;
  }
}

template <typename T = float>
inline static void
addition_cl_internal(ClContext::SharedPtrClKernel kernel, const T *input,
                     T *res, unsigned int size_input, unsigned int size_res,
                     bool use_svm = false) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t dim1_size = sizeof(T) * size_input;
  size_t dim2_size = sizeof(T) * size_res;

  if (use_svm) {
    // SVM-direct: input/res are GPU-resident pointers; accumulate in place
    // (res += input) with no host round-trip (residency path). Coarse-grained
    // SVM needs explicit coherence: release host mappings so the GPU sees the
    // current host-side contents (res was just host-written by the copy of the
    // first addend), then re-map res after the dispatch for the host read.
    blas_cc->command_queue_inst_.enqueueSVMUnmap(const_cast<T *>(input));
    blas_cc->command_queue_inst_.enqueueSVMUnmap(res);
    if (!kernel->SetKernelSVMArguments(0, input))
      return;
    if (!kernel->SetKernelSVMArguments(1, res))
      return;
  } else {
    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, input);
    if (!result) {
      return;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim2_size, res);
    if (!result) {
      return;
    }

    result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      return;
    }

    result = kernel->SetKernelArguments(1, clbuffInstance.getOutBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      return;
    }
  }

  result = kernel->SetKernelArguments(2, &size_input, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(3, &size_res, sizeof(int));
  if (!result) {
    return;
  }

  // lws was {1,1,1} ("test-value") -> one work-item per work-group, i.e. ~1/64
  // of the SIMD wave used (measured: addition_cl_fp16 at ~1.5 GB/s). Use a
  // full work-group; round the global size up to a multiple of lws -- both
  // addition_cl / addition_cl_fp16 guard `if (idx < size_res)` so the
  // rounded-up tail work-items are no-ops.
  const int add_lws = 64;
  const int add_gws = (((int)size_res + add_lws - 1) / add_lws) * add_lws;
  const int work_groups_count[3] = {add_gws, 1, 1};
  const int work_group_size[3] = {add_lws, 1, 1};
  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  if (!use_svm) {
    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, dim2_size, res);

    if (!result) {
      return;
    }
  } else {
    // re-map the in-place result so the host sees the GPU-written values.
    // Kept BLOCKING: an async map here measured faster but corrupted output
    // (coherence race) — removing the pair entirely needs a GPU-resident
    // producer+consumer, not an async flip.
    blas_cc->command_queue_inst_.enqueueSVMMap(res, dim2_size, true);
  }
}

template <typename T = float>
inline static void rmsnorm_cl_internal(ClContext::SharedPtrClKernel kernel,
                                       const T *input, const T *gamma,
                                       T *result, const T epsilon,
                                       unsigned int height, unsigned int width,
                                       const bool use_svm = true) {
  unsigned dim_in = height * width;
  unsigned dim_gamma = width;
  unsigned size_in = dim_in * sizeof(T);
  unsigned size_gamma = dim_gamma * sizeof(T);

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  if (use_svm) {
    if (!kernel->SetKernelSVMArguments(0, input)) {
      return;
    }
    if (!kernel->SetKernelSVMArguments(1, result)) {
      return;
    }
    if (!kernel->SetKernelSVMArguments(2, gamma)) {
      return;
    }
  } else {
    auto &clbuffInstance = ClBufferManager::Global();
    if (!clbuffInstance.getInBufferA()->WriteDataRegion(
          blas_cc->command_queue_inst_, size_in, input)) {
      return;
    }
    if (!clbuffInstance.getInBufferB()->WriteDataRegion(
          blas_cc->command_queue_inst_, size_gamma, gamma)) {
      return;
    }

    if (!kernel->SetKernelArguments(
          0, &clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem))) {
      return;
    }
    if (!kernel->SetKernelArguments(
          1, &clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem))) {
      return;
    }
    if (!kernel->SetKernelArguments(
          2, &clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem))) {
      return;
    }
  }

  if (!kernel->SetKernelArguments(3, &epsilon, sizeof(float))) {
    return;
  }
  if (!kernel->SetKernelArguments(4, &height, sizeof(int))) {
    return;
  }
  if (!kernel->SetKernelArguments(5, &width, sizeof(int))) {
    return;
  }
#ifdef __ANDROID__
  constexpr int SUBGROUP_SIZE = 64;
#else
  constexpr int SUBGROUP_SIZE = 32;
#endif
  const int work_groups_count[3] = {static_cast<int>(height) * SUBGROUP_SIZE, 1,
                                    1};

  const int work_group_size[3] = {SUBGROUP_SIZE, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kernel, work_groups_count,
                                                    work_group_size)) {
    return;
  }

  if (!use_svm) {
    auto &clbuffInstance = ClBufferManager::Global();
    if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
          blas_cc->command_queue_inst_, size_in, result)) {
      return;
    }
  } else {
    blas_cc->command_queue_inst_.enqueueSVMMap(result, size_in, false);
  }
}

template <typename T = float>
inline static void sscal_cl_internal(ClContext::SharedPtrClKernel kernel, T *X,
                                     const unsigned int N, const float alpha) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t x_size = N * sizeof(T);

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, x_size, X);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, &alpha, sizeof(float));
  if (!result) {
    return;
  }

  const int work_groups_count[3] = {(int)N, 1, 1};
  const int work_group_size[3] = {1, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, x_size, X);
  if (!result) {
    return;
  }
}

template <typename T = float>
inline static void transpose_cl_axis_internal(
  ClContext::SharedPtrClKernel kernel, const T *in, T *res,
  unsigned int input_batch_size, unsigned int input_channels,
  unsigned int input_height, unsigned int input_width, unsigned int axis) {

  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t dim_size =
    sizeof(T) * input_batch_size * input_height * input_width * input_channels;

  result = clbuffInstance.getInBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim_size, in);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim_size, res);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(2, &input_batch_size, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(3, &input_channels, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(4, &input_height, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(5, &input_width, sizeof(int));
  if (!result) {
    return;
  }

  int work_groups_count[3] = {(int)input_height, (int)input_width, 1};
  if (axis == 2)
    work_groups_count[0] = (int)input_channels;

  const int work_group_size[3] = {1, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, dim_size, res);
  if (!result) {
    return;
  }
}

} // namespace nntrainer

#endif
