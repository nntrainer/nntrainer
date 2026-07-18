// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file	clblast_interface.cpp
 * @date	12 May 2025
 * @brief	CLBlast library interface
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <stdexcept>

#ifdef ENABLE_CLBLAST
#define CL_EXT_SUFFIX__VERSION_2_0_DEPRECATED // to disable deprecation warnings
#include "clblast.h"
#endif
#include "clblast_interface.h"

#include "opencl_loader.h"

namespace nntrainer {

#ifndef ENABLE_CLBLAST
// CLBlast was excluded from this build (-Denable-clblast=false). All
// general-purpose BLAS-on-OpenCL routes throw if invoked. The paper-aligned
// v8c int8/int4 GPU path does not depend on these routines — it has its own
// dedicated kernels (gemm_int8_v8c_cl + quantize_act_v8c_*_cl).
#define NNTR_CLBLAST_STUB(NAME)                                                \
  do {                                                                         \
    throw std::runtime_error(                                                  \
      "nntrainer was built with -Denable-clblast=false: " NAME " is "          \
      "unavailable. Re-build with -Denable-clblast=true to use the "           \
      "CLBlast-backed OpenCL BLAS routes, or switch the caller to a "          \
      "non-CLBlast GPU path.");                                                \
  } while (0)
#endif

#ifdef ENABLE_CLBLAST
void scal_cl(const unsigned int N, const float alpha, float *X,
             unsigned int incX) {
  auto *clblast_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clBuffManagerInst = ClBufferManager::Global();
  auto command_queue = clblast_cc->command_queue_inst_.GetCommandQueue();

  clBuffManagerInst.getOutBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Scal<float>(N, alpha, clBuffManagerInst.getOutBufferA()->GetBuffer(),
                       0, incX, &command_queue);

  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);
}

void copy_cl(const unsigned int N, const float *X, float *Y, unsigned int incX,
             unsigned int incY) {
  auto *clblast_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clBuffManagerInst = ClBufferManager::Global();
  auto command_queue = clblast_cc->command_queue_inst_.GetCommandQueue();

  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Copy<float>(N, clBuffManagerInst.getInBufferA()->GetBuffer(), 0,
                       incX, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                       incY, &command_queue);

  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), Y);
}

void axpy_cl(const unsigned int N, const float alpha, const float *X, float *Y,
             unsigned int incX, unsigned int incY) {
  auto *clblast_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clBuffManagerInst = ClBufferManager::Global();
  auto command_queue = clblast_cc->command_queue_inst_.GetCommandQueue();

  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clBuffManagerInst.getOutBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), Y);

  clblast::Axpy<float>(N, alpha, clBuffManagerInst.getInBufferA()->GetBuffer(),
                       0, incX, clBuffManagerInst.getOutBufferA()->GetBuffer(),
                       0, incY, &command_queue);

  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), Y);
}

float dot_cl(const unsigned int N, const float *X, const float *Y,
             unsigned int incX, unsigned int incY) {
  auto *clblast_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clBuffManagerInst = ClBufferManager::Global();
  auto command_queue = clblast_cc->command_queue_inst_.GetCommandQueue();

  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clBuffManagerInst.getInBufferB()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), Y);

  clblast::Dot<float>(N, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                      clBuffManagerInst.getInBufferA()->GetBuffer(), 0, incX,
                      clBuffManagerInst.getInBufferB()->GetBuffer(), 0, incY,
                      &command_queue);

  float result;
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, sizeof(float), &result);
  return result;
}

float nrm2_cl(const unsigned int N, const float *X, unsigned int incX) {
  auto *clblast_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clBuffManagerInst = ClBufferManager::Global();
  auto command_queue = clblast_cc->command_queue_inst_.GetCommandQueue();

  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Nrm2<float>(N, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                       clBuffManagerInst.getInBufferA()->GetBuffer(), 0, incX,
                       &command_queue);

  float result;
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, sizeof(float), &result);

  return result;
}

float asum_cl(const unsigned int N, const float *X, unsigned int incX) {
  auto *clblast_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clBuffManagerInst = ClBufferManager::Global();
  auto command_queue = clblast_cc->command_queue_inst_.GetCommandQueue();

  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Asum<float>(N, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                       clBuffManagerInst.getInBufferA()->GetBuffer(), 0, incX,
                       &command_queue);

  float result;
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, sizeof(float), &result);

  return result;
}

int amax_cl(const unsigned int N, const float *X, unsigned int incX) {
  auto *clblast_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clBuffManagerInst = ClBufferManager::Global();
  auto command_queue = clblast_cc->command_queue_inst_.GetCommandQueue();

  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Amax<float>(N, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                       clBuffManagerInst.getInBufferA()->GetBuffer(), 0, incX,
                       &command_queue);

  int result;
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, sizeof(int), &result);

  return result;
}

int amin_cl(const unsigned int N, const float *X, unsigned int incX) {
  auto *clblast_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clBuffManagerInst = ClBufferManager::Global();
  auto command_queue = clblast_cc->command_queue_inst_.GetCommandQueue();

  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Amin<float>(N, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                       clBuffManagerInst.getInBufferA()->GetBuffer(), 0, incX,
                       &command_queue);

  int result;
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, sizeof(int), &result);

  return result;
}

void gemv_cl(const unsigned int layout, bool TransA, const unsigned int M,
             const unsigned int N, const float alpha, const float *A,
             const unsigned int lda, const float *X, const float beta, float *Y,
             unsigned int incX, unsigned int incY) {
  auto *clblast_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clBuffManagerInst = ClBufferManager::Global();
  auto command_queue = clblast_cc->command_queue_inst_.GetCommandQueue();

  clblast::Transpose transA =
    (TransA) ? clblast::Transpose::kYes : clblast::Transpose::kNo;

  const size_t a_size = M * N;
  const size_t x_size = (TransA ? M : N) * incX;
  const size_t y_size = (TransA ? N : M) * incY;

  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, a_size * sizeof(float), A);

  clBuffManagerInst.getOutBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, y_size * sizeof(float), Y);

  clBuffManagerInst.getInBufferB()->WriteDataRegion(
    clblast_cc->command_queue_inst_, x_size * sizeof(float), X);

  auto s = clblast::Gemv<float>(
    clblast::Layout::kRowMajor, transA, M, N, alpha,
    clBuffManagerInst.getInBufferA()->GetBuffer(), 0, lda,
    clBuffManagerInst.getInBufferB()->GetBuffer(), 0, incX, beta,
    clBuffManagerInst.getOutBufferA()->GetBuffer(), 0, incY, &command_queue);

  opencl::clFinish(clblast_cc->command_queue_inst_.GetCommandQueue());

  opencl::clEnqueueReadBuffer(clblast_cc->command_queue_inst_.GetCommandQueue(),
                              clBuffManagerInst.getOutBufferA()->GetBuffer(),
                              CL_TRUE, 0, y_size * sizeof(float), Y, 0, nullptr,
                              nullptr);
}

void gemm_cl(const unsigned int layout, bool TransA, bool TransB,
             const unsigned int M, const unsigned int N, const unsigned int K,
             const float alpha, const float *A, const unsigned int lda,
             const float *B, const unsigned int ldb, const float beta, float *C,
             const unsigned int ldc) {
  auto *clblast_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clBuffManagerInst = ClBufferManager::Global();
  auto command_queue = clblast_cc->command_queue_inst_.GetCommandQueue();

  clblast::Transpose transA =
    (TransA) ? clblast::Transpose::kYes : clblast::Transpose::kNo;
  clblast::Transpose transB =
    (TransB) ? clblast::Transpose::kYes : clblast::Transpose::kNo;

  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, M * K * sizeof(float), A);

  clBuffManagerInst.getInBufferB()->WriteDataRegion(
    clblast_cc->command_queue_inst_, K * N * sizeof(float), B);

  clBuffManagerInst.getOutBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, M * N * sizeof(float), C);

  // layout is currently fixed to RowMajor
  clblast::Gemm<float>(
    clblast::Layout::kRowMajor, transA, transB, M, N, K, alpha,
    clBuffManagerInst.getInBufferA()->GetBuffer(), 0, lda,
    clBuffManagerInst.getInBufferB()->GetBuffer(), 0, ldb, beta,
    clBuffManagerInst.getOutBufferA()->GetBuffer(), 0, ldc, &command_queue);

  // Read the result back to C
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, M * N * sizeof(float), C);
}

void gemm_batched_cl(const unsigned int layout, bool TransA, bool TransB,
                     const unsigned int M, const unsigned int N,
                     const unsigned int K, const float *alpha, const float *A,
                     const unsigned int lda, const float *B,
                     const unsigned int ldb, const float *beta, float *C,
                     const unsigned int ldc, const unsigned int batch_size) {
  throw std::runtime_error("gemm_batched_cl is not implemented");
}

void im2col_cl(const unsigned int C, const unsigned int H, const unsigned int W,
               const unsigned int kernel_h, const unsigned int kernel_w,
               const unsigned int pad_h, const unsigned int pad_w,
               const unsigned int stride_h, const unsigned int stride_w,
               const unsigned int dilation_h, const unsigned int dilation_w,
               const float *input, float *output) {
  throw std::runtime_error("im2col_cl is not implemented");
}
#else  // ENABLE_CLBLAST

void scal_cl(const unsigned int, const float, float *, unsigned int) {
  NNTR_CLBLAST_STUB("scal_cl");
}
void copy_cl(const unsigned int, const float *, float *, unsigned int,
             unsigned int) {
  NNTR_CLBLAST_STUB("copy_cl");
}
void axpy_cl(const unsigned int, const float, const float *, float *,
             unsigned int, unsigned int) {
  NNTR_CLBLAST_STUB("axpy_cl");
}
float dot_cl(const unsigned int, const float *, const float *, unsigned int,
             unsigned int) {
  NNTR_CLBLAST_STUB("dot_cl");
}
float nrm2_cl(const unsigned int, const float *, unsigned int) {
  NNTR_CLBLAST_STUB("nrm2_cl");
}
float asum_cl(const unsigned int, const float *, unsigned int) {
  NNTR_CLBLAST_STUB("asum_cl");
}
int amax_cl(const unsigned int, const float *, unsigned int) {
  NNTR_CLBLAST_STUB("amax_cl");
}
int amin_cl(const unsigned int, const float *, unsigned int) {
  NNTR_CLBLAST_STUB("amin_cl");
}
void gemv_cl(const unsigned int, bool, const unsigned int, const unsigned int,
             const float, const float *, const unsigned int, const float *,
             const float, float *, unsigned int, unsigned int) {
  NNTR_CLBLAST_STUB("gemv_cl");
}
void gemm_cl(const unsigned int, bool, bool, const unsigned int,
             const unsigned int, const unsigned int, const float, const float *,
             const unsigned int, const float *, const unsigned int, const float,
             float *, const unsigned int) {
  NNTR_CLBLAST_STUB("gemm_cl");
}
void gemm_batched_cl(const unsigned int, bool, bool, const unsigned int,
                     const unsigned int, const unsigned int, const float *,
                     const float *, const unsigned int, const float *,
                     const unsigned int, const float *, float *,
                     const unsigned int, const unsigned int) {
  NNTR_CLBLAST_STUB("gemm_batched_cl");
}
void im2col_cl(const unsigned int, const unsigned int, const unsigned int,
               const unsigned int, const unsigned int, const unsigned int,
               const unsigned int, const unsigned int, const unsigned int,
               const unsigned int, const unsigned int, const float *, float *) {
  NNTR_CLBLAST_STUB("im2col_cl");
}
#endif // ENABLE_CLBLAST

} // namespace nntrainer
