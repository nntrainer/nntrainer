// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	unittest_opencl_kernels_blas.cpp
 * @date	6 June 2024
 * @brief	Test setup for blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

#include "fallback_internal.h"
#include "gelu_cl_op.h"
#include "int4_utils.h"
#include "layernorm_cl_op.h"
#include "nntrainer_test_util.h"
#include "q4_0_utils.h"
#include "swiglu_cl_op.h"
#include "tensor_dim.h"
#include "timer.h"
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <cpu_backend.h>
#include <fp16.h>
#include <layer_context.h>
#include <tensor.h>

using namespace nntrainer;

// -----
// Functions
// -----

static std::pair<float, float>
dotCL_sgemv_test_func(const int batch, const int channel, const int height,
                      const int width, const int height_b, const int width_b,
                      const bool transA, const bool transB, const float alpha,
                      const int MOD) {
  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  return {mseError, static_cast<float>(cosSim)};
}

#ifdef ENABLE_FP16
static std::pair<float, float> dotCL_sgemv_fp16_test_func(
  const int batch, const int channel, const int height, const int width,
  const int height_b, const int width_b, const bool transA, const bool transB,
  const float alpha, const int MOD) {
  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp16, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp16, B_fp16, transA, transB);
  nntrainer::Tensor C_fp16 = A_fp16.dot(B_fp16, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  return {mseError, static_cast<float>(cosSim)};
}

static std::pair<float, float>
dot_gemm_fp16(const int batch, const int channel, const int height,
              const int width, const int height_b, const int width_b,
              const bool transA, const bool transB, const float alpha,
              const int MOD) {
  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);

  nntrainer::Tensor C = dotCl(A, B, transA, transB);
  nntrainer::Tensor C_fp16 = A.dot(B, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  return {mseError, static_cast<float>(cosSim)};
}
#endif

TEST(blas_kernels, dotCL_sgemv_M_1_1) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_M_1_2) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_1) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 768;
  const int width_b = 1;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_2) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 2048;
  const int width_b = 1;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_n) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  EXPECT_THROW(dotCl(A_fp32, B_fp32, transA, transB), std::runtime_error);
}

TEST(blas_kernels, dotCL_sgemv_N_1_M_1_1) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 1;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  auto gen_data = [](nntrainer::Tensor x) {
    auto ptr = x.getData();
    for (unsigned int i = 0; i < x.size(); ++i) {
      ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  };

  gen_data(A_fp32), gen_data(B_fp32);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float err = mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-5 * width;

  EXPECT_IN_RANGE(err, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_M_1_2) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 1;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-5 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_noTrans) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transB) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_transA) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transAB) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = true;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, addition_i) {
  const int batch = 1;
  const int channel = 1;
  const int height = 64;
  const int width = 3072;

  const int batch_b = 1;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch_b, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor C_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor D_fp32(batch_b, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(B_fp32, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);
  GEN_TEST_INPUT(C_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(D_fp32, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);

  A_fp32.add_i(B_fp32);
  add_i_cl(C_fp32, D_fp32);

  float mseError =
    mse<float>(A_fp32.getData<float>(), C_fp32.getData<float>(), A_fp32.size());

  double cosSim = cosine_similarity<float>(
    A_fp32.getData<float>(), C_fp32.getData<float>(), A_fp32.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, l2norm) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 768;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  GEN_TEST_INPUT(B_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  float gpu_result = nrm2Cl(A_fp32);
  float cpu_result = B_fp32.l2norm();

  EXPECT_FLOAT_EQ(gpu_result, cpu_result);
}

TEST(blas_kernels, absolute_sum) {
  const int batch = 1;
  const int channel = 1;
  const int height = 32;
  const int width = 32;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  float cpu_result = 0.0f;
  for (size_t i = 0; i < A_fp32.size(); ++i) {
    cpu_result += fabs(A_fp32.getData<float>()[i]);
  }

  float gpu_result = asumCl(A_fp32);

  EXPECT_FLOAT_EQ(cpu_result, gpu_result);
}

TEST(blas_kernels, rmsnorm_fp32) {
  const int batch = 1;
  const int channel = 1;
  const int height = 67;
  const int width = 3072;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor in_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width, t_type_nchw_fp32);
  nntrainer::Tensor out_cl_fp32(batch, channel, height, width,
                                t_type_nchw_fp32);
  nntrainer::Tensor out_ref_fp32(batch, channel, height, width,
                                 t_type_nchw_fp32);

  /// Initialize CPU input data
  GEN_TEST_INPUT(in_fp32, ((i * (batch * height * channel) +
                            j * (batch * height) + k * (width) + l + 1) %
                           MOD) *
                            alpha);
  for (int l = 0; l < width; ++l) {
    float val = ((l + 1) % MOD) * alpha;
    gamma_fp32.setValue(0, 0, 0, l, val);
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  /// Initialize GPU input/ouput data
  void *in_fp32_svm = allocateSVM(in_fp32.size() * sizeof(float));
  void *gamma_fp32_svm = allocateSVM(gamma_fp32.size() * sizeof(float));
  void *out_fp32_svm = allocateSVM(out_cl_fp32.size() * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(
    in_fp32_svm, in_fp32.size() * sizeof(float), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(
    gamma_fp32_svm, gamma_fp32.size() * sizeof(float), false);

  std::memcpy(in_fp32_svm, in_fp32.getData<float>(),
              in_fp32.size() * sizeof(float));
  std::memcpy(gamma_fp32_svm, gamma_fp32.getData<float>(),
              gamma_fp32.size() * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMUnmap(in_fp32_svm);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(gamma_fp32_svm);

  static constexpr uint32_t run_count = 500;
  static constexpr float kEpsilon = 0.001f;

  Timer timer1{};
  for (unsigned int i = 0; i < run_count; ++i) {
    rmsnorm_cl((float *)in_fp32_svm, (float *)gamma_fp32_svm,
               (float *)out_fp32_svm, kEpsilon,
               in_fp32.batch() * in_fp32.channel() * in_fp32.height(),
               in_fp32.width(), true);
  }
  auto t2_cl = timer1.GetElapsedMilliseconds();

  Timer timer2{};
  for (unsigned int i = 0; i < run_count; ++i) {
    std::function<float(float)> f = [](float x) { return 1 / std::sqrt(x); };
    auto t = in_fp32.multiply(in_fp32).average(3).add(kEpsilon);
    t.apply_i(f);
    in_fp32.multiply(t, out_ref_fp32);
    out_ref_fp32.multiply_i(gamma_fp32);
  }
  auto t2_ref = timer2.GetElapsedMilliseconds();

  std::cout << "RMSNorm time : GPU = " << t2_cl / (run_count * 1.0f) << " ms"
            << std::endl;

  std::cout << "RMSNorm time : CPU = " << t2_ref / (run_count * 1.0f) << " ms"
            << std::endl;

  blas_cc->command_queue_inst_.enqueueSVMMap(
    out_fp32_svm, out_cl_fp32.size() * sizeof(float), false);

  float mseError = mse<float>((float *)out_fp32_svm,
                              out_ref_fp32.getData<float>(), height * width);

  double cosSim = cosine_similarity<float>(
    (float *)out_fp32_svm, out_ref_fp32.getData<float>(), height * width);

  const float epsilon = 1e-3 * width;

  freeSVM(out_fp32_svm);
  freeSVM(in_fp32_svm);
  freeSVM(gamma_fp32_svm);

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

// LayerNorm / GELU whole-ops. These go through the op entry points
// (layernorm_cl_op / gelu_cl_op) rather than the former blas_kernels wrappers,
// i.e. exactly the symbols ClComputeOps::layer_norm / ::activation forward to,
// so they cover the argument binding AND the kernel math. Tensors here come
// from the default (host) pool, so the ops take their host-bounce branch --
// the SVM-direct branch is the same kernel with the pointers bound directly.
TEST(blas_kernels, layernorm_fp32) {
  // Small, fully-deterministic LayerNorm on the GPU checked element-wise
  // against a plain CPU reference LayerNorm.
  const unsigned int height = 3;
  const unsigned int width = 8;
  const float kEpsilon = 0.001f;

  nntrainer::init_backend();
  (void)nntrainer::Engine::Global().getRegisteredContext("gpu");

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                             nntrainer::Tdatatype::FP32};
  nntrainer::Tensor in(1, 1, height, width, t_fp32);
  nntrainer::Tensor out(1, 1, height, width, t_fp32);
  nntrainer::Tensor gamma(1, 1, 1, width, t_fp32);
  nntrainer::Tensor beta(1, 1, 1, width, t_fp32);

  // Deterministic, non-trivial input/gamma/beta (rows have different means so
  // the mean-subtraction path is actually exercised).
  float *ip = in.getData<float>();
  for (unsigned int h = 0; h < height; ++h)
    for (unsigned int w = 0; w < width; ++w)
      ip[h * width + w] = ((h * width + w) % 7) * 0.5f - 1.0f;
  for (unsigned int w = 0; w < width; ++w) {
    gamma.getData<float>()[w] = 0.5f + 0.1f * w;
    beta.getData<float>()[w] = -0.3f + 0.05f * w;
  }

  nntrainer::layernorm_cl_op(in, out, gamma, beta, kEpsilon, height,
                             /*row_offset=*/0);

  /// CPU reference LayerNorm (mean, var, (x-mean)/sqrt(var+eps)*gamma + beta)
  float maxAbsErr = 0.0f;
  for (unsigned int h = 0; h < height; ++h) {
    double mean = 0.0;
    for (unsigned int w = 0; w < width; ++w)
      mean += ip[h * width + w];
    mean /= width;
    double var = 0.0;
    for (unsigned int w = 0; w < width; ++w) {
      double d = ip[h * width + w] - mean;
      var += d * d;
    }
    var /= width;
    double inv = 1.0 / std::sqrt(var + (double)kEpsilon);
    for (unsigned int w = 0; w < width; ++w) {
      double ref =
        (ip[h * width + w] - mean) * inv * gamma.getData<float>()[w] +
        beta.getData<float>()[w];
      float gpu = out.getData<float>()[h * width + w];
      float err = std::fabs(gpu - (float)ref);
      if (err > maxAbsErr)
        maxAbsErr = err;
    }
  }

  std::cout << "LayerNorm max abs error : " << maxAbsErr << std::endl;

  EXPECT_LT(maxAbsErr, 1e-4f);
}

TEST(blas_kernels, layernorm_fp32_row_window) {
  // The (active_rows, row_offset) window must leave the rows outside it
  // untouched -- this is the decode/step contract the neutral Layer relies on.
  const unsigned int height = 4;
  const unsigned int width = 8;
  const float kEpsilon = 0.001f;

  nntrainer::init_backend();
  (void)nntrainer::Engine::Global().getRegisteredContext("gpu");

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                             nntrainer::Tdatatype::FP32};
  nntrainer::Tensor in(1, 1, height, width, t_fp32);
  nntrainer::Tensor out(1, 1, height, width, t_fp32);
  nntrainer::Tensor gamma(1, 1, 1, width, t_fp32);
  nntrainer::Tensor beta(1, 1, 1, width, t_fp32);

  float *ip = in.getData<float>();
  for (unsigned int i = 0; i < height * width; ++i)
    ip[i] = (i % 5) * 0.75f - 1.5f;
  gamma.setValue(1.0f);
  beta.setValue(0.0f);
  out.setValue(-99.0f);

  nntrainer::layernorm_cl_op(in, out, gamma, beta, kEpsilon, /*active_rows=*/2,
                             /*row_offset=*/1);

  for (unsigned int h = 0; h < height; ++h) {
    for (unsigned int w = 0; w < width; ++w) {
      const float got = out.getData<float>()[h * width + w];
      if (h == 0 || h == 3) {
        EXPECT_FLOAT_EQ(got, -99.0f) << "row window leaked at h=" << h;
      } else {
        double mean = 0.0;
        for (unsigned int k = 0; k < width; ++k)
          mean += ip[h * width + k];
        mean /= width;
        double var = 0.0;
        for (unsigned int k = 0; k < width; ++k) {
          double d = ip[h * width + k] - mean;
          var += d * d;
        }
        var /= width;
        const double ref =
          (ip[h * width + w] - mean) / std::sqrt(var + (double)kEpsilon);
        EXPECT_NEAR(got, (float)ref, 1e-4f);
      }
    }
  }
}

TEST(blas_kernels, gelu_fp32) {
  // Fixed FP32 input of 24 values spanning negatives/zero/positives.
  const std::vector<float> host_in = {
    -6.0f, -4.5f,  -3.0f, -2.5f, -2.0f, -1.5f, -1.0f, -0.75f,
    -0.5f, -0.25f, -0.1f, 0.0f,  0.1f,  0.25f, 0.5f,  0.75f,
    1.0f,  1.5f,   2.0f,  2.5f,  3.0f,  4.5f,  6.0f,  8.0f};
  const unsigned int num_elems = (unsigned int)host_in.size();

  nntrainer::init_backend();
  (void)nntrainer::Engine::Global().getRegisteredContext("gpu");

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                             nntrainer::Tdatatype::FP32};
  nntrainer::Tensor in(1, 1, 1, num_elems, t_fp32);
  nntrainer::Tensor out(1, 1, 1, num_elems, t_fp32);
  std::memcpy(in.getData<float>(), host_in.data(), num_elems * sizeof(float));

  // Double-precision CPU reference for both GELU variants.
  auto ref_gelu = [](double x, int mode) -> double {
    if (mode == 1) {
      // tanh approximation (ACT_TANH_GELU)
      const double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
      return 0.5 * x * (1.0 + std::tanh(inner));
    }
    // erf-based exact GELU (ACT_GELU)
    return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752));
  };

  for (int mode = 0; mode <= 1; ++mode) {
    out.setValue(0.0f);
    nntrainer::gelu_cl_op(in, out, mode, /*active_rows=*/1, /*row_offset=*/0);

    float max_abs_err = 0.0f;
    for (unsigned int i = 0; i < num_elems; ++i) {
      const double ref = ref_gelu((double)host_in[i], mode);
      const float got = out.getData<float>()[i];
      const float err = std::fabs((float)(got - ref));
      if (err > max_abs_err)
        max_abs_err = err;
    }

    std::cout << "GELU mode " << mode << " (" << (mode ? "tanh" : "erf")
              << ") max abs error = " << max_abs_err << std::endl;

    EXPECT_LT(max_abs_err, 1e-4f);
  }
}

TEST(blas_kernels, swiglu_layer_fp32_67_3072) {
  const int batch = 1;
  const int channel = 1;
  const int height = 67;
  const int width = 3072;

  const int dim = width * height;

  const int batch_b = 1;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  /// Initialize CPU input/output
  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch_b, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor out_ref_fp32(batch, channel, height, width,
                                 t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(B_fp32, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);

  /// Initialize GPU input/output
  void *gpu_in1 = allocateSVM(dim * sizeof(float));
  void *gpu_in2 = allocateSVM(dim * sizeof(float));
  void *gpu_dst = allocateSVM(dim * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(gpu_in1, dim * sizeof(float),
                                             false);
  blas_cc->command_queue_inst_.enqueueSVMMap(gpu_in2, dim * sizeof(float),
                                             false);

  std::memcpy(gpu_in1, A_fp32.getData(), dim * sizeof(float));
  std::memcpy(gpu_in2, B_fp32.getData(), dim * sizeof(float));

  static constexpr uint32_t run_count = 500;

  auto t1_cl = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::swiglu_cl((float *)gpu_in1, (float *)gpu_in2, (float *)gpu_dst,
                         width, height, true);
  }
  auto t2_cl = std::chrono::high_resolution_clock::now();

  auto t1_ref = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::swiglu(height * width, out_ref_fp32.getData(), A_fp32.getData(),
                      B_fp32.getData());
  }
  auto t2_ref = std::chrono::high_resolution_clock::now();

  auto dt_cl =
    std::chrono::duration_cast<std::chrono::microseconds>(t2_cl - t1_cl);
  auto dt_ref =
    std::chrono::duration_cast<std::chrono::microseconds>(t2_ref - t1_ref);

  std::cout << "Swiglu time : GPU = " << dt_cl.count() / (run_count * 1000.0f)
            << " ms" << std::endl;

  std::cout << "Swiglu time : CPU = " << dt_ref.count() / (run_count * 1000.0f)
            << " ms" << std::endl;

  float mseError =
    mse<float>((float *)gpu_dst, out_ref_fp32.getData<float>(), height * width);

  double cosSim = cosine_similarity<float>(
    (float *)gpu_dst, out_ref_fp32.getData<float>(), height * width);

  const float epsilon = 1e-3 * width;

  freeSVM(gpu_in1);
  freeSVM(gpu_in2);
  freeSVM(gpu_dst);

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

#ifdef ENABLE_FP16

TEST(blas_kernels, dotCL_sgemv_M_1_1_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_M_1_2_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_1_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 768;
  const int width_b = 1;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_2_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 2048;
  const int width_b = 1;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_n_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp16, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  EXPECT_THROW(dotCl(A_fp16, B_fp16, transA, transB), std::runtime_error);
}

TEST(blas_kernels, sgemv_3072_20120_noTrans) {
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 3072;

  int height_b = 3072;
  int width_b = 20120;

  bool transA = false;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, multiply_i) {
  const int batch = 1;
  const int channel = 1;
  const int height = 2;
  const int width = 11;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  // fp16
  multiplyCl(input, 0.1);

  // fp32
  multiplyCl(input_fp32, 0.1);

  float mseError = mse<_FP16>(input.getData<_FP16>(),
                              input_fp32.getData<float>(), input.size());

  double cosSim = cosine_similarity<_FP16>(
    input.getData<_FP16>(), input_fp32.getData<float>(), input.size());

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_noTrans_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transB_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_transA_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transAB_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = true;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, addition_i_fp16) {
  const int batch = 12;
  const int channel = 1;
  const int height = 26;
  const int width = 26;

  const int batch_b = 1;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch_b, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor C_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor D_fp16(batch_b, channel, height, width, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(B_fp16, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);
  GEN_TEST_INPUT(C_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(D_fp16, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);

  A_fp16.add_i(B_fp16);
  add_i_cl(C_fp16, D_fp16);

  float mseError =
    mse<_FP16>(A_fp16.getData<_FP16>(), C_fp16.getData<_FP16>(), A_fp16.size());

  double cosSim = cosine_similarity<_FP16>(
    A_fp16.getData<_FP16>(), C_fp16.getData<_FP16>(), A_fp16.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

/**
 * @brief The FP16 scalar multiply on the shared plane: both operands are SVM,
 *        the kernel scales every element, and the map the call leaves behind
 *        is what makes the result readable on the host.
 */
TEST(blas_kernels, scalar_mul_cl_fp16_shared_plane) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  constexpr unsigned int n = 512;
  constexpr float scalar = 0.375f;
  const size_t bytes = n * sizeof(_FP16);

  auto *in_svm = static_cast<_FP16 *>(allocateSVM(bytes));
  auto *out_svm = static_cast<_FP16 *>(allocateSVM(bytes));
  ASSERT_NE(in_svm, nullptr);
  ASSERT_NE(out_svm, nullptr);

  // Host writes through a map, exactly as a host-side producer would; the
  // call itself owns the unmap, so the test must not take it away.
  blas_cc->command_queue_inst_.enqueueSVMMap(in_svm, bytes, false);
  blas_cc->command_queue_inst_.enqueueSVMMap(out_svm, bytes, false);
  std::vector<float> expected(n);
  for (unsigned int i = 0; i < n; ++i) {
    const float v = 0.5f + 0.01f * static_cast<float>(i % 97);
    in_svm[i] = static_cast<_FP16>(v);
    out_svm[i] = static_cast<_FP16>(-1.0f);
    expected[i] = static_cast<float>(static_cast<_FP16>(v)) * scalar;
  }

  scalar_mul_cl_fp16(in_svm, out_svm, scalar, n, /*use_svm=*/true);

  const float epsilon = 1e-2f;
  for (unsigned int i = 0; i < n; ++i)
    EXPECT_NEAR(static_cast<float>(out_svm[i]), expected[i], epsilon);

  freeSVM(out_svm);
  freeSVM(in_svm);
}

/**
 * @brief A host pointer is not a kernel operand, so the call is refused
 *        instead of dispatching with its two buffer arguments never bound.
 *        The destination has to come back untouched -- a partial dispatch
 *        would be a driver error report on a kernel that is fine.
 */
TEST(blas_kernels, scalar_mul_cl_fp16_refuses_host_only_operands) {
  constexpr unsigned int n = 64;

  std::vector<_FP16> input(n, static_cast<_FP16>(2.0f));
  std::vector<_FP16> output(n, static_cast<_FP16>(-1.0f));

  scalar_mul_cl_fp16(input.data(), output.data(), 3.0f, n, /*use_svm=*/false);

  for (unsigned int i = 0; i < n; ++i)
    EXPECT_FLOAT_EQ(static_cast<float>(output[i]), -1.0f);
}
#endif

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
