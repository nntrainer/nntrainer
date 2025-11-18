// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_cuda.cpp
 * @date   18 Nov 2025
 * @brief  Unit test for CUDA operations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nntrainer_test_util.h>
#include <rmsnorm_cuda.h>
#include <tensor.h>
#include <tensor_dim.h>

using namespace nntrainer;

/**
 * @brief Helper function to generate test data
 *
 * @tparam T data type
 * @param size data length
 * @param min_val minimum value
 * @param max_val maximum value
 * @return std::vector<T> random vector
 */
template <typename T>
static inline std::vector<T> generate_test_data(size_t size, T min_val,
                                                T max_val) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

/**
 * @brief Test for rmsnorm_cuda function
 */
TEST(nntrainer_CUDA, rmsnorm_cuda_1) {
  const int batch = 1;
  const int channel = 1;
  const int height = 67;
  const int width = 3072;

  const float epsilon = 1e-6;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  /// Initialize CPU input data
  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor gamma(1, 1, 1, width, t_type_nchw_fp32);
  nntrainer::Tensor output_cuda(batch, channel, height, width,
                                t_type_nchw_fp32);
  nntrainer::Tensor output_ref(batch, channel, height, width, t_type_nchw_fp32);

  /// Generate test data
  auto input_data = generate_test_data<float>(input.size(), -1.0f, 1.0f);
  auto gamma_data = generate_test_data<float>(gamma.size(), 0.5f, 2.0f);

  std::copy(input_data.begin(), input_data.end(), input.getData<float>());
  std::copy(gamma_data.begin(), gamma_data.end(), gamma.getData<float>());

  /// Allocate CUDA memory
  float *d_input = nullptr, *d_gamma = nullptr, *d_output = nullptr;
  size_t input_size = input.size() * sizeof(float);
  size_t gamma_size = gamma.size() * sizeof(float);
  size_t output_size = output_cuda.size() * sizeof(float);

  cudaMalloc((void **)&d_input, input_size);
  cudaMalloc((void **)&d_gamma, gamma_size);
  cudaMalloc((void **)&d_output, output_size);

  /// Copy data to CUDA memory
  cudaMemcpy(d_input, input.getData<float>(), input_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, gamma.getData<float>(), gamma_size,
             cudaMemcpyHostToDevice);

  /// Reference implementation using CPU
  std::function<float(float)> f = [](float x) { return 1 / std::sqrt(x); };
  auto t = input.multiply(input).average(3).add(epsilon);
  t.apply_i(f);
  input.multiply(t, output_ref);
  output_ref.multiply_i(gamma);

  /// Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /// CUDA implementation
  rmsnorm_cuda(d_input, d_gamma, d_output, epsilon,
               input.batch() * input.channel() * input.height(), input.width());

  /// Record start time
  cudaEventRecord(start);

  /// CUDA implementation
  rmsnorm_cuda(d_input, d_gamma, d_output, epsilon,
               input.batch() * input.channel() * input.height(), input.width());

  /// Record stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  /// Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "RMSNorm CUDA kernel execution time for input size (" << batch
            << ", " << channel << ", " << height << ", " << width
            << "): " << milliseconds << " ms" << std::endl;

  /// Copy result back to host
  cudaMemcpy(output_cuda.getData<float>(), d_output, output_size,
             cudaMemcpyDeviceToHost);

  /// Destroy CUDA events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  /// Free CUDA memory
  cudaFree(d_input);
  cudaFree(d_gamma);
  cudaFree(d_output);

  /// Compare results
  float mseError = mse<float>(output_cuda.getData<float>(),
                              output_ref.getData<float>(), output_cuda.size());

  double cosSim =
    cosine_similarity<float>(output_cuda.getData<float>(),
                             output_ref.getData<float>(), output_cuda.size());

  const float error_threshold = 1e-5;
  const float cosine_threshold = 0.999;

  EXPECT_LE(mseError, error_threshold);
  EXPECT_GE(cosSim, cosine_threshold);
}

/**
 * @brief Test for rmsnorm_cuda function with different dimensions
 */
TEST(nntrainer_CUDA, rmsnorm_cuda_2) {
  const int batch = 2;
  const int channel = 3;
  const int height = 32;
  const int width = 1024;

  const float epsilon = 1e-6;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  /// Initialize CPU input data
  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor gamma(1, 1, 1, width, t_type_nchw_fp32);
  nntrainer::Tensor output_cuda(batch, channel, height, width,
                                t_type_nchw_fp32);
  nntrainer::Tensor output_ref(batch, channel, height, width, t_type_nchw_fp32);

  /// Generate test data
  auto input_data = generate_test_data<float>(input.size(), -2.0f, 2.0f);
  auto gamma_data = generate_test_data<float>(gamma.size(), 0.1f, 1.5f);

  std::copy(input_data.begin(), input_data.end(), input.getData<float>());
  std::copy(gamma_data.begin(), gamma_data.end(), gamma.getData<float>());

  /// Allocate CUDA memory
  float *d_input = nullptr, *d_gamma = nullptr, *d_output = nullptr;
  size_t input_size = input.size() * sizeof(float);
  size_t gamma_size = gamma.size() * sizeof(float);
  size_t output_size = output_cuda.size() * sizeof(float);

  cudaMalloc((void **)&d_input, input_size);
  cudaMalloc((void **)&d_gamma, gamma_size);
  cudaMalloc((void **)&d_output, output_size);

  /// Copy data to CUDA memory
  cudaMemcpy(d_input, input.getData<float>(), input_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, gamma.getData<float>(), gamma_size,
             cudaMemcpyHostToDevice);

  /// Reference implementation using CPU
  std::function<float(float)> f = [](float x) { return 1 / std::sqrt(x); };
  auto t = input.multiply(input).average(3).add(epsilon);
  t.apply_i(f);
  input.multiply(t, output_ref);
  output_ref.multiply_i(gamma);

  /// Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /// Record start time
  cudaEventRecord(start);

  /// CUDA implementation
  rmsnorm_cuda(d_input, d_gamma, d_output, epsilon,
               input.batch() * input.channel() * input.height(), input.width());

  /// Record stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  /// Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "RMSNorm CUDA kernel execution time for input size (" << batch
            << ", " << channel << ", " << height << ", " << width
            << "): " << milliseconds << " ms" << std::endl;

  /// Copy result back to host
  cudaMemcpy(output_cuda.getData<float>(), d_output, output_size,
             cudaMemcpyDeviceToHost);

  /// Destroy CUDA events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  /// Free CUDA memory
  cudaFree(d_input);
  cudaFree(d_gamma);
  cudaFree(d_output);

  /// Compare results
  float mseError = mse<float>(output_cuda.getData<float>(),
                              output_ref.getData<float>(), output_cuda.size());

  double cosSim =
    cosine_similarity<float>(output_cuda.getData<float>(),
                             output_ref.getData<float>(), output_cuda.size());

  const float error_threshold = 1e-5;
  const float cosine_threshold = 0.999;

  EXPECT_LE(mseError, error_threshold);
  EXPECT_GE(cosSim, cosine_threshold);
}

/**
 * @brief Test for rmsnorm_cuda function with small epsilon
 */
TEST(nntrainer_CUDA, rmsnorm_cuda_3) {
  const int batch = 1;
  const int channel = 1;
  const int height = 10;
  const int width = 128;

  const float epsilon = 1e-12;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  /// Initialize CPU input data
  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor gamma(1, 1, 1, width, t_type_nchw_fp32);
  nntrainer::Tensor output_cuda(batch, channel, height, width,
                                t_type_nchw_fp32);
  nntrainer::Tensor output_ref(batch, channel, height, width, t_type_nchw_fp32);

  /// Generate test data
  auto input_data = generate_test_data<float>(input.size(), -0.5f, 0.5f);
  auto gamma_data = generate_test_data<float>(gamma.size(), 0.8f, 1.2f);

  std::copy(input_data.begin(), input_data.end(), input.getData<float>());
  std::copy(gamma_data.begin(), gamma_data.end(), gamma.getData<float>());

  /// Allocate CUDA memory
  float *d_input = nullptr, *d_gamma = nullptr, *d_output = nullptr;
  size_t input_size = input.size() * sizeof(float);
  size_t gamma_size = gamma.size() * sizeof(float);
  size_t output_size = output_cuda.size() * sizeof(float);

  cudaMalloc((void **)&d_input, input_size);
  cudaMalloc((void **)&d_gamma, gamma_size);
  cudaMalloc((void **)&d_output, output_size);

  /// Copy data to CUDA memory
  cudaMemcpy(d_input, input.getData<float>(), input_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, gamma.getData<float>(), gamma_size,
             cudaMemcpyHostToDevice);

  /// Reference implementation using CPU
  std::function<float(float)> f = [](float x) { return 1 / std::sqrt(x); };
  auto t = input.multiply(input).average(3).add(epsilon);
  t.apply_i(f);
  input.multiply(t, output_ref);
  output_ref.multiply_i(gamma);

  /// Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /// Record start time
  cudaEventRecord(start);

  /// CUDA implementation
  rmsnorm_cuda(d_input, d_gamma, d_output, epsilon,
               input.batch() * input.channel() * input.height(), input.width());

  /// Record stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  /// Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "RMSNorm CUDA kernel execution time for input size (" << batch
            << ", " << channel << ", " << height << ", " << width
            << "): " << milliseconds << " ms" << std::endl;

  /// Copy result back to host
  cudaMemcpy(output_cuda.getData<float>(), d_output, output_size,
             cudaMemcpyDeviceToHost);

  /// Destroy CUDA events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  /// Free CUDA memory
  cudaFree(d_input);
  cudaFree(d_gamma);
  cudaFree(d_output);

  /// Compare results
  float mseError = mse<float>(output_cuda.getData<float>(),
                              output_ref.getData<float>(), output_cuda.size());

  double cosSim =
    cosine_similarity<float>(output_cuda.getData<float>(),
                             output_ref.getData<float>(), output_cuda.size());

  const float error_threshold = 1e-5;
  const float cosine_threshold = 0.999;

  EXPECT_LE(mseError, error_threshold);
  EXPECT_GE(cosSim, cosine_threshold);
}

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
