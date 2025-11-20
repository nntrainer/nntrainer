// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_cuda_addition.cpp
 * @date   20 Nov 2025
 * @brief  Unit test for CUDA addition operations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>
#include <nntrainer_test_util.h>
#include <tensor.h>
#include <tensor_dim.h>
#include "addition_cuda.h"
#include <cuda_runtime.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(error) << std::endl;                     \
      FAIL();                                                                  \
    }                                                                          \
  } while (0)

using namespace nntrainer;


/**
 * @brief Test for addition_cuda function
 */
TEST(nntrainer_CUDA, addition_cuda) {
  const int size_input = 1024;
  const int size_res = 1024;

  // Allocate host memory
  std::vector<float> input_data(size_input);
  std::vector<float> res_data(size_res);
  std::vector<float> expected_data(size_res);

  // Allocate device memory
  float *d_input = nullptr;
  float *d_res = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, size_input * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_res, size_res * sizeof(float)));

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Call CUDA function 10 times
  for (int i = 0; i < 10; ++i) {
    // Initialize input data
    for (int j = 0; j < size_input; ++j) {
      input_data[j] = static_cast<float>((j + i) % 100) / 100.0f;
    }

    // Initialize result data
    for (int j = 0; j < size_res; ++j) {
      res_data[j] = static_cast<float>((j + i) % 50) / 100.0f;
      expected_data[j] = res_data[j] + input_data[j % size_input];
    }

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, input_data.data(), size_input * sizeof(float),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res, res_data.data(), size_res * sizeof(float),
               cudaMemcpyHostToDevice));

    if (i == 0) {
      // First call without timing
      addition_cuda(d_input, d_res, size_input, size_res);
    } else {
      // Subsequent calls with timing
      CUDA_CHECK(cudaEventRecord(start));
      addition_cuda(d_input, d_res, size_input, size_res);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));

      float milliseconds = 0;
      CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
      std::cout << "addition_cuda kernel execution time for call " << (i + 1) 
                << ": " << milliseconds << " ms" << std::endl;
    }

    // Copy result back to host
    std::vector<float> result_data(size_res);
    CUDA_CHECK(cudaMemcpy(result_data.data(), d_res, size_res * sizeof(float),
               cudaMemcpyDeviceToHost));

    // Check results (only for the last iteration)
    if (i == 9) {
      float mseError =
          mse<float>(result_data.data(), expected_data.data(), size_res);

      double cosSim = cosine_similarity<float>(result_data.data(),
                                               expected_data.data(), size_res);

      const float epsilon = 1e-5;

      if (mseError > epsilon) {
        std::cout << "MSE Error: " << mseError << std::endl;
      }
      EXPECT_IN_RANGE(mseError, 0, epsilon);
      
      if ((float)cosSim < 0.99) {
        std::cout << "Cosine Similarity: " << (float)cosSim << std::endl;
      }
      EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
    }
  }

  // Destroy CUDA events
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Free device memory
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_res));
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
