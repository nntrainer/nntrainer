// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_rope_reference.cpp
 * @date   27 January 2025
 * @brief  Unit test for RoPE reference implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author [Your Name] <[Your Email]>
 * @bug    No known bugs except for NYI items
 *
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "rope_reference.h"

// Helper to convert half precision (uint16_t) to float back for comparison
// Assuming standard IEEE 754 half-precision binary16
static float half_to_float(uint16_t h) {
    uint16_t h_exp = (h >> 10) & 0x1F;
    uint16_t h_sig = h & 0x3FF;
    
    if (h_exp == 0) {
        if (h_sig == 0) return 0.0f; // Signed zero not handled separately here
        return std::pow(2.0f, -14.0f) * (h_sig / 1024.0f); // Subnormal
    } else if (h_exp == 0x1F) {
        return h_sig == 0 ? INFINITY : NAN;
    } else {
        return std::pow(2.0f, (float)h_exp - 15.0f) * (1.0f + h_sig / 1024.0f);
    }
}

// Since we reuse the logic in the source code which uses _cvtsh_ss or similar intrinsics,
// we might not exact match with the simple helper above if there are rounding differences.
// But valid FP16 values should be close enough.
// Actually, to verify correctness properly, we should assume the scalar implementation in
// rotary_embedding_ref is the ground truth (or vice versa) and they should behave identically
// aside from potential minor precision differences if AVX2 uses different rounding.
// However, since both use similar logic structure, we expect very close results.

class RoPEReferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup random number generation
        std::random_device rd;
        gen = std::mt19937(rd());
        dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    }

    std::vector<float> create_random_vector(size_t size) {
        std::vector<float> vec(size);
        for (size_t i = 0; i < size; ++i) {
            vec[i] = dist(gen);
        }
        return vec;
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};


#ifdef ENABLE_CUDA
#include "rope.h"
#include <cuda_runtime.h>
#endif

// Parameter structure for RoPE tests
struct RoPETestParam {
    unsigned int width;
    unsigned int dim;
};

class RoPEMemoryTest : public ::testing::TestWithParam<RoPETestParam> {
protected:
    void SetUp() override {
        // Setup random number generation
        std::random_device rd;
        gen = std::mt19937(rd());
        dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    }

    std::vector<float> create_random_vector(size_t size) {
        std::vector<float> vec(size);
        for (size_t i = 0; i < size; ++i) {
            vec[i] = dist(gen);
        }
        return vec;
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

TEST_P(RoPEMemoryTest, ComparePerformance_AVX2_vs_CUDA) {
#ifndef ENABLE_CUDA
    GTEST_SKIP() << "CUDA not enabled, skipping performance comparison test";
#endif

    const auto& param = GetParam();
    const unsigned int WIDTH = param.width;
    const unsigned int DIM = param.dim;
    const unsigned int HALF = DIM / 2;

    // Prepare Data
    std::vector<float> input = create_random_vector(WIDTH);
    std::vector<float> cos_ = create_random_vector(HALF);
    std::vector<float> sin_ = create_random_vector(HALF);

    // Host outputs
    std::vector<uint16_t> output_avx2(WIDTH);
    std::vector<uint16_t> output_cuda(WIDTH, 0);

    // Device Memory
    float *d_input, *d_cos, *d_sin;
    uint16_t *d_output;
    
    ASSERT_EQ(cudaMalloc(&d_input, WIDTH * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_cos, HALF * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sin, HALF * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, WIDTH * sizeof(uint16_t)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_input, input.data(), WIDTH * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_cos, cos_.data(), HALF * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_sin, sin_.data(), HALF * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    // 1. Measure AVX2 Performance
    int iterations = 200;
    // Warmup
    for(int i=0; i<50; ++i)
        nntrainer::rotary_embedding_avx2_ref(output_avx2.data(), WIDTH, DIM, HALF, input.data(), cos_.data(), sin_.data(), false);
    
    // Benchmark AVX2
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        nntrainer::rotary_embedding_avx2_ref(output_avx2.data(), WIDTH, DIM, HALF, input.data(), cos_.data(), sin_.data(), false);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;
    double avg_time_cpu = duration_cpu.count() / iterations;

    // 2. Measure CUDA Performance
    // Warmup
    for(int i=0; i<50; ++i)
        nntrainer::rotary_embedding_cuda(d_output, WIDTH, DIM, HALF, d_input, d_cos, d_sin, false, 0);
    cudaDeviceSynchronize();

    // Benchmark CUDA
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    cudaEventRecord(start_evt);
    for(int i=0; i<iterations; ++i) {
        nntrainer::rotary_embedding_cuda(d_output, WIDTH, DIM, HALF, d_input, d_cos, d_sin, false, 0);
    }
    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);

    float total_ms_cuda = 0;
    cudaEventElapsedTime(&total_ms_cuda, start_evt, stop_evt);
    double avg_time_cuda = total_ms_cuda / iterations;

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);

    // Log Results
    std::cout << "[Benchmark] Width=" << WIDTH << ", Dim=" << DIM << std::endl;
    std::cout << "  AVX2 Time: " << avg_time_cpu << " ms" << std::endl;
    std::cout << "  CUDA Time: " << avg_time_cuda << " ms" << std::endl;
    if (avg_time_cuda < avg_time_cpu)
        std::cout << "  Speedup: " << avg_time_cpu / avg_time_cuda << "x" << std::endl;
    else
        std::cout << "  Slowdown: " << avg_time_cuda / avg_time_cpu << "x" << std::endl;


    // 3. Verify Correctness
    ASSERT_EQ(cudaMemcpy(output_cuda.data(), d_output, WIDTH * sizeof(uint16_t), cudaMemcpyDeviceToHost), cudaSuccess);
    
    int mismatch_count = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < WIDTH; ++i) {
        float val_cuda = half_to_float(output_cuda[i]);
        float val_scalar = half_to_float(output_avx2[i]);
        float diff = std::abs(val_cuda - val_scalar);
        
        if (diff > 1e-2) { 
            mismatch_count++;
             if (mismatch_count < 10) {
                 std::cout << "Mismatch at " << i << ": CUDA=" << val_cuda 
                           << ", AVX2=" << val_scalar << ", Diff=" << diff << std::endl;
             }
        }
        if (diff > max_diff) max_diff = diff;
    }
    
    cudaFree(d_input);
    cudaFree(d_cos);
    cudaFree(d_sin);
    cudaFree(d_output);

    EXPECT_EQ(mismatch_count, 0) << "Result mismatch between AVX2 and CUDA. Max Diff: " << max_diff;
}

INSTANTIATE_TEST_SUITE_P(
    RoPEBenchmark,
    RoPEMemoryTest,
    ::testing::Values(
        RoPETestParam{128 * 64, 128},    // Small
        RoPETestParam{1024 * 64, 128},   // Medium
        RoPETestParam{4096 * 64, 128},   // Large
        RoPETestParam{8192 * 64, 128}    // Very Large
    )
);


/**
 * @brief Main function for running tests
 */
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
