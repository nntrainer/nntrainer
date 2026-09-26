// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_cuda_kernels_layernorm.cpp
 * @date    27 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   First CUDA kernel unittest in this tree. Validates the NVRTC
 *          LayerNorm kernels (cuda_layernorm_fp32/fp16) against a
 *          double-precision host reference LayerNorm; this run doubles as
 *          the mandatory NVRTC device probe for the new kernel strings
 *          (registration/compilation errors surface as ASSERT_TRUE failures,
 *          not silent skips).
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cuda_layernorm.h>

namespace {

static bool cudaAvailable() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

// Host ports of cuda_layernorm.cpp's ln_h2f/ln_f2h device helpers (plain
// C++, memcpy instead of __float_as_int/__int_as_float) so the fp16 test can
// quantize its inputs/reference on the host the same way the device kernel
// reads/writes them.
static float h2f(unsigned short h) {
  unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, o;
  if (e == 0u) {
    if (m == 0u) {
      o = s;
    } else {
      int x = -1;
      do {
        m <<= 1;
        x++;
      } while ((m & 0x400u) == 0u);
      m &= 0x3FFu;
      o = s | ((unsigned int)(127 - 15 - x) << 23) | (m << 13);
    }
  } else if (e == 0x1Fu) {
    o = s | 0x7F800000u | (m << 13);
  } else {
    o = s | ((e + (127u - 15u)) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &o, sizeof(f));
  return f;
}

static unsigned short f2h(float f) {
  unsigned int xi;
  std::memcpy(&xi, &f, sizeof(xi));
  unsigned int s = (xi >> 16) & 0x8000u, mant = xi & 0x7FFFFFu;
  int e = (int)((xi >> 23) & 0xFFu);
  if (e == 0xFF)
    return (unsigned short)(s | 0x7C00u | (mant ? 0x200u : 0u));
  int exp = e - 127 + 15;
  if (exp >= 0x1F)
    return (unsigned short)(s | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10)
      return (unsigned short)s;
    mant |= 0x800000u;
    int sh = 14 - exp;
    unsigned int hh = mant >> sh, rem = mant & ((1u << sh) - 1u),
                 half = 1u << (sh - 1);
    if (rem > half || (rem == half && (hh & 1u)))
      hh++;
    return (unsigned short)(s | hh);
  }
  unsigned int hh = ((unsigned int)exp << 10) | (mant >> 13),
               rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (hh & 1u)))
    hh++;
  return (unsigned short)(s | hh);
}

} // namespace

TEST(cuda_kernels, layernorm_fp32) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";

  const float kEpsilon = 0.001f;
  // (rows, width): 515 exercises the 256-thread stride's scalar tail, 4096
  // exercises multiple stride iterations per thread.
  const std::vector<std::pair<unsigned int, unsigned int>> shapes = {
    {3, 8}, {4, 515}, {2, 4096}};

  for (auto &shape : shapes) {
    const unsigned int rows = shape.first, width = shape.second;

    float *in = nullptr, *gamma = nullptr, *beta = nullptr, *out = nullptr;
    ASSERT_EQ(cudaMallocManaged(&in, (size_t)rows * width * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&gamma, (size_t)width * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&beta, (size_t)width * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&out, (size_t)rows * width * sizeof(float)),
              cudaSuccess);

    std::vector<float> h_in((size_t)rows * width), h_gamma(width),
      h_beta(width);
    for (unsigned int w = 0; w < width; ++w) {
      h_gamma[w] = 0.5f + 0.1f * (float)(w % 16);
      h_beta[w] = -0.3f + 0.05f * (float)(w % 16);
    }
    for (unsigned int r = 0; r < rows; ++r)
      for (unsigned int w = 0; w < width; ++w)
        h_in[(size_t)r * width + w] =
          (float)((r * width + w) % 7) * 0.5f - 1.0f;

    std::memcpy(in, h_in.data(), h_in.size() * sizeof(float));
    std::memcpy(gamma, h_gamma.data(), h_gamma.size() * sizeof(float));
    std::memcpy(beta, h_beta.data(), h_beta.size() * sizeof(float));

    ASSERT_TRUE(nntrainer::cuda::cuda_layernorm_fp32(in, gamma, beta, out,
                                                     kEpsilon, rows, width));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Double-precision CPU reference LayerNorm.
    float maxAbsErr = 0.0f;
    for (unsigned int r = 0; r < rows; ++r) {
      double mean = 0.0;
      for (unsigned int w = 0; w < width; ++w)
        mean += h_in[(size_t)r * width + w];
      mean /= width;
      double var = 0.0;
      for (unsigned int w = 0; w < width; ++w) {
        double d = h_in[(size_t)r * width + w] - mean;
        var += d * d;
      }
      var /= width;
      double inv = 1.0 / std::sqrt(var + (double)kEpsilon);
      for (unsigned int w = 0; w < width; ++w) {
        double ref =
          (h_in[(size_t)r * width + w] - mean) * inv * h_gamma[w] + h_beta[w];
        float gpu = out[(size_t)r * width + w];
        float err = std::fabs(gpu - (float)ref);
        if (err > maxAbsErr)
          maxAbsErr = err;
      }
    }

    printf("CUDA LayerNorm fp32 %ux%u max abs error : %e\n", rows, width,
           maxAbsErr);
    EXPECT_LT(maxAbsErr, 1e-4f);

    cudaFree(in);
    cudaFree(gamma);
    cudaFree(beta);
    cudaFree(out);
  }
}

/**
 * @brief fp16 I/O variant of the kernel test
 */
TEST(cuda_kernels, layernorm_fp16) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";

  const float kEpsilon = 0.001f;
  const std::vector<std::pair<unsigned int, unsigned int>> shapes = {
    {3, 8}, {4, 515}, {2, 4096}};

  for (auto &shape : shapes) {
    const unsigned int rows = shape.first, width = shape.second;

    unsigned short *in = nullptr, *gamma = nullptr, *beta = nullptr,
                   *out = nullptr;
    ASSERT_EQ(
      cudaMallocManaged(&in, (size_t)rows * width * sizeof(unsigned short)),
      cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&gamma, (size_t)width * sizeof(unsigned short)),
              cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&beta, (size_t)width * sizeof(unsigned short)),
              cudaSuccess);
    ASSERT_EQ(
      cudaMallocManaged(&out, (size_t)rows * width * sizeof(unsigned short)),
      cudaSuccess);

    // Quantize inputs to fp16 up front, then build the reference from these
    // stored (already-rounded) values -- isolates kernel error from input
    // rounding error.
    std::vector<float> h_in_f((size_t)rows * width), h_gamma_f(width),
      h_beta_f(width);
    std::vector<unsigned short> h_in(rows * (size_t)width), h_gamma(width),
      h_beta(width);
    for (unsigned int w = 0; w < width; ++w) {
      float g = 0.5f + 0.1f * (float)(w % 16);
      float b = -0.3f + 0.05f * (float)(w % 16);
      h_gamma[w] = f2h(g);
      h_beta[w] = f2h(b);
      h_gamma_f[w] = h2f(h_gamma[w]);
      h_beta_f[w] = h2f(h_beta[w]);
    }
    for (unsigned int r = 0; r < rows; ++r) {
      for (unsigned int w = 0; w < width; ++w) {
        float x = (float)((r * width + w) % 7) * 0.5f - 1.0f;
        unsigned short xh = f2h(x);
        h_in[(size_t)r * width + w] = xh;
        h_in_f[(size_t)r * width + w] = h2f(xh);
      }
    }

    std::memcpy(in, h_in.data(), h_in.size() * sizeof(unsigned short));
    std::memcpy(gamma, h_gamma.data(), h_gamma.size() * sizeof(unsigned short));
    std::memcpy(beta, h_beta.data(), h_beta.size() * sizeof(unsigned short));

    ASSERT_TRUE(nntrainer::cuda::cuda_layernorm_fp16(in, gamma, beta, out,
                                                     kEpsilon, rows, width));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Double-precision CPU reference LayerNorm over the fp16-quantized
    // inputs (h_in_f/h_gamma_f/h_beta_f).
    float maxAbsErr = 0.0f;
    for (unsigned int r = 0; r < rows; ++r) {
      double mean = 0.0;
      for (unsigned int w = 0; w < width; ++w)
        mean += h_in_f[(size_t)r * width + w];
      mean /= width;
      double var = 0.0;
      for (unsigned int w = 0; w < width; ++w) {
        double d = h_in_f[(size_t)r * width + w] - mean;
        var += d * d;
      }
      var /= width;
      double inv = 1.0 / std::sqrt(var + (double)kEpsilon);
      for (unsigned int w = 0; w < width; ++w) {
        double ref =
          (h_in_f[(size_t)r * width + w] - mean) * inv * h_gamma_f[w] +
          h_beta_f[w];
        float gpu = h2f(out[(size_t)r * width + w]);
        float err = std::fabs(gpu - (float)ref);
        if (err > maxAbsErr)
          maxAbsErr = err;
      }
    }

    printf("CUDA LayerNorm fp16 %ux%u max abs error : %e\n", rows, width,
           maxAbsErr);
    EXPECT_LT(maxAbsErr, 2e-2f);

    cudaFree(in);
    cudaFree(gamma);
    cudaFree(beta);
    cudaFree(out);
  }
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
