// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_cuda_kernels_gelu.cpp
 * @date    27 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Validates the NVRTC GELU kernels (cuda_gelu_fp32/fp16) against a
 *          double-precision host reference GELU (both erf-exact and tanh
 *          modes); this run doubles as the NVRTC device probe for the new
 *          kernel strings (registration/compilation errors surface as
 *          ASSERT_TRUE failures, not silent skips).
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cuda_gelu.h>

namespace {

static bool cudaAvailable() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

// Host ports of cuda_gelu.cpp's gelu_h2f/gelu_f2h device helpers (plain C++,
// memcpy instead of __float_as_int/__int_as_float) so the fp16 test can
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

// Double-precision CPU reference GELU (mode 0 = erf-exact, mode 1 = tanh
// approximation); same constants as the device/host kernels.
static double ref_gelu(double x, int mode) {
  if (mode == 1) {
    const double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std::tanh(inner));
  }
  return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752));
}

// Fixed input of 24 values spanning negatives/zero/positives (same vector as
// the OpenCL blas_kernels.gelu_fp32 test).
static const std::vector<float> kHostIn = {
  -6.0f, -4.5f,  -3.0f, -2.5f, -2.0f, -1.5f, -1.0f, -0.75f,
  -0.5f, -0.25f, -0.1f, 0.0f,  0.1f,  0.25f, 0.5f,  0.75f,
  1.0f,  1.5f,   2.0f,  2.5f,  3.0f,  4.5f,  6.0f,  8.0f};

} // namespace

TEST(cuda_kernels, gelu_fp32) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";

  const unsigned int num_elems = (unsigned int)kHostIn.size();

  float *in = nullptr, *out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&in, (size_t)num_elems * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&out, (size_t)num_elems * sizeof(float)),
            cudaSuccess);

  std::memcpy(in, kHostIn.data(), kHostIn.size() * sizeof(float));

  for (int mode = 0; mode <= 1; ++mode) {
    ASSERT_TRUE(nntrainer::cuda::cuda_gelu_fp32(in, out, mode, num_elems));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float maxAbsErr = 0.0f;
    for (unsigned int i = 0; i < num_elems; ++i) {
      const double ref = ref_gelu((double)kHostIn[i], mode);
      const float gpu = out[i];
      const float err = std::fabs((float)(gpu - ref));
      if (err > maxAbsErr)
        maxAbsErr = err;
    }

    printf("CUDA GELU fp32 mode %d (%s) max abs error : %e\n", mode,
           mode == 1 ? "tanh" : "erf", maxAbsErr);
    EXPECT_LT(maxAbsErr, 1e-4f);
  }

  cudaFree(in);
  cudaFree(out);
}

/**
 * @brief fp16 I/O variant of the kernel test
 */
TEST(cuda_kernels, gelu_fp16) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";

  const unsigned int num_elems = (unsigned int)kHostIn.size();

  unsigned short *in = nullptr, *out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&in, (size_t)num_elems * sizeof(unsigned short)),
            cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&out, (size_t)num_elems * sizeof(unsigned short)),
            cudaSuccess);

  // Quantize inputs to fp16 up front, then build the reference from these
  // stored (already-rounded) values -- isolates kernel error from input
  // rounding error.
  std::vector<float> h_in_f(num_elems);
  std::vector<unsigned short> h_in(num_elems);
  for (unsigned int i = 0; i < num_elems; ++i) {
    unsigned short xh = f2h(kHostIn[i]);
    h_in[i] = xh;
    h_in_f[i] = h2f(xh);
  }

  std::memcpy(in, h_in.data(), h_in.size() * sizeof(unsigned short));

  for (int mode = 0; mode <= 1; ++mode) {
    ASSERT_TRUE(nntrainer::cuda::cuda_gelu_fp16(in, out, mode, num_elems));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float maxAbsErr = 0.0f;
    for (unsigned int i = 0; i < num_elems; ++i) {
      const double ref = ref_gelu((double)h_in_f[i], mode);
      const float gpu = h2f(out[i]);
      const float err = std::fabs((float)(gpu - ref));
      if (err > maxAbsErr)
        maxAbsErr = err;
    }

    printf("CUDA GELU fp16 mode %d (%s) max abs error : %e\n", mode,
           mode == 1 ? "tanh" : "erf", maxAbsErr);
    EXPECT_LT(maxAbsErr, 2e-2f);
  }

  cudaFree(in);
  cudaFree(out);
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
