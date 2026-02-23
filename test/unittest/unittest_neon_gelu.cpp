// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeong Gwon Hong <h0g1.hong@samsung.com>
 *
 * @file unittest_neon_gelu.cpp
 * @date   05 Feb 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeong Gwon Hong <h0g1.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Unittest for accuracy and time of swiglu and gelu kernels
 *
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <cpu_backend.h> // tanh_gelu, swiglu

namespace {

static inline float ref_tanh_gelu(float x) {
  const float half = 0.5f;
  const float c = 0.044715f;
  const float k = 0.7978845608f; // sqrt(2/pi)
  float x3 = x * x * x;
  return half * x * (1.0f + std::tanh(k * (x + c * x3)));
}

static inline float ref_tanh_gelu_mul(float x, float y) {
  const float half = 0.5f;
  const float c = 0.044715f;
  const float k = 0.7978845608f; // sqrt(2/pi)
  float x3 = x * x * x;
  return half * x * (1.0f + std::tanh(k * (x + c * x3))) * y;
}

static inline float ref_swiglu(float y, float z, float alpha) {
  float denom = 1.0f + std::exp(-alpha * y);
  return (y / denom) * z;
}

static inline float abs_err(float a, float b) { return std::fabs(a - b); }

static inline void expect_close(float got, float ref, float abs_tol,
                                float rel_tol) {
  float ae = abs_err(got, ref);
  EXPECT_TRUE(ae <= abs_tol) << "got=" << got << " ref=" << ref
                             << " abs_err=" << ae << " abs_tol=" << abs_tol;
}

} // namespace

TEST(ActivationNeon, TanhGeluAccuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), y_ref(N);
  for (size_t i = 0; i < N; ++i)
    x[i] = dist(rng);

  nntrainer::tanh_gelu(static_cast<unsigned int>(N), x.data(), y.data());

  // Reference
  for (size_t i = 0; i < N; ++i)
    y_ref[i] = ref_tanh_gelu(x[i]);

  // Tolerance, use abs_tol only now
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;

  // Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(y[i], y_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, TanhGeluv2Accuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(123);
  // Input Range
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), y_ref(N);
  for (size_t i = 0; i < N; ++i)
    x[i] = dist(rng);

  // DUT
  nntrainer::tanh_gelu_v2(static_cast<unsigned int>(N), x.data(),
                          y.data()); // 3

  // Reference
  for (size_t i = 0; i < N; ++i)
    y_ref[i] = ref_tanh_gelu(x[i]);

  // Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;

  // Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(y[i], y_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, SwiGluAccuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(456);
  // Input Range
  std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
  std::uniform_real_distribution<float> dist_z(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), z(N), x_ref(N);

  for (size_t i = 0; i < N; ++i) {
    y[i] = dist_y(rng);
    z[i] = dist_z(rng);
  }

  const float alpha = 1.0f; // Parametrize it if necessary
  nntrainer::swiglu(static_cast<unsigned int>(N), x.data(), y.data(), z.data(),
                    alpha); // 4

  for (size_t i = 0; i < N; ++i)
    x_ref[i] = ref_swiglu(y[i], z[i], alpha);

  // Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;

  // Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(x[i], x_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, TanhGeluMulAccuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(456);
  // Input Range
  std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
  std::uniform_real_distribution<float> dist_z(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), z(N), x_ref(N);

  for (size_t i = 0; i < N; ++i) {
    y[i] = dist_y(rng);
    z[i] = dist_z(rng);
  }

  nntrainer::tanh_gelu_mul(static_cast<unsigned int>(N), x.data(), y.data(),
                           z.data());

  for (size_t i = 0; i < N; ++i)
    x_ref[i] = ref_tanh_gelu_mul(y[i], z[i]);

  // Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;

  // Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(x[i], x_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeonPerf, TanhGeluVsSwiGluTime) {
  constexpr size_t N = 8192;
  constexpr int iters = 1000;

  std::mt19937 rng(789);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  std::uniform_real_distribution<float> distz(-5.0f, 5.0f);

  std::vector<float> x(N * iters), y(N * iters), z(N * iters), r(N * iters),
    out(N * iters);
  std::vector<float> x_d(N * iters), y_d(N * iters), z_d(N * iters),
    r_d(N * iters), out_d(N * iters);

  std::vector<std::vector<float>> inputs;

  // Input data
  float *px = x.data();
  float *py = y.data();
  float *pz = z.data();
  float *pr = r.data();
  float *pout = out.data();

  // Dummy data for fair comparison of kernel execution time
  float *px_d = x_d.data();
  float *py_d = y_d.data();
  float *pz_d = z_d.data();
  float *pr_d = r_d.data();
  float *pout_d = out_d.data();

  for (size_t i = 0; i < N * iters; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
    z[i] = distz(rng);
    x_d[i] = dist(rng);
    y_d[i] = dist(rng);
    z_d[i] = distz(rng);
  }

  auto bench = [&](const char *name, auto &&fn) {
    // timed
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      fn();
      // Next data
      px += N;
      py += N;
      pz += N;
      pr += N;
      pout += N;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    px = x.data();
    py = y.data();
    pz = z.data();
    pr = r.data();
    pout = out.data();

    // checksum to prevent DCE
    float acc = 0.f;
    for (size_t i = 0; i < N; i += 4096)
      acc += out[i];

    auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    double ms = ns / 1e6;
    std::cout << "[PERF] " << name << " : " << ms << " ms total, "
              << (ms / iters) << " ms/iter, "
              << "checksum=" << acc << "\n";
  };

  const float alpha = 1.0f;

  bench("swiglu(alpha=1)", [&]() {
    nntrainer::swiglu(static_cast<unsigned int>(N), pout, py, pz,
                      alpha); // 6
  });

  bench("tanh_gelu", [&]() {
    nntrainer::tanh_gelu(static_cast<unsigned int>(N), px, pout); // 5
  });

  bench("tanh_gelu_v2", [&]() {
    nntrainer::tanh_gelu_v2(static_cast<unsigned int>(N), px, pout); // 5
  });
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
