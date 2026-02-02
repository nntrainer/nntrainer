/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	unittest_neon_gelu.cpp
 * @date	02 Feb 2026
 * @brief	This is the unittest code for several kernels for activation function
 * @see		https://github.com/nntrainer/
 * @author	h0g1 <h0g1.hong@samsung.com>
 * @bug		No known bugs except for NYI items
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
  // Official formula of tanh_gelu
  // 0.5 * X * (1 + tanh(sqrt(2/pi) * (X + 0.044715 * X^3)))
  const float half = 0.5f;
  const float c = 0.044715f;
  const float k = 0.7978845608f; // sqrt(2/pi)
  float x3 = x * x * x;
  return half * x * (1.0f + std::tanh(k * (x + c * x3)));
}

static inline float ref_tanh_gelu_mul(float x, float y) {
  // Official formula of tanh_gelu
  // 0.5 * X * (1 + tanh(sqrt(2/pi) * (X + 0.044715 * X^3)))
  const float half = 0.5f;
  const float c = 0.044715f;
  const float k = 0.7978845608f; // sqrt(2/pi)
  float x3 = x * x * x;
  return half * x * (1.0f + std::tanh(k * (x + c * x3))) * y;
}

static inline float ref_swiglu(float y, float z, float alpha) {
  // (y / (1 + exp(-alpha*y))) * z
  float denom = 1.0f + std::exp(-alpha * y);
  return (y / denom) * z ;
}

static inline float abs_err(float a, float b) { return std::fabs(a - b); }


static inline void expect_close(float got, float ref,
                                float abs_tol, float rel_tol) {
  float ae = abs_err(got, ref);
  EXPECT_TRUE(ae <= abs_tol)
      << "got=" << got << " ref=" << ref
      << " abs_err=" << ae
      << " abs_tol=" << abs_tol;
}

} // namespace

TEST(ActivationNeon, TanhGeluAccuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(123);
  //Input Range
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), y_ref(N);
  for (size_t i = 0; i < N; ++i) x[i] = dist(rng);


  nntrainer::tanh_gelu(static_cast<unsigned int>(N), x.data(), y.data()); 

  // Reference
  for (size_t i = 0; i < N; ++i) y_ref[i] = ref_tanh_gelu(x[i]);

  //Tolerance, use abs_tol only now
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;

  // Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(y[i], y_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, TanhGelux2Accuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(123);
  // Input Range
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), y_ref(N);
  for (size_t i = 0; i < N; ++i) x[i] = dist(rng);

  // DUT
  nntrainer::tanh_gelu_unrolledx2(static_cast<unsigned int>(N), x.data(), y.data()); // 3

  // Reference
  for (size_t i = 0; i < N; ++i) y_ref[i] = ref_tanh_gelu(x[i]);

  //Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;

  // Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(y[i], y_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, TanhGelux4Accuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(123);
  // Input Range
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), y_ref(N);
  for (size_t i = 0; i < N; ++i) x[i] = dist(rng);

  // DUT
  nntrainer::tanh_gelu_unrolledx4(static_cast<unsigned int>(N), x.data(), y.data()); // 3

  // Reference
  for (size_t i = 0; i < N; ++i) y_ref[i] = ref_tanh_gelu(x[i]);

  // Tolerance
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
  for (size_t i = 0; i < N; ++i) x[i] = dist(rng);

  // DUT
  nntrainer::tanh_gelu_v2(static_cast<unsigned int>(N), x.data(), y.data()); // 3

  // Reference
  for (size_t i = 0; i < N; ++i) y_ref[i] = ref_tanh_gelu(x[i]);

  // Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;

  // Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(y[i], y_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, TanhGeluv2x2Accuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(123);
  // Input Range
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), y_ref(N);
  for (size_t i = 0; i < N; ++i) x[i] = dist(rng);

  // DUT
  nntrainer::tanh_gelu_v2_unrolledx2(static_cast<unsigned int>(N), x.data(), y.data()); // 3

  // Reference
  for (size_t i = 0; i < N; ++i) y_ref[i] = ref_tanh_gelu(x[i]);

  // Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;

  // Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(y[i], y_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, TanhGeluv2x4Accuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(123);
  // Input Range
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), y_ref(N);
  for (size_t i = 0; i < N; ++i) x[i] = dist(rng);

  // DUT
  nntrainer::tanh_gelu_v2_unrolledx4(static_cast<unsigned int>(N), x.data(), y.data()); // 3

  // Reference
  for (size_t i = 0; i < N; ++i) y_ref[i] = ref_tanh_gelu(x[i]);

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
  //Input Range
  std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
  std::uniform_real_distribution<float> dist_z(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), z(N), x_ref(N);

  for (size_t i = 0; i < N; ++i) {
    y[i] = dist_y(rng);
    z[i] = dist_z(rng);
  }

  const float alpha = 1.0f; // Parametrize it if necessary
  nntrainer::swiglu(static_cast<unsigned int>(N), x.data(), y.data(), z.data(), alpha); // 4

  for (size_t i = 0; i < N; ++i) x_ref[i] = ref_swiglu(y[i], z[i], alpha);

  //Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;
  
  //Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(x[i], x_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, SwiGlux2Accuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(456);
  std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
  std::uniform_real_distribution<float> dist_z(-3.0f, 3.0f);

  std::vector<float> x(N), y(N), z(N), x_ref(N);

  for (size_t i = 0; i < N; ++i) {
    y[i] = dist_y(rng);
    z[i] = dist_z(rng);
  }

  const float alpha = 1.0f; // Parametrize it if necessary
  nntrainer::swiglu_unrolledx2(static_cast<unsigned int>(N), x.data(), y.data(), z.data(), alpha); // 4

  for (size_t i = 0; i < N; ++i) x_ref[i] = ref_swiglu(y[i], z[i], alpha);

  // Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;

  for (size_t i = 0; i < N; ++i) {
    expect_close(x[i], x_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, SwiGlux4Accuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(456);
  std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
  std::uniform_real_distribution<float> dist_z(-3.0f, 3.0f);

  std::vector<float> x(N), y(N), z(N), x_ref(N);

  for (size_t i = 0; i < N; ++i) {
    y[i] = dist_y(rng);
    z[i] = dist_z(rng);
  }

  const float alpha = 1.0f; // Parametrize it if necessary
  nntrainer::swiglu_unrolledx4(static_cast<unsigned int>(N), x.data(), y.data(), z.data(), alpha); // 4

  for (size_t i = 0; i < N; ++i) x_ref[i] = ref_swiglu(y[i], z[i], alpha);

  // Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;

  for (size_t i = 0; i < N; ++i) {
    expect_close(x[i], x_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, TanhGeluV2MulAccuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(456);
  //Input Range
  std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
  std::uniform_real_distribution<float> dist_z(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), z(N), x_ref(N);

  for (size_t i = 0; i < N; ++i) {
    y[i] = dist_y(rng);
    z[i] = dist_z(rng);
  }

  nntrainer::tanh_gelu_v2_mul(static_cast<unsigned int>(N), x.data(), y.data(), z.data()); 

  for (size_t i = 0; i < N; ++i) x_ref[i] = ref_tanh_gelu_mul(y[i], z[i]);

  //Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;
  
  //Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(x[i], x_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, TanhGeluV2Mulx2Accuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(456);
  //Input Range
  std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
  std::uniform_real_distribution<float> dist_z(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), z(N), x_ref(N);

  for (size_t i = 0; i < N; ++i) {
    y[i] = dist_y(rng);
    z[i] = dist_z(rng);
  }

  nntrainer::tanh_gelu_v2_mul_unrolledx2(static_cast<unsigned int>(N), x.data(), y.data(), z.data()); 

  for (size_t i = 0; i < N; ++i) x_ref[i] = ref_tanh_gelu_mul(y[i], z[i]);

  //Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;
  
  //Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(x[i], x_ref[i], abs_tol, rel_tol);
  }
}

TEST(ActivationNeon, TanhGeluV2Mulx4Accuracy) {
  constexpr size_t N = 4096;

  std::mt19937 rng(456);
  //Input Range
  std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
  std::uniform_real_distribution<float> dist_z(-10.0f, 10.0f);

  std::vector<float> x(N), y(N), z(N), x_ref(N);

  for (size_t i = 0; i < N; ++i) {
    y[i] = dist_y(rng);
    z[i] = dist_z(rng);
  }

  nntrainer::tanh_gelu_v2_mul_unrolledx4(static_cast<unsigned int>(N), x.data(), y.data(), z.data()); 

  for (size_t i = 0; i < N; ++i) x_ref[i] = ref_tanh_gelu_mul(y[i], z[i]);

  //Tolerance
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-5f;
  
  //Test for each case
  for (size_t i = 0; i < N; ++i) {
    expect_close(x[i], x_ref[i], abs_tol, rel_tol);
  }
}



TEST(ActivationNeonPerf, TanhGeluVsSwiGluTime) {
  constexpr size_t N = 8192;     
  constexpr int iters = 200;              

  std::mt19937 rng(789);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> distz(-1.0f, 1.0f);

  std::vector<float> x(N*iters), y(N*iters), z(N*iters), r(N*iters), out(N*iters);
  std::vector<float> x_d(N*iters), y_d(N*iters), z_d(N*iters), r_d(N*iters), out_d(N*iters);

  std::vector<std::vector<float>> inputs;
  
  //Input data
  float *px = x.data();
  float *py = y.data();
  float *pz = z.data();
  float *pr = r.data();
  float *pout = out.data();

  //Dummy data for fair comparison of kernel execution time
  float *px_d = x_d.data();
  float *py_d = y_d.data();
  float *pz_d = z_d.data();
  float *pr_d = r_d.data();
  float *pout_d = out_d.data();

  for (size_t i = 0; i < N*iters; ++i) {
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
      //Next data
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
    volatile float acc = 0.f;
    for (size_t i = 0; i < N; i += 4096) acc += out[i];

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    double ms = ns / 1e6;
    std::cout << "[PERF] " << name << " : "
              << ms << " ms total, "
              << (ms / iters) << " ms/iter, "
              << "checksum=" << acc << "\n";
  };

  auto bench_dummy = [&](const char *name, auto &&fn) {
   
 
    for (int i = 0; i < iters; ++i) {
      fn();
      px_d += N;
      py_d += N;
      pz_d += N;
      pr_d += N;
      pout_d += N;
    }
    

    px_d = x_d.data();
    py_d = y_d.data();
    pz_d = z_d.data();
    pr_d = r_d.data();
    pout_d = out_d.data();

    // checksum to prevent DCE
    volatile float acc = 0.f;
    for (size_t i = 0; i < N; i += 4096) acc += out[i];
 
  };


  const float alpha = 1.0f;


  bench_dummy("swiglu(alpha=1)", [&](){
    nntrainer::swiglu(static_cast<unsigned int>(N), pout_d, py_d, pz_d, alpha); // 6
  });

  bench("swiglu(alpha=1)", [&](){
    nntrainer::swiglu(static_cast<unsigned int>(N), pout, py, pz, alpha); // 6
  });

  bench_dummy("swiglu(alpha=1)_unrolledx2", [&](){
    nntrainer::swiglu_unrolledx2(static_cast<unsigned int>(N), pout_d, py_d, pz_d, alpha); // 6
  });


  bench("swiglu(alpha=1)_unrolledx2", [&](){
    nntrainer::swiglu_unrolledx2(static_cast<unsigned int>(N), pout, py, pz, alpha); // 6
  });

  bench_dummy("swiglu(alpha=1)_unrolledx4", [&](){
    nntrainer::swiglu_unrolledx4(static_cast<unsigned int>(N), pout_d, py_d, pz_d, alpha); // 6
  });

  bench("swiglu(alpha=1)_unrolledx4", [&](){
    nntrainer::swiglu_unrolledx4(static_cast<unsigned int>(N), pout, py, pz, alpha); // 6
  });
  
  bench_dummy("tanh_gelu", [&](){
    nntrainer::tanh_gelu(static_cast<unsigned int>(N), px_d, pout_d); // 5
  });

  bench("tanh_gelu", [&](){
    nntrainer::tanh_gelu(static_cast<unsigned int>(N), px, pout); // 5
  });

  bench_dummy("tanh_gelu_unrolledx2", [&](){
    nntrainer::tanh_gelu_unrolledx2(static_cast<unsigned int>(N), px_d, pout_d); // 5    
  });

  bench("tanh_gelu_unrolledx2", [&](){
    nntrainer::tanh_gelu_unrolledx2(static_cast<unsigned int>(N), px, pout); // 5    
  });

  bench_dummy("tanh_gelu_unrolledx4", [&](){
    nntrainer::tanh_gelu_unrolledx4(static_cast<unsigned int>(N), px_d, pout_d); // 5
    
  });

  bench("tanh_gelu_unrolledx4", [&](){
    nntrainer::tanh_gelu_unrolledx4(static_cast<unsigned int>(N), px, pout); // 5
    
  });

  


  bench_dummy("tanh_gelu_v2", [&](){
    nntrainer::tanh_gelu_v2(static_cast<unsigned int>(N), px, pout); // 5
    
  });

  bench("tanh_gelu_v2", [&](){
    nntrainer::tanh_gelu_v2(static_cast<unsigned int>(N), px, pout); // 5
    
  });

  bench_dummy("tanh_gelu_v2_unrolledx2", [&](){
    nntrainer::tanh_gelu_v2_unrolledx2(static_cast<unsigned int>(N), px_d, pout_d); // 5
    
  });
  

  bench("tanh_gelu_v2_unrolledx2", [&](){
    nntrainer::tanh_gelu_v2_unrolledx2(static_cast<unsigned int>(N), px, pout); // 5
    
  });

  bench_dummy("tanh_gelu_v2_unrolledx4", [&](){
    nntrainer::tanh_gelu_v2_unrolledx4(static_cast<unsigned int>(N), px_d, pout_d); // 5
    
  });


  bench("tanh_gelu_v2_unrolledx4", [&](){
    nntrainer::tanh_gelu_v2_unrolledx4(static_cast<unsigned int>(N), px, pout); // 5
    
  });

  




  
  
  

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}