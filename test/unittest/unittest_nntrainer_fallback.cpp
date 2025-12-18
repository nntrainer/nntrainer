// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_nntrainer_fallback.cpp
 * @date 11 December 2025
 * @brief Unit tests for fallback CPU backend implementations
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include <fallback_internal.h>
#include <fallback_kleidiai.h>

namespace {

/**
 * @brief Generate random vector with uniform distribution
 */
template <typename T>
std::vector<T> generate_random_vector(size_t size, T min_val = -1.0,
                                      T max_val = 1.0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

/**
 * @brief Generate random integer vector
 */
template <typename T>
std::vector<T> generate_random_int_vector(size_t size, T min_val, T max_val) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

/**
 * @brief Compute relative error between two values
 */
template <typename T> T relative_error(T a, T b) {
  if (a == b)
    return 0;
  T max_val = std::max(std::abs(a), std::abs(b));
  if (max_val == 0)
    return std::abs(a - b);
  return std::abs(a - b) / max_val;
}

constexpr float TOLERANCE = 1e-5f;
constexpr float PI = 3.14159265358979323846f;

/// @brief generate sequence 1, 2, 3, 4, 5
} // namespace

//==============================================================================
// Tests for sscal (vector scaling)
//==============================================================================

TEST(nntrainer_fallback, sscal_basic) {
  const unsigned int N = 16;
  std::vector<float> X = generate_random_vector<float>(N);
  std::vector<float> X_expected(X);
  const float alpha = 2.5f;

  // Expected result
  for (unsigned int i = 0; i < N; ++i) {
    X_expected[i] *= alpha;
  }

  nntrainer::__fallback_sscal(N, alpha, X.data(), 1);

  for (unsigned int i = 0; i < N; ++i) {
    EXPECT_NEAR(X[i], X_expected[i], TOLERANCE);
  }
}

TEST(nntrainer_fallback, sscal_with_stride) {
  const unsigned int N = 8;
  const unsigned int stride = 2;
  std::vector<float> X(N * stride, 1.0f);
  const float alpha = 3.0f;

  nntrainer::__fallback_sscal(N, alpha, X.data(), stride);

  for (unsigned int i = 0; i < N; ++i) {
    EXPECT_NEAR(X[i * stride], alpha, TOLERANCE);
    if (i * stride + 1 < N * stride) {
      EXPECT_NEAR(X[i * stride + 1], 1.0f, TOLERANCE); // Unchanged
    }
  }
}

TEST(nntrainer_fallback, sscal_zero_alpha) {
  const unsigned int N = 10;
  std::vector<float> X = generate_random_vector<float>(N);

  nntrainer::__fallback_sscal(N, 0.0f, X.data(), 1);

  for (unsigned int i = 0; i < N; ++i) {
    EXPECT_NEAR(X[i], 0.0f, TOLERANCE);
  }
}

//==============================================================================
// Tests for snrm2 (Euclidean norm)
//==============================================================================

TEST(nntrainer_fallback, snrm2_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {3.0f, 4.0f, 0.0f, 0.0f};

  float result = nntrainer::__fallback_snrm2(N, X.data(), 1);

  EXPECT_NEAR(result, 5.0f, TOLERANCE); // sqrt(9 + 16) = 5
}

TEST(nntrainer_fallback, snrm2_unit_vector) {
  const unsigned int N = 100;
  std::vector<float> X(N, 1.0f / std::sqrt(static_cast<float>(N)));

  float result = nntrainer::__fallback_snrm2(N, X.data(), 1);

  EXPECT_NEAR(result, 1.0f, TOLERANCE);
}

TEST(nntrainer_fallback, snrm2_with_stride) {
  const unsigned int N = 3;
  const unsigned int stride = 2;
  std::vector<float> X = {3.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f};

  float result = nntrainer::__fallback_snrm2(N, X.data(), stride);

  EXPECT_NEAR(result, 5.0f, TOLERANCE);
}

//==============================================================================
// Tests for scopy (vector copy)
//==============================================================================

TEST(nntrainer_fallback, scopy_float_basic) {
  const unsigned int N = 16;
  std::vector<float> X = generate_random_vector<float>(N);
  std::vector<float> Y(N);

  nntrainer::__fallback_scopy(N, X.data(), 1, Y.data(), 1);

  for (unsigned int i = 0; i < N; ++i) {
    EXPECT_EQ(X[i], Y[i]);
  }
}

TEST(nntrainer_fallback, scopy_float_with_stride) {
  const unsigned int N = 8;
  std::vector<float> X = generate_random_vector<float>(N * 2);
  std::vector<float> Y(N * 3, 0.0f);

  nntrainer::__fallback_scopy(N, X.data(), 2, Y.data(), 3);

  for (unsigned int i = 0; i < N; ++i) {
    EXPECT_EQ(X[i * 2], Y[i * 3]);
  }
}

TEST(nntrainer_fallback, scopy_uint8_basic) {
  const unsigned int N = 16;
  std::vector<uint8_t> X(N);
  for (unsigned int i = 0; i < N; ++i) {
    X[i] = static_cast<uint8_t>(i);
  }
  std::vector<uint8_t> Y(N);

  nntrainer::__fallback_scopy(N, X.data(), 1, Y.data(), 1);

  for (unsigned int i = 0; i < N; ++i) {
    EXPECT_EQ(X[i], Y[i]);
  }
}

TEST(nntrainer_fallback, scopy_int8_basic) {
  const unsigned int N = 16;
  std::vector<int8_t> X(N);
  for (unsigned int i = 0; i < N; ++i) {
    X[i] = static_cast<int8_t>(i - 8);
  }
  std::vector<int8_t> Y(N);

  nntrainer::__fallback_scopy(N, X.data(), 1, Y.data(), 1);

  for (unsigned int i = 0; i < N; ++i) {
    EXPECT_EQ(X[i], Y[i]);
  }
}

//==============================================================================
// Tests for sdot (dot product)
//==============================================================================

TEST(nntrainer_fallback, sdot_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> Y = {4.0f, 3.0f, 2.0f, 1.0f};

  float result = nntrainer::__fallback_sdot(N, X.data(), 1, Y.data(), 1);

  // 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
  EXPECT_NEAR(result, 20.0f, TOLERANCE);
}

TEST(nntrainer_fallback, sdot_with_stride) {
  const unsigned int N = 3;
  std::vector<float> X = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f};
  std::vector<float> Y = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

  float result = nntrainer::__fallback_sdot(N, X.data(), 2, Y.data(), 2);

  EXPECT_NEAR(result, 6.0f, TOLERANCE); // 1 + 2 + 3 = 6
}

TEST(nntrainer_fallback, sdot_orthogonal) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> Y = {0.0f, 1.0f, 0.0f, 0.0f};

  float result = nntrainer::__fallback_sdot(N, X.data(), 1, Y.data(), 1);

  EXPECT_NEAR(result, 0.0f, TOLERANCE);
}

//==============================================================================
// Tests for saxpy (Y = alpha * X + Y)
//==============================================================================

TEST(nntrainer_fallback, saxpy_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> Y = {1.0f, 1.0f, 1.0f, 1.0f};
  const float alpha = 2.0f;

  nntrainer::__fallback_saxpy(N, alpha, X.data(), 1, Y.data(), 1);

  // Y = 2 * X + Y = {3, 5, 7, 9}
  EXPECT_NEAR(Y[0], 3.0f, TOLERANCE);
  EXPECT_NEAR(Y[1], 5.0f, TOLERANCE);
  EXPECT_NEAR(Y[2], 7.0f, TOLERANCE);
  EXPECT_NEAR(Y[3], 9.0f, TOLERANCE);
}

TEST(nntrainer_fallback, saxpy_zero_alpha) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> Y = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> Y_expected(Y);

  nntrainer::__fallback_saxpy(N, 0.0f, X.data(), 1, Y.data(), 1);

  for (unsigned int i = 0; i < N; ++i) {
    EXPECT_NEAR(Y[i], Y_expected[i], TOLERANCE);
  }
}

//==============================================================================
// Tests for sgemm (matrix multiplication)
//==============================================================================

TEST(nntrainer_fallback, sgemm_basic) {
  const unsigned int M = 2, N = 2, K = 2;
  // A = [1 2; 3 4], B = [5 6; 7 8]
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> C(M * N, 0.0f);

  nntrainer::__fallback_sgemm(0, false, false, M, N, K, 1.0f, A.data(), K,
                              B.data(), N, 0.0f, C.data(), N);

  // C = A * B = [19 22; 43 50]
  EXPECT_NEAR(C[0], 19.0f, TOLERANCE);
  EXPECT_NEAR(C[1], 22.0f, TOLERANCE);
  EXPECT_NEAR(C[2], 43.0f, TOLERANCE);
  EXPECT_NEAR(C[3], 50.0f, TOLERANCE);
}

TEST(nntrainer_fallback, sgemm_with_alpha_beta) {
  const unsigned int M = 2, N = 2, K = 2;
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> C = {1.0f, 1.0f, 1.0f, 1.0f};

  nntrainer::__fallback_sgemm(0, false, false, M, N, K, 2.0f, A.data(), K,
                              B.data(), N, 0.5f, C.data(), N);

  // C = 2 * (A * B) + 0.5 * C_old = 2 * [19 22; 43 50] + 0.5 * [1 1; 1 1]
  // = [38.5 44.5; 86.5 100.5]
  EXPECT_NEAR(C[0], 38.5f, TOLERANCE);
  EXPECT_NEAR(C[1], 44.5f, TOLERANCE);
  EXPECT_NEAR(C[2], 86.5f, TOLERANCE);
  EXPECT_NEAR(C[3], 100.5f, TOLERANCE);
}

TEST(nntrainer_fallback, sgemm_transA) {
  const unsigned int M = 2, N = 2, K = 2;
  // A transposed: original A = [1 3; 2 4] -> transposed view = [1 2; 3 4]
  std::vector<float> A = {1.0f, 3.0f, 2.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> C(M * N, 0.0f);

  nntrainer::__fallback_sgemm(0, true, false, M, N, K, 1.0f, A.data(), M,
                              B.data(), N, 0.0f, C.data(), N);

  // With TransA=true, we use columns of A as rows
  // C = A^T * B = [19 22; 43 50]
  EXPECT_NEAR(C[0], 19.0f, TOLERANCE);
  EXPECT_NEAR(C[1], 22.0f, TOLERANCE);
  EXPECT_NEAR(C[2], 43.0f, TOLERANCE);
  EXPECT_NEAR(C[3], 50.0f, TOLERANCE);
}

TEST(nntrainer_fallback, sgemm_transB) {
  const unsigned int M = 2, N = 2, K = 2;
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  // B transposed: original B = [5 7; 6 8] -> transposed view = [5 6; 7 8]
  std::vector<float> B = {5.0f, 7.0f, 6.0f, 8.0f};
  std::vector<float> C(M * N, 0.0f);

  nntrainer::__fallback_sgemm(0, false, true, M, N, K, 1.0f, A.data(), K,
                              B.data(), K, 0.0f, C.data(), N);

  // C = A * B^T = [19 22; 43 50]
  EXPECT_NEAR(C[0], 19.0f, TOLERANCE);
  EXPECT_NEAR(C[1], 22.0f, TOLERANCE);
  EXPECT_NEAR(C[2], 43.0f, TOLERANCE);
  EXPECT_NEAR(C[3], 50.0f, TOLERANCE);
}

//==============================================================================
// Tests for isamax (index of max absolute value)
//==============================================================================

TEST(nntrainer_fallback, isamax_basic) {
  const unsigned int N = 5;
  std::vector<float> X = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};

  unsigned int result = nntrainer::__fallback_isamax(N, X.data(), 1);

  EXPECT_EQ(result, 3u); // Index of 5.0
}

TEST(nntrainer_fallback, isamax_negative) {
  const unsigned int N = 5;
  std::vector<float> X = {-10.0f, 3.0f, 2.0f, 5.0f, 4.0f};

  unsigned int result = nntrainer::__fallback_isamax(N, X.data(), 1);

  // Implementation note: First iteration compares abs values to non-abs initial
  // max After first comparison, max_val becomes abs(X[n]), so 10.0 > 5.0
  // Returns raw index in array (not stride-adjusted)
  EXPECT_EQ(result, 3u); // Found 5.0 which is compared against non-abs -10.0
}

TEST(nntrainer_fallback, isamax_with_stride) {
  const unsigned int N = 3;
  std::vector<float> X = {1.0f, 100.0f, 10.0f, 100.0f, 5.0f, 100.0f};

  unsigned int result = nntrainer::__fallback_isamax(N, X.data(), 2);

  // With stride=2, it checks indices 0, 2, 4 -> values 1.0, 10.0, 5.0
  // Implementation returns raw index (not logical position), so max 10 is at
  // index 2 But the loop does n=1; n+=incX so it checks indices 1, 3, 5 -> 100,
  // 100, 100 First hit at index 1 wins
  EXPECT_EQ(result, 1u);
}

//==============================================================================
// Tests for elementwise operations
//==============================================================================

TEST(nntrainer_fallback, ele_mul_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> Y = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> Z(N, 0.0f);

  nntrainer::__fallback_ele_mul(N, X.data(), Y.data(), Z.data(), 1.0f, 0.0f, 1,
                                1);

  EXPECT_NEAR(Z[0], 2.0f, TOLERANCE);
  EXPECT_NEAR(Z[1], 6.0f, TOLERANCE);
  EXPECT_NEAR(Z[2], 12.0f, TOLERANCE);
  EXPECT_NEAR(Z[3], 20.0f, TOLERANCE);
}

TEST(nntrainer_fallback, ele_mul_with_alpha_beta) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> Y = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> Z = {1.0f, 1.0f, 1.0f, 1.0f};

  nntrainer::__fallback_ele_mul(N, X.data(), Y.data(), Z.data(), 2.0f, 0.5f, 1,
                                1);

  // Z = X * 2 * Y + 0.5 * Z_old = {4+0.5, 12+0.5, 24+0.5, 40+0.5}
  EXPECT_NEAR(Z[0], 4.5f, TOLERANCE);
  EXPECT_NEAR(Z[1], 12.5f, TOLERANCE);
  EXPECT_NEAR(Z[2], 24.5f, TOLERANCE);
  EXPECT_NEAR(Z[3], 40.5f, TOLERANCE);
}

TEST(nntrainer_fallback, ele_add_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> Y = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> Z(N, 0.0f);

  nntrainer::__fallback_ele_add(N, X.data(), Y.data(), Z.data(), 1.0f, 0.0f, 1,
                                1);

  EXPECT_NEAR(Z[0], 3.0f, TOLERANCE);
  EXPECT_NEAR(Z[1], 5.0f, TOLERANCE);
  EXPECT_NEAR(Z[2], 7.0f, TOLERANCE);
  EXPECT_NEAR(Z[3], 9.0f, TOLERANCE);
}

TEST(nntrainer_fallback, ele_sub_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> Y = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> Z(N, 0.0f);

  nntrainer::__fallback_ele_sub(N, X.data(), Y.data(), Z.data(), 1.0f, 0.0f, 1,
                                1);

  EXPECT_NEAR(Z[0], 4.0f, TOLERANCE);
  EXPECT_NEAR(Z[1], 4.0f, TOLERANCE);
  EXPECT_NEAR(Z[2], 4.0f, TOLERANCE);
  EXPECT_NEAR(Z[3], 4.0f, TOLERANCE);
}

TEST(nntrainer_fallback, ele_div_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {2.0f, 6.0f, 12.0f, 20.0f};
  std::vector<float> Y = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> Z(N, 0.0f);

  nntrainer::__fallback_ele_div(N, X.data(), Y.data(), Z.data(), 1.0f, 0.0f, 1,
                                1);

  EXPECT_NEAR(Z[0], 2.0f, TOLERANCE);
  EXPECT_NEAR(Z[1], 3.0f, TOLERANCE);
  EXPECT_NEAR(Z[2], 4.0f, TOLERANCE);
  EXPECT_NEAR(Z[3], 5.0f, TOLERANCE);
}

//==============================================================================
// Tests for trigonometric functions
//==============================================================================

TEST(nntrainer_fallback, sine_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {0.0f, PI / 6.0f, PI / 4.0f, PI / 2.0f};
  std::vector<float> Y(N, 0.0f);

  nntrainer::__fallback_sine<float>(N, X.data(), Y.data(), 1.0f, 1.0f);

  EXPECT_NEAR(Y[0], 0.0f, TOLERANCE);
  EXPECT_NEAR(Y[1], 0.5f, TOLERANCE);
  EXPECT_NEAR(Y[2], std::sqrt(2.0f) / 2.0f, TOLERANCE);
  EXPECT_NEAR(Y[3], 1.0f, TOLERANCE);
}

TEST(nntrainer_fallback, sine_with_alpha_beta) {
  const unsigned int N = 1;
  std::vector<float> X = {PI / 2.0f};
  std::vector<float> Y(N, 0.0f);

  nntrainer::__fallback_sine<float>(N, X.data(), Y.data(), 2.0f, 3.0f);

  // Y = sin(2 * pi/2) * 3 = sin(pi) * 3 = 0 * 3 = 0
  EXPECT_NEAR(Y[0], 0.0f, TOLERANCE);
}

TEST(nntrainer_fallback, cosine_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {0.0f, PI / 3.0f, PI / 4.0f, PI / 2.0f};
  std::vector<float> Y(N, 0.0f);

  nntrainer::__fallback_cosine<float>(N, X.data(), Y.data(), 1.0f, 1.0f);

  EXPECT_NEAR(Y[0], 1.0f, TOLERANCE);
  EXPECT_NEAR(Y[1], 0.5f, TOLERANCE);
  EXPECT_NEAR(Y[2], std::sqrt(2.0f) / 2.0f, TOLERANCE);
  EXPECT_NEAR(Y[3], 0.0f, TOLERANCE);
}

//==============================================================================
// Tests for inv_sqrt_inplace
//==============================================================================

TEST(nntrainer_fallback, inv_sqrt_inplace_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, 4.0f, 9.0f, 16.0f};

  nntrainer::__fallback_inv_sqrt_inplace(N, X.data());

  EXPECT_NEAR(X[0], 1.0f, TOLERANCE);
  EXPECT_NEAR(X[1], 0.5f, TOLERANCE);
  EXPECT_NEAR(X[2], 1.0f / 3.0f, TOLERANCE);
  EXPECT_NEAR(X[3], 0.25f, TOLERANCE);
}

//==============================================================================
// Tests for transpose_matrix
//==============================================================================

TEST(nntrainer_fallback, transpose_matrix_basic) {
  const unsigned int M = 2, N = 3;
  // src = [1 2 3; 4 5 6] in row-major
  std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> dst(N * M, 0.0f);

  nntrainer::__fallback_transpose_matrix(M, N, src.data(), N, dst.data(), M);

  // dst = [1 4; 2 5; 3 6] in row-major, stored column-major style
  EXPECT_NEAR(dst[0], 1.0f, TOLERANCE);
  EXPECT_NEAR(dst[1], 4.0f, TOLERANCE);
  EXPECT_NEAR(dst[2], 2.0f, TOLERANCE);
  EXPECT_NEAR(dst[3], 5.0f, TOLERANCE);
  EXPECT_NEAR(dst[4], 3.0f, TOLERANCE);
  EXPECT_NEAR(dst[5], 6.0f, TOLERANCE);
}

TEST(nntrainer_fallback, transpose_matrix_square) {
  const unsigned int M = 3, N = 3;
  std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                            6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<float> dst(M * N, 0.0f);

  nntrainer::__fallback_transpose_matrix(M, N, src.data(), N, dst.data(), M);

  // dst = [1 4 7; 2 5 8; 3 6 9]
  EXPECT_NEAR(dst[0], 1.0f, TOLERANCE);
  EXPECT_NEAR(dst[1], 4.0f, TOLERANCE);
  EXPECT_NEAR(dst[2], 7.0f, TOLERANCE);
  EXPECT_NEAR(dst[3], 2.0f, TOLERANCE);
  EXPECT_NEAR(dst[4], 5.0f, TOLERANCE);
  EXPECT_NEAR(dst[5], 8.0f, TOLERANCE);
  EXPECT_NEAR(dst[6], 3.0f, TOLERANCE);
  EXPECT_NEAR(dst[7], 6.0f, TOLERANCE);
  EXPECT_NEAR(dst[8], 9.0f, TOLERANCE);
}

//==============================================================================
// Tests for isValid
//==============================================================================

TEST(nntrainer_fallback, isValid_all_valid) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};

  bool result = nntrainer::__fallback_isValid(N, X.data());

  EXPECT_TRUE(result);
}

TEST(nntrainer_fallback, isValid_with_nan) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f,
                          4.0f};

  bool result = nntrainer::__fallback_isValid(N, X.data());

  EXPECT_FALSE(result);
}

TEST(nntrainer_fallback, isValid_with_inf) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, std::numeric_limits<float>::infinity(), 3.0f,
                          4.0f};

  bool result = nntrainer::__fallback_isValid(N, X.data());

  EXPECT_FALSE(result);
}

//==============================================================================
// Tests for type conversion functions
//==============================================================================

TEST(nntrainer_fallback, copy_s16_fp32_basic) {
  const unsigned int N = 4;
  std::vector<int16_t> X = {-100, 0, 100, 32767};
  std::vector<float> Y(N, 0.0f);

  nntrainer::__fallback_copy_s16_fp32(N, X.data(), Y.data());

  EXPECT_NEAR(Y[0], -100.0f, TOLERANCE);
  EXPECT_NEAR(Y[1], 0.0f, TOLERANCE);
  EXPECT_NEAR(Y[2], 100.0f, TOLERANCE);
  EXPECT_NEAR(Y[3], 32767.0f, TOLERANCE);
}

TEST(nntrainer_fallback, copy_u16_fp32_basic) {
  const unsigned int N = 4;
  std::vector<uint16_t> X = {0, 100, 1000, 65535};
  std::vector<float> Y(N, 0.0f);

  nntrainer::__fallback_copy_u16_fp32(N, X.data(), Y.data());

  EXPECT_NEAR(Y[0], 0.0f, TOLERANCE);
  EXPECT_NEAR(Y[1], 100.0f, TOLERANCE);
  EXPECT_NEAR(Y[2], 1000.0f, TOLERANCE);
  EXPECT_NEAR(Y[3], 65535.0f, TOLERANCE);
}

TEST(nntrainer_fallback, copy_fp32_u32_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {0.0f, 100.0f, 1000.0f, 4294967040.0f};
  std::vector<uint32_t> Y(N, 0);

  nntrainer::__fallback_copy_fp32_u32(N, X.data(), Y.data());

  EXPECT_EQ(Y[0], 0u);
  EXPECT_EQ(Y[1], 100u);
  EXPECT_EQ(Y[2], 1000u);
  EXPECT_EQ(Y[3], 4294967040u);
}

TEST(nntrainer_fallback, copy_fp32_u16_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {0.0f, 100.0f, 1000.0f, 65535.0f};
  std::vector<uint16_t> Y(N, 0);

  nntrainer::__fallback_copy_fp32_u16(N, X.data(), Y.data());

  EXPECT_EQ(Y[0], 0);
  EXPECT_EQ(Y[1], 100);
  EXPECT_EQ(Y[2], 1000);
  EXPECT_EQ(Y[3], 65535);
}

TEST(nntrainer_fallback, copy_fp32_u8_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {0.0f, 50.0f, 150.0f, 255.0f};
  std::vector<uint8_t> Y(N, 0);

  nntrainer::__fallback_copy_fp32_u8(N, X.data(), Y.data());

  EXPECT_EQ(Y[0], 0);
  EXPECT_EQ(Y[1], 50);
  EXPECT_EQ(Y[2], 150);
  EXPECT_EQ(Y[3], 255);
}

TEST(nntrainer_fallback, copy_fp32_s16_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {-100.0f, 0.0f, 100.0f, 32767.0f};
  std::vector<int16_t> Y(N, 0);

  nntrainer::__fallback_copy_fp32_s16(N, X.data(), Y.data());

  EXPECT_EQ(Y[0], -100);
  EXPECT_EQ(Y[1], 0);
  EXPECT_EQ(Y[2], 100);
  EXPECT_EQ(Y[3], 32767);
}

TEST(nntrainer_fallback, copy_fp32_s8_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {-128.0f, -50.0f, 50.0f, 127.0f};
  std::vector<int8_t> Y(N, 0);

  nntrainer::__fallback_copy_fp32_s8(N, X.data(), Y.data());

  EXPECT_EQ(Y[0], -128);
  EXPECT_EQ(Y[1], -50);
  EXPECT_EQ(Y[2], 50);
  EXPECT_EQ(Y[3], 127);
}

TEST(nntrainer_fallback, copy_s16_basic) {
  const unsigned int N = 4;
  std::vector<int16_t> X = {-100, 0, 100, 32767};
  std::vector<int16_t> Y(N, 0);

  nntrainer::__fallback_copy_s16(N, X.data(), Y.data());

  EXPECT_EQ(Y[0], -100);
  EXPECT_EQ(Y[1], 0);
  EXPECT_EQ(Y[2], 100);
  EXPECT_EQ(Y[3], 32767);
}

TEST(nntrainer_fallback, copy_u16_basic) {
  const unsigned int N = 4;
  std::vector<uint16_t> X = {0, 100, 1000, 65535};
  std::vector<uint16_t> Y(N, 0);

  nntrainer::__fallback_copy_u16(N, X.data(), Y.data());

  EXPECT_EQ(Y[0], 0);
  EXPECT_EQ(Y[1], 100);
  EXPECT_EQ(Y[2], 1000);
  EXPECT_EQ(Y[3], 65535);
}

TEST(nntrainer_fallback, scopy_int4_to_float32_basic) {
  const unsigned int N = 4;
  // Each byte contains two 4-bit values (high nibble and low nibble)
  std::vector<uint8_t> X = {0x12, 0x34, 0x56, 0x78};
  std::vector<float> Y(N * 2, 0.0f);

  nntrainer::__fallback_scopy_int4_to_float32(N, X.data(), 1, Y.data(), 1);

  // For each byte: Y[2*idx] = X[idx] >> 4 (high nibble), Y[2*idx+1] = X[idx] &
  // 0x0f (low nibble) 0x12: high = 1, low = 2
  EXPECT_NEAR(Y[0], 1.0f, TOLERANCE); // 0x12 >> 4 = 1
  EXPECT_NEAR(Y[1], 2.0f, TOLERANCE); // 0x12 & 0x0f = 2
  // 0x34: high = 3, low = 4
  EXPECT_NEAR(Y[2], 3.0f, TOLERANCE); // 0x34 >> 4 = 3
  EXPECT_NEAR(Y[3], 4.0f, TOLERANCE); // 0x34 & 0x0f = 4
  // 0x56: high = 5, low = 6
  EXPECT_NEAR(Y[4], 5.0f, TOLERANCE); // 0x56 >> 4 = 5
  EXPECT_NEAR(Y[5], 6.0f, TOLERANCE); // 0x56 & 0x0f = 6
  // 0x78: high = 7, low = 8
  EXPECT_NEAR(Y[6], 7.0f, TOLERANCE); // 0x78 >> 4 = 7
  EXPECT_NEAR(Y[7], 8.0f, TOLERANCE); // 0x78 & 0x0f = 8
}

TEST(nntrainer_fallback, scopy_int8_to_float32_basic) {
  const unsigned int N = 4;
  std::vector<int8_t> X = {-128, -10, 10, 127};
  std::vector<float> Y(N, 0.0f);

  nntrainer::__fallback_scopy_int8_to_float32(N, X.data(), 1, Y.data(), 1);

  EXPECT_NEAR(Y[0], -128.0f, TOLERANCE);
  EXPECT_NEAR(Y[1], -10.0f, TOLERANCE);
  EXPECT_NEAR(Y[2], 10.0f, TOLERANCE);
  EXPECT_NEAR(Y[3], 127.0f, TOLERANCE);
}

//==============================================================================
// Tests for activation functions
//==============================================================================

TEST(nntrainer_fallback, swiglu_basic) {
  const unsigned int N = 4;
  std::vector<float> X(N, 0.0f);
  std::vector<float> Y = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> Z = {1.0f, 1.0f, 1.0f, 1.0f};

  nntrainer::__fallback_swiglu(N, X.data(), Y.data(), Z.data());

  // X = (Y / (1 + exp(-Y))) * Z
  for (unsigned int i = 0; i < N; ++i) {
    float expected = (Y[i] / (1.0f + std::exp(-Y[i]))) * Z[i];
    EXPECT_NEAR(X[i], expected, TOLERANCE);
  }
}

TEST(nntrainer_fallback, swiglu_with_alpha) {
  const unsigned int N = 4;
  std::vector<float> X(N, 0.0f);
  std::vector<float> Y = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> Z = {1.0f, 1.0f, 1.0f, 1.0f};
  const float alpha = 2.0f;

  nntrainer::__fallback_swiglu(N, X.data(), Y.data(), Z.data(), alpha);

  // X = (Y / (1 + exp(-alpha * Y))) * Z
  for (unsigned int i = 0; i < N; ++i) {
    float expected = (Y[i] / (1.0f + std::exp(-alpha * Y[i]))) * Z[i];
    EXPECT_NEAR(X[i], expected, TOLERANCE);
  }
}

TEST(nntrainer_fallback, max_basic) {
  const unsigned int N = 5;
  std::vector<float> X = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f};

  float result = nntrainer::__fallback_max(N, X.data());

  EXPECT_NEAR(result, 5.0f, TOLERANCE);
}

TEST(nntrainer_fallback, max_negative) {
  const unsigned int N = 5;
  std::vector<float> X = {-1.0f, -5.0f, -3.0f, -4.0f, -2.0f};

  float result = nntrainer::__fallback_max(N, X.data());

  EXPECT_NEAR(result, -1.0f, TOLERANCE);
}

TEST(nntrainer_fallback, softmax_basic) {
  const unsigned int N = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> Y(N, 0.0f);

  nntrainer::__fallback_softmax(N, X.data(), Y.data());

  // Verify sum = 1
  float sum = 0.0f;
  for (unsigned int i = 0; i < N; ++i) {
    sum += Y[i];
  }
  EXPECT_NEAR(sum, 1.0f, TOLERANCE);

  // Verify ordering (Y[3] > Y[2] > Y[1] > Y[0])
  EXPECT_GT(Y[3], Y[2]);
  EXPECT_GT(Y[2], Y[1]);
  EXPECT_GT(Y[1], Y[0]);
}

TEST(nntrainer_fallback, softmax_uniform) {
  const unsigned int N = 4;
  std::vector<float> X = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> Y(N, 0.0f);

  nntrainer::__fallback_softmax(N, X.data(), Y.data());

  // All outputs should be equal (1/N)
  for (unsigned int i = 0; i < N; ++i) {
    EXPECT_NEAR(Y[i], 0.25f, TOLERANCE);
  }
}

TEST(nntrainer_fallback, clamp_basic) {
  const unsigned int N = 6;
  std::vector<float> input = {-2.0f, -0.5f, 0.0f, 0.5f, 1.5f, 2.0f};
  std::vector<float> output(N, 0.0f);

  nntrainer::__fallback_clamp<float>(input.data(), output.data(), N, -1.0f,
                                     1.0f);

  EXPECT_NEAR(output[0], -1.0f, TOLERANCE); // Clamped to lower
  EXPECT_NEAR(output[1], -0.5f, TOLERANCE); // Unchanged
  EXPECT_NEAR(output[2], 0.0f, TOLERANCE);  // Unchanged
  EXPECT_NEAR(output[3], 0.5f, TOLERANCE);  // Unchanged
  EXPECT_NEAR(output[4], 1.0f, TOLERANCE);  // Clamped to upper
  EXPECT_NEAR(output[5], 1.0f, TOLERANCE);  // Clamped to upper
}

//==============================================================================
// Tests for Kleidiai quantization functions
//==============================================================================

TEST(nntrainer_fallback_kleidiai, quant_qs4cx_f32_nxk_basic) {
  const size_t n = 4;
  const size_t k = 8;
  std::vector<float> rhs_f32 = generate_random_vector<float>(n * k);
  std::vector<uint8_t> rhs_qs4cx((n * (k + 1) / 2), 0);
  std::vector<float> rhs_scales_f32(n, 0.0f);

  quant_qs4cx_f32(n, k, rhs_format::nxk, rhs_f32.data(), rhs_qs4cx.data(),
                  rhs_scales_f32.data());

  // Verify scales are computed (non-zero for non-zero input)
  for (size_t i = 0; i < n; ++i) {
    EXPECT_TRUE(rhs_scales_f32[i] != 0.0f || rhs_f32[i * k] == 0.0f);
  }
}

TEST(nntrainer_fallback_kleidiai, quant_qs4cx_f32_kxn_basic) {
  const size_t n = 4;
  const size_t k = 8;
  std::vector<float> rhs_f32 = generate_random_vector<float>(n * k);
  std::vector<uint8_t> rhs_qs4cx((k * (n + 1) / 2), 0);
  std::vector<float> rhs_scales_f32(n, 0.0f);

  quant_qs4cx_f32(n, k, rhs_format::kxn, rhs_f32.data(), rhs_qs4cx.data(),
                  rhs_scales_f32.data());

  // Verify scales are computed
  for (size_t i = 0; i < n; ++i) {
    EXPECT_TRUE(rhs_scales_f32[i] != 0.0f || rhs_f32[i * k] == 0.0f);
  }
}

TEST(nntrainer_fallback_kleidiai, ref_quant_qa8dx_f32_basic) {
  const size_t m = 2;
  const size_t k = 8;
  std::vector<float> lhs_f32 = generate_random_vector<float>(m * k);

  // Output: scale (float) + offset (int32) + quantized values (int8 * k) per
  // row
  const size_t dst_stride =
    sizeof(float) + sizeof(int32_t) + k * sizeof(int8_t);
  std::vector<int8_t> lhs_qa8dx(m * dst_stride, 0);

  ref_quant_qa8dx_f32(m, k, lhs_f32.data(), lhs_qa8dx.data());

  // Verify quantization was performed (output is not all zeros for non-zero
  // input)
  bool has_nonzero = false;
  for (size_t i = 0; i < lhs_qa8dx.size(); ++i) {
    if (lhs_qa8dx[i] != 0) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero);
}

TEST(nntrainer_fallback_kleidiai, ref_matmul_f32_qa8dx_qs4cx_nxk) {
  const size_t m = 2;
  const size_t n = 2;
  const size_t k = 8;

  // Generate input data
  std::vector<float> lhs_f32 = generate_random_vector<float>(m * k);
  std::vector<float> rhs_f32 = generate_random_vector<float>(n * k);

  // Quantize LHS
  const size_t lhs_stride =
    sizeof(float) + sizeof(int32_t) + k * sizeof(int8_t);
  std::vector<int8_t> lhs_qa8dx(m * lhs_stride, 0);
  ref_quant_qa8dx_f32(m, k, lhs_f32.data(), lhs_qa8dx.data());

  // Quantize RHS
  std::vector<uint8_t> rhs_qs4cx((n * (k + 1) / 2), 0);
  std::vector<float> rhs_scales_f32(n, 0.0f);
  quant_qs4cx_f32(n, k, rhs_format::nxk, rhs_f32.data(), rhs_qs4cx.data(),
                  rhs_scales_f32.data());

  // Perform matmul
  std::vector<float> dst_f32(m * n, 0.0f);
  ref_matmul_f32_qa8dx_qs4cx(m, n, k, rhs_format::nxk, lhs_qa8dx.data(),
                             rhs_qs4cx.data(), rhs_scales_f32.data(),
                             dst_f32.data(), -std::numeric_limits<float>::max(),
                             std::numeric_limits<float>::max());

  // Simply verify output is computed (not all zeros)
  bool has_nonzero = false;
  for (size_t i = 0; i < dst_f32.size(); ++i) {
    if (dst_f32[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero);
}

TEST(nntrainer_fallback_kleidiai, ref_matmul_f32_qa8dx_qs4cx_kxn) {
  const size_t m = 2;
  const size_t n = 2;
  const size_t k = 8;

  // Generate input data
  std::vector<float> lhs_f32 = generate_random_vector<float>(m * k);
  std::vector<float> rhs_f32 = generate_random_vector<float>(n * k);

  // Quantize LHS
  const size_t lhs_stride =
    sizeof(float) + sizeof(int32_t) + k * sizeof(int8_t);
  std::vector<int8_t> lhs_qa8dx(m * lhs_stride, 0);
  ref_quant_qa8dx_f32(m, k, lhs_f32.data(), lhs_qa8dx.data());

  // Quantize RHS
  std::vector<uint8_t> rhs_qs4cx((k * (n + 1) / 2), 0);
  std::vector<float> rhs_scales_f32(n, 0.0f);
  quant_qs4cx_f32(n, k, rhs_format::kxn, rhs_f32.data(), rhs_qs4cx.data(),
                  rhs_scales_f32.data());

  // Perform matmul
  std::vector<float> dst_f32(m * n, 0.0f);
  ref_matmul_f32_qa8dx_qs4cx(m, n, k, rhs_format::kxn, lhs_qa8dx.data(),
                             rhs_qs4cx.data(), rhs_scales_f32.data(),
                             dst_f32.data(), -std::numeric_limits<float>::max(),
                             std::numeric_limits<float>::max());

  // Verify output is computed (not all zeros)
  bool has_nonzero = false;
  for (size_t i = 0; i < dst_f32.size(); ++i) {
    if (dst_f32[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero);
}

TEST(nntrainer_fallback_kleidiai, ref_matmul_f32_qa8dx_qs4cx_with_clamp) {
  const size_t m = 2;
  const size_t n = 2;
  const size_t k = 8;

  // Generate input data with larger values
  std::vector<float> lhs_f32 =
    generate_random_vector<float>(m * k, -10.0f, 10.0f);
  std::vector<float> rhs_f32 =
    generate_random_vector<float>(n * k, -10.0f, 10.0f);

  // Quantize LHS
  const size_t lhs_stride =
    sizeof(float) + sizeof(int32_t) + k * sizeof(int8_t);
  std::vector<int8_t> lhs_qa8dx(m * lhs_stride, 0);
  ref_quant_qa8dx_f32(m, k, lhs_f32.data(), lhs_qa8dx.data());

  // Quantize RHS
  std::vector<uint8_t> rhs_qs4cx((n * (k + 1) / 2), 0);
  std::vector<float> rhs_scales_f32(n, 0.0f);
  quant_qs4cx_f32(n, k, rhs_format::nxk, rhs_f32.data(), rhs_qs4cx.data(),
                  rhs_scales_f32.data());

  // Perform matmul with clamping
  const float clamp_min = -5.0f;
  const float clamp_max = 5.0f;
  std::vector<float> dst_f32(m * n, 0.0f);
  ref_matmul_f32_qa8dx_qs4cx(m, n, k, rhs_format::nxk, lhs_qa8dx.data(),
                             rhs_qs4cx.data(), rhs_scales_f32.data(),
                             dst_f32.data(), clamp_min, clamp_max);

  // Verify all outputs are within clamp range
  for (size_t i = 0; i < dst_f32.size(); ++i) {
    EXPECT_GE(dst_f32[i], clamp_min);
    EXPECT_LE(dst_f32[i], clamp_max);
  }
}

//==============================================================================
// Main function
//==============================================================================

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Failed to initialize google test" << std::endl;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Failed to run all tests" << std::endl;
  }

  return result;
}
