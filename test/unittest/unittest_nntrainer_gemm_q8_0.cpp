// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_gemm_q8_0.cpp
 * @date   14 May 2026
 * @brief  Correctness tests for gemm_q8_0_fp32 / __fallback_gemm_q8_0.
 *         Compares the scalar Q8_0 GEMM against a plain FP32 reference
 *         GEMM, expecting agreement to within Q8_0 quantisation tolerance.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include <cpu_backend.h>
#include <fallback_internal.h>
#include <ggml_interface.h>
#include <nntr_ggml_impl.h>
#include <q8_0_tensor.h>

namespace {

constexpr unsigned int QK = 32;

// Reference Q8_0 quantisation (per-row, 32-element block).
// Mirrors GGML's quantize_row_q8_0_ref exactly: per-block fp16 scale =
// amax / 127, then int8 = round(x / scale).
static void quantize_row_q8_0_ref(const float *x, nntrainer::block_q8_0 *y,
                                  unsigned int k) {
  const unsigned int nb = k / QK;
  for (unsigned int b = 0; b < nb; ++b) {
    float amax = 0.0f;
    for (unsigned int l = 0; l < QK; ++l) {
      amax = std::max(amax, std::fabs(x[b * QK + l]));
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    // Pack scale: fp32 -> fp16 bit cast (IEEE 754 half).
    uint32_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FFu;
    if (exp <= 0) {
      y[b].d = static_cast<uint16_t>(sign);
    } else if (exp >= 31) {
      y[b].d = static_cast<uint16_t>(sign | (0x1Fu << 10) | mant);
    } else {
      y[b].d =
        static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
    }
    for (unsigned int l = 0; l < QK; ++l) {
      int32_t q = static_cast<int32_t>(std::round(x[b * QK + l] * id));
      q = std::max(-127, std::min(127, q));
      y[b].qs[l] = static_cast<int8_t>(q);
    }
  }
}

// Plain FP32 reference GEMM: C = A * B.T (A is MxK, B is NxK, C is MxN).
static void gemm_fp32_ref(unsigned int M, unsigned int N, unsigned int K,
                          const float *A, const float *B, float *C) {
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (unsigned int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[n * K + k];
      }
      C[m * N + n] = acc;
    }
  }
}

/**
 * @struct Shape
 * @brief GEMM dimensions shape parameter.
 */
struct Shape {
  unsigned int M, N, K;
};

/**
 * @class GemmQ8_0Param
 * @brief Parameterized test fixture for Q8_0 GEMM.
 */
class GemmQ8_0Param : public ::testing::TestWithParam<Shape> {};

} // namespace

TEST_P(GemmQ8_0Param, scalar_kernel_matches_fp32_reference_within_q8_0_tol) {
  const Shape sh = GetParam();
  ASSERT_EQ(sh.K % QK, 0u) << "K must be a multiple of 32";

  // Deterministic RNG so failures are reproducible.
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> A(static_cast<size_t>(sh.M) * sh.K);
  std::vector<float> B_fp32(static_cast<size_t>(sh.N) * sh.K);
  for (auto &v : A)
    v = dist(rng);
  for (auto &v : B_fp32)
    v = dist(rng);

  // Quantise B row-by-row to Q8_0 (offline).
  const unsigned int nb_per_row = sh.K / QK;
  std::vector<nntrainer::block_q8_0> B_q8(static_cast<size_t>(sh.N) *
                                          nb_per_row);
  for (unsigned int n = 0; n < sh.N; ++n) {
    quantize_row_q8_0_ref(B_fp32.data() + static_cast<size_t>(n) * sh.K,
                          B_q8.data() + static_cast<size_t>(n) * nb_per_row,
                          sh.K);
  }

  std::vector<float> C_q8(static_cast<size_t>(sh.M) * sh.N, 0.0f);
  std::vector<float> C_ref(static_cast<size_t>(sh.M) * sh.N, 0.0f);

  // GEMM under test: A FP32 (auto-quantised inside) * B Q8_0 -> C FP32.
  nntrainer::__fallback_gemm_q8_0<float>(sh.M, sh.N, sh.K, A.data(), sh.K,
                                         B_q8.data(), sh.K, C_q8.data(), sh.N);

  // Reference: FP32 x FP32, no quantisation.
  gemm_fp32_ref(sh.M, sh.N, sh.K, A.data(), B_fp32.data(), C_ref.data());

  // Tolerance: Q8_0 (8-bit) on both operands -> per-element relative error
  // bounded by ~2 / 127 ~= 1.6%, scaled by sqrt(K) for the accumulated dot.
  // We use an absolute tolerance proportional to K and the input scale.
  const float input_amax = 1.0f;               // dist range is [-1, 1]
  const float rel_err_per_dim = 2.0f / 127.0f; // Q8_0 quantisation step
  const float abs_tol =
    rel_err_per_dim * input_amax * input_amax * static_cast<float>(sh.K);

  size_t max_abs_off_idx = 0;
  float max_abs_off = 0.0f;
  for (size_t i = 0; i < C_q8.size(); ++i) {
    const float diff = std::fabs(C_q8[i] - C_ref[i]);
    if (diff > max_abs_off) {
      max_abs_off = diff;
      max_abs_off_idx = i;
    }
  }
  EXPECT_LE(max_abs_off, abs_tol)
    << "max |C_q8 - C_ref| = " << max_abs_off << " at idx " << max_abs_off_idx
    << " exceeds tolerance " << abs_tol << " (M=" << sh.M << " N=" << sh.N
    << " K=" << sh.K << ")";
}

TEST_P(GemmQ8_0Param, identity_weight_reproduces_input_row_sums) {
  // With B = identity (after Q8_0 quantisation), C[m, n] should be very
  // close to A[m, n]. This validates that the per-block scale and the int8
  // dot product cancel correctly.
  const Shape sh = GetParam();
  if (sh.K != sh.N) {
    GTEST_SKIP() << "identity-weight check only defined for square (N == K)";
  }
  ASSERT_EQ(sh.K % QK, 0u);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> A(static_cast<size_t>(sh.M) * sh.K);
  for (auto &v : A)
    v = dist(rng);

  // Build identity FP32 B (N x K with K == N) and quantise to Q8_0.
  std::vector<float> B_fp32(static_cast<size_t>(sh.N) * sh.K, 0.0f);
  for (unsigned int n = 0; n < sh.N; ++n) {
    B_fp32[static_cast<size_t>(n) * sh.K + n] = 1.0f;
  }

  const unsigned int nb_per_row = sh.K / QK;
  std::vector<nntrainer::block_q8_0> B_q8(static_cast<size_t>(sh.N) *
                                          nb_per_row);
  for (unsigned int n = 0; n < sh.N; ++n) {
    quantize_row_q8_0_ref(B_fp32.data() + static_cast<size_t>(n) * sh.K,
                          B_q8.data() + static_cast<size_t>(n) * nb_per_row,
                          sh.K);
  }

  std::vector<float> C(static_cast<size_t>(sh.M) * sh.N, 0.0f);
  nntrainer::__fallback_gemm_q8_0<float>(sh.M, sh.N, sh.K, A.data(), sh.K,
                                         B_q8.data(), sh.K, C.data(), sh.N);

  // With identity weight, only the diagonal Q8_0 block is non-zero;
  // expect C ~= A (within Q8_0 quantisation error on both A and that
  // single non-zero weight entry).
  const float abs_tol = 0.05f; // generous for an int8 round-trip
  for (size_t i = 0; i < A.size(); ++i) {
    ASSERT_LE(std::fabs(C[i] - A[i]), abs_tol)
      << "C[" << i << "]=" << C[i] << " vs A[" << i << "]=" << A[i];
  }
}

INSTANTIATE_TEST_SUITE_P(GemmQ8_0Shapes, GemmQ8_0Param,
                         ::testing::Values(Shape{1, 32, 32}, Shape{1, 64, 64},
                                           Shape{4, 32, 64}, Shape{8, 128, 128},
                                           Shape{2, 96, 96}));

// SIMD path: __ggml_q8_0_q8_0_GEMM uses the AVX2 nntr_gemm_q8_0_q8_0 kernel,
// which shares mul_sum_i8_pairs_acc_int32x8 with the Q4_0 kernel — identical
// int8 dot product instruction mix.
TEST_P(GemmQ8_0Param, simd_kernel_matches_scalar_oracle) {
  const Shape sh = GetParam();
  ASSERT_EQ(sh.K % QK, 0u);

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> A(static_cast<size_t>(sh.M) * sh.K);
  std::vector<float> B_fp32(static_cast<size_t>(sh.N) * sh.K);
  for (auto &v : A)
    v = dist(rng);
  for (auto &v : B_fp32)
    v = dist(rng);

  const unsigned int nb_per_row = sh.K / QK;
  std::vector<nntrainer::block_q8_0> B_q8(static_cast<size_t>(sh.N) *
                                          nb_per_row);
  for (unsigned int n = 0; n < sh.N; ++n) {
    quantize_row_q8_0_ref(B_fp32.data() + static_cast<size_t>(n) * sh.K,
                          B_q8.data() + static_cast<size_t>(n) * nb_per_row,
                          sh.K);
  }

  std::vector<float> C_scalar(static_cast<size_t>(sh.M) * sh.N, 0.0f);
  std::vector<float> C_simd(static_cast<size_t>(sh.M) * sh.N, 0.0f);

  nntrainer::__fallback_gemm_q8_0<float>(
    sh.M, sh.N, sh.K, A.data(), sh.K, B_q8.data(), sh.K, C_scalar.data(), sh.N);

  nntrainer::__ggml_q8_0_q8_0_GEMM(sh.M, sh.N, sh.K, A.data(), sh.K,
                                   B_q8.data(), sh.K, C_simd.data(), sh.N);

  // Both kernels do the same arithmetic on the same Q8_0 blocks; the only
  // possible discrepancy is fp32 accumulation order. A tight epsilon catches
  // any structural bug while tolerating that.
  const float abs_tol = 1e-3f;
  for (size_t i = 0; i < C_scalar.size(); ++i) {
    ASSERT_LE(std::fabs(C_simd[i] - C_scalar[i]), abs_tol)
      << "idx=" << i << "  simd=" << C_simd[i] << "  scalar=" << C_scalar[i];
  }
}

int main(int argc, char **argv) {
  // Initialise the fp16 <-> fp32 lookup table that the GGML quantised path
  // depends on. In production this is invoked by init_backend() during model
  // setup, but unit tests reach the GEMM kernels directly so we have to do it
  // here. Without this nntr_fp16_to_fp32() returns 0 for every input.
  nntrainer::__ggml_init();

  int result = -1;
  try {
    ::testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS" << std::endl;
  }
  return result;
}
