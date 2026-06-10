// SPDX-License-Identifier: Apache-2.0
/**
 * @file  benchmark_x86_backend.cpp
 * @date  24 March 2026
 * @brief Performance benchmarks for cpu compute backend
 * @see   https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho
 * @bug   No known bugs except for NYI items
 */

#include "benchmark_utils.h"
#include "nntrainer_test_util.h"

#include <cpu_backend.h>
#include <fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// ============================================================================
// Global benchmark configuration (settable via command-line)
//
//   --bench_iters=N     Number of measured iterations per test (default: 1000)
//   --bench_warmup=N    Number of warmup iterations (default: 10)
//   --bench_sizes=S     Comma-separated element sizes (default: 256,1024,4096)
// ============================================================================

static unsigned int g_bench_iters = 1000;
static unsigned int g_bench_warmup = 10;
static std::vector<unsigned int> g_bench_sizes = {256, 1024, 4096};

// ============================================================================
// Helpers
// ============================================================================

/**
 * @brief Parse an unsigned integer benchmark option.
 *
 * @param text input option value
 * @param value parsed output value
 * @param allow_zero true if zero is accepted
 * @return true if parsing succeeded
 */
static bool parse_uint_arg(const std::string &text, unsigned int &value,
                           bool allow_zero) {
  if (text.empty()) {
    return false;
  }

  unsigned long long parsed = 0;
  char extra = '\0';
  std::istringstream ss(text);
  if (!(ss >> parsed) || (ss >> extra)) {
    return false;
  }
  if ((!allow_zero && parsed == 0) ||
      parsed > std::numeric_limits<unsigned int>::max()) {
    return false;
  }

  value = static_cast<unsigned int>(parsed);
  return true;
}

/**
 * @brief Format a numeric GoogleTest property value without integer narrowing.
 *
 * @param value numeric value to record
 * @return string-formatted property value
 */
static std::string make_record_property_value(double value) {
  return std::to_string(value);
}

/**
 * @brief Convert FP32 values to raw FP16 bit-pattern values.
 *
 * @param f32_vec input FP32 vector
 * @return converted FP16 bit-pattern vector
 */
static inline std::vector<uint16_t>
convert_f32_to_f16_u16(const std::vector<float> &f32_vec) {
  std::vector<uint16_t> vec(f32_vec.size());
  for (size_t i = 0; i < f32_vec.size(); i++) {
    vec[i] = nntrainer::compute_fp32_to_fp16(f32_vec[i]);
  }
  return vec;
}

// ============================================================================
// Element-wise Benchmarks (FP32 vs FP16 interleaved)
// ============================================================================

/**
 * @brief Test fixture for element-wise operation benchmarks
 */
class Bench_EleOps : public ::testing::Test {};

TEST_F(Bench_EleOps, ele_sub) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));

    {
      auto X = generate_random_vector<float>(N);
      auto Y = generate_random_vector<float>(N, 0.1f, 1.0f);
      std::vector<float> Z(N);

      auto stats = bench::measure(
        [&]() {
          nntrainer::ele_sub(N, X.data(), Y.data(), Z.data(), 1.f, 0.f, 1, 1);
        },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      m.total_bytes = 3 * N * sizeof(float);
      bench::report("ele_sub", "FP32", "N=" + std::to_string(N), stats, m);
    }

#ifdef ENABLE_FP16
    {
      auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
      auto Y_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
      std::vector<uint16_t> Z(N);

      auto stats = bench::measure(
        [&]() {
          nntrainer::ele_sub(N, (const _FP16 *)X_u16.data(),
                             (const _FP16 *)Y_u16.data(), (_FP16 *)Z.data(),
                             1.f, 0.f, 1, 1);
        },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      m.total_bytes = 3 * N * sizeof(uint16_t);
      bench::report("ele_sub", "FP16", "N=" + std::to_string(N), stats, m);
    }
#endif
  }
}

TEST_F(Bench_EleOps, ele_div) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));

    {
      auto X = generate_random_vector<float>(N);
      auto Y = generate_random_vector<float>(N, 0.1f, 2.0f);
      std::vector<float> Z(N);

      auto stats = bench::measure(
        [&]() {
          nntrainer::ele_div(N, X.data(), Y.data(), Z.data(), 1.f, 0.f, 1, 1);
        },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      m.total_bytes = 3 * N * sizeof(float);
      bench::report("ele_div", "FP32", "N=" + std::to_string(N), stats, m);
    }

#ifdef ENABLE_FP16
    {
      auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
      auto Y_u16 =
        convert_f32_to_f16_u16(generate_random_vector<float>(N, 0.1f, 2.0f));
      std::vector<uint16_t> Z(N);

      auto stats = bench::measure(
        [&]() {
          nntrainer::ele_div(N, (const _FP16 *)X_u16.data(),
                             (const _FP16 *)Y_u16.data(), (_FP16 *)Z.data(),
                             1.f, 0.f, 1, 1);
        },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      m.total_bytes = 3 * N * sizeof(uint16_t);
      bench::report("ele_div", "FP16", "N=" + std::to_string(N), stats, m);
    }
#endif
  }
}

#ifdef ENABLE_FP16
TEST_F(Bench_EleOps, ele_mul) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    auto Y_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    std::vector<uint16_t> Z(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::ele_mul(N, (const _FP16 *)X_u16.data(),
                           (const _FP16 *)Y_u16.data(), (_FP16 *)Z.data(), 1.f,
                           0.f, 1, 1);
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = 3 * N * sizeof(uint16_t);
    bench::report("ele_mul", "FP16", "N=" + std::to_string(N), stats, m);
  }
}

TEST_F(Bench_EleOps, ele_add) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    auto Y_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    std::vector<uint16_t> Z(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::ele_add(N, (const _FP16 *)X_u16.data(),
                           (const _FP16 *)Y_u16.data(), (_FP16 *)Z.data(), 1.f,
                           0.f, 1, 1);
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = 3 * N * sizeof(uint16_t);
    bench::report("ele_add", "FP16", "N=" + std::to_string(N), stats, m);
  }
}
#endif

// ============================================================================
// Activation / Reduction Benchmarks (FP32 vs FP16 interleaved)
// ============================================================================

/**
 * @brief Test fixture for activation and reduction benchmarks
 */
class Bench_Activations : public ::testing::Test {};

TEST_F(Bench_Activations, softmax) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));

    {
      auto X = generate_random_vector<float>(N);
      std::vector<float> Y(N);

      auto stats =
        bench::measure([&]() { nntrainer::softmax(N, X.data(), Y.data()); },
                       g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("softmax", "FP32", "N=" + std::to_string(N), stats, m);
    }

#ifdef ENABLE_FP16
    {
      auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
      std::vector<uint16_t> Y(N);

      auto stats = bench::measure(
        [&]() {
          nntrainer::softmax(N, (_FP16 *)X_u16.data(), (_FP16 *)Y.data());
        },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("softmax", "FP16", "N=" + std::to_string(N), stats, m);
    }
#endif
  }
}

TEST_F(Bench_Activations, tanh_gelu) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto X = generate_random_vector<float>(N);
    std::vector<float> Y(N);

    auto stats =
      bench::measure([&]() { nntrainer::tanh_gelu(N, X.data(), Y.data()); },
                     g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    bench::report("tanh_gelu", "FP32", "N=" + std::to_string(N), stats, m);
  }
}

TEST_F(Bench_Activations, tanh_gelu_mul) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto Y_in = generate_random_vector<float>(N);
    auto Z_in = generate_random_vector<float>(N);
    std::vector<float> X(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::tanh_gelu_mul(N, X.data(), Y_in.data(), Z_in.data());
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    bench::report("tanh_gelu_mul", "FP32", "N=" + std::to_string(N), stats, m);
  }
}

TEST_F(Bench_Activations, inv_sqrt_inplace) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));

    {
      auto X_orig = generate_random_vector<float>(N, 0.01f, 10.0f);
      auto X_tmp = X_orig;

      auto stats = bench::measure_with_setup(
        [&]() { std::copy(X_orig.begin(), X_orig.end(), X_tmp.begin()); },
        [&]() { nntrainer::inv_sqrt_inplace(N, X_tmp.data()); }, g_bench_warmup,
        g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("inv_sqrt_inplace", "FP32", "N=" + std::to_string(N), stats,
                    m);
    }

#ifdef ENABLE_FP16
    {
      auto X_orig =
        convert_f32_to_f16_u16(generate_random_vector<float>(N, 0.01f, 10.0f));
      auto X_u16 = X_orig;

      auto stats = bench::measure_with_setup(
        [&]() { X_u16 = X_orig; },
        [&]() { nntrainer::inv_sqrt_inplace(N, (_FP16 *)X_u16.data()); },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("inv_sqrt_inplace", "FP16", "N=" + std::to_string(N), stats,
                    m);
    }
#endif
  }
}

TEST_F(Bench_Activations, max_val) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));

    {
      auto X = generate_random_vector<float>(N);

      float val = 0.0f;
      auto stats =
        bench::measure([&]() { val = nntrainer::max_val(N, X.data()); },
                       g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("max_val", "FP32", "N=" + std::to_string(N), stats, m);
    }

#ifdef ENABLE_FP16
    {
      auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));

      auto stats =
        bench::measure([&]() { nntrainer::max_val(N, (_FP16 *)X_u16.data()); },
                       g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("max_val", "FP16", "N=" + std::to_string(N), stats, m);
    }
#endif
  }
}

#ifdef ENABLE_FP16
TEST_F(Bench_Activations, swiglu) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto Y_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    auto Z_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    std::vector<uint16_t> X(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::swiglu(N, (_FP16 *)X.data(), (_FP16 *)Y_u16.data(),
                          (_FP16 *)Z_u16.data());
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    bench::report("swiglu", "FP16", "N=" + std::to_string(N), stats, m);
  }
}
#endif

// ============================================================================
// RMS Norm Benchmarks (FP32 vs FP16 interleaved)
// ============================================================================

/**
 * @brief Test fixture for RMS normalization benchmarks
 */
class Bench_RmsNorm
  : public ::testing::TestWithParam<std::tuple<unsigned int, unsigned int>> {};

TEST_P(Bench_RmsNorm, rms_norm) {
  auto [H, W] = GetParam();
  const unsigned int H_v = H;
  const unsigned int W_v = W;
  const unsigned int N = H * W;
  float epsilon = 1e-6f;
  std::string sz = "H=" + std::to_string(H) + ",W=" + std::to_string(W);

  {
    auto X = generate_random_vector<float>(N);
    std::vector<float> Y(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(X.data(), Y.data(), H_v,
                                                     W_v, epsilon);
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = 2 * N * sizeof(float);
    bench::report("rms_norm", "FP32", sz, stats, m);
  }

#ifdef ENABLE_FP16
  {
    auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    std::vector<uint16_t> Y(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::rms_norm_wrt_width_fp16_intrinsic<_FP16>(
          (const _FP16 *)X_u16.data(), (_FP16 *)Y.data(), H_v, W_v, epsilon);
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = 2 * N * sizeof(uint16_t);
    bench::report("rms_norm", "FP16", sz, stats, m);
  }
#endif
}

GTEST_PARAMETER_TEST(
  Dims, Bench_RmsNorm,
  ::testing::Values(std::make_tuple(1u, 128u), std::make_tuple(1u, 512u),
                    std::make_tuple(4u, 256u), std::make_tuple(4u, 1024u),
                    std::make_tuple(16u, 256u), std::make_tuple(16u, 1024u),
                    std::make_tuple(16u, 3072u)));

// ============================================================================
// FP16-only Benchmarks
// ============================================================================

#ifdef ENABLE_FP16

// ---- FP16 BLAS-like ----

/**
 * @brief Test fixture for FP16 BLAS operation benchmarks
 */
class Bench_FP16_BLAS : public ::testing::Test {};

TEST_F(Bench_FP16_BLAS, sdot) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    auto Y_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));

    auto stats = bench::measure(
      [&]() {
        nntrainer::sdot(N, (const _FP16 *)X_u16.data(), 1,
                        (const _FP16 *)Y_u16.data(), 1);
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = 2 * N * sizeof(uint16_t);
    bench::report("sdot", "FP16", "N=" + std::to_string(N), stats, m);
  }
}

TEST_F(Bench_FP16_BLAS, saxpy) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    auto Y_orig = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    auto Y_u16 = Y_orig;

    auto stats = bench::measure_with_setup(
      [&]() { Y_u16 = Y_orig; },
      [&]() {
        nntrainer::saxpy(N, 2.0f, (const _FP16 *)X_u16.data(), 1,
                         (_FP16 *)Y_u16.data(), 1);
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = 3 * N * sizeof(uint16_t);
    bench::report("saxpy", "FP16", "N=" + std::to_string(N), stats, m);
  }
}

TEST_F(Bench_FP16_BLAS, sscal) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto X_orig = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    auto X_u16 = X_orig;

    auto stats = bench::measure_with_setup(
      [&]() { X_u16 = X_orig; },
      [&]() { nntrainer::sscal(N, 0.5f, (_FP16 *)X_u16.data(), 1); },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = N * sizeof(uint16_t);
    bench::report("sscal", "FP16", "N=" + std::to_string(N), stats, m);
  }
}

// ============================================================================
// GEMM Benchmarks (Performance)
// ============================================================================

using HgemmParams =
  std::tuple<unsigned int, unsigned int, unsigned int, bool, bool>;
using GemvParams = std::tuple<unsigned int, unsigned int>;

/**
 * @brief Make a compact GEMM dimension label for benchmark output.
 *
 * @param M output row count
 * @param N output column count
 * @param K reduction dimension
 * @param TransA transpose flag for matrix A
 * @param TransB transpose flag for matrix B
 * @return formatted benchmark dimension label
 */
static std::string make_hgemm_size_label(unsigned int M, unsigned int N,
                                         unsigned int K, bool TransA,
                                         bool TransB) {
  return std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) +
         " " + (TransA ? "T" : "N") + (TransB ? "T" : "N");
}

/**
 * @brief Test fixture for x86 FP16 HGEMM benchmarks
 */
class Bench_X86_FP16_HGEMM : public ::testing::TestWithParam<HgemmParams> {};

TEST_P(Bench_X86_FP16_HGEMM, via_sgemm_dispatch) {
  auto [M, N, K, TransA, TransB] = GetParam();
  const unsigned int M_v = M;
  const unsigned int N_v = N;
  const unsigned int K_v = K;
  const bool TransA_v = TransA;
  const bool TransB_v = TransB;
  const unsigned int lda = TransA ? M : K;
  const unsigned int ldb = TransB ? K : N;
  const unsigned int ldc = N;
  const std::size_t a_size = static_cast<std::size_t>(TransA ? K : M) * lda;
  const std::size_t b_size = static_cast<std::size_t>(TransB ? N : K) * ldb;
  const std::size_t c_size = static_cast<std::size_t>(M) * ldc;

  auto A_fp32 = generate_random_vector<float>(a_size, -0.25f, 0.25f);
  auto B_fp32 = generate_random_vector<float>(b_size, -0.25f, 0.25f);
  std::vector<float> C_fp32(c_size);
  auto A_u16 = convert_f32_to_f16_u16(A_fp32);
  auto B_u16 = convert_f32_to_f16_u16(B_fp32);
  std::vector<uint16_t> C_u16(c_size);

  auto fp32_stats = bench::measure(
    [&]() {
      nntrainer::sgemm(0, TransA_v, TransB_v, M_v, N_v, K_v, 1.0f,
                       A_fp32.data(), lda, B_fp32.data(), ldb, 0.0f,
                       C_fp32.data(), ldc);
    },
    g_bench_warmup, g_bench_iters);

  auto fp16_stats = bench::measure(
    [&]() {
      // The public FP16 sgemm path dispatches to the x86 hgemm implementation.
      nntrainer::sgemm(0, TransA_v, TransB_v, M_v, N_v, K_v, 1.0f,
                       (const _FP16 *)A_u16.data(), lda,
                       (const _FP16 *)B_u16.data(), ldb, 0.0f,
                       (_FP16 *)C_u16.data(), ldc);
    },
    g_bench_warmup, g_bench_iters);

  std::string sz = make_hgemm_size_label(M, N, K, TransA, TransB);
  bench::Metrics m;
  m.flop_count = 2.0 * M * N * K;
  bench::report("fp32_sgemm", "FP32", sz, fp32_stats, m);
  bench::report("x86_fp16_hgemm", "FP16", sz, fp16_stats, m);
  bench::compare("x86_fp16_hgemm", "FP16", sz, fp32_stats, fp16_stats);

  RecordProperty("fp32_gflops",
                 make_record_property_value(
                   (2.0 * M * N * K / (fp32_stats.avg_ns * 1e-9)) / 1e9));
  RecordProperty("fp16_gflops",
                 make_record_property_value(
                   (2.0 * M * N * K / (fp16_stats.avg_ns * 1e-9)) / 1e9));
  RecordProperty("fp16_speedup", make_record_property_value(fp32_stats.avg_ns /
                                                            fp16_stats.avg_ns));
}

/**
 * @brief Test fixture for FP16 GEMV benchmarks
 */
class Bench_FP16_GEMV : public ::testing::TestWithParam<GemvParams> {};

TEST_P(Bench_FP16_GEMV, sgemv) {
  auto [M, N] = GetParam();
  const unsigned int M_v = M;
  const unsigned int N_v = N;
  auto A_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(M * N));
  auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
  auto Y_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(M));

  auto stats = bench::measure(
    [&]() {
      nntrainer::sgemv(0, false, M_v, N_v, 1.0f, (const _FP16 *)A_u16.data(),
                       N_v, (const _FP16 *)X_u16.data(), 1, 0.0f,
                       (_FP16 *)Y_u16.data(), 1);
    },
    g_bench_warmup, g_bench_iters);

  std::string sz = std::to_string(M) + "x" + std::to_string(N);
  bench::Metrics m;
  m.flop_count = 2.0 * M * N;
  bench::report("sgemv", "FP16", sz, stats, m);
}

GTEST_PARAMETER_TEST(
  HgemmDims, Bench_X86_FP16_HGEMM,
  ::testing::Values(HgemmParams{64, 64, 64, false, false},
                    HgemmParams{256, 256, 256, false, false},
                    HgemmParams{1024, 1024, 1024, false, false},
                    HgemmParams{97, 53, 71, false, false},
                    HgemmParams{7, 17, 33, false, false},
                    HgemmParams{1, 768, 1024, false, false},
                    HgemmParams{1, 3072, 512, false, false},
                    HgemmParams{1, 4096, 4096, false, false},
                    HgemmParams{16, 1024, 1024, false, false},
                    HgemmParams{16, 4096, 4096, false, false},
                    HgemmParams{128, 4096, 4096, false, false},
                    HgemmParams{256, 1024, 1024, false, false},
                    HgemmParams{256, 3072, 3072, false, false},
                    HgemmParams{1024, 3072, 3072, false, false},
                    HgemmParams{64, 64, 64, false, true},
                    HgemmParams{64, 64, 64, true, false},
                    HgemmParams{64, 64, 64, true, true}));

GTEST_PARAMETER_TEST(GemvDims, Bench_FP16_GEMV,
                     ::testing::Values(GemvParams{1, 768}, GemvParams{16, 1024},
                                       GemvParams{256, 1024},
                                       GemvParams{256, 3072},
                                       GemvParams{1024, 3072}));

#endif // ENABLE_FP16

// ============================================================================
// Attention Kernel Benchmarks
//
// These model the real prefill attention dataflow of
// MHACoreLayer::one_batch_incremental_forwarding: for every query row the three
// leaf kernels run in sequence, chained through a shared score buffer:
//
//   Q [seq_len, num_q_heads, head_dim]
//     --compute_kcaches (Q.K^T)------>  scores [T rows, num_q_heads]
//     --softmax_row_inplace---------->  scores (in place)
//     --compute_*vcache* (scores.V)-->  attn_out [seq_len, num_q_heads,
//     head_dim]
//
// where num_q_heads = num_cache_head * gqa_size and the score-buffer row count
//   T = seq_len*(seq_len+1)/2  (causal)   or   seq_len*seq_len  (non-causal).
// For query row i the kernel reads/writes one row slice: the Q/attn_out offset
// is i*num_q_heads*head_dim and the score offset is
// calc_attn_index(i)*num_q_heads (causal) or i*seq_len*num_q_heads
// (non-causal), matching mha_core exactly.
//
// At runtime this per-row loop is parallelized with ThreadManager; the
// benchmark runs it serially to isolate per-kernel cost (the bench build is
// NNTR_NUM_THREADS=1, and P4 measures per-kernel before/after). All three
// kernels share the same parameter set for directly comparable shapes.
//
// NOTE: work scales with the full prefill (~O(seq_len) more than a single
// call), so large seq_len configs may need a lower --bench_iters.
//
// KV cache is always kept as 16-bit regardless of enable-fp16:
//   enable-fp16=false: Q=FP32, K/V=FP16 stored in uint16_t  (Set A below)
//   enable-fp16=true : Q=FP16, K/V=FP16                     (Set B below)
// ============================================================================

// {seq_len, num_cache_head, gqa_size, head_dim, causal}
using AttnParams = std::tuple<int, int, int, int, bool>;

// Row-tile size for compute_kcaches: an internal tuning knob (not a model
// dimension), kept fixed so the shared configs vary only model shapes.
static constexpr int kAttnTileSize = 4;

// Causal score rows preceding query row i: i*(i+1)/2 (== MHACoreLayer's
// calc_attn_index, the offset into the packed triangular score buffer).
static inline size_t attn_causal_index(int i) {
  return static_cast<size_t>(i) * (static_cast<size_t>(i) + 1) / 2;
}

// Total score rows across all query rows for a given seq_len / causality.
static inline size_t attn_score_rows(int seq_len, bool causal) {
  const size_t s = static_cast<size_t>(seq_len);
  return causal ? s * (s + 1) / 2 : s * s;
}

static std::string attn_size_str(int seq_len, int num_cache_head, int gqa_size,
                                 int head_dim, bool causal) {
  return "seq=" + std::to_string(seq_len) +
         ",nch=" + std::to_string(num_cache_head) +
         ",gqa=" + std::to_string(gqa_size) +
         ",hd=" + std::to_string(head_dim) + (causal ? ",causal" : ",full");
}

static const auto kAttnConfigs = ::testing::Values(
  // Causal (primary path: IsCausal defaults to true for decoder LLMs).
  // {seq_len, num_cache_head, gqa_size, head_dim, causal}
  AttnParams{128, 8, 1, 64, true}, AttnParams{128, 8, 4, 128, true},
  AttnParams{512, 8, 1, 64, true}, AttnParams{512, 8, 4, 128, true},
  AttnParams{1024, 8, 1, 128, true}, AttnParams{1024, 8, 4, 128, true},
  AttnParams{2048, 8, 1, 128, true}, AttnParams{2048, 8, 4, 128, true},
  // Non-causal (shape coverage); small seq_len bounds the seq_len^2 score
  // buffer.
  AttnParams{128, 8, 4, 128, false}, AttnParams{512, 8, 1, 128, false});

// ---- Set A: FP32 Q + uint16(FP16-bits) KV cache (available in all builds)
// ----

/**
 * @brief Test fixture for compute_kcaches benchmarks (FP32 Q, uint16 KV)
 */
class Bench_KCache : public ::testing::TestWithParam<AttnParams> {};

TEST_P(Bench_KCache, compute_kcaches) {
  auto [seq_len, num_cache_head, gqa_size, head_dim, causal] = GetParam();
  // Plain local copies: clang (-Werror) rejects capturing structured-binding
  // names inside the [&] lambdas below.
  const int seq_len_v = seq_len;
  const int num_cache_head_v = num_cache_head;
  const int gqa_size_v = gqa_size;
  const int head_dim_v = head_dim;
  const bool causal_v = causal;
  const int total_heads = num_cache_head * gqa_size;
  const size_t score_rows = attn_score_rows(seq_len, causal);

  // Full attention input Q: [seq_len, num_q_heads, head_dim].
  auto q_f32 = generate_random_vector<float>(static_cast<size_t>(seq_len) *
                                             total_heads * head_dim);
  auto kcache_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(
    static_cast<size_t>(num_cache_head) * seq_len * head_dim));
  // Scores (== softmax / vcache input): packed [T rows, num_q_heads].
  std::vector<float> scores(score_rows * total_heads, 0.0f);

  auto stats = bench::measure(
    [&]() {
      for (int i = 0; i < seq_len_v; ++i) {
        const int num_rows = causal_v ? i + 1 : seq_len_v;
        const size_t in_off = static_cast<size_t>(i) * total_heads * head_dim_v;
        const size_t out_off = (causal_v ? attn_causal_index(i)
                                         : static_cast<size_t>(i) * seq_len_v) *
                               total_heads;
        nntrainer::compute_kcaches<uint16_t>(
          q_f32.data() + in_off, kcache_u16.data(), scores.data() + out_off,
          num_rows, num_cache_head_v, head_dim_v, gqa_size_v, kAttnTileSize,
          static_cast<size_t>(num_rows));
      }
    },
    g_bench_warmup, g_bench_iters);

  bench::report(
    "compute_kcaches", "FP32",
    attn_size_str(seq_len, num_cache_head, gqa_size, head_dim, causal), stats);

  RecordProperty("latency_ns", make_record_property_value(stats.avg_ns));
}

GTEST_PARAMETER_TEST(Configs, Bench_KCache, kAttnConfigs);

/**
 * @brief Test fixture for multihead row-softmax benchmarks (between kcache and
 * vcache in the attention dataflow)
 */
class Bench_SoftmaxRow : public ::testing::TestWithParam<AttnParams> {};

TEST_P(Bench_SoftmaxRow, softmax_row_inplace) {
  auto [seq_len, num_cache_head, gqa_size, head_dim, causal] = GetParam();
  // Plain local copies: clang (-Werror) rejects capturing structured-binding
  // names inside the [&] lambdas below.
  const int seq_len_v = seq_len;
  const bool causal_v = causal;
  const int total_heads = num_cache_head * gqa_size;
  const size_t score_rows = attn_score_rows(seq_len, causal);
  const std::string sz =
    attn_size_str(seq_len, num_cache_head, gqa_size, head_dim, causal);

  // Softmax operates over the score buffer (== compute_kcaches output): each
  // query row i normalizes its own [calc_attn_index(i), calc_attn_index(i+1))
  // row range, mirroring MHACoreLayer::softmax_triangle.
  {
    auto scores_orig = generate_random_vector<float>(score_rows * total_heads);
    auto scores = scores_orig;

    auto stats = bench::measure_with_setup(
      [&]() {
        std::copy(scores_orig.begin(), scores_orig.end(), scores.begin());
      },
      [&]() {
        for (int i = 0; i < seq_len_v; ++i) {
          const size_t start_row = causal_v
                                     ? attn_causal_index(i)
                                     : static_cast<size_t>(i) * seq_len_v;
          const size_t end_row = causal_v
                                   ? attn_causal_index(i + 1)
                                   : static_cast<size_t>(i + 1) * seq_len_v;
          nntrainer::softmax_row_inplace(scores.data(), start_row, end_row,
                                         static_cast<size_t>(total_heads));
        }
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = score_rows * total_heads;
    bench::report("softmax_row_inplace", "FP32", sz, stats, m);
  }

#if defined(ENABLE_FP16) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
  {
    auto scores_orig = convert_f32_to_f16_u16(
      generate_random_vector<float>(score_rows * total_heads));
    auto scores = scores_orig;

    auto stats = bench::measure_with_setup(
      [&]() { scores = scores_orig; },
      [&]() {
        for (int i = 0; i < seq_len_v; ++i) {
          const size_t start_row = causal_v
                                     ? attn_causal_index(i)
                                     : static_cast<size_t>(i) * seq_len_v;
          const size_t end_row = causal_v
                                   ? attn_causal_index(i + 1)
                                   : static_cast<size_t>(i + 1) * seq_len_v;
          nntrainer::softmax_row_inplace(
            (_FP16 *)scores.data(), start_row, end_row,
            static_cast<size_t>(total_heads), static_cast<_FP16 *>(nullptr));
        }
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = score_rows * total_heads;
    bench::report("softmax_row_inplace", "FP16", sz, stats, m);
  }
#endif
}

GTEST_PARAMETER_TEST(Configs, Bench_SoftmaxRow, kAttnConfigs);

/**
 * @brief Test fixture for attention kernel benchmarks (FP32 Q, uint16 KV)
 */
class Bench_Attention : public ::testing::TestWithParam<AttnParams> {};

TEST_P(Bench_Attention, compute_fp16vcache) {
  auto [seq_len, num_cache_head, gqa_size, head_dim, causal] = GetParam();
  // Plain local copies: clang (-Werror) rejects capturing structured-binding
  // names inside the [&] lambda below.
  const int seq_len_v = seq_len;
  const int num_cache_head_v = num_cache_head;
  const int gqa_size_v = gqa_size;
  const int head_dim_v = head_dim;
  const bool causal_v = causal;
  const int total_heads = num_cache_head * gqa_size;
  const size_t score_rows = attn_score_rows(seq_len, causal);

  // Attention weights (== compute_kcaches output): packed [T rows,
  // num_q_heads].
  auto scores_f32 = generate_random_vector<float>(score_rows * total_heads);
  auto vcache_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(
    static_cast<size_t>(num_cache_head) * seq_len * head_dim));
  // Attention output: [seq_len, num_q_heads, head_dim].
  std::vector<float> attn_out(
    static_cast<size_t>(seq_len) * total_heads * head_dim, 0.0f);

  auto stats = bench::measure(
    [&]() {
      for (int i = 0; i < seq_len_v; ++i) {
        const int row_num = causal_v ? i : seq_len_v - 1;
        const size_t in_off = (causal_v ? attn_causal_index(i)
                                        : static_cast<size_t>(i) * seq_len_v) *
                              total_heads;
        const size_t out_off =
          static_cast<size_t>(i) * total_heads * head_dim_v;
        nntrainer::compute_fp16vcache_fp32_transposed(
          row_num, scores_f32.data() + in_off, vcache_u16.data(),
          attn_out.data() + out_off, num_cache_head_v, gqa_size_v, head_dim_v,
          static_cast<size_t>(row_num + 1));
      }
    },
    g_bench_warmup, g_bench_iters);

  bench::report(
    "compute_fp16vcache", "FP32",
    attn_size_str(seq_len, num_cache_head, gqa_size, head_dim, causal), stats);

  RecordProperty("latency_ns", make_record_property_value(stats.avg_ns));
}

GTEST_PARAMETER_TEST(Configs, Bench_Attention, kAttnConfigs);

// ---- Set B: FP16 Q + FP16 KV cache (enable-fp16=true only) ----

#if defined(ENABLE_FP16) && (defined(__ARM_NEON) || defined(__ARM_NEON__))

/**
 * @brief Test fixture for compute_kcaches benchmarks (FP16 Q, FP16 KV)
 */
class Bench_KCache_FP16 : public ::testing::TestWithParam<AttnParams> {};

TEST_P(Bench_KCache_FP16, compute_kcaches) {
  auto [seq_len, num_cache_head, gqa_size, head_dim, causal] = GetParam();
  // Plain local copies: clang (-Werror) rejects capturing structured-binding
  // names inside the [&] lambda below.
  const int seq_len_v = seq_len;
  const int num_cache_head_v = num_cache_head;
  const int gqa_size_v = gqa_size;
  const int head_dim_v = head_dim;
  const bool causal_v = causal;
  const int total_heads = num_cache_head * gqa_size;
  const size_t score_rows = attn_score_rows(seq_len, causal);

  auto q_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(
    static_cast<size_t>(seq_len) * total_heads * head_dim));
  auto kcache_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(
    static_cast<size_t>(num_cache_head) * seq_len * head_dim));
  std::vector<uint16_t> scores(score_rows * total_heads, 0);

  auto stats = bench::measure(
    [&]() {
      for (int i = 0; i < seq_len_v; ++i) {
        const int num_rows = causal_v ? i + 1 : seq_len_v;
        const size_t in_off = static_cast<size_t>(i) * total_heads * head_dim_v;
        const size_t out_off = (causal_v ? attn_causal_index(i)
                                         : static_cast<size_t>(i) * seq_len_v) *
                               total_heads;
        nntrainer::compute_kcaches(
          (const _FP16 *)q_u16.data() + in_off,
          (const _FP16 *)kcache_u16.data(), (_FP16 *)scores.data() + out_off,
          num_rows, num_cache_head_v, head_dim_v, gqa_size_v, kAttnTileSize,
          static_cast<size_t>(num_rows));
      }
    },
    g_bench_warmup, g_bench_iters);

  bench::report(
    "compute_kcaches", "FP16",
    attn_size_str(seq_len, num_cache_head, gqa_size, head_dim, causal), stats);

  RecordProperty("latency_ns", make_record_property_value(stats.avg_ns));
}

GTEST_PARAMETER_TEST(Configs, Bench_KCache_FP16, kAttnConfigs);

/**
 * @brief Test fixture for attention kernel benchmarks (FP16 Q, FP16 KV)
 */
class Bench_Attention_FP16 : public ::testing::TestWithParam<AttnParams> {};

TEST_P(Bench_Attention_FP16, compute_fp16vcache_transposed) {
  auto [seq_len, num_cache_head, gqa_size, head_dim, causal] = GetParam();
  // Plain local copies: clang (-Werror) rejects capturing structured-binding
  // names inside the [&] lambda below.
  const int seq_len_v = seq_len;
  const int num_cache_head_v = num_cache_head;
  const int gqa_size_v = gqa_size;
  const int head_dim_v = head_dim;
  const bool causal_v = causal;
  const int total_heads = num_cache_head * gqa_size;
  const size_t score_rows = attn_score_rows(seq_len, causal);

  auto scores_u16 = convert_f32_to_f16_u16(
    generate_random_vector<float>(score_rows * total_heads));
  auto vcache_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(
    static_cast<size_t>(num_cache_head) * seq_len * head_dim));
  std::vector<uint16_t> attn_out(
    static_cast<size_t>(seq_len) * total_heads * head_dim, 0);

  auto stats = bench::measure(
    [&]() {
      for (int i = 0; i < seq_len_v; ++i) {
        const int row_num = causal_v ? i : seq_len_v - 1;
        const size_t in_off = (causal_v ? attn_causal_index(i)
                                        : static_cast<size_t>(i) * seq_len_v) *
                              total_heads;
        const size_t out_off =
          static_cast<size_t>(i) * total_heads * head_dim_v;
        nntrainer::compute_fp16vcache_transposed(
          row_num, (const _FP16 *)scores_u16.data() + in_off,
          (const _FP16 *)vcache_u16.data(), (_FP16 *)attn_out.data() + out_off,
          num_cache_head_v, gqa_size_v, head_dim_v,
          static_cast<size_t>(row_num + 1));
      }
    },
    g_bench_warmup, g_bench_iters);

  bench::report(
    "compute_fp16vcache_transposed", "FP16",
    attn_size_str(seq_len, num_cache_head, gqa_size, head_dim, causal), stats);

  RecordProperty("latency_ns", make_record_property_value(stats.avg_ns));
}

GTEST_PARAMETER_TEST(Configs, Bench_Attention_FP16, kAttnConfigs);

#endif // defined(ENABLE_FP16) && (defined(__ARM_NEON) || defined(__ARM_NEON__))

// ============================================================================
// Main
// ============================================================================

/**
 * @brief Parse benchmark-specific command-line arguments.
 *
 * @param argc argument count, updated after removing benchmark options
 * @param argv argument array, compacted in-place for GoogleTest
 */
static void parse_bench_args(int *argc, char **argv) {
  int out = 1;
  for (int i = 1; i < *argc; ++i) {
    if (strncmp(argv[i], "--bench_iters=", 14) == 0) {
      unsigned int v = 0;
      if (parse_uint_arg(argv[i] + 14, v, false)) {
        g_bench_iters = v;
      }
    } else if (strncmp(argv[i], "--bench_warmup=", 15) == 0) {
      unsigned int v = 0;
      if (parse_uint_arg(argv[i] + 15, v, true)) {
        g_bench_warmup = v;
      }
    } else if (strncmp(argv[i], "--bench_sizes=", 14) == 0) {
      std::vector<unsigned int> parsed_sizes;
      std::istringstream ss(argv[i] + 14);
      std::string token;
      while (std::getline(ss, token, ',')) {
        unsigned int v = 0;
        if (parse_uint_arg(token, v, false)) {
          parsed_sizes.push_back(v);
        }
      }
      if (!parsed_sizes.empty()) {
        g_bench_sizes = parsed_sizes;
      }
    } else {
      argv[out++] = argv[i];
    }
  }
  *argc = out;
}

/**
 * @brief Benchmark binary entry point.
 *
 * @param argc argument count
 * @param argv argument array
 * @return GoogleTest result code
 */
int main(int argc, char **argv) {
  parse_bench_args(&argc, argv);

  nntrainer::init_backend();
  ::testing::InitGoogleTest(&argc, argv);

  std::cout << "  Config: iters=" << g_bench_iters
            << ", warmup=" << g_bench_warmup << ", sizes={";
  for (size_t i = 0; i < g_bench_sizes.size(); ++i) {
    if (i > 0)
      std::cout << ",";
    std::cout << g_bench_sizes[i];
  }
  std::cout << "}" << std::endl;

  bench::print_separator("nntrainer CPU Backend Benchmark");
  bench::print_header();

  return RUN_ALL_TESTS();
}
