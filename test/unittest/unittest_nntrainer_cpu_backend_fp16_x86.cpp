// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_cpu_backend_fp16_x86.cpp
 * @date	16 June 2026
 * @brief	x86-only unit tests for the cache-blocked FP16 GEMM/GEMV backend
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 * @note Split out of unittest_nntrainer_cpu_backend_fp16.cpp (which is built
 * for Android only) so the x86 cache-blocked GEMM/GEMV tests build under meson
 * without touching that file. The whole body is gated on x86, so the target is
 * harmless to build on other architectures.
 */

#include "nntrainer_test_util.h"

#include <cpu_backend.h>
#include <fallback_internal.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <hgemm_common.h>
#include <hgemm_pack.h>
#ifdef ENABLE_TEST
#include <hgemm_test.h>
#define X86_HGEMM_WORKSPACE_STATS_AVAILABLE 1
#endif

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  double err_sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(static_cast<float>(
                                      src[i * N + j] - src2[i * N + j])));
      err_sum += std::abs(static_cast<float>(src[i * N + j] - src2[i * N + j]));
    }
  }
  return max_diff;
}

/// FP16 sgemm path: exercises the x86 cache-blocked GEMM.
/// Reference is a local FP32 GEMM loop to avoid using the optimized backend as
/// the oracle for this backend test.
static void run_sgemm_fp16_hgemm_test(unsigned int M, unsigned int N,
                                      unsigned int K, bool TransA = false,
                                      bool TransB = false, float alpha = 1.0F,
                                      float beta = 0.0F,
                                      unsigned int lda_extra = 0,
                                      unsigned int ldb_extra = 0,
                                      unsigned int ldc_extra = 0) {
  nntrainer::init_backend();

  const unsigned int lda = std::max(1u, (TransA ? M : K) + lda_extra);
  const unsigned int ldb = std::max(1u, (TransB ? K : N) + ldb_extra);
  const unsigned int ldc = std::max(1u, N + ldc_extra);
  const unsigned int a_rows = TransA ? K : M;
  const unsigned int b_rows = TransB ? N : K;
  const std::size_t a_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(a_rows) * lda);
  const std::size_t b_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(b_rows) * ldb);
  const std::size_t c_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(M) * ldc);

  auto A_fp16 = generate_random_vector<_FP16>(a_size, -0.25F, 0.25F);
  auto B_fp16 = generate_random_vector<_FP16>(b_size, -0.25F, 0.25F);
  auto C_fp16 = generate_random_vector<_FP16>(c_size, -0.25F, 0.25F);
  auto C_before = C_fp16;

  std::vector<float> C_fp32_ref(c_size);
  for (std::size_t i = 0; i < c_size; ++i) {
    C_fp32_ref[i] = static_cast<float>(C_before[i]);
  }

  if (M != 0 && N != 0) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N; ++n) {
        float acc = 0.0F;
        for (unsigned int k = 0; k < K; ++k) {
          const float a = static_cast<float>(TransA ? A_fp16[k * lda + m]
                                                    : A_fp16[m * lda + k]);
          const float b = static_cast<float>(TransB ? B_fp16[n * ldb + k]
                                                    : B_fp16[k * ldb + n]);
          acc += a * b;
        }
        const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
        C_fp32_ref[idx] = alpha * acc + beta * C_fp32_ref[idx];
      }
    }
  }

  // System under test: FP16 sgemm, routed to x86::hgemm_fp16 by the x86
  // backend dispatcher for row-major inputs.
  nntrainer::sgemm(0, TransA, TransB, M, N, K, alpha, A_fp16.data(), lda,
                   B_fp16.data(), ldb, beta, C_fp16.data(), ldc);

  if (M == 0 || N == 0) {
    for (std::size_t i = 0; i < c_size; ++i) {
      EXPECT_EQ(static_cast<float>(C_fp16[i]), static_cast<float>(C_before[i]))
        << "zero-dimension GEMM touched C at i=" << i << " M=" << M
        << " N=" << N << " K=" << K << " TransA=" << TransA
        << " TransB=" << TransB;
    }
    return;
  }

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
      const float got = static_cast<float>(C_fp16[idx]);
      const float ref = C_fp32_ref[idx];
      const float abs_diff = std::abs(got - ref);
      const float rel_diff = abs_diff / std::max(1.0F, std::abs(ref));
      EXPECT_TRUE(abs_diff <= 1e-2F || rel_diff <= 1e-2F)
        << "mismatch at m=" << m << " n=" << n << " M=" << M << " N=" << N
        << " K=" << K << " TransA=" << TransA << " TransB=" << TransB
        << " alpha=" << alpha << " beta=" << beta << " lda=" << lda
        << " ldb=" << ldb << " ldc=" << ldc << " got=" << got << " ref=" << ref
        << " abs_diff=" << abs_diff << " rel_diff=" << rel_diff;
    }

    for (unsigned int n = N; n < ldc; ++n) {
      const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
      EXPECT_EQ(static_cast<float>(C_fp16[idx]),
                static_cast<float>(C_before[idx]))
        << "C padding was modified at m=" << m << " n=" << n << " ldc=" << ldc;
    }
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_aligned_12x32x32) {
  run_sgemm_fp16_hgemm_test(12, 32, 32);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_aligned_256x512x128) {
  run_sgemm_fp16_hgemm_test(256, 512, 128);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_aligned_64x64x64) {
  run_sgemm_fp16_hgemm_test(64, 64, 64);
}

// Large shapes that exceed the 8M-FLOP parallel threshold and force the blocked
// path's panel subdivision: square (N-subdivision) and tall-thin
// (M-subdivision). Run the whole binary under NNTR_NUM_THREADS>1 to exercise
// the multi-threaded panel grid.
TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_parallel_square_512x512x512) {
  run_sgemm_fp16_hgemm_test(512, 512, 512);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_parallel_tall_thin_1024x96x256) {
  run_sgemm_fp16_hgemm_test(1024, 96, 256);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_parallel_wide_96x1024x256) {
  run_sgemm_fp16_hgemm_test(96, 1024, 256);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_parallel_square_alpha_beta_640x384x320) {
  run_sgemm_fp16_hgemm_test(640, 384, 320, false, false, 0.75F, 0.5F);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_parallel_transAB_384x320x512) {
  run_sgemm_fp16_hgemm_test(384, 320, 512, true, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_unaligned_7x17x33) {
  // Exercises both M-edge (7 = 6 + 1) and N-edge (17 = 16 + 1) cleanup.
  run_sgemm_fp16_hgemm_test(7, 17, 33);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_unaligned_13x33x65) {
  run_sgemm_fp16_hgemm_test(13, 33, 65);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_noTrans_alpha_beta_13x33x65) {
  run_sgemm_fp16_hgemm_test(13, 33, 65, false, false, -0.75F, 0.25F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_transA_unaligned_13x33x65) {
  run_sgemm_fp16_hgemm_test(13, 33, 65, true, false, 0.5F, -0.125F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_transB_unaligned_13x33x65) {
  run_sgemm_fp16_hgemm_test(13, 33, 65, false, true, 1.25F, 0.5F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_transAB_unaligned_13x33x65) {
  run_sgemm_fp16_hgemm_test(13, 33, 65, true, true, -1.0F, 0.125F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_alpha_beta_boundary_cases) {
  /// @brief parameter set for one sub-case of this test
  struct Case {
    float alpha;
    float beta;
  };

  const std::vector<Case> cases = {
    {0.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {2.5F, -1.0F}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("alpha=" + std::to_string(tc.alpha) +
                 " beta=" + std::to_string(tc.beta));
    run_sgemm_fp16_hgemm_test(13, 33, 65, false, false, tc.alpha, tc.beta);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_zero_dimension_cases) {
  run_sgemm_fp16_hgemm_test(0, 17, 33, false, false, 1.0F, 0.0F);
  run_sgemm_fp16_hgemm_test(7, 0, 33, false, false, 1.0F, 0.0F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_zero_k_beta_cases) {
  /// @brief parameter set for one sub-case of this test
  struct Case {
    float beta;
  };

  const std::vector<Case> cases = {{0.0F}, {1.0F}, {-0.5F}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("K=0 beta=" + std::to_string(tc.beta));
    run_sgemm_fp16_hgemm_test(7, 17, 0, false, false, 1.0F, tc.beta);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_prime_size_all_transposes) {
  /// @brief parameter set for one sub-case of this test
  struct Case {
    bool trans_a;
    bool trans_b;
  };

  const std::vector<Case> cases = {
    {false, false}, {false, true}, {true, false}, {true, true}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("TransA=" + std::to_string(tc.trans_a) +
                 " TransB=" + std::to_string(tc.trans_b));
    run_sgemm_fp16_hgemm_test(97, 53, 71, tc.trans_a, tc.trans_b, 0.75F,
                              -0.25F);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_small_tile_cases) {
  run_sgemm_fp16_hgemm_test(1, 1, 1, false, false, 1.0F, 0.0F);
  run_sgemm_fp16_hgemm_test(5, 15, 7, false, false, 1.25F, -0.5F);
  run_sgemm_fp16_hgemm_test(5, 15, 7, true, true, -0.75F, 1.0F);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_row_fast_path_cases) {
  run_sgemm_fp16_hgemm_test(1, 257, 129, false, false, 0.75F, -0.25F, 3, 5, 7);
  run_sgemm_fp16_hgemm_test(1, 257, 129, true, false, -0.5F, 0.125F, 2, 4, 3);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_small_k_notrans_large_padded) {
  run_sgemm_fp16_hgemm_test(129, 257, 4, false, false, -0.75F, 0.25F, 3, 5, 7);
  run_sgemm_fp16_hgemm_test(129, 257, 4, true, false, 0.5F, -0.125F, 2, 6, 3);
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_skinny_direct_cases) {
  run_sgemm_fp16_hgemm_test(129, 1, 67, false, false, 1.25F, -0.5F, 5, 3, 2);
  run_sgemm_fp16_hgemm_test(97, 2, 71, true, true, -0.75F, 0.25F, 4, 6, 3);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_skinny_fast_path_all_transposes) {
  /// @brief parameter set for one sub-case of this test
  struct Case {
    unsigned int n;
    bool trans_a;
    bool trans_b;
  };

  const std::vector<Case> cases = {
    {1, false, false}, {1, false, true}, {1, true, false}, {1, true, true},
    {2, false, false}, {2, false, true}, {2, true, false}, {2, true, true}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("N=" + std::to_string(tc.n) +
                 " TransA=" + std::to_string(tc.trans_a) +
                 " TransB=" + std::to_string(tc.trans_b));
    run_sgemm_fp16_hgemm_test(131, tc.n, 67, tc.trans_a, tc.trans_b, 0.75F,
                              -0.25F, 3, 5, 7);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemm_fp16_row_transB_fast_path_cases) {
  run_sgemm_fp16_hgemm_test(1, 257, 129, false, true, 0.75F, -0.25F, 3, 5, 7);
  run_sgemm_fp16_hgemm_test(1, 257, 129, true, true, -0.5F, 0.125F, 2, 4, 3);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_padded_lda_ldb_ldc_all_transposes) {
  /// @brief parameter set for one sub-case of this test
  struct Case {
    bool trans_a;
    bool trans_b;
    float alpha;
    float beta;
    unsigned int lda_extra;
    unsigned int ldb_extra;
    unsigned int ldc_extra;
  };

  const std::vector<Case> cases = {
    {false, false, 1.0F, 0.0F, 4, 7, 5},
    {false, true, -0.5F, 0.125F, 6, 2, 1},
    {true, false, 0.75F, -0.25F, 3, 8, 4},
    {true, true, -1.25F, 0.5F, 5, 3, 6},
  };

  for (const auto &tc : cases) {
    SCOPED_TRACE("TransA=" + std::to_string(tc.trans_a) +
                 " TransB=" + std::to_string(tc.trans_b));
    run_sgemm_fp16_hgemm_test(13, 33, 65, tc.trans_a, tc.trans_b, tc.alpha,
                              tc.beta, tc.lda_extra, tc.ldb_extra,
                              tc.ldc_extra);
  }
}

#ifdef X86_HGEMM_WORKSPACE_STATS_AVAILABLE
TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_reuse_warmed_shape) {
  run_sgemm_fp16_hgemm_test(64, 64, 64);

  nntrainer::hgemm::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(64, 64, 64);
  auto same_stats =
    nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(same_stats.total_realloc_count, 0u);

  nntrainer::hgemm::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(7, 17, 33);
  auto smaller_stats =
    nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(smaller_stats.total_realloc_count, 0u);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_uses_c32_panel_and_packed_panels) {
  nntrainer::hgemm::internal::testing::clear_hgemm_workspace();

  const auto &block = nntrainer::hgemm::internal::get_hgemm_block_sizes();
  const unsigned int M = block.m + 1;
  const unsigned int N = block.n + 1;
  const unsigned int K = 5;

  run_sgemm_fp16_hgemm_test(M, N, K, false, false, 0.75F, 0.25F);
  const auto stats =
    nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();

  const auto round_up_to = [](unsigned int value, unsigned int tile) {
    return ((value + tile - 1) / tile) * tile;
  };

  const unsigned int panel_m = std::min<unsigned int>(M, block.m);
  const unsigned int panel_n = std::min<unsigned int>(N, block.n);
  const unsigned int k_block = std::min<unsigned int>(K, block.k);
  const unsigned int tile_m =
    std::min<unsigned int>(panel_m, std::min(block.m, block.c_m));

  const std::size_t expected_c32_capacity =
    static_cast<std::size_t>(round_up_to(panel_m, X86_HGEMM_MR)) *
    round_up_to(panel_n, X86_HGEMM_NR);
  const std::size_t full_c32_capacity =
    static_cast<std::size_t>(round_up_to(M, X86_HGEMM_MR)) *
    round_up_to(N, X86_HGEMM_NR);
  const std::size_t expected_pack_a_capacity =
    static_cast<std::size_t>(round_up_to(tile_m, X86_HGEMM_MR)) * k_block;
  const std::size_t expected_pack_b_capacity =
    static_cast<std::size_t>(k_block) * round_up_to(panel_n, X86_HGEMM_NR);

  EXPECT_EQ(stats.c32_capacity, expected_c32_capacity);
  EXPECT_LT(stats.c32_capacity, full_c32_capacity);
  EXPECT_EQ(stats.pack_a_capacity, expected_pack_a_capacity);
  EXPECT_EQ(stats.pack_b_capacity, expected_pack_b_capacity);
  EXPECT_EQ(stats.scratch_capacity, 0u);
  EXPECT_EQ(stats.total_capacity_bytes,
            (stats.c32_capacity + stats.pack_a_capacity +
             stats.pack_b_capacity + stats.scratch_capacity) *
              sizeof(float));
  EXPECT_EQ(stats.total_realloc_count,
            stats.c32_realloc_count + stats.pack_a_realloc_count +
              stats.pack_b_realloc_count + stats.scratch_realloc_count);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_row_fast_path_uses_scratch_only) {
  nntrainer::hgemm::internal::testing::clear_hgemm_workspace();

  const unsigned int N = 257;
  run_sgemm_fp16_hgemm_test(1, N, 129, false, false, 0.75F, -0.25F, 3, 5, 7);
  auto stats = nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();

  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, N);
  EXPECT_EQ(stats.c32_realloc_count, 0u);
  EXPECT_EQ(stats.pack_a_realloc_count, 0u);
  EXPECT_EQ(stats.pack_b_realloc_count, 0u);
  EXPECT_EQ(stats.scratch_realloc_count, 1u);
  EXPECT_EQ(stats.total_realloc_count, 1u);
  EXPECT_EQ(stats.total_capacity_bytes, stats.scratch_capacity * sizeof(float));

  nntrainer::hgemm::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(1, N, 129, true, false, -0.5F, 0.125F, 2, 4, 3);
  stats = nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();

  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, N);
  EXPECT_EQ(stats.scratch_realloc_count, 0u);
  EXPECT_EQ(stats.total_realloc_count, 0u);
  EXPECT_EQ(stats.total_capacity_bytes, stats.scratch_capacity * sizeof(float));
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_row_transB_fast_path_uses_scratch_only) {
  nntrainer::hgemm::internal::testing::clear_hgemm_workspace();

  const unsigned int K = 129;
  run_sgemm_fp16_hgemm_test(1, 257, K, false, true, 0.75F, -0.25F, 3, 5, 7);
  auto stats = nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();

  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, K);
  EXPECT_EQ(stats.c32_realloc_count, 0u);
  EXPECT_EQ(stats.pack_a_realloc_count, 0u);
  EXPECT_EQ(stats.pack_b_realloc_count, 0u);
  EXPECT_EQ(stats.scratch_realloc_count, 1u);
  EXPECT_EQ(stats.total_realloc_count, 1u);
  EXPECT_EQ(stats.total_capacity_bytes, stats.scratch_capacity * sizeof(float));

  nntrainer::hgemm::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(1, 257, K, true, true, -0.5F, 0.125F, 2, 4, 3);
  stats = nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();

  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, K);
  EXPECT_EQ(stats.scratch_realloc_count, 0u);
  EXPECT_EQ(stats.total_realloc_count, 0u);
  EXPECT_EQ(stats.total_capacity_bytes, stats.scratch_capacity * sizeof(float));
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_no_allocation_for_fast_small_shapes) {
  nntrainer::hgemm::internal::testing::clear_hgemm_workspace();
  run_sgemm_fp16_hgemm_test(129, 257, 4, false, false, -0.75F, 0.25F, 3, 5, 7);
  auto stats = nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, 0u);
  EXPECT_EQ(stats.total_realloc_count, 0u);

  nntrainer::hgemm::internal::testing::clear_hgemm_workspace();
  run_sgemm_fp16_hgemm_test(131, 2, 67, true, true, 0.75F, -0.25F, 3, 5, 7);
  stats = nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(stats.c32_capacity, 0u);
  EXPECT_EQ(stats.pack_a_capacity, 0u);
  EXPECT_EQ(stats.pack_b_capacity, 0u);
  EXPECT_EQ(stats.scratch_capacity, 0u);
  EXPECT_EQ(stats.total_realloc_count, 0u);
}

template <typename SrcT> static void run_packing_B_N16_trans_test() {
  const unsigned int K = 13;
  const unsigned int N = X86_HGEMM_NR;
  const unsigned int stride = K + 5;
  auto src = generate_random_vector<SrcT>(static_cast<std::size_t>(N) * stride,
                                          -1.0F, 1.0F);
  std::vector<float> dst(static_cast<std::size_t>(K) * X86_HGEMM_NR, -123.0F);

  nntrainer::hgemm::internal::packing_B_N16_trans(K, N, src.data(), stride,
                                                  dst.data());

  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int n = 0; n < N; ++n) {
      EXPECT_FLOAT_EQ(
        dst[static_cast<std::size_t>(k) * X86_HGEMM_NR + n],
        static_cast<float>(src[static_cast<std::size_t>(n) * stride + k]))
        << "mismatch at k=" << k << " n=" << n;
    }
  }
}

TEST(nntrainer_cpu_backend_standalone, hgemm_pack_B_N16_trans_fp16) {
  run_packing_B_N16_trans_test<_FP16>();
}

TEST(nntrainer_cpu_backend_standalone, hgemm_pack_B_N16_trans_float) {
  run_packing_B_N16_trans_test<float>();
}

TEST(nntrainer_cpu_backend_standalone,
     sgemm_fp16_workspace_no_allocation_for_degenerate_paths) {
  nntrainer::hgemm::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(13, 33, 65, false, false, 0.0F, 0.5F);
  auto alpha_zero_stats =
    nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(alpha_zero_stats.total_realloc_count, 0u);

  nntrainer::hgemm::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(13, 33, 0, false, false, 1.0F, -0.5F);
  auto k_zero_stats =
    nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(k_zero_stats.total_realloc_count, 0u);

  nntrainer::hgemm::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(0, 33, 65, false, false, 1.0F, 0.0F);
  auto m_zero_stats =
    nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(m_zero_stats.total_realloc_count, 0u);

  nntrainer::hgemm::internal::testing::reset_hgemm_workspace_stats();
  run_sgemm_fp16_hgemm_test(13, 0, 65, false, false, 1.0F, 0.0F);
  auto n_zero_stats =
    nntrainer::hgemm::internal::testing::get_hgemm_workspace_stats();
  EXPECT_EQ(n_zero_stats.total_realloc_count, 0u);
}
#endif

/// FP16 sgemv path: exercises the x86 AVX2 hgemv kernel.
/// Reference is a local FP32 GEMV loop to avoid using the optimized backend
/// as the oracle for this backend test.
static void run_sgemv_fp16_hgemv_test(unsigned int M, unsigned int N,
                                      bool TransA = false, float alpha = 1.0F,
                                      float beta = 0.0F,
                                      unsigned int lda_extra = 0,
                                      unsigned int incX = 1,
                                      unsigned int incY = 1) {
  nntrainer::init_backend();

  const unsigned int lda = std::max(1u, N + lda_extra);
  const unsigned int lenX = TransA ? M : N;
  const unsigned int lenY = TransA ? N : M;

  const std::size_t a_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(M) * lda);
  const std::size_t x_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenX) * std::max(incX, 1u));
  const std::size_t y_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenY) * std::max(incY, 1u));

  auto A_fp16 = generate_random_vector<_FP16>(a_size, -0.25F, 0.25F);
  auto X_fp16 = generate_random_vector<_FP16>(x_size, -0.25F, 0.25F);
  auto Y_fp16 = generate_random_vector<_FP16>(y_size, -0.25F, 0.25F);
  auto Y_before = Y_fp16;

  std::vector<float> Y_fp32_ref(y_size);
  for (std::size_t i = 0; i < y_size; ++i) {
    Y_fp32_ref[i] = static_cast<float>(Y_before[i]);
  }

  if (M != 0 && N != 0) {
    if (!TransA) {
      for (unsigned int i = 0; i < M; ++i) {
        float acc = 0.0F;
        for (unsigned int j = 0; j < N; ++j) {
          acc += static_cast<float>(A_fp16[i * lda + j]) *
                 static_cast<float>(X_fp16[j * incX]);
        }
        const std::size_t y_idx = static_cast<std::size_t>(i) * incY;
        Y_fp32_ref[y_idx] = alpha * acc + beta * Y_fp32_ref[y_idx];
      }
    } else {
      for (unsigned int j = 0; j < N; ++j) {
        float acc = 0.0F;
        for (unsigned int i = 0; i < M; ++i) {
          acc += static_cast<float>(A_fp16[i * lda + j]) *
                 static_cast<float>(X_fp16[i * incX]);
        }
        const std::size_t y_idx = static_cast<std::size_t>(j) * incY;
        Y_fp32_ref[y_idx] = alpha * acc + beta * Y_fp32_ref[y_idx];
      }
    }
  }

  // System under test: FP16 sgemv, routed to x86::hgemv by the x86 backend
  // dispatcher for row-major inputs.
  nntrainer::sgemv(0, TransA, M, N, alpha, A_fp16.data(), lda, X_fp16.data(),
                   incX, beta, Y_fp16.data(), incY);

  if (M == 0 || N == 0) {
    for (std::size_t i = 0; i < y_size; ++i) {
      EXPECT_EQ(static_cast<float>(Y_fp16[i]), static_cast<float>(Y_before[i]))
        << "zero-dimension GEMV touched Y at i=" << i << " M=" << M
        << " N=" << N << " TransA=" << TransA;
    }
    return;
  }

  for (unsigned int k = 0; k < lenY; ++k) {
    const std::size_t y_idx = static_cast<std::size_t>(k) * incY;
    const float got = static_cast<float>(Y_fp16[y_idx]);
    const float ref = Y_fp32_ref[y_idx];
    const float abs_diff = std::abs(got - ref);
    const float rel_diff = abs_diff / std::max(1.0F, std::abs(ref));
    EXPECT_TRUE(abs_diff <= 1e-2F || rel_diff <= 1e-2F)
      << "mismatch at k=" << k << " M=" << M << " N=" << N
      << " TransA=" << TransA << " alpha=" << alpha << " beta=" << beta
      << " lda=" << lda << " incX=" << incX << " incY=" << incY
      << " got=" << got << " ref=" << ref << " abs_diff=" << abs_diff
      << " rel_diff=" << rel_diff;
  }

  // Y gap slots (only present when incY > 1) must be untouched.
  if (incY > 1) {
    for (unsigned int k = 0; k < lenY; ++k) {
      const std::size_t base = static_cast<std::size_t>(k) * incY;
      for (unsigned int g = 1; g < incY; ++g) {
        const std::size_t gap_idx = base + g;
        if (gap_idx >= y_size) {
          break;
        }
        EXPECT_EQ(static_cast<float>(Y_fp16[gap_idx]),
                  static_cast<float>(Y_before[gap_idx]))
          << "Y gap was modified at idx=" << gap_idx << " k=" << k << " g=" << g
          << " incY=" << incY;
      }
    }
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_aligned_8x64) {
  // N multiple of 16: 16-element main loop only, no tails.
  run_sgemv_fp16_hgemv_test(8, 64);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_aligned_16x128) {
  run_sgemv_fp16_hgemv_test(16, 128);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_n8_tail_5x24) {
  // N=24 = 16 + 8: exercises the 8-element tail after the 16-element loop.
  run_sgemv_fp16_hgemv_test(5, 24);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_scalar_tail_7x31) {
  // N=31 = 16 + 8 + 7: exercises 16-loop, 8-tail, then scalar tail of 7.
  run_sgemv_fp16_hgemv_test(7, 31);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemv_fp16_noTrans_scalar_tail_only_5x9) {
  // N=9: skips 16-loop and 8-tail, scalar tail of 9.
  run_sgemv_fp16_hgemv_test(5, 9);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_transA_aligned_16x64) {
  run_sgemv_fp16_hgemv_test(16, 64, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_transA_unaligned_13x33) {
  // M=13 (AXPY row sweep tail), N=33 (8-element tail + scalar tail in
  // the FP32-scratch axpy update).
  run_sgemv_fp16_hgemv_test(13, 33, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_alpha_beta_13x33) {
  run_sgemv_fp16_hgemv_test(13, 33, false, -0.75F, 0.25F);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_transA_alpha_beta_13x33) {
  run_sgemv_fp16_hgemv_test(13, 33, true, 0.5F, -0.125F);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_lda_noTrans_7x17) {
  // lda = N + 7: row-padded A.
  run_sgemv_fp16_hgemv_test(7, 17, false, 1.0F, 0.0F, 7);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_lda_transA_7x17) {
  run_sgemv_fp16_hgemv_test(7, 17, true, 1.0F, 0.0F, 7);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_incX_noTrans_7x17) {
  // incX=2 forces the non-contiguous X path.
  run_sgemv_fp16_hgemv_test(7, 17, false, 1.0F, 0.0F, 0, 2, 1);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_incY_noTrans_7x17) {
  // incY=3 exercises Y-gap preservation in the writeback path.
  run_sgemv_fp16_hgemv_test(7, 17, false, 1.0F, 0.0F, 0, 1, 3);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemv_fp16_strided_lda_incX_incY_transA_7x17) {
  // All three strides at once on the TransA path (non-contiguous Y32 init,
  // non-contiguous incX read, non-contiguous incY writeback).
  run_sgemv_fp16_hgemv_test(7, 17, true, 0.5F, -0.25F, 3, 2, 4);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_alpha_beta_boundary_cases) {
  /// @brief parameter set for one sub-case of this test
  struct Case {
    float alpha;
    float beta;
  };

  const std::vector<Case> cases = {
    {0.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {2.5F, -1.0F}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("alpha=" + std::to_string(tc.alpha) +
                 " beta=" + std::to_string(tc.beta));
    run_sgemv_fp16_hgemv_test(13, 33, false, tc.alpha, tc.beta);
    run_sgemv_fp16_hgemv_test(13, 33, true, tc.alpha, tc.beta);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_zero_dimension_cases) {
  run_sgemv_fp16_hgemv_test(0, 17, false, 1.0F, 0.0F);
  run_sgemv_fp16_hgemv_test(7, 0, false, 1.0F, 0.0F);
  run_sgemv_fp16_hgemv_test(0, 17, true, 1.0F, 0.0F);
  run_sgemv_fp16_hgemv_test(7, 0, true, 1.0F, 0.0F);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_single_row_or_col) {
  run_sgemv_fp16_hgemv_test(1, 33, false);
  run_sgemv_fp16_hgemv_test(33, 1, false);
  run_sgemv_fp16_hgemv_test(1, 33, true);
  run_sgemv_fp16_hgemv_test(33, 1, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_beta_zero_does_not_read_Y) {
  // BLAS rule: when beta == 0, Y must not be read. Seed Y with NaN; if the
  // kernel reads it, NaN propagates into the result.
  nntrainer::init_backend();

  const _FP16 nan_v = static_cast<_FP16>(std::nan(""));

  auto check_path = [&](bool TransA) {
    const unsigned int M = 7;
    const unsigned int N = 17;
    const unsigned int lenY = TransA ? N : M;

    auto A = generate_random_vector<_FP16>(M * N, -0.25F, 0.25F);
    auto X = generate_random_vector<_FP16>(TransA ? M : N, -0.25F, 0.25F);
    std::vector<_FP16> Y(lenY, nan_v);

    nntrainer::sgemv(0, TransA, M, N, 1.0F, A.data(), N, X.data(), 1, 0.0F,
                     Y.data(), 1);

    for (unsigned int k = 0; k < lenY; ++k) {
      const float yv = static_cast<float>(Y[k]);
      EXPECT_FALSE(std::isnan(yv)) << "beta=0 path read NaN-seeded Y at k=" << k
                                   << " TransA=" << TransA << " value=" << yv;
    }
  };

  check_path(false);
  check_path(true);
}

/// Mixed-precision GEMM path (shgemm: A=FP32,B=FP16 / hsgemm: A=FP16,B=FP32),
/// FP32 output. Reference accumulates in double so the only approximation
/// under test is the kernel's FP32 blocked summation; both sides read the
/// identical generated operands, so operand rounding is not a discrepancy.
template <typename AType, typename BType>
static void run_mixed_sgemm_test(unsigned int M, unsigned int N, unsigned int K,
                                 bool TransA = false, bool TransB = false,
                                 float alpha = 1.0F, float beta = 0.0F,
                                 unsigned int lda_extra = 0,
                                 unsigned int ldb_extra = 0,
                                 unsigned int ldc_extra = 0) {
  static_assert(std::is_same_v<AType, float> != std::is_same_v<BType, float>,
                "exactly one operand must be FP32 for mixed-precision GEMM");
  nntrainer::init_backend();

  const unsigned int lda = std::max(1u, (TransA ? M : K) + lda_extra);
  const unsigned int ldb = std::max(1u, (TransB ? K : N) + ldb_extra);
  const unsigned int ldc = std::max(1u, N + ldc_extra);
  const unsigned int a_rows = TransA ? K : M;
  const unsigned int b_rows = TransB ? N : K;
  const std::size_t a_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(a_rows) * lda);
  const std::size_t b_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(b_rows) * ldb);
  const std::size_t c_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(M) * ldc);

  auto A = generate_random_vector<AType>(a_size, -0.25F, 0.25F);
  auto B = generate_random_vector<BType>(b_size, -0.25F, 0.25F);
  auto C = generate_random_vector<float>(c_size, -0.25F, 0.25F);
  auto C_before = C;

  std::vector<float> C_ref(c_size);
  for (std::size_t i = 0; i < c_size; ++i) {
    C_ref[i] = C_before[i];
  }

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (unsigned int k = 0; k < K; ++k) {
        const double a = static_cast<double>(
          static_cast<float>(TransA ? A[k * lda + m] : A[m * lda + k]));
        const double b = static_cast<double>(
          static_cast<float>(TransB ? B[n * ldb + k] : B[k * ldb + n]));
        acc += a * b;
      }
      const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
      C_ref[idx] = static_cast<float>(static_cast<double>(alpha) * acc +
                                      static_cast<double>(beta) * C_ref[idx]);
    }
  }

  if constexpr (std::is_same_v<AType, float>) {
    nntrainer::shgemm(0, TransA, TransB, M, N, K, alpha, A.data(), lda,
                      B.data(), ldb, beta, C.data(), ldc);
  } else {
    nntrainer::hsgemm(0, TransA, TransB, M, N, K, alpha, A.data(), lda,
                      B.data(), ldb, beta, C.data(), ldc);
  }

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
      const float got = C[idx];
      const float ref = C_ref[idx];
      const float abs_diff = std::abs(got - ref);
      const float rel_diff = abs_diff / std::max(1.0F, std::abs(ref));
      EXPECT_TRUE(abs_diff <= 1e-3F || rel_diff <= 1e-3F)
        << "mismatch at m=" << m << " n=" << n << " M=" << M << " N=" << N
        << " K=" << K << " TransA=" << TransA << " TransB=" << TransB
        << " alpha=" << alpha << " beta=" << beta << " got=" << got
        << " ref=" << ref << " abs_diff=" << abs_diff
        << " rel_diff=" << rel_diff;
    }
    for (unsigned int n = N; n < ldc; ++n) {
      const std::size_t idx = static_cast<std::size_t>(m) * ldc + n;
      EXPECT_FLOAT_EQ(C[idx], C_before[idx])
        << "C padding was modified at m=" << m << " n=" << n << " ldc=" << ldc;
    }
  }
}

TEST(nntrainer_cpu_backend_standalone, shgemm_fp32xfp16_noTrans_64x64x64) {
  run_mixed_sgemm_test<float, _FP16>(64, 64, 64);
}
TEST(nntrainer_cpu_backend_standalone, shgemm_fp32xfp16_noTrans_256x512x128) {
  run_mixed_sgemm_test<float, _FP16>(256, 512, 128);
}
TEST(nntrainer_cpu_backend_standalone, shgemm_fp32xfp16_unaligned_13x33x65) {
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65);
}
TEST(nntrainer_cpu_backend_standalone,
     shgemm_fp32xfp16_all_transposes_13x33x65) {
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65, true, false);
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65, false, true);
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65, true, true);
}
TEST(nntrainer_cpu_backend_standalone, shgemm_fp32xfp16_alpha_beta_padded) {
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65, false, false, 0.5F, 0.25F, 4,
                                     7, 5);
  run_mixed_sgemm_test<float, _FP16>(13, 33, 65, true, true, -1.25F, 0.5F, 5, 3,
                                     6);
}
TEST(nntrainer_cpu_backend_standalone, hsgemm_fp16xfp32_noTrans_64x64x64) {
  run_mixed_sgemm_test<_FP16, float>(64, 64, 64);
}
TEST(nntrainer_cpu_backend_standalone, hsgemm_fp16xfp32_noTrans_256x512x128) {
  run_mixed_sgemm_test<_FP16, float>(256, 512, 128);
}
TEST(nntrainer_cpu_backend_standalone, hsgemm_fp16xfp32_unaligned_13x33x65) {
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65);
}
TEST(nntrainer_cpu_backend_standalone,
     hsgemm_fp16xfp32_all_transposes_13x33x65) {
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65, true, false);
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65, false, true);
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65, true, true);
}
TEST(nntrainer_cpu_backend_standalone, hsgemm_fp16xfp32_alpha_beta_padded) {
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65, false, false, 0.5F, 0.25F, 4,
                                     7, 5);
  run_mixed_sgemm_test<_FP16, float>(13, 33, 65, true, true, -1.25F, 0.5F, 5, 3,
                                     6);
}

/// Mixed-precision GEMV path (shgemv: A=FP32,X=FP16 / hsgemv: A=FP16,X=FP32),
/// FP32 output. Reference accumulates in double; see run_mixed_sgemm_test.
template <typename MatT, typename VecT>
static void run_mixed_sgemv_test(unsigned int M, unsigned int N,
                                 bool TransA = false, float alpha = 1.0F,
                                 float beta = 0.0F, unsigned int lda_extra = 0,
                                 unsigned int incX = 1, unsigned int incY = 1) {
  static_assert(std::is_same_v<MatT, float> != std::is_same_v<VecT, float>,
                "exactly one operand must be FP32 for mixed-precision GEMV");
  nntrainer::init_backend();

  const unsigned int lda = std::max(1u, N + lda_extra);
  const unsigned int lenX = TransA ? M : N;
  const unsigned int lenY = TransA ? N : M;

  const std::size_t a_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(M) * lda);
  const std::size_t x_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenX) * std::max(incX, 1u));
  const std::size_t y_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenY) * std::max(incY, 1u));

  auto A = generate_random_vector<MatT>(a_size, -0.25F, 0.25F);
  auto X = generate_random_vector<VecT>(x_size, -0.25F, 0.25F);
  auto Y = generate_random_vector<float>(y_size, -0.25F, 0.25F);
  auto Y_before = Y;

  std::vector<float> Y_ref(y_size);
  for (std::size_t i = 0; i < y_size; ++i) {
    Y_ref[i] = Y_before[i];
  }

  if (!TransA) {
    for (unsigned int i = 0; i < M; ++i) {
      double acc = 0.0;
      for (unsigned int j = 0; j < N; ++j) {
        acc += static_cast<double>(static_cast<float>(A[i * lda + j])) *
               static_cast<double>(static_cast<float>(X[j * incX]));
      }
      const std::size_t y_idx = static_cast<std::size_t>(i) * incY;
      Y_ref[y_idx] =
        static_cast<float>(static_cast<double>(alpha) * acc +
                           static_cast<double>(beta) * Y_ref[y_idx]);
    }
  } else {
    for (unsigned int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (unsigned int i = 0; i < M; ++i) {
        acc += static_cast<double>(static_cast<float>(A[i * lda + j])) *
               static_cast<double>(static_cast<float>(X[i * incX]));
      }
      const std::size_t y_idx = static_cast<std::size_t>(j) * incY;
      Y_ref[y_idx] =
        static_cast<float>(static_cast<double>(alpha) * acc +
                           static_cast<double>(beta) * Y_ref[y_idx]);
    }
  }

  if constexpr (std::is_same_v<MatT, float>) {
    nntrainer::shgemv(0, TransA, M, N, alpha, A.data(), lda, X.data(), incX,
                      beta, Y.data(), incY);
  } else {
    nntrainer::hsgemv(0, TransA, M, N, alpha, A.data(), lda, X.data(), incX,
                      beta, Y.data(), incY);
  }

  for (unsigned int k = 0; k < lenY; ++k) {
    const std::size_t y_idx = static_cast<std::size_t>(k) * incY;
    const float got = Y[y_idx];
    const float ref = Y_ref[y_idx];
    const float abs_diff = std::abs(got - ref);
    const float rel_diff = abs_diff / std::max(1.0F, std::abs(ref));
    EXPECT_TRUE(abs_diff <= 1e-3F || rel_diff <= 1e-3F)
      << "mismatch at k=" << k << " M=" << M << " N=" << N
      << " TransA=" << TransA << " alpha=" << alpha << " beta=" << beta
      << " incX=" << incX << " incY=" << incY << " got=" << got
      << " ref=" << ref << " abs_diff=" << abs_diff << " rel_diff=" << rel_diff;
  }
}

TEST(nntrainer_cpu_backend_standalone, shgemv_fp32xfp16_noTrans_16x128) {
  run_mixed_sgemv_test<float, _FP16>(16, 128);
}
TEST(nntrainer_cpu_backend_standalone,
     shgemv_fp32xfp16_transA_unaligned_13x33) {
  run_mixed_sgemv_test<float, _FP16>(13, 33, true);
}
TEST(nntrainer_cpu_backend_standalone, shgemv_fp32xfp16_alpha_beta_13x33) {
  run_mixed_sgemv_test<float, _FP16>(13, 33, false, 0.5F, 0.25F);
  run_mixed_sgemv_test<float, _FP16>(13, 33, true, 0.5F, 0.25F);
}
TEST(nntrainer_cpu_backend_standalone, hsgemv_fp16xfp32_noTrans_16x128) {
  run_mixed_sgemv_test<_FP16, float>(16, 128);
}
TEST(nntrainer_cpu_backend_standalone,
     hsgemv_fp16xfp32_transA_unaligned_13x33) {
  run_mixed_sgemv_test<_FP16, float>(13, 33, true);
}
TEST(nntrainer_cpu_backend_standalone,
     hsgemv_fp16xfp32_strided_incX_incY_7x17) {
  run_mixed_sgemv_test<_FP16, float>(7, 17, false, 1.0F, 0.0F, 3, 2, 2);
  run_mixed_sgemv_test<float, _FP16>(7, 17, true, 0.75F, 0.5F, 0, 2, 2);
}

#endif // defined(__x86_64__) || defined(_M_X64)

int main(int argc, char **argv) {
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
