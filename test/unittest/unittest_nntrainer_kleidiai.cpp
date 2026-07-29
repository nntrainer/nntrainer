// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_kleidiai.cpp
 * @date	04 June 2026
 * @brief	This is unittest for kleidiai
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jaemin Shin <jaemin980311@gmail.com>
 * @bug		No known bugs except for NYI items
 */

#include <cfloat>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "int4_utils.h"
#include "kai/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "kai/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon.h"
#include "kleidiai_interface.h"
#include "nntrainer_test_util.h"
#include <cpu_backend.h>
#include <fallback_internal.h>
#include <gtest/gtest.h>
#include <neon_impl.h>
#include <thread_manager.h>

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;
using std::chrono::steady_clock;

#define QK4_0 32
/**
 * @brief q4_0 block
 */
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0_testonly;

/**
 * @brief gemm_qai8dxp_qsi4cxp_rhs_unpacked
 */
class gemm_qai8dxp_qsi4cxp_rhs_unpacked
  : public ::testing::TestWithParam<std::tuple<bool, size_t, size_t, size_t>> {
};

TEST_P(gemm_qai8dxp_qsi4cxp_rhs_unpacked, check_ukernels) {
  auto [is_nxk, M, N, K] = GetParam();

  // test data
  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> ref_dst(M * N);
  std::vector<float> dst(M * N);

  // quantize rhs weight
  const size_t rhs_native_size_qs4cx = is_nxk
                                         ? N * ((K + 1) / 2) * sizeof(uint8_t)
                                         : K * ((N + 1) / 2) * sizeof(uint8_t);
  const size_t rhs_scales_size_f32 = N * sizeof(float);

  uint8_t *rhs_native_mtx_qs4cx = new uint8_t[rhs_native_size_qs4cx];
  uint8_t *rhs_scales_f32 = new uint8_t[rhs_scales_size_f32];

  nntrainer::quant_qs4cx_f32(N, K, weight.data(), rhs_native_mtx_qs4cx,
                             rhs_scales_f32, is_nxk);

  // calculate reference output
  std::vector<int8_t> lhs_qa8dx(M * (K + sizeof(int32_t) + sizeof(float)));
  nntrainer::__fallback_quant_qa8dx_f32(M, K, activation.data(),
                                        lhs_qa8dx.data());
  if (is_nxk) {
    nntrainer::__fallback_matmul_mxn_mxk_nxk_f32_qa8dx_qs4cx(
      M, N, K, lhs_qa8dx.data(), rhs_native_mtx_qs4cx, (float *)rhs_scales_f32,
      ref_dst.data());
  } else {
    nntrainer::__fallback_matmul_mxn_mxk_kxn_f32_qa8dx_qs4cx(
      M, N, K, lhs_qa8dx.data(), rhs_native_mtx_qs4cx, (float *)rhs_scales_f32,
      ref_dst.data());
  }

  // get number of ukernels
  const size_t num_idx_variants =
    nntrainer::__kai_get_num_ukernel_variants_qai8dxp_qsi4cxp();

  // Run GEMM!
  for (size_t idx_variant = 0; idx_variant < num_idx_variants; idx_variant++) {
    nntrainer::gemm_qai8dxp_qsi4cxp_rhs_unpacked(
      M, N, K, activation.data(), rhs_native_mtx_qs4cx, rhs_scales_f32,
      dst.data(), idx_variant, is_nxk);

    for (size_t i = 0; i < M * N; i++) {
      EXPECT_NEAR(ref_dst[i], dst[i], 1e-4);
    }
  }

  delete[] rhs_native_mtx_qs4cx;
  delete[] rhs_scales_f32;
}

INSTANTIATE_TEST_SUITE_P(
  nntrainer_kleidiai, gemm_qai8dxp_qsi4cxp_rhs_unpacked,
  ::testing::Values(std::make_tuple(true, 1, 512, 3072),
                    std::make_tuple(false, 1, 512, 3072),
                    std::make_tuple(true, 768, 768, 768),
                    std::make_tuple(false, 768, 768, 768),
                    std::make_tuple(true, 512, 2048, 768),
                    std::make_tuple(false, 512, 2048, 768),
                    std::make_tuple(true, 3072, 512, 512),
                    std::make_tuple(false, 3072, 512, 512)),
  [](const ::testing::TestParamInfo<
     gemm_qai8dxp_qsi4cxp_rhs_unpacked::ParamType> &info) {
    bool is_nxk = std::get<0>(info.param);
    size_t M = std::get<1>(info.param);
    size_t N = std::get<2>(info.param);
    size_t K = std::get<3>(info.param);
    std::string ret = std::string((is_nxk) ? "nxk_" : "kxn_") +
                      std::to_string(M) + "_" + std::to_string(N) + "_" +
                      std::to_string(K);
    return ret;
  });

/**
 * @brief gemm_qai8dxp_qsi4cxp
 */
class gemm_qai8dxp_qsi4cxp
  : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

TEST_P(gemm_qai8dxp_qsi4cxp, check_ukernels) {
  auto [M, N, K] = GetParam();
  constexpr float eps = 1e-5;

  // test data
  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> ref_dst(M * N);
  std::vector<float> dst(M * N);

  // quantize rhs weight
  const size_t rhs_native_size_qs4cx = N * ((K + 1) / 2) * sizeof(uint8_t);
  const size_t rhs_scales_size_f32 = N * sizeof(float);

  uint8_t *rhs_native_mtx_qs4cx = new uint8_t[rhs_native_size_qs4cx];
  uint8_t *rhs_scales_f32 = new uint8_t[rhs_scales_size_f32];

  nntrainer::quant_qs4cx_f32(N, K, weight.data(), rhs_native_mtx_qs4cx,
                             rhs_scales_f32, true);

  // calculate reference output
  std::vector<int8_t> lhs_qa8dx(M * (K + sizeof(int32_t) + sizeof(float)));
  nntrainer::__fallback_quant_qa8dx_f32(M, K, activation.data(),
                                        lhs_qa8dx.data());

  nntrainer::__fallback_matmul_mxn_mxk_nxk_f32_qa8dx_qs4cx(
    M, N, K, lhs_qa8dx.data(), rhs_native_mtx_qs4cx, (float *)rhs_scales_f32,
    ref_dst.data());

  // find max packed rhs size and allocate
  const size_t num_idx_variants =
    nntrainer::__kai_get_num_ukernel_variants_qai8dxp_qsi4cxp();
  size_t max_rhs_packed_size = 0;
  for (size_t idx_variant = 0; idx_variant < num_idx_variants; idx_variant++) {
    size_t rhs_packed_size =
      nntrainer::get_rhs_packed_size_qsi4cxp_qs4cxs1s0(N, K, idx_variant, true);
    max_rhs_packed_size = std::max(max_rhs_packed_size, rhs_packed_size);
  }

  uint8_t *rhs_packed_mtx_qs4cx = new uint8_t[max_rhs_packed_size];

  // Run GEMM!
  for (size_t idx_variant = 0; idx_variant < num_idx_variants; idx_variant++) {
    // pack rhs offline
    nntrainer::rhs_pack_qsi4cxp_qs4cxs1s0(N, K, rhs_packed_mtx_qs4cx,
                                          rhs_native_mtx_qs4cx, rhs_scales_f32,
                                          idx_variant, true);

    // check packed gemm only
    nntrainer::gemm_qai8dxp_qsi4cxp(M, N, K, activation.data(),
                                    rhs_packed_mtx_qs4cx, dst.data(),
                                    idx_variant);

    for (size_t i = 0; i < M * N; i++) {
      EXPECT_NEAR(ref_dst[i], dst[i], 1e-4);
    }
  }

  delete[] rhs_packed_mtx_qs4cx;
  delete[] rhs_native_mtx_qs4cx;
  delete[] rhs_scales_f32;
}

INSTANTIATE_TEST_SUITE_P(
  nntrainer_kleidiai, gemm_qai8dxp_qsi4cxp,
  ::testing::Values(std::make_tuple(1, 512, 3072),
                    std::make_tuple(768, 768, 768),
                    std::make_tuple(512, 2048, 768),
                    std::make_tuple(3072, 512, 512)),
  [](const ::testing::TestParamInfo<gemm_qai8dxp_qsi4cxp::ParamType> &info) {
    size_t M = std::get<0>(info.param);
    size_t N = std::get<1>(info.param);
    size_t K = std::get<2>(info.param);
    std::string ret =
      std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);
    return ret;
  });

#ifdef ENABLE_FP16

void test_f16_qai8dxp_qsi4cxp_cosine_similarity(size_t M, size_t N, size_t K) {
  constexpr double cosine_threshold = 0.99;

  std::vector<float> activation_f32 = generate_random_vector<float>(M * K);
  std::vector<__fp16> activation_f16(M * K);
  for (size_t i = 0; i < M * K; i++) {
    activation_f16[i] = static_cast<__fp16>(activation_f32[i]);
  }

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> ref_dst(M * N);
  std::vector<__fp16> dst_f16(M * N);
  std::vector<float> dst(M * N);

  const size_t rhs_native_size_qs4cx = N * ((K + 1) / 2) * sizeof(uint8_t);
  const size_t rhs_scales_size_f32 = N * sizeof(float);

  uint8_t *rhs_native_mtx_qs4cx = new uint8_t[rhs_native_size_qs4cx];
  uint8_t *rhs_scales_f32 = new uint8_t[rhs_scales_size_f32];

  nntrainer::quant_qs4cx_f32(N, K, weight.data(), rhs_native_mtx_qs4cx,
                             rhs_scales_f32, true);

  // Calculate reference output using pure F32xF32->F32 GEMM/GEMV
  if (M == 1) {
    // GEMV: ref_dst[n] = sum_k(activation_f32[k] * weight[n, k])
    for (size_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; k++) {
        sum += activation_f32[k] * weight[n * K + k];
      }
      ref_dst[n] = sum;
    }
  } else {
    // GEMM: ref_dst[m, n] = sum_k(activation_f32[m, k] * weight[n, k])
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
          sum += activation_f32[m * K + k] * weight[n * K + k];
        }
        ref_dst[m * N + n] = sum;
      }
    }
  }

  const size_t num_idx_variants =
    nntrainer::__kai_get_num_ukernel_variants_f16_qai8dxp_qsi4cxp();
  size_t max_rhs_packed_size = 0;
  for (size_t idx_variant = 0; idx_variant < num_idx_variants; idx_variant++) {
    size_t rhs_packed_size =
      nntrainer::__kai_get_rhs_packed_size_f16_qsi4cxp_qs4cxs1s0(
        N, K, idx_variant, true);
    max_rhs_packed_size = std::max(max_rhs_packed_size, rhs_packed_size);
  }

  uint8_t *rhs_packed_mtx_qs4cx = new uint8_t[max_rhs_packed_size];

  // Test each F16 variant
  for (size_t idx_variant = 0; idx_variant < num_idx_variants; idx_variant++) {
    nntrainer::__kai_rhs_pack_f16_qsi4cxp_qs4cxs1s0(
      N, K, rhs_packed_mtx_qs4cx, rhs_native_mtx_qs4cx, rhs_scales_f32,
      idx_variant, true);

    // Direct F16 GEMM call (outputs __fp16)
    nntrainer::__kai_gemm_f16_qai8dxp_qsi4cxp(M, N, K, activation_f16.data(),
                                              rhs_packed_mtx_qs4cx,
                                              dst_f16.data(), idx_variant);

    // Convert F16 output to F32 for cosine similarity comparison
    for (size_t i = 0; i < M * N; i++) {
      dst[i] = static_cast<float>(dst_f16[i]);
    }

    // Compute cosine similarity
    double cosine_sim = cosine_similarity(ref_dst.data(), dst.data(), M * N);

    // Expect high cosine similarity (directional alignment)
    EXPECT_GE(cosine_sim, cosine_threshold)
      << "Low cosine similarity at F16 variant " << idx_variant;
  }

  delete[] rhs_packed_mtx_qs4cx;
  delete[] rhs_native_mtx_qs4cx;
  delete[] rhs_scales_f32;
}

TEST(nntrainer_kleidiai, gemv_f16_qai8dxp_qsi4cxp_cosine_similarity) {
  test_f16_qai8dxp_qsi4cxp_cosine_similarity(1, 512, 3072);
}

TEST(nntrainer_kleidiai, gemm_f16_qai8dxp_qsi4cxp_cosine_similarity) {
  test_f16_qai8dxp_qsi4cxp_cosine_similarity(768, 768, 768);
}

#endif

/**
 * @brief Clean and invalidate the given buffer from the caches.
 */
static void evict_from_caches(const void *buf, size_t bytes) {
  // align down to the line base so the last line is covered even when
  // buf is not 64B-aligned
  uintptr_t p = reinterpret_cast<uintptr_t>(buf) & ~static_cast<uintptr_t>(63);
  const uintptr_t end = reinterpret_cast<uintptr_t>(buf) + bytes;
  for (; p < end; p += 64) {
    __asm__ volatile("dc civac, %0" ::"r"(p) : "memory");
  }
  __asm__ volatile("dsb sy" ::: "memory");
}

/**
 * @brief Benchmark comparison of three GEMM implementations
 *
 * Compares latency of:
 * - nntr_gemm_qai8dxp_qsi4cxp_packed (KleidiAI with channel-wise quant)
 * - gemm_q4_0<float> (GGML-style Q4_0 GEMM)
 */
void run_gemm_benchmark_comparison(const size_t M, const size_t N,
                                   const size_t K,
                                   const size_t warmup_iters = 3,
                                   const size_t test_iters = 5,
                                   bool print = false) {
  nntrainer::init_backend();

  // Generate random data
  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);

  // Output buffers
  std::vector<float> dst_qai8dxp(M * N);
  std::vector<float> dst_q4_0(M * N);

  // ============================================================
  // Setup 1: gemm_q4_0<float> (GGML-style Q4_0)
  // ============================================================
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t q4_0_block_size = 32;
  size_t q4_0_data_size = q4_0_type_size * N / q4_0_block_size;
  q4_0_data_size *= K;
  std::vector<char> q4_0_offline_qWeight(q4_0_data_size);
  nntrainer::quantize_q4_0(weight.data(), (void *)q4_0_offline_qWeight.data(),
                           N, K, nullptr);

  std::vector<char> q4_0_repacked_qWeight(q4_0_data_size);
  nntrainer::repack_q4_0(q4_0_repacked_qWeight.data(),
                         q4_0_offline_qWeight.data(), q4_0_data_size, N, K);

  // ============================================================
  // Setup 2: qai8dxp_qsi4cxp (KleidiAI with dynamic block)
  // ============================================================
  const size_t rhs_native_size_qs4cx = N * ((K + 1) / 2);
  const size_t rhs_scales_size_f32 = N * sizeof(float);

  std::vector<uint8_t> rhs_native_mtx_qs4cx(rhs_native_size_qs4cx);
  std::vector<uint8_t> rhs_scales_f32(rhs_scales_size_f32);
  nntrainer::quant_qs4cx_f32(N, K, weight.data(), rhs_native_mtx_qs4cx.data(),
                             rhs_scales_f32.data(), true);

  const size_t num_idx_variants =
    nntrainer::__kai_get_num_ukernel_variants_qai8dxp_qsi4cxp();
  size_t max_rhs_packed_size = 0;
  for (size_t idx_variant = 0; idx_variant < num_idx_variants; idx_variant++) {
    size_t rhs_packed_size =
      nntrainer::get_rhs_packed_size_qsi4cxp_qs4cxs1s0(N, K, idx_variant, true);
    max_rhs_packed_size = std::max(max_rhs_packed_size, rhs_packed_size);
  }

  std::vector<uint8_t> packed_weight_qai8dxp(max_rhs_packed_size);

  // ============================================================
  // Benchmark: gemm_q4_0<float>
  // ============================================================

  // Keep the RHS cold, LHS warm as in inference:
  // - For each iteration, RHS will be evicted from caches before running gemm.
  // - LHS will stay as it is.
  nanoseconds total_time_q4_0 = nanoseconds(0);
  for (size_t i = 0; i < warmup_iters; ++i) {
    evict_from_caches(q4_0_repacked_qWeight.data(), q4_0_data_size);
    nntrainer::gemm_q4_0<float>(M, N, K, activation.data(), K,
                                (void *)q4_0_repacked_qWeight.data(), N,
                                dst_q4_0.data(), N);
  }

  for (size_t i = 0; i < test_iters; ++i) {
    evict_from_caches(q4_0_repacked_qWeight.data(), q4_0_data_size);
    auto t1 = steady_clock::now();
    nntrainer::gemm_q4_0<float>(M, N, K, activation.data(), K,
                                (void *)q4_0_repacked_qWeight.data(), N,
                                dst_q4_0.data(), N);
    auto t2 = steady_clock::now();
    total_time_q4_0 += duration_cast<nanoseconds>(t2 - t1);
  }

  // ============================================================
  // Benchmark: qai8dxp_qsi4cxp_packed
  // ============================================================
  std::vector<nanoseconds> total_time_qai8dxp(num_idx_variants, nanoseconds(0));

  for (size_t idx_variant = 0; idx_variant < num_idx_variants; idx_variant++) {
    // rhs pack
    nntrainer::rhs_pack_qsi4cxp_qs4cxs1s0(
      N, K, packed_weight_qai8dxp.data(), rhs_native_mtx_qs4cx.data(),
      rhs_scales_f32.data(), idx_variant, true);

    // cold-RHS eviction, same scheme as the q4_0 benchmark above
    const size_t rhs_packed_size =
      nntrainer::get_rhs_packed_size_qsi4cxp_qs4cxs1s0(N, K, idx_variant, true);

    // warm up
    for (size_t i = 0; i < warmup_iters; ++i) {
      evict_from_caches(packed_weight_qai8dxp.data(), rhs_packed_size);
      nntrainer::gemm_qai8dxp_qsi4cxp(M, N, K, activation.data(),
                                      packed_weight_qai8dxp.data(),
                                      dst_qai8dxp.data(), idx_variant);
    }

    nanoseconds local_time = nanoseconds(0);

    for (size_t i = 0; i < test_iters; ++i) {
      evict_from_caches(packed_weight_qai8dxp.data(), rhs_packed_size);
      auto t1 = steady_clock::now();
      nntrainer::gemm_qai8dxp_qsi4cxp(M, N, K, activation.data(),
                                      packed_weight_qai8dxp.data(),
                                      dst_qai8dxp.data(), idx_variant);
      auto t2 = steady_clock::now();
      local_time += duration_cast<nanoseconds>(t2 - t1);
    }
    total_time_qai8dxp[idx_variant] = local_time;
  }

  // ============================================================
  // Print results
  // ============================================================
  auto avg_ns_q4_0 = total_time_q4_0.count() / test_iters;
  std::cout << "\n-----------------------------------------" << std::endl;
  std::cout << "[RESULT] Average latency over " << test_iters
            << " iterations:" << std::endl;
  std::cout << "  gemm_q4_0<float>:         " << avg_ns_q4_0 << " ns ("
            << avg_ns_q4_0 / 1'000 << " us, " << avg_ns_q4_0 / 1'000'000
            << " ms)" << std::endl;
  std::cout << "-----------------------------------------" << std::endl;
  for (size_t i = 0; i < num_idx_variants; i++) {
    auto avg_ns_qai8dxp = total_time_qai8dxp[i].count() / test_iters;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "ukernel[" << i << "] "
              << nntrainer::__kai_get_num_ukernel_name_qai8dxp_qsi4cxp(i)
              << std::endl;
    std::cout << "  qai8dxp_qsi4cxp_packed:   " << avg_ns_qai8dxp << " ns ("
              << avg_ns_qai8dxp / 1'000 << " us, " << avg_ns_qai8dxp / 1'000'000
              << " ms)" << std::endl;
  }
}

// Benchmarks, not correctness tests: DISABLED from the default
// Run on demand with `--bench`
TEST(nntrainer_kleidiai, DISABLED_gemm_benchmark_comparison_1x2560x4096) {
  run_gemm_benchmark_comparison(1, 2560, 4096);
}

TEST(nntrainer_kleidiai, DISABLED_gemm_benchmark_comparison_1024x2560x4096) {
  run_gemm_benchmark_comparison(1024, 2560, 4096);
}

/**
 * @brief rhs_pack_nxk_qsi4cxp_qs4cxs1s0 NEON optimization test
 *
 * Checks that kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon produces a
 * byte-identical packed buffer to the reference implementation for
 * (nr, kr, sr) = (8, 16, 2), and measures the latency of both.
 */
class rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon
  : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {};

TEST_P(rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon, correctness_and_perf) {
  auto [N, K] = GetParam();

  constexpr size_t nr = 8;
  constexpr size_t kr = 16;
  constexpr size_t sr = 2;
  constexpr size_t warmup_iters = 5;
  constexpr size_t test_iters = 20;

  // Random 4-bit source data (two nibbles per byte), scales and bias
  const size_t rhs_stride = (K + 1) / 2;
  std::vector<uint8_t> rhs_native(N * rhs_stride);
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &b : rhs_native) {
    b = static_cast<uint8_t>(dist(gen));
  }
  std::vector<float> scales = generate_random_vector<float>(N, 0.001F, 1.0F);

  const size_t packed_size =
    kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(N, K, nr, kr, sr);
  std::vector<uint8_t> packed_ref(packed_size, 0xA5);
  std::vector<uint8_t> packed_opt(packed_size, 0x5A);

  struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params;
  params.lhs_zero_point = 1;
  params.rhs_zero_point = 8;

  // Correctness
  kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(1, N, K, nr, kr, sr, rhs_native.data(),
                                         NULL, scales.data(), packed_ref.data(),
                                         0, &params);
  kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon(
    1, N, K, nr, kr, sr, rhs_native.data(), NULL, scales.data(),
    packed_opt.data(), 0, &params);

  for (size_t i = 0; i < packed_size; i++) {
    ASSERT_EQ(packed_ref[i], packed_opt[i])
      << "first mismatch at byte " << i << " / " << packed_size;
  }

  // Performance
  for (size_t i = 0; i < warmup_iters; ++i) {
    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
      1, N, K, nr, kr, sr, rhs_native.data(), NULL, scales.data(),
      packed_ref.data(), 0, &params);
  }
  nanoseconds time_ref = nanoseconds(0);
  for (size_t i = 0; i < test_iters; ++i) {
    auto t1 = steady_clock::now();
    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
      1, N, K, nr, kr, sr, rhs_native.data(), NULL, scales.data(),
      packed_ref.data(), 0, &params);
    auto t2 = steady_clock::now();
    time_ref += duration_cast<nanoseconds>(t2 - t1);
  }

  for (size_t i = 0; i < warmup_iters; ++i) {
    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon(
      1, N, K, nr, kr, sr, rhs_native.data(), NULL, scales.data(),
      packed_opt.data(), 0, &params);
  }
  nanoseconds time_opt = nanoseconds(0);
  for (size_t i = 0; i < test_iters; ++i) {
    auto t1 = steady_clock::now();
    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon(
      1, N, K, nr, kr, sr, rhs_native.data(), NULL, scales.data(),
      packed_opt.data(), 0, &params);
    auto t2 = steady_clock::now();
    time_opt += duration_cast<nanoseconds>(t2 - t1);
  }

  const auto avg_ref = time_ref.count() / test_iters;
  const auto avg_opt = time_opt.count() / test_iters;
  std::cout << "\n-----------------------------------------" << std::endl;
  std::cout << "[RESULT] rhs_pack_nxk_qsi4cxp (n, k) = (" << N << ", " << K
            << "), average over " << test_iters << " iterations:" << std::endl;
  std::cout << "  reference: " << avg_ref << " ns (" << avg_ref / 1'000
            << " us)" << std::endl;
  std::cout << "  neon:      " << avg_opt << " ns (" << avg_opt / 1'000
            << " us)" << std::endl;
  std::cout << "  speedup:   " << static_cast<double>(avg_ref) / avg_opt << "x"
            << std::endl;
  std::cout << "-----------------------------------------" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
  nntrainer_kleidiai, rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon,
  ::testing::Values(std::make_tuple(1024, 1024), std::make_tuple(1024, 2048),
                    std::make_tuple(1024, 3072), std::make_tuple(2048, 1024),
                    std::make_tuple(3072, 1024),
                    // edge cases: n % nr != 0, k % 32 != 0
                    std::make_tuple(1000, 1000), std::make_tuple(9, 62)),
  [](const ::testing::TestParamInfo<
     rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon::ParamType> &info) {
    size_t N = std::get<0>(info.param);
    size_t K = std::get<1>(info.param);
    return std::to_string(N) + "_" + std::to_string(K);
  });

static void compare_dequantize_row_qs4cx_fallback_vs_neon(size_t n, size_t k) {
  std::vector<float> rhs_f32 =
    generate_random_vector<float>(n * k, -10.0f, 10.0f);

  const size_t rhs_qs4cx_stride = (k + 1) / 2;
  std::vector<uint8_t> rhs_qs4cx(n * rhs_qs4cx_stride, 0);
  std::vector<float> rhs_scales_f32(n, 0.0f);

  nntrainer::__fallback_quant_nxk_qs4cx_f32(
    n, k, rhs_f32.data(), rhs_qs4cx.data(), rhs_scales_f32.data());

  std::vector<float> row_out_fallback(k, 0.0f);
  std::vector<float> row_out_neon(k, 0.0f);

  for (size_t n_idx = 0; n_idx < n; ++n_idx) {
    // Reference fallback implementation
    nntrainer::__fallback_dequantize_row_qs4cx_f32(n_idx, k, rhs_qs4cx.data(),
                                                   rhs_scales_f32.data(),
                                                   row_out_fallback.data());

    // Test NEON implementation
    nntrainer::neon::dequantize_row_qs4cx_neon(
      n_idx, k, rhs_qs4cx.data(), rhs_scales_f32.data(), row_out_neon.data());

    // Compare results: should be byte-identical
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      EXPECT_FLOAT_EQ(row_out_fallback[k_idx], row_out_neon[k_idx])
        << "Mismatch at n=" << n_idx << ", k=" << k_idx;
      EXPECT_EQ(std::memcmp(&row_out_fallback[k_idx], &row_out_neon[k_idx],
                            sizeof(float)),
                0)
        << "Not byte-identical at n=" << n_idx << ", k=" << k_idx;
    }
  }
}

TEST(nntrainer_arm_neon, dequantize_row_qs4cx_vs_fallback) {
  compare_dequantize_row_qs4cx_fallback_vs_neon(/*n=*/4, /*k=*/8);
}

TEST(nntrainer_arm_neon, dequantize_row_qs4cx_vs_fallback_odd_k) {
  compare_dequantize_row_qs4cx_fallback_vs_neon(/*n=*/4, /*k=*/7);
}

TEST(nntrainer_arm_neon, dequantize_row_qs4cx_vs_fallback_large) {
  compare_dequantize_row_qs4cx_fallback_vs_neon(/*n=*/16, /*k=*/256);
}

TEST(nntrainer_arm_neon, dequantize_row_qs4cx_vs_fallback_large_odd) {
  compare_dequantize_row_qs4cx_fallback_vs_neon(/*n=*/16, /*k=*/255);
}

TEST(nntrainer_arm_neon, DISABLED_dequantize_row_qs4cx_perf_vs_fallback) {
  const size_t n = 64;
  const size_t k = 4096;
  const size_t warmup_iterations = 3;
  const size_t iterations = 20;
  const size_t n_idx = 31; // fix row to test

  std::vector<float> rhs_f32 =
    generate_random_vector<float>(n * k, -10.0f, 10.0f);

  const size_t rhs_qs4cx_stride = (k + 1) / 2;
  std::vector<uint8_t> rhs_qs4cx(n * rhs_qs4cx_stride, 0);
  std::vector<float> rhs_scales_f32(n, 0.0f);

  nntrainer::__fallback_quant_nxk_qs4cx_f32(
    n, k, rhs_f32.data(), rhs_qs4cx.data(), rhs_scales_f32.data());

  std::vector<float> out_fallback(k, 0.0f);
  std::vector<float> out_neon(k, 0.0f);

  auto run_fallback = [&]() {
    nntrainer::__fallback_dequantize_row_qs4cx_f32(
      n_idx, k, rhs_qs4cx.data(), rhs_scales_f32.data(), out_fallback.data());
  };
  auto run_neon = [&]() {
    nntrainer::neon::dequantize_row_qs4cx_neon(
      n_idx, k, rhs_qs4cx.data(), rhs_scales_f32.data(), out_neon.data());
  };

  for (size_t it = 0; it < warmup_iterations; ++it) {
    run_fallback();
    run_neon();
  }

  // Interleave the two implementations so DVFS / thermal drift affects both
  // equally
  int64_t fallback_ns = 0;
  int64_t neon_ns = 0;
  for (size_t it = 0; it < iterations; ++it) {
    auto t1 = std::chrono::high_resolution_clock::now();
    run_fallback();
    auto t2 = std::chrono::high_resolution_clock::now();
    run_neon();
    auto t3 = std::chrono::high_resolution_clock::now();
    fallback_ns +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    neon_ns +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
  }

  EXPECT_EQ(
    std::memcmp(out_fallback.data(), out_neon.data(), k * sizeof(float)), 0)
    << "NEON output is not byte-identical to fallback";

  const double iters = static_cast<double>(iterations);
  std::cout << "[INFO] dequantize_row_qs4cx n=" << n << " k=" << k
            << " (avg of " << iterations << " iters)" << std::endl;
  std::cout << "[INFO] fallback: " << fallback_ns / iters / 1000.0 << " us, "
            << std::endl;
  std::cout << "[INFO] neon    : " << neon_ns / iters / 1000.0 << " us, "
            << std::endl;
  std::cout << "[INFO] speedup : "
            << static_cast<double>(fallback_ns) / static_cast<double>(neon_ns)
            << "x" << std::endl;
}

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  // `--bench`: run the DISABLED_ benchmark tests
  // When it is applied, filter is narrowed to the benchmarks only.
  // It can be overriden by `--gtest_filter` option.
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--bench") {
      ::testing::GTEST_FLAG(also_run_disabled_tests) = true;
      if (::testing::GTEST_FLAG(filter) == "*") {
        ::testing::GTEST_FLAG(filter) = "*gemm_benchmark*";
      }
    }
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
