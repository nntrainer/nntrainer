// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_ggml_arm.cpp
 * @date	09 July 2026
 * @brief	This is unittest for ARM-specific nntr_ggml_impl functions
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jaemin Shin <jaemin980311@gmail.com>
 * @bug		No known bugs except for NYI items
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <arm_neon.h>

#include <gtest/gtest.h>

#include <ggml_interface.h>
#include <nntr_ggml_impl.h>
#include <nntrainer_test_util.h>
#include <thread_manager.h>

// Convert a float to its fp16 bit pattern the same way the production kernels
// do. This test only runs on aarch64, where the library's fp32->fp16 helpers
// reduce to a hardware __fp16 cast, so doing the cast directly keeps the
// reference block deltas byte-for-byte identical to the production ones.
static inline uint16_t fp32_to_fp16_bits(float f) {
  __fp16 h = (__fp16)f;
  uint16_t bits;
  memcpy(&bits, &h, sizeof(bits));
  return bits;
}

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

/**
 * @brief q8_0x4 block (four interleaved q8_0 blocks)
 */
typedef struct {
  uint16_t d[4];  // deltas (fp16) of the four q8_0 blocks
  int8_t qs[128]; // quants, interleaved by 8 bytes per block
} block_q8_0x4_testonly;

static_assert(sizeof(block_q8_0x4_testonly) == 136,
              "block_q8_0x4 layout mismatch");

/**
 * @brief q8_0 block
 */
typedef struct {
  uint16_t d;    // delta (fp16)
  int8_t qs[32]; // quants
} block_q8_0_testonly;

static_assert(sizeof(block_q8_0_testonly) == 34, "block_q8_0 layout mismatch");

/**
 * @brief q4_0 block
 */
typedef struct {
  uint16_t d;     // delta (fp16)
  uint8_t qs[16]; // nibbles / quants
} block_q4_0_testonly;

static_assert(sizeof(block_q4_0_testonly) == 18, "block_q4_0 layout mismatch");

/**
 * @brief Reference (scalar per-lane store) implementation of
 * nntr_quantize_mat_q8_0_4x8, retained here to validate the vectorized
 * production kernel byte-for-byte.
 */
static void nntr_quantize_mat_q8_0_4x8_ref(const float *__restrict x,
                                           void *__restrict vy, int64_t k) {
  const int nb = k / 32;

  block_q8_0x4_testonly *__restrict y = (block_q8_0x4_testonly *)vy;

  float32x4_t srcv[4][8];
  float id[4];

  for (int i = 0; i < nb; i++) {
    float32x4_t asrcv[8];
    float32x4_t amaxv[8];

    for (int row_iter = 0; row_iter < 4; row_iter++) {
      for (int j = 0; j < 8; j++)
        srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
      for (int j = 0; j < 8; j++)
        asrcv[j] = vabsq_f32(srcv[row_iter][j]);

      for (int j = 0; j < 4; j++)
        amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
      for (int j = 0; j < 2; j++)
        amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
      for (int j = 0; j < 1; j++)
        amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

      const float amax = vmaxvq_f32(amaxv[0]);

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = fp32_to_fp16_bits(d);
    }

    for (int j = 0; j < 4; j++) {
      float32x4_t v = vmulq_n_f32(srcv[0][2 * j], id[0]);
      int32x4_t vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 0] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 1] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 2] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 3] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[0][2 * j + 1], id[0]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

      v = vmulq_n_f32(srcv[1][2 * j], id[1]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[1][2 * j + 1], id[1]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

      v = vmulq_n_f32(srcv[2][2 * j], id[2]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[2][2 * j + 1], id[2]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

      v = vmulq_n_f32(srcv[3][2 * j], id[3]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[3][2 * j + 1], id[3]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
    }
  }
}

/**
 * @brief Reference (scalar per-lane store) implementation of
 * nntr_quantize_row_q8_0, retained here to validate the vectorized production
 * kernel byte-for-byte.
 */
static void nntr_quantize_row_q8_0_ref(const float *__restrict x,
                                       void *__restrict vy, int64_t k) {
  const int64_t nb = k / 32;

  block_q8_0_testonly *__restrict y = (block_q8_0_testonly *)vy;

  for (int64_t i = 0; i < nb; i++) {
    float32x4_t srcv[8];
    float32x4_t asrcv[8];
    float32x4_t amaxv[8];

    for (int j = 0; j < 8; j++)
      srcv[j] = vld1q_f32(x + i * 32 + 4 * j);
    for (int j = 0; j < 8; j++)
      asrcv[j] = vabsq_f32(srcv[j]);

    for (int j = 0; j < 4; j++)
      amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
    for (int j = 0; j < 2; j++)
      amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
    for (int j = 0; j < 1; j++)
      amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

    const float amax = vmaxvq_f32(amaxv[0]);

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = fp32_to_fp16_bits(d);

    for (int j = 0; j < 8; j++) {
      const float32x4_t v = vmulq_n_f32(srcv[j], id);
      const int32x4_t vi = vcvtnq_s32_f32(v);

      y[i].qs[4 * j + 0] = vgetq_lane_s32(vi, 0);
      y[i].qs[4 * j + 1] = vgetq_lane_s32(vi, 1);
      y[i].qs[4 * j + 2] = vgetq_lane_s32(vi, 2);
      y[i].qs[4 * j + 3] = vgetq_lane_s32(vi, 3);
    }
  }
}

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
 * @brief Generate rows x k random activations
 */
static std::vector<float> generate_activations(int64_t k, unsigned int seed,
                                               int rows = 4) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> vec(rows * k);
  for (auto &v : vec)
    v = dist(gen);
  return vec;
}

/**
 * @brief Latency statistics (ns)
 */
struct latency_stats {
  double min_ns;
  double max_ns;
  double median_ns;
  double mean_ns;
  double stdev_ns;
};

/**
 * @brief Compute latency statistics
 */
static latency_stats compute_latency_stats(std::vector<int64_t> ns) {
  std::sort(ns.begin(), ns.end());
  const size_t n = ns.size();
  latency_stats s;
  s.min_ns = (double)ns.front();
  s.max_ns = (double)ns.back();
  s.median_ns = (n % 2) ? (double)ns[n / 2]
                        : ((double)ns[n / 2 - 1] + (double)ns[n / 2]) / 2.0;
  s.mean_ns = std::accumulate(ns.begin(), ns.end(), 0.0) / n;
  double sq = 0.0;
  for (int64_t v : ns)
    sq += ((double)v - s.mean_ns) * ((double)v - s.mean_ns);
  s.stdev_ns = (n > 1) ? std::sqrt(sq / (n - 1)) : 0.0;
  return s;
}

/**
 * @brief Print latency statistics in us
 */
static void print_latency_stats(const latency_stats &s) {
  std::cout << "[INFO]   min " << s.min_ns / 1000.0 << " / median "
            << s.median_ns / 1000.0 << " / mean " << s.mean_ns / 1000.0
            << " / max " << s.max_ns / 1000.0 << " us, stdev "
            << s.stdev_ns / 1000.0 << " us" << std::endl;
}

/**
 * @brief Print per-call samples in us
 */
static void print_samples_us(const std::vector<int64_t> &ns) {
  std::cout << "[INFO]   samples(us):";
  for (int64_t v : ns)
    std::cout << " " << v / 1000.0;
  std::cout << std::endl;
}

/**
 * @brief Compare nntr_quantize_mat_q8_0_4x8_ref and nntr_quantize_mat_q8_0_4x8
 */
static void expect_opt_matches_reference(const std::vector<float> &src,
                                         int64_t k) {
  const size_t out_size = (k / 32) * sizeof(block_q8_0x4_testonly);
  std::vector<uint8_t> ref(out_size, 0xAA);
  std::vector<uint8_t> opt(out_size, 0x55);

  nntr_quantize_mat_q8_0_4x8_ref(src.data(), ref.data(), k);
  nntr_quantize_mat_q8_0_4x8(src.data(), opt.data(), k);

  if (memcmp(ref.data(), opt.data(), out_size) == 0)
    return;
  for (size_t i = 0; i < out_size; ++i)
    ASSERT_EQ((int)ref[i], (int)opt[i])
      << "first mismatching byte at offset " << i << " (k=" << k << ")";
}

/**
 * @brief Compare nntr_quantize_row_q8_0_ref and nntr_quantize_row_q8_0
 */
static void expect_row_opt_matches_reference(const std::vector<float> &src,
                                             int64_t k) {
  const size_t out_size = (k / 32) * sizeof(block_q8_0_testonly);
  std::vector<uint8_t> ref(out_size, 0xAA);
  std::vector<uint8_t> opt(out_size, 0x55);

  nntr_quantize_row_q8_0_ref(src.data(), ref.data(), k);
  nntr_quantize_row_q8_0(src.data(), opt.data(), k);

  if (memcmp(ref.data(), opt.data(), out_size) == 0)
    return;
  for (size_t i = 0; i < out_size; ++i)
    ASSERT_EQ((int)ref[i], (int)opt[i])
      << "first mismatching byte at offset " << i << " (k=" << k << ")";
}

TEST(nntrainer_ggml_arm, quantize_row_q8_0_opt_random) {
  nntr_ggml_init();

  unsigned int seed = 77;
  for (int64_t k : {32, 256, 1024, 4096, 8192}) {
    auto src = generate_activations(k, seed++, 1);
    expect_row_opt_matches_reference(src, k);
  }
}

TEST(nntrainer_ggml_arm, quantize_row_q8_0_opt_edge_cases) {
  nntr_ggml_init();

  const int64_t k = 256;
  auto src = generate_activations(k, 77, 1);

  // block 1 all zero: exercises the d == 0 / id == 0 path
  for (int j = 32; j < 64; ++j)
    src[j] = 0.0f;
  // extreme magnitudes
  src[64] = 1e30f;
  src[65] = -1e30f;
  src[96] = 1e-30f;
  // lands on a round-to-nearest boundary after scaling
  src[128] = 63.5f;

  expect_row_opt_matches_reference(src, k);
}

TEST(nntrainer_ggml_arm, quantize_mat_q8_0_4x8_opt_random) {
  nntr_ggml_init();

  unsigned int seed = 42;
  for (int64_t k : {32, 256, 1024, 4096, 8192}) {
    auto src = generate_activations(k, seed++);
    expect_opt_matches_reference(src, k);
  }
}

TEST(nntrainer_ggml_arm, quantize_mat_q8_0_4x8_opt_edge_cases) {
  nntr_ggml_init();

  const int64_t k = 256;
  auto src = generate_activations(k, 42);

  // row 1, block 0 all zero: exercises the d == 0 / id == 0 path
  for (int j = 0; j < 32; ++j)
    src[k + j] = 0.0f;
  // extreme magnitudes
  src[32] = 1e30f;
  src[33] = -1e30f;
  src[2 * k + 5] = 1e-30f;
  // lands on a round-to-nearest boundary after scaling
  src[3 * k + 7] = 63.5f;

  expect_opt_matches_reference(src, k);
}

TEST(nntrainer_ggml_arm, DISABLED_quantize_mat_q8_0_4x8_benchmark) {
  nntr_ggml_init();

  const int64_t k = 4096;
  const int warmup_iters = 500, iters = 100;
  auto src = generate_activations(k, 42);
  const size_t src_bytes = src.size() * sizeof(float);
  std::vector<uint8_t> dst((k / 32) * sizeof(block_q8_0x4_testonly));

  // single-threaded kernel, so no pool wake is needed
  auto run_side = [&](auto fn) {
    for (int it = 0; it < warmup_iters; ++it) {
      evict_from_caches(src.data(), src_bytes);
      fn(src.data(), dst.data(), k);
    }
    std::vector<int64_t> samples;
    samples.reserve(iters);
    for (int it = 0; it < iters; ++it) {
      evict_from_caches(src.data(), src_bytes);

      auto t0 = high_resolution_clock::now();
      fn(src.data(), dst.data(), k);
      auto t1 = high_resolution_clock::now();
      samples.push_back(duration_cast<nanoseconds>(t1 - t0).count());
    }
    return samples;
  };

  const auto samples = run_side(nntr_quantize_mat_q8_0_4x8);

  const latency_stats st = compute_latency_stats(samples);
  std::cout << "[INFO] quantize_mat_q8_0_4x8 (k=" << k << ", 4 rows, cold src, "
            << iters << " iters)" << std::endl;
  print_latency_stats(st);
  print_samples_us(samples);
}

static float test_q8_0_4x4(const size_t M, const size_t N, const size_t K) {
  nntr_ggml_init();

  auto A = generate_activations(K, 11, M);
  auto W = generate_activations(K, 22, N);

  std::vector<float> C_ref(M * N, 0.0f);

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < K; k++) {
        C_ref[i * N + j] += A[i * K + k] * W[j * K + k];
      }
    }
  }

  const size_t q4_size = (size_t)N * K / 32 * sizeof(block_q8_0_testonly);
  std::vector<char> Q4(q4_size), B(q4_size);
  nntr_quantize_q8_0(W.data(), Q4.data(), N, K, nullptr);
  nntr_repack_q8_0_to_q8_0_4_bl(B.data(), 4, Q4.data(), q4_size, N, K);

  std::vector<float> C(M * N, 0.0f);

  nntrainer::__ggml_q8_0_4x4_q8_0_GEMM(M, N, K, A.data(), K, B.data(), N,
                                       C.data(), N);

  return cosine_similarity(C_ref.data(), C.data(), M * N);
}

static float test_q8_0_4x8(const size_t M, const size_t N, const size_t K) {
  nntr_ggml_init();

  auto A = generate_activations(K, 11, M);
  auto W = generate_activations(K, 22, N);

  std::vector<float> C_ref(M * N, 0.0f);

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < K; k++) {
        C_ref[i * N + j] += A[i * K + k] * W[j * K + k];
      }
    }
  }

  const size_t q4_size = (size_t)N * K / 32 * sizeof(block_q8_0_testonly);
  std::vector<char> Q4(q4_size), B(q4_size);
  nntr_quantize_q8_0(W.data(), Q4.data(), N, K, nullptr);
  nntr_repack_q8_0_to_q8_0_4_bl(B.data(), 8, Q4.data(), q4_size, N, K);

  std::vector<float> C(M * N, 0.0f);

  nntrainer::__ggml_q8_0_4x8_q8_0_GEMM(M, N, K, A.data(), K, B.data(), N,
                                       C.data(), N);

  return cosine_similarity(C_ref.data(), C.data(), M * N);
}

TEST(nntrainer_ggml_arm, gemv_q8_0_4x4_1x128x128) {
  const size_t M = 1, N = 128, K = 128;
  float cos = test_q8_0_4x4(M, N, K);

  EXPECT_GT(cos, 0.999f);
}

TEST(nntrainer_ggml_arm, gemv_q8_0_4x8_1x128x128) {
  const size_t M = 1, N = 128, K = 128;
  float cos = test_q8_0_4x8(M, N, K);

  EXPECT_GT(cos, 0.999f);
}

TEST(nntrainer_ggml_arm, gemm_q8_0_4x4_32x128x128) {
  const size_t M = 32, N = 128, K = 128;
  float cos = test_q8_0_4x4(M, N, K);

  EXPECT_GT(cos, 0.999f);
}

TEST(nntrainer_ggml_arm, gemm_q8_0_4x8_32x128x128) {
  const size_t M = 32, N = 128, K = 128;
  float cos = test_q8_0_4x8(M, N, K);

  EXPECT_GT(cos, 0.999f);
}

TEST(nntrainer_ggml_arm, gemm_q8_0_4x4_35x128x128) {
  const size_t M = 35, N = 128, K = 128;
  float cos = test_q8_0_4x4(M, N, K);

  EXPECT_GT(cos, 0.999f);
}

TEST(nntrainer_ggml_arm, gemm_q8_0_4x8_35x128x128) {
  const size_t M = 35, N = 128, K = 128;
  float cos = test_q8_0_4x8(M, N, K);

  EXPECT_GT(cos, 0.999f);
}

TEST(nntrainer_ggml_arm, gemm_q8_0_4x4_128x128x128) {
  const size_t M = 128, N = 128, K = 128;
  float cos = test_q8_0_4x4(M, N, K);

  EXPECT_GT(cos, 0.999f);
}

TEST(nntrainer_ggml_arm, gemm_q8_0_4x8_128x128x128) {
  const size_t M = 128, N = 128, K = 128;
  float cos = test_q8_0_4x8(M, N, K);

  EXPECT_GT(cos, 0.999f);
}

TEST(nntrainer_ggml_arm, DISABLED_quantize_row_q8_0_benchmark) {
  nntr_ggml_init();

  const int64_t k = 4096;
  const int warmup_iters = 500, iters = 100;
  auto src = generate_activations(k, 77, 1);
  const size_t src_bytes = src.size() * sizeof(float);
  std::vector<uint8_t> dst((k / 32) * sizeof(block_q8_0_testonly));

  // single-threaded kernel, so no pool wake is needed
  auto run_side = [&](auto fn) {
    for (int it = 0; it < warmup_iters; ++it) {
      evict_from_caches(src.data(), src_bytes);
      fn(src.data(), dst.data(), k);
    }
    std::vector<int64_t> samples;
    samples.reserve(iters);
    for (int it = 0; it < iters; ++it) {
      evict_from_caches(src.data(), src_bytes);

      auto t0 = high_resolution_clock::now();
      fn(src.data(), dst.data(), k);
      auto t1 = high_resolution_clock::now();
      samples.push_back(duration_cast<nanoseconds>(t1 - t0).count());
    }
    return samples;
  };

  const auto samples = run_side(nntr_quantize_row_q8_0);

  const latency_stats st = compute_latency_stats(samples);
  std::cout << "[INFO] quantize_row_q8_0 (k=" << k << ", cold src, " << iters
            << " iters)" << std::endl;
  print_latency_stats(st);
  print_samples_us(samples);
}

TEST(nntrainer_ggml_arm, DISABLED_gemv_q4_0_4x8_benchmark) {
  nntr_ggml_init();

  const unsigned int N = 2560, K = 4096;
  const int warmup_iters = 3, iters = 10;

  auto A = generate_activations(K, 11, 1);
  auto W = generate_activations(K, 22, N);

  // offline weight quantization + repack to q4_0x4 layout
  const size_t q4_size = (size_t)N * K / 32 * sizeof(block_q4_0_testonly);
  std::vector<char> Q4(q4_size), B(q4_size);
  nntr_quantize_q4_0(W.data(), Q4.data(), N, K, nullptr);
  nntr_repack_q4_0_to_q4_0_4_bl(B.data(), 8, Q4.data(), q4_size, N, K);

  std::vector<float> C(N, 0.0f);

  auto &tm = nntrainer::ThreadManager::Global();
  // wake workers that fell into deep idle during the ~ms eviction
  auto wake_workers = [&]() {
    tm.parallel_for(0, tm.getComputeThreadCount(), [](size_t) {});
  };

  // warmup
  for (int i = 0; i < warmup_iters; ++i) {
    evict_from_caches(B.data(), q4_size);
    nntrainer::__ggml_q4_0_4x8_q8_0_GEMM<float>(1, N, K, A.data(), K, B.data(),
                                                N, C.data(), N);
  }

  std::vector<int64_t> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    evict_from_caches(B.data(), q4_size);
    wake_workers();

    auto t0 = high_resolution_clock::now();
    nntrainer::__ggml_q4_0_4x8_q8_0_GEMM<float>(1, N, K, A.data(), K, B.data(),
                                                N, C.data(), N);
    auto t1 = high_resolution_clock::now();
    samples.push_back(duration_cast<nanoseconds>(t1 - t0).count());
  }

  const latency_stats st = compute_latency_stats(samples);
  std::cout << "[INFO] __ggml_q4_0_4x8_q8_0_GEMM GEMV path (N=" << N
            << ", K=" << K << ", cold RHS, " << iters << " iters)" << std::endl;
  print_latency_stats(st);
  print_samples_us(samples);
  std::cout << "[INFO]   weight stream "
            << (double)q4_size / st.median_ns // bytes/ns == GB/s
            << " GB/s (median)" << std::endl;
}

TEST(nntrainer_ggml_arm, DISABLED_gemm_q4_0_4x8_benchmark) {
  nntr_ggml_init();

  const unsigned int M = 1024, N = 2560, K = 4096;
  const int warmup_iters = 10, iters = 10;

  auto A = generate_activations(K, 33, M);
  auto W = generate_activations(K, 44, N);

  // offline weight quantization + repack to q4_0x4 layout
  const size_t q4_size = (size_t)N * K / 32 * sizeof(block_q4_0_testonly);
  std::vector<char> Q4(q4_size), B(q4_size);
  nntr_quantize_q4_0(W.data(), Q4.data(), N, K, nullptr);
  nntr_repack_q4_0_to_q4_0_4_bl(B.data(), 8, Q4.data(), q4_size, N, K);

  std::vector<float> C((size_t)M * N, 0.0f);

  auto &tm = nntrainer::ThreadManager::Global();
  // wake workers that fell into deep idle during the ~ms eviction
  auto wake_workers = [&]() {
    tm.parallel_for(0, tm.getComputeThreadCount(), [](size_t) {});
  };

  // warmup
  for (int i = 0; i < warmup_iters; ++i) {
    evict_from_caches(B.data(), q4_size);
    nntrainer::__ggml_q4_0_4x8_q8_0_GEMM<float>(M, N, K, A.data(), K, B.data(),
                                                N, C.data(), N);
  }

  std::vector<int64_t> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    evict_from_caches(B.data(), q4_size);
    wake_workers();

    auto t0 = high_resolution_clock::now();
    nntrainer::__ggml_q4_0_4x8_q8_0_GEMM<float>(M, N, K, A.data(), K, B.data(),
                                                N, C.data(), N);
    auto t1 = high_resolution_clock::now();
    samples.push_back(duration_cast<nanoseconds>(t1 - t0).count());
  }

  const latency_stats st = compute_latency_stats(samples);
  std::cout << "[INFO] __ggml_q4_0_4x8_q8_0_GEMM GEMM path (M=" << M
            << ", N=" << N << ", K=" << K << ", cold RHS, " << iters
            << " iters)" << std::endl;
  print_latency_stats(st);
  print_samples_us(samples);
  std::cout << "[INFO]   "
            << 2.0 * M * N * K / st.median_ns // flop/ns == GFLOPS
            << " GFLOPS (median)" << std::endl;
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
        ::testing::GTEST_FLAG(filter) = "*benchmark*";
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
