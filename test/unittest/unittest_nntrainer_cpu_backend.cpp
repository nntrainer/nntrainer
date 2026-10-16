// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_cpu_backend.cpp
 * @date	03 April 2025
 * @brief	This is unittest for cpu_backend standalone
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "../htp/host/q4_gemv_cases.h"
#include "htp_act_quant.h"
#include "htp_q4_0_convert.h"
#include "htp_wh_layout.h"
#include "htp_wh_palette.h"
#include "int4_utils.h"
#include "m1_ops_det.h"
#include "nntrainer_test_util.h"
#include "q4_0_utils.h"
#include <cfloat>
#include <cmath>
#include <cpu_backend.h>
#include <cstring>
#include <fallback_internal.h>
#include <fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <nntr_ggml_impl.h>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tensor.h>
#include <thread>
#include <vector>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

#if defined(ENABLE_FP16) || defined(__AVX2__)
static inline std::vector<uint16_t>
convert_vector_f32_to_f16_as_uint16(std::vector<float> f32_vec) {
  std::vector<uint16_t> vec(f32_vec.size());
  for (size_t i = 0; i < f32_vec.size(); i++) {
    vec[i] = nntrainer::compute_fp32_to_fp16(f32_vec[i]);
  }
  return vec;
}

static inline std::vector<float>
convert_vector_f16_as_uint16_to_f32(std::vector<uint16_t> uint16_vec) {
  std::vector<float> vec(uint16_vec.size());
  for (size_t i = 0; i < uint16_vec.size(); i++) {
    vec[i] = nntrainer::compute_fp16_to_fp32(uint16_vec[i]);
  }
  return vec;
}
#endif

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  double err_sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(src[i * N + j] - src2[i * N + j]));
      err_sum += std::abs(src[i * N + j] - src2[i * N + j]);
    }
  }
  // std::cout << "err_sum : " << err_sum << std::endl;
  return max_diff;
}

#define QK4_0 32
#define Q4_0 32
/**
 * @brief q4_0 block
 *
 */
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0_testonly;
/**
 * @brief q4_K block
 *
 */
typedef struct {
  union {
    struct {
      int16_t d;    // super-block scale for quantized scales
      int16_t dmin; // super-block scale for quantized mins
    };
    uint32_t dm;
  };
  uint8_t scales[12];  // scales and mins, quantized with 6 bits
  uint8_t qs[256 / 2]; // 4--bit quants
} block_q4_K_testonly;
/**
 * @brief q8_K block
 *
 */
typedef struct {
  float d;                 // delta
  int8_t qs[256];          // quants
  int16_t bsums[256 / 16]; // sum of quants in groups of 16
} block_q8_K_testonly;
/**
 * @brief q4_Kx8 block
 *
 */
struct block_q4_Kx8_testonly {
  int16_t d[8];       // super-block scale for quantized scales
  int16_t dmin[8];    // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};

#define QK_K 256
typedef struct {
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  uint16_t d;               // super-block scale
} block_q6_K_testonly;

/**
 * @brief Elementwise-addition unittest : Vanilla example for formulating a TC
 * in unittest_nntrainer_cpu_backend.cpp
 *
 */
TEST(nntrainer_cpu_backend_standalone, ele_add) {
  const unsigned int TEST_SIZE = 100;
  float alpha = 1.F;
  float beta = 0.F;
  unsigned int i_stride = 1;
  unsigned int o_stride = 1;

  std::vector<float> lhs = generate_random_vector<float>(TEST_SIZE);
  std::vector<float> rhs = generate_random_vector<float>(TEST_SIZE);
  std::vector<float> dst(TEST_SIZE);

  const float *lhs_ptr = (const float *)lhs.data();
  const float *rhs_ptr = (const float *)rhs.data();
  float *dst_ptr = (float *)dst.data();

  nntrainer::ele_add(TEST_SIZE, lhs_ptr, rhs_ptr, dst_ptr, alpha, beta,
                     i_stride, o_stride);

  for (unsigned int i = 0; i < TEST_SIZE; ++i) {
    EXPECT_EQ(dst[i], lhs[i] + rhs[i]);
  }
}

template <typename T = float>
float compute_mse(const uint32_t M, const uint32_t N, std::vector<T> &ref_dst,
                  std::vector<T> &dst, bool print = false) {
  auto mean_squared_error = mse<T, T>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity<T, T>(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff<T>(ref_dst.data(), dst.data(), M, N);

  auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
  auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);
  if (print) {
    std::cout << "[INFO]            MSE: " << mean_squared_error
              << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
              << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  }
  return mean_squared_error;
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__AVX2__)
static std::vector<float> make_causal_conv_input(unsigned int B, unsigned int H,
                                                 unsigned int W) {
  std::vector<float> input(static_cast<size_t>(B) * H * W);
  for (size_t i = 0; i < input.size(); ++i) {
    const int v = static_cast<int>((i * 17 + 5) % 23) - 11;
    input[i] = static_cast<float>(v) * 0.125f;
  }
  return input;
}

static std::vector<float> make_causal_conv_weight(unsigned int W) {
  std::vector<float> weight(3 * W);
  for (size_t i = 0; i < weight.size(); ++i) {
    const int v = static_cast<int>((i * 7 + 3) % 19) - 9;
    weight[i] = static_cast<float>(v) * 0.03125f;
  }
  return weight;
}

static std::vector<float> make_causal_conv_bias(unsigned int W) {
  std::vector<float> bias(W);
  for (size_t i = 0; i < bias.size(); ++i) {
    const int v = static_cast<int>((i * 5 + 1) % 13) - 6;
    bias[i] = static_cast<float>(v) * 0.0625f;
  }
  return bias;
}

static std::vector<float> reference_causal_depthwise_conv1d_k3(
  const std::vector<float> &input, const std::vector<float> &weight,
  const float *bias, unsigned int B, unsigned int H, unsigned int W) {
  std::vector<float> output(static_cast<size_t>(B) * H * W, 0.0f);
  const float *w0 = weight.data();
  const float *w1 = weight.data() + W;
  const float *w2 = weight.data() + 2 * W;

  for (unsigned int b = 0; b < B; ++b) {
    const float *x_base = input.data() + static_cast<size_t>(b) * H * W;
    float *y_base = output.data() + static_cast<size_t>(b) * H * W;

    for (unsigned int t = 0; t < H; ++t) {
      for (unsigned int c = 0; c < W; ++c) {
        const float cur = x_base[static_cast<size_t>(t) * W + c];
        const float prev1 =
          t >= 1 ? x_base[static_cast<size_t>(t - 1) * W + c] : 0.0f;
        const float prev2 =
          t >= 2 ? x_base[static_cast<size_t>(t - 2) * W + c] : 0.0f;
        float acc = w0[c] * cur + w1[c] * prev1 + w2[c] * prev2;
        if (bias) {
          acc += bias[c];
        }
        y_base[static_cast<size_t>(t) * W + c] = acc;
      }
    }
  }

  return output;
}

TEST(nntrainer_cpu_backend_standalone,
     causal_depthwise_conv1d_k3_prefill_matches_reference_with_bias) {
  nntrainer::init_backend();

  constexpr unsigned int B = 2;
  constexpr unsigned int H = 5;
  constexpr unsigned int W = 45;

  const std::vector<float> input = make_causal_conv_input(B, H, W);
  const std::vector<float> weight = make_causal_conv_weight(W);
  const std::vector<float> bias = make_causal_conv_bias(W);
  const std::vector<float> expected =
    reference_causal_depthwise_conv1d_k3(input, weight, bias.data(), B, H, W);

  std::vector<float> output(expected.size(), 0.0f);
  nntrainer::causal_depthwise_conv1d_k3(input.data(), weight.data(),
                                        bias.data(), output.data(), B, H, W);

  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], expected[i], 1.0e-6f);
  }
}

TEST(nntrainer_cpu_backend_standalone,
     causal_depthwise_conv1d_k3_decode_matches_prefill_and_updates_state) {
  nntrainer::init_backend();

  constexpr unsigned int B = 1;
  constexpr unsigned int H = 6;
  constexpr unsigned int W = 45;

  const std::vector<float> input = make_causal_conv_input(B, H, W);
  const std::vector<float> weight = make_causal_conv_weight(W);
  const std::vector<float> expected =
    reference_causal_depthwise_conv1d_k3(input, weight, nullptr, B, H, W);

  std::vector<float> prefill_output(expected.size(), 0.0f);
  nntrainer::causal_depthwise_conv1d_k3(input.data(), weight.data(), nullptr,
                                        prefill_output.data(), B, H, W);

  std::vector<float> state(2 * W, 0.0f);
  std::vector<float> y_cur(W, 0.0f);

  for (unsigned int t = 0; t < H; ++t) {
    const float *x_cur = input.data() + static_cast<size_t>(t) * W;
    nntrainer::causal_depthwise_conv1d_k3_decode(x_cur, weight.data(),
                                                 state.data(), y_cur.data(), W);

    const float *prefill_y = prefill_output.data() + static_cast<size_t>(t) * W;
    const float *expected_y = expected.data() + static_cast<size_t>(t) * W;
    for (unsigned int c = 0; c < W; ++c) {
      EXPECT_NEAR(y_cur[c], expected_y[c], 1.0e-6f);
      EXPECT_NEAR(y_cur[c], prefill_y[c], 1.0e-6f);

      const float expected_s0 =
        t >= 1 ? input[static_cast<size_t>(t - 1) * W + c] : 0.0f;
      const float expected_s1 = input[static_cast<size_t>(t) * W + c];
      EXPECT_NEAR(state[c], expected_s0, 1.0e-6f);
      EXPECT_NEAR(state[W + c], expected_s1, 1.0e-6f);
    }
  }
}
#endif

TEST(nntrainer_cpu_backend_standalone, q4_K_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q4_K(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q4_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);
}

TEST(nntrainer_cpu_backend_standalone, q6_K_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * ne0 / q6_k_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q6_K(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q6_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);
}

TEST(nntrainer_cpu_backend_standalone, q4_0_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q4_0_block_size = QK4_0;
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t num_blocks = (K * N) / q4_0_block_size;
  size_t data_size = num_blocks * q4_0_type_size;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q4_0(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q4_0(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);
}

TEST(nntrainer_cpu_backend_standalone, q4_0_rowwise_gemv) {
  nntrainer::init_backend();

  // N deliberately does not align to the q4_0x4/q4_0x8 packed kernels. The
  // row-wise path must support arbitrary vocabulary sizes.
  const unsigned int N = 37;
  const unsigned int K = 256;
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> activation = generate_random_vector<float>(K);

  const size_t weight_size =
    static_cast<size_t>(N) * K / QK4_0 * sizeof(block_q4_0_testonly);
  std::vector<char> q4_weight(weight_size);
  nntrainer::quantize_q4_0(weight.data(), q4_weight.data(), N, K, nullptr);

  std::vector<char> q8_activation(nntrainer::q4_0_gemv_activation_size(K));
  nntrainer::quantize_q4_0_gemv_activation(K, activation.data(),
                                           q8_activation.data());

  std::vector<float> output(N);
  const unsigned int row_split = 17;
  nntrainer::gemv_q4_0_rowwise_range(0, row_split, K, q8_activation.data(),
                                     q4_weight.data(), output.data());
  nntrainer::gemv_q4_0_rowwise_range(row_split, N, K, q8_activation.data(),
                                     q4_weight.data(), output.data());

  std::vector<float> dequant_activation(K);
  nntr_dequantize_row_q8_0(q8_activation.data(), dequant_activation.data(), K);

  std::vector<float> dequant_weight(K);
  const size_t weight_row_size = weight_size / N;
  for (unsigned int row = 0; row < N; ++row) {
    nntrainer::dequantize_row_q4_0(q4_weight.data() + row * weight_row_size,
                                   dequant_weight.data(), K);
    const float expected = nntrainer::sdot(K, dequant_weight.data(), 1,
                                           dequant_activation.data(), 1);
    EXPECT_NEAR(output[row], expected, 1.0e-3f) << "row=" << row;
  }
}

/**
 * @brief The lm_head's two paths agree
 *
 * A tied model stores one canonical Q4_0 weight so the embedding lookup can
 * address a token's row, which leaves the lm_head on the per-row vec_dot at
 * 2.9 GB/s against 22 for a repacked FC (doc 46 sections 44 and 45).
 * TieWordEmbedding fixes that by keeping a blocked twin and calling the same
 * gemm_q4_0 the fully_connected layers use.
 *
 * This is that substitution, on the shape of the real lm_head's inner
 * dimension: repack the canonical rows with nrow = vocabulary and k =
 * hidden, then check gemm_q4_0 against gemv_q4_0_rowwise_range on the same
 * bytes. Nothing downstream would catch a disagreement -- the layer has no
 * fallback to compare against at runtime, and wrong logits read as
 * plausible-but-wrong text, not as a failure.
 */
TEST(nntrainer_cpu_backend_standalone, q4_0_blocked_lmhead_matches_rowwise) {
  nntrainer::init_backend();

  // vocab % 4 and hidden % 32, which is what the layer checks before it
  // builds the twin; hidden is the model's real 2048.
  const unsigned int vocab = 512;
  const unsigned int hidden = 2048;
  std::vector<float> weight = generate_random_vector<float>(vocab * hidden);
  std::vector<float> activation = generate_random_vector<float>(hidden);

  const size_t weight_size =
    static_cast<size_t>(vocab) * hidden / QK4_0 * sizeof(block_q4_0_testonly);
  std::vector<char> canonical(weight_size);
  nntrainer::quantize_q4_0(weight.data(), canonical.data(), vocab, hidden,
                           nullptr);

  // The row-wise path, as the layer ran it before.
  std::vector<char> q8_activation(nntrainer::q4_0_gemv_activation_size(hidden));
  nntrainer::quantize_q4_0_gemv_activation(hidden, activation.data(),
                                           q8_activation.data());
  std::vector<float> want(vocab, 0.0f);
  nntrainer::gemv_q4_0_rowwise_range(0, vocab, hidden, q8_activation.data(),
                                     canonical.data(), want.data());

  // The blocked path. Same argument order the layer uses: nrow is the
  // vocabulary, k is the hidden size, and nothing is transposed because the
  // tied weight is already [vocab, hidden].
  std::vector<char> blocked(weight_size);
  nntrainer::repack_q4_0(blocked.data(), canonical.data(), weight_size, vocab,
                         hidden);
  std::vector<float> got(vocab, 0.0f);
  nntrainer::gemm_q4_0<float>(1, vocab, hidden, activation.data(), hidden,
                              blocked.data(), vocab, got.data(), vocab);

  // Both quantize the activation to Q8_0 and reduce in int32, so they differ
  // only in the order the per-block f32 scales accumulate.
  size_t bad = 0;
  for (unsigned int v = 0; v < vocab; ++v) {
    const float tol = 1.0e-3f * std::max(1.0f, std::fabs(want[v]));
    if (std::fabs(got[v] - want[v]) > tol) {
      if (bad < 4)
        ADD_FAILURE() << "logit " << v << ": blocked " << got[v]
                      << " vs rowwise " << want[v];
      ++bad;
    }
  }
  EXPECT_EQ(bad, 0u) << bad << " of " << vocab << " logits disagree -- the "
                     << "repack argument order or the gemm leading "
                     << "dimensions are wrong";
}

/**
 * @brief Test quantize -> repack -> unpack -> dequantize pipeline.
 *
 * Verifies that unpack_q4_0 correctly reverses the repack_q4_0 operation
 * so that the subsequent dequantize_row_q4_0 produces the same result as
 * direct quantize -> dequantize (without repack)
 *
 * NOTE: repack_q4_0 produces q4_0x4 on ARM and q4_0x8 on x86 backends.
 *       unpack_q4_0 is the matching inverse for each backend.
 */
TEST(nntrainer_cpu_backend_standalone, q4_0_repack_unpack_dequantize) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);

  // --- Path A: quantize -> dequantize (reference) ---
  int64_t q4_0_block_size = QK4_0;
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  size_t data_size =
    (static_cast<size_t>(K) * N / q4_0_block_size) * q4_0_type_size;

  std::vector<char> q4_weight(data_size);
  nntrainer::quantize_q4_0(weight.data(), q4_weight.data(), N, K, nullptr);

  std::vector<float> ref_dequantized(N * K);
  nntrainer::dequantize_row_q4_0(q4_weight.data(), ref_dequantized.data(),
                                 static_cast<int64_t>(K) * N);

  // --- Path B: quantize -> repack -> unpack -> dequantize ---
  std::vector<char> repacked(data_size);
  nntrainer::repack_q4_0(repacked.data(), q4_weight.data(), data_size, N, K);

  std::vector<char> unpacked(data_size);
  nntrainer::unpack_q4_0(repacked.data(), unpacked.data(), data_size, N, K);

  std::vector<float> roundtrip_dequantized(N * K);
  nntrainer::dequantize_row_q4_0(unpacked.data(), roundtrip_dequantized.data(),
                                 static_cast<int64_t>(K) * N);

  // --- Verify: Path A == Path B (bit-exact after XOR round-trip) ---
  for (unsigned int i = 0; i < N * K; ++i) {
    EXPECT_EQ(ref_dequantized[i], roundtrip_dequantized[i])
      << "Mismatch at index " << i;
  }
}

/**
 * @brief The standalone legacy-QINT4 record reader
 *   (Int4Utils::readLegacyQint4RecordToQs4cx) must reproduce, byte for byte,
 *   what the in-place QINT4 -> QS4CX transcode produces (sectionAToPlain plus
 *   an fp16 -> fp32 scale widening). This is what lets the reader replace the
 *   removed Int4QTensor load path.
 */
TEST(nntrainer_cpu_backend_standalone, qint4_legacy_reader_to_qs4cx) {
  nntrainer::init_backend();

  // Section A packing needs N % 4 == 0 and K % 32 == 0.
  const std::vector<std::pair<size_t, size_t>> shapes = {
    {512, 768}, {2048, 2048}, {256, 2048}, {4, 32}, {1024, 2560}};

  for (auto &shape : shapes) {
    const size_t N = shape.first;
    const size_t K = shape.second;
    std::vector<float> weight = generate_random_vector<float>(N * K);

    // What a legacy QINT4 (0x06) .bin stores: Section A nibbles + fp16 scales.
    std::vector<uint8_t> sec_a;
    std::vector<uint16_t> fp16_scales;
    nntrainer::Int4Utils::quantizeAndPackKai(weight.data(), N, K, sec_a,
                                             fp16_scales);

    // Reference: exactly what the QINT4 -> QS4CX transcode does for a record
    // whose payload is sec_a and whose scales are fp16_scales.
    const size_t plain_bytes = N * ((K + 1) / 2);
    std::vector<uint8_t> ref_plain(plain_bytes);
    nntrainer::Int4Utils::sectionAToPlain(sec_a.data(), N, K, ref_plain.data());
    std::vector<float> ref_scales(N);
    for (size_t n = 0; n < N; ++n)
      ref_scales[n] = nntrainer::compute_fp16_to_fp32(fp16_scales[n]);

    // Assemble the on-disk record: [u16 = 0x06][sec_a nibbles][fp16 scales].
    const uint16_t scheme = 0x0006; // QScheme::QS4CX
    std::vector<uint8_t> record;
    record.insert(record.end(), (const uint8_t *)&scheme,
                  (const uint8_t *)&scheme + sizeof(uint16_t));
    record.insert(record.end(), sec_a.begin(), sec_a.end());
    record.insert(record.end(), (const uint8_t *)fp16_scales.data(),
                  (const uint8_t *)fp16_scales.data() +
                    fp16_scales.size() * sizeof(uint16_t));

    // Standalone reader output (no Int4QTensor).
    std::vector<uint8_t> got_plain(plain_bytes);
    std::vector<float> got_scales(N);
    nntrainer::Int4Utils::readLegacyQint4RecordToQs4cx(
      record.data(), record.size(), N, K, got_plain.data(), got_scales.data());

    ASSERT_EQ(got_plain, ref_plain)
      << "plain nibble mismatch for N=" << N << " K=" << K;
    for (size_t n = 0; n < N; ++n)
      ASSERT_EQ(got_scales[n], ref_scales[n])
        << "fp32 scale mismatch n=" << n << " (N=" << N << " K=" << K << ")";
  }
}

/**
 * @brief The fused, row-range fp16-activation rhs builder
 *   (Int4Utils::packPlainToKaiRhsPackedRows) must give, byte for byte, the
 *   buffer the two-step path gives (packPlainToSectionA, then
 *   assembleKaiRhsPacked over fp16-rounded scales), whether it is filled in
 *   one call or in disjoint pieces. Covers a partial last super-row
 *   (N % 4 != 0) and a K that is not a multiple of 32.
 */
TEST(nntrainer_cpu_backend_standalone, qs4cx_kai_rhs_rows_match_two_step) {
  nntrainer::init_backend();

  const std::vector<std::pair<size_t, size_t>> shapes = {
    {4, 32}, {10, 64}, {64, 96}, {130, 80}, {1536, 512}};

  for (auto &shape : shapes) {
    const size_t N = shape.first;
    const size_t K = shape.second;
    const size_t row_bytes = (K + 1) / 2;

    std::mt19937 rng(static_cast<unsigned int>(N * 131 + K));
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(1e-4f, 0.1f);
    std::vector<uint8_t> plain(N * row_bytes);
    for (auto &b : plain)
      b = static_cast<uint8_t>(byte_dist(rng));
    std::vector<float> scales(N);
    for (auto &s : scales)
      s = scale_dist(rng);

    // Two-step reference.
    std::vector<uint8_t> sec_a(
      nntrainer::Int4Utils::kaiNibblePayloadBytes(N, K));
    nntrainer::Int4Utils::packPlainToSectionA(plain.data(), N, K, sec_a.data());
    std::vector<uint16_t> fp16_scales(N);
    for (size_t n = 0; n < N; ++n)
      fp16_scales[n] = nntrainer::compute_fp32_to_fp16(scales[n]);
    std::vector<uint8_t> ref;
    nntrainer::Int4Utils::assembleKaiRhsPacked(sec_a.data(), fp16_scales.data(),
                                               N, K, ref);
    ASSERT_EQ(ref.size(), nntrainer::Int4Utils::kaiRhsPackedBytes(N, K));

    const size_t super_rows =
      (N + nntrainer::Int4Utils::KAI_NR - 1) / nntrainer::Int4Utils::KAI_NR;

    // One call over every super-row, into a poisoned buffer.
    std::vector<uint8_t> whole(ref.size(), 0x5A);
    nntrainer::Int4Utils::packPlainToKaiRhsPackedRows(
      plain.data(), scales.data(), N, K, 0, super_rows, whole.data());
    EXPECT_EQ(whole, ref) << "single call mismatch N=" << N << " K=" << K;

    // Disjoint pieces of 3 super-rows, filled back to front; the first piece
    // filled may run past the end, which is clamped.
    std::vector<uint8_t> pieces(ref.size(), 0xA5);
    for (size_t sr = (super_rows - 1) / 3 * 3 + 3; sr >= 3; sr -= 3)
      nntrainer::Int4Utils::packPlainToKaiRhsPackedRows(
        plain.data(), scales.data(), N, K, sr - 3, sr, pieces.data());
    EXPECT_EQ(pieces, ref) << "split fill mismatch N=" << N << " K=" << K;
  }
}

/**
 * @brief htp_qs4cx_from_q4_0x4 against the weight it was requantized from.
 *
 * This is a real requantization (Q4_0's per-32-block scale down to qs4cx's
 * one-scale-per-column), not a bit reshuffle, so some accuracy loss beyond
 * Q4_0's own is expected and is what this test measures rather than
 * assumes -- see htp_q4_0_convert.h. Forces the ARM (q4_0x4) repack layout
 * explicitly rather than relying on this host's default, because that is
 * the only layout htp_qs4cx_from_q4_0x4 understands (ENABLE_HEXKL only
 * ever builds for Android/ARM64); q4_0_utils.cpp's dequantizeQ4_0x4 is
 * portable, so this is fully host-testable without a device.
 */
TEST(nntrainer_cpu_backend_standalone, htp_qs4cx_from_q4_0x4_accuracy) {
  nntrainer::init_backend();

  const unsigned int K = 768; // must be a multiple of 32 (Q4_0 block size)
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);

  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  size_t data_size = (static_cast<size_t>(K) * N / QK4_0) * q4_0_type_size;

  std::vector<char> q4_weight(data_size);
  nntrainer::quantize_q4_0(weight.data(), q4_weight.data(), N, K, nullptr);

  std::vector<char> repacked(data_size);
  nntrainer::repack_q4_0(repacked.data(), q4_weight.data(), data_size, N, K,
                         ml::train::ISA::ARM);

  std::vector<int8_t> q_w4_i8(static_cast<size_t>(K) * N);
  std::vector<float> w_scale(N);
  std::vector<int32_t> colsum_w(N);
  nntrainer::htp_qs4cx_from_q4_0x4(repacked.data(), K, N, q_w4_i8.data(),
                                   w_scale.data(), colsum_w.data());

  // colsum_w must match a direct sum over the emitted int4 values -- this
  // is the term HexKL's dequant uses to correct for unsigned activations,
  // so a mismatch here is a silent wrong-answer bug, not a rounding one.
  for (unsigned int n = 0; n < N; ++n) {
    int32_t sum = 0;
    for (unsigned int k = 0; k < K; ++k)
      sum += q_w4_i8[static_cast<size_t>(k) * N + n];
    EXPECT_EQ(sum, colsum_w[n]) << "colsum mismatch at n=" << n;
  }

  // Reconstruct and compare against the ORIGINAL fp32 weight (not Q4_0's own
  // dequant) so the printed number is the whole double-quantization cost,
  // the number that actually reaches the matmul.
  float max_abs_err = 0.0f, sum_abs_err = 0.0f;
  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = 0; k < K; ++k) {
      const float approx = q_w4_i8[static_cast<size_t>(k) * N + n] * w_scale[n];
      const float err =
        std::fabs(approx - weight[static_cast<size_t>(n) * K + k]);
      max_abs_err = std::max(max_abs_err, err);
      sum_abs_err += err;
    }
  }
  const float mean_abs_err = sum_abs_err / (static_cast<float>(N) * K);
  std::cout << "htp_qs4cx_from_q4_0x4: max_abs_err=" << max_abs_err
            << " mean_abs_err=" << mean_abs_err << std::endl;

  // generate_random_vector's range is the implicit full scale here; a 4-bit
  // per-channel quantizer's own noise floor is ~1/30 of a channel's range,
  // and Q4_0's is comparable, so bound generously above their sum rather
  // than pin an exact number -- this is a smoke bound against a layout or
  // sign bug, not a tight accuracy contract.
  EXPECT_LT(max_abs_err, 0.5f);
  EXPECT_LT(mean_abs_err, 0.1f);
}

/**
 * @brief The offline WH packer sees the same values the load path does
 *
 * quantize_stream's QS4CX_WH branch unpacks quant_qs4cx_f32's nibbles itself
 * -- q + 8 stored unsigned, even k in the low half -- and sums them per
 * output channel before calling whPack. htp_qs4cx_from_packed does the same
 * unpacking at load time and is device-verified, so it is the reference: if
 * the two disagree, a model quantized to QS4CX_WH holds different weights
 * than the same model quantized to QS4CX, and nothing downstream would say
 * so. whPack itself is checked against the DSP's own bake by
 * unittest_hvx_mm_u8i4's WhPackReferenceMatchesDspBake.
 */
TEST(nntrainer_cpu_backend_standalone, wh_pack_unpacks_like_the_load_path) {
  nntrainer::init_backend();

  const unsigned int K = 64, N = 96; // not square, both multiples of 32
  std::vector<float> weight = generate_random_vector<float>(N * K);

  const size_t nibble_bytes = static_cast<size_t>(N) * ((K + 1) / 2);
  std::vector<uint8_t> qs4cx(nibble_bytes + N * sizeof(float));
  float *scales = reinterpret_cast<float *>(qs4cx.data() + nibble_bytes);
  nntrainer::quant_qs4cx_f32(N, K, weight.data(), qs4cx.data(), scales, true);

  std::vector<int8_t> want_rm(static_cast<size_t>(K) * N);
  std::vector<float> want_scale(N);
  std::vector<int32_t> want_colsum(N);
  nntrainer::htp_qs4cx_from_packed(qs4cx.data(), scales, K, N, want_rm.data(),
                                   want_scale.data(), want_colsum.data());

  // The quantizer's own unpack, character for character with what it writes.
  std::vector<int8_t> rm(static_cast<size_t>(K) * N);
  std::vector<int32_t> colsum(N, 0);
  const size_t stride = (K + 1) / 2;
  for (size_t n = 0; n < N; ++n) {
    const uint8_t *row = qs4cx.data() + n * stride;
    int32_t sum = 0;
    for (size_t k = 0; k < K; ++k) {
      const uint8_t byte = row[k >> 1];
      const uint8_t nibble = (k & 1u) ? (byte >> 4) : (byte & 0x0Fu);
      const int8_t q = static_cast<int8_t>(static_cast<int32_t>(nibble) - 8);
      rm[k * N + n] = q;
      sum += q;
    }
    colsum[n] = sum;
  }
  EXPECT_EQ(rm, want_rm);
  EXPECT_EQ(colsum, want_colsum);

  // A column sum is at most K values in [-8, 7], so f32 carries it exactly --
  // which is what lets it travel in the file beside the scales.
  for (unsigned int n = 0; n < N; ++n) {
    EXPECT_EQ(static_cast<int32_t>(static_cast<float>(colsum[n])), colsum[n]);
  }

  // And the packing is reversible: every value lands where whSlot says, so
  // reading it back gives the weight again. whUnpack is that read, and the
  // reference path NNTR_MOE_DIFF dequantizes a whole expert through it, so
  // the round trip is what says those reference values mean anything.
  std::vector<uint8_t> wh(nntrainer::whBytes(K, N));
  nntrainer::whPack(rm.data(), K, N, wh.data());
  EXPECT_EQ(wh.size(), nibble_bytes);

  std::vector<int8_t> back(static_cast<size_t>(K) * N);
  nntrainer::whUnpack(wh.data(), K, N, back.data());
  EXPECT_EQ(back, rm) << "whPack and whUnpack disagree about where values go";
}

/**
 * @brief QS4CX multiplication respects the platform's packing contract.
 *
 * ARM kernels consume a packed RHS. The x86 fallback rejects packing and
 * multiplies the original nibbles and scales. Check both contracts before
 * comparing the product against dequantized weights. Activation quantization
 * requires an SNR bound rather than equality.
 */
TEST(nntrainer_cpu_backend_standalone, qs4cx_tensor_packs_and_dots) {
  nntrainer::init_backend();

  const unsigned int M = 4, K = 64, N = 96;
  std::vector<float> w_nk = generate_random_vector<float>(N * K); // [N][K]
  nntrainer::TensorDim wdim(1, 1, K, N,
                            {ml::train::TensorDim::Format::NCHW,
                             ml::train::TensorDim::DataType::QS4CX});
  nntrainer::Tensor w(wdim, true);
  nntrainer::quant_qs4cx_f32(N, K, w_nk.data(), w.getData<uint8_t>(),
                             w.getScale<float>(), true);
#if defined(__aarch64__) || defined(_M_ARM64)
  ASSERT_NO_THROW(w.pack());
#else
  EXPECT_THROW(w.pack(), std::runtime_error);
  EXPECT_FALSE(w.isPacked());
#endif

  std::vector<float> deq(static_cast<size_t>(N) * K);
  nntrainer::dequant_qs4cx_f32(N, K, w.getData<uint8_t>(), w.getScale<float>(),
                               deq.data(), true);

  nntrainer::Tensor x(1, 1, M, K);
  std::vector<float> xv = generate_random_vector<float>(M * K);
  std::copy(xv.begin(), xv.end(), x.getData<float>());
  nntrainer::Tensor y(1, 1, M, N);
  x.dot(w, y);

  double signal = 0.0, noise = 0.0;
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double ref = 0.0;
      for (unsigned int k = 0; k < K; ++k)
        ref += (double)xv[m * K + k] * deq[static_cast<size_t>(n) * K + k];
      const double d = (double)y.getData<float>()[m * N + n] - ref;
      signal += ref * ref;
      noise += d * d;
    }
  }
  EXPECT_GT(10.0 * std::log10(signal / noise), 30.0);
}

/**
 * @brief The 2-bit palette really is the best four codes, not a good guess
 *
 * whPaletteQuantize decides what a 2-bit expert weighs, and the whole 2-bit
 * effort is gated on one perplexity number measured with it. A k-means that
 * settled into a local minimum would make that number an argument about the
 * fitter rather than about 2 bits, so this checks the claim the header makes:
 * with sixteen candidate codes the nearest-level partition is contiguous, so
 * the DP is exact. Brute force over all 1820 four-subsets says whether it is.
 */
TEST(nntrainer_cpu_backend_standalone, wh_palette_is_optimal_and_restricts) {
  const unsigned int K = 128, N = 64;
  const unsigned int group_k = 32;

  // Codes shaped like a real quantized weight: bell-shaped over [-8, 7].
  std::mt19937 rng(20260923u);
  std::normal_distribution<float> bell(0.0f, 3.0f);
  std::vector<int8_t> rm(static_cast<size_t>(K) * N);
  for (auto &v : rm) {
    int q = static_cast<int>(std::lround(bell(rng)));
    v = static_cast<int8_t>(q < -8 ? -8 : (q > 7 ? 7 : q));
  }
  const std::vector<int8_t> original = rm;

  std::vector<int8_t> pal(
    nntrainer::whPaletteEntries(K, N, group_k, /*per_column=*/false));
  nntrainer::whPaletteQuantize(rm.data(), K, N, group_k, /*per_column=*/false,
                               pal.data());

  const unsigned int groups = nntrainer::whPaletteGroups(K, group_k);
  ASSERT_EQ(pal.size(), groups * nntrainer::WH_PALETTE_LEVELS);

  for (unsigned int g = 0; g < groups; ++g) {
    const int8_t *entry = pal.data() + g * nntrainer::WH_PALETTE_LEVELS;

    // Every value in the group is one of the four, and the nearest one.
    uint64_t hist[nntrainer::WH_PALETTE_CODES] = {0};
    for (unsigned int k = g * group_k; k < (g + 1) * group_k; ++k) {
      for (unsigned int n = 0; n < N; ++n) {
        const size_t i = static_cast<size_t>(k) * N + n;
        ++hist[original[i] + 8];
        EXPECT_EQ(rm[i],
                  nntrainer::detail::whPaletteNearest(original[i], entry));
      }
    }

    const auto error = [&hist](const int8_t p[nntrainer::WH_PALETTE_LEVELS]) {
      double e = 0.0;
      for (int v = 0; v < (int)nntrainer::WH_PALETTE_CODES; ++v) {
        if (!hist[v])
          continue;
        const int8_t q = static_cast<int8_t>(v - 8);
        const double d = q - nntrainer::detail::whPaletteNearest(q, p);
        e += static_cast<double>(hist[v]) * d * d;
      }
      return e;
    };

    double best = std::numeric_limits<double>::max();
    for (int a = 0; a < 16; ++a)
      for (int b = a + 1; b < 16; ++b)
        for (int c = b + 1; c < 16; ++c)
          for (int d = c + 1; d < 16; ++d) {
            const int8_t p[4] = {
              static_cast<int8_t>(a - 8), static_cast<int8_t>(b - 8),
              static_cast<int8_t>(c - 8), static_cast<int8_t>(d - 8)};
            best = std::min(best, error(p));
          }
    EXPECT_LE(error(entry), best + 1e-9) << "group " << g << " is not optimal";
  }

  // And the point of all this: two bits an entry is enough to name them.
  std::set<int8_t> used(rm.begin(), rm.end());
  EXPECT_LE(used.size(), groups * nntrainer::WH_PALETTE_LEVELS);
}

/**
 * @brief whSourcePageRange never hands back a byte it was not given
 *
 * The HTP arena drops a weight's pages once the bytes are copied in, and
 * every weight is a slice of ONE pool allocation shared with its neighbours.
 * An off-by-one page here is not a crash, it is the neighbouring weight
 * reading as zeros -- a silently wrong matmul. So the property under test is
 * containment, checked on the shapes this model actually registers plus the
 * degenerate ones.
 */
TEST(nntrainer_cpu_backend_standalone, wh_source_page_range_stays_inside) {
  const size_t ps = 4096;
  // A pool slice can start anywhere; 512 is the DSP's weight alignment and
  // the smallest offset a real weight is likely to sit at.
  const uintptr_t bases[] = {0x10000000u, 0x10000200u, 0x10000fffu};
  const size_t lens[] = {
    0u,       1u, ps - 1u, ps, ps + 1u,
    1835008u, // 1792x2048 down, whBytes
    3670016u  // 2048x3584 gate_up, the model's largest
  };

  for (uintptr_t base : bases) {
    for (size_t len : lens) {
      const void *src = reinterpret_cast<const void *>(base);
      uintptr_t begin = 0;
      size_t span = 0;
      nntrainer::whSourcePageRange(src, len, ps, &begin, &span);
      if (span == 0)
        continue;
      EXPECT_GE(begin, base) << "released below the weight";
      EXPECT_LE(begin + span, base + len) << "released past the weight";
      EXPECT_EQ(begin % ps, 0u) << "madvise needs a page-aligned start";
      EXPECT_EQ(span % ps, 0u) << "madvise needs a whole number of pages";
      // Whatever is dropped, at most one page at each end is not.
      EXPECT_LE(len - span, 2u * ps) << "gave up more than the two end pages";
    }
  }

  // A weight smaller than a page, or one that covers no whole page, releases
  // nothing at all rather than guessing.
  uintptr_t begin = 0;
  size_t span = 0;
  nntrainer::whSourcePageRange(reinterpret_cast<const void *>(0x10000200u),
                               ps - 1u, ps, &begin, &span);
  EXPECT_EQ(span, 0u);
}

/**
 * @brief htp_qs4cx_from_packed rearranges, it does not requantize
 *
 * A weight quantized once from FP32 straight into QS4CX and then handed to
 * HexKL must be at least as accurate as the same weight routed through
 * Q4_0 first, because the second route quantizes twice. This test pins
 * that ordering rather than just checking the new path runs.
 */
TEST(nntrainer_cpu_backend_standalone, htp_qs4cx_from_packed_accuracy) {
  nntrainer::init_backend();

  const unsigned int K = 768; // multiple of 32 so the Q4_0 comparison is legal
  const unsigned int N = 512;

  // [N][K] row-major: one output channel per row, which is what both
  // quantize_q4_0(.., N, K, ..) and quant_qs4cx_f32(N, K, .., is_nxk=true)
  // expect, so the same buffer feeds both routes.
  std::vector<float> weight = generate_random_vector<float>(N * K);

  // --- Route A: FP32 -> QS4CX (one quantization), then rearrange ---
  const size_t nibble_bytes = static_cast<size_t>(N) * ((K + 1) / 2);
  std::vector<uint8_t> qs4cx(nibble_bytes + N * sizeof(float));
  float *qs4cx_scales = reinterpret_cast<float *>(qs4cx.data() + nibble_bytes);
  nntrainer::quant_qs4cx_f32(N, K, weight.data(), qs4cx.data(), qs4cx_scales,
                             true);

  std::vector<int8_t> q_direct(static_cast<size_t>(K) * N);
  std::vector<float> scale_direct(N);
  std::vector<int32_t> colsum_direct(N);
  nntrainer::htp_qs4cx_from_packed(qs4cx.data(), qs4cx_scales, K, N,
                                   q_direct.data(), scale_direct.data(),
                                   colsum_direct.data());

  // colsum is HexKL's correction term for unsigned activations; a wrong one
  // is a silent wrong answer, so recompute it independently.
  for (unsigned int n = 0; n < N; ++n) {
    int32_t sum = 0;
    for (unsigned int k = 0; k < K; ++k)
      sum += q_direct[static_cast<size_t>(k) * N + n];
    EXPECT_EQ(sum, colsum_direct[n]) << "colsum mismatch at n=" << n;
  }

  // Every emitted value must be a legal int4.
  for (size_t i = 0; i < q_direct.size(); ++i) {
    ASSERT_GE(q_direct[i], -8);
    ASSERT_LE(q_direct[i], 7);
  }

  // --- Route B: FP32 -> Q4_0 -> qs4cx (two quantizations) ---
  const size_t q4_0_bytes =
    (static_cast<size_t>(K) * N / QK4_0) * sizeof(block_q4_0_testonly);
  std::vector<char> q4_weight(q4_0_bytes);
  nntrainer::quantize_q4_0(weight.data(), q4_weight.data(), N, K, nullptr);
  std::vector<char> repacked(q4_0_bytes);
  nntrainer::repack_q4_0(repacked.data(), q4_weight.data(), q4_0_bytes, N, K,
                         ml::train::ISA::ARM);

  std::vector<int8_t> q_hop(static_cast<size_t>(K) * N);
  std::vector<float> scale_hop(N);
  std::vector<int32_t> colsum_hop(N);
  nntrainer::htp_qs4cx_from_q4_0x4(repacked.data(), K, N, q_hop.data(),
                                   scale_hop.data(), colsum_hop.data());

  // --- Compare both reconstructions against the ORIGINAL fp32 ---
  auto mean_abs_err = [&](const std::vector<int8_t> &q,
                          const std::vector<float> &scale) {
    double sum_abs_err = 0.0;
    for (unsigned int n = 0; n < N; ++n) {
      for (unsigned int k = 0; k < K; ++k) {
        const float approx = q[static_cast<size_t>(k) * N + n] * scale[n];
        sum_abs_err +=
          std::fabs(approx - weight[static_cast<size_t>(n) * K + k]);
      }
    }
    return static_cast<float>(sum_abs_err / (static_cast<double>(N) * K));
  };

  const float err_direct = mean_abs_err(q_direct, scale_direct);
  const float err_hop = mean_abs_err(q_hop, scale_hop);
  std::cout << "htp_qs4cx_from_packed: mean_abs_err=" << err_direct
            << " (direct)  vs " << err_hop << " (via Q4_0)" << std::endl;

  // The whole reason for quantizing the FFN straight to QS4CX: one
  // quantization must beat two. Strict, because the comparison is on the
  // same weight with the same per-channel scheme -- the only difference is
  // the extra Q4_0 hop.
  EXPECT_LT(err_direct, err_hop);
}

/**
 * @brief htp_quant_pack_u8_ah's own math, checked in isolation.
 *
 * This cannot compare against the real HVX path bit-for-bit -- that only
 * exists on-device (nntr_hvx_mm_u8i4_from_f32 returns act_u8_ah/act_scale/
 * act_zp for exactly that comparison; see docs/htp_attention's u8-boundary
 * task for the device-side differential gate). What this test can check
 * without a device: every invariant the DSP path's own contract states --
 * per-row scale/zp bounds, the AH tile addressing formula, padding rows
 * zeroed, and that dequantizing the packed bytes reproduces the input
 * within one quantization step's error.
 */
TEST(nntrainer_cpu_backend_standalone, htp_quant_pack_u8_ah_accuracy) {
  const uint32_t M = 37; // deliberately not a multiple of HTP_ACT_TILE_ROW
  const uint32_t K = 128;

  std::vector<float> x = generate_random_vector<float>(M * K);

  const uint32_t m_pad = nntrainer::htp_act_m_pad(M);
  ASSERT_EQ(m_pad, 64u);

  std::vector<uint8_t> out_ah(static_cast<size_t>(m_pad) * K);
  std::vector<float> act_scale(m_pad);
  std::vector<int32_t> act_zp(m_pad);
  nntrainer::htp_quant_pack_u8_ah(x.data(), M, K, out_ah.data(),
                                  act_scale.data(), act_zp.data());

  const uint32_t n_ktiles = K / nntrainer::HTP_ACT_TILE_INNER;
  float max_abs_err = 0.0f;
  for (uint32_t m = 0; m < M; ++m) {
    ASSERT_GE(act_zp[m], 0) << "m=" << m;
    ASSERT_LE(act_zp[m], 255) << "m=" << m;
    ASSERT_GT(act_scale[m], 0.0f) << "m=" << m;

    const uint32_t rb = m / nntrainer::HTP_ACT_TILE_ROW;
    const uint32_t r = m % nntrainer::HTP_ACT_TILE_ROW;
    for (uint32_t k = 0; k < K; ++k) {
      const uint32_t kt = k / nntrainer::HTP_ACT_TILE_INNER;
      const uint32_t c = k % nntrainer::HTP_ACT_TILE_INNER;
      const size_t idx =
        (static_cast<size_t>(rb) * n_ktiles + kt) *
          nntrainer::HTP_ACT_TILE_BYTES +
        static_cast<size_t>(r) * nntrainer::HTP_ACT_TILE_INNER + c;
      const uint8_t q = out_ah[idx];
      const float approx = act_scale[m] * (static_cast<float>(q) - act_zp[m]);
      const float err = std::fabs(approx - x[static_cast<size_t>(m) * K + k]);
      max_abs_err = std::max(max_abs_err, err);
    }
  }
  std::cout << "htp_quant_pack_u8_ah: max_abs_err=" << max_abs_err << std::endl;
  // One quantization step is scale (range/255); generate_random_vector's
  // range makes this a loose smoke bound against a layout or rounding bug,
  // not a tight accuracy contract -- same spirit as the qs4cx tests above.
  EXPECT_LT(max_abs_err, 0.5f);

  // Padding rows (M..m_pad) must be exactly zero -- the DSP dequant path
  // never reads them, but a stray nonzero byte here would mean the tile
  // addressing formula overlapped a real row into padding space.
  for (uint32_t m = M; m < m_pad; ++m) {
    EXPECT_FLOAT_EQ(act_scale[m], 1.0f) << "m=" << m;
    EXPECT_EQ(act_zp[m], 0) << "m=" << m;
    const uint32_t rb = m / nntrainer::HTP_ACT_TILE_ROW;
    const uint32_t r = m % nntrainer::HTP_ACT_TILE_ROW;
    for (uint32_t kt = 0; kt < n_ktiles; ++kt) {
      const size_t base =
        (static_cast<size_t>(rb) * n_ktiles + kt) *
          nntrainer::HTP_ACT_TILE_BYTES +
        static_cast<size_t>(r) * nntrainer::HTP_ACT_TILE_INNER;
      for (uint32_t c = 0; c < nntrainer::HTP_ACT_TILE_INNER; ++c) {
        EXPECT_EQ(out_ah[base + c], 0) << "m=" << m << " k=" << (kt * 32 + c);
      }
    }
  }
}

float test_gemm_q4_0(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // needed to initialize f16 tables

  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t q4_0_block_size = 32;
  int64_t q4_0_num_blocks = (K * N) / q4_0_block_size;
  size_t q4_0_data_size = q4_0_type_size * N / q4_0_block_size;
  q4_0_data_size *= K;
  std::vector<char> q4_0_offline_qWeight = std::vector<char>(q4_0_data_size);

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  char *q4_0_offline_qWeight_ptr = (char *)q4_0_offline_qWeight.data();
  nntrainer::quantize_q4_0(weights, (void *)q4_0_offline_qWeight_ptr, N, K,
                           nullptr);

  // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the
  // model weights. It's a one-time operation)
  std::vector<char> q4_0_repacked_qWeight = std::vector<char>(q4_0_data_size);
  nntrainer::repack_q4_0(q4_0_repacked_qWeight.data(), q4_0_offline_qWeight_ptr,
                         q4_0_data_size, N, K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(static_cast<size_t>(M) * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_0: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

float test_gemm_q4_K(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * N / q4_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  nntrainer::quantize_q4_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the
  // model weights. It's a one-time operation)
  std::vector<char> repacked_qWeight = std::vector<char>(data_size);
  nntrainer::repack_q4_K(repacked_qWeight.data(), offline_qWeight_ptr,
                         data_size, N, K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(static_cast<size_t>(M) * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
                       N, dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

float test_gemm_q6_K(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * N / q6_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  nntrainer::quantize_q6_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  // Step2. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(static_cast<size_t>(M) * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q6_K(M, N, K, activations, K, (void *)offline_qWeight_ptr, N,
                       dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q6_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

static void run_quant_test(const uint32_t M, const uint32_t K, const uint32_t N,
                           float &q4_0_mse, float &q4_k_mse, float &q6_k_mse,
                           bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "[INFO] Quantization Test (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
  }
  ///@note A(M, K) * W.T(N, K) = (M, N)
  ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

  ///@note q4_K GEMM is a Row-Major, transB GEMM
  std::vector<float> activation =
    generate_random_vector<float>(static_cast<size_t>(M) * K);
  std::vector<float> weight =
    generate_random_vector<float>(static_cast<size_t>(N) * K);
  std::vector<float> ref_dst(static_cast<size_t>(M) * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm :    " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }
  q4_0_mse =
    test_gemm_q4_0(M, K, N, weight.data(), activation.data(), ref_dst, print);
  q4_k_mse =
    test_gemm_q4_K(M, K, N, weight.data(), activation.data(), ref_dst, print);
  q6_k_mse =
    test_gemm_q6_K(M, K, N, weight.data(), activation.data(), ref_dst, print);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_457x512x512) {
  const unsigned int M = 457;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_458x512x512) {
  const unsigned int M = 458;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_459x512x512) {
  const unsigned int M = 459;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_512x512x512) {
  const unsigned int M = 512;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x512x512) {
  const unsigned int M = 1;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

static void run_vec_dot_test(const uint32_t K, bool print = false) {
  const int TEST_CNT = 20;
  nanoseconds ref_time = (nanoseconds)0;
  nanoseconds q6_k_time = (nanoseconds)0;
  float ref_result, result;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> activation = generate_random_vector<float, false>(K);
    std::vector<float> weight = generate_random_vector<float, false>(K);

    {
      // GROUND TRUTH sdot for reference
      auto t1 = high_resolution_clock::now();
      ref_result = nntrainer::sdot(K, weight.data(), 1, activation.data(), 1);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_time += dt;
      }
    }

    // Quantization of weights
    int64_t num_blocks = K / 256;
    size_t q6_k_data_size = num_blocks * sizeof(block_q6_K_testonly);
    std::vector<char> q6_K_weight = std::vector<char>(q6_k_data_size);
    nntrainer::quantize_row_q6_K(weight.data(), q6_K_weight.data(), K);

    // Quantization of activations
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    int q8_K_activation_size = sizeof(block_q8_K_testonly) * blocks_per_row;
    std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
    nntrainer::quantize_row_q8_K(activation.data(), v_q8_activation.data(), K);

    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      result =
        nntrainer::dot_q6_K_q8_K(K, q6_K_weight.data(), v_q8_activation.data());
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        q6_k_time += dt;
      }
    }
    EXPECT_NEAR(result, ref_result, 0.25 * K / 256);
  }
  if (print) {
    std::cout << "[INFO] dot_q6_K_q8_K: TEST CNT: " << TEST_CNT << ", K: " << K
              << ", Average ref_time: " << ref_time.count() / TEST_CNT
              << " ns, Average q6_k_time: " << q6_k_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_1024) {
  const uint32_t K = 1024;
  run_vec_dot_test(K);
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_2560) {
  const uint32_t K = 2560;
  run_vec_dot_test(K);
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_10240) {
  const uint32_t K = 10240;
  run_vec_dot_test(K);
}

static void run_ele_mul_test(const unsigned int N, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride,
                             bool print = false) {
  // Z = X ⊙ alpha * Y + beta * Z
  const int TEST_CNT = 20;
  nanoseconds ref_mul_time = (nanoseconds)0;
  nanoseconds mul_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> X =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Y = generate_random_vector<float, false>(
      std::max<size_t>(1, (size_t)N * i_stride));
    std::vector<float> Z =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Z_ref = Z;
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::__fallback_ele_mul(N, X.data(), Y.data(), Z_ref.data(), alpha,
                                    beta, i_stride, o_stride);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_mul_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::ele_mul(N, X.data(), Y.data(), Z.data(), alpha, beta, i_stride,
                         o_stride);
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        mul_time += dt;
      }
    }

    auto mean_squared_error = compute_mse(1, N, Z_ref, Z, false);
    ASSERT_LE(mean_squared_error, 0.00001f);
  }

  if (print) {
    std::cout << "[INFO] ele_mul: TEST CNT: " << TEST_CNT << ", N: " << N
              << ", i_stride: " << i_stride
              << ", Average ref_time: " << ref_mul_time.count() / TEST_CNT
              << " ns, Average mul_time: " << mul_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, ele_mul_3072_istr_0) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 0;
  const unsigned int o_stride = 1;
  run_ele_mul_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_mul_3072_istr_1) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 1;
  const unsigned int o_stride = 1;
  run_ele_mul_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_mul_3072_istr_16_ostr_16) {
  const unsigned int N = 3072;
  const float alpha = 3.f;
  const float beta = 2.f;
  const unsigned int i_stride = 16;
  const unsigned int o_stride = 16;
  run_ele_mul_test(N, alpha, beta, i_stride, o_stride);
}

static void run_ele_add_test(const unsigned int N, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride,
                             bool print = false) {
  // Z = X ⊙ alpha * Y + beta * Z
  const int TEST_CNT = 20;
  nanoseconds ref_add_time = (nanoseconds)0;
  nanoseconds add_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> X =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Y = generate_random_vector<float, false>(
      std::max<size_t>(1, (size_t)N * i_stride));
    std::vector<float> Z =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Z_ref = Z;
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::__fallback_ele_add(N, X.data(), Y.data(), Z_ref.data(), alpha,
                                    beta, i_stride, o_stride);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_add_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::ele_add(N, X.data(), Y.data(), Z.data(), alpha, beta, i_stride,
                         o_stride);
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        add_time += dt;
      }
    }
    auto mean_squared_error = compute_mse(1, N, Z_ref, Z, false);
    ASSERT_LE(mean_squared_error, 0.00001f);
  }
  if (print) {
    std::cout << "[INFO] ele_add: TEST CNT: " << TEST_CNT << ", N: " << N
              << ", i_stride: " << i_stride
              << ", Average ref_time: " << ref_add_time.count() / TEST_CNT
              << " ns, Average add_time: " << add_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, ele_add_3072_istr_0) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 0;
  const unsigned int o_stride = 1;
  run_ele_add_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_add_3072_istr_1) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 1;
  const unsigned int o_stride = 1;
  run_ele_add_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_add_3072_istr_16_ostrid_16) {
  const unsigned int N = 3072;
  const float alpha = 3.f;
  const float beta = 2.f;
  const unsigned int i_stride = 16;
  const unsigned int o_stride = 16;
  run_ele_add_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, softmax_row_inplace) {
  size_t start_row = 0;
  size_t end_row = 3;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  std::vector<float> qk_out = {
    -2.509198f, 5.930860f,  9.014286f,  -6.331305f, 4.639878f,  5.593820f,
    1.973170f,  1.937003f,  -6.879627f, -1.083345f, -6.880110f, -8.000502f,
    -8.838327f, -0.815022f, 7.323523f,  -3.325828f, 2.022300f,  -7.142663f,
    4.161452f,  3.017770f,  -9.588310f, -8.871769f, 9.398197f,  4.439975f,
    6.648853f,  8.771055f,  -5.753218f, -9.984425f, -6.363501f, 9.844232f};
  std::vector<float> ref_out = {
    0.986697f, 0.999999f, 0.405184f, 0.000021f, 0.043301f, 0.040031f,
    0.487615f, 0.999879f, 0.000016f, 0.000018f, 0.012472f, 0.000001f,
    0.000000f, 0.005194f, 0.633859f, 0.000005f, 0.512170f, 0.000114f,
    0.999957f, 0.001083f, 0.000831f, 0.000000f, 0.594816f, 0.994785f,
    0.322840f, 0.959963f, 0.000215f, 0.000007f, 0.000027f, 0.998899f};

  nntrainer::softmax_row_inplace(qk_out.data(), start_row, end_row, num_heads);

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, softmax_row_inplace_sink) {
  size_t start_row = 0;
  size_t end_row = 2;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  std::vector<float> qk_out = {-6.880110f, -8.000502f, -8.838327f, -0.815022f,
                               7.323523f,  -3.325828f, 2.022300f,  -7.142663f,
                               4.161452f,  3.017770f,  -9.588310f, -8.871769f,
                               9.398197f,  4.439975f,  6.648853f,  8.771055f,
                               -5.753218f, -9.984425f, -6.363501f, 9.844232f};
  std::vector<float> sink = {-2.509198f, 5.930860f, 9.014286f, -6.331305f,
                             4.639878f,  5.593820f, 1.973170f, 1.937003f,
                             -6.879627f, -1.083345f};
  std::vector<float> ref_out = {
    0.012472f, 0.000001f, 0.000000f, 0.005194f, 0.633859f, 0.000005f, 0.512170f,
    0.000114f, 0.999957f, 0.001083f, 0.000831f, 0.000000f, 0.594816f, 0.994785f,
    0.322840f, 0.959963f, 0.000215f, 0.000007f, 0.000027f, 0.998899f};

  nntrainer::softmax_row_inplace(qk_out.data(), start_row, end_row, num_heads,
                                 sink.data());

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, softmax_row) {
  size_t start_row = 0;
  size_t end_row = 3;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  std::vector<float> qk_out = {
    -2.509198f, 5.930860f,  9.014286f,  -6.331305f, 4.639878f,  5.593820f,
    1.973170f,  1.937003f,  -6.879627f, -1.083345f, -6.880110f, -8.000502f,
    -8.838327f, -0.815022f, 7.323523f,  -3.325828f, 2.022300f,  -7.142663f,
    4.161452f,  3.017770f,  -9.588310f, -8.871769f, 9.398197f,  4.439975f,
    6.648853f,  8.771055f,  -5.753218f, -9.984425f, -6.363501f, 9.844232f};
  std::vector<float> ref_out = {
    0.986697f, 0.999999f, 0.405184f, 0.000021f, 0.043301f, 0.040031f,
    0.487615f, 0.999879f, 0.000016f, 0.000018f, 0.012472f, 0.000001f,
    0.000000f, 0.005194f, 0.633859f, 0.000005f, 0.512170f, 0.000114f,
    0.999957f, 0.001083f, 0.000831f, 0.000000f, 0.594816f, 0.994785f,
    0.322840f, 0.959963f, 0.000215f, 0.000007f, 0.000027f, 0.998899f};

  nntrainer::softmax_row(qk_out.data(), start_row, end_row, num_heads);

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0001f);
  }
}

#if defined(ENABLE_FP16) || defined(__AVX2__)
TEST(nntrainer_cpu_backend_standalone, compute_kcaches) {
  int num_rows = 1;
  int N = 2;
  int head_dim = 10;
  int group_size = 4;
  int tile_size = 16;
  size_t in_size = N * group_size * head_dim;
  size_t kcache_size = num_rows * N * head_dim;
  size_t output_size = num_rows * N * group_size;
  std::vector<float> in = {
    -2.509198f, 5.930860f,  9.014286f,  -6.331305f, 4.639878f,  5.593820f,
    1.973170f,  1.937003f,  -6.879627f, -1.083345f, -6.880110f, -8.000502f,
    -8.838327f, -0.815022f, 7.323523f,  -3.325828f, 2.022300f,  -7.142663f,
    4.161452f,  3.017770f,  -9.588310f, -8.871769f, 9.398197f,  4.439975f,
    6.648853f,  8.771055f,  -5.753218f, -9.984425f, -6.363501f, 9.844232f,
    -6.331910f, 2.349631f,  -3.915155f, 2.233063f,  0.495129f,  -9.858674f,
    -1.361099f, -9.538752f, -4.175417f, 0.495493f,  2.237058f,  -2.002780f,
    -7.210123f, -9.066687f, -4.157107f, 9.475111f,  -2.672763f, -5.344573f,
    -0.878600f, -8.187871f, 5.703520f,  2.367721f,  -6.006524f, -2.350760f,
    0.284688f,  9.664618f,  1.848291f,  -0.664742f, -9.070992f, 7.198808f,
    2.150897f,  3.606151f,  -6.589518f, -0.990015f, -8.698968f, -9.734701f,
    8.977711f,  8.844034f,  9.312640f,  1.265764f,  6.167947f,  -2.291670f,
    -3.907725f, -9.680675f, -8.046557f, -5.382123f, 3.684660f,  -5.179491f,
    -1.196950f, 3.665271f};
  std::vector<uint16_t> kcache = {3751, 7967, 9507, 1842, 7322, 7799, 5990,
                                  5972, 1568, 4463, 1568, 1008, 590,  4597,
                                  8663, 3343, 6015, 1437, 7083, 6512};
  std::vector<float> ref_out = {0.089252f,  -0.072949f, 0.058948f,  -0.045583f,
                                -0.025812f, -0.002068f, -0.014971f, -0.028027f};
  std::vector<float> output(output_size);

  nntrainer::compute_kcaches<uint16_t>(in.data(), kcache.data(), output.data(),
                                       num_rows, N, head_dim, group_size,
                                       tile_size);

  for (size_t i = 0; i < output_size; i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_out_null) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;
  bool only_convert_to_fp16 = false;

  std::vector<float> inout = {
    -0.250920f, 0.593086f,  0.901429f,  -0.633130f, 0.463988f,  0.559382f,
    0.197317f,  0.193700f,  -0.687963f, -0.108334f, -0.688011f, -0.800050f,
    -0.883833f, -0.081502f, 0.732352f,  -0.332583f, 0.202230f,  -0.714266f,
    0.416145f,  0.301777f,  -0.958831f, -0.887177f, 0.939820f,  0.443998f,
    0.664885f,  0.877105f,  -0.575322f, -0.998442f, -0.636350f, 0.984423f,
    -0.633191f, 0.234963f,  -0.391515f, 0.223306f,  0.049513f,  -0.985867f,
    -0.136110f, -0.953875f, -0.417542f, 0.049549f};
  std::vector<float> cos_ = {-0.508061f, 0.998221f,  -0.733869f, 0.164825f,
                             0.297530f,  -0.591437f, -0.165907f, -0.999895f,
                             -0.163179f, 0.249787};
  std::vector<float> sin_ = {0.809128f,  -0.390829f, -0.112627f, 0.377992f,
                             -0.994569f, -0.641599f, -0.927266f, -0.401338f,
                             0.244335f,  -0.414953f};
  std::vector<float> ref_out = {
    0.684172f,  0.279348f,  -0.761074f, -0.073549f, 0.866425f,  -0.544224f,
    0.154785f,  -0.480342f, 0.010582f,  0.098163f,  0.146526f,  -1.030422f,
    0.547092f,  -0.252752f, -0.243571f, -0.162197f, -0.216517f, 0.636452f,
    -0.236000f, 0.120334f,  0.999478f,  -0.793769f, -0.733800f, -0.011226f,
    0.247067f,  -1.151284f, -0.030760f, 0.615511f,  0.205859f,  0.266457f,
    -0.454117f, 0.581280f,  0.181472f,  0.204634f,  -0.646542f, 0.020328f,
    0.556058f,  1.354487f,  -0.087349f, -0.396113f};

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(), nullptr,
                                      cos_.data(), sin_.data(),
                                      only_convert_to_fp16);

  for (size_t i = 0; i < inout.size(); i++) {
    EXPECT_NEAR(ref_out[i], inout[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_out_uint16) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;
  bool only_convert_to_fp16 = false;

  std::vector<float> inout = {
    -0.250920f, 0.593086f,  0.901429f,  -0.633130f, 0.463988f,  0.559382f,
    0.197317f,  0.193700f,  -0.687963f, -0.108334f, -0.688011f, -0.800050f,
    -0.883833f, -0.081502f, 0.732352f,  -0.332583f, 0.202230f,  -0.714266f,
    0.416145f,  0.301777f,  -0.958831f, -0.887177f, 0.939820f,  0.443998f,
    0.664885f,  0.877105f,  -0.575322f, -0.998442f, -0.636350f, 0.984423f,
    -0.633191f, 0.234963f,  -0.391515f, 0.223306f,  0.049513f,  -0.985867f,
    -0.136110f, -0.953875f, -0.417542f, 0.049549f};
  std::vector<float> cos_ = {-0.508061f, 0.998221f,  -0.733869f, 0.164825f,
                             0.297530f,  -0.591437f, -0.165907f, -0.999895f,
                             -0.163179f, 0.249787f};
  std::vector<float> sin_ = {0.809128f,  -0.390829f, -0.112627f, 0.377992f,
                             -0.994569f, -0.641599f, -0.927266f, -0.401338f,
                             0.244335f,  -0.414953f};
  std::vector<float> ref_out = {
    0.684172f,  0.279348f,  -0.761074f, -0.073549f, 0.866425f,  -0.544224f,
    0.154785f,  -0.480342f, 0.010582f,  0.098163f,  0.146526f,  -1.030422f,
    0.547092f,  -0.252752f, -0.243571f, -0.162197f, -0.216517f, 0.636452f,
    -0.236000f, 0.120334f,  0.999478f,  -0.793769f, -0.733800f, -0.011226f,
    0.247067f,  -1.151284f, -0.030760f, 0.615511f,  0.205859f,  0.266457f,
    -0.454117f, 0.581280f,  0.181472f,  0.204634f,  -0.646542f, 0.020328f,
    0.556058f,  1.354487f,  -0.087349f, -0.396113f};
  std::vector<uint16_t> output(width);

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(),
                                      output.data(), cos_.data(), sin_.data(),
                                      only_convert_to_fp16);

  std::vector<float> f32_output = convert_vector_f16_as_uint16_to_f32(output);
  for (size_t i = 0; i < f32_output.size(); i++) {
    EXPECT_NEAR(ref_out[i], f32_output[i], 0.0003f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_fp16vcache_fp32_transposed) {
  int row_num = 1;
  int num_cache_head = 2;
  int gqa_size = 2;
  int head_dim = 9;
  int max_iter = 2;
  size_t in_size = max_iter * max_iter * num_cache_head * gqa_size;
  size_t vcache_size = max_iter * num_cache_head * head_dim;
  size_t output_size = num_cache_head * gqa_size * head_dim;

  std::vector<float> in = {0.092400f,  -0.414656f, -0.271581f, -0.300455f,
                           0.317950f,  -0.176271f, 0.138504f,  -0.361754f,
                           -0.024215f, -0.968064f, 0.360932f,  0.384471f};
  std::vector<float> f32_vcache = {
    -0.250920f, 0.593086f,  0.901429f,  -0.633130f, 0.463988f,  0.559382f,
    0.197317f,  0.193700f,  -0.687963f, -0.108334f, -0.688011f, -0.800050f,
    -0.883833f, -0.081502f, 0.732352f,  -0.332583f, 0.202230f,  -0.714266f,
    0.416145f,  0.301777f,  -0.958831f, -0.887177f, 0.939820f,  0.443998f,
    0.664885f,  0.877105f,  -0.575322f, -0.998442f, -0.636350f, 0.984423f,
    -0.633191f, 0.234963f,  -0.391515f, 0.223306f,  0.049513f,  -0.985867f};
  std::vector<uint16_t> vcache =
    convert_vector_f32_to_f16_as_uint16(f32_vcache);
  std::vector<float> ref_out = {
    0.109160f,  0.150761f,  -0.221623f, -0.340605f, 0.341716f,  0.192903f,
    0.229677f,  0.296728f,  -0.246453f, 0.030694f,  -0.299190f, -0.204716f,
    0.418990f,  -0.358029f, -0.310309f, -0.199024f, -0.234911f, 0.386668f,
    -0.108879f, 0.098724f,  0.353685f,  0.152306f,  0.054675f,  -0.253151f,
    0.121229f,  -0.048077f, 0.057462f,  0.393775f,  0.436868f,  -0.115650f,
    0.494638f,  -0.060525f, -0.078396f, 0.019140f,  -0.078680f, 0.571263f};
  std::vector<float> output(output_size);

  nntrainer::compute_fp16vcache_fp32_transposed(
    row_num, in.data(), vcache.data(), output.data(), num_cache_head, gqa_size,
    head_dim);

  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0001f);
  }
}
#endif

#if defined(ENABLE_FP16) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
TEST(nntrainer_cpu_backend_standalone, softmax_row_inplace_fp16) {
  size_t start_row = 0;
  size_t end_row = 3;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  // auto qk_out = generate_random_vector<float>(qk_out_size, -10, 10);
  std::vector<__fp16> qk_out = {
    -2.509198, 5.930860,  9.014286,  -6.331305, 4.639878,  5.593820,
    1.973170,  1.937003,  -6.879627, -1.083345, -6.880110, -8.000502,
    -8.838327, -0.815022, 7.323523,  -3.325828, 2.022300,  -7.142663,
    4.161452,  3.017770,  -9.588310, -8.871769, 9.398197,  4.439975,
    6.648853,  8.771055,  -5.753218, -9.984425, -6.363501, 9.844232};
  std::vector<__fp16> ref_out = {
    0.986697, 0.999999, 0.405184, 0.000021, 0.043301, 0.040031,
    0.487615, 0.999879, 0.000016, 0.000018, 0.012472, 0.000001,
    0.000000, 0.005194, 0.633859, 0.000005, 0.512170, 0.000114,
    0.999957, 0.001083, 0.000831, 0.000000, 0.594816, 0.994785,
    0.322840, 0.959963, 0.000215, 0.000007, 0.000027, 0.998899};

  nntrainer::softmax_row_inplace(qk_out.data(), start_row, end_row, num_heads);

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0005f);
  }
}

TEST(nntrainer_cpu_backend_standalone, softmax_row_inplace_fp16_sink) {
  size_t start_row = 0;
  size_t end_row = 2;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  std::vector<__fp16> qk_out = {-6.880110f, -8.000502f, -8.838327f, -0.815022f,
                                7.323523f,  -3.325828f, 2.022300f,  -7.142663f,
                                4.161452f,  3.017770f,  -9.588310f, -8.871769f,
                                9.398197f,  4.439975f,  6.648853f,  8.771055f,
                                -5.753218f, -9.984425f, -6.363501f, 9.844232f};
  std::vector<__fp16> sink = {-2.509198f, 5.930860f, 9.014286f, -6.331305f,
                              4.639878f,  5.593820f, 1.973170f, 1.937003f,
                              -6.879627f, -1.083345f};
  std::vector<__fp16> ref_out = {
    0.012472f, 0.000001f, 0.000000f, 0.005194f, 0.633859f, 0.000005f, 0.512170f,
    0.000114f, 0.999957f, 0.001083f, 0.000831f, 0.000000f, 0.594816f, 0.994785f,
    0.322840f, 0.959963f, 0.000215f, 0.000007f, 0.000027f, 0.998899f};

  nntrainer::softmax_row_inplace(qk_out.data(), start_row, end_row, num_heads,
                                 sink.data());

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0005f);
  }
}

TEST(nntrainer_cpu_backend_standalone, softmax_row_fp16) {
  size_t start_row = 0;
  size_t end_row = 3;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  // auto qk_out = generate_random_vector<float>(qk_out_size, -10, 10);
  std::vector<__fp16> qk_out = {
    -2.509198, 5.930860,  9.014286,  -6.331305, 4.639878,  5.593820,
    1.973170,  1.937003,  -6.879627, -1.083345, -6.880110, -8.000502,
    -8.838327, -0.815022, 7.323523,  -3.325828, 2.022300,  -7.142663,
    4.161452,  3.017770,  -9.588310, -8.871769, 9.398197,  4.439975,
    6.648853,  8.771055,  -5.753218, -9.984425, -6.363501, 9.844232};
  std::vector<__fp16> ref_out = {
    0.986697, 0.999999, 0.405184, 0.000021, 0.043301, 0.040031,
    0.487615, 0.999879, 0.000016, 0.000018, 0.012472, 0.000001,
    0.000000, 0.005194, 0.633859, 0.000005, 0.512170, 0.000114,
    0.999957, 0.001083, 0.000831, 0.000000, 0.594816, 0.994785,
    0.322840, 0.959963, 0.000215, 0.000007, 0.000027, 0.998899};

  nntrainer::softmax_row(qk_out.data(), start_row, end_row, num_heads);

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0005f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_kcaches_fp16) {
  int num_rows = 1;
  int N = 2;
  int head_dim = 10;
  int group_size = 4;
  int tile_size = 16;
  size_t in_size = N * group_size * head_dim;
  size_t kcache_size = num_rows * N * head_dim;
  size_t output_size = num_rows * N * group_size;

  std::vector<__fp16> in = {
    -2.509198, 5.930860,  9.014286,  -6.331305, 4.639878,  5.593820,  1.973170,
    1.937003,  -6.879627, -1.083345, -6.880110, -8.000502, -8.838327, -0.815022,
    7.323523,  -3.325828, 2.022300,  -7.142663, 4.161452,  3.017770,  -9.588310,
    -8.871769, 9.398197,  4.439975,  6.648853,  8.771055,  -5.753218, -9.984425,
    -6.363501, 9.844232,  -6.331910, 2.349631,  -3.915155, 2.233063,  0.495129,
    -9.858674, -1.361099, -9.538752, -4.175417, 0.495493,  2.237058,  -2.002780,
    -7.210123, -9.066687, -4.157107, 9.475111,  -2.672763, -5.344573, -0.878600,
    -8.187871, 5.703520,  2.367721,  -6.006524, -2.350760, 0.284688,  9.664618,
    1.848291,  -0.664742, -9.070992, 7.198808,  2.150897,  3.606151,  -6.589518,
    -0.990015, -8.698968, -9.734701, 8.977711,  8.844034,  9.312640,  1.265764,
    6.167947,  -2.291670, -3.907725, -9.680675, -8.046557, -5.382123, 3.684660,
    -5.179491, -1.196950, 3.665271};
  std::vector<__fp16> kcache = {
    0.000406, 0.006954, 0.020065, 0.000110, 0.004494, 0.006313, 0.001806,
    0.001789, 0.000093, 0.000663, 0.000093, 0.000060, 0.000035, 0.000727,
    0.011406, 0.000309, 0.001830, 0.000086, 0.003744, 0.002655};
  std::vector<__fp16> ref_out = {0.089252,  -0.072949, 0.058948,  -0.045583,
                                 -0.025812, -0.002068, -0.014971, -0.028027};
  std::vector<__fp16> output(output_size);

  nntrainer::compute_kcaches(in.data(), kcache.data(), output.data(), num_rows,
                             N, head_dim, group_size, tile_size);

  for (size_t i = 0; i < output_size; i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_out_null_fp16) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;

  std::vector<__fp16> inout = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,  0.197317,
    0.193700,  -0.687963, -0.108334, -0.688011, -0.800050, -0.883833, -0.081502,
    0.732352,  -0.332583, 0.202230,  -0.714266, 0.416145,  0.301777,  -0.958831,
    -0.887177, 0.939820,  0.443998,  0.664885,  0.877105,  -0.575322, -0.998442,
    -0.636350, 0.984423,  -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,
    -0.985867, -0.136110, -0.953875, -0.417542, 0.049549};
  std::vector<__fp16> cos_ = {-0.508061, 0.998221,  -0.733869, 0.164825,
                              0.297530,  -0.591437, -0.165907, -0.999895,
                              -0.163179, 0.249787};
  std::vector<__fp16> sin_ = {0.809128,  -0.390829, -0.112627, 0.377992,
                              -0.994569, -0.641599, -0.927266, -0.401338,
                              0.244335,  -0.414953};
  std::vector<__fp16> ref_out = {
    0.684172,  0.279348,  -0.761074, -0.073549, 0.866425,  -0.544224, 0.154785,
    -0.480342, 0.010582,  0.098163,  0.146526,  -1.030422, 0.547092,  -0.252752,
    -0.243571, -0.162197, -0.216517, 0.636452,  -0.236000, 0.120334,  0.999478,
    -0.793769, -0.733800, -0.011226, 0.247067,  -1.151284, -0.030760, 0.615511,
    0.205859,  0.266457,  -0.454117, 0.581280,  0.181472,  0.204634,  -0.646542,
    0.020328,  0.556058,  1.354487,  -0.087349, -0.396113};

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(), nullptr,
                                      cos_.data(), sin_.data());

  for (size_t i = 0; i < inout.size(); i++) {
    EXPECT_NEAR(ref_out[i], inout[i], 0.001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_fp16) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;

  std::vector<__fp16> inout = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,  0.197317,
    0.193700,  -0.687963, -0.108334, -0.688011, -0.800050, -0.883833, -0.081502,
    0.732352,  -0.332583, 0.202230,  -0.714266, 0.416145,  0.301777,  -0.958831,
    -0.887177, 0.939820,  0.443998,  0.664885,  0.877105,  -0.575322, -0.998442,
    -0.636350, 0.984423,  -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,
    -0.985867, -0.136110, -0.953875, -0.417542, 0.049549};
  std::vector<__fp16> cos_ = {-0.508061, 0.998221,  -0.733869, 0.164825,
                              0.297530,  -0.591437, -0.165907, -0.999895,
                              -0.163179, 0.249787};
  std::vector<__fp16> sin_ = {0.809128,  -0.390829, -0.112627, 0.377992,
                              -0.994569, -0.641599, -0.927266, -0.401338,
                              0.244335,  -0.414953};
  std::vector<__fp16> ref_out = {
    0.684172,  0.279348,  -0.761074, -0.073549, 0.866425,  -0.544224, 0.154785,
    -0.480342, 0.010582,  0.098163,  0.146526,  -1.030422, 0.547092,  -0.252752,
    -0.243571, -0.162197, -0.216517, 0.636452,  -0.236000, 0.120334,  0.999478,
    -0.793769, -0.733800, -0.011226, 0.247067,  -1.151284, -0.030760, 0.615511,
    0.205859,  0.266457,  -0.454117, 0.581280,  0.181472,  0.204634,  -0.646542,
    0.020328,  0.556058,  1.354487,  -0.087349, -0.396113};
  std::vector<__fp16> output(width);

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(),
                                      output.data(), cos_.data(), sin_.data());

  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_fp16vcache_transposed_fp16) {
  int row_num = 1;
  int num_cache_head = 2;
  int gqa_size = 2;
  int head_dim = 9;
  int max_iter = 2;
  size_t in_size = max_iter * max_iter * num_cache_head * gqa_size;
  size_t vcache_size = max_iter * num_cache_head * head_dim;
  size_t output_size = num_cache_head * gqa_size * head_dim;

  std::vector<__fp16> in = {0.092400,  -0.414656, -0.271581, -0.300455,
                            0.317950,  -0.176271, 0.138504,  -0.361754,
                            -0.024215, -0.968064, 0.360932,  0.384471};
  std::vector<__fp16> vcache = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,
    0.197317,  0.193700,  -0.687963, -0.108334, -0.688011, -0.800050,
    -0.883833, -0.081502, 0.732352,  -0.332583, 0.202230,  -0.714266,
    0.416145,  0.301777,  -0.958831, -0.887177, 0.939820,  0.443998,
    0.664885,  0.877105,  -0.575322, -0.998442, -0.636350, 0.984423,
    -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,  -0.985867};
  std::vector<__fp16> ref_out = {
    0.109160,  0.150761,  -0.221623, -0.340605, 0.341716,  0.192903,
    0.229677,  0.296728,  -0.246453, 0.030694,  -0.299190, -0.204716,
    0.418990,  -0.358029, -0.310309, -0.199024, -0.234911, 0.386668,
    -0.108879, 0.098724,  0.353685,  0.152306,  0.054675,  -0.253151,
    0.121229,  -0.048077, 0.057462,  0.393775,  0.436868,  -0.115650,
    0.494638,  -0.060525, -0.078396, 0.019140,  -0.078680, 0.571263};
  std::vector<__fp16> output(output_size);

  nntrainer::compute_fp16vcache_transposed(row_num, in.data(), vcache.data(),
                                           output.data(), num_cache_head,
                                           gqa_size, head_dim);

  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0004f);
  }
}
#endif

static void run_clamp_test(const unsigned int N, float lower_bound,
                           float upper_bound, bool print = false) {
  const int TEST_CNT = 20;
  nanoseconds ref_mul_time = (nanoseconds)0;
  nanoseconds mul_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> X = generate_random_vector<float, false>(N, -10, 10);
    std::vector<float> Y = generate_random_vector<float, false>(N);
    std::vector<float> Y_ref = generate_random_vector<float, false>(N);
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::__fallback_clamp(X.data(), Y_ref.data(), N, lower_bound,
                                  upper_bound);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_mul_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::clamp(X.data(), Y.data(), N, lower_bound, upper_bound);
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        mul_time += dt;
      }
    }

    auto mean_squared_error = compute_mse(1, N, Y_ref, Y, print);
    ASSERT_LE(mean_squared_error, 0.00001f);
  }

  if (print) {
    std::cout << "[INFO] clamp: TEST CNT: " << TEST_CNT << ", N: " << N
              << ", Average ref_time: " << ref_mul_time.count() / TEST_CNT
              << " ns, Average mul_time: " << mul_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, clamp_3072_0_1) {
  const unsigned int N = 3072;
  float lower_bound = 0.F;
  float upper_bound = 1.F;
  run_clamp_test(N, lower_bound, upper_bound, false);
}

static void run_transform_int4_test_(const uint32_t K, const uint32_t N,
                                     const int scale_group_size,
                                     bool use_ones = false) {
  nntrainer::init_backend();
  const size_t q4_data_size = K * N / Q4_0 * sizeof(block_q4_0);
  std::vector<float> weight_fp32;
  if (use_ones) {
    float ones_ratio = 0.1f;
    weight_fp32 = generate_01_vector(static_cast<size_t>(N) * K, ones_ratio);
  } else {
    weight_fp32 = generate_random_vector<float, false>(
      static_cast<size_t>(N) * K, -1.0, 1.0);
  }

  bool print = false;
  if (print && use_ones) {
    printMatrixI("weight_fp32", weight_fp32.data(), N, K);
  }

  std::vector<uint8_t> osv32_weights;
  std::vector<uint16_t> osv32_scales;
  nntrainer::Int4Utils::quantizeAndRepack(
    weight_fp32.data(), N, K, scale_group_size, osv32_weights, osv32_scales);

  // MAIN TEST - direct transform Int4 data (osv32_isv2) ---> Q4_0x
  int run_cnt = 100;
  std::vector<uint8_t> dst_q4_0x(q4_data_size);
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < run_cnt; i++) {
    nntrainer::transform_int4_osv32_isv2_to_q4_0(
      N, K, osv32_weights.data(), osv32_scales.data(), scale_group_size,
      dst_q4_0x.data());
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto exec_time =
    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  std::cout << "Time: " << (double)exec_time.count() / (1000 * run_cnt) << " ms"
            << std::endl;

  // Check MSE quantized values
  std::vector<uint8_t> unpacked_weights_q4(q4_data_size);
  nntrainer::unpack_q4_0(dst_q4_0x.data(), unpacked_weights_q4.data(),
                         q4_data_size, N, K);
  std::vector<float> dequantized_weights_q4(static_cast<size_t>(N) * K);
  nntrainer::dequantize_row_q4_0(unpacked_weights_q4.data(),
                                 dequantized_weights_q4.data(),
                                 static_cast<int64_t>(N) * K);
  if (print && use_ones) {
    printMatrixI("dequantized_weights_q4 I", dequantized_weights_q4.data(), N,
                 K);
  }
  float mse_direct_transform_q4 =
    mse<float>(weight_fp32.data(), dequantized_weights_q4.data(), N * K);
  std::cout << "MSE direct transform Q4_0: " << std::setprecision(10)
            << mse_direct_transform_q4 << std::endl;
  // MAIN TEST - END

  // Reference solution - Int4 data (osv32_isv2) --> FP32 --> quantization to
  // Q4_0x8 For checking difference of accuracy
  std::vector<float> dequant_weight_fp32(static_cast<size_t>(N) * K);
  nntrainer::Int4Utils::dequantizePacked(osv32_weights, osv32_scales, N, K,
                                         scale_group_size, dequant_weight_fp32);
  std::vector<uint8_t> tmp_q4_weight(q4_data_size);
  nntrainer::quantize_q4_0(dequant_weight_fp32.data(), tmp_q4_weight.data(), N,
                           K, nullptr);
  std::vector<uint8_t> ref_q4_0x8(q4_data_size);
  nntrainer::repack_q4_0(ref_q4_0x8.data(), tmp_q4_weight.data(), q4_data_size,
                         N, K);
  // Check MSE quantized values
  std::vector<uint8_t> unpacked_ref_weights_q4(q4_data_size);
  nntrainer::unpack_q4_0(ref_q4_0x8.data(), unpacked_ref_weights_q4.data(),
                         q4_data_size, N, K);
  std::vector<float> dequantized_ref_weights_q4(static_cast<size_t>(N) * K);
  nntrainer::dequantize_row_q4_0(unpacked_ref_weights_q4.data(),
                                 dequantized_ref_weights_q4.data(),
                                 static_cast<int64_t>(N) * K);
  float mse_fp32_transform_q4 =
    mse<float>(weight_fp32.data(), dequantized_ref_weights_q4.data(), N * K);
  std::cout << "MSE FP32 transform Q4_0:   " << std::setprecision(10)
            << mse_fp32_transform_q4 << std::endl;
  // Reference solution - END

  float mse_dequant_int4_vs_trans_q4 = mse<float>(
    dequant_weight_fp32.data(), dequantized_weights_q4.data(), N * K);
  std::cout << "MSE dequant Int4 vs direct transform Q4_0: "
            << std::setprecision(10) << mse_dequant_int4_vs_trans_q4
            << std::endl;

  // This is a proof that the transformQ4_0x_FromInt4() is lossless computation.
  const float epsilon = 0.00001f;
  EXPECT_IN_RANGE(mse_dequant_int4_vs_trans_q4, 0, epsilon);

  // Other MSEs
  const float epsilon_direct = (use_ones) ? 0.00001f : 0.002f;
  EXPECT_IN_RANGE(mse_direct_transform_q4, 0, epsilon_direct);
  const float epsilon_fp32 = (use_ones) ? 0.00001f : 0.004f;
  EXPECT_IN_RANGE(mse_fp32_transform_q4, 0, epsilon_fp32);
  if (!use_ones) {
    EXPECT_LE(mse_direct_transform_q4, mse_fp32_transform_q4);
  }

  // Additional test with using GEMMs
  const uint32_t M = 4;
  const uint32_t input_size = M * K;
  const int INT4_BLOCK_N_SIZE = 32;
  uint32_t alignN = nntrainer::align(N, INT4_BLOCK_N_SIZE);
  std::vector<float> input =
    generate_random_vector<float, false>(input_size, -1.0, 1.0);

  // SGEMM - reference
  std::vector<float> ref_output_fp32(M * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, input.data(), K,
                   weight_fp32.data(), K, 0.F, ref_output_fp32.data(), N);

  // GEMM INT4->transform->Q4
  std::vector<float> int4_trans_q4_output_fp32(M * N);
  nntrainer::gemm_q4_0(M, N, K, input.data(), K, dst_q4_0x.data(), N,
                       int4_trans_q4_output_fp32.data(), N);
  float mse_int4_trans_q4 =
    mse<float>(ref_output_fp32.data(), int4_trans_q4_output_fp32.data(), M * N);
  std::cout << "MSE GEMM INT4->transform->Q4:            "
            << std::setprecision(10) << mse_int4_trans_q4 << std::endl;

  // GEMM INT4->dequant->FP32->quant->Q4
  std::vector<float> int4_fp32_q4_output_fp32(M * N);
  nntrainer::gemm_q4_0(M, N, K, input.data(), K, ref_q4_0x8.data(), N,
                       int4_fp32_q4_output_fp32.data(), N);
  float mse_int4_fp32_q4 =
    mse<float>(ref_output_fp32.data(), int4_fp32_q4_output_fp32.data(), M * N);
  std::cout << "MSE GEMM INT4->dequant->FP32->quant->Q4: "
            << std::setprecision(10) << mse_int4_fp32_q4 << std::endl;
}

#define DECLARE_transform_int4_test_K_N(K, N, G)                               \
  TEST(nntrainer_blas_kernel,                                                  \
       _transform_int4_test_K##K##_N##N##_Group##G##_RandomOnes) {             \
    run_transform_int4_test_(K, N, G, true);                                   \
  }                                                                            \
  TEST(nntrainer_blas_kernel,                                                  \
       _transform_int4_test_K##K##_N##N##_Group##G##_RandomFloat) {            \
    run_transform_int4_test_(K, N, G, false);                                  \
  }

DECLARE_transform_int4_test_K_N(128, 40, 32);
DECLARE_transform_int4_test_K_N(128, 40, 64);
DECLARE_transform_int4_test_K_N(256, 40, 128);
DECLARE_transform_int4_test_K_N(32, 40, 32);
DECLARE_transform_int4_test_K_N(32, 48, 32);
DECLARE_transform_int4_test_K_N(64, 40, 32);
DECLARE_transform_int4_test_K_N(320, 640, 32);
DECLARE_transform_int4_test_K_N(1024, 640, 32);
DECLARE_transform_int4_test_K_N(1024, 648, 32);
DECLARE_transform_int4_test_K_N(1024, 648, 64);
DECLARE_transform_int4_test_K_N(1024, 648, 128);
DECLARE_transform_int4_test_K_N(3072, 8192, 32);

#if defined(__x86_64__) || defined(_M_X64)
/**
 * @brief Random plain QS4CX weight: N rows of ceil(K/2) bytes (every nibble
 * value 0..15 occurs, including the bytes' unused high nibble at odd K) plus
 * N per-row scales of mixed magnitude.
 */
static void make_qs4cx_weight(size_t N, size_t K, std::mt19937 &gen,
                              std::vector<uint8_t> &w,
                              std::vector<float> &scale) {
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> sc(1e-3f, 0.2f);
  w.resize(N * ((K + 1) / 2));
  for (auto &b : w)
    b = static_cast<uint8_t>(byte(gen));
  scale.resize(N);
  for (auto &v : scale)
    v = sc(gen);
}

/**
 * @brief The x86 qai8dx x qs4cx fp32 GEMM against the scalar reference
 * (online qa8dx quantization + __fallback_matmul_mxn_mxk_nxk_f32_qa8dx_qs4cx).
 * The dot products are exact integers, so the results must match bit for
 * bit, including odd M / N / K tails.
 */
static void run_qs4cx_f32_vs_reference(size_t M, size_t N, size_t K) {
  std::mt19937 gen(static_cast<unsigned>(M * 7919 + N * 131 + K));
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<float> lhs(M * K);
  for (auto &v : lhs)
    v = dist(gen);
  std::vector<uint8_t> w;
  std::vector<float> scale;
  make_qs4cx_weight(N, K, gen, w, scale);

  std::vector<float> out(M * N, -1.0f), ref(M * N, -2.0f);
  nntrainer::gemm_qai8dxp_qsi4cxp_rhs_unpacked(
    M, N, K, lhs.data(), w.data(), scale.data(), out.data(), 0, true);

  std::vector<int8_t> lhs_q(M * (K + sizeof(int32_t) + sizeof(float)));
  nntrainer::__fallback_quant_qa8dx_f32(M, K, lhs.data(), lhs_q.data());
  nntrainer::__fallback_matmul_mxn_mxk_nxk_f32_qa8dx_qs4cx(
    M, N, K, lhs_q.data(), w.data(), scale.data(), ref.data(), -FLT_MAX,
    FLT_MAX);

  EXPECT_EQ(std::memcmp(out.data(), ref.data(), out.size() * sizeof(float)), 0)
    << "M=" << M << " N=" << N << " K=" << K;
}

TEST(nntrainer_cpu_backend_standalone, qs4cx_f32_gemm_vs_reference) {
  run_qs4cx_f32_vs_reference(1, 1, 1);
  run_qs4cx_f32_vs_reference(1, 37, 129);
  run_qs4cx_f32_vs_reference(3, 70, 64);
  run_qs4cx_f32_vs_reference(5, 33, 31);
  run_qs4cx_f32_vs_reference(17, 96, 257);
  run_qs4cx_f32_vs_reference(2, 300, 1536);
}

#ifdef ENABLE_FP16
/**
 * @brief The x86 QS4CX x fp16-activation GEMM against the scalar reference
 * __fallback_gemm_qs4cx_fp16(). The kernel keeps the reference's ascending-k
 * fp32 summation per output, so the fp16 results must match bit for bit,
 * with and without beta, across odd M / N / K tails.
 */
static void run_qs4cx_fp16_vs_reference(size_t M, size_t N, size_t K,
                                        float beta) {
  std::mt19937 gen(static_cast<unsigned>(M * 7919 + N * 131 + K + 17));
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  std::vector<_FP16> lhs(M * K);
  for (auto &v : lhs)
    v = static_cast<_FP16>(dist(gen));
  std::vector<uint8_t> w;
  std::vector<float> scale;
  make_qs4cx_weight(N, K, gen, w, scale);

  std::vector<_FP16> out(M * N), ref(M * N);
  for (size_t i = 0; i < M * N; ++i)
    out[i] = ref[i] = static_cast<_FP16>(dist(gen));

  nntrainer::gemm_qs4cx_fp16(M, N, K, lhs.data(), w.data(), scale.data(),
                             out.data(), beta);
  nntrainer::__fallback_gemm_qs4cx_fp16(M, N, K, lhs.data(), w.data(),
                                        scale.data(), ref.data(), beta);

  EXPECT_EQ(std::memcmp(out.data(), ref.data(), out.size() * sizeof(_FP16)), 0)
    << "M=" << M << " N=" << N << " K=" << K << " beta=" << beta;
}

TEST(nntrainer_cpu_backend_standalone, qs4cx_fp16_gemm_vs_reference) {
  run_qs4cx_fp16_vs_reference(1, 1, 1, 0.0f);
  run_qs4cx_fp16_vs_reference(1, 37, 129, 0.0f);
  run_qs4cx_fp16_vs_reference(3, 70, 64, 0.0f);
  run_qs4cx_fp16_vs_reference(5, 33, 63, 1.0f);
  run_qs4cx_fp16_vs_reference(9, 8, 65, 0.5f);
  run_qs4cx_fp16_vs_reference(17, 96, 257, 0.0f);
  run_qs4cx_fp16_vs_reference(1, 1000, 1536, 0.0f);
  run_qs4cx_fp16_vs_reference(300, 72, 200, 0.0f);
}
#endif
#endif

/* ---- [#164] the HTP RMSNorm spec equals the shipped CPU RMSNorm --------- */

#if defined(__aarch64__)
/**
 * @brief The Android CPU's decode norm, call for call: the exported
 *        rms_norm_wrt_width_fp32_intrinsic over @a rows rows of @a W, then
 *        Tensor::multiply_i with a 1 x W gamma (RMSNormLayer at W = 2048,
 *        QKVLayer's headNorm at W = 64).
 */
static void cpu_rmsnorm_rows(const float *x, const float *g, float *y,
                             unsigned rows, unsigned W, float eps) {
  nntrainer::Tensor out(1, 1, rows, W);
  nntrainer::Tensor gamma(1, 1, 1, W);
  std::memcpy(gamma.getData<float>(), g, W * sizeof(float));
  nntrainer::rms_norm_wrt_width_fp32_intrinsic(x, out.getData<float>(), rows, W,
                                               eps);
  out.multiply_i(gamma);
  std::memcpy(y, out.getData<float>(), (size_t)rows * W * sizeof(float));
}

/** @brief Elements of @a n rows that differ between the CPU and
 *         m1_rmsnorm_det (chunk = W, one gamma). */
static unsigned rmsnorm_cpu_vs_spec(const float *x, const float *g,
                                    unsigned rows, unsigned W, float eps) {
  std::vector<float> yc((size_t)rows * W), ys((size_t)rows * W);
  cpu_rmsnorm_rows(x, g, yc.data(), rows, W, eps);
  m1_rmsnorm_det(x, g, ys.data(), rows * W, W, eps, nullptr);
  unsigned bad = 0;
  for (size_t i = 0; i < yc.size(); ++i)
    bad += std::memcmp(&yc[i], &ys[i], sizeof(float)) != 0;
  return bad;
}
#endif

/**
 * @brief G0 of plan 164: m1_ops_det.h's RMSNorm (the HTP kernel's spec) is
 *        bit-identical to the CPU function in this libnntrainer.so, at
 *        W = 2048 (1 row a call) and W = 64 (40 rows a call), on 20 000
 *        random rows whose magnitudes span 2^-20 .. 2^20 and the +-1e-39
 *        row. The spec is the aarch64 order; the host's AVX2 order is not.
 */
TEST(RmsNormCpuOrder, MatchesNeon) {
#if !defined(__aarch64__)
  GTEST_SKIP() << "the spec is the aarch64 NEON order";
#else
  const float eps = 1e-5f;
  std::mt19937 rng(0x164u);
  std::uniform_real_distribution<float> mant(0.5f, 1.0f), gd(0.5f, 1.5f);
  std::vector<float> g(2048), x(2048 * 40);
  for (auto &v : g)
    v = gd(rng);
  unsigned rows = 0, bad = 0;
  const unsigned shapes[2][2] = {{2048u, 1u}, {64u, 40u}};
  for (const auto &sh : shapes) {
    const unsigned W = sh[0], R = sh[1];
    /* 1 000 calls of each shape: 1 000 + 40 000 rows */
    for (unsigned call = 0; call < 1000u; ++call, rows += R) {
      for (unsigned r = 0; r < R; ++r) {
        const int e = (int)((call * R + r) % 41u) - 20;
        for (unsigned i = 0; i < W; ++i) {
          const float m = mant(rng);
          x[r * W + i] = std::ldexp((rng() & 1u) ? -m : m, e);
        }
      }
      bad += rmsnorm_cpu_vs_spec(x.data(), g.data(), R, W, eps);
    }
    for (unsigned i = 0; i < W * R; ++i)
      x[i] = (i & 1u) ? -1e-39f : 1e-39f;
    const unsigned bad_sub = rmsnorm_cpu_vs_spec(x.data(), g.data(), R, W, eps);
    std::printf("RmsNormCpuOrder W=%u rows/call=%u subnormal(+-1e-39) bad=%u\n",
                W, R, bad_sub);
    bad += bad_sub;
    rows += R;
  }
  std::printf("RmsNormCpuOrder rows=%u bad=%u\n", rows, bad);
  EXPECT_EQ(bad, 0u);
#endif
}

/**
 * @brief G0's replay: NNTR_NORM_REPLAY=<norm.bin> (a dev/norm-shadow dump)
 *        with gamma_rms.f32 (49 x 2048) and gamma_qk.f32 (6 x (q 64 | k 64))
 *        in the same directory; every record's input through the CPU and
 *        the spec, bit for bit. Skipped when unset.
 */
TEST(RmsNormCpuOrder, ReplayNormShadow) {
#if !defined(__aarch64__)
  GTEST_SKIP() << "the spec is the aarch64 NEON order";
#else
  const char *path = std::getenv("NNTR_NORM_REPLAY");
  if (!path)
    GTEST_SKIP() << "NNTR_NORM_REPLAY not set";
  const std::string file(path);
  const std::string dir = file.substr(0, file.find_last_of('/') + 1);
  auto load = [](const std::string &p, size_t n) {
    std::vector<float> v(n);
    std::FILE *f = std::fopen(p.c_str(), "rb");
    const size_t got = f ? std::fread(v.data(), sizeof(float), n, f) : 0;
    if (f)
      std::fclose(f);
    return got == n ? v : std::vector<float>();
  };
  const std::vector<float> grms = load(dir + "gamma_rms.f32", 49u * 2048u);
  const std::vector<float> gqk = load(dir + "gamma_qk.f32", 6u * 128u);
  ASSERT_FALSE(grms.empty() || gqk.empty()) << "gammas next to " << file;
  std::FILE *f = std::fopen(path, "rb");
  ASSERT_NE(f, nullptr) << file;
  const float eps = 1e-5f;
  unsigned h[4], k_rms = 0, k_qk = 0, n_rms = 0, bad_rms = 0, n_qk = 0,
                 bad_qk = 0;
  std::vector<float> in(8192), cpu(8192), other(8192);
  while (std::fread(h, sizeof(unsigned), 4, f) == 4) {
    ASSERT_TRUE(h[2] <= 8192u && h[3] <= 8192u && h[3] <= h[2] &&
                std::fread(in.data(), 4, h[2], f) == h[2] &&
                std::fread(cpu.data(), 4, h[3], f) == h[3] &&
                std::fread(other.data(), 4, h[3], f) == h[3]);
    if (h[0] != 1u) {
      ASSERT_EQ(h[2], 2048u);
      ++n_rms;
      bad_rms +=
        rmsnorm_cpu_vs_spec(in.data(), grms.data() + (k_rms++ % 49u) * 2048u, 1,
                            2048u, eps) != 0;
    } else {
      const unsigned wk = h[2] - h[3], wq = h[3] - wk;
      ASSERT_TRUE(wk > 0u && wk <= h[3] - wk && wq % 64u == 0u &&
                  wk % 64u == 0u)
        << "QK_NORM record of " << h[2] << " / " << h[3];
      const float *gq = gqk.data() + (k_qk++ % 6u) * 128u;
      n_qk += h[3] / 64u;
      bad_qk += rmsnorm_cpu_vs_spec(in.data(), gq, wq / 64u, 64u, eps) != 0;
      bad_qk +=
        rmsnorm_cpu_vs_spec(in.data() + wq, gq + 64, wk / 64u, 64u, eps) != 0;
    }
  }
  std::fclose(f);
  std::printf("RmsNormCpuOrder replay %s rms=%u bad=%u qk_heads=%u "
              "bad_calls=%u\n",
              path, n_rms, bad_rms, n_qk, bad_qk);
  EXPECT_EQ(bad_rms + bad_qk, 0u);
  EXPECT_GT(n_rms + n_qk, 0u);
#endif
}

/* ---- [#132 PR 2] G0: the CPU-order specs equal the shipped CPU ---------- */

#if defined(__aarch64__)
/** @brief Rows per G0 shape: NNTR_G0_ROWS, default 2000 (plan 132 G0). */
static unsigned g0_rows(unsigned dflt = 2000u) {
  const char *e = std::getenv("NNTR_G0_ROWS");
  return e ? static_cast<unsigned>(std::atoi(e)) : dflt;
}
#endif

/**
 * @brief nntr_quantize_row_q8_0 (this libnntrainer.so) against
 *        q8_0_quant_cpu_det, block for block: the stored f16 d and the 32
 *        quants, on q4_gemv_cases.h's rows (block magnitudes 2^-20 ..
 *        2^20, zero, tiny and quantizer-rounding blocks).
 */
TEST(Q8QuantCpuOrder, MatchesNeon) {
#if !defined(__aarch64__)
  GTEST_SKIP() << "the spec is the aarch64 order";
#else
  const unsigned rows = g0_rows();
  unsigned bad = 0, n = 0;
  for (unsigned K : {2048u, 7168u}) {
    std::vector<float> x(K);
    std::vector<uint8_t> qb((K / 32u) * 34u);
    std::vector<int8_t> q(K);
    std::vector<uint16_t> d(K / 32u);
    for (unsigned r = 0; r < rows; ++r, ++n) {
      make_row(x.data(), K, static_cast<int>(r % 4u), static_cast<int>(r));
      nntr_quantize_row_q8_0(x.data(), qb.data(), K);
      q8_0_quant_cpu_det(x.data(), K, q.data(), d.data());
      for (unsigned b = 0; b < K / 32u; ++b) {
        const uint8_t *blk = qb.data() + b * 34u;
        bad += std::memcmp(blk, &d[b], 2) != 0 ||
               std::memcmp(blk + 2, q.data() + b * 32u, 32) != 0;
      }
    }
  }
  std::printf("Q8QuantCpuOrder rows=%u bad_blocks=%u\n", n, bad);
  EXPECT_EQ(bad, 0u);
#endif
}

/**
 * @brief gemm_q4_0<float> at M = 1 (the decode FC and lm_head) against
 *        q4_gemv_cpu_det on the same canonical weight, repacked by this
 *        library's repack_q4_0: the five FC shapes, NNTR_G0_ROWS rows each.
 *        Also q4_0_from_q4_0x4 inverts the repack (the shadow's reorder).
 */
TEST(Q4GemvCpuOrder, MatchesNeon) {
#if !defined(__aarch64__)
  GTEST_SKIP() << "the spec is the aarch64 order";
#else
  const unsigned rows = g0_rows();
  const unsigned shapes[5][2] = {{2048u, 6144u},
                                 {2048u, 2048u},
                                 {2048u, 512u},
                                 {7168u, 2048u},
                                 {2048u, 7168u}};
  unsigned bad_total = 0;
  for (const auto &sh : shapes) {
    const unsigned K = sh[0], N = sh[1], nb = K / 32u;
    const size_t bytes = static_cast<size_t>(N) * nb * 18u;
    std::vector<uint8_t> w(bytes), x4(bytes), back(bytes);
    make_weights(w.data(), K, N);
    std::memcpy(x4.data(), w.data(), bytes);
    nntrainer::repack_q4_0(x4.data(), w.data(), bytes, N, K);
    q4_0_from_q4_0x4(x4.data(), K, N, back.data());
    const bool layout_ok = std::memcmp(back.data(), w.data(), bytes) == 0;
    std::vector<float> x(K), yc(N), ys(N);
    std::vector<int8_t> q(K);
    std::vector<uint16_t> d(nb);
    unsigned bad = 0;
    for (unsigned r = 0; r < rows; ++r) {
      make_row(x.data(), K, static_cast<int>(r % 4u), static_cast<int>(r));
      nntrainer::gemm_q4_0<float>(1, N, K, x.data(), K, x4.data(), N, yc.data(),
                                  N);
      q8_0_quant_cpu_det(x.data(), K, q.data(), d.data());
      q4_gemv_cpu_det(w.data(), q.data(), d.data(), K, N, ys.data());
      for (unsigned n = 0; n < N; ++n)
        bad += std::memcmp(&yc[n], &ys[n], 4) != 0;
    }
    std::printf("Q4GemvCpuOrder K=%u N=%u rows=%u bad=%u layout=%s\n", K, N,
                rows, bad, layout_ok ? "ok" : "BAD");
    EXPECT_TRUE(layout_ok) << "q4_0_from_q4_0x4 does not invert repack_q4_0";
    bad_total += bad;
  }
  EXPECT_EQ(bad_total, 0u);
#endif
}

/**
 * @brief nntrainer::swiglu (neon::swiglu, exp_ps) against
 *        m1_swiglu_cpu_det at n = 7168 (the dense FFN), gate values over
 *        the FFN range, the clamp, far tails, subnormals and every floor
 *        boundary of exp_ps's fx.
 */
TEST(SwigluCpuOrder, MatchesNeon) {
#if !defined(__aarch64__)
  GTEST_SKIP() << "the spec is the aarch64 order";
#else
  const unsigned N = 7168u, rows = g0_rows();
  std::mt19937 rng(0x132u);
  std::vector<float> y(N), z(N), oc(N), os(N);
  unsigned bad = 0;
  for (unsigned r = 0; r < rows; ++r) {
    const float span = r % 3u == 0u ? 8.0f : r % 3u == 1u ? 100.0f : 1e4f;
    std::uniform_real_distribution<float> yd(-span, span), zd(-4.0f, 4.0f);
    for (unsigned i = 0; i < N; ++i) {
      y[i] = yd(rng);
      z[i] = zd(rng);
    }
    if (r == 1u)
      for (unsigned i = 0; i < N; ++i)
        y[i] = (i & 1u) ? -1e-39f : 1e-39f;
    if (r == 2u)
      for (unsigned i = 0; i < N; ++i) {
        const int k = static_cast<int>(i / 56u) - 127;
        const float x0 = static_cast<float>((static_cast<double>(k) - 0.5) /
                                            1.4426950408889634);
        y[i] = -cpu_det_float(cpu_det_bits(x0) + (i % 56u) - 28u);
      }
    nntrainer::swiglu(N, oc.data(), y.data(), z.data());
    m1_swiglu_cpu_det(y.data(), z.data(), os.data(), N);
    for (unsigned i = 0; i < N; ++i)
      bad += std::memcmp(&oc[i], &os[i], 4) != 0;
  }
  std::printf("SwigluCpuOrder n=%u rows=%u bad=%u\n", N, rows, bad);
  EXPECT_EQ(bad, 0u);
#endif
}

/**
 * @brief The router's logits: Tensor::dot of a 1 x 2048 row with the
 *        2048 x 32 gate (Lfm2MoELayer's input.dot(gate_weights), OpenBLAS
 *        sgemv_n) against m1_router_cpu_det's fused chains, magnitudes
 *        2^-20 .. 2^20; and the selection of the same logits against a
 *        copy of buildExpertAssignments' code (bionic expf, divides).
 */
TEST(SgemvNCpuOrder, MatchesRouter) {
#if !defined(__aarch64__) || defined(_WIN32)
  // The spec is the order of the Android aarch64 OpenBLAS sgemv_n. The
  // Windows ARM64 build links another OpenBLAS build (nntrainer-windows-
  // resource) that sums in another order: on that CI every logit differed
  // in its bits while every selection matched.
  GTEST_SKIP() << "the spec is the aarch64 order of the Android OpenBLAS";
#else
  const unsigned K = 2048u, E = 32u, top_k = 4u, rows = g0_rows();
  std::mt19937 rng(0x13232u);
  std::uniform_real_distribution<float> wd(-0.05f, 0.05f), bd(-0.01f, 0.01f),
    md(-2.0f, 2.0f);
  nntrainer::Tensor in(1, 1, 1, K), gate(1, 1, K, E), out(1, 1, 1, E);
  float *wp = gate.getData<float>();
  for (unsigned i = 0; i < K * E; ++i)
    wp[i] = wd(rng);
  std::vector<float> bias(E), lg(E);
  for (auto &b : bias)
    b = bd(rng);
  std::vector<uint32_t> sel(top_k);
  std::vector<float> wt(top_k);
  unsigned bad_logit = 0, bad_sel = 0;
  for (unsigned r = 0; r < rows; ++r) {
    float *x = in.getData<float>();
    for (unsigned k = 0; k < K; ++k)
      x[k] = std::ldexp(md(rng), static_cast<int>(r % 41u) - 20);
    in.dot(gate, out);
    m1_router_cpu_det(x, wp, bias.data(), K, E, top_k, lg.data(), sel.data(),
                      wt.data());
    const float *lc = out.getData<float>();
    bad_logit += std::memcmp(lc, lg.data(), E * 4) != 0;
    // buildExpertAssignments on the CPU's own logits
    std::vector<std::pair<float, int>> sc(E);
    for (unsigned e = 0; e < E; ++e) {
      const float s = 1.0f / (1.0f + std::exp(-lc[e]));
      sc[e] = {s + bias[e], static_cast<int>(e)};
    }
    std::partial_sort(
      sc.begin(), sc.begin() + top_k, sc.end(),
      [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
      });
    for (unsigned k = 0; k < top_k; ++k)
      bad_sel += sel[k] != static_cast<uint32_t>(sc[k].second);
  }
  std::printf("SgemvNCpuOrder K=%u E=%u rows=%u bad_logits=%u bad_sel=%u "
              "(the weights' -ffast-math sum lives in libcausallm_core.so: "
              "G2's router records hold it)\n",
              K, E, rows, bad_logit, bad_sel);
  EXPECT_EQ(bad_logit, 0u);
  EXPECT_EQ(bad_sel, 0u);
#endif
}

/**
 * @brief m1_expf_bionic_det against this device's libm expf on all 2^32
 *        inputs (NaNs skipped), eight threads. The first 16 differences are
 *        printed: the plan's stop rule reads them.
 */
TEST(ExpfBionic, AllInputs) {
#if !defined(__aarch64__) || defined(_WIN32)
  // Windows links the UCRT, not bionic, and its ARM64 runner does not get
  // through the 2^32 inputs in the test's 90 s.
  GTEST_SKIP() << "pins the port to bionic (Android's libm)";
#else
  float (*volatile libm_expf)(float) = expf;
  const unsigned T = 8u;
  std::vector<uint64_t> bad(T, 0), n(T, 0);
  std::vector<std::vector<uint32_t>> first(T);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < T; ++t) {
    th.emplace_back([&, t] {
      const uint64_t lo = (1ull << 32) / T * t, hi = (1ull << 32) / T * (t + 1);
      for (uint64_t u = lo; u < hi; ++u) {
        const float x = cpu_det_float(static_cast<uint32_t>(u));
        if (x != x)
          continue;
        const float a = m1_expf_bionic_det(x), b = libm_expf(x);
        ++n[t];
        if (std::memcmp(&a, &b, 4) != 0) {
          ++bad[t];
          if (first[t].size() < 16u)
            first[t].push_back(static_cast<uint32_t>(u));
        }
      }
    });
  }
  for (auto &t : th)
    t.join();
  uint64_t nb = 0, nn = 0;
  for (unsigned t = 0; t < T; ++t) {
    nb += bad[t];
    nn += n[t];
    for (uint32_t u : first[t]) {
      if (u == 0u)
        continue;
      const float x = cpu_det_float(u);
      std::printf("ExpfBionic diff x=%08x port=%08x libm=%08x\n", u,
                  cpu_det_bits(m1_expf_bionic_det(x)),
                  cpu_det_bits(libm_expf(x)));
    }
  }
  std::printf("ExpfBionic inputs=%llu bad=%llu\n",
              static_cast<unsigned long long>(nn),
              static_cast<unsigned long long>(nb));
  EXPECT_EQ(nb, 0u);
#endif
}

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
