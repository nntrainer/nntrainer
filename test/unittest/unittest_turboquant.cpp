// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_turboquant.cpp
 * @date   28 March 2026
 * @brief  Unit tests for TurboQuant (norm + rotation + Lloyd-Max codebook)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <fallback_internal.h>
#include <gtest/gtest.h>
#include <turboquant_utils.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

/***************************************************
 * Test Hadamard Transform-related utils
 ***************************************************/

/**
 * @brief Test hadamard_transform is orthogonal: applying twice returns
 * original. i.e., Test H(Hx)) == x
 */
TEST(turboquant, hadamard_roundtrip) {
  constexpr int n = 64;
  std::vector<float> x(n), orig(n);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < n; ++i)
    x[i] = orig[i] = dist(gen);

  nntrainer::hadamard_transform(x.data(), n);
  nntrainer::hadamard_transform(x.data(), n);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], orig[i], 1e-5f) << "Roundtrip failed at index " << i;
  }
}

/**
 * @brief Test apply_rotation / apply_inverse_rotation roundtrip
 *        i.e., Test R^-1(R(x)) == x
 */
TEST(turboquant, rotation_roundtrip) {
  constexpr int n = 128;
  std::vector<float> signs(n), input(n), rotated(n);

  nntrainer::generate_random_signs(signs.data(), n, 0xDEADBEEF);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (int i = 0; i < n; ++i)
    input[i] = dist(gen);

  nntrainer::apply_rotation(input.data(), rotated.data(), signs.data(), n);
  nntrainer::apply_inverse_rotation(rotated.data(), signs.data(), n);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(rotated[i], input[i], 1e-5f)
      << "Rotation roundtrip failed at " << i;
  }
}

/**
 * @brief Test rotation preserves L2 norm (orthogonal transform).
 *        i.e., Test || R(x) ||^2 == ||x||^2
 */
TEST(turboquant, rotation_preserves_norm) {
  constexpr int n = 64;
  std::vector<float> signs(n), input(n), rotated(n);

  nntrainer::generate_random_signs(signs.data(), n, 0x12345678);

  std::mt19937 gen(99);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  float norm_sq_in = 0.0f;
  for (int i = 0; i < n; ++i) {
    input[i] = dist(gen);
    norm_sq_in += input[i] * input[i];
  }

  nntrainer::apply_rotation(input.data(), rotated.data(), signs.data(), n);

  float norm_sq_out = 0.0f;
  for (int i = 0; i < n; ++i)
    norm_sq_out += rotated[i] * rotated[i];

  EXPECT_NEAR(norm_sq_out, norm_sq_in, 1e-3f) << "Rotation changed the norm";
}

/***************************************************
 * Test Lloyd-Max Codebook-related utils
 ***************************************************/

/**
 * @brief Test Lloyd-Max codebook symmetry and monotonicity.
 *        Test CODEBOOK_D128
 */
TEST(turboquant, codebook_properties) {
  const auto &cb = nntrainer::CODEBOOK_D128;

  // Test c[i] == -c[15-i]
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(cb.centroids[i], -cb.centroids[15 - i], 1e-7f)
      << "Centroids not symmetric at " << i;
  }

  // Test monotonically increasing
  for (int i = 0; i < 15; ++i) {
    EXPECT_LT(cb.centroids[i], cb.centroids[i + 1])
      << "Centroids not monotonic at " << i;
  }

  for (int i = 0; i < 14; ++i) {
    EXPECT_LT(cb.boundaries[i], cb.boundaries[i + 1])
      << "Boundaries not monotonic at " << i;
  }

  // 0 at the end of the boundary
  EXPECT_FLOAT_EQ(cb.boundaries[7], 0.0f);
}

/**
 * @brief Test lloydmax_quantize for known values.
 *        Test quantization for the known value:
 *          quantize(0) \in {7, 8}
 */
TEST(turboquant, lloydmax_quantize_known) {
  const auto &cb = nntrainer::CODEBOOK_D128;

  uint8_t q0 = nntrainer::lloydmax_quantize(0.0f, cb);
  EXPECT_TRUE(q0 == 7 || q0 == 8)
    << "Zero mapped to unexpected bin " << (int)q0;

  uint8_t q_pos = nntrainer::lloydmax_quantize(1.0f, cb);
  EXPECT_EQ(q_pos, 15);

  uint8_t q_neg = nntrainer::lloydmax_quantize(-1.0f, cb);
  EXPECT_EQ(q_neg, 0);
}

/***************************************************
 * Qunatize/Dequantize
 ***************************************************/

/**
 * @brief Test turboquant_quantize_head -> turboquant_dequantize_head roundtrip.
 */
TEST(turboquant, quantize_dequantize_roundtrip) {
  constexpr int head_dim = 128;
  std::vector<float> signs(head_dim);
  nntrainer::generate_random_signs(signs.data(), head_dim, 0xCAFE);

  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> input(head_dim), output(head_dim);
  for (auto &v : input)
    v = dist(gen);

  std::vector<uint8_t> packed(head_dim / 2);
  float norm;
  const auto &cb = nntrainer::get_codebook(head_dim);

  nntrainer::turboquant_quantize_head(input.data(), head_dim, packed.data(),
                                      &norm, signs.data(), cb);
  nntrainer::turboquant_dequantize_head(packed.data(), norm, head_dim,
                                        output.data(), signs.data(), cb);

  float input_norm = 0.0f;
  for (auto v : input)
    input_norm += v * v;
  input_norm = std::sqrt(input_norm);
  EXPECT_NEAR(norm, input_norm, 1e-6f);

  float max_diff = 0.0f, sum_sq = 0.0f;
  for (int i = 0; i < head_dim; ++i) {
    float d = std::fabs(input[i] - output[i]);
    max_diff = std::max(max_diff, d);
    sum_sq += d * d;
  }
  float rmse = std::sqrt(sum_sq / head_dim);

  EXPECT_LT(rmse, 0.15f) << "RMSE too large for 4-bit quantization";
  EXPECT_LT(max_diff, 0.5f) << "Max diff too large";
}

/**
 * @brief Test quantize_kv_turboquant multi-head pipeline.
 */
TEST(turboquant, quantize_multihead) {
  constexpr int head_dim = 64;
  constexpr int num_heads = 4;
  constexpr int total = num_heads * head_dim;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> input(total);
  for (auto &v : input)
    v = dist(gen);

  std::vector<uint8_t> packed(total / 2);
  std::vector<float> norms(num_heads);
  std::vector<float> signs(head_dim);
  nntrainer::generate_random_signs(signs.data(), head_dim, 0x5EED);

  nntrainer::quantize_kv_turboquant(input.data(), packed.data(), norms.data(),
                                    signs.data(), head_dim, num_heads);

  const auto &cb = nntrainer::get_codebook(head_dim);
  for (int h = 0; h < num_heads; ++h) {
    float expected_norm = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      float v = input[h * head_dim + i];
      expected_norm += v * v;
    }
    expected_norm = std::sqrt(expected_norm);
    EXPECT_NEAR(norms[h], expected_norm, 1e-5f) << "Head " << h;
  }

  for (int h = 0; h < num_heads; ++h) {
    std::vector<float> output(head_dim);
    nntrainer::turboquant_dequantize_head(packed.data() + h * head_dim / 2,
                                          norms[h], head_dim, output.data(),
                                          signs.data(), cb);
    float max_diff = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      float d = std::fabs(input[h * head_dim + i] - output[i]);
      max_diff = std::max(max_diff, d);
    }
    EXPECT_LT(max_diff, 0.5f) << "Head " << h << " reconstruction too coarse";
  }
}

/**
 * @brief Test compute_kcaches_packed4 against FP32 reference.
 */
TEST(turboquant, kcaches_vs_fp32) {
  constexpr int num_heads_Q = 8;
  constexpr int num_heads_KV = 2;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int head_dim = 64;
  constexpr int num_rows = 16;

  std::mt19937 gen(55);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> query(num_heads_Q * head_dim);
  std::vector<float> keys(num_rows * num_heads_KV * head_dim);
  for (auto &v : query)
    v = dist(gen);
  for (auto &v : keys)
    v = dist(gen);

  // FP32 reference: Q * K^T / sqrt(d)
  std::vector<float> ref_scores(num_rows * num_heads_Q, 0.0f);
  float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
  for (int n = 0; n < num_heads_KV; ++n) {
    for (int r = 0; r < num_rows; ++r) {
      for (int g = 0; g < gqa_size; ++g) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dot += query[(n * gqa_size + g) * head_dim + d] *
                 keys[r * num_heads_KV * head_dim + n * head_dim + d];
        ref_scores[r * num_heads_Q + n * gqa_size + g] = dot * inv_sqrt_d;
      }
    }
  }

  // TurboQuant
  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xBEEF);

  int packed_row_bytes = num_heads_KV * head_dim / 2;
  std::vector<uint8_t> pk(num_rows * packed_row_bytes);
  std::vector<float> knorms(num_rows * num_heads_KV);

  for (int r = 0; r < num_rows; ++r) {
    nntrainer::quantize_kv_turboquant(keys.data() + r * num_heads_KV * head_dim,
                                      pk.data() + r * packed_row_bytes,
                                      knorms.data() + r * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
  }

  std::vector<float> tq_scores(num_rows * num_heads_Q, 0.0f);
  nntrainer::compute_kcaches_packed4(query.data(), pk.data(), knorms.data(),
                                     tq_scores.data(), num_rows, num_heads_KV,
                                     head_dim, gqa_size, 4, rot_signs.data());

  float max_diff = 0.0f;
  for (size_t i = 0; i < ref_scores.size(); ++i) {
    float d = std::fabs(ref_scores[i] - tq_scores[i]);
    max_diff = std::max(max_diff, d);
  }

  EXPECT_LT(max_diff, 0.5f) << "compute_kcaches_packed4 error too large";
}

/**
 * @brief Test compute_vcache_packed4 against FP32 reference.
 */
TEST(turboquant, vcache_vs_fp32) {
  constexpr int num_heads_Q = 4;
  constexpr int num_heads_KV = 2;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int head_dim = 64;
  constexpr int num_rows = 8;
  constexpr int row_num = num_rows - 1;

  std::mt19937 gen(77);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> values(num_rows * num_heads_KV * head_dim);
  for (auto &v : values)
    v = dist(gen);

  std::vector<float> attn(num_rows * num_heads_Q);
  for (auto &v : attn)
    v = std::fabs(dist(gen));
  for (int h = 0; h < num_heads_Q; ++h) {
    float sum = 0.0f;
    for (int r = 0; r < num_rows; ++r)
      sum += attn[r * num_heads_Q + h];
    for (int r = 0; r < num_rows; ++r)
      attn[r * num_heads_Q + h] /= sum;
  }

  // FP32 reference
  std::vector<float> ref_out(num_heads_Q * head_dim, 0.0f);
  for (int n = 0; n < num_heads_KV; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      int qh = n * gqa_size + g;
      for (int r = 0; r < num_rows; ++r) {
        float w = attn[r * num_heads_Q + qh];
        for (int d = 0; d < head_dim; ++d)
          ref_out[qh * head_dim + d] +=
            w * values[r * num_heads_KV * head_dim + n * head_dim + d];
      }
    }
  }

  // TurboQuant v2 path
  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xFACE);

  int packed_row_bytes = num_heads_KV * head_dim / 2;
  std::vector<uint8_t> pv(num_rows * packed_row_bytes);
  std::vector<float> vnorms(num_rows * num_heads_KV);

  for (int r = 0; r < num_rows; ++r) {
    nntrainer::quantize_kv_turboquant(
      values.data() + r * num_heads_KV * head_dim,
      pv.data() + r * packed_row_bytes, vnorms.data() + r * num_heads_KV,
      rot_signs.data(), head_dim, num_heads_KV);
  }

  std::vector<float> tq_out(num_heads_Q * head_dim, 0.0f);
  nntrainer::compute_vcache_packed4(row_num, attn.data(), pv.data(),
                                    vnorms.data(), tq_out.data(), num_heads_KV,
                                    gqa_size, head_dim, rot_signs.data());

  float max_diff = 0.0f;
  for (size_t i = 0; i < ref_out.size(); ++i) {
    float d = std::fabs(ref_out[i] - tq_out[i]);
    max_diff = std::max(max_diff, d);
  }

  EXPECT_LT(max_diff, 0.5f) << "compute_vcache_packed4 error too large";
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return result;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}
