// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_turboquant_mha_integration.cpp
 * @date   28 March 2026
 * @brief  Integration test: MHACoreLayer with use_turboquant=true
 *         Instantiates the layer, runs incremental_forwarding, verifies no
 * crash.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <turboquant_utils.h>

#ifdef USE_BLAS
extern "C" void openblas_set_num_threads(int num_threads);
extern "C" int openblas_get_num_threads(void);
#endif

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

static void fp32_reference_attention(
  const float *query,  // [num_heads_Q * head_dim]
  const float *keys,   // [num_rows * num_heads_KV * head_dim]
  const float *values, // [num_rows * num_heads_KV * head_dim]
  float *output,       // [num_heads_Q * head_dim]
  int num_rows, int num_heads_Q, int num_heads_KV, int head_dim) {

  int gqa_size = num_heads_Q / num_heads_KV;
  float scale = 1.0f / std::sqrt((float)head_dim);

  for (int n = 0; n < num_heads_KV; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      int qh = n * gqa_size + g; // query head index
      const float *q = query + qh * head_dim;

      // Q*K^T
      std::vector<float> scores(num_rows);
      for (int r = 0; r < num_rows; ++r) {
        const float *k = keys + (r * num_heads_KV + n) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dot += q[d] * k[d];
        scores[r] = dot * scale;
      }

      // Softmax
      float max_s = *std::max_element(scores.begin(), scores.end());
      float sum_exp = 0.0f;
      for (int r = 0; r < num_rows; ++r) {
        scores[r] = std::exp(scores[r] - max_s);
        sum_exp += scores[r];
      }
      for (int r = 0; r < num_rows; ++r)
        scores[r] /= sum_exp;

      // Attn * V
      float *out = output + qh * head_dim;
      std::fill(out, out + head_dim, 0.0f);
      for (int r = 0; r < num_rows; ++r) {
        const float *v = values + (r * num_heads_KV + n) * head_dim;
        for (int d = 0; d < head_dim; ++d)
          out[d] += scores[r] * v[d];
      }
    }
  }
}

// Forward declaration
static void turboquant_attention(const float *query, const float *keys,
                                 const float *values, float *output,
                                 int num_rows, int num_heads_Q,
                                 int num_heads_KV, int head_dim);

/**
 * @brief TurboQuant v2 pipeline: quantize KV → packed attention.
 *        Same Q, K, V as reference, but K/V go through v2 quantization.
 */
TEST(turboquant_logit_compare, small_config) {
  constexpr int num_heads_Q = 4;
  constexpr int num_heads_KV = 2;
  constexpr int head_dim = 64;
  constexpr int context_len = 16;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  int kv_width = num_heads_KV * head_dim;
  int q_size = num_heads_Q * head_dim;

  std::vector<float> query(q_size);
  std::vector<float> keys(context_len * kv_width);
  std::vector<float> values(context_len * kv_width);
  for (auto &v : query)
    v = dist(gen);
  for (auto &v : keys)
    v = dist(gen);
  for (auto &v : values)
    v = dist(gen);

  std::vector<float> ref_out(q_size, 0.0f);
  std::vector<float> tq_out(q_size, 0.0f);

  fp32_reference_attention(query.data(), keys.data(), values.data(),
                           ref_out.data(), context_len, num_heads_Q,
                           num_heads_KV, head_dim);
  turboquant_attention(query.data(), keys.data(), values.data(), tq_out.data(),
                       context_len, num_heads_Q, num_heads_KV, head_dim);

  // Per-element comparison
  float max_diff = 0, sum_diff = 0, sum_sq_diff = 0;
  float max_ref = 0;
  int worst_idx = -1;

  for (int i = 0; i < q_size; ++i) {
    float diff = std::fabs(ref_out[i] - tq_out[i]);
    sum_diff += diff;
    sum_sq_diff += diff * diff;
    max_ref = std::max(max_ref, std::fabs(ref_out[i]));
    if (diff > max_diff) {
      max_diff = diff;
      worst_idx = i;
    }
  }
  float avg_diff = sum_diff / q_size;
  float rmse = std::sqrt(sum_sq_diff / q_size);
  float rel_max = (max_ref > 0) ? max_diff / max_ref : 0;

  std::cout << "\n=== Logit Comparison: small (heads_Q=4, heads_KV=2, dim=64, "
               "ctx=16) ===\n"
            << "  max_abs_diff   = " << max_diff << " (at index " << worst_idx
            << ")\n"
            << "  avg_abs_diff   = " << avg_diff << "\n"
            << "  rmse           = " << rmse << "\n"
            << "  max_rel_error  = " << (rel_max * 100) << "%\n"
            << "  max |ref|      = " << max_ref << "\n"
            << "  ref[worst]     = " << ref_out[worst_idx]
            << "  tq[worst]      = " << tq_out[worst_idx] << "\n";

  // Per-head breakdown
  int gqa = num_heads_Q / num_heads_KV;
  for (int h = 0; h < num_heads_Q; ++h) {
    float h_max = 0, h_sum = 0;
    for (int d = 0; d < head_dim; ++d) {
      float diff =
        std::fabs(ref_out[h * head_dim + d] - tq_out[h * head_dim + d]);
      h_max = std::max(h_max, diff);
      h_sum += diff;
    }
    std::cout << "  head " << h << ": max_diff=" << h_max
              << "  avg_diff=" << (h_sum / head_dim) << "\n";
  }

  EXPECT_LT(max_diff, 0.2f) << "Small config logit diff too large";
  EXPECT_LT(rmse, 0.05f) << "Small config RMSE too large";
}

/**
 * @brief Compare with Qwen3-1.7B-like dimensions.
 */
TEST(turboquant_logit_compare, qwen3_like) {
  constexpr int num_heads_Q = 16;
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;
  constexpr int context_len = 64;

  std::mt19937 gen(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  int kv_width = num_heads_KV * head_dim;
  int q_size = num_heads_Q * head_dim;

  std::vector<float> query(q_size);
  std::vector<float> keys(context_len * kv_width);
  std::vector<float> values(context_len * kv_width);
  for (auto &v : query)
    v = dist(gen);
  for (auto &v : keys)
    v = dist(gen);
  for (auto &v : values)
    v = dist(gen);

  std::vector<float> ref_out(q_size, 0.0f);
  std::vector<float> tq_out(q_size, 0.0f);

  fp32_reference_attention(query.data(), keys.data(), values.data(),
                           ref_out.data(), context_len, num_heads_Q,
                           num_heads_KV, head_dim);
  turboquant_attention(query.data(), keys.data(), values.data(), tq_out.data(),
                       context_len, num_heads_Q, num_heads_KV, head_dim);

  float max_diff = 0, sum_sq = 0, max_ref = 0;
  for (int i = 0; i < q_size; ++i) {
    float diff = std::fabs(ref_out[i] - tq_out[i]);
    max_diff = std::max(max_diff, diff);
    sum_sq += diff * diff;
    max_ref = std::max(max_ref, std::fabs(ref_out[i]));
  }
  float rmse = std::sqrt(sum_sq / q_size);
  float rel_max = (max_ref > 0) ? max_diff / max_ref : 0;

  std::cout << "\n=== Logit Comparison: qwen3-like (heads_Q=16, heads_KV=4, "
               "dim=128, ctx=64) ===\n"
            << "  max_abs_diff   = " << max_diff << "\n"
            << "  rmse           = " << rmse << "\n"
            << "  max_rel_error  = " << (rel_max * 100) << "%\n"
            << "  max |ref|      = " << max_ref << "\n";

  EXPECT_LT(max_diff, 0.2f) << "Qwen3-like logit diff too large";
  EXPECT_LT(rmse, 0.05f) << "Qwen3-like RMSE too large";
}

/**
 * @brief Compare across increasing context lengths.
 *        Shows how error scales with sequence length.
 */
TEST(turboquant_logit_compare, error_vs_context_length) {
  constexpr int num_heads_Q = 8;
  constexpr int num_heads_KV = 2;
  constexpr int head_dim = 128;

  std::cout
    << "\n=== Error vs Context Length (heads_Q=8, heads_KV=2, dim=128) ===\n"
    << "  ctx_len  max_diff    rmse      rel_max%\n";

  int q_size = num_heads_Q * head_dim;
  int kv_width = num_heads_KV * head_dim;

  for (int ctx : {4, 16, 64, 128, 256, 512}) {
    std::mt19937 gen(ctx * 7 + 1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> query(q_size);
    std::vector<float> keys(ctx * kv_width);
    std::vector<float> values(ctx * kv_width);
    for (auto &v : query)
      v = dist(gen);
    for (auto &v : keys)
      v = dist(gen);
    for (auto &v : values)
      v = dist(gen);

    std::vector<float> ref_out(q_size, 0.0f);
    std::vector<float> tq_out(q_size, 0.0f);

    fp32_reference_attention(query.data(), keys.data(), values.data(),
                             ref_out.data(), ctx, num_heads_Q, num_heads_KV,
                             head_dim);
    turboquant_attention(query.data(), keys.data(), values.data(),
                         tq_out.data(), ctx, num_heads_Q, num_heads_KV,
                         head_dim);

    float max_diff = 0, sum_sq = 0, max_ref = 0;
    for (int i = 0; i < q_size; ++i) {
      float diff = std::fabs(ref_out[i] - tq_out[i]);
      max_diff = std::max(max_diff, diff);
      sum_sq += diff * diff;
      max_ref = std::max(max_ref, std::fabs(ref_out[i]));
    }
    float rmse = std::sqrt(sum_sq / q_size);
    float rel = (max_ref > 0) ? max_diff / max_ref * 100 : 0;

    printf("  %5d    %.6f  %.6f  %.2f%%\n", ctx, max_diff, rmse, rel);

    EXPECT_LT(max_diff, 0.3f) << "ctx=" << ctx << " logit diff too large";
  }
}

/**
 * @brief TurboQuant v2 pipeline: norm + rotation + Lloyd-Max codebook.
 *        Paper Algorithm 1 (MSE-optimal).
 */
static void turboquant_attention(const float *query, const float *keys,
                                 const float *values, float *output,
                                 int num_rows, int num_heads_Q,
                                 int num_heads_KV, int head_dim) {

  int gqa_size = num_heads_Q / num_heads_KV;
  int kv_width = num_heads_KV * head_dim;
  int packed_row_bytes = kv_width / 2;

  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  // Quantize K and V with v2 (norm + rotation + Lloyd-Max)
  std::vector<uint8_t> pk(num_rows * packed_row_bytes);
  std::vector<float> k_norms(num_rows * num_heads_KV);
  std::vector<uint8_t> pv(num_rows * packed_row_bytes);
  std::vector<float> v_norms(num_rows * num_heads_KV);

  for (int r = 0; r < num_rows; ++r) {
    nntrainer::quantize_kv_turboquant(keys + r * kv_width,
                                      pk.data() + r * packed_row_bytes,
                                      k_norms.data() + r * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
    nntrainer::quantize_kv_turboquant(values + r * kv_width,
                                      pv.data() + r * packed_row_bytes,
                                      v_norms.data() + r * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
  }

  // Q*K^T
  std::vector<float> attn(num_rows * num_heads_Q, 0.0f);
  nntrainer::compute_kcaches_packed4(query, pk.data(), k_norms.data(),
                                     attn.data(), num_rows, num_heads_KV,
                                     head_dim, gqa_size, 4, rot_signs.data());

  // Softmax
  for (int h = 0; h < num_heads_Q; ++h) {
    float max_val = -1e30f;
    for (int r = 0; r < num_rows; ++r)
      max_val = std::max(max_val, attn[r * num_heads_Q + h]);
    float sum_exp = 0.0f;
    for (int r = 0; r < num_rows; ++r) {
      attn[r * num_heads_Q + h] = std::exp(attn[r * num_heads_Q + h] - max_val);
      sum_exp += attn[r * num_heads_Q + h];
    }
    for (int r = 0; r < num_rows; ++r)
      attn[r * num_heads_Q + h] /= sum_exp;
  }

  // Attn * V
  std::fill(output, output + num_heads_Q * head_dim, 0.0f);
  nntrainer::compute_vcache_packed4(num_rows - 1, attn.data(), pv.data(),
                                    v_norms.data(), output, num_heads_KV,
                                    gqa_size, head_dim, rot_signs.data());
}

TEST(turboquant_qwen3_sim, prefill_then_decode) {
  constexpr int num_heads_Q = 8;
  constexpr int num_heads_KV = 2;
  constexpr int head_dim = 64;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int prefill_len = 16; // prompt length
  constexpr int decode_len = 16;  // tokens to generate
  constexpr int max_seq = prefill_len + decode_len;

  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int q_width = num_heads_Q * head_dim;
  constexpr int packed_row = kv_width / 2;

  std::mt19937 gen(20260329);
  std::normal_distribution<float> normal(0.0f, 0.3f);
  std::uniform_real_distribution<float> outlier_val(-4.0f, 4.0f);

  auto gen_llm_vec = [&](float *dst, int n) {
    for (int i = 0; i < n; ++i)
      dst[i] = normal(gen);
    for (int i = 0; i < n / 20; ++i)
      dst[std::abs((int)gen()) % n] = outlier_val(gen);
  };

  // Rotation signs (shared, generated once)
  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  // FP32 KV cache (ground truth)
  std::vector<float> fp32_kcache(max_seq * kv_width, 0.0f);
  std::vector<float> fp32_vcache(max_seq * kv_width, 0.0f);

  // TurboQuant v2 packed cache
  std::vector<uint8_t> tq_kcache(max_seq * packed_row, 0);
  std::vector<float> tq_knorms(max_seq * num_heads_KV, 0.0f);
  std::vector<uint8_t> tq_vcache(max_seq * packed_row, 0);
  std::vector<float> tq_vnorms(max_seq * num_heads_KV, 0.0f);

  // ========== PREFILL ==========
  // Generate all prompt K,V at once (same data for both paths)
  std::vector<float> prompt_keys(prefill_len * kv_width);
  std::vector<float> prompt_values(prefill_len * kv_width);
  for (int t = 0; t < prefill_len; ++t) {
    gen_llm_vec(prompt_keys.data() + t * kv_width, kv_width);
    gen_llm_vec(prompt_values.data() + t * kv_width, kv_width);
  }

  // Store in FP32 cache
  std::copy(prompt_keys.begin(), prompt_keys.end(), fp32_kcache.begin());
  std::copy(prompt_values.begin(), prompt_values.end(), fp32_vcache.begin());

  // Quantize into TQ v2 cache (all prefill tokens at once)
  for (int t = 0; t < prefill_len; ++t) {
    nntrainer::quantize_kv_turboquant(prompt_keys.data() + t * kv_width,
                                      tq_kcache.data() + t * packed_row,
                                      tq_knorms.data() + t * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
    nntrainer::quantize_kv_turboquant(prompt_values.data() + t * kv_width,
                                      tq_vcache.data() + t * packed_row,
                                      tq_vnorms.data() + t * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
  }

  std::cout << "\n=== Prefill(" << prefill_len << ") + Decode(" << decode_len
            << "): FP32 vs TurboQuant v2 ===\n"
            << "  Config: heads_Q=" << num_heads_Q
            << ", heads_KV=" << num_heads_KV << ", dim=" << head_dim << "\n"
            << "  step  ctx_len  max_diff  cosine_sim\n";

  // ========== DECODE ==========
  float total_cosine = 0;
  float worst_max_diff = 0;

  for (int d = 0; d < decode_len; ++d) {
    int pos = prefill_len + d; // current position in sequence
    int ctx_len = pos + 1; // total context (prefill + decoded so far + current)

    // Generate new K,V for this decode token (same data for both)
    std::vector<float> new_k(kv_width), new_v(kv_width);
    gen_llm_vec(new_k.data(), kv_width);
    gen_llm_vec(new_v.data(), kv_width);

    // Append to FP32 cache
    std::copy(new_k.begin(), new_k.end(), fp32_kcache.begin() + pos * kv_width);
    std::copy(new_v.begin(), new_v.end(), fp32_vcache.begin() + pos * kv_width);

    // Quantize and append to TQ cache
    nntrainer::quantize_kv_turboquant(new_k.data(),
                                      tq_kcache.data() + pos * packed_row,
                                      tq_knorms.data() + pos * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
    nntrainer::quantize_kv_turboquant(new_v.data(),
                                      tq_vcache.data() + pos * packed_row,
                                      tq_vnorms.data() + pos * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);

    // Generate query (only query is new at decode time)
    std::vector<float> query(q_width);
    gen_llm_vec(query.data(), q_width);

    // ---- FP32 attention (ground truth) ----
    std::vector<float> fp32_out(q_width, 0.0f);
    fp32_reference_attention(query.data(), fp32_kcache.data(),
                             fp32_vcache.data(), fp32_out.data(), ctx_len,
                             num_heads_Q, num_heads_KV, head_dim);

    // ---- TQ v2 attention (using packed prefill + decode cache) ----
    // Q * K_packed^T (reads ALL ctx_len rows from packed cache)
    std::vector<float> tq_scores(ctx_len * num_heads_Q, 0.0f);
    nntrainer::compute_kcaches_packed4(
      query.data(), tq_kcache.data(), tq_knorms.data(), tq_scores.data(),
      ctx_len, num_heads_KV, head_dim, gqa_size, 4, rot_signs.data());

    // Softmax
    for (int h = 0; h < num_heads_Q; ++h) {
      float mx = -1e30f;
      for (int r = 0; r < ctx_len; ++r)
        mx = std::max(mx, tq_scores[r * num_heads_Q + h]);
      float se = 0;
      for (int r = 0; r < ctx_len; ++r) {
        tq_scores[r * num_heads_Q + h] =
          std::exp(tq_scores[r * num_heads_Q + h] - mx);
        se += tq_scores[r * num_heads_Q + h];
      }
      for (int r = 0; r < ctx_len; ++r)
        tq_scores[r * num_heads_Q + h] /= se;
    }

    // Attn * V_packed (reads ALL ctx_len rows from packed value cache)
    std::vector<float> tq_out(q_width, 0.0f);
    nntrainer::compute_vcache_packed4(
      pos, tq_scores.data(), tq_vcache.data(), tq_vnorms.data(), tq_out.data(),
      num_heads_KV, gqa_size, head_dim, rot_signs.data());

    // Compare
    float max_diff = 0, dot_ab = 0, dot_aa = 0, dot_bb = 0;
    for (int i = 0; i < q_width; ++i) {
      ASSERT_TRUE(std::isfinite(tq_out[i]))
        << "Decode " << d << ": non-finite at " << i;
      float diff = std::fabs(fp32_out[i] - tq_out[i]);
      max_diff = std::max(max_diff, diff);
      dot_ab += fp32_out[i] * tq_out[i];
      dot_aa += fp32_out[i] * fp32_out[i];
      dot_bb += tq_out[i] * tq_out[i];
    }
    float cosine = (dot_aa > 0 && dot_bb > 0)
                     ? dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb))
                     : 0.0f;

    total_cosine += cosine;
    worst_max_diff = std::max(worst_max_diff, max_diff);

    printf("  %4d  %7d  %.6f  %.6f\n", d, ctx_len, max_diff, cosine);
  }

  float avg_cosine = total_cosine / decode_len;
  std::cout << "\n  Summary:\n"
            << "  avg_cosine_sim  = " << avg_cosine << "\n"
            << "  worst_max_diff  = " << worst_max_diff << "\n"
            << "  prefill tokens  = " << prefill_len << " (packed once)\n"
            << "  decode tokens   = " << decode_len << "\n";

  EXPECT_GT(avg_cosine, 0.95f)
    << "Prefill+decode cosine sim too low: " << avg_cosine;
}

/**
 * @brief Qwen3-1.7B-like prefill+decode with larger dimensions.
 */
TEST(turboquant_qwen3_sim, prefill_decode_1_7b_like) {
  constexpr int num_heads_Q = 16;
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int prefill_len = 32;
  constexpr int decode_len = 32;
  constexpr int max_seq = prefill_len + decode_len;

  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int q_width = num_heads_Q * head_dim;
  constexpr int packed_row = kv_width / 2;

  std::mt19937 gen(42);
  std::normal_distribution<float> normal(0.0f, 0.3f);
  std::uniform_real_distribution<float> outlier_val(-5.0f, 5.0f);

  auto gen_llm_vec = [&](float *dst, int n) {
    for (int i = 0; i < n; ++i)
      dst[i] = normal(gen);
    for (int i = 0; i < n / 20; ++i)
      dst[std::abs((int)gen()) % n] = outlier_val(gen);
  };

  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  std::vector<float> fp32_kc(max_seq * kv_width, 0.0f);
  std::vector<float> fp32_vc(max_seq * kv_width, 0.0f);
  std::vector<uint8_t> tq_kc(max_seq * packed_row, 0);
  std::vector<float> tq_kn(max_seq * num_heads_KV, 0.0f);
  std::vector<uint8_t> tq_vc(max_seq * packed_row, 0);
  std::vector<float> tq_vn(max_seq * num_heads_KV, 0.0f);

  // Prefill
  for (int t = 0; t < prefill_len; ++t) {
    gen_llm_vec(fp32_kc.data() + t * kv_width, kv_width);
    gen_llm_vec(fp32_vc.data() + t * kv_width, kv_width);
    nntrainer::quantize_kv_turboquant(fp32_kc.data() + t * kv_width,
                                      tq_kc.data() + t * packed_row,
                                      tq_kn.data() + t * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
    nntrainer::quantize_kv_turboquant(fp32_vc.data() + t * kv_width,
                                      tq_vc.data() + t * packed_row,
                                      tq_vn.data() + t * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
  }

  float total_cosine = 0;

  std::cout << "\n=== Qwen3-1.7B-like Prefill(" << prefill_len << ")+Decode("
            << decode_len << ") ===\n"
            << "  step  ctx_len  max_diff  cosine_sim\n";

  // Decode
  for (int d = 0; d < decode_len; ++d) {
    int pos = prefill_len + d;
    int ctx_len = pos + 1;

    gen_llm_vec(fp32_kc.data() + pos * kv_width, kv_width);
    gen_llm_vec(fp32_vc.data() + pos * kv_width, kv_width);
    nntrainer::quantize_kv_turboquant(fp32_kc.data() + pos * kv_width,
                                      tq_kc.data() + pos * packed_row,
                                      tq_kn.data() + pos * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
    nntrainer::quantize_kv_turboquant(fp32_vc.data() + pos * kv_width,
                                      tq_vc.data() + pos * packed_row,
                                      tq_vn.data() + pos * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);

    std::vector<float> query(q_width);
    gen_llm_vec(query.data(), q_width);

    // FP32 ref
    std::vector<float> fp32_out(q_width, 0.0f);
    fp32_reference_attention(query.data(), fp32_kc.data(), fp32_vc.data(),
                             fp32_out.data(), ctx_len, num_heads_Q,
                             num_heads_KV, head_dim);

    // TQ v2
    std::vector<float> tq_scores(ctx_len * num_heads_Q, 0.0f);
    nntrainer::compute_kcaches_packed4(query.data(), tq_kc.data(), tq_kn.data(),
                                       tq_scores.data(), ctx_len, num_heads_KV,
                                       head_dim, gqa_size, 4, rot_signs.data());

    for (int h = 0; h < num_heads_Q; ++h) {
      float mx = -1e30f;
      for (int r = 0; r < ctx_len; ++r)
        mx = std::max(mx, tq_scores[r * num_heads_Q + h]);
      float se = 0;
      for (int r = 0; r < ctx_len; ++r) {
        tq_scores[r * num_heads_Q + h] =
          std::exp(tq_scores[r * num_heads_Q + h] - mx);
        se += tq_scores[r * num_heads_Q + h];
      }
      for (int r = 0; r < ctx_len; ++r)
        tq_scores[r * num_heads_Q + h] /= se;
    }

    std::vector<float> tq_out(q_width, 0.0f);
    nntrainer::compute_vcache_packed4(pos, tq_scores.data(), tq_vc.data(),
                                      tq_vn.data(), tq_out.data(), num_heads_KV,
                                      gqa_size, head_dim, rot_signs.data());

    float max_diff = 0, dot_ab = 0, dot_aa = 0, dot_bb = 0;
    for (int i = 0; i < q_width; ++i) {
      ASSERT_TRUE(std::isfinite(tq_out[i]));
      float diff = std::fabs(fp32_out[i] - tq_out[i]);
      max_diff = std::max(max_diff, diff);
      dot_ab += fp32_out[i] * tq_out[i];
      dot_aa += fp32_out[i] * fp32_out[i];
      dot_bb += tq_out[i] * tq_out[i];
    }
    float cosine = (dot_aa > 0 && dot_bb > 0)
                     ? dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb))
                     : 0.0f;
    total_cosine += cosine;

    if (d < 5 || d == decode_len - 1)
      printf("  %4d  %7d  %.6f  %.6f\n", d, ctx_len, max_diff, cosine);
    else if (d == 5)
      printf("  ...   ...\n");
  }

  float avg_cosine = total_cosine / decode_len;
  std::cout << "  avg_cosine_sim = " << avg_cosine << "\n";

  EXPECT_GT(avg_cosine, 0.95f)
    << "Qwen3-1.7B prefill+decode cosine too low: " << avg_cosine;
}

/**
 * @brief Step-by-step MHA pipeline comparison that mirrors mha_core.cpp
 * EXACTLY.
 *
 *  This test calls compute_kcaches_packed4 and compute_vcache_packed4
 *  the SAME way mha_core.cpp does:
 *    - Per KV-head loop (head_start, head_end) for OpenMP-like parallelization
 *    - Explicit local_window_size parameter
 *    - Same output tensor layout: [row * num_heads_Q + head_q]
 *
 *  Compares intermediate results:
 *    Step 1: Q*K^T scores (before softmax)
 *    Step 2: Softmax weights
 *    Step 3: V*attn output (final)
 */
TEST(turboquant_mha_core_pipeline, single_token_stepwise_comparison) {
  constexpr int num_heads_Q = 16;
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int tile_size = 4;
  constexpr int context_len = 64;
  constexpr size_t local_window_size = 4096; // typical Qwen3 config

  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int q_width = num_heads_Q * head_dim;
  constexpr int packed_row = kv_width / 2;

  std::mt19937 gen(20260330);
  std::normal_distribution<float> normal(0.0f, 0.3f);
  std::uniform_real_distribution<float> outlier(-4.0f, 4.0f);

  auto gen_llm = [&](float *dst, int n) {
    for (int i = 0; i < n; ++i)
      dst[i] = normal(gen);
    for (int i = 0; i < n / 20; ++i)
      dst[std::abs((int)gen()) % n] = outlier(gen);
  };

  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  // Generate KV data (same for both paths)
  std::vector<float> fp32_keys(context_len * kv_width);
  std::vector<float> fp32_values(context_len * kv_width);
  for (int t = 0; t < context_len; ++t) {
    gen_llm(fp32_keys.data() + t * kv_width, kv_width);
    gen_llm(fp32_values.data() + t * kv_width, kv_width);
  }

  // TurboQuant v2 cache
  std::vector<uint8_t> tq_kc(context_len * packed_row);
  std::vector<float> tq_kn(context_len * num_heads_KV);
  std::vector<uint8_t> tq_vc(context_len * packed_row);
  std::vector<float> tq_vn(context_len * num_heads_KV);

  for (int t = 0; t < context_len; ++t) {
    nntrainer::quantize_kv_turboquant(fp32_keys.data() + t * kv_width,
                                      tq_kc.data() + t * packed_row,
                                      tq_kn.data() + t * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
    nntrainer::quantize_kv_turboquant(fp32_values.data() + t * kv_width,
                                      tq_vc.data() + t * packed_row,
                                      tq_vn.data() + t * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
  }

  // Decode at position (context_len - 1) looking at all context
  int from = context_len - 1;
  int to = context_len;
  int row_to_compute = from + 1; // causal: from + 1

  std::vector<float> query(q_width);
  gen_llm(query.data(), q_width);

  // ======== STEP 1: Q*K^T scores ========
  // FP32 reference
  float scale = 1.0f / std::sqrt((float)head_dim);
  std::vector<float> ref_scores(row_to_compute * num_heads_Q, 0.0f);
  for (int n = 0; n < num_heads_KV; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      int qh = n * gqa_size + g;
      const float *q = query.data() + qh * head_dim;
      for (int r = 0; r < row_to_compute; ++r) {
        const float *k = fp32_keys.data() + (r * num_heads_KV + n) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dot += q[d] * k[d];
        ref_scores[r * num_heads_Q + qh] = dot * scale;
      }
    }
  }

  // TQ v2 scores - per KV-head loop (matching mha_core.cpp OpenMP pattern)
  std::vector<float> tq_scores(row_to_compute * num_heads_Q, 0.0f);
  for (int head_kv = 0; head_kv < num_heads_KV; ++head_kv) {
    nntrainer::compute_kcaches_packed4(
      query.data(), tq_kc.data(), tq_kn.data(), tq_scores.data(),
      row_to_compute, num_heads_KV, head_dim, gqa_size, tile_size,
      rot_signs.data(), local_window_size, head_kv, head_kv + 1);
  }

  // Compare scores
  float score_max_diff = 0, score_sq = 0;
  for (int i = 0; i < row_to_compute * num_heads_Q; ++i) {
    float diff = std::fabs(ref_scores[i] - tq_scores[i]);
    score_max_diff = std::max(score_max_diff, diff);
    score_sq += diff * diff;
  }
  float score_rmse = std::sqrt(score_sq / (row_to_compute * num_heads_Q));

  // ======== STEP 2: Softmax ========
  // FP32 ref softmax
  std::vector<float> ref_attn = ref_scores;
  for (int h = 0; h < num_heads_Q; ++h) {
    float mx = -1e30f;
    for (int r = 0; r < row_to_compute; ++r)
      mx = std::max(mx, ref_attn[r * num_heads_Q + h]);
    float se = 0;
    for (int r = 0; r < row_to_compute; ++r) {
      ref_attn[r * num_heads_Q + h] =
        std::exp(ref_attn[r * num_heads_Q + h] - mx);
      se += ref_attn[r * num_heads_Q + h];
    }
    for (int r = 0; r < row_to_compute; ++r)
      ref_attn[r * num_heads_Q + h] /= se;
  }

  // TQ softmax
  std::vector<float> tq_attn = tq_scores;
  for (int h = 0; h < num_heads_Q; ++h) {
    float mx = -1e30f;
    for (int r = 0; r < row_to_compute; ++r)
      mx = std::max(mx, tq_attn[r * num_heads_Q + h]);
    float se = 0;
    for (int r = 0; r < row_to_compute; ++r) {
      tq_attn[r * num_heads_Q + h] =
        std::exp(tq_attn[r * num_heads_Q + h] - mx);
      se += tq_attn[r * num_heads_Q + h];
    }
    for (int r = 0; r < row_to_compute; ++r)
      tq_attn[r * num_heads_Q + h] /= se;
  }

  // Compare softmax weights
  float attn_max_diff = 0, attn_sq = 0;
  for (int i = 0; i < row_to_compute * num_heads_Q; ++i) {
    float diff = std::fabs(ref_attn[i] - tq_attn[i]);
    attn_max_diff = std::max(attn_max_diff, diff);
    attn_sq += diff * diff;
  }
  float attn_rmse = std::sqrt(attn_sq / (row_to_compute * num_heads_Q));

  // ======== STEP 3: V*attn output ========
  // FP32 reference
  std::vector<float> ref_out(q_width, 0.0f);
  for (int n = 0; n < num_heads_KV; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      int qh = n * gqa_size + g;
      float *out = ref_out.data() + qh * head_dim;
      for (int r = 0; r < row_to_compute; ++r) {
        float w = ref_attn[r * num_heads_Q + qh];
        const float *v = fp32_values.data() + (r * num_heads_KV + n) * head_dim;
        for (int d = 0; d < head_dim; ++d)
          out[d] += w * v[d];
      }
    }
  }

  // TQ v2 output - per KV-head loop (matching mha_core.cpp)
  std::vector<float> tq_out(q_width, 0.0f);
  int row_num = to - 1;
  for (int head_kv = 0; head_kv < num_heads_KV; ++head_kv) {
    nntrainer::compute_vcache_packed4(row_num, tq_attn.data(), tq_vc.data(),
                                      tq_vn.data(), tq_out.data(), num_heads_KV,
                                      gqa_size, head_dim, rot_signs.data(),
                                      local_window_size, head_kv, head_kv + 1);
  }

  // Compare final output
  float out_max_diff = 0, out_sq = 0, dot_ab = 0, dot_aa = 0, dot_bb = 0;
  for (int i = 0; i < q_width; ++i) {
    ASSERT_TRUE(std::isfinite(tq_out[i])) << "Non-finite at " << i;
    float diff = std::fabs(ref_out[i] - tq_out[i]);
    out_max_diff = std::max(out_max_diff, diff);
    out_sq += diff * diff;
    dot_ab += ref_out[i] * tq_out[i];
    dot_aa += ref_out[i] * ref_out[i];
    dot_bb += tq_out[i] * tq_out[i];
  }
  float out_rmse = std::sqrt(out_sq / q_width);
  float cosine = (dot_aa > 0 && dot_bb > 0)
                   ? dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb))
                   : 0.0f;

  std::cout
    << "\n=== MHA Core Pipeline Step-by-Step (single-token decode) ===\n"
    << "  Config: heads_Q=" << num_heads_Q << ", heads_KV=" << num_heads_KV
    << ", dim=" << head_dim << ", ctx=" << context_len << "\n"
    << "  Step 1 (Q*K^T scores):  max_diff=" << score_max_diff
    << "  rmse=" << score_rmse << "\n"
    << "  Step 2 (softmax wts):   max_diff=" << attn_max_diff
    << "  rmse=" << attn_rmse << "\n"
    << "  Step 3 (V*attn output): max_diff=" << out_max_diff
    << "  rmse=" << out_rmse << "  cosine=" << cosine << "\n";

  // Per-head breakdown of final output
  std::cout << "  Per-head output error:\n";
  for (int h = 0; h < num_heads_Q; ++h) {
    float h_max = 0, h_sq = 0;
    for (int d = 0; d < head_dim; ++d) {
      float diff =
        std::fabs(ref_out[h * head_dim + d] - tq_out[h * head_dim + d]);
      h_max = std::max(h_max, diff);
      h_sq += diff * diff;
    }
    float h_rmse = std::sqrt(h_sq / head_dim);
    if (h < 4 || h == num_heads_Q - 1)
      printf("    head %2d: max_diff=%.6f  rmse=%.6f\n", h, h_max, h_rmse);
    else if (h == 4)
      printf("    ...\n");
  }

  EXPECT_GT(cosine, 0.98f) << "Pipeline cosine too low";
  EXPECT_LT(out_rmse, 0.05f) << "Pipeline RMSE too high";
}

/**
 * @brief Test with small local_window_size to verify windowed attention
 *        consistency between TQ v2 compute functions and the expected
 *        mha_core.cpp softmax_triangle behavior.
 *
 *  When local_window_size < context_len, kcache only computes scores
 *  for the last local_window_size rows. Softmax should only process
 *  those rows. This test verifies the output is correct.
 */
TEST(turboquant_mha_core_pipeline, windowed_attention_consistency) {
  constexpr int num_heads_Q = 8;
  constexpr int num_heads_KV = 2;
  constexpr int head_dim = 128;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int tile_size = 4;
  constexpr int context_len = 64;
  constexpr size_t local_window_size = 16; // small window

  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int q_width = num_heads_Q * head_dim;
  constexpr int packed_row = kv_width / 2;

  std::mt19937 gen(12345);
  std::normal_distribution<float> normal(0.0f, 0.3f);

  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  // Build KV cache
  std::vector<float> fp32_keys(context_len * kv_width);
  std::vector<float> fp32_values(context_len * kv_width);
  for (auto &v : fp32_keys)
    v = normal(gen);
  for (auto &v : fp32_values)
    v = normal(gen);

  std::vector<uint8_t> tq_kc(context_len * packed_row);
  std::vector<float> tq_kn(context_len * num_heads_KV);
  std::vector<uint8_t> tq_vc(context_len * packed_row);
  std::vector<float> tq_vn(context_len * num_heads_KV);

  for (int t = 0; t < context_len; ++t) {
    nntrainer::quantize_kv_turboquant(fp32_keys.data() + t * kv_width,
                                      tq_kc.data() + t * packed_row,
                                      tq_kn.data() + t * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
    nntrainer::quantize_kv_turboquant(fp32_values.data() + t * kv_width,
                                      tq_vc.data() + t * packed_row,
                                      tq_vn.data() + t * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
  }

  // Decode at the last position
  int from = context_len - 1;
  int row_to_compute = from + 1; // causal

  std::vector<float> query(q_width);
  for (auto &v : query)
    v = normal(gen);

  // ---- TQ path with windowing (as mha_core.cpp does) ----
  // Allocate out_ like mha_core.cpp: size = row_to_compute * num_heads_Q
  // But setZero (critical for windowed case!)
  std::vector<float> tq_scores(row_to_compute * num_heads_Q, 0.0f);

  for (int head_kv = 0; head_kv < num_heads_KV; ++head_kv) {
    nntrainer::compute_kcaches_packed4(
      query.data(), tq_kc.data(), tq_kn.data(), tq_scores.data(),
      row_to_compute, num_heads_KV, head_dim, gqa_size, tile_size,
      rot_signs.data(), local_window_size, head_kv, head_kv + 1);
  }

  // Verify layout: compute_kcaches_packed4 writes scores at indices
  // 0..row_cnt-1 (compacted), where row_cnt = min(row_to_compute, LWS).
  // Indices row_cnt..row_to_compute-1 should still be 0 (from our setZero).
  int row_cnt = row_to_compute < (int)local_window_size
                  ? row_to_compute
                  : (int)local_window_size;
  ASSERT_LT(row_cnt, row_to_compute) << "Test requires context > window";

  bool zeros_after_window = true;
  for (int r = row_cnt; r < row_to_compute; ++r) {
    for (int h = 0; h < num_heads_Q; ++h) {
      if (tq_scores[r * num_heads_Q + h] != 0.0f) {
        zeros_after_window = false;
        break;
      }
    }
  }
  EXPECT_TRUE(zeros_after_window)
    << "Scores after row_cnt should be 0 (from setZero)";

  // Softmax: mimic softmax_triangle for row==1, causal
  // end_row = from < local_window_size ? from + 1 : local_window_size
  int end_row =
    from < (int)local_window_size ? from + 1 : (int)local_window_size;

  // Only softmax the first end_row elements (matching mha_core.cpp)
  for (int h = 0; h < num_heads_Q; ++h) {
    float mx = -1e30f;
    for (int r = 0; r < end_row; ++r)
      mx = std::max(mx, tq_scores[r * num_heads_Q + h]);
    float se = 0;
    for (int r = 0; r < end_row; ++r) {
      tq_scores[r * num_heads_Q + h] =
        std::exp(tq_scores[r * num_heads_Q + h] - mx);
      se += tq_scores[r * num_heads_Q + h];
    }
    for (int r = 0; r < end_row; ++r)
      tq_scores[r * num_heads_Q + h] /= se;
  }

  // V computation with windowing
  std::vector<float> tq_out(q_width, 0.0f);
  for (int head_kv = 0; head_kv < num_heads_KV; ++head_kv) {
    nntrainer::compute_vcache_packed4(from, tq_scores.data(), tq_vc.data(),
                                      tq_vn.data(), tq_out.data(), num_heads_KV,
                                      gqa_size, head_dim, rot_signs.data(),
                                      local_window_size, head_kv, head_kv + 1);
  }

  // ---- FP32 windowed reference ----
  // Only attend to the last local_window_size positions
  int ws = row_to_compute < (int)local_window_size
             ? 0
             : row_to_compute - (int)local_window_size;
  std::vector<float> ref_out(q_width, 0.0f);
  float scale = 1.0f / std::sqrt((float)head_dim);

  for (int n = 0; n < num_heads_KV; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      int qh = n * gqa_size + g;
      const float *q = query.data() + qh * head_dim;

      // Q*K^T for window only
      std::vector<float> scores;
      for (int r = ws; r < row_to_compute; ++r) {
        const float *k = fp32_keys.data() + (r * num_heads_KV + n) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dot += q[d] * k[d];
        scores.push_back(dot * scale);
      }

      // Softmax
      float mx = *std::max_element(scores.begin(), scores.end());
      float se = 0;
      for (auto &s : scores) {
        s = std::exp(s - mx);
        se += s;
      }
      for (auto &s : scores)
        s /= se;

      // V aggregation
      float *out = ref_out.data() + qh * head_dim;
      for (int r = ws; r < row_to_compute; ++r) {
        float w = scores[r - ws];
        const float *v = fp32_values.data() + (r * num_heads_KV + n) * head_dim;
        for (int d = 0; d < head_dim; ++d)
          out[d] += w * v[d];
      }
    }
  }

  // Compare
  float max_diff = 0, sum_sq = 0, dot_ab = 0, dot_aa = 0, dot_bb = 0;
  for (int i = 0; i < q_width; ++i) {
    ASSERT_TRUE(std::isfinite(tq_out[i]));
    float diff = std::fabs(ref_out[i] - tq_out[i]);
    max_diff = std::max(max_diff, diff);
    sum_sq += diff * diff;
    dot_ab += ref_out[i] * tq_out[i];
    dot_aa += ref_out[i] * ref_out[i];
    dot_bb += tq_out[i] * tq_out[i];
  }
  float rmse = std::sqrt(sum_sq / q_width);
  float cosine = dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb));

  std::cout << "\n=== Windowed Attention (window=" << local_window_size
            << ", ctx=" << context_len << ") ===\n"
            << "  max_diff=" << max_diff << "  rmse=" << rmse
            << "  cosine=" << cosine << "\n";

  EXPECT_GT(cosine, 0.98f) << "Windowed attention cosine too low";
}

/**
 * @brief Multi-step decode test with per-head parallelization.
 *        Verifies that calling compute_kcaches_packed4 per KV-head
 *        produces the same result as calling it all-at-once.
 */
TEST(turboquant_mha_core_pipeline, per_head_vs_allheads) {
  constexpr int num_heads_Q = 16;
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int context_len = 32;
  constexpr size_t local_window_size = UINT_MAX;

  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int q_width = num_heads_Q * head_dim;
  constexpr int packed_row = kv_width / 2;

  std::mt19937 gen(77777);
  std::normal_distribution<float> normal(0.0f, 0.5f);

  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  // Build cache
  std::vector<uint8_t> tq_kc(context_len * packed_row);
  std::vector<float> tq_kn(context_len * num_heads_KV);
  std::vector<uint8_t> tq_vc(context_len * packed_row);
  std::vector<float> tq_vn(context_len * num_heads_KV);

  std::vector<float> kv_data(kv_width);
  for (int t = 0; t < context_len; ++t) {
    for (auto &v : kv_data)
      v = normal(gen);
    nntrainer::quantize_kv_turboquant(kv_data.data(),
                                      tq_kc.data() + t * packed_row,
                                      tq_kn.data() + t * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
    for (auto &v : kv_data)
      v = normal(gen);
    nntrainer::quantize_kv_turboquant(kv_data.data(),
                                      tq_vc.data() + t * packed_row,
                                      tq_vn.data() + t * num_heads_KV,
                                      rot_signs.data(), head_dim, num_heads_KV);
  }

  std::vector<float> query(q_width);
  for (auto &v : query)
    v = normal(gen);

  int row_to_compute = context_len;

  // All-at-once (default: head_start=0, head_end=-1)
  std::vector<float> scores_all(row_to_compute * num_heads_Q, 0.0f);
  nntrainer::compute_kcaches_packed4(
    query.data(), tq_kc.data(), tq_kn.data(), scores_all.data(), row_to_compute,
    num_heads_KV, head_dim, gqa_size, 4, rot_signs.data(), local_window_size);

  // Per KV-head (matching mha_core.cpp OpenMP pattern)
  std::vector<float> scores_per(row_to_compute * num_heads_Q, 0.0f);
  for (int hkv = 0; hkv < num_heads_KV; ++hkv) {
    nntrainer::compute_kcaches_packed4(
      query.data(), tq_kc.data(), tq_kn.data(), scores_per.data(),
      row_to_compute, num_heads_KV, head_dim, gqa_size, 4, rot_signs.data(),
      local_window_size, hkv, hkv + 1);
  }

  // Should be bit-exact
  float max_diff = 0;
  for (int i = 0; i < row_to_compute * num_heads_Q; ++i) {
    float diff = std::fabs(scores_all[i] - scores_per[i]);
    max_diff = std::max(max_diff, diff);
  }

  std::cout << "\n=== Per-head vs All-heads (kcache scores) ===\n"
            << "  max_diff = " << max_diff << " (should be 0 or near-0)\n";
  EXPECT_LT(max_diff, 1e-6f) << "Per-head and all-heads should match exactly";

  // Same for V computation
  // First softmax the scores
  std::vector<float> attn_all = scores_all;
  std::vector<float> attn_per = scores_per;
  for (auto *attn : {&attn_all, &attn_per}) {
    for (int h = 0; h < num_heads_Q; ++h) {
      float mx = -1e30f;
      for (int r = 0; r < row_to_compute; ++r)
        mx = std::max(mx, (*attn)[r * num_heads_Q + h]);
      float se = 0;
      for (int r = 0; r < row_to_compute; ++r) {
        (*attn)[r * num_heads_Q + h] =
          std::exp((*attn)[r * num_heads_Q + h] - mx);
        se += (*attn)[r * num_heads_Q + h];
      }
      for (int r = 0; r < row_to_compute; ++r)
        (*attn)[r * num_heads_Q + h] /= se;
    }
  }

  std::vector<float> vout_all(q_width, 0.0f);
  nntrainer::compute_vcache_packed4(context_len - 1, attn_all.data(),
                                    tq_vc.data(), tq_vn.data(), vout_all.data(),
                                    num_heads_KV, gqa_size, head_dim,
                                    rot_signs.data(), local_window_size);

  std::vector<float> vout_per(q_width, 0.0f);
  for (int hkv = 0; hkv < num_heads_KV; ++hkv) {
    nntrainer::compute_vcache_packed4(
      context_len - 1, attn_per.data(), tq_vc.data(), tq_vn.data(),
      vout_per.data(), num_heads_KV, gqa_size, head_dim, rot_signs.data(),
      local_window_size, hkv, hkv + 1);
  }

  float vmax_diff = 0;
  for (int i = 0; i < q_width; ++i) {
    float diff = std::fabs(vout_all[i] - vout_per[i]);
    vmax_diff = std::max(vmax_diff, diff);
  }

  std::cout << "  vcache per-head vs all-heads max_diff = " << vmax_diff
            << "\n";
  EXPECT_LT(vmax_diff, 1e-5f) << "V per-head and all-heads should match";
}

/**
 * @brief Dequantize-then-recompute test: verify that TQ v2 quantization
 *        + dequantization preserves vectors within expected error bounds.
 *        This isolates quantization error from the attention computation.
 */
TEST(turboquant_mha_core_pipeline, quantize_dequantize_fidelity) {
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;
  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int packed_row = kv_width / 2;

  std::mt19937 gen(42);
  std::normal_distribution<float> normal(0.0f, 0.3f);
  std::uniform_real_distribution<float> outlier(-5.0f, 5.0f);

  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  int num_samples = 100;
  float total_cosine = 0, worst_cosine = 1.0f;
  float total_rel_error = 0, worst_rel_error = 0;

  for (int s = 0; s < num_samples; ++s) {
    // Generate one KV vector
    std::vector<float> orig(kv_width);
    for (auto &v : orig)
      v = normal(gen);
    if (s % 5 == 0) { // add outliers
      for (int i = 0; i < kv_width / 20; ++i)
        orig[std::abs((int)gen()) % kv_width] = outlier(gen);
    }

    // Quantize
    std::vector<uint8_t> packed(packed_row);
    std::vector<float> norms(num_heads_KV);
    nntrainer::quantize_kv_turboquant(orig.data(), packed.data(), norms.data(),
                                      rot_signs.data(), head_dim, num_heads_KV);

    // Dequantize (use turboquant_utils.h dequantize function)
    std::vector<float> recon(kv_width, 0.0f);
    for (int h = 0; h < num_heads_KV; ++h) {
      nntrainer::turboquant_dequantize_head(
        packed.data() + h * head_dim / 2, norms[h], head_dim,
        recon.data() + h * head_dim, rot_signs.data(),
        nntrainer::get_codebook(head_dim));
    }

    // Compare
    float dot_ab = 0, dot_aa = 0, dot_bb = 0;
    for (int i = 0; i < kv_width; ++i) {
      dot_ab += orig[i] * recon[i];
      dot_aa += orig[i] * orig[i];
      dot_bb += recon[i] * recon[i];
    }
    float cosine = (dot_aa > 0 && dot_bb > 0)
                     ? dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb))
                     : 0.0f;
    float norm_orig = std::sqrt(dot_aa);
    float norm_diff = 0;
    for (int i = 0; i < kv_width; ++i) {
      float d = orig[i] - recon[i];
      norm_diff += d * d;
    }
    norm_diff = std::sqrt(norm_diff);
    float rel_err = norm_orig > 0 ? norm_diff / norm_orig : 0;

    total_cosine += cosine;
    worst_cosine = std::min(worst_cosine, cosine);
    total_rel_error += rel_err;
    worst_rel_error = std::max(worst_rel_error, rel_err);
  }

  float avg_cosine = total_cosine / num_samples;
  float avg_rel = total_rel_error / num_samples;

  std::cout << "\n=== Quantize-Dequantize Fidelity (v2, " << num_samples
            << " samples) ===\n"
            << "  Config: heads_KV=" << num_heads_KV << ", dim=" << head_dim
            << "\n"
            << "  avg_cosine   = " << avg_cosine << "\n"
            << "  worst_cosine = " << worst_cosine << "\n"
            << "  avg_rel_err  = " << (avg_rel * 100) << "%\n"
            << "  worst_rel_err= " << (worst_rel_error * 100) << "%\n";

  EXPECT_GT(avg_cosine, 0.95f) << "Average quant-dequant cosine too low";
  EXPECT_GT(worst_cosine, 0.85f) << "Worst-case quant-dequant cosine too low";
}

/**
 * @brief Verify TurboQuant v2 produces bit-exact results regardless of
 *        OMP_NUM_THREADS. Reproduces the reported bug where results differ
 *        between OMP_NUM_THREADS=4 and OMP_NUM_THREADS=8.
 *
 *        Tests the OMP parallel for pattern used in mha_core.cpp (lines 827,
 * 850): each OMP thread calls compute_kcaches/vcache_packed4 with a
 * single-head range.
 */
TEST(turboquant_mha_core_pipeline, omp_thread_count_determinism) {
  // Qwen3-1.7B-like config
  constexpr int num_heads_Q = 16;
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int tile_size = 4;
  constexpr int context_len = 64;
  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int packed_row_bytes = kv_width / 2;
  constexpr size_t local_window_size = 4096;

  std::mt19937 gen(99999);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Generate deterministic rot_signs
  std::vector<float> rot_signs(head_dim);
  for (int i = 0; i < head_dim; ++i)
    rot_signs[i] = (gen() % 2) ? 1.0f : -1.0f;

  // Build quantized KV cache using v2
  std::vector<uint8_t> packed_kcache(context_len * packed_row_bytes);
  std::vector<float> kcache_norms(context_len * num_heads_KV);
  std::vector<uint8_t> packed_vcache(context_len * packed_row_bytes);
  std::vector<float> vcache_norms(context_len * num_heads_KV);

  for (int row = 0; row < context_len; ++row) {
    std::vector<float> k_data(kv_width), v_data(kv_width);
    for (auto &v : k_data)
      v = dist(gen);
    for (auto &v : v_data)
      v = dist(gen);

    nntrainer::quantize_kv_turboquant(
      k_data.data(), packed_kcache.data() + row * packed_row_bytes,
      kcache_norms.data() + row * num_heads_KV, rot_signs.data(), head_dim,
      num_heads_KV);
    nntrainer::quantize_kv_turboquant(
      v_data.data(), packed_vcache.data() + row * packed_row_bytes,
      vcache_norms.data() + row * num_heads_KV, rot_signs.data(), head_dim,
      num_heads_KV);
  }

  // Query
  std::vector<float> query(num_heads_Q * head_dim);
  for (auto &v : query)
    v = dist(gen);

  // ---- Run kcache with different OMP thread counts ----
  auto run_kcache_omp = [&](int nthreads) -> std::vector<float> {
    int row_to_compute = context_len;
    std::vector<float> out(row_to_compute * num_heads_Q, 0.0f);

    omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(static)
    for (int head_kv = 0; head_kv < num_heads_KV; ++head_kv) {
      nntrainer::compute_kcaches_packed4(
        query.data(), packed_kcache.data(), kcache_norms.data(), out.data(),
        row_to_compute, num_heads_KV, head_dim, gqa_size, tile_size,
        rot_signs.data(), local_window_size, head_kv, head_kv + 1);
    }
    return out;
  };

  std::vector<float> kcache_t1 = run_kcache_omp(1);
  std::vector<float> kcache_t4 = run_kcache_omp(4);
  std::vector<float> kcache_t8 = run_kcache_omp(8);

  std::cout << "\n=== OMP Thread Count Determinism (kcache) ===" << std::endl;

  int kcache_diffs_1v4 = 0, kcache_diffs_1v8 = 0, kcache_diffs_4v8 = 0;
  float max_diff_1v4 = 0, max_diff_1v8 = 0, max_diff_4v8 = 0;
  for (size_t i = 0; i < kcache_t1.size(); ++i) {
    float d14 = std::fabs(kcache_t1[i] - kcache_t4[i]);
    float d18 = std::fabs(kcache_t1[i] - kcache_t8[i]);
    float d48 = std::fabs(kcache_t4[i] - kcache_t8[i]);
    if (d14 > 0)
      kcache_diffs_1v4++;
    if (d18 > 0)
      kcache_diffs_1v8++;
    if (d48 > 0)
      kcache_diffs_4v8++;
    max_diff_1v4 = std::max(max_diff_1v4, d14);
    max_diff_1v8 = std::max(max_diff_1v8, d18);
    max_diff_4v8 = std::max(max_diff_4v8, d48);
  }

  std::cout << "  1 vs 4 threads: " << kcache_diffs_1v4
            << " diffs, max=" << max_diff_1v4 << std::endl;
  std::cout << "  1 vs 8 threads: " << kcache_diffs_1v8
            << " diffs, max=" << max_diff_1v8 << std::endl;
  std::cout << "  4 vs 8 threads: " << kcache_diffs_4v8
            << " diffs, max=" << max_diff_4v8 << std::endl;

  EXPECT_EQ(kcache_diffs_1v4, 0)
    << "kcache results differ between 1 and 4 threads!";
  EXPECT_EQ(kcache_diffs_1v8, 0)
    << "kcache results differ between 1 and 8 threads!";
  EXPECT_EQ(kcache_diffs_4v8, 0)
    << "kcache results differ between 4 and 8 threads!";

  // ---- Softmax (serial, deterministic) ----
  auto softmax = [&](std::vector<float> &attn, int num_rows) {
    for (int h = 0; h < num_heads_Q; ++h) {
      float max_val = -1e30f;
      for (int r = 0; r < num_rows; ++r)
        max_val = std::max(max_val, attn[r * num_heads_Q + h]);
      float sum_exp = 0.0f;
      for (int r = 0; r < num_rows; ++r) {
        attn[r * num_heads_Q + h] =
          std::exp(attn[r * num_heads_Q + h] - max_val);
        sum_exp += attn[r * num_heads_Q + h];
      }
      for (int r = 0; r < num_rows; ++r)
        attn[r * num_heads_Q + h] /= sum_exp;
    }
  };

  // Use kcache_t1 for softmax (deterministic baseline)
  softmax(kcache_t1, context_len);

  // ---- Run vcache with different OMP thread counts ----
  auto run_vcache_omp =
    [&](int nthreads, const std::vector<float> &attn) -> std::vector<float> {
    int out_dim = num_heads_KV * gqa_size * head_dim;
    std::vector<float> out(out_dim, 0.0f);
    int row_num = context_len - 1;

    omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(static)
    for (int head_kv = 0; head_kv < num_heads_KV; ++head_kv) {
      nntrainer::compute_vcache_packed4(
        row_num, attn.data(), packed_vcache.data(), vcache_norms.data(),
        out.data(), num_heads_KV, gqa_size, head_dim, rot_signs.data(),
        local_window_size, head_kv, head_kv + 1);
    }
    return out;
  };

  std::vector<float> vcache_t1 = run_vcache_omp(1, kcache_t1);
  std::vector<float> vcache_t4 = run_vcache_omp(4, kcache_t1);
  std::vector<float> vcache_t8 = run_vcache_omp(8, kcache_t1);

  std::cout << "\n=== OMP Thread Count Determinism (vcache) ===" << std::endl;

  int vcache_diffs_1v4 = 0, vcache_diffs_1v8 = 0, vcache_diffs_4v8 = 0;
  float vmax_14 = 0, vmax_18 = 0, vmax_48 = 0;
  for (size_t i = 0; i < vcache_t1.size(); ++i) {
    float d14 = std::fabs(vcache_t1[i] - vcache_t4[i]);
    float d18 = std::fabs(vcache_t1[i] - vcache_t8[i]);
    float d48 = std::fabs(vcache_t4[i] - vcache_t8[i]);
    if (d14 > 0)
      vcache_diffs_1v4++;
    if (d18 > 0)
      vcache_diffs_1v8++;
    if (d48 > 0)
      vcache_diffs_4v8++;
    vmax_14 = std::max(vmax_14, d14);
    vmax_18 = std::max(vmax_18, d18);
    vmax_48 = std::max(vmax_48, d48);
  }

  std::cout << "  1 vs 4 threads: " << vcache_diffs_1v4
            << " diffs, max=" << vmax_14 << std::endl;
  std::cout << "  1 vs 8 threads: " << vcache_diffs_1v8
            << " diffs, max=" << vmax_18 << std::endl;
  std::cout << "  4 vs 8 threads: " << vcache_diffs_4v8
            << " diffs, max=" << vmax_48 << std::endl;

  EXPECT_EQ(vcache_diffs_1v4, 0)
    << "vcache results differ between 1 and 4 threads!";
  EXPECT_EQ(vcache_diffs_1v8, 0)
    << "vcache results differ between 1 and 8 threads!";
  EXPECT_EQ(vcache_diffs_4v8, 0)
    << "vcache results differ between 4 and 8 threads!";

  // Restore default
  omp_set_num_threads(omp_get_max_threads());
}

/**
 * @brief Demonstrate that OpenBLAS sgemv produces different results with
 *        different thread counts, and show that these differences get amplified
 *        by 4-bit TurboQuant quantization.
 *
 *        This is the root cause of the reported OMP_NUM_THREADS sensitivity:
 *        OpenBLAS (openblas-pthread) uses OMP_NUM_THREADS as fallback for its
 *        internal thread count. Different thread counts → different FP
 *        reduction orders → different sgemv results → different quantized
 *        outputs.
 */
TEST(turboquant_mha_core_pipeline, openblas_quantization_amplification) {
#ifdef USE_BLAS
  constexpr int hidden = 2048; // Qwen3-1.7B hidden size
  constexpr int head_dim = 128;
  constexpr int num_heads_KV = 4;
  constexpr int kv_width = num_heads_KV * head_dim;

  std::mt19937 gen(77777);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  // Simulate FP32 weight matrix (hidden × kv_width) for K projection
  std::vector<float> weight(hidden * kv_width);
  for (auto &w : weight)
    w = dist(gen);

  // Simulate FP32 input (1 × hidden)
  std::vector<float> input(hidden);
  for (auto &v : input)
    v = dist(gen);

  // rot_signs for TurboQuant
  std::vector<float> rot_signs(head_dim);
  for (int i = 0; i < head_dim; ++i)
    rot_signs[i] = (gen() % 2) ? 1.0f : -1.0f;

  auto run_sgemv_then_quantize = [&](int blas_threads) {
    openblas_set_num_threads(blas_threads);

    // sgemv: output = input * weight^T → (1 × kv_width)
    std::vector<float> kv_out(kv_width, 0.0f);
    nntrainer::sgemv(0 /* RowMajor */, false, kv_width, hidden, 1.0f,
                     weight.data(), hidden, input.data(), 1, 0.0f,
                     kv_out.data(), 1);

    // Quantize with TurboQuant v2
    std::vector<uint8_t> packed(kv_width / 2);
    std::vector<float> norms(num_heads_KV);
    nntrainer::quantize_kv_turboquant(kv_out.data(), packed.data(),
                                      norms.data(), rot_signs.data(), head_dim,
                                      num_heads_KV);

    return std::make_tuple(kv_out, packed, norms);
  };

  auto [kv_1t, packed_1t, norms_1t] = run_sgemv_then_quantize(1);
  auto [kv_4t, packed_4t, norms_4t] = run_sgemv_then_quantize(4);
  auto [kv_8t, packed_8t, norms_8t] = run_sgemv_then_quantize(8);

  // Check sgemv result differences
  int sgemv_diffs_1v4 = 0, sgemv_diffs_1v8 = 0;
  float sgemv_maxdiff_1v4 = 0, sgemv_maxdiff_1v8 = 0;
  for (int i = 0; i < kv_width; ++i) {
    float d14 = std::fabs(kv_1t[i] - kv_4t[i]);
    float d18 = std::fabs(kv_1t[i] - kv_8t[i]);
    if (d14 > 0)
      sgemv_diffs_1v4++;
    if (d18 > 0)
      sgemv_diffs_1v8++;
    sgemv_maxdiff_1v4 = std::max(sgemv_maxdiff_1v4, d14);
    sgemv_maxdiff_1v8 = std::max(sgemv_maxdiff_1v8, d18);
  }

  std::cout << "\n=== OpenBLAS sgemv Non-Determinism ===" << std::endl;
  std::cout << "  1 vs 4 threads: " << sgemv_diffs_1v4
            << " diffs, max=" << sgemv_maxdiff_1v4 << std::endl;
  std::cout << "  1 vs 8 threads: " << sgemv_diffs_1v8
            << " diffs, max=" << sgemv_maxdiff_1v8 << std::endl;

  // Check quantized result differences (amplification)
  int quant_diffs_1v4 = 0, quant_diffs_1v8 = 0;
  for (size_t i = 0; i < packed_1t.size(); ++i) {
    if (packed_1t[i] != packed_4t[i])
      quant_diffs_1v4++;
    if (packed_1t[i] != packed_8t[i])
      quant_diffs_1v8++;
  }

  float norm_diffs_1v4 = 0, norm_diffs_1v8 = 0;
  for (int i = 0; i < num_heads_KV; ++i) {
    norm_diffs_1v4 += std::fabs(norms_1t[i] - norms_4t[i]);
    norm_diffs_1v8 += std::fabs(norms_1t[i] - norms_8t[i]);
  }

  std::cout << "\n=== Quantization Amplification ===" << std::endl;
  std::cout << "  Packed bytes differ (1v4): " << quant_diffs_1v4 << " / "
            << packed_1t.size() << std::endl;
  std::cout << "  Packed bytes differ (1v8): " << quant_diffs_1v8 << " / "
            << packed_1t.size() << std::endl;
  std::cout << "  Norm total abs diff (1v4): " << norm_diffs_1v4 << std::endl;
  std::cout << "  Norm total abs diff (1v8): " << norm_diffs_1v8 << std::endl;

  // Also test: does sgemm produce different results?
  // (used for prefill with seq_len > 1)
  constexpr int seq_len = 4;
  std::vector<float> multi_input(seq_len * hidden);
  for (auto &v : multi_input)
    v = dist(gen);

  auto run_sgemm = [&](int blas_threads) {
    openblas_set_num_threads(blas_threads);
    std::vector<float> out(seq_len * kv_width, 0.0f);
    nntrainer::sgemm(0 /* RowMajor */, false, true, seq_len, kv_width, hidden,
                     1.0f, multi_input.data(), hidden, weight.data(), hidden,
                     0.0f, out.data(), kv_width);
    return out;
  };

  auto gemm_1t = run_sgemm(1);
  auto gemm_4t = run_sgemm(4);
  auto gemm_8t = run_sgemm(8);

  int gemm_diffs_1v4 = 0, gemm_diffs_1v8 = 0;
  float gemm_maxdiff_1v4 = 0, gemm_maxdiff_1v8 = 0;
  for (size_t i = 0; i < gemm_1t.size(); ++i) {
    float d14 = std::fabs(gemm_1t[i] - gemm_4t[i]);
    float d18 = std::fabs(gemm_1t[i] - gemm_8t[i]);
    if (d14 > 0)
      gemm_diffs_1v4++;
    if (d18 > 0)
      gemm_diffs_1v8++;
    gemm_maxdiff_1v4 = std::max(gemm_maxdiff_1v4, d14);
    gemm_maxdiff_1v8 = std::max(gemm_maxdiff_1v8, d18);
  }

  std::cout << "\n=== OpenBLAS sgemm Non-Determinism ===" << std::endl;
  std::cout << "  1 vs 4 threads: " << gemm_diffs_1v4
            << " diffs, max=" << gemm_maxdiff_1v4 << std::endl;
  std::cout << "  1 vs 8 threads: " << gemm_diffs_1v8
            << " diffs, max=" << gemm_maxdiff_1v8 << std::endl;

  if (sgemv_diffs_1v4 > 0 || sgemv_diffs_1v8 > 0 || gemm_diffs_1v4 > 0 ||
      gemm_diffs_1v8 > 0) {
    std::cout << "\n  >>> ROOT CAUSE CONFIRMED: OpenBLAS sgemv produces "
                 "different results\n"
              << "      with different thread counts. 4-bit quantization "
                 "amplifies these\n"
              << "      tiny FP differences into different quantized "
                 "representations.\n"
              << "  >>> FIX: Set OPENBLAS_NUM_THREADS=<fixed> to decouple from "
                 "OMP_NUM_THREADS,\n"
              << "      or call openblas_set_num_threads(N) at init.\n";
  } else {
    std::cout << "\n  >>> OpenBLAS sgemv is deterministic on this platform "
                 "(single-threaded BLAS?)\n";
  }

  // Restore
  openblas_set_num_threads(1);
#else
  std::cout << "  [SKIPPED] No BLAS support compiled" << std::endl;
#endif
}

GTEST_API_ int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
