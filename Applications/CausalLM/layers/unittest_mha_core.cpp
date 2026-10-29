// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_mha_core.cpp
 * @date   29 October 2026
 * @brief  Characterization tests for causallm::MHACoreLayer on the host path
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * @details Runs MHACoreLayer in its external-KV-cache mode (the mode every
 * CausalLM model uses) over a prefill followed by decode steps, and compares
 * each step's output with a plain reference written from the attention formula:
 *
 *   out = softmax(rope(Q) . rope(K)^T / sqrt(head_dim) + mask) . V
 *
 * The reference rounds K and V to fp16 the way the layer's KV cache stores
 * them, so the two agree to fp32 rounding. These tests pin the current
 * host-path behaviour before the backend dispatch inside the layer is moved
 * to ComputeOps (#4451); a refactor of the layer must keep them passing.
 */

#include <mha_core.h>

#include <fp16.h>
#include <layer_context.h>
#include <var_grad.h>
#include <weight.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

/** @brief One configuration of the layer and the steps fed to it. */
struct MhaCase {
  std::string name;
  unsigned int heads_q;
  unsigned int heads_kv;
  unsigned int head_dim;
  /** rows per incremental_forwarding call: prefill chunks, then 1s */
  std::vector<unsigned int> steps;
  bool causal = true;
  bool gemm = false;              /**< use_gemm_attention */
  unsigned int window = UINT_MAX; /**< sliding_window */
  unsigned int theta = 10000;     /**< rope_theta; 0 disables RoPE */
  float softcap = 0.0f;           /**< attn_logit_softcapping */
  bool sink = false;              /**< use_sink */
  std::string rope_type = "default";
  float rope_factor = 1.0f;        /**< rope_scaling_factor */
  float partial_rotary = 1.0f;     /**< rope_partial_rotary_factor */
  unsigned int rope_orig_max = 64; /**< rope_scaling_max_position_embeddings */
  /**
   * Whether the layer applies softcap on this path. The host flash arm
   * (gemm_attention) does not: it ignores attn_logit_softcapping, while the
   * per-row arm and the OpenCL arms apply it. Pinned as-is; a fix flips it.
   */
  bool softcap_applied = true;
};

std::ostream &operator<<(std::ostream &os, const MhaCase &c) {
  return os << c.name;
}

constexpr float kPi = 3.14159265358979323846f;

/** @brief Round a value through fp16 storage, as the KV cache does. */
float roundFp16(float v) {
  return nntrainer::compute_fp16_to_fp32(nntrainer::compute_fp32_to_fp16(v));
}

/**
 * @brief RoPE inverse frequencies and cos/sin scale, written from the
 *        formulas the layer documents for each rope_scaling_type.
 */
struct RopeTable {
  std::vector<float> inv_freq; /**< head_dim / 2 entries */
  float scaling = 1.0f;        /**< multiplies cos and sin */
};

RopeTable makeRopeTable(const MhaCase &c) {
  const unsigned int half = c.head_dim / 2;
  const float d = static_cast<float>(c.head_dim);
  RopeTable t;
  t.inv_freq.resize(half);

  if (c.rope_type == "yarn") {
    // YaRN with beta_fast = 32, beta_slow = 1, no truncation of the ramp
    // bounds and attention scaling 0.1 * ln(factor) + 1.
    const float s = c.rope_factor;
    t.scaling = s <= 1.0f ? 1.0f : 0.1f * std::log(s) + 1.0f;
    auto correction_dim = [&](float rotations) {
      return (d * std::log(c.rope_orig_max / (rotations * 2 * kPi))) /
             (2 * std::log(static_cast<float>(c.theta)));
    };
    float low = correction_dim(32.0f);
    float high = correction_dim(1.0f);
    if (low == high)
      high += 0.001f;
    for (unsigned int i = 0; i < half; ++i) {
      const float extra = 1.0f / std::pow(c.theta, 2 * i / d);
      const float inter = extra / s;
      const float ramp = std::clamp((i - low) / (high - low), 0.0f, 1.0f);
      const float keep_extra = 1.0f - ramp;
      t.inv_freq[i] = extra * keep_extra + inter * (1.0f - keep_extra);
    }
    return t;
  }

  if (c.rope_type == "proportional" || c.partial_rotary != 1.0f) {
    // Only the first partial_rotary * head_dim / 2 pairs rotate; every
    // frequency is divided by rope_scaling_factor.
    const unsigned int rotated =
      static_cast<unsigned int>(c.partial_rotary * c.head_dim / 2.0f);
    for (unsigned int i = 0; i < half; ++i)
      t.inv_freq[i] =
        (i < rotated ? 1.0f / std::pow(c.theta, 2 * i / d) : 0.0f) /
        c.rope_factor;
    return t;
  }

  for (unsigned int i = 0; i < half; ++i)
    t.inv_freq[i] = 1.0f / std::pow(c.theta, 2 * i / d);
  return t;
}

/** @brief Rotate one head (rotate-half layout) to position @a pos. */
void ropeHead(const RopeTable &t, float *x, unsigned int head_dim,
              unsigned int pos) {
  const unsigned int half = head_dim / 2;
  for (unsigned int i = 0; i < half; ++i) {
    const float angle = pos * t.inv_freq[i];
    const float cs = std::cos(angle) * t.scaling;
    const float sn = std::sin(angle) * t.scaling;
    const float a = x[i];
    const float b = x[i + half];
    x[i] = a * cs - b * sn;
    x[i + half] = a * sn + b * cs;
  }
}

/**
 * @brief Plain attention over the whole history, one query row at a time.
 *
 * @param q   rows x (heads_q * head_dim), query rows for positions
 *            [from, from + rows)
 * @param k,v total x (heads_kv * head_dim), every key/value written so far
 * @param sink per-head sink logits, or empty
 */
std::vector<float> referenceAttention(const MhaCase &c, const RopeTable &rope,
                                      std::vector<float> q,
                                      const std::vector<float> &k,
                                      const std::vector<float> &v,
                                      const std::vector<float> &sink,
                                      unsigned int from, unsigned int rows) {
  const unsigned int D = c.head_dim;
  const unsigned int gqa = c.heads_q / c.heads_kv;
  const unsigned int total = from + rows;
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
  std::vector<float> out(rows * c.heads_q * D, 0.0f);

  for (unsigned int r = 0; r < rows; ++r) {
    const unsigned int pos = from + r;
    for (unsigned int h = 0; h < c.heads_q; ++h) {
      float *qh = q.data() + (r * c.heads_q + h) * D;
      if (c.theta > 0)
        ropeHead(rope, qh, D, pos);
      const unsigned int kh = h / gqa;

      // Visible keys: causal keeps j <= pos, the window keeps pos - j < W.
      unsigned int last = c.causal ? pos : total - 1;
      unsigned int first = 0;
      if (c.causal && c.window != UINT_MAX && pos + 1 > c.window)
        first = pos + 1 - c.window;

      std::vector<float> score;
      for (unsigned int j = first; j <= last; ++j) {
        const float *kj = k.data() + (j * c.heads_kv + kh) * D;
        float s = 0.0f;
        for (unsigned int i = 0; i < D; ++i)
          s += qh[i] * kj[i];
        s *= inv_sqrt_d;
        if (c.softcap > 0.0f && c.softcap_applied)
          s = std::tanh(s / c.softcap) * c.softcap;
        score.push_back(s);
      }

      float mx = *std::max_element(score.begin(), score.end());
      if (!sink.empty())
        mx = std::max(mx, sink[h]);
      float denom = sink.empty() ? 0.0f : std::exp(sink[h] - mx);
      for (float &s : score) {
        s = std::exp(s - mx);
        denom += s;
      }

      float *oh = out.data() + (r * c.heads_q + h) * D;
      for (unsigned int j = first; j <= last; ++j) {
        const float p = score[j - first] / denom;
        const float *vj = v.data() + (j * c.heads_kv + kh) * D;
        for (unsigned int i = 0; i < D; ++i)
          oh[i] += p * vj[i];
      }
    }
  }
  return out;
}

template <typename T> std::vector<T *> pointers(std::vector<T> &items) {
  std::vector<T *> view;
  for (auto &item : items)
    view.push_back(&item);
  return view;
}

nntrainer::TensorDim dim4(unsigned int h, unsigned int w,
                          nntrainer::TensorDim::DataType dtype) {
  return nntrainer::TensorDim({1, 1, h, w, nntrainer::Tformat::NCHW, dtype});
}

class MhaCoreHostPath : public ::testing::TestWithParam<MhaCase> {};

/**
 * Feeds every step of the case through incremental_forwarding and checks
 * each step's output rows against referenceAttention().
 */
TEST_P(MhaCoreHostPath, MatchesReferenceAttention) {
  const MhaCase &c = GetParam();
  const unsigned int D = c.head_dim;
  const unsigned int q_width = c.heads_q * D;
  const unsigned int kv_width = c.heads_kv * D;
  const unsigned int plane = *std::max_element(c.steps.begin(), c.steps.end());
  const unsigned int total =
    std::accumulate(c.steps.begin(), c.steps.end(), 0u);
  const unsigned int max_seq = total + 4;

  causallm::MHACoreLayer layer;
  std::vector<std::string> props = {
    "num_heads=" + std::to_string(c.heads_q),
    "num_heads_kv=" + std::to_string(c.heads_kv),
    "max_timestep=" + std::to_string(max_seq),
    "init_seq_len=" + std::to_string(plane),
    "max_position_embeddings=" + std::to_string(max_seq),
    "rope_theta=" + std::to_string(c.theta),
    "is_causal=" + std::string(c.causal ? "true" : "false"),
    "use_gemm_attention=" + std::string(c.gemm ? "true" : "false"),
    "use_sink=" + std::string(c.sink ? "true" : "false"),
    "rope_scaling_type=" + c.rope_type,
    "rope_scaling_factor=" + std::to_string(c.rope_factor),
    "rope_partial_rotary_factor=" + std::to_string(c.partial_rotary),
    "rope_scaling_max_position_embeddings=" + std::to_string(c.rope_orig_max)};
  if (c.window != UINT_MAX)
    props.push_back("sliding_window=" + std::to_string(c.window));
  if (c.softcap > 0.0f)
    props.push_back("attn_logit_softcapping=" + std::to_string(c.softcap));
  layer.setProperty(props);

  // Inputs: query, key, value (one plane of rows each), then the external
  // key/value caches. Without enable-fp16 the caches hold fp16 bits as UINT16,
  // as Transformer::createKVCachePlaceholders declares them.
  const auto fp32 = nntrainer::TensorDim::DataType::FP32;
#ifdef ENABLE_FP16
  const auto cache_dtype = nntrainer::TensorDim::DataType::FP16;
#else
  const auto cache_dtype = nntrainer::TensorDim::DataType::UINT16;
#endif
  nntrainer::InitLayerContext init_context(
    {dim4(plane, q_width, fp32), dim4(plane, kv_width, fp32),
     dim4(plane, kv_width, fp32), dim4(max_seq, kv_width, cache_dtype),
     dim4(max_seq, kv_width, cache_dtype)},
    {true}, false, "mha_core", "", 0.0f, {"NCHW", "FP32", "FP32"}, 1.0f,
    ml::train::ExecutionMode::INFERENCE);
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  for (const auto &spec : init_context.getWeightsSpec())
    weights.emplace_back(spec, true);
  std::vector<nntrainer::Var_Grad> tensors;
  for (const auto &spec : init_context.getTensorsSpec())
    tensors.emplace_back(spec, true);
  std::vector<nntrainer::Var_Grad> inputs;
  for (const auto &d : init_context.getInputDimensions())
    inputs.emplace_back(d, nntrainer::Initializer::ZEROS, false, true, "in");
  std::vector<nntrainer::Var_Grad> outputs;
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::ZEROS, false, true, "out");

  std::mt19937 rng(0x5eed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> sink;
  if (c.sink) {
    ASSERT_EQ(weights.size(), 1u);
    float *w = weights[0].getVariableRef().getData<float>();
    for (unsigned int h = 0; h < c.heads_q; ++h)
      sink.push_back(w[h] = dist(rng));
  }

  nntrainer::RunLayerContext run_context(
    "mha_core", false, 0.0f, false, 1.0f, nullptr, false, pointers(weights),
    pointers(inputs), pointers(outputs), pointers(tensors));

  const RopeTable rope = makeRopeTable(c);
  std::vector<float> k_hist; // rotated, fp16-rounded keys of every position
  std::vector<float> v_hist; // fp16-rounded values of every position

  unsigned int from = 0;
  for (unsigned int rows : c.steps) {
    std::vector<float> q(rows * q_width), k(rows * kv_width),
      v(rows * kv_width);
    for (auto &x : q)
      x = dist(rng);
    for (auto &x : k)
      x = dist(rng);
    for (auto &x : v)
      x = dist(rng);

    std::copy(q.begin(), q.end(), run_context.getInput(0).getData<float>());
    std::copy(k.begin(), k.end(), run_context.getInput(1).getData<float>());
    std::copy(v.begin(), v.end(), run_context.getInput(2).getData<float>());

    for (unsigned int r = 0; r < rows; ++r) {
      for (unsigned int h = 0; h < c.heads_kv; ++h) {
        float *kh = k.data() + (r * c.heads_kv + h) * D;
        if (c.theta > 0)
          ropeHead(rope, kh, D, from + r);
      }
    }
    for (float x : k)
      k_hist.push_back(roundFp16(x));
    for (float x : v)
      v_hist.push_back(roundFp16(x));

    layer.incremental_forwarding(run_context, from, from + rows, false);

    const std::vector<float> expected =
      referenceAttention(c, rope, q, k_hist, v_hist, sink, from, rows);
    const float *actual = run_context.getOutput(0).getData<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_NEAR(actual[i], expected[i], 1e-4f)
        << "step [" << from << ", " << from + rows << ") element " << i;
    }
    from += rows;
  }
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
  CausalLm, MhaCoreHostPath,
  ::testing::Values(
    MhaCase{"mha_rowwise", 2, 2, 16, {5, 1, 1, 1}},
    MhaCase{"mha_gemm", 2, 2, 16, {5, 1, 1, 1}, true, true},
    MhaCase{"gqa_rowwise", 4, 2, 16, {5, 1, 1, 1}},
    MhaCase{"gqa_gemm", 4, 2, 16, {5, 1, 1, 1}, true, true},
    MhaCase{"chunked_prefill_rowwise", 4, 2, 16, {4, 4, 1, 1}},
    MhaCase{"chunked_prefill_gemm", 4, 2, 16, {4, 4, 1, 1}, true, true},
    MhaCase{"window_rowwise", 4, 2, 16, {6, 1, 1, 1}, true, false, 3},
    MhaCase{"window_gemm", 4, 2, 16, {6, 1, 1, 1}, true, true, 3},
    MhaCase{"no_rope_rowwise", 2, 2, 16, {5, 1, 1}, true, false, UINT_MAX, 0},
    MhaCase{"no_rope_gemm", 2, 2, 16, {5, 1, 1}, true, true, UINT_MAX, 0},
    MhaCase{"softcap_rowwise", 2, 2, 16, {5, 1, 1}, true, false, UINT_MAX,
            10000, 0.5f},
    MhaCase{"softcap_gemm_not_applied", 2, 2, 16, {5, 1, 1}, true, true,
            UINT_MAX, 10000, 0.5f, false, "default", 1.0f, 1.0f, 64, false},
    MhaCase{"encoder_rowwise", 4, 2, 16, {6}, false, false},
    MhaCase{"encoder_gemm", 4, 2, 16, {6}, false, true},
    MhaCase{"sink", 4, 2, 16, {5, 1, 1}, true, false, UINT_MAX, 10000, 0.0f,
            true},
    MhaCase{"yarn", 2, 2, 16, {5, 1, 1}, true, false, UINT_MAX, 10000, 0.0f,
            false, "yarn", 4.0f},
    MhaCase{"proportional", 2, 2, 16, {5, 1, 1}, true, false, UINT_MAX, 10000,
            0.0f, false, "proportional", 2.0f, 0.5f}),
  [](const ::testing::TestParamInfo<MhaCase> &info) {
    return info.param.name;
  });
// clang-format on

} // namespace
