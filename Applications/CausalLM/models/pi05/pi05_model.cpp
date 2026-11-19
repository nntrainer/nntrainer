// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   pi05_model.cpp
 * @brief  pi0.5 flow-matching model: implementation.
 * @date   12 Nov 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "pi05_model.h"

#include "llm_util.hpp"
#include "paligemma/paligemma_causallm.h"

#include <cpu_backend.h>
#include <fp16.h>
#include <nntrainer_error.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread_manager.h>

namespace causallm {
namespace {
/** Row reductions retain their FP32 order; independent action rows share the
 * operator pool. Modulation is broadcast, without a horizon-sized gate copy. */
void adaptiveNormRows(const float *input, const float *modulation,
                      float *output, unsigned int rows, unsigned int width,
                      float eps) {
  const auto normalize = [&](size_t row) {
    const float *x = input + row * width;
    float *y = output + row * width;
    float ssq = 0.0f;
    for (unsigned int i = 0; i < width; ++i)
      ssq += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ssq / width + eps);
    for (unsigned int i = 0; i < width; ++i)
      y[i] = x[i] * inv * (1.0f + modulation[i]) + modulation[width + i];
  };
  if (rows >= 8)
    nntrainer::ThreadManager::Global().parallel_for(0, rows, normalize);
  else
    for (unsigned int row = 0; row < rows; ++row)
      normalize(row);
}
void gatedResidualRows(const float *input, const float *branch,
                       const float *gate, float *output, unsigned int rows,
                       unsigned int width) {
  const auto add = [&](size_t row) {
    const size_t offset = row * width;
    for (unsigned int i = 0; i < width; ++i)
      output[offset + i] = input[offset + i] + branch[offset + i] * gate[i];
  };
  if (rows >= 8)
    nntrainer::ThreadManager::Global().parallel_for(0, rows, add);
  else
    for (unsigned int row = 0; row < rows; ++row)
      add(row);
}
} // namespace

void PI05Model::dumpDiagnostic(const std::string &name,
                               const std::vector<float> &values) {
  if (const char *dir = std::getenv("NNTR_PI05_DIAGNOSTICS")) {
    std::ofstream f(std::string(dir) + "/" + name + ".bin", std::ios::binary);
    f.write(reinterpret_cast<const char *>(values.data()),
            values.size() * sizeof(float));
    if (!f)
      throw std::runtime_error("failed writing pi05 diagnostic " + name);
  }
}

json &PI05Model::sanitizeConfig(json &cfg) {
  // Establish the VLA allocation policy before the virtual Transformer base
  // initializes CUDA, including callers that construct PI05 without vision.
  if (causallm_engine() == "cuda")
    configurePaliGemmaCuda();
  // pi0.5 ships one HF config with paligemma text_config + vision_config.
  // Merge text_config up (top level wins) — the backbone ctor re-sanitizes.
  if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
    for (auto it = cfg["text_config"].begin(); it != cfg["text_config"].end();
         ++it)
      if (!cfg.contains(it.key()))
        cfg[it.key()] = it.value();
  }

  // backbone defaults (paligemma-3b-pt-224 gemma1 text)
  if (!cfg.contains("rms_norm_eps"))
    cfg["rms_norm_eps"] = 1e-6;
  if (!cfg.contains("rope_theta"))
    cfg["rope_theta"] = 10000;
  if (!cfg.contains("tie_word_embeddings"))
    cfg["tie_word_embeddings"] = true;
  if (!cfg.contains("max_position_embeddings"))
    cfg["max_position_embeddings"] = 8192;

  // expert / flow defaults (openpi gemma_300m + pi05 flow head)
  if (!cfg.contains("expert_hidden_size"))
    cfg["expert_hidden_size"] = 1024;
  if (!cfg.contains("expert_intermediate_size"))
    cfg["expert_intermediate_size"] = 4096;
  if (!cfg.contains("expert_head_dim"))
    cfg["expert_head_dim"] = 256;
  if (!cfg.contains("expert_num_attention_heads"))
    cfg["expert_num_attention_heads"] = 8;
  if (!cfg.contains("expert_num_key_value_heads"))
    cfg["expert_num_key_value_heads"] = 1;
  if (!cfg.contains("action_dim"))
    cfg["action_dim"] = 32;
  if (!cfg.contains("action_horizon"))
    cfg["action_horizon"] = 50;
  if (!cfg.contains("num_flow_steps"))
    cfg["num_flow_steps"] = 10;
  if (!cfg.contains("time_min_period"))
    cfg["time_min_period"] = 4e-3;
  if (!cfg.contains("time_max_period"))
    cfg["time_max_period"] = 4.0;
  return cfg;
}

json &PI05Model::sanitizeGenerationConfig(json &gen_cfg, const json &cfg) {
  (void)cfg;
  if (!gen_cfg.contains("do_sample"))
    gen_cfg["do_sample"] = false;
  return gen_cfg;
}

PI05Model::PI05Model(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(sanitizeConfig(cfg),
              sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg,
              ModelType::CAUSALLM) {
  // The statics mutate the jsons IN PLACE before the base ctor binds them
  // (gemma3 pattern) — the base setupParameters therefore sees the
  // flattened text_config. Keep copies for the backbone producer and the
  // pi0.5-specific keys.
  sanitized_cfg_ = cfg;
  gen_cfg_ = generation_cfg;
  nntr_cfg_ = nntr_cfg;
  expert_file_name_ = nntr_cfg.contains("expert_file_name")
                        ? nntr_cfg["expert_file_name"].get<std::string>()
                        : "";

  // pi0.5 does not run the text-generation path; no tokenizer is needed.
  tokenizer = nullptr;

  EXPERT_DIM = sanitized_cfg_["expert_hidden_size"].get<unsigned int>();
  EXPERT_MLP_DIM =
    sanitized_cfg_["expert_intermediate_size"].get<unsigned int>();
  EXPERT_HEADS =
    sanitized_cfg_["expert_num_attention_heads"].get<unsigned int>();
  EXPERT_KV_HEADS =
    sanitized_cfg_["expert_num_key_value_heads"].get<unsigned int>();
  EXPERT_HEAD_DIM = sanitized_cfg_["expert_head_dim"].get<unsigned int>();
  ACTION_DIM = sanitized_cfg_["action_dim"].get<unsigned int>();
  ACTION_HORIZON = sanitized_cfg_["action_horizon"].get<unsigned int>();
  // double, NOT float: 0.004f = 0.004000000189989805..., and that 4.7e-8
  // relative error shifts the highest-frequency time angle by 7.5e-5 rad —
  // enough to corrupt the adaRMS cond through the time MLP (measured).
  TIME_MIN_PERIOD = sanitized_cfg_["time_min_period"].get<double>();
  TIME_MAX_PERIOD = sanitized_cfg_["time_max_period"].get<double>();
  RMS_EPS = sanitized_cfg_["rms_norm_eps"].get<float>();
  ROPE_THETA = sanitized_cfg_["rope_theta"].get<float>();

  kv_width_ = EXPERT_KV_HEADS * EXPERT_HEAD_DIM;
  prefix_len_ = 0;
  vision_rows_ = 0;
}

std::pair<Tensor, Tensor> PI05Model::constructModel() {
  // pi0.5's compute is split between the backbone graph (owned by
  // backbone_, a real PaliGemmaCausalLM in bypass mode) and the host expert
  // loop. The base contract still wants a compiled graph from this model —
  // build the smallest legal one: one input, straight to the output.
  LayerHandle input_layer = createLayer(
    "input",
    {withKey("name", "input0"),
     withKey("input_dtype", MODEL_TENSOR_TYPE.find("-FP16") != std::string::npos
                              ? "FP16"
                              : "FP32"),
     withKey("input_shape",
             "1:" + std::to_string(INIT_SEQ_LEN) + ":" + std::to_string(DIM))});
  Tensor x = input_layer(Tensor());
  // A bare input-as-output graph is a path compile() is not exercised with
  // anywhere else; give the trivial net one identity hop.
  LayerHandle flat = createLayer("flatten", {withKey("name", "flat0")});
  Tensor y = flat(x);

  // The backbone producer: its ctor re-sanitizes, keeps USE_EMBEDDING off
  // unless nntr_config says so, and owns the embedding table + real graph.
  // The base Transformer does NOT retain the config jsons, so PI05Model
  // keeps its own copies (captured in the ctor).
  json bcfg = sanitized_cfg_;
  json bgen = gen_cfg_;
  json bnntr = nntr_cfg_;
  bnntr["prefill_only"] = true;
  backbone_ = std::make_unique<PaliGemmaCausalLM>(bcfg, bgen, bnntr);
  backbone_->initialize();

  return {x, y};
}

void PI05Model::load_weight(const std::string &weight_file) {
  // backbone weights first (bypass layout: head table last)
  if (!backbone_)
    throw std::runtime_error("pi05: constructModel() must run before "
                             "load_weight()");
  backbone_->load_weight(weight_file);

  // expert + flow-head records: separate bin in graph order
  std::ifstream f(expert_file_name_, std::ios::binary);
  NNTR_THROW_IF(!f.is_open(), std::invalid_argument)
    << "pi05: cannot open expert bin " << expert_file_name_;

  auto read_vec = [&f](size_t n) {
    std::vector<float> v(n);
    f.read(reinterpret_cast<char *>(v.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    return v;
  };

  const size_t d = EXPERT_DIM, m = EXPERT_MLP_DIM;
  const size_t qw = EXPERT_HEADS * EXPERT_HEAD_DIM;
  const size_t kw = kv_width_;
  layers_.clear();
  for (size_t li = 0; li < NUM_LAYERS; ++li) {
    ExpertLayer l;
    l.n1_w = read_vec(d * 3 * d); // adaRMS dense [d, 3d] in-major
    l.n1_b = read_vec(3 * d);
    l.qw = read_vec(d * qw); // in-major FC records
    l.kw = read_vec(d * kw);
    l.vw = read_vec(d * kw);
    l.ow = read_vec(qw * d);
    l.n2_w = read_vec(d * 3 * d);
    l.n2_b = read_vec(3 * d);
    l.gw = read_vec(d * m); // gate_proj
    l.uw = read_vec(d * m); // up_proj
    l.dw = read_vec(m * d); // down_proj
    layers_.push_back(std::move(l));
  }
  fn_w = read_vec(d * 3 * d);
  fn_b = read_vec(3 * d);
  act_in_w = read_vec(ACTION_DIM * d);
  act_in_b = read_vec(d);
  act_out_w = read_vec(d * ACTION_DIM);
  act_out_b = read_vec(ACTION_DIM);
  t_in_w = read_vec(d * d);
  t_in_b = read_vec(d);
  t_out_w = read_vec(d * d);
  t_out_b = read_vec(d);

  NNTR_THROW_IF(!f.good(), std::invalid_argument)
    << "pi05: expert bin truncated";
  expert_weights_transposed_ = false;
  const char *pack_env = std::getenv("NNTR_PI05_CPU_PACK");
  if (causallm_engine() == "cpu" && (!pack_env || pack_env[0] != '0')) {
    // Replace each private expert matrix with its output-major layout. Only
    // one matrix-sized temporary exists at a time; no second model is kept.
    auto pack = [](std::vector<float> &weight, unsigned int in,
                   unsigned int out) {
      std::vector<float> transposed(weight.size());
      nntrainer::transpose_matrix(in, out, weight.data(), out,
                                  transposed.data(), in);
      weight.swap(transposed);
    };
    for (auto &layer : layers_) {
      pack(layer.n1_w, d, 3 * d);
      pack(layer.n2_w, d, 3 * d);
      pack(layer.qw, d, qw);
      pack(layer.kw, d, kw);
      pack(layer.vw, d, kw);
      pack(layer.ow, qw, d);
      pack(layer.gw, d, m);
      pack(layer.uw, d, m);
      pack(layer.dw, m, d);
    }
    pack(fn_w, d, 3 * d);
    pack(act_in_w, ACTION_DIM, d);
    pack(act_out_w, d, ACTION_DIM);
    pack(t_in_w, d, d);
    pack(t_out_w, d, d);
    expert_weights_transposed_ = true;
  }
  expert_mlp_fused_ = false;
#ifdef USE_MKL
  const char *fused_env = std::getenv("NNTR_PI05_CPU_FUSED_MLP");
  if (expert_weights_transposed_ && (!fused_env || fused_env[0] != '0')) {
    for (auto &layer : layers_) {
      layer.qw.reserve(layer.qw.size() + layer.kw.size() + layer.vw.size());
      layer.qw.insert(layer.qw.end(), layer.kw.begin(), layer.kw.end());
      layer.qw.insert(layer.qw.end(), layer.vw.begin(), layer.vw.end());
      std::vector<float>().swap(layer.kw);
      std::vector<float>().swap(layer.vw);
      layer.gw.insert(layer.gw.end(), layer.uw.begin(), layer.uw.end());
      std::vector<float>().swap(layer.uw);
    }
    expert_mlp_fused_ = true;
  }
#endif
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  cuda_expert_.reset();
#endif
  expert_loaded_ = true;
  modulation_steps_ = 0;
  time_conditions_.clear();
  modulation_cache_.clear();
  const unsigned int default_steps =
    sanitized_cfg_.value("num_flow_steps", 10u);
  if (default_steps > 0 && default_steps <= 1000)
    prepareModulations(default_steps);
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  if (causallm_engine() == "cuda")
    loadCudaExpertWeights();
#endif
}

void PI05Model::repack_weight() {
  if (backbone_)
    backbone_->repack_weight();
}

void PI05Model::matvec(const std::vector<float> &w, const std::vector<float> &x,
                       std::vector<float> &y, unsigned int in_dim,
                       unsigned int out_dim) const {
  matmat(w, x, y, 1, in_dim, out_dim);
}

void PI05Model::matmat(const std::vector<float> &w, const std::vector<float> &x,
                       std::vector<float> &y, unsigned int rows,
                       unsigned int in_dim, unsigned int out_dim) const {
  y.resize(static_cast<size_t>(rows) * out_dim);
  nntrainer::sgemm(
    static_cast<unsigned int>(nntrainer::TStorageOrder::ROW_MAJOR), false,
    expert_weights_transposed_, rows, out_dim, in_dim, 1.0f, x.data(), in_dim,
    w.data(), expert_weights_transposed_ ? in_dim : out_dim, 0.0f, y.data(),
    out_dim);
}

std::vector<float> PI05Model::timeCondition(float t) const {
  const unsigned int half = EXPERT_DIM / 2;
  std::vector<float> emb(EXPERT_DIM);
  // openpi computes the whole embedding in float64 (torch.linspace float64
  // + float64 sin/cos, cast to float32 only at the end). At the small-i end
  // the angle reaches thousands of radians, where float32 rounding swings
  // sin/cos by ~0.1 — enough to corrupt the adaRMS cond through the MLP.
  for (unsigned int i = 0; i < half; ++i) {
    // torch.linspace(0, 1, half): spacing 1/(half-1), LAST point exactly 1
    const double fraction =
      (half > 1) ? static_cast<double>(i) / (half - 1) : 0.0;
    const double period = static_cast<double>(TIME_MIN_PERIOD) *
                          std::pow(static_cast<double>(TIME_MAX_PERIOD) /
                                     static_cast<double>(TIME_MIN_PERIOD),
                                   fraction);
    const double angle =
      (2.0 * std::acos(-1.0) / period) * static_cast<double>(t);
    emb[i] = static_cast<float>(std::sin(angle));
    emb[half + i] = static_cast<float>(std::cos(angle));
  }

  // time MLP: in -> silu -> out -> silu (openpi time_mlp_func). The matvec
  // is bias-free — add the loaded biases explicitly (the tiny fixture's
  // zero biases hid this once; 3B has nonzero ones).
  std::vector<float> h, out;
  matvec(t_in_w, emb, h, EXPERT_DIM, EXPERT_DIM);
  for (unsigned int i = 0; i < EXPERT_DIM; ++i) {
    h[i] += t_in_b[i];
    h[i] = h[i] / (1.0f + std::exp(-h[i]));
  }
  matvec(t_out_w, h, out, EXPERT_DIM, EXPERT_DIM);
  for (unsigned int i = 0; i < EXPERT_DIM; ++i) {
    out[i] += t_out_b[i];
    out[i] = out[i] / (1.0f + std::exp(-out[i]));
  }
  return out;
}

void PI05Model::prepareModulations(unsigned int steps) {
  if (modulation_steps_ == steps)
    return;
  time_conditions_.clear();
  modulation_cache_.clear();
  std::vector<float> conditions;
  conditions.reserve(static_cast<size_t>(steps) * EXPERT_DIM);
  const float dt = -1.0f / steps;
  float t = 1.0f;
  for (unsigned int step = 0; step < steps; ++step) {
    time_conditions_.push_back(timeCondition(t));
    const auto &cond = time_conditions_.back();
    conditions.insert(conditions.end(), cond.begin(), cond.end());
    t += dt;
  }
  const unsigned int width = 3 * EXPERT_DIM;
  auto project = [&](const std::vector<float> &weight,
                     const std::vector<float> &bias) {
    std::vector<float> projected;
    matmat(weight, conditions, projected, steps, EXPERT_DIM, width);
    for (size_t i = 0; i < projected.size(); ++i)
      projected[i] += bias[i % width];
    modulation_cache_.push_back(std::move(projected));
  };
  for (const auto &layer : layers_) {
    project(layer.n1_w, layer.n1_b);
    project(layer.n2_w, layer.n2_b);
  }
  project(fn_w, fn_b);
  modulation_steps_ = steps;
}

std::vector<float>
PI05Model::sampleActions(const std::vector<unsigned int> &prompt_ids,
                         const std::vector<float> &noise,
                         unsigned int num_steps) {
  if (num_steps == 0)
    throw std::invalid_argument("pi05: num_steps must be positive");
  if (!expert_loaded_)
    throw std::runtime_error("pi05: expert weights not loaded");
  if (noise.size() != static_cast<size_t>(ACTION_HORIZON) * ACTION_DIM)
    throw std::invalid_argument("pi05: noise must be [horizon, action_dim]");
  // openpi order puts the image rows FIRST, so bos is not necessarily the
  // first id — require that the language rows carry exactly one bos.
  if (std::count(prompt_ids.begin(), prompt_ids.end(), 2u) != 1)
    throw std::invalid_argument("pi05: prompt_ids must carry one bos token");

  prepareModulations(num_steps);

  // ---- 1. compose the merged prefix plane ----
  auto *pg = backbone_.get();
  std::vector<float> prefix_plane;
  unsigned int n_prefix = 0;
  {
    const unsigned int dim = DIM;
    // openpi embed_prefix order: IMAGE rows first, then the language rows
    // (bos + prompt). RoPE positions follow this order, so the 3B e2e and
    // the tiny fixture must share it. <image> placeholders inside
    // prompt_ids consume the injected vision rows wherever they appear.
    size_t vision_cursor = 0;
    const unsigned int image_token =
      sanitized_cfg_.contains("image_token_index")
        ? sanitized_cfg_["image_token_index"].get<unsigned int>()
        : 257152u;
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
      if (prompt_ids[i] == image_token) {
        if (vision_cursor + DIM > vision_embs_.size())
          throw std::runtime_error("pi05: not enough injected vision rows");
        prefix_plane.insert(prefix_plane.end(),
                            vision_embs_.begin() + vision_cursor,
                            vision_embs_.begin() + vision_cursor + DIM);
        vision_cursor += DIM;
      } else {
        // lookupEmbedding(unsigned) returns a scaled FP32 row copy.
        std::vector<float> row = pg->lookupEmbedding(prompt_ids[i]);
        prefix_plane.insert(prefix_plane.end(), row.begin(), row.end());
      }
      ++n_prefix;
    }
  }
  prefix_len_ = n_prefix;

  // Bind and feed the cache by graph input name, then read the manager's
  // layer-indexed tensors. No dependence on graph declaration or name order.
  pg->prefillEmbeddings(prefix_plane.data(), n_prefix);

  const size_t cache_rows = n_prefix + ACTION_HORIZON;
  k_cache_.assign(NUM_LAYERS, {});
  v_cache_.assign(NUM_LAYERS, {});
  for (size_t li = 0; li < NUM_LAYERS; ++li) {
    k_cache_[li].assign(cache_rows * kv_width_, 0.0f);
    v_cache_[li].assign(cache_rows * kv_width_, 0.0f);
    // Preserve FP32 prefix K/V, including post-RoPE keys, across Euler steps.
    const auto &prefix = pg->prefixKV(li);
    dumpDiagnostic("bb" + std::to_string(li) + "_k", prefix.key);
    dumpDiagnostic("bb" + std::to_string(li) + "_v", prefix.value);
    std::copy(prefix.key.begin(), prefix.key.end(), k_cache_[li].begin());
    std::copy(prefix.value.begin(), prefix.value.end(), v_cache_[li].begin());
  }
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  if (causallm_engine() == "cuda")
    prepareCudaExpert();
#endif

  // ---- 4. Euler flow loop ----
  std::vector<float> x_t = noise;
  const float dt = -1.0f / num_steps;
  per_step_v_t_.clear();
  for (unsigned int step = 0; step < num_steps; ++step) {
    const auto &cond = time_conditions_.at(step);
    if (per_step_v_t_.empty())
      dumpDiagnostic("cond0", cond);
    std::vector<float> v_t;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    if (causallm_engine() == "cuda")
      v_t = denoiseStepCuda(x_t, cond);
    else
#endif
      v_t = denoiseStep(x_t, cond);
    per_step_v_t_.push_back(v_t);
    for (size_t i = 0; i < x_t.size(); ++i)
      x_t[i] += dt * v_t[i];
  }
  if (std::getenv("NNTR_PI05_DIAGNOSTICS")) {
    std::vector<float> all;
    for (const auto &step : per_step_v_t_)
      all.insert(all.end(), step.begin(), step.end());
    dumpDiagnostic("v_t", all);
    dumpDiagnostic("actions", x_t);
  }
  return x_t;
}

std::vector<float> PI05Model::denoiseStep(const std::vector<float> &x_t,
                                          const std::vector<float> &cond) {
  (void)cond;
  const unsigned int T = ACTION_HORIZON;
  const unsigned int d = EXPERT_DIM;
  const unsigned int m = EXPERT_MLP_DIM;
  const unsigned int hd = EXPERT_HEAD_DIM;
  const unsigned int H = EXPERT_HEADS;
  const unsigned int KVH = EXPERT_KV_HEADS;
  const unsigned int kw = kv_width_;
  const unsigned int Tb = prefix_len_;
  const unsigned int Te = T;
  const unsigned int Tt = Tb + Te;

  // ---- embed the suffix: action_in_proj ----
  std::vector<float> suffix;
  matmat(act_in_w, x_t, suffix, T, ACTION_DIM, d);
  for (size_t i = 0; i < suffix.size(); ++i)
    suffix[i] += act_in_b[i % d];

  // ---- expert stream ----
  std::vector<float> x_e = suffix;

  std::vector<float> positions(Te);
  for (unsigned int i = 0; i < Te; ++i)
    positions[i] = static_cast<float>(Tb + i);

  // precompute rope cos/sin for the suffix positions
  const unsigned int half = hd / 2;
  std::vector<float> cos_t(Te * half), sin_t(Te * half);
  for (unsigned int p = 0; p < Te; ++p) {
    for (unsigned int j = 0; j < half; ++j) {
      const float freq =
        1.0f / std::pow(ROPE_THETA, (2.0f * j) / static_cast<float>(hd));
      const float ang = positions[p] * freq;
      cos_t[p * half + j] = std::cos(ang);
      sin_t[p * half + j] = std::sin(ang);
    }
  }

  // The expert has constant widths across layers. Every GEMM uses beta=0,
  // so these per-step workspaces are overwritten rather than zeroed/reallocated
  // for each layer. No input-dependent state survives a denoise step.
  std::vector<float> n1(T * d), n2(T * d), after1(T * d), qs(T * H * hd),
    ks(T * kw), vs(T * kw), projected, att_out(T * H * hd), att_proj,
    act(T * m), mlp(T * d), gate_up, g, u,
    attention_scores(static_cast<size_t>(H) * T * Tt);
  for (size_t li = 0; li < NUM_LAYERS; ++li) {
    const ExpertLayer &L = layers_[li];
    const float *mod1 = modulation(2 * li);
    const float *mod2 = modulation(2 * li + 1);
    adaptiveNormRows(x_e.data(), mod1, n1.data(), T, d, RMS_EPS);

    if (li == 0 && per_step_v_t_.empty())
      dumpDiagnostic("l0_attn_in", n1);
    // expert Q/K/V of the suffix rows
    if (expert_mlp_fused_) {
      const size_t width = H * hd + 2 * kw;
      matmat(L.qw, n1, projected, T, d, width);
      qs.resize(T * H * hd);
      ks.resize(T * kw);
      vs.resize(T * kw);
      for (size_t r = 0; r < T; ++r) {
        auto *row = projected.data() + r * width;
        std::copy_n(row, H * hd, qs.data() + r * H * hd);
        std::copy_n(row + H * hd, kw, ks.data() + r * kw);
        std::copy_n(row + H * hd + kw, kw, vs.data() + r * kw);
      }
    } else {
      matmat(L.qw, n1, qs, T, d, H * hd);
      matmat(L.kw, n1, ks, T, d, KVH * hd);
      matmat(L.vw, n1, vs, T, d, KVH * hd);
    }

    // apply RoPE to q/k BEFORE the attention (non-interleaved rotate-half;
    // positions = prefix_len + r). The backbone's cached K/V were already
    // roped by the backbone graph, so only the suffix needs this.
    for (unsigned int r = 0; r < T; ++r) {
      for (unsigned int h = 0; h < H; ++h) {
        float *q = qs.data() + ((size_t)r * H + h) * hd;
        for (unsigned int j = 0; j < half; ++j) {
          const float c = cos_t[r * half + j], s = sin_t[r * half + j];
          const float x1 = q[j], x2 = q[half + j];
          q[j] = x1 * c - x2 * s;
          q[half + j] = x1 * s + x2 * c;
        }
      }
      for (unsigned int kh = 0; kh < KVH; ++kh) {
        float *k = ks.data() + ((size_t)r * KVH + kh) * hd;
        for (unsigned int j = 0; j < half; ++j) {
          const float c = cos_t[r * half + j], s = sin_t[r * half + j];
          const float x1 = k[j], x2 = k[half + j];
          k[j] = x1 * c - x2 * s;
          k[half + j] = x1 * s + x2 * c;
        }
      }
    }

    // append the suffix K/V to the shared cache (this step's rows)
    for (unsigned int r = 0; r < T; ++r) {
      std::copy(ks.begin() + (size_t)r * kw, ks.begin() + (size_t)(r + 1) * kw,
                k_cache_[li].data() + (size_t)(Tb + r) * kw);
      std::copy(vs.begin() + (size_t)r * kw, vs.begin() + (size_t)(r + 1) * kw,
                v_cache_[li].data() + (size_t)(Tb + r) * kw);
    }

    // The prefix mask keeps backbone queries independent of action rows.
    // Each expert query attends the frozen prefix and this step's action K/V.
    {
      const auto order =
        static_cast<unsigned int>(nntrainer::TStorageOrder::ROW_MAJOR);
      const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
      const auto compute_head = [&](size_t h) {
        float *scores = attention_scores.data() + h * T * Tt;
        const unsigned int kh = h / (H / KVH);
        nntrainer::sgemm(order, false, true, Tt, T, hd, scale,
                         k_cache_[li].data() + kh * hd, kw, qs.data() + h * hd,
                         H * hd, 0.0f, scores, T);
        nntrainer::softmax_row_inplace<float>(scores, 0, Tt, T);
        nntrainer::sgemm(order, true, false, T, hd, Tt, 1.0f, scores, T,
                         v_cache_[li].data() + kh * hd, kw, 0.0f,
                         att_out.data() + h * hd, H * hd);
      };
#ifdef USE_MKL
      nntrainer::ThreadManager::Global().parallel_for(0, H, compute_head);
#else
      for (unsigned int h = 0; h < H; ++h)
        compute_head(h);
#endif
    }

    matmat(L.ow, att_out, att_proj, T, H * hd, d);

    gatedResidualRows(x_e.data(), att_proj.data(), mod1 + 2 * d, after1.data(),
                      T, d);

    if (li == 0 && per_step_v_t_.empty())
      dumpDiagnostic("l0_after_attn", after1);
    adaptiveNormRows(after1.data(), mod2, n2.data(), T, d, RMS_EPS);

    // MLP: gate -> gelu(tanh) -> mul up -> down
    if (expert_mlp_fused_) {
      matmat(L.gw, n2, gate_up, T, d, 2 * EXPERT_MLP_DIM);
      nntrainer::ThreadManager::Global().parallel_for(0, T, [&](size_t r) {
        auto *gate = gate_up.data() + r * 2 * m;
        nntrainer::tanh_gelu_mul(m, act.data() + r * m, gate, gate + m);
      });
    } else {
      matmat(L.gw, n2, g, T, d, EXPERT_MLP_DIM);
      matmat(L.uw, n2, u, T, d, EXPERT_MLP_DIM);
      nntrainer::tanh_gelu_mul(act.size(), act.data(), g.data(), u.data());
    }
    matmat(L.dw, act, mlp, T, EXPERT_MLP_DIM, d);

    gatedResidualRows(after1.data(), mlp.data(), mod2 + 2 * d, x_e.data(), T,
                      d);
  }

  adaptiveNormRows(x_e.data(), modulation(2 * NUM_LAYERS), x_e.data(), T, d,
                   RMS_EPS);

  // action_out_proj on the last horizon rows (all of them here)
  std::vector<float> v_t;
  matmat(act_out_w, x_e, v_t, T, d, ACTION_DIM);

  // runtime-gated per-step trace (compile-time macro removed on purpose:
  // an #ifdef'd dump can never be switched on at runtime)

  for (unsigned int r = 0; r < T; ++r)
    for (unsigned int o = 0; o < ACTION_DIM; ++o)
      v_t[(size_t)r * ACTION_DIM + o] += act_out_b[o];

  return v_t;
}

} // namespace causallm
