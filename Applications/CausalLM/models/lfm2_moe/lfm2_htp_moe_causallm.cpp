// SPDX-License-Identifier: Apache-2.0
/** @file lfm2_htp_moe_causallm.cpp
 * @brief LFM2 hybrid MoE graph with TL2 experts and affine WH projections.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#include "lfm2_htp_moe_causallm.h"
#include <algorithm>
#include <app_context.h>
#include <conv_block_layer.h>
#include <cstring>
#include <engine.h>
#include <fstream>
#include <htp_mha_core.h>
#include <layer_context.h>
#include <layer_node.h>
#include <limits>
#include <llm_util.hpp>
#include <map>
#include <model.h>
#include <stdexcept>
#ifdef ENABLE_HEXKL
#include <htp_backend.h>
#include <htp_graph_desc.h>
#endif

namespace causallm {
Lfm2HtpMoECausalLM::Lfm2HtpMoECausalLM(json &cfg, json &gen, json &nntr) :
  Transformer(sanitizeConfig(cfg), sanitizeGenerationConfig(gen, cfg), nntr,
              ModelType::CAUSALLM),
  Gemma4HtpCausalLM(cfg, gen, nntr) {
  setupParameters(cfg, gen, nntr);
}

void Lfm2HtpMoECausalLM::setupParameters(json &cfg, json &gen, json &nntr) {
  Gemma4HtpCausalLM::setupParameters(cfg, gen, nntr);
  if (nntr.value("lfm2_weight_format", std::string()) != "htp_fused_v1" ||
      cfg.value("model_type", std::string()) != "lfm2_moe")
    throw std::invalid_argument(
      "LFM2 MoE requires lfm2_weight_format=htp_fused_v1");
  dense_layers_ = cfg.at("num_dense_layers").get<unsigned>();
  NUM_EXPERTS = cfg.at("num_experts").get<unsigned>();
  NUM_EXPERTS_PER_TOK = cfg.at("num_experts_per_tok").get<unsigned>();
  MOE_INTERMEDIATE_SIZE = cfg.at("moe_intermediate_size").get<unsigned>();
  MOE_LAYER_DTYPE = nntr.at("moe_layer_dtype").get<std::string>();
  MOE_ENGINE = nntr.value("moe_engine", std::string("htp"));
  MOE_CACHE_EXPERTS = nntr.value("moe_cache_experts", 16u);
  ENABLE_MOE_BLOCK = true;
  EMBEDDING_SCALE = 1.0f;
  // Use per-row INT8 prefill for LFM2's raw V projections.
  ATTENTION_KV_DTYPE = "q8_row";
  NORM_EPS = cfg.value("norm_eps", 1e-5f);
  htp_execution_profile_ = true;
  if (BATCH_SIZE != 1 || INIT_SEQ_LEN <= 0 || INIT_SEQ_LEN > MAX_SEQ_LEN ||
      TIE_WORD_EMBEDDINGS || dense_layers_ > unsigned(NUM_LAYERS) ||
      layer_types.size() != size_t(NUM_LAYERS) || HEAD_DIM != 64 ||
      NUM_EXPERTS == 0 || NUM_EXPERTS > 64 || NUM_EXPERTS % 32 ||
      NUM_EXPERTS_PER_TOK == 0 || NUM_EXPERTS_PER_TOK > NUM_EXPERTS ||
      cfg.value("conv_L_cache", 3) != 3 || cfg.value("conv_bias", false) ||
      prefillChunk() != 0 || !cfg.value("norm_topk_prob", true) ||
      cfg.value("routed_scaling_factor", 1.0f) != 1.0f ||
      !cfg.value("use_expert_bias", true) || MEMORY_SWAP ||
      nntr.value("kv_window_ring", false) ||
      nntr.value("model_tensor_type", std::string("FP32-FP32")) != "FP32-FP32")
    throw std::invalid_argument("Unsupported LFM2 HTP geometry or precision");
  for (const auto &kind : layer_types)
    if (kind != "conv" && kind != "full_attention")
      throw std::invalid_argument("Unsupported LFM2 layer type");
  for (const auto &engine : {MOE_ENGINE, ATTN_PROJ_ENGINE, FFN_ENGINE,
                             ATTENTION_ENGINE, LMHEAD_ENGINE})
    if (engine != "htp")
      throw std::invalid_argument(
        "LFM2 fused profile requires all HTP engines");
  if (MOE_LAYER_DTYPE != "TL2CX_WH" || FC_LAYER_DTYPE != "QS4CX_WH_AFFINE" ||
      LMHEAD_DTYPE != "QS4CX_WH_AFFINE" || EMBEDDING_DTYPE != "FP32" ||
      nntr.value("short_conv_dtype", std::string()) != "FP32")
    throw std::invalid_argument("LFM2 fused profile requires TL2, affine WH, "
                                "FP32 embedding and FP32 conv");
#ifdef ENABLE_HEXKL
  nntrainer::get_htp_ops()->set_moe_geglu(false);
#else
  throw std::invalid_argument("LFM2 HTP requires an ENABLE_HEXKL build");
#endif
}

void Lfm2HtpMoECausalLM::registerCustomLayers() {
  Gemma4HtpCausalLM::registerCustomLayers();
  auto *ctx = static_cast<nntrainer::AppContext *>(
    nntrainer::Engine::Global().getRegisteredContext("cpu"));
  try {
    ctx->registerFactory(nntrainer::createLayer<ConvBlockLayer>);
  } catch (const std::invalid_argument &) {
    // The shared fused layer can already have been registered by another model.
  }
}

Tensor Lfm2HtpMoECausalLM::attentionBlock(unsigned l, Tensor input) {
  const std::string p = "layer" + std::to_string(l);
  LayerHandle qkv(createLayer(
    "htp_qkv_layer",
    {withKey("name", p + "_qkv"), withKey("q_unit", NUM_HEADS * HEAD_DIM),
     withKey("k_unit", NUM_KEY_VALUE_HEADS * HEAD_DIM),
     withKey("v_unit", NUM_KEY_VALUE_HEADS * HEAD_DIM),
     withKey("feature_size", HEAD_DIM),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("in_norm", "true"),
     withKey("v_norm", "false"), withKey("q_scale", "1"),
     withKey("rope", "true"), withKey("rope_theta", ROPE_THETA),
     withKey("max_timestep", MAX_SEQ_LEN),
     withKey("weight_dtype", FC_LAYER_DTYPE), withKey("engine", "htp")}));
  Tensor projections = qkv(input);
  auto cache =
    createGemma4KVCachePlaceholders(l, NUM_KEY_VALUE_HEADS * HEAD_DIM);
  LayerHandle attn(createAttentionCore(
    {withKey("name", p + "_attention"), withKey("num_heads", NUM_HEADS),
     withKey("num_heads_kv", NUM_KEY_VALUE_HEADS),
     withKey("max_timestep", MAX_SEQ_LEN),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("sliding_window", std::numeric_limits<unsigned>::max()),
     withKey("use_rope", "false"), withKey("rope_theta", ROPE_THETA),
     withKey("max_new_tokens", NUM_TO_GENERATE),
     withKey("is_causal", "true")}));
  Tensor a = attn({projections.output(0), projections.output(1),
                   projections.output(2), cache.first, cache.second});
  LayerHandle out(createLayer(
    "fully_connected",
    {withKey("name", p + "_attention_out"), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", "htp")}));
  return out(a);
}

std::pair<Tensor, Tensor> Lfm2HtpMoECausalLM::constructModel() {
  Tensor x({1, 1, 1, static_cast<unsigned>(INIT_SEQ_LEN)}, "input0");
  LayerHandle embedding(createLayer(
    "embedding_layer", buildEmbeddingLayerProperties("embedding0", NUM_VOCAB,
                                                     DIM, "FP32", 1.0f, "")));
  Tensor h = embedding(x);
  for (unsigned l = 0; l < unsigned(NUM_LAYERS); ++l) {
    const std::string p = "layer" + std::to_string(l);
    Tensor branch;
    if (layer_types[l] == "conv") {
      LayerHandle norm(
        createLayer("rms_norm", {withKey("name", p + "_conv_norm"),
                                 withKey("epsilon", std::to_string(NORM_EPS)),
                                 withKey("packed", "false")}));
      LayerHandle conv(createLayer(
        "conv_block",
        {withKey("name", p + "_conv"), withKey("unit", DIM),
         withKey("weight_dtype", FC_LAYER_DTYPE), withKey("engine", "htp")}));
      branch = conv(norm(h));
    } else {
      branch = attentionBlock(l, h);
    }
    LayerHandle add(
      createLayer("residual_add", {withKey("name", p + "_operator_add"),
                                   withKey("engine", "htp")}));
    h = add({h, branch});
    if (l < dense_layers_) {
      LayerHandle ffn(createLayer(
        "dense_ffn",
        {withKey("name", p + "_ffn"), withKey("unit", INTERMEDIATE_SIZE),
         withKey("gate_first", "true"), withKey("glu_activation", "swish"),
         withKey("in_norm", "true"), withKey("out_norm", "false"),
         withKey("epsilon", std::to_string(NORM_EPS)),
         withKey("weight_dtype", FC_LAYER_DTYPE), withKey("engine", "htp")}));
      branch = ffn(h);
    } else {
      std::vector<std::string> props = {
        withKey("name", p + "_sparse_moe"),
        withKey("unit", MOE_INTERMEDIATE_SIZE),
        withKey("num_experts", NUM_EXPERTS),
        withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
        withKey("moe_activation", "swish"),
        withKey("router_type", "sigmoid_bias"),
        withKey("in_norm", "true"),
        withKey("epsilon", std::to_string(NORM_EPS)),
        withKey("weight_dtype", MOE_LAYER_DTYPE),
        withKey("engine", "htp")};
      if (MOE_CACHE_EXPERTS)
        props.push_back(withKey("cache_experts", MOE_CACHE_EXPERTS));
      LayerHandle moe(createLayer("lfm2_moe", props));
      branch = moe(h);
    }
    LayerHandle ffn_add(
      createLayer("residual_add",
                  {withKey("name", p + "_ffn_add"), withKey("engine", "htp")}));
    h = ffn_add({h, branch});
  }
  LayerHandle norm(
    createLayer("rms_norm", {withKey("name", "output_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false")}));
  LayerHandle head(createLayer(
    "lm_head",
    {withKey("name", "output_of_causallm"), withKey("unit", NUM_VOCAB),
     withKey("disable_bias", "true"), withKey("weight_dtype", LMHEAD_DTYPE),
     withKey("engine", "htp")}));
  return {x, head(norm(h))};
}

void Lfm2HtpMoECausalLM::allocateAndBindKVCache() {
  // The generic binder permits conv layers without KV placeholders.
  CausalLM::allocateAndBindKVCache();
}

void Lfm2HtpMoECausalLM::load_weight(const std::string &path) {
  if (prepared_)
    throw std::logic_error("Create a new LFM2 handle to reload weights");
  loading_ = true;
  try {
    CausalLM::load_weight(path);
  } catch (...) {
    loading_ = false;
    throw;
  }
  loading_ = false;
  prepareDecodeGraph();
  repack_weight();
}

void Lfm2HtpMoECausalLM::repack_weight() {
  if (loading_ || prepared_)
    return;
  Gemma4HtpTransformer::repack_weight();
#ifdef ENABLE_HEXKL
  nntrainer::get_htp_ops()->finish_decode_graph_q4_0();
#endif
  prepared_ = true;
  weight_repacked_ = true;
}

void Lfm2HtpMoECausalLM::forEachConvBlock(
  const std::function<void(ConvBlockLayer &, nntrainer::RunLayerContext &)>
    &fn) {
  model->forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &rc, void *) {
      auto *node = dynamic_cast<nntrainer::LayerNode *>(&layer);
      if (node)
        if (auto *conv = dynamic_cast<ConvBlockLayer *>(node->getLayer()))
          fn(*conv, rc);
    },
    nullptr);
}

void Lfm2HtpMoECausalLM::run(const WSTR prompt, bool sample, const WSTR system,
                             const WSTR tail, bool log) {
#ifdef ENABLE_HEXKL
  // Each prompt is independent, as Gemma4HtpMoECausalLM::run does it: the
  // DSP's decode state (attention caches, conv states), the host conv states
  // and the host KV slabs decode may have dropped are reset.
  if (ran_) {
    nntrainer::htp_begin_prompt();
    model->forEachLayer(
      [](ml::train::Layer &layer, nntrainer::RunLayerContext &rc, void *) {
        auto *node = dynamic_cast<nntrainer::LayerNode *>(&layer);
        if (!node)
          return;
        if (auto *mha = dynamic_cast<HtpMHACoreLayer *>(node->getLayer()))
          mha->beginHtpPrompt();
        else if (auto *conv = dynamic_cast<ConvBlockLayer *>(node->getLayer()))
          conv->beginPrompt(rc);
      },
      nullptr);
    kv_cache = KVCacheManager{};
    kv_cache_bound = false;
  }
  // a precomputed cache (system_prompt.kvcache) stays on: the run saves
  // or loads it at position 0
  if (USE_KVCACHE)
    setPrecomputedKVCache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);
  else
    setSessionTokenPosition(0);
#endif
  ran_ = true;
  CausalLM::run(prompt, sample, system, tail, log);
}

namespace {
/** <path>.conv: this header, then each conv block's state ([2, C] FP32) in
 *  layer order. A snapshot of another geometry or position is refused. */
struct ConvSnapshotHeader {
  char magic[8];
  uint32_t version, layers, width, pos;
};
constexpr char kConvMagic[8] = {'N', 'N', 'T', 'R', 'C', 'O', 'N', 'V'};
constexpr uint32_t kConvVersion = 1;
} // namespace

void Lfm2HtpMoECausalLM::save_kvcache(std::string path, int to) {
  CausalLM::save_kvcache(path, to);
  std::vector<nntrainer::Tensor *> states;
  forEachConvBlock([&](ConvBlockLayer &conv, nntrainer::RunLayerContext &rc) {
    states.push_back(&conv.convState(rc));
  });
  if (states.empty())
    return;
  const std::string conv_path = path + ".conv";
  std::ofstream f(conv_path, std::ios::binary);
  ConvSnapshotHeader h{};
  std::memcpy(h.magic, kConvMagic, sizeof(h.magic));
  h.version = kConvVersion;
  h.layers = static_cast<uint32_t>(states.size());
  h.width = states.front()->width();
  h.pos = static_cast<uint32_t>(to);
  f.write(reinterpret_cast<const char *>(&h), sizeof(h));
  for (auto *t : states)
    f.write(t->getData<char>(), t->bytes());
  if (!f)
    throw std::runtime_error("save_kvcache: cannot write " + conv_path);
}

void Lfm2HtpMoECausalLM::load_kvcache(std::string path, int to) {
  CausalLM::load_kvcache(path, to);
  std::vector<nntrainer::Tensor *> states;
  forEachConvBlock([&](ConvBlockLayer &conv, nntrainer::RunLayerContext &rc) {
    states.push_back(&conv.convState(rc));
  });
  if (states.empty())
    return;
  // Without the conv states the loaded position is wrong: refused, as a
  // KV file of another model is.
  const std::string conv_path = path + ".conv";
  std::ifstream f(conv_path, std::ios::binary);
  ConvSnapshotHeader h{};
  f.read(reinterpret_cast<char *>(&h), sizeof(h));
  if (!f || std::memcmp(h.magic, kConvMagic, sizeof(h.magic)) != 0 ||
      h.version != kConvVersion)
    throw std::runtime_error("load_kvcache: " + conv_path +
                             " is missing or not an LFM2 conv snapshot; "
                             "save the KV cache again with this build");
  if (h.layers != states.size() || h.width != states.front()->width() ||
      h.pos != static_cast<uint32_t>(to))
    throw std::runtime_error(
      "load_kvcache: " + conv_path + " holds " + std::to_string(h.layers) +
      " layers x " + std::to_string(h.width) + " at position " +
      std::to_string(h.pos) + ", this model wants " +
      std::to_string(states.size()) + " x " +
      std::to_string(states.front()->width()) + " at " + std::to_string(to));
  for (auto *t : states)
    f.read(t->getData<char>(), t->bytes());
  if (!f)
    throw std::runtime_error("load_kvcache: " + conv_path + " is truncated");
}

void Lfm2HtpMoECausalLM::prepareDecodeGraph() {
#ifdef ENABLE_HEXKL
  std::map<std::string, nntrainer::Tensor *> weights;
  model->forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &rc, void *) {
      for (auto *w : rc.getWeights()) {
        const auto &name = w->getName();
        weights[name] = &w->getVariableRef();
        const auto colon = name.rfind(':');
        if (colon != std::string::npos)
          weights.emplace(layer.getName() + name.substr(colon),
                          &w->getVariableRef());
      }
    },
    nullptr);
  auto weight = [&](const std::string &name) -> nntrainer::Tensor & {
    auto it = weights.find(name);
    if (it == weights.end())
      throw std::runtime_error("Missing LFM2 weight " + name);
    return *it->second;
  };
  auto f32 = [&](const std::string &name, size_t count) {
    auto &t = weight(name);
    if (t.getDataType() != nntrainer::Tdatatype::FP32 || t.size() != count)
      throw std::runtime_error("Invalid LFM2 FP32 parameter " + name);
    return t.getData<float>();
  };
  htp_graph_lfm2_shape s{};
  s.n_layers = NUM_LAYERS;
  s.n_dense_layers = dense_layers_;
  s.hidden = DIM;
  s.inter_dense = INTERMEDIATE_SIZE;
  s.inter_moe = MOE_INTERMEDIATE_SIZE;
  s.n_experts = NUM_EXPERTS;
  s.top_k = NUM_EXPERTS_PER_TOK;
  s.n_heads = NUM_HEADS;
  s.n_kv_heads = NUM_KEY_VALUE_HEADS;
  s.head_dim = HEAD_DIM;
  s.vocab = NUM_VOCAB;
  s.max_seq = MAX_SEQ_LEN;
  s.eps = NORM_EPS;
  std::vector<uint8_t> attn(NUM_LAYERS);
  for (unsigned l = 0; l < s.n_layers; ++l)
    attn[l] = layer_types[l] == "full_attention";
  std::vector<uint32_t> words(
    htp_graph_words_for(s.n_layers, HTP_GRAPH_MAX_OPS));
  unsigned n = htp_graph_lfm2_build(words.data(), words.size(), &s, attn.data(),
                                    HTP_GRAPH_KINDS_ALL);
  if (!n)
    throw std::runtime_error("Invalid LFM2 HTP decode graph");
  words.resize(n);
  for (unsigned i = 0; i < words[3]; ++i)
    if (htp_graph_op_at(words.data(), i)->kind == HTP_OP_DENSE_FFN)
      htp_graph_op_at(words.data(), i)->feed |= HTP_GRAPH_FEED_WH;
  auto *ops = nntrainer::get_htp_ops();
  if (!ops->set_decode_graph_desc(words))
    throw std::runtime_error("HTP rejected LFM2 graph");
  auto hand = [&](unsigned i, unsigned kind, const float *v, unsigned count) {
    if (!ops->set_decode_graph_param(i, kind, v, count))
      throw std::runtime_error("HTP rejected LFM2 parameter");
  };
  auto wh = [&](const std::string &name) {
    auto &t = weight(name);
    if (t.getDataType() != nntrainer::Tdatatype::QS4CX_WH_AFFINE ||
        !ops->set_fc_wh_source(t.getData(), t.height(), t.width(), true) ||
        !ops->add_decode_graph_qs4cx(t.getData<char>(), t.getScale<float>(),
                                     t.height(), t.width()))
      throw std::runtime_error("HTP rejected LFM2 affine weight " + name);
  };
  params_.clear();
  unsigned layer = HTP_GRAPH_NO_OP, norm = 0, fc = 0;
  for (unsigned i = 0; i < words[3]; ++i) {
    const auto *op = htp_graph_op_cat(words.data(), i);
    if (layer != op->layer) {
      layer = op->layer;
      norm = fc = 0;
    }
    const std::string p = "layer" + std::to_string(layer);
    switch (op->kind) {
    case HTP_OP_RMSNORM: {
      const std::string name =
        layer == s.n_layers ? "output_norm:gamma"
        : norm++ == 0
          ? (attn[layer] ? p + "_qkv:in_norm_gamma" : p + "_conv_norm:gamma")
          : p + (layer < dense_layers_ ? "_ffn:in_norm_gamma"
                                       : "_sparse_moe:in_norm_gamma");
      hand(i, HTP_GRAPH_PARAM_GAMMA, f32(name, DIM), DIM);
      break;
    }
    case HTP_OP_QK_NORM: {
      std::vector<float> g(2 * HEAD_DIM);
      std::copy_n(f32(p + "_qkv:q_norm_gamma", HEAD_DIM), HEAD_DIM, g.begin());
      std::copy_n(f32(p + "_qkv:k_norm_gamma", HEAD_DIM), HEAD_DIM,
                  g.begin() + HEAD_DIM);
      params_.push_back(std::move(g));
      hand(i, HTP_GRAPH_PARAM_GAMMA, params_.back().data(), 2 * HEAD_DIM);
      break;
    }
    case HTP_OP_ROUTER_TOPK:
      hand(i, HTP_GRAPH_PARAM_ROUTER_W,
           f32(p + "_sparse_moe:gate", DIM * NUM_EXPERTS), DIM * NUM_EXPERTS);
      hand(i, HTP_GRAPH_PARAM_ROUTER_BIAS,
           f32(p + "_sparse_moe:expert_bias", NUM_EXPERTS), NUM_EXPERTS);
      break;
    case HTP_OP_FC:
      if (!attn[layer])
        wh(p + (fc++ == 0 ? "_conv:in_proj" : "_conv:out_proj"));
      else if (fc++ == 0) {
        wh(p + "_qkv:qweight");
        wh(p + "_qkv:kweight");
        wh(p + "_qkv:vweight");
      } else
        wh(p + "_attention_out:weight");
      break;
    case HTP_OP_DENSE_FFN:
      wh(p + "_ffn:up");
      wh(p + "_ffn:gate");
      wh(p + "_ffn:down");
      break;
    case HTP_OP_LM_HEAD:
      wh("output_of_causallm:weight");
      break;
    default:
      break;
    }
  }
#endif
}
} // namespace causallm
