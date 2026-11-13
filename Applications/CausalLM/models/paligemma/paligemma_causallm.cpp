// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   paligemma_causallm.cpp
 * @brief  PaliGemma (Gemma text decoder) causal language model implementation.
 * @date   12 Nov 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "paligemma_causallm.h"
#include "native_linear.h"
#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>
#include <tensor_api.h>

#include <algorithm>
#include <chrono>
#include <cpu_backend.h>
#include <cpu_memory_hints.h>
#include <fp16.h>
#include <fstream>
#include <iostream>

#include <compute_ops.h>
#include <cstring>
#include <embedding_layer.h>
#include <footprint_sampler.h> // per-run honest peak (RssAnon + accelerator)
#include <layer_context.h>
#include <layer_impl.h>
#include <layer_node.h>
#include <mha_core.h>
#include <node_exporter.h>
#include <rms_norm.h>

using ml::train::Tensor;

namespace causallm {
namespace {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
/** Keep the supplied embedding plane in the device activation pool. A plain
 * in-place input aliases host memory, forcing the first norm and residual to
 * host fallbacks which cannot be recorded by a CUDA graph. */
class DeviceEmbeddingInput final : public nntrainer::LayerImpl {
public:
  const std::string getType() const override { return "input"; }
  void finalize(nntrainer::InitLayerContext &ctx) override {
    ctx.setOutputDimensions(ctx.getInputDimensions());
  }
  void forwarding(nntrainer::RunLayerContext &ctx, bool training) override {
    incremental_forwarding(ctx, 0, ctx.getInput(0).height(), training);
  }
  void calcDerivative(nntrainer::RunLayerContext &) override {
    throw std::runtime_error("embedding input is inference-only");
  }
  bool supportBackwarding() const override { return false; }
  bool supportInPlace() const override { return false; }
  nntrainer::InPlaceType initializeInPlace() override {
    return nntrainer::InPlaceType::NONE;
  }
  void incremental_forwarding(nntrainer::RunLayerContext &ctx,
                              unsigned int from, unsigned int to,
                              bool training) override {
    (void)training;
    auto &input = ctx.getInput(0);
    auto &out = ctx.getOutput(0);
    if (nntrainer::cuda::dev_only(out.getData()))
      throw std::invalid_argument(
        "embedding input requires host-readable CUDA activations");
    nntrainer::cuda::StreamManager::Global().finishIfAsync();
    const size_t bytes =
      static_cast<size_t>(to - from) * input.width() *
      (input.getDataType() == nntrainer::Tdatatype::FP16 ? 2 : 4);
    for (unsigned int b = 0; b < input.batch(); ++b)
      std::memcpy(
        out.getData<char>() +
          b * out.getDim().getFeatureLen() *
            (input.getDataType() == nntrainer::Tdatatype::FP16 ? 2 : 4),
        input.getData<char>() +
          b * input.getDim().getFeatureLen() *
            (input.getDataType() == nntrainer::Tdatatype::FP16 ? 2 : 4),
        bytes);
  }
};
#endif

/** Vocabulary projection for the last active row only. */
class PaliGemmaHead : public PreparedVlaLayer {
public:
  explicit PaliGemmaHead(bool transposed = false, bool external = false) :
    transposed_(transposed), external_(external) {}
  void bindCpuEmbedding(const std::vector<float> *embedding) {
    cpu_embedding_ = embedding;
  }
  static constexpr const char *type = "paligemma_head";
  const std::string getType() const override { return type; }
  void setProperty(const std::vector<std::string> &values) override {
    auto remain = nntrainer::loadProperties(values, props_);
    LayerImpl::setProperty(remain);
  }
  void finalize(nntrainer::InitLayerContext &ctx) override {
    auto in = ctx.getInputDimensions().at(0);
    auto out = in;
    const auto units = std::get<nntrainer::props::Unit>(props_).get();
    out.height(1);
    out.width(units);
    out.setTensorType({ctx.getFormat(), ctx.getActivationDataType()});
    ctx.setOutputDimensions({out});
    // The CPU head can use the existing FP32 lookup table directly. This
    // checkpoint profile ends at the final norm and carries no head record.
    if (external_)
      return;
    nntrainer::TensorDim wd(1, 1, in.width(), units,
                            nntrainer::TensorDim::TensorType(
                              ctx.getFormat(), ctx.getWeightDataType()));
    if (transposed_) {
      wd.height(units);
      wd.width(in.width());
    }
    weight_ = ctx.requestWeight(wd, nntrainer::Initializer::NONE,
                                nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                                "weight", true);
  }
  void forwarding(nntrainer::RunLayerContext &ctx, bool training) override {
    incremental_forwarding(ctx, 0, ctx.getInput(0).height(), training);
  }
  void incremental_forwarding(nntrainer::RunLayerContext &ctx,
                              unsigned int from, unsigned int to,
                              bool training) override {
    auto &input = ctx.getInput(0);
    auto &output = ctx.getOutput(0);
    auto row = input.getDim();
    row.batch(1);
    row.height(1);
    auto out_row = output.getDim();
    out_row.batch(1);
    const size_t last = std::min<size_t>(to - from, input.height()) - 1;
    for (unsigned int b = 0; b < input.batch(); ++b) {
      auto x = input.getSharedDataTensor(
        row, b * input.getDim().getFeatureLen() + last * input.width(), true);
      auto y = output.getSharedDataTensor(
        out_row, b * output.getDim().getFeatureLen(), true);
      if (cpu_embedding_ && x.getDataType() == nntrainer::Tdatatype::FP32) {
        // The tied embedding already has the contiguous [vocabulary, hidden]
        // layout wanted by GEMV; reuse it without a second transposed copy.
#ifdef USE_MKL
        nntrainer::sgemm(
          static_cast<unsigned int>(nntrainer::TStorageOrder::ROW_MAJOR), false,
          true, 1, y.width(), x.width(), 1.0f, x.getData<float>(), x.width(),
          cpu_embedding_->data(), x.width(), 0.0f, y.getData<float>(),
          y.width());
#else
        nntrainer::sgemv(
          static_cast<unsigned int>(nntrainer::TStorageOrder::ROW_MAJOR), false,
          y.width(), x.width(), 1.0f, cpu_embedding_->data(), x.width(),
          x.getData<float>(), 1, 0.0f, y.getData<float>(), 1);
#endif
      } else if (external_)
        throw std::runtime_error("external head embedding has not been loaded");
      else if (transposed_)
        NativeLinear::project(x, ctx.getWeight(weight_), y);
      else
        x.getOps()->fc(x, ctx.getWeight(weight_), y);
    }
  }
  void calcDerivative(nntrainer::RunLayerContext &) override {
    throw std::runtime_error("PaliGemmaHead is inference-only");
  }
  bool supportBackwarding() const override { return false; }
  void exportTo(nntrainer::Exporter &e,
                const ml::train::ExportMethods &method) const override {
    LayerImpl::exportTo(e, method);
    e.saveResult(props_, method, this);
  }

private:
  bool transposed_ = false;
  bool external_ = false;
  const std::vector<float> *cpu_embedding_ = nullptr;
  std::tuple<nntrainer::props::Unit> props_;
  unsigned int weight_ = 0;
};

/** Route every compute node through the selected engine. */
LayerHandle createStandardModelLayer(const std::string &type,
                                     std::vector<std::string> properties) {
  const bool host_only = type == "input" || type == "conv2d" ||
                         type == "weight" || type == "reshape" ||
                         type == "permute" || type == "multiply";
  properties.emplace_back(
    withKey("engine", host_only ? "cpu" : causallm_engine()));
  return createLayer(type, properties);
}
} // namespace

LayerHandle
PaliGemmaTransformer::createModelLayer(const std::string &type,
                                       std::vector<std::string> properties) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  if (USE_EMBEDDING && type == "input" && causallm_engine() == "cuda") {
    properties.emplace_back(withKey("engine", "cuda"));
    std::shared_ptr<ml::train::Layer> node = nntrainer::createLayerNode(
      std::make_unique<DeviceEmbeddingInput>(), properties);
    return LayerHandle(std::move(node));
  }
#endif
  if (WEIGHT_TRANSPOSED && type == "fully_connected") {
    properties.emplace_back(withKey("engine", causallm_engine()));
    std::shared_ptr<ml::train::Layer> node =
      nntrainer::createLayerNode(std::make_unique<NativeLinear>(), properties);
    return LayerHandle(std::move(node));
  }
  return createStandardModelLayer(type, std::move(properties));
}

json &PaliGemmaTransformer::sanitizeConfig(json &cfg) {
  // Patch embedding, shape transforms and FP32 elementwise fallbacks use
  // host-readable activations. Establish this model profile before CUDA
  // context initialization selects the generic device-only defaults.
  if (causallm_engine() == "cuda") {
    configurePaliGemmaCuda();
  }
  // PaliGemma's config.json nests the text decoder under "text_config" and
  // omits keys the Gemma1 decoder needs. Flatten: merge text_config keys up
  // into the top level WITHOUT overwriting keys already there (the top-level
  // vocab_size/hidden_size/projection_dim agree with the text side anyway),
  // then fill the Gemma1 defaults the file leaves out.
  if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
    for (auto it = cfg["text_config"].begin(); it != cfg["text_config"].end();
         ++it) {
      if (!cfg.contains(it.key()))
        cfg[it.key()] = it.value();
    }
  }
  if (!cfg.contains("rms_norm_eps"))
    cfg["rms_norm_eps"] = 1e-6;
  if (!cfg.contains("rope_theta"))
    cfg["rope_theta"] = 10000;
  if (!cfg.contains("tie_word_embeddings"))
    cfg["tie_word_embeddings"] = true;
  if (!cfg.contains("max_position_embeddings"))
    cfg["max_position_embeddings"] = 8192;
  if (!cfg.contains("head_dim") && cfg.contains("hidden_size") &&
      cfg.contains("num_attention_heads")) {
    cfg["head_dim"] =
      cfg["hidden_size"].get<int>() / cfg["num_attention_heads"].get<int>();
  }
  // Prefix-LM: the graph's attention is bidirectional (is_causal=false); the
  // causal suffix is orchestrated per-token in run_with_embeddings().
  cfg["is_causal"] = false;
  return cfg;
}

json &PaliGemmaTransformer::sanitizeGenerationConfig(json &gen_cfg,
                                                     const json &cfg) {
  if (!gen_cfg.contains("eos_token_id")) {
    if (cfg.contains("eos_token_id")) {
      auto eos = cfg["eos_token_id"];
      if (eos.is_number()) {
        gen_cfg["eos_token_id"] =
          std::vector<unsigned int>{eos.get<unsigned int>()};
      } else {
        gen_cfg["eos_token_id"] = eos;
      }
    }
  } else {
    auto eos = gen_cfg["eos_token_id"];
    if (eos.is_number()) {
      gen_cfg["eos_token_id"] =
        std::vector<unsigned int>{eos.get<unsigned int>()};
    }
  }

  return gen_cfg;
}

Tensor PaliGemmaTransformer::createAttention(int layer_id, int seq_len,
                                             int heads, int head_dim,
                                             Tensor query, Tensor key,
                                             Tensor value) {
  if (!PREFILL_ONLY && !WEIGHT_TRANSPOSED)
    return Transformer::createAttention(layer_id, seq_len, heads, head_dim,
                                        query, key, value);
  auto project = [this, layer_id](const std::string &name, unsigned int width,
                                  Tensor input) {
    auto fc = createModelLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + name),
       withKey("unit", width), withKey("disable_bias", "true"),
       withKey("weight_dtype", FC_LAYER_DTYPE)});
    return fc(input);
  };
  Tensor q, k, v;
  if (WEIGHT_TRANSPOSED && FUSE_QKV && !HALF_ACTIVATIONS &&
      causallm_engine() == "cpu") {
    std::shared_ptr<ml::train::Layer> node = nntrainer::createLayerNode(
      std::make_unique<NativeQKV>(heads * head_dim,
                                  heads * head_dim / GQA_SIZE),
      {withKey("name", "layer" + std::to_string(layer_id) + "_qkv"),
       withKey("engine", "cpu"), withKey("weight_dtype", FC_LAYER_DTYPE)});
    LayerHandle projection(node);
    auto outputs = projection(query);
    q = outputs.output(0);
    k = outputs.output(1);
    v = outputs.output(2);
  } else {
    q = project("_wq", heads * head_dim, query);
    k = project("_wk", heads * head_dim / GQA_SIZE, key);
    v = project("_wv", heads * head_dim / GQA_SIZE, value);
  }
  if (!PREFILL_ONLY) {
    LayerHandle mha(createLayer(
      "mha_core",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
       withKey("num_heads", heads), withKey("num_heads_kv", heads / GQA_SIZE),
       withKey("max_timestep", MAX_SEQ_LEN),
       withKey("init_seq_len", INIT_SEQ_LEN),
       withKey("sliding_window", getLayerSlidingWindow(layer_id)),
       withKey("rope_theta", ROPE_THETA),
       withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
       withKey("max_new_tokens", NUM_TO_GENERATE),
       withKey("is_causal", "false"),
       withKey("use_gemm_attention",
               causallm_gemm_attention(USE_FLASH_ATTENTION) ? "true"
                                                            : "false")}));
    auto a = wireAttentionKVCache(layer_id, heads, mha, q, k, v, false);
    return project("_attention_out", DIM, a);
  }
  if (prefix_cache_.size() <= static_cast<size_t>(layer_id))
    prefix_cache_.resize(layer_id + 1);
  prefix_cache_[layer_id] = std::make_shared<PrefixKV>();
  LayerHandle attention(createPrefixAttention(
    "layer" + std::to_string(layer_id) + "_attention", heads, heads / GQA_SIZE,
    head_dim, ROPE_THETA, prefix_cache_[layer_id]));
  return project("_attention_out", DIM, attention({q, k, v}));
}

Tensor PaliGemmaTransformer::createMlp(const int layer_id, int dim,
                                       int hidden_dim, Tensor input) {

  if (WEIGHT_TRANSPOSED && FUSE_MLP) {
    std::shared_ptr<ml::train::Layer> node = nntrainer::createLayerNode(
      std::make_unique<NativeLinear>(true),
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate_up"),
       withKey("unit", hidden_dim), withKey("disable_bias", "true"),
       withKey("engine", causallm_engine()),
       withKey("weight_dtype", FC_LAYER_DTYPE)});
    LayerHandle gate_up(node);
    LayerHandle down(createModelLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
       withKey("unit", dim), withKey("disable_bias", "true"),
       withKey("weight_dtype", FC_LAYER_DTYPE)}));
    return down(gate_up(input));
  }

  // Gate projection
  LayerHandle ffn_gate(createModelLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor gate = ffn_gate(input);

  // Up projection
  LayerHandle ffn_up(createModelLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor up = ffn_up(input);

  // Keep GeGLU on the selected device instead of returning the MLP plane
  // to the host between activation and multiplication.
  LayerHandle activation(createModelLayer(
    "geglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu")}));
  Tensor geglu = activation({gate, up});

  // Down projection
  LayerHandle ffn_down(createModelLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  return ffn_down(geglu);
}

std::pair<Tensor, Tensor> PaliGemmaCausalLM::constructModel() {

  auto &engine = nntrainer::Engine::Global();
  for (const std::string &backend : {std::string("cpu"), causallm_engine()}) {
    try {
      engine.registerLayerFactory(
        backend, nntrainer::createLayer<PaliGemmaHead>, PaliGemmaHead::type);
    } catch (const std::invalid_argument &) {
      // A second model in this process reuses the registered factory.
    }
  }
  Tensor input_tensor;
  Tensor x;

  // Always create embedding0 for: weight loading, lookupEmbedding(),
  // and TIE_WORD_EMBEDDINGS shared_from reference.
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
  LayerHandle embedding(createModelLayer(
    embedding_type,
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME)));

  if (!USE_EMBEDDING) {
    // Standard path: token IDs → embedding → transformer
    LayerHandle input_layer = createModelLayer(
      "input", {withKey("name", "input0"),
                withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))});
    input_tensor = input_layer(Tensor());
    x = embedding(input_tensor);
  } else {
    // Embedding bypass: pre-computed embeddings → transformer directly.
    // embedding0 exists in graph but is disconnected from the data path,
    // so it is not executed during forward but its weights are still loaded.
    LayerHandle input_layer = createModelLayer(
      "input", {withKey("name", "input0"),
                withKey("input_dtype", HALF_ACTIVATIONS ? "FP16" : "FP32"),
                withKey("input_shape", "1:" + std::to_string(INIT_SEQ_LEN) +
                                         ":" + std::to_string(DIM))});
    input_tensor = input_layer(Tensor());
    x = input_tensor;
  }

  // Standard pre-norm Gemma1 decoder blocks (norm -> attn -> residual ->
  // norm -> mlp -> residual)
  for (int i = 0; i < NUM_LAYERS; ++i) {
    x = createTransformerDecoderBlock(i, x);
  }

  // Final normalization
  LayerHandle output_norm(
    createModelLayer("rms_norm", {withKey("name", "output_norm"),
                                  withKey("epsilon", std::to_string(NORM_EPS)),
                                  withKey("packed", "false")}));
  Tensor norm_out = output_norm(x);
  if (PREFILL_ONLY)
    return {input_tensor, norm_out};

  // LM head. In connected (token-id) mode a tied head shares embedding0's
  // weight by name (TieWordEmbedding::read() loads it in embedding mode). In
  // bypass (use_embedding) mode embedding0 never joins the graph, and
  // TieWordEmbedding::read() loads weights ONLY in embedding mode — an
  // lm_head-mode record would stay zero-filled and every logit comes out
  // exactly 0.0 (observed e2e). So the bypass head is a plain fully_connected
  // carrying its OWN [vocab, dim] record; weight_converter.py's bypass layout
  // writes the table as the file's last record to match the load walk's
  // graph order.
  const std::string lmhead_type = (TIE_WORD_EMBEDDINGS && !USE_EMBEDDING)
                                    ? "tie_word_embeddings"
                                    : PaliGemmaHead::type;

  std::vector<std::string> lmhead_props = {
    withKey("name", "output_of_causallm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
  };

  // The tie head shares embedding0's weight tensor BY NAME. That binding
  // only works when embedding0 actually joins the graph: a layer that is
  // never called is not a graph node, its weight is never allocated+loaded,
  // and the head silently multiplies by a zero tensor — e2e produced exactly
  // 0.0 logits while the same weights ran fine on the connected (token-id)
  // path. With use_embedding=true the embedding stays out of the graph
  // (LFM2 bypass pattern), so the head carries its OWN weight record here;
  // the converter appends the [vocab, dim] table as the file's last record
  // (weight_converter.py --bypass_head) to match the load walk's graph
  // order.
  if (TIE_WORD_EMBEDDINGS && !USE_EMBEDDING)
    lmhead_props.emplace_back(withKey("shared_from", "embedding0"));

  std::shared_ptr<ml::train::Layer> head_node;
  bind_cpu_head_ = {};
  if (lmhead_type == PaliGemmaHead::type) {
    auto head =
      std::make_unique<PaliGemmaHead>(WEIGHT_TRANSPOSED, CPU_EXTERNAL_HEAD);
    auto *binding = head.get();
    bind_cpu_head_ = [binding](const std::vector<float> *weights) {
      binding->bindCpuEmbedding(weights);
    };
    lmhead_props.emplace_back(withKey("engine", causallm_engine()));
    head_node = nntrainer::createLayerNode(std::move(head), lmhead_props);
  } else {
    head_node = createModelLayer(lmhead_type, lmhead_props);
  }
  LayerHandle lmhead(head_node);
  Tensor output = lmhead(norm_out);

  return {input_tensor, output};
}

std::vector<float>
PaliGemmaCausalLM::lookupEmbedding(unsigned int token_id) const {
  if (token_id >= NUM_VOCAB) {
    throw std::invalid_argument(
      "lookupEmbedding: token_id (" + std::to_string(token_id) +
      ") exceeds NUM_VOCAB (" + std::to_string(NUM_VOCAB) + ")");
  }

  if (!embedding_weight_cached_) {
    throw std::runtime_error("lookupEmbedding: embedding weight not cached. "
                             "Ensure load_weight() has been called.");
  }

  std::vector<float> result(DIM);

  if (embedding_weight_dtype_ == nntrainer::TensorDim::DataType::FP32) {
    const float *data = embedding_weight_cache_.data();
    std::copy(data + static_cast<size_t>(token_id) * DIM,
              data + static_cast<size_t>(token_id + 1) * DIM, result.begin());
  } else if (embedding_weight_dtype_ == nntrainer::TensorDim::DataType::Q4_0) {
    const void *row_ptr =
      embedding_weight_cache_uint8_.data() + embedding_row_bytes_ * token_id;
    nntrainer::dequantize_row_q4_0(row_ptr, result.data(), DIM);
  } else if (embedding_weight_dtype_ == nntrainer::TensorDim::DataType::Q6_K) {
    const void *row_ptr =
      embedding_weight_cache_uint8_.data() + embedding_row_bytes_ * token_id;
    nntrainer::dequantize_row_q6_K(row_ptr, result.data(), DIM);
  } else {
    throw std::runtime_error("lookupEmbedding: unsupported embedding dtype");
  }

  // Apply embedding scale (consistent with EmbeddingLayer). The image
  // embeddings bypass this: HF scales only the TEXT token embeddings
  // (GemmaTextScaledWordEmbedding) and injects the projector output raw.
  if (EMBEDDING_SCALE != 1.0f) {
    for (auto &v : result)
      v *= EMBEDDING_SCALE;
  }

  return result;
}

size_t PaliGemmaCausalLM::embeddingBytesPerToken() const {
  if (!embedding_weight_cached_) {
    return 0;
  }
  // The composer writes one row per token slot into the merged embedding
  // plane: text rows come from lookupEmbedding() (dequantized + scaled to
  // DIM floats) and image rows are the producer's raw FP32 output of the
  // same width (HF injects the projector output unscaled).
  return static_cast<size_t>(DIM) * sizeof(float);
}

const void *PaliGemmaCausalLM::lookupEmbedding(int token_id) const {
  if (token_id < 0 || static_cast<unsigned int>(token_id) >= NUM_VOCAB ||
      !embedding_weight_cached_) {
    return nullptr;
  }
  lookup_scratch_ = lookupEmbedding(static_cast<unsigned int>(token_id));
  return lookup_scratch_.data();
}

void PaliGemmaCausalLM::run(const WSTR prompt, bool do_sample,
                            const WSTR system_prompt, const WSTR tail_prompt,
                            bool log_output) {
  if (!USE_EMBEDDING) {
    CausalLM::run(prompt, do_sample, system_prompt, tail_prompt, log_output);
    return;
  }

  if (!is_initialized) {
    throw std::runtime_error(
      "PaliGemmaCausalLM model is not initialized. Please call "
      "initialize() before run().");
  }

  if (!embedding_weight_cached_) {
    throw std::runtime_error("PaliGemmaCausalLM::run: embedding weight not "
                             "cached. Ensure load_weight() has been called.");
  }

  const std::string full_prompt = system_prompt + prompt + tail_prompt;
  if (log_output)
    std::cout << full_prompt << std::endl;

  ensureTokenizer(); // join the async load
  // add_special_tokens=true prepends <bos> (TemplateProcessing post-processor)
  auto token_ids = tokenizer->Encode(full_prompt, /*add_special_tokens=*/true);

  // Clamp to INIT_SEQ_LEN (the model's prefill capacity)
  const size_t max_tokens = static_cast<size_t>(INIT_SEQ_LEN);
  if (token_ids.size() > max_tokens)
    token_ids.resize(max_tokens);

  const size_t n_tokens = token_ids.size();

  // Build flat embedding buffer: [n_tokens * DIM]. Text-only prompt: every
  // row goes through lookupEmbedding() (scaled). The image-conditional
  // caller builds [emb(bos), image embeds, emb(text)...] itself and calls
  // run_with_embeddings() with the prefix length instead.
  std::vector<float> inputs_embeds(n_tokens * static_cast<size_t>(DIM));
  for (size_t t = 0; t < n_tokens; ++t) {
    const unsigned int tok = static_cast<unsigned int>(token_ids[t]);
    std::vector<float> emb = lookupEmbedding(tok);
    std::copy(emb.begin(), emb.end(),
              inputs_embeds.begin() + static_cast<ptrdiff_t>(t * DIM));
  }

  // Collect token IDs as seed_tokens for repetition penalty tracking
  std::vector<int> seed_tokens(token_ids.begin(), token_ids.end());

  // Text-only prompt: treat the whole prompt as the prefix block.
  run_with_embeddings(inputs_embeds.data(), n_tokens, n_tokens, seed_tokens,
                      do_sample, log_output);
}

void PaliGemmaCausalLM::run_with_embeddings(const void *inputs_embeds,
                                            size_t n_tokens, size_t prefix_len,
                                            std::vector<int> seed_tokens,
                                            bool do_sample, bool log_output) {

  auto start_total = std::chrono::high_resolution_clock::now();
  /** The peak belongs to THIS request -- see causal_lm.cpp. */
  FootprintSampler::get().arm();

  if (!is_initialized) {
    throw std::runtime_error(
      "PaliGemmaCausalLM model is not initialized. Please call "
      "initialize() before run_with_embeddings().");
  }
  if (!USE_EMBEDDING) {
    throw std::runtime_error(
      "run_with_embeddings() requires USE_EMBEDDING=true. "
      "Set use_embedding=true in nntr_config.");
  }
  if (prefix_len > n_tokens)
    throw std::invalid_argument("prefix_len (" + std::to_string(prefix_len) +
                                ") exceeds n_tokens (" +
                                std::to_string(n_tokens) + ")");

  const auto *embeds = static_cast<const float *>(inputs_embeds);
  const unsigned int input_len = static_cast<unsigned int>(n_tokens);
  const unsigned int prefix = static_cast<unsigned int>(prefix_len);
  if (BATCH_SIZE != 1)
    throw std::invalid_argument(
      "PaliGemma embedding generation requires batch_size=1");
  if (global_token_len >= MAX_SEQ_LEN ||
      n_tokens >= static_cast<size_t>(MAX_SEQ_LEN - global_token_len) ||
      seed_tokens.size() > MAX_SEQ_LEN)
    throw std::invalid_argument(
      "PaliGemma prompt leaves no room for a generated token");

  if (input_len > static_cast<unsigned int>(INIT_SEQ_LEN)) {
    throw std::invalid_argument("n_tokens (" + std::to_string(input_len) +
                                ") exceeds INIT_SEQ_LEN (" +
                                std::to_string(INIT_SEQ_LEN) + ")");
  }

  if (MAX_SEQ_LEN < INIT_SEQ_LEN) {
    throw std::invalid_argument(
      "MAX_SEQ_LEN must be greater than or equal to INIT_SEQ_LEN");
  }

  // run_with_embeddings can be driven directly (mm driver, quick API) with
  // Encode() never called, so join the async tokenizer load here —
  // registerOutputs() dereferences the tokenizer unconditionally.
  ensureTokenizer();

  // Same per-run seed rule as CausalLM::run().
  beginSamplingRun();

  // Allocate the host-owned KV cache and bind it to mha_core's external cache
  // input slots. Idempotent: only the first call does work.
  allocateAndBindKVCache();

  output_list.clear();
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    output_list.push_back("");
  }

  // Store seed token IDs into ids_history for repetition penalty support
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (size_t i = 0; i < seed_tokens.size(); ++i) {
      ids_history[b * MAX_SEQ_LEN + i] =
        static_cast<unsigned int>(seed_tokens[i]);
    }
  }

  /**
   * Build inference inputs: [input_data, kv_cache_pointers...]
   * KV cache pointers are sorted by name for consistent ordering.
   */
  std::vector<uint16_t> half_input;
  auto build_inference_inputs = [this, &half_input](float *input_data,
                                                    unsigned int active_rows) {
    if (HALF_ACTIVATIONS) {
      half_input.resize(static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * DIM);
      std::transform(input_data,
                     input_data + static_cast<size_t>(active_rows) * DIM,
                     half_input.begin(), nntrainer::compute_fp32_to_fp16);
      input_data = reinterpret_cast<float *>(half_input.data());
    }
    std::vector<std::pair<std::string, float *>> cache_inputs;
    cache_inputs.reserve(static_cast<size_t>(NUM_LAYERS) * 2);
    for (int i = 0; i < NUM_LAYERS; ++i) {
      cache_inputs.emplace_back(
        "cache_k_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
      cache_inputs.emplace_back(
        "cache_v_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
    }
    std::sort(
      cache_inputs.begin(), cache_inputs.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    std::vector<float *> inference_inputs;
    inference_inputs.reserve(1 + cache_inputs.size());
    inference_inputs.push_back(input_data);
    for (const auto &cache_input : cache_inputs)
      inference_inputs.push_back(cache_input.second);
    return inference_inputs;
  };

  /**
   * Variables for Log
   */
  unsigned int generation_cnt = 0;

  if (n_tokens == 0)
    throw std::invalid_argument("run_with_embeddings: n_tokens must be > 0");

  auto start_prefill = std::chrono::high_resolution_clock::now();

  /**
   * PREFILL — the HF PaliGemma prefix-LM mask, realized in two phases:
   *
   *   1. PREFIX BLOCK: rows [0, prefix) of the prompt (bos + image embeds)
   *      go through ONE bidirectional forward (is_causal=false): every
   *      prefix row attends every prefix row — the bidirectional block of
   *      the HF mask. mha_core fills the KV cache rows [from, to) and the
   *      graph output carries the LAST computed row's logits.
   *   2. TEXT SUFFIX: rows [prefix, n_tokens) go one row per forward step
   *      (seq_len == 1): the step's query attends cache rows [0, from+1),
   *      i.e. the whole prefix plus the suffix rows so far — the causal
   *      suffix of the HF mask. The last suffix row's logits seed
   *      generation.
   *
   * A prefix-only prompt (prefix == n_tokens) takes its logits straight
   * from the single prefix forward.
   */
  std::vector<float> first_logits;
  const unsigned int prefill_from = global_token_len;
  const size_t plane_size =
    static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * DIM;

  if (prefix > 0) {
    std::vector<float> padded_embeds(plane_size, 0.0f);
    const size_t copy_size = static_cast<size_t>(BATCH_SIZE) * prefix * DIM;
    std::copy(embeds, embeds + copy_size, padded_embeds.begin());

    auto input = build_inference_inputs(padded_embeds.data(), prefix);
    std::vector<float *> label;

    setKVCachePosition(prefill_from);
    std::vector<float *> outputs = incrementalInference(
      BATCH_SIZE, input, prefix, prefill_from, prefill_from + prefix);
    if (prefix == input_len) {
      // PaliGemmaHead projects only the final active row into output row zero.
      ensureHostLogits(outputs[0]);
      first_logits.assign(outputs[0], outputs[0] + NUM_VOCAB);
    }
    for (auto *out : outputs)
      releaseLogitsBuf(out);
  }

  std::vector<float> step_plane(plane_size, 0.0f);
  for (unsigned int pos = prefix; pos < input_len; ++pos) {
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      std::copy(embeds + (static_cast<size_t>(b) * input_len + pos) * DIM,
                embeds + (static_cast<size_t>(b) * input_len + pos + 1) * DIM,
                step_plane.data() +
                  static_cast<size_t>(b) * INIT_SEQ_LEN * DIM);
    }

    auto input = build_inference_inputs(step_plane.data(), 1);
    std::vector<float *> label;

    const unsigned int from = prefill_from + pos;
    std::vector<float *> outputs =
      incrementalInference(BATCH_SIZE, input, 1, from, from + 1);
    if (pos + 1 == input_len) {
      ensureHostLogits(outputs[0]);
      first_logits.assign(outputs[0], outputs[0] + NUM_VOCAB);
    }
    for (auto *out : outputs)
      releaseLogitsBuf(out);
  }

  auto finish_prefill = std::chrono::high_resolution_clock::now();
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);

  /**
   * TOKEN GENERATION — the first token from the last prompt row's logits,
   * then per-token decode steps exactly like LFM2's loop.
   */
  generated_ids_.clear();
  std::vector<bool> eos_list(BATCH_SIZE, false);
  if (log_output && std::getenv("NNTR_PALIGEMMA_DEBUG")) {
    // e2e diagnosis: input-plane sanity + first prefill logits vs the HF
    // golden top-5.
    std::cout << "[paligemma-debug] input plane row0[0..4]:";
    for (int i = 0; i < 5; ++i)
      std::cout << " " << embeds[i];
    std::cout << " row_last[0..4]:";
    for (int i = 0; i < 5; ++i)
      std::cout << " "
                << embeds[(static_cast<size_t>(input_len) - 1) * DIM + i];
    std::cout << std::endl;
    // e2e diagnosis: first prefill logits vs the HF golden top-5.
    std::vector<std::pair<float, unsigned int>> top;
    for (unsigned int v = 0; v < NUM_VOCAB; ++v)
      top.emplace_back(first_logits[v], v);
    std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                      [](auto &a, auto &b) { return a.first > b.first; });
    std::cout << "[paligemma-debug] first_logits top5:";
    for (int i = 0; i < 5; ++i)
      std::cout << " (" << top[i].second << "," << top[i].first << ")";
    std::cout << " max="
              << *std::max_element(first_logits.begin(), first_logits.end())
              << std::endl;
  }
  prefill_logits_ = first_logits;
  std::vector<unsigned int> id_list(
    generate(first_logits.data(), do_sample, 1.0f, ids_history, input_len));

  generated_ids_.push_back(id_list[0]);
  registerOutputs(tokenizer, id_list, input_len, eos_list, log_output);

  // Update ids_history with the first generated token
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    ids_history[b * MAX_SEQ_LEN + input_len] = id_list[b];
  }

  const bool first_is_eos = std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(),
                                      id_list[0]) != EOS_TOKEN_ID.end();
  auto start_generation = std::chrono::high_resolution_clock::now();

  // Same window/budget guard as Lfm2CausalLM::run_with_embeddings().
  const unsigned int generation_budget =
    NUM_TO_GENERATE > 0 ? static_cast<unsigned int>(NUM_TO_GENERATE)
                        : MAX_SEQ_LEN;
  const unsigned int generation_begin = input_len + 1;
  const unsigned int available_rows = MAX_SEQ_LEN - global_token_len;
  const unsigned int generation_end =
    generation_begin < available_rows
      ? generation_begin +
          std::min(available_rows - generation_begin, generation_budget - 1)
      : generation_begin;

  std::vector<float> gen_input(plane_size, 0.0f);
  for (unsigned int token_generation_idx = generation_begin;
       token_generation_idx < generation_end && !first_is_eos;
       ++token_generation_idx) {

    // Look up embedding for the last generated token and fill gen_input
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      std::vector<float> embed = lookupEmbedding(id_list[b]);
      std::copy(embed.begin(), embed.end(),
                gen_input.data() + static_cast<size_t>(b) * INIT_SEQ_LEN * DIM);
    }

    auto input = build_inference_inputs(gen_input.data(), 1);
    allocateAndBindKVCache();

    // The 4th argument is the declared seq capacity of the input plane, not
    // this step's row count — LFM2's decode loop passes input_len here too.
    auto output_interval = incrementalInference(
      BATCH_SIZE, input, input_len, token_generation_idx - 1 + global_token_len,
      token_generation_idx + global_token_len);

    id_list = generate(output_interval[0], do_sample);
    generated_ids_.push_back(id_list[0]);

    registerOutputs(tokenizer, id_list, token_generation_idx, eos_list,
                    log_output);
    ++generation_cnt;

    // Update ids_history with the generated token
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      ids_history[b * MAX_SEQ_LEN + token_generation_idx] = id_list[b];
    }

    for (auto *out : output_interval) {
      releaseLogitsBuf(out);
    }

    // Check EOS
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j] && (std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(),
                                     id_list[j]) != EOS_TOKEN_ID.end())) {
        eos_list[j] = true;
      }
    }

    bool is_finish = true;
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j]) {
        is_finish = false;
        break;
      }
    }

    if (is_finish) {
      break;
    }
  }

  global_token_len += (generation_cnt + input_len);

  auto finish_generation = std::chrono::high_resolution_clock::now();
  auto generation_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(finish_generation -
                                                          start_generation);

  auto finish_total = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_total - start_total);
  const size_t maxrss_kb = getPeakMemoryKb();
  const size_t honest_kb = FootprintSampler::get().finish();
  const size_t peak_memory = resolvePeakMemoryKb(honest_kb, maxrss_kb);

  if (log_output) {

    std::cout << "\n\n";
    std::cout << "=================[ LLM with NNTrainer ]===================\n";
    std::cout << "prefill: " << input_len << " tokens, "
              << prefill_duration.count() << " ms, "
              << ((double)input_len / prefill_duration.count() * 1000)
              << " TPS\n";
    std::cout << "generation: " << generation_cnt << " tokens, "
              << generation_duration.count() << " ms, "
              << ((double)generation_cnt / generation_duration.count() * 1000)
              << " TPS\n";
    std::cout << "total: " << total_duration.count() << " ms\n";
    std::cout << "peak memory: " << peak_memory << " KB\n";
    std::cout << "  (honest RssAnon+accel: " << honest_kb
              << " KB, ru_maxrss: " << maxrss_kb << " KB)\n";
    std::cout << "==========================================================\n";
  }

  performance_metrics.prefill_tokens = input_len;
  performance_metrics.prefill_duration_ms = prefill_duration.count();
  performance_metrics.generation_tokens = generation_cnt;
  performance_metrics.generation_duration_ms = generation_duration.count();
  performance_metrics.total_duration_ms = total_duration.count();
  performance_metrics.peak_memory_kb = peak_memory;
  performance_metrics.peak_rss_kb = maxrss_kb;

  has_run_ = true;
}

void PaliGemmaCausalLM::load_weight(const std::string &weight_path) {
  Transformer::load_weight(weight_path);

  if (USE_EMBEDDING) {
    loadEmbeddingWeight();
    if (causallm_engine() == "cpu" && TIE_WORD_EMBEDDINGS &&
        LMHEAD_DTYPE == "FP32" && !HALF_ACTIVATIONS &&
        embedding_weight_dtype_ == nntrainer::Tdatatype::FP32 && bind_cpu_head_)
      bind_cpu_head_(&embedding_weight_cache_);
  }
}

void PaliGemmaCausalLM::loadEmbeddingWeight() {
  if (EMBEDDING_BIN_PATH.empty()) {
    throw std::runtime_error(
      "loadEmbeddingWeight: embedding_bin_path is not set. "
      "Add embedding_bin_path to nntr_config.");
  }

  // Determine dtype from EMBEDDING_DTYPE string
  if (EMBEDDING_DTYPE == "FP32" || EMBEDDING_DTYPE == "fp32") {
    embedding_weight_dtype_ = nntrainer::TensorDim::DataType::FP32;
  } else if (EMBEDDING_DTYPE == "Q4_0" || EMBEDDING_DTYPE == "q4_0") {
    embedding_weight_dtype_ = nntrainer::TensorDim::DataType::Q4_0;
  } else if (EMBEDDING_DTYPE == "Q6_K" || EMBEDDING_DTYPE == "q6_K") {
    embedding_weight_dtype_ = nntrainer::TensorDim::DataType::Q6_K;
  } else {
    throw std::runtime_error(
      "loadEmbeddingWeight: unsupported embedding dtype: " + EMBEDDING_DTYPE);
  }

  std::ifstream file(EMBEDDING_BIN_PATH, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error(
      "loadEmbeddingWeight: failed to open embedding bin: " +
      EMBEDDING_BIN_PATH);
  }

  if (embedding_weight_dtype_ == nntrainer::TensorDim::DataType::FP32) {
    size_t expected_size = static_cast<size_t>(NUM_VOCAB) * DIM;
    size_t file_size = static_cast<size_t>(file.tellg());
    if (file_size != expected_size * sizeof(float)) {
      throw std::runtime_error(
        "loadEmbeddingWeight: FP32 file size mismatch. Expected " +
        std::to_string(expected_size * sizeof(float)) + " bytes, got " +
        std::to_string(file_size) + " bytes.");
    }
    embedding_weight_cache_.resize(expected_size);
    nntrainer::hintCpuHugePages(embedding_weight_cache_.data(),
                                expected_size * sizeof(float));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(embedding_weight_cache_.data()),
              static_cast<std::streamsize>(expected_size * sizeof(float)));
  } else {
    // Compute row bytes for quantized types (consistent with
    // EmbeddingLayer::incremental_forwarding)
    if (embedding_weight_dtype_ == nntrainer::TensorDim::DataType::Q4_0) {
      int num_blocks = (DIM + 32 - 1) / 32;
      embedding_row_bytes_ = static_cast<size_t>(18 * num_blocks);
    } else { // Q6_K
      int num_blocks = (DIM + 256 - 1) / 256;
      embedding_row_bytes_ = static_cast<size_t>(210 * num_blocks);
    }

    size_t total_bytes = embedding_row_bytes_ * NUM_VOCAB;
    size_t file_size = static_cast<size_t>(file.tellg());
    if (file_size != total_bytes) {
      throw std::runtime_error(
        "loadEmbeddingWeight: quantized file size mismatch. Expected " +
        std::to_string(total_bytes) + " bytes, got " +
        std::to_string(file_size) + " bytes.");
    }
    embedding_weight_cache_uint8_.resize(total_bytes);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(embedding_weight_cache_uint8_.data()),
              static_cast<std::streamsize>(total_bytes));
  }

  embedding_weight_cached_ = true;
}

void PaliGemmaCausalLM::prefillEmbeddings(const float *inputs_embeds,
                                          size_t n_tokens) {
  if (!inputs_embeds || n_tokens == 0 ||
      n_tokens > static_cast<size_t>(INIT_SEQ_LEN))
    throw std::invalid_argument(
      "prefillEmbeddings: invalid input or prefill capacity");
  beginSamplingRun();
  ensureTokenizer();
  if (!PREFILL_ONLY) {
    allocateAndBindKVCache();
    setKVCachePosition(0);
  }
  std::vector<float> padded(
    static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * DIM, 0.0f);
  std::copy_n(inputs_embeds, n_tokens * DIM, padded.begin());
  std::vector<uint16_t> half_input;
  float *data = padded.data();
  if (HALF_ACTIVATIONS) {
    half_input.resize(padded.size());
    std::transform(padded.begin(), padded.end(), half_input.begin(),
                   nntrainer::compute_fp32_to_fp16);
    data = reinterpret_cast<float *>(half_input.data());
  }
  std::vector<float *> inputs{data};
  if (!PREFILL_ONLY)
    for (int i = 0; i < NUM_LAYERS; ++i) {
      inputs.push_back(
        reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
      inputs.push_back(
        reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
    }
  auto outputs =
    incrementalInference(BATCH_SIZE, inputs, n_tokens, 0, n_tokens);
  for (auto *out : outputs)
    releaseLogitsBuf(out);
}

causallm::KVCacheManager &PaliGemmaCausalLM::kvCache() { return kv_cache; }
} // namespace causallm
