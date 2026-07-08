// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   bert_decoder.cpp
 * @date   18 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  General-purpose BERT decoder — embeddings + causal self-attention
 *         with external KV cache (Task 4).
 *
 * Graph structure (this task):
 *   word_emb + pos_emb + type_emb → Addition → LayerNorm (emb_ln)
 *   → N x [ Q/K/V FC + mha_core(causal, no-rope, ext-KV) + O FC
 *           + Addition + LayerNorm ] (dec_layer{i}_self_*)
 *
 * Weight order matches collect_decoder in weight_converter.py:
 *   word_emb:weight, pos_emb:weight, type_emb:weight,
 *   emb_ln:gamma, emb_ln:beta,
 *   per layer: dec_layer{i}_self_q (w,b), _self_k (w,b), _self_v (w,b),
 *              _self_out (w,b), _self_ln (gamma,beta)
 */

#include <bert_decoder.h>
#include <llm_util.hpp>
#include <model.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace causallm {

namespace {
/**
 * @brief Convert a float to string with full precision (needed for 1e-12
 *        which rounds to 0 with default float-to-string conversion).
 */
std::string toStringPrecise(float v) {
  std::ostringstream oss;
  oss << std::setprecision(20) << v;
  return oss.str();
}
} // anonymous namespace

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------

void BertDecoder::initialize() {
  registerCustomLayers();

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
                      withKey("model_tensor_type", MODEL_TENSOR_TYPE)});

  auto [inputs, output] = constructDecoderGraph();
  std::vector<Tensor> outputs = {output};
  if (model->compile(inputs, outputs, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("BertDecoder: model compilation failed.");
  }

  is_initialized = true;
}

// ---------------------------------------------------------------------------
// allocateAndBindKVCache
// ---------------------------------------------------------------------------

void BertDecoder::allocateAndBindKVCache() {
  if (!kv_cache_.isAllocated()) {
    // dtype matches mha_core's cache placeholders (UINT16 on non-FP16 builds).
#ifdef ENABLE_FP16
    const auto cache_dtype = ml::train::TensorDim::DataType::FP16;
#else
    const auto cache_dtype = ml::train::TensorDim::DataType::UINT16;
#endif

    kv_cache_.allocate(static_cast<unsigned int>(BD_NUM_LAYERS), BD_BATCH_SIZE,
                       BD_MAX_SEQ_LEN, static_cast<unsigned int>(BD_NUM_HEADS),
                       static_cast<unsigned int>(BD_HEAD_DIM), cache_dtype);
    kv_cache_bound_ = false;
  }

  if (kv_cache_bound_)
    return;

  // Bind each (layer, K|V) buffer into the corresponding input layer declared
  // by createKVCachePlaceholders().  Names must match what that function
  // registers with the model.
  for (int i = 0; i < BD_NUM_LAYERS; ++i) {
    auto &kc = kv_cache_.getKeyCache(static_cast<unsigned int>(i));
    auto &vc = kv_cache_.getValueCache(static_cast<unsigned int>(i));

    auto find_cache_placeholder =
      [this](const std::string &base_name) -> nntrainer::Tensor * {
      for (const auto &suffix : {":0", ":input0", ":out0", ""}) {
        auto *t = model->getTensor(base_name + suffix);
        if (t != nullptr)
          return t;
      }
      return nullptr;
    };

    // Primary lookup: mha_core input slots 3 and 4 for self-attention
    auto *kp =
      model->getTensor("dec_layer" + std::to_string(i) + "_self_attn:input3");
    auto *vp =
      model->getTensor("dec_layer" + std::to_string(i) + "_self_attn:input4");
    if (kp == nullptr)
      kp = find_cache_placeholder("cache_k_l" + std::to_string(i));
    if (vp == nullptr)
      vp = find_cache_placeholder("cache_v_l" + std::to_string(i));

    if (kp == nullptr || vp == nullptr) {
      throw std::runtime_error(
        "BertDecoder::allocateAndBindKVCache: cache_k_l" + std::to_string(i) +
        " / cache_v_l" + std::to_string(i) +
        " input placeholder not found in compiled graph");
    }

    kp->setData(kc.getMemoryData(), kc.getOffset(), false);
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }

  kv_cache_bound_ = true;
}

// ---------------------------------------------------------------------------
// setKVCachePosition
// ---------------------------------------------------------------------------

void BertDecoder::setKVCachePosition(unsigned int pos) {
  kv_cache_.setPosition(pos);
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [pos](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == causallm::MHACoreLayer::type)
        l.setProperty({"cache_index=" + std::to_string(pos)});
    };
  model->forEachLayer(fn, nullptr);
}

// ---------------------------------------------------------------------------
// constructDecoderGraph
// ---------------------------------------------------------------------------

std::pair<std::vector<Tensor>, Tensor> BertDecoder::constructDecoderGraph() {

  // ===== Named inputs =====
  Tensor input0({1, 1, 1, BD_INIT_SEQ_LEN}, "input0");
  Tensor position_ids({1, 1, 1, BD_INIT_SEQ_LEN}, "position_ids");
  Tensor token_type_ids({1, 1, 1, BD_INIT_SEQ_LEN}, "token_type_ids");

  // Encoder hidden states [1,1,196,256] — unused until Task 5 cross-attn, but
  // declared here so the input signature remains stable across tasks.
  Tensor encoder_hidden({1, 1, 196, BD_DIM}, "encoder_hidden");

  // ===== Embeddings =====
  // word_emb: in_dim=30522, out_dim=256 (NOT tie_word_embeddings; LM tie Task
  // 6)
  LayerHandle word_emb_layer(createLayer(
    "embedding_layer",
    {withKey("name", "word_emb"), withKey("in_dim", BD_NUM_VOCAB),
     withKey("weight_dtype", EMBEDDING_DTYPE), withKey("out_dim", BD_DIM)}));
  Tensor word = word_emb_layer(input0);

  // pos_emb: in_dim=512, out_dim=256
  LayerHandle pos_emb_layer(createLayer(
    "embedding_layer",
    {withKey("name", "pos_emb"), withKey("in_dim", BD_MAX_POSITION_EMBEDDINGS),
     withKey("weight_dtype", EMBEDDING_DTYPE), withKey("out_dim", BD_DIM)}));
  Tensor pos = pos_emb_layer(position_ids);

  // type_emb: in_dim=2, out_dim=256
  LayerHandle type_emb_layer(createLayer(
    "embedding_layer",
    {withKey("name", "type_emb"), withKey("in_dim", BD_TYPE_VOCAB_SIZE),
     withKey("weight_dtype", EMBEDDING_DTYPE), withKey("out_dim", BD_DIM)}));
  Tensor tok_type = type_emb_layer(token_type_ids);

  // Sum word + position + token_type embeddings
  LayerHandle emb_sum(createLayer("addition", {withKey("name", "emb_sum")}));
  Tensor h = emb_sum({word, pos, tok_type});

  // Embedding LayerNorm (emb_ln): eps=1e-12, axis=3
  LayerHandle emb_ln(createLayer(
    "layer_normalization", {withKey("name", "emb_ln"),
                            withKey("epsilon", toStringPrecise(BD_NORM_EPS)),
                            withKey("axis", 3), withKey("packed", "false")}));
  h = emb_ln(h);

  // ===== Decoder layers =====
  // createSelfAttentionBlock creates KV cache placeholders (cache_k_l{i},
  // cache_v_l{i}) as input leaf nodes inside createKVCachePlaceholders().
  // They are reachable via backward graph traversal from h and do NOT need
  // to be listed in the primary inputs vector.
  // Block order: self-attn(+res+norm) -> cross-attn(+res+norm) -> FFN.
  // Cross-attn cross_cache_k_l{i}/cross_cache_v_l{i} placeholders are leaf
  // input nodes (created in createCrossCachePlaceholders), auto-discovered by
  // the backward graph traversal like the self-attn KV cache.
  for (int i = 0; i < BD_NUM_LAYERS; ++i) {
    h = createSelfAttentionBlock(i, h);
    h = createCrossAttentionBlock(i, h, encoder_hidden);
    h = createFFNBlock(i, h);
  }

  // ===== LM Head =====
  // lmhead_dense: unit=256, bias (maps predictions.transform.dense)
  LayerHandle lmh_dense(
    createLayer("fully_connected",
                {withKey("name", "lmhead_dense"), withKey("unit", BD_DIM),
                 withKey("disable_bias", "false")}));
  h = lmh_dense(h);

  // lmhead_gelu: activation=gelu (maps predictions.transform activation)
  LayerHandle lmh_gelu(
    createLayer("activation", {withKey("name", "lmhead_gelu"),
                               withKey("activation", "gelu")}));
  h = lmh_gelu(h);

  // lmhead_ln: LayerNorm eps=1e-12 (maps predictions.transform.LayerNorm)
  LayerHandle lmh_ln(createLayer(
    "layer_normalization", {withKey("name", "lmhead_ln"),
                            withKey("epsilon", toStringPrecise(BD_NORM_EPS)),
                            withKey("axis", 3), withKey("packed", "false")}));
  h = lmh_ln(h);

  // lm_head_proj: tie_word_embeddings (shares word_emb weights, transpose=true)
  // unit=30522, disable_bias=true, shared_from=word_emb (the embedding layer).
  // The shared weight tensor (word_emb:Embedding) must be requested with the
  // SAME dtype by both sites. tie_word_embeddings otherwise resolves its weight
  // dtype from the model's GLOBAL weight type (e.g. Q4_0), while the embedding
  // layer uses its per-layer EMBEDDING_DTYPE — a mismatch (e.g. embedding Q6_K
  // vs global Q4_0) makes requestOrExtend throw "tensor dimension mismatch for
  // word_emb:Embedding". Pin the tie layer to EMBEDDING_DTYPE so they agree.
  LayerHandle lm_proj(createLayer(
    "tie_word_embeddings",
    {withKey("name", "lm_head_proj"), withKey("unit", BD_NUM_VOCAB),
     withKey("disable_bias", "true"), withKey("weight_dtype", EMBEDDING_DTYPE),
     withKey("shared_from", "word_emb")}));
  Tensor logits = lm_proj(h);

  // lm_head_bias: learnable bias [1,1,1,30522] loaded from weight file
  // The weight_name "lmhead_bias" matches the key written by collect_decoder.
  LayerHandle lmbias(createLayer(
    "weight",
    {withKey("name", "lm_head_bias/weights"),
     withKey("dim", "1:1:1:" + std::to_string(BD_NUM_VOCAB)),
     withKey("tensor_dtype", "FP32"), withKey("weight_name", "lmhead_bias")}));
  Tensor bias_t = lmbias(h);

  // lm_head_bias_add: logits + bias
  LayerHandle addb(
    createLayer("addition", {withKey("name", "lm_head_bias_add")}));
  Tensor logits_out = addb({logits, bias_t});

  // Primary named inputs (KV cache placeholders are leaf nodes,
  // auto-discovered)
  std::vector<Tensor> graph_inputs = {input0, position_ids, token_type_ids,
                                      encoder_hidden};
  return {graph_inputs, logits_out};
}

// ---------------------------------------------------------------------------
// createSelfAttentionBlock
// ---------------------------------------------------------------------------

Tensor BertDecoder::createSelfAttentionBlock(int layer_id, Tensor h) {
  const std::string pfx = "dec_layer" + std::to_string(layer_id);

  // Q projection: unit=256, bias enabled
  // Name: dec_layer{i}_self_q (matches collect_decoder
  // pfx+"_self_q:weight/bias")
  LayerHandle wq(
    createLayer("fully_connected", {withKey("name", pfx + "_self_q"),
                                    withKey("unit", BD_HEAD_DIM * BD_NUM_HEADS),
                                    withKey("disable_bias", "false")}));
  Tensor q = wq(h);

  // K projection: unit=256, bias enabled (no GQA: same as Q dim)
  LayerHandle wk(
    createLayer("fully_connected",
                {withKey("name", pfx + "_self_k"),
                 withKey("unit", BD_HEAD_DIM * BD_NUM_HEADS / BD_GQA_SIZE),
                 withKey("disable_bias", "false")}));
  Tensor k = wk(h);

  // V projection: unit=256, bias enabled
  LayerHandle wv(
    createLayer("fully_connected",
                {withKey("name", pfx + "_self_v"),
                 withKey("unit", BD_HEAD_DIM * BD_NUM_HEADS / BD_GQA_SIZE),
                 withKey("disable_bias", "false")}));
  Tensor v = wv(h);

  // External KV cache placeholders (created as input layers, bound at runtime
  // via allocateAndBindKVCache).  Names: cache_k_l{layer_id},
  // cache_v_l{layer_id}
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, BD_NUM_HEADS);

  // MHA core: causal=true, use_rope=false (CRITICAL: rope_theta=0 +
  // use_rope=true
  //           caused NaN in the encoder — always false here)
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", pfx + "_self_attn"), withKey("num_heads", BD_NUM_HEADS),
     withKey("num_heads_kv", BD_NUM_HEADS / BD_GQA_SIZE),
     withKey("max_timestep", std::to_string(BD_MAX_SEQ_LEN)),
     withKey("rope_theta", ROPE_THETA), withKey("use_rope", "false"),
     withKey("is_causal", "true")}));
  Tensor a = mha({q, k, v, cache_k, cache_v});

  // Output projection (attention.output.dense): unit=256, bias enabled
  LayerHandle wo(
    createLayer("fully_connected",
                {withKey("name", pfx + "_self_out"), withKey("unit", BD_DIM),
                 withKey("disable_bias", "false")}));
  Tensor attn_out = wo(a);

  // Residual: h + attn_out
  LayerHandle res_add(
    createLayer("addition", {withKey("name", pfx + "_self_res")}));
  Tensor h_res = res_add({h, attn_out});

  // Post-attention LayerNorm (attention.output.LayerNorm): eps=1e-12, axis=3
  LayerHandle ln(createLayer("layer_normalization",
                             {withKey("name", pfx + "_self_ln"),
                              withKey("epsilon", toStringPrecise(BD_NORM_EPS)),
                              withKey("axis", 3), withKey("packed", "false")}));

  return ln(h_res);
}

// ---------------------------------------------------------------------------
// createCrossCachePlaceholders
// ---------------------------------------------------------------------------

std::pair<Tensor, Tensor>
BertDecoder::createCrossCachePlaceholders(int layer_id, unsigned int enc_len) {
  const unsigned int kv_width =
    static_cast<unsigned int>(BD_HEAD_DIM * BD_NUM_HEADS / BD_GQA_SIZE);
  const std::string cache_shape = std::to_string(BD_BATCH_SIZE) +
                                  ":1:" + std::to_string(enc_len) + ":" +
                                  std::to_string(kv_width);
#ifdef ENABLE_FP16
  // ARM/Android FP16 build. Previously these were bare Tensors (getType() !=
  // "input"), so they were NOT collected into runDecoderForward's
  // ordered_input_names and the prefilled cross-cache buffer was never fed to
  // the model — unlike the self-attn KV cache, the cross-cache has no separate
  // setData() binding, so cross-attn read uninitialized/zero memory and the
  // decode path became image-agnostic. Create real "input" LAYER nodes (FP16)
  // so getType()=="input", they are collected, and they receive the prefilled
  // cross-cache buffer through the input vector — exactly as on desktop.
  // mapExternalTensor maps FP16 and UINT16 buffers identically (raw uint16_t,
  // no conversion), so the fp16-bit buffer flows through unchanged and matches
  // how mha_core reads the FP16 self-attn KV cache in cross_attention=true
  // mode.
  const char *cross_cache_dtype = "FP16";
#else
  // Desktop (non-FP16) build: UINT16 storage holding fp16 bits, same as the
  // self-attn KV cache encoding.
  const char *cross_cache_dtype = "UINT16";
#endif
  LayerHandle cross_k_input(createLayer(
    "input", {withKey("name", "cross_cache_k_l" + std::to_string(layer_id)),
              withKey("input_shape", cache_shape),
              withKey("input_dtype", cross_cache_dtype)}));
  LayerHandle cross_v_input(createLayer(
    "input", {withKey("name", "cross_cache_v_l" + std::to_string(layer_id)),
              withKey("input_shape", cache_shape),
              withKey("input_dtype", cross_cache_dtype)}));
  return {cross_k_input(Tensor()), cross_v_input(Tensor())};
}

// ---------------------------------------------------------------------------
// createCrossAttentionBlock
// ---------------------------------------------------------------------------

Tensor BertDecoder::createCrossAttentionBlock(int layer_id, Tensor h,
                                              Tensor encoder_hidden) {
  const std::string pfx = "dec_layer" + std::to_string(layer_id);

  // Q projection (bias) on the post-self-attn-norm decoder hidden.
  LayerHandle wq(
    createLayer("fully_connected", {withKey("name", pfx + "_cross_q"),
                                    withKey("unit", BD_HEAD_DIM * BD_NUM_HEADS),
                                    withKey("disable_bias", "false")}));
  Tensor q = wq(h);

  // K / V projections (bias) on the encoder hidden states. Their outputs fill
  // the cross-cache once (mechanism finalized in Task 7).
  LayerHandle wk(
    createLayer("fully_connected",
                {withKey("name", pfx + "_cross_k"),
                 withKey("unit", BD_HEAD_DIM * BD_NUM_HEADS / BD_GQA_SIZE),
                 withKey("disable_bias", "false")}));
  Tensor k = wk(encoder_hidden);

  LayerHandle wv(
    createLayer("fully_connected",
                {withKey("name", pfx + "_cross_v"),
                 withKey("unit", BD_HEAD_DIM * BD_NUM_HEADS / BD_GQA_SIZE),
                 withKey("disable_bias", "false")}));
  Tensor v = wv(encoder_hidden);

  // Static read-only cross-cache placeholders (input leaf nodes).
  auto [cross_k, cross_v] = createCrossCachePlaceholders(layer_id, BD_ENC_LEN);

  // MHA core in external-cache mode, cross_attention=true: query attends over
  // all BD_ENC_LEN cached rows; cache is neither written nor advanced here.
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", pfx + "_cross_attn"), withKey("num_heads", BD_NUM_HEADS),
     withKey("num_heads_kv", BD_NUM_HEADS / BD_GQA_SIZE),
     withKey("max_timestep", std::to_string(BD_ENC_LEN)),
     withKey("rope_theta", ROPE_THETA), withKey("use_rope", "false"),
     withKey("is_causal", "false"), withKey("cross_attention", "true")}));
  Tensor a = mha({q, k, v, cross_k, cross_v});

  // Output projection (crossattention.output.dense): bias enabled.
  LayerHandle wo(
    createLayer("fully_connected",
                {withKey("name", pfx + "_cross_out"), withKey("unit", BD_DIM),
                 withKey("disable_bias", "false")}));
  Tensor attn_out = wo(a);

  // Residual: h + attn_out
  LayerHandle res_add(
    createLayer("addition", {withKey("name", pfx + "_cross_res")}));
  Tensor h_res = res_add({h, attn_out});

  // Output LayerNorm (crossattention.output.LayerNorm): eps=1e-12, axis=3
  LayerHandle ln(createLayer("layer_normalization",
                             {withKey("name", pfx + "_cross_ln"),
                              withKey("epsilon", toStringPrecise(BD_NORM_EPS)),
                              withKey("axis", 3), withKey("packed", "false")}));

  return ln(h_res);
}

// ---------------------------------------------------------------------------
// decodeStep  (parity test helper)
// ---------------------------------------------------------------------------

void BertDecoder::allocateCrossCacheBuffers() {
  const size_t cross_size =
    static_cast<size_t>(BD_ENC_LEN) *
    static_cast<size_t>(BD_HEAD_DIM * BD_NUM_HEADS / BD_GQA_SIZE);
  if (cross_cache_k_.empty()) {
    cross_cache_k_.assign(BD_NUM_LAYERS, std::vector<uint16_t>(cross_size, 0));
    cross_cache_v_.assign(BD_NUM_LAYERS, std::vector<uint16_t>(cross_size, 0));
  }
}

// ---------------------------------------------------------------------------
// runDecoderForward  (shared by prefillCrossCache + decodeStep)
// ---------------------------------------------------------------------------

std::vector<float *> BertDecoder::runDecoderForward(
  const float *enc_hidden, int token_id, std::vector<uint16_t> *cross_k_bufs,
  std::vector<uint16_t> *cross_v_bufs, unsigned int pos) {
  allocateAndBindKVCache();
  // cache_index = pos: the self-attn K/V for this step are written at slot pos,
  // so the cache grows across the autoregressive loop (decodeStep uses pos=0).
  setKVCachePosition(pos);

  std::vector<float> in_token_buf(1, static_cast<float>(token_id));
  std::vector<float> in_pos_buf(1, static_cast<float>(pos));
  std::vector<float> in_type_buf(1, 0.0f);

  // name -> ptr lookup
  std::unordered_map<std::string, float *> name_to_ptr;
  name_to_ptr["input0"] = in_token_buf.data();
  name_to_ptr["position_ids"] = in_pos_buf.data();
  name_to_ptr["token_type_ids"] = in_type_buf.data();
  name_to_ptr["encoder_hidden"] = const_cast<float *>(enc_hidden);
  for (int i = 0; i < BD_NUM_LAYERS; ++i) {
    name_to_ptr["cache_k_l" + std::to_string(i)] = reinterpret_cast<float *>(
      kv_cache_.getKeyCache(static_cast<unsigned int>(i)).getData());
    name_to_ptr["cache_v_l" + std::to_string(i)] = reinterpret_cast<float *>(
      kv_cache_.getValueCache(static_cast<unsigned int>(i)).getData());
    // Cross-cache buffers are 16-bit storage holding fp16 bits (UINT16 dtype on
    // desktop, FP16 dtype on ARM — see createCrossCachePlaceholders /
    // prefillCrossCache for why the dtype must diverge so copyData encodes real
    // fp16 on both platforms). They are fed through the input vector exactly
    // like the self-attn KV cache; the cross-cache input LAYER nodes (created
    // in createCrossCachePlaceholders) are collected below via
    // getType()=="input". mha_core reads them in cross_attention=true
    // (read-only) mode.
    name_to_ptr["cross_cache_k_l" + std::to_string(i)] =
      reinterpret_cast<float *>(cross_k_bufs[i].data());
    name_to_ptr["cross_cache_v_l" + std::to_string(i)] =
      reinterpret_cast<float *>(cross_v_bufs[i].data());
  }

  // Collect ordered input layer names (same order incremental_inference
  // expects)
  std::vector<std::string> ordered_input_names;
  {
    std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &,
                       void *)>
      collect_fn =
        [&](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
          if (l.getType() == "input")
            ordered_input_names.push_back(l.getName());
        };
    model->forEachLayer(collect_fn, nullptr);
  }

  std::vector<float *> inference_inputs;
  inference_inputs.reserve(ordered_input_names.size());
  for (const auto &n : ordered_input_names) {
    auto it = name_to_ptr.find(n);
    if (it == name_to_ptr.end())
      throw std::runtime_error("runDecoderForward: unknown input name '" + n +
                               "'");
    inference_inputs.push_back(it->second);
  }

  std::vector<float *> labels;
  return model->incremental_inference(BD_BATCH_SIZE, inference_inputs, labels,
                                      1, pos, pos + 1, false);
}

// ---------------------------------------------------------------------------
// prefillCrossCache
// ---------------------------------------------------------------------------
//
// Populates the per-layer cross-attention K/V cache with the REAL encoder K/V
// = encoder_hidden @ Wk_cross(+bk) / @ Wv_cross(+bv).
//
// Mechanism (direct-weight projection — no throwaway forward):
//   For each layer, compute K = encoder_hidden @ Wk_cross + bk and
//   V = encoder_hidden @ Wv_cross + bv directly from the layer's persistent
//   _cross_k / _cross_v FC weights (the same input.dot(weight)+add_i(bias) math
//   FullyConnectedLayer::forwarding performs), then store the FP32 result into
//   the cross-cache buffers via Tensor::copyData — the EXACT same FP32 ->
//   fp16/UINT16 encoding and per-row head layout the self-attn KV-cache write
//   path uses. No hand-encoded fp16 bits, no layout guesswork.
//
// Computing from persistent weights (rather than reading the _cross_k/_cross_v
// FC output tensors after a decoder forward) is required: under INFERENCE
// memory planning those intermediate output buffers are aliased across layers,
// so a post-forward read returns the wrong layer's data. Loaded weights never
// alias. No throwaway forward is run, so token_id is unused here.
//
// After this call the buffers hold persistent encoder K/V; the subsequent
// decode forwards read them in cross_attention=true (read-only) mode.
//
// This is the reusable mechanism Task 7's orchestrator calls once per image.
void BertDecoder::prefillCrossCache(const float *enc_hidden, int token_id) {
  if (!is_initialized)
    throw std::runtime_error("BertDecoder::prefillCrossCache: not initialized");

  (void)token_id; // direct-weight path needs no throwaway forward
  allocateCrossCacheBuffers();

  const unsigned int kv_width =
    static_cast<unsigned int>(BD_HEAD_DIM * BD_NUM_HEADS / BD_GQA_SIZE);

  // encoder_hidden as an FP32 tensor [1,1,196,256]. dot() below reproduces the
  // FC layer math exactly (FullyConnectedLayer::forwarding does
  // input.dot(weight, false, false) then += bias).
  ml::train::TensorDim enc_dim(
    {BD_BATCH_SIZE, 1, BD_ENC_LEN, static_cast<unsigned int>(BD_DIM)},
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP32});
  nntrainer::Tensor enc = nntrainer::Tensor::Map<float>(
    const_cast<float *>(enc_hidden),
    static_cast<unsigned int>(BD_ENC_LEN * BD_DIM * sizeof(float)), enc_dim, 0);

  // Destination cache tensor dtype MUST drive copyData() through the proper
  // FP32 -> fp16 *bit conversion*, matching how mha_core reads the cache and
  // how the self-attn KV-cache write path encodes fp16.
  //
  // Platform divergence (the actual ARM bug, beyond the input-vector wiring):
  //   * x86 (UINT16 dst): copy_fp32_u16 -> avx2::copy_f32_f16 -> real
  //     FP32->FP16 conversion. Correct fp16 bits (e.g. -2.043 -> 0xC016).
  //   * ARM (UINT16 dst): copy_fp32_u16 -> __fallback_copy_fp32_u16 does
  //     `Y[i] = static_cast<uint16_t>(X[i])` — an INTEGER truncation, NOT an
  //     fp16 conversion. Small projections (|x|<1) collapse to 0/1, so the
  //     cross-cache was effectively zero and cross-attn read ~0 on device.
  // Using an FP16 dst on ARM routes copyData() through HalfTensor::copyData ->
  // scopy_fp32_to_fp16 (real fp16 conversion), exactly like the self-attn
  // KV-cache write. The buffer storage stays 16-bit (uint16_t) either way.
#ifdef ENABLE_FP16
  const auto cross_cache_store_dtype = ml::train::TensorDim::DataType::FP16;
#else
  const auto cross_cache_store_dtype = ml::train::TensorDim::DataType::UINT16;
#endif
  ml::train::TensorDim cache_dim(
    {BD_BATCH_SIZE, 1, BD_ENC_LEN, kv_width},
    {ml::train::TensorDim::Format::NCHW, cross_cache_store_dtype});

  // Compute K/V = encoder_hidden @ W(+b) directly from each layer's persistent
  // _cross_k / _cross_v FC weights, then store into the cross-cache buffers via
  // Tensor::copyData (FP32 -> fp16/UINT16) — the EXACT encoding + per-row head
  // layout (num_heads * head_dim contiguous) the self-attn KV-cache write uses.
  //
  // Computing from persistent weights (not from intermediate FC output tensors)
  // is deliberate: under INFERENCE memory planning the graph aliases the
  // cross_k/_cross_v output buffers across layers, so reading them back after a
  // forward returns the wrong layer's data. The loaded weights never alias.
  auto project = [&](const std::string &fc, nntrainer::Tensor &dst_cache) {
    nntrainer::Tensor *w = model->getTensor(fc + ":weight");
    nntrainer::Tensor *b = model->getTensor(fc + ":bias");
    if (w == nullptr || b == nullptr)
      throw std::runtime_error("prefillCrossCache: weight/bias for " + fc +
                               " not found in compiled graph");

    // Pre-allocate the FP32 output. For an FP32 weight, Tensor::dot ->
    // dotFloat allocates `proj` itself (via calculateFlattenDot). For a
    // quantized weight (Q4_0/Q6_K/Q4_K), Tensor::dot routes to dotQnK, which
    // writes directly into output.getData<float>() WITHOUT allocating it —
    // passing a default-constructed (empty) tensor segfaults in the quantized
    // GEMM. Allocate proj = [1,1,BD_ENC_LEN,kv_width] FP32 up front so both the
    // FP32 and the quantized paths have a valid destination buffer.
    nntrainer::TensorDim proj_dim(
      {BD_BATCH_SIZE, 1, BD_ENC_LEN, kv_width},
      {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP32});
    nntrainer::Tensor proj(proj_dim, true /* allocate */);
    enc.dot(*w, proj, false, false);
    proj.add_i(*b); // broadcast bias over rows
    dst_cache.copyData(proj);
  };

  for (int i = 0; i < BD_NUM_LAYERS; ++i) {
    const std::string pfx = "dec_layer" + std::to_string(i);
    nntrainer::Tensor k_cache = nntrainer::Tensor::Map<uint16_t>(
      cross_cache_k_[i].data(),
      static_cast<unsigned int>(cross_cache_k_[i].size() * sizeof(uint16_t)),
      cache_dim, 0);
    nntrainer::Tensor v_cache = nntrainer::Tensor::Map<uint16_t>(
      cross_cache_v_[i].data(),
      static_cast<unsigned int>(cross_cache_v_[i].size() * sizeof(uint16_t)),
      cache_dim, 0);
    project(pfx + "_cross_k", k_cache);
    project(pfx + "_cross_v", v_cache);
  }

  cross_cache_prefilled_ = true;
}

// ---------------------------------------------------------------------------
// decodeToken  (autoregressive single-step decode)
// ---------------------------------------------------------------------------

std::vector<float> BertDecoder::decodeToken(const float *enc_hidden,
                                            int token_id, unsigned int pos) {
  if (!is_initialized)
    throw std::runtime_error("BertDecoder::decodeToken: not initialized");
  if (!cross_cache_prefilled_)
    throw std::runtime_error(
      "BertDecoder::decodeToken: prefillCrossCache() must be called once per "
      "image before decoding");

  auto out = runDecoderForward(enc_hidden, token_id, cross_cache_k_.data(),
                               cross_cache_v_.data(), pos);
  if (out.empty() || out[0] == nullptr)
    throw std::runtime_error(
      "BertDecoder::decodeToken: incremental_inference returned null");

  std::vector<float> logits(out[0], out[0] + BD_NUM_VOCAB);
  for (auto *p : out) {
    if (p != nullptr)
      delete[] p;
  }
  return logits;
}

// ---------------------------------------------------------------------------
// decodeStep  (parity test helper)
// ---------------------------------------------------------------------------

int BertDecoder::decodeStep(const float *enc_hidden, int token_id,
                            const std::string &out_npy) {
  if (!is_initialized)
    throw std::runtime_error("BertDecoder::decodeStep: not initialized");

  // Populate the cross-attention K/V cache with the real encoder K/V once,
  // then run the decode reading that populated cache (read-only).
  prefillCrossCache(enc_hidden, token_id);

  auto out = runDecoderForward(enc_hidden, token_id, cross_cache_k_.data(),
                               cross_cache_v_.data());

  if (out.empty() || out[0] == nullptr)
    throw std::runtime_error("decodeStep: incremental_inference returned null");

  const int vocab = static_cast<int>(BD_NUM_VOCAB);

  // Compute argmax
  int argmax_idx = 0;
  float maxv = out[0][0];
  for (int k = 1; k < vocab; ++k) {
    if (out[0][k] > maxv) {
      maxv = out[0][k];
      argmax_idx = k;
    }
  }
  std::cout << "[decodeStep] nntrainer argmax = " << argmax_idx
            << " (val=" << maxv << ")\n";
  std::cout << "[decodeStep] logits[0..3]: " << out[0][0] << " " << out[0][1]
            << " " << out[0][2] << " " << out[0][3] << "\n";

  // Sanity: the layer-0 cross-attn output (mha output, fed into _cross_out FC)
  // must be NONZERO and image-dependent now that the cross-cache holds real
  // encoder K/V. With the old zero-filled cross-cache it was bias-only.
  if (auto *xo = model->getTensor("dec_layer0_cross_out:input0")) {
    const float *d = xo->getData<float>();
    const size_t n = xo->getDim().getFeatureLen();
    float amax = 0.0f, asum = 0.0f;
    for (size_t j = 0; j < n; ++j) {
      float a = std::abs(d[j]);
      amax = a > amax ? a : amax;
      asum += a;
    }
    std::cout << "[decodeStep] layer0 cross-attn output: max|x|=" << amax
              << " mean|x|=" << (n ? asum / n : 0.0f) << " first4=" << d[0]
              << " " << d[1] << " " << d[2] << " " << d[3] << "\n";
  }

  // Save logits to .npy
  if (!out_npy.empty()) {
    std::ofstream npy(out_npy, std::ios::binary);
    if (!npy)
      throw std::runtime_error("decodeStep: cannot open output npy: " +
                               out_npy);
    const char magic[] = "\x93NUMPY\x01\x00";
    npy.write(magic, 8);
    std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 1, " +
      std::to_string(vocab) + "), }";
    header += '\n';
    while ((8 + 2 + header.size()) % 64 != 0)
      header.insert(header.size() - 1, 1, ' ');
    uint16_t hlen = static_cast<uint16_t>(header.size());
    npy.write(reinterpret_cast<const char *>(&hlen), 2);
    npy.write(header.c_str(), static_cast<std::streamsize>(header.size()));
    npy.write(reinterpret_cast<const char *>(out[0]),
              static_cast<std::streamsize>(vocab * sizeof(float)));
    std::cout << "[decodeStep] saved logits to " << out_npy << "\n";
  }

  for (auto *p : out) {
    if (p)
      delete[] p;
  }
  return argmax_idx;
}

// ---------------------------------------------------------------------------
// createFFNBlock
// ---------------------------------------------------------------------------

Tensor BertDecoder::createFFNBlock(int layer_id, Tensor h) {
  const std::string pfx = "dec_layer" + std::to_string(layer_id);

  // Intermediate (gate) dense: unit=1024, bias enabled (intermediate.dense)
  // Name: dec_layer{i}_ffn_inter (matches collect_decoder
  // pfx+"_ffn_inter:weight/bias")
  LayerHandle ffn_inter(
    createLayer("fully_connected", {withKey("name", pfx + "_ffn_inter"),
                                    withKey("unit", BD_INTERMEDIATE_SIZE),
                                    withKey("disable_bias", "false")}));
  Tensor inter = ffn_inter(h);

  // GELU activation (matches BERT intermediate activation)
  LayerHandle ffn_gelu(
    createLayer("activation", {withKey("name", pfx + "_ffn_gelu"),
                               withKey("activation", "gelu")}));
  inter = ffn_gelu(inter);

  // Output dense: unit=256, bias enabled (output.dense)
  // Name: dec_layer{i}_ffn_out (matches collect_decoder
  // pfx+"_ffn_out:weight/bias")
  LayerHandle ffn_out(
    createLayer("fully_connected",
                {withKey("name", pfx + "_ffn_out"), withKey("unit", BD_DIM),
                 withKey("disable_bias", "false")}));
  Tensor ffn_out_t = ffn_out(inter);

  // Residual: h + ffn_out
  LayerHandle res_add(
    createLayer("addition", {withKey("name", pfx + "_ffn_res")}));
  Tensor h_res = res_add({h, ffn_out_t});

  // Post-FFN LayerNorm (output.LayerNorm): eps=1e-12, axis=3
  // Name: dec_layer{i}_ffn_ln (matches collect_decoder
  // pfx+"_ffn_ln:gamma/beta")
  LayerHandle ln(createLayer("layer_normalization",
                             {withKey("name", pfx + "_ffn_ln"),
                              withKey("epsilon", toStringPrecise(BD_NORM_EPS)),
                              withKey("axis", 3), withKey("packed", "false")}));

  return ln(h_res);
}

// ---------------------------------------------------------------------------
// smokeTest
// ---------------------------------------------------------------------------

bool BertDecoder::smokeTest() {
  std::cout << "[smokeTest] entered\n";
  std::cout.flush();
  if (!is_initialized) {
    std::cerr << "[smokeTest] FAIL: not initialized\n";
    return false;
  }

  std::cout << "[smokeTest] Allocating + binding KV cache...\n";
  std::cout.flush();
  try {
    allocateAndBindKVCache();
  } catch (const std::exception &e) {
    std::cerr << "[smokeTest] FAIL allocate/bind: " << e.what() << "\n";
    return false;
  }
  std::cout << "[smokeTest] KV cache allocated + bound (" << BD_NUM_LAYERS
            << " layers, kv_width=" << BD_NUM_HEADS * BD_HEAD_DIM / BD_GQA_SIZE
            << ")\n";
  std::cout.flush();

  // Set cache position to 0 (first token)
  setKVCachePosition(0);

  // Build a name -> pointer map, then order by model's getInputDimension().
  // CRITICAL: do NOT sort alphabetically — the model registers inputs in a
  // specific order (primary inputs first, then KV cache leaf nodes in DFS
  // order).  Using getInputDimension() to query the model's registered input
  // names ensures each buffer is passed to the correct slot.
  //
  // Sizes match the model input dims:
  //   input0, position_ids, token_type_ids : [1,1,1,1] -> 1 float each
  //   encoder_hidden : [1,1,196,256]       -> 196*256 floats
  //   cache_k/v_l{i} : [1,1,512,256]       -> 512*256 floats (from kv_cache_)
  std::vector<float> in_token_buf(1, 101.0f); // [CLS]
  std::vector<float> in_pos_buf(1, 0.0f);
  std::vector<float> in_type_buf(1, 0.0f);
  std::vector<float> in_enc(static_cast<size_t>(196) * BD_DIM, 0.0f);

  // Dummy zero-filled cross-attention K/V cache buffers (UINT16 placeholders,
  // 196 x 256 elements => 2 bytes each). Task 7 fills these from the encoder
  // K/V projections; for the self-attn smoke we just bind valid storage so the
  // graph runs end-to-end. One shared zero buffer per K and V is sufficient
  // since the smoke only checks the self-attn path does not regress.
  std::vector<uint16_t> cross_k_dummy(static_cast<size_t>(196) * BD_DIM, 0);
  std::vector<uint16_t> cross_v_dummy(static_cast<size_t>(196) * BD_DIM, 0);

  // Build name -> ptr lookup table
  std::unordered_map<std::string, float *> name_to_ptr;
  name_to_ptr["input0"] = in_token_buf.data();
  name_to_ptr["position_ids"] = in_pos_buf.data();
  name_to_ptr["token_type_ids"] = in_type_buf.data();
  name_to_ptr["encoder_hidden"] = in_enc.data();
  for (int i = 0; i < BD_NUM_LAYERS; ++i) {
    name_to_ptr["cache_k_l" + std::to_string(i)] = reinterpret_cast<float *>(
      kv_cache_.getKeyCache(static_cast<unsigned int>(i)).getData());
    name_to_ptr["cache_v_l" + std::to_string(i)] = reinterpret_cast<float *>(
      kv_cache_.getValueCache(static_cast<unsigned int>(i)).getData());
    name_to_ptr["cross_cache_k_l" + std::to_string(i)] =
      reinterpret_cast<float *>(cross_k_dummy.data());
    name_to_ptr["cross_cache_v_l" + std::to_string(i)] =
      reinterpret_cast<float *>(cross_v_dummy.data());
  }

  // Enumerate model input layers in their registered order using forEachLayer.
  // This avoids alphabetical sort mismatch — the model registers:
  //   (1) primary inputs first (input0, position_ids, token_type_ids,
  //       encoder_hidden) in the order passed to compile(),
  //   (2) then KV cache input layers in DFS topological order.
  // We collect only "input"-type layers (those with no non-empty input_layers).
  std::vector<std::string> ordered_input_names;
  {
    std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &,
                       void *)>
      collect_inputs =
        [&](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
          if (l.getType() == "input") {
            ordered_input_names.push_back(l.getName());
          }
        };
    model->forEachLayer(collect_inputs, nullptr);
  }

  auto in_dims = model->getInputDimension();
  std::cout << "[smokeTest] model input slots (" << ordered_input_names.size()
            << "):\n";
  std::vector<float *> inference_inputs;
  inference_inputs.reserve(ordered_input_names.size());
  for (size_t idx = 0; idx < ordered_input_names.size(); ++idx) {
    const std::string &n = ordered_input_names[idx];
    const auto &dim =
      idx < in_dims.size() ? in_dims[idx] : ml::train::TensorDim{};
    std::cout << "[smokeTest]   [" << idx << "] " << n << " dim=" << dim
              << "\n";
    auto it = name_to_ptr.find(n);
    if (it == name_to_ptr.end()) {
      std::cerr << "[smokeTest] FAIL: unknown input name '" << n << "'\n";
      return false;
    }
    inference_inputs.push_back(it->second);
  }
  std::cout.flush();

  std::vector<float *> labels;
  std::cout
    << "[smokeTest] Running incremental_inference(seq_len=1, pos=0)...\n";
  std::cout.flush();

  {
    auto out = model->incremental_inference(BD_BATCH_SIZE, inference_inputs,
                                            labels, 1, 0, 1, false);
    std::cout << "[smokeTest] incremental_inference OK, output ptrs: "
              << out.size() << "\n";
    std::cout.flush();
    for (auto *p : out) {
      if (p != nullptr)
        delete[] p;
    }
  }

  std::cout << "[smokeTest] BertDecoder PASSED\n";
  std::cout.flush();
  return true;
}

} // namespace causallm
