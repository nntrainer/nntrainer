// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   siglip_vision.cpp
 * @brief  SiglipVisionModel — the vision half of PaliGemma (tower +
 *         multimodal projector), producing LLM-space image embeddings.
 * @date   12 Nov 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "siglip_vision.h"
#include "native_linear.h"
#include "prefix_attention.h"
// Declaration-only include: the stb_image implementation is compiled once in
// timm_vit_transformer.cpp (one STB_IMAGE_IMPLEMENTATION translation unit per
// target; defining it here too would duplicate its symbols at link time).
#include "stb_image.inc"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <factory.h>
#include <iomanip>
#include <llm_util.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace causallm {
namespace {
struct CubicWeights {
  int first;
  std::vector<int> weights;
};

std::vector<CubicWeights> cubicWeights(int input, int output) {
  constexpr double precision = 4194304.0;
  const double scale = static_cast<double>(input) / output;
  const double stretch = std::max(1.0, scale);
  std::vector<CubicWeights> table(output);
  for (int i = 0; i < output; ++i) {
    const double center = (i + 0.5) * scale;
    const int first = std::max(0, static_cast<int>(center - 2 * stretch + 0.5));
    const int end =
      std::min(input, static_cast<int>(center + 2 * stretch + 0.5));
    std::vector<double> weights(end - first);
    double sum = 0;
    for (int j = first; j < end; ++j) {
      const double x = std::abs((j + 0.5 - center) / stretch);
      const double w = x < 1 ? 1 + x * x * (1.5 * x - 2.5)
                             : (x < 2 ? 2 + x * (-4 + x * (2.5 - 0.5 * x)) : 0);
      weights[j - first] = w;
      sum += w;
    }
    table[i].first = first;
    for (double w : weights)
      table[i].weights.push_back(
        static_cast<int>(std::round(w / sum * precision)));
  }
  return table;
}

std::vector<unsigned char> resizeRGB(const unsigned char *input, int width,
                                     int height, int size) {
  const auto horizontal = cubicWeights(width, size);
  const auto vertical = cubicWeights(height, size);
  std::vector<unsigned char> temp(static_cast<size_t>(height) * size * 3);
  std::vector<unsigned char> output(static_cast<size_t>(size) * size * 3);
  auto clip = [](int value) {
    return static_cast<unsigned char>(std::clamp(value / 4194304, 0, 255));
  };
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < size; ++x)
      for (int c = 0; c < 3; ++c) {
        int accum = 2097152;
        const auto &filter = horizontal[x];
        for (size_t k = 0; k < filter.weights.size(); ++k)
          accum +=
            input[(static_cast<size_t>(y) * width + filter.first + k) * 3 + c] *
            filter.weights[k];
        temp[(static_cast<size_t>(y) * size + x) * 3 + c] = clip(accum);
      }
  for (int y = 0; y < size; ++y)
    for (int x = 0; x < size; ++x)
      for (int c = 0; c < 3; ++c) {
        int accum = 2097152;
        const auto &filter = vertical[y];
        for (size_t k = 0; k < filter.weights.size(); ++k)
          accum +=
            temp[((filter.first + k) * size + x) * 3 + c] * filter.weights[k];
        output[(static_cast<size_t>(y) * size + x) * 3 + c] = clip(accum);
      }
  return output;
}

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
SiglipVisionModel::createModelLayer(const std::string &type,
                                    std::vector<std::string> properties) {
  if (WEIGHT_TRANSPOSED && type == "fully_connected") {
    properties.emplace_back(withKey("engine", causallm_engine()));
    std::shared_ptr<ml::train::Layer> node =
      nntrainer::createLayerNode(std::make_unique<NativeLinear>(), properties);
    return LayerHandle(std::move(node));
  }
  return createStandardModelLayer(type, std::move(properties));
}

json &SiglipVisionModel::sanitizeVisionConfig(json &cfg) {
  if (causallm_engine() == "cuda") {
    configurePaliGemmaCuda();
  }

  // PaliGemma's top level carries the TEXT side's dims (hidden_size 2048,
  // projection_dim 2048); the tower's real dims live in vision_config. For
  // the vision graph, vision_config values must WIN over the top level —
  // the reverse of the text decoder's merge.
  if (cfg.contains("vision_config") && cfg["vision_config"].is_object()) {
    for (auto it = cfg["vision_config"].begin();
         it != cfg["vision_config"].end(); ++it) {
      cfg[it.key()] = it.value();
    }
  }
  if (!cfg.contains("layer_norm_eps"))
    cfg["layer_norm_eps"] = 1e-6;
  // SiglipVisionConfig default hidden_act — the config file omits it.
  if (!cfg.contains("hidden_act"))
    cfg["hidden_act"] = "gelu_pytorch_tanh";
  // image_size is not stored in PaliGemma's vision_config: recover it from
  // the patch count (num_image_tokens = (image_size / patch_size)^2).
  if (!cfg.contains("image_size") && cfg.contains("num_image_tokens") &&
      cfg.contains("patch_size")) {
    const unsigned int patches = cfg["num_image_tokens"].get<unsigned int>();
    const unsigned int side = static_cast<unsigned int>(
      std::lround(std::sqrt(static_cast<double>(patches))));
    cfg["image_size"] = side * cfg["patch_size"].get<unsigned int>();
  }
  if (!cfg.contains("num_patches") && cfg.contains("image_size") &&
      cfg.contains("patch_size")) {
    const unsigned int side = cfg["image_size"].get<unsigned int>() /
                              cfg["patch_size"].get<unsigned int>();
    cfg["num_patches"] = side * side;
  }
  return cfg;
}

void SiglipVisionModel::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  (void)generation_cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");

  NUM_VOCAB = cfg.value("vocab_size", 0);
  DIM = cfg.value("hidden_size", 1152);
  INTERMEDIATE_SIZE = cfg.value("intermediate_size", 4304);
  NUM_LAYERS = cfg.value("num_hidden_layers", 27);
  NUM_HEADS = cfg.value("num_attention_heads", 16);
  HEAD_DIM = cfg.value("head_dim", DIM / NUM_HEADS);
  NUM_KEY_VALUE_HEADS = cfg.value("num_key_value_heads", NUM_HEADS);
  MAX_POSITION_EMBEDDINGS = cfg.value("max_position_embeddings", 257);
  ROPE_THETA = cfg.value("rope_theta", 0);
  TIE_WORD_EMBEDDINGS = cfg.value("tie_word_embeddings", false);
  NORM_EPS = cfg.value("layer_norm_eps", 1e-6);
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  IS_CAUSAL = cfg.value("is_causal", false);
  SLIDING_WINDOW =
    cfg.contains("sliding_window") && !cfg["sliding_window"].is_null()
      ? cfg["sliding_window"].get<unsigned int>()
      : UINT_MAX;

  WEIGHT_TRANSPOSED = nntr_cfg.value("weight_transposed", false);
  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", 256);
  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", 256);
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;

  IMG_SIZE = cfg.value("image_size", 224);
  PATCH_SIZE = cfg.value("patch_size", 14);
  NUM_PATCHES = cfg.value("num_patches", 256);
  IMG_CHANNELS = 3;
  // The projector maps vision_hidden -> LLM hidden; projection_dim sits at
  // the top level of PaliGemma's config.json (not inside vision_config).
  PROJECTION_DIM = cfg.value("projection_dim", DIM);
}

Tensor SiglipVisionModel::createPatchEmbed(Tensor input) {
  const int embed_dim = DIM;

  LayerHandle conv(createModelLayer(
    "conv2d", {withKey("name", "patch_embed/conv"),
               withKey("kernel_size", {std::to_string(PATCH_SIZE),
                                       std::to_string(PATCH_SIZE)}),
               withKey("filters", std::to_string(embed_dim)),
               withKey("stride", {std::to_string(PATCH_SIZE),
                                  std::to_string(PATCH_SIZE)}),
               withKey("padding", "valid")}));
  Tensor h = conv(input);

  LayerHandle flatten(createModelLayer(
    "reshape", {withKey("name", "patch_embed/flatten"),
                withKey("target_shape", "1:" + std::to_string(embed_dim) + ":" +
                                          std::to_string(NUM_PATCHES))}));
  h = flatten(h);

  LayerHandle transpose(
    createModelLayer("permute", {withKey("name", "patch_embed/transpose"),
                                 withKey("direction", {1, 3, 2})}));
  h = transpose(h);

  LayerHandle pos_embed(createModelLayer(
    "weight",
    {withKey("name", "pos_embed/weights"),
     withKey("dim", "1:1:" + std::to_string(NUM_PATCHES) + ":" +
                      std::to_string(embed_dim)),
     withKey("tensor_dtype", "FP32"), withKey("weight_name", "pos_embed")}));
  Tensor pos = pos_embed(input);

  LayerHandle add(
    createModelLayer("addition", {withKey("name", "pos_embed/add")}));
  return add({h, pos});
}

Tensor SiglipVisionModel::createAttention(const int layer_id, Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id) + "_";

  LayerHandle norm(createModelLayer(
    "layer_normalization",
    {withKey("name", prefix + "attention_norm"), withKey("axis", "3"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = norm(input);

  auto q = prefix + "qkv_q", k = prefix + "qkv_k", v = prefix + "qkv_v",
       a = prefix + "attention", o = prefix + "attention_out";

  LayerHandle q_proj(
    createModelLayer("fully_connected",
                     {withKey("name", q), withKey("unit", std::to_string(DIM)),
                      withKey("disable_bias", "false")}));
  LayerHandle k_proj(
    createModelLayer("fully_connected",
                     {withKey("name", k), withKey("unit", std::to_string(DIM)),
                      withKey("disable_bias", "false")}));
  LayerHandle v_proj(
    createModelLayer("fully_connected",
                     {withKey("name", v), withKey("unit", std::to_string(DIM)),
                      withKey("disable_bias", "false")}));

  Tensor query = q_proj(normed);
  Tensor key = k_proj(normed);
  Tensor value = v_proj(normed);

  LayerHandle attention(
    createPrefixAttention(a, NUM_HEADS, NUM_HEADS, HEAD_DIM, 0.0f, nullptr));
  Tensor context = attention({query, key, value});

  LayerHandle out_proj(
    createModelLayer("fully_connected",
                     {withKey("name", o), withKey("unit", std::to_string(DIM)),
                      withKey("disable_bias", "false")}));
  return out_proj(context);
}

Tensor SiglipVisionModel::createMlp(const int layer_id, Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id) + "_";

  LayerHandle norm(createModelLayer(
    "layer_normalization",
    {withKey("name", prefix + "ffn_norm"), withKey("axis", "3"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor h = norm(input);

  LayerHandle fc_up(createModelLayer(
    "fully_connected", {withKey("name", prefix + "ffn_up"),
                        withKey("unit", std::to_string(INTERMEDIATE_SIZE)),
                        withKey("disable_bias", "false")}));
  h = fc_up(h);

  // SiglipVisionConfig hidden_act = gelu_pytorch_tanh == "tanh_gelu"
  LayerHandle gelu(
    createModelLayer("activation", {withKey("name", prefix + "ffn_gelu"),
                                    withKey("activation", "tanh_gelu")}));
  h = gelu(h);

  LayerHandle fc_down(
    createModelLayer("fully_connected", {withKey("name", prefix + "ffn_down"),
                                         withKey("unit", std::to_string(DIM)),
                                         withKey("disable_bias", "false")}));
  return fc_down(h);
}

Tensor SiglipVisionModel::createTransformerDecoderBlock(const int layer_id,
                                                        Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id) + "_";

  Tensor att_out = createAttention(layer_id, input);
  LayerHandle attention_res(createModelLayer(
    "addition", {withKey("name", prefix + "attention_residual")}));
  Tensor residual = attention_res({input, att_out});

  Tensor mlp_out = createMlp(layer_id, residual);
  LayerHandle ffn_res(
    createModelLayer("addition", {withKey("name", prefix + "ffn_residual")}));
  return ffn_res({residual, mlp_out});
}

std::pair<Tensor, Tensor> SiglipVisionModel::constructModel() {
  Tensor input({BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE}, "input0");
  Tensor h = createPatchEmbed(input);

  for (int i = 0; i < NUM_LAYERS; i++) {
    h = createTransformerDecoderBlock(i, h);
  }

  // SiglipVisionModel applies post_layernorm to the final hidden states
  // (no pooling head: vision_use_head = false).
  LayerHandle output_norm(
    createModelLayer("layer_normalization",
                     {withKey("name", "output_norm"), withKey("axis", "3"),
                      withKey("epsilon", std::to_string(NORM_EPS)),
                      withKey("packed", "false")}));
  h = output_norm(h);

  // Multi-modal projector: Linear(vision_hidden -> projection_dim, bias).
  LayerHandle projector(createModelLayer(
    "fully_connected", {withKey("name", "multi_modal_projector"),
                        withKey("unit", std::to_string(PROJECTION_DIM)),
                        withKey("disable_bias", "false")}));
  h = projector(h);

  return {input, h};
}

void SiglipVisionModel::registerCustomLayers() {
  Transformer::registerCustomLayers();
}

std::vector<float>
SiglipVisionModel::preprocessImage(const std::string &path) const {
  int width, height, channels;
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image(
    stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb),
    stbi_image_free);
  if (!image)
    throw std::runtime_error("Failed to load image: " + path);
  const auto resized = resizeRGB(image.get(), width, height, IMG_SIZE);
  const size_t plane = static_cast<size_t>(IMG_SIZE) * IMG_SIZE;
  std::vector<float> result(3 * plane);
  for (size_t i = 0; i < plane; ++i)
    for (size_t c = 0; c < 3; ++c)
      result[c * plane + i] =
        (static_cast<float>(resized[i * 3 + c]) / 255.0f - 0.5f) / 0.5f;
  return result;
}

/**
 * @brief Run vision inference on an image file path.
 */
void SiglipVisionModel::run(const WSTR prompt, bool do_sample,
                            const WSTR system_prompt, const WSTR tail_prompt,
                            bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  (void)log_output;

  if (!is_initialized) {
    throw std::runtime_error("SiglipVisionModel is not initialized. Please "
                             "call initialize() before run().");
  }

  auto image_data = preprocessImage(std::string(prompt));

  std::vector<float *> input;
  input.push_back(image_data.data());
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);

  std::cout << std::setprecision(9) << "First 10 values: ";
  const int print_count = static_cast<int>(PROJECTION_DIM) > 10
                            ? 10
                            : static_cast<int>(PROJECTION_DIM);
  for (int i = 0; i < print_count; ++i) {
    std::cout << "[" << i << "]=" << output[0][i] << " ";
  }
  std::cout << std::endl;

  for (auto *out : output)
    delete[] out;
}

multimodal_pointer
SiglipVisionModel::run_image(const WSTR prompt, multimodal_pointer image,
                             int image_height, int image_width, bool do_sample,
                             const WSTR system_prompt, const WSTR tail_prompt,
                             bool log_output) {
  (void)image_height;
  (void)image_width;
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  (void)log_output;

  if (!is_initialized) {
    throw std::runtime_error("SiglipVisionModel is not initialized. Please "
                             "call initialize() before run_image().");
  }

  const size_t expected_floats =
    static_cast<size_t>(IMG_CHANNELS) * IMG_SIZE * IMG_SIZE;
  std::vector<float> image_data;
  if (image.first && image.second == expected_floats * sizeof(float)) {
    const auto *pixels = static_cast<const float *>(image.first);
    image_data.assign(pixels, pixels + expected_floats);
  } else {
    image_data = preprocessImage(std::string(prompt));
  }

  std::vector<float *> input;
  input.push_back(image_data.data());
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);

  // One batch row: [NUM_PATCHES, PROJECTION_DIM] floats — LLM-space image
  // embeddings.
  const size_t floats =
    static_cast<size_t>(NUM_PATCHES) * static_cast<size_t>(PROJECTION_DIM);
  float *embeds = new float[floats];
  std::copy(output[0], output[0] + floats, embeds);
  for (auto *out : output)
    delete[] out;

  return {embeds, floats * sizeof(float)};
}

} // namespace causallm
