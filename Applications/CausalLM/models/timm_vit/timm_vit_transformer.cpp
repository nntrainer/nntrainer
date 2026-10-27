// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   timm_vit_transformer.cpp
 * @date   28 Jan 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief   This timm_vit_transformer.cpp constructs a class for timm ViT model
 * compatible with the PyTorch timm library.
 */

#include "timm_vit_transformer.h"
#include <app_context.h>
#include <engine.h>
#include <xi_pooling.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.inc"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <factory.h>
#include <fstream>
#include <iomanip>
#include <llm_util.hpp>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace causallm {

/**
 * @brief Resize an interleaved image buffer with bilinear interpolation.
 */
static std::vector<unsigned char> resizeImage(const unsigned char *src,
                                              int src_w, int src_h,
                                              int channels, int dst_w,
                                              int dst_h) {
  std::vector<unsigned char> dst(dst_w * dst_h * channels);
  float x_ratio = static_cast<float>(src_w) / static_cast<float>(dst_w);
  float y_ratio = static_cast<float>(src_h) / static_cast<float>(dst_h);

  for (int y = 0; y < dst_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      float px = x * x_ratio;
      float py = y * y_ratio;
      int x0 = static_cast<int>(std::floor(px));
      int y0 = static_cast<int>(std::floor(py));
      int x1 = std::min(x0 + 1, src_w - 1);
      int y1 = std::min(y0 + 1, src_h - 1);
      float fx = px - x0;
      float fy = py - y0;

      for (int c = 0; c < channels; ++c) {
        float v00 = src[(y0 * src_w + x0) * channels + c];
        float v10 = src[(y0 * src_w + x1) * channels + c];
        float v01 = src[(y1 * src_w + x0) * channels + c];
        float v11 = src[(y1 * src_w + x1) * channels + c];
        float v0 = v00 * (1.0f - fx) + v10 * fx;
        float v1 = v01 * (1.0f - fx) + v11 * fx;
        dst[(y * dst_w + x) * channels + c] =
          static_cast<unsigned char>(std::round(v0 * (1.0f - fy) + v1 * fy));
      }
    }
  }

  return dst;
}

/**
 * @brief Load, resize, and normalize an image into CHW float data.
 */
static std::vector<float> loadAndPreprocessImage(const std::string &filepath,
                                                 int target_width,
                                                 int target_height,
                                                 bool normalize) {
  int width, height, channels;
  unsigned char *image =
    stbi_load(filepath.c_str(), &width, &height, &channels, STBI_default);
  if (!image) {
    throw std::runtime_error("Failed to load image: " + filepath);
  }

  unsigned char *data = image;
  std::vector<unsigned char> resized_data;
  if (width != target_width || height != target_height) {
    resized_data =
      resizeImage(image, width, height, channels, target_width, target_height);
    data = resized_data.data();
  }

  std::vector<unsigned char> rgb_data;
  unsigned char *rgb = data;
  if (channels == 1) {
    rgb_data.resize(target_width * target_height * 3);
    for (int i = 0; i < target_width * target_height; ++i) {
      rgb_data[i * 3] = data[i];
      rgb_data[i * 3 + 1] = data[i];
      rgb_data[i * 3 + 2] = data[i];
    }
    rgb = rgb_data.data();
  } else if (channels == 4) {
    rgb_data.resize(target_width * target_height * 3);
    for (int i = 0; i < target_width * target_height; ++i) {
      rgb_data[i * 3] = data[i * 4];
      rgb_data[i * 3 + 1] = data[i * 4 + 1];
      rgb_data[i * 3 + 2] = data[i * 4 + 2];
    }
    rgb = rgb_data.data();
  } else if (channels != 3) {
    stbi_image_free(image);
    throw std::runtime_error("Unsupported number of channels: " +
                             std::to_string(channels));
  }

  std::vector<float> output(3 * target_height * target_width);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < target_height; ++y) {
      for (int x = 0; x < target_width; ++x) {
        unsigned char val = rgb[y * target_width * 3 + x * 3 + c];
        float pixel =
          normalize ? (val / 255.0f - 0.5f) / 0.5f : static_cast<float>(val);
        output[c * target_height * target_width + y * target_width + x] = pixel;
      }
    }
  }

  stbi_image_free(image);
  return output;
}

/**
 * @brief Set ViT-specific parameters from model and runtime configs.
 */
void TimmViTTransformer::setupParameters(json &cfg, json &generation_cfg,
                                         json &nntr_cfg) {
  (void)generation_cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");

  NUM_VOCAB = cfg.value("vocab_size", 1000);
  DIM = cfg.value("hidden_size", 768);
  INTERMEDIATE_SIZE = cfg.value("intermediate_size", 3072);
  NUM_LAYERS = cfg.value("num_hidden_layers", 12);
  NUM_HEADS = cfg.value("num_attention_heads", 12);
  HEAD_DIM = cfg.value("head_dim", DIM / NUM_HEADS);
  NUM_KEY_VALUE_HEADS = cfg.value("num_key_value_heads", NUM_HEADS);
  MAX_POSITION_EMBEDDINGS = cfg.value("max_position_embeddings", 196);
  ROPE_THETA = cfg.value("rope_theta", 0);
  TIE_WORD_EMBEDDINGS = cfg.value("tie_word_embeddings", false);
  NORM_EPS = cfg.value("norm_eps", 1e-6);
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  IS_CAUSAL = cfg.value("is_causal", false);
  SLIDING_WINDOW =
    cfg.contains("sliding_window") && !cfg["sliding_window"].is_null()
      ? cfg["sliding_window"].get<unsigned int>()
      : UINT_MAX;

  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", 224);
  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", 224);
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;

  // Input geometry. `img_size` covers the square-image case; models with a
  // non-square input (an audio mel-spectrogram is n_mels x frames) set
  // input_height / input_width instead.
  const unsigned int img_size = cfg.value("img_size", 224);
  INPUT_HEIGHT = cfg.value("input_height", img_size);
  INPUT_WIDTH = cfg.value("input_width", img_size);
  IMG_CHANNELS = cfg.value("in_chans", 3u);

  PATCH_SIZE = cfg.value("patch_size", 16);
  PATCH_STRIDE = cfg.value("patch_stride", PATCH_SIZE);

  // Patch grid = the conv output geometry, floor((L - K) / S) + 1.
  GRID_H = (INPUT_HEIGHT - PATCH_SIZE) / PATCH_STRIDE + 1;
  GRID_W = (INPUT_WIDTH - PATCH_SIZE) / PATCH_STRIDE + 1;
  NUM_PATCHES = cfg.value("num_patches", GRID_H * GRID_W);

  USE_INPUT_NORM = cfg.value("use_input_norm", false);
  INPUT_NORM_EPS = cfg.value("input_norm_eps", 1e-5f);

  NUM_CLASSES = cfg.value("num_classes", 0u);
  POOLING = cfg.value("pooling", std::string("mean"));
  HEAD_SIGMOID = cfg.value("head_sigmoid", false);
  HEAD_NORM_EPS = cfg.value("head_norm_eps", 1e-5f);
  POOLING_HIDDEN_SIZE = cfg.value("pooling_hidden_size", DIM);
  POOLING_NORM_EPS = cfg.value("pooling_norm_eps", 1e-5f);
  HEAD_DTYPE = nntr_cfg.contains("head_dtype")
                 ? nntr_cfg["head_dtype"].get<std::string>()
                 : nntr_cfg.value("lmhead_dtype", std::string("FP32"));

  // MODEL_TENSOR_TYPE is the fallback weight dtype for layers that do not name
  // one themselves. In a ViT that is the patch conv, the LayerNorms, the
  // positional embedding table and the optional input BatchNorm -- none of
  // which have a block-quantized representation. Their shapes are also not
  // 32-divisible in general (a 527-unit classifier, a 4x63 grid), so inheriting
  // a quantized global type makes the graph fail to build. The FC layers and
  // the head carry their own explicit dtype, so pin the weight half to FP32 and
  // keep only the activation half from the config.
  const size_t dash = MODEL_TENSOR_TYPE.find('-');
  const std::string act_type = dash == std::string::npos
                                 ? std::string("FP32")
                                 : MODEL_TENSOR_TYPE.substr(dash + 1);
  MODEL_TENSOR_TYPE = "FP32-" + act_type;
}

/**
 * @brief Create patch embedding and positional embedding layers.
 */
Tensor TimmViTTransformer::createPatchEmbed(Tensor input) {
  const int embed_dim = DIM;

  Tensor src = input;

  // Optional height-axis BatchNorm on the raw input. Axis 2 is the height of
  // the NCHW input, so the running statistics are per input row -- for CED,
  // per mel frequency bin, which is exactly what its BatchNorm2d(n_mels)
  // (applied with frequency permuted into the channel slot) computes.
  if (USE_INPUT_NORM) {
    LayerHandle input_norm(
      createLayer("batch_normalization",
                  {withKey("name", "input_norm"), withKey("axis", "2"),
                   withKey("epsilon", std::to_string(INPUT_NORM_EPS))}));
    src = input_norm(src);
  }

  LayerHandle conv(
    createLayer("conv2d", {withKey("name", "patch_embed/conv"),
                           withKey("kernel_size", {std::to_string(PATCH_SIZE),
                                                   std::to_string(PATCH_SIZE)}),
                           withKey("filters", std::to_string(embed_dim)),
                           withKey("stride", {std::to_string(PATCH_STRIDE),
                                              std::to_string(PATCH_STRIDE)}),
                           withKey("padding", "valid")}));
  Tensor h = conv(src);

  LayerHandle flatten(createLayer(
    "reshape", {withKey("name", "patch_embed/flatten"),
                withKey("target_shape", "1:" + std::to_string(embed_dim) + ":" +
                                          std::to_string(NUM_PATCHES))}));
  h = flatten(h);

  LayerHandle transpose(
    createLayer("permute", {withKey("name", "patch_embed/transpose"),
                            withKey("direction", {1, 3, 2})}));
  h = transpose(h);

  LayerHandle pos_embed(
    createLayer("weight", {withKey("name", "pos_embed/weights"),
                           withKey("dim", "1:1:" + std::to_string(NUM_PATCHES) +
                                            ":" + std::to_string(embed_dim)),
                           withKey("tensor_dtype", "FP32"),
                           withKey("weight_name", "pos_embed")}));
  Tensor pos = pos_embed(input);

  LayerHandle add(createLayer("addition", {withKey("name", "pos_embed/add")}));
  return add({h, pos});
}

/**
 * @brief Create a pre-normalized self-attention block.
 */
Tensor TimmViTTransformer::createAttention(const int layer_id, Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id) + "_";

  LayerHandle norm(createLayer("layer_normalization",
                               {withKey("name", prefix + "attention_norm"),
                                withKey("axis", "3"),
                                withKey("epsilon", std::to_string(NORM_EPS)),
                                withKey("packed", "false")}));
  Tensor normed = norm(input);

  auto q = prefix + "qkv_q", k = prefix + "qkv_k", v = prefix + "qkv_v",
       a = prefix + "attention", o = prefix + "attention_out";

  LayerHandle q_proj(
    createLayer("fully_connected",
                {withKey("name", q), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false"),
                 withKey("weight_dtype", FC_LAYER_DTYPE)}));
  LayerHandle k_proj(
    createLayer("fully_connected",
                {withKey("name", k), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false"),
                 withKey("weight_dtype", FC_LAYER_DTYPE)}));
  LayerHandle v_proj(
    createLayer("fully_connected",
                {withKey("name", v), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false"),
                 withKey("weight_dtype", FC_LAYER_DTYPE)}));

  Tensor query = q_proj(normed);
  Tensor key = k_proj(normed);
  Tensor value = v_proj(normed);

  LayerHandle attention(createLayer(
    "mha_core",
    {withKey("name", a), withKey("num_heads", std::to_string(NUM_HEADS)),
     withKey("num_heads_kv", std::to_string(NUM_HEADS)),
     withKey("max_timestep", std::to_string(NUM_PATCHES + 1)),
     withKey("is_causal", "false"),
     // ViT self-attention is position-encoding free -- positions enter through
     // the learned pos_embed added in createPatchEmbed(). mha_core's use_rope
     // property defaults to true, so it must be disabled explicitly; leaving
     // it on rotates q/k with a theta of 0 and produces NaN.
     withKey("use_rope", "false"),
     withKey("rope_theta", std::to_string(ROPE_THETA))}));
  Tensor context = attention({query, key, value});

  LayerHandle out_proj(
    createLayer("fully_connected",
                {withKey("name", o), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false"),
                 withKey("weight_dtype", FC_LAYER_DTYPE)}));
  return out_proj(context);
}

/**
 * @brief Create a pre-normalized feed-forward block.
 */
Tensor TimmViTTransformer::createMlp(const int layer_id, Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id) + "_";

  LayerHandle norm(
    createLayer("layer_normalization",
                {withKey("name", prefix + "ffn_norm"), withKey("axis", "3"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));
  Tensor h = norm(input);

  LayerHandle fc_up(createLayer(
    "fully_connected", {withKey("name", prefix + "ffn_up"),
                        withKey("unit", std::to_string(INTERMEDIATE_SIZE)),
                        withKey("disable_bias", "false"),
                        withKey("weight_dtype", FC_LAYER_DTYPE)}));
  h = fc_up(h);

  LayerHandle gelu(
    createLayer("activation", {withKey("name", prefix + "ffn_gelu"),
                               withKey("activation", "gelu")}));
  h = gelu(h);

  LayerHandle fc_down(
    createLayer("fully_connected", {withKey("name", prefix + "ffn_down"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false"),
                                    withKey("weight_dtype", FC_LAYER_DTYPE)}));
  return fc_down(h);
}

/**
 * @brief Create one ViT transformer block with residual connections.
 */
Tensor TimmViTTransformer::createTransformerDecoderBlock(const int layer_id,
                                                         Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id) + "_";

  Tensor att_out = createAttention(layer_id, input);
  LayerHandle attention_res(
    createLayer("addition", {withKey("name", prefix + "attention_residual")}));
  Tensor residual = attention_res({input, att_out});

  Tensor mlp_out = createMlp(layer_id, residual);
  LayerHandle ffn_res(
    createLayer("addition", {withKey("name", prefix + "ffn_residual")}));
  return ffn_res({residual, mlp_out});
}

/**
 * @brief Create the optional pooling + LayerNorm + classifier head.
 */
Tensor TimmViTTransformer::createHead(Tensor input) {
  if (NUM_CLASSES == 0) {
    // Feature-extraction configuration: the encoder's final LayerNorm output
    // is the model output, so there is nothing to append.
    return input;
  }

  Tensor h = input;

  // Token pooling. The encoder output is [batch, 1, tokens, dim], so the token
  // axis is 2.
  if (POOLING == "mean") {
    LayerHandle pool(createLayer(
      "reduce_mean", {withKey("name", "head/pool"), withKey("axis", "2")}));
    h = pool(h);
  } else if (POOLING == "xi") {
    // Gaussian posterior inference pooling: a learned per-frame precision
    // decides each frame's weight instead of averaging uniformly.
    LayerHandle pool(
      createLayer("xi_pooling",
                  {withKey("name", "head/pool"),
                   withKey("hidden_size", std::to_string(POOLING_HIDDEN_SIZE)),
                   withKey("epsilon", std::to_string(POOLING_NORM_EPS))}));
    h = pool(h);
  } else if (!POOLING.empty() && POOLING != "none") {
    throw std::invalid_argument("Unsupported pooling mode for ViT head: " +
                                POOLING);
  }

  // The head LayerNorm is a plain nn.LayerNorm, so its epsilon is PyTorch's
  // 1e-5 default -- not the 1e-6 the encoder blocks are built with.
  LayerHandle norm(createLayer(
    "layer_normalization", {withKey("name", "head/norm"), withKey("axis", "3"),
                            withKey("epsilon", std::to_string(HEAD_NORM_EPS)),
                            withKey("packed", "false")}));
  h = norm(h);

  LayerHandle classifier(createLayer(
    "fully_connected",
    {withKey("name", "head/classifier"),
     withKey("unit", std::to_string(NUM_CLASSES)),
     withKey("disable_bias", "false"), withKey("weight_dtype", HEAD_DTYPE)}));
  h = classifier(h);

  if (HEAD_SIGMOID) {
    LayerHandle sigmoid(
      createLayer("activation", {withKey("name", "head/sigmoid"),
                                 withKey("activation", "sigmoid")}));
    h = sigmoid(h);
  }

  return h;
}

/**
 * @brief Construct the symbolic ViT inference graph.
 */
std::pair<Tensor, Tensor> TimmViTTransformer::constructModel() {
  Tensor input({BATCH_SIZE, IMG_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH}, "input0");
  Tensor h = createPatchEmbed(input);

  for (int i = 0; i < NUM_LAYERS; i++) {
    h = createTransformerDecoderBlock(i, h);
  }

  LayerHandle output_norm(
    createLayer("layer_normalization",
                {withKey("name", "output_norm"), withKey("axis", "3"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));
  h = output_norm(h);

  h = createHead(h);

  return {input, h};
}

/**
 * @brief Register layers used by this model.
 */
void TimmViTTransformer::registerCustomLayers() {
  Transformer::registerCustomLayers();

  static std::once_flag registered;
  std::call_once(registered, []() {
    const auto &ct_engine = nntrainer::Engine::Global();
    auto *app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));
    app_context->registerFactory(
      nntrainer::createLayer<causallm::XiPoolingLayer>);
  });
}

/**
 * @brief Run ViT inference on an image file path.
 */
void TimmViTTransformer::run(const WSTR prompt, bool do_sample,
                             const WSTR system_prompt, const WSTR tail_prompt,
                             bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  (void)log_output;

  if (!is_initialized) {
    throw std::runtime_error("TimmViT model is not initialized. Please call "
                             "initialize() before run().");
  }

  unsigned int img_h = INPUT_HEIGHT;
  unsigned int img_w = INPUT_WIDTH;

  std::string image_path_str(prompt);
  std::vector<float> image_data =
    loadAndPreprocessImage(image_path_str, img_w, img_h, true);

  std::vector<float *> input;
  input.push_back(image_data.data());
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);

  std::cout << std::setprecision(9) << "First 10 values: ";
  const int print_count = DIM > 10 ? 10 : static_cast<int>(DIM);
  for (int i = 0; i < print_count; ++i) {
    std::cout << "[" << i << "]=" << output[0][i] << " ";
  }
  std::cout << std::endl;

  const size_t out_count = static_cast<size_t>(NUM_PATCHES) * DIM;

  // Optional dump of the full [NUM_PATCHES, DIM] feature map, for offline
  // comparison against a framework reference.
  if (const char *dump_path = std::getenv("VIT_OUT_BIN")) {
    std::ofstream dump(dump_path, std::ios::binary);
    if (!dump) {
      std::cerr << "[VIT_OUT_BIN] cannot open " << dump_path << std::endl;
    } else {
      dump.write(reinterpret_cast<const char *>(output[0]),
                 static_cast<std::streamsize>(out_count * sizeof(float)));
      std::cout << "[VIT_OUT_BIN] wrote " << out_count << " floats to "
                << dump_path << std::endl;
    }
  }

  // Optional verification against an FP32 reference of the same shape.
  if (const char *ref_path = std::getenv("VIT_REF_BIN")) {
    std::ifstream ref(ref_path, std::ios::binary | std::ios::ate);
    if (!ref) {
      std::cerr << "[VIT_REF_BIN] cannot open " << ref_path << std::endl;
      return;
    }
    const size_t ref_count = static_cast<size_t>(ref.tellg()) / sizeof(float);
    if (ref_count != out_count) {
      std::cerr << "[VIT_REF_BIN] size mismatch: reference has " << ref_count
                << " floats, output has " << out_count << std::endl;
      return;
    }
    ref.seekg(0);
    std::vector<float> ref_data(ref_count);
    ref.read(reinterpret_cast<char *>(ref_data.data()),
             static_cast<std::streamsize>(ref_count * sizeof(float)));

    double max_abs_diff = 0.0;
    double sum_sq_diff = 0.0;
    size_t max_idx = 0;
    size_t nan_count = 0;
    for (size_t i = 0; i < out_count; ++i) {
      const float got = output[0][i];
      // Bit-pattern NaN test: -ffast-math folds std::isnan() to false.
      uint32_t bits;
      std::memcpy(&bits, &got, sizeof(bits));
      if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        ++nan_count;
        continue;
      }
      const double diff = std::fabs(static_cast<double>(got) - ref_data[i]);
      sum_sq_diff += diff * diff;
      if (diff > max_abs_diff) {
        max_abs_diff = diff;
        max_idx = i;
      }
    }
    std::cout << "[VIT_REF_BIN] elements=" << out_count << " nan=" << nan_count
              << " max_abs_diff=" << max_abs_diff << " (at token "
              << max_idx / DIM << ", dim " << max_idx % DIM
              << ") rms_diff=" << std::sqrt(sum_sq_diff / out_count)
              << std::endl;
    std::cout << "[VIT_REF_BIN] "
              << ((nan_count == 0 && max_abs_diff < 1e-3) ? "PASS" : "FAIL")
              << std::endl;
  }
}

} // namespace causallm
