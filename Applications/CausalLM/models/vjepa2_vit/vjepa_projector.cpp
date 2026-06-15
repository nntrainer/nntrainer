// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   vjepa_projector.cpp
 * @date   1 June 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  V-JEPA 2.1 Projector / Merger (vision hidden → LLM embedding space).
 */

#include "vjepa_projector.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <app_context.h>
#include <engine.h>
#include <factory.h>
#include <llm_util.hpp>
#include <vjepa_gelu_layer.h>
#include <vjepa_layernorm_layer.h>

namespace causallm {

/**
 * @brief Set projector parameters from config.
 */
void VjepaProjector::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  (void)generation_cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");

  // Vision encoder dimensions
  VISION_DIM = cfg.value("vision_hidden_size", 768);
  DOWNSAMPLE_FACTOR = cfg.value("downsample_factor", 2);

  // Merger MLP dimensions
  INPUT_DIM = VISION_DIM * (DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR);
  MERGER_HIDDEN_1 = cfg.value("merger_hidden_1", 3072);
  MERGER_HIDDEN_2 = cfg.value("merger_hidden_2", 1536);
  NUM_MERGER_FC = nntr_cfg.value("num_merger_fc", cfg.value("num_merger_fc", 2u));
  TEXT_DIM = cfg.value("hidden_size", 1024);

  // Token counts
  unsigned int num_frames = cfg.value("num_frames", 16);
  unsigned int tubelet_size = cfg.value("tubelet_size", 2);
  unsigned int img_size = cfg.value("img_size", 384);
  unsigned int patch_size = cfg.value("patch_size", 16);

  TEMPORAL_DIM = num_frames / tubelet_size;
  SPATIAL_H = img_size / patch_size;
  SPATIAL_W = img_size / patch_size;
  NUM_TOKENS = TEMPORAL_DIM * SPATIAL_H * SPATIAL_W;
  OUTPUT_TOKENS = TEMPORAL_DIM * (SPATIAL_H / DOWNSAMPLE_FACTOR) *
                  (SPATIAL_W / DOWNSAMPLE_FACTOR);

  // Override Transformer base class params
  DIM = TEXT_DIM;
  INTERMEDIATE_SIZE = MERGER_HIDDEN_1;
  NUM_LAYERS = 0;
  NUM_HEADS = 1;
  HEAD_DIM = TEXT_DIM;
  NUM_KEY_VALUE_HEADS = 1;
  GQA_SIZE = 1;
  NORM_EPS = 1e-5;
  TIE_WORD_EMBEDDINGS = false;
  IS_CAUSAL = false;

  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", OUTPUT_TOKENS);
  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", OUTPUT_TOKENS);
  NUM_TO_GENERATE = 0;
  MEMORY_SWAP = false;
  FSU_LOOKAHEAD = 1;
}

/**
 * @brief Construct the symbolic projector graph.
 *
 * Supports two merger architectures controlled by NUM_MERGER_FC:
 *
 * NUM_MERGER_FC=2 (PyTorch reference, 2-FC merger):
 *   input[B, 1, OUTPUT_TOKENS, INPUT_DIM]
 *     → LayerNorm(INPUT_DIM)
 *     → FC(INPUT_DIM → MERGER_HIDDEN_1, bias) → GELU
 *     → FC(MERGER_HIDDEN_1 → TEXT_DIM, bias)
 *   → output[B, 1, OUTPUT_TOKENS, TEXT_DIM]
 *
 * NUM_MERGER_FC=3 (legacy 3-FC merger):
 *   input[B, 1, OUTPUT_TOKENS, INPUT_DIM]
 *     → LayerNorm(INPUT_DIM)
 *     → FC(INPUT_DIM → MERGER_HIDDEN_1, bias) → GELU
 *     → FC(MERGER_HIDDEN_1 → MERGER_HIDDEN_2, bias) → GELU
 *     → LayerNorm(MERGER_HIDDEN_2)
 *     → FC(MERGER_HIDDEN_2 → TEXT_DIM, bias)
 *     → LayerNorm(TEXT_DIM)
 *   → output[B, 1, OUTPUT_TOKENS, TEXT_DIM]
 */
std::pair<Tensor, Tensor> VjepaProjector::constructModel() {
  // Input: after pixel_unshuffle [B, 1, OUTPUT_TOKENS, INPUT_DIM]
  Tensor input({BATCH_SIZE, 1, OUTPUT_TOKENS, INPUT_DIM}, "input0");
  Tensor h = input;

  // LayerNorm(INPUT_DIM)
  LayerHandle ln1(createLayer(
    "vjepa_layernorm",
    {withKey("name", "merger_ln1"),
     withKey("epsilon", "1e-5")}));
  h = ln1(h);

  // FC1: INPUT_DIM → MERGER_HIDDEN_1
  LayerHandle fc1(createLayer(
    "fully_connected",
    {withKey("name", "merger_fc1"),
     withKey("unit", std::to_string(MERGER_HIDDEN_1)),
     withKey("disable_bias", "false")}));
  h = fc1(h);

  // GELU
  LayerHandle gelu1(
    createLayer("vjepa_gelu", {withKey("name", "merger_gelu1")}));
  h = gelu1(h);

  if (NUM_MERGER_FC == 2) {
    // 2-FC merger: FC2 goes directly to TEXT_DIM
    LayerHandle fc2(createLayer(
      "fully_connected",
      {withKey("name", "merger_fc2"),
       withKey("unit", std::to_string(TEXT_DIM)),
       withKey("disable_bias", "false")}));
    h = fc2(h);
  } else {
    // 3-FC merger (legacy): FC2 → GELU → LN → FC3 → LN
    LayerHandle fc2(createLayer(
      "fully_connected",
      {withKey("name", "merger_fc2"),
       withKey("unit", std::to_string(MERGER_HIDDEN_2)),
       withKey("disable_bias", "false")}));
    h = fc2(h);

    // GELU
    LayerHandle gelu2(
      createLayer("vjepa_gelu", {withKey("name", "merger_gelu2")}));
    h = gelu2(h);

    // LayerNorm(MERGER_HIDDEN_2)
    LayerHandle ln2(createLayer(
      "vjepa_layernorm",
      {withKey("name", "merger_ln2"),
       withKey("epsilon", "1e-5")}));
    h = ln2(h);

    // FC3: MERGER_HIDDEN_2 → TEXT_DIM
    LayerHandle fc3(createLayer(
      "fully_connected",
      {withKey("name", "merger_fc3"),
       withKey("unit", std::to_string(TEXT_DIM)),
       withKey("disable_bias", "false")}));
    h = fc3(h);

    // LayerNorm(TEXT_DIM)
    LayerHandle ln3(createLayer(
      "vjepa_layernorm",
      {withKey("name", "merger_ln3"),
       withKey("epsilon", "1e-5")}));
    h = ln3(h);
  }

  return {input, h};
}

/**
 * @brief Register custom layers (vjepa_gelu, vjepa_layernorm).
 */
void VjepaProjector::registerCustomLayers() {
  Transformer::registerCustomLayers();

  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::VjepaGeluLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register vjepa_gelu factory: " << e.what()
              << std::endl;
  }
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::VjepaLayerNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register vjepa_layernorm factory: " << e.what()
              << std::endl;
  }
}

/**
 * @brief Apply pixel_unshuffle on vision embeddings.
 *
 * Converts [B, T*H*W, C] → [B, T*(H/f)*(W/f), C*f*f]
 * where f = DOWNSAMPLE_FACTOR.
 *
 * The VJEPA2ViT output is organized as [T, H, W, C] tokens flattened to
 * [T*H*W, C]. We need to rearrange spatial dimensions:
 *   For each temporal position t:
 *     Input:  [H, W, C] tokens
 *     Output: [H/f, W/f, C*f*f] tokens
 *     where adjacent 2x2 spatial blocks are merged into single tokens.
 */
void VjepaProjector::pixelUnshuffle(const float *input, float *output,
                                    unsigned int /*num_tokens*/,
                                    unsigned int temporal_dim,
                                    unsigned int spatial_h,
                                    unsigned int spatial_w,
                                    unsigned int vision_dim,
                                    unsigned int factor) {
  const unsigned int out_h = spatial_h / factor;
  const unsigned int out_w = spatial_w / factor;
  const unsigned int out_dim = vision_dim * factor * factor;

  for (unsigned int t = 0; t < temporal_dim; ++t) {
    for (unsigned int oh = 0; oh < out_h; ++oh) {
      for (unsigned int ow = 0; ow < out_w; ++ow) {
        // Output token index
        unsigned int out_token_idx =
          t * out_h * out_w + oh * out_w + ow;

        // For each output channel, gather from the 2x2 input block
        for (unsigned int c = 0; c < vision_dim; ++c) {
          for (unsigned int fh = 0; fh < factor; ++fh) {
            for (unsigned int fw = 0; fw < factor; ++fw) {
              unsigned int ih = oh * factor + fh;
              unsigned int iw = ow * factor + fw;

              // Input token index
              unsigned int in_token_idx =
                t * spatial_h * spatial_w + ih * spatial_w + iw;

              // Output channel index (matches Python: fh, fw, c ordering)
              unsigned int out_c =
                (fh * factor + fw) * vision_dim + c;

              output[out_token_idx * out_dim + out_c] =
                input[in_token_idx * vision_dim + c];
            }
          }
        }
      }
    }
  }
}

/**
 * @brief Run the projector on VJEPA2 encoder output.
 */
multimodal_pointer VjepaProjector::run(const float *vision_embeds,
                                      unsigned int num_tokens,
                                      bool log_output) {
  if (!is_initialized) {
    throw std::runtime_error(
      "VjepaProjector model is not initialized. Please call "
      "initialize() before run().");
  }

  if (num_tokens != NUM_TOKENS) {
    throw std::runtime_error(
      "VjepaProjector::run: num_tokens (" + std::to_string(num_tokens) +
      ") does not match NUM_TOKENS (" + std::to_string(NUM_TOKENS) + ").");
  }

  // Step 1: Apply pixel_unshuffle
  unshuffled_.resize(
    static_cast<size_t>(OUTPUT_TOKENS) * INPUT_DIM);
  pixelUnshuffle(vision_embeds, unshuffled_.data(),
                 num_tokens, TEMPORAL_DIM, SPATIAL_H, SPATIAL_W,
                 VISION_DIM, DOWNSAMPLE_FACTOR);

  // Debug: print first 5 values after pixel_unshuffle
  if (log_output) {
    std::cout << "  After pixel_unshuffle, first 5: ";
    for (int i = 0; i < 5; ++i)
      std::cout << unshuffled_[i] << " ";
    std::cout << "\n";
  }

  // Step 2: Run merger MLP
  std::vector<float *> input_ptrs;
  input_ptrs.push_back(unshuffled_.data());
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input_ptrs, label, OUTPUT_TOKENS, 0, OUTPUT_TOKENS, false);

  // Step 3: Store output
  last_output_.assign(output[0], output[0] + OUTPUT_TOKENS * TEXT_DIM);

  if (log_output) {
    std::cout << std::setprecision(9)
              << "Projector: first 10 values (first token): ";
    const int print_count = TEXT_DIM > 10 ? 10 : TEXT_DIM;
    for (int i = 0; i < print_count; ++i) {
      std::cout << "[" << i << "]=" << output[0][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  Input tokens: " << NUM_TOKENS
              << " → Output tokens: " << OUTPUT_TOKENS << "\n";
    std::cout << "  Input dim: " << INPUT_DIM
              << " → Output dim: " << TEXT_DIM << "\n";
  }

  return {last_output_.data(), last_output_.size() * sizeof(float)};
}

} // namespace causallm
