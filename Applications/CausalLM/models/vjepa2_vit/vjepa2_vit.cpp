// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vjepa2_vit.cpp
 * @date   21 May 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  V-JEPA 2.1 ViT encoder (ViT-B/16, video) for nntrainer.
 */

#include "vjepa2_vit.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include <app_context.h>
#include <engine.h>
#include <factory.h>
#include <llm_util.hpp>
#include <vjepa_gelu_layer.h>
#include <vjepa_layernorm_layer.h>
#include <vjepa_rope_layer.h>

namespace causallm {

/**
 * @brief Set ViT-specific parameters from model and runtime configs.
 */
void VJEPA2ViT::setupParameters(json &cfg, json &generation_cfg,
                                json &nntr_cfg) {
  (void)generation_cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");

  DIM = cfg.value("hidden_size", 768);
  INTERMEDIATE_SIZE = cfg.value("intermediate_size", 3072);
  NUM_LAYERS = cfg.value("num_hidden_layers", 12);
  NUM_HEADS = cfg.value("num_attention_heads", 12);
  HEAD_DIM = cfg.value("head_dim", DIM / NUM_HEADS);
  NUM_KEY_VALUE_HEADS = cfg.value("num_key_value_heads", NUM_HEADS);
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;
  NORM_EPS = cfg.value("norm_eps", 1e-6);
  TIE_WORD_EMBEDDINGS = false;
  IS_CAUSAL = false;

  // Vision / video geometry
  IMG_SIZE = cfg.value("img_size", 384);
  PATCH_SIZE = cfg.value("patch_size", 16);
  TUBELET = cfg.value("tubelet_size", 2);
  NUM_FRAMES = cfg.value("num_frames", 64);
  IN_CHANS = cfg.value("in_chans", 3);
  PRETRAINED_GRID = cfg.value("pretrained_grid_size", 256 / PATCH_SIZE);
  INTERPOLATE_ROPE = cfg.value("interpolate_rope", true);

  GRID_T = NUM_FRAMES / TUBELET;
  GRID_H = IMG_SIZE / PATCH_SIZE;
  GRID_W = IMG_SIZE / PATCH_SIZE;
  NUM_PATCHES = GRID_T * GRID_H * GRID_W;
  PATCH_VEC = IN_CHANS * TUBELET * PATCH_SIZE * PATCH_SIZE;

  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", NUM_PATCHES);
  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", NUM_PATCHES);
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
}

/**
 * @brief Extract non-overlapping tubelets matching Conv3d weight layout.
 *
 * Source video buffer is laid out [C, T, H, W] (C-order). Each output token
 * corresponds to one (t', h', w') tubelet and is a PATCH_VEC vector ordered as
 * (in_chan, kt, kh, kw) so that a single fully_connected with the reshaped
 * Conv3d weight reproduces PatchEmbed3D exactly. Tokens are ordered
 * n = t' * (GRID_H * GRID_W) + h' * GRID_W + w', matching the flatten(2) of the
 * reference PatchEmbed3D.
 */
std::vector<float> VJEPA2ViT::patchify(const std::vector<float> &video) const {
  const unsigned int T = NUM_FRAMES, H = IMG_SIZE, W = IMG_SIZE;
  std::vector<float> tokens(static_cast<size_t>(NUM_PATCHES) * PATCH_VEC);

  for (unsigned int tt = 0; tt < GRID_T; ++tt) {
    for (unsigned int hh = 0; hh < GRID_H; ++hh) {
      for (unsigned int ww = 0; ww < GRID_W; ++ww) {
        const size_t token =
          (static_cast<size_t>(tt) * GRID_H + hh) * GRID_W + ww;
        float *dst = tokens.data() + token * PATCH_VEC;
        size_t k = 0;
        for (unsigned int c = 0; c < IN_CHANS; ++c) {
          for (unsigned int kt = 0; kt < TUBELET; ++kt) {
            const unsigned int frame = tt * TUBELET + kt;
            for (unsigned int kh = 0; kh < PATCH_SIZE; ++kh) {
              const unsigned int row = hh * PATCH_SIZE + kh;
              for (unsigned int kw = 0; kw < PATCH_SIZE; ++kw) {
                const unsigned int col = ww * PATCH_SIZE + kw;
                const size_t vidx =
                  ((static_cast<size_t>(c) * T + frame) * H + row) * W + col;
                dst[k++] = video[vidx];
              }
            }
          }
        }
      }
    }
  }
  return tokens;
}

/**
 * @brief Create the tubelet patch-embedding projection (Conv3d-equivalent FC).
 */
Tensor VJEPA2ViT::createPatchEmbed(Tensor input) {
  // patch_embed is a plain FC, so it follows the model's global weight dtype
  // (FP32 for the FP32 model, Q4_0 for the quantized model — nntr_quantize
  // includes "patch_embed/proj" in its FC dtype map).
  LayerHandle proj(
    createLayer("fully_connected", {withKey("name", "patch_embed/proj"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  return proj(input);
}

/**
 * @brief Create a pre-normalized self-attention block with 3D RoPE.
 */
Tensor VJEPA2ViT::createAttention(const int layer_id, Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id) + "_";

  LayerHandle norm(createLayer("vjepa_layernorm",
                               {withKey("name", prefix + "attention_norm"),
                                withKey("epsilon", std::to_string(NORM_EPS))}));
  Tensor normed = norm(input);

  // Names follow the LLM convention (wq/wk/wv/attention_out, ffn_up/ffn_down)
  // so the nntr_quantize layer-dtype map targets them for Q4_0.
  const std::string q = prefix + "wq", k = prefix + "wk", v = prefix + "wv",
                    a = prefix + "attention", o = prefix + "attention_out";

  LayerHandle q_proj(
    createLayer("fully_connected",
                {withKey("name", q), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false")}));
  LayerHandle k_proj(
    createLayer("fully_connected",
                {withKey("name", k), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false")}));
  LayerHandle v_proj(
    createLayer("fully_connected",
                {withKey("name", v), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false")}));

  Tensor query = q_proj(normed);
  Tensor key = k_proj(normed);
  Tensor value = v_proj(normed);

  // 3D axial RoPE on Q and K (mha_core runs with rope disabled).
  auto rope_props = [&](const std::string &name) {
    return std::vector<std::string>{
      withKey("name", name),
      withKey("num_heads", std::to_string(NUM_HEADS)),
      withKey("grid_t", std::to_string(GRID_T)),
      withKey("grid_h", std::to_string(GRID_H)),
      withKey("grid_w", std::to_string(GRID_W)),
      withKey("rope_theta", "10000.0"),
      withKey("pretrained_grid_size", std::to_string(PRETRAINED_GRID)),
      withKey("interpolate_rope", INTERPOLATE_ROPE ? "true" : "false")};
  };
  LayerHandle q_rope(createLayer("vjepa_rope", rope_props(prefix + "q_rope")));
  LayerHandle k_rope(createLayer("vjepa_rope", rope_props(prefix + "k_rope")));
  query = q_rope(query);
  key = k_rope(key);

  LayerHandle attention(createLayer(
    "mha_core",
    {withKey("name", a), withKey("num_heads", std::to_string(NUM_HEADS)),
     withKey("num_heads_KV", std::to_string(NUM_HEADS)),
     withKey("max_timestep", std::to_string(NUM_PATCHES + 1)),
     withKey("is_causal", "false"), withKey("rope_theta", "0"),
     // V-JEPA applies 3D axial RoPE in the custom vjepa_rope layer *before*
     // mha_core, so mha_core must NOT apply its own rotary embedding. Upstream
     // gates the internal rope on use_rope (default true) rather than the old
     // rope_theta>0 check, so disable it explicitly here — otherwise rope runs
     // with theta=0 and produces inf/NaN frequencies.
     withKey("use_rope", "false"),
     // V-JEPA-2 block-0 attention logits reach ~457k, past FP16's 65504
     // ceiling. The fmlal-widening QK kernel computes the FP16 Q*K products in
     // FP32 accumulators, so the wide logits never overflow — no separate
     // use_fp32_scores path is needed.
     withKey("use_gemm_attention", "true")}));
  Tensor context = attention({query, key, value});

  LayerHandle out_proj(
    createLayer("fully_connected",
                {withKey("name", o), withKey("unit", std::to_string(DIM)),
                 withKey("disable_bias", "false")}));
  return out_proj(context);
}

/**
 * @brief Create a pre-normalized GELU feed-forward block.
 */
Tensor VJEPA2ViT::createMlp(const int layer_id, Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id) + "_";

  LayerHandle norm(createLayer("vjepa_layernorm",
                               {withKey("name", prefix + "ffn_norm"),
                                withKey("epsilon", std::to_string(NORM_EPS))}));
  Tensor h = norm(input);

  LayerHandle fc_up(createLayer(
    "fully_connected", {withKey("name", prefix + "ffn_up"),
                        withKey("unit", std::to_string(INTERMEDIATE_SIZE)),
                        withKey("disable_bias", "false")}));
  h = fc_up(h);

  // Core "activation" gelu has no FP16 path (ActiFunc::gelu<half> calls the
  // FP32 gelu_v2 on FP16 data and crashes), so use the custom token-parallel
  // vjepa_gelu which casts each FP16 element up to FP32 for the exact-erf eval.
  LayerHandle gelu(
    createLayer("vjepa_gelu", {withKey("name", prefix + "ffn_gelu")}));
  h = gelu(h);

  LayerHandle fc_down(
    createLayer("fully_connected", {withKey("name", prefix + "ffn_down"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  return fc_down(h);
}

/**
 * @brief Create one ViT transformer block with residual connections.
 */
Tensor VJEPA2ViT::createTransformerDecoderBlock(const int layer_id,
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
 * @brief Construct the symbolic ViT inference graph.
 */
std::pair<Tensor, Tensor> VJEPA2ViT::constructModel() {
  // Host-side patchified tokens: [B, 1, NUM_PATCHES, PATCH_VEC]
  Tensor input({BATCH_SIZE, 1, NUM_PATCHES, PATCH_VEC}, "input0");
  Tensor h = createPatchEmbed(input);

  for (int i = 0; i < NUM_LAYERS; i++) {
    h = createTransformerDecoderBlock(i, h);
  }

  LayerHandle output_norm(createLayer(
    "vjepa_layernorm", {withKey("name", "output_norm"),
                        withKey("epsilon", std::to_string(NORM_EPS))}));
  h = output_norm(h);

  return {input, h};
}

/**
 * @brief Register layers used by this model.
 */
void VJEPA2ViT::registerCustomLayers() {
  Transformer::registerCustomLayers();

  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::VjepaRopeLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register vjepa_rope factory: " << e.what()
              << std::endl;
  }
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
 * @brief Run the encoder on a preprocessed video tensor file.
 *
 * @param prompt path to a raw float32 file holding a [C, T, H, W] video tensor
 *               (already resized and normalized to the model's expectations).
 */
void VJEPA2ViT::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                    const WSTR tail_prompt, bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  (void)log_output;

  if (!is_initialized) {
    throw std::runtime_error("VJEPA2ViT model is not initialized. Please call "
                             "initialize() before run().");
  }

  const size_t expected =
    static_cast<size_t>(IN_CHANS) * NUM_FRAMES * IMG_SIZE * IMG_SIZE;

  std::ifstream f(std::string(prompt), std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open video tensor file: " +
                             std::string(prompt));
  }
  std::vector<float> video(expected);
  f.read(reinterpret_cast<char *>(video.data()), expected * sizeof(float));
  if (static_cast<size_t>(f.gcount()) != expected * sizeof(float)) {
    throw std::runtime_error(
      "Video tensor file size mismatch; expected " + std::to_string(expected) +
      " float32 values ([C,T,H,W] = [" + std::to_string(IN_CHANS) + "," +
      std::to_string(NUM_FRAMES) + "," + std::to_string(IMG_SIZE) + "," +
      std::to_string(IMG_SIZE) + "]).");
  }

  std::vector<float> tokens = patchify(video);

  std::vector<float *> input;
  input.push_back(tokens.data());
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);

  std::cout << std::setprecision(9) << "First 10 values (last token): ";
  const int print_count = DIM > 10 ? 10 : DIM;
  for (int i = 0; i < print_count; ++i) {
    std::cout << "[" << i << "]=" << output[0][i] << " ";
  }
  std::cout << std::endl;

  // Dump the last-token hidden state (DIM floats) for offline verification.
  const std::string dump_path = std::string(prompt) + ".nntr_out.bin";
  std::ofstream of(dump_path, std::ios::binary);
  if (of.is_open()) {
    of.write(reinterpret_cast<const char *>(output[0]), DIM * sizeof(float));
    std::cout << "Wrote last-token output [" << DIM << "] to " << dump_path
              << std::endl;
  }
}

} // namespace causallm
