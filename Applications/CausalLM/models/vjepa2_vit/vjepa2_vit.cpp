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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
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

// Note: STB_IMAGE_IMPLEMENTATION is defined in timm_vit_transformer.cpp;
// here we only need the declarations from video_util.h.
#include "../../video_util.h"

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
  NUM_FRAMES = cfg.value("num_frames", 16);
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
  LayerHandle proj(createLayer(
    "fully_connected", {withKey("name", "patch_embed/proj"),
                        withKey("unit", std::to_string(DIM)),
                        withKey("disable_bias", "false")}));
  return proj(input);
}

/**
 * @brief Create a pre-normalized self-attention block with 3D RoPE.
 */
Tensor VJEPA2ViT::createAttention(const int layer_id, Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id) + "_";

  LayerHandle norm(
    createLayer("vjepa_layernorm",
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
     withKey("use_rope", "false"),
     // V-JEPA-2 block-0 attention logits reach ~457k, past FP16's 65504 ceiling.
     // The fmlal-widening QK kernel computes the FP16 Q*K products in FP32
     // accumulators, so the wide logits ghtnever overflow — no separate
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

  LayerHandle norm(
    createLayer("vjepa_layernorm",
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

  LayerHandle output_norm(
    createLayer("vjepa_layernorm",
                {withKey("name", "output_norm"),
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

multimodal_pointer
VJEPA2ViT::run_image(const std::vector<std::vector<float>> &images,
                     unsigned int original_height,
                     unsigned int original_width,
                     bool log_output) {
  (void)original_height;
  (void)original_width;

  if (!is_initialized) {
    throw std::runtime_error("VJEPA2ViT model is not initialized. Please call "
                             "initialize() before run_image().");
  }

  if (images.size() != NUM_FRAMES) {
    throw std::runtime_error(
      "VJEPA2ViT::run_image: frame count mismatch; got " +
      std::to_string(images.size()) + ", expected " +
      std::to_string(NUM_FRAMES));
  }

  const size_t frame_plane = static_cast<size_t>(IMG_SIZE) * IMG_SIZE;
  const size_t frame_size = static_cast<size_t>(IN_CHANS) * frame_plane;
  std::vector<float> video(static_cast<size_t>(IN_CHANS) * NUM_FRAMES *
                           IMG_SIZE * IMG_SIZE);

  for (unsigned int t = 0; t < NUM_FRAMES; ++t) {
    if (images[t].size() != frame_size) {
      throw std::runtime_error(
        "VJEPA2ViT::run_image: frame " + std::to_string(t) +
        " size mismatch; got " + std::to_string(images[t].size()) +
        ", expected " + std::to_string(frame_size));
    }
    for (unsigned int c = 0; c < IN_CHANS; ++c) {
      const size_t src_offset = static_cast<size_t>(c) * frame_plane;
      const size_t dst_offset =
        (static_cast<size_t>(c) * NUM_FRAMES + t) * frame_plane;
      std::copy_n(images[t].data() + src_offset, frame_plane,
                  video.data() + dst_offset);
    }
  }

  std::vector<float> tokens = patchify(video);
  std::vector<float *> input;
  input.push_back(tokens.data());
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);

  const size_t n_out = static_cast<size_t>(BATCH_SIZE) * NUM_PATCHES * DIM;
  last_output_.assign(output[0], output[0] + n_out);

  if (log_output) {
    std::cout << "[VJEPA2ViT] features [" << NUM_PATCHES << "x" << DIM
              << "], first 10 values: ";
    const int print_count = DIM > 10 ? 10 : static_cast<int>(DIM);
    for (int i = 0; i < print_count; ++i) {
      std::cout << "[" << i << "]=" << std::setprecision(9)
                << last_output_[i] << " ";
    }
    std::cout << std::endl;
  }

  return {last_output_.data(), last_output_.size() * sizeof(float)};
}

multimodal_pointer
VJEPA2ViT::run_with_bin(const std::string &video_bin_path,
                        bool log_output) {
  if (!is_initialized) {
    throw std::runtime_error("VJEPA2ViT model is not initialized. Please call "
                             "initialize() before run_with_bin().");
  }

  if (video_bin_path.empty()) {
    throw std::invalid_argument(
      "video_bin_path must not be empty in run_with_bin().");
  }

  // Load the raw float32 [C, T, H, W] binary file
  std::vector<float> video = loadVideoFromBin(video_bin_path, IN_CHANS,
                                               NUM_FRAMES, IMG_SIZE, IMG_SIZE);

  if (log_output) {
    std::cout << "[VJEPA2ViT] Loaded video from binary: " << video_bin_path
              << " (" << video.size() << " floats)" << std::endl;
  }

  // Patchify and run inference
  std::vector<float> tokens = patchify(video);

  std::vector<float *> input;
  input.push_back(tokens.data());
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);

  const size_t n_out = static_cast<size_t>(BATCH_SIZE) * NUM_PATCHES * DIM;
  last_output_.assign(output[0], output[0] + n_out);

  if (log_output) {
    std::cout << "[VJEPA2ViT] features [" << NUM_PATCHES << "x" << DIM
              << "], first 10 values: ";
    const int print_count = DIM > 10 ? 10 : static_cast<int>(DIM);
    for (int i = 0; i < print_count; ++i) {
      std::cout << "[" << i << "]=" << std::setprecision(9)
                << last_output_[i] << " ";
    }
    std::cout << std::endl;
  }

  return {last_output_.data(), last_output_.size() * sizeof(float)};
}

/**
 * @brief Run the encoder on video loaded from a directory of image frames
 *        or a raw binary tensor file.
 *
 * If @p video_dir is non-empty, image frames are loaded from the directory,
 * resized to IMG_SIZE×IMG_SIZE, and assembled into a [C,T,H,W] float buffer.
 * Otherwise, @p video_bin_path is used to read a pre-built raw binary tensor.
 *
 * After loading, the video is patchified and fed through the ViT encoder.
 * The first 10 values of the last-token output are printed, and the full
 * last-token hidden state is dumped to "<video_path>.nntr_out.bin".
 */
namespace {
/**
 * @brief Print statistics (min, max, mean, NaN/Inf count) of a float buffer.
 */
void print_tensor_stats(const std::string &label, const float *data,
                        size_t count) {
  size_t nan_count = 0, inf_count = 0;
  float vmin = std::numeric_limits<float>::max();
  float vmax = -std::numeric_limits<float>::max();
  double vsum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    float v = data[i];
    if (std::isnan(v)) {
      ++nan_count;
      continue;
    }
    if (std::isinf(v)) {
      ++inf_count;
      continue;
    }
    if (v < vmin)
      vmin = v;
    if (v > vmax)
      vmax = v;
    vsum += v;
  }
  size_t valid = count - nan_count - inf_count;
  double mean = valid > 0 ? vsum / valid : 0.0;
  std::cout << "[DEBUG] " << label << " (" << count << " elements): "
            << "min=" << std::setprecision(6) << vmin
            << ", max=" << vmax
            << ", mean=" << mean
            << ", NaN=" << nan_count
            << ", Inf=" << inf_count << std::endl;
}
} // anonymous namespace

void VJEPA2ViT::run_with_video(const std::string &video_dir,
                                const std::string &video_bin_path,
                                bool normalize) {
  if (!is_initialized) {
    throw std::runtime_error("VJEPA2ViT model is not initialized. Please call "
                             "initialize() before run_with_video().");
  }

  // Load video tensor: either from a directory of frames or from a binary file
  std::vector<float> video;
  std::string video_source; // for logging / dump path

  if (!video_dir.empty()) {
    std::cout << "Loading video from image frames in: " << video_dir
              << std::endl;
    video = loadAndPreprocessVideo(video_dir,
                                   static_cast<int>(IMG_SIZE),
                                   static_cast<int>(IMG_SIZE),
                                   NUM_FRAMES, NORM_IMAGENET);
    video_source = video_dir;
  } else if (!video_bin_path.empty()) {
    std::cout << "Loading video from binary file: " << video_bin_path
              << std::endl;
    video = loadVideoFromBin(video_bin_path, IN_CHANS, NUM_FRAMES,
                             IMG_SIZE, IMG_SIZE);
    video_source = video_bin_path;
  } else {
    throw std::invalid_argument(
      "Either video_dir or video_bin_path must be provided to "
      "run_with_video().");
  }

  const size_t expected =
    static_cast<size_t>(IN_CHANS) * NUM_FRAMES * IMG_SIZE * IMG_SIZE;
  if (video.size() != expected) {
    throw std::runtime_error(
      "Loaded video size mismatch: got " + std::to_string(video.size()) +
      ", expected " + std::to_string(expected) +
      " ([C,T,H,W] = [" + std::to_string(IN_CHANS) + "," +
      std::to_string(NUM_FRAMES) + "," + std::to_string(IMG_SIZE) + "," +
      std::to_string(IMG_SIZE) + "]).");
  }

  std::cout << "Video loaded: [C=" << IN_CHANS << ", T=" << NUM_FRAMES
            << ", H=" << IMG_SIZE << ", W=" << IMG_SIZE << "] ("
            << video.size() << " floats)" << std::endl;

  // Debug: raw video tensor stats
  print_tensor_stats("Raw video tensor", video.data(), video.size());
  // Debug: first 10 values of video
  std::cout << "[DEBUG] Raw video first 10: ";
  for (int i = 0; i < 10; ++i)
    std::cout << std::setprecision(6) << video[i] << " ";
  std::cout << std::endl;

  // Patchify and run inference
  std::vector<float> tokens = patchify(video);

  // Debug: patchified tokens = input to patch_embed/proj FC
  print_tensor_stats("patch_embed/proj INPUT (patchified tokens)", tokens.data(), tokens.size());

  // Print first 10 input tokens (patches)
  const int n_print = NUM_PATCHES > 10 ? 10 : NUM_PATCHES;
  std::cout << "[Input] First " << n_print << " tokens (each " << PATCH_VEC
            << " dims, showing first 5 values):" << std::endl;
  for (int i = 0; i < n_print; ++i) {
    std::cout << "  token[" << i << "]: ";
    const int n_vals = PATCH_VEC > 5 ? 5 : PATCH_VEC;
    for (int j = 0; j < n_vals; ++j) {
      std::cout << std::setprecision(6) << tokens[i * PATCH_VEC + j] << " ";
    }
    std::cout << "..." << std::endl;
  }

  std::vector<float *> input;
  input.push_back(tokens.data());
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);

  // Print patch_embed/proj weight stats
  {
    std::shared_ptr<ml::train::Layer> proj_layer;
    int ret = model->getLayer("patch_embed/proj", &proj_layer);
    if (ret == 0 && proj_layer) {
      std::vector<float *> weights;
      std::vector<ml::train::TensorDim> dims;
      proj_layer->getWeights(weights, dims);
      for (size_t w = 0; w < weights.size(); ++w) {
        size_t wsize = dims[w].getDataLen();
        print_tensor_stats("patch_embed/proj weight[" + std::to_string(w) +
                             "] (" + std::to_string(dims[w].batch()) + "x" +
                             std::to_string(dims[w].channel()) + "x" +
                             std::to_string(dims[w].height()) + "x" +
                             std::to_string(dims[w].width()) + ")",
                           weights[w], wsize);
        // Print first 5 and last 5 values
        {
          const float *wd = weights[w];
          size_t n5 = wsize > 5 ? 5 : wsize;
          std::cout << "[DEBUG]   first " << n5 << ": ";
          for (size_t i = 0; i < n5; ++i)
            std::cout << std::setprecision(6) << wd[i] << " ";
          std::cout << std::endl;
          if (wsize > 5) {
            std::cout << "[DEBUG]   last 5: ";
            for (size_t i = wsize - 5; i < wsize; ++i)
              std::cout << std::setprecision(6) << wd[i] << " ";
            std::cout << std::endl;
          }
        }
      }
    } else {
      std::cout << "[DEBUG] patch_embed/proj layer not found." << std::endl;
    }
  }

  // Debug: output tensor stats
  print_tensor_stats("Output tensor (last token)", output[0], DIM);

  std::cout << std::setprecision(9) << "First 10 values (last token): ";
  const int print_count = DIM > 10 ? 10 : DIM;
  for (int i = 0; i < print_count; ++i) {
    std::cout << "[" << i << "]=" << output[0][i] << " ";
  }
  std::cout << std::endl;

  // Dump the last-token hidden state (DIM floats) for offline verification.
  const std::string dump_path = video_source + ".nntr_out.bin";
  std::ofstream of(dump_path, std::ios::binary);
  if (of.is_open()) {
    of.write(reinterpret_cast<const char *>(output[0]), DIM * sizeof(float));
    std::cout << "Wrote last-token output [" << DIM << "] to " << dump_path
              << std::endl;
  }
}

} // namespace causallm
