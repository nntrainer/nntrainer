// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   vjepa_lfm2_vl.cpp
 * @date   9 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  V-JEPA 2.1 + Projector + LFM2 multimodal video-language model.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "vjepa_lfm2_vl.h"

#include <llm_util.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace causallm {

namespace {

unsigned int valueForAnyKey(const json &cfg,
                            std::initializer_list<const char *> keys,
                            unsigned int default_value) {
  for (const auto *key : keys) {
    if (cfg.contains(key) && !cfg[key].is_null())
      return cfg[key].get<unsigned int>();
  }
  return default_value;
}

unsigned int imageSizeFromVisionConfig(const json &vision_cfg) {
  return valueForAnyKey(vision_cfg, {"image_size", "img_size"}, 256u);
}

unsigned int textHiddenSizeFromConfig(const json &cfg, const json &text_cfg) {
  return valueForAnyKey(cfg, {"hidden_size"},
                        valueForAnyKey(text_cfg, {"hidden_size"}, 1024u));
}

} // namespace

/* -------------------------------------------------------------------------
 * Config splitting
 * ---------------------------------------------------------------------- */

std::pair<json, json>
VjepaLfm2ForConditionalGeneration::splitConfig(const json &top) {
  // text_config: prefer explicit "text_config" key; fallback to top-level
  json text_cfg = top.contains("text_config") ? top["text_config"] : top;

  // vision_config: must be present
  if (!top.contains("vision_config"))
    throw std::invalid_argument(
      "VjepaLfm2ForConditionalGeneration: config.json missing "
      "'vision_config'");
  json vision_cfg = top["vision_config"];

  return {text_cfg, vision_cfg};
}

json VjepaLfm2ForConditionalGeneration::buildProjectorConfig(
  const json &cfg) const {
  auto [text_cfg, vision_cfg] = splitConfig(cfg);

  unsigned int vit_embed = vision_cfg.value("hidden_size", 768u);
  unsigned int lm_hidden = textHiddenSizeFromConfig(cfg, text_cfg);
  unsigned int img_size = imageSizeFromVisionConfig(vision_cfg);
  unsigned int patch_size = vision_cfg.value("patch_size", 16u);
  unsigned int num_frames = vision_cfg.contains("num_frames")
                              ? vision_cfg["num_frames"].get<unsigned int>()
                              : 16u;
  unsigned int tubelet_size = vision_cfg.value("tubelet_size", 2u);
  unsigned int grid_t = num_frames / tubelet_size;
  unsigned int grid_h = img_size / patch_size;
  unsigned int grid_w = img_size / patch_size;
  unsigned int num_patches = grid_t * grid_h * grid_w;
  unsigned int output_tokens =
    num_patches / (downsample_factor_ * downsample_factor_);
  unsigned int input_dim = vit_embed * downsample_factor_ * downsample_factor_;
  unsigned int merger_hidden_1 =
    valueForAnyKey(cfg, {"merger_hidden_1"}, input_dim);
  unsigned int merger_hidden_2 =
    valueForAnyKey(cfg, {"merger_hidden_2"}, input_dim / 2u);

  json proj_cfg;
  proj_cfg["vision_hidden_size"] = vit_embed;
  proj_cfg["hidden_size"] = lm_hidden;
  proj_cfg["img_size"] = img_size;
  proj_cfg["patch_size"] = patch_size;
  proj_cfg["num_frames"] = num_frames;
  proj_cfg["tubelet_size"] = tubelet_size;
  proj_cfg["downsample_factor"] = downsample_factor_;
  proj_cfg["num_patches"] = num_patches;
  proj_cfg["output_tokens"] = output_tokens;
  proj_cfg["input_dim"] = input_dim;
  proj_cfg["output_dim"] = lm_hidden;
  proj_cfg["merger_hidden_1"] = merger_hidden_1;
  proj_cfg["merger_hidden_2"] = merger_hidden_2;
  proj_cfg["num_merger_fc"] = cfg.value("num_merger_fc", 2u);
  proj_cfg["projector_bias"] = cfg.value("projector_bias", true);
  proj_cfg["projector_hidden_act"] = cfg.value("projector_hidden_act", "gelu");
  proj_cfg["projector_use_layernorm"] =
    cfg.value("projector_use_layernorm", false);

  return proj_cfg;
}

/* -------------------------------------------------------------------------
 * Constructor
 * ---------------------------------------------------------------------- */

VjepaLfm2ForConditionalGeneration::VjepaLfm2ForConditionalGeneration(
  json &cfg, json &generation_cfg, json &nntr_cfg)
  : cfg_(cfg),
    generation_cfg_(generation_cfg),
    nntr_cfg_(nntr_cfg) {

  auto [text_cfg, vision_cfg] = splitConfig(cfg);
  if (!vision_cfg.contains("img_size") && vision_cfg.contains("image_size"))
    vision_cfg["img_size"] = vision_cfg["image_size"];
  if (!vision_cfg.contains("in_chans") && vision_cfg.contains("num_channels"))
    vision_cfg["in_chans"] = vision_cfg["num_channels"];

  // Allow nntr_config overrides for num_frames and num_video_tags.
  // Also propagate to cfg_ so buildProjectorConfig() picks up the override.
  if (nntr_cfg_.contains("vision_config") && nntr_cfg_["vision_config"].is_object()) {
    const auto &vis_nntr = nntr_cfg_["vision_config"];
    if (vis_nntr.contains("num_frames")) {
      vision_cfg["num_frames"] = vis_nntr["num_frames"];
      cfg_["vision_config"]["num_frames"] = vis_nntr["num_frames"];
    }
    if (vis_nntr.contains("num_video_tags"))
      num_video_tags_ = vis_nntr["num_video_tags"].get<unsigned int>();
    else
      num_video_tags_ = cfg.value("num_video_tags", 12u);
  } else {
    num_video_tags_ = cfg.value("num_video_tags", 12u);
  }

  downsample_factor_ = cfg.value("downsample_factor", 2u);
  video_token_id_ = cfg.value("video_token_id", 64400);
  image_token_id_ = cfg.value("image_token_id", 396);
  projector_hidden_size_ = cfg.value("projector_hidden_size", 2048u);

  // Helper: build component-specific nntr_cfg by merging common options
  // with component-specific overrides from vision_config/connector_config/llm_config.
  auto buildComponentNntrCfg = [&](const std::string &config_key) -> json {
    json result = nntr_cfg_; // start with common options
    if (nntr_cfg_.contains(config_key) && nntr_cfg_[config_key].is_object()) {
      result.merge_patch(nntr_cfg_[config_key]); // component overrides
    }
    return result;
  };

  // ── Vision encoder ────────────────────────────────────────────────
  json proj_cfg = buildProjectorConfig(cfg_);
  json vis_nntr = buildComponentNntrCfg("vision_config");
  vis_nntr["model_type"] = "Model";
  vis_nntr["skip_tokenizer"] = true;
  vis_nntr["num_to_generate"] = 0;
  vis_nntr["init_seq_len"] = proj_cfg["num_patches"];
  vis_nntr["max_seq_len"] = proj_cfg["num_patches"];

  const std::string vision_type = vis_nntr.value("vision_model_type",
                                        vis_nntr.value("model_type", "VJEPA2ViT"));
  if (vision_type == "VJEPA2ViT") {
    vjepa_ =
      std::make_unique<VJEPA2ViT>(vision_cfg, generation_cfg_, vis_nntr);
  } else {
    throw std::invalid_argument(
      "VjepaLfm2ForConditionalGeneration: unsupported vision model_type: " +
      vision_type);
  }

  // ── Connector (Projector) ─────────────────────────────────────────
  json conn_nntr = buildComponentNntrCfg("connector_config");
  conn_nntr["model_type"] = "Model";
  conn_nntr["skip_tokenizer"] = true;
  conn_nntr["num_to_generate"] = 0;
  conn_nntr["init_seq_len"] = proj_cfg["output_tokens"];
  conn_nntr["max_seq_len"] = proj_cfg["output_tokens"];

  const std::string connector_type = conn_nntr.value("connector_model_type",
                                        conn_nntr.value("model_type", "VjepaProjector"));
  if (connector_type == "VjepaProjector") {
    projector_ =
      std::make_unique<VjepaProjector>(proj_cfg, generation_cfg_, conn_nntr);
  } else {
    throw std::invalid_argument(
      "VjepaLfm2ForConditionalGeneration: unsupported connector model_type: " +
      connector_type);
  }

  // ── LM decoder ───────────────────────────────────────────────────
  json lm_nntr = buildComponentNntrCfg("llm_config");
  lm_nntr["model_type"] = "causallm";
  lm_nntr["use_embedding"] = true;

  const std::string llm_type = lm_nntr.value("llm_model_type",
                                   lm_nntr.value("model_type", "Lfm2CausalLM"));
  if (llm_type == "Lfm2CausalLM") {
    lm_ = std::make_unique<Lfm2CausalLM>(text_cfg, generation_cfg_, lm_nntr);
  } else {
    throw std::invalid_argument(
      "VjepaLfm2ForConditionalGeneration: unsupported LLM model_type: " +
      llm_type);
  }
}

/* -------------------------------------------------------------------------
 * initialize
 * ---------------------------------------------------------------------- */

void VjepaLfm2ForConditionalGeneration::initialize() {
  vjepa_->initialize();
  projector_->initialize();
  lm_->initialize();
  initialized_ = true;
}

void VjepaLfm2ForConditionalGeneration::initialize(
  const std::string &native_lib_dir) {
  vjepa_->initialize(native_lib_dir);
  projector_->initialize(native_lib_dir);
  lm_->initialize(native_lib_dir);
  initialized_ = true;
}

/* -------------------------------------------------------------------------
 * load_weight
 * ---------------------------------------------------------------------- */

void VjepaLfm2ForConditionalGeneration::load_weight(
  const std::string &weight_path) {

  // Helper: resolve model file from component config (new) or top-level (legacy)
  auto resolveModelFile = [&](const std::string &component_key,
                              const std::string &component_file_key,
                              const std::string &legacy_key) -> std::string {
    // New structure: component_config.model_file
    if (nntr_cfg_.contains(component_key) &&
        nntr_cfg_[component_key].is_object() &&
        nntr_cfg_[component_key].contains(component_file_key)) {
      return weight_path + "/" +
             nntr_cfg_[component_key][component_file_key].get<std::string>();
    }
    // Legacy: top-level key
    if (nntr_cfg_.contains(legacy_key)) {
      return weight_path + "/" + nntr_cfg_[legacy_key].get<std::string>();
    }
    return "";
  };

  std::cout << "Start Weight Loading" << std::endl;

  // LM weights
  std::string lm_file = resolveModelFile("llm_config", "model_file",
                                          "model_file_name");
  if (!lm_file.empty()) {
    lm_->load_weight(lm_file);
  }

  std::cout << "LLM Loading is completed." << std::endl;

  // ViT weights
  std::string vit_file = resolveModelFile("vision_config", "model_file",
                                           "vision_model_file");
  if (!vit_file.empty()) {
    vjepa_->load_weight(vit_file);
  }

  std::cout << "ViT Loading is completed." << std::endl;

  // Projector weights
  std::string proj_file = resolveModelFile("connector_config", "model_file",
                                            "projector_model_file");
  if (!proj_file.empty()) {
    projector_->load_weight(proj_file);
  }

  std::cout << "Projector Loading is completed." << std::endl;
}

/* -------------------------------------------------------------------------
 * Chat template with <video> placeholders
 * ---------------------------------------------------------------------- */

std::vector<std::string>
VjepaLfm2ForConditionalGeneration::applyVideoChatTemplate(
  const std::string &prompt, float video_duration) const {
  std::vector<std::string> segments;

  // First segment: system prompt + start of user
  segments.push_back("<|startoftext|><|im_start|>system\nYou are a helpful "
                     "assistant.<|im_end|>\n<|im_start|>user\n");

  // Segments between <video> tags (timestamps)
  const float time_per_video = video_duration / num_video_tags_;
  for (unsigned int i = 0; i < num_video_tags_; ++i) {
    char timestamp[32];
    std::snprintf(timestamp, sizeof(timestamp), "<%.1f seconds>",
                  i * time_per_video);
    segments.push_back(std::string(timestamp));
  }

  // Last segment: prompt + end tokens
  segments.push_back(prompt + "<|im_end|>\n<|im_start|>assistant\n");

  return segments;
}

/* -------------------------------------------------------------------------
 * Merge text + video embeddings
 * ---------------------------------------------------------------------- */

std::pair<std::vector<float>, unsigned int>
VjepaLfm2ForConditionalGeneration::mergeTextVideoEmbeddings(
  const std::vector<std::string> &text_segments,
  const float *video_embeds, unsigned int num_video_tokens,
  unsigned int vision_tokens_per_video) {

  auto *tok = lm_->getTokenizer();
  if (!tok) {
    throw std::runtime_error(
      "VjepaLfm2ForConditionalGeneration: LLM tokenizer not available");
  }

  auto [text_cfg, vision_cfg] = splitConfig(cfg_);
  const unsigned int text_dim = textHiddenSizeFromConfig(cfg_, text_cfg);
  const unsigned int init_seq_len =
    nntr_cfg_.value("init_seq_len", 4096u);
  const unsigned int batch_size = nntr_cfg_.value("batch_size", 1u);

  std::vector<float> inputs_embeds(
    static_cast<size_t>(batch_size) * init_seq_len * text_dim, 0.0f);

  size_t embed_offset = 0;
  size_t video_idx = 0;
  const size_t max_tokens = static_cast<size_t>(batch_size) * init_seq_len;

  auto ensure_capacity = [&](size_t tokens_to_add) {
    const size_t used_tokens = embed_offset / text_dim;
    if (used_tokens + tokens_to_add > max_tokens) {
      throw std::runtime_error(
        "VjepaLfm2ForConditionalGeneration: merged embedding length exceeds "
        "init_seq_len capacity (" +
        std::to_string(used_tokens + tokens_to_add) + " > " +
        std::to_string(max_tokens) + ")");
    }
  };

  for (size_t seg_i = 0; seg_i < text_segments.size(); ++seg_i) {
    // Tokenize and embed this text segment
    if (!text_segments[seg_i].empty()) {
      auto enc =
        tok->Encode(text_segments[seg_i], /*add_special_token=*/false);
      ensure_capacity(enc.size());
      for (auto id : enc) {
        auto emb = lm_->lookupEmbedding(static_cast<unsigned int>(id));
        std::copy(emb.begin(), emb.end(),
                  inputs_embeds.data() + embed_offset);
        embed_offset += text_dim;
      }
    }

    // Insert video embeddings after each timestamp segment (segments 1..num_video_tags_)
    // Segments layout: [0]=system_text, [1..N]=timestamps, [N+1]=prompt_end
    // Video embeddings go after each timestamp, not after system_text.
    if (seg_i >= 1 && seg_i <= num_video_tags_) {
      size_t vision_start = video_idx * vision_tokens_per_video;
      if (vision_start + vision_tokens_per_video > num_video_tokens) {
        throw std::runtime_error(
          "VjepaLfm2ForConditionalGeneration: video embedding slice exceeds "
          "projector output token count");
      }
      ensure_capacity(vision_tokens_per_video);
      for (size_t v = 0; v < vision_tokens_per_video; ++v) {
        std::copy(video_embeds + (vision_start + v) * text_dim,
                  video_embeds + (vision_start + v + 1) * text_dim,
                  inputs_embeds.data() + embed_offset);
        embed_offset += text_dim;
      }
      video_idx++;
    }
  }

  unsigned int actual_total_tokens =
    static_cast<unsigned int>(embed_offset / text_dim);
  return {std::move(inputs_embeds), actual_total_tokens};
}

/* -------------------------------------------------------------------------
 * runVisionToLM (common projector + merge + LM path)
 * ---------------------------------------------------------------------- */

void VjepaLfm2ForConditionalGeneration::runVisionToLM(
  const void *vision_ptr, unsigned int num_patches,
  const std::string &prompt, bool do_sample, bool log_output) {

  auto [text_cfg, vision_cfg] = splitConfig(cfg_);
  const unsigned int num_frames = vision_cfg.contains("num_frames")
                                    ? vision_cfg["num_frames"].get<unsigned int>()
                                    : 16u;
  const unsigned int text_dim = textHiddenSizeFromConfig(cfg_, text_cfg);
  const unsigned int output_tokens =
    num_patches / (downsample_factor_ * downsample_factor_);

  if (output_tokens == 0 || num_video_tags_ == 0 ||
      output_tokens % num_video_tags_ != 0) {
    throw std::runtime_error(
      "VjepaLfm2ForConditionalGeneration: projected token count (" +
      std::to_string(output_tokens) +
      ") must be non-zero and divisible by num_video_tags (" +
      std::to_string(num_video_tags_) + ")");
  }

  // ── 1. Projector ─────────────────────────────────────────────────
  auto [proj_ptr, proj_size] =
    projector_->run(static_cast<const float *>(vision_ptr), num_patches,
                    log_output);

  if (log_output) {
    std::cout << "[VJepaLFM2-VL] Vision tokens: " << num_patches
              << " x " << vision_cfg.value("hidden_size", 768u) << "\n";
    std::cout << "[VJepaLFM2-VL] Projected tokens: " << output_tokens
              << " x " << text_dim << "\n";
  }

  // ── 2. Build chat template and merge embeddings ──────────────────
  const float video_duration =
    static_cast<float>(num_frames) / vision_cfg.value("target_fps", 4u);
  const unsigned int vision_tokens_per_video = output_tokens / num_video_tags_;

  auto text_segments = applyVideoChatTemplate(prompt, video_duration);

  const float *proj_data = static_cast<const float *>(proj_ptr);
  auto [inputs_embeds, actual_total_tokens] = mergeTextVideoEmbeddings(
    text_segments, proj_data, output_tokens, vision_tokens_per_video);

  // Build seed tokens for repetition penalty tracking
  // Same interleave order as mergeTextVideoEmbeddings: video after timestamps
  auto *tok = lm_->getTokenizer();
  std::vector<int> seed_tokens;
  for (size_t seg_i = 0; seg_i < text_segments.size(); ++seg_i) {
    if (!text_segments[seg_i].empty()) {
      auto enc =
        tok->Encode(text_segments[seg_i], /*add_special_token=*/false);
      for (auto id : enc)
        seed_tokens.push_back(id);
    }
    if (seg_i >= 1 && seg_i <= num_video_tags_)
      seed_tokens.insert(seed_tokens.end(), vision_tokens_per_video, 0);
  }

  if (log_output) {
    std::cout << "[VJepaLFM2-VL] Vision tokens per <video>: "
              << vision_tokens_per_video << "\n";
    std::cout << "[VJepaLFM2-VL] Total vision tokens: " << output_tokens
              << "\n";
    std::cout << "[VJepaLFM2-VL] Actual total tokens: "
              << actual_total_tokens << "\n";
  }

  // ── 3. Run LFM2 inference ────────────────────────────────────────
  lm_->run_with_embeddings(inputs_embeds.data(), actual_total_tokens,
                           seed_tokens, do_sample, log_output);
}

/* -------------------------------------------------------------------------
 * run_video (with pre-loaded frames)
 * ---------------------------------------------------------------------- */

void VjepaLfm2ForConditionalGeneration::run_video(
  const std::vector<std::vector<float>> &frames,
  const std::string &prompt, bool do_sample, bool log_output) {
  if (!initialized_) {
    throw std::runtime_error(
      "VjepaLfm2ForConditionalGeneration: call initialize() first");
  }

  auto [text_cfg, vision_cfg] = splitConfig(cfg_);
  const unsigned int num_frames = vision_cfg.contains("num_frames")
                                    ? vision_cfg["num_frames"].get<unsigned int>()
                                    : 16u;
  const unsigned int img_size = imageSizeFromVisionConfig(vision_cfg);
  const unsigned int tubelet_size = vision_cfg.value("tubelet_size", 2u);
  const unsigned int patch_size = vision_cfg.value("patch_size", 16u);
  const unsigned int grid_t = num_frames / tubelet_size;
  const unsigned int grid_h = img_size / patch_size;
  const unsigned int grid_w = img_size / patch_size;
  const unsigned int num_patches = grid_t * grid_h * grid_w;

  if (frames.size() != num_frames) {
    throw std::runtime_error(
      "VjepaLfm2ForConditionalGeneration::run_video: frame count mismatch; "
      "got " +
      std::to_string(frames.size()) + ", expected " +
      std::to_string(num_frames));
  }
  const size_t expected_frame_size = static_cast<size_t>(3) * img_size * img_size;
  for (size_t i = 0; i < frames.size(); ++i) {
    if (frames[i].size() != expected_frame_size) {
      throw std::runtime_error(
        "VjepaLfm2ForConditionalGeneration::run_video: frame " +
        std::to_string(i) + " size mismatch; got " +
        std::to_string(frames[i].size()) + ", expected " +
        std::to_string(expected_frame_size));
    }
  }

  // ── 1. VJEPA2 ViT Encoder ────────────────────────────────────────
  auto [vision_ptr, vision_size] =
    vjepa_->run_image(frames, img_size, img_size, log_output);

  // ── 2. Projector + merge + LM ─────────────────────────────────────
  runVisionToLM(vision_ptr, num_patches, prompt, do_sample, log_output);
}

/* -------------------------------------------------------------------------
 * run_video_bin (with .bin file path)
 * ---------------------------------------------------------------------- */

void VjepaLfm2ForConditionalGeneration::run_video_bin(
  const std::string &video_bin_path, int numFrames, int frameHeight,
  int frameWidth, const std::string &prompt, bool do_sample,
  bool log_output) {
  (void)numFrames;
  (void)frameHeight;
  (void)frameWidth;

  if (!initialized_) {
    throw std::runtime_error(
      "VjepaLfm2ForConditionalGeneration: call initialize() first");
  }

  // ── 1. VJEPA2 ViT Encoder (loads bin, patchifies, runs inference) ──
  auto [vision_ptr, vision_size] =
    vjepa_->run_with_bin(video_bin_path, log_output);

  // Compute num_patches from vision config
  auto [text_cfg, vision_cfg] = splitConfig(cfg_);
  const unsigned int num_frames = vision_cfg.contains("num_frames")
                                    ? vision_cfg["num_frames"].get<unsigned int>()
                                    : 16u;
  const unsigned int tubelet_size = vision_cfg.value("tubelet_size", 2u);
  const unsigned int img_size = imageSizeFromVisionConfig(vision_cfg);
  const unsigned int patch_size = vision_cfg.value("patch_size", 16u);
  const unsigned int grid_t = num_frames / tubelet_size;
  const unsigned int grid_h = img_size / patch_size;
  const unsigned int grid_w = img_size / patch_size;
  const unsigned int num_patches = grid_t * grid_h * grid_w;

  // ── 2. Projector + merge + LM ─────────────────────────────────────
  runVisionToLM(vision_ptr, num_patches, prompt, do_sample, log_output);
}

/* -------------------------------------------------------------------------
 * run (text-only, delegates to LLM)
 * ---------------------------------------------------------------------- */

void VjepaLfm2ForConditionalGeneration::run(const WSTR prompt, bool do_sample,
                                             const WSTR system_prompt,
                                             const WSTR tail_prompt,
                                             bool log_output) {
  if (!initialized_) {
    throw std::runtime_error(
      "VjepaLfm2ForConditionalGeneration: call initialize() first");
  }
  lm_->run(prompt, do_sample, system_prompt, tail_prompt, log_output);
}

/* -------------------------------------------------------------------------
 * Delegated LLM interface
 * ---------------------------------------------------------------------- */

const std::vector<unsigned int> &
VjepaLfm2ForConditionalGeneration::getGeneratedIds() const {
  return lm_->getGeneratedIds();
}

tokenizers::Tokenizer *
VjepaLfm2ForConditionalGeneration::getTokenizer() {
  return lm_->getTokenizer();
}

std::string
VjepaLfm2ForConditionalGeneration::getOutput(int batch_idx) const {
  return lm_->getOutput(batch_idx);
}

bool VjepaLfm2ForConditionalGeneration::hasRun() const {
  return lm_->hasRun();
}

TransformerPerformanceMetrics
VjepaLfm2ForConditionalGeneration::getPerformanceMetrics() const {
  return lm_->getPerformanceMetrics();
}

void VjepaLfm2ForConditionalGeneration::requestStop() { lm_->requestStop(); }

void VjepaLfm2ForConditionalGeneration::setStreamer(BaseStreamer *streamer) {
  lm_->setStreamer(streamer);
}

int VjepaLfm2ForConditionalGeneration::getKvLen() const {
  return lm_->getKvLen();
}

size_t VjepaLfm2ForConditionalGeneration::embeddingBytesPerToken() const {
  return lm_->embeddingBytesPerToken();
}

std::pair<float, int>
VjepaLfm2ForConditionalGeneration::get_embedding_info() {
  return lm_->get_embedding_info();
}

void VjepaLfm2ForConditionalGeneration::run_with_embeddings(
  const void *prefill_embeds, size_t n_tokens,
  std::vector<int> seed_tokens, bool do_sample, bool log_output) {
  lm_->run_with_embeddings(prefill_embeds, n_tokens, seed_tokens, do_sample,
                           log_output);
}

} // namespace causallm
