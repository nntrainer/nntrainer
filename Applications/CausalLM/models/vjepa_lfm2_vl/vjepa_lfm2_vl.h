// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   vjepa_lfm2_vl.h
 * @date   9 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  V-JEPA 2.1 + Projector + LFM2 multimodal video-language model.
 *
 *         Pipeline:
 *           run_video():  frames → VJEPA2ViT::run_image()
 *           run_video_bin(): .bin → VJEPA2ViT::run_with_bin()
 *           → runVisionToLM() → VjepaProjector::run()
 *           → merge text+vision embeddings
 *           → Lfm2CausalLM::run_with_embeddings() → generate tokens
 *
 *         Single-class approach: all components are owned by this class,
 *         encapsulating the full video-language pipeline.
 *
 *         Config structure (config.json):
 *         {
 *           "architectures": ["Lfm2VLVJepa21BModel"],
 *           "model_type": "vora_lfm2_vl_vjepa2_1_b",
 *           "hidden_size": 1024,
 *           "downsample_factor": 2,
 *           "projector_hidden_size": 2048,
 *           "projector_bias": true,
 *           "projector_hidden_act": "gelu",
 *           "projector_use_layernorm": false,
 *           "video_token_id": 64400,
 *           "image_token_id": 396,
 *           "text_config": { ...LFM2 LM config... },
 *           "vision_config": { ...VJEPA2 ViT config... }
 *         }
 *
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __VJEPA_LFM2_VL_H__
#define __VJEPA_LFM2_VL_H__

#include <lfm2_causallm.h>
#include <transformer.h>
#include <vjepa2_vit/vjepa2_vit.h>
#include <vjepa2_vit/vjepa_projector.h>

#include <memory>
#include <string>
#include <vector>

namespace causallm {

/**
 * @brief V-JEPA 2.1 + LFM2 Video-Language model
 *
 * Owns:
 *   - VJEPA2ViT          (vision encoder)
 *   - VjepaProjector     (vision → text embedding space projection)
 *   - Lfm2CausalLM       (text decoder LM)
 */
class VjepaLfm2ForConditionalGeneration : virtual public Transformer {

public:
  static constexpr const char *architectures = "Lfm2VLVJepa21BModel";

  /**
   * @brief Construct the VL model from config / nntr_config.
   */
  VjepaLfm2ForConditionalGeneration(json &cfg, json &generation_cfg,
                                     json &nntr_cfg);

  /**
   * @brief Destroy the VL model.
   */
  virtual ~VjepaLfm2ForConditionalGeneration() = default;

  /**
   * @brief Initialize all sub-models (build graphs, compile).
   */
  void initialize() override;
  void initialize(const std::string &native_lib_dir) override;

  /**
   * @brief Load weights for ViT, projector, and LM.
   */
  void load_weight(const std::string &weight_path) override;

  /**
   * @brief Run the full VL pipeline with a preprocessed .bin video file.
   *
   * The .bin file is raw float32 in [C, T, H, W] layout.
   *
   * @param video_bin_path  Path to preprocessed .bin video tensor file
   * @param numFrames       Number of frames in the file
   * @param frameHeight     Frame height (default 256)
   * @param frameWidth      Frame width (default 256)
   * @param prompt          Text prompt
   * @param do_sample       Whether to sample during generation
   * @param log_output      Whether to log output
   */
  void run_video_bin(const std::string &video_bin_path, int numFrames,
                     int frameHeight, int frameWidth,
                     const std::string &prompt, bool do_sample = false,
                     bool log_output = true);

  /**
   * @brief Run the full VL pipeline with pre-loaded frames.
   *
   * @param frames          Vector of frame tensors, each [C, H, W]
   * @param prompt          Text prompt
   * @param do_sample       Whether to sample during generation
   * @param log_output      Whether to log output
   */
  void run_video(const std::vector<std::vector<float>> &frames,
                 const std::string &prompt, bool do_sample = false,
                 bool log_output = true);

  /**
   * @brief Run with text-only prompt (no video). Delegates to LLM.
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(),
           const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

  // ── Delegated LLM interface ──────────────────────────────────────────

  const std::vector<unsigned int> &getGeneratedIds() const;
  tokenizers::Tokenizer *getTokenizer();
  std::string getOutput(int batch_idx = 0) const override;
  bool hasRun() const;
  TransformerPerformanceMetrics getPerformanceMetrics() const;
  void requestStop() override;
  void setStreamer(::BaseStreamer *streamer) override;
  int getKvLen() const override;
  size_t embeddingBytesPerToken() const override;
  std::pair<float, int> get_embedding_info() override;
  void run_with_embeddings(const void *prefill_embeds, size_t n_tokens,
                           std::vector<int> seed_tokens, bool do_sample,
                           bool log_output) override;

private:
  json cfg_;
  json generation_cfg_;
  json nntr_cfg_;

  unsigned int num_video_tags_{12};
  unsigned int downsample_factor_{2};
  int video_token_id_{64400};
  int image_token_id_{396};
  unsigned int projector_hidden_size_{2048};

  std::unique_ptr<VJEPA2ViT> vjepa_;
  std::unique_ptr<VjepaProjector> projector_;
  std::unique_ptr<Lfm2CausalLM> lm_;

  bool initialized_{false};

  /**
   * @brief Extract text_config and vision_config from the top-level config.
   */
  static std::pair<json, json> splitConfig(const json &cfg);

  /**
   * @brief Build projector config from top-level VL config fields.
   */
  json buildProjectorConfig(const json &cfg) const;

  /**
   * @brief Build chat template text segments with <video> placeholders.
   */
  std::vector<std::string>
  applyVideoChatTemplate(const std::string &prompt,
                         float video_duration) const;

  /**
   * @brief Merge text token embeddings and vision embeddings.
   */
  std::pair<std::vector<float>, unsigned int>
  mergeTextVideoEmbeddings(
    const std::vector<std::string> &text_segments,
    const float *video_embeds, unsigned int num_video_tokens,
    unsigned int vision_tokens_per_video);

  /**
   * @brief Run the projector + merge + LM pipeline from vision encoder output.
   *
   * Common path shared by run_video() and run_video_bin().
   *
   * @param vision_ptr    Pointer to vision encoder output (NUM_PATCHES * DIM floats).
   * @param num_patches   Number of vision patches from the encoder.
   * @param prompt        Text prompt.
   * @param do_sample     Whether to sample during generation.
   * @param log_output    Whether to log output.
   */
  void runVisionToLM(const void *vision_ptr, unsigned int num_patches,
                     const std::string &prompt, bool do_sample,
                     bool log_output);
};

} // namespace causallm

#endif /* __VJEPA_LFM2_VL_H__ */
