// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   siglip_vision.h
 * @brief  SiglipVisionModel — the vision half of PaliGemma (tower +
 *         multimodal projector), producing LLM-space image embeddings.
 * @date   12 Nov 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __SIGLIP_VISION_H__
#define __SIGLIP_VISION_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief SiglipVisionModel class
 *
 * HF SiglipVisionModel (siglip_vision_model, no pooling head) followed by the
 * PaliGemma multi-modal projector Linear(vision_hidden -> projection_dim,
 * bias). The graph is the TimmViT pattern: conv patch embed -> +pos embed ->
 * N pre-norm blocks (LayerNorm -> biased FC q/k/v -> non-causal mha_core
 * (no rope) -> biased FC out -> residual; LayerNorm -> biased FC up ->
 * tanh_gelu -> biased FC down -> residual) -> final post_layernorm ->
 * projector.
 *
 * Output: NUM_PATCHES (256) rows of projection_dim (2048) floats in LLM
 * space, ready to be injected into the text decoder's embedding plane
 * (raw — HF scales only the TEXT token embeddings).
 */
class SiglipVisionModel : virtual public Transformer {

public:
  static constexpr const char *architectures = "SiglipVisionModel";

  SiglipVisionModel(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeVisionConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~SiglipVisionModel() = default;

protected:
  bool WEIGHT_TRANSPOSED = false;
  LayerHandle createModelLayer(const std::string &type,
                               std::vector<std::string> properties);
  /**
   * @brief Flatten PaliGemma's nested config into the vision view
   *
   * config.json carries the tower under "vision_config" and omits
   * image_size (derived: sqrt(num_image_tokens) * patch_size), the LayerNorm
   * epsilon (1e-6) and the hidden activation (gelu_pytorch_tanh). The
   * projector keys (projection_dim) stay at the top level.
   */
  static json &sanitizeVisionConfig(json &cfg);

  /**
   * @brief Set vision model parameters from the flattened configs
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Construct the SigLIP+projector graph
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Create the patch + position embedding sub-graph
   */
  Tensor createPatchEmbed(Tensor input);

  /**
   * @brief Create one pre-norm SigLIP attention block
   */
  Tensor createAttention(const int layer_id, Tensor input);

  /**
   * @brief Create one pre-norm SigLIP MLP block
   */
  Tensor createMlp(const int layer_id, Tensor input);

  /**
   * @brief Create one encoder block with residual connections
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  /**
   * @brief Register custom layers (the core CausalLM set covers mha_core)
   */
  void registerCustomLayers() override;

public:
  /** @brief Decode RGB, antialiased bicubic resize, and normalize to CHW. */
  std::vector<float> preprocessImage(const std::string &path) const;

  /**
   * @brief Run vision inference on an image file path and print the first
   *        output values (verification entry point, TimmViT style).
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

  /**
   * @brief Embedding-PRODUCER: encode an image into LLM-space embeddings
   *        ([NUM_PATCHES, projection_dim] floats) and return a heap buffer
   *        (caller frees).
   * @note  @p image, when carrying a preprocessed CHW float buffer of
   *        3 * IMG_SIZE * IMG_SIZE floats (e.g. the HF processor's
   *        pixel_values), is fed verbatim; otherwise the image file at
   *        @p prompt is loaded (antialiased bicubic resize + (v/255-0.5)/0.5).
   */
  multimodal_pointer run_image(const WSTR prompt, multimodal_pointer image,
                               int image_height, int image_width,
                               bool do_sample = false,
                               const WSTR system_prompt = WSTR(),
                               const WSTR tail_prompt = WSTR(),
                               bool log_output = true) override;

private:
  unsigned int IMG_SIZE = 224;        /**< Image height/width */
  unsigned int PATCH_SIZE = 14;       /**< Patch height/width */
  unsigned int NUM_PATCHES = 256;     /**< Number of patches */
  unsigned int IMG_CHANNELS = 3;      /**< Image channels (RGB) */
  unsigned int PROJECTION_DIM = 2048; /**< Projector output dim (LLM hidden) */
};

} // namespace causallm

#endif /* __SIGLIP_VISION_H__ */
