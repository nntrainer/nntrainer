// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   timm_vit_transformer.h
 * @date   28 Jan 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief   This timm_vit_transformer.h constructs a class for timm ViT model
 * compatible with the PyTorch timm library.
 */

#ifndef __TIMM_VIT_TRANSFORMER_H__
#define __TIMM_VIT_TRANSFORMER_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief TimmViTTransformer class
 */
class TimmViTTransformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "TimmViT";

  /**
   * @brief Construct a TimmViTTransformer object.
   */
  TimmViTTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Destroy the TimmViTTransformer object.
   */
  virtual ~TimmViTTransformer() = default;

public:
  /**
   * @brief Create patch embedding layers for an input image tensor.
   */
  Tensor createPatchEmbed(Tensor input);

  /**
   * @brief Create the optional classifier head (pool + norm + linear).
   *
   * A no-op when NUM_CLASSES == 0, which is the feature-extraction case: the
   * graph then ends at the encoder's final LayerNorm.
   */
  Tensor createHead(Tensor input);

  // The 2-arg ViT variants below take a single input tensor, while the base
  // Transformer::createAttention / createMlp expose the (seq_len, n_heads,
  // ..., ...) causal-LM signatures. Bring the base overloads into scope so
  // the ViT-specific 2-arg versions overload them instead of hiding them
  // (which -Werror=overloaded-virtual rejects).
  using Transformer::createAttention;
  using Transformer::createMlp;

  /**
   * @brief Create a ViT self-attention block for a transformer layer.
   */
  Tensor createAttention(const int layer_id, Tensor input);

  /**
   * @brief Create a ViT MLP block for a transformer layer.
   */
  Tensor createMlp(const int layer_id, Tensor input);

protected:
  /**
   * @brief Construct the ViT graph and return input/output tensors.
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Set model parameters from HuggingFace and nntrainer configs.
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Create a ViT transformer encoder block.
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  /**
   * @brief Register custom layers required by the base transformer.
   */
  void registerCustomLayers() override;

  /**
   * @brief Run the model (override for ViT specific behavior)
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

protected:
  // Input geometry. Height/width are kept separate so non-square inputs work
  // (an audio mel-spectrogram is n_mels x frames, e.g. 64 x 1012); `img_size`
  // in the config still sets both at once for square image models.
  unsigned int INPUT_HEIGHT = 224; /**< Input height (n_mels for audio) */
  unsigned int INPUT_WIDTH = 224;  /**< Input width (frames for audio) */
  unsigned int PATCH_SIZE = 16;    /**< Patch height/width */
  unsigned int PATCH_STRIDE = 16;  /**< Patch stride */
  unsigned int GRID_H = 14;        /**< Patches along height */
  unsigned int GRID_W = 14;        /**< Patches along width */
  unsigned int NUM_PATCHES = 196;  /**< GRID_H * GRID_W */
  unsigned int IMG_CHANNELS = 3;   /**< Input channels (1 for a spectrogram) */

  // Optional per-row input normalization: a BatchNorm applied across the
  // height axis of the raw input, before the patch conv. CED normalizes its
  // mel-spectrogram this way (BatchNorm2d(n_mels) with frequency as the
  // channel axis); image ViTs have nothing there.
  bool USE_INPUT_NORM = false;  /**< Prepend a height-axis BatchNorm */
  float INPUT_NORM_EPS = 1e-5f; /**< Epsilon of that BatchNorm */

  // Optional classifier head. NUM_CLASSES == 0 means "no head": the graph
  // output is the encoder feature map, which is what the timm SigLIP ViT
  // configuration uses.
  unsigned int NUM_CLASSES = 0;           /**< Head output units, 0 = no head */
  std::string POOLING = "mean";           /**< Token pooling before the head */
  unsigned int POOLING_HIDDEN_SIZE = 256; /**< xi pooling hidden width */
  float POOLING_NORM_EPS = 1e-5f;         /**< xi pooling BatchNorm epsilon */
  bool HEAD_SIGMOID = false;   /**< Apply sigmoid to the head output */
  float HEAD_NORM_EPS = 1e-5f; /**< Head LayerNorm epsilon */
  // The final projection's dtype is controlled separately from the encoder's
  // FC layers, the same split LLMs use for their LM head: a small output
  // projection is often the one worth leaving unquantized.
  std::string HEAD_DTYPE = "FP32"; /**< Head classifier weight dtype */
};

} // namespace causallm

#endif /* __TIMM_VIT_TRANSFORMER_H__ */
