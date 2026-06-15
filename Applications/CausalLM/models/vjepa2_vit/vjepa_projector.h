// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   vjepa_projector.h
 * @date   1 June 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  V-JEPA 2.1 Projector / Merger (vision hidden → LLM embedding space).
 *
 * VoRA merger architecture (matching Lfm2VLVJepa21BModel):
 *
 *   pixel_unshuffle(factor=2):
 *     [B, T, H, W, C] → [B, T, H/2, W/2, C*4]
 *     tokens: 4608 → 1152,  dim: 768 → 3072
 *
 *   merger (NUM_MERGER_FC=2, default):
 *     LayerNorm(3072)
 *     FC(3072 → 3072, bias) → GELU
 *     FC(3072 → 1024, bias)
 *
 *   merger (NUM_MERGER_FC=3, legacy):
 *     LayerNorm(3072)
 *     FC(3072 → 3072, bias) → GELU
 *     FC(3072 → 1536, bias) → GELU
 *     LayerNorm(1536)
 *     FC(1536 → 1024, bias)
 *     LayerNorm(1024)
 */

#ifndef __VJEPA_PROJECTOR_H__
#define __VJEPA_PROJECTOR_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief V-JEPA 2.1 Projector (Merger) model
 *
 * Standalone model that projects VJEPA2 encoder hidden states into the LLM
 * embedding space.  Constructed as a minimal Transformer (ModelType::MODEL)
 * with the VoRA merger graph.
 */
class VjepaProjector : virtual public Transformer {

public:
  static constexpr const char *architectures = "VjepaProjector";

  /**
   * @brief Construct a VjepaProjector object.
   */
  VjepaProjector(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Destroy the VjepaProjector object.
   */
  virtual ~VjepaProjector() = default;

  /**
   * @brief Run the projector on VJEPA2 encoder output.
   *
   * Applies pixel_unshuffle internally, then runs the merger MLP.
   *
   * @param vision_embeds  Pointer to NUM_PATCHES * VISION_DIM float values
   *                       (output from VJEPA2ViT::run_image).
   * @param num_tokens     Number of vision tokens (NUM_PATCHES, e.g. 4608).
   * @param log_output     Whether to log output values.
   * @return multimodal_pointer {data_ptr, size_in_bytes} pointing to the
   *         projected embeddings (OUTPUT_TOKENS * TEXT_DIM floats).
   *         Valid until the next call or destruction.
   */
  multimodal_pointer run(const float *vision_embeds, unsigned int num_tokens,
                         bool log_output = true);

  /**
   * @brief Apply pixel_unshuffle on vision embeddings.
   *
   * Reshapes [B, T*H*W, C] → [B, T, H/2, W/2, C*4] → [B, T*(H/2)*(W/2), C*4]
   *
   * @param input     Input data [B, num_tokens, vision_dim]
   * @param output    Output data [B, output_tokens, input_dim] (pre-allocated)
   * @param num_tokens   Total number of input tokens
   * @param temporal_dim Temporal dimension (num_frames / tubelet_size)
   * @param spatial_h    Spatial height (img_size / patch_size)
   * @param spatial_w    Spatial width (img_size / patch_size)
   * @param vision_dim   Vision hidden size (768)
   * @param factor       Downsample factor (2)
   */
  static void pixelUnshuffle(const float *input, float *output,
                              unsigned int num_tokens,
                              unsigned int temporal_dim,
                              unsigned int spatial_h,
                              unsigned int spatial_w,
                              unsigned int vision_dim,
                              unsigned int factor);

protected:
  /**
   * @brief Construct the symbolic projector graph.
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Set projector parameters from config.
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Register custom layers (GELU, LayerNorm).
   */
  void registerCustomLayers() override;

private:
  unsigned int VISION_DIM = 768;       /**< Vision encoder hidden size */
  unsigned int DOWNSAMPLE_FACTOR = 2;  /**< Pixel unshuffle factor */
  unsigned int MERGER_HIDDEN_1 = 3072; /**< Merger FC1 hidden size */
  unsigned int MERGER_HIDDEN_2 = 1536; /**< Merger FC2 hidden size (3-FC only) */
  unsigned int NUM_MERGER_FC = 2;      /**< Number of merger FC layers (2 or 3) */
  unsigned int TEXT_DIM = 1024;        /**< Text model hidden size */
  unsigned int NUM_TOKENS = 4608;      /**< Number of input vision tokens */
  unsigned int OUTPUT_TOKENS = 1152;   /**< Number of output tokens after unshuffle */
  unsigned int INPUT_DIM = 3072;       /**< Input dim after unshuffle (768*4) */

  unsigned int TEMPORAL_DIM = 8;       /**< T = num_frames / tubelet_size */
  unsigned int SPATIAL_H = 24;         /**< H = img_size / patch_size */
  unsigned int SPATIAL_W = 24;         /**< W = img_size / patch_size */

  /** Output from the last run() call. */
  std::vector<float> last_output_;

  /** Buffer for pixel_unshuffle output. */
  std::vector<float> unshuffled_;
};

} // namespace causallm

#endif /* __VJEPA_PROJECTOR_H__ */
