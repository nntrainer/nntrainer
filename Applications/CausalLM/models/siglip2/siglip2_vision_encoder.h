// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   siglip2_vision_encoder.h
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  SigLIP2 ViT-B/16 vision encoder with enc_to_dec projection.
 *         Produces [1, 196, 256] projected encoder hidden states.
 */

#ifndef __SIGLIP2_VISION_ENCODER_H__
#define __SIGLIP2_VISION_ENCODER_H__

#include <transformer.h>
#include <vector>

namespace causallm {

/**
 * @brief Siglip2VisionEncoder class
 *
 * Implements the SigLIP2 ViT-B/16 vision encoder. Architecture:
 *   - Conv2D patch embed (with bias)
 *   - Learned position embedding
 *   - 12x transformer blocks (pre-LN, separate Q/K/V, tanh_gelu MLP)
 *   - Post layer norm
 *   - Linear projection enc_to_dec_proj: [768 -> 256]
 *
 * Output: [1, 196, 256] projected encoder hidden states.
 */
class Siglip2VisionEncoder : virtual public Transformer {

public:
  static constexpr const char *architectures = "Siglip2VisionEncoder";
  static constexpr unsigned int ENC_TO_DEC_DIM = 256;

  /**
   * @brief Construct a Siglip2VisionEncoder object.
   */
  Siglip2VisionEncoder(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Destroy the Siglip2VisionEncoder object.
   */
  virtual ~Siglip2VisionEncoder() = default;

  /**
   * @brief Run the encoder on an image and return projected hidden states.
   * @param image_path Path to input image file.
   * @return Projected encoder hidden states [1, 196, 256] as raw float buffer.
   *         The buffer is owned by this object and valid until the next call.
   */
  std::vector<float> encode(const std::string &image_path);

  /**
   * @brief Run the encoder on pre-loaded CHW float pixels (bypass C++ resize).
   *
   * Accepts a pre-normalized [1,3,IMG_SIZE,IMG_SIZE] float32 buffer (e.g.
   * from PyTorch/PIL preprocessing) so encoder math can be compared against
   * a golden produced with identical input pixels.
   *
   * @param pixel_data Pointer to 1*3*IMG_SIZE*IMG_SIZE floats in CHW order.
   * @param pixel_count Total number of floats (must equal 3*IMG_SIZE*IMG_SIZE).
   * @return Projected encoder hidden states [1, 196, 256].
   *
   * @note Internal/debug-only tooling — not part of the production interface.
   *       Production callers (Task 5+) use only initialize(), load_weight(),
   *       and encode(). This method exists solely for parity verification
   *       (verify_parity.py --dump) and may be removed or restricted in
   *       future releases.
   */
  std::vector<float> encodePixels(const float *pixel_data, size_t pixel_count);

protected:
  /**
   * @brief Create patch embedding layers.
   */
  Tensor createPatchEmbed(Tensor input);

  /**
   * @brief Create a pre-normalized self-attention block.
   */
  Tensor createAttention(const int layer_id, Tensor input);

  /**
   * @brief Create a pre-normalized feed-forward block with tanh_gelu.
   */
  Tensor createMlp(const int layer_id, Tensor input);

protected:
  /**
   * @brief Construct the SigLIP2 encoder graph and return input/output tensors.
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Set model parameters from HuggingFace and nntrainer configs.
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Create a transformer encoder block with residual connections.
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  /**
   * @brief Register custom layers required by the base transformer.
   */
  void registerCustomLayers() override;

  /**
   * @brief Run the encoder on an image (override for encoder-specific
   * behavior).
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

private:
  unsigned int IMG_SIZE = 224;    /**< Image height/width */
  unsigned int PATCH_SIZE = 16;   /**< Patch height/width */
  unsigned int NUM_PATCHES = 196; /**< Number of patches */
  unsigned int IMG_CHANNELS = 3;  /**< Image channels (RGB) */

  /** Owned buffer for the last encode() call output. */
  std::vector<float> enc_output_buf_;
};

} // namespace causallm

#endif /* __SIGLIP2_VISION_ENCODER_H__ */
