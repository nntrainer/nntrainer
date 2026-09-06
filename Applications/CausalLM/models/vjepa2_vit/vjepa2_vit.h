// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vjepa2_vit.h
 * @date   21 May 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  V-JEPA 2.1 ViT encoder (ViT-B/16, video) for nntrainer.
 *
 * @note   Faithful reproduction of `vjepa2_1_vit_base_384` encoder:
 *           - 3D tubelet patch embedding (Conv3d, non-overlapping) implemented
 *             as a host-side patchify + a single fully_connected projection.
 *           - 3D axial RoPE applied to Q/K via the custom `vjepa_rope` layer;
 *             mha_core runs with rope disabled (rope_theta=0, is_causal=false).
 *           - GELU MLP, LayerNorm (eps 1e-6), qkv_bias=True, no CLS token,
 *             no learned positional embedding.
 *           - The video modality embedding is folded into the patch-embed bias
 *             by the weight converter (constant added to every token).
 */

#ifndef __VJEPA2_VIT_H__
#define __VJEPA2_VIT_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief V-JEPA2 ViT encoder model
 */
class VJEPA2ViT : virtual public Transformer {

public:
  static constexpr const char *architectures = "VJEPA2ViT";

  /**
   * @brief Construct a VJEPA2ViT object.
   */
  VJEPA2ViT(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Destroy the VJEPA2ViT object.
   */
  virtual ~VJEPA2ViT() = default;

public:
  /**
   * @brief Create the tubelet patch-embedding projection.
   */
  Tensor createPatchEmbed(Tensor input);

  /**
   * @brief Create a pre-normalized self-attention block with 3D RoPE.
   */
  Tensor createAttention(const int layer_id, Tensor input);

  /**
   * @brief Create a pre-normalized GELU feed-forward block.
   */
  Tensor createMlp(const int layer_id, Tensor input);

protected:
  /**
   * @brief Construct the symbolic ViT inference graph.
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Set model parameters from HuggingFace and nntrainer configs.
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Create one ViT transformer block with residual connections.
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  /**
   * @brief Register custom layers used by this model (vjepa_rope, mha_core).
   */
  void registerCustomLayers() override;

  /**
   * @brief Run the encoder on a preprocessed video tensor file.
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

private:
  unsigned int IMG_SIZE = 384;      /**< Image height/width */
  unsigned int PATCH_SIZE = 16;     /**< Spatial patch size */
  unsigned int TUBELET = 2;         /**< Temporal tubelet size */
  unsigned int NUM_FRAMES = 64;     /**< Number of input frames */
  unsigned int IN_CHANS = 3;        /**< Input channels (RGB) */
  unsigned int GRID_T = 32;         /**< Temporal grid (NUM_FRAMES / TUBELET) */
  unsigned int GRID_H = 24;         /**< Height grid (IMG_SIZE / PATCH_SIZE) */
  unsigned int GRID_W = 24;         /**< Width grid (IMG_SIZE / PATCH_SIZE) */
  unsigned int NUM_PATCHES = 18432; /**< GRID_T * GRID_H * GRID_W */
  unsigned int PATCH_VEC = 1536;    /**< IN_CHANS * TUBELET * PATCH_SIZE^2 */
  unsigned int PRETRAINED_GRID = 16; /**< 256 / PATCH_SIZE for rope interp */
  bool INTERPOLATE_ROPE = true;      /**< V-JEPA 2.1 uses rope interpolation */

  /**
   * @brief Extract non-overlapping tubelets from a [C,T,H,W] float buffer into
   *        a [NUM_PATCHES, PATCH_VEC] token matrix, ordered to match Conv3d.
   */
  std::vector<float> patchify(const std::vector<float> &video) const;
};

} // namespace causallm

#endif /* __VJEPA2_VIT_H__ */
