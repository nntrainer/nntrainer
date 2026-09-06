// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vjepa_rope_layer.h
 * @date   21 May 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  3D axial Rotary Positional Embedding layer for V-JEPA2 ViT.
 *
 * @note   This layer reproduces V-JEPA 2.1's `rotate_queries_or_keys`
 *         (see app/vjepa_2_1/models/utils/modules.py). It is attached right
 *         after the Q (or K) projection and before mha_core, which must be run
 *         with rope disabled (rope_theta=0). The per-head feature dimension is
 *         split into three contiguous axis slices of size `d_dim` each
 *         (depth/frame, height, width); each slice is rotated using the token's
 *         (frame, height, width) grid index. Any tail dimension beyond
 *         `3 * d_dim` is left unrotated. Because the (T, H, W) grid is fixed at
 *         inference time, the cos/sin tables are constants precomputed once in
 *         finalize(); the layer therefore owns no trainable weights.
 */

#ifndef __VJEPA_ROPE_LAYER_H__
#define __VJEPA_ROPE_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>
#include <vector>

#include <base_properties.h>

namespace causallm {

namespace props {

/**
 * @brief Number of attention heads (used to derive head_dim = width/num_heads)
 */
class VjepaNumHeads : public nntrainer::PositiveIntegerProperty {
public:
  VjepaNumHeads(unsigned int value = 12) { set(value); };
  static constexpr const char *key = "num_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Temporal grid size (number of tubelet frames, T = num_frames/tubelet)
 */
class VjepaGridT : public nntrainer::PositiveIntegerProperty {
public:
  VjepaGridT(unsigned int value = 1) { set(value); };
  static constexpr const char *key = "grid_t";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Height grid size in patches (H_patches = img_h/patch_size)
 */
class VjepaGridH : public nntrainer::PositiveIntegerProperty {
public:
  VjepaGridH(unsigned int value = 1) { set(value); };
  static constexpr const char *key = "grid_h";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Width grid size in patches (W_patches = img_w/patch_size)
 */
class VjepaGridW : public nntrainer::PositiveIntegerProperty {
public:
  VjepaGridW(unsigned int value = 1) { set(value); };
  static constexpr const char *key = "grid_w";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Rope base theta (default 10000)
 */
class VjepaRopeTheta : public nntrainer::Property<float> {
public:
  VjepaRopeTheta(float value = 10000.0f) { set(value); };
  static constexpr const char *key = "rope_theta";
  using prop_tag = nntrainer::float_prop_tag;
};

/**
 * @brief Pretrained grid size for rope position interpolation
 *        (e.g. 256/patch_size = 16 for patch_size 16)
 */
class VjepaPretrainedGridSize : public nntrainer::PositiveIntegerProperty {
public:
  VjepaPretrainedGridSize(unsigned int value = 16) { set(value); };
  static constexpr const char *key = "pretrained_grid_size";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Whether to interpolate the height/width rope positions to the
 *        pretrained grid (V-JEPA 2.1 sets interpolate_rope=True)
 */
class VjepaInterpolateRope : public nntrainer::Property<bool> {
public:
  VjepaInterpolateRope(bool value = false) { set(value); };
  static constexpr const char *key = "interpolate_rope";
  using prop_tag = nntrainer::bool_prop_tag;
};

} // namespace props

/**
 * @brief 3D axial Rotary Positional Embedding layer for V-JEPA2 ViT encoder.
 */
WIN_EXPORT class VjepaRopeLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new VjepaRopeLayer object
   */
  WIN_EXPORT VjepaRopeLayer() :
    Layer(),
    vjepa_rope_props(props::VjepaNumHeads(), props::VjepaGridT(),
                     props::VjepaGridH(), props::VjepaGridW(),
                     props::VjepaRopeTheta(), props::VjepaPretrainedGridSize(),
                     props::VjepaInterpolateRope()),
    head_dim(0),
    d_dim(0) {}

  /**
   * @brief Destroy the VjepaRopeLayer object
   */
  WIN_EXPORT ~VjepaRopeLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {
    exporter.saveResult(vjepa_rope_props, method, this);
  };

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return VjepaRopeLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "vjepa_rope";

private:
  std::tuple<props::VjepaNumHeads, props::VjepaGridT, props::VjepaGridH,
             props::VjepaGridW, props::VjepaRopeTheta,
             props::VjepaPretrainedGridSize, props::VjepaInterpolateRope>
    vjepa_rope_props;

  unsigned int head_dim; /**< per-head feature dimension (width / num_heads) */
  unsigned int d_dim;    /**< per-axis rotated dimension = 2*((head_dim/3)/2) */

  /** constant rotation tables, sized [num_tokens][head_dim]. cos/sin are
   *  repeat-interleaved across pairs and set to 1/0 for the unrotated tail. */
  std::vector<std::vector<float>> cos_table;
  std::vector<std::vector<float>> sin_table;

  /**
   * @brief Precompute the cos/sin rotation tables from the grid configuration.
   */
  void precompute_tables();

  /**
   * @brief Apply the 3D axial rotation to one [num_tokens, width] block.
   * @param in   input data pointer (row-major, width-contiguous)
   * @param out  output data pointer
   * @param from absolute token index of the first row
   * @param rows number of token rows to process
   * @param width feature width (= num_heads * head_dim)
   */
  template <typename T>
  void rotate(const T *in, T *out, unsigned int from, unsigned int rows,
              unsigned int width) const;
};

} // namespace causallm

#endif /* __VJEPA_ROPE_LAYER_H__ */
