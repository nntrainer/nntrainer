// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   vision_embedding_layer.h
 * @date   17 April 2026
 * @brief  This is the Vision Embedding Layer Class of Neural Network.
 * @details The Vision Embedding Layer maps a sequence of token IDs into their
 *          corresponding word embeddings and dynamically blends in precomputed
 *          image embeddings (e.g., from a Vision Transformer/ViT) wherever
 *          the token ID matches the configured `image_start_token`.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __VISION_EMBEDDING_LAYER_H__
#define __VISION_EMBEDDING_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {
/**
 * @brief Special token for image property
 *
 */
class ImageStartToken : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "image_start_token";                     /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};
} // namespace props

/**
 * @class   VisionEmbeddingLayer
 * @brief   VisionEmbeddingLayer
 * @todo    Support setBatch for VisionEmbeddingLayer
 */
WIN_EXPORT class VisionEmbeddingLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  WIN_EXPORT VisionEmbeddingLayer();

  /**
   * @brief     Destructor of Embedding Layer
   */
  WIN_EXPORT ~VisionEmbeddingLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] VisionEmbeddingLayer &&
   */
  WIN_EXPORT
  VisionEmbeddingLayer(VisionEmbeddingLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs VisionEmbeddingLayer to be moved.
   */
  WIN_EXPORT VisionEmbeddingLayer &
  operator=(VisionEmbeddingLayer &&rhs) = default;

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
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return VisionEmbeddingLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydic Layer::save()
   */
  WIN_EXPORT void save(
    std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    nntrainer::TensorDim::DataType dtype = nntrainer::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "vision_embedding_layer";

private:
  std::tuple<nntrainer::props::InDim, nntrainer::props::OutDim,
             props::ImageStartToken, nntrainer::props::Scale>
    vision_embedding_props;
  unsigned int weight_idx;
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __VISION_EMBEDDING_LAYER_H__ */
