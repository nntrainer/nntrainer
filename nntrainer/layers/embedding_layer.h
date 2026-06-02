// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding_layer.h
 * @date   04 March 2021
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __EMBEDDING_LAYER_H__
#define __EMBEDDING_LAYER_H__
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

/**
 * @class   EmbeddingLayer
 * @brief   EmbeddingLayer
 * @todo    Support setBatch for EmbeddingLayer
 */
WIN_EXPORT class EmbeddingLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  WIN_EXPORT EmbeddingLayer();

  /**
   * @brief     Destructor of Embedding Layer
   */
  WIN_EXPORT ~EmbeddingLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] EmbeddingLayer &&
   */
  WIN_EXPORT EmbeddingLayer(EmbeddingLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs EmbeddingLayer to be moved.
   */
  WIN_EXPORT EmbeddingLayer &operator=(EmbeddingLayer &&rhs) = default;

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
    return EmbeddingLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::save()
   */
  WIN_EXPORT void save(
    std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    nntrainer::TensorDim::DataType dtype = nntrainer::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  inline static const std::string type = "embedding_layer";

private:
  std::tuple<nntrainer::props::InDim, nntrainer::props::OutDim,
             nntrainer::props::Scale>
    embedding_props;
  unsigned int weight_idx;
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __EMBEDDING_LAYER_H__ */
