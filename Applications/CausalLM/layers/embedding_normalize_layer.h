// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   embedding_normalize_layer.h
 * @date   06 Jan 2026
 * @brief  This is Embedding Normalize Layer Class
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __EMBEDDING_NORMALIZE_LAYER_H__
#define __EMBEDDING_NORMALIZE_LAYER_H__

#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_impl.h>

namespace causallm {

/**
 * @class   EmbeddingNormalizeLayer
 * @brief   Embedding Normalize Layer
 */
WIN_EXPORT class EmbeddingNormalizeLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of EmbeddingNormalizeLayer
   */
  WIN_EXPORT EmbeddingNormalizeLayer();

  /**
   * @brief     Destructor of EmbeddingNormalizeLayer
   */
  WIN_EXPORT ~EmbeddingNormalizeLayer() = default;

  /**
   * @copydoc   Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc   Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
   * @copydoc   Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc   Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc   Layer::calcGradient(RunLayerContext &context)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc   Layer::exportTo(Exporter &exporter, const ExportMethods &method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc   Layer::getType()
   */
  const std::string getType() const override {
    return EmbeddingNormalizeLayer::type;
  }

  /**
   * @copydoc   Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "embedding_normalize";
};

} // namespace causallm

#endif /* __EMBEDDING_NORMALIZE_LAYER_H__ */
