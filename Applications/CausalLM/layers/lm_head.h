// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lm_head.h
 * @date   16 Jan 2026
 * @brief  This is LM_Head Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LM_HEAD_H__
#define __LM_HEAD_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_devel.h>
#include <layer_impl.h>

namespace causallm {

/**
 * @class   LMHead layer
 * @brief   LMHead layer
 */
WIN_EXPORT class LmHeadLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  WIN_EXPORT LmHeadLayer();

  /**
   * @brief     Destructor of Embedding Layer
   */
  WIN_EXPORT ~LmHeadLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] LmHeadLayer &&
   */
  WIN_EXPORT LmHeadLayer(LmHeadLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs LmHeadLayer to be moved.
   */
  WIN_EXPORT LmHeadLayer &operator=(LmHeadLayer &&rhs) = default;

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
    return LmHeadLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; }

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "lm_head";

private:
  std::tuple<nntrainer::props::Unit> lmhead_props;
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
};
} // namespace causallm

#endif
#endif
