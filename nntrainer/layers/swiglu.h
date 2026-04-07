// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   swiglu.h
 * @date   14 July 2023
 * @brief  Implementation of custom SwiGLU activation function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __SWIGLU_LAYER_H__
#define __SWIGLU_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace nntrainer {

/**
 * @brief A SwiGLU layer for llama.
 *
 */
class SwiGLULayer final : public Layer {
public:
  /**
   * @brief Construct a new custom SwiGLU layer object
   *
   */
  SwiGLULayer() = default;

  /**
   * @brief Destroy the custom SwiGLU layer object
   *
   */
  ~SwiGLULayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};
  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return SwiGLULayer::type; };
  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override{};

  void updateTensorsByInputDimensions(
    RunLayerContext &context, std::vector<TensorDim> input_dimensions) override;

  inline static const std::string type = "swiglu";
};

} // namespace nntrainer

#endif /* __SWIGLU_LAYER_H__ */
