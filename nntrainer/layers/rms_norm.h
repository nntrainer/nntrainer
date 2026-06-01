// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.h
 * @date   11 July 2025
 * @brief  Implementation of RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_RMS_NORM_LAYER_H__
#define __NNTRAINER_RMS_NORM_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

#include <common_properties.h>
#include <connection.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace nntrainer {

/**
 * @brief Enum class for RMSParams index
 */
enum class RMSParams { gamma };

/**
 * @brief A RMS normalization layer
 *
 */
class RMSNormLayer final : public Layer {
public:
  /**
   * @brief Construct a new RMS normalization layer object
   *
   */
  RMSNormLayer() : Layer(), wt_idx({0}) {}

  /**
   * @brief Destroy the RMS normalization layer object
   *
   */
  ~RMSNormLayer() {}

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
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

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
  const std::string getType() const override { return RMSNormLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, rms_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[rms_norm] Unknown Layer Properties count " +
           std::to_string(values.size());
  };

  void updateTensorsByInputDimensions(
    RunLayerContext &context, std::vector<TensorDim> input_dimensions) override;

  inline static const std::string type = "rms_norm";

private:
  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::GammaInitializer, props::Epsilon> rms_props;
};

} // namespace nntrainer

#endif /* __NNTRAINER_RMS_NORM_LAYER_H__ */
