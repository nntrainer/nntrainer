// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   reshaped_rms_norm.h
 * @date   15 July 2025
 * @brief  Implementation of RMS normalization function with reshaping.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __RESHAPED_RMS_NORM_LAYER_H__
#define __RESHAPED_RMS_NORM_LAYER_H__

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
 * @brief Enum class for ReshapedRMSParams index
 */
enum class ReshapedRMSParams { gamma };

/**
 * @brief A Reshaped RMS normalization layer
 *
 * This layer performs RMS normalization after reshaping the input.
 * The input is reshaped from (batch, channel, height, width) to
 * (batch, channel, height * width / feature_size, feature_size)
 * before applying RMS normalization.
 */
class ReshapedRMSNormLayer final : public Layer {
public:
  /**
   * @brief Construct a new Reshaped RMS normalization layer object
   *
   */
  ReshapedRMSNormLayer() : Layer(), wt_idx({0}) {}

  /**
   * @brief Destroy the Reshaped RMS normalization layer object
   *
   */
  ~ReshapedRMSNormLayer() {}

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
  const std::string getType() const override {
    return ReshapedRMSNormLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, rms_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[reshaped_rms_norm] Unknown Layer Properties count " +
           std::to_string(values.size());
  };

  void updateTensorsByInputDimensions(
    RunLayerContext &context, std::vector<TensorDim> input_dimensions) override;

  inline static const std::string type = "reshaped_rms_norm";

private:
  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::GammaInitializer, props::Epsilon, props::FeatureSize>
    rms_props;
};

} // namespace nntrainer

#endif /* __RESHAPED_RMS_NORM_LAYER_H__ */
