// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Joonseok Oh <jrock.oh@samsung.com>
 *
 * @file   rms_reverse_norm.h
 * @date   27 March 2026
 * @brief  This is Reverse RMS Norm Layer Class
 * @see    https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __RMS_REVERSE_NORM_LAYER_H__
#define __RMS_REVERSE_NORM_LAYER_H__

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

#include <causallm_common_properties.h>
#include <connection.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace causallm {

namespace props {

/**
 * @brief RMS_REVERSE_NORM_WEIGHT_INIT Initialization Enumeration Information
 *
 */
WIN_EXPORT class RMS_REVERSE_NORM_WEIGHT_INIT final
  : public nntrainer::EnumProperty<nntrainer::props::InitializerInfo> {
public:
  /**
   * @brief Construct a RMS_REVERSE_NORM_WEIGHT_INIT object
   */
  WIN_EXPORT RMS_REVERSE_NORM_WEIGHT_INIT(
    nntrainer::Initializer value = nntrainer::Initializer::ONES) {
    set(value);
  };

  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "weight_initializer";
};

/**
 * @brief RMS_REVERSE_NORM_OUTSCALE_INIT Initialization Enumeration Information
 *
 */
WIN_EXPORT class RMS_REVERSE_NORM_OUTSCALE_INIT final
  : public nntrainer::EnumProperty<nntrainer::props::InitializerInfo> {
public:
  /**
   * @brief Construct a RMS_REVERSE_NORM_OUTSCALE_INIT object
   */
  WIN_EXPORT RMS_REVERSE_NORM_OUTSCALE_INIT(
    nntrainer::Initializer value = nntrainer::Initializer::ONES) {
    set(value);
  };

  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "outscale_initializer";
};

}; // namespace props

/**
 * @brief A ReverseRMS normalization layer.
 *        Order of operations: input * weight → normalize → multiply by
 * out_scale
 *
 */
WIN_EXPORT class RMSReverseNormLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new ReverseRMS normalization layer object
   *
   */
  WIN_EXPORT RMSReverseNormLayer() : Layer() {}

  /**
   * @brief Destroy the ReverseRMS normalization layer object
   *
   */
  WIN_EXPORT ~RMSReverseNormLayer() {}

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
  WIN_EXPORT bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return RMSReverseNormLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = nntrainer::loadProperties(values, rms_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[rms_reverse_norm] Unknown Layer Properties count " +
           std::to_string(values.size());
  };

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "rms_reverse_norm";

private:
  std::array<unsigned int, 2> wt_idx;
  std::tuple<props::RMS_REVERSE_NORM_WEIGHT_INIT,
             props::RMS_REVERSE_NORM_OUTSCALE_INIT, nntrainer::props::Epsilon,
             nntrainer::props::SkipPrefill>
    rms_props;
  bool skip_prefill = false;
};

} // namespace causallm

#endif /* RMS_REVERSE_NORM_LAYER_H__ */
