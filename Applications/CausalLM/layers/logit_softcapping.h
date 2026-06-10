// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   logit_softcapping.h
 * @date   8 April 2026
 * @brief  Implementation of final logit softcapping layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __LOGIT_SOFTCAPPING_LAYER_H__
#define __LOGIT_SOFTCAPPING_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <acti_func.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <tensor.h>

namespace causallm {

namespace props {

/**
 * @brief Activation type for softcapping
 */
class LogitSoftcapActivation final
  : public nntrainer::EnumProperty<nntrainer::props::ActivationTypeInfo> {
public:
  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "activation_type";
};

/**
 * @brief Number of rows from the front to apply softcapping
 */
class ApplyRows : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key = "apply_rows";
  using prop_tag = nntrainer::uint_prop_tag;
  ApplyRows(unsigned int value = 0) { set(value); }
};

/**
 * @brief Softcap value used in x / softcap -> activation -> * softcap
 */
class SoftcapValue : public nntrainer::Property<float> {
public:
  static constexpr const char *key = "softcap_value";
  using prop_tag = nntrainer::float_prop_tag;
  SoftcapValue(float value = 1.0f) { set(value); }
};

} // namespace props

/**
 * @brief Final logit softcapping layer
 */
WIN_EXPORT class LogitSoftCappingLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT LogitSoftCappingLayer() : Layer(), acti_func() {}

  WIN_EXPORT ~LogitSoftCappingLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT bool supportBackwarding() const override { return false; };

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  WIN_EXPORT const std::string getType() const override {
    return LogitSoftCappingLayer::type;
  };

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, logit_softcap_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[logit_softcapping] Unknown Layer Properties count " +
           std::to_string(remain_props.size());
  };

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "logit_softcapping";

private:
  using PropTypes =
    std::tuple<props::LogitSoftcapActivation, props::ApplyRows,
               props::SoftcapValue, nntrainer::props::SkipPrefill>;

  void applyOnRange(nntrainer::RunLayerContext &context, unsigned int from,
                    unsigned int to);

  PropTypes logit_softcap_props;
  nntrainer::ActiFunc acti_func;
  bool skip_prefill = false;
};

} // namespace causallm

#endif /* __LOGIT_SOFTCAPPING_LAYER_H__ */
