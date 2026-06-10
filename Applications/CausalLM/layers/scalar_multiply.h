// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   scalar_multiply.h
 * @date   7 April 2026
 * @brief  Implementation of scalar multiplication layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This layer multiplies input tensor by a scalar value provided as
 * property or weight.
 */

#ifndef __SCALAR_MULTIPLY_LAYER_H__
#define __SCALAR_MULTIPLY_LAYER_H__

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
#include <common_properties.h>
#include <connection.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace causallm {

namespace props {

/**
 * @brief ScalarMultiplier property for the scalar value to multiply
 */
class ScalarMultiplier : public nntrainer::Property<float> {
public:
  static constexpr const char *key = "multiplier"; /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag;      /**< property type */
  ScalarMultiplier(float value = 1.0f) { set(value); }
};

/**
 * @brief UseWeight property to determine whether to load scalar from weight
 * file
 */
class UseWeight : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "use_weight"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;       /**< property type */
  UseWeight(bool value = false) { set(value); }
};

} // namespace props

/**
 * @brief A scalar multiplication layer that multiplies input tensor by a
 * scalar. The scalar can be provided either as a property or loaded from a
 * weight file.
 *
 */
WIN_EXPORT class ScalarMultiplyLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new scalar multiply layer object
   *
   */
  WIN_EXPORT ScalarMultiplyLayer() : Layer(), wt_idx({0}) {}

  /**
   * @brief Destroy the scalar multiply layer object
   *
   */
  WIN_EXPORT ~ScalarMultiplyLayer() {}

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
           const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return ScalarMultiplyLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, scalar_multiply_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[scalar_multiply] Unknown Layer Properties count " +
           std::to_string(values.size());
  };

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "scalar_multiply";

private:
  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::ScalarMultiplier, props::UseWeight,
             nntrainer::props::SkipPrefill>
    scalar_multiply_props;
  bool skip_prefill = false;
};

} // namespace causallm

#endif /* __SCALAR_MULTIPLY_LAYER_H__ */
