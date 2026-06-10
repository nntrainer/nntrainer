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
#include <utility>

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

namespace causallm {

/**
 * @brief A SwiGLU layer for llama.
 *
 */
WIN_EXPORT class SwiGLULayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new custom SwiGLU layer object
   *
   */
  WIN_EXPORT SwiGLULayer() :
    Layer(), swiglu_props(nntrainer::props::SkipPrefill()) {}

  /**
   * @brief Destroy the custom SwiGLU layer object
   *
   */
  WIN_EXPORT ~SwiGLULayer() {}

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
    return SwiGLULayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, swiglu_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[swiglu] Unknown Layer Properties count " +
           std::to_string(values.size());
  };

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "swiglu";

private:
  std::tuple<nntrainer::props::SkipPrefill> swiglu_props;
  bool skip_prefill = false;
};

} // namespace causallm

#endif /* __SWIGLU_LAYER_H__ */
