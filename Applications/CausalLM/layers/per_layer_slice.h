// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   per_layer_slice.h
 * @date   07 Apr 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Selects per-layer input chunk from packed per-layer embedding tensor.
 */

#ifndef __PER_LAYER_SLICE_H__
#define __PER_LAYER_SLICE_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <base_properties.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace causallm {

namespace props {
/**
 * @brief Layer index used to select a per-layer tensor slice.
 */
class LayerIndex : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key = "layer_index";
  using prop_tag = nntrainer::uint_prop_tag;
  LayerIndex(unsigned int value = 0) { set(value); }
};
} // namespace props

/**
 * @brief Layer that selects one layer-specific chunk from packed input.
 */
WIN_EXPORT class PerLayerSliceLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT PerLayerSliceLayer() :
    Layer(),
    slice_props(props::FeatureSize(), props::LayerIndex(),
                nntrainer::props::SkipPrefill()) {}

  WIN_EXPORT ~PerLayerSliceLayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  WIN_EXPORT const std::string getType() const override { return type; }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, slice_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[per_layer_slice] Unknown Layer Properties count " +
           std::to_string(values.size());
  }

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "per_layer_slice";

private:
  std::tuple<props::FeatureSize, props::LayerIndex,
             nntrainer::props::SkipPrefill>
    slice_props;
  bool skip_prefill = false;
};

} // namespace causallm

#endif
