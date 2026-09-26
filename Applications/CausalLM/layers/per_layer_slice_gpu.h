// SPDX-License-Identifier: Apache-2.0
/**
 * @file   per_layer_slice_gpu.h
 * @date   17 Jun 2026
 * @brief  GPU variant of the CausalLM PerLayerSlice layer. Same type string
 *         ("per_layer_slice") as the CPU class, registered on cl_context so
 *         engine=gpu routes here. Gathers this layer's per-layer-input slice
 *         on the GPU (per_layer_slice_cl_fp16) so the activation stays
 *         on-device -- a host round-trip here breaks GPU residency (the gemma4
 *         incoherence root cause). FP32 / non-SVM falls back to a host memcpy.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __PER_LAYER_SLICE_GPU_LAYER_H__
#define __PER_LAYER_SLICE_GPU_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <tensor.h>

#include <per_layer_slice.h> // reuse props::FeatureSize / props::LayerIndex

namespace causallm {

WIN_EXPORT class PerLayerSliceLayerGPU final : public nntrainer::Layer {
public:
  WIN_EXPORT PerLayerSliceLayerGPU() : Layer() {}
  WIN_EXPORT ~PerLayerSliceLayerGPU() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override {}

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override {
    throw std::runtime_error("PerLayerSliceLayerGPU: backward not supported");
  }

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  WIN_EXPORT const std::string getType() const override {
    return PerLayerSliceLayerGPU::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, slice_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[per_layer_slice_gpu] Unknown Layer Properties count " +
           std::to_string(values.size());
  }

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override {
    auto out_dim = input_dimensions[0];
    out_dim.width(std::get<props::FeatureSize>(slice_props).get());
    context.updateInput(0, input_dimensions[0]);
    context.updateOutput(0, out_dim);
  }

  inline static const std::string type = "per_layer_slice";

private:
  bool skip_prefill = false;
  std::tuple<props::FeatureSize, props::LayerIndex,
             nntrainer::props::SkipPrefill>
    slice_props;
};

} // namespace causallm

#endif /* __PER_LAYER_SLICE_GPU_LAYER_H__ */
