// SPDX-License-Identifier: Apache-2.0
/**
 * @file   rms_norm_gpu.h
 * @date   29 May 2026
 * @brief  GPU variant of the CausalLM custom RMSNorm. Same layer type
 *         string ("rms_norm") as the CPU version, registered on
 *         cl_context so engine=gpu routes here. Uses only GPU compute
 *         + raw host pointers (no Tensor::multiply / Tensor::add_i /
 *         Tensor::inv_sqrt_i) because those CPU Tensor ops crash on
 *         gpu-context-allocated tensors — same family of bug as the
 *         AdditionLayerCL FP32 fast-path workaround.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __RMS_NORM_GPU_LAYER_H__
#define __RMS_NORM_GPU_LAYER_H__

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

/**
 * @brief GPU-friendly RMSNorm. Type string matches the CPU class so
 *        engine=gpu createLayer routes here when this class is
 *        registered on cl_context.
 */
WIN_EXPORT class RMSNormLayerGPU final : public nntrainer::Layer {
public:
  WIN_EXPORT RMSNormLayerGPU() : Layer(), wt_idx({0}) {}
  WIN_EXPORT ~RMSNormLayerGPU() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override {}

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override {
    throw std::runtime_error("RMSNormLayerGPU: backward not supported");
  }

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  WIN_EXPORT const std::string getType() const override {
    return RMSNormLayerGPU::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, rms_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[rms_norm_gpu] Unknown Layer Properties count " +
           std::to_string(values.size());
  }

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  // SAME type string as the CPU class — different contexts (app_context
  // vs cl_context) hold separate factories so this isn't a conflict.
  inline static const std::string type = "rms_norm";

private:
  std::array<unsigned int, 1> wt_idx;
  bool skip_prefill =
    false; /**< skip compute during prefill (Gemma4 KV-share) */
  std::tuple<props::RMS_NORM_GAMMA_INIT, nntrainer::props::Epsilon,
             nntrainer::props::SkipPrefill>
    rms_props;
};

} // namespace causallm

#endif /* __RMS_NORM_GPU_LAYER_H__ */
