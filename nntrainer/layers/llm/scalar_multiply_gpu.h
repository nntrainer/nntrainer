// SPDX-License-Identifier: Apache-2.0
/**
 * @file   scalar_multiply_gpu.h
 * @date   17 Jun 2026
 * @brief  GPU variant of the CausalLM ScalarMultiply layer. Same type string
 *         ("scalar_multiply") as the CPU class, registered on cl_context so
 *         engine=gpu routes here. Runs out = in * scalar as a GPU kernel
 *         (scalar_mul_cl_fp16, SVM/cl_mem resident) so the activation stays
 *         on-device -- a host round-trip here breaks the GPU residency chain
 *         (the gemma4 incoherence root cause). FP32 / non-SVM falls back to a
 *         raw-pointer host loop (no Tensor::multiply, which crashes on
 *         gpu-context-allocated tensors).
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __SCALAR_MULTIPLY_GPU_LAYER_H__
#define __SCALAR_MULTIPLY_GPU_LAYER_H__

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

#include <scalar_multiply.h> // reuse props::ScalarMultiplier / props::UseWeight

namespace nntrainer {

/**
 * @brief GPU variant of ScalarMultiplyLayer: scales its input by a scalar
 *        (or a weight) entirely on the accelerator.
 */
WIN_EXPORT class ScalarMultiplyLayerGPU final : public nntrainer::Layer {
public:
  WIN_EXPORT ScalarMultiplyLayerGPU() : Layer(), wt_idx({0}) {}
  WIN_EXPORT ~ScalarMultiplyLayerGPU() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override {}

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override {
    throw std::runtime_error("ScalarMultiplyLayerGPU: backward not supported");
  }

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  WIN_EXPORT const std::string getType() const override {
    return ScalarMultiplyLayerGPU::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, scalar_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[scalar_multiply_gpu] Unknown Layer Properties count " +
           std::to_string(values.size());
  }

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override {
    context.updateInput(0, input_dimensions[0]);
    context.updateOutput(0, input_dimensions[0]);
  }

  // SAME type string as the CPU class — different contexts hold separate
  // factories (app_context vs cl_context), so this is not a conflict.
  inline static const std::string type = "scalar_multiply";

private:
  std::array<unsigned int, 1> wt_idx;
  bool skip_prefill = false;
  std::tuple<props::ScalarMultiplier, props::UseWeight,
             nntrainer::props::SkipPrefill>
    scalar_props;
};

} // namespace nntrainer

#endif /* __SCALAR_MULTIPLY_GPU_LAYER_H__ */
