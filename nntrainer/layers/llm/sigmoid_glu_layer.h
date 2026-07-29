// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   sigmoid_glu_layer.h
 * @date   06 July 2026
 * @brief  Backend-neutral Sigmoid-GLU: out = sigmoid(gate) * up.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details A single thin Layer that owns structure/shape/orchestration and
 * delegates the kernel to the active backend's ComputeOps whole-op
 * (in1.getOps()->sigmoid_glu(...)): CPU -> CpuComputeOps::sigmoid_glu (host
 * loop, fp32-accumulated), OpenCL -> ClComputeOps::sigmoid_glu, CUDA ->
 * CudaComputeOps::sigmoid_glu. Fusing sigmoid+multiply into one whole-op keeps
 * the standalone sigmoid/multiply ops off the GPU (they are unregistered
 * there). Two inputs {gate, up}; type "sigmoid_glu". Mirrors SwiGLULayer.
 * Registered on all three core contexts (cpu/gpu/cuda). Serves models with a
 * sigmoid-gated attention output.
 */

#ifndef __SIGMOID_GLU_LAYER_H__
#define __SIGMOID_GLU_LAYER_H__
#ifdef __cplusplus

#include <tuple>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class SigmoidGluLayer
 * @brief Sigmoid-GLU (sigmoid(gate) * up), backend-neutral via ComputeOps.
 */
class SigmoidGluLayer final : public Layer {
public:
  /**
   * @brief Construct a new SigmoidGluLayer object
   */
  SigmoidGluLayer() :
    Layer(), sigmoid_glu_props(props::Print(), props::SkipPrefill()) {}

  /**
   * @brief Destroy the SigmoidGluLayer object
   */
  ~SigmoidGluLayer() {}

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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return SigmoidGluLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "sigmoid_glu";

private:
  // SkipPrefill: KV-share attention layers skip the prefill gate.
  // props::Print kept for the base layer.
  std::tuple<props::Print, props::SkipPrefill> sigmoid_glu_props;
  bool skip_prefill = false;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SIGMOID_GLU_LAYER_H__ */
