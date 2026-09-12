// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   sigmoid_glu_layer.h
 * @date   12 September 2026
 * @brief  Backend-neutral sigmoid-gated linear unit: out = sigmoid(gate) * up.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details The third member of the gated-activation family in this directory,
 * next to GeGLULayer (gelu_tanh(gate) * up) and SwiGLULayer (silu(gate) * up);
 * only the squashing function differs. A single thin Layer that owns
 * structure/shape/orchestration and delegates the kernel to the active
 * backend's ComputeOps whole-op (in1.getOps()->sigmoid_glu(...)): CPU ->
 * CpuComputeOps::sigmoid_glu (host loop, fp32-accumulated), OpenCL ->
 * ClComputeOps::sigmoid_glu, CUDA -> CudaComputeOps::sigmoid_glu. Fusing the
 * sigmoid and the multiply into one whole-op keeps the standalone
 * sigmoid/multiply ops off the accelerators, where they are not registered.
 * Two inputs {gate, up}; type "sigmoid_glu".
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
 * @brief sigmoid-gated linear unit (sigmoid(gate) * up), backend-neutral via
 *        ComputeOps.
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
  bool skip_prefill =
    false; /**< skip compute during prefill (KV-shared blocks) */
  std::tuple<props::Print, props::SkipPrefill> sigmoid_glu_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SIGMOID_GLU_LAYER_H__ */
