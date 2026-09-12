// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   sigmoid_add_layer.h
 * @date   12 September 2026
 * @brief  Backend-neutral sigmoid-gated add: out = sigmoid(gate) + in2.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details The additive member of the gated-activation family in this
 * directory: the gate is squashed exactly as in SigmoidGluLayer, but the
 * second operand is ADDED to the gate's activation instead of being scaled by
 * it. Architectures that mix a side signal into the hidden state -- a
 * per-layer input embedding, for instance -- need this form rather than the
 * multiplicative one. A single thin Layer that owns
 * structure/shape/orchestration and delegates the kernel to the active
 * backend's ComputeOps whole-op (in1.getOps()->sigmoid_add(...)): CPU ->
 * CpuComputeOps::sigmoid_add (host loop, fp32-accumulated), OpenCL ->
 * ClComputeOps::sigmoid_add, CUDA -> CudaComputeOps::sigmoid_add. Fusing the
 * sigmoid and the add into one whole-op keeps the standalone sigmoid op off
 * the accelerators, where it is not registered. Two inputs {gate, in2}; type
 * "sigmoid_add".
 */

#ifndef __SIGMOID_ADD_LAYER_H__
#define __SIGMOID_ADD_LAYER_H__
#ifdef __cplusplus

#include <tuple>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class SigmoidAddLayer
 * @brief sigmoid-gated add (sigmoid(gate) + in2), backend-neutral via
 *        ComputeOps.
 */
class SigmoidAddLayer final : public Layer {
public:
  /**
   * @brief Construct a new SigmoidAddLayer object
   */
  SigmoidAddLayer() :
    Layer(), sigmoid_add_props(props::Print(), props::SkipPrefill()) {}

  /**
   * @brief Destroy the SigmoidAddLayer object
   */
  ~SigmoidAddLayer() {}

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
  const std::string getType() const override { return SigmoidAddLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "sigmoid_add";

private:
  bool skip_prefill =
    false; /**< skip compute during prefill (KV-shared blocks) */
  std::tuple<props::Print, props::SkipPrefill> sigmoid_add_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SIGMOID_ADD_LAYER_H__ */
