// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   sigmoid_add_layer.h
 * @date   06 July 2026
 * @brief  Backend-neutral Sigmoid-add: out = sigmoid(gate) + emb.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details A single thin Layer that owns structure/shape/orchestration and
 * delegates the kernel to the active backend's ComputeOps whole-op
 * (in1.getOps()->sigmoid_add(...)): CPU -> CpuComputeOps::sigmoid_add (host
 * loop, fp32-accumulated), OpenCL -> ClComputeOps::sigmoid_add, CUDA ->
 * CudaComputeOps::sigmoid_add. Fusing sigmoid+add into one whole-op keeps the
 * standalone sigmoid op off the GPU (unregistered there). Two inputs
 * {gate, emb}; type "sigmoid_add". Mirrors SwiGLULayer. Registered on all
 * three core contexts (cpu/gpu/cuda). Serves models with a sigmoid-gated
 * per-layer-embedding (PLE) mix.
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
 * @brief Sigmoid-add (sigmoid(gate) + emb), backend-neutral via ComputeOps.
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
  // SkipPrefill: KV-share PLE layers skip the prefill mix.
  // props::Print kept for the base layer.
  std::tuple<props::Print, props::SkipPrefill> sigmoid_add_props;
  bool skip_prefill = false;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SIGMOID_ADD_LAYER_H__ */
