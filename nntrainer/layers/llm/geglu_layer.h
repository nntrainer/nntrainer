// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   geglu_layer.h
 * @date   29 June 2026
 * @brief  Backend-neutral GeGLU activation: out = gelu_tanh(gate) * up.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details A single thin Layer that owns structure/shape/orchestration and
 * delegates the kernel to the active backend's ComputeOps whole-op
 * (in1.getOps()->geglu(...)): OpenCL -> ClComputeOps::geglu (cl_mem/SVM
 * residency), CUDA -> CudaComputeOps::geglu (device fp16 / host-on-UVM), CPU ->
 * CpuComputeOps::geglu (host loop). Replaces the former GeGLULayerCl and
 * CudaGeGLULayer forks. Two inputs {gate, up}; type "geglu".
 */

#ifndef __GEGLU_LAYER_H__
#define __GEGLU_LAYER_H__
#ifdef __cplusplus

#include <tuple>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class GeGLULayer
 * @brief GeGLU (gelu_tanh(gate) * up), backend-neutral via ComputeOps.
 */
class GeGLULayer final : public Layer {
public:
  /**
   * @brief Construct a new GeGLULayer object
   */
  GeGLULayer() : Layer(), geglu_props(props::Print(), props::SkipPrefill()) {}

  /**
   * @brief Destroy the GeGLULayer object
   */
  ~GeGLULayer() {}

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
  const std::string getType() const override { return GeGLULayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "geglu";

private:
  bool skip_prefill =
    false; /**< skip compute during prefill (Gemma4 KV-share) */
  std::tuple<props::Print, props::SkipPrefill> geglu_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __GEGLU_LAYER_H__ */
