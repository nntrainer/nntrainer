// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   swiglu_layer.h
 * @date   29 June 2026
 * @brief  Backend-neutral SwiGLU activation: out = silu(gate) * up.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details A single thin Layer that owns structure/shape/orchestration and
 * delegates the kernel to the active backend's ComputeOps whole-op
 * (in1.getOps()->swiglu(...)): OpenCL -> ClComputeOps::swiglu (cl_mem/SVM
 * residency), CPU -> CpuComputeOps::swiglu (host loop). Replaces the former
 * OpenCL SwiGLULayerCl. Two inputs {gate, up}; type "swiglu". Registered on the
 * "gpu" context only — the CPU/CUDA engines use the app-side causallm::SwiGLU
 * layer (a separate registration), so this collapse is OpenCL-scoped. [T7]
 */

#ifndef __SWIGLU_LAYER_H__
#define __SWIGLU_LAYER_H__
#ifdef __cplusplus

#include <tuple>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class SwiGLULayer
 * @brief SwiGLU (silu(gate) * up), backend-neutral via ComputeOps.
 */
class SwiGLULayer final : public Layer {
public:
  /**
   * @brief Construct a new SwiGLULayer object
   */
  SwiGLULayer() : Layer(), swiglu_props(props::Print(), props::SkipPrefill()) {}

  /**
   * @brief Destroy the SwiGLULayer object
   */
  ~SwiGLULayer() {}

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
  const std::string getType() const override { return SwiGLULayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "swiglu";

private:
  // SkipPrefill: gemma-style KV-share layers skip the prefill activation [T12,
  // merged from the former app fork]. props::Print kept for the base layer.
  std::tuple<props::Print, props::SkipPrefill> swiglu_props;
  bool skip_prefill = false;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SWIGLU_LAYER_H__ */
