// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rms_norm_layer.h
 * @date   29 Jul 2026
 * @brief  Backend-neutral RMS normalization:
 *         out = in * rsqrt(mean(in^2) + eps) * gamma.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Inference only (calcDerivative throws).
 *
 * @details One thin Layer that owns shape/step orchestration and delegates the
 * whole kernel to the active backend's ComputeOps
 * (in.getOps()->rms_norm(...)): CPU -> CpuComputeOps::rms_norm (the
 * arch-dispatched width-wise intrinsics), CUDA -> CudaComputeOps::rms_norm
 * (fp16 device kernel for decode row counts, else its host fallback). Every
 * impl accumulates the sum of squares in FP32 — an fp16 activation row with
 * large elements squares past the fp16 max otherwise, zeroing the row.
 * Type "rms_norm"; replaces the per-backend layer fork the cuda context
 * used to bind under the same type string.
 */

#ifndef __NNTRAINER_RMS_NORM_LAYER_H__
#define __NNTRAINER_RMS_NORM_LAYER_H__
#ifdef __cplusplus

#include <array>
#include <tuple>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace nntrainer {

/**
 * @class RMSNormLayer
 * @brief RMS normalization (x * rsqrt(mean(x^2)+eps) * gamma), backend-neutral
 *        via the ComputeOps whole-op.
 */
class RMSNormLayer final : public Layer {
public:
  /**
   * @brief Construct a new RMSNormLayer object
   */
  RMSNormLayer() : Layer(), wt_idx({0}) {}

  /**
   * @brief Destroy the RMSNormLayer object
   */
  ~RMSNormLayer() {}

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
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return RMSNormLayer::type; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override {
    // Parse the props we use (epsilon, skip_prefill); silently drop the rest
    // (e.g. a "packed=false" hint affects only gamma packing at load time, not
    // the x*rsqrt*gamma forward math).
    loadProperties(values, rms_props);
  }

  /**
   * @copydoc Layer::updateTensorsByInputDimensions(RunLayerContext &context,
   * std::vector<TensorDim> input_dimensions)
   */
  void updateTensorsByInputDimensions(
    RunLayerContext &context, std::vector<TensorDim> input_dimensions) override;

  inline static const std::string type = "rms_norm";

private:
  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::Epsilon, props::SkipPrefill> rms_props;
  bool skip_prefill = false;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NNTRAINER_RMS_NORM_LAYER_H__ */
