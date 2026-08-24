// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.h
 * @date   11 July 2025
 * @brief  Implementation of RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This layer only supports inference mode.
 */

#ifndef __RMS_NORM_LAYER_H__
#define __RMS_NORM_LAYER_H__

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
 * @brief A custom RMS normalization layer for llama.
 *
 */
WIN_EXPORT class RMSNormLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new custom RMS normalization layer object
   *
   */
  WIN_EXPORT RMSNormLayer() : Layer(), wt_idx({0}) {}

  /**
   * @brief Destroy the custom RMS normalization layer object
   *
   */
  WIN_EXPORT ~RMSNormLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return RMSNormLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, rms_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[rms_norm] Unknown Layer Properties count " +
           std::to_string(values.size());
  };

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "rms_norm";

private:
  /**
   * @brief Lazily allocate a persistent rpcmem copy of gamma and register it
   * with the Hexagon bridge, so nntr_htp_bridge_rms_norm's gamma operand is a
   * zero-copy pool hit instead of the bridge's staging memcpy fallback.
   * gamma is immutable after weight load, so this copies it once and reuses
   * the pointer for the rest of the layer's lifetime.
   *
   * @param gamma the layer's gamma weight tensor (FP32)
   * @return the rpcmem-backed copy, or nullptr if rpcmem/the bridge is
   * unavailable (caller should fall back to gamma's own pointer)
   */
  float *getOrCreateGammaRpcmem(const nntrainer::Tensor &gamma);


  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::RMS_NORM_GAMMA_INIT, nntrainer::props::Epsilon,
             nntrainer::props::SkipPrefill>
    rms_props;
  bool skip_prefill = false;

  // Zero-copy staging for gamma: gamma is a weight tensor, and weights are
  // deliberately kept off the graph's rpcmem-backed activation pool (see
  // docs/backend_guide - CMA budget concern for the large GEMM weight
  // matrices). RMSNorm's gamma is tiny by comparison (one row, width floats)
  // so it gets its own small persistent rpcmem copy instead, populated once
  // on first DSP dispatch. See get_gamma_rpcmem() in rms_norm.cpp.
  float *gamma_rpcmem = nullptr;
  unsigned int gamma_rpcmem_width = 0;
};

} // namespace causallm

#endif /* __CAUSALLM_RMS_NORM_LAYER_H__ */
