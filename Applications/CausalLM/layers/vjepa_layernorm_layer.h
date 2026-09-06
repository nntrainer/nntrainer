// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vjepa_layernorm_layer.h
 * @date   22 May 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  LayerNorm over the last (width) axis, parallelized per-token over the
 *         ThreadManager pool. Numerically matches the core layer_normalization
 *         (axis=3): y = (x-mean)/sqrt(var+eps)*gamma + beta, var biased. The
 *         core layer runs single-threaded; this splits rows across workers.
 */

#ifndef __VJEPA_LAYERNORM_LAYER_H__
#define __VJEPA_LAYERNORM_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <base_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <tuple>
#include <vector>

namespace causallm {

namespace props {
/** @brief epsilon added to variance for numerical stability */
class VjepaLnEpsilon : public nntrainer::Property<float> {
public:
  VjepaLnEpsilon(float value = 1e-6f) { set(value); };
  static constexpr const char *key = "epsilon";
  using prop_tag = nntrainer::float_prop_tag;
};
} // namespace props

/** @brief LayerNorm over the last (width) axis, parallelized per-token over the
 * thread pool. */
WIN_EXPORT class VjepaLayerNormLayer final : public nntrainer::Layer {
public:
  VjepaLayerNormLayer() : layernorm_props(props::VjepaLnEpsilon()) {}
  ~VjepaLayerNormLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT bool supportBackwarding() const override { return false; };

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  WIN_EXPORT const std::string getType() const override {
    return VjepaLayerNormLayer::type;
  };

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "vjepa_layernorm";

private:
  std::tuple<props::VjepaLnEpsilon> layernorm_props;
  std::array<unsigned int, 2> wt_idx; /**< 0: gamma, 1: beta */
};

} // namespace causallm

#endif /* __VJEPA_LAYERNORM_LAYER_H__ */
