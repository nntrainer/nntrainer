// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Anirudh Rajapakshe <anirudh.rajapakshe@samsung.com>
 *
 * @file   fused_ffn_layer.h
 * @date   4 August 2026
 * @brief  Fused FFN Layer with SwiGLU activation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Anirudh Rajapakshe <anirudh.rajapakshe@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __FUSED_FFN_LAYER_H__
#define __FUSED_FFN_LAYER_H__

#pragma once
#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include <common_properties.h>
#include <layer_impl.h>

#include <array>

namespace causallm {

namespace props {
class HiddenDim : public nntrainer::Property<int> {
public:
  static constexpr const char *key = "hidden_dim";
  using prop_tag = nntrainer::int_prop_tag;
  HiddenDim(int value = 0) { set(value); }
};

class OutputDim : public nntrainer::Property<int> {
public:
  static constexpr const char *key = "output_dim";
  using prop_tag = nntrainer::int_prop_tag;
  OutputDim(int value = 0) { set(value); }
};
} // namespace props

class WIN_EXPORT FusedFFNLayer : public nntrainer::LayerImpl {
public:
  FusedFFNLayer();
  ~FusedFFNLayer() = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;
  void setProperty(const std::vector<std::string> &values) override;

  const std::string getType() const override {
    return FusedFFNLayer::type;
  };
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "fused_ffn";

private:
  std::tuple<nntrainer::props::Unit, nntrainer::props::DisableBias,
             nntrainer::props::WeightInitializer, props::HiddenDim,
             props::OutputDim>
    ffn_props;
  std::array<unsigned int, 3> weight_idx;
};

} // namespace causallm

#endif /* __FUSED_FFN_LAYER_H__ */
