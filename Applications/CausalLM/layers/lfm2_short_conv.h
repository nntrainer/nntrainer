// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_short_conv.h
 * @date   4 May 2026
 * @brief  LFM2 short convolution layer for CausalLM inference
 */

#ifndef __LFM2_SHORT_CONV_H__
#define __LFM2_SHORT_CONV_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <array>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

/**
 * @brief LFM2 short convolution operator.
 *
 * This layer implements Lfm2ShortConv:
 *   in_proj(x) -> split(B, C, x) -> depthwise causal conv(B * x)
 *   -> C * conv_out -> out_proj.
 */
WIN_EXPORT class Lfm2ShortConvLayer final : public nntrainer::LayerImpl {
public:
  WIN_EXPORT Lfm2ShortConvLayer();
  WIN_EXPORT ~Lfm2ShortConvLayer() = default;

  WIN_EXPORT Lfm2ShortConvLayer(Lfm2ShortConvLayer &&rhs) noexcept = default;
  WIN_EXPORT Lfm2ShortConvLayer &
  operator=(Lfm2ShortConvLayer &&rhs) = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  WIN_EXPORT const std::string getType() const override {
    return Lfm2ShortConvLayer::type;
  }

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                           unsigned int batch) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "lfm2_short_conv";

private:
  std::tuple<nntrainer::props::Unit, nntrainer::props::KernelSize> conv_props;
  std::array<unsigned int, 3> weight_idx;
  std::array<unsigned int, 1> tensor_idx;

  unsigned int hidden_size;
  unsigned int kernel_size;
};

} // namespace causallm

#endif // __LFM2_SHORT_CONV_H__
