// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vjepa_gelu_layer.h
 * @date   22 May 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  GELU activation parallelized over tokens (uses the same NEON gelu_v2
 *         as the core activation layer, but splits the work across the
 *         ThreadManager pool — the core activation runs single-threaded).
 */

#ifndef __VJEPA_GELU_LAYER_H__
#define __VJEPA_GELU_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <vector>

namespace causallm {

/** @brief GELU activation parallelized over tokens via the ThreadManager pool.
 */
WIN_EXPORT class VjepaGeluLayer final : public nntrainer::Layer {
public:
  VjepaGeluLayer() = default;
  ~VjepaGeluLayer() = default;

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
    return VjepaGeluLayer::type;
  };

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "vjepa_gelu";
};

} // namespace causallm

#endif /* __VJEPA_GELU_LAYER_H__ */
