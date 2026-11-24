// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   adarms_layer.h
 * @brief  adaRMSNorm for the pi0.5 action expert (openpi's patched
 *         GemmaRMSNorm): takes [x, cond], outputs [normed, gate].
 * @date   12 Nov 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * normed = x * rsqrt(mean(x^2) + eps)
 * dense(cond) -> scale | shift | gate   (Dense(3*width) + bias, zeros-init
 *                                        in openpi; loaded from the bin)
 * normed_out = normed * (1 + scale) + shift
 * The gate is returned as a SECOND output; the residual site consumes it:
 * out = x + branch(normed_out) * gate  (openpi _gated_residual).
 *
 * The layer owns one weight: the dense weight [width, 3*width] in the
 * nntrainer FC layout (in-major, transposed from PyTorch [3w, w]) followed
 * by its bias [3*width]. Converter order must match.
 */
#ifndef __ADARMS_LAYER_H__
#define __ADARMS_LAYER_H__

#pragma once
#ifdef _WIN32
#define ADARMS_EXPORT __declspec(dllexport)
#else
#define ADARMS_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

#include <connection.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

#include <string>
#include <vector>

namespace causallm {

/**
 * @brief Adaptive RMS normalization (pi0.5 expert).
 */
ADARMS_EXPORT class AdaRMSLayer final : public nntrainer::Layer {
public:
  AdaRMSLayer() : Layer(), eps(1e-6f) {}

  static constexpr const char *type = "adarms";

  const std::string getType() const override { return type; }

  bool supportBackwarding() const override { return false; }

  void finalize(nntrainer::InitLayerContext &context) override;

  void setProperty(const std::vector<std::string> &values) override;

  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  void calcDerivative(nntrainer::RunLayerContext &context) override {
    throw nntrainer::exception::not_supported(
      "calcDerivative for adarms layer is not supported");
  }

  void calcGradient(nntrainer::RunLayerContext &context) override {}

  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

private:
  float eps; /**< epsilon of the rms normalization */
  unsigned int dense_wt_idx;
  unsigned int dense_bias_idx;

  /**
   * @brief shared body of forwarding / incremental_forwarding: processes one
   *        batch slice of the inputs into the outputs.
   */
  void run_body(nntrainer::RunLayerContext &context, unsigned int batch,
                unsigned int height, unsigned int width);
};

} // namespace causallm

#endif // __ADARMS_LAYER_H__
