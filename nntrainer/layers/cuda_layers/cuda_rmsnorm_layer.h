// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cuda_rmsnorm_layer.h
 * @date   22 Jun 2026
 * @brief  RMS normalization for the CUDA backend (FP32-safe sum-of-squares).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Drop-in replacement for the host causallm::RMSNormLayer on the cuda context.
 * Same math (x * rsqrt(mean(x^2)+eps) * gamma) but the sum-of-squares is
 * accumulated in FP32, so an FP16 residual element |x|>~256 does not overflow
 * to +Inf (the host FP16 path squares in FP16: gemma4 layer pre_ffn_norm sees
 * |x|~1688, 1688^2 overflows -> row zeroed -> garbage). UVM is host-coherent,
 * so this runs on the host over the managed buffers; a device kernel is a later
 * perf refinement.
 */

#ifndef __CUDA_RMS_NORM_LAYER_H__
#define __CUDA_RMS_NORM_LAYER_H__

#include <common_properties.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <tuple>

namespace nntrainer {

class CudaRMSNormLayer final : public nntrainer::Layer {
public:
  CudaRMSNormLayer() : Layer(), wt_idx({0}) {}
  ~CudaRMSNormLayer() {}

  void finalize(InitLayerContext &context) override;
  void forwarding(RunLayerContext &context, bool training) override;
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;
  void calcDerivative(RunLayerContext &context) override {}
  bool supportBackwarding() const override { return false; }

  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  const std::string getType() const override { return CudaRMSNormLayer::type; }

  void setProperty(const std::vector<std::string> &values) override {
    // Parse the props we use (epsilon, skip_prefill); silently drop the rest
    // (e.g. gemma4 passes "packed=false", which affects only gamma packing at
    // load time, not the x*rsqrt*gamma forward math).
    loadProperties(values, rms_props);
  }

  /** @brief layer type string. Deliberately the same string the OpenCL
   *  RMSNormLayerCl registers under, so a graph moves between backends by
   *  changing engine= and nothing else. */
  static constexpr const char *type = "rmsnorm";

private:
  std::array<unsigned int, 1> wt_idx;
  std::tuple<nntrainer::props::Epsilon, nntrainer::props::SkipPrefill>
    rms_props;
  bool skip_prefill = false;
};

} // namespace nntrainer

#endif // __CUDA_RMS_NORM_LAYER_H__
