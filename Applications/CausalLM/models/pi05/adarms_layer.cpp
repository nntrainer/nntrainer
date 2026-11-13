// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   adarms_layer.cpp
 * @brief  Implementation of the pi0.5 expert adaRMSNorm layer.
 * @date   12 Nov 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "adarms_layer.h"

#include <nntrainer_error.h>
#include <util_func.h>

#include <stdexcept>

namespace causallm {

namespace {
enum AdaRMSParams { dense_weight, dense_bias };
} // namespace

void AdaRMSLayer::setProperty(const std::vector<std::string> &values) {
  // This layer carries one scalar property ("epsilon=1e-06"); parse it by
  // hand — loadProperties needs a property tuple this layer does not have.
  for (const auto &v : values) {
    const std::string prefix = "epsilon=";
    if (v.rfind(prefix, 0) == 0)
      eps = std::stof(v.substr(prefix.size()));
    else
      throw std::invalid_argument("adarms: unknown property " + v);
  }
}

void AdaRMSLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "adarms takes [x, cond] (2 inputs)";

  auto const &x_dim = context.getInputDimensions()[0];
  auto const &c_dim = context.getInputDimensions()[1];

  NNTR_THROW_IF(x_dim.channel() != 1 || c_dim.channel() != 1,
                std::invalid_argument)
    << "adarms expects NCHW tensors with channel 1";
  // x: [B, 1, T, W]; cond: [B, 1, 1, W] (broadcast over the sequence)
  NNTR_THROW_IF(x_dim.width() != c_dim.width(), std::invalid_argument)
    << "adarms cond width must equal x width";

  const unsigned int w = x_dim.width();
  context.setOutputDimensions({x_dim, c_dim}); // [normed, gate]

  const bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  ml::train::TensorDim w_dim(
    1, is_nchw ? 1 : w, is_nchw ? w : 1, is_nchw ? 3 * w : 3 * w,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  ml::train::TensorDim b_dim(
    1, is_nchw ? 1 : 1, 1, is_nchw ? 3 * w : 3 * w,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  dense_wt_idx = context.requestWeight(w_dim, nntrainer::Initializer::NONE,
                                       nntrainer::WeightRegularizer::NONE, 1.0f,
                                       0.0f, "dense_weight", true);
  dense_bias_idx = context.requestWeight(b_dim, nntrainer::Initializer::NONE,
                                         nntrainer::WeightRegularizer::NONE,
                                         1.0f, 0.0f, "dense_bias", true);
}

void AdaRMSLayer::exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  // No LayerImpl props to export; the dense weights are plain requestWeight
  // records discovered by index.
  (void)exporter;
  (void)method;
}

void AdaRMSLayer::run_body(nntrainer::RunLayerContext &context,
                           unsigned int batch, unsigned int height,
                           unsigned int width) {
  nntrainer::Tensor &x = context.getInput(0);
  nntrainer::Tensor &cond = context.getInput(1);
  nntrainer::Tensor &normed_out = context.getOutput(0);
  nntrainer::Tensor &gate_out = context.getOutput(1);
  nntrainer::Tensor &dw = context.getWeight(dense_wt_idx);
  nntrainer::Tensor &db = context.getWeight(dense_bias_idx);

  const float *xp = x.getData<float>();
  const float *cp = cond.getData<float>();
  const float *dwp = dw.getData<float>();
  const float *dbp = db.getData<float>();
  float *np_ = normed_out.getData<float>();
  float *gp = gate_out.getData<float>();

  // gate = dense(cond)[:, width : 2*width] ... layout: dense weight is
  // [width, 3*width] (in-major). dense(cond) = cond @ dw + db gives 3*width
  // outputs: [0,w)=scale, [w,2w)=shift, [2w,3w)=gate.
  std::vector<float> mod(3 * width);
  for (unsigned int o = 0; o < 3 * width; ++o) {
    float acc = dbp[o];
    for (unsigned int i = 0; i < width; ++i)
      acc += cp[i] * dwp[i * (3 * (size_t)width) + o];
    mod[o] = acc;
  }
  // copy scale/shift/gate into the gate output plane (gate rows), keep
  // scale/shift local
  const float *scale = mod.data();
  const float *shift = mod.data() + width;
  const float *gate = mod.data() + 2 * width;

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int t = 0; t < height; ++t) {
      const float *xrow = xp + ((size_t)b * height + t) * width;
      float *nrow = np_ + ((size_t)b * height + t) * width;
      float *grow = gp + ((size_t)b * height + t) * width;
      float ssq = 0.0f;
      for (unsigned int i = 0; i < width; ++i)
        ssq += xrow[i] * xrow[i];
      const float inv = 1.0f / std::sqrt(ssq / width + eps);
      for (unsigned int i = 0; i < width; ++i)
        nrow[i] = xrow[i] * inv * (1.0f + scale[i]) + shift[i];
      for (unsigned int i = 0; i < width; ++i)
        grow[i] = gate[i];
    }
  }
}

void AdaRMSLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  auto const &x_dim = context.getInput(0).getDim();
  run_body(context, x_dim.batch(), x_dim.height(), x_dim.width());
}

void AdaRMSLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  auto const &x_dim = context.getInput(0).getDim();
  run_body(context, x_dim.batch(), to - from, x_dim.width());
}

} // namespace causallm
