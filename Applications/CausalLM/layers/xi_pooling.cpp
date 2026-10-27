// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   xi_pooling.cpp
 * @date   24 August 2026
 * @brief  Xi-vector Gaussian posterior inference pooling
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "xi_pooling.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

/** Reference clamp bounds on the log-precision, from the upstream
 *  implementation. They keep exp() in range for the softmax below. */
static constexpr float CLAMP_MIN = -15.0f;
static constexpr float CLAMP_MAX = 15.0f;

/** torch.nn.Softplus(beta=1, threshold=20): linear above the threshold. */
static inline float softplus(float x) {
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

XiPoolingLayer::XiPoolingLayer() :
  Layer(), xi_props(props::XiHiddenSize(), nntrainer::props::Epsilon(1e-5f)) {
  wt_idx.fill(std::numeric_limits<unsigned int>::max());
}

void XiPoolingLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, xi_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[xi_pooling] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void XiPoolingLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[xi_pooling] expects exactly one input";

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  dim = in_dim.width();
  hidden = std::get<props::XiHiddenSize>(xi_props).get();

  // The sequence collapses to a single position; the channel width is kept.
  nntrainer::TensorDim out_dim = in_dim;
  out_dim.height(1);
  context.setOutputDimensions({out_dim});

  // Every parameter is stored unquantized as FP32 in the weight file, so
  // request FP32 regardless of the activation dtype -- declaring these FP16
  // would reinterpret the on-disk FP32 bytes and corrupt them.
  const auto ttype = nntrainer::TensorDim::TensorType(
    context.getFormat(), nntrainer::TensorDim::DataType::FP32);
  auto request = [&](unsigned int h, unsigned int w, const char *name) {
    nntrainer::TensorDim d(1, 1, h, w, ttype);
    return context.requestWeight(
      d, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, name, true);
  };

  wt_idx[lin1_w] = request(hidden, dim, "lin1_weight");
  wt_idx[lin1_b] = request(1, hidden, "lin1_bias");
  wt_idx[bn_mean] = request(1, hidden, "bn_moving_mean");
  wt_idx[bn_var] = request(1, hidden, "bn_moving_variance");
  wt_idx[bn_gamma] = request(1, hidden, "bn_gamma");
  wt_idx[bn_beta] = request(1, hidden, "bn_beta");
  wt_idx[lin2_w] = request(dim, hidden, "lin2_weight");
  wt_idx[lin2_b] = request(1, dim, "lin2_bias");
  wt_idx[prior_mean] = request(1, dim, "prior_mean");
  wt_idx[prior_logpr] = request(1, dim, "prior_logprec");
}

void XiPoolingLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  incremental_forwarding(context, 0, in.getDim().height(), training);
}

void XiPoolingLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  (void)training;

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  // Pooling is a reduction over the whole sequence, so a partial range has no
  // meaning here: always consume every row that exists.
  (void)from;
  (void)to;

  const nntrainer::TensorDim in_dim = in.getDim();
  const unsigned int batch = in_dim.batch();
  const unsigned int tokens = in_dim.height();
  const float epsilon = std::get<nntrainer::props::Epsilon>(xi_props).get();

  const float *w1 = context.getWeight(wt_idx[lin1_w]).getData<float>();
  const float *b1 = context.getWeight(wt_idx[lin1_b]).getData<float>();
  const float *mu = context.getWeight(wt_idx[bn_mean]).getData<float>();
  const float *var = context.getWeight(wt_idx[bn_var]).getData<float>();
  const float *gamma = context.getWeight(wt_idx[bn_gamma]).getData<float>();
  const float *beta = context.getWeight(wt_idx[bn_beta]).getData<float>();
  const float *w2 = context.getWeight(wt_idx[lin2_w]).getData<float>();
  const float *b2 = context.getWeight(wt_idx[lin2_b]).getData<float>();
  const float *p_mean = context.getWeight(wt_idx[prior_mean]).getData<float>();
  const float *p_logpr =
    context.getWeight(wt_idx[prior_logpr]).getData<float>();

  // BatchNorm at inference is a per-channel affine, so fold it once per call
  // rather than per frame: r_norm = bn_a * r + bn_b.
  std::vector<float> bn_a(hidden), bn_b(hidden);
  for (unsigned int h = 0; h < hidden; ++h) {
    bn_a[h] = gamma[h] / std::sqrt(var[h] + epsilon);
    bn_b[h] = beta[h] - mu[h] * bn_a[h];
  }

  std::vector<float> r(hidden);
  std::vector<float> logprec(static_cast<size_t>(dim) * tokens);

  for (unsigned int b = 0; b < batch; ++b) {
    const float *x =
      in.getData<float>() + static_cast<size_t>(b) * tokens * dim;
    float *y = out.getData<float>() + static_cast<size_t>(b) * dim;

    // Per-frame log-precision estimate.
    for (unsigned int t = 0; t < tokens; ++t) {
      const float *xt = x + static_cast<size_t>(t) * dim;
      for (unsigned int h = 0; h < hidden; ++h) {
        const float *row = w1 + static_cast<size_t>(h) * dim;
        float acc = b1[h];
        for (unsigned int d = 0; d < dim; ++d)
          acc += row[d] * xt[d];
        acc = acc > 0.0f ? acc : 0.0f; // ReLU
        r[h] = bn_a[h] * acc + bn_b[h];
      }
      for (unsigned int d = 0; d < dim; ++d) {
        const float *row = w2 + static_cast<size_t>(d) * hidden;
        float acc = b2[d];
        for (unsigned int h = 0; h < hidden; ++h)
          acc += row[h] * r[h];
        // 2 log softplus(.), clamped. softplus is positive, so the log is
        // finite except for a hard underflow to 0, which the clamp absorbs.
        const float lp = 2.0f * std::log(softplus(acc));
        logprec[static_cast<size_t>(d) * tokens + t] =
          std::min(CLAMP_MAX, std::max(CLAMP_MIN, lp));
      }
    }

    // Softmax over the tokens plus the one prior column, per channel, then the
    // matching weighted sum. Max-subtracted for numerical stability; the
    // reference exponentiates directly, which is safe only because of the
    // clamp above, and the shift cancels exactly in the ratio.
    for (unsigned int d = 0; d < dim; ++d) {
      const float *lp = &logprec[static_cast<size_t>(d) * tokens];
      float m = p_logpr[d];
      for (unsigned int t = 0; t < tokens; ++t)
        m = std::max(m, lp[t]);

      float denom = std::exp(p_logpr[d] - m);
      float num = p_mean[d] * denom;
      for (unsigned int t = 0; t < tokens; ++t) {
        const float e = std::exp(lp[t] - m);
        denom += e;
        num += x[static_cast<size_t>(t) * dim + d] * e;
      }
      y[d] = num / denom;
    }
  }
}

void XiPoolingLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[xi_pooling] calcDerivative is not supported");
}

#ifdef PLUGGABLE

nntrainer::Layer *create_xi_pooling_layer() {
  auto layer = new XiPoolingLayer();
  return layer;
}

void destroy_xi_pooling_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_xi_pooling_layer,
                                                   destroy_xi_pooling_layer};
}

#endif

} // namespace causallm
