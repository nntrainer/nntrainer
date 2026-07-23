// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rtmcc_gau.cpp
 * @date   14 July 2026
 * @brief  RTMCC Gated Attention Unit (GAU) layer implementation.
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#include <cmath>
#include <vector>

#include "rtmcc_gau.h"

namespace quick_ai {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RTMCCGauLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gau_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[rtmcc_gau] Unknown Layer Properties count "
    << std::to_string(values.size());
}

void RTMCCGauLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &in_dim = context.getInputDimensions()[0];

  // Input token tensor is [B, 1, N, D]: N tokens (keypoints), D features.
  num_token = in_dim.height();
  in_dims = in_dim.width();

  e_dims = std::get<props::GauExpansion>(gau_props).get();
  s_dims = std::get<props::GauKeyDim>(gau_props).get();

  // Output has the same shape as input (residual GAU).
  context.setOutputDimensions({in_dim});

  auto weight_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), nntrainer::TensorDim::DataType::FP32);

  // uv projection weight: [D, 2e + s] (matmul weight, K=D rows, N cols).
  nntrainer::TensorDim uv_dim(1, 1, in_dims, 2 * e_dims + s_dims, weight_type);
  wt_idx[uv_w] = context.requestWeight(
    uv_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "uv_weight", true);

  // o projection weight: [e, D].
  nntrainer::TensorDim o_dim(1, 1, e_dims, in_dims, weight_type);
  wt_idx[o_w] = context.requestWeight(
    o_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "o_weight", true);

  // gamma, beta: [2, s].
  nntrainer::TensorDim affine_dim(1, 1, 2, s_dims, weight_type);
  wt_idx[gamma] = context.requestWeight(
    affine_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
  wt_idx[beta] = context.requestWeight(
    affine_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "beta", true);

  // ScaleNorm scalar gain g: [1].
  nntrainer::TensorDim g_dim(1, 1, 1, 1, weight_type);
  wt_idx[ln_g] = context.requestWeight(
    g_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "ln_g", true);

  // residual per-channel scale: [D].
  nntrainer::TensorDim rs_dim(1, 1, 1, in_dims, weight_type);
  wt_idx[res_scale] = context.requestWeight(
    rs_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "res_scale", true);
}

void RTMCCGauLayer::forwarding(nntrainer::RunLayerContext &context,
                               bool training) {
  incremental_forwarding(context, 0, num_token, training);
}

void RTMCCGauLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  using nntrainer::Tensor;
  using DataType = ml::train::TensorDim::DataType;

  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &w_uv = context.getWeight(wt_idx[uv_w]);
  Tensor &w_o = context.getWeight(wt_idx[o_w]);
  Tensor &t_gamma = context.getWeight(wt_idx[gamma]);
  Tensor &t_beta = context.getWeight(wt_idx[beta]);
  Tensor &t_g = context.getWeight(wt_idx[ln_g]);
  Tensor &t_rs = context.getWeight(wt_idx[res_scale]);

  NNTR_THROW_IF(in.getDataType() != DataType::FP32, std::invalid_argument)
    << "[rtmcc_gau] only FP32 activations are supported";

  const unsigned int N = num_token;
  const unsigned int D = in_dims;
  const unsigned int E = e_dims;
  const unsigned int S = s_dims;
  const float eps = std::get<nntrainer::props::Epsilon>(gau_props).get();
  const float scalenorm_scale = 1.0f / std::sqrt(static_cast<float>(D));
  const float sqrt_s = std::sqrt(static_cast<float>(S));

  auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };

  const float g_val = t_g.getData<float>()[0];
  const float *gamma_p = t_gamma.getData<float>(); // [2, S]
  const float *beta_p = t_beta.getData<float>();   // [2, S]
  const float *rs_p = t_rs.getData<float>();       // [D]

  const nntrainer::TensorDim in_dim = in.getDim();
  const unsigned int batch = in_dim.batch();

  nntrainer::TensorDim step_dim(1, 1, N, D, in_dim.getTensorType());

  for (unsigned int b = 0; b < batch; ++b) {
    Tensor x = in.getSharedDataTensor(step_dim, b * in_dim.getFeatureLen(),
                                      true);
    Tensor y = out.getSharedDataTensor(step_dim, b * in_dim.getFeatureLen(),
                                       true);
    const float *xp = x.getData<float>();

    // 1. ScaleNorm: per-token RMS over feature dim, scaled by scalar g.
    Tensor x_ln(step_dim, true);
    float *lnp = x_ln.getData<float>();
    for (unsigned int n = 0; n < N; ++n) {
      const float *row = xp + n * D;
      float ss = 0.0f;
      for (unsigned int d = 0; d < D; ++d)
        ss += row[d] * row[d];
      float norm = std::sqrt(ss) * scalenorm_scale;
      if (norm < eps)
        norm = eps;
      float inv = g_val / norm;
      float *orow = lnp + n * D;
      for (unsigned int d = 0; d < D; ++d)
        orow[d] = row[d] * inv;
    }

    // 2. uv = SiLU(x_ln @ Wuv)  -> [N, 2E + S]
    Tensor uv = x_ln.dot(w_uv); // [1,1,N,2E+S]
    float *uvp = uv.getData<float>();
    const unsigned int UV = 2 * E + S;
    for (unsigned int i = 0; i < N * UV; ++i)
      uvp[i] = silu(uvp[i]);

    // 3. split into u [N,E], v [N,E], base [N,S]; build q,k [N,S].
    nntrainer::TensorDim eDim(1, 1, N, E, in_dim.getTensorType());
    nntrainer::TensorDim sDim(1, 1, N, S, in_dim.getTensorType());
    Tensor u(eDim, true), v(eDim, true), q(sDim, true), k(sDim, true);
    float *up = u.getData<float>();
    float *vp = v.getData<float>();
    float *qp = q.getData<float>();
    float *kp = k.getData<float>();
    for (unsigned int n = 0; n < N; ++n) {
      const float *urow = uvp + n * UV;
      std::copy(urow, urow + E, up + n * E);
      std::copy(urow + E, urow + 2 * E, vp + n * E);
      const float *base = urow + 2 * E; // [S]
      float *qrow = qp + n * S;
      float *krow = kp + n * S;
      for (unsigned int s = 0; s < S; ++s) {
        qrow[s] = base[s] * gamma_p[s] + beta_p[s];             // row 0
        krow[s] = base[s] * gamma_p[S + s] + beta_p[S + s];     // row 1
      }
    }

    // 4. qk = q @ k^T -> [N,N]; kernel = relu(qk / sqrt_s)^2.
    Tensor qk = q.dot(k, false, true); // [1,1,N,N]
    float *qkp = qk.getData<float>();
    for (unsigned int i = 0; i < N * N; ++i) {
      float r = qkp[i] / sqrt_s;
      r = r > 0.0f ? r : 0.0f;
      qkp[i] = r * r;
    }

    // 5. attn = kernel @ v -> [N,E]; gated = u * attn.
    Tensor attn = qk.dot(v); // [1,1,N,E]
    float *ap = attn.getData<float>();
    for (unsigned int i = 0; i < N * E; ++i)
      ap[i] *= up[i];

    // 6. out_gau = gated @ Wo -> [N,D].
    Tensor out_gau = attn.dot(w_o); // [1,1,N,D]
    const float *ogp = out_gau.getData<float>();

    // 7. y = res_scale * x + out_gau.
    float *yp = y.getData<float>();
    for (unsigned int n = 0; n < N; ++n) {
      const float *xr = xp + n * D;
      const float *or_ = ogp + n * D;
      float *yr = yp + n * D;
      for (unsigned int d = 0; d < D; ++d)
        yr[d] = rs_p[d] * xr[d] + or_[d];
    }
  }
}

void RTMCCGauLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[rtmcc_gau] training is not supported");
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rtmcc_gau_layer() { return new RTMCCGauLayer(); }

void destroy_rtmcc_gau_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rtmcc_gau_layer,
                                                   destroy_rtmcc_gau_layer};
}

#endif

} // namespace quick_ai
