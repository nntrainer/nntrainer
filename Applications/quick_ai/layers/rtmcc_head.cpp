// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rtmcc_head.cpp
 * @date   15 July 2026
 * @brief  RTMCC SimCC pose head layer implementation (NCHW/NHWC-agnostic).
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <act_simd.h> // nntr_silu_inplace_f32 (shared NEON expf-based SiLU)

#include "rtmcc_head.h"

namespace quick_ai {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RTMCCHeadLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, head_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[rtmcc_head] Unknown Layer Properties count "
    << std::to_string(values.size());
}

void RTMCCHeadLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &in_dim = context.getInputDimensions()[0];

  // Conv output [B, K, H, W]: K keypoints (tokens), H*W spatial (feature).
  num_token = in_dim.channel();
  flatten = in_dim.height() * in_dim.width();

  e_dims = std::get<props::RtmccExpansion>(head_props).get();
  s_dims = std::get<props::RtmccKeyDim>(head_props).get();
  hidden = std::get<props::RtmccHidden>(head_props).get();
  simcc = std::get<props::RtmccSimcc>(head_props).get();
  is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);

  // Output: [B, 1, 2*K, SIMCC] (cls_x rows over cls_y rows). C=1 makes the
  // storage identical in NCHW and NHWC, so main.cpp decodes [2*K, SIMCC].
  // Always FP32 so the decoder reads float* regardless of the activation dtype
  // (the FP16-activation W8A16 path still yields an FP32 pose output).
  auto out_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), nntrainer::TensorDim::DataType::FP32);
  nntrainer::TensorDim out_dim(in_dim.batch(), 1, 2 * num_token, simcc,
                               out_type);
  context.setOutputDimensions({out_dim});

  // Head weights + all internal token tensors are kept NCHW-typed regardless
  // of the model format: the input gather below is format-aware, but Tensor::dot
  // extracts its M/N/K from the (logical) last two dims assuming NCHW, so a
  // NHWC-typed operand would mis-shape the GEMM.
  auto wtype = nntrainer::TensorDim::TensorType(
    nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP32);
  auto req = [&](const nntrainer::TensorDim &d, const std::string &name) {
    return context.requestWeight(d, nntrainer::props::InitializerInfo::Enum::NONE,
                                 nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                                 name, true);
  };

  wt_idx[mlp_ln_g] = req({1, 1, 1, 1, wtype}, "mlp_ln_g");
  wt_idx[mlp_w] = req({1, 1, flatten, hidden, wtype}, "mlp_w");
  wt_idx[gau_uv] = req({1, 1, hidden, 2 * e_dims + s_dims, wtype}, "gau_uv");
  wt_idx[gau_o] = req({1, 1, e_dims, hidden, wtype}, "gau_o");
  wt_idx[gau_gamma] = req({1, 1, 2, s_dims, wtype}, "gau_gamma");
  wt_idx[gau_beta] = req({1, 1, 2, s_dims, wtype}, "gau_beta");
  wt_idx[gau_ln_g] = req({1, 1, 1, 1, wtype}, "gau_ln_g");
  wt_idx[gau_res_scale] = req({1, 1, 1, hidden, wtype}, "gau_res_scale");
  wt_idx[cls_x] = req({1, 1, hidden, simcc, wtype}, "cls_x");
  wt_idx[cls_y] = req({1, 1, hidden, simcc, wtype}, "cls_y");
}

void RTMCCHeadLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  incremental_forwarding(context, 0, num_token, training);
}

void RTMCCHeadLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int, unsigned int, bool) {
  using nntrainer::Tensor;
  using DataType = ml::train::TensorDim::DataType;

  Tensor &in_raw = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  // W8A8: the feature map may arrive as a per-tensor-scale QINT8 activation
  // (the producing conv is int8-resident). The head is an FP32 island, so
  // dequantize once up front and run the FP32 path unchanged.
  Tensor in = in_raw;
  std::vector<float> w8a8_deq;
  if (in_raw.getDataType() == DataType::QINT8) {
    const int8_t *q = in_raw.getData<int8_t>();
    const float s = in_raw.getScale<float>()[0];
    w8a8_deq.resize(in_raw.size());
    for (size_t i = 0; i < w8a8_deq.size(); ++i)
      w8a8_deq[i] = s * (float)q[i];
    nntrainer::TensorDim d = in_raw.getDim();
    d.setDataType(DataType::FP32);
    in = nntrainer::Tensor::Map<float>(w8a8_deq.data(),
                                       w8a8_deq.size() * sizeof(float), d);
  }

  const bool in_fp16 = (in.getDataType() == DataType::FP16);
#ifndef ENABLE_FP16
  NNTR_THROW_IF(in_fp16, std::invalid_argument)
    << "[rtmcc_head] FP16 activations require an ENABLE_FP16 build";
#endif
  NNTR_THROW_IF(!in_fp16 && in.getDataType() != DataType::FP32,
                std::invalid_argument)
    << "[rtmcc_head] only FP32/FP16 activations are supported";

  const unsigned int K = num_token;
  const unsigned int F = flatten;
  const unsigned int D = hidden;
  const unsigned int E = e_dims;
  const unsigned int S = s_dims;
  const unsigned int UV = 2 * E + S;
  const float eps = std::get<nntrainer::props::Epsilon>(head_props).get();
  const float mlp_scale = 1.0f / std::sqrt(static_cast<float>(F));
  const float gau_scale = 1.0f / std::sqrt(static_cast<float>(D));
  const float sqrt_s = std::sqrt(static_cast<float>(S));


  Tensor &w_mlp_g = context.getWeight(wt_idx[mlp_ln_g]);
  Tensor &w_mlp = context.getWeight(wt_idx[mlp_w]);
  Tensor &w_uv = context.getWeight(wt_idx[gau_uv]);
  Tensor &w_o = context.getWeight(wt_idx[gau_o]);
  Tensor &t_gamma = context.getWeight(wt_idx[gau_gamma]);
  Tensor &t_beta = context.getWeight(wt_idx[gau_beta]);
  Tensor &w_gau_g = context.getWeight(wt_idx[gau_ln_g]);
  Tensor &t_rs = context.getWeight(wt_idx[gau_res_scale]);
  Tensor &w_cx = context.getWeight(wt_idx[cls_x]);
  Tensor &w_cy = context.getWeight(wt_idx[cls_y]);

  const float mlp_g = w_mlp_g.getData<float>()[0];
  const float gau_g = w_gau_g.getData<float>()[0];
  const float *gamma_p = t_gamma.getData<float>();
  const float *beta_p = t_beta.getData<float>();
  const float *rs_p = t_rs.getData<float>();

  const nntrainer::TensorDim in_dim = in.getDim();
  const unsigned int batch = in_dim.batch();
  const float *in_all = in_fp16 ? nullptr : in.getData<float>();
#ifdef ENABLE_FP16
  const _FP16 *in_all16 = in_fp16 ? in.getData<_FP16>() : nullptr;
#endif
  float *out_all = out.getData<float>();
  const size_t in_feat = static_cast<size_t>(K) * F;
  const size_t out_feat = static_cast<size_t>(2 * K) * simcc;

  // read conv-output element (keypoint k, spatial s) as float, format-aware.
  auto read_in = [&](size_t base, unsigned int k, unsigned int s) -> float {
    size_t idx = is_nchw ? (static_cast<size_t>(k) * F + s)
                         : (static_cast<size_t>(s) * K + k);
#ifdef ENABLE_FP16
    if (in_fp16)
      return static_cast<float>(in_all16[base + idx]);
#endif
    return in_all[base + idx];
  };

  // Internal token tensors are NCHW so Tensor::dot shapes the GEMM correctly.
  const auto nchw = nntrainer::TensorDim::TensorType(
    nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP32);
  nntrainer::TensorDim kf(1, 1, K, F, nchw);
  nntrainer::TensorDim kd(1, 1, K, D, nchw);

  for (unsigned int b = 0; b < batch; ++b) {
    const size_t base = static_cast<size_t>(b) * in_feat;
    float *outb = out_all + b * out_feat;

    // 1. Gather tokens X[K, F] from the conv output (format-aware) and apply
    //    ScaleNorm (RMS over F, scalar gain mlp_g).
    Tensor X(kf, true);
    float *xp = X.getData<float>();
    for (unsigned int k = 0; k < K; ++k) {
      float ss = 0.0f;
      for (unsigned int s = 0; s < F; ++s) {
        float v = read_in(base, k, s);
        xp[k * F + s] = v;
        ss += v * v;
      }
      float norm = std::sqrt(ss) * mlp_scale;
      if (norm < eps)
        norm = eps;
      float inv = mlp_g / norm;
      for (unsigned int s = 0; s < F; ++s)
        xp[k * F + s] *= inv;
    }


    // 2. mlp: H = X @ Wmlp  -> [K, D]
    Tensor H = X.dot(w_mlp);

    // 3. GAU on H -> G [K, D].
    // 3a. ScaleNorm over D (scalar gau_g).
    Tensor H_ln(kd, true);
    const float *hp = H.getData<float>();
    float *hlp = H_ln.getData<float>();
    for (unsigned int k = 0; k < K; ++k) {
      float ss = 0.0f;
      for (unsigned int d = 0; d < D; ++d)
        ss += hp[k * D + d] * hp[k * D + d];
      float norm = std::sqrt(ss) * gau_scale;
      if (norm < eps)
        norm = eps;
      float inv = gau_g / norm;
      for (unsigned int d = 0; d < D; ++d)
        hlp[k * D + d] = hp[k * D + d] * inv;
    }


    // 3b. uv = SiLU(H_ln @ Wuv) ; split u, v, base ; build q, k.
    Tensor uv = H_ln.dot(w_uv);
    float *uvp = uv.getData<float>();
    nntrainer::nntr_silu_inplace_f32(uvp, (size_t)K * UV);

    nntrainer::TensorDim ke(1, 1, K, E, nchw);
    nntrainer::TensorDim ks(1, 1, K, S, nchw);
    Tensor u(ke, true), v(ke, true), q(ks, true), kk(ks, true);
    float *up = u.getData<float>();
    float *vp = v.getData<float>();
    float *qp = q.getData<float>();
    float *kp = kk.getData<float>();
    for (unsigned int k = 0; k < K; ++k) {
      const float *row = uvp + k * UV;
      std::copy(row, row + E, up + k * E);
      std::copy(row + E, row + 2 * E, vp + k * E);
      const float *base = row + 2 * E;
      for (unsigned int s = 0; s < S; ++s) {
        qp[k * S + s] = base[s] * gamma_p[s] + beta_p[s];
        kp[k * S + s] = base[s] * gamma_p[S + s] + beta_p[S + s];
      }
    }

    // 3c. kernel = relu(q k^T / sqrt_s)^2 ; attn = kernel @ v ; gated = u*attn.
    Tensor qk = q.dot(kk, false, true); // [K, K]
    float *qkp = qk.getData<float>();
    for (unsigned int i = 0; i < K * K; ++i) {
      float r = qkp[i] / sqrt_s;
      r = r > 0.0f ? r : 0.0f;
      qkp[i] = r * r;
    }
    Tensor attn = qk.dot(v); // [K, E]
    float *ap = attn.getData<float>();
    for (unsigned int i = 0; i < K * E; ++i)
      ap[i] *= up[i];

    // 3d. out_gau = gated @ Wo ; G = res_scale * H + out_gau.
    Tensor out_gau = attn.dot(w_o); // [K, D]
    const float *ogp = out_gau.getData<float>();
    Tensor G(kd, true);
    float *gp = G.getData<float>();
    for (unsigned int k = 0; k < K; ++k)
      for (unsigned int d = 0; d < D; ++d)
        gp[k * D + d] = rs_p[d] * hp[k * D + d] + ogp[k * D + d];

    // 4. cls_x, cls_y ; write [cls_x rows ; cls_y rows] into the output.
    Tensor px = G.dot(w_cx); // [K, simcc]
    Tensor py = G.dot(w_cy); // [K, simcc]
    const float *pxp = px.getData<float>();
    const float *pyp = py.getData<float>();
    std::copy(pxp, pxp + K * simcc, outb);
    std::copy(pyp, pyp + K * simcc, outb + static_cast<size_t>(K) * simcc);
  }
}

void RTMCCHeadLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[rtmcc_head] training is not supported");
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rtmcc_head_layer() { return new RTMCCHeadLayer(); }

void destroy_rtmcc_head_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rtmcc_head_layer,
                                                   destroy_rtmcc_head_layer};
}

#endif

} // namespace quick_ai
