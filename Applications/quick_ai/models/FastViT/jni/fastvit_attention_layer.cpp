// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   fastvit_attention_layer.cpp
 * @date   15 July 2026
 * @brief  Multi-head attention custom layer for FastViT-S12 stage 3.
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#include "fastvit_attention_layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace fastvit {

void FastViTAttentionLayer::setProperty(
  const std::vector<std::string> &values) {
  nntrainer::LayerImpl::setProperty(values);
}

void FastViTAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &in_dim = context.getInputDimensions()[0];
  dim_ = in_dim.channel();
  unsigned int C = dim_ / 3;
  head_dim_ = C / num_heads_;

  nntrainer::TensorDim out_dim(in_dim.batch(), C, in_dim.height(),
                               in_dim.width(), in_dim.getFormat(),
                               in_dim.getDataType());
  context.setOutputDimensions({out_dim});
}

void FastViTAttentionLayer::forwarding(nntrainer::RunLayerContext &context,
                                       bool training) {
  const nntrainer::Tensor &in = context.getInput(0);
  nntrainer::Tensor &out = context.getOutput(0);

  const auto &in_dim = in.getDim();
  int B = in_dim.batch();
  int C3 = in_dim.channel(); // 3 * C
  int H = in_dim.height();
  int W = in_dim.width();
  int C = C3 / 3; // C = 512
  int num_heads = num_heads_;
  int head_dim = C / num_heads; // 32
  float scale = 1.0f / std::sqrt((float)head_dim);

  // The attention computation assumes NCHW memory layout.
  // If the tensor is NHWC, convert to NCHW first, then back.
  if (in_dim.getFormat() == nntrainer::TensorDim::Format::NHWC) {
    nntrainer::Tensor in_nchw(
      in_dim.batch(), in_dim.channel(), in_dim.height(), in_dim.width(),
      nntrainer::TensorDim::Format::NCHW, in_dim.getDataType());
    in_nchw.copyData(in);

    nntrainer::Tensor &out_ref = context.getOutput(0);
    nntrainer::Tensor out_nchw(
      out_ref.getDim().batch(), out_ref.getDim().channel(),
      out_ref.getDim().height(), out_ref.getDim().width(),
      nntrainer::TensorDim::Format::NCHW, out_ref.getDim().getDataType());

    const float *in_data = in_nchw.getData();
    float *out_data = out_nchw.getData();

    multiHeadAttention(in_data, out_data, B, C, H, W, num_heads, head_dim,
                       scale);

    // Copy back to NHWC output
    out_ref.copyData(out_nchw);
  } else {
    const float *in_data = in.getData();
    float *out_data = out.getData();

    multiHeadAttention(in_data, out_data, B, C, H, W, num_heads, head_dim,
                       scale);
  }
}

void FastViTAttentionLayer::multiHeadAttention(const float *qkv, float *out,
                                               int B, int C, int H, int W,
                                               int num_heads, int head_dim,
                                               float scale) {
  int N = H * W;
  int C3 = 3 * C;

  for (int b = 0; b < B; ++b) {
    const float *Q_base = qkv + b * C3 * H * W;
    const float *K_base = Q_base + C * H * W;
    const float *V_base = K_base + C * H * W;

    for (int h = 0; h < num_heads; ++h) {
      std::vector<float> attn_scores(N * N);

      // Compute Q * K^T * scale
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            float q_val = Q_base[(h * head_dim + d) * N + i];
            float k_val = K_base[(h * head_dim + d) * N + j];
            dot += q_val * k_val;
          }
          attn_scores[i * N + j] = dot * scale;
        }
      }

      // Softmax over j (last dim)
      for (int i = 0; i < N; ++i) {
        float max_val = attn_scores[i * N];
        for (int j = 1; j < N; ++j) {
          max_val = std::max(max_val, attn_scores[i * N + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
          attn_scores[i * N + j] = std::exp(attn_scores[i * N + j] - max_val);
          sum += attn_scores[i * N + j];
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < N; ++j) {
          attn_scores[i * N + j] *= inv_sum;
        }
      }

      // Compute out = attn @ V
      float *out_base = out + b * C * H * W;
      for (int i = 0; i < N; ++i) {
        for (int d = 0; d < head_dim; ++d) {
          float val = 0.0f;
          for (int j = 0; j < N; ++j) {
            val += attn_scores[i * N + j] * V_base[(h * head_dim + d) * N + j];
          }
          out_base[(h * head_dim + d) * N + i] = val;
        }
      }
    }
  }
}

} // namespace fastvit
