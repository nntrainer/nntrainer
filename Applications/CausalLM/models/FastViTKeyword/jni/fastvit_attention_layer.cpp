// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   fastvit_attention_layer.cpp
 * @date   7 July 2026
 * @brief  Multi-head attention custom layer for FastViT-S12 stage 3.
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
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

namespace fastvit_keyword {

void FastViTAttentionLayer::setProperty(
  const std::vector<std::string> &values) {
  nntrainer::LayerImpl::setProperty(values);
}

void FastViTAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &in_dim = context.getInputDimensions()[0];
  dim_ = in_dim.channel();
  // dim_ = 3 * C (e.g. 1536 for C=512)
  // C = dim_ / 3
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

  // The input tensor may be FP16 when the model uses FP32-FP16 tensor type.
  // Convert to FP32 for the attention computation, then write back.
  nntrainer::Tensor in_fp32 = in;
  if (in.getDataType() != ml::train::TensorDim::DataType::FP32) {
    in_fp32 = in.clone(ml::train::TensorDim::DataType::FP32);
  }

  const float *in_data = in_fp32.getData();

  // Output may also be FP16 — compute in FP32 then cast back.
  nntrainer::TensorDim out_dim = out.getDim();
  nntrainer::Tensor out_fp32(
    out_dim.batch(), out_dim.channel(), out_dim.height(), out_dim.width(),
    out_dim.getFormat(), ml::train::TensorDim::DataType::FP32);
  float *out_data = out_fp32.getData();

  multiHeadAttention(in_data, out_data, B, C, H, W, num_heads, head_dim, scale);

  // Cast FP32 result back to the output tensor's dtype
  if (out.getDataType() != ml::train::TensorDim::DataType::FP32) {
    out.copy(out_fp32);
  }
}

void FastViTAttentionLayer::multiHeadAttention(const float *qkv, float *out,
                                               int B, int C, int H, int W,
                                               int num_heads, int head_dim,
                                               float scale) {
  int N = H * W;
  int C3 = 3 * C;

  // PHYSICAL LAYOUT: this layer runs inside the NHWC backbone model
  // (video_pipeline.cpp sets tensor_format=NHWC for the FastViTKeyword
  // backbone). nntrainer stores NHWC tensors physically as [B, H, W, C] with
  // strides {H*W*C, W*C, C, 1} (tensor_dim.cpp computeStrides, NHWC branch) —
  // i.e. the CHANNEL axis is the innermost (stride 1), so a logical
  // coordinate (b, c, y, x) lives at offset  b*(N*C) + (y*W+x)*C + c.
  //
  // The qkv tensor has logical shape [B, 3*C, H, W] but is physically
  // [B, H, W, 3*C] with per-batch offset b*(N*C3) and the channel stored as
  // the last contiguous run of C3 floats per spatial location. Within that C3
  // run: Q = channels [0, C), K = [C, 2C), V = [2C, 3C).
  //
  // The output tensor has logical shape [B, C, H, W], physically [B, H, W, C].
  //
  // NOTE: the previous implementation indexed as Q_base[(h*hd+d)*N + i], which
  // assumed NCHW physical layout (channel stride N). Under NHWC that read
  // completely wrong memory → produced a consistent but incorrect result in
  // every dtype (FP32 and W8A16 idx=455 vs the ONNX reference idx=327). This
  // is the keyword accuracy root cause; the index math below is the fix.

  // For each batch element
  for (int b = 0; b < B; ++b) {
    const float *qkv_b = qkv + b * N * C3; // physical base of this batch

    // For each head
    for (int h = 0; h < num_heads; ++h) {
      // Logical channel index for this head's d-th component:
      //   Q: c_q = h*head_dim + d        (in [0, C))
      //   K: c_k = C + h*head_dim + d    (in [C, 2C))
      //   V: c_v = 2*C + h*head_dim + d  (in [2C, 3C))
      // Physical offset of logical (c, y, x): (y*W + x)*C3 + c.
      // With n := y*W + x, the inner-channel offset for head h is:
      //   q_off(d, n) = n*C3 + (h*head_dim + d)
      //   k_off(d, n) = n*C3 + (C + h*head_dim + d)
      //   v_off(d, n) = n*C3 + (2*C + h*head_dim + d)

      // Compute attention scores: attn[N, N] = (Q * scale) @ K^T
      // attn[i, j] = scale * sum_d Q[i, d] * K[j, d]
      std::vector<float> attn_scores(N * N);

      const int hd_off = h * head_dim; // logical channel offset within Q (and
                                       // within K +0, V +2C ranges)
      for (int i = 0; i < N; ++i) {
        const int q_i = i * C3 + hd_off; // Q[i, d] at q_i + d
        for (int j = 0; j < N; ++j) {
          const int k_j =
            j * C3 + C + hd_off; // K[j, d] at k_j + d (channels [C, 2C))
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            dot += qkv_b[q_i + d] * qkv_b[k_j + d];
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

      // Compute out = attn @ V.
      // Output is NHWC [B, H, W, C]: physical offset of logical (b, c, y, x) is
      // b*(N*C) + (y*W + x)*C + c = b*(N*C) + i*C + c.
      // V[j, d] is at qkv_b[j*C3 + (2C + h*head_dim + d)].
      float *out_base = out + b * N * C;
      const int v_chan_off = 2 * C + hd_off; // V[j, d] channel = v_chan_off + d
      for (int i = 0; i < N; ++i) {
        for (int d = 0; d < head_dim; ++d) {
          const int v_j_d = v_chan_off + d;
          float val = 0.0f;
          for (int j = 0; j < N; ++j) {
            val += attn_scores[i * N + j] * qkv_b[j * C3 + v_j_d];
          }
          out_base[i * C + (hd_off + d)] = val;
        }
      }
    }
  }
}

} // namespace fastvit_keyword
