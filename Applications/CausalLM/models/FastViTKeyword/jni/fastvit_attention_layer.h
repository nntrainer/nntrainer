// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   fastvit_attention_layer.h
 * @date   7 July 2026
 * @brief  Multi-head attention custom layer for FastViT-S12 stage 3.
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This layer implements the attention mechanism used in timm's FastViT
 * AttentionBlock (stages.3.blocks.*). It is weight-free: the qkv projection
 * and output projection are done by separate 1x1 conv layers in the graph.
 *
 * Input : qkv tensor [B, 3*C, H, W]  (channels: Q[0:C], K[C:2C], V[2C:3C])
 * Output: attention  [B, C, H, W]    (same C as input, C=512 for sa12)
 *
 * Attention computation (matching timm Attention.forward, non-fused path):
 *   B, C, H, W = shape
 *   N = H * W
 *   Q = qkv[0:C].reshape(B, N, num_heads, head_dim).permute(B, num_heads, N,
 * head_dim) K = qkv[C:2C].reshape(B, N, num_heads, head_dim)  # no permute
 * (used as B, nh, hd, N) V = qkv[2C:3C].reshape(B, N, num_heads,
 * head_dim).permute(B, num_heads, N, head_dim) scale = 1 / sqrt(head_dim) attn
 * = (Q * scale) @ K^T  -> [B, nh, N, N] attn = softmax(attn, dim=-1) out = attn
 * @ V            -> [B, nh, N, head_dim] out = out.transpose(1,2).reshape(B, N,
 * C).transpose(-2,-1).reshape(B, C, H, W)
 *
 * For sa12: C=512, num_heads=16, head_dim=32, scale=1/sqrt(32)≈0.17677
 */

#ifndef __FASTVIT_ATTENTION_LAYER_H__
#define __FASTVIT_ATTENTION_LAYER_H__

#include <string>
#include <vector>

#include <layer_devel.h>
#include <layer_impl.h>
#include <node_exporter.h>

namespace fastvit_keyword {

/**
 * @class FastViTAttentionLayer
 * @brief Multi-head attention for FastViT-S12 stage 3 (weight-free).
 *
 * Input: [B, 3*C, H, W] where C = num_heads * head_dim
 * Output: [B, C, H, W]
 *
 * The attention follows timm's Attention.forward (non-fused path):
 *   Q = qkv[:, 0:C]      reshaped to [B, nh, N, hd]
 *   K = qkv[:, C:2C]     reshaped to [B, nh, hd, N]  (transposed)
 *   V = qkv[:, 2C:3C]    reshaped to [B, nh, N, hd]
 *   attn = softmax((Q * scale) @ K^T, dim=-1)
 *   out = attn @ V        -> [B, nh, N, hd] -> reshape to [B, C, H, W]
 */
class FastViTAttentionLayer : public nntrainer::LayerImpl {
public:
  FastViTAttentionLayer() : dim_(0), num_heads_(16) {}
  ~FastViTAttentionLayer() override = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override {}
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}
  const std::string getType() const override {
    return FastViTAttentionLayer::type;
  }
  void setProperty(const std::vector<std::string> &values) override;
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "fastvit_attention";

private:
  /**
   * @brief Multi-head attention forward pass.
   *
   * @param qkv    Input [B, 3*C, H, W] (Q|K|V concatenated along channel)
   * @param out    Output [B, C, H, W]
   * @param B      Batch size
   * @param C      Channels (= num_heads * head_dim)
   * @param H      Height
   * @param W      Width
   * @param num_heads  Number of attention heads
   * @param head_dim    Dimension per head
   * @param scale   Attention scale (1/sqrt(head_dim))
   */
  static void multiHeadAttention(const float *qkv, float *out, int B, int C,
                                 int H, int W, int num_heads, int head_dim,
                                 float scale);

  unsigned int dim_;       ///< input channels (= 3 * C, e.g. 1536)
  unsigned int num_heads_; ///< number of attention heads (16 for sa12)
  unsigned int head_dim_;  ///< dimension per head (32 for sa12)
};

} // namespace fastvit_keyword

#endif // __FASTVIT_ATTENTION_LAYER_H__
