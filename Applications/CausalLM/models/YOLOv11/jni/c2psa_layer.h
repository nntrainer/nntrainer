// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   c2psa_layer.h
 * @date   18 June 2026
 * @brief  PSA spatial multi-head attention custom layer for YOLOv11 C2PSA.
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs
 *
 * The C2PSA block (model.10) is assembled from standard nntrainer layers
 * (conv2d, depthwiseconv2d, batch_normalization, addition, concat, slice) —
 * see buildC2PSA() in main.cpp. The ONLY genuinely-novel op is the spatial
 * multi-head attention, kept here as a (weight-free) custom layer.
 *
 * Input : qkv tensor [B, 512, H, W]  (channels: Q[0:128], K[128:256],
 * V[256:512]) Output: attention  [B, 256, H, W]  (= nh*vd, nh=4, kd=32, vd=64)
 *   score[i,j] = softmax_j( sum_d Q[h,d,i]*K[h,d,j] / sqrt(kd) )
 *   out[h,d,i] = sum_j score[i,j] * V[h,d,j]
 */

#ifndef __C2PSA_LAYER_H__
#define __C2PSA_LAYER_H__

#include <string>
#include <vector>

#include <layer_devel.h>
#include <layer_impl.h>
#include <node_exporter.h>

namespace yolov11 {

/**
 * @class PSAAttentionLayer
 * @brief Spatial multi-head attention for YOLOv11 C2PSA (weight-free).
 *        Hardcoded nh=4, kd=32, vd=64: input 512 ch (Q128/K128/V256) -> 256 ch.
 */
class PSAAttentionLayer : public nntrainer::LayerImpl {
public:
  PSAAttentionLayer() : dim_(0) {}
  ~PSAAttentionLayer() override = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override {}
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}
  const std::string getType() const override { return PSAAttentionLayer::type; }
  void setProperty(const std::vector<std::string> &values) override {
    nntrainer::LayerImpl::setProperty(values);
  }
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "psa_attention";

private:
  /**
   * @brief Spatial multi-head attention.
   * Q: [nh, kd, N], K: [nh, kd, N], V: [nh, vd, N], out: [nh, vd, N].
   */
  static void multiHeadAttention(const float *Q, const float *K, const float *V,
                                 float *out, int nh, int kd, int vd, int N);

  unsigned int dim_; ///< input channels (= 512)
  static constexpr unsigned int NUM_HEADS = 4;
  static constexpr unsigned int KD = 32; ///< key dim per head
  static constexpr unsigned int VD = 64; ///< val dim per head
};

} // namespace yolov11

#endif // __C2PSA_LAYER_H__
