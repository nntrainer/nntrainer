// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   fastvit_backbone_graph.h
 * @date   15 July 2026
 * @brief  FastViT-S12 backbone graph block builders (inline, header-only).
 *
 * Builds the FastViT-S12 backbone (general feature extractor): stem + 4 stages
 * + final_conv, producing a [1, 1024, 10, 10] feature map for 320x320 input.
 *
 * The model is ported from deep-vision-models
 * (backbone/fastvit.py — standalone FastViT, num_features=1024):
 *   - timm FastViT-S12 (fused/reparameterized inference form)
 *   - forward_default returns [stage0..3, final] multi-scale feature list
 *
 * Architecture (fused, 320x320 input):
 *   stem0: Conv2d(3,64,k3,s2,p1,g1) + GELU          -> [1,64,160,160]
 *   stem1: Conv2d(64,64,k3,s2,p1,g64) + GELU        -> [1,64,80,80]
 *   stem2: Conv2d(64,64,k1,s1,p0,g1) + GELU          -> [1,64,80,80]
 *   stage0: 2x RepMixerBlock(64)                      -> [1,64,80,80]
 *   s1_down: Conv(64,128,k7,s2,g64) + Conv(128,128,k1) + GELU -> [1,128,40,40]
 *   stage1: 2x RepMixerBlock(128)                     -> [1,128,40,40]
 *   s2_down: Conv(128,256,k7,s2,g128) + Conv(256,256,k1) + GELU ->
 * [1,256,20,20] stage2: 6x RepMixerBlock(256)                     ->
 * [1,256,20,20] s3_down: Conv(256,512,k7,s2,g256) + Conv(512,512,k1) + GELU ->
 * [1,512,10,10] s3_posemb: Conv(512,512,k7,s1,g512)               ->
 * [1,512,10,10] stage3: 2x AttentionBlock(512, nh=16, hd=32)     ->
 * [1,512,10,10] final_conv: Conv(512,1024,k3,s1,g512) + SE(1024,rd=64) + GELU
 * -> [1,1024,10,10]
 *
 * All RepConv/BN pairs are fused (reparameterized) at inference time, so the
 * nntrainer graph uses single biased convolutions. Layer_scale gamma values
 * are folded into the preceding conv weights/biases at conversion time.
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#ifndef __FASTVIT_BACKBONE_GRAPH_H__
#define __FASTVIT_BACKBONE_GRAPH_H__

#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <tensor_api.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;

namespace fastvit {

/// @brief Global weight dtype for quantized conv layers (Q4_0, Q8_0, or FP32).
inline std::string &quantWeightDtype() {
  static std::string dtype = "Q4_0";
  return dtype;
}

// ===== Primitive graph block builders =====

/**
 * @brief Build a Conv2d + GELU block (BN already fused into conv at convert
 * time).
 */
inline Tensor convGelu(const std::string &name, int in_ch, int out_ch, int k,
                       int stride, int padding, int groups, Tensor input,
                       bool conv_q40 = false) {
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding),
    nntrainer::withKey("groups", groups)};
  if (conv_q40)
    conv_props.push_back(
      nntrainer::withKey("weight_dtype", quantWeightDtype()));
  LayerHandle conv(createLayer("conv2d", conv_props));
  auto x = conv(input);
  LayerHandle gelu(
    createLayer("activation", {nntrainer::withKey("name", name + "/gelu"),
                               nntrainer::withKey("activation", "gelu")}));
  return gelu(x);
}

/** @brief Build a Conv2d only (no BN, no activation) — for fused depthwise
 * token mixer. */
inline Tensor convOnly(const std::string &name, int in_ch, int out_ch, int k,
                       int stride, int padding, int groups, Tensor input,
                       bool conv_q40 = false) {
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding),
    nntrainer::withKey("groups", groups)};
  if (conv_q40)
    conv_props.push_back(
      nntrainer::withKey("weight_dtype", quantWeightDtype()));
  LayerHandle conv(createLayer("conv2d", conv_props));
  return conv(input);
}

/** @brief Build a depthwise Conv2d (BN fused at conversion time, no
 * activation). */
inline Tensor dwConvBn(const std::string &name, int ch, int k, Tensor input,
                       bool conv_q40 = false) {
  return convOnly(name + "/dw", ch, ch, k, 1, k / 2, ch, input, conv_q40);
}

/** @brief Build a 1x1 Conv2d with bias (used as Linear replacement) + GELU. */
inline Tensor conv1x1Gelu(const std::string &name, int in_ch, int out_ch,
                          Tensor input, bool conv_q40 = false) {
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {1, 1}),
    nntrainer::withKey("filters", out_ch), nntrainer::withKey("stride", {1, 1}),
    nntrainer::withKey("padding", 0)};
  if (conv_q40)
    conv_props.push_back(
      nntrainer::withKey("weight_dtype", quantWeightDtype()));
  LayerHandle conv(createLayer("conv2d", conv_props));
  auto x = conv(input);
  LayerHandle gelu(
    createLayer("activation", {nntrainer::withKey("name", name + "/gelu"),
                               nntrainer::withKey("activation", "gelu")}));
  return gelu(x);
}

/**
 * @brief Build a 1x1 Conv2d (no activation) — used as Linear replacement.
 * @param no_bias  If true, disable bias.
 */
inline Tensor conv1x1Only(const std::string &name, int in_ch, int out_ch,
                          Tensor input, bool no_bias = false,
                          bool conv_q40 = false) {
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {1, 1}),
    nntrainer::withKey("filters", out_ch), nntrainer::withKey("stride", {1, 1}),
    nntrainer::withKey("padding", 0)};
  if (no_bias)
    conv_props.push_back(nntrainer::withKey("disable_bias", "true"));
  if (conv_q40)
    conv_props.push_back(
      nntrainer::withKey("weight_dtype", quantWeightDtype()));
  LayerHandle conv(createLayer("conv2d", conv_props));
  return conv(input);
}

/** @brief Elementwise addition of two tensors. */
inline Tensor addT(const std::string &name, Tensor a, Tensor b) {
  LayerHandle l(createLayer("Addition", {nntrainer::withKey("name", name)}));
  return l({a, b});
}

/**
 * @brief Build a RepMixerBlock (fused inference form).
 *
 * Forward: x = reparam_conv(x); x = x + layer_scale(mlp(x))
 * layer_scale gamma is folded into mlp_fc2 weights at conversion time.
 */
inline Tensor repMixerBlock(const std::string &name, int ch, Tensor input,
                            bool conv_q40 = false) {
  auto x = convOnly(name + "/tm", ch, ch, 3, 1, 1, ch, input, conv_q40);
  auto mlp_conv = dwConvBn(name + "/mlp_conv", ch, 7, x, conv_q40);
  auto mlp_fc1 = conv1x1Gelu(name + "/mlp_fc1", ch, ch * 4, mlp_conv, conv_q40);
  auto mlp_fc2 =
    conv1x1Only(name + "/mlp_fc2", ch * 4, ch, mlp_fc1, false, conv_q40);
  return addT(name + "/add", x, mlp_fc2);
}

/**
 * @brief Build a downsample block (fused inference form).
 * proj.0: dw 7x7 conv (s2, groups=in_ch); proj.1: 1x1 conv + GELU
 */
inline Tensor downsampleBlock(const std::string &name, int in_ch, int out_ch,
                              Tensor input, bool conv_q40 = false) {
  auto x =
    convOnly(name + "/down0", in_ch, out_ch, 7, 2, 3, in_ch, input, conv_q40);
  return convGelu(name + "/down1", out_ch, out_ch, 1, 1, 0, 1, x, conv_q40);
}

/**
 * @brief Build an AttentionBlock (fused inference form, stage 3).
 *
 * Forward:
 *   x = x + layer_scale_1(token_mixer(x))
 *   x = x + layer_scale_2(mlp(x))
 *
 * In the source model's fused form (AttentionBlockOptimized), the pre-qkv
 * BatchNorm2d (`norm`) is folded INTO the qkv 1x1 conv via
 * `reparameterize_bn_conv(norm, qkv)`, and the `norm` module is deleted.
 * Therefore the nntrainer graph has NO batch_normalization layer here, and the
 * qkv conv carries the folded bias.
 */
inline Tensor attentionBlock(const std::string &name, int ch, Tensor input,
                             bool conv_q40 = false) {
  // No batch_normalization: the pre-qkv BN was folded into qkv at conversion
  // time (AttentionBlockOptimized.fuse -> reparameterize_bn_conv(norm, qkv)).
  auto qkv = conv1x1Only(name + "/qkv", ch, ch * 3, input, false, conv_q40);

  LayerHandle attn(createLayer("fastvit_attention",
                               {nntrainer::withKey("name", name + "/attn")}));
  auto attn_out = attn(qkv);

  // layer_scale_1 gamma folded into proj weights at conversion time
  auto proj = conv1x1Only(name + "/proj", ch, ch, attn_out, false, conv_q40);
  auto x = addT(name + "/res1", input, proj);

  // layer_scale_2 gamma folded into mlp_fc2 weights at conversion time
  auto mlp_conv = dwConvBn(name + "/mlp_conv", ch, 7, x, conv_q40);
  auto mlp_fc1 = conv1x1Gelu(name + "/mlp_fc1", ch, ch * 4, mlp_conv, conv_q40);
  auto mlp_fc2 =
    conv1x1Only(name + "/mlp_fc2", ch * 4, ch, mlp_fc1, false, conv_q40);
  return addT(name + "/res2", x, mlp_fc2);
}

/**
 * @brief Build the SE (Squeeze-and-Excitation) module.
 *   x_se = gap(x) -> fc1+ReLU -> fc2+Sigmoid -> out = x * x_se
 */
inline Tensor seModule(const std::string &name, int ch, int rd, Tensor input,
                       bool conv_q40 = false) {
  LayerHandle gap_h(
    createLayer("reduce_mean", {nntrainer::withKey("name", name + "/gap_h"),
                                nntrainer::withKey("axis", 2)}));
  LayerHandle gap_w(
    createLayer("reduce_mean", {nntrainer::withKey("name", name + "/gap_w"),
                                nntrainer::withKey("axis", 3)}));
  auto pooled = gap_w(gap_h(input));

  auto fc1 = convOnly(name + "/se_fc1", ch, rd, 1, 1, 0, 1, pooled, conv_q40);
  LayerHandle relu(
    createLayer("activation", {nntrainer::withKey("name", name + "/se_relu"),
                               nntrainer::withKey("activation", "relu")}));
  auto relu_out = relu(fc1);

  auto fc2 = convOnly(name + "/se_fc2", rd, ch, 1, 1, 0, 1, relu_out, conv_q40);
  LayerHandle sigmoid(
    createLayer("activation", {nntrainer::withKey("name", name + "/se_sigmoid"),
                               nntrainer::withKey("activation", "sigmoid")}));
  auto sig_out = sigmoid(fc2);

  LayerHandle mul(
    createLayer("multiply", {nntrainer::withKey("name", name + "/se_mul")}));
  return mul({input, sig_out});
}

/** @brief Build the final conv block (dw3x3 reparam + SE + GELU). */
inline Tensor finalConvBlock(const std::string &name, int in_ch, int out_ch,
                             Tensor input, bool conv_q40 = false) {
  auto conv = convOnly(name, in_ch, out_ch, 3, 1, 1, in_ch, input, conv_q40);
  auto se = seModule(name + "/se", out_ch, out_ch / 16, conv, conv_q40);
  LayerHandle gelu(
    createLayer("activation", {nntrainer::withKey("name", name + "/gelu"),
                               nntrainer::withKey("activation", "gelu")}));
  return gelu(se);
}

// ===== Top-level whole-network builder =====

/**
 * @brief Build the FastViT-S12 backbone (stem + 4 stages + final_conv).
 * @param xIn  Input symbolic tensor [1, 3, 320, 320]
 * @return Output symbolic tensor [1, 1024, 10, 10]
 */
inline Tensor buildBackbone(Tensor xIn, bool conv_q40 = false) {
  // === Stem (3 layers) ===
  auto x =
    convGelu("stem0", 3, 64, 3, 2, 1, 1, xIn, conv_q40);   // -> [1,64,160,160]
  x = convGelu("stem1", 64, 64, 3, 2, 1, 64, x, conv_q40); // -> [1,64,80,80]
  x = convGelu("stem2", 64, 64, 1, 1, 0, 1, x, conv_q40);  // -> [1,64,80,80]

  // === Stage 0: 2x RepMixerBlock(64) -> [1,64,80,80] ===
  x = repMixerBlock("s0b0", 64, x, conv_q40);
  x = repMixerBlock("s0b1", 64, x, conv_q40);

  // === Stage 1: downsample + 2x RepMixerBlock(128) -> [1,128,40,40] ===
  x = downsampleBlock("s1_down", 64, 128, x, conv_q40);
  x = repMixerBlock("s1b0", 128, x, conv_q40);
  x = repMixerBlock("s1b1", 128, x, conv_q40);

  // === Stage 2: downsample + 6x RepMixerBlock(256) -> [1,256,20,20] ===
  x = downsampleBlock("s2_down", 128, 256, x, conv_q40);
  for (int b = 0; b < 6; ++b)
    x = repMixerBlock("s2b" + std::to_string(b), 256, x, conv_q40);

  // === Stage 3: downsample + pos_emb + 2x AttentionBlock(512) -> [1,512,10,10]
  // ===
  x = downsampleBlock("s3_down", 256, 512, x, conv_q40);
  x = convOnly("s3_posemb", 512, 512, 7, 1, 3, 512, x, conv_q40);
  x = attentionBlock("s3b0", 512, x, conv_q40);
  x = attentionBlock("s3b1", 512, x, conv_q40);

  // === Final conv: 512 -> 1024 (dw3x3 reparam + SE + GELU) -> [1,1024,10,10]
  // ===
  x = finalConvBlock("final_conv", 512, 1024, x, conv_q40);
  return x;
}

} // namespace fastvit

#endif // __FASTVIT_BACKBONE_GRAPH_H__
