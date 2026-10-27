// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   fastvit_keyword_graph.h
 * @date   7 July 2026
 * @brief  FastViTKeyword graph block builders (inline, header-only).
 *
 * Builds the FastViT-S12 backbone + MLP classification head for the
 * keyword classification task (507-class multi-label).
 *
 * The model is ported from deep-vision-models (dev/keyword/train-inference):
 *   - Backbone: timm FastViT-S12 (fused/reparameterized inference form)
 *   - Head: Linear(1024,1024) + LayerNorm(1024) + GELU + Dropout +
 * Linear(1024,1019)
 *   - Output: (logits[507], features[512]) = split of final_layer output
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
 * -> [1,1024,10,10] global_avg_pool: mean(dim=[2,3])                  ->
 * [1,1024] head_mlp: Linear(1024,1024) + LayerNorm(1024) + GELU -> [1,1024]
 *   head_final: Linear(1024,1019)                      -> [1,1019]
 *   output: split -> logits[1,507], features[1,512]
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __FASTVIT_KEYWORD_GRAPH_H__
#define __FASTVIT_KEYWORD_GRAPH_H__

#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <tensor_api.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;

namespace fastvit_keyword {

/// Channel axis for concat/slice (always logical axis 1 = channel)
inline int chAxis() { return 1; }

/// Global weight dtype for Q8_0 quantization. Set to "Q8_0" to enable
/// W8A16 (Q8_0 weights + FP16 activations). Default "FP32".
inline std::string &quantWeightDtype() {
  static std::string dtype = "FP32";
  return dtype;
}

/// Collect names of conv layers eligible for Q8_0 quantization.
/// A conv is eligible when out_ch % 32 == 0 and (in_ch * k * k) % 32 == 0
/// (Q8_0 block size = 32).
inline std::vector<std::string> &quantizableConvs() {
  static std::vector<std::string> names;
  return names;
}

/// Check if a conv layer is eligible for Q8_0 quantization.
/// Q8_0 requires: groups==1 (quant_matmul_filter), out_ch%32==0,
/// and (in_ch*k*k)%32==0 (Q8_0 block size = 32).
inline bool convQuantEligible(int in_ch, int out_ch, int k, int groups = 1) {
  return groups == 1 && out_ch % 32 == 0 && (in_ch * k * k) % 32 == 0;
}

// ===== Primitive graph block builders =====

/**
 * @brief Build a Conv2d + GELU block (BN already fused into conv at convert
 * time).
 *
 * @param name    Layer-name prefix
 * @param in_ch   Input channels
 * @param out_ch  Output channels
 * @param k       Kernel size (square)
 * @param stride  Stride
 * @param padding Padding
 * @param groups  Groups (1 for standard, ch for depthwise)
 * @param input   Input symbolic tensor
 * @return Output symbolic tensor after conv+GELU
 */
inline Tensor convGelu(const std::string &name, int in_ch, int out_ch, int k,
                       int stride, int padding, int groups, Tensor input) {
  const bool eligible = convQuantEligible(in_ch, out_ch, k, groups);
  if (eligible)
    quantizableConvs().push_back(name + "/conv");
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding),
    nntrainer::withKey("groups", groups)};
  if (quantWeightDtype() != "FP32" && eligible)
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
 * @brief Build a Conv2d only (no BN, no activation) — for fused depthwise token
 * mixer.
 */
inline Tensor convOnly(const std::string &name, int in_ch, int out_ch, int k,
                       int stride, int padding, int groups, Tensor input) {
  const bool eligible = convQuantEligible(in_ch, out_ch, k, groups);
  if (eligible)
    quantizableConvs().push_back(name + "/conv");
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding),
    nntrainer::withKey("groups", groups)};
  if (quantWeightDtype() != "FP32" && eligible)
    conv_props.push_back(
      nntrainer::withKey("weight_dtype", quantWeightDtype()));
  LayerHandle conv(createLayer("conv2d", conv_props));
  return conv(input);
}

/**
 * @brief Build a depthwise Conv2d (BN fused at conversion time, no activation).
 */
inline Tensor dwConvBn(const std::string &name, int ch, int k, Tensor input) {
  // BN is fused into conv at conversion time (model.fuse())
  return convOnly(name + "/dw", ch, ch, k, 1, k / 2, ch, input);
}

/**
 * @brief Build a 1x1 Conv2d with bias (used as Linear replacement) + GELU.
 */
inline Tensor conv1x1Gelu(const std::string &name, int in_ch, int out_ch,
                          Tensor input) {
  const bool eligible = convQuantEligible(in_ch, out_ch, 1, 1);
  if (eligible)
    quantizableConvs().push_back(name + "/conv");
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {1, 1}),
    nntrainer::withKey("filters", out_ch), nntrainer::withKey("stride", {1, 1}),
    nntrainer::withKey("padding", 0)};
  if (quantWeightDtype() != "FP32" && eligible)
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
 * @param no_bias  If true, disable bias (for qkv which has bias=False in timm).
 */
inline Tensor conv1x1Only(const std::string &name, int in_ch, int out_ch,
                          Tensor input, bool no_bias = false) {
  const bool eligible = convQuantEligible(in_ch, out_ch, 1, 1);
  if (eligible)
    quantizableConvs().push_back(name + "/conv");
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {1, 1}),
    nntrainer::withKey("filters", out_ch), nntrainer::withKey("stride", {1, 1}),
    nntrainer::withKey("padding", 0)};
  if (no_bias)
    conv_props.push_back(nntrainer::withKey("disable_bias", "true"));
  if (quantWeightDtype() != "FP32" && eligible)
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

/** @brief Channel-axis slice [start0, end0) — slice layer uses 1-indexed. */
inline Tensor sliceCh(const std::string &name, int start0, int end0,
                      Tensor in) {
  LayerHandle s(
    createLayer("slice", {nntrainer::withKey("name", name),
                          nntrainer::withKey("axis", chAxis()),
                          nntrainer::withKey("start_index", start0 + 1),
                          nntrainer::withKey("end_index", end0 + 1)}));
  return s(in);
}

/** @brief Channel-axis concat. */
inline Tensor concatCh(const std::string &name,
                       const std::vector<Tensor> &ins) {
  LayerHandle l(createLayer("concat", {nntrainer::withKey("name", name),
                                       nntrainer::withKey("axis", chAxis())}));
  return l(ins);
}

/**
 * @brief Build a RepMixerBlock (fused inference form).
 *
 * Forward: x = reparam_conv(x); x = x + layer_scale(mlp(x))
 *
 * reparam_conv: dw 3x3 conv (groups=ch, bias, no act)
 * mlp: dw 7x7 conv + BN, then 1x1 fc1 + GELU, then 1x1 fc2
 * layer_scale: elementwise multiply by gamma [C,1,1]
 *
 * @param name  Layer-name prefix
 * @param ch    Channels (input == output)
 * @param input Input symbolic tensor
 * @return Output symbolic tensor
 */
inline Tensor repMixerBlock(const std::string &name, int ch, Tensor input) {
  // token_mixer: dw 3x3 reparam_conv (no act)
  auto x = convOnly(name + "/tm", ch, ch, 3, 1, 1, ch, input);

  // mlp: dw 7x7 conv (BN fused), fc1(1x1)+GELU, fc2(1x1)
  // layer_scale gamma is folded into mlp_fc2 weights at conversion time
  auto mlp_conv = dwConvBn(name + "/mlp_conv", ch, 7, x);
  auto mlp_fc1 = conv1x1Gelu(name + "/mlp_fc1", ch, ch * 4, mlp_conv);
  auto mlp_fc2 = conv1x1Only(name + "/mlp_fc2", ch * 4, ch, mlp_fc1);

  // residual add (layer_scale already folded into mlp_fc2)
  return addT(name + "/add", x, mlp_fc2);
}

/**
 * @brief Build a downsample block (fused inference form).
 *
 * proj.0: dw 7x7 conv (s2, groups=in_ch, bias) — fused from
 * large_conv+small_conv proj.1: 1x1 conv (bias) + GELU
 *
 * @param name    Layer-name prefix
 * @param in_ch   Input channels
 * @param out_ch  Output channels
 * @param input   Input symbolic tensor
 * @return Output symbolic tensor
 */
inline Tensor downsampleBlock(const std::string &name, int in_ch, int out_ch,
                              Tensor input, int expose_down0_stop = 0) {
  // dw 7x7 conv, stride 2 (fused large+small conv)
  auto x = convOnly(name + "/down0", in_ch, out_ch, 7, 2, 3, in_ch, input);
  if (expose_down0_stop)
    return x; // bisect: after down0 only
  // 1x1 conv + GELU
  return convGelu(name + "/down1", out_ch, out_ch, 1, 1, 0, 1, x);
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
 * qkv conv carries the folded bias. token_mixer: qkv (1x1 conv, 512->1536,
 * biased), attention, proj (1x1 conv, 512->512, bias). Attention: 16 heads,
 * head_dim=32, scale=1/sqrt(32). mlp: dw 7x7 conv + BN, fc1(1x1,
 * 512->2048)+GELU, fc2(1x1, 2048->512). layer_scale_1, layer_scale_2 gamma
 * [512,1,1] are folded into proj / mlp_fc2 weights at conversion time.
 *
 * @param name  Layer-name prefix
 * @param ch    Channels (512)
 * @param input Input symbolic tensor
 * @return Output symbolic tensor
 */
inline Tensor attentionBlock(const std::string &name, int ch, Tensor input) {
  // No batch_normalization: the pre-qkv BN was folded into qkv at conversion
  // time (AttentionBlockOptimized.fuse -> reparameterize_bn_conv(norm, qkv)).

  // qkv: 1x1 conv (512 -> 1536, WITH bias — the folded norm bias)
  auto qkv = conv1x1Only(name + "/qkv", ch, ch * 3, input);

  // attention (custom layer: input qkv [B, 1536, H, W] -> output [B, 512, H,
  // W]) num_heads=16 is hardcoded in FastViTAttentionLayer (matching
  // FastViT-S12)
  LayerHandle attn(createLayer("fastvit_attention",
                               {nntrainer::withKey("name", name + "/attn")}));
  auto attn_out = attn(qkv);

  // proj: 1x1 conv (512 -> 512, bias)
  // layer_scale_1 gamma is folded into proj weights at conversion time
  auto proj = conv1x1Only(name + "/proj", ch, ch, attn_out);

  // residual add 1 (layer_scale_1 already folded into proj)
  auto x = addT(name + "/res1", input, proj);

  // mlp: dw 7x7 conv (BN fused), fc1(1x1, 512->2048)+GELU, fc2(1x1, 2048->512)
  // layer_scale_2 gamma is folded into mlp_fc2 weights at conversion time
  auto mlp_conv = dwConvBn(name + "/mlp_conv", ch, 7, x);
  auto mlp_fc1 = conv1x1Gelu(name + "/mlp_fc1", ch, ch * 4, mlp_conv);
  auto mlp_fc2 = conv1x1Only(name + "/mlp_fc2", ch * 4, ch, mlp_fc1);

  // residual add 2 (layer_scale_2 already folded into mlp_fc2)
  return addT(name + "/res2", x, mlp_fc2);
}

/**
 * @brief Build the SE (Squeeze-and-Excitation) module.
 *
 * Forward:
 *   x_se = global_avg_pool(x)           -> [B, C, 1, 1]
 *   x_se = fc1(x_se) + ReLU             -> [B, rd, 1, 1]
 *   x_se = fc2(x_se) + Sigmoid          -> [B, C, 1, 1]
 *   out = x * x_se
 *
 * @param name  Layer-name prefix
 * @param ch    Channels (1024)
 * @param rd    Reduction channels (64)
 * @param input Input symbolic tensor
 * @return Output symbolic tensor
 */
inline Tensor seModule(const std::string &name, int ch, int rd, Tensor input) {
  // global average pool: [B, C, H, W] -> [B, C, 1, 1]
  // Cascaded single-axis reductions to support NNTrainer's
  // props::ReduceDimension
  LayerHandle gap_h(
    createLayer("reduce_mean", {nntrainer::withKey("name", name + "/gap_h"),
                                nntrainer::withKey("axis", 2)}));
  LayerHandle gap_w(
    createLayer("reduce_mean", {nntrainer::withKey("name", name + "/gap_w"),
                                nntrainer::withKey("axis", 3)}));
  auto pooled = gap_w(gap_h(input));

  // fc1: 1x1 conv (C -> rd) + ReLU
  auto fc1 = convOnly(name + "/se_fc1", ch, rd, 1, 1, 0, 1, pooled);
  LayerHandle relu(
    createLayer("activation", {nntrainer::withKey("name", name + "/se_relu"),
                               nntrainer::withKey("activation", "relu")}));
  auto relu_out = relu(fc1);

  // fc2: 1x1 conv (rd -> C) + Sigmoid
  auto fc2 = convOnly(name + "/se_fc2", rd, ch, 1, 1, 0, 1, relu_out);
  LayerHandle sigmoid(
    createLayer("activation", {nntrainer::withKey("name", name + "/se_sigmoid"),
                               nntrainer::withKey("activation", "sigmoid")}));
  auto sig_out = sigmoid(fc2);

  // multiply input by SE weights
  LayerHandle mul(
    createLayer("multiply", {nntrainer::withKey("name", name + "/se_mul")}));
  return mul({input, sig_out});
}

/**
 * @brief Build the final conv block (MobileOneBlock with SE, fused).
 *
 * Forward: out = GELU(SE(reparam_conv(x)))
 * reparam_conv: dw 3x3 conv (groups=512, bias) — fused from kxk+scale
 *
 * @param name  Layer-name prefix
 * @param in_ch  Input channels (512)
 * @param out_ch Output channels (1024)
 * @param input  Input symbolic tensor
 * @return Output symbolic tensor
 */
inline Tensor finalConvBlock(const std::string &name, int in_ch, int out_ch,
                             Tensor input) {
  // dw 3x3 reparam_conv (groups=in_ch, bias)
  // convOnly already appends "/conv", so layer name = name/conv
  auto conv = convOnly(name, in_ch, out_ch, 3, 1, 1, in_ch, input);
  // SE module
  auto se = seModule(name + "/se", out_ch, out_ch / 16, conv);
  // GELU
  LayerHandle gelu(
    createLayer("activation", {nntrainer::withKey("name", name + "/gelu"),
                               nntrainer::withKey("activation", "gelu")}));
  return gelu(se);
}

// ===== Top-level whole-network builders =====

/**
 * @brief Build the FastViT-S12 backbone (stem + 4 stages + final_conv).
 *
 * @param xIn  Input symbolic tensor [1, 3, 320, 320]
 * @return Output symbolic tensor [1, 1024, 10, 10]
 */
inline Tensor buildBackbone(Tensor xIn, int stop_at = -1);

/**
 * @brief Build backbone optionally truncated after a stage (for golden-ref
 * stage bisection). stop_at<0 = full backbone.
 *   stop_at 0 -> returns output after stage0 (2x RepMixer 64)  [1,64,80,80]
 *   stop_at 1 -> after stage1                                  [1,128,40,40]
 *   stop_at 2 -> after stage2                                  [1,256,20,20]
 *   stop_at 3 -> after stage3 (attention)                      [1,512,10,10]
 */
inline Tensor buildBackbone(Tensor xIn, int stop_at) {
  // === Stem (3 layers) ===
  auto x = convGelu("stem0", 3, 64, 3, 2, 1, 1, xIn); // -> [1,64,160,160]
  x = convGelu("stem1", 64, 64, 3, 2, 1, 64, x);      // -> [1,64,80,80]
  x = convGelu("stem2", 64, 64, 1, 1, 0, 1, x);       // -> [1,64,80,80]

  // === Stage 0: 2x RepMixerBlock(64) -> [1,64,80,80] ===
  x = repMixerBlock("s0b0", 64, x);
  x = repMixerBlock("s0b1", 64, x);
  if (stop_at == 0)
    return x;

  // === Stage 1: downsample + 2x RepMixerBlock(128) -> [1,128,40,40] ===
  x = downsampleBlock("s1_down", 64, 128, x,
                      /*expose_down0_stop=*/(stop_at == -12));
  if (stop_at == -12)
    return x;
  if (stop_at == -10)
    return x; // after s1_downsample only [1,128,40,40]
  x = repMixerBlock("s1b0", 128, x);
  if (stop_at == -11)
    return x; // after s1 block0 [1,128,40,40]
  x = repMixerBlock("s1b1", 128, x);
  if (stop_at == 1)
    return x;

  // === Stage 2: downsample + 6x RepMixerBlock(256) -> [1,256,20,20] ===
  x = downsampleBlock("s2_down", 128, 256, x);
  if (stop_at == -20)
    return x; // after s2 downsample only [1,256,20,20]
  for (int b = 0; b < 6; ++b)
    x = repMixerBlock("s2b" + std::to_string(b), 256, x);
  if (stop_at == 2)
    return x;

  // === Stage 3: downsample + pos_emb + 2x AttentionBlock(512) -> [1,512,10,10]
  // === FastViT-S12 stage3 downsample (256->512, stride2), then
  // RepConditionalPosEnc (depthwise 7x7 conv, groups=512) applied to the
  // downsampled features, then 2 reparameterized AttentionBlocks.
  x = downsampleBlock("s3_down", 256, 512, x);
  if (stop_at == -30)
    return x; // after s3 downsample only [1,512,10,10]
  x = convOnly("s3_posemb", 512, 512, 7, 1, 3, 512, x);
  x = attentionBlock("s3b0", 512, x);
  x = attentionBlock("s3b1", 512, x);
  if (stop_at == 3)
    return x;

  // === Final conv: 512 -> 1024 (dw3x3 reparam + SE + GELU) -> [1,1024,10,10]
  // ===
  x = finalConvBlock("final_conv", 512, 1024, x);
  return x;
}

/**
 * @brief Build the classification head (MLP).
 *
 * Forward:
 *   x = global_avg_pool(backbone_out)     -> [1, 1024]
 *   x = Linear(1024, 1024) + LayerNorm + GELU  -> [1, 1024]
 *   x = Linear(1024, 1019)                 -> [1, 1019]
 *   output = split(x, [507, 512])          -> logits[1,507], features[1,512]
 *
 * @param backbone_out  Backbone output [1, 1024, H, W]
 * @param proj_target_dim  Logit dimension (507)
 * @param proj_feature_dim Feature dimension (512)
 * @return {logits, features} symbolic tensors
 */
inline std::vector<Tensor> buildHead(Tensor backbone_out, int proj_target_dim,
                                     int proj_feature_dim) {
  // Global average pool: [1, C, H, W] -> [1, C, 1, 1]
  // Cascaded single-axis reductions to support NNTrainer's
  // props::ReduceDimension
  LayerHandle gap_h(
    createLayer("reduce_mean", {nntrainer::withKey("name", "head/gap_h"),
                                nntrainer::withKey("axis", 2)}));
  LayerHandle gap_w(
    createLayer("reduce_mean", {nntrainer::withKey("name", "head/gap_w"),
                                nntrainer::withKey("axis", 3)}));
  auto pooled = gap_w(gap_h(backbone_out));

  // Reshape [1, C, 1, 1] -> [1, C, 1, 1] (already correct shape for 1x1 conv)
  // fc1: 1x1 conv (1024 -> 1024) — Linear replacement
  auto fc1 = conv1x1Only("head/mlp_fc1", 1024, 1024, pooled);

  // LayerNorm(1024)
  LayerHandle ln(createLayer("layer_normalization",
                             {nntrainer::withKey("name", "head/mlp_ln"),
                              nntrainer::withKey("axis", 1),
                              nntrainer::withKey("epsilon", 1e-5)}));
  auto ln_out = ln(fc1);

  // GELU
  LayerHandle gelu(
    createLayer("activation", {nntrainer::withKey("name", "head/gelu"),
                               nntrainer::withKey("activation", "gelu")}));
  auto gelu_out = gelu(ln_out);

  // final_layer: 1x1 conv (1024 -> 1019) — Linear replacement
  int out_dim = proj_target_dim + proj_feature_dim; // 1019
  auto final_out = conv1x1Only("head/final_fc", 1024, out_dim, gelu_out);

  // Split: logits [1, 507, 1, 1], features [1, 512, 1, 1]
  auto logits = sliceCh("head/logits", 0, proj_target_dim, final_out);
  auto features = sliceCh("head/features", proj_target_dim, out_dim, final_out);

  return {logits, features};
}

} // namespace fastvit_keyword

#endif // __FASTVIT_KEYWORD_GRAPH_H__
