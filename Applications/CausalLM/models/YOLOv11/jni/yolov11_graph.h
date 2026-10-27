// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   yolov11_graph.h
 * @date   25 June 2026
 * @brief  YOLOv11 graph block builders (inline, header-only).
 *
 * Extracted from main.cpp so the quantization-aware model class
 * (YOLOv11Model) can reuse the same graph construction without
 * duplicating code.
 *
 * Usage:
 *   #include "yolov11_graph.h"
 *   auto cfg = yolov11::ModelConfig::v11m();
 *   bool q40 = (std::getenv("YOLO_CONV_Q40") != nullptr);
 *   Tensor m4, m6;
 *   auto m10 = yolov11::buildBackbone(input, m4, m6, cfg, q40);
 *   auto outputs = yolov11::buildHead(m4, m6, m10, nc, cfg, q40);
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __YOLOV11_GRAPH_H__
#define __YOLOV11_GRAPH_H__

#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <tensor_api.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;

namespace yolov11 {

// When set (non-null), the conv block builders append the layer name of every
// Q4_0-eligible conv filter (out_ch and in_ch*k*k both 32-aligned) to this
// sink. The offline quantizer (main.cpp --quantize mode) uses it to build the
// per-layer dtype map for model->save() without enumerating the compiled
// graph. Default null = no collection. Eligibility here MUST match the
// conv_q40 gate below and quantize_q4_0_conv.py.
inline std::vector<std::string> *&quantConvSink() {
  static std::vector<std::string> *sink = nullptr;
  return sink;
}

// Weight dtype applied to Q-eligible convs when the conv_q40 gate is on. The
// eligibility test and quantization block layout are identical for Q4_0 and
// Q8_0 (both use QK=32 blocks, same 32-alignment rule); only the stored
// precision differs, so the same conv_q40 flag selects "quantize this conv"
// and this accessor selects the precision. main.cpp sets it per preset
// ("Q4_0" for w4a8/w4a16, "Q8_0" for w8a16). Default "Q4_0" preserves prior
// behavior.
inline std::string &quantWeightDtype() {
  static std::string dtype = "Q4_0";
  return dtype;
}

// Channel axis for concat/slice is ALWAYS logical axis 1 (channel), regardless
// of NCHW vs NHWC physical layout: TensorDim keeps dim[] in [N,C,H,W] order and
// getValue/getValue resolve the physical offset per format. So no change is
// needed when YOLO_NHWC flips the graph layout.
inline int chAxis() { return 1; }

/**
 * @brief Model architecture configuration (widths scale with model size).
 *
 * YOLOv11 has 5 backbone stages. The channel widths at each stage scale
 * with the model size (n/s/m/l/x). All other channels in the backbone, FPN
 * head, and C2PSA are derived from these 5 values.
 *
 *   stage:  0    1    2    3    4
 *   v11m:  64   128  256  512  512
 *   v11s:  32    64  128  256  512
 *
 * C2PSA always operates at stage-4 channels (512 for both v11m and v11s).
 * The detect head's pi_ch (input feature channels) for P3/P4/P5 are
 * stages[2], stages[3], stages[4] respectively.
 */
struct ModelConfig {
  int w[5];    ///< backbone widths per stage (0=stem, 1, 2=P3, 3=P4, 4=P5)
  bool c3k[8]; ///< c3k flag per C3k2 block (m2,m4,m6,m8,m13,m16,m19,m22)

  /** Construct from a 5-element width array + 8 c3k flags. */
  constexpr ModelConfig(int w0, int w1, int w2, int w3, int w4, bool c0,
                        bool c1, bool c2, bool c3, bool c4, bool c5, bool c6,
                        bool c7) :
    w{w0, w1, w2, w3, w4}, c3k{c0, c1, c2, c3, c4, c5, c6, c7} {}

  /** v11m preset (default): all C3k2 blocks use C3k. */
  static constexpr ModelConfig v11m() {
    return {64,   128,  256,  512,  512,  true, true,
            true, true, true, true, true, true};
  }

  /** v11s preset: C3k only for backbone m6/m8 and head m22. */
  static constexpr ModelConfig v11s() {
    return {32,   64,   128,   256,   512,   false, false,
            true, true, false, false, false, true};
  }
};

// ===== Primitive graph block builders =====

/**
 * @brief Build a (BN-fused) Conv2d + SiLU sub-graph block.
 *
 * BatchNorm is folded into the convolution at convert time, so this is a
 * single biased conv followed by a standalone SiLU.
 *
 * @param name      Layer-name prefix
 * @param in_ch     Input channels
 * @param out_ch    Output channels
 * @param k         Kernel size (square)
 * @param stride    Stride
 * @param padding   Padding
 * @param input     Input symbolic tensor
 * @param conv_q40  If true, eligible convs get weight_dtype=Q4_0.
 *                  Eligibility: groups==1 && out_ch>1 && out_ch%32==0 &&
 *                  (in_ch*k*k)%32==0 (must match quantize_q4_0_conv.py).
 * @return Output symbolic tensor after conv+SiLU
 */
inline Tensor convBnSilu(const std::string &name, int in_ch, int out_ch, int k,
                         int stride, int padding, Tensor input,
                         bool conv_q40 = false) {
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding),
    // Fuse SiLU into the conv epilogue (no separate Activation layer => one
    // less full-tensor read+write pass per conv). See Conv2DLayer::forwarding.
    nntrainer::withKey("fused_activation", "swish")};
  if (out_ch > 1 && out_ch % 32 == 0 && (in_ch * k * k) % 32 == 0) {
    if (conv_q40)
      conv_props.push_back(
        nntrainer::withKey("weight_dtype", quantWeightDtype()));
    if (quantConvSink() != nullptr)
      quantConvSink()->push_back(name + "/conv");
  }
  LayerHandle conv(createLayer("conv2d", conv_props));
  return conv(input);
}

/**
 * @brief Build a Bottleneck sub-graph (cv1 3x3 + cv2 3x3 + residual add).
 *
 * @param ch        Input/output channels (residual path)
 * @param inner_ch  Inner (bottleneck) channels. For plain Bottleneck
 *                  (c3k=False) this is ch/2 (expansion=0.5). For C3k inner
 *                  bottlenecks this equals ch (expansion=1.0).
 */
inline Tensor bottleneck(const std::string &name, int ch, int inner_ch,
                         Tensor input, bool conv_q40 = false) {
  auto h = convBnSilu(name + "/cv1", ch, inner_ch, 3, 1, 1, input, conv_q40);
  h = convBnSilu(name + "/cv2", inner_ch, ch, 3, 1, 1, h, conv_q40);
  LayerHandle add(
    createLayer("Addition", {nntrainer::withKey("name", name + "/add")}));
  return add({h, input});
}

/**
 * @brief Build a C3k sub-graph.
 */
inline Tensor c3kBlock(const std::string &name, int in_ch, int inner_ch,
                       int out_ch, Tensor input, bool conv_q40 = false) {
  auto inner =
    convBnSilu(name + "/cv1", in_ch, inner_ch, 1, 1, 0, input, conv_q40);
  // C3k inner bottlenecks use expansion=1.0 (inner_ch → inner_ch → inner_ch)
  inner = bottleneck(name + "/inner0", inner_ch, inner_ch, inner, conv_q40);
  inner = bottleneck(name + "/inner1", inner_ch, inner_ch, inner, conv_q40);
  auto skip =
    convBnSilu(name + "/cv2", in_ch, inner_ch, 1, 1, 0, input, conv_q40);

  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", chAxis())}));
  auto concat_out = cat({inner, skip});

  return convBnSilu(name + "/cv3", 2 * inner_ch, out_ch, 1, 1, 0, concat_out,
                    conv_q40);
}

/**
 * @brief Build a C3k2 sub-graph.
 *
 * @param c3k  If true, the inner block is a C3k (cv1+2*bottleneck+cv2+cv3);
 *             if false, it is a plain Bottleneck (cv1+cv2). In ultralytics,
 *             c3k=True when c >= c3k_threshold (typically 256).
 */
inline Tensor c3k2Block(const std::string &name, int in_ch, int out_ch, int c,
                        Tensor input, bool conv_q40 = false, bool c3k = false) {
  auto y = convBnSilu(name + "/cv1", in_ch, 2 * c, 1, 1, 0, input, conv_q40);

  LayerHandle sliceA(
    createLayer("slice", {nntrainer::withKey("name", name + "/slice_a"),
                          nntrainer::withKey("axis", chAxis()),
                          nntrainer::withKey("start_index", 1),
                          nntrainer::withKey("end_index", c + 1)}));
  auto y_a = sliceA(y);

  LayerHandle sliceB(
    createLayer("slice", {nntrainer::withKey("name", name + "/slice_b"),
                          nntrainer::withKey("axis", chAxis()),
                          nntrainer::withKey("start_index", c + 1),
                          nntrainer::withKey("end_index", 2 * c + 1)}));
  auto y_b = sliceB(y);

  Tensor y_c;
  if (c3k) {
    y_c = c3kBlock(name + "/m0", c, c / 2, c, y_b, conv_q40);
  } else {
    // Plain Bottleneck: cv1(3x3) + cv2(3x3) + residual add
    // Expansion=0.5: inner channels = c/2
    y_c = bottleneck(name + "/m0", c, c / 2, y_b, conv_q40);
  }

  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", chAxis())}));
  auto concat_out = cat({y_a, y_b, y_c});

  return convBnSilu(name + "/cv2", 3 * c, out_ch, 1, 1, 0, concat_out,
                    conv_q40);
}

/**
 * @brief Build a (BN-fused) Conv2d sub-graph block (NO activation).
 *
 * @param conv_q40  If true, eligible convs get weight_dtype=Q4_0.
 */
inline Tensor convBnOnly(const std::string &name, int in_ch, int out_ch, int k,
                         int stride, int padding, Tensor input,
                         bool conv_q40 = false) {
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding)};
  if (out_ch > 1 && out_ch % 32 == 0 && (in_ch * k * k) % 32 == 0) {
    if (conv_q40)
      conv_props.push_back(
        nntrainer::withKey("weight_dtype", quantWeightDtype()));
    if (quantConvSink() != nullptr)
      quantConvSink()->push_back(name + "/conv");
  }
  LayerHandle conv(createLayer("conv2d", conv_props));
  return conv(input);
}

/**
 * @brief Build a MaxPool2d layer with explicit padding.
 */
inline Tensor maxPool(const std::string &name, int k, Tensor input) {
  int p = k / 2;
  std::string pad_str = std::to_string(p) + "," + std::to_string(p) + "," +
                        std::to_string(p) + "," + std::to_string(p);
  LayerHandle pool(
    createLayer("pooling2d", {nntrainer::withKey("name", name),
                              nntrainer::withKey("pooling", "max"),
                              nntrainer::withKey("pool_size", {k, k}),
                              nntrainer::withKey("stride", {1, 1}),
                              nntrainer::withKey("padding", pad_str)}));
  return pool(input);
}

/**
 * @brief Build an SPPF sub-graph.
 */
inline Tensor sppfBlock(const std::string &name, int in_ch, Tensor input,
                        bool conv_q40 = false) {
  int half = in_ch / 2;
  auto x = convBnOnly(name + "/cv1", in_ch, half, 1, 1, 0, input, conv_q40);

  auto p1 = maxPool(name + "/pool1", 5, x);
  auto p2 = maxPool(name + "/pool2", 5, p1);
  auto p3 = maxPool(name + "/pool3", 5, p2);

  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", chAxis())}));
  auto concat_out = cat({x, p1, p2, p3});

  return convBnSilu(name + "/cv2", half * 4, in_ch, 1, 1, 0, concat_out,
                    conv_q40);
}

// ===== Detect head graph builders =====

/**
 * @brief 1x1 Conv2d with bias, no BN, no activation (detect output conv).
 *
 * @param conv_q40  If true, eligible 1x1 convs get weight_dtype=Q4_0.
 *                  Note: out_ch for box branch is 64 (divisible by 32) but
 *                  in_ch (64) * 1 * 1 = 64 is also divisible by 32, so it
 *                  can be Q4_0. cls branch out_ch=1 is NOT eligible
 *                  (out_ch>1 check).
 */
inline Tensor convBias1x1(const std::string &name, int out_ch, int in_ch,
                          Tensor input, bool conv_q40 = false) {
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {1, 1}),
    nntrainer::withKey("filters", out_ch), nntrainer::withKey("stride", {1, 1}),
    nntrainer::withKey("padding", 0)};
  if (out_ch > 1 && out_ch % 32 == 0 && (in_ch * 1 * 1) % 32 == 0) {
    if (conv_q40)
      conv_props.push_back(
        nntrainer::withKey("weight_dtype", quantWeightDtype()));
    if (quantConvSink() != nullptr)
      quantConvSink()->push_back(name + "/conv");
  }
  LayerHandle conv(createLayer("conv2d", conv_props));
  return conv(input);
}

/**
 * @brief Depthwise 3x3 (pad 1) + SiLU, BN folded. Always FP32 (never Q4_0).
 */
inline Tensor dwConvBnSilu(const std::string &name, int ch, Tensor input) {
  LayerHandle dw(createLayer(
    "conv2d",
    {nntrainer::withKey("name", name + "/dw"),
     nntrainer::withKey("kernel_size", {3, 3}),
     nntrainer::withKey("filters", ch), nntrainer::withKey("groups", ch),
     nntrainer::withKey("stride", {1, 1}), nntrainer::withKey("padding", 1),
     // Fuse SiLU into the depthwise conv epilogue (no separate Activation
     // layer). See Conv2DLayer::forwarding.
     nntrainer::withKey("fused_activation", "swish")}));
  return dw(input);
}

/**
 * @brief Depthwise 3x3 (pad 1), NO activation. Always FP32 (never Q4_0).
 */
inline Tensor dwConvBnOnly(const std::string &name, int ch, Tensor input) {
  LayerHandle dw(createLayer(
    "conv2d",
    {nntrainer::withKey("name", name + "/dw"),
     nntrainer::withKey("kernel_size", {3, 3}),
     nntrainer::withKey("filters", ch), nntrainer::withKey("groups", ch),
     nntrainer::withKey("stride", {1, 1}), nntrainer::withKey("padding", 1)}));
  return dw(input);
}

/**
 * @brief Build one Detect scale -> raw logits [1, 64+nc, H, W].
 *
 * @param s       name prefix (e.g. "det0")
 * @param pi_ch   input feature channels (w[2] for P3, w[3] for P4, w[4] for P5)
 * @param nc      number of detection classes (cls output channels)
 * @param cls_ch  cls branch intermediate channels (w[2]: 256 for v11m, 128 for
 * v11s)
 * @param in      input feature map
 * @param conv_q40  If true, eligible convs use Q4_0.
 */
inline Tensor detectScale(const std::string &s, int pi_ch, int nc, int cls_ch,
                          Tensor in, bool conv_q40 = false) {
  // box branch (cv2): pi_ch->64->64->64 (1x1 output)
  // Box channels = 4*reg_max = 64, fixed by YOLOv11 architecture.
  auto x = convBnSilu(s + "/cv2_0", pi_ch, 64, 3, 1, 1, in, conv_q40);
  x = convBnSilu(s + "/cv2_1", 64, 64, 3, 1, 1, x, conv_q40);
  auto box = convBias1x1(s + "/cv2_2", 64, 64, x, conv_q40);

  // cls branch (cv3): depthwise-separable x2 then 1x1 (out_ch=nc)
  // cls_ch scales with model size (w[2]: 256 for v11m, 128 for v11s).
  auto c = dwConvBnSilu(s + "/cv3_0_dw", pi_ch, in);
  c = convBnSilu(s + "/cv3_0_pw", pi_ch, cls_ch, 1, 1, 0, c, conv_q40);
  c = dwConvBnSilu(s + "/cv3_1_dw", cls_ch, c);
  c = convBnSilu(s + "/cv3_1_pw", cls_ch, cls_ch, 1, 1, 0, c, conv_q40);
  auto cls = convBias1x1(s + "/cv3_2", nc, cls_ch, c, conv_q40);

  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", s + "/out"),
                           nntrainer::withKey("axis", chAxis())}));
  return cat({box, cls});
}

// ===== Utility builders used by C2PSA and head =====

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

/** @brief Elementwise addition of two tensors. */
inline Tensor addT(const std::string &name, Tensor a, Tensor b) {
  LayerHandle l(createLayer("Addition", {nntrainer::withKey("name", name)}));
  return l({a, b});
}

/** @brief Upsample (nearest, x2). */
inline Tensor upsampleX2(const std::string &name, Tensor in) {
  LayerHandle l(
    createLayer("upsample2d", {nntrainer::withKey("name", name),
                               nntrainer::withKey("upsample", "nearest"),
                               nntrainer::withKey("kernel_size", "2,2")}));
  return l(in);
}

/** @brief Channel-axis concat. */
inline Tensor concatCh(const std::string &name,
                       const std::vector<Tensor> &ins) {
  LayerHandle l(createLayer("concat", {nntrainer::withKey("name", name),
                                       nntrainer::withKey("axis", chAxis())}));
  return l(ins);
}

/**
 * @brief Build the C2PSA block (model.10) from standard layers + psa_attention.
 *
 * C2PSA always operates at w[4] channels (512 for both v11m and v11s).
 * The attention half-width is w[4]/2 = 256.
 *
 * @param n       name prefix
 * @param x       input tensor
 * @param cfg     model configuration (uses w[4])
 * @param conv_q40  If true, eligible group=1 convs use Q4_0.
 */
inline Tensor buildC2PSA(const std::string &n, Tensor x, const ModelConfig &cfg,
                         bool conv_q40 = false) {
  const int W = cfg.w[4];     // 512
  const int H = cfg.w[4] / 2; // 256 (attention half-width)
  auto cv1 = yolov11::convBnSilu(n + "/cv1", W, W, 1, 1, 0, x, conv_q40);
  auto a = sliceCh(n + "/slice_a", 0, H, cv1);
  auto b = sliceCh(n + "/slice_b", H, W, cv1);

  auto qkv = yolov11::convBnOnly(n + "/qkv", H, W, 1, 1, 0, b, conv_q40);

  std::vector<Tensor> v_parts;
  for (int hh = 0; hh < 4; ++hh)
    v_parts.push_back(sliceCh(n + "/slice_v" + std::to_string(hh),
                              hh * (H / 2) + H / 4, hh * (H / 2) + H / 2, qkv));

  LayerHandle vcat(
    createLayer("concat", {nntrainer::withKey("name", n + "/vcat"),
                           nntrainer::withKey("axis", chAxis())}));
  auto v = vcat(v_parts);
  auto pe = yolov11::dwConvBnOnly(n + "/pe", H, v);

  LayerHandle att(
    createLayer("psa_attention", {nntrainer::withKey("name", n + "/attn")}));
  auto attn = att(qkv);
  auto attn_pe = addT(n + "/add_pe", attn, pe);
  auto proj =
    yolov11::convBnOnly(n + "/proj", H, H, 1, 1, 0, attn_pe, conv_q40);
  auto b1 = addT(n + "/res1", b, proj);

  auto ffn0 = yolov11::convBnSilu(n + "/ffn0", H, W, 1, 1, 0, b1, conv_q40);
  auto ffn1 = yolov11::convBnOnly(n + "/ffn1", W, H, 1, 1, 0, ffn0, conv_q40);
  auto b2 = addT(n + "/res2", b1, ffn1);

  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", n + "/cat"),
                           nntrainer::withKey("axis", chAxis())}));
  auto cc = cat({a, b2});
  return yolov11::convBnSilu(n + "/cv2", W, W, 1, 1, 0, cc, conv_q40);
}

// ===== Top-level whole-network builders =====

/**
 * @brief Build the backbone (model.0-10) and return m10 (C2PSA output).
 *
 * Also outputs m4 and m6 (FPN skip connections).
 *
 * @param xIn       Input symbolic tensor [1, 3, imgsz, imgsz]
 * @param m4_out    [out] C3k2 block 4 output (P3 skip)
 * @param m6_out    [out] C3k2 block 6 output (P4 skip)
 * @param cfg       Model configuration (widths per stage)
 * @param conv_q40  If true, eligible convs use Q4_0.
 * @return m10 (C2PSA output, P5 feature)
 */
inline Tensor buildBackbone(Tensor xIn, Tensor &m4_out, Tensor &m6_out,
                            const ModelConfig &cfg, bool conv_q40 = false) {
  auto h = yolov11::convBnSilu("conv0", 3, cfg.w[0], 3, 2, 1, xIn, conv_q40);
  h = yolov11::convBnSilu("conv1", cfg.w[0], cfg.w[1], 3, 2, 1, h, conv_q40);
  h = yolov11::c3k2Block("m2", cfg.w[1], cfg.w[2], cfg.w[0], h, conv_q40,
                         cfg.c3k[0]);
  h = yolov11::convBnSilu("conv3", cfg.w[2], cfg.w[2], 3, 2, 1, h, conv_q40);
  m4_out = yolov11::c3k2Block("m4", cfg.w[2], cfg.w[3], cfg.w[1], h, conv_q40,
                              cfg.c3k[1]);
  h =
    yolov11::convBnSilu("conv5", cfg.w[3], cfg.w[3], 3, 2, 1, m4_out, conv_q40);
  m6_out = yolov11::c3k2Block("m6", cfg.w[3], cfg.w[3], cfg.w[2], h, conv_q40,
                              cfg.c3k[2]);
  h =
    yolov11::convBnSilu("conv7", cfg.w[3], cfg.w[4], 3, 2, 1, m6_out, conv_q40);
  h = yolov11::c3k2Block("m8", cfg.w[4], cfg.w[4], cfg.w[4] / 2, h, conv_q40,
                         cfg.c3k[3]);
  h = yolov11::sppfBlock("m9", cfg.w[4], h, conv_q40);
  return buildC2PSA("m10", h, cfg, conv_q40);
}

/**
 * @brief Build the FPN head (model.11-22) and 3-scale Detect head (model.23).
 *
 * @param m4       C3k2 block 4 output (P3 skip from backbone)
 * @param m6       C3k2 block 6 output (P4 skip from backbone)
 * @param m10      C2PSA output (P5 feature from backbone)
 * @param nc       Number of detection classes
 * @param cfg      Model configuration (widths per stage)
 * @param conv_q40 If true, eligible convs use Q4_0.
 * @return {P3, P4, P5} raw detection logits [1, 64+nc, H, W] each
 */
inline std::vector<Tensor> buildHead(Tensor m4, Tensor m6, Tensor m10, int nc,
                                     const ModelConfig &cfg,
                                     bool conv_q40 = false) {
  // Head channel sizes derived from backbone widths.
  // m13: in = w[4]+w[3], out = w[3], c = w[3]/2
  // m16: in = w[3]+w[2], out = w[2], c = w[2]/2
  // m19: in = w[2]+w[3], out = w[3], c = w[3]/2
  // m22: in = w[3]+w[4], out = w[4], c = w[4]/2
  auto m11 = upsampleX2("m11", m10);
  auto m12 = concatCh("m12", {m11, m6});
  auto m13 = yolov11::c3k2Block("m13", cfg.w[4] + cfg.w[3], cfg.w[3],
                                cfg.w[3] / 2, m12, conv_q40, cfg.c3k[4]);

  auto m14 = upsampleX2("m14", m13);
  auto m15 = concatCh("m15", {m14, m4});
  auto m16 = yolov11::c3k2Block("m16", cfg.w[3] + cfg.w[2], cfg.w[2],
                                cfg.w[2] / 2, m15, conv_q40, cfg.c3k[5]);

  auto m17 =
    yolov11::convBnSilu("m17", cfg.w[2], cfg.w[2], 3, 2, 1, m16, conv_q40);
  auto m18 = concatCh("m18", {m17, m13});
  auto m19 = yolov11::c3k2Block("m19", cfg.w[2] + cfg.w[3], cfg.w[3],
                                cfg.w[3] / 2, m18, conv_q40, cfg.c3k[6]);

  auto m20 =
    yolov11::convBnSilu("m20", cfg.w[3], cfg.w[3], 3, 2, 1, m19, conv_q40);
  auto m21 = concatCh("m21", {m20, m10});
  auto m22 = yolov11::c3k2Block("m22", cfg.w[3] + cfg.w[4], cfg.w[4],
                                cfg.w[4] / 2, m21, conv_q40, cfg.c3k[7]);

  // Detect head: pi_ch = w[2] for P3, w[3] for P4, w[4] for P5
  // cls_ch = w[2] (256 for v11m, 128 for v11s)
  return {detectScale("det0", cfg.w[2], nc, cfg.w[2], m16, conv_q40),
          detectScale("det1", cfg.w[3], nc, cfg.w[2], m19, conv_q40),
          detectScale("det2", cfg.w[4], nc, cfg.w[2], m22, conv_q40)};
}

} // namespace yolov11

#endif // __YOLOV11_GRAPH_H__
