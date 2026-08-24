// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   yolov7_tiny_graph.h
 * @date   24 August 2026
 * @brief  YOLOv7-Tiny Object Detection graph builders (inline, header-only).
 *         Backbone : YOLOv7-tiny CSP (widen 1.0 -> stem 32 ... deepest 512)
 *         Neck     : FPN/PAN (SPPCSPCTiny + up x2 + down x2, 3 outputs)
 *         Head     : 3-scale Detect head (nc = 4)
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#ifndef __YOLOV7_TINY_GRAPH_H__
#define __YOLOV7_TINY_GRAPH_H__

#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <tensor_api.h>
#include <util_func.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;

namespace yolov7_tiny {

inline int chAxis() { return 1; }

inline std::string &quantWeightDtype() {
  static std::string dtype = "Q8_0";
  return dtype;
}

inline std::vector<std::string> &quantizableConvs() {
  static std::vector<std::string> names;
  return names;
}

inline bool convQuantEligible(int in_ch, int out_ch, int k) {
  return out_ch % 32 == 0 && (in_ch * k * k) % 32 == 0;
}

// Conv2D (biased)
inline Tensor conv(const std::string &name, int in_ch, int out_ch, int k,
                   int stride, int padding, Tensor input, bool conv_q = false) {
  std::vector<std::string> props = {
    nntrainer::withKey("name", name),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding)};
  const bool eligible = convQuantEligible(in_ch, out_ch, k);
  if (eligible)
    quantizableConvs().push_back(name);
  if (conv_q && eligible)
    props.push_back(nntrainer::withKey("weight_dtype", quantWeightDtype()));
  LayerHandle c(createLayer("conv2d", props));
  return c(input);
}

// Conv + BN + SiLU (Swish) Fused equivalent
inline Tensor convBnLeaky(const std::string &name, int in_ch, int out_ch, int k,
                          int stride, int padding, Tensor input,
                          bool conv_q = false) {
  auto x = conv(name, in_ch, out_ch, k, stride, padding, input, conv_q);
  LayerHandle act(createLayer(
    "activation", {nntrainer::withKey("name", name + "/silu"),
                   nntrainer::withKey("activation", "swish")}));
  return act(x);
}


inline Tensor concat(const std::string &name, std::vector<Tensor> xs) {
  LayerHandle c(createLayer("concat", {nntrainer::withKey("name", name),
                                       nntrainer::withKey("axis", chAxis())}));
  return c(xs);
}

inline Tensor maxpool(const std::string &name, int k, int stride, int pad,
                      Tensor input) {
  LayerHandle mp(createLayer(
    "pooling2d", {nntrainer::withKey("name", name),
                  nntrainer::withKey("pooling", "max"),
                  nntrainer::withKey("pool_size", {k, k}),
                  nntrainer::withKey("stride", {stride, stride}),
                  nntrainer::withKey("padding", pad)}));
  return mp(input);
}

// ELAN Block for YOLOv7-Tiny
inline Tensor elan(const std::string &name, int c_in, int c_out,
                   int c_bottleneck, int n_blocks, Tensor input,
                   bool conv_q = false) {
  auto x1 = convBnLeaky(name + ".conv1", c_in, c_bottleneck, 1, 1, 0, input,
                        conv_q);
  auto x = convBnLeaky(name + ".conv2", c_in, c_bottleneck, 1, 1, 0, input,
                       conv_q);
  std::vector<Tensor> cbs;
  auto prev = x;
  for (int j = 0; j < n_blocks; ++j) {
    prev = convBnLeaky(name + ".conv_blocks." + std::to_string(j), c_bottleneck,
                       c_bottleneck, 3, 1, 1, prev, conv_q);
    cbs.push_back(prev);
  }
  std::vector<Tensor> to_cat;
  for (int j = n_blocks - 1; j >= 0; --j)
    to_cat.push_back(cbs[j]);
  to_cat.push_back(x);
  to_cat.push_back(x1);
  auto cat = concat(name + "/cat", to_cat);
  int last_in = (n_blocks + 2) * c_bottleneck;
  return convBnLeaky(name + ".last_conv", last_in, c_out, 1, 1, 0, cat, conv_q);
}

// SPPCSPCTiny
inline Tensor sppcspc(const std::string &name, int c_in, int c_out,
                      Tensor input, bool conv_q = false) {
  int c_hidden = c_out;
  auto cv1 = convBnLeaky(name + ".cv1", c_in, c_hidden, 1, 1, 0, input, conv_q);
  auto cv2 = convBnLeaky(name + ".cv2", c_in, c_hidden, 1, 1, 0, input, conv_q);
  auto m5 = maxpool(name + "/m_5", 5, 1, 2, cv1);
  auto m9 = maxpool(name + "/m_9", 9, 1, 4, cv1);
  auto m13 = maxpool(name + "/m_13", 13, 1, 6, cv1);
  // PyTorch: cat([x1] + [m(x1) for m in self.m], 1) = [cv1, m5, m9, m13]
  auto y1 = convBnLeaky(name + ".cv3", 4 * c_hidden, c_hidden, 1, 1, 0,
                        concat(name + "/cat3", {cv1, m5, m9, m13}), conv_q);
  // PyTorch: cat((y1, y2), dim=1)
  return convBnLeaky(name + ".cv4", 2 * c_hidden, c_out, 1, 1, 0,
                     concat(name + "/cat4", {y1, cv2}), conv_q);
}


// UpSampleBlock
inline Tensor upSample(const std::string &name, int c_in, int c_out,
                       int route_ch, Tensor input, Tensor route,
                       bool conv_q = false) {
  int c_hidden = c_out / 2;
  auto c1 = convBnLeaky(name + ".base.conv1", c_in, c_hidden, 1, 1, 0, input, conv_q);
  LayerHandle up(
    createLayer("upsample2d", {nntrainer::withKey("name", name + "/up"),
                               nntrainer::withKey("upsample", "nearest"),
                               nntrainer::withKey("kernel_size", "2,2")}));
  auto x_up = up(c1);
  auto c2 =
    convBnLeaky(name + ".base.conv2", route_ch, c_hidden, 1, 1, 0, route, conv_q);
  // PyTorch: cat((x1, x), dim=1) where x1=conv2(route), x=up(conv1(input))
  return concat(name + "/cat", {c2, x_up});
}


// DownSampleBlock
inline Tensor downSample(const std::string &name, int c_in, int c_out,
                         int route_ch, Tensor input, Tensor route,
                         bool conv_q = false) {
  auto c1 = convBnLeaky(name + ".base.conv", c_in, c_out, 3, 2, 1, input, conv_q);
  return concat(name + "/cat", {c1, route});
}

// ---- backbone -------------------------------------------------------------
inline std::vector<Tensor> buildBackbone(Tensor xIn, bool conv_q = false) {
  const std::string p = "backbone.backbone.blocks";
  std::vector<Tensor> nodes;

  // block 0: stem 3 -> 32
  auto b0 = convBnLeaky(p + ".0.0", 3, 32, 3, 2, 1, xIn, conv_q);
  nodes.push_back(b0);

  // block 1: DownConv 32 -> 64 (s2), ELAN 64 -> 64
  auto b1_base = convBnLeaky(p + ".1.base", 32, 64, 3, 2, 1, b0, conv_q);
  auto b1 = elan(p + ".1.elan", 64, 64, 32, 2, b1_base, conv_q);
  nodes.push_back(b1);

  // blocks 2-4: maxpool (s2) + ELAN
  int chs[3] = {64, 128, 256};
  int outs[3] = {128, 256, 512};
  auto prev = b1;
  for (int i = 2; i <= 4; ++i) {
    auto mp = maxpool(p + "." + std::to_string(i) + "/mp", 2, 2, 0, prev);
    auto e = elan(p + "." + std::to_string(i) + ".elan", chs[i-2], outs[i-2], chs[i-2], 2, mp,
                  conv_q);
    nodes.push_back(e);
    prev = e;
  }
  return nodes;
}

// ---- neck -----------------------------------------------------------------
inline std::vector<Tensor> buildNeck(const std::string &prefix,
                                     const std::vector<Tensor> &nodes,
                                     bool conv_q = false) {
  auto b2 = nodes[2]; // 128, stride 8
  auto b3 = nodes[3]; // 256, stride 16
  auto b4 = nodes[4]; // 512, stride 32

  auto spp = sppcspc(prefix + ".spp", 512, 256, b4, conv_q); // 256

  // feature_up.0: up(spp) + route b3(256) -> elan -> 128
  auto up0_cat = upSample(prefix + ".feature_up.0", 256, 256, 256, spp, b3, conv_q);
  auto up0 = elan(prefix + ".feature_up.0.elan", 256, 128, 64, 2, up0_cat, conv_q);

  // feature_up.1: up(up0) + route b2(128) -> elan -> 64
  auto up1_cat = upSample(prefix + ".feature_up.1", 128, 128, 128, up0, b2, conv_q);
  auto up1 = elan(prefix + ".feature_up.1.elan", 128, 64, 32, 2, up1_cat, conv_q);

  // feature_down.0: down(up1) + route up0(128) -> elan -> 128
  auto down0_cat = downSample(prefix + ".feature_down.0", 64, 128, 128, up1, up0, conv_q);
  auto down0 = elan(prefix + ".feature_down.0.elan", 256, 128, 64, 2, down0_cat, conv_q);

  // feature_down.1: down(down0) + route spp(256) -> elan -> 256
  auto down1_cat = downSample(prefix + ".feature_down.1", 128, 256, 256, down0, spp, conv_q);
  auto down1 = elan(prefix + ".feature_down.1.elan", 512, 256, 128, 2, down1_cat, conv_q);

  // ends layers (stride 8, 16, 32 final convs)
  auto p3_end = convBnLeaky(prefix + ".ends.0", 64, 128, 3, 1, 1, up1, conv_q);
  auto p4_end = convBnLeaky(prefix + ".ends.1", 128, 256, 3, 1, 1, down0, conv_q);
  auto p5_end = convBnLeaky(prefix + ".ends.2", 256, 512, 3, 1, 1, down1, conv_q);

  return {p3_end, p4_end, p5_end};
}

// ---- head (3-scale Detect head) -------------------------------------------
inline std::vector<Tensor> buildHead(const std::vector<Tensor> &neck_nodes,
                                     int nc, bool conv_q = false) {
  const std::string p = "head";
  int na = 3;
  int out_ch = na * (nc + 5);

  auto det0 = conv(p + ".m.0", 128, out_ch, 1, 1, 0, neck_nodes[0], conv_q);
  auto det1 = conv(p + ".m.1", 256, out_ch, 1, 1, 0, neck_nodes[1], conv_q);
  auto det2 = conv(p + ".m.2", 512, out_ch, 1, 1, 0, neck_nodes[2], conv_q);

  return {det0, det1, det2};
}

// ---- full network ---------------------------------------------------------
inline std::vector<Tensor> buildBackboneNeckHead(Tensor xIn, int nc, bool conv_q = false) {
  auto backbone = buildBackbone(xIn, conv_q);
  auto neck = buildNeck("backbone.features", backbone, conv_q);
  auto head = buildHead(neck, nc, conv_q);
  return head;
}

} // namespace yolov7_tiny

#endif // __YOLOV7_TINY_GRAPH_H__
