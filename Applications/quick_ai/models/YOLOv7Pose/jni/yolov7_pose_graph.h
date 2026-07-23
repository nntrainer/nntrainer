// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   yolov7_pose_graph.h
 * @date   14 July 2026
 * @brief  YOLOv7ReIDtiny pose+ReID graph builders (inline, header-only).
 *
 *         Backbone : YOLOv7-tiny CSP (widen 1.5 -> stem 48 ... deepest 768)
 *         Neck     : two independent FPNs (pose "features", reid
 *                    "features_feat"), each SPPCSPCTiny + up x2 + down x2,
 *                    single stride-32 end (768ch @ 10x10).
 *         Head     : RTMCC SimCC pose head (-> [1, 2*NKPT, SIMCC_BINS]) and
 *                    a ReID embedding head (-> [1, EMBED_DIM]).
 *
 *         Weight names mirror the (fused) PyTorch module hierarchy of
 *         YOLOv7ReIDtiny so the safetensors produced by weight_converter.py
 *         bind by name. See res/yolov7_pose/ for the reference model.
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#ifndef __YOLOV7_POSE_GRAPH_H__
#define __YOLOV7_POSE_GRAPH_H__

#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <tensor_api.h>
#include <util_func.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;

namespace yolov7_pose {

// ---- model hyper-parameters (widen_factor = 1.5, img 320, nkpt 87) --------
inline constexpr int NKPT = 87;
inline constexpr int SIMCC_BINS = 640;   // img_size * simcc_split_ratio(2.0)
inline constexpr int GAU_HIDDEN = 256;   // token feature dim D
inline constexpr int GAU_S = 128;        // key dim s
inline constexpr int GAU_E = 512;        // expansion e (= D * 2)
inline constexpr int EMBED_DIM = 128;    // reid embedding
inline constexpr int FEATMAP = 10;       // 320 / 32
inline constexpr int C_STEM = 48;        // round(32 * 1.5)

inline int chAxis() { return 1; }

inline std::string &quantWeightDtype() {
  static std::string dtype = "Q8_0";
  return dtype;
}

// Names of conv layers whose weights are block-quantizable (out_ch and CRS both
// divisible by 32). Populated during graph construction; consumed by
// nntr_quantize (Yolov7Pose::getQuantizableLayerNames).
inline std::vector<std::string> &quantizableConvs() {
  static std::vector<std::string> names;
  return names;
}

// True when a conv weight forms a real matmul matrix divisible by the block
// size (Q8_0 block = 32). BOTH out_ch and CRS = in_ch*k*k must be multiples of
// 32, and both constraints are real:
//   - the quantized filter is stored as TensorDim(1,1,CRS,out_ch) and Q8_0_Tensor
//     block-quantizes along width == out_ch, so out_ch % 32 is enforced by the
//     container (a width=48 tensor throws at construction);
//   - the indirect GEMM derives nb = CRS/32 and drops any tail, so CRS % 32.
// (An earlier attempt to relax out_ch to %4 crashed on out_ch=48 for exactly the
// first reason -- widths like the ELAN bottleneck 48 need out_ch padding, which
// the plain eligibility guard cannot express.)
inline bool convQuantEligible(int in_ch, int out_ch, int k) {
  return out_ch % 32 == 0 && (in_ch * k * k) % 32 == 0;
}

// Conv2D (biased) + optional swish (SiLU) fused activation.
// @a name is the PyTorch Conv module path (e.g.
// "backbone.backbone.blocks.1.elan.conv1"); the nntrainer conv layer takes the
// same name so weight_converter.py maps "<name>.conv.weight" -> "<name>:filter".
inline Tensor conv(const std::string &name, int in_ch, int out_ch, int k,
                   int stride, int padding, Tensor input, bool act,
                   bool conv_q = false) {
  std::vector<std::string> props = {
    nntrainer::withKey("name", name),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding)};
  if (act)
    props.push_back(nntrainer::withKey("fused_activation", "swish"));
  const bool eligible = convQuantEligible(in_ch, out_ch, k);
  if (eligible)
    quantizableConvs().push_back(name);
  // Quantize only weights that form a real matmul matrix divisible by the
  // block size (Q8_0 block = 32).
  if (conv_q && eligible)
    props.push_back(nntrainer::withKey("weight_dtype", quantWeightDtype()));
  LayerHandle c(createLayer("conv2d", props));
  return c(input);
}

inline Tensor convBnSilu(const std::string &name, int in_ch, int out_ch, int k,
                         int stride, int padding, Tensor input,
                         bool conv_q = false) {
  return conv(name, in_ch, out_ch, k, stride, padding, input, true, conv_q);
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

// ELAN_S0: two 1x1 stems + N sequential 3x3 blocks, concat(reversed) -> 1x1.
// cat order (PyTorch) is [cb_{n-1}, .., cb_0, conv2, conv1] = 4*c_bottleneck.
inline Tensor elan(const std::string &name, int c_in, int c_out,
                   int c_bottleneck, int n_blocks, Tensor input,
                   bool conv_q = false) {
  auto x1 = convBnSilu(name + ".conv1", c_in, c_bottleneck, 1, 1, 0, input,
                       conv_q);
  auto x = convBnSilu(name + ".conv2", c_in, c_bottleneck, 1, 1, 0, input,
                      conv_q);
  std::vector<Tensor> cbs;
  auto prev = x;
  for (int j = 0; j < n_blocks; ++j) {
    prev = convBnSilu(name + ".conv_blocks." + std::to_string(j), c_bottleneck,
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
  return convBnSilu(name + ".last_conv", last_in, c_out, 1, 1, 0, cat, conv_q);
}

// SPPCSPCTiny: cv1/cv2 stems, cv3 fuses maxpool pyramid, cv4 fuses.
inline Tensor sppcspc(const std::string &name, int c_in, int c_out,
                      Tensor input, bool conv_q = false) {
  int c_hidden = c_out; // int(2 * c_out * 0.5)
  auto cv1 = convBnSilu(name + ".cv1", c_in, c_hidden, 1, 1, 0, input, conv_q);
  auto cv2 = convBnSilu(name + ".cv2", c_in, c_hidden, 1, 1, 0, input, conv_q);
  auto m5 = maxpool(name + "/m_5", 5, 1, 2, cv1);
  auto m9 = maxpool(name + "/m_9", 9, 1, 4, cv1);
  auto m13 = maxpool(name + "/m_13", 13, 1, 6, cv1);
  auto y1 = convBnSilu(name + ".cv3", 4 * c_hidden, c_hidden, 1, 1, 0,
                       concat(name + "/cat3", {cv1, m5, m9, m13}), conv_q);
  return convBnSilu(name + ".cv4", 2 * c_hidden, c_out, 1, 1, 0,
                    concat(name + "/cat4", {y1, cv2}), conv_q);
}

// UpSampleBlock: conv1 -> nearest upsample, conv2 on route, concat.
inline Tensor upSample(const std::string &name, int c_in, int c_out,
                       int route_ch, Tensor input, Tensor route,
                       bool conv_q = false) {
  int c_hidden = c_out / 2;
  auto c1 = convBnSilu(name + ".conv1", c_in, c_hidden, 1, 1, 0, input, conv_q);
  LayerHandle up(
    createLayer("upsample2d", {nntrainer::withKey("name", name + "/up"),
                               nntrainer::withKey("upsample", "nearest"),
                               nntrainer::withKey("kernel_size", "2,2")}));
  auto x_up = up(c1);
  auto c2 =
    convBnSilu(name + ".conv2", route_ch, c_hidden, 1, 1, 0, route, conv_q);
  return concat(name + "/cat", {c2, x_up});
}

// DownSampleRouteS0: conv (stride2) then concat with route.
inline Tensor downRoute(const std::string &name, int c_in, int c_out,
                        int route_ch, Tensor input, Tensor route,
                        bool conv_q = false) {
  int c_hidden = c_out / 2;
  auto d = convBnSilu(name + ".conv", c_in, c_hidden, 3, 2, 1, input, conv_q);
  return concat(name + "/cat", {d, route});
}

// One BlockS2 = sample block ("base") + ELAN. ELAN bottleneck = c_elan_out/2.
inline Tensor blockUp(const std::string &name, int c_in, int c_sample_out,
                      int c_elan_out, int route_ch, Tensor input, Tensor route,
                      bool conv_q = false) {
  auto s = upSample(name + ".base", c_in, c_sample_out, route_ch, input, route,
                    conv_q);
  return elan(name + ".elan", c_sample_out, c_elan_out, c_elan_out / 2, 2, s,
              conv_q);
}

inline Tensor blockDown(const std::string &name, int c_in, int c_sample_out,
                        int c_elan_out, int route_ch, Tensor input,
                        Tensor route, bool conv_q = false) {
  auto s = downRoute(name + ".base", c_in, c_sample_out, route_ch, input, route,
                     conv_q);
  return elan(name + ".elan", c_sample_out, c_elan_out, c_elan_out / 2, 2, s,
              conv_q);
}

// ---- backbone -------------------------------------------------------------
// YOLOv7-tiny CSP (widen 1.5). Stage nodes: [stem 48, 96, 192, 384, 768].
// block 1 downsamples with a strided conv ("base"); blocks 2-4 downsample with
// a parameter-free maxpool and the ELAN doubles the channels.
inline std::vector<Tensor> buildBackbone(Tensor xIn, bool conv_q = false) {
  const std::string p = "backbone.backbone.blocks";
  std::vector<Tensor> nodes;

  // block 0: stem (StemS0, single 3x3 s2)  3 -> 48
  auto b0 = convBnSilu(p + ".0.0", 3, 48, 3, 2, 1, xIn, conv_q);
  nodes.push_back(b0);

  // block 1: base DownConv 48 -> 96 (s2), ELAN 96 -> 96 (bottleneck 48)
  auto b1_base = convBnSilu(p + ".1.base", 48, 96, 3, 2, 1, b0, conv_q);
  auto b1 = elan(p + ".1.elan", 96, 96, 48, 2, b1_base, conv_q);
  nodes.push_back(b1);

  // blocks 2-4: maxpool (s2) + ELAN that doubles the channels.
  int ch = 96;
  auto prev = b1;
  for (int i = 2; i <= 4; ++i) {
    int out = ch * 2;
    auto mp = maxpool(p + "." + std::to_string(i) + "/mp", 2, 2, 0, prev);
    auto e = elan(p + "." + std::to_string(i) + ".elan", ch, out, ch, 2, mp,
                  conv_q);
    nodes.push_back(e);
    prev = e;
    ch = out;
  }
  return nodes;
}

// ---- neck (one FPN) -------------------------------------------------------
// prefix is "backbone.features" (pose) or "backbone.features_feat" (reid).
// Node channels: spp 384, up0 192, up1 96, down0 192, down1 384, end 768.
// Returns the single stride-32 end [768ch, 10x10].
inline Tensor buildNeck(const std::string &prefix,
                        const std::vector<Tensor> &nodes, bool conv_q = false) {
  auto b2 = nodes[2]; // 192, stride 8
  auto b3 = nodes[3]; // 384, stride 16
  auto b4 = nodes[4]; // 768, stride 32

  auto spp = sppcspc(prefix + ".spp", 768, 384, b4, conv_q); // 384

  // feature_up.0: up(spp) + route b3(384) -> base 384 ; elan -> 192
  auto up0 = blockUp(prefix + ".feature_up.0", 384, 384, 192, 384, spp, b3,
                     conv_q);
  // feature_up.1: up(up0) + route b2(192) -> base 192 ; elan -> 96
  auto up1 = blockUp(prefix + ".feature_up.1", 192, 192, 96, 192, up0, b2,
                     conv_q);
  // feature_down.0: down(up1) + route up0(192) -> base 384 ; elan -> 192
  auto down0 = blockDown(prefix + ".feature_down.0", 96, 384, 192, 192, up1,
                         up0, conv_q);
  // feature_down.1: down(down0) + route spp(384) -> base 768 ; elan -> 384
  auto down1 = blockDown(prefix + ".feature_down.1", 192, 768, 384, 384, down0,
                         spp, conv_q);
  // ends.0: 3x3 conv 384 -> 768
  return convBnSilu(prefix + ".ends.0", 384, 768, 3, 1, 1, down1, conv_q);
}

// ---- RTMCC SimCC pose head ------------------------------------------------
// Input: [1, 768, 10, 10]. Output: [1, 1, 2*NKPT, SIMCC_BINS] (cls_x || cls_y).
// The whole head after the final conv is one layout-independent custom layer,
// so the graph stays channel-last (NHWC) end-to-end with no transposes.
inline Tensor buildPoseHead(Tensor feat, bool conv_q = false) {
  const std::string p = "head";
  // final_layer: 7x7 conv 768 -> NKPT, pad 3  => [1, NKPT, 10, 10]
  auto fc = conv(p + ".final_layer", 16 * C_STEM, NKPT, 7, 1, 3, feat,
                 /*act=*/false, conv_q);
  LayerHandle head(createLayer(
    "rtmcc_head", {nntrainer::withKey("name", p),
                   nntrainer::withKey("gau_expansion", GAU_E),
                   nntrainer::withKey("gau_key_dim", GAU_S),
                   nntrainer::withKey("gau_hidden", GAU_HIDDEN),
                   nntrainer::withKey("simcc_bins", SIMCC_BINS),
                   nntrainer::withKey("epsilon", "0.00001")}));
  return head(fc);
}

// ---- ReID embedding head --------------------------------------------------
// Input: [1, 768, 10, 10]. Output: [1, EMBED_DIM].
inline Tensor buildReidHead(Tensor feat) {
  const std::string p = "head_feat";
  LayerHandle pool(createLayer(
    "pooling2d", {nntrainer::withKey("name", p + "/gap"),
                  nntrainer::withKey("pooling", "global_average")}));
  auto pooled = pool(feat); // [1, 768, 1, 1]
  // move the 768 channels to the last (width) axis for the FC.
  LayerHandle rs(createLayer(
    "reshape", {nntrainer::withKey("name", p + "/flatten"),
                nntrainer::withKey("target_shape",
                                   "1:1:" + std::to_string(16 * C_STEM))}));
  auto flat = rs(pooled); // [1, 1, 1, 768]
  LayerHandle fc(createLayer(
    "fully_connected", {nntrainer::withKey("name", p + ".fc"),
                        nntrainer::withKey("unit", EMBED_DIM),
                        nntrainer::withKey("disable_bias", "false")}));
  return fc(flat); // [1, 1, 1, EMBED_DIM]
}

} // namespace yolov7_pose

#endif // __YOLOV7_POSE_GRAPH_H__
