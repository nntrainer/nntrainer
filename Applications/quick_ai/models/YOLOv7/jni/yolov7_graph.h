// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   yolov7_graph.h
 * @date   13 July 2026
 * @brief  YOLOv7 box-detector graph block builders (inline, header-only).
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#ifndef __YOLOV7_GRAPH_H__
#define __YOLOV7_GRAPH_H__

#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <tensor_api.h>
#include <util_func.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;

namespace yolov7 {

inline int chAxis() { return 1; }

inline std::string &quantWeightDtype() {
  static std::string dtype = "Q4_0";
  return dtype;
}

// Helper for biased Conv2D + SiLU (Swish) activation
inline Tensor convBnSilu(const std::string &name, int in_ch, int out_ch, int k,
                         int stride, int padding, Tensor input,
                         bool conv_q40 = false) {
  std::vector<std::string> props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding),
    nntrainer::withKey("fused_activation", "swish")};
  if (out_ch > 1 && out_ch % 32 == 0 && (in_ch * k * k) % 32 == 0) {
    if (conv_q40) {
      props.push_back(nntrainer::withKey("weight_dtype", quantWeightDtype()));
    }
  }
  LayerHandle conv(createLayer("conv2d", props));
  return conv(input);
}

// Helper for biased Conv2D without activation
inline Tensor convNoAct(const std::string &name, int in_ch, int out_ch, int k,
                        int stride, int padding, Tensor input,
                        bool conv_q40 = false) {
  std::vector<std::string> props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding)};
  if (out_ch > 1 && out_ch % 32 == 0 && (in_ch * k * k) % 32 == 0) {
    if (conv_q40) {
      props.push_back(nntrainer::withKey("weight_dtype", quantWeightDtype()));
    }
  }
  LayerHandle conv(createLayer("conv2d", props));
  return conv(input);
}

// Stem block: StemS1 (outputs half resolution)
inline Tensor buildStemS1(const std::string &name, Tensor input,
                          bool conv_q40 = false) {
  auto x = convBnSilu(name + "/0", 3, 32, 3, 1, 1, input, conv_q40);
  x = convBnSilu(name + "/1", 32, 64, 3, 2, 1, x, conv_q40);
  return convBnSilu(name + "/2", 64, 64, 3, 1, 1, x, conv_q40);
}

// DownConv helper
inline Tensor downConv(const std::string &name, int in_ch, int out_ch,
                       Tensor input, bool conv_q40 = false) {
  // convBnSilu already appends "/conv" to the name it is given, so pass `name`
  // directly. Passing `name + "/conv"` here would double it to ".../conv/conv",
  // which no longer matches the safetensors weight key ".../conv:filter" and the
  // weight loader silently skips it (random-init conv -> divergent backbone).
  return convBnSilu(name, in_ch, out_ch, 3, 2, 1, input, conv_q40);
}

// DownSampleRouteS1 helper
inline Tensor downSampleRouteS1(const std::string &name, int in_ch, int out_ch,
                                Tensor input, Tensor route = Tensor(),
                                bool conv_q40 = false) {
  int c_hidden = route.empty() ? (out_ch / 2) : (out_ch / 4);

  // x1 branch: maxpool + conv1
  LayerHandle mp(
    createLayer("pooling2d", {nntrainer::withKey("name", name + "/mp"),
                              nntrainer::withKey("pooling", "max"),
                              nntrainer::withKey("pool_size", {2, 2}),
                              nntrainer::withKey("stride", {2, 2}),
                              nntrainer::withKey("padding", 0)}));
  auto x1 = mp(input);
  x1 = convBnSilu(name + "/conv1", in_ch, c_hidden, 1, 1, 0, x1, conv_q40);

  // x branch: conv2 + conv3
  auto x =
    convBnSilu(name + "/conv2", in_ch, c_hidden, 1, 1, 0, input, conv_q40);
  x = convBnSilu(name + "/conv3", c_hidden, c_hidden, 3, 2, 1, x, conv_q40);

  // Concat
  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", chAxis())}));
  if (!route.empty()) {
    return cat({x, x1, route});
  } else {
    return cat({x, x1});
  }
}

// ELAN Block helper: ELAN_S1 (or ELAN_S2)
inline Tensor elanBlock(const std::string &name, int in_ch, int out_ch,
                        int c_bottleneck, int n_blocks, Tensor input,
                        bool is_elan_s1 = true, bool conv_q40 = false) {
  auto x1 =
    convBnSilu(name + "/conv1", in_ch, c_bottleneck, 1, 1, 0, input, conv_q40);
  auto x =
    convBnSilu(name + "/conv2", in_ch, c_bottleneck, 1, 1, 0, input, conv_q40);

  int c_out_block = is_elan_s1 ? c_bottleneck : (c_bottleneck / 2);

  std::vector<Tensor> cb;
  auto prev = x;
  int curr_ch = c_bottleneck;
  for (int j = 0; j < n_blocks; ++j) {
    prev = convBnSilu(name + "/conv_blocks/" + std::to_string(j), curr_ch,
                      c_out_block, 3, 1, 1, prev, conv_q40);
    cb.push_back(prev);
    curr_ch = c_out_block;
  }

  std::vector<Tensor> to_cat;
  if (is_elan_s1) {
    // ELAN_S1 uses cb1 and cb3
    to_cat = {cb[3], cb[1], x, x1};
  } else {
    // ELAN_S2 uses cb3, cb2, cb1, cb0
    to_cat = {cb[3], cb[2], cb[1], cb[0], x, x1};
  }

  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", chAxis())}));
  auto concatenated = cat(to_cat);
  int last_in_ch = 4 * c_bottleneck;
  return convBnSilu(name + "/last_conv", last_in_ch, out_ch, 1, 1, 0,
                    concatenated, conv_q40);
}

// SPPCSPC Block helper
inline Tensor sppcspcBlock(const std::string &name, int in_ch, int out_ch,
                           Tensor input, bool conv_q40 = false) {
  int c_hidden = out_ch; // 512
  auto cv1 =
    convBnSilu(name + "/cv1", in_ch, c_hidden, 1, 1, 0, input, conv_q40);
  auto cv2 =
    convBnSilu(name + "/cv2", in_ch, c_hidden, 1, 1, 0, input, conv_q40);
  auto cv3 =
    convBnSilu(name + "/cv3", c_hidden, c_hidden, 3, 1, 1, cv1, conv_q40);
  auto cv4 =
    convBnSilu(name + "/cv4", c_hidden, c_hidden, 1, 1, 0, cv3, conv_q40);

  // maxpool branches with k=5, 9, 13
  auto mp_pool = [&](int k) {
    LayerHandle mp(createLayer(
      "pooling2d",
      {nntrainer::withKey("name", name + "/mp_" + std::to_string(k)),
       nntrainer::withKey("pooling", "max"),
       nntrainer::withKey("pool_size", {k, k}),
       nntrainer::withKey("stride", {1, 1}),
       nntrainer::withKey("padding", k / 2)}));
    return mp(cv4);
  };
  auto m5 = mp_pool(5);
  auto m9 = mp_pool(9);
  auto m13 = mp_pool(13);

  LayerHandle cat5(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat5"),
                           nntrainer::withKey("axis", chAxis())}));
  auto concatenated5 = cat5({cv4, m5, m9, m13});
  auto cv5 = convBnSilu(name + "/cv5", 4 * c_hidden, c_hidden, 1, 1, 0,
                        concatenated5, conv_q40);
  auto cv6 =
    convBnSilu(name + "/cv6", c_hidden, c_hidden, 3, 1, 1, cv5, conv_q40);

  LayerHandle cat7(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat7"),
                           nntrainer::withKey("axis", chAxis())}));
  auto concatenated7 = cat7({cv6, cv2});
  return convBnSilu(name + "/cv7", 2 * c_hidden, out_ch, 1, 1, 0, concatenated7,
                    conv_q40);
}

// UpSampleBlock helper
inline Tensor upSampleBlock(const std::string &name, int in_ch, int out_ch,
                            int route_ch, Tensor input, Tensor route,
                            bool conv_q40 = false) {
  int c_hidden = out_ch / 2;
  auto conv1 =
    convBnSilu(name + "/conv1", in_ch, c_hidden, 1, 1, 0, input, conv_q40);

  LayerHandle up(
    createLayer("upsample2d", {nntrainer::withKey("name", name + "/up"),
                               nntrainer::withKey("upsample", "nearest"),
                               nntrainer::withKey("kernel_size", "2,2")}));
  auto x_up = up(conv1);

  auto conv2 =
    convBnSilu(name + "/conv2", route_ch, c_hidden, 1, 1, 0, route, conv_q40);

  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", chAxis())}));
  return cat({conv2, x_up});
}

// Build whole YOLOv7 backbone and return intermediate nodes
inline Tensor buildBackbone(Tensor xIn, std::vector<Tensor> &features,
                            bool conv_q40 = false) {
  // Block 0: Stem
  auto b0 = buildStemS1("backbone/blocks/0", xIn, conv_q40);
  features.push_back(b0);

  // Block 1:
  auto b1_base = downConv("backbone/blocks/1/base", 64, 128, b0, conv_q40);
  auto b1 = elanBlock("backbone/blocks/1/elan", 128, 256, 64, 4, b1_base, true,
                      conv_q40);
  features.push_back(b1);

  // Block 2:
  auto b2_base = downSampleRouteS1("backbone/blocks/2/base", 256, 256, b1,
                                   Tensor(), conv_q40);
  auto b2 = elanBlock("backbone/blocks/2/elan", 256, 512, 128, 4, b2_base, true,
                      conv_q40);
  features.push_back(b2);

  // Block 3:
  auto b3_base = downSampleRouteS1("backbone/blocks/3/base", 512, 512, b2,
                                   Tensor(), conv_q40);
  auto b3 = elanBlock("backbone/blocks/3/elan", 512, 1024, 256, 4, b3_base,
                      true, conv_q40);
  features.push_back(b3);

  // Block 4:
  auto b4_base = downSampleRouteS1("backbone/blocks/4/base", 1024, 1024, b3,
                                   Tensor(), conv_q40);
  auto b4 = elanBlock("backbone/blocks/4/elan", 1024, 1024, 256, 4, b4_base,
                      true, conv_q40);
  features.push_back(b4);

  return b4;
}

// Build whole YOLOv7 FPN head and detection ends
inline std::vector<Tensor> buildHead(const std::vector<Tensor> &features,
                                     bool conv_q40 = false) {
  auto b4 = features[4];

  // SPP Block
  auto spp_out = sppcspcBlock("features/spp", 1024, 512, b4, conv_q40);

  // Feature Up 0
  auto up0_base = upSampleBlock("features/feature_up/0/base", 512, 512, 1024,
                                spp_out, features[3], conv_q40);
  auto up0 = elanBlock("features/feature_up/0/elan", 512, 256, 256, 4, up0_base,
                       false, conv_q40); // ELAN_S2

  // Feature Up 1
  auto up1_base = upSampleBlock("features/feature_up/1/base", 256, 256, 512,
                                up0, features[2], conv_q40);
  auto up1 = elanBlock("features/feature_up/1/elan", 256, 128, 128, 4, up1_base,
                       false, conv_q40); // ELAN_S2

  // Feature Down 0
  auto down0_base = downSampleRouteS1("features/feature_down/0/base", 128, 512,
                                      up1, up0, conv_q40);
  auto down0 = elanBlock("features/feature_down/0/elan", 512, 256, 256, 4,
                         down0_base, false, conv_q40); // ELAN_S2

  // Feature Down 1
  auto down1_base = downSampleRouteS1("features/feature_down/1/base", 256, 1024,
                                      down0, spp_out, conv_q40);
  auto down1 = elanBlock("features/feature_down/1/elan", 1024, 512, 512, 4,
                         down1_base, false, conv_q40); // ELAN_S2

  // Ends with SiLU activation
  auto end0 = convBnSilu("features/ends/0", 128, 256, 3, 1, 1, up1, conv_q40);
  auto end1 = convBnSilu("features/ends/1", 256, 512, 3, 1, 1, down0, conv_q40);
  auto end2 =
    convBnSilu("features/ends/2", 512, 1024, 3, 1, 1, down1, conv_q40);

  // Heads without activation
  auto det0 = convNoAct("det0", 256, 30, 1, 1, 0, end0, conv_q40);
  auto det1 = convNoAct("det1", 512, 30, 1, 1, 0, end1, conv_q40);
  auto det2 = convNoAct("det2", 1024, 30, 1, 1, 0, end2, conv_q40);

  return {det0, det1, det2};
}

} // namespace yolov7

#endif // __YOLOV7_GRAPH_H__
