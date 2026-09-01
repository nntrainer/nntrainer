// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   18 June 2026
 * @brief  YOLOv11m detection inference example on nntrainer.
 *         Builds the full model (backbone + FPN head + 3-scale Detect head),
 *         loads converted weights, runs one forward pass, and post-processes
 *         (DFL decode + NMS) into final detection boxes.
 *
 *         Usage: yolov11_infer [RES_DIR] [INPUT_BIN]
 *           RES_DIR   dir with weights/ and input bins
 *                     (default: Applications/CausalLM/models/YOLOv11/res)
 *           INPUT_BIN [1,3,IMGSZ,IMGSZ] float32 NCHW (default:
 * RES_DIR/input_<IMGSZ>.bin) Set env YOLO_VERIFY=1 to also compare raw logits /
 * decoded output to PyTorch references (ref_p3/p4/p5.bin, ref_decoded.bin) when
 * present.
 *
 *         Env vars:
 *           YOLO_IMGSZ  Input image size (square, default 832).
 *           YOLO_NC     Number of detection classes (default 1).
 *           YOLO_REG_MAX  DFL regression max (default 16, fixed by YOLOv11).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include "c2psa_layer.h"
#include "yolov11_graph.h"
#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <tensor.h>
#include <tensor_api.h>

// Optional direct image input. Enabled only when stb_image.h is present (the
// build defines YOLO_WITH_STB_IMAGE via meson fs.exists). stb_image.h is NOT
// committed; download it to enable .jpg/.png input:
//   curl -fsSL
//   https://raw.githubusercontent.com/nothings/stb/master/stb_image.h \
//     -o Applications/CausalLM/models/YOLOv11/jni/stb_image.h
// Without it the example still builds and runs on .bin input.
#ifdef YOLO_WITH_STB_IMAGE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma GCC diagnostic pop
#endif

// OpenBLAS runtime thread-count API (pthread build). Declared here to avoid
// pulling the cblas backend header into the app; see the call in main().
extern "C" void openblas_set_num_threads(int);

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

// Graph block builders are defined in yolov11_graph.h (included above).
// Post-processing (DFL decode, NMS) remains here.

namespace yolov11 {

// ===== Post-processing (DFL decode + dist2bbox + NMS) =====
// ---------------------------------------------------------------------------
// Anchor generation (ultralytics tal.py: make_anchors)
//
// For each scale with grid (H, W) and stride s:
//   sx = [0.5, 1.5, ..., W-0.5]
//   sy = [0.5, 1.5, ..., H-0.5]
//   row-major meshgrid: y outer, x inner → [H*W, 2] (x,y) in grid units
// Concat P3 → P4 → P5.
// ---------------------------------------------------------------------------
struct ScaleInfo {
  int H;
  int W;
  float stride;
};

/**
 * @brief Generate anchor points and stride tensor.
 * @param scales  Vector of (H, W, stride) for each detection scale.
 * @param[out] anchors     [num_anchors, 2] (x, y) in grid units.
 * @param[out] strides_out [num_anchors] stride per anchor.
 */
inline void makeAnchors(const std::vector<ScaleInfo> &scales,
                        std::vector<float> &anchors,
                        std::vector<float> &strides_out) {
  size_t total = 0;
  for (const auto &s : scales) {
    total += static_cast<size_t>(s.H) * s.W;
  }
  anchors.resize(total * 2);
  strides_out.resize(total);

  size_t off = 0;
  for (const auto &s : scales) {
    for (int iy = 0; iy < s.H; ++iy) {
      for (int ix = 0; ix < s.W; ++ix) {
        anchors[off * 2 + 0] = static_cast<float>(ix) + 0.5f;
        anchors[off * 2 + 1] = static_cast<float>(iy) + 0.5f;
        strides_out[off] = s.stride;
        ++off;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// DFL (Distribution Focal Loss) decode
// ---------------------------------------------------------------------------
inline void dfl(const float *raw_box, int reg_max, int N,
                std::vector<float> &dist) {
  dist.resize(4 * N);
  for (int c = 0; c < 4; ++c) {
    for (int a = 0; a < N; ++a) {
      float logits[16];
      for (int k = 0; k < reg_max; ++k) {
        logits[k] = raw_box[(c * reg_max + k) * N + a];
      }
      float max_logit = *std::max_element(logits, logits + reg_max);
      float sum = 0.0f;
      float exp_v[16];
      for (int k = 0; k < reg_max; ++k) {
        exp_v[k] = std::exp(logits[k] - max_logit);
        sum += exp_v[k];
      }
      float val = 0.0f;
      for (int k = 0; k < reg_max; ++k) {
        val += (exp_v[k] / sum) * static_cast<float>(k);
      }
      dist[c * N + a] = val;
    }
  }
}

// ---------------------------------------------------------------------------
// dist2bbox + stride multiply → XYWH pixels at 832-scale
// ---------------------------------------------------------------------------
inline void dist2bbox(const std::vector<float> &dist, int N,
                      const std::vector<float> &anchors,
                      const std::vector<float> &strides,
                      std::vector<float> &decoded_box) {
  const float *lt_x = dist.data() + 0 * N;
  const float *lt_y = dist.data() + 1 * N;
  const float *rb_x = dist.data() + 2 * N;
  const float *rb_y = dist.data() + 3 * N;

  decoded_box.resize(4 * N);
  float *cx = decoded_box.data() + 0 * N;
  float *cy = decoded_box.data() + 1 * N;
  float *w = decoded_box.data() + 2 * N;
  float *h = decoded_box.data() + 3 * N;

  for (int a = 0; a < N; ++a) {
    float ax = anchors[a * 2 + 0];
    float ay = anchors[a * 2 + 1];
    float s = strides[a];

    float x1 = ax - lt_x[a];
    float y1 = ay - lt_y[a];
    float x2 = ax + rb_x[a];
    float y2 = ay + rb_y[a];

    cx[a] = (x1 + x2) * 0.5f * s;
    cy[a] = (y1 + y2) * 0.5f * s;
    w[a] = (x2 - x1) * s;
    h[a] = (y2 - y1) * s;
  }
}

// ---------------------------------------------------------------------------
// Full post-processing pipeline for one scale
// ---------------------------------------------------------------------------
inline void decodeOneScale(const float *raw, int H, int W, float stride,
                           const std::vector<float> &anchors,
                           const std::vector<float> &strides_vec,
                           int anchor_off, int N_total, int box_ch, int nc,
                           std::vector<float> &decoded) {
  const int N = H * W;
  const int reg_max = box_ch / 4;

  const float *raw_box = raw;
  const float *raw_cls = raw + box_ch * N;

  std::vector<float> dist;
  dfl(raw_box, reg_max, N, dist);

  std::vector<float> scale_anchors(N * 2);
  std::vector<float> scale_strides(N);
  for (int a = 0; a < N; ++a) {
    scale_anchors[a * 2 + 0] = anchors[(anchor_off + a) * 2 + 0];
    scale_anchors[a * 2 + 1] = anchors[(anchor_off + a) * 2 + 1];
    scale_strides[a] = strides_vec[anchor_off + a];
  }

  std::vector<float> decoded_box;
  dist2bbox(dist, N, scale_anchors, scale_strides, decoded_box);

  for (int c = 0; c < 4; ++c) {
    for (int a = 0; a < N; ++a) {
      decoded[c * N_total + anchor_off + a] = decoded_box[c * N + a];
    }
  }
  // Per-class sigmoid scores: decoded[(4 + cls) * N_total + anchor_off + a]
  for (int cls = 0; cls < nc; ++cls) {
    for (int a = 0; a < N; ++a) {
      decoded[(4 + cls) * N_total + anchor_off + a] =
        1.0f / (1.0f + std::exp(-raw_cls[cls * N + a]));
    }
  }
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------
struct Detection {
  float x1, y1, x2, y2, conf;
  int cls;
};

inline float iou(const Detection &a, const Detection &b) {
  float ix1 = std::max(a.x1, b.x1);
  float iy1 = std::max(a.y1, b.y1);
  float ix2 = std::min(a.x2, b.x2);
  float iy2 = std::min(a.y2, b.y2);
  float inter_w = std::max(0.0f, ix2 - ix1);
  float inter_h = std::max(0.0f, iy2 - iy1);
  float inter = inter_w * inter_h;
  if (inter == 0.0f)
    return 0.0f;
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter);
}

/**
 * @brief Run per-class NMS on decoded predictions.
 * @param decoded    [(4+nc), N_total] (cx, cy, w, h, cls0_score, cls1_score,
 * ...)
 * @param N_total    total anchors
 * @param nc         number of classes
 * @param conf_thres confidence threshold
 * @param iou_thres  IoU threshold
 * @param max_det    max detections to keep
 * @return sorted (descending conf) list of Detection
 */
inline std::vector<Detection> nms(const std::vector<float> &decoded,
                                  int N_total, int nc, float conf_thres,
                                  float iou_thres, int max_det) {
  const float *cx_ptr = decoded.data() + 0 * N_total;
  const float *cy_ptr = decoded.data() + 1 * N_total;
  const float *w_ptr = decoded.data() + 2 * N_total;
  const float *h_ptr = decoded.data() + 3 * N_total;

  // Build candidate list: for each anchor, find the best class score;
  // if it exceeds conf_thres, emit a Detection with that class.
  std::vector<Detection> candidates;
  candidates.reserve(512);
  for (int a = 0; a < N_total; ++a) {
    float best_score = 0.0f;
    int best_cls = 0;
    for (int cls = 0; cls < nc; ++cls) {
      float score = decoded[(4 + cls) * N_total + a];
      if (score > best_score) {
        best_score = score;
        best_cls = cls;
      }
    }
    if (best_score <= conf_thres)
      continue;
    float cx = cx_ptr[a];
    float cy = cy_ptr[a];
    float bw = w_ptr[a];
    float bh = h_ptr[a];
    float x1 = cx - bw * 0.5f;
    float y1 = cy - bh * 0.5f;
    float x2 = cx + bw * 0.5f;
    float y2 = cy + bh * 0.5f;
    candidates.push_back({x1, y1, x2, y2, best_score, best_cls});
  }

  // Sort descending by conf
  std::sort(
    candidates.begin(), candidates.end(),
    [](const Detection &a, const Detection &b) { return a.conf > b.conf; });

  // Greedy per-class NMS: only suppress within the same class
  std::vector<bool> suppressed(candidates.size(), false);
  std::vector<Detection> result;
  result.reserve(max_det);

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i])
      continue;
    result.push_back(candidates[i]);
    if (static_cast<int>(result.size()) >= max_det)
      break;
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (suppressed[j] || candidates[j].cls != candidates[i].cls)
        continue;
      if (iou(candidates[i], candidates[j]) > iou_thres) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

} // namespace yolov11

namespace {

// Resource directory (yolov11m.safetensors + input/reference bins). Overridable
// via argv[1]; default assumes the binary runs from the nntrainer project root.
std::string RES_DIR = "Applications/CausalLM/models/YOLOv11/res";

/**
 * @brief Load binary file as float vector
 */
std::vector<float> loadBin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Cannot open: " + path);
  }
  f.seekg(0, std::ios::end);
  size_t n = f.tellg() / sizeof(float);
  f.seekg(0);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
  return v;
}

#ifdef YOLO_WITH_STB_IMAGE
/** @brief True if the path looks like an image we can decode with stb_image. */
bool isImagePath(const std::string &p) {
  auto lower = p;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  for (const char *e : {".jpg", ".jpeg", ".png", ".bmp"})
    if (lower.size() >= std::strlen(e) &&
        lower.compare(lower.size() - std::strlen(e), std::strlen(e), e) == 0)
      return true;
  return false;
}

/**
 * @brief Decode an image and letterbox it to [1,3,size,size] float32 CHW RGB.
 *
 * Mirrors ultralytics letterbox: resize (bilinear, half-pixel centers) keeping
 * aspect ratio, pad the rest with 114, /255. Note the bilinear resampler is not
 * bit-identical to OpenCV's, so detections on a real image are very close to
 * but not exactly equal to the PyTorch/cv2 path; for an exact match feed the
 * .bin written by PyTorch/run_pytorch.py instead.
 */
std::vector<float> loadImageLetterbox(const std::string &path, int size = 832,
                                      float pad = 114.0f) {
  int w = 0, h = 0, c = 0;
  unsigned char *img = stbi_load(path.c_str(), &w, &h, &c, 3); // force RGB
  if (!img)
    throw std::runtime_error("Cannot decode image: " + path);

  const float r = std::min((float)size / h, (float)size / w);
  const int nh = (int)std::round(h * r), nw = (int)std::round(w * r);
  const int top = (size - nh) / 2, left = (size - nw) / 2;

  std::vector<float> out(3UL * size * size, pad / 255.0f);
  for (int oy = 0; oy < nh; ++oy) {
    float sy = (oy + 0.5f) * h / nh - 0.5f; // cv2 INTER_LINEAR mapping
    int y0 = (int)std::floor(sy);
    float fy = sy - y0;
    int y0c = std::min(std::max(y0, 0), h - 1), y1c = std::min(y0 + 1, h - 1);
    for (int ox = 0; ox < nw; ++ox) {
      float sx = (ox + 0.5f) * w / nw - 0.5f;
      int x0 = (int)std::floor(sx);
      float fx = sx - x0;
      int x0c = std::min(std::max(x0, 0), w - 1), x1c = std::min(x0 + 1, w - 1);
      for (int ch = 0; ch < 3; ++ch) {
        auto px = [&](int yy, int xx) {
          return (float)img[(yy * w + xx) * 3 + ch];
        };
        float v = px(y0c, x0c) * (1 - fx) * (1 - fy) +
                  px(y0c, x1c) * fx * (1 - fy) + px(y1c, x0c) * (1 - fx) * fy +
                  px(y1c, x1c) * fx * fy;
        out[(size_t)ch * size * size + (size_t)(top + oy) * size +
            (left + ox)] = v / 255.0f;
      }
    }
  }
  stbi_image_free(img);
  std::cout << "image " << path << " (" << w << "x" << h << ") letterboxed to "
            << size << "x" << size << std::endl;
  return out;
}
#endif // YOLO_WITH_STB_IMAGE

/** @brief Register the YOLOv11 custom layers with the global AppContext.
 *  Only C2PSA is custom; the Detect head uses standard nntrainer layers. */
void registerCustomLayers() {
  auto &app_ctx = nntrainer::AppContext::Global();
  app_ctx.registerFactory(nntrainer::createLayer<yolov11::PSAAttentionLayer>);
}

/** @brief Optionally compare a logit tensor to a PyTorch reference .bin. */
void verifyAgainst(const std::string &ref_name, const float *out, size_t n) {
  std::ifstream f(RES_DIR + "/" + ref_name, std::ios::binary);
  if (!f) {
    std::cout << "  [verify] " << ref_name << " not found, skipped"
              << std::endl;
    return;
  }
  auto ref = loadBin(RES_DIR + "/" + ref_name);
  float max_diff = 0.0f;
  for (size_t i = 0; i < n && i < ref.size(); ++i)
    max_diff = std::max(max_diff, std::abs(out[i] - ref[i]));
  std::cout << "  [verify] " << ref_name << ": max_abs_diff=" << max_diff
            << std::endl;
}

// Report peak resident set size (VmHWM) from /proc/self/status. Linux/Android
// only; silently does nothing elsewhere or if the file is unreadable.
inline void printPeakRSS() {
#if defined(__linux__)
  std::ifstream st("/proc/self/status");
  if (!st.is_open())
    return;
  std::string line;
  while (std::getline(st, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      std::cout << "Peak RSS: " << line.substr(6) << std::endl;
      return;
    }
  }
#endif
}

} // namespace

int main(int argc, char *argv[]) {
  try {
    // Force the BLAS backend (used by the C2PSA attention sgemm and the detect
    // head) to a single thread. nntrainer's conv path drives its own
    // ThreadManager pool; letting OpenBLAS spawn a competing thread pool over-
    // subscribes the cores and both slows the conv path and inflates RSS
    // (~+25MB). OpenBLAS reads OPENBLAS_NUM_THREADS at its dlopen constructor
    // (before main), so setenv() here is too late; the pthread build honors the
    // runtime openblas_set_num_threads() API instead. Small GEMMs stay cache-
    // friendly without a competing pool.
    openblas_set_num_threads(1);

    if (argc > 1)
      RES_DIR = argv[1];

    // Image size (square input). Default 832, overridable via YOLO_IMGSZ.
    const int imgsz = std::getenv("YOLO_IMGSZ")
                        ? std::max(32, std::atoi(std::getenv("YOLO_IMGSZ")))
                        : 832;

    // Number of detection classes. Default 1, overridable via YOLO_NC.
    const int nc = std::getenv("YOLO_NC")
                     ? std::max(1, std::atoi(std::getenv("YOLO_NC")))
                     : 1;

    // DFL regression max (fixed by YOLOv11 architecture at 16).
    const int reg_max = 16;

    // Box channels = 4 * reg_max = 64. Output channels = box + nc.
    const int box_ch = 4 * reg_max;
    const int out_ch = box_ch + nc;

    // Model architecture preset (v11m / v11s). Default v11m.
    std::string model_preset =
      std::getenv("YOLO_MODEL") ? std::getenv("YOLO_MODEL") : "v11m";
    yolov11::ModelConfig cfg = yolov11::ModelConfig::v11m();
    if (model_preset == "v11s" || model_preset == "V11S") {
      cfg = yolov11::ModelConfig::v11s();
    }

    const std::string input_path =
      (argc > 2) ? argv[2]
                 : (RES_DIR + "/input_" + std::to_string(imgsz) + ".bin");
    const bool verify = std::getenv("YOLO_VERIFY") != nullptr;

    std::cout << "[YOLO] imgsz=" << imgsz << " nc=" << nc
              << " out_ch=" << out_ch << std::endl;

    registerCustomLayers();

    // Build the full model: input -> backbone -> head -> 3 detect outputs.
    ModelHandle model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", "1")});

    // YOLO_TENSOR_TYPE accepts either a raw dtype pair (e.g. "FP32-FP16") or a
    // named quantization preset:
    //   w4a16 — Q4_0 weights + FP16 activations + NHWC layout (best latency)
    //   w4a8  — Q4_0 weights + Q8_0 activations + NHWC layout (experimental)
    //   w8a16 — Q8_0 weights + FP16 activations + NHWC layout (best accuracy)
    // Default (unset) is FP32-FP32.  YOLOv11's input is a float image; for an
    // FP16-activation model the InputLayer must be declared FP16 (PR#4000).
    bool fp16_act = false;
    bool preset_q40 = false;  // implied by w4a16/w4a8 presets
    bool preset_nhwc = false; // implied by w4a16/w4a8 presets

    // Offline quantization mode (YOLO_QUANTIZE_OUT set) must build the graph in
    // plain FP32 so the FP32 source weights load and Conv2DLayer::save can
    // quantize them ("quantized save only supports FP32 source weight").  The
    // preset is still parsed here, but in quantize mode it only selects the
    // TARGET weight dtype written to the file — it must not switch the live
    // graph to FP16/NHWC/quantized weights.
    const bool quantize_mode = (std::getenv("YOLO_QUANTIZE_OUT") != nullptr);
    auto quant_target_dtype = ml::train::TensorDim::DataType::Q4_0;
    if (const char *tt = std::getenv("YOLO_TENSOR_TYPE")) {
      std::string tts = tt;
      if (tts == "w4a16" || tts == "W4A16") {
        if (!quantize_mode) {
          model->setProperty(
            {nntrainer::withKey("model_tensor_type", "FP32-FP16")});
          fp16_act = true;
          preset_q40 = true;
          preset_nhwc = true;
        }
        std::cout << "[YOLO] preset = w4a16 (Q4_0 weights + FP16 act + NHWC)"
                  << std::endl;
      } else if (tts == "w4a8" || tts == "W4A8") {
        if (!quantize_mode) {
          model->setProperty(
            {nntrainer::withKey("model_tensor_type", "FP32-FP16")});
          fp16_act = true;
          preset_q40 = true;
          preset_nhwc = true;
          setenv("NNTR_CONV_Q8ACT", "1", 1);
        }
        std::cout << "[YOLO] preset = w4a8 (Q4_0 weights + Q8_0 act + NHWC)"
                  << std::endl;
      } else if (tts == "w8a16" || tts == "W8A16") {
        quant_target_dtype = ml::train::TensorDim::DataType::Q8_0;
        if (!quantize_mode) {
          model->setProperty(
            {nntrainer::withKey("model_tensor_type", "FP32-FP16")});
          fp16_act = true;
          preset_q40 = true;
          preset_nhwc = true;
          yolov11::quantWeightDtype() = "Q8_0";
        }
        std::cout << "[YOLO] preset = w8a16 (Q8_0 weights + FP16 act + NHWC)"
                  << (quantize_mode ? " [quantize mode: FP32 graph, Q8_0 target]"
                                    : "")
                  << std::endl;
      } else {
        model->setProperty({nntrainer::withKey("model_tensor_type", tt)});
        auto dash = tts.find('-');
        std::string act =
          (dash == std::string::npos) ? tts : tts.substr(dash + 1);
        fp16_act = (act == "FP16");
        std::cout << "[YOLO] model_tensor_type = " << tt
                  << " (fp16_act=" << (fp16_act ? "1" : "0") << ")"
                  << std::endl;
      }
    }

    // NHWC layout: activated by w4a16/w4a8 presets or explicit YOLO_NHWC.
    if (preset_nhwc || std::getenv("YOLO_NHWC")) {
      model->setProperty({nntrainer::withKey("tensor_format", "NHWC")});
      std::cout << "[YOLO] tensor_format = NHWC" << std::endl;
    }

    // Q4_0 weight path: activated by preset or explicit YOLO_CONV_Q40.
    const bool conv_q40 =
      !quantize_mode &&
      (preset_q40 || (std::getenv("YOLO_CONV_Q40") != nullptr));

    // In quantize mode, collect the Q4_0-eligible conv layer names as the graph
    // is built (single source of truth for eligibility) to drive the per-layer
    // dtype map for model->save().
    std::vector<std::string> q_conv_names;
    if (quantize_mode)
      yolov11::quantConvSink() = &q_conv_names;

    // Declare the input tensor's dtype to match the activation dtype so the
    // synthesized InputLayer emits FP16 output for an FP16-activation model.
    // The input tensor's declared format must match the graph layout so the
    // synthesized InputLayer emits the right physical layout to conv0. Under an
    // NHWC preset the whole graph is channel-last, so declare the input NHWC
    // too (and feed NHWC-ordered bytes below).
    const bool in_nhwc = (preset_nhwc || std::getenv("YOLO_NHWC"));
    const auto in_fmt = in_nhwc ? ml::train::TensorDim::Format::NHWC
                                : ml::train::TensorDim::Format::NCHW;
    auto x =
      fp16_act
        ? Tensor(ml::train::TensorDim(1, 3, imgsz, imgsz, in_fmt,
                                      ml::train::TensorDim::DataType::FP16),
                 "input0")
        : Tensor(ml::train::TensorDim(1, 3, imgsz, imgsz, in_fmt,
                                      ml::train::TensorDim::DataType::FP32),
                 "input0");
    Tensor m4, m6;
    auto m10 = yolov11::buildBackbone(x, m4, m6, cfg, conv_q40);
    auto outputs =
      yolov11::buildHead(m4, m6, m10, nc, cfg, conv_q40); // {P3, P4, P5}

    yolov11::quantConvSink() = nullptr;

    if (int ret =
          model->compile(x, outputs, ml::train::ExecutionMode::INFERENCE))
      throw std::runtime_error("compile failed: " + std::to_string(ret));
    // Load every weight from the single nntrainer safetensors produced by
    // PyTorch/convert_weights.py (tensor names match the model weight names).
    // YOLO_WEIGHTS overrides the file (absolute, or relative to RES_DIR) so a
    // baseline and a fused/quantized model can be compared without rebuilding.
    std::string weights_path = RES_DIR + "/yolov11m.safetensors";
    if (const char *wenv = std::getenv("YOLO_WEIGHTS")) {
      weights_path =
        (wenv[0] == '/') ? std::string(wenv) : RES_DIR + "/" + wenv;
    }
    model->load(weights_path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
    std::cout << "Model built and weights loaded (" << weights_path << ")."
              << std::endl;

    // Offline quantization: re-save with the framework's general per-layer
    // quantizer. dtype=Q4_0 + empty layer map => every layer is targeted, and
    // Conv2DLayer::save quantizes the eligible conv filters (out_ch & CRS both
    // 32-aligned) to the [CRS, out_ch] Q4_0 matmul weight + ISA repack, while
    // biases / ineligible filters / weight-free layers stay FP32. This is the
    // framework equivalent of the offline python script.
    if (quantize_mode) {
      const std::string out_q = std::getenv("YOLO_QUANTIZE_OUT");
      ml::train::ISA isa = ml::train::ISA::DEFAULT;
      if (const char *ie = std::getenv("YOLO_QUANTIZE_ISA")) {
        std::string s = ie;
        if (s == "arm" || s == "ARM")
          isa = ml::train::ISA::ARM;
        else if (s == "x86" || s == "X86")
          isa = ml::train::ISA::X86;
      }
      // SAFETENSORS save requires the global dtype to be NONE; quantization is
      // driven by the per-layer map (conv filters -> Q4_0). Conv2DLayer::save
      // does the conv -> [CRS, out_ch] Q4_0 repack; ineligible/bias stay FP32.
      std::map<std::string, ml::train::TensorDim::DataType> dmap;
      for (const auto &n : q_conv_names)
        dmap[n] = quant_target_dtype;
      model->save(out_q, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS,
                  ml::train::TensorDim::DataType::NONE, dmap, isa);
      std::cout << "[YOLO] quantized " << dmap.size() << " conv filters -> "
                << out_q << " (isa="
                << (std::getenv("YOLO_QUANTIZE_ISA")
                      ? std::getenv("YOLO_QUANTIZE_ISA")
                      : "default")
                << ")" << std::endl;
      return 0;
    }

    // Run one forward pass on the input.
    // argv[2] may be a raw [1,3,imgsz,imgsz] float32 .bin (e.g. from
    // run_pytorch.py), or — when built with stb_image.h present — an image
    // (.jpg/.png/...) which is decoded + letterboxed here.
#ifdef YOLO_WITH_STB_IMAGE
    auto input = isImagePath(input_path) ? loadImageLetterbox(input_path)
                                         : loadBin(input_path);
#else
    auto input = loadBin(input_path);
#endif
    // The inference() API is FP32 by contract: always hand it the plain FP32
    // image. When the graph input is declared FP16 the framework converts
    // FP32->FP16 at the binding boundary (mapExternalTensor) through the Tensor
    // system — no app-side conversion.
    // When the graph is NHWC, the input bytes must also be NHWC-ordered
    // ([N,H,W,C]); input_832.bin is stored NCHW, so transpose here.
    if (preset_nhwc || std::getenv("YOLO_NHWC")) {
      const int C = 3, H = imgsz, W = imgsz;
      std::vector<float> nhwc(input.size());
      for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
            nhwc[(h * W + w) * C + c] = input[(c * H + h) * W + w];
      input.swap(nhwc);
    }
    std::vector<float *> in_ptr = {input.data()};

    // Inference timing. YOLO_BENCH_ITERS (default 1) controls how many timed
    // forward passes to run; the average wall-clock is reported and the last
    // run's outputs feed post-processing. More iters give a stabler number.
    int bench_iters =
      std::getenv("YOLO_BENCH_ITERS")
        ? std::max(1, std::atoi(std::getenv("YOLO_BENCH_ITERS")))
        : 1;
    std::vector<float *> outs;
    double total_ms = 0.0;
    for (int it = 0; it < bench_iters; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      outs = model->inference(1, in_ptr, std::vector<float *>());
      auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::cout << "Inference done (" << outs.size() << " scale outputs)."
              << std::endl;
    std::cout << "Inference time: " << (total_ms / bench_iters)
              << " ms (avg over " << bench_iters << " iters)" << std::endl;
    printPeakRSS();

    // Post-process: DFL decode + dist2bbox + sigmoid -> [5, N] then NMS.
    // YOLOv11 uses 3 detection scales at strides 8, 16, 32. Grid sizes are
    // derived from imgsz / stride (must be divisible by 32 for 3 downsamples).
    const int s3 = imgsz / 8;  // P3 grid (stride 8)
    const int s4 = imgsz / 16; // P4 grid (stride 16)
    const int s5 = imgsz / 32; // P5 grid (stride 32)
    std::vector<yolov11::ScaleInfo> scales = {
      {s3, s3, 8.0f}, {s4, s4, 16.0f}, {s5, s5, 32.0f}};
    const int N_total = s3 * s3 + s4 * s4 + s5 * s5;
    std::vector<float> anchors, strides;
    yolov11::makeAnchors(scales, anchors, strides);

    // When the graph runs NHWC, each scale output tensor is stored channel-last
    // in memory ((h*W+w)*C + c), but decodeOneScale expects the NCHW channel-
    // major layout (c*N + a). Transpose each output to NCHW before decoding.
    // Detect output channels = 4*reg_max (DFL box) + nc.
    const bool out_nhwc = preset_nhwc || std::getenv("YOLO_NHWC");
    const int OUT_CH = out_ch;
    // Decoded layout: [(4 + nc), N_total] — 4 box coords + nc class scores
    std::vector<float> decoded(static_cast<size_t>(4 + nc) * N_total, 0.0f);
    int off = 0;
    for (size_t i = 0; i < scales.size(); ++i) {
      const float *raw = outs[i];
      std::vector<float> nchw_buf;
      if (out_nhwc) {
        const int N = scales[i].H * scales[i].W;
        nchw_buf.resize(static_cast<size_t>(OUT_CH) * N);
        for (int a = 0; a < N; ++a)
          for (int c = 0; c < OUT_CH; ++c)
            nchw_buf[static_cast<size_t>(c) * N + a] =
              raw[static_cast<size_t>(a) * OUT_CH + c];
        raw = nchw_buf.data();
      }
      yolov11::decodeOneScale(raw, scales[i].H, scales[i].W, scales[i].stride,
                              anchors, strides, off, N_total, box_ch, nc,
                              decoded);
      off += scales[i].H * scales[i].W;
    }

    const float conf_thres =
      std::getenv("YOLO_CONF") ? std::stof(std::getenv("YOLO_CONF")) : 0.25f;
    const float iou_thres =
      std::getenv("YOLO_IOU") ? std::stof(std::getenv("YOLO_IOU")) : 0.70f;
    auto dets = yolov11::nms(decoded, N_total, nc, conf_thres, iou_thres, 300);

    // JSON output — same field names as the PyTorch reference JSON
    std::cout << "\n[";
    for (size_t i = 0; i < dets.size(); ++i) {
      const auto &d = dets[i];
      if (i)
        std::cout << ",";
      std::printf("\n  {\"x1\": %.6g, \"y1\": %.6g, \"x2\": %.6g,"
                  " \"y2\": %.6g, \"conf\": %.6g, \"cls\": %d}",
                  d.x1, d.y1, d.x2, d.y2, d.conf, d.cls);
    }
    std::cout << (dets.empty() ? "" : "\n") << "]" << std::endl;

    if (verify) {
      std::cout << "\nVerification vs PyTorch references:" << std::endl;
      const size_t ns[3] = {static_cast<size_t>(out_ch) * s3 * s3,
                            static_cast<size_t>(out_ch) * s4 * s4,
                            static_cast<size_t>(out_ch) * s5 * s5};
      const char *names[3] = {"ref_p3.bin", "ref_p4.bin", "ref_p5.bin"};
      for (int i = 0; i < 3; ++i)
        verifyAgainst(names[i], outs[i], ns[i]);
      verifyAgainst("ref_decoded.bin", decoded.data(), decoded.size());
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
