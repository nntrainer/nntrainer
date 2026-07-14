// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   13 July 2026
 * @brief  YOLOv7 box-detector (320x320, nc=5) inference example on nntrainer.
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
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

#include "yolov7_graph.h"
#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <tensor.h>
#include <tensor_api.h>

// Optional direct image input. Enabled only when stb_image.h is present.
#ifdef YOLO_WITH_STB_IMAGE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma GCC diagnostic pop
#endif

extern "C" void openblas_set_num_threads(int);

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

namespace yolov7 {

struct Detection {
  float x1, y1, x2, y2, conf;
  int cls;
};

// Sigmoid activation helper
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// IoU helper
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

// YOLOv7 anchors in pixel space (model's stride-divided anchor buffer × stride),
// extracted from weights/yolov7-package-apc-sstk-v001-ratio4.pt IDetect.anchors.
// These are NOT the stock COCO anchors — this model was trained with its own
// anchor set, so decoding with stock anchors shifts box w/h and produces spurious
// boxes that diverge from the PyTorch reference.
const float ANCHORS[3][3][2] = {
  {{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}},   // det0, stride 8
  {{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}},  // det1, stride 16
  {{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}} // det2, stride 32
};

const float STRIDES[3] = {8.0f, 16.0f, 32.0f};

// Decode YOLOv7 model outputs for all 3 scales
inline std::vector<Detection>
decodeAllScales(const std::vector<float *> &outs,
                const std::vector<std::array<int, 2>> &grid_sizes,
                float conf_thres, bool is_nhwc) {
  std::vector<Detection> candidates;
  candidates.reserve(1000);

  for (int s = 0; s < 3; ++s) {
    const float *raw = outs[s];
    int H = grid_sizes[s][0];
    int W = grid_sizes[s][1];
    float stride = STRIDES[s];
    int OUT_CH = 30;

    std::vector<float> nchw_buf;
    if (is_nhwc) {
      int N_pix = H * W;
      nchw_buf.resize(static_cast<size_t>(OUT_CH) * N_pix);
      for (int a = 0; a < N_pix; ++a) {
        for (int c = 0; c < OUT_CH; ++c) {
          nchw_buf[static_cast<size_t>(c) * N_pix + a] =
            raw[static_cast<size_t>(a) * OUT_CH + c];
        }
      }
      raw = nchw_buf.data();
    }

    const float *buf = raw;

    // For each anchor
    for (int a = 0; a < 3; ++a) {
      float aw = ANCHORS[s][a][0];
      float ah = ANCHORS[s][a][1];

      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          // Objectness score at index 4 (0-based) of the 10 outputs per anchor
          float raw_obj = buf[(a * 10 + 4) * H * W + y * W + x];
          if (std::isnan(raw_obj))
            continue;
          float obj = sigmoid(raw_obj);
          if (obj < conf_thres)
            continue;

          // Box offsets
          float raw_tx = buf[(a * 10 + 0) * H * W + y * W + x];
          float raw_ty = buf[(a * 10 + 1) * H * W + y * W + x];
          float raw_tw = buf[(a * 10 + 2) * H * W + y * W + x];
          float raw_th = buf[(a * 10 + 3) * H * W + y * W + x];
          if (std::isnan(raw_tx) || std::isnan(raw_ty) || std::isnan(raw_tw) ||
              std::isnan(raw_th))
            continue;

          // Decoding formulas
          float bx =
            (sigmoid(raw_tx) * 2.0f - 0.5f + static_cast<float>(x)) * stride;
          float by =
            (sigmoid(raw_ty) * 2.0f - 0.5f + static_cast<float>(y)) * stride;
          float bw = (sigmoid(raw_tw) * 2.0f) * (sigmoid(raw_tw) * 2.0f) * aw;
          float bh = (sigmoid(raw_th) * 2.0f) * (sigmoid(raw_th) * 2.0f) * ah;

          float x1 = bx - bw / 2.0f;
          float y1 = by - bh / 2.0f;
          float x2 = bx + bw / 2.0f;
          float y2 = by + bh / 2.0f;

          // Classes (5 classes, indices 5..9)
          for (int c = 0; c < 5; ++c) {
            float raw_cls = buf[(a * 10 + 5 + c) * H * W + y * W + x];
            if (std::isnan(raw_cls))
              continue;
            float score = sigmoid(raw_cls) * obj;
            if (score > conf_thres) {
              candidates.push_back({x1, y1, x2, y2, score, c});
            }
          }
        }
      }
    }
  }
  return candidates;
}

// Class-specific NMS
inline std::vector<Detection> nms(std::vector<Detection> &candidates,
                                  float iou_thres, int max_det) {
  // Sort descending by conf
  std::sort(
    candidates.begin(), candidates.end(),
    [](const Detection &a, const Detection &b) { return a.conf > b.conf; });

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
      if (!suppressed[j] && candidates[i].cls == candidates[j].cls) {
        if (iou(candidates[i], candidates[j]) > iou_thres) {
          suppressed[j] = true;
        }
      }
    }
  }
  return result;
}

} // namespace yolov7

namespace {

std::string RES_DIR = "/home/seungbaek/Downloads/video/detector/res";

std::vector<float> loadBin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open: " + path);
  f.seekg(0, std::ios::end);
  size_t n = f.tellg() / sizeof(float);
  f.seekg(0);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
  return v;
}

#ifdef YOLO_WITH_STB_IMAGE
bool isImagePath(const std::string &path) {
  auto ext = path.substr(path.find_last_of(".") + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext == "jpg" || ext == "jpeg" || ext == "png";
}

std::vector<float> loadImageLetterbox(const std::string &path) {
  int w, h, c;
  unsigned char *data = stbi_load(path.c_str(), &w, &h, &c, 3);
  if (!data)
    throw std::runtime_error("stbi_load failed: " + path);

  const int target = 320;
  float r =
    std::min(static_cast<float>(target) / w, static_cast<float>(target) / h);
  int nw = std::round(w * r);
  int nh = std::round(h * r);

  int pad_w = (target - nw) / 2;
  int pad_h = (target - nh) / 2;

  std::vector<float> out(target * target * 3,
                         114.0f / 255.0f); // default gray pad

  for (int dy = 0; dy < nh; ++dy) {
    int sy = std::min(static_cast<int>(std::round(dy / r)), h - 1);
    for (int dx = 0; dy < nw; ++dx) {
      int sx = std::min(static_cast<int>(std::round(dx / r)), w - 1);
      int s_idx = (sy * w + sx) * 3;
      int d_idx_r = (0 * target + (dy + pad_h)) * target + (dx + pad_w);
      int d_idx_g = (1 * target + (dy + pad_h)) * target + (dx + pad_w);
      int d_idx_b = (2 * target + (dy + pad_h)) * target + (dx + pad_w);

      out[d_idx_r] = data[s_idx + 0] / 255.0f;
      out[d_idx_g] = data[s_idx + 1] / 255.0f;
      out[d_idx_b] = data[s_idx + 2] / 255.0f;
    }
  }

  stbi_image_free(data);
  return out;
}
#endif

void printPeakRSS() {
#ifdef __linux__
  if (std::ifstream f("/proc/self/status"); f) {
    std::string line;
    while (std::getline(f, line)) {
      if (line.rfind("VmPeak:", 0) == 0 || line.rfind("VmHWM:", 0) == 0) {
        std::cout << "  " << line << std::endl;
      }
    }
  }
#endif
}

} // namespace

int main(int argc, char **argv) {
  openblas_set_num_threads(1);

  if (argc > 1) {
    RES_DIR = argv[1];
  }

  std::string input_path = RES_DIR + "/input_320.bin";
  if (argc > 2) {
    input_path = argv[2];
  }

  std::cout << "[YOLOv7] RES_DIR   = " << RES_DIR << std::endl;
  std::cout << "[YOLOv7] INPUT_BIN = " << input_path << std::endl;

  try {
    // Force AppContext creation
    auto &app_ctx = nntrainer::AppContext::Global();
    (void)app_ctx;

    // Use neural net factory method to instantiate the Model
    ModelHandle model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", "1")});

    bool fp16_act = false;
    bool preset_nhwc = false;
    bool preset_q40 = false;
    if (const char *tt = std::getenv("YOLO_TENSOR_TYPE")) {
      std::string tts = tt;
      if (tts == "w4a16" || tts == "W4A16") {
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP16")});
        fp16_act = true;
        preset_nhwc = true;
        preset_q40 = true;
        yolov7::quantWeightDtype() = "Q4_0";
        std::cout << "[YOLOv7] Preset = w4a16 (Q4_0 weights + FP16 act + NHWC)"
                  << std::endl;
      } else if (tts == "w4a8" || tts == "W4A8") {
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP16")});
        fp16_act = true;
        preset_nhwc = true;
        preset_q40 = true;
        yolov7::quantWeightDtype() = "Q4_0";
        std::cout
          << "[YOLOv7] Preset = w4a8 (Q4_0 weights + static Q8_0 act + NHWC)"
          << std::endl;
      } else if (tts == "w8a16" || tts == "W8A16") {
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP16")});
        fp16_act = true;
        preset_nhwc = true;
        preset_q40 = true;
        yolov7::quantWeightDtype() = "Q8_0";
        std::cout << "[YOLOv7] Preset = w8a16 (Q8_0 weights + FP16 act + NHWC)"
                  << std::endl;
      } else if (tts == "w8a32" || tts == "W8A32") {
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP32")});
        fp16_act = false;
        preset_nhwc = true;
        preset_q40 = true;
        yolov7::quantWeightDtype() = "Q8_0";
        std::cout << "[YOLOv7] Preset = w8a32 (Q8_0 weights + FP32 act + NHWC)"
                  << std::endl;
      } else if (tts == "QINT8-FP16") {
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "QINT8-FP16")});
        fp16_act = true;
        std::cout << "[YOLOv7] Preset = QINT8-FP16 (Legacy)" << std::endl;
      } else {
        model->setProperty({nntrainer::withKey("model_tensor_type", tt)});
        if (tts.find("FP16") != std::string::npos) {
          fp16_act = true;
        }
        std::cout << "[YOLOv7] model_tensor_type = " << tt << std::endl;
      }
    }

    bool format_nhwc = false;
    if (preset_nhwc || std::getenv("YOLO_NHWC")) {
      model->setProperty({nntrainer::withKey("tensor_format", "NHWC")});
      format_nhwc = true;
      std::cout << "[YOLOv7] tensor_format = NHWC" << std::endl;
    }

    auto input_dtype = fp16_act ? ml::train::TensorDim::DataType::FP16
                                : ml::train::TensorDim::DataType::FP32;

    // Input tensor shape [1, 3, 320, 320]
    auto x =
      Tensor(ml::train::TensorDim(
               1, 3, 320, 320, ml::train::TensorDim::Format::NCHW, input_dtype),
             "input0");

    // Build the functional graph with quantization flags
    std::vector<Tensor> features;
    auto b4 = yolov7::buildBackbone(x, features, preset_q40);
    auto outputs = yolov7::buildHead(features, preset_q40);

    // Compile model for inference
    if (int ret =
          model->compile(x, outputs, ml::train::ExecutionMode::INFERENCE)) {
      throw std::runtime_error("compile failed: " + std::to_string(ret));
    }

    // Load converted safetensors weights
    std::string weights_path = RES_DIR + "/yolov7.safetensors";
    if (const char *wenv = std::getenv("YOLO_WEIGHTS")) {
      weights_path =
        (wenv[0] == '/') ? std::string(wenv) : RES_DIR + "/" + wenv;
    }
    model->load(weights_path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
    std::cout << "[YOLOv7] Model built and weights loaded (" << weights_path
              << ")." << std::endl;

    // Load input data
#ifdef YOLO_WITH_STB_IMAGE
    auto input = isImagePath(input_path) ? loadImageLetterbox(input_path)
                                         : loadBin(input_path);
#else
    auto input = loadBin(input_path);
#endif

    if (format_nhwc) {
      const int C = 3, H = 320, W = 320;
      std::vector<float> nhwc(input.size());
      for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
            nhwc[(h * W + w) * C + c] = input[(c * H + h) * W + w];
      input.swap(nhwc);
    }

    std::vector<float *> in_ptr = {input.data()};

    // Warm-up and inference benchmarking
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
    std::cout << "[YOLOv7] Inference done (" << outs.size() << " outputs)."
              << std::endl;

    std::cout << "[YOLOv7] Inference time: " << (total_ms / bench_iters)
              << " ms (avg over " << bench_iters << " iters)" << std::endl;
    printPeakRSS();

    // Decode and NMS
    std::vector<std::array<int, 2>> grid_sizes = {{40, 40}, {20, 20}, {10, 10}};
    float conf_thres =
      std::getenv("YOLO_CONF") ? std::stof(std::getenv("YOLO_CONF")) : 0.001f;
    float iou_thres =
      std::getenv("YOLO_IOU") ? std::stof(std::getenv("YOLO_IOU")) : 0.65f;

    auto candidates =
      yolov7::decodeAllScales(outs, grid_sizes, conf_thres, format_nhwc);
    auto dets = yolov7::nms(candidates, iou_thres, 300);

    // JSON format output matching evaluate.py references
    std::cout << "\n[";
    for (size_t i = 0; i < dets.size(); ++i) {
      const auto &d = dets[i];
      if (i)
        std::cout << ",";
      std::printf("\n  {\"x1\": %.6g, \"y1\": %.6g, \"x2\": %.6g, \"y2\": "
                  "%.6g, \"conf\": %.6g, \"cls\": %d}",
                  d.x1, d.y1, d.x2, d.y2, d.conf, d.cls);
    }
    std::cout << (dets.empty() ? "" : "\n") << "]" << std::endl;

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
