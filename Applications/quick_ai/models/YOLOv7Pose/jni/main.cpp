// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   14 July 2026
 * @brief  YOLOv7ReIDtiny pose+ReID (320x320, nkpt=87) inference on nntrainer.
 *
 *         Presets (env YOLO_TENSOR_TYPE):
 *           w32a32   : FP32 weights + FP32 activations  (stage 1, reference)
 *           w8a32    : Q8_0 weights + FP32 activations, NHWC (stage 2)
 *         Weights are loaded from a safetensors produced by weight_converter.py
 *         (and, for w8a32, quantized by nntr_quantize --conv_dtype Q8_0).
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "yolov7_pose_graph.h"
#include "rtmcc_head.h"
#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <tensor_api.h>

extern "C" void openblas_set_num_threads(int);

using ml::train::createLayer;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

namespace {

using namespace yolov7_pose;

constexpr int INPUT_SIZE = 320;

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

struct Keypoint {
  float x, y, score;
};

// Decode RTMPose SimCC: pose is [2*NKPT, SIMCC_BINS] (cls_x rows then cls_y).
std::vector<Keypoint> decodeSimcc(const float *pose) {
  std::vector<Keypoint> kpts(NKPT);
  for (int k = 0; k < NKPT; ++k) {
    const float *rx = pose + static_cast<size_t>(k) * SIMCC_BINS;
    const float *ry = pose + static_cast<size_t>(NKPT + k) * SIMCC_BINS;
    int bx = 0, by = 0;
    float mx = rx[0], my = ry[0];
    for (int i = 1; i < SIMCC_BINS; ++i) {
      if (rx[i] > mx) { mx = rx[i]; bx = i; }
      if (ry[i] > my) { my = ry[i]; by = i; }
    }
    float score = std::min(mx, my);
    if (score <= 0.0f) {
      kpts[k] = {-1.0f, -1.0f, score};
    } else {
      kpts[k] = {bx / 2.0f, by / 2.0f, score};
    }
  }
  return kpts;
}

// Peak resident set size (VmHWM) in KB, or 0 if unavailable.
long peakRSSKB() {
#ifdef __linux__
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      long kb = 0;
      std::sscanf(line.c_str(), "VmHWM: %ld kB", &kb);
      return kb;
    }
  }
#endif
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  auto t_start = std::chrono::steady_clock::now();
  openblas_set_num_threads(4);

  std::string res_dir = (argc > 1) ? argv[1] : ".";
  std::string input_path =
    (argc > 2) ? argv[2] : res_dir + "/input_320.bin";
  std::cout << "[Pose] RES_DIR=" << res_dir << " INPUT=" << input_path
            << std::endl;

  try {
    auto &app_ctx = nntrainer::AppContext::Global();
    (void)app_ctx;

    // Register custom layers used by the pose head.
    {
      auto &ct_engine = nntrainer::Engine::Global();
      auto ctx = static_cast<nntrainer::AppContext *>(
        ct_engine.getRegisteredContext("cpu"));
      auto tryReg = [&](auto fn) {
        try {
          ctx->registerFactory(fn);
        } catch (std::invalid_argument &e) {
          std::cerr << "register: " << e.what() << std::endl;
        }
      };
      tryReg(nntrainer::createLayer<quick_ai::RTMCCHeadLayer>);
    }

    ModelHandle model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", "1")});

    bool preset_nhwc = false;
    bool preset_q = false;
    bool fp16_act = false;
    std::string weights_default = "yolov7_pose.safetensors";
    if (const char *tt = std::getenv("YOLO_TENSOR_TYPE")) {
      std::string s = tt;
      if (s == "w8a16" || s == "W8A16") {
        // Real on-device Q8_0 path: the ARM indirect conv only decodes Q8_0
        // weights against an FP16 activation, so Q8_0 requires FP16 act.
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP16")});
        preset_nhwc = true;
        preset_q = true;
        fp16_act = true;
        yolov7_pose::quantWeightDtype() = "Q8_0";
        weights_default = "yolov7_pose_q8_0.safetensors";
        std::cout << "[Pose] Preset=w8a16 (Q8_0 weights + FP16 act + NHWC)"
                  << std::endl;
      } else if (s == "w32a16" || s == "W32A16") {
        // Diagnostic: FP16 activations WITHOUT Q8_0 weights (NHWC). Isolates
        // FP16-activation overflow from the Q8_0 conv kernel.
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP16")});
        preset_nhwc = true;
        fp16_act = true;
        std::cout << "[Pose] Preset=w32a16 (FP32 weights + FP16 act + NHWC, "
                     "diagnostic)"
                  << std::endl;
      } else if (s == "w8a8" || s == "W8A8") {
        // Q8_0 weights + int8-resident activations (W8A8_DESIGN.md). Same
        // graph and weight file as w8a32; the env flag makes every Q8_0 conv
        // emit a per-tensor-scale QINT8 activation that the next layers
        // (Q8_0 convs, concat, max-pool, nearest-upsample) consume directly.
        // FP32 convs and the head dequantize on entry, so the FP32 islands
        // (stem/blocks.1/head) behave exactly like w8a32.
        setenv("NNTR_W8A8", "1", 1);
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP32")});
        preset_nhwc = true;
        preset_q = true;
        // NNTR_W8A8_FP32W: keep conv weights FP32 in the file and let the
        // per-channel path quantize them ONCE, directly from FP32, at load
        // (__ggml_q8ch_prepare_conv_weight's fp32_src path) with an FP32 per-channel scale.
        // This matches the S0 simulation's per-channel scheme exactly (81/87);
        // a Q8_0 conv file instead double-quantizes (per-block int8 in the file
        // -> per-channel requant at load), which costs the borderline keypoint
        // (80/87). Compute stays int8 (weights are quantized once, cached), so
        // there is no speed change -- only the on-disk weights are larger.
        const bool w8a8_fp32w = std::getenv("NNTR_W8A8_FP32W") != nullptr;
        if (w8a8_fp32w) {
          yolov7_pose::quantWeightDtype() = "FP32";
          weights_default = "yolov7_pose.safetensors";
          std::cout
            << "[Pose] Preset=w8a8 (FP32 weights, load-time per-channel int8 "
               "+ int8 act + NHWC)"
            << std::endl;
        } else {
          yolov7_pose::quantWeightDtype() = "Q8_0";
          weights_default = "yolov7_pose_q8_0.safetensors";
          std::cout << "[Pose] Preset=w8a8 (Q8_0 weights + int8 act + NHWC)"
                    << std::endl;
        }
      } else if (s == "w8a32" || s == "W8A32") {
        // Q8_0 weights + FP32 activations (NHWC). The FP32-activation Q8_0
        // indirect conv kernel (FloatTensor::convQ4_0Indirect ->
        // __ggml_q8_0_q8_0_indirect_GEMM_fp32) keeps activations in FP32 between
        // layers (no FP16 rounding accumulation) with int8 SMMLA compute, so
        // this recovers ~FP32 pose accuracy at int8 speed. Same weight file as
        // w8a16 (no re-quantize).
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP32")});
        preset_nhwc = true;
        preset_q = true;
        yolov7_pose::quantWeightDtype() = "Q8_0";
        weights_default = "yolov7_pose_q8_0.safetensors";
        std::cout << "[Pose] Preset=w8a32 (Q8_0 weights + FP32 act + NHWC)"
                  << std::endl;
      } else { // w32a32 default
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP32")});
        std::cout << "[Pose] Preset=w32a32 (FP32 weights + FP32 act + NCHW)"
                  << std::endl;
      }
    } else {
      model->setProperty(
        {nntrainer::withKey("model_tensor_type", "FP32-FP32")});
      std::cout << "[Pose] Preset=w32a32 (default)" << std::endl;
    }

    bool format_nhwc = false;
    // YOLO_FORCE_NCHW=1 keeps NCHW even for NHWC presets (x86 has no NHWC
    // quantized-conv kernel; used to check Q8_0 numerics on x86).
    if (std::getenv("YOLO_FORCE_NCHW"))
      preset_nhwc = false;
    if (preset_nhwc || std::getenv("YOLO_NHWC")) {
      model->setProperty({nntrainer::withKey("tensor_format", "NHWC")});
      format_nhwc = true;
      std::cout << "[Pose] tensor_format=NHWC" << std::endl;
    }

    auto input_dtype = fp16_act ? ml::train::TensorDim::DataType::FP16
                                : ml::train::TensorDim::DataType::FP32;
    auto x = Tensor(
      ml::train::TensorDim(1, 3, INPUT_SIZE, INPUT_SIZE,
                           ml::train::TensorDim::Format::NCHW, input_dtype),
      "input0");

    // pose_base_v311.pt is pose-only (no ReID head). Enable the ReID branch
    // (second neck "features_feat" + head_feat) only for a merged checkpoint.
    bool with_reid = std::getenv("YOLO_WITH_REID") != nullptr;

    auto nodes = yolov7_pose::buildBackbone(x, preset_q);
    auto pose_feat = yolov7_pose::buildNeck("backbone.features", nodes, preset_q);
    auto pose_out = yolov7_pose::buildPoseHead(pose_feat, preset_q);

    std::vector<Tensor> graph_outputs = {pose_out};
    if (with_reid) {
      auto reid_feat =
        yolov7_pose::buildNeck("backbone.features_feat", nodes, preset_q);
      graph_outputs.push_back(yolov7_pose::buildReidHead(reid_feat));
    }
    std::cout << "[Pose] ReID head: " << (with_reid ? "on" : "off") << std::endl;

    auto t_compile0 = std::chrono::steady_clock::now();
    // Everything from process start to here: binary/library startup, arg
    // parsing, graph construction (buildBackbone/Neck/Head), property setup.
    // Timed so the e2e figure decomposes fully (ORT's session-create load
    // covers the equivalent work on its side).
    double setup_ms =
      std::chrono::duration<double, std::milli>(t_compile0 - t_start).count();
    if (int ret = model->compile(x, graph_outputs,
                                 ml::train::ExecutionMode::INFERENCE))
      throw std::runtime_error("compile failed: " + std::to_string(ret));
    double compile_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_compile0)
        .count();

    std::string weights_path = res_dir + "/" + weights_default;
    if (const char *w = std::getenv("YOLO_WEIGHTS"))
      weights_path = (w[0] == '/') ? std::string(w) : res_dir + "/" + w;
    auto t_load0 = std::chrono::steady_clock::now();
    model->load(weights_path,
                ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
    double load_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t_load0)
                       .count();
    std::cout << "[Pose] Weights loaded: " << weights_path << std::endl;

    auto input = loadBin(input_path);
    if (input.size() != static_cast<size_t>(3 * INPUT_SIZE * INPUT_SIZE))
      throw std::runtime_error("input size mismatch: " +
                               std::to_string(input.size()));

    if (format_nhwc) {
      const int C = 3, H = INPUT_SIZE, W = INPUT_SIZE;
      std::vector<float> nhwc(input.size());
      for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
            nhwc[(h * W + w) * C + c] = input[(c * H + h) * W + w];
      input.swap(nhwc);
    }

    std::vector<float *> in_ptr = {input.data()};
    int iters = std::getenv("YOLO_BENCH_ITERS")
                  ? std::max(1, std::atoi(std::getenv("YOLO_BENCH_ITERS")))
                  : 1;
    std::vector<float *> outs;
    // One warmup inference. Some presets (e.g. w8a8 per-channel) build a cached
    // weight representation on the first forward, and the first inference also
    // triggers the activation-pool allocation + first-touch; timing that as
    // "inference" conflates one-time setup with steady-state latency. A real
    // deployment warms up at init the same way. This one-time cost is reported
    // separately (init) so the e2e figure decomposes fully instead of leaving a
    // large untimed gap. Skip with YOLO_NO_WARMUP.
    double warmup_ms = 0.0;
    if (!std::getenv("YOLO_NO_WARMUP")) {
      auto t_warm0 = std::chrono::steady_clock::now();
      outs = model->inference(1, in_ptr, std::vector<float *>());
      warmup_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_warm0)
                    .count();
    }
    // Drop any layer-profile samples from the warmup so the report reflects
    // steady-state only (NNTR_LAYER_PROFILE; no-op when unset).
    double total_ms = 0.0;
    for (int it = 0; it < iters; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      outs = model->inference(1, in_ptr, std::vector<float *>());
      auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double infer_ms = total_ms / iters;

    // outs[0] = pose [1, 2*NKPT, SIMCC_BINS]; outs[1] = reid [1, EMBED_DIM]
    // (only when the ReID branch is enabled).
    const bool has_reid = outs.size() > 1;
    if (const char *dump = std::getenv("POSE_DUMP")) {
      std::string base = dump;
      std::ofstream pf(base + "_pose.bin", std::ios::binary);
      pf.write(reinterpret_cast<const char *>(outs[0]),
               sizeof(float) * 2 * NKPT * SIMCC_BINS);
      if (has_reid) {
        std::ofstream rf(base + "_reid.bin", std::ios::binary);
        rf.write(reinterpret_cast<const char *>(outs[1]),
                 sizeof(float) * EMBED_DIM);
      }
      std::cout << "[Pose] raw outputs dumped to " << base << "_pose.bin"
                << (has_reid ? " (+_reid.bin)" : "") << std::endl;
    }

    auto kpts = decodeSimcc(outs[0]);

    float score_thr =
      std::getenv("POSE_THR") ? std::stof(std::getenv("POSE_THR")) : 0.5f;
    int visible = 0;
    for (auto &k : kpts)
      if (k.score >= score_thr && k.x >= 0)
        ++visible;
    std::cout << "[Pose] visible keypoints: " << visible << "/" << NKPT
              << " (thr=" << score_thr << ")" << std::endl;

    std::cout << "[";
    for (int k = 0; k < NKPT; ++k) {
      if (k)
        std::cout << ",";
      std::printf("\n  {\"i\": %d, \"x\": %.4g, \"y\": %.4g, \"s\": %.4g}", k,
                  kpts[k].x, kpts[k].y, kpts[k].score);
    }
    std::cout << "\n]" << std::endl;

    if (has_reid) {
      float norm = 0.0f;
      for (int i = 0; i < EMBED_DIM; ++i)
        norm += outs[1][i] * outs[1][i];
      std::cout << "[ReID] embed_dim=" << EMBED_DIM
                << " l2norm=" << std::sqrt(norm) << std::endl;
    }

    long peak_kb = peakRSSKB();
    double e2e_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t_start)
                      .count();
    // init = one-time model setup that a real deployment pays once at startup:
    // graph build (setup) + compile + weight load + warmup (which builds the
    // per-channel weight cache and allocates/first-touches the activation
    // pool). ORT folds the equivalent into its "load" (session create), so
    // report init alongside it for an apples-to-apples e2e decomposition.
    double init_ms = setup_ms + compile_ms + load_ms + warmup_ms;
    std::printf(
      "\n================[ YOLOv7 Pose with NNTrainer ]================\n");
    std::printf("init:      %.1f ms (setup %.1f + compile %.1f + load %.1f + "
                "warmup %.1f)\n",
                init_ms, setup_ms, compile_ms, load_ms, warmup_ms);
    std::printf("inference: %.3f ms (avg over %d iter%s, steady-state)\n",
                infer_ms, iters, iters > 1 ? "s" : "");
    std::printf("keypoints: %d/%d visible\n", visible, NKPT);
    std::printf("peak memory: %ld KB\n", peak_kb);
    std::printf(
      "=============================================================\n");
    std::printf("[e2e time]: %.0f ms  (init %.0f + %d x infer %.1f)\n", e2e_ms,
                init_ms, iters, infer_ms);
    std::printf("Max Resident Set Size: %ld KB\n", peak_kb);

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
