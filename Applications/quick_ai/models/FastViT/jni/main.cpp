// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   15 July 2026
 * @brief  FastViT-S12 backbone inference example on nntrainer.
 *
 * Builds the general FastViT-S12 backbone (stem + 4 stages + final_conv),
 * loads converted weights, runs one forward pass, and outputs the
 * [1,1024,10,10] feature map. When FASTVIT_VERIFY=1 (or KEYWORD_VERIFY=1),
 * compares the backbone output against a PyTorch reference .bin file.
 *
 * Usage: fastvit_backbone_infer [RES_DIR] [INPUT_BIN]
 *   RES_DIR   dir with weights/ and input bins
 *             (default: Applications/CausalLM/models/FastViT/res)
 *   INPUT_BIN [1,3,320,320] float32 NCHW (default: RES_DIR/input.bin)
 *
 * Env vars:
 *   FASTVIT_IMGSZ  Input image size (square, default 320).
 *   FASTVIT_VERIFY If set, compare output to PyTorch reference.
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "fastvit_attention_layer.h"
#include "fastvit_backbone_graph.h"
#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <tensor.h>
#include <tensor_api.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

namespace {

std::string RES_DIR = "Applications/CausalLM/models/FastViT/res";

/** @brief Load binary file as float vector */
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

/** @brief Register the FastViT custom layers with the global AppContext. */
void registerCustomLayers() {
  auto &app_ctx = nntrainer::AppContext::Global();
  app_ctx.registerFactory(
    nntrainer::createLayer<fastvit::FastViTAttentionLayer>);
}

/** @brief Optionally compare a tensor to a PyTorch reference .bin. */
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

} // namespace

int main(int argc, char *argv[]) {
  try {
    if (argc > 1)
      RES_DIR = argv[1];

    const int imgsz = std::getenv("FASTVIT_IMGSZ")
                        ? std::max(32, std::atoi(std::getenv("FASTVIT_IMGSZ")))
                        : 320;

    const std::string input_path =
      (argc > 2) ? argv[2] : (RES_DIR + "/input.bin");
    const bool verify = std::getenv("FASTVIT_VERIFY") != nullptr ||
                        std::getenv("KEYWORD_VERIFY") != nullptr;

    std::cout << "[FastViT] imgsz=" << imgsz << " (backbone-only)" << std::endl;

    registerCustomLayers();

    // Build the backbone only: input -> backbone -> [1,1024,10,10]
    ModelHandle model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", "1")});

    // --- Tensor type / quantization presets ---
    bool preset_nhwc = false;
    bool preset_q40 = false;
    if (const char *tt = std::getenv("FASTVIT_TENSOR_TYPE")) {
      std::string tts = tt;
      if (tts == "w8a8" || tts == "W8A8") {
        // W8A8: int8-resident activations between conv layers.
        // Uses FP32-FP32 model_tensor_type (no FP16 dependency) with
        // NNTR_W8A8 env flag enabling per-tensor int8 activation quantization.
        // NNTR_W8A8_FP32W: keep conv weights FP32 in the file and let the
        // per-channel path quantize them at load time (best accuracy).
        setenv("NNTR_W8A8", "1", 1);
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP32")});
        preset_nhwc = true;
        preset_q40 = true;
        const bool w8a8_fp32w = std::getenv("NNTR_W8A8_FP32W") != nullptr;
        if (w8a8_fp32w) {
          fastvit::quantWeightDtype() = "FP32";
          std::cout << "[FastViT] Preset = w8a8 (FP32 weights, load-time "
                       "per-channel int8 + int8 act + NHWC)"
                    << std::endl;
        } else {
          fastvit::quantWeightDtype() = "Q8_0";
          std::cout
            << "[FastViT] Preset = w8a8 (Q8_0 weights + int8 act + NHWC)"
            << std::endl;
        }
      } else if (tts == "w8a32" || tts == "W8A32") {
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP32")});
        preset_nhwc = true;
        preset_q40 = true;
        fastvit::quantWeightDtype() = "Q8_0";
        std::cout << "[FastViT] Preset = w8a32 (Q8_0 weights + FP32 act + NHWC)"
                  << std::endl;
      } else {
        model->setProperty({nntrainer::withKey("model_tensor_type", tts)});
        std::cout << "[FastViT] model_tensor_type = " << tts << std::endl;
      }
    }

    if (preset_nhwc || std::getenv("FASTVIT_NHWC")) {
      model->setProperty({nntrainer::withKey("tensor_format", "NHWC")});
      std::cout << "[FastViT] tensor_format = NHWC" << std::endl;
    }

    auto x = Tensor(ml::train::TensorDim(1, 3, imgsz, imgsz,
                                         ml::train::TensorDim::Format::NCHW,
                                         ml::train::TensorDim::DataType::FP32),
                    "input0");

    auto backbone_out = fastvit::buildBackbone(x, preset_q40);

    if (int ret = model->compile(x, {backbone_out},
                                 ml::train::ExecutionMode::INFERENCE))
      throw std::runtime_error("compile failed: " + std::to_string(ret));

    // Load weights
    std::string weights_path = RES_DIR + "/fastvit_backbone.safetensors";
    if (const char *wenv = std::getenv("FASTVIT_WEIGHTS")) {
      weights_path =
        (wenv[0] == '/') ? std::string(wenv) : RES_DIR + "/" + wenv;
    }
    model->load(weights_path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
    std::cout << "Model built and weights loaded (" << weights_path << ")."
              << std::endl;

    // Load input
    auto input = loadBin(input_path);
    std::cout << "Input loaded from: " << input_path
              << " (size=" << input.size() << ")" << std::endl;
    std::vector<float *> in_ptr = {input.data()};

    // Inference timing
    int bench_iters =
      std::getenv("FASTVIT_BENCH_ITERS")
        ? std::max(1, std::atoi(std::getenv("FASTVIT_BENCH_ITERS")))
        : 1;
    std::vector<float *> outs;
    double total_ms = 0.0;
    for (int it = 0; it < bench_iters; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      outs = model->inference(1, in_ptr, std::vector<float *>());
      auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::cout << "Inference done (" << outs.size() << " outputs)." << std::endl;
    std::cout << "Inference time: " << (total_ms / bench_iters)
              << " ms (avg over " << bench_iters << " iters)" << std::endl;

    // Output: backbone_out [1, 1024, 10, 10] = 102400 floats
    const float *feat = outs[0];
    size_t feat_n = 1024 * 10 * 10;
    std::cout << "\n[Backbone feature] shape=[1,1024,10,10]"
              << " mean=" <<
      [&] {
        double s = 0.0;
        for (size_t i = 0; i < feat_n; ++i)
          s += feat[i];
        return s / feat_n;
      }() << std::endl;

    // Verification

    if (verify) {
      std::cout << "\nVerification vs PyTorch references:" << std::endl;
      verifyAgainst("ref_backbone_out.bin", feat, feat_n);
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
