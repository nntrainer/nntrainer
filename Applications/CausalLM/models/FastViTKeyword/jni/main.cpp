// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main_split.cpp
 * @date   31 July 2026
 * @brief  FastViTKeyword inference with split NHWC(backbone) + NCHW(head).
 *
 * Runs the FastViT-S12 backbone in NHWC format (channel-last optimized Q8_0
 * W8A16), then transposes the backbone output to NCHW for the MLP head
 * (channel-first optimized Q8_0 W8A16). This matches the deployment
 * architecture: vision conv layers use NHWC Q8_0, transformer/FC layers use
 * NCHW Q8_0.
 *
 * Architecture:
 *   Model 1 (NHWC): input[1,3,320,320] -> backbone -> [1,1024,10,10] (NHWC)
 *   Transpose: [1,1024,10,10] (NHWC) -> [1,1024,10,10] (NCHW)
 *   Model 2 (NCHW): [1,1024,10,10] -> GAP -> head -> {logits[507],
 * features[512]}
 *
 * Usage: fastvit_keyword_split_infer [RES_DIR] [INPUT_BIN]
 *   RES_DIR   dir with weights/ and input bins
 *   INPUT_BIN [1,3,320,320] float32 NCHW (default: RES_DIR/input.bin)
 *
 * Env vars:
 *   KEYWORD_IMGSZ       Input image size (square, default 320).
 *   KEYWORD_VERIFY      If set, compare outputs to PyTorch references.
 *   KEYWORD_WEIGHTS     Override weights path (default:
 * fastvit_keyword_q8_0.safetensors) KEYWORD_NHWC_WEIGHTS  Override backbone
 * weights path KEYWORD_NCHW_WEIGHTS  Override head weights path
 *
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
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include "fastvit_attention_layer.h"
#include "fastvit_keyword_graph.h"
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

std::string RES_DIR = "Applications/CausalLM/models/FastViTKeyword/res";

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

/** @brief Save a float buffer to a binary file. */

/** @brief Register the FastViT custom layers with the global AppContext. */
void registerCustomLayers() {
  auto &app_ctx = nntrainer::AppContext::Global();
  app_ctx.registerFactory(
    nntrainer::createLayer<fastvit_keyword::FastViTAttentionLayer>);
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

    const int imgsz = std::getenv("KEYWORD_IMGSZ")
                        ? std::max(32, std::atoi(std::getenv("KEYWORD_IMGSZ")))
                        : 320;

    const int proj_target_dim = 507;
    const int proj_feature_dim = 512;

    const std::string input_path =
      (argc > 2) ? argv[2] : (RES_DIR + "/input.bin");
    const bool verify = std::getenv("KEYWORD_VERIFY") != nullptr ||
                        std::getenv("YOLO_VERIFY") != nullptr;

    // Weight paths: split into backbone (NHWC) and head (NCHW)
    std::string backbone_weights =
      RES_DIR + "/fastvit_keyword_q8_0.safetensors";
    if (const char *wenv = std::getenv("KEYWORD_NHWC_WEIGHTS"))
      backbone_weights =
        (wenv[0] == '/') ? std::string(wenv) : RES_DIR + "/" + wenv;

    std::string head_weights =
      RES_DIR + "/fastvit_keyword_head_q8_0.safetensors";
    if (const char *wenv = std::getenv("KEYWORD_NCHW_WEIGHTS"))
      head_weights =
        (wenv[0] == '/') ? std::string(wenv) : RES_DIR + "/" + wenv;

    std::cout << "[FastViTKeyword-Split] imgsz=" << imgsz
              << " proj_target=" << proj_target_dim
              << " proj_feature=" << proj_feature_dim << std::endl;
    std::cout << "  Backbone (NHWC Q8_0): " << backbone_weights << std::endl;
    std::cout << "  Head (NCHW Q8_0):    " << head_weights << std::endl;

    registerCustomLayers();

    // ====================================================================
    // Model 1: FastViT Backbone (NHWC format, Q8_0 W8A16)
    // ====================================================================
    std::cout << "\n=== Building Model 1: FastViT Backbone (NHWC Q8_0) ==="
              << std::endl;

    ModelHandle backbone_model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    backbone_model->setProperty(
      {nntrainer::withKey("batch_size", "1"),
       nntrainer::withKey("tensor_format", "NHWC"),
       nntrainer::withKey("model_tensor_type", "FP32-FP32")});

    // Build backbone graph with Q8_0 weight dtype
    fastvit_keyword::quantizableConvs().clear();
    fastvit_keyword::quantWeightDtype() = "Q8_0";

    // Input: NCHW format (will be converted to NHWC by the graph).
    // The model tensor_format=NHWC tells nntrainer to lay out all activation
    // tensors in NHWC internally. Input is provided as NCHW float32 and
    // the graph handles the conversion.
    auto backbone_input =
      Tensor(ml::train::TensorDim(1, 3, imgsz, imgsz,
                                  ml::train::TensorDim::Format::NHWC,
                                  ml::train::TensorDim::DataType::FP32),
             "input0");

    auto backbone_out = fastvit_keyword::buildBackbone(backbone_input);

    if (int ret = backbone_model->compile(backbone_input, backbone_out,
                                          ml::train::ExecutionMode::INFERENCE))
      throw std::runtime_error("Backbone model compile failed: " +
                               std::to_string(ret));

    backbone_model->load(backbone_weights,
                         ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
    std::cout << "Backbone model built and weights loaded." << std::endl;

    // ====================================================================
    // Model 2: Keyword Head (NCHW format, Q8_0 W8A16)
    // ====================================================================
    std::cout << "\n=== Building Model 2: Keyword Head (NCHW Q8_0) ==="
              << std::endl;

    ModelHandle head_model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    head_model->setProperty(
      {nntrainer::withKey("batch_size", "1"),
       nntrainer::withKey("tensor_format", "NCHW"),
       nntrainer::withKey("model_tensor_type", "FP32-FP32")});

    // Head input: backbone output [1, 1024, 10, 10] in NCHW format
    auto head_input = Tensor(
      ml::train::TensorDim(1, 1024, 10, 10, ml::train::TensorDim::Format::NCHW,
                           ml::train::TensorDim::DataType::FP32),
      "head_input0");

    auto head_outputs =
      fastvit_keyword::buildHead(head_input, proj_target_dim, proj_feature_dim);

    if (int ret = head_model->compile(head_input, head_outputs,
                                      ml::train::ExecutionMode::INFERENCE))
      throw std::runtime_error("Head model compile failed: " +
                               std::to_string(ret));

    head_model->load(head_weights,
                     ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
    std::cout << "Head model built and weights loaded." << std::endl;

    // ====================================================================
    // Load input and run pipeline
    // ====================================================================
    auto input = loadBin(input_path);
    std::cout << "\nInput loaded from: " << input_path
              << " (size=" << input.size() << ")" << std::endl;
    std::vector<float *> in_ptr = {input.data()};

    // --- Run backbone (NHWC) ---
    std::cout << "\n--- Running backbone (NHWC) ---" << std::endl;
    double total_ms = 0.0;
    int bench_iters =
      std::getenv("KEYWORD_BENCH_ITERS")
        ? std::max(1, std::atoi(std::getenv("KEYWORD_BENCH_ITERS")))
        : 1;

    std::vector<float *> backbone_outs;
    for (int it = 0; it < bench_iters; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      backbone_outs =
        backbone_model->inference(1, in_ptr, std::vector<float *>());
      auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::cout << "Backbone inference: " << (total_ms / bench_iters)
              << " ms (avg over " << bench_iters << " iters)" << std::endl;

    // backbone_outs[0] = [1, 1024, 10, 10] in NHWC physical layout
    // We need to transpose it to NCHW for the head model
    // NHWC: [N, H, W, C] -> NCHW: [N, C, H, W]
    // For a [1, 1024, 10, 10] tensor this is a transpose of the last 3 dims
    const float *backbone_feat = backbone_outs[0];
    size_t feat_n = 1024 * 10 * 10;

    // The backbone output in NHWC format is [1, 10, 10, 1024] physically.
    // We need to transpose to NCHW [1, 1024, 10, 10] for the head input.
    // Since the head input expects NCHW [1, 1024, 10, 10], we do the transpose.
    std::vector<float> backbone_nchw(1024 * 10 * 10);
    const int C = 1024, H = 10, W = 10;
    // NHWC: data[h*W*C + w*C + c]
    // NCHW: data[c*H*W + h*W + w]
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          backbone_nchw[c * H * W + h * W + w] =
            backbone_feat[h * W * C + w * C + c];
        }
      }
    }

    // Verify backbone output against reference if available
    if (verify) {
      // The reference is in NCHW format, so compare with our transposed output
      std::cout << "\n[Backbone verification vs PyTorch reference (NCHW)]:"
                << std::endl;
      verifyAgainst("ref_backbone_out.bin", backbone_nchw.data(), feat_n);
    }

    // --- Run head (NCHW) ---
    std::cout << "\n--- Running head (NCHW) ---" << std::endl;
    double head_ms = 0.0;
    std::vector<float *> head_outs;
    std::vector<float *> head_in_ptr = {backbone_nchw.data()};
    for (int it = 0; it < bench_iters; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      head_outs = head_model->inference(1, head_in_ptr, std::vector<float *>());
      auto t1 = std::chrono::steady_clock::now();
      head_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::cout << "Head inference: " << (head_ms / bench_iters)
              << " ms (avg over " << bench_iters << " iters)" << std::endl;
    std::cout << "Total inference: " << ((total_ms + head_ms) / bench_iters)
              << " ms" << std::endl;

    // Output: head_outs[0] = logits [1, 507, 1, 1], head_outs[1] = features [1,
    // 512, 1, 1]
    const float *logits = head_outs[0];
    const float *features = head_outs[1];

    // Sigmoid for probabilities
    std::vector<float> probs(proj_target_dim);
    for (int i = 0; i < proj_target_dim; ++i)
      probs[i] = 1.0f / (1.0f + std::exp(-logits[i]));

    // Print top-20
    std::vector<int> indices(proj_target_dim);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + 20, indices.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    std::cout << "\n[TOP-20 Keywords]" << std::endl;
    for (int i = 0; i < 20 && i < proj_target_dim; ++i) {
      int idx = indices[i];
      std::printf("  %2d. idx=%3d  prob=%.4f\n", i + 1, idx, probs[idx]);
    }

    // Print threshold-based predictions
    float threshold = 0.5f;
    if (const char *t = std::getenv("KEYWORD_THRESHOLD"))
      threshold = std::stof(t);

    std::cout << "\n[Predicted threshold >= " << threshold << "]" << std::endl;
    bool any = false;
    for (int i = 0; i < proj_target_dim; ++i) {
      if (probs[i] >= threshold) {
        std::printf("  idx=%3d  prob=%.4f\n", i, probs[i]);
        any = true;
      }
    }
    if (!any)
      std::cout << "  (no keywords above threshold)" << std::endl;

    // Verification
    if (verify) {
      std::cout << "\nVerification vs PyTorch references:" << std::endl;
      verifyAgainst("ref_logits.bin", logits, proj_target_dim);
      verifyAgainst("ref_features.bin", features, proj_feature_dim);
      verifyAgainst("ref_sigmoid.bin", probs.data(), proj_target_dim);
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
