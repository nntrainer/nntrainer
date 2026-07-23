// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   yolov7_pose.h
 * @date   14 July 2026
 * @brief  Lightweight quick_ai::Model wrapper for YOLOv7ReIDtiny pose+ReID,
 *         used by nntr_quantize to build the graph, load the FP32 weights and
 *         re-save them quantized (e.g. Q8_0 conv weights). Inference itself is
 *         driven by the standalone yolov7_pose_infer executable.
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#ifndef __YOLOV7_POSE_MODEL_H__
#define __YOLOV7_POSE_MODEL_H__

#include <iostream>
#include <string>
#include <vector>

#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>

#include "jni/yolov7_pose_graph.h"
#include "model_base.h"
#include "rtmcc_head.h"

namespace quick_ai {

/**
 * @brief YOLOv7ReIDtiny pose model (quantization/build wrapper).
 */
class Yolov7Pose : public Model {
public:
  Yolov7Pose(json &cfg, json &generation_cfg, json &nntr_cfg) {
    // conv_dtype is read by nntr_quantize from nntr_config.json; nothing else
    // is required from the config for a build/load/save cycle.
    (void)cfg;
    (void)generation_cfg;
    (void)nntr_cfg;
  }

  void registerCustomLayers() {
    // Force lazy AppContext initialization first; otherwise a later lazy init
    // (triggered during compile) would wipe factories registered here.
    (void)nntrainer::AppContext::Global();
    auto &ct_engine = nntrainer::Engine::Global();
    auto app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));
    auto tryRegister = [&](auto factory_fn) {
      try {
        app_context->registerFactory(factory_fn);
      } catch (std::invalid_argument &e) {
        std::cerr << "failed to register factory: " << e.what() << std::endl;
      }
    };
    tryRegister(nntrainer::createLayer<quick_ai::RTMCCHeadLayer>);
  }

  void initialize() override {
    registerCustomLayers();

    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", "1"),
                        nntrainer::withKey("model_tensor_type", "FP32-FP32")});

    // Build FP32 graph (no weight_dtype); collect quantizable conv names.
    yolov7_pose::quantizableConvs().clear();
    auto x = ml::train::Tensor(
      ml::train::TensorDim(1, 3, 320, 320,
                           ml::train::TensorDim::Format::NCHW,
                           ml::train::TensorDim::DataType::FP32),
      "input0");
    auto nodes = yolov7_pose::buildBackbone(x, /*conv_q=*/false);
    auto pose_feat =
      yolov7_pose::buildNeck("backbone.features", nodes, false);
    auto pose_out = yolov7_pose::buildPoseHead(pose_feat, false);
    quantizable_convs_ = yolov7_pose::quantizableConvs();

    std::vector<ml::train::Tensor> outs = {pose_out};
    if (model->compile(x, outs, ml::train::ExecutionMode::INFERENCE))
      throw std::runtime_error("Yolov7Pose compile failed");

    is_initialized = true;
  }

  std::vector<std::string> getQuantizableLayerNames() const override {
    return quantizable_convs_;
  }

  // Vision model: no text generation. run() exists only to satisfy the
  // quick_ai::Model interface; nntr_quantize never calls it.
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override {
    throw std::runtime_error("Yolov7Pose::run is not supported");
  }

private:
  std::vector<std::string> quantizable_convs_;
};

} // namespace quick_ai

#endif /* __YOLOV7_POSE_MODEL_H__ */
