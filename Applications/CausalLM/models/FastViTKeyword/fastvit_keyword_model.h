// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   fastvit_keyword_model.h
 * @date   28 July 2026
 * @brief  Lightweight causallm::Model wrapper for FastViTKeyword, used by
 *         nntr_quantize to build the graph, load FP32 weights and re-save them
 *         quantized (e.g. Q8_0 conv weights for W8A16). Inference itself is
 *         driven by the standalone fastvit_keyword_infer executable.
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __FASTVIT_KEYWORD_MODEL_H__
#define __FASTVIT_KEYWORD_MODEL_H__

#include <iostream>
#include <string>
#include <vector>

#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>

#include "jni/fastvit_attention_layer.h"
#include "jni/fastvit_keyword_graph.h"
#include "transformer.h"

namespace causallm {

/**
 * @brief FastViTKeyword model (quantization/build wrapper).
 */
class FastViTKeywordModel : public Transformer {
public:
  FastViTKeywordModel(json &cfg, json &generation_cfg, json &nntr_cfg) {
    (void)cfg;
    (void)generation_cfg;
    (void)nntr_cfg;
  }

  void registerCustomLayers() {
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
    tryRegister(nntrainer::createLayer<fastvit_keyword::FastViTAttentionLayer>);
  }

  void initialize() override {
    registerCustomLayers();

    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", "1"),
                        nntrainer::withKey("model_tensor_type", "FP32-FP32")});

    // Build FP32 graph (no weight_dtype); collect quantizable conv names.
    // NCHW format matches the source FP32 safetensors file.
    fastvit_keyword::quantizableConvs().clear();
    fastvit_keyword::quantWeightDtype() = "FP32";

    const int imgsz = 320;
    auto x = ml::train::Tensor(
      ml::train::TensorDim(1, 3, imgsz, imgsz,
                           ml::train::TensorDim::Format::NCHW,
                           ml::train::TensorDim::DataType::FP32),
      "input0");

    auto backbone_out = fastvit_keyword::buildBackbone(x);
    auto outputs = fastvit_keyword::buildHead(backbone_out, 507, 512);
    quantizable_convs_ = fastvit_keyword::quantizableConvs();

    if (model->compile(x, outputs, ml::train::ExecutionMode::INFERENCE))
      throw std::runtime_error("FastViTKeywordModel compile failed");

    is_initialized = true;
  }

  std::vector<std::string> getQuantizableLayerNames() const override {
    return quantizable_convs_;
  }

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override {
    throw std::runtime_error("FastViTKeywordModel::run is not supported");
  }

private:
  std::vector<std::string> quantizable_convs_;
};

} // namespace causallm

#endif /* __FASTVIT_KEYWORD_MODEL_H__ */
