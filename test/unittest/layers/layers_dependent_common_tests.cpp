// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file layer_common_tests.cpp
 * @date 02 July 2021
 * @brief Common test for nntrainer layers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <layers_common_tests.h>

#include <app_context.h>
#include <layer_devel.h>
#include <layer_node.h>

#ifdef ENABLE_OPENCL
#include <cl_context.h>
#endif

constexpr unsigned SAMPLE_TRIES = 10;

TEST_P(LayerSemantics, createFromAppContext_pn) {
  auto &eg = nntrainer::Engine::Global();
  auto ac =
    static_cast<nntrainer::AppContext *>(eg.getRegisteredContext("cpu"));
  if (!(options & LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT)) {
    ac->registerFactory<nntrainer::Layer>(std::get<0>(GetParam()));
  }

  EXPECT_EQ(ac->createObject<nntrainer::Layer>(expected_type)->getType(),
            expected_type);
}

TEST_P(LayerPropertySemantics, setPropertiesInvalid_n) {
  auto lnode = nntrainer::createLayerNode(expected_type);
  EXPECT_THROW(layer->setProperty({valid_properties}), std::invalid_argument);
}

TEST_P(LayerSemantics, setPropertiesInvalid_01_n) {
  auto lnode = nntrainer::createLayerNode(expected_type);
  /** must not crash */
  EXPECT_THROW(layer->setProperty({"unknown_props=2"}), std::invalid_argument);
}

TEST_P(LayerSemantics, setPropertiesInvalid_02_n) {
  auto lnode = nntrainer::createLayerNode(expected_type);
  /** must not crash */
  EXPECT_THROW(layer->setProperty({"unknown_props:2"}), std::invalid_argument);
}

TEST_P(LayerSemantics, finalizeValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type);
  std::vector<std::string> props = {"name=test"};
  std::string input_shape = "input_shape=1:1:1";
  if (expected_type == "weight") {
    input_shape = "weight_dim=1:1:1";
  }
  std::string input_layers = "input_layers=a";
  for (auto idx = 1u; idx < num_inputs; idx++) {
    if (expected_type != "weight") {
      input_shape += ",1:1:1";
      input_layers += ",a";
    }
  }
  props.push_back(input_shape);
  props.push_back(input_layers);
  lnode->setProperty(props);
  lnode->setOutputLayers({"dummy"});

  EXPECT_NO_THROW(lnode->setProperty(valid_properties));

  if (!must_fail) {
    nntrainer::InitLayerContext init_context = lnode->finalize();

    for (auto const &spec : init_context.getOutSpecs())
      EXPECT_GT(spec.variable_spec.dim.getDataLen(), size_t(0));
    for (auto const &ws : init_context.getWeightsSpec())
      EXPECT_GT(std::get<0>(ws).getDataLen(), size_t(0));
    for (auto const &ts : init_context.getTensorsSpec())
      EXPECT_GT(std::get<0>(ts).getDataLen(), size_t(0));
  } else {
    EXPECT_THROW(lnode->finalize(), nntrainer::exception::not_supported);
  }
}

TEST_P(LayerSemantics, getTypeValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type);
  std::string type;

  EXPECT_NO_THROW(type = lnode->getType());
  EXPECT_GT(type.size(), size_t(0));
}

TEST_P(LayerSemantics, gettersValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type);

  EXPECT_NO_THROW(lnode->supportInPlace());
  EXPECT_NO_THROW(lnode->requireLabel());
  EXPECT_NO_THROW(lnode->supportBackwarding());
}

TEST_P(LayerSemantics, setBatchValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type);
  std::vector<std::string> props = {"name=test"};
  std::string input_shape = "input_shape=1:1:1";
  if (expected_type == "weight") {
    input_shape = "weight_dim=1:1:1";
  }
  std::string input_layers = "input_layers=a";
  for (auto idx = 1u; idx < num_inputs; idx++) {
    if (expected_type != "weight") {
      input_shape += ",1:1:1";
      input_layers += ",a";
    }
  }
  props.push_back(input_shape);
  props.push_back(input_layers);
  lnode->setProperty(props);
  lnode->setOutputLayers({"dummy"});

  EXPECT_NO_THROW(lnode->setProperty(valid_properties));

  if (!must_fail) {
    EXPECT_NO_THROW(lnode->finalize());
  } else {
    EXPECT_THROW(lnode->finalize(), nntrainer::exception::not_supported);
  }
}

#ifdef ENABLE_OPENCL
TEST_P(LayerSemanticsGpu, createFromClContext_pn) {
  auto &eg = nntrainer::Engine::Global();
  auto cc = static_cast<nntrainer::ClContext *>(eg.getRegisteredContext("gpu"));
  if (!(options & LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT)) {
    cc->registerFactory<nntrainer::Layer>(std::get<0>(GetParam()));
  }

  EXPECT_EQ(cc->createObject<nntrainer::Layer>(expected_type)->getType(),
            expected_type);
}

/**
 * TODO: Fix the LayerPropertySemantics test errors for GPU
 * In current, layers tested with LayerPropertySemantics are not
 * supported in GPU.
 */
// TEST_P(LayerPropertySemanticsGpu, setPropertiesInvalid_n_gpu) {
// auto lnode =
// nntrainer::createLayerNode(expected_type, {}, ComputeEngine::GPU);
// EXPECT_THROW(layer->setProperty({valid_properties}), std::invalid_argument);
// }

TEST_P(LayerSemanticsGpu, setPropertiesInvalid_n) {
  auto lnode = nntrainer::createLayerNode(expected_type, {"engine=gpu"});
  /** must not crash */
  EXPECT_THROW(layer->setProperty({"unknown_props=2"}), std::invalid_argument);
}

TEST_P(LayerSemanticsGpu, finalizeValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type, {"engine=gpu"});
  std::vector<std::string> props = {"name=test"};
  std::string input_shape = "input_shape=1:1:1";
  if (expected_type == "weight") {
    input_shape = "weight_dim=1:1:1";
  }
  std::string input_layers = "input_layers=a";
  for (auto idx = 1u; idx < num_inputs; idx++) {
    if (expected_type != "weight") {
      input_shape += ",1:1:1";
      input_layers += ",a";
    }
  }
  props.push_back(input_shape);
  props.push_back(input_layers);
  lnode->setProperty(props);
  lnode->setOutputLayers({"dummy"});

  EXPECT_NO_THROW(lnode->setProperty(valid_properties));

  if (!must_fail) {
    nntrainer::InitLayerContext init_context = lnode->finalize();

    for (auto const &spec : init_context.getOutSpecs())
      EXPECT_GT(spec.variable_spec.dim.getDataLen(), size_t(0));
    for (auto const &ws : init_context.getWeightsSpec())
      EXPECT_GT(std::get<0>(ws).getDataLen(), size_t(0));
    for (auto const &ts : init_context.getTensorsSpec())
      EXPECT_GT(std::get<0>(ts).getDataLen(), size_t(0));
  } else {
    EXPECT_THROW(lnode->finalize(), nntrainer::exception::not_supported);
  }
}

TEST_P(LayerSemanticsGpu, getTypeValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type, {"engine=gpu"});
  std::string type;

  EXPECT_NO_THROW(type = lnode->getType());
  EXPECT_GT(type.size(), size_t(0));
}

TEST_P(LayerSemanticsGpu, gettersValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type, {"engine=gpu"});

  EXPECT_NO_THROW(lnode->supportInPlace());
  EXPECT_NO_THROW(lnode->requireLabel());
  EXPECT_NO_THROW(lnode->supportBackwarding());
}

TEST_P(LayerSemanticsGpu, setBatchValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type, {"engine=gpu"});
  std::vector<std::string> props = {"name=test"};
  std::string input_shape = "input_shape=1:1:1";
  if (expected_type == "weight") {
    input_shape = "weight_dim=1:1:1";
  }
  std::string input_layers = "input_layers=a";
  for (auto idx = 1u; idx < num_inputs; idx++) {
    if (expected_type != "weight") {
      input_shape += ",1:1:1";
      input_layers += ",a";
    }
  }
  props.push_back(input_shape);
  props.push_back(input_layers);
  lnode->setProperty(props);
  lnode->setOutputLayers({"dummy"});

  EXPECT_NO_THROW(lnode->setProperty(valid_properties));

  if (!must_fail) {
    EXPECT_NO_THROW(lnode->finalize());
  } else {
    EXPECT_THROW(lnode->finalize(), nntrainer::exception::not_supported);
  }
}
#endif
