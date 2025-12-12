// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file unittest_nntrainer_graph.cpp
 * @date 29 Oct 2020
 * @brief NNTrainer graph test.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <fc_layer.h>
#include <graph_core.h>
#include <gtest/gtest.h>
#include <ini_wrapper.h>
#include <input_layer.h>
#include <layer_node.h>
#include <multiout_layer.h>
#include <neuralnet.h>
#include <util_func.h>

#include "nntrainer_test_util.h"

using LayerRepresentation = std::pair<std::string, std::vector<std::string>>;
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using ml::train::createLayer;

namespace initest {
typedef enum {
  LOAD = 1 << 0,   /**< should fail at load */
  INIT = 1 << 1,   /**< should fail at init */
  REINIT = 1 << 2, /**< should fail at reinit */
} IniFailAt;
};

template <typename T>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  std::mt19937 gen(42);
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

/**
 * @brief Graph Tester
 *
 */
class nntrainerGraphTest
  : public ::testing::TestWithParam<
      std::tuple<const char *, const nntrainer::IniWrapper::Sections, int>> {

protected:
  nntrainerGraphTest() : failAt(0), name("") {}
  virtual void SetUp() {
    name = std::string(std::get<0>(GetParam()));
    std::cout << "starting test case : " << name << std::endl << std::endl;

    const auto &sections = std::get<1>(GetParam());

    ini = nntrainer::IniWrapper(name, sections);

    failAt = std::get<2>(GetParam());
    ini.save_ini();
  }

  virtual void TearDown() { ini.erase_ini(); }

  std::string getIniName() { return ini.getIniName(); }

  bool failAtLoad() { return failAt & initest::IniFailAt::LOAD; }

  bool failAtInit() { return failAt & initest::IniFailAt::INIT; }

  bool failAtReinit() { return failAt & initest::IniFailAt::REINIT; }

  nntrainer::NeuralNetwork NN;

private:
  int failAt;
  std::string name;
  nntrainer::IniWrapper ini;
};

/**
 * @brief check given ini is failing/suceeding at load
 */
TEST_P(nntrainerGraphTest, loadConfig) {
  std::cout << std::get<0>(GetParam()) << std::endl;
  int status = NN.loadFromConfig(getIniName());

  int batch = 16;
  int channel = 3;
  int height = 32;
  int width = 32;

  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  status = NN.compile();

  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  status = NN.initialize();
  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  status = NN.reinitialize();
  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  status = NN.allocate();
  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  NN.forwarding({MAKE_SHARED_TENSOR(input)});

  nntrainer::Tensor output(batch, 1, 1, 10);

  output.setZero();

  for (int i = 0; i < batch; ++i)
    output.setValue(i, 0, 0, 3, 1.0);

  NN.forwarding({MAKE_SHARED_TENSOR(input)}, {MAKE_SHARED_TENSOR(output)});
  NN.backwarding(1);
}

static nntrainer::IniSection nw_base("model", "Type = NeuralNetwork | "
                                              "batch_size = 16 | "
                                              "loss = cross");

static nntrainer::IniSection sgd("Optimizer", "Type = sgd |"
                                              "Learning_rate = 1");

static nntrainer::IniSection input("inputlayer", "Type = input |"
                                                 "Input_Shape = 3:32:32");

static nntrainer::IniSection conv2d8("conv2d8", "Type = conv2d |"
                                                "input_layers=inputlayer |"
                                                "bias_initializer = zeros |"
                                                "Activation = relu |"
                                                "filters = 32 |"
                                                "kernel_size = 3,3 |"
                                                "stride = 1,1 |"
                                                "padding = 0,0");

static nntrainer::IniSection conv2d9("conv2d9", "Type = conv2d |"
                                                "input_layers=conv2d8 |"
                                                "bias_initializer = zeros |"
                                                "Activation = relu |"
                                                "filters = 64 |"
                                                "kernel_size = 3,3 |"
                                                "stride = 1,1 |"
                                                "padding = 0,0");

static nntrainer::IniSection pooling2("pooling2", "Type = pooling2d |"
                                                  "input_layers = conv2d9 |"
                                                  "pool_size = 3, 3 |"
                                                  "stride = 3, 3 |"
                                                  "padding = 0, 0 |"
                                                  "pooling=max");

static nntrainer::IniSection out0("out0", "Type = multiout |"
                                          "input_layers = pooling2");

static nntrainer::IniSection conv2d10("conv2d10", "Type = conv2d |"
                                                  "input_layers=out0 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu |"
                                                  "filters = 64 |"
                                                  "kernel_size = 3,3 |"
                                                  "stride = 1,1 |"
                                                  "padding = 1,1");

static nntrainer::IniSection conv2d11("conv2d11", "Type = conv2d |"
                                                  "input_layers=conv2d10 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu |"
                                                  "filters = 64 |"
                                                  "kernel_size = 3,3 |"
                                                  "stride = 1,1 |"
                                                  "padding = 1,1");

static nntrainer::IniSection addition0("addition0",
                                       "Type=addition |"
                                       "input_layers = conv2d11, out0 ");

static nntrainer::IniSection out1("out1", "Type = multiout |"
                                          "input_layers = addition0");

static nntrainer::IniSection conv2d12("conv2d12", "Type = conv2d |"
                                                  "input_layers=out1 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu |"
                                                  "filters = 64 |"
                                                  "kernel_size = 3,3 |"
                                                  "stride = 1,1 |"
                                                  "padding = 1,1");

static nntrainer::IniSection conv2d13("conv2d13", "Type = conv2d |"
                                                  "input_layers=conv2d12 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu |"
                                                  "filters = 64 |"
                                                  "kernel_size = 3,3 |"
                                                  "stride = 1,1 |"
                                                  "padding = 1,1");

static nntrainer::IniSection addition1("addition1",
                                       "Type=addition |"
                                       "input_layers = conv2d13, out1 ");

static nntrainer::IniSection conv2d14("conv2d14", "Type = conv2d |"
                                                  "input_layers=addition1 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu |"
                                                  "filters = 64 |"
                                                  "kernel_size = 3,3 |"
                                                  "stride = 1,1 |"
                                                  "padding = 0,0");

static nntrainer::IniSection
  pooling3("pooling3", "Type = pooling2d |"
                       "input_layers = conv2d14 |"
                       "pooling=global_average | flatten = true");

static nntrainer::IniSection fclayer0("fclayer0", "Type = fully_connected |"
                                                  "Unit = 256 |"
                                                  "input_layers = pooling3 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu");

static nntrainer::IniSection fclayer1("fclayer1", "Type = fully_connected |"
                                                  "Unit = 10 |"
                                                  "input_layers = fclayer0 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = softmax");

static int SUCCESS = 0;

/**
 * @brief make ini test case from given parameter
 */
std::tuple<const char *, const nntrainer::IniWrapper::Sections, int>
mkIniTc(const char *name, const nntrainer::IniWrapper::Sections vec, int flag) {
  return std::make_tuple(name, vec, flag);
}

GTEST_PARAMETER_TEST(nntrainerIniAutoTests, nntrainerGraphTest,
                     ::testing::Values(mkIniTc(
                       "basic_p",
                       {nw_base, sgd, input, conv2d8, conv2d9, pooling2, out0,
                        conv2d10, conv2d11, addition0, out1, conv2d12, conv2d13,
                        addition1, conv2d14, pooling3, fclayer0, fclayer1},
                       SUCCESS)));

TEST(nntrainerGraphUnitTest, cross_with_relu) {
  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=1:1:1"});
  auto relu0 = LayerRepresentation(
    "activation", {"name=relu0", "activation=relu", "input_layers=in0"});

  auto g = makeGraph({input0, relu0});

  nntrainer::NetworkGraph ng;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "cross")});

  for (auto &node : g) {
    EXPECT_NO_THROW(nn_model->addLayer(node));
  }

  EXPECT_NE(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_NE(nn_model->initialize(), ML_ERROR_NONE);
}

TEST(nntrainerGraphUnitTest, compile_twice) {
  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=1:1:1"});
  auto relu0 = LayerRepresentation(
    "activation", {"name=relu0", "activation=softmax", "input_layers=in0"});

  auto g = makeGraph({input0, relu0});

  nntrainer::NetworkGraph ng;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "cross")});

  for (auto &node : g) {
    EXPECT_NO_THROW(nn_model->addLayer(node));
  }

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  EXPECT_EQ(nn_model->setOptimizer(std::move(optimizer)), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->initialize(), ML_ERROR_NONE);
  try {
    nn_model->compile();
  } catch (const std::exception &e) {
    EXPECT_STREQ(e.what(), "cannot remap identifiers after finalized");
  }
}

TEST(nntrainerGraphUnitTest, call_functions) {
  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=1:1:1"});
  auto relu0 = LayerRepresentation(
    "activation", {"name=relu0", "activation=softmax", "input_layers=in0"});

  auto g = makeGraph({input0, relu0});

  nntrainer::NetworkGraph ng;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "cross")});

  for (auto &node : g) {
    EXPECT_NO_THROW(nn_model->addLayer(node));
  }

  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  try {
    for (auto &node : g) {
      nn_model->addLayer(node);
    }
  } catch (const std::exception &e) {
    EXPECT_STREQ(e.what(), "Cannot modify graph after compile");
  }
}

TEST(nntrainerGraphUnitTest, NoLossLayerWhenInferenceMode) {
  std::unique_ptr<ml::train::Model> model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  model->addLayer(ml::train::createLayer(
    "input", {nntrainer::withKey("name", "input0"),
              nntrainer::withKey("input_shape", "1:1:256")}));

  for (int i = 0; i < 3; ++i) {
    model->addLayer(ml::train::createLayer(
      "fully_connected",
      {nntrainer::withKey("unit", 1024),
       nntrainer::withKey("weight_initializer", "xavier_uniform"),
       nntrainer::withKey("bias_initializer", "zeros")}));
  }
  model->addLayer(ml::train::createLayer(
    "fully_connected",
    {nntrainer::withKey("unit", 100),
     nntrainer::withKey("weight_initializer", "xavier_uniform"),
     nntrainer::withKey("bias_initializer", "zeros")}));

  model->setProperty({nntrainer::withKey("batch_size", 1),
                      nntrainer::withKey("epochs", 1),
                      nntrainer::withKey("fsu", "false"),
                      nntrainer::withKey("model_tensor_type", "FP32-FP32")});

  int status = model->compile(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = model->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float input[256];

  for (unsigned int i = 0; i < 256; ++i) {
    input[i] = i;
  }

  std::vector<float *> in;
  std::vector<float *> ans;

  in.push_back(input);

  ans = model->inference(1, in);

  in.clear();
  ans.clear();
}

void check_sorted_graph(nntrainer::GraphCore graph,
                        std::vector<std::string> expected_layer_names) {
  int expected_graph_size = expected_layer_names.size();
  EXPECT_EQ(graph.size(), expected_graph_size);
  for (int i = 0; i < expected_graph_size; i++) {
    EXPECT_EQ(graph.getSortedNode(i)->getName(), expected_layer_names[i]);
    EXPECT_EQ(graph.getSortedNodeIdx(expected_layer_names[i]), i);
  }
}

TEST(nntrainerGraphUnitTest, topological_sort_simple) {
  nntrainer::GraphCore graph;

  auto input_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::InputLayer>());
  input_layer->setName("input");

  auto fc1_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc1_layer->setName("fc1");

  auto fc2_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc2_layer->setName("fc2");

  graph.addNode(input_layer);
  graph.addNode(fc1_layer);
  graph.addNode(fc2_layer);
  // Create simple linear graph: input -> fc1 -> fc2 -> output
  fc1_layer->setProperty({"input_layers=input"});
  fc2_layer->setProperty({"input_layers=fc1"});

  EXPECT_NO_THROW(graph.topologicalSort());
  check_sorted_graph(graph, {"input", "fc1", "fc2"});
}

TEST(nntrainerGraphUnitTest, topological_sort_branching) {
  nntrainer::GraphCore graph;

  auto input_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::InputLayer>());
  input_layer->setName("input");

  auto fc1_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc1_layer->setName("fc1");

  auto fc2_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc2_layer->setName("fc2");

  auto fc3_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc3_layer->setName("fc3");

  auto fc4_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc4_layer->setName("fc4");

  graph.addNode(input_layer);
  graph.addNode(fc1_layer);
  graph.addNode(fc2_layer);
  graph.addNode(fc3_layer);
  graph.addNode(fc4_layer);

  // input -> fc1 -> fc4
  // input (branch) -> fc3 -> fc2
  fc1_layer->setProperty({"input_layers=input"});
  fc2_layer->setProperty({"input_layers=fc3"});
  fc3_layer->setProperty({"input_layers=input"});
  fc4_layer->setProperty({"input_layers=fc1"});

  EXPECT_NO_THROW(graph.topologicalSort());
  check_sorted_graph(graph, {"input", "fc1", "fc3", "fc2", "fc4"});
}

TEST(nntrainerGraphUnitTest, topological_sort_non_connected) {
  nntrainer::GraphCore graph;

  auto input1_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::InputLayer>());
  input1_layer->setName("input1");

  auto input2_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::InputLayer>());
  input2_layer->setName("input2");

  auto fc1_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc1_layer->setName("fc1");

  auto fc2_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc2_layer->setName("fc2");

  graph.addNode(input1_layer);
  graph.addNode(input2_layer);
  graph.addNode(fc1_layer);
  graph.addNode(fc2_layer);

  // input1 -> fc1
  // input2 -> fc2
  fc1_layer->setProperty({"input_layers=input1"});
  fc2_layer->setProperty({"input_layers=input2"});

  EXPECT_NO_THROW(graph.topologicalSort());
  check_sorted_graph(graph, {"input1", "input2", "fc1", "fc2"});
}

TEST(nntrainerGraphUnitTest, topological_sort_cycle_detection) {
  nntrainer::GraphCore graph;

  auto fc1_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc1_layer->setName("fc1");

  auto fc2_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc2_layer->setName("fc2");

  graph.addNode(fc1_layer);
  graph.addNode(fc2_layer);

  // fc1 <-> fc2
  fc1_layer->setProperty({"input_layers=fc2"});
  fc2_layer->setProperty({"input_layers=fc1"});

  // throw if there is cycle?
  EXPECT_NO_THROW(graph.topologicalSort());
  check_sorted_graph(graph, {"fc2", "fc1"});
}

TEST(nntrainerGraphUnitTest, topological_sort_multiple_inputs) {
  nntrainer::GraphCore graph;

  auto input1_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::InputLayer>());
  input1_layer->setName("input1");

  auto input2_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::InputLayer>());
  input2_layer->setName("input2");

  auto input3_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::InputLayer>());
  input3_layer->setName("input3");

  auto fc_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc_layer->setName("fc");

  graph.addNode(input1_layer);
  graph.addNode(input2_layer);
  graph.addNode(input3_layer);
  graph.addNode(fc_layer);

  // {input1, input2, input3} -> fc
  fc_layer->setProperty({"input_layers=input1,input2,input3"});

  EXPECT_NO_THROW(graph.topologicalSort());
  check_sorted_graph(graph, {"input1", "input2", "input3", "fc"});
}

TEST(nntrainerGraphUnitTest, topological_sort_multiple_outputs) {
  nntrainer::GraphCore graph;

  auto input_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::InputLayer>());
  input_layer->setName("input");

  auto multiout_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::MultiOutLayer>());
  multiout_layer->setName("multiout");

  auto fc1_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc1_layer->setName("fc1");

  auto fc2_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc2_layer->setName("fc2");

  auto fc3_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc3_layer->setName("fc3");

  auto fc4_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc4_layer->setName("fc4");

  graph.addNode(input_layer);
  graph.addNode(multiout_layer);
  graph.addNode(fc1_layer);
  graph.addNode(fc2_layer);
  graph.addNode(fc3_layer);
  graph.addNode(fc4_layer);

  // input -> multiout -> {fc1 -> fc3, fc2, fc4}
  multiout_layer->setProperty({"input_layers=input"});
  fc1_layer->setProperty({"input_layers=multiout"});
  fc2_layer->setProperty({"input_layers=multiout"});
  fc3_layer->setProperty({"input_layers=fc1"});
  fc4_layer->setProperty({"input_layers=multiout"});

  EXPECT_NO_THROW(graph.topologicalSort());
  check_sorted_graph(graph, {"input", "multiout", "fc1", "fc2", "fc3", "fc4"});
}

TEST(nntrainerGraphUnitTest, topological_sort_complex_multi_output) {
  nntrainer::GraphCore graph;

  auto input_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::InputLayer>());
  input_layer->setName("input");

  auto multiout1_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::MultiOutLayer>());
  multiout1_layer->setName("multiout1");

  auto multiout2_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::MultiOutLayer>());
  multiout2_layer->setName("multiout2");

  auto fc1_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc1_layer->setName("fc1");

  auto fc2_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc2_layer->setName("fc2");

  auto fc3_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc3_layer->setName("fc3");

  auto fc4_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc4_layer->setName("fc4");

  auto fc5_layer = std::make_shared<nntrainer::LayerNode>(
    std::make_unique<nntrainer::FullyConnectedLayer>());
  fc5_layer->setName("fc5");

  graph.addNode(input_layer);
  graph.addNode(fc1_layer);
  graph.addNode(fc2_layer);
  graph.addNode(fc3_layer);
  graph.addNode(fc4_layer);
  graph.addNode(fc5_layer);
  graph.addNode(multiout1_layer);
  graph.addNode(multiout2_layer);

  // input -> {multiout1, multiout2}
  // multiout1 -> {fc1, fc2}
  // multiout2 -> {fc4, fc5}
  // {fc2, fc4} -> fc3
  multiout1_layer->setProperty({"input_layers=input"});
  multiout2_layer->setProperty({"input_layers=input"});
  fc1_layer->setProperty({"input_layers=multiout1"});
  fc2_layer->setProperty({"input_layers=multiout1"});
  fc3_layer->setProperty({"input_layers=fc2,fc4"});
  fc5_layer->setProperty({"input_layers=multiout2"});
  fc4_layer->setProperty({"input_layers=multiout2"});

  EXPECT_NO_THROW(graph.topologicalSort());
  check_sorted_graph(graph, {"input", "multiout1", "fc1", "fc2", "multiout2",
                             "fc4", "fc3", "fc5"});
}

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
