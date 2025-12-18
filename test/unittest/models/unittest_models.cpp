// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_models.cpp
 * @date 25 Nov 2021
 * @brief unittest models for v2 version
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <ini_wrapper.h>
#include <memory>
#include <neuralnet.h>
#include <nntrainer_test_util.h>

#include <models_golden_test.h>

using namespace nntrainer;

static inline constexpr const int NOT_USED_ = 1;

static IniSection nn_base("model", "type = NeuralNetwork");
static std::string fc_base = "type = Fully_connected";
static std::string red_mean_base = "type = reduce_mean";
static IniSection sgd_base("optimizer", "Type = sgd");
static IniSection constant_loss("loss", "type = constant_derivative");
static IniSection act_base("activation", "Type = Activation");

IniWrapper reduce_mean_last("reduce_mean_last",
                            {
                              nn_base + "batch_size=3",
                              sgd_base + "learning_rate=0.1",
                              IniSection("fc_1") + fc_base +
                                "unit=7 | input_shape=1:1:2",
                              IniSection("red_mean") + red_mean_base + "axis=3",
                              constant_loss,
                            });

IniWrapper fc_relu_decay(
  "fc_relu_decay",
  {nn_base + "Loss=mse | batch_size = 3", sgd_base + "learning_rate = 0.1",
   IniSection("input") + "type=input" + "input_shape = 1:1:3",
   IniSection("dense") + fc_base + "unit = 10" + "weight_decay=0.9",
   IniSection("act") + act_base + "Activation = relu",
   IniSection("dense_1") + fc_base + "unit = 2" + "bias_decay=0.9",
   IniSection("act_1") + act_base + "Activation = sigmoid"});

/**
 * @brief get function to make model with non-trainable fc layer
 * @param[in] idx index of the fc layer to be non-trainable
 * @retval function to make model with non-trainable fc layer
 */
std::function<std::unique_ptr<NeuralNetwork>()>
getFuncToMakeNonTrainableFc(int idx) {

  std::string fc1_trainable = (idx == 1) ? "trainable=false" : "trainable=true";
  std::string fc2_trainable = (idx == 2) ? "trainable=false" : "trainable=true";
  std::string fc3_trainable = (idx == 3) ? "trainable=false" : "trainable=true";

  return [fc1_trainable, fc2_trainable, fc3_trainable]() {
    std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

    nn->setProperty({"batch_size=3"});

    auto outer_graph = makeGraph({
      {"input", {"name=in", "input_shape=1:1:3"}},
      {"fully_connected",
       {"name=fc1", "input_layers=in", "unit=10", "activation=relu",
        fc1_trainable}},
      {"fully_connected",
       {"name=fc2", "input_layers=fc1", "unit=10", "activation=relu",
        fc2_trainable}},
      {"fully_connected",
       {"name=fc3", "input_layers=fc2", "unit=2", "activation=sigmoid",
        fc3_trainable}},
      {"mse", {"name=loss", "input_layers=fc3"}},
    });

    for (auto &node : outer_graph) {
      nn->addLayer(node);
    }

    nn->setOptimizer(
      ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
    nn->setProperty({"input_layers=in", "label_layers=loss"});

    return nn;
  };
}

static auto makeNonTrainableFcIdx1 = getFuncToMakeNonTrainableFc(1);
static auto makeNonTrainableFcIdx2 = getFuncToMakeNonTrainableFc(2);
static auto makeNonTrainableFcIdx3 = getFuncToMakeNonTrainableFc(3);

static std::unique_ptr<NeuralNetwork> makeMolAttention() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=in3", "input_shape=1:1:5"}},
    {"input", {"name=in2", "input_shape=1:4:6"}},
    {"input", {"name=in1", "input_shape=1:1:6"}},
    {"mol_attention",
     {"name=mol", "input_layers=in1,in2,in3", "unit=8", "mol_k=5"}},
    {"constant_derivative", {"name=loss1", "input_layers=mol(0)"}},
    {"constant_derivative", {"name=loss2", "input_layers=mol(1)"}},
  });

  nn->setProperty({"label_layers=loss1,loss2"});
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMolAttentionMasked() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=in4", "input_shape=1:1:1"}},
    {"input", {"name=in3", "input_shape=1:1:5"}},
    {"input", {"name=in2", "input_shape=1:4:6"}},
    {"input", {"name=in1", "input_shape=1:1:6"}},
    {"mol_attention",
     {"name=mol", "input_layers=in1,in2,in3,in4", "unit=8", "mol_k=5"}},
    {"constant_derivative", {"name=loss1", "input_layers=mol(0)"}},
    {"constant_derivative", {"name=loss2", "input_layers=mol(1)"}},
  });

  nn->setProperty({"label_layers=loss1,loss2"});
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork>
makeMultiHeadAttention_disable_need_weights() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:3:6"}},
    {"input", {"name=input_1", "input_shape=1:2:6"}},
    {"input", {"name=input_2", "input_shape=1:2:6"}},
    {"multi_head_attention",
     {"name=multi_head_attention", "input_layers=input_0, input_1, input_2",
      "disable_bias=true", "num_heads=2"}},
    {"mse", {"name=loss", "input_layers=multi_head_attention"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty({"input_layers=input_0, input_1, input_2"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMultiHeadAttention() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:3:6"}},
    {"input", {"name=input_1", "input_shape=1:2:6"}},
    {"input", {"name=input_2", "input_shape=1:2:6"}},
    {"multi_head_attention",
     {"name=multi_head_attention", "input_layers=input_0, input_1, input_2",
      "num_heads=2", "return_attention_weight=after"}},
    {"mse", {"name=loss1", "input_layers=multi_head_attention(0)"}},
    {"mse", {"name=loss2", "input_layers=multi_head_attention(1)"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty(
    {"input_layers=input_0, input_1, input_2", "label_layers=loss1, loss2"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMultiHeadAttention_kdim_vdim() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:3:6"}},
    {"input", {"name=input_1", "input_shape=1:2:4"}},
    {"input", {"name=input_2", "input_shape=1:2:5"}},
    {"multi_head_attention",
     {"name=multi_head_attention", "input_layers=input_0, input_1, input_2",
      "num_heads=2", "return_attention_weight=after"}},
    {"mse", {"name=loss1", "input_layers=multi_head_attention(0)"}},
    {"mse", {"name=loss2", "input_layers=multi_head_attention(1)"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty(
    {"input_layers=input_0, input_1, input_2", "label_layers=loss1, loss2"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMultiHeadAttention_float_attn_mask() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:3:6"}},
    {"input", {"name=input_1", "input_shape=1:2:6"}},
    {"input", {"name=input_2", "input_shape=1:2:6"}},
    {"input", {"name=input_3", "input_shape=2:3:2"}},
    {"multi_head_attention",
     {"name=multi_head_attention",
      "input_layers=input_0, input_1, input_2, input_3", "num_heads=2",
      "return_attention_weight=after"}},
    {"mse", {"name=loss1", "input_layers=multi_head_attention(0)"}},
    {"mse", {"name=loss2", "input_layers=multi_head_attention(1)"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty({"input_layers=input_0, input_1, input_2, input_3",
                   "label_layers=loss1, loss2"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMultiHeadAttention_self_attention() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:3:6"}},
    {"multi_head_attention",
     {"name=multi_head_attention", "input_layers=input_0, input_0, input_0",
      "num_heads=2", "return_attention_weight=after"}},
    {"mse", {"name=loss1", "input_layers=multi_head_attention(0)"}},
    {"mse", {"name=loss2", "input_layers=multi_head_attention(1)"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty({"input_layers=input_0", "label_layers=loss1, loss2"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makePositionalEncoding() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=5:1:6"}},
    {"reshape", {"name=reshape", "target_shape=1:5:6"}},
    {"positional_encoding", {"name=positional_encoding", "max_timestep=7"}},
    {"multi_head_attention",
     {"name=multi_head_attention",
      "input_layers=positional_encoding, positional_encoding, "
      "positional_encoding",
      "num_heads=2"}},
    {"mse", {"name=loss", "input_layers=multi_head_attention(0)"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeTransformerEncoderLayer() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:5:6"}},
    {"multi_head_attention",
     {"name=multi_head_attention", "input_layers=input_0, input_0, input_0",
      "num_heads=2"}},
    {"addition", {"name=add1", "input_layers=input_0, multi_head_attention"}},
    {"layer_normalization", {"name=ln1", "axis=3", "epsilon=1e-5"}},
    {"fully_connected", {"name=fc1", "unit=7", "activation=relu"}},
    {"fully_connected", {"name=fc2", "unit=6"}},
    {"addition", {"name=add2", "input_layers=ln1, fc2"}},
    {"layer_normalization", {"name=ln2", "axis=3", "epsilon=1e-5"}},
    {"mse", {"name=loss", "input_layers=ln2"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty({"input_layers=input_0", "label_layers=loss"});

  return nn;
}

static std::unique_ptr<NeuralNetwork>
makeTransformerEncoderLayer_float_attn_mask() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:5:6"}},
    {"input", {"name=input_1", "input_shape=2:5:5"}},
    {"multi_head_attention",
     {"name=multi_head_attention",
      "input_layers=input_0, input_0, input_0, input_1", "num_heads=2"}},
    {"addition", {"name=add1", "input_layers=input_0, multi_head_attention"}},
    {"layer_normalization", {"name=ln1", "axis=3", "epsilon=1e-5"}},
    {"fully_connected", {"name=fc1", "unit=7", "activation=relu"}},
    {"fully_connected", {"name=fc2", "unit=6"}},
    {"addition", {"name=add2", "input_layers=ln1, fc2"}},
    {"layer_normalization", {"name=ln2", "axis=3", "epsilon=1e-5"}},
    {"mse", {"name=loss", "input_layers=ln2"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty({"input_layers=input_0, input_1", "label_layers=loss"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeTransformerDecoderLayer() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:5:6"}},
    {"input", {"name=input_1", "input_shape=1:4:6"}},
    {"multi_head_attention",
     {"name=masked_multi_head_attention",
      "input_layers=input_0, input_0, input_0", "num_heads=2"}},
    {"addition",
     {"name=add1", "input_layers=input_0, masked_multi_head_attention"}},
    {"layer_normalization", {"name=ln1", "axis=3", "epsilon=1e-5"}},
    {"multi_head_attention",
     {"name=multi_head_attention", "input_layers=ln1, input_1, input_1",
      "num_heads=2"}},
    {"addition", {"name=add2", "input_layers=ln1, multi_head_attention"}},
    {"layer_normalization", {"name=ln2", "axis=3", "epsilon=1e-5"}},
    {"fully_connected", {"name=fc1", "unit=7", "activation=relu"}},
    {"fully_connected", {"name=fc2", "unit=6"}},
    {"addition", {"name=add3", "input_layers=ln2, fc2"}},
    {"layer_normalization", {"name=ln3", "axis=3", "epsilon=1e-5"}},
    {"mse", {"name=loss", "input_layers=ln3"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty({"input_layers=input_0, input_1", "label_layers=loss"});

  return nn;
}

static std::unique_ptr<NeuralNetwork>
makeTransformerDecoderLayer_float_attn_mask() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:5:6"}},
    {"input", {"name=input_1", "input_shape=1:4:6"}},
    {"input", {"name=input_2", "input_shape=2:5:5"}},
    {"input", {"name=input_3", "input_shape=2:5:4"}},
    {"multi_head_attention",
     {"name=masked_multi_head_attention",
      "input_layers=input_0, input_0, input_0, input_2", "num_heads=2"}},
    {"addition",
     {"name=add1", "input_layers=input_0, masked_multi_head_attention"}},
    {"layer_normalization", {"name=ln1", "axis=3", "epsilon=1e-5"}},
    {"multi_head_attention",
     {"name=multi_head_attention",
      "input_layers=ln1, input_1, input_1, input_3", "num_heads=2"}},
    {"addition", {"name=add2", "input_layers=ln1, multi_head_attention"}},
    {"layer_normalization", {"name=ln2", "axis=3", "epsilon=1e-5"}},
    {"fully_connected", {"name=fc1", "unit=7", "activation=relu"}},
    {"fully_connected", {"name=fc2", "unit=6"}},
    {"addition", {"name=add3", "input_layers=ln2, fc2"}},
    {"layer_normalization", {"name=ln3", "axis=3", "epsilon=1e-5"}},
    {"mse", {"name=loss", "input_layers=ln3"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty(
    {"input_layers=input_0, input_1, input_2, input_3", "label_layers=loss"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeTransformer_single_layer() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto decoder_input = makeGraph({
    {"input", {"name=decoder_input", "input_shape=1:4:6"}},
  });

  for (auto &node : decoder_input) {
    nn->addLayer(node);
  }

  auto decoder_layer = makeGraph({
    {"multiout", {"name=decoder_layer1/multi_out1"}},
    {"multi_head_attention",
     {"name=decoder_layer1/masked_multi_head_attention",
      "input_layers=decoder_layer1/multi_out1(0), "
      "decoder_layer1/multi_out1(1), decoder_layer1/multi_out1(2)",
      "num_heads=2"}},
    {"addition",
     {"name=decoder_layer1/add1",
      "input_layers=decoder_layer1/multi_out1(3), "
      "decoder_layer1/masked_multi_head_attention"}},
    {"layer_normalization",
     {"name=decoder_layer1/ln1", "axis=3", "epsilon=1e-5"}},
    {"multiout", {"name=decoder_layer1/multi_out2"}},
    {"multi_head_attention",
     {"name=decoder_layer1/multi_head_attention",
      "input_layers=decoder_layer1/multi_out2(0), encoder_output(0), "
      "encoder_output(1)",
      "num_heads=2"}},
    {"addition",
     {"name=decoder_layer1/add2", "input_layers=decoder_layer1/multi_out2(1), "
                                  "decoder_layer1/multi_head_attention"}},
    {"layer_normalization",
     {"name=decoder_layer1/ln2", "axis=3", "epsilon=1e-5"}},
    {"multiout", {"name=decoder_layer1/multi_out3"}},
    {"fully_connected",
     {"name=decoder_layer1/fc1", "input_layers=decoder_layer1/multi_out3(0)",
      "unit=7", "activation=relu"}},
    {"fully_connected", {"name=decoder_layer1/fc2", "unit=6"}},
    {"addition",
     {"name=add3",
      "input_layers=decoder_layer1/multi_out3(1), decoder_layer1/fc2"}},
    {"layer_normalization",
     {"name=decoder_layer1/ln3", "axis=3", "epsilon=1e-5"}},
  });

  for (auto &node : decoder_layer) {
    nn->addLayer(node);
  }

  auto decoder_output = makeGraph({
    {"layer_normalization",
     {"name=decoder_layer_normalization", "axis=3", "epsilon=1e-5"}},
    {"mse", {"name=loss"}},
  });

  for (auto &node : decoder_output) {
    nn->addLayer(node);
  }

  auto encoder_input = makeGraph({
    {"input", {"name=encoder_input", "input_shape=1:5:6"}},
  });

  for (auto &node : encoder_input) {
    nn->addLayer(node);
  }

  auto encoder = makeGraph({
    {"multiout", {"name=encoder_layer1/multi_out1"}},
    {"multi_head_attention",
     {"name=encoder_layer1/multi_head_attention",
      "input_layers=encoder_layer1/multi_out1(0), "
      "encoder_layer1/multi_out1(1), encoder_layer1/multi_out1(2)",
      "num_heads=2"}},
    {"addition",
     {"name=encoder_layer1/add1", "input_layers=encoder_layer1/multi_out1(3), "
                                  "encoder_layer1/multi_head_attention"}},
    {"layer_normalization",
     {"name=encoder_layer1/ln1", "axis=3", "epsilon=1e-5"}},
    {"multiout", {"name=encoder_layer1/multi_out2"}},
    {"fully_connected",
     {"name=encoder_layer1/fc1", "input_layers=encoder_layer1/multi_out2(0)",
      "unit=7", "activation=relu"}},
    {"fully_connected", {"name=encoder_layer1/fc2", "unit=6"}},
    {"addition",
     {"name=add2",
      "input_layers=encoder_layer1/multi_out2(1), encoder_layer1/fc2"}},
    {"layer_normalization", {"name=ln2", "axis=3", "epsilon=1e-5"}},
  });

  for (auto &node : encoder) {
    nn->addLayer(node);
  }

  auto encoder_output = makeGraph({
    {"layer_normalization",
     {"name=encoder_layer_normalization", "axis=3", "epsilon=1e-5"}},
    {"multiout", {"name=encoder_output"}},
  });

  for (auto &node : encoder_output) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty(
    {"input_layers=encoder_input, decoder_input", "label_layers=loss"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeTransformer_stack_layer() {
  const unsigned int num_encoder_layer = 2;
  const unsigned int num_decoder_layer = 2;
  const unsigned int batch_size = 3;
  const unsigned int num_heads = 2;
  const unsigned int encoder_timestep = 5;
  const unsigned int decoder_timestep = 4;
  const unsigned int model_dim = 6;
  const unsigned int fc_unit = 7;

  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=" + std::to_string(batch_size)});

  auto decoder_input = makeGraph({
    {"input",
     {"name=decoder_input",
      "input_shape=1:" + std::to_string(decoder_timestep) + ":" +
        std::to_string(model_dim)}},
  });

  for (auto &node : decoder_input) {
    nn->addLayer(node);
  }

  for (unsigned int i = 0; i < num_decoder_layer; ++i) {
    auto decoder_layer = makeGraph({
      {"multiout", {"name=decoder_layer" + std::to_string(i) + "/multi_out1"}},
      {"multi_head_attention",
       {"name=decoder_layer" + std::to_string(i) +
          "/masked_multi_head_attention",
        "input_layers=decoder_layer" + std::to_string(i) +
          "/multi_out1(0), decoder_layer" + std::to_string(i) +
          "/multi_out1(1), decoder_layer" + std::to_string(i) +
          "/multi_out1(2)",
        "num_heads=" + std::to_string(num_heads)}},
      {"addition",
       {"name=decoder_layer" + std::to_string(i) + "/add1",
        "input_layers=decoder_layer" + std::to_string(i) +
          "/multi_out1(3), decoder_layer" + std::to_string(i) +
          "/masked_multi_head_attention"}},
      {"layer_normalization",
       {"name=decoder_layer" + std::to_string(i) + "/ln1", "axis=3",
        "epsilon=1e-5"}},
      {"multiout", {"name=decoder_layer" + std::to_string(i) + "/multi_out2"}},
      {"multi_head_attention",
       {"name=decoder_layer" + std::to_string(i) + "/multi_head_attention",
        "input_layers=decoder_layer" + std::to_string(i) +
          "/multi_out2(0), encoder_output(0), encoder_output(1)",
        "num_heads=" + std::to_string(num_heads)}},
      {"addition",
       {"name=decoder_layer" + std::to_string(i) + "/add2",
        "input_layers=decoder_layer" + std::to_string(i) +
          "/multi_out2(1), decoder_layer" + std::to_string(i) +
          "/multi_head_attention"}},
      {"layer_normalization",
       {"name=decoder_layer" + std::to_string(i) + "/ln2", "axis=3",
        "epsilon=1e-5"}},
      {"multiout", {"name=decoder_layer" + std::to_string(i) + "/multi_out3"}},
      {"fully_connected",
       {"name=decoder_layer" + std::to_string(i) + "/fc1",
        "input_layers=decoder_layer" + std::to_string(i) + "/multi_out3(0)",
        "unit=" + std::to_string(fc_unit), "activation=relu"}},
      {"fully_connected",
       {"name=decoder_layer" + std::to_string(i) + "/fc2",
        "unit=" + std::to_string(model_dim)}},
      {"addition",
       {"name=decoder_layer" + std::to_string(i) + "/add3",
        "input_layers=decoder_layer" + std::to_string(i) +
          "/multi_out3(1), decoder_layer" + std::to_string(i) + "/fc2"}},
      {"layer_normalization",
       {"name=decoder_layer" + std::to_string(i) + "/ln3", "axis=3",
        "epsilon=1e-5"}},
    });

    for (auto &node : decoder_layer) {
      nn->addLayer(node);
    }
  }

  auto decoder_output = makeGraph({
    {"layer_normalization",
     {"name=decoder_layer_normalization", "axis=3", "epsilon=1e-5"}},
    {"mse", {"name=loss"}},
  });

  for (auto &node : decoder_output) {
    nn->addLayer(node);
  }

  auto encoder_input = makeGraph({
    {"input",
     {"name=encoder_input",
      "input_shape=1:" + std::to_string(encoder_timestep) + ":" +
        std::to_string(model_dim)}},
  });

  for (auto &node : encoder_input) {
    nn->addLayer(node);
  }

  for (unsigned int i = 0; i < num_encoder_layer; ++i) {
    auto encoder_layer = makeGraph({
      {"multiout", {"name=encoder_layer" + std::to_string(i) + "/multi_out1"}},
      {"multi_head_attention",
       {"name=encoder_layer" + std::to_string(i) + "/multi_head_attention",
        "input_layers=encoder_layer" + std::to_string(i) +
          "/multi_out1(0), encoder_layer" + std::to_string(i) +
          "/multi_out1(1), encoder_layer" + std::to_string(i) +
          "/multi_out1(2)",
        "num_heads=" + std::to_string(num_heads)}},
      {"addition",
       {"name=encoder_layer" + std::to_string(i) + "/add1",
        "input_layers=encoder_layer" + std::to_string(i) +
          "/multi_out1(3), encoder_layer" + std::to_string(i) +
          "/multi_head_attention"}},
      {"layer_normalization",
       {"name=encoder_layer" + std::to_string(i) + "/ln1", "axis=3",
        "epsilon=1e-5"}},
      {"multiout", {"name=encoder_layer" + std::to_string(i) + "/multi_out2"}},
      {"fully_connected",
       {"name=encoder_layer" + std::to_string(i) + "/fc1",
        "input_layers=encoder_layer" + std::to_string(i) + "/multi_out2(0)",
        "unit=" + std::to_string(fc_unit), "activation=relu"}},
      {"fully_connected",
       {"name=encoder_layer" + std::to_string(i) + "/fc2",
        "unit=" + std::to_string(model_dim)}},
      {"addition",
       {"name=encoder_layer" + std::to_string(i) + "/add2",
        "input_layers=encoder_layer" + std::to_string(i) +
          "/multi_out2(1), encoder_layer" + std::to_string(i) + "/fc2"}},
      {"layer_normalization",
       {"name=encoder_layer" + std::to_string(i) + "/ln2", "axis=3",
        "epsilon=1e-5"}},
    });

    for (auto &node : encoder_layer) {
      nn->addLayer(node);
    }
  }

  auto encoder_output = makeGraph({
    {"layer_normalization",
     {"name=encoder_layer_normalization", "axis=3", "epsilon=1e-5"}},
    {"multiout", {"name=encoder_output"}},
  });

  for (auto &node : encoder_output) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty(
    {"input_layers=encoder_input, decoder_input", "label_layers=loss"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeTransformer_float_attn_mask() {
  const unsigned int num_encoder_layer = 2;
  const unsigned int num_decoder_layer = 2;
  const unsigned int batch_size = 3;
  const unsigned int num_heads = 2;
  const unsigned int encoder_timestep = 5;
  const unsigned int decoder_timestep = 4;
  const unsigned int model_dim = 6;
  const unsigned int fc_unit = 7;

  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=" + std::to_string(batch_size)});

  auto mask_input = makeGraph({
    {"input",
     {"name=memory_mask", "input_shape=" + std::to_string(num_heads) + ":" +
                            std::to_string(decoder_timestep) + ":" +
                            std::to_string(encoder_timestep)}},
    {"input",
     {"name=tgt_mask", "input_shape=" + std::to_string(num_heads) + ":" +
                         std::to_string(decoder_timestep) + ":" +
                         std::to_string(decoder_timestep)}},
    {"input",
     {"name=src_mask", "input_shape=" + std::to_string(num_heads) + ":" +
                         std::to_string(encoder_timestep) + ":" +
                         std::to_string(encoder_timestep)}},
  });

  for (auto &node : mask_input) {
    nn->addLayer(node);
  }

  auto decoder_input = makeGraph({
    {"input",
     {"name=decoder_input",
      "input_shape=1:" + std::to_string(decoder_timestep) + ":" +
        std::to_string(model_dim)}},
  });

  for (auto &node : decoder_input) {
    nn->addLayer(node);
  }

  for (unsigned int i = 0; i < num_decoder_layer; ++i) {
    auto decoder_layer = makeGraph({
      {"multiout", {"name=decoder_layer" + std::to_string(i) + "/multi_out1"}},
      {"multi_head_attention",
       {"name=decoder_layer" + std::to_string(i) +
          "/masked_multi_head_attention",
        "input_layers=decoder_layer" + std::to_string(i) +
          "/multi_out1(0), decoder_layer" + std::to_string(i) +
          "/multi_out1(1), decoder_layer" + std::to_string(i) +
          "/multi_out1(2), tgt_mask",
        "num_heads=" + std::to_string(num_heads)}},
      {"addition",
       {"name=decoder_layer" + std::to_string(i) + "/add1",
        "input_layers=decoder_layer" + std::to_string(i) +
          "/multi_out1(3), decoder_layer" + std::to_string(i) +
          "/masked_multi_head_attention"}},
      {"layer_normalization",
       {"name=decoder_layer" + std::to_string(i) + "/ln1", "axis=3",
        "epsilon=1e-5"}},
      {"multiout", {"name=decoder_layer" + std::to_string(i) + "/multi_out2"}},
      {"multi_head_attention",
       {"name=decoder_layer" + std::to_string(i) + "/multi_head_attention",
        "input_layers=decoder_layer" + std::to_string(i) +
          "/multi_out2(0), encoder_output(0), encoder_output(1), memory_mask",
        "num_heads=" + std::to_string(num_heads)}},
      {"addition",
       {"name=decoder_layer" + std::to_string(i) + "/add2",
        "input_layers=decoder_layer" + std::to_string(i) +
          "/multi_out2(1), decoder_layer" + std::to_string(i) +
          "/multi_head_attention"}},
      {"layer_normalization",
       {"name=decoder_layer" + std::to_string(i) + "/ln2", "axis=3",
        "epsilon=1e-5"}},
      {"multiout", {"name=decoder_layer" + std::to_string(i) + "/multi_out3"}},
      {"fully_connected",
       {"name=decoder_layer" + std::to_string(i) + "/fc1",
        "input_layers=decoder_layer" + std::to_string(i) + "/multi_out3(0)",
        "unit=" + std::to_string(fc_unit), "activation=relu"}},
      {"fully_connected",
       {"name=decoder_layer" + std::to_string(i) + "/fc2",
        "unit=" + std::to_string(model_dim)}},
      {"addition",
       {"name=decoder_layer" + std::to_string(i) + "/add3",
        "input_layers=decoder_layer" + std::to_string(i) +
          "/multi_out3(1), decoder_layer" + std::to_string(i) + "/fc2"}},
      {"layer_normalization",
       {"name=decoder_layer" + std::to_string(i) + "/ln3", "axis=3",
        "epsilon=1e-5"}},
    });

    for (auto &node : decoder_layer) {
      nn->addLayer(node);
    }
  }

  auto decoder_output = makeGraph({
    {"layer_normalization",
     {"name=decoder_layer_normalization", "axis=3", "epsilon=1e-5"}},
    {"mse", {"name=loss"}},
  });

  for (auto &node : decoder_output) {
    nn->addLayer(node);
  }

  auto encoder_input = makeGraph({
    {"input", {"name=encoder_input", "input_shape=1:5:6"}},
  });

  for (auto &node : encoder_input) {
    nn->addLayer(node);
  }

  for (unsigned int i = 0; i < num_encoder_layer; ++i) {
    auto encoder_layer = makeGraph({
      {"multiout", {"name=encoder_layer" + std::to_string(i) + "/multi_out1"}},
      {"multi_head_attention",
       {"name=encoder_layer" + std::to_string(i) + "/multi_head_attention",
        "input_layers=encoder_layer" + std::to_string(i) +
          "/multi_out1(0), encoder_layer" + std::to_string(i) +
          "/multi_out1(1), encoder_layer" + std::to_string(i) +
          "/multi_out1(2), src_mask",
        "num_heads=" + std::to_string(num_heads)}},
      {"addition",
       {"name=encoder_layer" + std::to_string(i) + "/add1",
        "input_layers=encoder_layer" + std::to_string(i) +
          "/multi_out1(3), encoder_layer" + std::to_string(i) +
          "/multi_head_attention"}},
      {"layer_normalization",
       {"name=encoder_layer" + std::to_string(i) + "/ln1", "axis=3",
        "epsilon=1e-5"}},
      {"multiout", {"name=encoder_layer" + std::to_string(i) + "/multi_out2"}},
      {"fully_connected",
       {"name=encoder_layer" + std::to_string(i) + "/fc1",
        "input_layers=encoder_layer" + std::to_string(i) + "/multi_out2(0)",
        "unit==" + std::to_string(fc_unit), "activation=relu"}},
      {"fully_connected",
       {"name=encoder_layer" + std::to_string(i) + "/fc2",
        "unit=" + std::to_string(model_dim)}},
      {"addition",
       {"name=encoder_layer" + std::to_string(i) + "/add2",
        "input_layers=encoder_layer" + std::to_string(i) +
          "/multi_out2(1), encoder_layer" + std::to_string(i) + "/fc2"}},
      {"layer_normalization",
       {"name=encoder_layer" + std::to_string(i) + "/ln2", "axis=3",
        "epsilon=1e-5"}},
    });

    for (auto &node : encoder_layer) {
      nn->addLayer(node);
    }
  }

  auto encoder_output = makeGraph({
    {"layer_normalization",
     {"name=encoder_layer_normalization", "axis=3", "epsilon=1e-5"}},
    {"multiout", {"name=encoder_output"}},
  });

  for (auto &node : encoder_output) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty({"input_layers=encoder_input, decoder_input, src_mask, "
                   "tgt_mask, memory_mask",
                   "label_layers=loss"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeAddOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:1:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"add", {"name=add_layer", "input_layers=in,fc"}},
               {"mse", {"name=loss", "input_layers=add_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeSubtractOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:1:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"subtract", {"name=subtract_layer", "input_layers=in,fc"}},
               {"mse", {"name=loss", "input_layers=subtract_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMultiplyOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:1:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"multiply", {"name=multiply_layer", "input_layers=in,fc"}},
               {"mse", {"name=loss", "input_layers=multiply_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeDivideOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:1:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"divide", {"name=divide_layer", "input_layers=in,fc"}},
               {"mse", {"name=loss", "input_layers=divide_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makePowOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:1:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"pow", {"name=pow_layer", "exponent=3", "input_layers=fc"}},
               {"mse", {"name=loss", "input_layers=pow_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeSQRTOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:1:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"sqrt", {"name=sqrt_layer", "input_layers=fc"}},
               {"mse", {"name=loss", "input_layers=sqrt_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeNegOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:1:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"negative", {"name=negative_layer", "input_layers=fc"}},
               {"mse", {"name=loss", "input_layers=negative_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeReduceSumOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph = makeGraph(
    {{"input", {"name=in", "input_shape=1:3:2"}},
     {"fully_connected", {"name=fc", "unit=7", "input_layers=in"}},
     {"reduce_sum", {"name=reduce_sum_layer", "input_layers=fc", "axis=3"}},
     {"mse", {"name=loss", "input_layers=reduce_sum_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

std::unique_ptr<NeuralNetwork> makeCosineOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:1:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"cos", {"name=cosine_layer", "input_layers=fc"}},
               {"mse", {"name=loss", "input_layers=cosine_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeSineOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:1:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"sin", {"name=sine_layer", "input_layers=fc"}},
               {"mse", {"name=loss", "input_layers=sine_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

std::unique_ptr<NeuralNetwork> makeTangentOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:1:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"tan", {"name=tangent_layer", "input_layers=fc"}},
               {"mse", {"name=loss", "input_layers=tangent_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMatMulOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph =
    makeGraph({{"input", {"name=in", "input_shape=1:2:2"}},
               {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
               {"matmul", {"name=mm", "input_layers=in,fc"}},
               {"mse", {"name=loss", "input_layers=mm"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeChannelShuffleOperation() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

  auto outer_graph = makeGraph(
    {{"conv2d",
      {"name=conv", "input_shape=1:8:4:4", "filters=8", "kernel_size=1,1"}},
     {"channel_shuffle",
      {"name=channel_shuffle_layer", "split_number=4", "input_layers=conv"}},
     {"mse", {"name=loss", "input_layers=channel_shuffle_layer"}}});

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setProperty({"batch_size=1"});
  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeFCSGD() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=2"});

  auto outer_graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:3"}},
    {"fully_connected", {"name=fc", "unit=4"}},
    {"mse", {"name=loss"}},
  });

  nn->setProperty({"label_layers=loss"});
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate=0.001"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeFCAdam() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=2"});

  auto outer_graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:3"}},
    {"fully_connected", {"name=fc", "unit=4"}},
    {"mse", {"name=loss"}},
  });

  nn->setProperty({"label_layers=loss"});
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer(
    "adam", {"learning_rate=0.1", "beta1=0.9", "beta2=0.999", "epsilon=1e-8"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeFCAdamW() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=2"});

  auto outer_graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:3"}},
    {"fully_connected",
     {"name=fc", "unit=4", "weight_decay=0.01", "bias_decay=0.01"}},
    {"mse", {"name=loss"}},
  });

  nn->setProperty({"label_layers=loss"});
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(
    ml::train::createOptimizer("adamw", {"learning_rate=0.1", "beta1=0.9",
                                         "beta2=0.999", "epsilon=1e-8"}));
  return nn;
}

GTEST_PARAMETER_TEST(
  model, nntrainerModelTest,
  ::testing::ValuesIn({
    mkModelIniTc(reduce_mean_last, DIM_UNUSED, NOT_USED_,
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeMolAttention, "mol_attention",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeMolAttentionMasked, "mol_attention_masked",
                 ModelTestOption::COMPARE_RUN_V2),
    mkModelTc_V2(makeMultiHeadAttention_disable_need_weights,
                 "multi_head_attention_disable_need_weights",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeMultiHeadAttention, "multi_head_attention",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeMultiHeadAttention_kdim_vdim,
                 "multi_head_attention_kdim_vdim", ModelTestOption::ALL_V2),
    mkModelTc_V2(makeMultiHeadAttention_float_attn_mask,
                 "multi_head_attention_float_attn_mask",
                 ModelTestOption::ALL_V2),
    /** @todo:change model if bool type tensor is supported */
    //  This unit test was commented out because it didn't work properly and
    //  caused errors.
    //  mkModelTc_V2(makeMultiHeadAttention_float_attn_mask,
    //               "multi_head_attention_pseudo_bool_attn_mask",
    //               ModelTestOption::ALL_V2),
    mkModelTc_V2(makeMultiHeadAttention_self_attention,
                 "multi_head_attention_self_attention",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makePositionalEncoding, "positional_encoding",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeTransformerEncoderLayer, "transformer_encoder_layer",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeTransformerEncoderLayer_float_attn_mask,
                 "transformer_encoder_layer_float_attn_mask",
                 ModelTestOption::ALL_V2),
    /** @todo:change model if bool type tensor is supported */
    //  This unit test was commented out because it didn't work properly and
    //  caused errors.
    //  mkModelTc_V2(makeTransformerEncoderLayer_float_attn_mask,
    //  "transformer_encoder_layer_pseudo_bool_attn_mask",
    //  ModelTestOption::ALL_V2),
    mkModelTc_V2(makeTransformerDecoderLayer, "transformer_decoder_layer",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeTransformerDecoderLayer_float_attn_mask,
                 "transformer_decoder_layer_float_attn_mask",
                 ModelTestOption::ALL_V2),
    /** @todo:change model if bool type tensor is supported */
    //  This unit test was commented out because it didn't work properly and
    //  caused errors.
    //  mkModelTc_V2(makeTransformerDecoderLayer_float_attn_mask,
    //               "transformer_decoder_layer_pseudo_bool_attn_mask",
    //               ModelTestOption::ALL_V2),
    mkModelTc_V2(makeTransformer_single_layer, "transformer_single",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeTransformer_stack_layer, "transformer_stack",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeTransformer_float_attn_mask, "transformer_float_attn_mask",
                 ModelTestOption::ALL_V2),
    //  This unit test was commented out because it didn't work properly and
    //  caused errors.
    //  mkModelTc_V2(makeTransformer_float_attn_mask,
    //               "transformer_pseudo_bool_attn_mask",
    //               ModelTestOption::ALL_V2),
    mkModelIniTc(fc_relu_decay, DIM_UNUSED, NOT_USED_,
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeNonTrainableFcIdx1, "non_trainable_fc_idx1",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeNonTrainableFcIdx2, "non_trainable_fc_idx2",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeNonTrainableFcIdx3, "non_trainable_fc_idx3",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeAddOperation, "add_operation", ModelTestOption::ALL_V2),
    mkModelTc_V2(makeSubtractOperation, "subtract_operation",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeMultiplyOperation, "multiply_operation",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeDivideOperation, "divide_operation",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makePowOperation, "pow_operation", ModelTestOption::ALL_V2),
    mkModelTc_V2(makeSQRTOperation, "sqrt_operation", ModelTestOption::ALL_V2),
    mkModelTc_V2(makeNegOperation, "neg_operation", ModelTestOption::ALL_V2),
    mkModelTc_V2(makeReduceSumOperation, "reduce_sum_operation",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeSineOperation, "sine_operation", ModelTestOption::ALL_V2),
    mkModelTc_V2(makeCosineOperation, "cosine_operation",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeTangentOperation, "tangent_operation",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeMatMulOperation, "matmul_operation",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeChannelShuffleOperation, "channel_shuffle",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeFCSGD, "fc_sgd", ModelTestOption::ALL_V2),
    mkModelTc_V2(makeFCAdam, "fc_adam", ModelTestOption::ALL_V2),
    mkModelTc_V2(makeFCAdamW, "fc_adamw", ModelTestOption::ALL_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info)
    -> const auto & { return std::get<1>(info.param); });

#ifdef NDK_BUILD

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
#endif
