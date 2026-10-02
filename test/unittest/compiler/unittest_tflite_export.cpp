// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_tflite_export.cpp
 * @date 23 November 2023
 * @brief export test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <vector>

#include <app_context.h>
#include <flatten_realizer.h>
#include <layer.h>
#include <model.h>
#include <node_exporter.h>
#include <optimizer.h>
#include <realizer.h>
#include <stdlib.h>
#include <util_func.h>

#include <nntrainer_test_util.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tflite_interpreter.h>
#endif

using LayerRepresentation = std::pair<std::string, std::vector<std::string>>;
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using ml::train::createLayer;

std::vector<float> out;
std::vector<float> ans;
std::vector<float *> in_f;
std::vector<float *> l_f;

unsigned int seed = 0;

/**
 * @brief Run TF Lite model with given input_vector's data
 *
 * @param tf_file_name tflite file name
 * @param input_vector input data
 * @return std::vector<float> output of tflite
 */
std::vector<float> run_tflite(std::string tf_file_name,
                              std::vector<float> input_vector) {
  std::vector<float> ret_vector;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> tf_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(tf_file_name.c_str());

  EXPECT_NE(model, nullptr);
  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  EXPECT_NE(tf_interpreter, nullptr);

  EXPECT_EQ(tf_interpreter->AllocateTensors(), kTfLiteOk);

  auto &in_indices = tf_interpreter->inputs();
  float *tf_input = tf_interpreter->typed_input_tensor<float>(0);

  for (unsigned int i = 0; i < input_vector.size(); i++) {
    tf_input[i] = input_vector[i];
  }

  int status = tf_interpreter->Invoke();

  auto out_indices = tf_interpreter->outputs();
  auto num_outputs = out_indices.size();
  auto out_tensor = tf_interpreter->tensor(out_indices[0]);
  auto out_size = out_tensor->bytes / sizeof(float);

  float *tf_output = tf_interpreter->typed_output_tensor<float>(0);
  for (size_t idx = 0; idx < out_size; idx++) {
    ret_vector.push_back(tf_output[idx]);
  }
  EXPECT_EQ(status, TfLiteStatus::kTfLiteOk);

  return ret_vector;
}

void data_clear() {
  out.clear();
  ans.clear();
  in_f.clear();
  l_f.clear();
}

/**
 * @brief Simple Fully Connected Layer export TEST
 */
TEST(nntrainerInterpreterTflite, simple_fc) {

  nntrainer::TfliteInterpreter interpreter;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});

  nn_model->addLayer(
    createLayer("input", {nntrainer::withKey("name", "in0"),
                          nntrainer::withKey("input_shape", "1:1:1:1")}));
  nn_model->addLayer(
    createLayer("fully_connected", {nntrainer::withKey("name", "fc0"),
                                    nntrainer::withKey("unit", 2)}));
  nn_model->addLayer(
    createLayer("fully_connected", {nntrainer::withKey("name", "fc1"),
                                    nntrainer::withKey("unit", 1)}));

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  EXPECT_EQ(nn_model->setOptimizer(std::move(optimizer)), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->initialize(), ML_ERROR_NONE);

  data_clear();
  unsigned int data_size = 1 * 1 * 1 * 1;
  std::vector<float> input_data;
  float *nntr_input = new float[data_size];

  for (unsigned int i = 0; i < data_size; i++) {
    auto rand_float = static_cast<float>(rand_r(&seed) / (RAND_MAX + 1.0));
    input_data.push_back(rand_float);
    nntr_input[i] = rand_float;
  }

  in_f.push_back(nntr_input);
  auto answer_f = nn_model->inference(1, in_f, l_f);
  for (auto element : answer_f) {
    ans.push_back(*element);
  }
  nn_model->exports(ml::train::ExportMethods::METHOD_TFLITE,
                    "simple_fc.tflite");

  out = run_tflite("simple_fc.tflite", input_data);

  for (size_t i = 0; i < out.size(); i++)
    EXPECT_NEAR(out[i], ans[i], 0.000001f);

  const size_t error_buflen = 100;
  char error_buf[error_buflen];
  if (remove("simple_fc.tflite")) {
    std::cerr << "remove tflite "
              << "simple_fc.tflite"
              << "failed, reason: "
              << SAFE_STRERROR(errno, error_buf, error_buflen);
  }
  delete[] nntr_input;
}

/**
 * @brief Flatten Test for export (NCHW -> NHWC)
 *
 */
TEST(nntrainerInterpreterTflite, flatten_test) {

  nntrainer::TfliteInterpreter interpreter;
  nntrainer::FlattenRealizer fr;
  data_clear();

  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=3:2:4"});

  auto flat = LayerRepresentation("flatten", {"name=flat", "input_layers=in0"});

  auto g = fr.realize(makeGraph({input0, flat}));

  nntrainer::NetworkGraph ng;

  /// @todo Disable Support Inplace  --> should be support Inplace()
  ng.setMemoryOptimizations(false);

  for (auto &node : g) {
    ng.addLayer(node);
  }
  EXPECT_EQ(ng.compile(""), ML_ERROR_NONE);
  EXPECT_EQ(ng.initialize(), ML_ERROR_NONE);

  ng.allocateTensors(nntrainer::ExecutionMode::INFERENCE);
  interpreter.serialize(g, "flatten_test.tflite");
  ng.deallocateTensors();

  std::vector<float> input_data;

  int count = 0;
  for (int i = 0; i < 3 * 2 * 4; i++) {
    input_data.push_back(count);
    count++;
  }

  out = run_tflite("flatten_test.tflite", input_data);

  std::vector<float> ans = {0, 8,  16, 1, 9,  17, 2, 10, 18, 3, 11, 19,
                            4, 12, 20, 5, 13, 21, 6, 14, 22, 7, 15, 23};

  for (size_t i = 0; i < out.size(); i++)
    EXPECT_NEAR(out[i], ans[i], 0.000001f);

  const size_t error_buflen = 100;
  char error_buf[error_buflen];
  if (remove("flatten_test.tflite")) {
    std::cerr << "remove tflite "
              << "flatten_test.tflite"
              << "failed, reason: "
              << SAFE_STRERROR(errno, error_buf, error_buflen);
  }
}

/**
 * @brief Resnet Part export TEST
 */
TEST(nntrainerInterpreterTflite, part_of_resnet_0) {

  nntrainer::TfliteInterpreter interpreter;

  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=1:1:1"});

  auto averagepool0 = LayerRepresentation(
    "pooling2d", {"name=averagepool0", "input_layers=in0", "pooling=average",
                  "pool_size=1,1", "stride=1,1", "padding=valid"});

  auto reshape0 =
    LayerRepresentation("reshape", {"name=reshape0", "target_shape=1:1:1",
                                    "input_layers=averagepool0"});

  auto fc0 = LayerRepresentation(
    "fully_connected", {"name=fc0", "unit=1", "input_layers=reshape0",
                        "bias_initializer=ones", "weight_initializer=ones"});

  auto softmax0 = LayerRepresentation(
    "activation", {"name=softmax0", "activation=softmax", "input_layers=fc0"});

  auto g = makeGraph({input0, averagepool0, reshape0, fc0, softmax0});

  nntrainer::NetworkGraph ng;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});

  for (auto &node : g) {
    nn_model->addLayer(node);
  }

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  EXPECT_EQ(nn_model->setOptimizer(std::move(optimizer)), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->initialize(), ML_ERROR_NONE);

  data_clear();
  unsigned int data_size = 1 * 1 * 1 * 1;
  std::vector<float> input_data;
  float *nntr_input = new float[data_size];

  for (unsigned int i = 0; i < data_size; i++) {
    auto rand_float = static_cast<float>(rand_r(&seed) / (RAND_MAX + 1.0));
    input_data.push_back(rand_float);
    nntr_input[i] = rand_float;
  }
  in_f.push_back(nntr_input);

  auto answer_f = nn_model->inference(1, in_f, l_f);
  for (auto element : answer_f) {
    ans.push_back(*element);
  }

  nn_model->exports(ml::train::ExportMethods::METHOD_TFLITE,
                    "part_of_resnet.tflite");

  out = run_tflite("part_of_resnet.tflite", input_data);

  for (size_t i = 0; i < out.size(); i++)
    EXPECT_NEAR(out[i], ans[i], 0.000001f);

  if (remove("part_of_resnet.tflite")) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    std::cerr << "remove ini "
              << "part_of_resnet.tflite"
              << "failed, reason: "
              << SAFE_STRERROR(errno, error_buf, error_buflen);
  }
  delete[] nntr_input;
}

/**
 * @brief MNIST Model export test
 */
TEST(nntrainerInterpreterTflite, MNIST_FULL_TEST) {

  nntrainer::TfliteInterpreter interpreter;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});
  nn_model->setProperty({nntrainer::withKey("memory_optimization", "false")});

  nn_model->addLayer(
    createLayer("input", {nntrainer::withKey("name", "in0"),
                          nntrainer::withKey("input_shape", "1:1:28:28")}));

  nn_model->addLayer(createLayer(
    "conv2d",
    {nntrainer::withKey("name", "conv0"), nntrainer::withKey("filters", 6),
     nntrainer::withKey("kernel_size", {5, 5}),
     nntrainer::withKey("stride", {1, 1}),
     nntrainer::withKey("padding", "same"),
     nntrainer::withKey("bias_initializer", "zeros"),
     nntrainer::withKey("weight_initializer", "xavier_uniform"),
     nntrainer::withKey("activation", "relu")}));

  nn_model->addLayer(
    createLayer("pooling2d", {nntrainer::withKey("name", "pooling2d_p1"),
                              nntrainer::withKey("pooling", "average"),
                              nntrainer::withKey("pool_size", {2, 2}),
                              nntrainer::withKey("stride", {2, 2}),
                              nntrainer::withKey("padding", "same")}));

  nn_model->addLayer(createLayer(
    "conv2d",
    {nntrainer::withKey("name", "conv0"), nntrainer::withKey("filters", 12),
     nntrainer::withKey("kernel_size", {5, 5}),
     nntrainer::withKey("stride", {1, 1}),
     nntrainer::withKey("padding", "same"),
     nntrainer::withKey("bias_initializer", "zeros"),
     nntrainer::withKey("weight_initializer", "xavier_uniform"),
     nntrainer::withKey("activation", "relu")}));

  nn_model->addLayer(
    createLayer("pooling2d", {nntrainer::withKey("name", "pooling2d_p1"),
                              nntrainer::withKey("pooling", "average"),
                              nntrainer::withKey("pool_size", {2, 2}),
                              nntrainer::withKey("stride", {2, 2}),
                              nntrainer::withKey("padding", "same")}));

  nn_model->addLayer(createLayer("flatten"));

  nn_model->addLayer(
    createLayer("fully_connected", {nntrainer::withKey("name", "fc0"),
                                    nntrainer::withKey("unit", 10)}));

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  EXPECT_EQ(nn_model->setOptimizer(std::move(optimizer)), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->initialize(), ML_ERROR_NONE);

  data_clear();
  unsigned int data_size = 1 * 1 * 28 * 28;
  std::vector<float> input_data;
  float nntr_input[28 * 28];

  for (unsigned int i = 0; i < data_size; i++) {
    auto rand_float = static_cast<float>(rand_r(&seed) / (RAND_MAX + 1.0));
    input_data.push_back(rand_float);
    nntr_input[i] = rand_float;
  }

  in_f.push_back(nntr_input);
  auto answer_f = nn_model->inference(1, in_f, l_f);

  for (auto element : answer_f) {
    ans.push_back(*element);
  }
  nn_model->exports(ml::train::ExportMethods::METHOD_TFLITE,
                    "MNIST_FULL_TEST.tflite");

  out = run_tflite("MNIST_FULL_TEST.tflite", input_data);

  for (size_t i = 0; i < ans.size(); i++) {
    EXPECT_NEAR(out[i], ans[i], 0.000001f);
  }

  if (remove("MNIST_FULL_TEST.tflite")) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    std::cerr << "remove tflite "
              << "MNIST_FULL_TEST.tflite"
              << "failed, reason: "
              << SAFE_STRERROR(errno, error_buf, error_buflen);
  }
}

/**
 * @brief Simple Fully Connected Layer export TEST with dropout Layer
 */
TEST(nntrainerInterpreterTflite, SIMPLE_FC_WITH_DROPOUT) {

  nntrainer::TfliteInterpreter interpreter;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});

  nn_model->addLayer(
    createLayer("input", {nntrainer::withKey("name", "in0"),
                          nntrainer::withKey("input_shape", "1:1:1:1")}));
  nn_model->addLayer(
    createLayer("fully_connected", {nntrainer::withKey("name", "fc0"),
                                    nntrainer::withKey("unit", 2)}));
  nn_model->addLayer(
    createLayer("dropout", {nntrainer::withKey("name", "dropout_1"),
                            nntrainer::withKey("dropout_rate", "0.3")}));
  nn_model->addLayer(
    createLayer("fully_connected", {nntrainer::withKey("name", "fc1"),
                                    nntrainer::withKey("unit", 1)}));

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  EXPECT_EQ(nn_model->setOptimizer(std::move(optimizer)), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->initialize(), ML_ERROR_NONE);

  data_clear();
  unsigned int data_size = 1 * 1 * 1 * 1;
  std::vector<float> input_data;
  float *nntr_input = new float[data_size];

  for (unsigned int i = 0; i < data_size; i++) {
    auto rand_float = static_cast<float>(rand_r(&seed) / (RAND_MAX + 1.0));
    input_data.push_back(rand_float);
    nntr_input[i] = rand_float;
  }

  in_f.push_back(nntr_input);
  auto answer_f = nn_model->inference(1, in_f, l_f);
  for (auto element : answer_f) {
    ans.push_back(*element);
  }
  nn_model->exports(ml::train::ExportMethods::METHOD_TFLITE,
                    "simple_fc_dropout.tflite");

  out = run_tflite("simple_fc_dropout.tflite", input_data);

  for (size_t i = 0; i < out.size(); i++)
    EXPECT_NEAR(out[i], ans[i], 0.000001f);

  const size_t error_buflen = 100;
  char error_buf[error_buflen];
  if (remove("simple_fc_dropout.tflite")) {
    std::cerr << "remove tflite "
              << "simple_fc_dropout.tflite"
              << "failed, reason: "
              << SAFE_STRERROR(errno, error_buf, error_buflen);
  }
  delete[] nntr_input;
}

/**
 * @brief Create a model whose batch normalization follows a non-trainable
 * layer, so that the exporter converts it to a fused MUL + ADD pair instead of
 * folding it into the previous layer
 *
 * @param bn_is_last true to make the batch normalization the last layer
 * @return ModelHandle compiled and initialized model with a 4:1:1 input
 */
static ModelHandle makeBnAfterNonTrainableModel(bool bn_is_last) {
  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});

  nn_model->addLayer(
    createLayer("input", {nntrainer::withKey("name", "in0"),
                          nntrainer::withKey("input_shape", "4:1:1")}));
  nn_model->addLayer(
    createLayer("pooling2d", {nntrainer::withKey("name", "pool0"),
                              nntrainer::withKey("pooling", "average"),
                              nntrainer::withKey("pool_size", {1, 1}),
                              nntrainer::withKey("stride", {1, 1}),
                              nntrainer::withKey("padding", "valid")}));
  nn_model->addLayer(createLayer(
    "batch_normalization",
    {nntrainer::withKey("name", "bn0"), nntrainer::withKey("epsilon", "0.5"),
     nntrainer::withKey("moving_mean_initializer", "he_uniform"),
     nntrainer::withKey("gamma_initializer", "he_uniform"),
     nntrainer::withKey("beta_initializer", "he_uniform")}));

  if (!bn_is_last) {
    nn_model->addLayer(
      createLayer("activation", {nntrainer::withKey("name", "relu0"),
                                 nntrainer::withKey("activation", "relu")}));
    nn_model->addLayer(
      createLayer("pooling2d", {nntrainer::withKey("name", "pool1"),
                                nntrainer::withKey("pooling", "average"),
                                nntrainer::withKey("pool_size", {1, 1}),
                                nntrainer::withKey("stride", {1, 1}),
                                nntrainer::withKey("padding", "valid")}));
  }

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  EXPECT_EQ(nn_model->setOptimizer(std::move(optimizer)), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->initialize(), ML_ERROR_NONE);

  return nn_model;
}

/**
 * @brief Copy the weights (moving mean, moving variance, gamma, beta) of the
 * batch normalization layer "bn0" out of the model
 */
static std::vector<std::vector<float>> getBnWeights(ModelHandle &nn_model) {
  std::shared_ptr<ml::train::Layer> bn;
  EXPECT_EQ(nn_model->getLayer("bn0", &bn), ML_ERROR_NONE);

  std::vector<float *> weights;
  std::vector<ml::train::TensorDim> dims;
  bn->getWeights(weights, dims);
  EXPECT_EQ(weights.size(), 4u);

  std::vector<std::vector<float>> copied;
  for (size_t i = 0; i < weights.size(); i++)
    copied.emplace_back(weights[i], weights[i] + dims[i].getDataLen());
  return copied;
}

/**
 * @brief Export batch normalization after a non-trainable layer (fused MUL +
 * ADD) and check that the export leaves the model weights intact
 */
TEST(nntrainerInterpreterTflite, bn_after_non_trainable_fused_mul_add) {
  const std::string file_name = "bn_after_non_trainable.tflite";
  ModelHandle nn_model = makeBnAfterNonTrainableModel(false);

  std::vector<float> input_data;
  for (unsigned int i = 0; i < 4; i++) {
    input_data.push_back(static_cast<float>(rand_r(&seed) / (RAND_MAX + 1.0)) -
                         0.5f);
  }

  std::vector<float *> inputs = {input_data.data()};
  std::vector<float *> labels;
  auto outputs = nn_model->inference(1, inputs, labels);
  ASSERT_EQ(outputs.size(), 1u);
  std::vector<float> answer(outputs[0], outputs[0] + input_data.size());

  auto bn_weights = getBnWeights(nn_model);

  nn_model->exports(ml::train::ExportMethods::METHOD_TFLITE, file_name);

  auto tflite_out = run_tflite(file_name, input_data);
  ASSERT_EQ(tflite_out.size(), answer.size());
  for (size_t i = 0; i < answer.size(); i++)
    EXPECT_NEAR(tflite_out[i], answer[i], 0.00001f);

  EXPECT_EQ(getBnWeights(nn_model), bn_weights);

  if (remove(file_name.c_str())) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    std::cerr << "remove tflite " << file_name << " failed, reason: "
              << SAFE_STRERROR(errno, error_buf, error_buflen);
  }
}

/**
 * @brief Exporting batch normalization after a non-trainable layer without a
 * following activation is rejected, and the failed export must neither write
 * a file nor release the model weights
 */
TEST(nntrainerInterpreterTflite, bn_after_non_trainable_as_last_layer_n) {
  const std::string file_name = "bn_after_non_trainable_last.tflite";
  ModelHandle nn_model = makeBnAfterNonTrainableModel(true);

  auto bn_weights = getBnWeights(nn_model);
  remove(file_name.c_str());

  try {
    nn_model->exports(ml::train::ExportMethods::METHOD_TFLITE, file_name);
    ADD_FAILURE() << "exporting a trailing batch normalization must fail";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("cannot be the last layer"),
              std::string::npos)
      << e.what();
  }

  EXPECT_EQ(getBnWeights(nn_model), bn_weights);
  EXPECT_NE(remove(file_name.c_str()), 0)
    << "the failed export left " << file_name << " behind";
}
