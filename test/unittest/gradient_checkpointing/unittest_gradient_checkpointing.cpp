// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Yeonjae Kim <duswo1120@snu.ac.kr>
 * Copyright (C) 2025 Hoyeon Jo <jhy213@snu.ac.kr>
 *
 * @file unittest_gradient_checkpointing.cpp
 * @date 16 December 2025
 * @brief gradient checkpointing test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Yeonjae Kim <duswo1120@snu.ac.kr>
 * @author Hoyeon Jo <jhy213@snu.ac.kr>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <neuralnet.h>

class SimpleDataGenerator {
public:
  SimpleDataGenerator(int _batch_size, int _seq_len, int _num_batches) :
    batch_size(_batch_size),
    seq_len(_seq_len),
    num_batches(_num_batches),
    current_batch(0) {

    // Allocate memory for input and label
    input_data.resize(batch_size * seq_len);
    label_data.resize(batch_size * seq_len);

    // Initialize with simple pattern
    for (int i = 0; i < batch_size * seq_len; i++) {
      input_data[i] = static_cast<float>(i % 10) / 10.0f;
      label_data[i] = static_cast<float>((i + 1) % 10) / 10.0f;
    }
  }

  void next(float **input, float **label, bool *last) {
    // Always provide data
    *input = input_data.data();
    *label = label_data.data();

    // Set last flag when we reach the end
    *last = (current_batch >= num_batches - 1);

    current_batch++;

    // Reset for next epoch
    if (current_batch >= num_batches) {
      current_batch = 0;
    }
  }

  void reset() { current_batch = 0; }

private:
  int batch_size;
  int seq_len;
  int num_batches;
  int current_batch;
  std::vector<float> input_data;
  std::vector<float> label_data;
};

int dataset_cb(float **input, float **label, bool *last, void *user_data) {
  auto *generator = reinterpret_cast<SimpleDataGenerator *>(user_data);
  generator->next(input, label, last);
  // Always return success
  return 0;
}

TEST(nntrainer_gradient_checkpointing, gradient_checkpointing_verification_01) {
  // Transformer layers with relu
  std::unique_ptr<ml::train::Model> model;
  std::unique_ptr<ml::train::Optimizer> optimizer;
  std::unique_ptr<ml::train::Dataset> dataset;

  EXPECT_NO_THROW(model = ml::train::createModel(
                    ml::train::ModelType::NEURAL_NET, {"loss=mse"}));

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "input",
    {"input_shape=1:" + std::to_string(32) + ":1", "name=input_tokens"})));

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "fully_connected", {"unit=" + std::to_string(64), "name=embedding"})));

  for (int i = 0; i < 2; i++) {
    std::string prefix = "layer" + std::to_string(i);

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "multiout", {"name=" + prefix + "/ln_multiout1"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "layer_normalization", {"axis=3", "name=" + prefix + "/ln1"})));

    EXPECT_NO_THROW(model->addLayer(
      ml::train::createLayer("multiout", {"name=" + prefix + "/multi_out1"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "multi_head_attention",
      {"name=" + prefix + "/mha",
       "input_layers=" + prefix + "/multi_out1(0)," + prefix +
         "/multi_out1(1)," + prefix + "/multi_out1(2)",
       "num_heads=" + std::to_string(4)})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "addition",
      {"name=" + prefix + "/add1",
       "input_layers=" + prefix + "/ln_multiout1(1)," + prefix + "/mha"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "multiout", {"name=" + prefix + "/ln_multiout2"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "layer_normalization", {"axis=3", "name=" + prefix + "/ln2"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "fully_connected", {"unit=" + std::to_string(256), "activation=relu",
                          "name=" + prefix + "/fc1"})));

    EXPECT_NO_THROW(model->addLayer(
      ml::train::createLayer("fully_connected", {"unit=" + std::to_string(64),
                                                 "name=" + prefix + "/fc2"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "addition",
      {"name=" + prefix + "/add2",
       "input_layers=" + prefix + "/ln_multiout2(1)," + prefix + "/fc2"})));
  }

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "layer_normalization", {"axis=3", "name=final_ln"})));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("fully_connected", {"unit=1", "name=output"})));

  for (int i = 0; i < 2; i++) {
    std::string prefix = "layer" + std::to_string(i);
    std::vector<std::string> block_layers{prefix + "/ln_multiout1",
                                          prefix + "/ln1",
                                          prefix + "/multi_out1",
                                          prefix + "/mha",
                                          prefix + "/add1",
                                          prefix + "/ln_multiout2",
                                          prefix + "/ln2",
                                          prefix + "/fc1",
                                          prefix + "/fc1/activation_realized",
                                          prefix + "/fc2",
                                          prefix + "/add2"};
    EXPECT_NO_THROW(model->addCheckpointBlock(block_layers));
  }

  EXPECT_NO_THROW(model->setProperty(
    {"batch_size=" + std::to_string(4), "epochs=" + std::to_string(2)}));

  EXPECT_NO_THROW(
    optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"}));
  EXPECT_NO_THROW(model->setOptimizer(std::move(optimizer)));

  EXPECT_EQ(model->compile(), ML_ERROR_NONE);

  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);

  EXPECT_NO_THROW(dataset = ml::train::createDataset(
                    ml::train::DatasetType::GENERATOR, dataset_cb,
                    new SimpleDataGenerator(4, 32, 8)));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                              std::move(dataset)),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(model->setGCVerify());
  EXPECT_NO_THROW(model->train());
}

TEST(nntrainer_gradient_checkpointing, gradient_checkpointing_verification_02) {
  // Transformer layers with gelu
  std::unique_ptr<ml::train::Model> model;
  std::unique_ptr<ml::train::Optimizer> optimizer;
  std::unique_ptr<ml::train::Dataset> dataset;

  EXPECT_NO_THROW(model = ml::train::createModel(
                    ml::train::ModelType::NEURAL_NET, {"loss=mse"}));

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "input",
    {"input_shape=1:" + std::to_string(32) + ":1", "name=input_tokens"})));

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "fully_connected", {"unit=" + std::to_string(64), "name=embedding"})));

  for (int i = 0; i < 2; i++) {
    std::string prefix = "layer" + std::to_string(i);

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "multiout", {"name=" + prefix + "/ln_multiout1"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "layer_normalization", {"axis=3", "name=" + prefix + "/ln1"})));

    EXPECT_NO_THROW(model->addLayer(
      ml::train::createLayer("multiout", {"name=" + prefix + "/multi_out1"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "multi_head_attention",
      {"name=" + prefix + "/mha",
       "input_layers=" + prefix + "/multi_out1(0)," + prefix +
         "/multi_out1(1)," + prefix + "/multi_out1(2)",
       "num_heads=" + std::to_string(4)})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "addition",
      {"name=" + prefix + "/add1",
       "input_layers=" + prefix + "/ln_multiout1(1)," + prefix + "/mha"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "multiout", {"name=" + prefix + "/ln_multiout2"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "layer_normalization", {"axis=3", "name=" + prefix + "/ln2"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "fully_connected", {"unit=" + std::to_string(256), "activation=gelu",
                          "name=" + prefix + "/fc1"})));

    EXPECT_NO_THROW(model->addLayer(
      ml::train::createLayer("fully_connected", {"unit=" + std::to_string(64),
                                                 "name=" + prefix + "/fc2"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "addition",
      {"name=" + prefix + "/add2",
       "input_layers=" + prefix + "/ln_multiout2(1)," + prefix + "/fc2"})));
  }

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "layer_normalization", {"axis=3", "name=final_ln"})));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("fully_connected", {"unit=1", "name=output"})));

  for (int i = 0; i < 2; i++) {
    std::string prefix = "layer" + std::to_string(i);
    std::vector<std::string> block_layers{prefix + "/ln_multiout1",
                                          prefix + "/ln1",
                                          prefix + "/multi_out1",
                                          prefix + "/mha",
                                          prefix + "/add1",
                                          prefix + "/ln_multiout2",
                                          prefix + "/ln2",
                                          prefix + "/fc1",
                                          prefix + "/fc1/activation_realized",
                                          prefix + "/fc2",
                                          prefix + "/add2"};
    EXPECT_NO_THROW(model->addCheckpointBlock(block_layers));
  }

  EXPECT_NO_THROW(model->setProperty(
    {"batch_size=" + std::to_string(4), "epochs=" + std::to_string(2)}));

  EXPECT_NO_THROW(
    optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"}));
  EXPECT_NO_THROW(model->setOptimizer(std::move(optimizer)));

  EXPECT_EQ(model->compile(), ML_ERROR_NONE);

  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);

  EXPECT_NO_THROW(dataset = ml::train::createDataset(
                    ml::train::DatasetType::GENERATOR, dataset_cb,
                    new SimpleDataGenerator(4, 32, 8)));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                              std::move(dataset)),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(model->setGCVerify());
  EXPECT_NO_THROW(model->train());
}

TEST(nntrainer_gradient_checkpointing, gradient_checkpointing_verification_03) {
  // Transformer layers with swiglu
  std::unique_ptr<ml::train::Model> model;
  std::unique_ptr<ml::train::Optimizer> optimizer;
  std::unique_ptr<ml::train::Dataset> dataset;

  EXPECT_NO_THROW(model = ml::train::createModel(
                    ml::train::ModelType::NEURAL_NET, {"loss=mse"}));

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "input",
    {"input_shape=1:" + std::to_string(32) + ":1", "name=input_tokens"})));

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "fully_connected", {"unit=" + std::to_string(64), "name=embedding"})));

  for (int i = 0; i < 2; i++) {
    std::string prefix = "layer" + std::to_string(i);

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "multiout", {"name=" + prefix + "/ln_multiout1"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "layer_normalization", {"axis=3", "name=" + prefix + "/ln1"})));

    EXPECT_NO_THROW(model->addLayer(
      ml::train::createLayer("multiout", {"name=" + prefix + "/multi_out1"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "multi_head_attention",
      {"name=" + prefix + "/mha",
       "input_layers=" + prefix + "/multi_out1(0)," + prefix +
         "/multi_out1(1)," + prefix + "/multi_out1(2)",
       "num_heads=" + std::to_string(4)})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "addition",
      {"name=" + prefix + "/add1",
       "input_layers=" + prefix + "/ln_multiout1(1)," + prefix + "/mha"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "multiout", {"name=" + prefix + "/ln_multiout2"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "layer_normalization", {"axis=3", "name=" + prefix + "/ln2"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "fully_connected",
      {"unit=" + std::to_string(256), "name=" + prefix + "/gate_proj"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "activation", {"activation=swish", "name=" + prefix + "/gate_act"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "fully_connected",
      {"unit=" + std::to_string(256), "input_layers=" + prefix + "/ln2",
       "name=" + prefix + "/up_proj"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "multiply",
      {"name=" + prefix + "/glu_mul",
       "input_layers=" + prefix + "/gate_act," + prefix + "/up_proj"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "fully_connected",
      {"unit=" + std::to_string(64), "name=" + prefix + "/down_proj"})));

    EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
      "addition", {"name=" + prefix + "/add2", "input_layers=" + prefix +
                                                 "/ln_multiout2(1)," + prefix +
                                                 "/down_proj"})));
  }

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "layer_normalization", {"axis=3", "name=final_ln"})));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("fully_connected", {"unit=1", "name=output"})));

  for (int i = 0; i < 2; i++) {
    std::string prefix = "layer" + std::to_string(i);
    std::vector<std::string> block_layers{
      prefix + "/ln_multiout1", prefix + "/ln1",       prefix + "/multi_out1",
      prefix + "/mha",          prefix + "/add1",      prefix + "/ln_multiout2",
      prefix + "/ln2",          prefix + "/gate_proj", prefix + "/gate_act",
      prefix + "/up_proj",      prefix + "/glu_mul",   prefix + "/down_proj",
      prefix + "/add2"};
    EXPECT_NO_THROW(model->addCheckpointBlock(block_layers));
  }

  EXPECT_NO_THROW(model->setProperty(
    {"batch_size=" + std::to_string(4), "epochs=" + std::to_string(2)}));

  EXPECT_NO_THROW(
    optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"}));
  EXPECT_NO_THROW(model->setOptimizer(std::move(optimizer)));

  EXPECT_EQ(model->compile(), ML_ERROR_NONE);

  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);

  EXPECT_NO_THROW(dataset = ml::train::createDataset(
                    ml::train::DatasetType::GENERATOR, dataset_cb,
                    new SimpleDataGenerator(4, 32, 8)));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                              std::move(dataset)),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(model->setGCVerify());
  EXPECT_NO_THROW(model->train());
}

/**
 * @brief Main gtest
 */
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
