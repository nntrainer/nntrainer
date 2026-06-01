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

#include <cmath>

#include <memory>
#include <neuralnet.h>
#include <stdexcept>
#include <vector>

/**
 * @brief Simple fixed-pattern data generator used by gradient checkpointing
 * tests to feed repeatable input/label batches.
 */
class SimpleDataGenerator {
public:
  SimpleDataGenerator(int _batch_size, int _seq_len, int _num_batches) :
    batch_size(_batch_size),
    seq_len(_seq_len),
    num_batches(_num_batches),
    current_batch(0) {

    const auto data_size =
      static_cast<std::vector<float>::size_type>(batch_size) *
      static_cast<std::vector<float>::size_type>(seq_len);

    // Allocate memory for input and label
    input_data.resize(data_size);
    label_data.resize(data_size);

    // Initialize with simple pattern
    for (std::vector<float>::size_type i = 0; i < data_size; i++) {
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

/**
 * @brief Dataset callback for SimpleDataGenerator
 */
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

  auto generator = std::make_unique<SimpleDataGenerator>(4, 32, 8);
  EXPECT_NO_THROW(dataset =
                    ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                             dataset_cb, generator.get()));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                              std::move(dataset)),
            ML_ERROR_NONE);

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

  auto generator = std::make_unique<SimpleDataGenerator>(4, 32, 8);
  EXPECT_NO_THROW(dataset =
                    ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                             dataset_cb, generator.get()));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                              std::move(dataset)),
            ML_ERROR_NONE);

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

  auto generator = std::make_unique<SimpleDataGenerator>(4, 32, 8);
  EXPECT_NO_THROW(dataset =
                    ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                             dataset_cb, generator.get()));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                              std::move(dataset)),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(model->train());
}

TEST(nntrainer_gradient_checkpointing, gradient_checkpointing_verification_04) {
  // A checkpoint block that skips the middle layer should fail connectivity
  std::unique_ptr<ml::train::Model> model;
  std::unique_ptr<ml::train::Optimizer> optimizer;

  EXPECT_NO_THROW(model = ml::train::createModel(
                    ml::train::ModelType::NEURAL_NET, {"loss=mse"}));

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "input",
    {"input_shape=1:" + std::to_string(8) + ":1", "name=input_tokens"})));

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "fully_connected", {"unit=" + std::to_string(16), "name=fc1"})));

  EXPECT_NO_THROW(model->addLayer(ml::train::createLayer(
    "fully_connected", {"unit=" + std::to_string(16), "name=fc2"})));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("fully_connected", {"unit=1", "name=fc3"})));

  EXPECT_EQ(model->addCheckpointBlock({"fc1", "fc3"}), ML_ERROR_NONE);

  EXPECT_NO_THROW(model->setProperty(
    {"batch_size=" + std::to_string(4), "epochs=" + std::to_string(1)}));

  EXPECT_NO_THROW(
    optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"}));
  EXPECT_NO_THROW(model->setOptimizer(std::move(optimizer)));

  EXPECT_THROW(model->compile(), std::invalid_argument);
}

TEST(nntrainer_gradient_checkpointing, gradient_checkpointing_correctness_01) {
  // Train an identical small model with and without gradient checkpointing.
  // Both models use weight_initializer=ones and bias_initializer=zeros so they
  // start from the same deterministic weights. The final losses must match
  // within a relative tolerance of 1e-3.

  auto make_model =
    [](bool with_checkpoint) -> std::unique_ptr<ml::train::Model> {
    std::unique_ptr<ml::train::Model> model;
    model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});

    model->addLayer(
      ml::train::createLayer("input", {"input_shape=1:1:8", "name=in"}));

    model->addLayer(ml::train::createLayer(
      "fully_connected",
      {"unit=16", "activation=gelu", "name=fc1", "weight_initializer=ones",
       "bias_initializer=zeros"}));

    model->addLayer(ml::train::createLayer(
      "fully_connected", {"unit=8", "name=fc2", "weight_initializer=ones",
                          "bias_initializer=zeros"}));

    if (with_checkpoint) {
      EXPECT_NO_THROW(
        model->addCheckpointBlock({"fc1", "fc1/activation_realized", "fc2"}));
    }

    model->setProperty({"batch_size=4", "epochs=1"});

    auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.01"});
    model->setOptimizer(std::move(optimizer));

    return model;
  };

  auto model_ref = make_model(false);
  auto model_ckpt = make_model(true);

  EXPECT_EQ(model_ref->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model_ref->initialize(), ML_ERROR_NONE);

  EXPECT_EQ(model_ckpt->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model_ckpt->initialize(), ML_ERROR_NONE);

  // Fixed deterministic input/label data (same for both models).
  std::vector<float> input_buf(4 * 8);
  std::vector<float> label_buf(4 * 8);
  for (int i = 0; i < 4 * 8; ++i) {
    input_buf[i] = static_cast<float>(i % 8) * 0.1f;
    label_buf[i] = static_cast<float>((i + 1) % 8) * 0.1f;
  }

  // Dataset callback using local buffers via raw pointer capture.
  struct DataCtx {
    float *input;
    float *label;
    int calls;
    int max_calls;
  };
  DataCtx ctx_ref{input_buf.data(), label_buf.data(), 0, 4};
  DataCtx ctx_ckpt{input_buf.data(), label_buf.data(), 0, 4};

  auto gen_cb = [](float **in, float **lbl, bool *last, void *ud) -> int {
    auto *c = static_cast<DataCtx *>(ud);
    *in = c->input;
    *lbl = c->label;
    *last = (c->calls >= c->max_calls - 1);
    c->calls++;
    if (c->calls >= c->max_calls)
      c->calls = 0;
    return 0;
  };

  auto ds_ref = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                         gen_cb, &ctx_ref);
  auto ds_ckpt = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                          gen_cb, &ctx_ckpt);

  EXPECT_EQ(model_ref->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                                  std::move(ds_ref)),
            ML_ERROR_NONE);
  EXPECT_EQ(model_ckpt->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                                   std::move(ds_ckpt)),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(model_ref->train());
  EXPECT_NO_THROW(model_ckpt->train());

  float loss_ref = model_ref->getLoss();
  float loss_ckpt = model_ckpt->getLoss();

  // Both must have converged to a finite, non-negative loss.
  EXPECT_TRUE(std::isfinite(loss_ref));
  EXPECT_TRUE(std::isfinite(loss_ckpt));
  EXPECT_GE(loss_ref, 0.0f);

  // Losses must match within 1e-3 relative tolerance.
  float denom = std::max(std::abs(loss_ref), std::abs(loss_ckpt)) + 1e-8f;
  float rel_err = std::abs(loss_ref - loss_ckpt) / denom;
  EXPECT_LT(rel_err, 1e-3f)
    << "loss_ref=" << loss_ref << " loss_ckpt=" << loss_ckpt;
}

TEST(nntrainer_gradient_checkpointing,
     gradient_checkpointing_stateful_layer_rejection_01) {
  // BatchNormalization is stateful (updates running mean/var) and must be
  // rejected at compile() when placed inside a checkpoint block.
  std::unique_ptr<ml::train::Model> model;
  std::unique_ptr<ml::train::Optimizer> optimizer;

  EXPECT_NO_THROW(model = ml::train::createModel(
                    ml::train::ModelType::NEURAL_NET, {"loss=mse"}));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("input", {"input_shape=1:1:8", "name=in"})));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("fully_connected", {"unit=8", "name=fc1"})));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("batch_normalization", {"name=bn1"})));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("fully_connected", {"unit=1", "name=fc2"})));

  EXPECT_EQ(model->addCheckpointBlock({"fc1", "bn1", "fc2"}), ML_ERROR_NONE);

  model->setProperty({"batch_size=4", "epochs=1"});

  EXPECT_NO_THROW(optimizer =
                    ml::train::createOptimizer("sgd", {"learning_rate=0.01"}));
  EXPECT_NO_THROW(model->setOptimizer(std::move(optimizer)));

  EXPECT_THROW(model->compile(), std::invalid_argument);
}

TEST(nntrainer_gradient_checkpointing,
     gradient_checkpointing_stateful_layer_rejection_02) {
  // Dropout is stateful (RNG state changes on each forward) and must be
  // rejected at compile() when placed inside a checkpoint block.
  std::unique_ptr<ml::train::Model> model;
  std::unique_ptr<ml::train::Optimizer> optimizer;

  EXPECT_NO_THROW(model = ml::train::createModel(
                    ml::train::ModelType::NEURAL_NET, {"loss=mse"}));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("input", {"input_shape=1:1:8", "name=in"})));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("fully_connected", {"unit=8", "name=fc1"})));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("dropout", {"dropout_rate=0.5", "name=drop1"})));

  EXPECT_NO_THROW(model->addLayer(
    ml::train::createLayer("fully_connected", {"unit=1", "name=fc2"})));

  EXPECT_EQ(model->addCheckpointBlock({"fc1", "drop1", "fc2"}), ML_ERROR_NONE);

  model->setProperty({"batch_size=4", "epochs=1"});

  EXPECT_NO_THROW(optimizer =
                    ml::train::createOptimizer("sgd", {"learning_rate=0.01"}));
  EXPECT_NO_THROW(model->setOptimizer(std::move(optimizer)));

  EXPECT_THROW(model->compile(), std::invalid_argument);
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
