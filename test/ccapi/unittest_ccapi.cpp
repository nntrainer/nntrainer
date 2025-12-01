// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file        unittest_ccapi.cc
 * @date        19 October 2020
 * @brief       cc API Unit tests.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <iostream>

#include <dataset.h>
#include <ini_wrapper.h>
#include <layer.h>
#include <model.h>
#include <nntrainer_error.h>
#include <nntrainer_test_util.h>
#include <optimizer.h>

static const std::string getTestResPath(const std::string &file) {
  return getResPath(file, {"test"});
}

// Add unittest for train fail without optimizer, but inference pass

/**
 * @brief Neural Network Model Construct Test
 */
TEST(ccapi_model, construct_01_n) {
  EXPECT_THROW(ml::train::createModel(ml::train::ModelType::UNKNOWN),
               std::invalid_argument);

  EXPECT_THROW(ml::train::createModel((ml::train::ModelType)100),
               std::invalid_argument);
}

/**
 * @brief Neural Network Model Construct Test
 */
TEST(ccapi_model, construct_02_p) {
  EXPECT_NO_THROW(ml::train::createModel(ml::train::ModelType::NEURAL_NET));
}

/**
 * @brief Neural Network Layer Construct Test
 */
TEST(ccapi_layer, construct_01_n) {
  EXPECT_THROW(ml::train::createLayer("unknown type"), std::invalid_argument);
}

/**
 * @brief Neural Network Layer Contruct Test
 */
TEST(ccapi_layer, construct_02_p) {
  std::shared_ptr<ml::train::Layer> layer;

  EXPECT_NO_THROW(layer = ml::train::layer::Input());
  EXPECT_EQ(layer->getType(), "input");

  EXPECT_NO_THROW(layer = ml::train::layer::WeightLayer());
  EXPECT_EQ(layer->getType(), "weight");

  EXPECT_NO_THROW(layer = ml::train::layer::AddLayer());
  EXPECT_EQ(layer->getType(), "add");

  EXPECT_NO_THROW(layer = ml::train::layer::SubtractLayer());
  EXPECT_EQ(layer->getType(), "subtract");

  EXPECT_NO_THROW(layer = ml::train::layer::MultiplyLayer());
  EXPECT_EQ(layer->getType(), "multiply");

  EXPECT_NO_THROW(layer = ml::train::layer::FullyConnected());
  EXPECT_EQ(layer->getType(), "fully_connected");

  EXPECT_NO_THROW(layer = ml::train::layer::BatchNormalization());
  EXPECT_EQ(layer->getType(), "batch_normalization");

  EXPECT_NO_THROW(layer = ml::train::layer::Convolution2D());
  EXPECT_EQ(layer->getType(), "conv2d");

  EXPECT_NO_THROW(layer = ml::train::layer::Pooling2D());
  EXPECT_EQ(layer->getType(), "pooling2d");

  EXPECT_NO_THROW(layer = ml::train::layer::Flatten());
  EXPECT_EQ(layer->getType(), "flatten");

  EXPECT_NO_THROW(layer = ml::train::layer::Addition());
  EXPECT_EQ(layer->getType(), "addition");

  EXPECT_NO_THROW(layer = ml::train::layer::Concat());
  EXPECT_EQ(layer->getType(), "concat");

  EXPECT_NO_THROW(layer = ml::train::layer::MultiOut());
  EXPECT_EQ(layer->getType(), "multiout");

  EXPECT_NO_THROW(layer = ml::train::layer::MoLAttention());
  EXPECT_EQ(layer->getType(), "mol_attention");

  EXPECT_NO_THROW(layer = ml::train::layer::ReduceMean());
  EXPECT_EQ(layer->getType(), "reduce_mean");

#ifdef ENABLE_NNSTREAMER_BACKBONE
  EXPECT_NO_THROW(layer = ml::train::layer::BackboneNNStreamer());
  EXPECT_EQ(layer->getType(), "backbone_nnstreamer");
#endif

#ifdef ENABLE_TFLITE_BACKBONE
  EXPECT_NO_THROW(layer = ml::train::layer::BackboneTFLite());
  EXPECT_EQ(layer->getType(), "backbone_tflite");
#endif

  EXPECT_NO_THROW(layer = ml::train::layer::ReLU());
  EXPECT_EQ(layer->getType(), "activation");

  EXPECT_NO_THROW(layer = ml::train::layer::Tanh());
  EXPECT_EQ(layer->getType(), "activation");

  EXPECT_NO_THROW(layer = ml::train::layer::Sigmoid());
  EXPECT_EQ(layer->getType(), "activation");

  EXPECT_NO_THROW(layer = ml::train::layer::Softmax());
  EXPECT_EQ(layer->getType(), "activation");
}

/**
 * @brief Neural Network Loss Layer Contruct Test
 */
TEST(ccapi_layer, construct_03_p) {
  std::shared_ptr<ml::train::Layer> layer;

  EXPECT_NO_THROW(layer = ml::train::loss::MSE());
  EXPECT_EQ(layer->getType(), "mse");

  EXPECT_NO_THROW(layer = ml::train::loss::CrossEntropySigmoid());
  EXPECT_EQ(layer->getType(), "cross_sigmoid");

  EXPECT_NO_THROW(layer = ml::train::loss::CrossEntropySoftmax());
  EXPECT_EQ(layer->getType(), "cross_softmax");
}

/**
 * @brief Neural Network Optimizer Contruct Test
 */
TEST(ccapi_optimizer, construct_01_n) {
  EXPECT_THROW(ml::train::createOptimizer("Not existing type"),
               std::invalid_argument);
}

/**
 * @brief Neural Network Optimizer Contruct Test
 */
TEST(ccapi_optimizer, construct_02_p) {
  EXPECT_NO_THROW(ml::train::optimizer::Adam());
  EXPECT_NO_THROW(ml::train::optimizer::SGD());
  EXPECT_NO_THROW(ml::train::optimizer::AdamW());
  EXPECT_NO_THROW(ml::train::optimizer::Lion());
}

/**
 * @brief Neural Network Dataset Contruct Test
 */
TEST(ccapi_dataset, construct_01_n) {
  EXPECT_THROW(ml::train::createDataset(ml::train::DatasetType::UNKNOWN),
               std::invalid_argument);
}

/**
 * @brief Neural Network Dataset Contruct Test
 */
TEST(ccapi_dataset, construct_02_n) {
  EXPECT_THROW(ml::train::createDataset(ml::train::DatasetType::GENERATOR),
               std::invalid_argument);
}

static nntrainer::IniSection model_base("Model", "Type = NeuralNetwork"
                                                 " | Epochs = 1"
                                                 " | Loss = cross"
                                                 " | Save_Path = 'model.bin'"
                                                 " | batch_size = 32");

static nntrainer::IniSection optimizer("Optimizer", "Type = adam"
                                                    " | beta1 = 0.9"
                                                    " | beta2 = 0.9999"
                                                    " | epsilon = 1e-7");

static nntrainer::IniSection learning_rate("LearningRateScheduler",
                                           "Type = exponential"
                                           " | Learning_rate = 0.0001"
                                           " | Decay_rate = 0.96"
                                           " | Decay_steps = 1000");

static nntrainer::IniSection dataset("Dataset", "BufferSize=100"
                                                " | TrainData = trainingSet.dat"
                                                " | ValidData = valSet.dat"
                                                " | LabelData = label.dat");

static nntrainer::IniSection inputlayer("inputlayer",
                                        "Type = input"
                                        "| Input_Shape = 1:1:62720"
                                        "| Normalization = true"
                                        "| Activation = sigmoid");

static nntrainer::IniSection outputlayer("outputlayer",
                                         "Type = fully_connected"
                                         "| input_layers = inputlayer"
                                         "| Unit = 10"
                                         "| bias_initializer = zeros"
                                         "| Activation = softmax");

/**
 * @brief Neural Network Model Training
 */
TEST(nntrainer_ccapi, train_with_config_01_p) {
  std::unique_ptr<ml::train::Model> model;
  ScopedIni s("ccapi_test_train_01_p",
              {model_base + "batch_size = 16", optimizer, learning_rate,
               dataset + "-BufferSize", inputlayer, outputlayer});

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_EQ(model->loadFromConfig(s.getIniName()), ML_ERROR_NONE);
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->train());

  EXPECT_NEAR(model->getTrainingLoss(), 4.13896, tolerance);
  EXPECT_NEAR(model->getValidationLoss(), 3.64587, tolerance);
}

/**
 * @brief Neural Network Model Training
 */
TEST(nntrainer_ccapi, train_dataset_with_file_01_p) {
  std::unique_ptr<ml::train::Model> model;
  std::shared_ptr<ml::train::Layer> layer;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  std::unique_ptr<ml::train::LearningRateScheduler> lrs;
  std::shared_ptr<ml::train::Dataset> dataset;

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_NO_THROW(layer = ml::train::layer::Input(
                    {"input_shape=1:1:62720", "normalization=true"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    layer = ml::train::layer::FullyConnected(
      {"unit= 10", "activation=softmax", "bias_initializer=zeros",
       "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
       "weight_initializer=xavier_uniform", "input_layers=input0"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(optimizer = ml::train::optimizer::Adam(
                    {"beta1=0.002", "beta2=0.001", "epsilon=1e-7"}));

  EXPECT_NO_THROW(
    lrs = ml::train::optimizer::learning_rate::Exponential(
      {"learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000"}));

  EXPECT_NO_THROW(optimizer->setLearningRateScheduler(std::move(lrs)));
  EXPECT_NO_THROW(model->setOptimizer(optimizer));

  EXPECT_NO_THROW(
    dataset = ml::train::createDataset(
      ml::train::DatasetType::FILE, getTestResPath("trainingSet.dat").c_str()));
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(
    dataset = ml::train::createDataset(ml::train::DatasetType::FILE,
                                       getTestResPath("valSet.dat").c_str()));
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(model->setProperty(
    {"loss=cross", "batch_size=16", "epochs=2", "save_path=model.bin"}));
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->train());

  EXPECT_NEAR(model->getTrainingLoss(), 2.1782395, tolerance);
  EXPECT_NEAR(model->getValidationLoss(), 2.2059061, tolerance);
}

/**
 * @brief Neural Network Model Training
 */
TEST(nntrainer_ccapi, train_dataset_with_generator_01_p) {
  std::unique_ptr<ml::train::Model> model;
  std::shared_ptr<ml::train::Layer> layer;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  std::shared_ptr<ml::train::Dataset> dataset;

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_NO_THROW(layer = ml::train::layer::Input(
                    {"input_shape=1:1:62720", "normalization=true"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    layer = ml::train::layer::FullyConnected(
      {"unit= 10", "activation=softmax", "bias_initializer=zeros",
       "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
       "weight_initializer=xavier_uniform", "input_layers=input0"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    optimizer = ml::train::optimizer::Adam(
      {"learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
       "beta1=0.002", "beta2=0.001", "epsilon=1e-7"}));
  EXPECT_NO_THROW(model->setOptimizer(optimizer));

  auto train_data = createTrainData();
  auto valid_data = createValidData();
  EXPECT_NO_THROW(dataset = ml::train::createDataset(
                    ml::train::DatasetType::GENERATOR, getSample, &train_data));
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(dataset = ml::train::createDataset(
                    ml::train::DatasetType::GENERATOR, getSample, &valid_data));
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(model->setProperty(
    {"loss=cross", "batch_size=16", "epochs=2", "save_path=model.bin"}));
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->train());

  EXPECT_NEAR(model->getTrainingLoss(), 2.238682, tolerance);
  EXPECT_NEAR(model->getValidationLoss(), 2.0042247, tolerance);
}

/**
 * @brief Neural Network Model Training
 */
TEST(nntrainer_ccapi, train_batch_size_update_after) {
  std::unique_ptr<ml::train::Model> model;
  std::shared_ptr<ml::train::Layer> layer;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  std::shared_ptr<ml::train::Dataset> dataset;

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_NO_THROW(layer = ml::train::layer::Input(
                    {"input_shape=1:1:62720", "normalization=true"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    layer = ml::train::layer::FullyConnected(
      {"unit= 10", "activation=softmax", "bias_initializer=zeros",
       "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
       "weight_initializer=xavier_uniform", "input_layers=input0"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    optimizer = ml::train::optimizer::Adam(
      {"learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
       "beta1=0.002", "beta2=0.001", "epsilon=1e-7"}));
  EXPECT_NO_THROW(model->setOptimizer(optimizer));

  EXPECT_NO_THROW(
    dataset = ml::train::createDataset(
      ml::train::DatasetType::FILE, getTestResPath("trainingSet.dat").c_str()));
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(
    dataset = ml::train::createDataset(ml::train::DatasetType::FILE,
                                       getTestResPath("valSet.dat").c_str()));
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(
    model->setProperty({"loss=cross", "batch_size=16", "epochs=1"}));

  /** Update batch size after compile */
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->setProperty({"batch_size=128"}));

  /** Update batch size after initialize */
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->setProperty({"batch_size=8"}));
  EXPECT_NO_THROW(model->train());

  /** Update batch size after train */
  EXPECT_NO_THROW(model->setProperty({"batch_size=16"}));
  EXPECT_NO_THROW(model->train());

  /** Update batch size after train */
  EXPECT_NO_THROW(model->setProperty({"batch_size=32"}));
  EXPECT_NO_THROW(model->train());

  /** Update batch size after train */
  EXPECT_NO_THROW(model->setProperty({"batch_size=4"}));
  EXPECT_NO_THROW(model->train());

  EXPECT_NEAR(model->getTrainingLoss(), 1.9332184, tolerance);
  EXPECT_NEAR(model->getValidationLoss(), 2.179843, tolerance);
}

/**
 * @brief Neural Network Model Training
 * @note Compilation without any argument sets default execution mode as train.
 *       The train mode requires optimizer to be set. Thus, both initialize and
 *       train throw errors.
 */
TEST(nntrainer_ccapi, train_with_config_02_n) {
  std::unique_ptr<ml::train::Model> model;
  ScopedIni s("ccapi_test_train_01_p",
              {model_base + "batch_size = 16", dataset + "-BufferSize",
               inputlayer, outputlayer});

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_EQ(model->loadFromConfig(s.getIniName()), ML_ERROR_NONE);
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_INVALID_PARAMETER);
  EXPECT_EQ(model->train(), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_ccapi, save_ini_p) {
  std::unique_ptr<ml::train::Model> model;
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  ScopedIni s("ccapi_simple_ini",
              {model_base + "batch_size = 16", optimizer, learning_rate,
               dataset + "-BufferSize", inputlayer, outputlayer});

  std::shared_ptr<ml::train::Dataset> dataset = ml::train::createDataset(
    ml::train::DatasetType::FILE, getTestResPath("trainingSet.dat").c_str());
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset),
            ML_ERROR_NONE);

  EXPECT_EQ(model->loadFromConfig(s.getIniName()), ML_ERROR_NONE);
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
  auto saved_ini_name = s.getIniName() + "_saved";
  if (remove(saved_ini_name.c_str())) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    std::cerr << "remove ini " << saved_ini_name << "failed, reason: "
              << SAFE_STRERROR(errno, error_buf, error_buflen);
  }

  model->save(saved_ini_name, ml::train::ModelFormat::MODEL_FORMAT_INI);
}

TEST(nntrainer_ccapi, model_copy_01_p) {
  std::unique_ptr<ml::train::Model> model;
  std::unique_ptr<ml::train::Model> c_model;

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  ScopedIni s("ccapi_simple_ini",
              {model_base + "batch_size = 16", optimizer, learning_rate,
               dataset + "-BufferSize", inputlayer, outputlayer});
  std::shared_ptr<ml::train::Dataset> dataset = ml::train::createDataset(
    ml::train::DatasetType::FILE, getTestResPath("trainingSet.dat").c_str());
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset),
            ML_ERROR_NONE);

  EXPECT_EQ(model->loadFromConfig(s.getIniName()), ML_ERROR_NONE);

  c_model = copyConfiguration(*model);

  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);

  EXPECT_EQ(c_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(c_model->initialize(), ML_ERROR_NONE);

  std::shared_ptr<ml::train::Layer> l;
  std::shared_ptr<ml::train::Layer> c_l;

  model->getLayer("inputlayer", &l);
  c_model->getLayer("inputlayer", &c_l);

  EXPECT_EQ(l->getName(), c_l->getName());
  EXPECT_EQ(l->getType(), c_l->getType());
}

TEST(nntrainer_ccapi, model_copy_01_n) {
  std::unique_ptr<ml::train::Model> model;
  std::unique_ptr<ml::train::Model> c_model;

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  ScopedIni s("ccapi_simple_ini",
              {model_base + "batch_size = 16", optimizer, learning_rate,
               dataset + "-BufferSize", inputlayer, outputlayer});
  std::shared_ptr<ml::train::Dataset> dataset = ml::train::createDataset(
    ml::train::DatasetType::FILE, getTestResPath("trainingSet.dat").c_str());
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset),
            ML_ERROR_NONE);

  EXPECT_EQ(model->loadFromConfig(s.getIniName()), ML_ERROR_NONE);

  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);

  // copyConfiguration is not allowed after finalize
  EXPECT_THROW(copyConfiguration(*model), std::invalid_argument);
}

TEST(nntrainer_ccapi, model_copy_02_p) {

  std::unique_ptr<ml::train::Model> model;
  std::unique_ptr<ml::train::Model> c_model;
  std::shared_ptr<ml::train::Layer> layer;
  std::shared_ptr<ml::train::Layer> c_layer;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  std::unique_ptr<ml::train::LearningRateScheduler> lrs;
  std::shared_ptr<ml::train::Dataset> dataset;
  std::shared_ptr<ml::train::Dataset> c_dataset;

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_NO_THROW(
    layer = ml::train::layer::Input(
      {"name=input0", "input_shape=1:1:62720", "normalization=true"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    layer = ml::train::layer::FullyConnected(
      {"name=fc", "unit= 10", "activation=softmax", "bias_initializer=zeros",
       "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
       "weight_initializer=xavier_uniform", "input_layers=input0"}));

  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(optimizer = ml::train::optimizer::Adam(
                    {"beta1=0.002", "beta2=0.001", "epsilon=1e-7"}));

  EXPECT_NO_THROW(
    lrs = ml::train::optimizer::learning_rate::Exponential(
      {"learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000"}));

  EXPECT_NO_THROW(optimizer->setLearningRateScheduler(std::move(lrs)));
  EXPECT_NO_THROW(model->setOptimizer(optimizer));

  EXPECT_NO_THROW(
    dataset = ml::train::createDataset(
      ml::train::DatasetType::FILE, getTestResPath("trainingSet.dat").c_str()));
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(
    dataset = ml::train::createDataset(ml::train::DatasetType::FILE,
                                       getTestResPath("valSet.dat").c_str()));
  EXPECT_NO_THROW(dataset->setProperty({"buffer_size=100"}));
  EXPECT_EQ(model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset),
            ML_ERROR_NONE);

  EXPECT_NO_THROW(model->setProperty(
    {"loss=cross", "batch_size=16", "epochs=2", "save_path=model.bin"}));

  c_model = copyConfiguration(*model);

  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);

  EXPECT_NO_THROW(model->train());

  EXPECT_NO_THROW(
    c_dataset = ml::train::createDataset(
      ml::train::DatasetType::FILE, getTestResPath("trainingSet.dat").c_str()));

  EXPECT_NO_THROW(c_dataset->setProperty({"buffer_size=10"}));

  EXPECT_EQ(
    c_model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset),
    ML_ERROR_NONE);

  EXPECT_NO_THROW(
    c_dataset = ml::train::createDataset(ml::train::DatasetType::FILE,
                                         getTestResPath("valSet.dat").c_str()));

  EXPECT_NO_THROW(c_dataset->setProperty({"buffer_size=10"}));
  EXPECT_EQ(
    c_model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset),
    ML_ERROR_NONE);

  EXPECT_EQ(c_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(c_model->initialize(), ML_ERROR_NONE);

  EXPECT_NO_THROW(c_model->train());

  std::shared_ptr<ml::train::Layer> l;
  std::shared_ptr<ml::train::Layer> c_l;

  model->getLayer("fc", &l);
  c_model->getLayer("fc", &c_l);

  EXPECT_EQ(l->getName(), c_l->getName());
  EXPECT_EQ(l->getType(), c_l->getType());

  std::vector<float *> l_weight;
  std::vector<float *> cl_weight;

  l_weight = l->getWeights();
  cl_weight = c_l->getWeights();

  EXPECT_EQ(l_weight.size(), cl_weight.size());

  c_l->setWeights(l_weight);
  cl_weight = c_l->getWeights();

  for (unsigned int i = 0; i < 10; ++i) {
    // Weight of Fully Connected Layer
    EXPECT_EQ(cl_weight[0][i], l_weight[0][i]);
    // Bias of Fully Connected Layer
    EXPECT_EQ(cl_weight[1][i], l_weight[1][i]);
  }

  // Run after set the weights
  EXPECT_NO_THROW(c_model->train());

  std::vector<float *> one_weights;
  float *w_one = new float[62720 * 10];
  float *b_one = new float[10];
  for (unsigned int i = 0; i < 62720 * 10; ++i)
    w_one[i] = 1.0;
  for (unsigned int i = 0; i < 10; ++i)
    b_one[i] = 1.0;

  one_weights.push_back(w_one);
  one_weights.push_back(b_one);

  c_model->getLayer("fc", &c_l);

  c_l->setWeights(one_weights);
  cl_weight = c_l->getWeights();

  for (unsigned int i = 0; i < 62720 * 10; ++i)
    EXPECT_EQ(cl_weight[0][i], 1.0);
  for (unsigned int i = 0; i < 10; ++i)
    EXPECT_EQ(cl_weight[1][i], 1.0);

  // Run after set the weights again
  EXPECT_NO_THROW(c_model->train());
  delete[] w_one;
  delete[] b_one;
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  try {
    nntrainer::Engine::Global().setWorkingDirectory(getTestResPath(""));
  } catch (std::invalid_argument &e) {
    std::cout << "failed to get test res path\n";
  }

  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cout << "Failed to init gtest" << std::endl;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cout << "Failed to run the tests" << std::endl;
  }

  return result;
}
