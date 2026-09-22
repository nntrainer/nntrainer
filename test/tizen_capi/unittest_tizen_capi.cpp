/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file        unittest_tizen_capi.cc
 * @date        03 April 2020
 * @brief       Unit test utility.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <gtest/gtest.h>
#include <string.h>

#include <nntrainer.h>
#include <nntrainer_internal.h>
#include <nntrainer_test_util.h>

static const std::string getTestResPath(const std::string &file) {
  return getResPath(file, {"test"});
}

static nntrainer::IniSection model_base("Model", "Type = NeuralNetwork"
                                                 " | Epochs = 1"
                                                 " | Loss = cross"
                                                 " | Save_Path = 'model.bin'"
                                                 " | batch_size = 32");

static nntrainer::IniSection optimizer("Optimizer", "Type = adam"
                                                    " | Learning_rate = 0.0001"
                                                    " | Decay_rate = 0.96"
                                                    " | Decay_steps = 1000"
                                                    " | beta1 = 0.9"
                                                    " | beta2 = 0.9999"
                                                    " | epsilon = 1e-7");

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
 * @brief Compare the training statistics
 */
static void nntrainer_capi_model_comp_metrics(ml_train_model_h model,
                                              float train_loss,
                                              float valid_loss,
                                              float valid_accuracy) {
  int status = ML_ERROR_NONE;
  char *summary1, *summary2, *summary3 = nullptr;

  /** Compare training statistics */
  status = ml_train_model_get_summary(
    model, (ml_train_summary_type_e)ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS,
    &summary1);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_NEAR(std::strtof(summary1, nullptr), train_loss, tolerance);
  free(summary1);

  status = ml_train_model_get_summary(
    model, (ml_train_summary_type_e)ML_TRAIN_SUMMARY_MODEL_VALID_LOSS,
    &summary2);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_NEAR(std::strtof(summary2, nullptr), valid_loss, tolerance);
  free(summary2);

  status = ml_train_model_get_summary(
    model, (ml_train_summary_type_e)ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY,
    &summary3);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_NEAR(std::strtof(summary3, nullptr), valid_accuracy, tolerance);
  free(summary3);
}

/**
 * @brief Neural Network Model Contruct / Destruct Test (possitive test )
 */
TEST(nntrainer_capi_nnmodel, construct_destruct_01_p) {
  ml_train_model_h handle;
  int status;
  status = ml_train_model_construct(&handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Destruct Test (negative test )
 */
TEST(nntrainer_capi_nnmodel, construct_destruct_02_n) {
  ml_train_model_h handle = NULL;
  int status;
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Construct (negative test)
 */
TEST(nntrainer_capi_nnmodel, construct_destruct_03_n) {
  int status;
  status = ml_train_model_construct(NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_01_p) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_compile_01_p",
              {model_base, optimizer, dataset, inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_02_n) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_compile_02_n.ini";
  status = ml_train_model_compile(NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, compile_03_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layers[2];
  ml_train_layer_h get_layer;
  ml_train_optimizer_h optimizer;
  ml_train_lr_scheduler_h lr_scheduler;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape=1:1:62720",
                                       "normalization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Find layer based on default name */
  status = ml_train_model_get_layer(model, "input0", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(get_layer, layers[0]);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_initializer=zeros",
    "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
    "weight_initializer=xavier_uniform", "name=fc100", "input_layers=input0",
    NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Find layer based on set name */
  status = ml_train_model_get_layer(model, "fc100", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(get_layer, layers[1]);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(optimizer, "beta1=0.002",
                                           "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_lr_scheduler_create(&lr_scheduler,
                                        ML_TRAIN_LR_SCHEDULER_TYPE_EXPONENTIAL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_lr_scheduler_set_property(
    lr_scheduler, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_lr_scheduler(optimizer, lr_scheduler);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_optimizer(model, optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(model, "loss=cross", "batch_size=32", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, compile_04_n) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layers[3];
  ml_train_layer_h get_layer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape=1:1:62720",
                                       "normalization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Find layer before adding */
  status = ml_train_model_get_layer(model, "input0", &get_layer);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Find layer based on default name */
  status = ml_train_model_get_layer(model, "input0", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Create another layer with same name, different type */
  status = ml_train_layer_set_property(layers[1], "name=input0", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Not add layer with existing name */
  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_set_property(layers[1], "name=fc0", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** add layer with different name, different layer type */
  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Find layer based on default name */
  status = ml_train_model_get_layer(model, "fc0", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[2], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Create another layer with same name, same type */
  status = ml_train_layer_set_property(layers[2], "name=fc0", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** add layer with different name, different layer type */
  status = ml_train_model_add_layer(model, layers[2]);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_set_property(layers[2], "name=fc1", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** add layer with different name, different layer type */
  status = ml_train_model_add_layer(model, layers[2]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_layer(model, "fc1", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, construct_conf_01_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "/test/cannot_find.ini";
  status = ml_train_model_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, construct_conf_02_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_construct_conf_02_n",
              {model_base, optimizer, dataset, inputlayer + "Input_Shape=1:1:0",
               outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_with_single_param_01_p) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_compile_with_single_param_01_p",
              {model_base, optimizer, dataset, inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status =
    ml_train_model_compile_with_single_param(handle, "loss=cross|epochs=2");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_with_single_param_02_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_compile_with_single_param_02_n",
              {model_base, optimizer, dataset, inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status =
    ml_train_model_compile_with_single_param(handle, "loss=cross epochs=2");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_with_single_param_03_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_compile_with_single_param_03_n",
              {model_base, optimizer, dataset, inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status =
    ml_train_model_compile_with_single_param(handle, "loss=cross,epochs=2");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_with_single_param_04_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_compile_with_single_param_04_n",
              {model_base, optimizer, dataset, inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status =
    ml_train_model_compile_with_single_param(handle, "loss=cross!epochs=2");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}
/**
 * @brief Neural Network Model Train Test
 */
TEST(nntrainer_capi_nnmodel, train_01_p) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_train_01_p",
              {model_base + "batch_size = 16", optimizer,
               dataset + "-BufferSize", inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_run(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Compare training statistics */
  nntrainer_capi_model_comp_metrics(handle, 3.911289, 2.933979, 10.4167);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Train Test
 */
TEST(nntrainer_capi_nnmodel, train_02_n) {
  int status = ML_ERROR_NONE;
  status = ml_train_model_run(NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Train Test
 */
TEST(nntrainer_capi_nnmodel, train_03_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;
  ScopedIni s("capi_test_train_03_n",
              {model_base + "batch_size = 16", optimizer,
               dataset + "-BufferSize", inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_run(handle, "loss=cross", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Train Test
 */
TEST(nntrainer_capi_nnmodel, train_with_single_param_01_p) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s(
    "capi_test_train_with_single_param_01_p",
    {model_base, optimizer, dataset + "-BufferSize", inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_train_model_run_with_single_param(handle, "epochs=2|batch_size=16");
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Compare training statistics */
  nntrainer_capi_model_comp_metrics(handle, 3.77080, 3.18020, 10.4167);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Train Test
 */
TEST(nntrainer_capi_nnmodel, train_with_single_param_02_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s(
    "capi_test_train_with_single_param_02_n",
    {model_base, optimizer, dataset + "-BufferSize", inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_run_with_single_param(
    handle, "epochs=2 batch_size=16 save_path=capi_tizen_model.bin");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Train Test
 */
TEST(nntrainer_capi_nnmodel, train_with_single_param_03_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s(
    "capi_test_train_with_single_param_03_n",
    {model_base, optimizer, dataset + "-BufferSize", inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_run_with_single_param(
    handle, "epochs=2,batch_size=16,save_path=capi_tizen_model.bin");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Train Test
 */
TEST(nntrainer_capi_nnmodel, train_with_single_param_04_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s(
    "capi_test_train_with_single_param_04_n",
    {model_base, optimizer, dataset + "-BufferSize", inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_run_with_single_param(
    handle, "epochs=2!batch_size=16!save_path=capi_tizen_model.bin");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_01_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layer, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layer, "input_shape=1:1:6270", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layer, "normalization = true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_02_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layer, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layer, "input_shape=1:1:6270",
                                       "normalization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_03_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layer, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layer, "input_shape=1:1:62720",
                                       "activation=sigmoid", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_destroy(layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_04_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layers[2];

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape=1:1:62720",
                                       "normalization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_initializer=zeros",
    "weight_regularizer=l2norm", "weight_regularizer_constant=0.005", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_destroy(layers[0]);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_destroy(layers[1]);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_05_n) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layer = NULL;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layer);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add layer test
 */
TEST(nntrainer_capi_nnmodel, addLayer_06_n) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = NULL;
  ml_train_layer_h layer = NULL;

  ScopedIni s("capi_test_addLayer_06_n",
              {model_base, optimizer, dataset, inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(model, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layer, ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layer, "unit= 10", "activation=softmax", "bias_initializer=zeros",
    "weight_regularizer=l2norm", "weight_regularizer_constant=0.005", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layer);
  EXPECT_EQ(status, ML_ERROR_NOT_SUPPORTED);

  status = ml_train_layer_destroy(layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Get Layer Test
 */
TEST(nntrainer_capi_nnmodel, getLayer_01_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h add_layer;
  ml_train_layer_h get_layer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&add_layer, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(add_layer, "name=inputlayer", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, add_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_layer(model, "inputlayer", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Get Layer Test
 */
TEST(nntrainer_capi_nnmodel, getLayer_02_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h get_layer;

  std::string default_name = "inputlayer", modified_name = "renamed_inputlayer";
  char *default_summary, *modified_summary = nullptr;

  ScopedIni s("getLayer_02_p", {model_base, inputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_train_model_get_summary(model, ML_TRAIN_SUMMARY_MODEL, &default_summary);
  EXPECT_EQ(status, ML_ERROR_NONE);

  std::string default_summary_str(default_summary);
  EXPECT_NE(default_summary_str.find(default_name), std::string::npos);
  free(default_summary);

  status = ml_train_model_get_layer(model, default_name.c_str(), &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(get_layer,
                                       ("name=" + modified_name).c_str(), NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /// @todo check layer with layer property (#1875)
  status = ml_train_model_get_summary(model, ML_TRAIN_SUMMARY_MODEL,
                                      &modified_summary);
  EXPECT_EQ(status, ML_ERROR_NONE);

  std::string modified_summary_str(modified_summary);
  EXPECT_NE(modified_summary_str.find(modified_name), std::string::npos);
  free(modified_summary);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Get Layer Test
 */
TEST(nntrainer_capi_nnmodel, getLayer_03_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h get_layer;

  ScopedIni s("getLayer_03_p", {model_base, inputlayer, optimizer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(model, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_layer(model, "inputlayer", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Get Layer Test
 */
TEST(nntrainer_capi_nnmodel, getLayer_04_n) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h add_layer;
  ml_train_layer_h get_layer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&add_layer, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(add_layer, "name=inputlayer", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, add_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_layer(model, "unknown", &get_layer);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Get Layer Test
 */
TEST(nntrainer_capi_nnmodel, getLayer_05_n) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h get_layer;

  std::string default_name = "inputlayer", modified_name = "renamed_inputlayer";
  char *default_summary, *modified_summary = nullptr;

  ScopedIni s("getLayer_05_p", {model_base, inputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_train_model_get_summary(model, ML_TRAIN_SUMMARY_MODEL, &default_summary);
  EXPECT_EQ(status, ML_ERROR_NONE);

  std::string default_summary_str(default_summary);
  EXPECT_NE(default_summary_str.find(default_name), std::string::npos);
  free(default_summary);

  status = ml_train_model_get_layer(model, default_name.c_str(), &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(get_layer,
                                       ("name=" + modified_name).c_str(), NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  ///@todo need to fix bug (Unable to get renamed layer)
  status = ml_train_model_get_layer(model, modified_name.c_str(), &get_layer);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Get Weight  Test
 */
TEST(nntrainer_capi_nnmodel, getWeight_01) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;
  const unsigned int MAXDIM = 4;

  ml_tensors_info_h weight_info;
  ml_tensors_data_h weights;
  unsigned int num_weights;
  const unsigned int weight_dim_expected[2][MAXDIM] = {{1, 1, 62720, 10},
                                                       {1, 1, 1, 10}};
  ml_tensor_dimension dim;

  ScopedIni s("capi_test_get_weight_01",
              {model_base, optimizer, dataset, inputlayer, outputlayer});

  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_train_model_get_weight(handle, "outputlayer", &weights, &weight_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count(weight_info, &num_weights);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(num_weights, 2ul);

  for (unsigned int idx = 0; idx < num_weights; ++idx) {
    ml_tensors_info_get_tensor_dimension(weight_info, idx, dim);
    for (unsigned int i = 0; i < MAXDIM; ++i) {
      EXPECT_EQ(dim[i], weight_dim_expected[idx][i]);
    }
  }

  status = ml_tensors_info_destroy(weight_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *t;
  size_t size = 627200 * sizeof(float);

  status = ml_tensors_data_get_tensor_data(weights, 0, (void **)&t, &size);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_tensors_data_destroy(weights);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, create_optimizer_01_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_optimizer_h optimizer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_destroy(optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, create_optimizer_02_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape=1:1:62720",
                                       "normalization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_initializer=zeros",
    "weight_regularizer=l2norm", "weight_regularizer_constant=0.005", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_destroy(optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, create_optimizer_03_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape=1:1:62720",
                                       "normalization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_initializer=zeros",
    "weight_regularizer=l2norm", "weight_regularizer_constant=0.005", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_optimizer(model, optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_destroy(optimizer);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, train_with_file_01_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;
  ml_train_dataset_h dataset;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_train_layer_set_property(layers[0], "input_shape=1:1:62720",
                                "normalization=true", "name=inputlayer", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_initializer=zeros",
    "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
    "weight_initializer=xavier_uniform", "name = fc1",
    "input_layers=inputlayer", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_optimizer(model, optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_create_with_file(
    &dataset, getTestResPath("trainingSet.dat").c_str(),
    getTestResPath("valSet.dat").c_str(), NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property(dataset, "buffer_size=100", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_dataset(model, dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(model, "loss=cross", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_run(model, "epochs=2", "batch_size=16",
                              "save_path=capi_tizen_model.bin", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Compare training statistics */
  nntrainer_capi_model_comp_metrics(model, 2.11176, 2.21936, 16.6667);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, train_with_generator_01_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;
  ml_train_dataset_h dataset;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_train_layer_set_property(layers[0], "input_shape=1:1:62720",
                                "normalization=true", "name=inputlayer", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_initializer=zeros",
    "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
    "weight_initializer=xavier_uniform", "name=fc1", "input_layers=inputlayer",
    NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_optimizer(model, optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  auto train_data = createTrainData();
  auto valid_data = createValidData();

  status = ml_train_dataset_create_with_generator(&dataset, getSample,
                                                  getSample, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property(dataset, "buffer_size=100", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property_for_mode(
    dataset, ML_TRAIN_DATASET_MODE_TRAIN, "user_data", &train_data, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property_for_mode(
    dataset, ML_TRAIN_DATASET_MODE_VALID, "user_data", &valid_data, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_dataset(model, dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(model, "loss=cross", "batch_size=16", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_run(
    model, "epochs=2 | save_path=capi_tizen_model.bin", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Compare training statistics */
  nntrainer_capi_model_comp_metrics(model, 2.20755, 1.98047, 58.3333);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

static int constant_generator_cb(float **outVec, float **outLabel, bool *last,
                                 void *user_data) {
  static int count = 0;
  unsigned int feature_size = 100;
  unsigned int num_class = 10;

  for (unsigned int i = 0; i < feature_size; ++i) {
    outVec[0][i] = 0.0f;
  }

  outLabel[0][0] = 1.0f;
  for (unsigned int j = 1; j < num_class; ++j) {
    outLabel[0][j] = 0.0f;
  }

  count++;
  if (count == 9) {
    *last = true;
    count = 0;
  } else {
    *last = false;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Neural Network Model generator Test
 */
TEST(nntrainer_capi_nnmodel, train_with_generator_02_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = nullptr;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;
  ml_train_dataset_h dataset;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_train_layer_set_property(layers[0], "input_shape=1:1:100",
                                "normalization=true", "name=inputlayer", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_train_layer_set_property(layers[1], "unit=10", "activation=softmax",
                                "name=fc1", "input_layers=inputlayer", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_create_with_generator(
    &dataset, constant_generator_cb, NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property(dataset, "buffer_size=9", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_dataset(model, dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.9", "beta2=0.9999", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_optimizer(model, optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(model, "loss=cross", "batch_size=9", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_run(model, "epochs=1", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Summary Test summary verbosity of tensor
 */
TEST(nntrainer_capi_summary, summary_01_p) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_compile_01_p",
              {model_base, optimizer, dataset, inputlayer, outputlayer});
  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  char *sum = NULL;
  status = ml_train_model_get_summary(handle, ML_TRAIN_SUMMARY_TENSOR, &sum);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_GT(strlen(sum), 90u);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  free(sum);
}

/**
 * @brief Neural Network Model Summary Test summary with tensor
 */
TEST(nntrainer_capi_summary, summary_02_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_compile_01_p",
              {model_base, optimizer, dataset, inputlayer, outputlayer});
  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_summary(handle, ML_TRAIN_SUMMARY_TENSOR, NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_capi_nnmodel, get_input_output_dimension_01_p) {
  ml_train_model_h handle = NULL;
  ml_tensors_info_h input_info, output_info;

  unsigned int input_count, output_count;
  const unsigned int MAXDIM = 4;
  const unsigned int input_dim_expected[MAXDIM] = {32, 1, 1, 62720};
  const unsigned int output_dim_expected[MAXDIM] = {32, 1, 1, 10};
  ml_tensor_dimension dim;

  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_get_input_dimension_01_p",
              {model_base, optimizer, dataset, inputlayer, outputlayer});
  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_input_tensors_info(handle, &input_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count(input_info, &input_count);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(input_count, 1ul);

  ml_tensors_info_get_tensor_dimension(input_info, 0, dim);
  for (unsigned int i = 0; i < MAXDIM; ++i) {
    EXPECT_EQ(dim[i], input_dim_expected[i]);
  }

  status = ml_train_model_get_output_tensors_info(handle, &output_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count(output_info, &output_count);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(output_count, 1ul);

  ml_tensors_info_get_tensor_dimension(output_info, 0, dim);
  for (unsigned int i = 0; i < MAXDIM; ++i) {
    EXPECT_EQ(dim[i], output_dim_expected[i]);
  }

  status = ml_tensors_info_destroy(input_info);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy(output_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_capi_nnmodel, get_input_output_dimension_02_p) {
  ml_train_model_h handle = NULL;
  ml_tensors_info_h input_info, output_info;

  unsigned int input_count, output_count;
  const unsigned int MAXDIM = 4;
  const unsigned int input_dim_expected[MAXDIM] = {32, 1, 1, 62720};
  const unsigned int output_dim_expected[MAXDIM] = {32, 1, 1, 10};
  ml_tensor_dimension dim;

  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_get_input_dimension_02_p",
              {model_base, optimizer, dataset, inputlayer, outputlayer});
  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_input_tensors_info(handle, &input_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_output_tensors_info(handle, &output_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count(input_info, &input_count);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(input_count, 1ul);

  ml_tensors_info_get_tensor_dimension(input_info, 0, dim);
  for (unsigned int i = 0; i < MAXDIM; ++i) {
    EXPECT_EQ(dim[i], input_dim_expected[i]);
  }

  status = ml_tensors_info_get_count(output_info, &output_count);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(output_count, 1ul);

  ml_tensors_info_get_tensor_dimension(output_info, 0, dim);
  for (unsigned int i = 0; i < MAXDIM; ++i) {
    EXPECT_EQ(dim[i], output_dim_expected[i]);
  }

  status = ml_tensors_info_destroy(input_info);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy(output_info);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_capi_nnmodel, get_input_output_dimension_03_n) {
  ml_train_model_h handle = NULL;
  ml_tensors_info_h input_info, output_info;

  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_get_input_dimension_03_n",
              {model_base, optimizer, dataset, inputlayer, outputlayer});
  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(ml_train_model_get_input_tensors_info(handle, &input_info),
            ML_ERROR_INVALID_PARAMETER);
  EXPECT_EQ(ml_train_model_get_output_tensors_info(handle, &output_info),
            ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_capi_nnmodel, get_input_output_dimension_04_n) {
  ml_train_model_h handle = NULL;
  ml_tensors_info_h input_info, output_info;

  EXPECT_EQ(ml_train_model_get_input_tensors_info(handle, &input_info),
            ML_ERROR_INVALID_PARAMETER);
  EXPECT_EQ(ml_train_model_get_output_tensors_info(handle, &output_info),
            ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_capi_nnmodel, get_input_output_dimension_05_n) {
  ml_train_model_h handle = NULL;
  ml_tensors_info_h input_info, output_info;

  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_get_input_dimension_05_n",
              {model_base, optimizer, dataset, inputlayer, outputlayer});
  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  handle = NULL;

  EXPECT_EQ(ml_train_model_get_input_tensors_info(handle, &input_info),
            ML_ERROR_INVALID_PARAMETER);
  EXPECT_EQ(ml_train_model_get_output_tensors_info(handle, &output_info),
            ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_capi_nnmodel, get_input_output_dimension_06_n) {
  ml_train_model_h handle = NULL;
  ml_tensors_info_h input_info, output_info;

  unsigned int input_count, output_count;
  const unsigned int MAXDIM = 4;
  const unsigned int input_dim_expected[MAXDIM] = {32, 1, 1, 62720};
  const unsigned int output_dim_expected[MAXDIM] = {32, 1, 1, 10};
  ml_tensor_dimension dim;

  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_get_input_dimension_06_n",
              {model_base, optimizer, dataset, inputlayer, outputlayer});
  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_input_tensors_info(handle, &input_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count(input_info, &input_count);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(input_count, 1ul);

  ml_tensors_info_get_tensor_dimension(input_info, 0, dim);
  for (unsigned int i = 0; i < MAXDIM; ++i) {
    EXPECT_NE(dim[i], input_dim_expected[i] - 1);
    EXPECT_NE(dim[i], input_dim_expected[i] + 1);
  }

  status = ml_train_model_get_output_tensors_info(handle, &output_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count(output_info, &output_count);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(output_count, 1ul);

  ml_tensors_info_get_tensor_dimension(output_info, 0, dim);
  for (unsigned int i = 0; i < MAXDIM; ++i) {
    EXPECT_NE(dim[i], output_dim_expected[i] - 1);
    EXPECT_NE(dim[i], output_dim_expected[i] + 1);
  }

  status = ml_tensors_info_destroy(input_info);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy(output_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief The tensors info built by the C-API must describe the model
 *        dimension only. A dimension buffer shorter than #ml_tensor_dimension
 *        lets ml-api pick up whatever follows it, which shows up as an
 *        inflated tensor byte size or as a non-empty rank above
 *        TensorDim::MAXDIM.
 */
TEST(nntrainer_capi_nnmodel, get_tensors_info_size_01_p) {
  ml_train_model_h handle = NULL;
  ml_tensors_info_h input_info = NULL;
  ml_tensors_info_h output_info = NULL;
  ml_tensors_info_h weight_info = NULL;
  ml_tensors_data_h weights = NULL;
  size_t data_size;
  ml_tensor_dimension dim = {0};

  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_get_tensors_info_size_01_p",
              {model_base, optimizer, dataset, inputlayer, outputlayer});
  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  ASSERT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_input_tensors_info(handle, &input_info);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = ml_tensors_info_get_tensor_size(input_info, 0, &data_size);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(data_size, 32u * 62720u * sizeof(float));

  status = ml_train_model_get_output_tensors_info(handle, &output_info);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = ml_tensors_info_get_tensor_size(output_info, 0, &data_size);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(data_size, 32u * 10u * sizeof(float));

  status =
    ml_train_model_get_weight(handle, "outputlayer", &weights, &weight_info);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = ml_tensors_info_get_tensor_size(weight_info, 0, &data_size);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(data_size, 62720u * 10u * sizeof(float));
  status = ml_tensors_info_get_tensor_size(weight_info, 1, &data_size);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(data_size, 10u * sizeof(float));

  status = ml_tensors_info_get_tensor_dimension(input_info, 0, dim);
  EXPECT_EQ(status, ML_ERROR_NONE);
  for (size_t i = ml::train::TensorDim::MAXDIM; i < ML_TENSOR_RANK_LIMIT; ++i) {
    EXPECT_EQ(dim[i], 0u);
  }

  status = ml_tensors_data_destroy(weights);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy(weight_info);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy(input_info);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy(output_info);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Get Weight Test with an unknown layer
 */
TEST(nntrainer_capi_nnmodel, getWeight_02_n) {
  ml_train_model_h handle = NULL;
  ml_tensors_info_h weight_info = NULL;
  ml_tensors_data_h weights = NULL;

  int status = ML_ERROR_NONE;

  ScopedIni s("capi_test_get_weight_02_n",
              {model_base, optimizer, dataset, inputlayer, outputlayer});
  status = ml_train_model_construct_with_conf(s.getIniName().c_str(), &handle);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  ASSERT_EQ(status, ML_ERROR_NONE);

  status =
    ml_train_model_get_weight(handle, "no_such_layer", &weights, &weight_info);
  EXPECT_NE(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  try {
    nntrainer::Engine::Global().setWorkingDirectory(getTestResPath(""));
  } catch (std::invalid_argument &e) {
    ml_loge("Failed to get test res path\n");
  }
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    ml_loge("Failed to init gtest\n");
  }

  /** ignore tizen feature check while running the testcases */
  set_feature_state(ML_FEATURE, SUPPORTED);
  set_feature_state(ML_FEATURE_INFERENCE, SUPPORTED);
  set_feature_state(ML_FEATURE_TRAINING, SUPPORTED);

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    ml_loge("Failed to run test.\n");
  }

  /** reset tizen feature check state */
  set_feature_state(ML_FEATURE, NOT_CHECKED_YET);
  set_feature_state(ML_FEATURE_INFERENCE, NOT_CHECKED_YET);
  set_feature_state(ML_FEATURE_TRAINING, NOT_CHECKED_YET);

  return result;
}
