// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   unittest_nntrainer_models.cpp
 * @date   19 Oct 2020
 * @brief  Model multi iteration, itegrated test
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <input_layer.h>
#include <layer.h>
#include <neuralnet.h>

#include <models_golden_test.h>
#include <nntrainer_test_util.h>

static nntrainer::IniSection nn_base("model", "type = NeuralNetwork");
static std::string input_base = "type = input";
static std::string fc_base = "type = Fully_connected";
static std::string conv_base = "type = conv2d | stride = 1,1 | padding = 0,0";
static std::string rnn_base = "type = rnn";
static std::string lstm_base = "type = lstm";
static std::string gru_base = "type = gru";
static std::string pooling_base = "type = pooling2d | padding = 0,0";
static std::string preprocess_flip_base = "type = preprocess_flip";
static std::string preprocess_l2norm_base = "type = preprocess_l2norm";
static std::string preprocess_translate_base = "type = preprocess_translate";
static std::string mse_base = "type = mse";
static std::string cross_base = "type = cross";
static std::string cross_softmax_base = "type = cross_softmax";

static std::string adam_base = "optimizer=adam | beta1 = 0.9 | beta2 = 0.999 | "
                               "epsilon = 1e-7";

static nntrainer::IniSection act_base("activation", "Type = Activation");
static nntrainer::IniSection softmax_base = act_base + "Activation = softmax";
static nntrainer::IniSection sigmoid_base = act_base + "Activation = sigmoid";
static nntrainer::IniSection relu_base = act_base + "Activation = relu";
static nntrainer::IniSection bn_base("bn", "Type=batch_normalization");
static nntrainer::IniSection sgd_base("optimizer", "Type = sgd");

static nntrainer::IniSection nn_base_nhwc = nn_base + "tensor_type=NHWC";
static nntrainer::IniSection nn_base_nchw = nn_base + "tensor_type=NCHW";

using I = nntrainer::IniSection;
using INI = nntrainer::IniWrapper;

/**
 * This is just a wrapper for an ini file with save / erase attached.
 * for example, fc_softmax_mse contains following ini file representation as a
 * series of IniSection
 *
 * [model]
 * Type = NeuralNetwork
 * Learning_rate = 1
 * Optimizer = sgd
 * Loss = mse
 * batch_Size = 3
 *
 * [input_1]
 * Type = input
 * Input_Shape = 1:1:3
 *
 * [dense]
 * Type = fully_connected
 * Unit = 5
 *
 * [activation]
 * Type = Activation
 * Activation = softmax
 *
 * [dense]
 * Type = fully_connected
 * Unit = 10
 *
 * [activation]
 * Type = Activation
 * Activation = softmax
 */
// clang-format off

// TODO: update some models to use loss at the end as a layer
// and check for all cases

INI fc_sigmoid_baseline(
  "fc_sigmoid",
  {nn_base_nchw + "batch_size = 3",
   sgd_base + "learning_rate = 1",
   I("input") + input_base + "input_shape = 1:1:3",
   I("dense") + fc_base + "unit = 5",
   I("act") + sigmoid_base,
   I("dense_1") + fc_base + "unit = 10"});

INI fc_sigmoid_mse =
  INI("fc_sigmoid_mse") + fc_sigmoid_baseline + softmax_base + "model/loss=mse";

INI fc_sigmoid_mse__1 =
  INI("fc_sigmoid_mse__1") + fc_sigmoid_baseline + softmax_base +  I("loss", mse_base);

INI fc_sigmoid_baseline_clipped_at_0(
  "fc_sigmoid",
  {nn_base + "batch_size = 3",
   sgd_base + "learning_rate = 1",
   I("input") + input_base + "input_shape = 1:1:3",
   I("dense") + fc_base + "unit = 5" + "clip_grad_by_norm = 0.0",
   I("act") + sigmoid_base,
   I("dense_1") + fc_base + "unit = 10" + "clip_grad_by_norm = 0.0"});

INI fc_sigmoid_mse__2 =
  INI("fc_sigmoid_mse__2") + fc_sigmoid_baseline_clipped_at_0 + softmax_base +  I("loss", mse_base);

INI fc_sigmoid_baseline_clipped_too_high(
  "fc_sigmoid",
  {nn_base + "batch_size = 3",
   sgd_base + "learning_rate = 1",
   I("input") + input_base + "input_shape = 1:1:3",
   I("dense") + fc_base + "unit = 5" + "clip_grad_by_norm = 10000.0",
   I("act") + sigmoid_base,
   I("dense_1") + fc_base + "unit = 10" + "clip_grad_by_norm = 10000.0"});

INI fc_sigmoid_mse__3 =
  INI("fc_sigmoid_mse__3") + fc_sigmoid_baseline_clipped_too_high + softmax_base +  I("loss", mse_base);

INI fc_sigmoid_cross =
  INI("fc_sigmoid_cross") + fc_sigmoid_baseline + softmax_base + "model/loss=cross";

INI fc_sigmoid_cross__1 =
  INI("fc_sigmoid_cross__1") + fc_sigmoid_baseline + I("loss", cross_softmax_base);

INI fc_relu_baseline(
  "fc_relu",
  {nn_base + "Loss=mse | batch_size = 3",
   sgd_base + "learning_rate = 0.1",
   I("input") + input_base + "input_shape = 1:1:3",
   I("dense") + fc_base + "unit = 10",
   I("act") + relu_base,
   I("dense_1") + fc_base + "unit = 2",
   I("act_1") + sigmoid_base + "input_layers=dense" + "input_layers=dense_1"});

INI fc_relu_mse =
  INI("fc_relu_mse") + fc_relu_baseline + "model/loss=mse";

INI fc_relu_mse__1 =
  INI("fc_relu_mse__1") + fc_relu_baseline + I("loss", mse_base);

INI fc_leaky_relu_mse = INI("fc_relu_leaky_relu") + fc_relu_baseline + "act/activation=leaky_relu";

INI fc_bn_sigmoid_cross(
  "fc_bn_sigmoid_cross",
  {nn_base + "loss=cross | batch_size = 3",
   sgd_base + "learning_rate = 1",
   I("input") + input_base + "input_shape = 1:1:3",
   I("dense") + fc_base + "unit = 10" + "input_layers=input",
   I("bn") + bn_base + "input_layers=dense",
   I("act") + sigmoid_base + "input_layers=bn",
   I("dense_2") + fc_base + "unit = 10" + "input_layers=act",
   I("act_3") + softmax_base + "input_layers=dense_2"});

INI fc_bn_sigmoid_mse =
  INI("fc_bn_sigmoid_mse") + fc_bn_sigmoid_cross + "model/loss=mse";

std::string mnist_pooling =
  pooling_base + "| pool_size=2,2 | stride=2,2 | pooling=average | padding=0,0";

INI mnist_conv_cross(
  "mnist_conv_cross",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("conv2d_c1_layer") + conv_base + "kernel_size=3,4 | filters=2" +"input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("pool_1") + mnist_pooling+"input_layers=act_1",
    I("flatten", "type=flatten")+"input_layers=pool_1" ,
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_3") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_1x1(
  "conv_1x1",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("conv2d_c1_layer") + conv_base + "kernel_size=1,1 | filters=4",
    I("act_1") + sigmoid_base,
    I("flatten", "type=flatten") ,
    I("outputlayer") + fc_base + "unit = 10",
    I("act_2") + softmax_base
  }
);

INI conv_input_matches_kernel(
  "conv_input_matches_kernel",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("conv2d_c1_layer") + conv_base + "kernel_size=4,5 | filters=4" +"input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_basic(
  "conv_basic",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:5:3",
    I("conv2d_c1") + conv_base +
            "kernel_size = 3,3 | filters=4" + "input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_same_padding(
  "conv_same_padding",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:5:3",
    I("conv2d_c1") + conv_base +
            "kernel_size = 3,3 | filters=4 | padding =same" + "input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_multi_stride(
  "conv_multi_stride",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:5:3",
    I("conv2d_c1") + conv_base +
            "kernel_size = 3,3 | filters=4 | stride=2,2" + "input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_uneven_strides(
  "conv_uneven_strides",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("conv2d_c1") + conv_base +
            "kernel_size = 3,3 | filters=4 | stride=3,3" + "input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_uneven_strides2(
  "conv_uneven_strides2",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
    I("input") + input_base + "input_shape=2:4:4",
    I("conv2d_c1") + conv_base + "kernel_size = 2,2 | filters=2 | stride=1,2",
    I("act_1") + sigmoid_base,
    I("flatten", "type=flatten"),
    I("outputlayer") + fc_base + "unit = 10",
    I("act_2") + softmax_base
  }
);

INI conv_uneven_strides3(
  "conv_uneven_strides3",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
    I("input") + input_base + "input_shape=2:4:4",
    I("conv2d_c1") + conv_base + "kernel_size = 2,2 | filters=2 | stride=2,1",
    I("act_1") + sigmoid_base,
    I("flatten", "type=flatten"),
    I("outputlayer") + fc_base + "unit = 10",
    I("act_2") + softmax_base
  }
);

INI conv_bn(
  "conv_bn",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
    I("input_layer") + input_base + "input_shape=2:3:5",
    I("conv2d_c1") + conv_base + "kernel_size = 2,2 | filters=2",
    I("bn") + bn_base,
    I("act_1") + relu_base,
    I("flatten", "type=flatten"),
    I("outputlayer") + fc_base + "unit = 10",
    I("act_2") + softmax_base
  }
);

INI conv_same_padding_multi_stride(
  "conv_same_padding_multi_stride",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:5:3",
    I("conv2d_c1") + conv_base +
            "kernel_size = 3,3 | filters=4 | stride=2,2 | padding=same" + "input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_no_loss(
  "conv_no_loss",
  {
    nn_base + "batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("conv2d_c1_layer") + conv_base + "kernel_size=4,5 | filters=4" +"input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("flatten", "type=flatten")+"input_layers=act_1" ,
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI pooling_max_same_padding(
  "pooling_max_same_padding",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=max | pool_size = 3,3 | padding =same" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_max_same_padding_multi_stride(
  "pooling_max_same_padding_multi_stride",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:3:5",
    I("pooling_1") + pooling_base +
            "pooling=max | pool_size = 3,3 | padding =1 | stride=2,2" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_max_valid_padding(
  "pooling_max_valid_padding",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=max | pool_size = 3,3 | padding =valid" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_avg_same_padding(
  "pooling_avg_same_padding",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=average | pool_size = 3,3 | padding =1,1,1,1" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_avg_valid_padding(
  "pooling_avg_valid_padding",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=average | pool_size = 3,3 | padding =valid" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_avg_same_padding_multi_stride(
  "pooling_avg_same_padding_multi_stride",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:3:5",
    I("pooling_1") + pooling_base +
            "pooling=average | pool_size = 3,3 | padding =same | stride=2,2" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_global_avg(
  "pooling_global_avg",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=global_average" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_global_max(
  "pooling_global_max",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=global_max" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI preprocess_flip_validate(
  "preprocess_flip_validate",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("preprocess_flip") + preprocess_flip_base +
            "flip_direction=vertical" + "input_layers=input",
    I("conv2d_c1_layer") + conv_base + "kernel_size=3,4 | filters=2" +"input_layers=preprocess_flip",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("pool_1") + mnist_pooling+"input_layers=act_1",
    I("flatten", "type=flatten")+"input_layers=pool_1" ,
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_3") + softmax_base +"input_layers=outputlayer"
  }
);

INI preprocess_translate(
  "preprocess_translate",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("preprocess_translate") + preprocess_translate_base +
            "random_translate=0.5" + "input_layers=input",
    I("conv2d_c1_layer") + conv_base + "kernel_size=3,4 | filters=2" +"input_layers=preprocess_translate",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("pool_1") + mnist_pooling+"input_layers=act_1",
    I("flatten", "type=flatten")+"input_layers=pool_1" ,
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_3") + softmax_base +"input_layers=outputlayer"
  }
);

INI preprocess_l2norm_validate(
  "preprocess_l2norm_validate",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:1:20",
    I("preprocess_l2norm") + preprocess_l2norm_base + "input_layers=input",
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=preprocess_l2norm",
    I("act_3") + softmax_base +"input_layers=outputlayer"
  }
);


INI mnist_conv_cross_one_input = INI("mnist_conv_cross_one_input") + mnist_conv_cross + "model/batch_size=1";

INI fc_softmax_mse_distribute(
  "fc_softmax_mse_distribute",
  {
    nn_base + "loss=mse | batch_size = 3",
    sgd_base + "learning_rate = 1",
    I("input") + input_base + "input_shape = 1:5:5",
    I("dense") + fc_base + "unit = 3"+"activation=softmax"+"distribute=true"
  }
);

INI fc_softmax_cross_distribute(
  "fc_softmax_cross_distribute",
  {
    nn_base + "loss=cross | batch_size = 3",
    sgd_base + "learning_rate = 1",
    I("input") + input_base + "input_shape = 1:5:5",
    I("dense") + fc_base + "unit = 3"+"activation=softmax"+"distribute=true"
  }
);

INI fc_sigmoid_cross_distribute(
  "fc_sigmoid_cross_distribute",
  {
    nn_base + "loss=cross | batch_size = 3",
    sgd_base + "learning_rate = 1",
    I("input") + input_base + "input_shape = 1:5:5",
    I("dense") + fc_base + "unit = 3"+"activation=sigmoid"+"distribute=true"
  }
);

INI addition_resnet_like(
  "addition_resnet_like",
  {
    nn_base + "loss=mse | batch_size = 3",
    sgd_base + "learning_rate = 0.1",
    I("x") + input_base + "input_shape = 2:3:5",
    I("addition_a1") + conv_base
      + "filters=4 | kernel_size=3,3 | stride=2,2 | padding=1,1",
    I("addition_a2") + relu_base,
    I("addition_a3") + conv_base + "filters=4 | kernel_size=3,3 | padding=1,1",
    I("addition_b1") + conv_base
      + "filters=4 | kernel_size=1,1 | stride=2,2"
      + "input_layers=x",
    I("addition_c1", "type=addition | input_layers=addition_a3, addition_b1"),
    I("addition_c2", "type=flatten"),
    I("addition_c3") + fc_base + "unit=10",
    I("addition_c4") + softmax_base,
  }
);

INI addition_resnet_like__1(
  "addition_resnet_like__1",
  {
    nn_base + "loss=mse | batch_size = 3",
    sgd_base + "learning_rate = 0.1",
    I("x") + input_base + "input_shape = 2:3:5",
    I("addition_a1") + conv_base
      + "filters=4 | kernel_size=3,3 | stride=2,2 | padding=1,1",
    I("addition_a2") + relu_base,
    I("addition_a3") + conv_base + "filters=4 | kernel_size=3,3 | padding=1,1",
    I("addition_b1") + conv_base
      + "filters=4 | kernel_size=1,1 | stride=2,2"
      + "input_layers=x",
    I("identity", "type=identity | input_layers=addition_a3, addition_b1"),
    I("addition_c1", "type=addition | input_layers=identity(0), identity(1)"),
    I("addition_c2", "type=flatten"),
    I("addition_c3") + fc_base + "unit=10",
    I("addition_c4") + softmax_base,
  }
);

INI lstm_basic(
  "lstm_basic",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:1:1",
    I("lstm") + lstm_base +
      "unit = 1" + "input_layers=input" + "integrate_bias=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=lstm"
  }
);

INI lstm_return_sequence(
  "lstm_return_sequence",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("lstm") + lstm_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=lstm"
  }
);

INI lstm_return_sequence_with_batch(
  "lstm_return_sequence_with_batch",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("lstm") + lstm_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=lstm"
  }
);
INI rnn_basic(
  "rnn_basic",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:1:1",
    I("rnn") + rnn_base +
      "unit = 2" + "input_layers=input" + "integrate_bias=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=rnn"
  }
);

INI rnn_return_sequences(
  "rnn_return_sequences",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("rnn") + rnn_base +
      "unit = 2" + "input_layers=input" + "return_sequences=true" + "integrate_bias=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=rnn"
  }
);

INI multi_lstm_return_sequence(
  "multi_lstm_return_sequence",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("lstm") + lstm_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true",
    I("lstm2") + lstm_base +
      "unit = 2" + "input_layers=lstm" + "integrate_bias=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=lstm2"
  }
);

INI multi_lstm_return_sequence_with_batch(
  "multi_lstm_return_sequence_with_batch",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("lstm") + lstm_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true",
    I("lstm2") + lstm_base +
      "unit = 2" + "input_layers=lstm" + "integrate_bias=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=lstm2"
  }
);

INI rnn_return_sequence_with_batch(
  "rnn_return_sequence_with_batch",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("rnn") + rnn_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=rnn"
  }
);

INI multi_rnn_return_sequence(
  "multi_rnn_return_sequence",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("rnn") + rnn_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true",
    I("rnn2") + rnn_base +
      "unit = 2" + "input_layers=rnn" + "integrate_bias=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=rnn2"
  }
);

INI multi_rnn_return_sequence_with_batch(
  "multi_rnn_return_sequence_with_batch",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("rnn") + rnn_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true",
    I("rnn2") + rnn_base +
      "unit = 2" + "input_layers=rnn" + "integrate_bias=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=rnn2"
  }
);

INI gru_basic(
  "gru_basic",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:1:4",
    I("gru") + gru_base +
      "unit = 3" + "input_layers=input" + "integrate_bias=true" + "reset_after=false",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=gru"
  }
);

INI gru_return_sequence(
  "gru_return_sequence",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:4",
    I("gru") + gru_base +
      "unit = 3" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true" + "reset_after=false",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=gru"
  }
);

INI gru_return_sequence_with_batch(
  "gru_return_sequence_with_batch",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:4",
    I("gru") + gru_base +
      "unit = 3" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true" + "reset_after=false",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=gru"
  }
);

INI multi_gru_return_sequence(
  "multi_gru_return_sequence",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:4",
    I("gru") + gru_base +
      "unit = 3" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true" + "reset_after=false",
    I("gru2") + gru_base +
      "unit = 3" + "input_layers=gru" + "integrate_bias=true" + "reset_after=false",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=gru2"
  }
);

INI multi_gru_return_sequence_with_batch(
  "multi_gru_return_sequence_with_batch",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:4",
    I("gru") + gru_base +
      "unit = 3" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=true" + "reset_after=false",
    I("gru2") + gru_base +
      "unit = 3" + "input_layers=gru" + "integrate_bias=true" + "reset_after=false",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=gru2"
  }
);

// Check reset_after
INI gru_reset_after_basic(
  "gru_reset_after_basic",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:1:4",
    I("gru") + gru_base +
      "unit = 3" + "input_layers=input" + "integrate_bias=false" + "reset_after=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=gru"
  }
);

INI gru_reset_after_return_sequence(
  "gru_reset_after_return_sequence",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:4",
    I("gru") + gru_base +
      "unit = 3" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=false" + "reset_after=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=gru"
  }
);

INI gru_reset_after_return_sequence_with_batch(
  "gru_reset_after_return_sequence_with_batch",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:4",
    I("gru") + gru_base +
      "unit = 3" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=false" + "reset_after=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=gru"
  }
);

INI multi_gru_reset_after_return_sequence(
  "multi_gru_reset_after_return_sequence",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:4",
    I("gru") + gru_base +
      "unit = 3" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=false" + "reset_after=true",
    I("gru2") + gru_base +
      "unit = 3" + "input_layers=gru" + "integrate_bias=false" + "reset_after=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=gru2"
  }
);

INI multi_gru_reset_after_return_sequence_with_batch(
  "multi_gru_reset_after_return_sequence_with_batch",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:4",
    I("gru") + gru_base +
      "unit = 3" + "input_layers=input"+ "return_sequences=true" + "integrate_bias=false" + "reset_after=true",
    I("gru2") + gru_base +
      "unit = 3" + "input_layers=gru" + "integrate_bias=false" + "reset_after=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=gru2"
  }
);

INI multiple_output_model(
  "multiple_output_model",
  {
    nn_base + "loss=mse | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("x") + input_base + "input_shape = 2:3:5",
    I("multiout_a1") + conv_base
      + "filters=4 | kernel_size=3,3 | stride=2,2 | padding=1,1",
    I("multiout_a2") + relu_base,
    I("multiout_a3") + conv_base + "filters=4 | kernel_size=3,3 | padding=1,1",
    I("multiout_a4", "type=flatten"),
    I("multiout_a5") + fc_base + "unit=10",
    I("multiout_a6") + softmax_base,
    I("multiout_b1") + conv_base
      + "filters=4 | kernel_size=1,1 | stride=2,2"
      + "input_layers=x",
    I("multiout_b2", "type=flatten"),
    I("multiout_b3") + fc_base + "unit=10",
    I("multiout_b4") + softmax_base
  }
);

INI multiout_model(
  "multiout_model",
  {
    nn_base + "loss=mse | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("x") + input_base + "input_shape = 1:10",
    I("fc") + fc_base + "unit = 2",
    I("fc1") + fc_base
      + "unit=2 | input_layers=fc",
    I("fc2") + fc_base
      + "unit=2 | input_layers=fc",
    I("add1", "type=addition | input_layers=fc1, fc2"),
    I("fc3") + fc_base + "unit=3",
    I("sm") + softmax_base
  }
);

/**
 * @brief helper function to make model testcase
 *
 * @param nntrainer::TensorDim label dimension
 * @param int Iteration
 * @param options options
 */
auto mkResNet18Tc(const unsigned int iteration,
               ModelTestOption options = ModelTestOption::ALL) {
  unsigned int batch_size = 2;
  unsigned int num_class = 100;
  unsigned int count = 0;
  nntrainer::IniWrapper::Sections layers;

  /** get unique name for a layer */
  auto getName = [&count]() -> std::string {
    if (count == 21)
      std::cout << "mimatch" << std::endl;
    return "layer" + std::to_string(++count);
    };
  auto getPreviousName = [&count]() -> std::string { return "layer" + std::to_string(count); };

  /** add blocks */
  auto addBlock = [&layers, &getName, &getPreviousName] (
    unsigned int filters, unsigned int kernel_size, bool downsample) {
    std::string filter_str = "filters=" + std::to_string(filters);
    std::string kernel_str = "kernel_size=" + std::to_string(kernel_size) + "," + std::to_string(kernel_size);
    std::string kernel1_str = "kernel_size=1,1";
    std::string stride1_str = "stride=1,1";
    std::string stride2_str = "stride=2,2";
    std::string padding_str = "padding=same";
    std::string input_name = getPreviousName();
    std::string in_layer_str = "input_layers=" + input_name;
    std::string stride_str = stride1_str;
    if (downsample)
      stride_str = stride2_str;

    /** skip connection */
    std::string b1_name = input_name;
    if (downsample) {
      b1_name = getName();
      layers.push_back(I(b1_name) + conv_base + filter_str +
      kernel1_str + stride_str + padding_str + in_layer_str);
    }

    /** main connection */
    layers.push_back(I(getName()) + conv_base + filter_str +
    kernel_str + stride_str + padding_str + in_layer_str);
    layers.push_back(I(getName()) + bn_base);
    layers.push_back(I(getName()) + relu_base);
    std::string a1_name = getName();
    layers.push_back(I(a1_name) + conv_base + filter_str +
    kernel_str + stride1_str + padding_str);

    /** add the two connections */
    layers.push_back(I(getName()) + "type=addition" + ("input_layers=" + b1_name + "," + a1_name));
    layers.push_back(I(getName()) + bn_base);
    layers.push_back(I(getName()) + relu_base);
  };

  layers.push_back(nn_base + ("loss=cross | batch_size = " + std::to_string(batch_size)));
  layers.push_back(sgd_base + "learning_rate = 0.1");
  /** prefix for resnet model */
  layers.push_back(I(getName()) + input_base + "input_shape = 3:32:32");
  layers.push_back(I(getName()) + conv_base + "kernel_size=3,3 | filters=64 | padding=same");
  layers.push_back(I(getName()) + bn_base);
  layers.push_back(I(getName()) + relu_base);
  /** add all the blocks */
  addBlock(64, 3, false);
  addBlock(64, 3, false);
  addBlock(128, 3, true);
  addBlock(128, 3, false);
  addBlock(256, 3, true);
  addBlock(256, 3, false);
  addBlock(512, 3, true);
  addBlock(512, 3, false);
  /** add suffix for resnet model */
  layers.push_back(I(getName()) + pooling_base + "pooling = average | pool_size=4,4");
  layers.push_back(I(getName()) + "type=flatten");
  layers.push_back(I(getName()) + fc_base + "unit=100");
  layers.push_back(I(getName()) + softmax_base);

  return std::tuple<const nntrainer::IniWrapper, const nntrainer::TensorDim,
                    const unsigned int, ModelTestOption>(
    nntrainer::IniWrapper("ResNet18", layers), nntrainer::TensorDim({batch_size, 1,1, num_class}), iteration, options);
}

GTEST_PARAMETER_TEST(
  nntrainerModelAutoTests, nntrainerModelTest, ::testing::ValuesIn(
    {
      mkModelIniTc(fc_sigmoid_mse, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(fc_sigmoid_mse__1, "3:1:1:10", 1, ModelTestOption::ALL),
      mkModelIniTc(fc_sigmoid_mse__2, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(fc_sigmoid_mse__3, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(fc_sigmoid_cross, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(fc_sigmoid_cross__1, "3:1:1:10", 1, ModelTestOption::ALL),
      mkModelIniTc(fc_relu_mse, "3:1:1:2", 10, ModelTestOption::ALL),
      mkModelIniTc(fc_leaky_relu_mse, "3:1:1:2", 10, ModelTestOption::SAVE_AND_LOAD_INI),
      mkModelIniTc(fc_relu_mse__1, "3:1:1:2", 1, ModelTestOption::ALL),
      /// @todo bn with custom initializer
      mkModelIniTc(fc_bn_sigmoid_cross, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(fc_bn_sigmoid_mse, "3:1:1:10", 10, ModelTestOption::ALL),

      /**< single conv2d layer test */
      mkModelIniTc(conv_1x1, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(conv_input_matches_kernel, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(conv_basic, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(conv_same_padding, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(conv_multi_stride, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(conv_uneven_strides, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(conv_uneven_strides2, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(conv_uneven_strides3, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(conv_bn, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(conv_same_padding_multi_stride, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(conv_no_loss, "3:1:1:10", 1, ModelTestOption::NO_THROW_RUN),

      /**< single pooling layer test */
      mkModelIniTc(pooling_max_same_padding, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(pooling_max_same_padding_multi_stride, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(pooling_max_valid_padding, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(pooling_avg_same_padding, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(pooling_avg_same_padding_multi_stride, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(pooling_avg_valid_padding, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(pooling_global_avg, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(pooling_global_max, "3:1:1:10", 10, ModelTestOption::ALL),

      /**< conv pool combined tests */
      mkModelIniTc(mnist_conv_cross, "3:1:1:10", 10, ModelTestOption::ALL),
      mkModelIniTc(mnist_conv_cross_one_input, "1:1:1:10", 10, ModelTestOption::ALL),

      /**< augmentation layer */
      mkModelIniTc(preprocess_flip_validate, "3:1:1:10", 10, ModelTestOption::NO_THROW_RUN),

      mkModelIniTc(preprocess_l2norm_validate, "3:1:1:10", 10, ModelTestOption::NO_THROW_RUN),

      /**< Addition test */
      mkModelIniTc(addition_resnet_like, "3:1:1:10", 10, ModelTestOption::COMPARE), // Todo: Enable option to ALL
      mkModelIniTc(addition_resnet_like__1, "3:1:1:10", 10, ModelTestOption::COMPARE), // Todo: Enable option to ALL

      /** Multiout test */
      mkModelIniTc(multiout_model, "3:1:1:3", 10, ModelTestOption::COMPARE), // Todo: Enable option to ALL

      /// #1192 time distribution inference bug
      mkModelIniTc(fc_softmax_mse_distribute, "3:1:5:3", 1, ModelTestOption::NO_THROW_RUN),
      mkModelIniTc(fc_softmax_cross_distribute, "3:1:5:3", 1, ModelTestOption::NO_THROW_RUN),
      mkModelIniTc(fc_sigmoid_cross_distribute, "3:1:5:3", 1, ModelTestOption::NO_THROW_RUN),
      mkModelIniTc(lstm_basic, "1:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(lstm_return_sequence, "1:1:2:1", 10, ModelTestOption::ALL),
      mkModelIniTc(lstm_return_sequence_with_batch, "2:1:2:1", 10, ModelTestOption::ALL),
      mkModelIniTc(multi_lstm_return_sequence, "1:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(multi_lstm_return_sequence_with_batch, "2:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(rnn_basic, "1:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(rnn_return_sequences, "1:1:2:1", 10, ModelTestOption::ALL),
      mkModelIniTc(rnn_return_sequence_with_batch, "2:1:2:1", 10, ModelTestOption::ALL),
      mkModelIniTc(multi_rnn_return_sequence, "1:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(multi_rnn_return_sequence_with_batch, "2:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(gru_basic, "1:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(gru_return_sequence, "1:1:2:1", 10, ModelTestOption::ALL),
      mkModelIniTc(gru_return_sequence_with_batch, "2:1:2:1", 10, ModelTestOption::ALL),
      mkModelIniTc(multi_gru_return_sequence, "1:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(multi_gru_return_sequence_with_batch, "2:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(gru_reset_after_basic, "1:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(gru_reset_after_return_sequence, "1:1:2:1", 10, ModelTestOption::ALL),
      mkModelIniTc(gru_reset_after_return_sequence_with_batch, "2:1:2:1", 10, ModelTestOption::ALL),
      mkModelIniTc(multi_gru_reset_after_return_sequence, "1:1:1:1", 10, ModelTestOption::ALL),
      mkModelIniTc(multi_gru_reset_after_return_sequence_with_batch, "2:1:1:1", 10, ModelTestOption::ALL),

      /**< multi output test */
      mkModelIniTc(multiple_output_model, "3:1:1:10", 10, ModelTestOption::COMPARE) // Todo: Enable option to ALL
      /** resnet model */
      // this must match training (verify only forwarding output values) for 2 iterations with tolerance 1.2e-4
      // mkResNet18Tc(2, ModelTestOption::COMPARE)
    }
), [](const testing::TestParamInfo<nntrainerModelTest::ParamType>& info) -> const auto &{
 return std::get<1>(info.param);
});
// clang-format on

#if defined(ENABLE_DATA_AUGMENTATION_OPENCV)
GTEST_PARAMETER_TEST(
  nntrainerModelAutoTestsAugmentationCV, nntrainerModelTest,
  ::testing::ValuesIn({mkModelIniTc(preprocess_translate, "3:1:1:10", 10,
                                    ModelTestOption::NO_THROW_RUN)}),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info)
    -> const auto & { return std::get<1>(info.param); });
#endif

/**
 * @brief Read or save the model before initialize
 */
TEST(nntrainerModels, read_save_01_n) {
  nntrainer::NeuralNetwork NN;
  std::shared_ptr<nntrainer::LayerNode> layer_node =
    nntrainer::createLayerNode(nntrainer::InputLayer::type,
                               {"input_shape=1:1:62720", "normalization=true"});

  EXPECT_NO_THROW(NN.addLayer(layer_node));
  EXPECT_NO_THROW(NN.setProperty({"loss=mse"}));

  EXPECT_THROW(NN.load("model.bin"), std::runtime_error);
  EXPECT_THROW(NN.save("model.bin"), std::runtime_error);

  EXPECT_EQ(NN.compile(), ML_ERROR_NONE);

  EXPECT_THROW(NN.load("model.bin"), std::runtime_error);
  EXPECT_THROW(NN.save("model.bin"), std::runtime_error);
}

TEST(nntrainerModels, loadFromLayersBackbone_p) {
  std::vector<std::shared_ptr<ml::train::Layer>> reference;
  reference.emplace_back(
    ml::train::layer::FullyConnected({"name=fc1", "input_shape=3:1:2"}));
  reference.emplace_back(
    ml::train::layer::FullyConnected({"name=fc2", "input_layers=fc1"}));

  nntrainer::NeuralNetwork nn;
  nn.addWithReferenceLayers(reference, "backbone", {}, {"fc1"}, {"fc2"},
                            ml::train::ReferenceLayersType::BACKBONE, {});

  nn.compile();
  auto graph = nn.getFlatGraph();
  for (unsigned int i = 0; i < graph.size(); ++i) {
    EXPECT_EQ(graph.at(i)->getName(), "backbone/" + reference.at(i)->getName());
  };
}

TEST(nntrainerModels, loadFromLayersRecurrent_p) {
  std::vector<std::shared_ptr<ml::train::Layer>> reference;
  reference.emplace_back(ml::train::layer::FullyConnected({"name=fc1"}));
  reference.emplace_back(
    ml::train::layer::FullyConnected({"name=fc2", "input_layers=fc1"}));

  nntrainer::NeuralNetwork nn;
  nn.addWithReferenceLayers(reference, "recurrent", {"out_source"}, {"fc1"},
                            {"fc2"}, ml::train::ReferenceLayersType::RECURRENT,
                            {
                              "unroll_for=3",
                              "as_sequence=fc2",
                              "recurrent_input=fc1",
                              "recurrent_output=fc2",
                            });

  std::vector<std::string> expected_node_names = {
    "recurrent/fc1/0",        "recurrent/fc2/0", "recurrent/fc1/1",
    "recurrent/fc2/1",        "recurrent/fc1/2", "recurrent/fc2/2",
    "recurrent/fc2/concat_0", "recurrent/fc2"};
  std::vector<std::string> expected_input_layers = {
    "out_source" /**< input added with external_input */,
    "recurrent/fc1/0",
    "recurrent/fc2/0",
    "recurrent/fc1/1",
    "recurrent/fc2/1",
    "recurrent/fc1/2",
    "recurrent/fc2/0" /**< out source's first input */,
    "recurrent/fc2/concat_0", /**< identity's input */
  };

  auto graph = nn.getFlatGraph();
  for (unsigned int i = 0; i < graph.size(); ++i) {
    /// comment below intended
    // std::cout << *graph.at(i);
    EXPECT_EQ(graph.at(i)->getName(), expected_node_names.at(i)) << "at " << i;
    EXPECT_EQ(graph.at(i)->getInputConnectionName(0),
              expected_input_layers.at(i))
      << "at " << i;
  };
}

/**
 * @brief Regression test for incremental_inference step selection.
 *
 * For sequence outputs, output_hidden_state=false must return the last step.
 */
TEST(nntrainerModels, incrementalInferenceReturnsLastStep_p) {
  nntrainer::NeuralNetwork nn;

  fc_softmax_mse_distribute.save_ini();
  nn.load(fc_softmax_mse_distribute.getIniName(),
          ml::train::ModelFormat::MODEL_FORMAT_INI);
  fc_softmax_mse_distribute.erase_ini();

  EXPECT_EQ(nn.compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn.initialize(), ML_ERROR_NONE);

  auto in_dim = nn.getInputDimension();
  auto out_dim = nn.getOutputDimension();
  ASSERT_EQ(in_dim.size(), 1u);
  ASSERT_EQ(out_dim.size(), 1u);
  ASSERT_GT(out_dim[0].height(), 1u);

  const unsigned int width = out_dim[0].width();
  const unsigned int sequence_len = out_dim[0].height();
  std::vector<float> input(in_dim[0].getDataLen(), 0.0f);
  std::vector<float *> input_ptr{input.data()};
  std::vector<float> first_step(width);
  std::vector<float> last_step(width);

  bool distinguished_steps = false;
  for (unsigned int trial = 0; trial < 4; ++trial) {
    for (unsigned int i = 0; i < input.size(); ++i) {
      input[i] = static_cast<float>((trial + 1) * (i + 1));
    }

    auto full_hidden =
      nn.incremental_inference(1, input_ptr, {}, sequence_len, 0, sequence_len,
                               true);
    ASSERT_EQ(full_hidden.size(), 1u);

    for (unsigned int i = 0; i < width; ++i) {
      first_step[i] = full_hidden[0][i];
      last_step[i] = full_hidden[0][(sequence_len - 1) * width + i];
    }

    for (unsigned int i = 0; i < width; ++i) {
      if (std::fabs(first_step[i] - last_step[i]) > 1e-5f) {
        distinguished_steps = true;
        break;
      }
    }
    if (distinguished_steps)
      break;
  }

  ASSERT_TRUE(distinguished_steps)
    << "Failed to generate distinguishable first/last sequence outputs";

  auto reduced =
    nn.incremental_inference(1, input_ptr, {}, sequence_len, 0, sequence_len,
                             false);
  ASSERT_EQ(reduced.size(), 1u);

  for (unsigned int i = 0; i < width; ++i) {
    EXPECT_NEAR(reduced[0][i], last_step[i], 1e-5f)
      << "Mismatch at width index " << i;
  }
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
