// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_lstm.cpp
 * @date 8 July 2021
 * @brief LSTM Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <lstm.h>

auto semantic_lstm = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>, nntrainer::LSTMLayer::type,
  {"unit=1"}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false,
  1);

GTEST_PARAMETER_TEST(LSTM, LayerSemantics, ::testing::Values(semantic_lstm));

auto lstm_single_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true"}, "3:1:1:7",
  "lstm_single_step.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto lstm_multi_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true"}, "3:1:4:7", "lstm_multi_step.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto lstm_single_step_seq = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true", "return_sequences=true"}, "3:1:1:7",
  "lstm_single_step_seq.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto lstm_multi_step_seq = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true", "return_sequences=true"}, "3:1:4:7",
  "lstm_multi_step_seq.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto lstm_multi_step_seq_act_orig = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true", "return_sequences=true",
   "hidden_state_activation=tanh", "recurrent_activation=sigmoid"},
  "3:1:4:7", "lstm_multi_step_seq.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto lstm_multi_step_seq_act = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true", "return_sequences=true",
   "hidden_state_activation=sigmoid", "recurrent_activation=tanh"},
  "3:1:4:7", "lstm_multi_step_seq_act.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(LSTM, LayerGoldenTest,
                     ::testing::Values(lstm_single_step, lstm_multi_step,
                                       lstm_single_step_seq,
                                       lstm_multi_step_seq,
                                       lstm_multi_step_seq_act_orig,
                                       lstm_multi_step_seq_act));

#ifdef ENABLE_FP16
auto lstm_single_step_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true"}, "3:1:1:7",
  "lstm_single_step_w16a16.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp16", "fp16");

auto lstm_multi_step_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true"}, "3:1:4:7",
  "lstm_multi_step_w16a16.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp16", "fp16");

auto lstm_single_step_seq_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true", "return_sequences=true"}, "3:1:1:7",
  "lstm_single_step_seq_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto lstm_multi_step_seq_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true", "return_sequences=true"}, "3:1:4:7",
  "lstm_multi_step_seq_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto lstm_multi_step_seq_act_orig_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true", "return_sequences=true",
   "hidden_state_activation=tanh", "recurrent_activation=sigmoid"},
  "3:1:4:7", "lstm_multi_step_seq_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto lstm_multi_step_seq_act_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMLayer>,
  {"unit=5", "integrate_bias=true", "return_sequences=true",
   "hidden_state_activation=sigmoid", "recurrent_activation=tanh"},
  "3:1:4:7", "lstm_multi_step_seq_act_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(LSTM16, LayerGoldenTest,
                     ::testing::Values(lstm_single_step_w16a16,
                                       lstm_multi_step_w16a16,
                                       lstm_single_step_seq_w16a16,
                                       lstm_multi_step_seq_w16a16,
                                       lstm_multi_step_seq_act_orig_w16a16,
                                       lstm_multi_step_seq_act_w16a16));
#endif
