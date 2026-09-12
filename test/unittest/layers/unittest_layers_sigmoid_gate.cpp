// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file unittest_layers_sigmoid_gate.cpp
 * @date 12 September 2026
 * @brief Sigmoid-gated GLU / add layer semantics
 * @see https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 *
 * @details Both layers are reached the way every other layer is -- by type
 * string through the registered context's factory -- so the semantics suite is
 * what proves the registration landed: it creates each type from the
 * application context, sets its properties, and finalizes it with two inputs.
 * The maths itself is covered backend by backend in
 * unittest_sigmoid_gate_ops.cpp.
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <sigmoid_add_layer.h>
#include <sigmoid_glu_layer.h>

auto semantic_sigmoid_glu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SigmoidGluLayer>,
  nntrainer::SigmoidGluLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

auto semantic_sigmoid_add = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SigmoidAddLayer>,
  nntrainer::SigmoidAddLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

// skip_prefill is the one property either layer takes beyond the base ones; a
// graph that declares it must not be rejected by the layer that has to honour
// it.
auto semantic_sigmoid_glu_skip_prefill = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SigmoidGluLayer>,
  nntrainer::SigmoidGluLayer::type, {"skip_prefill=true"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

auto semantic_sigmoid_add_skip_prefill = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SigmoidAddLayer>,
  nntrainer::SigmoidAddLayer::type, {"skip_prefill=true"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(SigmoidGate, LayerSemantics,
                     ::testing::Values(semantic_sigmoid_glu,
                                       semantic_sigmoid_add,
                                       semantic_sigmoid_glu_skip_prefill,
                                       semantic_sigmoid_add_skip_prefill));
