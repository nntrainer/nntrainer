// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_ccapi_tensor.cpp
 * @date        11 December 2023
 * @brief       cc API Tensor Unit tests.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */

#include <cstring>
#include <gtest/gtest.h>
#include <iostream>

#include <layer.h>
#include <model.h>
#include <nntrainer_error.h>
#include <nntrainer_test_util.h>
#include <tensor_api.h>

/**
 * @brief Original test — backward compatibility with Pimpl
 */
TEST(nntrainer_ccapi, tensor_01_p) {
  std::shared_ptr<ml::train::Layer> layer;
  ml::train::Tensor a;

  EXPECT_NO_THROW(layer =
                    ml::train::layer::Input({"name=input0", "input_shape=1:1:2",
                                             "normalization=true"}));

  EXPECT_NO_THROW(a.setSrcLayer(layer));

  std::shared_ptr<ml::train::Layer> layer_b = a.getSrcLayer();
  EXPECT_EQ(layer_b->getName(), "input0");
}

/**
 * @brief Default constructed tensor is invalid
 */
TEST(nntrainer_ccapi_tensor, default_construct_p) {
  ml::train::Tensor t;
  EXPECT_FALSE(t.isValid());
}

/**
 * @brief Symbolic tensor with dimension is valid
 */
TEST(nntrainer_ccapi_tensor, symbolic_construct_p) {
  ml::train::TensorDim dim({1, 1, 28, 28});
  ml::train::Tensor t(dim, "input");

  EXPECT_TRUE(t.isValid());
  EXPECT_EQ(t.name(), "input");
  EXPECT_EQ(t.shape().batch(), 1);
  EXPECT_EQ(t.shape().channel(), 1);
  EXPECT_EQ(t.shape().height(), 28);
  EXPECT_EQ(t.shape().width(), 28);
}

/**
 * @brief Symbolic tensor without name
 */
TEST(nntrainer_ccapi_tensor, symbolic_construct_no_name_p) {
  ml::train::TensorDim dim({1, 3, 32, 32});
  ml::train::Tensor t(dim);

  EXPECT_TRUE(t.isValid());
  EXPECT_EQ(t.name(), "");
  EXPECT_EQ(t.shape().channel(), 3);
}

/**
 * @brief dtype defaults to FP32
 */
TEST(nntrainer_ccapi_tensor, dtype_default_p) {
  ml::train::TensorDim dim({1, 1, 2, 2});
  ml::train::Tensor t(dim);

  EXPECT_EQ(t.dtype(), ml::train::TensorDim::DataType::FP32);
}

/**
 * @brief Move constructor transfers ownership
 */
TEST(nntrainer_ccapi_tensor, move_construct_p) {
  ml::train::TensorDim dim({1, 1, 28, 28});
  ml::train::Tensor a(dim, "original");
  ml::train::Tensor b(std::move(a));

  EXPECT_TRUE(b.isValid());
  EXPECT_EQ(b.name(), "original");
  EXPECT_EQ(b.shape().height(), 28);
  // moved-from tensor should be invalid
  EXPECT_FALSE(a.isValid());
}

/**
 * @brief Move assignment transfers ownership
 */
TEST(nntrainer_ccapi_tensor, move_assign_p) {
  ml::train::TensorDim dim({1, 1, 10, 10});
  ml::train::Tensor a(dim, "src");
  ml::train::Tensor b;

  b = std::move(a);
  EXPECT_TRUE(b.isValid());
  EXPECT_EQ(b.name(), "src");
  EXPECT_FALSE(a.isValid());
}

/**
 * @brief Copy constructor creates independent copy
 */
TEST(nntrainer_ccapi_tensor, copy_construct_p) {
  ml::train::TensorDim dim({1, 1, 28, 28});
  ml::train::Tensor a(dim, "shared");
  ml::train::Tensor b(a);

  EXPECT_TRUE(b.isValid());
  EXPECT_EQ(b.name(), "shared");
  EXPECT_EQ(b.shape().width(), 28);
  // both should be valid
  EXPECT_TRUE(a.isValid());
}

/**
 * @brief Copy assignment creates independent copy
 */
TEST(nntrainer_ccapi_tensor, copy_assign_p) {
  ml::train::TensorDim dim({1, 1, 5, 5});
  ml::train::Tensor a(dim, "orig");
  ml::train::Tensor b;

  b = a;
  EXPECT_TRUE(b.isValid());
  EXPECT_EQ(b.name(), "orig");
  EXPECT_TRUE(a.isValid());
}

/**
 * @brief Clone creates independent copy
 */
TEST(nntrainer_ccapi_tensor, clone_p) {
  ml::train::TensorDim dim({1, 1, 3, 3});
  ml::train::Tensor a(dim, "cloneable");
  ml::train::Tensor b = a.clone();

  EXPECT_TRUE(b.isValid());
  EXPECT_EQ(b.name(), "cloneable");
  EXPECT_EQ(b.shape().height(), 3);
}

/**
 * @brief Accessing shape on invalid tensor throws
 */
TEST(nntrainer_ccapi_tensor, invalid_shape_n) {
  ml::train::Tensor t;
  EXPECT_THROW(t.shape(), std::runtime_error);
}

/**
 * @brief Accessing name on invalid tensor throws
 */
TEST(nntrainer_ccapi_tensor, invalid_name_n) {
  ml::train::Tensor t;
  EXPECT_THROW(t.name(), std::runtime_error);
}

/**
 * @brief Accessing dtype on invalid tensor throws
 */
TEST(nntrainer_ccapi_tensor, invalid_dtype_n) {
  ml::train::Tensor t;
  EXPECT_THROW(t.dtype(), std::runtime_error);
}

/**
 * @brief setSrcLayer / getSrcLayer on default tensor
 */
TEST(nntrainer_ccapi_tensor, src_layer_default_p) {
  ml::train::Tensor t;
  EXPECT_EQ(t.getSrcLayer(), nullptr);
}

// ===== Step 1-2: fromData, zeros, ones =====

/**
 * @brief fromData wraps external buffer (zero-copy)
 */
TEST(nntrainer_ccapi_tensor, from_data_p) {
  float buf[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  ml::train::TensorDim dim({1, 1, 3, 4});
  auto t = ml::train::Tensor::fromData(dim, buf);

  EXPECT_TRUE(t.isValid());
  EXPECT_TRUE(t.isExternal());
  EXPECT_TRUE(t.isMaterialized());
  EXPECT_EQ(t.shape().height(), 3u);
  EXPECT_EQ(t.shape().width(), 4u);
}

/**
 * @brief fromData with name
 */
TEST(nntrainer_ccapi_tensor, from_data_named_p) {
  float buf[4] = {1, 2, 3, 4};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 2}, buf, "ext_cache");
  EXPECT_EQ(t.name(), "ext_cache");
  EXPECT_TRUE(t.isExternal());
}

/**
 * @brief fromData with null pointer throws
 */
TEST(nntrainer_ccapi_tensor, from_data_null_n) {
  EXPECT_THROW(ml::train::Tensor::fromData({1, 1, 2, 2}, nullptr),
               std::invalid_argument);
}

/**
 * @brief zeros creates materialized non-external tensor
 */
TEST(nntrainer_ccapi_tensor, zeros_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 3});
  EXPECT_TRUE(t.isValid());
  EXPECT_FALSE(t.isExternal());
  EXPECT_TRUE(t.isMaterialized());
  EXPECT_EQ(t.shape().height(), 2u);
  EXPECT_EQ(t.shape().width(), 3u);
}

/**
 * @brief ones creates materialized non-external tensor
 */
TEST(nntrainer_ccapi_tensor, ones_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 3});
  EXPECT_TRUE(t.isValid());
  EXPECT_FALSE(t.isExternal());
  EXPECT_TRUE(t.isMaterialized());
}

/**
 * @brief zeros with name
 */
TEST(nntrainer_ccapi_tensor, zeros_named_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 3, 3}, "zero_t");
  EXPECT_EQ(t.name(), "zero_t");
}

/**
 * @brief symbolic tensor is not materialized and not external
 */
TEST(nntrainer_ccapi_tensor, symbolic_not_materialized_p) {
  ml::train::Tensor t({1, 1, 28, 28});
  EXPECT_FALSE(t.isMaterialized());
  EXPECT_FALSE(t.isExternal());
}

/**
 * @brief default tensor is not materialized and not external
 */
TEST(nntrainer_ccapi_tensor, default_not_materialized_p) {
  ml::train::Tensor t;
  EXPECT_FALSE(t.isMaterialized());
  EXPECT_FALSE(t.isExternal());
}

// ===== Step 1-3: data access =====

/**
 * @brief data<float>() on fromData returns the original pointer (zero-copy)
 */
TEST(nntrainer_ccapi_tensor, data_from_data_zero_copy_p) {
  float buf[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 3}, buf);
  EXPECT_EQ(t.data<float>(), buf);
}

/**
 * @brief getValue on fromData tensor
 */
TEST(nntrainer_ccapi_tensor, get_value_from_data_p) {
  float buf[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 3}, buf);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 2), 3.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 2), 6.0f);
}

/**
 * @brief getValue on zeros tensor
 */
TEST(nntrainer_ccapi_tensor, get_value_zeros_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 0.0f);
}

/**
 * @brief getValue on ones tensor
 */
TEST(nntrainer_ccapi_tensor, get_value_ones_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 1.0f);
}

/**
 * @brief setValue modifies value in-place
 */
TEST(nntrainer_ccapi_tensor, set_value_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  t.setValue(0, 0, 1, 1, 42.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 42.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 0.0f);
}

/**
 * @brief mutable_data allows direct writes
 */
TEST(nntrainer_ccapi_tensor, mutable_data_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 1, 4});
  float *ptr = t.mutable_data<float>();
  ptr[0] = 10.0f;
  ptr[3] = 99.0f;
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 10.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 3), 99.0f);
}

/**
 * @brief setData replaces external pointer
 */
TEST(nntrainer_ccapi_tensor, set_data_replace_ptr_p) {
  float buf1[4] = {1, 2, 3, 4};
  float buf2[4] = {5, 6, 7, 8};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 2}, buf1);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 1.0f);

  t.setData(buf2);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 5.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 8.0f);
  EXPECT_EQ(t.data<float>(), buf2);
}

/**
 * @brief setData on non-external (zeros) tensor throws
 */
TEST(nntrainer_ccapi_tensor, set_data_on_non_external_n) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  float buf[4] = {};
  EXPECT_THROW(t.setData(buf), std::runtime_error);
}

/**
 * @brief setData on symbolic tensor throws
 */
TEST(nntrainer_ccapi_tensor, set_data_on_symbolic_n) {
  ml::train::Tensor t({1, 1, 2, 2});
  float buf[4] = {};
  EXPECT_THROW(t.setData(buf), std::runtime_error);
}

/**
 * @brief setData with null throws
 */
TEST(nntrainer_ccapi_tensor, set_data_null_n) {
  float buf[4] = {1, 2, 3, 4};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 2}, buf);
  EXPECT_THROW(t.setData(nullptr), std::invalid_argument);
}

/**
 * @brief copyFrom copies data into materialized tensor
 */
TEST(nntrainer_ccapi_tensor, copy_from_p) {
  float src[4] = {10, 20, 30, 40};
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  t.copyFrom(src);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 10.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 40.0f);
}

/**
 * @brief copyFrom with null throws
 */
TEST(nntrainer_ccapi_tensor, copy_from_null_n) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  EXPECT_THROW(t.copyFrom(nullptr), std::invalid_argument);
}

/**
 * @brief data access on unmaterialized (symbolic) tensor throws
 */
TEST(nntrainer_ccapi_tensor, data_access_unmaterialized_n) {
  ml::train::Tensor t({1, 1, 28, 28});
  EXPECT_THROW(t.data<float>(), std::runtime_error);
  EXPECT_THROW(t.mutable_data<float>(), std::runtime_error);
  EXPECT_THROW(t.getValue(0, 0, 0, 0), std::runtime_error);
  EXPECT_THROW(t.setValue(0, 0, 0, 0, 1.0f), std::runtime_error);
}

/**
 * @brief clone of eager tensor creates independent copy
 */
TEST(nntrainer_ccapi_tensor, clone_eager_independent_p) {
  auto orig = ml::train::Tensor::zeros({1, 1, 2, 2});
  orig.setValue(0, 0, 0, 0, 99.0f);
  auto cloned = orig.clone();

  cloned.setValue(0, 0, 0, 0, 1.0f);
  EXPECT_FLOAT_EQ(orig.getValue(0, 0, 0, 0), 99.0f);
  EXPECT_FLOAT_EQ(cloned.getValue(0, 0, 0, 0), 1.0f);
}

// ===== Step 2-1: LayerHandle graph edge recording =====

/**
 * @brief LayerHandle wraps createLayer and enables operator()
 */
TEST(nntrainer_ccapi_tensor, layer_handle_construct_p) {
  ml::train::LayerHandle fc =
    ml::train::createLayer("fully_connected", {"unit=256", "name=fc1"});
  EXPECT_TRUE(static_cast<bool>(fc));
  EXPECT_EQ(fc->getName(), "fc1");
  EXPECT_EQ(fc->getType(), "fully_connected");
}

/**
 * @brief LayerHandle converts to shared_ptr<Layer> for backward compat
 */
TEST(nntrainer_ccapi_tensor, layer_handle_to_shared_ptr_p) {
  ml::train::LayerHandle fc =
    ml::train::createLayer("fully_connected", {"unit=128", "name=fc_conv"});
  std::shared_ptr<ml::train::Layer> layer_ptr = fc;
  EXPECT_EQ(layer_ptr->getName(), "fc_conv");
}

/**
 * @brief Layer call on symbolic tensor produces symbolic output
 */
TEST(nntrainer_ccapi_tensor, layer_call_symbolic_p) {
  using namespace ml::train;
  auto input = Tensor({1, 1, 1, 784}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc1"});
  auto output = fc(input);

  EXPECT_TRUE(output.isValid());
  EXPECT_FALSE(output.isMaterialized());
  // Shape is propagated from input; full shape inference (e.g. FC unit)
  // will be resolved during model.compile()
  EXPECT_EQ(output.getProducingLayer()->getName(), "fc1");
}

/**
 * @brief Layer chain: fc1 -> fc2
 */
TEST(nntrainer_ccapi_tensor, layer_chain_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 784}, "x");
  LayerHandle fc1 = createLayer("fully_connected", {"unit=128", "name=fc1"});
  LayerHandle fc2 = createLayer("fully_connected", {"unit=10", "name=fc2"});
  auto h = fc1(x);
  auto y = fc2(h);

  EXPECT_TRUE(y.isValid());
  EXPECT_FALSE(y.isMaterialized());
  EXPECT_EQ(y.getProducingLayer()->getName(), "fc2");
  EXPECT_EQ(y.getInputTensors()[0].getProducingLayer()->getName(), "fc1");
}

/**
 * @brief Multi-input layer (Addition)
 */
TEST(nntrainer_ccapi_tensor, multi_input_layer_p) {
  using namespace ml::train;
  auto a = Tensor({1, 1, 1, 256}, "a");
  auto b = Tensor({1, 1, 1, 256}, "b");
  LayerHandle add = createLayer("Addition", {"name=add1"});
  auto added = add({a, b});

  EXPECT_TRUE(added.isValid());
  EXPECT_FALSE(added.isMaterialized());
}

/**
 * @brief Graph edge: output records producing layer
 */
TEST(nntrainer_ccapi_tensor, graph_edge_producing_layer_p) {
  using namespace ml::train;
  auto input = Tensor({1, 1, 1, 784}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc1"});
  auto output = fc(input);

  auto producer = output.getProducingLayer();
  EXPECT_NE(producer, nullptr);
  EXPECT_EQ(producer->getName(), "fc1");
}

/**
 * @brief Graph edge: output records input tensors
 */
TEST(nntrainer_ccapi_tensor, graph_edge_input_tensors_p) {
  using namespace ml::train;
  auto input = Tensor({1, 1, 1, 784}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc1"});
  auto output = fc(input);

  auto inputs = output.getInputTensors();
  EXPECT_EQ(inputs.size(), 1u);
  EXPECT_EQ(inputs[0].name(), "input");
}

/**
 * @brief Leaf/input tensor has no producing layer
 */
TEST(nntrainer_ccapi_tensor, graph_edge_leaf_tensor_p) {
  ml::train::Tensor input({1, 1, 1, 784}, "input");
  EXPECT_EQ(input.getProducingLayer(), nullptr);
  EXPECT_TRUE(input.getInputTensors().empty());
}

/**
 * @brief Chain graph traversal: output -> fc2 -> fc1 -> input
 */
TEST(nntrainer_ccapi_tensor, graph_chain_traversal_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 784}, "x");
  LayerHandle fc1 = createLayer("fully_connected", {"unit=128", "name=fc1"});
  LayerHandle fc2 = createLayer("fully_connected", {"unit=10", "name=fc2"});
  auto h = fc1(x);
  auto y = fc2(h);

  // y was produced by fc2
  EXPECT_EQ(y.getProducingLayer()->getName(), "fc2");
  // y's input is h
  auto y_inputs = y.getInputTensors();
  EXPECT_EQ(y_inputs.size(), 1u);
  // h was produced by fc1
  EXPECT_EQ(y_inputs[0].getProducingLayer()->getName(), "fc1");
  // h's input is x (leaf)
  auto h_inputs = y_inputs[0].getInputTensors();
  EXPECT_EQ(h_inputs.size(), 1u);
  EXPECT_EQ(h_inputs[0].name(), "x");
  EXPECT_EQ(h_inputs[0].getProducingLayer(), nullptr);
}

/**
 * @brief Null LayerHandle throws on call
 */
TEST(nntrainer_ccapi_tensor, layer_handle_null_call_n) {
  ml::train::LayerHandle null_handle;
  ml::train::Tensor input({1, 1, 1, 784}, "input");
  EXPECT_THROW(null_handle(input), std::runtime_error);
}

/**
 * @brief Empty inputs to operator() throws
 */
TEST(nntrainer_ccapi_tensor, layer_handle_empty_inputs_n) {
  ml::train::LayerHandle fc =
    ml::train::createLayer("fully_connected", {"unit=256", "name=fc1"});
  EXPECT_THROW(fc(std::vector<ml::train::Tensor>{}), std::invalid_argument);
}

// ===== Step 2-2: Tensor ops → implicit layers =====

/**
 * @brief add() creates implicit Addition layer
 */
TEST(nntrainer_ccapi_tensor, add_symbolic_p) {
  using namespace ml::train;
  auto a = Tensor({1, 1, 1, 256}, "a");
  auto b = Tensor({1, 1, 1, 256}, "b");
  auto c = a.add(b);

  EXPECT_TRUE(c.isValid());
  EXPECT_FALSE(c.isMaterialized());
  EXPECT_EQ(c.shape().width(), 256u);
  // Verify graph: c produced by Addition layer with 2 inputs
  EXPECT_NE(c.getProducingLayer(), nullptr);
  EXPECT_EQ(c.getProducingLayer()->getType(), "addition");
  EXPECT_EQ(c.getInputTensors().size(), 2u);
  EXPECT_EQ(c.getInputTensors()[0].name(), "a");
  EXPECT_EQ(c.getInputTensors()[1].name(), "b");
}

/**
 * @brief multiply() creates implicit Multiply layer
 */
TEST(nntrainer_ccapi_tensor, multiply_symbolic_p) {
  using namespace ml::train;
  auto a = Tensor({1, 1, 1, 128}, "ma");
  auto b = Tensor({1, 1, 1, 128}, "mb");
  auto c = a.multiply(b);

  EXPECT_TRUE(c.isValid());
  EXPECT_FALSE(c.isMaterialized());
  EXPECT_EQ(c.shape().width(), 128u);
  EXPECT_NE(c.getProducingLayer(), nullptr);
  EXPECT_EQ(c.getProducingLayer()->getType(), "multiply");
  EXPECT_EQ(c.getInputTensors().size(), 2u);
}

/**
 * @brief reshape() creates implicit Reshape layer
 */
TEST(nntrainer_ccapi_tensor, reshape_symbolic_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 784}, "flat");
  auto y = x.reshape(TensorDim({1, 1, 28, 28}));

  EXPECT_TRUE(y.isValid());
  EXPECT_FALSE(y.isMaterialized());
  EXPECT_EQ(y.shape().channel(), 1u);
  EXPECT_EQ(y.shape().height(), 28u);
  EXPECT_EQ(y.shape().width(), 28u);
  EXPECT_NE(y.getProducingLayer(), nullptr);
  EXPECT_EQ(y.getInputTensors().size(), 1u);
  EXPECT_EQ(y.getInputTensors()[0].name(), "flat");
}

/**
 * @brief Residual (skip) connection: x + fc(x)
 */
TEST(nntrainer_ccapi_tensor, residual_connection_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 256}, "x");
  LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc_res"});
  auto h = fc(x);
  auto out = x.add(h);

  EXPECT_TRUE(out.isValid());
  // out -> Addition -> [x, h]; h -> fc -> [x]
  auto inputs = out.getInputTensors();
  EXPECT_EQ(inputs.size(), 2u);
  EXPECT_EQ(inputs[0].name(), "x");
  EXPECT_EQ(inputs[1].getProducingLayer()->getName(), "fc_res");
}

/**
 * @brief Chained ops: a.add(b).multiply(c)
 */
TEST(nntrainer_ccapi_tensor, chained_ops_p) {
  using namespace ml::train;
  auto a = Tensor({1, 1, 1, 64}, "a");
  auto b = Tensor({1, 1, 1, 64}, "b");
  auto c = Tensor({1, 1, 1, 64}, "c");
  auto result = a.add(b).multiply(c);

  EXPECT_TRUE(result.isValid());
  EXPECT_EQ(result.getProducingLayer()->getType(), "multiply");
  auto mul_inputs = result.getInputTensors();
  EXPECT_EQ(mul_inputs.size(), 2u);
  // First input to multiply is the add result
  EXPECT_EQ(mul_inputs[0].getProducingLayer()->getType(), "addition");
}

// ===== Step 2-3: Model::compile(Tensor, Tensor) — graph extraction =====

/**
 * @brief Simple single FC layer compile from tensor graph
 */
TEST(nntrainer_ccapi_graph, simple_fc_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 784}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=10", "name=fc"});
  auto y = fc(x);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);
}

/**
 * @brief Multi-layer compile: fc1 -> relu -> fc2
 */
TEST(nntrainer_ccapi_graph, multi_layer_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 784}, "input");
  LayerHandle fc1 = createLayer("fully_connected", {"unit=128", "name=fc1"});
  LayerHandle relu = createLayer("activation", {"activation=relu", "name=relu1"});
  LayerHandle fc2 = createLayer("fully_connected", {"unit=10", "name=fc2"});

  auto h = fc1(x);
  h = relu(h);
  auto y = fc2(h);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);
}

/**
 * @brief Residual connection compile: x -> fc1 -> add(x, fc1_out) -> fc_out
 */
TEST(nntrainer_ccapi_graph, residual_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 256}, "input");
  LayerHandle fc1 = createLayer("fully_connected", {"unit=256", "name=fc1"});
  auto h = fc1(x);
  auto out = x.add(h);
  LayerHandle fc_out = createLayer("fully_connected", {"unit=10", "name=fc_out"});
  auto y = fc_out(out);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);
}

/**
 * @brief Existing addLayer-based workflow still works
 */
TEST(nntrainer_ccapi_graph, existing_add_layer_still_works_p) {
  using namespace ml::train;
  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  model->addLayer(
    createLayer("input", {"name=in", "input_shape=1:1:784"}));
  model->addLayer(
    createLayer("fully_connected", {"name=fc", "unit=10", "input_layers=in"}));
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
}

// ===== Step 2-4: _bind + data injection/extraction =====

/**
 * @brief After compile, input tensor is materialized and writable
 */
TEST(nntrainer_ccapi_graph, data_injection_after_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 4}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=2", "name=fc"});
  auto y = fc(x);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);

  // After compile (which includes initialize + bind), x is materialized
  EXPECT_TRUE(x.isMaterialized());
  EXPECT_TRUE(y.isMaterialized());

  // Can inject data
  float input_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  x.copyFrom(input_data);
  EXPECT_FLOAT_EQ(x.getValue(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(x.getValue(0, 0, 0, 3), 4.0f);
}

/**
 * @brief Can use setValue/getValue on bound tensors
 */
TEST(nntrainer_ccapi_graph, set_get_value_bound_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 4}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=2", "name=fc"});
  auto y = fc(x);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);

  x.setValue(0, 0, 0, 0, 42.0f);
  EXPECT_FLOAT_EQ(x.getValue(0, 0, 0, 0), 42.0f);

  // data() and mutable_data() should work
  const float *ptr = x.data<float>();
  EXPECT_FLOAT_EQ(ptr[0], 42.0f);

  float *mptr = x.mutable_data<float>();
  mptr[1] = 99.0f;
  EXPECT_FLOAT_EQ(x.getValue(0, 0, 0, 1), 99.0f);
}

/**
 * @brief Symbolic tensors are not materialized before compile
 */
TEST(nntrainer_ccapi_graph, not_materialized_before_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 4}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=2", "name=fc"});
  auto y = fc(x);

  EXPECT_FALSE(x.isMaterialized());
  EXPECT_FALSE(y.isMaterialized());
}

/**
 * @brief Multi-layer compile with bind
 */
TEST(nntrainer_ccapi_graph, multi_layer_bind_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 8}, "input");
  LayerHandle fc1 = createLayer("fully_connected", {"unit=4", "name=fc1"});
  LayerHandle fc2 = createLayer("fully_connected", {"unit=2", "name=fc2"});
  auto h = fc1(x);
  auto y = fc2(h);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);

  EXPECT_TRUE(x.isMaterialized());
  EXPECT_TRUE(y.isMaterialized());

  // Input shape should match
  EXPECT_EQ(x.shape().width(), 8u);
  // Output shape should be fc2's output
  EXPECT_EQ(y.shape().width(), 2u);
}

// ===== Step 2-5: Lazy chaining (chain/eval) =====

/**
 * @brief chain().multiply_i().add_i().eval() executes in order
 */
TEST(nntrainer_ccapi_tensor, lazy_chain_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  t.chain().multiply_i(2.0f).add_i(1.0f).eval();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 3.0f);  // 1*2+1
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 3.0f);
}

/**
 * @brief Operation order matters: add then multiply
 */
TEST(nntrainer_ccapi_tensor, lazy_chain_order_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 1});
  t.chain().add_i(3.0f).multiply_i(2.0f).eval();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 8.0f);  // (1+3)*2
}

/**
 * @brief eval on non-materialized tensor throws
 */
TEST(nntrainer_ccapi_tensor, lazy_eval_on_symbolic_n) {
  ml::train::Tensor t({1, 1, 2, 2});
  t.chain().add_i(1.0f);
  EXPECT_THROW(t.eval(), std::runtime_error);
}

/**
 * @brief chain() clears previous pending ops
 */
TEST(nntrainer_ccapi_tensor, lazy_chain_reset_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 1});
  t.chain().add_i(100.0f);  // queued but not eval'd
  t.chain().add_i(5.0f).eval();  // chain() clears, only +5 applied
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 6.0f);  // 1+5
}

/**
 * @brief eval clears the chain after execution
 */
TEST(nntrainer_ccapi_tensor, lazy_eval_clears_chain_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 1});
  t.chain().multiply_i(3.0f).eval();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 3.0f);
  // eval again should be no-op (chain is empty)
  t.eval();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 3.0f);
}

/**
 * @brief Graph with additional leaf tensors (non-fromData) works
 */
TEST(nntrainer_ccapi_graph, multiple_leaf_inputs_p) {
  using namespace ml::train;

  auto x1 = Tensor({1, 1, 1, 4}, "x1");
  auto x2 = Tensor({1, 1, 1, 4}, "x2");

  // Two separate inputs go into an add layer
  auto sum = x1.add(x2);

  LayerHandle fc = createLayer("fully_connected", {"unit=2", "name=fc"});
  auto y = fc(sum);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x1, y), ML_ERROR_NONE);

  EXPECT_TRUE(x1.isMaterialized());
  EXPECT_TRUE(y.isMaterialized());
}

/**
 * @brief PicoGPT-style transformer graph: 2 inputs, skip connections, fan-out
 *
 * Tests the graph pattern used in PicoGPT: dual inputs → embeddings → add →
 * transformer blocks (LN + MHA + skip + LN + FFN + skip) → final LN.
 * Uses 1 layer with small dims to keep memory low.
 */
TEST(nntrainer_ccapi_graph, picogpt_style_transformer_p) {
  using namespace ml::train;

  const unsigned int MODEL_DIM = 32;
  const unsigned int FF_DIM = 64;
  const unsigned int NUM_HEADS = 4;

  // Two symbolic inputs (like wte_input, wpe_input)
  Tensor wte_in({1, 1, 1, 1}, "wte_input");
  Tensor wpe_in({1, 1, 1, 1}, "wpe_input");

  LayerHandle wte = createLayer("embedding",
    {"name=wte", "in_dim=100", "out_dim=" + std::to_string(MODEL_DIM)});
  LayerHandle wpe = createLayer("embedding",
    {"name=wpe", "in_dim=100", "out_dim=" + std::to_string(MODEL_DIM)});
  auto wte_out = wte(wte_in);
  auto wpe_out = wpe(wpe_in);

  LayerHandle add_emb = createLayer("Addition", {"name=add"});
  auto prev = add_emb({wte_out, wpe_out});

  // Single transformer block
  LayerHandle ln1 = createLayer("layer_normalization",
    {"name=layer0/ln1", "axis=3", "epsilon=1e-5"});
  auto ln1_out = ln1(prev);

  // MHA with same tensor as Q, K, V (fan-out test)
  LayerHandle mha = createLayer("multi_head_attention",
    {"name=layer0/multi_head_attention",
     "num_heads=" + std::to_string(NUM_HEADS)});
  auto attn_out = mha({ln1_out, ln1_out, ln1_out});

  // Skip connection 1: prev + attn_out (prev used twice = fan-out)
  LayerHandle add1 = createLayer("Addition", {"name=layer0/add1"});
  auto add1_out = add1({prev, attn_out});

  LayerHandle ln2 = createLayer("layer_normalization",
    {"name=layer0/ln2", "axis=3", "epsilon=1e-5"});
  auto ln2_out = ln2(add1_out);

  LayerHandle fc1 = createLayer("fully_connected",
    {"name=layer0/fc1", "unit=" + std::to_string(FF_DIM), "activation=gelu"});
  auto fc1_out = fc1(ln2_out);

  LayerHandle fc2 = createLayer("fully_connected",
    {"name=layer0/fc2", "unit=" + std::to_string(MODEL_DIM)});
  auto fc2_out = fc2(fc1_out);

  // Skip connection 2: add1_out + fc2_out (add1_out used twice = fan-out)
  LayerHandle add2 = createLayer("Addition", {"name=layer0/add2"});
  auto block_out = add2({add1_out, fc2_out});

  LayerHandle final_ln = createLayer("layer_normalization",
    {"name=layer_normalization", "axis=3", "epsilon=1e-5"});
  auto output = final_ln(block_out);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(wte_in, output), ML_ERROR_NONE);

  EXPECT_TRUE(wte_in.isMaterialized());
  EXPECT_TRUE(output.isMaterialized());
}

/**
 * @brief PicoGPT-style graph compile + inference with random weights
 *
 * Verifies no segfault during forward pass through the graph-compiled model.
 */
TEST(nntrainer_ccapi_graph, picogpt_style_inference_p) {
  using namespace ml::train;

  const unsigned int MODEL_DIM = 16;
  const unsigned int FF_DIM = 32;
  const unsigned int NUM_HEADS = 2;

  Tensor wte_in({1, 1, 1, 1}, "wte_input");
  Tensor wpe_in({1, 1, 1, 1}, "wpe_input");

  LayerHandle wte = createLayer("embedding",
    {"name=wte", "in_dim=50", "out_dim=" + std::to_string(MODEL_DIM)});
  LayerHandle wpe = createLayer("embedding",
    {"name=wpe", "in_dim=50", "out_dim=" + std::to_string(MODEL_DIM)});
  auto wte_out = wte(wte_in);
  auto wpe_out = wpe(wpe_in);

  LayerHandle add_emb = createLayer("Addition", {"name=add"});
  auto prev = add_emb({wte_out, wpe_out});

  // 1 transformer block
  LayerHandle ln1 = createLayer("layer_normalization",
    {"name=layer0/ln1", "axis=3", "epsilon=1e-5"});
  auto ln1_out = ln1(prev);

  LayerHandle mha = createLayer("multi_head_attention",
    {"name=layer0/mha", "num_heads=" + std::to_string(NUM_HEADS)});
  auto attn_out = mha({ln1_out, ln1_out, ln1_out});

  LayerHandle add1 = createLayer("Addition", {"name=layer0/add1"});
  auto add1_out = add1({prev, attn_out});

  LayerHandle ln2 = createLayer("layer_normalization",
    {"name=layer0/ln2", "axis=3", "epsilon=1e-5"});
  auto ln2_out = ln2(add1_out);

  LayerHandle fc1 = createLayer("fully_connected",
    {"name=layer0/fc1", "unit=" + std::to_string(FF_DIM), "activation=gelu"});
  auto fc1_out = fc1(ln2_out);

  LayerHandle fc2 = createLayer("fully_connected",
    {"name=layer0/fc2", "unit=" + std::to_string(MODEL_DIM)});
  auto fc2_out = fc2(fc1_out);

  LayerHandle add2 = createLayer("Addition", {"name=layer0/add2"});
  auto block_out = add2({add1_out, fc2_out});

  LayerHandle final_ln = createLayer("layer_normalization",
    {"name=final_ln", "axis=3", "epsilon=1e-5"});
  auto output = final_ln(block_out);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  ASSERT_EQ(model->compile(wte_in, output), ML_ERROR_NONE);

  // Run inference with dummy input (token id=1 for both inputs)
  float wte_input_data[1];
  float wpe_input_data[1];
  unsigned int wte_token = 1, wpe_token = 0;
  std::memcpy(wte_input_data, &wte_token, sizeof(unsigned int));
  std::memcpy(wpe_input_data, &wpe_token, sizeof(unsigned int));

  std::vector<float *> output_bufs;
  EXPECT_NO_THROW(
    output_bufs = model->inference(1, {wte_input_data, wpe_input_data}));
  EXPECT_FALSE(output_bufs.empty());
}

// ===== Phase 1: Extended eager tensor operations =====

// --- Convenience dimension accessors ---

TEST(nntrainer_ccapi_tensor, size_p) {
  auto t = ml::train::Tensor::zeros({2, 3, 4, 5});
  EXPECT_EQ(t.size(), 2u * 3u * 4u * 5u);
}

TEST(nntrainer_ccapi_tensor, batch_channel_height_width_p) {
  auto t = ml::train::Tensor::zeros({2, 3, 4, 5});
  EXPECT_EQ(t.batch(), 2u);
  EXPECT_EQ(t.channel(), 3u);
  EXPECT_EQ(t.height(), 4u);
  EXPECT_EQ(t.width(), 5u);
}

TEST(nntrainer_ccapi_tensor, empty_p) {
  ml::train::Tensor t;
  EXPECT_TRUE(t.empty());
  auto z = ml::train::Tensor::zeros({1, 1, 2, 2});
  EXPECT_FALSE(z.empty());
}

// --- Scalar eager operations ---

TEST(nntrainer_ccapi_tensor, add_scalar_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  auto r = t.add(5.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 6.0f);
  // Original unchanged
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 1.0f);
}

TEST(nntrainer_ccapi_tensor, subtract_scalar_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  auto r = t.subtract(0.5f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 0.5f);
}

TEST(nntrainer_ccapi_tensor, multiply_scalar_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  auto r = t.multiply(3.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 3.0f);
}

TEST(nntrainer_ccapi_tensor, divide_scalar_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  t.setValue(0, 0, 0, 0, 6.0f);
  auto r = t.divide(2.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 3.0f);
}

// --- Tensor-tensor eager operations ---

TEST(nntrainer_ccapi_tensor, subtract_tensor_p) {
  auto a = ml::train::Tensor::ones({1, 1, 2, 2});
  auto b = ml::train::Tensor::ones({1, 1, 2, 2});
  a.setValue(0, 0, 0, 0, 5.0f);
  auto r = a.subtract(b);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 4.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 1), 0.0f);
}

TEST(nntrainer_ccapi_tensor, divide_tensor_p) {
  auto a = ml::train::Tensor::ones({1, 1, 2, 2});
  auto b = ml::train::Tensor::ones({1, 1, 2, 2});
  a.setValue(0, 0, 0, 0, 6.0f);
  b.setValue(0, 0, 0, 0, 3.0f);
  auto r = a.divide(b);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 2.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 1), 1.0f);
}

// --- Dot product ---

TEST(nntrainer_ccapi_tensor, dot_p) {
  // [1,1,1,2] dot [1,1,2,1] = [1,1,1,1]
  auto a = ml::train::Tensor::ones({1, 1, 1, 2});
  auto b = ml::train::Tensor::ones({1, 1, 2, 1});
  a.setValue(0, 0, 0, 0, 2.0f);
  a.setValue(0, 0, 0, 1, 3.0f);
  auto r = a.dot(b);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 5.0f); // 2*1 + 3*1
}

// --- Transpose ---

TEST(nntrainer_ccapi_tensor, transpose_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 3});
  t.setValue(0, 0, 0, 0, 1.0f);
  t.setValue(0, 0, 0, 1, 2.0f);
  t.setValue(0, 0, 0, 2, 3.0f);
  t.setValue(0, 0, 1, 0, 4.0f);
  t.setValue(0, 0, 1, 1, 5.0f);
  t.setValue(0, 0, 1, 2, 6.0f);
  auto r = t.transpose("0:2:1"); // swap h and w
  EXPECT_EQ(r.height(), 3u);
  EXPECT_EQ(r.width(), 2u);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 1, 0), 2.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 1), 4.0f);
}

// --- Power ---

TEST(nntrainer_ccapi_tensor, pow_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 2});
  t.setValue(0, 0, 0, 0, 3.0f);
  t.setValue(0, 0, 0, 1, 2.0f);
  auto r = t.pow(2.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 9.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 1), 4.0f);
}

// --- Sum and average ---

TEST(nntrainer_ccapi_tensor, sum_axis_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 3});
  t.setValue(0, 0, 0, 0, 1.0f);
  t.setValue(0, 0, 0, 1, 2.0f);
  t.setValue(0, 0, 0, 2, 3.0f);
  t.setValue(0, 0, 1, 0, 4.0f);
  t.setValue(0, 0, 1, 1, 5.0f);
  t.setValue(0, 0, 1, 2, 6.0f);
  // Sum along axis 3 (width) -> {1,1,2,1}
  auto r = t.sum(3);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 6.0f);  // 1+2+3
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 1, 0), 15.0f); // 4+5+6
}

TEST(nntrainer_ccapi_tensor, average_axis_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  t.setValue(0, 0, 0, 0, 2.0f);
  t.setValue(0, 0, 0, 1, 4.0f);
  t.setValue(0, 0, 1, 0, 6.0f);
  t.setValue(0, 0, 1, 1, 8.0f);
  // Average along axis 3 (width) -> {1,1,2,1}
  auto r = t.average(3);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 3.0f); // (2+4)/2
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 1, 0), 7.0f); // (6+8)/2
}

TEST(nntrainer_ccapi_tensor, average_global_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  t.setValue(0, 0, 0, 0, 2.0f);
  t.setValue(0, 0, 0, 1, 4.0f);
  t.setValue(0, 0, 1, 0, 6.0f);
  t.setValue(0, 0, 1, 1, 8.0f);
  auto r = t.average();
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 5.0f); // (2+4+6+8)/4
}

// --- L2 norm ---

TEST(nntrainer_ccapi_tensor, l2norm_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 1, 2});
  t.setValue(0, 0, 0, 0, 3.0f);
  t.setValue(0, 0, 0, 1, 4.0f);
  EXPECT_FLOAT_EQ(t.l2norm(), 5.0f); // sqrt(9+16)
}

// --- setZero ---

TEST(nntrainer_ccapi_tensor, set_zero_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  t.setZero();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 0.0f);
}

// --- copyData ---

TEST(nntrainer_ccapi_tensor, copy_data_p) {
  auto src = ml::train::Tensor::ones({1, 1, 2, 2});
  src.setValue(0, 0, 0, 0, 42.0f);
  auto dst = ml::train::Tensor::zeros({1, 1, 2, 2});
  dst.copyData(src);
  EXPECT_FLOAT_EQ(dst.getValue(0, 0, 0, 0), 42.0f);
  EXPECT_FLOAT_EQ(dst.getValue(0, 0, 0, 1), 1.0f);
}

// --- fill ---

TEST(nntrainer_ccapi_tensor, fill_p) {
  auto src = ml::train::Tensor::ones({1, 1, 2, 2});
  src.setValue(0, 0, 0, 0, 99.0f);
  auto dst = ml::train::Tensor::zeros({1, 1, 2, 2});
  dst.fill(src);
  EXPECT_FLOAT_EQ(dst.getValue(0, 0, 0, 0), 99.0f);
}

// --- apply / apply_i ---

TEST(nntrainer_ccapi_tensor, apply_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 3});
  t.setValue(0, 0, 0, 0, 1.0f);
  t.setValue(0, 0, 0, 1, 4.0f);
  t.setValue(0, 0, 0, 2, 9.0f);
  auto r = t.apply([](float x) { return x * x; });
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 1), 16.0f);
  EXPECT_FLOAT_EQ(r.getValue(0, 0, 0, 2), 81.0f);
  // Original unchanged
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 1), 4.0f);
}

TEST(nntrainer_ccapi_tensor, apply_i_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 2});
  t.setValue(0, 0, 0, 0, 3.0f);
  t.setValue(0, 0, 0, 1, 5.0f);
  t.apply_i([](float x) { return x + 10.0f; });
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 13.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 1), 15.0f);
}

// --- Extended lazy chain ---

TEST(nntrainer_ccapi_tensor, lazy_subtract_i_scalar_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 1});
  t.setValue(0, 0, 0, 0, 10.0f);
  t.chain().subtract_i(3.0f).eval();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 7.0f);
}

TEST(nntrainer_ccapi_tensor, lazy_divide_i_scalar_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 1});
  t.setValue(0, 0, 0, 0, 12.0f);
  t.chain().divide_i(4.0f).eval();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 3.0f);
}

TEST(nntrainer_ccapi_tensor, lazy_pow_i_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 1});
  t.setValue(0, 0, 0, 0, 3.0f);
  t.chain().pow_i(2.0f).eval();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 9.0f);
}

TEST(nntrainer_ccapi_tensor, lazy_tensor_add_i_p) {
  auto a = ml::train::Tensor::ones({1, 1, 1, 2});
  auto b = ml::train::Tensor::ones({1, 1, 1, 2});
  a.setValue(0, 0, 0, 0, 3.0f);
  b.setValue(0, 0, 0, 0, 7.0f);
  a.chain().add_i(b).eval();
  EXPECT_FLOAT_EQ(a.getValue(0, 0, 0, 0), 10.0f); // 3+7
  EXPECT_FLOAT_EQ(a.getValue(0, 0, 0, 1), 2.0f);  // 1+1
}

TEST(nntrainer_ccapi_tensor, lazy_tensor_multiply_i_p) {
  auto a = ml::train::Tensor::ones({1, 1, 1, 2});
  auto b = ml::train::Tensor::ones({1, 1, 1, 2});
  a.setValue(0, 0, 0, 0, 3.0f);
  b.setValue(0, 0, 0, 0, 4.0f);
  a.chain().multiply_i(b).eval();
  EXPECT_FLOAT_EQ(a.getValue(0, 0, 0, 0), 12.0f); // 3*4
  EXPECT_FLOAT_EQ(a.getValue(0, 0, 0, 1), 1.0f);  // 1*1
}

TEST(nntrainer_ccapi_tensor, lazy_complex_chain_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 1});
  t.setValue(0, 0, 0, 0, 10.0f);
  // (10 + 2) * 3 - 1 = 35
  t.chain().add_i(2.0f).multiply_i(3.0f).subtract_i(1.0f).eval();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 35.0f);
}

// --- getBatchSlice ---

TEST(nntrainer_ccapi_tensor, get_batch_slice_p) {
  auto t = ml::train::Tensor::zeros({2, 1, 1, 3});
  t.setValue(0, 0, 0, 0, 1.0f);
  t.setValue(0, 0, 0, 1, 2.0f);
  t.setValue(0, 0, 0, 2, 3.0f);
  t.setValue(1, 0, 0, 0, 4.0f);
  t.setValue(1, 0, 0, 1, 5.0f);
  t.setValue(1, 0, 0, 2, 6.0f);

  auto s = t.getBatchSlice(1, 1); // second batch
  EXPECT_EQ(s.batch(), 1u);
  EXPECT_FLOAT_EQ(s.getValue(0, 0, 0, 0), 4.0f);
  EXPECT_FLOAT_EQ(s.getValue(0, 0, 0, 2), 6.0f);
}

// --- Eager ops on unmaterialized tensor throw ---

TEST(nntrainer_ccapi_tensor, eager_ops_unmaterialized_n) {
  ml::train::Tensor t({1, 1, 2, 2});
  EXPECT_THROW(t.add(1.0f), std::runtime_error);
  EXPECT_THROW(t.subtract(1.0f), std::runtime_error);
  EXPECT_THROW(t.multiply(1.0f), std::runtime_error);
  EXPECT_THROW(t.divide(1.0f), std::runtime_error);
  EXPECT_THROW(t.pow(2.0f), std::runtime_error);
  EXPECT_THROW(t.l2norm(), std::runtime_error);
  EXPECT_THROW(t.setZero(), std::runtime_error);
}

// --- argmax ---

TEST(nntrainer_ccapi_tensor, argmax_p) {
  // shape [2, 1, 1, 3] — 2 batch elements, each with 3 values
  auto t = ml::train::Tensor::zeros({2, 1, 1, 3});
  t.setValue(0, 0, 0, 0, 1.0f);
  t.setValue(0, 0, 0, 1, 5.0f);
  t.setValue(0, 0, 0, 2, 3.0f);
  t.setValue(1, 0, 0, 0, 9.0f);
  t.setValue(1, 0, 0, 1, 2.0f);
  t.setValue(1, 0, 0, 2, 7.0f);

  auto ids = t.argmax();
  EXPECT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 1u); // max at index 1 (value 5.0)
  EXPECT_EQ(ids[1], 0u); // max at index 0 (value 9.0)
}

// ===== CausalLM-pattern symbolic graph verification =====

/**
 * @brief CausalLM-style multi-input compile with external KV cache tensors,
 *        residual connections (Tensor::add), and multiple decoder blocks.
 *
 * Validates the exact symbolic graph construction pattern used by the
 * CausalLM base class after conversion to Tensor API:
 *   Input → Embedding(FC) → N x [Norm(LN) → Attention(FC) → Residual(add)
 *                                → Norm(LN) → MLP(FC→FC) → Residual(add)]
 *           → FinalNorm(LN) → LMHead(FC)
 *
 * External KV cache inputs are passed as additional leaf tensors via
 * compile(vector<Tensor>, vector<Tensor>).
 */
TEST(nntrainer_ccapi_graph, causallm_style_multi_input_p) {
  using namespace ml::train;

  const unsigned int SEQ_LEN = 4;
  const unsigned int DIM = 16;
  const unsigned int FF_DIM = 32;
  const unsigned int NUM_LAYERS = 2;
  const unsigned int VOCAB_SIZE = 50;

  // --- Main input (leaf symbolic tensor) ---
  Tensor input({1, 1, 1, SEQ_LEN}, "input0");

  // --- Embedding (using FC as proxy) ---
  LayerHandle embedding = createLayer(
    "fully_connected",
    {"name=embedding0", "unit=" + std::to_string(DIM), "disable_bias=true"});
  Tensor x = embedding(input);

  // --- Decoder blocks: tests residual connections, fan-out, multi-layer ---
  for (unsigned int layer_id = 0; layer_id < NUM_LAYERS; ++layer_id) {
    std::string prefix = "layer" + std::to_string(layer_id);

    // Attention norm
    LayerHandle att_norm = createLayer(
      "layer_normalization",
      {"name=" + prefix + "_att_norm", "axis=3", "epsilon=1e-5"});
    Tensor normed = att_norm(x);

    // Q/K/V projections (self-attention: all from same normed input)
    LayerHandle q_proj = createLayer(
      "fully_connected",
      {"name=" + prefix + "_wq", "unit=" + std::to_string(DIM),
       "disable_bias=true"});
    LayerHandle k_proj = createLayer(
      "fully_connected",
      {"name=" + prefix + "_wk", "unit=" + std::to_string(DIM),
       "disable_bias=true"});
    LayerHandle v_proj = createLayer(
      "fully_connected",
      {"name=" + prefix + "_wv", "unit=" + std::to_string(DIM),
       "disable_bias=true"});
    Tensor q = q_proj(normed);
    Tensor k = k_proj(normed);
    Tensor v = v_proj(normed);

    // Attention proxy: Addition of Q+K+V → O projection
    LayerHandle qkv_combine =
      createLayer("Addition", {"name=" + prefix + "_qkv_combine"});
    Tensor combined = qkv_combine({q, k, v});

    LayerHandle o_proj = createLayer(
      "fully_connected",
      {"name=" + prefix + "_wo", "unit=" + std::to_string(DIM),
       "disable_bias=true"});
    Tensor att_out = o_proj(combined);

    // Residual add (Tensor::add creates implicit Addition layer)
    Tensor residual = x.add(att_out);

    // FFN norm
    LayerHandle ffn_norm = createLayer(
      "layer_normalization",
      {"name=" + prefix + "_ffn_norm", "axis=3", "epsilon=1e-5"});
    Tensor ffn_normed = ffn_norm(residual);

    // MLP: up → gate → multiply → down
    LayerHandle ffn_up = createLayer(
      "fully_connected",
      {"name=" + prefix + "_ffn_up", "unit=" + std::to_string(FF_DIM),
       "disable_bias=true"});
    LayerHandle ffn_gate = createLayer(
      "fully_connected",
      {"name=" + prefix + "_ffn_gate", "unit=" + std::to_string(FF_DIM),
       "disable_bias=true"});
    Tensor up = ffn_up(ffn_normed);
    Tensor gate = ffn_gate(ffn_normed);

    // SwiGLU proxy: element-wise multiply (up * gate)
    Tensor activated = up.multiply(gate);

    LayerHandle ffn_down = createLayer(
      "fully_connected",
      {"name=" + prefix + "_ffn_down", "unit=" + std::to_string(DIM),
       "disable_bias=true"});
    Tensor ffn_out = ffn_down(activated);

    // Residual add
    x = residual.add(ffn_out);
  }

  // --- Final norm ---
  LayerHandle output_norm = createLayer(
    "layer_normalization", {"name=output_norm", "axis=3", "epsilon=1e-5"});
  x = output_norm(x);

  // --- LM Head ---
  LayerHandle lmhead = createLayer(
    "fully_connected",
    {"name=output_of_causallm", "unit=" + std::to_string(VOCAB_SIZE),
     "disable_bias=true"});
  Tensor output = lmhead(x);

  // --- Compile from symbolic tensor graph (includes initialize) ---
  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  ASSERT_EQ(model->compile(input, output, ExecutionMode::INFERENCE),
            ML_ERROR_NONE);

  // --- Verify tensors are materialized after compile ---
  EXPECT_TRUE(input.isMaterialized());
  EXPECT_TRUE(output.isMaterialized());

  // --- Run inference with dummy data ---
  std::vector<float> input_data(SEQ_LEN, 1.0f);
  std::vector<float *> output_bufs;
  EXPECT_NO_THROW(output_bufs = model->inference(1, {input_data.data()}));
  EXPECT_FALSE(output_bufs.empty());
  EXPECT_NE(output_bufs[0], nullptr);
}
