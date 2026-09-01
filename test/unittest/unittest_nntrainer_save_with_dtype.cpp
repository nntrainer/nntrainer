// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   unittest_nntrainer_save_with_dtype.cpp
 * @date   04 March 2026
 * @brief  Unit tests for NONE DataType and save-with-dtype feature
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <input_layer.h>
#include <layer.h>
#include <model.h>
#include <neuralnet.h>
#include <optimizer.h>
#include <tensor.h>
#include <tensor_dim.h>

#include <nntrainer_test_util.h>

using TensorDim = ml::train::TensorDim;
using DataType = TensorDim::DataType;
using Format = TensorDim::Format;
using ModelFormat = ml::train::ModelFormat;

/**
 * @brief Helper to create and return an initialized NeuralNetwork
 *        using addLayer API.
 *        FC layer weight dim = (1, 1, input_width, units).
 *        Q4_0 requires: units % 32 == 0 (Q4_0_Tensor width constraint).
 * @param input_width width of input_shape (1:1:input_width)
 * @param units number of FC output units
 */
static std::unique_ptr<nntrainer::NeuralNetwork>
createInitializedNN(unsigned int input_width = 3, unsigned int units = 5) {
  auto nn = std::make_unique<nntrainer::NeuralNetwork>();

  nn->addLayer(ml::train::layer::Input(
    {"name=input", "input_shape=1:1:" + std::to_string(input_width)}));
  nn->addLayer(ml::train::layer::FullyConnected(
    {"name=dense", "unit=" + std::to_string(units)}));

  nn->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn->setProperty({"loss=mse", "batch_size=1"});

  nn->compile();
  nn->initialize();
  return nn;
}

/**
 * @brief Helper to create an initialized NN with two FC layers
 * @param input_width width of input_shape
 * @param units1 number of units in first FC layer
 * @param units2 number of units in second FC layer
 */
static std::unique_ptr<nntrainer::NeuralNetwork>
createTwoLayerNN(unsigned int input_width, unsigned int units1,
                 unsigned int units2) {
  auto nn = std::make_unique<nntrainer::NeuralNetwork>();

  nn->addLayer(ml::train::layer::Input(
    {"name=input", "input_shape=1:1:" + std::to_string(input_width)}));
  nn->addLayer(ml::train::layer::FullyConnected(
    {"name=dense1", "unit=" + std::to_string(units1)}));
  nn->addLayer(ml::train::layer::FullyConnected(
    {"name=dense2", "unit=" + std::to_string(units2)}));

  nn->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn->setProperty({"loss=mse", "batch_size=1"});

  nn->compile();
  nn->initialize();
  return nn;
}

// =============================================================================
// Save with dtype Tests (Commit: [Feat] introduce save with dtype)
// =============================================================================

/**
 * @brief Save before initialization should throw (with default params)
 */
TEST(SaveWithDtype, save_before_init_default_params_n) {
  nntrainer::NeuralNetwork NN;
  std::shared_ptr<nntrainer::LayerNode> layer_node = nntrainer::createLayerNode(
    nntrainer::InputLayer::type, {"input_shape=1:1:3", "normalization=true"});

  EXPECT_NO_THROW(NN.addLayer(layer_node));
  EXPECT_NO_THROW(NN.setProperty({"loss=mse"}));

  EXPECT_THROW(NN.save("test_model.bin"), std::runtime_error);
}

/**
 * @brief Save before initialization should throw (with explicit Q4_0 dtype)
 */
TEST(SaveWithDtype, save_before_init_with_dtype_n) {
  nntrainer::NeuralNetwork NN;
  std::shared_ptr<nntrainer::LayerNode> layer_node = nntrainer::createLayerNode(
    nntrainer::InputLayer::type, {"input_shape=1:1:3", "normalization=true"});

  EXPECT_NO_THROW(NN.addLayer(layer_node));
  EXPECT_NO_THROW(NN.setProperty({"loss=mse"}));

  EXPECT_THROW(
    NN.save("test_model.bin", ModelFormat::MODEL_FORMAT_BIN, DataType::Q4_0),
    std::runtime_error);
}

/**
 * @brief Save before initialization should throw (with layer_dtype_map)
 */
TEST(SaveWithDtype, save_before_init_with_layer_dtype_map_n) {
  nntrainer::NeuralNetwork NN;
  std::shared_ptr<nntrainer::LayerNode> layer_node = nntrainer::createLayerNode(
    nntrainer::InputLayer::type, {"input_shape=1:1:3", "normalization=true"});

  EXPECT_NO_THROW(NN.addLayer(layer_node));
  EXPECT_NO_THROW(NN.setProperty({"loss=mse"}));

  std::map<std::string, DataType> dtype_map = {{"dense", DataType::Q4_0}};
  EXPECT_THROW(NN.save("test_model.bin", ModelFormat::MODEL_FORMAT_BIN,
                       DataType::NONE, dtype_map),
               std::runtime_error);
}

/**
 * @brief Save with non-BIN format and non-NONE dtype should throw
 */
TEST(SaveWithDtype, save_ini_format_with_dtype_throws_n) {
  auto nn = createInitializedNN();

  EXPECT_THROW(
    nn->save("test_model.ini", ModelFormat::MODEL_FORMAT_INI, DataType::Q4_0),
    std::runtime_error);
}

/**
 * @brief Save with INI_WITH_BIN format and non-NONE dtype should throw
 */
TEST(SaveWithDtype, save_ini_with_bin_format_with_dtype_throws_n) {
  auto nn = createInitializedNN();

  EXPECT_THROW(nn->save("test_model.ini",
                        ModelFormat::MODEL_FORMAT_INI_WITH_BIN, DataType::Q4_0),
               std::runtime_error);
}

/**
 * @brief Save with BIN format and NONE dtype (default) should succeed
 */
TEST(SaveWithDtype, save_bin_format_default_dtype_p) {
  auto nn = createInitializedNN();

  EXPECT_NO_THROW(nn->save("test_default_dtype.bin",
                           ModelFormat::MODEL_FORMAT_BIN, DataType::NONE));
  remove("test_default_dtype.bin");
}

/**
 * @brief Save with default parameters should succeed (backward compatibility)
 */
TEST(SaveWithDtype, save_backward_compatible_default_params_p) {
  auto nn = createInitializedNN();

  EXPECT_NO_THROW(nn->save("test_backward_compat.bin"));
  remove("test_backward_compat.bin");
}

/**
 * @brief Save with BIN format and explicit NONE dtype and empty map succeeds
 */
TEST(SaveWithDtype, save_bin_format_none_dtype_empty_map_p) {
  auto nn = createInitializedNN();

  std::map<std::string, DataType> empty_map;
  EXPECT_NO_THROW(nn->save("test_none_empty_map.bin",
                           ModelFormat::MODEL_FORMAT_BIN, DataType::NONE,
                           empty_map));
  remove("test_none_empty_map.bin");
}

/**
 * @brief Save with INI format and NONE dtype should succeed (NONE is default)
 */
TEST(SaveWithDtype, save_ini_format_with_none_dtype_p) {
  auto nn = createInitializedNN();

  EXPECT_NO_THROW(nn->save("test_ini_none.ini", ModelFormat::MODEL_FORMAT_INI,
                           DataType::NONE));
  remove("test_ini_none.ini");
}

/**
 * @brief Saving with BIN format and FP32 dtype should succeed
 *        (FP32 matches the default weight type, so weights are saved as-is)
 */
TEST(SaveWithDtype, save_bin_format_fp32_dtype_p) {
  auto nn = createInitializedNN();

  EXPECT_NO_THROW(nn->save("test_fp32_dtype.bin", ModelFormat::MODEL_FORMAT_BIN,
                           DataType::FP32));
  remove("test_fp32_dtype.bin");
}

/**
 * @brief Verify that save with BIN format produces a non-empty file
 */
TEST(SaveWithDtype, save_bin_produces_nonempty_file_p) {
  auto nn = createInitializedNN();

  std::string file_path = "test_nonempty.bin";
  EXPECT_NO_THROW(nn->save(file_path, ModelFormat::MODEL_FORMAT_BIN));

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.is_open());
  EXPECT_GT(file.tellg(), 0);
  file.close();

  remove(file_path.c_str());
}

/**
 * @brief Save with FP16 dtype should throw (unsupported conversion)
 */
TEST(SaveWithDtype, save_bin_with_fp16_dtype_throws_n) {
  auto nn = createInitializedNN();

  EXPECT_THROW(
    nn->save("test_fp16.bin", ModelFormat::MODEL_FORMAT_BIN, DataType::FP16),
    std::runtime_error);
  remove("test_fp16.bin");
}

/**
 * @brief Save with QINT8 dtype should throw (unsupported conversion)
 */
TEST(SaveWithDtype, save_bin_with_qint8_dtype_throws_n) {
  auto nn = createInitializedNN();

  EXPECT_THROW(
    nn->save("test_qint8.bin", ModelFormat::MODEL_FORMAT_BIN, DataType::QINT8),
    std::runtime_error);
  remove("test_qint8.bin");
}

// =============================================================================
// Q4_0 dimension-dependent tests
//
// FC layer weight dim = (1, 1, input_width, units).
// Q4_0_Tensor constructor requires: batch=1, channel=1, width % 32 == 0.
// quantize_q4_0 requires: (nrow * n_per_row) % 32 == 0.
// Therefore, the critical constraint is: units (width) must be divisible by 32.
//
// File size formulas (MODEL_FORMAT_BIN, default TRAIN execution mode):
//   Per FC layer (H=input_width, W=units):
//     FP32:  weight = H*W*4,           bias = W*4
//     Q4_0:  weight = (H*W)/32 * 18,   bias = W*4 (stays FP32 when height==1)
//   Trailing metadata: epoch_idx(4) + iter(4) = 8 bytes
// =============================================================================

/// epoch_idx + iter written at end of bin file in TRAIN mode
static constexpr std::streamsize TRAIN_METADATA_SIZE = 8;

/**
 * @brief Q4_0 save succeeds when units=32, input=32
 *        weight=(1,1,32,32): width=32 is divisible by 32
 */
TEST(SaveWithDtypeQ4, save_q4_0_units32_input32_p) {
  auto nn = createInitializedNN(32, 32);

  std::string file_path = "test_q4_32_32.bin";
  EXPECT_NO_THROW(
    nn->save(file_path, ModelFormat::MODEL_FORMAT_BIN, DataType::Q4_0));

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.is_open());
  EXPECT_GT(file.tellg(), 0);
  file.close();

  remove(file_path.c_str());
}

/**
 * @brief Q4_0 save succeeds when units=64, input=32
 *        weight=(1,1,32,64): width=64 is divisible by 32
 */
TEST(SaveWithDtypeQ4, save_q4_0_units64_input32_p) {
  auto nn = createInitializedNN(32, 64);

  std::string file_path = "test_q4_32_64.bin";
  EXPECT_NO_THROW(
    nn->save(file_path, ModelFormat::MODEL_FORMAT_BIN, DataType::Q4_0));

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.is_open());
  EXPECT_GT(file.tellg(), 0);
  file.close();

  remove(file_path.c_str());
}

/**
 * @brief Q4_0 save succeeds when units=32, input=64
 *        weight=(1,1,64,32): width=32 is divisible by 32
 */
TEST(SaveWithDtypeQ4, save_q4_0_units32_input64_p) {
  auto nn = createInitializedNN(64, 32);

  std::string file_path = "test_q4_64_32.bin";
  EXPECT_NO_THROW(
    nn->save(file_path, ModelFormat::MODEL_FORMAT_BIN, DataType::Q4_0));

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.is_open());
  EXPECT_GT(file.tellg(), 0);
  file.close();

  remove(file_path.c_str());
}

/**
 * @brief Q4_0 save fails when units=5 (not divisible by 32)
 *        weight=(1,1,3,5): width=5 is not divisible by 32
 */
TEST(SaveWithDtypeQ4, save_q4_0_units5_input3_n) {
  auto nn = createInitializedNN(3, 5);

  EXPECT_THROW(
    nn->save("test_q4_3_5.bin", ModelFormat::MODEL_FORMAT_BIN, DataType::Q4_0),
    std::invalid_argument);
  remove("test_q4_3_5.bin");
}

/**
 * @brief Q4_0 save fails when units=16 (not divisible by 32)
 *        weight=(1,1,32,16): width=16 is not divisible by 32
 */
TEST(SaveWithDtypeQ4, save_q4_0_units16_input32_n) {
  auto nn = createInitializedNN(32, 16);

  EXPECT_THROW(nn->save("test_q4_32_16.bin", ModelFormat::MODEL_FORMAT_BIN,
                        DataType::Q4_0),
               std::invalid_argument);
  remove("test_q4_32_16.bin");
}

/**
 * @brief Q4_0 save fails when units=48 (not divisible by 32... wait, 48/32
 *        is not integer). Actually 48 is NOT divisible by 32, so this fails.
 *        weight=(1,1,32,48): width=48 is not divisible by 32
 */
TEST(SaveWithDtypeQ4, save_q4_0_units48_input32_n) {
  auto nn = createInitializedNN(32, 48);

  EXPECT_THROW(nn->save("test_q4_32_48.bin", ModelFormat::MODEL_FORMAT_BIN,
                        DataType::Q4_0),
               std::invalid_argument);
  remove("test_q4_32_48.bin");
}

/**
 * @brief Q4_0 save succeeds when units=128 (divisible by 32)
 *        weight=(1,1,32,128): width=128 is divisible by 32
 */
TEST(SaveWithDtypeQ4, save_q4_0_units128_input32_p) {
  auto nn = createInitializedNN(32, 128);

  std::string file_path = "test_q4_32_128.bin";
  EXPECT_NO_THROW(
    nn->save(file_path, ModelFormat::MODEL_FORMAT_BIN, DataType::Q4_0));

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.is_open());
  EXPECT_GT(file.tellg(), 0);
  file.close();

  remove(file_path.c_str());
}

/**
 * @brief Q4_0 bin file must have the exact expected byte size.
 *        Model: input(1:1:32) -> dense(unit=32)
 *        FC weight: (1,1,32,32), bias: (1,1,1,32)
 *
 *        FP32: weight = 32*32*4 = 4096, bias = 32*4 = 128
 *              total = 4224 bytes
 *
 *        Q4_0: weight = (32*32)/32 * 18 = 576 (quantized)
 *              bias   = 32*4 = 128 (stays FP32, height==1)
 *              total  = 704 bytes
 */
TEST(SaveWithDtypeQ4, save_q4_0_exact_file_size_p) {
  const unsigned int H = 32, W = 32;
  auto nn = createInitializedNN(H, W);

  std::string fp32_path = "test_fp32_size.bin";
  std::string q4_path = "test_q4_size.bin";

  EXPECT_NO_THROW(
    nn->save(fp32_path, ModelFormat::MODEL_FORMAT_BIN, DataType::NONE));
  EXPECT_NO_THROW(
    nn->save(q4_path, ModelFormat::MODEL_FORMAT_BIN, DataType::Q4_0));

  std::ifstream fp32_file(fp32_path, std::ios::binary | std::ios::ate);
  std::ifstream q4_file(q4_path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(fp32_file.is_open());
  EXPECT_TRUE(q4_file.is_open());

  const std::streamsize expected_fp32 = (H * W + W) * 4 + TRAIN_METADATA_SIZE;
  const std::streamsize expected_q4 =
    (H * W / 32) * 18 + W * 4 + TRAIN_METADATA_SIZE;

  EXPECT_EQ(fp32_file.tellg(), expected_fp32);
  EXPECT_EQ(q4_file.tellg(), expected_q4);
  EXPECT_LT(q4_file.tellg(), fp32_file.tellg());

  fp32_file.close();
  q4_file.close();

  remove(fp32_path.c_str());
  remove(q4_path.c_str());
}

/**
 * @brief Q4_0 save with NONE dtype (default) still saves as FP32 for a
 *        Q4_0-compatible model, preserving backward compatibility
 */
TEST(SaveWithDtypeQ4, save_none_dtype_same_as_fp32_p) {
  auto nn = createInitializedNN(32, 32);

  std::string none_path = "test_none_path.bin";
  std::string fp32_path = "test_fp32_path.bin";

  EXPECT_NO_THROW(
    nn->save(none_path, ModelFormat::MODEL_FORMAT_BIN, DataType::NONE));
  EXPECT_NO_THROW(
    nn->save(fp32_path, ModelFormat::MODEL_FORMAT_BIN, DataType::FP32));

  std::ifstream none_file(none_path, std::ios::binary | std::ios::ate);
  std::ifstream fp32_file(fp32_path, std::ios::binary | std::ios::ate);

  EXPECT_EQ(none_file.tellg(), fp32_file.tellg());

  none_file.close();
  fp32_file.close();

  remove(none_path.c_str());
  remove(fp32_path.c_str());
}

/**
 * @brief layer_dtype_map allows Q4_0 only for a specific Q4_0-compatible layer,
 *        while others stay as FP32. Verify exact file size.
 *
 *        Model: input(1:1:32) -> dense1(unit=32) -> dense2(unit=5)
 *        dense1 (Q4_0): (32*32)/32*18 + 32*4 = 576+128 = 704
 *        dense2 (FP32): (32*5+5)*4 = 660
 *        total = 1364
 */
TEST(SaveWithDtypeQ4, save_layer_dtype_map_compatible_layer_only_p) {
  auto nn = createTwoLayerNN(32, 32, 5);

  std::string file_path = "test_q4_map_compat.bin";
  std::map<std::string, DataType> dtype_map = {{"dense1", DataType::Q4_0}};

  EXPECT_NO_THROW(nn->save(file_path, ModelFormat::MODEL_FORMAT_BIN,
                           DataType::NONE, dtype_map));

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.is_open());

  const std::streamsize expected =
    (32 * 32 / 32) * 18 + 32 * 4 + // dense1: Q4_0 weight + FP32 bias
    (32 * 5 + 5) * 4 +             // dense2: FP32 weight + FP32 bias
    TRAIN_METADATA_SIZE;
  EXPECT_EQ(file.tellg(), expected);
  file.close();

  remove(file_path.c_str());
}

/**
 * @brief layer_dtype_map applying Q4_0 to an incompatible layer should throw
 *        dense2 weight: (1,1,32,5) - NOT Q4_0 compatible (5 % 32 != 0)
 */
TEST(SaveWithDtypeQ4, save_layer_dtype_map_incompatible_layer_n) {
  auto nn = createTwoLayerNN(32, 32, 5);

  std::string file_path = "test_q4_map_incompat.bin";
  std::map<std::string, DataType> dtype_map = {{"dense2", DataType::Q4_0}};

  EXPECT_THROW(nn->save(file_path, ModelFormat::MODEL_FORMAT_BIN,
                        DataType::NONE, dtype_map),
               std::invalid_argument);
  remove(file_path.c_str());
}

/**
 * @brief Global Q4_0 dtype fails when any layer has incompatible dimensions
 *        dense2 weight: (1,1,32,5) - NOT Q4_0 compatible
 */
TEST(SaveWithDtypeQ4, save_global_q4_0_with_incompatible_layer_n) {
  auto nn = createTwoLayerNN(32, 32, 5);

  EXPECT_THROW(nn->save("test_q4_global.bin", ModelFormat::MODEL_FORMAT_BIN,
                        DataType::Q4_0),
               std::invalid_argument);
  remove("test_q4_global.bin");
}

/**
 * @brief Global Q4_0 dtype succeeds when all layers have compatible dimensions.
 *        Verify exact file size.
 *
 *        Model: input(1:1:32) -> dense1(unit=32) -> dense2(unit=64)
 *        dense1: Q4_0 weight = (32*32)/32*18 = 576, bias FP32 = 32*4 = 128
 *        dense2: Q4_0 weight = (32*64)/32*18 = 1152, bias FP32 = 64*4 = 256
 *        total = 576+128+1152+256 = 2112
 */
TEST(SaveWithDtypeQ4, save_global_q4_0_all_compatible_p) {
  auto nn = createTwoLayerNN(32, 32, 64);

  std::string file_path = "test_q4_global_compat.bin";
  EXPECT_NO_THROW(
    nn->save(file_path, ModelFormat::MODEL_FORMAT_BIN, DataType::Q4_0));

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.is_open());

  const std::streamsize expected =
    (32 * 32 / 32) * 18 + 32 * 4 + // dense1: Q4_0 weight + FP32 bias
    (32 * 64 / 32) * 18 + 64 * 4 + // dense2: Q4_0 weight + FP32 bias
    TRAIN_METADATA_SIZE;
  EXPECT_EQ(file.tellg(), expected);
  file.close();

  remove(file_path.c_str());
}

/**
 * @brief layer_dtype_map overrides global dtype: global=NONE, layer=Q4_0
 *        Only the specified layer should be quantized.
 *        Verify exact file sizes.
 *
 *        Model: input(1:1:32) -> dense1(unit=32) -> dense2(unit=64)
 *        dense1: weight(1,1,32,32), bias(1,1,1,32)
 *        dense2: weight(1,1,32,64), bias(1,1,1,64)
 *
 *        FP32 total: (32*32+32)*4 + (32*64+64)*4 = 4224 + 8448 = 12672
 *
 *        Map (dense1=Q4_0, dense2=FP32):
 *          dense1: (32*32)/32*18 + 32*4 = 576+128 = 704
 *          dense2: (32*64+64)*4 = 8448
 *          total = 9152
 */
TEST(SaveWithDtypeQ4, save_layer_dtype_map_overrides_global_p) {
  auto nn = createTwoLayerNN(32, 32, 64);

  std::string global_none_path = "test_q4_override_none.bin";
  std::string map_q4_path = "test_q4_override_map.bin";

  // Save all as FP32 (global NONE)
  EXPECT_NO_THROW(
    nn->save(global_none_path, ModelFormat::MODEL_FORMAT_BIN, DataType::NONE));

  // Save dense1 as Q4_0 via map, rest as FP32 (global NONE)
  std::map<std::string, DataType> dtype_map = {{"dense1", DataType::Q4_0}};
  EXPECT_NO_THROW(nn->save(map_q4_path, ModelFormat::MODEL_FORMAT_BIN,
                           DataType::NONE, dtype_map));

  std::ifstream none_file(global_none_path, std::ios::binary | std::ios::ate);
  std::ifstream map_file(map_q4_path, std::ios::binary | std::ios::ate);

  const std::streamsize expected_fp32 =
    (32 * 32 + 32) * 4 + (32 * 64 + 64) * 4 + TRAIN_METADATA_SIZE;
  const std::streamsize expected_map =
    (32 * 32 / 32) * 18 + 32 * 4 + (32 * 64 + 64) * 4 + TRAIN_METADATA_SIZE;

  EXPECT_EQ(none_file.tellg(), expected_fp32);
  EXPECT_EQ(map_file.tellg(), expected_map);

  none_file.close();
  map_file.close();

  remove(global_none_path.c_str());
  remove(map_q4_path.c_str());
}

/**
 * @brief layer_dtype_map can exclude a layer from global Q4_0 by setting FP32.
 *        Verify exact file size.
 *
 *        Global: Q4_0, but dense2 is overridden to FP32 via map
 *        Model: input(1:1:32) -> dense1(unit=32) -> dense2(unit=64)
 *
 *        dense1 (Q4_0): (32*32)/32*18 + 32*4 = 576+128 = 704
 *        dense2 (FP32): (32*64+64)*4 = 8448
 *        total = 9152
 */
TEST(SaveWithDtypeQ4, save_layer_dtype_map_exclude_from_global_q4_p) {
  auto nn = createTwoLayerNN(32, 32, 64);

  std::string file_path = "test_q4_map_exclude.bin";
  std::map<std::string, DataType> dtype_map = {{"dense2", DataType::FP32}};

  EXPECT_NO_THROW(nn->save(file_path, ModelFormat::MODEL_FORMAT_BIN,
                           DataType::Q4_0, dtype_map));

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.is_open());

  const std::streamsize expected =
    (32 * 32 / 32) * 18 + 32 * 4 + (32 * 64 + 64) * 4 + TRAIN_METADATA_SIZE;
  EXPECT_EQ(file.tellg(), expected);
  file.close();

  remove(file_path.c_str());
}

/**
 * @brief Q4_0 save fails when input=16 (height not divisible by 32)
 *        weight=(1,1,16,32): height=16 is not divisible by 32
 *        quantize_q4_0 requires n_per_row (=height) % 32 == 0
 */
TEST(SaveWithDtypeQ4, save_q4_0_units32_input16_n) {
  auto nn = createInitializedNN(16, 32);

  EXPECT_THROW(nn->save("test_q4_16_32.bin", ModelFormat::MODEL_FORMAT_BIN,
                        DataType::Q4_0),
               std::invalid_argument);
  remove("test_q4_16_32.bin");
}

/**
 * @brief Q4_0 save fails when units=1 (trivially not divisible by 32)
 *        weight=(1,1,32,1): width=1 not divisible by 32
 */
TEST(SaveWithDtypeQ4, save_q4_0_units1_n) {
  auto nn = createInitializedNN(32, 1);

  EXPECT_THROW(
    nn->save("test_q4_32_1.bin", ModelFormat::MODEL_FORMAT_BIN, DataType::Q4_0),
    std::invalid_argument);
  remove("test_q4_32_1.bin");
}

// =============================================================================
// Quantize-Save-Load-Inference Comparison Tests
//
// These tests verify that a model saved with quantized dtype can be loaded
// back and produce inference results close to the original FP32 model.
// Due to quantization error, results are compared with a tolerance.
//
// When saving with Q4_0, weights are quantized to Q4_0 format on disk.
// To load them back, the receiving model must be configured with
// model_tensor_type="Q4_0-FP32" and compiled/initialized in INFERENCE mode
// so that its weight tensors match the Q4_0 layout.
// =============================================================================

using ExecutionMode = ml::train::ExecutionMode;

/**
 * @brief Helper: build a deterministic input tensor from a seed.
 */
static nntrainer::Tensor buildInput(unsigned int width,
                                    unsigned int seed = 42) {
  nntrainer::TensorDim dim({1, 1, 1, width});
  nntrainer::Tensor input(dim);
  srand(seed);
  for (unsigned int w = 0; w < width; ++w)
    input.setValue(0, 0, 0, w,
                   static_cast<float>(rand()) / static_cast<float>(RAND_MAX) -
                     0.5f);
  return input;
}

/**
 * @brief Helper: run inference on a nntrainer::NeuralNetwork and return
 *        a copy of the output tensor.
 */
static nntrainer::Tensor runInference(nntrainer::NeuralNetwork &nn,
                                      const nntrainer::Tensor &input) {
  nntrainer::sharedConstTensors in = {MAKE_SHARED_TENSOR(input)};
  nntrainer::sharedConstTensors out = nn.inference(in, false);
  return out[0]->clone();
}

/**
 * @brief Save model as Q4_0, load it into a Q4_0-typed inference model,
 *        run inference, and compare with original FP32 inference.
 *        Model: input(1:1:32) -> dense(unit=32)
 *        Weight: (1,1,32,32) — fully Q4_0-compatible.
 *
 *        Uses ml::train::createModel API for the Q4_0 inference model,
 *        following the pattern used in integration_test_fsu.cpp.
 */
TEST(SaveWithDtypeInference, save_q4_0_load_inference_compare_p) {
  const unsigned int input_width = 32;
  const unsigned int units = 32;

  // --- Step 1: create FP32 model, run inference ---
  auto nn_orig = createInitializedNN(input_width, units);
  nntrainer::Tensor input = buildInput(input_width);
  nntrainer::Tensor out_orig = runInference(*nn_orig, input);

  // --- Step 2: save weights as Q4_0 ---
  std::string q4_path = "test_infer_q4.bin";
  ASSERT_NO_THROW(
    nn_orig->save(q4_path, ModelFormat::MODEL_FORMAT_BIN, DataType::Q4_0));

  // --- Step 3: create a Q4_0-typed model for inference and load ---
  auto nn_q4 =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
  nn_q4->addLayer(ml::train::createLayer(
    "input", {"name=input", "input_shape=1:1:" + std::to_string(input_width)}));
  nn_q4->addLayer(ml::train::createLayer(
    "fully_connected", {"name=dense", "unit=" + std::to_string(units)}));
  nn_q4->setProperty({"batch_size=1", "model_tensor_type=Q4_0-FP32"});
  ASSERT_EQ(nn_q4->compile(ExecutionMode::INFERENCE), ML_ERROR_NONE);
  ASSERT_EQ(nn_q4->initialize(ExecutionMode::INFERENCE), ML_ERROR_NONE);
  ASSERT_NO_THROW(nn_q4->load(q4_path, ModelFormat::MODEL_FORMAT_BIN));

  // --- Step 4: run inference on loaded Q4_0 model ---
  float *input_data = input.getData<float>();
  std::vector<float *> in_raw = {input_data};
  std::vector<float *> answer = nn_q4->inference(1, in_raw);

  // --- Step 5: compare outputs ---
  for (unsigned int l = 0; l < units; ++l) {
    float orig_val = out_orig.getValue<float>(0, 0, 0, l);
    float load_val = answer[0][l];
    EXPECT_NEAR(orig_val, load_val, 0.5f) << "Mismatch at output index " << l;
  }

  remove(q4_path.c_str());
}

/**
 * @brief Partial quantization via layer_dtype_map: dense1=Q4_0, dense2=FP32.
 *        Save, load into a matching inference model, and compare with
 *        original FP32 inference output.
 *
 *        Model: input(1:1:32) -> dense1(unit=32) -> dense2(unit=64)
 *        dense1 weight: (1,1,32,32) — Q4_0 compatible
 *        dense2 weight: (1,1,32,64) — stays FP32
 *
 *        The receiving inference model uses model_tensor_type=Q4_0-FP32
 *        globally, then overrides dense2 with weight_dtype=FP32.
 */
TEST(SaveWithDtypeInference, save_partial_q4_load_inference_compare_p) {
  const unsigned int input_width = 32;
  const unsigned int units1 = 32;
  const unsigned int units2 = 64;

  // --- Step 1: create FP32 model, run inference ---
  auto nn_orig = createTwoLayerNN(input_width, units1, units2);
  nntrainer::Tensor input = buildInput(input_width);
  nntrainer::Tensor out_orig = runInference(*nn_orig, input);

  // --- Step 2: save with partial quantization (dense1=Q4_0, dense2=FP32) ---
  std::string save_path = "test_infer_partial_q4.bin";
  std::map<std::string, DataType> dtype_map = {{"dense1", DataType::Q4_0}};
  ASSERT_NO_THROW(nn_orig->save(save_path, ModelFormat::MODEL_FORMAT_BIN,
                                DataType::NONE, dtype_map));

  // --- Step 3: verify exact file size ---
  {
    std::ifstream f(save_path, std::ios::binary | std::ios::ate);
    const std::streamsize expected =
      (32 * 32 / 32) * 18 + 32 * 4 + // dense1: Q4_0 weight + FP32 bias
      (32 * 64 + 64) * 4 +           // dense2: FP32 weight + FP32 bias
      TRAIN_METADATA_SIZE;
    EXPECT_EQ(f.tellg(), expected);
  }

  // --- Step 4: create a matching inference model ---
  //     Global model_tensor_type=Q4_0-FP32, but dense2 overridden to FP32
  auto nn_load =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
  nn_load->addLayer(ml::train::createLayer(
    "input", {"name=input", "input_shape=1:1:" + std::to_string(input_width)}));
  nn_load->addLayer(ml::train::createLayer(
    "fully_connected", {"name=dense1", "unit=" + std::to_string(units1)}));
  nn_load->addLayer(ml::train::createLayer(
    "fully_connected",
    {"name=dense2", "unit=" + std::to_string(units2), "weight_dtype=FP32"}));
  nn_load->setProperty({"batch_size=1", "model_tensor_type=Q4_0-FP32"});
  ASSERT_EQ(nn_load->compile(ExecutionMode::INFERENCE), ML_ERROR_NONE);
  ASSERT_EQ(nn_load->initialize(ExecutionMode::INFERENCE), ML_ERROR_NONE);
  ASSERT_NO_THROW(nn_load->load(save_path, ModelFormat::MODEL_FORMAT_BIN));

  // --- Step 5: run inference on loaded model ---
  float *input_data = input.getData<float>();
  std::vector<float *> in_raw = {input_data};
  std::vector<float *> answer = nn_load->inference(1, in_raw);

  // --- Step 6: compare outputs ---
  // Only dense1 is quantized → error comes from first layer only.
  for (unsigned int l = 0; l < units2; ++l) {
    float orig_val = out_orig.getValue<float>(0, 0, 0, l);
    float load_val = answer[0][l];
    EXPECT_NEAR(orig_val, load_val, 1.0f) << "Mismatch at output index " << l;
  }

  remove(save_path.c_str());
}

/**
 * @brief Save a model as QS4CX, then load it back and compare inference with
 *        the original FP32 model.
 *
 *        The QS4CX record carries no version and the .bin no per-tensor dtype,
 *        so the reader tells the trimmed record stride (N * ceil(K/2) nibbles
 *        + N fp32 scales, what the writer emits) from the padded one
 *        (N * (K + 1) / 2 nibbles, floor(N/2) of them pad) by which of the two
 *        totals the file size fits. This locks both halves: the file must have
 *        the trimmed size, and reading it back must land the per-channel
 *        scales where the GEMM looks for them -- a stride mix-up reads the
 *        scales out of the nibbles and the output is nowhere near.
 *
 *        Model: input(1:1:32) -> dense(unit=32, no bias), so the file holds
 *        exactly one QS4CX record, with an even K where the two strides differ
 *        (they coincide for odd K).
 */
TEST(SaveWithDtypeInference, save_qs4cx_trimmed_record_load_p) {
  const unsigned int K = 32; // input width, even: the two strides differ here
  const unsigned int N = 32; // units

  auto nn_orig = std::make_unique<nntrainer::NeuralNetwork>();
  nn_orig->addLayer(ml::train::layer::Input(
    {"name=input", "input_shape=1:1:" + std::to_string(K)}));
  nn_orig->addLayer(ml::train::layer::FullyConnected(
    {"name=dense", "unit=" + std::to_string(N), "disable_bias=true"}));
  nn_orig->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn_orig->setProperty({"loss=mse", "batch_size=1"});
  ASSERT_EQ(nn_orig->compile(), ML_ERROR_NONE);
  ASSERT_EQ(nn_orig->initialize(), ML_ERROR_NONE);

  nntrainer::Tensor input = buildInput(K);
  nntrainer::Tensor out_orig = runInference(*nn_orig, input);

  std::string qs4cx_path = "test_infer_qs4cx.bin";
  ASSERT_NO_THROW(
    nn_orig->save(qs4cx_path, ModelFormat::MODEL_FORMAT_BIN, DataType::QS4CX));

  const std::streamsize trimmed =
    N * ((K + 1) / 2) + N * 4 + TRAIN_METADATA_SIZE;
  const std::streamsize padded = N * (K + 1) / 2 + N * 4 + TRAIN_METADATA_SIZE;
  ASSERT_NE(trimmed, padded);

  std::ifstream qs4cx_file(qs4cx_path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(qs4cx_file.is_open());
  EXPECT_EQ(qs4cx_file.tellg(), trimmed);
  qs4cx_file.close();

  auto nn_qs4cx =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
  nn_qs4cx->addLayer(ml::train::createLayer(
    "input", {"name=input", "input_shape=1:1:" + std::to_string(K)}));
  nn_qs4cx->addLayer(ml::train::createLayer(
    "fully_connected",
    {"name=dense", "unit=" + std::to_string(N), "disable_bias=true"}));
  nn_qs4cx->setProperty({"batch_size=1", "model_tensor_type=QS4CX-FP32"});
  ASSERT_EQ(nn_qs4cx->compile(ExecutionMode::INFERENCE), ML_ERROR_NONE);
  ASSERT_EQ(nn_qs4cx->initialize(ExecutionMode::INFERENCE), ML_ERROR_NONE);
  ASSERT_NO_THROW(nn_qs4cx->load(qs4cx_path, ModelFormat::MODEL_FORMAT_BIN));

  float *input_data = input.getData<float>();
  std::vector<float *> in_raw = {input_data};
  std::vector<float *> answer = nn_qs4cx->inference(1, in_raw);

  for (unsigned int l = 0; l < N; ++l) {
    float orig_val = out_orig.getValue<float>(0, 0, 0, l);
    float load_val = answer[0][l];
    EXPECT_NEAR(orig_val, load_val, 0.5f) << "Mismatch at output index " << l;
  }

  remove(qs4cx_path.c_str());
}

/**
 * @brief A QS4CX save leaves the bias FP32, because that is the type the
 *        reader asks for.
 *
 *        The bias of a dense layer is (1, 1, 1, N), so its height is 1. A
 *        layer whose weight is quantized requests its bias as FP32 rather
 *        than as the weight type (FullyConnectedLayer::finalize spells this
 *        out), so N floats are exactly what load() goes on to read. Writing a
 *        QS4CX record there instead would put N * ceil(1/2) nibble bytes plus
 *        N scales where the reader expects N floats, and every byte after it
 *        would be read from the wrong offset.
 *
 *        Model: input(1:1:32) -> dense(unit=32) with the bias left enabled,
 *        so the file holds a quantized weight followed by an FP32 bias rather
 *        than one record alone.
 *
 * @note  Inference is compared as well as the file size: a size alone cannot
 *        tell a correctly placed bias from one whose bytes are read at the
 *        right offset with the wrong type.
 */
TEST(SaveWithDtypeInference, save_qs4cx_bias_record_load_p) {
  const unsigned int K = 32; // input width
  const unsigned int N = 32; // units

  auto nn_orig = std::make_unique<nntrainer::NeuralNetwork>();
  nn_orig->addLayer(ml::train::layer::Input(
    {"name=input", "input_shape=1:1:" + std::to_string(K)}));
  nn_orig->addLayer(ml::train::layer::FullyConnected(
    {"name=dense", "unit=" + std::to_string(N)}));
  nn_orig->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn_orig->setProperty({"loss=mse", "batch_size=1"});
  ASSERT_EQ(nn_orig->compile(), ML_ERROR_NONE);
  ASSERT_EQ(nn_orig->initialize(), ML_ERROR_NONE);

  nntrainer::Tensor input = buildInput(K);
  nntrainer::Tensor out_orig = runInference(*nn_orig, input);

  std::string qs4cx_path = "test_infer_qs4cx_bias.bin";
  ASSERT_NO_THROW(
    nn_orig->save(qs4cx_path, ModelFormat::MODEL_FORMAT_BIN, DataType::QS4CX));

  // The weight (K, N) is a QS4CX record; the bias (1, N) stays N floats.
  const std::streamsize weight_record = N * ((K + 1) / 2) + N * 4;
  const std::streamsize bias_record = N * 4;

  std::ifstream qs4cx_file(qs4cx_path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(qs4cx_file.is_open());
  EXPECT_EQ(qs4cx_file.tellg(),
            weight_record + bias_record + TRAIN_METADATA_SIZE);
  qs4cx_file.close();

  auto nn_qs4cx =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
  nn_qs4cx->addLayer(ml::train::createLayer(
    "input", {"name=input", "input_shape=1:1:" + std::to_string(K)}));
  nn_qs4cx->addLayer(ml::train::createLayer(
    "fully_connected", {"name=dense", "unit=" + std::to_string(N)}));
  nn_qs4cx->setProperty({"batch_size=1", "model_tensor_type=QS4CX-FP32"});
  ASSERT_EQ(nn_qs4cx->compile(ExecutionMode::INFERENCE), ML_ERROR_NONE);
  ASSERT_EQ(nn_qs4cx->initialize(ExecutionMode::INFERENCE), ML_ERROR_NONE);

  // Each record is the size the reader assumes, so the bias is read from the
  // offset the weight before it ends at and the file is fully consumed.
  ASSERT_NO_THROW(nn_qs4cx->load(qs4cx_path, ModelFormat::MODEL_FORMAT_BIN));

  float *input_data = input.getData<float>();
  std::vector<float *> in_raw = {input_data};
  std::vector<float *> answer = nn_qs4cx->inference(1, in_raw);

  for (unsigned int l = 0; l < N; ++l) {
    float orig_val = out_orig.getValue<float>(0, 0, 0, l);
    float load_val = answer[0][l];
    EXPECT_NEAR(orig_val, load_val, 0.5f) << "Mismatch at output index " << l;
  }

  remove(qs4cx_path.c_str());
}

// =============================================================================
// Main function
// =============================================================================

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Failed to initialize google test" << std::endl;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Failed to run all tests" << std::endl;
  }

  return result;
}
