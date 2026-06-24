// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_embedding_sidecar_lut.cpp
 * @date   11 June 2026
 * @brief  CausalLM EmbeddingLayer sidecar LUT tests
 */

#include <embedding_layer.h>
#include <layer_context.h>
#include <var_grad.h>
#include <weight.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

class TempDir {
public:
  explicit TempDir(const std::string &name) {
    path_ = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TempDir() { std::filesystem::remove_all(path_); }

  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

void writeBytes(const std::filesystem::path &path,
                const std::vector<uint8_t> &bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

void writeU16(const std::filesystem::path &path,
              const std::vector<uint16_t> &values) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(values.data()),
            values.size() * sizeof(uint16_t));
}

void writeText(const std::filesystem::path &path, const std::string &text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  ASSERT_TRUE(out.is_open());
  out << text;
}

std::vector<nntrainer::Weight *>
makeWeightView(std::vector<nntrainer::Weight> &weights) {
  std::vector<nntrainer::Weight *> view;
  view.reserve(weights.size());
  for (auto &weight : weights)
    view.push_back(&weight);
  return view;
}

std::vector<nntrainer::Var_Grad *>
makeVarGradView(std::vector<nntrainer::Var_Grad> &vars) {
  std::vector<nntrainer::Var_Grad *> view;
  view.reserve(vars.size());
  for (auto &var : vars)
    view.push_back(&var);
  return view;
}

nntrainer::RunLayerContext
makeRunContext(std::vector<nntrainer::Weight> &weights,
               std::vector<nntrainer::Var_Grad> &inputs,
               std::vector<nntrainer::Var_Grad> &outputs,
               std::vector<nntrainer::Var_Grad> &tensors) {
  return nntrainer::RunLayerContext(
    "embedding_sidecar_lut", true, 0.0f, false, 1.0f, nullptr, false,
    makeWeightView(weights), makeVarGradView(inputs), makeVarGradView(outputs),
    makeVarGradView(tensors));
}

nntrainer::TensorDim makeInputDim(nntrainer::TensorDim::DataType dtype,
                                  unsigned int width) {
  return nntrainer::TensorDim(
    {1, 1, 1, width, nntrainer::Tformat::NCHW, dtype});
}

} // namespace

TEST(CausalLmEmbeddingSidecarLut, rawUint16RequiresExactHintedSize) {
  TempDir dir("nntrainer_embedding_sidecar_raw_u16");
  const auto raw_path = dir.path() / "embedding.u16";
  writeU16(raw_path, {1, 2, 3, 4, 5, 6});

  auto lut = causallm::get_or_load_quant_lut(raw_path.string(), 2, 3);

  ASSERT_NE(lut, nullptr);
  EXPECT_TRUE(lut->is_raw_u16);
  EXPECT_EQ(lut->in_dim, 2u);
  EXPECT_EQ(lut->out_dim, 3u);
  EXPECT_EQ(lut->bytes.size(), 6u * sizeof(uint16_t));

  const auto bad_path = dir.path() / "bad_embedding.u16";
  writeU16(bad_path, {1, 2, 3, 4, 5, 6});
  EXPECT_THROW(causallm::get_or_load_quant_lut(bad_path.string(), 2, 4),
               std::runtime_error);
}

TEST(CausalLmEmbeddingSidecarLut, ufixed8ManifestUsesRelativePathAndDims) {
  TempDir dir("nntrainer_embedding_sidecar_ufixed8");
  const auto lut_path = dir.path() / "tables" / "embedding.bin";
  const auto manifest_path = dir.path() / "manifest.json";

  writeBytes(lut_path, {0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb});
  writeText(manifest_path, R"({
    "lut-path": "tables/embedding.bin",
    "size": 4,
    "quant-param": { "scale": 0.5, "offset": -1 }
  })");

  auto lut = causallm::get_or_load_quant_lut(manifest_path.string(), 3, 4);

  ASSERT_NE(lut, nullptr);
  EXPECT_FALSE(lut->is_raw_u16);
  EXPECT_FALSE(lut->is_signed4);
  EXPECT_EQ(lut->in_dim, 3u);
  EXPECT_EQ(lut->out_dim, 4u);
  EXPECT_FLOAT_EQ(lut->scale, 0.5f);
  EXPECT_EQ(lut->offset, -1);
  EXPECT_EQ(lut->bytes.size(), 6u);
}

TEST(CausalLmEmbeddingSidecarLut, sfixed4ManifestParsesAndValidatesRowScale) {
  TempDir dir("nntrainer_embedding_sidecar_sfixed4");
  const auto lut_path = dir.path() / "embedding.s4";
  const auto manifest_path = dir.path() / "manifest.json";

  writeBytes(lut_path, {0x21, 0x43, 0x65, 0x87});
  writeText(manifest_path, R"({
    "lut-path": "embedding.s4",
    "size": 4,
    "datatype": "sfixed4",
    "quant-param": { "scale": [0.5, 1.5] }
  })");

  auto lut = causallm::get_or_load_quant_lut(manifest_path.string(), 2, 4);

  ASSERT_NE(lut, nullptr);
  EXPECT_FALSE(lut->is_raw_u16);
  EXPECT_TRUE(lut->is_signed4);
  EXPECT_EQ(lut->in_dim, 2u);
  EXPECT_EQ(lut->out_dim, 4u);
  ASSERT_EQ(lut->row_scales.size(), 2u);
  EXPECT_FLOAT_EQ(lut->row_scales[0], 0.5f);
  EXPECT_FLOAT_EQ(lut->row_scales[1], 1.5f);

  const auto bad_manifest_path = dir.path() / "bad_manifest.json";
  writeText(bad_manifest_path, R"({
    "lut-path": "embedding.s4",
    "size": 4,
    "datatype": "sfixed4",
    "quant-param": { "scale": [0.5] }
  })");

  EXPECT_THROW(causallm::get_or_load_quant_lut(bad_manifest_path.string(), 2,
                                               4),
               std::invalid_argument);
}

TEST(CausalLmEmbeddingSidecarLut, unsupportedManifestDatatypeThrows) {
  TempDir dir("nntrainer_embedding_sidecar_unsupported");
  const auto lut_path = dir.path() / "embedding.bin";
  const auto manifest_path = dir.path() / "manifest.json";

  writeBytes(lut_path, {0x21, 0x43});
  writeText(manifest_path, R"({
    "lut-path": "embedding.bin",
    "size": 4,
    "datatype": "int8",
    "quant-param": { "scale": 0.5, "offset": 0 }
  })");

  EXPECT_THROW(causallm::get_or_load_quant_lut(manifest_path.string(), 1, 4),
               std::runtime_error);
}

TEST(CausalLmEmbeddingSidecarLut, ufixed8DecodeRequantsToUint16) {
  TempDir dir("nntrainer_embedding_sidecar_decode");
  const auto lut_path = dir.path() / "embedding.bin";
  const auto manifest_path = dir.path() / "manifest.json";

  writeBytes(lut_path, {0x21, 0x43, 0x65, 0x87});
  writeText(manifest_path, R"({
    "lut-path": "embedding.bin",
    "size": 4,
    "datatype": "ufixed8",
    "quant-param": { "scale": 0.5, "offset": -1 }
  })");

  auto lut = causallm::get_or_load_quant_lut(manifest_path.string(), 2, 4);

  uint16_t output[4] = {};
  causallm::decode_quant_lut_row_to_uint16(*lut, 1, 2.0f, 0.25f, -3, output,
                                           4);

  EXPECT_EQ(output[0], 19);
  EXPECT_EQ(output[1], 23);
  EXPECT_EQ(output[2], 27);
  EXPECT_EQ(output[3], 31);
}

TEST(CausalLmEmbeddingSidecarLut,
     rawUint16LayerForcesFp32TokenInputAndCopiesUint16Rows) {
  TempDir dir("nntrainer_embedding_sidecar_layer_raw_u16");
  const auto raw_path = dir.path() / "embedding.u16";
  writeU16(raw_path, {10, 11, 12, 20, 21, 22, 30, 31, 32});

  causallm::EmbeddingLayer layer;
  layer.setProperty({"in_dim=3", "out_dim=3",
                     "quantized_lut_path=" + raw_path.string()});

  nntrainer::InitLayerContext init_context(
    {makeInputDim(nntrainer::TensorDim::DataType::UINT16, 2)}, {true}, false,
    "embedding_sidecar_lut", "", 0.0f, {"NCHW", "FP32", "UINT16"});

  ASSERT_NO_THROW(layer.finalize(init_context));
  ASSERT_EQ(init_context.getInputDimensions().size(), 1u);
  EXPECT_EQ(init_context.getInputDimensions()[0].getDataType(),
            nntrainer::TensorDim::DataType::FP32);
  ASSERT_EQ(init_context.getOutSpecs().size(), 1u);
  EXPECT_EQ(init_context.getOutSpecs()[0].variable_spec.dim.getDataType(),
            nntrainer::TensorDim::DataType::UINT16);
  EXPECT_EQ(init_context.getNumWeights(), 0u);

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  float *input_data = inputs[0].getVariableRef().getData<float>();
  input_data[0] = 2.0f;
  input_data[1] = 0.0f;

  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  layer.forwarding(run_context, false);

  const uint16_t *output = run_context.getOutput(0).getData<uint16_t>();
  EXPECT_EQ(output[0], 30);
  EXPECT_EQ(output[1], 31);
  EXPECT_EQ(output[2], 32);
  EXPECT_EQ(output[3], 10);
  EXPECT_EQ(output[4], 11);
  EXPECT_EQ(output[5], 12);
}

TEST(CausalLmEmbeddingSidecarLut,
     ufixed8LayerForwardingRequantsToUint16ActivationOutput) {
  TempDir dir("nntrainer_embedding_sidecar_layer_ufixed8");
  const auto lut_path = dir.path() / "embedding.bin";
  const auto manifest_path = dir.path() / "manifest.json";

  writeBytes(lut_path, {0x21, 0x43, 0x65, 0x87});
  writeText(manifest_path, R"({
    "lut-path": "embedding.bin",
    "size": 4,
    "datatype": "ufixed8",
    "quant-param": { "scale": 0.5, "offset": -1 }
  })");

  causallm::EmbeddingLayer layer;
  layer.setProperty({"in_dim=2", "out_dim=4", "scale=2.0",
                     "quantized_lut_path=" + manifest_path.string(),
                     "output_quant_scale=0.25", "output_quant_offset=-3"});

  nntrainer::InitLayerContext init_context(
    {makeInputDim(nntrainer::TensorDim::DataType::UINT16, 1)}, {true}, false,
    "embedding_sidecar_lut", "", 0.0f, {"NCHW", "FP32", "UINT16"});

  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  inputs[0].getVariableRef().getData<float>()[0] = 1.0f;

  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  layer.forwarding(run_context, false);

  const uint16_t *output = run_context.getOutput(0).getData<uint16_t>();
  EXPECT_EQ(output[0], 19);
  EXPECT_EQ(output[1], 23);
  EXPECT_EQ(output[2], 27);
  EXPECT_EQ(output[3], 31);
}

TEST(CausalLmEmbeddingSidecarLut,
     sfixed4LayerForwardingDecodesSignedNibblesToFp32Output) {
  TempDir dir("nntrainer_embedding_sidecar_layer_sfixed4");
  const auto lut_path = dir.path() / "embedding.s4";
  const auto manifest_path = dir.path() / "manifest.json";

  writeBytes(lut_path, {0x21, 0x43, 0x87, 0xf0});
  writeText(manifest_path, R"({
    "lut-path": "embedding.s4",
    "size": 4,
    "datatype": "sfixed4",
    "quant-param": { "scale": [0.5, 2.0] }
  })");

  causallm::EmbeddingLayer layer;
  layer.setProperty({"in_dim=2", "out_dim=4", "scale=1.5",
                     "quantized_lut_path=" + manifest_path.string()});

  nntrainer::InitLayerContext init_context(
    {makeInputDim(nntrainer::TensorDim::DataType::FP32, 1)}, {true}, false,
    "embedding_sidecar_lut", "", 0.0f, {"NCHW", "FP32", "FP32"});

  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  inputs[0].getVariableRef().getData<float>()[0] = 1.0f;

  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  layer.forwarding(run_context, false);

  const float *output = run_context.getOutput(0).getData<float>();
  EXPECT_FLOAT_EQ(output[0], 21.0f);
  EXPECT_FLOAT_EQ(output[1], -24.0f);
  EXPECT_FLOAT_EQ(output[2], 0.0f);
  EXPECT_FLOAT_EQ(output[3], -3.0f);
}

TEST(CausalLmEmbeddingSidecarLut,
     rawUint16IncrementalForwardingUsesCompactStepInput) {
  TempDir dir("nntrainer_embedding_sidecar_incremental_raw_u16");
  const auto raw_path = dir.path() / "embedding.u16";
  writeU16(raw_path,
           {10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42});

  causallm::EmbeddingLayer layer;
  layer.setProperty({"in_dim=4", "out_dim=3",
                     "quantized_lut_path=" + raw_path.string()});

  nntrainer::InitLayerContext init_context(
    {makeInputDim(nntrainer::TensorDim::DataType::UINT16, 3)}, {true}, false,
    "embedding_sidecar_lut", "", 0.0f, {"NCHW", "FP32", "UINT16"});

  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  float *input_data = inputs[0].getVariableRef().getData<float>();
  input_data[0] = 3.0f;
  input_data[1] = 1.0f;
  input_data[2] = 2.0f;

  uint16_t *output_data = outputs[0].getVariableRef().getData<uint16_t>();
  for (unsigned int i = 0; i < 9; ++i)
    output_data[i] = 999;

  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  layer.incremental_forwarding(run_context, 7, 8, false);

  const uint16_t *output = run_context.getOutput(0).getData<uint16_t>();
  EXPECT_EQ(output[0], 40);
  EXPECT_EQ(output[1], 41);
  EXPECT_EQ(output[2], 42);
  EXPECT_EQ(output[3], 999);
  EXPECT_EQ(output[4], 999);
  EXPECT_EQ(output[5], 999);
  EXPECT_EQ(output[6], 999);
  EXPECT_EQ(output[7], 999);
  EXPECT_EQ(output[8], 999);
}

TEST(CausalLmEmbeddingSidecarLut, nonSidecarForwardingPreservesNoOpBehavior) {
  causallm::EmbeddingLayer layer;
  layer.setProperty({"in_dim=3", "out_dim=2"});

  nntrainer::InitLayerContext init_context(
    {makeInputDim(nntrainer::TensorDim::DataType::FP32, 1)}, {true}, false,
    "embedding_sidecar_lut", "", 0.0f, {"NCHW", "FP32", "FP32"});

  ASSERT_NO_THROW(layer.finalize(init_context));
  ASSERT_EQ(init_context.getWeightsSpec().size(), 1u);

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;

  nntrainer::Tensor weight_tensor(
    nntrainer::TensorDim({1, 1, 3, 2, nntrainer::Tformat::NCHW,
                          nntrainer::TensorDim::DataType::FP32}),
    true);
  weights.emplace_back(weight_tensor, nntrainer::Tensor(), nntrainer::Tensor(),
                       "Embedding");
  float *weight_data = weights[0].getVariableRef().getData<float>();
  for (unsigned int i = 0; i < 6; ++i)
    weight_data[i] = static_cast<float>(i + 1);

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  inputs[0].getVariableRef().getData<float>()[0] = 1.0f;
  float *output_data = outputs[0].getVariableRef().getData<float>();
  output_data[0] = -7.0f;
  output_data[1] = -8.0f;

  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  layer.forwarding(run_context, false);

  EXPECT_FLOAT_EQ(run_context.getOutput(0).getData<float>()[0], -7.0f);
  EXPECT_FLOAT_EQ(run_context.getOutput(0).getData<float>()[1], -8.0f);
}
