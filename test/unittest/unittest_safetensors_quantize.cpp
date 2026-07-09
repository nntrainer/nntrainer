// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_safetensors_quantize.cpp
 * @date   26 May 2026
 * @brief  Unit tests for quantized weight support in the safetensors format.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <layer.h>
#include <model.h>
#include <neuralnet.h>
#include <optimizer.h>
#include <safetensors_util.h>
#include <tensor_dim.h>

namespace st = nntrainer::safetensors;
using DataType = ml::train::TensorDim::DataType;
using ModelFormat = ml::train::ModelFormat;

// ===========================================================================
// safetensors_util: dtype mapping
// ===========================================================================

TEST(SafetensorsUtil, quantized_dtype_maps_to_u8_p) {
  EXPECT_STREQ(st::dtypeToString(DataType::Q4_0), "U8");
  EXPECT_STREQ(st::dtypeToString(DataType::Q4_K), "U8");
  EXPECT_STREQ(st::dtypeToString(DataType::Q6_K), "U8");
  EXPECT_STREQ(st::dtypeToString(DataType::FP32), "F32");
  EXPECT_STREQ(st::dtypeToString(DataType::FP16), "F16");
}

TEST(SafetensorsUtil, is_quantized_p) {
  EXPECT_TRUE(st::isQuantized(DataType::Q4_0));
  EXPECT_TRUE(st::isQuantized(DataType::Q4_K));
  EXPECT_TRUE(st::isQuantized(DataType::Q6_K));
  EXPECT_FALSE(st::isQuantized(DataType::FP32));
  EXPECT_FALSE(st::isQuantized(DataType::FP16));
}

TEST(SafetensorsUtil, nntr_dtype_name_round_trip_p) {
  for (auto dt : {DataType::FP32, DataType::FP16, DataType::Q4_0,
                  DataType::Q4_K, DataType::Q6_K}) {
    const std::string name = st::nntrDtypeName(dt);
    EXPECT_EQ(st::nntrDtypeFromName(name), dt);
  }
  // Accept standard safetensors aliases too.
  EXPECT_EQ(st::nntrDtypeFromName("F32"), DataType::FP32);
  EXPECT_EQ(st::nntrDtypeFromName("U8"), DataType::UINT8);
}

TEST(SafetensorsUtil, nntr_dtype_from_unknown_name_n) {
  EXPECT_THROW(st::nntrDtypeFromName("NOT_A_TYPE"), std::invalid_argument);
}

// ===========================================================================
// safetensors_util: header build / parse round-trip
// ===========================================================================

TEST(SafetensorsUtil, build_parse_round_trip_p) {
  std::vector<st::TensorEntry> entries;

  st::TensorEntry fp32;
  fp32.name = "dense:weight";
  fp32.dtype = "F32";
  fp32.shape = {1, 1, 4, 8};
  fp32.offset_start = 0;
  fp32.offset_end = 128;
  entries.push_back(fp32);

  st::TensorEntry quant;
  quant.name = "fc:weight";
  quant.dtype = "U8";
  quant.shape = {576};
  quant.offset_start = 128;
  quant.offset_end = 704;
  quant.nntr_dtype = "Q4_0";
  quant.nntr_shape = {1, 1, 32, 32};
  entries.push_back(quant);

  std::map<std::string, std::string> meta = {
    {"nntr_format", "nntr-safetensors-v1"}};
  const std::string header = st::buildHeader(entries, meta);

  // Offsets-only parser still works (used by the weight loader).
  auto offsets = st::parseHeader(header);
  ASSERT_EQ(offsets.count("dense:weight"), 1u);
  ASSERT_EQ(offsets.count("fc:weight"), 1u);
  EXPECT_EQ(offsets["dense:weight"].first, 0u);
  EXPECT_EQ(offsets["dense:weight"].second, 128u);
  EXPECT_EQ(offsets["fc:weight"].first, 128u);
  EXPECT_EQ(offsets["fc:weight"].second, 576u);

  // Full parser recovers the extension fields.
  auto parsed = st::parseHeaderEntries(header);
  ASSERT_EQ(parsed.size(), 2u);

  const st::TensorEntry *q = nullptr;
  for (const auto &e : parsed)
    if (e.name == "fc:weight")
      q = &e;
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(q->dtype, "U8");
  EXPECT_EQ(q->nntr_dtype, "Q4_0");
  ASSERT_EQ(q->nntr_shape.size(), 4u);
  EXPECT_EQ(q->nntr_shape[2], 32u);
  EXPECT_EQ(q->nntr_shape[3], 32u);
  EXPECT_EQ(q->offset_end - q->offset_start, 576u);

  // Metadata round-trips.
  auto md = st::parseMetadata(header);
  EXPECT_EQ(md["nntr_format"], "nntr-safetensors-v1");
}

TEST(SafetensorsUtil, inspect_reports_quant_type_p) {
  std::vector<st::TensorEntry> entries;
  st::TensorEntry quant;
  quant.name = "fc:weight";
  quant.dtype = "U8";
  quant.shape = {576};
  quant.offset_start = 0;
  quant.offset_end = 576;
  quant.nntr_dtype = "Q4_0";
  quant.nntr_shape = {1, 1, 32, 32};
  entries.push_back(quant);

  const std::string header =
    st::buildHeader(entries, {{"nntr_format", "nntr-safetensors-v1"}});
  const std::string report = st::inspect(header);

  EXPECT_NE(report.find("Q4_0"), std::string::npos);
  EXPECT_NE(report.find("fc:weight"), std::string::npos);
  EXPECT_NE(report.find("nntr_format"), std::string::npos);
}

// ===========================================================================
// End-to-end: BIN and safetensors produce byte-identical quantized payloads
// ===========================================================================

static std::unique_ptr<nntrainer::NeuralNetwork>
createFcNN(unsigned int input_width, unsigned int units) {
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

static std::vector<char> readFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

/**
 * @brief A Q4_0 weight stored in safetensors must be byte-identical to the
 *        same weight stored in the BIN format, and carry U8 + nntr_dtype tags.
 */
TEST(SafetensorsQuant, q4_0_payload_matches_bin_p) {
  const unsigned int W = 32; // weight dim (1,1,W,U); W,U % 32 == 0
  const unsigned int U = 32;

  // Save the *same* initialized model in both formats with the same dtype map
  // so the only possible difference is the serialization path.
  std::map<std::string, DataType> dtype_map = {{"dense", DataType::Q4_0}};

  auto nn = createFcNN(W, U);
  const std::string bin_path = "st_quant_test.bin";
  ASSERT_NO_THROW(nn->save(bin_path, ModelFormat::MODEL_FORMAT_BIN,
                           DataType::NONE, dtype_map));

  const std::string st_path = "st_quant_test.safetensors";
  ASSERT_NO_THROW(nn->save(st_path, ModelFormat::MODEL_FORMAT_SAFETENSORS,
                           DataType::NONE, dtype_map));

  // Parse the safetensors header.
  std::ifstream stf(st_path, std::ios::binary);
  ASSERT_TRUE(stf.is_open());
  uint64_t header_size = 0;
  stf.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
  std::string header_json(header_size, '\0');
  stf.read(header_json.data(), static_cast<std::streamsize>(header_size));

  auto entries = st::parseHeaderEntries(header_json);
  const st::TensorEntry *q = nullptr;
  for (const auto &e : entries)
    if (e.nntr_dtype == "Q4_0")
      q = &e;
  ASSERT_NE(q, nullptr) << "quantized weight entry not found in header";
  EXPECT_EQ(q->dtype, "U8");
  ASSERT_EQ(q->nntr_shape.size(), 4u);
  EXPECT_EQ(q->nntr_shape[2], W);
  EXPECT_EQ(q->nntr_shape[3], U);

  // Expected Q4_0 byte size: (H*W)/32 * 18.
  const size_t expected_bytes = static_cast<size_t>(W) * U / 32 * 18;
  EXPECT_EQ(q->offset_end - q->offset_start, expected_bytes);

  // Extract the quantized payload from the safetensors data section.
  const size_t data_base = sizeof(uint64_t) + header_size;
  std::vector<char> st_bytes = readFile(st_path);
  ASSERT_GE(st_bytes.size(), data_base + q->offset_end);
  std::vector<char> st_weight(st_bytes.begin() + data_base + q->offset_start,
                              st_bytes.begin() + data_base + q->offset_end);

  // The BIN file begins with the (graph-order first) quantized weight.
  std::vector<char> bin_bytes = readFile(bin_path);
  ASSERT_GE(bin_bytes.size(), expected_bytes);
  std::vector<char> bin_weight(bin_bytes.begin(),
                               bin_bytes.begin() + expected_bytes);

  EXPECT_EQ(st_weight, bin_weight)
    << "safetensors quantized payload differs from BIN payload";

  remove(bin_path.c_str());
  remove(st_path.c_str());
}

/**
 * @brief Saving an FP32 model to safetensors keeps standard F32 tags (no
 *        nntr extension fields) so the file stays a plain safetensors file.
 */
TEST(SafetensorsQuant, fp32_save_has_no_nntr_extension_p) {
  auto nn = createFcNN(8, 16);
  const std::string st_path = "st_fp32_test.safetensors";
  ASSERT_NO_THROW(
    nn->save(st_path, ModelFormat::MODEL_FORMAT_SAFETENSORS, DataType::NONE));

  std::ifstream stf(st_path, std::ios::binary);
  ASSERT_TRUE(stf.is_open());
  uint64_t header_size = 0;
  stf.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
  std::string header_json(header_size, '\0');
  stf.read(header_json.data(), static_cast<std::streamsize>(header_size));

  auto entries = st::parseHeaderEntries(header_json);
  ASSERT_FALSE(entries.empty());
  for (const auto &e : entries) {
    EXPECT_EQ(e.dtype, "F32");
    EXPECT_TRUE(e.nntr_dtype.empty());
  }

  remove(st_path.c_str());
}

/**
 * @brief A Q4_0 safetensors records the target ISA in __metadata__ so the
 *        (ISA-specific) repack layout is identifiable. Explicit ISA::ARM must
 *        be tagged "arm" even when produced on a non-ARM host.
 */
TEST(SafetensorsQuant, q4_0_records_target_isa_p) {
  const unsigned int W = 32;
  const unsigned int U = 32;
  std::map<std::string, DataType> dtype_map = {{"dense", DataType::Q4_0}};

  auto nn = createFcNN(W, U);
  const std::string st_path = "st_isa_test.safetensors";
  ASSERT_NO_THROW(nn->save(st_path, ModelFormat::MODEL_FORMAT_SAFETENSORS,
                           DataType::NONE, dtype_map, ml::train::ISA::ARM));

  std::ifstream stf(st_path, std::ios::binary);
  ASSERT_TRUE(stf.is_open());
  uint64_t header_size = 0;
  stf.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
  std::string header_json(header_size, '\0');
  stf.read(header_json.data(), static_cast<std::streamsize>(header_size));

  auto md = st::parseMetadata(header_json);
  ASSERT_EQ(md.count("nntr_q4_0_isa"), 1u);
  EXPECT_EQ(md["nntr_q4_0_isa"], "arm");

  remove(st_path.c_str());
}

/**
 * @brief A Q8_0 safetensors records the target ISA in __metadata__: the FC
 *        weight layout is ISA-specific (x86: plain rows, ARM: q8_0x4
 *        interleaved), so the tag makes a file quantized for the wrong
 *        platform identifiable before it is loaded.
 */
TEST(SafetensorsQuant, q8_0_records_target_isa_p) {
  const unsigned int W = 32;
  const unsigned int U = 32;
  std::map<std::string, DataType> dtype_map = {{"dense", DataType::Q8_0}};

  auto nn = createFcNN(W, U);
  const std::string st_path = "st_q8_isa_test.safetensors";
  ASSERT_NO_THROW(nn->save(st_path, ModelFormat::MODEL_FORMAT_SAFETENSORS,
                           DataType::NONE, dtype_map, ml::train::ISA::ARM));

  std::ifstream stf(st_path, std::ios::binary);
  ASSERT_TRUE(stf.is_open());
  uint64_t header_size = 0;
  stf.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
  std::string header_json(header_size, '\0');
  stf.read(header_json.data(), static_cast<std::streamsize>(header_size));

  auto md = st::parseMetadata(header_json);
  ASSERT_EQ(md.count("nntr_q8_0_isa"), 1u);
  EXPECT_EQ(md["nntr_q8_0_isa"], "arm");
  EXPECT_EQ(md.count("nntr_q4_0_isa"), 0u); // no Q4_0 tensor in this file

  remove(st_path.c_str());
}

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
