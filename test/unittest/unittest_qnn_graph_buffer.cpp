// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * @file   unittest_qnn_graph_buffer.cpp
 * @date   30 Sep 2026
 * @brief  Unit tests for the QNN graph layer buffer collection
 * (qnn_graph_buffer.h), which QNNGraph::forwarding() uses; runs without the
 * QNN SDK
 * @see    https://github.com/nntrainer/nntrainer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug    No known bugs
 */
#include <gtest/gtest.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <qnn_graph_buffer.h>
#include <tensor.h>

using nntrainer::QnnBufferPtr;
using nntrainer::Tensor;
using nntrainer::TensorDim;
using Tdatatype = ml::train::TensorDim::DataType;
using Tformat = ml::train::TensorDim::Format;

namespace {

/**
 * @brief Allocate a small named tensor of the given data type
 */
Tensor makeTensor(Tdatatype type, const std::string &name = "t") {
  return Tensor(TensorDim(1, 1, 1, 4, Tformat::NCHW, type), true,
                nntrainer::Initializer::NONE, name);
}

/**
 * @brief Call resetQnnBuffers() on a list of tensors, the way
 * QNNGraph::forwarding() calls it on the layer inputs or outputs
 */
void reset(std::vector<QnnBufferPtr> &buffers, std::vector<Tensor> &tensors) {
  nntrainer::resetQnnBuffers(
    buffers, static_cast<unsigned int>(tensors.size()),
    [&](unsigned int i) -> Tensor & { return tensors[i]; });
}

/**
 * @brief Check that @a buffers holds exactly the FP32 data pointers of
 * @a tensors, in order
 */
void expectFp32Buffers(const std::vector<QnnBufferPtr> &buffers,
                       const std::vector<Tensor> &tensors) {
  ASSERT_EQ(buffers.size(), tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    ASSERT_TRUE(std::holds_alternative<float *>(buffers[i]));
    EXPECT_EQ(std::get<float *>(buffers[i]), tensors[i].getData<float>());
  }
}

} // namespace

/**
 * @brief FP32 tensors are handed over as float *
 */
TEST(QnnGraphBuffer, get_buffer_fp32) {
  Tensor t = makeTensor(Tdatatype::FP32);
  QnnBufferPtr p = nntrainer::getQnnBufferPtr(t);
  ASSERT_TRUE(std::holds_alternative<float *>(p));
  EXPECT_EQ(std::get<float *>(p), t.getData<float>());
}

/**
 * @brief UINT8 tensors are handed over as uint8_t *
 */
TEST(QnnGraphBuffer, get_buffer_uint8) {
  Tensor t = makeTensor(Tdatatype::UINT8);
  QnnBufferPtr p = nntrainer::getQnnBufferPtr(t);
  ASSERT_TRUE(std::holds_alternative<uint8_t *>(p));
  EXPECT_EQ(std::get<uint8_t *>(p), t.getData<uint8_t>());
}

/**
 * @brief UINT4 tensors are handed over as uint8_t *
 */
TEST(QnnGraphBuffer, get_buffer_uint4) {
  Tensor t = makeTensor(Tdatatype::UINT4);
  QnnBufferPtr p = nntrainer::getQnnBufferPtr(t);
  ASSERT_TRUE(std::holds_alternative<uint8_t *>(p));
  EXPECT_EQ(std::get<uint8_t *>(p), t.getData<uint8_t>());
}

/**
 * @brief UINT16 tensors are handed over as uint16_t *
 */
TEST(QnnGraphBuffer, get_buffer_uint16) {
  Tensor t = makeTensor(Tdatatype::UINT16);
  QnnBufferPtr p = nntrainer::getQnnBufferPtr(t);
  ASSERT_TRUE(std::holds_alternative<uint16_t *>(p));
  EXPECT_EQ(std::get<uint16_t *>(p), t.getData<uint16_t>());
}

/**
 * @brief UINT32 is not supported; the error names the tensor
 */
TEST(QnnGraphBuffer, get_buffer_uint32_n) {
  Tensor t = makeTensor(Tdatatype::UINT32, "bad_input");
  try {
    nntrainer::getQnnBufferPtr(t);
    FAIL() << "std::invalid_argument expected";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("bad_input"), std::string::npos);
  }
}

/**
 * @brief QINT8 is not supported
 */
TEST(QnnGraphBuffer, get_buffer_qint8_n) {
  Tensor t = makeTensor(Tdatatype::QINT8);
  EXPECT_THROW(nntrainer::getQnnBufferPtr(t), std::invalid_argument);
}

/**
 * @brief A second forward with re-allocated input and output tensors hands
 * over the new addresses, and repeated forwards do not accumulate entries
 */
TEST(QnnGraphBuffer, reset_uses_current_addresses) {
  std::vector<QnnBufferPtr> in_buffers, out_buffers;

  std::vector<Tensor> in1 = {makeTensor(Tdatatype::FP32),
                             makeTensor(Tdatatype::FP32)};
  std::vector<Tensor> out1 = {makeTensor(Tdatatype::FP32)};
  reset(in_buffers, in1);
  reset(out_buffers, out1);
  expectFp32Buffers(in_buffers, in1);
  expectFp32Buffers(out_buffers, out1);

  /** the first set stays alive, so the new tensors get different addresses */
  std::vector<Tensor> in2 = {makeTensor(Tdatatype::FP32),
                             makeTensor(Tdatatype::FP32)};
  std::vector<Tensor> out2 = {makeTensor(Tdatatype::FP32)};
  for (int iter = 0; iter < 3; ++iter) {
    reset(in_buffers, in2);
    reset(out_buffers, out2);
    expectFp32Buffers(in_buffers, in2);
    expectFp32Buffers(out_buffers, out2);
  }
  EXPECT_NE(std::get<float *>(in_buffers[0]), in1[0].getData<float>());
  EXPECT_NE(std::get<float *>(out_buffers[0]), out1[0].getData<float>());
}

/**
 * @brief Mixed supported data types keep their order and pointer types
 */
TEST(QnnGraphBuffer, reset_mixed_types) {
  std::vector<QnnBufferPtr> buffers;
  std::vector<Tensor> tensors = {makeTensor(Tdatatype::UINT8),
                                 makeTensor(Tdatatype::UINT16),
                                 makeTensor(Tdatatype::FP32)};
  reset(buffers, tensors);
  ASSERT_EQ(buffers.size(), 3u);
  EXPECT_EQ(std::get<uint8_t *>(buffers[0]), tensors[0].getData<uint8_t>());
  EXPECT_EQ(std::get<uint16_t *>(buffers[1]), tensors[1].getData<uint16_t>());
  EXPECT_EQ(std::get<float *>(buffers[2]), tensors[2].getData<float>());
}

/**
 * @brief Resetting with no tensors drops the previous entries
 */
TEST(QnnGraphBuffer, reset_empty_clears) {
  std::vector<QnnBufferPtr> buffers;
  std::vector<Tensor> tensors = {makeTensor(Tdatatype::FP32)};
  reset(buffers, tensors);
  ASSERT_EQ(buffers.size(), 1u);

  std::vector<Tensor> none;
  reset(buffers, none);
  EXPECT_TRUE(buffers.empty());
}

/**
 * @brief An unsupported input after a supported one throws, and the next
 * reset with valid tensors discards the partial state
 */
TEST(QnnGraphBuffer, reset_unsupported_input_n) {
  std::vector<QnnBufferPtr> in_buffers;
  std::vector<Tensor> bad = {makeTensor(Tdatatype::FP32),
                             makeTensor(Tdatatype::UINT32)};
  EXPECT_THROW(reset(in_buffers, bad), std::invalid_argument);

  std::vector<Tensor> good = {makeTensor(Tdatatype::FP32)};
  reset(in_buffers, good);
  expectFp32Buffers(in_buffers, good);
}

/**
 * @brief Same as reset_unsupported_input_n on the output list, starting from
 * a list filled by an earlier successful forward
 */
TEST(QnnGraphBuffer, reset_unsupported_output_n) {
  std::vector<QnnBufferPtr> out_buffers;
  std::vector<Tensor> first = {makeTensor(Tdatatype::FP32),
                               makeTensor(Tdatatype::FP32)};
  reset(out_buffers, first);
  ASSERT_EQ(out_buffers.size(), 2u);

  std::vector<Tensor> bad = {makeTensor(Tdatatype::UINT16),
                             makeTensor(Tdatatype::QINT8)};
  EXPECT_THROW(reset(out_buffers, bad), std::invalid_argument);

  std::vector<Tensor> good = {makeTensor(Tdatatype::FP32)};
  reset(out_buffers, good);
  expectFp32Buffers(out_buffers, good);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
