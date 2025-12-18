// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_tensor_types.cpp
 * @date 11 December 2025
 * @brief Unit tests for various tensor types (CharTensor, ShortTensor,
 * UIntTensor, Int4QTensor)
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <sstream>

#include <char_tensor.h>
#include <short_tensor.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace {
constexpr float TOLERANCE = 1e-5f;
} // namespace

//==============================================================================
// CharTensor Tests (int8_t based tensor)
//==============================================================================

TEST(CharTensor, constructor_default) {
  nntrainer::CharTensor t("test", nntrainer::Tformat::NCHW);
  EXPECT_EQ(t.getName(), "test");
}

TEST(CharTensor, constructor_with_dim) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::CharTensor t(dim, true);

  EXPECT_EQ(t.batch(), 1u);
  EXPECT_EQ(t.channel(), 1u);
  EXPECT_EQ(t.height(), 2u);
  EXPECT_EQ(t.width(), 3u);
}

TEST(CharTensor, setZero) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::CharTensor t(dim, true);

  t.setZero();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 0);
  }
}

TEST(CharTensor, setValue_getValue) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::CharTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, 20.0f);
  t.setValue(0, 0, 1, 0, 30.0f);
  t.setValue(0, 0, 1, 1, 40.0f);

  EXPECT_EQ(t.getValue(0, 0, 0, 0), 10);
  EXPECT_EQ(t.getValue(0, 0, 0, 1), 20);
  EXPECT_EQ(t.getValue(0, 0, 1, 0), 30);
  EXPECT_EQ(t.getValue(0, 0, 1, 1), 40);
}

TEST(CharTensor, initialize_zeros) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::CharTensor t(dim, true, nntrainer::Initializer::ZEROS);
  t.initialize();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 0);
  }
}

TEST(CharTensor, initialize_ones) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::CharTensor t(dim, true, nntrainer::Initializer::ONES);
  t.initialize();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 1);
  }
}

TEST(CharTensor, argmax) {
  nntrainer::TensorDim dim(2, 1, 1, 4);
  nntrainer::CharTensor t(dim, true);

  // Batch 0: [10, 50, 30, 20] -> max at index 1
  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, 50.0f);
  t.setValue(0, 0, 0, 2, 30.0f);
  t.setValue(0, 0, 0, 3, 20.0f);

  // Batch 1: [5, 15, 25, 100] -> max at index 3
  t.setValue(1, 0, 0, 0, 5.0f);
  t.setValue(1, 0, 0, 1, 15.0f);
  t.setValue(1, 0, 0, 2, 25.0f);
  t.setValue(1, 0, 0, 3, 100.0f);

  auto result = t.argmax();
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 1u);
  EXPECT_EQ(result[1], 3u);
}

TEST(CharTensor, argmin) {
  nntrainer::TensorDim dim(2, 1, 1, 4);
  nntrainer::CharTensor t(dim, true);

  // Batch 0: [10, 50, 5, 20] -> min at index 2
  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, 50.0f);
  t.setValue(0, 0, 0, 2, 5.0f);
  t.setValue(0, 0, 0, 3, 20.0f);

  // Batch 1: [1, 15, 25, 100] -> min at index 0
  t.setValue(1, 0, 0, 0, 1.0f);
  t.setValue(1, 0, 0, 1, 15.0f);
  t.setValue(1, 0, 0, 2, 25.0f);
  t.setValue(1, 0, 0, 3, 100.0f);

  auto result = t.argmin();
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 2u);
  EXPECT_EQ(result[1], 0u);
}

TEST(CharTensor, maxValue) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::CharTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, 127.0f);
  t.setValue(0, 0, 0, 2, 30.0f);
  t.setValue(0, 0, 1, 0, -50.0f);
  t.setValue(0, 0, 1, 1, -128.0f);
  t.setValue(0, 0, 1, 2, 0.0f);

  EXPECT_EQ(t.maxValue(), 127.0f);
}

TEST(CharTensor, minValue) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::CharTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, 127.0f);
  t.setValue(0, 0, 0, 2, 30.0f);
  t.setValue(0, 0, 1, 0, -50.0f);
  t.setValue(0, 0, 1, 1, -128.0f);
  t.setValue(0, 0, 1, 2, 0.0f);

  EXPECT_EQ(t.minValue(), -128.0f);
}

TEST(CharTensor, max_abs) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::CharTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, -120.0f); // abs = 120 (max)
  t.setValue(0, 0, 1, 0, 50.0f);
  t.setValue(0, 0, 1, 1, -30.0f);

  EXPECT_EQ(t.max_abs(), 120.0f);
}

TEST(CharTensor, operator_equal) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::CharTensor t1(dim, true);
  nntrainer::CharTensor t2(dim, true);

  t1.setValue(0, 0, 0, 0, 10.0f);
  t1.setValue(0, 0, 0, 1, 20.0f);
  t1.setValue(0, 0, 1, 0, 30.0f);
  t1.setValue(0, 0, 1, 1, 40.0f);

  t2.setValue(0, 0, 0, 0, 10.0f);
  t2.setValue(0, 0, 0, 1, 20.0f);
  t2.setValue(0, 0, 1, 0, 30.0f);
  t2.setValue(0, 0, 1, 1, 40.0f);

  EXPECT_TRUE(t1 == t2);
}

TEST(CharTensor, operator_not_equal) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::CharTensor t1(dim, true);
  nntrainer::CharTensor t2(dim, true);

  t1.setValue(0, 0, 0, 0, 10.0f);
  t2.setValue(0, 0, 0, 0, 99.0f);

  EXPECT_FALSE(t1 == t2);
}

TEST(CharTensor, addValue) {
  nntrainer::TensorDim dim(1, 1, 1, 1);
  nntrainer::CharTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 10.0f);
  t.addValue(0, 0, 0, 0, 5.0f, 1.0f);

  EXPECT_EQ(t.getValue(0, 0, 0, 0), 15);
}

TEST(CharTensor, getMemoryBytes) {
  nntrainer::TensorDim dim(1, 1, 4, 4); // 16 elements
  nntrainer::CharTensor t(dim, true);

  // CharTensor uses int8_t (1 byte per element) plus scale data
  // Memory size varies based on quantization scheme, just verify it's non-zero
  EXPECT_GT(t.getMemoryBytes(), 0u);
}

TEST(CharTensor, print) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::CharTensor t(dim, true, nntrainer::Initializer::ZEROS,
                          "test_tensor");
  t.initialize();

  std::ostringstream oss;
  t.print(oss);

  std::string output = oss.str();
  // Just verify print doesn't throw and produces some output
  EXPECT_FALSE(output.empty());
}

//==============================================================================
// ShortTensor Tests (int16_t based tensor)
//==============================================================================

TEST(ShortTensor, constructor_default) {
  nntrainer::ShortTensor t("test", nntrainer::Tformat::NCHW);
  EXPECT_EQ(t.getName(), "test");
}

TEST(ShortTensor, constructor_with_dim) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::ShortTensor t(dim, true);

  EXPECT_EQ(t.batch(), 1u);
  EXPECT_EQ(t.channel(), 1u);
  EXPECT_EQ(t.height(), 2u);
  EXPECT_EQ(t.width(), 3u);
}

TEST(ShortTensor, setZero) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::ShortTensor t(dim, true);

  t.setZero();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 0);
  }
}

TEST(ShortTensor, setValue_getValue) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::ShortTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 1000.0f);
  t.setValue(0, 0, 0, 1, 2000.0f);
  t.setValue(0, 0, 1, 0, -3000.0f);
  t.setValue(0, 0, 1, 1, 32767.0f);

  EXPECT_EQ(t.getValue(0, 0, 0, 0), 1000);
  EXPECT_EQ(t.getValue(0, 0, 0, 1), 2000);
  EXPECT_EQ(t.getValue(0, 0, 1, 0), -3000);
  EXPECT_EQ(t.getValue(0, 0, 1, 1), 32767);
}

TEST(ShortTensor, initialize_zeros) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::ShortTensor t(dim, true, nntrainer::Initializer::ZEROS);
  t.initialize();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 0);
  }
}

TEST(ShortTensor, initialize_ones) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::ShortTensor t(dim, true, nntrainer::Initializer::ONES);
  t.initialize();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 1);
  }
}

TEST(ShortTensor, argmax) {
  nntrainer::TensorDim dim(2, 1, 1, 4);
  nntrainer::ShortTensor t(dim, true);

  // Batch 0: [100, 5000, 300, 200] -> max at index 1
  t.setValue(0, 0, 0, 0, 100.0f);
  t.setValue(0, 0, 0, 1, 5000.0f);
  t.setValue(0, 0, 0, 2, 300.0f);
  t.setValue(0, 0, 0, 3, 200.0f);

  // Batch 1: [50, 150, 250, 10000] -> max at index 3
  t.setValue(1, 0, 0, 0, 50.0f);
  t.setValue(1, 0, 0, 1, 150.0f);
  t.setValue(1, 0, 0, 2, 250.0f);
  t.setValue(1, 0, 0, 3, 10000.0f);

  auto result = t.argmax();
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 1u);
  EXPECT_EQ(result[1], 3u);
}

TEST(ShortTensor, argmin) {
  nntrainer::TensorDim dim(2, 1, 1, 4);
  nntrainer::ShortTensor t(dim, true);

  // Batch 0: [100, 5000, 50, 200] -> min at index 2
  t.setValue(0, 0, 0, 0, 100.0f);
  t.setValue(0, 0, 0, 1, 5000.0f);
  t.setValue(0, 0, 0, 2, 50.0f);
  t.setValue(0, 0, 0, 3, 200.0f);

  // Batch 1: [10, 150, 250, 10000] -> min at index 0
  t.setValue(1, 0, 0, 0, 10.0f);
  t.setValue(1, 0, 0, 1, 150.0f);
  t.setValue(1, 0, 0, 2, 250.0f);
  t.setValue(1, 0, 0, 3, 10000.0f);

  auto result = t.argmin();
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 2u);
  EXPECT_EQ(result[1], 0u);
}

TEST(ShortTensor, maxValue) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::ShortTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 100.0f);
  t.setValue(0, 0, 0, 1, 32767.0f);
  t.setValue(0, 0, 0, 2, 300.0f);
  t.setValue(0, 0, 1, 0, -5000.0f);
  t.setValue(0, 0, 1, 1, -32768.0f);
  t.setValue(0, 0, 1, 2, 0.0f);

  EXPECT_EQ(t.maxValue(), 32767.0f);
}

TEST(ShortTensor, minValue) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::ShortTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 100.0f);
  t.setValue(0, 0, 0, 1, 32767.0f);
  t.setValue(0, 0, 0, 2, 300.0f);
  t.setValue(0, 0, 1, 0, -5000.0f);
  t.setValue(0, 0, 1, 1, -32768.0f);
  t.setValue(0, 0, 1, 2, 0.0f);

  EXPECT_EQ(t.minValue(), -32768.0f);
}

TEST(ShortTensor, max_abs) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::ShortTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 100.0f);
  t.setValue(0, 0, 0, 1, -12000.0f); // abs = 12000 (max)
  t.setValue(0, 0, 1, 0, 5000.0f);
  t.setValue(0, 0, 1, 1, -3000.0f);

  EXPECT_EQ(t.max_abs(), 12000.0f);
}

TEST(ShortTensor, operator_equal) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::ShortTensor t1(dim, true);
  nntrainer::ShortTensor t2(dim, true);

  t1.setValue(0, 0, 0, 0, 1000.0f);
  t1.setValue(0, 0, 0, 1, 2000.0f);
  t1.setValue(0, 0, 1, 0, 3000.0f);
  t1.setValue(0, 0, 1, 1, 4000.0f);

  t2.setValue(0, 0, 0, 0, 1000.0f);
  t2.setValue(0, 0, 0, 1, 2000.0f);
  t2.setValue(0, 0, 1, 0, 3000.0f);
  t2.setValue(0, 0, 1, 1, 4000.0f);

  EXPECT_TRUE(t1 == t2);
}

TEST(ShortTensor, addValue) {
  nntrainer::TensorDim dim(1, 1, 1, 1);
  nntrainer::ShortTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 1000.0f);
  t.addValue(0, 0, 0, 0, 500.0f, 1.0f);

  EXPECT_EQ(t.getValue(0, 0, 0, 0), 1500);
}

TEST(ShortTensor, getMemoryBytes) {
  nntrainer::TensorDim dim(1, 1, 4, 4); // 16 elements
  nntrainer::ShortTensor t(dim, true);

  // ShortTensor uses int16_t (2 bytes per element) plus scale data
  // Memory size varies based on quantization scheme, just verify it's non-zero
  EXPECT_GT(t.getMemoryBytes(), 0u);
}

TEST(ShortTensor, print) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::ShortTensor t(dim, true, nntrainer::Initializer::ZEROS,
                           "test_short");
  t.initialize();

  std::ostringstream oss;
  t.print(oss);

  std::string output = oss.str();
  // Just verify print doesn't throw and produces some output
  EXPECT_FALSE(output.empty());
}

//==============================================================================
// Main function
//==============================================================================

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
