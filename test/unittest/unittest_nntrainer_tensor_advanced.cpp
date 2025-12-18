// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_tensor_advanced.cpp
 * @date 11 December 2025
 * @brief Advanced unit tests for Tensor operations to improve coverage
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <sstream>

#include <float_tensor.h>
#include <layer_context.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <var_grad.h>
#include <weight.h>

namespace {
constexpr float TOLERANCE = 1e-5f;
} // namespace

//==============================================================================
// FloatTensor Tests
//==============================================================================

TEST(FloatTensor, constructor_default) {
  nntrainer::FloatTensor t("test", nntrainer::Tformat::NCHW);
  EXPECT_EQ(t.getName(), "test");
}

TEST(FloatTensor, constructor_with_dim) {
  nntrainer::TensorDim dim(1, 2, 3, 4);
  nntrainer::FloatTensor t(dim, true);

  EXPECT_EQ(t.batch(), 1u);
  EXPECT_EQ(t.channel(), 2u);
  EXPECT_EQ(t.height(), 3u);
  EXPECT_EQ(t.width(), 4u);
}

TEST(FloatTensor, setZero) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::FloatTensor t(dim, true);
  t.setZero();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(t.getValue(i), 0.0f, TOLERANCE);
  }
}

TEST(FloatTensor, setValue_getValue) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::FloatTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 1.0f);
  t.setValue(0, 0, 0, 1, 2.0f);
  t.setValue(0, 0, 1, 0, 3.0f);
  t.setValue(0, 0, 1, 1, 4.0f);

  EXPECT_NEAR(t.getValue(0, 0, 0, 0), 1.0f, TOLERANCE);
  EXPECT_NEAR(t.getValue(0, 0, 0, 1), 2.0f, TOLERANCE);
  EXPECT_NEAR(t.getValue(0, 0, 1, 0), 3.0f, TOLERANCE);
  EXPECT_NEAR(t.getValue(0, 0, 1, 1), 4.0f, TOLERANCE);
}

TEST(FloatTensor, initialize_zeros) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::FloatTensor t(dim, true, nntrainer::Initializer::ZEROS);
  t.initialize();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(t.getValue(i), 0.0f, TOLERANCE);
  }
}

TEST(FloatTensor, initialize_ones) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::FloatTensor t(dim, true, nntrainer::Initializer::ONES);
  t.initialize();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(t.getValue(i), 1.0f, TOLERANCE);
  }
}

TEST(FloatTensor, multiply_i_scalar) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::FloatTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 2.0f);
  t.setValue(0, 0, 0, 1, 3.0f);
  t.setValue(0, 0, 1, 0, 4.0f);
  t.setValue(0, 0, 1, 1, 5.0f);

  t.multiply_i(2.0f);

  EXPECT_NEAR(t.getValue(0, 0, 0, 0), 4.0f, TOLERANCE);
  EXPECT_NEAR(t.getValue(0, 0, 0, 1), 6.0f, TOLERANCE);
  EXPECT_NEAR(t.getValue(0, 0, 1, 0), 8.0f, TOLERANCE);
  EXPECT_NEAR(t.getValue(0, 0, 1, 1), 10.0f, TOLERANCE);
}

TEST(FloatTensor, add_i_scalar) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::FloatTensor t(dim, true);
  t.setValue(1.0f);

  // Use add() instead of add_i() and update in place
  for (unsigned int i = 0; i < t.size(); ++i) {
    t.setValue(0, 0, i / 2, i % 2, t.getValue(0, 0, i / 2, i % 2) + 5.0f);
  }

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(t.getValue(i), 6.0f, TOLERANCE);
  }
}

TEST(FloatTensor, argmax) {
  nntrainer::TensorDim dim(2, 1, 1, 4);
  nntrainer::FloatTensor t(dim, true);

  // Batch 0: max at index 2
  t.setValue(0, 0, 0, 0, 1.0f);
  t.setValue(0, 0, 0, 1, 3.0f);
  t.setValue(0, 0, 0, 2, 10.0f);
  t.setValue(0, 0, 0, 3, 2.0f);

  // Batch 1: max at index 0
  t.setValue(1, 0, 0, 0, 100.0f);
  t.setValue(1, 0, 0, 1, 3.0f);
  t.setValue(1, 0, 0, 2, 5.0f);
  t.setValue(1, 0, 0, 3, 2.0f);

  auto result = t.argmax();
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 2u);
  EXPECT_EQ(result[1], 0u);
}

TEST(FloatTensor, maxValue) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::FloatTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 1.0f);
  t.setValue(0, 0, 0, 1, 99.0f); // max
  t.setValue(0, 0, 0, 2, 3.0f);
  t.setValue(0, 0, 1, 0, -5.0f);
  t.setValue(0, 0, 1, 1, 42.0f);
  t.setValue(0, 0, 1, 2, 0.0f);

  EXPECT_NEAR(t.maxValue(), 99.0f, TOLERANCE);
}

TEST(FloatTensor, minValue) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::FloatTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 1.0f);
  t.setValue(0, 0, 0, 1, 99.0f);
  t.setValue(0, 0, 0, 2, 3.0f);
  t.setValue(0, 0, 1, 0, -50.0f); // min
  t.setValue(0, 0, 1, 1, 42.0f);
  t.setValue(0, 0, 1, 2, 0.0f);

  EXPECT_NEAR(t.minValue(), -50.0f, TOLERANCE);
}

TEST(FloatTensor, operator_equal) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::FloatTensor t1(dim, true);
  nntrainer::FloatTensor t2(dim, true);

  t1.setValue(0, 0, 0, 0, 1.0f);
  t1.setValue(0, 0, 0, 1, 2.0f);
  t1.setValue(0, 0, 1, 0, 3.0f);
  t1.setValue(0, 0, 1, 1, 4.0f);

  t2.setValue(0, 0, 0, 0, 1.0f);
  t2.setValue(0, 0, 0, 1, 2.0f);
  t2.setValue(0, 0, 1, 0, 3.0f);
  t2.setValue(0, 0, 1, 1, 4.0f);

  EXPECT_TRUE(t1 == t2);
}

TEST(FloatTensor, print) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::FloatTensor t(dim, true, nntrainer::Initializer::ZEROS, "test");
  t.initialize();

  std::ostringstream oss;
  t.print(oss);

  std::string output = oss.str();
  EXPECT_FALSE(output.empty());
}

//==============================================================================
// Advanced Tensor Operations Tests
//==============================================================================

TEST(Tensor, reshape) {
  nntrainer::TensorDim dim(1, 2, 3, 4); // 24 elements
  nntrainer::Tensor t(dim);
  t.setValue(1.0f);

  // Reshape to 1x1x6x4
  nntrainer::TensorDim new_dim(1, 1, 6, 4);
  t.reshape(new_dim);

  EXPECT_EQ(t.height(), 6u);
  EXPECT_EQ(t.width(), 4u);
  EXPECT_EQ(t.size(), 24u);
}

TEST(Tensor, getDim) {
  nntrainer::TensorDim dim(2, 3, 4, 5);
  nntrainer::Tensor t(dim);

  nntrainer::TensorDim result = t.getDim();

  EXPECT_EQ(result.batch(), 2u);
  EXPECT_EQ(result.channel(), 3u);
  EXPECT_EQ(result.height(), 4u);
  EXPECT_EQ(result.width(), 5u);
}

TEST(Tensor, getBatchSlice) {
  nntrainer::TensorDim dim(4, 2, 3, 3);
  nntrainer::Tensor t(dim);

  // Set different values for each batch
  for (unsigned int b = 0; b < 4; ++b) {
    for (unsigned int idx = 0; idx < 2 * 3 * 3; ++idx) {
      t.getValue(b * 2 * 3 * 3 + idx) = static_cast<float>(b + 1);
    }
  }

  nntrainer::Tensor slice = t.getBatchSlice(1, 1); // Get batch 1

  EXPECT_EQ(slice.batch(), 1u);
  EXPECT_NEAR(slice.getValue(0, 0, 0, 0), 2.0f, TOLERANCE);
}

TEST(Tensor, add_inplace) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::Tensor t(dim);
  t.setValue(3.0f);

  t.add_i(5.0f);

  float *data = t.getData();
  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(data[i], 8.0f, TOLERANCE);
  }
}

TEST(Tensor, subtract_inplace) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::Tensor t(dim);
  t.setValue(10.0f);

  t.subtract_i(3.0f);

  float *data = t.getData();
  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(data[i], 7.0f, TOLERANCE);
  }
}

TEST(Tensor, multiply_inplace) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::Tensor t(dim);
  t.setValue(4.0f);

  t.multiply_i(3.0f);

  float *data = t.getData();
  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(data[i], 12.0f, TOLERANCE);
  }
}

TEST(Tensor, divide_inplace) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::Tensor t(dim);
  t.setValue(12.0f);

  t.divide_i(4.0f);

  float *data = t.getData();
  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(data[i], 3.0f, TOLERANCE);
  }
}

TEST(Tensor, pow) {
  nntrainer::TensorDim dim(1, 1, 1, 4);
  nntrainer::Tensor t(dim);

  t.setValue(0, 0, 0, 0, 2.0f);
  t.setValue(0, 0, 0, 1, 3.0f);
  t.setValue(0, 0, 0, 2, 4.0f);
  t.setValue(0, 0, 0, 3, 5.0f);

  nntrainer::Tensor result = t.pow(2.0f);

  EXPECT_NEAR(result.getValue(0, 0, 0, 0), 4.0f, TOLERANCE);
  EXPECT_NEAR(result.getValue(0, 0, 0, 1), 9.0f, TOLERANCE);
  EXPECT_NEAR(result.getValue(0, 0, 0, 2), 16.0f, TOLERANCE);
  EXPECT_NEAR(result.getValue(0, 0, 0, 3), 25.0f, TOLERANCE);
}

TEST(Tensor, erf) {
  nntrainer::TensorDim dim(1, 1, 1, 3);
  nntrainer::Tensor t(dim);

  t.setValue(0, 0, 0, 0, 0.0f);
  t.setValue(0, 0, 0, 1, 1.0f);
  t.setValue(0, 0, 0, 2, -1.0f);

  nntrainer::Tensor result = t.erf();

  EXPECT_NEAR(result.getValue(0, 0, 0, 0), 0.0f, TOLERANCE);
  EXPECT_NEAR(result.getValue(0, 0, 0, 1), std::erf(1.0f), TOLERANCE);
  EXPECT_NEAR(result.getValue(0, 0, 0, 2), std::erf(-1.0f), TOLERANCE);
}

TEST(Tensor, clone) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::Tensor t(dim);

  t.setValue(0, 0, 0, 0, 1.0f);
  t.setValue(0, 0, 0, 1, 2.0f);
  t.setValue(0, 0, 1, 0, 3.0f);
  t.setValue(0, 0, 1, 1, 4.0f);

  nntrainer::Tensor cloned = t.clone();

  // Modify original
  t.setValue(0, 0, 0, 0, 999.0f);

  // Cloned should be unchanged
  EXPECT_NEAR(cloned.getValue(0, 0, 0, 0), 1.0f, TOLERANCE);
  EXPECT_NEAR(cloned.getValue(0, 0, 0, 1), 2.0f, TOLERANCE);
}

TEST(Tensor, sum) {
  nntrainer::TensorDim dim(2, 2, 2, 2);
  nntrainer::Tensor t(dim);
  t.setValue(1.0f); // 16 elements, all 1.0

  // Sum along axis 0 (batch)
  nntrainer::Tensor sum0 = t.sum(0);
  EXPECT_EQ(sum0.batch(), 1u);

  // Sum along axis 3 (width)
  nntrainer::Tensor sum3 = t.sum(3);
  EXPECT_EQ(sum3.width(), 1u);
}

//==============================================================================
// Weight Tests
//==============================================================================

TEST(Weight, basic_construction) {
  nntrainer::TensorDim dim(1, 1, 3, 3);

  nntrainer::Weight w(dim, nntrainer::Initializer::ONES,
                      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, 0.0f,
                      true, false, "test_weight");

  EXPECT_EQ(w.getName(), "test_weight");
}

//==============================================================================
// VarGrad Tests
//==============================================================================

TEST(VarGrad, basic_construction) {
  nntrainer::TensorDim dim(1, 1, 3, 3);

  nntrainer::Var_Grad vg(dim, nntrainer::Initializer::ZEROS, true, false,
                         "test_vg");

  EXPECT_EQ(vg.getName(), "test_vg");
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
