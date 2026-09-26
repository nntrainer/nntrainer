// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_sigmoid_gate_ops.cpp
 * @date   12 September 2026
 * @brief  ComputeOps::sigmoid_glu / ::sigmoid_add -- the closed-form contract
 *         every backend table is measured against.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details No golden file is involved: both ops are closed-form, so every
 * expectation below is recomputed in double from the definition
 * (sigmoid(x) = 1/(1+exp(-x))) next to one exactly representable anchor
 * (sigmoid(0) = 0.5). The row window is asserted on both sides -- the rows
 * inside it, and the sentinel left untouched outside it -- because a whole-op
 * that quietly ignores active_rows / row_offset still produces plausible
 * numbers for the window it did process. unittest_sigmoid_gate_ops_cl.cpp
 * measures the OpenCL table against the same expectations.
 */

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <compute_ops.h>
#include <context_data.h>
#include <tensor.h>

namespace {

constexpr float POISON = -123456.0f;

/** @brief attach an ops table to a tensor, the way a LayerNode does */
void attachOps(nntrainer::Tensor &t, nntrainer::ComputeOps *ops) {
  auto ct = std::make_shared<nntrainer::ContextData>();
  ct->setComputeOps(ops);
  t.setContextData(ct);
}

/**
 * @brief Fill the destination with a sentinel, so a dispatch that silently
 *        does nothing cannot pass as a correct result.
 */
void poison(nntrainer::Tensor &t) {
  for (unsigned int i = 0; i < t.size(); ++i)
    t.getData()[i] = POISON;
}

/** deterministic, sign-varied inputs with non-trivial magnitudes */
float genValue(unsigned int i) {
  return std::sin(0.41f * (float)i) * (1.0f + 0.02f * (float)(i % 13));
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

} // namespace

/**
 * @brief sigmoid(0) == 0.5 exactly, so this case needs no tolerance at all:
 *        out must be the second operand halved / offset by a half.
 */
TEST(SigmoidGateOps, ZeroGateIsExactlyAHalf) {
  const unsigned int W = 4;
  nntrainer::Tensor gate({1, 1, 1, W}, true);
  nntrainer::Tensor other({1, 1, 1, W}, true);
  nntrainer::Tensor out({1, 1, 1, W}, true);

  gate.setValue(0.0f);
  const float rhs[W] = {2.0f, -3.0f, 0.5f, 8.0f};
  for (unsigned int i = 0; i < W; ++i)
    other.getData()[i] = rhs[i];

  auto *ops = nntrainer::get_cpu_ops();
  attachOps(gate, ops);
  attachOps(other, ops);
  attachOps(out, ops);

  poison(out);
  ASSERT_NO_THROW(gate.getOps()->sigmoid_glu(gate, other, out, 1, 0));
  for (unsigned int i = 0; i < W; ++i)
    EXPECT_FLOAT_EQ(out.getData()[i], 0.5f * rhs[i]) << "element " << i;

  poison(out);
  ASSERT_NO_THROW(gate.getOps()->sigmoid_add(gate, other, out, 1, 0));
  for (unsigned int i = 0; i < W; ++i)
    EXPECT_FLOAT_EQ(out.getData()[i], 0.5f + rhs[i]) << "element " << i;
}

/**
 * @brief The two ops differ only in how the second operand is combined:
 *        multiplied for sigmoid_glu, added for sigmoid_add. Both are checked
 *        against the definition recomputed in double.
 */
TEST(SigmoidGateOps, MatchesTheClosedForm) {
  const unsigned int H = 3, W = 16;
  nntrainer::Tensor gate({1, 1, H, W}, true);
  nntrainer::Tensor other({1, 1, H, W}, true);
  nntrainer::Tensor out({1, 1, H, W}, true);

  for (unsigned int i = 0; i < gate.size(); ++i) {
    gate.getData()[i] = genValue(i);
    other.getData()[i] = genValue(i + 7) * 3.0f;
  }

  auto *ops = nntrainer::get_cpu_ops();
  attachOps(gate, ops);
  attachOps(other, ops);
  attachOps(out, ops);

  poison(out);
  ASSERT_NO_THROW(gate.getOps()->sigmoid_glu(gate, other, out, H, 0));
  for (unsigned int i = 0; i < gate.size(); ++i)
    EXPECT_NEAR(out.getData()[i],
                (float)(sigmoid(gate.getData()[i]) * other.getData()[i]), 1e-6)
      << "element " << i;

  poison(out);
  ASSERT_NO_THROW(gate.getOps()->sigmoid_add(gate, other, out, H, 0));
  for (unsigned int i = 0; i < gate.size(); ++i)
    EXPECT_NEAR(out.getData()[i],
                (float)(sigmoid(gate.getData()[i]) + other.getData()[i]), 1e-6)
      << "element " << i;
}

/**
 * @brief active_rows / row_offset select a contiguous row window. Everything
 *        outside it must still hold the sentinel: the decode path relies on
 *        exactly one row being touched per step.
 */
TEST(SigmoidGateOps, HonorsTheRowWindow) {
  const unsigned int H = 6, W = 8, OFF = 2, N = 3;
  nntrainer::Tensor gate({1, 1, H, W}, true);
  nntrainer::Tensor other({1, 1, H, W}, true);
  nntrainer::Tensor out({1, 1, H, W}, true);

  for (unsigned int i = 0; i < gate.size(); ++i) {
    gate.getData()[i] = genValue(i);
    other.getData()[i] = genValue(i + 5);
  }

  auto *ops = nntrainer::get_cpu_ops();
  attachOps(gate, ops);
  attachOps(other, ops);
  attachOps(out, ops);

  poison(out);
  ASSERT_NO_THROW(gate.getOps()->sigmoid_glu(gate, other, out, N, OFF));

  for (unsigned int r = 0; r < H; ++r) {
    for (unsigned int c = 0; c < W; ++c) {
      const size_t i = (size_t)r * W + c;
      if (r >= OFF && r < OFF + N)
        EXPECT_NEAR(out.getData()[i],
                    (float)(sigmoid(gate.getData()[i]) * other.getData()[i]),
                    1e-6)
          << "row " << r << " element " << c;
      else
        EXPECT_FLOAT_EQ(out.getData()[i], POISON)
          << "row " << r << " was written outside the window";
    }
  }
}

/**
 * @brief active_rows == 0 is the empty window a skipped prefill step passes
 *        down; it must be a no-op rather than a one-row dispatch.
 */
TEST(SigmoidGateOps, EmptyWindowWritesNothing) {
  const unsigned int W = 8;
  nntrainer::Tensor gate({1, 1, 1, W}, true);
  nntrainer::Tensor other({1, 1, 1, W}, true);
  nntrainer::Tensor out({1, 1, 1, W}, true);
  gate.setValue(0.0f);
  other.setValue(1.0f);

  auto *ops = nntrainer::get_cpu_ops();
  attachOps(gate, ops);
  attachOps(other, ops);
  attachOps(out, ops);

  poison(out);
  ASSERT_NO_THROW(gate.getOps()->sigmoid_glu(gate, other, out, 0, 0));
  ASSERT_NO_THROW(gate.getOps()->sigmoid_add(gate, other, out, 0, 0));
  for (unsigned int i = 0; i < W; ++i)
    EXPECT_FLOAT_EQ(out.getData()[i], POISON) << "element " << i;
}

/**
 * @brief The reference tables index the second operand and the destination
 *        with the first one's shape and dtype, so a disagreeing operand has to
 *        be rejected rather than reinterpreted.
 */
TEST(SigmoidGateOps, RejectsMismatchedShapes) {
  nntrainer::Tensor gate({1, 1, 2, 8}, true);
  nntrainer::Tensor narrow({1, 1, 2, 4}, true);
  nntrainer::Tensor out({1, 1, 2, 8}, true);
  gate.setValue(0.0f);
  narrow.setValue(1.0f);
  out.setValue(0.0f);

  auto *ops = nntrainer::get_cpu_ops();
  attachOps(gate, ops);
  attachOps(narrow, ops);
  attachOps(out, ops);

  EXPECT_THROW(gate.getOps()->sigmoid_glu(gate, narrow, out, 2, 0),
               std::invalid_argument);
  EXPECT_THROW(gate.getOps()->sigmoid_add(gate, narrow, out, 2, 0),
               std::invalid_argument);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
