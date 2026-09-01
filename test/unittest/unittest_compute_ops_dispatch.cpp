// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_compute_ops_dispatch.cpp
 * @date   25 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Verify that a per-Context ComputeOps installed via ContextData
 *         actually reaches Tensor::dot / multiply / add through the
 *         tensor's attached ContextData. End-to-end check that
 *         vendor backend dispatch works with virtual dispatch (no
 *         preprocessor branches at the call site).
 */

#include <common_properties.h>
#include <compute_ops.h>
#include <context_data.h>
#include <gtest/gtest.h>
#include <tensor.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <stdexcept>

#ifdef ENABLE_HEXKL
#include <htp_backend.h>
#endif

namespace {

/**
 * @brief Per-test counters incremented by the mock subclass.
 *
 * The mock forwards each op to the real (CPU) backend so that result
 * correctness is preserved; the counters confirm dispatch reached the
 * mock and not the global singleton directly.
 */
struct CallCounters {
  std::atomic<int> sgemm{0};
  std::atomic<int> sgemv{0};
  std::atomic<int> ele_mul{0};
  std::atomic<int> ele_add{0};
  std::atomic<int> scopy{0};
  std::atomic<int> layer_norm{0};
  std::atomic<int> activation{0};
  std::atomic<int> residual_op{0};
  std::atomic<int> swiglu{0};
};

/**
 * @brief Mock ComputeOps subclass: forwards to a "real" backend
 *        (the global one) for correctness while bumping per-op counters.
 *
 * Because every base-class default just throws, this only overrides
 * the ops the tests exercise. Anything the test setup happens to
 * trigger that's not overridden here would throw — the tests stay
 * inside the overridden subset (sgemm/sgemv/ele_mul/ele_add/scopy).
 */
class MockComputeOps : public nntrainer::ComputeOps {
public:
  MockComputeOps(nntrainer::ComputeOps *real, CallCounters *c) :
    real_(real), counters_(c) {}

  void sgemm_fp32(unsigned int o, bool tA, bool tB, unsigned int M,
                  unsigned int N, unsigned int K, float a, const float *A,
                  unsigned int lda, const float *B, unsigned int ldb, float b,
                  float *C, unsigned int ldc) override {
    counters_->sgemm++;
    real_->sgemm_fp32(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void sgemv_fp32(unsigned int o, bool tA, unsigned int M, unsigned int N,
                  float a, const float *A, unsigned int lda, const float *X,
                  unsigned int iX, float b, float *Y,
                  unsigned int iY) override {
    counters_->sgemv++;
    real_->sgemv_fp32(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }
  void ele_mul_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    counters_->ele_mul++;
    real_->ele_mul_fp32(N, X, Y, Z, a, b, is, os);
  }
  void ele_add_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    counters_->ele_add++;
    real_->ele_add_fp32(N, X, Y, Z, a, b, is, os);
  }
  void scopy_fp32(unsigned int N, const float *X, unsigned int iX, float *Y,
                  unsigned int iY) override {
    counters_->scopy++;
    real_->scopy_fp32(N, X, iX, Y, iY);
  }

  // Whole-ops. The neutral LayerNormalizationLayer / ActivationLayer /
  // AdditionLayer reach a backend through these, so a ContextData-attached
  // table has to see them.
  void layer_norm(const nntrainer::Tensor &in, nntrainer::Tensor &out,
                  const nntrainer::Tensor &gamma, const nntrainer::Tensor &beta,
                  float epsilon, unsigned int active_rows,
                  unsigned int row_offset) override {
    counters_->layer_norm++;
    real_->layer_norm(in, out, gamma, beta, epsilon, active_rows, row_offset);
  }
  void activation(const nntrainer::Tensor &in, nntrainer::Tensor &out,
                  int act_type, unsigned int active_rows,
                  unsigned int row_offset) override {
    counters_->activation++;
    real_->activation(in, out, act_type, active_rows, row_offset);
  }
  void residual_op(nntrainer::Tensor &hidden, const nntrainer::Tensor &input,
                   bool accumulate) override {
    counters_->residual_op++;
    real_->residual_op(hidden, input, accumulate);
  }
  void swiglu(const nntrainer::Tensor &in1, const nntrainer::Tensor &in2,
              nntrainer::Tensor &out, unsigned int active_rows,
              unsigned int row_offset) override {
    counters_->swiglu++;
    real_->swiglu(in1, in2, out, active_rows, row_offset);
  }

private:
  nntrainer::ComputeOps *real_;
  CallCounters *counters_;
};

/**
 * @brief Test fixture wiring a MockComputeOps into a fresh ContextData for
 *        each test, so dispatch tests can assert on per-call counters.
 */
class ComputeOpsDispatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    counters = std::make_unique<CallCounters>();
    nntrainer::ensureComputeOps();
    mock_ops = std::make_unique<MockComputeOps>(nntrainer::getComputeOps(),
                                                counters.get());
    ct_data = std::make_shared<nntrainer::ContextData>();
    ct_data->setComputeOps(mock_ops.get());
  }

  std::unique_ptr<CallCounters> counters;
  std::unique_ptr<MockComputeOps> mock_ops;
  std::shared_ptr<nntrainer::ContextData> ct_data;
};

} // namespace

/**
 * @brief A Tensor with no attached ContextData should fall back to the
 *        global ops; no mock counter increments.
 */
TEST_F(ComputeOpsDispatchTest, FallbackToGlobalWhenNoContextData) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  auto out = a.dot(b);

  EXPECT_EQ(counters->sgemm.load(), 0);
  EXPECT_EQ(counters->sgemv.load(), 0);
}

/**
 * @brief When a ContextData with the mock subclass is attached to a
 *        Tensor, calling .dot dispatches through the mock.
 */
TEST_F(ComputeOpsDispatchTest, DotDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  a.setContextData(ct_data);
  auto out = a.dot(b);

  EXPECT_GT(counters->sgemm.load() + counters->sgemv.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 4.0f);
}

/**
 * @brief Element-wise multiply through the attached ContextData should
 *        invoke the mock's ele_mul_fp32.
 */
TEST_F(ComputeOpsDispatchTest, MultiplyDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.multiply(b, out);

  EXPECT_GT(counters->ele_mul.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 6.0f);
}

/**
 * @brief Element-wise add through the attached ContextData should
 *        invoke the mock's ele_add_fp32.
 */
TEST_F(ComputeOpsDispatchTest, AddDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.add(b, out);

  EXPECT_GT(counters->ele_add.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 5.0f);
}

/**
 * @brief Result tensor of a binary op inherits ContextData from `this`,
 *        so a chained op on the result keeps dispatching through the
 *        same backend.
 */
TEST_F(ComputeOpsDispatchTest, ResultInheritsContextDataFromOperand) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  a.setContextData(ct_data);

  auto first = a.dot(b);
  EXPECT_EQ(first.getContextData().get(), ct_data.get());

  int before = counters->sgemm.load() + counters->sgemv.load();
  auto second = first.dot(b);
  int after = counters->sgemm.load() + counters->sgemv.load();
  EXPECT_GT(after, before);
}

/**
 * @brief Replacing the attached ContextData with a different subclass
 *        rebinds dispatch on subsequent calls — the runtime swap
 *        property required for hot-swapping vendor contexts.
 */
TEST_F(ComputeOpsDispatchTest, SwappingContextDataRebindsDispatch) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out1(1, 1, 1, 8);
  nntrainer::Tensor out2(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.multiply(b, out1);
  int after_first = counters->ele_mul.load();
  EXPECT_GT(after_first, 0);

  // Swap to a fresh ContextData carrying its own MockComputeOps with
  // independent counters. Subsequent ops bump fresh counters only.
  auto fresh_counters = std::make_unique<CallCounters>();
  auto fresh_mock = std::make_unique<MockComputeOps>(nntrainer::getComputeOps(),
                                                     fresh_counters.get());
  auto fresh_ct = std::make_shared<nntrainer::ContextData>();
  fresh_ct->setComputeOps(fresh_mock.get());
  a.setContextData(fresh_ct);

  a.multiply(b, out2);
  EXPECT_EQ(counters->ele_mul.load(), after_first);
  EXPECT_GT(fresh_counters->ele_mul.load(), 0);
}

/**
 * @brief Cross-vendor mismatch — when two operands of a binary op
 *        carry DIFFERENT ContextData (e.g. one CPU-resident tensor,
 *        one OpenCL-resident tensor), the op must throw rather than
 *        silently dispatch through one side's ops onto the other
 *        side's incompatible memory. This is the assertion that
 *        protects against the most insidious mixed-backend bug.
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpThrowsOnContextMismatch) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  // Two distinct ContextData instances simulate two different vendor
  // contexts (e.g. CPU + OpenCL) — same kind of mock here, but the
  // identity of the ContextData pointers differs.
  auto ct_other = std::make_shared<nntrainer::ContextData>();
  ct_other->setComputeOps(mock_ops.get());

  a.setContextData(ct_data);
  b.setContextData(ct_other);

  EXPECT_THROW(a.multiply(b, out), std::invalid_argument);
  EXPECT_THROW(a.add(b, out), std::invalid_argument);
  EXPECT_THROW(a.divide(b, out), std::invalid_argument);
}

/**
 * @brief Same ContextData identity on both operands → no throw.
 *        Confirms the mismatch check is keyed on identity, not on
 *        nullness alone (which would be backward-compatibility break).
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpAcceptsSameContext) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  b.setContextData(ct_data); // same instance, not a copy

  EXPECT_NO_THROW(a.multiply(b, out));
}

/**
 * @brief One operand has no ContextData → permissive (legacy code
 *        path). A tensor created without ever touching ContextData
 *        falls back to the global ops table; binary-op'ing it with
 *        a context-attached tensor must NOT throw — that would break
 *        every existing test and call site.
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpAcceptsOneSideUnattached) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  // b has no ContextData — legacy code path

  EXPECT_NO_THROW(a.multiply(b, out));
}

/**
 * @brief Tensor::to(target_ct) deep-copies and re-tags. Result owns
 *        the new ContextData; original is unchanged. After to(), a
 *        previously-mismatched binary op on the migrated tensor
 *        succeeds.
 */
TEST_F(ComputeOpsDispatchTest, ToMigratesContextDataAndUnblocksOp) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  auto ct_other = std::make_shared<nntrainer::ContextData>();
  ct_other->setComputeOps(mock_ops.get());

  a.setContextData(ct_data);
  b.setContextData(ct_other);

  // Migrate b onto a's context. Original b stays on ct_other.
  nntrainer::Tensor b_migrated = b.to(ct_data);
  EXPECT_EQ(b_migrated.getContextData().get(), ct_data.get());
  EXPECT_EQ(b.getContextData().get(), ct_other.get()); // unchanged

  // Now a.multiply(b_migrated) is on the same context — no throw.
  EXPECT_NO_THROW(a.multiply(b_migrated, out));
}

/**
 * @brief layer_norm reaches the ContextData-attached table and produces the
 *        reference LayerNorm. This is the mechanical proof that the neutral
 *        LayerNormalizationLayer's in.getOps()->layer_norm(...) lands on the
 *        backend table rather than the global one.
 */
TEST_F(ComputeOpsDispatchTest, LayerNormDispatchesThroughAttachedContextOps) {
  constexpr unsigned int H = 3, W = 8;
  const float eps = 1e-3f;

  nntrainer::Tensor in(1, 1, H, W);
  nntrainer::Tensor out(1, 1, H, W);
  nntrainer::Tensor gamma(1, 1, 1, W);
  nntrainer::Tensor beta(1, 1, 1, W);

  float *ip = in.getData<float>();
  for (unsigned int h = 0; h < H; ++h)
    for (unsigned int w = 0; w < W; ++w)
      ip[h * W + w] = ((h * W + w) % 7) * 0.5f - 1.0f;
  for (unsigned int w = 0; w < W; ++w) {
    gamma.getData<float>()[w] = 0.5f + 0.1f * w;
    beta.getData<float>()[w] = -0.3f + 0.05f * w;
  }

  in.setContextData(ct_data);
  in.getOps()->layer_norm(in, out, gamma, beta, eps, H, /*row_offset=*/0);

  EXPECT_EQ(counters->layer_norm.load(), 1);

  for (unsigned int h = 0; h < H; ++h) {
    double mean = 0.0;
    for (unsigned int w = 0; w < W; ++w)
      mean += ip[h * W + w];
    mean /= W;
    double var = 0.0;
    for (unsigned int w = 0; w < W; ++w) {
      const double d = ip[h * W + w] - mean;
      var += d * d;
    }
    var /= W;
    const double inv = 1.0 / std::sqrt(var + (double)eps);
    for (unsigned int w = 0; w < W; ++w) {
      const double ref =
        (ip[h * W + w] - mean) * inv * gamma.getData<float>()[w] +
        beta.getData<float>()[w];
      EXPECT_NEAR(out.getData<float>()[h * W + w], (float)ref, 1e-5f);
    }
  }
}

/**
 * @brief activation reaches the attached table, honours the
 *        (active_rows, row_offset) window, and computes GELU.
 */
TEST_F(ComputeOpsDispatchTest, ActivationDispatchesThroughAttachedContextOps) {
  constexpr unsigned int H = 4, W = 4;
  nntrainer::Tensor in(1, 1, H, W);
  nntrainer::Tensor out(1, 1, H, W);
  float *ip = in.getData<float>();
  for (unsigned int i = 0; i < H * W; ++i)
    ip[i] = -2.0f + 0.25f * i;
  out.setValue(-99.0f);

  in.setContextData(ct_data);
  // window = rows [1, 3): the rows outside it must stay untouched.
  in.getOps()->activation(in, out, (int)nntrainer::ActivationType::ACT_GELU,
                          /*active_rows=*/2, /*row_offset=*/1);

  EXPECT_EQ(counters->activation.load(), 1);

  for (unsigned int i = 0; i < H * W; ++i) {
    const float got = out.getData<float>()[i];
    if (i < W || i >= 3 * W) {
      EXPECT_FLOAT_EQ(got, -99.0f) << "row window leaked at " << i;
    } else {
      const double x = ip[i];
      const double ref = 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752));
      EXPECT_NEAR(got, (float)ref, 1e-5f);
    }
  }
}

/**
 * @brief supports_activation() is the non-throwing capability query; the base
 *        table answers yes for every mode, since the CPU services them all.
 */
TEST_F(ComputeOpsDispatchTest, SupportsActivationDefaultsToTrue) {
  nntrainer::Tensor a(1, 1, 1, 4);
  a.setContextData(ct_data);
  EXPECT_TRUE(
    a.getOps()->supports_activation((int)nntrainer::ActivationType::ACT_GELU));
  EXPECT_TRUE(
    a.getOps()->supports_activation((int)nntrainer::ActivationType::ACT_RELU));
}

/**
 * @brief residual_op reaches the attached table and gives the neutral
 *        AdditionLayer's two modes: overwrite for the first operand,
 *        accumulate for the rest.
 */
TEST_F(ComputeOpsDispatchTest, ResidualOpDispatchesThroughAttachedContextOps) {
  constexpr unsigned int N = 6;
  nntrainer::Tensor hidden(1, 1, 1, N);
  nntrainer::Tensor a(1, 1, 1, N);
  nntrainer::Tensor b(1, 1, 1, N);
  hidden.setValue(-99.0f);
  for (unsigned int i = 0; i < N; ++i) {
    a.getData<float>()[i] = (float)i;
    b.getData<float>()[i] = 10.0f * (float)i;
  }

  hidden.setContextData(ct_data);
  hidden.getOps()->residual_op(hidden, a, /*accumulate=*/false);
  hidden.getOps()->residual_op(hidden, b, /*accumulate=*/true);

  EXPECT_EQ(counters->residual_op.load(), 2);
  for (unsigned int i = 0; i < N; ++i)
    EXPECT_FLOAT_EQ(hidden.getData<float>()[i], 11.0f * (float)i);
}

/**
 * @brief A two-operand whole-op (swiglu stands in for the gated family)
 *        reaches the attached table and respects the row window.
 */
TEST_F(ComputeOpsDispatchTest, SwigluDispatchesThroughAttachedContextOps) {
  constexpr unsigned int H = 3, W = 4;
  nntrainer::Tensor gate(1, 1, H, W);
  nntrainer::Tensor up(1, 1, H, W);
  nntrainer::Tensor out(1, 1, H, W);
  for (unsigned int i = 0; i < H * W; ++i) {
    gate.getData<float>()[i] = -1.5f + 0.25f * i;
    up.getData<float>()[i] = 2.0f - 0.1f * i;
  }
  out.setValue(-99.0f);

  gate.setContextData(ct_data);
  gate.getOps()->swiglu(gate, up, out, /*active_rows=*/1, /*row_offset=*/1);

  EXPECT_EQ(counters->swiglu.load(), 1);
  for (unsigned int i = 0; i < H * W; ++i) {
    const float got = out.getData<float>()[i];
    if (i < W || i >= 2 * W) {
      EXPECT_FLOAT_EQ(got, -99.0f) << "row window leaked at " << i;
    } else {
      const double g = gate.getData<float>()[i];
      const double ref = (g / (1.0 + std::exp(-g))) * up.getData<float>()[i];
      EXPECT_NEAR(got, (float)ref, 1e-5f);
    }
  }
}

/**
 * @brief An op the table does not implement reports it instead of computing
 *        something else. The mock overrides only part of the whole-op family,
 *        so the rest still hit the throwing defaults.
 */
TEST_F(ComputeOpsDispatchTest, UnimplementedWholeOpThrows) {
  nntrainer::Tensor a(1, 1, 1, 4);
  nntrainer::Tensor b(1, 1, 1, 4);
  nntrainer::Tensor out(1, 1, 1, 4);
  a.setContextData(ct_data);
  EXPECT_THROW(a.getOps()->geglu(a, b, out, 1, 0), std::runtime_error);
}

#ifdef ENABLE_HEXKL
/**
 * @brief The HTP ops table resolves and every op on it still computes.
 *
 * No kernel is wired to the NPU yet, so an engine="htp" model has to keep
 * working by falling through to the CPU implementation. Asserting a real
 * result -- rather than merely that the pointer is non-null -- is what
 * catches a table whose entries throw, which is what the base ComputeOps
 * defaults do. The accelerator predicates arrive with the first kernel.
 */
TEST(HtpBackendSmoke, TableFallsThroughToCpu) {
  auto *htp = nntrainer::get_htp_ops();
  ASSERT_NE(htp, nullptr);
  nntrainer::ensureComputeOps();

  constexpr unsigned int N = 4;
  const float x[N] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float y[N] = {10.0f, 20.0f, 30.0f, 40.0f};
  float z[N] = {0.0f, 0.0f, 0.0f, 0.0f};

  ASSERT_NO_THROW(htp->ele_add_fp32(N, x, y, z, 1.0f, 0.0f, 1, 1));
  for (unsigned int i = 0; i < N; ++i)
    EXPECT_FLOAT_EQ(z[i], x[i] + y[i]);
}
#endif // ENABLE_HEXKL

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
