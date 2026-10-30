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
#include <fp16.h>
#include <gtest/gtest.h>
#include <quantizer.h>
#include <tensor.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

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
  std::atomic<int> rotary_embedding{0};
  std::atomic<int> kv_cache_write{0};
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
  void rotary_embedding(const nntrainer::Tensor &in, nntrainer::Tensor &out,
                        unsigned int head_dim,
                        const nntrainer::RotaryTable &table,
                        unsigned int start_pos, bool rotate) override {
    counters_->rotary_embedding++;
    real_->rotary_embedding(in, out, head_dim, table, start_pos, rotate);
  }
  void kv_cache_write(const nntrainer::Tensor &k, const nntrainer::Tensor &v,
                      const nntrainer::KVCacheView &kv, unsigned int pos,
                      const nntrainer::RotaryTable *rope) override {
    counters_->kv_cache_write++;
    real_->kv_cache_write(k, v, kv, pos, rope);
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
 * @brief A Q4_0 dot on a context-attached tensor must still compute.
 *
 * The scalar Q4_0 GEMM is a CPU-table op. An accelerator ComputeOps only
 * implements the accel-only ops it advertises through supports_*() -- the mock
 * here, like every real accelerator table, implements none of the Q4_0 ops and
 * inherits the base defaults, which throw. M == 1 is the decode step, which
 * never takes the accel batch path, so before the fix this threw
 * "ComputeOps::gemm_q4_0_fp32 not implemented by this backend" on every decode
 * step of a Q4_0 model running on a gpu context.
 */
TEST_F(ComputeOpsDispatchTest, ScalarQ4_0GemmTakesTheCpuTable) {
  nntrainer::init_backend();

  constexpr unsigned int K = 512;
  constexpr unsigned int N = 512;

  nntrainer::Tensor w_fp32(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  for (unsigned int k = 0; k < K; ++k)
    for (unsigned int n = 0; n < N; ++n)
      w_fp32.setValue(0, 0, k, n,
                      0.01f * static_cast<float>((k + 3 * n) % 7) - 0.03f);

  auto quantizer =
    nntrainer::Quantization::createQuantizer(nntrainer::QScheme::Q4_0);
  nntrainer::Tensor w_q40 =
    quantizer->quantize(w_fp32, nntrainer::Tdatatype::Q4_0);
  ASSERT_EQ(w_q40.getDataType(), nntrainer::Tdatatype::Q4_0);

  // M == 1 is the decode step.
  nntrainer::Tensor act(1, 1, 1, K);
  for (unsigned int k = 0; k < K; ++k)
    act.setValue(0, 0, 0, k, 0.5f - 0.001f * static_cast<float>(k % 11));

  // Reference: no ContextData, so getOps() already resolves to the CPU table.
  nntrainer::Tensor expected(1, 1, 1, N);
  act.dot(w_q40, expected);

  nntrainer::Tensor act_ctx(act);
  act_ctx.setContextData(ct_data);

  nntrainer::Tensor got(1, 1, 1, N);
  ASSERT_NO_THROW(act_ctx.dot(w_q40, got));

  for (unsigned int n = 0; n < N; ++n)
    EXPECT_FLOAT_EQ(got.getValue<float>(0, 0, 0, n),
                    expected.getValue<float>(0, 0, 0, n));
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

namespace {

/**
 * @brief cos / sin rows for positions [0, positions) and angles
 *        theta_i = 10000^(-2i / head_dim), in the RotaryTable layout.
 */
struct RotaryRows {
  std::vector<float> cos, sin;
  nntrainer::RotaryTable table;

  RotaryRows(unsigned int positions, unsigned int head_dim) {
    const unsigned int half = head_dim / 2;
    for (unsigned int p = 0; p < positions; ++p) {
      for (unsigned int i = 0; i < half; ++i) {
        const double angle =
          p * std::pow(10000.0, -2.0 * i / static_cast<double>(head_dim));
        cos.push_back((float)std::cos(angle));
        sin.push_back((float)std::sin(angle));
      }
    }
    table.cos = cos.data();
    table.sin = sin.data();
    table.positions = positions;
    table.half = half;
  }
};

/** @brief Rotate-half reference of one element of a [rows, W] input. */
float rotatedRef(const float *x, const RotaryRows &rows, unsigned int W,
                 unsigned int head_dim, unsigned int start_pos, unsigned int r,
                 unsigned int w) {
  const unsigned int half = head_dim / 2;
  const unsigned int head = w / head_dim * head_dim;
  const unsigned int i = (w - head) % half;
  const size_t t = (size_t)(start_pos + r) * half + i;
  const float a = x[r * W + head + i];
  const float b = x[r * W + head + i + half];
  return (w - head) < half ? a * rows.cos[t] - b * rows.sin[t]
                           : a * rows.sin[t] + b * rows.cos[t];
}

} // namespace

/**
 * @brief rotary_embedding reaches the attached table and rotates every head of
 *        every row by the table row of its position, in place.
 */
TEST_F(ComputeOpsDispatchTest, RotaryEmbeddingDispatchesThroughAttachedOps) {
  constexpr unsigned int H = 3, head_dim = 8, W = 2 * head_dim, start = 5;
  const RotaryRows rows(start + H, head_dim);
  nntrainer::Tensor x(1, 1, H, W);
  for (unsigned int i = 0; i < H * W; ++i)
    x.getData<float>()[i] = 0.1f * i - 1.0f;
  const std::vector<float> before(x.getData<float>(),
                                  x.getData<float>() + H * W);

  x.setContextData(ct_data);
  x.getOps()->rotary_embedding(x, x, head_dim, rows.table, start, true);

  EXPECT_EQ(counters->rotary_embedding.load(), 1);
  for (unsigned int r = 0; r < H; ++r)
    for (unsigned int w = 0; w < W; ++w)
      EXPECT_NEAR(x.getData<float>()[r * W + w],
                  rotatedRef(before.data(), rows, W, head_dim, start, r, w),
                  1e-5f)
        << "row " << r << " col " << w;
}

/**
 * @brief Into fp16 bits (a KV-cache write): rotated with rotate == true,
 *        converted unchanged with rotate == false. The input is left as is.
 */
TEST_F(ComputeOpsDispatchTest, RotaryEmbeddingWritesFp16Bits) {
  constexpr unsigned int H = 2, head_dim = 8, W = head_dim, start = 3;
  const RotaryRows rows(start + H, head_dim);
  nntrainer::Tensor x(1, 1, H, W);
  for (unsigned int i = 0; i < H * W; ++i)
    x.getData<float>()[i] = 0.25f * i - 2.0f;
  const std::vector<float> before(x.getData<float>(),
                                  x.getData<float>() + H * W);
  const nntrainer::TensorDim fp16_bits(
    {1, 1, H, W, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16});
  nntrainer::Tensor rotated(fp16_bits);
  nntrainer::Tensor copied(fp16_bits);

  x.getOps()->rotary_embedding(x, rotated, head_dim, rows.table, start, true);
  x.getOps()->rotary_embedding(x, copied, head_dim, rows.table, start, false);

  for (unsigned int r = 0; r < H; ++r) {
    for (unsigned int w = 0; w < W; ++w) {
      const unsigned int i = r * W + w;
      EXPECT_FLOAT_EQ(x.getData<float>()[i], before[i]);
      EXPECT_NEAR(
        nntrainer::compute_fp16_to_fp32(rotated.getData<uint16_t>()[i]),
        rotatedRef(before.data(), rows, W, head_dim, start, r, w), 2e-3f);
      EXPECT_EQ(copied.getData<uint16_t>()[i],
                nntrainer::compute_fp32_to_fp16(before[i]));
    }
  }
}

#ifdef ENABLE_FP16
/**
 * @brief An FP16 input reads the fp16 table and rotates into an FP16 output,
 *        in place or not; it does not take the copy-only (rotate == false)
 *        form, which exists for the FP32 value write.
 */
TEST_F(ComputeOpsDispatchTest, RotaryEmbeddingRotatesFp16Input) {
  constexpr unsigned int H = 3, head_dim = 16, W = 2 * head_dim, start = 4;
  RotaryRows rows(start + H, head_dim);
  std::vector<uint16_t> cos16, sin16;
  for (float v : rows.cos)
    cos16.push_back(nntrainer::compute_fp32_to_fp16(v));
  for (float v : rows.sin)
    sin16.push_back(nntrainer::compute_fp32_to_fp16(v));
  rows.table.cos_fp16 = cos16.data();
  rows.table.sin_fp16 = sin16.data();

  const nntrainer::TensorDim dim(
    {1, 1, H, W, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
  nntrainer::Tensor x(dim), out(dim);
  std::vector<float> before;
  for (unsigned int i = 0; i < H * W; ++i) {
    x.getData<_FP16>()[i] = static_cast<_FP16>(0.1f * i - 1.5f);
    before.push_back(static_cast<float>(x.getData<_FP16>()[i]));
  }

  x.setContextData(ct_data);
  x.getOps()->rotary_embedding(x, out, head_dim, rows.table, start, true);
  EXPECT_EQ(counters->rotary_embedding.load(), 1);
  for (unsigned int r = 0; r < H; ++r)
    for (unsigned int w = 0; w < W; ++w)
      EXPECT_NEAR(static_cast<float>(out.getData<_FP16>()[r * W + w]),
                  rotatedRef(before.data(), rows, W, head_dim, start, r, w),
                  2e-2f)
        << "row " << r << " col " << w;

  x.getOps()->rotary_embedding(x, x, head_dim, rows.table, start, true);
  for (unsigned int i = 0; i < H * W; ++i)
    EXPECT_EQ(static_cast<float>(x.getData<_FP16>()[i]),
              static_cast<float>(out.getData<_FP16>()[i]));

  EXPECT_THROW(
    x.getOps()->rotary_embedding(x, out, head_dim, rows.table, start, false),
    std::invalid_argument);
}
#endif

/**
 * @brief A position past the end of the table is rejected instead of reading
 *        beyond it.
 */
TEST_F(ComputeOpsDispatchTest, RotaryEmbeddingRejectsPositionsPastTable) {
  constexpr unsigned int H = 2, head_dim = 8;
  const RotaryRows rows(4, head_dim);
  nntrainer::Tensor x(1, 1, H, head_dim);
  EXPECT_NO_THROW(
    x.getOps()->rotary_embedding(x, x, head_dim, rows.table, 2, true));
  EXPECT_THROW(
    x.getOps()->rotary_embedding(x, x, head_dim, rows.table, 3, true),
    std::invalid_argument);
}

/**
 * @brief Each batch's rows start again at start_pos: height index h is at
 *        position start_pos + h in every batch.
 */
TEST_F(ComputeOpsDispatchTest, RotaryEmbeddingRestartsPositionsPerBatch) {
  constexpr unsigned int B = 2, H = 3, head_dim = 8, W = head_dim, start = 2;
  const RotaryRows rows(start + H, head_dim);
  nntrainer::Tensor x(B, 1, H, W);
  for (unsigned int i = 0; i < B * H * W; ++i)
    x.getData<float>()[i] = 0.05f * i - 1.0f;
  const std::vector<float> before(x.getData<float>(),
                                  x.getData<float>() + B * H * W);

  x.getOps()->rotary_embedding(x, x, head_dim, rows.table, start, true);

  for (unsigned int b = 0; b < B; ++b)
    for (unsigned int h = 0; h < H; ++h)
      for (unsigned int w = 0; w < W; ++w)
        EXPECT_NEAR(
          x.getData<float>()[(b * H + h) * W + w],
          rotatedRef(before.data() + b * H * W, rows, W, head_dim, start, h, w),
          1e-5f)
          << "batch " << b << " row " << h << " col " << w;
}

/**
 * @brief A head_dim that cannot tile the row, or a missing fp32 table, is
 *        rejected instead of looping or reading past the row.
 */
TEST_F(ComputeOpsDispatchTest, RotaryEmbeddingRejectsBadHeadDimOrTable) {
  const RotaryRows rows(4, 8);
  nntrainer::Tensor x(1, 1, 1, 12);
  for (unsigned int head_dim : {0u, 5u, 8u})
    EXPECT_THROW(
      x.getOps()->rotary_embedding(x, x, head_dim, rows.table, 0, true),
      std::invalid_argument)
      << "head_dim " << head_dim;

  // The copy-only form reads no table, so only the head_dim check stops a
  // head_dim of 0 (the kernels' head loop would not advance) or an odd one
  // that divides the row.
  const nntrainer::TensorDim bits15(
    {1, 1, 1, 15, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16});
  nntrainer::Tensor z(1, 1, 1, 15), z_bits(bits15);
  for (unsigned int head_dim : {0u, 5u})
    EXPECT_THROW(
      x.getOps()->rotary_embedding(z, z_bits, head_dim, rows.table, 0, false),
      std::invalid_argument)
      << "copy-only, head_dim " << head_dim;

  nntrainer::Tensor y(1, 1, 1, 8);
  nntrainer::RotaryTable no_fp32 = rows.table;
  no_fp32.cos = nullptr;
  no_fp32.sin = nullptr;
  EXPECT_THROW(x.getOps()->rotary_embedding(y, y, 8, no_fp32, 0, true),
               std::invalid_argument);
  EXPECT_NO_THROW(x.getOps()->rotary_embedding(y, y, 8, no_fp32, 0, false));
}

namespace {

/** @brief A [1, 1, rows, width] cache plane of fp16 bits, every bit 0xABCD. */
nntrainer::Tensor cachePlane(unsigned int rows, unsigned int width) {
  nntrainer::Tensor t(
    nntrainer::TensorDim({1, 1, rows, width, nntrainer::Tformat::NCHW,
                          nntrainer::Tdatatype::UINT16}));
  for (unsigned int i = 0; i < rows * width; ++i)
    t.getData<uint16_t>()[i] = 0xABCD;
  return t;
}

/** @brief A [1, 1, rows, width] FP32 step with distinct values. */
nntrainer::Tensor stepRows(unsigned int rows, unsigned int width, float base) {
  nntrainer::Tensor t(1, 1, rows, width);
  for (unsigned int i = 0; i < rows * width; ++i)
    t.getData<float>()[i] = base + 0.05f * i;
  return t;
}

} // namespace

/**
 * @brief kv_cache_write reaches the attached table and writes rows
 *        [pos, pos + rows): K rotated at its absolute positions, V converted
 *        to fp16 bits, every other cache row left alone.
 */
TEST_F(ComputeOpsDispatchTest, KvCacheWriteDispatchesAndPlacesRows) {
  constexpr unsigned int R = 2, head_dim = 8, W = 2 * head_dim, pos = 3;
  const RotaryRows rows(pos + R, head_dim);
  nntrainer::Tensor k = stepRows(R, W, -1.0f), v = stepRows(R, W, 0.5f);
  nntrainer::Tensor key = cachePlane(6, W), value = cachePlane(6, W);
  nntrainer::KVCacheView kv;
  kv.key = &key;
  kv.value = &value;
  kv.head_dim = head_dim;

  k.setContextData(ct_data);
  k.getOps()->kv_cache_write(k, v, kv, pos, &rows.table);

  EXPECT_EQ(counters->kv_cache_write.load(), 1);
  for (unsigned int r = 0; r < 6; ++r) {
    for (unsigned int w = 0; w < W; ++w) {
      const uint16_t kb = key.getData<uint16_t>()[r * W + w];
      const uint16_t vb = value.getData<uint16_t>()[r * W + w];
      if (r < pos || r >= pos + R) {
        EXPECT_EQ(kb, 0xABCD) << "row " << r;
        EXPECT_EQ(vb, 0xABCD) << "row " << r;
        continue;
      }
      const unsigned int s = r - pos;
      // rotatedRef indexes the table by start + s, the absolute position.
      EXPECT_NEAR(nntrainer::compute_fp16_to_fp32(kb),
                  rotatedRef(k.getData<float>(), rows, W, head_dim, pos, s, w),
                  2e-3f)
        << "row " << r << " col " << w;
      EXPECT_EQ(vb,
                nntrainer::compute_fp32_to_fp16(v.getData<float>()[s * W + w]))
        << "row " << r << " col " << w;
    }
  }
}

/**
 * @brief On a ringed cache absolute row n lands in row n % ring_cap, while K
 *        still rotates by its absolute position; a step that would cross the
 *        ring seam is rejected.
 */
TEST_F(ComputeOpsDispatchTest, KvCacheWriteWrapsOnARing) {
  constexpr unsigned int R = 2, head_dim = 8, W = head_dim, ring = 4, pos = 6;
  const RotaryRows rows(pos + R + 1, head_dim);
  nntrainer::Tensor k = stepRows(R, W, 0.25f), v = stepRows(R, W, -0.75f);
  nntrainer::Tensor key = cachePlane(ring, W), value = cachePlane(ring, W);
  nntrainer::KVCacheView kv;
  kv.key = &key;
  kv.value = &value;
  kv.head_dim = head_dim;
  kv.ring_cap = ring;

  k.getOps()->kv_cache_write(k, v, kv, pos, &rows.table);

  for (unsigned int r = 0; r < ring; ++r) {
    for (unsigned int w = 0; w < W; ++w) {
      const uint16_t kb = key.getData<uint16_t>()[r * W + w];
      if (r < pos % ring) {
        EXPECT_EQ(kb, 0xABCD) << "row " << r;
        continue;
      }
      const unsigned int s = r - pos % ring;
      EXPECT_NEAR(nntrainer::compute_fp16_to_fp32(kb),
                  rotatedRef(k.getData<float>(), rows, W, head_dim, pos, s, w),
                  2e-3f)
        << "row " << r << " col " << w;
    }
  }

  // Absolute 7 and 8 are ring rows 3 and 0: across the seam.
  EXPECT_THROW(k.getOps()->kv_cache_write(k, v, kv, pos + 1, &rows.table),
               std::invalid_argument);
}

/**
 * @brief A null plane is left to the caller; a write past the last cache row
 *        is rejected.
 */
TEST_F(ComputeOpsDispatchTest, KvCacheWriteSkipsNullPlaneAndRejects) {
  constexpr unsigned int R = 2, head_dim = 8, W = head_dim;
  nntrainer::Tensor k = stepRows(R, W, 0.0f), v = stepRows(R, W, 1.0f);
  nntrainer::Tensor key = cachePlane(4, W), value = cachePlane(4, W);
  nntrainer::KVCacheView kv;
  kv.value = &value;
  kv.head_dim = head_dim;

  k.getOps()->kv_cache_write(k, v, kv, 0, nullptr);
  EXPECT_EQ(value.getData<uint16_t>()[0],
            nntrainer::compute_fp32_to_fp16(v.getData<float>()[0]));
  EXPECT_EQ(key.getData<uint16_t>()[0], 0xABCD);

  kv.key = &key;
  EXPECT_THROW(k.getOps()->kv_cache_write(k, v, kv, 3, nullptr),
               std::invalid_argument);
}

/**
 * @brief Without RoPE an odd head_dim is allowed, and every element of K
 *        and V is still converted to fp16 bits.
 */
TEST_F(ComputeOpsDispatchTest, KvCacheWriteConvertsOddHeadDim) {
  constexpr unsigned int R = 2, head_dim = 5, W = 3 * head_dim;
  nntrainer::Tensor k = stepRows(R, W, -0.4f), v = stepRows(R, W, 0.3f);
  nntrainer::Tensor key = cachePlane(3, W), value = cachePlane(3, W);
  nntrainer::KVCacheView kv;
  kv.key = &key;
  kv.value = &value;
  kv.head_dim = head_dim;

  k.getOps()->kv_cache_write(k, v, kv, 1, nullptr);
  for (unsigned int i = 0; i < R * W; ++i) {
    EXPECT_EQ(key.getData<uint16_t>()[W + i],
              nntrainer::compute_fp32_to_fp16(k.getData<float>()[i]));
    EXPECT_EQ(value.getData<uint16_t>()[W + i],
              nntrainer::compute_fp32_to_fp16(v.getData<float>()[i]));
  }
  EXPECT_EQ(key.getData<uint16_t>()[0], 0xABCD);
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
