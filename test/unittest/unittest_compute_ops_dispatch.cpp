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

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
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
  std::atomic<int> attention{0};
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
  void attention(const nntrainer::Tensor &q, const nntrainer::KVCacheView &kv,
                 nntrainer::Tensor &out,
                 const nntrainer::AttentionOpParams &p) override {
    counters_->attention++;
    real_->attention(q, kv, out, p);
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
 * @brief attention reaches the attached table and computes
 *        softmax(softcap(q . k / sqrt(d))) . v per query row over the keys
 *        its causal window sees, with two query heads on one KV head; a
 *        ringed cache and a kv_len that is not q_pos + rows are rejected.
 */
TEST_F(ComputeOpsDispatchTest, AttentionDispatchesAndMatchesTheFormula) {
  constexpr unsigned int D = 4, Hq = 2, R = 2, keys = 3, window = 2;
  constexpr float cap = 0.5f;
  nntrainer::Tensor q = stepRows(R, Hq * D, -0.2f);
  nntrainer::Tensor key = cachePlane(keys, D), value = cachePlane(keys, D);
  for (unsigned int i = 0; i < keys * D; ++i) {
    key.getData<uint16_t>()[i] =
      nntrainer::compute_fp32_to_fp16(0.1f * i - 0.6f);
    value.getData<uint16_t>()[i] =
      nntrainer::compute_fp32_to_fp16(0.3f - 0.05f * i);
  }
  nntrainer::Tensor out(1, 1, R, Hq * D);
  nntrainer::KVCacheView kv;
  kv.key = &key;
  kv.value = &value;
  kv.head_dim = D;
  nntrainer::AttentionOpParams p;
  p.num_heads_q = Hq;
  p.q_pos = keys - R;
  p.kv_len = keys;
  p.window = window;
  p.softcap = cap;

  q.setContextData(ct_data);
  q.getOps()->attention(q, kv, out, p);
  EXPECT_EQ(counters->attention.load(), 1);

  for (unsigned int r = 0; r < R; ++r) {
    const unsigned int pos = p.q_pos + r;
    const unsigned int first = pos + 1 > window ? pos + 1 - window : 0;
    for (unsigned int h = 0; h < Hq; ++h) {
      const float *qh = q.getData<float>() + (r * Hq + h) * D;
      std::vector<float> s;
      float mx = -INFINITY;
      for (unsigned int j = first; j <= pos; ++j) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < D; ++d)
          dot += qh[d] * nntrainer::compute_fp16_to_fp32(
                           key.getData<uint16_t>()[j * D + d]);
        s.push_back(cap * std::tanh(dot / std::sqrt((float)D) / cap));
        mx = std::max(mx, s.back());
      }
      float sum = 0.0f;
      for (float &x : s)
        sum += (x = std::exp(x - mx));
      for (unsigned int d = 0; d < D; ++d) {
        float expect = 0.0f;
        for (unsigned int j = first; j <= pos; ++j)
          expect += s[j - first] / sum *
                    nntrainer::compute_fp16_to_fp32(
                      value.getData<uint16_t>()[j * D + d]);
        EXPECT_NEAR(out.getData<float>()[(r * Hq + h) * D + d], expect, 1e-4f)
          << "row " << r << " head " << h << " dim " << d;
      }
    }
  }

  kv.ring_cap = keys;
  EXPECT_THROW(q.getOps()->attention(q, kv, out, p), std::invalid_argument);
  kv.ring_cap = 0;
  p.kv_len = keys + 1;
  EXPECT_THROW(q.getOps()->attention(q, kv, out, p), std::invalid_argument);
}

namespace {

/** @brief The cache a per-row attention case reads. */
enum class AttnCache { FP32, FP16Bits, Int8 };

/** @brief One configuration of the per-row attention op. */
struct AttnCase {
  const char *name;
  bool fp16_q; /**< FP16 query and output (FP32 otherwise) */
  AttnCache cache;
  unsigned int rows;  /**< query rows: 1 is a decode step */
  unsigned int q_pos; /**< absolute position of query row 0 */
  bool causal;
  size_t window; /**< UINT_MAX for none */
  float softcap; /**< 0 for none */
  bool sink;
  bool flash = false; /**< allow the flash arm (AttentionOpParams::flash) */
};

/**
 * @brief Set (or, for a null value, unset) an environment variable for one
 *        scope, restoring it after.
 */
class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : name_(name) {
    if (const char *old = std::getenv(name)) {
      had_ = true;
      old_ = old;
    }
    set(value);
  }
  ~ScopedEnv() { set(had_ ? old_.c_str() : nullptr); }

private:
  void set(const char *value) {
#ifdef _WIN32
    _putenv_s(name_, value ? value : "");
#else
    if (value)
      setenv(name_, value, 1);
    else
      unsetenv(name_);
#endif
  }

  const char *name_;
  bool had_ = false;
  std::string old_;
};

/** @brief A tensor of the given dtype holding @p v (fp16 bits for UINT16). */
nntrainer::Tensor makeTensor(unsigned int h, unsigned int w,
                             nntrainer::Tdatatype dt,
                             const std::vector<float> &v) {
  nntrainer::Tensor t(
    nntrainer::TensorDim({1, 1, h, w, nntrainer::Tformat::NCHW, dt}));
  for (size_t i = 0; i < v.size(); ++i) {
    if (dt == nntrainer::Tdatatype::FP32)
      t.getData<float>()[i] = v[i];
    else
      t.getData<uint16_t>()[i] = nntrainer::compute_fp32_to_fp16(v[i]);
  }
  return t;
}

/** @brief @p x as the fp16 value it is stored as. */
float roundFp16(float x) {
  return nntrainer::compute_fp16_to_fp32(nntrainer::compute_fp32_to_fp16(x));
}

/**
 * @brief Run ComputeOps::attention on one case and compare every output with
 *        softmax(softcap(q . k / sqrt(d))) . v over the keys each row sees,
 *        computed in fp32 from the values the tensors actually hold.
 */
void checkAttention(const AttnCase &c, float tol) {
  SCOPED_TRACE(c.name);
  constexpr unsigned int D = 4, Hq = 4, Hkv = 2, G = Hq / Hkv;
  const unsigned int keys = c.q_pos + c.rows;
  const auto q_dt =
    c.fp16_q ? nntrainer::Tdatatype::FP16 : nntrainer::Tdatatype::FP32;

  std::vector<float> q(c.rows * Hq * D), k(keys * Hkv * D), v(keys * Hkv * D),
    sink(Hq);
  // Scores of a few units, so a softcap and a window change the softmax
  // visibly and a dropped one fails the tolerance.
  for (size_t i = 0; i < q.size(); ++i)
    q[i] = 2.0f * std::sin(0.7f * i + 0.1f);
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = 0.5f * std::cos(0.37f * i);
    v[i] = 0.4f * std::sin(0.23f * i + 1.0f);
  }
  for (unsigned int h = 0; h < Hq; ++h)
    sink[h] = 0.25f * h - 0.3f;
  if (c.fp16_q) {
    for (float &x : q)
      x = roundFp16(x);
    for (float &x : sink)
      x = roundFp16(x);
  }

  // The cache, and the values it holds as the reference reads them.
  nntrainer::Tensor key, value;
  std::vector<uint16_t> kscale, vscale;
  std::vector<float> kref = k, vref = v;
  if (c.cache == AttnCache::Int8) {
    key = nntrainer::Tensor(
      nntrainer::TensorDim({1, 1, keys, Hkv * D, nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::UINT8}));
    value = nntrainer::Tensor(
      nntrainer::TensorDim({1, 1, keys, Hkv * D, nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::UINT8}));
    auto quantize = [&](const std::vector<float> &x, nntrainer::Tensor &t,
                        std::vector<uint16_t> &scale, std::vector<float> &ref) {
      int8_t *dst = reinterpret_cast<int8_t *>(t.getData<uint8_t>());
      for (unsigned int j = 0; j < keys * Hkv; ++j) {
        float amax = 0.0f;
        for (unsigned int d = 0; d < D; ++d)
          amax = std::max(amax, std::fabs(x[j * D + d]));
        scale.push_back(nntrainer::compute_fp32_to_fp16(amax / 127.0f));
        const float s = nntrainer::compute_fp16_to_fp32(scale.back());
        for (unsigned int d = 0; d < D; ++d) {
          const int qv = static_cast<int>(std::lround(x[j * D + d] / s));
          dst[j * D + d] =
            static_cast<int8_t>(std::max(-127, std::min(127, qv)));
          ref[j * D + d] = dst[j * D + d] * s;
        }
      }
    };
    quantize(k, key, kscale, kref);
    quantize(v, value, vscale, vref);
  } else {
    const auto kv_dt = c.cache == AttnCache::FP32
                         ? nntrainer::Tdatatype::FP32
                         : (c.fp16_q ? nntrainer::Tdatatype::FP16
                                     : nntrainer::Tdatatype::UINT16);
    key = makeTensor(keys, Hkv * D, kv_dt, k);
    value = makeTensor(keys, Hkv * D, kv_dt, v);
    if (c.cache == AttnCache::FP16Bits) {
      for (float &x : kref)
        x = roundFp16(x);
      for (float &x : vref)
        x = roundFp16(x);
    }
  }

  nntrainer::Tensor qt = makeTensor(c.rows, Hq * D, q_dt, q);
  nntrainer::Tensor st = makeTensor(1, Hq, q_dt, sink);
  nntrainer::Tensor out(nntrainer::TensorDim(
    {1, 1, c.rows, Hq * D, nntrainer::Tformat::NCHW, q_dt}));
  nntrainer::KVCacheView kv;
  kv.key = &key;
  kv.value = &value;
  kv.head_dim = D;
  kv.key_scale = kscale.empty() ? nullptr : kscale.data();
  kv.value_scale = vscale.empty() ? nullptr : vscale.data();
  nntrainer::AttentionOpParams p;
  p.num_heads_q = Hq;
  p.q_pos = c.q_pos;
  p.kv_len = keys;
  p.causal = c.causal;
  p.window = c.window;
  p.softcap = c.softcap;
  p.sink = c.sink ? &st : nullptr;
  p.flash = c.flash;
  qt.getOps()->attention(qt, kv, out, p);

  for (unsigned int r = 0; r < c.rows; ++r) {
    const unsigned int pos = c.q_pos + r;
    const unsigned int last = c.causal ? pos : keys - 1;
    const unsigned int first =
      c.causal && pos + 1 > c.window ? pos + 1 - c.window : 0;
    for (unsigned int h = 0; h < Hq; ++h) {
      const unsigned int kh = h / G;
      std::vector<float> s;
      float mx = c.sink ? sink[h] : -INFINITY;
      for (unsigned int j = first; j <= last; ++j) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < D; ++d)
          dot += q[(r * Hq + h) * D + d] * kref[(j * Hkv + kh) * D + d];
        float x = dot / std::sqrt(static_cast<float>(D));
        if (c.softcap > 0.0f)
          x = c.softcap * std::tanh(x / c.softcap);
        s.push_back(x);
        mx = std::max(mx, x);
      }
      float sum = c.sink ? std::exp(sink[h] - mx) : 0.0f;
      for (float &x : s)
        sum += (x = std::exp(x - mx));
      for (unsigned int d = 0; d < D; ++d) {
        float expect = 0.0f;
        for (unsigned int j = first; j <= last; ++j)
          expect += s[j - first] / sum * vref[(j * Hkv + kh) * D + d];
        const size_t o = (r * Hq + h) * D + d;
        const float got =
          c.fp16_q ? nntrainer::compute_fp16_to_fp32(out.getData<uint16_t>()[o])
                   : out.getData<float>()[o];
        EXPECT_NEAR(got, expect, tol)
          << "row " << r << " head " << h << " dim " << d;
      }
    }
  }
}

} // namespace

/**
 * @brief The per-row arm on an FP32 query, for each cache it reads (FP32,
 *        fp16 bits), decode and prefill steps, with a window, softcap, sink
 *        or non-causal rows.
 */
TEST_F(ComputeOpsDispatchTest, AttentionFp32QueryPaths) {
  const AttnCase cases[] = {
    {"fp32_cache_decode", false, AttnCache::FP32, 1, 4, true, 3, 0.5f, false},
    {"fp32_cache_prefill", false, AttnCache::FP32, 3, 2, true, 3, 0.0f, false},
    {"fp16_bits_sink_decode", false, AttnCache::FP16Bits, 1, 3, true, UINT_MAX,
     0.0f, true},
    {"fp16_bits_sink_prefill", false, AttnCache::FP16Bits, 3, 1, true, UINT_MAX,
     0.0f, true},
    {"fp16_bits_non_causal", false, AttnCache::FP16Bits, 3, 0, false, UINT_MAX,
     0.0f, false},
  };
  for (const AttnCase &c : cases)
    checkAttention(c, 1e-4f);
}

#ifdef ENABLE_FP16
/**
 * @brief The per-row arm on an FP16 query (the Android fp16 path), over an
 *        FP16 cache and an int8 cache with per-(row, head) scales, decode and
 *        prefill steps, with a window, softcap, sink or non-causal rows. Held
 *        to 1e-2, as the FP16 cases of unittest_mha_core.
 */
TEST_F(ComputeOpsDispatchTest, AttentionFp16QueryPaths) {
  const AttnCase cases[] = {
    {"fp16_softcap_window_prefill", true, AttnCache::FP16Bits, 3, 2, true, 3,
     0.5f, false},
    {"fp16_sink_decode", true, AttnCache::FP16Bits, 1, 3, true, UINT_MAX, 0.0f,
     true},
    {"fp16_sink_prefill", true, AttnCache::FP16Bits, 3, 1, true, UINT_MAX, 0.0f,
     true},
    {"fp16_non_causal", true, AttnCache::FP16Bits, 3, 0, false, UINT_MAX, 0.0f,
     false},
    {"int8_decode", true, AttnCache::Int8, 1, 4, true, UINT_MAX, 0.0f, false},
    {"int8_window_prefill", true, AttnCache::Int8, 3, 2, true, 3, 0.0f, false},
  };
  for (const AttnCase &c : cases)
    checkAttention(c, 1e-2f);
}
#endif

/**
 * @brief The flash arm, allowed by AttentionOpParams::flash: a 40-row step
 *        takes it on every architecture (ARM from 32 rows, x86 for every
 *        step), over an fp16-bits cache, with a window and softcap, and with
 *        non-causal rows. 8 x 8 flash tiles split the step over several query
 *        and key blocks, so the online-softmax rescale across key blocks
 *        runs. The decode row goes to the platform's faster arm, a sink and
 *        an FP32 cache to the per-row arm; each must still match the formula.
 *        x86 keeps the probabilities in fp32 (1e-4); the ARM form stores
 *        them in fp16 (2e-3).
 */
TEST_F(ComputeOpsDispatchTest, AttentionFlashArm) {
  ScopedEnv bq("VJEPA_BQ", "8"), bk("VJEPA_BK", "8");
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  const float tol = 1e-4f;
#else
  const float tol = 2e-3f;
#endif
  const AttnCase cases[] = {
    {"flash_softcap_window_prefill", false, AttnCache::FP16Bits, 40, 3, true, 5,
     0.5f, false, true},
    {"flash_non_causal", false, AttnCache::FP16Bits, 40, 0, false, UINT_MAX,
     0.0f, false, true},
    {"flash_chunk_after_prefill", false, AttnCache::FP16Bits, 40, 40, true,
     UINT_MAX, 0.0f, false, true},
    {"flash_decode_row", false, AttnCache::FP16Bits, 1, 40, true, UINT_MAX,
     0.0f, false, true},
    {"flash_sink_takes_per_row", false, AttnCache::FP16Bits, 40, 2, true,
     UINT_MAX, 0.0f, true, true},
    {"flash_fp32_cache_takes_per_row", false, AttnCache::FP32, 40, 2, true,
     UINT_MAX, 0.0f, false, true},
  };
  for (const AttnCase &c : cases)
    checkAttention(c, tol);
}

#ifdef ENABLE_FP16
/**
 * @brief The flash arm on an FP16 query (the all-FP16 NEON form on ARM, the
 *        widened FP32 form on x86) over an FP16 cache, with a softcap and a
 *        window, and over an int8 cache, which it dequantizes per tile. Held
 *        to 1e-2, as the FP16 cases of unittest_mha_core.
 */
TEST_F(ComputeOpsDispatchTest, AttentionFlashArmFp16Query) {
  ScopedEnv bq("VJEPA_BQ", "8"), bk("VJEPA_BK", "8");
  const AttnCase cases[] = {
    {"flash_fp16_softcap_prefill", true, AttnCache::FP16Bits, 40, 2, true,
     UINT_MAX, 0.1f, false, true},
    {"flash_fp16_window_prefill", true, AttnCache::FP16Bits, 40, 0, true, 7,
     0.0f, false, true},
    {"flash_int8_prefill", true, AttnCache::Int8, 40, 1, true, UINT_MAX, 0.0f,
     false, true},
  };
  for (const AttnCase &c : cases)
    checkAttention(c, 1e-2f);
}
#endif

/**
 * @brief attention rejects a kv_len past the cache rows (a K or V plane
 *        shorter than it), dtypes no arm reads (an FP32 query over an int8
 *        cache, an FP16 query over an FP32 cache, K and V of different dtypes,
 *        an output of another dtype than the query), and a sink of another
 *        size or dtype than num_heads_q values of the query's.
 */
TEST_F(ComputeOpsDispatchTest, AttentionRejectsShortCacheAndBadDtypes) {
  constexpr unsigned int D = 4, R = 2, keys = 3;
  nntrainer::Tensor q = stepRows(R, D, 0.1f);
  nntrainer::Tensor out(1, 1, R, D);
  nntrainer::Tensor key = cachePlane(keys, D), value = cachePlane(keys, D);
  nntrainer::KVCacheView kv;
  kv.key = &key;
  kv.value = &value;
  kv.head_dim = D;
  nntrainer::AttentionOpParams p;
  p.num_heads_q = 1;
  p.q_pos = keys - R;
  p.kv_len = keys;
  EXPECT_NO_THROW(q.getOps()->attention(q, kv, out, p));

  nntrainer::Tensor short_value = cachePlane(keys - 1, D);
  kv.value = &short_value;
  EXPECT_THROW(q.getOps()->attention(q, kv, out, p), std::invalid_argument);
  kv.value = &value;

  nntrainer::Tensor short_key = cachePlane(keys - 1, D);
  kv.key = &short_key;
  EXPECT_THROW(q.getOps()->attention(q, kv, out, p), std::invalid_argument);
  kv.key = &key;

  nntrainer::Tensor sink(1, 1, 1, 1);
  p.sink = &sink;
  EXPECT_NO_THROW(q.getOps()->attention(q, kv, out, p));
  nntrainer::Tensor sink_wide(1, 1, 1, 2);
  p.sink = &sink_wide;
  EXPECT_THROW(q.getOps()->attention(q, kv, out, p), std::invalid_argument);
  nntrainer::Tensor sink16(nntrainer::TensorDim(
    {1, 1, 1, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16}));
  p.sink = &sink16;
  EXPECT_THROW(q.getOps()->attention(q, kv, out, p), std::invalid_argument);
  p.sink = nullptr;

  nntrainer::Tensor fp32_value(1, 1, keys, D);
  kv.value = &fp32_value;
  EXPECT_THROW(q.getOps()->attention(q, kv, out, p), std::invalid_argument);
  kv.value = &value;

  const nntrainer::TensorDim u8(
    {1, 1, keys, D, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT8});
  nntrainer::Tensor key8(u8), value8(u8);
  kv.key = &key8;
  kv.value = &value8;
  EXPECT_THROW(q.getOps()->attention(q, kv, out, p), std::invalid_argument);
  kv.key = &key;
  kv.value = &value;

  nntrainer::Tensor out16(nntrainer::TensorDim(
    {1, 1, R, D, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16}));
  EXPECT_THROW(q.getOps()->attention(q, kv, out16, p), std::invalid_argument);

#ifdef ENABLE_FP16
  const nntrainer::TensorDim f16(
    {1, 1, R, D, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
  nntrainer::Tensor q16(f16), out_f16(f16);
  nntrainer::Tensor fp32_key(1, 1, keys, D), fp32_val(1, 1, keys, D);
  kv.key = &fp32_key;
  kv.value = &fp32_val;
  EXPECT_THROW(q16.getOps()->attention(q16, kv, out_f16, p),
               std::invalid_argument);
  kv.key = &key;
  kv.value = &value;
#endif
}

/**
 * @brief On a KOhwi key plane kv_cache_write puts each head's rotated rows at
 *        that head's stride (head h, row r at (h * height + r) * head_dim),
 *        leaves the other rows alone, and writes V concat as before.
 */
TEST_F(ComputeOpsDispatchTest, KvCacheWriteScattersAKOhwiKeyPlane) {
  constexpr unsigned int R = 2, head_dim = 8, H = 2, W = H * head_dim, pos = 3,
                         height = 6;
  const RotaryRows rows(pos + R, head_dim);
  nntrainer::Tensor k = stepRows(R, W, -1.0f), v = stepRows(R, W, 0.5f);
  nntrainer::Tensor key = cachePlane(height, W), value = cachePlane(height, W);
  nntrainer::KVCacheView kv;
  kv.key = &key;
  kv.value = &value;
  kv.head_dim = head_dim;
  kv.layout = nntrainer::KVCacheLayout::KOhwi;
  k.getOps()->kv_cache_write(k, v, kv, pos, &rows.table);

  for (unsigned int h = 0; h < H; ++h) {
    for (unsigned int r = 0; r < height; ++r) {
      for (unsigned int d = 0; d < head_dim; ++d) {
        const uint16_t kb =
          key.getData<uint16_t>()[((size_t)h * height + r) * head_dim + d];
        if (r < pos || r >= pos + R) {
          EXPECT_EQ(kb, 0xABCD) << "head " << h << " row " << r;
          continue;
        }
        EXPECT_NEAR(nntrainer::compute_fp16_to_fp32(kb),
                    rotatedRef(k.getData<float>(), rows, W, head_dim, pos,
                               r - pos, h * head_dim + d),
                    2e-3f)
          << "head " << h << " row " << r << " dim " << d;
      }
    }
  }
  for (unsigned int r = 0; r < height; ++r) {
    for (unsigned int w = 0; w < W; ++w) {
      const uint16_t vb = value.getData<uint16_t>()[r * W + w];
      if (r < pos || r >= pos + R)
        EXPECT_EQ(vb, 0xABCD) << "row " << r;
      else
        EXPECT_EQ(vb, nntrainer::compute_fp32_to_fp16(
                        v.getData<float>()[(r - pos) * W + w]))
          << "row " << r << " col " << w;
    }
  }
}

/**
 * @brief attention reads a KOhwi key plane as it reads the concat one: the
 *        same rows written in both layouts give the same output, bit for bit.
 */
TEST_F(ComputeOpsDispatchTest, AttentionReadsAKOhwiKeyPlane) {
  constexpr unsigned int R = 5, head_dim = 4, H = 2, W = H * head_dim,
                         Hq = 2 * H, height = 8;
  const RotaryRows rows(R, head_dim);
  nntrainer::Tensor k = stepRows(R, W, -0.6f), v = stepRows(R, W, 0.3f);
  nntrainer::Tensor key_c = cachePlane(height, W),
                    value_c = cachePlane(height, W);
  nntrainer::Tensor key_o = cachePlane(height, W),
                    value_o = cachePlane(height, W);
  nntrainer::KVCacheView concat, ohwi;
  concat.key = &key_c;
  concat.value = &value_c;
  concat.head_dim = head_dim;
  ohwi = concat;
  ohwi.key = &key_o;
  ohwi.value = &value_o;
  ohwi.layout = nntrainer::KVCacheLayout::KOhwi;
  k.getOps()->kv_cache_write(k, v, concat, 0, &rows.table);
  k.getOps()->kv_cache_write(k, v, ohwi, 0, &rows.table);

  nntrainer::Tensor q = stepRows(R, Hq * head_dim, 0.2f);
  nntrainer::Tensor out_c(1, 1, R, Hq * head_dim),
    out_o(1, 1, R, Hq * head_dim);
  nntrainer::AttentionOpParams p;
  p.num_heads_q = Hq;
  p.q_pos = 0;
  p.kv_len = R;
  q.getOps()->attention(q, concat, out_c, p);
  q.getOps()->attention(q, ohwi, out_o, p);
  for (size_t i = 0; i < out_c.size(); ++i)
    EXPECT_EQ(out_c.getData<float>()[i], out_o.getData<float>()[i])
      << "element " << i;
}

/**
 * @brief On an int8 (UINT8) cache kv_cache_write quantizes each (row, head)
 *        of K and V symmetrically to int8 with the fp16 scale amax / 127 at
 *        the same row of the scale planes, leaves the other rows and scales
 *        alone, and refuses the write without scales.
 */
TEST_F(ComputeOpsDispatchTest, KvCacheWriteQuantizesAnInt8Cache) {
  constexpr unsigned int R = 2, head_dim = 4, H = 2, W = H * head_dim, pos = 2,
                         height = 5;
  nntrainer::Tensor k = stepRows(R, W, -1.0f), v = stepRows(R, W, 0.5f);
  const nntrainer::TensorDim u8(
    {1, 1, height, W, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT8});
  nntrainer::Tensor key(u8), value(u8);
  std::fill_n(key.getData<uint8_t>(), height * W, 0x55);
  std::fill_n(value.getData<uint8_t>(), height * W, 0x55);
  std::vector<uint16_t> kscale(height * H, 0xABCD), vscale(height * H, 0xABCD);
  nntrainer::KVCacheView kv;
  kv.key = &key;
  kv.value = &value;
  kv.head_dim = head_dim;
  kv.key_scale = kscale.data();
  kv.value_scale = vscale.data();
  k.getOps()->kv_cache_write(k, v, kv, pos, nullptr);

  auto expectPlane = [&](const nntrainer::Tensor &plane,
                         const std::vector<uint16_t> &scale,
                         const nntrainer::Tensor &in, const char *name) {
    for (unsigned int r = 0; r < height; ++r) {
      for (unsigned int h = 0; h < H; ++h) {
        const int8_t *row =
          reinterpret_cast<const int8_t *>(plane.getData<uint8_t>()) + r * W +
          h * head_dim;
        if (r < pos || r >= pos + R) {
          for (unsigned int d = 0; d < head_dim; ++d)
            EXPECT_EQ(static_cast<uint8_t>(row[d]), 0x55)
              << name << " row " << r;
          EXPECT_EQ(scale[r * H + h], 0xABCD) << name << " row " << r;
          continue;
        }
        float x[head_dim], amax = 0.0f;
        for (unsigned int d = 0; d < head_dim; ++d) {
          x[d] =
            roundFp16(in.getData<float>()[(r - pos) * W + h * head_dim + d]);
          amax = std::max(amax, std::fabs(x[d]));
        }
        EXPECT_EQ(scale[r * H + h],
                  nntrainer::compute_fp32_to_fp16(amax / 127.0f))
          << name << " row " << r << " head " << h;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (unsigned int d = 0; d < head_dim; ++d)
          EXPECT_EQ(static_cast<int>(row[d]),
                    std::max(-127, std::min(127, static_cast<int>(
                                                   std::round(x[d] * inv)))))
            << name << " row " << r << " head " << h << " dim " << d;
      }
    }
  };
  expectPlane(key, kscale, k, "key");
  expectPlane(value, vscale, v, "value");

  kv.key_scale = nullptr;
  EXPECT_THROW(k.getOps()->kv_cache_write(k, v, kv, pos, nullptr),
               std::invalid_argument);
}

#ifdef ENABLE_FP16
/**
 * @brief What kv_cache_write quantizes, attention reads back: an FP16 query
 *        over the int8 cache the op wrote matches the per-row arm over an
 *        fp16 cache holding the dequantized rows, within fp16. (The FP16 step
 *        is written unrotated: the fp32-only table of RotaryRows is for FP32
 *        steps, and the rotation is covered by the other write tests.)
 */
TEST_F(ComputeOpsDispatchTest, AttentionReadsWhatKvCacheWriteQuantized) {
  constexpr unsigned int R = 4, head_dim = 4, H = 2, W = H * head_dim,
                         Hq = 2 * H;
  std::vector<float> kf(R * W), vf(R * W), qf(R * Hq * head_dim);
  for (size_t i = 0; i < kf.size(); ++i) {
    kf[i] = roundFp16(0.9f * std::sin(0.31f * i));
    vf[i] = roundFp16(0.7f * std::cos(0.17f * i + 0.5f));
  }
  for (size_t i = 0; i < qf.size(); ++i)
    qf[i] = roundFp16(0.5f * std::sin(0.23f * i + 1.0f));
  nntrainer::Tensor k = makeTensor(R, W, nntrainer::Tdatatype::FP16, kf);
  nntrainer::Tensor v = makeTensor(R, W, nntrainer::Tdatatype::FP16, vf);
  nntrainer::Tensor q =
    makeTensor(R, Hq * head_dim, nntrainer::Tdatatype::FP16, qf);

  const nntrainer::TensorDim u8(
    {1, 1, R, W, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT8});
  nntrainer::Tensor key8(u8), value8(u8);
  std::vector<uint16_t> kscale(R * H), vscale(R * H);
  nntrainer::KVCacheView int8;
  int8.key = &key8;
  int8.value = &value8;
  int8.head_dim = head_dim;
  int8.key_scale = kscale.data();
  int8.value_scale = vscale.data();
  k.getOps()->kv_cache_write(k, v, int8, 0, nullptr);

  // The dequantized rows, as an fp16 cache for the reference run.
  std::vector<float> kd(R * W), vd(R * W);
  for (unsigned int r = 0; r < R; ++r)
    for (unsigned int h = 0; h < H; ++h)
      for (unsigned int d = 0; d < head_dim; ++d) {
        const size_t i = r * W + h * head_dim + d;
        kd[i] = reinterpret_cast<const int8_t *>(key8.getData<uint8_t>())[i] *
                nntrainer::compute_fp16_to_fp32(kscale[r * H + h]);
        vd[i] = reinterpret_cast<const int8_t *>(value8.getData<uint8_t>())[i] *
                nntrainer::compute_fp16_to_fp32(vscale[r * H + h]);
      }
  nntrainer::Tensor key16 = makeTensor(R, W, nntrainer::Tdatatype::FP16, kd);
  nntrainer::Tensor value16 = makeTensor(R, W, nntrainer::Tdatatype::FP16, vd);
  nntrainer::KVCacheView fp16;
  fp16.key = &key16;
  fp16.value = &value16;
  fp16.head_dim = head_dim;

  nntrainer::Tensor out8(
    nntrainer::TensorDim({1, 1, R, Hq * head_dim, nntrainer::Tformat::NCHW,
                          nntrainer::Tdatatype::FP16}));
  nntrainer::Tensor out16(out8.getDim());
  nntrainer::AttentionOpParams p;
  p.num_heads_q = Hq;
  p.q_pos = 0;
  p.kv_len = R;
  q.getOps()->attention(q, int8, out8, p);
  q.getOps()->attention(q, fp16, out16, p);
  for (size_t i = 0; i < out8.size(); ++i)
    EXPECT_NEAR(nntrainer::compute_fp16_to_fp32(out8.getData<uint16_t>()[i]),
                nntrainer::compute_fp16_to_fp32(out16.getData<uint16_t>()[i]),
                1e-2f)
      << "element " << i;
}
#endif

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
