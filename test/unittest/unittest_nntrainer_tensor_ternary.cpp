// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_nntrainer_tensor_ternary.cpp
 * @date 02 October 2026
 * @brief Unit tests for the TQ2CX ternary weight tensor (TernaryTensor)
 * @see https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 *
 * @details Covers the three things the scaffolding has to get right before any
 * kernel work starts:
 *   1. the layout arithmetic is a pure function of the dims (no padded /
 *      trimmed ambiguity, which is the QS4CX trap at qs4cx_tensor.h:204-220);
 *   2. pack -> unpack round-trips losslessly for a genuinely ternary source;
 *   3. the reference dot agrees with a dense float GEMM, both through the
 *      weight-as-receiver TernaryTensor::dot and through the
 *      activation-as-receiver FloatTensor::dot dispatch an FC layer takes.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include <cpu_backend.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <ternary_tensor.h>
#include <tq2cx_kernels.h>
#include <tq2cx_layout.h>

namespace {

constexpr float TOLERANCE = 1e-4f;

using TD = nntrainer::TensorDim;

/**
 * @brief Build a dense [N, K] weight whose every value is in {-s, 0, +s} with
 *        a per-row scale s, at the measured Gemma4 code distribution
 *        (~25.4% +1 / 49.2% 0 / 25.3% -1).
 * @param[out] scales per-row scale actually used
 */
std::vector<float> makeTernaryDense(size_t N, size_t K,
                                    std::vector<float> &scales,
                                    unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::uniform_real_distribution<float> sdist(0.01f, 0.5f);

  std::vector<float> dense(N * K);
  scales.resize(N);
  for (size_t n = 0; n < N; ++n) {
    const float s = sdist(rng);
    scales[n] = s;
    bool saw_max = false;
    for (size_t k = 0; k < K; ++k) {
      const float r = u(rng);
      int code = (r < 0.254f) ? 1 : ((r < 0.746f) ? 0 : -1);
      dense[n * K + k] = static_cast<float>(code) * s;
      saw_max |= (code != 0);
    }
    // tq2cx_pack_ref_fp32 derives the scale as max|row|, so the row has to
    // contain at least one +-s for the recovered scale to equal s.
    if (!saw_max) {
      dense[n * K] = s;
    }
  }
  return dense;
}

/**
 * @brief Dense fp32 reference GEMM, out[M,N] = lhs[M,K] * W[K,N] with W given
 *        as [N,K] row major. Accumulates in the SAME order as
 *        tq2cx_gemm_ref_fp32 so the comparison can be tight.
 */
std::vector<float> denseGemm(size_t M, size_t N, size_t K,
                             const std::vector<float> &lhs,
                             const std::vector<float> &w_nk) {
  std::vector<float> out(M * N, 0.0f);
  for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < M; ++m) {
      float acc = 0.0f;
      for (size_t k = 0; k < K; ++k)
        acc += lhs[m * K + k] * w_nk[n * K + k];
      out[m * N + n] = acc;
    }
  }
  return out;
}

} // namespace

//==============================================================================
// Layout arithmetic
//==============================================================================

TEST(TernaryTensor, layout_is_a_pure_function_of_dims) {
  // One stride expression, derived from the dims alone -- the property QS4CX
  // lacks. 32 taps per 8-byte unit, no row padding, OUT fp32 scales per
  // expert record.
  EXPECT_EQ(nntrainer::TernaryTensor::unitsPerRow(32), 1u);
  EXPECT_EQ(nntrainer::TernaryTensor::unitsPerRow(33), 2u);
  EXPECT_EQ(nntrainer::TernaryTensor::unitsPerRow(2816), 88u);
  EXPECT_EQ(nntrainer::TernaryTensor::unitsPerRow(704), 22u);

  EXPECT_EQ(nntrainer::TernaryTensor::rowBytes(2816), 704u);
  EXPECT_EQ(nntrainer::TernaryTensor::rowBytes(704), 176u);
  // No padding: a row is always a whole number of 8-byte units.
  EXPECT_EQ(nntrainer::TernaryTensor::rowBytes(2816) % 8u, 0u);

  // Exactly 2.00 bits/weight for the mask half.
  EXPECT_EQ(nntrainer::TernaryTensor::maskBytes(2816, 1408),
            (size_t)1408 * 2816 / 4);
  EXPECT_EQ(nntrainer::TernaryTensor::recordBytes(2816, 1408),
            (size_t)1408 * 2816 / 4 + (size_t)1408 * sizeof(float));
}

TEST(TernaryTensor, byte_math_for_the_gemma4_shapes) {
  // gate_up_proj [E=128, OUT=1408, IN=2816]
  EXPECT_EQ(nntrainer::TernaryTensor::totalBytes(128, 2816, 1408), 127598592u);
  // down_proj [E=128, OUT=2816, IN=704]
  EXPECT_EQ(nntrainer::TernaryTensor::totalBytes(128, 704, 2816), 64880640u);
  // One decoder layer's pair, and all 30 layers.
  const size_t per_layer =
    nntrainer::TernaryTensor::totalBytes(128, 2816, 1408) +
    nntrainer::TernaryTensor::totalBytes(128, 704, 2816);
  EXPECT_EQ(per_layer, 192479232u);
  EXPECT_EQ(per_layer * 30, 5774376960u);
}

TEST(TernaryTensor, size_matches_the_static_arithmetic) {
  // [batch=1, channel=E, height=IN, width=OUT]
  TD dim(1, 3, 64, 5, {TD::Format::NCHW, TD::DataType::TQ2CX});
  nntrainer::TernaryTensor t(dim, true);

  EXPECT_EQ(t.height(), 64u);
  EXPECT_EQ(t.width(), 5u);
  EXPECT_EQ(t.channel(), 3u);
  EXPECT_EQ(t.size(), nntrainer::TernaryTensor::totalBytes(3, 64, 5));
  EXPECT_EQ(t.getMemoryBytes(), t.size());
  EXPECT_EQ(t.scale_size(), 15u); // E * OUT
  EXPECT_EQ(t.q_scheme(), nntrainer::QScheme::TQ2CX);
}

TEST(TernaryTensor, batch_greater_than_one_is_rejected) {
  TD dim(2, 1, 64, 4, {TD::Format::NCHW, TD::DataType::TQ2CX});
  EXPECT_THROW(nntrainer::TernaryTensor(dim, true), std::invalid_argument);
}

TEST(TernaryTensor, datatype_size_is_one_byte) {
  // size() already returns BYTES, so bytes() == size() * 1 must hold or every
  // save/read offset is wrong.
  TD dim(1, 1, 64, 4, {TD::Format::NCHW, TD::DataType::TQ2CX});
  EXPECT_EQ(dim.getDataTypeSize(), 1u);
}

TEST(TernaryTensor, setZero_gives_an_all_zero_weight) {
  TD dim(1, 2, 96, 4, {TD::Format::NCHW, TD::DataType::TQ2CX});
  nntrainer::TernaryTensor t(dim, true);
  t.setZero();

  // Zero is a representable code (both planes clear), so this is a real zero
  // weight and not merely a cleared buffer.
  EXPECT_TRUE(t.validate());
  for (size_t e = 0; e < 2; ++e)
    for (size_t n = 0; n < 4; ++n)
      for (size_t k = 0; k < 96; ++k)
        EXPECT_FLOAT_EQ(t.decode(e, n, k), 0.0f);
  EXPECT_FLOAT_EQ(t.max_abs(), 0.0f);
}

TEST(TernaryTensor, initialize_zeroes_the_record) {
  TD dim(1, 1, 64, 3, {TD::Format::NCHW, TD::DataType::TQ2CX});
  nntrainer::TernaryTensor t(dim, true);
  t.initialize();
  EXPECT_TRUE(t.validate());
}

//==============================================================================
// pack -> unpack round trip
//==============================================================================

TEST(TernaryTensor, pack_unpack_round_trips_losslessly) {
  // K is deliberately NOT a multiple of 32 so the tail-bit invariant is
  // exercised.
  constexpr size_t N = 7, K = 100;

  std::vector<float> scales;
  auto dense = makeTernaryDense(N, K, scales, 1234u);

  std::vector<uint8_t> record(nntrainer::TernaryTensor::recordBytes(K, N));
  nntrainer::tq2cx_pack_ref_fp32(N, K, dense.data(), K, record.data());

  TD dim(1, 1, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX});
  nntrainer::TernaryTensor t(dim, record.data());

  // format invariants: (pos & neg) == 0, and the 28 dead bits of the last unit
  // of every row are clear in both planes
  EXPECT_TRUE(t.validate());

  // scales recovered exactly (max|row| of a +-s row is s)
  for (size_t n = 0; n < N; ++n)
    EXPECT_FLOAT_EQ(t.getScaleOfExpert(0)[n], scales[n]);

  // every weight recovered exactly -- no tolerance, this is lossless
  for (size_t n = 0; n < N; ++n)
    for (size_t k = 0; k < K; ++k)
      EXPECT_FLOAT_EQ(t.decode(0, n, k), dense[n * K + k])
        << "n=" << n << " k=" << k;
}

TEST(TernaryTensor, illegal_both_planes_set_is_detected) {
  constexpr size_t N = 2, K = 64;
  std::vector<uint8_t> record(nntrainer::TernaryTensor::recordBytes(K, N), 0);
  auto *units = reinterpret_cast<nntrainer::TQ2CX_MaskUnit *>(record.data());

  TD dim(1, 1, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX});

  units[0].pos = 0x1u;
  units[0].neg = 0x1u; // the (1,1) combination: not a code
  nntrainer::TernaryTensor bad(dim, record.data());
  EXPECT_FALSE(bad.validate());

  units[0].neg = 0x2u; // disjoint again
  nntrainer::TernaryTensor good(dim, record.data());
  EXPECT_TRUE(good.validate());
}

TEST(TernaryTensor, dirty_tail_bits_are_detected) {
  constexpr size_t N = 1, K = 40; // 2 units, 24 dead bits in the second
  std::vector<uint8_t> record(nntrainer::TernaryTensor::recordBytes(K, N), 0);
  auto *units = reinterpret_cast<nntrainer::TQ2CX_MaskUnit *>(record.data());

  TD dim(1, 1, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX});

  units[1].pos = 1u << 8; // tap 40: past the end
  nntrainer::TernaryTensor bad(dim, record.data());
  EXPECT_FALSE(bad.validate());

  units[1].pos = 1u << 7; // tap 39: the last real tap
  nntrainer::TernaryTensor good(dim, record.data());
  EXPECT_TRUE(good.validate());
}

TEST(TernaryTensor, stacked_expert_record_is_a_pure_subrange) {
  // The whole point of the per-expert record layout: expert e's bytes are
  // byte-identical to a standalone single-expert record, so an MoE runtime
  // that wants 2D per-expert weights slices with pointer arithmetic only.
  constexpr size_t E = 4, N = 5, K = 96;

  const size_t rec = nntrainer::TernaryTensor::recordBytes(K, N);
  std::vector<uint8_t> stacked(E * rec);
  std::vector<std::vector<float>> dense(E);
  std::vector<std::vector<float>> scales(E);

  for (size_t e = 0; e < E; ++e) {
    dense[e] = makeTernaryDense(N, K, scales[e], 7000u + (unsigned)e);
    nntrainer::tq2cx_pack_ref_fp32(N, K, dense[e].data(), K,
                                   stacked.data() + e * rec);
  }

  TD sdim(1, E, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX});
  nntrainer::TernaryTensor st(sdim, stacked.data());
  EXPECT_EQ(st.size(), E * rec);
  EXPECT_TRUE(st.validate());

  for (size_t e = 0; e < E; ++e) {
    // byte-for-byte equal to the standalone record
    TD one(1, 1, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX});
    nntrainer::TernaryTensor solo(one, stacked.data() + e * rec);
    EXPECT_EQ(0, std::memcmp(st.expertRecord(e), solo.getData(), rec));

    for (size_t n = 0; n < N; ++n) {
      EXPECT_FLOAT_EQ(st.getScaleOfExpert(e)[n], scales[e][n]);
      // flat getScale(idx) must agree with the per-expert view
      EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(st.getScale(e * N + n)),
                      scales[e][n]);
      for (size_t k = 0; k < K; ++k)
        EXPECT_FLOAT_EQ(st.decode(e, n, k), dense[e][n * K + k]);
    }
  }
}

//==============================================================================
// save / read round trip
//==============================================================================

TEST(TernaryTensor, save_read_round_trip) {
  constexpr size_t E = 2, N = 6, K = 96;
  const size_t rec = nntrainer::TernaryTensor::recordBytes(K, N);

  std::vector<uint8_t> stacked(E * rec);
  std::vector<std::vector<float>> dense(E);
  std::vector<std::vector<float>> scales(E);
  for (size_t e = 0; e < E; ++e) {
    dense[e] = makeTernaryDense(N, K, scales[e], 99u + (unsigned)e);
    nntrainer::tq2cx_pack_ref_fp32(N, K, dense[e].data(), K,
                                   stacked.data() + e * rec);
  }

  TD dim(1, E, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX});
  nntrainer::TernaryTensor src(dim, stacked.data());

  const char *path = "tq2cx_roundtrip.bin";
  {
    std::ofstream f(path, std::ios::out | std::ios::binary);
    ASSERT_TRUE(f.good());
    src.save(f);
  }

  // The on-disk record must be exactly size() bytes: no qparam header, no
  // padding, nothing version dependent.
  {
    std::ifstream f(path, std::ios::in | std::ios::binary | std::ios::ate);
    ASSERT_TRUE(f.good());
    EXPECT_EQ(static_cast<size_t>(f.tellg()), src.size());
  }

  nntrainer::TernaryTensor dst(dim, true);
  {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    ASSERT_TRUE(f.good());
    dst.read(f, 0, false);
  }
  std::remove(path);

  EXPECT_TRUE(dst.validate());
  EXPECT_EQ(0, std::memcmp(src.getData(), dst.getData(), src.size()));
  for (size_t e = 0; e < E; ++e)
    for (size_t n = 0; n < N; ++n)
      for (size_t k = 0; k < K; ++k)
        EXPECT_FLOAT_EQ(dst.decode(e, n, k), dense[e][n * K + k]);
}

//==============================================================================
// reference dot vs dense float GEMM
//==============================================================================

TEST(TernaryTensor, gemm_ref_matches_dense_float_gemv) {
  constexpr size_t M = 1, N = 9, K = 160; // M == 1: the decode-step GEMV

  std::vector<float> scales;
  auto dense = makeTernaryDense(N, K, scales, 4242u);

  std::vector<uint8_t> record(nntrainer::TernaryTensor::recordBytes(K, N));
  nntrainer::tq2cx_pack_ref_fp32(N, K, dense.data(), K, record.data());

  std::mt19937 rng(5u);
  std::uniform_real_distribution<float> a(-1.0f, 1.0f);
  std::vector<float> lhs(M * K);
  for (auto &v : lhs)
    v = a(rng);

  auto expected = denseGemm(M, N, K, lhs, dense);

  std::vector<float> got(M * N, 0.0f);
  nntrainer::tq2cx_gemm_ref_fp32(M, N, K, lhs.data(), K, record.data(),
                                 got.data(), N, 0.0f);

  for (size_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(got[i], expected[i], TOLERANCE) << "i=" << i;
}

TEST(TernaryTensor, gemm_ref_matches_dense_float_gemm) {
  constexpr size_t M = 5, N = 9, K = 100; // K % 32 != 0

  std::vector<float> scales;
  auto dense = makeTernaryDense(N, K, scales, 777u);

  std::vector<uint8_t> record(nntrainer::TernaryTensor::recordBytes(K, N));
  nntrainer::tq2cx_pack_ref_fp32(N, K, dense.data(), K, record.data());

  std::mt19937 rng(11u);
  std::uniform_real_distribution<float> a(-2.0f, 2.0f);
  std::vector<float> lhs(M * K);
  for (auto &v : lhs)
    v = a(rng);

  auto expected = denseGemm(M, N, K, lhs, dense);

  std::vector<float> got(M * N, 0.0f);
  nntrainer::tq2cx_gemm_ref_fp32(M, N, K, lhs.data(), K, record.data(),
                                 got.data(), N, 0.0f);

  for (size_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(got[i], expected[i], TOLERANCE) << "i=" << i;
}

TEST(TernaryTensor, gemm_ref_honours_beta) {
  constexpr size_t M = 2, N = 4, K = 64;

  std::vector<float> scales;
  auto dense = makeTernaryDense(N, K, scales, 31u);
  std::vector<uint8_t> record(nntrainer::TernaryTensor::recordBytes(K, N));
  nntrainer::tq2cx_pack_ref_fp32(N, K, dense.data(), K, record.data());

  std::vector<float> lhs(M * K, 1.0f);
  auto expected = denseGemm(M, N, K, lhs, dense);

  std::vector<float> got(M * N, 3.0f);
  nntrainer::tq2cx_gemm_ref_fp32(M, N, K, lhs.data(), K, record.data(),
                                 got.data(), N, 2.0f);

  for (size_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(got[i], expected[i] + 6.0f, TOLERANCE) << "i=" << i;
}

TEST(TernaryTensor, dot_through_the_FloatTensor_dispatch) {
  // The path an FC layer takes: activation.dot(weight), which lands in
  // FloatTensor::dot's switch on the WEIGHT's dtype.
  constexpr size_t M = 3, N = 8, K = 96;

  std::vector<float> scales;
  auto dense = makeTernaryDense(N, K, scales, 606u);
  std::vector<uint8_t> record(nntrainer::TernaryTensor::recordBytes(K, N));
  nntrainer::tq2cx_pack_ref_fp32(N, K, dense.data(), K, record.data());

  std::mt19937 rng(13u);
  std::uniform_real_distribution<float> a(-1.0f, 1.0f);
  std::vector<float> lhs(M * K);
  for (auto &v : lhs)
    v = a(rng);
  auto expected = denseGemm(M, N, K, lhs, dense);

  nntrainer::Tensor act(TD(1, 1, M, K, {TD::Format::NCHW, TD::DataType::FP32}));
  act.allocate();
  std::memcpy(act.getData<float>(), lhs.data(), lhs.size() * sizeof(float));

  // weight TensorDim is the logical FC weight [K=in, N=out]; the bytes are the
  // transpose [N][K], same convention QS4CX uses.
  nntrainer::Tensor w(TD(1, 1, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX}),
                      record.data());

  nntrainer::Tensor out(TD(1, 1, M, N, {TD::Format::NCHW, TD::DataType::FP32}));
  out.allocate();
  out.setZero();

  act.dot(w, out, false, false, 0.0f);
  for (size_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(out.getData<float>()[i], expected[i], TOLERANCE) << "i=" << i;
}

TEST(TernaryTensor, dot_weight_as_receiver) {
  // The BCQTensor-style form, weight.dot(activation) -- same kernel.
  constexpr size_t M = 2, N = 6, K = 64;

  std::vector<float> scales;
  auto dense = makeTernaryDense(N, K, scales, 808u);
  std::vector<uint8_t> record(nntrainer::TernaryTensor::recordBytes(K, N));
  nntrainer::tq2cx_pack_ref_fp32(N, K, dense.data(), K, record.data());

  std::vector<float> lhs(M * K, 0.5f);
  auto expected = denseGemm(M, N, K, lhs, dense);

  nntrainer::Tensor act(TD(1, 1, M, K, {TD::Format::NCHW, TD::DataType::FP32}));
  act.allocate();
  std::fill(act.getData<float>(), act.getData<float>() + M * K, 0.5f);

  nntrainer::Tensor w(TD(1, 1, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX}),
                      record.data());
  nntrainer::Tensor out(TD(1, 1, M, N, {TD::Format::NCHW, TD::DataType::FP32}));
  out.allocate();
  out.setZero();

  w.dot(act, out, false, false, 0.0f);
  for (size_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(out.getData<float>()[i], expected[i], TOLERANCE) << "i=" << i;
}

TEST(TernaryTensor, dot_rejects_transpose_and_stacked_experts) {
  constexpr size_t N = 4, K = 64;
  std::vector<uint8_t> record(nntrainer::TernaryTensor::recordBytes(K, N), 0);

  nntrainer::Tensor act(TD(1, 1, 1, K, {TD::Format::NCHW, TD::DataType::FP32}));
  act.allocate();
  act.setZero();
  nntrainer::Tensor out(TD(1, 1, 1, N, {TD::Format::NCHW, TD::DataType::FP32}));
  out.allocate();
  out.setZero();

  nntrainer::Tensor w(TD(1, 1, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX}),
                      record.data());
  EXPECT_THROW(act.dot(w, out, true, false, 0.0f), std::invalid_argument);
  EXPECT_THROW(act.dot(w, out, false, true, 0.0f), std::invalid_argument);

  // A stacked weight has no single operand at this boundary; it must refuse
  // rather than silently use expert 0.
  std::vector<uint8_t> stacked(2 * nntrainer::TernaryTensor::recordBytes(K, N),
                               0);
  nntrainer::Tensor ws(TD(1, 2, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX}),
                       stacked.data());
  EXPECT_THROW(act.dot(ws, out, false, false, 0.0f), std::invalid_argument);
}

//==============================================================================
// registration surface
//==============================================================================

TEST(TernaryTensor, dtype_string_round_trips_through_the_property_table) {
  // TensorDataTypeInfo is what every ini/json "weight_dtype" goes through.
  TD dim(1, 1, 64, 4, {TD::Format::NCHW, TD::DataType::TQ2CX});
  std::stringstream ss;
  ss << dim;
  EXPECT_NE(ss.str().find("TQ2CX"), std::string::npos);
}

//==============================================================================
// batched TQ2CX GEMM (ComputeOps / arch tiles)
//==============================================================================

/**
 * @brief shapes for the batch kernel tests: {M, K, N}
 */
class TernaryBatchKernelTest
  : public ::testing::TestWithParam<
      std::tuple<unsigned int, unsigned int, unsigned int>> {};

/**
 * @brief pack -> validate -> batched dot. Every dispatch path must give
 *        exactly the bits of the single-threaded scalar reference tile, the
 *        result must not change between runs, and it must stay within 1e-5
 *        (relative, RMS) of a double precision dense product.
 */
TEST_P(TernaryBatchKernelTest, matches_scalar_reference_p) {
  const unsigned int M = std::get<0>(GetParam());
  const unsigned int K = std::get<1>(GetParam());
  const unsigned int N = std::get<2>(GetParam());

  std::vector<float> scales;
  const auto dense = makeTernaryDense(N, K, scales, 7 + K + N);
  const TD wdim(1, 1, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX});
  std::vector<uint8_t> record(nntrainer::TernaryTensor::recordBytes(K, N));
  nntrainer::tq2cx_pack_ref_fp32(N, K, dense.data(), K, record.data());
  nntrainer::TernaryTensor packed(wdim, record.data());
  ASSERT_TRUE(packed.validate());
  nntrainer::Tensor w(wdim, record.data());

  std::mt19937 rng(31 + M);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  nntrainer::Tensor act(1, 1, M, K);
  for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i)
    act.getData<float>()[i] = nd(rng);
  const float *A = act.getData<float>();

  const size_t len = static_cast<size_t>(M) * N;
  std::vector<float> c_ref(len, -1.0f), c_arch(len, -2.0f), c_ops(len, -3.0f),
    c_again(len, -4.0f), c_oracle(len, 0.0f);
  std::vector<void *> Ws = {w.getData()};
  const std::vector<unsigned int> Ns = {N};
  auto out = [](std::vector<float> &v) {
    return std::vector<float *>{v.data()};
  };

  nntrainer::tq2cx_gemm_batch(M, Ns, K, A, K, Ws, out(c_ref), Ns,
                              nntrainer::tq2cx_tile_ref, false);
  nntrainer::gemm_tq2cx_batch(M, Ns, K, A, K, Ws, out(c_arch), Ns);
  nntrainer::gemm_tq2cx_batch(M, Ns, K, A, K, Ws, out(c_again), Ns);
  auto *ops = nntrainer::getComputeOps();
  ASSERT_NE(ops, nullptr);
  ASSERT_TRUE(ops->supports_gemm_tq2cx_batch_fp32());
  ops->gemm_tq2cx_batch_fp32(Ws, const_cast<float *>(A), out(c_ops), M, Ns, K);
  nntrainer::tq2cx_gemm_ref_fp32(M, N, K, A, K, w.getData(), c_oracle.data(), N,
                                 0.0f);

  const size_t bytes = len * sizeof(float);
  EXPECT_EQ(std::memcmp(c_ref.data(), c_arch.data(), bytes), 0)
    << "arch tile differs from the scalar reference tile";
  EXPECT_EQ(std::memcmp(c_arch.data(), c_again.data(), bytes), 0)
    << "two runs of the threaded kernel differ";
  EXPECT_EQ(std::memcmp(c_ref.data(), c_ops.data(), bytes), 0)
    << "ComputeOps dispatch differs from the scalar reference tile";
  EXPECT_EQ(std::memcmp(c_ref.data(), c_oracle.data(), bytes), 0)
    << "batch reference differs from tq2cx_gemm_ref_fp32";

  // the FC-layer path: activation.dot(weight)
  nntrainer::Tensor y(1, 1, M, N);
  act.dot(w, y, false, false, 0.0f);
  EXPECT_EQ(std::memcmp(c_ref.data(), y.getData<float>(), bytes), 0)
    << "FloatTensor::dot differs from the scalar reference tile";

  double num = 0.0, den = 0.0;
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double r = 0.0;
      for (unsigned int k = 0; k < K; ++k)
        r +=
          static_cast<double>(A[(size_t)m * K + k]) * dense[(size_t)n * K + k];
      const double d = r - c_ref[(size_t)m * N + n];
      num += d * d;
      den += r * r;
    }
  }
  EXPECT_LE(std::sqrt(num / den), 1e-5);
}

INSTANTIATE_TEST_SUITE_P(ExpertShapes, TernaryBatchKernelTest,
                         ::testing::Combine(::testing::Values(1u, 2u, 8u, 33u),
                                            ::testing::Values(704u, 2816u),
                                            ::testing::Values(704u, 2816u)));

INSTANTIATE_TEST_SUITE_P(OddShapes, TernaryBatchKernelTest,
                         ::testing::Values(std::make_tuple(5u, 100u, 13u),
                                           std::make_tuple(3u, 40u, 24u),
                                           std::make_tuple(17u, 64u, 57u)));

//==============================================================================
// stored layout v2 ("t2r16v2")
//==============================================================================

namespace {

/**
 * @brief Run with the process's stored TQ2CX layout set to @a l, restore on
 *        scope exit.
 */
struct StoredLayoutScope {
  nntrainer::Tq2cxLayout saved; /**< layout to restore */
  explicit StoredLayoutScope(const nntrainer::Tq2cxLayout &l) :
    saved(nntrainer::tq2cxStoredLayout()) {
    nntrainer::setTq2cxStoredLayout(l);
  }
  ~StoredLayoutScope() { nntrainer::setTq2cxStoredLayout(saved); }
};

} // namespace

TEST(TernaryLayout, ids_parse_and_print) {
  nntrainer::Tq2cxLayout l;
  ASSERT_TRUE(nntrainer::tq2cxLayoutFromName("t2r16v2", l));
  EXPECT_EQ(l, nntrainer::TQ2CX_LAYOUT_T2R16V2);
  EXPECT_EQ(nntrainer::tq2cxLayoutName(l), "t2r16v2");
  ASSERT_TRUE(nntrainer::tq2cxLayoutFromName("", l));
  EXPECT_EQ(l, nntrainer::TQ2CX_LAYOUT_RM);
  ASSERT_TRUE(nntrainer::tq2cxLayoutFromName("rm", l));
  EXPECT_EQ(nntrainer::tq2cxLayoutName(l), "rm");
  EXPECT_FALSE(nntrainer::tq2cxLayoutFromName("t2r32v4", l));
  EXPECT_FALSE(nntrainer::tq2cxLayoutFromName("b3r16v2", l));
  EXPECT_EQ(nntrainer::tq2cxStoredLayout(), nntrainer::TQ2CX_LAYOUT_RM);
}

TEST(TernaryLayout, record_sizes) {
  const auto v2 = nntrainer::TQ2CX_LAYOUT_T2R16V2;
  // the two expert shapes: 16 | N and 64-byte records, so v2 == v1 in size
  EXPECT_EQ(nntrainer::tq2cxRecordBytes(v2, 704, 2816), 498432u);
  EXPECT_EQ(nntrainer::tq2cxRecordBytes(v2, 2816, 704), 506880u);
  EXPECT_EQ(nntrainer::TernaryTensor::recordBytes(2816, 704, v2),
            nntrainer::TernaryTensor::recordBytes(2816, 704));
  // N = 13 pads to one 16-row tile, scales after the padded masks, record
  // padded to 64 bytes
  EXPECT_EQ(nntrainer::tq2cxMaskBytes(v2, 13, 100), 16u * 4 * 8);
  EXPECT_EQ(nntrainer::tq2cxRecordBytes(v2, 13, 100), 576u);
  EXPECT_EQ(nntrainer::tq2cxRecordBytes(nntrainer::TQ2CX_LAYOUT_RM, 13, 100),
            13u * 4 * 8 + 13 * 4);
}

TEST(TernaryLayout, unit_index_is_the_spec) {
  // t2r16v2, N = 20 (two tiles), U = 5 (blocks of 2, 2, 1)
  const auto v2 = nntrainer::TQ2CX_LAYOUT_T2R16V2;
  const size_t U = 5;
  // tile 0, block 0: rows 0..15 x units 0, 1
  EXPECT_EQ(nntrainer::tq2cxUnitIndex(v2, 0, 0, U), 0u);
  EXPECT_EQ(nntrainer::tq2cxUnitIndex(v2, 0, 1, U), 1u);
  EXPECT_EQ(nntrainer::tq2cxUnitIndex(v2, 1, 0, U), 2u);
  EXPECT_EQ(nntrainer::tq2cxUnitIndex(v2, 15, 1, U), 31u);
  // block 1 starts after 16 x 2 units; tail block 2 is one unit wide
  EXPECT_EQ(nntrainer::tq2cxUnitIndex(v2, 0, 2, U), 32u);
  EXPECT_EQ(nntrainer::tq2cxUnitIndex(v2, 3, 4, U), 64u + 3u);
  // tile 1 starts at 16 * U
  EXPECT_EQ(nntrainer::tq2cxUnitIndex(v2, 16, 0, U), 80u);
  EXPECT_EQ(nntrainer::tq2cxUnitIndex(v2, 19, 4, U), 80u + 64u + 3u);
  std::vector<int> hit(32 * U, 0);
  for (size_t n = 0; n < 20; ++n)
    for (size_t u = 0; u < U; ++u)
      ++hit[nntrainer::tq2cxUnitIndex(v2, n, u, U)];
  for (size_t i = 0; i < hit.size(); ++i)
    EXPECT_LE(hit[i], 1) << "unit slot " << i << " used twice";
}

/**
 * @brief shapes for the v2 tests: {M, K, N}
 */
class TernaryLayoutV2Test
  : public ::testing::TestWithParam<
      std::tuple<unsigned int, unsigned int, unsigned int>> {};

/**
 * @brief v1 record -> v2 -> v1 is the identity, and every kernel reading the
 *        v2 record (scalar reference, arch tile, ComputeOps, the FC dot)
 *        returns exactly the bits of the scalar reference over the v1 record:
 *        the layout moves units, not values or the summation order.
 */
TEST_P(TernaryLayoutV2Test, v2_kernels_match_row_major_reference_p) {
  const unsigned int M = std::get<0>(GetParam());
  const unsigned int K = std::get<1>(GetParam());
  const unsigned int N = std::get<2>(GetParam());
  const auto rm = nntrainer::TQ2CX_LAYOUT_RM;
  const auto v2 = nntrainer::TQ2CX_LAYOUT_T2R16V2;

  std::vector<float> scales;
  const auto dense = makeTernaryDense(N, K, scales, 11 + K + 3 * N);
  std::vector<uint8_t> rec1(nntrainer::tq2cxRecordBytes(rm, N, K));
  nntrainer::tq2cx_pack_ref_fp32(N, K, dense.data(), K, rec1.data(), rm);

  // converter round trip; the direct v2 pack writes the same bytes
  std::vector<uint8_t> rec2(nntrainer::tq2cxRecordBytes(v2, N, K), 0xAB);
  nntrainer::tq2cxRelayout(rm, v2, rec1.data(), rec2.data(), N, K);
  std::vector<uint8_t> back(rec1.size(), 0xCD);
  nntrainer::tq2cxRelayout(v2, rm, rec2.data(), back.data(), N, K);
  ASSERT_EQ(back, rec1);
  std::vector<uint8_t> packed2(rec2.size(), 0x11);
  nntrainer::tq2cx_pack_ref_fp32(N, K, dense.data(), K, packed2.data(), v2);
  ASSERT_EQ(packed2, rec2);

  std::mt19937 rng(57 + M);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  nntrainer::Tensor act(1, 1, M, K);
  for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i)
    act.getData<float>()[i] = nd(rng);
  const float *A = act.getData<float>();
  const size_t len = static_cast<size_t>(M) * N, bytes = len * sizeof(float);
  const std::vector<unsigned int> Ns = {N};
  auto out = [](std::vector<float> &v) {
    return std::vector<float *>{v.data()};
  };

  // the expected bits: scalar reference tile over the row-major record
  std::vector<float> c_ref(len, -1.0f);
  {
    std::vector<void *> W1 = {rec1.data()};
    nntrainer::tq2cx_gemm_batch(M, Ns, K, A, K, W1, out(c_ref), Ns,
                                nntrainer::tq2cx_tile_ref, false, rm);
  }

  StoredLayoutScope scope(v2);
  const TD wdim(1, 1, K, N, {TD::Format::NCHW, TD::DataType::TQ2CX});
  nntrainer::Tensor w(wdim, rec2.data());
  ASSERT_EQ(w.bytes(), rec2.size());
  nntrainer::TernaryTensor t2(wdim, rec2.data());
  ASSERT_TRUE(t2.validate());
  for (unsigned int n = 0; n < N; n += 3)
    for (unsigned int k = 0; k < K; k += 7)
      ASSERT_EQ(t2.decode(0, n, k), dense[(size_t)n * K + k]);

  std::vector<void *> W2 = {w.getData()};
  std::vector<float> c_tile(len, -2.0f), c_arch(len, -3.0f), c_ops(len, -4.0f),
    c_oracle(len, 0.0f);
  nntrainer::tq2cx_gemm_batch(M, Ns, K, A, K, W2, out(c_tile), Ns,
                              nntrainer::tq2cx_tile_ref, false);
  nntrainer::gemm_tq2cx_batch(M, Ns, K, A, K, W2, out(c_arch), Ns);
  auto *ops = nntrainer::getComputeOps();
  ASSERT_NE(ops, nullptr);
  ops->gemm_tq2cx_batch_fp32(W2, const_cast<float *>(A), out(c_ops), M, Ns, K);
  nntrainer::tq2cx_gemm_ref_fp32(M, N, K, A, K, w.getData(), c_oracle.data(), N,
                                 0.0f);
  EXPECT_EQ(std::memcmp(c_ref.data(), c_tile.data(), bytes), 0)
    << "v2 scalar tile differs from the row-major reference";
  EXPECT_EQ(std::memcmp(c_ref.data(), c_arch.data(), bytes), 0)
    << "v2 arch tile differs from the row-major reference";
  EXPECT_EQ(std::memcmp(c_ref.data(), c_ops.data(), bytes), 0)
    << "v2 ComputeOps dispatch differs from the row-major reference";
  EXPECT_EQ(std::memcmp(c_ref.data(), c_oracle.data(), bytes), 0)
    << "v2 tq2cx_gemm_ref_fp32 differs from the row-major reference";

  nntrainer::Tensor y(1, 1, M, N);
  act.dot(w, y, false, false, 0.0f);
  EXPECT_EQ(std::memcmp(c_ref.data(), y.getData<float>(), bytes), 0)
    << "v2 FloatTensor::dot differs from the row-major reference";
}

INSTANTIATE_TEST_SUITE_P(ExpertShapes, TernaryLayoutV2Test,
                         ::testing::Combine(::testing::Values(1u, 8u, 33u),
                                            ::testing::Values(704u, 2816u),
                                            ::testing::Values(704u, 2816u)));

// N % 16 != 0 (padded rows), an odd unit count (one-unit tail block),
// K % 32 != 0, N below one tile
INSTANTIATE_TEST_SUITE_P(OddShapes, TernaryLayoutV2Test,
                         ::testing::Values(std::make_tuple(5u, 100u, 13u),
                                           std::make_tuple(3u, 96u, 40u),
                                           std::make_tuple(17u, 64u, 57u),
                                           std::make_tuple(2u, 160u, 100u),
                                           std::make_tuple(4u, 32u, 8u)));

int main(int argc, char **argv) {
  int result = -1;

  testing::InitGoogleTest(&argc, argv);
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
