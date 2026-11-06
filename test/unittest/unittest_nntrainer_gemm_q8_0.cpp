// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_gemm_q8_0.cpp
 * @date   26 October 2026
 * @brief  Unit tests for the Q8_0 weight x FP32 activation GEMM
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include <compute_ops.h>
#include <cpu_backend.h>
#include <fallback_internal.h>
#include <fp16.h>
#include <ggml_interface.h>
#include <q8_0_tensor.h>
#include <quantizer.h>
#include <tensor.h>

namespace {

/// Reference q8_0 quantization of one row, as GGML does it.
void quantizeRowRef(const float *x, nntrainer::block_q8_0 *y, unsigned int k) {
  for (unsigned int b = 0; b < k / QK8_0; ++b) {
    float amax = 0.0f;
    for (int l = 0; l < QK8_0; ++l)
      amax = std::max(amax, std::fabs(x[b * QK8_0 + l]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    y[b].d = nntrainer::compute_fp32_to_fp16(d);
    for (int l = 0; l < QK8_0; ++l)
      y[b].qs[l] = static_cast<int8_t>(std::round(x[b * QK8_0 + l] * id));
  }
}

std::vector<float> randomVector(size_t n, unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

/// Weight [N x K] in fp32, as q8_0 rows and repacked to q8_0x4.
struct Weight {
  std::vector<float> fp32;
  std::vector<nntrainer::block_q8_0> q8;
  std::vector<nntrainer::block_q8_0> q8x4;
};

Weight makeWeight(unsigned int N, unsigned int K, unsigned int seed) {
  Weight w{randomVector(static_cast<size_t>(N) * K, seed), {}, {}};
  w.q8.resize(static_cast<size_t>(N) * K / QK8_0);
  for (unsigned int n = 0; n < N; ++n)
    quantizeRowRef(w.fp32.data() + static_cast<size_t>(n) * K,
                   w.q8.data() + static_cast<size_t>(n) * K / QK8_0, K);
  w.q8x4.resize(w.q8.size());
  nntrainer::repack_q8_0(w.q8x4.data(), w.q8.data(), w.q8.size() * Q8_0_SIZE, N,
                         K);
  return w;
}

/// C = A * B.T with A [M x K] and B [N x K].
std::vector<float> gemmRef(unsigned int M, unsigned int N, unsigned int K,
                           const float *A, const float *B) {
  std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
  for (unsigned int m = 0; m < M; ++m)
    for (unsigned int n = 0; n < N; ++n)
      for (unsigned int k = 0; k < K; ++k)
        C[m * N + n] += A[m * K + k] * B[n * K + k];
  return C;
}

/// Both operands are 8-bit, so each product is off by about 2 / 127.
float q8Tolerance(unsigned int K) { return 2.0f / 127.0f * K; }

struct Shape {
  unsigned int M, N, K;
};

class GemmQ8_0 : public ::testing::TestWithParam<Shape> {};

} // namespace

TEST_P(GemmQ8_0, scalar_matches_fp32_reference) {
  const Shape sh = GetParam();
  const auto A = randomVector(static_cast<size_t>(sh.M) * sh.K, 1);
  const Weight W = makeWeight(sh.N, sh.K, 2);

  std::vector<float> C(static_cast<size_t>(sh.M) * sh.N);
  nntrainer::__fallback_gemm_q8_0<float>(sh.M, sh.N, sh.K, A.data(), sh.K,
                                         W.q8x4.data(), sh.K, C.data(), sh.N);

  const auto ref = gemmRef(sh.M, sh.N, sh.K, A.data(), W.fp32.data());
  for (size_t i = 0; i < C.size(); ++i)
    EXPECT_NEAR(C[i], ref[i], q8Tolerance(sh.K)) << "index " << i;
}

TEST_P(GemmQ8_0, simd_matches_scalar) {
  const Shape sh = GetParam();
  const auto A = randomVector(static_cast<size_t>(sh.M) * sh.K, 3);
  const Weight W = makeWeight(sh.N, sh.K, 4);

  std::vector<float> scalar(static_cast<size_t>(sh.M) * sh.N);
  std::vector<float> simd(scalar.size());
  nntrainer::__fallback_gemm_q8_0<float>(
    sh.M, sh.N, sh.K, A.data(), sh.K, W.q8x4.data(), sh.K, scalar.data(), sh.N);
  nntrainer::getComputeOps()->gemm_q8_0_fp32(
    sh.M, sh.N, sh.K, A.data(), sh.K, W.q8x4.data(), sh.K, simd.data(), sh.N);

  // Same int8 math on the same blocks; only the fp32 summation order differs.
  for (size_t i = 0; i < scalar.size(); ++i)
    EXPECT_NEAR(simd[i], scalar[i], 1e-3f) << "index " << i;
}

// M = 1 is the GEMV path, M = 3 has no 4-row tile at all, M = 5 leaves a row
// tail after 4-row tiles, N = 36 leaves a partial 16-column tile, and M = 37
// spans several 16-row thread tiles.
INSTANTIATE_TEST_SUITE_P(Shapes, GemmQ8_0,
                         ::testing::Values(Shape{1, 32, 32}, Shape{1, 64, 128},
                                           Shape{3, 36, 64}, Shape{4, 32, 64},
                                           Shape{5, 36, 96}, Shape{8, 128, 128},
                                           Shape{37, 36, 96}));

TEST(GemmQ8_0, repack_matches_ggml_q8_0x4) {
  constexpr unsigned int N = 8, K = 64;
  const Weight W = makeWeight(N, K, 8);

  std::vector<nntrainer::block_q8_0> plain = W.q8, ggml(W.q8.size());
  nntrainer::__ggml_repack_q8_0_to_q8_0_4x8(ggml.data(), plain.data(),
                                            plain.size() * Q8_0_SIZE, N, K);
  EXPECT_EQ(std::memcmp(ggml.data(), W.q8x4.data(), ggml.size() * Q8_0_SIZE),
            0);
}

TEST(GemmQ8_0, rejects_unsupported_shapes_n) {
  std::vector<float> A(4 * 64, 1.0f), C(4 * 32);
  std::vector<nntrainer::block_q8_0> B(32 * 2);
  auto *ops = nntrainer::getComputeOps();

  // K = 48 is not a whole number of blocks.
  EXPECT_THROW(
    ops->gemm_q8_0_fp32(4, 32, 48, A.data(), 48, B.data(), 48, C.data(), 32),
    std::invalid_argument);
  // Weight rows come in groups of 4.
  EXPECT_THROW(
    ops->gemm_q8_0_fp32(4, 30, 32, A.data(), 32, B.data(), 32, C.data(), 30),
    std::invalid_argument);
  // Weight rows have to be packed.
  EXPECT_THROW(
    ops->gemm_q8_0_fp32(4, 32, 32, A.data(), 32, B.data(), 64, C.data(), 32),
    std::invalid_argument);
  // The scalar version checks the same.
  EXPECT_THROW(nntrainer::__fallback_gemm_q8_0<float>(
                 4, 32, 32, A.data(), 32, B.data(), 64, C.data(), 32),
               std::invalid_argument);
}

TEST(GemmQ8_0Tensor, dot_rejects_partial_block_rows_n) {
  // A Q8_0 weight only checks its width, so a 48-row weight can exist.
  nntrainer::Tensor W(
    {1, 1, 48, 32, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q8_0}},
    true);
  nntrainer::Tensor A(1, 1, 2, 48);
  nntrainer::Tensor out(1, 1, 2, 32);
  EXPECT_THROW(A.dot(W, out), std::invalid_argument);
}

TEST(GemmQ8_0Tensor, dot_with_quantized_weight_matches_fp32) {
  constexpr unsigned int M = 4, K = 96, N = 64;

  nntrainer::Tensor A(1, 1, M, K);
  nntrainer::Tensor W(1, 1, K, N);
  const auto a = randomVector(M * K, 5);
  const auto w = randomVector(K * N, 6);
  std::copy(a.begin(), a.end(), A.getData<float>());
  std::copy(w.begin(), w.end(), W.getData<float>());

  nntrainer::GgmlQuantizer quantizer(nntrainer::QScheme::Q8_0);
  nntrainer::Tensor W_q8 =
    quantizer.quantize(W, nntrainer::TensorDim::DataType::Q8_0);

  // The quantized dot writes into the given output, so allocate it.
  nntrainer::Tensor out(1, 1, M, N);
  A.dot(W_q8, out);
  nntrainer::Tensor ref = A.dot(W);

  const float *o = out.getData<float>();
  const float *r = ref.getData<float>();
  for (size_t i = 0; i < out.size(); ++i)
    EXPECT_NEAR(o[i], r[i], q8Tolerance(K)) << "index " << i;
}

namespace {

/// A [b x 1 x h x K] activation and a quantized [K x N] weight.
struct DotCase {
  nntrainer::Tensor A, W, W_q8;
};

DotCase makeDotCase(unsigned int b, unsigned int h, unsigned int K,
                    unsigned int N) {
  DotCase c{nntrainer::Tensor(b, 1, h, K), nntrainer::Tensor(1, 1, K, N), {}};
  const auto a = randomVector(static_cast<size_t>(b) * h * K, 9);
  const auto w = randomVector(static_cast<size_t>(K) * N, 10);
  std::copy(a.begin(), a.end(), c.A.getData<float>());
  std::copy(w.begin(), w.end(), c.W.getData<float>());
  nntrainer::GgmlQuantizer quantizer(nntrainer::QScheme::Q8_0);
  c.W_q8 = quantizer.quantize(c.W, nntrainer::TensorDim::DataType::Q8_0);
  return c;
}

void expectNearTensor(nntrainer::Tensor &out, nntrainer::Tensor &ref,
                      unsigned int K) {
  ASSERT_EQ(out.getDim(), ref.getDim());
  const float *o = out.getData<float>();
  const float *r = ref.getData<float>();
  for (size_t i = 0; i < out.size(); ++i)
    EXPECT_NEAR(o[i], r[i], q8Tolerance(K)) << "index " << i;
}

} // namespace

TEST(GemmQ8_0Tensor, dot_returning_output_allocates_it) {
  DotCase c = makeDotCase(1, 4, 64, 32);
  nntrainer::Tensor out = c.A.dot(c.W_q8);
  nntrainer::Tensor ref = c.A.dot(c.W);
  expectNearTensor(out, ref, 64);

  // An empty output passed in is allocated the same way.
  nntrainer::Tensor empty;
  c.A.dot(c.W_q8, empty);
  expectNearTensor(empty, ref, 64);
}

TEST(GemmQ8_0Tensor, dot_covers_every_batch_row) {
  DotCase c = makeDotCase(3, 2, 64, 32);
  nntrainer::Tensor out = c.A.dot(c.W_q8);
  nntrainer::Tensor ref = c.A.dot(c.W);
  expectNearTensor(out, ref, 64);
}

TEST(GemmQ8_0Tensor, dot_rejects_unsupported_calls_n) {
  DotCase c = makeDotCase(1, 4, 64, 32);
  nntrainer::Tensor out(1, 1, 4, 32);

  // The weight's K has to match the activation's.
  nntrainer::Tensor A_wide(1, 1, 4, 96);
  EXPECT_THROW(A_wide.dot(c.W_q8, out), std::invalid_argument);
  // The output is overwritten, so beta cannot be honored.
  EXPECT_THROW(c.A.dot(c.W_q8, out, false, false, 1.0f), std::invalid_argument);
  EXPECT_THROW(c.A.dot(c.W_q8, out, false, true), std::invalid_argument);
  // The output has to hold M x N values.
  nntrainer::Tensor short_out(1, 1, 2, 32);
  EXPECT_THROW(c.A.dot(c.W_q8, short_out), std::invalid_argument);
  // Only NCHW activations are supported.
  nntrainer::Tensor A_nhwc(
    nntrainer::TensorDim(
      1, 64, 4, 1, {nntrainer::Tformat::NHWC, nntrainer::Tdatatype::FP32}),
    true);
  EXPECT_THROW(A_nhwc.dot(c.W_q8, out), std::invalid_argument);
}

int main(int argc, char **argv) {
  // Sets up the ops table and the fp16 lookup table the q8_0 kernels use.
  nntrainer::init_backend();

  int result = -1;
  try {
    ::testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS" << std::endl;
  }
  return result;
}
