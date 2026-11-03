// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_qs4acx.cpp
 * @date	02 October 2026
 * @brief	Unit tests for the QS4ACX (asymmetric per-channel 4-bit) weight
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <vector>

#include <cpu_backend.h>
#include <qs4acx_tensor.h>
#include <tensor.h>

namespace {

using TD = nntrainer::TensorDim;

/**
 * @brief random asymmetric uint4 weight w[k][n] = s[n] * (u - z[n]) packed
 *        as a QS4ACX record; also returns the dense [N, K] weight
 */
std::vector<uint8_t> makeRecord(size_t K, size_t N, unsigned int seed,
                                bool symmetric, std::vector<float> &dense) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> code(0, 15), zero(2, 14);
  std::uniform_real_distribution<float> step(0.001f, 0.05f);
  const size_t stride = (K + 1) / 2;
  std::vector<uint8_t> rec(nntrainer::QS4ACX_Tensor::recordBytes(K, N), 0);
  float *scale = reinterpret_cast<float *>(rec.data() + N * stride);
  float *zp = scale + N;
  dense.assign(N * K, 0.0f);
  for (size_t n = 0; n < N; ++n) {
    scale[n] = step(rng);
    zp[n] = symmetric ? 8.0f : static_cast<float>(zero(rng));
    for (size_t k = 0; k < K; ++k) {
      const int u = code(rng);
      rec[n * stride + k / 2] |=
        static_cast<uint8_t>(k % 2 == 0 ? u : (u << 4));
      dense[n * K + k] = scale[n] * (static_cast<float>(u) - zp[n]);
    }
  }
  return rec;
}

std::vector<float> randomAct(size_t n, unsigned int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto &x : v)
    x = d(rng);
  return v;
}

} // namespace

TEST(nntrainer_qs4acx, size_and_registration_p) {
  const TD dim(1, 1, 64, 24, {TD::Format::NCHW, TD::DataType::QS4ACX});
  nntrainer::Tensor t(dim, true);
  EXPECT_EQ(t.getMemoryBytes(), 24u * 32u + 2u * 24u * sizeof(float));
  EXPECT_EQ(dim.getDataTypeSize(), 1u);
  std::stringstream ss;
  ss << dim;
  EXPECT_NE(ss.str().find("QS4ACX"), std::string::npos);
}

/**
 * @brief the record prefix is a QS4CX record: the QS4CX decoder gives
 *        scale * c, and the zero point adds scale * (8 - zp)
 */
TEST(nntrainer_qs4acx, prefix_decodes_as_qs4cx_p) {
  const size_t K = 96, N = 40;
  std::vector<float> dense;
  auto rec = makeRecord(K, N, 3, false, dense);
  const float *scale =
    reinterpret_cast<const float *>(rec.data() + N * ((K + 1) / 2));
  const float *zp = scale + N;
  std::vector<float> row(K);
  for (size_t n = 0; n < N; ++n) {
    nntrainer::dequantize_row_qs4cx(n, K, rec.data(),
                                    const_cast<float *>(scale), row.data());
    for (size_t k = 0; k < K; ++k)
      EXPECT_NEAR(row[k] + scale[n] * (8.0f - zp[n]), dense[n * K + k],
                  1e-5f * scale[n]);
  }
}

/**
 * @brief random asymmetric weights: reference GEMM and the FloatTensor::dot
 *        path agree bit for bit, run to run, and with a double precision
 *        dense product (fp32 accumulation noise only)
 */
TEST(nntrainer_qs4acx, matches_dense_reference_p) {
  for (auto [M, K, N] : std::vector<std::tuple<size_t, size_t, size_t>>{
         {1, 2816, 704}, {7, 704, 2816}, {3, 100, 33}}) {
    std::vector<float> dense;
    auto rec = makeRecord(K, N, 11 + K, false, dense);
    const auto A = randomAct(M * K, 5 + M);

    std::vector<float> c1(M * N, -1.0f), c2(M * N, -2.0f);
    nntrainer::qs4acx_gemm_ref_fp32(M, N, K, A.data(), K, rec.data(), nullptr,
                                    c1.data(), N);
    nntrainer::qs4acx_gemm_ref_fp32(M, N, K, A.data(), K, rec.data(), nullptr,
                                    c2.data(), N);
    EXPECT_EQ(std::memcmp(c1.data(), c2.data(), c1.size() * sizeof(float)), 0);

    nntrainer::Tensor w(
      TD(1, 1, K, N, {TD::Format::NCHW, TD::DataType::QS4ACX}), rec.data());
    nntrainer::Tensor x(1, 1, M, K);
    std::memcpy(x.getData<float>(), A.data(), A.size() * sizeof(float));
    nntrainer::Tensor y(1, 1, M, N);
    x.dot(w, y, false, false, 0.0f);
    EXPECT_EQ(
      std::memcmp(c1.data(), y.getData<float>(), c1.size() * sizeof(float)), 0);

    double num = 0.0, den = 0.0;
    for (size_t m = 0; m < M; ++m)
      for (size_t n = 0; n < N; ++n) {
        double r = 0.0;
        for (size_t k = 0; k < K; ++k)
          r += static_cast<double>(A[m * K + k]) * dense[n * K + k];
        const double d = r - c1[m * N + n];
        num += d * d;
        den += r * r;
      }
    EXPECT_LE(std::sqrt(num / den), 1e-5) << "M=" << M << " K=" << K;
  }
}

/**
 * @brief zp == 8 everywhere: the correction is exactly zero and the result
 *        is bit-identical to the symmetric (QS4CX) product computed in the
 *        same order
 */
TEST(nntrainer_qs4acx, zero_point_8_is_qs4cx_p) {
  const size_t M = 4, K = 704, N = 128;
  std::vector<float> dense;
  auto rec = makeRecord(K, N, 21, true, dense);
  const auto A = randomAct(M * K, 9);
  const size_t stride = (K + 1) / 2;
  const float *scale = reinterpret_cast<const float *>(rec.data() + N * stride);

  std::vector<float> out(M * N), sym(M * N);
  nntrainer::qs4acx_gemm_ref_fp32(M, N, K, A.data(), K, rec.data(), nullptr,
                                  out.data(), N);
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        const uint8_t b = rec[n * stride + k / 2];
        const int u = (k % 2 == 0) ? (b & 0x0F) : (b >> 4);
        acc = std::fma(static_cast<float>(u - 8), A[m * K + k], acc);
      }
      sym[m * N + n] = scale[n] * acc;
    }
  EXPECT_EQ(std::memcmp(out.data(), sym.data(), out.size() * sizeof(float)), 0);
}

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
