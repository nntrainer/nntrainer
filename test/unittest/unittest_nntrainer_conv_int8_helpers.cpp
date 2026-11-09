// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_conv_int8_helpers.cpp
 * @date   26 October 2026
 * @brief  Unit tests for conv_indirect.h, act_simd.h and int8_gemm.h
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include <act_simd.h>
#include <conv_indirect.h>
#include <int8_gemm.h>

namespace {

std::vector<float> randomVector(size_t n, unsigned int seed, float lo = -1.0f,
                                float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

/// 3x3 conv on a 5x6 input with padding, stride 2 and dilation 1 or 2.
nntrainer::ConvGatherParams convParams(int dil, bool nhwc) {
  nntrainer::ConvGatherParams p{};
  p.in_ch = 3;
  p.in_h = 5;
  p.in_w = 6;
  p.k_h = 3;
  p.k_w = 3;
  p.pad_t = 1;
  p.pad_l = 1;
  p.stride_h = 2;
  p.stride_w = 2;
  p.dil_h = dil;
  p.dil_w = dil;
  p.out_w = (p.in_w + 2 * p.pad_l - dil * (p.k_w - 1) - 1) / p.stride_w + 1;
  p.is_nhwc = nhwc;
  return p;
}

int outRows(const nntrainer::ConvGatherParams &p) {
  const int out_h =
    (p.in_h + 2 * p.pad_t - p.dil_h * (p.k_h - 1) - 1) / p.stride_h + 1;
  return out_h * p.out_w;
}

/// Naive im2col row: [c][kh][kw] order, or [kh][kw][c] when taps_last.
std::vector<float> im2colRef(const std::vector<float> &in,
                             const nntrainer::ConvGatherParams &p, int m,
                             bool taps_last) {
  const int oh = m / p.out_w, ow = m % p.out_w;
  std::vector<float> row((size_t)p.in_ch * p.k_h * p.k_w, 0.0f);
  for (int c = 0; c < p.in_ch; ++c)
    for (int kh = 0; kh < p.k_h; ++kh)
      for (int kw = 0; kw < p.k_w; ++kw) {
        const int h = oh * p.stride_h - p.pad_t + kh * p.dil_h;
        const int w = ow * p.stride_w - p.pad_l + kw * p.dil_w;
        if (h < 0 || h >= p.in_h || w < 0 || w >= p.in_w)
          continue;
        const float v = p.is_nhwc ? in[(h * p.in_w + w) * p.in_ch + c]
                                  : in[(c * p.in_h + h) * p.in_w + w];
        const int k = taps_last ? (kh * p.k_w + kw) * p.in_ch + c
                                : (c * p.k_h + kh) * p.k_w + kw;
        row[k] = v;
      }
  return row;
}

/// Q8_0 weight [N x K] where every block of a row shares one scale.
std::vector<uint8_t> perChannelWeight(const std::vector<float> &w,
                                      unsigned int N, unsigned int K,
                                      std::vector<float> &scales) {
  constexpr unsigned int QK = nntrainer::int8_gemm::QK;
  constexpr unsigned int BB = nntrainer::int8_gemm::BLOCK_BYTES;
  std::vector<uint8_t> out((size_t)N * (K / QK) * BB);
  scales.assign(N, 0.0f);
  for (unsigned int n = 0; n < N; ++n) {
    float amax = 0.0f;
    for (unsigned int k = 0; k < K; ++k)
      amax = std::max(amax, std::fabs(w[n * K + k]));
    // scale is a power of two so it is exact in fp16
    const int e = static_cast<int>(std::ceil(std::log2(amax / 127.0f)));
    const float s = std::ldexp(1.0f, e);
    scales[n] = s;
    const uint16_t d = static_cast<uint16_t>((e + 15) << 10);
    for (unsigned int b = 0; b < K / QK; ++b) {
      uint8_t *blk = out.data() + ((size_t)n * (K / QK) + b) * BB;
      std::memcpy(blk, &d, 2);
      for (unsigned int l = 0; l < QK; ++l)
        blk[2 + l] = static_cast<uint8_t>(
          static_cast<int8_t>(std::round(w[n * K + b * QK + l] / s)));
    }
  }
  return out;
}

} // namespace

TEST(ConvIndirect, gather_matches_im2col) {
  for (bool nhwc : {false, true})
    for (int dil : {1, 2}) {
      const auto p = convParams(dil, nhwc);
      const auto in = randomVector((size_t)p.in_ch * p.in_h * p.in_w, 1);
      const int K = p.in_ch * p.k_h * p.k_w, rows = outRows(p);

      std::vector<float> got((size_t)rows * K, -1.0f);
      nntrainer::gather_conv_act_rows<float>(got.data(), in.data(), p, 0, rows);

      for (int m = 0; m < rows; ++m) {
        const auto ref = im2colRef(in, p, m, false);
        for (int k = 0; k < K; ++k)
          ASSERT_EQ(got[(size_t)m * K + k], ref[k])
            << "nhwc " << nhwc << ", dil " << dil << ", row " << m;
      }
    }
}

TEST(ConvIndirect, gather_pads_row_tail) {
  const auto p = convParams(1, false);
  const auto in = randomVector((size_t)p.in_ch * p.in_h * p.in_w, 2);
  const int K = p.in_ch * p.k_h * p.k_w, stride = 32;

  std::vector<float> got((size_t)2 * stride, -1.0f);
  nntrainer::gather_conv_act_rows<float>(got.data(), in.data(), p, 3, 2,
                                         stride);
  for (int r = 0; r < 2; ++r)
    for (int k = K; k < stride; ++k)
      EXPECT_EQ(got[r * stride + k], 0.0f);
}

TEST(ConvIndirect, taps_last_gather_matches_im2col) {
  for (int dil : {1, 2}) {
    const auto p = convParams(dil, true);
    const auto in = randomVector((size_t)p.in_ch * p.in_h * p.in_w, 3);
    const int K = p.in_ch * p.k_h * p.k_w, rows = outRows(p);

    std::vector<float> got((size_t)rows * K, -1.0f);
    nntrainer::gather_conv_act_rows_tapslast<float>(got.data(), in.data(), p, 0,
                                                    rows);
    for (int m = 0; m < rows; ++m) {
      const auto ref = im2colRef(in, p, m, true);
      for (int k = 0; k < K; ++k)
        ASSERT_EQ(got[(size_t)m * K + k], ref[k]) << "dil " << dil;
    }
  }
}

TEST(ConvIndirect, repack_q8_0_interleaves_four_rows) {
  constexpr unsigned int N = 4, K = 64, nb = K / 32, BB = 34;
  std::vector<uint8_t> src(N * nb * BB);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(i * 7 + 1);
  std::vector<uint8_t> dst(src.size());
  nntrainer::repack_q8_0(dst.data(), src.data(), N, K);

  // block j of the super-row: d[4], then qs[32 * sub + 8 * row + c]
  for (unsigned int j = 0; j < nb; ++j) {
    const uint8_t *sb = dst.data() + j * 4 * BB;
    for (unsigned int r = 0; r < N; ++r) {
      const uint8_t *blk = src.data() + (r * nb + j) * BB;
      EXPECT_EQ(std::memcmp(sb + 2 * r, blk, 2), 0);
      for (unsigned int sub = 0; sub < 4; ++sub)
        EXPECT_EQ(std::memcmp(sb + 8 + 32 * sub + 8 * r, blk + 2 + 8 * sub, 8),
                  0);
    }
  }
}

namespace {

/// NHWC Q8_0 input: in_h x in_w pixels of in_ch / 32 blocks with random bytes.
std::vector<uint8_t> q8Input(const nntrainer::ConvGatherParams &p) {
  std::vector<uint8_t> in((size_t)p.in_h * p.in_w * (p.in_ch / 32) * 34);
  std::mt19937 rng(9);
  for (auto &b : in)
    b = static_cast<uint8_t>(rng());
  return in;
}

/// Block b of output row m, or an all-zero block for padding.
std::vector<uint8_t> q8RefBlock(const std::vector<uint8_t> &in,
                                const nntrainer::ConvGatherParams &p, int m,
                                int b) {
  const int ch_blocks = p.in_ch / 32;
  const int kh = b / ch_blocks / p.k_w, kw = b / ch_blocks % p.k_w;
  const int cb = b % ch_blocks;
  const int h = (m / p.out_w) * p.stride_h - p.pad_t + kh * p.dil_h;
  const int w = (m % p.out_w) * p.stride_w - p.pad_l + kw * p.dil_w;
  if (h < 0 || h >= p.in_h || w < 0 || w >= p.in_w)
    return std::vector<uint8_t>(34, 0);
  const uint8_t *src =
    in.data() + ((size_t)(h * p.in_w + w) * ch_blocks + cb) * 34;
  return std::vector<uint8_t>(src, src + 34);
}

nntrainer::ConvGatherParams q8Params(int dil) {
  auto p = convParams(dil, true);
  p.in_ch = 64;
  return p;
}

} // namespace

TEST(ConvIndirect, q8_0_single_row_gather) {
  for (int dil : {1, 2}) {
    const auto p = q8Params(dil);
    const auto in = q8Input(p);
    const int nb = p.k_h * p.k_w * (p.in_ch / 32);

    for (int m = 0; m < outRows(p); ++m) {
      std::vector<uint8_t> row((size_t)nb * 34, 0xAA);
      nntrainer::gather_conv_act_rows_q8_0_single(row.data(), in.data(), p, m);
      for (int b = 0; b < nb; ++b)
        ASSERT_EQ(std::vector<uint8_t>(row.begin() + b * 34,
                                       row.begin() + (b + 1) * 34),
                  q8RefBlock(in, p, m, b))
          << "dil " << dil << ", row " << m << ", block " << b;
    }
  }
}

TEST(ConvIndirect, q8_0_four_row_gather_interleaves_rows) {
  const auto p = q8Params(1);
  const auto in = q8Input(p);
  const int nb = p.k_h * p.k_w * (p.in_ch / 32);

  // A full tile, and a 3-row tail whose fourth row must be zero.
  for (int nrows : {4, 3}) {
    const int m0 = 2;
    std::vector<uint8_t> tile((size_t)nb * 4 * 34, 0xAA);
    nntrainer::gather_conv_act_rows_q8_0(tile.data(), in.data(), p, m0, nrows);

    for (int b = 0; b < nb; ++b) {
      const uint8_t *sb = tile.data() + (size_t)b * 4 * 34;
      for (int r = 0; r < 4; ++r) {
        const auto ref = r < nrows ? q8RefBlock(in, p, m0 + r, b)
                                   : std::vector<uint8_t>(34, 0);
        // d[4] first, then qs[32 * j + 8 * r + lane]
        EXPECT_EQ(std::memcmp(sb + 2 * r, ref.data(), 2), 0);
        for (int j = 0; j < 4; ++j)
          EXPECT_EQ(
            std::memcmp(sb + 8 + 32 * j + 8 * r, ref.data() + 2 + 8 * j, 8), 0)
            << "nrows " << nrows << ", block " << b << ", row " << r;
      }
    }
  }
}

TEST(ConvIndirect, typed_gather_wrappers) {
  const auto p = convParams(1, false);
  const auto in = randomVector((size_t)p.in_ch * p.in_h * p.in_w, 10);
  const int K = p.in_ch * p.k_h * p.k_w;

  std::vector<float> a((size_t)2 * K), b((size_t)2 * K);
  nntrainer::gather_conv_act_rows<float>(a.data(), in.data(), p, 1, 2);
  nntrainer::gather_conv_act_rows_fp32(b.data(), in.data(), p, 1, 2);
  EXPECT_EQ(a, b);
#ifdef ENABLE_FP16
  std::vector<_FP16> in16(in.size()), c((size_t)2 * K);
  for (size_t i = 0; i < in.size(); ++i)
    in16[i] = static_cast<_FP16>(in[i]);
  nntrainer::gather_conv_act_rows_fp16(c.data(), in16.data(), p, 1, 2);
  for (int k = 0; k < 2 * K; ++k)
    EXPECT_EQ(static_cast<float>(c[k]), static_cast<float>((_FP16)a[k]));
#endif
}

TEST(ActSimd, silu_matches_reference) {
  auto x = randomVector(1027, 4, -30.0f, 30.0f);
  const auto in = x;
  nntrainer::nntr_silu_inplace_f32(x.data(), x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    const double ref = in[i] / (1.0 + std::exp(-(double)in[i]));
    EXPECT_NEAR(x[i], ref, 1e-5 * std::max(1.0, std::fabs(ref)));
  }
}

TEST(ActSimd, quantize_affine_i8_matches_formula) {
  const auto x = randomVector(77, 5, -3.0f, 5.0f);
  const float inv = 255.0f / 8.0f, off = 3.0f;
  std::vector<int8_t> q(x.size());
  nntrainer::nntr_quantize_affine_i8(x.data(), q.data(), x.size(), inv, off);
  for (size_t i = 0; i < x.size(); ++i) {
    float ref = std::round((x[i] + off) * inv) - 128.f;
    ref = std::max(-128.f, std::min(127.f, ref));
    EXPECT_EQ(q[i], static_cast<int8_t>(ref)) << "index " << i;
  }
}

TEST(ActSimd, absmax) {
  auto x = randomVector(131, 6);
  x[97] = -7.5f;
  EXPECT_EQ(nntrainer::nntr_absmax_f32(x.data(), x.size()), 7.5f);
}

TEST(Int8Gemm, per_channel_matches_fp32_reference) {
  constexpr unsigned int M = 3, N = 8, K = 64;
  const auto A = randomVector(M * K, 7);
  const auto W = randomVector(N * K, 8);
  std::vector<float> scales;
  const auto Wq = perChannelWeight(W, N, K, scales);

  const auto w = nntrainer::int8_gemm::prepareWeight(Wq.data(), N, K);
  ASSERT_TRUE(w.usable);
  EXPECT_EQ(w.scale, scales);

  std::vector<float> C(M * N);
  ASSERT_TRUE(nntrainer::int8_gemm::gemmPerChannelA8(M, N, K, A.data(),
                                                     Wq.data(), w, C.data()));

  // 8-bit activations and weights: about 2 / 127 of each |a * w| term.
  for (unsigned int m = 0; m < M; ++m)
    for (unsigned int n = 0; n < N; ++n) {
      float ref = 0.0f;
      for (unsigned int k = 0; k < K; ++k)
        ref += A[m * K + k] * W[n * K + k];
      EXPECT_NEAR(C[m * N + n], ref, 2.0f / 127.0f * K);
    }

  // The int8-input variant gives the same result for the same activation.
  std::vector<int8_t> Aq;
  const float a_scale =
    nntrainer::int8_gemm::quantizeActivation(A.data(), A.size(), Aq);
  std::vector<float> C2(M * N);
  ASSERT_TRUE(nntrainer::int8_gemm::gemmPerChannelA8_q8in(
    M, N, K, Aq.data(), a_scale, Wq.data(), w, C2.data()));
  EXPECT_EQ(C, C2);
}

TEST(Int8Gemm, subnormal_scale) {
  // A channel whose weights are all below ~7.8e-3 gets a subnormal fp16
  // scale; 0x0200 is 512 * 2^-24.
  constexpr unsigned int N = 1, K = 32, BB = 34;
  const uint16_t d = 0x0200;
  const float scale = 512.0f / 16777216.0f;
  std::vector<uint8_t> Wq(BB);
  std::memcpy(Wq.data(), &d, 2);
  for (unsigned int l = 0; l < K; ++l)
    Wq[2 + l] = static_cast<uint8_t>(static_cast<int8_t>(l * 4 - 64));

  const auto w = nntrainer::int8_gemm::prepareWeight(Wq.data(), N, K);
  ASSERT_TRUE(w.usable);
  EXPECT_EQ(w.scale[0], scale);

  std::vector<float> A(K, 1.0f), C(1);
  ASSERT_TRUE(nntrainer::int8_gemm::gemmPerChannelA8(1, N, K, A.data(),
                                                     Wq.data(), w, C.data()));
  float ref = 0.0f;
  for (unsigned int l = 0; l < K; ++l)
    ref += static_cast<int8_t>(Wq[2 + l]) * scale;
  EXPECT_FLOAT_EQ(C[0], ref);
}

TEST(Int8Gemm, rejects_per_block_weight) {
  constexpr unsigned int N = 2, K = 64, BB = 34;
  std::vector<uint8_t> Wq(N * (K / 32) * BB, 0);
  const uint16_t d0 = 0x3C00, d1 = 0x4000; // two scales in row 0
  std::memcpy(Wq.data(), &d0, 2);
  std::memcpy(Wq.data() + BB, &d1, 2);

  const auto w = nntrainer::int8_gemm::prepareWeight(Wq.data(), N, K);
  EXPECT_FALSE(w.usable);

  std::vector<float> A(K, 1.0f), C(N, -1.0f);
  EXPECT_FALSE(nntrainer::int8_gemm::gemmPerChannelA8(1, N, K, A.data(),
                                                      Wq.data(), w, C.data()));
  EXPECT_EQ(C, std::vector<float>(N, -1.0f));
}

TEST(Int8Gemm, rejects_scales_for_another_shape) {
  constexpr unsigned int N = 2, K = 32, BB = 34;
  std::vector<uint8_t> Wq(4 * BB, 0);
  const uint16_t one = 0x3C00;
  for (unsigned int n = 0; n < 4; ++n)
    std::memcpy(Wq.data() + n * BB, &one, 2);

  // Scales prepared for 2 rows must not be used for 4.
  const auto w = nntrainer::int8_gemm::prepareWeight(Wq.data(), N, K);
  std::vector<float> A(K, 1.0f), C(4, -1.0f);
  EXPECT_FALSE(nntrainer::int8_gemm::gemmPerChannelA8(1, 4, K, A.data(),
                                                      Wq.data(), w, C.data()));
}

int main(int argc, char **argv) {
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
