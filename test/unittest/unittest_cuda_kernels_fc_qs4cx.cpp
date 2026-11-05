// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_cuda_kernels_fc_qs4cx.cpp
 * @date    24 Aug 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Differential test for the three CUDA QS4CX FC routes (dequant-GEMM
 *          floor, w4a8 dp4a, cuBLAS int8) against a double-precision host
 *          dequant-GEMM. The three routes are numerically different by
 *          construction -- the floor keeps the activation in FP32 while the
 *          other two quantize it to int8 per row -- so each is checked against
 *          the reference at its own tolerance rather than against each other.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cuda_fc_qs4cx.h>

namespace {

static bool cudaAvailable() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

/** @brief half -> float, matching the device kernels' bit-exact decode. */
static float h2f(unsigned short h) {
  unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, o;
  if (e == 0u) {
    if (m == 0u) {
      o = s;
    } else {
      int x = -1;
      do {
        m <<= 1;
        x++;
      } while ((m & 0x400u) == 0u);
      m &= 0x3FFu;
      o = s | ((unsigned int)(127 - 15 - x) << 23) | (m << 13);
    }
  } else if (e == 0x1Fu) {
    o = s | 0x7F800000u | (m << 13);
  } else {
    o = s | ((e + (127u - 15u)) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &o, sizeof(f));
  return f;
}

/** @brief float -> half, round-to-nearest-even. */
static unsigned short f2h(float f) {
  unsigned int xi;
  std::memcpy(&xi, &f, sizeof(xi));
  unsigned int s = (xi >> 16) & 0x8000u, mant = xi & 0x7FFFFFu;
  int e = (int)((xi >> 23) & 0xFFu);
  if (e == 0xFF)
    return (unsigned short)(s | 0x7C00u | (mant ? 0x200u : 0u));
  int exp = e - 127 + 15;
  if (exp >= 0x1F)
    return (unsigned short)(s | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10)
      return (unsigned short)s;
    mant |= 0x800000u;
    int sh = 14 - exp;
    unsigned int hh = mant >> sh, rem = mant & ((1u << sh) - 1u),
                 half = 1u << (sh - 1);
    if (rem > half || (rem == half && (hh & 1u)))
      hh++;
    return (unsigned short)(s | hh);
  }
  unsigned int hh = ((unsigned int)exp << 10) | (mant >> 13),
               rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (hh & 1u)))
    hh++;
  return (unsigned short)(s | hh);
}

/**
 * @brief One QS4CX test weight: the plain payload is [N][(K+1)/2] packed signed
 *        nibbles (even k = low nibble, odd k = high nibble, stored biased by 8)
 *        plus one FP16 scale per output channel.
 */
struct TestWeight {
  std::vector<unsigned char> plain;
  std::vector<unsigned short> scales;
  std::vector<float> dense; ///< [N][K] dequantized reference
};

static TestWeight makeWeight(unsigned int N, unsigned int K, unsigned seed) {
  const unsigned int Kh = (K + 1u) / 2u;
  TestWeight w;
  w.plain.assign((size_t)N * Kh, 0);
  w.scales.resize(N);
  w.dense.assign((size_t)N * K, 0.f);

  unsigned int rng = seed | 1u;
  auto next = [&rng]() {
    rng = rng * 1664525u + 1013904223u;
    return (rng >> 16) & 0xFFFFu;
  };

  for (unsigned int n = 0; n < N; ++n) {
    const float sc = 0.002f + 0.0001f * (float)(n % 17);
    w.scales[n] = f2h(sc);
    const float sc_h = h2f(w.scales[n]);
    for (unsigned int k = 0; k < K; ++k) {
      const int nib = (int)(next() % 16u); // biased int4: value = nib - 8
      unsigned char &b = w.plain[(size_t)n * Kh + (k >> 1)];
      if (k & 1u)
        b = (unsigned char)((b & 0x0Fu) | ((unsigned)nib << 4));
      else
        b = (unsigned char)((b & 0xF0u) | (unsigned)nib);
      w.dense[(size_t)n * K + k] = (float)(nib - 8) * sc_h;
    }
  }
  return w;
}

static std::vector<float> makeActivation(unsigned int M, unsigned int K,
                                         unsigned seed) {
  std::vector<float> x((size_t)M * K);
  unsigned int rng = seed | 1u;
  for (size_t i = 0; i < x.size(); ++i) {
    rng = rng * 1103515245u + 12345u;
    x[i] = ((float)((rng >> 16) & 0x7FFFu) / 16383.5f) - 1.0f;
  }
  return x;
}

/** @brief Y[M,N] = X[M,K] * Wdense[N,K]^T, accumulated in double. */
static std::vector<double> refGemm(const std::vector<float> &X,
                                   const std::vector<float> &Wdense,
                                   unsigned int M, unsigned int N,
                                   unsigned int K) {
  std::vector<double> y((size_t)M * N, 0.0);
  for (unsigned int m = 0; m < M; ++m)
    for (unsigned int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (unsigned int k = 0; k < K; ++k)
        acc += (double)X[(size_t)m * K + k] * (double)Wdense[(size_t)n * K + k];
      y[(size_t)m * N + n] = acc;
    }
  return y;
}

/** @brief Largest |ref| in the reference, used to normalize the error. */
static double refScale(const std::vector<double> &ref) {
  double s = 0.0;
  for (double v : ref)
    s = std::max(s, std::fabs(v));
  return s > 0.0 ? s : 1.0;
}

struct Shape {
  unsigned int M, N, K;
};

// NOTE: nothing allocated in this file is freed. The FC path caches its derived
// device copies keyed by the plain payload's ADDRESS -- sound for a model whose
// weights outlive the run, but a test that frees a weight and allocates another
// gets the same address back and hits a cache built for the previous shape.
// Holding every buffer for the life of the test binary keeps the addresses
// distinct.

// Decode-shaped (M=1), small-batch and prefill-shaped windows, all with K a
// multiple of 4 -- the shape every real projection width has.
static const std::vector<Shape> kShapes = {
  {1, 64, 128}, {4, 96, 256}, {32, 128, 512}};

// Contraction widths that are NOT a multiple of 4, which the int8 routes read
// four channels at a time. Both parities of (K+1)/2 are covered: an odd
// half-width also puts every second weight row on an odd address.
static const std::vector<Shape> kRaggedShapes = {
  {1, 8, 7}, {1, 8, 19}, {1, 8, 31}, {1, 8, 131},
  {1, 8, 5}, {1, 8, 17}, {1, 8, 33}, {1, 8, 129}};

} // namespace

TEST(cuda_kernels, fc_qs4cx_dequant_gemm_fp32) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";

  for (const Shape &s : kShapes) {
    const TestWeight w = makeWeight(s.N, s.K, 0xC0FFEEu + s.N);
    const std::vector<float> hx = makeActivation(s.M, s.K, 0xBEEFu + s.M);
    const std::vector<double> ref = refGemm(hx, w.dense, s.M, s.N, s.K);

    float *X = nullptr, *Y = nullptr;
    unsigned char *W = nullptr;
    unsigned short *S = nullptr;
    ASSERT_EQ(cudaMallocManaged(&X, hx.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&Y, (size_t)s.M * s.N * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&W, w.plain.size()), cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&S, w.scales.size() * sizeof(unsigned short)),
              cudaSuccess);
    std::memcpy(X, hx.data(), hx.size() * sizeof(float));
    std::memcpy(W, w.plain.data(), w.plain.size());
    std::memcpy(S, w.scales.data(), w.scales.size() * sizeof(unsigned short));
    std::memset(Y, 0, (size_t)s.M * s.N * sizeof(float));

    ASSERT_TRUE(
      nntrainer::cuda::cuda_fc_qs4cx_gemm_fp32(X, W, S, Y, s.M, s.N, s.K));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    double maxErr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
      maxErr = std::max(maxErr, std::fabs((double)Y[i] - ref[i]));
    const double rel = maxErr / refScale(ref);
    printf("CUDA QS4CX dequant-GEMM fp32 %ux%ux%u max rel error : %e\n", s.M,
           s.N, s.K, rel);
    // FP32 accumulation in a different order than the host reference.
    EXPECT_LT(rel, 1e-5);
  }
}

TEST(cuda_kernels, fc_qs4cx_dp4a_fp16) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";

  for (const Shape &s : kShapes) {
    const TestWeight w = makeWeight(s.N, s.K, 0xC0FFEEu + s.N);
    const std::vector<float> hx = makeActivation(s.M, s.K, 0xBEEFu + s.M);
    // The activation reaches the kernel as FP16, so quantize the reference
    // input the same way rather than blaming the kernel for the conversion.
    std::vector<float> hxq(hx.size());
    for (size_t i = 0; i < hx.size(); ++i)
      hxq[i] = h2f(f2h(hx[i]));
    const std::vector<double> ref = refGemm(hxq, w.dense, s.M, s.N, s.K);

    unsigned short *X = nullptr, *Y = nullptr, *S = nullptr;
    unsigned char *W = nullptr;
    ASSERT_EQ(cudaMallocManaged(&X, hx.size() * sizeof(unsigned short)),
              cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&Y, (size_t)s.M * s.N * sizeof(unsigned short)),
              cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&W, w.plain.size()), cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&S, w.scales.size() * sizeof(unsigned short)),
              cudaSuccess);
    for (size_t i = 0; i < hx.size(); ++i)
      X[i] = f2h(hx[i]);
    std::memcpy(W, w.plain.data(), w.plain.size());
    std::memcpy(S, w.scales.data(), w.scales.size() * sizeof(unsigned short));
    std::memset(Y, 0, (size_t)s.M * s.N * sizeof(unsigned short));

    ASSERT_TRUE(
      nntrainer::cuda::cuda_fc_qs4cx_dp4a_gemm_fp16(X, W, S, Y, s.M, s.N, s.K));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    double maxErr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
      maxErr = std::max(maxErr, std::fabs((double)h2f(Y[i]) - ref[i]));
    const double rel = maxErr / refScale(ref);
    printf("CUDA QS4CX dp4a w4a8 fp16 %ux%ux%u max rel error : %e\n", s.M, s.N,
           s.K, rel);
    // The activation is quantized to int8 per row, so this route is a w4a8
    // approximation of the reference, not a reassociation of it.
    EXPECT_LT(rel, 3e-2);
  }
}

TEST(cuda_kernels, fc_qs4cx_cublas_i8_fp16) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";

  // The Tensor-Core route is the prefill-shaped one; a decode-shaped M=1 GEMM
  // never selects it in production.
  const Shape s = {32, 128, 512};
  const TestWeight w = makeWeight(s.N, s.K, 0xC0FFEEu + s.N);
  const std::vector<float> hx = makeActivation(s.M, s.K, 0xBEEFu + s.M);
  std::vector<float> hxq(hx.size());
  for (size_t i = 0; i < hx.size(); ++i)
    hxq[i] = h2f(f2h(hx[i]));
  const std::vector<double> ref = refGemm(hxq, w.dense, s.M, s.N, s.K);

  unsigned short *X = nullptr, *Y = nullptr, *S = nullptr;
  unsigned char *W = nullptr;
  ASSERT_EQ(cudaMallocManaged(&X, hx.size() * sizeof(unsigned short)),
            cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&Y, (size_t)s.M * s.N * sizeof(unsigned short)),
            cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&W, w.plain.size()), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&S, w.scales.size() * sizeof(unsigned short)),
            cudaSuccess);
  for (size_t i = 0; i < hx.size(); ++i)
    X[i] = f2h(hx[i]);
  std::memcpy(W, w.plain.data(), w.plain.size());
  std::memcpy(S, w.scales.data(), w.scales.size() * sizeof(unsigned short));
  std::memset(Y, 0, (size_t)s.M * s.N * sizeof(unsigned short));

  if (!nntrainer::cuda::cuda_fc_qs4cx_cublas_i8_gemm_fp16(X, W, S, Y, s.M, s.N,
                                                          s.K))
    GTEST_SKIP() << "cuBLAS int8 route unavailable on this device";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  double maxErr = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    maxErr = std::max(maxErr, std::fabs((double)h2f(Y[i]) - ref[i]));
  const double rel = maxErr / refScale(ref);
  printf("CUDA QS4CX cuBLAS int8 fp16 %ux%ux%u max rel error : %e\n", s.M, s.N,
         s.K, rel);
  // Same w4a8 quantization as dp4a; the int32 accumulation is exact, so this
  // must land in the same band.
  EXPECT_LT(rel, 3e-2);
}

TEST(cuda_kernels, fc_qs4cx_dp4a_ragged_k) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";

  // Every weight stays allocated for the whole test. The derived device caches
  // are keyed by the plain payload's ADDRESS, which is sound for a model whose
  // weights outlive the run, but a loop that frees and re-allocates would get
  // the same address back and hit a cache built for the previous shape.
  struct Buf {
    unsigned short *X, *Y, *S;
    unsigned char *W;
  };
  std::vector<Buf> bufs(kRaggedShapes.size());
  std::vector<std::vector<double>> refs(kRaggedShapes.size());

  for (size_t si = 0; si < kRaggedShapes.size(); ++si) {
    const Shape &s = kRaggedShapes[si];
    const TestWeight w = makeWeight(s.N, s.K, 0xC0FFEEu + s.K);
    const std::vector<float> hx = makeActivation(s.M, s.K, 0xBEEFu + s.K);
    std::vector<float> hxq(hx.size());
    for (size_t i = 0; i < hx.size(); ++i)
      hxq[i] = h2f(f2h(hx[i]));
    refs[si] = refGemm(hxq, w.dense, s.M, s.N, s.K);

    Buf &b = bufs[si];
    ASSERT_EQ(cudaMallocManaged(&b.X, hx.size() * sizeof(unsigned short)),
              cudaSuccess);
    ASSERT_EQ(
      cudaMallocManaged(&b.Y, (size_t)s.M * s.N * sizeof(unsigned short)),
      cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&b.W, w.plain.size()), cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&b.S, w.scales.size() * sizeof(unsigned short)),
              cudaSuccess);
    for (size_t i = 0; i < hx.size(); ++i)
      b.X[i] = f2h(hx[i]);
    std::memcpy(b.W, w.plain.data(), w.plain.size());
    std::memcpy(b.S, w.scales.data(), w.scales.size() * sizeof(unsigned short));
    std::memset(b.Y, 0, (size_t)s.M * s.N * sizeof(unsigned short));
  }

  for (size_t si = 0; si < kRaggedShapes.size(); ++si) {
    const Shape &s = kRaggedShapes[si];
    Buf &b = bufs[si];
    ASSERT_TRUE(nntrainer::cuda::cuda_fc_qs4cx_dp4a_gemm_fp16(
      b.X, b.W, b.S, b.Y, s.M, s.N, s.K));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    double maxErr = 0.0;
    for (size_t i = 0; i < refs[si].size(); ++i)
      maxErr = std::max(maxErr, std::fabs((double)h2f(b.Y[i]) - refs[si][i]));
    const double rel = maxErr / refScale(refs[si]);
    printf("CUDA QS4CX dp4a ragged K %ux%ux%u max rel error : %e\n", s.M, s.N,
           s.K, rel);
    // Dropping the tail channels shows up as a gross error, not a small one:
    // a K=7 contraction that stops at 4 is missing three of seven terms.
    EXPECT_LT(rel, 3e-2);
  }
}

namespace {

/**
 * @brief A per-channel ASYMMETRIC int4 weight w = s*(q - z), q in [0, 15],
 *        z in [3, 12], in the plain QS4CX layout (nibble q = signed code
 *        q - 8, fp16 scale s) plus its integer zero points z[n].
 */
struct AsymWeight {
  std::vector<unsigned char> plain;
  std::vector<unsigned short> scales;
  std::vector<float> scales32, zp;
  std::vector<float> dense; ///< [N][K] asymmetric weight
};

static AsymWeight makeAsymWeight(unsigned int N, unsigned int K,
                                 unsigned seed) {
  const unsigned int Kh = (K + 1u) / 2u;
  AsymWeight w;
  w.plain.assign((size_t)N * Kh, 0);
  w.scales.resize(N);
  w.scales32.resize(N);
  w.zp.resize(N);
  w.dense.assign((size_t)N * K, 0.f);
  unsigned int rng = seed | 1u;
  auto next = [&rng]() {
    rng = rng * 1664525u + 1013904223u;
    return (rng >> 16) & 0xFFFFu;
  };
  for (unsigned int n = 0; n < N; ++n) {
    w.scales[n] = f2h(0.002f + 0.0001f * (float)(n % 17));
    const float s = h2f(w.scales[n]);
    const int z = 3 + (int)(next() % 10u);
    w.scales32[n] = s;
    w.zp[n] = (float)z;
    for (unsigned int k = 0; k < K; ++k) {
      const int q = (int)(next() % 16u);
      unsigned char &b = w.plain[(size_t)n * Kh + (k >> 1)];
      if (k & 1u)
        b = (unsigned char)((b & 0x0Fu) | ((unsigned)q << 4));
      else
        b = (unsigned char)((b & 0xF0u) | (unsigned)q);
      w.dense[(size_t)n * K + k] = (float)(q - z) * s;
    }
  }
  return w;
}

enum class Route { DP4A, CUBLAS, FPACT };

/**
 * @brief Run one route on an asymmetric weight before and after registering
 *        its zero points. Without them the output must miss the asymmetric
 *        reference by far more than the w4a8 band; with them land inside it.
 * @return false when the route is unavailable (skip).
 */
static bool runAsym(Route r, unsigned int M, unsigned int N, unsigned int K,
                    unsigned seed) {
  const AsymWeight w = makeAsymWeight(N, K, seed);
  const std::vector<float> hx = makeActivation(M, K, seed + 5u);
  std::vector<float> hxq(hx.size());
  for (size_t i = 0; i < hx.size(); ++i)
    hxq[i] = h2f(f2h(hx[i]));
  const std::vector<double> ref = refGemm(hxq, w.dense, M, N, K);

  unsigned short *X = nullptr, *Y = nullptr, *S = nullptr;
  unsigned char *W = nullptr;
  float *S32 = nullptr;
  EXPECT_EQ(cudaMallocManaged(&X, hx.size() * sizeof(unsigned short)),
            cudaSuccess);
  EXPECT_EQ(cudaMallocManaged(&Y, (size_t)M * N * sizeof(unsigned short)),
            cudaSuccess);
  EXPECT_EQ(cudaMallocManaged(&W, w.plain.size()), cudaSuccess);
  EXPECT_EQ(cudaMallocManaged(&S, (size_t)N * sizeof(unsigned short)),
            cudaSuccess);
  EXPECT_EQ(cudaMallocManaged(&S32, (size_t)N * sizeof(float)), cudaSuccess);
  for (size_t i = 0; i < hx.size(); ++i)
    X[i] = f2h(hx[i]);
  std::memcpy(W, w.plain.data(), w.plain.size());
  std::memcpy(S, w.scales.data(), (size_t)N * sizeof(unsigned short));
  std::memcpy(S32, w.scales32.data(), (size_t)N * sizeof(float));

  auto run = [&]() {
    std::memset(Y, 0, (size_t)M * N * sizeof(unsigned short));
    bool ok = false;
    switch (r) {
    case Route::DP4A:
      ok = nntrainer::cuda::cuda_fc_qs4cx_dp4a_gemm_fp16(X, W, S, Y, M, N, K);
      break;
    case Route::CUBLAS:
      ok =
        nntrainer::cuda::cuda_fc_qs4cx_cublas_i8_gemm_fp16(X, W, S, Y, M, N, K);
      break;
    case Route::FPACT:
      ok = nntrainer::cuda::cuda_fc_qs4cx_fpact_gemv_fp16(X, W, S32, Y, N, K);
      break;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    double e = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
      e = std::max(e, std::fabs((double)h2f(Y[i]) - ref[i]));
    return ok ? e / refScale(ref) : -1.0;
  };

  const double before = run();
  if (before < 0.0)
    return false;
  if (r == Route::FPACT) {
    // The loader-side form, c[n] = s[n] * (8 - z[n]).
    std::vector<float> c(N);
    for (unsigned int n = 0; n < N; ++n)
      c[n] = w.scales32[n] * (8.0f - w.zp[n]);
    EXPECT_TRUE(nntrainer::cuda::cuda_fc_qs4cx_set_zero_point_correction(
      W, w.scales32.data(), c.data(), N));
  } else {
    EXPECT_TRUE(
      nntrainer::cuda::cuda_fc_qs4cx_set_weight_zero_points(W, w.zp.data(), N));
  }
  const double after = run();
  printf("CUDA QS4CX asym route %d %ux%ux%u rel error: without zp %e "
         "with zp %e\n",
         (int)r, M, N, K, before, after);
  EXPECT_GT(before, 0.1);
  EXPECT_LT(after, r == Route::FPACT ? 1e-2 : 3e-2);
  return true;
}

} // namespace

TEST(cuda_kernels, fc_qs4cx_asym_zero_point_dp4a) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";
  // A symmetric weight's output is recorded before any zero points exist and
  // again after: registering them turns on the quantisers' row-sum output,
  // which must not move a byte of a symmetric FC.
  const TestWeight sw = makeWeight(64, 256, 0xA11CEu);
  const std::vector<float> hx = makeActivation(4, 256, 0x5EEDu);
  // One activation buffer per row count: the staged-quant handoff is keyed
  // by (pointer, K), so two row counts on one pointer would share a quant.
  unsigned short *Xs[2] = {nullptr, nullptr}, *Y = nullptr, *S = nullptr;
  unsigned char *W = nullptr;
  for (auto &X : Xs) {
    ASSERT_EQ(cudaMallocManaged(&X, hx.size() * sizeof(unsigned short)),
              cudaSuccess);
    for (size_t i = 0; i < hx.size(); ++i)
      X[i] = f2h(hx[i]);
  }
  ASSERT_EQ(cudaMallocManaged(&Y, 4 * 64 * sizeof(unsigned short)),
            cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&W, sw.plain.size()), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&S, sw.scales.size() * sizeof(unsigned short)),
            cudaSuccess);
  std::memcpy(W, sw.plain.data(), sw.plain.size());
  std::memcpy(S, sw.scales.data(), sw.scales.size() * sizeof(unsigned short));
  std::vector<unsigned short> sym_before[2];
  for (unsigned M : {1u, 4u}) {
    ASSERT_TRUE(nntrainer::cuda::cuda_fc_qs4cx_dp4a_gemm_fp16(Xs[M == 4], W, S,
                                                              Y, M, 64, 256));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    sym_before[M == 4].assign(Y, Y + M * 64);
  }

  for (const Shape &s : kShapes)
    ASSERT_TRUE(runAsym(Route::DP4A, s.M, s.N, s.K, 0xA5A5u + s.N));

  for (unsigned M : {1u, 4u}) {
    ASSERT_TRUE(nntrainer::cuda::cuda_fc_qs4cx_dp4a_gemm_fp16(Xs[M == 4], W, S,
                                                              Y, M, 64, 256));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(0, std::memcmp(Y, sym_before[M == 4].data(),
                             M * 64 * sizeof(unsigned short)))
      << "symmetric M=" << M << " output changed";
  }
}

TEST(cuda_kernels, fc_qs4cx_asym_zero_point_cublas_i8) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";
  if (!runAsym(Route::CUBLAS, 32, 128, 512, 0xB0B0u))
    GTEST_SKIP() << "cuBLAS int8 route unavailable on this device";
}

TEST(cuda_kernels, fc_qs4cx_asym_zero_point_fpact_gemv) {
  if (!cudaAvailable())
    GTEST_SKIP() << "no CUDA device";
  ASSERT_TRUE(runAsym(Route::FPACT, 1, 256, 512, 0xC0C0u));
}

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
