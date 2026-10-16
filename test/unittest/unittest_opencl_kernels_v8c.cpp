// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	unittest_opencl_kernels_v8c.cpp
 * @date	30 August 2026
 * @brief	Numerical and cache-boundary tests for the int8 x int4 v8c GEMM
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @details The v8c path is a GPU replacement for a host dot over a QS4CX
 * weight, so the host dot is its reference: both read the SAME tensor, so any
 * disagreement beyond the int8 activation quantization error is the kernel's.
 * Every case skips rather than fails when no OpenCL device can take the
 * dispatch, so a machine without a GPU still reports green.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include <blas_kernel_interface.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace {

/**
 * @brief Deterministic pseudo-random byte stream, so a failure is reproducible
 * without carrying a golden blob into the tree.
 */
class Lcg {
public:
  explicit Lcg(uint32_t seed) : s_(seed) {}
  /** @brief next value in [0, 2^31) */
  uint32_t next() {
    s_ = s_ * 1103515245u + 12345u;
    return (s_ >> 1);
  }
  /** @brief next value in [lo, hi) */
  float uniform(float lo, float hi) {
    return lo + (hi - lo) * (float)(next() % 4096u) / 4096.0f;
  }

private:
  uint32_t s_;
};

/**
 * @brief Build a QS4CX weight of shape (K, N) filled with reproducible
 * nibbles and per-output-channel scales.
 * @details The nibble layout is not interpreted here on purpose: the point of
 * the comparison below is that the GPU kernel and the host dot agree on
 * whatever the layout is, which is exactly the contract dotCl_v8c has to
 * honour to be a drop-in for Tensor::dot.
 */
nntrainer::Tensor makeQs4cxWeight(unsigned int K, unsigned int N,
                                  const std::string &name, uint32_t seed) {
  nntrainer::TensorDim dim(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QS4CX});
  nntrainer::Tensor w(dim, true, nntrainer::Initializer::NONE, name);

  uint8_t *nibbles = w.getData<uint8_t>();
  float *scales = w.getScale<float>();
  EXPECT_NE(nibbles, nullptr);
  EXPECT_NE(scales, nullptr);

  Lcg rng(seed);
  const size_t nibble_bytes = (size_t)N * ((K + 1) / 2);
  for (size_t i = 0; i < nibble_bytes; ++i)
    nibbles[i] = (uint8_t)(rng.next() & 0xFF);
  for (unsigned int n = 0; n < N; ++n)
    scales[n] = rng.uniform(0.005f, 0.02f);

  return w;
}

/**
 * @brief Fill an FP32 activation with reproducible values in [-1, 1).
 */
void fillActivation(nntrainer::Tensor &t, uint32_t seed) {
  Lcg rng(seed);
  float *p = t.getData<float>();
  for (size_t i = 0; i < t.getDim().getDataLen(); ++i)
    p[i] = rng.uniform(-1.0f, 1.0f);
}

/**
 * @brief Largest |gpu - ref| relative to the reference's own magnitude.
 * @details A plain elementwise relative error is meaningless where the
 * reference is near zero, which a random int4 GEMM output frequently is, so
 * the denominator is the row's peak magnitude.
 */
float relativeError(const nntrainer::Tensor &gpu, const nntrainer::Tensor &ref,
                    unsigned int M, unsigned int N) {
  const float *g = gpu.getData<float>();
  const float *r = ref.getData<float>();
  float worst = 0.0f;
  for (unsigned int m = 0; m < M; ++m) {
    float peak = 1e-6f;
    for (unsigned int n = 0; n < N; ++n)
      peak = std::max(peak, std::fabs(r[(size_t)m * N + n]));
    for (unsigned int n = 0; n < N; ++n) {
      const size_t i = (size_t)m * N + n;
      worst = std::max(worst, std::fabs(g[i] - r[i]) / peak);
    }
  }
  return worst;
}

/**
 * @brief Both sides quantize the activation to int8 per row, so agreement is
 * close but not exact. Measured worst case across the shapes below is under
 * 1e-2 of the row peak, and the bound is set at twice that: it stays well
 * below the scale of a kernel that indexes, unpacks or accumulates wrongly,
 * every one of which misses by O(1) relative to the row peak.
 */
constexpr float kQuantTolerance = 0.02f;

/**
 * @brief Run one (M, K, N) case through both paths and compare.
 * @return false when the GPU declined the dispatch, so the caller can skip.
 */
bool runCase(unsigned int M, unsigned int K, unsigned int N,
             const std::string &weight_name, uint32_t seed, float *err_out) {
  nntrainer::Tensor weight = makeQs4cxWeight(K, N, weight_name, seed);
  nntrainer::Tensor input(nntrainer::TensorDim(1, 1, M, K), true);
  fillActivation(input, seed + 7u);

  nntrainer::Tensor ref(nntrainer::TensorDim(1, 1, M, N), true);
  nntrainer::Tensor gpu(nntrainer::TensorDim(1, 1, M, N), true);
  ref.setZero();
  gpu.setZero();

  input.dot(weight, ref, false, false);
  if (!nntrainer::dotCl_v8c(input, weight, gpu))
    return false;

  *err_out = relativeError(gpu, ref, M, N);
  return true;
}

} // namespace

/**
 * @brief Decode shape: M = 1 takes the GEMV route through the kernel family.
 */
TEST(v8cGemm, gemv_m1_matches_host_dot_p) {
  float err = 0.0f;
  if (!runCase(1, 128, 32, "v8c_test_gemv_w", 0x51ED0001u, &err))
    GTEST_SKIP() << "no OpenCL device took the v8c dispatch";
  EXPECT_LT(err, kQuantTolerance);
}

/**
 * @brief Prefill shape below the tile alignment: M = 3 exercises the padded
 * rows and the kernel's valid-row store guard.
 */
TEST(v8cGemm, gemm_m3_matches_host_dot_p) {
  float err = 0.0f;
  if (!runCase(3, 128, 32, "v8c_test_m3_w", 0x51ED0002u, &err))
    GTEST_SKIP() << "no OpenCL device took the v8c dispatch";
  EXPECT_LT(err, kQuantTolerance);
}

/**
 * @brief Prefill shape past the M_pad alignment, on a wider K and N.
 */
TEST(v8cGemm, gemm_m64_matches_host_dot_p) {
  float err = 0.0f;
  if (!runCase(64, 256, 64, "v8c_test_m64_w", 0x51ED0003u, &err))
    GTEST_SKIP() << "no OpenCL device took the v8c dispatch";
  EXPECT_LT(err, kQuantTolerance);
}

/**
 * @brief A weight the kernel cannot take is declined, not thrown.
 * @details K = 100 is not a multiple of 32, so the entry point must return
 * false and leave the host fallback reachable.
 */
TEST(v8cGemm, declines_unsupported_shape_n) {
  nntrainer::Tensor weight =
    makeQs4cxWeight(100, 32, "v8c_test_decline_w", 0x51ED0004u);
  nntrainer::Tensor input(nntrainer::TensorDim(1, 1, 1, 100), true);
  fillActivation(input, 11u);
  nntrainer::Tensor out(nntrainer::TensorDim(1, 1, 1, 32), true);
  out.setZero();

  EXPECT_FALSE(nntrainer::dotCl_v8c(input, weight, out));
}

/**
 * @brief The shared-quant cache must not serve a previous pass's activation.
 *
 * @details Two weights share one activation tensor, which is what makes the
 * cache worth having (wq/wk/wv read one post-norm activation). The second
 * "pass" rewrites that tensor IN PLACE, so the cache key -- the host address
 * -- is bit-identical while the data behind it is not: this is the tensor-pool
 * recycling the cache has to survive. Both weights took part in the first
 * pass, so the pass boundary is detectable here.
 */
TEST(v8cGemm, quant_cache_survives_in_place_activation_rewrite_p) {
  constexpr unsigned int M = 4, K = 128, N = 32;
  nntrainer::Tensor wa = makeQs4cxWeight(K, N, "v8c_cache_wa", 0x51ED0010u);
  nntrainer::Tensor wb = makeQs4cxWeight(K, N, "v8c_cache_wb", 0x51ED0011u);
  nntrainer::Tensor input(nntrainer::TensorDim(1, 1, M, K), true);
  nntrainer::Tensor out(nntrainer::TensorDim(1, 1, M, N), true);
  nntrainer::Tensor ref(nntrainer::TensorDim(1, 1, M, N), true);

  // Pass one.
  fillActivation(input, 101u);
  out.setZero();
  if (!nntrainer::dotCl_v8c(input, wa, out))
    GTEST_SKIP() << "no OpenCL device took the v8c dispatch";
  out.setZero();
  ASSERT_TRUE(nntrainer::dotCl_v8c(input, wb, out));

  // Pass two, same activation address, different contents.
  fillActivation(input, 202u);
  ref.setZero();
  input.dot(wa, ref, false, false);
  out.setZero();
  ASSERT_TRUE(nntrainer::dotCl_v8c(input, wa, out));
  EXPECT_LT(relativeError(out, ref, M, N), kQuantTolerance);

  ref.setZero();
  input.dot(wb, ref, false, false);
  out.setZero();
  ASSERT_TRUE(nntrainer::dotCl_v8c(input, wb, out));
  EXPECT_LT(relativeError(out, ref, M, N), kQuantTolerance);
}

/**
 * @brief A pass that opens with a weight the previous pass never dispatched.
 *
 * @details This is the expert-routing / conditionally-skipped-layer shape, and
 * the one the pass-boundary heuristic cannot infer: no weight repeats, so no
 * boundary is detected, and the cached int8 from the previous pass still keys
 * as valid. The dispatch must refuse the cache rather than multiply the stale
 * quantization, which is what this asserts -- the third weight's result has to
 * follow the activation it was actually given.
 */
TEST(v8cGemm, quant_cache_refused_when_pass_opens_with_a_new_weight_p) {
  constexpr unsigned int M = 4, K = 128, N = 32;
  nntrainer::Tensor wa = makeQs4cxWeight(K, N, "v8c_moe_wa", 0x51ED0020u);
  nntrainer::Tensor wb = makeQs4cxWeight(K, N, "v8c_moe_wb", 0x51ED0021u);
  nntrainer::Tensor wc = makeQs4cxWeight(K, N, "v8c_moe_wc", 0x51ED0022u);
  nntrainer::Tensor input(nntrainer::TensorDim(1, 1, M, K), true);
  nntrainer::Tensor out(nntrainer::TensorDim(1, 1, M, N), true);
  nntrainer::Tensor ref(nntrainer::TensorDim(1, 1, M, N), true);

  // Pass one dispatches wa then wb.
  fillActivation(input, 303u);
  out.setZero();
  if (!nntrainer::dotCl_v8c(input, wa, out))
    GTEST_SKIP() << "no OpenCL device took the v8c dispatch";
  out.setZero();
  ASSERT_TRUE(nntrainer::dotCl_v8c(input, wb, out));

  // Pass two opens with wc, dispatched for the first time, over an activation
  // rewritten in place. Nothing in the generation state distinguishes this
  // from a second FC of pass one.
  fillActivation(input, 404u);
  ref.setZero();
  input.dot(wc, ref, false, false);
  out.setZero();
  ASSERT_TRUE(nntrainer::dotCl_v8c(input, wc, out));
  EXPECT_LT(relativeError(out, ref, M, N), kQuantTolerance);
}

namespace {

/**
 * @brief A per-channel ASYMMETRIC int4 weight in the QS4CX layout plus its
 * zero-point correction.
 * @details w[n][k] = s[n] * (q - z[n]) with q in [0, 15] and z[n] in [3, 12].
 * The tensor carries nibble q (= signed code q - 8) and scale s[n]; `zp`
 * holds the integer zero points as floats. `dense` is the exact [N][K]
 * asymmetric weight and `dense_sym` the symmetric reading s * (q - 8) of the
 * same bytes, both for the host reference.
 */
struct AsymWeight {
  nntrainer::Tensor w;
  std::vector<float> zp;
  std::vector<double> dense, dense_sym;
};

AsymWeight makeAsymWeight(unsigned int K, unsigned int N,
                          const std::string &name, uint32_t seed) {
  AsymWeight a;
  nntrainer::TensorDim dim(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QS4CX});
  a.w = nntrainer::Tensor(dim, true, nntrainer::Initializer::NONE, name);
  uint8_t *nib = a.w.getData<uint8_t>();
  float *scales = a.w.getScale<float>();
  const unsigned int Kh = (K + 1) / 2;
  Lcg rng(seed);
  a.zp.resize(N);
  a.dense.assign((size_t)N * K, 0.0);
  a.dense_sym.assign((size_t)N * K, 0.0);
  for (unsigned int n = 0; n < N; ++n) {
    const float s = rng.uniform(0.005f, 0.02f);
    const int z = 3 + (int)(rng.next() % 10u); // [3, 12]
    scales[n] = s;
    a.zp[n] = (float)z;
    for (unsigned int k = 0; k < K; ++k) {
      const int q = (int)(rng.next() % 16u);
      uint8_t &b = nib[(size_t)n * Kh + k / 2];
      b = (k & 1u) ? (uint8_t)((b & 0x0Fu) | (q << 4))
                   : (uint8_t)((b & 0xF0u) | q);
      a.dense[(size_t)n * K + k] = (double)s * (q - z);
      a.dense_sym[(size_t)n * K + k] = (double)s * (q - 8);
    }
  }
  return a;
}

/** @brief y[m][n] = sum_k x[m][k] * W[n][k] in double. */
std::vector<double> refDense(const float *x, const std::vector<double> &W,
                             unsigned int M, unsigned int K, unsigned int N) {
  std::vector<double> y((size_t)M * N, 0.0);
  for (unsigned int m = 0; m < M; ++m)
    for (unsigned int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (unsigned int k = 0; k < K; ++k)
        acc += (double)x[(size_t)m * K + k] * W[(size_t)n * K + k];
      y[(size_t)m * N + n] = acc;
    }
  return y;
}

/** @brief Largest |out - ref| relative to each row's peak |ref|. */
float relErr(const float *out, const std::vector<double> &ref, unsigned int M,
             unsigned int N) {
  float worst = 0.0f;
  for (unsigned int m = 0; m < M; ++m) {
    double peak = 1e-6;
    for (unsigned int n = 0; n < N; ++n)
      peak = std::max(peak, std::fabs(ref[(size_t)m * N + n]));
    for (unsigned int n = 0; n < N; ++n) {
      const size_t i = (size_t)m * N + n;
      worst =
        std::max(worst, (float)(std::fabs((double)out[i] - ref[i]) / peak));
    }
  }
  return worst;
}

/**
 * @brief One asymmetric-weight case: the dispatch without zero points must
 * equal the symmetric reading of the bytes, and the one with them the
 * asymmetric weight -- each within the int8 activation quantisation error,
 * while the symmetric output misses the asymmetric reference by far more.
 * @return false when no OpenCL device took the dispatch.
 */
bool runAsymCase(unsigned int M, unsigned int K, unsigned int N,
                 const std::string &name, uint32_t seed,
                 bool via_correction = false) {
  AsymWeight a = makeAsymWeight(K, N, name, seed);
  nntrainer::Tensor input(nntrainer::TensorDim(1, 1, M, K), true);
  fillActivation(input, seed + 3u);
  const std::vector<double> ref =
    refDense(input.getData<float>(), a.dense, M, K, N);
  const std::vector<double> ref_sym =
    refDense(input.getData<float>(), a.dense_sym, M, K, N);

  nntrainer::Tensor sym(nntrainer::TensorDim(1, 1, M, N), true);
  sym.setZero();
  if (!nntrainer::dotCl_v8c(input, a.w, sym))
    return false;
  EXPECT_LT(relErr(sym.getData<float>(), ref_sym, M, N), kQuantTolerance)
    << name << ": symmetric reading";
  EXPECT_GT(relErr(sym.getData<float>(), ref, M, N), 10 * kQuantTolerance)
    << name << ": the case must need the correction";

  if (via_correction) {
    // The loader-side form: c[n] = s[n] * (8 - z[n]).
    std::vector<float> c(N);
    const float *sc = a.w.getScale<float>();
    for (unsigned int n = 0; n < N; ++n)
      c[n] = sc[n] * (8.0f - a.zp[n]);
    EXPECT_TRUE(
      nntrainer::dotCl_v8c_set_zero_point_correction(a.w, c.data(), N));
  } else {
    EXPECT_TRUE(
      nntrainer::dotCl_v8c_set_weight_zero_points(a.w, a.zp.data(), N));
  }
  nntrainer::Tensor out(nntrainer::TensorDim(1, 1, M, N), true);
  out.setZero();
  EXPECT_TRUE(nntrainer::dotCl_v8c(input, a.w, out));
  const float err = relErr(out.getData<float>(), ref, M, N);
  std::cout << name << " M=" << M << " K=" << K << " N=" << N
            << " corrected rel err " << err << std::endl;
  EXPECT_LT(err, kQuantTolerance) << name << ": corrected";

  // Removing the zero points restores the symmetric output bit for bit.
  EXPECT_TRUE(nntrainer::dotCl_v8c_set_weight_zero_points(a.w, nullptr, N));
  nntrainer::Tensor again(nntrainer::TensorDim(1, 1, M, N), true);
  again.setZero();
  EXPECT_TRUE(nntrainer::dotCl_v8c(input, a.w, again));
  EXPECT_EQ(0, std::memcmp(again.getData<float>(), sym.getData<float>(),
                           sizeof(float) * (size_t)M * N))
    << name << ": uncorrected output changed";
  return true;
}

} // namespace

/**
 * @brief Per-channel asymmetric weight through the decode GEMV (M = 1).
 */
TEST(v8cGemm, asym_zero_point_gemv_m1_p) {
  if (!runAsymCase(1, 256, 64, "v8c_asym_m1_w", 0x5A5A0001u))
    GTEST_SKIP() << "no OpenCL device took the v8c dispatch";
}

/**
 * @brief Asymmetric weight through the tiled GEMM with padded rows (M = 3),
 * registered through the c[n] = s[n] * (8 - z[n]) form.
 */
TEST(v8cGemm, asym_zero_point_gemm_m3_p) {
  if (!runAsymCase(3, 256, 64, "v8c_asym_m3_w", 0x5A5A0002u, true))
    GTEST_SKIP() << "no OpenCL device took the v8c dispatch";
}

/**
 * @brief Asymmetric weight through the prefill GEMM (the matrix-engine kernel
 * where the device has one, the dp4a kernel otherwise).
 */
TEST(v8cGemm, asym_zero_point_gemm_m70_p) {
  if (!runAsymCase(70, 256, 128, "v8c_asym_m70_w", 0x5A5A0003u))
    GTEST_SKIP() << "no OpenCL device took the v8c dispatch";
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
