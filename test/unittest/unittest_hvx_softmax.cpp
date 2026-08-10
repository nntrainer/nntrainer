// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_hvx_softmax.cpp
 * @date   05 Aug 2026
 * @brief  Device test: HVX exp and softmax match a double reference
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Runs on an Android device only. Requires libnntr_hvx_skel.so on
 * ADSP_LIBRARY_PATH. See test/htp/build.sh.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "nntr_hvx.h"

#include "mha_htp_host_model.h"

namespace {

/**
 * @brief Offset AEEStdErr.h adds to every AEE_* code on the DSP side.
 *
 * DSP-side skel code is compiled with __hexagon__ defined, where
 * AEEStdErr.h adds this offset to every AEE_* code before it crosses the
 * FastRPC boundary. This host binary is not compiled with __hexagon__, so
 * AEE_EBADPARM here is plain 14 -- the offset has to be added back when
 * checking an error the DSP returned.
 */
constexpr int kDspOffset = 0x80000400;

/** @brief Render a FastRPC error as hex so the code is searchable. */
std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

/**
 * @brief Softmax reference in double, row by row.
 *
 * double rather than the nntrainer CPU softmax: comparing two f32
 * approximations would blur where the error comes from.
 */
std::vector<float> ref_softmax(const std::vector<float> &x, uint32_t m,
                               uint32_t k, float scale) {
  std::vector<float> y(x.size());
  std::vector<double> e(k);

  for (uint32_t r = 0; r < m; ++r) {
    const float *xr = x.data() + static_cast<size_t>(r) * k;
    double mx = -std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < k; ++i) {
      mx = std::max(mx, static_cast<double>(xr[i]) * scale);
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < k; ++i) {
      e[i] = std::exp(static_cast<double>(xr[i]) * scale - mx);
      sum += e[i];
    }
    for (uint32_t i = 0; i < k; ++i) {
      y[static_cast<size_t>(r) * k + i] = static_cast<float>(e[i] / sum);
    }
  }
  return y;
}

/**
 * @brief Opens an unsigned-PD CDSP session for each test.
 *
 * A failure here is a hard FAIL rather than a skip: proving the DSP comes
 * up on the device is part of what this test measures, so a quiet skip
 * would report success for it.
 */
class HtpSession : public ::testing::Test {
protected:
  void SetUp() override {
    remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                     &unsigned_pd, sizeof(unsigned_pd));
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);

    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &handle_);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "nntr_hvx_open failed: " << hex(err)
      << " -- is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?";
  }

  void TearDown() override {
    if (handle_) {
      nntr_hvx_close(handle_);
    }
  }

  remote_handle64 handle_ = 0;
};

class HvxExp : public HtpSession {};
class HvxSoftmax : public HtpSession {};

TEST_F(HvxExp, RejectsNonVectorLength) {
  const int n = 33;
  std::vector<float> in(n, 1.0f), out(n, 0.0f);

  int err = nntr_hvx_exp_f32(handle_, in.data(), n, out.data(), n);
  EXPECT_EQ(err, AEE_EBADPARM + kDspOffset)
    << "expected EBADPARM, got " << hex(err);
}

TEST_F(HvxExp, MatchesDoubleOverTheNormalRange) {
  // 8192 is a multiple of 32. Sweep stops at -87: below that the true
  // value is subnormal and the kernel flushes it to zero by contract.
  const int n = 8192;
  std::vector<float> in(n), out(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    in[i] = -87.0f + 175.0f * static_cast<float>(i) / static_cast<float>(n - 1);
  }

  int err = nntr_hvx_exp_f32(handle_, in.data(), n, out.data(), n);
  ASSERT_EQ(err, AEE_SUCCESS) << "exp_f32 failed: " << hex(err);

  double worst = 0.0;
  int worst_i = 0;
  for (int i = 0; i < n; ++i) {
    const double ref = std::exp(static_cast<double>(in[i]));
    const double rel = std::abs(static_cast<double>(out[i]) - ref) / ref;
    if (rel > worst) {
      worst = rel;
      worst_i = i;
    }
  }
  EXPECT_LT(worst, 1e-6) << "worst relative error at x=" << in[worst_i]
                         << ": got " << out[worst_i] << ", want "
                         << std::exp(static_cast<double>(in[worst_i]));
}

TEST_F(HvxExp, ExactlyOneAtZero) {
  const int n = 32;
  const std::vector<float> in(n, 0.0f);
  std::vector<float> out(n, -1.0f);

  int err = nntr_hvx_exp_f32(handle_, in.data(), n, out.data(), n);
  ASSERT_EQ(err, AEE_SUCCESS) << "exp_f32 failed: " << hex(err);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], 1.0f) << "lane " << i;
  }
}

TEST_F(HvxExp, FlushesFarNegativeToZero) {
  // softmax feeds x - max, which can be arbitrarily negative. The low
  // clamp inside hvx_exp_sf is what keeps the range reduction from
  // overflowing its own valid domain on these.
  const int n = 32;
  std::vector<float> in(n, -90.0f);
  in[1] = -200.0f;
  in[2] = -1e30f;
  in[3] = -3.4e38f;
  std::vector<float> out(n, 1.0f);

  int err = nntr_hvx_exp_f32(handle_, in.data(), n, out.data(), n);
  ASSERT_EQ(err, AEE_SUCCESS) << "exp_f32 failed: " << hex(err);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], 0.0f) << "lane " << i << " x=" << in[i];
  }
}

TEST_F(HvxSoftmax, RejectsLengthMismatch) {
  const uint32_t m = 2, k = 32;
  std::vector<float> in(m * k, 1.0f), out(k, 0.0f);

  int err = nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, in.data(),
                                 static_cast<int>(in.size()), out.data(),
                                 static_cast<int>(out.size()));
  EXPECT_EQ(err, AEE_EBADPARM + kDspOffset)
    << "expected EBADPARM, got " << hex(err);
}

TEST_F(HvxSoftmax, MatchesDoubleForOneFullRow) {
  const uint32_t m = 1, k = 1024;
  std::vector<float> in(m * k), out(m * k, 0.0f);

  std::mt19937 rng(20260805u);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
  for (auto &v : in) {
    v = dist(rng);
  }

  int err = nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, in.data(),
                                 static_cast<int>(in.size()), out.data(),
                                 static_cast<int>(out.size()));
  ASSERT_EQ(err, AEE_SUCCESS) << "softmax_f32 failed: " << hex(err);

  const std::vector<float> ref = ref_softmax(in, m, k, 1.0f);
  double worst = 0.0;
  double sum = 0.0;
  for (uint32_t i = 0; i < k; ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(out[i]) - ref[i]));
    sum += out[i];
  }
  EXPECT_LT(worst, 1e-6) << "worst absolute error";
  EXPECT_NEAR(sum, 1.0, 1e-6) << "row does not sum to 1";
}

TEST_F(HvxSoftmax, IsInvariantToAConstantShift) {
  // Adding a constant to every element must not change the result. This is
  // what the max subtraction buys. exp(100) overflows f32, so 1e2 still
  // forces max subtraction; going larger (e.g. 1e4) rounds f32 inputs to
  // ULP ~0.001 on the host before the DSP sees them, making 1e-6
  // unachievable regardless of kernel precision.
  const uint32_t m = 1, k = 256;
  std::vector<float> base(m * k), shifted(m * k);
  std::vector<float> out_base(m * k, 0.0f), out_shift(m * k, 0.0f);

  std::mt19937 rng(7u);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (uint32_t i = 0; i < k; ++i) {
    base[i] = dist(rng);
    shifted[i] = base[i] + 1e2f;
  }

  const int n = static_cast<int>(m * k);
  ASSERT_EQ(nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, base.data(), n,
                                 out_base.data(), n),
            AEE_SUCCESS);
  ASSERT_EQ(nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, shifted.data(), n,
                                 out_shift.data(), n),
            AEE_SUCCESS);

  for (uint32_t i = 0; i < k; ++i) {
    EXPECT_NEAR(out_base[i], out_shift[i], 1e-6) << "lane " << i;
  }
}

TEST_F(HvxSoftmax, SpreadsUniformlyForEqualInputs) {
  const uint32_t m = 1, k = 64;
  const std::vector<float> in(m * k, 5.0f);
  std::vector<float> out(m * k, 0.0f);

  const int n = static_cast<int>(m * k);
  ASSERT_EQ(
    nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, in.data(), n, out.data(), n),
    AEE_SUCCESS);

  for (uint32_t i = 0; i < k; ++i) {
    EXPECT_NEAR(out[i], 1.0f / 64.0f, 1e-6) << "lane " << i;
  }
}

TEST_F(HvxSoftmax, CollapsesOntoADominantElement) {
  // Every other term underflows exp. Without the guard in hvx_exp_sf these
  // come back as garbage rather than zero.
  const uint32_t m = 1, k = 32;
  std::vector<float> in(m * k, 0.0f);
  in[7] = 100.0f;
  std::vector<float> out(m * k, -1.0f);

  const int n = static_cast<int>(m * k);
  ASSERT_EQ(
    nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, in.data(), n, out.data(), n),
    AEE_SUCCESS);

  for (uint32_t i = 0; i < k; ++i) {
    EXPECT_NEAR(out[i], i == 7 ? 1.0f : 0.0f, 1e-6) << "lane " << i;
  }
}

TEST_F(HvxSoftmax, HandlesRowLengthsThatAreNotWholeVectors) {
  // 1 is below one vector; 31 is one short; 33 is one over; 100 is three
  // vectors plus four.
  for (const uint32_t k : {1u, 31u, 33u, 100u}) {
    const uint32_t m = 1;
    std::vector<float> in(k), out(k, -1.0f);

    std::mt19937 rng(k);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    for (auto &v : in) {
      v = dist(rng);
    }

    const int n = static_cast<int>(k);
    ASSERT_EQ(nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, in.data(), n,
                                   out.data(), n),
              AEE_SUCCESS)
      << "k=" << k;

    const std::vector<float> ref = ref_softmax(in, m, k, 1.0f);
    double sum = 0.0;
    for (uint32_t i = 0; i < k; ++i) {
      EXPECT_NEAR(out[i], ref[i], 1e-6) << "k=" << k << " lane " << i;
      sum += out[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-6) << "k=" << k << " does not sum to 1";
  }
}

TEST_F(HvxSoftmax, DoesNotWritePastTheEndOfTheBuffer) {
  // k=33 means the tail vector covers 32 lanes but only 1 is real. A
  // full-vector store would clobber 31 floats of whatever follows.
  const uint32_t m = 1, k = 33;
  const uint32_t guard = 64;
  std::vector<float> buf(k + guard, 12345.0f);
  std::vector<float> in(k, 1.0f);

  const int n = static_cast<int>(k);
  ASSERT_EQ(
    nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, in.data(), n, buf.data(), n),
    AEE_SUCCESS);

  for (uint32_t i = 0; i < guard; ++i) {
    EXPECT_EQ(buf[k + i], 12345.0f) << "clobbered guard word " << i;
  }
}

TEST_F(HvxSoftmax, KeepsRowsIndependent) {
  // Different distributions per row: if one row's max or sum leaked into
  // the next, these would not all normalize to 1.
  const uint32_t m = 8, k = 100;
  std::vector<float> in(m * k), out(m * k, -1.0f);

  std::mt19937 rng(31u);
  for (uint32_t r = 0; r < m; ++r) {
    std::uniform_real_distribution<float> dist(-2.0f * (r + 1), 2.0f * (r + 1));
    for (uint32_t i = 0; i < k; ++i) {
      in[(size_t)r * k + i] = dist(rng);
    }
  }

  const int n = static_cast<int>(in.size());
  ASSERT_EQ(
    nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, in.data(), n, out.data(), n),
    AEE_SUCCESS);

  const std::vector<float> ref = ref_softmax(in, m, k, 1.0f);
  for (uint32_t r = 0; r < m; ++r) {
    double sum = 0.0;
    for (uint32_t i = 0; i < k; ++i) {
      const size_t j = (size_t)r * k + i;
      EXPECT_NEAR(out[j], ref[j], 1e-6) << "row " << r << " lane " << i;
      sum += out[j];
    }
    EXPECT_NEAR(sum, 1.0, 1e-6) << "row " << r << " does not sum to 1";
  }
}

TEST_F(HvxSoftmax, MatchesOutOfPlaceWhenComputedInPlace) {
  const uint32_t m = 4, k = 65;
  std::vector<float> in(m * k);

  std::mt19937 rng(99u);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  for (auto &v : in) {
    v = dist(rng);
  }

  std::vector<float> out_of_place(m * k, 0.0f);
  std::vector<float> in_place = in;

  const int n = static_cast<int>(in.size());
  ASSERT_EQ(nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, in.data(), n,
                                 out_of_place.data(), n),
            AEE_SUCCESS);
  ASSERT_EQ(nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, in_place.data(), n,
                                 in_place.data(), n),
            AEE_SUCCESS);

  for (uint32_t i = 0; i < m * k; ++i) {
    EXPECT_EQ(in_place[i], out_of_place[i]) << "lane " << i;
  }
}

TEST_F(HvxSoftmax, HandlesANegativeScale) {
  // A negative scale flips which element is the maximum. Taking
  // scale*max(x) instead of max(x*scale) would subtract the wrong value
  // and overflow exp.
  const uint32_t m = 2, k = 48;
  std::vector<float> in(m * k), out(m * k, -1.0f);

  std::mt19937 rng(5u);
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  for (auto &v : in) {
    v = dist(rng);
  }

  const int n = static_cast<int>(in.size());
  ASSERT_EQ(
    nntr_hvx_softmax_f32(handle_, m, k, 0u, -1.0f, in.data(), n, out.data(), n),
    AEE_SUCCESS);

  const std::vector<float> ref = ref_softmax(in, m, k, -1.0f);
  for (uint32_t i = 0; i < m * k; ++i) {
    EXPECT_NEAR(out[i], ref[i], 1e-6) << "lane " << i;
  }
}

TEST_F(HvxSoftmax, LeavesRowsBeforeTheRangeAlone) {
  // m_first=2 of 5 rows. FastRPC zeroes the rout buffer before sending it
  // to the DSP, so the untouched rows come back 0 regardless -- an in-place
  // (inrout) IDL entry would be needed to see them directly. Instead, call
  // twice -- (0, M) and (m_first, M) -- and compare the rows the range call
  // should have produced.
  const uint32_t m = 5, k = 64, m_first = 2;
  std::vector<float> in(m * k, 1.0f);
  std::vector<float> out_full(m * k, 0.0f);
  std::vector<float> out_range(m * k, 0.0f);

  const int n = static_cast<int>(in.size());
  ASSERT_EQ(nntr_hvx_softmax_f32(handle_, m, k, 0u, 1.0f, in.data(), n,
                                 out_full.data(), n),
            AEE_SUCCESS);
  ASSERT_EQ(nntr_hvx_softmax_f32(handle_, m, k, m_first, 1.0f, in.data(), n,
                                 out_range.data(), n),
            AEE_SUCCESS);

  for (uint32_t i = m_first * k; i < m * k; ++i) {
    EXPECT_NEAR(out_range[i], out_full[i], 1e-6)
      << "range result differs from full result at lane " << i;
  }
}

} // namespace

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}

/**===========================================================================
 * Blocked masked softmax -- PHASE B of the fused attention loop.
 *
 * The oracle is mha_softmax_blocked_ref, the same host model the whole flash
 * loop is specified against, so this gates the DSP kernel against the
 * specification rather than against a second opinion written beside it.
 *
 * Nothing asserts per element: the whole matrix runs, the worst case is
 * carried out, and the bounds are checked once at the end. A per-element
 * assertion aborts at the first shape and then reports nothing about the
 * later ones, which is how a matrix silently stops covering what it claims.
 *=========================================================================*/

class HvxSoftmaxBlocked : public HtpSession {};

namespace {

/** @brief Worst case seen across the matrix, with the shape that produced it.
 */
struct BlockedWorst {
  double p_abs = 0.0;    /**< max |p_dev - p_ref|, absolute */
  double l_margin = 0.0; /**< max |l_dev - l_ref| / summation-order bound */
  double l_abs = 0.0;    /**< the raw l difference behind l_margin */
  std::string p_where, l_where;
};

/**
 * @brief Bound on how far two summation ORDERS of the same n non-negative f32
 *        values may legitimately land apart.
 *
 * Two terms, and the first device run showed both are needed:
 *
 *   n * eps * sum  -- reassociation. The host model adds in kv order, one
 *     scalar at a time; the kernel adds into 32 qf32 lanes and reduces at the
 *     end. Neither is "the" answer; both approximate the same exact sum.
 *     Dominates on long rows (kv=1023 landed 1.06e-6 relative here).
 *
 *   1e-6 * sum     -- hvx_exp_sf's own documented relative error. Every term
 *     of l is an exp, and on a SHORT row there is no reassociation at all, so
 *     this is the only term left. Leaving it out put the worst margin at
 *     0.998 on a two-term row -- passing, but one rounding away from red and
 *     for a reason that has nothing to do with summation order.
 *
 * A logic bug (a mask off by one, a dropped block, a mishandled sink) moves l
 * by whole terms, i.e. O(1/n) to O(1) relative, and is still caught by orders
 * of magnitude.
 */
double sum_order_bound(uint32_t n_terms, double sum) {
  return ((double)n_terms * 1.1920928955078125e-7 + 1e-6) * sum;
}

/** @brief One case of the blocked softmax matrix, device against host model. */
void check_blocked(remote_handle64 handle, uint32_t kv, uint32_t T, uint32_t M,
                   uint32_t window, bool with_sink, BlockedWorst *w) {
  const uint32_t n_seg = (kv + T - 1u) / T;
  const uint32_t band = n_seg * M * T;
  const float scale = 0.125f;

  std::mt19937 rng(kv * 7919u + T * 131u + M * 17u + window +
                   (with_sink ? 1u : 0u));
  std::uniform_real_distribution<float> d(-6.0f, 6.0f);

  /** Poison the padded tail past kv: a masking bug reads a value that cannot
   * be mistaken for a plausible score. */
  std::vector<float> in(band, 1e30f);
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t pos = 0; pos < kv; ++pos) {
      in[(size_t)(pos / T) * M * T + (size_t)m * T + (pos % T)] = d(rng);
    }
  }

  std::vector<uint32_t> begin(M), end(M);
  for (uint32_t m = 0; m < M; ++m) {
    const uint32_t e = kv - (m % kv);
    end[m] = e;
    begin[m] = (window != 0u && window < e) ? (e - window) : 0u;
  }
  std::vector<float> sink(M);
  for (uint32_t m = 0; m < M; ++m) {
    sink[m] = d(rng) * 0.3f;
  }

  std::vector<float> out(band, 0.0f), l(M, 0.0f);
  const int err = nntr_hvx_softmax_blocked_f32(
    handle, n_seg, T, M, scale, in.data(), (int)band, begin.data(), (int)M,
    end.data(), (int)M, with_sink ? sink.data() : nullptr,
    with_sink ? (int)M : 0, out.data(), (int)band, l.data(), (int)M);

  std::ostringstream shape;
  shape << "kv=" << kv << " T=" << T << " M=" << M << " window=" << window
        << " sink=" << (with_sink ? 1 : 0);
  ASSERT_EQ(err, AEE_SUCCESS) << shape.str() << " -> " << hex(err);

  std::vector<const float *> seg(n_seg);
  for (uint32_t j = 0; j < n_seg; ++j) {
    seg[j] = in.data() + (size_t)j * M * T;
  }
  const MhaSoftmaxOut ref = mha_softmax_blocked_ref(
    seg, n_seg, T, M, scale, begin, end, with_sink ? sink.data() : nullptr);

  for (size_t i = 0; i < band; ++i) {
    const double e = std::fabs((double)out[i] - (double)ref.p[i]);
    if (e > w->p_abs) {
      w->p_abs = e;
      w->p_where = shape.str();
    }
  }
  for (uint32_t m = 0; m < M; ++m) {
    const double e = std::fabs((double)l[m] - (double)ref.l[m]);
    const double bound = sum_order_bound(end[m] - begin[m], (double)ref.l[m]);
    const double margin = (bound > 0.0) ? (e / bound) : e;
    if (margin > w->l_margin) {
      w->l_margin = margin;
      w->l_abs = e;
      w->l_where = shape.str() + " m=" + std::to_string(m);
    }
  }
}

} // namespace

TEST_F(HvxSoftmaxBlocked, MatchesHostModelOverTheShapeMatrix) {
  BlockedWorst w;
  size_t cases = 0;

  for (uint32_t kv : {1u, 31u, 32u, 33u, 255u, 256u, 257u, 1023u, 1024u}) {
    for (uint32_t T : {32u, 256u}) {
      for (uint32_t M : {1u, 7u, 64u}) {
        /** window T-1, T, T+1 is the off-by-one trap in the block arithmetic
         * and is mandatory; kv-1 and kv exercise the whole-band ends. */
        for (uint32_t window : {0u, 1u, T - 1u, T, T + 1u, kv - 1u, kv}) {
          for (int s = 0; s < 2; ++s) {
            check_blocked(handle_, kv, T, M, window, s != 0, &w);
            if (::testing::Test::HasFatalFailure()) {
              return;
            }
            ++cases;
          }
        }
      }
    }
  }

  std::cout << "BLOCKED_FIELD path=softmax_blocked field=cases value=" << cases
            << std::endl;
  std::cout << "BLOCKED_FIELD path=softmax_blocked field=max_abs_err_p value="
            << w.p_abs << std::endl;
  std::cout << "BLOCKED_FIELD path=softmax_blocked field=max_abs_err_l value="
            << w.l_abs << std::endl;
  std::cout
    << "BLOCKED_FIELD path=softmax_blocked field=l_sum_order_margin value="
    << w.l_margin << std::endl;

  /** p is bounded absolutely: values are in (0, 1] and hvx_exp_sf's own spec is
   * a relative error of 1e-6 over this domain, so 1e-6 absolute is the kernel
   * inheriting exactly its exp's accuracy and nothing worse. */
  EXPECT_LE(w.p_abs, 1e-6) << "worst at " << w.p_where;
  /** l is bounded by reassociation, not by a picked number -- see
   * sum_order_bound. A margin at or under 1 means the two implementations
   * differ only in the order they added the same values. */
  EXPECT_LE(w.l_margin, 1.0)
    << "worst at " << w.l_where << ", raw diff " << w.l_abs;
}

TEST_F(HvxSoftmaxBlocked, RejectsBadShapes) {
  std::vector<float> band(4 * 8, 0.0f), out(4 * 8, 0.0f), l(4, 0.0f);
  std::vector<uint32_t> b(4, 0u), e(4, 8u);

  EXPECT_EQ(nntr_hvx_softmax_blocked_f32(handle_, 0u, 8u, 4u, 1.0f, band.data(),
                                         32, b.data(), 4, e.data(), 4, nullptr,
                                         0, out.data(), 32, l.data(), 4),
            AEE_EBADPARM + kDspOffset)
    << "n_seg = 0 must be rejected";
  EXPECT_EQ(nntr_hvx_softmax_blocked_f32(handle_, 1u, 8u, 4u, 1.0f, band.data(),
                                         31, b.data(), 4, e.data(), 4, nullptr,
                                         0, out.data(), 32, l.data(), 4),
            AEE_EBADPARM + kDspOffset)
    << "band length must equal n_seg*M*T";
}
