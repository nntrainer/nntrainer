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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "m1_ops_det.h"
#include "nntr_hvx.h"
#include "swiglu_det.h"

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

class HvxSwigluDet : public HtpSession {};

namespace {

/**
 * @brief The scalar specification, taken from the header both machines
 *        implement rather than copied.
 *
 * An earlier version of this file carried its own transcription of
 * swiglu_det.h's arithmetic. That is one copy too many: the whole point of
 * the test is to catch a divergence between implementations of one
 * specification, and a private copy of the specification cannot do that --
 * it can only catch divergence from itself. swiglu_det.h's scalar path
 * stores every intermediate through a volatile for exactly the reason this
 * test needed to, so nothing is lost by deferring to it.
 */
inline float exp_det_ref(float x) { return swiglu_det_exp(x); }
inline float recip_det_ref(float d) { return swiglu_det_recip(d); }
inline float swiglu_det_ref(float g, float u) { return swiglu_det_one(g, u); }
inline float dadd(float a, float b) { return swiglu_det_add(a, b); }
inline float dsub(float a, float b) { return swiglu_det_sub(a, b); }

inline int32_t bits_of(float f) {
  int32_t i;
  std::memcpy(&i, &f, sizeof(i));
  return i;
}

} // namespace

TEST_F(HvxSwigluDet, RejectsNonVectorLength) {
  const int n = 33;
  std::vector<float> g(n, 1.0f), u(n, 1.0f), o(n), e(n), r(n);
  int err = nntr_hvx_swiglu_det_f32(handle_, g.data(), n, u.data(), n, o.data(),
                                    n, e.data(), n, r.data(), n);
  EXPECT_EQ(err, AEE_EBADPARM + kDspOffset)
    << "expected EBADPARM, got " << hex(err);
}

/**
 * @brief [A1] The gate the whole deterministic-SwiGLU approach rests on.
 *
 * Not an SNR test and not a tolerance test -- the output must be BIT
 * IDENTICAL to a scalar reference running the same specification. That is
 * the only property that drives the quantization flip count to zero (doc 44
 * section 3.3), and a value that is merely close does not have it.
 *
 * A failure here is informative rather than fatal: the per-stage counts say
 * whether HVX's Vsf multiply and add round differently from ARM's f32 ones
 * (exp and recip both off), or whether only the tail does. If exp_det
 * mismatches, hvx_exp_f32.h's remark that chained Vsf loses precision meant
 * Vsf is not IEEE-correctly-rounded, and bit identity is not reachable on
 * this hardware -- which is worth knowing before anything is built on it.
 */
TEST_F(HvxSwigluDet, MatchesScalarBitExact) {
  // A spread that covers what a real SwiGLU sees plus both clamps: the
  // saturating tails, the sign change, and the dense band around zero
  // where sigmoid actually varies.
  const int n = 8192;
  std::vector<float> g(n), u(n), o(n, 0.0f), e(n, 0.0f), r(n, 0.0f);
  std::mt19937 rng(0xA1A1A1A1u);
  std::uniform_real_distribution<float> small(-8.0f, 8.0f);
  for (int i = 0; i < n; ++i) {
    if (i < 64) {
      g[i] = -200.0f + 3.0f * static_cast<float>(i); // through both clamps
    } else if (i < 128) {
      g[i] = 100.0f - 3.0f * static_cast<float>(i - 64);
    } else {
      g[i] = small(rng);
    }
    u[i] = small(rng);
  }

  int err = nntr_hvx_swiglu_det_f32(handle_, g.data(), n, u.data(), n, o.data(),
                                    n, e.data(), n, r.data(), n);
  ASSERT_EQ(err, AEE_SUCCESS) << "swiglu_det_f32 failed: " << hex(err);

  int bad_exp = 0, bad_recip = 0, bad_out = 0;
  int first_bad = -1;
  for (int i = 0; i < n; ++i) {
    const float ref_e = exp_det_ref(dsub(0.0f, g[i]));
    const float ref_r = recip_det_ref(dadd(1.0f, ref_e));
    const float ref_o = swiglu_det_ref(g[i], u[i]);
    const bool de = bits_of(e[i]) != bits_of(ref_e);
    const bool dr = bits_of(r[i]) != bits_of(ref_r);
    const bool dobad = bits_of(o[i]) != bits_of(ref_o);
    bad_exp += de ? 1 : 0;
    bad_recip += dr ? 1 : 0;
    bad_out += dobad ? 1 : 0;
    if (first_bad < 0 && (de || dr || dobad)) {
      first_bad = i;
      std::cout << "SWIGLU_DET first mismatch i=" << i << " g=" << g[i]
                << " u=" << u[i] << "\n  exp   dsp=" << std::hexfloat << e[i]
                << " ref=" << ref_e << "\n  recip dsp=" << r[i]
                << " ref=" << ref_r << "\n  out   dsp=" << o[i]
                << " ref=" << ref_o << std::defaultfloat << std::endl;
    }
  }
  std::cout << "SWIGLU_DET_FIELD bad_exp=" << bad_exp
            << " bad_recip=" << bad_recip << " bad_out=" << bad_out << " of "
            << n << std::endl;

  EXPECT_EQ(bad_exp, 0) << "hvx_exp_det_sf differs from the scalar spec -- "
                           "HVX Vsf does not round like ARM f32";
  EXPECT_EQ(bad_recip, 0) << "hvx_recip_det_sf differs from the scalar spec";
  EXPECT_EQ(bad_out, 0) << "hvx_swiglu_det_sf differs from the scalar spec";
}

/**
 * @brief [A1] The ARM half of the same specification, on the same data.
 *
 * This test needs no DSP: it compares swiglu_det.h's NEON path against
 * swiglu_det.h's scalar path, both on this phone's own CPU. What it is
 * really looking for is a fused multiply-add. The scalar path stores every
 * intermediate through a volatile and cannot be contracted; the NEON path
 * has only `#pragma clang fp contract(off)` between it and an fmla that
 * would round once where the DSP rounds twice. That failure is silent --
 * the kernel still computes an accurate SwiGLU, just not the same bits --
 * so nothing but a comparison like this one would find it.
 *
 * Together with MatchesScalarBitExact this closes the triangle: DSP ==
 * scalar and NEON == scalar give DSP == NEON, which is the property the
 * fused MoE path actually needs.
 */
TEST(SwigluDetNeon, MatchesScalar) {
#ifndef SWIGLU_DET_HAS_NEON
  GTEST_SKIP() << "built without the NEON path -- nothing to compare";
#else
  const int n = 8192;
  std::vector<float> g(n), u(n), out(n, 0.0f);
  std::mt19937 rng(0xA1A1A1A1u);
  std::uniform_real_distribution<float> small(-8.0f, 8.0f);
  for (int i = 0; i < n; ++i) {
    // The same spread MatchesScalarBitExact uses, including the five
    // inputs that land on the exp clamp -- the exact place the DSP side
    // was wrong.
    if (i < 64) {
      g[i] = -200.0f + 3.0f * static_cast<float>(i);
    } else if (i < 128) {
      g[i] = 100.0f - 3.0f * static_cast<float>(i - 64);
    } else {
      g[i] = small(rng);
    }
    u[i] = small(rng);
  }

  swiglu_det(static_cast<unsigned int>(n), out.data(), g.data(), u.data());

  int bad = 0;
  int first_bad = -1;
  for (int i = 0; i < n; ++i) {
    const float ref = swiglu_det_one(g[i], u[i]);
    if (bits_of(out.data()[i]) != bits_of(ref)) {
      ++bad;
      if (first_bad < 0) {
        first_bad = i;
        std::cout << "SWIGLU_DET_NEON first mismatch i=" << i << " g=" << g[i]
                  << " u=" << u[i] << std::hexfloat << " neon=" << out[i]
                  << " scalar=" << ref << std::defaultfloat << std::endl;
      }
    }
  }
  std::cout << "SWIGLU_DET_NEON_FIELD bad=" << bad << " of " << n << std::endl;
  EXPECT_EQ(bad, 0)
    << "the NEON path does not match the scalar specification -- the usual "
       "cause is the compiler contracting a multiply and an add into an fmla";
#endif
}

class HvxM1Ops : public HtpSession {};

namespace {

/**
 * @brief Inputs for the M=1 small-op tests: the host check's spread
 *        (test/htp/host/m1_ops_host_check.c) -- random rows plus a zero, an
 *        all-subnormal (+-1e-39: HVX keeps subnormals, rule 24) and a
 *        large (|x| ~ 1e4) row, so the device is judged on what the host
 *        check already passed.
 */
void m1_fill_row(float *x, int n, int kind, std::mt19937 &rng) {
  std::uniform_real_distribution<float> small(-3.0f, 3.0f);
  std::uniform_real_distribution<float> large(-1e4f, 1e4f);
  for (int i = 0; i < n; ++i) {
    switch (kind) {
    case 1:
      x[i] = 0.0f;
      break;
    case 2:
      x[i] = (i & 1) ? -1e-39f : 1e-39f;
      break;
    case 3:
      x[i] = large(rng);
      break;
    default:
      x[i] = small(rng);
    }
  }
}

/** @brief cos[32] | sin[32] for one position at LFM2.5's rope_theta. */
void m1_rope_cs(float *cs, uint32_t pos) {
  for (int i = 0; i < 32; ++i) {
    const double inv_freq = std::pow(5e6, -(2.0 * i) / 64.0);
    const double ang = static_cast<double>(pos) * inv_freq;
    cs[i] = static_cast<float>(std::cos(ang));
    cs[32 + i] = static_cast<float>(std::sin(ang));
  }
}

int m1_count_bad(const std::vector<float> &dsp, const std::vector<float> &ref,
                 const char *what) {
  int bad = 0;
  for (size_t i = 0; i < dsp.size(); ++i) {
    if (bits_of(dsp[i]) != bits_of(ref[i])) {
      if (bad == 0) {
        std::cout << "M1_OPS " << what << " first mismatch i=" << i
                  << std::hexfloat << " dsp=" << dsp[i] << " ref=" << ref[i]
                  << std::defaultfloat << std::endl;
      }
      ++bad;
    }
  }
  return bad;
}

/** @brief One RMSNorm shape on the device against m1_ops_det.h; returns the
 *         output and the row-scale bad counts through the arguments. */
void m1_rmsnorm_case(remote_handle64 handle, uint32_t n, uint32_t chunk,
                     const std::vector<float> &x, int *bad_y, int *bad_rs) {
  const uint32_t nchunk = n / chunk;
  std::mt19937 rng(0x82000002u);
  std::uniform_real_distribution<float> gd(0.5f, 1.5f);
  std::vector<float> gamma(chunk), y(n, 0.0f), rs(nchunk, 0.0f);
  for (auto &g : gamma) {
    g = gd(rng);
  }
  const int err = nntr_hvx_rmsnorm_det_f32(
    handle, chunk, 1e-5f, x.data(), static_cast<int>(n), gamma.data(),
    static_cast<int>(chunk), y.data(), static_cast<int>(n), rs.data(),
    static_cast<int>(nchunk));
  ASSERT_EQ(err, AEE_SUCCESS) << "rmsnorm_det_f32 failed: " << hex(err)
                              << " (0x8000040e = stale skel, rule 3)";
  std::vector<float> y_ref(n), rs_ref(nchunk);
  m1_rmsnorm_det(x.data(), gamma.data(), y_ref.data(), n, chunk, 1e-5f,
                 rs_ref.data());
  *bad_rs = m1_count_bad(rs, rs_ref, "rmsnorm row_scale");
  *bad_y = m1_count_bad(y, y_ref, "rmsnorm y");
}

} // namespace

/**
 * @brief A bad shape is AEE_EINVALIDFORMAT from the entry's own check.
 *        AEE_EBADPARM here means the skel predates these three methods
 *        (rule 3) -- rebuild it before reading the other cases.
 */
TEST_F(HvxM1Ops, RejectsBadShapes) {
  std::vector<float> x(2048, 1.0f), gamma(48, 1.0f), y(2048), rs(1);
  // chunk 48: a multiple of 16 but neither of 32 nor a power of two.
  int err =
    nntr_hvx_rmsnorm_det_f32(handle_, 48u, 1e-5f, x.data(), 2048, gamma.data(),
                             48, y.data(), 2048, rs.data(), 1);
  EXPECT_EQ(err, AEE_EINVALIDFORMAT + kDspOffset)
    << "rmsnorm chunk 48: got " << hex(err);

  std::vector<float> cs(32, 0.0f), qk(128, 1.0f), yq(128);
  err = nntr_hvx_rope64_det_f32(handle_, 1u, cs.data(), 32, qk.data(), 128,
                                yq.data(), 128);
  EXPECT_EQ(err, AEE_EINVALIDFORMAT + kDspOffset)
    << "rope cs of 32: got " << hex(err);

  const int C = 64;
  std::vector<float> abc(3 * C, 1.0f), w(3 * C, 1.0f), st(2 * C, 0.0f), out(C),
    so(2 * C);
  // state_out must be three rows (the kernel's scratch row), not two.
  err = nntr_hvx_conv_gate_m1_f32(handle_, abc.data(), 3 * C, w.data(), 3 * C,
                                  st.data(), 2 * C, out.data(), C, so.data(),
                                  2 * C);
  EXPECT_EQ(err, AEE_EINVALIDFORMAT + kDspOffset)
    << "conv state_out of 2C: got " << hex(err);
}

/**
 * @brief The hidden RMSNorm (n = chunk = 2048) on the four row kinds, bit
 *        for bit against the scalar spec. row_scale is compared first: a
 *        bad count there puts the divergence in the reduction or the
 *        rsqrt, a clean row_scale with bad y puts it in the scaling.
 *
 * The last row (|x| ~ 3e38, sum of squares overflows to inf) is printed
 * and not gated: rule 24's unexplained overflow case, where the host
 * emulation has no device confirmation of the inf/NaN encodings.
 */
TEST_F(HvxM1Ops, RmsnormMatchesDetBitExact) {
  const uint32_t n = 2048;
  std::mt19937 rng(0x82000001u);
  std::vector<float> x(n);
  for (int kind = 0; kind < 4; ++kind) {
    m1_fill_row(x.data(), static_cast<int>(n), kind, rng);
    int bad_y = -1, bad_rs = -1;
    ASSERT_NO_FATAL_FAILURE(m1_rmsnorm_case(handle_, n, n, x, &bad_y, &bad_rs));
    std::cout << "M1_OPS_FIELD rmsnorm kind=" << kind << " bad_y=" << bad_y
              << " bad_row_scale=" << bad_rs << " of " << n << std::endl;
    EXPECT_EQ(bad_rs, 0) << "kind " << kind
                         << ": the reduction or rsqrt differs from the spec";
    EXPECT_EQ(bad_y, 0) << "kind " << kind << ": y differs from the spec";
  }
  std::uniform_real_distribution<float> huge(-3e38f, 3e38f);
  for (auto &v : x) {
    v = huge(rng);
  }
  int bad_y = -1, bad_rs = -1;
  ASSERT_NO_FATAL_FAILURE(m1_rmsnorm_case(handle_, n, n, x, &bad_y, &bad_rs));
  std::cout << "M1_OPS_FIELD rmsnorm overflow_row (not gated) bad_y=" << bad_y
            << " bad_row_scale=" << bad_rs << " of " << n << std::endl;
}

/** @brief The per-head q/k norm: 32 x 64 and 8 x 64 with one shared
 *         64-float gamma, heads 0..2 the fixed kinds. */
TEST_F(HvxM1Ops, QkNormMatchesDetBitExact) {
  std::mt19937 rng(0x82000003u);
  for (uint32_t heads : {32u, 8u}) {
    const uint32_t n = heads * 64u;
    std::vector<float> x(n);
    for (uint32_t h = 0; h < heads; ++h) {
      m1_fill_row(x.data() + h * 64u, 64,
                  (h < 3u) ? static_cast<int>(h) + 1 : 0, rng);
    }
    int bad_y = -1, bad_rs = -1;
    ASSERT_NO_FATAL_FAILURE(
      m1_rmsnorm_case(handle_, n, 64u, x, &bad_y, &bad_rs));
    std::cout << "M1_OPS_FIELD qk_norm heads=" << heads << " bad_y=" << bad_y
              << " bad_row_scale=" << bad_rs << " of " << n << std::endl;
    EXPECT_EQ(bad_rs, 0) << heads << " heads: row scale differs";
    EXPECT_EQ(bad_y, 0) << heads << " heads: y differs";
  }
}

/** @brief RoPE on 32 q + 8 k heads at positions 0, 1, 511, 1023, 4095;
 *         position 0 is also the identity bit for bit. */
TEST_F(HvxM1Ops, Rope64MatchesDetBitExact) {
  const uint32_t n_q = 32, n_k = 8, n = (n_q + n_k) * 64u;
  std::mt19937 rng(0x82000004u);
  std::vector<float> qk(n), y(n), y_ref(n), cs(64);
  for (uint32_t h = 0; h < n_q + n_k; ++h) {
    m1_fill_row(qk.data() + h * 64u, 64, (h < 3u) ? static_cast<int>(h) + 1 : 0,
                rng);
  }
  for (uint32_t pos : {0u, 1u, 511u, 1023u, 4095u}) {
    m1_rope_cs(cs.data(), pos);
    std::fill(y.begin(), y.end(), 0.0f);
    const int err = nntr_hvx_rope64_det_f32(handle_, n_q, cs.data(), 64,
                                            qk.data(), static_cast<int>(n),
                                            y.data(), static_cast<int>(n));
    ASSERT_EQ(err, AEE_SUCCESS) << "rope64_det_f32 failed: " << hex(err);
    y_ref = qk;
    for (uint32_t h = 0; h < n_q + n_k; ++h) {
      m1_rope64_det(y_ref.data() + h * 64u, cs.data());
    }
    const int bad = m1_count_bad(y, y_ref, "rope64");
    std::cout << "M1_OPS_FIELD rope64 pos=" << pos << " bad=" << bad << " of "
              << n << std::endl;
    EXPECT_EQ(bad, 0) << "pos " << pos << ": differs from the spec";
    if (pos == 0u) {
      EXPECT_EQ(m1_count_bad(y, qk, "rope64 identity"), 0)
        << "position 0 is not the identity";
    }
  }
}

/**
 * @brief The conv1d + gate as a chain of 8 tokens from a zero state, each
 *        call's out and new state against the spec's chain; the chain
 *        equals one prefill-shape hvx_conv_gate_f32 call by construction
 *        (the same kernel at m_count = 1; m1_ops_host_check proves it).
 */
TEST_F(HvxM1Ops, ConvGateM1MatchesDetBitExact) {
  const int C = 2048;
  std::mt19937 rng(0x82000005u);
  std::uniform_real_distribution<float> wd(-1.0f, 1.0f);
  std::vector<float> w(3 * C), abc(3 * C), out(C), state_in(2 * C, 0.0f),
    state_out(3 * C), out_ref(C), state_ref(2 * C, 0.0f);
  for (auto &v : w) {
    v = wd(rng);
  }
  int bad_out = 0, bad_state = 0;
  for (int t = 0; t < 8; ++t) {
    m1_fill_row(abc.data(), 3 * C, (t >= 1 && t <= 3) ? t : 0, rng);
    std::fill(out.begin(), out.end(), 0.0f);
    const int err = nntr_hvx_conv_gate_m1_f32(
      handle_, abc.data(), 3 * C, w.data(), 3 * C, state_in.data(), 2 * C,
      out.data(), C, state_out.data(), 3 * C);
    ASSERT_EQ(err, AEE_SUCCESS) << "conv_gate_m1_f32 failed: " << hex(err);
    m1_conv_gate_det(abc.data(), state_ref.data(), w.data(), out_ref.data(),
                     static_cast<uint32_t>(C));
    bad_out += m1_count_bad(out, out_ref, "conv out");
    std::copy(state_out.begin(), state_out.begin() + 2 * C, state_in.begin());
    bad_state += m1_count_bad(state_in, state_ref, "conv state");
  }
  std::cout << "M1_OPS_FIELD conv_gate_m1 C=" << C
            << " chain=8 bad_out=" << bad_out << " bad_state=" << bad_state
            << std::endl;
  EXPECT_EQ(bad_out, 0) << "conv1d + gate differs from the spec";
  EXPECT_EQ(bad_state, 0) << "the conv state shift differs from the spec";
}

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
