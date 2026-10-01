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
#include <chrono>
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

#include "../htp/host/q4_gemv_cases.h"
#include "m1_ops_det.h"
#include "nntr_hvx.h"
#include "q4_gemv_native_det.h"
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
inline float geglu_det_ref(float g, float u) { return geglu_det_one(g, u); }
inline float dmul(float a, float b) { return swiglu_det_mul(a, b); }
/** t = g (C0 + C1 g^2), as both geglu_det_one and the skel compute it */
inline float geglu_t_ref(float g) {
  return dmul(g, dadd(GEGLU_DET_C0, dmul(GEGLU_DET_C1, dmul(g, g))));
}
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
  int err = nntr_hvx_swiglu_det_f32(handle_, 0u, g.data(), n, u.data(), n,
                                    o.data(), n, e.data(), n, r.data(), n);
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

  int err = nntr_hvx_swiglu_det_f32(handle_, 0u, g.data(), n, u.data(), n,
                                    o.data(), n, e.data(), n, r.data(), n);
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
 * @brief The GeGLU twin of the gate above (doc 55: Gemma-4's experts):
 *        hvx_geglu_det_sf against geglu_det_one, bit for bit, with the
 *        exp and reciprocal of t = g (C0 + C1 g^2) reported the same way.
 */
TEST_F(HvxSwigluDet, GegluMatchesScalarBitExact) {
  const int n = 8192;
  std::vector<float> g(n), u(n), o(n, 0.0f), e(n, 0.0f), r(n, 0.0f);
  std::mt19937 rng(0xB2B2B2B2u);
  std::uniform_real_distribution<float> small(-8.0f, 8.0f);
  for (int i = 0; i < n; ++i) {
    if (i < 64) {
      g[i] = -40.0f + 1.25f * static_cast<float>(i); // t through both clamps
    } else {
      g[i] = small(rng);
    }
    u[i] = small(rng);
  }

  int err = nntr_hvx_swiglu_det_f32(handle_, 1u, g.data(), n, u.data(), n,
                                    o.data(), n, e.data(), n, r.data(), n);
  ASSERT_EQ(err, AEE_SUCCESS) << "swiglu_det_f32(gelu) failed: " << hex(err);

  int bad_exp = 0, bad_recip = 0, bad_out = 0, first_bad = -1;
  for (int i = 0; i < n; ++i) {
    const float t = geglu_t_ref(g[i]);
    const float ref_e = exp_det_ref(dsub(0.0f, t));
    const float ref_r = recip_det_ref(dadd(1.0f, ref_e));
    const float ref_o = geglu_det_ref(g[i], u[i]);
    const bool de = bits_of(e[i]) != bits_of(ref_e);
    const bool dr = bits_of(r[i]) != bits_of(ref_r);
    const bool dobad = bits_of(o[i]) != bits_of(ref_o);
    bad_exp += de ? 1 : 0;
    bad_recip += dr ? 1 : 0;
    bad_out += dobad ? 1 : 0;
    if (first_bad < 0 && (de || dr || dobad)) {
      first_bad = i;
      std::cout << "GEGLU_DET first mismatch i=" << i << " g=" << g[i]
                << " u=" << u[i] << "\n  exp   dsp=" << std::hexfloat << e[i]
                << " ref=" << ref_e << "\n  recip dsp=" << r[i]
                << " ref=" << ref_r << "\n  out   dsp=" << o[i]
                << " ref=" << ref_o << std::defaultfloat << std::endl;
    }
  }
  std::cout << "GEGLU_DET_FIELD bad_exp=" << bad_exp
            << " bad_recip=" << bad_recip << " bad_out=" << bad_out << " of "
            << n << std::endl;
  EXPECT_EQ(bad_exp, 0);
  EXPECT_EQ(bad_recip, 0);
  EXPECT_EQ(bad_out, 0);
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
 *        bad count there puts the divergence in the scalar row scale
 *        (sffma chains, integer sqrt / reciprocal; #164), a clean
 *        row_scale with bad y puts it in the scaling.
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
  // [#164] G0's magnitude sweep: one row per 2^e, e = -20 .. 20, elements
  // +-[0.5, 1) 2^e -- the spec is the CPU's order, so any normal-range
  // magnitude the model can produce must hold, not only the kinds above.
  std::uniform_real_distribution<float> mant(0.5f, 1.0f);
  int sweep_y = 0, sweep_rs = 0;
  for (int e = -20; e <= 20; ++e) {
    for (auto &v : x) {
      const float m = mant(rng);
      v = std::ldexp((rng() & 1u) ? -m : m, e);
    }
    int bad_y = -1, bad_rs = -1;
    ASSERT_NO_FATAL_FAILURE(m1_rmsnorm_case(handle_, n, n, x, &bad_y, &bad_rs));
    sweep_y += bad_y;
    sweep_rs += bad_rs;
  }
  std::cout << "M1_OPS_FIELD rmsnorm sweep 2^-20..2^20 rows=41 bad_y="
            << sweep_y << " bad_row_scale=" << sweep_rs << std::endl;
  EXPECT_EQ(sweep_rs, 0) << "sweep: the row scale differs from the spec";
  EXPECT_EQ(sweep_y, 0) << "sweep: y differs from the spec";

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
  // [#164] the magnitude sweep: head h of 40 at 2^e, e = h % 41 - 20,
  // twice with the second call shifted by 20 (all 41 magnitudes covered)
  std::uniform_real_distribution<float> mant(0.5f, 1.0f);
  int sweep_y = 0, sweep_rs = 0;
  for (uint32_t shift : {0u, 20u}) {
    const uint32_t n = 40u * 64u;
    std::vector<float> x(n);
    for (uint32_t i = 0; i < n; ++i) {
      const int e = static_cast<int>((i / 64u + shift) % 41u) - 20;
      const float m = mant(rng);
      x[i] = std::ldexp((rng() & 1u) ? -m : m, e);
    }
    int bad_y = -1, bad_rs = -1;
    ASSERT_NO_FATAL_FAILURE(
      m1_rmsnorm_case(handle_, n, 64u, x, &bad_y, &bad_rs));
    sweep_y += bad_y;
    sweep_rs += bad_rs;
  }
  std::cout << "M1_OPS_FIELD qk_norm sweep 2^-20..2^20 heads=80 bad_y="
            << sweep_y << " bad_row_scale=" << sweep_rs << std::endl;
  EXPECT_EQ(sweep_rs, 0) << "sweep: row scale differs";
  EXPECT_EQ(sweep_y, 0) << "sweep: y differs";
}

/** @brief RoPE (fp16 since #152) on 32 q + 8 k heads at positions 0, 1,
 *         511, 1023, 4095; position 0 is also the fp16-rounded input, by
 *         value (a -0 input with a negative b comes out +0, as the CPU's
 *         fsub does). */
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
      int bad_id = 0;
      for (uint32_t i = 0; i < n; ++i) {
        bad_id += y[i] != attn_m1_det_rne16(qk[i]);
      }
      EXPECT_EQ(bad_id, 0) << "position 0 is not the fp16 identity";
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

/**
 * @brief [#132] The MoE router at K 2048 / E 32 / top 4 (LFM2.5) and K 128
 *        and 64 / E 4 / top 2 (the fixtures), on random, exact-tie and
 *        1-ulp near-tie rows (m1_ops_host_check's construction): logits,
 *        selection order and routing weights bit for bit against
 *        m1_router_cpu_det (the Android CPU's order since #132 PR 2). A bad
 *        count in logits alone puts the divergence in the sffma chains;
 *        clean logits with bad weights put it in the expf port / integer
 *        divide on the DSP's scalar unit; a bad selection with clean logits
 *        is the tie rule. 0x8000040e is a skel that predates the method
 *        (rule 3).
 */
TEST_F(HvxM1Ops, RouterTopkMatchesDetBitExact) {
  struct Shape {
    uint32_t K, E, top_k;
  };
  std::mt19937 rng(0x13200001u);
  std::uniform_real_distribution<float> xd(-2.0f, 2.0f), wd(-0.05f, 0.05f),
    bd(-0.01f, 0.01f);
  for (const Shape sh :
       {Shape{2048u, 32u, 4u}, Shape{128u, 4u, 2u}, Shape{64u, 4u, 2u}}) {
    const uint32_t K = sh.K, E = sh.E, top_k = sh.top_k;
    int bad_logits = 0, bad_sel = 0, bad_weight = 0;
    for (int kind = 0; kind < 3; ++kind) {
      for (int rep = 0; rep < 8; ++rep) {
        std::vector<float> x(K), w(static_cast<size_t>(K) * E), bias(E);
        for (auto &v : x)
          v = xd(rng);
        for (auto &v : w)
          v = wd(rng);
        for (auto &v : bias)
          v = bd(rng);
        if (kind != 0) {
          // e1 < e2 share a column and meet at the last selected place
          const uint32_t e1 = rng() % (E - 1u);
          const uint32_t e2 = e1 + 1u + rng() % (E - 1u - e1);
          for (uint32_t k = 0; k < K; ++k)
            w[static_cast<size_t>(k) * E + e2] =
              w[static_cast<size_t>(k) * E + e1];
          for (uint32_t e = 0, n = 0; e < E && n + 1u < top_k; ++e) {
            if (e != e1 && e != e2) {
              bias[e] = 2.0f;
              ++n;
            }
          }
          bias[e1] = 1.0f;
          bias[e2] = kind == 1 ? 1.0f : std::nextafter(1.0f, 2.0f);
        }
        std::vector<float> logits(E, 0.0f), weight(top_k, 0.0f);
        std::vector<uint32_t> sel(top_k, 0u);
        const int err = nntr_hvx_router_topk_det_f32(
          handle_, top_k, x.data(), static_cast<int>(K), w.data(),
          static_cast<int>(w.size()), bias.data(), static_cast<int>(E),
          logits.data(), static_cast<int>(E), sel.data(),
          static_cast<int>(top_k), weight.data(), static_cast<int>(top_k));
        ASSERT_EQ(err, AEE_SUCCESS)
          << "router_topk_det_f32 failed: " << hex(err)
          << " (0x8000040e = stale skel, rule 3)";
        std::vector<float> logits_ref(E), weight_ref(top_k);
        std::vector<uint32_t> sel_ref(top_k);
        m1_router_cpu_det(x.data(), w.data(), bias.data(), K, E, top_k,
                          logits_ref.data(), sel_ref.data(), weight_ref.data());
        bad_logits += m1_count_bad(logits, logits_ref, "router logits");
        bad_weight += m1_count_bad(weight, weight_ref, "router weight");
        bad_sel += sel == sel_ref ? 0 : 1;
      }
    }
    std::cout << "M1_OPS_FIELD router_topk K=" << K << " E=" << E
              << " top_k=" << top_k << " rows=24 bad_logits=" << bad_logits
              << " bad_sel=" << bad_sel << " bad_weight=" << bad_weight
              << std::endl;
    EXPECT_EQ(bad_logits, 0) << "K " << K << ": logits differ from the spec";
    EXPECT_EQ(bad_sel, 0) << "K " << K << ": the selection differs";
    EXPECT_EQ(bad_weight, 0) << "K " << K << ": routing weights differ";
  }
}

/* ---- [#132 PR 2] the CPU-exact Q4_0 FC and the CPU-order small ops ---- */

class HvxFcQ4 : public HtpSession {};

namespace {

/** @brief fc_q4m1_f32's variant word (nntr_hvx.idl): the feed bit only
 *  since the second sitting (the kernel is the intrinsics one). */
constexpr uint32_t FC_FEED_VTCM = 1u << 16;

struct FcVariant {
  const char *name;
  uint32_t word;
};

/** @brief The kernel(s) G1 / G3 run; the first sitting's hvx_native and
 *  sffma* returned 0 on silicon and are gone (HvxFcQ4.SfProbe says why). */
const FcVariant kFcVariants[] = {{"hvx_intrin", 0u}};
constexpr int kNV = sizeof(kFcVariants) / sizeof(kFcVariants[0]);

/** @brief A K x N canonical Q4_0 weight from q4_gemv_cases.h, registered
 *         on the DSP as Q4M1. */
struct FcWeight {
  uint32_t K = 0, N = 0, h = 0;
  std::vector<uint8_t> canon;
};

void fc_register(remote_handle64 handle, uint32_t K, uint32_t N, FcWeight *w) {
  w->K = K;
  w->N = N;
  w->canon.resize(static_cast<size_t>(N) * (K / 32u) * 18u);
  make_weights(w->canon.data(), K, N);
  std::vector<uint8_t> m1(q4m1_bytes(K, N));
  q4m1_from_q4_0(w->canon.data(), K, N, m1.data());
  const int err = nntr_hvx_q4m1_register(handle, K, N, m1.data(),
                                         static_cast<int>(m1.size()), &w->h);
  ASSERT_EQ(err, AEE_SUCCESS) << "q4m1_register failed: " << hex(err)
                              << " (0x8000040e = stale skel, rule 3)";
}

/** @brief The spec's output for x. */
std::vector<float> fc_spec(const FcWeight &w, const std::vector<float> &x) {
  std::vector<int8_t> q(w.K);
  std::vector<uint16_t> d(w.K / 32u);
  std::vector<float> y(w.N);
  q8_0_quant_cpu_det(x.data(), w.K, q.data(), d.data());
  q4_gemv_cpu_det(w.canon.data(), q.data(), d.data(), w.K, w.N, y.data());
  return y;
}

/** @brief One fc_q4m1_f32 call; stats (8 words) out. */
void fc_run(remote_handle64 handle, const FcWeight &w, uint32_t variant,
            uint32_t lanes, uint32_t reps, const std::vector<float> &x,
            std::vector<float> *y, std::vector<uint32_t> *stats) {
  y->assign(w.N, 0.0f);
  stats->assign(8, 0u);
  const int err = nntr_hvx_fc_q4m1_f32(
    handle, w.h, variant, lanes, reps, x.data(), static_cast<int>(w.K),
    y->data(), static_cast<int>(w.N), stats->data(), 8);
  ASSERT_EQ(err, AEE_SUCCESS) << "fc_q4m1_f32 failed: " << hex(err);
}

} // namespace

/**
 * @brief G1 (plan 132): the DSP's Q4_0 FC equals q4_gemv_cpu_det bit for
 *        bit -- the five FC shapes, 8 rows each from q4_gemv_cases.h
 *        (block magnitudes 2^-20 .. 2^20, zero / tiny / quantizer-edge
 *        blocks; negative, f16-subnormal, zero and power-of-two d_w, an
 *        exact cancellation), the intrinsics kernel with its own vector
 *        quantizer, both feeds, 6 lanes.
 */
TEST_F(HvxFcQ4, MatchesSpecBitExact) {
  const uint32_t shapes[5][2] = {{2048u, 6144u},
                                 {2048u, 2048u},
                                 {2048u, 512u},
                                 {7168u, 2048u},
                                 {2048u, 7168u}};
  int bad_total = 0;
  for (const auto &sh : shapes) {
    FcWeight w;
    ASSERT_NO_FATAL_FAILURE(fc_register(handle_, sh[0], sh[1], &w));
    std::vector<float> x(w.K), y;
    std::vector<uint32_t> st;
    for (const FcVariant &v : kFcVariants) {
      for (uint32_t feed : {0u, FC_FEED_VTCM}) {
        int bad = 0;
        for (int r = 0; r < 8; ++r) {
          make_row(x.data(), w.K, r % 4, r);
          const std::vector<float> ref = fc_spec(w, x);
          ASSERT_NO_FATAL_FAILURE(
            fc_run(handle_, w, v.word | feed, 6u, 1u, x, &y, &st));
          bad += m1_count_bad(y, ref, v.name);
        }
        std::cout << "FC_Q4_FIELD K=" << w.K << " N=" << w.N
                  << " variant=" << v.name
                  << " feed=" << (feed ? "vtcm" : "direct")
                  << " rows=8 bad=" << bad << std::endl;
        EXPECT_EQ(bad, 0) << v.name << " K=" << w.K << " N=" << w.N;
        bad_total += bad;
      }
    }
    EXPECT_EQ(nntr_hvx_q4m1_release(handle_, w.h), AEE_SUCCESS);
  }
  std::cout << "FC_Q4_FIELD total bad=" << bad_total << std::endl;
}

/**
 * @brief [#194 L1, htp_moe_ppl] The native pair (fc_q4m1_f32 variant bit
 *        18: hvx_q4m1_prep_vec + hvx_q4m1_gemv_groups_native) equals
 *        q4_gemv_native_det.h bit for bit on the five FC shapes, 8 rows of
 *        every kind each plus a row of blocks around the quantizer's 2^-60
 *        cut, the three feeds, 6 lanes -- the device canary beside
 *        MatchesSpecBitExact (a broken E0 is a broken build; a failure
 *        here is the hf widening product or the qf32 -> hf narrowing not
 *        being what hvx_emu assumes). Prints the GEMV us of one call.
 */
TEST_F(HvxFcQ4, NativeMatchesSpec) {
  constexpr uint32_t FC_NATIVE = 1u << 18, FC_FEED_L2 = 1u << 17;
  const uint32_t shapes[5][2] = {{2048u, 6144u},
                                 {2048u, 2048u},
                                 {2048u, 512u},
                                 {7168u, 2048u},
                                 {2048u, 7168u}};
  int bad_total = 0;
  for (const auto &sh : shapes) {
    FcWeight w;
    ASSERT_NO_FATAL_FAILURE(fc_register(handle_, sh[0], sh[1], &w));
    std::vector<float> x(w.K), y, ref(w.N);
    std::vector<int8_t> q(w.K);
    std::vector<uint16_t> d(w.K / 32u);
    std::vector<uint32_t> st;
    for (uint32_t feed : {0u, FC_FEED_VTCM, FC_FEED_L2}) {
      int bad = 0;
      for (int r = 0; r < 9; ++r) {
        if (r < 8) {
          make_row(x.data(), w.K, r % 4, r);
        } else {
          for (uint32_t i = 0; i < w.K; ++i)
            x[i] = std::ldexp(frand(-1.0f, 1.0f),
                              -60 + static_cast<int>((i / 32u) % 5u) - 2);
        }
        q8_0_quant_native_det(x.data(), w.K, q.data(), d.data());
        q4_gemv_native_det(w.canon.data(), q.data(), d.data(), w.K, w.N,
                           ref.data());
        ASSERT_NO_FATAL_FAILURE(
          fc_run(handle_, w, FC_NATIVE | feed, 6u, 1u, x, &y, &st));
        bad += m1_count_bad(y, ref, "native");
      }
      std::cout << "FC_Q4_NATIVE K=" << w.K << " N=" << w.N << " feed="
                << (feed == 0u             ? "direct"
                    : feed == FC_FEED_VTCM ? "vtcm"
                                           : "l2")
                << " rows=9 bad=" << bad << " gemv_us=" << st[0]
                << " quant_us=" << st[1] << std::endl;
      EXPECT_EQ(bad, 0) << "native K=" << w.K << " N=" << w.N;
      bad_total += bad;
    }
    EXPECT_EQ(nntr_hvx_q4m1_release(handle_, w.h), AEE_SUCCESS);
  }
  std::cout << "FC_Q4_NATIVE total bad=" << bad_total << std::endl;
}

/** @brief The FC's vector quantizer (q8_quant_f32 runs hvx_q4m1_prep) on
 *         400 + 100 rows of every kind plus rows around its 2^-100 scalar
 *         fallback, and the scalar SwiGLU and argmax, against their specs
 *         (q8_quant_f32, swiglu_cpu_f32, argmax_f32). */
TEST_F(HvxFcQ4, SmallOpsMatchSpec) {
  int bad_q = 0, bad_s = 0, bad_a = 0;
  for (uint32_t K : {2048u, 7168u}) {
    std::vector<float> x(K);
    std::vector<int8_t> q(K), q_ref(K);
    std::vector<uint16_t> d(K / 32u), d_ref(K / 32u);
    const int rows = K == 2048u ? 400 : 100;
    for (int r = 0; r < rows; ++r) {
      make_row(x.data(), K, r % 4, r);
      if (r % 50 == 49)
        for (uint32_t i = 0; i < K; ++i)
          x[i] = std::ldexp(frand(-1.0f, 1.0f),
                            -100 + static_cast<int>((i / 32u) % 5u) - 2);
      ASSERT_EQ(nntr_hvx_q8_quant_f32(handle_, x.data(), static_cast<int>(K),
                                      q.data(), static_cast<int>(K), d.data(),
                                      static_cast<int>(K / 32u)),
                AEE_SUCCESS);
      q8_0_quant_cpu_det(x.data(), K, q_ref.data(), d_ref.data());
      bad_q += q != q_ref || d != d_ref;
    }
  }
  std::mt19937 rng(0x13205u);
  const uint32_t N = 7168u;
  std::vector<float> y(N), z(N), o(N), o_ref(N);
  for (int r = 0; r < 8; ++r) {
    const float span = r < 3 ? 8.0f : r < 6 ? 100.0f : 1e4f;
    std::uniform_real_distribution<float> yd(-span, span), zd(-4.0f, 4.0f);
    for (uint32_t i = 0; i < N; ++i) {
      y[i] = yd(rng);
      z[i] = zd(rng);
      if (r == 7)
        y[i] = -cpu_det_float(
          cpu_det_bits(static_cast<float>(
            (static_cast<double>(static_cast<int>(i / 56u) - 127) - 0.5) /
            1.4426950408889634)) +
          (i % 56u) - 28u);
    }
    ASSERT_EQ(nntr_hvx_swiglu_cpu_f32(handle_, y.data(), static_cast<int>(N),
                                      z.data(), static_cast<int>(N), o.data(),
                                      static_cast<int>(N)),
              AEE_SUCCESS);
    m1_swiglu_cpu_det(y.data(), z.data(), o_ref.data(), N);
    bad_s += m1_count_bad(o, o_ref, "swiglu_cpu");
  }
  std::vector<float> lg(128000);
  std::uniform_real_distribution<float> ld(-20.0f, 20.0f);
  for (int r = 0; r < 4; ++r) {
    for (auto &v : lg)
      v = ld(rng);
    lg[1000u + 7u * r] = 50.0f; /* a tie: the first must win */
    lg[90000u + r] = 50.0f;
    uint32_t idx = 0;
    ASSERT_EQ(nntr_hvx_argmax_f32(handle_, lg.data(),
                                  static_cast<int>(lg.size()), &idx),
              AEE_SUCCESS);
    bad_a +=
      idx != m1_argmax_first(lg.data(), 128000u) || idx != 1000u + 7u * r;
  }
  std::cout << "FC_Q4_FIELD small ops: q8_quant bad_rows=" << bad_q
            << " swiglu_cpu bad=" << bad_s << " argmax bad=" << bad_a
            << std::endl;
  EXPECT_EQ(bad_q, 0);
  EXPECT_EQ(bad_s, 0);
  EXPECT_EQ(bad_a, 0);
}

/**
 * @brief The scalar divide the vector quantizer relies on (sf_probe op 6,
 *        the Hexagon's sfrecipa / sffixup sequence) against the integer
 *        cpu_det_div_rn: amax / 127 for every mantissa at amax exponents
 *        -100, 0 and 20, and 1 / d for every mantissa at d exponents -107,
 *        0 and 12 (both quotients depend on the mantissa alone in that
 *        range), 6 x 2^23 divides in 2^20-element calls.
 */
TEST_F(HvxFcQ4, ScalarDivide) {
  const uint32_t chunk = 1u << 20;
  std::vector<float> a(chunk), b(chunk), o(chunk);
  uint64_t bad = 0, n = 0;
  const int ea[3] = {-100, 0, 20}, ed[3] = {-107, 0, 12};
  for (int kind = 0; kind < 2; ++kind)
    for (int t = 0; t < 3; ++t)
      for (uint32_t m0 = 0; m0 < (1u << 23); m0 += chunk) {
        for (uint32_t i = 0; i < chunk; ++i) {
          const uint32_t m = m0 + i;
          if (kind == 0) {
            a[i] =
              cpu_det_float((static_cast<uint32_t>(ea[t] + 127) << 23) | m);
            b[i] = 127.0f;
          } else {
            a[i] = 1.0f;
            b[i] =
              cpu_det_float((static_cast<uint32_t>(ed[t] + 127) << 23) | m);
          }
        }
        ASSERT_EQ(nntr_hvx_sf_probe(
                    handle_, 6u, a.data(), static_cast<int>(chunk), b.data(),
                    static_cast<int>(chunk), o.data(), static_cast<int>(chunk)),
                  AEE_SUCCESS)
          << "sf_probe (0x8000040e = stale skel, rule 3)";
        for (uint32_t i = 0; i < chunk; ++i, ++n) {
          const float r = cpu_det_div_rn(a[i], b[i]);
          if (cpu_det_bits(r) != cpu_det_bits(o[i])) {
            if (bad < 8)
              std::printf("SCALAR_DIV diff %08x / %08x dsp=%08x spec=%08x\n",
                          cpu_det_bits(a[i]), cpu_det_bits(b[i]),
                          cpu_det_bits(o[i]), cpu_det_bits(r));
            ++bad;
          }
        }
      }
  // [#132 Part B E5f] the SwiGLU's y / (exp_ps(-y) + 1) now takes the same
  // divide: 16 x 2^20 general pairs -- a any finite f32 of either sign
  // (subnormal included), b in [1, 2^127] (the denominator's range), and a
  // quarter with b any finite nonzero -- against cpu_det_div_rn
  std::mt19937 rng(0x5d1fu);
  for (int c = 0; c < 16; ++c) {
    for (uint32_t i = 0; i < chunk; ++i) {
      uint32_t ua = rng() & 0xFFFFFFFFu;
      if ((ua & 0x7F800000u) == 0x7F800000u)
        ua &= 0xBF7FFFFFu; /* finite */
      uint32_t ub = (c % 4 == 3)
                      ? (rng() & 0xFFFFFFFFu)
                      : ((127u + rng() % 128u) << 23) | (rng() & 0x7FFFFFu);
      if ((ub & 0x7F800000u) == 0x7F800000u || (ub & 0x7FFFFFFFu) == 0u)
        ub = 0x3F800000u;
      a[i] = cpu_det_float(ua);
      b[i] = cpu_det_float(ub);
    }
    ASSERT_EQ(nntr_hvx_sf_probe(handle_, 6u, a.data(), static_cast<int>(chunk),
                                b.data(), static_cast<int>(chunk), o.data(),
                                static_cast<int>(chunk)),
              AEE_SUCCESS);
    for (uint32_t i = 0; i < chunk; ++i, ++n) {
      const float r = cpu_det_div_rn(a[i], b[i]);
      if (cpu_det_bits(r) != cpu_det_bits(o[i])) {
        if (bad < 16)
          std::printf("SCALAR_DIV general diff %08x / %08x dsp=%08x "
                      "spec=%08x\n",
                      cpu_det_bits(a[i]), cpu_det_bits(b[i]),
                      cpu_det_bits(o[i]), cpu_det_bits(r));
        ++bad;
      }
    }
  }
  std::printf("SCALAR_DIV divides=%llu bad=%llu\n",
              static_cast<unsigned long long>(n),
              static_cast<unsigned long long>(bad));
  EXPECT_EQ(bad, 0u);
}

/**
 * @brief Why the first sitting's hvx_native and sffma variants read 0 on
 *        silicon (the ISS matched): each f32 operation on the DSP,
 *        sf_probe's IEEE-form asm vadd / vsub / vmpy (.sf = .sf op .sf)
 *        against the Q6_Vsf_* intrinsics (qf32 op + conversion) and the
 *        scalar sffma, (1) on 256 random normal pairs and (2) on the
 *        operands of the kernel's first failing step -- column group 0,
 *        block 0 of the 2048 x 32 G1 weight on row kind 0: the split
 *        product P1 = F1 * S1, P2 = F2 * S2 and uh = P1 + P2 (the
 *        Fast2Sum head), computed here by the host's IEEE ops. Prints per
 *        op the lanes that differ from the host and one lane in hex.
 */
TEST_F(HvxFcQ4, SfProbe) {
  auto probe = [&](uint32_t op, const std::vector<float> &a,
                   const std::vector<float> &b) {
    std::vector<float> o(a.size(), -1.0f);
    const int err = nntr_hvx_sf_probe(
      handle_, op, a.data(), static_cast<int>(a.size()), b.data(),
      static_cast<int>(b.size()), o.data(), static_cast<int>(o.size()));
    EXPECT_EQ(err, AEE_SUCCESS) << "sf_probe op " << op << ": " << hex(err);
    return o;
  };
  auto report = [&](const char *set, const char *name, uint32_t op,
                    const std::vector<float> &a, const std::vector<float> &b,
                    const std::vector<float> &ref) {
    const std::vector<float> o = probe(op, a, b);
    int bad = 0, first = -1;
    for (size_t i = 0; i < o.size(); ++i)
      if (cpu_det_bits(o[i]) != cpu_det_bits(ref[i])) {
        ++bad;
        first = first < 0 ? static_cast<int>(i) : first;
      }
    const size_t i = first < 0 ? 0u : static_cast<size_t>(first);
    std::printf("SF_PROBE %s op=%s bad=%d/%zu lane=%zu a=%08x b=%08x "
                "dsp=%08x host=%08x\n",
                set, name, bad, o.size(), i, cpu_det_bits(a[i]),
                cpu_det_bits(b[i]), cpu_det_bits(o[i]), cpu_det_bits(ref[i]));
    return bad;
  };
  auto run_set = [&](const char *set, const std::vector<float> &a,
                     const std::vector<float> &b, bool sffma) {
    std::vector<float> add(a.size()), sub(a.size()), mul(a.size()),
      fma1(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      volatile float x = a[i], y = b[i];
      volatile float s0 = x + y, s1 = x - y, s2 = x * y;
      add[i] = s0;
      sub[i] = s1;
      mul[i] = s2;
      fma1[i] = std::fma(a[i], b[i], 1.0f);
    }
    int intrin_bad = report(set, "intrin_vadd", 3u, a, b, add);
    intrin_bad += report(set, "intrin_vsub", 4u, a, b, sub);
    intrin_bad += report(set, "intrin_vmpy", 5u, a, b, mul);
    report(set, "asm_ieee_vadd", 0u, a, b, add);
    report(set, "asm_ieee_vsub", 1u, a, b, sub);
    report(set, "asm_ieee_vmpy", 2u, a, b, mul);
    if (sffma)
      intrin_bad += report(set, "scalar_sffma", 7u, a, b, fma1);
    return intrin_bad;
  };
  std::mt19937 rng(0x5f5f5f5fu);
  std::uniform_real_distribution<float> md(1.0f, 2.0f);
  std::vector<float> a(256), b(256);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = std::ldexp((rng() & 1u) ? -md(rng) : md(rng),
                      static_cast<int>(rng() % 60u) - 30);
    b[i] = std::ldexp((rng() & 1u) ? -md(rng) : md(rng),
                      static_cast<int>(rng() % 60u) - 30);
  }
  int intrin_bad = run_set("random", a, b, true);
  // the kernel's first step on the G1 data: group 0, block 0
  const uint32_t K = 2048u;
  std::vector<uint8_t> w(static_cast<size_t>(32u) * (K / 32u) * 18u);
  make_weights(w.data(), K, 32u);
  std::vector<float> x(K);
  make_row(x.data(), K, 0, 0);
  std::vector<int8_t> q(K);
  std::vector<uint16_t> d(K / 32u);
  q8_0_quant_cpu_det(x.data(), K, q.data(), d.data());
  const uint32_t ea = (d[0] >> 10) & 31u;
  const int32_t ma = static_cast<int32_t>((d[0] & 1023u) | (ea ? 1024u : 0u));
  std::vector<float> F1(32), S1(32), F2(32), S2(32), P1(32), P2(32);
  for (uint32_t l = 0; l < 32u; ++l) {
    const uint8_t *blk = w.data() + static_cast<size_t>(l) * (K / 32u) * 18u;
    const int32_t isum = q4_cpu_block_isum(blk + 2, q.data());
    const uint32_t h = q4_cpu_block_d(blk), E = (h >> 10) & 31u;
    int32_t mw = static_cast<int32_t>((h & 1023u) | (E ? 1024u : 0u));
    int32_t T = isum * mw;
    T = (h & 0x8000u) ? -T : T;
    const int32_t Th = T >> 13, Tl = T & 0x1fff;
    const uint32_t s2 = ((E ? E : 1u) + (ea ? ea : 1u) - 50u + 127u) << 23;
    F1[l] = static_cast<float>(Th * ma);
    F2[l] = static_cast<float>(Tl * ma);
    S2[l] = cpu_det_float(s2);
    S1[l] = cpu_det_float(s2 + (13u << 23));
    P1[l] = F1[l] * S1[l];
    P2[l] = F2[l] * S2[l];
  }
  intrin_bad += run_set("step_P1=F1*S1", F1, S1, false);
  intrin_bad += run_set("step_P2=F2*S2", F2, S2, false);
  intrin_bad += run_set("step_uh=P1+P2", P1, P2, false);
  std::printf("SF_PROBE intrinsics+sffma bad=%d (the kernel's ops; 0 "
              "expected)\n",
              intrin_bad);
  EXPECT_EQ(intrin_bad, 0);
}

/**
 * @brief G3 (plan 132): the exact FC's rate on silicon, reported, not
 *        gated (it feeds decision D). The per-token FC set of LFM2.5
 *        (453 M weights) and the tied lm_head (262 M, measured as one
 *        16384-row slice and scaled by 128000 / 16384), the kernel,
 *        weights read directly from the heap or fed to VTCM by per-lane
 *        DMA (src_bypass), 1 / 2 / 4 / 6 lanes; 3 timed calls after one
 *        warm-up. Per cell: us per call, cycles per (column, 32-block)
 *        step per lane, GB/s; per variant, feed and lane count the
 *        projected ms per token of the whole set (FC_RATE_PROJ). Every
 *        output is bit-compared to the spec as well.
 */
TEST_F(HvxFcQ4, Rate) {
  struct Shape {
    uint32_t K, N;
    double per_token;
  };
  const Shape shapes[6] = {
    {2048u, 6144u, 18.0}, {2048u, 2048u, 30.0},
    {2048u, 512u, 12.0},  {2048u, 7168u, 4.0},
    {7168u, 2048u, 2.0},  {2048u, 16384u, 128000.0 / 16384.0}};
  const uint32_t lane_set[4] = {1u, 2u, 4u, 6u};
  // [variant][feed][lanes] -> ms per token (FC and lm_head), quant ms
  double ms[kNV][2][4] = {}, ms_lm[kNV][2][4] = {}, ms_q[kNV][2][4] = {};
  int bad_total = 0;
  for (const Shape &sh : shapes) {
    FcWeight w;
    ASSERT_NO_FATAL_FAILURE(fc_register(handle_, sh.K, sh.N, &w));
    std::vector<float> x(w.K), y;
    std::vector<uint32_t> st;
    make_row(x.data(), w.K, 0, 1);
    const std::vector<float> ref = fc_spec(w, x);
    const double steps = static_cast<double>(w.N) * (w.K / 32u);
    const double gbytes = static_cast<double>(w.N) * w.K * 18.0 / 32.0;
    for (int vi = 0; vi < kNV; ++vi) {
      for (int fi = 0; fi < 2; ++fi) {
        for (int li = 0; li < 4; ++li) {
          const uint32_t word = kFcVariants[vi].word | (fi ? FC_FEED_VTCM : 0u);
          ASSERT_NO_FATAL_FAILURE(
            fc_run(handle_, w, word, lane_set[li], 1u, x, &y, &st));
          ASSERT_NO_FATAL_FAILURE(
            fc_run(handle_, w, word, lane_set[li], 3u, x, &y, &st));
          const int bad = m1_count_bad(y, ref, kFcVariants[vi].name);
          bad_total += bad;
          const double us = st[0] / 3.0, qus = st[1] / 3.0;
          const double pcyc =
            (static_cast<double>(st[3]) * 4294967296.0 + st[2]) / 3.0;
          const double cyc_step = pcyc * st[4] / steps;
          std::cout << std::fixed << std::setprecision(2) << "FC_RATE K=" << w.K
                    << " N=" << w.N << " variant=" << kFcVariants[vi].name
                    << " feed=" << (fi ? "vtcm" : "direct")
                    << " lanes=" << st[4] << " us_per_call=" << us
                    << " quant_us=" << qus << " pcyc_per_call=" << pcyc
                    << " cyc_per_colstep_lane=" << cyc_step
                    << " GBps=" << gbytes / (us * 1e3) << " bad=" << bad
                    << std::defaultfloat << std::endl;
          const double tok = sh.per_token * us / 1000.0;
          if (w.N == 16384u)
            ms_lm[vi][fi][li] += tok;
          else
            ms[vi][fi][li] += tok;
          ms_q[vi][fi][li] += sh.per_token * qus / 1000.0;
        }
      }
    }
    EXPECT_EQ(nntr_hvx_q4m1_release(handle_, w.h), AEE_SUCCESS);
  }
  for (int vi = 0; vi < kNV; ++vi)
    for (int fi = 0; fi < 2; ++fi)
      for (int li = 0; li < 4; ++li)
        std::cout << std::fixed << std::setprecision(3)
                  << "FC_RATE_PROJ variant=" << kFcVariants[vi].name
                  << " feed=" << (fi ? "vtcm" : "direct")
                  << " lanes=" << lane_set[li]
                  << " ms_per_token=" << ms[vi][fi][li] + ms_lm[vi][fi][li]
                  << " (fc=" << ms[vi][fi][li]
                  << " lm_head=" << ms_lm[vi][fi][li]
                  << ") quant_ms=" << ms_q[vi][fi][li]
                  << " cpu_reference_ms=7.4" << std::defaultfloat << std::endl;
  std::cout << "FC_RATE bad_total=" << bad_total << std::endl;
  EXPECT_EQ(bad_total, 0);
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
