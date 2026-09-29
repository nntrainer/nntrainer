// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_hvx_attn.cpp
 * @date   07 Aug 2026
 * @brief  Device gate for PHASE A (S = Q.Kt) of the fused attention path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The oracle is mha_htp_host_scores -- the same PHASE A mha_htp_host_forward
 * runs, not a second opinion written beside it. Both sides do identical
 * integer arithmetic (u8 activation x iX weight -> int32 -> the same dequant
 * formula), so agreement should be near exact; the tolerance asserted is
 * MHA_HTP_U8_TASKS.md Task 3's, and the observed value is reported so the
 * margin is visible rather than assumed.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "nntr_hvx.h"

#include "../htp/host/attn_m1_cases.h"
#include "../htp/nntr_attn_m1_probe.h"
#include "attn_m1_det.h"
#include "htp_rpc_bench.h"
#include "mha_htp_host_model.h"
#include "swiglu_det.h"

namespace {

constexpr int kDspOffset = 0x80000400;

/**
 * @brief Opens an unsigned-PD CDSP session for each test.
 *
 * Same shape as unittest_hvx_softmax.cpp's fixture: a failure here is a hard
 * FAIL rather than a skip, because proving the DSP comes up is part of what
 * this test measures.
 */
class HtpSession : public ::testing::Test {
protected:
  void SetUp() override {
    int err = htp_enable_unsigned_pd();
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);

    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &handle_);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "nntr_hvx_open failed: " << hex(err)
      << " -- is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?";

    std::cout << "ATTN_FIELD path=env field=rpc_poll_qos value="
              << htp_set_latency_qos(handle_) << std::endl;
  }

  void TearDown() override {
    if (handle_) {
      nntr_hvx_close(handle_);
    }
  }

  remote_handle64 handle_ = 0;
};

class HvxAttnScores : public HtpSession {};

/** @brief fp32 -> fp16 bits, matching what the KV cache holds. */
uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t man = x & 0x7FFFFFu;
  if (exp <= 0) {
    return (uint16_t)sign;
  }
  if (exp >= 31) {
    return (uint16_t)(sign | 0x7C00u);
  }
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

struct AttnCfg {
  uint32_t kv_len, n_query, gqa, nch, head_dim, T;
  hexkl_w_width w_k, w_v;
  uint32_t M_band = 64u; /**< must be <= T (property 5) */
};

const char *wname(hexkl_w_width w) { return w == HEXKL_W_I4 ? "I4" : "I8"; }

/**
 * @brief Registers, appends the whole cache, and returns the device S band for
 *        one head, alongside the host model's.
 */
void run_case(remote_handle64 handle, const AttnCfg &c, uint32_t head,
              std::vector<float> *dev, std::vector<float> *ref) {
  const uint32_t M = c.n_query * c.gqa;
  const uint32_t n_blocks = (c.kv_len + c.T - 1u) / c.T;
  const uint32_t band = n_blocks * M * c.T;

  std::mt19937 rng(c.kv_len * 7919u + c.n_query * 131u + c.gqa * 17u +
                   c.head_dim + head);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);

  /** Realistic KV: a shared per-channel component plus per-position noise.
   * i.i.d. content makes the attention output a near-total cancellation and
   * tells you nothing (MHA_HTP_PLAN.md §9.2). */
  std::vector<float> base(c.nch * c.head_dim);
  for (float &x : base) {
    x = d(rng);
  }
  std::vector<uint16_t> k16((size_t)c.kv_len * c.nch * c.head_dim);
  std::vector<uint16_t> v16(k16.size());
  for (uint32_t r = 0; r < c.kv_len; ++r) {
    for (uint32_t h = 0; h < c.nch; ++h) {
      for (uint32_t dd = 0; dd < c.head_dim; ++dd) {
        const size_t i = ((size_t)r * c.nch + h) * c.head_dim + dd;
        k16[i] = f32_to_f16(base[h * c.head_dim + dd] + 0.5f * d(rng));
        v16[i] = f32_to_f16(base[h * c.head_dim + dd] + 0.5f * d(rng));
      }
    }
  }
  std::vector<float> q_band((size_t)M * c.head_dim);
  for (float &x : q_band) {
    x = d(rng);
  }

  uint32_t h_attn = 0;
  int err =
    nntr_hvx_attn_register(handle, c.nch, c.gqa, c.head_dim, c.kv_len, c.T,
                           c.M_band, (uint32_t)c.w_k, (uint32_t)c.w_v, &h_attn);
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_register: " << hex(err);

  err = nntr_hvx_attn_kv_append(handle, h_attn, 0u, c.kv_len, k16.data(),
                                (int)k16.size(), v16.data(), (int)v16.size());
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_kv_append: " << hex(err);

  dev->assign(band, 0.0f);
  err = nntr_hvx_attn_scores_debug(handle, h_attn, head, M, q_band.data(),
                                   (int)q_band.size(), dev->data(), (int)band);
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_scores_debug: " << hex(err);

  err = nntr_hvx_attn_release(handle, h_attn);
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_release: " << hex(err);

  ref->assign(band, 0.0f);
  mha_htp_host_scores(c.kv_len, c.nch, c.head_dim, c.T, M, head, c.w_k,
                      q_band.data(), k16.data(), ref->data());
}

/**
 * @brief max|a - b| normalized by max|b|, and @a den_out is that denominator.
 *
 * The denominator is returned, not swallowed, because a zero one makes this
 * metric report a perfect 0.00e+00 for a comparison of two all-zero buffers.
 * The caller asserts it is non-zero, so "matched exactly" cannot be a way of
 * saying "computed nothing".
 */
double max_rel(const std::vector<float> &a, const std::vector<float> &b,
               double *den_out) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    num = std::max(num, std::fabs((double)a[i] - (double)b[i]));
    den = std::max(den, std::fabs((double)b[i]));
  }
  *den_out = den;
  return (den > 0.0) ? num / den : num;
}

double tol_for(hexkl_w_width w_k) {
  /** Task 3's out tolerances, applied to the stage that feeds them. S is where
   * K's width shows up, so the K half of the pair is what selects. */
  return (w_k == HEXKL_W_I8) ? 5e-3 : 5e-2;
}

} // namespace

TEST_F(HvxAttnScores, MatchesHostModelOverTheShapeMatrix) {
  const hexkl_w_width widths[4][2] = {{HEXKL_W_I8, HEXKL_W_I8},
                                      {HEXKL_W_I4, HEXKL_W_I4},
                                      {HEXKL_W_I8, HEXKL_W_I4},
                                      {HEXKL_W_I4, HEXKL_W_I8}};
  double worst = 0.0;
  double weakest_ref = 0.0; /* smallest S dynamic range any case produced */
  std::string worst_where;
  size_t cases = 0;

  std::cout << "\nS band vs host model -- max|dev - host| / max|host|\n"
            << "w_k,w_v are the Kt and V CACHE widths, in that order. "
               "Activations are u8 throughout,\n"
            << "so I4 means a u8 x i4 matmul -- nothing here is u4.\n"
            << "shape (kv,nq,gqa,nch,hd,T)     w_k,w_v   max_rel_err\n";

  for (uint32_t kv : {32u, 33u, 256u, 257u, 1024u}) {
    for (uint32_t nq : {1u, 33u, 128u}) {
      if (nq > kv) {
        continue; /* kv_len is kv_from + n_query; more queries than positions
                     is not representable */
      }
      for (uint32_t gqa : {1u, 8u}) {
        for (uint32_t head_dim : {64u, 128u}) {
          for (uint32_t T : {64u, 256u}) {
            for (int w = 0; w < 4; ++w) {
              const AttnCfg c{kv,       nq, gqa,          1u,
                              head_dim, T,  widths[w][0], widths[w][1]};
              std::vector<float> dev, ref;
              run_case(handle_, c, 0u, &dev, &ref);
              if (::testing::Test::HasFatalFailure()) {
                return;
              }
              double den = 0.0;
              const double e = max_rel(dev, ref, &den);
              /** A case whose reference S is all zeros would report a perfect
               * match while proving nothing. */
              ASSERT_GT(den, 0.0)
                << "reference S is identically zero -- the comparison is "
                << "vacuous for kv=" << kv << " nq=" << nq << " T=" << T;
              if (cases == 0 || den < weakest_ref) {
                weakest_ref = den;
              }

              std::ostringstream shape;
              shape << kv << "," << nq << "," << gqa << ",1," << head_dim << ","
                    << T;
              if (e > worst) {
                worst = e;
                worst_where =
                  shape.str() + " " + wname(c.w_k) + "," + wname(c.w_v);
              }
              if (w == 0 || e > 1e-6) {
                std::cout << std::left << std::setw(30) << shape.str()
                          << std::setw(10)
                          << (std::string(wname(c.w_k)) + "," + wname(c.w_v))
                          << std::scientific << std::setprecision(2) << e
                          << "\n";
              }
              EXPECT_LE(e, tol_for(c.w_k))
                << shape.str() << " (" << wname(c.w_k) << "," << wname(c.w_v)
                << ")";
              ++cases;
            }
          }
        }
      }
    }
  }

  std::cout << "ATTN_FIELD path=scores field=cases value=" << cases
            << std::endl;
  std::cout << "ATTN_FIELD path=scores field=max_rel_err value=" << worst
            << std::endl;
  std::cout << "ATTN_FIELD path=scores field=worst_shape value=" << worst_where
            << std::endl;
  /** Publish the weakest denominator so a future 0.00e+00 can be read as
     "bit-identical" rather than "both sides were empty". */
  std::cout << "ATTN_FIELD path=scores field=min_ref_dynamic_range value="
            << weakest_ref << std::endl;
}

/**
 * @brief V's width must not reach S at all.
 *
 * S = Q.Kt never touches V, so the two runs must be BITWISE identical. A
 * difference here is a plumbing bug -- the wrong registry or the wrong ops
 * being handed to the score call -- not numerical noise, which is why this is
 * bit-exact rather than a tolerance.
 */
TEST_F(HvxAttnScores, VWidthDoesNotReachS) {
  for (uint32_t kv : {33u, 257u}) {
    for (hexkl_w_width w_k : {HEXKL_W_I8, HEXKL_W_I4}) {
      const AttnCfg a{kv, 33u, 8u, 1u, 128u, 64u, w_k, HEXKL_W_I8};
      const AttnCfg b{kv, 33u, 8u, 1u, 128u, 64u, w_k, HEXKL_W_I4};
      std::vector<float> da, ra, db, rb;

      run_case(handle_, a, 0u, &da, &ra);
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
      run_case(handle_, b, 0u, &db, &rb);
      if (::testing::Test::HasFatalFailure()) {
        return;
      }

      ASSERT_EQ(da.size(), db.size());
      for (size_t i = 0; i < da.size(); ++i) {
        ASSERT_EQ(std::memcmp(&da[i], &db[i], sizeof(float)), 0)
          << "kv=" << kv << " w_k=" << wname(w_k) << " i=" << i
          << " w_v=I8 gave " << da[i] << ", w_v=I4 gave " << db[i];
      }
    }
  }
  std::cout << "ATTN_FIELD path=scores field=v_width_invariant value=1"
            << std::endl;
}

/**
 * @brief A released and re-registered layer reproduces its own result.
 *
 * Catches a registry slot that is not really reset -- stale WH bytes or a
 * handle that outlives its release would show up as a difference on the second
 * pass, and only there.
 */
TEST_F(HvxAttnScores, LifecycleIsRepeatable) {
  for (hexkl_w_width w : {HEXKL_W_I8, HEXKL_W_I4}) {
    /** kv 257 with T 64 crosses several block boundaries and leaves a partial
     * tail block, which is where a stale registration would survive. */
    const AttnCfg c{257u, 33u, 8u, 1u, 128u, 64u, w, w};
    std::vector<float> first, ref1, second, ref2;

    run_case(handle_, c, 0u, &first, &ref1);
    if (::testing::Test::HasFatalFailure()) {
      return;
    }
    run_case(handle_, c, 0u, &second, &ref2);
    if (::testing::Test::HasFatalFailure()) {
      return;
    }

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i) {
      ASSERT_EQ(std::memcmp(&first[i], &second[i], sizeof(float)), 0)
        << "width=" << wname(w) << " i=" << i;
    }
  }
  std::cout << "ATTN_FIELD path=scores field=lifecycle_repeatable value=1"
            << std::endl;
}

TEST_F(HvxAttnScores, RejectsBadShapes) {
  uint32_t h = 0;
  /* T must be a multiple of 32: hexkl_mm_u8iX_plan enforces N % 32 == 0. */
  EXPECT_EQ(nntr_hvx_attn_register(handle_, 1u, 1u, 128u, 64u, 48u, 48u,
                                   (uint32_t)HEXKL_W_I8, (uint32_t)HEXKL_W_I8,
                                   &h),
            AEE_EBADPARM + kDspOffset)
    << "T not a multiple of 32 must be rejected";
  EXPECT_EQ(nntr_hvx_attn_register(handle_, 1u, 1u, 128u, 64u, 64u, 64u, 5u,
                                   (uint32_t)HEXKL_W_I8, &h),
            AEE_EBADPARM + kDspOffset)
    << "width other than 4 or 8 must be rejected";
  /** Property 5: M_band > T is the one that misbehaves silently until block
   * skipping lands, so it has to be rejected at registration. */
  EXPECT_EQ(nntr_hvx_attn_register(handle_, 1u, 1u, 128u, 256u, 64u, 128u,
                                   (uint32_t)HEXKL_W_I8, (uint32_t)HEXKL_W_I8,
                                   &h),
            AEE_EBADPARM + kDspOffset)
    << "M_band > T must be rejected";
}

/**
 * @brief The fused forward against the host model: PHASE A + B + C in ONE
 *        FastRPC round trip.
 *
 * This is the stage that replaces compute_kcaches + softmax_triangle +
 * compute_fp16vcache_transposed in mha_core's incremental path, so the oracle
 * is mha_htp_host_forward -- the same loop, same quantization points, same
 * mask arithmetic, scalar instead of HMX/HVX/DMA.
 */
TEST_F(HvxAttnScores, FusedForwardMatchesHostModel) {
  struct FwdCfg {
    uint32_t kv_len, n_query, gqa, nch, head_dim, T, M_band, window;
    int causal, sink;
  };
  /** Qwen3-0.6B's per-head shape (nch 8 / gqa 2 / head_dim 128) plus the
   * boundary cases the block arithmetic hides in: a partial tail block, a
   * window on either side of T, and decode as well as prefill. */
  const FwdCfg cfgs[] = {
    {256u, 1u, 2u, 8u, 128u, 256u, 64u, 0u, 1, 0},
    {257u, 33u, 2u, 8u, 128u, 256u, 64u, 0u, 1, 0},
    {512u, 128u, 2u, 8u, 128u, 256u, 64u, 0u, 1, 0},
    {257u, 33u, 8u, 1u, 64u, 64u, 64u, 63u, 1, 0},
    {257u, 33u, 8u, 1u, 64u, 64u, 64u, 64u, 1, 1},
    {257u, 33u, 8u, 1u, 64u, 64u, 64u, 65u, 0, 1},
    {1024u, 1u, 2u, 8u, 128u, 256u, 64u, 0u, 1, 1},
  };
  const hexkl_w_width widths[4][2] = {{HEXKL_W_I8, HEXKL_W_I8},
                                      {HEXKL_W_I4, HEXKL_W_I4},
                                      {HEXKL_W_I8, HEXKL_W_I4},
                                      {HEXKL_W_I4, HEXKL_W_I8}};
  double worst = 0.0;
  size_t cases = 0;

  std::cout << "\nfused forward vs host model -- max|dev - host| / max|host|\n"
            << "w_k,w_v are the Kt and V CACHE widths, in that order. "
               "Activations are u8 throughout,\n"
            << "so I4 means a u8 x i4 matmul -- nothing here is u4.\n"
            << "shape (kv,nq,gqa,nch,hd,T,Mb,win,c,sk)  w_k,w_v  max_rel_err\n";

  for (const FwdCfg &f : cfgs) {
    const uint32_t nHq = f.nch * f.gqa;
    const uint32_t kv_from = f.kv_len - f.n_query;

    std::mt19937 rng(f.kv_len * 7919u + f.n_query * 131u + f.gqa);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    std::vector<float> base(f.nch * f.head_dim);
    for (float &x : base) {
      x = d(rng);
    }
    std::vector<uint16_t> k16((size_t)f.kv_len * f.nch * f.head_dim);
    std::vector<uint16_t> v16(k16.size());
    for (uint32_t r = 0; r < f.kv_len; ++r) {
      for (uint32_t h = 0; h < f.nch; ++h) {
        for (uint32_t dd = 0; dd < f.head_dim; ++dd) {
          const size_t i = ((size_t)r * f.nch + h) * f.head_dim + dd;
          k16[i] = f32_to_f16(base[h * f.head_dim + dd] + 0.5f * d(rng));
          v16[i] = f32_to_f16(base[h * f.head_dim + dd] + 0.5f * d(rng));
        }
      }
    }
    std::vector<float> q((size_t)f.n_query * nHq * f.head_dim);
    for (float &x : q) {
      x = d(rng);
    }
    std::vector<float> sink(nHq);
    for (float &x : sink) {
      x = d(rng) * 0.3f;
    }
    const float scale = 1.0f / std::sqrt((float)f.head_dim);

    for (int w = 0; w < 4; ++w) {
      uint32_t h_attn = 0;
      int err = nntr_hvx_attn_register(
        handle_, f.nch, f.gqa, f.head_dim, f.kv_len, f.T, f.M_band,
        (uint32_t)widths[w][0], (uint32_t)widths[w][1], &h_attn);
      ASSERT_EQ(err, AEE_SUCCESS) << "attn_register: " << hex(err);
      err =
        nntr_hvx_attn_kv_append(handle_, h_attn, 0u, f.kv_len, k16.data(),
                                (int)k16.size(), v16.data(), (int)v16.size());
      ASSERT_EQ(err, AEE_SUCCESS) << "attn_kv_append: " << hex(err);

      std::vector<float> dev(q.size(), 0.0f);
      err = nntr_hvx_attn_forward(
        handle_, h_attn, kv_from, f.n_query, scale, (uint32_t)f.causal,
        f.window, f.sink ? sink.data() : nullptr, f.sink ? (int)nHq : 0,
        q.data(), (int)q.size(), dev.data(), (int)dev.size());
      ASSERT_EQ(err, AEE_SUCCESS) << "attn_forward: " << hex(err);
      ASSERT_EQ(nntr_hvx_attn_release(handle_, h_attn), AEE_SUCCESS);

      std::vector<float> ref(q.size(), 0.0f);
      mha_htp_host_forward(
        f.n_query, kv_from, f.nch, f.gqa, f.head_dim, f.T, f.M_band,
        f.causal != 0, f.window, f.sink ? sink.data() : nullptr, widths[w][0],
        widths[w][1], q.data(), k16.data(), v16.data(), ref.data());

      double den = 0.0;
      const double e = max_rel(dev, ref, &den);
      ASSERT_GT(den, 0.0) << "reference output is identically zero";
      worst = std::max(worst, e);

      std::ostringstream shape;
      shape << f.kv_len << "," << f.n_query << "," << f.gqa << "," << f.nch
            << "," << f.head_dim << "," << f.T << "," << f.M_band << ","
            << f.window << "," << f.causal << "," << f.sink;
      std::cout << std::left << std::setw(40) << shape.str() << std::setw(9)
                << (std::string(wname(widths[w][0])) + "," +
                    wname(widths[w][1]))
                << std::scientific << std::setprecision(2) << e << "\n";

      /** Task 3's fixed tolerances, on the same quantity they were written
       * for. */
      const double tol = (widths[w][0] == HEXKL_W_I8)
                           ? ((widths[w][1] == HEXKL_W_I8) ? 5e-3 : 2e-2)
                           : 5e-2;
      EXPECT_LE(e, tol) << shape.str() << " (" << wname(widths[w][0]) << ","
                        << wname(widths[w][1]) << ")";
      ++cases;
    }
  }

  std::cout << "ATTN_FIELD path=forward field=cases value=" << cases
            << std::endl;
  std::cout << "ATTN_FIELD path=forward field=max_rel_err value=" << worst
            << std::endl;
}

/**
 * @brief End-to-end cost of one fused attention layer, in microseconds.
 *
 * Timed on the ARM side around the FastRPC call, so the number INCLUDES
 * transport -- which is the honest figure, and the one MHA_HTP_PLAN.md §2's
 * arithmetic predicts against. Everything the optimisation work already
 * landed is inside it: the DMA ring with cross-block weight prefetch (PHASE A
 * passes every Kt block as one handle list), the session-scoped HMX, the
 * async accumulator copy-out and the vectorised quant/dequant.
 *
 * Rules this follows, each one a bug this project already shipped and found
 * (§9.7):
 *   - the correctness check is NOT in the timed region; a diff left inside one
 *     once produced a phantom 146 us slowdown;
 *   - three consecutive runs, all three printed, because thermal state moves
 *     these numbers;
 *   - field=... value=... marker lines with the unit in the field name, after
 *     a report script keyed by column position printed a stale number for
 *     every shape past the first, twice;
 *   - nothing is asserted. The reviewer compares against §2's prediction.
 *
 * NOT yet included: the softmax does not run on the worker pool, and P.V does
 * not prefetch across blocks (n_handles=1 per call). Both are Task 10, and
 * both need this breakdown first rather than a guess.
 */
TEST_F(HvxAttnScores, FusedForwardPerLayerCost) {
  /** Qwen3-0.6B: num_key_value_heads 8, num_attention_heads 16 (gqa 2),
     head_dim 128, 28 layers -- register_qwen3_0_6b() in
     Applications/CausalLM/api/model_config.cpp. */
  const uint32_t nch = 8u, gqa = 2u, head_dim = 128u, T = 256u, M_band = 64u;
  const uint32_t layers = 28u;
  const hexkl_w_width widths[4][2] = {{HEXKL_W_I8, HEXKL_W_I8},
                                      {HEXKL_W_I4, HEXKL_W_I4},
                                      {HEXKL_W_I8, HEXKL_W_I4},
                                      {HEXKL_W_I4, HEXKL_W_I8}};
  const uint32_t nHq = nch * gqa;

  std::cout << "\nQwen3-0.6B attention, one fused layer end to end "
            << "(includes FastRPC transport)\n"
            << "w_k,w_v are the Kt and V CACHE widths, in that order. "
               "Activations are u8 throughout,\n"
            << "so I4 means a u8 x i4 matmul -- nothing here is u4.\n"
            << "10 iterations; avg/min/max are over ALL 10, run 1 included, "
               "which is the\n"
            << "convention the QNN net-run numbers this gets compared against "
               "use. init\n"
            << "is register + kv_append, the one-time setup a graph prepare "
               "corresponds to.\n"
            << "probed is the same work with the DSP breakdown "
               "instrumentation on.\n"
            << "kv_len regime  w_k,w_v    avg1-10      min      max     "
               "init    probed\n";

  for (uint32_t kv_len : {512u, 1024u}) {
    for (uint32_t n_query : {1u, 128u}) {
      const uint32_t kv_from = kv_len - n_query;

      std::mt19937 rng(kv_len * 31u + n_query);
      std::uniform_real_distribution<float> d(-1.0f, 1.0f);
      std::vector<float> base(nch * head_dim);
      for (float &x : base) {
        x = d(rng);
      }
      std::vector<uint16_t> k16((size_t)kv_len * nch * head_dim);
      std::vector<uint16_t> v16(k16.size());
      for (size_t i = 0; i < k16.size(); ++i) {
        const size_t ch = i % ((size_t)nch * head_dim);
        k16[i] = f32_to_f16(base[ch] + 0.5f * d(rng));
        v16[i] = f32_to_f16(base[ch] + 0.5f * d(rng));
      }
      const size_t qn = (size_t)n_query * nHq * head_dim;
      /** q and out ride in every timed call -- 8 KB each way at decode, 1 MB
       * at prefill. See RpcBuf for why these two are ION-backed. */
      RpcBuf qbuf(qn * sizeof(float));
      RpcBuf obuf(qn * sizeof(float));
      float *q = (float *)qbuf.p;
      float *out = (float *)obuf.p;
      for (size_t qi = 0; qi < qn; ++qi) {
        q[qi] = d(rng);
      }
      std::memset(out, 0, qn * sizeof(float));
      /** 1 = ION-backed; 0 = the device library exports no rpcmem, or the
       * alloc failed -- either way these runs used plain heap. */
      std::cout << "ATTN_FIELD path=env field=ion_buffers value="
                << ((qbuf.ion && obuf.ion) ? 1 : 0) << std::endl;
      const float scale = 1.0f / std::sqrt((float)head_dim);

      for (int w = 0; w < 4; ++w) {
        uint32_t h = 0;
        /** init: everything that happens once per layer before a single token
         * is served -- the handle, and quantizing plus registering every KV
         * block. The analogue of a QNN graph prepare, and reported the same
         * way, separate from the per-inference numbers. */
        const auto init_t0 = std::chrono::steady_clock::now();
        const int rc_reg = nntr_hvx_attn_register(
          handle_, nch, gqa, head_dim, kv_len, T, M_band,
          (uint32_t)widths[w][0], (uint32_t)widths[w][1], &h);
        const int rc_app = (rc_reg == AEE_SUCCESS)
                             ? nntr_hvx_attn_kv_append(
                                 handle_, h, 0u, kv_len, k16.data(),
                                 (int)k16.size(), v16.data(), (int)v16.size())
                             : rc_reg;
        const auto init_t1 = std::chrono::steady_clock::now();
        ASSERT_EQ(rc_reg, AEE_SUCCESS);
        ASSERT_EQ(rc_app, AEE_SUCCESS);
        const double init_us =
          std::chrono::duration<double, std::micro>(init_t1 - init_t0).count();

        /** The STEP append: the KV rows this inference contributes, which is
         * 1 row at decode and n_query at prefill. init above appended the
         * whole cache at once, which is not what a running model pays -- it
         * pays this, every token, in every layer, before forward can run.
         * Re-appending rows already present is idempotent (the affected
         * blocks are released and re-registered from the same shadow data)
         * and kv_len does not move, so this measures the real per-step cost
         * without disturbing the forwards below. */
        constexpr int kIters = 10;
        const size_t row_elems = (size_t)nch * head_dim;
        double app[kIters];
        for (int r = 0; r < kIters; ++r) {
          const auto t0 = std::chrono::steady_clock::now();
          const int err = nntr_hvx_attn_kv_append(
            handle_, h, kv_from, n_query, k16.data() + kv_from * row_elems,
            (int)((size_t)n_query * row_elems),
            v16.data() + kv_from * row_elems,
            (int)((size_t)n_query * row_elems));
          const auto t1 = std::chrono::steady_clock::now();
          ASSERT_EQ(err, AEE_SUCCESS);
          app[r] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        double app_sum = 0.0;
        for (int r = 0; r < kIters; ++r) {
          app_sum += app[r];
        }
        const double app_avg = app_sum / (double)kIters;

        /** PRODUCTION path: attn_forward passes no stage_us, so the DSP's
         * stage timers and the in-loop probes are both off. That is what a
         * real caller pays and therefore what the headline reports; the
         * instrumented runs below are for the breakdown, and the gap between
         * them is published rather than folded into the headline.
         *
         * Ten iterations, statistics over ALL TEN including the first --
         * the convention the QNN numbers this gets compared against are
         * quoted under. Run 1 pays page faults on the freshly registered
         * shadows and so pulls the average up; it is also reported on its
         * own, as us_first, so how much it pulls stays visible. */
        double iter[kIters];
        for (int r = 0; r < kIters; ++r) {
          const auto t0 = std::chrono::steady_clock::now();
          const int err =
            nntr_hvx_attn_forward(handle_, h, kv_from, n_query, scale, 1u, 0u,
                                  nullptr, 0, q, (int)qn, out, (int)qn);
          const auto t1 = std::chrono::steady_clock::now();
          ASSERT_EQ(err, AEE_SUCCESS);
          iter[r] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        double sum = 0.0, lo = iter[0], hi = iter[0];
        for (int r = 0; r < kIters; ++r) {
          sum += iter[r];
          lo = std::min(lo, iter[r]);
          hi = std::max(hi, iter[r]);
        }
        const double avg = sum / (double)kIters;

        double us[3];
        std::vector<uint32_t> stage_of[3];
        for (int r = 0; r < 3; ++r) {
          /** Mirrors HEXKL_ATTN_N_STAGES; the DSP header is not includable
             from the ARM side, so the skel rejects a stale count with
             AEE_EBADPARM rather than silently truncating. */
          std::vector<uint32_t> stage(13, 0u);
          const auto t0 = std::chrono::steady_clock::now();
          const int err = nntr_hvx_attn_forward_timed(
            handle_, h, kv_from, n_query, scale, 1u, 0u, nullptr, 0, q, (int)qn,
            out, (int)qn, stage.data(), (int)stage.size());
          const auto t1 = std::chrono::steady_clock::now();
          /* Checked after the clock stops, deliberately. */
          ASSERT_EQ(err, AEE_SUCCESS);
          us[r] = std::chrono::duration<double, std::micro>(t1 - t0).count();
          stage_of[r] = stage;
        }
        ASSERT_EQ(nntr_hvx_attn_release(handle_, h), AEE_SUCCESS);

        /** MEDIAN of the three, not the min. Wall clock on this device is
         * bimodal at roughly +-25% (same shape, same binary, run to run --
         * DVFS residency, not our code), so the min publishes whichever run
         * happened to land in the fast state. MHA_HTP_PLAN.md §9.7 already
         * requires printing all three runs; this only changes which one gets
         * published as the number, and it stays a single measured run rather
         * than a mean so the stage breakdown beside it comes from that same
         * run and still sums to its own dsp_total. */
        int ord[3] = {0, 1, 2};
        std::sort(ord, ord + 3, [&us](int a, int b) { return us[a] < us[b]; });
        const double med = avg;               /* production, uninstrumented */
        const double med_probed = us[ord[1]]; /* instrumented, for the parts */
        const std::vector<uint32_t> &med_stage = stage_of[ord[1]];
        std::ostringstream pair;
        pair << wname(widths[w][0]) << "," << wname(widths[w][1]);
        std::cout << std::left << std::setw(7) << kv_len << std::setw(9)
                  << (n_query == 1 ? "decode" : "prefill") << std::setw(10)
                  << pair.str() << std::right << std::fixed
                  << std::setprecision(1) << std::setw(9) << avg << std::setw(9)
                  << lo << std::setw(9) << hi << std::setw(9) << init_us
                  << std::setw(10) << med_probed << "\n"
                  << std::left;
        std::cout << "ATTN_FIELD path=forward_"
                  << (n_query == 1 ? "dec" : "pre") << "_kv" << kv_len << "_"
                  << wname(widths[w][0]) << wname(widths[w][1])
                  << " field=us_per_layer value=" << med << std::endl;
        std::cout << "ATTN_FIELD path=forward_"
                  << (n_query == 1 ? "dec" : "pre") << "_kv" << kv_len << "_"
                  << wname(widths[w][0]) << wname(widths[w][1])
                  << " field=us_per_token_all_layers value=" << med * layers
                  << std::endl;

        /** DSP-internal breakdown. The host timed the whole call and this
         * timed the inside, so (med - dsp_total) is the MEASURED FastRPC
         * transport rather than an assumed 404 us. */
        static const char *kStage[13] = {
          "qk_us", "softmax_us", "pv_us", "accum_us", "gather_us",
          "dsp_total_us",
          "layer_run_calls" /* Tier 0 probes, subdividing qk+pv: */,
          "acc_read_us", "acc_copy_us", "dequant_us", "quant_us", "drain_us",
          /** not a time: the accumulator row stride the in-place tile dequant
             derived, or 0 when layer_run fell back to the vendor copy */
          "acc_stride"};
        std::ostringstream path;
        path << "forward_" << (n_query == 1 ? "dec" : "pre") << "_kv" << kv_len
             << "_" << wname(widths[w][0]) << wname(widths[w][1]);
        for (int st = 0; st < 13; ++st) {
          std::cout << "ATTN_STAGE path=" << path.str()
                    << " field=" << kStage[st] << " value=" << med_stage[st]
                    << std::endl;
        }
        /** dsp_total came from the INSTRUMENTED run, so subtract it from that
         * run's wall -- mixing it with the production wall would fold the
         * instrumentation into the transport estimate. */
        std::cout << "ATTN_STAGE path=" << path.str()
                  << " field=transport_us value="
                  << (med_probed - (double)med_stage[5]) << std::endl;
        std::cout << "ATTN_STAGE path=" << path.str()
                  << " field=probe_overhead_us value=" << (med_probed - med)
                  << std::endl;
        static const char *kRun[7] = {"us_avg_1_10", "us_min",  "us_max",
                                      "us_first",    "init_us", "append_us",
                                      "step_us"};
        /** step_us is what a running model actually pays per token per layer:
         * this step's KV append plus the forward. The forward alone is
         * us_avg_1_10. */
        const double run_v[7] = {avg,     lo,      hi,           iter[0],
                                 init_us, app_avg, app_avg + avg};
        for (int f = 0; f < 7; ++f) {
          std::cout << "ATTN_FIELD path=" << path.str() << " field=" << kRun[f]
                    << " value=" << run_v[f] << std::endl;
        }
      }
    }
  }
}

/* ---- [#81] decode attention at m=1 with the DSP-resident KV cache -------- */

class HvxAttnM1 : public HtpSession {};

namespace {

/** @brief LFM2.5's attention shape: 32 q heads, 8 kv heads, head_dim 64. */
constexpr uint32_t kM1Kv = 8, kM1Gqa = 4, kM1Hd = 64, kM1Nq = kM1Kv * kM1Gqa;
constexpr float kM1Scale = 0.125f;

int32_t m1_bits_of(float f) {
  int32_t i;
  std::memcpy(&i, &f, sizeof(i));
  return i;
}

/** @brief q rows from attn_m1_cases.h, the host check's inputs: heads 0..2
 *         zero / fp16-subnormal / large, head 3 soft, the last two the
 *         adversarial score heads. */
void m1_fill_q(std::vector<float> &q, amc_rng &rng) {
  q.assign((size_t)kM1Nq * kM1Hd, 0.0f);
  amc_fill_q(&rng, q.data(), kM1Nq, kM1Hd);
}

/** @brief L rows of k and v, [L][kv][head_dim], from attn_m1_cases.h:
 *         positions 0 / 1 / 2 the fixed kinds, then the adversarial score
 *         rows; with @a q, also the PV midpoint cases planted for it (the
 *         count is returned; the timing tests skip the search). */
uint32_t m1_fill_kv(std::vector<float> &k, std::vector<float> &v, uint32_t L,
                    amc_rng &rng, const std::vector<float> *q = nullptr) {
  const size_t row = (size_t)kM1Kv * kM1Hd;
  k.assign(L * row, 0.0f);
  v.assign(L * row, 0.0f);
  amc_fill_kv(&rng, k.data(), v.data(), L, kM1Kv, kM1Gqa, kM1Hd);
  return q ? amc_plant_pv(q->data(), k.data(), v.data(), L, kM1Kv, kM1Gqa,
                          kM1Hd)
           : 0u;
}

/** @brief attn_m1_det.h's output and stats for L rows. */
void m1_spec(const std::vector<float> &q, const std::vector<float> &k,
             const std::vector<float> &v, uint32_t L, uint32_t max_seq,
             std::vector<float> *out, std::vector<float> *stats) {
  std::vector<float> kt((size_t)kM1Kv * kM1Hd * max_seq, 0.0f);
  std::vector<float> vv((size_t)kM1Kv * max_seq * kM1Hd, 0.0f);
  std::vector<float> e(L);
  for (uint32_t p = 0; p < L; ++p) {
    for (uint32_t h = 0; h < kM1Kv; ++h) {
      attn_m1_det_append(kt.data() + (size_t)h * kM1Hd * max_seq,
                         vv.data() + (size_t)h * max_seq * kM1Hd, kM1Hd,
                         max_seq, p, k.data() + ((size_t)p * kM1Kv + h) * kM1Hd,
                         v.data() + ((size_t)p * kM1Kv + h) * kM1Hd);
    }
  }
  out->assign((size_t)kM1Nq * kM1Hd, 0.0f);
  stats->assign(2u * kM1Nq, 0.0f);
  attn_m1_det_forward(q.data(), kt.data(), vv.data(), kM1Kv, kM1Gqa, kM1Hd,
                      max_seq, L, kM1Scale, e.data(), out->data(),
                      stats->data());
}

/** @brief max_abs_err / max|V| of @a out against softmax(q.Kt*scale).V in
 *         double (printed, not asserted: rule 25). */
double m1_double_ref_err(const std::vector<float> &q,
                         const std::vector<float> &k,
                         const std::vector<float> &v, uint32_t L,
                         const std::vector<float> &out) {
  std::vector<double> s(L);
  double worst = 0.0;
  for (uint32_t hq = 0; hq < kM1Nq; ++hq) {
    const uint32_t h = hq / kM1Gqa;
    double m = -INFINITY, den = 0.0;
    for (uint32_t p = 0; p < L; ++p) {
      double acc = 0.0;
      for (uint32_t d = 0; d < kM1Hd; ++d) {
        const size_t i = ((size_t)p * kM1Kv + h) * kM1Hd + d;
        acc += (double)q[(size_t)hq * kM1Hd + d] * (double)k[i];
        den = std::max(den, std::fabs((double)v[i]));
      }
      s[p] = acc * (double)kM1Scale;
      m = std::max(m, s[p]);
    }
    double l = 0.0;
    for (uint32_t p = 0; p < L; ++p) {
      s[p] = std::exp(s[p] - m);
      l += s[p];
    }
    for (uint32_t d = 0; d < kM1Hd; ++d) {
      double o = 0.0;
      for (uint32_t p = 0; p < L; ++p) {
        o += s[p] * (double)v[((size_t)p * kM1Kv + h) * kM1Hd + d];
      }
      o /= l;
      const double err = std::fabs((double)out[(size_t)hq * kM1Hd + d] - o);
      if (den > 0.0) {
        worst = std::max(worst, err / den);
      }
    }
  }
  return worst;
}

int m1_count_bad(const std::vector<float> &dsp, const std::vector<float> &ref,
                 const char *what) {
  int bad = 0;
  for (size_t i = 0; i < dsp.size(); ++i) {
    if (m1_bits_of(dsp[i]) != m1_bits_of(ref[i])) {
      if (bad == 0) {
        std::cout << "ATTN_M1 " << what << " first mismatch i=" << i
                  << std::hexfloat << " dsp=" << dsp[i] << " ref=" << ref[i]
                  << std::defaultfloat << std::endl;
      }
      ++bad;
    }
  }
  return bad;
}

/** @brief kv_append of L-1 rows to @a layer from position 0 (a rewind when
 *         the layer already holds rows), then forward of the last. */
int m1_run(remote_handle64 handle, uint32_t layer, const std::vector<float> &q,
           const std::vector<float> &k, const std::vector<float> &v, uint32_t L,
           std::vector<float> *out, std::vector<float> *stats) {
  const size_t row = (size_t)kM1Kv * kM1Hd;
  int err = nntr_hvx_attn_m1_kv_append(handle, layer, 0u, L - 1u, k.data(),
                                       (int)((L - 1u) * row), v.data(),
                                       (int)((L - 1u) * row));
  if (err != AEE_SUCCESS) {
    return err;
  }
  out->assign((size_t)kM1Nq * kM1Hd, 0.0f);
  return nntr_hvx_attn_m1_forward(
    handle, layer, L - 1u, kM1Scale, q.data(), (int)q.size(),
    k.data() + (size_t)(L - 1u) * row, (int)row,
    v.data() + (size_t)(L - 1u) * row, (int)row, out->data(), (int)out->size(),
    stats ? stats->data() : nullptr, stats ? (int)stats->size() : 0);
}

} // namespace

/**
 * @brief A bad shape is AEE_EINVALIDFORMAT, a hole or a missing cache
 *        AEE_EBADSTATE, from the entries' own checks. AEE_EBADPARM
 *        (0x8000040e) here means the skel predates these four methods
 *        (rule 3) -- rebuild it before reading the other cases.
 */
TEST_F(HvxAttnM1, RejectsBadShapes) {
  std::vector<float> q((size_t)kM1Nq * kM1Hd, 1.0f),
    k((size_t)kM1Kv * kM1Hd, 1.0f), v(k.size(), 1.0f), y(q.size());
  int err = nntr_hvx_attn_m1_forward(
    handle_, 0u, 0u, kM1Scale, q.data(), (int)q.size(), k.data(), (int)k.size(),
    v.data(), (int)v.size(), y.data(), (int)y.size(), nullptr, 0);
  EXPECT_EQ(err, AEE_EBADSTATE + kDspOffset)
    << "forward without a cache: got " << hex(err);
  // max_seq 100: not a multiple of 32.
  err = nntr_hvx_attn_m1_register(handle_, 2u, kM1Kv, kM1Gqa, kM1Hd, 100u);
  EXPECT_EQ(err, AEE_EINVALIDFORMAT + kDspOffset)
    << "register max_seq 100: got " << hex(err);
  // head_dim 32 and 128: the fp16 CPU order is head_dim 64 only (#152).
  for (uint32_t hd : {32u, 128u}) {
    err = nntr_hvx_attn_m1_register(handle_, 2u, kM1Kv, kM1Gqa, hd, 1024u);
    EXPECT_EQ(err, AEE_EINVALIDFORMAT + kDspOffset)
      << "register head_dim " << hd << ": got " << hex(err);
  }
  err = nntr_hvx_attn_m1_register(handle_, 2u, kM1Kv, kM1Gqa, kM1Hd, 1024u);
  ASSERT_EQ(err, AEE_SUCCESS) << "register: " << hex(err);
  err = nntr_hvx_attn_m1_register(handle_, 2u, kM1Kv, kM1Gqa, kM1Hd, 1024u);
  EXPECT_EQ(err, AEE_EBADSTATE + kDspOffset)
    << "second register: got " << hex(err);
  // q one head short.
  err = nntr_hvx_attn_m1_forward(handle_, 0u, 0u, kM1Scale, q.data(),
                                 (int)q.size() - kM1Hd, k.data(), (int)k.size(),
                                 v.data(), (int)v.size(), y.data(),
                                 (int)y.size(), nullptr, 0);
  EXPECT_EQ(err, AEE_EINVALIDFORMAT + kDspOffset)
    << "forward q of 31 heads: got " << hex(err);
  // A hole: position 1 of an empty layer.
  err = nntr_hvx_attn_m1_forward(
    handle_, 0u, 1u, kM1Scale, q.data(), (int)q.size(), k.data(), (int)k.size(),
    v.data(), (int)v.size(), y.data(), (int)y.size(), nullptr, 0);
  EXPECT_EQ(err, AEE_EBADSTATE + kDspOffset) << "hole: got " << hex(err);
  err = nntr_hvx_attn_m1_forward(handle_, 0u, 1024u, kM1Scale, q.data(),
                                 (int)q.size(), k.data(), (int)k.size(),
                                 v.data(), (int)v.size(), y.data(),
                                 (int)y.size(), nullptr, 0);
  EXPECT_EQ(err, AEE_EINVALIDFORMAT + kDspOffset)
    << "pos == max_seq: got " << hex(err);
  err = nntr_hvx_attn_m1_release(handle_);
  EXPECT_EQ(err, AEE_SUCCESS) << "release: " << hex(err);
  err = nntr_hvx_attn_m1_release(handle_);
  EXPECT_EQ(err, AEE_EBADSTATE + kDspOffset)
    << "second release: got " << hex(err);
}

/**
 * @brief The eight lengths of the host check against attn_m1_det.h compiled
 *        into this binary, bit for bit, on a cache registered at the
 *        LFM2.5 shape and max_seq 2048 (the 48 MiB allocation of plan 81
 *        section 3.1, proved once here), with attn_m1_cases.h's rows: the
 *        fused-FMA midpoint cases (score_cases, pv_cases) are where a
 *        silicon Vsf op that is not IEEE would show first (#152). stats
 *        (m, l) are compared too and printed on a mismatch, so a bad count
 *        names the stage; the double-reference error is printed, not
 *        asserted (rule 25).
 */
TEST_F(HvxAttnM1, MatchesDetSpecBitExact) {
  const uint32_t max_seq = 2048u, layer = 5u;
  int err =
    nntr_hvx_attn_m1_register(handle_, 6u, kM1Kv, kM1Gqa, kM1Hd, max_seq);
  ASSERT_EQ(err, AEE_SUCCESS)
    << "register 6 x 8 x 64 x 2048 (48 MiB): " << hex(err)
    << " (0x8000040e = stale skel)";
  amc_rng rng{0x81000001u};
  std::vector<float> q, k, v, out, stats(2u * kM1Nq), out_ref, stats_ref;
  m1_fill_q(q, rng);
  for (uint32_t L : {1u, 63u, 64u, 65u, 512u, 513u, 1024u, 1536u}) {
    const uint32_t planted = m1_fill_kv(k, v, L, rng, &q);
    std::fill(stats.begin(), stats.end(), 0.0f);
    err = m1_run(handle_, layer, q, k, v, L, &out, &stats);
    ASSERT_EQ(err, AEE_SUCCESS) << "L=" << L << ": " << hex(err);
    m1_spec(q, k, v, L, max_seq, &out_ref, &stats_ref);
    const int bad_stats = m1_count_bad(stats, stats_ref, "stats (m, l)");
    const int bad = m1_count_bad(out, out_ref, "out");
    std::cout << "ATTN_M1_FIELD L=" << L << " bad=" << bad
              << " bad_stats=" << bad_stats << " of " << out.size()
              << " pv_cases=" << planted
              << " score_cases=" << (L > 3u ? 2u * (L - 3u) : 0u)
              << " err/max|V|(double)=" << m1_double_ref_err(q, k, v, L, out)
              << std::endl;
    EXPECT_EQ(bad_stats, 0)
      << "L=" << L << ": the max or the sum differs from the spec";
    EXPECT_EQ(bad, 0) << "L=" << L << ": out differs from the spec";
  }
  EXPECT_EQ(nntr_hvx_attn_m1_release(handle_), AEE_SUCCESS);
}

/** @brief 65 forward calls from an empty layer vs one kv_append of 64 rows
 *         plus a forward on another layer: the last outputs byte-equal (the
 *         host check also compares the cache bytes, which the entries do
 *         not expose). */
TEST_F(HvxAttnM1, AppendChainEqualsBulk) {
  const uint32_t L = 65u;
  int err = nntr_hvx_attn_m1_register(handle_, 2u, kM1Kv, kM1Gqa, kM1Hd, 1024u);
  ASSERT_EQ(err, AEE_SUCCESS) << "register: " << hex(err);
  amc_rng rng{0x81000002u};
  std::vector<float> q, k, v, out_a((size_t)kM1Nq * kM1Hd), out_b;
  m1_fill_q(q, rng);
  m1_fill_kv(k, v, L, rng);
  const size_t row = (size_t)kM1Kv * kM1Hd;
  for (uint32_t p = 0; p < L; ++p) {
    err = nntr_hvx_attn_m1_forward(handle_, 0u, p, kM1Scale, q.data(),
                                   (int)q.size(), k.data() + p * row, (int)row,
                                   v.data() + p * row, (int)row, out_a.data(),
                                   (int)out_a.size(), nullptr, 0);
    ASSERT_EQ(err, AEE_SUCCESS) << "chain pos " << p << ": " << hex(err);
  }
  err = m1_run(handle_, 1u, q, k, v, L, &out_b, nullptr);
  ASSERT_EQ(err, AEE_SUCCESS) << "bulk: " << hex(err);
  const int bad = m1_count_bad(out_a, out_b, "chain vs bulk");
  std::cout << "ATTN_M1_FIELD append_chain L=" << L << " bad=" << bad
            << std::endl;
  EXPECT_EQ(bad, 0) << "the append chain's last output differs from bulk";
  EXPECT_EQ(nntr_hvx_attn_m1_release(handle_), AEE_SUCCESS);
}

namespace {

/** @brief One forward at the last position of @a layer, host-timed, with
 *         the phase words (#146) requested when @a words is non-null. */
int m1_timed_forward(remote_handle64 handle, uint32_t layer, uint32_t L,
                     const std::vector<float> &q, const std::vector<float> &k,
                     const std::vector<float> &v, std::vector<float> *out,
                     std::vector<uint32_t> *words, double *us) {
  const size_t row = (size_t)kM1Kv * kM1Hd;
  std::vector<float> stats(words ? 2u * kM1Nq + ATTN_M1_PROF_WORDS : 0u);
  const auto t0 = std::chrono::steady_clock::now();
  const int err = nntr_hvx_attn_m1_forward(
    handle, layer, L - 1u, kM1Scale, q.data(), (int)q.size(),
    k.data() + (size_t)(L - 1u) * row, (int)row,
    v.data() + (size_t)(L - 1u) * row, (int)row, out->data(), (int)out->size(),
    words ? stats.data() : nullptr, (int)stats.size());
  const auto t1 = std::chrono::steady_clock::now();
  *us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  if (words) {
    words->resize(ATTN_M1_PROF_WORDS);
    std::memcpy(words->data(), stats.data() + 2u * kM1Nq,
                ATTN_M1_PROF_WORDS * sizeof(uint32_t));
  }
  return err;
}

template <typename T> T m1_median(std::vector<T> x) {
  std::sort(x.begin(), x.end());
  return x[x.size() / 2];
}

/** @brief The ATTN_M1_PHASE line: the median of each word over the calls.
 *         dsp_us is CALL_QT / 19.2; mhz is (APPEND + POOL) pcycles per
 *         dsp_us of the same call -- the effective clock, a little low
 *         since the entry's checks and the reduce sit outside both. */
void m1_print_phase(uint32_t pos, const char *which,
                    const std::vector<std::vector<uint32_t>> &calls) {
  auto word = [&](uint32_t w) {
    std::vector<uint32_t> x;
    for (const auto &c : calls) {
      x.push_back(c[w]);
    }
    return m1_median(x);
  };
  std::vector<double> mhz;
  for (const auto &c : calls) {
    const double us = c[ATTN_M1_PROF_CALL_QT] / 19.2;
    mhz.push_back(
      us > 0.0 ? (c[ATTN_M1_PROF_APPEND] + (double)c[ATTN_M1_PROF_POOL]) / us
               : 0.0);
  }
  std::cout << "ATTN_M1_PHASE pos=" << pos << " " << which
            << " append=" << word(ATTN_M1_PROF_APPEND)
            << " scores=" << word(ATTN_M1_PROF_SCORES)
            << " softmax=" << word(ATTN_M1_PROF_SOFTMAX)
            << " pv=" << word(ATTN_M1_PROF_PV)
            << " busy_max=" << word(ATTN_M1_PROF_BUSY_MAX)
            << " start_max=" << word(ATTN_M1_PROF_START_MAX)
            << " pool=" << word(ATTN_M1_PROF_POOL)
            << " lanes=" << word(ATTN_M1_PROF_LANES)
            << " dsp_us=" << word(ATTN_M1_PROF_CALL_QT) / 19.2
            << " mhz=" << m1_median(mhz) << " (pcycles, median of "
            << calls.size() << ")" << std::endl;
}

constexpr const char *kM1StaleProf =
  " (AEE_EINVALIDFORMAT here = a skel without the #146 phase words)";

} // namespace

/** @brief The per-layer forward cost at pos 511, 1023 and 1535 (median of
 *         10 calls, host-timed, transport included): the first read of plan
 *         81 section 0's 0.5 / 1.0 ms per token estimate for 6 layers.
 *         Printed, not asserted. Since #146 each position also prints the
 *         phase split of 10 more calls with the words requested (warm: the
 *         same layer every call), and then the same positions COLD: 6
 *         layers at max_seq 2048 and the layer rotated per call, so each
 *         call's slab (4.19 MB at pos 1023) was evicted by the other five,
 *         as in the model (plan 146 section 3.1). The cold us= is taken
 *         with the words requested; the warm us= line is unchanged. Since
 *         #170 the warm cache is max_seq 2048 too (pos 1535 needs it). */
TEST_F(HvxAttnM1, PerLayerCost) {
  int err = nntr_hvx_attn_m1_register(handle_, 2u, kM1Kv, kM1Gqa, kM1Hd, 2048u);
  ASSERT_EQ(err, AEE_SUCCESS) << "register: " << hex(err);
  amc_rng rng{0x81000003u};
  std::vector<float> q, k, v, out((size_t)kM1Nq * kM1Hd);
  m1_fill_q(q, rng);
  const size_t row = (size_t)kM1Kv * kM1Hd;
  for (uint32_t L : {512u, 1024u, 1536u}) {
    m1_fill_kv(k, v, L, rng);
    err = nntr_hvx_attn_m1_kv_append(handle_, 0u, 0u, L - 1u, k.data(),
                                     (int)((L - 1u) * row), v.data(),
                                     (int)((L - 1u) * row));
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_append: " << hex(err);
    std::vector<double> us;
    for (int it = 0; it < 10; ++it) {
      double t = 0.0;
      err = m1_timed_forward(handle_, 0u, L, q, k, v, &out, nullptr, &t);
      ASSERT_EQ(err, AEE_SUCCESS) << "forward: " << hex(err);
      us.push_back(t);
    }
    std::sort(us.begin(), us.end());
    std::cout << "ATTN_M1_FIELD pos=" << (L - 1u) << " us=" << us[us.size() / 2]
              << " us_min=" << us.front() << " (host-timed, median of 10)"
              << std::endl;
    std::vector<std::vector<uint32_t>> calls(10);
    for (auto &w : calls) {
      double t = 0.0;
      err = m1_timed_forward(handle_, 0u, L, q, k, v, &out, &w, &t);
      ASSERT_EQ(err, AEE_SUCCESS)
        << "forward with the phase words: " << hex(err) << kM1StaleProf;
    }
    m1_print_phase(L - 1u, "warm", calls);
  }
  EXPECT_EQ(nntr_hvx_attn_m1_release(handle_), AEE_SUCCESS);

  constexpr uint32_t kLayers = 6u;
  err =
    nntr_hvx_attn_m1_register(handle_, kLayers, kM1Kv, kM1Gqa, kM1Hd, 2048u);
  ASSERT_EQ(err, AEE_SUCCESS) << "register 6 x 2048: " << hex(err);
  for (uint32_t L : {512u, 1024u, 1536u}) {
    m1_fill_kv(k, v, L, rng);
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      err = nntr_hvx_attn_m1_kv_append(handle_, layer, 0u, L - 1u, k.data(),
                                       (int)((L - 1u) * row), v.data(),
                                       (int)((L - 1u) * row));
      ASSERT_EQ(err, AEE_SUCCESS)
        << "kv_append layer " << layer << ": " << hex(err);
    }
    /* One round of 6 to put every layer's slab behind the other five, then
       10 timed calls, each on the least recently used layer. */
    std::vector<double> us;
    std::vector<std::vector<uint32_t>> calls;
    for (uint32_t it = 0; it < kLayers + 10u; ++it) {
      std::vector<uint32_t> w;
      double t = 0.0;
      err = m1_timed_forward(handle_, it % kLayers, L, q, k, v, &out, &w, &t);
      ASSERT_EQ(err, AEE_SUCCESS)
        << "cold forward: " << hex(err) << kM1StaleProf;
      if (it >= kLayers) {
        us.push_back(t);
        calls.push_back(w);
      }
    }
    std::sort(us.begin(), us.end());
    std::cout << "ATTN_M1_FIELD cold pos=" << (L - 1u)
              << " us=" << us[us.size() / 2] << " us_min=" << us.front()
              << " (host-timed, median of 10, 6 layers rotated)" << std::endl;
    m1_print_phase(L - 1u, "cold", calls);
  }
  EXPECT_EQ(nntr_hvx_attn_m1_release(handle_), AEE_SUCCESS);
}

/* ---- [#170] the fp16-lane primitives on silicon (plan 170 step 1, S1) --- */

class HvxAttnM1Probe : public HtpSession {};

namespace {

/** @brief One semantics op of attn_m1_probe over a.size() lanes (padded
 *         to a multiple of 64 with lane 0's operands). */
int probe_sem(remote_handle64 h, uint32_t op, std::vector<uint16_t> a,
              std::vector<uint16_t> b, std::vector<uint16_t> c,
              std::vector<uint16_t> *y) {
  const size_t n = a.size(), np = (n + 63u) / 64u * 64u;
  a.resize(np, a[0]);
  b.resize(np, b[0]);
  c.resize(np, c[0]);
  y->assign(np, 0u);
  const int err =
    nntr_hvx_attn_m1_probe(h, op, 1u, 1u, a.data(), (int)np, b.data(), (int)np,
                           c.data(), (int)np, y->data(), (int)np, nullptr, 0);
  y->resize(n);
  return err;
}

/** @brief hvx_hf_fma on (c, a, b) triples against attn_m1_det_fma16, bit
 *         for bit; triples whose fused result is past 65504 are skipped.
 *         Prints the ATTN_M1_PROBE qfma line; returns the bad count. */
int probe_qfma(remote_handle64 h, const char *name,
               const std::vector<uint16_t> &c, const std::vector<uint16_t> &a,
               const std::vector<uint16_t> &b) {
  std::vector<uint16_t> y;
  const int err = probe_sem(h, ATTN_M1_PROBE_QFMA, a, b, c, &y);
  EXPECT_EQ(err, AEE_SUCCESS)
    << "attn_m1_probe qfma: " << hex(err) << " (0x8000040e = stale skel)";
  if (err != AEE_SUCCESS) {
    return -1;
  }
  int n = 0, hz = 0, sub = 0, bad = 0;
  for (size_t i = 0; i < c.size(); ++i) {
    const float fc = amc_h2f(c[i]), fa = amc_h2f(a[i]), fb = amc_h2f(b[i]);
    const float ref = attn_m1_det_fma16(fc, fa, fb);
    if (std::fabs(ref) > 65504.0f) {
      continue;
    }
    ++n;
    hz += amc_is_midpoint_case(fc, fa, fb);
    sub += std::fabs(ref) < std::ldexp(1.0f, -14);
    if (y[i] != amc_f2h(ref)) {
      if (bad < 4) {
        std::cout << "ATTN_M1_PROBE qfma " << name << " mismatch c=" << std::hex
                  << c[i] << " a=" << a[i] << " b=" << b[i] << " got=" << y[i]
                  << " want=" << amc_f2h(ref) << std::dec << std::endl;
      }
      ++bad;
    }
  }
  std::cout << "ATTN_M1_PROBE qfma " << name << " n=" << n << " hazards=" << hz
            << " subnormal=" << sub << " bad=" << bad << std::endl;
  return bad;
}

/** @brief The case file of tools/htp/attn_fma_cases.py: uint16 (c, a, b)
 *         triples from #136's dump_attn. NNTR_ATTN_FMA_CASES, else
 *         ./attn_fma_cases.bin. */
bool read_fma_cases(std::vector<uint16_t> *c, std::vector<uint16_t> *a,
                    std::vector<uint16_t> *b, std::string *path) {
  const char *env = std::getenv("NNTR_ATTN_FMA_CASES");
  *path = env ? env : "attn_fma_cases.bin";
  FILE *f = std::fopen(path->c_str(), "rb");
  if (!f) {
    return false;
  }
  uint16_t t[3];
  while (std::fread(t, sizeof(uint16_t), 3, f) == 3) {
    c->push_back(t[0]);
    a->push_back(t[1]);
    b->push_back(t[2]);
  }
  std::fclose(f);
  return !c->empty();
}

} // namespace

/**
 * @brief G1 of plan 170: hvx_attn_m1_hf.h on silicon against attn_m1_det.h
 *        compiled into this binary, bit for bit.
 *   - qfma, the one-rounding FMA: the #136-derived case file (every
 *     double-rounding hazard and inexact midpoint of a real-data replay,
 *     plus a sample), attn_m1_cases.h's adversarial (76,800) and zero /
 *     sign (25,600) families. Its bad=0 is what the design rests on
 *     (plan 170 step 2's stop rule).
 *   - hf add / sub / mul / max, * 0.125 and 0 + x over every finite fp16
 *     a x 4 random b (max by value).
 *   - exp16 (the vector exp_ps) at all 31,745 fp16 d <= 0.
 *   - div16 on the host check's hard quotients (rne16(e * recip_det(l))
 *     off by one, or e / l a tie), found here with swiglu_det_recip, the
 *     scalar twin of hvx_recip_det_sf.
 * Each row prints an ATTN_M1_PROBE line; every row is also asserted.
 */
TEST_F(HvxAttnM1Probe, Semantics) {
  std::vector<uint16_t> c, a, b, y;
  std::string path;
  ASSERT_TRUE(read_fma_cases(&c, &a, &b, &path))
    << "no fma case file at " << path
    << " (tools/htp/attn_fma_cases.py; set NNTR_ATTN_FMA_CASES)";
  EXPECT_EQ(probe_qfma(handle_, "cases", c, a, b), 0);
  amc_rng r{0x17000001u};
  c.assign(76800u, 0u), a.assign(76800u, 0u), b.assign(76800u, 0u);
  for (uint32_t i = 0; i < 76800u; ++i) {
    amc_fma_adversarial(&r, i, &c[i], &a[i], &b[i]);
  }
  EXPECT_EQ(probe_qfma(handle_, "adversarial", c, a, b), 0);
  c.assign(25600u, 0u), a.assign(25600u, 0u), b.assign(25600u, 0u);
  for (uint32_t i = 0; i < 25600u; ++i) {
    amc_fma_zero_sign(&r, &c[i], &a[i], &b[i]);
  }
  EXPECT_EQ(probe_qfma(handle_, "zero_sign", c, a, b), 0);

  /* hf ops: every finite fp16 a (both signs) x 4 random b */
  a.clear(), b.clear();
  for (uint32_t rep = 0; rep < 4u; ++rep) {
    for (uint32_t x = 0; x < 2u * 0x7C00u; ++x) {
      a.push_back((uint16_t)(((x & 1u) << 15) | (x >> 1)));
      const uint32_t k = amc_next(&r);
      b.push_back(
        (uint16_t)((k & 0x8000u) | ((k & 3u) == 0u ? 0u : (k >> 2) % 0x7C00u)));
    }
  }
  static const struct {
    uint32_t op;
    const char *name;
  } kOps[] = {
    {ATTN_M1_PROBE_ADD, "add"},         {ATTN_M1_PROBE_SUB, "sub"},
    {ATTN_M1_PROBE_MUL, "mul"},         {ATTN_M1_PROBE_MAX, "max"},
    {ATTN_M1_PROBE_EIGHTH, "mul0.125"}, {ATTN_M1_PROBE_ZPLUS, "zero_plus"}};
  for (const auto &o : kOps) {
    const int err = probe_sem(handle_, o.op, a, b, a, &y);
    ASSERT_EQ(err, AEE_SUCCESS) << o.name << ": " << hex(err);
    int n = 0, bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      const float x = amc_h2f(a[i]), z = amc_h2f(b[i]);
      float ref = 0.0f;
      switch (o.op) {
      case ATTN_M1_PROBE_ADD:
        ref = attn_m1_det_rne16(attn_m1_det_add(x, z));
        break;
      case ATTN_M1_PROBE_SUB:
        ref = attn_m1_det_rne16(attn_m1_det_sub(x, z));
        break;
      case ATTN_M1_PROBE_MUL:
        ref = attn_m1_det_rne16(attn_m1_det_mul(x, z));
        break;
      case ATTN_M1_PROBE_MAX:
        ref = x > z ? x : z;
        break;
      case ATTN_M1_PROBE_EIGHTH:
        ref = attn_m1_det_rne16(attn_m1_det_mul(x, 0.125f));
        break;
      default:
        ref = attn_m1_det_add(0.0f, x);
        break;
      }
      if (std::fabs(ref) > 65504.0f) {
        continue;
      }
      ++n;
      bad +=
        o.op == ATTN_M1_PROBE_MAX ? amc_h2f(y[i]) != ref : y[i] != amc_f2h(ref);
    }
    std::cout << "ATTN_M1_PROBE hf " << o.name << " n=" << n << " bad=" << bad
              << std::endl;
    EXPECT_EQ(bad, 0) << o.name;
  }

  /* exp16 at every fp16 d <= 0: +0, then -0 .. -65504 */
  a.assign(1u, 0u);
  for (uint32_t x = 0x8000u; x < 0xFC00u; ++x) {
    a.push_back((uint16_t)x);
  }
  ASSERT_EQ(probe_sem(handle_, ATTN_M1_PROBE_EXP16, a, a, a, &y), AEE_SUCCESS);
  int bad_exp = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    bad_exp += y[i] != amc_f2h(attn_m1_det_exp16(amc_h2f(a[i])));
  }
  std::cout << "ATTN_M1_PROBE exp16 n=" << a.size() << " bad=" << bad_exp
            << std::endl;
  EXPECT_EQ(bad_exp, 0);

  /* div16 on the hard quotients of l in [1, 2048], e in [0, l] */
  a.clear(), b.clear();
  std::vector<uint16_t> want;
  for (uint32_t lb = 0x3C00u; lb <= 0x6800u; ++lb) {
    const float l = amc_h2f((uint16_t)lb), rc = swiglu_det_recip(l);
    for (uint32_t eb = 0; eb < 0x7C00u; ++eb) {
      const float e = amc_h2f((uint16_t)eb);
      if (e > l) {
        break;
      }
      const float w = attn_m1_det_rne16(attn_m1_det_div(e, l));
      const float c0 = attn_m1_det_rne16(attn_m1_det_mul(e, rc));
      const double q = (double)e / (double)l;
      const float qf = (float)q;
      const bool tie = (double)qf == q && attn_m1_det_rne16(qf) != qf &&
                       (qf >= std::ldexp(1.0f, -14)
                          ? (attn_m1_det_bits(qf) & 0x1FFFu) == 0x1000u
                          : std::fmod(q * 33554432.0, 2.0) == 1.0);
      if (c0 != w || tie) {
        a.push_back((uint16_t)eb);
        b.push_back((uint16_t)lb);
        want.push_back(amc_f2h(w));
      }
    }
  }
  ASSERT_FALSE(a.empty());
  ASSERT_EQ(probe_sem(handle_, ATTN_M1_PROBE_DIV16, a, b, a, &y), AEE_SUCCESS);
  int bad_div = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    bad_div += y[i] != want[i];
  }
  std::cout << "ATTN_M1_PROBE div16 hard n=" << a.size() << " bad=" << bad_div
            << std::endl;
  EXPECT_EQ(bad_div, 0);

  /* A length that is not a multiple of 64 is refused: the entry answers
     AEE_EINVALIDFORMAT, but on silicon the FastRPC layer answers AEE_ERPC
     (0x80000600) before the entry runs (LEDGER rule 38, S1 2026-09-29).
     Either is a refusal; AEE_EBADPARM would be a stale skel. */
  std::vector<uint16_t> s63(63u, 0u);
  const int err = nntr_hvx_attn_m1_probe(
    handle_, ATTN_M1_PROBE_ADD, 1u, 1u, s63.data(), 63, s63.data(), 63,
    s63.data(), 63, s63.data(), 63, nullptr, 0);
  EXPECT_TRUE(err == AEE_EINVALIDFORMAT + kDspOffset || err == (int)0x80000600u)
    << "n=63: " << hex(err);
}

/**
 * @brief The cost side of S1 (plan 170 step 2 cell (b)): each cost op at
 *        1 / 2 / 4 / 6 pool lanes, one warm-up run then `reps` timed runs.
 *        pcyc_per_fma64 is the wall pcycles per 64-lane FMA of all lanes
 *        together (the throughput the cost model needs); lane_pcyc_per_fma64
 *        is the lane-summed pcycles per FMA (#152's 24.6 for today's loop at
 *        6 lanes, 32-lane FMAs counted in pairs); mhz is the pcycles per
 *        qtimer microsecond; gbps the FETCH pair's read rate. Since #170
 *        round 2 the kernel's leads on cold slabs (pv4_cold*, scores1_cold*:
 *        each against pv4 / scores1, the hot-vs-cold ratio per lead) and
 *        scores1_splat (q splatted per scalar load, against scores1).
 *        Printed, not asserted, except that each call succeeds on the lanes
 *        asked for.
 */
TEST_F(HvxAttnM1Probe, Cost) {
  static const struct {
    uint32_t op;
    const char *name;
    uint32_t reps;
  } kOps[] = {{ATTN_M1_PROBE_FMA16_SF, "fma16_sf", 20u},
              {ATTN_M1_PROBE_SCORES1, "scores1", 20u},
              {ATTN_M1_PROBE_SCORES2, "scores2", 20u},
              {ATTN_M1_PROBE_PV4, "pv4", 20u},
              {ATTN_M1_PROBE_FETCH, "fetch", 5u},
              {ATTN_M1_PROBE_FETCH_L2F, "fetch_l2f", 5u},
              {ATTN_M1_PROBE_PV4_COLD, "pv4_cold", 5u},
              {ATTN_M1_PROBE_PV4_COLD_L2F, "pv4_cold_l2f", 5u},
              {ATTN_M1_PROBE_SCORES1_COLD, "scores1_cold", 5u},
              {ATTN_M1_PROBE_SCORES1_COLD_L2F, "scores1_cold_l2f", 5u},
              {ATTN_M1_PROBE_SCORES1_SPLAT, "scores1_splat", 20u}};
  for (const auto &o : kOps) {
    for (uint32_t lanes : {1u, 2u, 4u, 6u}) {
      std::vector<uint32_t> w(ATTN_M1_PROBE_WORDS, 0u);
      int err = AEE_SUCCESS;
      for (uint32_t reps : {1u, o.reps}) { /* warm-up, then timed */
        err = nntr_hvx_attn_m1_probe(handle_, o.op, lanes, reps, nullptr, 0,
                                     nullptr, 0, nullptr, 0, nullptr, 0,
                                     w.data(), (int)w.size());
        ASSERT_EQ(err, AEE_SUCCESS) << o.name << " lanes=" << lanes << ": "
                                    << hex(err) << " (0x8000040e = stale skel)";
      }
      auto w64 = [&](uint32_t i) {
        return (double)w[i] + 4294967296.0 * (double)w[i + 1u];
      };
      const double wall = w64(ATTN_M1_PROBE_W_WALL);
      const double us = w64(ATTN_M1_PROBE_W_QT) / 19.2;
      const double busy = w64(ATTN_M1_PROBE_W_BUSY_SUM);
      const double n_fma = (double)w[ATTN_M1_PROBE_W_FMA64] *
                           w[ATTN_M1_PROBE_W_LANES] * w[ATTN_M1_PROBE_W_REPS];
      const double bytes = (double)w[ATTN_M1_PROBE_W_BYTES] *
                           w[ATTN_M1_PROBE_W_LANES] * w[ATTN_M1_PROBE_W_REPS];
      std::cout << "ATTN_M1_PROBE_COST op=" << o.name << " lanes=" << lanes
                << " ran=" << w[ATTN_M1_PROBE_W_LANES]
                << " reps=" << w[ATTN_M1_PROBE_W_REPS];
      if (n_fma > 0.0) {
        std::cout << " pcyc_per_fma64=" << wall / n_fma
                  << " lane_pcyc_per_fma64=" << busy / n_fma;
      }
      if (bytes > 0.0) {
        std::cout << " gbps=" << bytes / (us * 1e3);
      }
      std::cout << " mhz=" << (us > 0.0 ? wall / us : 0.0) << " us=" << us
                << " busy_max=" << w[ATTN_M1_PROBE_W_BUSY_MAX]
                << " sink=" << w[ATTN_M1_PROBE_W_SINK] << std::endl;
      EXPECT_EQ(w[ATTN_M1_PROBE_W_LANES], lanes) << o.name;
    }
  }
}

int main(int argc, char **argv) {
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
