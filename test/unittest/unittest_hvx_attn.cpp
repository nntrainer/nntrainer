// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file unittest_hvx_attn.cpp
 * @date 07 Aug 2026
 * @brief Device gate for PHASE A (S = Q.Kt) of the fused attention path
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs except for NYI items
 *
 * The oracle is mha_htp_host_scores the same PHASE A mha_htp_host_forward
 * runs, not a second opinion written beside it. Both sides do identical
 * integer arithmetic (u8 activation x iX weight -> int32 -> the same dequant
 * formula), so agreement should be near exact; the tolerance asserted is
 * the task's, and the observed value is reported so the
 * margin is visible rather than assumed. */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

#include "htp_rpc_bench.h"
#include "mha_htp_host_model.h"

namespace {

constexpr int kDspOffset = 0x80000400;

/**
 * @brief Opens an unsigned-PD CDSP session for each test.
 *
 * Same shape as unittest_hvx_softmax.cpp's fixture: a failure here is a hard
 * FAIL rather than a skip, because proving the DSP comes up is part of what
 * this test measures. */
class HtpSession : public ::testing::Test {
protected:
  void SetUp override {
    int err = htp_enable_unsigned_pd;
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);

    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str, &handle_);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "nntr_hvx_open failed: " << hex(err)
      << " is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?";

    std::cout << "ATTN_FIELD path=env field=rpc_poll_qos value="
              << htp_set_latency_qos(handle_) << std::endl;
  }

  void TearDown override {
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
 * one head, alongside the host model's. */
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
   * tells you nothing. */
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

  err = nntr_hvx_attn_kv_append(handle, h_attn, 0u, c.kv_len, k16.data,
                                (int)k16.size(), v16.data, (int)v16.size());
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_kv_append: " << hex(err);

  dev->assign(band, 0.0f);
  err = nntr_hvx_attn_scores_debug(handle, h_attn, head, M, q_band.data,
                                   (int)q_band.size(), dev->data, (int)band);
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_scores_debug: " << hex(err);

  err = nntr_hvx_attn_release(handle, h_attn);
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_release: " << hex(err);

  ref->assign(band, 0.0f);
  mha_htp_host_scores(c.kv_len, c.nch, c.head_dim, c.T, M, head, c.w_k,
                      q_band.data, k16.data, ref->data);
}

/**
 * @brief max|a - b| normalized by max|b|, and @a den_out is that denominator.
 *
 * The denominator is returned, not swallowed, because a zero one makes this
 * metric report a perfect 0.00e+00 for a comparison of two all-zero buffers.
 * The caller asserts it is non-zero, so "matched exactly" cannot be a way of
 * saying "computed nothing". */
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

  std::cout << "\nS band vs host model max|dev - host| / max|host|\n"
            << "w_k,w_v are the Kt and V CACHE widths, in that order. "
               "Activations are u8 throughout,\n"
            << "so I4 means a u8 x i4 matmul nothing here is u4.\n"
            << "shape (kv,nq,gqa,nch,hd,T) w_k,w_v max_rel_err\n";

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
              if (::testing::Test::HasFatalFailure) {
                return;
              }
              double den = 0.0;
              const double e = max_rel(dev, ref, &den);
              /** A case whose reference S is all zeros would report a perfect
               * match while proving nothing. */
              ASSERT_GT(den, 0.0)
                << "reference S is identically zero the comparison is "
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
                  shape.str + " " + wname(c.w_k) + "," + wname(c.w_v);
              }
              if (w == 0 || e > 1e-6) {
                std::cout << std::left << std::setw(30) << shape.str
                          << std::setw(10)
                          << (std::string(wname(c.w_k)) + "," + wname(c.w_v))
                          << std::scientific << std::setprecision(2) << e
                          << "\n";
              }
              EXPECT_LE(e, tol_for(c.w_k)) << shape.str << " (" << wname(c.w_k)
                                           << "," << wname(c.w_v) << ")";
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
 * difference here is a plumbing bug the wrong registry or the wrong ops
 * being handed to the score call not numerical noise, which is why this is
 * bit-exact rather than a tolerance. */
TEST_F(HvxAttnScores, VWidthDoesNotReachS) {
  for (uint32_t kv : {33u, 257u}) {
    for (hexkl_w_width w_k : {HEXKL_W_I8, HEXKL_W_I4}) {
      const AttnCfg a{kv, 33u, 8u, 1u, 128u, 64u, w_k, HEXKL_W_I8};
      const AttnCfg b{kv, 33u, 8u, 1u, 128u, 64u, w_k, HEXKL_W_I4};
      std::vector<float> da, ra, db, rb;

      run_case(handle_, a, 0u, &da, &ra);
      if (::testing::Test::HasFatalFailure) {
        return;
      }
      run_case(handle_, b, 0u, &db, &rb);
      if (::testing::Test::HasFatalFailure) {
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
 * Catches a registry slot that is not really reset stale WH bytes or a
 * handle that outlives its release would show up as a difference on the second
 * pass, and only there. */
TEST_F(HvxAttnScores, LifecycleIsRepeatable) {
  for (hexkl_w_width w : {HEXKL_W_I8, HEXKL_W_I4}) {
    /** kv 257 with T 64 crosses several block boundaries and leaves a partial
     * tail block, which is where a stale registration would survive. */
    const AttnCfg c{257u, 33u, 8u, 1u, 128u, 64u, w, w};
    std::vector<float> first, ref1, second, ref2;

    run_case(handle_, c, 0u, &first, &ref1);
    if (::testing::Test::HasFatalFailure) {
      return;
    }
    run_case(handle_, c, 0u, &second, &ref2);
    if (::testing::Test::HasFatalFailure) {
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
 * FastRPC round trip.
 *
 * This is the stage that replaces compute_kcaches + softmax_triangle +
 * compute_fp16vcache_transposed in mha_core's incremental path, so the oracle
 * is mha_htp_host_forward the same loop, same quantization points, same
 * mask arithmetic, scalar instead of HMX/HVX/DMA. */
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

  std::cout << "\nfused forward vs host model max|dev - host| / max|host|\n"
            << "w_k,w_v are the Kt and V CACHE widths, in that order. "
               "Activations are u8 throughout,\n"
            << "so I4 means a u8 x i4 matmul nothing here is u4.\n"
            << "shape (kv,nq,gqa,nch,hd,T,Mb,win,c,sk) w_k,w_v max_rel_err\n";

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
      err = nntr_hvx_attn_kv_append(handle_, h_attn, 0u, f.kv_len, k16.data,
                                    (int)k16.size(), v16.data, (int)v16.size());
      ASSERT_EQ(err, AEE_SUCCESS) << "attn_kv_append: " << hex(err);

      std::vector<float> dev(q.size(), 0.0f);
      err = nntr_hvx_attn_forward(
        handle_, h_attn, kv_from, f.n_query, scale, (uint32_t)f.causal,
        f.window, f.sink ? sink.data : nullptr, f.sink ? (int)nHq : 0, q.data,
        (int)q.size(), dev.data, (int)dev.size());
      ASSERT_EQ(err, AEE_SUCCESS) << "attn_forward: " << hex(err);
      ASSERT_EQ(nntr_hvx_attn_release(handle_, h_attn), AEE_SUCCESS);

      std::vector<float> ref(q.size(), 0.0f);
      mha_htp_host_forward(f.n_query, kv_from, f.nch, f.gqa, f.head_dim, f.T,
                           f.M_band, f.causal != 0, f.window,
                           f.sink ? sink.data : nullptr, widths[w][0],
                           widths[w][1], q.data, k16.data, v16.data, ref.data);

      double den = 0.0;
      const double e = max_rel(dev, ref, &den);
      ASSERT_GT(den, 0.0) << "reference output is identically zero";
      worst = std::max(worst, e);

      std::ostringstream shape;
      shape << f.kv_len << "," << f.n_query << "," << f.gqa << "," << f.nch
            << "," << f.head_dim << "," << f.T << "," << f.M_band << ","
            << f.window << "," << f.causal << "," << f.sink;
      std::cout << std::left << std::setw(40) << shape.str << std::setw(9)
                << (std::string(wname(widths[w][0])) + "," +
                    wname(widths[w][1]))
                << std::scientific << std::setprecision(2) << e << "\n";

      /** Task 3's fixed tolerances, on the same quantity they were written
       * for. */
      const double tol = (widths[w][0] == HEXKL_W_I8)
                           ? ((widths[w][1] == HEXKL_W_I8) ? 5e-3 : 2e-2)
                           : 5e-2;
      EXPECT_LE(e, tol) << shape.str << " (" << wname(widths[w][0]) << ","
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
 * transport which is the honest figure, and the one 's
 * arithmetic predicts against. Everything the optimisation work already
 * landed is inside it: the DMA ring with cross-block weight prefetch (PHASE A
 * passes every Kt block as one handle list), the session-scoped HMX, the
 * async accumulator copy-out and the vectorised quant/dequant.
 *
 * Rules this follows, each one a bug this project already shipped and found
 *:
 * - the correctness check is NOT in the timed region; a diff left inside one
 * once produced a phantom 146 us slowdown;
 * - three consecutive runs, all three printed, because thermal state moves
 * these numbers;
 * - field=... value=... marker lines with the unit in the field name, after
 * a report script keyed by column position printed a stale number for
 * every shape past the first, twice;
 * - nothing is asserted. The reviewer compares against 's prediction.
 *
 * NOT yet included: the softmax does not run on the worker pool, and P.V does
 * not prefetch across blocks (n_handles=1 per call). Both are Task 10, and
 * both need this breakdown first rather than a guess. */
TEST_F(HvxAttnScores, FusedForwardPerLayerCost) {
  /** Qwen3-0.6B: num_key_value_heads 8, num_attention_heads 16 (gqa 2),
  head_dim 128, 28 layers register_qwen3_0_6b in
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
            << "so I4 means a u8 x i4 matmul nothing here is u4.\n"
            << "10 iterations; avg/min/max are over ALL 10, run 1 included, "
               "which is the\n"
            << "convention the QNN net-run numbers this gets compared against "
               "use. init\n"
            << "is register + kv_append, the one-time setup a graph prepare "
               "corresponds to.\n"
            << "probed is the same work with the DSP breakdown "
               "instrumentation on.\n"
            << "kv_len regime w_k,w_v avg1-10 min max "
               "init probed\n";

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
      /** q and out ride in every timed call 8 KB each way at decode, 1 MB
       * at prefill. RpcBuf for why these two are ION-backed. */
      RpcBuf qbuf(qn * sizeof(float));
      RpcBuf obuf(qn * sizeof(float));
      float *q = (float *)qbuf.p;
      float *out = (float *)obuf.p;
      for (size_t qi = 0; qi < qn; ++qi) {
        q[qi] = d(rng);
      }
      std::memset(out, 0, qn * sizeof(float));
      /** 1 = ION-backed; 0 = the device library exports no rpcmem, or the
       * alloc failed either way these runs used plain heap. */
      std::cout << "ATTN_FIELD path=env field=ion_buffers value="
                << ((qbuf.ion && obuf.ion) ? 1 : 0) << std::endl;
      const float scale = 1.0f / std::sqrt((float)head_dim);

      for (int w = 0; w < 4; ++w) {
        uint32_t h = 0;
        /** init: everything that happens once per layer before a single token
         * is served the handle, and quantizing plus registering every KV
         * block. The analogue of a QNN graph prepare, and reported the same
         * way, separate from the per-inference numbers. */
        const auto init_t0 = std::chrono::steady_clock::now;
        const int rc_reg = nntr_hvx_attn_register(
          handle_, nch, gqa, head_dim, kv_len, T, M_band,
          (uint32_t)widths[w][0], (uint32_t)widths[w][1], &h);
        const int rc_app = (rc_reg == AEE_SUCCESS)
                             ? nntr_hvx_attn_kv_append(
                                 handle_, h, 0u, kv_len, k16.data,
                                 (int)k16.size(), v16.data, (int)v16.size())
                             : rc_reg;
        const auto init_t1 = std::chrono::steady_clock::now;
        ASSERT_EQ(rc_reg, AEE_SUCCESS);
        ASSERT_EQ(rc_app, AEE_SUCCESS);
        const double init_us =
          std::chrono::duration<double, std::micro>(init_t1 - init_t0).count;

        /** The STEP append: the KV rows this inference contributes, which is
         * 1 row at decode and n_query at prefill. init above appended the
         * whole cache at once, which is not what a running model pays it
         * pays this, every token, in every layer, before forward can run.
         * Re-appending rows already present is idempotent (the affected
         * blocks are released and re-registered from the same shadow data)
         * and kv_len does not move, so this measures the real per-step cost
         * without disturbing the forwards below. */
        constexpr int kIters = 10;
        const size_t row_elems = (size_t)nch * head_dim;
        double app[kIters];
        for (int r = 0; r < kIters; ++r) {
          const auto t0 = std::chrono::steady_clock::now;
          const int err = nntr_hvx_attn_kv_append(
            handle_, h, kv_from, n_query, k16.data + kv_from * row_elems,
            (int)((size_t)n_query * row_elems), v16.data + kv_from * row_elems,
            (int)((size_t)n_query * row_elems));
          const auto t1 = std::chrono::steady_clock::now;
          ASSERT_EQ(err, AEE_SUCCESS);
          app[r] = std::chrono::duration<double, std::micro>(t1 - t0).count;
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
         * Ten iterations, statistics over ALL TEN including the first * the
         * convention the QNN numbers this gets compared against are quoted
         * under. Run 1 pays page faults on the freshly registered shadows and
         * so pulls the average up; it is also reported on its own, as us_first,
         * so how much it pulls stays visible. */
        double iter[kIters];
        for (int r = 0; r < kIters; ++r) {
          const auto t0 = std::chrono::steady_clock::now;
          const int err =
            nntr_hvx_attn_forward(handle_, h, kv_from, n_query, scale, 1u, 0u,
                                  nullptr, 0, q, (int)qn, out, (int)qn);
          const auto t1 = std::chrono::steady_clock::now;
          ASSERT_EQ(err, AEE_SUCCESS);
          iter[r] = std::chrono::duration<double, std::micro>(t1 - t0).count;
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
          const auto t0 = std::chrono::steady_clock::now;
          const int err = nntr_hvx_attn_forward_timed(
            handle_, h, kv_from, n_query, scale, 1u, 0u, nullptr, 0, q, (int)qn,
            out, (int)qn, stage.data, (int)stage.size());
          const auto t1 = std::chrono::steady_clock::now;
          /* Checked after the clock stops, deliberately. */
          ASSERT_EQ(err, AEE_SUCCESS);
          us[r] = std::chrono::duration<double, std::micro>(t1 - t0).count;
          stage_of[r] = stage;
        }
        ASSERT_EQ(nntr_hvx_attn_release(handle_, h), AEE_SUCCESS);

        /** MEDIAN of the three, not the min. Wall clock on this device is
         * bimodal at roughly +-25% (same shape, same binary, run to run * DVFS
         * residency, not our code), so the min publishes whichever run happened
         * to land in the fast state. already requires printing all three runs;
         * this only changes which one gets published as the number, and it
         * stays a single measured run rather than a mean so the stage breakdown
         * beside it comes from that same run and still sums to its own
         * dsp_total. */
        int ord[3] = {0, 1, 2};
        std::sort(ord, ord + 3, [&us](int a, int b) { return us[a] < us[b]; });
        const double med = avg;               /* production, uninstrumented */
        const double med_probed = us[ord[1]]; /* instrumented, for the parts */
        const std::vector<uint32_t> &med_stage = stage_of[ord[1]];
        std::ostringstream pair;
        pair << wname(widths[w][0]) << "," << wname(widths[w][1]);
        std::cout << std::left << std::setw(7) << kv_len << std::setw(9)
                  << (n_query == 1 ? "decode" : "prefill") << std::setw(10)
                  << pair.str << std::right << std::fixed
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
          std::cout << "ATTN_STAGE path=" << path.str << " field=" << kStage[st]
                    << " value=" << med_stage[st] << std::endl;
        }
        /** dsp_total came from the INSTRUMENTED run, so subtract it from that
         * run's wall mixing it with the production wall would fold the
         * instrumentation into the transport estimate. */
        std::cout << "ATTN_STAGE path=" << path.str
                  << " field=transport_us value="
                  << (med_probed - (double)med_stage[5]) << std::endl;
        std::cout << "ATTN_STAGE path=" << path.str
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
          std::cout << "ATTN_FIELD path=" << path.str << " field=" << kRun[f]
                    << " value=" << run_v[f] << std::endl;
        }
      }
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
    result = RUN_ALL_TESTS;
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS" << std::endl;
  }
  return result;
}
