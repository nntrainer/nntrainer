// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_hvx_attn.cpp
 * @date   07 Aug 2026
 * @brief  Device checks for DSP-resident M=1 attention and its probes
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
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

} // namespace

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
            << " exp=" << word(ATTN_M1_PROF_EXP)
            << " et=" << word(ATTN_M1_PROF_ET)
            << " max=" << word(ATTN_M1_PROF_MAX)
            << " sum=" << word(ATTN_M1_PROF_SUM)
            << " div=" << word(ATTN_M1_PROF_DIV)
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
    /** One round of 6 to put every layer's slab behind the other five, then
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

  /** A length that is not a multiple of 64 is refused: the entry answers
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
