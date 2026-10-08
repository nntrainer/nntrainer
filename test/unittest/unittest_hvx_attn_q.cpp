// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   unittest_hvx_attn_q.cpp
 * @date   23 Sep 2026
 * @brief  Device test: quantized (A8W8 / A8W4) KV cache and attention on HMX
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Runs on an Android device only. Requires libnntr_hvx_skel.so on
 * ADSP_LIBRARY_PATH. See test/htp/build.sh.
 *
 * Phase Q1 (docs/backend_guide/htp_backend/21_quantized_attention_plan.md):
 * the DSP quantizes appended rows exactly as the same C does on the ARM
 * side, and the WH tiles it bakes with HexKL multiply on HMX to the
 * integers a plain int matmul over those masters gives -- through the
 * probed int32 accumulator layout.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "hexkl_attn_q_plan.h"
#include "hexkl_kv_q.h"
#include "hvx_attn_q_model.h"
#include "hvx_attn_test_util.h"
#include "hvx_softmax_q.h"
#include "nntr_hvx.h"

namespace {

using namespace hvx_test;

class HvxAttnQ : public hvx_test::SessionTest {
protected:
  /**
   * @brief Registers a cache of @a kind on both sides, appends the same
   *        rows to both (two chunks plus a rewrite), and returns the DSP
   *        handle. The host table is the reference the dump is compared to.
   */
  void FillBoth(uint32_t kind, uint32_t rows, uint32_t n_kv, uint32_t hd,
                hexkl_kv_q_table &host, uint32_t &host_h, uint32_t &dsp_h,
                bool fixed = false) {
    const uint32_t width = n_kv * hd;
    std::vector<uint16_t> k(static_cast<size_t>(rows) * width), v(k.size());
    fill_hf(k, 0x0A8Bu + kind, 1.0f);
    fill_hf(v, 0x0A8Cu + kind, 1.0f);
    std::vector<uint16_t> k2(static_cast<size_t>(3) * width), v2(k2.size());
    fill_hf(k2, 0x0A8Du, 1.0f);
    fill_hf(v2, 0x0A8Eu, 1.0f);

    ASSERT_EQ(hexkl_kv_q_register(&host, static_cast<hexkl_kv_q_kind>(kind),
                                  rows, n_kv, hd, &host_h),
              0);
    int err = nntr_hvx_kv_register_q(handle_, kind, rows, n_kv, hd, &dsp_h);
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_register_q failed: " << hex(err);
    if (fixed) {
      // Scales from the data's own maxima, per head for K and per (head,
      // dim) for V, slightly tightened so the clamp is exercised.
      std::vector<float> s_k(n_kv, 0.0f),
        s_v(static_cast<size_t>(n_kv) * hd, 0.0f);
      for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t n = 0; n < n_kv; ++n) {
          for (uint32_t d = 0; d < hd; ++d) {
            const size_t at = static_cast<size_t>(r) * width + n * hd + d;
            s_k[n] = std::max(s_k[n], std::fabs(hf_to_f32(k[at])));
            s_v[n * hd + d] =
              std::max(s_v[n * hd + d], std::fabs(hf_to_f32(v[at])));
          }
        }
      }
      for (auto &x : s_k) {
        x = x * 0.9f / 127.0f;
      }
      for (auto &x : s_v) {
        x = std::max(x, 1e-3f) * 0.9f / 127.0f;
      }
      ASSERT_EQ(
        hexkl_kv_q_set_fixed_scales(&host, host_h, s_k.data(), s_v.data()), 0);
      err = nntr_hvx_kv_set_fixed_scales_q(
        handle_, dsp_h, s_k.data(), static_cast<int>(s_k.size()), s_v.data(),
        static_cast<int>(s_v.size()));
      ASSERT_EQ(err, AEE_SUCCESS)
        << "kv_set_fixed_scales_q failed: " << hex(err);
    }

    const uint32_t split = rows / 2;
    auto append = [&](uint32_t row0, uint32_t n, const uint16_t *kr,
                      const uint16_t *vr) {
      ASSERT_EQ(
        hexkl_kv_q_append(&host, host_h, row0, n, kr, vr, nullptr, nullptr), 0);
      const int e = nntr_hvx_kv_append_q(handle_, dsp_h, row0, kr,
                                         static_cast<int>(n * width), vr,
                                         static_cast<int>(n * width));
      ASSERT_EQ(e, AEE_SUCCESS) << "kv_append_q failed: " << hex(e);
    };
    append(0, split, k.data(), v.data());
    append(split, rows - split, k.data() + static_cast<size_t>(split) * width,
           v.data() + static_cast<size_t>(split) * width);
    append(10, 3, k2.data(), v2.data());
  }

  /** @brief kv_dump_q of all rows, compared bit for bit with the host. */
  void DumpMatches(const hexkl_kv_q_table &host, uint32_t host_h,
                   uint32_t dsp_h, std::vector<int8_t> &kq,
                   std::vector<int8_t> &vq) {
    const hexkl_kv_q *kv = hexkl_kv_q_get(&host, host_h);
    ASSERT_NE(kv, nullptr);
    const uint32_t rows = kv->max_rows, n_kv = kv->n_head_kv, hd = kv->head_dim;
    const size_t values = static_cast<size_t>(rows) * n_kv * hd;
    // One K scale, one colsum and one V scale per (row, head).
    const size_t heads = static_cast<size_t>(rows) * n_kv;
    kq.assign(values, 0);
    vq.assign(values, 0);
    std::vector<float> sk(heads), sv(heads);
    std::vector<int32_t> cs(heads);
    int err = nntr_hvx_kv_dump_q(
      handle_, dsp_h, 0, rows, kq.data(), static_cast<int>(values), vq.data(),
      static_cast<int>(values), sk.data(), static_cast<int>(heads), cs.data(),
      static_cast<int>(heads), sv.data(), static_cast<int>(heads));
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_dump_q failed: " << hex(err);

    std::vector<int8_t> hkq(values), hvq(values);
    std::vector<float> hsk(heads), hsv(heads);
    std::vector<int32_t> hcs(heads);
    ASSERT_EQ(hexkl_kv_q_dump(kv, 0, rows, hkq.data(), hvq.data(), hsk.data(),
                              hcs.data(), hsv.data()),
              0);
    // The DSP quantizes on HVX with the same scales; a value may differ by
    // one step where the f32 product lands on a rounding tie the scalar
    // path resolves differently. Scales are bit-identical, and colsum has
    // to be the sum of the values the DSP actually stored.
    size_t k_diff = 0, v_diff = 0, k_far = 0, v_far = 0;
    for (size_t i = 0; i < values; ++i) {
      const int dk = std::abs(kq[i] - hkq[i]), dv = std::abs(vq[i] - hvq[i]);
      k_diff += dk != 0;
      v_diff += dv != 0;
      k_far += dk > 1;
      v_far += dv > 1;
    }
    EXPECT_EQ(k_far, 0u) << "K values off by more than one step";
    EXPECT_EQ(v_far, 0u) << "V values off by more than one step";
    EXPECT_LE(k_diff, values / 500) << "K one-step differences: " << k_diff;
    EXPECT_LE(v_diff, values / 500) << "V one-step differences: " << v_diff;
    EXPECT_EQ(std::memcmp(sk.data(), hsk.data(), heads * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(sv.data(), hsv.data(), heads * sizeof(float)), 0);
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t n = 0; n < n_kv; ++n) {
        int32_t sum = 0;
        for (uint32_t d = 0; d < hd; ++d) {
          sum += kq[static_cast<size_t>(r) * n_kv * hd + n * hd + d];
        }
        ASSERT_EQ(cs[r * n_kv + n], sum) << "colsum row " << r << " head " << n;
      }
    }
  }

  /**
   * @brief HMX over the baked tiles of (head n, column tile c) against a
   *        plain int matmul over the dumped masters. Exact.
   */
  void TileMatmulMatches(uint32_t dsp_h, const hexkl_kv_q *kv,
                         const std::vector<int8_t> &kq,
                         const std::vector<int8_t> &vq, uint32_t n,
                         uint32_t c) {
    SCOPED_TRACE("head " + std::to_string(n) + " col tile " +
                 std::to_string(c));
    const uint32_t hd = kv->head_dim, width = kv->n_head_kv * hd;
    std::vector<uint8_t> act_s(static_cast<size_t>(64) * hd), act_p(64 * 32);
    uint32_t s = 0xACC00001u + n * 7u + c;
    for (auto &x : act_s) {
      s = s * 1664525u + 1013904223u;
      x = static_cast<uint8_t>(s >> 24);
    }
    for (auto &x : act_p) {
      s = s * 1664525u + 1013904223u;
      x = static_cast<uint8_t>(s >> 24);
    }
    std::vector<int32_t> got_s(64 * 32), got_o(static_cast<size_t>(64) * hd);
    const int err = nntr_hvx_probe_kv_q_mm(
      handle_, dsp_h, n, c, act_s.data(), static_cast<int>(act_s.size()),
      act_p.data(), static_cast<int>(act_p.size()), got_s.data(),
      static_cast<int>(got_s.size()), got_o.data(),
      static_cast<int>(got_o.size()));
    ASSERT_EQ(err, AEE_SUCCESS) << "probe_kv_q_mm failed: " << hex(err);

    size_t s_bad = 0, o_bad = 0;
    for (uint32_t r = 0; r < 64; ++r) {
      for (uint32_t k = 0; k < 32; ++k) {
        int32_t acc = 0;
        const int8_t *krow =
          &kq[static_cast<size_t>(32 * c + k) * width + n * hd];
        for (uint32_t d = 0; d < hd; ++d) {
          acc += static_cast<int32_t>(act_s[static_cast<size_t>(r) * hd + d]) *
                 krow[d];
        }
        s_bad += got_s[r * 32 + k] != acc;
      }
      for (uint32_t d = 0; d < hd; ++d) {
        int32_t acc = 0;
        for (uint32_t k = 0; k < 32; ++k) {
          acc += static_cast<int32_t>(act_p[r * 32 + k]) *
                 vq[static_cast<size_t>(32 * c + k) * width + n * hd + d];
        }
        o_bad += got_o[static_cast<size_t>(r) * hd + d] != acc;
      }
    }
    EXPECT_EQ(s_bad, 0u) << "Q.K^T tile: HMX over baked K^T tiles != masters";
    EXPECT_EQ(o_bad, 0u) << "P.V tile: HMX over baked V tiles != masters";
  }

  void RunKind(uint32_t kind, uint32_t rows, uint32_t n_kv, uint32_t hd,
               bool fixed = false) {
    SCOPED_TRACE("kind " + std::to_string(kind) + " rows " +
                 std::to_string(rows) + " n_kv " + std::to_string(n_kv) +
                 " hd " + std::to_string(hd) + (fixed ? " fixed" : ""));
    hexkl_kv_q_table host{};
    uint32_t host_h = 0, dsp_h = 0;
    FillBoth(kind, rows, n_kv, hd, host, host_h, dsp_h, fixed);
    if (::testing::Test::HasFatalFailure()) {
      return;
    }
    std::vector<int8_t> kq, vq;
    DumpMatches(host, host_h, dsp_h, kq, vq);
    const hexkl_kv_q *kv = hexkl_kv_q_get(&host, host_h);
    // A full column tile, and the partial last one (zeros past the rows).
    TileMatmulMatches(dsp_h, kv, kq, vq, n_kv - 1, 1);
    TileMatmulMatches(dsp_h, kv, kq, vq, 0, (rows - 1) / 32);
    EXPECT_EQ(nntr_hvx_kv_release_q(handle_, dsp_h), AEE_SUCCESS);
    hexkl_kv_q_release(&host, host_h);
  }
};

constexpr int kQStatCount = 12;
const char *const kQStatNames[kQStatCount] = {
  "qprep", "dma",  "qk",    "dequant",  "softmax",  "pquant",
  "pv",    "oupd", "store", "n_blocks", "us_total", "kcycles"};

/**
 * @brief Attention over a quantized cache against the f32 reference over
 *        the exact fp16 rows the cache was built from, so the SNR includes
 *        the cache's own quantization error.
 */
class HvxAttnQPrefill : public hvx_test::SessionTest {
protected:
  void RunShape(uint32_t kind, const AttnShape &s, double min_snr_db,
                bool report = false) {
    SCOPED_TRACE(
      "kind=" + std::to_string(kind) + " n_q=" + std::to_string(s.n_q) +
      " from=" + std::to_string(s.cache_from) + " to=" +
      std::to_string(s.cache_to) + " hq=" + std::to_string(s.n_head_q) +
      " hkv=" + std::to_string(s.n_head_kv) +
      " hd=" + std::to_string(s.head_dim) + " win=" + std::to_string(s.window) +
      " cap=" + std::to_string(s.softcap) +
      " sink=" + std::to_string(s.use_sink));
    const size_t q_elems = static_cast<size_t>(s.n_q) * s.n_head_q * s.head_dim;
    const uint32_t width = s.n_head_kv * s.head_dim;
    const size_t kv_elems = static_cast<size_t>(s.cache_to) * width;
    std::vector<float> q(q_elems);
    fill_deterministic(q, 0xA77E0001u, 3.0f);
    std::vector<uint16_t> k(kv_elems), v(kv_elems);
    fill_hf(k, 0xA77E0002u, 1.0f);
    fill_hf(v, 0xA77E0003u, 1.0f);
    std::vector<float> sinks;
    if (s.use_sink) {
      sinks.resize(s.n_head_q);
      fill_deterministic(sinks, 0xA77E0004u, 2.0f);
    }
    std::vector<float> want;
    ref_attention(s, q, k, v, sinks, want);

    uint32_t h = 0;
    int err = nntr_hvx_kv_register_q(handle_, kind, s.cache_to, s.n_head_kv,
                                     s.head_dim, &h);
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_register_q failed: " << hex(err);
    // Appended in two chunks, as a prefill step then a follow-on would.
    const uint32_t split = s.cache_to / 2;
    err = nntr_hvx_kv_append_q(handle_, h, 0, k.data(),
                               static_cast<int>(split * width), v.data(),
                               static_cast<int>(split * width));
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_append_q failed: " << hex(err);
    err = nntr_hvx_kv_append_q(handle_, h, split,
                               k.data() + static_cast<size_t>(split) * width,
                               static_cast<int>((s.cache_to - split) * width),
                               v.data() + static_cast<size_t>(split) * width,
                               static_cast<int>((s.cache_to - split) * width));
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_append_q failed: " << hex(err);

    std::vector<float> got(q_elems, 0.0f);
    std::vector<uint32_t> stats(kQStatCount, 0);
    err = nntr_hvx_attn_q_prefill(
      handle_, h, s.n_q, s.cache_from, s.cache_to, s.n_head_q, s.window, s.br,
      s.bc, s.softcap, q.data(), static_cast<int>(q.size()), sinks.data(),
      static_cast<int>(sinks.size()), got.data(), static_cast<int>(got.size()),
      stats.data(), kQStatCount);
    EXPECT_EQ(nntr_hvx_kv_release_q(handle_, h), AEE_SUCCESS);
    ASSERT_EQ(err, AEE_SUCCESS) << "attn_q_prefill failed: " << hex(err);

    for (size_t i = 0; i < got.size(); ++i) {
      ASSERT_TRUE(std::isfinite(got[i])) << "non-finite output at " << i;
    }
    // Three numbers: DSP vs exact (the gate), the CPU int model vs exact
    // (what the quantization alone costs), DSP vs model (kernel error).
    hexkl_attn_f16_tiling tl = {s.br, s.bc, 0, 0};
    hexkl_attn_f16_shape sh = {s.n_q,      s.cache_from, s.cache_to,
                               s.n_head_q, s.n_head_kv,  s.head_dim,
                               s.window,   s.softcap};
    if (s.br == 0 || s.bc == 0) {
      hexkl_attn_f16_choose_tiling(&sh, &tl);
    }
    std::vector<float> model;
    model_attention_q(knobs_for_kind(kind), s, q, k, v, sinks, tl.bc, model);
    const double snr = snr_db(want, got);
    const double snr_model = snr_db(want, model);
    const double snr_vs_model = snr_db(model, got);
    // The kernel is gated on reproducing the model (the quantization noise
    // is the scheme's, not the kernel's); the floor only catches a scheme
    // change that would make the numbers meaningless.
    EXPECT_GT(snr_vs_model, 45.0) << "DSP vs model " << snr_vs_model << " dB";
    EXPECT_GT(snr, snr_model - 1.0)
      << "SNR " << snr << " dB vs model " << snr_model << " dB";
    EXPECT_GT(snr, min_snr_db) << "SNR " << snr << " dB below the floor";
    std::cout << "ATTN_Q_FIELD kind=" << kind << " shape=" << s.n_q << "x"
              << s.cache_to << "x" << s.n_head_q << "/" << s.n_head_kv << "x"
              << s.head_dim << " bc=" << tl.bc << " field=snr_db value=" << snr
              << " model=" << snr_model << " dsp_vs_model=" << snr_vs_model
              << "\n";
    if (report) {
      for (int i = 0; i < kQStatCount; ++i) {
        std::cout << "ATTN_Q_FIELD kind=" << kind << " shape=" << s.n_q << "x"
                  << s.cache_to << "x" << s.n_head_q << "/" << s.n_head_kv
                  << "x" << s.head_dim << " field=" << kQStatNames[i]
                  << " value=" << stats[i] << "\n";
      }
      if (stats[10] > 0) {
        std::cout << "ATTN_Q_FIELD kind=" << kind << " shape=" << s.n_q << "x"
                  << s.cache_to << "x" << s.n_head_q << "/" << s.n_head_kv
                  << "x" << s.head_dim << " field=clock_mhz value="
                  << static_cast<double>(stats[11]) * 1000.0 / stats[10]
                  << "\n";
      }
    }
  }
};

constexpr int kQ2StatCount = 12;
const char *const kQ2StatNames[kQ2StatCount] = {
  "qprep",    "dma",      "qk",      "softmax",  "pv",         "epilogue",
  "us_total", "n_blocks", "kcycles", "hmx_wait", "head_setup", "submit"};

/**
 * @brief The row-blocked kernel (plan 23, R4) over a fixed-scale cache
 *        against the f32 reference over the exact fp16 rows. The cache's
 *        scales come from the data's maxima, per head for K and per (head,
 *        dim) for V, as the quantized model's encodings would supply them.
 */
class HvxAttnQ2 : public hvx_test::SessionTest {
protected:
  void RunShape(const AttnShape &s, double min_snr_db, bool report = false) {
    SCOPED_TRACE(
      "n_q=" + std::to_string(s.n_q) + " from=" + std::to_string(s.cache_from) +
      " to=" + std::to_string(s.cache_to) + " hq=" +
      std::to_string(s.n_head_q) + " hkv=" + std::to_string(s.n_head_kv) +
      " hd=" + std::to_string(s.head_dim) + " win=" + std::to_string(s.window));
    const size_t q_elems = static_cast<size_t>(s.n_q) * s.n_head_q * s.head_dim;
    const uint32_t width = s.n_head_kv * s.head_dim;
    const size_t kv_elems = static_cast<size_t>(s.cache_to) * width;
    std::vector<float> q(q_elems);
    fill_deterministic(q, 0xA77E0001u, 3.0f);
    std::vector<uint16_t> k(kv_elems), v(kv_elems);
    fill_hf(k, 0xA77E0002u, 1.0f);
    fill_hf(v, 0xA77E0003u, 1.0f);
    std::vector<float> sinks;
    std::vector<float> want;
    ref_attention(s, q, k, v, sinks, want);

    std::vector<float> s_k(s.n_head_kv, 0.0f), s_v(width, 0.0f);
    for (uint32_t r = 0; r < s.cache_to; ++r) {
      for (uint32_t n = 0; n < s.n_head_kv; ++n) {
        for (uint32_t d = 0; d < s.head_dim; ++d) {
          const size_t at = static_cast<size_t>(r) * width + n * s.head_dim + d;
          s_k[n] = std::max(s_k[n], std::fabs(hf_to_f32(k[at])));
          s_v[n * s.head_dim + d] =
            std::max(s_v[n * s.head_dim + d], std::fabs(hf_to_f32(v[at])));
        }
      }
    }
    for (auto &x : s_k) {
      x = x / 127.0f;
    }
    for (auto &x : s_v) {
      x = std::max(x, 1e-6f) / 127.0f;
    }

    uint32_t h = 0;
    int err = nntr_hvx_kv_register_q(handle_, 0, s.cache_to, s.n_head_kv,
                                     s.head_dim, &h);
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_register_q failed: " << hex(err);
    err = nntr_hvx_kv_set_fixed_scales_q(
      handle_, h, s_k.data(), static_cast<int>(s_k.size()), s_v.data(),
      static_cast<int>(s_v.size()));
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_set_fixed_scales_q failed: " << hex(err);
    const uint32_t split = s.cache_to / 2;
    err = nntr_hvx_kv_append_q(handle_, h, 0, k.data(),
                               static_cast<int>(split * width), v.data(),
                               static_cast<int>(split * width));
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_append_q failed: " << hex(err);
    err = nntr_hvx_kv_append_q(handle_, h, split,
                               k.data() + static_cast<size_t>(split) * width,
                               static_cast<int>((s.cache_to - split) * width),
                               v.data() + static_cast<size_t>(split) * width,
                               static_cast<int>((s.cache_to - split) * width));
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_append_q failed: " << hex(err);

    // Per-head Q scales as a quantized model's encodings would give them.
    std::vector<float> q_scale(s.n_head_q, 0.0f);
    for (uint32_t r = 0; r < s.n_q; ++r) {
      for (uint32_t hh = 0; hh < s.n_head_q; ++hh) {
        for (uint32_t d = 0; d < s.head_dim; ++d) {
          q_scale[hh] = std::max(
            q_scale[hh],
            std::fabs(
              q[(static_cast<size_t>(r) * s.n_head_q + hh) * s.head_dim + d]));
        }
      }
    }
    for (auto &x : q_scale) {
      x = x > 0.0f ? x / 127.0f : 1.0f;
    }
    std::vector<float> got(q_elems, 0.0f);
    std::vector<uint32_t> stats(kQ2StatCount, 0);
    err = nntr_hvx_attn_q2_prefill(
      handle_, h, s.n_q, s.cache_from, s.cache_to, s.n_head_q, s.window,
      q.data(), static_cast<int>(q.size()), q_scale.data(),
      static_cast<int>(q_scale.size()), got.data(),
      static_cast<int>(got.size()), stats.data(), kQ2StatCount);
    EXPECT_EQ(nntr_hvx_kv_release_q(handle_, h), AEE_SUCCESS);
    ASSERT_EQ(err, AEE_SUCCESS) << "attn_q2_prefill failed: " << hex(err);
    for (size_t i = 0; i < got.size(); ++i) {
      ASSERT_TRUE(std::isfinite(got[i])) << "non-finite output at " << i;
    }
    const double snr = snr_db(want, got);
    std::cout << "ATTN_Q2_FIELD shape=" << s.n_q << "x" << s.cache_to << "x"
              << s.n_head_q << "/" << s.n_head_kv << "x" << s.head_dim
              << " win=" << s.window << " field=snr_db value=" << snr << "\n";
    EXPECT_GT(snr, min_snr_db) << "SNR " << snr << " dB below the floor";
    if (report) {
      for (int i = 0; i < kQ2StatCount; ++i) {
        std::cout << "ATTN_Q2_FIELD shape=" << s.n_q << "x" << s.cache_to << "x"
                  << s.n_head_q << "/" << s.n_head_kv << "x" << s.head_dim
                  << " win=" << s.window << " field=" << kQ2StatNames[i]
                  << " value=" << stats[i] << "\n";
      }
    }
  }
};

/** @brief a8 P on a fixed 1/256 grid gives ~33 dB on synthetic data (the
 *         host softmax test); the int8 K/V/Q terms sit above that. */
constexpr double kSnrQ2 = 28.0;

TEST_F(HvxAttnQ2, SmallCausal) {
  RunShape({128, 0, 128, 4, 2, 128, 0, 0, 0}, kSnrQ2);
}

TEST_F(HvxAttnQ2, ChunkedWithHistory) {
  // Rows 768..1023 over a 1024-row cache: the second chunk of a prefill.
  RunShape({256, 768, 1024, 8, 4, 128, 0, 0, 0}, kSnrQ2);
}

TEST_F(HvxAttnQ2, WindowedGemmaSliding1024) {
  RunShape({1024, 0, 1024, 16, 8, 256, 1024, 0, 0}, kSnrQ2, true);
}

TEST_F(HvxAttnQ2, PaddedLastBlock) {
  // 100 rows: the second block has 36 valid rows.
  RunShape({100, 200, 300, 4, 2, 64, 0, 0, 0}, kSnrQ2);
}

TEST_F(HvxAttnQ2, ReportSliding4096Chunk) {
  // The last 1024-row chunk of a 4096 prefill, sliding layer shape.
  RunShape({1024, 3072, 4096, 16, 8, 256, 1024, 0, 0}, kSnrQ2, true);
}

TEST_F(HvxAttnQ2, ReportGlobal4096Chunk) {
  // The last 512-row chunk over the full 4096 cache, global layer shape.
  RunShape({512, 3584, 4096, 16, 2, 512, 0, 0, 0}, kSnrQ2, true);
}

/**
 * @brief Absolute floors, from the CPU model on this test's uniform random
 *        data: every int8 term sits at its theoretical 48 dB and the u8 P
 *        at 41-44 dB, summing to 39-44 dB for A8W8; any int4 term is 23 dB
 *        on uniform data whatever its scale granularity, so A8W4 lands at
 *        20 dB here. The gate that matters is agreement with the model.
 */
constexpr double kSnrQ8 = 36.0;
constexpr double kSnrQ4 = 18.0;

TEST_F(HvxAttnQPrefill, Int8Basic) {
  RunShape(0, {64, 0, 64, 1, 1, 64, 0, 16, 32}, kSnrQ8);
  RunShape(0, {128, 896, 1024, 16, 4, 128, 0, 16, 128}, kSnrQ8);
  RunShape(0, {100, 0, 100, 8, 8, 64, 0, 8, 32}, kSnrQ8);
}

TEST_F(HvxAttnQPrefill, Int8Options) {
  // GQA with a q block straddling cache blocks, window, softcap, sink.
  RunShape(0, {70, 30, 100, 8, 2, 64, 0, 16, 32}, kSnrQ8);
  RunShape(0, {128, 896, 1024, 16, 4, 128, 256, 16, 64}, kSnrQ8);
  RunShape(0, {64, 448, 512, 8, 4, 128, 0, 16, 64, 30.0f}, kSnrQ8);
  RunShape(0, {64, 448, 512, 8, 4, 128, 0, 16, 64, 0.0f, true}, kSnrQ8);
  // hd 32 and 256 (one and eight groups).
  RunShape(0, {40, 0, 40, 2, 2, 32, 0, 32, 32}, kSnrQ8);
  RunShape(0, {32, 96, 128, 4, 2, 256, 0, 16, 32}, kSnrQ8);
}

TEST_F(HvxAttnQPrefill, Int8AutoTiling) {
  RunShape(0, {128, 896, 1024, 16, 4, 128, 0, 0, 0}, kSnrQ8);
  RunShape(0, {100, 0, 100, 8, 8, 64, 0, 0, 0}, kSnrQ8);
}

TEST_F(HvxAttnQPrefill, Int4Basic) {
  RunShape(1, {64, 0, 64, 1, 1, 64, 0, 16, 32}, kSnrQ4);
  RunShape(1, {128, 896, 1024, 16, 4, 128, 0, 16, 128}, kSnrQ4);
  RunShape(1, {70, 30, 100, 8, 2, 64, 0, 16, 32}, kSnrQ4);
  RunShape(1, {64, 448, 512, 8, 4, 128, 0, 16, 64, 30.0f, true}, kSnrQ4);
}

TEST_F(HvxAttnQPrefill, ReportPhaseTimes) {
  RunShape(0, {128, 896, 1024, 16, 4, 128, 0, 16, 128}, kSnrQ8, true);
  RunShape(1, {128, 896, 1024, 16, 4, 128, 0, 16, 128}, kSnrQ4, true);
  RunShape(0, {32, 4064, 4096, 16, 4, 128, 0, 16, 256}, kSnrQ8, true);
}

/**
 * @brief The decode kernel against the exact reference and the CPU model
 *        with its own arithmetic: 32-row P' blocks, f32 scores.
 */
class HvxAttnQDecode : public hvx_test::SessionTest {
protected:
  void RunDecode(uint32_t kind, const AttnShape &s, double floor_db,
                 bool report = false) {
    SCOPED_TRACE(
      "decode kind=" + std::to_string(kind) + " n_q=" + std::to_string(s.n_q) +
      " from=" + std::to_string(s.cache_from) + " to=" +
      std::to_string(s.cache_to) + " hq=" + std::to_string(s.n_head_q) +
      " hkv=" + std::to_string(s.n_head_kv) +
      " hd=" + std::to_string(s.head_dim) + " win=" + std::to_string(s.window) +
      " cap=" + std::to_string(s.softcap) +
      " sink=" + std::to_string(s.use_sink));
    const size_t q_elems = static_cast<size_t>(s.n_q) * s.n_head_q * s.head_dim;
    const uint32_t width = s.n_head_kv * s.head_dim;
    const size_t kv_elems = static_cast<size_t>(s.cache_to) * width;
    std::vector<float> q(q_elems);
    fill_deterministic(q, 0xDEC0DE01u, 3.0f);
    std::vector<uint16_t> k(kv_elems), v(kv_elems);
    fill_hf(k, 0xDEC0DE02u, 1.0f);
    fill_hf(v, 0xDEC0DE03u, 1.0f);
    std::vector<float> sinks;
    if (s.use_sink) {
      sinks.resize(s.n_head_q);
      fill_deterministic(sinks, 0xDEC0DE04u, 2.0f);
    }
    std::vector<float> want;
    ref_attention(s, q, k, v, sinks, want);

    uint32_t h = 0;
    int err = nntr_hvx_kv_register_q(handle_, kind, s.cache_to, s.n_head_kv,
                                     s.head_dim, &h);
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_register_q failed: " << hex(err);
    err =
      nntr_hvx_kv_append_q(handle_, h, 0, k.data(), static_cast<int>(kv_elems),
                           v.data(), static_cast<int>(kv_elems));
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_append_q failed: " << hex(err);

    std::vector<float> got(q_elems, 0.0f);
    std::vector<uint32_t> stats(2, 0);
    err = nntr_hvx_attn_q_decode(handle_, h, s.n_q, s.cache_from, s.cache_to,
                                 s.n_head_q, s.window, s.softcap, q.data(),
                                 static_cast<int>(q.size()), sinks.data(),
                                 static_cast<int>(sinks.size()), got.data(),
                                 static_cast<int>(got.size()), stats.data(), 2);
    EXPECT_EQ(nntr_hvx_kv_release_q(handle_, h), AEE_SUCCESS);
    ASSERT_EQ(err, AEE_SUCCESS) << "attn_q_decode failed: " << hex(err);
    for (size_t i = 0; i < got.size(); ++i) {
      ASSERT_TRUE(std::isfinite(got[i])) << "non-finite output at " << i;
    }

    QModelKnobs kn = knobs_for_kind(kind);
    kn.s_hf = false;
    std::vector<float> model;
    model_attention_q(kn, s, q, k, v, sinks, 32, model);
    const double snr = snr_db(want, got);
    const double snr_model = snr_db(want, model);
    const double snr_vs_model = snr_db(model, got);
    EXPECT_GT(snr_vs_model, 45.0) << "DSP vs model " << snr_vs_model << " dB";
    EXPECT_GT(snr, snr_model - 1.0)
      << "SNR " << snr << " dB vs model " << snr_model << " dB";
    EXPECT_GT(snr, floor_db) << "SNR " << snr << " dB below the floor";
    std::cout << "ATTN_Q_FIELD path=decode kind=" << kind << " shape=" << s.n_q
              << "x" << s.cache_to << "x" << s.n_head_q << "/" << s.n_head_kv
              << "x" << s.head_dim << " field=snr_db value=" << snr
              << " model=" << snr_model << " dsp_vs_model=" << snr_vs_model
              << "\n";
    if (report) {
      std::cout << "ATTN_Q_FIELD path=decode kind=" << kind
                << " shape=" << s.n_q << "x" << s.cache_to << "x" << s.n_head_q
                << "/" << s.n_head_kv << "x" << s.head_dim
                << " field=total_us value=" << stats[0] << "\n";
    }
  }
};

TEST_F(HvxAttnQDecode, Int8SingleTokenGqa) {
  RunDecode(0, {1, 511, 512, 16, 4, 128, 0, 0, 0}, kSnrQ8);
  // Ragged: 500 rows is not a block multiple.
  RunDecode(0, {1, 499, 500, 16, 4, 128, 0, 0, 0}, kSnrQ8);
}

TEST_F(HvxAttnQDecode, Int8FewTokensHeadDim32To96) {
  RunDecode(0, {4, 300, 304, 8, 8, 64, 0, 0, 0}, kSnrQ8);
  RunDecode(0, {3, 100, 103, 8, 2, 96, 0, 0, 0}, kSnrQ8);
  RunDecode(0, {2, 60, 62, 4, 2, 32, 0, 0, 0}, kSnrQ8);
}

TEST_F(HvxAttnQDecode, Int8WindowSinkSoftcap) {
  AttnShape s{2, 700, 702, 8, 4, 128, 256, 0, 0};
  RunDecode(0, s, kSnrQ8);
  s.use_sink = true;
  RunDecode(0, s, kSnrQ8);
  s.softcap = 30.0f;
  RunDecode(0, s, kSnrQ8);
}

TEST_F(HvxAttnQDecode, Int4) {
  RunDecode(1, {1, 511, 512, 16, 4, 128, 0, 0, 0}, kSnrQ4);
  RunDecode(1, {3, 100, 103, 8, 2, 96, 0, 0, 0}, kSnrQ4);
  AttnShape s{2, 700, 702, 8, 4, 128, 256, 0, 0, 30.0f, true};
  RunDecode(1, s, kSnrQ4);
}

TEST_F(HvxAttnQDecode, ReportTimes) {
  RunDecode(0, {1, 1023, 1024, 16, 4, 128, 0, 0, 0}, kSnrQ8, true);
  RunDecode(0, {1, 4095, 4096, 16, 4, 128, 0, 0, 0}, kSnrQ8, true);
  RunDecode(1, {1, 4095, 4096, 16, 4, 128, 0, 0, 0}, kSnrQ4, true);
}

/**
 * @brief attn_q_step as the model uses it: one row appended and attended
 *        per call, with the append's breakdown, at two cache depths and for
 *        both kinds.
 */
TEST_F(HvxAttnQDecode, StepTiming) {
  for (uint32_t kind : {0u, 1u}) {
    for (uint32_t rows : {512u, 4096u}) {
      const uint32_t n_kv = 8, hd = 128, n_q_heads = 16, width = n_kv * hd;
      std::vector<uint16_t> k(static_cast<size_t>(rows) * width), v(k.size());
      fill_hf(k, 0x57E90001u, 1.0f);
      fill_hf(v, 0x57E90002u, 1.0f);
      std::vector<float> q(static_cast<size_t>(n_q_heads) * hd);
      fill_deterministic(q, 0x57E90003u, 3.0f);
      uint32_t h = 0;
      ASSERT_EQ(nntr_hvx_kv_register_q(handle_, kind, rows, n_kv, hd, &h),
                AEE_SUCCESS);
      ASSERT_EQ(nntr_hvx_kv_append_q(
                  handle_, h, 0, k.data(), static_cast<int>((rows - 1) * width),
                  v.data(), static_cast<int>((rows - 1) * width)),
                AEE_SUCCESS);
      std::vector<float> out(q.size());
      std::vector<uint32_t> st(8, 0), acc(8, 0);
      const int reps = 8;
      for (int r = 0; r < reps; ++r) {
        const int err = nntr_hvx_attn_q_step(
          handle_, h, rows - 1,
          k.data() + static_cast<size_t>(rows - 1) * width,
          static_cast<int>(width),
          v.data() + static_cast<size_t>(rows - 1) * width,
          static_cast<int>(width), 1, rows - 1, rows, n_q_heads, 0, 0.0f,
          q.data(), static_cast<int>(q.size()), nullptr, 0, out.data(),
          static_cast<int>(out.size()), st.data(), 8);
        ASSERT_EQ(err, AEE_SUCCESS) << hex(err);
        for (int i = 0; i < 7; ++i) {
          acc[i] += st[i];
        }
      }
      std::cout << "ATTN_Q_FIELD path=step kind=" << kind << " rows=" << rows
                << " append_us=" << acc[0] / reps
                << " attn_us=" << acc[1] / reps << " total_us=" << acc[3] / reps
                << " quant_us=" << acc[4] / reps
                << " stage_us=" << acc[5] / reps << " bake_us=" << acc[6] / reps
                << "\n";
      EXPECT_EQ(nntr_hvx_kv_release_q(handle_, h), AEE_SUCCESS);
    }
  }
}

/**
 * A long prompt in one step: the whole prefill's rows appended and attended
 * in a single call, the shape a 3k-token prompt hands the layer.
 */
TEST_F(HvxAttnQDecode, StepTimingLongPrefill) {
  const uint32_t rows = 3072, n_kv = 8, hd = 128, n_q_heads = 16;
  const uint32_t width = n_kv * hd;
  std::vector<uint16_t> k(static_cast<size_t>(rows) * width), v(k.size());
  fill_hf(k, 0x57E90011u, 1.0f);
  fill_hf(v, 0x57E90012u, 1.0f);
  std::vector<float> q(static_cast<size_t>(rows) * n_q_heads * hd);
  fill_deterministic(q, 0x57E90013u, 3.0f);
  std::vector<float> out(q.size());
  std::vector<uint32_t> st(8, 0);
  uint32_t h = 0;
  ASSERT_EQ(nntr_hvx_kv_register_q(handle_, 0, rows + 64, n_kv, hd, &h),
            AEE_SUCCESS);
  const int err = nntr_hvx_attn_q_step(
    handle_, h, 0, k.data(), static_cast<int>(k.size()), v.data(),
    static_cast<int>(v.size()), rows, 0, rows, n_q_heads, 0, 0.0f, q.data(),
    static_cast<int>(q.size()), nullptr, 0, out.data(),
    static_cast<int>(out.size()), st.data(), 8);
  std::cout << "ATTN_Q_FIELD path=step_prefill kind=0 rows=" << rows
            << " err=" << hex(err) << " append_us=" << st[0]
            << " attn_us=" << st[1] << " total_us=" << st[3]
            << " quant_us=" << st[4] << " stage_us=" << st[5]
            << " bake_us=" << st[6] << " appended=" << st[7] << "\n";
  EXPECT_EQ(err, AEE_SUCCESS) << hex(err);
  EXPECT_EQ(nntr_hvx_kv_release_q(handle_, h), AEE_SUCCESS);
}

TEST_F(HvxAttnQ, AccumulatorLayoutIsRowMajorStrided) {
  std::vector<uint32_t> layout(3, 0);
  const int err = nntr_hvx_probe_acc_i32_layout(handle_, layout.data(), 3);
  ASSERT_EQ(err, AEE_SUCCESS) << hex(err);
  std::cout << "ATTN_Q_FIELD field=acc_layout usable=" << layout[0]
            << " base=" << layout[1] << " row_stride=" << layout[2] << "\n";
  // The in-place accumulator read every int kernel here relies on.
  EXPECT_EQ(layout[0], 1u) << "int32 accumulator readout is not row-major "
                              "strided on this part; see hexkl_acc_tile.h";
}

TEST_F(HvxAttnQ, ReportSessionInfo) {
  // The session's info row (nntr_hvx.idl): 8 slots, [1] the VTCM arena
  // in bytes, [7] the fp16 HMX rate.
  uint32_t info[8] = {0};
  const int err = nntr_hvx_session_info(handle_, info, 8);
  ASSERT_EQ(err, AEE_SUCCESS) << hex(err);
  const uint32_t vtcm_size = info[1], hmx_fp16_rate = info[7];
  std::cout << "ATTN_Q_FIELD field=session vtcm_size=" << vtcm_size
            << " hmx_fp16_rate=" << hmx_fp16_rate << "\n";
  EXPECT_GT(vtcm_size, 0u);
}

/**
 * R1 of 23_attention_v2_plan.md: two convert passes with scales s and
 * s/256 give the low 16 bits of floor(acc * s / 512), at the accumulator
 * magnitudes an hd=256 Q.K^T produces. Coherent K rows (all +1 or all -1,
 * quantized to +-127) against all-255 activation rows push |acc| to
 * 255 * 127 * 256 ~ 2^23; the other rows are random.
 */
TEST_F(HvxAttnQ, ConvertPlanesAreLow16BitsOfScaledAccumulator) {
  const uint32_t rows = 96, n_kv = 1, hd = 256;
  std::vector<uint16_t> k(static_cast<size_t>(rows) * hd), v(k.size());
  fill_hf(k, 0x0C01u, 1.0f);
  fill_hf(v, 0x0C02u, 1.0f);
  for (uint32_t j = 0; j < rows; ++j) {
    if (j % 4 == 0 || j % 4 == 1) {
      const uint16_t one = f32_to_hf(j % 4 == 0 ? 1.0f : -1.0f);
      std::fill(k.begin() + static_cast<size_t>(j) * hd,
                k.begin() + static_cast<size_t>(j + 1) * hd, one);
    }
  }
  uint32_t h = 0;
  ASSERT_EQ(nntr_hvx_kv_register_q(handle_, 0, rows, n_kv, hd, &h),
            AEE_SUCCESS);
  ASSERT_EQ(nntr_hvx_kv_append_q(handle_, h, 0, k.data(),
                                 static_cast<int>(k.size()), v.data(),
                                 static_cast<int>(v.size())),
            AEE_SUCCESS);

  std::vector<uint8_t> act(static_cast<size_t>(64) * hd);
  uint32_t s = 0xC0FFEE01u;
  for (uint32_t r = 0; r < 64; ++r) {
    for (uint32_t d = 0; d < hd; ++d) {
      s = s * 1664525u + 1013904223u;
      const uint8_t rnd = static_cast<uint8_t>(s >> 24);
      act[static_cast<size_t>(r) * hd + d] = r % 4 == 0   ? 255
                                             : r % 4 == 1 ? 0
                                             : r % 4 == 2 ? 128
                                                          : rnd;
    }
  }

  // 512 is the identity (plane0 = acc & 0xff); the others scale a 2^23
  // accumulator into and around the int16 range, two of them not powers
  // of two.
  const float scales[] = {512.0f, 2.0f, 1.0f, 0.75f, 1.5f, 0.00390625f};
  int32_t acc_max = 0;
  uint32_t cyc_pass = 0, cyc_drain = 0, cyc_sync = 0;
  for (uint32_t c = 0; c < rows / 32; ++c) {
    for (float sc : scales) {
      SCOPED_TRACE("col tile " + std::to_string(c) + " scale " +
                   std::to_string(sc));
      const uint16_t s0 = f32_to_hf(sc), s1 = f32_to_hf(sc / 256.0f);
      std::vector<uint8_t> p0(2048), p1(2048);
      std::vector<int16_t> s16(2048);
      std::vector<int32_t> acc(2048);
      std::vector<uint32_t> st(3, 0);
      const int err = nntr_hvx_probe_cvt(
        handle_, h, 0, c, act.data(), static_cast<int>(act.size()), s0, s1, 0,
        1000, p0.data(), 2048, p1.data(), 2048, s16.data(), 2048, acc.data(),
        2048, st.data(), 3);
      ASSERT_EQ(err, AEE_SUCCESS) << hex(err);
      cyc_pass = st[0];
      cyc_drain = st[1];
      cyc_sync = st[2];
      // Each plane on its own: its byte against the model, as a signed
      // difference modulo 256. The combined int16 against the zip the DSP
      // computed on HVX.
      size_t bad0 = 0, bad1 = 0, zip_bad = 0, shown = 0;
      int max_e0 = 0, max_e1 = 0;
      for (uint32_t p = 0; p < 2048; ++p) {
        acc_max = std::max(acc_max, std::abs(acc[p]));
        const double vd = static_cast<double>(acc[p]) *
                          static_cast<double>(hf_to_f32(s0)) / 512.0;
        const int64_t fl = static_cast<int64_t>(std::floor(vd));
        const int e0 = static_cast<int8_t>(
          static_cast<uint8_t>(p0[p] - static_cast<uint8_t>(fl & 0xff)));
        const int e1 = static_cast<int8_t>(
          static_cast<uint8_t>(p1[p] - static_cast<uint8_t>((fl >> 8) & 0xff)));
        bad0 += e0 != 0;
        bad1 += e1 != 0;
        max_e0 = std::max(max_e0, std::abs(e0));
        max_e1 = std::max(max_e1, std::abs(e1));
        if ((e0 != 0 || e1 != 0) && shown++ < 3) {
          std::cout << "  cvt off p=" << p << " acc=" << acc[p]
                    << " exact=" << vd << " e_lo=" << e0 << " e_hi=" << e1
                    << "\n";
        }
        const int16_t zipped =
          static_cast<int16_t>(static_cast<uint16_t>(p0[p] | (p1[p] << 8)));
        zip_bad += s16[p] != zipped;
      }
      std::cout << "ATTN_Q_FIELD field=cvt_scale c=" << c << " scale=" << sc
                << " bad_lo=" << bad0 << " bad_hi=" << bad1
                << " max_err_lo=" << max_e0 << " max_err_hi=" << max_e1 << "\n";
      EXPECT_EQ(zip_bad, 0u) << "HVX zip of the two planes != lo | hi << 8";
      // What R1 found (23_attention_v2_plan.md): a power-of-two scale is a
      // shift and exact at every magnitude; any other scale floors a
      // product carrying only a few guard bits, so each plane can be one
      // step low near an integer boundary -- which in the high plane is a
      // 256-step error in the combined value. Hence: convert scales are
      // powers of two, the residual factor goes to HVX.
      const bool pow2 = std::ldexp(1.0f, std::ilogb(sc)) == sc;
      if (pow2) {
        EXPECT_EQ(bad0, 0u) << "low byte plane";
        EXPECT_EQ(bad1, 0u) << "high byte plane";
      } else {
        EXPECT_LE(max_e0, 1) << "non-power-of-two: low plane off by > 1";
        EXPECT_LE(max_e1, 1) << "non-power-of-two: high plane off by > 1";
      }
    }
  }
  std::cout << "ATTN_Q_FIELD field=cvt acc_max=" << acc_max
            << " cycles_per_pass_issue=" << cyc_pass
            << " cycles_per_pass_synced=" << cyc_sync
            << " cycles_library_drain_scalar_copy=" << cyc_drain << "\n";
  EXPECT_GT(acc_max, 4000000) << "the probe did not reach 2^22 accumulators";
  EXPECT_EQ(nntr_hvx_kv_release_q(handle_, h), AEE_SUCCESS);
}

/**
 * R3 of 23_attention_v2_plan.md: the HVX integer softmax against the
 * scalar definition, bit for bit, over blocks with every tile class:
 * fully visible, diagonal, window edge, short cache, padding rows.
 */
struct SoftmaxCase {
  const char *name;
  uint32_t ct, col0, n_cols, row0, n_rows, window;
};

TEST_F(HvxAttnQ, SoftmaxQMatchesReferenceBitExact) {
  const SoftmaxCase cases[] = {
    {"causal", 35, 0, 1120, 1000, 64, 0},
    {"window", 36, 992, 1152, 2048, 64, 1024},
    {"padding", 3, 0, 70, 30, 40, 0},
    {"dense4096", 128, 0, 4096, 4032, 64, 0},
  };
  const uint32_t F = 8;
  int k = 0;
  uint16_t rho = 0;
  ASSERT_EQ(hvx_softmax_q_scale(1.0f / 1316.0f, F, &k, &rho), 0);
  uint32_t seed = 0x50f7u;
  for (const SoftmaxCase &c : cases) {
    SCOPED_TRACE(c.name);
    const size_t n = static_cast<size_t>(c.ct) * HVX_SOFTMAX_Q_TILE;
    std::vector<int16_t> s(n), corr(static_cast<size_t>(c.ct + 1) * 32, 0);
    for (auto &x : s) {
      seed = seed * 1664525u + 1013904223u;
      // Mostly moderate scores, a few far above, so every shift occurs.
      const int32_t v = static_cast<int32_t>(seed >> 16) - 32768;
      x = static_cast<int16_t>((seed & 0x3f) == 0 ? v / 2 + 8000 : v / 8);
    }
    for (uint32_t i = 0; i < c.ct * 32; ++i) {
      seed = seed * 1664525u + 1013904223u;
      corr[i] = static_cast<int16_t>(static_cast<int32_t>(seed >> 20) - 2048);
    }
    hvx_softmax_q_block b{};
    b.n_col_tiles = c.ct;
    b.col0 = c.col0;
    b.n_cols = c.n_cols;
    b.row0 = c.row0;
    b.n_rows = c.n_rows;
    b.window = c.window;
    b.rho_q15 = rho;
    b.frac_bits = static_cast<uint8_t>(F);
    std::vector<uint8_t> p_ref(n, 0xAA), p_dsp(n, 0x55);
    std::vector<int32_t> rs_ref(HVX_SOFTMAX_Q_ROWSUM_WORDS, -1),
      rs_dsp(HVX_SOFTMAX_Q_ROWSUM_WORDS, -2);
    hvx_softmax_q_ref(&b, s.data(), corr.data(), p_ref.data(), rs_ref.data());
    std::vector<uint32_t> st(5, 0);
    const int err = nntr_hvx_probe_softmax_q(
      handle_, c.ct, c.col0, c.n_cols, c.row0, c.n_rows, c.window, rho, F,
      s.data(), static_cast<int>(s.size()), corr.data(),
      static_cast<int>(corr.size()), p_dsp.data(), static_cast<int>(n),
      rs_dsp.data(), static_cast<int>(rs_dsp.size()), st.data(), 5);
    ASSERT_EQ(err, AEE_SUCCESS) << hex(err);
    size_t bad = 0, shown = 0;
    for (size_t i = 0; i < n; ++i) {
      if (p_ref[i] != p_dsp[i]) {
        ++bad;
        if (shown++ < 4) {
          std::cout << "  softmax mismatch tile " << i / HVX_SOFTMAX_Q_TILE
                    << " row " << (i % HVX_SOFTMAX_Q_TILE) / 32 << " col "
                    << i % 32 << " ref " << int(p_ref[i]) << " dsp "
                    << int(p_dsp[i]) << " s=" << s[i] << "\n";
        }
      }
    }
    size_t bad_rs = 0;
    for (uint32_t r = 0; r < 64; ++r) {
      const uint32_t at = hvx_softmax_q_rowsum_index(r);
      bad_rs += rs_ref[at] != rs_dsp[at];
    }
    std::cout << "ATTN_Q_FIELD field=softmax_q case=" << c.name
              << " tiles=" << c.ct << " cycles=" << st[0]
              << " cycles_per_tile=" << st[0] / c.ct << " bad_p=" << bad
              << " bad_rowsum=" << bad_rs << "\n";
    EXPECT_EQ(bad, 0u) << "P' differs from the reference";
    EXPECT_EQ(bad_rs, 0u) << "row sums differ from the reference";
  }
}

TEST_F(HvxAttnQ, ReportHvxRate) {
  std::vector<uint32_t> st(12, 0);
  const int err = nntr_hvx_probe_hvx_rate(handle_, 100000, st.data(), 12);
  ASSERT_EQ(err, AEE_SUCCESS) << hex(err);
  std::cout << "ATTN_Q_FIELD field=hvx_rate cycles_per_vadd=" << st[0]
            << " cycles_per_vmpy_q15=" << st[1]
            << " cycles_per_exp2_chain=" << st[2]
            << " cycles_per_vtcm_load=" << st[3] << " stream_load_use=" << st[4]
            << " stream_load_ahead=" << st[5] << " pass1_stride4k=" << st[6]
            << " pass1_stride128=" << st[7] << " pass2_wmajor=" << st[8]
            << " pass2_tilemajor=" << st[9] << " pass2_padded=" << st[10]
            << " pass1_padded=" << st[11] << "\n";
}

TEST_F(HvxAttnQ, ReportWhI8Permutation) {
  std::vector<uint16_t> pos(1024, 0);
  const int err = nntr_hvx_probe_wh_i8_pos(handle_, pos.data(), 1024);
  ASSERT_EQ(err, AEE_SUCCESS) << hex(err);
  std::cout << "ATTN_Q_FIELD field=wh_i8_pos";
  for (uint32_t i = 0; i < 1024; ++i) {
    std::cout << (i ? "," : " ") << pos[i];
  }
  std::cout << "\n";
}

TEST_F(HvxAttnQ, Int8CacheMatchesHostAndMultiplies) {
  RunKind(0, 70, 2, 64);
  RunKind(0, 100, 4, 128);
}

/** R4a: fixed scales and the global layer's head_dim, both on the DSP. */
TEST_F(HvxAttnQ, Int8FixedScalesHeadDim512MatchesHostAndMultiplies) {
  RunKind(0, 70, 2, 512, true);
}

TEST_F(HvxAttnQ, Int4CacheMatchesHostAndMultiplies) {
  RunKind(1, 70, 2, 64);
  RunKind(1, 100, 4, 128);
}

TEST_F(HvxAttnQ, RejectsBadParameters) {
  uint32_t h = 0;
  EXPECT_TRUE(is_badparm(nntr_hvx_kv_register_q(handle_, 2, 64, 2, 64, &h)));
  EXPECT_TRUE(is_badparm(nntr_hvx_kv_register_q(handle_, 0, 64, 2, 100, &h)));
  ASSERT_EQ(nntr_hvx_kv_register_q(handle_, 0, 64, 2, 64, &h), AEE_SUCCESS);
  std::vector<uint16_t> rows(2 * 128, 0);
  // Length not a multiple of the row width.
  EXPECT_TRUE(is_badparm(
    nntr_hvx_kv_append_q(handle_, h, 0, rows.data(), 100, rows.data(), 100)));
  // Past max_rows.
  EXPECT_TRUE(is_badparm(
    nntr_hvx_kv_append_q(handle_, h, 63, rows.data(), 256, rows.data(), 256)));
  EXPECT_EQ(nntr_hvx_kv_release_q(handle_, h), AEE_SUCCESS);
  EXPECT_TRUE(is_badparm(nntr_hvx_kv_release_q(handle_, h)));
}

} // namespace

/**
 * @brief Main gtest (this tree's googletest_main static library is
 *        gtest-all only; every device test carries its own main).
 */
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
