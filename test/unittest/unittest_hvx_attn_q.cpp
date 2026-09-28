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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "hexkl_attn_q_plan.h"
#include "hexkl_kv_q.h"
#include "hvx_attn_q_model.h"
#include "hvx_attn_test_util.h"
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
                hexkl_kv_q_table &host, uint32_t &host_h, uint32_t &dsp_h) {
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

    const uint32_t split = rows / 2;
    auto append = [&](uint32_t row0, uint32_t n, const uint16_t *kr,
                      const uint16_t *vr) {
      ASSERT_EQ(hexkl_kv_q_append(&host, host_h, row0, n, kr, vr, nullptr), 0);
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
    const uint32_t rows = kv->max_rows, n_kv = kv->n_head_kv, hd = kv->head_dim,
                   dt = kv->n_dot_tiles;
    const size_t values = static_cast<size_t>(rows) * n_kv * hd;
    const size_t heads = static_cast<size_t>(rows) * n_kv;
    kq.assign(values, 0);
    vq.assign(values, 0);
    std::vector<float> sk(heads), sv(heads * dt);
    std::vector<int32_t> cs(heads);
    int err = nntr_hvx_kv_dump_q(
      handle_, dsp_h, 0, rows, kq.data(), static_cast<int>(values), vq.data(),
      static_cast<int>(values), sk.data(), static_cast<int>(heads), cs.data(),
      static_cast<int>(heads), sv.data(), static_cast<int>(heads * dt));
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_dump_q failed: " << hex(err);

    std::vector<int8_t> hkq(values), hvq(values);
    std::vector<float> hsk(heads), hsv(heads * dt);
    std::vector<int32_t> hcs(heads);
    ASSERT_EQ(hexkl_kv_q_dump(kv, 0, rows, hkq.data(), hvq.data(), hsk.data(),
                              hcs.data(), hsv.data()),
              0);
    size_t k_diff = 0, v_diff = 0;
    for (size_t i = 0; i < values; ++i) {
      k_diff += kq[i] != hkq[i];
      v_diff += vq[i] != hvq[i];
    }
    EXPECT_EQ(k_diff, 0u) << "K values differ from the host quantizer";
    EXPECT_EQ(v_diff, 0u) << "V values differ from the host quantizer";
    EXPECT_EQ(std::memcmp(sk.data(), hsk.data(), heads * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(cs.data(), hcs.data(), heads * sizeof(int32_t)), 0);
    EXPECT_EQ(std::memcmp(sv.data(), hsv.data(), heads * dt * sizeof(float)),
              0);
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

  void RunKind(uint32_t kind, uint32_t rows, uint32_t n_kv, uint32_t hd) {
    SCOPED_TRACE("kind " + std::to_string(kind) + " rows " +
                 std::to_string(rows) + " n_kv " + std::to_string(n_kv) +
                 " hd " + std::to_string(hd));
    hexkl_kv_q_table host{};
    uint32_t host_h = 0, dsp_h = 0;
    FillBoth(kind, rows, n_kv, hd, host, host_h, dsp_h);
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

constexpr int kQStatCount = 10;
const char *const kQStatNames[kQStatCount] = {
  "qprep",  "dma", "qk",   "dequant", "softmax",
  "pquant", "pv",  "oupd", "store",   "n_blocks"};

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
    }
  }
};

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

TEST_F(HvxAttnQ, Int8CacheMatchesHostAndMultiplies) {
  RunKind(0, 70, 2, 64);
  RunKind(0, 100, 4, 128);
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
