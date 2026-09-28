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

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "hexkl_kv_q.h"
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
