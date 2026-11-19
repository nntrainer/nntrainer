// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_kv_snap.cpp
 * @date	02 October 2026
 * @brief	Unit tests for the fused RoPE + int8 snap + fp16 KV-cache store
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include <cpu_backend.h>
#include <fallback_internal.h>
#include <fp16.h>

namespace {

struct Case {
  unsigned int heads, dim;
  bool rope;
};

void runCase(const Case &c, unsigned int seed) {
  const unsigned int width = c.heads * c.dim, half = c.dim / 2;
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> sd(0.002f, 0.02f), ang(-3.14f, 3.14f);
  std::vector<float> x(width), cos_(half), sin_(half), scales(c.heads);
  for (auto &v : x)
    v = nd(rng) * 0.6f; // some values beyond 127 * s: exercises the clamp
  for (unsigned int k = 0; k < half; ++k) {
    const float a = ang(rng);
    cos_[k] = std::cos(a);
    sin_[k] = std::sin(a);
  }
  for (auto &s : scales)
    s = sd(rng);

  std::vector<uint16_t> out(width, 0xFFFF), ref(width, 0xEEEE);
  const float *cp = c.rope ? cos_.data() : nullptr;
  nntrainer::compute_rotary_emb_value_snap(
    width, c.dim, half, x.data(), out.data(), cp, sin_.data(), scales.data());
  nntrainer::__fallback_compute_rotary_emb_value_snap(
    width, c.dim, half, x.data(), ref.data(), cp, sin_.data(), scales.data());
  ASSERT_EQ(std::memcmp(out.data(), ref.data(), width * sizeof(uint16_t)), 0)
    << "dispatched kernel differs from the scalar reference";

  // every stored value is fp16(q * s) with an integer q in [-128, 127], and
  // q = nearbyint(x' / s) clamped, x' the (rotated) input
  for (unsigned int i = 0; i < width; ++i) {
    const unsigned int h = i / c.dim, k = i % c.dim;
    float xr = x[i];
    if (c.rope) {
      const unsigned int kk = k % half, base = h * c.dim;
      const float a = x[base + kk], b = x[base + kk + half];
      xr =
        (k < half) ? a * cos_[kk] - b * sin_[kk] : a * sin_[kk] + b * cos_[kk];
    }
    const float s = scales[h];
    const float q = std::min(127.0f, std::max(-128.0f, std::nearbyint(xr / s)));
    const float got = nntrainer::compute_fp16_to_fp32(out[i]);
    const double want = static_cast<double>(q) * static_cast<double>(s);
    EXPECT_NEAR(got, want, std::fabs(want) * 1e-3 + 1e-7) << "i=" << i;
  }
}

} // namespace

TEST(nntrainer_kv_snap, sliding_shape_with_rope_p) {
  runCase({8, 256, true}, 1);
}

TEST(nntrainer_kv_snap, global_shape_with_rope_p) {
  runCase({2, 512, true}, 2);
}

TEST(nntrainer_kv_snap, value_path_without_rope_p) {
  runCase({8, 256, false}, 3);
}

TEST(nntrainer_kv_snap, odd_head_dim_tail_p) { runCase({3, 20, true}, 4); }

int main(int argc, char **argv) {
  int result = -1;
  testing::InitGoogleTest(&argc, argv);
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}
