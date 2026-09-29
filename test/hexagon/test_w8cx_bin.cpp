// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_w8cx_bin.cpp
 * @date	31 August 2026
 * @brief	x86 self-check for the W8_CX checkpoint reader: size arithmetic,
 *		scale blocks finite/positive, RMSNorm gammas near 1, int8
 *		payload non-degenerate; with a second, W4CX file: the
 *		W4CX header's int8 set, int4 codes in [-7, 7] on the six
 *		projections, int8 embed / down, and the w4cx_down8 layout it
 *		maps to. Not gtest; self-contained main.
 *
 * Build: ./tools/hexagon/build_host_x86.sh test_w8cx_bin
 * Run:   ./build_x86_hexagon/test_w8cx_bin <w8cx.bin> [<w4cx_down8.bin>]
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "qwen3_w8cx_bin.h"

using nntrainer::hexagon::HexModelConfig;
using nntrainer::hexagon::HexModelWeights;
using nntrainer::hexagon::kHexAllI8;
using nntrainer::hexagon::kHexW4cxDown8I8;
using nntrainer::hexagon::kQwen3_0_6b;
using nntrainer::hexagon::Qwen3W8cxBin;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__);   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

bool all_positive_finite(const float *p, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (!(std::isfinite(p[i]) && p[i] > 0.0f))
      return false;
  return true;
}

float mean_abs(const float *p, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; ++i)
    s += std::fabs(p[i]);
  return static_cast<float>(s / static_cast<double>(n));
}

float nonzero_frac(const int8_t *p, size_t n) {
  size_t nz = 0;
  for (size_t i = 0; i < n; ++i)
    nz += p[i] != 0;
  return static_cast<float>(nz) / static_cast<float>(n);
}

/** Every code inside [lo, hi]; an int4 tensor read as int8 or the
 * reverse fails this on the first row. */
bool codes_within(const int8_t *p, size_t n, int lo, int hi) {
  for (size_t i = 0; i < n; ++i)
    if (p[i] < lo || p[i] > hi)
      return false;
  return true;
}

/** An int8 tensor uses the range: some |code| > 7 in the first rows. */
bool has_wide_codes(const int8_t *p, size_t n) {
  return !codes_within(p, n, -7, 7);
}

bool gamma_sane(const float *p, size_t n) {
  float m = mean_abs(p, n);
  // qwen3-0.6b gammas range from ~0.17 (layer 0) to ~14.6 (layer 27).
  return m > 0.01f && m < 100.0f;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("usage: %s <w8cx.bin> [<w4cx_down8.bin>]\n", argv[0]);
    return 2;
  }

  // 1. shape arithmetic, independent of the file
  CHECK(Qwen3W8cxBin::expected_size(kQwen3_0_6b) == 598230528ull);

  Qwen3W8cxBin bin(argv[1], kQwen3_0_6b); // throws on size mismatch
  const HexModelWeights &w = bin.weights();
  CHECK(w.layers.size() == 28u);

  // 2. every scale block is finite and strictly positive - a cursor that
  //    drifted by one tensor reads int8 payload as float and fails here
  CHECK(all_positive_finite(w.embed_s, kQwen3_0_6b.vocab));
  for (const auto &l : w.layers) {
    CHECK(all_positive_finite(l.wq_s, 2048));
    CHECK(all_positive_finite(l.wk_s, 1024));
    CHECK(all_positive_finite(l.wv_s, 1024));
    CHECK(all_positive_finite(l.wo_s, 1024));
    CHECK(all_positive_finite(l.w_gate_s, 3072));
    CHECK(all_positive_finite(l.w_up_s, 3072));
    CHECK(all_positive_finite(l.w_down_s, 1024));
  }

  // 3. RMSNorm gammas of a trained model are O(1) - catches a cursor that
  //    landed on int8 payload (which reads as denormal/garbage floats)
  for (const auto &l : w.layers) {
    CHECK(gamma_sane(l.attn_norm, 1024));
    CHECK(gamma_sane(l.ffn_norm, 1024));
    CHECK(gamma_sane(l.q_norm, 128));
    CHECK(gamma_sane(l.k_norm, 128));
  }
  CHECK(gamma_sane(w.final_norm, 1024));

  // 4. int8 payload is not degenerate (a zero block means a wrong offset)
  CHECK(nonzero_frac(w.embed, 151936ull * 1024) > 0.5f);
  for (const auto &l : w.layers)
    CHECK(nonzero_frac(l.wq, 2048ull * 1024) > 0.5f);

  // 5. A header-less file is the W8 layout: all classes int8, tiled32.
  CHECK(w.i8_mask == kHexAllI8);
  {
    HexModelConfig c = kQwen3_0_6b;
    bin.apply_layout(c);
    CHECK(c.weight_layout == NNTR_HTP_WEIGHT_LAYOUT_TILED32);
  }

  // 6. make_w4cx_bin.py --i8-tensors down,embed (hvx_w4cx): the same
  //    stream behind the W4CX header; the six projections are int4 codes,
  //    embed / down int8.
  if (argc >= 3) {
    Qwen3W8cxBin b4(argv[2], kQwen3_0_6b);
    const HexModelWeights &w4 = b4.weights();
    CHECK(w4.i8_mask == kHexW4cxDown8I8);
    HexModelConfig c = kQwen3_0_6b;
    b4.apply_layout(c);
    CHECK(c.weight_layout == NNTR_HTP_WEIGHT_LAYOUT_W4CX_DOWN8 &&
          c.i8_mask == kHexW4cxDown8I8);
    CHECK(all_positive_finite(w4.embed_s, kQwen3_0_6b.vocab));
    CHECK(has_wide_codes(w4.embed, 1024ull * 64));
    for (const auto &l : w4.layers) {
      CHECK(codes_within(l.wq, 2048ull * 1024, -7, 7));
      CHECK(codes_within(l.wk, 1024ull * 1024, -7, 7));
      CHECK(codes_within(l.wv, 1024ull * 1024, -7, 7));
      CHECK(codes_within(l.wo, 1024ull * 2048, -7, 7));
      CHECK(codes_within(l.w_gate, 3072ull * 1024, -7, 7));
      CHECK(codes_within(l.w_up, 3072ull * 1024, -7, 7));
      CHECK(has_wide_codes(l.w_down, 3072ull * 64));
      CHECK(all_positive_finite(l.wq_s, 2048));
      CHECK(all_positive_finite(l.w_down_s, 1024));
      CHECK(gamma_sane(l.attn_norm, 1024));
    }
    CHECK(gamma_sane(w4.final_norm, 1024));
    std::printf("W4CX_BIN_TEST PASS\n");
  }

  std::printf("W8CX_BIN_TEST PASS\n");
  return 0;
}
