// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nntr_htp_bridge_check.cpp
 * @date   24 July 2026
 * @brief  Standalone diagnostic: runs one Q4_0 GEMM through
 * HexagonComputeOps::gemm_q4_0_accel_fp32 (the cDSP bridge) and compares it
 * against nntrainer's own CPU gemm_q4_0_fp32 for the same random inputs.
 * Deliberately NOT wired into meson - this is meant to be compiled directly
 * against an already-built libnntrainer.so with a single compiler
 * invocation (see the comment block below), so it can be rebuilt and pushed
 * quickly while iterating on the bridge, without touching the Android.mk/
 * ndk-build pipeline at all.
 *
 * Build (Android arm64, against an NDK + an already-built libnntrainer.so):
 *
 *   $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang++ \
 *       -std=c++17 -O2 -fPIC \
 *       -I nntrainer -I nntrainer/tensor/cpu_backend -I nntrainer/hexagon \
 *       -I builddir/android_build_result/include \
 *       tools/nntr_htp_bridge_check.cpp \
 *       -L builddir/android_build_result/lib/arm64-v8a -lnntrainer \
 *       -o nntr_htp_bridge_check
 *
 * (adjust -I/-L to wherever your build actually put libnntrainer.so and its
 * headers - the paths above match tools/package_android.sh's output layout.)
 *
 * Run (on-device, same directory as libnntrainer.so and libggml-hexagon.so,
 * or with LD_LIBRARY_PATH covering both):
 *
 *   $ adb push nntr_htp_bridge_check /data/local/tmp/nntrainer/causallm/
 *   $ adb shell "cd /data/local/tmp/nntrainer/causallm && \
 *       LD_LIBRARY_PATH=/data/local/tmp:. ./nntr_htp_bridge_check"
 *
 * Exit code 0 = all cases passed. Nonzero = at least one case failed or
 * threw - read the printed output for which case and why.
 */

#include <compute_ops.h>
#include <hexagon_compute_ops.h>
#include <hexagon_repack.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Case {
  const char *name;
  unsigned int M, N, K;
};

// Shapes (confirmed against test/unittest/unittest_nntrainer_cpu_backend.cpp's
// run_quant_test/test_gemm_q4_0 - see the shape note in nntr-htp-bridge.cpp):
// activation = [M, K] (M rows), weight = [N, K] (N rows, quantized/repacked
// per row), output = [M, N].
//
// K must be a multiple of 256 (block_q4_0's 32-element blocks, tiled 8 at a
// time by the q4x4x2 repack - see hexagon_repack.h). N (weight rows) must
// stay under the bridge's 16K VTCM guard; M (activation rows) under its
// 1024 guard (see nntr-htp-bridge.cpp). M must also be a multiple of 4 -
// the CPU reference kernel's own NEON tiling (ncols_interleaved=4 in
// nntr_ggml_impl_neon.cpp) asserts on this; not a Hexagon-specific
// constraint. Real decode (1 activation row) goes through a separate GEMV
// path in nntrainer, not gemm_q4_0_fp32 - this tool only exercises the GEMM
// path both CPU and Hexagon share.
const Case CASES[] = {
  {"small", 64, 8, 512},
  {"qwen3-0.6b-ish FC (M=64,N=1024,K=1024)", 64, 1024, 1024},
  {"qwen3-0.6b-ish FC (M=32 prefill-like,N=1024,K=1024)", 32, 1024, 1024},
};

bool run_case(const Case &c) {
  std::printf("--- %s (M=%u N=%u K=%u) ---\n", c.name, c.M, c.N, c.K);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Random fp32 "weight" [N, K] (N rows), quantized to standard block_q4_0 -
  // this is exactly what nntrainer's own quantize path produces before
  // repacking.
  std::vector<float> weight_fp32((size_t)c.N * c.K);
  for (auto &v : weight_fp32)
    v = dist(rng);

  auto *cpu_ops = nntrainer::get_cpu_ops();
  auto *hexagon_ops = nntrainer::get_hexagon_ops();

  // Q4_0 byte size: 18 bytes per 32-element block (2-byte fp16 delta + 16
  // packed nibble bytes), so (K/32)*18 bytes per row.
  size_t q4_0_row_bytes = (c.K / 32) * 18;
  std::vector<uint8_t> weight_q4_0((size_t)c.N * q4_0_row_bytes);
  cpu_ops->quantize_q4_0(weight_fp32.data(), weight_q4_0.data(), c.N, c.K,
                          nullptr);

  // Repack to the HTP q4x4x2 tile layout the bridge expects as input -
  // this is what nntrainer's real quantize-time repack already does; here
  // we do it inline since this tool starts from freshly quantized bytes.
  std::vector<uint8_t> weight_htp(weight_q4_0.size());
  nntrainer::repack_q4_0_to_htp_q4x4x2(weight_htp.data(), weight_q4_0.data(),
                                        weight_q4_0.size(), c.N, c.K);

  // Random fp32 activation [M, K] (M rows) - see the shape note in
  // nntr-htp-bridge.cpp.
  std::vector<float> act((size_t)c.M * c.K);
  for (auto &v : act)
    v = dist(rng);

  std::vector<float> cpu_out((size_t)c.M * c.N, 0.0f);
  std::vector<float> hexagon_out((size_t)c.M * c.N, 0.0f);

  try {
    cpu_ops->gemm_q4_0_fp32(c.M, c.N, c.K, act.data(), c.K,
                             weight_q4_0.data(), c.N, cpu_out.data(), c.N);
  } catch (const std::exception &e) {
    std::printf("  CPU reference threw: %s\n", e.what());
    return false;
  }

  try {
    hexagon_ops->gemm_q4_0_accel_fp32(weight_htp.data(), act.data(),
                                       hexagon_out.data(), c.M, c.N, c.K);
  } catch (const std::exception &e) {
    std::printf("  Hexagon bridge threw: %s\n", e.what());
    return false;
  }

  double max_abs_err = 0.0, sum_abs_err = 0.0;
  size_t worst_idx = 0;
  for (size_t i = 0; i < cpu_out.size(); ++i) {
    double err = std::fabs((double)cpu_out[i] - (double)hexagon_out[i]);
    sum_abs_err += err;
    if (err > max_abs_err) {
      max_abs_err = err;
      worst_idx = i;
    }
  }
  double mean_abs_err = sum_abs_err / cpu_out.size();

  // Q4_0 has real, expected quantization error even between two correct
  // implementations if they round differently - this is a generous
  // tolerance meant to catch "wrong tensor shape/stride" or "garbage from
  // the DSP" bugs, not to certify bit-exact numerics.
  const double TOLERANCE = 0.5;
  bool pass = max_abs_err < TOLERANCE;

  std::printf("  max_abs_err=%.6f (at idx %zu: cpu=%.6f hexagon=%.6f) "
              "mean_abs_err=%.6f -> %s\n",
              max_abs_err, worst_idx, cpu_out[worst_idx],
              hexagon_out[worst_idx], mean_abs_err, pass ? "PASS" : "FAIL");

  return pass;
}

} // namespace

int main() {
  std::printf(
    "nntr_htp_bridge_check: comparing HexagonComputeOps::"
    "gemm_q4_0_accel_fp32 against the CPU reference for %zu case(s).\n"
    "If this crashes or hangs instead of printing PASS/FAIL, check "
    "'adb logcat | grep -i \"ggml-hex|fastrpc|cdsprpc\"' in another shell "
    "before re-running.\n\n",
    sizeof(CASES) / sizeof(CASES[0]));

  bool all_pass = true;
  for (const auto &c : CASES) {
    if (!run_case(c)) {
      all_pass = false;
    }
    std::printf("\n");
  }

  std::printf(all_pass ? "ALL CASES PASSED\n" : "AT LEAST ONE CASE FAILED\n");
  return all_pass ? 0 : 1;
}
