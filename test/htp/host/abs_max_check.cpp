// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   abs_max_check.cpp
 * @date   8 October 2026
 * @brief  Applications/CausalLM/layers/abs_max.h against the scalar loops
 *         the q2 calibration had, bit for bit, on every fp16 pattern and on
 *         f32 with NaNs. run_host_checks.sh runs it natively and, when an
 *         aarch64 g++ and qemu-aarch64 are installed, on the NEON path.
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <abs_max.h>
#include <cstdio>
#include <cstring>
#include <vector>
// Scalar reference: the loops calibrate_q2_scales had.
static float ref_f16(const uint16_t *x, unsigned n, float m) {
  for (unsigned i = 0; i < n; ++i)
    m = std::max(m, std::fabs(nntrainer::compute_fp16_to_fp32(x[i])));
  return m;
}
static float ref_f32(const float *x, unsigned n, float m) {
  for (unsigned i = 0; i < n; ++i)
    m = std::max(m, std::fabs(x[i]));
  return m;
}
int main() {
  uint32_t s = 12345;
  auto rnd = [&] {
    s = s * 1664525u + 1013904223u;
    return s;
  };
  int bad = 0, cases = 0;
  for (unsigned n : {1u, 7u, 8u, 9u, 31u, 256u, 512u, 2048u, 4096u}) {
    for (int t = 0; t < 200; ++t) {
      std::vector<uint16_t> h(n);
      std::vector<float> f(n);
      for (unsigned i = 0; i < n; ++i) {
        h[i] = (uint16_t)rnd(); // every pattern: NaN, inf, subnormal
        uint32_t b = rnd();
        if (t % 3)
          b &= 0xBFFFFFFFu; // mostly finite floats
        std::memcpy(&f[i], &b, 4);
      }
      float m0 = (t % 5) * 0.25f;
      ++cases;
      float a = causallm::abs_max_f16(h.data(), n, m0),
            b = ref_f16(h.data(), n, m0);
      float c = causallm::abs_max_f32(f.data(), n, m0),
            d = ref_f32(f.data(), n, m0);
      std::vector<float> l1(n, m0), l2(n, m0);
      causallm::abs_max_f16_lanes(h.data(), n, l1.data());
      for (unsigned i = 0; i < n; ++i)
        l2[i] =
          std::max(l2[i], std::fabs(nntrainer::compute_fp16_to_fp32(h[i])));
      if (std::memcmp(&a, &b, 4) || std::memcmp(&c, &d, 4) ||
          std::memcmp(l1.data(), l2.data(), n * 4)) {
        if (++bad < 5)
          printf("n=%u t=%d f16 %a/%a f32 %a/%a\n", n, t, a, b, c, d);
      }
    }
  }
  printf("abs_max: %d cases, %d bad\n%s\n", cases, bad,
         bad ? "ABS MAX FAILED" : "ABS MAX OK");
  return bad != 0;
}
