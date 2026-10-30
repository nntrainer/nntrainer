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
  // min / max: exact. quant: the scalar loop mha_core had, within one
  // code (the NEON multiply and add may round differently from a fused
  // scalar under -ffast-math). dequant: exact.
  int q_off1 = 0, q_bad = 0, mm_bad = 0, dq_bad = 0;
  long q_n = 0;
  for (unsigned n : {1u, 7u, 8u, 9u, 31u, 256u, 512u}) {
    for (int t = 0; t < 200; ++t) {
      std::vector<float> f(n);
      for (unsigned i = 0; i < n; ++i)
        f[i] = ((int)(rnd() % 20001) - 10000) * (0.0005f * (1 + t % 7));
      float lo = 1e30f, hi = -1e30f, rlo = 1e30f, rhi = -1e30f;
      causallm::min_max_f32(f.data(), n, lo, hi);
      for (unsigned i = 0; i < n; ++i) {
        rlo = std::min(rlo, f[i]);
        rhi = std::max(rhi, f[i]);
      }
      mm_bad += lo != rlo || hi != rhi;
      const float scale = std::max(rhi - rlo, 1e-6f) / 65535.0f;
      const float zp = std::round(-rlo / scale), inv = 1.0f / scale;
      std::vector<uint16_t> u(n);
      causallm::quant_u16_f32(f.data(), n, inv, zp, u.data());
      for (unsigned i = 0; i < n; ++i) {
        float v = f[i] * inv + zp + 0.5f;
        v = v < 0.0f ? 0.0f : v > 65535.0f ? 65535.0f : v;
        const int d = (int)u[i] - (int)static_cast<uint16_t>(v);
        q_off1 += d != 0;
        q_bad += d > 1 || d < -1;
        ++q_n;
      }
      std::vector<float> o(n);
      causallm::dequant_u16_f32(u.data(), n, scale, 32768.0f, o.data());
      for (unsigned i = 0; i < n; ++i) {
        const float r = (static_cast<float>(u[i]) - 32768.0f) * scale;
        dq_bad += std::memcmp(&r, &o[i], 4) != 0;
      }
    }
  }
  printf("min_max bad %d, quant %ld values: %d off by one, %d worse, "
         "dequant bad %d\n",
         mm_bad, q_n, q_off1, q_bad, dq_bad);
  bad += mm_bad + q_bad + dq_bad;
  printf("abs_max: %d cases, %d bad\n%s\n", cases, bad,
         bad ? "ABS MAX FAILED" : "ABS MAX OK");
  return bad != 0;
}
