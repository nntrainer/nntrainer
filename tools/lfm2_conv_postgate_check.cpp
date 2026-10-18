// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_conv_postgate_check.cpp
 * @brief LFM ARM convolution/post-gate versus the current core kernel.
 * @author NNTrainer contributors
 * @bug Actual ARM execution is required for FMA parity and timing.
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

// Exercise the same internal helper without adding an app testing API.
#include "../Applications/CausalLM/layers/conv_block_layer.cpp"

namespace {
void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

void fill(std::vector<float> &data, uint32_t seed, bool special) {
  const uint32_t edge[] = {0u,          0x80000000u, 0x00000001u, 0x80000001u,
                           0x3f800000u, 0xbf800000u, 0x00800000u, 0x80800000u,
                           0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
                           0x7fc00001u, 0xffc00123u, 0x3f800001u, 0xbf800001u};
  for (size_t i = 0; i < data.size(); ++i) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    uint32_t bits =
      (seed & 0x807fffffu) | ((117u + static_cast<unsigned>(i % 20u)) << 23u);
    if (special)
      bits = edge[(i + seed % 16u) % 16u];
    std::memcpy(data.data() + i, &bits, sizeof(bits));
  }
}

void reference(const float *g, const float *w, const float *p, float *out,
               unsigned rows) {
  constexpr unsigned C = 2048;
  nntrainer::causal_depthwise_conv1d_k3(g, w, nullptr, out, 1, rows, C);
  nntrainer::ThreadManager::Global().parallel_for(
    0, static_cast<size_t>(rows), [&](size_t r) {
      const float *b = p + r * 3u * C + C;
      float *dst = out + r * C;
      for (unsigned c = 0; c < C; ++c)
        dst[c] = b[c] * dst[c];
    });
}

void equal(const std::vector<float> &ref, const std::vector<float> &got,
           unsigned rows, bool special) {
  if (std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0)
    return;
  unsigned differences = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    uint32_t a, b;
    std::memcpy(&a, ref.data() + i, sizeof(a));
    std::memcpy(&b, got.data() + i, sizeof(b));
    if (a != b && differences++ < 8)
      std::fprintf(stderr, "M=%u special=%u index=%zu ref=%08x got=%08x\n",
                   rows, special, i, a, b);
  }
  throw std::runtime_error("convolution/post-gate is not bit-identical");
}

void run(unsigned rows, bool special) {
  constexpr unsigned C = 2048;
  const size_t n = static_cast<size_t>(rows) * C;
  std::vector<float> g(n), w(3u * C), p(3u * n), got(n), ref(n);
  fill(g, 0x1234567u, special);
  fill(w, 0x89abcdu, special);
  fill(p, 0x1234567u, special);
  const auto saved_g = g, saved_w = w, saved_p = p;
  reference(g.data(), w.data(), p.data(), ref.data(), rows);
  check(causallm::lfmConvPostGatePrefill(g.data(), w.data(), p.data(),
                                         got.data(), rows, C),
        "valid LFM prefill rejected");
  equal(ref, got, rows, special);
  check(std::memcmp(g.data(), saved_g.data(), n * sizeof(float)) == 0 &&
          std::memcmp(w.data(), saved_w.data(), w.size() * sizeof(float)) ==
            0 &&
          std::memcmp(p.data(), saved_p.data(), p.size() * sizeof(float)) == 0,
        "convolution changed an input or its state-source rows");
  std::printf("CONV_POSTGATE M=%u special=%u exact=1 inputs_unchanged=1\n",
              rows, special);
}

void guards() {
  constexpr unsigned C = 2048, M = 16;
  std::vector<float> g(M * C + 1u, 1.0f), w(3u * C, 1.0f);
  std::vector<float> p(3u * M * C, 1.0f), out(M * C, 7.0f);
  const auto call = [&](const float *x, const float *weights, const float *proj,
                        float *y, unsigned rows, unsigned channels) {
    return causallm::lfmConvPostGatePrefill(x, weights, proj, y, rows,
                                            channels);
  };
  for (unsigned m : {0u, 1u, 2u, 15u})
    check(!call(g.data(), w.data(), p.data(), out.data(), m, C),
          "small/decode shape selected");
  check(!call(g.data(), w.data(), p.data(), out.data(), M, 2816),
        "Gemma width selected");
  check(!call(g.data(), w.data(), p.data(), g.data(), M, C), "gated alias");
  check(!call(g.data(), w.data(), p.data(), g.data() + 1, M, C),
        "partial gated alias");
  check(!call(g.data() + 1, w.data(), p.data(), g.data(), M, C),
        "reverse partial gated alias");
  check(!call(g.data(), w.data(), p.data(), p.data() + C, M, C),
        "projection alias");
  check(!call(g.data(), w.data(), p.data(), w.data(), M, C), "weight alias");
  check(!call(nullptr, w.data(), p.data(), out.data(), M, C), "null input");
  check(!call(g.data(), nullptr, p.data(), out.data(), M, C), "null weights");
  check(!call(g.data(), w.data(), nullptr, out.data(), M, C),
        "null projection");
  check(!call(g.data(), w.data(), p.data(), nullptr, M, C), "null output");
  check(std::all_of(out.begin(), out.end(), [](float v) { return v == 7.0f; }),
        "rejected shape wrote output");
  std::puts("CONV_POSTGATE_GUARDS small/decode/Gemma/alias/partial/null PASS");
}

template <typename F> double time_us(F fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double, std::micro>(
           std::chrono::steady_clock::now() - start)
    .count();
}

void bench(unsigned rows, unsigned repeat) {
  const size_t n = static_cast<size_t>(rows) * 2048u;
  std::vector<float> g(n), w(3u * 2048u), p(3u * n), got(n), ref(n);
  fill(g, 0x1234567u, false);
  fill(w, 0x89abcdu, false);
  fill(p, 0x1234567u, false);
  const auto baseline = [&] {
    reference(g.data(), w.data(), p.data(), ref.data(), rows);
  };
  const auto candidate = [&] {
    check(causallm::lfmConvPostGatePrefill(g.data(), w.data(), p.data(),
                                           got.data(), rows, 2048),
          "benchmark shape rejected");
  };
  for (unsigned i = 0; i < 3; ++i) {
    baseline();
    candidate();
  }
  std::vector<double> old_us, new_us;
  for (unsigned i = 0; i < repeat; ++i) {
    // Alternate execution order to reduce consistent warm-cache bias.
    if (i % 2u) {
      new_us.push_back(time_us(candidate));
      old_us.push_back(time_us(baseline));
    } else {
      old_us.push_back(time_us(baseline));
      new_us.push_back(time_us(candidate));
    }
  }
  equal(ref, got, rows, false);
  std::sort(old_us.begin(), old_us.end());
  std::sort(new_us.begin(), new_us.end());
  std::printf("CONV_POSTGATE_BENCH M=%u n=%u old_us=%.3f new_us=%.3f "
              "ratio=%.4f old_range=%.3f:%.3f new_range=%.3f:%.3f\n",
              rows, repeat, old_us[repeat / 2u], new_us[repeat / 2u],
              new_us[repeat / 2u] / old_us[repeat / 2u], old_us.front(),
              old_us.back(), new_us.front(), new_us.back());
}
} // namespace

int main(int argc, char **argv) {
  try {
    check(argc <= 2, "usage: lfm2_conv_postgate_check [benchmark-repetitions]");
    const unsigned repeat = argc == 2 ? std::stoul(argv[1]) : 0;
    check(repeat <= 1000, "too many repetitions");
    for (unsigned m : {16u, 17u, 31u, 64u, 1024u, 2048u}) {
      run(m, false);
      run(m, true);
    }
    guards();
    std::puts("CONV_POSTGATE_PASS cases=12");
    if (repeat) {
      bench(1024u, repeat);
      bench(2048u, repeat);
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "CONV_POSTGATE_FAIL: %s\n", e.what());
    return 1;
  }
}
