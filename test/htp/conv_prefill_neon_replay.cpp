// SPDX-License-Identifier: Apache-2.0
/**
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
/** @file conv_prefill_neon_replay.cpp
 * @brief Exercise wide prefill, scalar tails, bias and in-place fallback.
 * Compare the binary output against the baseline library on ARM.
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace nntrainer::neon {
void causal_depthwise_conv1d_k3(const float *, const float *, const float *,
                                float *, unsigned, unsigned, unsigned);
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  FILE *file = std::fopen(argv[1], "wb");
  if (!file)
    return 2;
  unsigned seed = 71, cases = 0;
  auto sample = [&]() {
    seed = seed * 1664525u + 1013904223u;
    return (float(seed >> 8) / 16777216.f - 0.5f) * 4.f;
  };
  for (unsigned width : {3u, 4u, 127u, 128u, 132u, 2048u})
    for (unsigned rows : {1u, 2u, 15u, 16u, 65u, 1024u})
      for (unsigned batch : {1u, 2u})
        for (bool bias : {false, true})
          for (bool inplace : {false, true}) {
            std::vector<float> x(size_t(batch) * rows * width), out(x.size());
            std::vector<float> w(3 * width), b(width);
            for (float &v : x)
              v = sample();
            for (float &v : w)
              v = sample();
            for (float &v : b)
              v = sample();
            float *dst = inplace ? x.data() : out.data();
            nntrainer::neon::causal_depthwise_conv1d_k3(
              x.data(), w.data(), bias ? b.data() : nullptr, dst, batch, rows,
              width);
            if (std::fwrite(dst, sizeof(float), x.size(), file) != x.size())
              return 1;
            ++cases;
          }
  std::fclose(file);
  for (unsigned rows : {1024u, 2048u}) {
    constexpr unsigned width = 2048;
    std::vector<float> x(size_t(rows) * width, 0.25f), out(x.size());
    std::vector<float> w(3 * width, 0.2f);
    std::vector<double> times;
    for (unsigned repeat = 0; repeat < 12; ++repeat) {
      const auto start = std::chrono::steady_clock::now();
      nntrainer::neon::causal_depthwise_conv1d_k3(x.data(), w.data(), nullptr,
                                                  out.data(), 1, rows, width);
      const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - start)
                          .count();
      if (repeat >= 2)
        times.push_back(us);
    }
    std::sort(times.begin(), times.end());
    std::printf("BENCH rows=%u median_us=%.3f\n", rows,
                times[times.size() / 2]);
  }
  std::printf("CONV %u fixtures completed; compare binary outputs\n", cases);
}
