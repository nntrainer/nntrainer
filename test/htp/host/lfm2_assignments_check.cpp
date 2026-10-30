// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_assignments_check.cpp
 * @brief Exact assignment order and weights across the parallel threshold.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug No known bugs.
 */
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace nntrainer {
struct Tensor {
  std::vector<float> data;
  template <typename T> const T *getData() const { return data.data(); }
};
struct ThreadManager {
  unsigned count = 1;
  static ThreadManager &Global() {
    static ThreadManager tm;
    return tm;
  }
  unsigned getComputeThreadCount() const { return count; }
  template <typename F> void parallel_for(size_t begin, size_t end, F f) {
    std::vector<std::thread> threads;
    for (size_t i = begin; i < end; ++i)
      threads.emplace_back(f, i);
    for (auto &thread : threads)
      thread.join();
  }
};
} // namespace nntrainer
using Assignments = std::vector<std::vector<std::pair<unsigned, float>>>;
static constexpr bool NORM_TOPK_PROB = true;
static constexpr float ROUTED_SCALING_FACTOR = 1.0f;
static constexpr unsigned EXTRA_TOPK = 5;
struct Layer {
  unsigned topk, num_experts;
  bool softmax_router;
  void reference(const nntrainer::Tensor &, const nntrainer::Tensor &, unsigned,
                 Assignments &, std::vector<int> *);
  void candidate(const nntrainer::Tensor &, const nntrainer::Tensor &, unsigned,
                 Assignments &, std::vector<int> *);
};
#include "candidate.inc"
#include "reference.inc"
int main() {
  unsigned cases = 0;
  for (unsigned workers : {1u, 3u, 8u})
    for (unsigned rows : {1u, 127u, 128u, 129u, 1024u})
      for (unsigned experts : {32u, 64u, 128u})
        for (unsigned top : {1u, 4u, 8u})
          for (bool softmax : {false, true})
            for (unsigned pattern = 0; pattern < 4; ++pattern) {
              nntrainer::ThreadManager::Global().count = workers;
              nntrainer::Tensor x, bias;
              x.data.resize(size_t(rows) * experts);
              bias.data.resize(experts);
              for (unsigned e = 0; e < experts; ++e)
                bias.data[e] = softmax        ? 0.125f * (1 + e % 13)
                               : pattern == 2 ? 0.0f
                                              : (int(e % 7) - 3) * 0.015625f;
              for (size_t i = 0; i < x.data.size(); ++i)
                x.data[i] =
                  pattern == 0   ? 0.0f
                  : pattern == 1 ? (int(i % 67) - 33) * 0.25f
                  : pattern == 2
                    ? (i % 2 ? 100.0f : -100.0f)
                    : (int((i * 1664525u + 1013904223u) % 65537) - 32768) /
                        4096.0f;
              Layer layer{top, experts, softmax};
              for (bool hints : {false, true}) {
                Assignments want(experts), got(experts);
                std::vector<int> want_hints{719}, got_hints{719};
                layer.reference(x, bias, rows, want,
                                hints ? &want_hints : nullptr);
                layer.candidate(x, bias, rows, got,
                                hints ? &got_hints : nullptr);
                assert(want_hints == got_hints);
                for (unsigned e = 0; e < experts; ++e) {
                  assert(want[e].size() == got[e].size());
                  for (size_t i = 0; i < want[e].size(); ++i) {
                    assert(want[e][i].first == got[e][i].first);
                    assert(std::memcmp(&want[e][i].second, &got[e][i].second,
                                       sizeof(float)) == 0);
                  }
                }
                ++cases;
              }
            }
  std::printf("Assignments exact row order/weights/ties/hints: %u cases PASS\n",
              cases);
}
