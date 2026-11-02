// SPDX-License-Identifier: Apache-2.0
/** @file htp_worker_policy.h
 * @brief Fixed per-process HTP reader configuration and thread placement.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_HTP_WORKER_POLICY_H
#define NNTR_HTP_WORKER_POLICY_H
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif
namespace nntrainer {
class HtpWorkerPolicy {
public:
  struct PrefetchKnobs {
    size_t readers = 6;
    std::vector<int> cpus;
  };
  static const PrefetchKnobs &prefetchKnobs() {
    static const PrefetchKnobs k = [] {
      PrefetchKnobs r;
      if (const char *v = std::getenv("NNTR_MOE_PREFETCH_READERS"))
        r.readers = std::max<size_t>(1, std::strtoul(v, nullptr, 10));
      if (const char *v = std::getenv("NNTR_MOE_PREFETCH_CPUS")) {
        for (const char *p = v; *p != '\0';) {
          char *end = nullptr;
          const long c = std::strtol(p, &end, 10);
          if (end == p)
            break;
          if (c >= 0 && c < 32)
            r.cpus.push_back(static_cast<int>(c));
          p = (*end == ',') ? end + 1 : end;
        }
      }
      return r;
    }();
    return k;
  }
  /** Match the prefetch worker policy without inheriting a caller pinned
   *  to a single compute core. An explicit prefetch CPU list is retained. */
  static std::vector<int> backgroundCpus(uint32_t caller) {
    auto cpus = prefetchKnobs().cpus;
#if defined(__linux__)
    if (cpus.empty() && caller) {
      const long n = sysconf(_SC_NPROCESSORS_CONF);
      for (int c = 0; c < n && c < 32; ++c)
        if ((caller & (1u << c)) == 0)
          cpus.push_back(c);
    }
#else
    (void)caller;
#endif
    return cpus;
  }
  /** @brief This thread's core as a bit, 0 where that is not known. */
  static uint32_t cpuBit() {
#if defined(__linux__)
    const int c = sched_getcpu();
    return (c >= 0 && c < 32) ? (1u << c) : 0u;
#else
    return 0u;
#endif
  }
  static void pinToCpus(const std::vector<int> &cpus) {
#if defined(__linux__)
    if (cpus.empty())
      return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
      CPU_SET(c, &set);
    sched_setaffinity(0, sizeof(set), &set);
#else
    (void)cpus;
#endif
  }
};
} // namespace nntrainer
#endif
