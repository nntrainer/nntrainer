// SPDX-License-Identifier: Apache-2.0
/** @file cpu_memory_hints.h @brief Optional hints for large host allocations.
 */
#ifndef NNTRAINER_CPU_MEMORY_HINTS_H
#define NNTRAINER_CPU_MEMORY_HINTS_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#if defined(__linux__) && defined(__x86_64__)
#include <sys/mman.h>
#endif

namespace nntrainer {
/** Suggest transparent huge pages for complete 2 MiB pages inside an owned
 * allocation. This never changes system policy or touches adjacent allocations.
 * Other platforms and an unavailable hint retain the normal allocation. */
inline void hintCpuHugePages(void *memory, size_t bytes) {
#if defined(__linux__) && defined(__x86_64__)
  static const bool enabled = [] {
    const char *value = std::getenv("NNTR_CPU_HUGE_PAGES");
    return value && value[0] == '1';
  }();
  constexpr uintptr_t page = 2 * 1024 * 1024;
  if (!enabled || !memory || bytes < 2 * page)
    return;
  const auto address = reinterpret_cast<uintptr_t>(memory);
  const auto begin = (address + page - 1) & ~(page - 1);
  const auto end = (address + bytes) & ~(page - 1);
  if (end > begin)
    madvise(reinterpret_cast<void *>(begin), end - begin, MADV_HUGEPAGE);
#else
  (void)memory;
  (void)bytes;
#endif
}
} // namespace nntrainer
#endif
