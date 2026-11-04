// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qnn_row_readahead.cpp
 * @date   06 September 2026
 * @brief  Batched page-cache warm-up for token-indexed sidecar tables
 * @bug    No known bugs except for NYI items
 */

#include "qnn_row_readahead.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace causallm {

namespace {

size_t page_size() {
#if defined(__linux__) || defined(__ANDROID__)
  const long size = ::sysconf(_SC_PAGESIZE);
  return size > 0 ? static_cast<size_t>(size) : 4096;
#else
  return 4096;
#endif
}
constexpr size_t kDefaultCapBytes = 32u * 1024u * 1024u;

bool trace_on() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_QNN_TRACE");
    return e != nullptr && e[0] != '0';
  }();
  return on;
}

void report(const char *tag, const RowReadaheadStats &s, size_t distinct_seen) {
  if (!trace_on())
    return;
  if (s.capped) {
    std::cout << "[RA] " << tag << " skipped: " << distinct_seen
              << " rows would exceed cap " << (row_readahead_cap_bytes() >> 20)
              << " MiB" << std::endl;
    return;
  }
  std::cout << "[RA] " << tag << " rows=" << s.rows << " bytes=" << s.bytes
            << " pages<=" << s.pages << std::endl;
}

/**
 * Walk `tokens` in order, calling `emit(row_index)` once per distinct valid id.
 * Returns the stats; `emit` is only called when the batch is under the cap, so
 * the walk is done twice (a count pass, then an issue pass) rather than
 * issuing I/O we might have to regret.
 */
template <typename Emit>
RowReadaheadStats walk_rows(const int *tokens, size_t count, size_t row_bytes,
                            size_t row_count, Emit &&emit) {
  RowReadaheadStats st;
  if (tokens == nullptr || count == 0 || row_bytes == 0 || row_count == 0)
    return st;

  std::vector<bool> seen(row_count, false);
  size_t distinct = 0;
  for (size_t i = 0; i < count; ++i) {
    const int id = tokens[i];
    if (id < 0 || static_cast<size_t>(id) >= row_count)
      continue;
    if (seen[static_cast<size_t>(id)])
      continue;
    seen[static_cast<size_t>(id)] = true;
    ++distinct;
  }
  if (distinct == 0)
    return st;

  if (distinct > row_readahead_cap_bytes() / row_bytes) {
    st.capped = true;
    st.rows = distinct;
    return st;
  }

  // Second pass in prompt order; `seen` is reused as a "not yet issued" map.
  for (size_t i = 0; i < count; ++i) {
    const int id = tokens[i];
    if (id < 0 || static_cast<size_t>(id) >= row_count)
      continue;
    const size_t row = static_cast<size_t>(id);
    if (!seen[row])
      continue;
    seen[row] = false;
    emit(row);
    st.rows += 1;
    st.bytes += row_bytes;
  }
  return st;
}

} // namespace

bool row_readahead_enabled() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_PLE_PREFILL_RA");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

size_t row_readahead_cap_bytes() {
  static const size_t cap = []() -> size_t {
    const char *e = std::getenv("NNTR_PLE_PREFILL_RA_CAP_MIB");
    if (e == nullptr)
      return kDefaultCapBytes;
    const long v = std::strtol(e, nullptr, 10);
    if (v <= 0)
      return kDefaultCapBytes;
    return static_cast<size_t>(v) * 1024u * 1024u;
  }();
  return cap;
}

RowReadaheadStats row_readahead_file(int fd, const int *tokens, size_t count,
                                     size_t row_bytes, size_t row_count,
                                     size_t base_offset, const char *tag) {
  RowReadaheadStats st;
  if (fd < 0)
    return st;

  const size_t kPageSize = page_size();
  size_t pages = 0;
  st = walk_rows(tokens, count, row_bytes, row_count, [&](size_t row) {
    const size_t off = base_offset + row * row_bytes;
#if defined(__linux__) || defined(__ANDROID__)
    (void)::readahead(fd, static_cast<off_t>(off), row_bytes);
#endif
    const size_t first = static_cast<size_t>(off) / kPageSize;
    const size_t last = (static_cast<size_t>(off) + row_bytes - 1) / kPageSize;
    pages += last - first + 1;
  });
  st.pages = pages;
  report(tag, st, st.rows);
  return st;
}

RowReadaheadStats row_readahead_map(const void *base, size_t map_size,
                                    const int *tokens, size_t count,
                                    size_t row_bytes, size_t row_count,
                                    size_t base_offset, const char *tag) {
  RowReadaheadStats st;
  if (base == nullptr)
    return st;

  const size_t kPageSize = page_size();
  size_t pages = 0;
  st = walk_rows(tokens, count, row_bytes, row_count, [&](size_t row) {
    const size_t off = base_offset + row * row_bytes;
    if (off + row_bytes > map_size)
      return;
    const size_t start = off & ~(kPageSize - 1);
    const size_t end = (off + row_bytes + kPageSize - 1) & ~(kPageSize - 1);
    void *addr = const_cast<char *>(static_cast<const char *>(base)) + start;
#if defined(MADV_WILLNEED)
    (void)::madvise(addr, end - start, MADV_WILLNEED);
#elif defined(POSIX_MADV_WILLNEED)
    (void)::posix_madvise(addr, end - start, POSIX_MADV_WILLNEED);
#else
    (void)addr;
#endif
    pages += (end - start) / kPageSize;
  });
  st.pages = pages;
  report(tag, st, st.rows);
  return st;
}

} // namespace causallm
