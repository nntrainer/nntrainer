// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file weight_file_reader.cpp
 * @brief Bounded file reads for accelerator weights, including unaligned direct
 * IO.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <weight_file_reader.h>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace nntrainer {

namespace {
thread_local WeightReadTiming *read_timing = nullptr;
}

WeightReadTiming *weightSetReadTiming(WeightReadTiming *timing) {
  auto *previous = read_timing;
  read_timing = timing;
  return previous;
}

/** -1 preserves the existing loader; 0 bypasses the page cache; 1 bounds it. */
int weightPageCachePolicy() {
  const char *value = std::getenv("NNTR_HTP_PAGE_CACHE");
  if (!value)
    return -1;
  if (std::strcmp(value, "0") == 0)
    return 0;
  if (std::strcmp(value, "1") == 0)
    return 1;
  throw std::invalid_argument("NNTR_HTP_PAGE_CACHE must be 0 or 1");
}

#if defined(__linux__)
/** Open a separate direct-IO description: never change flags on another reader.
 */
int openWeightFile(const char *path) {
  int flags = O_RDONLY | O_CLOEXEC;
  if (weightPageCachePolicy() == 0)
    flags |= O_DIRECT;
  int fd = ::open(path, flags);
  if (fd < 0)
    throw std::runtime_error(std::string("Cannot open weight file: ") +
                             std::strerror(errno));
  if (weightPageCachePolicy() >= 0) {
    // Discard old, clean pages for this model only, including prior runs.
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
  }
  return fd;
}

/** Keep at most 2 GB of recently read model-file ranges in the kernel cache.
 * Advice is per file, never a system-wide drop_caches. The kernel can retain
 * pages mapped by other users, so actual residency must also be measured.
 */
class WeightPageCache {
  using File = std::pair<dev_t, ino_t>;
  using Key = std::tuple<dev_t, ino_t, uint64_t>;
  static constexpr size_t grain = 1u << 20;
  static constexpr size_t limit = 2000000000ull / grain;
  std::mutex mutex_;
  struct FileEntry {
    int fd;
    size_t blocks;
  };
  std::map<File, FileEntry> files_;
  std::list<Key> order_;
  std::map<Key, std::list<Key>::iterator> entries_;

public:
  ~WeightPageCache() {
    for (auto &f : files_)
      ::close(f.second.fd);
  }
  static WeightPageCache &get() {
    static WeightPageCache cache;
    return cache;
  }
  void touch(int fd, uint64_t offset, size_t bytes) {
    if (!bytes)
      return;
    struct stat st;
    if (::fstat(fd, &st))
      throw std::runtime_error("Cannot identify weight cache file");
    std::lock_guard<std::mutex> lock(mutex_);
    File file{st.st_dev, st.st_ino};
    if (!files_.count(file)) {
      int copy = ::dup(fd);
      if (copy < 0)
        throw std::runtime_error("Cannot retain weight cache fd");
      files_.emplace(file, FileEntry{copy, 0});
    }
    uint64_t last = (offset + bytes - 1) / grain;
    for (uint64_t block = offset / grain; block <= last; ++block) {
      Key key{st.st_dev, st.st_ino, block};
      auto old = entries_.find(key);
      if (old != entries_.end()) {
        order_.splice(order_.end(), order_, old->second);
      } else {
        order_.push_back(key);
        entries_.emplace(key, std::prev(order_.end()));
        ++files_.at(file).blocks;
      }
      while (entries_.size() > limit) {
        Key victim = order_.front();
        auto old_file = files_.find({std::get<0>(victim), std::get<1>(victim)});
        ::posix_fadvise(old_file->second.fd, std::get<2>(victim) * grain, grain,
                        POSIX_FADV_DONTNEED);
        if (--old_file->second.blocks == 0) {
          ::close(old_file->second.fd);
          files_.erase(old_file);
        }
        entries_.erase(victim);
        order_.pop_front();
      }
    }
  }
};

/**
 * Read exactly a range. Direct IO uses a bounded, aligned bounce buffer,
 * including for DMA mappings that cannot be pinned by the filesystem. The
 * file's final block may be short, but every requested byte must exist. Returns
 * errno, or ENODATA for a short file. No buffered fallback: callers must not
 * silently enable the page cache.
 */
int weightPreadScatter(int fd, const WeightReadSlice *slices, size_t count,
                       uint64_t offset) {
  constexpr size_t max_slices = 4;
  if (!slices || !count || count > max_slices)
    return EINVAL;
  const uint64_t max_offset = std::numeric_limits<off_t>::max();
  if (offset > max_offset)
    return EOVERFLOW;
  size_t bytes = 0;
  iovec parts[max_slices];
  for (size_t i = 0; i < count; ++i) {
    if (slices[i].bytes > std::numeric_limits<size_t>::max() - bytes ||
        slices[i].bytes > max_offset - offset - bytes)
      return EOVERFLOW;
    if (!slices[i].data && slices[i].bytes)
      return EINVAL;
    parts[i] = {slices[i].data, slices[i].bytes};
    bytes += slices[i].bytes;
  }
  int flags = ::fcntl(fd, F_GETFL);
  if (flags < 0)
    return errno;
  size_t part = 0;
  // Advance both destinations after a buffered read, or scatter the direct
  // read's scratch buffer. Metadata never has to be read back from DMA memory.
  auto consume = [&](const uint8_t *source, size_t count) {
    while (count) {
      while (!parts[part].iov_len)
        ++part;
      const size_t take = std::min(count, parts[part].iov_len);
      if (source) {
        std::memcpy(parts[part].iov_base, source, take);
        source += take;
      }
      parts[part].iov_base =
        static_cast<uint8_t *>(parts[part].iov_base) + take;
      parts[part].iov_len -= take;
      count -= take;
    }
  };
  if (!(flags & O_DIRECT)) {
    if (weightPageCachePolicy() == 0)
      return EINVAL;
    while (bytes) {
      iovec batch[max_slices];
      int batch_count = 0;
      size_t left = std::min(bytes, size_t(1) << 20);
      for (size_t i = part; i < count && left; ++i) {
        const size_t take = std::min(left, parts[i].iov_len);
        if (take) {
          batch[batch_count++] = {parts[i].iov_base, take};
          left -= take;
        }
      }
      ssize_t got = ::preadv(fd, batch, batch_count, offset);
      if (got < 0) {
        if (errno == EINTR)
          continue;
        return errno;
      }
      if (!got)
        return ENODATA;
      if (weightPageCachePolicy() == 1)
        WeightPageCache::get().touch(fd, offset, static_cast<size_t>(got));
      consume(nullptr, static_cast<size_t>(got));
      bytes -= static_cast<size_t>(got);
      offset += static_cast<size_t>(got);
    }
    return 0;
  }
  // 4 KiB meets the block-device and filesystem direct-IO alignment used by
  // Android. Larger VM page sizes do not change the on-disk sector alignment.
  // A 2 MiB window also holds a native TL2 expert's adjacent gate/up and
  // down records, including metadata, so confirmed misses need only one IO.
  constexpr size_t alignment = WEIGHT_DIRECT_ALIGNMENT;
  constexpr size_t capacity = WEIGHT_DIRECT_WINDOW_BYTES;
  struct Scratch {
    void *data = nullptr;
    ~Scratch() { std::free(data); }
  };
  thread_local Scratch scratch;
  while (bytes) {
    const size_t skip = offset % alignment;
    const size_t take = std::min(bytes, capacity - skip);
    const size_t request =
      (skip + take + alignment - 1) / alignment * alignment;
    // FastRPC mappings can be CPU-addressable without supporting the page
    // pinning required by direct IO (pread then returns EFAULT). Always read
    // through bounded ordinary memory, including aligned DMA destinations.
    if (!scratch.data) {
      int rc = ::posix_memalign(&scratch.data, alignment, capacity);
      if (rc)
        return rc;
    }
    void *target = scratch.data;
    const auto start = read_timing ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    ssize_t got;
    do {
      got = ::pread(fd, target, request, static_cast<off_t>(offset - skip));
    } while (got < 0 && errno == EINTR);
    const int read_errno = got < 0 ? errno : 0;
    if (read_timing) {
      read_timing->io_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
      ++read_timing->calls;
      read_timing->requested_bytes += request;
    }
    if (got < 0)
      return read_errno;
    if (static_cast<size_t>(got) < skip + take)
      return ENODATA;
    const auto copy_start = read_timing
                              ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    consume(static_cast<uint8_t *>(target) + skip, take);
    if (read_timing)
      read_timing->copy_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - copy_start)
          .count();
    offset += take;
    bytes -= take;
  }
  return 0;
}
#else
int openWeightFile(const char *) {
  throw std::runtime_error("NNTR_HTP_PAGE_CACHE requires Linux/Android");
}
int weightPreadScatter(int, const WeightReadSlice *, size_t, uint64_t) {
  return ENOSYS;
}
#endif

int weightPreadSplit(int fd, void *dst, size_t bytes, void *tail,
                     size_t tail_bytes, uint64_t offset) {
  const WeightReadSlice slices[] = {{dst, bytes}, {tail, tail_bytes}};
  return weightPreadScatter(fd, slices, 2, offset);
}

int weightPread(int fd, void *dst, size_t bytes, uint64_t offset) {
  return weightPreadSplit(fd, dst, bytes, nullptr, 0, offset);
}

} // namespace nntrainer
