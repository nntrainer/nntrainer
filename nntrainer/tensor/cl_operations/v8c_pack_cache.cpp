// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   v8c_pack_cache.cpp
 * @date   31 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Derive-once disk cache for the v8c GEMM weight pack (see header).
 */

#include "v8c_pack_cache.h"

#include <nntrainer_log.h>

#if !defined(_WIN32)

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace nntrainer {
namespace v8c_pack {
namespace {

// On-disk format ------------------------------------------------------------
// [header @0, 64B][blob regions ...][index @header.index_off]
// The header's index_off stays 0 until the finalize rename, so a temp file
// that never finalized (crash, IO error) can never validate.
// '02': the index record gained the source fingerprint, so a v01 pack (whose
// records cannot carry one) must be rejected rather than reinterpreted.
constexpr char kMagic[8] = {'N', 'T', 'V', '8', 'P', 'C', '0', '2'};
constexpr size_t kBlobStart = 4096; // header page reserved
// Upper bound on the index a pack may declare, so a corrupted count cannot
// drive a multi-gigabyte resize before the bounds check has a chance to run.
// One record per weight tensor; the largest models in scope are far below it.
constexpr uint32_t kMaxIndexRecords = 1u << 20;

struct FileHeader {
  char magic[8];
  uint64_t src_size;
  int64_t src_mtime_ns;
  uint64_t index_off;
  uint32_t index_count;
  uint32_t pad;
  uint64_t reserved[3];
};
static_assert(sizeof(FileHeader) == 64, "pack header must stay 64B");

struct IndexRecord {
  uint64_t name_fnv;
  uint32_t N;
  uint32_t K;
  uint64_t row_bytes;
  uint64_t payload_off;
  uint64_t payload_len;
  uint64_t rowsum_off;
  uint64_t rowsum_len;
  uint64_t payload_sample_fnv;
  uint64_t rowsum_fnv;
  uint64_t src_sample_fnv;
};
static_assert(sizeof(IndexRecord) == 80, "pack index record must stay 80B");

uint64_t fnv1a64(const void *data, size_t len,
                 uint64_t h = 1469598103934665603ull) {
  const uint8_t *p = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < len; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

// Sampled payload checksum: both 64KB ends + 16 evenly spaced 4KB interior
// pages. Bounded ~192KB per record regardless of payload size, so validation
// never costs the launch what the cache saves. Truncation is caught by the
// exact-length key; this guards header rot and gross corruption (an atomic
// temp+rename write path means torn files never validate in the first place).
uint64_t sample_fnv(const uint8_t *p, size_t len) {
  constexpr size_t kEnd = 64u << 10, kPage = 4096, kProbes = 16;
  uint64_t h = 1469598103934665603ull;
  if (len <= 2 * kEnd + kProbes * kPage)
    return fnv1a64(p, len, h);
  h = fnv1a64(p, kEnd, h);
  const size_t lo = kEnd, hi = len - kEnd;
  for (size_t i = 0; i < kProbes; ++i) {
    size_t off = lo + (hi - lo) * i / kProbes;
    off &= ~(kPage - 1);
    h = fnv1a64(p + off, kPage, h);
  }
  return fnv1a64(p + len - kEnd, kEnd, h);
}

bool cache_enabled() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_V8C_PACK_CACHE");
    return !(e && e[0] == '0');
  }();
  return on;
}

size_t min_payload_bytes() {
  static const size_t v = []() -> size_t {
    const char *e = std::getenv("NNTR_V8C_PACK_CACHE_MIN_MB");
    return (size_t)(e ? atol(e) : 64) << 20;
  }();
  return v;
}

struct Manager {
  std::mutex mtx;

  // active source identity
  std::string pack_path; // final pack location for the active source
  uint64_t src_size = 0;
  int64_t src_mtime_ns = 0;

  // read side (valid pack mapped)
  int map_fd = -1;
  uint8_t *map = nullptr;
  size_t map_len = 0;
  std::vector<IndexRecord> index;
  bool pack_valid = false;

  // write side (no valid pack -> rewrite armed)
  bool write_armed = false;
  int tmp_fd = -1;
  std::string tmp_path;
  uint64_t append_off = kBlobStart;
  std::vector<IndexRecord> pending;
  bool write_failed = false;

  std::thread finalizer;

  ~Manager() { join_finalizer(); }

  void join_finalizer() {
    if (finalizer.joinable())
      finalizer.join();
  }

  void drop_map() {
    if (map && map != MAP_FAILED)
      ::munmap(map, map_len);
    map = nullptr;
    map_len = 0;
    if (map_fd >= 0)
      ::close(map_fd);
    map_fd = -1;
    index.clear();
    pack_valid = false;
  }

  void drop_tmp(bool unlink_file) {
    if (tmp_fd >= 0)
      ::close(tmp_fd);
    tmp_fd = -1;
    if (unlink_file && !tmp_path.empty())
      ::unlink(tmp_path.c_str());
    tmp_path.clear();
    pending.clear();
    append_off = kBlobStart;
    write_failed = false;
  }
};

Manager &mgr() {
  static Manager m;
  return m;
}

// Fallback cache dir for read-only model dirs (same convention as the
// tokenizer snapshot cache): $XDG_CACHE_HOME|~/.cache + /nntrainer/v8cpack.
std::string fallback_pack_path(const std::string &src, uint64_t size,
                               int64_t mtime_ns) {
  const char *xdg = std::getenv("XDG_CACHE_HOME");
  std::string base;
  if (xdg && *xdg) {
    base = xdg;
  } else {
    const char *home = std::getenv("HOME");
    if (!home || !*home)
      return std::string();
    base = std::string(home) + "/.cache";
  }
  base += "/nntrainer";
  ::mkdir(base.c_str(), 0755);
  base += "/v8cpack";
  ::mkdir(base.c_str(), 0755);
  uint64_t h = fnv1a64(src.data(), src.size());
  h = fnv1a64(&size, sizeof(size), h);
  h = fnv1a64(&mtime_ns, sizeof(mtime_ns), h);
  char name[32];
  std::snprintf(name, sizeof(name), "/%016llx.v8cpack", (unsigned long long)h);
  return base + name;
}

// Validate + mmap a pack file for the active source. Returns true on success.
bool try_open_pack(Manager &m, const std::string &path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return false;
  struct stat st {};
  if (::fstat(fd, &st) != 0 || (size_t)st.st_size < sizeof(FileHeader)) {
    ::close(fd);
    return false;
  }
  const size_t len = (size_t)st.st_size;
  void *p = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) {
    ::close(fd);
    return false;
  }
  const uint8_t *base = static_cast<const uint8_t *>(p);
  FileHeader h{};
  std::memcpy(&h, base, sizeof(h));
  // Every offset and length below is read verbatim out of a file that lives
  // outside the trust boundary (next to the model, or under a cache directory
  // an environment variable names). Bounds are therefore checked in the
  // SUBTRACTION form: `off + len > file_len` wraps for a crafted or corrupted
  // pair whose sum exceeds 2^64 and would then pass, leaving the reads below
  // to run off the end of the mapping.
  const uint64_t flen = (uint64_t)len;
  const uint64_t index_bytes = (uint64_t)h.index_count * sizeof(IndexRecord);
  bool ok = std::memcmp(h.magic, kMagic, sizeof(kMagic)) == 0 &&
            h.src_size == m.src_size && h.src_mtime_ns == m.src_mtime_ns &&
            h.index_off >= kBlobStart && h.index_count > 0 &&
            h.index_count <= kMaxIndexRecords && index_bytes <= flen &&
            h.index_off <= flen - index_bytes;
  if (ok) {
    m.index.resize(h.index_count);
    std::memcpy(m.index.data(), base + h.index_off,
                h.index_count * sizeof(IndexRecord));
    for (const auto &r : m.index) {
      if (r.payload_off < kBlobStart || r.payload_len > flen ||
          r.payload_off > flen - r.payload_len || r.rowsum_off < kBlobStart ||
          r.rowsum_len > flen || r.rowsum_off > flen - r.rowsum_len) {
        ok = false;
        break;
      }
    }
  }
  if (!ok) {
    ::munmap(p, len);
    ::close(fd);
    m.index.clear();
    return false;
  }
  m.map_fd = fd;
  m.map = static_cast<uint8_t *>(p);
  m.map_len = len;
  m.pack_valid = true;
  // Kick readahead for the payloads: the giant builds run first in the load
  // (largest-first hand-out), so cold-page faults would sit on the critical
  // path otherwise.
  (void)::posix_madvise(p, len, POSIX_MADV_WILLNEED);
  return true;
}

} // namespace

struct RecordWriter {
  IndexRecord rec{};
  uint64_t base_off = 0;
  bool failed = false;
};

void set_source(const char *model_bin_path) {
  if (!cache_enabled() || !model_bin_path || !*model_bin_path)
    return;
  Manager &m = mgr();
  std::lock_guard<std::mutex> lock(m.mtx);
  m.join_finalizer();
  m.drop_map();
  m.drop_tmp(true);
  m.write_armed = false;
  m.pack_path.clear();

  struct stat st {};
  if (::stat(model_bin_path, &st) != 0)
    return;
  m.src_size = (uint64_t)st.st_size;
  m.src_mtime_ns =
    (int64_t)st.st_mtim.tv_sec * 1000000000ll + st.st_mtim.tv_nsec;

  const std::string primary = std::string(model_bin_path) + ".v8cpack";
  if (try_open_pack(m, primary)) {
    m.pack_path = primary;
  } else {
    const std::string fb =
      fallback_pack_path(model_bin_path, m.src_size, m.src_mtime_ns);
    if (!fb.empty() && try_open_pack(m, fb)) {
      m.pack_path = fb;
    } else {
      // no usable pack: arm a rewrite. Prefer next-to-model; fall back to the
      // cache dir when the model dir refuses the temp file.
      m.pack_path = primary;
      m.write_armed = true;
    }
  }
  ml_logi("v8c pack cache: %s (%s)",
          m.pack_valid ? "pack mapped" : "no pack, derive and write",
          m.pack_path.c_str());
}

uint64_t source_fingerprint(const void *data, size_t len) {
  return sample_fnv(static_cast<const uint8_t *>(data), len);
}

bool lookup(const char *name, unsigned int N, unsigned int K, size_t row_bytes,
            size_t payload_len, uint64_t src_fnv, Hit &out) {
  if (!cache_enabled() || !name || !*name)
    return false;
  Manager &m = mgr();
  if (!m.pack_valid || payload_len < min_payload_bytes())
    return false;
  const uint64_t nh = fnv1a64(name, std::strlen(name));
  for (const auto &r : m.index) {
    if (r.name_fnv != nh || r.N != N || r.K != K || r.row_bytes != row_bytes ||
        r.payload_len != payload_len || r.src_sample_fnv != src_fnv)
      continue;
    if (r.rowsum_len != (uint64_t)N * sizeof(int32_t))
      return false;
    const uint8_t *payload = m.map + r.payload_off;
    const uint8_t *rowsum = m.map + r.rowsum_off;
    if (sample_fnv(payload, payload_len) != r.payload_sample_fnv ||
        fnv1a64(rowsum, r.rowsum_len) != r.rowsum_fnv) {
      // corrupt record: silent per-record miss (derive as before). The rest
      // of the pack stays usable; no partial rewrite ever clobbers it.
      ml_logw("v8c pack cache: checksum mismatch for %s; deriving", name);
      return false;
    }
    out.payload = payload;
    out.rowsum = reinterpret_cast<const int32_t *>(rowsum);
    out.payload_len = payload_len;
    return true;
  }
  return false;
}

void payload_consumed(const Hit &hit) {
  if (!hit.payload || hit.payload_len == 0)
    return;
  const size_t page = 4096;
  uintptr_t lo = ((uintptr_t)hit.payload + page - 1) & ~(page - 1);
  uintptr_t hi = ((uintptr_t)hit.payload + hit.payload_len) & ~(page - 1);
  if (hi > lo)
    (void)::madvise((void *)lo, hi - lo, MADV_DONTNEED); // clean file pages
}

RecordWriter *begin_record(const char *name, unsigned int N, unsigned int K,
                           size_t row_bytes, size_t payload_len,
                           uint64_t src_fnv) {
  if (!cache_enabled() || !name || !*name)
    return nullptr;
  if (payload_len < min_payload_bytes())
    return nullptr;
  Manager &m = mgr();
  std::lock_guard<std::mutex> lock(m.mtx);
  if (!m.write_armed || m.write_failed)
    return nullptr;
  if (m.tmp_fd < 0) {
    // lazy temp creation, next-to-model first, cache dir second
    std::string t = m.pack_path + ".tmp." + std::to_string((long)::getpid());
    int fd = ::open(t.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0644);
    if (fd < 0) {
      const std::string fb = fallback_pack_path(
        m.pack_path.substr(0, m.pack_path.size() - 8 /* ".v8cpack" */),
        m.src_size, m.src_mtime_ns);
      if (!fb.empty()) {
        t = fb + ".tmp." + std::to_string((long)::getpid());
        fd = ::open(t.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0644);
        if (fd >= 0)
          m.pack_path = fb;
      }
      if (fd < 0) {
        m.write_failed = true;
        return nullptr;
      }
    }
    m.tmp_fd = fd;
    m.tmp_path = t;
    m.append_off = kBlobStart;
  }
  auto *rw = new RecordWriter();
  rw->rec.name_fnv = fnv1a64(name, std::strlen(name));
  rw->rec.src_sample_fnv = src_fnv;
  rw->rec.N = N;
  rw->rec.K = K;
  rw->rec.row_bytes = row_bytes;
  rw->rec.payload_len = payload_len;
  rw->rec.rowsum_len = (uint64_t)N * sizeof(int32_t);
  rw->rec.payload_off = m.append_off;
  rw->rec.rowsum_off = m.append_off + payload_len;
  // page-align the next record's region so DONTNEED trims stay per-record
  m.append_off = (rw->rec.rowsum_off + rw->rec.rowsum_len + 4095) & ~4095ull;
  rw->base_off = rw->rec.payload_off;
  return rw;
}

void record_write(RecordWriter *rw, size_t payload_off, const void *data,
                  size_t len) {
  if (!rw || rw->failed)
    return;
  Manager &m = mgr();
  const int fd = m.tmp_fd; // stable while records are in flight
  if (fd < 0) {
    rw->failed = true;
    return;
  }
  const uint8_t *p = static_cast<const uint8_t *>(data);
  size_t off = rw->base_off + payload_off;
  while (len > 0) {
    ssize_t w = ::pwrite(fd, p, len, (off_t)off);
    if (w <= 0) {
      rw->failed = true;
      return;
    }
    p += w;
    off += (size_t)w;
    len -= (size_t)w;
  }
}

void commit_record(RecordWriter *rw, const int32_t *rowsum, size_t count) {
  if (!rw)
    return;
  Manager &m = mgr();
  bool ok = !rw->failed && rowsum && count == rw->rec.N;
  if (ok) {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(rowsum);
    size_t len = rw->rec.rowsum_len, off = rw->rec.rowsum_off;
    while (len > 0) {
      ssize_t w = ::pwrite(m.tmp_fd, p, len, (off_t)off);
      if (w <= 0) {
        ok = false;
        break;
      }
      p += w;
      off += (size_t)w;
      len -= (size_t)w;
    }
  }
  if (ok) {
    // checksum from the file (page-cache hot: we just wrote it)
    std::vector<uint8_t> buf(
      rw->rec.payload_len < (192u << 10) ? rw->rec.payload_len : (192u << 10));
    // read back through a bounded window for the sampled fnv
    // (small payloads: exact; large: same sampling as lookup)
    if (rw->rec.payload_len <= buf.size()) {
      ok = ::pread(m.tmp_fd, buf.data(), rw->rec.payload_len,
                   (off_t)rw->rec.payload_off) == (ssize_t)rw->rec.payload_len;
      if (ok)
        rw->rec.payload_sample_fnv =
          sample_fnv(buf.data(), rw->rec.payload_len);
    } else {
      // mmap the record region read-only; simpler than replaying the sampling
      // arithmetic through pread windows, and dropped right after.
      const size_t page = 4096;
      const size_t moff = rw->rec.payload_off & ~(page - 1);
      const size_t shift = rw->rec.payload_off - moff;
      const size_t mlen = shift + rw->rec.payload_len;
      void *p =
        ::mmap(nullptr, mlen, PROT_READ, MAP_PRIVATE, m.tmp_fd, (off_t)moff);
      if (p == MAP_FAILED) {
        ok = false;
      } else {
        rw->rec.payload_sample_fnv = sample_fnv(
          static_cast<const uint8_t *>(p) + shift, rw->rec.payload_len);
        ::munmap(p, mlen);
      }
    }
  }
  if (ok)
    rw->rec.rowsum_fnv = fnv1a64(rowsum, rw->rec.rowsum_len);
  {
    std::lock_guard<std::mutex> lock(m.mtx);
    if (ok)
      m.pending.push_back(rw->rec);
    else
      m.write_failed = true;
  }
  delete rw;
}

void abort_record(RecordWriter *rw) {
  if (!rw)
    return;
  delete rw; // its blob region is simply never indexed
}

void load_complete() {
  Manager &m = mgr();
  std::lock_guard<std::mutex> lock(m.mtx);
  if (m.tmp_fd < 0)
    return;
  if (m.write_failed || m.pending.empty()) {
    m.drop_tmp(true);
    return;
  }
  // Hand the finalize (index + header + fsync + rename) to a background
  // thread: it costs a dirty-page writeback of the whole pack, which has no
  // business on the first-token path. The thread owns copies/fd; it touches
  // no model memory, so lifetime is trivial. Joined in the Manager dtor at
  // exit (a detached writer's rename never lands when a one-shot CLI exits
  // first -- same lesson as the tokenizer snapshot writer).
  m.join_finalizer();
  int fd = m.tmp_fd;
  std::string tmp = m.tmp_path;
  std::string final_path = m.pack_path;
  std::vector<IndexRecord> recs;
  recs.swap(m.pending);
  uint64_t index_off = m.append_off;
  uint64_t src_size = m.src_size;
  int64_t src_mtime_ns = m.src_mtime_ns;
  m.tmp_fd = -1; // ownership moved to the finalizer
  m.tmp_path.clear();
  m.finalizer = std::thread([fd, tmp, final_path, recs = std::move(recs),
                             index_off, src_size, src_mtime_ns]() {
    bool ok = true;
    {
      const uint8_t *p = reinterpret_cast<const uint8_t *>(recs.data());
      size_t len = recs.size() * sizeof(IndexRecord), off = index_off;
      while (ok && len > 0) {
        ssize_t w = ::pwrite(fd, p, len, (off_t)off);
        if (w <= 0)
          ok = false;
        else {
          p += w;
          off += (size_t)w;
          len -= (size_t)w;
        }
      }
    }
    if (ok) {
      FileHeader h{};
      std::memcpy(h.magic, kMagic, sizeof(kMagic));
      h.src_size = src_size;
      h.src_mtime_ns = src_mtime_ns;
      h.index_off = index_off;
      h.index_count = (uint32_t)recs.size();
      ok = ::pwrite(fd, &h, sizeof(h), 0) == (ssize_t)sizeof(h);
    }
    if (ok)
      ok = ::fsync(fd) == 0;
    ::close(fd);
    if (ok)
      ok = ::rename(tmp.c_str(), final_path.c_str()) == 0;
    if (!ok)
      ::unlink(tmp.c_str());
    if (ok)
      ml_logi("v8c pack cache: wrote %zu records", recs.size());
    else
      ml_logw("v8c pack cache: write failed; the next launch derives again");
  });
}

} // namespace v8c_pack
} // namespace nntrainer

#else /* _WIN32: no-op stubs (POSIX file plumbing; the derive path is the      \
         behavior with the cache absent, which is exactly the fallback) */

namespace nntrainer {
namespace v8c_pack {
struct RecordWriter {};
void set_source(const char *) {}
void load_complete() {}
bool lookup(const char *, unsigned int, unsigned int, size_t, size_t, uint64_t,
            Hit &) {
  return false;
}
uint64_t source_fingerprint(const void *, size_t) { return 0; }
void payload_consumed(const Hit &) {}
RecordWriter *begin_record(const char *, unsigned int, unsigned int, size_t,
                           size_t, uint64_t) {
  return nullptr;
}
void record_write(RecordWriter *, size_t, const void *, size_t) {}
void commit_record(RecordWriter *, const int32_t *, size_t) {}
void abort_record(RecordWriter *) {}
} // namespace v8c_pack
} // namespace nntrainer

#endif
