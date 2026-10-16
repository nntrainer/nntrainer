// SPDX-License-Identifier: Apache-2.0
/** @file htp_prepared_tl2.h
 * @brief Offline symmetric QS2 / TL2 weights, one representation per slot.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __HTP_PREPARED_TL2_H__
#define __HTP_PREPARED_TL2_H__
#include "htp_prepared_qs2.h"
#include "htp_tl2_opts.h"
#include <atomic>
#include <cstdio>
#include <thread>
#if defined(__ANDROID__) || defined(__linux__)
#include <sys/uio.h>
#endif

namespace nntrainer {
#if defined(__ANDROID__) || defined(__linux__)
/** NNTRTL21: 64-byte header followed by 48-byte entries, little endian.
 * Both payloads have arena-ready scales, integer sums and a symmetric
 * {-1,0,1} or legacy {-7,0,7} palette, which records the dot-product magnitude.
 * Without a deployment receipt, source CRCs are checked on first use and
 * payload CRCs on every read. Receipts bind full validation to unchanged
 * files. The model and artifact must remain immutable during a run.
 * CRC is accidental corruption detection, not authentication. */
class PreparedTl2 {
  struct Entry {
    uint64_t source, qs2, tl2;
    uint32_t K, N, qs2_crc, tl2_crc, source_crc;
    std::atomic<bool> verified{false};
  };
  int fd_ = -1;
  PreparedQs2Stamp source_stamp_;
  std::unique_ptr<PreparedQs2Stamp> stamp_;
  std::unordered_map<uint64_t, std::unique_ptr<Entry>> entries_;
  bool receipt_ = false;
  const bool direct_read_ = [] {
    const char *v = std::getenv("NNTR_HTP_TL2_DIRECT_READ");
    return !v || std::strcmp(v, "0") != 0;
  }();

  // Native codes and the arena tail are separated by unused QS2 capacity.
  // Scatter the contiguous file record directly into these destinations.
  static void readScatter(int fd, uint64_t off, iovec *parts, int count) {
    while (count) {
      const ssize_t n = preadv(fd, parts, count, static_cast<off_t>(off));
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
        throw std::runtime_error("prepared TL2: incomplete scatter read");
      off += static_cast<uint64_t>(n);
      size_t left = static_cast<size_t>(n);
      while (count && left >= parts->iov_len) {
        left -= parts->iov_len;
        ++parts;
        --count;
      }
      if (count && left) {
        parts->iov_base = static_cast<uint8_t *>(parts->iov_base) + left;
        parts->iov_len -= left;
      }
    }
  }
  std::thread discard_thread_;
  static void put64(uint8_t *p, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i)
      p[i] = uint8_t(value >> (8u * i));
  }
  std::array<uint8_t, 128> receiptBytes() const {
    std::array<uint8_t, 128> data{};
    std::memcpy(data.data(), "TL2V0001", 8);
    const auto source = source_stamp_.words(), artifact = stamp_->words();
    for (unsigned i = 0; i < 7; ++i) {
      put64(data.data() + 8u + 8u * i, source[i]);
      put64(data.data() + 64u + 8u * i, artifact[i]);
    }
    put64(data.data() + 120, preparedQs2Crc(data.data(), 120));
    return data;
  }

public:
  PreparedTl2(const char *path, int source_fd, bool allow_receipt = true) :
    source_stamp_(source_fd) {
    fd_ = open(path, O_RDONLY | O_CLOEXEC);
    if (fd_ < 0)
      throw std::runtime_error("prepared TL2: cannot open artifact");
    try {
      stamp_.reset(new PreparedQs2Stamp(fd_));
      uint8_t header[64];
      preparedQs2Read(fd_, header, sizeof(header), 0);
      const uint32_t count = preparedFcU32(header + 12);
      if (std::memcmp(header, "NNTRTL21", 8) ||
          preparedFcU32(header + 8) != 1u || !count || count > 65536u ||
          preparedFcU64(header + 16) !=
            static_cast<uint64_t>(source_stamp_.value.st_size) ||
          preparedFcU64(header + 24) !=
            static_cast<uint64_t>(stamp_->value.st_size))
        throw std::runtime_error("prepared TL2: invalid header or source size");
      std::vector<uint8_t> index(size_t(count) * 48u);
      preparedQs2Read(fd_, index.data(), index.size(), 64u);
      if (preparedQs2Crc(index.data(), index.size()) !=
          preparedFcU32(header + 32))
        throw std::runtime_error("prepared TL2: corrupt index");
      uint64_t end = 64u + index.size();
      for (uint32_t i = 0; i < count; ++i) {
        const uint8_t *p = index.data() + size_t(i) * 48u;
        std::unique_ptr<Entry> e(new Entry);
        e->source = preparedFcU64(p);
        e->K = preparedFcU32(p + 8);
        e->N = preparedFcU32(p + 12);
        e->qs2 = preparedFcU64(p + 16);
        e->tl2 = preparedFcU64(p + 24);
        e->qs2_crc = preparedFcU32(p + 32);
        e->tl2_crc = preparedFcU32(p + 36);
        e->source_crc = preparedFcU32(p + 40);
        const uint64_t codes = uint64_t(e->K) * e->N / 4u;
        const uint64_t native = htp_tl2_native_bytes(e->K, e->N);
        const uint64_t tail = 8u * e->N + 4u;
        if (!e->K || !e->N || e->K % 32u || e->N % 32u || e->K > 8192u ||
            e->N > 8192u || native > codes ||
            e->source > uint64_t(source_stamp_.value.st_size) ||
            codes + tail > uint64_t(source_stamp_.value.st_size) - e->source ||
            e->qs2 < end || e->tl2 < e->qs2 || e->tl2 - e->qs2 < codes + tail ||
            e->tl2 > uint64_t(stamp_->value.st_size) ||
            native + tail > uint64_t(stamp_->value.st_size) - e->tl2)
          throw std::runtime_error("prepared TL2: invalid weight bounds");
        end = e->tl2 + native + tail;
        const uint64_t key = e->source;
        if (!entries_.emplace(key, std::move(e)).second)
          throw std::runtime_error("prepared TL2: duplicate source offset");
      }
      if (allow_receipt) {
        std::ifstream file(std::string(path) + ".validated",
                           std::ios::binary | std::ios::ate);
        std::array<uint8_t, 128> data{};
        if (file && file.tellg() == 128) {
          file.seekg(0);
          receipt_ = bool(file.read(reinterpret_cast<char *>(data.data()),
                                    data.size())) &&
                     data == receiptBytes();
        }
      }
    } catch (...) {
      close(fd_);
      fd_ = -1;
      throw;
    }
  }
  ~PreparedTl2() {
    if (discard_thread_.joinable())
      discard_thread_.join();
    if (fd_ >= 0)
      close(fd_);
  }
  PreparedTl2(const PreparedTl2 &) = delete;
  PreparedTl2 &operator=(const PreparedTl2 &) = delete;
  bool validatedReceipt() const { return receipt_; }

  /** Deployment-only full validation. Bind the receipt to both exact open
   * file versions. A stale/missing receipt falls back to per-weight CRCs.
   * Every runtime read still checks device/inode/size/mtime/ctime. */
  size_t validate(int source_fd, const std::string &receipt_path) {
    receipt_ = false;
    std::vector<uint64_t> offsets;
    for (const auto &item : entries_)
      offsets.push_back(item.first);
    std::sort(offsets.begin(), offsets.end());
    std::vector<uint8_t> arena;
    for (uint64_t off : offsets) {
      Entry &e = *entries_.at(off);
      e.verified.store(false);
      arena.resize(size_t(e.K) * e.N / 4u + 8u * e.N + 4u);
      read(source_fd, off, e.K, e.N, false, arena.data());
      read(source_fd, off, e.K, e.N, true, arena.data());
    }
    if (!(PreparedQs2Stamp(source_fd) == source_stamp_) ||
        !(PreparedQs2Stamp(fd_) == *stamp_))
      throw std::runtime_error("prepared TL2: files changed while validating");
    const auto data = receiptBytes();
    const std::string temporary = receipt_path + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out.write(reinterpret_cast<const char *>(data.data()), data.size()))
      throw std::runtime_error("prepared TL2: cannot write validation receipt");
    out.close();
    if (!out || std::rename(temporary.c_str(), receipt_path.c_str()))
      throw std::runtime_error(
        "prepared TL2: cannot publish validation receipt");
    return offsets.size();
  }

  /** Discard obsolete representation pages at the idle phase boundary.
   * Only full pages inside each payload are discarded: adjacent native/QS2
   * records may share a boundary page. This does not evict active weights. */
  void discard(bool native) const {
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
      return;
    const uint64_t mask = static_cast<uint64_t>(page) - 1u;
    for (const auto &item : entries_) {
      const Entry &e = *item.second;
      const uint64_t start = native ? e.tl2 : e.qs2;
      const uint64_t size =
        (native ? htp_tl2_native_bytes(e.K, e.N) : uint64_t(e.K) * e.N / 4u) +
        8u * e.N + 4u;
      const uint64_t lo = (start + mask) & ~mask;
      const uint64_t hi = (start + size) & ~mask;
      if (hi > lo)
        (void)posix_fadvise(fd_, static_cast<off_t>(lo),
                            static_cast<off_t>(hi - lo), POSIX_FADV_DONTNEED);
    }
  }

  /** Advice runs alongside decode. Join before a phase reversal or close,
   * so no detached task can touch a closed/reused descriptor. */
  void discardAsync(bool native) {
    if (discard_thread_.joinable())
      discard_thread_.join();
    discard_thread_ = std::thread([this, native] { discard(native); });
  }

  void read(int source_fd, uint64_t off, uint32_t K, uint32_t N, bool native,
            uint8_t *arena, int8_t *palette = nullptr) {
    if (!(PreparedQs2Stamp(source_fd) == source_stamp_) ||
        !(PreparedQs2Stamp(fd_) == *stamp_))
      throw std::runtime_error("prepared TL2: file changed during inference");
    const auto it = entries_.find(off);
    if (it == entries_.end() || it->second->K != K || it->second->N != N)
      throw std::runtime_error("prepared TL2: source weight not in index");
    Entry &e = *it->second;
    thread_local std::vector<uint8_t> data;
    const size_t codes = size_t(K) * N / 4u, tail = 8u * N + 4u;
    if (!receipt_ && !e.verified.load(std::memory_order_acquire)) {
      data.resize(codes + tail);
      preparedQs2Read(source_fd, data.data(), data.size(), off);
      if (preparedQs2Crc(data.data(), data.size()) != e.source_crc)
        throw std::runtime_error("prepared TL2: source checksum mismatch");
      e.verified.store(true, std::memory_order_release);
    }
    const size_t bytes = native ? htp_tl2_native_bytes(K, N) : codes;
    if (receipt_ && direct_read_) {
      iovec parts[] = {{arena, bytes},
                       {arena + codes, palette ? tail - 4u : tail},
                       {palette, 4u}};
      readScatter(fd_, native ? e.tl2 : e.qs2, parts, palette ? 3 : 2);
      return;
    }
    data.resize(bytes + tail);
    preparedQs2Read(fd_, data.data(), data.size(), native ? e.tl2 : e.qs2);
    if (!receipt_ && preparedQs2Crc(data.data(), data.size()) !=
                       (native ? e.tl2_crc : e.qs2_crc))
      throw std::runtime_error("prepared TL2: payload checksum mismatch");
    std::memcpy(arena, data.data(), bytes);
    std::memcpy(arena + codes, data.data() + bytes, palette ? tail - 4u : tail);
    if (palette)
      std::memcpy(palette, data.data() + bytes + tail - 4u, 4u);
  }
};
#endif
} // namespace nntrainer
#endif
