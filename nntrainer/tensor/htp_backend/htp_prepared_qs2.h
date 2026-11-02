// SPDX-License-Identifier: Apache-2.0
/** @file htp_prepared_qs2.h
 * @brief CRC and file I/O helpers for prepared weight artifacts.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __HTP_PREPARED_QS2_H__
#define __HTP_PREPARED_QS2_H__
#include "htp_prepared_fc.h"
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#if defined(__ANDROID__) || defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__aarch64__) && (defined(__ANDROID__) || defined(__linux__))
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace nntrainer {
#if defined(__aarch64__) && (defined(__ANDROID__) || defined(__linux__))
__attribute__((target("+crc"))) inline uint32_t
preparedQs2CrcArm(const void *data, size_t size) {
  const auto *p = static_cast<const uint8_t *>(data);
  uint32_t c = ~0u;
  while (size >= 8u) {
    uint64_t word;
    std::memcpy(&word, p, sizeof(word));
    __asm__("crc32x %w0, %w0, %1" : "+r"(c) : "r"(word));
    p += 8u;
    size -= 8u;
  }
  while (size--) {
    const uint32_t byte = *p++;
    __asm__("crc32b %w0, %w0, %w1" : "+r"(c) : "r"(byte));
  }
  return ~c;
}
#endif
inline uint32_t preparedQs2Crc(const void *data, size_t size) {
#if defined(__aarch64__) && (defined(__ANDROID__) || defined(__linux__))
  static const bool crc = (getauxval(AT_HWCAP) & HWCAP_CRC32) != 0;
  if (crc)
    return preparedQs2CrcArm(data, size);
#endif
  static const auto table = [] {
    std::array<std::array<uint32_t, 256>, 8> t{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (unsigned b = 0; b < 8; ++b)
        c = (c >> 1) ^ ((0u - (c & 1u)) & 0xedb88320u);
      t[0][i] = c;
    }
    for (unsigned j = 1; j < 8; ++j)
      for (unsigned i = 0; i < 256; ++i)
        t[j][i] = t[0][t[j - 1][i] & 255u] ^ (t[j - 1][i] >> 8);
    return t;
  }();
  uint32_t c = ~0u;
  const auto *p = static_cast<const uint8_t *>(data);
  size_t i = 0;
  for (; size - i >= 8u; i += 8u) {
    c ^= preparedFcU32(p + i);
    c = table[7][c & 255u] ^ table[6][(c >> 8) & 255u] ^
        table[5][(c >> 16) & 255u] ^ table[4][c >> 24] ^ table[3][p[i + 4]] ^
        table[2][p[i + 5]] ^ table[1][p[i + 6]] ^ table[0][p[i + 7]];
  }
  for (; i < size; ++i)
    c = table[0][(c ^ p[i]) & 255u] ^ (c >> 8);
  return ~c;
}

#if defined(__ANDROID__) || defined(__linux__)
/** Identity and change stamp of an open file. Artifacts and models must not
 * be modified while inference reads them. Replacement between loads is
 * detected by inode/ctime/mtime; validation is reused only for that version.
 * CRC is corruption detection, not authentication against malicious writes. */
struct PreparedQs2Stamp {
  struct stat value {};
  explicit PreparedQs2Stamp(int fd) {
    if (fstat(fd, &value))
      throw std::runtime_error("prepared QS2: fstat failed");
  }
  std::array<uint64_t, 7> words() const {
    return {{static_cast<uint64_t>(value.st_dev),
             static_cast<uint64_t>(value.st_ino),
             static_cast<uint64_t>(value.st_size),
             static_cast<uint64_t>(value.st_mtim.tv_sec),
             static_cast<uint64_t>(value.st_mtim.tv_nsec),
             static_cast<uint64_t>(value.st_ctim.tv_sec),
             static_cast<uint64_t>(value.st_ctim.tv_nsec)}};
  }
  bool operator==(const PreparedQs2Stamp &b) const {
    return value.st_dev == b.value.st_dev && value.st_ino == b.value.st_ino &&
           value.st_size == b.value.st_size &&
           value.st_mtim.tv_sec == b.value.st_mtim.tv_sec &&
           value.st_mtim.tv_nsec == b.value.st_mtim.tv_nsec &&
           value.st_ctim.tv_sec == b.value.st_ctim.tv_sec &&
           value.st_ctim.tv_nsec == b.value.st_ctim.tv_nsec;
  }
};

inline void preparedQs2Read(int fd, void *data, size_t bytes, uint64_t offset) {
  auto *p = static_cast<uint8_t *>(data);
  while (bytes) {
    const ssize_t n = pread(fd, p, bytes, static_cast<off_t>(offset));
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      throw std::runtime_error("prepared QS2: truncated or unreadable file");
    p += n;
    offset += static_cast<uint64_t>(n);
    bytes -= static_cast<size_t>(n);
  }
}

#endif
} // namespace nntrainer
#endif
