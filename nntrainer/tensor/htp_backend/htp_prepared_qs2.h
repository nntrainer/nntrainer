// SPDX-License-Identifier: Apache-2.0
/** @file htp_prepared_qs2.h
 * @brief Checked reader for offline QS2 decode rows. Models stay unchanged.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __HTP_PREPARED_QS2_H__
#define __HTP_PREPARED_QS2_H__
#include "htp_prepared_fc.h"
#include "htp_qs2_lut_layout.h"
#include <array>
#include <cerrno>
#include <memory>
#include <mutex>
#include <unordered_map>
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

/** Same little-endian field widths as the prepared FC header, with magic
 * NNTRI201, version=1, layout=1 (K-major pair LUT); hash fields hold CRC32.
 * Source includes compact WH, palette, f32 scales and f32 sums. Payload
 * contains K*N/4 rearranged code bytes only. CRCs detect stale/corrupt
 * artifacts, not adversarial substitution. Every load checks both CRCs. */
inline std::vector<uint8_t> readPreparedQs2(const std::string &path,
                                            uint64_t source_offset, uint32_t K,
                                            uint32_t N, const void *source,
                                            size_t source_bytes) {
  auto fail = [&path](const char *reason) {
    throw std::runtime_error("prepared QS2: " + path + ": " + reason);
  };
  const uint64_t bytes = uint64_t(K) * N / 4u;
  if (!source || !K || !N || K % 32 || N % 128 || bytes > kPreparedFcMaxBytes ||
      source_bytes != bytes + 4u + uint64_t(N) * 8u)
    fail("unsupported source shape");
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() != std::streamoff(bytes + 64u))
    fail("missing or incorrectly sized artifact");
  input.seekg(0);
  uint8_t h[64];
  if (!input.read(reinterpret_cast<char *>(h), sizeof(h)))
    fail("truncated header");
  if (std::memcmp(h, "NNTRI201", 8) || preparedFcU32(h + 8) != 1u ||
      preparedFcU32(h + 12) != 1u || preparedFcU64(h + 16) != source_offset ||
      preparedFcU64(h + 24) != source_bytes || preparedFcU32(h + 40) != K ||
      preparedFcU32(h + 44) != N || preparedFcU64(h + 48) != bytes ||
      preparedFcU64(h + 32) != preparedQs2Crc(source, source_bytes))
    fail("format, layout, shape or source checksum mismatch");
  std::vector<uint8_t> data(static_cast<size_t>(bytes));
  if (!input.read(reinterpret_cast<char *>(data.data()), data.size()) ||
      preparedFcU64(h + 56) != preparedQs2Crc(data.data(), data.size()))
    fail("payload checksum mismatch");
  return data;
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

/** Deployment-time verification receipt. It records the exact local model
 * and artifact identities whose full CRCs validate_qs2_lut checked. It is
 * optional: absent, stale or malformed receipts fall back to full validation.
 * Like the CRCs, this is not authentication against malicious replacement.
 * The receipt is loaded once per directory/process; per-load open-file
 * stamps still invalidate it after replacement or in-place modification. */
inline bool preparedQs2ReceiptMatch(const std::string &path,
                                    const PreparedQs2Stamp &model,
                                    const PreparedQs2Stamp &artifact,
                                    const uint8_t *header) {
  struct Receipt {
    std::vector<uint8_t> bytes;
    std::unordered_map<uint64_t, size_t> records;
    explicit Receipt(const std::string &name) {
      std::ifstream file(name, std::ios::binary | std::ios::ate);
      if (!file || file.tellg() < 76 || file.tellg() > (16 << 20))
        return;
      bytes.resize(static_cast<size_t>(file.tellg()));
      file.seekg(0);
      if (!file.read(reinterpret_cast<char *>(bytes.data()), bytes.size()) ||
          std::memcmp(bytes.data(), "NNTRV201", 8) ||
          preparedFcU64(bytes.data() + 8) != (bytes.size() - 76u) / 120u ||
          (bytes.size() - 76u) % 120u ||
          preparedFcU32(bytes.data() + bytes.size() - 4u) !=
            preparedQs2Crc(bytes.data(), bytes.size() - 4u)) {
        bytes.clear();
        return;
      }
      for (size_t i = 72u; i + 120u <= bytes.size() - 4u; i += 120u) {
        if (!records.emplace(preparedFcU64(bytes.data() + i + 16u), i).second) {
          records.clear();
          bytes.clear();
          return;
        }
      }
    }
  };
  const size_t slash = path.find_last_of('/');
  const std::string name =
    (slash == std::string::npos ? "." : path.substr(0, slash)) +
    "/.nntri2.validated";
  static std::mutex mutex;
  static std::unordered_map<std::string, std::shared_ptr<const Receipt>>
    receipts;
  std::shared_ptr<const Receipt> receipt;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto &slot = receipts[name];
    if (!slot)
      slot = std::make_shared<Receipt>(name);
    receipt = slot;
  }
  if (receipt->bytes.empty())
    return false;
  const auto match = [](const uint8_t *p, const PreparedQs2Stamp &stamp) {
    const auto words = stamp.words();
    for (size_t i = 0; i < words.size(); ++i)
      if (preparedFcU64(p + 8u * i) != words[i])
        return false;
    return true;
  };
  if (!match(receipt->bytes.data() + 16u, model))
    return false;
  const auto it = receipt->records.find(preparedFcU64(header + 16u));
  if (it == receipt->records.end())
    return false;
  const auto *record = receipt->bytes.data() + it->second;
  return !std::memcmp(record, header, 64u) && match(record + 64u, artifact);
}

/** Version 2: one LUT code image followed by the source palette/scales/sums.
 * The first load validates both full CRCs. Subsequent loads of unchanged
 * open-file identities read only the prepared payload. With arena supplied,
 * the code bytes go directly from pread to ION, avoiding the cached bounce
 * buffer; only the metadata tail in data is valid on return. Without arena,
 * data contains the entire payload (portable validation and test callers).
 * The small validation cache stores identities, never weight payloads. */
inline void readPreparedQs2Cached(const std::string &path, int source_fd,
                                  uint64_t source_offset, uint32_t K,
                                  uint32_t N, std::vector<uint8_t> &data,
                                  uint8_t *arena = nullptr,
                                  bool allow_receipt = true) {
  struct Fd {
    int fd;
    explicit Fd(const std::string &name) :
      fd(open(name.c_str(), O_RDONLY | O_CLOEXEC)) {}
    ~Fd() {
      if (fd >= 0)
        close(fd);
    }
  } file(path);
  const uint64_t codes = uint64_t(K) * N / 4u;
  const uint64_t bytes = codes + 4u + uint64_t(N) * 8u;
  if (file.fd < 0 || !K || !N || K % 32u || N % 128u ||
      codes > kPreparedFcMaxBytes)
    throw std::runtime_error(
      "prepared QS2: missing artifact or invalid shape: " + path);
  const PreparedQs2Stamp model(source_fd), artifact(file.fd);
  if (artifact.value.st_size != static_cast<off_t>(bytes + 64u) ||
      source_offset > static_cast<uint64_t>(model.value.st_size) ||
      bytes > static_cast<uint64_t>(model.value.st_size) - source_offset)
    throw std::runtime_error("prepared QS2: incorrect file size: " + path);
  uint8_t header[64];
  preparedQs2Read(file.fd, header, sizeof(header), 0);
  if (std::memcmp(header, "NNTRI202", 8) || preparedFcU32(header + 8) != 2u ||
      preparedFcU32(header + 12) != 2u ||
      preparedFcU64(header + 16) != source_offset ||
      preparedFcU64(header + 24) != bytes || preparedFcU32(header + 40) != K ||
      preparedFcU32(header + 44) != N || preparedFcU64(header + 48) != bytes)
    throw std::runtime_error(
      "prepared QS2: version 2 single-layout artifact required: " + path);
  struct Validation {
    std::mutex mutex;
    bool valid = false;
    PreparedQs2Stamp model, artifact;
    std::array<uint8_t, 64> header{};
    Validation(int source, int file) : model(source), artifact(file) {}
  };
  static std::mutex map_mutex;
  static std::unordered_map<std::string, std::shared_ptr<Validation>> validated;
  std::shared_ptr<Validation> entry;
  {
    std::lock_guard<std::mutex> lock(map_mutex);
    auto &slot = validated[path];
    if (!slot)
      slot = std::make_shared<Validation>(source_fd, file.fd);
    entry = slot;
  }
  std::lock_guard<std::mutex> lock(entry->mutex);
  const bool reuse =
    (entry->valid && entry->model == model && entry->artifact == artifact &&
     !std::memcmp(entry->header.data(), header, 64)) ||
    (allow_receipt && preparedQs2ReceiptMatch(path, model, artifact, header));
  data.resize(static_cast<size_t>(bytes));
  if (reuse && arena) {
    // Do not bounce megabytes through cached scratch and memcpy to ION.
    // Metadata is small and still converted in cached memory by the caller.
    preparedQs2Read(file.fd, arena, static_cast<size_t>(codes), 64u);
    preparedQs2Read(file.fd, data.data() + codes,
                    static_cast<size_t>(bytes - codes), 64u + codes);
  } else {
    preparedQs2Read(file.fd, data.data(), data.size(), 64u);
  }
  if (!reuse) {
    entry->valid = false;
    thread_local std::vector<uint8_t> source;
    source.resize(data.size());
    preparedQs2Read(source_fd, source.data(), source.size(), source_offset);
    if (preparedFcU64(header + 32) !=
          preparedQs2Crc(source.data(), source.size()) ||
        preparedFcU64(header + 56) !=
          preparedQs2Crc(data.data(), data.size()) ||
        std::memcmp(source.data() + codes, data.data() + codes, bytes - codes))
      throw std::runtime_error(
        "prepared QS2: source/payload checksum mismatch: " + path);
  }
  if (!(model == PreparedQs2Stamp(source_fd)) ||
      !(artifact == PreparedQs2Stamp(file.fd)))
    throw std::runtime_error("prepared QS2: file changed during read: " + path);
  if (arena && !reuse)
    std::memcpy(arena, data.data(), static_cast<size_t>(codes));
  entry->model = model;
  entry->artifact = artifact;
  std::memcpy(entry->header.data(), header, 64);
  entry->valid = true;
}
#endif
} // namespace nntrainer
#endif
