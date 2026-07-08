// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   tokenizer_cache_util.h
 * @brief  Common helpers for compact tokenizer cache blobs
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef TOKENIZER_CACHE_UTIL_H_
#define TOKENIZER_CACHE_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tokenizers {
namespace cache_util {

/**
 * @brief Tokenizer backend identifier stored in the cache blob header
 */
enum class CacheKind : uint32_t {
  WordPiece = 1,
  BPE = 2,
};

constexpr char kCacheMagic[] = {'Q', 'A', 'I', 'T', 'O', 'K', 'C', 'H'};
constexpr size_t kCacheMagicSize = sizeof(kCacheMagic);
constexpr uint32_t kCacheVersion = 3;
constexpr uint32_t kMaxU16 = 0xffff;
constexpr uint32_t kMaxU24 = 0xffffff;

inline std::string InvalidCacheMessage(const char *cache_name,
                                       const char *reason) {
  return std::string("Invalid ") + cache_name + " cache: " + reason;
}

inline bool IsLittleEndianHost() {
  const uint32_t value = 1;
  return *reinterpret_cast<const unsigned char *>(&value) == 1;
}

inline void AppendU32(std::string &out, uint32_t value) {
  out.push_back(static_cast<char>(value & 0xff));
  out.push_back(static_cast<char>((value >> 8) & 0xff));
  out.push_back(static_cast<char>((value >> 16) & 0xff));
  out.push_back(static_cast<char>((value >> 24) & 0xff));
}

inline void AppendU16(std::string &out, uint32_t value,
                      const char *cache_name) {
  if (value > kMaxU16) {
    throw std::runtime_error(InvalidCacheMessage(cache_name, "u16 overflow"));
  }
  out.push_back(static_cast<char>(value & 0xff));
  out.push_back(static_cast<char>((value >> 8) & 0xff));
}

inline void AppendU24(std::string &out, uint32_t value,
                      const char *cache_name) {
  if (value > kMaxU24) {
    throw std::runtime_error(InvalidCacheMessage(cache_name, "u24 overflow"));
  }
  out.push_back(static_cast<char>(value & 0xff));
  out.push_back(static_cast<char>((value >> 8) & 0xff));
  out.push_back(static_cast<char>((value >> 16) & 0xff));
}

inline void CheckMagic(const std::string &blob, const char *magic,
                       size_t magic_size, const char *cache_name) {
  if (blob.size() < magic_size ||
      std::memcmp(blob.data(), magic, magic_size) != 0) {
    throw std::runtime_error(InvalidCacheMessage(cache_name, "bad magic"));
  }
}

inline void AppendHeader(std::string &out, CacheKind kind) {
  out.append(kCacheMagic, kCacheMagicSize);
  AppendU32(out, kCacheVersion);
  AppendU32(out, static_cast<uint32_t>(kind));
}

inline uint32_t ReadU32(const std::string &blob, size_t &offset,
                        const char *cache_name) {
  if (offset + 4 > blob.size()) {
    throw std::runtime_error(InvalidCacheMessage(cache_name, "truncated u32"));
  }

  const unsigned char *p =
    reinterpret_cast<const unsigned char *>(blob.data() + offset);
  offset += 4;
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

inline uint32_t ReadU16(const std::string &blob, size_t &offset,
                        const char *cache_name) {
  if (offset + 2 > blob.size()) {
    throw std::runtime_error(InvalidCacheMessage(cache_name, "truncated u16"));
  }

  const unsigned char *p =
    reinterpret_cast<const unsigned char *>(blob.data() + offset);
  offset += 2;
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
}

inline uint32_t ReadU24(const std::string &blob, size_t &offset,
                        const char *cache_name) {
  if (offset + 3 > blob.size()) {
    throw std::runtime_error(InvalidCacheMessage(cache_name, "truncated u24"));
  }

  const unsigned char *p =
    reinterpret_cast<const unsigned char *>(blob.data() + offset);
  offset += 3;
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16);
}

inline std::string ReadBytes(const std::string &blob, size_t &offset,
                             size_t len, const char *cache_name) {
  if (offset + len > blob.size()) {
    throw std::runtime_error(
      InvalidCacheMessage(cache_name, "truncated bytes"));
  }

  std::string out(blob.data() + offset, len);
  offset += len;
  return out;
}

template <typename T>
inline void ReadTrivialVector(const std::string &blob, size_t &offset,
                              size_t count, std::vector<T> &out,
                              const char *cache_name) {
  static_assert(std::is_trivially_copyable<T>::value,
                "cache array bulk read requires trivially copyable elements");

  if (offset > blob.size()) {
    throw std::runtime_error(
      InvalidCacheMessage(cache_name, "truncated array"));
  }
  if (count > (blob.size() - offset) / sizeof(T)) {
    throw std::runtime_error(
      InvalidCacheMessage(cache_name, "truncated array"));
  }
  const size_t bytes = count * sizeof(T);
  if (!IsLittleEndianHost()) {
    throw std::runtime_error(
      InvalidCacheMessage(cache_name, "unsupported host endian"));
  }

  out.resize(count);
  if (bytes > 0) {
    std::memcpy(out.data(), blob.data() + offset, bytes);
  }
  offset += bytes;
}

inline void ReadU32Vector(const std::string &blob, size_t &offset, size_t count,
                          std::vector<uint32_t> &out, const char *cache_name) {
  if (IsLittleEndianHost()) {
    ReadTrivialVector(blob, offset, count, out, cache_name);
    return;
  }

  out.resize(count);
  for (uint32_t &value : out) {
    value = ReadU32(blob, offset, cache_name);
  }
}

inline size_t ReadHeader(const std::string &blob, CacheKind expected_kind,
                         const char *cache_name) {
  CheckMagic(blob, kCacheMagic, kCacheMagicSize, cache_name);

  size_t offset = kCacheMagicSize;
  const uint32_t version = ReadU32(blob, offset, cache_name);
  if (version != kCacheVersion) {
    throw std::runtime_error(
      InvalidCacheMessage(cache_name, "unsupported version"));
  }

  const auto kind = static_cast<CacheKind>(ReadU32(blob, offset, cache_name));
  if (kind != expected_kind) {
    throw std::runtime_error(
      InvalidCacheMessage(cache_name, "wrong tokenizer kind"));
  }

  return offset;
}

} // namespace cache_util
} // namespace tokenizers

#endif // TOKENIZER_CACHE_UTIL_H_
