// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file htp_prepared_fc.h
 * @brief Portable reader for offline-prepared QS4CX_WH FC slices.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __HTP_PREPARED_FC_H__
#define __HTP_PREPARED_FC_H__

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nntrainer {

// All fields are little endian. The 64-byte header is:
// magic[8]="NNTRWH01", version:u32, slices:u32, source_offset:u64,
// source_bytes:u64, source_fnv1a64:u64, K:u32, N:u32, payload_bytes:u64,
// payload_fnv1a64:u64. The payload begins with slices records of
// {column_start:u32, columns:u32, offset_in_payload:u64}. Each slice starts
// at a 512-byte boundary and contains WH bytes, f32 scales, i32 column sums.
// Source hashes cover the original WH bytes, f32 scales and f32 column sums.
// FNV checks detect stale/corrupt local artifacts; they are not signatures.
constexpr uint64_t kPreparedFcFnvOffset = UINT64_C(14695981039346656037);
constexpr uint64_t kPreparedFcFnvPrime = UINT64_C(1099511628211);
constexpr uint64_t kPreparedFcMaxBytes = UINT64_C(256) << 20;

inline uint64_t preparedFcHash(const void *data, size_t size,
                               uint64_t hash = kPreparedFcFnvOffset) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < size; ++i)
    hash = (hash ^ bytes[i]) * kPreparedFcFnvPrime;
  return hash;
}

inline uint32_t preparedFcSliceCols(uint32_t K) {
  const uint32_t tiles = K / 32u;
  const uint32_t n_tiles = tiles == 0 ? 0 : ((2u << 20) / 512u) / tiles;
  const uint32_t cols = n_tiles == 0 ? 32u : n_tiles * 32u;
  return cols >= 512u ? (cols / 512u) * 512u : cols;
}

inline uint64_t preparedFcSourceHash(const void *data, const float *scales,
                                     uint32_t K, uint32_t N) {
  const uint64_t bytes = uint64_t(K) * N / 2;
  if (!data || !scales || !K || !N || K % 32 || N % 32 ||
      bytes + uint64_t(N) * 8 > kPreparedFcMaxBytes)
    throw std::invalid_argument("prepared FC: unsupported source shape");
  return preparedFcHash(scales, size_t(N) * 8,
                        preparedFcHash(data, static_cast<size_t>(bytes)));
}

inline uint32_t preparedFcU32(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

inline uint64_t preparedFcU64(const uint8_t *p) {
  return preparedFcU32(p) | (uint64_t(preparedFcU32(p + 4)) << 32);
}

struct PreparedFcSlice {
  uint32_t column_start;
  uint32_t columns;
  uint64_t offset;
};

struct PreparedFcFile {
  std::vector<uint8_t> payload;
  std::vector<PreparedFcSlice> slices;
};

inline PreparedFcFile readPreparedFc(const std::string &path,
                                     uint64_t source_offset, uint32_t K,
                                     uint32_t N, uint64_t source_hash) {
  auto fail = [&path](const char *reason) {
    throw std::runtime_error("prepared FC: " + path + ": " + reason);
  };
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    fail("cannot open required offline artifact");
  const auto file_bytes = input.tellg();
  if (file_bytes < 64 || uint64_t(file_bytes) > kPreparedFcMaxBytes + 64)
    fail("invalid file size");
  input.seekg(0);
  uint8_t header[64];
  if (!input.read(reinterpret_cast<char *>(header), sizeof(header)))
    fail("truncated header");
  if (std::memcmp(header, "NNTRWH01", 8) != 0 || preparedFcU32(header + 8) != 1)
    fail("unsupported format");
  if (!K || !N || K % 32 || N % 32 || preparedFcU32(header + 40) != K ||
      preparedFcU32(header + 44) != N ||
      preparedFcU64(header + 16) != source_offset ||
      preparedFcU64(header + 24) != uint64_t(K) * N / 2 + uint64_t(N) * 8 ||
      preparedFcU64(header + 32) != source_hash)
    fail("source shape, offset or full weight checksum mismatch");
  const uint64_t payload_bytes = preparedFcU64(header + 48);
  const uint32_t count = preparedFcU32(header + 12);
  const uint32_t cap = preparedFcSliceCols(K);
  if (payload_bytes > kPreparedFcMaxBytes ||
      payload_bytes != uint64_t(file_bytes) - 64 || count == 0 ||
      count > 4096 || uint64_t(count) * 16 > payload_bytes ||
      count != (uint64_t(N) + cap - 1) / cap)
    fail("invalid payload size or slice count");
  PreparedFcFile result;
  result.payload.resize(static_cast<size_t>(payload_bytes));
  if (!input.read(reinterpret_cast<char *>(result.payload.data()),
                  static_cast<std::streamsize>(payload_bytes)))
    fail("truncated payload");
  if (preparedFcHash(result.payload.data(), result.payload.size()) !=
      preparedFcU64(header + 56))
    fail("payload checksum mismatch");
  uint32_t column = 0;
  uint64_t end = uint64_t(count) * 16;
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t *record = result.payload.data() + size_t(i) * 16;
    PreparedFcSlice slice{preparedFcU32(record), preparedFcU32(record + 4),
                          preparedFcU64(record + 8)};
    const uint32_t columns = std::min(cap, N - column);
    const uint64_t bytes = uint64_t(K) * columns / 2 + uint64_t(columns) * 8;
    const uint64_t aligned = (end + 511) & ~UINT64_C(511);
    if (slice.column_start != column || slice.columns != columns ||
        slice.offset != aligned || slice.offset > payload_bytes ||
        bytes > payload_bytes - slice.offset)
      fail("invalid slice dimensions, ordering or bounds");
    result.slices.push_back(slice);
    column += columns;
    end = slice.offset + bytes;
  }
  if (column != N || end != payload_bytes)
    fail("unexpected trailing payload");
  return result;
}

} // namespace nntrainer
#endif // __HTP_PREPARED_FC_H__
