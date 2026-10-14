// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file weight_file_reader_check.cpp
 * @brief Direct-IO model reads: unaligned ranges, EOF, concurrency and
 * residency.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <weight_file_reader.h>

int main(int argc, char **argv) {
  std::string name =
    std::string(argc > 1 ? argv[1] : "/tmp") + "/nntr-weight-read-XXXXXX";
  char *path = name.data();
  int fd = mkstemp(path);
  assert(fd >= 0);
  const size_t size = 3 * (1u << 20) + 123;
  std::vector<unsigned char> original(size);
  for (size_t i = 0; i < size; ++i)
    original[i] = (i * 37 + i / 31) % 251;
  assert(write(fd, original.data(), size) == static_cast<ssize_t>(size));
  assert(fsync(fd) == 0);
  setenv("NNTR_HTP_PAGE_CACHE", "0", 1);
  int direct = nntrainer::openWeightFile(path);
  auto verify = [&](size_t offset, size_t bytes) {
    std::vector<unsigned char> data(bytes + 2, 0xff);
    assert(nntrainer::weightPread(direct, data.data() + 1, bytes, offset) == 0);
    assert(data.front() == 0xff && data.back() == 0xff);
    assert(std::memcmp(data.data() + 1, original.data() + offset, bytes) == 0);
  };
  verify(3, size - 3);
  verify(size - 17, 17);
  std::thread a([&] { verify(4093, 8193); });
  std::thread b([&] { verify(98765, 1000000); });
  a.join();
  b.join();
  void *aligned = nullptr;
  assert(posix_memalign(&aligned, 4096, 2 * (1u << 20)) == 0);
  assert(nntrainer::weightPread(direct, aligned, 2 * (1u << 20), 4096) == 0);
  assert(std::memcmp(aligned, original.data() + 4096, 2 * (1u << 20)) == 0);
  free(aligned);
  unsigned char tail[32];
  assert(nntrainer::weightPread(direct, tail, sizeof(tail), size - 16) ==
         ENODATA);
  assert(nntrainer::weightPread(direct, tail, sizeof(tail), UINT64_MAX) ==
         EOVERFLOW);
  assert(nntrainer::weightPread(fd, tail, sizeof(tail), 0) == EINVAL);
  auto verifySplit = [&](int input, size_t offset, size_t first, size_t last) {
    std::vector<unsigned char> code(first + 2, 0xff), meta(last + 2, 0xfe);
    assert(nntrainer::weightPreadSplit(input, code.data() + 1, first,
                                       meta.data() + 1, last, offset) == 0);
    assert(code.front() == 0xff && code.back() == 0xff);
    assert(meta.front() == 0xfe && meta.back() == 0xfe);
    assert(std::memcmp(code.data() + 1, original.data() + offset, first) == 0);
    assert(std::memcmp(meta.data() + 1, original.data() + offset + first,
                       last) == 0);
  };
  verifySplit(direct, 7, 1002500, 11268);
  verifySplit(direct, 3, 2 * (1u << 20), size - 3 - 2 * (1u << 20));
  verifySplit(direct, size - 17, 0, 17);
  verifySplit(direct, size - 17, 17, 0);
  assert(nntrainer::weightPreadSplit(direct, tail, 16, tail + 16, 16,
                                     size - 16) == ENODATA);
  assert(nntrainer::weightPreadSplit(direct, tail, 16, tail + 16, SIZE_MAX,
                                     0) == EOVERFLOW);
  const size_t page = sysconf(_SC_PAGESIZE);
  std::vector<unsigned char> pages((size + page - 1) / page);
  void *mapping = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
  assert(mapping != MAP_FAILED);
  assert(mincore(mapping, size, pages.data()) == 0);
  for (auto p : pages)
    assert((p & 1) == 0);
  setenv("NNTR_HTP_PAGE_CACHE", "1", 1);
  verifySplit(fd, 3, 1002500, 11268);
  verifySplit(fd, 3, 2 * (1u << 20), size - 3 - 2 * (1u << 20));
  verifySplit(fd, size - 17, 0, 17);
  assert(nntrainer::weightPread(fd, tail, sizeof(tail), 7) == 0);
  assert(std::memcmp(tail, original.data() + 7, sizeof(tail)) == 0);
  // A sparse file exercises the 2 GB LRU boundary using only a few MB of
  // storage. Each one-byte read occupies a distinct 1 MiB accounting range.
  constexpr size_t grain = 1u << 20, blocks = 1920;
  assert(ftruncate(fd, blocks * grain) == 0);
  const unsigned char value = 91;
  for (size_t block = 0; block < blocks; ++block)
    assert(pwrite(fd, &value, 1, block * grain) == 1);
  assert(fsync(fd) == 0);
  int cached = nntrainer::openWeightFile(path);
  for (size_t block = 0; block < blocks; ++block) {
    unsigned char got = 0;
    assert(nntrainer::weightPread(cached, &got, 1, block * grain) == 0);
    assert(got == value);
  }
  void *sparse = mmap(nullptr, blocks * grain, PROT_READ, MAP_SHARED, fd, 0);
  assert(sparse != MAP_FAILED);
  unsigned char residency = 0;
  assert(mincore(sparse, page, &residency) == 0);
  assert((residency & 1) == 0); // Oldest range was evicted.
  assert(mincore(static_cast<char *>(sparse) + (blocks - 1) * grain, page,
                 &residency) == 0);
  assert((residency & 1) == 1); // Most recent range remains cached.
  assert(nntrainer::weightPread(cached, tail, 1, 0) == 0 && tail[0] == value);
  assert(mincore(sparse, page, &residency) == 0 && (residency & 1));
  munmap(sparse, blocks * grain);
  close(cached);
  setenv("NNTR_HTP_PAGE_CACHE", "invalid", 1);
  bool rejected = false;
  try {
    nntrainer::weightPageCachePolicy();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
  munmap(mapping, size);
  close(direct);
  close(fd);
  unlink(path);
  puts("WEIGHT FILE READ OK: direct/cache, unaligned EOF, parallel reads, no "
       "cache population");
}
