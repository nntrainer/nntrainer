// SPDX-License-Identifier: Apache-2.0
/** @file prepare_tl2.cpp
 * @brief Prepare checked symmetric QS2 and native TL2 expert payloads.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "htp_prepared_tl2.h"
#include "htp_tl2_native_packing.h"
#include <fstream>
#include <iostream>
#include <thread>

#if defined(__ANDROID__) || defined(__linux__)
using namespace nntrainer;
struct Weight {
  uint64_t off, qs2, tl2;
  uint32_t K, N, source_crc, qs2_crc, tl2_crc;
};
static void put(uint8_t *p, uint64_t v, unsigned bytes) {
  for (unsigned i = 0; i < bytes; ++i)
    p[i] = uint8_t(v >> (8u * i));
}
static void writeAt(int fd, const void *data, size_t bytes, uint64_t off) {
  const auto *p = static_cast<const uint8_t *>(data);
  while (bytes) {
    const ssize_t n = pwrite(fd, p, bytes, off);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      throw std::runtime_error("write failed");
    bytes -= n;
    off += n;
    p += n;
  }
}
int main(int argc, char **argv) {
  if (argc == 4 && std::string(argv[1]) == "--validate") {
    const int src = open(argv[2], O_RDONLY | O_CLOEXEC);
    if (src < 0) {
      std::cerr << "Cannot open source\n";
      return 1;
    }
    try {
      PreparedTl2 prepared(argv[3], src, false);
      const size_t count =
        prepared.validate(src, std::string(argv[3]) + ".validated");
      close(src);
      std::cout << "Validated " << count << " weights\n";
      return 0;
    } catch (const std::exception &e) {
      close(src);
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
  if (argc != 4) {
    std::cerr << "Usage: prepare_tl2 SOURCE MODEL_WEIGHTS.tsv OUTPUT\n"
                 "       prepare_tl2 --validate SOURCE OUTPUT\n"
                 "TSV columns: original_offset K N. OUTPUT must not exist.\n";
    return 2;
  }
  const int src = open(argv[1], O_RDONLY | O_CLOEXEC);
  const int dst = open(argv[3], O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0644);
  if (src < 0 || dst < 0) {
    std::cerr << "Cannot open files\n";
    return 1;
  }
  try {
    PreparedQs2Stamp stamp(src);
    std::ifstream spec(argv[2]);
    std::vector<Weight> weights;
    Weight w{};
    while (spec >> w.off >> w.K >> w.N) {
      if (!w.K || !w.N || w.K % 32 || w.N % 32 || w.K > 8192 || w.N > 8192 ||
          htp_tl2_native_bytes(w.K, w.N) > uint64_t(w.K) * w.N / 4u)
        throw std::runtime_error("unsupported compact TL2 shape");
      weights.push_back(w);
    }
    if (!spec.eof() || weights.empty() || weights.size() > 65536)
      throw std::runtime_error("invalid weight catalog");
    uint64_t pos = (64 + weights.size() * 48 + 4095) & ~uint64_t(4095);
    for (auto &v : weights) {
      const uint64_t codes = uint64_t(v.K) * v.N / 4u, tail = 8u * v.N + 4u;
      if (v.off > uint64_t(stamp.value.st_size) ||
          codes + tail > uint64_t(stamp.value.st_size) - v.off)
        throw std::runtime_error("source bounds");
      v.qs2 = pos;
      v.tl2 = (pos + codes + tail + 4095) & ~uint64_t(4095);
      pos = (v.tl2 + htp_tl2_native_bytes(v.K, v.N) + tail + 4095) &
            ~uint64_t(4095);
    }
    if (ftruncate(dst, pos))
      throw std::runtime_error("ftruncate failed");
    std::atomic<size_t> next{0}, done{0};
    std::mutex error_mutex;
    std::exception_ptr error;
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < 8; ++t)
      workers.emplace_back([&] {
        try {
          for (size_t i = next.fetch_add(1); i < weights.size();
               i = next.fetch_add(1)) {
            Weight &v = weights[i];
            const size_t codes = size_t(v.K) * v.N / 4u, tail = 8u * v.N + 4u;
            std::vector<uint8_t> source(codes + tail), native,
              qs2(codes + tail);
            preparedQs2Read(src, source.data(), source.size(), v.off);
            v.source_crc = preparedQs2Crc(source.data(), source.size());
            prepareSymmetricTl2(source.data(), v.K, v.N, &native);
            std::memcpy(qs2.data(), source.data(), codes);
            std::memcpy(qs2.data() + codes, source.data() + codes + 4,
                        4u * v.N);
            for (uint32_t n = 0; n < v.N; ++n) {
              float f;
              std::memcpy(&f, source.data() + codes + 4u + 4u * (v.N + n), 4u);
              const int32_t s = static_cast<int32_t>(f);
              std::memcpy(qs2.data() + codes + 4u * (v.N + n), &s, 4u);
            }
            std::memcpy(qs2.data() + codes + 8u * v.N, source.data() + codes,
                        4u);
            native.insert(native.end(), qs2.begin() + codes, qs2.end());
            v.qs2_crc = preparedQs2Crc(qs2.data(), qs2.size());
            v.tl2_crc = preparedQs2Crc(native.data(), native.size());
            writeAt(dst, qs2.data(), qs2.size(), v.qs2);
            writeAt(dst, native.data(), native.size(), v.tl2);
            const size_t count = done.fetch_add(1) + 1;
            if (count % 512 == 0)
              std::cerr << "Prepared " << count << '/' << weights.size()
                        << '\n';
          }
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          error = std::current_exception();
        }
      });
    for (auto &t : workers)
      t.join();
    if (error)
      std::rethrow_exception(error);
    std::vector<uint8_t> index(weights.size() * 48u, 0);
    for (size_t i = 0; i < weights.size(); ++i) {
      const Weight &v = weights[i];
      uint8_t *p = index.data() + i * 48u;
      put(p, v.off, 8);
      put(p + 8, v.K, 4);
      put(p + 12, v.N, 4);
      put(p + 16, v.qs2, 8);
      put(p + 24, v.tl2, 8);
      put(p + 32, v.qs2_crc, 4);
      put(p + 36, v.tl2_crc, 4);
      put(p + 40, v.source_crc, 4);
    }
    uint8_t header[64] = {};
    std::memcpy(header, "NNTRTL21", 8);
    put(header + 8, 1, 4);
    put(header + 12, weights.size(), 4);
    put(header + 16, stamp.value.st_size, 8);
    put(header + 24, pos, 8);
    put(header + 32, preparedQs2Crc(index.data(), index.size()), 4);
    writeAt(dst, index.data(), index.size(), 64);
    writeAt(dst, header, sizeof(header), 0);
    fsync(dst);
    std::cout << weights.size() << " weights, " << pos << " bytes\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    close(src);
    close(dst);
    return 1;
  }
  close(src);
  close(dst);
}

#else
int main() {
  std::cerr << "This deployment tool requires Linux or Android\n";
  return 2;
}
#endif
