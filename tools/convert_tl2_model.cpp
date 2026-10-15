// SPDX-License-Identifier: Apache-2.0
/** @file convert_tl2_model.cpp
 * @brief Replace QS2 experts with native-only TL2 in a positional model.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#include "htp_prepared_tl2.h"
#include "htp_tl2_native_packing.h"
#include "hvx_gemv_u8tl2.h"
#include <cmath>
#include <iostream>
#include <thread>

#if defined(__ANDROID__) || defined(__linux__)
using namespace nntrainer;
struct NativeWeight {
  uint64_t source, target;
  uint32_t K, N;
};
static void writeAll(int fd, const void *buffer, size_t size, uint64_t offset) {
  const auto *data = static_cast<const uint8_t *>(buffer);
  while (size) {
    const ssize_t n = pwrite(fd, data, size, offset);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      throw std::runtime_error("TL2 output write failed");
    data += n;
    size -= n;
    offset += n;
  }
}
int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: convert_tl2_model SOURCE EXPERTS.tsv OUTPUT\n"
                 "TSV: original_offset K N. OUTPUT must not exist.\n";
    return 2;
  }
  const int src = open(argv[1], O_RDONLY | O_CLOEXEC);
  if (src < 0) {
    std::cerr << "Cannot open source\n";
    return 1;
  }
  const int dst = open(argv[3], O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0644);
  if (dst < 0) {
    close(src);
    std::cerr << "Cannot open source/output\n";
    return 1;
  }
  try {
    const PreparedQs2Stamp stamp(src);
    std::ifstream catalog(argv[2]);
    std::vector<NativeWeight> weights;
    NativeWeight w{};
    uint64_t input = 0, output = 0;
    std::vector<uint8_t> copy(1u << 20);
    auto copyRange = [&](uint64_t end) {
      while (input < end) {
        const size_t n = std::min<uint64_t>(copy.size(), end - input);
        preparedQs2Read(src, copy.data(), n, input);
        writeAll(dst, copy.data(), n, output);
        input += n;
        output += n;
      }
    };
    while (catalog >> w.source >> w.K >> w.N) {
      const uint64_t old = uint64_t(w.K) * w.N / 4u + 4u + 8u * w.N;
      const uint64_t native = htp_tl2_native_bytes(w.K, w.N);
      if (!w.K || !w.N || w.K % 32u || w.N % 128u || w.K > 8192u ||
          w.N > 8192u || native > uint64_t(w.K) * w.N / 4u ||
          w.source < input || w.source > uint64_t(stamp.value.st_size) ||
          old > uint64_t(stamp.value.st_size) - w.source)
        throw std::runtime_error("Invalid expert extent/shape");
      copyRange(w.source);
      w.target = output;
      weights.push_back(w);
      input += old;
      output += native + 8u * w.N;
    }
    if (!catalog.eof() || weights.empty() || weights.size() > 65536u)
      throw std::runtime_error("Invalid expert catalog");
    copyRange(stamp.value.st_size);
    if (ftruncate(dst, output))
      throw std::runtime_error("Cannot size output");
    std::atomic<size_t> next{0}, done{0};
    std::mutex error_mutex;
    std::exception_ptr error;
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < 8u; ++t)
      workers.emplace_back([&] {
        try {
          for (size_t i = next.fetch_add(1); i < weights.size();
               i = next.fetch_add(1)) {
            const auto &v = weights[i];
            const size_t codes = size_t(v.K) * v.N / 4u;
            std::vector<uint8_t> source(codes + 4u + 8u * v.N), native;
            preparedQs2Read(src, source.data(), source.size(), v.source);
            const int8_t palette[4] = {-1, 0, 1, 1};
            if (std::memcmp(source.data() + codes, palette, 4u))
              throw std::runtime_error("Expected exact pure ternary palette");
            prepareSymmetricTl2(source.data(), v.K, v.N, &native);
            writeAll(dst, native.data(), native.size(), v.target);
            writeAll(dst, source.data() + codes + 4u, 8u * v.N,
                     v.target + native.size());
            // Read the serialized record and independently decode it.
            std::vector<uint8_t> serialized(native.size() + 8u * v.N);
            preparedQs2Read(dst, serialized.data(), serialized.size(),
                            v.target);
            if (std::memcmp(serialized.data() + native.size(),
                            source.data() + codes + 4u, 8u * v.N))
              throw std::runtime_error("TL2 scale/sum copy mismatch");
            std::vector<int8_t> decoded(size_t(v.K) * v.N);
            if (hvx_tl2_unpack(serialized.data(), v.K, v.N, decoded.data()))
              throw std::runtime_error("Invalid serialized TL2 codes");
            for (uint32_t n = 0; n < v.N; ++n) {
              int sum = 0;
              for (uint32_t k = 0; k < v.K; ++k) {
                const uint32_t r = k % 32u;
                const size_t at =
                  (size_t(k / 32u) * (v.N / 32u) + n / 32u) * 256u +
                  r / 16u * 128u + n % 32u * 4u + r % 4u;
                const unsigned q = (source[at] >> (2u * (r / 4u % 4u))) & 3u;
                if (decoded[size_t(k) * v.N + n] != palette[q])
                  throw std::runtime_error(
                    "TL2 weight reconstruction mismatch");
                sum += palette[q];
              }
              float scale, stored_sum;
              std::memcpy(&scale, source.data() + codes + 4u + 4u * n, 4u);
              std::memcpy(&stored_sum,
                          source.data() + codes + 4u + 4u * (v.N + n), 4u);
              if (!std::isfinite(scale) || scale < 0 || stored_sum != sum)
                throw std::runtime_error("Invalid expert scale/column sum");
            }
            const size_t count = done.fetch_add(1) + 1;
            if (count % 512u == 0)
              std::cerr << "Converted and verified " << count << '/'
                        << weights.size() << '\n';
          }
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          error = std::current_exception();
        }
      });
    for (auto &thread : workers)
      thread.join();
    if (error)
      std::rethrow_exception(error);
    if (!(PreparedQs2Stamp(src) == stamp) || fsync(dst))
      throw std::runtime_error("Source changed or output sync failed");
    std::cout << "Native-only TL2 model: " << weights.size() << " experts, "
              << output << " bytes\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    close(src);
    close(dst);
    unlink(argv[3]);
    return 1;
  }
  close(src);
  close(dst);
  return 0;
}
#else
int main() { return 2; }
#endif
