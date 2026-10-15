// SPDX-License-Identifier: Apache-2.0
/** @file tl2_prepared_check.cpp
 * @brief Independent values and corruption checks for offline native TL2.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "htp_prepared_tl2.h"
#include "hvx_gemv_u8tl2.h"
#include <cassert>
#include <iostream>
using namespace nntrainer;
constexpr uint32_t K = 256, N = 256;
constexpr size_t codes = size_t(K) * N / 4, bytes = codes + 8 * N + 4;
static size_t at(uint32_t k, uint32_t n) {
  return (size_t(k / 32) * (N / 32) + n / 32) * 256 + (k % 32 / 16) * 128 +
         (n % 32) * 4 + k % 4;
}
static unsigned code(uint32_t k, uint32_t n) {
  return (k * 71 + n * 19 + k * n) % 4;
}
static int weight(uint32_t k, uint32_t n) {
  return code(k, n) < 2 ? -1 : code(k, n) == 2 ? 0 : 1;
}
template <class F> static void reject(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(failed);
}
int main(int argc, char **argv) {
  assert(argc == 3);
  if (std::string(argv[1]) == "--fixture" ||
      std::string(argv[1]) == "--unit-fixture") {
    const bool unit = std::string(argv[1]) == "--unit-fixture";
    std::vector<uint8_t> data(bytes, 0);
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t n = 0; n < N; ++n)
        data[at(k, n)] |= code(k, n) << (2 * ((k % 32 / 4) % 4));
    const int8_t pal[4] = {static_cast<int8_t>(unit ? -1 : -8),
                           static_cast<int8_t>(unit ? -1 : -7), 0,
                           static_cast<int8_t>(unit ? 1 : 7)};
    std::memcpy(data.data() + codes, pal, 4);
    for (uint32_t n = 0; n < N; ++n) {
      float scale = float(n + 1) / 1024, sum = 0;
      for (uint32_t k = 0; k < K; ++k)
        sum += pal[code(k, n)];
      std::memcpy(data.data() + codes + 4 + 4 * n, &scale, 4);
      std::memcpy(data.data() + codes + 4 + 4 * (N + n), &sum, 4);
    }
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<char *>(data.data()), data.size());
    return 0;
  }
  const int src = open(argv[1], O_RDWR), dst = open(argv[2], O_RDWR);
  assert(src >= 0 && dst >= 0);
  int8_t source_palette[4];
  assert(pread(src, source_palette, 4, codes) == 4);
  const int magnitude = source_palette[0] == -1 ? 1 : 7;
  std::vector<uint8_t> arena(bytes + 128, 0xa5);
  std::vector<int8_t> dense(size_t(K) * N);
  {
    PreparedTl2 reader(argv[2], src);
    // Repeated decode -> prefill -> decode switches reuse exactly one slot.
    for (const bool native : {false, true, false, true}) {
      reader.read(src, 0, K, N, native, arena.data());
      if (native)
        assert(hvx_tl2_unpack(arena.data(), K, N, dense.data()) == 0);
      for (uint32_t n = 0; n < N; ++n) {
        int sum = 0;
        for (uint32_t k = 0; k < K; ++k) {
          if (native)
            assert(dense[size_t(k) * N + n] == weight(k, n));
          else
            assert(((arena[at(k, n)] >> (2 * ((k % 32 / 4) % 4))) & 3) ==
                   code(k, n));
          sum += magnitude * weight(k, n);
        }
        float scale;
        int32_t got;
        std::memcpy(&scale, arena.data() + codes + 4 * n, 4);
        std::memcpy(&got, arena.data() + codes + 4 * (N + n), 4);
        assert(scale == float(n + 1) / 1024 && got == sum);
      }
      for (size_t i = bytes; i < arena.size(); ++i)
        assert(arena[i] == 0xa5);
      // PR #4416 keeps the palette outside the arena. An exact-sized slot
      // must keep its trailing canary intact in both representations.
      std::vector<uint8_t> compact(bytes - 4u + 128u, 0xa5);
      int8_t palette[4];
      reader.read(src, 0, K, N, native, compact.data(), palette);
      const size_t weight_bytes = native ? hvx_tl2_weight_bytes(K, N) : codes;
      assert(std::memcmp(compact.data(), arena.data(), weight_bytes) == 0);
      assert(
        std::memcmp(compact.data() + codes, arena.data() + codes, 8u * N) == 0);
      assert(palette[0] == -magnitude && palette[1] == -magnitude &&
             palette[2] == 0 && palette[3] == magnitude);
      for (size_t i = bytes - 4u; i < compact.size(); ++i)
        assert(compact[i] == 0xa5);
      reader.discardAsync(!native);
    }
    reject([&] { reader.read(src, 1, K, N, true, arena.data()); });
    reject([&] { reader.read(src, 0, K, N / 2, true, arena.data()); });
    const std::string receipt = std::string(argv[2]) + ".validated";
    assert(!reader.validatedReceipt());
    assert(reader.validate(src, receipt) == 1u);
    {
      PreparedTl2 fast(argv[2], src);
      assert(fast.validatedReceipt());
      for (const bool native : {false, true}) {
        std::vector<uint8_t> expected(bytes + 128, 0xa5);
        std::vector<uint8_t> actual(bytes + 128, 0xa5);
        std::vector<uint8_t> compact(bytes - 4u + 128u, 0xa5);
        int8_t palette[4]{};
        reader.read(src, 0, K, N, native, expected.data());
        fast.read(src, 0, K, N, native, actual.data());
        assert(expected == actual); // Includes padding and trailing canaries.
        fast.read(src, 0, K, N, native, compact.data(), palette);
        assert(std::memcmp(compact.data(), expected.data(), bytes - 4u) == 0);
        assert(std::memcmp(palette, expected.data() + bytes - 4u, 4u) == 0);
        for (size_t i = bytes - 4u; i < compact.size(); ++i)
          assert(compact[i] == 0xa5);
      }
      fast.read(src, 0, K, N, true, arena.data());
      assert(hvx_tl2_unpack(arena.data(), K, N, dense.data()) == 0);
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t n = 0; n < N; ++n)
          assert(dense[size_t(k) * N + n] == weight(k, n));
    }
    {
      std::ofstream corrupt(receipt, std::ios::binary);
      corrupt << "bad";
    }
    {
      PreparedTl2 fallback(argv[2], src);
      assert(!fallback.validatedReceipt());
      fallback.read(src, 0, K, N, false, arena.data());
    }
    assert(reader.validate(src, receipt) == 1u);
    uint8_t v;
    assert(pread(src, &v, 1, 0) == 1);
    v ^= 1;
    assert(pwrite(src, &v, 1, 0) == 1);
    reject([&] { reader.read(src, 0, K, N, true, arena.data()); });
    reject([&] {
      PreparedTl2 bad(argv[2], src);
      assert(!bad.validatedReceipt());
      bad.read(src, 0, K, N, true, arena.data());
    });
    v ^= 1;
    assert(pwrite(src, &v, 1, 0) == 1);
  }
  uint8_t entry[48];
  assert(pread(dst, entry, 48, 64) == 48);
  const uint64_t payload = preparedFcU64(entry + 24);
  uint8_t v;
  assert(pread(dst, &v, 1, payload) == 1);
  v ^= 1;
  assert(pwrite(dst, &v, 1, payload) == 1);
  reject([&] {
    PreparedTl2 bad(argv[2], src);
    bad.read(src, 0, K, N, true, arena.data());
  });
  assert(ftruncate(dst, payload + 10) == 0);
  reject([&] { PreparedTl2 bad(argv[2], src); });
  close(src);
  close(dst);
  std::cout << "TL2 PREPARED CHECK OK (values, phase reuse, source/payload "
               "corruption, bounds)\n";
}
