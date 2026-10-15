// SPDX-License-Identifier: Apache-2.0
/** @file tl2_native_packing_check.cpp
 * @brief Compare QS2-to-TL2 conversion with independent logical weights.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "htp_tl2_native_packing.h"
#include "hvx_gemv_u8tl2.h"
#include <cassert>
#include <cstdio>

static void check(uint32_t K, uint32_t N, bool unit) {
  const size_t bytes = static_cast<size_t>(K) * N / 4u;
  std::vector<uint8_t> file(bytes + 4u + 8u * N, 0u), native;
  std::vector<int8_t> dense(static_cast<size_t>(K) * N);
  std::vector<int32_t> sums(N);
  const int8_t palette[4] = {static_cast<int8_t>(unit ? -1 : -8),
                             static_cast<int8_t>(unit ? -1 : -7), 0,
                             static_cast<int8_t>(unit ? 1 : 7)};
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n) {
      const uint32_t q = (k * 7u + n * 3u + k / 5u) % 4u;
      const uint32_t r = k % 32u,
                     old = (r / 8u) * 128u + (n % 32u) * 4u + r % 4u;
      const size_t at =
        (static_cast<size_t>(k / 32u) * (N / 32u) + n / 32u) * 256u +
        (old / 256u) * 128u + old % 128u;
      file[at] |= q << (2u * ((r / 4u) % 2u) + 4u * ((r / 8u) % 2u));
      const int value = q < 2 ? -1 : q == 2 ? 0 : 1;
      dense[static_cast<size_t>(k) * N + n] = value;
      sums[n] += (unit ? 1 : 7) * value;
    }
  std::memcpy(file.data() + bytes, palette, 4u);
  for (uint32_t n = 0; n < N; ++n) {
    const float scale = float(n + 1) / 1024;
    std::memcpy(file.data() + bytes + 4u + 4u * n, &scale, 4u);
  }
  std::vector<uint8_t> baseline(file);
  nntrainer::prepareSymmetricTl2(baseline.data(), K, N, nullptr);
  nntrainer::prepareSymmetricTl2(file.data(), K, N, &native);
  assert(baseline == file);
  std::vector<uint8_t> expected(hvx_tl2_weight_bytes(K, N));
  assert(hvx_tl2_pack(dense.data(), K, N, expected.data()) == 0);
  assert(native == expected);
  const auto *prepared_palette =
    reinterpret_cast<const int8_t *>(file.data() + bytes);
  assert(htp_tl2_palette_magnitude(prepared_palette) == (unit ? 1u : 7u));
  for (uint32_t n = 0; n < N; ++n) {
    float sum, scale;
    std::memcpy(&scale, file.data() + bytes + 4u + 4u * n, 4u);
    assert(scale == float(n + 1) / 1024);
    std::memcpy(&sum, file.data() + bytes + 4u + 4u * (N + n), 4u);
    assert(sum == sums[n]);
  }
  std::printf("Native TL2 packing exact K=%u N=%u magnitude=%u\n", K, N,
              unit ? 1u : 7u);
}

int main() {
  const int8_t mixed[] = {-1, 0, 7, 7}, asymmetric[] = {-8, -7, 0, 7};
  const int8_t zero[] = {0, 0, 0, 0};
  assert(htp_tl2_palette_magnitude(mixed) == 0);
  assert(htp_tl2_palette_magnitude(asymmetric) == 0);
  assert(htp_tl2_palette_magnitude(zero) == 1);
  for (bool unit : {false, true}) {
    check(32, 32, unit);
    check(256, 192, unit);
    check(2816, 1408, unit);
    check(704, 2816, unit);
  }
  return 0;
}
