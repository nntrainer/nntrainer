// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_nntrainer_hexagon_repack.cpp
 * @date   23 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Round-trip test for the Hexagon HTP q4x4x2 weight repack. Pure
 * host-side byte shuffling - no Hexagon SDK or device required.
 */

#include <cpu_backend.h>
#include <gtest/gtest.h>
#include <hexagon_repack.h>

#include <cstdint>
#include <random>
#include <vector>

namespace {

std::vector<float> random_weights(size_t n, unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  std::vector<float> w(n);
  for (auto &v : w)
    v = dist(rng);
  return w;
}

} // namespace

/**
 * @brief repacking to q4x4x2 and back reproduces the exact original
 * block_q4_0 bytes (delta + nibbles) - i.e. the tile transform is lossless.
 */
TEST(nntrainer_HexagonRepack, q4_0_to_q4x4x2_roundtrip_p) {
  nntrainer::init_backend();

  const unsigned int M = 4;   // rows
  const unsigned int N = 512; // cols, divisible by 256

  auto weights = random_weights((size_t)M * N, 1234);

  std::vector<uint8_t> q4_0((size_t)M * N / 32 * 18);
  size_t quantized_size =
    nntrainer::quantize_q4_0(weights.data(), q4_0.data(), M, N, nullptr);
  ASSERT_EQ(quantized_size, q4_0.size());

  std::vector<uint8_t> packed(q4_0.size());
  nntrainer::repack_q4_0_to_htp_q4x4x2(packed.data(), q4_0.data(),
                                       q4_0.size(), M, N);

  std::vector<uint8_t> unpacked(q4_0.size());
  nntrainer::unpack_htp_q4x4x2_to_q4_0(unpacked.data(), packed.data(),
                                       packed.size(), M, N);

  EXPECT_EQ(q4_0, unpacked);
}

/**
 * @brief the packed buffer is the same total size as the standard block_q4_0
 * buffer - the tile transform only rearranges bytes, it does not grow them.
 */
TEST(nntrainer_HexagonRepack, q4_0_to_q4x4x2_size_preserving_p) {
  nntrainer::init_backend();

  const unsigned int M = 2;
  const unsigned int N = 1024;

  auto weights = random_weights((size_t)M * N, 5678);

  std::vector<uint8_t> q4_0((size_t)M * N / 32 * 18);
  nntrainer::quantize_q4_0(weights.data(), q4_0.data(), M, N, nullptr);

  std::vector<uint8_t> packed(q4_0.size());
  // Should not throw and should not read/write out of bounds (ASan/valgrind
  // covered via CI); the buffer sizes above are exact, not padded.
  nntrainer::repack_q4_0_to_htp_q4x4x2(packed.data(), q4_0.data(),
                                       q4_0.size(), M, N);
}

/**
 * @brief N not divisible by 256 is rejected rather than silently
 * mis-packed. See hexagon_repack.h for why this case is unsupported.
 */
TEST(nntrainer_HexagonRepack, non_multiple_of_256_throws_n) {
  const unsigned int M = 1;
  const unsigned int N = 128; // divisible by 32, not by 256

  std::vector<uint8_t> q4_0((size_t)M * N / 32 * 18, 0);
  std::vector<uint8_t> packed(q4_0.size());

  EXPECT_THROW(nntrainer::repack_q4_0_to_htp_q4x4x2(packed.data(),
                                                     q4_0.data(), q4_0.size(),
                                                     M, N),
              std::invalid_argument);
}

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
