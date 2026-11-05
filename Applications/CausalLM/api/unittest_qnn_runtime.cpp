// SPDX-License-Identifier: Apache-2.0
/** @file unittest_qnn_runtime.cpp
 * @brief Host regression checks for the shared QNN runtime; no DSP required.
 */
#include "../qnn/generate_qnn_utils.h"
#include "../qnn/qnn_kv_cache_manager.h"
#include "../qnn/quick_dot_ai_qnn.h"
#include <gtest/gtest.h>
#include <thread_manager.h>

// Host-only RoPE checks use heap buffers instead of Android RPCMEM.
void *allocate(size_t bytes) { return std::malloc(bytes); }
void deallocate(void *pointer) { std::free(pointer); }

TEST(QnnRuntime, QuantizationRoundsAndSaturates) {
  float input[] = {-100.0f, 0.24f, 0.25f, 0.75f, 40000.0f};
  uint16_t output[5] = {};
  causallm::Quick_Dot_AI_QNN::quantize_uint16_memcpy(input, output, 5, 0.5f, 0);
  EXPECT_EQ((std::vector<uint16_t>{output, output + 5}),
            (std::vector<uint16_t>{0, 0, 1, 2, 65535}));
}

TEST(QnnRuntime, GreedyTiesAndRngAdvance) {
  for (int length : {1, 7, 512, 513, 1025}) {
    std::vector<uint16_t> logits(length, 0);
    const int first = length / 2;
    logits[first] = 65535;
    logits.back() = 65535;
    reseed_sampling_rng();
    auto expected_rng = rng;
    (void)expected_rng();
    EXPECT_EQ(
      ::sample(logits.data(), length, nullptr, 0, 1.0f, 0, 1.0f, 1.0f, 1.0f, 1),
      first);
    EXPECT_EQ(rng(), expected_rng());
  }
}

TEST(QnnRuntime, ReseedingReproducesSampling) {
  uint16_t logits[] = {1, 2, 3, 4};
  std::vector<int> first, second;
  for (auto *result : {&first, &second}) {
    reseed_sampling_rng();
    for (int i = 0; i < 32; ++i)
      result->push_back(
        ::sample(logits, 4, nullptr, 0, 0.1f, 0, 1.0f, 1.0f, 1.0f, 4));
  }
  EXPECT_EQ(first, second);
}

TEST(QnnRuntime, OneRowKeyAppendPreservesOtherSlots) {
  uint8_t input[] = {11, 12, 21, 22, 31, 32};
  uint8_t output[12] = {};
  process_key(input, 1, 3, output, 2, 4, 2);
  EXPECT_EQ((std::vector<uint8_t>{output, output + 12}),
            (std::vector<uint8_t>{0, 0, 11, 0, 0, 0, 21, 0, 0, 0, 31, 0}));
}

TEST(QnnRuntime, RingSyncRejectsOverflowBeforeWorkers) {
  nntrainer::ThreadManagerConfig config;
  config.compute_threads = 2;
  config.enable_affinity = false;
  nntrainer::ThreadManager::setConfig(config);
  causallm::QnnKvCacheManager cache;
  cache.addLayerRowLength(8);
  std::vector<uint8_t> generation(8 * 128, 9), prefill(4 * 128, 0);
  const int index = cache.addGenerationCache("key", generation.data(),
                                             generation.size(), 8, true);
  // More than one entry exercises the parallel region.
  cache.addPrefillCache(prefill.data(), prefill.size(), 4, index, true);
  std::vector<uint8_t> prefill2(4 * 128, 0);
  cache.addPrefillCache(prefill2.data(), prefill2.size(), 4, index, true);
  cache.setLength(5);
  EXPECT_THROW(cache.syncGenerationToPrefillRing(), std::runtime_error);
  EXPECT_EQ(prefill[0], 128);
  cache.setLength(3);
  EXPECT_NO_THROW(cache.syncGenerationToPrefillRing());
  EXPECT_EQ(prefill[0], 9);
  EXPECT_EQ(prefill[3], 128);
  EXPECT_EQ(prefill2[0], 9);
}

TEST(QnnRuntime, PartialRopeLeavesNonRotaryLanesAtIdentity) {
  auto [cosine, sine] = get_cos_sin(3, 8, 1000000, "proportional", 0.25, 1, 16);
  for (int row = 0; row < 3; ++row) {
    for (int lane = 2; lane < 8; ++lane) {
      EXPECT_EQ(cosine[row * 8 + lane], 65535);
      EXPECT_EQ(sine[row * 8 + lane], 32768);
    }
  }
  deallocate(cosine);
  deallocate(sine);
}
