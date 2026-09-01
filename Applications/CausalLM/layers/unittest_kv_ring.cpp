// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_kv_ring.cpp
 * @date   01 September 2026
 * @brief  Host tests for the sliding-window KV ring rule (kv_ring.h)
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details The ring rule is consumed by the model side (KV allocation height)
 * and by the layer side (cache-row modulo map). A disagreement between them is
 * an out-of-bounds write rather than a wrong answer, so these tests pin the
 * rule itself: the capacity formula, the invariants that make it safe, the
 * boundary returns, the row wrap, and the chunk clamp. They need no GPU and run
 * in CI.
 *
 * NOT covered here, and deliberately so: chunked prefill producing the same
 * logits as a single-block prefill. That comparison needs a GPU attention arm
 * (the ring only turns on where one resolves), so it stays a device test.
 */

#include <kv_ring.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

/** @brief RAII setter/restorer for one environment variable. */
class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : key(name) {
    const char *old = std::getenv(name);
    had_old = (old != nullptr);
    if (had_old)
      old_value = old;
    if (value == nullptr)
      ::unsetenv(name);
    else
      ::setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_old)
      ::setenv(key.c_str(), old_value.c_str(), 1);
    else
      ::unsetenv(key.c_str());
  }

private:
  std::string key;
  std::string old_value;
  bool had_old;
};

/**
 * @brief Put the process in a state where the ring is requested AND a
 *        ring-aware attention arm resolves, so kvRingCap() can return non-zero.
 * @details engine=cuda + NNTR_CUDA_ATTN is the one combination whose answer
 * does not depend on whether this build defines ENABLE_OPENCL.
 */
class RingOn {
public:
  RingOn() :
    ring("NNTR_KV_WINDOW_RING", "1"),
    engine("NNTR_ENGINE", "cuda"),
    arm("NNTR_CUDA_ATTN", "1"),
    int8("NNTR_KV_INT8", nullptr),
    chunk("NNTR_PREFILL_CHUNK", nullptr) {}

private:
  ScopedEnv ring, engine, arm, int8, chunk;
};

} // namespace

/**
 * @brief The ring is opt-in: nothing turns it on without
 *        NNTR_KV_WINDOW_RING.
 */
TEST(KVRing, disabled_by_default) {
  ScopedEnv ring("NNTR_KV_WINDOW_RING", nullptr);
  ScopedEnv engine("NNTR_ENGINE", "cuda");
  ScopedEnv arm("NNTR_CUDA_ATTN", "1");
  EXPECT_FALSE(causallm::kvRingEnabled());
  EXPECT_EQ(causallm::kvRingCap(512, 32768, 4096), 0u);
  // and with the ring off, chunking is off too
  ScopedEnv chunk("NNTR_PREFILL_CHUNK", nullptr);
  EXPECT_EQ(causallm::requestedPrefillChunk(), 0u);
}

/** @brief '0' is an explicit opt-out, and so is any other falsy spelling. */
TEST(KVRing, explicit_zero_disables) {
  ScopedEnv engine("NNTR_ENGINE", "cuda");
  ScopedEnv arm("NNTR_CUDA_ATTN", "1");
  ScopedEnv ring("NNTR_KV_WINDOW_RING", "0");
  EXPECT_FALSE(causallm::kvRingEnabled());
}

/**
 * @brief A requested ring is refused when no ring-aware attention arm
 *        resolves, so the linear full-height cache stays in place.
 */
TEST(KVRing, refused_without_a_ring_aware_arm) {
  ScopedEnv ring("NNTR_KV_WINDOW_RING", "1");
  ScopedEnv engine("NNTR_ENGINE", "cuda");
  {
    ScopedEnv arm("NNTR_CUDA_ATTN", nullptr);
    EXPECT_FALSE(causallm::kvRingArmAvailable());
    EXPECT_FALSE(causallm::kvRingEnabled());
    EXPECT_EQ(causallm::kvRingCap(512, 32768, 4096), 0u);
  }
  {
    ScopedEnv arm("NNTR_CUDA_ATTN", "1");
    EXPECT_TRUE(causallm::kvRingArmAvailable());
    EXPECT_TRUE(causallm::kvRingEnabled());
  }
  // the cpu engine can never host the ring, whatever else is set
  ScopedEnv cpu("NNTR_ENGINE", "cpu");
  ScopedEnv arm("NNTR_CUDA_ATTN", "1");
  EXPECT_FALSE(causallm::kvRingEngineEligible());
  EXPECT_FALSE(causallm::kvRingEnabled());
}

/**
 * @brief The capacity formula, pinned value by value.
 * @details cap = (W / C + 2) * C, or 0 when that would not shrink max_seq.
 * Any reimplementation of the rule (a kernel-side copy, a future refactor) must
 * reproduce this table exactly; the two consumers sizing and indexing the same
 * buffer differently is a heap overwrite.
 */
TEST(KVRing, capacity_table) {
  RingOn on;
  struct Row {
    unsigned int W;
    unsigned int max_seq;
    unsigned int C;
    unsigned int expected;
  };
  const std::vector<Row> table = {
    // W      max_seq    C      expected = (W / C + 2) * C, or 0
    {512, 32768, 4096, 8192},   // W < C   -> 2C
    {1024, 32768, 1024, 3072},  // W == C  -> 3C
    {4096, 32768, 1024, 6144},  // W == 4C -> 6C
    {512, 32768, 512, 1536},    //
    {512, 32768, 1024, 2048},   //
    {2048, 32768, 4096, 8192},  //
    {8192, 65536, 4096, 16384}, // W == 2C -> 4C
    {4096, 16384, 4096, 12288}, // 3C < max_seq -> shrinks, keep it
    {1024, 8192, 4096, 0},      // 2C == max_seq -> no benefit
    {1024, 8193, 4096, 8192},   // one row of benefit is still benefit
  };
  for (const auto &r : table)
    EXPECT_EQ(causallm::kvRingCap(r.W, r.max_seq, r.C), r.expected)
      << "W=" << r.W << " max_seq=" << r.max_seq << " C=" << r.C;
}

/**
 * @brief The two invariants that make a ringed write safe, over a sweep.
 * @details cap is a multiple of C (a C-aligned chunk write never straddles the
 * wrap seam) and cap >= W + C (the live window [pos-W+1, pos+C) never
 * self-collides mod cap). A cap violating either is silent corruption.
 */
TEST(KVRing, capacity_invariants) {
  RingOn on;
  for (unsigned int C : {256u, 512u, 1024u, 2048u, 4096u}) {
    for (unsigned int W : {128u, 512u, 1000u, 1024u, 4096u, 8192u}) {
      for (unsigned int max_seq : {8192u, 16384u, 32768u, 131072u}) {
        const unsigned int cap = causallm::kvRingCap(W, max_seq, C);
        if (cap == 0u)
          continue; // no ring for this cell
        EXPECT_EQ(cap % C, 0u) << "W=" << W << " C=" << C;
        EXPECT_GE(cap, W + C) << "W=" << W << " C=" << C;
        EXPECT_LT(cap, max_seq) << "W=" << W << " C=" << C;
      }
    }
  }
}

/** @brief Every documented boundary that must return 0 (no ring). */
TEST(KVRing, boundary_returns_zero) {
  RingOn on;
  EXPECT_EQ(causallm::kvRingCap(0, 32768, 4096), 0u);     // full attention
  EXPECT_EQ(causallm::kvRingCap(32768, 32768, 4096), 0u); // W == max_seq
  EXPECT_EQ(causallm::kvRingCap(40000, 32768, 4096), 0u); // W > max_seq
  EXPECT_EQ(causallm::kvRingCap(512, 32768, 0), 0u);      // no chunking
  EXPECT_EQ(causallm::kvRingCap(512, 4096, 4096), 0u);    // cap >= max_seq
  EXPECT_EQ(causallm::kvRingCap(512, 1024, 4096), 0u);    // cap > max_seq
}

/**
 * @brief The host row map is exactly the kernels' `n % ring_cap`.
 * @details mha_core::cacheRow() and every ring-aware kernel must agree on the
 * physical row for an absolute position, including across the seam and for
 * several full wraps.
 */
TEST(KVRing, cache_row_wraps_like_the_kernels) {
  const unsigned int cap = 3072;
  for (unsigned long n = 0; n < 4ul * cap + 7ul; ++n)
    ASSERT_EQ(causallm::kvCacheRow(n, cap), n % cap) << "n=" << n;
  // cap == 0 is the identity: ring off must be bit-identical to the linear path
  for (unsigned long n : {0ul, 1ul, 4095ul, 1000000ul})
    EXPECT_EQ(causallm::kvCacheRow(n, 0), n);
  // the seam itself
  EXPECT_EQ(causallm::kvCacheRow(cap - 1, cap), cap - 1);
  EXPECT_EQ(causallm::kvCacheRow(cap, cap), 0ul);
  EXPECT_EQ(causallm::kvCacheRow(cap + 1, cap), 1ul);
}

/**
 * @brief The chunk the prefill runs is the request clamped to the activation
 *        plane, and both the model and the layer must read that same number.
 */
TEST(KVRing, effective_chunk_clamps_to_the_plane) {
  RingOn on;
  ScopedEnv chunk("NNTR_PREFILL_CHUNK", "4096");
  EXPECT_EQ(causallm::requestedPrefillChunk(), 4096u);
  EXPECT_EQ(causallm::effectivePrefillChunk(1024), 1024u);
  EXPECT_EQ(causallm::effectivePrefillChunk(4096), 4096u);
  EXPECT_EQ(causallm::effectivePrefillChunk(8192), 4096u);
  EXPECT_EQ(causallm::effectivePrefillChunk(0), 4096u); // plane unknown
  // sizing the ring off the unclamped request would over-allocate by 4x
  EXPECT_EQ(
    causallm::kvRingCap(512, 32768, causallm::effectivePrefillChunk(1024)),
    2048u);
  EXPECT_EQ(causallm::kvRingCap(512, 32768, causallm::requestedPrefillChunk()),
            8192u);
}

/**
 * @brief A non-positive or unparseable NNTR_PREFILL_CHUNK is rejected, not
 *        wrapped into a ~4e9 unsigned that the (W/C + 2) * C arithmetic eats.
 */
TEST(KVRing, rejects_non_positive_chunk) {
  RingOn on;
  for (const char *bad : {"-1", "0", "abc", "-4096", "12x"}) {
    ScopedEnv chunk("NNTR_PREFILL_CHUNK", bad);
    // rejected => falls back to the ring's own 4096, never to a huge value
    EXPECT_EQ(causallm::requestedPrefillChunk(), 4096u) << "value=" << bad;
    EXPECT_EQ(causallm::effectivePrefillChunk(1024), 1024u) << "value=" << bad;
  }
  ScopedEnv good("NNTR_PREFILL_CHUNK", "2048");
  EXPECT_EQ(causallm::requestedPrefillChunk(), 2048u);
}

/**
 * @brief The per-layer eligibility truth table, which the model side and the
 *        layer side both feed from their own view of the same two facts.
 */
TEST(KVRing, layer_eligibility_truth_table) {
  RingOn on;
  EXPECT_TRUE(causallm::kvRingLayerEligible(/*sink=*/false, /*external=*/true));
  EXPECT_FALSE(causallm::kvRingLayerEligible(/*sink=*/true, /*external=*/true));
  EXPECT_FALSE(
    causallm::kvRingLayerEligible(/*sink=*/false, /*external=*/false));
  EXPECT_FALSE(
    causallm::kvRingLayerEligible(/*sink=*/true, /*external=*/false));
  // the int8 KV cache is allocated at full max_seq and written with absolute
  // rows on the layer side, so it must disqualify the ring on BOTH sides
  ScopedEnv int8("NNTR_KV_INT8", "1");
  EXPECT_FALSE(
    causallm::kvRingLayerEligible(/*sink=*/false, /*external=*/true));
}

/**
 * @brief The read view the layer builds always fits the allocation the model
 *        made, over a sweep of positions.
 * @details Transformer::getKVCacheRows() allocates max(cap, max_seq) rows and
 * MHACoreLayer clamps its attention read view to min(cache_to, cap). Both are
 * reproduced here from the same kv_ring.h entry points; the assertion is the
 * property that a drift between them would break -- the view never runs off the
 * end of the buffer, and a ringed layer never reads more rows than it stores.
 */
TEST(KVRing, read_view_fits_the_allocation) {
  RingOn on;
  for (unsigned int C : {512u, 1024u, 4096u}) {
    ScopedEnv chunk("NNTR_PREFILL_CHUNK", std::to_string(C).c_str());
    for (unsigned int plane : {0u, 1024u, 4096u, 32768u}) {
      const unsigned int chunk_run = causallm::effectivePrefillChunk(plane);
      for (unsigned int W : {0u, 512u, 4096u, 32768u}) {
        for (unsigned int max_seq : {4096u, 32768u}) {
          const unsigned int cap =
            causallm::kvRingLayerEligible(/*sink=*/false, /*external=*/true)
              ? causallm::kvRingCap(W, max_seq, chunk_run)
              : 0u;
          const unsigned int rows = cap ? cap : max_seq; // getKVCacheRows()
          for (unsigned int cache_to = 1; cache_to <= max_seq;
               cache_to += (max_seq / 8)) {
            const unsigned int read_rows =
              cap ? std::min(cache_to, cap) : cache_to; // the layer's view
            ASSERT_LE(read_rows, rows)
              << "W=" << W << " C=" << C << " plane=" << plane
              << " max_seq=" << max_seq << " cache_to=" << cache_to;
            if (cap != 0u) {
              ASSERT_LE(read_rows, cap);
            }
          }
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
