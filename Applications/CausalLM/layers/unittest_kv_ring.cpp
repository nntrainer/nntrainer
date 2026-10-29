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

#if defined(ENABLE_OPENCL)
/**
 * @brief On OpenCL the ring-aware flash arms live in the concat-layout block:
 *        NNTR_MHA_GPU opens it, NNTR_KV_OHWI and the two_conv image arm close
 *        it; the OHWI image arm (Adreno) keeps it open.
 * @details NNTR_KV_OHWI used to be listed as a precondition. It is the
 * opposite: the OHWI K scatter places rows by absolute position (a heap
 * overwrite on a Wcap-high plane) and its readers are linear.
 */
TEST(KVRing, opencl_arm_rule) {
  ScopedEnv ring("NNTR_KV_WINDOW_RING", "1");
  ScopedEnv engine("NNTR_ENGINE", "gpu");
  ScopedEnv ohwi("NNTR_KV_OHWI", nullptr);
  ScopedEnv img("NNTR_KV_IMG_ATTN", nullptr);
  ScopedEnv img2("NNTR_MHA_GPU_IMG", nullptr);
  {
    ScopedEnv mha("NNTR_MHA_GPU", nullptr);
    EXPECT_FALSE(causallm::kvRingArmAvailable());
  }
  ScopedEnv mha("NNTR_MHA_GPU", "1");
  EXPECT_TRUE(causallm::kvRingArmAvailable());
  EXPECT_TRUE(causallm::kvRingEnabled());
  {
    ScopedEnv on("NNTR_KV_OHWI", "1");
    EXPECT_FALSE(causallm::kvRingArmAvailable());
  }
  {
    // The Adreno bundle: the OHWI image arm serves the ring through its
    // sliding mirror, unless the staged (mirror-only) store or the control
    // arm is selected.
    ScopedEnv on("NNTR_KV_IMG_ATTN", "1");
    ScopedEnv stage("NNTR_KV_STAGE", nullptr);
    ScopedEnv ctl("NNTR_KV_IMG_RING", nullptr);
    EXPECT_TRUE(causallm::kvRingArmAvailable());
    {
      ScopedEnv ring_unset("NNTR_KV_WINDOW_RING", nullptr);
      EXPECT_TRUE(causallm::kvRingEnabled(true));
    }
    {
      ScopedEnv off("NNTR_KV_IMG_RING", "0");
      EXPECT_FALSE(causallm::kvRingArmAvailable());
    }
    {
      ScopedEnv st("NNTR_KV_STAGE", "1");
      ScopedEnv nst("NNTR_NO_KV_STAGE", nullptr);
      EXPECT_FALSE(causallm::kvRingArmAvailable());
    }
  }
  {
    ScopedEnv on("NNTR_MHA_GPU_IMG", "1");
    EXPECT_FALSE(causallm::kvRingArmAvailable());
  }
}
#endif

/**
 * @brief A model may make the ring its default; the environment still wins.
 * @details Unset => the model's default decides; '0' opts out of a model
 * default; '1' opts in without one. A model default that no arm can serve is
 * refused exactly like an explicit request (and keeps the linear cache).
 */
TEST(KVRing, model_default_and_env_precedence) {
  ScopedEnv engine("NNTR_ENGINE", "cuda");
  ScopedEnv arm("NNTR_CUDA_ATTN", "1");
  ScopedEnv int8("NNTR_KV_INT8", nullptr);
  ScopedEnv chunk("NNTR_PREFILL_CHUNK", nullptr);
  {
    ScopedEnv ring("NNTR_KV_WINDOW_RING", nullptr);
    EXPECT_FALSE(causallm::kvRingEnabled());
    EXPECT_FALSE(causallm::kvRingEnabled(false));
    EXPECT_TRUE(causallm::kvRingEnabled(true));
    // chunking and the capacity follow the same default
    EXPECT_EQ(causallm::requestedPrefillChunk(false), 0u);
    EXPECT_EQ(causallm::requestedPrefillChunk(true), 4096u);
    EXPECT_EQ(causallm::effectivePrefillChunk(1024, true), 1024u);
    EXPECT_EQ(causallm::kvRingCap(512, 32768, 1024, false), 0u);
    EXPECT_EQ(causallm::kvRingCap(512, 32768, 1024, true), 2048u);
    // a 2K context with a 1024-row chunk: the ring would not shrink the plane
    EXPECT_EQ(causallm::kvRingCap(512, 2048, 1024, true), 0u);
  }
  {
    ScopedEnv ring("NNTR_KV_WINDOW_RING", "0");
    EXPECT_TRUE(causallm::kvRingOptedOut());
    EXPECT_FALSE(causallm::kvRingEnabled(true));
    EXPECT_EQ(causallm::requestedPrefillChunk(true), 0u);
    EXPECT_EQ(causallm::kvRingCap(512, 32768, 1024, true), 0u);
  }
  {
    // only an explicit, non-truthy value is the opt-out (it also withdraws
    // Transformer::chunkLongPrompts()); unset or truthy is not
    ScopedEnv ring("NNTR_KV_WINDOW_RING", nullptr);
    EXPECT_FALSE(causallm::kvRingOptedOut());
  }
  {
    ScopedEnv ring("NNTR_KV_WINDOW_RING", "1");
    EXPECT_FALSE(causallm::kvRingOptedOut());
  }
  {
    ScopedEnv ring("NNTR_KV_WINDOW_RING", "1");
    EXPECT_TRUE(causallm::kvRingEnabled(false));
  }
  {
    ScopedEnv ring("NNTR_KV_WINDOW_RING", nullptr);
    ScopedEnv no_arm("NNTR_CUDA_ATTN", nullptr);
    EXPECT_FALSE(causallm::kvRingEnabled(true));
    ScopedEnv cpu("NNTR_ENGINE", "cpu");
    EXPECT_FALSE(causallm::kvRingEnabled(true));
  }
}

/**
 * @brief The capacity formula, pinned value by value.
 * @details cap = the smallest multiple of C that is >= W + C (W, C on the
 * 64-row grid; (W / C + 2) * C off it), or 0 when that would not shrink
 * max_seq. Any reimplementation of the rule (a kernel-side copy, a future
 * refactor) must reproduce this table exactly; the two consumers sizing and
 * indexing the same buffer differently is a heap overwrite.
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
    // W      max_seq    C      expected = ceil(W / C) * C + C, or 0
    {512, 32768, 4096, 8192},   // W < C   -> 2C
    {1024, 32768, 1024, 2048},  // W == C  -> 2C
    {1024, 2560, 1024, 2048},   // the Gemma4 MoE pack at plane 1024
    {1024, 2560, 512, 1536},    // ... and at plane 512
    {4096, 32768, 1024, 5120},  // W == 4C -> 5C
    {512, 32768, 512, 1024},    //
    {512, 32768, 1024, 2048},   //
    {2048, 32768, 4096, 8192},  //
    {8192, 65536, 4096, 12288}, // W == 2C -> 3C
    {4096, 16384, 4096, 8192},  // 2C < max_seq -> shrinks, keep it
    {1024, 8192, 4096, 0},      // 2C == max_seq -> no benefit
    {1024, 8193, 4096, 8192},   // one row of benefit is still benefit
    {1000, 32768, 1024, 2048},  // W off the 64 grid -> (W / C + 2) * C
    {1024, 32768, 1000, 3000},  // C off the 64 grid -> (W / C + 2) * C
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
 * @brief The plane a request gets is the prompt, on the 64-row grid, bounded by
 *        the validated block and by the window, and never under the pack's own
 *        height.
 * @details The plane is charged on EVERY run while the block it enables is only
 * worth paying for on a run whose prompt is that long (measured on Adreno: a
 * 919-token prompt at plane 4096 costs +375 MiB of honest footprint over plane
 * 1024 and buys nothing -- one block either way). These cases are the policy:
 * a short prompt must come back at the pack's height, a long one at the cap,
 * and nothing may exceed the window.
 */
TEST(KVRing, prefill_plane_follows_the_prompt) {
  const unsigned int pack = 1024u, msl = 16896u;
  // short prompt -> the pack's plane, unchanged (today's footprint)
  EXPECT_EQ(causallm::prefillPlaneFor(0, pack, pack, msl), pack); // no number
  EXPECT_EQ(causallm::prefillPlaneFor(1, pack, pack, msl), pack);
  EXPECT_EQ(causallm::prefillPlaneFor(919, pack, pack, msl), pack);
  EXPECT_EQ(causallm::prefillPlaneFor(1024, pack, pack, msl), pack);
  // past the pack's plane -> the 64-row grid, not the raw token count
  EXPECT_EQ(causallm::prefillPlaneFor(1025, pack, pack, msl), 1088u);
  EXPECT_EQ(causallm::prefillPlaneFor(2047, pack, pack, msl), 2048u);
  EXPECT_EQ(causallm::prefillPlaneFor(2048, pack, pack, msl), 2048u);
  EXPECT_EQ(causallm::prefillPlaneFor(3925, pack, pack, msl), 3968u);
  // at and past the validated block -> the block, and a longer prompt is
  // chunked at it rather than given a taller plane
  EXPECT_EQ(causallm::prefillPlaneFor(4096, pack, pack, msl),
            causallm::kPrefillBlockCap);
  EXPECT_EQ(causallm::prefillPlaneFor(16203, pack, pack, msl),
            causallm::kPrefillBlockCap);
  EXPECT_EQ(causallm::prefillPlaneFor(0xFFFFFFFFu, pack, pack, msl),
            causallm::kPrefillBlockCap); // the round-up must not wrap
  // the window bounds the block from above, below the cap
  EXPECT_EQ(causallm::prefillPlaneFor(8000, 512u, 512u, 2048u), 2048u);
  // a pack that ships a TALL plane keeps it whatever the prompt is ...
  EXPECT_EQ(causallm::prefillPlaneFor(100, 4096u, 4096u, msl), 4096u);
  // ... but the floor cannot climb over the window either
  EXPECT_EQ(causallm::prefillPlaneFor(100, 2048u, 8192u, 2048u), 2048u);
}

/**
 * @brief A bigger block is taken only on an arm it was measured to pay off on,
 *        and NNTR_PREFILL_GROW forces the answer either way.
 */
TEST(KVRing, prefill_block_pays_only_on_the_arms_it_was_measured_on) {
  {
    ScopedEnv eng("NNTR_ENGINE", "cpu");
    ScopedEnv img("NNTR_KV_IMG_ATTN", nullptr);
    ScopedEnv cu("NNTR_CUDA_ATTN", nullptr);
    EXPECT_FALSE(causallm::prefillBlockPays());
    // the Intel Xe flash/XMX arm: gpu engine WITHOUT the image arm -> no
    ScopedEnv gpu("NNTR_ENGINE", "gpu");
    EXPECT_FALSE(causallm::prefillBlockPays());
    { // the Adreno image arm -> ALSO no, since the re-measurement with the
      // prefill/decode boundary drain armed: the block-4096 prefill TPS win was
      // the undrained queue, and end to end the tall plane is 2.4-5.8% slower
      // and +140..+637 MiB at every length on both models. See
      // causallm::prefillBlockPays() for the table.
      ScopedEnv on("NNTR_KV_IMG_ATTN", "1");
      EXPECT_FALSE(causallm::prefillBlockPays());
    }
    { // cuda follows its own attention arm
      ScopedEnv cuda("NNTR_ENGINE", "cuda");
      EXPECT_FALSE(causallm::prefillBlockPays());
      ScopedEnv attn("NNTR_CUDA_ATTN", "1");
      EXPECT_TRUE(causallm::prefillBlockPays());
    }
    // the env forces both directions, so an arm can be A/B'd without a build
    ScopedEnv off("NNTR_PREFILL_GROW", "0");
    EXPECT_FALSE(causallm::prefillBlockPays());
  }
  ScopedEnv cpu("NNTR_ENGINE", "cpu");
  ScopedEnv on("NNTR_PREFILL_GROW", "1");
  EXPECT_TRUE(causallm::prefillBlockPays());
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
 * @details Transformer::getKVCacheRows() allocates `cap ? cap : max_seq` rows
 * and MHACoreLayer clamps a ringed layer's attention read view to
 * min(cache_to, cap). Both are
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

/**
 * @brief The Adreno OHWI mirror height rule: one launch's span, not the ring.
 * @details kvMirrorRows() is what decoupled the mirror from Wcap. The mirror is
 * the linear window the image kernels address, so it holds the rows ONE launch
 * reads -- keys [f+1-W, f+S) with the base floored to the 64-row grid, i.e.
 * (W-1) + C + 63 rows -- while Wcap is a multiple of C and so up to 4x taller.
 * A reimplementation must reproduce this table; too SMALL a mirror makes
 * mha_core's mirror_fits false and a ringed layer then throws.
 */
TEST(KVRing, mirror_rows_table) {
  RingOn on;
  struct Row {
    unsigned int W;
    unsigned int C;
    unsigned int max_seq;
    unsigned int expected;
  };
  const std::vector<Row> table = {
    // W     C      max_seq   expected mirror rows (ring cap in the comment)
    {512, 4096, 32768, 4672},   // cap 8192  -> 4672  (the 0.3B/1.5B model cell)
    {512, 1024, 32768, 1600},   // cap 2048  -> 1600
    {512, 512, 32768, 1024},    // cap 1024  -> 1024 (Wcap-bounded)
    {1024, 1024, 32768, 2048},  // cap 2048  -> 2048 (Wcap-bounded)
    {1024, 512, 2560, 1536},    // cap 1536  -> 1536 (Gemma4 MoE, plane 512)
    {4096, 1024, 32768, 5120},  // cap 5120  -> 5120 (Wcap-bounded)
    {2048, 4096, 32768, 6208},  // cap 8192  -> 6208
    {8192, 4096, 65536, 12288}, // cap 12288 -> 12288 (Wcap-bounded)
  };
  for (const auto &r : table) {
    const unsigned int cap = causallm::kvRingCap(r.W, r.max_seq, r.C);
    ASSERT_NE(cap, 0u) << "W=" << r.W << " C=" << r.C;
    EXPECT_EQ(causallm::kvMirrorRows(cap, r.W, r.C), r.expected)
      << "W=" << r.W << " C=" << r.C << " cap=" << cap;
  }
  // A linear layer (no ring) keeps its own derivation -- 0 means "not mine".
  EXPECT_EQ(causallm::kvMirrorRows(0u, 512u, 4096u), 0u);
  EXPECT_EQ(causallm::kvMirrorRows(8192u, 0u, 4096u), 0u); // full attention
  EXPECT_EQ(causallm::kvMirrorRows(8192u, 512u, 0u), 0u);  // no chunking
  // Never taller than the ring it back-fills from.
  EXPECT_EQ(causallm::kvMirrorRows(2048u, 1024u, 4096u), 2048u);
  // The control arm restores the Wcap-high mirror.
  {
    ScopedEnv tight("NNTR_KV_MIRROR_TIGHT", "0");
    EXPECT_EQ(causallm::kvMirrorRows(8192u, 512u, 4096u), 8192u);
  }
}

/**
 * @brief A mirror always holds one whole launch, over a sweep.
 * @details The span a launch at [f, f+S), S <= C, needs is
 * (f+S) - base. With the exact floor base = f+1-W that is W-1+C, which every
 * mirror must hold (mha_core's mirror_fits is false otherwise and a ringed
 * layer throws). With the 64-floored base it is up to W-1+C+63, which is what
 * the rule asks for -- and gets, except where Wcap itself is tighter than that
 * (a small C against a W that is not a multiple of it), where mha_core already
 * falls back to the exact floor.
 */
TEST(KVRing, mirror_rows_holds_one_launch) {
  RingOn on;
  for (unsigned int C : {256u, 512u, 1024u, 2048u, 4096u}) {
    for (unsigned int W : {128u, 512u, 1000u, 1024u, 4096u, 8192u}) {
      for (unsigned int max_seq : {8192u, 16384u, 32768u, 131072u}) {
        const unsigned int cap = causallm::kvRingCap(W, max_seq, C);
        if (cap == 0u)
          continue;
        const unsigned int rows = causallm::kvMirrorRows(cap, W, C);
        ASSERT_LE(rows, cap) << "W=" << W << " C=" << C;
        ASSERT_GE(rows, W + C) << "W=" << W << " C=" << C; // the exact floor
        if (rows == cap) {
          // Wcap-bounded. Off the 64 grid the 64-aligned base may not fit and
          // mha_core takes the exact floor there. On it, a launch the producer
          // phases on the C grid (it never crosses a multiple of C) still fits
          // with the 64-floored base, which is what keeps ring on == ring off
          // bit for bit on the Adreno image arm.
          if ((W % 64u) != 0u || (C % 64u) != 0u)
            continue;
          for (unsigned int phase : {0u, 1u, 63u, 64u, C / 2u, C - 1u}) {
            const unsigned int f = 8u * C + phase;
            const unsigned int S = C - phase; // up to the next C boundary
            const unsigned int need_lo = f + 1u > W ? f + 1u - W : 0u;
            const unsigned int base = need_lo & ~63u;
            ASSERT_LE(f + S - base, rows)
              << "W=" << W << " C=" << C << " phase=" << phase;
          }
          continue;
        }
        // the worst launch: f is 1 past a 64 boundary, S == C
        for (unsigned int phase : {0u, 1u, 32u, 63u}) {
          const unsigned int f = 8u * C + phase;
          const unsigned int need_lo = f + 1u > W ? f + 1u - W : 0u;
          const unsigned int base = need_lo & ~63u;
          ASSERT_LE(f + C - base, rows)
            << "W=" << W << " C=" << C << " phase=" << phase;
        }
      }
    }
  }
}

/**
 * @brief The DECODE mirror height: W plus one slack band, never the cap.
 * @details This is the height the mirror is re-materialised at when prefill
 * hands over to decode, and the number that makes a big prefill block free:
 * it is a function of W alone, so it does not move when the block does.
 */
TEST(KVRing, decode_mirror_rows_table) {
  EXPECT_EQ(causallm::kvDecodeMirrorRows(512u), 1024u);
  EXPECT_EQ(causallm::kvDecodeMirrorRows(1000u), 1536u); // W -> 1024 grid
  EXPECT_EQ(causallm::kvDecodeMirrorRows(1024u), 1536u);
  EXPECT_EQ(causallm::kvDecodeMirrorRows(4096u), 4608u);
  EXPECT_EQ(causallm::kvDecodeMirrorRows(0u), 0u); // full attention: no resize
  {
    ScopedEnv s("NNTR_KV_DECODE_SLACK", "0"); // control arm: one mirror height
    EXPECT_EQ(causallm::kvDecodeMirrorRows(512u), 0u);
  }
  {
    ScopedEnv s("NNTR_KV_DECODE_SLACK", "256");
    EXPECT_EQ(causallm::kvDecodeMirrorRows(512u), 768u);
  }
  // it must hold one decode step: the query row plus the window, with the base
  // floored to the 64-row grid
  for (unsigned int W : {64u, 128u, 512u, 1000u, 1024u, 4096u}) {
    const unsigned int rows = causallm::kvDecodeMirrorRows(W);
    ASSERT_NE(rows, 0u) << "W=" << W;
    ASSERT_GE(rows, W + 64u) << "W=" << W;
  }
}

/**
 * @brief The eager-slide fallback bound, for a mirror that cannot be resized.
 */
TEST(KVRing, decode_span_table) {
  EXPECT_EQ(causallm::kvDecodeMirrorSpan(512u, 8192u), 1024u);
  EXPECT_EQ(causallm::kvDecodeMirrorSpan(512u, 4672u), 1024u);
  EXPECT_EQ(causallm::kvDecodeMirrorSpan(1000u, 8192u), 1536u); // W -> 1024
  EXPECT_EQ(causallm::kvDecodeMirrorSpan(4096u, 8192u), 4608u);
  // nothing to bound: the mirror is already no taller than the band
  EXPECT_EQ(causallm::kvDecodeMirrorSpan(512u, 1024u), 0u);
  EXPECT_EQ(causallm::kvDecodeMirrorSpan(512u, 1088u), 1024u);
  EXPECT_EQ(causallm::kvDecodeMirrorSpan(0u, 8192u), 0u); // full attention
  EXPECT_EQ(causallm::kvDecodeMirrorSpan(512u, 0u), 0u);  // no mirror
  {
    ScopedEnv s("NNTR_KV_DECODE_SLACK", "0"); // control arm: slide when full
    EXPECT_EQ(causallm::kvDecodeMirrorSpan(512u, 8192u), 0u);
  }
  {
    ScopedEnv s("NNTR_KV_DECODE_SLACK", "256");
    EXPECT_EQ(causallm::kvDecodeMirrorSpan(512u, 8192u), 768u);
  }
  {
    ScopedEnv s("NNTR_KV_DECODE_SLACK", "100"); // rounded up to the 64 grid
    EXPECT_EQ(causallm::kvDecodeMirrorSpan(512u, 8192u), 640u);
  }
  {
    ScopedEnv s("NNTR_KV_DECODE_SLACK", "8"); // floored at one 64-row band
    EXPECT_EQ(causallm::kvDecodeMirrorSpan(512u, 8192u), 576u);
  }
  {
    ScopedEnv s("NNTR_KV_DECODE_SLACK", "-3"); // rejected -> the default
    EXPECT_EQ(causallm::kvDecodeMirrorSpan(512u, 8192u), 1024u);
  }
}

/**
 * @brief PREFILL capacity scales with the block; the DECODE window does not.
 * @details Wcap is a multiple of the prefill chunk because a C-aligned chunk
 * write must not straddle the wrap seam, so a bigger block is a bigger ring,
 * and the mirror used to be Wcap rows high for the same reason. What ONE launch
 * reads back is (W-1) + C + 63 (kvMirrorRows) and what ONE DECODE step reads is
 * W + 63 + 1 (kvDecodeMirrorRows) -- the second is a function of W alone, which
 * is the property this pins: three numbers derived from the same W, two of
 * which move with the block and one of which must not.
 */
TEST(KVRing, decode_capacity_is_independent_of_the_prefill_block) {
  RingOn on;
  const unsigned int W = 512u, max_seq = 32768u;
  for (unsigned int C : {512u, 1024u, 2048u, 4096u}) {
    const unsigned int cap = causallm::kvRingCap(W, max_seq, C);
    const unsigned int rows = causallm::kvMirrorRows(cap, W, C);
    const unsigned int dec = causallm::kvDecodeMirrorRows(W);
    // prefill side: both the ring and the mirror grow with the block
    EXPECT_EQ(cap % C, 0u) << "C=" << C;
    EXPECT_GE(cap, W + C) << "C=" << C;
    EXPECT_GE(rows, W + C) << "C=" << C;
    // decode side: one answer for every block, and never the larger one
    // (at C == W the tight ring is W + C rows, which is exactly W + slack)
    EXPECT_EQ(dec, 1024u) << "C=" << C;
    EXPECT_LE(dec, cap) << "C=" << C;
    EXPECT_LE(dec, rows) << "C=" << C;
    // the eager-slide fallback agrees wherever the mirror is the taller of the
    // two (it is only consulted when the mirror was not resized), and has
    // nothing to bound where the mirror is already that tight
    EXPECT_EQ(causallm::kvDecodeMirrorSpan(W, rows), dec < rows ? dec : 0u)
      << "C=" << C;
  }
  // and the ring cap really does move with the block, so the invariance above
  // is a decoupling rather than a constant on both sides
  EXPECT_NE(causallm::kvRingCap(W, max_seq, 1024u),
            causallm::kvRingCap(W, max_seq, 4096u));
}

/**
 * @brief The mirror->slab boundary sync never addresses a row it does not own.
 * @details sync_kv_slab used to take the ABSOLUTE range to both sides: it wrote
 * the concat slab at row `abs` and read the mirror at row `abs`. On a ringed
 * layer the slab is only `ring_cap` rows tall, so that write left the layer's
 * allocation as soon as abs >= ring_cap -- a latent out-of-bounds write that
 * nothing read back on the image path. The invariants pinned here are the two
 * that make it safe again: every emitted slab row is < ring_cap, and every
 * emitted mirror row is < mirror_rows.
 */
TEST(KVRing, slab_sync_segments_stay_inside_both_allocations) {
  // (a) ring off, mirror at base 0: ONE segment with the old arguments. The
  // pre-fix call is the thing that must not change.
  {
    const auto s = causallm::kvSlabSyncSegments(0, 843, /*ring_cap=*/0,
                                                /*mirror_base=*/0,
                                                /*mirror_rows=*/2048);
    ASSERT_EQ(s.size(), 1u);
    EXPECT_EQ(s[0].mirror_row, 0u);
    EXPECT_EQ(s[0].slab_row, 0u);
    EXPECT_EQ(s[0].rows, 843u);
  }
  // (b) the bug: a ringed layer past the cap. Every row must land inside the
  // ring, and the mirror side must be base-relative.
  {
    const unsigned int cap = 3072u, base = 2048u, rows = 1536u;
    const auto s = causallm::kvSlabSyncSegments(3000, 3200, cap, base, rows);
    unsigned int total = 0;
    for (const auto &g : s) {
      EXPECT_LT(g.slab_row, cap);
      EXPECT_LE((unsigned long)g.slab_row + g.rows, (unsigned long)cap)
        << "a segment straddles the ring seam";
      EXPECT_LE((unsigned long)g.mirror_row + g.rows, (unsigned long)rows)
        << "a segment reads past the mirror";
      total += g.rows;
    }
    EXPECT_EQ(total, 200u);
    // the seam is really crossed here, so this is two pieces
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(s[0].slab_row, 3000u % cap);
    EXPECT_EQ(s[0].rows, cap - (3000u % cap));
    EXPECT_EQ(s[1].slab_row, 0u);
    EXPECT_EQ(s[0].mirror_row, 3000u - base);
    EXPECT_EQ(s[1].mirror_row, 3000u - base + s[0].rows);
  }
  // (c) rows the mirror does not hold are DROPPED, not read out of bounds.
  {
    const auto below = causallm::kvSlabSyncSegments(0, 100, 3072, 2048, 1536);
    EXPECT_TRUE(below.empty()) << "rows below the mirror base were emitted";
    const auto above =
      causallm::kvSlabSyncSegments(2048, 2048 + 4096, 8192, 2048, 1536);
    unsigned int total = 0;
    for (const auto &g : above)
      total += g.rows;
    EXPECT_EQ(total, 1536u) << "the sync ran past the mirror height";
    const auto empty = causallm::kvSlabSyncSegments(500, 500, 0, 0, 2048);
    EXPECT_TRUE(empty.empty());
  }
  // (d) exhaustive: for a small ring/mirror, no emitted row ever leaves either
  // allocation and the pieces tile the clamped range contiguously.
  {
    const unsigned int cap = 16u, rows = 12u;
    for (unsigned int base = 0; base <= 40u; base += 4u)
      for (unsigned int from = 0; from < 48u; ++from)
        for (unsigned int to = from; to < 52u; ++to) {
          const auto s =
            causallm::kvSlabSyncSegments(from, to, cap, base, rows);
          unsigned int prev_abs = 0;
          bool first = true;
          for (const auto &g : s) {
            ASSERT_LE((unsigned long)g.slab_row + g.rows, (unsigned long)cap)
              << " base=" << base << " [" << from << "," << to << ")";
            ASSERT_LE((unsigned long)g.mirror_row + g.rows, (unsigned long)rows)
              << " base=" << base << " [" << from << "," << to << ")";
            ASSERT_GT(g.rows, 0u);
            const unsigned int abs = g.mirror_row + base;
            if (!first) {
              ASSERT_EQ(abs, prev_abs) << "the pieces are not contiguous";
            }
            prev_abs = abs + g.rows;
            first = false;
          }
        }
  }
}

/**
 * @brief The attention arm rule: the fast image arm is the DEFAULT, and
 *        NNTR_DETERMINISTIC=1 is the opt-in reproducible per-layer arm.
 * @details Measured (Adreno 840, per-node output hashes): 20 of 21 diverging
 * run pairs originate in a FULL-attention layer's attention op, and re-measured
 * a windowed read is not clean either (2 divergences in 28 runs of the 1.5B
 * 3925 cell, both first differing at a windowed layer's attention op), so the
 * bound is 0 and the reproducible arm puts EVERY layer on the flash kernels.
 * That arm was the default until its cost was measured (+20~22 % end-to-end,
 * prefill 2~4x slower on Adreno at equal tokens); the fast arm is the default
 * now and reproducibility is one env away. The per-layer predicate stays, and
 * stays pinned, because the bound is one env away too.
 *
 * Pinned here because the whole contract is "no environment = the fast arm,
 * NNTR_DETERMINISTIC=1 = no layer on the image path", and both halves are
 * one-line predicates that a later refactor could invert without any test
 * noticing.
 */
TEST(KVRing, det_arm_per_layer_rule) {
  const size_t kFull = (size_t)UINT_MAX;
  const unsigned int MAXT = 16896u;
  ScopedEnv det("NNTR_DETERMINISTIC", nullptr);
  ScopedEnv allow("NNTR_ALLOW_NONDETERMINISTIC", nullptr);
  ScopedEnv wmax("NNTR_DET_IMG_WINDOW_MAX", nullptr);
  causallm::packAllowsNondeterministic() = false;

  { // the image bundle is not in play at all -> no layer takes it, any arm
    ScopedEnv img("NNTR_KV_IMG_ATTN", nullptr);
    EXPECT_EQ(causallm::detArm(), causallm::DetArm::kFast);
    EXPECT_FALSE(causallm::imageAttnRequested());
    EXPECT_FALSE(causallm::imageAttnLayer(512, MAXT));
    EXPECT_FALSE(causallm::imageAttnLayer(kFull, MAXT));
    ScopedEnv on("NNTR_DETERMINISTIC", "1");
    EXPECT_FALSE(causallm::imageAttnRequested());
    EXPECT_FALSE(causallm::imageAttnLayer(512, MAXT));
  }
  { // value-checked, as before
    ScopedEnv img("NNTR_KV_IMG_ATTN", "0");
    EXPECT_FALSE(causallm::imageAttnRequested());
    EXPECT_FALSE(causallm::imageAttnLayer(512, MAXT));
  }

  ScopedEnv img("NNTR_KV_IMG_ATTN", "1");

  // THE DEFAULT, no environment: the fast arm, EVERY layer on the image path
  // (windowed, full, however the window is spelled).
  EXPECT_EQ(causallm::detArm(), causallm::DetArm::kFast);
  EXPECT_FALSE(causallm::determinismFirst());
  EXPECT_TRUE(causallm::imageAttnRequested());
  EXPECT_TRUE(causallm::imageAttnLayer(512, MAXT));   // 0.3B/1.5B sliding
  EXPECT_TRUE(causallm::imageAttnLayer(kFull, MAXT)); // 0.3B L7/L15 etc.
  EXPECT_TRUE(causallm::imageAttnLayer(1024, MAXT));  // E2B sliding
  EXPECT_TRUE(causallm::imageAttnLayer(MAXT, MAXT));
  EXPECT_TRUE(causallm::imageAttnLayer(0, MAXT));
  { // the bound is not consulted on the fast arm
    ScopedEnv w0("NNTR_DET_IMG_WINDOW_MAX", "0");
    EXPECT_TRUE(causallm::imageAttnLayer(512, MAXT));
    EXPECT_TRUE(causallm::imageAttnLayer(kFull, MAXT));
  }
  { // the explicit spellings of the default resolve to it, unchanged
    ScopedEnv on("NNTR_ALLOW_NONDETERMINISTIC", "1");
    EXPECT_EQ(causallm::detArm(), causallm::DetArm::kFast);
    EXPECT_FALSE(causallm::determinismFirst());
    EXPECT_TRUE(causallm::imageAttnLayer(512, MAXT));
    EXPECT_TRUE(causallm::imageAttnLayer(kFull, MAXT));
  }
  {
    ScopedEnv off("NNTR_DETERMINISTIC", "0");
    EXPECT_EQ(causallm::detArm(), causallm::DetArm::kFast);
    EXPECT_TRUE(causallm::imageAttnLayer(kFull, MAXT));
  }
  { // the pack's own opt-out (nntr_config "allow_nondeterministic") too
    causallm::packAllowsNondeterministic() = true;
    EXPECT_EQ(causallm::detArm(), causallm::DetArm::kFast);
    EXPECT_TRUE(causallm::imageAttnLayer(kFull, MAXT));
    causallm::packAllowsNondeterministic() = false;
    EXPECT_EQ(causallm::detArm(), causallm::DetArm::kFast);
  }

  // NNTR_DETERMINISTIC=1: the reproducible per-layer arm. NO layer takes the
  // image path, because the bound is 0 -- the image read is not reproducible
  // at any window width (the pass walks the chunk plus the window, and a W=512
  // layer was measured originating divergences at chunk 0). The bundle is
  // still "requested" at process level so the ring rule and the program build
  // behave as before.
  ScopedEnv on("NNTR_DETERMINISTIC", "1");
  EXPECT_EQ(causallm::detArm(), causallm::DetArm::kPerLayer);
  EXPECT_TRUE(causallm::determinismFirst());
  EXPECT_TRUE(causallm::imageAttnRequested()); // the bundle is still in play
  EXPECT_EQ(causallm::detImageWindowMax(), 0u);
  EXPECT_FALSE(causallm::imageAttnLayer(512, MAXT));   // 0.3B/1.5B sliding
  EXPECT_FALSE(causallm::imageAttnLayer(kFull, MAXT)); // 0.3B L7/L15 etc.
  EXPECT_FALSE(causallm::imageAttnLayer(1024, MAXT));  // E2B sliding
  // A window at least as wide as the context IS a full-attention read, however
  // it is spelled: the model may hand the window value through verbatim.
  EXPECT_FALSE(causallm::imageAttnLayer(MAXT, MAXT));
  EXPECT_FALSE(causallm::imageAttnLayer(MAXT + 1u, MAXT));
  // A bound of 0 must not read as "a zero-wide window qualifies": 0 spells
  // "no window", which is a full read.
  EXPECT_FALSE(causallm::imageAttnLayer(0, MAXT));
  { // the bound is overridable for experiments, both ways -- 512 restores the
    // per-layer split exactly, which is the A/B the cost rows were taken on.
    ScopedEnv w("NNTR_DET_IMG_WINDOW_MAX", "512");
    EXPECT_EQ(causallm::detImageWindowMax(), 512u);
    EXPECT_TRUE(causallm::imageAttnLayer(512, MAXT));   // the W=512 models
    EXPECT_FALSE(causallm::imageAttnLayer(1024, MAXT)); // E2B
    EXPECT_FALSE(causallm::imageAttnLayer(8192, MAXT));
    EXPECT_FALSE(causallm::imageAttnLayer(kFull, MAXT));
    EXPECT_FALSE(causallm::imageAttnLayer(0, MAXT));
    { // max_timestep unknown (0) must not turn every layer into a full one
      EXPECT_TRUE(causallm::imageAttnLayer(512, 0u));
      EXPECT_FALSE(causallm::imageAttnLayer(kFull, 0u));
    }
  }
  {
    ScopedEnv w("NNTR_DET_IMG_WINDOW_MAX", "8192");
    EXPECT_TRUE(causallm::imageAttnLayer(8192, MAXT));
    EXPECT_TRUE(causallm::imageAttnLayer(1024, MAXT));
  }
  { // and 0 is what the arm already is
    ScopedEnv w0("NNTR_DET_IMG_WINDOW_MAX", "0");
    EXPECT_FALSE(causallm::imageAttnLayer(512, MAXT));
  }
  { // an explicit request outranks the pack's opt-out and the old opt-out
    causallm::packAllowsNondeterministic() = true;
    ScopedEnv allow1("NNTR_ALLOW_NONDETERMINISTIC", "1");
    EXPECT_EQ(causallm::detArm(), causallm::DetArm::kPerLayer);
    EXPECT_FALSE(causallm::imageAttnLayer(512, MAXT));
    causallm::packAllowsNondeterministic() = false;
  }
}

/**
 * @brief The ring survives BOTH arms.
 * @details On the fast default the windowed layers are on the image path with
 * their sliding mirrors -- the ring's validated reader -- and a full-attention
 * layer is not ringed anyway (kvRingCap returns 0 for it), so store and reader
 * match per layer. On the reproducible arm no layer is on the image kernels and
 * the flash prefill/decode kernels modulo-map the cache row; that configuration
 * (bundle requested, ring on) was measured bit-identical run to run on every
 * cell, so the ring is not refused there either -- refusing it only cost
 * memory. Pinned so neither arm silently loses the memory win.
 *
 * Both arms are OpenCL arms: a build without ENABLE_OPENCL has no ring-aware
 * arm for the gpu engine, so there the ring is refused on either arm and the
 * layer keeps the linear cache (kvRingCap returns 0).
 */
TEST(KVRing, ring_survives_both_arms) {
  ScopedEnv ring("NNTR_KV_WINDOW_RING", "1");
  ScopedEnv engine("NNTR_ENGINE", "gpu");
  ScopedEnv mha("NNTR_MHA_GPU", "1");
  ScopedEnv ohwi("NNTR_KV_OHWI", nullptr);
  ScopedEnv img2("NNTR_MHA_GPU_IMG", nullptr);
  ScopedEnv img("NNTR_KV_IMG_ATTN", "1");
  ScopedEnv stage("NNTR_KV_STAGE", nullptr);
  ScopedEnv ctl("NNTR_KV_IMG_RING", nullptr);
  ScopedEnv allow("NNTR_ALLOW_NONDETERMINISTIC", nullptr);
  causallm::packAllowsNondeterministic() = false;
#if defined(ENABLE_OPENCL)
  const bool ring_arm = true;
#else
  const bool ring_arm = false; // no OpenCL arm to serve the gpu engine
#endif
  {
    ScopedEnv det("NNTR_DETERMINISTIC", nullptr); // the fast default
    EXPECT_EQ(causallm::kvRingArmAvailable(), ring_arm);
    // a windowed layer is ringed; a full-attention one never is, on any arm
    EXPECT_EQ(causallm::kvRingCap(512, 16896, 4096, true) > 0u, ring_arm);
    EXPECT_EQ(causallm::kvRingCap(UINT_MAX, 16896, 4096, true), 0u);
  }
  {
    ScopedEnv det("NNTR_DETERMINISTIC", "1"); // the reproducible arm
    EXPECT_EQ(causallm::kvRingArmAvailable(), ring_arm);
    EXPECT_EQ(causallm::kvRingCap(512, 16896, 4096, true) > 0u, ring_arm);
    EXPECT_EQ(causallm::kvRingCap(UINT_MAX, 16896, 4096, true), 0u);
  }
  { // the control arms still close the ring under the image bundle
    ScopedEnv det("NNTR_DETERMINISTIC", nullptr);
    ScopedEnv ir("NNTR_KV_IMG_RING", "0");
    EXPECT_FALSE(causallm::kvRingArmAvailable());
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
