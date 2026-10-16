// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file unittest_memory_planning.cpp
 * @date 11 August 2021
 * @brief Memory Planner Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>
#include <memory>

#include <basic_planner.h>
#include <memory_planner.h>
#include <memory_pool.h>
#include <nntrainer_test_util.h>
#include <numeric>
#include <optimized_v1_planner.h>

constexpr unsigned int MEM_BYTES = 128;
constexpr unsigned int MEM_QUANT = 100;
constexpr unsigned int INTERVAL_SIZE = 5;

/**
 * @class MemoryPlannerValidate
 * @brief Memory planning validation
 */
class MemoryPlannerValidate
  : public ::testing::TestWithParam<std::shared_ptr<nntrainer::MemoryPlanner>> {
public:
  /**
   * @brief SetUp test cases here
   *
   */
  void SetUp(void) { planner = GetParam(); }

  /**
   * @brief Release test resources
   *
   */
  void TearDown(void) {}

  std::shared_ptr<nntrainer::MemoryPlanner> planner;
};

/**
 * @brief Validate the provided layout does not overlap
 */
static bool validateNoOverlap(const std::vector<size_t> &memory_size,
                              const std::vector<size_t> &memory_offset,
                              unsigned int size) {
  std::vector<bool> mem_overlap(size, false);
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    for (unsigned int mem = memory_offset[idx];
         mem < memory_offset[idx] + memory_size[idx]; mem++) {
      if (mem_overlap[mem])
        return false;
      mem_overlap[mem] = true;
    }
  }

  /** This ensures that there are no unnecessary gaps in between */
  EXPECT_EQ(mem_overlap.size(),
            std::accumulate(mem_overlap.begin(), mem_overlap.end(), 0u));

  return true;
}

/**
 * @brief Validate the provided layout does completely overlap
 */
static bool validateAllOverlap(const std::vector<size_t> &memory_size,
                               const std::vector<size_t> &memory_offset) {
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    if (memory_offset[idx] != memory_offset[0]) {
      return false;
    }
  }

  return true;
}

/**
 * @brief Validate the provided layout does partially overlap
 *
 * @note this test assumes that the memory validity is sorted such that start
 * validity at index idx = idx, and end validity at index idx > idx.
 */
static bool validateIntervalOverlap(
  const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
  const std::vector<size_t> &memory_size,
  const std::vector<size_t> &memory_offset) {
  std::vector<unsigned int> valid_intervals;
  std::vector<unsigned int> sorted_by_validity(memory_size.size());

  /** sort the intervals by their validity for comparison */
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    sorted_by_validity[idx] = idx;
  }
  std::sort(
    sorted_by_validity.begin(), sorted_by_validity.end(),
    [&memory_validity](auto const &idx1, auto const &idx2) -> unsigned int {
      if (memory_validity[idx1].first == memory_validity[idx2].first)
        return memory_validity[idx1].second < memory_validity[idx2].second;
      return memory_validity[idx1].first < memory_validity[idx2].first;
    });

  for (unsigned int idx : sorted_by_validity) {
    /**
     * intervals which have finished before the start of the current intervals
     * must be popped
     */
    auto expired_intervals = std::remove_if(
      valid_intervals.begin(), valid_intervals.end(),
      [memory_validity, idx](auto const &jdx) {
        return memory_validity[jdx].second <= memory_validity[idx].first;
      });
    valid_intervals.erase(expired_intervals, valid_intervals.end());

    /** this allocation must not overlap with any existing valid allocations */
    auto new_mem_start = memory_offset[idx];
    auto new_mem_end = memory_offset[idx] + memory_size[idx];
    for (auto const &vi : valid_intervals) {
      auto vi_mem_start = memory_offset[vi];
      auto vi_mem_end = memory_offset[vi] + memory_size[vi];
      EXPECT_TRUE(vi_mem_start >= new_mem_end || vi_mem_end <= new_mem_start);
    }

    valid_intervals.push_back(idx);
  }

  return true;
}

/**
 * @brief Validate the provided layout does not overflow outside the given
 * size
 */
static bool validateOverflow(const std::vector<size_t> &memory_size,
                             const std::vector<size_t> &memory_offset,
                             size_t pool_size) {
  for (unsigned int idx = 0; idx < memory_size.size(); idx++)
    if (memory_offset[idx] + memory_size[idx] > pool_size)
      return false;

  return true;
}

/**
 * @brief Test memory planning with full overlap of requested memories
 * @details validityIsFullOverlapping - validify is the full overlapping memory
 * requests handled correctly.
 */
TEST_P(MemoryPlannerValidate, full_overlap) {
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);

  std::vector<size_t> memory_size(MEM_QUANT);
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    memory_size[idx] = dist(rng);
  }
  std::vector<std::pair<unsigned int, unsigned int>> memory_validity(MEM_QUANT,
                                                                     {1, 2});
  std::vector<size_t> memory_offset;

  std::vector<bool> memory_is_wgrad;
  size_t pool_size = planner->planLayout(memory_size, memory_validity,
                                         memory_offset, memory_is_wgrad, 0);

  EXPECT_EQ(pool_size,
            std::accumulate(memory_size.begin(), memory_size.end(), 0u));
  EXPECT_TRUE(validateNoOverlap(memory_size, memory_offset, pool_size));
  EXPECT_TRUE(validateOverflow(memory_size, memory_offset, pool_size));
}

/**
 * @brief Test memory planning with no overlap of requested memories
 */
TEST_P(MemoryPlannerValidate, none_overlap) {
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);

  std::vector<size_t> memory_size(MEM_QUANT);
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    memory_size[idx] = dist(rng);
  }

  std::vector<std::pair<unsigned int, unsigned int>> memory_validity(MEM_QUANT);
  for (unsigned int idx = 0; idx < memory_validity.size(); idx++) {
    memory_validity[idx] = {idx, idx + 1};
  }

  std::vector<size_t> memory_offset;
  std::vector<bool> memory_is_wgrad;
  size_t pool_size = planner->planLayout(memory_size, memory_validity,
                                         memory_offset, memory_is_wgrad, 0);

  EXPECT_TRUE(validateOverflow(memory_size, memory_offset, pool_size));
  if (planner->getType() == nntrainer::BasicPlanner::type) {
    EXPECT_EQ(pool_size,
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
    EXPECT_TRUE(validateNoOverlap(memory_size, memory_offset, pool_size));
  } else {
    EXPECT_EQ(pool_size,
              *std::max_element(memory_size.begin(), memory_size.end()));
    EXPECT_TRUE(validateAllOverlap(memory_size, memory_offset));
  }
}

/**
 * @brief Test memory planning with partial overlap of requested memories
 */
TEST_P(MemoryPlannerValidate, partial_overlap) {
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);
  std::uniform_int_distribution<unsigned int> dist_interval(1, INTERVAL_SIZE);

  std::vector<size_t> memory_size(MEM_QUANT);
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    memory_size[idx] = dist(rng);
  }

  std::vector<std::pair<unsigned int, unsigned int>> memory_validity(MEM_QUANT);
  for (unsigned int idx = 0; idx < memory_validity.size(); idx++) {
    memory_validity[idx] = {idx, idx + dist_interval(rng)};
  }
  std::shuffle(memory_validity.begin(), memory_validity.end(),
               std::default_random_engine(0));

  std::vector<size_t> memory_offset;
  std::vector<bool> memory_is_wgrad;
  size_t pool_size = planner->planLayout(memory_size, memory_validity,
                                         memory_offset, memory_is_wgrad, 0);

  EXPECT_TRUE(validateOverflow(memory_size, memory_offset, pool_size));
  if (planner->getType() == nntrainer::BasicPlanner::type) {
    EXPECT_EQ(pool_size,
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
    EXPECT_TRUE(validateNoOverlap(memory_size, memory_offset, pool_size));
  } else {
    EXPECT_GE(pool_size,
              *std::max_element(memory_size.begin(), memory_size.end()));
    EXPECT_LE(pool_size,
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
    EXPECT_TRUE(
      validateIntervalOverlap(memory_validity, memory_size, memory_offset));
  }
}
/**
 * @brief validate memory full overlap
 */
TEST_P(MemoryPlannerValidate, validate_memory_full_overlap) {
  nntrainer::MemoryPool pool;
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);
  std::uniform_int_distribution<unsigned int> dist_interval(1, INTERVAL_SIZE);

  std::vector<unsigned int> tokens(MEM_QUANT);
  std::vector<size_t> memory_size(MEM_QUANT);
  std::vector<std::shared_ptr<nntrainer::MemoryData>> ptrs(MEM_QUANT);

  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    memory_size[idx] = dist(rng);
    unsigned int start = 1;
    unsigned int end = start + dist_interval(rng);
    EXPECT_NO_THROW(tokens[idx] =
                      pool.requestMemory(memory_size[idx], start, end));
  }

  EXPECT_NO_THROW(pool.planLayout(*planner.get()));
  EXPECT_EQ(pool.size(),
            std::accumulate(memory_size.begin(), memory_size.end(), 0u));
  EXPECT_NO_THROW(pool.allocate());

  for (unsigned int idx = 0; idx < MEM_QUANT; idx++)
    EXPECT_NO_THROW(ptrs[idx] = pool.getMemory(tokens[idx]));

  /** write data to memory */
  for (unsigned int idx = 0; idx < MEM_QUANT; idx++)
    memset(ptrs[idx]->getAddr(), idx, memory_size[idx]);

  /** verify data in memory */
  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    std::vector<unsigned char> golden(memory_size[idx], idx);
    EXPECT_EQ(0, memcmp(ptrs[idx]->getAddr(), golden.data(), memory_size[idx]));
  }

  pool.deallocate();
}

/**
 * @brief validate memory no overlap
 */
TEST_P(MemoryPlannerValidate, validate_memory_no_overlap) {
  nntrainer::MemoryPool pool;
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);
  std::uniform_int_distribution<unsigned int> dist_interval(1, INTERVAL_SIZE);

  std::vector<unsigned int> tokens(MEM_QUANT);
  std::vector<size_t> memory_size(MEM_QUANT);
  std::vector<std::shared_ptr<nntrainer::MemoryData>> ptrs(MEM_QUANT);

  unsigned int prev_idx = 0;
  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    memory_size[idx] = dist(rng);
    unsigned int start = prev_idx;
    unsigned int end = start + dist_interval(rng);
    EXPECT_NO_THROW(tokens[idx] =
                      pool.requestMemory(memory_size[idx], start, end));
    prev_idx = end;
  }

  EXPECT_NO_THROW(pool.planLayout(*planner.get()));
  if (planner->getType() == nntrainer::BasicPlanner::type) {
    EXPECT_EQ(pool.size(),
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
  } else {
    EXPECT_EQ(pool.size(),
              *std::max_element(memory_size.begin(), memory_size.end()));
  }
  EXPECT_NO_THROW(pool.allocate());

  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    EXPECT_NO_THROW(ptrs[idx] = pool.getMemory(tokens[idx]));
  }

  // TODO need to investigate what's wrong with OptimizedV1Planner
  if (planner->getType() != nntrainer::OptimizedV1Planner::type) {
    /** write data to memory */
    for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
      std::memset(ptrs[idx]->getAddr(), idx, memory_size[idx]);
    }

    /** verify data in memory */
    for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
      std::vector<unsigned char> golden(memory_size[idx], idx);
      EXPECT_EQ(
        0, std::memcmp(ptrs[idx]->getAddr(), golden.data(), memory_size[idx]));
    }
  }

  pool.deallocate();
}

/**
 * @brief validate memory partial overlap
 */
TEST_P(MemoryPlannerValidate, validate_memory_partial_overlap) {
  nntrainer::MemoryPool pool;
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);
  std::uniform_int_distribution<unsigned int> dist_interval(1, INTERVAL_SIZE);
  std::uniform_int_distribution<unsigned int> dist_interval_start(1, 100);

  std::vector<unsigned int> tokens(MEM_QUANT);
  std::vector<size_t> memory_size(MEM_QUANT);
  std::vector<std::shared_ptr<nntrainer::MemoryData>> ptrs(MEM_QUANT);

  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    memory_size[idx] = dist(rng);
    unsigned int start = dist_interval_start(rng);
    unsigned int end = start + dist_interval(rng);
    EXPECT_NO_THROW(tokens[idx] =
                      pool.requestMemory(memory_size[idx], start, end));
  }

  EXPECT_NO_THROW(pool.planLayout(*planner.get()));
  if (planner->getType() == nntrainer::BasicPlanner::type) {
    EXPECT_EQ(pool.size(),
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
  } else {
    EXPECT_GE(pool.size(),
              *std::max_element(memory_size.begin(), memory_size.end()));
    EXPECT_LE(pool.size(),
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
  }
  EXPECT_NO_THROW(pool.allocate());

  for (unsigned int idx = 0; idx < MEM_QUANT; idx++)
    EXPECT_NO_THROW(ptrs[idx] = pool.getMemory(tokens[idx]));

  // TODO need to investigate what's wrong with OptimizedV1Planner
  if (planner->getType() != nntrainer::OptimizedV1Planner::type) {
    /** write data to memory */
    for (unsigned int idx = 0; idx < MEM_QUANT; idx++)
      memset(ptrs[idx]->getAddr(), idx, memory_size[idx]);

    /** verify data in memory */
    for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
      std::vector<unsigned char> golden(memory_size[idx], idx);
      EXPECT_EQ(0,
                memcmp(ptrs[idx]->getAddr(), golden.data(), memory_size[idx]));
    }
  }

  pool.deallocate();
}

/**
 * @brief Reuse and split a larger expired block below a live residual.
 */
TEST(MemoryPlanner, reuse_larger_expired_block) {
  nntrainer::OptimizedV1Planner planner(true);
  const std::vector<size_t> sizes{128, 32, 48, 80};
  const std::vector<std::pair<unsigned int, unsigned int>> validity{
    {0, 1}, {0, 4}, {1, 3}, {1, 3}};
  std::vector<size_t> offsets;
  std::vector<bool> gradients;
  const size_t bytes =
    planner.planLayout(sizes, validity, offsets, gradients, 0);
  EXPECT_EQ(bytes, 160u);
  EXPECT_TRUE(validateIntervalOverlap(validity, sizes, offsets));
  EXPECT_TRUE(validateOverflow(sizes, offsets, bytes));
}

/**
 * @brief Adjacent expired blocks form one reusable gap.
 */
TEST(MemoryPlanner, coalesce_expired_blocks) {
  nntrainer::OptimizedV1Planner planner(true);
  const std::vector<size_t> sizes{32, 48, 16, 64};
  const std::vector<std::pair<unsigned int, unsigned int>> validity{
    {0, 1}, {0, 2}, {0, 4}, {2, 3}};
  std::vector<size_t> offsets;
  std::vector<bool> gradients;
  const size_t bytes =
    planner.planLayout(sizes, validity, offsets, gradients, 0);
  EXPECT_EQ(bytes, 96u);
  EXPECT_TRUE(validateIntervalOverlap(validity, sizes, offsets));
  EXPECT_TRUE(validateOverflow(sizes, offsets, bytes));
}

GTEST_PARAMETER_TEST(
  MemoryPlanner, MemoryPlannerValidate,
  ::testing::Values(std::make_shared<nntrainer::BasicPlanner>(),
                    std::make_shared<nntrainer::OptimizedV1Planner>(),
                    std::make_shared<nntrainer::OptimizedV1Planner>(true)));
