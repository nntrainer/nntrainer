// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file unittest_memory_pool.cpp
 * @date 11 August 2021
 * @brief Memory Pool Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <mem_allocator.h>

#include <gtest/gtest.h>

#include <basic_planner.h>
#include <cache_pool.h>
#include <memory_pool.h>
#include <nntrainer_test_util.h>
#include <optimized_v2_planner.h>

/**
 * @brief MemoryPool Test Class
 */
class MemoryPoolTest
  : public ::testing::TestWithParam<std::shared_ptr<nntrainer::MemoryPool>> {
public:
  void SetUp(void) { pool = GetParam(); }

  void TearDown(void) { EXPECT_NO_THROW(pool->clear()); }

  std::shared_ptr<nntrainer::MemoryPool> pool;
};

/**
 * @brief creation and destruction
 */
TEST_P(MemoryPoolTest, create_destroy) {
  EXPECT_NO_THROW(nntrainer::MemoryPool());
}

/**
 * @brief request 0 sized memory
 */
TEST_P(MemoryPoolTest, request_mem_01_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.requestMemory(0, 1, 2), std::invalid_argument);
}

/**
 * @brief request memory when starts after it ends
 */
TEST_P(MemoryPoolTest, request_mem_02_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.requestMemory(1, 3, 2), std::invalid_argument);
}

/**
 * @brief request memory with 0 valid time
 */
TEST_P(MemoryPoolTest, request_mem_03_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.requestMemory(1, 4, 4), std::invalid_argument);
}

/**
 * @brief request memory after allocate
 */
TEST(MemoryPool, request_mem_04_n) {
  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.requestMemory(1, 4, 5));
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_THROW(pool.requestMemory(1, 5, 6), std::invalid_argument);
}

/**
 * @brief request memory
 */
TEST_P(MemoryPoolTest, request_mem_04_p) {
  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.requestMemory(1, 4, 5));
}

/**
 * @brief plan layout without reqeustMemory
 */
TEST_P(MemoryPoolTest, plan_layout_01_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.planLayout(nntrainer::BasicPlanner()), std::runtime_error);
}

/**
 * @brief plan layout after allocate
 */
TEST_P(MemoryPoolTest, plan_layout_02_p) {
  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.requestMemory(1, 4, 5));
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_THROW(pool.planLayout(nntrainer::BasicPlanner()), std::runtime_error);
}

/**
 * @brief plan layout
 */
TEST_P(MemoryPoolTest, plan_layout_03_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_EQ(1u, pool.size());

  pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_EQ(2u, pool.size());
}

/**
 * @brief deallocate
 */
TEST_P(MemoryPoolTest, deallocate_01_p) {
  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.deallocate());

  pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief allocate without requestMemory
 */
TEST_P(MemoryPoolTest, allocate_01_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.allocate(), std::runtime_error);
}

/**
 * @brief allocate without planLayout
 */
TEST_P(MemoryPoolTest, allocate_02_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 4, 5);
  EXPECT_THROW(pool.allocate(), std::runtime_error);
}

/**
 * @brief allocate aftrer allocate
 */
TEST_P(MemoryPoolTest, allocate_03_n) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_THROW(pool.allocate(), std::runtime_error);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief allocate
 */
TEST_P(MemoryPoolTest, allocate_04_n) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(3, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_EQ(3u, pool.size());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NO_THROW(pool.deallocate());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief size of the pool
 */
TEST_P(MemoryPoolTest, size_01_p) {
  nntrainer::MemoryPool pool;

  EXPECT_EQ(pool.size(), 0u);
}

/**
 * @brief size of the pool
 */
TEST_P(MemoryPoolTest, size_02_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 4, 5);
  EXPECT_EQ(pool.size(), 0u);
}

/**
 * @brief size of the pool
 */
TEST_P(MemoryPoolTest, size_03_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 4, 5);
  EXPECT_EQ(pool.size(), 0u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.size(), 1u);

  pool.allocate();
  EXPECT_EQ(pool.size(), 1u);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief min requirement
 */
TEST_P(MemoryPoolTest, min_mem_req_01_p) {
  nntrainer::MemoryPool pool;

  EXPECT_EQ(pool.minMemoryRequirement(), 0u);
}

/**
 * @brief min requirement
 */
TEST_P(MemoryPoolTest, min_mem_req_02_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 4, 5);
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);
}

/**
 * @brief min requirement
 */
TEST_P(MemoryPoolTest, min_mem_req_03_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 4, 5);
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  /** exact overlap */
  pool.requestMemory(2, 4, 5);
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  /** ending overlap */
  pool.requestMemory(3, 2, 5);
  EXPECT_EQ(pool.minMemoryRequirement(), 6u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 6u);

  /** start overlap */
  pool.requestMemory(4, 4, 8);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 10u);

  /** complete overlap */
  pool.requestMemory(5, 1, 10);
  EXPECT_EQ(pool.minMemoryRequirement(), 15u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 15u);
}

/**
 * @brief min requirement
 */
TEST_P(MemoryPoolTest, min_mem_req_04_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 5, 10);
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  /** partial overlap */
  pool.requestMemory(2, 1, 8);
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  /** ending overlap */
  pool.requestMemory(3, 7, 12);
  EXPECT_EQ(pool.minMemoryRequirement(), 6u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 6u);
}

/**
 * @brief min requirement
 */
TEST_P(MemoryPoolTest, min_mem_req_05_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 5, 10);
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  /** partial overlap */
  pool.requestMemory(2, 1, 8);
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  /** ending overlap with matching ends */
  pool.requestMemory(3, 8, 12);
  EXPECT_EQ(pool.minMemoryRequirement(), 4u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 4u);
}

/**
 * @brief min requirement
 */
TEST_P(MemoryPoolTest, min_mem_req_06_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 5, 10);
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  /** partial overlap */
  pool.requestMemory(2, 1, 5);
  EXPECT_EQ(pool.minMemoryRequirement(), 2u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 2u);

  /** ending overlap with matching ends */
  pool.requestMemory(3, 10, 12);
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  /** ending overlap with matching ends */
  pool.requestMemory(1, 12, 13);
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);
}

/**
 * @brief get memory
 */
TEST_P(MemoryPoolTest, get_memory_01_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.getMemory(1), std::invalid_argument);
}

/**
 * @brief get memory
 */
TEST_P(MemoryPoolTest, get_memory_02_n) {
  nntrainer::MemoryPool pool;

  auto idx = pool.requestMemory(1, 4, 5);
  EXPECT_THROW(pool.getMemory(idx), std::invalid_argument);
}

/**
 * @brief get memory
 */
TEST_P(MemoryPoolTest, get_memory_03_n) {
  nntrainer::MemoryPool pool;

  auto idx = pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_ANY_THROW(pool.getMemory(idx + 1));

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief get memory
 */
TEST_P(MemoryPoolTest, get_memory_04_p) {
  nntrainer::MemoryPool pool;
  std::shared_ptr<nntrainer::MemoryData> mem;

  auto idx = pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NO_THROW(mem = pool.getMemory(idx));
  EXPECT_NE(mem, nullptr);

  EXPECT_NO_THROW(pool.deallocate());
}

GTEST_PARAMETER_TEST(
  MemoryPool, MemoryPoolTest,
  ::testing::Values(std::make_shared<nntrainer::MemoryPool>(),
                    std::make_shared<nntrainer::CachePool>("tmp pool")));

namespace {

/**
 * @brief Test allocator that records every alloc/free call so the
 *        unit test can assert MemoryPool routes through it instead
 *        of going around to libc.
 */
class CountingAllocator : public nntrainer::MemAllocator {
public:
  std::atomic<int> alloc_count{0};
  std::atomic<int> free_count{0};
  std::vector<void *> allocated_ptrs;
  std::vector<void *> freed_ptrs;
  std::vector<size_t> allocated_sizes;

  void alloc(void **ptr, size_t size, size_t alignment) override {
    alloc_count++;
    nntrainer::MemAllocator::alloc(ptr, size, alignment);
    allocated_ptrs.push_back(*ptr);
    allocated_sizes.push_back(size);
  }
  void free(void *ptr) override {
    if (ptr != nullptr) {
      free_count++;
      freed_ptrs.push_back(ptr);
    }
    nntrainer::MemAllocator::free(ptr);
  }
  std::string getName() override { return "counting"; }
};

class QnnCountingAllocator : public CountingAllocator {
public:
  std::string getName() override { return "qnn"; }
};

class ForcedOffsetPlanner : public nntrainer::MemoryPlanner {
public:
  explicit ForcedOffsetPlanner(std::vector<size_t> offsets_) :
    offsets(std::move(offsets_)) {}

  size_t planLayout(
    const std::vector<size_t> &memory_size,
    const std::vector<std::pair<unsigned int, unsigned int>> &,
    std::vector<size_t> &memory_offset, std::vector<bool> &,
    size_t) const override {
    if (offsets.size() != memory_size.size())
      throw std::runtime_error("ForcedOffsetPlanner offset count mismatch");

    memory_offset = offsets;
    size_t required_size = 0;
    for (unsigned int idx = 0; idx < memory_size.size(); idx++)
      required_size = std::max(required_size, offsets[idx] + memory_size[idx]);

    return required_size;
  }

  const std::string getType() const override { return "forced-offset"; }

private:
  std::vector<size_t> offsets;
};

class ExpectWGradCountPlanner : public nntrainer::MemoryPlanner {
public:
  explicit ExpectWGradCountPlanner(size_t expected_n_wgrad_) :
    expected_n_wgrad(expected_n_wgrad_) {}

  size_t planLayout(
    const std::vector<size_t> &memory_size,
    const std::vector<std::pair<unsigned int, unsigned int>> &,
    std::vector<size_t> &memory_offset, std::vector<bool> &,
    size_t n_wgrad) const override {
    if (n_wgrad != expected_n_wgrad)
      throw std::runtime_error("unexpected weight-gradient request count");

    memory_offset.assign(memory_size.size(), 0);
    size_t required_size = 0;
    for (size_t size : memory_size)
      required_size = std::max(required_size, size);

    return required_size;
  }

  const std::string getType() const override { return "expect-wgrad-count"; }

private:
  size_t expected_n_wgrad;
};

} // namespace

/**
 * @brief MemoryPool routes allocate()/deallocate() through the
 *        injected MemAllocator (1 alloc, 1 matching free).
 */
TEST(MemoryPoolAllocator, allocate_routes_through_injected_allocator) {
  auto counter = std::make_shared<CountingAllocator>();
  nntrainer::MemoryPool pool(counter);

  EXPECT_NO_THROW(pool.requestMemory(1024, 4, 5));
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));

  EXPECT_EQ(counter->alloc_count.load(), 0);
  EXPECT_NO_THROW(pool.allocate());
  EXPECT_EQ(counter->alloc_count.load(), 1);
  EXPECT_EQ(counter->free_count.load(), 0);

  EXPECT_NO_THROW(pool.deallocate());
  EXPECT_EQ(counter->free_count.load(), 1);
}

/**
 * @brief QNN MemoryPool allocates one backend buffer per distinct planned
 *        offset instead of returning interior pointers into a single pool.
 */
TEST(MemoryPoolAllocator, qnn_allocate_uses_distinct_planned_offsets) {
  auto counter = std::make_shared<QnnCountingAllocator>();
  nntrainer::MemoryPool pool(counter);

  auto first = pool.requestMemory(64, 1, 2);
  auto second = pool.requestMemory(64, 2, 3);
  EXPECT_NO_THROW(pool.planLayout(ForcedOffsetPlanner({0, 16})));

  EXPECT_NO_THROW(pool.allocate());

  auto ptrs = pool.getMemoryPtrs();
  ASSERT_EQ(ptrs.size(), 2u);
  ASSERT_EQ(counter->allocated_ptrs.size(), 2u);
  EXPECT_TRUE(pool.isAllocated());
  EXPECT_EQ(counter->alloc_count.load(), 2);
  EXPECT_EQ(counter->allocated_sizes[0], 64u);
  EXPECT_EQ(counter->allocated_sizes[1], 64u);
  EXPECT_EQ(pool.getMemoryPoolAddress(), nullptr);
  EXPECT_EQ(ptrs[first - 1], counter->allocated_ptrs[0]);
  EXPECT_EQ(ptrs[second - 1], counter->allocated_ptrs[1]);
  EXPECT_NE(ptrs[second - 1],
            static_cast<char *>(counter->allocated_ptrs[0]) + 16);

  auto first_mem = pool.getMemory(first);
  auto second_mem = pool.getMemory(second);
  ASSERT_NE(first_mem, nullptr);
  ASSERT_NE(second_mem, nullptr);
  EXPECT_EQ(first_mem->getAddr<void>(), counter->allocated_ptrs[0]);
  EXPECT_EQ(second_mem->getAddr<void>(), counter->allocated_ptrs[1]);

  EXPECT_NO_THROW(pool.deallocate());
  EXPECT_FALSE(pool.isAllocated());
  EXPECT_EQ(counter->free_count.load(), 2);
  ASSERT_EQ(counter->freed_ptrs.size(), 2u);
  EXPECT_EQ(counter->freed_ptrs[0], counter->allocated_ptrs[0]);
  EXPECT_EQ(counter->freed_ptrs[1], counter->allocated_ptrs[1]);
}

/**
 * @brief QNN MemoryPool shares a backend allocation for requests that reuse
 *        the same planned offset and sizes it for the largest alias.
 */
TEST(MemoryPoolAllocator, qnn_allocate_reuses_matching_planned_offset) {
  auto counter = std::make_shared<QnnCountingAllocator>();
  nntrainer::MemoryPool pool(counter);

  auto first = pool.requestMemory(64, 1, 2);
  auto second = pool.requestMemory(128, 2, 3);
  EXPECT_NO_THROW(pool.planLayout(ForcedOffsetPlanner({0, 0})));

  EXPECT_NO_THROW(pool.allocate());

  auto ptrs = pool.getMemoryPtrs();
  ASSERT_EQ(ptrs.size(), 2u);
  ASSERT_EQ(counter->allocated_ptrs.size(), 1u);
  ASSERT_EQ(counter->allocated_sizes.size(), 1u);
  EXPECT_TRUE(pool.isAllocated());
  EXPECT_EQ(counter->alloc_count.load(), 1);
  EXPECT_EQ(counter->allocated_sizes[0], 128u);
  EXPECT_EQ(pool.getMemoryPoolAddress(), nullptr);
  EXPECT_EQ(ptrs[first - 1], counter->allocated_ptrs[0]);
  EXPECT_EQ(ptrs[second - 1], counter->allocated_ptrs[0]);

  auto first_mem = pool.getMemory(first);
  auto second_mem = pool.getMemory(second);
  ASSERT_NE(first_mem, nullptr);
  ASSERT_NE(second_mem, nullptr);
  EXPECT_EQ(first_mem->getAddr<void>(), counter->allocated_ptrs[0]);
  EXPECT_EQ(second_mem->getAddr<void>(), counter->allocated_ptrs[0]);

  EXPECT_NO_THROW(pool.deallocate());
  EXPECT_FALSE(pool.isAllocated());
  EXPECT_EQ(counter->free_count.load(), 1);
}

/**
 * @brief QNN MemoryPool keeps mem_pool null, so allocate() must still reject
 *        a second call while owned buffers exist.
 */
TEST(MemoryPoolAllocator, qnn_allocate_twice_throws_with_owned_buffers) {
  auto counter = std::make_shared<QnnCountingAllocator>();
  nntrainer::MemoryPool pool(counter);

  EXPECT_NO_THROW(pool.requestMemory(64, 1, 2));
  EXPECT_NO_THROW(pool.planLayout(ForcedOffsetPlanner({0})));

  EXPECT_NO_THROW(pool.allocate());
  EXPECT_EQ(pool.getMemoryPoolAddress(), nullptr);
  EXPECT_THROW(pool.allocate(), std::runtime_error);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief QNN MemoryPool must reject re-planning while allocated even though
 *        mem_pool remains null and ownership lives in owned_buffers_.
 */
TEST(MemoryPoolAllocator, qnn_plan_layout_after_allocate_throws) {
  auto counter = std::make_shared<QnnCountingAllocator>();
  nntrainer::MemoryPool pool(counter);

  EXPECT_NO_THROW(pool.requestMemory(64, 1, 2));
  EXPECT_NO_THROW(pool.planLayout(ForcedOffsetPlanner({0})));

  EXPECT_NO_THROW(pool.allocate());
  EXPECT_EQ(pool.getMemoryPoolAddress(), nullptr);
  EXPECT_THROW(pool.planLayout(ForcedOffsetPlanner({0})), std::runtime_error);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief QNN MemoryPool can reuse an existing planned layout after
 *        deallocate(), matching the default contiguous allocator lifecycle.
 */
TEST(MemoryPoolAllocator, qnn_allocate_after_deallocate_reuses_layout) {
  auto counter = std::make_shared<QnnCountingAllocator>();
  nntrainer::MemoryPool pool(counter);

  auto first = pool.requestMemory(64, 1, 2);
  auto second = pool.requestMemory(128, 2, 3);
  EXPECT_NO_THROW(pool.planLayout(ForcedOffsetPlanner({0, 16})));

  EXPECT_NO_THROW(pool.allocate());
  auto first_ptrs = pool.getMemoryPtrs();
  ASSERT_EQ(first_ptrs.size(), 2u);
  EXPECT_EQ(first_ptrs[first - 1], counter->allocated_ptrs[0]);
  EXPECT_EQ(first_ptrs[second - 1], counter->allocated_ptrs[1]);
  EXPECT_NO_THROW(pool.deallocate());

  EXPECT_NO_THROW(pool.allocate());
  auto second_ptrs = pool.getMemoryPtrs();
  ASSERT_EQ(second_ptrs.size(), 2u);
  ASSERT_EQ(counter->allocated_ptrs.size(), 4u);
  ASSERT_EQ(counter->allocated_sizes.size(), 4u);
  EXPECT_EQ(counter->allocated_sizes[2], 64u);
  EXPECT_EQ(counter->allocated_sizes[3], 128u);
  EXPECT_EQ(second_ptrs[first - 1], counter->allocated_ptrs[2]);
  EXPECT_EQ(second_ptrs[second - 1], counter->allocated_ptrs[3]);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief Adding a new memory request after deallocate() invalidates the old
 *        planned layout; callers must run planLayout() again before allocate().
 */
TEST(MemoryPoolAllocator, qnn_request_after_deallocate_requires_new_layout) {
  auto counter = std::make_shared<QnnCountingAllocator>();
  nntrainer::MemoryPool pool(counter);

  EXPECT_NO_THROW(pool.requestMemory(64, 1, 2));
  EXPECT_NO_THROW(pool.planLayout(ForcedOffsetPlanner({0})));
  EXPECT_NO_THROW(pool.allocate());
  EXPECT_NO_THROW(pool.deallocate());

  EXPECT_NO_THROW(pool.requestMemory(128, 3, 4));
  EXPECT_THROW(pool.allocate(), std::runtime_error);

  EXPECT_NO_THROW(pool.planLayout(ForcedOffsetPlanner({0})));
  EXPECT_NO_THROW(pool.allocate());
  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief Starting a new request set after deallocate() resets stale
 *        weight-gradient request counts before optimized replanning.
 */
TEST(MemoryPoolAllocator, request_after_deallocate_resets_wgrad_count) {
  auto counter = std::make_shared<QnnCountingAllocator>();
  nntrainer::MemoryPool pool(counter);

  EXPECT_NO_THROW(pool.requestMemory(64, 1, 3, {},
                                     nntrainer::TensorLifespan::MAX_LIFESPAN,
                                     true));
  EXPECT_NO_THROW(pool.requestMemory(64, 2, 4, {},
                                     nntrainer::TensorLifespan::MAX_LIFESPAN,
                                     true));
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());
  EXPECT_NO_THROW(pool.deallocate());

  EXPECT_NO_THROW(pool.requestMemory(128, 3, 4));
  EXPECT_NO_THROW(pool.planLayout(ExpectWGradCountPlanner(0)));
  EXPECT_NO_THROW(pool.planLayout(nntrainer::OptimizedV2Planner()));
  EXPECT_NO_THROW(pool.allocate());
  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief Starting a new request set after deallocate() also resets stale
 *        weight-gradient counts when no layout had been planned yet.
 */
TEST(MemoryPoolAllocator,
     request_after_unplanned_deallocate_resets_wgrad_count) {
  {
    nntrainer::MemoryPool check_pool;
    EXPECT_NO_THROW(check_pool.requestMemory(
      64, 1, 3, {}, nntrainer::TensorLifespan::MAX_LIFESPAN, true));
    EXPECT_NO_THROW(check_pool.planLayout(ExpectWGradCountPlanner(1)));
  }

  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.requestMemory(64, 1, 3, {},
                                     nntrainer::TensorLifespan::MAX_LIFESPAN,
                                     true));
  EXPECT_NO_THROW(pool.deallocate());

  EXPECT_NO_THROW(pool.requestMemory(128, 3, 4));
  EXPECT_NO_THROW(pool.planLayout(nntrainer::OptimizedV2Planner()));
  EXPECT_NO_THROW(pool.allocate());
  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief getAllocator() returns the injected allocator.
 */
TEST(MemoryPoolAllocator, get_allocator_returns_injected) {
  auto counter = std::make_shared<CountingAllocator>();
  nntrainer::MemoryPool pool(counter);
  EXPECT_EQ(pool.getAllocator().get(), counter.get());
  EXPECT_EQ(pool.getAllocator()->getName(), "counting");
}

/**
 * @brief deallocate() is a no-op on a never-allocated pool — must
 *        not call free() on a null mem_pool. Regression test for
 *        the rewrite that replaced the explicit `if (mem_pool)`
 *        gate with owned_buffers_ tracking.
 */
TEST(MemoryPoolAllocator, deallocate_without_allocate_is_safe) {
  auto counter = std::make_shared<CountingAllocator>();
  nntrainer::MemoryPool pool(counter);

  EXPECT_NO_THROW(pool.deallocate());
  EXPECT_EQ(counter->free_count.load(), 0);
}

/**
 * @brief Default-constructed MemoryPool keeps using base CPU
 *        MemAllocator (backward compatibility).
 */
TEST(MemoryPoolAllocator, default_pool_uses_cpu_allocator) {
  nntrainer::MemoryPool pool;
  ASSERT_NE(pool.getAllocator(), nullptr);
  EXPECT_EQ(pool.getAllocator()->getName(), "cpu");
}
