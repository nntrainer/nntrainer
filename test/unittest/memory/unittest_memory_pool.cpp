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

#include <atomic>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <mem_allocator.h>

#include <gtest/gtest.h>

#include <basic_planner.h>
#include <cache_pool.h>
#include <memory_pool.h>
#include <nntrainer_test_util.h>

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

  void alloc(void **ptr, size_t size, size_t alignment) override {
    alloc_count++;
    nntrainer::MemAllocator::alloc(ptr, size, alignment);
  }
  void free(void *ptr) override {
    if (ptr != nullptr)
      free_count++;
    nntrainer::MemAllocator::free(ptr);
  }
  std::string getName() override { return "counting"; }
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
