// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_thread_manager.cpp
 * @date        20 March 2026
 * @brief       Unit test for ThreadManager
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <thread_manager.h>

// ─── ThreadManager parallel_for Tests ───────────────────────

TEST(ThreadManager, ParallelForEmptyRange) {
  auto &tm = nntrainer::ThreadManager::Global();
  std::atomic<int> count{0};
  tm.parallel_for(0, 0, [&](size_t) { count.fetch_add(1); });
  EXPECT_EQ(count.load(), 0);
}

TEST(ThreadManager, ParallelForSingleElement) {
  auto &tm = nntrainer::ThreadManager::Global();
  std::atomic<int> count{0};
  tm.parallel_for(0, 1, [&](size_t) { count.fetch_add(1); });
  EXPECT_EQ(count.load(), 1);
}

TEST(ThreadManager, ParallelForAllElementsExecuted) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t N = 1000;
  std::vector<std::atomic<int>> flags(N);
  for (size_t i = 0; i < N; ++i)
    flags[i].store(0);

  tm.parallel_for(0, N, [&](size_t i) { flags[i].fetch_add(1); });

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(flags[i].load(), 1)
      << "Index " << i << " not executed exactly once";
  }
}

TEST(ThreadManager, ParallelForWithOffset) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t BEGIN = 10;
  const size_t END = 50;
  std::vector<std::atomic<int>> flags(END);
  for (size_t i = 0; i < END; ++i)
    flags[i].store(0);

  tm.parallel_for(BEGIN, END, [&](size_t i) { flags[i].fetch_add(1); });

  for (size_t i = 0; i < BEGIN; ++i)
    EXPECT_EQ(flags[i].load(), 0);
  for (size_t i = BEGIN; i < END; ++i)
    EXPECT_EQ(flags[i].load(), 1);
}

TEST(ThreadManager, ParallelForAccumulation) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t N = 10000;
  std::atomic<size_t> sum{0};

  tm.parallel_for(
    0, N, [&](size_t i) { sum.fetch_add(i, std::memory_order_relaxed); });

  size_t expected = (N * (N - 1)) / 2;
  EXPECT_EQ(sum.load(), expected);
}

TEST(ThreadManager, ParallelForMultipleCalls) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t N = 100;

  for (int round = 0; round < 20; ++round) {
    std::atomic<int> count{0};
    tm.parallel_for(0, N, [&](size_t) { count.fetch_add(1); });
    EXPECT_EQ(count.load(), (int)N) << "Failed at round " << round;
  }
}

// ─── ThreadManager Query Tests ──────────────────────────────

TEST(ThreadManager, ThreadCounts) {
  auto &tm = nntrainer::ThreadManager::Global();
  EXPECT_GT(tm.getComputeThreadCount(), 0u);

  // compute workers should be <= hardware_concurrency
  unsigned int hw = std::thread::hardware_concurrency();
  if (hw > 1) {
    EXPECT_LE(tm.getComputeThreadCount(), hw);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
