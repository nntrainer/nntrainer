// SPDX-License-Identifier: Apache-2.0
/**
 * @file expert_read_pool_check.cpp
 * @brief Demand read scheduling, error joining and reuse checks.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#include "htp_expert_read_pool.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <future>

using nntrainer::HtpExpertReadPool;

static void checkpoint(std::future<void> &event) {
  assert(event.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  event.get();
}

static void reuseAndFailure() {
  for (size_t readers : {1u, 2u, 4u, 8u}) {
    HtpExpertReadPool pool(readers);
    std::array<std::atomic<unsigned>, 64> seen{};
    for (unsigned batch = 0; batch < 250; ++batch) {
      for (size_t count : {0u, 1u, 2u, 3u, 9u, 64u}) {
        for (auto &value : seen)
          value = 0;
        pool.run(count, [&](size_t i) { ++seen[i]; });
        for (size_t i = 0; i < seen.size(); ++i)
          assert(seen[i] == unsigned(i < count));
      }
    }
    std::atomic<unsigned> done{0};
    bool caught = false;
    try {
      pool.run(64, [&](size_t i) {
        ++done;
        if (i == 7)
          throw std::runtime_error("test read failure");
      });
    } catch (const std::runtime_error &error) {
      caught = std::string(error.what()) == "test read failure";
    }
    assert(caught && done == 64);
    done = 0;
    pool.run(64, [&](size_t) { ++done; });
    assert(done == 64);
  }
}

static void boundedAndJoined() {
  std::promise<void> all_started, release;
  auto started = all_started.get_future();
  auto allowed = release.get_future().share();
  std::atomic<unsigned> active{0}, init{0}, completed{0};
  HtpExpertReadPool pool(4, [&] { ++init; });
  auto batch = std::async(std::launch::async, [&] {
    bool caught = false;
    try {
      pool.run(4, [&](size_t i) {
        if (++active == 4)
          all_started.set_value();
        allowed.wait();
        --active;
        ++completed;
        if (i == 0)
          throw std::runtime_error("joined");
      });
    } catch (const std::runtime_error &) {
      caught = true;
    }
    assert(caught && completed == 4 && active == 0);
  });
  checkpoint(started);
  assert(active == 4 && init == 3);
  assert(batch.wait_for(std::chrono::milliseconds(5)) ==
         std::future_status::timeout);
  release.set_value();
  checkpoint(batch);
}

static void serializedCallers() {
  HtpExpertReadPool pool(3);
  std::promise<void> first_entered, release;
  auto entered = first_entered.get_future();
  auto allowed = release.get_future().share();
  std::atomic<unsigned> first{0}, second{0};
  auto a = std::async(std::launch::async, [&] {
    pool.run(6, [&](size_t) {
      if (++first == 1)
        first_entered.set_value();
      allowed.wait();
    });
  });
  checkpoint(entered);
  auto b = std::async(std::launch::async, [&] {
    pool.run(9, [&](size_t) {
      assert(first == 6);
      ++second;
    });
  });
  assert(b.wait_for(std::chrono::milliseconds(5)) ==
         std::future_status::timeout);
  assert(second == 0);
  release.set_value();
  checkpoint(a);
  checkpoint(b);
  assert(second == 9);
}

int main() {
  bool rejected = false;
  try {
    HtpExpertReadPool invalid(0);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
  reuseAndFailure();
  boundedAndJoined();
  serializedCallers();
  std::puts("expert_read_pool_check: PASS");
}
