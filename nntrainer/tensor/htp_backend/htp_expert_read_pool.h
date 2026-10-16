// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_expert_read_pool.h
 * @brief Synchronous bounded workers for demand expert reads.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#ifndef NNTR_HTP_EXPERT_READ_POOL_H
#define NNTR_HTP_EXPERT_READ_POOL_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace nntrainer {

/** The calling thread also reads. Workers sleep between batches, instead of
 * waking the model's full compute pool and spinning after a small I/O batch.
 * Calls are serialized; all callbacks finish before run() returns or throws.
 * As with other thread pools, destruction must not race a caller of run().
 */
class HtpExpertReadPool {
public:
  /** @param readers Maximum readers including the calling thread.
   * @param initialize Optional worker affinity setup; must not throw. */
  explicit HtpExpertReadPool(size_t readers,
                             std::function<void()> initialize = {}) {
    if (readers == 0)
      throw std::invalid_argument("Expert read pool needs a reader");
    try {
      for (size_t i = 1; i < readers; ++i)
        workers_.emplace_back([this, initialize] {
          if (initialize)
            initialize();
          work();
        });
    } catch (...) {
      stop();
      throw;
    }
  }
  ~HtpExpertReadPool() { stop(); }
  HtpExpertReadPool(const HtpExpertReadPool &) = delete;
  HtpExpertReadPool &operator=(const HtpExpertReadPool &) = delete;

  /** Runs each index once, joins every worker, then rethrows the first error.
   */
  void run(size_t count, std::function<void(size_t)> read) {
    std::lock_guard<std::mutex> dispatch(dispatch_);
    if (count <= 1 || workers_.empty()) {
      std::exception_ptr error;
      for (size_t i = 0; i < count; ++i) {
        try {
          read(i);
        } catch (...) {
          if (!error)
            error = std::current_exception();
        }
      }
      if (error)
        std::rethrow_exception(error);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      read_ = std::move(read);
      count_ = count;
      next_.store(0, std::memory_order_relaxed);
      error_ = nullptr;
      pending_ = workers_.size();
      ++generation_;
    }
    ready_.notify_all();
    readJobs();
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
    read_ = {};
    auto error = error_;
    lock.unlock();
    if (error)
      std::rethrow_exception(error);
  }

private:
  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    ready_.notify_all();
    for (auto &worker : workers_)
      if (worker.joinable())
        worker.join();
  }

  void readJobs() {
    for (size_t i;
         (i = next_.fetch_add(1, std::memory_order_relaxed)) < count_;) {
      try {
        read_(i);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_)
          error_ = std::current_exception();
      }
    }
  }

  void work() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock,
                    [this, seen] { return stop_ || generation_ != seen; });
        if (stop_)
          return;
        seen = generation_;
      }
      readJobs();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--pending_ == 0)
          done_.notify_one();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex dispatch_, mutex_;
  std::condition_variable ready_, done_;
  std::function<void(size_t)> read_;
  std::atomic<size_t> next_{0};
  size_t count_ = 0, pending_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
  std::exception_ptr error_;
};

} // namespace nntrainer

#endif // NNTR_HTP_EXPERT_READ_POOL_H
