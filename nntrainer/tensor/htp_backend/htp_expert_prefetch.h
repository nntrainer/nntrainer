// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_expert_prefetch.h
 * @brief Bounded speculative reads into reusable expert buffers.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#ifndef NNTR_HTP_EXPERT_PREFETCH_H
#define NNTR_HTP_EXPERT_PREFETCH_H

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace nntrainer {

/** Workers own a buffer until its read completes, even after cancellation.
 * Submission and consumption serialize metadata. A caller may exchange a ready
 * buffer with a retired demand buffer; no payload copy or DSP operation is
 * needed. Readers may stop between I/O operations when their cancellation
 * check is true; failed or canceled guesses fall back to demand I/O.
 */
template <typename Buffer> class HtpExpertPrefetch {
public:
  struct Stats {
    uint64_t submitted = 0, completed = 0, used = 0, late = 0;
    uint64_t discarded = 0, failed = 0, full = 0, joined = 0;
    uint64_t cancelled = 0;  /**< Aborted callbacks, excluded from completed. */
    uint64_t handed_off = 0; /**< Queued guesses replaced by demand reads. */
  };
  enum class ReadResult { Complete, Cancelled, Failed };
  using Reader = std::function<bool(Buffer &)>;
  /** Valid only during its reader callback; do not retain or call it later. */
  using CancelCheck = std::function<bool()>;
  using CancellableReader =
    std::function<ReadResult(Buffer &, const CancelCheck &)>;

  /** Retain the original reader API for work that cannot stop partway. */
  HtpExpertPrefetch(std::vector<Buffer> buffers, size_t readers, Reader read) :
    HtpExpertPrefetch(
      std::move(buffers), readers,
      [read = std::move(read)](Buffer &buffer, const CancelCheck &) {
        return read(buffer) ? ReadResult::Complete : ReadResult::Failed;
      }) {}

  HtpExpertPrefetch(std::vector<Buffer> buffers, size_t readers,
                    CancellableReader read) :
    read_(std::move(read)) {
    if (buffers.empty() || readers == 0)
      throw std::invalid_argument("Expert prefetch needs buffers and readers");
    for (auto &buffer : buffers)
      slots_.push_back({std::move(buffer)});
    try {
      for (size_t i = 0; i < std::min(readers, slots_.size()); ++i)
        workers_.emplace_back([this] { work(); });
    } catch (...) {
      stop();
      throw;
    }
  }
  ~HtpExpertPrefetch() { stop(); }
  HtpExpertPrefetch(const HtpExpertPrefetch &) = delete;
  HtpExpertPrefetch &operator=(const HtpExpertPrefetch &) = delete;

  /** A bounded submission. initialize changes only a free buffer's metadata. */
  template <typename Initialize>
  bool submit(const void *key, uint64_t epoch, Initialize initialize) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_ || epoch < min_epoch_)
      return false;
    for (const auto &s : slots_)
      if (s.state != Free && !s.retired && s.key == key)
        return false;
    for (size_t i = 0; i < slots_.size(); ++i) {
      auto &s = slots_[i];
      if (s.state != Free)
        continue;
      initialize(s.buffer);
      queue_.push_back(i);
      s.key = key;
      s.epoch = epoch;
      s.retired = false;
      s.claimed = false;
      s.state = Queued;
      ++stats_.submitted;
      ready_.notify_one();
      return true;
    }
    ++stats_.full;
    return false;
  }

  /** Never waits for a guess. consume may swap in a retired demand buffer. */
  template <typename Consume> bool take(const void *key, Consume consume) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &s : slots_) {
      if (s.state == Free || s.retired || s.claimed || s.key != key)
        continue;
      if (s.state != Ready) {
        ++stats_.late;
        return false;
      }
      consume(s.buffer);
      s.state = Free;
      ++stats_.used;
      return true;
    }
    return false;
  }

  /** Reserve a real miss's speculative read. Queued demand goes first; an
   * in-flight read is joined after unrelated demand I/O instead of duplicated.
   * The sole consumer must finish claims with takeClaimed or drain.
   * With queued_to_demand, an unread queued guess is removed and false is
   * returned: the caller reads its own demand buffer. A worker that already
   * owns the slot is always claimed normally. No buffer crosses this API.
   * The removed key stays reserved until expiry/drain to reject late duplicate
   * submissions while the caller finishes claiming its current miss batch. */
  bool claim(const void *key, bool queued_to_demand = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < slots_.size(); ++i) {
      auto &s = slots_[i];
      if (s.state == Free || s.state == HandedOff || s.retired || s.claimed ||
          s.key != key)
        continue;
      if (queued_to_demand && s.state == Queued) {
        queue_.erase(std::find(queue_.begin(), queue_.end(), i));
        s.state = HandedOff;
        ++stats_.late;
        ++stats_.handed_off;
        return false;
      }
      s.claimed = true;
      if (s.state != Ready) {
        ++stats_.late;
        ++stats_.joined;
      }
      if (s.state == Queued) {
        queue_.erase(std::find(queue_.begin(), queue_.end(), i));
        queue_.push_front(i);
      }
      return true;
    }
    return false;
  }

  template <typename Consume>
  bool takeClaimed(const void *key, Consume consume) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto &s : slots_) {
      if (!s.claimed || s.key != key)
        continue;
      done_.wait(lock,
                 [&s] { return s.state != Reading && s.state != Queued; });
      const bool ok = s.state == Ready && !s.retired;
      if (ok) {
        consume(s.buffer);
        ++stats_.used;
      }
      s.claimed = false;
      s.state = Free;
      return ok;
    }
    return false;
  }

  /** Cancel stale guesses without recycling buffers that workers still own. */
  void expireBefore(uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_epoch_ = std::max(min_epoch_, epoch);
    expire(epoch);
  }

  /** Join outstanding I/O before changing representation or unmapping arenas.
   */
  void drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    expire(UINT64_MAX, true);
    done_.wait(lock, [this] {
      return std::none_of(slots_.begin(), slots_.end(),
                          [](const Slot &s) { return s.state == Reading; });
    });
  }

  Stats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

private:
  enum State { Free, Queued, Reading, Ready, Failed, HandedOff };
  struct Slot {
    Buffer buffer;
    const void *key = nullptr;
    uint64_t epoch = 0;
    State state = Free;
    bool retired = false;
    bool claimed = false;
  };
  void expire(uint64_t epoch, bool cancel_claims = false) {
    for (auto &s : slots_)
      if (s.state != Free && !s.retired && s.epoch < epoch &&
          (!s.claimed || cancel_claims)) {
        ++stats_.discarded;
        s.retired = true;
        if (s.state != Reading)
          s.state = Free;
      }
    queue_.erase(
      std::remove_if(queue_.begin(), queue_.end(),
                     [this](size_t i) { return slots_[i].state != Queued; }),
      queue_.end());
  }
  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      expire(UINT64_MAX, true);
    }
    ready_.notify_all();
    for (auto &worker : workers_)
      if (worker.joinable())
        worker.join();
  }
  void work() {
    for (;;) {
      size_t i;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_)
          return;
        i = queue_.front();
        queue_.pop_front();
        slots_[i].state = Reading;
      }
      ReadResult result = ReadResult::Failed;
      try {
        const CancelCheck cancelled = [this, i] {
          std::lock_guard<std::mutex> lock(mutex_);
          // Expiry protects claimed demand, even when its epoch is old.
          // Reading slots cannot be reused until this callback returns.
          return stop_ || slots_[i].retired;
        };
        result = read_(slots_[i].buffer, cancelled);
      } catch (...) {
        // Speculation is optional; the synchronous path reports I/O errors.
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &s = slots_[i];
        if (result == ReadResult::Cancelled)
          ++stats_.cancelled;
        else
          ++stats_.completed;
        stats_.failed += result == ReadResult::Failed;
        const bool ok = result == ReadResult::Complete;
        s.state = s.retired ? Free : ok ? Ready : s.claimed ? Failed : Free;
      }
      done_.notify_all();
    }
  }
  CancellableReader read_;
  std::vector<Slot> slots_;
  std::deque<size_t> queue_;
  std::vector<std::thread> workers_;
  mutable std::mutex mutex_;
  std::condition_variable ready_, done_;
  uint64_t min_epoch_ = 0;
  bool stop_ = false;
  Stats stats_;
};
} // namespace nntrainer
#endif
