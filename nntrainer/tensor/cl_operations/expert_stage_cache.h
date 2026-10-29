// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   expert_stage_cache.h
 * @date   09 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Expert records read ahead of need into host memory: the
 *         prediction of a later decode step, served to that step's misses.
 *
 * @details A streamed MoE layer's decode step reads the experts it misses
 * from the weight file into device slots, on its critical path. When a
 * prediction of those experts exists earlier (the next layer's router run on
 * this layer's router row, "pre-gating"), the records can be read ahead into
 * host memory and copied into the slots when the step needs them. This class
 * holds those records. It is a hint only:
 *
 *  - It never touches the residency policy or the device slots: a staged
 *    record is not resident, is not "used", refreshes no recency and evicts
 *    nothing. The step's own acquire() decides its misses exactly as without
 *    it; only how a miss's bytes arrive changes (host copy instead of a file
 *    read), so outputs are bit for bit the same.
 *  - Its reads are low-priority detached reads of the pool
 *    (ExpertReaderPool::submitLow): a reader takes one only when no demand
 *    read or posted task is queued, and nobody ever waits on the pool for
 *    them. At most `window` reads per layer are submitted at once; each
 *    finished read submits the layer's next (from the reader thread, without
 *    waiting).
 *  - claim() is what the step does for a miss: a ready record is served; a
 *    record whose read nobody started is taken over (the queued read is
 *    cancelled through its once flag and the step reads the record itself,
 *    as a demand read); a record being read is waited for (a read that is
 *    running, never a task); anything else is a plain miss.
 *  - close(layer) ends the layer's step: what it did not claim is dropped
 *    (counted wasted when read or being read, cancelled when not started),
 *    and the layer's generation advances, so a prediction made for the step
 *    that just ended and staged late is refused (stage() with the old
 *    generation).
 *
 * Thread-safe (one mutex; the pool is called with it held, and calls back
 * only outside its own lock).
 */

#ifndef __EXPERT_STAGE_CACHE_H__
#define __EXPERT_STAGE_CACHE_H__

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <expert_reader_pool.h>

namespace nntrainer {

class ExpertStageCache {
public:
  /** where a layer's records are */
  struct Source {
    int fd = -1;
    const std::vector<uint64_t> *offset = nullptr; /**< expert -> offset */
    size_t record_bytes = 0;
  };

  /** state of an entry (under the cache mutex) */
  enum State : int {
    kPending = 0, /**< staged, its read not submitted yet (window full) */
    kQueued,      /**< submitted; queued or being read */
    kReady,       /**< read: buf + data_offset hold the record */
    kFailed,      /**< the read failed */
    kCancelled,   /**< never read (taken over, dropped or shut down) */
  };

  /** one record read ahead */
  struct Entry {
    unsigned int layer = 0, expert = 0;
    uint64_t file_offset = 0;
    AlignedBuffer buf;
    size_t data_offset = 0; /**< the record is buf.data() + data_offset */
    std::atomic<bool> once{false}; /**< the read's once flag */
    int state = kPending;
    bool claimed = false; /**< a step took it (it recycles the buffer) */
    bool in_map = true;   /**< still findable by claim() */
    const uint8_t *data() const { return buf.data() + data_offset; }
  };

  /** what claim() found */
  enum class Claim {
    none,      /**< not staged (or failed): read it */
    ready,     /**< served: copy entry->data() */
    in_flight, /**< being read: waitReady(), then copy (or read on false) */
    take_over, /**< its read had not started and is cancelled: read it */
  };
  struct Taken {
    Claim kind = Claim::none;
    std::shared_ptr<Entry> entry;
  };

  /** totals since construction (or resetStats) */
  struct Stats {
    std::atomic<unsigned long long> predictions{0}; /**< stage() calls */
    std::atomic<unsigned long long> predicted{0};   /**< experts offered */
    std::atomic<unsigned long long> staged{0};      /**< entries created */
    std::atomic<unsigned long long> reads{0};       /**< reads that did I/O */
    std::atomic<unsigned long long> served{0};      /**< ready at claim */
    std::atomic<unsigned long long> waited{0};      /**< served after a wait */
    std::atomic<unsigned long long> wait_us{0};     /**< time in waitReady */
    std::atomic<unsigned long long> taken_over{0};
    std::atomic<unsigned long long> wasted{0};    /**< read, never claimed */
    std::atomic<unsigned long long> cancelled{0}; /**< dropped unread */
    std::atomic<unsigned long long> late{0};      /**< refused predictions */
    std::atomic<unsigned long long> failed{0};
  };

  /**
   * @param pool the reader pool (outlives this)
   * @param window low-priority reads submitted at once per layer (>= 1)
   */
  ExpertStageCache(ExpertReaderPool &pool, unsigned int window) :
    pool_(pool), window_(window == 0 ? 1 : window) {}

  ~ExpertStageCache() {
    // reads still queued hold shared references to their entries; nothing
    // here points into them
    std::lock_guard<std::mutex> lk(mu_);
    map_.clear();
  }

  ExpertStageCache(const ExpertStageCache &) = delete;
  ExpertStageCache &operator=(const ExpertStageCache &) = delete;

  unsigned int window() const { return window_; }

  /** @brief the generation of @a layer: close() calls so far */
  uint64_t generation(unsigned int layer) {
    std::lock_guard<std::mutex> lk(mu_);
    return gen_[layer];
  }

  /**
   * @brief Read ahead the records of @a experts (rank order, most likely
   *        first; the caller left out those resident) for @a layer's next
   *        step. Refused (0, counted late) when @a layer was closed since
   *        @a gen was taken: the step the prediction was for has run.
   * @return the entries created
   */
  size_t stage(unsigned int layer, uint64_t gen, const Source &src,
               const std::vector<unsigned int> &experts) {
    std::lock_guard<std::mutex> lk(mu_);
    stats_.predictions++;
    stats_.predicted += experts.size();
    if (gen_[layer] != gen) {
      stats_.late++;
      return 0;
    }
    size_t n = 0;
    for (unsigned int e : experts) {
      if (src.offset == nullptr || e >= src.offset->size() || src.fd < 0 ||
          map_.count(key(layer, e)) != 0)
        continue;
      auto ent = std::make_shared<Entry>();
      ent->layer = layer;
      ent->expert = e;
      ent->file_offset = (*src.offset)[e];
      map_.emplace(key(layer, e), ent);
      Layer &l = layers_[layer];
      l.pending.push_back(ent);
      l.src = src;
      ++n;
    }
    stats_.staged += n;
    issueLocked(layer);
    return n;
  }

  /**
   * @brief The step of @a layer misses @a expert: see Claim. Removes the
   *        entry from the cache (a second claim finds nothing).
   */
  Taken claim(unsigned int layer, unsigned int expert) {
    std::lock_guard<std::mutex> lk(mu_);
    Taken t;
    auto it = map_.find(key(layer, expert));
    if (it == map_.end())
      return t;
    std::shared_ptr<Entry> e = std::move(it->second);
    map_.erase(it);
    e->in_map = false;
    e->claimed = true;
    switch (e->state) {
    case kReady:
      t.kind = Claim::ready;
      stats_.served++;
      break;
    case kPending:
      e->state = kCancelled; // never submitted: the issue loop skips it
      e->once.store(true);
      t.kind = Claim::take_over;
      stats_.taken_over++;
      break;
    case kQueued:
      if (!e->once.exchange(true)) {
        // nobody started it: the pool skips it, the step reads it
        t.kind = Claim::take_over;
        stats_.taken_over++;
      } else {
        t.kind = Claim::in_flight;
      }
      break;
    default:
      t.kind = Claim::none;
      break;
    }
    t.entry = std::move(e);
    return t;
  }

  /**
   * @brief Wait until the read of an in_flight claim ends (a read running on
   *        another thread: it always ends). true = the record is in the
   *        entry; false = the read failed (read it yourself).
   */
  bool waitReady(Entry &e) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&e] { return e.state != kQueued && e.state != kPending; });
    stats_.wait_us +=
      (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0)
        .count();
    if (e.state == kReady) {
      stats_.served++;
      stats_.waited++;
      return true;
    }
    return false;
  }

  /** @brief a claimed entry is done with: its buffer is reused */
  void release(const std::shared_ptr<Entry> &e) {
    if (!e)
      return;
    std::lock_guard<std::mutex> lk(mu_);
    if (e->state != kQueued)
      recycleLocked(*e);
  }

  /**
   * @brief @a layer's step is done: drop what it did not claim, advance the
   *        generation. Returns the entries dropped.
   */
  size_t close(unsigned int layer) {
    std::lock_guard<std::mutex> lk(mu_);
    ++gen_[layer];
    size_t n = 0;
    for (auto it = map_.lower_bound(key(layer, 0));
         it != map_.end() && (unsigned int)(it->first >> 32) == layer;) {
      Entry &e = *it->second;
      e.in_map = false;
      if (e.state == kReady) {
        stats_.wasted++;
        recycleLocked(e);
      } else if (e.state == kPending) {
        e.state = kCancelled;
        e.once.store(true);
        stats_.cancelled++;
      } else if (e.state == kQueued) {
        if (!e.once.exchange(true))
          stats_.cancelled++; // the pool skips it
        else
          stats_.wasted++; // being read; recycled when it ends
      }
      it = map_.erase(it);
      ++n;
    }
    auto lt = layers_.find(layer);
    if (lt != layers_.end())
      lt->second.pending.clear();
    return n;
  }

  /** @brief entries findable now (tests) */
  size_t size() {
    std::lock_guard<std::mutex> lk(mu_);
    return map_.size();
  }

  /** @brief reads of @a layer submitted and not finished (tests) */
  unsigned int inFlight(unsigned int layer) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = layers_.find(layer);
    return it == layers_.end() ? 0 : it->second.inflight;
  }

  Stats &stats() { return stats_; }

private:
  struct Layer {
    std::deque<std::shared_ptr<Entry>> pending;
    unsigned int inflight = 0;
    Source src;
  };

  static uint64_t key(unsigned int layer, unsigned int expert) {
    return ((uint64_t)layer << 32) | expert;
  }

  /** submit @a layer's pending reads up to the window. Caller holds mu_ */
  void issueLocked(unsigned int layer) {
    Layer &l = layers_[layer];
    std::shared_ptr<std::vector<ExpertRead>> batch;
    while (l.inflight < window_ && !l.pending.empty()) {
      std::shared_ptr<Entry> e = std::move(l.pending.front());
      l.pending.pop_front();
      if (e->state != kPending)
        continue; // taken over or dropped before it was submitted
      const size_t need = ExpertReaderPool::bounceBytes(l.src.record_bytes);
      if (!free_.empty()) {
        e->buf = std::move(free_.back());
        free_.pop_back();
      }
      try {
        e->buf.reserve(need);
      } catch (...) {
        e->state = kFailed;
        stats_.failed++;
        cv_.notify_all();
        continue;
      }
      e->state = kQueued;
      ++l.inflight;
      if (!batch)
        batch = std::make_shared<std::vector<ExpertRead>>();
      batch->emplace_back();
      ExpertRead &r = batch->back();
      r.fd = l.src.fd;
      r.offset = e->file_offset;
      r.bytes = l.src.record_bytes;
      r.bounce = &e->buf;
      r.once = &e->once;
      r.finished = [this, e](ExpertRead &rd) { finished(e, rd); };
    }
    if (batch)
      pool_.submitLow(std::move(batch));
  }

  /** a read of @a e ended (reader thread, outside the pool's lock) */
  void finished(const std::shared_ptr<Entry> &e, const ExpertRead &rd) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = layers_[e->layer];
    if (l.inflight > 0)
      --l.inflight;
    if (rd.error == 0) {
      e->data_offset = rd.data_offset;
      e->state = kReady;
      stats_.reads++;
    } else if (rd.error == ECANCELED) {
      e->state = kCancelled;
    } else {
      e->state = kFailed;
      stats_.failed++;
    }
    // nobody will copy from it: dropped while read, or taken over
    if (!e->in_map && !e->claimed)
      recycleLocked(*e);
    else if (e->claimed && e->state != kReady)
      recycleLocked(*e);
    cv_.notify_all();
    issueLocked(e->layer);
  }

  /** keep @a e's buffer for the next read (bounded). Caller holds mu_ */
  void recycleLocked(Entry &e) {
    if (e.buf.capacity() == 0)
      return;
    if (free_.size() < kMaxFree)
      free_.push_back(std::move(e.buf));
    else
      e.buf = AlignedBuffer();
  }

  static constexpr size_t kMaxFree = 32;

  ExpertReaderPool &pool_;
  const unsigned int window_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::map<uint64_t, std::shared_ptr<Entry>> map_;
  std::unordered_map<unsigned int, Layer> layers_;
  std::unordered_map<unsigned int, uint64_t> gen_;
  std::vector<AlignedBuffer> free_;
  Stats stats_;
};

} // namespace nntrainer

#endif // __EXPERT_STAGE_CACHE_H__
