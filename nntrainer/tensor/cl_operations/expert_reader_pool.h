// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   expert_reader_pool.h
 * @date   08 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Dedicated reader threads that stream expert records from the model
 *         file into page-aligned host bounce buffers.
 *
 * @details The OpenCL backend fills its expert slots from the weight file at
 * run time (a MoE layer whose experts do not all stay resident). A fill is
 *
 *   pread(record) -> bounce buffer -> clEnqueueWriteBuffer(slot)
 *
 * and this pool is the first arrow. It owns R reader threads of its own
 * (default 4): they are not ThreadManager workers, because a read has to keep
 * going while the caller enqueues kernels or while ThreadManager's workers
 * run a host op. Where the platform exposes thread affinity the readers are
 * kept off the core of the thread that created the pool.
 *
 * Requests are submitted in batches and complete in FIFO order of
 * submission; each batch gets a ticket the caller waits on. A read goes to a
 * caller-owned AlignedBuffer (4 KiB aligned). With O_DIRECT (Linux / Android,
 * default on) the pool reads the 4 KiB-aligned superset of the record through
 * a second descriptor opened on the same file and reports where the record
 * starts inside the bounce (data_offset). When O_DIRECT is not available
 * (other platforms, a file system that refuses it, NNTR_MOE_ODIRECT=0) the
 * read is a buffered positional read of the record itself, placed at the
 * same data_offset, so callers see one layout either way. A read may also go
 * straight to a caller's own 4 KiB-aligned memory (ExpertRead::dst, e.g. an
 * expert slot in fine-grain SVM): the same layout, no bounce and no copy.
 * Batches the caller does not wait for are submitted detached
 * (submitDetached / submitLow) and report through ExpertRead::finished.
 *
 * A short read or an I/O error never yields a partial record: the request
 * carries the error, and wait() throws for the batch.
 */

#ifndef __EXPERT_READER_POOL_H__
#define __EXPERT_READER_POOL_H__

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nntrainer {

/**
 * @brief A host buffer whose start is 4 KiB aligned (O_DIRECT destination).
 */
class AlignedBuffer {
public:
  static constexpr size_t kAlign = 4096; /**< alignment of data() */

  AlignedBuffer() = default;
  /** @brief @a bytes of capacity, rounded up to the alignment */
  explicit AlignedBuffer(size_t bytes) { reserve(bytes); }
  ~AlignedBuffer() { release(); }
  AlignedBuffer(const AlignedBuffer &) = delete;
  AlignedBuffer &operator=(const AlignedBuffer &) = delete;
  AlignedBuffer(AlignedBuffer &&o) noexcept;
  AlignedBuffer &operator=(AlignedBuffer &&o) noexcept;

  /** @brief grow to at least @a bytes (contents are not kept) */
  void reserve(size_t bytes);
  uint8_t *data() { return ptr_; }
  const uint8_t *data() const { return ptr_; }
  size_t capacity() const { return cap_; }

private:
  void release();
  uint8_t *ptr_ = nullptr;
  size_t cap_ = 0;
};

/**
 * @brief One record to read.
 */
struct ExpertRead {
  int fd = -1;                     /**< file to read (kept open by caller) */
  uint64_t offset = 0;             /**< file offset of the record */
  size_t bytes = 0;                /**< record bytes */
  AlignedBuffer *bounce = nullptr; /**< destination, capacity at least
                                        ExpertReaderPool::bounceBytes(bytes);
                                        null with consume: the reader's own */
  /** optional, in place of a bounce: the caller's own destination (e.g. an
   *  expert slot the device reads), 4 KiB aligned (else the read fails with
   *  EINVAL) and at least bounceBytes(bytes) long. The aligned superset is
   *  read straight into it (O_DIRECT), or buffered the record alone at
   *  dst + data_offset: either way the record ends at dst + offset % 4096,
   *  with no copy. consume, when set, gets that address */
  uint8_t *dst = nullptr;
  /** optional: run on the reader thread once the record is read, with the
   *  record's bytes (valid only during the call); a non-zero return (an
   *  errno) or an exception fails the read. With a null bounce the record
   *  is read into the reader thread's own staging buffer, which is reused
   *  as soon as consume returns */
  std::function<int(const uint8_t *record, const ExpertRead &r)> consume;
  /** optional: a read that more than one party may take (a read ahead that
   *  the step needing it may read itself instead, or drop). The thread
   *  about to read exchanges it to true; when it was true already the read
   *  is skipped (no I/O, no consume) and ends with error ECANCELED.
   *  Exchanging it to true first therefore cancels a read nobody started
   *  (and tells the party that it may read the record itself); a read that
   *  started completes. Owned by the caller, alive until the read is done */
  std::atomic<bool> *once = nullptr;
  /** optional: run on the thread that ran the read, after the read and its
   *  consume, whatever the outcome (error set, ECANCELED for a skipped
   *  once read). A detached read reports its end only through this */
  std::function<void(ExpertRead &r)> finished;
  // ---- filled by the pool ----
  size_t data_offset = 0; /**< the record starts at bounce->data() + this */
  int error = 0;          /**< 0, an errno, or -1 for an end of file */
  bool direct = false;    /**< read through O_DIRECT */
};

/**
 * @brief R dedicated reader threads, FIFO batches, completion tickets.
 */
class ExpertReaderPool {
public:
  using Ticket = uint64_t;

  /**
   * @brief Start @a readers threads (at least one).
   * @param direct try O_DIRECT reads (ignored where unavailable)
   */
  explicit ExpertReaderPool(unsigned int readers = 4, bool direct = true);
  ~ExpertReaderPool();
  ExpertReaderPool(const ExpertReaderPool &) = delete;
  ExpertReaderPool &operator=(const ExpertReaderPool &) = delete;

  /**
   * @brief The pool of the process: NNTR_MOE_READERS threads (default 4; 2
   *        on Android, which reach the UFS ceiling; the streamed decode
   *        without the host round trip, NNTR_G4MOE_FSU_FLAGS, grows it to 3
   *        there when NNTR_MOE_READERS is unset), O_DIRECT unless
   *        NNTR_MOE_ODIRECT=0. Created on first use.
   */
  static ExpertReaderPool &global();

  /** @brief bounce capacity that holds the aligned superset of @a bytes */
  static size_t bounceBytes(size_t bytes) {
    return (bytes + 2 * AlignedBuffer::kAlign - 1) / AlignedBuffer::kAlign *
           AlignedBuffer::kAlign;
  }

  /**
   * @brief Queue every read of @a batch (taken in order by the readers).
   * @param batch kept alive and unchanged by the caller until wait() returns
   * @return the ticket to wait on; an empty batch is complete at once
   */
  Ticket submit(std::vector<ExpertRead> *batch);

  /**
   * @brief Queue the reads of @a batch at low priority and detached: a
   *        reader takes one only when no other read or posted task is
   *        queued, a helping waiter never takes one, and nobody waits on
   *        them (no ticket: each read reports its end through
   *        ExpertRead::finished; the pool keeps @a batch alive until its
   *        last read is done). For reads ahead of need (a prefetch): they
   *        never delay a read a step waits for that is queued, and they never
   *        make a reader thread wait. A read already running when a demand
   *        read is queued still completes first.
   */
  void submitLow(std::shared_ptr<std::vector<ExpertRead>> batch);

  /**
   * @brief Queue the reads of @a batch detached, like submitLow, at normal
   *        priority (@a low false: in FIFO order with the other reads and
   *        posted tasks; a helping waiter may run one) or at low priority
   *        (@a low true: submitLow). Nobody waits on them: each read reports
   *        its end through ExpertRead::finished; the pool keeps @a batch
   *        alive until its last read is done.
   */
  void submitDetached(std::shared_ptr<std::vector<ExpertRead>> batch, bool low);

  /** @brief low-priority reads queued and not started (tests, stats) */
  size_t lowQueued();

  /** @brief batches not retired yet (tests: a detached batch is retired at
   *         its last read, whoever ran it) */
  size_t batchesInFlight();

  /**
   * @brief Block until every read of ticket @a t is done.
   * @param help run queued reads on the calling thread while waiting (the
   *        batch's own first, then those of other batches; never a posted
   *        task, never a low-priority read): a task on a reader thread then
   *        reads alongside the other
   *        readers instead of idling, and never waits on reads that only a
   *        reader busy with another task could serve -- N readers make
   *        progress with any number of posted tasks that wait with help
   * @throw std::runtime_error when a read of the batch failed (the message
   *        names the first failure); the batch is still retired
   */
  void wait(Ticket t, bool help = false);

  /** @brief true once every read of @a t is done (or @a t was waited) */
  bool done(Ticket t);

  /**
   * @brief Queue @a task to run once on a reader thread, in FIFO order with
   *        the reads. Detached: there is no ticket, the task reports its own
   *        completion and errors (an exception it throws is dropped). A task
   *        that reads through the pool submits them and waits with help.
   */
  void post(std::function<void()> task);

  unsigned int readers() const {
    return nthreads_.load(std::memory_order_acquire);
  }

  /**
   * @brief Grow the pool to @a n reader threads (at most 16); a pool that
   *        has as many or more keeps its count. The new readers take work
   *        from the same queues as the others. Any thread.
   */
  void ensureReaders(unsigned int n);
  bool directEnabled() const { return direct_; }

  /** @brief one read on the calling thread (what a reader runs), then its
   *         consume callback when it has one */
  void readNow(ExpertRead &r);

  /**
   * @brief Drop the O_DIRECT descriptor the pool opened for @a fd (call when
   *        the caller closes @a fd). Not required for correctness: a read
   *        checks that @a fd still names the file the cached descriptor was
   *        opened on and reopens it otherwise.
   */
  void forgetFd(int fd);

private:
  struct Batch {
    std::vector<ExpertRead> *reads = nullptr;
    std::function<void()> task; /**< a posted task instead of reads */
    size_t done = 0;
    /** submitLow: the pool's reference; retired at the last read */
    std::shared_ptr<std::vector<ExpertRead>> owned;
  };

  void loop();
  void readInto(ExpertRead &r);
  /** the read of (batch, index) is done: count it, retire a finished
   *  low-priority batch (its reference moves to @a release, dropped by the
   *  caller outside the lock). Caller holds mu_ */
  void retire(std::pair<Ticket, size_t> job,
              std::shared_ptr<std::vector<ExpertRead>> &release);
  int directFd(int fd);

  bool direct_;
  std::mutex mu_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::deque<std::pair<Ticket, size_t>> jobs_; /**< (batch, read index) */
  std::deque<std::pair<Ticket, size_t>> low_;  /**< submitLow reads */
  std::map<Ticket, Batch> batches_;
  Ticket next_ = 1;
  bool stop_ = false;
  std::vector<std::thread> threads_; /**< grown under mu_ */
  std::atomic<unsigned int> nthreads_{0};
  int caller_cpu_ = -1; /**< the readers keep off the creator's core */

  /** the O_DIRECT descriptor opened for a caller's fd, and the file (device,
   *  inode) that fd named then: a descriptor number is reused once the
   *  caller closes it, so the entry is valid only while fd still names the
   *  same file */
  struct DirectFd {
    int dfd = -1; /**< O_DIRECT descriptor, or -1 (O_DIRECT refused) */
    uint64_t dev = 0, ino = 0;
  };
  std::mutex fd_mu_;
  std::unordered_map<int, DirectFd> direct_fds_; /**< caller fd -> entry */
};

} // namespace nntrainer

#endif // __EXPERT_READER_POOL_H__
