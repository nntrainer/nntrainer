// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   expert_reader_pool.cpp
 * @date   08 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Dedicated reader threads for expert records (see the header).
 */

#include <expert_reader_pool.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif

namespace nntrainer {

// ---------------------------------------------------------------------------
// AlignedBuffer
// ---------------------------------------------------------------------------

AlignedBuffer::AlignedBuffer(AlignedBuffer &&o) noexcept :
  ptr_(o.ptr_), cap_(o.cap_) {
  o.ptr_ = nullptr;
  o.cap_ = 0;
}

AlignedBuffer &AlignedBuffer::operator=(AlignedBuffer &&o) noexcept {
  if (this != &o) {
    release();
    ptr_ = o.ptr_;
    cap_ = o.cap_;
    o.ptr_ = nullptr;
    o.cap_ = 0;
  }
  return *this;
}

void AlignedBuffer::reserve(size_t bytes) {
  if (bytes <= cap_)
    return;
  release();
  const size_t cap = (bytes + kAlign - 1) / kAlign * kAlign;
#if defined(_WIN32)
  void *p = _aligned_malloc(cap, kAlign);
#else
  // Pages of their own, given back at release: a bounce is ~1.5 MiB, and from
  // the allocator such blocks come out of per-thread heaps that keep them
  // after free (measured: +184 MiB RssAnon after the load-time fills of 30
  // layers on 8 loader threads).
  void *p = ::mmap(nullptr, cap, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED)
    p = nullptr;
#endif
  if (p == nullptr)
    throw std::bad_alloc();
  ptr_ = static_cast<uint8_t *>(p);
  cap_ = cap;
}

void AlignedBuffer::release() {
  if (ptr_ == nullptr)
    return;
#if defined(_WIN32)
  _aligned_free(ptr_);
#else
  ::munmap(ptr_, cap_);
#endif
  ptr_ = nullptr;
  cap_ = 0;
}

namespace {

/**
 * @brief Positional read of exactly @a len bytes unless the file ends first.
 * @return 0, an errno, or -1 when the file ended before @a min_len bytes
 */
int preadSome(int fd, uint8_t *dst, size_t len, uint64_t off, size_t min_len) {
  size_t got = 0;
  while (got < len) {
#if defined(_WIN32)
    HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE)
      return EBADF;
    OVERLAPPED ov{};
    const uint64_t at = off + got;
    ov.Offset = static_cast<DWORD>(at & 0xFFFFFFFFu);
    ov.OffsetHigh = static_cast<DWORD>(at >> 32);
    const size_t want = std::min<size_t>(len - got, 1u << 30);
    DWORD n = 0;
    if (!ReadFile(h, dst + got, static_cast<DWORD>(want), &n, &ov)) {
      const DWORD e = GetLastError();
      if (e == ERROR_HANDLE_EOF)
        break;
      return EIO;
    }
#else
    const ssize_t n =
      ::pread(fd, dst + got, len - got, static_cast<off_t>(off + got));
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return errno != 0 ? errno : EIO;
    }
#endif
    if (n == 0)
      break;
    got += static_cast<size_t>(n);
  }
  return got >= min_len ? 0 : -1;
}

/** this thread's core, or -1 where that is not known */
int currentCpu() {
#if defined(__linux__)
  return sched_getcpu();
#else
  return -1;
#endif
}

/** keep this thread off @a avoid (when the affinity calls exist) */
void pinOffCpu(int avoid) {
#if defined(__linux__)
  if (avoid < 0)
    return;
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0)
    return;
  if (!CPU_ISSET(avoid, &set))
    return;
  CPU_CLR(avoid, &set);
  if (CPU_COUNT(&set) == 0)
    return; // the caller's core is the only one we may use
  (void)sched_setaffinity(0, sizeof(set), &set);
#else
  (void)avoid;
#endif
}

} // namespace

// ---------------------------------------------------------------------------
// ExpertReaderPool
// ---------------------------------------------------------------------------

ExpertReaderPool::ExpertReaderPool(unsigned int readers, bool direct) :
#if defined(__linux__) && defined(O_DIRECT)
  direct_(direct)
#else
  direct_(false)
#endif
{
  (void)direct;
  caller_cpu_ = currentCpu();
  ensureReaders(std::max(1u, readers));
}

void ExpertReaderPool::ensureReaders(unsigned int n) {
  std::lock_guard<std::mutex> lk(mu_);
  if (stop_)
    return;
  n = std::min(16u, n);
  const int caller = caller_cpu_;
  while (threads_.size() < n)
    threads_.emplace_back([this, caller] {
      pinOffCpu(caller);
      loop();
    });
  nthreads_.store(static_cast<unsigned int>(threads_.size()),
                  std::memory_order_release);
}

ExpertReaderPool::~ExpertReaderPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  work_cv_.notify_all();
  for (auto &t : threads_)
    if (t.joinable())
      t.join();
#if !defined(_WIN32)
  for (auto &kv : direct_fds_)
    if (kv.second.dfd >= 0)
      ::close(kv.second.dfd);
#endif
}

ExpertReaderPool &ExpertReaderPool::global() {
  static ExpertReaderPool pool(
    []() {
  // Android: two readers reach the UFS ceiling (3.4 GB/s measured on
  // an SM8850 phone with O_DIRECT 1.43 MiB records); more only add
  // threads. Elsewhere four (NVMe keeps scaling to four streams).
#if defined(__ANDROID__)
      constexpr long kDefaultReaders = 2;
#else
      constexpr long kDefaultReaders = 4;
#endif
      const char *e = std::getenv("NNTR_MOE_READERS");
      const long v =
        e != nullptr ? std::strtol(e, nullptr, 10) : kDefaultReaders;
      return static_cast<unsigned int>(std::min(16L, std::max(1L, v)));
    }(),
    []() {
      const char *e = std::getenv("NNTR_MOE_ODIRECT");
      return e == nullptr || e[0] != '0';
    }());
  return pool;
}

int ExpertReaderPool::directFd(int fd) {
#if defined(__linux__) && defined(O_DIRECT)
  struct stat st;
  if (::fstat(fd, &st) != 0)
    return -1; // not an open descriptor: the buffered read reports it
  const uint64_t dev = static_cast<uint64_t>(st.st_dev);
  const uint64_t ino = static_cast<uint64_t>(st.st_ino);
  std::lock_guard<std::mutex> lk(fd_mu_);
  auto it = direct_fds_.find(fd);
  if (it != direct_fds_.end()) {
    if (it->second.dev == dev && it->second.ino == ino)
      return it->second.dfd;
    // the caller closed fd and the number now names another file
    if (it->second.dfd >= 0)
      ::close(it->second.dfd);
    direct_fds_.erase(it);
  }
  // a second open file description on the same file, with O_DIRECT; the
  // caller's descriptor (and its flags) is left as it is
  const std::string path = "/proc/self/fd/" + std::to_string(fd);
  DirectFd d;
  d.dfd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  d.dev = dev;
  d.ino = ino;
  if (d.dfd >= 0) {
    // the path is resolved again by open(): keep the descriptor only when it
    // is the same file as fd
    struct stat dst;
    if (::fstat(d.dfd, &dst) != 0 || static_cast<uint64_t>(dst.st_dev) != dev ||
        static_cast<uint64_t>(dst.st_ino) != ino) {
      ::close(d.dfd);
      d.dfd = -1;
    }
  }
  direct_fds_.emplace(fd, d);
  return d.dfd;
#else
  (void)fd;
  return -1;
#endif
}

void ExpertReaderPool::forgetFd(int fd) {
#if !defined(_WIN32)
  std::lock_guard<std::mutex> lk(fd_mu_);
  auto it = direct_fds_.find(fd);
  if (it == direct_fds_.end())
    return;
  if (it->second.dfd >= 0)
    ::close(it->second.dfd);
  direct_fds_.erase(it);
#else
  (void)fd;
#endif
}

void ExpertReaderPool::readNow(ExpertRead &r) {
  if (r.once != nullptr && r.once->exchange(true)) {
    // another party took (or dropped) this read before it started
    r.error = ECANCELED;
    if (r.finished)
      r.finished(r);
    return;
  }
  // a read with a consumer and no destination stages in this thread's buffer
  thread_local AlignedBuffer own;
  AlignedBuffer *const given = r.bounce;
  if (r.dst == nullptr && given == nullptr && r.consume && r.fd >= 0) {
    try {
      own.reserve(bounceBytes(r.bytes));
    } catch (...) {
      r.error = ENOMEM;
      return;
    }
    r.bounce = &own;
  }
  readInto(r);
  if (r.error == 0 && r.consume) {
    try {
      r.error = r.consume(
        (r.dst != nullptr ? r.dst : r.bounce->data()) + r.data_offset, r);
    } catch (...) {
      r.error = EIO;
    }
  }
  r.bounce = given;
  if (r.finished) {
    try {
      r.finished(r);
    } catch (...) {
    }
  }
}

void ExpertReaderPool::readInto(ExpertRead &r) {
  r.error = 0;
  r.direct = false;
  const uint64_t a = AlignedBuffer::kAlign;
  uint8_t *const base = r.dst != nullptr      ? r.dst
                        : r.bounce != nullptr ? r.bounce->data()
                                              : nullptr;
  if (base == nullptr || r.fd < 0 ||
      (r.dst == nullptr && r.bounce->capacity() < bounceBytes(r.bytes)) ||
      reinterpret_cast<uintptr_t>(base) % a != 0) {
    r.error = EINVAL;
    return;
  }
  const uint64_t lo = r.offset / a * a;
  r.data_offset = static_cast<size_t>(r.offset - lo);
  if (direct_) {
    const int dfd = directFd(r.fd);
    if (dfd >= 0) {
      const uint64_t hi = (r.offset + r.bytes + a - 1) / a * a;
      const int rc = preadSome(dfd, base, static_cast<size_t>(hi - lo), lo,
                               r.data_offset + r.bytes);
      if (rc == 0) {
        r.direct = true;
        return;
      }
      // EFAULT: the destination is memory O_DIRECT cannot pin (a caller's
      // dst in device-shared memory the kernel cannot map, e.g. coarse-grain
      // SVM on Intel NEO): the buffered read below copies into it instead
      if (rc != EINVAL && !(rc == EFAULT && r.dst != nullptr)) {
        r.error = rc;
        return;
      }
      // EINVAL: this file (system) refuses the aligned direct read; fall
      // through to the buffered read
    }
  }
  r.error = preadSome(r.fd, base + r.data_offset, r.bytes, r.offset, r.bytes);
}

void ExpertReaderPool::retire(
  std::pair<Ticket, size_t> job,
  std::shared_ptr<std::vector<ExpertRead>> &release) {
  auto it = batches_.find(job.first);
  if (it == batches_.end())
    return;
  ++it->second.done;
  if (it->second.owned && it->second.done == it->second.owned->size()) {
    release = std::move(it->second.owned);
    batches_.erase(it);
  }
}

void ExpertReaderPool::loop() {
  for (;;) {
    std::pair<Ticket, size_t> job;
    ExpertRead *r = nullptr;
    std::function<void()> task;
    bool drop = false; // a low-priority read left at shutdown: not read
    {
      std::unique_lock<std::mutex> lk(mu_);
      work_cv_.wait(
        lk, [this] { return stop_ || !jobs_.empty() || !low_.empty(); });
      if (stop_ && jobs_.empty() && low_.empty())
        return;
      // a low-priority read only when nothing else is queued
      auto &from = jobs_.empty() ? low_ : jobs_;
      drop = stop_ && &from == &low_;
      job = from.front();
      from.pop_front();
      auto it = batches_.find(job.first);
      if (it->second.task) {
        // a posted task is detached: nobody waits on its ticket
        task = std::move(it->second.task);
        batches_.erase(it);
      } else {
        r = &(*it->second.reads)[job.second];
      }
    }
    if (task) {
      try {
        task();
      } catch (...) {
      }
      continue;
    }
    if (drop) {
      r->error = ECANCELED;
      if (r->finished) {
        try {
          r->finished(*r);
        } catch (...) {
        }
      }
    } else {
      readNow(*r);
    }
    std::shared_ptr<std::vector<ExpertRead>> release;
    {
      std::lock_guard<std::mutex> lk(mu_);
      retire(job, release);
    }
    done_cv_.notify_all();
  }
}

void ExpertReaderPool::submitLow(
  std::shared_ptr<std::vector<ExpertRead>> batch) {
  submitDetached(std::move(batch), /*low=*/true);
}

void ExpertReaderPool::submitDetached(
  std::shared_ptr<std::vector<ExpertRead>> batch, bool low) {
  if (!batch || batch->empty())
    return;
  std::lock_guard<std::mutex> lk(mu_);
  const Ticket t = next_++;
  Batch b;
  b.reads = batch.get();
  b.owned = std::move(batch);
  const size_t n = b.reads->size();
  batches_.emplace(t, std::move(b));
  auto &q = low ? low_ : jobs_;
  for (size_t i = 0; i < n; ++i)
    q.emplace_back(t, i);
  work_cv_.notify_all();
}

size_t ExpertReaderPool::batchesInFlight() {
  std::lock_guard<std::mutex> lk(mu_);
  return batches_.size();
}

size_t ExpertReaderPool::lowQueued() {
  std::lock_guard<std::mutex> lk(mu_);
  return low_.size();
}

ExpertReaderPool::Ticket
ExpertReaderPool::submit(std::vector<ExpertRead> *batch) {
  std::lock_guard<std::mutex> lk(mu_);
  const Ticket t = next_++;
  Batch b;
  b.reads = batch;
  batches_.emplace(t, b);
  const size_t n = batch != nullptr ? batch->size() : 0;
  for (size_t i = 0; i < n; ++i)
    jobs_.emplace_back(t, i);
  if (n != 0)
    work_cv_.notify_all();
  return t;
}

bool ExpertReaderPool::done(Ticket t) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = batches_.find(t);
  if (it == batches_.end())
    return true;
  const size_t n = it->second.reads != nullptr ? it->second.reads->size() : 0;
  return it->second.done == n;
}

void ExpertReaderPool::post(std::function<void()> task) {
  if (!task)
    return;
  std::lock_guard<std::mutex> lk(mu_);
  const Ticket t = next_++;
  Batch b;
  b.task = std::move(task);
  batches_.emplace(t, std::move(b));
  jobs_.emplace_back(t, 0);
  work_cv_.notify_all();
}

void ExpertReaderPool::wait(Ticket t, bool help) {
  std::vector<ExpertRead> *reads = nullptr;
  // detached batches whose last read this waiter ran: dropped unlocked
  std::vector<std::shared_ptr<std::vector<ExpertRead>>> retired;
  {
    std::unique_lock<std::mutex> lk(mu_);
    auto it = batches_.find(t);
    if (it == batches_.end())
      return;
    const size_t n = it->second.reads != nullptr ? it->second.reads->size() : 0;
    for (;;) {
      if (batches_.at(t).done == n)
        break;
      // help: run a queued read on this thread -- this batch's first, else
      // the oldest read of any batch. Posted tasks are skipped, not run
      // here: a waiting task would otherwise sit behind a task queued ahead
      // of its reads, which on a pool whose readers all run tasks nobody
      // takes (one reader, two posted tasks: the second task stays queued
      // ahead of the first task's reads). Skipping them, every helping
      // waiter serves its own reads, so N readers make progress with any
      // number of posted tasks; a skipped task runs when a reader is free.
      // Low-priority reads (submitLow) are in low_, never taken here: a
      // waiter serves the reads somebody waits for, not reads ahead.
      if (help) {
        auto pick = jobs_.end();
        for (auto j = jobs_.begin(); j != jobs_.end(); ++j) {
          if (j->first == t) {
            pick = j;
            break;
          }
          if (pick == jobs_.end()) {
            auto bt = batches_.find(j->first);
            if (bt != batches_.end() && bt->second.reads != nullptr)
              pick = j;
          }
        }
        if (pick != jobs_.end()) {
          const auto job = *pick;
          jobs_.erase(pick);
          ExpertRead *r = &(*batches_.at(job.first).reads)[job.second];
          lk.unlock();
          readNow(*r);
          lk.lock();
          // a detached batch (submitDetached) is retired at its last read
          std::shared_ptr<std::vector<ExpertRead>> release;
          retire(job, release);
          if (release)
            retired.push_back(std::move(release));
          done_cv_.notify_all();
          continue;
        }
      }
      done_cv_.wait(lk);
    }
    reads = batches_.at(t).reads;
    batches_.erase(t);
  }
  if (reads == nullptr)
    return;
  for (const ExpertRead &r : *reads)
    if (r.error != 0)
      throw std::runtime_error(
        "ExpertReaderPool: reading " + std::to_string(r.bytes) +
        " bytes at offset " + std::to_string(r.offset) + " of fd " +
        std::to_string(r.fd) + " failed (" +
        (r.error < 0 ? std::string("end of file")
                     : std::string(std::strerror(r.error))) +
        ")");
}

} // namespace nntrainer
