// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_expert_reader_pool.cpp
 * @date        08 October 2026
 * @brief       Unit test for the OpenCL backend's expert reader pool. Reads a
 *              temporary file only; needs no GPU.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include <gtest/gtest.h>

#include <expert_reader_pool.h>
#include <expert_stage_cache.h>

namespace {

/** byte i of the test file with contents @a seed */
uint8_t pattern(uint64_t i, uint64_t seed = 0) {
  return static_cast<uint8_t>(((i + seed * 7919u) * 2654435761u) >> 13);
}

/** a temporary file of @a bytes pattern bytes, removed at scope exit */
class TempFile {
public:
  explicit TempFile(size_t bytes, uint64_t seed = 0) {
#if defined(_WIN32)
    char name[L_tmpnam];
    path_ = std::tmpnam(name);
#else
    char tmpl[] = "/tmp/nntr_expert_reader_XXXXXX";
    const int fd = ::mkstemp(tmpl);
    if (fd < 0)
      throw std::runtime_error("mkstemp failed");
    ::close(fd);
    path_ = tmpl;
#endif
    FILE *f = std::fopen(path_.c_str(), "wb");
    if (f == nullptr)
      throw std::runtime_error("cannot create " + path_);
    std::vector<uint8_t> buf(bytes);
    for (size_t i = 0; i < bytes; ++i)
      buf[i] = pattern(i, seed);
    std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
#if defined(_WIN32)
    fd_ = ::_open(path_.c_str(), _O_RDONLY | _O_BINARY);
#else
    fd_ = ::open(path_.c_str(), O_RDONLY);
#endif
    if (fd_ < 0)
      throw std::runtime_error("cannot open " + path_);
  }
  ~TempFile() {
#if defined(_WIN32)
    ::_close(fd_);
#else
    ::close(fd_);
#endif
    std::remove(path_.c_str());
  }
  int fd() const { return fd_; }

private:
  std::string path_;
  int fd_ = -1;
};

/** true when @a r holds bytes [r.offset, r.offset + r.bytes) of the file
 *  with contents @a seed */
bool holdsRecord(const nntrainer::ExpertRead &r, uint64_t seed = 0) {
  const uint8_t *p = r.bounce->data() + r.data_offset;
  for (size_t i = 0; i < r.bytes; ++i)
    if (p[i] != pattern(r.offset + i, seed))
      return false;
  return true;
}

// odd sizes and offsets on purpose: the pack's records start at 2 mod 4
constexpr size_t kFile = 8u << 20;
constexpr size_t kRecord = 1503744;

} // namespace

TEST(ExpertReaderPool, AlignedBufferIsAligned) {
  nntrainer::AlignedBuffer b(12345);
  ASSERT_NE(b.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(b.data()) %
              nntrainer::AlignedBuffer::kAlign,
            0u);
  EXPECT_GE(b.capacity(), 12345u);
  EXPECT_EQ(b.capacity() % nntrainer::AlignedBuffer::kAlign, 0u);
}

TEST(ExpertReaderPool, BounceBytesCoversAlignedSuperset) {
  const size_t a = nntrainer::AlignedBuffer::kAlign;
  for (uint64_t off : {0ull, 2ull, 4094ull, 4096ull, 29703490ull})
    for (size_t n : {size_t(1), size_t(4096), kRecord}) {
      const uint64_t lo = off / a * a;
      const uint64_t hi = (off + n + a - 1) / a * a;
      EXPECT_GE(nntrainer::ExpertReaderPool::bounceBytes(n), hi - lo)
        << off << " " << n;
    }
}

TEST(ExpertReaderPool, ReadsRecordsAtUnalignedOffsets) {
  TempFile f(kFile);
  nntrainer::ExpertReaderPool pool(4, true);
  const uint64_t offs[] = {0, 2, 4096, 4098, 1000001, kFile - kRecord};
  std::vector<nntrainer::AlignedBuffer> bounce(6);
  std::vector<nntrainer::ExpertRead> batch(6);
  for (size_t i = 0; i < batch.size(); ++i) {
    bounce[i].reserve(nntrainer::ExpertReaderPool::bounceBytes(kRecord));
    batch[i].fd = f.fd();
    batch[i].offset = offs[i];
    batch[i].bytes = kRecord;
    batch[i].bounce = &bounce[i];
  }
  const auto t = pool.submit(&batch);
  ASSERT_NO_THROW(pool.wait(t));
  int direct = 0;
  for (const auto &r : batch) {
    EXPECT_EQ(r.error, 0);
    EXPECT_EQ(r.data_offset, r.offset % nntrainer::AlignedBuffer::kAlign);
    EXPECT_TRUE(holdsRecord(r)) << "offset " << r.offset;
    direct += r.direct ? 1 : 0;
  }
  // informational: a file system without O_DIRECT takes the buffered read
  RecordProperty("direct_reads", direct);
  std::cout << "[ INFO     ] " << direct << " of " << batch.size()
            << " reads went through O_DIRECT\n";
}

TEST(ExpertReaderPool, BufferedFallbackGivesSameLayout) {
  TempFile f(kFile);
  nntrainer::ExpertReaderPool pool(2, false);
  EXPECT_FALSE(pool.directEnabled());
  nntrainer::AlignedBuffer b(nntrainer::ExpertReaderPool::bounceBytes(777));
  std::vector<nntrainer::ExpertRead> batch(1);
  batch[0].fd = f.fd();
  batch[0].offset = 8190;
  batch[0].bytes = 777;
  batch[0].bounce = &b;
  pool.wait(pool.submit(&batch));
  EXPECT_FALSE(batch[0].direct);
  EXPECT_EQ(batch[0].data_offset, 8190u % nntrainer::AlignedBuffer::kAlign);
  EXPECT_TRUE(holdsRecord(batch[0]));
}

TEST(ExpertReaderPool, ManyBatchesCompleteWithTheirTickets) {
  TempFile f(kFile);
  nntrainer::ExpertReaderPool pool(3, true);
  constexpr size_t kBatches = 8, kPer = 5, kBytes = 70001;
  std::vector<std::vector<nntrainer::AlignedBuffer>> bounce(kBatches);
  std::vector<std::vector<nntrainer::ExpertRead>> batches(kBatches);
  std::vector<nntrainer::ExpertReaderPool::Ticket> tickets;
  for (size_t b = 0; b < kBatches; ++b) {
    bounce[b].resize(kPer);
    batches[b].resize(kPer);
    for (size_t i = 0; i < kPer; ++i) {
      bounce[b][i].reserve(nntrainer::ExpertReaderPool::bounceBytes(kBytes));
      auto &r = batches[b][i];
      r.fd = f.fd();
      r.offset = (b * kPer + i) * 98765 + 2;
      r.bytes = kBytes;
      r.bounce = &bounce[b][i];
    }
    tickets.push_back(pool.submit(&batches[b]));
  }
  for (size_t i = 1; i < tickets.size(); ++i)
    EXPECT_GT(tickets[i], tickets[i - 1]); // FIFO issue order
  // wait out of order: every ticket is still answered
  for (size_t b = kBatches; b-- > 0;) {
    ASSERT_NO_THROW(pool.wait(tickets[b]));
    EXPECT_TRUE(pool.done(tickets[b]));
    for (const auto &r : batches[b])
      EXPECT_TRUE(holdsRecord(r));
  }
}

TEST(ExpertReaderPool, EmptyBatchIsDone) {
  nntrainer::ExpertReaderPool pool(1, false);
  std::vector<nntrainer::ExpertRead> none;
  const auto t = pool.submit(&none);
  EXPECT_TRUE(pool.done(t));
  EXPECT_NO_THROW(pool.wait(t));
}

TEST(ExpertReaderPool, ReadPastEndThrows) {
  TempFile f(1 << 16);
  nntrainer::ExpertReaderPool pool(2, true);
  nntrainer::AlignedBuffer b(nntrainer::ExpertReaderPool::bounceBytes(4096));
  std::vector<nntrainer::ExpertRead> batch(1);
  batch[0].fd = f.fd();
  batch[0].offset = (1 << 16) - 100; // 100 bytes left, 4096 asked
  batch[0].bytes = 4096;
  batch[0].bounce = &b;
  EXPECT_THROW(pool.wait(pool.submit(&batch)), std::runtime_error);
  EXPECT_EQ(batch[0].error, -1);
}

TEST(ExpertReaderPool, BadArgumentsReportError) {
  nntrainer::ExpertReaderPool pool(1, false);
  nntrainer::AlignedBuffer small(16);
  std::vector<nntrainer::ExpertRead> batch(2);
  batch[0].fd = -1; // no file
  batch[0].bytes = 16;
  batch[0].bounce = &small;
  batch[1].fd = 0;
  batch[1].bytes = 1 << 20; // bounce too small
  batch[1].bounce = &small;
  EXPECT_THROW(pool.wait(pool.submit(&batch)), std::runtime_error);
  EXPECT_NE(batch[0].error, 0);
  EXPECT_NE(batch[1].error, 0);
}

TEST(ExpertReaderPool, DestructorJoinsWithWorkQueued) {
  TempFile f(kFile);
  std::vector<nntrainer::AlignedBuffer> bounce(16);
  std::vector<nntrainer::ExpertRead> batch(16);
  for (size_t i = 0; i < batch.size(); ++i) {
    bounce[i].reserve(nntrainer::ExpertReaderPool::bounceBytes(65536));
    batch[i].fd = f.fd();
    batch[i].offset = i * 65536 + 3;
    batch[i].bytes = 65536;
    batch[i].bounce = &bounce[i];
  }
  {
    nntrainer::ExpertReaderPool pool(2, true);
    pool.submit(&batch);
  } // must drain the queue and join, not hang or crash
  for (const auto &r : batch)
    EXPECT_TRUE(holdsRecord(r));
}

// A read with a consume callback and no bounce stages in the reader's own
// buffer and hands the record to the callback on the reader thread.
TEST(ExpertReaderPool, ConsumeGetsTheRecordOnTheReaderThread) {
  TempFile f(kFile, 3);
  nntrainer::ExpertReaderPool pool(3, true);
  constexpr size_t kN = 12, kBytes = 300001;
  std::vector<std::vector<uint8_t>> got(kN);
  std::vector<nntrainer::ExpertRead> batch(kN);
  const auto caller = std::this_thread::get_id();
  std::atomic<int> on_caller{0};
  for (size_t i = 0; i < kN; ++i) {
    auto &r = batch[i];
    r.fd = f.fd();
    r.offset = i * 400003 + 2;
    r.bytes = kBytes;
    r.consume = [&, i](const uint8_t *rec, const nntrainer::ExpertRead &rd) {
      if (std::this_thread::get_id() == caller)
        on_caller++;
      got[i].assign(rec, rec + rd.bytes);
      return 0;
    };
  }
  ASSERT_NO_THROW(pool.wait(pool.submit(&batch)));
  EXPECT_EQ(on_caller.load(), 0);
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(batch[i].bounce, nullptr); // the caller's field is left as is
    ASSERT_EQ(got[i].size(), kBytes);
    bool same = true;
    for (size_t j = 0; j < kBytes && same; ++j)
      same = got[i][j] == pattern(batch[i].offset + j, 3);
    EXPECT_TRUE(same) << "record " << i;
  }
  // a failing consumer fails the batch
  std::vector<nntrainer::ExpertRead> bad(2);
  for (auto &r : bad) {
    r.fd = f.fd();
    r.offset = 4096;
    r.bytes = 1000;
  }
  bad[0].consume = [](const uint8_t *, const nntrainer::ExpertRead &) {
    return 0;
  };
  bad[1].consume = [](const uint8_t *, const nntrainer::ExpertRead &) -> int {
    throw std::runtime_error("device write failed");
  };
  EXPECT_THROW(pool.wait(pool.submit(&bad)), std::runtime_error);
  EXPECT_EQ(bad[0].error, 0);
  EXPECT_NE(bad[1].error, 0);
}

// A posted task runs once on a reader thread, and may itself read through
// the pool, helping with its own batch; a throwing task leaves the pool
// working.
TEST(ExpertReaderPool, PostedTaskRunsOnAReaderAndMayReadThroughThePool) {
  TempFile f(kFile, 5);
  for (unsigned int readers : {1u, 2u, 3u}) {
    nntrainer::ExpertReaderPool pool(readers, true);
    const auto caller = std::this_thread::get_id();
    std::mutex mu;
    std::condition_variable cv;
    bool finished = false, on_caller = true, read_ok = false;
    int runs = 0;
    pool.post([&]() {
      bool ok = true;
      std::vector<nntrainer::AlignedBuffer> b;
      std::vector<nntrainer::ExpertRead> rd(4);
      for (size_t i = 0; i < rd.size(); ++i)
        b.emplace_back(nntrainer::ExpertReaderPool::bounceBytes(5000));
      for (size_t i = 0; i < rd.size(); ++i) {
        rd[i].fd = f.fd();
        rd[i].offset = 12345 + 7000 * i;
        rd[i].bytes = 5000;
        rd[i].bounce = &b[i];
      }
      // one reader: this task is the only one, it reads every record
      pool.wait(pool.submit(&rd), /*help=*/true);
      for (const auto &r : rd)
        ok = ok && holdsRecord(r, 5);
      std::lock_guard<std::mutex> lk(mu);
      ++runs;
      on_caller = std::this_thread::get_id() == caller;
      read_ok = ok;
      finished = true;
      cv.notify_all();
    });
    {
      std::unique_lock<std::mutex> lk(mu);
      ASSERT_TRUE(
        cv.wait_for(lk, std::chrono::seconds(10), [&] { return finished; }))
        << readers << " reader(s)";
    }
    EXPECT_EQ(runs, 1);
    EXPECT_FALSE(on_caller);
    EXPECT_TRUE(read_ok) << readers << " reader(s)";
    pool.post([]() { throw std::runtime_error("x"); });
    std::vector<nntrainer::ExpertRead> rd(1);
    nntrainer::AlignedBuffer bb(nntrainer::ExpertReaderPool::bounceBytes(100));
    rd[0].fd = f.fd();
    rd[0].offset = 64;
    rd[0].bytes = 100;
    rd[0].bounce = &bb;
    EXPECT_NO_THROW(pool.wait(pool.submit(&rd)));
    EXPECT_TRUE(holdsRecord(rd[0], 5));
  }
}

#if !defined(_WIN32)
// The pool keeps an O_DIRECT descriptor per caller descriptor. Once the caller
// closes its descriptor the number is reused for the next file it opens: a
// read through the reused number must read that file, not the first one.
/**
 * The process pool with NNTR_MOE_READERS=1 and two concurrent posted jobs
 * that each read through the pool and wait with help (the decode step's
 * job does this). The first job submits its reads only after the second
 * job is queued, so the second job sits in the queue ahead of the first
 * job's reads while the only reader runs the first job. The helping wait
 * must serve its own reads past the queued task, or nothing ever runs them.
 * A deadlock cannot be unwound (the reader is stuck in the job and the
 * process pool joins it at exit): the test then reports and exits.
 */
TEST(ExpertReaderPool, OneReaderTwoPostedJobsReadingThroughThePool) {
#if defined(_WIN32)
  _putenv_s("NNTR_MOE_READERS", "1");
#else
  ::setenv("NNTR_MOE_READERS", "1", 1);
#endif
  nntrainer::ExpertReaderPool &pool = nntrainer::ExpertReaderPool::global();
  ASSERT_EQ(pool.readers(), 1u)
    << "the process pool was created before this test set its reader count";
  static TempFile f(7 * kRecord, 9); // outlives a stuck job
  struct State {
    std::mutex mu;
    std::condition_variable cv;
    bool second_posted = false;
    int finished = 0;
    bool ok[2] = {false, false};
  };
  auto *s = new State(); // leaked when the jobs deadlock
  auto job = [s, &pool](int id) {
    if (id == 0) {
      // let the second job queue up ahead of this job's reads
      std::unique_lock<std::mutex> lk(s->mu);
      s->cv.wait(lk, [s] { return s->second_posted; });
    }
    std::vector<nntrainer::AlignedBuffer> b;
    std::vector<nntrainer::ExpertRead> rd(3);
    for (size_t i = 0; i < rd.size(); ++i)
      b.emplace_back(nntrainer::ExpertReaderPool::bounceBytes(kRecord));
    for (size_t i = 0; i < rd.size(); ++i) {
      rd[i].fd = f.fd();
      rd[i].offset = 4099 + (uint64_t)kRecord * (2 * i + id);
      rd[i].bytes = kRecord;
      rd[i].bounce = &b[i];
    }
    bool ok = true;
    try {
      pool.wait(pool.submit(&rd), /*help=*/true);
    } catch (...) {
      ok = false;
    }
    for (const auto &r : rd)
      ok = ok && holdsRecord(r, 9);
    std::lock_guard<std::mutex> lk(s->mu);
    s->ok[id] = ok;
    ++s->finished;
    s->cv.notify_all();
  };
  pool.post([job]() { job(0); });
  pool.post([job]() { job(1); });
  {
    std::lock_guard<std::mutex> lk(s->mu);
    s->second_posted = true;
  }
  s->cv.notify_all();
  bool done = false;
  {
    std::unique_lock<std::mutex> lk(s->mu);
    done = s->cv.wait_for(lk, std::chrono::seconds(20),
                          [s] { return s->finished == 2; });
  }
  if (!done) {
    std::fprintf(stderr, "[  FAILED  ] one reader, two posted jobs: deadlock "
                         "(a job waits on reads queued behind the other)\n");
    std::fflush(stderr);
    std::_Exit(1);
  }
  EXPECT_TRUE(s->ok[0]);
  EXPECT_TRUE(s->ok[1]);
  delete s;
}

namespace {

/** holds a pool's reader threads in a posted task until release() */
struct Latch {
  std::mutex mu;
  std::condition_variable cv;
  bool open = false;
  int entered = 0;
  void block(nntrainer::ExpertReaderPool &pool) {
    pool.post([this]() {
      std::unique_lock<std::mutex> lk(mu);
      ++entered;
      cv.notify_all();
      cv.wait(lk, [this] { return open; });
    });
  }
  void waitEntered(int n) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [this, n] { return entered >= n; });
  }
  void release() {
    {
      std::lock_guard<std::mutex> lk(mu);
      open = true;
    }
    cv.notify_all();
  }
};

/** @a n reads of @a f, each into its own bounce */
std::shared_ptr<std::vector<nntrainer::ExpertRead>>
lowBatch(const TempFile &f, std::vector<nntrainer::AlignedBuffer> &b, size_t n,
         uint64_t first) {
  auto v = std::make_shared<std::vector<nntrainer::ExpertRead>>(n);
  b.clear();
  for (size_t i = 0; i < n; ++i)
    b.emplace_back(nntrainer::ExpertReaderPool::bounceBytes(kRecord));
  for (size_t i = 0; i < n; ++i) {
    (*v)[i].fd = f.fd();
    (*v)[i].offset = first + (uint64_t)kRecord * i;
    (*v)[i].bytes = kRecord;
    (*v)[i].bounce = &b[i];
  }
  return v;
}

} // namespace

TEST(ExpertReaderPool, LowPriorityReadsRunAfterTheQueuedDemandReads) {
  TempFile f(kFile, 3);
  nntrainer::ExpertReaderPool pool(1, true);
  Latch latch;
  latch.block(pool);
  latch.waitEntered(1);
  std::mutex mu;
  std::vector<int> order;
  std::condition_variable cv;
  int low_done = 0;
  std::vector<nntrainer::AlignedBuffer> lb;
  auto low = lowBatch(f, lb, 2, 4098);
  for (auto &r : *low) {
    r.consume = [&](const uint8_t *, const nntrainer::ExpertRead &) {
      std::lock_guard<std::mutex> lk(mu);
      order.push_back(1);
      return 0;
    };
    r.finished = [&](nntrainer::ExpertRead &) {
      std::lock_guard<std::mutex> lk(mu);
      ++low_done;
      cv.notify_all();
    };
  }
  pool.submitLow(low); // queued first
  low.reset();         // the pool keeps the batch alive
  EXPECT_EQ(pool.lowQueued(), 2u);
  std::vector<nntrainer::AlignedBuffer> db;
  for (int i = 0; i < 2; ++i)
    db.emplace_back(nntrainer::ExpertReaderPool::bounceBytes(kRecord));
  std::vector<nntrainer::ExpertRead> demand(2);
  for (size_t i = 0; i < demand.size(); ++i) {
    demand[i].fd = f.fd();
    demand[i].offset = 3 * kRecord + 5 + i * kRecord;
    demand[i].bytes = kRecord;
    demand[i].bounce = &db[i];
    demand[i].consume = [&](const uint8_t *, const nntrainer::ExpertRead &) {
      std::lock_guard<std::mutex> lk(mu);
      order.push_back(0);
      return 0;
    };
  }
  const auto t = pool.submit(&demand);
  latch.release();
  pool.wait(t);
  {
    std::unique_lock<std::mutex> lk(mu);
    ASSERT_TRUE(
      cv.wait_for(lk, std::chrono::seconds(10), [&] { return low_done == 2; }));
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order, (std::vector<int>{0, 0, 1, 1}));
  }
  for (const auto &r : demand)
    EXPECT_TRUE(holdsRecord(r, 3));
  for (size_t i = 0; i < lb.size(); ++i) {
    nntrainer::ExpertRead r;
    r.offset = 4098 + (uint64_t)kRecord * i;
    r.bytes = kRecord;
    r.bounce = &lb[i];
    r.data_offset = r.offset % 4096; // the superset starts a page below
    EXPECT_TRUE(holdsRecord(r, 3));
  }
}

TEST(ExpertReaderPool, AHelpingWaiterNeverTakesALowPriorityRead) {
  TempFile f(kFile, 4);
  nntrainer::ExpertReaderPool pool(1, true);
  Latch latch;
  latch.block(pool);
  latch.waitEntered(1);
  std::vector<nntrainer::AlignedBuffer> lb;
  auto low = lowBatch(f, lb, 1, 77);
  std::atomic<int> low_done{0};
  (*low)[0].finished = [&](nntrainer::ExpertRead &) { ++low_done; };
  pool.submitLow(low);
  nntrainer::AlignedBuffer b(nntrainer::ExpertReaderPool::bounceBytes(kRecord));
  std::vector<nntrainer::ExpertRead> demand(1);
  demand[0].fd = f.fd();
  demand[0].offset = 2 * kRecord;
  demand[0].bytes = kRecord;
  demand[0].bounce = &b;
  // the only reader is held: this thread reads its own record, and only it
  pool.wait(pool.submit(&demand), /*help=*/true);
  EXPECT_TRUE(holdsRecord(demand[0], 4));
  EXPECT_EQ(pool.lowQueued(), 1u);
  EXPECT_EQ(low_done.load(), 0);
  latch.release();
  for (int i = 0; i < 1000 && low_done.load() == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(low_done.load(), 1);
  EXPECT_EQ((*low)[0].error, 0);
}

TEST(ExpertReaderPool, OnceFlagCancelsALowReadNobodyStarted) {
  TempFile f(kFile, 5);
  nntrainer::ExpertReaderPool pool(1, true);
  Latch latch;
  latch.block(pool);
  latch.waitEntered(1);
  std::vector<nntrainer::AlignedBuffer> lb;
  auto low = lowBatch(f, lb, 2, 4096);
  std::atomic<bool> once[2];
  once[0] = false;
  once[1] = false;
  std::atomic<int> consumed{0}, finished{0}, cancelled{0};
  for (int i = 0; i < 2; ++i) {
    (*low)[i].once = &once[i];
    (*low)[i].consume = [&](const uint8_t *, const nntrainer::ExpertRead &) {
      ++consumed;
      return 0;
    };
    (*low)[i].finished = [&](nntrainer::ExpertRead &r) {
      if (r.error == ECANCELED)
        ++cancelled;
      ++finished;
    };
  }
  pool.submitLow(low);
  // a party takes record 0 before any reader started it: the pool skips it
  EXPECT_FALSE(once[0].exchange(true));
  latch.release();
  for (int i = 0; i < 1000 && finished.load() < 2; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(finished.load(), 2);
  EXPECT_EQ(cancelled.load(), 1);
  EXPECT_EQ(consumed.load(), 1);
  EXPECT_EQ((*low)[0].error, ECANCELED);
  EXPECT_EQ((*low)[1].error, 0);
  // and once a read started, a late exchange finds it taken
  EXPECT_TRUE(once[1].exchange(true));
}

TEST(ExpertReaderPool, LowReadsLeftAtShutdownEndCancelled) {
  TempFile f(kFile, 6);
  std::atomic<int> cancelled{0}, read{0};
  {
    auto pool = std::make_unique<nntrainer::ExpertReaderPool>(1, true);
    Latch latch;
    latch.block(*pool);
    latch.waitEntered(1);
    std::vector<nntrainer::AlignedBuffer> lb;
    auto low = lowBatch(f, lb, 3, 0);
    for (auto &r : *low)
      r.finished = [&](nntrainer::ExpertRead &q) {
        if (q.error == ECANCELED)
          ++cancelled;
        else if (q.error == 0)
          ++read;
      };
    pool->submitLow(low);
    std::thread releaser([&latch]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      latch.release();
    });
    pool.reset(); // stop first, then the held reader is released
    releaser.join();
  }
  EXPECT_EQ(cancelled.load() + read.load(), 3);
  EXPECT_EQ(cancelled.load(), 3);
}

namespace {

/** record offsets of a fake layer: expert e at 4098 + e * kRecord */
std::vector<uint64_t> recordOffsets(unsigned int experts) {
  std::vector<uint64_t> o(experts);
  for (unsigned int e = 0; e < experts; ++e)
    o[e] = 4098 + (uint64_t)e * kRecord;
  return o;
}

/** true when @a p holds the record of expert @a e (recordOffsets) */
bool isRecord(const uint8_t *p, unsigned int e, uint64_t seed) {
  const uint64_t off = 4098 + (uint64_t)e * kRecord;
  for (size_t i = 0; i < kRecord; i += 997)
    if (p[i] != pattern(off + i, seed))
      return false;
  return p[kRecord - 1] == pattern(off + kRecord - 1, seed);
}

/** poll @a pred for up to 10 s */
template <typename P> bool eventually(P pred) {
  for (int i = 0; i < 2000; ++i) {
    if (pred())
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return pred();
}

} // namespace

// A staged record is never "used": only a claim by the step that misses it
// serves it; what the step does not claim is dropped at its close, and a
// prediction for a step that already ran is refused. The cache has no
// handle on any slot or residency policy: it cannot evict anything.
TEST(ExpertStageCache, AStagedRecordCountsOnlyWhenTheStepClaimsIt) {
  TempFile f(5 * kRecord + 8192, 11);
  const auto off = recordOffsets(5);
  nntrainer::ExpertReaderPool pool(2, true);
  nntrainer::ExpertStageCache c(pool, 2);
  nntrainer::ExpertStageCache::Source src{f.fd(), &off, kRecord};
  const uint64_t g = c.generation(7);
  EXPECT_EQ(c.stage(7, g, src, {3, 1, 4}), 3u);
  EXPECT_EQ(c.stage(7, g, src, {3}), 0u); // already staged
  ASSERT_TRUE(eventually([&] { return c.stats().reads.load() == 3; }));
  EXPECT_EQ(c.stats().served.load(), 0u);
  auto t = c.claim(7, 1);
  ASSERT_EQ(t.kind, nntrainer::ExpertStageCache::Claim::ready);
  EXPECT_TRUE(isRecord(t.entry->data(), 1, 11));
  c.release(t.entry);
  EXPECT_EQ(c.claim(7, 1).kind, nntrainer::ExpertStageCache::Claim::none);
  EXPECT_EQ(c.claim(7, 0).kind, nntrainer::ExpertStageCache::Claim::none);
  EXPECT_EQ(c.close(7), 2u); // 3 and 4: read, never claimed
  EXPECT_EQ(c.stats().served.load(), 1u);
  EXPECT_EQ(c.stats().wasted.load(), 2u);
  EXPECT_EQ(c.claim(7, 3).kind, nntrainer::ExpertStageCache::Claim::none);
  EXPECT_EQ(c.size(), 0u);
  // the prediction for the step that ran comes late: refused
  EXPECT_EQ(c.stage(7, g, src, {2}), 0u);
  EXPECT_EQ(c.stats().late.load(), 1u);
  EXPECT_EQ(c.generation(7), g + 1);
  // other layers are untouched by layer 7's close
  EXPECT_EQ(c.stage(8, c.generation(8), src, {2}), 1u);
  ASSERT_TRUE(eventually([&] { return c.inFlight(8) == 0; }));
  EXPECT_EQ(c.claim(8, 2).kind, nntrainer::ExpertStageCache::Claim::ready);
}

// The step's claim of a record whose read nobody started takes the read
// over (the queued low-priority read is skipped, no I/O twice); a record
// not yet submitted (window full) likewise.
TEST(ExpertStageCache, AClaimTakesOverAReadNobodyStarted) {
  TempFile f(5 * kRecord + 8192, 12);
  const auto off = recordOffsets(5);
  nntrainer::ExpertReaderPool pool(1, true);
  Latch latch;
  latch.block(pool);
  latch.waitEntered(1);
  nntrainer::ExpertStageCache c(pool, 2);
  nntrainer::ExpertStageCache::Source src{f.fd(), &off, kRecord};
  EXPECT_EQ(c.stage(3, c.generation(3), src, {0, 1, 2}), 3u);
  EXPECT_EQ(c.inFlight(3), 2u); // the window: 0 and 1 submitted, 2 pending
  EXPECT_EQ(pool.lowQueued(), 2u);
  EXPECT_EQ(c.claim(3, 0).kind, nntrainer::ExpertStageCache::Claim::take_over);
  EXPECT_EQ(c.claim(3, 2).kind, nntrainer::ExpertStageCache::Claim::take_over);
  latch.release();
  ASSERT_TRUE(eventually([&] { return c.inFlight(3) == 0; }));
  EXPECT_EQ(c.stats().reads.load(), 1u); // expert 1 only
  EXPECT_EQ(c.stats().taken_over.load(), 2u);
  auto t = c.claim(3, 1);
  ASSERT_EQ(t.kind, nntrainer::ExpertStageCache::Claim::ready);
  EXPECT_TRUE(isRecord(t.entry->data(), 1, 12));
  c.release(t.entry);
  EXPECT_EQ(c.close(3), 0u);
  EXPECT_EQ(c.stats().wasted.load(), 0u);
}

// At most `window` reads of a layer are submitted at once; each finished
// read submits the next from the reader thread.
TEST(ExpertStageCache, TheWindowBoundsTheReadsInFlightPerLayer) {
  TempFile f(8 * kRecord + 8192, 13);
  const auto off = recordOffsets(8);
  nntrainer::ExpertReaderPool pool(4, true);
  nntrainer::ExpertStageCache c(pool, 2);
  nntrainer::ExpertStageCache::Source src{f.fd(), &off, kRecord};
  std::atomic<unsigned int> peak{0};
  std::atomic<bool> stop{false};
  std::thread watch([&] {
    while (!stop.load()) {
      const unsigned int n = c.inFlight(5);
      unsigned int p = peak.load();
      while (n > p && !peak.compare_exchange_weak(p, n)) {
      }
      std::this_thread::yield();
    }
  });
  EXPECT_EQ(c.stage(5, c.generation(5), src, {0, 1, 2, 3, 4, 5, 6, 7}), 8u);
  EXPECT_LE(c.inFlight(5), 2u);
  ASSERT_TRUE(eventually([&] { return c.stats().reads.load() == 8; }));
  stop = true;
  watch.join();
  EXPECT_LE(peak.load(), 2u);
  EXPECT_GE(peak.load(), 1u);
  for (unsigned int e = 0; e < 8; ++e) {
    auto t = c.claim(5, e);
    ASSERT_EQ(t.kind, nntrainer::ExpertStageCache::Claim::ready);
    EXPECT_TRUE(isRecord(t.entry->data(), e, 13));
    c.release(t.entry);
  }
}

// The decode protocol on a pool of ONE reader (NNTR_MOE_READERS=1): every
// step job is a posted task that reads its misses (submit + helping wait),
// serves the misses staged for it, closes its layer and only then, on its
// way out, stages the prediction for the next layer -- it never waits on
// another task. The old form (a posted prefetch task waiting for the step's
// job) could leave the only reader blocked. Must finish, bytes exact.
TEST(ExpertStageCache, OneReaderDecodeWithReadAheadNeverDeadlocks) {
  for (unsigned int readers : {1u, 2u}) {
    TempFile f(16 * kRecord + 8192, 14);
    const auto off = recordOffsets(16);
    auto pool = std::make_unique<nntrainer::ExpertReaderPool>(readers, true);
    nntrainer::ExpertStageCache c(*pool, 2);
    nntrainer::ExpertStageCache::Source src{f.fd(), &off, kRecord};
    constexpr unsigned int kLayers = 6, kTokens = 4;
    struct Run {
      std::mutex mu;
      std::condition_variable cv;
      unsigned int steps = 0;
      bool bad = false;
    };
    auto run = std::make_shared<Run>();
    // misses of (layer, token): 3 experts; the prediction staged for the
    // next layer hits two of its misses and one expert it does not need
    auto misses = [](unsigned int l, unsigned int t) {
      return std::vector<unsigned int>{(l + t) % 16, (l + t + 5) % 16,
                                       (l + t + 9) % 16};
    };
    std::function<void(unsigned int, unsigned int)> step;
    step = [&, run](unsigned int l, unsigned int t) {
      bool ok = true;
      std::vector<nntrainer::ExpertRead> reads;
      std::vector<nntrainer::AlignedBuffer> bufs;
      std::vector<std::shared_ptr<nntrainer::ExpertStageCache::Entry>> fl;
      const auto need = misses(l, t);
      bufs.reserve(need.size());
      for (unsigned int e : need) {
        auto tk = c.claim(l, e);
        if (tk.kind == nntrainer::ExpertStageCache::Claim::ready) {
          ok = ok && isRecord(tk.entry->data(), e, 14);
          c.release(tk.entry);
          continue;
        }
        if (tk.kind == nntrainer::ExpertStageCache::Claim::in_flight) {
          fl.push_back(tk.entry);
          continue;
        }
        bufs.emplace_back(nntrainer::ExpertReaderPool::bounceBytes(kRecord));
        reads.emplace_back();
        reads.back().fd = f.fd();
        reads.back().offset = off[e];
        reads.back().bytes = kRecord;
        reads.back().bounce = &bufs.back();
      }
      try {
        pool->wait(pool->submit(&reads), /*help=*/true);
      } catch (...) {
        ok = false;
      }
      for (const auto &r : reads)
        ok = ok && holdsRecord(r, 14);
      for (auto &e : fl) {
        // a read running on another reader: it ends
        if (c.waitReady(*e))
          ok = ok && isRecord(e->data(), e->expert, 14);
        c.release(e);
      }
      c.close(l);
      // the completion path: the prediction for the next layer, staged
      // without waiting, then the next step is posted
      if (l + 1 < kLayers) {
        auto nx = misses(l + 1, t);
        c.stage(l + 1, c.generation(l + 1), src, {nx[2], nx[0], 15});
        pool->post([&step, l, t]() { step(l + 1, t); });
      } else if (t + 1 < kTokens) {
        pool->post([&step, t]() { step(0, t + 1); });
      }
      std::lock_guard<std::mutex> lk(run->mu);
      run->bad = run->bad || !ok;
      ++run->steps;
      run->cv.notify_all();
    };
    pool->post([&step]() { step(0, 0); });
    bool done = false;
    {
      std::unique_lock<std::mutex> lk(run->mu);
      done = run->cv.wait_for(lk, std::chrono::seconds(30),
                              [&] { return run->steps == kLayers * kTokens; });
    }
    if (!done) {
      std::fprintf(stderr,
                   "[  FAILED  ] %u reader(s): decode with read-ahead "
                   "deadlocked\n",
                   readers);
      std::fflush(stderr);
      std::_Exit(1);
    }
    EXPECT_FALSE(run->bad) << readers << " reader(s)";
    // every step's first prediction hit is served or taken over, never lost
    const auto &s = c.stats();
    EXPECT_EQ(s.served.load() + s.taken_over.load() + s.cancelled.load() +
                s.wasted.load() + s.failed.load() + s.late.load(),
              s.staged.load() + s.late.load())
      << readers << " reader(s)";
    pool.reset(); // joins the readers before the cache goes
  }
}

TEST(ExpertReaderPool, ReusedDescriptorNumberReadsTheNewFile) {
  nntrainer::ExpertReaderPool pool(2, true);
  nntrainer::AlignedBuffer b(nntrainer::ExpertReaderPool::bounceBytes(kRecord));
  std::vector<nntrainer::ExpertRead> batch(1);
  auto readAt = [&](int fd) {
    batch[0] = nntrainer::ExpertRead();
    batch[0].fd = fd;
    batch[0].offset = 4098;
    batch[0].bytes = kRecord;
    batch[0].bounce = &b;
    pool.wait(pool.submit(&batch));
  };
  int n = -1;
  {
    TempFile first(kFile, 0);
    n = ::dup(first.fd());
    ASSERT_GE(n, 0);
    ASSERT_NO_THROW(readAt(n));
    EXPECT_TRUE(holdsRecord(batch[0], 0));
    ::close(n); // number n is free again; the first file is deleted below
  }
  TempFile second(kFile, 7);
  ASSERT_EQ(::dup2(second.fd(), n), n); // the same number, another file
  ASSERT_NO_THROW(readAt(n));
  EXPECT_TRUE(holdsRecord(batch[0], 7)) << "read the closed file's bytes";
  EXPECT_FALSE(holdsRecord(batch[0], 0));
  // an explicit forget, then the same number again: still the second file
  pool.forgetFd(n);
  ASSERT_NO_THROW(readAt(n));
  EXPECT_TRUE(holdsRecord(batch[0], 7));
  ::close(n);
  // a closed number is reported, not read through a stale descriptor
  batch[0] = nntrainer::ExpertRead();
  batch[0].fd = n;
  batch[0].offset = 0;
  batch[0].bytes = 4096;
  batch[0].bounce = &b;
  EXPECT_THROW(pool.wait(pool.submit(&batch)), std::runtime_error);
}
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

namespace {

/** true when @a p holds bytes [off, off + n) of the file with contents
 *  @a seed */
bool holdsAt(const uint8_t *p, uint64_t off, size_t n, uint64_t seed) {
  for (size_t i = 0; i < n; ++i)
    if (p[i] != pattern(off + i, seed))
      return false;
  return true;
}

} // namespace

TEST(ExpertReaderPool, ReadsStraightIntoACallersAlignedDestination) {
  TempFile f(kFile, 11);
  nntrainer::ExpertReaderPool pool(2, true);
  const size_t a = nntrainer::AlignedBuffer::kAlign;
  // slots of the flags protocol: 4 KiB pages that hold any 64-byte aligned
  // record's O_DIRECT window
  const size_t stride = (kRecord + a - 64 + a - 1) / a * a;
  const uint64_t offs[] = {64, 4096 + 4032, 1000064,
                           (kFile - kRecord) / 64 * 64};
  nntrainer::AlignedBuffer slots(4 * stride);
  std::memset(slots.data(), 0xAB, 4 * stride);
  std::vector<nntrainer::ExpertRead> batch(4);
  std::atomic<int> consumed{0};
  for (size_t i = 0; i < 4; ++i) {
    batch[i].fd = f.fd();
    batch[i].offset = offs[i];
    batch[i].bytes = kRecord;
    batch[i].dst = slots.data() + i * stride;
    batch[i].consume = [&, i](const uint8_t *rec,
                              const nntrainer::ExpertRead &r) {
      // consume gets the record where it lies, no staging copy
      if (rec == slots.data() + i * stride + r.offset % a)
        ++consumed;
      return 0;
    };
  }
  ASSERT_NO_THROW(pool.wait(pool.submit(&batch)));
  EXPECT_EQ(consumed.load(), 4);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(batch[i].error, 0);
    EXPECT_EQ(batch[i].data_offset, offs[i] % a);
    EXPECT_LE(batch[i].data_offset + kRecord, stride);
    EXPECT_TRUE(
      holdsAt(slots.data() + i * stride + offs[i] % a, offs[i], kRecord, 11))
      << "slot " << i;
  }
  // a destination off the page: refused, nothing written
  std::vector<nntrainer::ExpertRead> bad(1);
  bad[0].fd = f.fd();
  bad[0].offset = 64;
  bad[0].bytes = 4096;
  bad[0].dst = slots.data() + 64;
  EXPECT_THROW(pool.wait(pool.submit(&bad)), std::runtime_error);
  EXPECT_EQ(bad[0].error, EINVAL);
}

TEST(ExpertReaderPool, DetachedReadsRunInOrderAndRetire) {
  TempFile f(kFile, 12);
  nntrainer::ExpertReaderPool pool(1, true);
  Latch latch;
  latch.block(pool);
  latch.waitEntered(1);
  std::mutex mu;
  std::vector<int> order;
  std::atomic<int> finished{0};
  std::vector<nntrainer::AlignedBuffer> hb, lb;
  auto high = lowBatch(f, hb, 2, 64);
  auto low = lowBatch(f, lb, 1, 128);
  for (size_t i = 0; i < 2; ++i)
    (*high)[i].finished = [&, i](nntrainer::ExpertRead &r) {
      std::lock_guard<std::mutex> lk(mu);
      order.push_back(r.error == 0 ? (int)i : -100);
      ++finished;
    };
  (*low)[0].finished = [&](nntrainer::ExpertRead &r) {
    std::lock_guard<std::mutex> lk(mu);
    order.push_back(r.error == 0 ? 10 : -100);
    ++finished;
  };
  pool.submitDetached(low, /*low=*/true);
  pool.submitDetached(high, /*low=*/false);
  low.reset();
  high.reset();
  latch.release();
  for (int i = 0; i < 1000 && finished.load() < 3; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  ASSERT_EQ(finished.load(), 3);
  EXPECT_EQ(order, (std::vector<int>{0, 1, 10})); // demand first, then low
  for (int i = 0; i < 200 && pool.batchesInFlight() != 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(pool.batchesInFlight(), 0u);
}

TEST(ExpertReaderPool, AHelpingWaiterRetiresADetachedReadItRan) {
  TempFile f(kFile, 13);
  nntrainer::ExpertReaderPool pool(1, true);
  // the reader sits in A's consume until D (detached) is done: only the
  // helping waiter of A can run D, and it must retire D's batch
  std::mutex mu;
  std::condition_variable cv;
  bool d_done = false, a_in = false;
  std::vector<nntrainer::AlignedBuffer> ab, db;
  std::vector<nntrainer::ExpertRead> a(1);
  nntrainer::AlignedBuffer abuf(nntrainer::ExpertReaderPool::bounceBytes(4096));
  a[0].fd = f.fd();
  a[0].offset = 4096;
  a[0].bytes = 4096;
  a[0].bounce = &abuf;
  a[0].consume = [&](const uint8_t *, const nntrainer::ExpertRead &) {
    std::unique_lock<std::mutex> lk(mu);
    a_in = true;
    cv.notify_all();
    cv.wait_for(lk, std::chrono::seconds(10), [&] { return d_done; });
    return d_done ? 0 : ETIMEDOUT;
  };
  const auto ta = pool.submit(&a);
  {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait_for(lk, std::chrono::seconds(10), [&] { return a_in; });
    ASSERT_TRUE(a_in);
  }
  auto d = lowBatch(f, db, 1, 8192);
  const std::thread::id me = std::this_thread::get_id();
  std::thread::id ran;
  (*d)[0].finished = [&](nntrainer::ExpertRead &r) {
    std::lock_guard<std::mutex> lk(mu);
    ran = std::this_thread::get_id();
    d_done = r.error == 0;
    cv.notify_all();
  };
  pool.submitDetached(d, /*low=*/false);
  d.reset();
  ASSERT_NO_THROW(pool.wait(ta, /*help=*/true));
  EXPECT_TRUE(d_done);
  EXPECT_EQ(ran, me);
  EXPECT_EQ(pool.batchesInFlight(), 0u);
}
