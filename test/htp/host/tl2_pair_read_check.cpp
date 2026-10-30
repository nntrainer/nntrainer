// SPDX-License-Identifier: Apache-2.0
/**
 * @file tl2_pair_read_check.cpp
 * @brief Native LFM2/Gemma4 direct reads and speculative buffer ownership.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#include <cstdio>

#if defined(__linux__)
#include "htp_expert_prefetch.h"
#include "htp_expert_read_pool.h"
#include "hvx_gemv_u8tl2.h"
#include "weight_file_reader.h"
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {
struct Weight {
  unsigned K, N;
  std::vector<unsigned char> codes;
  std::vector<float> tail;
  static int value(unsigned k, unsigned n) {
    return int((k * 19u + n * 23u + k * n) % 3u) - 1;
  }
  Weight(unsigned k, unsigned n) :
    K(k), N(n), codes(hvx_tl2_weight_bytes(k, n)), tail(2 * n) {
    std::vector<int8_t> q(size_t(k) * n);
    for (unsigned j = 0; j < n; ++j) {
      int sum = 0;
      for (unsigned i = 0; i < k; ++i) {
        q[size_t(i) * n + j] = value(i, j);
        sum += q[size_t(i) * n + j];
      }
      tail[j] = 0.001f * float(j % 17 + 1);
      tail[n + j] = float(sum);
    }
    assert(hvx_tl2_pack(q.data(), k, n, codes.data()) == 0);
  }
};

struct Payload {
  std::array<std::vector<unsigned char>, 4> parts;
};
struct Buffer {
  std::shared_ptr<Payload> payload;
  uint64_t offset = 0;
  bool gated = false;
  int rc = 0;
  unsigned handle = 0;
};
using Queue = nntrainer::HtpExpertPrefetch<Buffer>;

struct Gate {
  std::promise<void> entered, allow_io, read_done, allow_return;
  std::future<void> entered_event = entered.get_future();
  std::shared_future<void> io_event = allow_io.get_future().share();
  std::future<void> read_event = read_done.get_future();
  std::shared_future<void> return_event = allow_return.get_future().share();
};

static void checkpoint(std::future<void> &f) {
  assert(f.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  f.get();
}
static void completed(Queue &q, uint64_t n) {
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (q.stats().completed < n) {
    assert(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
}

struct Fixture {
  Weight gu, dn;
  std::array<size_t, 4> sizes;
  std::vector<unsigned char> file;
  uint64_t offsets[2]{};
  std::string path;
  int fd = -1, direct = -1;
  std::atomic<unsigned> calls{0};

  Fixture(const char *directory, unsigned K, unsigned inter, unsigned N) :
    gu(K, 2u * inter),
    dn(inter, N),
    sizes{gu.codes.size(), gu.tail.size() * sizeof(float), dn.codes.size(),
          dn.tail.size() * sizeof(float)},
    path(std::string(directory) + "/nntr-tl2-pair-XXXXXX") {
    const size_t bytes = sizes[0] + sizes[1] + sizes[2] + sizes[3];
    file.resize(37 + 2 * bytes);
    size_t at = 37;
    for (unsigned record = 0; record < 2; ++record) {
      offsets[record] = at;
      assert(bytes <= nntrainer::WEIGHT_DIRECT_WINDOW_BYTES -
                        at % nntrainer::WEIGHT_DIRECT_ALIGNMENT);
      for (const auto *w : {&gu, &dn}) {
        memcpy(file.data() + at, w->codes.data(), w->codes.size());
        at += w->codes.size();
        auto tail = w->tail;
        for (unsigned j = 0; j < w->N; ++j)
          tail[j] *= float(record + 1);
        memcpy(file.data() + at, tail.data(), tail.size() * sizeof(float));
        at += tail.size() * sizeof(float);
      }
    }
    fd = mkstemp(path.data());
    assert(fd >= 0);
    assert(write(fd, file.data(), file.size()) == ssize_t(file.size()));
    assert(fsync(fd) == 0);
    assert(setenv("NNTR_HTP_PAGE_CACHE", "0", 1) == 0);
    direct = nntrainer::openWeightFile(path.c_str());
    assert(direct >= 0);
  }
  ~Fixture() {
    const long page = sysconf(_SC_PAGESIZE);
    assert(page > 0);
    std::vector<unsigned char> resident((file.size() + page - 1) / page);
    void *mapping = mmap(nullptr, file.size(), PROT_READ, MAP_SHARED, fd, 0);
    assert(mapping != MAP_FAILED);
    assert(mincore(mapping, file.size(), resident.data()) == 0);
    for (auto v : resident)
      assert((v & 1) == 0);
    assert(munmap(mapping, file.size()) == 0);
    assert(close(direct) == 0);
    assert(close(fd) == 0);
    assert(unlink(path.c_str()) == 0);
  }
  std::shared_ptr<Payload> payload() const {
    auto out = std::make_shared<Payload>();
    for (size_t i = 0; i < sizes.size(); ++i)
      out->parts[i].assign(sizes[i] + 2, 0xA5);
    return out;
  }
  int read(Buffer &b, bool pair = true) {
    nntrainer::WeightReadSlice slices[4];
    for (size_t i = 0; i < sizes.size(); ++i)
      slices[i] = {b.payload->parts[i].data() + 1, sizes[i]};
    nntrainer::WeightReadTiming timing;
    assert(nntrainer::weightSetReadTiming(&timing) == nullptr);
    if (pair) {
      b.rc = nntrainer::weightPreadScatter(direct, slices, 4, b.offset);
    } else {
      b.rc = nntrainer::weightPreadSplit(direct, slices[0].data, sizes[0],
                                         slices[1].data, sizes[1], b.offset);
      if (b.rc == 0)
        b.rc = nntrainer::weightPreadSplit(direct, slices[2].data, sizes[2],
                                           slices[3].data, sizes[3],
                                           b.offset + sizes[0] + sizes[1]);
    }
    assert(nntrainer::weightSetReadTiming(nullptr) == &timing);
    ++calls;
    for (const auto &p : b.payload->parts)
      assert(p.front() == 0xA5 && p.back() == 0xA5);
    if (pair)
      assert(timing.calls == 1 &&
             timing.requested_bytes <= nntrainer::WEIGHT_DIRECT_WINDOW_BYTES);
    if (b.rc == 0) {
      assert(timing.calls == (pair ? 1u : 2u));
      check(b);
    } else {
      assert(b.rc == ENODATA);
      if (pair) {
        assert(timing.copy_ns == 0);
        for (const auto &p : b.payload->parts)
          for (auto byte : p)
            assert(byte == 0xA5); // Short single reads do not scatter.
      }
    }
    return b.rc;
  }
  void check(const Buffer &b) const {
    size_t at = b.offset;
    for (size_t i = 0; i < sizes.size(); ++i) {
      assert(memcmp(b.payload->parts[i].data() + 1, file.data() + at,
                    sizes[i]) == 0);
      at += sizes[i];
    }
  }
  Queue::ReadResult run(Buffer &b, Gate &g) {
    if (b.gated) {
      g.entered.set_value();
      g.io_event.wait();
    }
    const int rc = read(b);
    if (b.gated) {
      g.read_done.set_value();
      g.return_event.wait();
    }
    // An in-flight pair finishes under its worker even after epoch expiry.
    return rc == 0 ? Queue::ReadResult::Complete : Queue::ReadResult::Failed;
  }
};

static void splitPairAndMetadata(Fixture &f) {
  for (unsigned record = 0; record < 2; ++record) {
    Buffer pair{f.payload(), f.offsets[record]},
      split{f.payload(), pair.offset};
    assert(f.read(pair) == 0 && f.read(split, false) == 0);
    assert(pair.payload->parts == split.payload->parts);
    for (size_t wi = 0; wi < 2; ++wi) {
      const Weight &w = wi == 0 ? f.gu : f.dn;
      std::vector<int8_t> unpacked(size_t(w.K) * w.N);
      assert(hvx_tl2_unpack(pair.payload->parts[2 * wi].data() + 1, w.K, w.N,
                            unpacked.data()) == 0);
      std::vector<float> tail(2 * w.N);
      memcpy(tail.data(), pair.payload->parts[2 * wi + 1].data() + 1,
             tail.size() * sizeof(float));
      for (unsigned j = 0; j < w.N; ++j) {
        int sum = 0;
        for (unsigned i = 0; i < w.K; ++i) {
          assert(unpacked[size_t(i) * w.N + j] == Weight::value(i, j));
          sum += unpacked[size_t(i) * w.N + j];
        }
        assert(tail[j] == w.tail[j] * float(record + 1));
        assert(tail[w.N + j] == float(sum));
      }
    }
  }
  // A one-byte short GU, then a complete GU followed by a short DN.
  for (size_t bytes : {f.sizes[0] + f.sizes[1],
                       f.sizes[0] + f.sizes[1] + f.sizes[2] + f.sizes[3]}) {
    Buffer pair{f.payload(), f.file.size() - bytes + 1};
    Buffer split{f.payload(), pair.offset};
    assert(f.read(pair) == ENODATA && f.read(split, false) == ENODATA);
  }
}

static void prefillBatchJoin(Fixture &f) {
  nntrainer::HtpExpertReadPool pool(6);
  std::atomic<unsigned> read_ok{0};
  unsigned registered = 0;
  auto read = [&](size_t job, bool fail) {
    Buffer b{f.payload(),
             fail && job == 7 ? f.file.size() - 16 : f.offsets[job % 2]};
    if (f.read(b))
      throw std::runtime_error("short expert");
    ++read_ok;
  };
  pool.run(18, [&](size_t j) { read(j, false); });
  assert(read_ok == 18);
  registered += 18;
  bool failed = false;
  try {
    pool.run(18, [&](size_t j) { read(j, true); });
    registered += 18;
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(failed && read_ok == 35 && registered == 18);
  pool.run(6, [&](size_t j) { read(j, false); });
  assert(read_ok == 41); // All callbacks joined; the pool remains reusable.
}

static void retiredReadOwnsSlot(Fixture &f) {
  int keys[2]{};
  auto original = f.payload();
  Gate gate;
  Queue q({{original}}, 1, [&](Buffer &b, const Queue::CancelCheck &) {
    return f.run(b, gate);
  });
  assert(q.submit(keys, 1, [&](Buffer &b) {
    b.offset = f.offsets[0];
    b.gated = true;
  }));
  checkpoint(gate.entered_event);
  q.expireBefore(2);
  assert(!q.claim(keys));
  assert(!q.take(keys, [](Buffer &) { assert(false); }));
  assert(!q.submit(keys + 1, 2, [](Buffer &) { assert(false); }));
  gate.allow_io.set_value();
  checkpoint(gate.read_event);
  // I/O has finished, but the callback still owns the slot until return.
  assert(!q.submit(keys + 1, 2, [](Buffer &) { assert(false); }));
  gate.allow_return.set_value();
  completed(q, 1);
  assert(q.stats().discarded == 1 && q.stats().cancelled == 0);
  assert(!q.take(keys, [](Buffer &) { assert(false); }));
  assert(q.submit(keys + 1, 2, [&](Buffer &b) {
    assert(b.payload == original);
    b.offset = f.offsets[1];
    b.gated = false;
  }));
  completed(q, 2);
  assert(q.take(keys + 1, [&](Buffer &b) { f.check(b); }));
  assert(q.stats().used == 1 && q.stats().failed == 0);
}

static void claimedReadSurvivesExpiry(Fixture &f) {
  int key{};
  Gate gate;
  Queue q({{f.payload()}}, 1, [&](Buffer &b, const Queue::CancelCheck &) {
    return f.run(b, gate);
  });
  assert(q.submit(&key, 10, [&](Buffer &b) {
    b.offset = f.offsets[0];
    b.gated = true;
    b.handle = 42;
  }));
  checkpoint(gate.entered_event);
  assert(q.claim(&key));
  q.expireBefore(11);
  auto adopted = f.payload();
  auto retired = adopted;
  auto take = std::async(std::launch::async, [&] {
    return q.takeClaimed(&key, [&](Buffer &b) {
      f.check(b);
      assert(b.handle == 42);
      adopted.swap(b.payload); // Zero-copy adoption returns the retired slot.
    });
  });
  assert(take.wait_for(std::chrono::seconds(0)) == std::future_status::timeout);
  gate.allow_io.set_value();
  checkpoint(gate.read_event);
  assert(take.wait_for(std::chrono::seconds(0)) == std::future_status::timeout);
  gate.allow_return.set_value();
  assert(take.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  assert(take.get());
  assert(adopted != retired);
  assert(q.stats().used == 1 && q.stats().discarded == 0 &&
         q.stats().joined == 1);
  assert(q.submit(&key, 11, [&](Buffer &b) {
    assert(b.payload == retired);
    b.offset = f.offsets[1];
    b.gated = false;
  }));
  completed(q, 2);
  assert(q.take(&key, [&](Buffer &b) { f.check(b); }));
}

static void queuedExpiryAndGenerationReuse(Fixture &f) {
  int keys[3]{};
  Gate gate;
  const unsigned before = f.calls;
  Queue q(
    {{f.payload()}, {f.payload()}}, 1,
    [&](Buffer &b, const Queue::CancelCheck &) { return f.run(b, gate); });
  assert(q.submit(keys, 20, [&](Buffer &b) {
    b.offset = f.offsets[0];
    b.gated = true;
  }));
  checkpoint(gate.entered_event);
  assert(q.submit(keys + 1, 20, [&](Buffer &b) {
    b.offset = f.offsets[0];
    b.gated = false;
  }));
  q.expireBefore(21);
  // Queued stale work is removed; only the active old slot remains owned.
  assert(q.submit(keys + 2, 21, [&](Buffer &b) {
    b.offset = f.offsets[1];
    b.gated = false;
  }));
  gate.allow_io.set_value();
  checkpoint(gate.read_event);
  gate.allow_return.set_value();
  completed(q, 2);
  assert(f.calls == before + 2);
  assert(!q.take(keys + 1, [](Buffer &) { assert(false); }));
  assert(q.take(keys + 2, [&](Buffer &b) { f.check(b); }));
  assert(q.stats().submitted == 3 && q.stats().discarded == 2 &&
         q.stats().cancelled == 0);
}

static void eofIsNeverAdopted(Fixture &f) {
  int key{};
  Gate gate;
  auto original = f.payload();
  Queue q({{original}}, 1, [&](Buffer &b, const Queue::CancelCheck &) {
    return f.run(b, gate);
  });
  assert(q.submit(&key, 30, [&](Buffer &b) {
    b.offset = f.file.size() - 16;
    b.gated = true;
  }));
  checkpoint(gate.entered_event);
  assert(q.claim(&key));
  q.expireBefore(31);
  gate.allow_io.set_value();
  checkpoint(gate.read_event);
  for (const auto &p : original->parts)
    for (auto v : p)
      assert(v == 0xA5);
  gate.allow_return.set_value();
  assert(!q.takeClaimed(&key, [](Buffer &) { assert(false); }));
  assert(q.stats().failed == 1 && q.stats().used == 0);
  // The existing demand fallback also reports the short file, not success.
  Buffer fallback{f.payload(), f.file.size() - 16};
  assert(f.read(fallback) == ENODATA);
  assert(q.submit(&key, 31, [&](Buffer &b) {
    assert(b.payload == original);
    b.offset = f.offsets[1];
    b.gated = false;
  }));
  completed(q, 2);
  assert(q.take(&key, [&](Buffer &b) { f.check(b); }));
}

/** Queued demand can finish while the speculative reader is still occupied.
 * The existing demand slot owns the payload, including EOF/error cleanup. */
static void queuedDemandRead(Fixture &f, unsigned failure) {
  int keys[3]{};
  Gate gate;
  auto spare = f.payload();
  const unsigned before = f.calls;
  Queue q(
    {{f.payload()}, {spare}}, 1,
    [&](Buffer &b, const Queue::CancelCheck &) { return f.run(b, gate); });
  assert(q.submit(keys, 40, [&](Buffer &b) {
    b.offset = f.offsets[0];
    b.gated = true;
  }));
  checkpoint(gate.entered_event);
  const uint64_t offset = failure == 1 ? f.file.size() - 16 : f.offsets[1];
  assert(q.submit(keys + 1, 40, [&](Buffer &b) { b.offset = offset; }));
  assert(!q.claim(keys + 1, true));
  assert(!q.submit(keys + 1, 40, [](Buffer &) { assert(false); }));
  q.expireBefore(41);
  Buffer demand{f.payload(), offset, false, 0, 42};
  const auto retired = demand.payload;
  nntrainer::HtpExpertReadPool pool(4);
  std::atomic<unsigned> joined{0};
  bool failed = false, registered = false;
  try {
    pool.run(4, [&](size_t job) {
      ++joined;
      if (job != 0)
        return;
      if (failure == 2) {
        unsigned char byte = 0;
        const nntrainer::WeightReadSlice part{&byte, 1};
        demand.rc = nntrainer::weightPreadScatter(-1, &part, 1, offset);
        assert(demand.rc == EBADF);
      } else {
        f.read(demand);
      }
      if (demand.rc)
        throw std::runtime_error("demand read failure");
    });
    registered = true;
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(joined == 4 && failed == (failure != 0) && registered == !failure);
  assert(demand.payload == retired && demand.handle == 42);
  assert(!q.takeClaimed(keys + 1, [](Buffer &) { assert(false); }));
  assert(q.stats().completed == 0 && q.stats().handed_off == 1);
  for (const auto &part : spare->parts)
    for (auto byte : part)
      assert(byte == 0xA5); // The removed guess performed no I/O.
  gate.allow_io.set_value();
  checkpoint(gate.read_event);
  gate.allow_return.set_value();
  completed(q, 1);
  assert(f.calls == before + (failure == 2 ? 1u : 2u));
  assert(q.submit(keys + 2, 41, [&](Buffer &b) {
    b.offset = f.offsets[1];
    b.gated = false;
  }));
  completed(q, 2);
  assert(q.take(keys + 2, [&](Buffer &b) { f.check(b); }));
  pool.run(1, [&](size_t) {
    Buffer retry{f.payload(), f.offsets[1]};
    assert(f.read(retry) == 0);
  });
}

} // namespace

int main(int argc, char **argv) {
  assert(argc == 2); // The runner supplies its disposable fixture directory.
  for (const auto &shape : {std::array<unsigned, 3>{2048, 1536, 2048},
                            std::array<unsigned, 3>{2816, 704, 2816}}) {
    {
      Fixture f(argv[1], shape[0], shape[1], shape[2]);
      splitPairAndMetadata(f);
      prefillBatchJoin(f);
      retiredReadOwnsSlot(f);
      claimedReadSurvivesExpiry(f);
      queuedExpiryAndGenerationReuse(f);
      eofIsNeverAdopted(f);
      for (unsigned failure = 0; failure < 3; ++failure)
        queuedDemandRead(f, failure);
    }
    printf("TL2 PAIR K%u/I%u/N%u: native/scatter/EOF, batch join, "
           "retired/claimed ownership, epoch reuse/adopt, no page cache PASS\n",
           shape[0], shape[1], shape[2]);
  }
}
#else
int main() { puts("TL2 PAIR READ SKIP: Linux/Android direct I/O is required"); }
#endif
