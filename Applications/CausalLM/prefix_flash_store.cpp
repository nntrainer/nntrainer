// SPDX-License-Identifier: Apache-2.0
/** @file prefix_flash_store.cpp
 * @brief Asynchronous, bounded file backing of immutable prefill KV pages.
 */
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <list>
#include <mutex>
#include <prefix_flash_store.h>
#include <thread>
#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif
namespace causallm {
namespace {
using Clock = std::chrono::steady_clock;
constexpr uint64_t magic = 0x31464b5652544e4eULL;
constexpr size_t overhead = 6 * sizeof(uint64_t);
size_t bytesOf(const PrefixFlashStore::Snapshot &s) {
  size_t n = 0;
  for (const auto &l : s.layers)
    for (const auto &p : l.pages) {
      if (!p || p->key_view || p->value_view ||
          p->keySize() != p->valueSize() || p->keySize() > (SIZE_MAX - n) / 2)
        throw std::invalid_argument("Flash staging requires owned KV pages");
      n += p->keySize() * 2;
    }
  return n;
}
void hashBytes(uint64_t &hash, const unsigned char *p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ULL;
  }
}
void writeSnapshot(const std::string &path, const PrefixFlashStore::Snapshot &s,
                   size_t bytes) {
  std::ofstream f(path + ".tmp", std::ios::binary | std::ios::trunc);
  const uint64_t header[] = {magic, s.position, s.element_bytes,
                             s.layers.size(), bytes};
  f.write(reinterpret_cast<const char *>(header), sizeof(header));
  uint64_t hash = 14695981039346656037ULL;
  for (const auto &l : s.layers)
    for (const auto &p : l.pages)
      for (const auto *v : {&p->key, &p->value}) {
        hashBytes(hash, v->data(), v->size());
        f.write(reinterpret_cast<const char *>(v->data()), v->size());
      }
  f.write(reinterpret_cast<const char *>(&hash), sizeof(hash));
  f.close();
  if (!f)
    throw std::runtime_error("Cannot write prefix Flash backing");
#if defined(__linux__)
  // Complete storage writes on the background worker, then ask the OS to drop
  // clean file-cache pages. Restore tests should not just read our own writes
  // from the page cache. This advisory eviction is not a cold-device guarantee.
  const int fd = ::open((path + ".tmp").c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    throw std::runtime_error("Cannot open prefix backing for synchronization");
  const int rc = ::fdatasync(fd);
  if (rc == 0)
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  ::close(fd);
  if (rc != 0)
    throw std::runtime_error("Cannot synchronize prefix Flash backing");
#endif
  std::filesystem::rename(path + ".tmp", path);
}
std::shared_ptr<const PrefixFlashStore::Snapshot>
readSnapshot(const std::string &path, const PrefixFlashStore::Snapshot &layout,
             size_t bytes) {
  std::ifstream f(path, std::ios::binary);
  uint64_t header[5]{};
  f.read(reinterpret_cast<char *>(header), sizeof(header));
  if (!f || header[0] != magic || header[1] != layout.position ||
      header[2] != layout.element_bytes || header[3] != layout.layers.size() ||
      header[4] != bytes)
    throw std::runtime_error("Invalid prefix Flash header");
  auto s = std::make_shared<PrefixFlashStore::Snapshot>(layout);
  size_t total = 0;
  uint64_t hash = 14695981039346656037ULL;
  for (auto &l : s->layers)
    for (unsigned r = l.base; r < s->position; r += 64u) {
      const size_t n =
        size_t(std::min(64u, s->position - r)) * l.width * s->element_bytes;
      if (n > (bytes - total) / 2)
        throw std::runtime_error("Invalid prefix Flash size");
      auto page = std::make_shared<PrefixFlashStore::Snapshot::Page>();
      for (auto *v : {&page->key, &page->value}) {
        v->resize(n);
        f.read(reinterpret_cast<char *>(v->data()), n);
        if (!f)
          throw std::runtime_error("Truncated prefix Flash page");
        hashBytes(hash, v->data(), n);
      }
      total += n * 2;
      l.pages.push_back(std::move(page));
    }
  uint64_t stored = 0;
  f.read(reinterpret_cast<char *>(&stored), sizeof(stored));
  if (!f || total != bytes || stored != hash || f.peek() != EOF)
    throw std::runtime_error("Corrupt prefix Flash payload");
  return s;
}
void removeFile(const std::string &path) {
  std::error_code error;
  std::filesystem::remove(path, error);
  std::filesystem::remove(path + ".tmp", error);
}
} // namespace
struct PrefixFlashStore::Record {
  enum Status { Pending, Ready, Missing } status = Pending;
  std::string path;
  size_t bytes = 0, readers = 0;
  bool abandoned = false;
  uint64_t used = 0;
  std::shared_ptr<const Snapshot> payload;
  Snapshot layout;
};
struct PrefixFlashStore::State {
  mutable std::mutex mutex;
  std::condition_variable wake, drained;
  bool stop = false;
  size_t flash_budget = 0, staging_budget = 0;
  uint64_t tick = 0, next = 0;
  Stats stats;
  std::string directory;
  std::list<std::shared_ptr<Record>> records;
  std::deque<std::shared_ptr<Record>> queue;
  // Evict ready, unpinned files; queued/in-flight reservations stay charged.
  bool reserve(size_t bytes) {
    if (bytes > staging_budget || bytes > flash_budget ||
        stats.pending_bytes > staging_budget - bytes)
      return false;
    while (stats.flash_bytes > flash_budget - bytes) {
      auto victim = records.end();
      for (auto it = records.begin(); it != records.end(); ++it)
        if ((*it)->status == Record::Ready && !(*it)->readers &&
            (victim == records.end() || (*it)->used < (*victim)->used))
          victim = it;
      if (victim == records.end())
        return false;
      auto &r = **victim;
      r.status = Record::Missing;
      removeFile(r.path);
      stats.flash_bytes -= r.bytes + overhead;
      ++stats.evicted;
      records.erase(victim);
    }
    return true;
  }
};
struct PrefixFlashStore::Worker {
  std::thread thread;
};
PrefixFlashStore::PrefixFlashStore(const std::string &directory,
                                   size_t flash_budget, size_t staging_budget) :
  state_(std::make_shared<State>()), worker_(std::make_unique<Worker>()) {
  if (directory.empty() || flash_budget <= overhead ||
      staging_budget <= overhead)
    throw std::invalid_argument("Nonzero Flash and staging budgets required");
  std::filesystem::create_directories(directory);
  auto &s = *state_;
  s.flash_budget = flash_budget;
  s.staging_budget = staging_budget;
  for (unsigned i = 0; i < 100; ++i) {
    s.directory = (std::filesystem::path(directory) /
                   ("nntr-prefix-" +
                    std::to_string(Clock::now().time_since_epoch().count()) +
                    "-" + std::to_string(i)))
                    .string();
    if (std::filesystem::create_directory(s.directory))
      break;
    if (i == 99)
      throw std::runtime_error(
        "Cannot create private prefix backing directory");
  }
  worker_->thread = std::thread([state = state_] {
    auto &s = *state;
    for (;;) {
      std::shared_ptr<Record> r;
      {
        std::unique_lock<std::mutex> lock(s.mutex);
        s.wake.wait(lock, [&] { return s.stop || !s.queue.empty(); });
        if (s.queue.empty())
          break;
        r = s.queue.front();
        s.queue.pop_front();
      }
      const auto start = Clock::now();
      bool ok = true;
      try {
        writeSnapshot(r->path, *r->payload, r->bytes);
      } catch (...) {
        ok = false;
        removeFile(r->path);
      }
      {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.stats.write_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - start)
            .count();
        s.stats.pending_bytes -= r->bytes + overhead;
        r->payload.reset();
        r->status = ok && !r->abandoned ? Record::Ready : Record::Missing;
        if (ok && !r->abandoned)
          ++s.stats.written;
        else {
          if (!ok)
            ++s.stats.failed;
          removeFile(r->path);
          s.stats.flash_bytes -= r->bytes + overhead;
          s.records.remove(r);
        }
      }
      s.drained.notify_all();
    }
  });
}
PrefixFlashStore::~PrefixFlashStore() {
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->stop = true;
  }
  state_->wake.notify_all();
  worker_->thread.join();
  std::error_code error;
  std::filesystem::remove_all(state_->directory, error);
}
bool PrefixFlashStore::canStage(size_t bytes) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  const bool ok =
    bytes <= SIZE_MAX - overhead && state_->reserve(bytes + overhead);
  if (!ok)
    ++state_->stats.skipped;
  return ok;
}
std::shared_ptr<PrefixFlashStore::Entry>
PrefixFlashStore::submit(std::shared_ptr<const Snapshot> snapshot,
                         double staging_ms) {
  if (!snapshot)
    throw std::invalid_argument("Missing prefix snapshot");
  const auto bytes = bytesOf(*snapshot);
  auto r = std::make_shared<Record>();
  r->bytes = bytes;
  r->layout = *snapshot;
  for (auto &l : r->layout.layers)
    l.pages.clear();
  r->payload = std::move(snapshot);
  auto entry = std::make_shared<Entry>();
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (bytes > SIZE_MAX - overhead || !state_->reserve(bytes + overhead)) {
    ++state_->stats.skipped;
    return {};
  }
  r->path = (std::filesystem::path(state_->directory) /
             (std::to_string(++state_->next) + ".kv"))
              .string();
  r->used = ++state_->tick;
  entry->state_ = state_;
  entry->record_ = r;
  state_->records.push_back(r);
  state_->queue.push_back(r);
  auto &stats = state_->stats;
  stats.pending_bytes += bytes + overhead;
  stats.flash_bytes += bytes + overhead;
  stats.peak_pending_bytes =
    std::max(stats.pending_bytes, stats.peak_pending_bytes);
  ++stats.submitted;
  stats.staging_ms += staging_ms;
  state_->wake.notify_one();
  return entry;
}
PrefixFlashStore::Entry::~Entry() {
  const auto state = state_.lock();
  if (!state || !record_)
    return;
  std::lock_guard<std::mutex> lock(state->mutex);
  record_->abandoned = true;
  if (record_->status == Record::Ready) {
    record_->status = Record::Missing;
    state->stats.flash_bytes -= record_->bytes + overhead;
    removeFile(record_->path);
    state->records.remove(record_);
  }
}
bool PrefixFlashStore::Entry::available() const {
  const auto state = state_.lock();
  if (!state)
    return false;
  std::lock_guard<std::mutex> lock(state->mutex);
  return record_->status != Record::Missing;
}
std::shared_ptr<const PrefixFlashStore::Snapshot>
PrefixFlashStore::Entry::load() const {
  const auto state = state_.lock();
  if (!state)
    return {};
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (record_->status == Record::Missing)
      return {};
    record_->used = ++state->tick;
    if (record_->payload) {
      ++state->stats.staging_hits;
      return record_->payload;
    }
    ++record_->readers;
  }
  const auto start = Clock::now();
  std::shared_ptr<const Snapshot> payload;
  try {
    payload = readSnapshot(record_->path, record_->layout, record_->bytes);
  } catch (...) {
  }
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    --record_->readers;
    state->stats.read_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (payload)
      ++state->stats.flash_hits;
    else if (record_->status != Record::Missing) {
      record_->status = Record::Missing;
      state->stats.flash_bytes -= record_->bytes + overhead;
      ++state->stats.failed;
      removeFile(record_->path);
      state->records.remove(record_);
    }
  }
  return payload;
}
PrefixFlashStore::Stats PrefixFlashStore::stats() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->stats;
}
void PrefixFlashStore::drain() {
  std::unique_lock<std::mutex> lock(state_->mutex);
  state_->drained.wait(lock, [&] { return state_->stats.pending_bytes == 0; });
}
} // namespace causallm
