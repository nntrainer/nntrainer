// SPDX-License-Identifier: Apache-2.0
/** @file prefix_flash_store.h
 * @brief Bounded asynchronous backing for immutable native prefill boundaries.
 */
#ifndef NNTR_PREFIX_FLASH_STORE_H
#define NNTR_PREFIX_FLASH_STORE_H
#include <kv_cache_manager.h>
#include <memory>
#include <string>
namespace causallm {
/** Request-local radix backing, not a persistent cross-process model format.
 * Calibration and token identity stay with the radix frame. The worker only
 * reads owned immutable bytes; it never touches inference tensors or DSP RPCs.
 */
class PrefixFlashStore {
  struct State;
  struct Record;

public:
  using Snapshot = KVCacheManager::PrefixSnapshot;
  struct Stats {
    size_t pending_bytes = 0, peak_pending_bytes = 0, flash_bytes = 0;
    uint64_t submitted = 0, written = 0, skipped = 0, failed = 0;
    uint64_t evicted = 0, flash_hits = 0, staging_hits = 0;
    double staging_ms = 0, write_ms = 0, read_ms = 0;
  };
  class Entry {
  public:
    Entry() = default;
    ~Entry();
    Entry(const Entry &) = delete;
    Entry &operator=(const Entry &) = delete;
    bool available() const;
    std::shared_ptr<const Snapshot> load() const;
    /** Load a blob submitted with submitBlob(). */
    std::string loadBlob() const;

  private:
    friend class PrefixFlashStore;
    std::weak_ptr<State> state_;
    std::shared_ptr<Record> record_;
  };
  PrefixFlashStore(const std::string &directory, size_t flash_budget,
                   size_t staging_budget);
  ~PrefixFlashStore();
  PrefixFlashStore(const PrefixFlashStore &) = delete;
  PrefixFlashStore &operator=(const PrefixFlashStore &) = delete;
  /** Single inference producer. Reject before copying if budgets are full. */
  bool canStage(size_t payload_bytes);
  std::shared_ptr<Entry> submit(std::shared_ptr<const Snapshot> snapshot,
                                double staging_ms = 0);
  /** Back an immutable accelerator checkpoint using the same worker and
   * capacity limits as native prefill frames. */
  std::shared_ptr<Entry> submitBlob(const std::string &bytes,
                                    double staging_ms = 0);
  /** Synchronous payload codec for persistent baked assets. Views are valid
   * only until save returns. Layout metadata is supplied by the asset format.
   */
  static void saveSnapshot(const std::string &path, const Snapshot &snapshot);
  static std::shared_ptr<const Snapshot> loadSnapshot(const std::string &path,
                                                      const Snapshot &layout);
  Stats stats() const;
  /** Explicit diagnostic/request-boundary drain; inference never calls it. */
  void drain();

private:
  std::shared_ptr<State> state_;
  struct Worker;
  std::unique_ptr<Worker> worker_;
};
} // namespace causallm
#endif
