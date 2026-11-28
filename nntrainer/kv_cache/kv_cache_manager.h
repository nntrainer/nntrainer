// SPDX-License-Identifier: Apache-2.0
/** @file kv_cache_manager.h
 * @brief Internal synchronous CPU cache, single session and single live step.
 * @author Codex (OpenAI)
 * @bug Sessions are thread confined; asynchronous device access is unsupported.
 */
#ifndef NNTRAINER_CORE_KV_CACHE_MANAGER_H
#define NNTRAINER_CORE_KV_CACHE_MANAGER_H

#include <kv_cache/kv_cache_spec.h>
#include <kv_cache/kv_storage.h>

#include <memory>
#include <vector>

namespace nntrainer::kv {

struct SessionState;
struct ViewEpoch;
class KvStepScope;
class KvLayerAccess;

/** Byte range with strong allocation ownership and a checked generation.
 * Read data is const; writableData() additionally checks the live step.
 * Beginning another step invalidates earlier views before ring rows can change.
 * A raw pointer must not be retained past the synchronous operation.
 */
class KvView {
public:
  const void *data() const;
  void *writableData() const;
  size_t offsetBytes() const { return offset_; }
  size_t sizeBytes() const { return bytes_; }
  uint64_t bufferId() const { return storage_ ? storage_->identity() : 0; }

private:
  friend class KvCacheManager;
  void validate(bool write) const;
  std::shared_ptr<KvStorage> storage_;
  std::shared_ptr<ViewEpoch> epoch_;
  uint64_t generation_ = 0;
  uint64_t step_ = 0;
  size_t offset_ = 0;
  size_t bytes_ = 0;
};

/** Destruction aborts an uncommitted step, including on graph exceptions. */
class StepLease {
public:
  StepLease(StepLease &&other) noexcept;
  StepLease &operator=(StepLease &&other) noexcept;
  ~StepLease();
  StepLease(const StepLease &) = delete;
  StepLease &operator=(const StepLease &) = delete;
  size_t from() const { return from_; }
  size_t to() const { return to_; }

private:
  friend class KvCacheManager;
  StepLease(std::shared_ptr<SessionState> state, uint64_t id, size_t from,
            size_t to);
  void abort() noexcept;
  std::shared_ptr<SessionState> state_;
  uint64_t id_ = 0;
  size_t from_ = 0;
  size_t to_ = 0;
};

struct CacheStats {
  size_t backing_bytes = 0;
  size_t owner_layers = 0;
};

/** Geometry and publication only; attention/RoPE stay in the model/layer.
 * Thread confined. No asynchronous backend completion is advertised.
 */
class KvCacheManager {
public:
  KvCacheManager();
  ~KvCacheManager();
  KvCacheManager(const KvCacheManager &) = delete;
  KvCacheManager &operator=(const KvCacheManager &) = delete;

  void initialize(
    const CacheSpec &spec,
    std::shared_ptr<MemAllocator> allocator = std::make_shared<MemAllocator>());
  StepLease beginStep(size_t token_count);
  std::vector<KvView> writeKey(const StepLease &, size_t layer,
                               size_t batch = 0);
  std::vector<KvView> writeValue(const StepLease &, size_t layer,
                                 size_t batch = 0);
  std::vector<KvView> readKey(const StepLease &, size_t layer,
                              size_t batch = 0);
  std::vector<KvView> readValue(const StepLease &, size_t layer,
                                size_t batch = 0);
  void markReady(const StepLease &, size_t owner_layer);
  /** Bind checked accesses to one synchronous graph invocation. */
  std::unique_ptr<KvStepScope> bindStep(const StepLease &);
  void commit(StepLease &&);
  void abort(StepLease &&);
  void reset();
  /** Compatibility restore/truncate. Caller supplies matching cache bytes. */
  void setPosition(size_t position);
  size_t committedLength() const;
  bool valid() const;
  bool hasActiveStep() const;
  CacheStats stats() const;
  const LayerSpec &layerSpec(size_t layer) const;
  /** Persistent Tensor bridge. Its shared_ptr pins the allocation.
   * This compatibility bridge does not enforce step/generation checks on raw
   * Tensor access; the caller must confine mutable use to its synchronous step.
   */
  std::shared_ptr<MemoryData> keyMemory(size_t layer);
  std::shared_ptr<MemoryData> valueMemory(size_t layer);

private:
  friend class KvStepScope;
  friend class KvLayerAccess;
  void checkStep(const StepLease &) const;
  std::vector<KvView> views(const StepLease &, size_t layer, size_t batch,
                            bool key, bool write);
  std::shared_ptr<SessionState> state_;
};

/** Checked linear CPU access, held across the actual producer and consumer.
 * Tensor adapters must validate their exact raw slice before executing math.
 * This does not make arbitrary escaped raw Tensor pointers generation-safe.
 */
class KvLayerAccess {
public:
  void validateWrite(const void *key, const void *value, size_t bytes) const;
  void producerReady();
  void validateRead(const void *key, const void *value, size_t bytes) const;

private:
  friend class KvStepScope;
  KvLayerAccess(KvStepScope &, size_t layer);
  KvStepScope &checkedScope() const;
  KvStepScope *scope_;
  std::weak_ptr<char> scope_lifetime_;
  size_t layer_;
  std::vector<KvView> key_write_, value_write_, key_read_, value_read_;
};

/** Thread-confined bridge shared by graph and MHA DSOs through libnntrainer.
 * No nested graph execution, asynchronous backend or ring layout is supported.
 * The scope must be destroyed before its lease and cache manager.
 */
class KvStepScope {
public:
  ~KvStepScope();
  KvStepScope(const KvStepScope &) = delete;
  KvStepScope &operator=(const KvStepScope &) = delete;
  static std::unique_ptr<KvLayerAccess> access(const void *key_base,
                                               const void *value_base,
                                               size_t batch, size_t from,
                                               size_t to);

private:
  friend class KvCacheManager;
  friend class KvLayerAccess;
  KvStepScope(KvCacheManager &, const StepLease &);
  void validate() const;
  static thread_local KvStepScope *current_;
  KvCacheManager &manager_;
  const StepLease &step_;
  std::shared_ptr<char> lifetime_ = std::make_shared<char>(0);
};

} // namespace nntrainer::kv
#endif
