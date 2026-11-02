// SPDX-License-Identifier: Apache-2.0
/** @file kv_cache_manager.h
 * @brief Internal synchronous CPU cache: paged sessions, one live step.
 * @author Codex (OpenAI)
 * @bug Sessions are thread confined; asynchronous device access is unsupported.
 */
#ifndef NNTRAINER_CORE_KV_CACHE_MANAGER_H
#define NNTRAINER_CORE_KV_CACHE_MANAGER_H

#include <kv_cache/kv_cache_spec.h>
#include <kv_cache/kv_page_pool.h>
#include <kv_cache/kv_prefix_cache.h>
#include <kv_cache/kv_storage.h>

#include <iosfwd>
#include <map>
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
  /** Pages referenced by the active session, including prefix pages it shares
   * with other sessions or the prefix cache; excludes idle or view-only pool
   * pages. Summing it over sessions counts shared pages more than once. */
  size_t backing_bytes = 0;
  size_t owner_layers = 0;
};

/** Identifies one KV session of a manager. */
using SessionId = uint64_t;

/** CPU attention parameters. Queries and results use [token, query_head, dim].
 * RoPE must already have been applied to queries and keys by the producer.
 */
struct AttentionOptions {
  size_t query_heads = 0;
  bool causal = true;
  size_t window = 0; /**< Zero means all available history. */
  float scale = 0;   /**< Zero means 1/sqrt(head_dim). */
  float logit_softcap = 0;
  std::vector<float>
    sinks; /**< Optional denominator-only logit per query head. */
};

/** Cache ownership, publication and direct segmented CPU attention.
 *
 * A manager owns one or more sessions. A session is an independent sequence:
 * its own page table, committed position, step state and token ids. All
 * sessions of a manager share the spec, page pool, optional prefix cache and
 * the graph identity tensors, so a graph bound once can run any session; the
 * selected session receives every step, view and snapshot call. Additional
 * sessions require paged storage. Only the selected session can have a live
 * step, and selection cannot change while one is live.
 *
 * Thread confined. No asynchronous backend completion is advertised.
 */
class KvCacheManager {
public:
  /** The session that initialize() creates and selects. */
  static constexpr SessionId kDefaultSession = 0;

  KvCacheManager();
  ~KvCacheManager();
  KvCacheManager(const KvCacheManager &) = delete;
  KvCacheManager &operator=(const KvCacheManager &) = delete;

  /** page_tokens enables segmented CPU storage allocated as steps advance.
   * A prefix cache requires paged batch-1 geometry and uses its own pool;
   * page_pool, when also given, must be that pool. Replaces every session.
   */
  void initialize(
    const CacheSpec &spec,
    std::shared_ptr<MemAllocator> allocator = std::make_shared<MemAllocator>(),
    std::shared_ptr<KvPagePool> page_pool = nullptr,
    std::shared_ptr<KvPrefixCache> prefix_cache = nullptr);

  /** Create an empty paged session; it does not change the selection. */
  SessionId createSession();
  /** Route subsequent calls to an existing session. Requires no live step. */
  void selectSession(SessionId session);
  /** Release a session's pages. The selected session cannot be destroyed. */
  void destroySession(SessionId session);
  SessionId activeSession() const;
  size_t sessionCount() const;

  /** tokens, when given, holds the ids of the token_count rows this step
   * writes. Steps that carry ids keep the session's committed ids complete,
   * which prefix reuse and publication need; a step without ids ends the
   * known prefix until the session is truncated back into it.
   */
  StepLease beginStep(size_t token_count, const TokenId *tokens = nullptr);
  std::vector<KvView> writeKey(const StepLease &, size_t layer,
                               size_t batch = 0);
  std::vector<KvView> writeValue(const StepLease &, size_t layer,
                                 size_t batch = 0);
  std::vector<KvView> readKey(const StepLease &, size_t layer,
                              size_t batch = 0);
  std::vector<KvView> readValue(const StepLease &, size_t layer,
                                size_t batch = 0);
  /** Append already transformed token-major K/V in the cache's encoding.
   * bytes is the size of each plane for this batch and step, not both planes.
   * Input buffers must not overlap cache storage.
   * Call markReady after all batches have been produced.
   */
  void append(const StepLease &, size_t layer, const void *keys,
              const void *values, size_t bytes, size_t batch = 0);
  void markReady(const StepLease &, size_t owner_layer);
  /** Consume pages directly with online softmax; never gather cache history.
   * Queries cover exactly this step. Reads require markReady(), including
   * aliases. The returned FP32 output has the same layout and size as queries.
   */
  std::vector<float> attention(const StepLease &, size_t layer,
                               const std::vector<float> &queries,
                               const AttentionOptions &, size_t batch = 0);
  /** Bind checked accesses to one synchronous graph invocation. */
  std::unique_ptr<KvStepScope> bindStep(const StepLease &);
  /** Publish the step. With a prefix cache, every newly completed full page
   * whose token ids are known is published as a shared block.
   */
  void commit(StepLease &&);
  void abort(StepLease &&);
  void reset();
  /** Compatibility restore/truncate. Caller supplies matching cache bytes. */
  void setPosition(size_t position);
  /** Reuse as much of tokens[0, count) as is already computed, then continue
   * from the returned position (always <= count).
   *
   * The session keeps its own committed rows that match the leading tokens.
   * If the prefix cache holds a longer chain of full blocks, the session
   * instead adopts those shared pages. The remaining rows are released, as in
   * setPosition(). Pass one token fewer than the prompt to leave the last
   * prompt token to be computed for its logits. Requires paged storage.
   */
  size_t reusePrefix(const TokenId *tokens, size_t count);
  /** Committed token ids known for [0, knownTokenCount()). */
  std::vector<TokenId> committedTokens() const;
  size_t knownTokenCount() const;
  std::shared_ptr<KvPrefixCache> prefixCache() const;
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
  bool paged() const;
  /** Page width in tokens of the initialized spec (0 for continuous). */
  size_t pageTokens() const;
  /** One-row graph identity tensors, never used to hold cache history. */
  std::shared_ptr<MemoryData> graphKeyMemory(size_t layer);
  std::shared_ptr<MemoryData> graphValueMemory(size_t layer);
  void savePlane(std::ostream &, size_t layer, bool key, size_t length) const;
  /** Restore layer-major K/V planes atomically, preserving graph identities.
   * Snapshots carry no token ids, so restored rows cannot be reused or
   * published as prefixes.
   */
  void restorePlanes(std::istream &, size_t length);

private:
  friend class KvStepScope;
  friend class KvLayerAccess;
  void checkStep(const StepLease &) const;
  void requireIdle() const;
  /** Insert newly completed, id-known full pages into the prefix cache. */
  void publishPrefix() noexcept;
  std::vector<KvView> views(const StepLease &, size_t layer, size_t batch,
                            bool key, bool write);
  std::shared_ptr<SessionState> state_; /**< The selected session. */
  std::map<SessionId, std::shared_ptr<SessionState>> sessions_;
  SessionId active_ = kDefaultSession;
  SessionId next_session_ = kDefaultSession + 1;
};

/** Checked CPU access, held across the actual producer and consumer.
 * Tensor adapters must validate their exact raw slice before executing math.
 * This does not make arbitrary escaped raw Tensor pointers generation-safe.
 */
class KvLayerAccess {
public:
  bool paged() const;
  bool alias() const;
  const LayerSpec &spec() const;
  void append(const void *key, const void *value, size_t bytes);
  std::vector<float> attention(const std::vector<float> &,
                               const AttentionOptions &);
  /** Page width in tokens (0 for continuous storage). */
  size_t pageTokens() const;
  /** Ready-checked read views in token order (owner-resolved). Paged storage
   * yields one view per page; a page base is data() - offsetBytes().
   */
  std::vector<KvView> readKeyViews(size_t batch = 0) const;
  std::vector<KvView> readValueViews(size_t batch = 0) const;
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
  static bool pagedActive();
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
