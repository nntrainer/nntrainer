// SPDX-License-Identifier: Apache-2.0
/** @file unittest_kv_cache_core.cpp
 * @brief CPU ownership, checked geometry and transactional KV publication.
 * @author Codex (OpenAI)
 * @bug Covers synchronous CPU storage; device integration is not exercised.
 */
#include <fp16.h>
#include <gtest/gtest.h>
#include <kv_cache/kv_cache_manager.h>
#include <tensor.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

using namespace nntrainer::kv;

namespace {

CacheSpec spec(size_t capacity = 8, size_t max_tokens = 8) {
  LayerSpec layer;
  layer.num_kv_heads = 2;
  layer.head_dim = 2;
  layer.capacity = capacity;
  layer.max_tokens = max_tokens;
  layer.encoding = Encoding::Float32;
  layer.storage_type = StorageType::Float32;
  return {{layer}};
}

class CountingAllocator : public nntrainer::MemAllocator {
public:
  size_t allocations = 0;
  size_t frees = 0;
  size_t fail_at = std::numeric_limits<size_t>::max();
  void alloc(void **ptr, size_t bytes, size_t alignment) override {
    if (allocations == fail_at)
      throw std::bad_alloc();
    ++allocations;
    nntrainer::MemAllocator::alloc(ptr, bytes, alignment);
  }
  void free(void *ptr) override {
    ++frees;
    nntrainer::MemAllocator::free(ptr);
  }
};

void finish(KvCacheManager &manager, StepLease &&step) {
  manager.markReady(step, 0);
  manager.commit(std::move(step));
}

} // namespace

TEST(KvCacheCore, validateBeforeAllocating) {
  auto allocator = std::make_shared<CountingAllocator>();
  KvCacheManager manager;
  auto s = spec();
  s.layers.push_back(s.layers[0]);
  s.layers[1].head_dim = 0;
  EXPECT_THROW(manager.initialize(s, allocator), std::invalid_argument);
  EXPECT_EQ(allocator->allocations, 0u);
  s = spec();
  s.layers[0].batch = std::numeric_limits<size_t>::max();
  EXPECT_THROW(manager.initialize(s, allocator), std::overflow_error);
  s = spec();
  s.layers[0].encoding = Encoding::Float16;
  EXPECT_THROW(manager.initialize(s, allocator), std::invalid_argument);
  EXPECT_THROW(checkedAdd(std::numeric_limits<size_t>::max(), 1),
               std::overflow_error);
  EXPECT_EQ(allocator->allocations, 0u);
}

TEST(KvCacheCore, aliasGeometryValidatedBeforeAllocation) {
  auto allocator = std::make_shared<CountingAllocator>();
  KvCacheManager manager;
  auto s = spec();
  s.layers.push_back(s.layers[0]);
  s.layers[1].source = 0;
  s.layers[1].capacity = 4;
  EXPECT_THROW(manager.initialize(s, allocator), std::invalid_argument);
  s.layers[1] = s.layers[0];
  s.layers[1].source = 1;
  EXPECT_THROW(manager.initialize(s, allocator), std::invalid_argument);
  s.layers[1].source = -2;
  EXPECT_THROW(manager.initialize(s, allocator), std::invalid_argument);
  EXPECT_EQ(allocator->allocations, 0u);
}

TEST(KvCacheCore, aliasOwnsNoExtraStorageAndCannotWrite) {
  auto allocator = std::make_shared<CountingAllocator>();
  KvCacheManager manager;
  auto s = spec();
  s.layers.push_back(s.layers[0]);
  s.layers[1].source = 0;
  s.layers.push_back(s.layers[1]);
  s.layers[2].source = 1;
  manager.initialize(s, allocator);
  EXPECT_EQ(allocator->allocations, 2u);
  EXPECT_EQ(manager.stats().backing_bytes, 2 * s.layers[0].planeBytes());
  EXPECT_EQ(manager.stats().owner_layers, 1u);
  EXPECT_EQ(manager.keyMemory(0), manager.keyMemory(2));
  EXPECT_EQ(manager.valueMemory(0), manager.valueMemory(1));
  auto step = manager.beginStep(2);
  EXPECT_THROW(manager.writeKey(step, 1), std::out_of_range);
  EXPECT_THROW(manager.readKey(step, 2), std::logic_error);
  manager.markReady(step, 0);
  EXPECT_EQ(manager.readKey(step, 2)[0].data(),
            manager.writeKey(step, 0)[0].data());
  manager.commit(std::move(step));
}

TEST(KvCacheCore, tensorPinsAllocationAfterManagerAndAllocatorGoAway) {
  auto allocator = std::make_shared<CountingAllocator>();
  std::weak_ptr<CountingAllocator> weak = allocator;
  nntrainer::Tensor retained;
  {
    KvCacheManager manager;
    manager.initialize(spec(), allocator);
    retained = nntrainer::Tensor(ml::train::TensorDim({1, 1, 8, 4}), false);
    retained.setData(manager.keyMemory(0), 0, false);
    retained.getData<float>()[0] = 17.0f;
  }
  EXPECT_EQ(allocator->frees, 1u); // V died; Tensor still owns K.
  allocator.reset();
  ASSERT_FALSE(weak.expired());
  EXPECT_EQ(retained.getData<float>()[0], 17.0f);
  retained = nntrainer::Tensor();
  EXPECT_TRUE(weak.expired());
}

TEST(KvCacheCore,
     failedReinitializePreservesOldSessionAndReleasesPartialStorage) {
  auto allocator = std::make_shared<CountingAllocator>();
  KvCacheManager manager;
  manager.initialize(spec(), allocator);
  auto original = manager.keyMemory(0);
  manager.setPosition(3);
  allocator->fail_at = 3; // New K succeeds; new V fails.
  EXPECT_THROW(manager.initialize(spec(), allocator), std::bad_alloc);
  EXPECT_EQ(manager.committedLength(), 3u);
  EXPECT_EQ(manager.keyMemory(0), original);
  EXPECT_EQ(allocator->frees, 1u);
}

TEST(KvCacheCore, nullAllocationRejected) {
  class NullAllocator : public nntrainer::MemAllocator {
  public:
    void alloc(void **ptr, size_t, size_t) override { *ptr = nullptr; }
  };
  KvCacheManager manager;
  EXPECT_THROW(manager.initialize(spec(), std::make_shared<NullAllocator>()),
               std::bad_alloc);
}

TEST(KvCacheCore, deviceMemoryRejectedByCpuAdapter) {
  class DeviceAllocator : public CountingAllocator {
  public:
    bool isDeviceVisible() const override { return true; }
  };
  auto allocator = std::make_shared<DeviceAllocator>();
  KvCacheManager manager;
  EXPECT_THROW(manager.initialize(spec(), allocator), std::invalid_argument);
  EXPECT_EQ(allocator->allocations, 0u);
}

TEST(KvCacheCore, graphScopeChecksRealSlicesAndProducerCompletion) {
  KvCacheManager manager;
  auto s = spec();
  s.layers.push_back(s.layers[0]);
  manager.initialize(s);
  auto step = manager.beginStep(2);
  auto scope = manager.bindStep(step);
  auto *key = manager.keyMemory(0)->getAddr<void>();
  auto *value = manager.valueMemory(0)->getAddr<void>();
  auto access = KvStepScope::access(key, value, 0, 0, 2);
  ASSERT_NE(access, nullptr);
  const size_t bytes = 2 * s.layers[0].rowBytes();
  access->validateWrite(key, value, bytes);
  EXPECT_THROW(access->validateWrite(value, key, bytes), std::logic_error);
  EXPECT_THROW(access->validateRead(key, value, bytes), std::logic_error);
  std::memset(key, 0, bytes);
  std::memset(value, 0, bytes);
  access->producerReady();
  access->validateRead(key, value, bytes);
  EXPECT_THROW(manager.commit(std::move(step)), std::logic_error);
  EXPECT_THROW(KvStepScope::access(key, value, 0, 1, 2), std::logic_error);
  EXPECT_THROW(KvStepScope::access(value, key, 0, 0, 2), std::logic_error);
  auto other =
    KvStepScope::access(manager.keyMemory(1)->getAddr<void>(),
                        manager.valueMemory(1)->getAddr<void>(), 0, 0, 2);
  other->producerReady();
  manager.commit(std::move(step));
  EXPECT_THROW(access->validateRead(key, value, bytes), std::logic_error);
}

TEST(KvCacheCore, graphScopeRejectsStaleAccessAfterAbort) {
  KvCacheManager manager;
  manager.initialize(spec());
  auto step = manager.beginStep(1);
  auto scope = manager.bindStep(step);
  auto access =
    KvStepScope::access(manager.keyMemory(0)->getAddr<void>(),
                        manager.valueMemory(0)->getAddr<void>(), 0, 0, 1);
  manager.abort(std::move(step));
  EXPECT_THROW(access->producerReady(), std::logic_error);
  EXPECT_EQ(manager.committedLength(), 0u);
}

TEST(KvCacheCore, layerAccessCannotOutliveItsGraphScope) {
  KvCacheManager manager;
  manager.initialize(spec());
  auto step = manager.beginStep(1);
  auto scope = manager.bindStep(step);
  auto access =
    KvStepScope::access(manager.keyMemory(0)->getAddr<void>(),
                        manager.valueMemory(0)->getAddr<void>(), 0, 0, 1);
  scope.reset();
  auto next_scope = manager.bindStep(step);
  EXPECT_THROW(access->producerReady(), std::logic_error);
}

TEST(KvCacheCore, graphScopeUsesDeclaredHybridProducersAndRejectsRing) {
  KvCacheManager manager;
  auto s = spec();
  s.layers.push_back(s.layers[0]);
  s.layers[1].has_producer = false;
  manager.initialize(s);
  {
    auto step = manager.beginStep(1);
    auto scope = manager.bindStep(step);
    auto access =
      KvStepScope::access(manager.keyMemory(0)->getAddr<void>(),
                          manager.valueMemory(0)->getAddr<void>(), 0, 0, 1);
    EXPECT_THROW(KvStepScope::access(manager.keyMemory(1)->getAddr<void>(),
                                     manager.valueMemory(1)->getAddr<void>(), 0,
                                     0, 1),
                 std::out_of_range);
    access->producerReady();
    manager.commit(std::move(step));
  }
  manager.initialize(spec(4, 8));
  auto step = manager.beginStep(1);
  EXPECT_THROW(manager.bindStep(step), std::invalid_argument);
}

TEST(KvCacheCore, publishOnlyAfterAllOwnersAreReady) {
  KvCacheManager manager;
  auto s = spec();
  s.layers.push_back(s.layers[0]);
  manager.initialize(s);
  auto step = manager.beginStep(3);
  EXPECT_EQ(step.from(), 0u);
  EXPECT_EQ(step.to(), 3u);
  EXPECT_THROW(manager.commit(std::move(step)), std::logic_error);
  manager.markReady(step, 0);
  EXPECT_THROW(manager.commit(std::move(step)), std::logic_error);
  EXPECT_EQ(manager.committedLength(), 0u);
  manager.markReady(step, 1);
  manager.commit(std::move(step));
  EXPECT_EQ(manager.committedLength(), 3u);
  EXPECT_THROW(manager.commit(std::move(step)), std::logic_error);
}

TEST(KvCacheCore, activeStepExcludesResetSeekAndReallocation) {
  KvCacheManager manager;
  manager.initialize(spec());
  auto step = manager.beginStep(1);
  EXPECT_THROW(manager.beginStep(1), std::logic_error);
  EXPECT_THROW(manager.reset(), std::logic_error);
  EXPECT_THROW(manager.setPosition(0), std::logic_error);
  EXPECT_THROW(manager.initialize(spec()), std::logic_error);
  KvCacheManager other;
  other.initialize(spec());
  EXPECT_THROW(other.writeKey(step, 0), std::logic_error);
}

TEST(KvCacheCore, boundariesAndBatchOffsets) {
  KvCacheManager manager;
  auto s = spec();
  s.layers[0].batch = 2;
  manager.initialize(s);
  EXPECT_THROW(manager.beginStep(0), std::out_of_range);
  EXPECT_THROW(manager.beginStep(std::numeric_limits<size_t>::max()),
               std::out_of_range);
  auto step = manager.beginStep(8);
  auto first = manager.writeKey(step, 0, 0)[0];
  auto second = manager.writeKey(step, 0, 1)[0];
  EXPECT_EQ(second.offsetBytes(),
            s.layers[0].capacity * s.layers[0].rowBytes());
  EXPECT_EQ(second.sizeBytes(), first.sizeBytes());
  EXPECT_EQ(reinterpret_cast<uintptr_t>(first.data()) % 64, 0u);
  EXPECT_THROW(manager.writeKey(step, 0, 2), std::out_of_range);
  EXPECT_THROW(manager.writeKey(step, 1), std::out_of_range);
  finish(manager, std::move(step));
  EXPECT_THROW(first.writableData(), std::logic_error);
  EXPECT_THROW(manager.beginStep(1), std::out_of_range);
}

TEST(KvCacheCore, linearAbortPreservesPrefixAndInvalidatesViews) {
  KvCacheManager manager;
  manager.initialize(spec());
  finish(manager, manager.beginStep(3));
  auto step = manager.beginStep(2);
  auto view = manager.writeValue(step, 0)[0];
  std::memset(view.writableData(), 42, view.sizeBytes());
  manager.abort(std::move(step));
  EXPECT_TRUE(manager.valid());
  EXPECT_EQ(manager.committedLength(), 3u);
  EXPECT_THROW(view.data(), std::logic_error);
  auto retry = manager.beginStep(2);
  EXPECT_EQ(retry.from(), 3u);
  finish(manager, std::move(retry));
}

TEST(KvCacheCore, abandonedMovedLeaseAborts) {
  KvCacheManager manager;
  manager.initialize(spec());
  {
    auto step = manager.beginStep(2);
    auto moved = std::move(step);
    EXPECT_THROW(manager.writeKey(step, 0), std::logic_error);
  }
  EXPECT_TRUE(manager.valid());
  EXPECT_EQ(manager.committedLength(), 0u);
  finish(manager, manager.beginStep(1));
}

TEST(KvCacheCore, resetAndReallocateInvalidateOldViewsButRetainAllocation) {
  KvCacheManager manager;
  manager.initialize(spec());
  auto step = manager.beginStep(1);
  auto view = manager.writeKey(step, 0)[0];
  auto memory = manager.keyMemory(0);
  finish(manager, std::move(step));
  manager.reset();
  EXPECT_THROW(view.data(), std::logic_error);
  auto next = manager.beginStep(1);
  auto next_view = manager.writeKey(next, 0)[0];
  finish(manager, std::move(next));
  manager.initialize(spec());
  EXPECT_THROW(next_view.data(), std::logic_error);
  EXPECT_NE(memory, manager.keyMemory(0));
}

TEST(KvCacheCore, ringSeamSegmentsPreserveLogicalReadOrder) {
  KvCacheManager manager;
  manager.initialize(spec(4, 16));
  auto first = manager.beginStep(3);
  auto write = manager.writeKey(first, 0)[0];
  auto ptr = static_cast<float *>(write.writableData());
  for (size_t row = 0; row < 3; ++row)
    ptr[row * 4] = static_cast<float>(row);
  finish(manager, std::move(first));
  auto next = manager.beginStep(3);
  auto segments = manager.writeKey(next, 0);
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0].sizeBytes(), 16u);
  EXPECT_EQ(segments[1].sizeBytes(), 32u);
  size_t value = 3;
  for (auto &segment : segments) {
    auto data = static_cast<float *>(segment.writableData());
    for (size_t row = 0; row < segment.sizeBytes() / 16; ++row)
      data[row * 4] = static_cast<float>(value++);
  }
  manager.markReady(next, 0);
  auto read = manager.readKey(next, 0);
  ASSERT_EQ(read.size(), 2u);
  value = 2;
  for (const auto &segment : read) {
    auto data = static_cast<const float *>(segment.data());
    for (size_t row = 0; row < segment.sizeBytes() / 16; ++row)
      EXPECT_EQ(data[row * 4], static_cast<float>(value++));
  }
  manager.commit(std::move(next));
  EXPECT_EQ(manager.committedLength(), 6u);
}

TEST(KvCacheCore, overwrittenRingAbortRequiresReset) {
  KvCacheManager manager;
  manager.initialize(spec(4, 16));
  EXPECT_THROW(manager.beginStep(5), std::out_of_range);
  finish(manager, manager.beginStep(4));
  {
    auto step = manager.beginStep(1);
    auto view = manager.writeValue(step, 0)[0];
    std::memset(view.writableData(), 7, view.sizeBytes());
  }
  EXPECT_FALSE(manager.valid());
  EXPECT_THROW(manager.beginStep(1), std::logic_error);
  EXPECT_THROW(manager.setPosition(0), std::logic_error);
  manager.reset();
  EXPECT_TRUE(manager.valid());
  EXPECT_EQ(manager.committedLength(), 0u);
  finish(manager, manager.beginStep(1));
}

TEST(KvCacheCore, nextStepInvalidatesPriorRingReadViewsBeforeOverwrite) {
  KvCacheManager manager;
  manager.initialize(spec(4, 16));
  auto step = manager.beginStep(4);
  manager.markReady(step, 0);
  auto previous = manager.readKey(step, 0)[0];
  manager.commit(std::move(step));
  EXPECT_NO_THROW(previous.data());
  auto next = manager.beginStep(1);
  EXPECT_THROW(previous.data(), std::logic_error);
  auto write = manager.writeKey(next, 0)[0];
  std::memset(write.writableData(), 42, write.sizeBytes());
  EXPECT_THROW(previous.data(), std::logic_error);
  finish(manager, std::move(next));
}

TEST(KvCacheCore, wrapperRebindingCannotChangeOwnedAllocationOrCoreViews) {
  class RecordingAllocator : public nntrainer::MemAllocator {
  public:
    void *allocated = nullptr;
    void *released = nullptr;
    void alloc(void **ptr, size_t bytes, size_t alignment) override {
      nntrainer::MemAllocator::alloc(ptr, bytes, alignment);
      allocated = *ptr;
    }
    void free(void *ptr) override {
      released = ptr;
      nntrainer::MemAllocator::free(ptr);
    }
  };
  auto allocator = std::make_shared<RecordingAllocator>();
  {
    auto storage = KvStorage::create(16, allocator);
    auto memory = storage->memory();
    memory->setAddr(nullptr);
    EXPECT_EQ(storage->data(), allocator->allocated);
  }
  EXPECT_EQ(allocator->released, allocator->allocated);

  KvCacheManager manager;
  manager.initialize(spec());
  auto step = manager.beginStep(1);
  auto view = manager.writeKey(step, 0)[0];
  const void *base = view.data();
  manager.keyMemory(0)->setAddr(nullptr);
  EXPECT_EQ(view.data(), base);
  EXPECT_EQ(view.writableData(), base);
}

TEST(KvCacheCore, emptyOrMovedViewAccessIsRejected) {
  KvView empty;
  EXPECT_EQ(empty.bufferId(), 0u);
  EXPECT_THROW(empty.data(), std::logic_error);
  EXPECT_THROW(empty.writableData(), std::logic_error);
  KvCacheManager manager;
  manager.initialize(spec());
  auto step = manager.beginStep(1);
  auto view = manager.writeKey(step, 0)[0];
  auto moved = std::move(view);
  EXPECT_NE(moved.bufferId(), 0u);
  EXPECT_EQ(view.bufferId(), 0u);
  EXPECT_THROW(view.data(), std::logic_error);
}

TEST(KvCacheCore, pagedViewsAllocateOnDemandAndPreserveTokenOrder) {
  auto allocator = std::make_shared<CountingAllocator>();
  auto s = spec(8, 8);
  s.page_tokens = 2;
  s.layers.push_back(s.layers[0]);
  s.layers[1].source = 0;
  KvCacheManager manager;
  manager.initialize(s, allocator);
  EXPECT_EQ(allocator->allocations, 0u);
  auto step = manager.beginStep(3);
  auto write = manager.writeKey(step, 0);
  ASSERT_EQ(write.size(), 2u);
  EXPECT_EQ(allocator->allocations, 4u);
  size_t token = 0;
  for (auto &view : write) {
    auto data = static_cast<float *>(view.writableData());
    for (size_t row = 0; row < view.sizeBytes() / 16; ++row)
      data[row * 4] = static_cast<float>(++token);
  }
  manager.markReady(step, 0);
  auto read = manager.readKey(step, 1);
  ASSERT_EQ(read.size(), 2u);
  token = 0;
  for (auto &view : read) {
    auto data = static_cast<const float *>(view.data());
    for (size_t row = 0; row < view.sizeBytes() / 16; ++row)
      EXPECT_EQ(data[row * 4], static_cast<float>(++token));
  }
  EXPECT_THROW(manager.keyMemory(0), std::logic_error);
  EXPECT_NO_THROW(manager.bindStep(step));
  manager.commit(std::move(step));
  auto next = manager.beginStep(2);
  EXPECT_EQ(allocator->allocations, 6u);
  EXPECT_THROW(read[0].data(), std::logic_error);
  finish(manager, std::move(next));
  manager.reset();
  EXPECT_EQ(manager.stats().backing_bytes, 0u);
}

TEST(KvCacheCore, pagedAllocationFailureLeavesSessionUnchanged) {
  auto allocator = std::make_shared<CountingAllocator>();
  auto s = spec();
  s.page_tokens = 2;
  KvCacheManager manager;
  manager.initialize(s, allocator);
  allocator->fail_at = 1;
  EXPECT_THROW(manager.beginStep(2), std::bad_alloc);
  EXPECT_EQ(manager.committedLength(), 0u);
  EXPECT_FALSE(manager.hasActiveStep());
  EXPECT_EQ(manager.stats().backing_bytes, 0u);
  EXPECT_EQ(allocator->frees, 0u); // Failed admission leaves an idle pool page.
}

TEST(KvCacheCore, sharedPagePoolPinsViewsAndReusesReleasedPages) {
  auto pool = std::make_shared<KvPagePool>(128);
  auto s = spec();
  s.page_tokens = 2;
  KvCacheManager first, second;
  first.initialize(s, std::make_shared<nntrainer::MemAllocator>(), pool);
  second.initialize(s, std::make_shared<nntrainer::MemAllocator>(), pool);
  auto step = first.beginStep(1);
  auto retained = first.writeKey(step, 0);
  const auto identity = retained[0].bufferId();
  first.abort(std::move(step));
  EXPECT_THROW(second.beginStep(1), std::bad_alloc);
  EXPECT_THROW(retained[0].data(), std::logic_error);
  retained.clear();
  auto next = second.beginStep(1);
  EXPECT_EQ(second.writeKey(next, 0)[0].bufferId(), identity);
  EXPECT_EQ(pool->reservedBytes(), 128u);
  second.abort(std::move(next));
  pool->trim();
  EXPECT_EQ(pool->reservedBytes(), 0u);
}

TEST(KvCacheCore, pagedAttentionMatchesDenseReferenceAcrossStepsAndBatches) {
  auto s = spec(8, 8);
  s.layers[0].batch = 2;
  s.page_tokens = 2;
  s.layers.push_back(s.layers[0]);
  s.layers[1].source = 0;
  KvCacheManager manager;
  manager.initialize(s);
  AttentionOptions options;
  options.query_heads = 4;
  options.scale = 0.7f;
  options.logit_softcap = 1.5f;
  std::vector<float> history_k[2], history_v[2];
  for (size_t count : {3u, 1u, 3u}) {
    auto step = manager.beginStep(count);
    std::vector<float> queries(count * 8);
    for (size_t i = 0; i < queries.size(); ++i)
      queries[i] = std::sin(static_cast<float>(i + step.from()));
    for (size_t b = 0; b < 2; ++b) {
      std::vector<float> k(count * 4), v(count * 4);
      for (size_t i = 0; i < k.size(); ++i) {
        k[i] = std::cos(static_cast<float>(i + step.from() * 4 + b));
        v[i] = static_cast<float>(i + step.from() * 4 + b) / 13;
      }
      manager.append(step, 0, k.data(), v.data(), k.size() * sizeof(float), b);
      history_k[b].insert(history_k[b].end(), k.begin(), k.end());
      history_v[b].insert(history_v[b].end(), v.begin(), v.end());
    }
    EXPECT_THROW(manager.attention(step, 0, queries, options),
                 std::logic_error);
    manager.markReady(step, 0);
    for (bool causal : {true, false})
      for (size_t window : {0u, 2u})
        for (size_t b = 0; b < 2; ++b) {
          options.causal = causal;
          options.window = window;
          auto actual = manager.attention(step, 1, queries, options, b);
          for (size_t q = 0; q < count; ++q) {
            const size_t end = causal ? step.from() + q + 1 : step.to();
            const size_t start = window && end > window ? end - window : 0;
            for (size_t h = 0; h < 4; ++h) {
              std::vector<double> scores(end - start);
              for (size_t t = start; t < end; ++t) {
                double score = 0;
                for (size_t d = 0; d < 2; ++d)
                  score += static_cast<double>(queries[q * 8 + h * 2 + d]) *
                           history_k[b][t * 4 + (h / 2) * 2 + d];
                scores[t - start] =
                  1.5 * std::tanh(score * options.scale / 1.5);
              }
              double maximum = *std::max_element(scores.begin(), scores.end());
              double sum = 0;
              for (auto &score : scores) {
                score = std::exp(score - maximum);
                sum += score;
              }
              for (size_t d = 0; d < 2; ++d) {
                double expected = 0;
                for (size_t t = start; t < end; ++t)
                  expected += scores[t - start] *
                              history_v[b][t * 4 + (h / 2) * 2 + d] / sum;
                EXPECT_NEAR(actual[q * 8 + h * 2 + d], expected, 1e-5);
              }
            }
          }
        }
    finish(manager, std::move(step));
  }
}

TEST(KvCacheCore, pagedAttentionDecodesBinary16WithoutNativeHalf) {
  auto s = spec();
  s.page_tokens = 1;
  s.layers[0].encoding = Encoding::Float16;
  s.layers[0].storage_type = StorageType::UInt16;
  KvCacheManager manager;
  manager.initialize(s);
  auto step = manager.beginStep(2);
  uint16_t k[8] = {}; // Equal scores give uniform attention.
  uint16_t v[8] = {0x3c00, 0x4000, 0x4200, 0x4400,
                   0x4200, 0x4400, 0x4500, 0x4600};
  manager.append(step, 0, k, v, sizeof(k));
  manager.markReady(step, 0);
  AttentionOptions options;
  options.query_heads = 2;
  auto output = manager.attention(step, 0, std::vector<float>(8, 1), options);
  EXPECT_FLOAT_EQ(output[0], 1);
  EXPECT_FLOAT_EQ(output[4], 2);
  EXPECT_FLOAT_EQ(output[5], 3);
  EXPECT_FLOAT_EQ(output[6], 4);
  EXPECT_FLOAT_EQ(output[7], 5);
}

TEST(KvCacheCore,
     pagedAttentionVectorizedMatchesDenseReferenceForStandardHeadDim) {
  const size_t head_dim = 64;
  const size_t num_kv_heads = 2;
  const size_t query_heads = 4;
  const size_t page_tokens = 4;
  const size_t max_tokens = 16;

  CacheSpec s;
  LayerSpec layer;
  layer.num_kv_heads = num_kv_heads;
  layer.head_dim = head_dim;
  layer.capacity = max_tokens;
  layer.max_tokens = max_tokens;
  layer.encoding = Encoding::Float32;
  layer.storage_type = StorageType::Float32;
  s.layers = {layer};
  s.page_tokens = page_tokens;

  KvCacheManager manager;
  manager.initialize(s);

  AttentionOptions options;
  options.query_heads = query_heads;
  options.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  options.causal = true;

  std::vector<float> history_k, history_v;

  for (size_t count : {5u, 1u, 2u}) {
    auto step = manager.beginStep(count);
    const size_t width = query_heads * head_dim;
    std::vector<float> queries(count * width);
    for (size_t i = 0; i < queries.size(); ++i)
      queries[i] = std::sin(static_cast<float>(i + step.from() * 7 + 1));

    std::vector<float> k(count * num_kv_heads * head_dim);
    std::vector<float> v(count * num_kv_heads * head_dim);
    for (size_t i = 0; i < k.size(); ++i) {
      k[i] = std::cos(static_cast<float>(i + step.from() * 11 + 3)) * 0.1f;
      v[i] = static_cast<float>((i + 5) % 37) / 19.0f;
    }

    manager.append(step, 0, k.data(), v.data(), k.size() * sizeof(float));
    history_k.insert(history_k.end(), k.begin(), k.end());
    history_v.insert(history_v.end(), v.begin(), v.end());
    manager.markReady(step, 0);

    auto actual = manager.attention(step, 0, queries, options);

    const size_t group = query_heads / num_kv_heads;
    const size_t kv_width = num_kv_heads * head_dim;

    for (size_t q = 0; q < count; ++q) {
      const size_t end = step.from() + q + 1;
      for (size_t h = 0; h < query_heads; ++h) {
        std::vector<double> scores(end);
        for (size_t t = 0; t < end; ++t) {
          double score = 0;
          for (size_t d = 0; d < head_dim; ++d) {
            score +=
              static_cast<double>(queries[q * width + h * head_dim + d]) *
              history_k[t * kv_width + (h / group) * head_dim + d];
          }
          scores[t] = score * options.scale;
        }
        double maximum = *std::max_element(scores.begin(), scores.end());
        double sum = 0;
        for (double &sc : scores) {
          sc = std::exp(sc - maximum);
          sum += sc;
        }
        for (size_t d = 0; d < head_dim; ++d) {
          double expected = 0;
          for (size_t t = 0; t < end; ++t)
            expected += scores[t] *
                        history_v[t * kv_width + (h / group) * head_dim + d] /
                        sum;
          EXPECT_NEAR(actual[q * width + h * head_dim + d], expected, 1e-4);
        }
      }
    }
    finish(manager, std::move(step));
  }
}

TEST(KvCacheCore,
     pagedAttentionVectorizedFloat16MatchesDenseReferenceForStandardHeadDim) {
  const size_t head_dim = 64;
  const size_t num_kv_heads = 2;
  const size_t query_heads = 4;
  const size_t page_tokens = 4;
  const size_t max_tokens = 16;

  CacheSpec s;
  LayerSpec layer;
  layer.num_kv_heads = num_kv_heads;
  layer.head_dim = head_dim;
  layer.capacity = max_tokens;
  layer.max_tokens = max_tokens;
  layer.encoding = Encoding::Float16;
  layer.storage_type = StorageType::UInt16;
  s.layers = {layer};
  s.page_tokens = page_tokens;

  KvCacheManager manager;
  manager.initialize(s);

  AttentionOptions options;
  options.query_heads = query_heads;
  options.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  options.causal = true;

  std::vector<float> history_k, history_v;

  for (size_t count : {4u, 1u}) {
    auto step = manager.beginStep(count);
    const size_t width = query_heads * head_dim;
    std::vector<float> queries(count * width);
    for (size_t i = 0; i < queries.size(); ++i)
      queries[i] = std::sin(static_cast<float>(i + step.from() * 5 + 2));

    const size_t kv_elements = count * num_kv_heads * head_dim;
    std::vector<uint16_t> k_u16(kv_elements), v_u16(kv_elements);
    std::vector<float> k_f32(kv_elements), v_f32(kv_elements);

    for (size_t i = 0; i < kv_elements; ++i) {
      float k_val =
        std::cos(static_cast<float>(i + step.from() * 7 + 1)) * 0.1f;
      float v_val = static_cast<float>((i + 3) % 29) / 15.0f;
      uint16_t k_bits = nntrainer::compute_fp32_to_fp16(k_val);
      uint16_t v_bits = nntrainer::compute_fp32_to_fp16(v_val);
      k_u16[i] = k_bits;
      v_u16[i] = v_bits;
      k_f32[i] = nntrainer::compute_fp16_to_fp32(k_bits);
      v_f32[i] = nntrainer::compute_fp16_to_fp32(v_bits);
    }

    manager.append(step, 0, k_u16.data(), v_u16.data(),
                   kv_elements * sizeof(uint16_t));
    history_k.insert(history_k.end(), k_f32.begin(), k_f32.end());
    history_v.insert(history_v.end(), v_f32.begin(), v_f32.end());
    manager.markReady(step, 0);

    auto actual = manager.attention(step, 0, queries, options);

    const size_t group = query_heads / num_kv_heads;
    const size_t kv_width = num_kv_heads * head_dim;

    for (size_t q = 0; q < count; ++q) {
      const size_t end = step.from() + q + 1;
      for (size_t h = 0; h < query_heads; ++h) {
        std::vector<double> scores(end);
        for (size_t t = 0; t < end; ++t) {
          double score = 0;
          for (size_t d = 0; d < head_dim; ++d) {
            score +=
              static_cast<double>(queries[q * width + h * head_dim + d]) *
              history_k[t * kv_width + (h / group) * head_dim + d];
          }
          scores[t] = score * options.scale;
        }
        double maximum = *std::max_element(scores.begin(), scores.end());
        double sum = 0;
        for (double &sc : scores) {
          sc = std::exp(sc - maximum);
          sum += sc;
        }
        for (size_t d = 0; d < head_dim; ++d) {
          double expected = 0;
          for (size_t t = 0; t < end; ++t)
            expected += scores[t] *
                        history_v[t * kv_width + (h / group) * head_dim + d] /
                        sum;
          EXPECT_NEAR(actual[q * width + h * head_dim + d], expected, 2e-3);
        }
      }
    }
    finish(manager, std::move(step));
  }
}

TEST(KvCacheCore, pagedAttentionThreadedGroupsMatchDenseReference) {
  const size_t head_dim = 64, kv_heads = 2, query_heads = 8;
  const size_t width = query_heads * head_dim, kv_width = kv_heads * head_dim;
  const size_t group = query_heads / kv_heads, window = 20;
  CacheSpec s;
  LayerSpec layer;
  layer.num_kv_heads = kv_heads;
  layer.head_dim = head_dim;
  layer.capacity = layer.max_tokens = 64;
  layer.encoding = Encoding::Float16;
  layer.storage_type = StorageType::UInt16;
  s.layers = {layer};
  s.page_tokens = 8;
  KvCacheManager manager;
  manager.initialize(s);
  AttentionOptions options;
  options.query_heads = query_heads;
  options.window = window;
  options.sinks = {0.5f, -1, 0, 1, 2, -2, 0.25f, 0};
  std::vector<float> history_k, history_v;
  // A long prefill spreads groups across threads; decode splits the groups.
  for (size_t count : {48u, 1u, 3u}) {
    auto step = manager.beginStep(count);
    std::vector<float> queries(count * width);
    for (size_t i = 0; i < queries.size(); ++i)
      queries[i] =
        std::sin(static_cast<float>(i + step.from() * 3 + 1) * 0.37f);
    std::vector<uint16_t> k(count * kv_width), v(count * kv_width);
    for (size_t i = 0; i < k.size(); ++i) {
      k[i] = nntrainer::compute_fp32_to_fp16(
        std::cos(static_cast<float>(i + step.from() * kv_width) * 0.11f) *
        0.2f);
      v[i] = nntrainer::compute_fp32_to_fp16(
        static_cast<float>((i * 7 + step.from()) % 23) / 11.0f);
      history_k.push_back(nntrainer::compute_fp16_to_fp32(k[i]));
      history_v.push_back(nntrainer::compute_fp16_to_fp32(v[i]));
    }
    manager.append(step, 0, k.data(), v.data(), k.size() * sizeof(uint16_t));
    manager.markReady(step, 0);
    for (bool causal : {true, false}) {
      options.causal = causal;
      auto actual = manager.attention(step, 0, queries, options);
      for (size_t q = 0; q < count; ++q) {
        const size_t end = causal ? step.from() + q + 1 : step.to();
        const size_t start = end > window ? end - window : 0;
        for (size_t h = 0; h < query_heads; ++h) {
          const size_t kv_offset = (h / group) * head_dim;
          std::vector<double> scores(end - start);
          double maximum = options.sinks[h];
          const float *q_head = queries.data() + q * width + h * head_dim;
          for (size_t t = start; t < end; ++t) {
            double score = 0;
            for (size_t d = 0; d < head_dim; ++d)
              score += static_cast<double>(q_head[d]) *
                       history_k[t * kv_width + kv_offset + d];
            scores[t - start] = score / 8.0; // 1 / sqrt(64)
            maximum = std::max(maximum, scores[t - start]);
          }
          double sum = std::exp(options.sinks[h] - maximum);
          for (auto &score : scores) {
            score = std::exp(score - maximum);
            sum += score;
          }
          for (size_t d = 0; d < head_dim; ++d) {
            double expected = 0;
            for (size_t t = start; t < end; ++t)
              expected +=
                scores[t - start] * history_v[t * kv_width + kv_offset + d];
            EXPECT_NEAR(actual[q * width + h * head_dim + d], expected / sum,
                        2e-3);
          }
        }
      }
    }
    manager.commit(std::move(step));
  }
}

TEST(KvCacheCore, pagePoolFreesIdlePagesOfOtherSizesUnderBudget) {
  auto pool = std::make_shared<KvPagePool>(128);
  {
    auto first = pool->acquire(64);
    auto second = pool->acquire(64);
    EXPECT_EQ(pool->reservedBytes(), 128u);
    EXPECT_THROW(pool->acquire(32), std::bad_alloc); // Both pages are in use.
  }
  auto large = pool->acquire(128); // Idle 64-byte pages are freed first.
  EXPECT_EQ(pool->reservedBytes(), 128u);
  const auto identity = large->identity();
  large.reset();
  EXPECT_EQ(pool->acquire(128)->identity(), identity);
}

TEST(KvCacheCore, pagePoolHandlesMayOutliveThePool) {
  std::shared_ptr<KvStorage> page;
  {
    KvPagePool pool;
    page = pool.acquire(64);
  }
  ASSERT_EQ(page->bytes(), 64u);
  std::memset(page->data(), 1, 64);
  page.reset();
}

TEST(KvCacheCore, pagedTruncateReturnsWholeBlocksAndRejectsForwardSeek) {
  auto s = spec();
  s.page_tokens = 2;
  auto pool = std::make_shared<KvPagePool>();
  KvCacheManager manager;
  manager.initialize(s, std::make_shared<nntrainer::MemAllocator>(), pool);
  finish(manager, manager.beginStep(5));
  EXPECT_EQ(manager.stats().backing_bytes, 384u);
  manager.setPosition(1);
  EXPECT_EQ(manager.stats().backing_bytes, 128u);
  EXPECT_THROW(manager.setPosition(2), std::logic_error);
  pool->trim();
  EXPECT_EQ(pool->reservedBytes(), 128u);
  manager.reset();
  pool->trim();
  EXPECT_EQ(pool->reservedBytes(), 0u);
}

TEST(KvCacheCore, segmentedAttentionRejectsInvalidGeometryAndQueries) {
  auto s = spec();
  s.page_tokens = 2;
  KvCacheManager manager;
  manager.initialize(s);
  auto step = manager.beginStep(1);
  manager.markReady(step, 0);
  AttentionOptions options;
  options.query_heads = 3;
  EXPECT_THROW(manager.attention(step, 0, std::vector<float>(6), options),
               std::invalid_argument);
  options.query_heads = 2;
  EXPECT_THROW(manager.attention(step, 0, std::vector<float>(3), options),
               std::invalid_argument);
  options.scale = std::numeric_limits<float>::infinity();
  EXPECT_THROW(manager.attention(step, 0, std::vector<float>(4), options),
               std::invalid_argument);
  options.scale = 0;
  auto output =
    manager.attention(step, 0, std::vector<float>(4, 10000), options);
  for (float value : output)
    EXPECT_FLOAT_EQ(value, 0);
  EXPECT_THROW(manager.append(step, 0, nullptr, nullptr, 16),
               std::invalid_argument);
}

TEST(KvCacheCore, pagedGraphIdentitiesResolveAliasesAndRequireActiveLease) {
  auto s = spec();
  s.page_tokens = 2;
  s.layers.push_back(s.layers[0]);
  s.layers[1].source = 0;
  KvCacheManager manager;
  manager.initialize(s);
  auto k0 = manager.graphKeyMemory(0), v0 = manager.graphValueMemory(0);
  auto k1 = manager.graphKeyMemory(1), v1 = manager.graphValueMemory(1);
  ASSERT_NE(k0, k1);
  auto step = manager.beginStep(3);
  auto scope = manager.bindStep(step);
  auto owner = KvStepScope::access(k0->getAddr(), v0->getAddr(), 0, 0, 3);
  auto alias = KvStepScope::access(k1->getAddr(), v1->getAddr(), 0, 0, 3);
  ASSERT_TRUE(owner->paged());
  ASSERT_TRUE(alias->alias());
  std::vector<float> kv(12, 1), q(12, 2);
  owner->append(kv.data(), kv.data(), kv.size() * sizeof(float));
  AttentionOptions options;
  options.query_heads = 2;
  options.sinks = {0, 0};
  EXPECT_EQ(alias->attention(q, options), owner->attention(q, options));
  manager.commit(std::move(step));
  EXPECT_THROW(owner->attention(q, options), std::logic_error);
  scope.reset();
  EXPECT_THROW(owner->paged(), std::logic_error);
}

TEST(KvCacheCore, pagedSnapshotRestoreIsAtomicAndPreservesGraphIdentities) {
  auto s = spec();
  s.page_tokens = 2;
  KvCacheManager manager;
  manager.initialize(s);
  auto handle = manager.graphKeyMemory(0);
  auto step = manager.beginStep(3);
  std::vector<float> data(12, 5);
  manager.append(step, 0, data.data(), data.data(),
                 data.size() * sizeof(float));
  finish(manager, std::move(step));
  std::stringstream saved;
  manager.savePlane(saved, 0, true, 3);
  manager.savePlane(saved, 0, false, 3);
  std::stringstream truncated(saved.str().substr(0, saved.str().size() - 1));
  EXPECT_THROW(manager.restorePlanes(truncated, 3), std::runtime_error);
  EXPECT_EQ(manager.committedLength(), 3u);
  EXPECT_EQ(manager.graphKeyMemory(0), handle);
  manager.reset();
  manager.restorePlanes(saved, 3);
  EXPECT_EQ(manager.graphKeyMemory(0), handle);
  EXPECT_EQ(manager.committedLength(), 3u);
  auto next = manager.beginStep(1);
  std::vector<float> tail(4, 5);
  manager.append(next, 0, tail.data(), tail.data(), 16);
  manager.markReady(next, 0);
  AttentionOptions options;
  options.query_heads = 2;
  for (float value :
       manager.attention(next, 0, std::vector<float>(4, 1), options))
    EXPECT_FLOAT_EQ(value, 5);
}

TEST(KvCacheCore, pagedSinkContributesOnlyToSoftmaxDenominator) {
  auto s = spec();
  s.page_tokens = 1;
  KvCacheManager manager;
  manager.initialize(s);
  auto step = manager.beginStep(2);
  std::vector<float> keys(8), values(8, 6);
  manager.append(step, 0, keys.data(), values.data(), 32);
  manager.markReady(step, 0);
  AttentionOptions options;
  options.query_heads = 2;
  options.sinks = {0, 0};
  auto result = manager.attention(step, 0, std::vector<float>(8), options);
  EXPECT_FLOAT_EQ(result[0], 3);
  EXPECT_FLOAT_EQ(result[4], 4);
}

TEST(KvCacheCore, pagePoolAsksReclaimersBeforeFailing) {
  auto pool = std::make_shared<KvPagePool>(128);
  std::vector<std::shared_ptr<KvStorage>> held = {pool->acquire(64),
                                                  pool->acquire(64)};
  size_t calls = 0;
  const auto token = pool->addReclaimer([&held, &calls]() {
    ++calls;
    if (held.empty())
      return false;
    held.pop_back();
    return true;
  });
  auto page = pool->acquire(64); // Reuses the page the reclaimer released.
  EXPECT_EQ(calls, 1u);
  EXPECT_EQ(pool->reservedBytes(), 128u);
  pool->removeReclaimer(token);
  EXPECT_THROW(pool->acquire(64), std::bad_alloc);
  EXPECT_EQ(calls, 1u);
}

GTEST_API_ int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
