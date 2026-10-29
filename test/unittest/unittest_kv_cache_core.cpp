// SPDX-License-Identifier: Apache-2.0
/** @file unittest_kv_cache_core.cpp
 * @brief CPU ownership, checked geometry and transactional KV publication.
 * @author Codex (OpenAI)
 * @bug Covers synchronous CPU storage; device integration is not exercised.
 */
#include <gtest/gtest.h>
#include <kv_cache/kv_cache_manager.h>
#include <tensor.h>

#include <cstring>
#include <limits>

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

GTEST_API_ int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
