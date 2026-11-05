// SPDX-License-Identifier: Apache-2.0
/** @file prefill_prefix_check.cpp
 * @brief Immutable prefill pages, window layout and transactional validation.
 * Link with libcausallm and libnntrainer; no DSP/model weights required.
 */
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <kv_cache_manager.h>
#include <map>
#include <stdexcept>

static void require(bool ok) {
  if (!ok)
    throw std::runtime_error("Prefill prefix check failed");
}
int main() {
  using Manager = causallm::KVCacheManager;
  Manager cache;
  cache.allocate(2, 1, 512, std::vector<unsigned>{8, 4},
                 ml::train::TensorDim::DataType::UINT16,
                 ml::train::TensorDim::Format::NCHW, {192, 512}, {64, 0});
  auto fill = [&] {
    for (unsigned l = 0; l < 2; ++l) {
      auto &k = cache.getKeyCache(l), &v = cache.getValueCache(l);
      for (unsigned r = 0; r < k.height(); ++r)
        for (unsigned d = 0; d < k.width(); ++d) {
          const auto i = size_t(r) * k.width() + d;
          k.getData<uint16_t>()[i] = uint16_t((r + cache.getBase(l)) * 13 + d);
          v.getData<uint16_t>()[i] = uint16_t((r + cache.getBase(l)) * 17 + d);
        }
    }
  };
  fill();
  cache.setPosition(192);
  const auto a = cache.capturePrefix(64);
  const auto b = cache.capturePrefix(128, a.get());
  require(a->layers[0].pages[0] == b->layers[0].pages[0]);
  require(b->layers[0].pages[0] != b->layers[0].pages[1]);
  std::fill_n(cache.getKeyCache(0).getData<uint16_t>(), 128 * 8, 0xffff);
  cache.restorePrefix(*a);
  require(cache.getPosition() == 64);
  require(cache.getKeyCache(0).getData<uint16_t>()[63 * 8] == 63 * 13);
  cache.restorePrefix(*b);
  require(cache.getKeyCache(0).getData<uint16_t>()[127 * 8] == 127 * 13);
  require(cache.compact(128) && cache.getBase(0) == 64);
  fill();
  cache.setPosition(192);
  const auto partial = cache.capturePrefix(159, b.get());
  const auto c = cache.capturePrefix(192, partial.get());
  require(partial->layers[0].pages.back()->key.size() == 31u * 8u * 2u);
  require(c->layers[0].pages.back() != partial->layers[0].pages.back());
  cache.restorePrefix(*partial);
  require(cache.getPosition() == 159);
  require(cache.getKeyCache(0).getData<uint16_t>()[94 * 8] == 158 * 13);
  cache.restorePrefix(*c);
  require(c->layers[0].base == 64 && c->layers[0].pages.size() == 2);
  require(c->layers[0].pages[0] == b->layers[0].pages[1]);
  require(c->layers[1].pages[0] == b->layers[1].pages[0]);
  cache.restorePrefix(*a);
  cache.restorePrefix(*c);
  require(cache.getBase(0) == 64 && cache.getPosition() == 192);
  require(cache.getKeyCache(0).getData<uint16_t>()[0] == 64 * 13);
  require(cache.getKeyCache(0).getData<uint16_t>()[127 * 8] == 191 * 13);
  auto invalid = *c;
  invalid.layers[1].pages.clear();
  try {
    cache.restorePrefix(invalid);
    require(false);
  } catch (const std::invalid_argument &) {
    require(cache.getPosition() == 192 && cache.getBase(0) == 64);
  }
  // A read-only final view has exactly the same live KV and dropped-row maxima
  // as the original compaction, without modifying any prefill payload.
  Manager moved, viewed;
  for (auto *m : {&moved, &viewed}) {
    m->allocate(1, 1, 256, std::vector<unsigned>{8},
                ml::train::TensorDim::DataType::UINT16,
                ml::train::TensorDim::Format::NCHW, {192}, {64});
    for (unsigned i = 0; i < 192u * 8u; ++i) {
      m->getKeyCache(0).getData<uint16_t>()[i] = uint16_t(i);
      m->getValueCache(0).getData<uint16_t>()[i] = uint16_t(i + 3);
    }
    m->setPosition(159);
  }
  require(moved.compact(159));
  viewed.prepareDecodeView(159);
  require(viewed.getDecodeOffset(0) == 95);
  require(viewed.getKeyCache(0).getData<uint16_t>()[0] == 0);
  require(viewed.droppedAmaxK(0) == moved.droppedAmaxK(0));
  require(viewed.droppedAmaxV(0) == moved.droppedAmaxV(0));
  const auto mv = moved.capturePrefix(159), vw = viewed.capturePrefix(159);
  require(mv->layers[0].base == vw->layers[0].base);
  for (size_t i = 0; i < mv->layers[0].pages.size(); ++i) {
    const auto &x = mv->layers[0].pages[i], &y = vw->layers[0].pages[i];
    require(x->key == y->key && x->value == y->value);
  }
  try {
    viewed.compact(159);
    require(false);
  } catch (const std::logic_error &) {
  }
  viewed.restorePrefix(*vw);
  require(viewed.getDecodeOffset(0) == 0);
  require(viewed.getKeyCache(0).getData<uint16_t>()[0] == 95u * 8u);

  // Borrowed RPC slabs survive owner release. Sliding compaction freezes only
  // pages in the mutated layer before its original memmove.
  std::shared_ptr<const Manager::PrefixSnapshot> pinned, still_borrowed;
  std::map<void *, std::shared_ptr<void>> slabs;
  {
    Manager borrowed;
    borrowed.setSharedAllocator(
      [&](size_t bytes) -> void * {
        void *p = std::malloc(bytes);
        if (p)
          slabs.emplace(p, std::shared_ptr<void>(p, std::free));
        return p;
      },
      [&](void *p) { slabs.erase(p); });
    borrowed.setSharedRetain([&](void *p) { return slabs.at(p); });
    borrowed.allocate(1, 1, 256, std::vector<unsigned>{8},
                      ml::train::TensorDim::DataType::UINT16,
                      ml::train::TensorDim::Format::NCHW, {192}, {64});
    auto *k = borrowed.getKeyCache(0).getData<uint16_t>();
    auto *v = borrowed.getValueCache(0).getData<uint16_t>();
    for (unsigned i = 0; i < 192u * 8u; ++i)
      k[i] = v[i] = uint16_t(i);
    borrowed.setPosition(159);
    pinned = borrowed.capturePrefix(159);
    require(pinned->layers[0].pages[0]->key_view ==
            reinterpret_cast<unsigned char *>(k));
    const auto staged = borrowed.capturePrefix(159, pinned.get(), true);
    require(borrowed.prefixPayloadBytes(159) == 159u * 8u * 2u * 2u);
    for (const auto &page : staged->layers[0].pages)
      require(!page->key_view && !page->value_view && !page->key_owner);
    require(borrowed.compact(159)); // unaligned sliding base is preserved
    require(reinterpret_cast<const uint16_t *>(
              staged->layers[0].pages[0]->keyData())[10] == 10);
    require(!pinned->layers[0].pages[0]->key_view);
    const auto live = borrowed.capturePrefix(159);
    require(live->layers[0].base == 95);
    borrowed.restorePrefix(*live);
    require(borrowed.getPosition() == 159);
    still_borrowed = borrowed.capturePrefix(159);
    require(still_borrowed->layers[0].pages[0]->key_view != nullptr);
  }
  require(slabs.empty());
  require(reinterpret_cast<const uint16_t *>(
            still_borrowed->layers[0].pages[0]->keyData())[0] == 95u * 8u);
  still_borrowed.reset();
  require(reinterpret_cast<const uint16_t *>(
            pinned->layers[0].pages[0]->keyData())[10] == 10);
  std::cout << "Prefill shared pages, window layout, invalid restore: PASS\n";
}
