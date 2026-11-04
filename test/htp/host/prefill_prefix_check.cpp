// SPDX-License-Identifier: Apache-2.0
/** @file prefill_prefix_check.cpp
 * @brief Immutable prefill pages, window layout and transactional validation.
 * Link with libcausallm and libnntrainer; no DSP/model weights required.
 */
#include <algorithm>
#include <iostream>
#include <kv_cache_manager.h>
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
  const auto c = cache.capturePrefix(192, b.get());
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
  std::cout << "Prefill shared pages, window layout, invalid restore: PASS\n";
}
