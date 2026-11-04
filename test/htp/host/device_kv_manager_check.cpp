// SPDX-License-Identifier: Apache-2.0
/** @file device_kv_manager_check.cpp
 * @brief Session isolation, radix branch matching, LRU and failure recovery.
 */
#include "device_kv_manager.h"
#include <cassert>
#include <iostream>
#include <set>

int main() {
  using Manager = causallm::DeviceKVManager;
  uint32_t next = 1, restored = 0;
  bool fail_restore = false;
  std::set<uint32_t> live;
  {
    Manager m(
      [&](unsigned, unsigned) {
        live.insert(next);
        return next++;
      },
      [&](uint32_t id) {
        if (fail_restore)
          throw std::runtime_error("failure");
        assert(live.count(id));
        restored = id;
      },
      [&](uint32_t id) { assert(live.erase(id)); }, [&] { restored = 0; }, 3,
      3);
    m.setTokens({1, 2, 3});
    auto b = m.create();
    m.select(b, 3, 9);
    assert(restored == 0 && m.tokens().empty());
    m.setTokens({4, 5});
    m.select(0, 2, 8);
    assert(restored && m.tokens() == Manager::Tokens({1, 2, 3}));
    auto child = m.fork(3, 9);
    fail_restore = true;
    try {
      m.select(child, 3, 9);
      assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(m.active() == 0 && m.tokens() == Manager::Tokens({1, 2, 3}));
    fail_restore = false;
    m.select(child, 3, 9);
    assert(m.tokens() == Manager::Tokens({1, 2, 3}));
    m.destroy(b);
    // Splits inside a compressed edge, proper-prefix terminals and misses.
    m.publish({10, 20, 30, 40}, 4, 8);
    m.publish({10, 20, 31, 41}, 4, 8);
    m.publish({10, 20}, 2, 8);
    assert(m.stats().prefixes == 3 && m.stats().radix_nodes == 3);
    assert(m.reuse({10, 20, 30, 40, 50}) == 4);
    assert(m.reuse({10, 20, 30, 99}) == 2);
    assert(m.reuse({10, 99, 30}) == 0);
    assert(m.reuse({10, 20}) == 0); // must recompute at least one token
    m.publish({60, 61}, 2, 8);      // evicts least recent terminal
    assert(m.reuse({10, 20, 31, 41, 99}) == 2);
    assert(m.reuse({10, 20, 30, 40, 99}) == 4);
    const auto count = live.size();
    fail_restore = true;
    try {
      m.reuse({60, 61, 9});
      assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(live.size() == count);
    fail_restore = false;
    m.clearPrefixes();
    assert(m.stats().prefixes == 0 && m.stats().radix_nodes == 0);
  }
  assert(live.empty());
  std::cout
    << "Device sessions, compressed radix, LRU, failure isolation: PASS\n";
}
