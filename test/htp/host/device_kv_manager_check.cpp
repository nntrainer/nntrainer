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
      [&](uint32_t id, unsigned) {
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
  {
    std::set<uint32_t> prefix_handles;
    unsigned selected = 0;
    Manager m([](unsigned, unsigned) -> uint32_t { return 1; },
              [&](uint32_t id, unsigned pos) {
                assert(prefix_handles.count(id));
                selected = pos;
              },
              [](uint32_t) {}, [] {}, 2, 2);
    m.setPrefixLengths([](uint32_t) { return std::vector<unsigned>{4096}; });
    m.setPrefixRelease([&](uint32_t id) { assert(prefix_handles.erase(id)); });
    Manager::Tokens tokens(4096, 7);
    m.publish(tokens, 4096, 9, [&](unsigned pos, unsigned pending) {
      assert(pos == 4096 && pending == 9);
      prefix_handles.insert(42);
      return 42;
    });
    tokens.push_back(8);
    assert(m.reuse(tokens) == 4096 && selected == 4096);
    m.clearPrefixes();
    assert(prefix_handles.empty());
  }
  {
    std::set<uint32_t> handles;
    uint32_t selected = 0;
    Manager m([](unsigned, unsigned) -> uint32_t { return 1; },
              [&](uint32_t id, unsigned) { selected = id; }, [](uint32_t) {},
              [] {}, 2, 3);
    m.setPrefixCapture([&](unsigned, unsigned) {
      handles.insert(1);
      return 1;
    });
    m.setPrefixRelease([&](uint32_t id) { assert(handles.erase(id)); });
    m.setPrefixLengths([](uint32_t id) {
      return id == 1 ? std::vector<unsigned>{4095}
                     : std::vector<unsigned>{4096};
    });
    Manager::Tokens tokens(4096, 7);
    m.publish(tokens, 4096, 9);
    m.publish(tokens, 4096, 9, [&](unsigned, unsigned) {
      handles.insert(0x80000000u);
      return 0x80000000u;
    });
    tokens.push_back(8);
    assert(m.reuse(tokens) == 4096 && selected == 0x80000000u);
    m.clearPrefixes();
    assert(handles.empty());
  }
  {
    Manager m(
      [&](unsigned, unsigned) {
        live.insert(next);
        return next++;
      },
      [](uint32_t, unsigned) {}, [&](uint32_t id) { assert(live.erase(id)); },
      [] {}, 2, 2);
    m.setPrefixLengths([](uint32_t) { return std::vector<unsigned>{64, 95}; });
    m.setPrefixCompatible([](uint32_t, unsigned p, size_t n) {
      return p == 64 ? n > 96 : n == 96;
    });
    Manager::Tokens tokens(96, 7);
    m.publish(tokens, 96, 9);
    assert(m.reuse(tokens) == 95);
    tokens.resize(100, 7);
    assert(m.reuse(tokens) == 64); // reject longer token-only match
    tokens.resize(80);
    assert(m.reuse(tokens) == 0); // incompatible shorter call must miss
  }
  assert(live.empty());
  {
    std::set<uint32_t> device, host;
    uint32_t next_device = 1;
    {
      Manager m(
        [&](unsigned, unsigned) {
          device.insert(next_device);
          return next_device++;
        },
        [&](uint32_t id, unsigned position) {
          assert(position ? host.count(id) : device.count(id));
        },
        [&](uint32_t id) { assert(device.erase(id)); }, [] {}, 2, 1);
      m.setPrefixCapture([&](unsigned, unsigned) {
        assert(host.insert(1).second);
        return 1;
      });
      m.setPrefixRelease([&](uint32_t id) { assert(host.erase(id)); });
      m.publish({10, 20, 30}, 3, 9);
      assert(m.reuse({10, 20, 30, 40}) == 3);
      const auto child =
        m.fork(3, 9); // DSP handle 1 collides with host prefix 1
      m.select(child, 3, 9);
      m.clearPrefixes();
      assert(host.empty() && device.count(1));
      m.publish({40, 50, 60}, 3, 9);
      m.publish({70, 80, 90}, 3, 9); // host LRU eviction must not release DSP 1
      assert(device.count(1) && host.count(1));
    }
    assert(device.empty() && host.empty());
  }
  assert(live.empty());
  {
    unsigned selected = 0;
    Manager m(
      [&](unsigned, unsigned) {
        live.insert(next);
        return next++;
      },
      [&](uint32_t, unsigned length) { selected = length; },
      [&](uint32_t id) { assert(live.erase(id)); }, [] {}, 2, 2);
    m.setPrefixLengths([](uint32_t) { return std::vector<unsigned>{64, 128}; });
    Manager::Tokens source(256, 7);
    m.publish(source, 256, 9);
    // Decode-only endpoint and unavailable boundaries must not enter radix.
    source.push_back(8);
    assert(m.reuse(source) == 128 && selected == 128);
    source.resize(65);
    assert(m.reuse(source) == 64 && selected == 64);
    source.resize(64);
    assert(m.reuse(source) == 0);
    m.setMinimumSuffix(3);
    source.resize(130, 7);
    assert(m.reuse(source) == 64);
    source.resize(131, 7);
    assert(m.reuse(source) == 128);
  }
  assert(live.empty());
  {
    unsigned selected_length = 0;
    uint32_t capture_calls = 0;
    std::map<uint32_t, unsigned> first;
    Manager m(
      [&](unsigned position, unsigned) {
        ++capture_calls;
        first[next] = position > 1024 ? 2048 : 0;
        live.insert(next);
        return next++;
      },
      [&](uint32_t id, unsigned length) {
        assert(live.count(id));
        restored = id;
        selected_length = length;
      },
      [&](uint32_t id) { assert(live.erase(id)); }, [] {}, 2, 3,
      [&](uint32_t id) { return first.at(id); });
    Manager::Tokens source(1024);
    for (unsigned i = 0; i < source.size(); ++i)
      source[i] = i + 1;
    m.publish(source, 1024, 9);
    const uint32_t original = next - 1;
    assert(capture_calls == 1 && m.stats().radix_nodes == 16);
    Manager::Tokens divergent(source.begin(), source.begin() + 768);
    divergent.insert(divergent.end(), 100, 99999);
    assert(m.reuse(divergent) == 768);
    assert(restored == original && selected_length == 768);
    assert(m.tokens() == Manager::Tokens(source.begin(), source.begin() + 768));
    assert(capture_calls == 1); // no independently captured 768-token state
    divergent[750] = 77777;
    assert(m.reuse(divergent) == 704);
    assert(selected_length == 704);
    assert(m.reuse(source) == 960); // fresh logits require an input suffix
    Manager::Tokens long_source(3072);
    for (unsigned i = 0; i < long_source.size(); ++i)
      long_source[i] = i + 1;
    m.publish(long_source, 3072, 9);
    const uint32_t windowed = next - 1;
    divergent.assign(long_source.begin(), long_source.begin() + 2304);
    divergent.push_back(99999);
    assert(m.reuse(divergent) == 2304 && restored == windowed);
    // The newer windowed state cannot supply old sliding rows. The older
    // retained state still supplies its full-page terminals.
    divergent.assign(source.begin(), source.begin() + 768);
    divergent.push_back(99999);
    assert(m.reuse(divergent) == 768 && restored == original);
    m.clearPrefixes();
    m.publish(long_source, 3072, 9);
    assert(m.reuse(divergent) == 0); // required rows were actually evicted
  }
  {
    unsigned selected = 0;
    Manager m([](unsigned, unsigned) { return 1u; },
              [&](uint32_t, unsigned length) {
                if (length == 128)
                  throw Manager::PrefixUnavailable("corrupt/evicted Flash");
                selected = length;
              },
              [](uint32_t) {}, [] {}, 1, 1);
    m.setPrefixLengths([](uint32_t) { return std::vector<unsigned>{64, 128}; });
    Manager::Tokens source(160, 7);
    m.publish(source, 160, 0);
    auto branch = source;
    branch[150] = 8;
    assert(m.reuse(branch) == 64 && selected == 64);
    assert(m.tokens().size() == 64 && m.stats().hits == 1);
  }
  assert(live.empty());
  std::cout
    << "Device sessions, compressed radix, LRU, failure isolation: PASS\n";
}
