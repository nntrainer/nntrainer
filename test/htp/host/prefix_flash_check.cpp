// SPDX-License-Identifier: Apache-2.0
/** @file prefix_flash_check.cpp
 * @brief Async backing: exact bytes, overwrite isolation, LRU and corruption.
 */
#include <chrono>
#include <filesystem>
#include <iostream>
#include <prefix_flash_store.h>
#include <stdexcept>
using Store = causallm::PrefixFlashStore;
static void check(bool ok) {
  if (!ok)
    throw std::runtime_error("Prefix Flash check failed");
}
static std::shared_ptr<const Store::Snapshot> sample() {
  auto s = std::make_shared<Store::Snapshot>();
  s->position = 159;
  s->element_bytes = 2;
  s->layers.resize(2);
  for (unsigned l = 0; l < 2; ++l) {
    auto &layer = s->layers[l];
    layer.base = l ? 64 : 0;
    layer.width = 8;
    layer.dropped_k = {0.5f, 1.5f};
    layer.dropped_v = {2.5f, 3.5f};
    for (unsigned r = layer.base; r < s->position; r += 64) {
      auto p = std::make_shared<Store::Snapshot::Page>();
      p->key.resize(std::min(64u, s->position - r) * 8u * 2u);
      p->value.resize(p->key.size());
      for (size_t i = 0; i < p->key.size(); ++i) {
        p->key[i] = (r + i) % 256;
        p->value[i] = (l + 3 * i) % 256;
      }
      layer.pages.push_back(p);
    }
  }
  return s;
}
static void equal(const Store::Snapshot &a, const Store::Snapshot &b) {
  check(a.position == b.position && a.element_bytes == b.element_bytes &&
        a.layers.size() == b.layers.size());
  for (size_t l = 0; l < a.layers.size(); ++l) {
    const auto &x = a.layers[l], &y = b.layers[l];
    check(x.base == y.base && x.width == y.width &&
          x.dropped_k == y.dropped_k && x.dropped_v == y.dropped_v &&
          x.pages.size() == y.pages.size());
    for (size_t p = 0; p < x.pages.size(); ++p)
      check(x.pages[p]->key == y.pages[p]->key &&
            x.pages[p]->value == y.pages[p]->value);
  }
}
int main() {
  const auto dir =
    std::filesystem::temp_directory_path() /
    ("nntr-flash-check-" +
     std::to_string(
       std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(dir);
  try {
    auto s = sample();
    size_t bytes = 48;
    for (const auto &l : s->layers)
      for (const auto &p : l.pages)
        bytes += p->key.size() + p->value.size();
    std::shared_ptr<Store::Entry> retired;
    {
      Store store(dir.string(), bytes * 2, bytes);
      check(!store.canStage(bytes * 3));
      auto a = store.submit(s);
      check(a && a->available());
      equal(*s, *a->load());
      store.drain();
      equal(*s, *a->load());
      auto b = store.submit(s);
      check(b != nullptr);
      store.drain();
      equal(*s, *a->load()); // A is now more recent than B.
      auto c = store.submit(s);
      check(c != nullptr);
      store.drain();
      check(a->available() && !b->available() && !b->load());
      equal(*s, *c->load());
      check(store.stats().evicted == 1);
      a.reset();
      check(store.stats().flash_bytes == bytes);
      for (const auto &sub : std::filesystem::directory_iterator(dir))
        for (const auto &file : std::filesystem::directory_iterator(sub.path()))
          if (file.path().extension() == ".kv") {
            std::fstream f(file.path(),
                           std::ios::binary | std::ios::in | std::ios::out);
            f.seekp(48);
            f.put('\x7f');
          }
      check(!c->load() && !c->available());
      check(store.stats().failed == 1 && store.stats().flash_bytes == 0);
      retired = c;
      check(store.stats().peak_pending_bytes <= bytes);
      // Releasing a pending entry must not race the writer or leak its backing.
      auto canceled = store.submit(s);
      check(canceled != nullptr);
      canceled.reset();
      store.drain();
      check(store.stats().pending_bytes == 0 && store.stats().flash_bytes == 0);
    }
    check(!retired->available() && !retired->load());
    {
      Store blobs(dir.string(), 16384, 8192);
      for (const std::string &input :
           {std::string(), std::string("x"), std::string("a\0b\xff", 4),
            std::string(4097, 'z')}) {
        auto entry = blobs.submitBlob(input);
        check(entry && entry->loadBlob() == input);
        blobs.drain();
        check(entry->loadBlob() == input);
      }
    }
    {
      Store failed_writer(dir.string(), bytes, bytes);
      for (const auto &sub : std::filesystem::directory_iterator(dir))
        check(std::filesystem::remove(sub.path()));
      auto failed = failed_writer.submit(s);
      check(failed != nullptr);
      failed_writer.drain();
      check(!failed->available() && !failed->load());
      check(failed_writer.stats().failed == 1 &&
            failed_writer.stats().pending_bytes == 0 &&
            failed_writer.stats().flash_bytes == 0);
    }
    check(std::filesystem::is_empty(dir));
    std::filesystem::remove(dir);
    std::cout << "Async prefix backing: "
                 "bytes/window/LRU/budget/corruption/release PASS\n";
  } catch (...) {
    std::filesystem::remove_all(dir);
    throw;
  }
}
