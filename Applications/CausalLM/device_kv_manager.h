// SPDX-License-Identifier: Apache-2.0
/** @file device_kv_manager.h
 * @brief Serialized device sessions and token radix index of immutable states.
 * @author Codex <noreply@openai.com>
 */
#ifndef NNTR_DEVICE_KV_MANAGER_H
#define NNTR_DEVICE_KV_MANAGER_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace causallm {

/** The backend owns page tables, calibration and positions behind handles.
 * Radix terminals reference immutable prefix pages. A backend can supply only
 * its causally reproducible prefill boundaries and retain their payload apart
 * from decode session pages. Missing sliding history is never indexed.
 * Thread confined; all operations occur between synchronous inference calls.
 */
class DeviceKVManager {
public:
  /** Backing was evicted/corrupt; try an earlier candidate or recompute. */
  struct PrefixUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
  };
  using Tokens = std::vector<unsigned int>;
  using Capture = std::function<uint32_t(unsigned, unsigned)>;
  using Restore = std::function<void(uint32_t, unsigned)>;
  using PrefixMin = std::function<unsigned(uint32_t)>;
  using PrefixLengths = std::function<std::vector<unsigned>(uint32_t)>;
  using PrefixCompatible = std::function<bool(uint32_t, unsigned, size_t)>;
  using Release = std::function<void(uint32_t)>;
  struct Session {
    uint32_t handle = 0;
    Tokens tokens; /**< Known processed tokens; excludes pending feedback. */
  };
  struct Stats {
    size_t sessions = 0, prefixes = 0, radix_nodes = 0;
    uint64_t lookups = 0, hits = 0, reused_tokens = 0;
  };

  DeviceKVManager(Capture capture, Restore restore, Release release,
                  std::function<void()> reset, size_t max_sessions,
                  size_t max_prefixes, PrefixMin prefix_min = {}) :
    capture_(std::move(capture)),
    restore_(std::move(restore)),
    release_(std::move(release)),
    reset_(std::move(reset)),
    prefix_min_(std::move(prefix_min)),
    max_sessions_(max_sessions),
    max_prefixes_(max_prefixes) {
    if (!max_sessions || max_sessions > 120u ||
        max_prefixes > 120u - max_sessions)
      throw std::invalid_argument("Device KV handle limit is 120");
    sessions_.emplace(0, Session{});
  }
  /** Backend-supplied safe boundaries, e.g. causally calibrated prefill pages.
   * Install before publishing. The retained decode endpoint alone need not
   * be a valid prefill prefix. */
  void setPrefixLengths(PrefixLengths lengths) {
    if (!prefixes_.empty())
      throw std::logic_error("Set prefix boundaries before publication");
    prefix_lengths_ = std::move(lengths);
  }
  /** A token match must also preserve the backend's original call boundaries.
   */
  void setPrefixCompatible(PrefixCompatible compatible) {
    prefix_compatible_ = std::move(compatible);
  }
  /** Prefill payload pins belong to the prefix LRU, not to decode sessions. */
  void setPrefixCapture(Capture capture) {
    prefix_capture_ = std::move(capture);
  }
  /** Prefix handles may use a host-only namespace, separate from DSP sessions.
   */
  void setPrefixRelease(Release release) {
    prefix_release_ = std::move(release);
  }
  /** Leave enough input rows for the backend's block-prefill entry. */
  void setMinimumSuffix(size_t rows) {
    if (!rows)
      throw std::invalid_argument("A prefix hit must leave input rows");
    minimum_suffix_ = rows;
  }
  ~DeviceKVManager() {
    for (const auto &s : sessions_)
      releaseNoThrow(s.second.handle);
    for (const auto &p : prefixes_)
      releasePrefixNoThrow(p.second.handle);
  }
  DeviceKVManager(const DeviceKVManager &) = delete;
  DeviceKVManager &operator=(const DeviceKVManager &) = delete;

  uint32_t create() {
    if (sessions_.size() >= max_sessions_ || !next_session_)
      throw std::length_error("Device KV session limit reached");
    const uint32_t id = next_session_++;
    sessions_.emplace(id, Session{});
    return id;
  }
  uint32_t fork(unsigned position, unsigned pending) {
    const uint32_t id = create();
    try {
      Session child;
      child.tokens = sessions_.at(active_).tokens;
      if (position)
        child.handle = capture_(position, pending);
      sessions_.at(id) = std::move(child);
    } catch (...) {
      sessions_.erase(id);
      throw;
    }
    return id;
  }
  void select(uint32_t id, unsigned position, unsigned pending) {
    auto target = sessions_.find(id);
    if (target == sessions_.end())
      throw std::invalid_argument("Unknown device KV session");
    if (id == active_)
      return;
    uint32_t outgoing = position ? capture_(position, pending) : 0;
    try {
      if (target->second.handle)
        restore_(target->second.handle, 0);
      else {
        Tokens old_tokens = sessions_.at(active_).tokens;
        reset_();
        sessions_.at(active_).tokens = std::move(old_tokens);
      }
    } catch (...) {
      releaseNoThrow(outgoing);
      throw;
    }
    auto &old = sessions_.at(active_);
    releaseNoThrow(old.handle);
    old.handle = outgoing;
    active_ = id;
  }
  void destroy(uint32_t id) {
    if (id == active_)
      throw std::logic_error("Select another session before deleting it");
    auto it = sessions_.find(id);
    if (it == sessions_.end())
      throw std::invalid_argument("Unknown device KV session");
    if (it->second.handle)
      release_(it->second.handle);
    sessions_.erase(it);
  }
  uint32_t active() const { return active_; }
  const Tokens &tokens() const { return sessions_.at(active_).tokens; }
  void setTokens(Tokens tokens) {
    sessions_.at(active_).tokens = std::move(tokens);
  }
  void forgetTokens() { sessions_.at(active_).tokens.clear(); }

  /** Longest published state strictly before the final prompt token.
   * Leaves at least one token to produce fresh logits under this request's
   * sampling policy. Only use for a new conversation's complete input IDs. */
  size_t reuse(const Tokens &tokens) {
    ++stats_.lookups;
    const Node *node = &root_;
    size_t offset = 0;
    std::vector<std::pair<uint32_t, size_t>> candidates;
    while (offset < tokens.size()) {
      auto it = node->children.find(tokens[offset]);
      if (it == node->children.end())
        break;
      node = it->second.get();
      if (node->edge.size() > tokens.size() - offset ||
          !std::equal(node->edge.begin(), node->edge.end(),
                      tokens.begin() + offset))
        break;
      offset += node->edge.size();
      if (node->terminal && minimum_suffix_ <= tokens.size() - offset &&
          (!prefix_compatible_ ||
           prefix_compatible_(node->terminal, static_cast<unsigned>(offset),
                              tokens.size()))) {
        candidates.emplace_back(node->terminal, offset);
      }
    }
    for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
      auto &p = prefixes_.at(it->first);
      Tokens adopted(p.tokens.begin(), p.tokens.begin() + it->second);
      try {
        restore_(p.handle, static_cast<unsigned>(it->second));
      } catch (const PrefixUnavailable &) {
        continue;
      }
      sessions_.at(active_).tokens = std::move(adopted);
      p.used = ++clock_;
      ++stats_.hits;
      stats_.reused_tokens += it->second;
      return it->second;
    }
    return 0;
  }

  /** Publish the backend's safe prefixes (or full pages plus the endpoint).
   * Terminals borrow one handle; pages and boundary metadata can be shared. */
  void publish(const Tokens &tokens, unsigned position, unsigned pending) {
    if (!max_prefixes_ || tokens.empty() || tokens.size() != position)
      return;
    // Equal token histories can have different quantizer states depending on
    // prefill chunking. Retain the first valid state; never combine its pages
    // with another state's scales.
    for (auto &entry : prefixes_)
      if (entry.second.tokens == tokens) {
        entry.second.used = ++clock_;
        return;
      }
    // Evict BEFORE retaining to bound temporary handles as well as residency.
    if (prefixes_.size() == max_prefixes_) {
      auto victim = std::min_element(prefixes_.begin(), prefixes_.end(),
                                     [](const auto &a, const auto &b) {
                                       return a.second.used < b.second.used;
                                     });
      releasePrefix(victim->second.handle);
      erase(root_, victim->first);
      prefixes_.erase(victim);
      rebuild();
    }
    Prefix p{tokens,
             (prefix_capture_ ? prefix_capture_ : capture_)(position, pending),
             ++clock_, 0};
    try {
      p.first = prefix_min_ ? prefix_min_(p.handle) : 0;
      prefixes_.emplace(p.handle, p);
      rebuild();
    } catch (...) {
      prefixes_.erase(p.handle);
      releasePrefixNoThrow(p.handle);
      // Tree publication is rebuilt atomically, so an older index cannot
      // contain this unpublished handle.
      throw;
    }
  }
  void clearPrefixes() {
    for (auto it = prefixes_.begin(); it != prefixes_.end();) {
      releasePrefix(it->second.handle);
      erase(root_, it->first);
      it = prefixes_.erase(it);
    }
    root_ = Node{};
  }
  Stats stats() const {
    Stats out = stats_;
    out.sessions = sessions_.size();
    out.prefixes = prefixes_.size();
    out.radix_nodes = nodes(root_);
    return out;
  }

private:
  struct Node {
    Tokens edge;
    uint32_t terminal = 0;
    std::map<unsigned, std::unique_ptr<Node>> children;
  };
  struct Prefix {
    Tokens tokens;
    uint32_t handle;
    uint64_t used;
    unsigned first; /**< Minimum prefix with all required sliding rows. */
  };
  static size_t nodes(const Node &n) {
    size_t count = 0;
    for (const auto &c : n.children)
      count += 1 + nodes(*c.second);
    return count;
  }
  static void insert(Node &root, const Tokens &tokens, size_t count,
                     uint32_t id) {
    Node *node = &root;
    size_t offset = 0;
    while (offset < count) {
      auto it = node->children.find(tokens[offset]);
      if (it == node->children.end()) {
        auto leaf = std::make_unique<Node>();
        leaf->edge.assign(tokens.begin() + offset, tokens.begin() + count);
        leaf->terminal = id;
        node->children.emplace(tokens[offset], std::move(leaf));
        return;
      }
      Node *child = it->second.get();
      size_t common = 0;
      while (common < child->edge.size() && offset + common < count &&
             child->edge[common] == tokens[offset + common])
        ++common;
      if (common < child->edge.size()) {
        auto branch = std::make_unique<Node>();
        branch->edge.assign(child->edge.begin(), child->edge.begin() + common);
        child->edge.erase(child->edge.begin(), child->edge.begin() + common);
        branch->children.emplace(child->edge[0], std::move(it->second));
        it->second = std::move(branch);
      }
      node = it->second.get();
      offset += common;
    }
    node->terminal = id;
  }
  static void erase(Node &node, uint32_t handle) noexcept {
    if (node.terminal == handle)
      node.terminal = 0;
    for (auto it = node.children.begin(); it != node.children.end();) {
      erase(*it->second, handle);
      if (!it->second->terminal && it->second->children.empty())
        it = node.children.erase(it);
      else
        ++it;
    }
  }
  void rebuild() {
    Node next;
    for (const auto &p : prefixes_) {
      const auto &entry = p.second;
      if (prefix_lengths_) {
        for (unsigned count : prefix_lengths_(entry.handle)) {
          if (count == 0 || count > entry.tokens.size())
            throw std::logic_error("Invalid backend prefix boundary");
          insert(next, entry.tokens, count, p.first);
        }
        continue;
      }
      size_t first =
        std::max<size_t>(64, (size_t(entry.first) + 63u) / 64u * 64u);
      for (size_t count = first; count <= entry.tokens.size(); count += 64u)
        insert(next, entry.tokens, count, p.first);
      // A completed state may end in a partial page and is also reusable.
      if (entry.tokens.size() >= entry.first)
        insert(next, entry.tokens, entry.tokens.size(), p.first);
    }
    root_ = std::move(next);
  }
  void releaseNoThrow(uint32_t handle) noexcept {
    if (handle)
      try {
        release_(handle);
      } catch (...) { /* DSP teardown also frees handles. */
      }
  }
  void releasePrefix(uint32_t handle) {
    (prefix_release_ ? prefix_release_ : release_)(handle);
  }
  void releasePrefixNoThrow(uint32_t handle) noexcept {
    if (handle)
      try {
        releasePrefix(handle);
      } catch (...) {
      }
  }
  Capture capture_;
  Capture prefix_capture_;
  Restore restore_;
  Release release_;
  Release prefix_release_;
  std::function<void()> reset_;
  PrefixMin prefix_min_;
  PrefixLengths prefix_lengths_;
  PrefixCompatible prefix_compatible_;
  size_t max_sessions_, max_prefixes_;
  size_t minimum_suffix_ = 1;
  uint32_t next_session_ = 1, active_ = 0;
  uint64_t clock_ = 0;
  std::map<uint32_t, Session> sessions_;
  std::map<uint32_t, Prefix> prefixes_;
  Node root_;
  Stats stats_;
};
} // namespace causallm
#endif
