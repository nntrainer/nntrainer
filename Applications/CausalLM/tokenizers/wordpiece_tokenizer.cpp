// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   wordpiece_tokenizer.cpp
 * @brief  Compact WordPiece tokenizer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <tokenizers/tokenizer_cache_util.h>
#include <tokenizers_cpp.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace tokenizers {
namespace {

constexpr uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kPackedInvalidId = cache_util::kMaxU24;
constexpr char kCacheName[] = "WordPiece";
constexpr cache_util::CacheKind kCacheKind = cache_util::CacheKind::WordPiece;

using cache_util::AppendHeader;
using cache_util::AppendU16;
using cache_util::AppendU24;
using cache_util::AppendU32;
using cache_util::ReadBytes;
using cache_util::ReadHeader;
using cache_util::ReadU32;
using cache_util::ReadU32Vector;

uint32_t PackTokenId(uint32_t token_id) {
  if (token_id == kInvalidId) {
    return kPackedInvalidId;
  }
  if (token_id >= kPackedInvalidId) {
    throw std::runtime_error("Invalid WordPiece cache: token id overflow");
  }
  return token_id;
}

uint32_t UnpackTokenId(uint32_t token_id) {
  return token_id == kPackedInvalidId ? kInvalidId : token_id;
}

std::string ToLowerAscii(const std::string &text) {
  std::string out = text;
  for (char &c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

bool IsWhitespace(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

bool IsPunctuation(unsigned char c) {
  return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) ||
         (c >= 123 && c <= 126);
}

std::vector<std::string> BasicTokenize(const std::string &text,
                                       bool do_lower_case) {
  const std::string normalized = do_lower_case ? ToLowerAscii(text) : text;
  std::vector<std::string> tokens;
  std::string current;

  auto flush_current = [&]() {
    if (!current.empty()) {
      tokens.push_back(current);
      current.clear();
    }
  };

  for (unsigned char c : normalized) {
    if (IsWhitespace(c)) {
      flush_current();
    } else if (IsPunctuation(c)) {
      flush_current();
      tokens.emplace_back(1, static_cast<char>(c));
    } else {
      current.push_back(static_cast<char>(c));
    }
  }
  flush_current();

  return tokens;
}

std::vector<size_t> Utf8CharStarts(const std::string &text) {
  std::vector<size_t> starts;
  for (size_t i = 0; i < text.size();) {
    starts.push_back(i);
    const unsigned char c = static_cast<unsigned char>(text[i]);
    if (c < 0x80) {
      ++i;
    } else if ((c & 0xe0) == 0xc0 && i + 1 < text.size()) {
      i += 2;
    } else if ((c & 0xf0) == 0xe0 && i + 2 < text.size()) {
      i += 3;
    } else if ((c & 0xf8) == 0xf0 && i + 3 < text.size()) {
      i += 4;
    } else {
      ++i;
    }
  }
  starts.push_back(text.size());
  return starts;
}

/**
 * @brief Self-contained native WordPiece tokenizer with cache save/load
 * support
 */
class CompactWordPieceTokenizer : public Tokenizer {
public:
  /**
   * @brief Trie node used by the in-memory greedy-longest-match lookup
   */
  struct TrieNode {
    uint32_t first_edge = 0;
    uint32_t edge_count = 0;
    uint32_t token_id = kInvalidId;
  };
  static_assert(sizeof(TrieNode) == sizeof(uint32_t) * 3,
                "TrieNode in-memory layout must stay compact");

  /**
   * @brief Single byte-labeled outgoing edge of a TrieNode
   */
  struct TrieEdge {
    unsigned char byte = 0;
    uint32_t child = 0;
  };

  CompactWordPieceTokenizer(const std::string &vocab_blob, bool do_lower_case,
                            std::string unk_token,
                            std::string continuing_subword_prefix,
                            uint32_t max_input_chars_per_word,
                            std::string cls_token, std::string sep_token) :
    do_lower_case_(do_lower_case),
    unk_token_(std::move(unk_token)),
    continuing_subword_prefix_(std::move(continuing_subword_prefix)),
    cls_token_(std::move(cls_token)),
    sep_token_(std::move(sep_token)),
    max_input_chars_per_word_(max_input_chars_per_word) {
    LoadVocab(vocab_blob);
    BuildTrie();
    unk_id_ = LookupToken(unk_token_);
    cls_id_ = LookupToken(cls_token_);
    sep_id_ = LookupToken(sep_token_);
    if (unk_id_ == kInvalidId) {
      throw std::runtime_error("WordPiece vocab does not contain unk token: " +
                               unk_token_);
    }
  }

  explicit CompactWordPieceTokenizer(const std::string &cache_blob) {
    LoadCache(cache_blob);
  }

  std::vector<int32_t> Encode(const std::string &text) final {
    return Encode(text, false);
  }

  std::vector<int32_t> Encode(const std::string &text,
                              bool add_special_tokens) final {
    std::vector<int32_t> output;
    if (add_special_tokens && cls_id_ != kInvalidId) {
      output.push_back(static_cast<int32_t>(cls_id_));
    }

    for (const std::string &token : BasicTokenize(text, do_lower_case_)) {
      const auto pieces = TokenizeWord(token);
      output.insert(output.end(), pieces.begin(), pieces.end());
    }

    if (add_special_tokens && sep_id_ != kInvalidId) {
      output.push_back(static_cast<int32_t>(sep_id_));
    }

    return output;
  }

  std::string Decode(const std::vector<int32_t> &ids) final {
    std::string out;
    for (int32_t raw_id : ids) {
      if (raw_id < 0) {
        continue;
      }
      std::string token = IdToToken(raw_id);
      if (token.empty() || token == "[PAD]" || token == "[CLS]" ||
          token == "[SEP]") {
        continue;
      }

      if (token.rfind(continuing_subword_prefix_, 0) == 0) {
        out += token.substr(continuing_subword_prefix_.size());
      } else {
        if (!out.empty()) {
          out.push_back(' ');
        }
        out += token;
      }
    }
    return out;
  }

  size_t GetVocabSize() final {
    return token_offsets_.empty() ? 0 : token_offsets_.size() - 1;
  }

  std::string IdToToken(int32_t token_id) final {
    if (token_id < 0 ||
        static_cast<size_t>(token_id) + 1 >= token_offsets_.size()) {
      return "";
    }

    const uint32_t id = static_cast<uint32_t>(token_id);
    const uint32_t begin = token_offsets_[id];
    const uint32_t end = token_offsets_[id + 1];
    if (begin > end || end > token_bytes_.size()) {
      return "";
    }
    return token_bytes_.substr(begin, end - begin);
  }

  int32_t TokenToId(const std::string &token) final {
    const uint32_t id = LookupToken(token);
    return id == kInvalidId ? -1 : static_cast<int32_t>(id);
  }

  std::string SerializeToCache() const final {
    std::string out;
    AppendHeader(out, kCacheKind);
    AppendU32(out, do_lower_case_ ? 1 : 0);
    AppendU32(out, static_cast<uint32_t>(
                     token_offsets_.empty() ? 0 : token_offsets_.size() - 1));
    AppendU32(out, static_cast<uint32_t>(nodes_.size()));
    AppendU32(out, static_cast<uint32_t>(edges_.size()));
    AppendU32(out, static_cast<uint32_t>(token_bytes_.size()));
    AppendU32(out, unk_id_);
    AppendU32(out, cls_id_);
    AppendU32(out, sep_id_);
    AppendU32(out, max_input_chars_per_word_);
    AppendU32(out, static_cast<uint32_t>(unk_token_.size()));
    AppendU32(out, static_cast<uint32_t>(continuing_subword_prefix_.size()));
    AppendU32(out, static_cast<uint32_t>(cls_token_.size()));
    AppendU32(out, static_cast<uint32_t>(sep_token_.size()));
    out += unk_token_;
    out += continuing_subword_prefix_;
    out += cls_token_;
    out += sep_token_;

    for (uint32_t offset : token_offsets_) {
      AppendU32(out, offset);
    }
    out += token_bytes_;
    for (const auto &node : nodes_) {
      AppendU24(out, node.first_edge, kCacheName);
      AppendU16(out, node.edge_count, kCacheName);
      AppendU24(out, PackTokenId(node.token_id), kCacheName);
    }
    for (const auto &edge : edges_) {
      out.push_back(static_cast<char>(edge.byte));
      AppendU24(out, edge.child, kCacheName);
    }

    return out;
  }

private:
  /**
   * @brief Mutable trie node used while building the vocabulary trie
   */
  struct MutableNode {
    uint32_t token_id = kInvalidId;
    std::unordered_map<unsigned char, uint32_t> children;
  };

  void LoadVocab(const std::string &vocab_blob) {
    std::vector<std::string> tokens;
    size_t begin = 0;
    while (begin <= vocab_blob.size()) {
      size_t end = vocab_blob.find('\n', begin);
      if (end == std::string::npos) {
        end = vocab_blob.size();
      }

      std::string token = vocab_blob.substr(begin, end - begin);
      if (!token.empty() && token.back() == '\r') {
        token.pop_back();
      }
      if (!token.empty()) {
        tokens.push_back(std::move(token));
      }

      if (end == vocab_blob.size()) {
        break;
      }
      begin = end + 1;
    }

    if (tokens.empty()) {
      throw std::runtime_error("WordPiece vocab is empty");
    }

    token_offsets_.clear();
    token_bytes_.clear();
    token_offsets_.reserve(tokens.size() + 1);
    token_offsets_.push_back(0);
    for (const std::string &token : tokens) {
      token_bytes_ += token;
      token_offsets_.push_back(static_cast<uint32_t>(token_bytes_.size()));
    }
  }

  void BuildTrie() {
    std::vector<MutableNode> mutable_nodes(1);
    for (uint32_t id = 0; id + 1 < token_offsets_.size(); ++id) {
      const uint32_t begin = token_offsets_[id];
      const uint32_t end = token_offsets_[id + 1];
      uint32_t node = 0;
      for (uint32_t i = begin; i < end; ++i) {
        const unsigned char byte = static_cast<unsigned char>(token_bytes_[i]);
        auto it = mutable_nodes[node].children.find(byte);
        if (it == mutable_nodes[node].children.end()) {
          const uint32_t child = static_cast<uint32_t>(mutable_nodes.size());
          mutable_nodes[node].children[byte] = child;
          mutable_nodes.emplace_back();
          node = child;
        } else {
          node = it->second;
        }
      }
      mutable_nodes[node].token_id = id;
    }

    nodes_.assign(mutable_nodes.size(), TrieNode{});
    edges_.clear();
    for (size_t i = 0; i < mutable_nodes.size(); ++i) {
      std::vector<std::pair<unsigned char, uint32_t>> children(
        mutable_nodes[i].children.begin(), mutable_nodes[i].children.end());
      std::sort(
        children.begin(), children.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

      nodes_[i].first_edge = static_cast<uint32_t>(edges_.size());
      nodes_[i].edge_count = static_cast<uint32_t>(children.size());
      nodes_[i].token_id = mutable_nodes[i].token_id;
      for (const auto &[byte, child] : children) {
        edges_.push_back(TrieEdge{byte, child});
      }
    }
  }

  uint32_t LookupToken(const std::string &token) const {
    uint32_t node = 0;
    for (unsigned char byte : token) {
      const TrieNode &trie_node = nodes_[node];
      const auto begin = edges_.begin() + trie_node.first_edge;
      const auto end = begin + trie_node.edge_count;
      auto it = std::lower_bound(begin, end, byte,
                                 [](const TrieEdge &edge, unsigned char value) {
                                   return edge.byte < value;
                                 });
      if (it == end || it->byte != byte) {
        return kInvalidId;
      }
      node = it->child;
    }
    return nodes_[node].token_id;
  }

  std::vector<int32_t> TokenizeWord(const std::string &word) const {
    const auto char_starts = Utf8CharStarts(word);
    if (char_starts.size() <= 1 ||
        char_starts.size() - 1 > max_input_chars_per_word_) {
      return {static_cast<int32_t>(unk_id_)};
    }

    std::vector<int32_t> output;
    size_t start_idx = 0;
    while (start_idx + 1 < char_starts.size()) {
      size_t end_idx = char_starts.size() - 1;
      uint32_t cur_id = kInvalidId;
      size_t cur_end_idx = start_idx;

      while (start_idx < end_idx) {
        std::string candidate =
          word.substr(char_starts[start_idx],
                      char_starts[end_idx] - char_starts[start_idx]);
        if (start_idx > 0) {
          candidate = continuing_subword_prefix_ + candidate;
        }

        const uint32_t id = LookupToken(candidate);
        if (id != kInvalidId) {
          cur_id = id;
          cur_end_idx = end_idx;
          break;
        }
        --end_idx;
      }

      if (cur_id == kInvalidId) {
        return {static_cast<int32_t>(unk_id_)};
      }

      output.push_back(static_cast<int32_t>(cur_id));
      start_idx = cur_end_idx;
    }

    return output;
  }

  void LoadCache(const std::string &blob) {
    size_t offset = ReadHeader(blob, kCacheKind, kCacheName);

    const uint32_t flags = ReadU32(blob, offset, kCacheName);
    do_lower_case_ = (flags & 1) != 0;
    const uint32_t vocab_size = ReadU32(blob, offset, kCacheName);
    const uint32_t node_count = ReadU32(blob, offset, kCacheName);
    const uint32_t edge_count = ReadU32(blob, offset, kCacheName);
    const uint32_t token_bytes_size = ReadU32(blob, offset, kCacheName);
    unk_id_ = ReadU32(blob, offset, kCacheName);
    cls_id_ = ReadU32(blob, offset, kCacheName);
    sep_id_ = ReadU32(blob, offset, kCacheName);
    max_input_chars_per_word_ = ReadU32(blob, offset, kCacheName);
    const uint32_t unk_len = ReadU32(blob, offset, kCacheName);
    const uint32_t prefix_len = ReadU32(blob, offset, kCacheName);
    const uint32_t cls_len = ReadU32(blob, offset, kCacheName);
    const uint32_t sep_len = ReadU32(blob, offset, kCacheName);

    unk_token_ = ReadBytes(blob, offset, unk_len, kCacheName);
    continuing_subword_prefix_ =
      ReadBytes(blob, offset, prefix_len, kCacheName);
    cls_token_ = ReadBytes(blob, offset, cls_len, kCacheName);
    sep_token_ = ReadBytes(blob, offset, sep_len, kCacheName);

    ReadU32Vector(blob, offset, static_cast<size_t>(vocab_size) + 1,
                  token_offsets_, kCacheName);
    token_bytes_ = ReadBytes(blob, offset, token_bytes_size, kCacheName);

    LoadPackedNodes(blob, offset, node_count);
    LoadPackedEdges(blob, offset, edge_count);

    if (offset != blob.size()) {
      throw std::runtime_error("Invalid WordPiece cache: trailing bytes");
    }

    ValidateCache();
  }

  void LoadPackedNodes(const std::string &blob, size_t &offset,
                       uint32_t node_count) {
    constexpr size_t kPackedNodeSize = 8;
    if (offset > blob.size() || static_cast<size_t>(node_count) >
                                  (blob.size() - offset) / kPackedNodeSize) {
      throw std::runtime_error("Invalid WordPiece cache: truncated nodes");
    }

    nodes_.resize(node_count);
    const unsigned char *p =
      reinterpret_cast<const unsigned char *>(blob.data() + offset);
    for (auto &node : nodes_) {
      node.first_edge = static_cast<uint32_t>(p[0]) |
                        (static_cast<uint32_t>(p[1]) << 8) |
                        (static_cast<uint32_t>(p[2]) << 16);
      node.edge_count =
        static_cast<uint32_t>(p[3]) | (static_cast<uint32_t>(p[4]) << 8);
      const uint32_t token_id = static_cast<uint32_t>(p[5]) |
                                (static_cast<uint32_t>(p[6]) << 8) |
                                (static_cast<uint32_t>(p[7]) << 16);
      node.token_id = UnpackTokenId(token_id);
      p += kPackedNodeSize;
    }
    offset += static_cast<size_t>(node_count) * kPackedNodeSize;
  }

  void LoadPackedEdges(const std::string &blob, size_t &offset,
                       uint32_t edge_count) {
    constexpr size_t kPackedEdgeSize = 4;
    if (offset > blob.size() || static_cast<size_t>(edge_count) >
                                  (blob.size() - offset) / kPackedEdgeSize) {
      throw std::runtime_error("Invalid WordPiece cache: truncated edges");
    }

    edges_.resize(edge_count);
    const unsigned char *p =
      reinterpret_cast<const unsigned char *>(blob.data() + offset);
    for (auto &edge : edges_) {
      edge.byte = p[0];
      edge.child = static_cast<uint32_t>(p[1]) |
                   (static_cast<uint32_t>(p[2]) << 8) |
                   (static_cast<uint32_t>(p[3]) << 16);
      p += kPackedEdgeSize;
    }
    offset += static_cast<size_t>(edge_count) * kPackedEdgeSize;
  }

  void ValidateCache() const {
    const uint32_t vocab_size = static_cast<uint32_t>(
      token_offsets_.empty() ? 0 : token_offsets_.size() - 1);
    if (vocab_size == 0 || nodes_.empty()) {
      throw std::runtime_error("Invalid WordPiece cache: empty table");
    }
    if (unk_id_ == kInvalidId || unk_id_ >= vocab_size ||
        (cls_id_ != kInvalidId && cls_id_ >= vocab_size) ||
        (sep_id_ != kInvalidId && sep_id_ >= vocab_size)) {
      throw std::runtime_error("Invalid WordPiece cache: bad special token id");
    }
    if (token_offsets_.front() != 0 ||
        static_cast<size_t>(token_offsets_.back()) != token_bytes_.size()) {
      throw std::runtime_error("Invalid WordPiece cache: bad token offsets");
    }

    for (size_t i = 1; i < token_offsets_.size(); ++i) {
      if (token_offsets_[i - 1] > token_offsets_[i]) {
        throw std::runtime_error("Invalid WordPiece cache: unsorted offsets");
      }
    }
    for (const auto &node : nodes_) {
      if (static_cast<size_t>(node.first_edge) + node.edge_count >
          edges_.size()) {
        throw std::runtime_error("Invalid WordPiece cache: bad edge range");
      }
      if (node.token_id != kInvalidId && node.token_id >= vocab_size) {
        throw std::runtime_error("Invalid WordPiece cache: bad token id");
      }
    }
    for (const auto &edge : edges_) {
      if (static_cast<size_t>(edge.child) >= nodes_.size()) {
        throw std::runtime_error("Invalid WordPiece cache: bad child index");
      }
    }
  }

  bool do_lower_case_ = true;
  std::string unk_token_ = "[UNK]";
  std::string continuing_subword_prefix_ = "##";
  std::string cls_token_ = "[CLS]";
  std::string sep_token_ = "[SEP]";
  uint32_t max_input_chars_per_word_ = 100;
  uint32_t unk_id_ = kInvalidId;
  uint32_t cls_id_ = kInvalidId;
  uint32_t sep_id_ = kInvalidId;
  std::vector<uint32_t> token_offsets_;
  std::string token_bytes_;
  std::vector<TrieNode> nodes_;
  std::vector<TrieEdge> edges_;
};

} // namespace

std::unique_ptr<Tokenizer> Tokenizer::FromBlobWordPiece(
  const std::string &vocab_blob, bool do_lower_case,
  const std::string &unk_token, const std::string &continuing_subword_prefix,
  uint32_t max_input_chars_per_word, const std::string &cls_token,
  const std::string &sep_token) {
  return std::make_unique<CompactWordPieceTokenizer>(
    vocab_blob, do_lower_case, unk_token, continuing_subword_prefix,
    max_input_chars_per_word, cls_token, sep_token);
}

std::unique_ptr<Tokenizer>
Tokenizer::FromBlobWordPieceCache(const std::string &cache_blob) {
  return std::make_unique<CompactWordPieceTokenizer>(cache_blob);
}

} // namespace tokenizers
