// SPDX-License-Identifier: Apache-2.0
/** @file kv_cache_spec.h
 * @brief Internal, backend-independent KV geometry for contiguous CPU storage.
 * @author Codex (OpenAI)
 * @bug Only contiguous token-major CPU layouts are supported.
 */
#ifndef NNTRAINER_KV_CACHE_SPEC_H
#define NNTRAINER_KV_CACHE_SPEC_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nntrainer::kv {

/** Mathematical encoding and Tensor backing type are deliberately separate. */
enum class Encoding { Float32, Float16 };
enum class StorageType { Float32, Float16, UInt16 };

/** Checked arithmetic, including the allocator's alignment rounding. */
inline size_t checkedAdd(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a)
    throw std::overflow_error("KV byte size overflow");
  return a + b;
}

inline size_t checkedMultiply(size_t a, size_t b) {
  if (a && b > std::numeric_limits<size_t>::max() / a)
    throw std::overflow_error("KV byte size overflow");
  return a * b;
}

/** Physical order is [batch, token, kv_head, head_dim]; no GQA expansion. */
struct LayerSpec {
  size_t batch = 1;
  size_t num_kv_heads = 1;
  size_t head_dim = 0;
  size_t max_tokens = 0;
  size_t capacity = 0;
  int source = -1;
  bool has_producer =
    true; /**< False only for declared non-attention blocks. */
  Encoding encoding = Encoding::Float16;
  StorageType storage_type = StorageType::UInt16;

  size_t elementBytes() const { return encoding == Encoding::Float32 ? 4 : 2; }
  size_t rowBytes() const {
    return checkedMultiply(checkedMultiply(num_kv_heads, head_dim),
                           elementBytes());
  }
  size_t planeBytes() const {
    return checkedMultiply(checkedMultiply(batch, capacity), rowBytes());
  }
  bool sameGeometry(const LayerSpec &other) const {
    return batch == other.batch && num_kv_heads == other.num_kv_heads &&
           head_dim == other.head_dim && max_tokens == other.max_tokens &&
           capacity == other.capacity && encoding == other.encoding &&
           storage_type == other.storage_type;
  }
};

struct CacheSpec {
  std::vector<LayerSpec> layers;

  /** Validate the entire declaration before making any allocation. */
  size_t validate() const {
    if (layers.empty())
      throw std::invalid_argument("KV cache requires layers");
    size_t bytes = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
      const auto &s = layers[i];
      if (!s.batch || !s.num_kv_heads || !s.head_dim || !s.max_tokens ||
          !s.capacity || s.capacity > s.max_tokens ||
          s.max_tokens != layers.front().max_tokens)
        throw std::invalid_argument("Invalid KV geometry");
      if (!((s.encoding == Encoding::Float32 &&
             s.storage_type == StorageType::Float32) ||
            (s.encoding == Encoding::Float16 &&
             (s.storage_type == StorageType::Float16 ||
              s.storage_type == StorageType::UInt16))))
        throw std::invalid_argument("Invalid KV encoding/backing combination");
      const size_t plane = s.planeBytes();
      // Count actual aligned backing bytes, not just the logical payload.
      const size_t aligned = checkedAdd(plane, 63) & ~size_t{63};
      if (s.source < -1 ||
          (s.source >= 0 && (static_cast<size_t>(s.source) >= i ||
                             !s.sameGeometry(layers[s.source]) ||
                             !layers[s.source].has_producer)))
        throw std::invalid_argument("Invalid KV source or alias geometry");
      if (s.source < 0)
        bytes = checkedAdd(bytes, checkedMultiply(aligned, 2));
    }
    return bytes;
  }
};

} // namespace nntrainer::kv
#endif
