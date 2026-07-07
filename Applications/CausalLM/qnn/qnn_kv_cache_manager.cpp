#include "qnn_kv_cache_manager.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace causallm {
namespace {

constexpr char kKvCacheMagic[] = {'Q', 'A', 'I', 'Q', 'N', 'N', 'K', 'V'};
constexpr uint32_t kKvCacheVersion = 1;
constexpr uint8_t kKvCacheFill = 128;
// KV head_dim / per-position column count (matches generate_qnn_utils
// kQnnKvNumColumns); the head-major key / seq-major value layouts use it.
constexpr int kKvNumColumns = 128;

void write_bytes(std::ofstream &out, const void *data, std::size_t size) {
  out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  if (!out) {
    throw std::runtime_error("Failed to write QNN KV cache");
  }
}

template <typename T> void write_value(std::ofstream &out, const T &value) {
  write_bytes(out, &value, sizeof(T));
}

void write_string(std::ofstream &out, const std::string &value) {
  const uint32_t size = static_cast<uint32_t>(value.size());
  write_value(out, size);
  if (size > 0) {
    write_bytes(out, value.data(), size);
  }
}

void read_bytes(std::ifstream &in, void *data, std::size_t size) {
  in.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
  if (!in) {
    throw std::runtime_error("Failed to read QNN KV cache");
  }
}

template <typename T> T read_value(std::ifstream &in) {
  T value{};
  read_bytes(in, &value, sizeof(T));
  return value;
}

std::string read_string(std::ifstream &in) {
  const uint32_t size = read_value<uint32_t>(in);
  std::string value(size, '\0');
  if (size > 0) {
    read_bytes(in, &value[0], size);
  }
  return value;
}

} // namespace

void QnnKvCacheManager::clear() {
  generation_caches_.clear();
  prefill_caches_.clear();
  layer_row_lengths_.clear();
  prefill_output_kv_bindings_.clear();
  generation_output_kv_bindings_.clear();
  generation_kv_index_by_name_.clear();
  kv_columns_per_layer_.clear();
  kv_len_ = 0;
  committed_length_ = 0;
  tree_tail_start_ = 0;
}

void QnnKvCacheManager::addLayerRowLength(int row_length) {
  if (row_length <= 0) {
    throw std::runtime_error("Invalid QNN KV layer row length");
  }
  layer_row_lengths_.push_back(row_length);
}

int QnnKvCacheManager::addGenerationCache(const std::string &name,
                                          uint8_t *data, int byte_size,
                                          int row_length, bool is_key) {
  if (data == nullptr || byte_size <= 0 || row_length <= 0) {
    throw std::runtime_error("Invalid QNN generation KV cache tensor: " +
                             name);
  }

  const int index = static_cast<int>(generation_caches_.size());
  generation_caches_.push_back({name, data, byte_size, row_length, is_key});
  generation_kv_index_by_name_[name] = index;
  return index;
}

void QnnKvCacheManager::addPrefillCache(uint8_t *data, int byte_size,
                                        int row_length, int generation_index,
                                        bool is_key) {
  if (data == nullptr || byte_size <= 0 || row_length <= 0 ||
      generation_index < 0 ||
      generation_index >= static_cast<int>(generation_caches_.size())) {
    throw std::runtime_error("Invalid QNN prefill KV cache tensor");
  }

  prefill_caches_.push_back(
      {data, byte_size, row_length, generation_index, is_key});
}

void QnnKvCacheManager::setPrefillOutputBindings(
    std::vector<QnnKvOutputBinding> bindings) {
  prefill_output_kv_bindings_ = std::move(bindings);
}

void QnnKvCacheManager::setGenerationOutputBindings(
    std::vector<QnnKvOutputBinding> bindings) {
  generation_output_kv_bindings_ = std::move(bindings);
}

void QnnKvCacheManager::setLength(int length) {
  if (length < 0) {
    throw std::runtime_error("Invalid QNN KV cache length");
  }
  kv_len_ = length;
}

void QnnKvCacheManager::advance(int delta) {
  if (delta < 0 || kv_len_ + delta < 0) {
    throw std::runtime_error("Invalid QNN KV cache length delta");
  }
  kv_len_ += delta;
}

void QnnKvCacheManager::reset() {
  kv_len_ = 0;
  committed_length_ = 0;
  tree_tail_start_ = 0;
  for (const auto &cache : generation_caches_) {
    std::memset(cache.data, kKvCacheFill, cache.byte_size);
  }
  resetPrefillInputs();
}

void QnnKvCacheManager::resetPrefillInputs() {
  for (const auto &cache : prefill_caches_) {
    std::fill_n(cache.data, cache.byte_size, kKvCacheFill);
  }
}

void QnnKvCacheManager::syncGenerationToPrefill() {
  resetPrefillInputs();

  if (kv_len_ <= 0) {
    return;
  }

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(prefill_caches_.size()); i++) {
    const auto &prefill = prefill_caches_[i];
    const int generation_idx = prefill.generation_index;
    const int generation_layer_idx = generation_idx / 4;
    if (generation_idx < 0 ||
        generation_idx >= static_cast<int>(generation_caches_.size()) ||
        generation_layer_idx < 0 ||
        generation_layer_idx >= static_cast<int>(layer_row_lengths_.size())) {
      continue;
    }

    copy_kv_cache_window(prefill.data, prefill.row_length,
                         generation_caches_[generation_idx].data,
                         layer_row_lengths_[generation_layer_idx], kv_len_,
                         prefill.is_key);
  }
}

void QnnKvCacheManager::appendPrefillOutputs(
    const std::vector<IO_TensorType> &step_outputs, int target_position,
    int rows, int src_row_length, const std::string &graph_name) {
  std::vector<uint8_t *> kvs;
  kvs.reserve(generation_caches_.size());
  for (const auto &cache : generation_caches_) {
    kvs.push_back(cache.data);
  }

  append_outputs_to_kv_cache(step_outputs, prefill_output_kv_bindings_, kvs,
                             layer_row_lengths_, target_position, rows,
                             src_row_length, graph_name,
                             kv_columns_per_layer_.empty()
                               ? nullptr
                               : &kv_columns_per_layer_);
}

void QnnKvCacheManager::appendGenerationOutputs(
    const std::vector<IO_TensorType> &step_outputs, int target_position,
    int rows, int src_row_length, const std::string &graph_name) {
  std::vector<uint8_t *> kvs;
  kvs.reserve(generation_caches_.size());
  for (const auto &cache : generation_caches_) {
    kvs.push_back(cache.data);
  }

  append_outputs_to_kv_cache(step_outputs, generation_output_kv_bindings_, kvs,
                             layer_row_lengths_, target_position, rows,
                             src_row_length, graph_name,
                             kv_columns_per_layer_.empty()
                               ? nullptr
                               : &kv_columns_per_layer_);
}

void QnnKvCacheManager::appendAcceptedGenerationOutputsRing(
    const std::vector<IO_TensorType> &step_outputs,
    const std::vector<int32_t> &accepted_indices, int base_position,
    int src_row_length, const std::string &graph_name) {
  for (const auto &binding : generation_output_kv_bindings_) {
    if (binding.output_index < 0 ||
        binding.output_index >= static_cast<int>(step_outputs.size()) ||
        binding.kv_index < 0 ||
        binding.kv_index >= static_cast<int>(generation_caches_.size()) ||
        binding.layer_index < 0 ||
        binding.layer_index >= static_cast<int>(layer_row_lengths_.size()) ||
        (!kv_columns_per_layer_.empty() &&
         binding.layer_index >=
           static_cast<int>(kv_columns_per_layer_.size()))) {
      throw std::runtime_error(graph_name +
                               " output KV binding is out of range");
    }
  }

  const int keep = static_cast<int>(accepted_indices.size());
#pragma omp parallel for
  for (int binding_idx = 0;
       binding_idx < static_cast<int>(generation_output_kv_bindings_.size());
       binding_idx++) {
    const auto &binding = generation_output_kv_bindings_[binding_idx];
    const int dest_row_length = layer_row_lengths_[binding.layer_index];
    const int num_columns = kv_columns_per_layer_.empty()
                              ? kKvNumColumns
                              : kv_columns_per_layer_[binding.layer_index];
    auto output = std::get<uint8_t *>(step_outputs[binding.output_index]);
    auto dest = generation_caches_[binding.kv_index].data;

    for (int k = 0; k < keep; k++) {
      const int a = accepted_indices[k]; // node column in the verify batch
      const int slot =
        (base_position + k) % dest_row_length; // ring slot (sliding wrap)
      if (binding.is_key) {
        // head-major [num_columns=head_dim, dest_row_length=seq]: write one slot
        // column, gathering verify-output column a from [head_dim, src_row_len].
        for (int c = 0; c < num_columns; c++)
          dest[static_cast<size_t>(c) * dest_row_length + slot] =
            output[static_cast<size_t>(c) * src_row_length + a];
      } else {
        // seq-major [dest_row_length=seq, num_columns=head_dim]: write one slot
        // row, gathering verify-output row a from [src_row_len, head_dim].
        std::memcpy(dest + static_cast<size_t>(slot) * num_columns,
                    output + static_cast<size_t>(a) * num_columns, num_columns);
      }
    }
  }
}

void QnnKvCacheManager::save(const std::string &path,
                             const std::string &architecture) const {
  if (path.empty()) {
    throw std::runtime_error("QNN KV cache path is empty");
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open QNN KV cache for writing: " +
                             path);
  }

  write_bytes(out, kKvCacheMagic, sizeof(kKvCacheMagic));
  write_value(out, kKvCacheVersion);
  write_string(out, architecture);
  write_value(out, static_cast<int32_t>(kv_len_));

  write_value(out, static_cast<uint32_t>(layer_row_lengths_.size()));
  for (const int row_length : layer_row_lengths_) {
    write_value(out, static_cast<int32_t>(row_length));
  }

  write_value(out, static_cast<uint32_t>(generation_caches_.size()));
  for (const auto &cache : generation_caches_) {
    write_string(out, cache.name);
    write_value(out, static_cast<int32_t>(cache.byte_size));
    write_value(out, static_cast<int32_t>(cache.row_length));
    write_value(out, static_cast<uint8_t>(cache.is_key ? 1 : 0));
    write_bytes(out, cache.data, static_cast<std::size_t>(cache.byte_size));
  }
}

void QnnKvCacheManager::load(const std::string &path,
                             const std::string &architecture, int max_length) {
  if (path.empty()) {
    throw std::runtime_error("QNN KV cache path is empty");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open QNN KV cache for reading: " +
                             path);
  }

  char magic[sizeof(kKvCacheMagic)]{};
  read_bytes(in, magic, sizeof(magic));
  if (std::memcmp(magic, kKvCacheMagic, sizeof(kKvCacheMagic)) != 0) {
    throw std::runtime_error("Invalid QNN KV cache file");
  }

  const uint32_t version = read_value<uint32_t>(in);
  if (version != kKvCacheVersion) {
    throw std::runtime_error("Unsupported QNN KV cache version");
  }

  const std::string saved_architecture = read_string(in);
  if (saved_architecture != architecture) {
    throw std::runtime_error("QNN KV cache architecture mismatch");
  }

  const int32_t saved_kv_len = read_value<int32_t>(in);
  if (saved_kv_len < 0 || saved_kv_len > max_length) {
    throw std::runtime_error("QNN KV cache length is out of range");
  }

  const uint32_t layer_count = read_value<uint32_t>(in);
  if (layer_count != layer_row_lengths_.size()) {
    throw std::runtime_error("QNN KV cache layer count mismatch");
  }
  for (uint32_t i = 0; i < layer_count; i++) {
    const int32_t row_length = read_value<int32_t>(in);
    if (row_length != layer_row_lengths_[i]) {
      throw std::runtime_error("QNN KV cache layer row length mismatch");
    }
  }

  const uint32_t tensor_count = read_value<uint32_t>(in);
  if (tensor_count != generation_caches_.size()) {
    throw std::runtime_error("QNN KV cache tensor count mismatch");
  }

  std::vector<std::vector<uint8_t>> loaded_tensors;
  loaded_tensors.reserve(tensor_count);
  for (uint32_t i = 0; i < tensor_count; i++) {
    const std::string name = read_string(in);
    const int32_t byte_size = read_value<int32_t>(in);
    const int32_t row_length = read_value<int32_t>(in);
    const uint8_t is_key = read_value<uint8_t>(in);

    const auto &cache = generation_caches_[i];
    if (name != cache.name || byte_size != cache.byte_size ||
        row_length != cache.row_length || (is_key != 0) != cache.is_key) {
      throw std::runtime_error("QNN KV cache tensor metadata mismatch");
    }

    std::vector<uint8_t> tensor(static_cast<std::size_t>(byte_size));
    read_bytes(in, tensor.data(), tensor.size());
    loaded_tensors.push_back(std::move(tensor));
  }

  for (size_t i = 0; i < loaded_tensors.size(); i++) {
    std::memcpy(generation_caches_[i].data, loaded_tensors[i].data(),
                loaded_tensors[i].size());
  }
  kv_len_ = saved_kv_len;
  // Loaded data is fully committed context with no active draft tree tail, so
  // anchor both markers at the restored length (get_tree_tail_length() == 0).
  committed_length_ = saved_kv_len;
  tree_tail_start_ = saved_kv_len;
  resetPrefillInputs();
}

// ─── DDTree speculative decoding support ───

void QnnKvCacheManager::set_kv_columns_per_layer(std::vector<int> columns) {
  // Size must match the layer count: append_outputs_to_kv_cache() indexes
  // (*kv_columns)[binding.layer_index] without its own bounds check, so a
  // mismatch would be an out-of-bounds read.
  if (!layer_row_lengths_.empty() &&
      columns.size() != layer_row_lengths_.size()) {
    throw std::runtime_error(
      "KV columns-per-layer size (" + std::to_string(columns.size()) +
      ") must equal the layer count (" +
      std::to_string(layer_row_lengths_.size()) + ")");
  }
  for (const int c : columns) {
    if (c <= 0) {
      throw std::runtime_error("Invalid KV columns-per-layer value");
    }
  }
  kv_columns_per_layer_ = std::move(columns);
}

void QnnKvCacheManager::reserve_for_tree_tail(int num_tree_nodes) {
  // Mark the logical start of the draft tree tail. The QNN KV buffers are
  // fixed-size and already allocated, so there is no physical reservation to
  // do here; num_tree_nodes is accepted for symmetry with the accept/compact
  // phase (which will validate the appended window length against it) and is
  // intentionally unused for now.
  (void)num_tree_nodes;
  tree_tail_start_ = kv_len_;
}

void QnnKvCacheManager::compact_seq_major_appended_tail(
  const std::vector<int32_t> &keep_indices) {
  // Seq-major (contiguous) compaction: valid only when every K/V tensor is laid
  // out [.,.,seq,head_dim], i.e. one position is a contiguous `column`-byte row
  // at offset position * column. This holds for the whole draft cache (draft
  // keys are seq-major, unlike target keys), so it requires per-layer column
  // (bytes/position) info from set_kv_columns_per_layer().
  if (kv_columns_per_layer_.empty()) {
    throw std::runtime_error(
      "compact_seq_major_appended_tail requires set_kv_columns_per_layer()");
  }

  const int past = committed_length_;     // appended window starts here
  const int tail = kv_len_ - past;        // appended (tree) length
  const int keep = static_cast<int>(keep_indices.size());
  if (tail < 0) {
    throw std::runtime_error("compact_seq_major: kv_len_ < committed_length_");
  }
  if (tail == 0) {
    if (keep != 0) {
      throw std::runtime_error("compact_seq_major: keep with empty tail");
    }
    return;
  }
  if (keep == 0) {
    kv_len_ = past; // nothing kept: drop the whole appended tail
    return;
  }
  if (keep > tail) {
    throw std::runtime_error("compact_seq_major: keep_count exceeds tail");
  }

  // Absolute-slot compaction only works while no layer has ring-wrapped: a
  // position lives at physical offset (past+i)*col, which requires
  // past + tail <= the layer's seq capacity. Draft layers have MIXED capacity
  // (e.g. sliding 1024 vs full 12288), so once a sliding layer wraps, its tail
  // is a window — not addressable by absolute slot (cf. ddtree.py, which
  // refuses to compact windowed caches). Fail loudly until a sliding-window-
  // aware compaction lands (DDTree accept/compact phase).
  for (size_t layer = 0; layer < layer_row_lengths_.size(); ++layer) {
    if (past + tail > layer_row_lengths_[layer]) {
      throw std::runtime_error(
        "compact_seq_major: layer " + std::to_string(layer) +
        " appended window (" + std::to_string(past + tail) +
        ") exceeds seq capacity (" + std::to_string(layer_row_lengths_[layer]) +
        "); sliding-window-aware compaction is not yet implemented");
    }
  }

  // Validate keep_indices is strictly increasing within [0, tail). This is the
  // forward-gather precondition (new_i <= old_i ⇒ dst <= src, so each memcpy is
  // non-overlapping and the destination row is also in bounds once the source
  // is) and matches follow_verified_tree's BFS (parent < child) ordering.
  for (int i = 0; i < keep; ++i) {
    const int32_t v = keep_indices[i];
    if (v < 0 || v >= tail) {
      throw std::runtime_error("compact_seq_major: index outside tail");
    }
    if (i > 0 && v <= keep_indices[i - 1]) {
      throw std::runtime_error(
        "compact_seq_major: keep_indices must be strictly increasing");
    }
  }

  // generation_caches_ are grouped kv_per_layer (key_h0, key_h1, value_h0,
  // value_h1); the column count is per layer.
  constexpr int kv_per_layer = 4;
  if (generation_caches_.size() % kv_per_layer != 0) {
    throw std::runtime_error(
      "compact_seq_major: cache count not a multiple of kv_per_layer");
  }
  for (size_t idx = 0; idx < generation_caches_.size(); ++idx) {
    auto &cache = generation_caches_[idx];
    const size_t layer = idx / kv_per_layer;
    if (layer >= kv_columns_per_layer_.size()) {
      throw std::runtime_error("compact_seq_major: missing column count");
    }
    const int col = kv_columns_per_layer_[layer]; // bytes per position
    // Gather kept rows of the appended window into the front of the window.
    // keep_indices is strictly increasing (validated above), so new_i <= old_i
    // ⇒ dst <= src: a plain forward memcpy per row is safe and in bounds.
    for (int new_i = 0; new_i < keep; ++new_i) {
      const int32_t old_i = keep_indices[new_i];
      if (old_i == new_i) {
        continue; // already in place
      }
      const size_t dst = static_cast<size_t>(past + new_i) * col;
      const size_t src = static_cast<size_t>(past + old_i) * col;
      if (src + col > static_cast<size_t>(cache.byte_size)) {
        throw std::runtime_error("compact_seq_major: out of bounds");
      }
      std::memcpy(cache.data + dst, cache.data + src, col);
    }
  }

  kv_len_ = past + keep;
}

void QnnKvCacheManager::compact_cache_by_indices(
  const std::vector<int32_t> &keep_indices, int past_length,
  int window_length) {
  // Target verify compaction: keep only keep_indices of the appended window,
  // moving them to the window front. Keys are head-major [.,.,head_dim,seq] and
  // values are seq-major [.,.,seq,head_dim] (cf. process_key / process_value).
  const int keep = static_cast<int>(keep_indices.size());
  if (window_length < 0 || past_length < 0)
    throw std::runtime_error("compact: bad past/window");
  if (keep == 0 || window_length == 0 || keep == window_length)
    return; // nothing to gather, or identity
  if (keep > window_length)
    throw std::runtime_error("compact: keep_count exceeds window");
  for (int i = 0; i < keep; ++i) {
    if (keep_indices[i] < 0 || keep_indices[i] >= window_length)
      throw std::runtime_error("compact: index outside window");
    if (i > 0 && keep_indices[i] <= keep_indices[i - 1])
      throw std::runtime_error("compact: keep_indices must be increasing");
  }

  // head_dim columns (== append's num_columns); element bytes from byte_size.
  constexpr int kColumns = kKvNumColumns;
  constexpr int kv_per_layer = 4;
  if (generation_caches_.size() % kv_per_layer != 0)
    throw std::runtime_error("compact: cache count not a multiple of 4");

  for (size_t idx = 0; idx < generation_caches_.size(); ++idx) {
    auto &cache = generation_caches_[idx];
    const size_t layer = idx / kv_per_layer;
    if (layer >= layer_row_lengths_.size())
      throw std::runtime_error("compact: missing layer row length");
    const int seq_cap = layer_row_lengths_[layer];
    // Absolute-offset compaction is only valid before a layer ring-wraps; the
    // sliding layers (seq_cap < full) wrap once committed+window exceeds them.
    if (past_length + window_length > seq_cap)
      throw std::runtime_error(
        "compact: window exceeds seq capacity (" +
        std::to_string(past_length + window_length) + " > " +
        std::to_string(seq_cap) +
        "); sliding-window-aware compaction not yet implemented");
    if (cache.byte_size % (seq_cap * kColumns) != 0)
      throw std::runtime_error("compact: byte size not divisible by seq*cols");
    const int elem = cache.byte_size / (seq_cap * kColumns); // bytes/element

    if (cache.is_key) {
      // Head-major: element (column c, position p) at (c*seq_cap + p)*elem.
      for (int n = 0; n < keep; ++n) {
        const int old = keep_indices[n];
        if (old == n)
          continue;
        for (int c = 0; c < kColumns; ++c) {
          uint8_t *dst =
            cache.data + (static_cast<size_t>(c) * seq_cap + past_length + n) * elem;
          const uint8_t *src =
            cache.data + (static_cast<size_t>(c) * seq_cap + past_length + old) * elem;
          std::memcpy(dst, src, elem);
        }
      }
    } else {
      // Seq-major: position p is a contiguous (kColumns*elem)-byte row.
      const size_t row = static_cast<size_t>(kColumns) * elem;
      for (int n = 0; n < keep; ++n) {
        const int old = keep_indices[n];
        if (old == n)
          continue;
        std::memcpy(cache.data + (static_cast<size_t>(past_length) + n) * row,
                    cache.data + (static_cast<size_t>(past_length) + old) * row,
                    row);
      }
    }
  }
}

void QnnKvCacheManager::commit_tree_tail(int num_accepted_from_tree) {
  // Optional commitment-marker bookkeeping for callers that track committed_
  // length_/tree_tail_start_. The SD target accept path instead uses kv_len_
  // directly (compact_cache_by_indices + setLength), so it does not call this.
  committed_length_ += num_accepted_from_tree;
  tree_tail_start_ = kv_len_;
}

} // namespace causallm
