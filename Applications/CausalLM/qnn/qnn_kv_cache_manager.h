#ifndef __QNN_KV_CACHE_MANAGER_H__
#define __QNN_KV_CACHE_MANAGER_H__

#include "generate_qnn_utils.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace causallm {

class QnnKvCacheManager {
public:
  void clear();

  void addLayerRowLength(int row_length);
  int addGenerationCache(const std::string &name, uint8_t *data, int byte_size,
                         int row_length, bool is_key);
  void addPrefillCache(uint8_t *data, int byte_size, int row_length,
                       int generation_index, bool is_key);

  const std::unordered_map<std::string, int> &generationIndexByName() const {
    return generation_kv_index_by_name_;
  }

  void setPrefillOutputBindings(std::vector<QnnKvOutputBinding> bindings);
  void setGenerationOutputBindings(std::vector<QnnKvOutputBinding> bindings);

  size_t generationCacheCount() const { return generation_caches_.size(); }
  size_t prefillCacheCount() const { return prefill_caches_.size(); }
  size_t prefillOutputBindingCount() const {
    return prefill_output_kv_bindings_.size();
  }
  size_t generationOutputBindingCount() const {
    return generation_output_kv_bindings_.size();
  }

  int length() const { return kv_len_; }
  void setLength(int length);
  void advance(int delta);

  void reset();
  void resetPrefillInputs();
  void syncGenerationToPrefill();

  void appendPrefillOutputs(const std::vector<IO_TensorType> &step_outputs,
                            int target_position, int rows, int src_row_length,
                            const std::string &graph_name);
  void appendGenerationOutputs(const std::vector<IO_TensorType> &step_outputs,
                               int target_position, int rows,
                               int src_row_length,
                               const std::string &graph_name);

  // DDTree accept: append ONLY the accepted tree nodes to the TARGET cache at
  // ring slots, so the committed length can grow past the sliding-window
  // capacity. Accepted node k (verify-batch column accepted_indices[k]) is
  // written at absolute position base_position+k, stored at slot
  // (base_position+k) % seq_cap[layer] — the sliding layers wrap (dropping the
  // oldest), the full layer never wraps in context. Keys are head-major
  // [.,.,head_dim,seq] (one column gathered per head_dim from the verify output
  // [head_dim, src_row_length]); values are seq-major [.,.,seq,head_dim] (one
  // row gathered from [src_row_length, head_dim]). UFIXED8 (1 byte/elem) target
  // path only. Mirrors gauss4.cpp sd_target.cpp acceptAndAppend(); replaces the
  // append-whole-tree + compact_cache_by_indices() path (which cannot ring-wrap).
  // Does NOT change kv_len_ (the caller sets the new length).
  void appendAcceptedGenerationOutputsRing(
    const std::vector<IO_TensorType> &step_outputs,
    const std::vector<int32_t> &accepted_indices, int base_position,
    int src_row_length, const std::string &graph_name);

  void save(const std::string &path, const std::string &architecture) const;
  void load(const std::string &path, const std::string &architecture,
            int max_length);

  // ─── DDTree speculative decoding support ───
  void reserve_for_tree_tail(int num_tree_nodes);
  // Compact the appended verify window [past_length, past_length+window_length)
  // of the TARGET cache, keeping only keep_indices (relative to the window, must
  // be strictly increasing) at the front [past_length, past_length+keep). Layout
  // aware: keys are head-major [.,.,head_dim,seq] (gathered per head_dim column),
  // values are seq-major [.,.,seq,head_dim] (gathered as contiguous rows). Throws
  // for ring-wrapped sliding layers (past+window > seq capacity) until
  // sliding-window-aware compaction lands. kv_len_ is NOT changed (the caller
  // sets the new length).
  void compact_cache_by_indices(const std::vector<int32_t> &keep_indices,
                                int past_length, int window_length);
  void commit_tree_tail(int num_accepted_from_tree);

  int get_committed_length() const { return committed_length_; }
  int get_tree_tail_start() const { return tree_tail_start_; }
  int get_tree_tail_length() const { return kv_len_ - tree_tail_start_; }

  // Per-layer bytes-per-position override for append (process_value/key
  // `column`). Default (empty) => kQnnKvNumColumns (128), correct for the
  // target's UFIXED8 head_dim=128. The draft cache is 16-bit (head_dim=128 ⇒
  // 256 bytes/position), so it sets this to route through the seq-major
  // contiguous path with the right byte stride. `columns.size()` must equal the
  // layer count. See append_outputs_to_kv_cache(kv_columns).
  void set_kv_columns_per_layer(std::vector<int> columns);

  // Seq-major (contiguous) compaction of the appended tree tail for caches
  // whose every K/V tensor is laid out [.,.,seq,head_dim] — i.e. the draft
  // cache (draft keys are seq-major, unlike target keys). Keeps only
  // keep_indices (relative to the appended window starting at
  // committed_length_) and shrinks kv_len_ accordingly. The mixed-layout target
  // cache uses compact_cache_by_indices() instead.
  void compact_seq_major_appended_tail(const std::vector<int32_t> &keep_indices);

private:
  struct GenerationCache {
    std::string name;
    uint8_t *data = nullptr;
    int byte_size = 0;
    int row_length = 0;
    bool is_key = false;
  };

  struct PrefillCache {
    uint8_t *data = nullptr;
    int byte_size = 0;
    int row_length = 0;
    int generation_index = -1;
    bool is_key = false;
  };

  std::vector<GenerationCache> generation_caches_;
  std::vector<PrefillCache> prefill_caches_;
  std::vector<int> layer_row_lengths_;
  std::vector<QnnKvOutputBinding> prefill_output_kv_bindings_;
  std::vector<QnnKvOutputBinding> generation_output_kv_bindings_;
  std::unordered_map<std::string, int> generation_kv_index_by_name_;
  int kv_len_ = 0;

  // Per-layer bytes-per-position for append; empty => default 128 (target).
  std::vector<int> kv_columns_per_layer_;

  // DDTree speculative decoding state
  int committed_length_ = 0;  // KV rows up to this position are final (accepted)
  int tree_tail_start_ = 0;   // Start position of unverified draft tree in cache
};

} // namespace causallm

#endif // __QNN_KV_CACHE_MANAGER_H__
