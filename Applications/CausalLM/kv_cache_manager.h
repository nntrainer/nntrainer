// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   kv_cache_manager.h
 * @date   25 April 2026
 * @brief  KV Cache Manager for externalized KV cache management
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __KV_CACHE_MANAGER_H__
#define __KV_CACHE_MANAGER_H__

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <memory>

#include <memory_pool.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace causallm {

/**
 * @brief KV Cache Manager - manages KV cache externally for all attention
 *        layers in a transformer model.
 *
 * This class owns the KV cache memory and provides tensor views to mha_core
 * layers, replacing the internal cache allocation done by mha_core.
 *
 * Key responsibilities:
 * - Allocate KV cache buffers for all layers
 * - Track the current write position (cache_pos)
 * - Provide write-pointer tensor views for new K/V insertion
 * - Provide full-range tensor views for attention computation
 * - Save/load cache to/from files
 *
 * Future extensions:
 * - Cache eviction policies
 * - Cache compression / quantization
 * - Paged attention support
 */
class KVCacheManager {
public:
  KVCacheManager() = default;
  ~KVCacheManager() = default;

  // Non-copyable, movable
  KVCacheManager(const KVCacheManager &) = delete;
  KVCacheManager &operator=(const KVCacheManager &) = delete;
  KVCacheManager(KVCacheManager &&) = default;
  KVCacheManager &operator=(KVCacheManager &&) = default;

  /**
   * @brief Allocate KV cache for all layers
   * @param[in] num_layers number of attention layers
   * @param[in] batch_size batch size
   * @param[in] max_seq_len maximum sequence length (total cache capacity)
   * @param[in] num_heads_kv number of KV heads (for GQA)
   * @param[in] head_dim dimension per head
   * @param[in] dtype data type for cache tensors
   * @param[in] format tensor format
   */
  void allocate(
    unsigned int num_layers, unsigned int batch_size, unsigned int max_seq_len,
    unsigned int num_heads_kv, unsigned int head_dim,
    ml::train::TensorDim::DataType dtype = ml::train::TensorDim::DataType::FP16,
    ml::train::TensorDim::Format format = ml::train::TensorDim::Format::NCHW);

  /**
   * @brief Allocate KV cache with per-layer KV widths.
   * @param[in] num_layers number of attention layers
   * @param[in] batch_size batch size
   * @param[in] max_seq_len maximum sequence length
   * @param[in] kv_widths per-layer width (num_heads_kv * head_dim)
   * @param[in] dtype data type for cache tensors
   * @param[in] format tensor format
   */
  void allocate(
    unsigned int num_layers, unsigned int batch_size, unsigned int max_seq_len,
    const std::vector<unsigned int> &kv_widths,
    ml::train::TensorDim::DataType dtype = ml::train::TensorDim::DataType::FP16,
    ml::train::TensorDim::Format format = ml::train::TensorDim::Format::NCHW);

  /**
   * @brief Check if the manager has been allocated
   */
  bool isAllocated() const { return !layer_caches_.empty(); }

  /**
   * @brief Get current write position in the cache
   */
  unsigned int getPosition() const { return cache_pos_; }

  /**
   * @brief Set current write position (e.g., after loading pre-computed cache)
   * @param[in] pos new position
   */
  void setPosition(unsigned int pos);

  /**
   * @brief Advance the write position by step_size
   * @param[in] step_size number of positions to advance
   */
  void advance(unsigned int step_size);

  /**
   * @brief Reset position to 0 (for new inference session)
   */
  void reset();

  /**
   * @brief Get the full key cache tensor for a layer (for direct access)
   * @param[in] layer_idx attention layer index
   * @return reference to the full key cache tensor
   */
  nntrainer::Tensor &getKeyCache(unsigned int layer_idx);

  /**
   * @brief Get the full value cache tensor for a layer (for direct access)
   * @param[in] layer_idx attention layer index
   * @return reference to the full value cache tensor
   */
  nntrainer::Tensor &getValueCache(unsigned int layer_idx);

  /**
   * @brief Get a write-pointer view into key cache at current position
   *        for a specific batch and step_size.
   *        This is where new K values should be written.
   * @param[in] layer_idx attention layer index
   * @param[in] batch batch index
   * @param[in] step_size number of tokens to write
   * @return Tensor view pointing to the write location
   */
  nntrainer::Tensor getKeyCacheWriteView(unsigned int layer_idx,
                                         unsigned int batch,
                                         unsigned int step_size);

  /**
   * @brief Get a write-pointer view into value cache at current position
   * @param[in] layer_idx attention layer index
   * @param[in] batch batch index
   * @param[in] step_size number of tokens to write
   * @return Tensor view pointing to the write location
   */
  nntrainer::Tensor getValueCacheWriteView(unsigned int layer_idx,
                                           unsigned int batch,
                                           unsigned int step_size);

  /**
   * @brief Get a read view of key cache from position 0 to (cache_pos +
   * step_size) for attention computation (Q @ K^T).
   * @param[in] layer_idx attention layer index
   * @param[in] batch batch index
   * @param[in] read_len total length to read (typically cache_pos + step_size)
   * @return Tensor view covering [0, read_len)
   */
  nntrainer::Tensor getKeyCacheReadView(unsigned int layer_idx,
                                        unsigned int batch,
                                        unsigned int read_len);

  /**
   * @brief Get a read view of value cache from position 0 to read_len
   * @param[in] layer_idx attention layer index
   * @param[in] batch batch index
   * @param[in] read_len total length to read
   * @return Tensor view covering [0, read_len)
   */
  nntrainer::Tensor getValueCacheReadView(unsigned int layer_idx,
                                          unsigned int batch,
                                          unsigned int read_len);

  /**
   * @brief Save KV cache to file (all layers, up to current position)
   * @param[in] path file path
   */
  void save(const std::string &path) const;

  /**
   * @brief Save KV cache to file up to specified length
   * @param[in] path file path
   * @param[in] seq_len number of positions to save
   */
  void save(const std::string &path, unsigned int seq_len) const;

  /**
   * @brief Load KV cache from file
   * @param[in] path file path
   * @param[in] seq_len number of positions to load
   */
  void load(const std::string &path, unsigned int seq_len);

  /**
   * @brief Get number of layers
   */
  unsigned int getNumLayers() const {
    return static_cast<unsigned int>(layer_caches_.size());
  }

  /**
   * @brief Get maximum sequence length (cache capacity)
   */
  unsigned int getMaxSeqLen() const { return max_seq_len_; }

  /**
   * @brief Get batch size
   */
  unsigned int getBatchSize() const { return batch_size_; }

  /**
   * @brief Get the KV dimension width (num_heads_kv * head_dim)
   */
  unsigned int getKVWidth() const { return num_heads_kv_ * head_dim_; }

  /**
   * @brief [kv-share] Declare, per layer, which EARLIER layer owns the K/V
   *        storage this layer reads. A KV-shared layer (gemma4
   *        `num_kv_shared_layers`) recomputes nothing: it attends over the
   *        source layer's K/V plane. Reproducing the VALUES into a private
   *        slab (what this manager used to do) is byte-for-byte redundant --
   *        for a 35-layer config with num_kv_shared_layers=20, that duplicated
   *        more than half the whole KV plane, all of it device-resident
   *        because the SVM/UVM pool populates every page it reserves.
   *
   *        With a source declared, allocate() gives the layer NO pool token
   *        and NO new storage: its key/value Tensors are aliases (same
   *        MemoryData, same offset) onto the source's. The alias is exactly
   *        the operation the model's bind already performs when it hands the
   *        cache to the graph placeholder, so nothing downstream can tell the
   *        difference except the byte count.
   *
   *        Contract (checked in allocate(), which throws on violation):
   *          - sources[i] < i    -- a source is always allocated first, so one
   *                                 forward pass over the layers suffices;
   *          - sources[i] == -1  -- this layer owns its cache (the default);
   *          - the geometry (cap x width x dtype) of i and sources[i] must be
   *            identical, otherwise the alias would reinterpret the plane.
   *
   *        MUST be called BEFORE allocate(). An empty vector (the default)
   *        means every layer owns its cache -- bit-identical to the
   *        pre-aliasing behaviour.
   *
   *        Derive the vector from the SAME rule the graph builder uses to pick
   *        the shared-attention source (Gemma4Transformer::
   *        getSharedKVSourceLayer(), reached through the
   *        Transformer::getKVSourceLayer() hook). If allocation and graph
   *        building ever disagree, a layer silently attends over the wrong
   *        layer's K/V: fluent, wrong, and crash-free.
   *
   * @param[in] sources per-layer KV source layer id (-1 = owns its cache)
   */
  void setLayerKVSources(std::vector<int> sources) {
    layer_kv_sources_ = std::move(sources);
  }

  /**
   * @brief The layer whose K/V storage layer_idx aliases, or -1 when the layer
   *        owns its own cache.
   * @param[in] layer_idx attention layer index
   */
  int getLayerKVSource(unsigned int layer_idx) const {
    if (layer_idx < layer_kv_sources_.size())
      return layer_kv_sources_[layer_idx];
    return -1;
  }

  /**
   * @brief true when layer_idx's K/V tensors are views onto another layer's
   *        storage, i.e. contribute zero bytes of their own.
   * @param[in] layer_idx attention layer index
   */
  bool isLayerKVAliased(unsigned int layer_idx) const {
    return getLayerKVSource(layer_idx) >= 0;
  }

  /**
   * @brief [kv-window-ring] Set the per-layer physical row capacity. A
   * sliding-window layer under the ring stores only Wcap rows instead of
   * max_seq_len; pass caps[i]=Wcap for those layers and caps[i]=0 (or
   * max_seq_len) for the full ones. Must be called BEFORE allocate(). An empty
   * vector means every layer is full max_seq.
   */
  void setLayerCaps(std::vector<unsigned int> caps) {
    layer_caps_ = std::move(caps);
  }

  /**
   * @brief Physical row capacity of a layer's cache (Wcap for a ring layer,
   * else max_seq_len). The write/read code modulo-indexes against this.
   */
  unsigned int getLayerCap(unsigned int layer_idx) const {
    if (layer_idx < layer_caps_.size() && layer_caps_[layer_idx] > 0)
      return layer_caps_[layer_idx];
    return max_seq_len_;
  }

private:
  /**
   * @brief [kv-share] Validate layer_kv_sources_ against num_layers and the
   *        per-layer geometry, and throw if the declaration is unusable.
   *        Called once at the top of allocate(), i.e. BEFORE any memory is
   *        requested, so a bad map can never reach the pool.
   */
  void validateKVSources(unsigned int num_layers) const;

  /**
   * @brief [kv-share] Point layer `dst`'s K/V tensors at layer `src`'s
   *        storage. Shares the source's MemoryData shared_ptr (so teardown
   *        order is irrelevant on both the pool and the host branch) at the
   *        source's offset -- the same setData() the model bind performs.
   *        Never leaves a tensor empty: `cache_dim` is the layer's own
   *        geometry and is asserted equal to the source's.
   */
  void aliasLayerCache(unsigned int dst, unsigned int src,
                       const ml::train::TensorDim &cache_dim);

  /**
   * @brief [kv-share] One-time witness of the resolved layer -> source map,
   *        the bytes it kept out of the pool, and a live re-check that every
   *        alias actually resolves to its source's address. Emitted at the end
   *        of allocate() whenever sharing is declared.
   */
  void reportKVShare(unsigned int num_layers, unsigned int batch_size,
                     ml::train::TensorDim::DataType dtype,
                     const char *where) const;

  /**
   * @brief Per-layer cache storage
   */
  struct LayerCache {
    nntrainer::Tensor key_cache;   /**< (batch, 1, max_seq_len, kv_width) */
    nntrainer::Tensor value_cache; /**< (batch, 1, max_seq_len, kv_width) */
  };

  std::vector<LayerCache> layer_caches_; /**< per-layer KV caches */

  /**
   * @brief Optional SVM-backed memory pool. When NNTR_GPU_SVM_POOL is set and
   * the GPU (gpu-svm) allocator is available, the KV caches are allocated from
   * this pool so their MemoryData reports isSVM()=true — required for the
   * GPU flash attention path (mha_core). Null on the host (CPU) path.
   */
  std::shared_ptr<nntrainer::MemoryPool> svm_pool_;

  unsigned int cache_pos_ = 0;    /**< current write position */
  unsigned int batch_size_ = 0;   /**< batch size */
  unsigned int max_seq_len_ = 0;  /**< max sequence length */
  unsigned int num_heads_kv_ = 0; /**< number of KV heads */
  unsigned int head_dim_ = 0;     /**< head dimension */
  unsigned int kv_width_ = 0;     /**< num_heads_kv * head_dim */
  std::vector<unsigned int> kv_widths_;
  /** [kv-share] per-layer KV alias source (-1 = owns its cache) */
  std::vector<int> layer_kv_sources_;
  /** [kv-window-ring] per-layer physical row capacity (0 = full max_seq_len) */
  std::vector<unsigned int> layer_caps_;

  ml::train::TensorDim::DataType dtype_ = ml::train::TensorDim::DataType::FP16;
  ml::train::TensorDim::Format format_ = ml::train::TensorDim::Format::NCHW;
};

} // namespace causallm

#endif // __KV_CACHE_MANAGER_H__
