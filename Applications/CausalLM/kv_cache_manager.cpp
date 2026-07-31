// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   kv_cache_manager.cpp
 * @date   25 April 2026
 * @brief  KV Cache Manager implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "kv_cache_manager.h"

#include <stdexcept>

#include <dlfcn.h>

#include <cstdint>
#include <memory>

#include <llm_util.hpp>
#include <memory_data.h>
#include <nntrainer_log.h>

namespace causallm {

namespace {

/**
 * @brief Back the KV cache with rpcmem and register it with the Hexagon cDSP
 *        bridge, so the DSP can read K/V in place.
 *
 * Prerequisite for offloading attention's two matmuls (Q.K^T and scores.V):
 * those read the KV cache, and a DSP kernel cannot dereference an ordinary
 * host pointer at all (see docs/backend_guide/HEXAGON_NPU_PRIMER.md). The
 * alternative to this is memcpy-ing K and V into rpcmem per layer per
 * forward pass (~35 MB per 308-token prefill); allocating the cache in
 * rpcmem once and registering it is strictly better, since the cache is
 * allocated once and lives for the whole session.
 *
 * Resolved by dlopen/dlsym rather than by linking nntrainer's
 * HexagonRpcAllocator, deliberately: ENABLE_HEXAGON_CDSP is not propagated to
 * the CausalLM build (it lives only in nntrainer's own Android.mk /
 * extra_defines), so a compile-time dependency on hexagon_rpc_allocator.h
 * would break any build of this app against an nntrainer that was configured
 * without -Denable-hexagon-cdsp=true. Same dlopen pattern already used by
 * hexagon_compute_ops.cpp and hexagon_rpc_allocator.cpp.
 *
 * Returns nullptr if anything is unavailable, in which case the caller keeps
 * the ordinary host allocation - the CPU attention path stays correct, it
 * just cannot be offloaded.
 */
class KVCacheRpcMem {
public:
  static KVCacheRpcMem &global() {
    static KVCacheRpcMem inst;
    return inst;
  }

  bool usable() const { return alloc_ && register_pool_; }

  /** @brief rpcmem_alloc + register with the bridge; nullptr on any failure */
  void *allocAndRegister(size_t bytes) {
    if (!usable()) {
      return nullptr;
    }
    void *p = alloc_(kHeapIdSystem, kDefaultFlags, static_cast<int>(bytes));
    if (!p) {
      ml_logw("KVCacheManager: rpcmem_alloc(%zu) failed; KV cache stays on "
              "host memory (attention cannot be offloaded)",
              bytes);
      return nullptr;
    }
    if (register_pool_(p, bytes) != 0) {
      ml_logw("KVCacheManager: bridge rejected KV cache pool %p (%zu bytes); "
              "keeping it but the DSP will not see it",
              p, bytes);
    }
    return p;
  }

private:
  using AllocFn = void *(*)(int, uint32_t, int);
  using RegisterFn = int (*)(const void *, size_t);

  static constexpr int kHeapIdSystem = 25;
  static constexpr int kDefaultFlags = 1;

  KVCacheRpcMem() {
    void *rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rpc) {
      ml_logw("KVCacheManager: dlopen(libcdsprpc.so) failed: %s", dlerror());
      return;
    }
    alloc_ = reinterpret_cast<AllocFn>(dlsym(rpc, "rpcmem_alloc"));

    void *bridge = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!bridge) {
      ml_logw("KVCacheManager: dlopen(libggml-hexagon.so) failed: %s",
              dlerror());
      return;
    }
    register_pool_ = reinterpret_cast<RegisterFn>(
      dlsym(bridge, "nntr_htp_bridge_register_activation_pool"));
  }

  AllocFn alloc_ = nullptr;
  RegisterFn register_pool_ = nullptr;
};

} // namespace

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              unsigned int num_heads_kv, unsigned int head_dim,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format) {
  if (num_heads_kv == 0 || head_dim == 0) {
    throw std::invalid_argument(
      "KVCacheManager::allocate: all parameters must be > 0");
  }

  allocate(num_layers, batch_size, max_seq_len,
           std::vector<unsigned int>(num_layers, num_heads_kv * head_dim),
           dtype, format);

  num_heads_kv_ = num_heads_kv;
  head_dim_ = head_dim;
}

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              const std::vector<unsigned int> &kv_widths,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format) {
  if (num_layers == 0 || batch_size == 0 || max_seq_len == 0 ||
      kv_widths.size() != num_layers) {
    throw std::invalid_argument(
      "KVCacheManager::allocate: invalid layer, batch, or KV width count");
  }

  for (auto kv_width : kv_widths) {
    if (kv_width == 0) {
      throw std::invalid_argument(
        "KVCacheManager::allocate: KV widths must be > 0");
    }
  }

  batch_size_ = batch_size;
  max_seq_len_ = max_seq_len;
  num_heads_kv_ = 0;
  head_dim_ = 0;
  kv_width_ = kv_widths[0];
  kv_widths_ = kv_widths;
  dtype_ = dtype;
  format_ = format;
  cache_pos_ = 0;

  // Back the cache with rpcmem only when the cDSP engine is actually in use,
  // so a plain CPU run does not consume the (scarce) rpcmem/CMA pool for no
  // reason. Falls back to ordinary host allocation whenever unavailable.
  const bool use_rpcmem =
    useHexagonCdsp() && KVCacheRpcMem::global().usable();

  layer_caches_.resize(num_layers);

  // ONE rpcmem region for every layer's K and V, sub-allocated by offset -
  // deliberately not one allocation per tensor. HTP_OP_MAX_BUFS and
  // HTP_MAX_MMAPS are both 16 (S2 finding 9), so 2*num_layers separate
  // registrations is the exact anti-pattern S8's pooled weight arenas exist
  // to avoid: it burns a DSP mapping-cache slot per tensor and a
  // fastrpc_mmap per tensor. The bridge's find_ext_pool() matches any pointer
  // *inside* a registered range and add_tensor derives its wire offset as
  // (data - sbuf->base), so sub-allocated pointers need no extra registration.
  std::shared_ptr<nntrainer::MemoryData> pool;
  std::vector<size_t> k_off(num_layers), v_off(num_layers);
  if (use_rpcmem) {
    const size_t elem_sz = ml::train::TensorDim({1, 1, 1, 1}, {format, dtype})
                             .getDataTypeSize();
    // 128-byte (HVX) alignment between sub-allocations, expressed in elements.
    const size_t align_elems = elem_sz ? (128 / elem_sz) : 1;
    auto round_up = [align_elems](size_t n) {
      return align_elems ? ((n + align_elems - 1) / align_elems) * align_elems
                         : n;
    };

    size_t total_elems = 0;
    for (unsigned int i = 0; i < num_layers; ++i) {
      const size_t per =
        round_up(static_cast<size_t>(batch_size) * max_seq_len * kv_widths_[i]);
      k_off[i] = total_elems;
      total_elems += per;
      v_off[i] = total_elems;
      total_elems += per;
    }

    if (void *base =
          KVCacheRpcMem::global().allocAndRegister(total_elems * elem_sz)) {
      pool = std::make_shared<nntrainer::MemoryData>(base);
      ml_logi("KVCacheManager: KV cache in rpcmem, one %zu MiB pool for %u "
              "layers (DSP-visible)",
              (total_elems * elem_sz) / (1024 * 1024), num_layers);
    } else {
      ml_logw("KVCacheManager: falling back to host memory for the KV cache; "
              "attention cannot be offloaded to the cDSP");
    }
  }

  for (unsigned int i = 0; i < num_layers; ++i) {
    ml::train::TensorDim cache_dim({batch_size, 1, max_seq_len, kv_widths_[i]},
                                   {format, dtype});

    if (pool) {
      // alloc_now=false: the buffer comes from setData below, so do not let
      // the Tensor allocate host memory it would then leak. offset is in
      // elements, not bytes (see FloatTensor::getData).
      layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, false);
      layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, false);
      layer_caches_[i].key_cache.setData(pool, k_off[i], /*init=*/true);
      layer_caches_[i].value_cache.setData(pool, v_off[i], /*init=*/true);
    } else {
      layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, true);
      layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, true);
    }
  }
}

void KVCacheManager::setPosition(unsigned int pos) {
  if (pos > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::setPosition: pos exceeds max_seq_len");
  }
  cache_pos_ = pos;
}

void KVCacheManager::advance(unsigned int step_size) {
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::advance: position would exceed max_seq_len");
  }
  cache_pos_ += step_size;
}

void KVCacheManager::reset() { cache_pos_ = 0; }

nntrainer::Tensor &KVCacheManager::getKeyCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range("KVCacheManager::getKeyCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].key_cache;
}

nntrainer::Tensor &KVCacheManager::getValueCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range("KVCacheManager::getValueCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].value_cache;
}

nntrainer::Tensor KVCacheManager::getKeyCacheWriteView(unsigned int layer_idx,
                                                       unsigned int batch,
                                                       unsigned int step_size) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheWriteView: invalid layer_idx");
  }
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheWriteView: would exceed max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].key_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width;
  return cache.getSharedDataTensor(step_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getValueCacheWriteView(
  unsigned int layer_idx, unsigned int batch, unsigned int step_size) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheWriteView: invalid layer_idx");
  }
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheWriteView: would exceed max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].value_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width;
  return cache.getSharedDataTensor(step_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getKeyCacheReadView(unsigned int layer_idx,
                                                      unsigned int batch,
                                                      unsigned int read_len) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheReadView: invalid layer_idx");
  }
  if (read_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheReadView: read_len exceeds max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].key_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getValueCacheReadView(unsigned int layer_idx,
                                                        unsigned int batch,
                                                        unsigned int read_len) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheReadView: invalid layer_idx");
  }
  if (read_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheReadView: read_len exceeds max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].value_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}

void KVCacheManager::save(const std::string &path) const {
  save(path, cache_pos_);
}

void KVCacheManager::save(const std::string &path, unsigned int seq_len) const {
  if (layer_caches_.empty()) {
    throw std::runtime_error("KVCacheManager::save: not allocated");
  }
  if (seq_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::save: seq_len exceeds max_seq_len");
  }

  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("KVCacheManager::save: cannot open file: " + path);
  }

  for (const auto &lc : layer_caches_) {
    ml::train::TensorDim save_dim = lc.key_cache.getDim();
    save_dim.height(seq_len);

    nntrainer::Tensor k_slice = const_cast<nntrainer::Tensor &>(lc.key_cache)
                                  .getSharedDataTensor(save_dim, 0, true);
    nntrainer::Tensor v_slice = const_cast<nntrainer::Tensor &>(lc.value_cache)
                                  .getSharedDataTensor(save_dim, 0, true);

    k_slice.save(f);
    v_slice.save(f);
  }
}

void KVCacheManager::load(const std::string &path, unsigned int seq_len) {
  if (layer_caches_.empty()) {
    throw std::runtime_error("KVCacheManager::load: not allocated");
  }
  if (seq_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::load: seq_len exceeds max_seq_len");
  }

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("KVCacheManager::load: cannot open file: " + path);
  }

  for (auto &lc : layer_caches_) {
    ml::train::TensorDim load_dim = lc.key_cache.getDim();
    load_dim.height(seq_len);

    nntrainer::Tensor k_slice =
      lc.key_cache.getSharedDataTensor(load_dim, 0, true);
    nntrainer::Tensor v_slice =
      lc.value_cache.getSharedDataTensor(load_dim, 0, true);

    k_slice.read(f);
    v_slice.read(f);
  }

  cache_pos_ = seq_len;
}

} // namespace causallm
