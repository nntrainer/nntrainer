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

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <basic_planner.h>
#include <engine.h>
#include <llm_util.hpp> // causallm_engine() - the RESOLVED engine name
#include <mem_allocator.h>

namespace causallm {

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

  layer_caches_.resize(num_layers);

  // GPU-resident KV cache: when this model's engine has an allocator that can
  // hand its pointers to an OpenCL kernel, allocate the per-layer K/V out of a
  // MemoryPool on that allocator so their MemoryData reports isSVM()=true.
  // NNTR_GPU_SVM_POOL=0 reverts to the host cache for A/B.
  //
  // This is NOT an optimisation, it is a correctness precondition. mha_core
  // gates its whole GPU attention chain on `svm_ok` (every one of q/k/v/o
  // SVM-resident); with a plain-host cache that gate can never pass, so
  // attention always falls to the host GEMM -- and the host GEMM reads the
  // cache through host pointers while the K/V *producers* (the wk/wv FC
  // outputs, class GPU_CLMEM once any node is engine-stamped) write on the
  // device. The result is a coherent-looking but stale KV plane.
  //
  // The plane is chosen by CAPABILITY, never by backend name. MemAllocator::
  // isSVM() is the predicate core itself uses for exactly this question
  // (memory_pool.cpp stamps residency with allocator_->isSVM()), and its
  // contract is narrower than "unified memory": it means "this pointer may be
  // handed to an OpenCL kernel via clSetKernelArgSVMPointer" -- which is
  // precisely what mha_core's svm_ok gate requires. So:
  //   * ClSVMAllocator derives true (host-addressable && device-visible), and
  //     its own comment records that this is what replaced the former
  //     getName()=="gpu-svm" test. The gpu engine therefore selects the same
  //     allocator object it selected before, and the measured
  //     svm(q/k/v/o) 1001 -> 1111 result is preserved by construction.
  //   * CudaMemAllocator overrides isSVM() to false on purpose even though
  //     CUDA UVM is host-coherent, so engine=cuda keeps the host cache without
  //     this file ever spelling "cuda".
  //   * the plain host MemAllocator is not device-visible, so engine=cpu keeps
  //     the host cache for the same structural reason.
  //   * any future backend answers for itself instead of being enumerated
  //     here, which is what makes this win portable rather than OpenCL-only.
  //
  // The engine name is the RESOLVED one from causallm_engine() -- the same
  // value every node's engine= property carries -- not a raw getenv, so the
  // cache plane cannot disagree with the graph the caches are attached to
  // (a raw NNTR_ENGINE read also treats "gpu on a build without OpenCL" as
  // gpu, which the resolver refuses loudly).
  std::shared_ptr<nntrainer::MemAllocator> svm_alloc;
  const char *_svm_pool_env = std::getenv("NNTR_GPU_SVM_POOL");
  const bool svm_pool_on = !_svm_pool_env || std::atoi(_svm_pool_env) != 0;
  if (svm_pool_on) {
    const std::string engine = causallm_engine();
    auto allocs = nntrainer::Engine::Global().getAllocators();
    auto it = allocs.find(engine);
    if (it != allocs.end() && it->second && it->second->isSVM())
      svm_alloc = it->second;

    if (std::getenv("NNTR_KVSVM_TRACE")) {
      std::fprintf(
        stderr,
        "[kvsvm] engine=%s allocators=%zu found=%d name=%s "
        "isSVM=%d selected=%d\n",
        engine.c_str(), allocs.size(), (int)(it != allocs.end()),
        (it != allocs.end() && it->second) ? it->second->getName().c_str()
                                           : "(none)",
        (int)(it != allocs.end() && it->second && it->second->isSVM()),
        (int)(svm_alloc != nullptr));
      for (auto &kv : allocs)
        std::fprintf(stderr, "[kvsvm]   key=%s name=%s isSVM=%d\n",
                     kv.first.c_str(),
                     kv.second ? kv.second->getName().c_str() : "(null)",
                     (int)(kv.second && kv.second->isSVM()));
      std::fflush(stderr);
    }
  }

  const size_t elem_size =
    (dtype == ml::train::TensorDim::DataType::FP16) ? 2u : 4u;

  if (svm_alloc) {
    svm_pool_ = std::make_shared<nntrainer::MemoryPool>(svm_alloc);
    std::vector<unsigned int> tokens;
    tokens.reserve((size_t)num_layers * 2);
    // All caches are live for the whole run; BasicPlanner gives each its own
    // (non-overlapping) region so the total pool is the sum of them.
    for (unsigned int i = 0; i < num_layers; ++i) {
      const size_t bytes =
        (size_t)batch_size * max_seq_len * kv_widths_[i] * elem_size;
      tokens.push_back(svm_pool_->requestMemory(bytes, 1, 2)); // key
      tokens.push_back(svm_pool_->requestMemory(bytes, 1, 2)); // value
    }
    svm_pool_->planLayout(nntrainer::BasicPlanner());
    svm_pool_->allocate();

    for (unsigned int i = 0; i < num_layers; ++i) {
      ml::train::TensorDim cache_dim(
        {batch_size, 1, max_seq_len, kv_widths_[i]}, {format, dtype});
      layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, false);
      layer_caches_[i].key_cache.setData(svm_pool_->getMemory(tokens[2 * i]), 0,
                                         false);
      layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, false);
      layer_caches_[i].value_cache.setData(
        svm_pool_->getMemory(tokens[2 * i + 1]), 0, false);
    }
    return;
  }

  for (unsigned int i = 0; i < num_layers; ++i) {
    ml::train::TensorDim cache_dim({batch_size, 1, max_seq_len, kv_widths_[i]},
                                   {format, dtype});
    layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, true);
    layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, true);
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
