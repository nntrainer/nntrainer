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

#include <cstdlib>
#include <stdexcept>

#include <basic_planner.h>
#include <engine.h>
#include <mem_allocator.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_mem_allocator.h>
#endif

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

  // GPU-resident KV cache: when the graph runs on the SVM pool
  // (NNTR_GPU_SVM_POOL) and the gpu-svm allocator is available, allocate the
  // per-layer K/V from an SVM MemoryPool so their MemoryData reports
  // isSVM()=true. That is the precondition for mha_core's GPU flash attention
  // path (svm_ok); without it attention falls back to the host (CPU) GEMM and
  // is ~60x slower. Mirrors gpu_native's SVM K/V cache.
  // [engine=gpu fold] the SVM-resident pool is now the default — but ONLY for
  // the OpenCL gpu engine. On a CUDA build the gpu-svm (OpenCL) allocator is
  // also registered, so without the engine guard a cuda run would wrongly bind
  // the KV cache to it; the #if ENABLE_CUDA block below owns the cuda-uvm KV
  // cache. The env proxy (NNTR_GPU_SVM_POOL set ⇒ OpenCL) used to carry that
  // distinction; now it is explicit. NNTR_GPU_SVM_POOL=0 reverts to a host KV
  // cache.
  std::shared_ptr<nntrainer::MemAllocator> svm_alloc;
  const char *_svm_pool_env = std::getenv("NNTR_GPU_SVM_POOL");
  const char *_eng = std::getenv("NNTR_ENGINE");
  const bool svm_pool_on = !_svm_pool_env || std::atoi(_svm_pool_env) != 0;
  const bool gpu_engine =
    !_eng || (std::string(_eng) != "cpu" && std::string(_eng) != "cuda");
  if (svm_pool_on && gpu_engine) {
    auto allocs = nntrainer::Engine::Global().getAllocators();
    auto it = allocs.find("gpu");
    if (it != allocs.end() && it->second &&
        it->second->getName() == "gpu-svm") {
      svm_alloc = it->second;
    }
  }
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // CUDA UVM-resident KV cache: route the per-layer K/V through the cuda-uvm
  // (cudaMallocManaged) allocator so the cache is device-accessible. That lets
  // GPU attention read it without the per-call host->device mirror and lets GPU
  // RoPE write K straight into the device cache -- the precondition for a fully
  // on-GPU decode chain. Same pooled path as the OpenCL SVM cache below.
  // VALUE-checked (=0 disables), same contract as the other SAFE cuda env:
  // CudaContext auto-defaults it to "1" (setenv overwrite=0), so a presence
  // check made =0 impossible to honor -- the only way to force a plain-host KV
  // cache (WDDM, where the UVM setZero host-faults / attention can't reach a
  // managed cache) is an explicit NNTR_CUDA_KV_UVM=0.
  // Device-resident KV (NNTR_CUDA_KV_DEV=1, opt-in): the cache lives in
  // cudaMalloc DEVICE memory instead of UVM/pinned. WDDM (cMA==0) campaign
  // tier: managed KV hangs (remigration storm) and pinned KV pays PCIe on
  // every attention read; device-resident reads at VRAM speed. Steady state
  // has zero host KV-byte touches on the SAFE profile (GPU rope writes K,
  // GPU scalar-mul writes V -- the V-copy gate auto-routes for a device-only
  // destination -- and split-KV flash reads); setZero/save/load stage through
  // cuda::device_memset0/copy_any. Falls back to the UVM path if the cuda
  // engine is not registered. Takes precedence over NNTR_CUDA_KV_UVM.
  const char *kv_dev = std::getenv("NNTR_CUDA_KV_DEV");
  if (!svm_alloc && kv_dev != nullptr && kv_dev[0] == '1') {
    auto allocs = nntrainer::Engine::Global().getAllocators();
    if (allocs.find("cuda") != allocs.end()) {
      svm_alloc =
        std::make_shared<nntrainer::CudaMemAllocator>(/*device_only=*/true);
    }
  }

  const char *kv_uvm = std::getenv("NNTR_CUDA_KV_UVM");
  if (!svm_alloc && kv_uvm != nullptr && kv_uvm[0] != '0') {
    auto allocs = nntrainer::Engine::Global().getAllocators();
    auto it = allocs.find("cuda");
    if (it != allocs.end() && it->second &&
        it->second->getName() == "cuda-uvm") {
      svm_alloc = it->second;
    }
  }
#endif

  const size_t elem_size =
    (dtype == ml::train::TensorDim::DataType::FP16) ? 2u : 4u;

  if (svm_alloc) {
    svm_pool_ = std::make_shared<nntrainer::MemoryPool>(svm_alloc);
    std::vector<unsigned int> tokens;
    tokens.reserve((size_t)num_layers * 2);
    // All caches are live for the whole run; BasicPlanner gives each its own
    // (non-overlapping) region so the total pool is the sum. Size each region
    // per-layer so models with non-uniform KV widths stay correct.
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
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // Device-only KV (NNTR_CUDA_KV_DEV): host setZero would dereference a
      // cudaMalloc pointer -- zero on the device instead. Detected from the
      // pointer itself so the UVM/pinned/SVM paths keep the host memset.
      auto zero_kv = [](nntrainer::Tensor &t) {
        void *ptr = (void *)t.getData<char>();
        const auto md = t.getMemoryData();
        if (md && !md->isHostAddressable()) {
          if (!nntrainer::cuda::device_memset0(ptr, t.bytes()))
            throw std::runtime_error(
              "KVCacheManager: device memset of the KV cache failed");
        } else {
          t.setZero();
        }
      };
      zero_kv(layer_caches_[i].key_cache);
      zero_kv(layer_caches_[i].value_cache);
#else
      layer_caches_[i].key_cache.setZero();
      layer_caches_[i].value_cache.setZero();
#endif
    }
  } else {
    for (unsigned int i = 0; i < num_layers; ++i) {
      ml::train::TensorDim cache_dim(
        {batch_size, 1, max_seq_len, kv_widths_[i]}, {format, dtype});
      layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, true);
      layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, true);
      layer_caches_[i].key_cache.setZero();
      layer_caches_[i].value_cache.setZero();
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

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // Device-only KV: Tensor::save reads on the host -- stage D2H first.
    auto save_slice = [&f](nntrainer::Tensor &slice) {
      void *ptr = (void *)slice.getData<char>();
      const auto md = slice.getMemoryData();
      if (md && !md->isHostAddressable()) {
        nntrainer::Tensor host_t(slice.getDim(), true);
        if (!nntrainer::cuda::copy_any((void *)host_t.getData<char>(), ptr,
                                       host_t.bytes()))
          throw std::runtime_error(
            "KVCacheManager::save: D2H staging of the device KV failed");
        host_t.save(f);
      } else {
        slice.save(f);
      }
    };
    save_slice(k_slice);
    save_slice(v_slice);
#else
    k_slice.save(f);
    v_slice.save(f);
#endif
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

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // Device-only KV: Tensor::read writes on the host -- read into a host
    // temp, then push H2D.
    auto load_slice = [&f](nntrainer::Tensor &slice) {
      void *ptr = (void *)slice.getData<char>();
      const auto md = slice.getMemoryData();
      if (md && !md->isHostAddressable()) {
        nntrainer::Tensor host_t(slice.getDim(), true);
        host_t.read(f);
        if (!nntrainer::cuda::copy_any(
              ptr, (const void *)host_t.getData<char>(), host_t.bytes()))
          throw std::runtime_error(
            "KVCacheManager::load: H2D staging of the device KV failed");
      } else {
        slice.read(f);
      }
    };
    load_slice(k_slice);
    load_slice(v_slice);
#else
    k_slice.read(f);
    v_slice.read(f);
#endif
  }

  cache_pos_ = seq_len;
}

} // namespace causallm
