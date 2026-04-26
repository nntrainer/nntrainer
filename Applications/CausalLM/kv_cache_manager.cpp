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

namespace causallm {

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              unsigned int num_heads_kv, unsigned int head_dim,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format) {
  if (num_layers == 0 || batch_size == 0 || max_seq_len == 0 ||
      num_heads_kv == 0 || head_dim == 0) {
    throw std::invalid_argument(
      "KVCacheManager::allocate: all parameters must be > 0");
  }

  batch_size_ = batch_size;
  max_seq_len_ = max_seq_len;
  num_heads_kv_ = num_heads_kv;
  head_dim_ = head_dim;
  kv_width_ = num_heads_kv * head_dim;
  dtype_ = dtype;
  format_ = format;
  cache_pos_ = 0;

  ml::train::TensorDim cache_dim({batch_size, 1, max_seq_len, kv_width_},
                                 {format, dtype});

  layer_caches_.resize(num_layers);
  for (unsigned int i = 0; i < num_layers; ++i) {
    layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, true);
    layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, true);
    layer_caches_[i].key_cache.setZero();
    layer_caches_[i].value_cache.setZero();
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
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width_},
                                {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width_;
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
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width_},
                                {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width_;
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
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width_}, {format_, dtype_});

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
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width_}, {format_, dtype_});

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
