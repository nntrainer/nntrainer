// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cl_tensor_backing_pool.h
 * @date    27 May 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Process-singleton registry mapping logical tensor names to
 *          tv::TensorBacking instances, so GPU layers can hand off
 *          activations across layer boundaries without going through
 *          nntrainer::Tensor's CPU memory path.
 *
 *          Paper §3.6 (fused RoPE+Q/K/V + fused RMSNorm+residual) needs
 *          this — the output of one fused kernel is the input of the
 *          next, and both kernels read/write cl_mem directly.
 *
 *          Lifetime: the pool is process-global. Backings are reference-
 *          counted via shared_ptr. A backing inserted under a tensor
 *          name is reachable to any GPU layer in the same process until
 *          all owners drop their references. Model teardown should
 *          explicitly clear() the pool (no automatic gc — backings
 *          outlive the layer that produced them by design, since the
 *          consumer might still hold a reference).
 *
 *          This is intentionally minimal — single string-keyed map, no
 *          generation count, no inter-thread synchronization beyond a
 *          mutex around the registry itself. Real inter-layer
 *          dependency tracking lives in nntrainer's NetworkGraph; this
 *          pool just provides a name → backing lookup.
 */
#ifndef __NNTRAINER_CL_TENSOR_BACKING_POOL_H__
#define __NNTRAINER_CL_TENSOR_BACKING_POOL_H__

#include "cl_tensor_view.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace nntrainer::tv {

/**
 * @brief Process-singleton name → tv::TensorBacking registry used to hand
 *        GPU activations across layer boundaries without a CPU round trip.
 */
class TensorBackingPool {
public:
  /// Process-global singleton.
  static TensorBackingPool &Global();

  /// Register a backing under a tensor name. Replaces any existing entry.
  /// Returns the inserted shared_ptr so the caller has a strong reference.
  std::shared_ptr<TensorBacking> set(const std::string &name,
                                     std::shared_ptr<TensorBacking> backing);

  /// Look up a backing by name. Returns nullptr if not registered.
  std::shared_ptr<TensorBacking> get(const std::string &name) const;

  /// Remove an entry. No-op if not present.
  void erase(const std::string &name);

  /// Drop all entries (model teardown). After this every previously
  /// returned shared_ptr stays valid until the last reference drops.
  void clear();

  /// For diagnostics: how many entries currently registered.
  size_t size() const;

private:
  TensorBackingPool() = default;
  mutable std::mutex mtx_;
  std::unordered_map<std::string, std::shared_ptr<TensorBacking>> map_;
};

} // namespace nntrainer::tv

#endif // __NNTRAINER_CL_TENSOR_BACKING_POOL_H__
