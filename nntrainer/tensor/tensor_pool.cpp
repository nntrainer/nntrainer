// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   tensor_pool.cpp
 * @date   19 Aug 2021
 * @brief  This is TensorPool for all requested tensors
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 * @todo   add checks for request/updates that finalize is not done
 * @todo   check before allocate that finalize is done
 */

#include <cstdlib>
#include <memory_pool.h>
#include <nntrainer_log.h>
#include <residency_planner.h>
#include <residency_policy.h>
#include <tensor.h>
#include <tensor_pool.h>
#include <tensor_wrap_specs.h>
#include <util_func.h>

namespace nntrainer {

/**
 * @brief The single process-wide residency policy instance. Out of line here
 * so the library and the application that declares the model's residency
 * boundaries share one copy across the shared-object boundary.
 */
ResidencyPolicy &ResidencyPolicy::global() {
  static ResidencyPolicy instance;
  return instance;
}

/**
 * @brief     Request tensor with the given spec
 *
 * @note returns empty tensor which will be filled when allocate is called.
 * @note we assume that the caller checks if the exec_order and lifespan are
 * compatible.
 */
Tensor *TensorPool::request(const std::string &name, const TensorDim &dim,
                            const std::vector<unsigned int> &exec_order,
                            TensorLifespan lifespan, const Initializer &init,
                            bool is_weight_grad,
                            ml::train::LayerComputeEngine engine) {

  bool is_virtual = lifespan == TensorLifespan::VIRTUAL;
  lifespan = is_virtual ? TensorLifespan::UNMANAGED : lifespan;
  return registerRequestSpec(
    {is_weight_grad,
     std::make_unique<Tensor>(dim, false, init, name,
                              QScheme::PER_CHANNEL_AFFINE, is_virtual),
     TensorPool::SourceDetails{0, lifespan, exec_order, {}, engine}});
}

/**
 * @brief     Request tensor with the given spec
 *
 * @note returns empty tensor which will be filled when allocate is called.
 */
Tensor *TensorPool::placeholder(const std::string &name, const TensorDim &dim) {
  return request(name, dim, {}, TensorLifespan::UNMANAGED);
}

/**
 * @brief     Request tensor which has been already requested with the given
 * spec
 *
 * @note returns empty tensor which will be filled when allocate is called.
 * @note we assume that the caller checks if the exec_order and lifespan are
 * compatible.
 */
Tensor *TensorPool::view(const std::string &name, const std::string &reference,
                         const TensorDim &dim,
                         const std::vector<unsigned int> &exec_order,
                         TensorLifespan lifespan, const size_t offset,
                         ml::train::LayerComputeEngine consumer_engine) {
  auto &spec = getSourceSpec(reference);

  NNTR_THROW_IF(spec.tensor->getDataType() != dim.getDataType() ||
                  spec.tensor->getFormat() != dim.getFormat(),
                std::invalid_argument)
    << "view tensor type != source tensor type, view tensor type: " << dim
    << " source tensor: " << spec.tensor->getDim();

  unsigned adjusted_offset = std::visit(
    [](const auto &s) {
      using T = std::decay_t<decltype(s)>;
      if constexpr (std::is_same_v<T, SourceDetails>) {
        return 0u;
      } else if constexpr (std::is_same_v<T, DependentDetails>) {
        return s.offset;
      }
      return 0u;
    },
    pool[name_map.at(reference)].details);
  adjusted_offset += offset;

  NNTR_THROW_IF(spec.tensor->getDim().getDataLen() <
                  adjusted_offset + dim.getDataLen(),
                std::invalid_argument)
    << "view tensor size + offset > source tensor size, view tensor size: "
    << dim.getDataLen() << " offset: " << adjusted_offset
    << " source tensor: " << spec.tensor->getDim().getDataLen()
    << " name: " << spec.tensor->getName();

  expandLifespan(spec, exec_order, lifespan);
  {
    auto &src_details = std::get<SourceDetails>(spec.details);
    src_details.dependents.push_back(pool.size());
    /** A consumer that does not run on the device pulls the source back onto
     *  the shared plane: the placement has to be one every reader can reach. */
    src_details.all_consumers_device &=
      (consumer_engine == ml::train::LayerComputeEngine::GPU);
    src_details.view_count++;
  }

  /** @note below invalidates spec reference */
  /** @note in case of view of view, internal datastructure saves the src to
   * view index, not view to view reference in order to flatten depth */
  auto parent_idx = name_map.at(spec.tensor->getName());

  /** @note default is_weight_grad for view is false. view is for the
   * activation. */
  return registerRequestSpec(
    {false, std::make_unique<Tensor>(dim, false, Initializer::NONE, name),
     TensorPool::DependentDetails{parent_idx, adjusted_offset}});
}

/**
 * @brief finalize the requested tensors
 *
 * @details finalize the requested tensors, request memory for them and plan
 * layout for their allocations.
 */
void TensorPool::finalize(const MemoryPlanner &planner,
                          unsigned int start_order, unsigned int end_order) {
  mem_pool->clear();
  unsigned int bytes_requested = 0;
  /** if execution order is PERSIST_END_ORDER, then we think it has another
   * execution order for gradient clipping
   *  persist_end_order is for checking if the end order is updated */
  bool persist_end_order = false;
  unsigned int old_end_order = end_order;

  for (auto &spec : pool) {

    auto details = std::get_if<SourceDetails>(&spec.details);
    if (!details || details->lifespan == TensorLifespan::UNMANAGED ||
        details->exec_order.empty()) {
      continue;
    }
    details->token = 0;

    /**
     * 1. create the validity ranges for the all the requested tensors.
     * validity_start/validity_end should be a value in the exec order of the
     * given tensor or a value out of range so as to not request memory for
     * this tensor
     */
    unsigned int validity_start = end_order + 1;
    for (unsigned int idx = 0; idx < details->exec_order.size(); idx++) {
      if (details->exec_order[idx] >= start_order)
        validity_start = std::min(validity_start, details->exec_order[idx]);
      /** This is to enforce not to reach if the execution order is greater
       * than backwarding end order. e.g., for the input layer, the
       * backwarding is not reached but the exeuction order is assigned.
       * */
      if (details->exec_order[idx] > old_end_order &&
          details->exec_order[idx] != PERSIST_END_ORDER) {
        details->exec_order[idx] = PERSIST_END_ORDER - 1;
      }
    }
    unsigned int validity_end = validity_start;
    for (unsigned int idx = 0; idx < details->exec_order.size(); idx++) {
      if (details->exec_order[idx] == PERSIST_END_ORDER) {
        if (!persist_end_order) {
          end_order = end_order + 1;
          persist_end_order = true;
        }
        validity_end = end_order;
        details->exec_order[idx] = validity_end;
        break;
      }

      if (details->exec_order[idx] <= end_order) {
        validity_end = std::max(validity_end, details->exec_order[idx]);
      }
    }
    /**
     * use lifespan to update the validity.
     * if the validity is long term, the tensor must stay valid for the
     * complete duration.
     */
    if (isTensorLongTerm(details->lifespan)) {
      validity_start = start_order;
      validity_end = end_order;
    }

    /** 2. for each tensor request if it is in the provided range */
    if (validity_end < start_order || validity_start > end_order) {
      continue;
    }

    /**
     * 3. requestMemory for all the tensors and set their tokens
     * @note +1 is to make the validity_end exlusive in the interval range
     */
    details->token = mem_pool->requestMemory(
      spec.tensor->getMemoryBytes(), validity_start, validity_end + 1,
      details->exec_order, details->lifespan, spec.is_weight_grad);
#ifdef DEBUG
    if (details->token == 0)
      throw std::runtime_error("Received invalid token from memory pool");
#endif

    bytes_requested += spec.tensor->getMemoryBytes();
  }

  /** 4. finalizeLayout for the memory pool. */
  if (bytes_requested > 0) {
    double efficiency = mem_pool->planLayout(planner);
    ml_logd("Memory layout efficiency = %lf", efficiency);
  }
}

/**
 * @brief Set the batch size for the inputs/outputs of the layers
 */
void TensorPool::setBatchSize(const std::string &name, unsigned int batch) {
  if (name_map.find(name) == name_map.end())
    throw std::invalid_argument("Requested tensor not found");

  pool[name_map[name]].tensor->updateBatch(batch);
}

/**
 * @brief Allocate memory for all the managed tensors
 */
void TensorPool::allocate(bool init) {
  if (minMemoryRequirement() == 0)
    return;
  mem_pool->allocate();

  /** Configure the residency planner for this pool. What the allocator can
   *  produce decides what placements are available at all; the application's
   *  declared boundaries decide the two places the engine heuristic cannot
   *  see (see residency_policy.h). */
  const auto &policy = ResidencyPolicy::global();
  ResidencyPlanner planner;
  planner.device_backed = allocator_ && allocator_->isDeviceVisible();
  planner.device_pool =
    allocator_ && allocator_->supportsResidency(ResidencyClass::GPU_CLMEM);
  planner.raise =
    policy.raise_patterns.empty() ? nullptr : policy.raise_patterns.c_str();
  planner.lower =
    policy.lower_patterns.empty() ? nullptr : policy.lower_patterns.c_str();
  planner.exclude =
    policy.exclude_patterns.empty() ? nullptr : policy.exclude_patterns.c_str();
  /** NNTR_CLMEM_FANOUT=0 demotes a fan-out tensor off the device plane. The
   *  default keeps it eligible, so an unset environment classifies exactly as
   *  before (see SourceDetails::view_count). */
  planner.allow_fanout = [] {
    const char *e = std::getenv("NNTR_CLMEM_FANOUT");
    return !(e != nullptr && e[0] == '0');
  }();

  /** set the pointers using the token for all the tensors */
  for (auto &spec : pool) {
    auto details = std::get_if<SourceDetails>(&spec.details);
    if (!details || details->token == 0) {
      continue;
    }
    spec.tensor->setData(mem_pool->getMemory(details->token), 0, init);
    ml_logi("Memory Alloc Details (Tensor): %s : %zu : address %p",
            spec.tensor->getName().c_str(), spec.tensor->getMemoryBytes(),
            spec.tensor->getData());

    /** Record the placement on the freshly bound MemoryData. Views share that
     *  MemoryData through syncDependents below, so they inherit it and cannot
     *  disagree with the source about where the tensor is. A device placement
     *  the pool cannot honour falls back to the shared plane for the WHOLE
     *  tensor, which is a placement, not a half-conversion. */
    if (auto md = spec.tensor->getMemoryData()) {
      ResidencyClass cls = planner.classify(
        details->engine, details->all_consumers_device,
        spec.tensor->getDataType() == ml::train::TensorDim::DataType::FP16,
        spec.tensor->getInitializer() != Initializer::NONE,
        spec.tensor->getName(), details->view_count);
      /** The allocator has the last word on the class, because a placement
       *  is only available if the memory behind it is. Demote rather than
       *  refuse: the shared plane, and then the host plane, are always
       *  reachable by whoever was going to read the tensor. */
      if (allocator_ && !allocator_->supportsResidency(cls))
        cls = allocator_->supportsResidency(ResidencyClass::SVM)
                ? ResidencyClass::SVM
                : ResidencyClass::HOST;

      void *dev = nullptr;
      if (cls == ResidencyClass::GPU_CLMEM) {
        dev = mem_pool->deviceMemory(details->token);
        if (dev == nullptr)
          cls = ResidencyClass::SVM;
      }
      md->setResidency(cls, dev);
    }

    syncDependents(spec);
  }

  if (cache_loader) {
    cache_loader->init();
  }
}

/**
 * @brief Deallocate memory for all the managed tensors
 */
void TensorPool::deallocate() {
  if (cache_loader)
    cache_loader->finish();

  mem_pool->deallocate();

  /** nullify the data pointers for the tensors */
  for (auto &spec : pool) {
    spec.tensor->setData(nullptr);
  }
}

const std::vector<unsigned int> &
TensorPool::getExecutionOrder(const std::string &name) {
  return std::get<SourceDetails>(getSourceSpec(name).details).exec_order;
}

/**
 * @brief     Expand the lifespan of the tensor with the given name
 *
 */
TensorPool::RequestSpec &
TensorPool::expandLifespan(const std::string &name,
                           const std::vector<unsigned> &exec_order,
                           TensorLifespan lifespan) {
  auto &spec = getSourceSpec(name);
  expandLifespan(spec, exec_order, lifespan);
  return spec;
}

void TensorPool::expandLifespan(RequestSpec &spec,
                                const std::vector<unsigned int> &exec_order,
                                TensorLifespan lifespan) {
  auto &details = std::get<SourceDetails>(spec.details);
  NNTR_THROW_IF((details.lifespan != TensorLifespan::UNMANAGED &&
                 lifespan == TensorLifespan::UNMANAGED),
                std::invalid_argument)
    << "Extending to lifespan to unmanaged is not possible for name: "
    << spec.tensor->getName();

  if (details.lifespan != TensorLifespan::UNMANAGED) {
    /// update only if lifespan is unmanaged
    details.lifespan =
      enum_class_or<TensorLifespan>(details.lifespan, lifespan);
  }
  details.exec_order.insert(details.exec_order.end(), exec_order.begin(),
                            exec_order.end());
}

void TensorPool::syncDependents(const RequestSpec &spec) {
  /// @note syncing dependents of dependents is invalid and will throw.
  auto &dependents = std::get<SourceDetails>(spec.details).dependents;
  for (auto &dep : dependents) {
    auto &dep_spec = pool.at(dep);
    auto offset = std::get<DependentDetails>(dep_spec.details).offset;

    dep_spec.tensor->setData(spec.tensor->getMemoryData(),
                             spec.tensor->getOffset() + offset);
  }
}

Tensor *TensorPool::registerRequestSpec(RequestSpec &&spec) {
  auto &name = spec.tensor->getName();
  if (name_map.find(name) != name_map.end())
    throw std::invalid_argument("Cannot request tensor with same name");

  if (spec.tensor->empty())
    throw std::invalid_argument("Cannot request tensor with size 0");

  if (name.empty())
    throw std::invalid_argument("Cannot request tensor with empty name");

  pool.push_back(std::move(spec));
  name_map[name] = pool.size() - 1;

  return pool.back().tensor.get();
}

TensorPool::RequestSpec &TensorPool::getSourceSpec(const std::string &name) {
  RequestSpec *rs = &pool.at(name_map.at(name));
  while (auto dep_details = std::get_if<DependentDetails>(&rs->details)) {
    rs = &pool.at(dep_details->parent_idx);
  }

  return *rs;
}

void TensorPool::fillPlaceholder(const std::string &name, const Tensor &t) {
  auto &spec = getSourceSpec(name);
  auto &details = std::get<SourceDetails>(spec.details);
  NNTR_THROW_IF(details.lifespan != TensorLifespan::UNMANAGED,
                std::invalid_argument)
    << "Cannot set external tensor for non-zero lifespan for " << name;

  NNTR_THROW_IF(t.size() == 0 && t.getData(), std::invalid_argument)
    << "Error: setting invalid external tensor size 0 for " << name;

  NNTR_THROW_IF(t.size() != 0 && t.size() < spec.tensor->size(),
                std::invalid_argument)
    << "Error: setting external tensor of smaller size for "
    << spec.tensor->getName() << "(maybe view of " << name << ")";

  spec.tensor->setData(t.getMemoryData(), t.getOffset());
  syncDependents(spec);
}

Tensor *TensorPool::extend(const std::string &name, const TensorDim &dim,
                           const std::vector<unsigned int> &exec_order,
                           TensorLifespan lifespan) {
  NNTR_THROW_IF(!tensorExist(name), std::invalid_argument)
    << " cannot extend tensor which does not exist, name: " << name;
  auto &spec = getSourceSpec(name);
  NNTR_THROW_IF(dim != spec.tensor->getDim(), std::invalid_argument)
    << "Cannot extend tensor with different dimension";
  spec.is_weight_grad = false;
  expandLifespan(spec, exec_order, lifespan);
  return getTensor(name);
}

Tensor *TensorPool::requestOrExtend(const std::string &name,
                                    const TensorDim &dim,
                                    const std::vector<unsigned int> &exec_order,
                                    TensorLifespan lifespan,
                                    const Initializer &init,
                                    ml::train::LayerComputeEngine engine) {
  NNTR_THROW_IF(lifespan == TensorLifespan::UNMANAGED, std::invalid_argument)
    << "unmanaged life span is not supported";

  if (tensorExist(name)) {
    Tensor *t = getTensor(name);
    NNTR_THROW_IF(t->getDim() != dim, std::invalid_argument)
      << "tensor dimension mismatch for requestOrExtend name: " << name;
    NNTR_THROW_IF(t->getInitializer() != init, std::invalid_argument)
      << "tensor initializer mismatch for requestOrExtend name: " << name;
    return extend(name, dim, exec_order, lifespan);
  } else {
    return request(name, dim, exec_order, lifespan, init,
                   /*is_weight_grad=*/false, engine);
  }
}

void TensorPool::reidentifySource(const std::string &dest,
                                  const std::string &new_src,
                                  unsigned int offset) {
  /// @todo add test
  /// source tensor of dest tensor becomes a view of new_src
  auto &old_spec = getSourceSpec(dest);
  auto &old_details = std::get<SourceDetails>(old_spec.details);

  /// 1. extend new_src with old src
  auto &new_spec = getSourceSpec(new_src);
  expandLifespan(new_spec, old_details.exec_order, old_details.lifespan);
  auto &new_dependents = std::get<SourceDetails>(new_spec.details).dependents;
  new_dependents.insert(new_dependents.end(), old_details.dependents.begin(),
                        old_details.dependents.end());

  /// 2. calcaulate base offset from the new_src
  auto new_parent_idx = name_map.at(new_src);
  unsigned base_offset = std::visit(
    [](const auto &s) {
      using T = std::decay_t<decltype(s)>;
      if constexpr (std::is_same_v<T, SourceDetails>) {
        return 0u;
      } else if constexpr (std::is_same_v<T, DependentDetails>) {
        return s.offset;
      }
      return 0u;
    },
    pool[new_parent_idx].details);
  base_offset += offset;

  /// 3. transform parent idx/offset of old src's dependents base on the offset
  for (auto &dep : old_details.dependents) {
    auto &dep_spec = pool.at(dep);
    auto &details = std::get<DependentDetails>(dep_spec.details);
    details.offset += base_offset;
    details.parent_idx = new_parent_idx;
  }

  /// 4. replace old details to dependent srcs
  old_spec.details = DependentDetails{new_parent_idx, base_offset};
}

bool TensorPool::tensorExist(const std::string &name) {
  /// @todo consider use a helper function to check, eg) something like
  /// getTensor()
  return name_map.count(name);
}

/**
 * @brief     Check if the lifespan leads to long term valitidy
 *
 */
bool TensorPool::isTensorLongTerm(const TensorLifespan &lifespan) {
  switch (lifespan) {
  case TensorLifespan::EPOCH_LIFESPAN:
    [[fallthrough]];
  case TensorLifespan::FORWARD_INFER_LIFESPAN:
    [[fallthrough]];
  case TensorLifespan::MAX_LIFESPAN:
    return true;
  case TensorLifespan::FORWARD_FUNC_LIFESPAN:
    [[fallthrough]];
  case TensorLifespan::BACKWARD_FUNC_LIFESPAN:
    [[fallthrough]];
  case TensorLifespan::ITERATION_LIFESPAN:
    [[fallthrough]];
  case TensorLifespan::UNMANAGED:
    [[fallthrough]];
  default:
    return false;
  }
}

void TensorPool::flushCache() {
  if (auto pool = dynamic_cast<CachePool *>(mem_pool.get()))
    pool->flush();
}

void TensorPool::flushCacheExcept(unsigned int order) {
  if (auto pool = dynamic_cast<CachePool *>(mem_pool.get()))
    pool->flushExcept(order);
}

void TensorPool::loadCacheExec(unsigned int order) {
  if (dynamic_cast<CachePool *>(mem_pool.get()))
    cache_loader->loadAllinOrder(order);
}

int TensorPool::loadCacheExecAsync(
  unsigned int order, TaskExecutor::CompleteCallback complete_callback) {

  if (dynamic_cast<CachePool *>(mem_pool.get()))
    return cache_loader->loadAllinOrder(order);
  else
    return 0;
}

bool TensorPool::checkLoadComplete(unsigned int order) {
  if (dynamic_cast<CachePool *>(mem_pool.get()))
    return cache_loader->checkAllLoadComplete(order);
  else
    return true;
}

int TensorPool::flushCacheExecAsync(
  unsigned int order, TaskExecutor::CompleteCallback complete_callback) {
  if (dynamic_cast<CachePool *>(mem_pool.get()))
    return cache_loader->unloadAllinOrder(order);
  else
    return 0;
}

void TensorPool::loadCacheCancel(int id) {
  if (dynamic_cast<CachePool *>(mem_pool.get()) == nullptr)
    return;

  cache_loader->cancelAsync(id);
}

unsigned int TensorPool::inActive(unsigned int order) {
  return cache_loader->inActive(order);
}

} // namespace nntrainer
