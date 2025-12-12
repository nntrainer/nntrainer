// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   manager.cpp
 * @date   2 Dec 2020
 * @brief  This is NNtrainer manager for all weights, i/o and intermediate
 * tensors
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifdef __ANDROID__
#include <android/sharedmem.h>
#endif

#ifdef DEBUG
#include <cassert>
#endif
#include <fcntl.h>
#include <functional>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <activation_layer.h>
#include <basic_planner.h>
#include <bn_layer.h>
#include <graph_node.h>
#include <grucell.h>
#include <layer_node.h>
#include <layer_normalization_layer.h>
#include <loss/cross_entropy_sigmoid_loss_layer.h>
#include <loss/cross_entropy_softmax_loss_layer.h>
#include <loss/mse_loss_layer.h>
#include <manager.h>
#include <multiout_layer.h>
#include <nntrainer_log.h>
#include <optimized_v1_planner.h>
#include <optimized_v2_planner.h>
#include <optimized_v3_planner.h>
#include <tensor_pool.h>
#include <tensor_wrap_specs.h>
#include <util_func.h>
#include <var_grad.h>

#include "utils/mman_windows.h"

namespace nntrainer {

MMapedMemory::MMapedMemory(size_t size, bool allocate_fd_) :
  fd(-1), buf(nullptr), buf_size(0), allocate_fd(allocate_fd_) {

#ifndef __ANDROID__
  if (allocate_fd) {
    /// @todo create a file in tmpfs and bind to memfs
    /// memfd_create is not available for number of platforms so this is
    /// commented
    // auto fd_ = memfd_create("", 0);
    // if (fd_ < 0) {
    //   throw std::runtime_error("[Manager] creating mem fd failed");
    // }
    // if (ftruncate(fd_, size) < 0) {
    //   throw std::runtime_error("[Manager] truncating fd failed");
    // }
    ml_logi("[MMapedMemory] fd creation is not supported in this platform");
    allocate_fd = false;
  }
#endif
  int fd_ = -1;
  void *buf_ = nullptr;

  if (allocate_fd) {
#ifdef __ANDROID__
    /// unfortunately, memfd_create is not supported before android level 30
    fd_ = ASharedMemory_create("", size);
    if (fd_ < 0) {
      throw std::runtime_error("[MMapedMemory] creating mem fd failed");
    }

    if (ASharedMemory_setProt(fd_, PROT_READ | PROT_WRITE) < 0) {
      // unlink / close the given fd here
      close(fd_);
      throw std::runtime_error("[MMapedMemory] Setting prot failed");
    }

    buf_ = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
#endif
  } else {
    buf_ = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                fd_, 0);
  }

  if (buf_ == MAP_FAILED) {
#ifdef __ANDROID__
    if (fd_ != -1) {
      // unlink / close the given fd here
      close(fd_);
    }
#endif

    throw std::runtime_error("[MMapedMemory] mmap failed");
  }

  fd = fd_;
  buf = buf_;
  buf_size = size;

  ml_logd("[MMapedMemory] memory acquired size: %zu, fd: %d, addr: %p",
          buf_size, fd, buf);
}

MMapedMemory::~MMapedMemory() noexcept {
#ifdef DEBUG
  assert(buf_size > 0 && fd > 0);
#endif

  if (fd != -1) {
    if (close(fd) < 0) {
      ml_logw("[MMapedMemory] closing fd failed on destruction please check");
    }
  }

  if (buf != nullptr) {
    if (munmap(buf, buf_size) < 0) {
      ml_logw("[MMapedMemory] munmap failed on destruction please check");
    }
  }

  /// keeping the invariant although this is not necessary as of now
  fd = -1;
  buf = nullptr;
  buf_size = 0;
  ml_logd("[MMapedMemory] buf released");
}

void Manager::reinitialize() {
  inputs_v2.clear();
  outputs_v2.clear();
  tensors_v2.clear();
  tensor_pool.reinitialize();
}

void Manager::allocateWeights(unsigned int max_exec_order_, bool init) {
  max_exec_order = max_exec_order_;
  if (!weight_pool.isAllocated()) {
    finalizeTensorPool(weight_pool, 0, max_exec_order_);
    weight_pool.allocate(init);
  }
}

void Manager::deallocateWeights() { weight_pool.deallocate(); }

static Tensor *requestTensor_(const TensorSpecV2 &spec,
                              const GraphNode::ExecutionOrder &exec_order,
                              const std::string &scope, TensorPool &tp,
                              bool expose, bool trainable) {
  using RT = TensorSpecV2::RequestType;
  using LS = TensorLifespan;
  NNTR_THROW_IF(spec.request_type == RT::MAYBE_MODIFYING_VIEW,
                std::invalid_argument)
    << "Modifying view cannot be requested, the request type has to be "
       "delegated to either view or unique";

  auto [forward, calc_grad, calc_deriv, apply_grad] = exec_order;

  std::vector<unsigned> order = spec.additional_exec_order;
  if (expose) {
    order.push_back(TensorPool::PERSIST_END_ORDER);
  }

  const auto name = scope + ":" + spec.name;
  if (enum_class_or(spec.ls, LS::FORWARD_FUNC_LIFESPAN) == spec.ls) {
    order.push_back(forward);
  }
  if (enum_class_or(spec.ls, LS::CALC_GRAD_LIFESPAN) == spec.ls) {
    order.push_back(calc_grad);
  }
  if (enum_class_or(spec.ls, LS::CALC_DERIV_LIFESPAN) == spec.ls) {
    order.push_back(calc_deriv);
  }
  if (enum_class_or(spec.ls, LS::CALC_AGRAD_LIFESPAN) == spec.ls) {
    order.push_back(apply_grad);
  }

  switch (spec.request_type) {
  case RT::PLACEHOLDER:
    return tp.placeholder(name, spec.dim);
  case RT::UNIQUE:
    return tp.request(name, spec.dim, order, spec.ls, spec.initializer);
  case RT::SHARED:
    return tp.requestOrExtend(name, spec.dim, order, spec.ls, spec.initializer);
  case RT::READ_ONLY_VIEW:
    return tp.view(name, spec.reference_name, spec.dim, order, spec.ls);
  case RT::MAYBE_MODIFYING_VIEW:
  default:
    throw std::logic_error("requestTensor_ should not reach here");
  }

  return nullptr;
}

Var_Grad *Manager::requestTensor(const VarGradSpecV2 &spec,
                                 TensorGroupType identify_as,
                                 const GraphNode::ExecutionOrder &exec_order,
                                 const std::string &scope, bool expose_var,
                                 bool expose_grad) {
  NNTR_THROW_IF(identify_as == TensorGroupType::WEIGHT, std::invalid_argument)
    << "requestTensor with var grad spec cannot be identified as weights, use "
       "requestTensor with weight spec instead";

  NNTR_THROW_IF(identify_as == TensorGroupType::INPUT or
                  identify_as == TensorGroupType::TENSORS,
                nntrainer::exception::not_supported)
    << "Currently, input and tensors group type is not yet implemented, use "
       "requestInputs() requestTensors() instead";

  bool is_train_mode = (exec_mode == ExecutionMode::TRAIN) ? true : false;

  Tensor *var = requestTensor_(spec.variable_spec, exec_order, scope,
                               tensor_pool, expose_var, false);
  Tensor *grad = (spec.gradient_spec && is_train_mode)
                   ? requestTensor_(*spec.gradient_spec, exec_order, scope,
                                    tensor_pool, expose_grad, false)
                   : nullptr;

  /// @note as only supporting identify_as == TensorGroupType::output, only
  /// saves to outputs for now
  outputs_v2.push_back(std::make_unique<Var_Grad>(var, grad));

  return outputs_v2.back().get();
}

std::vector<Var_Grad *> Manager::requestTensors(
  const std::vector<VarGradSpecV2> &specs, TensorGroupType identify_as,
  const GraphNode::ExecutionOrder &exec_order, const std::string &scope,
  bool expose_var, bool expose_grad) {
  std::vector<Var_Grad *> ret;
  ret.reserve(specs.size());
  for (auto &spec : specs) {
    ret.push_back(requestTensor(spec, identify_as, exec_order, scope,
                                expose_var, expose_grad));
  }

  return ret;
}

/**
 * @brief Allocate memory for all the managed tensors
 */
void Manager::allocateTensors(unsigned int max_exec_order_) {
  allocateWeights(max_exec_order_);

  if (!tensor_pool.isAllocated()) {
    finalizeTensorPool(tensor_pool, 0, max_exec_order_);
    tensor_pool.allocate();
  }
}

/**
 * @brief Deallocate memory for all the managed tensors
 */
void Manager::deallocateTensors(bool dealloc_weights) {
  if (dealloc_weights) {
    deallocateWeights();
  }

  tensor_pool.deallocate();
}

#ifdef LAYER_V1
void Manager::initializeTensorsInference(unsigned int max_exec_order_) {
  /**
   * A single buffer (shared_inout) provides memory for inputs and outputs of a
   * layer. Further, the output of layer i shares memory with input with layer
   * i+1. So, each alternate layer allocates memory from either the start of the
   * buffer or the end of the buffer, and use_first_last tracks this
   *
   * @note Label for the last layer is not initialized in inference.
   * @note Input for the first layer is not initialized in inference.
   */
  // Initialize shared input/output memory for inference
  // @note Memory for label is not allocated here as inference doesnt has label
  if (enable_inference_inout_memory_opt)
    shared_inout = Tensor(TensorDim({max_shared_inout}), false);

  bool use_first_last = 0;
  for (unsigned int idx = 0; idx < in_outs.size(); idx++) {
    auto &l_io = in_outs[idx];
    unsigned int offset = 0;
    bool is_first_layer = idx == 0;

    // For flatten layer, do not assign new memory
    if (idx > 0 && is_flat_type[idx])
      use_first_last = 1 - use_first_last;

    // In inference mode, do not allocate the memory for the input of the
    // first layer. These is the first entry in the in_outs. Inference() will
    // override input tensors of the first layer
    if (is_first_layer)
      continue;

    for (auto &io : l_io) {
      Tensor shared_inout_cur = Tensor();
      if (enable_inference_inout_memory_opt) {
        // if optimized
        if (use_first_last) {
          // Create tensor with from the front of shared tensor
          shared_inout_cur =
            shared_inout.getSharedDataTensor(io->getDim(), offset);
        } else {
          // Create tensor with from the back of shared tensor
          shared_inout_cur = shared_inout.getSharedDataTensor(
            io->getDim(),
            max_shared_inout - io->getDim().getDataLen() - offset);
        }
        offset += io->getDim().getDataLen();
      }
      io->initialize(shared_inout_cur, Tensor(), false);
    }
    use_first_last = 1 - use_first_last;
  }
}

void Manager::initializeTensorsTrain(unsigned int max_exec_order_) {
  // Initialize gradients
  initializeGradients();

  // Initialize shared derivative memory
  if (max_derivative_size > 0 && enable_activation_memory_opt)
    shared_deriv = Tensor(TensorDim({max_derivative_size}), false);
  for (unsigned int idx = 0; idx < in_outs.size(); idx++) {
    auto &l_io = in_outs[idx];
    unsigned int offset = 0;
    bool is_last_layer = idx == in_outs.size() - 1;

    for (auto &io : l_io) {
      // Last layer requires separate memory allocations for output and label
      // (deriv)
      if (enable_derivative_memory_opt && !is_last_layer) {
        // Training Mode with optimizations
        if (enable_activation_memory_opt &&
            (is_rnn_type[idx] || is_act_type[idx])) {
          io->initialize(
            Tensor(), shared_deriv.getSharedDataTensor(io->getDim(), offset));
          offset += io->getDim().getDataLen();
        } else {
          io->initializeShared();
        }

      } else {
        // Training Mode without optimizations
        io->initialize(Tensor(), Tensor(), true);
      }
    }
  }
}
#endif

/**
 * @brief     Create weights with the given spec
 *
 */
std::vector<Weight *> Manager::requestWeights(
  const GraphNode &node, const std::vector<Weight::Spec> &weights_spec,
  bool trainable, const std::vector<std::string> &shared_names) {
  const auto [forwarding_order, calcGradient_order, calcDerivative_order,
              applyGradient_order] = node.getExecutionOrder();

  std::vector<unsigned int> default_var_exec_order(
    {forwarding_order, calcDerivative_order});

  /**
   *  TODO: This needs to be fixed. calcDerivative does not needs the gradient.
   *  However, current implementation of loss needs the gradient computation.
   *  and therefore, if we remove the calcDerivative order, then tests fails.
   */
  TensorLifespan var_ls;
  if (exec_mode != ExecutionMode::INFERENCE) {
    var_ls = TensorLifespan::MAX_LIFESPAN;
  } else {
    if (enable_fsu) {
      var_ls = TensorLifespan::FORWARD_FUNC_LIFESPAN;
    } else {
      var_ls = TensorLifespan::FORWARD_INFER_LIFESPAN;
    }
  }

  TensorLifespan grad_ls = TensorLifespan::BACKWARD_FUNC_LIFESPAN;

  std::vector<Weight *> ret;
  size_t current_size = weights_v2.size();

  for (unsigned int i = 0; i < weights_spec.size(); ++i) {
    auto &[dim_v, dim_g, t_initializer, w_reg, w_reg_const, decay,
           clip_by_global_norm, need_gradient, name, axis, loss_scale, is_mixed,
           is_virtual] = weights_spec.at(i);

    std::vector<unsigned int> var_exec_order;
    for (auto order : default_var_exec_order) {
      var_exec_order.push_back(order);
      if (exec_mode == ExecutionMode::INFERENCE)
        break;
    }
    // auto var_exec_order = default_var_exec_order;
    std::vector<unsigned int> grad_exec_order;

    if (trainable) {
      var_exec_order.reserve(var_exec_order.size() + 2);
      var_exec_order.push_back(calcGradient_order);
      var_exec_order.push_back(applyGradient_order);
      grad_exec_order.push_back(calcGradient_order);
      grad_exec_order.push_back(applyGradient_order);
    }

    /**
     * If the weight is supposed to be clip by global norm, extend its exec
     * order with the max exec order where it will be used for clipping and then
     * applied to the weight.
     */
    if (Weight::isGradientClipByGlobalNorm(clip_by_global_norm) ||
        isMixedPrecision()) {
      grad_exec_order.push_back(TensorPool::PERSIST_END_ORDER);
      // TODO: We need double check if it is OK not to add PERSIST_END_ORDER
      // here or add other conditions
      // var_exec_order.push_back(TensorPool::PERSIST_END_ORDER);
    }

    Tensor *var = nullptr, *grad = nullptr, *var32 = nullptr;
    bool is_dependent = !shared_names.empty();
    if (is_dependent) {
      /// shared_name is used and the original name is discarded
      const auto &shared_name = shared_names.at(i);
      /** case when shared names are given */
      var = weight_pool.requestOrExtend(shared_name, dim_v, var_exec_order,
                                        var_ls, t_initializer);
      if (trainable && need_gradient) {
        /** We cannot use the tensor scheduling for weight gradient if the
         * weight is shared. Weight Sharing means, the gradient is not temporal
         * for each layer anymore and it is hard to overwritten.
         */
        grad = tensor_pool.requestOrExtend(shared_name + Var_Grad::grad_suffix,
                                           dim_g, grad_exec_order, grad_ls,
                                           Initializer::ZEROS);

        if (var->getDataType() != ml::train::TensorDim::DataType::FP32) {
          TensorDim var32_dim(dim_v);
          var32_dim.setDataType(ml::train::TensorDim::DataType::FP32);
          std::vector<unsigned int> var32_exec_order;
          var32_exec_order.push_back(TensorPool::PERSIST_END_ORDER);

          var32 = weight_pool.requestOrExtend(shared_name + ":var32", var32_dim,
                                              var32_exec_order, var_ls,
                                              Initializer::ZEROS);
        }
      }
    } else {
      /** case requesting fresh weights */
      if (exec_mode == ExecutionMode::INFERENCE && enable_fsu) {
        for (unsigned int i = 0; i < fsu_lookahead; ++i) {
          int lah_order = (forwarding_order - (fsu_lookahead - i));
          var_exec_order.push_back(std::max(lah_order, 0));
        }
      }
      if (is_virtual) {
        var = weight_pool.request(name, dim_v, var_exec_order,
                                  TensorLifespan::VIRTUAL, t_initializer);
      } else {
        var = weight_pool.request(name, dim_v, var_exec_order, var_ls,
                                  t_initializer);
      }
      // }

      if (trainable && need_gradient) {
        /** is_wgrad is the index which is true when it is the gradient tensor
         * of weight. If it is true, memory planner schedule based on it to
         * reduce the memory.
         */
        bool is_wgrad = true;
        //        if (Weight::isGradientClipByGlobalNorm(clip_by_global_norm))
        //          is_wgrad = false;
        grad = tensor_pool.request(name + Var_Grad::grad_suffix, dim_g,
                                   grad_exec_order, grad_ls, Initializer::ZEROS,
                                   is_wgrad);
        if (var->getDataType() != ml::train::TensorDim::DataType::FP32) {
          TensorDim var32_dim(dim_v);
          var32_dim.setDataType(ml::train::TensorDim::DataType::FP32);
          std::vector<unsigned int> var32_exec_order;
          var32_exec_order.push_back(TensorPool::PERSIST_END_ORDER);
          var32 =
            weight_pool.request(name + ":var32", var32_dim, var32_exec_order,
                                var_ls, Initializer::ZEROS);
        }
      }
    }

    weights_v2.emplace_back(std::make_unique<Weight>(
      var, grad, var32, w_reg, w_reg_const, decay, is_dependent,
      clip_by_global_norm, axis, loss_scale, is_mixed));
  }

  std::transform(weights_v2.begin() + current_size, weights_v2.end(),
                 std::back_inserter(ret),
                 [](auto const &elem) { return elem.get(); });
  return ret;
}

/**
 * @brief     Create tensors with the given spec
 *
 */
std::vector<Var_Grad *> Manager::requestTensors(
  const GraphNode &node, const std::vector<Var_Grad::Spec> &tensors_spec,
  bool trainable, const std::vector<std::string> &shared_names) {
  const auto [forwarding_order, calcGradient_order, calcDerivative_order,
              applyGradient_order] = node.getExecutionOrder();

  std::vector<Var_Grad *> ret;
  size_t current_size = tensors_v2.size();
  bool is_train_mode = (exec_mode == ExecutionMode::TRAIN) ? true : false;

  for (unsigned int i = 0; i < tensors_spec.size(); ++i) {
    auto const &[dim, t_init, need_grad, name, tspan, t_engine] =
      tensors_spec.at(i);

    std::vector<unsigned int> var_exec_order;
    std::vector<unsigned int> grad_exec_order;

    /** usage for tensors */
    if (enum_class_logical_and(tspan, TensorLifespan::FORWARD_FUNC_LIFESPAN))
      var_exec_order.push_back(forwarding_order);

    /** usage for tensors gradient in backwarding */
    if (trainable && is_train_mode &&
        enum_class_logical_and(tspan, TensorLifespan::CALC_GRAD_LIFESPAN)) {
      var_exec_order.push_back(calcGradient_order);
      grad_exec_order.push_back(calcGradient_order);
    }

    if (is_train_mode &&
        enum_class_logical_and(tspan, TensorLifespan::CALC_DERIV_LIFESPAN)) {
      var_exec_order.push_back(calcDerivative_order);
      grad_exec_order.push_back(calcDerivative_order);
    }

    if (trainable && is_train_mode &&
        enum_class_logical_and(tspan, TensorLifespan::CALC_AGRAD_LIFESPAN)) {
      var_exec_order.push_back(applyGradient_order);
      grad_exec_order.push_back(applyGradient_order);
    }

    bool is_dependent = !shared_names.empty();
    Tensor *var = nullptr, *grad = nullptr;
    if (is_dependent) {
      const auto &shared_name = shared_names.at(i);
      var = tensor_pool.requestOrExtend(shared_name, dim, var_exec_order, tspan,
                                        t_init);
      if (need_grad && tspan > TensorLifespan::FORWARD_FUNC_LIFESPAN) {
        grad = tensor_pool.requestOrExtend(shared_name + Var_Grad::grad_suffix,
                                           dim, grad_exec_order, tspan,
                                           Initializer::ZEROS);
      }
    } else {
      var = tensor_pool.request(name, dim, var_exec_order, tspan, t_init);
      if (is_train_mode && need_grad &&
          tspan > TensorLifespan::FORWARD_FUNC_LIFESPAN) {
        grad = tensor_pool.request(name + Var_Grad::grad_suffix, /// name
                                   dim, grad_exec_order, tspan,
                                   Initializer::ZEROS /// tensor initializer
        );
      }
    }

    tensors_v2.emplace_back(std::make_unique<Var_Grad>(var, grad));
  }

  std::transform(tensors_v2.begin() + current_size, tensors_v2.end(),
                 std::back_inserter(ret),
                 [](auto const &elem) { return elem.get(); });
  return ret;
}

/**
 * @brief     Create tensors with the given spec
 */
std::vector<Var_Grad *>
Manager::requestInputs(const GraphNode &node,
                       const std::vector<TensorDim> &inputs_dim,
                       const std::vector<std::string> &outputs_name) {
  using RT = TensorSpecV2::RequestType;

  bool is_train_mode = exec_mode == ExecutionMode::TRAIN;

  TensorSpecV2 var_common_spec, grad_common_spec;
  if (is_train_mode) {
    var_common_spec.ls = TensorLifespan::FORWARD_GRAD_LIFESPAN;
  } else {
    var_common_spec.ls = TensorLifespan::FORWARD_FUNC_LIFESPAN;
  }

  grad_common_spec.ls = TensorLifespan::CALC_DERIV_LIFESPAN;
  /// @todo handle this inside layer
  /// @note ActivationLayer with supportInPlace()=false (e.g., swish, gelu) needs input in backward.
  ///       For these layers, extend input lifespan to FORWARD_DERIV_LIFESPAN.
  if (node.getType() == ActivationLayer::type) {
    if (node.supportInPlace())
      var_common_spec.ls = TensorLifespan::FORWARD_FUNC_LIFESPAN;
    else
      var_common_spec.ls = TensorLifespan::FORWARD_DERIV_LIFESPAN;
  } else if (node.getType() == MultiOutLayer::type or
             node.getType() == BatchNormalizationLayer::type or
             node.getType() == LayerNormalizationLayer::type or !node.getTrainable())
    var_common_spec.ls = TensorLifespan::FORWARD_FUNC_LIFESPAN;

  if (node.getType() == MSELossLayer::type or
      node.getType() == CrossEntropySoftmaxLossLayer::type or
      node.getType() == CrossEntropySigmoidLossLayer::type)
    var_common_spec.ls = TensorLifespan::FORWARD_DERIV_LIFESPAN;

  if (node.getType() == GRUCellLayer::type) {
    grad_common_spec.ls = TensorLifespan::CALC_GRAD_DERIV_LIFESPAN;
  }

  std::vector<Var_Grad *> ret;
  size_t current_size = inputs_v2.size();

  for (unsigned int idx = 0; idx < inputs_dim.size(); idx++) {
    TensorSpecV2 var_spec = var_common_spec, grad_spec = grad_common_spec;

    var_spec.name = std::string("input") + std::to_string(idx);
    var_spec.dim = inputs_dim[idx];

    grad_spec.name = var_spec.name + Var_Grad::grad_suffix;
    grad_spec.dim = inputs_dim[idx];

    if (!outputs_name.empty()) {
      grad_spec.request_type = var_spec.request_type = RT::READ_ONLY_VIEW;
      var_spec.reference_name = outputs_name[idx];
      grad_spec.reference_name = outputs_name[idx] + Var_Grad::grad_suffix;
    } else if (!node.getInputConnections().empty()) {
      grad_spec.request_type = var_spec.request_type = RT::UNIQUE;
    } else {
      var_spec.request_type = RT::PLACEHOLDER;

#ifdef ENABLE_TEST
      grad_spec.request_type = RT::UNIQUE;
#else
      grad_spec.request_type = RT::PLACEHOLDER;
#endif
    }
    inputs_v2.emplace_back(std::make_unique<Var_Grad>(
      requestTensor_(var_spec, node.getExecutionOrder(), node.getName(),
                     tensor_pool, false, node.getTrainable()),
      is_train_mode
        ? requestTensor_(grad_spec, node.getExecutionOrder(), node.getName(),
                         tensor_pool, false, node.getTrainable())
        : nullptr));
  }

  ret.reserve(inputs_dim.size());
  std::transform(inputs_v2.begin() + current_size, inputs_v2.end(),
                 std::back_inserter(ret),
                 [](auto const &elem) { return elem.get(); });

  return ret;
}

std::vector<unsigned int>
Manager::getTensorExecutionOrders(const std::string &name, bool is_weight) {

  return is_weight ? weight_pool.getExecutionOrder(name)
                   : tensor_pool.getExecutionOrder(name);
}

std::pair<unsigned int, unsigned int>
Manager::getMinMaxTensorExecutionOrder(const std::string &name,
                                       bool is_weight) {

  auto orders = is_weight ? weight_pool.getExecutionOrder(name)
                          : tensor_pool.getExecutionOrder(name);
  auto [min_, max_] = std::minmax_element(orders.begin(), orders.end());
  return {*min_, *max_};
}

unsigned int Manager::getSecondMaxTensorExecutionOrder(const std::string &name,
                                                       bool is_weight) {

  auto orders = is_weight ? weight_pool.getExecutionOrder(name)
                          : tensor_pool.getExecutionOrder(name);
  if (orders.size() < 2)
    throw std::runtime_error(
      "Requesting second last access with less than 2 exec orders");
  /** tensor pool exec order can have same exec order multiple times */
  std::sort(orders.begin(), orders.end());
  orders.erase(std::unique(orders.begin(), orders.end()), orders.end());
  return orders[orders.size() - 2];
}

bool Manager::isFirstAccess(const std::string &name, unsigned current_execution,
                            bool is_weight) {
  /// @todo add cache mechanism, eg) sort at finalizing requesting
  return getMinMaxTensorExecutionOrder(name, is_weight).first ==
         current_execution;
}

bool Manager::isLastAccess(const std::string &name, unsigned current_execution,
                           bool is_weight) {
  /// @todo add cache mechanism, eg) sort at finalizing requesting
  return getMinMaxTensorExecutionOrder(name, is_weight).second ==
         current_execution;
}

bool Manager::isSecondLastAccess(const std::string &name,
                                 unsigned current_execution, bool is_weight) {
  /// @todo add cache mechanism, eg) sort at finalizing requesting
  return getSecondMaxTensorExecutionOrder(name, is_weight) == current_execution;
}

/**
 * @brief     Create tensors with the given spec
 *
 */
std::vector<Tensor *> Manager::requestWeightOptimizerVariables(
  const std::vector<TensorDim> &dims, const std::string &name,
  const std::string &suffix, const TensorLifespan &lifespan, bool is_grad_clip,
  bool is_mixed_precision, Initializer initializer) {

  std::vector<Tensor *> ret;
  ret.reserve(dims.size());

  std::vector<unsigned int> exec;
  exec.reserve(1);
  if (is_grad_clip || is_mixed_precision) {
    exec.emplace_back(TensorPool::PERSIST_END_ORDER);
  } else {
    exec.emplace_back(getMinMaxTensorExecutionOrder(name, true).second);
  }

  /// @note this is assuming weight optimizer variables is treated as weight, if
  /// not, there is room to optimize below behavior
  for (unsigned int idx = 0; idx < dims.size(); idx++)
    ret.push_back(weight_pool.request(name + suffix + std::to_string(idx),
                                      dims[idx], exec, lifespan, initializer));

  return ret;
}

std::vector<Weight *>
Manager::getWeights(const std::function<bool(const Weight *)> &condition) {
  std::vector<Weight *> conditional_weights;

  for (auto &w : weights_v2) {
    if (!condition || condition(w.get()))
      conditional_weights.push_back(w.get());
  }
  return conditional_weights;
}

void Manager::flushCache() {
  if (!fsu_lookahead) {
    weight_pool.flushCache();
    tensor_pool.flushCache();
  }
}

bool Manager::checkLoadComplete(unsigned int order) {

  auto checkLoadCompleteAtPool = [](TensorPool &pool, unsigned int order) {
    return pool.checkLoadComplete(order);
  };

  if (exec_mode == ExecutionMode::TRAIN) {
    return checkLoadCompleteAtPool(weight_pool, order) &&
           checkLoadCompleteAtPool(tensor_pool, order);
  } else {
    return checkLoadCompleteAtPool(weight_pool, order);
  }
}

bool Manager::checkUnloadComplete(unsigned int order) {
  if (async_unload_tensor.count(order)) {
    auto &tasks = async_unload_tensor[order];
    std::unique_lock<std::mutex> lock(completed_unload_mutex);
    if (exec_mode == ExecutionMode::TRAIN) {
      auto w_fut = completed_unload_tensor[std::get<0>(tasks)].get_future();
      auto t_fut = completed_unload_tensor[std::get<1>(tasks)].get_future();
      lock.unlock();
      if (std::get<0>(tasks) != 0)
        w_fut.wait();
      if (std::get<1>(tasks) != 0)
        t_fut.wait();
    } else {
      auto w_fut = completed_unload_tensor[std::get<0>(tasks)].get_future();
      lock.unlock();
      if (std::get<0>(tasks) != 0)
        w_fut.wait();
    }
    async_unload_tensor.erase(order);
  }
  return true;
}

void Manager::LoadTensors(unsigned int order,
                          unsigned int remainder_lookahead) {

  auto loadTensorsAsync = [&](TensorPool &pool, unsigned int order) {
    return pool.loadCacheExecAsync(
      order, [&](int id, TaskExecutor::CompleteStatus status,
                 std::future<TaskExecutor::CompleteStatus> fut) {
        std::scoped_lock<std::mutex> lock(completed_load_mutex);
        completed_load_fut[id] = std::move(fut);
      });
  };

  auto enqueTasks = [&](unsigned int o) {
    auto load_weight = loadTensorsAsync(weight_pool, o);
    ml_logd("load weight is requested in LoadTensors with order - %d", o);
    int load_tensor = 0;

    if (exec_mode != ml::train::ExecutionMode::INFERENCE) {
      load_tensor = loadTensorsAsync(tensor_pool, o);
      ml_logd("load tensor is requested in LoadTensors with order - %d", o);
    }
    NNTR_THROW_IF(load_weight < 0 || load_tensor < 0, std::runtime_error)
      << "Fail to launch task";
  };

  if (order <= max_exec_order) {
    enqueTasks(order);
  }
}

void Manager::UnloadTensors(unsigned int order) {

  auto unloadTensorsAsync = [&](TensorPool &pool, unsigned int order) {
    return pool.flushCacheExecAsync(
      order, [&](int id, TaskExecutor::CompleteStatus status,
                 std::future<TaskExecutor::CompleteStatus> fut) {
        std::scoped_lock<std::mutex> lock(completed_unload_mutex);
        completed_unload_tensor[id].set_value(true);
      });
  };

  auto enqueTasks = [&](unsigned int o) {
    if (async_unload_tensor.count(o)) {
      ml_logd("Task unloadTensors (%d) is in progress", o);
      return;
    }
    auto unload_weight = unloadTensorsAsync(weight_pool, o);
    ml_logd("unload weight is requested in UnLoadTensors with order - %d", o);
    int unload_tensor = 0;
    if (exec_mode != ml::train::ExecutionMode::INFERENCE) {
      unload_tensor = unloadTensorsAsync(tensor_pool, o);
      ml_logd("unload tensor is requested in UnLoadTensors with order - %d", o);
    }
    NNTR_THROW_IF(unload_weight < 0 || unload_tensor < 0, std::runtime_error)
      << "Faile to launch task";
    async_unload_tensor[o] = std::make_tuple(unload_weight, unload_tensor);
  };

  enqueTasks(order);
}

void Manager::flushCacheExcept(unsigned int order) {
  auto loadAsync = [&](TensorPool &pool, unsigned int order) {
    return pool.loadCacheExecAsync(

      order, [&](int id, TaskExecutor::CompleteStatus status,
                 std::future<TaskExecutor::CompleteStatus> fu) {
        std::scoped_lock<std::mutex> lock(completed_mutex);
        completed[id].set_value(true);
      });
  };

  auto waitComplete = [&](unsigned int o) {
    auto &tasks = async_task_eos[o];

    std::unique_lock<std::mutex> lock(completed_mutex);
    auto w_fut = completed[std::get<0>(tasks)].get_future();
    auto t_fut = completed[std::get<1>(tasks)].get_future();
    lock.unlock();

    w_fut.wait();
    t_fut.wait();

    async_task_eos.erase(o);
  };

  // TODO: lookahead > 1 is required.
  if (fsu_lookahead == 1) {
    if (async_task_eos.count(order) == 1)
      waitComplete(order);

    auto load_weight = loadAsync(weight_pool, order + 1);
    auto load_tensor = loadAsync(tensor_pool, order + 1);

    NNTR_THROW_IF(load_weight < 0 || load_tensor < 0, std::runtime_error)
      << "Failed to launch preloading task";
    async_task_eos[order + 1] = std::make_tuple(load_weight, load_tensor);
  } else {
    weight_pool.flushCacheExcept(order);
    tensor_pool.flushCacheExcept(order);
  }
}

void Manager::finalizeTensorPool(TensorPool &pool, unsigned int start,
                                 unsigned int end) {
  if (enable_optimizations) {
    if (exec_mode == ExecutionMode::INFERENCE && enable_fsu) {
      //@todo change V3 and validate
      pool.finalize(OptimizedV1Planner(), start, end);
    } else {
      pool.finalize(OptimizedV1Planner(), start, end);
    }
  } else {
    pool.finalize(BasicPlanner(), start, end);
  }
}

unsigned int Manager::inActive(unsigned int order) {
  return weight_pool.inActive(order);
}

} // namespace nntrainer
