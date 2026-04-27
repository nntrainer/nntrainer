// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   manager.h
 * @date   30 Nov 2020
 * @brief  This is NNtrainer manager for all weights, i/o and intermediate
 * tensors
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 *
 * @details Manager assumes that the layer inouts are being tracked by the
 * manager in the order of the execution. If the order is not maintained, then
 * the optimizations cannot be performed and will result in wrong values.
 */

#ifndef __MANAGER_H__
#define __MANAGER_H__
#include "tensor.h"
#include "tensor_wrap_specs.h"
#ifdef __cplusplus

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <basic_planner.h>
#include <common.h>
#include <graph_node.h>
#include <tensor_pool.h>
#include <var_grad.h>
#include <weight.h>

#include "noncopyable.h"
#include "nonmovable.h"

namespace nntrainer {
using ExecutionMode = ml::train::ExecutionMode;

/**
 * @class MMappedMemory
 * @brief Memory Handler, that has mmaped memory with a file descriptor
 */
class MMapedMemory : public Noncopyable, public Nonmovable {
public:
  /**
   * @brief Construct a new MMapedMemory object
   *
   * @param size bytesize of the memory chunk
   * @param allocate_fd_ map a shared memory object to a file
   */
  MMapedMemory(size_t size, bool allocate_fd_ = false);

  /**
   * @brief Destroy the MMapedMemory object
   *
   */
  ~MMapedMemory() noexcept;

  /**
   * @brief Get the File descriptor.
   * Will return -1 except for android
   * @todo make this available for other platforms
   *
   * @return -1 if fd is not allocated (or unabled to allocate)
   */
  int getFd() noexcept { return fd; }

  /**
   * @brief get the size of managed memory
   *
   * @return size_t size
   */
  size_t size() noexcept { return buf_size; }

  /**
   * @brief get Typed buffer from the memory
   *
   * @tparam T Type to specify the buffer. return is reinterpreted to T*
   * @return T* Typed buffer, return nullptr if empty
   */
  template <typename T> T *typedBuffer() noexcept {
    return reinterpret_cast<T *>(buf);
  }

  void *data() noexcept { return typedBuffer<void>(); }

private:
  int fd;           /**< fd to access the shared_memory  */
  void *buf;        /**< buffer object when use_shared_memory */
  size_t buf_size;  /**< buffer size */
  bool allocate_fd; /**< option to choose to allocate an fd */
};

/**
 * @class   Manager
 * @brief   manager of nntrainer
 */
class Manager : public Noncopyable, public Nonmovable {

public:
  /**
   * @brief Tensor Group Type
   * @note this is not mutually exclusive list, a tensor might be identified as
   * input as well as output
   *
   */
  enum TensorGroupType {
    INPUT = 0,   /**< Input of an operation */
    OUTPUT = 1,  /**< Output of an operation */
    WEIGHT = 2,  /**< Weight of an operation */
    TENSORS = 3, /**< Extra states of an operation */
  };

  constexpr inline static unsigned NUM_TENSOR_GROUP_TYPE =
    4; /**< number of tensor group type */

  /**
   * @brief     Constructor of Manager
   */
  Manager() :
    enable_fsu(false),
    enable_optimizations(true),
    fsu_lookahead(0),
    tensor_format("NCHW"),
    tensor_dtype(split("FP32-FP32", getRegex("\\-"))),
    exec_mode(ExecutionMode::TRAIN) {}

  /**
   * @brief     Constructor of Manager
   *
   * @param allocator backend allocator from the engine's Context
   *        (CPU / SVM / RPC). Forwarded into both the weight and the
   *        tensor pools so all dynamically requested memory comes
   *        from the same source. Default = host CPU memory.
   */
  Manager(bool enable_fsu_, const std::string &fsu_path = "",
          unsigned int lookahead = 0, const std::string tensor_format_ = "NCHW",
          const std::string tensor_dtype_ = "FP32-FP32",
          ExecutionMode exec_mode_ = ExecutionMode::TRAIN,
          std::shared_ptr<MemAllocator> allocator =
            std::make_shared<MemAllocator>()) :
    weight_pool(enable_fsu_, fsu_path, "weight_pool", exec_mode_, allocator),
    tensor_pool(enable_fsu_ && (exec_mode_ == ExecutionMode::TRAIN), fsu_path,
                "tensor_pool", exec_mode_, allocator),
    enable_fsu(enable_fsu_),
    enable_optimizations(true),
    fsu_lookahead(lookahead),
    tensor_format(tensor_format_),
    tensor_dtype(split(tensor_dtype_, getRegex("\\-"))),
    exec_mode(exec_mode_) {}

  /**
   * @brief Move Construct a new Manager object
   *
   */
  Manager(Manager &&) noexcept;

  /**
   * @brief Move assign a new Manager object
   *
   * @return Manager& reference to newly assign
   */
  Manager &operator=(Manager &&) noexcept;

  /**
   * @brief     Destructor of Manager
   */
  ~Manager() = default;

  /**
   * @brief     Create weights with the given spec
   * @todo      The max_exec_order can be reduced to the max exec order which
   * updates gradient
   *
   * @param node Graph node to extract node identifiers/info
   * @param weights_spec Specification for the weights
   * @param trainable make the weight trainable if true
   * @param shared_names name to refer to when the weights are borrowed from the
   * original source. if not shared pass empty vector
   *
   * @return created weights list
   */
  std::vector<Weight *>
  requestWeights(const GraphNode &node,
                 const std::vector<Weight::Spec> &weights_spec, bool trainable,
                 const std::vector<std::string> &shared_names);

  /**
   * @brief     Create tensors with the given spec
   *
   * @param node Graph node to extract node identifiers/info
   * @param tensors_spec Specification for the tensors
   * @param trainable make the weight trainable if true
   * @param shared_names if tensor is shared, name is needed
   *
   * @return created tensors list
   */
  std::vector<Var_Grad *> requestTensors(
    const GraphNode &node, const std::vector<Var_Grad::Spec> &tensors_spec,
    bool trainable, const std::vector<std::string> &shared_names = {});

  /**
   * @brief     Create tensors with the given spec
   *
   * @param node Graph node to extract node identifiers/info
   * @param tensors_spec Specification for the tensors
   *
   * @return created tensors list
   */
  std::vector<Tensor *> requestWeightOptimizerVariables(
    const std::vector<TensorDim> &dims, const std::string &name,
    const std::string &suffix, const TensorLifespan &lifespan,
    bool is_grad_clip, bool is_mixed_type,
    Initializer initializer = Initializer::NONE);

  /**
   * @brief     Create tensors with the given spec
   *
   * @param node Graph node to extract node identifiers/info
   * @param inputs_dim Specification for the tensors
   * @param outputs_name Name of the already requested output tensors
   *
   * @return created tensors list
   *
   * @details create Var_Grads to be used as input of GraphNode with the
   * inputs_dim as their spec. If the outputs_name is provided, the returned
   * Var_Grad share tensors with the already allocated Var_Grad for outputs,
   * named with outputs_name. In this case, the input_dim and the shape of the
   * output_tensors must match. If the outputs_name are empty, then new tensors
   * will be allocated.
   */
  std::vector<Var_Grad *>
  requestInputs(const GraphNode &node, const std::vector<TensorDim> &inputs_dim,
                const std::vector<std::string> &outputs_name = {});

  /**
   * @brief     Get all the weights which match the above condition
   *
   * @return    return the weights with satisfying the above condition
   */
  std::vector<Weight *>
  getWeights(const std::function<bool(const Weight *)> &condition = nullptr);

  /**
   * @brief Get the tensor execution orders
   *
   * @param name name of the tensor
   * @param is_weight check if this should be queried in weight pool
   * @return std::vector<unsigned int>
   */
  std::vector<unsigned int> getTensorExecutionOrders(const std::string &name,
                                                     bool is_weight);

  /**
   * @brief Get the Min Max of a tensor execution order
   *
   * @param name name of the tensor
   * @param is_weight check if this should be queried in weight pool
   * @return std::pair<unsigned int, unsigned int>
   */
  std::pair<unsigned int, unsigned int>
  getMinMaxTensorExecutionOrder(const std::string &name, bool is_weight);

  /**
   * @brief Get the second max of a tensor execution order
   *
   * @param name name of the tensor
   * @param is_weight check if this should be queried in weight pool
   * @return 2nd max execution order value
   */
  unsigned int getSecondMaxTensorExecutionOrder(const std::string &name,
                                                bool is_weight);

  /**
   * @brief check if given execution order is the first access
   *
   * @param name tensor name
   * @param current_execution current execution
   * @param is_weight check if this should be queried in weight pool
   * @return bool true if given execution order first access
   */
  bool isFirstAccess(const std::string &name, unsigned current_execution,
                     bool is_weight = false);

  /**
   * @brief check if given execution order is the last access
   *
   * @param name tensor name
   * @param current_execution current execution
   * @param is_weight check if this should be queried in weight pool
   * @return bool ture if given execution order is the last access
   */
  bool isLastAccess(const std::string &name, unsigned current_execution,
                    bool is_weight = false);

  /**
   * @brief check if given execution order is the second last access
   *
   * @param name tensor name
   * @param current_execution current execution
   * @param is_weight check if this should be queried in weight pool
   * @return bool ture if given execution order is the second last access
   */
  bool isSecondLastAccess(const std::string &name, unsigned current_execution,
                          bool is_weight = false);

  /**
   * @brief   Check if the manager has allocated tensors
   *
   * @return true if tensors allocated, else false
   */
  bool isAllocated() const { return tensor_pool.isAllocated(); }

  /**
   * @brief Set the batch size for the inputs/outputs of the layers
   */
  void setBatchSize(unsigned int batch) {
    /**
     * All the tensors must be deallocated first by the called and then
     * allocated by the caller.
     */
    for (auto &in : inputs_v2)
      in->setBatchSize(batch);
    for (auto &out : outputs_v2)
      out->setBatchSize(batch);
  }

  /**
   * @brief Set the batch size for the given tensor
   *
   * @note this does not works for weights as they are supposed to be
   * independent of batch size.
   */
  void setBatchSize(const std::string &name, unsigned int batch) {
    tensor_pool.setBatchSize(name, batch);
  }

  /**
   * @brief Allocate memory for all the managed tensors
   *
   * @param[in] max_exec_order The maximum order of execution to determine
   * memory layout
   *
   * @note Any requested tensor which is not used inside the max_exec_order is
   * not initialized and will not be allocated. The initialization uses a memory
   * planner to plan the layout of all the tensors which are used at least once
   * before the max_exec_order.
   */
  void allocateTensors(unsigned int max_exec_order_);

  /**
   * @brief Deallocate memory for all the managed tensors
   */
  void deallocateTensors(bool dealloc_weights = false);

  /**
   * @brief Allocate memory for all the managed weights
   *
   * @param[in] max_exec_order The maximum order of execution to determine
   * memory layout
   *
   * @note Any requested tensor which is not used inside the max_exec_order is
   * not initialized and will not be allocated. The initialization uses a memory
   * planner to plan the layout of all the tensors which are used at least once
   * before the max_exec_order.
   *
   * @note this will make requests to the tensor pool and allocate the
   * corresponding weights
   */
  void allocateWeights(unsigned int max_exec_order_, bool init = true);

  /**
   * @brief Deallocate memory for all the weights
   */
  void deallocateWeights();

  /**
   * @brief Set optimizations for manager
   *
   * @param val true to enable, else false
   */
  void setOptimizations(bool val) { enable_optimizations = val; }

  /**
   * @brief Update externally dependent tensors
   *
   * @param name Name of the tensor
   * @param t External tensor
   */
  void fillPlaceholder(const std::string &name, const Tensor &t) {
    tensor_pool.fillPlaceholder(name, t);
  }

  /**
   * @brief Get the tensor of the given name
   *
   * @return ptr to the tensor with the given
   * @throws if no tensor is found with the given name
   */
  Tensor *getTensor(const std::string &name) {
    try {
      return tensor_pool.getTensor(name);
    } catch (...) {
      return weight_pool.getTensor(name);
    }
  }

  /**
   * @brief request Tensor with weight specification
   *
   * @param spec specification
   * @param identify_as identify as tensor as a group
   * @return Tensor* tensor
   */
  Tensor *requestTensor(const WeightSpecV2 &spec, TensorGroupType identify_as);

  /**
   * @brief request Tensor with variable + gradient specification
   *
   * @param spec specification
   * @param identify_as identify as tensor as a group
   * @param exec_order execution order to refer to
   * @param scope common scope to attach in front of current specification name
   * @param expose_var expose variable tensor out of graph, when allocation,
   * this tensor will be valid max_exec_order when allocation happens
   * @param expose_grad expose variable tensor out of graph, this tensor will be
   * valid max_exec_order when allocation happens
   * @return Tensor* tensor
   */
  Var_Grad *requestTensor(const VarGradSpecV2 &spec,
                          TensorGroupType identify_as,
                          const GraphNode::ExecutionOrder &exec_order,
                          const std::string &scope = "",
                          bool expose_var = false, bool expose_grad = false);

  /**
   * @brief request vector of tensors with variable + gradient specification
   *
   * @param spec specification
   * @param identify_as identify as tensor as a group
   * @param exec_order execution order to refer to
   * @param scope common scope to attach in front of current specification name
   * @param expose_var expose variable tensor out of graph, when
   * allocation, this tensor will be valid max_exec_order when allocation
   * happens
   * @param expose_grad expose variable tensor out of graph, this tensor will be
   * valid max_exec_order when allocation happens
   * @return Tensor* tensor
   */
  std::vector<Var_Grad *> requestTensors(
    const std::vector<VarGradSpecV2> &specs, TensorGroupType identify_as,
    const GraphNode::ExecutionOrder &exec_order, const std::string &scope = "",
    bool expose_var = false, bool expose_grad = false);

  /**
   * @brief flush cache data
   */
  void flushCache();

  /**
   * @brief flush cache data except the order
   *
   * @param order except execution order
   * @param lookahead preloading size
   * @note preloading loads execution order data asynchronously,
   *       for lookahead size. If new flush request arrives,
   *       it waits previous preloading is completed and invokes new one.
   */
  void flushCacheExcept(unsigned int order);

  /**
   * @brief load cache data for the execution order
   *
   * @param order execution order
   * @param remainder_lookahead remain look_ahead
   * @note preloading loads execution order data asynchronously,
   *       for lookahead size.
   */
  void LoadTensors(unsigned int order, unsigned int remainder_lookahead = 0);

  /**
   * @brief check completion of load data for the execution order
   *
   * @param order execution order
   * @note preloading tensors for execution order.
   */
  bool checkLoadComplete(unsigned int order);

  /**
   * @brief check completion of unload data for the execution order
   *
   * @param order execution order
   * @note preloading tensors for execution order.
   */
  bool checkUnloadComplete(unsigned int order);

  /**
   * @brief load cache data for the execution order
   *
   * @param order execution order
   * @param lookahead look ahead value to load from execution order
   * @note preloading loads execution order data asynchronously,
   *       for lookahead size.
   */
  void LoadFsuTensors(unsigned int order, unsigned int lookahead);

  /**
   * @brief flush load data for the execution order
   *
   * @param order execution order
   * @note flush tensors for execution order.
   */
  void UnloadTensors(unsigned int order);

  /**
   * @brief     reinitialize manager
   */
  void reinitialize();

  /**
   * @brief     set Execution Mode
   */
  void setExecutionMode(ExecutionMode mode = ExecutionMode::TRAIN) {
    exec_mode = mode;
  };

  /**
   * @brief     return if it is mixed precsion
   */
  bool isMixedPrecision() { return !istrequal(tensor_dtype[0], "FP32"); }

  /**
   * @brief set Inactive elems in order
   *
   */
  unsigned int inActive(unsigned int order);

  /**
   * @brief set FSU weight path
   *
   * @param path FSU weight file path
   */
  void setFsuWeightPath(std::string path) {
    weight_pool.setFsuWeightPath(path);
  }

  /**
   * @brief set weight file offset for FSU loading
   *
   * @param offsets weight file offset
   */
  void setWeightOffset(std::vector<std::pair<size_t, size_t>> offsets) {
    weight_pool.setWeightOffset(offsets);
  }

private:
  /** @todo: merge this list to one */
  std::vector<std::unique_ptr<Weight>> weights_v2; /**< weights for the layers
                                                    */
  std::vector<std::unique_ptr<Var_Grad>>
    inputs_v2; /**< inputs for the layers */
  std::vector<std::unique_ptr<Var_Grad>>
    outputs_v2; /**< outputs for the layers */
  std::vector<std::unique_ptr<Var_Grad>>
    tensors_v2; /**< extra tensors required by the layers */

  std::array<std::vector<std::unique_ptr<Var_Grad>>, NUM_TENSOR_GROUP_TYPE>
    tensor_book; /**< reference to tensor book kept */

  TensorPool weight_pool; /**< tensor pool to request tensors */
  TensorPool tensor_pool; /**< tensor pool to request tensors */

  /** async load task <execution order, weight completed id> */
  std::map<unsigned int, int> async_task_weight_load;

  /** async unload task <execution order, weight completed id> */
  std::map<unsigned int, int> async_task_weight_unload;

  /**< async tasks <execution order, <weight_pool completed id, tensor_pool
   * completed id>>
   */

  std::map<unsigned int, std::tuple<int, int>> async_task_eos;
  /**< async tasks <execution order, <weight_pool completed id, tensor_pool
   * completed id>>
   */
  std::map<unsigned int, std::tuple<int, int>> async_load_tensor;

  std::map<unsigned int, std::tuple<int, int>> async_unload_tensor;

  std::map<int, std::promise<bool>> completed;

  std::map<int, std::promise<bool>> completed_load_tensor;

  std::map<int, std::future<TaskExecutor::CompleteStatus>> completed_load_fut;

  std::map<int, std::promise<bool>> completed_unload_tensor;

  /**< async tasks completion <task id, promise> */
  std::mutex completed_mutex; /**< mutex for async tasks completion */

  std::mutex completed_load_mutex; /**< mutex for async tasks completion */

  std::mutex completed_unload_mutex; /**< mutex for async tasks completion */

  bool enable_fsu; /**< to enable fsu */

  bool enable_optimizations; /**< to enable memory optimizations */

  unsigned int fsu_lookahead; /** lookahead for memory fsu */

  std::string tensor_format;

  std::vector<std::string> tensor_dtype;

  ExecutionMode exec_mode;

  unsigned int max_exec_order = 0;

  /**
   * @brief Finalize the given tensor pool
   *
   * @param pool Tensor pool to finalize
   * @param start Start execution order
   * @param end End execution order
   */
  void finalizeTensorPool(TensorPool &pool, unsigned int start,
                          unsigned int end);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MANAGER_H__ */
