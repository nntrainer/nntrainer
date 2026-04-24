// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api.h
 * @date   11 December 2023
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Tensor interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_TENSOR_H__
#define __ML_TRAIN_TENSOR_H__

#if __cplusplus < MIN_CPP_VERSION
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus

#include <functional>
#include <layer.h>
#include <memory>
#include <string>
#include <tensor_dim.h>
#include <vector>

namespace ml {
namespace train {

/**
 * @class   Tensor
 * @brief   Public Tensor API using Pimpl pattern.
 *
 * Symbolic tensors (constructed with dim) represent graph placeholders.
 * Eager tensors (fromData/zeros/ones) hold actual data immediately.
 * After model compile, symbolic tensors get bound to internal storage.
 */
class Tensor {
public:
  /**
   * @brief Default constructor — creates an invalid/empty tensor
   */
  Tensor();

  /**
   * @brief Construct a symbolic tensor with given dimensions
   *
   * @param dim Tensor dimensions
   * @param name Optional name for graph identification
   */
  explicit Tensor(const TensorDim &dim, const std::string &name = "");

  /**
   * @brief Destructor
   */
  ~Tensor();

  /**
   * @brief Move constructor
   */
  Tensor(Tensor &&rhs) noexcept;

  /**
   * @brief Move assignment
   */
  Tensor &operator=(Tensor &&rhs) noexcept;

  /**
   * @brief Copy constructor (shallow — shares the same graph node)
   */
  Tensor(const Tensor &rhs);

  /**
   * @brief Copy assignment (shallow)
   */
  Tensor &operator=(const Tensor &rhs);

  /**
   * @brief Clone with deep copy of data (for eager tensors)
   *
   * @return Deep-copied Tensor
   */
  Tensor clone() const;

  /**
   * @brief Check if this tensor is valid (has been properly constructed)
   *
   * @return true if tensor has been constructed with dim or data
   */
  bool isValid() const;

  /**
   * @brief Get tensor dimensions
   *
   * @return TensorDim
   */
  const TensorDim &shape() const;

  /**
   * @brief Get tensor name
   *
   * @return name string
   */
  const std::string &name() const;

  /**
   * @brief Get data type
   *
   * @return TensorDim::DataType
   */
  TensorDim::DataType dtype() const;

  /**
   * @brief Check if this tensor wraps external (user-managed) memory
   *
   * @return true if created via fromData()
   */
  bool isExternal() const;

  /**
   * @brief Check if this tensor has actual data (eager or bound after compile)
   *
   * @return true if data is accessible via data<T>() / getValue()
   */
  bool isMaterialized() const;

  /**
   * @brief Get read-only pointer to the underlying data
   *
   * @tparam T Data type (default: float)
   * @return Pointer to data buffer
   * @throws std::runtime_error if tensor is not materialized
   */
  template <typename T = float> const T *data() const;

  /**
   * @brief Get mutable pointer to the underlying data
   *
   * @tparam T Data type (default: float)
   * @return Pointer to mutable data buffer
   * @throws std::runtime_error if tensor is not materialized
   */
  template <typename T = float> T *mutable_data();

  /**
   * @brief Get value at specific location
   *
   * @param b batch index
   * @param c channel index
   * @param h height index
   * @param w width index
   * @return float value
   * @throws std::runtime_error if tensor is not materialized
   */
  float getValue(unsigned int b, unsigned int c, unsigned int h,
                 unsigned int w) const;

  /**
   * @brief Set value at specific location
   *
   * @param b batch index
   * @param c channel index
   * @param h height index
   * @param w width index
   * @param value value to set
   * @throws std::runtime_error if tensor is not materialized
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value);

  /**
   * @brief Copy data from external buffer into this tensor
   *
   * @param src Source buffer (must have at least shape().getDataLen() bytes)
   * @throws std::runtime_error if tensor is not materialized
   */
  void copyFrom(const void *src);

  /**
   * @brief Replace the external data pointer (fromData tensors only)
   *
   * @param new_ptr New data pointer (must outlive the tensor)
   * @throws std::runtime_error if tensor is not external
   */
  void setData(void *new_ptr);

  /**
   * @brief Create a tensor wrapping external user-managed memory (zero-copy)
   *
   * @param dim Tensor dimensions
   * @param data Pointer to user-managed buffer (must outlive the tensor)
   * @param name Optional name
   * @return Tensor wrapping the external buffer
   */
  static Tensor fromData(const TensorDim &dim, void *data,
                         const std::string &name = "");

  /**
   * @brief Create a zero-initialized eager tensor
   *
   * @param dim Tensor dimensions
   * @param name Optional name
   * @return Tensor with all elements set to 0
   */
  static Tensor zeros(const TensorDim &dim, const std::string &name = "");

  /**
   * @brief Create an eager tensor initialized to ones
   *
   * @param dim Tensor dimensions
   * @param name Optional name
   * @return Tensor with all elements set to 1
   */
  static Tensor ones(const TensorDim &dim, const std::string &name = "");

  /**
   * @brief Set the source layer that produced this tensor
   *
   * @param l Source layer
   */
  void setSrcLayer(std::shared_ptr<Layer> l);

  /**
   * @brief Get the source layer that produced this tensor
   *
   * @return Source layer (may be nullptr)
   */
  std::shared_ptr<Layer> getSrcLayer() const;

  // --- Symbolic tensor operations (create implicit layers) ---

  /**
   * @brief Element-wise addition (creates implicit Addition layer)
   *
   * @param other Tensor to add
   * @return Output tensor connected to an Addition layer
   */
  Tensor add(const Tensor &other) const;

  /**
   * @brief Element-wise multiplication (creates implicit Multiply layer)
   *
   * @param other Tensor to multiply
   * @return Output tensor connected to a Multiply layer
   */
  Tensor multiply(const Tensor &other) const;

  /**
   * @brief Reshape tensor (creates implicit Reshape layer)
   *
   * @param new_shape Target dimensions
   * @return Output tensor connected to a Reshape layer
   */
  Tensor reshape(const TensorDim &new_shape) const;

  /**
   * @brief Get the layer that produced this tensor (graph edge info)
   *
   * @return Producing layer (nullptr if this is an input/leaf tensor)
   */
  std::shared_ptr<Layer> getProducingLayer() const;

  /**
   * @brief Get an indexed output of a multi-output layer (e.g., split)
   *
   * Creates a symbolic tensor referencing a specific output index of the
   * producing layer. Used for layers like split that produce multiple outputs.
   *
   * Example:
   *   auto split_out = split_layer(input);
   *   auto first = split_out.output(0);   // split(0)
   *   auto second = split_out.output(1);  // split(1)
   *
   * @param index Output index (0-based)
   * @return New tensor referencing the indexed output
   */
  Tensor output(unsigned int index) const;

  /**
   * @brief Get the input tensors that were fed to the producing layer
   *
   * @return Vector of input tensors (empty if this is an input/leaf tensor)
   */
  std::vector<Tensor> getInputTensors() const;

  // --- Lazy chaining ---

  /**
   * @brief Start a lazy operation chain (clears any pending chain)
   * @return Reference to this tensor for chaining
   */
  Tensor &chain();

  /**
   * @brief Queue in-place addition (lazy, applied on eval())
   * @param value Scalar to add
   * @return Reference to this tensor for chaining
   */
  Tensor &add_i(float value);

  /**
   * @brief Queue in-place tensor addition (lazy, applied on eval())
   * @param other Tensor to add
   * @param alpha Scaling factor for other (default 1.0)
   * @return Reference to this tensor for chaining
   */
  Tensor &add_i(const Tensor &other, float alpha = 1.0f);

  /**
   * @brief Queue in-place subtraction (lazy, applied on eval())
   * @param value Scalar to subtract
   * @return Reference to this tensor for chaining
   */
  Tensor &subtract_i(float value);

  /**
   * @brief Queue in-place tensor subtraction (lazy, applied on eval())
   * @param other Tensor to subtract
   * @return Reference to this tensor for chaining
   */
  Tensor &subtract_i(const Tensor &other);

  /**
   * @brief Queue in-place multiplication (lazy, applied on eval())
   * @param value Scalar to multiply
   * @return Reference to this tensor for chaining
   */
  Tensor &multiply_i(float value);

  /**
   * @brief Queue in-place tensor multiplication (lazy, applied on eval())
   * @param other Tensor to multiply element-wise
   * @return Reference to this tensor for chaining
   */
  Tensor &multiply_i(const Tensor &other);

  /**
   * @brief Queue in-place division (lazy, applied on eval())
   * @param value Scalar divisor
   * @return Reference to this tensor for chaining
   */
  Tensor &divide_i(float value);

  /**
   * @brief Queue in-place tensor division (lazy, applied on eval())
   * @param other Tensor divisor (element-wise)
   * @return Reference to this tensor for chaining
   */
  Tensor &divide_i(const Tensor &other);

  /**
   * @brief Queue in-place power (lazy, applied on eval())
   * @param exponent Power exponent
   * @return Reference to this tensor for chaining
   */
  Tensor &pow_i(float exponent);

  /**
   * @brief Queue in-place inverse square root (lazy, applied on eval())
   * @return Reference to this tensor for chaining
   */
  Tensor &inv_sqrt_i();

  /**
   * @brief Execute all queued operations on the materialized tensor
   * @return Reference to this tensor
   * @throws std::runtime_error if tensor is not materialized
   */
  Tensor &eval();

  // --- Eager operations returning new tensors (requires materialized) ---

  /**
   * @brief Element-wise scalar addition
   * @param value Scalar to add
   * @return New tensor with result
   */
  Tensor add(float value) const;

  /**
   * @brief Element-wise scalar subtraction
   * @param value Scalar to subtract
   * @return New tensor with result
   */
  Tensor subtract(float value) const;

  /**
   * @brief Element-wise tensor subtraction
   * @param other Tensor to subtract
   * @return New tensor with result
   */
  Tensor subtract(const Tensor &other) const;

  /**
   * @brief Element-wise scalar multiplication
   * @param value Scalar to multiply
   * @return New tensor with result
   */
  Tensor multiply(float value) const;

  /**
   * @brief Element-wise scalar division
   * @param value Scalar divisor
   * @return New tensor with result
   */
  Tensor divide(float value) const;

  /**
   * @brief Element-wise tensor division
   * @param other Tensor divisor
   * @return New tensor with result
   */
  Tensor divide(const Tensor &other) const;

  /**
   * @brief Matrix multiplication (dot product)
   * @param other Right-hand tensor
   * @param trans Whether to transpose other (default false)
   * @param trans_in Whether to transpose this (default false)
   * @return New tensor with result
   */
  Tensor dot(const Tensor &other, bool trans = false,
             bool trans_in = false) const;

  /**
   * @brief Transpose tensor
   * @param direction Permutation string (e.g., "0:2:1")
   * @return New transposed tensor
   */
  Tensor transpose(const std::string &direction) const;

  /**
   * @brief Power operation
   * @param exponent Exponent value
   * @return New tensor with result
   */
  Tensor pow(float exponent) const;

  /**
   * @brief Sum along axis
   * @param axis Axis to sum over
   * @param alpha Scaling factor (default 1.0)
   * @return New tensor with result
   */
  Tensor sum(unsigned int axis, float alpha = 1.0f) const;

  /**
   * @brief Average along axis
   * @param axis Axis to average over
   * @return New tensor with result
   */
  Tensor average(unsigned int axis) const;

  /**
   * @brief Global average of all elements
   * @return New 1-element tensor with result
   */
  Tensor average() const;

  /**
   * @brief L2 norm of the tensor
   * @return L2 norm value
   */
  float l2norm() const;

  /**
   * @brief Get indices of maximum values along the last axis
   * @return Vector of indices (one per batch element)
   */
  std::vector<unsigned int> argmax() const;

  // --- Tensor manipulation ---

  /**
   * @brief Get a slice along the batch dimension
   * @param offset Batch offset
   * @param size Number of batches
   * @return New tensor sharing data (view)
   */
  Tensor getBatchSlice(unsigned int offset, unsigned int size) const;

  /**
   * @brief Get a view of this tensor with different dimensions (shared data)
   * @param dim New dimensions for the view
   * @param offset Offset in elements from start
   * @return New tensor sharing data with different shape
   */
  Tensor getSharedDataTensor(const TensorDim &dim, size_t offset) const;

  /**
   * @brief Apply element-wise function returning new tensor
   * @param f Function mapping float to float
   * @return New tensor with transformed values
   */
  Tensor apply(std::function<float(float)> f) const;

  /**
   * @brief Apply element-wise function in-place
   * @param f Function mapping float to float
   */
  void apply_i(std::function<float(float)> f);

  /**
   * @brief Concatenate multiple tensors along an axis
   * @param tensors Tensors to concatenate
   * @param axis Concatenation axis
   * @return Concatenated tensor
   */
  static Tensor cat(const std::vector<Tensor> &tensors, int axis);

  // --- Immediate in-place operations (no lazy chain) ---

  /**
   * @brief Set all elements to zero
   */
  void setZero();

  /**
   * @brief Fill this tensor with data from another tensor
   * @param from Source tensor
   */
  void fill(const Tensor &from);

  /**
   * @brief Copy data from another tensor
   * @param from Source tensor
   */
  void copyData(const Tensor &from);

  // --- Convenience dimension accessors ---

  /**
   * @brief Get total number of elements
   * @return Total element count
   */
  size_t size() const;

  /**
   * @brief Check if tensor is empty (no elements)
   * @return true if empty
   */
  bool empty() const;

  /**
   * @brief Get batch dimension
   */
  size_t batch() const;

  /**
   * @brief Get channel dimension
   */
  size_t channel() const;

  /**
   * @brief Get height dimension
   */
  size_t height() const;

  /**
   * @brief Get width dimension
   */
  size_t width() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  /// Get internal nntrainer::Tensor pointer (throws if not materialized)
  void *getInternalPtr() const;
  /// Wrap an internal tensor result into a public Tensor
  static Tensor wrapResult(const void *internal_tensor);

  friend class LayerHandle;
  friend class Model;
};

/**
 * @class   LayerHandle
 * @brief   Callable wrapper around a Layer for graph construction.
 *
 * Wraps a shared_ptr<Layer> and provides operator() to create symbolic
 * tensor connections (graph edges). Implicitly constructible from
 * unique_ptr<Layer> so it works with createLayer() results.
 *
 * Usage:
 *   LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc1"});
 *   auto output = fc(input);
 */
class LayerHandle {
public:
  LayerHandle() = default;

  /**
   * @brief Construct from unique_ptr<Layer> (implicit, works with createLayer)
   */
  LayerHandle(std::unique_ptr<Layer> p) : ptr_(std::move(p)) {}

  /**
   * @brief Construct from shared_ptr<Layer>
   */
  LayerHandle(std::shared_ptr<Layer> p) : ptr_(std::move(p)) {}

  /**
   * @brief Implicit conversion to shared_ptr<Layer> for backward compatibility
   */
  operator std::shared_ptr<Layer>() const { return ptr_; }

  /**
   * @brief Access the underlying Layer
   */
  Layer *get() const { return ptr_.get(); }
  Layer &operator*() const { return *ptr_; }
  Layer *operator->() const { return ptr_.get(); }
  explicit operator bool() const { return static_cast<bool>(ptr_); }

  /**
   * @brief Get the underlying shared_ptr
   */
  std::shared_ptr<Layer> layer() const { return ptr_; }

  /**
   * @brief Call the layer with a single input tensor (graph construction)
   *
   * Creates a new symbolic output Tensor connected to this layer.
   *
   * @param input Input tensor
   * @return Output tensor with graph edge info
   */
  Tensor operator()(const Tensor &input);

  /**
   * @brief Call the layer with multiple input tensors (graph construction)
   *
   * @param inputs Vector of input tensors
   * @return Output tensor with graph edge info
   */
  Tensor operator()(const std::vector<Tensor> &inputs);

private:
  std::shared_ptr<Layer> ptr_;
};

} // namespace train
} // namespace ml
#endif // __ML_TRAIN_TENSOR_H__
