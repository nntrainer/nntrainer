// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   tensor_wrap_specs.h
 * @date   26 July 2021
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is specs for various tensor wrappers
 *
 */

#ifndef __TENSOR_WRAP_SPECS_H__
#define __TENSOR_WRAP_SPECS_H__

#include <memory>
#include <tuple>

#include <common.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief     Enumeration of Weight Regularizer
 * @todo      Update to TensorRegularizer
 */
enum class WeightRegularizer {
  L2NORM, /**< L2 norm regularization */
  NONE,   /**< no regularization */
  UNKNOWN /**< Unknown */
};

/**
 * @brief define the lifespan of the given tensor to reduce peak memory
 *
 */
enum class TensorLifespan {
  UNMANAGED = 0b000, /**< tensor with no lifespan, will not be allocated */
  FORWARD_FUNC_LIFESPAN = 0b001, /**< tensor must not be reset before during the
                           forward function call, eg. temporary tensors
                           needed during forward operations */
  CALC_DERIV_LIFESPAN = 0b010,   /**< must be valid during calcDerivative() */
  CALC_GRAD_LIFESPAN = 0b100, /**< tensor must be valid during calcGradient() */
  CALC_AGRAD_LIFESPAN =
    0b1000, /**< tensor must be valid during calcGradient() */
  CALC_GRAD_DERIV_LIFESPAN = 0b110, /**< tensor must not be reset before during
                             the calc_grad and clac_deriv call, eg. temporary
                             tensors needed during backward operations */
  CALC_GRAD_DERIV_AGRAD_LIFESPAN =
    0b1110,                      /**< tensor must not be reset before during
                   the calc_grad, clac_deriv and apply gradient call,
                   eg. temporary tensors needed during backward operations */
  FORWARD_GRAD_LIFESPAN = 0b101, /**< Forward + grad lifespan */
  FORWARD_GRAD_AGRAD_LIFESPAN =
    0b1101, /**< Forward + grad + apply gradient lifespan */
  FORWARD_DERIV_LIFESPAN = 0b011, /**< Forward + deriv lifespan */
  BACKWARD_FUNC_LIFESPAN =
    CALC_GRAD_DERIV_AGRAD_LIFESPAN, /**< Alias of CALC_GRAD_DERIV_AGRAD_LIFESPAN
                                     */
  ITERATION_LIFESPAN = 0b1111, /**< tensor must not be reset until the owning
                        layer finishes its execution in the current
                        iteration, eg. hidden memory/cells of RNN */
  EPOCH_LIFESPAN = 0b11111, /**< tensor must be valid before the epoch ends */
  FORWARD_INFER_LIFESPAN =
    0b100000,               /**< tensor is only used for only inference */
  MAX_LIFESPAN = 0b1111111, /**< tensor must not be reset until the end of the
                  model  execution, eg. layer weights */
  VIRTUAL = 0b11111111,     /**< virtual lifespan, tensor exists but does not
                               allocate memory */
};

/**
 * @brief Specification of the Weight as a tensor wrapper
 *
 * @details The tuple values are dimension, initializer, regularizer,
 * regularizer_constant, decay, clip gradient constant, need_gradient property,
 * name, output axis of the tensor object and loss Scale Factor, is_mixed,
 * is_virtual.
 *
 * @note virtual tensor doesn't allocate any memory but create an empty tensor
 */
typedef std::tuple<TensorDim, TensorDim, Initializer, WeightRegularizer, float,
                   float, float, bool, const std::string, unsigned int, float,
                   bool, bool>
  WeightSpec;

/**
 * @brief Specification of the Var_Grad (trainable tensor) as a tensor wrapper
 *
 * @details The tuple values are dimension, initializer, need_gradient property,
 * the name, and lifespan of the Var_Grad object.
 */
typedef std::tuple<TensorDim, Initializer, bool, const std::string,
                   TensorLifespan, ml::train::LayerComputeEngine>
  VarGradSpec;

/**
 * @brief Tensor Specification which describes how this tensor should be
 * allocated and managed
 *
 */
struct TensorSpecV2 {

  /**
   * @brief Tensor is being managed by nntrainer, this enum defines how the
   * value should be recognized inside nntrainer tensor managing scheme.
   *
   */
  enum class RequestType {
    PLACEHOLDER, /**< Placeholder defines that nntrainer should never care about
                    the memory inside the particualar tensor */
    UNIQUE, /**< Unique means a simple tensor that will be owned explicitly the
               current request */
    READ_ONLY_VIEW, /**< Readonly view defines a view of which ownership of @a
                       underlying memory is at another tensor, also hinting
                       nntrainer that the operation upon this particular tensor
                       will never change value of the underlying memory */
    MAYBE_MODIFYING_VIEW, /**< Maybe modifying view defines a (possible) view of
                       which ownership of @a underlying memory is at another
                       tensor, while hinting the nntrainer this tensor will do
                       some modification of the underlying memory. nntrainer
                       will try to make this particular tensor a view of the
                       stated reference. If making a view of reference is likely
                       to break the data integrity, nntrainer will request an
                       independent memory slot, in this case, it is user's
                       responsibility to copy the data. */
    SHARED, /**< Shared defines a shared tensor ownership for the given
               identifier, it is user's responsibility to guarantee that
               dimension and initializer of shared tensor if exactly same as the
               user will be agnostic about when and who will actually request
               the certain tensor. */
  };

  RequestType request_type = RequestType::UNIQUE; /**< Type of request */
  std::string name;                               /**< Identifier */
  TensorDim dim;                                  /**< dimension */
  TensorLifespan ls;                              /**< lifespan */
  Initializer initializer = Initializer::NONE;    /**< initializer */

  /** ONLY USED FOR READ_ONLY_VIEW, MAYBE_MODIFYING_VIEW */
  unsigned int offset = 0u;   /**< tensor offset */
  std::string reference_name; /**< reference name */

  /** ONLY FOR THE GRANULAR CONTROL OF LIFE OUTSIDE OF LAYER NODE */
  /// @todo make this as an opaque information with PIMPL
  std::vector<unsigned> additional_exec_order = {};

  /** compute engine of the producing layer, stamped by InitLayerContext so a
   * later residency step can derive where the tensor should live. CPU by
   * default, so nothing changes for non-GPU layers.
   */
  ml::train::LayerComputeEngine engine = ml::train::LayerComputeEngine::CPU;
};

/**
 * @brief variable + gradient specification
 *
 */
struct VarGradSpecV2 {

  /**
   * @brief Construct a new Var Grad Spec V2 object
   *
   */
  VarGradSpecV2() = default;

  /**
   * @brief Copy construct
   *
   * @param rhs
   */
  VarGradSpecV2(const VarGradSpecV2 &rhs) :
    variable_spec(rhs.variable_spec),
    gradient_spec(rhs.gradient_spec
                    ? std::make_unique<TensorSpecV2>(*rhs.gradient_spec)
                    : nullptr) {}

  /**
   * @brief copy assignment
   *
   * @param rhs
   * @return VarGradSpecV2&
   */
  VarGradSpecV2 &operator=(const VarGradSpecV2 &rhs) {
    variable_spec = rhs.variable_spec;
    gradient_spec = rhs.gradient_spec
                      ? std::make_unique<TensorSpecV2>(*rhs.gradient_spec)
                      : nullptr;
    return *this;
  }

  /**
   * @brief Move Construct
   *
   */
  VarGradSpecV2(VarGradSpecV2 &&) noexcept = default;
  VarGradSpecV2 &operator=(VarGradSpecV2 &&) noexcept = default;

  TensorSpecV2 variable_spec; /**< variable spec */
  std::unique_ptr<TensorSpecV2> gradient_spec =
    nullptr; /**< gradient spec, if null it cannot be trained*/
};

/**
 * @brief weight specification
 *
 */
struct WeightSpecV2 {
  VarGradSpecV2 vg_spec; /**< variable + graident specification */
  WeightRegularizer regularizer = WeightRegularizer::NONE; /**< regularizer */
  float regularizer_constant = 0.0f; /**< regularizer constant */
  float decay = 0.0f;                /**< decay constant */
  float clip_by_global_norm = 0.0f;  /**< clip the gradient by norm */
};

} // namespace nntrainer

#endif /** __TENSOR_WRAP_SPECS_H__ */
