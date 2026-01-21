// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   acti_func.cpp
 * @date   22 March 2021
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Function Class for Neural Network
 *
 */

#ifndef __ACTI_FUNC_H__
#define __ACTI_FUNC_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <cpu_backend.h>

#if defined(_WIN32)
#define _USE_MATH_DEFINES
#include <math.h>
#endif

namespace nntrainer {

class Tensor;

/**
 * @class   ActiFunc Class
 * @brief   ActiFunc Class
 */
class ActiFunc {

public:
  constexpr static inline float NEGATIVE_SLOPE = 0.01f;

  /**
   * @brief     Constructor of ActiFunc
   */
  template <typename T = float>
  ActiFunc(ActivationType at = ActivationType::ACT_NONE,
           bool is_inplace_ = true) :
    is_inplace(is_inplace_) {
    setActiFunc<T>(at);
  }

  /**
   * @brief     Destructor of ActiFunc
   */
  ~ActiFunc(){};

  /**
   * @brief setActivation by preset ActivationType
   *
   * @param[in] ActivationType
   */
  template <typename T = float> void setActiFunc(ActivationType acti_type) {
    activation_type = acti_type;

    switch (acti_type) {
    case ActivationType::ACT_TANH:
      this->setActivation<T>(tanhFloat<T>, tanhPrime<T>);
      break;
    case ActivationType::ACT_SIGMOID:
      this->setActivation<T>(sigmoid<T>, sigmoidPrime<T>);
      break;
    case ActivationType::ACT_SOFTMAX:
      this->setActivation<Tensor>(softmax<T>, softmaxPrime<T>);
      break;
    case ActivationType::ACT_RELU:
      this->setActivation<T>(relu<T>, reluPrime<T>);
      break;
    case ActivationType::ACT_LEAKY_RELU:
      this->setActivation<T>(leakyRelu<T>, leakyReluPrime<T>);
      break;
    case ActivationType::ACT_SWISH:
      is_inplace = false;
      this->setActivation<Tensor>(swish<T>, swishPrime<T>);
      break;
    case ActivationType::ACT_GELU:
      is_inplace = false;
      this->setActivation<Tensor>(gelu<T>, geluPrime<T>);
      break;
    case ActivationType::ACT_TANH_GELU:
      is_inplace = false;
      this->setActivation<Tensor>(tanhGelu<T>, tanhGeluPrime<T>);
      break;
    case ActivationType::ACT_SIGMOID_GELU:
      is_inplace = false;
      this->setActivation<Tensor>(sigmoidGelu<T>, sigmoidGeluPrime<T>);
      break;
    case ActivationType::ACT_ELU:
      this->setActivation<T>(elu<T>, eluPrime<T>);
      break;
    case ActivationType::ACT_SELU:
      this->setActivation<T>(selu<T>, seluPrime<T>);
      break;
    case ActivationType::ACT_SOFTPLUS:
      this->setActivation<T>(softplus<T>, softplusPrime<T>);
      break;
    case ActivationType::ACT_MISH:
      this->setActivation<T>(mish<T>, mishPrime<T>);
      break;
    case ActivationType::ACT_NONE:
      this->setActivation<T>(no_op<T>, no_op_prime<T>);
      break;
    case ActivationType::ACT_UNKNOWN:
    default:
      throw std::runtime_error("Error: Not Supported Activation Type");
    }
  }

  /**
   * @brief run function
   *
   * @param[in] input : input
   * @param[out] output : output
   */
  void run_fn(Tensor const &input, Tensor &output) { _act_fn(input, output); }

  /**
   * @brief run prime function
   *
   * @param[in] input input
   * @param[in] output output
   * @param[out] outgoing_derivative outgoing derivative
   * @param[in] incoming_derivative incoming derivative
   * @retVal    Tensor
   */
  Tensor &run_prime_fn(Tensor &input, Tensor &output,
                       Tensor &outgoing_derivative,
                       Tensor const &incoming_derivative) {
    return _act_prime_fn(input, output, outgoing_derivative,
                         incoming_derivative);
  }

  /**
   * @brief run prime function
   *
   * @param[in] output output
   * @param[out] outgoing_derivative outgoing derivative
   * @param[in] incoming_derivative incoming derivative
   * @retVal    Tensor
   */
  Tensor &run_prime_fn(Tensor &output, Tensor &outgoing_derivative,
                       Tensor const &incoming_derivative) {
    return _act_prime_fn(Tensor(), output, outgoing_derivative,
                         incoming_derivative);
  }

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const { return is_inplace; }

  /**
   * @brief       Calculate softmax for Tensor Type
   * @param[in] input input Tensor
   * @param[out] output output Tensor
   * @retval      Tensor
   */
  template <typename T = float>
  static Tensor &softmax(Tensor const &input, Tensor &output) {
    /**
     * shiftx_logit = logit - max_batch(logit)
     * softmax = exp(shiftx_logit) / (sum(exp(shiftx_logit)))
     *
     * @note softmax is applied on the last dimension
     */
    /** TODO: support strided operations */
    if (input.size() == output.size() &&
        input.getStrides() != output.getStrides())
      throw std::invalid_argument(
        "Softmax does not support operating on strided tensors");

    unsigned int width = input.width();
    unsigned int bch_size = input.getDim().getDataLen() / width;

    // copy will not executed in inplace case
    output.copy(input);

    T *output_data = output.getData<T>();

    // prevent overflow
    Tensor tmp(width, input.getTensorType());
    for (unsigned int i = 0; i < bch_size; i++) {
      T *ptr = output_data + i * width;

      // find max value and subtract it
      T max_value = *std::max_element(ptr, ptr + width);

      tmp.setValue(max_value);
      saxpy(width, -1, tmp.getData<T>(), 1, ptr, 1);
    }

    // take exp
    output.apply<T>(exp_util<T>, output);

    // take sum over the last dimension
    Tensor sum = output.sum(3);

    for (unsigned int i = 0; i < bch_size; i++) {
      T *ptr = output_data + i * width;
      std::transform(ptr, ptr + width, ptr,
                     std::bind(std::divides<T>(), std::placeholders::_1,
                               sum.getValue<T>(i)));
    }

    return output;
  }

  /**
   * @brief     Calculate derivative of softmax function
   * @param[in] output output tensor
   * @param[out] outgoing_derivative result of calculated derivative of softmax
   * @param[in] incoming_derivative incoming derivative tensor from next layer
   * @retVal    Tensor
   */

  template <typename T = float>
  static Tensor &softmaxPrime(Tensor const &output, Tensor &outgoing_derivative,
                              Tensor const &incoming_derivative = Tensor()) {
    /** TODO: support strided operations */

    if ((output.size() == outgoing_derivative.size() &&
         output.getStrides() != outgoing_derivative.getStrides()) ||
        (output.size() == incoming_derivative.size() &&
         output.getStrides() != incoming_derivative.getStrides()))
      throw std::invalid_argument(
        "SoftmaxPrime does not support operating on strided tensors");

    unsigned int batch = output.batch();
    unsigned int channel = output.channel();
    unsigned int height = output.height();
    unsigned int width = output.width();

    if (outgoing_derivative.empty())
      outgoing_derivative = Tensor(output.getDim());

    const T *output_data = output.getData<T>();
    const T *incoming_derivative_data = incoming_derivative.getData<T>();
    T *outgoing_derivative_data = outgoing_derivative.getData<T>();

    Tensor tmp = Tensor(width, output.getTensorType());
    T *tmp_data = tmp.getData<T>();
    unsigned int output_width_stride = output.getStrides()[3];
    for (unsigned int b = 0; b < batch; ++b) {
      int b_offset = b * channel * height * width;
      for (unsigned int c = 0; c < channel; ++c) {
        int bc_offset = b_offset + c * height * width;
        for (unsigned int h = 0; h < height; ++h) {
          int bch_offset = bc_offset + h * width;
          for (unsigned int w1 = 0; w1 < width; ++w1) {
            T sum = 0;
            for (unsigned int w2 = 0; w2 < width; ++w2) {
              T val;
              if (w1 == w2) {
                val = output_data[bch_offset + w2] *
                      ((T)1 - output_data[bch_offset + w1]);
              } else {
                val =
                  -output_data[bch_offset + w2] * output_data[bch_offset + w1];
              }
              if (!incoming_derivative.empty())
                val *= incoming_derivative_data[bch_offset + w2];
              sum += val;
            }
            tmp.setValue(0, 0, 0, w1, sum);
          }
          scopy(width, tmp_data, 1, outgoing_derivative_data + bch_offset,
                output_width_stride);
        }
      }
    }

    return outgoing_derivative;
  }

  /**
   * @brief     sigmoid activation function
   * @param[in] x input
   */
  template <typename T = float> static T sigmoid(T x) {
    return static_cast<T>(1.0 / (1.0 + exp_util<T>(-x)));
  }

  /**
   * @brief     derivative sigmoid function
   * @param[in] x input
   */
  template <typename T = float> static T sigmoidPrime(T x) {
    return static_cast<T>(x * (static_cast<T>(1.0) - x));
  }

  /**
   * @brief     tanh function for float type
   * @param[in] x input
   */
  template <typename T = float> static T tanhFloat(T x) {
    return static_cast<T>(2.0 * sigmoid<T>(static_cast<T>(2.0) * x) - 1.0);
  }

  /**
   * @brief     derivative tanh function
   * @param[in] x input
   */
  template <typename T = float> static T tanhPrime(T x) {
    return static_cast<T>(1.0 - x * x);
  }

  /**
   * @brief     relu activation function
   * @param[in] x input
   */
  template <typename T = float> static T relu(T x) {
    if (x <= 0)
      return 0;
    return x;
  }

  /**
   * @brief     derivative relu function
   * @param[in] x input
   */
  template <typename T = float> static T reluPrime(T x) {
    if (x <= 0)
      return 0;
    return 1;
  }

  /**
   * @brief     no_op function
   * @param[in] x input
   */
  template <typename T = float> static T no_op(T x) { return x; }

  /**
   * @brief     no_op function
   * @param[in] x input
   */
  template <typename T = float> static T no_op_prime(T x) { return 1; }

  /**
   * @brief leaky relu function
   * @note slope parameter is needed for leaky relu, but supporting property on
   * this class will need extensive refactoring. For now 0.01 is used for
   * negative slope.
   *
   * @param x input
   * @return float output
   */
  template <typename T = float> static T leakyRelu(T x) {
    return x >= static_cast<T>(0.0) ? x : static_cast<T>(NEGATIVE_SLOPE) * x;
  }

  /**
   * @brief leaky relu prime function
   * @note slope parameter is needed for leaky relu, but supporting property on
   * this class will need extensive refactoring. For now 0.01 is used for
   * negative slope.
   *
   * @param x input
   * @return float output
   */
  template <typename T = float> static T leakyReluPrime(T x) {
    return x >= static_cast<T>(0.0) ? static_cast<T>(1.0)
                                    : static_cast<T>(NEGATIVE_SLOPE);
  }

  /**
   * @brief     Softplus activation function
   * @tparam T type of an input/output
   * @param x input
   * @return T type output
   */
  template <typename T = float> static T softplus(T x) {
    /** TODO: Change beta to be a property */
    return static_cast<T>(log(1 + exp(beta * x)) / beta);
  }

  /**
   * @brief     derivative softplus function
   * @tparam T type of an input/output
   * @param x input
   * @return T type output
   */
  template <typename T = float> static T softplusPrime(T x) {
    return sigmoid<T>(static_cast<T>(beta * x));
  }

  /**
   * @brief     swish activation function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   */
  template <typename T = float>
  static Tensor &swish(Tensor const &t_in, Tensor &t_out) {
    t_in.apply<T>([&](T x) { return sigmoid<T>(x); }, t_out);
    t_out.multiply_i(t_in);

    return t_out;
  }

  /**
   * @brief     derivative swish function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   * @param[in] outgoing_derivative outgoing derivative
   * @param[in] incoming_derivative incoming derivative
   */
  template <typename T = float>
  static Tensor &swishPrime(Tensor const &t_in, Tensor const &t_out,
                            Tensor &outgoing_derivative,
                            Tensor const &incoming_derivative = Tensor()) {
    if (outgoing_derivative.empty())
      outgoing_derivative = Tensor(t_out.getDim());

    Tensor tmp = Tensor(t_out.getDim());
    t_in.apply<T>([&](T x) { return sigmoid(x); }, outgoing_derivative);
    t_out.apply<T>([&](T x) { return 1 - x; }, tmp);
    outgoing_derivative.multiply_i(tmp);
    outgoing_derivative.add_i(t_out);

    outgoing_derivative.multiply_i_strided(incoming_derivative);

    return outgoing_derivative;
  }

  /**
   * @brief     gelu activation function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   */
  template <typename T = float>
  static Tensor &gelu(Tensor const &t_in, Tensor &t_out) {
    double tmp = 1.0 / sqrt(2.0);
    t_in.apply<T>(
      [&](T x) { return static_cast<T>(0.5 * x * (1 + erf(x * tmp))); }, t_out);
    return t_out;
  }

  /**
   * @brief     derivative gelu function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   * @param[in] outgoing_derivative outgoing derivative
   * @param[in] incoming_derivative incoming derivative
   */
  template <typename T = float>
  static Tensor &geluPrime(Tensor const &t_in, Tensor const &t_out,
                           Tensor &outgoing_derivative,
                           Tensor const &incoming_derivative = Tensor()) {

    if (outgoing_derivative.empty())
      outgoing_derivative = Tensor(t_out.getDim());

    T tmp = static_cast<T>(1 / sqrt(2));
    t_in.apply<T>(
      [&](T x) {
        return static_cast<T>(
          0.5 * (1 + erf(x * tmp) +
                 x * ((2 / sqrt(M_PI)) * exp(-pow(x * tmp, 2))) * tmp));
      },
      outgoing_derivative);

    outgoing_derivative.multiply_i_strided(incoming_derivative);

    return outgoing_derivative;
  }

  /**
   * @brief     tanh-based gelu approximate function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   */
  template <typename T = float>
  static Tensor &tanhGelu(Tensor const &t_in, Tensor &t_out) {
    nntrainer::tanh_gelu(t_in.size(), t_in.getData<float>(),
                         t_out.getData<float>());
    return t_out;
  }

  /**
   * @brief     derivative of tanh-based gelu approximate function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   * @param[in] outgoing_derivative outgoing derivative
   * @param[in] incoming_derivative incoming derivative
   */
  template <typename T = float>
  static Tensor &tanhGeluPrime(Tensor const &t_in, Tensor const &t_out,
                               Tensor &outgoing_derivative,
                               Tensor const &incoming_derivative = Tensor()) {
    // NYI
    ml_logw("tanhGeluPrime which is calculate derivate of tanhGelu function is "
            "not yet implemented");
    return outgoing_derivative;
  }

  /**
   * @brief     sigmoid-based gelu approximate function (quick gelu)
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   */
  template <typename T = float>
  static Tensor &sigmoidGelu(Tensor const &t_in, Tensor &t_out) {
    t_in.apply<T>(
      [&](T x) {
        return static_cast<T>(x * (sigmoid<T>(static_cast<T>(1.702 * x))));
      },
      t_out);
    return t_out;
  }

  /**
   * @brief     derivative of sigmoid-based gelu approximate function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   * @param[in] outgoing_derivative outgoing derivative
   * @param[in] incoming_derivative incoming derivative
   */
  template <typename T = float>
  static Tensor &
  sigmoidGeluPrime(Tensor const &t_in, Tensor const &t_out,
                   Tensor &outgoing_derivative,
                   Tensor const &incoming_derivative = Tensor()) {
    // NYI
    ml_logw("sigmoidGeluPrime which is calculate derivate of sigmoidGelu "
            "function is not yet implemented");
    return outgoing_derivative;
  }

  /**
   * @brief elu function
   * @note alpha parameter is needed for elu, but supporting property on
   * this class will need extensive refactoring. For now 1.0 is used for
   * alpha.
   *
   * @tparam T type of an input/output
   * @param x input
   * @return T type output
   */
  template <typename T = float> static T elu(T x) {
    return x >= static_cast<T>(0.0) ? x : static_cast<T>(alpha * (exp(x) - 1));
  }

  /**
   * @brief elu prime function
   * @note alpha parameter is needed for elu, but supporting property on
   * this class will need extensive refactoring. For now 1.0 is used for
   * alpha.
   *
   * @tparam T type of an input/output
   * @param x input
   * @return T type output
   */
  template <typename T = float> static T eluPrime(T x) {
    return x >= static_cast<T>(0.0) ? static_cast<T>(1.0)
                                    : static_cast<T>(alpha * exp(x));
  }

  /**
   * @brief selu function
   * @tparam T type of an input/output
   * @param x input
   * @return T type output
   */
  template <typename T = float> static T selu(T x) {
    return x > static_cast<T>(0.0)
             ? static_cast<T>(selu_scale * x)
             : static_cast<T>(selu_scale * selu_alpha * (exp(x) - 1));
  }

  /**
   * @brief selu prime function
   * @tparam T type of an input/output
   * @param x input
   * @return T type output
   */
  template <typename T = float> static T seluPrime(T x) {
    return x > static_cast<T>(0.0)
             ? static_cast<T>(selu_scale)
             : static_cast<T>(selu_scale * selu_alpha * exp(x));
  }

  /**
   * @brief     mish activation function
   * @param[in] x input
   */
  template <typename T = float> static T mish(T x) {
    return static_cast<T>(x * tanhFloat<T>(softplus<T>(x)));
  }

  /**
   * @brief     mish prime function
   * @param[in] x input
   */
  template <typename T = float> static T mishPrime(T x) {
    return static_cast<T>(tanhFloat<T>(softplus<T>(x)) +
                          x * softplusPrime<T>(x) *
                            tanhPrime<T>(tanhFloat<T>(softplus<T>(x))));
  }

  /**
   * @brief setActivation by custom activation function
   * @note  apply derivative as this activation_prime_fn does not utilize
   * derivative
   * @param[in] std::function<Tensor(Tensor const &, Tensor &)> activation_fn
   * activation function to be used
   * @param[in] std::function<Tensor(Tensor const &, Tensor &)>
   * activation_prime_fn activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  template <typename funcParam = Tensor>
  int setActivation(
    std::function<funcParam &(funcParam const &, funcParam &)> const
      &activation_fn,
    std::function<funcParam &(funcParam &, funcParam &,
                              funcParam const &)> const &activation_prime_fn) {
    _act_fn = activation_fn;
    _act_prime_fn = [activation_prime_fn](
                      funcParam const &t_in, funcParam &t_out,
                      funcParam &outgoing_derivative,
                      funcParam const &incoming_derivative) -> funcParam & {
      return activation_prime_fn(t_out, outgoing_derivative,
                                 incoming_derivative);
    };

    return ML_ERROR_NONE;
  }

  /**
   * @brief setActivation by custom activation function
   * @note  derivative not applied here as this activation_prime_fn applies
   * derivative itself
   * @param[in] std::function<Tensor(Tensor const &, Tensor &)> activation_fn
   * activation function to be used
   * @param[in] std::function<Tensor(Tensor const &, Tensor &, Tensor const &)>
   * activation_prime_fn activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  template <typename funcParam = Tensor>
  int setActivation(
    std::function<funcParam &(funcParam const &, funcParam &)> const
      &activation_fn,
    std::function<funcParam &(funcParam const &, funcParam const &, funcParam &,
                              funcParam const &)> const &activation_prime_fn) {
    if (is_inplace)
      return ML_ERROR_INVALID_PARAMETER;

    _act_fn = activation_fn;
    _act_prime_fn = activation_prime_fn;

    return ML_ERROR_NONE;
  }

  /**
   * @brief setActivation by custom activation function
   * @note  derivative not applied here as this activation_prime_fn applies
   * derivative itself
   * @param[in] activation_fn activation function to be used
   * @param[in] activtion_prime_fn activation prime function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  template <typename funcParam = Tensor>
  int setActivation(
    std::function<funcParam &(funcParam const &, funcParam &)> const
      &activation_fn,
    std::function<funcParam &(funcParam &, funcParam &)> const
      &activation_prime_fn) {
    if (!is_inplace) {
      _act_prime_fn = [activation_prime_fn](
                        funcParam const &t_in, funcParam &t_out,
                        funcParam &outgoing_derivative,
                        funcParam const &incoming_derivative) -> funcParam & {
        /** @todo update this based on supportInPlace */
        activation_prime_fn(t_out, outgoing_derivative);
        outgoing_derivative.multiply_i_strided(incoming_derivative);

        return outgoing_derivative;
      };
    } else {
      _act_prime_fn = [activation_prime_fn](
                        funcParam const &t_in, funcParam &t_out,
                        funcParam &outgoing_derivative,
                        funcParam const &incoming_derivative) -> funcParam & {
        activation_prime_fn(t_out, t_out);
        incoming_derivative.multiply_strided(t_out, outgoing_derivative);

        return outgoing_derivative;
      };
    }

    return ML_ERROR_NONE;
  }

  /**
   * @brief setActivation by custom activation function
   * @note  apply derivative as this activation_prime_fn does not utilize
   * derivative
   * @param[in] std::function<float(float const &)> activation_fn activation
   *            function to be used
   * @param[in] std::function<float(float const &)> activation_prime_fn
   *            activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  template <typename funcParam = float>
  int setActivation(
    std::function<funcParam(funcParam const)> const &activation_fn,
    std::function<funcParam(funcParam const)> const &activation_prime_fn) {
    _act_fn = [activation_fn](Tensor const &x, Tensor &hidden) -> Tensor & {
      return x.apply(activation_fn, hidden);
    };
    if (!is_inplace) {
      _act_prime_fn =
        [activation_prime_fn](Tensor const &t_in, Tensor &t_out,
                              Tensor &outgoing_derivative,
                              Tensor const &incoming_derivative) -> Tensor & {
        /** @todo update this based on supportInPlace */
        t_out.apply(activation_prime_fn, outgoing_derivative);
        outgoing_derivative.multiply_i_strided(incoming_derivative);

        return outgoing_derivative;
      };
    } else {
      _act_prime_fn =
        [activation_prime_fn](Tensor const &t_in, Tensor &t_out,
                              Tensor &outgoing_derivative,
                              Tensor const &incoming_derivative) -> Tensor & {
        t_out.apply(activation_prime_fn, t_out);
        incoming_derivative.multiply_strided(t_out, outgoing_derivative);

        return outgoing_derivative;
      };
    }

    return ML_ERROR_NONE;
  }

  /**
   * @brief setActivation by custom activation function
   * @note  apply derivative as this activation_prime_fn does not utilize
   * derivative
   * @param[in] std::function<float(float const)> activation_fn activation
   * function to be used
   * @param[in] std::function<float(float const, float const)>
   * activation_prime_fn activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(
    std::function<float(float const)> const &activation_fn,
    std::function<float(float const, float const)> const &activation_prime_fn);

  /**
   * @brief   Notify that this layer will execute in-place
   *
   * @param val True if execute in-place, else false
   */
  void setInPlace(bool val) {
    if (val && !supportInPlace())
      throw std::runtime_error(
        "Error setting activation layer to work in-place");

    is_inplace = val;
  }

private:
  constexpr static inline float alpha = 1.0f; /**< alpha for elu */
  constexpr static inline float beta = 1.0f;  /**< beta for Softplus */
  constexpr static inline float selu_alpha = 1.67326324f; /**< alpha for selu */
  constexpr static inline float selu_scale = 1.05070098f; /**< scale for selu */

  std::function<Tensor &(Tensor const &, Tensor &)> _act_fn;
  std::function<Tensor &(Tensor const &, Tensor &, Tensor &, Tensor const &)>
    _act_prime_fn; /**< prime function with input and output*/

  ActivationType
    activation_type; /**< type of the activation represented by this */
  bool is_inplace;   /**< if this class should operate is_inplace */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ACTI_FUNC_H__ */
