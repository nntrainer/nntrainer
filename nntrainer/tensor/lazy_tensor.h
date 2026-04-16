// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   lazy_tensor.h
 * @date   05 Jun 2020
 * @brief  A lazy evaluation calculator for tensors
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LAZY_TENSOR_H__
#define __LAZY_TENSOR_H__
#ifdef __cplusplus

#include <tensor.h>
#include <vector>

namespace nntrainer {

/**
 * @class   LazyTensor a wrapper class for lazy calculation of tensor
 * @brief   calculation is delayed until Tensor LazyTensor::run() is
 *          called, can be contructed by Tensor::chain() method
 */
class LazyTensor {
public:
  /**
   * @brief Constructor of Lazy Tensor, Tensor is copied to gaurantee
   * immutability
   */
  LazyTensor(const Tensor &from) { target.copy(from); };

  /**
   * @brief     Wrapper method of add_i. see tensor.h for more detail
   * @param[in] value to be added
   * @retval    LazyTensor *this
   */
  LazyTensor &add_i(float const &value);

  /**
   * @brief     Wrapper method of add_i. see tensor.h for more detail
   * @param[in] m Tensor to be added
   * @retval    LazyTensor *this
   */
  LazyTensor &add_i(Tensor const &m, float const alpha = 1);

  /**
   * @brief     Wrapper method of subtract_i. see tensor.h for more detail
   * @param[in] m Tensor to subtract
   * @retval    LazyTensor *this
   */
  LazyTensor &subtract_i(Tensor const &m);

  /**
   * @brief     Wrapper method of subtract_i. see tensor.h for more detail
   * @param[in] value value to subtract
   * @retval    LazyTensor *this
   */
  LazyTensor &subtract_i(float const &value);

  /**
   * @brief Wrapper method of multiply_i. see tensor.h for more detail
   * @param[in] value to be added
   * @retval LazyTensor *this
   */
  LazyTensor &multiply_i(float const &value);

  /**
   * @brief     Wrapper method of multiply_i. see tensor.h for more detail
   * @param[in] m Tensor to be multiplied
   * @retval    LazyTensor *this
   */
  LazyTensor &multiply_i(Tensor const &m);

  /**
   * @brief     Wrapper method of divide_i. see tensor.h for more detail
   * @param[in] value divisor
   * @retval    LazyTensor *this
   */
  LazyTensor &divide_i(float const &value);

  /**
   * @brief     Wrapper method of divide_i. see tensor.h for more detail
   * @param[in] m Tensor to for division
   * @retval    LazyTensor *this
   */
  LazyTensor &divide_i(Tensor const &m);

  /**
   * @brief     Wrapper method of pow_i. see tensor.h for more detail
   * @param[in] exponent exponent value
   * @retval    LazyTensor *this
   */
  LazyTensor &pow_i(float exponent);

  /**
   * @brief     Wrapper method of sqrt_i. see tensor.h for more detail
   * @retval    LazyTensor *this
   */
  LazyTensor &sqrt_i();

  /**
   * @brief     Wrapper method of erf_i. see tensor.h for more detail
   * @retval    LazyTensor *this
   */
  LazyTensor &erf_i();

  /**
   * @brief     Wrapper method of exp_i. see tensor.h for more detail
   * @retval    LazyTensor *this
   */
  LazyTensor &exp_i();

  /**
   * @brief     Wrapper method of log_i. see tensor.h for more detail
   * @retval    LazyTensor *this
   */
  LazyTensor &log_i();

  /**
   * @brief     Wrapper method of clamp_i. see tensor.h for more detail
   * @param[in] min minimum value
   * @param[in] max maximum value
   * @retval    LazyTensor *this
   */
  LazyTensor &clamp_i(float min, float max);

  /**
   * @brief     Wrapper method of inv_sqrt_i. see tensor.h for more detail
   * @retval    LazyTensor *this
   */
  LazyTensor &inv_sqrt_i();

  /**
   * @brief     Wrapper method of neg. (memcopy happens)
   * @retval    LazyTensor *this
   */
  LazyTensor &neg();

  /**
   * @brief     Wrapper method of abs. (memcopy happens)
   * @retval    LazyTensor *this
   */
  LazyTensor &abs();

  /**
   * @brief     Wrapper method of sin. (memcopy happens)
   * @param[in] alpha scale factor (default 1.0)
   * @retval    LazyTensor *this
   */
  LazyTensor &sin(float alpha = 1.0);

  /**
   * @brief     Wrapper method of cos. (memcopy happens)
   * @param[in] alpha scale factor (default 1.0)
   * @retval    LazyTensor *this
   */
  LazyTensor &cos(float alpha = 1.0);

  /**
   * @brief     Wrapper method of tan. (memcopy happens)
   * @param[in] alpha scale factor (default 1.0)
   * @retval    LazyTensor *this
   */
  LazyTensor &tan(float alpha = 1.0);

  /**
   * @brief     Wrapper method of dot. see tensor.h for more detail (memcopy
   * happens)
   * @param[in] m Tensor
   * @retval    LazyTensor *this
   */
  LazyTensor &dot(Tensor const &m);

  /**
   * @brief     Wrapper method of transpose. see tensor.h for more detail
   * (memcopy happens)
   * @param[in] direction to transpose ex) 0:2:1
   * @retval    LazyTensor *this
   */
  LazyTensor &transpose(std::string direction);

  /**
   * @brief     Wrapper method of sum_by_batch. see tensor.h for more detail
   * (memcopy happens)
   * @retval    LazyTensor *this
   */
  LazyTensor &sum_by_batch();

  /**
   * @brief     Wrapper method of sum. see tensor.h for more detail (memcopy
   * happens) 0 : batch direction 1 : channel direction 2 : height direction 3 :
   * width direction
   * @retval    LazyTensor *this
   */
  LazyTensor &sum(int axis);

  /**
   * @brief     Wrapper method of average. see tensor.h for more detail (memcopy
   * happens) 0 : batch direction 1 : channel direction 2 : height direction 3 :
   * width direction
   * @retval    LazyTensor *this
   */
  LazyTensor &average(int axis);

  /**
   * @brief     Wrapper method of average. see tensor.h for more detail (memcopy
   * happens)
   * @retval    LazyTensor *this
   */
  LazyTensor &average();

  /**
   * @brief execute the call_chain to get the tensor
   * @retval calculated tensor
   */
  Tensor run();

private:
  /**< handle the data as a std::vector type */
  std::vector<std::function<int(Tensor &)>> call_chain;
  Tensor target;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LAZY_TENSOR_H__ */
