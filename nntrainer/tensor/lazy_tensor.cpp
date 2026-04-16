// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   lazy_tensor.cpp
 * @date   05 Jun 2020
 * @brief  A lazy evaluation calculator for tensors
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <lazy_tensor.h>
#include <nntrainer_error.h>

namespace nntrainer {

/**
 * @brief Wrapper method of add_i (immediate version of add)
 * @retval this
 */
LazyTensor &LazyTensor::add_i(float const &value) {
  call_chain.push_back(
    [value](Tensor &t) mutable -> int { return t.add_i(value); });
  return *this;
}
/**
 * @brief     Wrapper method of add_i. see tensor.h for more detail
 * @param[in] m Tensor to be added
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::add_i(Tensor const &m, float const alpha) {
  auto f = [&m, alpha](Tensor &t) mutable -> int { return t.add_i(m, alpha); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of subtract_i. see tensor.h for more detail
 * @param[in] m Tensor to subtract
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::subtract_i(Tensor const &m) {
  auto f = [&m](Tensor &t) mutable -> int { return t.subtract_i(m); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of subtract_i. see tensor.h for more detail
 * @param[in] value value to subtract
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::subtract_i(float const &value) {
  auto f = [value](Tensor &t) mutable -> int { return t.subtract_i(value); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief Wrapper method of multiply_i. see tensor.h for more detail
 * @param[in] value to be added
 * @retval LazyTensor *this
 */
LazyTensor &LazyTensor::multiply_i(float const &value) {
  auto f = [value](Tensor &t) mutable -> int { return t.multiply_i(value); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of multiply_i. see tensor.h for more detail
 * @param[in] m Tensor to be multiplied
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::multiply_i(Tensor const &m) {
  auto f = [&m](Tensor &t) mutable -> int { return t.multiply_i(m); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of divide_i. see tensor.h for more detail
 * @param[in] value divisor
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::divide_i(float const &value) {
  auto f = [value](Tensor &t) mutable -> int { return t.divide_i(value); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of divide_i. see tensor.h for more detail
 * @param[in] m Tensor to for division
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::divide_i(Tensor const &m) {
  auto f = [&m](Tensor &t) mutable -> int { return t.divide_i(m); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of pow_i. see tensor.h for more detail
 * @param[in] exponent exponent value
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::pow_i(float exponent) {
  auto f = [exponent](Tensor &t) mutable -> int { return t.pow_i(exponent); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of sqrt_i. see tensor.h for more detail
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::sqrt_i() {
  auto f = [](Tensor &t) mutable -> int { return t.sqrt_i(); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of erf_i. see tensor.h for more detail
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::erf_i() {
  auto f = [](Tensor &t) mutable -> int { return t.erf_i(); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of exp_i. see tensor.h for more detail
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::exp_i() {
  auto f = [](Tensor &t) mutable -> int { return t.exp_i(); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of log_i. see tensor.h for more detail
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::log_i() {
  auto f = [](Tensor &t) mutable -> int { return t.log_i(); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of clamp_i. see tensor.h for more detail
 * @param[in] min minimum value
 * @param[in] max maximum value
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::clamp_i(float min, float max) {
  auto f = [min, max](Tensor &t) mutable -> int {
    return t.clamp_i(min, max);
  };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of inv_sqrt_i. see tensor.h for more detail
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::inv_sqrt_i() {
  auto f = [](Tensor &t) mutable -> int {
    try {
      t.inv_sqrt_i();
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of neg. (memcopy happens)
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::neg() {
  auto f = [](Tensor &t) mutable -> int {
    try {
      t = t.neg();
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of abs. (memcopy happens)
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::abs() {
  auto f = [](Tensor &t) mutable -> int {
    try {
      Tensor out(t.getDim());
      t.abs(out);
      t = out;
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of sin. (memcopy happens)
 * @param[in] alpha scale factor
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::sin(float alpha) {
  auto f = [alpha](Tensor &t) mutable -> int {
    try {
      Tensor out(t.getDim());
      t.sin(out, alpha);
      t = out;
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of cos. (memcopy happens)
 * @param[in] alpha scale factor
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::cos(float alpha) {
  auto f = [alpha](Tensor &t) mutable -> int {
    try {
      Tensor out(t.getDim());
      t.cos(out, alpha);
      t = out;
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of tan. (memcopy happens)
 * @param[in] alpha scale factor
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::tan(float alpha) {
  auto f = [alpha](Tensor &t) mutable -> int {
    try {
      Tensor out(t.getDim());
      t.tan(out, alpha);
      t = out;
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of dot. see tensor.h for more detail (memcopy
 * happens)
 * @param[in] m Tensor
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::dot(Tensor const &m) {
  auto f = [&m](Tensor &t) mutable -> int {
    try {
      t = t.dot(m);
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of transpose. see tensor.h for more detail (memcopy
 * happens)
 * @param[in] direction to transpose ex) 0:2:1
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::transpose(std::string direction) {
  auto f = [direction](Tensor &t) mutable -> int {
    try {
      t = t.transpose(direction);
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of sum. see tensor.h for more detail (memcopy
 * happens)
 * @param[in] direction to transpose ex) 0:2:1
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::sum_by_batch() {
  auto f = [](Tensor &t) mutable -> int {
    try {
      t = t.sum_by_batch();
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of sum. see tensor.h for more detail (memcopy
 * happens) 0 : batch direction 1 : channel direction 2 : channel direction 3 :
 * channel direction
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::sum(int axis) {
  auto f = [axis](Tensor &t) mutable -> int {
    try {
      t = t.sum(axis);
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of average. see tensor.h for more detail (memcopy
 * happens)
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::average(int axis) {
  auto f = [axis](Tensor &t) mutable -> int {
    try {
      t = t.average(axis);
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of average. see tensor.h for more detail (memcopy
 * happens)
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::average() {
  auto f = [](Tensor &t) mutable -> int {
    try {
      t = t.average();
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief execute the call_chain to evaluate
 * @retval calculated tensor
 */
Tensor LazyTensor::run() {
  int status;
  for (auto &item : call_chain) {
    status = item(target);
    if (status != ML_ERROR_NONE) {
      throw std::runtime_error("Error: evaluation failed");
    }
  }
  return target;
}

} /* namespace nntrainer */
