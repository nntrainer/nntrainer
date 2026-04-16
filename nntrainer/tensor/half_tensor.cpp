// SPDX-License-Identifier: Apache-2.0
/**
 * @file	half_tensor.cpp
 * @date	01 December 2023
 * @brief	This is a HalfTensor class for 16-bit floating point calculation
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <iomanip>
#include <iostream>

#include <compute_ops.h>
#include <half_tensor.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

HalfTensor::HalfTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm, Tdatatype::FP16) {}

HalfTensor::HalfTensor(const TensorDim &d, bool alloc_now, Initializer init,
                       std::string name) :
  TensorBase(d, alloc_now, init, name) {
  if (alloc_now)
    allocate();
}

HalfTensor::HalfTensor(const TensorDim &d, const void *buf) :
  HalfTensor(d, true) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

bool HalfTensor::operator==(const HalfTensor &rhs) const {
  const _FP16 *_data = (_FP16 *)getData();
  const _FP16 *_rdata = (_FP16 *)rhs.getData();
  for (size_t i = 0; i < size(); ++i) {
    if (std::isnan((float)_data[i]) || std::isnan((float)_rdata[i]) ||
        std::fabs((float)(_data[i] - _rdata[i])) > epsilon)
      return false;
  }

  return true;
}

/// @todo support allocation by src_tensor
void HalfTensor::allocate() {
  if (empty() || data)
    /// already allocated
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    mem_data = new MemoryData((void *)(new _FP16[dim.getDataLen()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<_FP16>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void HalfTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *HalfTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<_FP16>() + offset;
}

void *HalfTensor::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<_FP16>() + offset + idx;
}

void *HalfTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((_FP16 *)getData())[i];
}

const void *HalfTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((_FP16 *)getData())[i];
}

const _FP16 &HalfTensor::getValue(unsigned int i) const {
  return ((_FP16 *)getData())[i];
}

_FP16 &HalfTensor::getValue(unsigned int i) { return ((_FP16 *)getData())[i]; }

const _FP16 &HalfTensor::getValue(unsigned int b, unsigned int c,
                                  unsigned int h, unsigned int w) const {
  return getValue(getIndex(b, c, h, w));
}

_FP16 &HalfTensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                            unsigned int w) {
  return getValue(getIndex(b, c, h, w));
}

void HalfTensor::setValue(float value) {
  _FP16 *data = (_FP16 *)getData();
  std::fill(data, data + size(), static_cast<_FP16>(value));
}

void HalfTensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w, float value) {
  ((_FP16 *)getData())[getIndex(b, c, h, w)] = static_cast<_FP16>(value);
}

void HalfTensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w, float value, float beta) {
  auto const &idx = getIndex(b, c, h, w);
  ((_FP16 *)getData())[idx] *= static_cast<_FP16>(beta);
  ((_FP16 *)getData())[idx] += static_cast<_FP16>(value);
}

void HalfTensor::setZero() {
  if (contiguous) {
    //  sscal(size(), 0, (_FP16 *)getData(), 1);
    /// @note we cannot use sscal, when we set zero. if the data is inf or
    /// NaN, then the inf or NaN still remain.
    memset((_FP16 *)getData(), 0, sizeof(_FP16) * size());
  } else {
    /// @todo implement apply_i
    // apply_i<_FP16>([](_FP16 val) -> _FP16 { return 0; });
    setValue(0);
  }
}

void HalfTensor::setRandNormal(float mean, float stddev) {
  setDist<std::normal_distribution<float>>(
    std::normal_distribution<float>(mean, stddev));
}

void HalfTensor::setRandUniform(float min, float max) {
  setDist<std::uniform_real_distribution<float>>(
    std::uniform_real_distribution<float>(min, max));
}

void HalfTensor::setRandBernoulli(float probability) {
  setDist<std::bernoulli_distribution>(
    std::bernoulli_distribution(probability));
}

void HalfTensor::initialize() {
  if (empty() || !isAllocated())
    return;

  unsigned int fan_in, fan_out;

  /// @fixme: when unit is equal to one, this does not work, we need to rely on
  /// effective dimension then actual numbers here. For now, some heuristics
  /// added to infer what would be fan_in/fan_out
  if (dim.batch() * dim.channel() * dim.height() == 1) {
    fan_out = fan_in = dim.width();
  } else if (dim.batch() * dim.channel() == 1) { /// fc layer - 2-D tensor
    fan_in = dim.height();
    fan_out = dim.width();
  } else { /// conv2d filters - 4d tensor, @todo extend this to > 4
    auto field_size = dim.height() * dim.width();

    // this also handles below cases.
    // 1. fan_in = fan_out = 1 as well.
    // 2. batch == 1, channel == 1 and height == 1, theoretical rank of 1
    fan_in = dim.channel() * field_size;
    fan_out = dim.batch() * field_size;
  }

  switch (initializer) {
  case Initializer::ZEROS:
    setZero();
    break;
  case Initializer::ONES:
    setValue(1.0f);
    break;
  case Initializer::LECUN_NORMAL:
    setRandNormal(0.0f, sqrtFloat(1.0f / fan_in));
    break;
  case Initializer::XAVIER_NORMAL:
    setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in + fan_out)));
    break;
  case Initializer::HE_NORMAL:
    setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in)));
    break;
  case Initializer::LECUN_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(1.0f / fan_in), sqrtFloat(1.0f / fan_in));
    break;
  case Initializer::XAVIER_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in + fan_out)),
                   sqrtFloat(6.0 / (fan_in + fan_out)));
    break;
  case Initializer::HE_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in)),
                   sqrtFloat(6.0 / (fan_in)));
  default:
    break;
  }

  putData();
}

void HalfTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

Tensor &HalfTensor::apply(std::function<_FP16(_FP16)> f, Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

  if (contiguous && output.getContiguous()) {
    const _FP16 *data = (_FP16 *)getData();
    _FP16 *rdata = output.getData<_FP16>();

    std::transform(data, data + size(), rdata, f);
  } else if (strides[3] == 1 && output.getStrides()[3] == 1) {
    /** @todo optimize this with combining these loops where stride is 1 */
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          _FP16 *out_data = output.getAddress<_FP16>(b, c, h, 0);
          const _FP16 *in_data = (_FP16 *)getAddress(getIndex(b, c, h, 0));
          std::transform(in_data, in_data + width(), out_data, f);
        }
      }
    }
  } else {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            output.setValue(b, c, h, w, f(getValue(b, c, h, w)));
          }
        }
      }
    }
  }

  return output;
}

Tensor HalfTensor::multiply_strided(Tensor const &m, Tensor &output,
                                    const float beta) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

  if (size() != m.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided multiplication does not support broadcasting");

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData<_FP16>() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData<_FP16>() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  if (strides[3] != 1 || m.getStrides()[3] != 1 ||
      output.getStrides()[3] != 1 || std::fpclassify(beta) != FP_ZERO) {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            output.addValue(
              b, c, h, w, getValue(b, c, h, w) * m.getValue<_FP16>(b, c, h, w),
              beta);
          }
        }
      }
    }
  } else {
    /** @todo optimize by combining these loops where stride is 1 */
    if (this->getFormat() == Tformat::NCHW) {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            _FP16 *out_data = output.getAddress<_FP16>(b, c, h, 0);
            const _FP16 *m_data = m.getAddress<_FP16>(b, c, h, 0);
            const _FP16 *in_data = (_FP16 *)getAddress(getIndex(b, c, h, 0));
            std::transform(in_data, in_data + width(), m_data, out_data,
                           std::multiplies<_FP16>());
          }
        }
      }
    } else {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            _FP16 *out_data = output.getAddress<_FP16>(b, 0, h, w);
            const _FP16 *m_data = m.getAddress<_FP16>(b, 0, h, w);
            const _FP16 *in_data = (_FP16 *)getAddress(getIndex(b, 0, h, w));
            std::transform(in_data, in_data + channel(), m_data, out_data,
                           std::multiplies<_FP16>());
          }
        }
      }
    }
  }

  return output;
}

int HalfTensor::multiply_i(float const &value, ComputeOps *ops) {
  _FP16 *data = (_FP16 *)getData();
  unsigned int len = size();
  getComputeOps()->sscal_fp16(len, value, data, 1);

  return ML_ERROR_NONE;
}

Tensor &HalfTensor::multiply(float const &value, Tensor &out,
                             ComputeOps *ops) const {
  auto f = std::bind(std::multiplies<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, out);
  return out;
}

Tensor &HalfTensor::multiply(Tensor const &m, Tensor &output,
                             const float beta, ComputeOps *ops) const {
  auto f = [&](const BroadcastInfo &e, const _FP16 *buf, const _FP16 *m_buf,
               _FP16 *out_buf) {
    getComputeOps()->ele_mul_fp16(e.buffer_size, buf, m_buf, out_buf, 1, beta, e.strides[3],
            strides[3]);
  };

  NNTR_THROW_IF(m.getFormat() != this->getFormat(), std::invalid_argument)
    << "Tensor Format of " << getName() << ":"
    << ((bool)(this->getFormat()) ? "NHWC" : "NCHW") << " is not match. ("
    << ((bool)(m.getFormat()) ? "NHWC" : "NCHW") << ")";

  NNTR_THROW_IF(!contiguous || !m.getContiguous() || !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  apply_broadcast(m, f, output);
  return output;
}

Tensor &HalfTensor::add_strided(Tensor const &input, Tensor &output,
                                const float beta) const {
  if (size() != input.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided multiplication does not support broadcasting");

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(input.getData<_FP16>() == nullptr, std::invalid_argument)
    << input.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData<_FP16>() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  if (strides[3] != 1 || input.getStrides()[3] != 1 ||
      output.getStrides()[3] != 1 || std::fpclassify(beta) != FP_ZERO) {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            output.setValue(b, c, h, w,
                            getValue(b, c, h, w) +
                              input.getValue<_FP16>(b, c, h, w) * beta);
          }
        }
      }
    }
  } else {
    /** @todo optimize by combining these loops where stride is 1 */
    if (this->getFormat() == Tformat::NCHW) {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            _FP16 *out_data = output.getAddress<_FP16>(b, c, h, 0);
            const _FP16 *in_data = input.getAddress<_FP16>(b, c, h, 0);
            const _FP16 *_data = (_FP16 *)getAddress(getIndex(b, c, h, 0));
            std::transform(_data, _data + width(), in_data, out_data,
                           std::plus<_FP16>());
          }
        }
      }
    } else {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            _FP16 *out_data = output.getAddress<_FP16>(b, 0, h, w);
            const _FP16 *in_data = input.getAddress<_FP16>(b, 0, h, w);
            const _FP16 *_data = (_FP16 *)getAddress(getIndex(b, 0, h, w));
            std::transform(_data, _data + channel(), in_data, out_data,
                           std::plus<_FP16>());
          }
        }
      }
    }
  }

  return output;
}

int HalfTensor::add_i_partial(unsigned int len, unsigned int addr_idx,
                              Tensor &m, unsigned int incX, unsigned int incY,
                              const Tensor alphas, unsigned int alpha_idx,
                              ComputeOps *ops) {
  auto *o = ops ? ops : getComputeOps();
  o->saxpy_fp16(len, alphas.getValue<_FP16>(alpha_idx), m.getData<_FP16>(), incX,
        (_FP16 *)getAddress(addr_idx), incY);

  return ML_ERROR_NONE;
}

Tensor &HalfTensor::add(float const &value, Tensor &output,
                        ComputeOps *ops) const {
  auto f = std::bind(std::plus<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, output);
  return output;
}

Tensor &HalfTensor::add(Tensor const &m, Tensor &output,
                        float const alpha, ComputeOps *ops) const {
  auto f = [&](const BroadcastInfo &e, const _FP16 *buf, const _FP16 *m_buf,
               _FP16 *out_buf) {
    getComputeOps()->ele_add_fp16(e.buffer_size, buf, m_buf, out_buf, alpha, 0, e.strides[3],
            strides[3]);
  };
  apply_broadcast(m, f, output);
  return output;
}

Tensor &HalfTensor::subtract(float const &value, Tensor &output) const {
  auto f = std::bind(std::minus<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, output);
  return output;
}

void HalfTensor::sum_by_batch(Tensor &output, ComputeOps *ops) const {
  auto *o = ops ? ops : getComputeOps();
  size_t feat_len = dim.getFeatureLen();
  size_t batch = dim.batch();

  const _FP16 *data = (_FP16 *)getData();
  _FP16 *out_data = output.getData<_FP16>();

  Tensor ones(1, 1, 1, feat_len, this->getTensorType());
  ones.setValue((_FP16)1.0);
  o->sgemv_fp16((unsigned int)dim.getStorageOrder(), false, batch, feat_len, 1, data,
        feat_len, ones.getData<_FP16>(), 1, 0.0, out_data, 1);
}

Tensor &HalfTensor::sum(unsigned int axis, Tensor &output, float alpha,
                        float beta) const {

  const _FP16 *data = (_FP16 *)getData();

  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  if (axis >= 4)
    throw std::out_of_range("Error: axis is invalid");

  if (dim.getDim()[axis] == 1 and alpha == 1.0 and !beta) {
    CREATE_IF_EMPTY_DIMS(output, dim);
    getComputeOps()->scopy_fp16(size(), (_FP16 *)getData(), 1, output.getData<_FP16>(), 1);
    return output;
  }

  switch (axis) {
  case 0: {
    CREATE_IF_EMPTY_DIMS(output, 1, dim.channel(), dim.height(), dim.width(),
                         this->getTensorType());
    size_t feat_len = dim.getFeatureLen();
    size_t batch = dim.batch();
    Tensor ones(1, 1, 1, batch, this->getTensorType());
    ones.setValue(alpha);
    getComputeOps()->sgemv_fp16((unsigned int)dim.getStorageOrder(), true, batch, feat_len, 1, data,
          feat_len, ones.getData<_FP16>(), 1, beta, output.getData<_FP16>(), 1);
  } break;
  case 1: {
    CREATE_IF_EMPTY_DIMS(output, dim[0], 1, dim[2], dim[3], getTensorType());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int feat_len = output.getDim().getDataLen();
      unsigned int t_axis = dim[1];
      Tensor ones(1, 1, 1, t_axis, this->getTensorType());
      ones.setValue(alpha);
      getComputeOps()->sgemv_fp16((unsigned int)dim.getStorageOrder(), false, feat_len, t_axis, 1,
            data, t_axis, ones.getData<_FP16>(), 1, beta,
            output.getData<_FP16>(), 1);
    } else {
      unsigned int feat_len = dim[2] * dim[3];
      unsigned int t_axis = dim[1];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = output.getData<_FP16>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        getComputeOps()->sgemv_fp16((unsigned int)dim.getStorageOrder(), true, t_axis, feat_len, 1,
              &data[k * dim.getFeatureLen()], feat_len, ones.getData<_FP16>(),
              1, beta, &rdata[k * feat_len], 1);
      }
    }
  } break;
  case 2: {
    CREATE_IF_EMPTY_DIMS(output, dim[0], dim[1], 1, dim[3], getTensorType());

    if (this->getFormat() == Tformat::NHWC) {
      unsigned int feat_len = dim[1] * dim[3];
      unsigned int t_axis = dim[2];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = output.getData<_FP16>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        getComputeOps()->sgemv_fp16((unsigned int)dim.getStorageOrder(), true, t_axis, feat_len, 1,
              &data[k * dim.getFeatureLen()], feat_len, ones.getData<_FP16>(),
              1, beta, &rdata[k * feat_len], 1);
      }
    } else {
      unsigned int t_3 = dim[3];
      unsigned int t_axis = dim[2];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = output.getData<_FP16>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        for (unsigned int c = 0; c < dim[1]; ++c) {
          unsigned int idx = k * dim.getFeatureLen() + c * dim[3] * dim[2];
          unsigned int ridx = k * output.getDim().getFeatureLen() + c * dim[3];
          getComputeOps()->sgemv_fp16((unsigned int)dim.getStorageOrder(), true, t_axis, t_3, 1,
                &data[idx], t_3, ones.getData<_FP16>(), 1, beta, &rdata[ridx],
                1);
        }
      }
    }
  } break;
  case 3: {
    CREATE_IF_EMPTY_DIMS(output, dim[0], dim[1], dim[2], 1, getTensorType());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int t_3 = dim[1];
      unsigned int t_axis = dim[3];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = output.getData<_FP16>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        for (unsigned int c = 0; c < dim[2]; ++c) {
          unsigned int idx = k * dim.getFeatureLen() + c * dim[3] * dim[1];
          unsigned int ridx = k * output.getDim().getFeatureLen() + c * dim[1];
          getComputeOps()->sgemv_fp16((unsigned int)dim.getStorageOrder(), true, t_axis, t_3, 1,
                &data[idx], t_3, ones.getData<_FP16>(), 1, beta, &rdata[ridx],
                1);
        }
      }
    } else {
      unsigned int m = output.getDim().getDataLen();
      unsigned int n = dim[3];
      Tensor ones(1, 1, 1, n, getTensorType());
      ones.setValue(alpha);
      getComputeOps()->sgemv_fp16((unsigned int)dim.getStorageOrder(), false, m, n, 1, data, n,
            ones.getData<_FP16>(), 1, beta, output.getData<_FP16>(), 1);
    }
  } break;
  default:
    throw std::out_of_range("Error: Dimension cannot exceed 3");
  }

  return output;
}

float HalfTensor::l2norm(ComputeOps *ops) const {
  auto *o = ops ? ops : getComputeOps();
  return o->snrm2_fp16(size(), (_FP16 *)getData(), 1);
}

Tensor &HalfTensor::abs(Tensor &output, ComputeOps *ops) const {
  auto f = [](_FP16 in) {
    return static_cast<_FP16>(std::abs(static_cast<float>(in)));
  };
  apply(f, output);
  return output;
}

Tensor &HalfTensor::pow(float exponent, Tensor &output) const {
  auto f = [exponent](float in) {
    return static_cast<_FP16>(powf(in, exponent));
  };
  apply(f, output);
  return output;
}

Tensor &HalfTensor::sqrt(Tensor &output) const {
  auto f = [](_FP16 in) {
    return static_cast<_FP16>(std::sqrt(static_cast<float>(in)));
  };
  apply(f, output);
  return output;
}

Tensor &HalfTensor::erf(Tensor &output) const {
  auto f = [](_FP16 in) {
    return static_cast<_FP16>(std::erf(static_cast<float>(in)));
  };
  apply(f, output);
  return output;
}

Tensor &HalfTensor::exp(Tensor &output) const {
  auto f = [](_FP16 in) {
    return static_cast<_FP16>(std::exp(static_cast<float>(in)));
  };
  apply(f, output);
  return output;
}

Tensor &HalfTensor::log(Tensor &output) const {
  auto f = [](_FP16 in) {
    return static_cast<_FP16>(std::log(static_cast<float>(in)));
  };
  apply(f, output);
  return output;
}

Tensor &HalfTensor::clamp(float min, float max, Tensor &output) const {
  auto f = [min, max](_FP16 in) {
    float val = static_cast<float>(in);
    return static_cast<_FP16>(std::min(std::max(val, min), max));
  };
  apply(f, output);
  return output;
}

void HalfTensor::tan(Tensor &output, float alpha) {
  auto f = [alpha](_FP16 in) -> _FP16 {
    return static_cast<_FP16>(std::tan(static_cast<float>(alpha * in)));
  };
  apply(f, output);
}

void HalfTensor::inv_sqrt(Tensor &out, ComputeOps *ops) {
  if (!contiguous) {
    apply(
      [](_FP16 val) -> _FP16 {
        return static_cast<_FP16>(1 / std::sqrt(static_cast<float>(val)));
      },
      out);
  } else {
    getComputeOps()->inv_sqrt_inplace_fp16(out.size(), out.getData<_FP16>());
  }
}

Tensor &HalfTensor::dotHalf(Tensor const &input, Tensor &output, bool trans,
                            bool trans_in, float beta, ComputeOps *ops) const {
  // Comment out with intension to support the calculation wrt. batch and height
  // direction. It supposes to have this->dim as [ BxCxH,W ] and input.dim is
  // [BxCxH,W] as well if (input.dim.rank() > 2) {
  //   throw exception::not_supported("Error: support only for rank of dot "
  //                                  "matrix <= 2");
  // }

  // Comment out with intension to support the calculation wrt. batch and height
  // direction of this tensor. It is OK as long as input is 2D
  if (trans && dim.rank() > 2) {
    ml_logw("Warning: support only for rank of dot matrix <= 2 with trans");
  }
  unsigned int first_three_flat, last_axis, input_first_three_flat,
    input_last_axis, M, N, K, lda, ldb, ldc;

  calculateFlattenDot(input, output, trans, trans_in, first_three_flat,
                      last_axis, input_first_three_flat, input_last_axis, M, N,
                      K, lda, ldb, ldc);

  const _FP16 *data = (_FP16 *)getData();
  const _FP16 *mdata = input.getData<_FP16>();
  _FP16 *rdata = output.getData<_FP16>();
  const float alpha = 1.0f;

  /// shortcut handling in case of vector
  /// for vector, (1 * K) == (K * 1) in current memory layout...
  /// and plaese note that N, K, M is a fixed place holder after considering
  /// transpose.
  /// For example, there is no case like (1 * K) X (1 * K) while
  /// (1 * K) X (1 * M) can be a case
  /// case1: (1 * K) X (K * 1)
  if (M == 1 && N == 1) {
    *rdata = getComputeOps()->sdot_fp16(K, data, 1, mdata, 1) +
             ((0.0f == beta) ? static_cast<_FP16>(0.0f)
                             : static_cast<_FP16>(beta) * *rdata);
  }
  /// case2: (M * K) X (K * 1)
  else if (N == 1) {
    getComputeOps()->sgemv_fp16((unsigned int)dim.getStorageOrder(), trans, first_three_flat,
          last_axis, alpha, data, lda, mdata, 1, beta, rdata, 1);
  }
  /// case3: (1 * K) X (K * N) = 1 * N = R
  /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
  /// Effectively a translation of sgemv
  else if (M == 1) {
    getComputeOps()->sgemv_fp16((unsigned int)dim.getStorageOrder(), !trans_in,
          input_first_three_flat, input_last_axis, alpha, mdata, ldb, data, 1,
          beta, rdata, 1);
  }
  /// case others: use sgemm
  else {
    getComputeOps()->sgemm_fp16((unsigned int)dim.getStorageOrder(), trans, trans_in, M, N, K, alpha,
          data, lda, mdata, ldb, beta, rdata, ldc);
  }

  return output;
}

Tensor &HalfTensor::dot(Tensor const &input, Tensor &output, bool trans,
                        bool trans_in, float beta, ComputeOps *ops) const {
  switch (input.getDataType()) {
  case Tdatatype::FP16:
    dotHalf(input, output, trans, trans_in, beta);
    break;
  case Tdatatype::Q4_0:
    break;
  default:
    throw std::invalid_argument("Error: unsupported datatype");
  }
  return output;
}

Tensor &HalfTensor::dotQnK(Tensor const &input, Tensor &output, bool trans,
                           bool trans_in, float beta, Tdatatype dtype,
                           ComputeOps *ops) const {
  auto *o = ops ? ops : getComputeOps();
  _FP16 *data = (_FP16 *)getData();
  uint8_t *mdata = input.getData<uint8_t>();
  _FP16 *rdata = output.getData<_FP16>();

  unsigned int M, N, K;
  M = getDim().height();
  K = getDim().width();
  N = trans_in ? input.getDim().height() : input.getDim().width();
  switch (dtype) {
  case Tdatatype::Q4_0:
    o->gemm_q4_0_fp16(M, N, K, data, K, (void *)mdata, N, rdata, N);
    break;
  default:
    throw std::invalid_argument("Error: unsupported datatype");
  }
  return output;
}

void HalfTensor::dropout_mask(float dropout) {
  _FP16 scale = static_cast<_FP16>(1.0 / (1 - dropout));
  _FP16 *data_ = (_FP16 *)getData();
  for (unsigned int i = 0; i < size(); ++i) {
    if (data_[i] >= dropout)
      data_[i] = scale;
    else
      data_[i] = 0;
  }
}

void HalfTensor::filter_mask(const Tensor &mask_len, bool reverse) {
  float fill_mask_val = 0.0;
  float en_mask_val = 1.0 - fill_mask_val;

  if (reverse) {
    fill_mask_val = 1.0;
    en_mask_val = 1.0 - fill_mask_val;
  }

  setValue(fill_mask_val);

  NNTR_THROW_IF(mask_len.batch() != batch(), std::invalid_argument)
    << "Number of filter masks mismatched";

  for (unsigned int b = 0; b < batch(); b++) {
    _FP16 *addr = (_FP16 *)getAddress(getIndex(b, 0, 0, 0));
    const unsigned int *mask_len_val =
      mask_len.getAddress<unsigned int>(b, 0, 0, 0);
    std::fill(addr, addr + (*mask_len_val), (_FP16)en_mask_val);
  }
}

void HalfTensor::zoneout_mask(Tensor &opposite, float zoneout) {
  _FP16 zoneout_fp16 = (_FP16)zoneout;
  opposite.setRandBernoulli(zoneout_fp16);

  _FP16 *data = (_FP16 *)getData();
  _FP16 *opposite_data = opposite.getData<_FP16>();

  for (unsigned int i = 0; i < size(); ++i) {
    if (opposite_data[i] > epsilon) {
      data[i] = (_FP16)0.0;
    } else {
      data[i] = (_FP16)1.0;
    }
  }
}

std::vector<Tensor> HalfTensor::split(std::vector<size_t> sizes, int axis) {
  size_t num_size = sizes.size();

  if (axis == -1) {
    axis = 3;
  }

  size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
  NNTR_THROW_IF(dim.getTensorDim(axis) != total_size, std::invalid_argument)
    << "given sum of sizes did not match with origin tensor dim, tensor dim: "
    << dim.getTensorDim(axis) << " total size: " << total_size;

  std::vector<TensorDim> ret_dims(num_size, dim);
  for (unsigned int i = 0; i < num_size; ++i) {
    ret_dims[i].setTensorDim(axis, sizes[i]);
  }

  bool is_format_nchw = (dim.getFormat() == Tformat::NCHW) ? true : false;
  std::vector<Tensor> ret;

  auto iter_value = [this, is_format_nchw](
                      std::array<size_t, 4> &loc,
                      const std::array<size_t, 4> &end_loc,
                      const std::array<size_t, 4> &reset_dim_arr) -> _FP16 & {
    auto &value = (is_format_nchw) ? getValue(loc[0], loc[1], loc[2], loc[3])
                                   : getValue(loc[0], loc[3], loc[1], loc[2]);
    for (int i = 3; i >= 0; --i) {
      loc[i]++;
      if (loc[i] == end_loc[i]) {
        loc[i] -= reset_dim_arr[i];
        continue;
      }
      break;
    }
    return value;
  };

  unsigned int accumulated_size = 0;
  for (unsigned int i = 0; i < num_size; ++i) {
    std::array<size_t, 4> loc = {0, 0, 0, 0};

    if (is_format_nchw) {
      loc[axis] += accumulated_size;
    } else {
      if (axis == 0) {
        loc[0] += accumulated_size;
      } else if (axis == 1) {
        loc[3] += accumulated_size;
      } else if (axis == 2 || axis == 3) {
        loc[axis - 1] += accumulated_size;
      }
    }

    ret.push_back(Tensor(ret_dims[i]));
    auto &ret_t = ret.back();

    std::array<size_t, 4> end_loc;

    if (is_format_nchw) {
      end_loc = {ret_dims[i].batch(), ret_dims[i].channel(),
                 ret_dims[i].height(), ret_dims[i].width()};
    } else {
      end_loc = {ret_dims[i].batch(), ret_dims[i].height(), ret_dims[i].width(),
                 ret_dims[i].channel()};
    }

    accumulated_size += sizes[i];

    if (is_format_nchw) {
      end_loc[axis] = accumulated_size;
    } else {
      if (axis == 0) {
        end_loc[0] = accumulated_size;
      } else if (axis == 1) {
        end_loc[3] = accumulated_size;
      } else if (axis == 2 || axis == 3) {
        end_loc[axis - 1] = accumulated_size;
      }
    }

    std::array<size_t, 4> reset_dim_arr;
    if (is_format_nchw) {
      reset_dim_arr = {ret_dims[i].batch(), ret_dims[i].channel(),
                       ret_dims[i].height(), ret_dims[i].width()};
    } else {
      reset_dim_arr = {ret_dims[i].batch(), ret_dims[i].height(),
                       ret_dims[i].width(), ret_dims[i].channel()};
    }

    ret_t.apply_i<_FP16>(
      [&iter_value, &loc, &end_loc, &reset_dim_arr](_FP16 _) {
        return iter_value(loc, end_loc, reset_dim_arr);
      });
  }

  return ret;
}

Tensor HalfTensor::concat(const std::vector<Tensor> &tensors, int axis,
                          Tensor &output) {
  bool is_format_nchw = (tensors.front().getDim().getFormat() == Tformat::NCHW);

  auto iter_value =
    [is_format_nchw](std::array<unsigned, 4> &loc,
                     const std::array<unsigned, 4> &start_loc, Tensor &t,
                     const std::array<unsigned, 4> &ref_dim_arr) -> _FP16 & {
    auto &value = is_format_nchw
                    ? t.getValue<_FP16>(loc[0], loc[1], loc[2], loc[3])
                    : t.getValue<_FP16>(loc[0], loc[3], loc[1], loc[2]);

    for (int i = 3; i >= 0; --i) {
      loc[i]++;
      if (loc[i] - start_loc[i] == ref_dim_arr[i]) {
        loc[i] = start_loc[i];
        continue;
      }
      break;
    }
    return value;
  };

  std::array<unsigned, 4> loc = {0, 0, 0, 0};
  for (auto &t : tensors) {
    std::array<unsigned, 4> start_loc = loc;
    std::array<unsigned, 4> tensor_dim_arr;
    TensorDim curr_dim = t.getDim();

    tensor_dim_arr[0] = curr_dim.getTensorDim(0);
    tensor_dim_arr[1] =
      is_format_nchw ? curr_dim.getTensorDim(1) : curr_dim.getTensorDim(2);
    tensor_dim_arr[2] =
      is_format_nchw ? curr_dim.getTensorDim(2) : curr_dim.getTensorDim(3);
    tensor_dim_arr[3] =
      is_format_nchw ? curr_dim.getTensorDim(3) : curr_dim.getTensorDim(1);

    for (size_t i = 0u, sz = t.size(); i < sz; ++i) {
      iter_value(loc, start_loc, output, tensor_dim_arr) = t.getValue<_FP16>(i);
    }

    if (is_format_nchw) {
      loc[axis] += curr_dim.getTensorDim(axis);
    } else {
      if (axis == 0) {
        loc[0] += curr_dim.getTensorDim(axis);
      } else if (axis == 1) {
        loc[3] += curr_dim.getTensorDim(axis);
      } else if (axis == 2 || axis == 3) {
        loc[axis - 1] += curr_dim.getTensorDim(axis);
      }
    }
  }
  return output;
}

void HalfTensor::print(std::ostream &out) const {
  const _FP16 *data = (_FP16 *)getData();
  unsigned int len = size();
  out << "data addr: " << data << '\n';
  out << dim;

  if (len > 100) {
    out << '[' << (float)data[0] << ' ' << (float)data[1] << ' '
        << (float)data[2] << " ... " << (float)data[len - 3] << ' '
        << (float)data[len - 2] << ' ' << (float)data[len - 1] << ']'
        << std::endl;
    return;
  }

  std::ios init(NULL);
  init.copyfmt(out);

  if (getFormat() == Tformat::NCHW) {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int l = 0; l < channel(); l++) {
        for (unsigned int i = 0; i < height(); i++) {
          for (unsigned int j = 0; j < width(); j++) {
            out << std::setw(10) << std::setprecision(10)
                << (float)data[getIndex(k, l, i, j)] << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  } else {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int i = 0; i < height(); i++) {
        for (unsigned int j = 0; j < width(); j++) {
          for (unsigned int l = 0; l < channel(); l++) {
            out << std::setw(10) << std::setprecision(10)
                << (float)data[getIndex(k, l, i, j)] << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  }
  out.copyfmt(init);
}

Tensor &HalfTensor::divide(float const &value, Tensor &output,
                           ComputeOps *ops) const {
  auto f = std::bind(std::divides<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, output);
  return output;
}

Tensor &HalfTensor::divide(Tensor const &m, Tensor &output, ComputeOps *ops) const {
  auto f = [&](const BroadcastInfo &e, const _FP16 *buf, const _FP16 *m_buf,
               _FP16 *out_buf) {
    getComputeOps()->ele_div_fp16(e.buffer_size, buf, m_buf, out_buf, 1, 0, e.strides[3], strides[3]);
  };

  apply_broadcast(m, f, output);
  return output;
}

void HalfTensor::copy(const Tensor &from, ComputeOps *ops) {
  reshape(from.getDim());
  copy(from.getData<_FP16>(), ops);
}

void HalfTensor::copyData(const Tensor &from, ComputeOps *ops) {
  auto *o = ops ? ops : getComputeOps();
  if (!contiguous) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (size() != from.size())
    throw std::invalid_argument("Size of tensor to copy must match");

  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    o->scopy_fp32_to_fp16(size(), from.getData<float>(), 1, (_FP16 *)getData(), 1);
    break;
  case ml::train::TensorDim::DataType::FP16:
    copy(from.getData<_FP16>(), ops);
    break;
  case ml::train::TensorDim::DataType::QINT8:
    o->scopy_int8_to_float16_s(from.size(), from.getData<int8_t>(), 1,
                          (_FP16 *)getData(), 1);
    break;
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

void HalfTensor::copy_with_stride(const Tensor &input, Tensor &output) {
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w, input.getValue<_FP16>(b, c, h, w));
        }
      }
    }
  }
}

std::vector<unsigned int> HalfTensor::argmax() const {
  std::vector<unsigned int> result;
  const _FP16 *data = (_FP16 *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();

  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; b++) {
    auto max_iter =
      std::max_element(data + b * feature_len, data + (b + 1) * feature_len);
    result[b] = std::distance(data, max_iter) - (b * feature_len);
  }

  return result;
}

std::vector<unsigned int> HalfTensor::argmin() const {
  std::vector<unsigned int> result;
  const _FP16 *data = (_FP16 *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();

  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; b++) {
    auto min_iter =
      std::min_element(data + b * feature_len, data + (b + 1) * feature_len);
    result[b] = std::distance(data, min_iter) - (b * feature_len);
  }
  return result;
}

float HalfTensor::max_abs(ComputeOps *ops) const {
  auto *o = ops ? ops : getComputeOps();
  const _FP16 *data = (_FP16 *)getData();
  unsigned int idx = o->isamax_fp16(size(), data, 1);
  return (float)(*(data + idx));
}

float HalfTensor::maxValue() const {
  const _FP16 *data = (_FP16 *)getData();
  return (float)*std::max_element(data, data + size());
}

float HalfTensor::minValue() const {
  const _FP16 *data = (_FP16 *)getData();
  return (float)*std::min_element(data, data + size());
}

Tensor &HalfTensor::transpose(const std::string &direction,
                              Tensor &output, ComputeOps *ops) const {
  unsigned int SL, SI, SJ, SK;

  output.reshape(dim.transpose(direction));

  int indexI = direction[0] - '0';
  int indexJ = direction[2] - '0';

  SL = dim.batch(), SI = dim.channel(), SJ = dim.height(), SK = dim.width();

  bool is_format_nchw = (getFormat() == Tformat::NCHW);

  const _FP16 *inptr = (_FP16 *)getData();
  _FP16 *outptr = output.getData<_FP16>();
  switch (indexI) {
  case 0:
    if (indexJ == 1) {
      if (is_format_nchw) {
        transposeloop(l, i, j, k, SL, SI, SJ, SK);
      } else {
        transposeloop_nhwc(l, j, k, i, SL, SJ, SK, SI);
      }
    } else {
      if (is_format_nchw) {
        for (unsigned int b = 0; b < batch(); ++b) {
          for (unsigned int c = 0; c < channel(); ++c) {
            getComputeOps()->transpose_matrix_fp16(
              height(), width(), (_FP16 *)getData() + getIndex(b, c, 0, 0),
              width(), (_FP16 *)output.getData() + output.getIndex(b, c, 0, 0),
              output.width());
          }
        }
      } else {
        transposeloop_nhwc(l, k, j, i, SL, SK, SJ, SI);
      }
    }
    break;
  case 1:
    if (indexJ == 0) {
      if (is_format_nchw) {
        transposeloop(l, j, i, k, SL, SJ, SI, SK);
      } else {
        transposeloop_nhwc(l, i, k, j, SL, SI, SK, SJ);
      }
    } else {
      if (is_format_nchw) {
        transposeloop(l, j, k, i, SL, SJ, SK, SI);
      } else {
        transposeloop_nhwc(l, k, i, j, SL, SK, SI, SJ);
      }
    }
    break;
  case 2:
    if (indexJ == 0) {
      if (is_format_nchw) {
        transposeloop(l, k, i, j, SL, SK, SI, SJ);
      } else {
        transposeloop_nhwc(l, i, j, k, SL, SI, SJ, SK);
      }
    } else {
      if (is_format_nchw) {
        transposeloop(l, k, j, i, SL, SK, SJ, SI);
      } else {
        transposeloop_nhwc(l, j, i, k, SL, SJ, SI, SK);
      }
    }
    break;
  }

  return output;
}

void HalfTensor::copy(const void *buf, ComputeOps *ops) {
  auto *o = ops ? ops : getComputeOps();
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  o->scopy_fp16(size(), (_FP16 *)buf, 1, (_FP16 *)getData(), 1);
}

void HalfTensor::apply_broadcast(
  Tensor const &m,
  std::function<void(const BroadcastInfo &e, const _FP16 *, const _FP16 *,
                     _FP16 *)>
    v_func,
  Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData<_FP16>() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData<_FP16>() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  /// shortcut to cover when dimension matches
  /// note that buffer_size, the last stride is only used in v_func but it
  /// might be changed
  if (dim == m.getDim()) {
    BroadcastInfo e;
    e.buffer_size = size();
    e.strides[3] = 1;
    v_func(e, (_FP16 *)getData(), m.getData<_FP16>(), output.getData<_FP16>());
    return;
  }

  return apply_broadcast_util(m, v_func, output, this->computeBroadcastInfo(m));
}

void HalfTensor::apply_broadcast_util(
  Tensor const &m,
  std::function<void(const BroadcastInfo &e, const _FP16 *, const _FP16 *,
                     _FP16 *)>
    v_func,
  Tensor &output, const BroadcastInfo &e, int cur_axis, size_t offset,
  size_t m_offset) const {

  const _FP16 *buf = (_FP16 *)this->getData();
  const _FP16 *m_buf = m.getData<_FP16>();
  _FP16 *out_buf = output.getData<_FP16>();

  if (e.buffer_axis == cur_axis) {
    v_func(e, buf + offset, m_buf + m_offset, out_buf + offset);
    return;
  }

  cur_axis++;
  for (unsigned int i = 0; i < dim.getTensorDim(cur_axis); ++i) {
    size_t next_offset = offset + i * strides[cur_axis];
    size_t next_m_offset = m_offset + i * e.strides[cur_axis];
    apply_broadcast_util(m, v_func, output, e, cur_axis, next_offset,
                         next_m_offset);
  }
}

bool HalfTensor::isValid(ComputeOps *ops) const {
  auto *o = ops ? ops : getComputeOps();
  return o->is_valid_fp16(dim.getDataLen(), (_FP16 *)getData());
}

} // namespace nntrainer
