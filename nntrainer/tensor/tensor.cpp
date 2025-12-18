// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor.cpp
 * @date	01 December 2023
 * @brief	This is a Tensor class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <numeric>

#include <char_tensor.h>
#include <float_tensor.h>
#include <int4_tensor.h>
#include <lazy_tensor.h>
#include <q4_0_tensor.h>
#include <q4_k_tensor.h>
#include <q6_k_tensor.h>
#include <short_tensor.h>
#include <tensor.h>
#include <uint4_tensor.h>
#include <uint_tensor.h>

#ifdef ENABLE_FP16
#include <half_tensor.h>
#endif

#ifdef ENABLE_BIQGEMM
#include <bcq_tensor.h>
#endif

#include <fcntl.h>

#if defined(__unix__) || defined(__ANDROID__) || defined(__arm__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nntrainer {

Tensor::Tensor(
  std::vector<std::vector<std::vector<std::vector<int16_t>>>> const &d,
  std::vector<float> const &scales, ml::train::TensorDim::TensorType t_type,
  QScheme qscheme_) {
  switch (qscheme_) {
  case QScheme::PER_TENSOR_AFFINE:
    break;
  case QScheme::PER_CHANNEL_AFFINE:
    break;
  default:
    break;
  }
  itensor_ = std::make_unique<ShortTensor>(d, scales, t_type.format, qscheme_);
}

Tensor::Tensor(
  std::vector<std::vector<std::vector<std::vector<int8_t>>>> const &d,
  std::vector<float> const &scales, ml::train::TensorDim::TensorType t_type,
  QScheme qscheme_) {
  if (t_type.data_type == Tdatatype::QINT4) {
    itensor_ =
      std::make_unique<Int4QTensor>(d, scales, t_type.format, qscheme_);
  } else if (t_type.data_type == Tdatatype::QINT8) {
    itensor_ = std::make_unique<CharTensor>(d, scales, t_type.format, qscheme_);
  } else {
    throw std::invalid_argument(
      "Error: Tensor cannot be constructed because the given data type is "
      "incorrect. The supported d_types are: QINT4, QINT8");
  }
}

Tensor::Tensor(
  std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
  ml::train::TensorDim::TensorType t_type) {
  itensor_ = std::make_unique<FloatTensor>(d, t_type.format);
}

Tensor::Tensor(
  std::vector<std::vector<std::vector<std::vector<uint8_t>>>> const &d,
  std::vector<float> const &scales,
  std::vector<unsigned int> const &zero_points,
  ml::train::TensorDim::TensorType t_type, QScheme qscheme_) {
  if (t_type.data_type == Tdatatype::UINT4) {
    itensor_ = std::make_unique<Uint4QTensor>(d, scales, zero_points,
                                              t_type.format, qscheme_);
  } else if (t_type.data_type == Tdatatype::UINT8) {
    itensor_ = std::make_unique<UInt8Tensor>(d, scales, zero_points,
                                             t_type.format, qscheme_);
  } else {
    throw std::invalid_argument(
      "Error: Tensor cannot be constructed because the given data type is "
      "incorrect. The supported d_types are: UINT4, UINT8");
  }
}

Tensor::Tensor(
  std::vector<std::vector<std::vector<std::vector<uint16_t>>>> const &d,
  std::vector<float> const &scales,
  std::vector<unsigned int> const &zero_points,
  ml::train::TensorDim::TensorType t_type, QScheme qscheme_) {
  itensor_ = std::make_unique<UInt16Tensor>(d, scales, zero_points,
                                            t_type.format, qscheme_);
}

Tensor::Tensor(
  std::vector<std::vector<std::vector<std::vector<uint32_t>>>> const &d,
  std::vector<float> const &scales,
  std::vector<unsigned int> const &zero_points,
  ml::train::TensorDim::TensorType t_type, QScheme qscheme_) {
  itensor_ = std::make_unique<UInt32Tensor>(d, scales, zero_points,
                                            t_type.format, qscheme_);
}

Tensor::Tensor(std::string name_, Tformat fm, Tdatatype d_type) {
  itensor_ = nullptr;

  if (d_type == Tdatatype::FP32) {
    itensor_ = std::make_unique<FloatTensor>(name_, fm);
  } else if (d_type == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor_ = std::make_unique<HalfTensor>(name_, fm);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else if (d_type == Tdatatype::Q4_K) {
    itensor_ = std::make_unique<Q4_K_Tensor>(name_, fm);
  } else if (d_type == Tdatatype::Q6_K) {
    itensor_ = std::make_unique<Q6_K_Tensor>(name_, fm);
  } else if (d_type == Tdatatype::Q4_0) {
    itensor_ = std::make_unique<Q4_0_Tensor>(name_, fm);
  } else if (d_type == Tdatatype::UINT4) {
    itensor_ = std::make_unique<Uint4QTensor>(name_, fm);
  } else if (d_type == Tdatatype::UINT8) {
    itensor_ = std::make_unique<UInt8Tensor>(name_, fm);
  } else if (d_type == Tdatatype::UINT16) {
    itensor_ = std::make_unique<UInt16Tensor>(name_, fm);
  } else if (d_type == Tdatatype::UINT32) {
    itensor_ = std::make_unique<UInt32Tensor>(name_, fm);
  } else if (d_type == Tdatatype::QINT16) {
    itensor_ = std::make_unique<ShortTensor>(name_, fm);
  } else if (d_type == Tdatatype::QINT8) {
    itensor_ = std::make_unique<CharTensor>(name_, fm);
  } else if (d_type == Tdatatype::QINT4) {
    itensor_ = std::make_unique<Int4QTensor>(name_, fm);
  } else if (d_type == Tdatatype::BCQ) {
#ifdef ENABLE_BIQGEMM
    itensor_ = std::make_unique<BCQTensor>(name_, fm);
#else
    throw std::invalid_argument("Error: enable-biqgemm is not activated. "
                                "Enable only if your system supports BiQGEMM.");
#endif
  } else {
    throw std::invalid_argument(
      "Error: Tensor cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

Tensor::Tensor(const TensorDim &d, bool alloc_now, Initializer init,
               std::string name, QScheme qscheme, bool is_virtual) {
  itensor_ = nullptr;
  this->is_virtual = is_virtual;

  if (d.getDataType() == Tdatatype::FP32) {
    itensor_ = std::make_unique<FloatTensor>(d, alloc_now, init, name);
  } else if (d.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor_ = std::make_unique<HalfTensor>(d, alloc_now, init, name);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else if (d.getDataType() == Tdatatype::Q4_K) {
    itensor_ = std::make_unique<Q4_K_Tensor>(d, alloc_now, init, name);
  } else if (d.getDataType() == Tdatatype::Q6_K) {
    itensor_ = std::make_unique<Q6_K_Tensor>(d, alloc_now, init, name);
  } else if (d.getDataType() == Tdatatype::Q4_0) {
    itensor_ = std::make_unique<Q4_0_Tensor>(d, alloc_now, init, name);
  } else if (d.getDataType() == Tdatatype::UINT4) {
    if (qscheme != QScheme::Q4_Kx8) {
      itensor_ =
        std::make_unique<Uint4QTensor>(d, alloc_now, init, name, qscheme);
    } else {
      itensor_ =
        std::make_unique<Q4_K_Tensor>(d, alloc_now, init, name, qscheme);
    }
  } else if (d.getDataType() == Tdatatype::UINT8) {
    itensor_ = std::make_unique<UInt8Tensor>(d, alloc_now, init, name);
  } else if (d.getDataType() == Tdatatype::UINT16) {
    itensor_ = std::make_unique<UInt16Tensor>(d, alloc_now, init, name);
  } else if (d.getDataType() == Tdatatype::UINT32) {
    itensor_ = std::make_unique<UInt32Tensor>(d, alloc_now, init, name);
  } else if (d.getDataType() == Tdatatype::QINT16) {
    itensor_ = std::make_unique<ShortTensor>(d, alloc_now, init, name, qscheme);
  } else if (d.getDataType() == Tdatatype::QINT8) {
    itensor_ = std::make_unique<CharTensor>(d, alloc_now, init, name, qscheme);
  } else if (d.getDataType() == Tdatatype::QINT4) {
    itensor_ = std::make_unique<Int4QTensor>(d, alloc_now, init, name, qscheme);
  } else if (d.getDataType() == Tdatatype::BCQ) {
#ifdef ENABLE_BIQGEMM
    itensor_ = std::make_unique<BCQTensor>(d, alloc_now, init, name);
#else
    throw std::invalid_argument("Error: enable-biqgemm is not activated. "
                                "Enable only if your system supports BiQGEMM.");
#endif
  } else {
    throw std::invalid_argument(
      "Error: Tensor cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

Tensor::Tensor(const TensorDim &d, const void *buf, QScheme qscheme) {
  itensor_ = nullptr;

  if (d.getDataType() == Tdatatype::FP32) {
    itensor_ = std::make_unique<FloatTensor>(d, buf);
  } else if (d.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor_ = std::make_unique<HalfTensor>(d, buf);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else if (d.getDataType() == Tdatatype::Q4_K) {
    itensor_ = std::make_unique<Q4_K_Tensor>(d, buf);
  } else if (d.getDataType() == Tdatatype::Q6_K) {
    itensor_ = std::make_unique<Q6_K_Tensor>(d, buf);
  } else if (d.getDataType() == Tdatatype::Q4_0) {
    itensor_ = std::make_unique<Q4_0_Tensor>(d, buf);
  } else if (d.getDataType() == Tdatatype::UINT4) {
    if (qscheme != QScheme::Q4_Kx8)
      itensor_ = std::make_unique<Uint4QTensor>(d, buf, qscheme);
    else
      itensor_ = std::make_unique<Q4_K_Tensor>(d, buf, qscheme);
  } else if (d.getDataType() == Tdatatype::UINT8) {
    itensor_ = std::make_unique<UInt8Tensor>(d, buf);
  } else if (d.getDataType() == Tdatatype::UINT16) {
    itensor_ = std::make_unique<UInt16Tensor>(d, buf);
  } else if (d.getDataType() == Tdatatype::UINT32) {
    itensor_ = std::make_unique<UInt32Tensor>(d, buf);
  } else if (d.getDataType() == Tdatatype::QINT16) {
    itensor_ = std::make_unique<ShortTensor>(d, buf, qscheme);
  } else if (d.getDataType() == Tdatatype::QINT8) {
    itensor_ = std::make_unique<CharTensor>(d, buf, qscheme);
  } else if (d.getDataType() == Tdatatype::QINT4) {
    itensor_ = std::make_unique<Int4QTensor>(d, buf);
  } else if (d.getDataType() == Tdatatype::BCQ) {
#ifdef ENABLE_BIQGEMM
    itensor_ = std::make_unique<BCQTensor>(d, buf);
#else
    throw std::invalid_argument("Error: enable-biqgemm is not activated. "
                                "Enable only if your system supports BiQGEMM.");
#endif
  } else {
    throw std::invalid_argument(
      "Error: Tensor cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

Tensor::Tensor(const Tensor &rhs) {
  if (rhs.getDataType() == Tdatatype::FP32) {
    itensor_ = std::make_unique<FloatTensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor_ = std::make_unique<HalfTensor>(*rhs.itensor_);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else if (rhs.getDataType() == Tdatatype::Q4_K) {
    itensor_ = std::make_unique<Q4_K_Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::Q6_K) {
    itensor_ = std::make_unique<Q6_K_Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::Q4_0) {
    itensor_ = std::make_unique<Q4_0_Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::UINT4) {
    itensor_ = std::make_unique<Uint4QTensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::UINT8) {
    itensor_ = std::make_unique<UInt8Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::UINT16) {
    itensor_ = std::make_unique<UInt16Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::UINT32) {
    itensor_ = std::make_unique<UInt32Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::QINT16) {
    itensor_ = std::make_unique<ShortTensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::QINT8) {
    itensor_ = std::make_unique<CharTensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::QINT4) {
    itensor_ = std::make_unique<Int4QTensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::BCQ) {
#ifdef ENABLE_BIQGEMM
    itensor_ = std::make_unique<BCQTensor>(*rhs.itensor_);
#else
    throw std::invalid_argument("Error: enable-biqgemm is not activated. "
                                "Enable only if your system supports BiQGEMM.");
#endif
  }

  /** copy tensor properties */
  this->is_virtual = rhs.is_virtual;
  this->fd = rhs.fd;
  this->read_offset = rhs.read_offset;
  this->mapped_ptr = rhs.mapped_ptr;
}

Tensor::Tensor(const std::unique_ptr<TensorBase> &rhs) {
  NNTR_THROW_IF(rhs.get() == nullptr, std::invalid_argument)
    << "Error: received a nullptr. Tensor cannot be constructed";

  if (rhs->getDataType() == Tdatatype::FP32) {
    itensor_ = std::make_unique<FloatTensor>(*rhs.get());
  } else if (rhs->getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor_ = std::make_unique<HalfTensor>(*rhs.get());
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else if (rhs->getDataType() == Tdatatype::UINT4) {
    itensor_ = std::make_unique<Uint4QTensor>(*rhs.get());
  } else if (rhs->getDataType() == Tdatatype::UINT8) {
    itensor_ = std::make_unique<UInt8Tensor>(*rhs.get());
  } else if (rhs->getDataType() == Tdatatype::UINT16) {
    itensor_ = std::make_unique<UInt16Tensor>(*rhs.get());
  } else if (rhs->getDataType() == Tdatatype::UINT32) {
    itensor_ = std::make_unique<UInt32Tensor>(*rhs.get());
  } else if (rhs->getDataType() == Tdatatype::QINT16) {
    itensor_ = std::make_unique<ShortTensor>(*rhs.get());
  } else if (rhs->getDataType() == Tdatatype::QINT8) {
    itensor_ = std::make_unique<CharTensor>(*rhs.get());
  } else if (rhs->getDataType() == Tdatatype::QINT4) {
    itensor_ = std::make_unique<Int4QTensor>(*rhs.get());
  } else if (rhs->getDataType() == Tdatatype::BCQ) {
#ifdef ENABLE_BIQGEMM
    itensor_ = std::make_unique<BCQTensor>(*rhs.get());
#else
    throw std::invalid_argument("Error: enable-biqgemm is not activated. "
                                "Enable only if your system supports BiQGEMM.");
#endif
  }
}

Tensor &Tensor::operator=(const Tensor &rhs) {
  if (rhs.getDataType() == Tdatatype::FP32) {
    itensor_ = std::make_unique<FloatTensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor_ = std::make_unique<HalfTensor>(*rhs.itensor_);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else if (rhs.getDataType() == Tdatatype::Q4_K) {
    itensor_ = std::make_unique<Q4_K_Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::Q6_K) {
    itensor_ = std::make_unique<Q6_K_Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::Q4_0) {
    itensor_ = std::make_unique<Q4_0_Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::UINT4) {
    itensor_ = std::make_unique<Uint4QTensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::UINT8) {
    itensor_ = std::make_unique<UInt8Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::UINT16) {
    itensor_ = std::make_unique<UInt16Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::UINT32) {
    itensor_ = std::make_unique<UInt32Tensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::QINT16) {
    itensor_ = std::make_unique<ShortTensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::QINT8) {
    itensor_ = std::make_unique<CharTensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::QINT4) {
    itensor_ = std::make_unique<Int4QTensor>(*rhs.itensor_);
  } else if (rhs.getDataType() == Tdatatype::BCQ) {
#ifdef ENABLE_BIQGEMM
    itensor_ = std::make_unique<BCQTensor>(*rhs.itensor_);
#else
    throw std::invalid_argument("Error: enable-biqgemm is not activated. "
                                "Enable only if your system supports BiQGEMM.");
#endif
  }

  /** copy tensor properties */
  this->is_virtual = rhs.is_virtual;
  this->fd = rhs.fd;
  this->read_offset = rhs.read_offset;
  this->mapped_ptr = rhs.mapped_ptr;
  return *this;
}

bool Tensor::operator==(const Tensor &rhs) const {
  /// compares tensor information
  if (*itensor_.get() == *rhs.itensor_.get()) {
    /// compares tensor data
    if (getDataType() == Tdatatype::FP32) {
      return itensorCompare<FloatTensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
      return itensorCompare<HalfTensor>(itensor_.get(), rhs.itensor_.get());
#else
      throw std::invalid_argument(
        "Error: HalfTensor cannot be created or used when FP16 is not enabled. "
        "Please check if the tensor data type is set properly.");
#endif
    } else if (getDataType() == Tdatatype::Q4_K) {
      return itensorCompare<Q4_K_Tensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::Q6_K) {
      return itensorCompare<Q6_K_Tensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::Q4_0) {
      return itensorCompare<Q4_0_Tensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::UINT4) {
      return itensorCompare<Uint4QTensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::UINT8) {
      return itensorCompare<UInt8Tensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::UINT16) {
      return itensorCompare<UInt16Tensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::UINT32) {
      return itensorCompare<UInt32Tensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::QINT16) {
      return itensorCompare<ShortTensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::QINT8) {
      return itensorCompare<CharTensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::QINT4) {
      return itensorCompare<Int4QTensor>(itensor_.get(), rhs.itensor_.get());
    } else if (getDataType() == Tdatatype::BCQ) {
#ifdef ENABLE_BIQGEMM
      return itensorCompare<BCQTensor>(itensor_.get(), rhs.itensor_.get());
#else
      throw std::invalid_argument(
        "Error: enable-biqgemm is not activated. "
        "Enable only if your system supports BiQGEMM.");
#endif
    }
  }
  return false;
}

void Tensor::allocate() { itensor_->allocate(); }

void Tensor::deallocate() { itensor_->deallocate(); }

bool Tensor::isAllocated() { return itensor_->isAllocated(); }

void Tensor::setValue(float value) { itensor_->setValue(value); }

void Tensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                      unsigned int w, float value) {
  itensor_->setValue(b, c, h, w, value);
}

void Tensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                      unsigned int w, float value, float beta) noexcept {
  itensor_->addValue(b, c, h, w, value, beta);
}

void Tensor::setZero() { itensor_->setZero(); }

void Tensor::setRandNormal(float mean, float stddev) {
  itensor_->setRandNormal(mean, stddev);
}

void Tensor::setRandUniform(float min, float max) {
  itensor_->setRandUniform(min, max);
}

void Tensor::setRandBernoulli(float probability) {
  itensor_->setRandBernoulli(probability);
}

void Tensor::initialize() { itensor_->initialize(); }

void Tensor::initialize(Initializer init) { itensor_->initialize(init); }

Tensor Tensor::apply(std::function<Tensor(Tensor)> f) const { return f(*this); }

Tensor &Tensor::apply(std::function<Tensor &(Tensor, Tensor &)> f,
                      Tensor &output) const {
  return f(*this, output);
}

int Tensor::multiply_i_strided(Tensor const &m, const float beta) {
  try {
    this->multiply_strided(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::multiply_strided(Tensor const &m, const float beta) const {
  Tensor t("", getFormat(), getDataType());
  return this->multiply_strided(m, t, beta);
}

Tensor &Tensor::multiply_strided(Tensor const &m, Tensor &output,
                                 const float beta) const {
  itensor_->multiply_strided(m, output, beta);
  return output;
}

int Tensor::multiply_i(float const &value) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  return itensor_->multiply_i(value);
}

Tensor Tensor::multiply(float const &value) const {
  Tensor t("", getFormat(), getDataType());
  return multiply(value, t);
}

Tensor &Tensor::multiply(float const &value, Tensor &out) const {
  itensor_->multiply(value, out);
  return out;
}

int Tensor::multiply_i(Tensor const &m, const float beta) {
  try {
    this->multiply(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::multiply(Tensor const &m, const float beta) const {
  Tensor t("", getFormat(), getDataType());
  return multiply(m, t, beta);
}

Tensor &Tensor::multiply(Tensor const &m, Tensor &output,
                         const float beta) const {
  NNTR_THROW_IF(m.getFormat() != this->getFormat(), std::invalid_argument)
    << "Tensor Format of " << getName() << ":"
    << ((bool)(this->getFormat()) ? "NHWC" : "NCHW") << " is not match. ("
    << ((bool)(m.getFormat()) ? "NHWC" : "NCHW") << ")";

  NNTR_THROW_IF(!getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  itensor_->multiply(m, output, beta);
  return output;
}

int Tensor::divide_i(float const &value) {
  if (value == 0.0f) {
    return ML_ERROR_INVALID_PARAMETER;
  }
  this->divide(value, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::divide(float const &value) const {
  Tensor output("", getFormat(), getDataType());
  return divide(value, output);
}

Tensor &Tensor::divide(float const &value, Tensor &output) const {
  /// @todo add unittest, ZeroDivisionError
  if (value == 0.0f) {
    std::stringstream ss;
    ss << "[Tensor] divide by value failed, value: " << value;
    throw std::invalid_argument(ss.str().c_str());
  }
  itensor_->divide(value, output);
  return output;
}

int Tensor::divide_i(Tensor const &m) {
  try {
    this->divide(m, *this);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::divide(Tensor const &m) const {
  Tensor output("", getFormat(), getDataType());
  return this->divide(m, output);
}

Tensor &Tensor::divide(Tensor const &m, Tensor &output) const {
  NNTR_THROW_IF(!getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot divide";
  itensor_->divide(m, output);
  return output;
}

int Tensor::add_i_strided(Tensor const &input, const float beta) {
  try {
    this->add_strided(input, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::add_strided(Tensor const &input, const float beta) const {
  Tensor output("", getFormat(), getDataType());
  return this->add_strided(input, output, beta);
}

Tensor &Tensor::add_strided(Tensor const &input, Tensor &output,
                            const float beta) const {
  CREATE_IF_EMPTY_DIMS(output, getDim(), nullptr);

  if (size() != input.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided addition does not support broadcasting");

  itensor_->add_strided(input, output, beta);

  return output;
}

int Tensor::add_i(float const &value) {
  this->add(value, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::add(float const &value) const {
  Tensor t("", getFormat(), getDataType());
  return add(value, t);
}

Tensor &Tensor::add(float const &value, Tensor &output) const {
  itensor_->add(value, output);
  return output;
}

int Tensor::add_i(Tensor const &m, float const alpha) {
  try {
    itensor_->add(m, *this, alpha);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }
  return ML_ERROR_NONE;
}

int Tensor::add_i_partial(unsigned int len, unsigned int addr_idx, Tensor &m,
                          unsigned int incX, unsigned int incY,
                          const Tensor alphas, unsigned int alpha_idx) {
  return itensor_->add_i_partial(len, addr_idx, m, incX, incY, alphas,
                                 alpha_idx);
}

Tensor Tensor::add(Tensor const &m, float const alpha) const {
  Tensor t("", getFormat(), getDataType());
  return this->add(m, t, alpha);
}

Tensor &Tensor::add(Tensor const &m, Tensor &output, float const alpha) const {
  NNTR_THROW_IF(m.getFormat() != this->getFormat(), std::invalid_argument)
    << "Tensor Format of " << getName() << ":"
    << ((bool)(this->getFormat()) ? "NHWC" : "NCHW") << " is not match. ("
    << ((bool)(m.getFormat()) ? "NHWC" : "NCHW") << ")";

  NNTR_THROW_IF(!itensor_->getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot add";
  itensor_->add(m, output, alpha);
  return output;
}

int Tensor::subtract_i(float const &value) {
  this->subtract(value, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::subtract(float const &value) const {
  Tensor output("", getFormat(), getDataType());
  return subtract(value, output);
}

Tensor &Tensor::subtract(float const &value, Tensor &output) const {
  itensor_->subtract(value, output);
  return output;
}

int Tensor::subtract_i(Tensor const &m) { return add_i(m, -1); }

Tensor Tensor::subtract(Tensor const &m) const {
  Tensor t("", getFormat(), getDataType());
  return this->subtract(m, t);
}

Tensor &Tensor::subtract(Tensor const &m, Tensor &output) const {
  return add(m, output, -1);
}

/**
 * This is to sum the Tensor data according to the dim.batch().
 * Therefore the result has M(dim.batch(), 1, 1, 1) dimension.
 */
Tensor Tensor::sum_by_batch() const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  Tensor output(batch(), 1, 1, 1, this->getFormat(), getDataType());
  itensor_->sum_by_batch(output);
  return output;
}

Tensor Tensor::sum(unsigned int axis, float alpha) const {
  Tensor output("", this->getFormat(), this->getDataType());
  return sum(axis, output, alpha, 0);
}

Tensor &Tensor::sum(unsigned int axis, Tensor &output, float alpha,
                    float beta) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  itensor_->sum(axis, output, alpha, beta);
  return output;
}

Tensor Tensor::sum(const std::vector<unsigned int> &axes, float alpha) const {
  Tensor output("", this->getFormat());
  return sum(axes, output, alpha);
}

Tensor &Tensor::sum(const std::vector<unsigned int> &axes, Tensor &output,
                    float alpha) const {
  if (axes.empty())
    throw std::invalid_argument("empty axes given");

  if (axes.size() == 1) {
    this->sum(axes[0], output, alpha);
  } else {

    /** club axes together */
    Tensor new_reshaped = Tensor(getDim());
    new_reshaped.copy(*this);
    std::vector<unsigned int> continuous_order = {0, 3, 1, 2};
    std::vector<unsigned int> new_axes = {axes[0]};

    for (unsigned int i = 1; i < axes.size(); ++i) {
      if (checkContinuous(axes[i - 1], axes[i])) {
        new_reshaped.mergeAxis(axes[i - 1], axes[i]);
        new_axes.back() = axes[i];
      } else {
        new_axes.push_back(axes[i]);
      }
    }

    Tensor ret = new_reshaped.sum(new_axes[0]);
    for (unsigned int i = 1; i < new_axes.size() - 1; ++i)
      ret = ret.sum(axes[i]);
    ret.sum(new_axes.back(), output, alpha);
  }
  return output;
}

Tensor &Tensor::abs(Tensor &output) const {
  if (size() != output.size() || getDataType() != output.getDataType() ||
      getFormat() != output.getFormat())
    throw std::invalid_argument(
      "Error: Tensor::abs requires output tensor to be same size, data type "
      "and format as input tensor.");
  return itensor_->abs(output);
}

Tensor Tensor::average(unsigned int axis) const {
  Tensor output("", this->getFormat(), this->getDataType());
  return average(axis, output);
}

Tensor &Tensor::average(unsigned int axis, Tensor &output) const {
  if (axis >= TensorDim::MAXDIM)
    throw std::out_of_range(
      "negative axis or axis more then MAXDIM is invalid");

  unsigned int axis_size = getDim()[axis];
  if (axis_size == 1)
    output.copy(*this);
  else
    this->sum(axis, output, 1.0 / ((float)axis_size));

  return output;
}

Tensor Tensor::average(const std::vector<unsigned int> &axes) const {
  Tensor output("", this->getFormat(), this->getDataType());
  return average(axes, output);
}

Tensor &Tensor::average(const std::vector<unsigned int> &axes,
                        Tensor &output) const {
  if (axes.empty())
    return this->average(output);

  TensorDim ret_shape(getTensorType());

  for (const auto &idx : axes) {
    if (idx >= TensorDim::MAXDIM) {
      throw std::out_of_range("axis more then MAXDIM is invalid");
    }
    ret_shape.setTensorDim(idx, getDim().getTensorDim(idx));
  }

  return this->sum(axes, output, 1.0 / (float)ret_shape.getDataLen());
}

Tensor Tensor::average() const {
  Tensor output = *this;
  unsigned int axis = 0;
  if (this->getFormat() == Tformat::NHWC) {
    output.reshape({1, getDim().getDataLen(), 1, 1, this->getTensorType()});
    axis = 1;
  } else {
    output.reshape({1, 1, 1, getDim().getDataLen(), this->getTensorType()});
    axis = 3;
  }
  return output.average(axis);
}

Tensor &Tensor::average(Tensor &output) const {
  Tensor result = *this;
  result.reshape({1, 1, 1, getDim().getDataLen()});
  return result.average(3, output);
}

int Tensor::pow_i(float exponent) {
  pow(exponent, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::pow(float exponent) const {
  Tensor output("", getFormat(), getDataType());
  return pow(exponent, output);
}

Tensor &Tensor::pow(float exponent, Tensor &output) const {
  itensor_->pow(exponent, output);
  return output;
}

int Tensor::sqrt_i() {
  this->sqrt(*this);
  return ML_ERROR_NONE;
}

Tensor Tensor::sqrt() const {
  Tensor output("", getFormat(), getDataType());
  return sqrt(output);
};

Tensor &Tensor::sqrt(Tensor &output) const {
  if (size() != output.size() || getDataType() != output.getDataType() ||
      getFormat() != output.getFormat())
    throw std::invalid_argument(
      "Error: Tensor::sqrt requires output tensor to be same size, data type "
      "and format as input tensor.");

  itensor_->sqrt(output);
  return output;
};

Tensor Tensor::neg() const {
  Tensor output("", getFormat(), getDataType());
  return neg(output);
};

Tensor &Tensor::neg(Tensor &output) const {
  if (size() != output.size() || getDataType() != output.getDataType() ||
      getFormat() != output.getFormat())
    throw std::invalid_argument(
      "Error: Tensor::sqrt requires output tensor to be same size, data type "
      "and format as input tensor.");

  itensor_->multiply(-1, output);
  return output;
};

int Tensor::erf_i() {
  erf(*this);
  return ML_ERROR_NONE;
}

Tensor Tensor::erf() const {
  Tensor output("", getFormat(), getDataType());
  return erf(output);
}

Tensor &Tensor::erf(Tensor &output) const {
  itensor_->erf(output);
  return output;
}

void Tensor::sin(Tensor &out, float alpha) const {
  if (size() != out.size())
    throw std::invalid_argument("Error: Size of out of Tensor::sin must match");

  itensor_->sin(out, alpha);
}

void Tensor::cos(Tensor &out, float alpha) const {
  if (size() != out.size())
    throw std::invalid_argument("Error: Size of out of Tensor::cos must match");

  itensor_->cos(out, alpha);
}

void Tensor::tan(Tensor &output, float alpha) const {
  if (size() != output.size() || getDataType() != output.getDataType() ||
      getFormat() != output.getFormat())
    throw std::invalid_argument(
      "Error: Tensor::abs requires output tensor to be same size, data type "
      "and format as input tensor.");

  itensor_->tan(output, alpha);
}

void Tensor::inv_sqrt_i() { itensor_->inv_sqrt(*this); }

Tensor Tensor::inv_sqrt(Tensor &out) const {
  itensor_->inv_sqrt(out);
  return out;
}

LazyTensor Tensor::chain() const { return LazyTensor(*this); }

float Tensor::l2norm() const { return itensor_->l2norm(); }

void Tensor::normalization_i() {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot do normalization.";

  const float min = minValue();
  const float max = maxValue();

  if (max == min) {
    Tensor tmp = *this;
    this->subtract_i(tmp);
  } else {
    this->subtract_i(min);
    this->divide_i(max - min);
  }
}

void Tensor::standardization_i() {
  Tensor mean_by_batch = this->sum_by_batch();
  mean_by_batch.divide_i(static_cast<float>(getDim().getFeatureLen()));

  this->subtract_i(mean_by_batch);
  Tensor std_dev_by_batch(batch(), 1, 1, 1, getFormat(), getDataType());
  std_dev_by_batch.setZero();

  /// @todo remove conditional statement
  if (getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *std_dev = std_dev_by_batch.getData<float>();

    for (unsigned int k = 0; k < batch(); ++k) {
      Tensor sub_this = this->getBatchSlice(k, 1);
      std_dev[k] = sub_this.l2norm();
    }
  } else if (getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *std_dev = std_dev_by_batch.getData<_FP16>();

    for (unsigned int k = 0; k < batch(); ++k) {
      Tensor sub_this = this->getBatchSlice(k, 1);
      std_dev[k] = static_cast<_FP16>(sub_this.l2norm());
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }

  std_dev_by_batch.divide_i(static_cast<float>(getDim().getFeatureLen()));
  this->divide_i(std_dev_by_batch);
}

void Tensor::dot(std::vector<Tensor *> input, std::vector<Tensor *> output,
                 bool trans, bool trans_in, float beta) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous. Cannot dot product.";

  itensor_->dot(input, output, trans, trans_in, beta);
}

Tensor Tensor::dot(Tensor const &input, bool trans, bool trans_in) const {
  Tensor output("", getFormat(), getDataType());
  dot(input, output, trans, trans_in);

  return output;
}

/**
 * @note: This dot product flattens the fist 3 axis for the purpose of
 * computation. So, while performing, these matrices are behaving as 2-D
 * matrices. The dimensions are restored while returning back the tensor
 * in case of trans is false.
 */
Tensor &Tensor::dot(Tensor const &input, Tensor &output, bool trans,
                    bool trans_in, float beta) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous. Cannot dot product.";

  itensor_->dot(input, output, trans, trans_in, beta);
  return output;
}

Tensor &Tensor::dot_deriv_wrt_1(Tensor const &m, Tensor const &output_deriv,
                                bool trans, bool trans_m, float beta) {
  bool deriv_trans_m = true;
  bool deriv_trans = false;
  /** @todo handle all cases of trans and trans_m */
  if (!trans && trans_m) {
    deriv_trans_m = false;
  }

  return output_deriv.dot(m, *this, deriv_trans, deriv_trans_m, beta);
}

/**
 * @brief compute the derivative wrt m in the m tensor
 * @note The caller tensor must be the same tensor as the one which called the
 * dot() product.
 */
Tensor &Tensor::dot_deriv_wrt_2(Tensor &m_deriv, Tensor const &output_deriv,
                                bool trans, bool trans_m, float beta) const {
  bool deriv_trans_m = false;
  bool deriv_trans = true;
  /** @todo handle all cases of trans and trans_m */

  if (!trans && trans_m) {
    output_deriv.dot(*this, m_deriv, deriv_trans, deriv_trans_m, beta);
    return m_deriv;
  } else {
    return dot(output_deriv, m_deriv, deriv_trans, deriv_trans_m, beta);
  }
}

Tensor &Tensor::dotBatched(Tensor const &m, Tensor &result, bool trans,
                           bool trans_m, float beta) const {
  if (!result.isAllocated())
    throw std::invalid_argument(
      "Output tensor must be preallocated for dotBatched operation");

  size_t lcm = std::lcm(batch(), m.batch());
  size_t group_size = lcm / batch();
  size_t m_group_size = lcm / m.batch();

  NNTR_THROW_IF(!((lcm == batch() || lcm == m.batch())), std::invalid_argument)
    << "The batch size of the given twon tensors must be the same"
       "or the bigger one should be a multiple of the smaller one";

  for (unsigned int b = 0; b < lcm; b++) {
    /** @todo try using transpose to speedup the operation */
    const Tensor this_b = this->getBatchSlice(b / group_size, 1);
    Tensor m_b = m.getBatchSlice(b / m_group_size, 1);
    Tensor result_b = result.getBatchSlice(b, 1);

    this_b.dot(m_b, result_b, trans, trans_m, beta);
  }

  return result;
}

Tensor &Tensor::dot_batched_deriv_wrt_1(Tensor const &m,
                                        Tensor const &output_deriv, bool trans,
                                        bool trans_m, float beta) {
  bool deriv_trans_m = true;
  bool deriv_trans = false;
  /** @todo handle all cases of trans and trans_m */
  if (!trans && trans_m) {
    deriv_trans_m = false;
  }

  return output_deriv.dotBatched(m, *this, deriv_trans, deriv_trans_m, beta);
}

Tensor &Tensor::dot_batched_deriv_wrt_2(Tensor &m_deriv,
                                        Tensor const &output_deriv, bool trans,
                                        bool trans_m, float beta) const {
  bool deriv_trans_m = false;
  bool deriv_trans = true;
  /** @todo handle all cases of trans and trans_m */

  if (!trans && trans_m) {
    output_deriv.dotBatched(*this, m_deriv, deriv_trans, deriv_trans_m, beta);
    return m_deriv;
  } else {
    return dotBatched(output_deriv, m_deriv, deriv_trans, deriv_trans_m, beta);
  }
}

Tensor Tensor::dropout_mask(float dropout) const {
  Tensor output(getDim());
  output.dropout_mask(dropout);
  return output;
}

void Tensor::dropout_mask(float dropout) {
  /// @todo add unittest
  NNTR_THROW_IF(dropout < 0 || dropout > 1, std::invalid_argument)
    << "[Tensor::dropout_mask] Dropout rate should be between 0 and 1";

  // if the rate is zero, no change is needed
  if (std::fpclassify(dropout) == FP_ZERO)
    return;

  setRandUniform(0.0, 1.0);
  itensor_->dropout_mask(dropout);
}

void Tensor::filter_mask(const Tensor &mask_len, bool reverse) {
  /// @todo add unittest
  itensor_->filter_mask(mask_len, reverse);
}

Tensor Tensor::zoneout_mask(float zoneout) {
  Tensor output(getDim());
  zoneout_mask(output, zoneout);
  return output;
}

void Tensor::zoneout_mask(Tensor &opposite, float zoneout) {
  NNTR_THROW_IF(getDim() != opposite.getDim(), std::invalid_argument)
    << "[Tensor::zoneout_mask] opposite dimension does not match";

  NNTR_THROW_IF(zoneout < 0 || zoneout > 1, std::invalid_argument)
    << "[Tensor::zoneout_mask] Zoneout rate should be between 0 and 1";

  // if the rate is zero, no change is needed
  if (std::fpclassify(zoneout) == FP_ZERO)
    return;

  itensor_->zoneout_mask(opposite, zoneout);
}

std::vector<Tensor> Tensor::split(unsigned num_size, int axis) {
  NNTR_THROW_IF(num_size == 0, std::invalid_argument)
    << "num size cannot be zero";

  if (axis == -1) {
    axis = 3;
  }

  NNTR_THROW_IF(!(0 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(getDim().getTensorDim(axis) % num_size != 0,
                std::invalid_argument)
    << "axis is not divisible by num_size, axis: " << axis
    << " num size: " << num_size;

  std::vector<size_t> sizes;
  sizes.resize(num_size);

  unsigned int sz = getDim().getTensorDim(axis) / num_size;
  std::fill(sizes.begin(), sizes.end(), sz);

  return split(sizes, axis);
}

std::vector<Tensor> Tensor::split(std::vector<size_t> sizes, int axis) {
  NNTR_THROW_IF(sizes.size() == 0, std::invalid_argument)
    << "num size cannot be zero";

  NNTR_THROW_IF(!(-1 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(
    std::any_of(sizes.begin(), sizes.end(), [](size_t sz) { return !sz; }),
    std::invalid_argument)
    << "among given sizes at least one of size is 0";

  return itensor_->split(sizes, axis);
}

Tensor Tensor::concat(const std::vector<Tensor> &tensors, int axis,
                      Tensor &output) {
  return itensor_->concat(tensors, axis, output);
}

Tensor Tensor::cat(const std::vector<Tensor> &tensors, int axis) {
  if (axis == -1) {
    axis = 3;
  }

  // Create an output tensor to store the concatenation result
  TensorDim out_dim = Tensor::calculateConcatOutputDim(tensors, axis);
  Tensor output = Tensor(out_dim);

  return output.concat(tensors, axis, output);
}

Tensor Tensor::cat(const std::vector<Tensor> &tensors, int axis,
                   Tensor &output) {
  if (axis == -1) {
    axis = 3;
  }

  // Check if the given output tensor dimension is valid
  TensorDim out_dim = Tensor::calculateConcatOutputDim(tensors, axis);

  NNTR_THROW_IF(out_dim != output.getDim(), std::invalid_argument)
    << "invalid output dim for concatenation " << output.getDim()
    << "expected output dim " << out_dim;

  return output.concat(tensors, axis, output);
}

void Tensor::print(std::ostream &out) const {
  printInstance(out, this);
  itensor_->print(out);
}

void Tensor::putData() const { itensor_->putData(); }

void Tensor::setData(const std::shared_ptr<MemoryData> buf, size_t off,
                     bool init) {
  itensor_->setMemoryData(buf, off);

  if (buf && init) {
    initialize();
  }
}

const std::shared_ptr<MemoryData> Tensor::getMemoryData() const {
  return itensor_->getMemoryData();
}

size_t Tensor::getOffset() const { return itensor_->getOffset(); }

void Tensor::copy(const Tensor &from) {
  /// @todo enable copy to non-contiguous tensor
  if (!itensor_->getContiguous() || !from.getContiguous()) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (from.size() != 0 && size() == from.size() &&
      scale_size() == from.scale_size() &&
      getDataType() == from.getDataType()) {
    // if tensor size and data type match, copy data
    itensor_->copy(from);
  } else {
    Tensor t = Tensor(from.getDim(), from.getData<char>());
    swap(t, *this);
  }
}

void Tensor::copyData(const Tensor &from) { itensor_->copyData(from); }

void Tensor::copy_with_stride(const Tensor &from) {
  if (itensor_->getDim() == from.getDim()) {
    // If the tensor dim matches, copy the data. This also applies to
    // uncontigous tensor.
    itensor_->copy_with_stride(from, *this);
  } else {
    // replace with a new tensor that has the same data as the given tensor
    Tensor t = Tensor(from.getDim(), true);
    itensor_->copy_with_stride(from, t);
    swap(t, *this);
  }
}

Tensor Tensor::getBatchSlice(size_t offset, unsigned int size) const {
  TensorDim dim_ = getDim();
  dim_.batch(size);

  return getSharedDataTensor(dim_, offset * this->getDim().getFeatureLen(),
                             true, "");
}

Tensor Tensor::getBatchSlice(const std::vector<unsigned int> &indices) const {

  // Validate tensor contiguity
  NNTR_THROW_IF(!this->getContiguous(), std::runtime_error)
    << "getBatchSlice requires contiguous tensor layer";

  // Validate indices vector is not empty
  NNTR_THROW_IF(indices.empty(), std::invalid_argument)
    << "Indices vector cannot be empty";

  // Validate indices
  const unsigned batch_size = getDim().batch();
  for (auto idx : indices) {
    NNTR_THROW_IF(idx >= batch_size, std::out_of_range)
      << "Batch index " << idx << " out of range [0," << batch_size << ")";
  }

  // Get original tensor dimensions
  const TensorDim &orig_dim = this->getDim();
  const size_t element_size = orig_dim.getDataTypeSize();

  // Calculate single batch size in elements
  const size_t single_batch_size = orig_dim.getFeatureLen();

  // Create output tensor with selected batches
  TensorDim new_dim = orig_dim;
  new_dim.batch(indices.size());
  Tensor output(new_dim);

  // Validate output tensor size
  const size_t output_bytes = output.bytes();
  const size_t single_batch_bytes = single_batch_size * element_size;

  // Get raw data pointers
  const unsigned char *src_data =
    static_cast<const unsigned char *>(this->getData<unsigned char>());
  unsigned char *dst_data =
    static_cast<unsigned char *>(output.getData<void>());

// Parallel copy using OpenMP
#pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    const unsigned batch_idx = indices[i];

    // Calculate memory offsets
    const size_t src_offset =
      static_cast<size_t>(batch_idx) * single_batch_bytes;
    const size_t dst_offset = static_cast<size_t>(i) * single_batch_bytes;

    // Bounds check for destination buffer
    NNTR_THROW_IF(dst_offset + single_batch_bytes > output_bytes,
                  std::runtime_error)
      << "Destination buffer overflow detected";

    // Perform memory copy
    std::memcpy(dst_data + dst_offset, src_data + src_offset,
                single_batch_bytes);
  }

  return output;
}

Tensor Tensor::clone() const {
  Tensor output(getName(), getFormat(), getDataType());
  output.copy(*this);
  return output;
}

Tensor Tensor::clone(ml::train::TensorDim::DataType type) const {
  if (getDataType() == type)
    return clone();
  TensorDim dim = getDim();
  dim.setDataType(type);
  Tensor output(dim, true);
  output.copyData(*this);
  output.setName(getName());
  return output;
}

void Tensor::readFSU() { itensor_->readFSU(); }

void Tensor::save(std::ostream &file) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot save.";

  itensor_->save(file);
}

void Tensor::read(std::ifstream &file, size_t start_offset,
                  bool read_from_offset, int file_fd) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot read.";

  // save the start_offset_info
  read_offset = start_offset;

  // Do not read now but save file_fd in tensor
  if (is_virtual) {
    fd = file_fd;
    return;
  }

  itensor_->read(file, start_offset, read_from_offset);
}

void Tensor::read(ReadSource src, size_t start_offset, bool read_from_offset) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot read.";

  itensor_->read(src, start_offset, read_from_offset);
}

std::vector<unsigned int> Tensor::argmax() const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot get argmax.";
  return itensor_->argmax();
}

std::vector<unsigned int> Tensor::argmin() const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot get argmin.";
  return itensor_->argmin();
}

std::pair<Tensor, Tensor> Tensor::topK(unsigned int k) const {

  // Create output tensor with modified W dimension
  TensorDim output_dim = getDim();
  TensorDim indices_dim = getDim();
  Tformat format = output_dim.getFormat();

  // Validate k is within width dimension size
  unsigned int width_size = output_dim.width();
  NNTR_THROW_IF(k == 0 || k > width_size, std::invalid_argument)
    << "k must be between 1 and width dimension size (" << width_size << ")";

  // Set new width dimension to k
  output_dim.width(k);
  indices_dim.width(k);
  indices_dim.setDataType(Tdatatype::UINT32); // Set indices data type to UINT32

  // Create output tensor
  Tensor output(output_dim);
  output.allocate();
  Tensor indices(indices_dim);
  indices.allocate();

  // Prepare output buffer
  void *output_data = output.getData<void>();
  uint32_t *indices_data = indices.getData<uint32_t>();

  // Call TopK implementation
  itensor_->topK(k, output_data, indices_data);

  return {output, indices};
}

float Tensor::max_abs() const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot get max_abs.";
  return itensor_->max_abs();
}

float Tensor::maxValue() const { return itensor_->maxValue(); }

float Tensor::minValue() const { return itensor_->minValue(); }

Tensor Tensor::transpose(const std::string &direction) const {
  Tensor output(getDim());
  transpose(direction, output);
  return output;
}

Tensor &Tensor::transpose(const std::string &direction, Tensor &output) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous. Cannot transpose.";

  if (output.getData<char>() == getData<char>()) {
    Tensor result = clone();
    return result.transpose(direction, output);
  }

  itensor_->transpose(direction, output);

  return output;
}

void Tensor::reshape(const TensorDim &d) { itensor_->reshape(d); }

void Tensor::fill(const Tensor &from, bool allocate) {
  if (allocate && this->empty()) {
    this->copy(from);
    return;
  }

  if (!from.getContiguous() || !getContiguous()) {
    /// @todo enable this if needed
    throw nntrainer::exception::not_supported(
      "[Tensor::fill] non-contiguous tensors are not supported");
  }

  if (getDim() != from.getDim()) {
    throw std::invalid_argument("[Tensor::fill] dimension must be the same");
  }

  if (getStrides() != from.getStrides()) {
    /// @todo length does not represent buffer size, there should be way to
    /// get the buffer size
    throw std::invalid_argument("[Tensor::fill] buffer size must be the same");
  }

  copyData(from);
}

TensorDim Tensor::getDim() const { return itensor_->getDim(); }

TensorDim::TensorType Tensor::getTensorType() const {
  return itensor_->getTensorType();
};

Initializer Tensor::getInitializer() const {
  return itensor_->getInitializer();
}

TensorDim::Format Tensor::getFormat() const { return itensor_->getFormat(); }

Tdatatype Tensor::getDataType() const { return itensor_->getDataType(); }

void Tensor::updateBatch(unsigned int batch) { itensor_->updateBatch(batch); }

void Tensor::updateDimension(TensorDim dimension) {
  itensor_->updateDimension(dimension);
}

const bool Tensor::getContiguous() const noexcept {
  return itensor_->getContiguous();
}

const std::array<size_t, TensorDim::MAXDIM>
Tensor::getStrides() const noexcept {
  return itensor_->getStrides();
}

bool Tensor::checkContinuous(unsigned int np1, unsigned int np2) const {
  if (np1 > 3 || np2 > 3) {
    throw std::invalid_argument(
      "Error: Input value must be within the range of 0 to 3.");
  }

  if (getFormat() == Tformat::NCHW) {
    if (np1 + 1 == np2)
      return true;
  } else {
    std::vector<unsigned int> continuous_order_nhwc = {0, 3, 1, 2};
    if (continuous_order_nhwc[np2] == continuous_order_nhwc[np1] + 1)
      return true;
  }

  return false;
}

void Tensor::setFileOffset(const size_t file_offset) {
  itensor_->setFileOffset(file_offset);
}

size_t Tensor::getFileOffset() const { return itensor_->getFileOffset(); }

void Tensor::setName(const std::string &name_) { itensor_->setName(name_); }

const std::string &Tensor::getName() const { return itensor_->getName(); }

size_t Tensor::getIndex(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w) const noexcept {
  return itensor_->getIndex(b, c, h, w);
}

size_t Tensor::size() const { return itensor_->size(); }

bool Tensor::empty() const { return itensor_->empty(); }

size_t Tensor::bytes() const { return itensor_->bytes(); }

size_t Tensor::getMemoryBytes() const { return itensor_->getMemoryBytes(); }

size_t Tensor::batch() const { return itensor_->batch(); }

size_t Tensor::channel() const { return itensor_->channel(); }

size_t Tensor::height() const { return itensor_->height(); }

size_t Tensor::width() const { return itensor_->width(); }

size_t Tensor::scale_size() const { return itensor_->scale_size(); }

QScheme Tensor::q_scheme() const { return itensor_->q_scheme(); }

void Tensor::mergeAxis(unsigned int axis1, unsigned int axis2) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot merge axis";

  if (axis2 != axis1 + 1)
    if (!checkContinuous(axis1, axis2))
      throw std::invalid_argument("axis2 must be axis1 + 1 for merging.");

  itensor_->mergeAxis(axis1, axis2);
}

void Tensor::createSharedDataTensor(const Tensor &src, Tensor &dest,
                                    size_t offset) const {
  itensor_->createSharedDataTensor(src.itensor_.get(), dest.itensor_.get(),
                                   offset);
}

Tensor Tensor::getSharedDataTensor(const TensorDim dim_, size_t offset,
                                   bool reset_stride,
                                   const std::string &name_) const {
  Tensor ret = *this;
  itensor_->getSharedDataTensor(dim_, offset, reset_stride, name_,
                                ret.itensor_.get());
  return ret;
}

void Tensor::activate() {

  NNTR_THROW_IF(!is_virtual, std::invalid_argument)
    << "non-virtual tensor cannot call activate()";
#if defined(_WIN32)
  NNTR_THROW_IF(true, std::invalid_argument)
    << "[Error/VirtualTensor] virtual tensor is not supported on Windows";
#else

  auto file_offset = getFileOffset();
  size_t off = (file_offset / 4096) * 4096;
  size_t diff = file_offset - off;
  size_t len = getMemoryBytes() + diff;

  mapped_ptr = mmap(NULL, len, PROT_READ, MAP_PRIVATE, this->fd, off);
#ifdef __ANDROID__
  madvise(mapped_ptr, len, MADV_WILLNEED);
#endif
  if (mapped_ptr == MAP_FAILED) {
    std::cerr << "[activate] mmap failed: " << strerror(errno) << std::endl;
  }
  itensor_->activate((void *)&((uint8_t *)mapped_ptr)[diff]);
#endif
}

void Tensor::deactivate() {

  NNTR_THROW_IF(!is_virtual, std::invalid_argument)
    << "non-virtual tensor cannot call deactivate()";
#if defined(_WIN32)
  NNTR_THROW_IF(true, std::invalid_argument)
    << "[Error/VirtualTensor] virtual tensor is not supported on Windows";
#else

  if (mapped_ptr == nullptr) {
    return;
  };

  auto file_offset = getFileOffset();
  size_t off = (file_offset / 4096) * 4096;
  size_t diff = file_offset - off;
  size_t len = getMemoryBytes() + diff;

  auto ret_munmap = munmap((void *)mapped_ptr, len);
  const size_t error_buflen = 100;
  char error_buf[error_buflen];
  NNTR_THROW_IF(ret_munmap == -1, std::runtime_error)
    << "[deactivate] munmap failed: "
    << SAFE_STRERROR(errno, error_buf, error_buflen);

  mapped_ptr = nullptr;
  itensor_->deactivate();
#endif
}

void Tensor::setTensorVar(TensorDim d, void *buf, size_t offset) {
  itensor_->setTensorVar(d, buf, offset);
}

TensorDim Tensor::calculateConcatOutputDim(const std::vector<Tensor> &tensors,
                                           int axis) {
  // Check axis, in which the tensors are concatenated, is valid.
  NNTR_THROW_IF(!(-1 <= axis && axis < 4), std::invalid_argument)
    << "cannot concatenate tensors along an axis: " << axis;

  // Check if the number of input tensors is valid.
  NNTR_THROW_IF(tensors.size() <= 1, std::invalid_argument)
    << "received an invalid tensor vector. size must be greater than 1.";

  auto out_dim = tensors.front().getDim();

  // Check if all tensor data types are the same.
  for (auto &t : tensors) {
    NNTR_THROW_IF(t.getDataType() != out_dim.getDataType(),
                  std::invalid_argument)
      << "cannot concatenate tensors with different data types.";
  }

  // Compute the dimensions of an output tensor.
  out_dim.setTensorDim(axis, 1);
  NNTR_THROW_IF(!std::all_of(tensors.begin(), tensors.end(),
                             [&out_dim, axis](const Tensor &t) {
                               auto cur_dim = t.getDim();
                               cur_dim.setTensorDim(axis, 1);
                               return out_dim == cur_dim;
                             }),
                std::invalid_argument)
    << " all tensor must have the same dimension except for the axis, out_dim: "
    << out_dim << " axis : " << axis;

  auto axis_dim = std::accumulate(tensors.begin(), tensors.end(), 0u,
                                  [axis](unsigned cur, const Tensor &t) {
                                    return cur += t.getDim().getTensorDim(axis);
                                  });

  out_dim.setTensorDim(axis, axis_dim);
  return out_dim;
}

std::ostream &operator<<(std::ostream &out, Tensor const &input) {
  input.print(out);
  return out;
}

} // namespace nntrainer
