// SPDX-License-Identifier: Apache-2.0
/**
 * @file	uint_tensor.h
 * @date	02 April 2024
 * @brief	This is UIntTensor class for unsigned integer calculation
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __UINT_TENSOR_H__
#define __UINT_TENSOR_H__
#ifdef __cplusplus

#include <iomanip>
#include <iostream>
#include <cpu_backend.h>
#include <tensor.h>
#include <tensor_base.h>

namespace nntrainer {

/**
 * @class UIntTensor class : Template <typename T>
 * @brief UIntTensor class for T-bit unsigned integer calculation
 * For typename, uint8_t, uint16_t, uint32_t are supported.
 */
template <typename T> class UIntTensor : public TensorBase {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  UIntTensor(std::string name_ = "", Tformat fm = Tformat::NCHW,
             QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE);

  /**
   * @brief Construct a new UIntTensor object
   *
   * @param d Tensor dim for this float tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   * @param qscheme_ quantization scheme
   */
  UIntTensor(const TensorDim &d, bool alloc_now,
             Initializer init = Initializer::NONE, std::string name = "",
             QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE);

  /**
   * @brief Construct a new UIntTensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   * @param qscheme_ quantization scheme
   */
  UIntTensor(const TensorDim &d, const void *buf = nullptr,
             QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE);

  /**
   * @brief Construct a new UIntTensor object
   *
   * @param d data for the Tensor
   * @param scales scale factors for the Tensor
   * @param zero_points zero_points for the Tensor
   * @param fm format for the Tensor
   * @param qscheme_ quantization scheme
   */
  UIntTensor(std::vector<std::vector<std::vector<std::vector<T>>>> const &d,
             std::vector<float> const &scales,
             std::vector<unsigned int> const &zero_points, Tformat fm,
             QScheme qscheme_);

  /**
   * @brief Construct a new UIntTensor object
   * @param rhs TensorBase object to copy
   */
  UIntTensor(TensorBase &rhs) :
    TensorBase(rhs), qscheme(QScheme::PER_TENSOR_AFFINE) {}

  /**
   * @brief Basic Destructor
   */
  ~UIntTensor() {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator==(const UIntTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator!=(const UIntTensor &rhs) const { return !(*this == rhs); }

  /**
   * @copydoc Tensor::allocate()
   */
  void allocate() override;

  /**
   * @copydoc Tensor::deallocate()
   */
  void deallocate() override;

  /**
   * @copydoc Tensor::getData()
   */
  void *getData() const override;

  /**
   * @copydoc Tensor::getData(size_t idx)
   */
  void *getData(size_t idx) const override;

  /**
   * @copydoc Tensor::getScale()
   */
  void *getScale() const override;

  /**
   * @copydoc Tensor::getScale(size_t idx)
   */
  void *getScale(size_t idx) const override;

  /**
   * @copydoc Tensor::getZeroPoint()
   */
  unsigned int *getZeroPoint() const override;

  /**
   * @copydoc Tensor::getZeroPoint(size_t idx)
   */
  unsigned int *getZeroPoint(size_t idx) const override;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  void *getAddress(unsigned int i) override;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  const void *getAddress(unsigned int i) const override;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  const T &getValue(unsigned int i) const;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  T &getValue(unsigned int i);

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  const T &getValue(unsigned int b, unsigned int c, unsigned int h,
                    unsigned int w) const;

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  T &getValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w);

  /**
   * @copydoc Tensor::setValue(float value)
   */
  void setValue(float value) override;

  /**
   * @copydoc Tensor::setValue(b, c, h, w, value)
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override;

  /**
   * @copydoc Tensor::addValue(b, c, h, w, value, beta)
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override;

  /**
   * @copydoc Tensor::setZero()
   */
  void setZero() override;

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize() override;

  /**
   * @copydoc Tensor::initialize(Initializer init)
   */
  void initialize(Initializer init) override;

  /**
   * @copydoc Tensor::copy(const Tensor &from)
   */
  void copy(const Tensor &from, ComputeOps *ops = nullptr) override;

  /**
   * @copydoc Tensor::copyData(const Tensor &from)
   */
  void copyData(const Tensor &from, ComputeOps *ops = nullptr) override;

  /**
   * @copydoc Tensor::copy_with_stride()
   */
  void copy_with_stride(const Tensor &input, Tensor &output) override;

  /**
   * @copydoc Tensor::save(std::ostream &file)
   */
  void save(std::ostream &file) override;

  /**
   * @copydoc Tensor::read(std::ifstream &file)
   */
  void read(std::ifstream &file, size_t start_offset,
            bool read_from_offset) override;

  /**
   * @copydoc Tensor::read(ReadSource src)
   */
  void read(ReadSource src, size_t start_offset,
            bool read_from_offset) override;

  /**
   * @copydoc Tensor::argmax()
   */
  std::vector<unsigned int> argmax() const override;

  /**
   * @copydoc Tensor::argmin()
   */
  std::vector<unsigned int> argmin() const override;

  /**
   * @copydoc Tensor::max_abs()
   */
  float max_abs(ComputeOps *ops = nullptr) const override;

  /**
   * @copydoc Tensor::maxValue()
   */
  float maxValue() const override;

  /**
   * @copydoc Tensor::minValue()
   */
  float minValue() const override;

  /**
   * @copydoc Tensor::print(std::ostream &out)
   */
  void print(std::ostream &out) const override;

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override;

  /**
   * @copydoc TensorBase::save_quantization_info()
   */
  void save_quantization_info(std::ostream &file) override;

  /**
   * @copydoc TensorBase::read_quantization_info()
   */
  void read_quantization_info(std::ifstream &file, size_t start_offset,
                              bool read_from_offset) override;

  /**
   * @copydoc TensorBase::read_quantization_info()
   */
  void read_quantization_info(ReadSource src, size_t start_offset,
                              bool read_from_offset) override;

  /**
   * @copydoc Tensor::scale_size()
   */
  size_t scale_size() const override;

  /**
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override;

private:
  /**
   * @brief quantization scheme
   */
  QScheme qscheme = QScheme::PER_TENSOR_AFFINE;

  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (UINT16)
   */
  std::string getStringDataType() const override {
    if (typeid(T) == typeid(uint8_t))
      return "UINT8";
    else if (typeid(T) == typeid(uint16_t))
      return "UINT16";
    else if (typeid(T) == typeid(uint32_t))
      return "UINT32";
    else
      throw std::runtime_error("unsupported type");
  }

  /**
   * @brief  check Tensor Data Type (to briefly support template)
   * @return Tdatatype
   */
  Tdatatype checkTensorDataType() const {
    if (typeid(T) == typeid(uint8_t))
      return Tdatatype::UINT8;
    else if (typeid(T) == typeid(uint16_t))
      return Tdatatype::UINT16;
    else if (typeid(T) == typeid(uint32_t))
      return Tdatatype::UINT32;
    else
      throw std::runtime_error("unsupported type");
  }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid(ComputeOps *ops = nullptr) const override { return true; }; // NYI
};

/******  Alias for UIntTensors ******/
using UInt8Tensor = UIntTensor<uint8_t>;
using UInt16Tensor = UIntTensor<uint16_t>;
using UInt32Tensor = UIntTensor<uint32_t>;

/****  Declare Template classes *****/
template class UIntTensor<uint8_t>;
template class UIntTensor<uint16_t>;
template class UIntTensor<uint32_t>;

/**
 * Define UIntTenosr's template class methods.
 * Template methods should be defined with declaration.
 * To this end, include implementation file
 */
#include <uint_tensor.cpp>

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __UINT_TENSOR_H__ */
