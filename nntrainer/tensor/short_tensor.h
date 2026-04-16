// SPDX-License-Identifier: Apache-2.0
/**
 * @file	short_tensor.h
 * @date	10 January 2025
 * @brief	This is ShortTensor class for 16-bit signed integer calculation
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __SHORT_TENSOR_H__
#define __SHORT_TENSOR_H__
#ifdef __cplusplus

#include <tensor_base.h>

namespace nntrainer {

/**
 * @class ShortTensor class
 * @brief ShortTensor class for 16-bit unsigned integer calculation
 */
class ShortTensor : public TensorBase {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  ShortTensor(std::string name_ = "", Tformat fm = Tformat::NCHW,
              QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE);

  /**
   * @brief Construct a new ShortTensor object
   *
   * @param d Tensor dim for this float tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  ShortTensor(const TensorDim &d, bool alloc_now,
              Initializer init = Initializer::NONE, std::string name = "",
              QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE);

  /**
   * @brief Construct a new ShortTensor object
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  ShortTensor(const TensorDim &d, const void *buf = nullptr,
              QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE);

  /**
   * @brief Construct a new ShortTensor object
   * @param d data for the Tensor
   * @param fm format for the Tensor
   */
  ShortTensor(
    std::vector<std::vector<std::vector<std::vector<int16_t>>>> const &d,
    std::vector<float> const &scales, Tformat fm, QScheme qscheme_);

  /**
   * @brief Construct a new ShortTensor object
   * @param rhs TensorBase object to copy
   */
  ShortTensor(TensorBase &rhs) :
    TensorBase(rhs), qscheme(QScheme::PER_TENSOR_AFFINE) {}

  /**
   * @brief Basic Destructor
   */
  ~ShortTensor() {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator==(const ShortTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator!=(const ShortTensor &rhs) const { return !(*this == rhs); }

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
  const int16_t &getValue(unsigned int i) const;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  int16_t &getValue(unsigned int i);

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  const int16_t &getValue(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w) const;

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  int16_t &getValue(unsigned int b, unsigned int c, unsigned int h,
                    unsigned int w);

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
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override;

  /**
   * @copydoc Tensor::print(std::ostream &out)
   */
  void print(std::ostream &out) const override;

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
   * @copydoc Tensor::scale_size()
   */
  size_t scale_size() const override;

  /**
   * @copydoc Tensor::scale_size()
   */
  QScheme q_scheme() const override;

private:
  /**
   * @brief quantization scheme
   */
  QScheme qscheme;

  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (QINT16)
   */
  std::string getStringDataType() const override { return "QINT16"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid(ComputeOps *ops = nullptr) const override { return true; };
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SHORT_TENSOR_H__ */
