// SPDX-License-Identifier: Apache-2.0
/**
 * @file	uint4_tensor.h
 * @date	20 March 2025
 * @brief	This is Uint4QTensor class for quantized 4-bit unsigned integer
 * calculation
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __UINT4_TENSOR_H__
#define __UINT4_TENSOR_H__
#ifdef __cplusplus

#include <quantizer.h>
#include <tensor_base.h>

namespace nntrainer {

/**
 * @class Uint4QTensor class
 * @brief Uint4QTensor class for quantized 4-bit integer calculation
 *
 * @note Uint4QTensor store uint4 data within the uint8 memory space.
 * Specifically, each uint8 value contains two uint4 values packed together.
 * The first four bits represent the first uint4 value, while the last four bits
 * represent the second uint4 value.
 * E.g., 01011001 (89) represents 0101 (+5) and 1001 (+9)
 */
class Uint4QTensor : public TensorBase {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  Uint4QTensor(std::string name_ = "", Tformat fm = Tformat::NCHW,
               QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE);

  /**
   * @brief Construct a new Uint4QTensor object
   *
   * @param d Tensor dim for this uint4 tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   * @param qscheme_ Quantization scheme of the tensor
   */
  Uint4QTensor(const TensorDim &d, bool alloc_now,
               Initializer init = Initializer::NONE, std::string name = "",
               QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE);

  /**
   * @brief Construct a new Uint4QTensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   * @param qscheme_ quantization scheme of the tensor
   */
  Uint4QTensor(const TensorDim &d, const void *buf = nullptr,
               QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE);

  /**
   * @brief Construct a new Uint4QTensor object
   *
   * @param d data for the Tensor
   * @param scales scale factors for the Tensor
   * @param fm format for the Tensor
   * @param qscheme_ quantization scheme of the tensor
   */
  Uint4QTensor(
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> const &d,
    std::vector<float> const &scales,
    std::vector<unsigned int> const &zero_points, Tformat fm, QScheme qscheme_);

  /**
   * @brief Construct a new Uint4QTensor object
   * @param rhs TensorBase object to copy
   */
  Uint4QTensor(TensorBase &rhs) :
    TensorBase(rhs), qscheme(QScheme::PER_TENSOR_AFFINE) {}

  /**
   * @brief Construct a new Uint4QTensor object
   * @param rhs TensorBase object to copy
   * @param qsceme_ qscheme_
   */
  Uint4QTensor(TensorBase &rhs, QScheme qscheme_) :
    TensorBase(rhs), qscheme(qscheme_) {}

  /**
   * @brief Basic Destructor
   */
  ~Uint4QTensor() {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator==(const Uint4QTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator!=(const Uint4QTensor &rhs) const { return !(*this == rhs); }

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
  const uint8_t getValue(unsigned int i) const;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  uint8_t getValue(unsigned int i);

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  const uint8_t getValue(unsigned int b, unsigned int c, unsigned int h,
                         unsigned int w) const;

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  uint8_t getValue(unsigned int b, unsigned int c, unsigned int h,
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
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override;

  /**
   * @copydoc Tensor::scale_size()
   */
  size_t scale_size() const override;

  /**
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override;

protected:
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
   * @return std::string of tensor data type (UINT4)
   */
  std::string getStringDataType() const override { return "UINT4"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid(ComputeOps *ops = nullptr) const override { return true; };
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __UINT4_TENSOR_H__ */
