// SPDX-License-Identifier: Apache-2.0
/**
 * @file	bcq_tensor.h
 * @date	06 December 2024
 * @brief	This is BCQTensor class for binary-code-based quantization
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __BCQ_TENSOR_H__
#define __BCQ_TENSOR_H__
#ifdef __cplusplus

#include <tensor_base.h>

#include "BiQGEMM.h"

namespace nntrainer {

/**
 * @class BCQTensor class
 * @brief BCQTensor class for the binary-code-based quantized weight (BCQ)
 */
class BCQTensor : public TensorBase {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  BCQTensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new BCQTensor object
   *
   * @param d Tensor dim for this bcq tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  BCQTensor(const TensorDim &d, bool alloc_now,
            Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new BCQTensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  BCQTensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new BCQTensor object
   * @param rhs TensorBase object to copy
   */
  BCQTensor(TensorBase &rhs) : TensorBase(rhs) {}

  /**
   * @brief Basic Destructor
   */
  ~BCQTensor() {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator==(const BCQTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator!=(const BCQTensor &rhs) const;

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
  const uint32_t &getValue(unsigned int i) const;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  uint32_t &getValue(unsigned int i);

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  const uint32_t &getValue(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w) const;

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  uint32_t &getValue(unsigned int b, unsigned int c, unsigned int h,
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
   * @copydoc Tensor::dot(Tensor const &input, Tensor &output, bool
   * trans, bool trans_in, float beta)
   *
   * @note BCQTensor::dot ignores trans, trans_in, and beta currently
   */
  Tensor &dot(Tensor const &input, Tensor &output, bool trans, bool trans_in,
              float beta, ComputeOps *ops = nullptr) const override;

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
   * @copydoc Tensor::readFSU()
   */
  void readFSU() override;

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
   * @copydoc TensorBase::save_quantization_info(std::ostream &file)
   */
  void save_quantization_info(std::ostream &file) override;

  /**
   * @copydoc TensorBase::read_quantization_info(std::ifstream &file)
   */
  void read_quantization_info(std::ifstream &file, size_t start_offset,
                              bool read_from_offset) override;

  /**
   * @copydoc TensorBase::size()
   */
  size_t size() const override;

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
   * @copydoc Tensor::scale_size()
   */
  size_t scale_size() const override;

private:
  /// @note this is an arbitrary value
  uint16_t quantized_bit_size_ = 3;
  std::unique_ptr<BiQGEMM::BCQW> bcq_weight_;

  /**
   * @brief create BCQW structure from current tensor
   */
  void createBCQW();

  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type
   */
  std::string getStringDataType() const override;

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid(ComputeOps *ops = nullptr) const override { return true; };

  /**
   * @brief print quantization scale factors
   */
  void printScales(std::ostream &out) const;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __BCQ_TENSOR_H__ */
