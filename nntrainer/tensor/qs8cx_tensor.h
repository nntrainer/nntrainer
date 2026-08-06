// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qs8cx_tensor.h
 * @date	30 July 2026
 * @brief	This is QS8CX_Tensor class for QS8CX quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jaemin Shin <jaemin980311@google.com>
 * @bug		No known bugs except for NYI items
 *
 * @note QS8CX refers to quantized symmetric signed 8-bit per-channel
 * quantization. The weight data is stored as N*K int8 values followed by N
 * per-channel float scales. As of now there is no 8-bit (qsi8cxp) pack/GEMM
 * backend kernel, so pack() / getPackedData() and the dot path throw a
 * not-yet-implemented error while the type is fully recognized elsewhere.
 */

#ifndef __QS8CX_TENSOR_H__
#define __QS8CX_TENSOR_H__
#ifdef __cplusplus

#include <quantizer.h>
#include <tensor_base.h>

namespace nntrainer {

/**
 * @class QS8CX_Tensor class
 * @brief QS8CX_Tensor class for QS8CX quantized tensor
 */
class QS8CX_Tensor : public TensorBase {

public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  QS8CX_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new QS8CX_Tensor object
   *
   * @param d Tensor dim for this qs8cx tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  QS8CX_Tensor(const TensorDim &d, bool alloc_now,
               Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new QS8CX_Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  QS8CX_Tensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new QS8CX_Tensor object
   * @param rhs TensorBase object to copy
   */
  QS8CX_Tensor(TensorBase &rhs) : TensorBase(rhs) {}

  /**
   * @copydoc Tensor::allocate()
   */
  void allocate() override;

  /**
   * @copydoc Tensor::deallocate()
   */
  void deallocate() override {
    data = nullptr;
    offset = 0;
  }

  /**
   * @copydoc Tensor::getData()
   */
  void *getData() const override;

  /**
   * @copydoc Tensor::getData()
   */
  void *getData(size_t idx) const override {
    throw std::invalid_argument(
      "QS8CX_Tensor::getData() is not supported. Use getData() instead.");
  }

  /**
   * @copydoc Tensor::getPackedData()
   */
  void *getPackedData() const override;

  /**
   * @copydoc Tensor::getAddress()
   */
  void *getAddress(unsigned int i) override {
    throw std::invalid_argument("QS8CX_Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  const void *getAddress(unsigned int i) const override {
    throw std::invalid_argument("QS8CX_Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(float value) override {
    throw std::invalid_argument("QS8CX_Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override {
    throw std::invalid_argument("QS8CX_Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::addValue()
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override {
    throw std::invalid_argument("QS8CX_Tensor::addValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setZero()
   */
  void setZero() override;

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize(Initializer init) override {
    throw std::invalid_argument("QS8CX_Tensor::initialize() is not supported.");
  }

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize() override;

  /**
   * @copydoc Tensor::print()
   */
  void print(std::ostream &out) const override;

  /**
   * @copydoc Tensor::copy()
   */
  void copy(const Tensor &from) override {
    throw std::invalid_argument("QS8CX_Tensor::copy() is not supported.");
  }

  /**
   * @copydoc Tensor::copyData()
   */
  void copyData(const Tensor &from) override {
    throw std::invalid_argument("QS8CX_Tensor::copyData() is not supported.");
  }

  /**
   * @copydoc Tensor::copy_with_stride()
   */
  void copy_with_stride(const Tensor &input, Tensor &output) override {
    throw std::invalid_argument(
      "QS8CX_Tensor::copy_with_stride() is not supported.");
  }

  /**
   * @copydoc Tensor::max_abs()
   */
  float max_abs() const override {
    throw std::invalid_argument("QS8CX_Tensor::max_abs() is not supported.");
  }

  /**
   * @copydoc Tensor::maxValue()
   */
  float maxValue() const override {
    throw std::invalid_argument("QS8CX_Tensor::maxValue() is not supported.");
  }

  /**
   * @copydoc Tensor::minValue()
   */
  float minValue() const override {
    throw std::invalid_argument("QS8CX_Tensor::minValue() is not supported.");
  }

  /**
   * @copydoc TensorBase::size()
   */
  size_t size() const override;

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override;

  /**
   * @copydoc Tensor::getScale()
   */
  void *getScale() const override;

  /**
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override;

  /**
   * @brief Eagerly pack the weight data after loading
   * @note Must be called after load_weight() to prepare for computation
   * @note Prepares weight data for efficient matrix multiplication
   */
  void pack() override;

private:
  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy_qs8cx(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (QS8CX)
   */
  std::string getStringDataType() const override { return "QS8CX"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid() const override { return true; }

  std::unique_ptr<uint8_t[]> packed_data = nullptr;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __QS8CX_TENSOR_H__ */
