// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_tensor.h
 * @date	22 July 2025
 * @brief	This is Q4_0_Tensor class for Q4_0 quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __Q4_0_TENSOR_H__
#define __Q4_0_TENSOR_H__
#ifdef __cplusplus

#include <quantizer.h>
#include <tensor_base.h>

namespace nntrainer {

#define QK4_0 32
/**
 * @brief Q4_0 Block
 * @note This is a structure for Q4_0 quantization.
 * This struct is not for use, only for reference.
 */
struct block_q4_0 {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
};

#define Q4_0_SIZE sizeof(struct block_q4_0)

/**
 * @class Q4_0_Tensor class
 * @brief Q4_0_Tensor class for Q4_0 quantized tensor
 */
class Q4_0_Tensor : public TensorBase {

public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  Q4_0_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new Q4_0_Tensor object
   *
   * @param d Tensor dim for this q4_k tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  Q4_0_Tensor(const TensorDim &d, bool alloc_now,
              Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new Q4_0_Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  Q4_0_Tensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new Q4_0_Tensor object
   * @param rhs TensorBase object to copy
   */
  Q4_0_Tensor(TensorBase &rhs) : TensorBase(rhs) {}

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
      "Q4_0_Tensor::getData() is not supported. Use getData() instead.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  void *getAddress(unsigned int i) override {
    throw std::invalid_argument("Q4_0_Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  const void *getAddress(unsigned int i) const override {
    throw std::invalid_argument("Q4_0_Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(float value) override {
    throw std::invalid_argument("Q4_0_Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override {
    throw std::invalid_argument("Q4_0_Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::addValue()
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override {
    throw std::invalid_argument("Q4_0_Tensor::addValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setZero()
   */
  void setZero() override;

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize(Initializer init) override {
    throw std::invalid_argument("Q4_0_Tensor::initialize() is not supported.");
  }

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize() override;

  /**
   * @copydoc Tensor::print()
   */
  void print(std::ostream &out) const override {
    throw std::invalid_argument("Q4_0_Tensor::print() is not supported.");
  }

  /**
   * @copydoc Tensor::copy()
   */
  void copy(const Tensor &from, ComputeOps *ops = nullptr) override {
    throw std::invalid_argument("Q4_0_Tensor::copy() is not supported.");
  }
  /**
   * @copydoc Tensor::copyData()
   */
  void copyData(const Tensor &from, ComputeOps *ops = nullptr) override {
    throw std::invalid_argument("Q4_0_Tensor::copyData() is not supported.");
  }

  /**
   * @copydoc Tensor::copy_with_stride()
   */
  void copy_with_stride(const Tensor &input, Tensor &output) override {
    throw std::invalid_argument(
      "Q4_0_Tensor::copy_with_stride() is not supported.");
  }

  /**
   * @copydoc Tensor::max_abs()
   */
  float max_abs(ComputeOps *ops = nullptr) const override {
    throw std::invalid_argument("Q4_0_Tensor::max_abs() is not supported.");
  }

  /**
   * @copydoc Tensor::maxValue()
   */
  float maxValue() const override {
    throw std::invalid_argument("Q4_0_Tensor::maxValue() is not supported.");
  }

  /**
   * @copydoc Tensor::minValue()
   */
  float minValue() const override {
    throw std::invalid_argument("Q4_0_Tensor::minValue() is not supported.");
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
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override;

private:
  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy_q40(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (Q6_K)
   */
  std::string getStringDataType() const override { return "Q4_0"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid(ComputeOps *ops = nullptr) const override { return true; }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __Q4_0_TENSOR_H__ */
