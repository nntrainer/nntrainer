// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q8_0_tensor.h
 * @date	14 May 2026
 * @brief	This is Q8_0_Tensor class for Q8_0 quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __Q8_0_TENSOR_H__
#define __Q8_0_TENSOR_H__
#ifdef __cplusplus

#include <quantizer.h>
#include <tensor_base.h>

namespace nntrainer {

#define QK8_0 32
/**
 * @brief Q8_0 Block (same layout as GGML's block_q8_0)
 * @note This struct is not for use, only for reference.
 */
struct block_q8_0 {
  uint16_t d;       // fp16 scale
  int8_t qs[QK8_0]; // quants
};

/// Fully qualified, so it also expands correctly outside nntrainer.
#define Q8_0_SIZE sizeof(::nntrainer::block_q8_0)

/**
 * @class Q8_0_Tensor class
 * @brief Q8_0_Tensor class for Q8_0 quantized tensor
 *
 * GgmlQuantizer stores the transposed weight as q8_0x4, four rows interleaved
 * block by block (see repack_q8_0), which is what the Q8_0 GEMM kernels read.
 */
class Q8_0_Tensor : public TensorBase {

public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  Q8_0_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new Q8_0_Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  Q8_0_Tensor(const TensorDim &d, bool alloc_now,
              Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new Q8_0_Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  Q8_0_Tensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new Q8_0_Tensor object
   * @param rhs TensorBase object to copy
   */
  Q8_0_Tensor(TensorBase &rhs) : TensorBase(rhs) {}

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
      "Q8_0_Tensor::getData() is not supported. Use getData() instead.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  void *getAddress(unsigned int i) override {
    throw std::invalid_argument("Q8_0_Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  const void *getAddress(unsigned int i) const override {
    throw std::invalid_argument("Q8_0_Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(float value) override {
    throw std::invalid_argument("Q8_0_Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override {
    throw std::invalid_argument("Q8_0_Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::addValue()
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override {
    throw std::invalid_argument("Q8_0_Tensor::addValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setZero()
   */
  void setZero() override;

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize(Initializer init) override {
    throw std::invalid_argument("Q8_0_Tensor::initialize() is not supported.");
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
    throw std::invalid_argument("Q8_0_Tensor::copy() is not supported.");
  }

  /**
   * @copydoc Tensor::copyData()
   */
  void copyData(const Tensor &from) override {
    throw std::invalid_argument("Q8_0_Tensor::copyData() is not supported.");
  }

  /**
   * @copydoc Tensor::copy_with_stride()
   */
  void copy_with_stride(const Tensor &input, Tensor &output) override {
    throw std::invalid_argument(
      "Q8_0_Tensor::copy_with_stride() is not supported.");
  }

  /**
   * @copydoc Tensor::max_abs()
   */
  float max_abs() const override {
    throw std::invalid_argument("Q8_0_Tensor::max_abs() is not supported.");
  }

  /**
   * @copydoc Tensor::maxValue()
   */
  float maxValue() const override {
    throw std::invalid_argument("Q8_0_Tensor::maxValue() is not supported.");
  }

  /**
   * @copydoc Tensor::minValue()
   */
  float minValue() const override {
    throw std::invalid_argument("Q8_0_Tensor::minValue() is not supported.");
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
  void copy_q80(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (Q8_0)
   */
  std::string getStringDataType() const override { return "Q8_0"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid() const override { return true; }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __Q8_0_TENSOR_H__ */
