// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_k_tensor.h
 * @date	23 April 2025
 * @brief	This is Q4_K_Tensor class for Q4_K quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __Q4_K_TENSOR_H__
#define __Q4_K_TENSOR_H__
#ifdef __cplusplus

#include <quantizer.h>
#include <uint4_tensor.h>

namespace nntrainer {

/**
 * @brief Q4_K Block
 * @note This is a structure for Q4_K quantization.
 * This struct is not for use, only for reference.
 */
struct block_q4_K {
  int16_t d[1];       // super-block scale for quantized scales
  int16_t dmin[1];    // super-block scale for quantized mins
  uint8_t scales[12]; // scales and mins, quantized with 6 bits
  uint8_t qs[128];    // 4--bit quants
};

#define Q4_K_SIZE 144    // sizeof(block_q4_K)
#define Q4_Kx8_SIZE 1152 // Q4_K_SIZE * 8

/**
 * @class Q4_K_Tensor class
 * @brief Q4_K_Tensor class for Q4_K quantized tensor
 */
class Q4_K_Tensor : public Uint4QTensor {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  Q4_K_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW,
              QScheme qscheme_ = QScheme::Q4_Kx8);

  /**
   * @brief Construct a new Q4_K_Tensor object
   *
   * @param d Tensor dim for this q4_k tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  Q4_K_Tensor(const TensorDim &d, bool alloc_now,
              Initializer init = Initializer::NONE, std::string name = "",
              QScheme qscheme_ = QScheme::Q4_Kx8);

  /**
   * @brief Construct a new Q4_K_Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  Q4_K_Tensor(const TensorDim &d, const void *buf = nullptr,
              QScheme qscheme_ = QScheme::Q4_Kx8);

  /**
   * @brief Construct a new Q4_K_Tensor object
   * @param rhs TensorBase object to copy
   */
  Q4_K_Tensor(TensorBase &rhs) : Uint4QTensor(rhs, QScheme::Q4_Kx8) {}

  /**
   * @copydoc Tensor::allocate()
   */
  void allocate() override;

  /**
   * @copydoc TensorBase::size()
   */
  size_t size() const override;

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override;

  /**
   * @copydoc Tensor::scale_size()
   */
  size_t scale_size() const override;

private:
  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy_q4k(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (Q4_K)
   */
  std::string getStringDataType() const override { return "Q4_K"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid(ComputeOps *ops = nullptr) const override { return true; }

}; // class Q4_K_Tensor

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __Q4_K_TENSOR_H__ */
