// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_tensor.cpp
 * @date	23 April 2025
 * @brief	This is Q4_0_Tensor class for Q4_0 quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <compute_ops.h>
#include <q4_0_tensor.h>
#include <tensor.h>

namespace nntrainer {

Q4_0_Tensor::Q4_0_Tensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm) {
  offset = 0;
}

Q4_0_Tensor::Q4_0_Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name) :
  TensorBase(d, false, init, name) {
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1 || d.width() % QK4_0 != 0,
                std::invalid_argument)
    << "Q4_0_Tensor must be 2 dimensional tensor with batch size 1 and "
       "width must be divisible by 32";

  if (alloc_now)
    allocate();
  offset = 0;
}

Q4_0_Tensor::Q4_0_Tensor(const TensorDim &d, const void *buf) :
  Q4_0_Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy_q40(buf);
  }
}

void Q4_0_Tensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    mem_data = new MemoryData((void *)(new uint8_t[size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<uint8_t>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void *Q4_0_Tensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint8_t>() + offset;
}

size_t Q4_0_Tensor::size() const {
  size_t num_blocks = height() * width() / QK4_0;
  return Q4_0_SIZE * num_blocks;
}

size_t Q4_0_Tensor::getMemoryBytes() const { return size() * sizeof(uint8_t); }

void Q4_0_Tensor::copy_q40(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  // copy tensor data
  getComputeOps()->scopy_u8(size(), (uint8_t *)buf, 1, (uint8_t *)getData(), 1);
}

void Q4_0_Tensor::setZero() {
  uint8_t *data = (uint8_t *)getData();
  std::fill(data, data + size(), 0);
}

void Q4_0_Tensor::initialize() {
  if (empty() || !isAllocated())
    return;

  setZero();
  putData();
}

QScheme Q4_0_Tensor::q_scheme() const { return QScheme::Q4_0; }

} // namespace nntrainer
