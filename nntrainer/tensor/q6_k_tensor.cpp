// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_k_tensor.cpp
 * @date	23 April 2025
 * @brief	This is Q6_K_Tensor class for Q4_K quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <compute_ops.h>
#include <q6_k_tensor.h>
#include <tensor.h>

namespace nntrainer {

Q6_K_Tensor::Q6_K_Tensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm) {}

Q6_K_Tensor::Q6_K_Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name) :
  TensorBase(d, false, init, name) {
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1 || d.width() % 256 != 0,
                std::invalid_argument)
    << "Q6_K_Tensor must be 2 dimensional tensor with batch size 1 and "
       "width must be divisible by 256";

  if (alloc_now)
    allocate();
}

Q6_K_Tensor::Q6_K_Tensor(const TensorDim &d, const void *buf) :
  Q6_K_Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy_q6k(buf);
  }
}

void Q6_K_Tensor::allocate() {
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

void *Q6_K_Tensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint8_t>() + offset;
}

size_t Q6_K_Tensor::size() const {
  size_t num_blocks = height() * width() / 256;
  return Q6_K_SIZE * num_blocks;
}

size_t Q6_K_Tensor::getMemoryBytes() const { return size() * sizeof(uint8_t); }

void Q6_K_Tensor::copy_q6k(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  // copy tensor data
  getComputeOps()->scopy_u8(size(), (uint8_t *)buf, 1, (uint8_t *)getData(), 1);
}

void Q6_K_Tensor::setZero() {
  uint8_t *data = (uint8_t *)getData();
  std::fill(data, data + size(), 0);
}

void Q6_K_Tensor::initialize() {
  if (empty() || !isAllocated())
    return;

  setZero();
  putData();
}

QScheme Q6_K_Tensor::q_scheme() const { return QScheme::Q6_K; }

} // namespace nntrainer
