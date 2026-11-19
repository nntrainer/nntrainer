// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q8_0_tensor.cpp
 * @date	14 May 2026
 * @brief	This is Q8_0_Tensor class for Q8_0 quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <q8_0_tensor.h>
#include <tensor.h>

namespace nntrainer {

Q8_0_Tensor::Q8_0_Tensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm) {
  offset = 0;
}

Q8_0_Tensor::Q8_0_Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name) :
  TensorBase(d, false, init, name) {
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1 || d.width() % QK8_0 != 0,
                std::invalid_argument)
    << "Q8_0_Tensor must be 2 dimensional tensor with batch size 1 and "
       "width must be divisible by 32";

  if (alloc_now)
    allocate();
  offset = 0;
}

Q8_0_Tensor::Q8_0_Tensor(const TensorDim &d, const void *buf) :
  Q8_0_Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy_q80(buf);
  }
}

void Q8_0_Tensor::allocate() {
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

void *Q8_0_Tensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  // offset counts elements; each QK8_0 elements take Q8_0_SIZE bytes.
  NNTR_THROW_IF(offset % QK8_0 != 0, std::invalid_argument)
    << getName() << ": Q8_0 offset " << offset
    << " is not a multiple of the block size";
  return data->getAddr<uint8_t>() + (offset / QK8_0) * Q8_0_SIZE;
}

size_t Q8_0_Tensor::size() const {
  size_t num_blocks = height() * width() / QK8_0;
  return Q8_0_SIZE * num_blocks;
}

size_t Q8_0_Tensor::getMemoryBytes() const { return size() * sizeof(uint8_t); }

void Q8_0_Tensor::copy_q80(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  // copy tensor data
  scopy(size(), (uint8_t *)buf, 1, (uint8_t *)getData(), 1);
}

void Q8_0_Tensor::setZero() {
  uint8_t *data = (uint8_t *)getData();
  std::fill(data, data + size(), 0);
}

void Q8_0_Tensor::initialize() {
  if (empty() || !isAllocated())
    return;

  setZero();
  putData();
}

void Q8_0_Tensor::print(std::ostream &out) const {
  out << "data addr: " << getData() << '\n';
  out << dim;
  out << "[Q8_0 data print skipped]" << std::endl;
}

QScheme Q8_0_Tensor::q_scheme() const { return QScheme::Q8_0; }

} // namespace nntrainer
