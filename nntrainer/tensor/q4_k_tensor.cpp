// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_k_tensor.cpp
 * @date	23 April 2025
 * @brief	This is Q4_K_Tensor class for Q4_K quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <compute_ops.h>
#include <q4_k_tensor.h>
#include <tensor.h>

namespace nntrainer {

Q4_K_Tensor::Q4_K_Tensor(std::string name_, Tformat fm, QScheme qscheme_) :
  Uint4QTensor(name_, fm, QScheme::Q4_Kx8) {}

Q4_K_Tensor::Q4_K_Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name, QScheme qscheme_) :
  Uint4QTensor(d, false, init, name, qscheme_) {
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1 ||
                  (d.height() % 256 != 0 && d.width() % 256 != 0),
                std::invalid_argument)
    << "Q4_K_Tensor must be 2 dimensional tensor with batch size 1 and "
       "height or width must be divisible by 256";

  if (qscheme_ == QScheme::Q4_Kx8) {
    NNTR_THROW_IF(d.height() % 8 != 0 || d.width() % 8 != 0,
                  std::invalid_argument)
      << "Q4_Kx8 Tensor must have height or width must be divisible by 8";
  }

  if (alloc_now)
    allocate();
}

Q4_K_Tensor::Q4_K_Tensor(const TensorDim &d, const void *buf,
                         QScheme qscheme_) :
  Q4_K_Tensor(d, true, Initializer::NONE, "", qscheme_) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy_q4k(buf);
  }
}

void Q4_K_Tensor::allocate() {
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

size_t Q4_K_Tensor::size() const {
  if (qscheme == QScheme::Q4_Kx8) {
    size_t num_blocks = height() * width() / (256 * 8);
    return Q4_Kx8_SIZE * num_blocks;
  } else {
    size_t num_blocks = height() * width() / 256;
    return Q4_K_SIZE * num_blocks;
  }
}

size_t Q4_K_Tensor::getMemoryBytes() const { return size() * sizeof(uint8_t); }

size_t Q4_K_Tensor::scale_size() const { return 0; }

void Q4_K_Tensor::copy_q4k(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  // copy tensor data
  getComputeOps()->scopy_u8(size(), (uint8_t *)buf, 1, (uint8_t *)getData(), 1);
}

} // namespace nntrainer
