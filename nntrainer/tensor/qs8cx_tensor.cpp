// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qs8cx_tensor.cpp
 * @date	30 July 2026
 * @brief	This is QS8CX_Tensor class for QS8CX quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jaemin Shin <jaemin980311@google.com>
 * @bug		No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <kleidiai_interface.h>
#include <qs8cx_tensor.h>
#include <tensor.h>

namespace nntrainer {

QS8CX_Tensor::QS8CX_Tensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm) {
  offset = 0;
}

QS8CX_Tensor::QS8CX_Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                           std::string name) :
  TensorBase(d, false, init, name) {
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1, std::invalid_argument)
    << "QS8CX_Tensor must be 2 dimensional tensor with batch size 1";

  if (alloc_now)
    allocate();
  offset = 0;
}

QS8CX_Tensor::QS8CX_Tensor(const TensorDim &d, const void *buf) :
  QS8CX_Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy_qs8cx(buf);
  }
}

void QS8CX_Tensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    allocateSrcTensor();
  } else {
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

void *QS8CX_Tensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint8_t>() + offset;
}

void QS8CX_Tensor::pack() {
  if (packed_data) {
    return;
  }

  size_t opt_kernel_idx = 3;
  const size_t K = height();
  const size_t N = width();

  size_t packed_size =
    nntrainer::__kai_get_rhs_packed_size_qsi8cxp_qsi8cx(N, K, opt_kernel_idx);
  packed_data = std::make_unique<uint8_t[]>(packed_size);

  nntrainer::__kai_rhs_pack_qsi8cxp_qsi8cx(N, K, packed_data.get(), getData(),
                                           getScale(), opt_kernel_idx);

  if (!packed_data) {
    throw std::runtime_error{"something wrong"};
  }
}

void *QS8CX_Tensor::getPackedData() const {
  if (!packed_data) {
    throw std::runtime_error{"pack before run model"};
  }

  return packed_data.get();
}

size_t QS8CX_Tensor::size() const {
  const size_t K = height();
  const size_t N = width();
  return N * K + N * sizeof(float);
}

size_t QS8CX_Tensor::getMemoryBytes() const { return size() * sizeof(uint8_t); }

void *QS8CX_Tensor::getScale() const {
  if (!data)
    return nullptr;

  data->validate();

  const size_t K = height();
  const size_t N = width();

  return ((int8_t *)getData()) + N * K;
}

void QS8CX_Tensor::copy_qs8cx(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  scopy(size(), (uint8_t *)buf, 1, (uint8_t *)getData(), 1);
}

void QS8CX_Tensor::setZero() {
  uint8_t *data = (uint8_t *)getData();
  std::fill(data, data + size(), 0);
}

void QS8CX_Tensor::initialize() {
  if (empty() || !isAllocated())
    return;

  setZero();
  putData();
}

void QS8CX_Tensor::print(std::ostream &out) const {
  out << "data addr: " << getData() << '\n';
  out << dim;
  out << "[QS8CX data print skipped]" << std::endl;
}

QScheme QS8CX_Tensor::q_scheme() const { return QScheme::QS8CX; }

} // namespace nntrainer
