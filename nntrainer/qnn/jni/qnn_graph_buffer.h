// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * @file   qnn_graph_buffer.h
 * @date   30 Sep 2026
 * @brief  Collects the tensor buffer addresses a QNN graph layer hands to QNN
 * @see    https://github.com/nntrainer/nntrainer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This header does not depend on the Qualcomm QNN SDK, so the logic can be
 * unit-tested in builds without enable-npu.
 */

#ifndef __QNN_GRAPH_BUFFER_H__
#define __QNN_GRAPH_BUFFER_H__

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <tensor.h>

namespace nntrainer {

/**
 * @brief Typed data pointer of a tensor handed to QNN
 */
using QnnBufferPtr =
  std::variant<std::monostate, uint8_t *, uint16_t *, float *>;

/**
 * @brief Get the data pointer of a tensor in the form QNN graph layers use
 * @param[in] t tensor
 * @return data pointer of @a t
 * @throw std::invalid_argument if the data type of @a t is not one of
 * UINT4, UINT8, UINT16 and FP32
 */
inline QnnBufferPtr getQnnBufferPtr(Tensor &t) {
  Tdatatype type = t.getDataType();
  switch (type) {
  case Tdatatype::UINT4:
  case Tdatatype::UINT8:
    return t.getData<uint8_t>();
  case Tdatatype::UINT16:
    return t.getData<uint16_t>();
  case Tdatatype::FP32:
    return t.getData<float>();
  default:
    throw std::invalid_argument("QNNGraph: unsupported data type " +
                                std::to_string(static_cast<int>(type)) +
                                " of tensor " + t.getName());
  }
}

/**
 * @brief Replace @a buffers with the data pointers of the given tensors
 * @details @a buffers is cleared first, so it never keeps addresses from a
 * previous call, which may belong to an activation pool that was re-allocated
 * since.
 * @param[out] buffers data pointers, one per tensor
 * @param[in] count number of tensors
 * @param[in] tensor_at callable taking an unsigned int index and returning the
 * tensor at that index as Tensor &
 * @throw std::invalid_argument if a tensor has an unsupported data type
 */
template <typename TensorAt>
void resetQnnBuffers(std::vector<QnnBufferPtr> &buffers, unsigned int count,
                     TensorAt tensor_at) {
  buffers.clear();
  for (unsigned int i = 0; i < count; ++i)
    buffers.push_back(getQnnBufferPtr(tensor_at(i)));
}

} // namespace nntrainer

#endif /* __QNN_GRAPH_BUFFER_H__ */
