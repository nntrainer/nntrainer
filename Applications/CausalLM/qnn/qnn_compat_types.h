// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qnn_compat_types.h
 * @brief  QuickAI-owned compatibility type aliases for the main-based
 * nntrainer.
 *
 * @note   The old QuickAI nntrainer fork used to expose two
 *         types that QuickAI's QNN product models depend on but that the
 *         upstream `main` nntrainer does not provide:
 *
 *           - ml::train::TensorDim::IO_TensorType : a variant over the various
 *             dtype pointers a QNN graph binds for its quantized inference I/O.
 *           - causallm::multimodal_pointer        : a {void*, size_t} pair used
 *             by the (currently excluded) multimodal QNN models.
 *
 *         To keep the QuickAI runtime logic untouched while compiling against
 *         the main nntrainer API, we re-declare these here, owned by QuickAI,
 *         with the *exact* definitions copied from the OLD nntrainer. Usage
 *         sites that referenced `ml::train::TensorDim::IO_TensorType` are
 *         retargeted to `causallm::IO_TensorType`.
 *
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __QNN_COMPAT_TYPES_H__
#define __QNN_COMPAT_TYPES_H__

#include <cstdint>
#include <utility>
#include <variant>

namespace causallm {

/**
 * @brief Variant over the dtype pointers a QNN graph binds for inference I/O.
 *
 * Derived from the OLD nntrainer `tensor_dim.h` IO_TensorType. The OLD type
 * additionally carried an `_FP16*` alternative, but `_FP16` is an nntrainer
 * internal typedef not exposed by the main nntrainer public headers, and none
 * of the QuickAI QNN models bind/`std::get` an `_FP16*` slot (FP16 QNN I/O is
 * carried as `uint16_t*`, see Quick_Dot_AI_QNN::get_qnn_input_data). So the
 * half-precision alternative is intentionally omitted to avoid depending on an
 * nntrainer-private type.
 */
using IO_TensorType =
  std::variant<float *, uint32_t *, uint16_t *, uint8_t *, int16_t *, int8_t *>;

} // namespace causallm

#endif /* __QNN_COMPAT_TYPES_H__ */
