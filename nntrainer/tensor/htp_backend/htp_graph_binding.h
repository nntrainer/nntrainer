// SPDX-License-Identifier: Apache-2.0
/** @file htp_graph_binding.h
 * @brief Typed weight roles derived from existing decode operation dataflow.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_HTP_GRAPH_BINDING_H
#define NNTR_HTP_GRAPH_BINDING_H
#include <htp_graph_desc.h>
#include <stdexcept>
#include <vector>
namespace nntrainer {
enum class HtpBindingRole {
  None,
  InputNorm,
  AttentionOutputNorm,
  MoeInputNorm,
  MoeOutputNorm,
  DenseInputNorm,
  DenseOutputNorm,
  FfnOutputNorm,
  OutputNorm,
  AttentionInput,
  AttentionOutput,
  ConvInput,
  ConvOutput
};
/** Roles are host metadata. No field or ordinal is added to the wire format. */
inline HtpBindingRole htpBindingRole(const std::vector<uint32_t> &words,
                                     unsigned index) {
  if (words.size() < HTP_GRAPH_HEADER_WORDS ||
      words[2] > HTP_GRAPH_MAX_LAYERS || words[3] > HTP_GRAPH_MAX_OPS ||
      index >= words[3] ||
      words.size() < htp_graph_words_for(words[2], words[3]))
    throw std::invalid_argument("HTP binding: invalid operation index");
  const auto *op = htp_graph_op_cat(words.data(), index);
  const unsigned previous =
    index && htp_graph_op_cat(words.data(), index - 1)->layer == op->layer
      ? htp_graph_op_cat(words.data(), index - 1)->kind
      : HTP_GRAPH_NO_OP;
  const unsigned next =
    index + 1 < words[3] &&
        htp_graph_op_cat(words.data(), index + 1)->layer == op->layer
      ? htp_graph_op_cat(words.data(), index + 1)->kind
      : HTP_GRAPH_NO_OP;
  using Role = HtpBindingRole;
  if (op->kind == HTP_OP_RMSNORM) {
    if (next == HTP_OP_LM_HEAD)
      return Role::OutputNorm;
    if (next == HTP_OP_FC)
      return Role::InputNorm;
    if (previous == HTP_OP_FC)
      return Role::AttentionOutputNorm;
    if (previous == HTP_OP_MOE)
      return Role::MoeOutputNorm;
    if (previous == HTP_OP_DENSE_FFN)
      return Role::DenseOutputNorm;
    if (next == HTP_OP_ROUTER_TOPK)
      return Role::MoeInputNorm;
    if (next == HTP_OP_DENSE_FFN)
      return Role::DenseInputNorm;
    if (previous == HTP_OP_ADD && next == HTP_OP_ADD)
      return Role::FfnOutputNorm;
  } else if (op->kind == HTP_OP_FC) {
    if (next == HTP_OP_QK_NORM)
      return Role::AttentionInput;
    if (next == HTP_OP_CONV1D_GATE)
      return Role::ConvInput;
    if (previous == HTP_OP_ATTN_M1)
      return Role::AttentionOutput;
    if (previous == HTP_OP_CONV1D_GATE)
      return Role::ConvOutput;
  } else {
    return Role::None;
  }
  throw std::invalid_argument("HTP binding: unsupported operation dataflow");
}
} // namespace nntrainer
#endif
