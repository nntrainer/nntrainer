// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file fusion_realizer.cpp
 * @date 29 June 2026
 * @brief NNTrainer graph realizer which fuses an activation epilogue into its
 *        preceding compute layer (conv+relu / fc+act).
 * @see https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <fusion_realizer.h>

#include <cstdlib>
#include <string>

#include <common_properties.h>
#include <layer_node.h>
#include <util_func.h>

namespace nntrainer {

GraphRepresentation
FusionRealizer::realize(const GraphRepresentation &reference) {
  /// shallow-copied output graph (same nodes; activation property is moved
  /// in-place onto the fusible compute nodes).
  GraphRepresentation processed(reference.begin(), reference.end());

  /// opt-in by default; NNTR_FUSE_ACT=0 disables the fusion (falls back to the
  /// standalone ActivationLayer that ActivationRealizer would split out).
  const char *gate = std::getenv("NNTR_FUSE_ACT");
  const bool enabled = (gate == nullptr) || (gate[0] != '0');
  if (!enabled)
    return processed;

  for (auto &node : processed) {
    /// only compute layers that own an inline fused-activation epilogue
    if (!istrequal(node->getType(), "fully_connected") &&
        !istrequal(node->getType(), "conv2d"))
      continue;

    /// a node that already carries a fused epilogue is left alone, as the
    /// header states. Such a node normally has activation=none and the arm
    /// below would skip it anyway; this makes code and doc agree for the one
    /// case they otherwise would not -- a node carrying both properties, where
    /// overwriting the existing fused_activation would silently change the
    /// layer's math.
    const std::string fused = node->getProperty("fused_activation");
    if (!fused.empty() && !istrequal(fused, "none"))
      continue;

    const ActivationType act = node->getActivationType();
    /// pointwise epilogues only: softmax is row-wise (needs the full row), and
    /// none/unknown are nothing to fuse.
    if (act == ActivationType::ACT_NONE || act == ActivationType::ACT_SOFTMAX ||
        act == ActivationType::ACT_UNKNOWN)
      continue;

    /// move the realization activation -> the layer-internal fused_activation,
    /// and clear the realization one so ActivationRealizer leaves it fused.
    props::Activation act_name;
    act_name.set(act);
    node->setProperty(
      {"fused_activation=" + to_string(act_name), "activation=none"});
  }

  return processed;
}

} // namespace nntrainer
