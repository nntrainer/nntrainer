// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file fusion_realizer.h
 * @date 29 June 2026
 * @brief NNTrainer graph realizer which FUSES an activation epilogue into its
 *        preceding compute layer (conv+relu / fc+act), eliminating the separate
 *        ActivationLayer node.
 * @see https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __FUSION_REALIZER_H__
#define __FUSION_REALIZER_H__

#include <realizer.h>

namespace nntrainer {

/**
 * @brief Graph realizer that fuses a layer's activation into its forward, so
 * the compute layer (fully_connected / conv2d) applies the activation inline
 *        after GEMM+bias instead of having it split out into a standalone
 *        ActivationLayer node.
 *
 * @note  This is the INVERSE of ActivationRealizer (which splits `activation=X`
 *        into a node). FusionRealizer therefore runs BEFORE ActivationRealizer:
 *        for a fusible compute node it moves the realization `activation`
 *        property onto a layer-internal `fused_activation` property and clears
 *        the realization one, so ActivationRealizer then has nothing to split.
 *        The graph topology is UNCHANGED (no node added/removed/rewired), and
 * the activation math is the same ActiFunc the ActivationLayer would run, so
 *        the result is value-identical — just one fewer node + intermediate
 *        tensor per fused site.
 *
 * @note  Gated by NNTR_FUSE_ACT (default ON; set NNTR_FUSE_ACT=0 to disable),
 *        and applied to inference graphs only: the fused forward is
 *        value-identical to the standalone node, but there is no fused
 *        backward, so a training graph must keep the separate node.
 *        Skips softmax / none / unknown (softmax is row-wise, not a pointwise
 *        epilogue), and skips a node that already carries a fused activation.
 */
class FusionRealizer final : public GraphRealizer {
public:
  /** @brief Construct a new Fusion Realizer object */
  FusionRealizer() = default;

  /** @brief Destroy the Fusion Realizer object */
  ~FusionRealizer() = default;

  /**
   * @copydoc GraphRealizer::realize(const GraphRepresentation &reference)
   * @note moves a fusible compute node's activation onto fused_activation; the
   *       returned graph has the same nodes/edges as the reference.
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;
};

} // namespace nntrainer

#endif // __FUSION_REALIZER_H__
