// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lm_head.h
 * @date   16 Jan 2026
 * @brief  This is LM_Head Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Anirudh Bocha <b.saianirud@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LM_HEAD_H__
#define __LM_HEAD_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_devel.h>
#include <layer_impl.h>
#include <vector>

namespace causallm {

/**
 * @brief Per-batch-index row (within the current input buffer) that
 *        forwarding() should project to vocab logits. An empty vector
 *        (default), or a batch index beyond its size, means "use the last
 *        row" (height - 1), matching incremental_forwarding's inference
 *        behavior.
 *
 * @details Under right-padded training the last *real* token is not at row
 *          (height - 1), so this must be set to that token's row before the
 *          forward pass -- and since different samples in the same batch pad
 *          to different lengths, that row is inherently per batch index, not
 *          one shared value. It must be written from the training thread
 *          during the forward pass itself — NOT from the dataset generator,
 *          which nntrainer runs on a separate producer thread that may
 *          prefetch ahead of the trainer.
 *
 * @note This standalone lm_head layer is only used by models that do NOT tie
 *       word embeddings; the tied path uses TieWordEmbedding, whose
 *       embedding-mode instance derives the row itself (see
 *       g_tie_embedding_lm_head_read_row). An untied model trained with
 *       right-padding needs an equivalent hook in its embedding layer --
 *       today nothing in this codebase populates this vector outside of
 *       unit tests, so untied-embedding training still needs that hook
 *       written before it can train correctly under right-padding.
 */
extern std::vector<unsigned int> g_lm_head_read_row;

/**
 * @class   LMHead layer
 * @brief   LMHead layer
 */
WIN_EXPORT class LmHeadLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  WIN_EXPORT LmHeadLayer();

  /**
   * @brief     Destructor of Embedding Layer
   */
  WIN_EXPORT ~LmHeadLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] LmHeadLayer &&
   */
  WIN_EXPORT LmHeadLayer(LmHeadLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs LmHeadLayer to be moved.
   */
  WIN_EXPORT LmHeadLayer &operator=(LmHeadLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return LmHeadLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   * @note Both calcDerivative and calcGradient are implemented. The LM
   *       head weight stays frozen under LoRA-only training (see
   *       causal_lm.cpp), in which case the framework never calls
   *       calcGradient; it is implemented so full fine-tuning works.
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; }

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "lm_head";

private:
  std::tuple<nntrainer::props::Unit> lmhead_props;
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
  bool skip_prefill = false;
};
} // namespace causallm

#endif
#endif
