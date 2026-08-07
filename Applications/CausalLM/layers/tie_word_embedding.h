// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   custom_tie_word_embedding_layer.h
 * @date   21 May 2025
 * @brief  This is Tie_Word_Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Anirudh Bocha <b.saianirud@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CUSTOM_TIE_WORD_EMBEDDING_H__
#define __CUSTOM_TIE_WORD_EMBEDDING_H__
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
 *        lm_head-mode forwarding() should project to vocab logits. An empty
 *        vector (default), or a batch index beyond its size, means "use the
 *        last row" (height - 1), matching incremental_forwarding_lmhead's
 *        inference behavior.
 *
 * @details Under right-padded training the last *real* token is not at
 *          row (height - 1), so the lm head must be told which row to read
 *          -- and since different samples in the same batch pad to
 *          different lengths, that row is inherently per batch index, not
 *          one shared value. The vector is produced by the *embedding*-mode
 *          instance of this same layer (see forwarding()), which is the
 *          first node in the graph and is the only place the raw token ids
 *          are visible: it scans every batch row for its own last non-pad
 *          id and records read_row[b] here, and the lm_head-mode instance
 *          reads it back later in the very same forward pass on the same
 *          thread. It is deliberately NOT thread_local and deliberately NOT
 *          set by the data pipeline: nntrainer runs the dataset generator on
 *          a separate producer thread which may also prefetch ahead of the
 *          trainer, so a value written there is both invisible to the
 *          training thread and liable to describe the wrong sample.
 *
 *          This is a separate variable from lm_head.h's
 *          `g_lm_head_read_row` to avoid an inter-shared-library link
 *          dependency between tie_word_embedding_layer.so and lm_head.so
 *          (a model only ever loads one of the two lm-head layer types,
 *          depending on whether it ties embeddings).
 */
extern std::vector<unsigned int> g_tie_embedding_lm_head_read_row;

/**
 * @class   TieWordEmbedding
 * @brief   TieWordEmbedding
 * @todo    Support setBatch for TieWordEmbedding
 */
WIN_EXPORT class TieWordEmbedding : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  WIN_EXPORT TieWordEmbedding();

  /**
   * @brief     Destructor of Embedding Layer
   */
  WIN_EXPORT ~TieWordEmbedding() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] TieWordEmbedding &&
   */
  WIN_EXPORT TieWordEmbedding(TieWordEmbedding &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs TieWordEmbedding to be moved.
   */
  WIN_EXPORT TieWordEmbedding &operator=(TieWordEmbedding &&rhs) = default;

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
    return TieWordEmbedding::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   * @note calcDerivative is implemented for lm_head mode only (FP32);
   *       embedding mode keeps throwing, since there is no meaningful
   *       gradient with respect to token indices and that instance is the
   *       first layer in the graph, so it is never invoked. calcGradient is
   *       implemented for BOTH modes and accumulates into the one shared
   *       tied weight; the framework only calls it when the layer is
   *       trainable (frozen under LoRA-only training).
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; }

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  /**
   * @copydoc Layer::read()
   */
  WIN_EXPORT void read(std::ifstream &file, nntrainer::RunLayerContext &context,
                       bool opt_var, ml::train::ExecutionMode mode,
                       bool trainable,
                       nntrainer::TensorDim::DataType definedWeightDataType,
                       bool fsu = false, size_t start_offset = 0,
                       bool read_from_offset = false,
                       int file_fd = -1) override;

  /**
   * @copydoc Layer::read() (ReadSource/mmap variant)
   */
  WIN_EXPORT void read(nntrainer::ReadSource src,
                       nntrainer::RunLayerContext &context, bool opt_var,
                       ml::train::ExecutionMode mode, bool trainable,
                       nntrainer::TensorDim::DataType definedWeightDataType,
                       bool fsu, size_t start_offset = 0,
                       bool read_from_offset = false,
                       int file_fd = -1) override;

  /**
   * @copydoc Layer::save()
   */
  WIN_EXPORT void save(
    std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    nntrainer::TensorDim::DataType dtype = nntrainer::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "tie_word_embeddings";

private:
  std::tuple<nntrainer::props::InDim, nntrainer::props::OutDim,
             nntrainer::props::Unit, nntrainer::props::Scale>
    tieword_embedding_props;
  enum mode { embedding, lm_head };
  enum mode mode_;
  std::array<unsigned int, 4> weight_idx; /**< indices of the weights */
  bool skip_prefill = false;

  WIN_EXPORT void finalize_embedding(nntrainer::InitLayerContext &context);
  WIN_EXPORT void finalize_lmhead(nntrainer::InitLayerContext &context);
  WIN_EXPORT void
  incremental_forwarding_embedding(nntrainer::RunLayerContext &context,
                                   unsigned int from, unsigned int to,
                                   bool training);
  WIN_EXPORT void
  incremental_forwarding_lmhead(nntrainer::RunLayerContext &context,
                                unsigned int from, unsigned int to,
                                bool training);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __CUSTOM_TIE_WORD_EMBEDDING_H__ */
