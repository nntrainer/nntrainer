// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding.h
 * @date   04 March 2021
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __EMBEDDING_LAYER_H__
#define __EMBEDDING_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_impl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace causallm {

namespace props {

/**
 * @brief Path to a sidecar embedding LUT.
 */
class QuantizedLutPath final : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "quantized_lut_path";
  using prop_tag = nntrainer::str_prop_tag;
};

/**
 * @brief Output requantization scale for sidecar LUT decoding.
 */
class OutputQuantScale final : public nntrainer::Property<float> {
public:
  static constexpr const char *key = "output_quant_scale";
  using prop_tag = nntrainer::float_prop_tag;
};

/**
 * @brief Output requantization offset for sidecar LUT decoding.
 */
class OutputQuantOffset final : public nntrainer::Property<int> {
public:
  static constexpr const char *key = "output_quant_offset";
  using prop_tag = nntrainer::int_prop_tag;
};

} // namespace props

/**
 * @brief Shared sidecar embedding LUT loaded from raw UINT16 or JSON manifest.
 */
struct QuantLut {
  std::vector<uint8_t> bytes;
  std::vector<float> row_scales;

  float scale = 1.0f;
  int offset = 0;
  size_t in_dim = 0;
  size_t out_dim = 0;

  bool is_raw_u16 = false;
  bool is_signed4 = false;
};

/**
 * @brief Load or return a cached sidecar embedding LUT by path.
 */
WIN_EXPORT std::shared_ptr<QuantLut>
get_or_load_quant_lut(const std::string &path, size_t in_dim_hint = 0,
                      size_t out_dim_hint = 0);

/**
 * @brief Decode one LUT row to FP32.
 */
WIN_EXPORT void decode_quant_lut_row_to_fp32(const QuantLut &lut,
                                             size_t token_idx,
                                             float layer_scale, float *output,
                                             size_t output_len);

/**
 * @brief Decode one LUT row to UINT16 using naive float clamping.
 */
WIN_EXPORT void decode_quant_lut_row_to_uint16(const QuantLut &lut,
                                               size_t token_idx,
                                               float layer_scale,
                                               uint16_t *output,
                                               size_t output_len);

/**
 * @brief Decode one LUT row to UINT16 with output requantization.
 */
WIN_EXPORT void decode_quant_lut_row_to_uint16(
  const QuantLut &lut, size_t token_idx, float layer_scale,
  float output_quant_scale, int output_quant_offset, uint16_t *output,
  size_t output_len);

/**
 * @class   EmbeddingLayer
 * @brief   EmbeddingLayer
 * @todo    Support setBatch for EmbeddingLayer
 */
WIN_EXPORT class EmbeddingLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  WIN_EXPORT EmbeddingLayer();

  /**
   * @brief     Destructor of Embedding Layer
   */
  WIN_EXPORT ~EmbeddingLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] EmbeddingLayer &&
   */
  WIN_EXPORT EmbeddingLayer(EmbeddingLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs EmbeddingLayer to be moved.
   */
  WIN_EXPORT EmbeddingLayer &operator=(EmbeddingLayer &&rhs) = default;

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
    return EmbeddingLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::save()
   */
  WIN_EXPORT void save(
    std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    nntrainer::TensorDim::DataType dtype = nntrainer::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  inline static const std::string type = "embedding_layer";

private:
  void forwardSidecarLut(nntrainer::RunLayerContext &context,
                         unsigned int from, unsigned int to);

  std::tuple<nntrainer::props::InDim, nntrainer::props::OutDim,
             nntrainer::props::Scale, props::QuantizedLutPath,
             props::OutputQuantScale, props::OutputQuantOffset>
    embedding_props;
  unsigned int weight_idx;
  std::shared_ptr<QuantLut> quant_lut;
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __EMBEDDING_H__ */
