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
#include <tensor_dim.h>

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

/**
 * @brief Where save() writes this layer's weight instead of the model file
 *        (sidecar extraction; used by nntr_quantize --ple_sidecar).
 */
class SidecarExportPath final : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "sidecar_export_path";
  using prop_tag = nntrainer::str_prop_tag;
};

} // namespace props

/**
 * @brief Shared sidecar embedding LUT loaded from raw UINT16, JSON manifest,
 *        or GGML (q4_0/q6_k) row payload.
 *
 * The payload is mmap'd read-only when possible (POSIX) so a multi-hundred-MB
 * table stays out of resident memory and rows are paged in on demand; `bytes`
 * is the fallback container (Windows / mmap failure). Always access the
 * payload through data()/payload_size().
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

  /// GGML row-block payload (Q4_0/Q6_K); NONE for the packed-4bit/raw formats.
  nntrainer::TensorDim::DataType ggml_dtype =
    nntrainer::TensorDim::DataType::NONE;
  size_t row_bytes = 0; ///< payload stride per row (ggml mode)

  /// sfixed4 only: how many equal-width scale blocks a row is split into.
  /// 1 = ONE scale per row (manifest quant-type "per-row-symmetric", the
  /// original layout). >1 is "per-row-per-block-symmetric" (the folded
  /// per-layer table: one block per decoder layer), where
  /// row_scales[row * blocks + col / (out_dim / blocks)] scales column `col`.
  size_t sfixed4_blocks = 1;

  void *mmap_ptr = nullptr;
  size_t mmap_len = 0;

  const uint8_t *data() const {
    return mmap_ptr ? static_cast<const uint8_t *>(mmap_ptr) : bytes.data();
  }
  size_t payload_size() const { return mmap_ptr ? mmap_len : bytes.size(); }

  QuantLut() = default;
  QuantLut(const QuantLut &) = delete;
  QuantLut &operator=(const QuantLut &) = delete;
  ~QuantLut();
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
WIN_EXPORT void
decode_quant_lut_row_to_uint16(const QuantLut &lut, size_t token_idx,
                               float layer_scale, float output_quant_scale,
                               int output_quant_offset, uint16_t *output,
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
  void forwardSidecarLut(nntrainer::RunLayerContext &context, unsigned int from,
                         unsigned int to);

  std::tuple<nntrainer::props::InDim, nntrainer::props::OutDim,
             nntrainer::props::Scale, props::QuantizedLutPath,
             props::OutputQuantScale, props::OutputQuantOffset,
             props::SidecarExportPath>
    embedding_props;
  unsigned int weight_idx;
  std::shared_ptr<QuantLut> quant_lut;
  /** CUDA dev-act pinned staging (cudaHostAlloc), PER INSTANCE. This was a
   *  function-scope static, which was safe while only the PLE used this class;
   *  once embedding0 became an EmbeddingLayer too, both layers shared one
   *  buffer and the second lookup overwrote the first one's still-in-flight
   *  async H2D copy => corrupted residual seed => CUDA garbage. */
  void *cuda_stage = nullptr;
  size_t cuda_stage_cap = 0; ///< capacity in BYTES (activation dtype varies)

  /// On-GPU LUT gather (CUDA M==1 decode): cuda_emb_gather handle for this
  /// layer's sidecar (-2 = not attempted, -1 = unavailable/refused, >= 0 =
  /// registered). Plain ints so non-CUDA builds need no guards.
  int cuda_gather_handle = -2;
  /// True when the LIVE M2-B decode graph captured this layer's gather kernel
  /// at the recorded epoch: the per-token feed then only publishes the token
  /// id (the replay performs the gather). A graph captured WITHOUT the gather
  /// (dispatch refused mid-capture) keeps false, so the feed still refreshes
  /// the host staging that graph depends on.
  bool cuda_gather_in_graph = false;
  unsigned cuda_gather_epoch = 0;
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __EMBEDDING_H__ */
