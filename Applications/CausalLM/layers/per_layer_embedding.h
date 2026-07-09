// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   per_layer_embedding.h
 * @date   17 June 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author haehun.yang <haehun.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Per-layer embedding loader/dequantizer, exposed both as a standalone
 *         engine (PerLayerEmbeddingEngine) and as an nntrainer multi-output
 *         layer (PerLayerEmbeddingLayer).
 */

#ifndef __PER_LAYER_EMBEDDING_H__
#define __PER_LAYER_EMBEDDING_H__
#ifdef __cplusplus

#pragma once

#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <base_properties.h>
#include <common_properties.h>
#include <cstdint>
#include <layer_context.h>
#include <layer_impl.h>
#include <memory>
#include <node_exporter.h>
#include <string>
#include <tuple>
#include <util_func.h>
#include <vector>

namespace causallm {

namespace props {

/**
 * @brief Path to the per-layer embedding file (JSON manifest or raw binary).
 */
class PleFilePath : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "ple_file_path";
  using prop_tag = nntrainer::str_prop_tag;
};

/**
 * @brief Number of output tensors to fill (one per consumed model layer).
 *        Optional for JSON manifests (auto-detected); REQUIRED for raw uint16.
 */
class PleLayerCount : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "ple_layer_count";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Number of elements per layer in each embedding row (default 256).
 */
class PlePerLayerWidth : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "ple_per_layer_width";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Comma-separated model-layer index for each output slot.
 *        Maps dense output slot l -> source model layer index. When empty,
 *        the identity mapping (l -> l) is used.
 */
class PleModelIndices : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "ple_model_indices";
  using prop_tag = nntrainer::str_prop_tag;
};

/**
 * @brief Comma-separated consumer-space quant scale per output slot.
 *        When empty, all scales default to 1.0 (no requant).
 */
class PleScales : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "ple_scales";
  using prop_tag = nntrainer::str_prop_tag;
};

/**
 * @brief Comma-separated consumer-space quant offset per output slot.
 *        When empty, all offsets default to 0.
 */
class PleOffsets : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "ple_offsets";
  using prop_tag = nntrainer::str_prop_tag;
};

} // namespace props

/**
 * @class PerLayerEmbeddingEngine
 * @brief Standalone per-layer embedding loader + dequantizer, independent of
 *        any nntrainer graph/RunLayerContext.
 *
 * The engine owns a memory-mapped per-layer embedding file and produces, for a
 * given token id, the per-layer UINT16 embedding chunks written into arbitrary
 * destination buffers. It is the single source of PLE logic shared by both the
 * nntrainer PerLayerEmbeddingLayer and the QNN runners (which dequant directly
 * into QNN graph input buffers).
 *
 * Three file formats (auto-detected by extension / manifest datatype):
 * - *.json + "sfixed4": rowwise signed 4-bit (two nibbles per byte), per-row-
 *   per-layer scales. A binary sidecar (`<manifest>.sf4cache`) is written on
 *   first parse so later cold starts skip the multi-million-float JSON parse.
 * - *.json + "ufixed8": tensorwise unsigned 8-bit (one byte per element),
 *   single scale/offset.
 * - raw uint16 binary: already in consumer space; per-layer fill is a memcpy.
 */
class PerLayerEmbeddingEngine {
public:
  WIN_EXPORT PerLayerEmbeddingEngine() = default;
  WIN_EXPORT ~PerLayerEmbeddingEngine() { close(); }

  // Move-only: uniquely owns a file descriptor and mmap region.
  WIN_EXPORT PerLayerEmbeddingEngine(PerLayerEmbeddingEngine &&rhs) noexcept;
  WIN_EXPORT PerLayerEmbeddingEngine &
  operator=(PerLayerEmbeddingEngine &&rhs) noexcept;

  /**
   * @brief Load and mmap the per-layer embedding file.
   * @param file_path manifest (*.json) or raw uint16 binary
   * @param per_layer_width elements per layer in a row (e.g. 256 / 192)
   * @param raw_layer_count_hint number of layers for the raw-uint16 format
   *        (the row layout is derived from it); ignored for JSON manifests
   */
  WIN_EXPORT void open(const std::string &file_path, size_t per_layer_width,
                       size_t raw_layer_count_hint = 0);

  /** @brief munmap the file and close the descriptor */
  WIN_EXPORT void close();

  /** @brief true once a file has been successfully mapped */
  WIN_EXPORT bool isOpen() const {
    return ple_mmap_ != nullptr || ple_u16_mmap_ != nullptr;
  }

  /** @brief number of model layers stored per row */
  WIN_EXPORT size_t numLayers() const { return ple_layers_; }
  /** @brief elements per layer */
  WIN_EXPORT size_t perLayerWidth() const { return ple_per_layer_; }
  /** @brief number of rows (vocabulary size) */
  WIN_EXPORT size_t vocab() const { return ple_vocab_; }

  /**
   * @brief Dequantize one token's per-layer embeddings into destination
   *        buffers (one buffer per output slot).
   *
   * For output slot @c l the source row chunk @c model_index[l] is dequantized
   * and requantized to the consumer space (@c consumer_scale[l],
   * @c consumer_offset[l]) and written as @c perLayerWidth() UINT16 values to
   * @c dsts[l] + @c dst_elem_offset.
   *
   * @param token_id row index into the embedding table
   * @param model_index per-slot source model-layer index
   * @param consumer_scale per-slot consumer-space scale
   * @param consumer_offset per-slot consumer-space offset
   * @param dsts per-slot destination base pointers
   * @param dst_elem_offset element offset added to each destination
   */
  WIN_EXPORT void fillToken(int token_id, const std::vector<int> &model_index,
                            const std::vector<float> &consumer_scale,
                            const std::vector<int> &consumer_offset,
                            const std::vector<uint16_t *> &dsts,
                            size_t dst_elem_offset) const;

private:
  // File I/O and mmap
  int ple_fd_ = -1;
  const uint8_t *ple_mmap_ = nullptr;
  const uint16_t *ple_u16_mmap_ = nullptr;
  size_t ple_file_size_ = 0;

  // Format detection
  bool ple_is_quantized_ =
    false;                      // manifest-backed (sfixed4/ufixed8) vs raw u16
  bool ple_is_signed4_ = false; // sfixed4 (signed 4-bit nibbles); else ufixed8

  // Layout metadata
  size_t ple_per_layer_ = 0; // width per layer
  size_t ple_row_elems_ = 0; // total row width (layers * per_layer)
  size_t ple_row_bytes_ = 0; // bytes per row
  size_t ple_layers_ = 0;    // model layers stored per row
  size_t ple_vocab_ = 0;     // number of rows

  // Source-side quantization parameters
  float ple_scale_ = 1.0f;                  // ufixed8 only
  int ple_offset_ = 0;                      // ufixed8 only
  std::vector<float> ple_row_layer_scales_; // sfixed4: [vocab][layers]

  void open_manifest_(const std::string &manifest_path);
  void open_raw_u16_(const std::string &file_path, size_t raw_layer_count_hint);
  // sfixed4 binary sidecar cache (skips the JSON scale-array parse).
  bool load_sf4_cache_(const std::string &manifest_path);
  void write_sf4_cache_(const std::string &manifest_path,
                        const std::string &lut_abs) const;
  void reset_();

  /** @brief sign-extend a 4-bit value (0..15 -> -8..7) */
  static inline int s4_decode(unsigned nib) {
    return (nib & 0x8u) ? static_cast<int>(nib) - 16 : static_cast<int>(nib);
  }
  static void dequant_sfixed4_requant_u16_(const uint8_t *packed, size_t elems,
                                           float row_scale, float out_scale,
                                           int out_offset, uint16_t *dst);
  static void dequant_bytes_requant_u16_(const uint8_t *src, size_t elems,
                                         float lut_scale, int lut_offset,
                                         float out_scale, int out_offset,
                                         uint16_t *dst);
};

/**
 * @class PerLayerEmbeddingLayer
 * @brief Multi-output nntrainer layer that distributes per-layer embeddings
 *        from a single file across multiple output tensors.
 *
 * Thin wrapper over PerLayerEmbeddingEngine.
 * Input: 1 tensor of token IDs (forced FP32 in finalize, matching
 * EmbeddingLayer). Outputs: ple_layer_count UINT16 tensors, one per consumed
 * model layer.
 *
 * Properties (see causallm::props above): ple_file_path (required),
 * ple_layer_count (req. for raw uint16), ple_per_layer_width (default 256),
 * ple_model_indices / ple_scales / ple_offsets (optional per-slot config).
 */
WIN_EXPORT class PerLayerEmbeddingLayer : public nntrainer::LayerImpl {
public:
  /** @brief Constructor */
  WIN_EXPORT PerLayerEmbeddingLayer();

  /** @brief Destructor (engine RAII releases the mapping) */
  WIN_EXPORT ~PerLayerEmbeddingLayer() = default;

  // Move-only (the engine owns the fd/mmap and is itself move-only).
  PerLayerEmbeddingLayer(PerLayerEmbeddingLayer &&rhs) noexcept = default;
  PerLayerEmbeddingLayer &
  operator=(PerLayerEmbeddingLayer &&rhs) noexcept = default;

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
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
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
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override { return type; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "per_layer_embedding_layer";

private:
  std::tuple<props::PleFilePath, props::PleLayerCount, props::PlePerLayerWidth,
             props::PleModelIndices, props::PleScales, props::PleOffsets>
    ple_props;

  PerLayerEmbeddingEngine engine_;

  size_t ple_per_layer_ = 0;
  size_t ple_output_count_ = 0;
  std::vector<int> ple_model_index_;      // output slot -> source model layer
  std::vector<float> ple_consumer_scale_; // requant scale per output slot
  std::vector<int> ple_consumer_offset_;  // requant offset per output slot
};

} // namespace causallm

#endif // __cplusplus
#endif // __PER_LAYER_EMBEDDING_H__
