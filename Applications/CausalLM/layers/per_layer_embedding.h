// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   per_layer_embedding.h
 * @date   17 June 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author haehun.yang <haehun.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Multi-output embedding layer that loads per-layer embeddings from a
 *         single file and distributes them across multiple output tensors.
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
 *        the identity mapping (l -> l) is used. See the source-indexing note
 *        in the original Gemma4_E2B_QNN::fill_prefill_ple_chunk_().
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
 * @class PerLayerEmbeddingLayer
 * @brief Multi-output embedding layer that loads per-layer embeddings from a
 *        single file and distributes them across multiple output tensors.
 *
 * Supports three file formats (auto-detected by extension / manifest datatype):
 * - *.json manifest with datatype "sfixed4": rowwise signed 4-bit quantized
 *   (two nibbles per byte), per-row-per-layer scales.
 * - *.json manifest with datatype "ufixed8": tensorwise unsigned 8-bit
 *   quantized (one byte per element), single scale/offset.
 * - raw uint16 binary: unquantized, already in consumer space.
 *
 * Input: 1 tensor of token IDs. The input dtype is forced to FP32 in
 * finalize() (matching EmbeddingLayer), since the layer is often the model
 * entry point and token IDs index a vocab that can exceed UINT16 range.
 * Outputs: ple_layer_count UINT16 tensors, one per consumed model layer.
 *
 * Properties (see causallm::props above):
 * - ple_file_path        : manifest or raw binary path (required)
 * - ple_layer_count      : number of output tensors (req. for raw uint16)
 * - ple_per_layer_width  : elements per layer (default 256)
 * - ple_model_indices    : output slot -> source model-layer index (optional)
 * - ple_scales           : consumer-space scale per output (optional, def 1.0)
 * - ple_offsets          : consumer-space offset per output (optional, def 0)
 */
WIN_EXPORT class PerLayerEmbeddingLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Constructor
   */
  WIN_EXPORT PerLayerEmbeddingLayer();

  /**
   * @brief Destructor
   */
  WIN_EXPORT ~PerLayerEmbeddingLayer();

  /**
   * @brief Move constructor. The layer uniquely owns a file descriptor and an
   * mmap region, so a defaulted move would shallow-copy them and let the
   * moved-from destructor munmap/close the still-referenced mapping (double
   * free / use-after-unmap). This transfers ownership and clears the source.
   * Copy operations are implicitly deleted by declaring these moves.
   */
  PerLayerEmbeddingLayer(PerLayerEmbeddingLayer &&rhs) noexcept;

  /**
   * @brief Move assignment. Releases this object's own mapping first, then
   * transfers ownership from @a rhs and clears it.
   */
  PerLayerEmbeddingLayer &operator=(PerLayerEmbeddingLayer &&rhs) noexcept;

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
  size_t ple_per_layer_ = 0;    // Width per layer
  size_t ple_row_elems_ = 0;    // Total vocab row width
  size_t ple_row_bytes_ = 0;    // Byte count per row
  size_t ple_layers_ = 0;       // Total model layer count in the file
  size_t ple_output_count_ = 0; // Number of output tensors to fill
  size_t ple_vocab_ = 0;        // Number of rows (vocab size) for bounds check

  // Per-output-slot mapping and consumer-space quant params
  std::vector<int> ple_model_index_;      // output slot -> source model layer
  std::vector<float> ple_consumer_scale_; // requant scale per output slot
  std::vector<int> ple_consumer_offset_;  // requant offset per output slot

  // Quantization parameters (source side)
  float ple_scale_ = 1.0f;
  int ple_offset_ = 0;
  std::vector<float> ple_row_layer_scales_;

  /**
   * @brief open and mmap the per-layer embedding file, parsing the manifest
   * @param filename path to the JSON manifest or raw uint16 binary
   */
  void open_ple_file_(const std::string &filename);

  /**
   * @brief munmap the embedding file and close its descriptor
   */
  void close_ple_file_();

  /**
   * @brief decode a signed 4-bit nibble (two's complement) to int
   * @param nib 4-bit value
   * @return sign-extended integer in [-8, 7]
   */
  static inline int s4_decode(unsigned nib) {
    return (nib & 0x8u) ? static_cast<int>(nib) - 16 : static_cast<int>(nib);
  }

  /**
   * @brief dequantize a signed-4bit row slice and requantize to UINT16
   * @param src packed nibble source
   * @param len number of elements
   * @param src_scale per-row source scale
   * @param dst_scale consumer-space scale
   * @param dst_offset consumer-space offset
   * @param dst UINT16 destination
   */
  void dequant_sfixed4_requant_u16_(const uint8_t *src, size_t len,
                                    float src_scale, float dst_scale,
                                    int dst_offset, uint16_t *dst) const;

  /**
   * @brief dequantize an unsigned 8-bit (one byte per element) row slice and
   *        requantize to UINT16
   * @param src byte source (one element per byte)
   * @param len number of elements
   * @param src_scale source LUT scale
   * @param src_offset source LUT offset
   * @param dst_scale consumer-space scale
   * @param dst_offset consumer-space offset
   * @param dst UINT16 destination
   */
  void dequant_bytes_requant_u16_(const uint8_t *src, size_t len,
                                  float src_scale, int src_offset,
                                  float dst_scale, int dst_offset,
                                  uint16_t *dst) const;
};

} // namespace causallm

#endif // __cplusplus
#endif // __PER_LAYER_EMBEDDING_H__
