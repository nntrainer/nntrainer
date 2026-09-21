// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qwen3_w8cx_bin.h
 * @date	31 August 2026
 * @brief	mmap view over a W8_CX / w4cx qwen3 .bin
 *		(tools/hexagon/make_w8cx_bin.py, or make_w4cx_bin.py on hvx_w4cx)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef __CAUSALLM_HEXAGON_QWEN3_W8CX_BIN_H__
#define __CAUSALLM_HEXAGON_QWEN3_W8CX_BIN_H__

#include <cstdint>
#include <string>

#include "graph_lowering.h"

namespace nntrainer::hexagon {

/** qwen3-0.6b checkpoint shape, the only W8_CX .bin this reader knows. */
inline const HexModelConfig kQwen3_0_6b = {
  /*n_layers=*/28,   /*n_heads=*/16,    /*n_kv_heads=*/8,
  /*head_dim=*/128,
  /*hidden=*/1024,   /*ffn=*/3072,      /*vocab=*/151936,
  /*max_seq=*/2048,
  /*max_chunk=*/128, /*rms_eps=*/1e-6f, /*rope_theta=*/1e6f};

/**
 * @brief The 64-byte header of a W4CX checkpoint (issue #65 S1,
 *        written by make_w4cx_bin.py on the hvx_w4cx branch). A W8
 *        .bin has no header. Only the int width of the tensors changes:
 *        an int4 tensor is still stored one code per byte ([-7, 7]) at the
 *        same size, so the stream after the header is laid out exactly as
 *        the W8 one and the reader is shared.
 */
struct Qwen3W4cxBinHeader {
  char magic[4];     /**< "W4CX" */
  uint32_t version;  /**< 1 */
  uint32_t n_layers; /**< layers in the file */
  uint32_t i8_mask;  /**< HexTensorBit set = int8 tensor class */
  uint32_t bits;     /**< 4: width of the classes not in i8_mask */
  uint8_t pad[44];
};
typedef char
  qwen3_w4cx_bin_header_size_check[(sizeof(Qwen3W4cxBinHeader) == 64) ? 1 : -1];

/**
 * @class Qwen3W8cxBin
 * @brief Read-only view over a W8_CX or w4cx .bin
 *        (tools/hexagon/make_w8cx_bin.py or make_w4cx_bin.py). mmaps the
 *        file and hands out non-owning pointers into it; the object must
 *        outlive the HexModelWeights it returns.
 *
 * The payload is a stream in graph layer order: embedding
 * (int8 [vocab][hidden] + fp32 [vocab]), then per layer attn_norm, wq,
 * q_norm, wk, k_norm, wv, wo, ffn_norm, ffn_up, ffn_gate, ffn_down (2D
 * tensors as int8 [N][K] + fp32 [N], norms as fp32), then output_norm.
 * A W8 file is that stream alone; a W4CX file prefixes the
 * Qwen3W4cxBinHeader and stores its int4 tensors as one code per byte.
 */
class Qwen3W8cxBin {
public:
  /** @throw std::runtime_error on open/size/structure mismatch. */
  Qwen3W8cxBin(const std::string &path, const HexModelConfig &cfg);
  ~Qwen3W8cxBin();
  Qwen3W8cxBin(const Qwen3W8cxBin &) = delete;
  Qwen3W8cxBin &operator=(const Qwen3W8cxBin &) = delete;

  const HexModelWeights &weights() const { return w_; }

  /** The WEIGHTS layout this checkpoint packs into: tiled32 for a W8
   * file, w4cx_down8 for a W4CX file (whose i8 set must include embed
   * and down until #65 S3). Sets cfg.weight_layout / cfg.i8_mask.
   * @throw std::runtime_error when no v5 layout fits the file's i8 set */
  void apply_layout(HexModelConfig &cfg) const;

  /** Byte size the payload must have for this shape (a W4CX file is
   * 64 bytes longer). */
  static uint64_t expected_size(const HexModelConfig &cfg);

private:
  int fd_ = -1;
  uint8_t *base_ = nullptr;
  uint64_t size_ = 0;
  HexModelWeights w_;
};

} // namespace nntrainer::hexagon
#endif // __CAUSALLM_HEXAGON_QWEN3_W8CX_BIN_H__
