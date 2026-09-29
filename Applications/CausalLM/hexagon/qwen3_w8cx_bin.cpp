// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qwen3_w8cx_bin.cpp
 * @date	31 August 2026
 * @brief	mmap view over a W8_CX / w4cx qwen3 .bin
 *		(tools/hexagon/make_w8cx_bin.py, or make_w4cx_bin.py on hvx_w4cx)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "qwen3_w8cx_bin.h"

#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nntrainer::hexagon {

namespace {

struct Cursor {
  uint8_t *p;
  uint8_t *end;
  const int8_t *i8(uint64_t n) { return (const int8_t *)take(n); }
  const float *f32(uint64_t n) {
    // fp32 blocks are only aligned when every preceding int8 blob is a
    // multiple of 4 bytes; true for qwen3-0.6b, asserted rather than assumed.
    assert(((uintptr_t)p & 3u) == 0);
    return (const float *)take(n * 4ull);
  }
  void *take(uint64_t bytes) {
    if ((uint64_t)(end - p) < bytes)
      throw std::runtime_error("w8cx bin: truncated");
    void *r = p;
    p += bytes;
    return r;
  }
};

/** int8 [n][k] blob followed by n fp32 scales. */
void quantized(Cursor &c, const int8_t *&q, const float *&s, uint64_t n,
               uint64_t k) {
  q = c.i8(n * k);
  s = c.f32(n);
}

uint64_t q_bytes(uint64_t n, uint64_t k) { return n * k + n * 4ull; }

} // namespace

uint64_t Qwen3W8cxBin::expected_size(const HexModelConfig &c) {
  const uint64_t qdim = (uint64_t)c.n_heads * c.head_dim;
  const uint64_t kvdim = (uint64_t)c.n_kv_heads * c.head_dim;
  const uint64_t per_layer =
    4ull * c.hidden + q_bytes(qdim, c.hidden) + 4ull * c.head_dim +
    q_bytes(kvdim, c.hidden) + 4ull * c.head_dim + q_bytes(kvdim, c.hidden) +
    q_bytes(c.hidden, qdim) + 4ull * c.hidden + q_bytes(c.ffn, c.hidden) +
    q_bytes(c.ffn, c.hidden) + q_bytes(c.hidden, c.ffn);
  return q_bytes(c.vocab, c.hidden) + per_layer * c.n_layers + 4ull * c.hidden;
}

Qwen3W8cxBin::Qwen3W8cxBin(const std::string &path, const HexModelConfig &cfg) {
  fd_ = open(path.c_str(), O_RDONLY);
  if (fd_ < 0)
    throw std::runtime_error("w8cx bin: cannot open " + path);
  struct stat st;
  if (fstat(fd_, &st) != 0) {
    close(fd_);
    throw std::runtime_error("w8cx bin: fstat failed " + path);
  }
  size_ = (uint64_t)st.st_size;
  const uint64_t want = expected_size(cfg);
  // A W4CX file is the same payload behind a 64-byte header.
  const bool w4 = size_ == want + sizeof(Qwen3W4cxBinHeader);
  if (size_ != want && !w4) {
    close(fd_);
    throw std::runtime_error(
      "w8cx bin: size " + std::to_string(size_) + " != expected " +
      std::to_string(want) + " (W8) or " +
      std::to_string(want + sizeof(Qwen3W4cxBinHeader)) + " (w4cx)");
  }
  void *m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (m == MAP_FAILED) {
    close(fd_);
    throw std::runtime_error("w8cx bin: mmap failed " + path);
  }
  base_ = (uint8_t *)m;

  const uint64_t qdim = (uint64_t)cfg.n_heads * cfg.head_dim;
  const uint64_t kvdim = (uint64_t)cfg.n_kv_heads * cfg.head_dim;
  Cursor c{base_, base_ + size_};

  w_.i8_mask = kHexAllI8;
  if (w4) {
    Qwen3W4cxBinHeader h;
    std::memcpy(&h, c.take(sizeof(h)), sizeof(h));
    if (std::memcmp(h.magic, "W4CX", 4) != 0 || h.version != 1u ||
        h.bits != 4u || h.n_layers != cfg.n_layers ||
        (h.i8_mask & ~kHexAllI8) != 0u)
      throw std::runtime_error("w8cx bin: bad w4cx header in " + path);
    w_.i8_mask = h.i8_mask;
  }

  quantized(c, w_.embed, w_.embed_s, cfg.vocab, cfg.hidden);
  w_.layers.resize(cfg.n_layers);
  for (auto &l : w_.layers) {
    l.attn_norm = c.f32(cfg.hidden);
    quantized(c, l.wq, l.wq_s, qdim, cfg.hidden);
    l.q_norm = c.f32(cfg.head_dim);
    quantized(c, l.wk, l.wk_s, kvdim, cfg.hidden);
    l.k_norm = c.f32(cfg.head_dim);
    quantized(c, l.wv, l.wv_s, kvdim, cfg.hidden);
    quantized(c, l.wo, l.wo_s, cfg.hidden, qdim);
    l.ffn_norm = c.f32(cfg.hidden);
    // nntrainer stores the mlp weights in up, gate order.
    quantized(c, l.w_up, l.w_up_s, cfg.ffn, cfg.hidden);
    quantized(c, l.w_gate, l.w_gate_s, cfg.ffn, cfg.hidden);
    quantized(c, l.w_down, l.w_down_s, cfg.hidden, cfg.ffn);
  }
  w_.final_norm = c.f32(cfg.hidden);
  if (c.p != c.end)
    throw std::runtime_error("w8cx bin: trailing bytes");
}

void Qwen3W8cxBin::apply_layout(HexModelConfig &cfg) const {
  cfg.i8_mask = w_.i8_mask;
  if (w_.i8_mask == kHexAllI8) {
    cfg.weight_layout = NNTR_HTP_WEIGHT_LAYOUT_TILED32;
    return;
  }
  if ((w_.i8_mask & kHexW4cxDown8I8) != kHexW4cxDown8I8)
    throw std::runtime_error("w8cx bin: a 4-bit embed or down needs the w4cx "
                             "layout of #65 S3; only w4cx_down8 (embed and "
                             "down int8) packs today");
  cfg.weight_layout = NNTR_HTP_WEIGHT_LAYOUT_W4CX_DOWN8;
}

Qwen3W8cxBin::~Qwen3W8cxBin() {
  if (base_)
    munmap(base_, size_);
  if (fd_ >= 0)
    close(fd_);
}

} // namespace nntrainer::hexagon
