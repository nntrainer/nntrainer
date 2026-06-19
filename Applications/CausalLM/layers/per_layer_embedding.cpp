// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   per_layer_embedding.cpp
 * @date   17 June 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author haehun.yang <haehun.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Per-layer embedding loader/dequantizer engine and the nntrainer
 *         multi-output layer wrapping it.
 */

#include "per_layer_embedding.h"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#if defined(_WIN32)
#include <io.h>
#include <mman_windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "json.hpp"

using json = nlohmann::json;

namespace causallm {

namespace {

constexpr size_t SINGLE_INOUT_IDX = 0;

/** @brief return true if string @a s ends with suffix @a suf */
bool ends_with(const std::string &s, const std::string &suf) {
  return s.size() >= suf.size() &&
         0 == s.compare(s.size() - suf.size(), suf.size(), suf);
}

/** @brief return the directory component of a path (empty if none) */
std::string dirname(const std::string &p) {
  auto pos = p.find_last_of('/');
  return (pos == std::string::npos) ? std::string() : p.substr(0, pos);
}

/** @brief rebase a relative path against the manifest's directory */
std::string rebase_relative_to_model_file(const std::string &path,
                                          const std::string &manifest_path) {
  if (path.empty() || path[0] == '/' || manifest_path.empty())
    return path;
  const std::string base = dirname(manifest_path);
  if (base.empty()) // bare manifest (e.g. "ple.json"): keep path relative
    return path;
  return base + "/" + path;
}

/** @brief parse a comma-separated list into a vector via the given converter */
template <typename T, typename Conv>
std::vector<T> parse_csv(const std::string &s, Conv conv) {
  std::vector<T> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t b = item.find_first_not_of(" \t");
    if (b == std::string::npos)
      continue;
    size_t e = item.find_last_not_of(" \t");
    out.push_back(conv(item.substr(b, e - b + 1)));
  }
  return out;
}

// ── sfixed4 binary sidecar cache ──────────────────────────────────
// The sfixed4 scale array is shape [vocab, num_layers] (e.g. 262144*35 ≈
// 9.1M floats); re-parsing it from JSON on every cold start dominates load
// time. After the first parse a compact binary is dumped next to the manifest
// and the JSON parse is skipped on later loads. Invalidated on mtime change.
constexpr char kPleSf4Magic[4] = {'P', 'S', '4', 'C'};
constexpr uint32_t kPleSf4Version = 1;

#pragma pack(push, 1)
struct PleSf4CacheHeader {
  char magic[4];
  uint32_t version;
  int64_t manifest_mtime;
  uint64_t row_elems;
  uint64_t layers;
  uint64_t scale_count;
  uint32_t lut_path_len;
};
#pragma pack(pop)

int64_t ple_file_mtime_(const std::string &p) {
  struct stat st {};
  return (::stat(p.c_str(), &st) == 0) ? static_cast<int64_t>(st.st_mtime) : -1;
}

std::string ple_sf4_cache_path_(const std::string &manifest_path) {
  return manifest_path + ".sf4cache";
}

} // namespace

// =====================================================================
// PerLayerEmbeddingEngine
// =====================================================================
PerLayerEmbeddingEngine::PerLayerEmbeddingEngine(
  PerLayerEmbeddingEngine &&rhs) noexcept :
  ple_fd_(rhs.ple_fd_),
  ple_mmap_(rhs.ple_mmap_),
  ple_u16_mmap_(rhs.ple_u16_mmap_),
  ple_file_size_(rhs.ple_file_size_),
  ple_is_quantized_(rhs.ple_is_quantized_),
  ple_is_signed4_(rhs.ple_is_signed4_),
  ple_per_layer_(rhs.ple_per_layer_),
  ple_row_elems_(rhs.ple_row_elems_),
  ple_row_bytes_(rhs.ple_row_bytes_),
  ple_layers_(rhs.ple_layers_),
  ple_vocab_(rhs.ple_vocab_),
  ple_scale_(rhs.ple_scale_),
  ple_offset_(rhs.ple_offset_),
  ple_row_layer_scales_(std::move(rhs.ple_row_layer_scales_)) {
  rhs.ple_fd_ = -1;
  rhs.ple_mmap_ = nullptr;
  rhs.ple_u16_mmap_ = nullptr;
  rhs.ple_file_size_ = 0;
}

PerLayerEmbeddingEngine &
PerLayerEmbeddingEngine::operator=(PerLayerEmbeddingEngine &&rhs) noexcept {
  if (this != &rhs) {
    close();
    ple_fd_ = rhs.ple_fd_;
    ple_mmap_ = rhs.ple_mmap_;
    ple_u16_mmap_ = rhs.ple_u16_mmap_;
    ple_file_size_ = rhs.ple_file_size_;
    ple_is_quantized_ = rhs.ple_is_quantized_;
    ple_is_signed4_ = rhs.ple_is_signed4_;
    ple_per_layer_ = rhs.ple_per_layer_;
    ple_row_elems_ = rhs.ple_row_elems_;
    ple_row_bytes_ = rhs.ple_row_bytes_;
    ple_layers_ = rhs.ple_layers_;
    ple_vocab_ = rhs.ple_vocab_;
    ple_scale_ = rhs.ple_scale_;
    ple_offset_ = rhs.ple_offset_;
    ple_row_layer_scales_ = std::move(rhs.ple_row_layer_scales_);
    rhs.ple_fd_ = -1;
    rhs.ple_mmap_ = nullptr;
    rhs.ple_u16_mmap_ = nullptr;
    rhs.ple_file_size_ = 0;
  }
  return *this;
}

void PerLayerEmbeddingEngine::close() {
  if (ple_mmap_) {
    munmap((void *)ple_mmap_, ple_file_size_);
    ple_mmap_ = nullptr;
  }
  if (ple_u16_mmap_) {
    munmap((void *)ple_u16_mmap_, ple_file_size_);
    ple_u16_mmap_ = nullptr;
  }
  if (ple_fd_ >= 0) {
    ::close(ple_fd_);
    ple_fd_ = -1;
  }
}

void PerLayerEmbeddingEngine::open(const std::string &file_path,
                                   size_t per_layer_width,
                                   size_t raw_layer_count_hint) {
  if (file_path.empty())
    throw std::runtime_error("PerLayerEmbeddingEngine: file path is empty");
  NNTR_THROW_IF(per_layer_width == 0, std::invalid_argument)
    << "PerLayerEmbeddingEngine: per_layer_width is 0";

  ple_per_layer_ = per_layer_width;
  ple_is_quantized_ = ends_with(file_path, ".json");

  if (ple_is_quantized_)
    open_manifest_(file_path);
  else
    open_raw_u16_(file_path, raw_layer_count_hint);
}

bool PerLayerEmbeddingEngine::load_sf4_cache_(
  const std::string &manifest_path) {
  const std::string cpath = ple_sf4_cache_path_(manifest_path);
  std::ifstream cf(cpath, std::ios::binary);
  if (!cf.is_open())
    return false;

  PleSf4CacheHeader h{};
  cf.read(reinterpret_cast<char *>(&h), sizeof(h));
  const bool hdr_ok = static_cast<bool>(cf) &&
                      std::memcmp(h.magic, kPleSf4Magic, 4) == 0 &&
                      h.version == kPleSf4Version &&
                      h.manifest_mtime == ple_file_mtime_(manifest_path);
  if (!hdr_ok)
    return false;

  std::string lut_abs(h.lut_path_len, '\0');
  cf.read(&lut_abs[0], h.lut_path_len);

  ple_is_signed4_ = true;
  ple_row_elems_ = h.row_elems;
  ple_layers_ = h.layers;
  ple_row_bytes_ = (ple_row_elems_ + 1) / 2;
  ple_scale_ = 1.0f;
  ple_offset_ = 0;
  ple_row_layer_scales_.resize(h.scale_count);
  cf.read(reinterpret_cast<char *>(ple_row_layer_scales_.data()),
          static_cast<std::streamsize>(h.scale_count * sizeof(float)));
  if (!static_cast<bool>(cf) || ple_layers_ == 0 ||
      ple_layers_ * ple_per_layer_ != ple_row_elems_) {
    reset_();
    return false;
  }

  ple_fd_ = ::open(lut_abs.c_str(), O_RDONLY);
  struct stat st;
  if (ple_fd_ < 0 || fstat(ple_fd_, &st) != 0) {
    reset_();
    return false;
  }
  ple_file_size_ = static_cast<size_t>(st.st_size);
  const size_t exp_vocab = ple_row_bytes_ ? ple_file_size_ / ple_row_bytes_ : 0;
  const size_t scl_vocab = ple_row_layer_scales_.size() / ple_layers_;
  if (ple_row_bytes_ == 0 || ple_file_size_ % ple_row_bytes_ != 0 ||
      scl_vocab != exp_vocab) {
    reset_();
    return false;
  }

  void *m = mmap(nullptr, ple_file_size_, PROT_READ, MAP_PRIVATE, ple_fd_, 0);
  if (m == MAP_FAILED) {
    reset_();
    return false;
  }
  ple_mmap_ = static_cast<const uint8_t *>(m);
  ple_vocab_ = exp_vocab;
#ifdef POSIX_MADV_RANDOM
  posix_madvise((void *)ple_mmap_, ple_file_size_, POSIX_MADV_RANDOM);
#endif
  ml_logd(
    "[PLE] sfixed4 from binary cache %s rows=%zu layers=%zu per_layer=%zu "
    "scales=%zu (JSON parse skipped)",
    cpath.c_str(), ple_vocab_, ple_layers_, ple_per_layer_,
    ple_row_layer_scales_.size());
  return true;
}

void PerLayerEmbeddingEngine::write_sf4_cache_(
  const std::string &manifest_path, const std::string &lut_abs) const {
  // Best-effort + atomic rename; a read-only model dir simply keeps JSON.
  const std::string cpath = ple_sf4_cache_path_(manifest_path);
  const std::string tmp = cpath + ".tmp";
  std::ofstream wf(tmp, std::ios::binary | std::ios::trunc);
  if (!wf.is_open())
    return;
  PleSf4CacheHeader h{};
  std::memcpy(h.magic, kPleSf4Magic, 4);
  h.version = kPleSf4Version;
  h.manifest_mtime = ple_file_mtime_(manifest_path);
  h.row_elems = ple_row_elems_;
  h.layers = ple_layers_;
  h.scale_count = ple_row_layer_scales_.size();
  h.lut_path_len = static_cast<uint32_t>(lut_abs.size());
  wf.write(reinterpret_cast<const char *>(&h), sizeof(h));
  wf.write(lut_abs.data(), static_cast<std::streamsize>(lut_abs.size()));
  wf.write(
    reinterpret_cast<const char *>(ple_row_layer_scales_.data()),
    static_cast<std::streamsize>(ple_row_layer_scales_.size() * sizeof(float)));
  const bool ok = static_cast<bool>(wf);
  wf.close();
  if (ok && wf) {
    if (std::rename(tmp.c_str(), cpath.c_str()) != 0)
      std::remove(tmp.c_str());
  } else {
    std::remove(tmp.c_str());
  }
}

void PerLayerEmbeddingEngine::open_manifest_(const std::string &manifest_path) {
  // Fast path: sfixed4 binary sidecar cache (skips JSON parse).
  if (load_sf4_cache_(manifest_path))
    return;

  std::ifstream mf(manifest_path);
  if (!mf.is_open())
    throw std::runtime_error("Failed to open PLE manifest: " + manifest_path);

  json j;
  mf >> j;

  const std::string lut_rel = j.at("lut-path").get<std::string>();
  const int row_elems = j.at("size").get<int>();
  const std::string datatype = j.value("datatype", std::string("ufixed8"));
  const auto &qp = j.at("quant-param");

  ple_is_signed4_ = (datatype == "sfixed4");
  if (!ple_is_signed4_ && datatype != "ufixed8")
    throw std::runtime_error("PLE: unsupported datatype: " + datatype);

  ple_row_elems_ = static_cast<size_t>(row_elems);
  // sfixed4 packs two 4-bit values per byte; ufixed8 is one byte per element.
  ple_row_bytes_ = ple_is_signed4_ ? (ple_row_elems_ + 1) / 2 : ple_row_elems_;
  ple_layers_ = ple_row_elems_ / ple_per_layer_;

  if (ple_layers_ * ple_per_layer_ != ple_row_elems_)
    throw std::runtime_error("PLE 'size' not divisible by per_layer width");

  if (ple_is_signed4_) {
    const auto &scale_arr = qp.at("scale");
    if (!scale_arr.is_array())
      throw std::runtime_error("PLE sfixed4: quant-param.scale must be array");
    ple_row_layer_scales_.clear();
    ple_row_layer_scales_.reserve(scale_arr.size());
    for (const auto &v : scale_arr)
      ple_row_layer_scales_.push_back(v.get<float>());
    if (ple_row_layer_scales_.size() % ple_layers_ != 0)
      throw std::runtime_error(
        "PLE sfixed4: scale array length not divisible by num_layers");
    ple_scale_ = 1.0f;
    ple_offset_ = 0;
  } else {
    ple_scale_ = qp.at("scale").get<float>();
    ple_offset_ = qp.at("offset").get<int>();
  }

  std::string lut_abs = rebase_relative_to_model_file(lut_rel, manifest_path);

  ple_fd_ = ::open(lut_abs.c_str(), O_RDONLY);
  if (ple_fd_ < 0)
    throw std::runtime_error("open PLE bin: " + lut_abs);

  struct stat st;
  if (fstat(ple_fd_, &st) < 0) {
    reset_();
    throw std::runtime_error("stat PLE bin: " + lut_abs);
  }
  ple_file_size_ = static_cast<size_t>(st.st_size);
  if (ple_row_bytes_ == 0 || ple_file_size_ == 0 ||
      ple_file_size_ % ple_row_bytes_ != 0) {
    reset_();
    throw std::runtime_error("PLE bin size not a positive multiple of row "
                             "bytes");
  }
  ple_vocab_ = ple_file_size_ / ple_row_bytes_;

  if (ple_is_signed4_) {
    const size_t scale_vocab = ple_row_layer_scales_.size() / ple_layers_;
    if (scale_vocab != ple_vocab_)
      throw std::runtime_error(
        "PLE sfixed4 scale vocab=" + std::to_string(scale_vocab) +
        " != bin vocab=" + std::to_string(ple_vocab_));
  }

  void *m = mmap(nullptr, ple_file_size_, PROT_READ, MAP_PRIVATE, ple_fd_, 0);
  if (m == MAP_FAILED) {
    reset_();
    throw std::runtime_error("mmap PLE bin: " + lut_abs);
  }
  ple_mmap_ = static_cast<const uint8_t *>(m);
#ifdef POSIX_MADV_RANDOM
  posix_madvise((void *)ple_mmap_, ple_file_size_, POSIX_MADV_RANDOM);
#endif

  if (ple_is_signed4_) {
    ml_logd("[PLE] sfixed4 (rowwise+layerwise) mmaped %s rows=%zu layers=%zu "
            "per_layer=%zu scales=%zu",
            lut_abs.c_str(), ple_vocab_, ple_layers_, ple_per_layer_,
            ple_row_layer_scales_.size());
    // Persist a sidecar so the next cold start skips the JSON scale parse.
    write_sf4_cache_(manifest_path, lut_abs);
  } else {
    ml_logd("[PLE] ufixed8 (tensorwise) mmaped %s rows=%zu layers=%zu "
            "per_layer=%zu scale=%f offset=%d",
            lut_abs.c_str(), ple_vocab_, ple_layers_, ple_per_layer_,
            ple_scale_, ple_offset_);
  }
}

void PerLayerEmbeddingEngine::open_raw_u16_(const std::string &file_path,
                                            size_t raw_layer_count_hint) {
  // raw UINT16: row = layers * per_layer uint16, layout derived from the
  // caller-provided layer count.
  if (raw_layer_count_hint == 0)
    throw std::runtime_error(
      "PLE raw uint16: layer count must be provided before loading");
  ple_layers_ = raw_layer_count_hint;
  ple_row_elems_ = ple_layers_ * ple_per_layer_;
  ple_row_bytes_ = ple_row_elems_ * sizeof(uint16_t);
  ple_is_signed4_ = false;
  ple_scale_ = 1.0f;
  ple_offset_ = 0;

  ple_fd_ = ::open(file_path.c_str(), O_RDONLY);
  if (ple_fd_ < 0)
    throw std::runtime_error("open PLE bin: " + file_path);

  struct stat st;
  if (fstat(ple_fd_, &st) < 0) {
    reset_();
    throw std::runtime_error("stat PLE bin: " + file_path);
  }
  ple_file_size_ = static_cast<size_t>(st.st_size);
  if (ple_row_bytes_ == 0 || ple_file_size_ == 0 ||
      ple_file_size_ % ple_row_bytes_ != 0) {
    reset_();
    throw std::runtime_error(
      "PLE raw uint16: file size not a positive multiple of row bytes "
      "(expected multiple of " +
      std::to_string(ple_row_bytes_) + ")");
  }
  ple_vocab_ = ple_file_size_ / ple_row_bytes_;

  void *m = mmap(nullptr, ple_file_size_, PROT_READ, MAP_PRIVATE, ple_fd_, 0);
  if (m == MAP_FAILED) {
    reset_();
    throw std::runtime_error("mmap PLE bin: " + file_path);
  }
  ple_u16_mmap_ = static_cast<const uint16_t *>(m);
#ifdef POSIX_MADV_RANDOM
  posix_madvise((void *)ple_u16_mmap_, ple_file_size_, POSIX_MADV_RANDOM);
#endif
  ml_logd("[PLE] raw uint16 mmaped %s rows=%zu layers=%zu per_layer=%zu",
          file_path.c_str(), ple_vocab_, ple_layers_, ple_per_layer_);
}

void PerLayerEmbeddingEngine::reset_() {
  if (ple_mmap_) {
    munmap((void *)ple_mmap_, ple_file_size_);
    ple_mmap_ = nullptr;
  }
  if (ple_u16_mmap_) {
    munmap((void *)ple_u16_mmap_, ple_file_size_);
    ple_u16_mmap_ = nullptr;
  }
  if (ple_fd_ >= 0) {
    ::close(ple_fd_);
    ple_fd_ = -1;
  }
  ple_file_size_ = 0;
  ple_vocab_ = 0;
}

void PerLayerEmbeddingEngine::fillToken(
  int token_id, const std::vector<int> &model_index,
  const std::vector<float> &consumer_scale,
  const std::vector<int> &consumer_offset, const std::vector<uint16_t *> &dsts,
  size_t dst_elem_offset) const {
  if (!isOpen())
    return;
  NNTR_THROW_IF(token_id < 0, std::invalid_argument)
    << "PerLayerEmbeddingEngine: negative token id " << token_id;
  const size_t tok = static_cast<size_t>(token_id);
  NNTR_THROW_IF(tok >= ple_vocab_, std::invalid_argument)
    << "PerLayerEmbeddingEngine: token id " << tok << " >= vocab "
    << ple_vocab_;

  const size_t per_layer = ple_per_layer_;
  const size_t slots = dsts.size();

  if (ple_is_quantized_) {
    const uint8_t *row = ple_mmap_ + tok * ple_row_bytes_;
    if (ple_is_signed4_) {
      const size_t per_layer_bytes = per_layer / 2;
      const float *row_scales =
        ple_row_layer_scales_.data() + tok * ple_layers_;
      for (size_t l = 0; l < slots; ++l) {
        const size_t ml = static_cast<size_t>(model_index[l]);
        dequant_sfixed4_requant_u16_(
          row + ml * per_layer_bytes, per_layer, row_scales[ml],
          consumer_scale[l], consumer_offset[l], dsts[l] + dst_elem_offset);
      }
    } else {
      const size_t per_layer_bytes = per_layer; // ufixed8: one byte per elem
      for (size_t l = 0; l < slots; ++l) {
        const size_t ml = static_cast<size_t>(model_index[l]);
        dequant_bytes_requant_u16_(
          row + ml * per_layer_bytes, per_layer, ple_scale_, ple_offset_,
          consumer_scale[l], consumer_offset[l], dsts[l] + dst_elem_offset);
      }
    }
  } else {
    const uint16_t *row = ple_u16_mmap_ + tok * ple_row_elems_;
    for (size_t l = 0; l < slots; ++l) {
      const size_t ml = static_cast<size_t>(model_index[l]);
      std::memcpy(dsts[l] + dst_elem_offset, row + ml * per_layer,
                  per_layer * sizeof(uint16_t));
    }
  }
}

void PerLayerEmbeddingEngine::dequant_sfixed4_requant_u16_(
  const uint8_t *packed, size_t elems, float row_scale, float out_scale,
  int out_offset, uint16_t *dst) {
  const float inv_out = 1.0f / out_scale;
  auto requant = [&](unsigned nib) -> uint16_t {
    const float f = static_cast<float>(s4_decode(nib)) * row_scale;
    int q = static_cast<int>(std::lrintf(f * inv_out)) - out_offset;
    return static_cast<uint16_t>(std::max(0, std::min(65535, q)));
  };
  const size_t whole = elems / 2;
  for (size_t i = 0; i < whole; ++i) {
    const uint8_t b = packed[i];
    dst[2 * i] = requant(b & 0x0F);
    dst[2 * i + 1] = requant((b >> 4) & 0x0F);
  }
  if (elems & 1)
    dst[2 * whole] = requant(packed[whole] & 0x0F);
}

void PerLayerEmbeddingEngine::dequant_bytes_requant_u16_(
  const uint8_t *src, size_t elems, float lut_scale, int lut_offset,
  float out_scale, int out_offset, uint16_t *dst) {
  const float inv_out = 1.0f / out_scale;
  for (size_t i = 0; i < elems; ++i) {
    const float f = (static_cast<float>(src[i]) + lut_offset) * lut_scale;
    int q = static_cast<int>(std::lrintf(f * inv_out)) - out_offset;
    dst[i] = static_cast<uint16_t>(std::max(0, std::min(65535, q)));
  }
}

// =====================================================================
// PerLayerEmbeddingLayer (thin wrapper over the engine)
// =====================================================================
PerLayerEmbeddingLayer::PerLayerEmbeddingLayer() :
  nntrainer::LayerImpl(),
  ple_props(props::PleFilePath(), props::PleLayerCount(),
            props::PlePerLayerWidth(), props::PleModelIndices(),
            props::PleScales(), props::PleOffsets()) {}

void PerLayerEmbeddingLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "PerLayerEmbeddingLayer takes only one input";

  // Force the token-ID input to FP32 (see EmbeddingLayer::finalize): the layer
  // is often the model entry point, so the input dim would otherwise inherit
  // the activation dtype (e.g. UINT16), but token IDs index a vocab that can
  // exceed UINT16 range.
  context.setInputDataType(nntrainer::TensorDim::DataType::FP32);

  auto &file_prop = std::get<props::PleFilePath>(ple_props);
  NNTR_THROW_IF(file_prop.empty(), std::invalid_argument)
    << "PerLayerEmbeddingLayer: ple_file_path property not set";

  ple_per_layer_ =
    std::get<props::PlePerLayerWidth>(ple_props).empty()
      ? 256
      : static_cast<size_t>(std::get<props::PlePerLayerWidth>(ple_props).get());

  // Output count must be known before open() for the raw-uint16 layout.
  auto &count_prop = std::get<props::PleLayerCount>(ple_props);
  const bool has_count = !count_prop.empty();
  if (has_count)
    ple_output_count_ = static_cast<size_t>(count_prop.get());

  engine_.open(file_prop.get(), ple_per_layer_, ple_output_count_);

  // JSON manifests expose the full model layer count; default the output count
  // to it when not explicitly given.
  if (!has_count)
    ple_output_count_ = engine_.numLayers();

  NNTR_THROW_IF(ple_output_count_ == 0, std::invalid_argument)
    << "PerLayerEmbeddingLayer: output count is 0";
  NNTR_THROW_IF(ple_output_count_ > engine_.numLayers(), std::invalid_argument)
    << "PerLayerEmbeddingLayer: output count (" << ple_output_count_
    << ") exceeds available layers (" << engine_.numLayers() << ")";

  // Resolve per-output model-layer index (identity if unset).
  auto &idx_prop = std::get<props::PleModelIndices>(ple_props);
  if (idx_prop.empty()) {
    ple_model_index_.resize(ple_output_count_);
    for (size_t l = 0; l < ple_output_count_; ++l)
      ple_model_index_[l] = static_cast<int>(l);
  } else {
    ple_model_index_ = parse_csv<int>(
      idx_prop.get(), [](const std::string &t) { return std::stoi(t); });
    NNTR_THROW_IF(ple_model_index_.size() != ple_output_count_,
                  std::invalid_argument)
      << "PerLayerEmbeddingLayer: ple_model_indices count ("
      << ple_model_index_.size() << ") != output count (" << ple_output_count_
      << ")";
  }
  for (size_t l = 0; l < ple_output_count_; ++l)
    NNTR_THROW_IF(ple_model_index_[l] < 0 ||
                    static_cast<size_t>(ple_model_index_[l]) >=
                      engine_.numLayers(),
                  std::invalid_argument)
      << "PerLayerEmbeddingLayer: model index " << ple_model_index_[l]
      << " out of range [0," << engine_.numLayers() << ")";

  // Resolve per-output consumer-space quant scale/offset (1.0 / 0 if unset).
  auto &scale_prop = std::get<props::PleScales>(ple_props);
  if (scale_prop.empty()) {
    ple_consumer_scale_.assign(ple_output_count_, 1.0f);
  } else {
    ple_consumer_scale_ = parse_csv<float>(
      scale_prop.get(), [](const std::string &t) { return std::stof(t); });
    NNTR_THROW_IF(ple_consumer_scale_.size() != ple_output_count_,
                  std::invalid_argument)
      << "PerLayerEmbeddingLayer: ple_scales count ("
      << ple_consumer_scale_.size() << ") != output count ("
      << ple_output_count_ << ")";
  }

  auto &offset_prop = std::get<props::PleOffsets>(ple_props);
  if (offset_prop.empty()) {
    ple_consumer_offset_.assign(ple_output_count_, 0);
  } else {
    ple_consumer_offset_ = parse_csv<int>(
      offset_prop.get(), [](const std::string &t) { return std::stoi(t); });
    NNTR_THROW_IF(ple_consumer_offset_.size() != ple_output_count_,
                  std::invalid_argument)
      << "PerLayerEmbeddingLayer: ple_offsets count ("
      << ple_consumer_offset_.size() << ") != output count ("
      << ple_output_count_ << ")";
  }

  // Each output carries the per-layer embedding for the whole sequence:
  //   [batch, 1, seq_len, ple_per_layer_], UINT16.
  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  nntrainer::TensorDim out_dim = in_dim;
  out_dim.height(in_dim.width());
  out_dim.width(ple_per_layer_);
  out_dim.setTensorType(
    {context.getFormat(), nntrainer::TensorDim::DataType::UINT16});

  std::vector<nntrainer::VarGradSpecV2> out_specs;
  out_specs.reserve(ple_output_count_);
  for (size_t l = 0; l < ple_output_count_; ++l)
    out_specs.push_back(nntrainer::InitLayerContext::outSpec(out_dim, "out"));
  context.requestOutputs(std::move(out_specs));
}

void PerLayerEmbeddingLayer::forwarding(nntrainer::RunLayerContext &context,
                                        bool training) {
  // Mirror incremental_forwarding over the full sequence width.
  auto &input = context.getInput(SINGLE_INOUT_IDX);
  incremental_forwarding(context, 0, input.width(), training);
}

void PerLayerEmbeddingLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  if (!engine_.isOpen())
    return;

  auto &input = context.getInput(SINGLE_INOUT_IDX);

  const size_t per_layer_elems = ple_per_layer_;
  const unsigned int b_size = input.batch();
  const unsigned int in_feature_len = input.getDim().getFeatureLen();
  const int iter = static_cast<int>(to - from);
  if (iter <= 0)
    return;

  // Per-batch element stride into each output tensor (all outputs share dims).
  const size_t out_feature_len =
    context.getOutput(SINGLE_INOUT_IDX).getDim().getFeatureLen();

  // Cache per-output base pointers once per batch to avoid repeated lookups.
  std::vector<uint16_t *> out_base(ple_output_count_);

  for (unsigned int b = 0; b < b_size; ++b) {
    for (size_t l = 0; l < ple_output_count_; ++l)
      out_base[l] = context.getOutput(l).getData<uint16_t>(b * out_feature_len);

    // Token IDs are stored as FP32 (forced in finalize). Like EmbeddingLayer
    // and the other CausalLM step-layers, the per-step input/output are indexed
    // from 0 over [0, to-from) (NOT offset by `from`); `from` only sizes iter.
    const float *token_data = input.getAddress<float>(b * in_feature_len);

    for (int t = 0; t < iter; ++t) {
      const int token_id = static_cast<int>(token_data[t]);
      engine_.fillToken(token_id, ple_model_index_, ple_consumer_scale_,
                        ple_consumer_offset_, out_base,
                        static_cast<size_t>(t) * per_layer_elems);
    }
  }
}

void PerLayerEmbeddingLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "PerLayerEmbeddingLayer does not support backwarding");
}

void PerLayerEmbeddingLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "PerLayerEmbeddingLayer does not support backwarding");
}

void PerLayerEmbeddingLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = nntrainer::loadProperties(values, ple_props);
  LayerImpl::setProperty(remain_props);
}

#ifdef PLUGGABLE
nntrainer::Layer *create_per_layer_embedding_layer() {
  return new PerLayerEmbeddingLayer();
}
void destroy_per_layer_embedding_layer(nntrainer::Layer *layer) {
  delete layer;
}
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_per_layer_embedding_layer, destroy_per_layer_embedding_layer};
}
#endif

} // namespace causallm
