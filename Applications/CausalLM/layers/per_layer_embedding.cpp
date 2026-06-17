// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   per_layer_embedding.cpp
 * @date   17 June 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author haehun.yang <haehun.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Loads per-layer embeddings from a single file and distributes them
 *         across multiple output tensors (one per consumed model layer).
 */

#include "per_layer_embedding.h"
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <sstream>
#include <sys/stat.h>
#if defined(_WIN32)
#include <io.h>
#include <mman_windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

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
    // trim surrounding whitespace
    size_t b = item.find_first_not_of(" \t");
    if (b == std::string::npos)
      continue;
    size_t e = item.find_last_not_of(" \t");
    out.push_back(conv(item.substr(b, e - b + 1)));
  }
  return out;
}

} // namespace

PerLayerEmbeddingLayer::PerLayerEmbeddingLayer() :
  nntrainer::LayerImpl(),
  ple_props(props::PleFilePath(), props::PleLayerCount(),
            props::PlePerLayerWidth(), props::PleModelIndices(),
            props::PleScales(), props::PleOffsets()) {}

PerLayerEmbeddingLayer::~PerLayerEmbeddingLayer() { close_ple_file_(); }

PerLayerEmbeddingLayer::PerLayerEmbeddingLayer(
  PerLayerEmbeddingLayer &&rhs) noexcept :
  nntrainer::LayerImpl(std::move(rhs)),
  ple_props(std::move(rhs.ple_props)),
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
  ple_output_count_(rhs.ple_output_count_),
  ple_vocab_(rhs.ple_vocab_),
  ple_model_index_(std::move(rhs.ple_model_index_)),
  ple_consumer_scale_(std::move(rhs.ple_consumer_scale_)),
  ple_consumer_offset_(std::move(rhs.ple_consumer_offset_)),
  ple_scale_(rhs.ple_scale_),
  ple_offset_(rhs.ple_offset_),
  ple_row_layer_scales_(std::move(rhs.ple_row_layer_scales_)) {
  // Clear the moved-from owning handles so only this object frees them.
  rhs.ple_fd_ = -1;
  rhs.ple_mmap_ = nullptr;
  rhs.ple_u16_mmap_ = nullptr;
  rhs.ple_file_size_ = 0;
}

PerLayerEmbeddingLayer &
PerLayerEmbeddingLayer::operator=(PerLayerEmbeddingLayer &&rhs) noexcept {
  if (this != &rhs) {
    // Release our own mapping before taking ownership of rhs's.
    close_ple_file_();
    nntrainer::LayerImpl::operator=(std::move(rhs));
    ple_props = std::move(rhs.ple_props);
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
    ple_output_count_ = rhs.ple_output_count_;
    ple_vocab_ = rhs.ple_vocab_;
    ple_model_index_ = std::move(rhs.ple_model_index_);
    ple_consumer_scale_ = std::move(rhs.ple_consumer_scale_);
    ple_consumer_offset_ = std::move(rhs.ple_consumer_offset_);
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
  const std::string ple_file_name = file_prop.get();

  ple_per_layer_ =
    std::get<props::PlePerLayerWidth>(ple_props).empty()
      ? 256
      : static_cast<size_t>(std::get<props::PlePerLayerWidth>(ple_props).get());
  NNTR_THROW_IF(ple_per_layer_ == 0, std::invalid_argument)
    << "PerLayerEmbeddingLayer: ple_per_layer_width is 0";

  // Output count must be known before open_ple_file_ for the raw-uint16 path,
  // whose row layout is derived from it.
  auto &count_prop = std::get<props::PleLayerCount>(ple_props);
  const bool has_count = !count_prop.empty();
  if (has_count)
    ple_output_count_ = static_cast<size_t>(count_prop.get());

  open_ple_file_(ple_file_name);

  // JSON manifests expose the full model layer count; default the output count
  // to it when not explicitly given.
  if (!has_count) {
    NNTR_THROW_IF(!ple_is_quantized_, std::invalid_argument)
      << "PerLayerEmbeddingLayer: ple_layer_count is required for raw uint16";
    ple_output_count_ = ple_layers_;
  }

  NNTR_THROW_IF(ple_output_count_ == 0, std::invalid_argument)
    << "PerLayerEmbeddingLayer: output count is 0";
  NNTR_THROW_IF(ple_output_count_ > ple_layers_, std::invalid_argument)
    << "PerLayerEmbeddingLayer: output count (" << ple_output_count_
    << ") exceeds available layers (" << ple_layers_ << ")";

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
                    static_cast<size_t>(ple_model_index_[l]) >= ple_layers_,
                  std::invalid_argument)
      << "PerLayerEmbeddingLayer: model index " << ple_model_index_[l]
      << " out of range [0," << ple_layers_ << ")";

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
  if (!ple_mmap_ && !ple_u16_mmap_)
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
      const float tok_f = token_data[t];
      NNTR_THROW_IF(tok_f < 0.0f, std::invalid_argument)
        << "PerLayerEmbeddingLayer: negative token id " << tok_f;
      const size_t token_id = static_cast<size_t>(tok_f);
      NNTR_THROW_IF(token_id >= ple_vocab_, std::invalid_argument)
        << "PerLayerEmbeddingLayer: token id " << token_id << " >= vocab "
        << ple_vocab_;

      const size_t dst_off = static_cast<size_t>(t) * per_layer_elems;

      if (ple_is_quantized_) {
        const uint8_t *row = ple_mmap_ + token_id * ple_row_bytes_;

        if (ple_is_signed4_) {
          // sfixed4: two signed 4-bit nibbles per byte (half a byte per elem).
          const size_t per_layer_bytes = per_layer_elems / 2;
          const float *row_scales =
            ple_row_layer_scales_.data() + token_id * ple_layers_;
          for (size_t l = 0; l < ple_output_count_; ++l) {
            const size_t ml = static_cast<size_t>(ple_model_index_[l]);
            dequant_sfixed4_requant_u16_(
              row + ml * per_layer_bytes, per_layer_elems, row_scales[ml],
              ple_consumer_scale_[l], ple_consumer_offset_[l],
              out_base[l] + dst_off);
          }
        } else {
          // ufixed8: one unsigned byte per element.
          const size_t per_layer_bytes = per_layer_elems;
          for (size_t l = 0; l < ple_output_count_; ++l) {
            const size_t ml = static_cast<size_t>(ple_model_index_[l]);
            dequant_bytes_requant_u16_(
              row + ml * per_layer_bytes, per_layer_elems, ple_scale_,
              ple_offset_, ple_consumer_scale_[l], ple_consumer_offset_[l],
              out_base[l] + dst_off);
          }
        }
      } else {
        // raw uint16: per-layer slice memcpy (already in consumer space).
        const uint16_t *row = ple_u16_mmap_ + token_id * ple_row_elems_;
        for (size_t l = 0; l < ple_output_count_; ++l) {
          const size_t ml = static_cast<size_t>(ple_model_index_[l]);
          std::memcpy(out_base[l] + dst_off, row + ml * per_layer_elems,
                      per_layer_elems * sizeof(uint16_t));
        }
      }
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

void PerLayerEmbeddingLayer::open_ple_file_(const std::string &ple_file_name) {
  if (ple_file_name.empty())
    throw std::runtime_error("PerLayerEmbeddingLayer: ple_file_name is empty");

  ple_is_quantized_ = ends_with(ple_file_name, ".json");

  if (ple_is_quantized_) {
    std::ifstream mf(ple_file_name);
    if (!mf.is_open())
      throw std::runtime_error("Failed to open PLE manifest: " + ple_file_name);

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
    ple_row_bytes_ =
      ple_is_signed4_ ? (ple_row_elems_ + 1) / 2 : ple_row_elems_;
    ple_layers_ = ple_row_elems_ / ple_per_layer_;

    if (ple_layers_ * ple_per_layer_ != ple_row_elems_)
      throw std::runtime_error("PLE 'size' not divisible by per_layer width");

    if (ple_is_signed4_) {
      const auto &scale_arr = qp.at("scale");
      if (!scale_arr.is_array())
        throw std::runtime_error(
          "PLE sfixed4: quant-param.scale must be an array");
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

    std::string lut_abs = rebase_relative_to_model_file(lut_rel, ple_file_name);

    ple_fd_ = open(lut_abs.c_str(), O_RDONLY);
    if (ple_fd_ < 0)
      throw std::runtime_error("open PLE bin: " + lut_abs);

    struct stat st;
    if (fstat(ple_fd_, &st) < 0) {
      ::close(ple_fd_);
      ple_fd_ = -1;
      throw std::runtime_error("stat PLE bin: " + lut_abs);
    }
    ple_file_size_ = static_cast<size_t>(st.st_size);
    if (ple_row_bytes_ == 0 || ple_file_size_ == 0 ||
        ple_file_size_ % ple_row_bytes_ != 0) {
      ::close(ple_fd_);
      ple_fd_ = -1;
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
      ::close(ple_fd_);
      ple_fd_ = -1;
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
    } else {
      ml_logd("[PLE] ufixed8 (tensorwise) mmaped %s rows=%zu layers=%zu "
              "per_layer=%zu scale=%f offset=%d",
              lut_abs.c_str(), ple_vocab_, ple_layers_, ple_per_layer_,
              ple_scale_, ple_offset_);
    }
    return;
  }

  // raw UINT16: no manifest. Row layout is derived from the output count, which
  // the caller (finalize) must have set before reaching here.
  if (ple_output_count_ == 0)
    throw std::runtime_error(
      "PLE raw uint16: ple_layer_count must be set before loading");
  ple_row_elems_ = ple_output_count_ * ple_per_layer_;
  ple_row_bytes_ = ple_row_elems_ * sizeof(uint16_t);
  ple_layers_ = ple_output_count_;
  ple_is_signed4_ = false;
  ple_scale_ = 1.0f;
  ple_offset_ = 0;

  ple_fd_ = open(ple_file_name.c_str(), O_RDONLY);
  if (ple_fd_ < 0)
    throw std::runtime_error("open PLE bin: " + ple_file_name);

  struct stat st;
  if (fstat(ple_fd_, &st) < 0) {
    ::close(ple_fd_);
    ple_fd_ = -1;
    throw std::runtime_error("stat PLE bin: " + ple_file_name);
  }
  ple_file_size_ = static_cast<size_t>(st.st_size);
  if (ple_row_bytes_ == 0 || ple_file_size_ == 0 ||
      ple_file_size_ % ple_row_bytes_ != 0) {
    ::close(ple_fd_);
    ple_fd_ = -1;
    throw std::runtime_error(
      "PLE bin size not a positive multiple of row bytes (expected multiple "
      "of " +
      std::to_string(ple_row_bytes_) + ", got " +
      std::to_string(ple_file_size_) + ")");
  }
  ple_vocab_ = ple_file_size_ / ple_row_bytes_;

  void *m = mmap(nullptr, ple_file_size_, PROT_READ, MAP_PRIVATE, ple_fd_, 0);
  if (m == MAP_FAILED) {
    ::close(ple_fd_);
    ple_fd_ = -1;
    throw std::runtime_error("mmap PLE bin: " + ple_file_name);
  }
  ple_u16_mmap_ = static_cast<const uint16_t *>(m);
#ifdef POSIX_MADV_RANDOM
  posix_madvise((void *)ple_u16_mmap_, ple_file_size_, POSIX_MADV_RANDOM);
#endif

  ml_logd("[PLE] raw uint16 mmaped %s rows=%zu layers=%zu per_layer=%zu",
          ple_file_name.c_str(), ple_vocab_, ple_layers_, ple_per_layer_);
}

void PerLayerEmbeddingLayer::close_ple_file_() {
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

void PerLayerEmbeddingLayer::dequant_sfixed4_requant_u16_(
  const uint8_t *packed, size_t elems, float row_scale, float out_scale,
  int out_offset, uint16_t *dst) const {
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

void PerLayerEmbeddingLayer::dequant_bytes_requant_u16_(
  const uint8_t *src, size_t elems, float lut_scale, int lut_offset,
  float out_scale, int out_offset, uint16_t *dst) const {
  const float inv_out = 1.0f / out_scale;
  for (size_t i = 0; i < elems; ++i) {
    const float f = (static_cast<float>(src[i]) + lut_offset) * lut_scale;
    int q = static_cast<int>(std::lrintf(f * inv_out)) - out_offset;
    dst[i] = static_cast<uint16_t>(std::max(0, std::min(65535, q)));
  }
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
