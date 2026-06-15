// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding.cpp
 * @date   04 March 2021
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This embedding layer supports FP32/FP16/Q6_K data type only.
 */

#include <embedding_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "json.hpp"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum EmbeddingParams { weight };

namespace {

// Path-keyed cache so two graphs (or two layers) that reference the same
// manifest share a single in-memory copy of the 4-bit LUT.
std::mutex g_lut_cache_mtx;
std::unordered_map<std::string, std::weak_ptr<QuantLut>> g_lut_cache;

std::string dirname(const std::string &p) {
  auto pos = p.find_last_of('/');
  return (pos == std::string::npos) ? std::string() : p.substr(0, pos);
}

std::string resolve_relative(const std::string &path,
                             const std::string &base_dir) {
  if (path.empty() || path[0] == '/' || base_dir.empty())
    return path;
  return base_dir + "/" + path;
}

inline uint16_t clamp_u16(float v) {
  return static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, v)));
}

// Sign-extend a 4-bit value (low or high nibble) to int.
//   nib in [0,7]   → 0..7
//   nib in [8,15]  → -8..-1
inline int s4(unsigned nib) {
  return (nib & 0x8u) ? static_cast<int>(nib) - 16 : static_cast<int>(nib);
}

bool ends_with(const std::string &s, const std::string &suf) {
  return s.size() >= suf.size() &&
         0 == s.compare(s.size() - suf.size(), suf.size(), suf);
}

std::shared_ptr<QuantLut>
load_ufixed8_manifest_(const std::string &manifest_path,
                       const nlohmann::json &j) {
  // Tensorwise unsigned 4-bit (legacy). One (scale, offset) for the
  // whole table.
  const std::string lut_rel = j.at("lut-path").get<std::string>();
  const int per_row = j.at("size").get<int>();
  const auto &qp = j.at("quant-param");

  auto lut = std::make_shared<QuantLut>();
  lut->is_raw_u16 = false;
  lut->is_signed4 = false;
  lut->scale = qp.at("scale").get<float>();
  lut->offset = qp.at("offset").get<int>();
  lut->out_dim = static_cast<size_t>(per_row);

  const std::string lut_abs = resolve_relative(lut_rel, dirname(manifest_path));

  std::ifstream bin(lut_abs, std::ios::binary | std::ios::ate);
  NNTR_THROW_IF(!bin.is_open(), std::runtime_error)
    << "Failed to open LUT binary: " << lut_abs;
  const std::streamsize sz = bin.tellg();
  bin.seekg(0, std::ios::beg);
  lut->bytes.resize(static_cast<size_t>(sz));
  bin.read(reinterpret_cast<char *>(lut->bytes.data()), sz);

  NNTR_THROW_IF(lut->out_dim == 0 || (2 * lut->bytes.size()) % lut->out_dim,
                std::runtime_error)
    << "LUT binary size " << lut->bytes.size()
    << " is not consistent with out_dim=" << lut->out_dim;
  lut->in_dim = (2 * lut->bytes.size()) / lut->out_dim;

  ml_logi("Loaded ufixed8 (tensorwise) LUT '%s' (in_dim=%zu, out_dim=%zu, "
          "scale=%f, offset=%d, bytes=%zu)",
          manifest_path.c_str(), lut->in_dim, lut->out_dim, lut->scale,
          lut->offset, lut->bytes.size());
  return lut;
}

std::shared_ptr<QuantLut>
load_sfixed4_manifest_(const std::string &manifest_path,
                       const nlohmann::json &j) {
  // Per-row signed 4-bit (-8..7), 2 packed per byte, no offset.
  // quant-param.scale is an array of length == in_dim (vocab size).
  const std::string lut_rel = j.at("lut-path").get<std::string>();
  const int per_row = j.at("size").get<int>();
  const auto &qp = j.at("quant-param");

  auto lut = std::make_shared<QuantLut>();
  lut->is_raw_u16 = false;
  lut->is_signed4 = true;
  lut->offset = 0;
  lut->out_dim = static_cast<size_t>(per_row);

  // scale array, one entry per vocab row.
  const auto &scale_arr = qp.at("scale");
  NNTR_THROW_IF(!scale_arr.is_array(), std::runtime_error)
    << "sfixed4 manifest expects quant-param.scale as an array";
  lut->row_scales.reserve(scale_arr.size());
  for (const auto &v : scale_arr)
    lut->row_scales.push_back(v.get<float>());

  const std::string lut_abs = resolve_relative(lut_rel, dirname(manifest_path));

  std::ifstream bin(lut_abs, std::ios::binary | std::ios::ate);
  NNTR_THROW_IF(!bin.is_open(), std::runtime_error)
    << "Failed to open LUT binary: " << lut_abs;
  const std::streamsize sz = bin.tellg();
  bin.seekg(0, std::ios::beg);
  lut->bytes.resize(static_cast<size_t>(sz));
  bin.read(reinterpret_cast<char *>(lut->bytes.data()), sz);

  NNTR_THROW_IF(lut->out_dim == 0 || (2 * lut->bytes.size()) % lut->out_dim,
                std::runtime_error)
    << "LUT binary size " << lut->bytes.size()
    << " is not consistent with out_dim=" << lut->out_dim;
  lut->in_dim = (2 * lut->bytes.size()) / lut->out_dim;

  NNTR_THROW_IF(lut->row_scales.size() != lut->in_dim, std::invalid_argument)
    << "sfixed4 row_scales.size=" << lut->row_scales.size()
    << " != in_dim=" << lut->in_dim;

  ml_logi("Loaded sfixed4 (rowwise) LUT '%s' (in_dim=%zu, out_dim=%zu, "
          "row_scales=%zu, bytes=%zu)",
          manifest_path.c_str(), lut->in_dim, lut->out_dim,
          lut->row_scales.size(), lut->bytes.size());
  return lut;
}

std::shared_ptr<QuantLut>
load_4bit_manifest_(const std::string &manifest_path) {
  // Dispatch by `datatype`: "ufixed8" = legacy tensorwise unsigned,
  // "sfixed4" = rowwise signed (per-row scale, no offset).
  std::ifstream f(manifest_path);
  NNTR_THROW_IF(!f.is_open(), std::runtime_error)
    << "Failed to open LUT manifest: " << manifest_path;

  nlohmann::json j;
  f >> j;

  const std::string datatype = j.value("datatype", std::string("ufixed8"));

  if (datatype == "sfixed4")
    return load_sfixed4_manifest_(manifest_path, j);
  if (datatype == "ufixed8")
    return load_ufixed8_manifest_(manifest_path, j);

  NNTR_THROW_IF(true, std::runtime_error)
    << "Unsupported LUT datatype: " << datatype
    << " (expected 'ufixed8' or 'sfixed4')";
  return nullptr; // unreachable
}

std::shared_ptr<QuantLut> load_raw_u16_(const std::string &bin_path,
                                        size_t in_dim, size_t out_dim) {
  NNTR_THROW_IF(in_dim == 0 || out_dim == 0, std::invalid_argument)
    << "Raw UINT16 embedding requires in_dim/out_dim hints from layer";

  std::ifstream bin(bin_path, std::ios::binary | std::ios::ate);
  NNTR_THROW_IF(!bin.is_open(), std::runtime_error)
    << "Failed to open raw UINT16 embedding: " << bin_path;
  const std::streamsize sz = bin.tellg();
  bin.seekg(0, std::ios::beg);

  const size_t expected = in_dim * out_dim * sizeof(uint16_t);
  NNTR_THROW_IF(static_cast<size_t>(sz) != expected, std::runtime_error)
    << "Raw UINT16 file size " << sz << " != in_dim*out_dim*2 = " << expected;

  auto lut = std::make_shared<QuantLut>();
  lut->is_raw_u16 = true;
  lut->in_dim = in_dim;
  lut->out_dim = out_dim;
  lut->bytes.resize(static_cast<size_t>(sz));
  bin.read(reinterpret_cast<char *>(lut->bytes.data()), sz);

  ml_logi("Loaded shared raw UINT16 embedding '%s' (in_dim=%zu, out_dim=%zu, "
          "bytes=%zu)",
          bin_path.c_str(), in_dim, out_dim, lut->bytes.size());
  return lut;
}
} // namespace

std::shared_ptr<QuantLut> get_or_load_quant_lut(const std::string &path,
                                                size_t in_dim_hint,
                                                size_t out_dim_hint) {
  std::lock_guard<std::mutex> lk(g_lut_cache_mtx);

  auto it = g_lut_cache.find(path);
  if (it != g_lut_cache.end()) {
    if (auto sp = it->second.lock())
      return sp;
    g_lut_cache.erase(it);
  }

  // Auto-detect mode by extension: `.json` → 4-bit manifest, otherwise
  // assume a raw UINT16 binary (consumer-space, no requant needed).
  std::shared_ptr<QuantLut> lut =
    ends_with(path, ".json") ? load_4bit_manifest_(path)
                             : load_raw_u16_(path, in_dim_hint, out_dim_hint);

  g_lut_cache[path] = lut;
  return lut;
}

EmbeddingLayer::EmbeddingLayer() :
  LayerImpl(),
  embedding_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
                  nntrainer::props::Scale(), props::QuantizedLutPath(),
                  props::OutputQuantScale(), props::OutputQuantOffset()),
  weight_idx(std::numeric_limits<unsigned>::max()) {}

void EmbeddingLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Embedding layer takes only one input";

  // Token IDs are integers — embedding caller is expected to provide FP32
  // input (e.g., via an explicit input layer with input_dtype=FP32). The
  // historical "must be FP32" check is removed so FP16-activation models
  // still construct, but the actual lookup expects integer-valued data.

  const nntrainer::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "Embedding layer takes only one for channel size";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  size_t in_dim =
    static_cast<size_t>(std::get<nntrainer::props::InDim>(embedding_props));
  size_t out_dim =
    static_cast<size_t>(std::get<nntrainer::props::OutDim>(embedding_props));

  // Tensorwise 4-bit LUT mode: load (or look up cached) shared LUT and
  // skip the standard managed weight allocation. The LUT is owned by
  // this layer via shared_ptr, and shared with any other layer that
  // references the same manifest path.
  auto &quant_path_prop = std::get<props::QuantizedLutPath>(embedding_props);
  if (!quant_path_prop.empty()) {
    // Hints are only consulted in raw-uint16 mode (no manifest); the
    // 4-bit path derives in_dim/out_dim from manifest+file size.
    quant_lut_ = get_or_load_quant_lut(quant_path_prop.get(), in_dim, out_dim);

    NNTR_THROW_IF(quant_lut_->in_dim != in_dim, std::invalid_argument)
      << "in_dim mismatch: layer=" << in_dim << " file=" << quant_lut_->in_dim;
    NNTR_THROW_IF(quant_lut_->out_dim != out_dim, std::invalid_argument)
      << "out_dim mismatch: layer=" << out_dim
      << " file=" << quant_lut_->out_dim;
  }

  nntrainer::TensorDim output_dim = input_dim;

  // output_dim expected as hidden x num input (batch size)
  output_dim.height(input_dim.width());
  output_dim.width(out_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  if (quant_lut_) {
    // No managed weight in LUT mode — embedding rows live in quant_lut_.
    return;
  }

  nntrainer::TensorDim dim = output_dim;

  dim.setTensorType({context.getFormat(), context.getWeightDataType()});

  dim.height(in_dim);
  dim.width(out_dim);
  dim.batch(1);

  weight_idx = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void EmbeddingLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, embedding_props);
  LayerImpl::setProperty(remain_props);
}

void EmbeddingLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  /// Mirror incremental_forwarding for the full input width (no from/to).
  unsigned int in_dim = std::get<nntrainer::props::InDim>(embedding_props);
  unsigned int out_dim = std::get<nntrainer::props::OutDim>(embedding_props);
  float scale = std::get<nntrainer::props::Scale>(embedding_props).empty()
                  ? 1.0f
                  : std::get<nntrainer::props::Scale>(embedding_props).get();

  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  unsigned int b_size = input_.batch();
  const unsigned int seq_len = input_.width();

  // -------------------------------------------------------------------
  // Tensorwise 4-bit LUT path: dequant 4-bit → float (LUT space) →
  // uint16 (consumer space) per token, mirroring incremental_forwarding.
  // -------------------------------------------------------------------
  if (quant_lut_) {
    NNTR_THROW_IF(out_dim != quant_lut_->out_dim, std::runtime_error)
      << "LUT out_dim drift";

    const auto out_dtype = hidden_.getDataType();

    // Raw UINT16 path: file is already in consumer space, so a per-token
    // memcpy is enough — no dequant, no requant, no scale/offset.
    if (quant_lut_->is_raw_u16) {
      NNTR_THROW_IF(out_dtype != nntrainer::TensorDim::DataType::UINT16,
                    std::runtime_error)
        << "Raw UINT16 embedding requires UINT16 output dtype";

      const uint16_t *table =
        reinterpret_cast<const uint16_t *>(quant_lut_->bytes.data());

      for (unsigned int b = 0; b < b_size; ++b) {
        const float *in_data =
          input_.getAddress<float>(b * input_.getDim().getFeatureLen());
        nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < (int)seq_len; ++i) {
          const size_t embed_idx = static_cast<size_t>(in_data[i]);
          if (embed_idx >= in_dim) {
            throw std::invalid_argument(
              "input word index is greater than in_dim");
          }
          uint16_t *dst = batchsliced_hidden.getData<uint16_t>() +
                          static_cast<size_t>(out_dim) * i;
          std::memcpy(dst, table + embed_idx * out_dim,
                      out_dim * sizeof(uint16_t));
        }
        // ── Debug: dump first 4 elems for first 32 tokens (raw u16) ──
        static bool dbg_u16_done = false;
        if (!dbg_u16_done && b == 0) {
          dbg_u16_done = true;
          uint16_t *p = batchsliced_hidden.getData<uint16_t>();
          const int n_dump = std::min<int>(32, (int)seq_len);
          for (int i = 0; i < n_dump; ++i) {
            const size_t tid = static_cast<size_t>(in_data[i]);
            std::cout << "[EMB-U16-DBG] pos=" << i << " tok=" << tid
                      << " emb[0..3]=";
            for (int k = 0; k < 4; ++k)
              std::cout << p[i * out_dim + k] << " ";
            std::cout << "\n";
          }
        }
      }
      return;
    }

    NNTR_THROW_IF(out_dim % 2 != 0, std::runtime_error)
      << "4-bit packed embedding requires out_dim to be even, got " << out_dim;

    const uint8_t *packed = quant_lut_->bytes.data();
    const size_t bytes_per_row = out_dim / 2;

    auto &out_scale_prop = std::get<props::OutputQuantScale>(embedding_props);
    auto &out_offset_prop = std::get<props::OutputQuantOffset>(embedding_props);
    const bool has_out_quant = !out_scale_prop.empty();
    const float out_scale = has_out_quant ? out_scale_prop.get() : 1.0f;
    const int out_offset =
      (!out_offset_prop.empty()) ? out_offset_prop.get() : 0;
    const float inv_out_scale = has_out_quant ? (1.0f / out_scale) : 1.0f;

    // ─── Per-row signed-4-bit (sfixed4) path ──────────────────────────
    // f = s4(nib) * row_scales[token_id] * scale (props::Scale modifier);
    // q16 = round(f / out_scale) - out_offset.
    if (quant_lut_->is_signed4 && !quant_lut_->row_scales.empty()) {
      const float *row_scales = quant_lut_->row_scales.data();

      for (unsigned int b = 0; b < b_size; ++b) {
        const float *in_data =
          input_.getAddress<float>(b * input_.getDim().getFeatureLen());
        nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < (int)seq_len; ++i) {
          const size_t embed_idx = static_cast<size_t>(in_data[i]);
          if (embed_idx >= in_dim) {
            throw std::invalid_argument(
              "input word index is greater than in_dim");
          }
          const uint8_t *row = packed + bytes_per_row * embed_idx;
          const float row_scale = row_scales[embed_idx] * scale;
          const size_t out_off = static_cast<size_t>(out_dim) * i;

          if (out_dtype == nntrainer::TensorDim::DataType::UINT16) {
            uint16_t *dst = batchsliced_hidden.getData<uint16_t>() + out_off;
            if (has_out_quant) {
              for (size_t k = 0; k < bytes_per_row; ++k) {
                const uint8_t byte = row[k];
                const float f_lo = s4(byte & 0x0F) * row_scale;
                const float f_hi = s4((byte >> 4) & 0x0F) * row_scale;
                const int q_lo =
                  static_cast<int>(std::lrintf(f_lo * inv_out_scale)) -
                  out_offset;
                const int q_hi =
                  static_cast<int>(std::lrintf(f_hi * inv_out_scale)) -
                  out_offset;
                dst[2 * k] =
                  static_cast<uint16_t>(std::max(0, std::min(65535, q_lo)));
                dst[2 * k + 1] =
                  static_cast<uint16_t>(std::max(0, std::min(65535, q_hi)));
              }
            } else {
              for (size_t k = 0; k < bytes_per_row; ++k) {
                const uint8_t byte = row[k];
                dst[2 * k] = clamp_u16(s4(byte & 0x0F) * row_scale);
                dst[2 * k + 1] = clamp_u16(s4((byte >> 4) & 0x0F) * row_scale);
              }
            }
          } else if (out_dtype == nntrainer::TensorDim::DataType::FP32) {
            float *dst = batchsliced_hidden.getData<float>() + out_off;
            for (size_t k = 0; k < bytes_per_row; ++k) {
              const uint8_t byte = row[k];
              dst[2 * k] = s4(byte & 0x0F) * row_scale;
              dst[2 * k + 1] = s4((byte >> 4) & 0x0F) * row_scale;
            }
#ifdef ENABLE_FP16
          } else if (out_dtype == nntrainer::TensorDim::DataType::FP16) {
            _FP16 *dst = batchsliced_hidden.getData<_FP16>() + out_off;
            for (size_t k = 0; k < bytes_per_row; ++k) {
              const uint8_t byte = row[k];
              dst[2 * k] = static_cast<_FP16>(s4(byte & 0x0F) * row_scale);
              dst[2 * k + 1] =
                static_cast<_FP16>(s4((byte >> 4) & 0x0F) * row_scale);
            }
#endif
          } else {
            throw std::runtime_error(
              "EmbeddingLayer sfixed4 mode: unsupported output dtype");
          }
        }
        // ── Debug: dump first 32 tokens' embedding (sfixed4) ──
        static bool dbg_s4 = false;
        if (!dbg_s4 && b == 0) {
          dbg_s4 = true;
          uint16_t *p = batchsliced_hidden.getData<uint16_t>();
          const int n_dump = std::min<int>(32, (int)seq_len);
          for (int i = 0; i < n_dump; ++i) {
            const size_t tid = static_cast<size_t>(in_data[i]);
            std::cout << "[EMB-S4-DBG] pos=" << i << " tok=" << tid
                      << " emb[0..3]=";
            for (int k = 0; k < 4; ++k)
              std::cout << p[i * out_dim + k] << " ";
            std::cout << "\n";
          }
        }
      }
      return;
    }

    // ─── Tensorwise unsigned-4-bit (ufixed8 legacy) path ──────────────
    const float lut_scale = quant_lut_->scale * scale;
    const int lut_offset = quant_lut_->offset;

    for (unsigned int b = 0; b < b_size; ++b) {
      const float *in_data =
        input_.getAddress<float>(b * input_.getDim().getFeatureLen());
      nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < (int)seq_len; ++i) {
        const size_t embed_idx = static_cast<size_t>(in_data[i]);
        if (embed_idx >= in_dim) {
          throw std::invalid_argument(
            "input word index is greater than in_dim");
        }

        const uint8_t *row = packed + bytes_per_row * embed_idx;
        const size_t out_off = static_cast<size_t>(out_dim) * i;

        if (out_dtype == nntrainer::TensorDim::DataType::UINT16) {
          uint16_t *dst = batchsliced_hidden.getData<uint16_t>() + out_off;
          if (has_out_quant) {
            for (size_t k = 0; k < bytes_per_row; ++k) {
              const uint8_t byte = row[k];
              const float f_lo =
                (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale;
              const float f_hi =
                (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) *
                lut_scale;
              const int q_lo =
                static_cast<int>(std::lrintf(f_lo * inv_out_scale)) -
                out_offset;
              const int q_hi =
                static_cast<int>(std::lrintf(f_hi * inv_out_scale)) -
                out_offset;
              dst[2 * k] =
                static_cast<uint16_t>(std::max(0, std::min(65535, q_lo)));
              dst[2 * k + 1] =
                static_cast<uint16_t>(std::max(0, std::min(65535, q_hi)));
            }
          } else {
            for (size_t k = 0; k < bytes_per_row; ++k) {
              const uint8_t byte = row[k];
              dst[2 * k] = clamp_u16(
                (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale);
              dst[2 * k + 1] = clamp_u16(
                (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) *
                lut_scale);
            }
          }
        } else if (out_dtype == nntrainer::TensorDim::DataType::FP32) {
          float *dst = batchsliced_hidden.getData<float>() + out_off;
          for (size_t k = 0; k < bytes_per_row; ++k) {
            const uint8_t byte = row[k];
            dst[2 * k] =
              (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale;
            dst[2 * k + 1] =
              (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) * lut_scale;
          }
#ifdef ENABLE_FP16
        } else if (out_dtype == nntrainer::TensorDim::DataType::FP16) {
          _FP16 *dst = batchsliced_hidden.getData<_FP16>() + out_off;
          for (size_t k = 0; k < bytes_per_row; ++k) {
            const uint8_t byte = row[k];
            dst[2 * k] = static_cast<_FP16>(
              (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale);
            dst[2 * k + 1] = static_cast<_FP16>(
              (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) *
              lut_scale);
          }
#endif
        } else {
          throw std::runtime_error(
            "EmbeddingLayer LUT mode: unsupported output dtype");
        }
      }
      static bool dbg_first = false;
      if (!dbg_first && b == 0) {
        dbg_first = true;
        uint16_t *p = batchsliced_hidden.getData<uint16_t>();
        // Dump first 32 tokens' embedding so user can compare per-token
        // values across ufixed8 vs uint16 modes (look for divergence on
        // specific token IDs like the ones for "/4" digits).
        const int n_dump = std::min<int>(32, (int)seq_len);
        for (int i = 0; i < n_dump; ++i) {
          const size_t tid = static_cast<size_t>(in_data[i]);
          std::cout << "[EMB-U8-DBG] pos=" << i << " tok=" << tid
                    << " emb[0..3]=";
          for (int k = 0; k < 4; ++k)
            std::cout << p[i * out_dim + k] << " ";
          std::cout << "\n";
        }
      }
    }
    return;
  }

  // -------------------------------------------------------------------
  // Non-LUT path (FP32/FP16 + Q4_0/Q6_K block dequant) mirroring
  // incremental_forwarding's loop body but over the full sequence.
  // -------------------------------------------------------------------
  nntrainer::Tensor &weight = context.getWeight(weight_idx);

  nntrainer::TensorDim out_tensor_dim =
    nntrainer::TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

  for (unsigned int b = 0; b < b_size; ++b) {
    float *in_data =
      input_.getAddress<float>(b * input_.getDim().getFeatureLen());
    nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < (int)seq_len; ++i) {
      size_t embed_idx = static_cast<size_t>(in_data[i]);
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      nntrainer::Tensor cur_weight =
        weight.getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);
      nntrainer::Tensor out_tensor =
        batchsliced_hidden.getSharedDataTensor(out_tensor_dim, out_dim * i);

      if (weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K) {
        int num_blocks_per_row = (weight.width() + 256 - 1) / 256;
        nntrainer::dequantize_row_q6_K(
          (void *)((char *)weight.getData<uint8_t>() +
                   (210 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      } else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
        int num_blocks_per_row = (weight.width() + 32 - 1) / 32;
        nntrainer::dequantize_row_q4_0(
          (void *)((char *)weight.getData<uint8_t>() +
                   (18 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      } else {
        out_tensor.copyData(cur_weight);
      }

      if (scale != 1.0f) {
        out_tensor.multiply_i(scale);
      }
    }
  }
}

void EmbeddingLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {

  /// @todo get input and output dimension from input_ and hidden itself
  unsigned int in_dim = std::get<nntrainer::props::InDim>(embedding_props);
  unsigned int out_dim = std::get<nntrainer::props::OutDim>(embedding_props);
  float scale = std::get<nntrainer::props::Scale>(embedding_props).empty()
                  ? 1.0f
                  : std::get<nntrainer::props::Scale>(embedding_props).get();

  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  unsigned int b_size = input_.batch();

  // -------------------------------------------------------------------
  // Tensorwise 4-bit LUT path: dequantize the embedding row directly into
  // the output (UINT16 or float). Bypasses the managed-weight tensor
  // entirely; the packed table is shared with peer graphs via the
  // QuantLut shared_ptr.
  // -------------------------------------------------------------------
  if (quant_lut_) {
    NNTR_THROW_IF(out_dim != quant_lut_->out_dim, std::runtime_error)
      << "LUT out_dim drift";

    const auto out_dtype = hidden_.getDataType();

    // Raw UINT16 path: per-token memcpy.
    if (quant_lut_->is_raw_u16) {
      NNTR_THROW_IF(out_dtype != nntrainer::TensorDim::DataType::UINT16,
                    std::runtime_error)
        << "Raw UINT16 embedding requires UINT16 output dtype";

      const uint16_t *table =
        reinterpret_cast<const uint16_t *>(quant_lut_->bytes.data());

      for (unsigned int b = 0; b < b_size; ++b) {
        const float *in_data =
          input_.getAddress<float>(b * input_.getDim().getFeatureLen());
        nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);
        const int iter = static_cast<int>(to - from);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < iter; ++i) {
          const size_t embed_idx = static_cast<size_t>(in_data[i]);
          if (embed_idx >= in_dim) {
            throw std::invalid_argument(
              "input word index is greater than in_dim");
          }
          uint16_t *dst = batchsliced_hidden.getData<uint16_t>() +
                          static_cast<size_t>(out_dim) * i;
          std::memcpy(dst, table + embed_idx * out_dim,
                      out_dim * sizeof(uint16_t));
        }
      }
      return;
    }

    NNTR_THROW_IF(out_dim % 2 != 0, std::runtime_error)
      << "4-bit packed embedding requires out_dim to be even, got " << out_dim;

    const uint8_t *packed = quant_lut_->bytes.data();
    const size_t bytes_per_row = out_dim / 2;

    // Two-step requant for UINT16 output:
    //   f      = <decode>(q4bit) * <scale>
    //   q16bit = round(f / out_scale - out_offset)  ← QNN convention
    // When the consumer's quant params are missing we fall back to a
    // naive clamp (only valid if LUT and consumer share quant space).
    auto &out_scale_prop = std::get<props::OutputQuantScale>(embedding_props);
    auto &out_offset_prop = std::get<props::OutputQuantOffset>(embedding_props);
    const bool has_out_quant = !out_scale_prop.empty();
    const float out_scale = has_out_quant ? out_scale_prop.get() : 1.0f;
    const int out_offset =
      (!out_offset_prop.empty()) ? out_offset_prop.get() : 0;
    const float inv_out_scale = has_out_quant ? (1.0f / out_scale) : 1.0f;

    // ─── Per-row signed-4-bit (sfixed4) path ──────────────────────────
    if (quant_lut_->is_signed4 && !quant_lut_->row_scales.empty()) {
      const float *row_scales = quant_lut_->row_scales.data();

      for (unsigned int b = 0; b < b_size; ++b) {
        const float *in_data =
          input_.getAddress<float>(b * input_.getDim().getFeatureLen());
        nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);
        const int iter = static_cast<int>(to - from);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < iter; ++i) {
          const size_t embed_idx = static_cast<size_t>(in_data[i]);
          if (embed_idx >= in_dim) {
            throw std::invalid_argument(
              "input word index is greater than in_dim");
          }
          const uint8_t *row = packed + bytes_per_row * embed_idx;
          const float row_scale = row_scales[embed_idx] * scale;
          const size_t out_off = static_cast<size_t>(out_dim) * i;

          if (out_dtype == nntrainer::TensorDim::DataType::UINT16) {
            uint16_t *dst = batchsliced_hidden.getData<uint16_t>() + out_off;
            if (has_out_quant) {
              for (size_t k = 0; k < bytes_per_row; ++k) {
                const uint8_t byte = row[k];
                const float f_lo = s4(byte & 0x0F) * row_scale;
                const float f_hi = s4((byte >> 4) & 0x0F) * row_scale;
                const int q_lo =
                  static_cast<int>(std::lrintf(f_lo * inv_out_scale)) -
                  out_offset;
                const int q_hi =
                  static_cast<int>(std::lrintf(f_hi * inv_out_scale)) -
                  out_offset;
                dst[2 * k] =
                  static_cast<uint16_t>(std::max(0, std::min(65535, q_lo)));
                dst[2 * k + 1] =
                  static_cast<uint16_t>(std::max(0, std::min(65535, q_hi)));
              }
            } else {
              for (size_t k = 0; k < bytes_per_row; ++k) {
                const uint8_t byte = row[k];
                dst[2 * k] = clamp_u16(s4(byte & 0x0F) * row_scale);
                dst[2 * k + 1] = clamp_u16(s4((byte >> 4) & 0x0F) * row_scale);
              }
            }
          } else if (out_dtype == nntrainer::TensorDim::DataType::FP32) {
            float *dst = batchsliced_hidden.getData<float>() + out_off;
            for (size_t k = 0; k < bytes_per_row; ++k) {
              const uint8_t byte = row[k];
              dst[2 * k] = s4(byte & 0x0F) * row_scale;
              dst[2 * k + 1] = s4((byte >> 4) & 0x0F) * row_scale;
            }
#ifdef ENABLE_FP16
          } else if (out_dtype == nntrainer::TensorDim::DataType::FP16) {
            _FP16 *dst = batchsliced_hidden.getData<_FP16>() + out_off;
            for (size_t k = 0; k < bytes_per_row; ++k) {
              const uint8_t byte = row[k];
              dst[2 * k] = static_cast<_FP16>(s4(byte & 0x0F) * row_scale);
              dst[2 * k + 1] =
                static_cast<_FP16>(s4((byte >> 4) & 0x0F) * row_scale);
            }
#endif
          } else {
            throw std::runtime_error(
              "EmbeddingLayer sfixed4 mode: unsupported output dtype");
          }
        }
      }
      return;
    }

    // ─── Tensorwise unsigned-4-bit (ufixed8 legacy) path ──────────────
    const float lut_scale = quant_lut_->scale * scale;
    const int lut_offset = quant_lut_->offset;

    // Token IDs are FP32 (forced by setInputDataType in finalize).
    for (unsigned int b = 0; b < b_size; ++b) {
      const float *in_data =
        input_.getAddress<float>(b * input_.getDim().getFeatureLen());
      nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

      const int iter = static_cast<int>(to - from);

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < iter; ++i) {
        const size_t embed_idx = static_cast<size_t>(in_data[i]);
        if (embed_idx >= in_dim) {
          throw std::invalid_argument(
            "input word index is greater than in_dim");
        }

        const uint8_t *row = packed + bytes_per_row * embed_idx;
        const size_t out_off = static_cast<size_t>(out_dim) * i;

        if (out_dtype == nntrainer::TensorDim::DataType::UINT16) {
          uint16_t *dst = batchsliced_hidden.getData<uint16_t>() + out_off;
          if (has_out_quant) {
            for (size_t k = 0; k < bytes_per_row; ++k) {
              const uint8_t byte = row[k];
              const float f_lo =
                (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale;
              const float f_hi =
                (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) *
                lut_scale;
              const int q_lo =
                static_cast<int>(std::lrintf(f_lo * inv_out_scale)) -
                out_offset;
              const int q_hi =
                static_cast<int>(std::lrintf(f_hi * inv_out_scale)) -
                out_offset;
              dst[2 * k] =
                static_cast<uint16_t>(std::max(0, std::min(65535, q_lo)));
              dst[2 * k + 1] =
                static_cast<uint16_t>(std::max(0, std::min(65535, q_hi)));
            }
          } else {
            // Naive clamp (legacy / same-quant-space).
            for (size_t k = 0; k < bytes_per_row; ++k) {
              const uint8_t byte = row[k];
              dst[2 * k] = clamp_u16(
                (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale);
              dst[2 * k + 1] = clamp_u16(
                (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) *
                lut_scale);
            }
          }
        } else if (out_dtype == nntrainer::TensorDim::DataType::FP32) {
          float *dst = batchsliced_hidden.getData<float>() + out_off;
          for (size_t k = 0; k < bytes_per_row; ++k) {
            const uint8_t byte = row[k];
            dst[2 * k] =
              (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale;
            dst[2 * k + 1] =
              (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) * lut_scale;
          }
#ifdef ENABLE_FP16
        } else if (out_dtype == nntrainer::TensorDim::DataType::FP16) {
          _FP16 *dst = batchsliced_hidden.getData<_FP16>() + out_off;
          for (size_t k = 0; k < bytes_per_row; ++k) {
            const uint8_t byte = row[k];
            dst[2 * k] = static_cast<_FP16>(
              (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale);
            dst[2 * k + 1] = static_cast<_FP16>(
              (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) *
              lut_scale);
          }
#endif
        } else {
          throw std::runtime_error(
            "EmbeddingLayer LUT mode: unsupported output dtype");
        }
      }
    }
    return;
  }

  // -------------------------------------------------------------------
  // Original non-LUT path (FP32/FP16 + Q4_0 / Q6_K block dequant).
  // -------------------------------------------------------------------
  nntrainer::Tensor &weight = context.getWeight(weight_idx);

  nntrainer::TensorDim out_tensor_dim =
    nntrainer::TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

  for (unsigned int b = 0; b < b_size; ++b) {
    float *in_data =
      input_.getAddress<float>(b * input_.getDim().getFeatureLen());
    nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

    int iter = to - from;

    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, static_cast<size_t>(iter), [&](size_t i) {
      size_t embed_idx = static_cast<size_t>(in_data[i]);
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      nntrainer::Tensor cur_weight =
        weight.getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);
      nntrainer::Tensor out_tensor =
        batchsliced_hidden.getSharedDataTensor(out_tensor_dim, out_dim * (i));

      if (weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 256 - 1) / 256;
        nntrainer::dequantize_row_q6_K(
          (void *)((char *)weight.getData<uint8_t>() +
                   (210 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      } else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 32 - 1) / 32;
        nntrainer::dequantize_row_q4_0(
          (void *)((char *)weight.getData<uint8_t>() +
                   (18 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      } else {
        out_tensor.copyData(cur_weight);
      }

      if (scale != 1.0f) {
        out_tensor.multiply_i(scale);
      }
    });

#ifdef DEBUG
    std::cout << context.getName() << " : "
              << "\n input:" << input_ << "\n weight: " << weight
              << "\n hidden: " << hidden_ << std::endl;
#endif
  }
}

void EmbeddingLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for Embedding layer is not supported");
}

void EmbeddingLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void EmbeddingLayer::exportTo(nntrainer::Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(embedding_props, method, this);
}

void EmbeddingLayer::save(std::ofstream &file,
                          nntrainer::RunLayerContext &run_context, bool opt_var,
                          ml::train::ExecutionMode mode, bool trainable,
                          nntrainer::TensorDim::DataType dtype,
                          ml::train::ISA target_isa) const {
  // @note shared weights are only be saved at the first access
  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (run_context.isGradientFirstAccess(i)) {
      auto &weight = run_context.getWeight(i);
      if (dtype == nntrainer::TensorDim::DataType::NONE ||
          weight.getDataType() == dtype)
        weight.save(file);
      else {
        NNTR_THROW_IF(weight.getDataType() !=
                        nntrainer::TensorDim::DataType::FP32,
                      std::runtime_error)
          << "Save with quantization only supports for FP32 weight.";
        ///@note The codelines below can be replaced with quantizer's
        /// quantize()
        nntrainer::TensorDim dim = weight.getDim();
        unsigned int K = dim.height();
        unsigned int N = dim.width();

        if (dtype == nntrainer::TensorDim::DataType::Q4_0) {

          // Skip quantization for bias-like tensors (1D with height == 1)
          // as they are not suitable for Q4_0 block quantization
          if (K == 1) {
            weight.save(file);
          } else {
            NNTR_THROW_IF(N % 32 != 0, std::invalid_argument)
              << "Q4_0 embedding quantization requires width to be "
                 "divisible by 32, but got width="
              << N;
            //////////////////////////////////////////////////////////////////
            ///@note Please note that Embedding layer doesn't need to be
            /// transposed!
            //////////////////////////////////////////////////////////////////
            nntrainer::Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                           {nntrainer::Tformat::NCHW, dtype});
            nntrainer::quantize_q4_0(weight.getData<float>(),
                                     quant_weight.getData<uint8_t>(), K, N,
                                     nullptr);
            quant_weight.save(file);
          }
        } else if (dtype == nntrainer::TensorDim::DataType::Q6_K) {
          //////////////////////////////////////////////////////////////////
          ///@note Please note that Embedding layer doesn't need to be
          /// transposed!
          //////////////////////////////////////////////////////////////////
          nntrainer::Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                         {nntrainer::Tformat::NCHW, dtype});
          nntrainer::quantize_q6_K(weight.getData<float>(),
                                   quant_weight.getData<uint8_t>(), K, N,
                                   nullptr);
          quant_weight.save(file);
        } else {
          NNTR_THROW_IF(true, std::runtime_error)
            << "This dtype is not supported in save with quantization";
        }
      }
    }
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_embedding_layer() {
  auto layer = new EmbeddingLayer();
  std::cout << "embedding layer created\n";
  return layer;
}

void destroy_embedding_layer(nntrainer::Layer *layer) {
  std::cout << "embeddinglayer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_embedding_layer,
                                                   destroy_embedding_layer};
}

#endif

} // namespace causallm
