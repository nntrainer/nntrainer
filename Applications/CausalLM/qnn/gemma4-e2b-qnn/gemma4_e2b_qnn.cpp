// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_e2b_qnn.cpp
 * @brief  QNN model implementation for Gemma 4 E2B with PLE.
 *         Follows the shared QNN KV-cache pattern; PLE handling layered on
 *         top.
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include "gemma4_e2b_qnn.h"
#include "android_memory_allocator.h"
#include "generate_qnn_utils.h"
#include "utf8_stream_util.h"

#include "streamer.h"
#include <llm_util.hpp>
#include <model.h>

#include <app_context.h>
#include <engine.h>
#include <factory.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "json.hpp"

using namespace causallm;

// =====================================================================
// Anonymous namespace helpers
// =====================================================================
namespace {

bool starts_with(const std::string &v, const std::string &p) {
  return v.compare(0, p.size(), p) == 0;
}

bool is_absolute_path(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

std::string dirname(const std::string &path) {
  auto pos = path.find_last_of('/');
  return (pos == std::string::npos) ? std::string() : path.substr(0, pos);
}

std::string rebase_relative_to_model_file(const std::string &path,
                                          const std::string &model_file) {
  if (path.empty() || is_absolute_path(path))
    return path;
  auto base = dirname(model_file);
  if (base.empty())
    return path;
  return base + "/" + path;
}

std::vector<int> read_eos_token_ids(const json &generation_cfg,
                                    const json &cfg) {
  const json *eos = nullptr;
  if (generation_cfg.contains("eos_token_id") &&
      !generation_cfg["eos_token_id"].is_null()) {
    eos = &generation_cfg["eos_token_id"];
  } else if (cfg.contains("eos_token_id") && !cfg["eos_token_id"].is_null()) {
    eos = &cfg["eos_token_id"];
  }

  if (eos == nullptr) {
    throw std::invalid_argument("missing eos_token_id");
  }
  if (eos->is_number_integer() || eos->is_number_unsigned()) {
    return {eos->get<int>()};
  }
  if (eos->is_array()) {
    return eos->get<std::vector<int>>();
  }

  throw std::invalid_argument("eos_token_id must be an integer or array");
}

// PLE 4-bit packed → uint16 (QNN consumer space) two-step requant.
// ufixed8 path: f = (q4bit + lut_offset) * lut_scale.
inline void dequant_nibbles_requant_u16(const uint8_t *packed, size_t elems,
                                        float lut_scale, int lut_offset,
                                        float out_scale, int out_offset,
                                        uint16_t *dst) {
  const float inv_out = 1.0f / out_scale;
  auto requant = [&](uint8_t nib) -> uint16_t {
    const float f = (static_cast<float>(nib) + lut_offset) * lut_scale;
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

// Sign-extend a 4-bit value (0..15 → -8..7).
inline int s4(unsigned nib) {
  return (nib & 0x8u) ? static_cast<int>(nib) - 16 : static_cast<int>(nib);
}

// PLE sfixed4 (per-row-per-layer) → uint16 requant. f = s4(nib) * row_scale.
inline void dequant_sfixed4_requant_u16(const uint8_t *packed, size_t elems,
                                        float row_scale, float out_scale,
                                        int out_offset, uint16_t *dst) {
  const float inv_out = 1.0f / out_scale;
  auto requant = [&](unsigned nib) -> uint16_t {
    const float f = static_cast<float>(s4(nib)) * row_scale;
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

} // namespace

// =====================================================================
// Auto-registration
// =====================================================================
__attribute__((constructor)) static void register_custom_models() {
  causallm::Factory::Instance().registerModel(
    "Gemma4_E2B_QNN", [](causallm::json cfg, causallm::json generation_cfg,
                         causallm::json nntr_cfg) {
      return std::make_unique<causallm::Gemma4_E2B_QNN>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
}

// =====================================================================
// PLE methods (dual-mode: 4-bit manifest OR raw uint16 bin)
// =====================================================================
namespace {
inline bool ends_with(const std::string &s, const std::string &suf) {
  return s.size() >= suf.size() &&
         0 == s.compare(s.size() - suf.size(), suf.size(), suf);
}

// ── PLE sfixed4 binary sidecar cache ──────────────────────────────
// The PLE scale array is shape [vocab, num_layers] (e.g. 262144*35 ≈
// 9.1M floats); re-parsing it from JSON with nlohmann on every cold
// start dominates load time. After the first parse we dump a compact
// binary next to the manifest and skip JSON entirely on later loads.
// Invalidated when the manifest's mtime changes.
constexpr char kPleSf4Magic[4] = {'P', 'S', '4', 'C'};
constexpr uint32_t kPleSf4Version = 1;

#pragma pack(push, 1)
struct PleSf4CacheHeader {
  char magic[4];
  uint32_t version;
  int64_t manifest_mtime;
  uint64_t row_elems;   // ple_row_elems_
  uint64_t layers;      // ple_layers_
  uint64_t scale_count; // ple_row_layer_scales_.size()
  uint32_t lut_path_len;
};
#pragma pack(pop)

inline int64_t ple_file_mtime_(const std::string &p) {
  struct stat st {};
  return (::stat(p.c_str(), &st) == 0) ? static_cast<int64_t>(st.st_mtime) : -1;
}

inline std::string ple_sf4_cache_path_(const std::string &manifest_path) {
  return manifest_path + ".sf4cache";
}
} // namespace

void Gemma4_E2B_QNN::open_ple_file_() {
  if (ple_file_name.empty())
    return;

  ple_is_4bit_ = ends_with(ple_file_name, ".json");
  ple_per_layer_ = 256;

  if (ple_is_4bit_) {
    // ── Fast path: sfixed4 binary sidecar cache (skips JSON parse) ──
    {
      const std::string cpath = ple_sf4_cache_path_(ple_file_name);
      std::ifstream cf(cpath, std::ios::binary);
      if (cf.is_open()) {
        PleSf4CacheHeader h{};
        cf.read(reinterpret_cast<char *>(&h), sizeof(h));
        const bool hdr_ok = static_cast<bool>(cf) &&
                            std::memcmp(h.magic, kPleSf4Magic, 4) == 0 &&
                            h.version == kPleSf4Version &&
                            h.manifest_mtime == ple_file_mtime_(ple_file_name);
        if (hdr_ok) {
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
          if (static_cast<bool>(cf) && ple_layers_ != 0 &&
              ple_layers_ * ple_per_layer_ == ple_row_elems_ &&
              generation_per_layer_dst_.size() <= ple_layers_) {
            ple_fd_ = open(lut_abs.c_str(), O_RDONLY);
            struct stat st;
            if (ple_fd_ >= 0 && fstat(ple_fd_, &st) == 0) {
              ple_file_size_ = static_cast<size_t>(st.st_size);
              const size_t exp_vocab = ple_file_size_ / ple_row_bytes_;
              const size_t scl_vocab =
                ple_row_layer_scales_.size() / ple_layers_;
              if (ple_file_size_ % ple_row_bytes_ == 0 &&
                  scl_vocab == exp_vocab) {
                void *m = mmap(nullptr, ple_file_size_, PROT_READ, MAP_PRIVATE,
                               ple_fd_, 0);
                if (m != MAP_FAILED) {
                  ple_mmap_ = static_cast<const uint8_t *>(m);
#ifdef POSIX_MADV_RANDOM
                  posix_madvise((void *)ple_mmap_, ple_file_size_,
                                POSIX_MADV_RANDOM);
#endif
                  std::cout << "[PLE] sfixed4 from binary cache " << cpath
                            << " rows=" << exp_vocab
                            << " layers=" << ple_layers_
                            << " per_layer=" << ple_per_layer_
                            << " scales=" << ple_row_layer_scales_.size()
                            << " (JSON parse skipped)" << std::endl;
                  return;
                }
              }
            }
            if (ple_fd_ >= 0) {
              ::close(ple_fd_);
              ple_fd_ = -1;
            }
          }
          // any validation failure → fall through to the JSON path
        }
      }
    }

    // ── 4-bit manifest: dispatch by `datatype` (ufixed8 / sfixed4) ──
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
    ple_row_bytes_ = (ple_row_elems_ + 1) / 2;
    ple_layers_ = ple_row_elems_ / ple_per_layer_;

    if (ple_layers_ * ple_per_layer_ != ple_row_elems_)
      throw std::runtime_error("PLE 'size' not divisible by 256");
    if (generation_per_layer_dst_.size() > ple_layers_)
      throw std::runtime_error("PLE layer count too small");

    if (ple_is_signed4_) {
      // Per-row-per-layer scale array; shape [vocab][layers] flat in
      // row-major. No offset (symmetric).
      const auto &scale_arr = qp.at("scale");
      if (!scale_arr.is_array())
        throw std::runtime_error(
          "PLE sfixed4: quant-param.scale must be an array");
      ple_row_layer_scales_.clear();
      ple_row_layer_scales_.reserve(scale_arr.size());
      for (const auto &v : scale_arr)
        ple_row_layer_scales_.push_back(v.get<float>());
      // Validate shape: must be a multiple of ple_layers_; vocab inferred.
      if (ple_row_layer_scales_.size() % ple_layers_ != 0)
        throw std::runtime_error(
          "PLE sfixed4: scale array length not divisible by num_layers");
      ple_scale_ = 1.0f; // unused
      ple_offset_ = 0;   // unused
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
    if (ple_file_size_ % ple_row_bytes_ != 0) {
      ::close(ple_fd_);
      ple_fd_ = -1;
      throw std::runtime_error("PLE bin size not multiple of row bytes");
    }

    // For sfixed4 the scale array's vocab dim must match the bin's row
    // count; for ufixed8 there is no per-row scale to validate.
    if (ple_is_signed4_) {
      const size_t expected_vocab = ple_file_size_ / ple_row_bytes_;
      const size_t scale_vocab = ple_row_layer_scales_.size() / ple_layers_;
      if (scale_vocab != expected_vocab)
        throw std::runtime_error(
          "PLE sfixed4 scale vocab=" + std::to_string(scale_vocab) +
          " != bin vocab=" + std::to_string(expected_vocab));
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
      std::cout << "[PLE] sfixed4 (rowwise+layerwise) mmaped " << lut_abs
                << " rows=" << (ple_file_size_ / ple_row_bytes_)
                << " layers=" << ple_layers_ << " per_layer=" << ple_per_layer_
                << " scales=" << ple_row_layer_scales_.size() << std::endl;
    } else {
      std::cout << "[PLE] ufixed8 (tensorwise) mmaped " << lut_abs
                << " rows=" << (ple_file_size_ / ple_row_bytes_)
                << " layers=" << ple_layers_ << " per_layer=" << ple_per_layer_
                << " scale=" << ple_scale_ << " offset=" << ple_offset_
                << std::endl;
    }

    if (ple_is_signed4_) {
      // Persist a binary sidecar so the next cold start skips the
      // multi-million-float JSON scale parse. Best-effort + atomic
      // rename; a read-only model dir simply keeps using JSON.
      const std::string cpath = ple_sf4_cache_path_(ple_file_name);
      const std::string tmp = cpath + ".tmp";
      std::cout << "[PLE-CACHE-DBG] writing sidecar: " << cpath
                << " scales=" << ple_row_layer_scales_.size() << " ("
                << (ple_row_layer_scales_.size() * sizeof(float)) << " bytes)"
                << std::endl;
      std::ofstream wf(tmp, std::ios::binary | std::ios::trunc);
      if (!wf.is_open()) {
        std::cout << "[PLE-CACHE-DBG] FAILED to open temp for write: " << tmp
                  << " (errno=" << errno << " " << std::strerror(errno)
                  << ") — keeping JSON path" << std::endl;
      } else {
        PleSf4CacheHeader h{};
        std::memcpy(h.magic, kPleSf4Magic, 4);
        h.version = kPleSf4Version;
        h.manifest_mtime = ple_file_mtime_(ple_file_name);
        h.row_elems = ple_row_elems_;
        h.layers = ple_layers_;
        h.scale_count = ple_row_layer_scales_.size();
        h.lut_path_len = static_cast<uint32_t>(lut_abs.size());
        wf.write(reinterpret_cast<const char *>(&h), sizeof(h));
        wf.write(lut_abs.data(), static_cast<std::streamsize>(lut_abs.size()));
        wf.write(reinterpret_cast<const char *>(ple_row_layer_scales_.data()),
                 static_cast<std::streamsize>(ple_row_layer_scales_.size() *
                                              sizeof(float)));
        const bool ok = static_cast<bool>(wf);
        wf.close();
        if (ok && wf) {
          if (std::rename(tmp.c_str(), cpath.c_str()) == 0) {
            std::cout << "[PLE-CACHE-DBG] sidecar written OK: " << cpath
                      << std::endl;
          } else {
            std::cout << "[PLE-CACHE-DBG] rename failed (errno=" << errno << " "
                      << std::strerror(errno) << "): " << tmp << " -> " << cpath
                      << std::endl;
            std::remove(tmp.c_str());
          }
        } else {
          std::cout << "[PLE-CACHE-DBG] write stream bad after "
                    << (ok ? "close" : "write") << " (errno=" << errno << " "
                    << std::strerror(errno) << ") — removing temp" << std::endl;
          std::remove(tmp.c_str());
        }
      }
    }
    return;
  }

  // ── raw UINT16: row = ple_layers * 256 uint16, no manifest ──
  // Derive layer count from the generation graph's per_layer_inputs_*
  // count (collected before open_ple_file_() is called).
  ple_layers_ = generation_per_layer_dst_.size();
  if (ple_layers_ == 0)
    throw std::runtime_error(
      "PLE raw uint16: no per_layer slots collected from generation graph");
  ple_row_elems_ = ple_layers_ * ple_per_layer_;
  ple_row_bytes_ = ple_row_elems_ * sizeof(uint16_t);

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
  if (ple_file_size_ % ple_row_bytes_ != 0) {
    ::close(ple_fd_);
    ple_fd_ = -1;
    throw std::runtime_error(
      "PLE raw uint16: file size not multiple of row bytes (expected " +
      std::to_string(ple_row_bytes_) + ")");
  }

  void *m = mmap(nullptr, ple_file_size_, PROT_READ, MAP_PRIVATE, ple_fd_, 0);
  if (m == MAP_FAILED) {
    ::close(ple_fd_);
    ple_fd_ = -1;
    throw std::runtime_error("mmap PLE bin: " + ple_file_name);
  }
  ple_u16_mmap_ = static_cast<const uint16_t *>(m);
  ple_mmap_ = static_cast<const uint8_t *>(m); // alias for cleanup
#ifdef POSIX_MADV_RANDOM
  posix_madvise(m, ple_file_size_, POSIX_MADV_RANDOM);
#endif
  std::cout << "[PLE] raw u16 mmaped " << ple_file_name
            << " rows=" << (ple_file_size_ / ple_row_bytes_)
            << " layers=" << ple_layers_ << " per_layer=" << ple_per_layer_
            << " (no requant)" << std::endl;
}

void Gemma4_E2B_QNN::close_ple_file_() {
  if (ple_mmap_) {
    munmap((void *)ple_mmap_, ple_file_size_);
    ple_mmap_ = nullptr;
    ple_u16_mmap_ = nullptr;
  }
  if (ple_fd_ >= 0) {
    ::close(ple_fd_);
    ple_fd_ = -1;
  }
}

void Gemma4_E2B_QNN::fill_prefill_ple_chunk_(const std::vector<int> &tokens,
                                             int chunk_idx, int chunk_len) {
  if (!ple_mmap_)
    return;
  const size_t L_pre = prefill_per_layer_dst_.size();
  const size_t per_layer_elems = ple_per_layer_;
  const int chunk_size_tokens = context_size;

  // The PLE binary is laid out per model-layer (`ple_layers_` chunks of
  // `per_layer_elems` per row). The prefill graph may expose a SUBSET of
  // model layers via `per_layer_inputs_N`, so source rows MUST be indexed
  // by the parsed N (model layer index), not by the dense slot index `l`.
  const int *pre_layer_idx = prefill_per_layer_model_index_.data();

  if (ple_is_4bit_) {
    const size_t per_layer_bytes = per_layer_elems / 2;
    if (ple_is_signed4_) {
      // sfixed4: per-row-per-layer scale lookup, signed nibble decode.
      const float *scales = ple_row_layer_scales_.data();
      for (int t = 0; t < chunk_size_tokens; ++t) {
        const int abs_idx = chunk_idx * chunk_size_tokens + t;
        const int token_id = (t < chunk_len) ? tokens[abs_idx] : padding_token;
        const uint8_t *row = ple_mmap_ + (size_t)token_id * ple_row_bytes_;
        const float *row_scales = scales + (size_t)token_id * ple_layers_;
        for (size_t l = 0; l < L_pre; ++l) {
          const size_t ml = (size_t)pre_layer_idx[l];
          uint16_t *dst =
            prefill_per_layer_dst_[l] + (size_t)t * per_layer_elems;
          dequant_sfixed4_requant_u16(
            row + ml * per_layer_bytes, per_layer_elems, row_scales[ml],
            prefill_per_layer_scale_[l], prefill_per_layer_offset_[l], dst);
        }
      }
    } else {
      for (int t = 0; t < chunk_size_tokens; ++t) {
        const int abs_idx = chunk_idx * chunk_size_tokens + t;
        const int token_id = (t < chunk_len) ? tokens[abs_idx] : padding_token;
        const uint8_t *row = ple_mmap_ + (size_t)token_id * ple_row_bytes_;
        for (size_t l = 0; l < L_pre; ++l) {
          const size_t ml = (size_t)pre_layer_idx[l];
          uint16_t *dst =
            prefill_per_layer_dst_[l] + (size_t)t * per_layer_elems;
          dequant_nibbles_requant_u16(row + ml * per_layer_bytes,
                                      per_layer_elems, ple_scale_, ple_offset_,
                                      prefill_per_layer_scale_[l],
                                      prefill_per_layer_offset_[l], dst);
        }
      }
    }
  } else {
    // raw uint16: per-layer slice memcpy. Source already in consumer space.
    for (int t = 0; t < chunk_size_tokens; ++t) {
      const int abs_idx = chunk_idx * chunk_size_tokens + t;
      const int token_id = (t < chunk_len) ? tokens[abs_idx] : padding_token;
      const uint16_t *row = ple_u16_mmap_ + (size_t)token_id * ple_row_elems_;
      for (size_t l = 0; l < L_pre; ++l) {
        const size_t ml = (size_t)pre_layer_idx[l];
        uint16_t *dst = prefill_per_layer_dst_[l] + (size_t)t * per_layer_elems;
        std::memcpy(dst, row + ml * per_layer_elems,
                    per_layer_elems * sizeof(uint16_t));
      }
    }
  }
}

void Gemma4_E2B_QNN::fill_generation_ple_(int token_id) {
  if (!ple_mmap_)
    return;
  const size_t L_gen = generation_per_layer_dst_.size();
  const size_t per_layer_elems = ple_per_layer_;
  const int *gen_layer_idx = generation_per_layer_model_index_.data();

  if (ple_is_4bit_) {
    const size_t per_layer_bytes = per_layer_elems / 2;
    const uint8_t *row = ple_mmap_ + (size_t)token_id * ple_row_bytes_;
    if (ple_is_signed4_) {
      const float *row_scales =
        ple_row_layer_scales_.data() + (size_t)token_id * ple_layers_;
      for (size_t l = 0; l < L_gen; ++l) {
        const size_t ml = (size_t)gen_layer_idx[l];
        dequant_sfixed4_requant_u16(
          row + ml * per_layer_bytes, per_layer_elems, row_scales[ml],
          generation_per_layer_scale_[l], generation_per_layer_offset_[l],
          generation_per_layer_dst_[l]);
      }
      // ── Debug: dump token's first decoded nibbles + scales (once) ──
      static bool dbg_done = false;
      if (!dbg_done) {
        dbg_done = true;
        std::cout << "[PLE-S4-DBG] token=" << token_id
                  << " row_offset=" << (size_t)token_id * ple_row_bytes_
                  << "\n[PLE-S4-DBG] L0 row_scale=" << row_scales[0]
                  << " L1=" << row_scales[1] << " L17=" << row_scales[17]
                  << " L34=" << row_scales[34] << "\n";
        std::cout << "[PLE-S4-DBG] L0 raw bytes [0..7]: ";
        for (int i = 0; i < 8; ++i)
          std::cout << std::hex << (int)row[i] << " ";
        std::cout << std::dec << "\n[PLE-S4-DBG] L0 nibbles s4 [0..15]: ";
        for (int i = 0; i < 8; ++i) {
          int lo = s4(row[i] & 0x0F);
          int hi = s4((row[i] >> 4) & 0x0F);
          std::cout << lo << " " << hi << " ";
        }
        std::cout << "\n[PLE-S4-DBG] L0 dst u16 [0..7]: ";
        for (int i = 0; i < 8; ++i)
          std::cout << generation_per_layer_dst_[0][i] << " ";
        std::cout << "\n[PLE-S4-DBG] L0 consumer scale="
                  << generation_per_layer_scale_[0]
                  << " offset=" << generation_per_layer_offset_[0] << "\n";
      }
    } else {
      for (size_t l = 0; l < L_gen; ++l) {
        const size_t ml = (size_t)gen_layer_idx[l];
        dequant_nibbles_requant_u16(
          row + ml * per_layer_bytes, per_layer_elems, ple_scale_, ple_offset_,
          generation_per_layer_scale_[l], generation_per_layer_offset_[l],
          generation_per_layer_dst_[l]);
      }
    }
  } else {
    const uint16_t *row = ple_u16_mmap_ + (size_t)token_id * ple_row_elems_;
    for (size_t l = 0; l < L_gen; ++l) {
      const size_t ml = (size_t)gen_layer_idx[l];
      std::memcpy(generation_per_layer_dst_[l], row + ml * per_layer_elems,
                  per_layer_elems * sizeof(uint16_t));
    }
  }
}

// =====================================================================
// Destructor
// =====================================================================
Gemma4_E2B_QNN::~Gemma4_E2B_QNN() {
  close_ple_file_();
  this->prefill_kv_zero_byte_.clear();
}

// =====================================================================
// KV cache helpers
// =====================================================================
void Gemma4_E2B_QNN::initialize_kv_cache() {
  kv_len = 0;
  conversation_started_ = false;
  for (int i = 0; i < (int)kvs.size(); ++i) {
    std::memcpy(kvs[i], fresh_kvs[i], kv_sizes[i]);
  }
  reset_prefill_kv_cache_inputs();
}

void Gemma4_E2B_QNN::reset_prefill_kv_cache_inputs() {
  for (int i = 0; i < (int)prefill_kvs.size(); ++i) {
    std::fill_n(prefill_kvs[i], prefill_kv_sizes[i], prefill_kv_zero_byte_[i]);
  }
}

void Gemma4_E2B_QNN::sync_generation_kv_cache_to_prefill() {
  reset_prefill_kv_cache_inputs();
  if (kv_len <= 0)
    return;

#pragma omp parallel for
  for (int i = 0; i < (int)prefill_kvs.size(); ++i) {
    int gen_idx = prefill_to_generation_kv_indices[i];
    int gen_layer = gen_idx / 2; // Gemma 4: 2 KV per layer
    if (gen_idx < 0 || gen_idx >= (int)kvs.size() || gen_layer < 0 ||
        gen_layer >= (int)kv_row_lengths.size())
      continue;

    copy_kv_cache_window(prefill_kvs[i], prefill_kv_row_lengths[i],
                         (uint8_t *)kvs[gen_idx], kv_row_lengths[gen_layer],
                         kv_len, prefill_kv_is_key[i] != 0,
                         kv_columns[gen_layer]);
  }
}

// =====================================================================
// initialize()
// =====================================================================
void Gemma4_E2B_QNN::initialize() {
  Quick_Dot_AI_QNN::initialize();
  LOGD("Quick_Dot_AI_QNN::initialize() done");

  std::string prefill_graph = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];

  auto &prefill_graph_info = models[prefill_graph].graph_info;
  auto &generation_graph_info = models[generation_graph].graph_info;
  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;

  if (prefill_inputs.size() != prefill_graph_info.raw_inputs.size())
    throw std::runtime_error("prefill input count mismatch");
  if (generation_inputs.size() != generation_graph_info.raw_inputs.size())
    throw std::runtime_error("generation input count mismatch");

  // ── Mask / RoPE element counts ──
  prefill_attention_mask_elements =
    GraphParser::get_named_tensor_elements_or_throw(
      prefill_graph_info.raw_inputs, "attention_mask");

  prefill_attention_mask_columns =
    GraphParser::get_tensor_info_or_throw(prefill_graph_info.raw_inputs,
                                          "attention_mask")
      .dimensions.back(); // 8192

  prefill_sliding_attention_mask_elements =
    GraphParser::get_named_tensor_elements_or_throw(
      prefill_graph_info.raw_inputs, "sliding_attention_mask");

  prefill_sliding_attention_mask_columns =
    GraphParser::get_tensor_info_or_throw(prefill_graph_info.raw_inputs,
                                          "sliding_attention_mask")
      .dimensions.back(); // 768

  generation_attention_mask_elements =
    GraphParser::get_named_tensor_elements_or_throw(
      generation_graph_info.raw_inputs, "attention_mask");
  generation_sliding_attention_mask_elements =
    GraphParser::get_named_tensor_elements_or_throw(
      generation_graph_info.raw_inputs, "sliding_attention_mask");

  generation_full_kv_past_length = generation_attention_mask_elements - 1;
  generation_sliding_kv_past_length =
    generation_sliding_attention_mask_elements - 1;

  pos_dim = GraphParser::get_tensor_info_or_throw(prefill_graph_info.raw_inputs,
                                                  "position_ids_cos")
              .dimensions.back();
  swa_pos_dim = GraphParser::get_tensor_info_or_throw(
                  prefill_graph_info.raw_inputs, "swa_position_ids_cos")
                  .dimensions.back();

  // ── Debug: confirm prefill and generation share the same pos/swa dims ──
  {
    const int gen_pos = GraphParser::get_tensor_info_or_throw(
                          generation_graph_info.raw_inputs, "position_ids_cos")
                          .dimensions.back();
    const int gen_swa =
      GraphParser::get_tensor_info_or_throw(generation_graph_info.raw_inputs,
                                            "swa_position_ids_cos")
        .dimensions.back();
    std::cout << "[ROPE-DBG] prefill pos_dim=" << pos_dim
              << " gen pos_dim=" << gen_pos
              << " | prefill swa_pos_dim=" << swa_pos_dim
              << " gen swa_pos_dim=" << gen_swa << std::endl;
    if (gen_pos != pos_dim || gen_swa != swa_pos_dim)
      std::cout << "[ROPE-DBG] !!! prefill/gen position dims DIFFER !!!"
                << std::endl;
  }

  rope_cache_seq_len =
    std::max(max_seq_len, generation_attention_mask_elements);

  // ── Bind input tensor pointers ──
  int prefill_input_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "input_embeds");
  int generation_input_idx = GraphParser::find_tensor_index(
    generation_graph_info.raw_inputs, "input_embeds");
  input_sample = std::get<float *>(prefill_inputs[prefill_input_idx]);
  generation_sample =
    std::get<float *>(generation_inputs[generation_input_idx]);

  attention_mask =
    std::get<uint16_t *>(prefill_inputs[GraphParser::find_tensor_index(
      prefill_graph_info.raw_inputs, "attention_mask")]);
  sliding_attention_mask =
    std::get<uint16_t *>(prefill_inputs[GraphParser::find_tensor_index(
      prefill_graph_info.raw_inputs, "sliding_attention_mask")]);
  generation_attention_mask =
    std::get<uint16_t *>(generation_inputs[GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "attention_mask")]);
  generation_sliding_attention_mask =
    std::get<uint16_t *>(generation_inputs[GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "sliding_attention_mask")]);

  prefill_position_ids_cos =
    std::get<uint16_t *>(prefill_inputs[GraphParser::find_tensor_index(
      prefill_graph_info.raw_inputs, "position_ids_cos")]);
  prefill_position_ids_sin =
    std::get<uint16_t *>(prefill_inputs[GraphParser::find_tensor_index(
      prefill_graph_info.raw_inputs, "position_ids_sin")]);
  generation_position_ids_cos =
    std::get<uint16_t *>(generation_inputs[GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "position_ids_cos")]);
  generation_position_ids_sin =
    std::get<uint16_t *>(generation_inputs[GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "position_ids_sin")]);
  prefill_swa_position_ids_cos =
    std::get<uint16_t *>(prefill_inputs[GraphParser::find_tensor_index(
      prefill_graph_info.raw_inputs, "swa_position_ids_cos")]);
  prefill_swa_position_ids_sin =
    std::get<uint16_t *>(prefill_inputs[GraphParser::find_tensor_index(
      prefill_graph_info.raw_inputs, "swa_position_ids_sin")]);
  generation_swa_position_ids_cos =
    std::get<uint16_t *>(generation_inputs[GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "swa_position_ids_cos")]);
  generation_swa_position_ids_sin =
    std::get<uint16_t *>(generation_inputs[GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "swa_position_ids_sin")]);

  // ── RoPE cache ──

  // std::tuple<uint16_t *, uint16_t *> cos_sin_tuple =
  //     get_cos_sin(rope_cache_seq_len, pos_dim, rope_theta);
  // position_ids_cos = std::get<0>(cos_sin_tuple);
  // position_ids_sin = std::get<1>(cos_sin_tuple);
  // allocated_ptrs_.insert(position_ids_cos);
  // allocated_ptrs_.insert(position_ids_sin);

  // std::tuple<uint16_t *, uint16_t *> swa_cos_sin_tuple =
  //     get_cos_sin(rope_cache_seq_len, swa_pos_dim, local_rope_theta);
  // swa_position_ids_cos = std::get<0>(swa_cos_sin_tuple);
  // swa_position_ids_sin = std::get<1>(swa_cos_sin_tuple);
  // allocated_ptrs_.insert(swa_position_ids_cos);
  // allocated_ptrs_.insert(swa_position_ids_sin);

  std::tuple<uint16_t *, uint16_t *> cos_sin_tuple =
    get_cos_sin(rope_cache_seq_len, pos_dim, rope_theta_full, rope_type_full,
                rope_partial_factor, rope_scaling_factor_full, g_head_dim);
  position_ids_cos = std::get<0>(cos_sin_tuple);
  position_ids_sin = std::get<1>(cos_sin_tuple);
  allocated_ptrs_.insert(position_ids_cos);
  allocated_ptrs_.insert(position_ids_sin);

  // ── Sliding window RoPE (default = no scaling) ──
  std::tuple<uint16_t *, uint16_t *> swa_cos_sin_tuple = get_cos_sin(
    rope_cache_seq_len, swa_pos_dim, rope_theta_sliding, rope_type_sliding,
    /*partial=*/1.0, rope_scaling_factor_sliding, l_head_dim);
  swa_position_ids_cos = std::get<0>(swa_cos_sin_tuple);
  swa_position_ids_sin = std::get<1>(swa_cos_sin_tuple);
  allocated_ptrs_.insert(swa_position_ids_cos);
  allocated_ptrs_.insert(swa_position_ids_sin);

  // ── PLE per-layer dst + scale/offset collection ──
  auto collect_per_layer =
    [](const GraphInfo &gi, std::vector<causallm::IO_TensorType> &inputs,
       std::vector<uint16_t *> &dsts, std::vector<float> &scales,
       std::vector<int> &offsets, std::vector<int> &model_indices) {
      std::map<int, std::tuple<uint16_t *, float, int>> by_index;
      for (size_t idx = 0; idx < gi.raw_inputs.size(); ++idx) {
        const auto &info = gi.raw_inputs[idx];
        const std::string prefix = "per_layer_inputs_";
        if (info.name.rfind(prefix, 0) != 0)
          continue;
        int n = std::stoi(info.name.substr(prefix.size()));
        by_index[n] = std::make_tuple(std::get<uint16_t *>(inputs[idx]),
                                      info.scale, info.offset);
      }
      dsts.clear();
      scales.clear();
      offsets.clear();
      model_indices.clear();
      for (auto &kv : by_index) {
        model_indices.push_back(kv.first);
        dsts.push_back(std::get<0>(kv.second));
        scales.push_back(std::get<1>(kv.second));
        offsets.push_back(std::get<2>(kv.second));
      }
    };
  collect_per_layer(prefill_graph_info, prefill_inputs, prefill_per_layer_dst_,
                    prefill_per_layer_scale_, prefill_per_layer_offset_,
                    prefill_per_layer_model_index_);
  collect_per_layer(generation_graph_info, generation_inputs,
                    generation_per_layer_dst_, generation_per_layer_scale_,
                    generation_per_layer_offset_,
                    generation_per_layer_model_index_);

  std::cout << "[PLE] prefill slots=" << prefill_per_layer_dst_.size()
            << " generation slots=" << generation_per_layer_dst_.size()
            << std::endl;
  std::cout << "[PLE] prefill model indices: ";
  for (int n : prefill_per_layer_model_index_)
    std::cout << n << " ";
  std::cout << "\n[PLE] generation model indices: ";
  for (int n : generation_per_layer_model_index_)
    std::cout << n << " ";
  std::cout << std::endl;

  open_ple_file_();

  // ── KV cache mapping (shared QNN KV-cache pattern, 2 KV per layer) ──
  this->kvs.clear();
  this->fresh_kvs.clear();
  this->kv_sizes.clear();
  this->kv_row_lengths.clear();
  this->kv_columns.clear();
  this->prefill_kvs.clear();
  this->prefill_kv_sizes.clear();
  this->prefill_kv_row_lengths.clear();
  this->prefill_to_generation_kv_indices.clear();
  this->prefill_kv_is_key.clear();
  this->prefill_output_kv_bindings.clear();
  this->generation_output_kv_bindings.clear();

  std::unordered_map<std::string, int> generation_kv_index_by_name;

  int kv_layer_count = 0;
  while (true) {
    const std::string name =
      "past_key_" + std::to_string(kv_layer_count) + "_h0_in";
    if (find_tensor_index_or_minus_one(generation_graph_info.raw_inputs, name) <
        0)
      break;
    kv_layer_count++;
  }

  std::cout << "[KV-DBG] kv_layer_count=" << kv_layer_count
            << " (num_hidden_layers=" << num_hidden_layers << ")" << std::endl;

  LOGD("KV layer count = %d (num_hidden_layers config = %d)", kv_layer_count,
       num_hidden_layers);
  for (int layer = 0; layer < kv_layer_count; ++layer) {
    const std::vector<std::string> kv_names = {
      "past_key_" + std::to_string(layer) + "_h0_in",
      "past_value_" + std::to_string(layer) + "_h0_in",
    };

    {
      const auto &gen_key_info = GraphParser::get_tensor_info_or_throw(
        generation_graph_info.raw_inputs, kv_names[0]);
      this->kv_row_lengths.push_back(gen_key_info.dimensions.back());
      this->kv_columns.push_back(gen_key_info.dimensions[2]);

      // ── Debug: dump KV tensor shape for first 6 + last layer ──
      if (layer < 6 || layer + 1 == kv_layer_count) {
        std::cout << "[KV-DBG] layer " << layer << " key dims=[";
        for (size_t i = 0; i < gen_key_info.dimensions.size(); ++i) {
          if (i)
            std::cout << ",";
          std::cout << gen_key_info.dimensions[i];
        }
        std::cout << "] → row_len=" << gen_key_info.dimensions.back()
                  << " columns(dim[2])=" << gen_key_info.dimensions[2]
                  << " dtype=" << gen_key_info.data_type
                  << " scale=" << gen_key_info.scale
                  << " offset=" << gen_key_info.offset << " size_bytes="
                  << GraphParser::get_tensor_size(gen_key_info) << "\n";
        const auto &gen_val_info = GraphParser::get_tensor_info_or_throw(
          generation_graph_info.raw_inputs, kv_names[1]);
        std::cout << "[KV-DBG] layer " << layer << " val dims=[";
        for (size_t i = 0; i < gen_val_info.dimensions.size(); ++i) {
          if (i)
            std::cout << ",";
          std::cout << gen_val_info.dimensions[i];
        }
        std::cout << "] dtype=" << gen_val_info.data_type
                  << " scale=" << gen_val_info.scale
                  << " offset=" << gen_val_info.offset << " size_bytes="
                  << GraphParser::get_tensor_size(gen_val_info) << "\n";
      }
    }

    for (const auto &name : kv_names) {
      int gen_idx =
        GraphParser::find_tensor_index(generation_graph_info.raw_inputs, name);
      int pre_idx =
        find_tensor_index_or_minus_one(prefill_graph_info.raw_inputs, name);
      const auto &gen_info = generation_graph_info.raw_inputs[gen_idx];
      int size = GraphParser::get_tensor_size(gen_info);

      // Layer-specific zero-point byte.
      int zq = std::max(0, std::min(255, -gen_info.offset));

      // ★ REUSE existing generation_inputs buffer (don't re-allocate).
      auto *current_kv = std::get<uint8_t *>(generation_inputs[gen_idx]);
      std::memset(current_kv, zq, size);

      // Separate fresh_kv to use as run-start reset state.
      auto *fresh_kv = static_cast<uint8_t *>(tracked_allocate(size));
      std::memset(fresh_kv, zq, size);
      allocated_ptrs_.insert(fresh_kv);

      int kv_input_index = (int)this->kvs.size();
      this->kvs.push_back((uint16_t *)current_kv);
      this->fresh_kvs.push_back((uint16_t *)fresh_kv);
      this->kv_sizes.push_back(size);
      generation_kv_index_by_name[name] = kv_input_index;

      if (pre_idx >= 0) {
        const auto &pre_info = prefill_graph_info.raw_inputs[pre_idx];
        const bool is_key = starts_with(name, "past_key_");
        const int pre_row_length =
          is_key ? pre_info.dimensions.back()
                 : pre_info.dimensions[pre_info.dimensions.size() - 2];

        this->prefill_kvs.push_back(
          std::get<uint8_t *>(prefill_inputs[pre_idx]));
        this->prefill_kv_sizes.push_back(
          GraphParser::get_tensor_size(pre_info));
        this->prefill_kv_row_lengths.push_back(pre_row_length);
        this->prefill_to_generation_kv_indices.push_back(kv_input_index);
        this->prefill_kv_is_key.push_back(is_key ? 1 : 0);

        // Layer-specific zero point for prefill reset (replaces the 128 fill).
        int pzq = std::max(0, std::min(255, -pre_info.offset));
        this->prefill_kv_zero_byte_.push_back((uint8_t)pzq);

        // Initialize prefill buffer to its layer-specific zero point.
        std::memset(std::get<uint8_t *>(prefill_inputs[pre_idx]), pzq,
                    GraphParser::get_tensor_size(pre_info));
      }
    }
  }

  prefill_output_kv_bindings =
    build_kv_output_bindings(prefill_graph_info.raw_outputs,
                             generation_kv_index_by_name, prefill_graph, 2);
  generation_output_kv_bindings =
    build_kv_output_bindings(generation_graph_info.raw_outputs,
                             generation_kv_index_by_name, generation_graph, 2);

  generation_logits_output_index =
    find_tensor_index_or_minus_one(generation_graph_info.raw_outputs, "logits");
  if (generation_logits_output_index < 0) {
    generation_logits_output_index =
      static_cast<int>(generation_graph_info.raw_outputs.size()) - 1;
  }

  LOGD("KV mapping: gen_inputs=%zu pre_inputs=%zu pre_outs=%zu gen_outs=%zu "
       "logits_output=%d",
       this->kvs.size(), this->prefill_kvs.size(),
       this->prefill_output_kv_bindings.size(),
       this->generation_output_kv_bindings.size(),
       generation_logits_output_index);

  // ── Debug: dump prefill/gen output KV bindings (layer indices) ──
  std::cout << "[KV-OUT-DBG] prefill output KV layers (key only): ";
  for (const auto &b : prefill_output_kv_bindings)
    if (b.is_key)
      std::cout << b.layer_index << " ";
  std::cout << "\n[KV-OUT-DBG] generation output KV layers (key only): ";
  for (const auto &b : generation_output_kv_bindings)
    if (b.is_key)
      std::cout << b.layer_index << " ";
  std::cout << "\n[KV-OUT-DBG] prefill bindings count="
            << prefill_output_kv_bindings.size()
            << " generation bindings count="
            << generation_output_kv_bindings.size() << std::endl;
  // ── Debug: dump prefill OUTPUT past_key shapes (sliding & full layers) ──
  // Confirms whether the output is per-chunk [head_dim, 256] or full
  // updated cache [head_dim, 7936+] or something else. The append code
  // assumes per-chunk with src_row_length=context_size=256.
  std::cout << "[KV-OUT-SHAPE-DBG] prefill output K shapes: ";
  for (const auto &b : prefill_output_kv_bindings) {
    if (!b.is_key)
      continue;
    if (b.layer_index > 5 && b.layer_index != 14)
      continue;
    const auto &info = prefill_graph_info.raw_outputs[b.output_index];
    std::cout << "L" << b.layer_index << "=[";
    for (size_t i = 0; i < info.dimensions.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << info.dimensions[i];
    }
    std::cout << "] ";
  }
  std::cout << "\n[KV-OUT-SHAPE-DBG] prefill output V shapes: ";
  for (const auto &b : prefill_output_kv_bindings) {
    if (b.is_key)
      continue;
    if (b.layer_index > 5 && b.layer_index != 14)
      continue;
    const auto &info = prefill_graph_info.raw_outputs[b.output_index];
    std::cout << "L" << b.layer_index << "=[";
    for (size_t i = 0; i < info.dimensions.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << info.dimensions[i];
    }
    std::cout << "] ";
  }
  std::cout << "\n[KV-OUT-SHAPE-DBG] generation output K shapes: ";
  for (const auto &b : generation_output_kv_bindings) {
    if (!b.is_key)
      continue;
    if (b.layer_index > 5 && b.layer_index != 14)
      continue;
    const auto &info = generation_graph_info.raw_outputs[b.output_index];
    std::cout << "L" << b.layer_index << "=[";
    for (size_t i = 0; i < info.dimensions.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << info.dimensions[i];
    }
    std::cout << "] ";
  }
  std::cout << std::endl;

  // ── Logit dequant params (overrides setupParameters defaults) ──
  const auto &logits_info = GraphParser::get_tensor_info_or_throw(
    generation_graph_info.raw_outputs, "logits");
  logit_scale = logits_info.scale;
  logit_offset = logits_info.offset;
  std::cout << "[LOGIT-DBG] dtype=" << logits_info.data_type
            << " scale=" << logit_scale << " offset=" << logit_offset
            << " | final_logit_softcapping=" << final_logit_softcapping
            << std::endl;
  // Diagnostic experiment: disable our externally-applied softcap if the
  // QNN graph already applies it internally (double-softcap would flatten
  // the distribution and produce structured-but-mixed-language gibberish).
  // Toggle by env var GEMMA4_DISABLE_SOFTCAP=1 to test.
  if (const char *env = std::getenv("GEMMA4_DISABLE_SOFTCAP");
      env && env[0] == '1') {
    std::cout << "[LOGIT-DBG] GEMMA4_DISABLE_SOFTCAP=1 → forcing softcap=0"
              << std::endl;
    final_logit_softcapping = 0.0f;
  }
  final_logit_softcapping = 0.0f;

  initialize_kv_cache();

  LOGD("----------------------- initialize() done");
}

// =====================================================================
// setupParameters
// =====================================================================
void Gemma4_E2B_QNN::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  Quick_Dot_AI_QNN::setupParameters(cfg, generation_cfg, nntr_cfg);

  num_hidden_layers = cfg["num_hidden_layers"].get<int>();
  hidden_size = cfg["hidden_size"].get<int>();
  vocab_size = cfg["vocab_size"].get<int>();
  max_seq_len = cfg["max_seq_len"].get<int>();
  sliding_window = cfg["sliding_window"].get<int>();
  context_size = cfg["context_size"].get<int>();
  g_head_dim = cfg["global_head_dim"].get<int>();
  l_head_dim = cfg["head_dim"].get<int>();
  head_dim = g_head_dim;

  // ─── RoPE parameters (new format preferred) ───
  if (cfg.contains("rope_parameters") && cfg["rope_parameters"].is_object()) {
    auto &rp = cfg["rope_parameters"];

    if (rp.contains("full_attention") && rp["full_attention"].is_object()) {
      auto &fa = rp["full_attention"];
      rope_theta_full = fa.value("rope_theta", 1000000.0f);
      rope_partial_factor = fa.value("partial_rotary_factor", 1.0f);
      rope_scaling_factor_full = fa.value("factor", 1.0f);
      rope_type_full = fa.value("rope_type", std::string("default"));
    }

    if (rp.contains("sliding_attention") &&
        rp["sliding_attention"].is_object()) {
      auto &sa = rp["sliding_attention"];
      rope_theta_sliding = sa.value("rope_theta", 10000.0f);
      rope_scaling_factor_sliding = sa.value("factor", 1.0f);
      rope_type_sliding = sa.value("rope_type", std::string("default"));
    }
  } else {
    // Legacy flat form
    rope_theta_full = cfg.value("rope_theta", 1000000.0f);
    rope_theta_sliding = cfg.value("local_rope_theta", 10000.0f);
    rope_scaling_factor_full = 1.0f;
    rope_scaling_factor_sliding = 1.0f;
  }

  // rope_theta       = rope_theta_full;
  // local_rope_theta = rope_theta_sliding;

  LOGD("RoPE full: theta=%f partial=%f factor=%f type=%s", rope_theta_full,
       rope_partial_factor, rope_scaling_factor_full, rope_type_full.c_str());
  LOGD("RoPE sliding: theta=%f factor=%f type=%s", rope_theta_sliding,
       rope_scaling_factor_sliding, rope_type_sliding.c_str());

  padding_token = generation_cfg["pad_token_id"].get<int>();
  eos_tokens = read_eos_token_ids(generation_cfg, cfg);
  temperature = generation_cfg["temperature"].get<float>();
  top_k = generation_cfg["top_k"].get<int>();
  top_p = generation_cfg["top_p"].get<float>();
  repetition_penalty = generation_cfg.value("repetition_penalty", 1.0f);
  logit_scale = generation_cfg.value("logit_scale", 1.0f);
  logit_offset = generation_cfg.value("logit_offset", 0);

  // Gemma final-logit soft-cap (0 disables). Without it the model
  // collapses into repetition since a few raw logits dominate softmax.
  final_logit_softcapping = cfg.value("final_logit_softcapping", 0.0f);
  LOGD("final_logit_softcapping = %f", final_logit_softcapping);

  lora_path = nntr_cfg.value("lora_path", "");
  lora_path = rebase_relative_to_model_file(lora_path, model_file_name);
  ple_file_name = nntr_cfg.value("ple_file_name", "");
  ple_file_name = rebase_relative_to_model_file(ple_file_name, model_file_name);
}

// =====================================================================
// run()
// =====================================================================
void Gemma4_E2B_QNN::run(const WSTR prompt, bool /*do_sample*/,
                         const WSTR /*system_prompt*/,
                         const WSTR /*tail_prompt*/, bool log_output) {
  last_output_.clear();

  resetKvCache();

  stop_requested_.store(false, std::memory_order_release);

  std::string prefill_graph = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];
  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;
  auto &prefill_model = models[prefill_graph].model_handle;
  auto &generation_model = models[generation_graph].model_handle;

  const std::string model_prompt = promptToUtf8(prompt);
  auto _input = tokenizer->Encode(model_prompt);
  if (_input.size() <= 1) {
    std::cout << "[Error] Empty input\n";
    return;
  }
  unsigned int input_len = _input.size() - 1;
  int token = _input.back();
  // Similar with n / m but ceiling of the float n / m result, instead of floor.
  auto divide_ceiling = [](int n, int m) {
    return (n % m != 0) ? ((n / m) + 1) : (n / m);
  };

  const unsigned int n_chunks = divide_ceiling(input_len, context_size);

  if (kv_len + (int)input_len >= generation_full_kv_past_length)
    throw std::runtime_error("Input prompt leaves no room for generation");

  std::vector<int> output;
  std::vector<causallm::IO_TensorType> outputs;

  auto append_generation_token_to_kv_cache = [&](int t) {
    if (kv_len >= generation_full_kv_past_length)
      return;
    fill_generation_inputs(
      generation_sample, t, generation_attention_mask,
      generation_attention_mask_elements, generation_sliding_attention_mask,
      generation_sliding_attention_mask_elements,
      generation_full_kv_past_length, generation_sliding_kv_past_length,
      generation_position_ids_cos, generation_position_ids_sin,
      position_ids_cos, position_ids_sin, pos_dim,
      generation_swa_position_ids_cos, generation_swa_position_ids_sin,
      swa_position_ids_cos, swa_position_ids_sin, swa_pos_dim, kv_len,
      rope_cache_seq_len);
    fill_generation_ple_(t);
    auto term = run_qnn_inference(generation_model, 1, generation_inputs,
                                  models[generation_graph].graph_info);
    std::vector<uint8_t *> kv_ptrs;
    for (auto *kv : kvs)
      kv_ptrs.push_back((uint8_t *)kv);
    append_outputs_to_kv_cache(term, generation_output_kv_bindings, kv_ptrs,
                               kv_row_lengths, kv_len, 1, 1, generation_graph,
                               &kv_columns);
    kv_len += 1;
  };

  // ── Prefill ──
  auto start_prefill = std::chrono::system_clock::now();

  for (int c = 0; c < (int)n_chunks; ++c) {
    int chunk_len = ((c + 1) * context_size < (int)input_len)
                      ? context_size
                      : ((int)input_len - c * context_size);

    sync_generation_kv_cache_to_prefill();

    for (int i = 0; i < context_size; ++i)
      input_sample[i] =
        (i < chunk_len) ? _input[c * context_size + i] : padding_token;

    fill_attention_mask_with_length(
      context_size, prefill_attention_mask_columns, chunk_len, attention_mask);
    // Past KV cap MUST reserve the trailing `context_size` cols for the
    // current chunk's causal triangle (those cols are written by
    // fill_attention_mask_with_length above). Using
    // generation_full_kv_past_length here would overlap chunk cols when
    // kv_len > prefill_attention_mask_columns - context_size, corrupting
    // the mask on multi-chunk prefill. Match the shared QNN KV-cache
    // convention:
    //   past_max = mask_columns - context_size
    {
      const int prefill_full_past_max =
        prefill_attention_mask_columns - context_size;
      fill_attention_mask_with_prev_length(
        context_size, prefill_attention_mask_columns,
        std::min(kv_len, prefill_full_past_max), attention_mask);
    }
    fill_attention_mask_with_length(context_size,
                                    prefill_sliding_attention_mask_columns,
                                    chunk_len, sliding_attention_mask);
    {
      const int prefill_sliding_past_max =
        prefill_sliding_attention_mask_columns - context_size;
      fill_attention_mask_with_prev_length(
        context_size, prefill_sliding_attention_mask_columns,
        std::min(kv_len, prefill_sliding_past_max), sliding_attention_mask);
    }

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * swa_pos_dim,
                65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * swa_pos_dim,
                32768);

    if (kv_len + chunk_len > rope_cache_seq_len)
      throw std::runtime_error("Prefill position out of rope cache");

    auto pos_off = kv_len * pos_dim;
    auto swa_pos_off = kv_len * swa_pos_dim;
    std::memcpy(prefill_position_ids_cos, position_ids_cos + pos_off,
                chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_position_ids_sin, position_ids_sin + pos_off,
                chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_cos,
                swa_position_ids_cos + swa_pos_off,
                chunk_len * swa_pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_sin,
                swa_position_ids_sin + swa_pos_off,
                chunk_len * swa_pos_dim * sizeof(uint16_t));

    fill_prefill_ple_chunk_(_input, c, chunk_len);

    outputs = run_qnn_inference(prefill_model, 1, prefill_inputs,
                                models[prefill_graph].graph_info);

    // ── Debug: dump raw prefill output for value layer 0 (first chunk only) ──
    if (c == 0) {
      for (const auto &b : prefill_output_kv_bindings) {
        if (b.is_key || b.layer_index != 0)
          continue;
        auto out = std::get<uint8_t *>(outputs[b.output_index]);
        std::cout << "\n=== RAW PREFILL OUTPUT for value L0 (first 10 bytes of "
                     "first 5 rows) ==="
                  << std::endl;
        for (int r = 0; r < 5 && r < chunk_len; ++r) {
          std::cout << "row[" << r << "][0..9]: ";
          for (int h = 0; h < 10; ++h) {
            std::cout << (int)out[r * 256 + h] << " ";
          }
          std::cout << std::endl;
        }
      }
    }

    {
      std::vector<uint8_t *> kv_ptrs;
      for (auto *kv : kvs)
        kv_ptrs.push_back((uint8_t *)kv);
      append_outputs_to_kv_cache(outputs, prefill_output_kv_bindings, kv_ptrs,
                                 kv_row_lengths, kv_len, chunk_len,
                                 context_size, prefill_graph, &kv_columns);
    }
    kv_len += chunk_len;
  }

  // ── Debug: Post-prefill KV cache state ──
  std::cout << "\n=== Post-prefill KV cache debug ===" << std::endl;
  std::cout << "KV cache layers: " << kvs.size() / 2 << std::endl;
  std::cout << "kv_len after prefill: " << kv_len << std::endl;

  // Dump entire KV cache for layer 0
  {
    int layer = 0;
    uint8_t *key_ptr = (uint8_t *)kvs[layer * 2];
    uint8_t *val_ptr = (uint8_t *)kvs[layer * 2 + 1];
    int row_len = kv_row_lengths[layer];
    int col = kv_columns[layer];
    std::cout << "\n=== Layer 0 FULL KV CACHE DUMP ===" << std::endl;
    std::cout << "Key layout: [head_dim=" << col << ", seq_len=" << row_len
              << "]" << std::endl;
    std::cout << "Value layout: [seq_len=" << row_len << ", head_dim=" << col
              << "]" << std::endl;

    // Dump key cache for first 5 positions
    std::cout << "\n--- Key cache (first 5 positions, first 10 head_dim values "
                 "each) ---"
              << std::endl;
    for (int pos = 0; pos < 5 && pos < kv_len; ++pos) {
      std::cout << "key[" << pos << "][0..9]: ";
      for (int h = 0; h < 10 && h < col; ++h) {
        std::cout << (int)key_ptr[h * row_len + pos] << " ";
      }
      std::cout << std::endl;
    }

    // Dump value cache for first 5 positions
    std::cout << "\n--- Value cache (first 5 positions, first 10 head_dim "
                 "values each) ---"
              << std::endl;
    for (int pos = 0; pos < 5 && pos < kv_len; ++pos) {
      std::cout << "value[" << pos << "][0..9]: ";
      for (int h = 0; h < 10 && h < col; ++h) {
        std::cout << (int)val_ptr[pos * col + h] << " ";
      }
      std::cout << std::endl;
    }
  }

  // ── Generation ──
  auto end_prefill = std::chrono::system_clock::now();
  auto start = std::chrono::system_clock::now();
  int idx;
  int prefill_len = kv_len;
  // Tokens decoded-but-not-yet-streamed because they end mid-character.
  std::vector<int32_t> pending_stream;
  for (idx = prefill_len; idx < generation_full_kv_past_length; ++idx) {
    fill_generation_inputs(
      generation_sample, token, generation_attention_mask,
      generation_attention_mask_elements, generation_sliding_attention_mask,
      generation_sliding_attention_mask_elements,
      generation_full_kv_past_length, generation_sliding_kv_past_length,
      generation_position_ids_cos, generation_position_ids_sin,
      position_ids_cos, position_ids_sin, pos_dim,
      generation_swa_position_ids_cos, generation_swa_position_ids_sin,
      swa_position_ids_cos, swa_position_ids_sin, swa_pos_dim, idx,
      rope_cache_seq_len);
    fill_generation_ple_(token);

    // ── Debug: dump generation inputs for first 3 steps ──
    if (idx - prefill_len < 3) {
      std::cout << "\n=== GENERATION STEP " << (idx - prefill_len)
                << " (position=" << idx << ", token=" << token
                << ") ===" << std::endl;

      // 1. Token input
      std::cout << "generation_sample[0] = " << generation_sample[0]
                << std::endl;

      // 2. Attention mask summary
      int full_mask_count = 0, sliding_mask_count = 0;
      for (int i = 0; i < generation_attention_mask_elements; ++i)
        if (generation_attention_mask[i] != 0)
          ++full_mask_count;
      for (int i = 0; i < generation_sliding_attention_mask_elements; ++i)
        if (generation_sliding_attention_mask[i] != 0)
          ++sliding_mask_count;
      std::cout << "attention_mask active: " << full_mask_count << "/"
                << generation_attention_mask_elements << std::endl;
      std::cout << "sliding_attention_mask active: " << sliding_mask_count
                << "/" << generation_sliding_attention_mask_elements
                << std::endl;

      // 3. Position IDs (first 10 values)
      std::cout << "position_ids_cos[0..9]: ";
      for (int i = 0; i < pos_dim; ++i)
        std::cout << generation_position_ids_cos[i] << " ";
      std::cout << std::endl;
      std::cout << "position_ids_sin[0..9]: ";
      for (int i = 0; i < pos_dim; ++i)
        std::cout << generation_position_ids_sin[i] << " ";
      std::cout << std::endl;

      // 4. SWA Position IDs (first 10 values)
      std::cout << "swa_position_ids_cos[0..9]: ";
      for (int i = 0; i < swa_pos_dim; ++i)
        std::cout << generation_swa_position_ids_cos[i] << " ";
      std::cout << std::endl;
      std::cout << "swa_position_ids_sin[0..9]: ";
      for (int i = 0; i < swa_pos_dim; ++i)
        std::cout << generation_swa_position_ids_sin[i] << " ";
      std::cout << std::endl;

      // 5. PLE input for first layer (first 10 values)
      if (!generation_per_layer_dst_.empty()) {
        std::cout << "PLE layer0[0..9]: ";
        for (int i = 0; i < 10; ++i)
          std::cout << generation_per_layer_dst_[0][i] << " ";
        std::cout << std::endl;
        std::cout << "PLE layer0 scale=" << generation_per_layer_scale_[0]
                  << " offset=" << generation_per_layer_offset_[0] << std::endl;
      }

      // 6. KV cache first layer first column (first 16 bytes at current
      // position)
      if (!kvs.empty() && kv_row_lengths.size() > 0) {
        uint8_t *kv0 = (uint8_t *)kvs[0];
        int row_len = kv_row_lengths[0];
        int col = kv_columns.size() > 0 ? kv_columns[0] : 0;
        std::cout << "KV[0] layer0: row_len=" << row_len << " col=" << col
                  << std::endl;
        std::cout << "KV[0] at position " << idx << " (bytes): ";
        for (int i = 0; i < 16 && (idx * col + i) < kv_sizes[0]; ++i)
          std::cout << (int)kv0[idx * col + i] << " ";
        std::cout << std::endl;
      }

      std::cout << "=== END DEBUG STEP " << (idx - prefill_len)
                << " ===" << std::endl;
    }

    outputs = run_qnn_inference(generation_model, 1, generation_inputs,
                                models[generation_graph].graph_info);

    // ── Debug: dump generation model outputs for first 3 steps ──
    if (idx - prefill_len < 3) {
      std::cout << "\n=== GENERATION MODEL OUTPUTS for step "
                << (idx - prefill_len) << " ===" << std::endl;

      // Dump KV outputs for layer 0 (key and value)
      for (const auto &b : generation_output_kv_bindings) {
        if (b.layer_index != 0)
          continue;
        auto out = std::get<uint8_t *>(outputs[b.output_index]);
        std::cout << "Output " << (b.is_key ? "KEY" : "VALUE")
                  << " L0 first 16 bytes: ";
        for (int i = 0; i < 16; ++i) {
          std::cout << (int)out[i] << " ";
        }
        std::cout << std::endl;
      }

      // Dump logits (first 10 values)
      auto logits =
        std::get<uint16_t *>(outputs[generation_logits_output_index]);
      std::cout << "Logits first 10 values (uint16): ";
      for (int i = 0; i < 10; ++i) {
        std::cout << logits[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "=== END GENERATION OUTPUTS ===" << std::endl;
    }

    {
      std::vector<uint8_t *> kv_ptrs;
      for (auto *kv : kvs)
        kv_ptrs.push_back((uint8_t *)kv);
      append_outputs_to_kv_cache(outputs, generation_output_kv_bindings,
                                 kv_ptrs, kv_row_lengths, idx, 1, 1,
                                 generation_graph, &kv_columns);
    }
    kv_len += 1;
    token = ::sample(
      std::get<uint16_t *>(outputs[generation_logits_output_index]), vocab_size,
      _input.data(), _input.size(), logit_scale, logit_offset,
      repetition_penalty, temperature, top_p, top_k, final_logit_softcapping);
    output.push_back(token);

    bool reached_eos = false;
    for (auto eos : eos_tokens) {
      if (token == eos) {
        reached_eos = true;
        break;
      }
    }
    if (reached_eos) {
      append_generation_token_to_kv_cache(token);
      break;
    }

    _input.push_back(token);

    // Accumulate and decode the whole pending run so a multi-byte character
    // (e.g. a Korean syllable split across byte-fallback tokens) is held until
    // complete instead of streaming a half-formed glyph. last_output_ and the
    // streamer only receive text once the character finishes. See
    // utf8_stream_util.h.
    pending_stream.push_back(token);
    std::string decoded = tokenizer->Decode(pending_stream);
    LOGD("%d : %s", token, decoded.c_str());
    if (!utf8stream::shouldHold(decoded, pending_stream.size())) {
      pending_stream.clear();
      last_output_ += decoded;
      if (streamer_) {
        if (streamer_put(streamer_, decoded.c_str()) != 0) {
          stop_requested_.store(true, std::memory_order_release);
          break;
        }
      } else if (log_output) {
        std::cout << decoded << std::flush;
      }
    }

    if (stop_requested_.load(std::memory_order_acquire))
      break;
  }

  // Flush any tail held while waiting for an incomplete character to finish.
  if (!pending_stream.empty()) {
    std::string tail = tokenizer->Decode(pending_stream);
    if (!tail.empty()) {
      last_output_ += tail;
      if (streamer_)
        streamer_put(streamer_, tail.c_str());
      else if (log_output)
        std::cout << tail << std::flush;
    }
  }

  if (streamer_)
    streamer_end(streamer_);
  auto end = std::chrono::system_clock::now();
  this->raw_exec_seconds = end - start;

  auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_prefill - start_prefill)
                      .count();
  auto gen_ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  unsigned int generated_tokens =
    (idx > prefill_len) ? static_cast<unsigned int>(idx - prefill_len) : 0U;

  performance_metrics.prefill_tokens = static_cast<unsigned int>(input_len);
  performance_metrics.prefill_duration_ms = static_cast<double>(prefill_ms);
  performance_metrics.generation_tokens = generated_tokens;
  performance_metrics.generation_duration_ms = static_cast<double>(gen_ms);
  performance_metrics.total_duration_ms =
    static_cast<double>(prefill_ms + gen_ms);
  performance_metrics.peak_memory_kb = getPeakMemoryKb();

  has_run_ = true;
  conversation_started_ = true;

  if (log_output) {
    std::cout << "\n\nGeneration exec_time : " << this->raw_exec_seconds.count()
              << ", token per second: "
              << generated_tokens / this->raw_exec_seconds.count()
              << ", token generation time average: "
              << this->raw_exec_seconds.count() /
                   std::max(1, static_cast<int>(generated_tokens))
              << std::endl;
  }
}
