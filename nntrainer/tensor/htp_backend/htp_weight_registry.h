// SPDX-License-Identifier: Apache-2.0
/** @file htp_weight_registry.h
 * @brief Model-lifetime host weight bindings and DSP handle indices.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_HTP_WEIGHT_REGISTRY_H
#define NNTR_HTP_WEIGHT_REGISTRY_H
#include <algorithm>
#include <htp_weight_io.h>
#include <htp_wh_layout.h>
#include <limits>
#include <mutex>
#include <q4_gemv_cpu_det.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
namespace nntrainer {
class HtpWeightRegistry {
public:
  HtpWeightRegistry() = default;
  HtpWeightRegistry(const HtpWeightRegistry &) = delete;
  HtpWeightRegistry &operator=(const HtpWeightRegistry &) = delete;
  ~HtpWeightRegistry() {
    if (fcwh_fd_ >= 0)
      ::close(fcwh_fd_);
  }
  struct ConvHandles {
    std::vector<uint32_t> h_in; /**< a, b, c */
    uint32_t h_out = 0;
  };
  struct FcHandles {
    std::vector<uint32_t> handles;
    std::vector<unsigned int> cols; /**< N of each handle */
  };
  struct DenseHandles {
    std::vector<uint32_t> h_gu, h_dn;
    unsigned int w = 0; /**< columns of I per chunk */
  };
  struct StaticQuant {
    float scale;
    int32_t zp;
  };
  std::mutex handle_mutex_;
  std::unordered_map<const void *, uint32_t> handle_cache_;
  /** FC weights by data pointer -> their handle(s); see get_or_register_fc.
   *  Values are never erased or moved, so the references it hands out stay
   *  valid (std::unordered_map keeps node addresses across rehash). */
  std::unordered_map<const void *, FcHandles> fc_cache_;
  /** Dense FFNs by their up weight's pointer; see get_or_register_dense. */
  std::unordered_map<const void *, DenseHandles> dense_cache_;
  /** Conv blocks by their in_proj's pointer; see get_or_register_conv. */
  std::unordered_map<const void *, ConvHandles> conv_cache_;
  /** The FC WH sidecar (set_fc_wh_file): its fd, path and index by
   *  fcWhKey, and the bytes its images took in the arena and on the heap. */
  int fcwh_fd_ = -1;
  std::string fcwh_name_;
  std::unordered_map<uint64_t, FcWhEntry> fcwh_;
  /** Main-file WH records keyed by their tensor data pointer. */
  std::unordered_map<const void *, FcWhEntry> direct_wh_;
  size_t fcwh_arena_bytes_ = 0, fcwh_heap_bytes_ = 0;
  /** WH bytes of the sidecar's images not registered yet, plus
   *  placeExistingOn's 4 KiB alignment for four handles an image (its most:
   *  a dense FFN's chunks). */
  size_t fcwhLeft() const {
    size_t all = 0;
    for (const auto &e : fcwh_)
      all += whBytes(e.second.K, e.second.N) + 4u * 4096u;
    for (const auto &e : direct_wh_)
      all +=
        whBytes(e.second.K, e.second.N) +
        ((e.second.N + fcSliceCols(e.second.K) - 1u) / fcSliceCols(e.second.K) +
         4u) *
          4096u;
    const size_t done = fcwh_arena_bytes_ + fcwh_heap_bytes_;
    return all > done ? all - done : 0u;
  }
  /** Handles registered from the sidecar: the FC slices plus the dense
   *  chunks' two each. */
  size_t fcwhHandles() const {
    if (fcwh_fd_ < 0)
      return 0;
    size_t n = 0;
    for (const auto &f : fc_cache_)
      n += f.second.handles.size();
    for (const auto &d : dense_cache_)
      n += d.second.h_gu.size() + d.second.h_dn.size();
    return n;
  }

  std::unordered_map<const void *, StaticQuant> static_quant_;
  static unsigned int fcSliceCols(unsigned int K) {
    // 2 MiB slices, 1 MiB from K = 4096 up. The slice's double buffer is
    // what the activation rows share VTCM with (fcMaxRows): at 2 MiB the
    // o-proj's 4096-deep weight left 896 rows a call, two calls for a
    // 1024-row prefill; at 1 MiB it is 1,472 and one call (the same bytes
    // stream through VTCM in narrower slices).
    // K = 2816 at 1 MiB would make the qkv call one for 2048
    // rows too (2,368 a call), but the MoE call after it then fails with
    // AEE_ENOMEMORY on the S26 Ultra at C = 32 (DSP memory, not yet found
    // which); the qkv call stays two calls there.
    const unsigned int kSliceBytes = K >= 4096u ? 1u << 20 : 2u << 20;
    const unsigned int k_tiles = K / 32u;
    const unsigned int n_tiles =
      k_tiles == 0 ? 0 : (kSliceBytes / 512u) / k_tiles;
    const unsigned int cols = n_tiles < 1u ? 32u : n_tiles * 32u;
    // A slice holds whole heads, so a per-head norm folded into the call
    // (mm_u8i4_layer_norm's post chunks) never straddles two handles: 512
    // columns is the widest head dim here and divides every smaller one.
    // a head dim that does not divide 512 needs its own
    // alignment passed down, not this constant.
    return cols >= 512u ? (cols / 512u) * 512u : cols;
  }
  const FcWhEntry *fcwhFind(const void *q4, uint32_t K, uint32_t N) const {
    const auto direct = direct_wh_.find(q4);
    if (direct != direct_wh_.end()) {
      if (direct->second.K != K || direct->second.N != N)
        throw std::invalid_argument(
          "QS4CX_WH FC source shape mismatch: requested " + std::to_string(K) +
          "x" + std::to_string(N) + " registered " +
          std::to_string(direct->second.K) + "x" +
          std::to_string(direct->second.N));
      return &direct->second;
    }
    if (fcwh_fd_ < 0)
      return nullptr;
    const size_t len = static_cast<size_t>(N) * (K / 32u) * Q4_CPU_BLOCK_BYTES;
    auto it = fcwh_.find(fcWhKey(q4, len));
    if (it == fcwh_.end() || it->second.K != K || it->second.N != N) {
      throw std::runtime_error(
        "fc_wh_file_name " + fcwh_name_ + ": no image for a " +
        std::to_string(K) + "x" + std::to_string(N) +
        " Q4_0 weight of the model file -- the sidecar is not the one "
        "nntr_quantize_stream --fc_wh_sidecar wrote with this model file");
    }
    return &it->second;
  }
  void fcwhRead(const FcWhEntry *entry, void *dst, size_t bytes,
                uint64_t offset) const {
    const auto direct = direct_wh_.find(
      reinterpret_cast<const void *>(static_cast<uintptr_t>(entry->key)));
    if (direct != direct_wh_.end() && &direct->second == entry) {
      if (offset > entry->bytes || bytes > entry->bytes - offset)
        throw std::out_of_range("QS4CX_WH FC source read outside record");
      std::memcpy(dst, static_cast<const uint8_t *>(direct->first) + offset,
                  bytes);
    } else {
      throwPread(preadAll(fcwh_fd_, dst, bytes, offset), "fc wh image");
    }
  }
  bool set_fc_wh_source(const void *data, unsigned K, unsigned N,
                        bool affine = false) {
    if (!data || K == 0 || N == 0 || K % WH_TILE || N % WH_TILE)
      throw std::invalid_argument("QS4CX_WH FC requires complete WH tiles");
    std::lock_guard<std::mutex> lock(handle_mutex_);
    FcWhEntry entry{};
    entry.K = K;
    entry.N = N;
    entry.key = reinterpret_cast<uintptr_t>(data);
    entry.bytes = whBytes(K, N) + (affine ? 12u : 8u) * size_t(N);
    auto inserted = direct_wh_.emplace(data, entry);
    if (!inserted.second &&
        (inserted.first->second.K != K || inserted.first->second.N != N ||
         inserted.first->second.bytes != entry.bytes))
      throw std::invalid_argument("QS4CX_WH FC source shape changed");
    return true;
  }
  bool set_fc_wh_file(const char *path) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    if (fcwh_fd_ >= 0)
      ::close(fcwh_fd_);
    fcwh_fd_ = -1;
    fcwh_.clear();
    fcwh_arena_bytes_ = fcwh_heap_bytes_ = 0; // fcwhLeft's, the banner's
    const std::string where = std::string("fc_wh_file_name ") + path + ": ";
    const int fd = openWeightFile(path);
    if (fd < 0)
      throw std::runtime_error(where + std::strerror(errno));
    std::unordered_map<uint64_t, FcWhEntry> index;
    try {
      const off_t end = ::lseek(fd, 0, SEEK_END);
      const uint64_t size = end < 0 ? 0u : static_cast<uint64_t>(end);
      char magic[sizeof(FCWH_MAGIC)];
      uint32_t vc[2] = {0, 0}; // version, count
      throwPread(preadAll(fd, magic, sizeof(magic), 0), "fc wh header");
      throwPread(preadAll(fd, vc, sizeof(vc), sizeof(magic)), "fc wh header");
      if (std::memcmp(magic, FCWH_MAGIC, sizeof(magic)) != 0 ||
          vc[0] != FCWH_VERSION || fcWhHeaderBytes(vc[1]) > size) {
        throw std::runtime_error("not an FC WH sidecar of version " +
                                 std::to_string(FCWH_VERSION));
      }
      std::vector<FcWhEntry> entries(vc[1]);
      throwPread(
        preadAll(fd, entries.data(), entries.size() * sizeof(FcWhEntry), 16u),
        "fc wh index");
      for (const FcWhEntry &e : entries) {
        const uint64_t want = whBytes(e.K, e.N) + 2u * sizeof(float) * e.N;
        if (e.K % WH_TILE != 0 || e.N % WH_TILE != 0 || e.bytes != want ||
            e.off < fcWhHeaderBytes(vc[1]) || e.off > size ||
            e.bytes > size - e.off || !index.emplace(e.key, e).second) {
          throw std::runtime_error("bad index entry " +
                                   std::string(e.name, strnlen(e.name, 64)));
        }
      }
    } catch (const std::exception &err) {
      ::close(fd);
      throw std::runtime_error(where + err.what());
    }
    fcwh_ = std::move(index);
    fcwh_fd_ = fd;
    fcwh_name_ = path;
    return true;
  }
  bool set_weight_static_quant(const void *key, float scale, int zp) {
    if (!key ||
        !(scale >= std::numeric_limits<float>::min() &&
          scale <= std::numeric_limits<float>::max() / 255.0f) ||
        zp < 0 || zp > 255)
      throw std::invalid_argument("Invalid static U8 encoding");
    std::lock_guard<std::mutex> lock(handle_mutex_);
    static_quant_[key] = {scale, zp};
    return true;
  }
};
} // namespace nntrainer
#endif
