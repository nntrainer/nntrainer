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
#include <layer_prof.h>
#include <memory_data.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <vector>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>

// Both helpers below are reached only from the FP16 embedding paths, whose own
// guards read `ENABLE_CUDA && ENABLE_FP16`. This guard has to match those call
// sites exactly: with CUDA on and FP16 off the definitions would have no
// caller, and -Werror=unused-function fails that build. Upstream CI runs no
// CUDA job, so only a local cuda=true / opencl=false / fp16=false build sees
// it.
#if defined(ENABLE_FP16)
namespace {
// NNTR_CUDA_ASYNC guard for the pinned embedding staging buffers: in async
// mode nothing drains the stream per-op, so the NEXT token's host dequant can
// rewrite (or cudaFreeHost) emb_stage while the PREVIOUS token's H2D from the
// same buffer is still in flight -> the consumer kernel reads torn rows
// (field: word-salad decode under ASYNC=1, coherent under sync). One event on
// the single backend stream marks the most recent staging H2D; stream FIFO
// means "last H2D done" implies every earlier one is done, so a single shared
// event safely guards both instances (embedding0 + per_layer_input_embedding).
// Skipped during graph capture: an in-capture cudaEventSynchronize is illegal
// and the captured H2D is replay-ordered by the graph itself.
cudaEvent_t g_emb_h2d_evt = nullptr;
bool g_emb_h2d_pending = false;

void emb_stage_h2d_record() {
  auto &sm = nntrainer::cuda::StreamManager::Global();
  if (sm.isCapturing())
    return;
  if (g_emb_h2d_evt == nullptr &&
      cudaEventCreateWithFlags(&g_emb_h2d_evt, cudaEventDisableTiming) !=
        cudaSuccess) {
    g_emb_h2d_evt = nullptr;
    cudaGetLastError();
    return;
  }
  if (cudaEventRecord(g_emb_h2d_evt, sm.GetStream()) == cudaSuccess)
    g_emb_h2d_pending = true;
  else
    cudaGetLastError();
}

void emb_stage_h2d_wait() {
  if (!g_emb_h2d_pending ||
      nntrainer::cuda::StreamManager::Global().isCapturing())
    return;
  cudaEventSynchronize(g_emb_h2d_evt);
  g_emb_h2d_pending = false;
}
} // namespace
#endif // ENABLE_FP16
#endif // ENABLE_CUDA

#include "../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#include <fcntl.h>        // _O_RDONLY, _O_BINARY
#include <io.h>           // _wopen, _close
#include <mman_windows.h> // mmap/munmap (MapViewOfFile), PROT_READ, MAP_PRIVATE
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum EmbeddingParams { weight };

namespace {

std::mutex quant_lut_cache_mutex;
std::unordered_map<std::string, std::weak_ptr<QuantLut>> quant_lut_cache;

bool hasJsonExtension(const std::string &path) {
  return std::filesystem::path(path).extension() == ".json";
}

std::filesystem::path resolveLutPath(const std::string &manifest_path,
                                     const std::string &lut_path) {
  std::filesystem::path path(lut_path);
  if (path.is_absolute())
    return path;

  return std::filesystem::path(manifest_path).parent_path() / path;
}

/**
 * @brief Attach the file's contents to the LUT — mmap'd read-only where
 *        possible so the table pages in on demand instead of residing in
 *        memory; falls back to a full read into lut.bytes.
 */
void attachPayload(QuantLut &lut, const std::filesystem::path &path);

std::vector<uint8_t> readBinaryFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  NNTR_THROW_IF(!file.is_open(), std::runtime_error)
    << "Failed to open LUT file: " << path.string();

  const auto pos = file.tellg();
  NNTR_THROW_IF(pos < 0, std::runtime_error)
    << "Failed to get LUT file size: " << path.string();

  const auto size = static_cast<size_t>(pos);
  std::vector<uint8_t> bytes(size);

  file.seekg(0, std::ios::beg);
  if (size > 0) {
    file.read(reinterpret_cast<char *>(bytes.data()),
              static_cast<std::streamsize>(size));
    NNTR_THROW_IF(static_cast<size_t>(file.gcount()) != size,
                  std::runtime_error)
      << "Failed to read complete LUT file: " << path.string();
  }

  return bytes;
}

void attachPayload(QuantLut &lut, const std::filesystem::path &path) {
#if !defined(_WIN32)
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd >= 0) {
    struct stat st {};
    if (::fstat(fd, &st) == 0 && st.st_size > 0) {
      void *ptr = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                         MAP_PRIVATE, fd, 0);
      if (ptr != MAP_FAILED) {
        // Token-id lookups are random access; don't let readahead pull the
        // whole table into the page cache.
        ::madvise(ptr, static_cast<size_t>(st.st_size), MADV_RANDOM);
        ::close(fd); // mapping keeps its own reference
        lut.mmap_ptr = ptr;
        lut.mmap_len = static_cast<size_t>(st.st_size);
        return;
      }
    }
    ::close(fd);
  }
#else
  // Windows: map the sidecar with MapViewOfFile via the mman shim
  // (utils/mman_windows.h) instead of slurping it whole. MapViewOfFile faults
  // pages on demand -- no whole-file readahead -- so the random token-id
  // lookups keep only the touched rows resident, the same win MADV_RANDOM gives
  // on POSIX (the shim has no madvise, and none is needed for that on-demand
  // behaviour). Without this, readBinaryFile below pulled the entire sidecar
  // into RAM (e.g. ~1 GB of PLE + ~0.4 GB of embedding), defeating the point of
  // -side.
  std::error_code ec;
  const auto fsize = std::filesystem::file_size(path, ec);
  if (!ec && fsize > 0) {
    int fd = ::_wopen(path.wstring().c_str(), _O_RDONLY | _O_BINARY);
    if (fd >= 0) {
      void *ptr = ::mmap(nullptr, static_cast<size_t>(fsize), PROT_READ,
                         MAP_PRIVATE, fd, 0);
      ::_close(fd); // the view keeps its own file-mapping reference
      if (ptr != MAP_FAILED) {
        lut.mmap_ptr = ptr;
        lut.mmap_len = static_cast<size_t>(fsize);
        return;
      }
    }
  }
#endif
  lut.bytes = readBinaryFile(path);
}

const nlohmann::json &requireJsonObjectField(const nlohmann::json &json,
                                             const char *field,
                                             const std::string &path) {
  NNTR_THROW_IF(!json.contains(field) || !json.at(field).is_object(),
                std::runtime_error)
    << "Malformed LUT manifest " << path << ": expected object field '" << field
    << "'";
  return json.at(field);
}

std::string requireJsonStringField(const nlohmann::json &json,
                                   const char *field, const std::string &path) {
  NNTR_THROW_IF(!json.contains(field) || !json.at(field).is_string(),
                std::runtime_error)
    << "Malformed LUT manifest " << path << ": expected string field '" << field
    << "'";
  return json.at(field).get<std::string>();
}

float requireJsonFloatField(const nlohmann::json &json, const char *field,
                            const std::string &path) {
  NNTR_THROW_IF(!json.contains(field) || !json.at(field).is_number(),
                std::runtime_error)
    << "Malformed LUT manifest " << path << ": expected numeric field '"
    << field << "'";
  return json.at(field).get<float>();
}

int requireJsonIntField(const nlohmann::json &json, const char *field,
                        const std::string &path) {
  NNTR_THROW_IF(!json.contains(field) || !(json.at(field).is_number_integer() ||
                                           json.at(field).is_number_unsigned()),
                std::runtime_error)
    << "Malformed LUT manifest " << path << ": expected integer field '"
    << field << "'";

  const long long value = json.at(field).get<long long>();
  NNTR_THROW_IF(value < std::numeric_limits<int>::min() ||
                  value > std::numeric_limits<int>::max(),
                std::runtime_error)
    << "Malformed LUT manifest " << path << ": integer field '" << field
    << "' is out of int range";
  return static_cast<int>(value);
}

size_t requireJsonSizeField(const nlohmann::json &json, const char *field,
                            const std::string &path) {
  NNTR_THROW_IF(!json.contains(field) || !(json.at(field).is_number_integer() ||
                                           json.at(field).is_number_unsigned()),
                std::runtime_error)
    << "Malformed LUT manifest " << path << ": expected integer field '"
    << field << "'";

  const long long value = json.at(field).get<long long>();
  NNTR_THROW_IF(value <= 0, std::invalid_argument)
    << "Malformed LUT manifest " << path << ": field '" << field
    << "' must be positive";
  return static_cast<size_t>(value);
}

void derivePacked4BitDimensions(QuantLut &lut,
                                const std::string &manifest_path) {
  NNTR_THROW_IF(lut.out_dim == 0, std::invalid_argument)
    << "Malformed LUT manifest " << manifest_path
    << ": size/out_dim must be positive";
  NNTR_THROW_IF(lut.out_dim % 2 != 0, std::invalid_argument)
    << "Malformed LUT manifest " << manifest_path
    << ": 4-bit packed LUT requires even out_dim, got " << lut.out_dim;

  const size_t bytes_per_row = lut.out_dim / 2;
  NNTR_THROW_IF(lut.payload_size() == 0 ||
                  lut.payload_size() % bytes_per_row != 0,
                std::runtime_error)
    << "LUT binary size " << lut.payload_size()
    << " is not consistent with out_dim=" << lut.out_dim;

  lut.in_dim = lut.payload_size() / bytes_per_row;
  NNTR_THROW_IF(lut.in_dim == 0, std::runtime_error)
    << "LUT binary has no rows: " << manifest_path;
}

std::shared_ptr<QuantLut> loadUfixed8Manifest(const std::string &manifest_path,
                                              const nlohmann::json &json) {
  const auto lut_path = requireJsonStringField(json, "lut-path", manifest_path);
  const auto &quant_param =
    requireJsonObjectField(json, "quant-param", manifest_path);

  auto lut = std::make_shared<QuantLut>();
  lut->out_dim = requireJsonSizeField(json, "size", manifest_path);
  lut->scale = requireJsonFloatField(quant_param, "scale", manifest_path);
  lut->offset = requireJsonIntField(quant_param, "offset", manifest_path);
  lut->is_raw_u16 = false;
  lut->is_signed4 = false;
  attachPayload(*lut, resolveLutPath(manifest_path, lut_path));

  derivePacked4BitDimensions(*lut, manifest_path);
  return lut;
}

std::shared_ptr<QuantLut> loadSfixed4Manifest(const std::string &manifest_path,
                                              const nlohmann::json &json) {
  const auto lut_path = requireJsonStringField(json, "lut-path", manifest_path);
  const auto &quant_param =
    requireJsonObjectField(json, "quant-param", manifest_path);
  NNTR_THROW_IF(!quant_param.contains("scale") ||
                  !quant_param.at("scale").is_array(),
                std::runtime_error)
    << "Malformed LUT manifest " << manifest_path
    << ": sfixed4 expects quant-param.scale array";

  auto lut = std::make_shared<QuantLut>();
  lut->out_dim = requireJsonSizeField(json, "size", manifest_path);
  lut->is_raw_u16 = false;
  lut->is_signed4 = true;
  attachPayload(*lut, resolveLutPath(manifest_path, lut_path));
  lut->row_scales.reserve(quant_param.at("scale").size());

  for (const auto &scale : quant_param.at("scale")) {
    NNTR_THROW_IF(!scale.is_number(), std::runtime_error)
      << "Malformed LUT manifest " << manifest_path
      << ": sfixed4 row scale must be numeric";
    lut->row_scales.push_back(scale.get<float>());
  }

  derivePacked4BitDimensions(*lut, manifest_path);
  NNTR_THROW_IF(lut->row_scales.size() != lut->in_dim, std::invalid_argument)
    << "sfixed4 row scale count " << lut->row_scales.size()
    << " does not match in_dim " << lut->in_dim << " for " << manifest_path;

  return lut;
}

/**
 * @brief GGML row-block sidecar: the payload is the byte-identical Q4_0/Q6_K
 *        row table an in-bin embedding weight would hold, so decode reuses
 *        dequantize_row_q{4_0,6_K} and the outputs match the in-bin path
 *        bit-exactly. Manifest:
 *          {"datatype": "q4_0"|"q6_k", "size": <out_dim>,
 *           "rows": <in_dim, optional>, "lut-path": "<payload>"}
 */
std::shared_ptr<QuantLut> loadGgmlManifest(const std::string &manifest_path,
                                           const nlohmann::json &json,
                                           nntrainer::TensorDim::DataType dt) {
  const auto lut_path = requireJsonStringField(json, "lut-path", manifest_path);

  auto lut = std::make_shared<QuantLut>();
  lut->out_dim = requireJsonSizeField(json, "size", manifest_path);
  lut->ggml_dtype = dt;

  const size_t block = (dt == nntrainer::TensorDim::DataType::Q6_K) ? 256 : 32;
  const size_t block_bytes =
    (dt == nntrainer::TensorDim::DataType::Q6_K) ? 210 : 18;
  NNTR_THROW_IF(lut->out_dim % block != 0, std::invalid_argument)
    << "Malformed LUT manifest " << manifest_path << ": size " << lut->out_dim
    << " must be a multiple of the " << block << "-wide quant block";
  lut->row_bytes = block_bytes * (lut->out_dim / block);

  attachPayload(*lut, resolveLutPath(manifest_path, lut_path));
  NNTR_THROW_IF(lut->payload_size() == 0 ||
                  lut->payload_size() % lut->row_bytes != 0,
                std::runtime_error)
    << "LUT binary size " << lut->payload_size()
    << " is not consistent with row stride " << lut->row_bytes << " for "
    << manifest_path;
  lut->in_dim = lut->payload_size() / lut->row_bytes;

  if (json.contains("rows")) {
    const size_t rows = requireJsonSizeField(json, "rows", manifest_path);
    NNTR_THROW_IF(rows != lut->in_dim, std::invalid_argument)
      << "LUT manifest " << manifest_path << " declares rows=" << rows
      << " but payload holds " << lut->in_dim;
  }
  return lut;
}

std::shared_ptr<QuantLut> loadJsonManifest(const std::string &manifest_path) {
  std::ifstream file(manifest_path);
  NNTR_THROW_IF(!file.is_open(), std::runtime_error)
    << "Failed to open LUT manifest: " << manifest_path;

  nlohmann::json json;
  try {
    file >> json;
  } catch (const nlohmann::json::exception &e) {
    std::ostringstream ss;
    ss << "Malformed LUT manifest " << manifest_path << ": " << e.what();
    throw std::runtime_error(ss.str());
  }

  NNTR_THROW_IF(!json.is_object(), std::runtime_error)
    << "Malformed LUT manifest " << manifest_path
    << ": top-level JSON must be an object";

  const std::string datatype =
    json.contains("datatype")
      ? requireJsonStringField(json, "datatype", manifest_path)
      : std::string("ufixed8");

  if (datatype == "ufixed8")
    return loadUfixed8Manifest(manifest_path, json);
  if (datatype == "sfixed4")
    return loadSfixed4Manifest(manifest_path, json);
  if (datatype == "q4_0")
    return loadGgmlManifest(manifest_path, json,
                            nntrainer::TensorDim::DataType::Q4_0);
  if (datatype == "q6_k")
    return loadGgmlManifest(manifest_path, json,
                            nntrainer::TensorDim::DataType::Q6_K);

  NNTR_THROW_IF(true, std::runtime_error)
    << "Unsupported LUT datatype '" << datatype << "' in " << manifest_path
    << " (expected ufixed8, sfixed4, q4_0, or q6_k)";
  return nullptr;
}

std::shared_ptr<QuantLut> loadRawU16(const std::string &path,
                                     size_t in_dim_hint, size_t out_dim_hint) {
  NNTR_THROW_IF(in_dim_hint == 0 || out_dim_hint == 0, std::invalid_argument)
    << "Raw UINT16 LUT requires non-zero in_dim/out_dim hints";
  NNTR_THROW_IF(in_dim_hint > std::numeric_limits<size_t>::max() /
                                out_dim_hint / sizeof(uint16_t),
                std::overflow_error)
    << "Raw UINT16 LUT size overflows size_t for " << path;

  const size_t expected_size = in_dim_hint * out_dim_hint * sizeof(uint16_t);
  auto lut = std::make_shared<QuantLut>();
  attachPayload(*lut, path);
  NNTR_THROW_IF(lut->payload_size() != expected_size, std::runtime_error)
    << "Raw UINT16 LUT file size " << lut->payload_size()
    << " does not match in_dim*out_dim*2 (" << expected_size << ") for "
    << path;

  lut->in_dim = in_dim_hint;
  lut->out_dim = out_dim_hint;
  lut->is_raw_u16 = true;
  return lut;
}

void validateHintedDimensions(const QuantLut &lut, const std::string &path,
                              size_t in_dim_hint, size_t out_dim_hint) {
  NNTR_THROW_IF(in_dim_hint != 0 && lut.in_dim != in_dim_hint,
                std::invalid_argument)
    << "LUT in_dim mismatch for " << path << ": expected " << in_dim_hint
    << ", file has " << lut.in_dim;
  NNTR_THROW_IF(out_dim_hint != 0 && lut.out_dim != out_dim_hint,
                std::invalid_argument)
    << "LUT out_dim mismatch for " << path << ": expected " << out_dim_hint
    << ", file has " << lut.out_dim;
}

int decodeSigned4(uint8_t nibble) {
  nibble &= 0x0fU;
  return (nibble & 0x08U) ? static_cast<int>(nibble) - 16
                          : static_cast<int>(nibble);
}

uint16_t clampFloatToU16(float value) {
  if (!std::isfinite(value))
    return value > 0.0f ? std::numeric_limits<uint16_t>::max() : 0;

  if (value <= 0.0f)
    return 0;
  if (value >= static_cast<float>(std::numeric_limits<uint16_t>::max()))
    return std::numeric_limits<uint16_t>::max();
  return static_cast<uint16_t>(value);
}

uint16_t clampRoundedToU16(double value) {
  if (!std::isfinite(value))
    return value > 0.0 ? std::numeric_limits<uint16_t>::max() : 0;

  if (value <= 0.0)
    return 0;
  if (value >= static_cast<double>(std::numeric_limits<uint16_t>::max()))
    return std::numeric_limits<uint16_t>::max();
  return static_cast<uint16_t>(value);
}

void validateDecodeArgs(const QuantLut &lut, size_t token_idx,
                        size_t output_len) {
  NNTR_THROW_IF(token_idx >= lut.in_dim, std::invalid_argument)
    << "input word index is greater than in_dim";
  NNTR_THROW_IF(output_len != lut.out_dim, std::invalid_argument)
    << "LUT decode output length " << output_len << " does not match out_dim "
    << lut.out_dim;
}

float decodePacked4BitValue(const QuantLut &lut, size_t token_idx,
                            uint8_t nibble, float layer_scale) {
  if (lut.is_signed4) {
    NNTR_THROW_IF(lut.row_scales.size() != lut.in_dim, std::runtime_error)
      << "sfixed4 LUT row scale count does not match in_dim";
    return static_cast<float>(decodeSigned4(nibble)) *
           lut.row_scales[token_idx] * layer_scale;
  }

  return (static_cast<float>(nibble & 0x0fU) + static_cast<float>(lut.offset)) *
         lut.scale * layer_scale;
}

template <typename T>
void decodePacked4BitRowToFloatType(const QuantLut &lut, size_t token_idx,
                                    float layer_scale, T *output,
                                    size_t output_len) {
  validateDecodeArgs(lut, token_idx, output_len);
  NNTR_THROW_IF(lut.is_raw_u16, std::runtime_error)
    << "Raw UINT16 LUT cannot be decoded to floating-point output";
  NNTR_THROW_IF(lut.out_dim % 2 != 0, std::runtime_error)
    << "4-bit packed LUT requires even out_dim, got " << lut.out_dim;

  const size_t bytes_per_row = lut.out_dim / 2;
  const uint8_t *row = lut.data() + token_idx * bytes_per_row;

  for (size_t i = 0; i < bytes_per_row; ++i) {
    const uint8_t byte = row[i];
    output[i * 2] = static_cast<T>(
      decodePacked4BitValue(lut, token_idx, byte & 0x0fU, layer_scale));
    output[i * 2 + 1] = static_cast<T>(
      decodePacked4BitValue(lut, token_idx, byte >> 4, layer_scale));
  }
}

} // namespace

QuantLut::~QuantLut() {
  // ::munmap resolves to the POSIX call or the mman_windows shim
  // (UnmapViewOfFile) depending on platform; both accept (ptr, len).
  if (mmap_ptr)
    ::munmap(mmap_ptr, mmap_len);
}

std::shared_ptr<QuantLut> get_or_load_quant_lut(const std::string &path,
                                                size_t in_dim_hint,
                                                size_t out_dim_hint) {
  std::lock_guard<std::mutex> lock(quant_lut_cache_mutex);

  auto cached = quant_lut_cache.find(path);
  if (cached != quant_lut_cache.end()) {
    if (auto lut = cached->second.lock()) {
      validateHintedDimensions(*lut, path, in_dim_hint, out_dim_hint);
      return lut;
    }
    quant_lut_cache.erase(cached);
  }

  auto lut = hasJsonExtension(path)
               ? loadJsonManifest(path)
               : loadRawU16(path, in_dim_hint, out_dim_hint);
  validateHintedDimensions(*lut, path, in_dim_hint, out_dim_hint);
  quant_lut_cache[path] = lut;
  return lut;
}

void decode_quant_lut_row_to_fp32(const QuantLut &lut, size_t token_idx,
                                  float layer_scale, float *output,
                                  size_t output_len) {
  decodePacked4BitRowToFloatType(lut, token_idx, layer_scale, output,
                                 output_len);
}

void decode_quant_lut_row_to_uint16(const QuantLut &lut, size_t token_idx,
                                    float layer_scale, uint16_t *output,
                                    size_t output_len) {
  validateDecodeArgs(lut, token_idx, output_len);

  if (lut.is_raw_u16) {
    const uint16_t *row =
      reinterpret_cast<const uint16_t *>(lut.data()) + token_idx * lut.out_dim;
    std::memcpy(output, row, lut.out_dim * sizeof(uint16_t));
    return;
  }

  NNTR_THROW_IF(lut.out_dim % 2 != 0, std::runtime_error)
    << "4-bit packed LUT requires even out_dim, got " << lut.out_dim;

  const size_t bytes_per_row = lut.out_dim / 2;
  const uint8_t *row = lut.data() + token_idx * bytes_per_row;

  for (size_t i = 0; i < bytes_per_row; ++i) {
    const uint8_t byte = row[i];
    output[i * 2] = clampFloatToU16(
      decodePacked4BitValue(lut, token_idx, byte & 0x0fU, layer_scale));
    output[i * 2 + 1] = clampFloatToU16(
      decodePacked4BitValue(lut, token_idx, byte >> 4, layer_scale));
  }
}

void decode_quant_lut_row_to_uint16(const QuantLut &lut, size_t token_idx,
                                    float layer_scale, float output_quant_scale,
                                    int output_quant_offset, uint16_t *output,
                                    size_t output_len) {
  validateDecodeArgs(lut, token_idx, output_len);

  if (lut.is_raw_u16) {
    decode_quant_lut_row_to_uint16(lut, token_idx, layer_scale, output,
                                   output_len);
    return;
  }

  NNTR_THROW_IF(output_quant_scale <= 0.0f, std::invalid_argument)
    << "output_quant_scale must be positive";
  NNTR_THROW_IF(lut.out_dim % 2 != 0, std::runtime_error)
    << "4-bit packed LUT requires even out_dim, got " << lut.out_dim;

  const size_t bytes_per_row = lut.out_dim / 2;
  const uint8_t *row = lut.data() + token_idx * bytes_per_row;

  for (size_t i = 0; i < bytes_per_row; ++i) {
    const uint8_t byte = row[i];
    const float lo =
      decodePacked4BitValue(lut, token_idx, byte & 0x0fU, layer_scale);
    const float hi =
      decodePacked4BitValue(lut, token_idx, byte >> 4, layer_scale);

    output[i * 2] = clampRoundedToU16(
      std::round(static_cast<double>(lo) / output_quant_scale) -
      output_quant_offset);
    output[i * 2 + 1] = clampRoundedToU16(
      std::round(static_cast<double>(hi) / output_quant_scale) -
      output_quant_offset);
  }
}

EmbeddingLayer::EmbeddingLayer() :
  LayerImpl(),
  embedding_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
                  nntrainer::props::Scale(), props::QuantizedLutPath(),
                  props::OutputQuantScale(), props::OutputQuantOffset(),
                  props::SidecarExportPath()),
  weight_idx(std::numeric_limits<unsigned>::max()) {}

void EmbeddingLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Embedding layer takes only one input";

  // Token IDs are integers — embedding caller provides FP32 input. The
  // historical "must be FP32" throw is removed so FP16-activation models still
  // construct (the lookup expects integer-valued data). [merge 2026-06-30: keep
  // OURS (no FP32-input throw) + adopt upstream's quantized-LUT sidecar
  // wiring.]
  auto &quantized_lut_path = std::get<props::QuantizedLutPath>(embedding_props);
  const bool has_quantized_lut = !quantized_lut_path.empty();
  context.setInputDataType(nntrainer::TensorDim::DataType::FP32);

  const nntrainer::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "Embedding layer takes only one for channel size";

  // [merge 2026-06-30] OURS: no FP32-input throw — our FP16-activation models
  // build input0 as plain FP16-default (no explicit FP32 input layer).
  // [merge 2026-07-08] upstream d87ff6dd3 pins the input DTYPE to FP32
  // unconditionally (kept above): token ids > 2048 are not exactly
  // representable in FP16 and our lookup reads ids via getAddress<float>.
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

  quant_lut.reset();
  if (has_quantized_lut) {
    quant_lut =
      get_or_load_quant_lut(quantized_lut_path.get(), in_dim, out_dim);
    NNTR_THROW_IF(quant_lut->in_dim != in_dim, std::invalid_argument)
      << "LUT in_dim mismatch: layer=" << in_dim
      << ", file=" << quant_lut->in_dim;
    NNTR_THROW_IF(quant_lut->out_dim != out_dim, std::invalid_argument)
      << "LUT out_dim mismatch: layer=" << out_dim
      << ", file=" << quant_lut->out_dim;
    NNTR_THROW_IF(quant_lut->is_raw_u16 &&
                    context.getActivationDataType() !=
                      nntrainer::TensorDim::DataType::UINT16,
                  std::invalid_argument)
      << "Raw UINT16 LUT requires UINT16 activation/output dtype";
  }

  nntrainer::TensorDim output_dim = input_dim;

  // output_dim expected as hidden x num input (batch size)
  output_dim.height(input_dim.width());
  output_dim.width(out_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  if (quant_lut)
    return;

  nntrainer::TensorDim dim = output_dim;

  dim.setTensorType({context.getFormat(), context.getWeightDataType()});

  dim.batch(1);

  /**
   * @note nntrainer's per-channel quantized tensor is in following shape
   * - quantized data: (H, W)
   * - scales: (W)
   *
   * For embedding, there are in_dim elements in scale.
   * So we should use (out_dim, in_dim) dimension although actual tensor
   * shape is (in_dim, out_dim)
   *
   * @todo Add other types that needs to be transposed
   * @todo Allow other axis can be a quantization axis
   */
  if (context.getWeightDataType() == nntrainer::TensorDim::DataType::QS4CX) {
    dim.height(out_dim);
    dim.width(in_dim);
  } else {
    dim.height(in_dim);
    dim.width(out_dim);
  }

  weight_idx = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void EmbeddingLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, embedding_props);
  LayerImpl::setProperty(remain_props);
}

void EmbeddingLayer::forwardSidecarLut(nntrainer::RunLayerContext &context,
                                       unsigned int from, unsigned int to) {
  NNTR_THROW_IF(!quant_lut, std::runtime_error)
    << "Embedding sidecar LUT is not loaded";
  NNTR_THROW_IF(quant_lut->ggml_dtype != nntrainer::TensorDim::DataType::NONE,
                std::runtime_error)
    << "GGML sidecar LUT must be decoded by incremental_forwarding";
  NNTR_THROW_IF(to < from, std::invalid_argument)
    << "Embedding incremental range is invalid";

  const unsigned int out_dim =
    std::get<nntrainer::props::OutDim>(embedding_props);
  const unsigned int iter = to - from;
  const float scale =
    std::get<nntrainer::props::Scale>(embedding_props).empty()
      ? 1.0f
      : std::get<nntrainer::props::Scale>(embedding_props).get();
  auto &output_quant_scale = std::get<props::OutputQuantScale>(embedding_props);
  auto &output_quant_offset =
    std::get<props::OutputQuantOffset>(embedding_props);
  const bool has_output_quant_scale = !output_quant_scale.empty();
  const float out_scale =
    has_output_quant_scale ? output_quant_scale.get() : 0.0f;
  const int out_offset =
    output_quant_offset.empty() ? 0 : output_quant_offset.get();

  NNTR_THROW_IF(has_output_quant_scale && out_scale <= 0.0f,
                std::invalid_argument)
    << "output_quant_scale must be positive";

  nntrainer::Tensor &hidden = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const auto output_dtype = hidden.getDataType();
  const unsigned int batch_size = input.batch();

  NNTR_THROW_IF(quant_lut->is_raw_u16 &&
                  output_dtype != nntrainer::TensorDim::DataType::UINT16,
                std::runtime_error)
    << "Raw UINT16 LUT requires UINT16 output dtype";

  auto &tm = nntrainer::ThreadManager::Global();

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    const float *input_data =
      input.getAddress<float>(batch * input.getDim().getFeatureLen());
    nntrainer::Tensor batch_hidden = hidden.getBatchSlice(batch, 1);

    tm.parallel_for(0, static_cast<size_t>(iter), [&](size_t i) {
      const size_t token_idx = static_cast<size_t>(input_data[i]);
      const size_t output_offset = static_cast<size_t>(out_dim) * i;

      if (output_dtype == nntrainer::TensorDim::DataType::UINT16) {
        auto output = batch_hidden.getData<uint16_t>() + output_offset;
        if (has_output_quant_scale) {
          decode_quant_lut_row_to_uint16(*quant_lut, token_idx, scale,
                                         out_scale, out_offset, output,
                                         out_dim);
        } else {
          decode_quant_lut_row_to_uint16(*quant_lut, token_idx, scale, output,
                                         out_dim);
        }
        return;
      }

      NNTR_THROW_IF(quant_lut->is_raw_u16, std::runtime_error)
        << "Raw UINT16 LUT requires UINT16 output dtype";

      if (output_dtype == nntrainer::TensorDim::DataType::FP32) {
        auto output = batch_hidden.getData<float>() + output_offset;
        decode_quant_lut_row_to_fp32(*quant_lut, token_idx, scale, output,
                                     out_dim);
        return;
      }

#ifdef ENABLE_FP16
      if (output_dtype == nntrainer::TensorDim::DataType::FP16) {
        auto output = batch_hidden.getData<_FP16>() + output_offset;
        decodePacked4BitRowToFloatType(*quant_lut, token_idx, scale, output,
                                       out_dim);
        return;
      }
#endif

      throw std::runtime_error(
        "Embedding sidecar LUT does not support output dtype");
    });
  }
}

void EmbeddingLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  nntrainer::LayerProfScope _prof("embedding_fwd", false);
  if (quant_lut) {
    nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
    if (quant_lut->ggml_dtype != nntrainer::TensorDim::DataType::NONE)
      incremental_forwarding(context, 0, input.width(), training);
    else
      forwardSidecarLut(context, 0, input.width());
  }
}

void EmbeddingLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  nntrainer::LayerProfScope _prof("embedding", (to - from) == 1);

  /// @todo get input and output dimension from input_ and hidden itself
  unsigned int in_dim = std::get<nntrainer::props::InDim>(embedding_props);
  unsigned int out_dim = std::get<nntrainer::props::OutDim>(embedding_props);
  float scale = std::get<nntrainer::props::Scale>(embedding_props).empty()
                  ? 1.0f
                  : std::get<nntrainer::props::Scale>(embedding_props).get();
  unsigned int _from = from;

  const bool ggml_lut =
    quant_lut && quant_lut->ggml_dtype != nntrainer::TensorDim::DataType::NONE;
  if (quant_lut && !ggml_lut) {
    forwardSidecarLut(context, from, to);
    return;
  }

  // A GGML-format sidecar (q4_0/q6_k manifest) is decoded by this SAME loop
  // as the in-bin weight — identical row bytes, identical dequant — so every
  // backend handoff below (CUDA dev-act staging, SVM/clmem raise) covers both
  // sources; only the row base pointer differs (mmap'd file vs weight tensor).
  nntrainer::Tensor *weight_p =
    ggml_lut ? nullptr : &context.getWeight(weight_idx);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  nntrainer::TensorDim out_tensor_dim =
    nntrainer::TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

  unsigned int b_size = input_.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    float *in_data =
      input_.getAddress<float>(b * input_.getDim().getFeatureLen());
    nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

    int iter = to - from;

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // Device-only activation pool (NNTR_CUDA_DEV_ACT): the PLE output is real
    // device memory (not host-addressable). Dequant into a host staging buffer
    // and push it H2D on the backend stream -- the CUDA mirror of the
    // device upload the OpenCL residency path performs for its own pool.
    // Persistent + PINNED host staging (was a local std::vector). Under
    // CUDA-graph stream capture a local vector fails twice: (a) a pageable
    // cudaMemcpyAsync is NOT capturable, and (b) the vector is freed when this
    // function returns, but the captured graph REPLAYS afterwards -- it would
    // copy from freed memory => garbage. A layer-lifetime pinned
    // (cudaHostAlloc) buffer is capturable and survives the replay. PER
    // INSTANCE (member, NOT a function static): embedding0 and the PLE both run
    // this method now, and a shared static let the PLE overwrite embedding0's
    // still-in-flight async copy (CUDA garbage). Grows monotonically (decode
    // iter==1; prefill iter<=max_seq_len); single sequence (b_size==1).
    _FP16 *&emb_stage = *reinterpret_cast<_FP16 **>(&cuda_stage);
    size_t &emb_stage_cap = cuda_stage_cap;
    bool emb_dev_only = false;
    if (nntrainer::cuda::engine_selected() &&
        hidden_.getDataType() == nntrainer::TensorDim::DataType::FP16) {
      cudaPointerAttributes pa{};
      emb_dev_only =
        cudaPointerGetAttributes(&pa, batchsliced_hidden.getData<_FP16>()) ==
          cudaSuccess &&
        pa.type == cudaMemoryTypeDevice;
      cudaGetLastError();
      if (emb_dev_only) {
        // Async-mode: the previous token's H2D from this pinned buffer may
        // still be in flight -- wait before the host rewrites or frees it.
        emb_stage_h2d_wait();
        size_t need = (size_t)iter * out_dim;
        if (need > emb_stage_cap) {
          if (emb_stage)
            cudaFreeHost(emb_stage);
          cudaHostAlloc((void **)&emb_stage, need * sizeof(_FP16),
                        cudaHostAllocDefault);
          emb_stage_cap = need;
        }
      }
    }
#endif

    const auto wt = ggml_lut ? quant_lut->ggml_dtype : weight_p->getDataType();
    const bool row_quant = (wt == nntrainer::TensorDim::DataType::Q6_K ||
                            wt == nntrainer::TensorDim::DataType::Q4_0);
    // Per-channel 4-bit symmetric embedding table (upstream QS4CX support).
    // finalize() stores it transposed as (out_dim, in_dim) so the per-channel
    // scale vector has in_dim entries; the row decoder takes the token id as
    // the channel index and out_dim as the row length.
    const bool qs4cx_w =
      !ggml_lut && wt == nntrainer::TensorDim::DataType::QS4CX;
    NNTR_THROW_IF(ggml_lut && !row_quant, std::runtime_error)
      << "GGML sidecar LUT supports only Q4_0/Q6_K payloads";
    const uint8_t *quant_table =
      row_quant ? (ggml_lut ? quant_lut->data() : weight_p->getData<uint8_t>())
                : nullptr;
    const size_t row_stride =
      (wt == nntrainer::TensorDim::DataType::Q6_K)
        ? 210 * ((static_cast<size_t>(out_dim) + 255) / 256)
        : 18 * ((static_cast<size_t>(out_dim) + 31) / 32);

#if !defined(_WIN32)
    // Cold-start I/O for the mmap'd sidecar: MADV_RANDOM disabled readahead,
    // so a ~1K-token prefill would otherwise pay ~1K synchronous major faults
    // serialized inside the workers (measured ~100-160ms on NVMe). Ask the
    // kernel to fault this batch's exact rows in asynchronously up front;
    // out-of-range ids are skipped here and rejected in the compute loop.
    if (ggml_lut && quant_lut->mmap_ptr && iter > 1) {
      static const uintptr_t pg_mask =
        ~static_cast<uintptr_t>(sysconf(_SC_PAGESIZE) - 1);
      for (int pi = 0; pi < iter; ++pi) {
        const size_t idx = static_cast<size_t>(in_data[pi]);
        if (idx >= in_dim)
          continue;
        const uint8_t *row = quant_table + row_stride * idx;
        const uintptr_t start = reinterpret_cast<uintptr_t>(row) & pg_mask;
        const uintptr_t end = reinterpret_cast<uintptr_t>(row) + row_stride;
        ::madvise(reinterpret_cast<void *>(start), end - start, MADV_WILLNEED);
      }
    }
#endif

    auto &tm = nntrainer::ThreadManager::Global();
    const size_t total = static_cast<size_t>(iter);
    const size_t max_workers = std::max<size_t>(1, tm.getComputeThreadCount());
    const size_t njobs = std::min(total, max_workers);
    tm.parallel_for(0, njobs, [&](size_t t) {
      // Chunked over tokens (prefill: total == prompt length) so each worker
      // reuses ONE fp32 scratch for its whole chunk instead of paying a heap
      // allocation per token; decode (total==1) stays on the caller thread.
      const size_t chunk_begin = t * total / njobs;
      const size_t chunk_end = (t + 1) * total / njobs;
      std::vector<float> tmp;
      if (row_quant || qs4cx_w)
        tmp.resize(out_dim);
      for (size_t i = chunk_begin; i < chunk_end; ++i) {
        size_t embed_idx = static_cast<size_t>(in_data[i]);
        if (embed_idx >= in_dim) {
          throw std::invalid_argument(
            "input word index is greater than in_dim");
        }

        nntrainer::Tensor out_tensor =
          batchsliced_hidden.getSharedDataTensor(out_tensor_dim, out_dim * (i));

        if (row_quant || qs4cx_w) {
          // dequantize_row_q{6_K,4_0,s4cx} ALWAYS writes out_dim FP32 values.
          // In an FP16-activation run out_tensor is FP16, so writing FP32
          // straight in (the old `out_tensor.getData()` == float*) overruns the
          // buffer 2x and corrupts every value => garbage PLE row added to
          // every layer => prompt-independent garbage output. Mirror
          // TieWordEmbedding: dequant into an FP32 scratch, then cast into the
          // output's real dtype, folding the embed scale.
          if (qs4cx_w) {
            nntrainer::dequantize_row_qs4cx(embed_idx, out_dim,
                                            weight_p->getData(),
                                            weight_p->getScale(), tmp.data());
          } else {
            const void *row = quant_table + row_stride * embed_idx;
            if (wt == nntrainer::TensorDim::DataType::Q6_K)
              nntrainer::dequantize_row_q6_K(const_cast<void *>(row),
                                             tmp.data(), out_dim);
            else
              nntrainer::dequantize_row_q4_0(const_cast<void *>(row),
                                             tmp.data(), out_dim);
          }
          if (out_tensor.getDataType() ==
              nntrainer::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
            _FP16 *o =
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
              emb_dev_only ? (emb_stage + (size_t)i * out_dim) :
#endif
                           out_tensor.getData<_FP16>();
            for (unsigned int k = 0; k < (unsigned int)out_dim; ++k)
              o[k] = static_cast<_FP16>(tmp[k] * scale);
#else
            throw std::invalid_argument(
              "FP16 out_tensor requires ENABLE_FP16");
#endif
          } else {
            float *o = out_tensor.getData<float>();
            for (unsigned int k = 0; k < (unsigned int)out_dim; ++k)
              o[k] = tmp[k] * scale;
          }
        } else if (wt == nntrainer::TensorDim::DataType::FP32 &&
                   out_tensor.getDataType() ==
                     nntrainer::TensorDim::DataType::FP16) {
          // FP32 weight row -> FP16 activation needs an explicit narrowing
          // cast; copyData would byte-copy out_dim*4 bytes into an out_dim*2
          // buffer.
#ifdef ENABLE_FP16
          nntrainer::Tensor cur_weight =
            weight_p->getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);
          const float *src = cur_weight.getData<float>();
          _FP16 *o =
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
            emb_dev_only ? (emb_stage + (size_t)i * out_dim) :
#endif
                         out_tensor.getData<_FP16>();
          for (unsigned int k = 0; k < (unsigned int)out_dim; ++k)
            o[k] = static_cast<_FP16>(src[k] * scale);
#else
          throw std::invalid_argument("FP16 out_tensor requires ENABLE_FP16");
#endif
        } else {
          nntrainer::Tensor cur_weight =
            weight_p->getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);
          out_tensor.copyData(cur_weight);
          if (scale != 1.0f) {
            out_tensor.multiply_i(scale);
          }
        }
      }
    });

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // Push the host-dequantized rows into the device-only output on the
    // backend stream, ordered before the GPU consumer reads them.
    if (emb_dev_only) {
      // Windows defaults to a fully synchronous upload. The asynchronous H2D
      // copy under a device-only activation pool was measured as a source of
      // run-to-run divergence there, while the synchronous copy costs nothing
      // measurable (one ~4 KB per-token DMA) and generated identical text
      // across every battery. NNTR_CUDA_EMB_SYNCCOPY=0 restores the
      // asynchronous copy for an A/B; other platforms keep it asynchronous.
      static const bool emb_synccopy = []() {
        const char *e = std::getenv("NNTR_CUDA_EMB_SYNCCOPY");
        if (e)
          return e[0] == '1';
#ifdef _WIN32
        return true;
#else
        return false;
#endif
      }();
      if (emb_synccopy &&
          !nntrainer::cuda::StreamManager::Global().isCapturing()) {
        cudaMemcpy(batchsliced_hidden.getData<_FP16>(), emb_stage,
                   (size_t)iter * out_dim * sizeof(_FP16),
                   cudaMemcpyHostToDevice);
      } else {
        cudaMemcpyAsync(batchsliced_hidden.getData<_FP16>(), emb_stage,
                        (size_t)iter * out_dim * sizeof(_FP16),
                        cudaMemcpyHostToDevice,
                        nntrainer::cuda::StreamManager::Global().GetStream());
        emb_stage_h2d_record();
      }
    }
#endif

#ifdef DEBUG
    std::cout << context.getName() << " : "
              << "\n input:" << input_ << "\n hidden: " << hidden_ << std::endl;
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
  // Sidecar extraction (nntr_quantize --ple_sidecar): this layer's table goes
  // to its own file — raw rows, no header; the manifest JSON is written by the
  // caller — and NOTHING is written to the model file, matching the load side
  // (quantized_lut_path => finalize requests no weight, so the bin must not
  // contain these bytes).
  auto &export_path = std::get<props::SidecarExportPath>(embedding_props);
  std::ofstream sidecar;
  std::ofstream &out = export_path.empty() ? file : sidecar;
  if (!export_path.empty()) {
    sidecar.open(export_path.get(), std::ios::binary | std::ios::trunc);
    NNTR_THROW_IF(!sidecar.is_open(), std::runtime_error)
      << "Failed to open sidecar export file: " << export_path.get();
  }

  // @note shared weights are only be saved at the first access
  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (run_context.isGradientFirstAccess(i)) {
      auto &weight = run_context.getWeight(i);
      if (dtype == nntrainer::TensorDim::DataType::NONE ||
          weight.getDataType() == dtype)
        weight.save(out);
      else {
        NNTR_THROW_IF(weight.getDataType() !=
                        nntrainer::TensorDim::DataType::FP32,
                      std::runtime_error)
          << "Save with quantization only supports for FP32 weight.";
        ///@note The codelines below can be replaced with quantizer's
        /// quantize()
        nntrainer::TensorDim dim = weight.getDim();
        size_t K = dim.height();
        size_t N = dim.width();

        if (dtype == nntrainer::TensorDim::DataType::Q4_0) {
          // NOTE: the former "skip quantization when K == 1" bypass was
          // removed upstream — it silently wrote FP32 bytes into a Q4_0 bin
          // and corrupted the file. Callers pick the dtype per tensor via the
          // quantize.cpp dtype_map instead.
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
          quant_weight.save(out);
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
          quant_weight.save(out);
        } else if (dtype == nntrainer::TensorDim::DataType::QS4CX) {
          /**
           * @note QS4CX tensor uses N as the number of scale.
           * In finalize(), we passed in_dim as width() which equals to number
           * of scale. So we don't need to swap N and K.
           */

          const size_t data_size = N * ((K + 1) / 2);
          const size_t scale_size = N * sizeof(float);

          // allocate packed size, not an unpacked size
          std::vector<uint8_t> rhs_q(data_size + scale_size);
          uint8_t *data = rhs_q.data();
          uint8_t *scale = data + data_size;

          nntrainer::quant_qs4cx_f32(N, K, weight.getData(), data, scale, true);
          out.write((const char *)data, data_size + scale_size);
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
