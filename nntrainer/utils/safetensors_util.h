// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  Util helpers for the safetensors format.
 * @file   safetensors_util.h
 * @date   18 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __SAFETENSORS_UTIL_H__
#define __SAFETENSORS_UTIL_H__
#ifdef __cplusplus

#include <map>
#include <string>
#include <tensor_dim.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nntrainer::safetensors {

/**
 * @brief Metadata entry for a single tensor in a safetensors file.
 *
 * @note For block-quantized tensors (Q4_0 / Q4_K / Q6_K) the standard
 * @c dtype is written as "U8" and @c shape carries the raw byte length, so
 * generic safetensors parsers can still read the payload as an opaque blob.
 * The native nntrainer data type and the pre-quantization (logical) shape are
 * preserved in the @c nntr_dtype and @c nntr_shape extension fields.
 */
struct TensorEntry {
  std::string name;
  std::string dtype;
  std::vector<size_t> shape;
  size_t offset_start;
  size_t offset_end;
  /** nntrainer-native dtype name ("Q4_0", "Q4_K", ...); empty when the
   *  standard @c dtype already describes the tensor faithfully. */
  std::string nntr_dtype;
  /** logical (pre-quantization) shape; empty for non-quantized tensors. */
  std::vector<size_t> nntr_shape;
};

/**
 * @brief Map an nntrainer DataType to the standard safetensors dtype string.
 * @note Block-quantized types map to "U8" (opaque byte blob).
 */
const char *dtypeToString(ml::train::TensorDim::DataType dtype);

/**
 * @brief Map an nntrainer DataType to its native name ("FP32", "Q4_0", ...).
 */
const char *nntrDtypeName(ml::train::TensorDim::DataType dtype);

/**
 * @brief Parse a native nntrainer dtype name back into a DataType.
 * @throw std::invalid_argument when the name is unknown.
 */
ml::train::TensorDim::DataType nntrDtypeFromName(const std::string &name);

/**
 * @brief Whether the data type is a block-quantized type (Q4_0/Q4_K/Q6_K).
 */
bool isQuantized(ml::train::TensorDim::DataType dtype);

/**
 * @brief Build a safetensors header JSON string from tensor entries.
 */
std::string buildHeader(const std::vector<TensorEntry> &entries);

/**
 * @brief Build a safetensors header JSON string with extra @c __metadata__.
 * @param entries  per-tensor entries
 * @param metadata extra string key/value pairs to embed under __metadata__
 */
std::string buildHeader(const std::vector<TensorEntry> &entries,
                        const std::map<std::string, std::string> &metadata);

/**
 * @brief Parse a safetensors header, returning name -> (offset_start, size).
 * @note Kept for the weight-loading path which only needs byte offsets.
 */
std::unordered_map<std::string, std::pair<size_t, size_t>>
parseHeader(const std::string &json);

/**
 * @brief Fully parse a safetensors header into per-tensor entries (including
 *        the @c nntr_dtype / @c nntr_shape extension fields).
 */
std::vector<TensorEntry> parseHeaderEntries(const std::string &json);

/**
 * @brief Parse only the @c __metadata__ block of a safetensors header.
 */
std::map<std::string, std::string> parseMetadata(const std::string &json);

/**
 * @brief Read the @c __metadata__ block of the safetensors file at @a path
 *        (the header only, not the data).
 * @throw std::runtime_error when the file or its header cannot be read
 */
std::map<std::string, std::string> readMetadata(const std::string &path);

/**
 * @brief Produce a human-readable summary of a safetensors header (metadata
 *        plus a per-tensor table) for inspection without loading weights.
 */
std::string inspect(const std::string &json);

} // namespace nntrainer::safetensors

#endif /* __cplusplus */
#endif /* __SAFETENSORS_UTIL_H__ */
