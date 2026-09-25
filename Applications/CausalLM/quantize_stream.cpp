// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   quantize_stream.cpp
 * @brief  Bounded-memory weight quantizer for supported CausalLM models.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cpu_backend.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#endif
#include "json.hpp"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace {

using json = nlohmann::json;

constexpr size_t QK4_0 = 32;
constexpr size_t Q4_0_BLOCK_BYTES = 18;
constexpr size_t QK_K = 256;
constexpr size_t Q4_K_BLOCK_BYTES = 144;
constexpr size_t Q6_K_BLOCK_BYTES = 210;
constexpr size_t MAX_TENSOR_BUFFER_BYTES = 64U * 1024U * 1024U;

using DType = ml::train::TensorDim::DataType;

struct QuantizationPlan {
  DType fc_dtype;
  DType embedding_dtype;
  DType lmhead_dtype;
  ml::train::ISA target_isa;
};

struct Qwen3MoePlan {
  size_t hidden_size;
  size_t vocab_size;
  size_t num_layers;
  size_t num_attention_heads;
  size_t num_key_value_heads;
  size_t head_dim;
  size_t intermediate_size;
  size_t num_experts;
  bool tied_embeddings;
};

struct Gemma4MoePlan {
  size_t hidden_size;
  size_t vocab_size;
  size_t num_layers;
  size_t num_attention_heads;
  size_t num_key_value_heads;
  size_t head_dim;
  size_t global_head_dim;
  size_t num_global_key_value_heads;
  size_t intermediate_size;
  size_t moe_intermediate_size;
  size_t num_experts;
  size_t per_layer_input_size;
  size_t per_layer_vocab_size;
  std::vector<std::string> layer_types;
  bool attention_k_eq_v;
  bool tied_embeddings;
};

std::string upper(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return value;
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

size_t checkedMultiply(size_t lhs, size_t rhs, const std::string &name) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error("Tensor size overflow for " + name);
  }
  return lhs * rhs;
}

size_t tensorElements(size_t rows, size_t columns, const std::string &name) {
  return checkedMultiply(rows, columns, name);
}

size_t tensorBytes(size_t rows, size_t columns, const std::string &name) {
  return checkedMultiply(tensorElements(rows, columns, name), sizeof(float),
                         name);
}

json readJson(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open " + path.string());
  }

  json value;
  file >> value;
  return value;
}

size_t getSize(const json &cfg, const std::string &key) {
  if (!cfg.contains(key) || !cfg[key].is_number_unsigned()) {
    throw std::runtime_error("config.json is missing unsigned numeric key: " +
                             key);
  }
  return cfg[key].get<size_t>();
}

std::string getArchitecture(const json &cfg) {
  if (!cfg.contains("architectures") || !cfg["architectures"].is_array() ||
      cfg["architectures"].empty()) {
    throw std::runtime_error("config.json is missing architectures[0]");
  }
  return cfg["architectures"][0].get<std::string>();
}

Qwen3MoePlan makeQwen3MoePlan(const json &cfg) {
  const std::string architecture = getArchitecture(cfg);
  if (architecture != "Qwen3MoeForCausalLM") {
    throw std::runtime_error(
      "nntr_quantize_stream only supports Qwen3MoeForCausalLM, but got " +
      architecture);
  }

  const size_t hidden_size = getSize(cfg, "hidden_size");
  const size_t num_attention_heads = getSize(cfg, "num_attention_heads");
  if (num_attention_heads == 0) {
    throw std::runtime_error("num_attention_heads must be greater than zero");
  }

  const size_t head_dim = cfg.contains("head_dim")
                            ? getSize(cfg, "head_dim")
                            : hidden_size / num_attention_heads;
  const size_t num_key_value_heads = cfg.contains("num_key_value_heads")
                                       ? getSize(cfg, "num_key_value_heads")
                                       : num_attention_heads;

  if (head_dim == 0 || num_key_value_heads == 0) {
    throw std::runtime_error(
      "head_dim and num_key_value_heads must be greater than zero");
  }

  return {
    hidden_size,
    getSize(cfg, "vocab_size"),
    getSize(cfg, "num_hidden_layers"),
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    getSize(cfg, "moe_intermediate_size"),
    getSize(cfg, "num_experts"),
    cfg.value("tie_word_embeddings", false),
  };
}

Gemma4MoePlan makeGemma4MoePlan(const json &root_cfg) {
  const std::string architecture = getArchitecture(root_cfg);
  if (architecture != "Gemma4ForCausalLM" &&
      architecture != "Gemma4ForConditionalGeneration") {
    throw std::runtime_error("Unsupported Gemma4 architecture: " +
                             architecture);
  }

  const json &cfg =
    root_cfg.contains("text_config") ? root_cfg.at("text_config") : root_cfg;
  if (!cfg.is_object())
    throw std::runtime_error("Gemma4 text_config must be an object");
  if (!cfg.value("enable_moe_block", false)) {
    throw std::runtime_error(
      "Gemma4 streaming quantization requires enable_moe_block=true");
  }

  const size_t hidden_size = getSize(cfg, "hidden_size");
  const size_t num_layers = getSize(cfg, "num_hidden_layers");
  const size_t num_attention_heads = getSize(cfg, "num_attention_heads");
  if (num_attention_heads == 0)
    throw std::runtime_error("num_attention_heads must be greater than zero");

  const size_t head_dim = cfg.contains("head_dim")
                            ? getSize(cfg, "head_dim")
                            : hidden_size / num_attention_heads;
  const size_t num_key_value_heads = cfg.contains("num_key_value_heads")
                                       ? getSize(cfg, "num_key_value_heads")
                                       : num_attention_heads;
  const size_t global_head_dim =
    cfg.contains("global_head_dim") && !cfg["global_head_dim"].is_null()
      ? getSize(cfg, "global_head_dim")
      : head_dim;
  const size_t num_global_key_value_heads =
    cfg.contains("num_global_key_value_heads") &&
        !cfg["num_global_key_value_heads"].is_null()
      ? getSize(cfg, "num_global_key_value_heads")
      : num_key_value_heads;

  const size_t num_kv_shared_layers =
    cfg.value("num_kv_shared_layers", size_t{0});
  if (num_kv_shared_layers != 0) {
    throw std::runtime_error(
      "Gemma4 MoE streaming quantization does not support shared KV layers");
  }

  std::vector<std::string> layer_types(num_layers,
                                       std::string("sliding_attention"));
  if (cfg.contains("layer_types")) {
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();
    if (layer_types.size() != num_layers) {
      throw std::runtime_error(
        "Gemma4 layer_types size must match num_hidden_layers");
    }
  }
  for (const auto &type : layer_types) {
    if (type != "sliding_attention" && type != "full_attention")
      throw std::runtime_error("Unsupported Gemma4 layer type: " + type);
  }

  const size_t per_layer_input_size =
    cfg.value("hidden_size_per_layer_input", size_t{0});
  const size_t per_layer_vocab_size =
    cfg.value("vocab_size_per_layer_input", getSize(cfg, "vocab_size"));
  if (per_layer_input_size != 0 && per_layer_vocab_size == 0) {
    throw std::runtime_error(
      "vocab_size_per_layer_input must be greater than zero");
  }

  const bool tied_embeddings = cfg.contains("tie_word_embeddings")
                                 ? cfg["tie_word_embeddings"].get<bool>()
                                 : root_cfg.value("tie_word_embeddings", true);
  return {
    hidden_size,
    getSize(cfg, "vocab_size"),
    num_layers,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    global_head_dim,
    num_global_key_value_heads,
    getSize(cfg, "intermediate_size"),
    getSize(cfg, "moe_intermediate_size"),
    getSize(cfg, "num_experts"),
    per_layer_input_size,
    per_layer_vocab_size,
    std::move(layer_types),
    cfg.value("attention_k_eq_v", false),
    tied_embeddings,
  };
}

DType parseDType(const std::string &value) {
  const std::string dtype = upper(value);
  if (dtype == "FP32")
    return DType::FP32;
  if (dtype == "Q4_0" || dtype == "Q40")
    return DType::Q4_0;
  if (dtype == "Q4_K" || dtype == "Q4K")
    return DType::Q4_K;
  if (dtype == "Q6_K" || dtype == "Q6K")
    return DType::Q6_K;
  if (dtype == "QS4CX") {
    // TODO: Add QS4CX streaming layout support for Qwen3 MoE weights.
    throw std::invalid_argument(
      "QS4CX is not supported by nntr_quantize_stream yet");
  }
  throw std::invalid_argument("Unsupported dtype: " + value +
                              " (supported: FP32, Q4_0, Q4_K, Q6_K)");
}

const char *dtypeName(DType dtype) {
  switch (dtype) {
  case DType::FP32:
    return "FP32";
  case DType::Q4_0:
    return "Q4_0";
  case DType::Q4_K:
    return "Q4_K";
  case DType::Q6_K:
    return "Q6_K";
  default:
    throw std::invalid_argument("Unknown dtype");
  }
}

std::string dtypeSuffix(DType dtype) {
  std::string suffix = lower(dtypeName(dtype));
  suffix.erase(std::remove(suffix.begin(), suffix.end(), '_'), suffix.end());
  return suffix;
}

ml::train::ISA parseIsa(const std::string &value) {
  const std::string isa = upper(value);
  if (isa == "DEFAULT")
    return ml::train::ISA::DEFAULT;
  if (isa == "X86")
    return ml::train::ISA::X86;
  if (isa == "ARM")
    return ml::train::ISA::ARM;
  throw std::invalid_argument("Unsupported ISA: " + value +
                              " (supported: DEFAULT, X86, ARM)");
}

const char *isaName(ml::train::ISA isa) {
  switch (isa) {
  case ml::train::ISA::DEFAULT:
    return "DEFAULT";
  case ml::train::ISA::X86:
    return "X86";
  case ml::train::ISA::ARM:
    return "ARM";
  }
  throw std::invalid_argument("Unknown ISA");
}

size_t quantizedSize(DType dtype, size_t rows, size_t columns, bool repack,
                     const std::string &name) {
  switch (dtype) {
  case DType::FP32:
    return tensorBytes(rows, columns, name);
  case DType::Q4_0:
    if (columns % QK4_0 != 0 || (repack && rows % QK4_0 != 0)) {
      throw std::invalid_argument(
        name + " Q4_0 shape must have columns divisible by 32" +
        (repack ? " and rows divisible by 32" : ""));
    }
    return checkedMultiply(checkedMultiply(rows, columns / QK4_0, name),
                           Q4_0_BLOCK_BYTES, name);
  case DType::Q4_K:
    if (columns % QK_K != 0 || (repack && rows % 8 != 0)) {
      throw std::invalid_argument(
        name + " Q4_K shape must have columns divisible by 256" +
        (repack ? " and rows divisible by 8" : ""));
    }
    return checkedMultiply(checkedMultiply(rows, columns / QK_K, name),
                           Q4_K_BLOCK_BYTES, name);
  case DType::Q6_K:
    if (columns % QK_K != 0) {
      throw std::invalid_argument(name +
                                  " Q6_K columns must be divisible by 256");
    }
    return checkedMultiply(checkedMultiply(rows, columns / QK_K, name),
                           Q6_K_BLOCK_BYTES, name);
  default:
    break;
  }
  throw std::invalid_argument("Unknown dtype for " + name);
}

class TensorWriter {
public:
  TensorWriter(std::ifstream &input, std::ofstream &output,
               ml::train::ISA target_isa) :
    input_(input), output_(output), target_isa_(target_isa) {}

  void copyFp32(size_t elements, const std::string &name) {
    copyBytes(checkedMultiply(elements, sizeof(float), name), name);
  }

  void discardFp32(size_t elements, const std::string &name) {
    discardBytes(checkedMultiply(elements, sizeof(float), name), name);
  }

  bool hasRemainingBytes() {
    input_.peek();
    const bool remaining = !input_.eof();
    input_.clear();
    return remaining;
  }

  void writeEmbedding(size_t rows, size_t columns, DType dtype,
                      const std::string &name) {
    if (dtype == DType::FP32) {
      copyBytes(tensorBytes(rows, columns, name), name);
      return;
    }
    if (dtype == DType::Q4_K) {
      throw std::invalid_argument(
        "Q4_K embedding is not supported by the CausalLM embedding layer");
    }
    quantizedSize(dtype, rows, columns, false, name);

    const size_t bytes_per_row = tensorBytes(1, columns, name);
    const size_t rows_per_block =
      std::max<size_t>(1, MAX_TENSOR_BUFFER_BYTES / bytes_per_row);

    for (size_t row = 0; row < rows;) {
      const size_t block_rows = std::min(rows_per_block, rows - row);
      std::vector<float> source(tensorElements(block_rows, columns, name));
      readFloats(source, name);
      writeQuantized(source, block_rows, columns, dtype, false, name);
      row += block_rows;
    }
  }

  /**
   * The FP32 NNTrainer file stores FC weights as [input, output]. Quantized FC
   * tensors are stored row-wise as [output, input], so only quantized output
   * is transposed. Large matrices (normally an untied LM head) use a seek-based
   * blocked transpose to cap memory use.
   */
  void writeFc(size_t input_size, size_t output_size, DType dtype,
               const std::string &name) {
    const size_t source_bytes = tensorBytes(input_size, output_size, name);
    if (dtype == DType::FP32) {
      copyBytes(source_bytes, name);
      return;
    }

    // Validate the complete shape before writing any part of this tensor.
    quantizedSize(dtype, output_size, input_size, true, name);

    if (source_bytes <= MAX_TENSOR_BUFFER_BYTES) {
      std::vector<float> source(tensorElements(input_size, output_size, name));
      readFloats(source, name);
      std::vector<float> transposed(source.size());
      for (size_t input = 0; input < input_size; ++input) {
        for (size_t output = 0; output < output_size; ++output) {
          transposed[output * input_size + input] =
            source[input * output_size + output];
        }
      }
      writeQuantized(transposed, output_size, input_size, dtype, true, name);
      return;
    }

    writeBlockedTransposed(input_size, output_size, dtype, name);
  }

  void requireEndOfFile() {
    const std::streampos consumed = input_.tellg();
    input_.peek();
    if (!input_.eof()) {
      throw std::runtime_error(
        "Input weight file has trailing unread bytes after offset " +
        std::to_string(static_cast<std::streamoff>(consumed)));
    }
  }

private:
  void writeBytes(const void *source, size_t bytes, const std::string &name) {
    if (bytes >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      throw std::overflow_error("Write size is too large for " + name);
    }
    output_.write(static_cast<const char *>(source),
                  static_cast<std::streamsize>(bytes));
    if (!output_)
      throw std::runtime_error("Failed to write " + name);
  }

  void copyBytes(size_t bytes, const std::string &name) {
    constexpr size_t COPY_BUFFER_BYTES = 16U * 1024U * 1024U;
    std::vector<char> buffer(std::min(COPY_BUFFER_BYTES, bytes));
    size_t remaining = bytes;
    while (remaining != 0) {
      const size_t chunk = std::min(buffer.size(), remaining);
      readExact(buffer.data(), chunk, name);
      writeBytes(buffer.data(), chunk, name);
      remaining -= chunk;
    }
  }

  void discardBytes(size_t bytes, const std::string &name) {
    constexpr size_t DISCARD_BUFFER_BYTES = 16U * 1024U * 1024U;
    std::vector<char> buffer(std::min(DISCARD_BUFFER_BYTES, bytes));
    size_t remaining = bytes;
    while (remaining != 0) {
      const size_t chunk = std::min(buffer.size(), remaining);
      readExact(buffer.data(), chunk, name);
      remaining -= chunk;
    }
  }

  void readExact(char *destination, size_t bytes, const std::string &name) {
    if (bytes >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      throw std::overflow_error("Read size is too large for " + name);
    }
    input_.read(destination, static_cast<std::streamsize>(bytes));
    if (input_.gcount() != static_cast<std::streamsize>(bytes)) {
      throw std::runtime_error("Unexpected EOF while reading " + name);
    }
  }

  void readFloats(std::vector<float> &destination, const std::string &name) {
    readExact(reinterpret_cast<char *>(destination.data()),
              checkedMultiply(destination.size(), sizeof(float), name), name);
  }

  void writeQuantized(const std::vector<float> &source, size_t rows,
                      size_t columns, DType dtype, bool repack,
                      const std::string &name) {
    const size_t output_size =
      quantizedSize(dtype, rows, columns, repack, name);
    if (dtype == DType::FP32) {
      throw std::invalid_argument("Internal FP32 quantization request for " +
                                  name);
    }

    std::vector<char> quantized(output_size);
    size_t written = 0;
    if (dtype == DType::Q4_0) {
      if (repack) {
        std::vector<char> plain(output_size);
        written = nntrainer::quantize_q4_0(
          source.data(), plain.data(), static_cast<int64_t>(rows),
          static_cast<int64_t>(columns), nullptr);
        nntrainer::repack_q4_0(quantized.data(), plain.data(), output_size,
                               static_cast<unsigned int>(rows),
                               static_cast<unsigned int>(columns), target_isa_);
      } else {
        written = nntrainer::quantize_q4_0(
          source.data(), quantized.data(), static_cast<int64_t>(rows),
          static_cast<int64_t>(columns), nullptr);
      }
    } else if (dtype == DType::Q4_K) {
      if (repack) {
        std::vector<char> plain(output_size);
        written = nntrainer::quantize_q4_K(
          source.data(), plain.data(), static_cast<int64_t>(rows),
          static_cast<int64_t>(columns), nullptr);
        nntrainer::repack_q4_K(quantized.data(), plain.data(), output_size,
                               static_cast<unsigned int>(rows),
                               static_cast<unsigned int>(columns));
      } else {
        written = nntrainer::quantize_q4_K(
          source.data(), quantized.data(), static_cast<int64_t>(rows),
          static_cast<int64_t>(columns), nullptr);
      }
    } else if (dtype == DType::Q6_K) {
      written = nntrainer::quantize_q6_K(
        source.data(), quantized.data(), static_cast<int64_t>(rows),
        static_cast<int64_t>(columns), nullptr);
    }

    if (written != output_size) {
      throw std::runtime_error("Unexpected quantized size for " + name +
                               ": expected " + std::to_string(output_size) +
                               ", got " + std::to_string(written));
    }

    writeBytes(quantized.data(), quantized.size(), name);
  }

  void writeBlockedTransposed(size_t input_size, size_t output_size,
                              DType dtype, const std::string &name) {
    const std::streampos tensor_start = input_.tellg();
    if (tensor_start < 0) {
      throw std::runtime_error("Failed to determine input offset for " + name);
    }

    size_t row_alignment = 1;
    if (dtype == DType::Q4_0)
      row_alignment = 32;
    else if (dtype == DType::Q4_K)
      row_alignment = 8;

    const size_t target_row_bytes = tensorBytes(1, input_size, name);
    size_t rows_per_block =
      std::max(row_alignment, MAX_TENSOR_BUFFER_BYTES / target_row_bytes);
    rows_per_block -= rows_per_block % row_alignment;
    rows_per_block = std::min(rows_per_block, output_size);

    std::vector<float> strip(rows_per_block);
    for (size_t output_begin = 0; output_begin < output_size;) {
      const size_t block_rows =
        std::min(rows_per_block, output_size - output_begin);
      std::vector<float> transposed(
        tensorElements(block_rows, input_size, name));

      for (size_t input = 0; input < input_size; ++input) {
        const size_t element_offset =
          checkedMultiply(input, output_size, name) + output_begin;
        const size_t byte_offset =
          checkedMultiply(element_offset, sizeof(float), name);
        input_.clear();
        input_.seekg(tensor_start + static_cast<std::streamoff>(byte_offset));
        if (!input_) {
          throw std::runtime_error("Failed to seek while reading " + name);
        }
        readExact(reinterpret_cast<char *>(strip.data()),
                  checkedMultiply(block_rows, sizeof(float), name), name);
        for (size_t output = 0; output < block_rows; ++output) {
          transposed[output * input_size + input] = strip[output];
        }
      }

      writeQuantized(transposed, block_rows, input_size, dtype, true, name);
      output_begin += block_rows;
    }

    const size_t source_bytes = tensorBytes(input_size, output_size, name);
    input_.clear();
    input_.seekg(tensor_start + static_cast<std::streamoff>(source_bytes));
    if (!input_) {
      throw std::runtime_error("Failed to advance past " + name);
    }
  }

  std::ifstream &input_;
  std::ofstream &output_;
  ml::train::ISA target_isa_;
};

void validateSourceConfig(const json &nntr_cfg) {
  const auto requireFp32 = [&nntr_cfg](const std::string &key) {
    if (nntr_cfg.contains(key) &&
        upper(nntr_cfg[key].get<std::string>()) != "FP32") {
      throw std::runtime_error("Source " + key + " must be FP32");
    }
  };

  requireFp32("fc_layer_dtype");
  requireFp32("embedding_dtype");
  requireFp32("lmhead_dtype");
  if (nntr_cfg.contains("model_tensor_type") &&
      upper(nntr_cfg["model_tensor_type"].get<std::string>()) != "FP32-FP32") {
    throw std::runtime_error(
      "Source model_tensor_type must be FP32-FP32 for streaming quantization");
  }
}

void writeQwen3Moe(TensorWriter &writer, const Qwen3MoePlan &model,
                   const QuantizationPlan &quant) {
  writer.writeEmbedding(model.vocab_size, model.hidden_size,
                        quant.embedding_dtype, "embedding0");

  const size_t query_width =
    checkedMultiply(model.num_attention_heads, model.head_dim, "query width");
  const size_t kv_width =
    checkedMultiply(model.num_key_value_heads, model.head_dim, "KV width");

  for (size_t layer = 0; layer < model.num_layers; ++layer) {
    const std::string prefix = "layer" + std::to_string(layer);
    writer.copyFp32(model.hidden_size, prefix + "_attention_norm");
    writer.writeFc(model.hidden_size, query_width, quant.fc_dtype,
                   prefix + "_wq");
    writer.copyFp32(model.head_dim, prefix + "_q_norm");
    writer.writeFc(model.hidden_size, kv_width, quant.fc_dtype, prefix + "_wk");
    writer.copyFp32(model.head_dim, prefix + "_k_norm");
    writer.writeFc(model.hidden_size, kv_width, quant.fc_dtype, prefix + "_wv");
    writer.writeFc(query_width, model.hidden_size, quant.fc_dtype,
                   prefix + "_attention_out");
    writer.copyFp32(model.hidden_size, prefix + "_ffn_norm");

    // The router is explicitly FP32 in qwen_moe_layer::finalize(). Keeping it
    // unquantized also avoids invalid Q4_0 shapes such as [hidden, 128].
    writer.copyFp32(
      tensorElements(model.hidden_size, model.num_experts, prefix + "_router"),
      prefix + "_router");

    for (size_t expert = 0; expert < model.num_experts; ++expert) {
      const std::string expert_prefix =
        prefix + "_expert" + std::to_string(expert);
      writer.writeFc(model.hidden_size, model.intermediate_size, quant.fc_dtype,
                     expert_prefix + "_up");
      writer.writeFc(model.hidden_size, model.intermediate_size, quant.fc_dtype,
                     expert_prefix + "_gate");
      writer.writeFc(model.intermediate_size, model.hidden_size, quant.fc_dtype,
                     expert_prefix + "_down");
    }

    std::cout << "  Quantized layer " << layer + 1 << "/" << model.num_layers
              << '\n';
  }

  writer.copyFp32(model.hidden_size, "output_norm");
  if (!model.tied_embeddings) {
    writer.writeFc(model.hidden_size, model.vocab_size, quant.lmhead_dtype,
                   "output_of_causallm");
  }
  writer.requireEndOfFile();
}

void writeGemma4Moe(TensorWriter &writer, const Gemma4MoePlan &model,
                    const QuantizationPlan &quant) {
  writer.writeEmbedding(model.vocab_size, model.hidden_size,
                        quant.embedding_dtype, "embedding0");

  for (size_t layer = 0; layer < model.num_layers; ++layer) {
    const std::string prefix = "layer" + std::to_string(layer);
    const bool is_sliding = model.layer_types[layer] == "sliding_attention";
    const size_t head_dim = is_sliding ? model.head_dim : model.global_head_dim;
    const size_t kv_heads = is_sliding || !model.attention_k_eq_v
                              ? model.num_key_value_heads
                              : model.num_global_key_value_heads;
    const size_t query_width = checkedMultiply(
      model.num_attention_heads, head_dim, prefix + " query width");
    const size_t kv_width =
      checkedMultiply(kv_heads, head_dim, prefix + " KV width");

    writer.copyFp32(model.hidden_size, prefix + "_attention_norm");
    writer.writeFc(model.hidden_size, query_width, quant.fc_dtype,
                   prefix + "_wq");
    writer.copyFp32(head_dim, prefix + "_q_norm");
    writer.writeFc(model.hidden_size, kv_width, quant.fc_dtype, prefix + "_wk");
    writer.copyFp32(head_dim, prefix + "_k_norm");
    if (!model.attention_k_eq_v || is_sliding) {
      writer.writeFc(model.hidden_size, kv_width, quant.fc_dtype,
                     prefix + "_wv");
    }
    writer.writeFc(query_width, model.hidden_size, quant.fc_dtype,
                   prefix + "_attention_out");

    writer.copyFp32(model.hidden_size, prefix + "_post_attention_norm");
    writer.copyFp32(model.hidden_size, prefix + "_pre_ffn_norm");
    writer.writeFc(model.hidden_size, model.intermediate_size, quant.fc_dtype,
                   prefix + "_ffn_gate");
    writer.writeFc(model.hidden_size, model.intermediate_size, quant.fc_dtype,
                   prefix + "_ffn_up");
    writer.writeFc(model.intermediate_size, model.hidden_size, quant.fc_dtype,
                   prefix + "_ffn_down");

    writer.copyFp32(model.hidden_size, prefix + "_post_ffn_norm_1");
    writer.copyFp32(model.hidden_size, prefix + "_pre_ffn_norm_2");
    writer.copyFp32(tensorElements(model.hidden_size, model.num_experts,
                                   prefix + "_sparse_moe router"),
                    prefix + "_sparse_moe router");
    writer.copyFp32(model.hidden_size, prefix + "_sparse_moe router_scale");
    writer.copyFp32(model.num_experts,
                    prefix + "_sparse_moe router_per_expert_scale");

    for (size_t expert = 0; expert < model.num_experts; ++expert) {
      const std::string expert_prefix =
        prefix + "_expert" + std::to_string(expert);
      writer.writeFc(model.hidden_size, model.moe_intermediate_size,
                     quant.fc_dtype, expert_prefix + "_gate");
      writer.writeFc(model.hidden_size, model.moe_intermediate_size,
                     quant.fc_dtype, expert_prefix + "_up");
      writer.writeFc(model.moe_intermediate_size, model.hidden_size,
                     quant.fc_dtype, expert_prefix + "_down");
    }

    writer.copyFp32(model.hidden_size, prefix + "_post_ffn_norm_2");
    writer.copyFp32(model.hidden_size, prefix + "_post_ffn_norm");

    if (model.per_layer_input_size != 0) {
      writer.writeFc(model.hidden_size, model.per_layer_input_size,
                     quant.fc_dtype, prefix + "_per_layer_input_gate");
      if (layer == 0) {
        const size_t total_per_layer_size =
          checkedMultiply(model.num_layers, model.per_layer_input_size,
                          "per-layer input total size");
        writer.writeEmbedding(model.per_layer_vocab_size, total_per_layer_size,
                              quant.fc_dtype, "per_layer_input_embedding");
        writer.writeFc(model.hidden_size, total_per_layer_size, quant.fc_dtype,
                       "per_layer_input_projection");
        writer.copyFp32(model.per_layer_input_size,
                        "per_layer_projection_norm");
      }
      writer.writeFc(model.per_layer_input_size, model.hidden_size,
                     quant.fc_dtype, prefix + "_per_layer_input_proj");
      writer.copyFp32(model.hidden_size, prefix + "_post_per_layer_input_norm");
    }
    writer.copyFp32(1, prefix + "_layer_scalar");

    std::cout << "  Quantized layer " << layer + 1 << "/" << model.num_layers
              << '\n';
  }

  writer.copyFp32(model.hidden_size, "output_norm");
  if (model.tied_embeddings) {
    // The Gemma4 converter retains a legacy trailing shared embedding, while
    // NNTrainer model saving emits only embedding0. Accept either FP32 source.
    if (writer.hasRemainingBytes()) {
      writer.discardFp32(tensorElements(model.vocab_size, model.hidden_size,
                                        "tied output_of_causallm"),
                         "tied output_of_causallm");
    }
  } else {
    writer.writeFc(model.hidden_size, model.vocab_size, quant.lmhead_dtype,
                   "output_of_causallm");
  }
  writer.requireEndOfFile();
}

std::string stripKnownDtypeSuffix(std::string base) {
  const std::vector<std::string> suffixes = {"_fp32", "_fp16", "_q40", "_q4_0",
                                             "_q4k",  "_q4_k", "_q6k", "_q6_k"};
  for (const auto &suffix : suffixes) {
    if (base.size() >= suffix.size() &&
        base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
      base.resize(base.size() - suffix.size());
      break;
    }
  }
  return base;
}

std::string defaultOutputBin(const std::string &input_bin,
                             const QuantizationPlan &quant) {
  std::filesystem::path input_path(input_bin);
  std::string base = stripKnownDtypeSuffix(input_path.stem().string());
  std::string result = base + "_" + dtypeSuffix(quant.fc_dtype);
  if (quant.embedding_dtype != quant.fc_dtype)
    result += "_embd" + dtypeSuffix(quant.embedding_dtype);
  if (quant.lmhead_dtype != quant.embedding_dtype)
    result += "_lmhead" + dtypeSuffix(quant.lmhead_dtype);
  result += "_" + lower(isaName(quant.target_isa)) + ".bin";
  return result;
}

std::filesystem::path
defaultOutputDirectory(const std::filesystem::path &model_dir,
                       const QuantizationPlan &quant) {
  return model_dir / ("quantized_fc_" + lower(dtypeName(quant.fc_dtype)) +
                      "_embd_" + lower(dtypeName(quant.embedding_dtype)) +
                      "_lmhead_" + lower(dtypeName(quant.lmhead_dtype)) +
                      "_isa_" + lower(isaName(quant.target_isa)));
}

void copyAuxiliaryFiles(const std::filesystem::path &model_dir,
                        const std::filesystem::path &output_dir) {
  const char *files[] = {
    "config.json",           "generation_config.json",  "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
    "spiece.model",          "sentencepiece.bpe.model", "modules.json"};
  for (const char *file : files) {
    const std::filesystem::path source = model_dir / file;
    if (std::filesystem::exists(source)) {
      std::filesystem::copy_file(
        source, output_dir / file,
        std::filesystem::copy_options::overwrite_existing);
    }
  }
}

void writeOutputConfig(const std::filesystem::path &model_dir,
                       const std::filesystem::path &output_dir,
                       const std::string &output_bin,
                       const QuantizationPlan &quant, json nntr_cfg) {
  nntr_cfg["model_file_name"] = output_bin;
  nntr_cfg["model_tensor_type"] =
    std::string(dtypeName(quant.fc_dtype)) + "-FP32";
  nntr_cfg["fc_layer_dtype"] = dtypeName(quant.fc_dtype);
  nntr_cfg["embedding_dtype"] = dtypeName(quant.embedding_dtype);
  nntr_cfg["lmhead_dtype"] = dtypeName(quant.lmhead_dtype);

  const std::filesystem::path output_config =
    std::filesystem::equivalent(model_dir, output_dir)
      ? output_dir / "nntr_config_quantized.json"
      : output_dir / "nntr_config.json";
  std::ofstream file(output_config);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open " + output_config.string());
  }
  file << nntr_cfg.dump(4) << '\n';
}

void printUsage(const char *program) {
  std::cout
    << "Usage: " << program << " <model_path> [options]\n\n"
    << "Stream-quantize a supported FP32 CausalLM .bin model without loading "
       "the full model.\n\n"
    << "Options:\n"
    << "  --output, -o <path>   Output directory (default: dtype/ISA "
       "subdirectory)\n"
    << "  --fc_dtype <type>     Attention/expert dtype (default: Q4_0)\n"
    << "  --embd_dtype <type>   Embedding dtype (default: FP32)\n"
    << "  --lmhead_dtype <type> LM head dtype (default: embedding dtype)\n"
    << "  --isa <target>        DEFAULT, X86, or ARM (default: DEFAULT)\n"
    << "  --output_bin <name>   Output .bin filename\n"
    << "  --config <path>       Read target dtype fields and filename from an "
       "nntr config\n"
    << "  -h, --help            Show this help\n\n"
    << "Architectures: Qwen3MoeForCausalLM, Gemma4ForCausalLM, "
       "Gemma4ForConditionalGeneration\n"
    << "Supported dtypes: FP32, Q4_0, Q4_K, Q6_K\n"
    << "Gemma4 MoE FC/expert weights currently support FP32 or Q4_0.\n";
}

int run(int argc, char **argv) {
  if (argc < 2) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }
  if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    printUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  const std::filesystem::path model_dir = argv[1];
  std::filesystem::path output_dir;
  std::string fc_dtype = "Q4_0";
  std::string embedding_dtype = "FP32";
  std::string lmhead_dtype;
  std::string target_isa = "DEFAULT";
  std::string output_bin;
  std::filesystem::path target_config;

  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto requireValue = [&](const std::string &option) {
      if (index + 1 >= argc)
        throw std::invalid_argument("Missing value for " + option);
      return std::string(argv[++index]);
    };

    if (argument == "--output" || argument == "-o")
      output_dir = requireValue(argument);
    else if (argument == "--fc_dtype")
      fc_dtype = requireValue(argument);
    else if (argument == "--embd_dtype")
      embedding_dtype = requireValue(argument);
    else if (argument == "--lmhead_dtype")
      lmhead_dtype = requireValue(argument);
    else if (argument == "--isa")
      target_isa = requireValue(argument);
    else if (argument == "--output_bin")
      output_bin = requireValue(argument);
    else if (argument == "--config")
      target_config = requireValue(argument);
    else if (argument == "--help" || argument == "-h") {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      throw std::invalid_argument("Unknown option: " + argument);
    }
  }

  const json cfg = readJson(model_dir / "config.json");
  json nntr_cfg = readJson(model_dir / "nntr_config.json");
  validateSourceConfig(nntr_cfg);
  const std::string architecture = getArchitecture(cfg);
  const bool is_qwen3_moe = architecture == "Qwen3MoeForCausalLM";
  const bool is_gemma4_moe = architecture == "Gemma4ForCausalLM" ||
                             architecture == "Gemma4ForConditionalGeneration";
  if (!is_qwen3_moe && !is_gemma4_moe)
    throw std::runtime_error("Unsupported architecture: " + architecture);

  Qwen3MoePlan qwen3_model{};
  Gemma4MoePlan gemma4_model{};
  if (is_qwen3_moe)
    qwen3_model = makeQwen3MoePlan(cfg);
  else
    gemma4_model = makeGemma4MoePlan(cfg);

  const bool tied_embeddings =
    is_qwen3_moe ? qwen3_model.tied_embeddings : gemma4_model.tied_embeddings;
  const size_t num_layers =
    is_qwen3_moe ? qwen3_model.num_layers : gemma4_model.num_layers;
  const size_t num_experts =
    is_qwen3_moe ? qwen3_model.num_experts : gemma4_model.num_experts;

  if (!target_config.empty()) {
    const json requested = readJson(target_config);
    if (requested.contains("fc_layer_dtype"))
      fc_dtype = requested["fc_layer_dtype"].get<std::string>();
    if (requested.contains("embedding_dtype"))
      embedding_dtype = requested["embedding_dtype"].get<std::string>();
    if (requested.contains("lmhead_dtype"))
      lmhead_dtype = requested["lmhead_dtype"].get<std::string>();
    if (requested.contains("model_file_name") && output_bin.empty())
      output_bin = requested["model_file_name"].get<std::string>();
    if (requested.contains("moe_cache_size"))
      nntr_cfg["moe_cache_size"] = requested["moe_cache_size"];
  }

  if (lmhead_dtype.empty())
    lmhead_dtype = embedding_dtype;
  const QuantizationPlan quant{parseDType(fc_dtype),
                               parseDType(embedding_dtype),
                               parseDType(lmhead_dtype), parseIsa(target_isa)};

  if (tied_embeddings && quant.embedding_dtype != quant.lmhead_dtype) {
    throw std::invalid_argument(
      "A tied model requires matching embedding and LM head dtypes");
  }
  if (is_gemma4_moe && quant.fc_dtype != DType::FP32 &&
      quant.fc_dtype != DType::Q4_0) {
    throw std::invalid_argument(
      "Gemma4 MoE FC/expert dtype must be FP32 or Q4_0");
  }
  const std::string input_bin =
    nntr_cfg.at("model_file_name").get<std::string>();
  if (std::filesystem::path(input_bin).extension() != ".bin") {
    throw std::invalid_argument(
      "Streaming quantization currently requires an NNTrainer .bin input");
  }
  if (output_dir.empty())
    output_dir = defaultOutputDirectory(model_dir, quant);
  std::filesystem::create_directories(output_dir);
  if (output_bin.empty())
    output_bin = defaultOutputBin(input_bin, quant);
  if (std::filesystem::path(output_bin).extension() != ".bin") {
    throw std::invalid_argument("--output_bin must use the .bin extension");
  }

  const std::filesystem::path input_path = model_dir / input_bin;
  const std::filesystem::path output_path = output_dir / output_bin;
  if (std::filesystem::exists(output_path) &&
      std::filesystem::equivalent(input_path, output_path)) {
    throw std::invalid_argument("Input and output weight paths must differ");
  }

  std::ifstream input(input_path, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Failed to open " + input_path.string());
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output.is_open())
    throw std::runtime_error("Failed to open " + output_path.string());

  std::cout << "NNTrainer CausalLM streaming quantizer\n"
            << "  Architecture: " << architecture << '\n'
            << "  Source: " << input_path << '\n'
            << "  Target: " << output_path << '\n'
            << "  Layers: " << num_layers << '\n'
            << "  Experts per layer: " << num_experts << '\n'
            << "  FC dtype: " << dtypeName(quant.fc_dtype) << '\n'
            << "  Embedding dtype: " << dtypeName(quant.embedding_dtype) << '\n'
            << "  LM head dtype: " << dtypeName(quant.lmhead_dtype) << '\n'
            << "  Target ISA: " << isaName(quant.target_isa) << '\n';

  TensorWriter writer(input, output, quant.target_isa);
  if (is_qwen3_moe)
    writeQwen3Moe(writer, qwen3_model, quant);
  else
    writeGemma4Moe(writer, gemma4_model, quant);
  output.close();
  if (!output)
    throw std::runtime_error("Failed to finalize " + output_path.string());

  if (!std::filesystem::equivalent(model_dir, output_dir))
    copyAuxiliaryFiles(model_dir, output_dir);
  writeOutputConfig(model_dir, output_dir, output_bin, quant, nntr_cfg);

  const uintmax_t input_size = std::filesystem::file_size(input_path);
  const uintmax_t output_size = std::filesystem::file_size(output_path);
  const double ratio = input_size == 0
                         ? 0.0
                         : 100.0 * static_cast<double>(output_size) /
                             static_cast<double>(input_size);
  std::cout << "  Source size: " << input_size / (1024 * 1024) << " MiB\n"
            << "  Output size: " << output_size / (1024 * 1024) << " MiB\n"
            << "  Output/source: " << std::fixed << std::setprecision(1)
            << ratio << "%\n"
            << "Streaming quantization complete\n";
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "[!] FATAL ERROR: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
