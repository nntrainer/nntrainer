// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   quantize.cpp
 * @date   04 March 2026
 * @brief  Quantization utility for CausalLM models.
 *         Reads a FP32 model and converts weights to a target data type,
 *         saving the quantized weights (.bin or .safetensors) and a new
 *         nntr_config.json.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @usage
 *   nntr_quantize <model_path> [options]
 *
 *   Required:
 *     <model_path>        Path to the model directory containing:
 *                           config.json, generation_config.json,
 *                           nntr_config.json, and the .bin weight file.
 *
 *   Options:
 *     --output, -o <path> Output directory (default: <model_path>)
 *     --fc_dtype <type>   Target dtype for FC layers (default: Q4_0)
 *     --embd_dtype <type> Target dtype for embedding layer (default: FP32)
 *     --lmhead_dtype <type> Target dtype for LM head layer (default: FP32)
 *     --conv_dtype <type>   Target dtype for conv layers (default: FP32;
 *                           vision models only, read from nntr_config if
 * absent)
 *     --output_bin <name> Output weight filename (auto-generated if omitted)
 *     --output_format <fmt> Output container: 'bin' (default) or 'safetensors'
 *     --no_sidecar        Keep the embedding / per-layer-embedding tables
 *                         inside the .bin. By default they are written to an
 *                         mmap'd sidecar file whenever the model qualifies
 *                         (Q4_0/Q6_K embedding; untied lm_head for embedding0;
 *                         a per-layer-embedding model for the PLE table).
 *
 *   Supported data types: FP32, FP16, Q4_0, Q4_K, Q6_K
 *
 *   Example:
 *     # Quantize Qwen3-4B to Q4_0 FC layers (embedding stays FP32):
 *     nntr_quantize /path/to/qwen3-4b --fc_dtype Q4_0
 *
 *     # Quantize with Q6_K embedding and Q4_0 FC layers:
 *     nntr_quantize /path/to/qwen3-4b --fc_dtype Q4_0 --embd_dtype Q6_K
 *
 *     # Quantize to a different output directory:
 *     nntr_quantize /path/to/qwen3-4b -o /output/qwen3-4b-q4
 *
 *     # Use a target nntr_config.json directly:
 *     nntr_quantize /path/to/qwen3-4b --config /path/to/target_nntr_config.json
 */

#include <algorithm>
#include <env_compat.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>
#include <tensor_dim.h>

#include "causal_lm.h"
#include "deberta_v2.h"
#include "embedding_gemma.h"
#include "gemma2_causallm.h"
#include "gemma3_causallm.h"
#include "gemma4_causallm.h"
#include "gemma4_moe_causallm.h"
#if !defined(_WIN32)
#include "gptoss_cached_slim_causallm.h"
#endif
#include "gptoss_causallm.h"
#include "lfm2_causallm.h"
#if !defined(_WIN32) && !defined(__ANDROID__)
#include "multilingual_tinybert_16mb.h"
#endif
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#include "xlm_roberta/xlm_roberta.h"
#if !defined(_WIN32)
#include "qwen3_cached_slim_moe_causallm.h"
#endif
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "siglip2/siglip2_vision_encoder.h"
#if !defined(_WIN32)
#include "bert_decoder/bert_decoder.h"
#include "screenai_caption/screenai_caption.h"
#endif
#if !defined(_WIN32) && !defined(__ANDROID__)
#include "ced/ced_transformer.h"
#include "timm_vit/timm_vit_transformer.h"
#endif
#include "FastViTKeyword/fastvit_keyword_model.h"
#include "YOLOv7Pose/yolov7_pose.h"

using json = nlohmann::json;
using DataType = ml::train::TensorDim::DataType;

namespace {

/**
 * @brief Map of string data type names to DataType enum values
 */
const std::map<std::string, DataType> dtype_str_map = {
  {"FP32", DataType::FP32},   {"FP16", DataType::FP16},
  {"Q4_0", DataType::Q4_0},   {"Q6_K", DataType::Q6_K},
  {"Q4_K", DataType::Q4_K},   {"Q8_0", DataType::Q8_0},
  {"QINT4", DataType::QINT4}, {"QS4CX", DataType::QS4CX},
  {"NONE", DataType::NONE},
};

/**
 * @brief Map of string ISA names to ISA enum values
 */
const std::map<std::string, ml::train::ISA> isa_str_map = {
  {"DEFAULT", ml::train::ISA::DEFAULT},
  {"X86", ml::train::ISA::X86},
  {"ARM", ml::train::ISA::ARM},
};

/**
 * @brief Convert string to ISA enum
 */
ml::train::ISA strToISA(const std::string &s) {
  std::string upper = s;
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  auto it = isa_str_map.find(upper);
  if (it == isa_str_map.end()) {
    throw std::invalid_argument("Unsupported ISA: " + s +
                                ". Supported: DEFAULT, X86, ARM");
  }
  return it->second;
}

/**
 * @brief Convert ISA enum to string
 */
std::string isaToStr(ml::train::ISA isa) {
  for (const auto &[key, val] : isa_str_map) {
    if (val == isa)
      return key;
  }
  return "DEFAULT";
}

/**
 * @brief Convert string to DataType enum
 */
DataType strToDataType(const std::string &s) {
  std::string upper = s;
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  auto it = dtype_str_map.find(upper);
  if (it == dtype_str_map.end()) {
    throw std::invalid_argument(
      "Unsupported data type: " + s +
      ". Supported: FP32, FP16, Q4_0, Q8_0, Q6_K, Q4_K, QS4CX");
  }
  return it->second;
}

/**
 * @brief Convert DataType enum to string
 */
std::string dataTypeToStr(DataType dt) {
  for (const auto &[key, val] : dtype_str_map) {
    if (val == dt)
      return key;
  }
  return "NONE";
}

/**
 * @brief Build model_tensor_type string from fc_dtype and the source's
 *        activation dtype. Format: "<weight_type>-<activation_type>".
 *        Quantization only changes weight bytes; the activation/compute dtype
 *        (and the packed=false norm weights that follow it) is preserved from
 *        the source model_tensor_type. So an FP16-FP16 source yields
 *        "<fc>-FP16" (fp16 norms), an FP32-FP32 source yields "<fc>-FP32".
 */
std::string buildModelTensorType(const std::string &fc_dtype,
                                 const std::string &src_tensor_type) {
  auto dash = src_tensor_type.find('-');
  std::string act =
    (dash != std::string::npos && dash + 1 < src_tensor_type.size())
      ? src_tensor_type.substr(dash + 1)
      : std::string("FP32");
  return fc_dtype + "-" + act;
}

/**
 * @brief Return lowercase copy of a string
 */
std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

/**
 * @brief Generate a descriptive output bin filename
 */
std::string generateOutputBinName(const std::string &original_bin,
                                  const std::string &fc_dtype,
                                  const std::string &embd_dtype,
                                  const std::string &target_isa) {
  // Extract model name from original (e.g., "nntr_qwen3_4b_fp32.bin" ->
  // "nntr_qwen3_4b"). Use the basename only so a relative model_file_name
  // (e.g. "../../YOLOv11/yolov11s.safetensors") doesn't leak path components
  // into the output filename.
  std::string base = original_bin;
  if (auto slash = base.find_last_of("/\\"); slash != std::string::npos)
    base = base.substr(slash + 1);
  // Remove .bin / .safetensors extension
  for (const auto *ext : {".safetensors", ".bin"}) {
    auto dot_pos = base.rfind(ext);
    if (dot_pos != std::string::npos &&
        dot_pos + std::string(ext).size() == base.size()) {
      base = base.substr(0, dot_pos);
      break;
    }
  }

  // Remove old dtype suffix patterns (e.g., _fp32, _q40_fp32)
  // Common patterns: _fp32, _fp16, _q40, _q6k, _q4k, etc.
  std::vector<std::string> dtype_suffixes = {"_fp32",  "_fp16", "_q40", "_q4_0",
                                             "_q6k",   "_q6_k", "_q4k", "_q4_k",
                                             "_qs4cx", "_q80",  "_q8_0"};
  for (const auto &suffix : dtype_suffixes) {
    auto pos = base.rfind(suffix);
    if (pos != std::string::npos && pos + suffix.size() == base.size()) {
      base = base.substr(0, pos);
      break;
    }
  }

  // Build new dtype suffix
  std::string fc_lower = fc_dtype;
  std::transform(fc_lower.begin(), fc_lower.end(), fc_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  // Replace _ for cleaner naming
  std::string fc_clean = fc_lower;
  fc_clean.erase(std::remove(fc_clean.begin(), fc_clean.end(), '_'),
                 fc_clean.end());

  std::string embd_lower = embd_dtype;
  std::transform(embd_lower.begin(), embd_lower.end(), embd_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::string embd_clean = embd_lower;
  embd_clean.erase(std::remove(embd_clean.begin(), embd_clean.end(), '_'),
                   embd_clean.end());

  if (embd_clean == fc_clean) {
    return base + "_" + fc_clean + "_" + target_isa + ".bin";
  }
  return base + "_" + fc_clean + "_embd" + embd_clean + "_" + target_isa +
         ".bin";
}

/**
 * @brief Resolve architecture name from config
 */
std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    // Already-resolved nntrainer class names — pass through
    if (architecture == "Qwen3Embedding" || architecture == "Qwen2Embedding" ||
        architecture == "EmbeddingGemma" ||
        architecture == "MultilingualTinyBert" || architecture == "DebertaV2" ||
        architecture == "XLMRobertaForMaskedLM")
      return architecture;

    if (architecture == "Qwen3ForCausalLM")
      return "Qwen3Embedding";
    else if (architecture == "Gemma3ForCausalLM" ||
             architecture == "Gemma3TextModel")
      return "EmbeddingGemma";
    else if (architecture == "Qwen2Model")
      return "Qwen2Embedding";
    else if (architecture == "BertForMaskedLM")
      return "MultilingualTinyBert";
    else if (architecture == "XLMRobertaModel")
      return "XLMRobertaForMaskedLM";
    else if (architecture == "DebertaV2ForMaskedLM")
      return "DebertaV2";
    else
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
  }

  if (architecture == "Gemma4ForConditionalGeneration") {
    return "Gemma4ForCausalLM";
  }

  // ScreenAI image-captioning model (SigLIP2 encoder + BERT decoder), exported
  // by HuggingFace as a VisionEncoderDecoderModel. Mirrors main.cpp's mapping.
  if (architecture == "VisionEncoderDecoderModel") {
    return "ScreenAICaption";
  }

  return architecture;
}

/**
 * @brief Return the final component of a dotted Python class name
 */
std::string getLastComponent(const std::string &type) {
  const size_t last_dot_pos = type.find_last_of('.');
  if (last_dot_pos == std::string::npos)
    return type;

  return type.substr(last_dot_pos + 1);
}

/**
 * @brief Register all CausalLM model factories
 */
void registerAllModels() {
  auto &factory = causallm::Factory::Instance();

  factory.registerModel("LlamaForCausalLM", [](json cfg, json generation_cfg,
                                               json nntr_cfg) {
    return std::make_unique<causallm::CausalLM>(cfg, generation_cfg, nntr_cfg);
  });
  factory.registerModel("Qwen2ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Qwen2CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Qwen2Embedding",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Qwen2Embedding>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Qwen3ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Qwen3CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Qwen3MoeForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Qwen3MoECausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Qwen3SlimMoeForCausalLM", [](json cfg,
                                                      json generation_cfg,
                                                      json nntr_cfg) {
    return std::make_unique<causallm::Qwen3SlimMoECausalLM>(cfg, generation_cfg,
                                                            nntr_cfg);
  });
#if !defined(_WIN32)
  factory.registerModel(
    "Qwen3CachedSlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  factory.registerModel("Qwen3Embedding",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Qwen3Embedding>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("GptOssForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::GptOssForCausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
#if !defined(_WIN32)
  factory.registerModel(
    "GptOssCachedSlimCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  factory.registerModel("Gemma2ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Gemma2CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Gemma3ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Gemma3CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel(
    "Gemma4ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      const json &text_cfg =
        cfg.contains("text_config") && cfg["text_config"].is_object()
          ? cfg["text_config"]
          : cfg;
      if (text_cfg.value("enable_moe_block", false)) {
        return std::unique_ptr<causallm::Transformer>(
          std::make_unique<causallm::Gemma4MoECausalLM>(cfg, generation_cfg,
                                                        nntr_cfg));
      }
      return std::unique_ptr<causallm::Transformer>(
        std::make_unique<causallm::Gemma4CausalLM>(cfg, generation_cfg,
                                                   nntr_cfg));
    });
  factory.registerModel("EmbeddingGemma",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::EmbeddingGemma>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Lfm2ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Lfm2CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("DebertaV2", [](json cfg, json generation_cfg,
                                        json nntr_cfg) {
    return std::make_unique<causallm::DebertaV2>(cfg, generation_cfg, nntr_cfg);
  });
  factory.registerModel(
    "Siglip2VisionEncoder", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Siglip2VisionEncoder>(
        cfg, generation_cfg, nntr_cfg);
    });
#if !defined(_WIN32) && !defined(__ANDROID__)
  factory.registerModel(
    "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::MultilingualTinyBert>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
#if !defined(_WIN32)
  factory.registerModel(
    "XLMRobertaForMaskedLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::XLMRobertaForMaskedLM>(
        cfg, generation_cfg, nntr_cfg);
    });
  factory.registerModel(
    "BertDecoder", [](json cfg, json generation_cfg, json nntr_cfg) {
      // BertDecoder hardcodes its dimensions; tensor dtypes come from
      // nntr_config (FP32 for the source model). The graph is built FP32 to
      // load FP32 weights; quantization happens at save via the dtype map.
      auto dec = std::make_unique<causallm::BertDecoder>();
      dec->setTensorTypes(nntr_cfg.value("model_tensor_type", "FP32-FP32"),
                          nntr_cfg.value("fc_layer_dtype", "FP32"),
                          nntr_cfg.value("embedding_dtype", "FP32"));
      return dec;
    });
  factory.registerModel("ScreenAICaption",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::ScreenAICaption>(
                            cfg, generation_cfg, nntr_cfg);
                        });
#endif
  // Vision models (conv-based, tokenizer-free). Mirrors nntr_causallm
  // (main.cpp) so nntr_quantize can quantize them through the same Factory.
#if !defined(_WIN32) && !defined(__ANDROID__)
  factory.registerModel("TimmViT",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::TimmViTTransformer>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("CedForAudioClassification",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::CedTransformer>(
                            cfg, generation_cfg, nntr_cfg);
                        });
#endif
  factory.registerModel("YOLOv7ReIDtiny",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Yolov7Pose>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("FastViTKeyword", [](json cfg, json generation_cfg,
                                             json nntr_cfg) {
    return std::make_unique<causallm::FastViTKeywordModel>(cfg, generation_cfg,
                                                           nntr_cfg);
  });
}

/**
 * @brief Print usage information
 */
void printUsage(const char *prog) {
  std::cout
    << "Usage: " << prog << " <model_path> [options]\n"
    << "\n"
    << "Quantize a CausalLM model from FP32 to a target data type.\n"
    << "\n"
    << "Required:\n"
    << "  <model_path>          Path to model directory containing:\n"
    << "                          config.json, generation_config.json,\n"
    << "                          nntr_config.json, and .bin weight file\n"
    << "\n"
    << "Options:\n"
    << "  --output, -o <path>   Output directory (default: <model_path>)\n"
    << "  --container <form>    QINT4 on-disk container: section_a (legacy,\n"
    << "                        default) or plain (engine-neutral, shared\n"
    << "                        with the upstream KAI CPU path)\n"
    << "  --fc_dtype <type>     Target dtype for FC layers (default: Q4_0)\n"
    << "  --embd_dtype <type>   Target dtype for embedding (default: FP32)\n"
    << "  --lmhead_dtype <type> Target dtype for LM head (default: same as "
       "embd_dtype)\n"
    << "  --act_dtype <type>    Activation dtype recorded in "
       "model_tensor_type\n"
    << "                        (default: FP32). Options: FP32, FP16. With "
       "Q8_0\n"
    << "                        weights, FP32 selects the W8A8 path (ggml "
       "re-quantizes\n"
    << "                        activations to Q8_0); FP16 selects W8A16, "
       "which is\n"
    << "                        markedly more accurate.\n"
    << "  --conv_dtype <type>   Target dtype for conv layers (default: FP32 / "
       "read from nntr_config).\n"
    << "                        Vision models only; surface conv names via the "
       "model's\n"
    << "                        getQuantizableLayerNames(). No effect on LLM "
       "backbones.\n"
    << "  --isa <arch>          Target instruction set architecture for "
       "quantized weights\n"
    << "                        (default: DEFAULT). Options: DEFAULT, X86, "
       "ARM.\n"
    << "  --output_bin <name>   Output weight filename (auto-generated if "
       "omitted)\n"
    << "  --output_format <fmt> Output container: 'bin' (default) or "
       "'safetensors'\n"
    << "  --config <path>       Use a target nntr_config.json instead of\n"
    << "                        individual dtype options. The fc_layer_dtype,\n"
    << "                        embedding_dtype, and lmhead_dtype fields\n"
    << "                        from this config will be used.\n"
    << "  --ple_sidecar         Require the per-layer-embedding (PLE) table\n"
    << "                        in a sidecar file (<bin>_ple.bin + _ple.json "
       "GGML\n"
    << "                        manifest); fail if the model does not "
       "qualify.\n"
    << "  --embd_sidecar        Same for the embedding0 token-embedding table\n"
    << "                        (<bin>_embd.bin + _embd.json).\n"
    << "  --no_ple_sidecar      Keep the PLE table inside the model .bin.\n"
    << "  --no_embd_sidecar     Keep the embedding0 table inside the model "
       ".bin.\n"
    << "  --no_sidecar          Both of the above (the former default "
       "layout).\n"
    << "                        By default each table goes to a sidecar "
       "whenever\n"
    << "                        the model qualifies, and stays in the .bin\n"
    << "                        otherwise. A sidecar is mmapped by the runtime "
       "and\n"
    << "                        only the rows of the tokens seen are read, so\n"
    << "                        no backend keeps a resident copy. It needs a\n"
    << "                        Q4_0 or Q6_K --embd_dtype and the 'bin' "
       "output\n"
    << "                        format; the PLE table needs a model with\n"
    << "                        hidden_size_per_layer_input (Gemma-4), and\n"
    << "                        embedding0 needs an UNTIED lm_head\n"
    << "                        (lmhead_untie=true) - a tied head scans every\n"
    << "                        table row per decode step, so a sidecar saves\n"
    << "                        nothing. The output nntr_config.json names "
       "the\n"
    << "                        sidecar (ple_file_name / "
       "embedding_file_name).\n"
    << "                        An already-quantized source with matching "
       "dtypes\n"
    << "                        is repacked bit-identically.\n"
    << "  --help, -h            Show this help message\n"
    << "\n"
    << "Supported data types: FP32, FP16, Q4_0, Q8_0, Q6_K, Q4_K, QS4CX\n"
    << "  int4 = QS4CX (per-channel int4: plain nibbles + fp32 scales) is the\n"
    << "  canonical int4 weight format for GPU/NPU FC/lm_head. The legacy "
       "QINT4\n"
    << "  target is still accepted, and existing QINT4-FP16 .bin files still "
       "load\n"
    << "  (transcoded losslessly to QS4CX at read time), but prefer QS4CX for "
       "new\n"
    << "  models.\n"
    << "Supported ISA options: DEFAULT (current platform), X86, ARM\n"
    << "\n"
    << "Examples:\n"
    << "  # Quantize FC layers to Q4_0 (default):\n"
    << "  " << prog << " /path/to/qwen3-4b\n"
    << "\n"
    << "  # Canonical int4 GPU/NPU recipe (FC+lm_head QS4CX, embedding Q6_K):\n"
    << "  " << prog
    << " /path/to/gemma4 --fc_dtype QS4CX --embd_dtype Q6_K --lmhead_dtype "
       "QS4CX\n"
    << "\n"
    << "  # Quantize FC layers to Q4_0 and embedding to Q6_K:\n"
    << "  " << prog << " /path/to/qwen3-4b --fc_dtype Q4_0 --embd_dtype Q6_K\n"
    << "\n"
    << "  # Quantize to ARM format for deployment on ARM devices:\n"
    << "  " << prog << " /path/to/qwen3-4b --isa ARM\n"
    << "\n"
    << "  # Quantize to X86 format for deployment on x86 devices:\n"
    << "  " << prog << " /path/to/qwen3-4b --isa X86\n"
    << "\n"
    << "  # Quantize to a different output directory:\n"
    << "  " << prog << " /path/to/qwen3-4b -o /output/qwen3-4b-q4\n"
    << "\n"
    << "  # Use a target nntr_config.json:\n"
    << "  " << prog
    << " /path/to/qwen3-4b --config /path/to/target_nntr_config.json\n";
}

/**
 * @brief Build the layer_dtype_map for the model based on target dtypes.
 *
 * Layer naming convention in Transformer:
 *   - embedding0          : embedding layer
 *   - layer{i}_wq/wk/wv  : attention Q/K/V projections (FC layers)
 *   - layer{i}_attention_out : attention output projection (FC layer)
 *   - layer{i}_ffn_up/gate/down : FFN layers (FC layers)
 *   - layer{i}_attention_norm, layer{i}_ffn_norm : RMSNorm layers
 *   - output_norm          : final RMSNorm
 *   - output_of_causallm   : LM head (FC layer)
 *
 * The dtype map assigns:
 *   - embedding0             -> embd_dtype
 *   - All FC layers (wq, wk, wv, attention_out, ffn_*) -> fc_dtype
 *   - output_of_causallm     -> lmhead_dtype
 *   - RMSNorm / other layers -> FP32 (not quantized)
 */
std::map<std::string, DataType>
buildLayerDtypeMap(int num_layers, DataType fc_dtype, DataType embd_dtype,
                   DataType lmhead_dtype, bool include_lmhead) {

  std::map<std::string, DataType> dtype_map;

  // Embedding layer
  if (embd_dtype != DataType::FP32 && embd_dtype != DataType::NONE) {
    dtype_map["embedding0"] = embd_dtype;
    dtype_map["position_embedding"] = embd_dtype;
    dtype_map["token_type_embedding"] = embd_dtype;
  }

  // ViT (TimmViT) patch-embedding projection is a plain FC layer.
  if (fc_dtype != DataType::FP32 && fc_dtype != DataType::NONE) {
    dtype_map["patch_embed/proj"] = fc_dtype;
  }
  // Gemma4 PLE input embedding is a lookup table (EmbeddingLayer), whose save
  // supports only Q4_0/Q6_K/FP32 — NOT QINT4 (a matmul-weight packing). Use
  // embd_dtype, not fc_dtype, so QINT4 FC runs don't choke on this embedding.
  if (embd_dtype != DataType::FP32 && embd_dtype != DataType::NONE) {
    dtype_map["per_layer_input_embedding"] = embd_dtype;
  }
  // Gemma4 PLE projection is a plain FC layer -> fc_dtype.
  dtype_map["per_layer_input_projection"] = fc_dtype;

  // Transformer decoder layers
  for (int i = 0; i < num_layers; ++i) {
    std::string prefix = "layer" + std::to_string(i);

    // Attention FC layers
    if (fc_dtype != DataType::FP32 && fc_dtype != DataType::NONE) {
      dtype_map[prefix + "_wq"] = fc_dtype;
      dtype_map[prefix + "_wk"] = fc_dtype;
      dtype_map[prefix + "_wv"] = fc_dtype;
      dtype_map[prefix + "_attention_out"] = fc_dtype;

      // ViT / CED encoder blocks: the fused qkv projection is split into three
      // separate FC layers named after the checkpoint's `attn.qkv` weight.
      dtype_map[prefix + "_qkv_q"] = fc_dtype;
      dtype_map[prefix + "_qkv_k"] = fc_dtype;
      dtype_map[prefix + "_qkv_v"] = fc_dtype;

      // Attention Gates
      dtype_map[prefix + "_attention_gate_down"] = fc_dtype;
      dtype_map[prefix + "_attention_gate_up"] = fc_dtype;

      // Attention Gates
      dtype_map[prefix + "_attention_gate_down"] = fc_dtype;
      dtype_map[prefix + "_attention_gate_up"] = fc_dtype;

      // FFN FC layers - version4
      dtype_map[prefix + "_ffn_gate_up"] = fc_dtype;
      dtype_map[prefix + "_ffn_gate_down"] = fc_dtype;
      dtype_map[prefix + "_ffn_linear_up"] = fc_dtype;

      // FFN FC layers - version3 (Qwen/Gemma LLMs)
      dtype_map[prefix + "_ffn_gate"] = fc_dtype;
      dtype_map[prefix + "_ffn_up"] = fc_dtype;
      dtype_map[prefix + "_ffn_down"] = fc_dtype;

      // Gemma4 MoE custom layer keeps its router weights in FP32 and
      // quantizes only the expert projections in its save override.
      dtype_map[prefix + "_sparse_moe"] = fc_dtype;

      dtype_map[prefix + "_ffn_output"] = fc_dtype;

      // LFM2 conv-block projections (causal_conv1d core stays FP32, but the
      // in/out projections follow fc_layer_dtype like any other FC layer).
      dtype_map[prefix + "_conv_in_proj"] = fc_dtype;
      dtype_map[prefix + "_conv_out_proj"] = fc_dtype;

      // FFN FC layers - BERT (BertTransformer)
      dtype_map[prefix + "_ffn_fc1"] = fc_dtype;

      // FFN FC layers - DeBERTa V2
      dtype_map[prefix + "_intermediate"] = fc_dtype;
      dtype_map[prefix + "_output_dense"] = fc_dtype;

      // for PLE
      dtype_map[prefix + "_per_layer_input_gate"] = fc_dtype;
      dtype_map[prefix + "_per_layer_input_proj"] = fc_dtype;
      // per-layer PLE embedding is a lookup table -> embd_dtype (EmbeddingLayer
      // save supports Q4_0/Q6_K/FP32, not QINT4).
      if (embd_dtype != DataType::FP32 && embd_dtype != DataType::NONE) {
        dtype_map[prefix + "_ple"] = embd_dtype;
      }

      dtype_map[prefix + "_ple_projection"] = fc_dtype;
      dtype_map[prefix + "_ple_input_gate"] = fc_dtype;
    }
  }

  // LM Head layer
  if (include_lmhead && lmhead_dtype != DataType::FP32 &&
      lmhead_dtype != DataType::NONE) {
    dtype_map["output_of_causallm"] = lmhead_dtype;
    // A vision/audio classifier head plays the same role as an LM head, so it
    // follows lmhead_dtype rather than fc_layer_dtype.
    dtype_map["head/classifier"] = lmhead_dtype;
  }

  return dtype_map;
}

/**
 * @brief Build the layer_dtype_map for the SigLIP2 vision encoder.
 *
 * The QWEN-style names produced by buildLayerDtypeMap() (layer{i}_wq,
 * output_of_causallm, ...) do NOT match the encoder layer names, so nothing
 * would be quantized. This builds the exact FC names emitted by
 * siglip2_vision_encoder.cpp.
 *
 * Encoder FCs (per layer i in [0, enc_layers)):
 *   enc_layer{i}_wq, _wk, _wv, _out, _fc1, _fc2
 * plus the single enc_to_dec_proj FC. All are 2D, so they take fc_dtype (Q4_0).
 *
 * For Q8_0 the patch_embed_conv is quantized too (stored [CRS, out_ch] by
 * Conv2DLayer::save, run via NCHW im2col + the interleaved int8 GEMM); for
 * Q4_0 it stays FP32 (a Q4_0 tensor must be 2D). pos_embedding and all
 * LayerNorms stay FP32.
 */
std::map<std::string, DataType> buildEncoderLayerDtypeMap(int enc_layers,
                                                          DataType fc_dtype) {
  std::map<std::string, DataType> dtype_map;

  const bool quant_fc =
    fc_dtype != DataType::FP32 && fc_dtype != DataType::NONE;

  if (quant_fc) {
    for (int i = 0; i < enc_layers; ++i) {
      const std::string pfx = "enc_layer" + std::to_string(i);
      dtype_map[pfx + "_wq"] = fc_dtype;
      dtype_map[pfx + "_wk"] = fc_dtype;
      dtype_map[pfx + "_wv"] = fc_dtype;
      dtype_map[pfx + "_out"] = fc_dtype;
      dtype_map[pfx + "_fc1"] = fc_dtype;
      dtype_map[pfx + "_fc2"] = fc_dtype;
    }
    dtype_map["enc_to_dec_proj"] = fc_dtype;
    // Q8_0 also covers the 16x16 stride-16 patch conv: with no overlap it is a
    // pure [196, 768] x [768, 768] matmul, stored [CRS, out_ch] by
    // Conv2DLayer::save and consumed by the same interleaved int8 GEMM as the
    // FCs (NCHW im2col + dotQnK). Q4_0 keeps the conv FP32 (unchanged).
    if (fc_dtype == DataType::Q8_0)
      dtype_map["patch_embed_conv"] = fc_dtype;
  }

  return dtype_map;
}

/**
 * @brief Build the layer_dtype_map for the BERT cross-attention decoder.
 *
 * The QWEN-style names produced by buildLayerDtypeMap() do NOT match the
 * decoder layer names, so this builds the exact FC / embedding names emitted by
 * bert_decoder.cpp.
 *
 * Decoder FCs (per layer i in [0, dec_layers), -> fc_dtype):
 *   dec_layer{i}_self_q, _self_k, _self_v, _self_out,
 *   dec_layer{i}_cross_q, _cross_k, _cross_v, _cross_out,
 *   dec_layer{i}_ffn_inter, _ffn_out
 * plus the LM-head transform FC lmhead_dense.
 *
 * Embeddings (-> embd_dtype): word_emb / pos_emb / type_emb embedding_layer
 * lookup tables. The embedding_layer supports Q4_0/Q8_0 and Q6_K lookup/save
 * (Q4_0/Q8_0 need width % 32 == 0 — BD_DIM = 256, OK), so embd_dtype is fully
 * configurable (FP32 / Q4_0 / Q8_0 / Q6_K) via --embd_dtype. The decoder graph
 * is built with EMBEDDING_DTYPE = embd_dtype (wired from nntr_config via
 * BertDecoder::setTensorTypes).
 *
 * Deliberately left FP32: all LayerNorms (emb_ln, *_self_ln, *_cross_ln,
 * *_ffn_ln, lmhead_ln), lm_head_proj (tied to word_emb) and lmhead_bias.
 */
std::map<std::string, DataType> buildDecoderLayerDtypeMap(int dec_layers,
                                                          DataType fc_dtype,
                                                          DataType embd_dtype) {
  std::map<std::string, DataType> dtype_map;

  const bool quant_fc =
    fc_dtype != DataType::FP32 && fc_dtype != DataType::NONE;
  const bool quant_embd =
    embd_dtype != DataType::FP32 && embd_dtype != DataType::NONE;

  if (quant_fc) {
    for (int i = 0; i < dec_layers; ++i) {
      const std::string pfx = "dec_layer" + std::to_string(i);
      dtype_map[pfx + "_self_q"] = fc_dtype;
      dtype_map[pfx + "_self_k"] = fc_dtype;
      dtype_map[pfx + "_self_v"] = fc_dtype;
      dtype_map[pfx + "_self_out"] = fc_dtype;
      dtype_map[pfx + "_cross_q"] = fc_dtype;
      dtype_map[pfx + "_cross_k"] = fc_dtype;
      dtype_map[pfx + "_cross_v"] = fc_dtype;
      dtype_map[pfx + "_cross_out"] = fc_dtype;
      dtype_map[pfx + "_ffn_inter"] = fc_dtype;
      dtype_map[pfx + "_ffn_out"] = fc_dtype;
    }
    dtype_map["lmhead_dense"] = fc_dtype;
  }

  if (quant_embd) {
    dtype_map["word_emb"] = embd_dtype;
    dtype_map["pos_emb"] = embd_dtype;
    dtype_map["type_emb"] = embd_dtype;
  }

  return dtype_map;
}

/**
 * @brief Build the layer_dtype_map for the ScreenAICaption model.
 *
 * ScreenAICaption is a two-sub-model pipeline (SigLIP2 vision encoder + BERT
 * decoder). The QWEN-style names produced by buildLayerDtypeMap() (layer{i}_wq,
 * output_of_causallm, ...) do NOT match the caption layer names, so nothing
 * would be quantized. This builds the exact FC / embedding names emitted by
 * siglip2_vision_encoder.cpp and bert_decoder.cpp.
 *
 * ENCODER FCs ARE QUANTIZED. The SigLIP2 encoder's FC layers (enc_layer{i}_wq/
 * _wk/_wv/_out/_fc1/_fc2 and enc_to_dec_proj) are 2D, so they take fc_dtype
 * (Q4_0) just like the decoder FCs — this is what shrinks the dominant ~343MB
 * FP32 encoder bin. The two NON-2D / pinned encoder weights stay FP32 and are
 * deliberately omitted from the map: patch_embed_conv is a 4D conv kernel (a
 * Q4_0 tensor must be 2D — "Q4_0_Tensor must be 2 dimensional ...") and
 * pos_embedding is FP32-pinned. The encoder graph pins both via explicit
 * per-layer dtypes (weight_dtype/tensor_dtype="FP32") so the quantized encoder
 * graph loads conv/pos as FP32 and the FCs as Q4_0.
 *
 * Deliberately left FP32 (NOT quantized), regardless of fc/embd dtype:
 *   - encoder patch_embed_conv (4D conv, explicitly weight_dtype FP32),
 *     pos_embedding (FP32-pinned), and all LayerNorms,
 *   - all decoder LayerNorms (emb_ln, *_self_ln, *_cross_ln, *_ffn_ln,
 *     lmhead_ln),
 *   - lm_head_proj (tied to word_emb, no own weight) and lmhead_bias.
 */
std::map<std::string, DataType> buildCaptionLayerDtypeMap(int enc_layers,
                                                          int dec_layers,
                                                          DataType fc_dtype,
                                                          DataType embd_dtype) {
  std::map<std::string, DataType> dtype_map;

  const bool quant_fc =
    fc_dtype != DataType::FP32 && fc_dtype != DataType::NONE;
  const bool quant_embd =
    embd_dtype != DataType::FP32 && embd_dtype != DataType::NONE;

  if (quant_fc) {
    // ---- Encoder (SigLIP2) FCs (2D — pos_embedding excluded, it stays FP32
    //      via the encoder graph's per-layer pin) ----
    for (int i = 0; i < enc_layers; ++i) {
      const std::string pfx = "enc_layer" + std::to_string(i);
      dtype_map[pfx + "_wq"] = fc_dtype;
      dtype_map[pfx + "_wk"] = fc_dtype;
      dtype_map[pfx + "_wv"] = fc_dtype;
      dtype_map[pfx + "_out"] = fc_dtype;
      dtype_map[pfx + "_fc1"] = fc_dtype;
      dtype_map[pfx + "_fc2"] = fc_dtype;
    }
    dtype_map["enc_to_dec_proj"] = fc_dtype;
    // Q8_0 also covers the 16x16 stride-16 patch conv (same rationale as
    // buildEncoderLayerDtypeMap). This is required, not just an optimization,
    // for an FP16-activation model: HalfTensor::dot only accepts an FP16 or
    // Q4_0/Q6_K/Q8_0 operand, so leaving the conv weight FP32 throws
    // "unsupported datatype" at the first patch-embed matmul.
    if (fc_dtype == DataType::Q8_0)
      dtype_map["patch_embed_conv"] = fc_dtype;

    // ---- Decoder (BERT) FCs ----
    for (int i = 0; i < dec_layers; ++i) {
      const std::string pfx = "dec_layer" + std::to_string(i);
      dtype_map[pfx + "_self_q"] = fc_dtype;
      dtype_map[pfx + "_self_k"] = fc_dtype;
      dtype_map[pfx + "_self_v"] = fc_dtype;
      dtype_map[pfx + "_self_out"] = fc_dtype;
      dtype_map[pfx + "_cross_q"] = fc_dtype;
      dtype_map[pfx + "_cross_k"] = fc_dtype;
      dtype_map[pfx + "_cross_v"] = fc_dtype;
      dtype_map[pfx + "_cross_out"] = fc_dtype;
      dtype_map[pfx + "_ffn_inter"] = fc_dtype;
      dtype_map[pfx + "_ffn_out"] = fc_dtype;
    }
    dtype_map["lmhead_dense"] = fc_dtype;
  }

  // ---- Decoder embedding lookup tables ----
  if (quant_embd) {
    dtype_map["word_emb"] = embd_dtype;
    dtype_map["pos_emb"] = embd_dtype;
    dtype_map["type_emb"] = embd_dtype;
  }

  return dtype_map;
}

/**
 * @brief Add SentenceTransformer module dtype overrides to the dtype map
 */
void addSentenceTransformerLayerDtypes(std::map<std::string, DataType> &map,
                                       const json &nntr_cfg,
                                       const std::string &model_path,
                                       DataType fc_dtype) {
  if (fc_dtype == DataType::FP32 || fc_dtype == DataType::NONE ||
      !nntr_cfg.contains("module_config_path")) {
    return;
  }

  std::filesystem::path modules_config_path =
    nntr_cfg["module_config_path"].get<std::string>();
  if (modules_config_path.is_relative()) {
    modules_config_path =
      std::filesystem::path(model_path) / modules_config_path;
  }

  json modules_json = causallm::LoadJsonFile(modules_config_path.string());
  auto modules = modules_json.get<std::vector<json>>();
  for (const auto &module : modules) {
    if (!module.contains("type"))
      continue;

    const std::string component =
      getLastComponent(module["type"].get<std::string>());
    if (component != "Dense")
      continue;

    std::string layer_name;
    if (module.contains("name")) {
      layer_name = module["name"].get<std::string>();
    } else if (module.contains("idx")) {
      layer_name = "sentence_module_" +
                   std::to_string(module["idx"].get<int>()) + "_" + component;
    } else {
      throw std::runtime_error(
        "Dense SentenceTransformer module has neither name nor idx.");
    }

    map[layer_name] = fc_dtype;
  }
}

/**
 * @brief How a table that can live in an mmap'd sidecar file is written.
 *        kAuto (default): sidecar when the model qualifies, in-bin otherwise.
 *        kOn: sidecar required (an unqualified model is an error).
 *        kOff: always inside the model file.
 */
enum class SidecarMode { kAuto, kOn, kOff };

/**
 * @brief Strip a trailing ".bin" from the output weight file name.
 */
std::string sidecarBaseName(const std::string &bin_name) {
  auto pos = bin_name.rfind(".bin");
  if (pos != std::string::npos && pos + 4 == bin_name.size())
    return bin_name.substr(0, pos);
  return bin_name;
}

} // anonymous namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  std::string first_arg = argv[1];
  if (first_arg == "--help" || first_arg == "-h") {
    printUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  // Parse arguments
  std::string model_path = argv[1];
  std::string output_dir = "";
  std::string fc_dtype_str = "Q4_0";
  std::string embd_dtype_str = "FP32";
  std::string lmhead_dtype_str = "";
  // Activation dtype of the emitted model_tensor_type
  // ("<weight>-<activation>"). FP32 activations against Q8_0 weights run ggml's
  // W8A8 path (activations are dynamically re-quantized to Q8_0 per block);
  // FP16 keeps the activations in half precision, which roughly halves the
  // end-to-end quantization error.
  std::string act_dtype_str = "FP32";
  std::string conv_dtype_str =
    ""; // empty = FP32 (default: don't quantize convs)
  std::string isa_str = "DEFAULT";
  std::string output_bin_name = "";
  std::string target_config_path = "";
  std::string output_format = "bin";
  SidecarMode ple_mode = SidecarMode::kAuto;
  SidecarMode embd_mode = SidecarMode::kAuto;
  std::optional<unsigned int> target_moe_cache_size;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--fc_dtype" && i + 1 < argc) {
      fc_dtype_str = argv[++i];
    } else if (arg == "--embd_dtype" && i + 1 < argc) {
      embd_dtype_str = argv[++i];
    } else if (arg == "--lmhead_dtype" && i + 1 < argc) {
      lmhead_dtype_str = argv[++i];
    } else if (arg == "--act_dtype" && i + 1 < argc) {
      act_dtype_str = argv[++i];
      if (act_dtype_str != "FP32" && act_dtype_str != "FP16") {
        std::cerr << "Unknown activation dtype: " << act_dtype_str
                  << " (expected 'FP32' or 'FP16')\n";
        return EXIT_FAILURE;
      }
    } else if (arg == "--conv_dtype" && i + 1 < argc) {
      conv_dtype_str = argv[++i];
    } else if (arg == "--isa" && i + 1 < argc) {
      isa_str = argv[++i];
    } else if (arg == "--output_bin" && i + 1 < argc) {
      output_bin_name = argv[++i];
    } else if (arg == "--output_format" && i + 1 < argc) {
      output_format = toLower(argv[++i]);
      if (output_format != "bin" && output_format != "safetensors") {
        std::cerr << "Unknown output format: " << output_format
                  << " (expected 'bin' or 'safetensors')\n";
        return EXIT_FAILURE;
      }
    } else if (arg == "--config" && i + 1 < argc) {
      target_config_path = argv[++i];
    } else if (arg == "--ple_sidecar") {
      ple_mode = SidecarMode::kOn;
    } else if (arg == "--embd_sidecar") {
      embd_mode = SidecarMode::kOn;
    } else if (arg == "--no_ple_sidecar") {
      ple_mode = SidecarMode::kOff;
    } else if (arg == "--no_embd_sidecar") {
      embd_mode = SidecarMode::kOff;
    } else if (arg == "--no_sidecar") {
      ple_mode = SidecarMode::kOff;
      embd_mode = SidecarMode::kOff;
    } else if (arg == "--container" && i + 1 < argc) {
      std::string container = argv[++i];
      if (container == "plain") {
        // Engine-neutral QINT4 record (PR#3978-compatible): plain nibbles +
        // fp32 scales; loaders repack per engine (CPU KAI / GPU Section A).
#if defined(_WIN32)
        _putenv_s("NNTR_QINT4_PLAIN", "1");
#else
        setenv("NNTR_QINT4_PLAIN", "1", 1);
#endif
      } else if (container != "section_a") {
        std::cerr << "Unknown --container (use plain or section_a): "
                  << container << "\n";
        return EXIT_FAILURE;
      }
    } else if (arg == "--int4_scale" && i + 1 < argc) {
      std::string formula = argv[++i];
      if (formula == "range15") {
        // PR#3978/KAI quantizer formula (plain container only).
#if defined(_WIN32)
        _putenv_s("NNTR_QINT4_RANGE15", "1");
#else
        setenv("NNTR_QINT4_RANGE15", "1", 1);
#endif
      } else if (formula != "absmax7") {
        std::cerr << "Unknown --int4_scale (use absmax7 or range15): "
                  << formula << "\n";
        return EXIT_FAILURE;
      }
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      printUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  try {
    // =========================================================================
    // Step 1: Load source configurations
    // =========================================================================
    std::cout << "==========================================================\n";
    std::cout << "  NNTrainer CausalLM Quantization Utility\n";
    std::cout << "==========================================================\n";
    std::cout << "[1/5] Loading configurations from: " << model_path << "\n";

    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    // generation_config.json is LLM-specific (sampling/eos). Vision models
    // (e.g. YOLOv11) don't carry one — load only when present, mirroring
    // nntr_causallm (main.cpp).
    json generation_cfg = json::object();
    {
      const std::string gen_cfg_path = model_path + "/generation_config.json";
      if (std::filesystem::exists(gen_cfg_path))
        generation_cfg = causallm::LoadJsonFile(gen_cfg_path);
    }
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    // If a target config is specified, read dtypes from it
    if (!target_config_path.empty()) {
      std::cout << "  Using target config: " << target_config_path << "\n";
      json target_cfg = causallm::LoadJsonFile(target_config_path);
      if (target_cfg.contains("fc_layer_dtype"))
        fc_dtype_str = target_cfg["fc_layer_dtype"].get<std::string>();
      if (target_cfg.contains("embedding_dtype"))
        embd_dtype_str = target_cfg["embedding_dtype"].get<std::string>();
      if (target_cfg.contains("lmhead_dtype"))
        lmhead_dtype_str = target_cfg["lmhead_dtype"].get<std::string>();
      if (target_cfg.contains("conv_dtype"))
        conv_dtype_str = target_cfg["conv_dtype"].get<std::string>();
      if (target_cfg.contains("model_file_name") && output_bin_name.empty())
        output_bin_name = target_cfg["model_file_name"].get<std::string>();
      if (target_cfg.contains("moe_cache_size"))
        target_moe_cache_size =
          target_cfg["moe_cache_size"].get<unsigned int>();
    }

    // Read conv dtype from the model's own nntr_config when not given on the
    // CLI. Vision models declare conv_dtype there; absent for LLMs (stays FP32
    // -> no conv quantization, identical to prior behavior).
    if (conv_dtype_str.empty() && nntr_cfg.contains("conv_dtype"))
      conv_dtype_str = nntr_cfg["conv_dtype"].get<std::string>();

    // Default lmhead_dtype to embd_dtype if not specified
    if (lmhead_dtype_str.empty())
      lmhead_dtype_str = embd_dtype_str;

    // Parse target ISA
    ml::train::ISA target_isa = strToISA(isa_str);

    // Parse target data types
    DataType fc_dtype = strToDataType(fc_dtype_str);
    DataType embd_dtype = strToDataType(embd_dtype_str);
    DataType lmhead_dtype = strToDataType(lmhead_dtype_str);
    // conv_dtype defaults to FP32 (no conv quantization) when unspecified —
    // LLM backbones have no convs, so this preserves their behavior entirely.
    DataType conv_dtype =
      conv_dtype_str.empty() ? DataType::FP32 : strToDataType(conv_dtype_str);

    // Validate source model is FP32
    std::string src_tensor_type =
      nntr_cfg["model_tensor_type"].get<std::string>();
    if (src_tensor_type != "FP32-FP32") {
      std::cerr << "[WARNING] Source model_tensor_type is '" << src_tensor_type
                << "', not 'FP32-FP32'.\n"
                << "  Quantization from non-FP32 models may produce unexpected "
                   "results.\n";
    }

    // Setup output directory
    if (output_dir.empty())
      output_dir = model_path;
    std::filesystem::create_directories(output_dir);

    // Determine output filename
    std::string original_bin = nntr_cfg["model_file_name"].get<std::string>();
    if (output_bin_name.empty()) {
      output_bin_name =
        generateOutputBinName(original_bin, dataTypeToStr(fc_dtype),
                              dataTypeToStr(embd_dtype), isa_str);
    }
    if (output_format == "safetensors") {
      // The output format is decided by the file extension on save, so make
      // sure the generated/explicit name ends with ".safetensors".
      const std::string bin_ext = ".bin";
      auto pos = output_bin_name.rfind(bin_ext);
      if (pos != std::string::npos &&
          pos + bin_ext.size() == output_bin_name.size())
        output_bin_name = output_bin_name.substr(0, pos) + ".safetensors";
      else if (output_bin_name.find(".safetensors") == std::string::npos)
        output_bin_name += ".safetensors";
    }

    std::string src_weight_path = model_path + "/" + original_bin;
    std::string dst_weight_path = output_dir + "/" + output_bin_name;

    // Sidecar layout for the PLE and embedding0 tables. By default (kAuto)
    // a table goes to its own mmap'd file whenever the model qualifies: the
    // runtime then reads only the rows of the tokens it sees instead of every
    // backend keeping a resident copy of the whole table (for Gemma-4 with a
    // Q6_K embedding that is about 2.1 GiB). The payload path is injected
    // into nntr_cfg so the model wires sidecar_export_path onto the layer
    // during save. --ple_sidecar / --embd_sidecar turn an unqualified model
    // into an error; --no_*sidecar keep the table inside the model file.
    const bool quantized_embd =
      (embd_dtype == DataType::Q4_0 || embd_dtype == DataType::Q6_K);
    auto sidecar_blocker = [&](bool table_present, const char *absent_why,
                               const char *existing_key) -> std::string {
      if (output_format != "bin")
        return "supports only the 'bin' output format";
      if (!quantized_embd)
        return "requires --embd_dtype Q4_0 or Q6_K (got " +
               dataTypeToStr(embd_dtype) + ")";
      if (!table_present)
        return absent_why;
      if (!nntr_cfg.value(existing_key, std::string()).empty())
        return std::string("source model already uses a sidecar (") +
               existing_key + " set); nothing to extract";
      return "";
    };
    auto resolve_mode = [](SidecarMode mode, const std::string &blocker,
                           const char *flag) {
      if (mode == SidecarMode::kOff)
        return false;
      if (blocker.empty())
        return true;
      if (mode == SidecarMode::kOn)
        throw std::invalid_argument(std::string(flag) + " " + blocker);
      return false;
    };

    const bool has_ple = cfg.contains("hidden_size_per_layer_input") &&
                         cfg["hidden_size_per_layer_input"].is_number() &&
                         cfg["hidden_size_per_layer_input"].get<int>() > 0;
    bool ple_sidecar =
      resolve_mode(ple_mode,
                   sidecar_blocker(has_ple,
                                   "requires a model with per-layer embeddings "
                                   "(config.json hidden_size_per_layer_input)",
                                   "ple_file_name"),
                   "--ple_sidecar");

    // embedding0 is gated on an UNTIED lm_head: a tied head shares the table
    // and scans every row per decode step, so a sidecar would fault the
    // whole file back in and save nothing.
    const bool untied =
      nntr_cfg.contains("lmhead_untie") && nntr_cfg["lmhead_untie"].get<bool>();
    bool embd_sidecar = resolve_mode(
      embd_mode,
      sidecar_blocker(untied,
                      "requires an untied lm_head (lmhead_untie=true); a "
                      "tied lm_head shares the embedding table and would "
                      "gain nothing",
                      "embedding_file_name"),
      "--embd_sidecar");

    const std::string sidecar_base = sidecarBaseName(output_bin_name);
    const std::string ple_payload_name = sidecar_base + "_ple.bin";
    const std::string ple_manifest_name = sidecar_base + "_ple.json";
    const std::string embd_payload_name = sidecar_base + "_embd.bin";
    const std::string embd_manifest_name = sidecar_base + "_embd.json";
    if (ple_sidecar)
      nntr_cfg["ple_sidecar_export"] = output_dir + "/" + ple_payload_name;
    if (embd_sidecar)
      nntr_cfg["embd_sidecar_export"] = output_dir + "/" + embd_payload_name;

    std::string architecture =
      cfg["architectures"].get<std::vector<std::string>>()[0];
    const json &network_cfg =
      cfg.contains("text_config") && cfg["text_config"].is_object()
        ? cfg["text_config"]
        : cfg;

    // The SigLIP2 vision encoder nests its layer count under cfg["encoder"] and
    // the BERT decoder under cfg["decoder"] when a combined config.json is
    // reused; fall back to the top-level field for a flat config.
    const bool is_encoder = (architecture == "Siglip2VisionEncoder");
    const bool is_decoder = (architecture == "BertDecoder");
    // ScreenAICaption (VisionEncoderDecoderModel) nests its layer counts under
    // cfg["encoder"]/cfg["decoder"]; there is no top-level num_hidden_layers.
    const bool is_caption = (architecture == "VisionEncoderDecoderModel" ||
                             architecture == "ScreenAICaption");
    int enc_layers = 0;
    int dec_layers = 0;
    int num_layers = 0;
    if (is_encoder) {
      num_layers = cfg.contains("encoder")
                     ? cfg.at("encoder").value("num_hidden_layers", 12)
                     : cfg.value("num_hidden_layers", 12);
    } else if (is_decoder) {
      num_layers = cfg.contains("decoder")
                     ? cfg.at("decoder").value("num_hidden_layers", 4)
                     : cfg.value("num_hidden_layers", 4);
    } else if (is_caption) {
      enc_layers = cfg.at("encoder").value("num_hidden_layers", 12);
      dec_layers = cfg.at("decoder").value("num_hidden_layers", 4);
      num_layers = enc_layers + dec_layers;
    } else {
      // LLM backbones report num_hidden_layers and timm-style ViT configs
      // (CED included) report depth; pure conv vision models report neither,
      // so fall back to 0: buildLayerDtypeMap's FC loop is then a no-op and
      // the model's getQuantizableLayerNames() supplies the conv names.
      num_layers = network_cfg.contains("num_hidden_layers")
                     ? network_cfg["num_hidden_layers"].get<int>()
                     : network_cfg.value("depth", 0);
    }

    std::cout << "  Architecture: " << architecture << "\n";
    std::cout << "  Num layers:   " << num_layers << "\n";
    std::cout << "  Source:       " << src_weight_path << "\n";
    std::cout << "  Target:       " << dst_weight_path << "\n";
    std::cout << "  FC dtype:     " << dataTypeToStr(fc_dtype) << "\n";
    std::cout << "  Embed dtype:  " << dataTypeToStr(embd_dtype) << "\n";
    std::cout << "  LMHead dtype: " << dataTypeToStr(lmhead_dtype) << "\n";
    std::cout << "  Target ISA:   " << isaToStr(target_isa) << "\n";
    std::cout << "  PLE table:    "
              << (ple_sidecar ? "sidecar " + ple_manifest_name
                              : std::string(has_ple ? "in model file" : "-"))
              << "\n";
    std::cout << "  Embd table:   "
              << (embd_sidecar ? "sidecar " + embd_manifest_name
                               : std::string("in model file"))
              << "\n";
    std::cout << "\n";

    // =========================================================================
    // Step 2: Register models & create model instance
    // =========================================================================
    std::cout << "[2/5] Creating and initializing model...\n";

    registerAllModels();

    if (nntr_cfg.contains("model_type")) {
      std::string model_type = nntr_cfg["model_type"].get<std::string>();
      architecture = resolve_architecture(model_type, architecture);
    }

    // Resolve paths in nntr_cfg against the model directory.
    // Relative paths are anchored to model_path (existing behaviour).
    // Absolute paths that don't exist in the current environment (e.g. a path
    // baked in on the build host but running on an Android device) fall back to
    // the bare filename resolved against model_path, so the file is found as
    // long as it lives next to nntr_config.json.
    // Keep the ORIGINAL (unresolved) values for the output config — baking the
    // build host's absolute paths into it breaks the model dir on-device.
    json orig_path_cfg;
    for (const char *key : {"module_config_path", "tokenizer_file"}) {
      if (nntr_cfg.contains(key))
        orig_path_cfg[key] = nntr_cfg[key];
    }
    for (const char *key : {"module_config_path", "tokenizer_file"}) {
      if (!nntr_cfg.contains(key))
        continue;
      std::filesystem::path p = nntr_cfg[key].get<std::string>();
      if (p.is_relative()) {
        nntr_cfg[key] = (std::filesystem::path(model_path) / p).string();
      } else if (!std::filesystem::exists(p)) {
        nntr_cfg[key] =
          (std::filesystem::path(model_path) / p.filename()).string();
      }
    }

    // Inject model_dir so image-only orchestrators (e.g. ScreenAICaption) can
    // resolve their encoder/decoder weight files relative to the model dir
    // (mirrors main.cpp). The base CausalLM models ignore this field.
    if (!nntr_cfg.contains("model_dir"))
      nntr_cfg["model_dir"] = model_path;

    auto model = causallm::Factory::Instance().create(architecture, cfg,
                                                      generation_cfg, nntr_cfg);
    if (!model) {
      throw std::runtime_error("Failed to create model for architecture: " +
                               architecture);
    }

    model->initialize();
    std::cout << "  Model initialized successfully.\n";

    // =========================================================================
    // Step 3: Load FP32 weights
    // =========================================================================
    std::cout << "[3/5] Loading FP32 weights from: " << src_weight_path << "\n";
    model->load_weight(src_weight_path);
    std::cout << "  Weights loaded successfully.\n";

    // =========================================================================
    // Step 4: Build layer dtype map and save quantized weights
    // =========================================================================
    std::cout << "[4/5] Quantizing and saving weights to: " << dst_weight_path
              << "\n";

    bool include_lmhead = true;
    if (nntr_cfg.contains("model_type") &&
        toLower(nntr_cfg["model_type"].get<std::string>()) == "embedding") {
      include_lmhead = false;
    }

    std::map<std::string, DataType> layer_dtype_map;
    if (is_encoder) {
      // SigLIP2 vision encoder: only its 2D FCs are quantized (conv/pos/LN stay
      // FP32). embd_dtype is unused (the encoder has no embedding lookup
      // table).
      layer_dtype_map = buildEncoderLayerDtypeMap(num_layers, fc_dtype);
    } else if (is_decoder) {
      // BERT decoder: FCs -> fc_dtype, word/pos/type embeddings -> embd_dtype
      // (Q4_0/Q8_0 or Q6_K), LayerNorms/tied-LM-head stay FP32.
      layer_dtype_map =
        buildDecoderLayerDtypeMap(num_layers, fc_dtype, embd_dtype);
    } else if (is_caption) {
      layer_dtype_map =
        buildCaptionLayerDtypeMap(enc_layers, dec_layers, fc_dtype, embd_dtype);
    } else {
      layer_dtype_map = buildLayerDtypeMap(num_layers, fc_dtype, embd_dtype,
                                           lmhead_dtype, include_lmhead);
      addSentenceTransformerLayerDtypes(layer_dtype_map, nntr_cfg, model_path,
                                        fc_dtype);
    }

    // Vision models surface their quantizable conv layer names via the model
    // virtual (the public API has no layer enumeration). When a conv target
    // dtype is requested, assign it to every named conv. FP32 (the default)
    // is skipped so the convs are saved as-is — identical to prior LLM runs.
    if (conv_dtype != DataType::FP32 && conv_dtype != DataType::NONE) {
      auto conv_names = model->getQuantizableLayerNames();
      for (const auto &name : conv_names)
        layer_dtype_map[name] = conv_dtype;
      std::cout << "  Conv dtype:    " << dataTypeToStr(conv_dtype) << " ("
                << conv_names.size() << " conv layers)\n";
    }

    std::cout << "  Layer dtype mapping (" << layer_dtype_map.size()
              << " layers targeted):\n";
    for (const auto &[name, dt] : layer_dtype_map) {
      std::cout << "    " << name << " -> " << dataTypeToStr(dt) << "\n";
    }

    model->save_weight(dst_weight_path, DataType::NONE, layer_dtype_map,
                       target_isa);

    // An architecture that does not route the table to the sidecar leaves it
    // in the model file; that is only an error when the sidecar was required.
    auto sidecar_written = [&](bool &enabled, SidecarMode mode,
                               const std::string &payload_name,
                               const std::string &what) {
      if (!enabled)
        return;
      const std::string path = output_dir + "/" + payload_name;
      if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0)
        return;
      if (mode == SidecarMode::kOn)
        throw std::runtime_error(what + " sidecar was not written - does " +
                                 architecture + " route it to a sidecar?");
      std::filesystem::remove(path);
      std::cout << "  " << what << " table kept in the model file ("
                << architecture << " writes no sidecar for it)\n";
      enabled = false;
    };
    sidecar_written(ple_sidecar, ple_mode, ple_payload_name, "PLE");
    sidecar_written(embd_sidecar, embd_mode, embd_payload_name, "embedding");

    if (ple_sidecar) {
      const std::string ple_payload_path = output_dir + "/" + ple_payload_name;

      // Manifest: GGML row payload; rows/size cross-checked by the loader
      // against the payload byte count.
      const bool q6k = (embd_dtype == DataType::Q6_K);
      const size_t out_dim = static_cast<size_t>(num_layers) *
                             cfg["hidden_size_per_layer_input"].get<size_t>();
      const size_t row_bytes =
        q6k ? 210 * ((out_dim + 255) / 256) : 18 * ((out_dim + 31) / 32);
      const size_t payload = std::filesystem::file_size(ple_payload_path);
      if (payload % row_bytes != 0)
        throw std::runtime_error("PLE sidecar size " + std::to_string(payload) +
                                 " is not a multiple of the row stride " +
                                 std::to_string(row_bytes));

      json manifest;
      manifest["datatype"] = q6k ? "q6_k" : "q4_0";
      manifest["size"] = out_dim;
      manifest["rows"] = payload / row_bytes;
      manifest["lut-path"] = ple_payload_name;
      std::ofstream mf(output_dir + "/" + ple_manifest_name);
      if (!mf.is_open())
        throw std::runtime_error("Failed to open PLE manifest for write");
      mf << manifest.dump(4) << std::endl;

      std::cout << "  PLE sidecar:  " << ple_payload_name << " ("
                << (payload / (1024 * 1024)) << " MB, " << payload / row_bytes
                << " rows x " << out_dim << ") + " << ple_manifest_name << "\n";
    }

    if (embd_sidecar) {
      const std::string embd_payload_path =
        output_dir + "/" + embd_payload_name;

      const bool q6k = (embd_dtype == DataType::Q6_K);
      const size_t hidden = cfg["hidden_size"].get<size_t>();
      const size_t row_bytes =
        q6k ? 210 * ((hidden + 255) / 256) : 18 * ((hidden + 31) / 32);
      const size_t payload = std::filesystem::file_size(embd_payload_path);
      if (payload % row_bytes != 0)
        throw std::runtime_error(
          "embedding sidecar size " + std::to_string(payload) +
          " is not a multiple of the row stride " + std::to_string(row_bytes));

      json manifest;
      manifest["datatype"] = q6k ? "q6_k" : "q4_0";
      manifest["size"] = hidden;
      manifest["rows"] = payload / row_bytes;
      manifest["lut-path"] = embd_payload_name;
      std::ofstream mf(output_dir + "/" + embd_manifest_name);
      if (!mf.is_open())
        throw std::runtime_error("Failed to open embd manifest for write");
      mf << manifest.dump(4) << std::endl;

      std::cout << "  EMBD sidecar: " << embd_payload_name << " ("
                << (payload / (1024 * 1024)) << " MB, " << payload / row_bytes
                << " rows x " << hidden << ") + " << embd_manifest_name << "\n";
    }

    // Report file size
    auto src_size = std::filesystem::file_size(src_weight_path);
    auto dst_size = std::filesystem::file_size(dst_weight_path);
    double ratio = static_cast<double>(dst_size) / src_size * 100.0;

    std::cout << "  Source size:  " << (src_size / (1024 * 1024)) << " MB\n";
    std::cout << "  Output size:  " << (dst_size / (1024 * 1024)) << " MB\n";
    std::cout << "  Compression:  " << std::fixed << std::setprecision(1)
              << ratio << "%\n";

    // =========================================================================
    // Step 5: Generate new nntr_config.json
    // =========================================================================
    std::cout << "[5/5] Generating nntr_config.json...\n";

    json new_nntr_cfg = nntr_cfg;
    new_nntr_cfg["model_file_name"] = output_bin_name;
    new_nntr_cfg["fc_layer_dtype"] = dataTypeToStr(fc_dtype);
    new_nntr_cfg["embedding_dtype"] = dataTypeToStr(embd_dtype);
    new_nntr_cfg["lmhead_dtype"] = dataTypeToStr(lmhead_dtype);
    if (target_moe_cache_size.has_value())
      new_nntr_cfg["moe_cache_size"] = *target_moe_cache_size;
    // The encoder's patch conv is quantized alongside the FCs for Q8_0 (the
    // NCHW conv runs the same interleaved int8 GEMM); record its dtype so the
    // runtime graph declares the matching weight tensor.
    if ((is_encoder || is_caption) && fc_dtype == DataType::Q8_0)
      new_nntr_cfg["patch_embed_dtype"] = dataTypeToStr(fc_dtype);
    // --act_dtype overrides the activation half of model_tensor_type (e.g.
    // W8A16 from an FP32 source); otherwise it follows the source model.
    if (conv_dtype != DataType::FP32 && conv_dtype != DataType::NONE)
      new_nntr_cfg["conv_dtype"] = dataTypeToStr(conv_dtype);
    new_nntr_cfg["model_tensor_type"] =
      act_dtype_str.empty()
        ? buildModelTensorType(dataTypeToStr(fc_dtype), src_tensor_type)
        : dataTypeToStr(fc_dtype) + "-" + act_dtype_str;
    // *_sidecar_export are extraction-time keys only — the runtime keys are
    // ple_file_name / embedding_file_name, pointing at the manifests next to
    // the model file.
    new_nntr_cfg.erase("ple_sidecar_export");
    new_nntr_cfg.erase("embd_sidecar_export");
    if (ple_sidecar)
      new_nntr_cfg["ple_file_name"] = ple_manifest_name;
    if (embd_sidecar)
      new_nntr_cfg["embedding_file_name"] = embd_manifest_name;
    for (auto &[key, value] : orig_path_cfg.items())
      new_nntr_cfg[key] = value;

    // model_dir was injected only to let the orchestrator resolve weight files
    // during this run; drop it so the saved config is portable (the runtime
    // re-injects it from the model directory).
    new_nntr_cfg.erase("model_dir");

    // For ScreenAICaption the runtime reads encoder_model_file_name /
    // decoder_model_file_name (not model_file_name) to find the two sub-model
    // weight files. ScreenAICaption::save_weight wrote them next to
    // dst_weight_path using the same dtype/ISA-tagged basename (encoder ==
    // output_bin_name; decoder == that name with the encoder base swapped for
    // the decoder base). Point the new config at the quantized files.
    if (is_caption) {
      const std::string enc_in =
        nntr_cfg.value("encoder_model_file_name", original_bin);
      const std::string dec_in =
        nntr_cfg.value("decoder_model_file_name", std::string());
      auto base_of = [](const std::string &name) {
        std::string s = std::filesystem::path(name).stem().string();
        for (const char *tag : {"_fp32", "_fp16", "_q40", "_q4_0", "_q6k",
                                "_q6_k", "_q4k", "_q4_k"}) {
          const std::string t(tag);
          if (s.size() >= t.size() &&
              s.compare(s.size() - t.size(), t.size(), t) == 0) {
            s = s.substr(0, s.size() - t.size());
            break;
          }
        }
        return s;
      };
      const std::string enc_base = base_of(enc_in);
      const std::string dec_base = base_of(dec_in);
      std::string dec_name;
      auto pos = output_bin_name.find(enc_base);
      if (!enc_base.empty() && pos != std::string::npos) {
        dec_name = output_bin_name.substr(0, pos) + dec_base +
                   output_bin_name.substr(pos + enc_base.size());
      } else {
        dec_name = dec_base + "_" + output_bin_name;
      }
      new_nntr_cfg["encoder_model_file_name"] = output_bin_name;
      new_nntr_cfg["decoder_model_file_name"] = dec_name;
    }

    std::string output_config_path = output_dir + "/nntr_config.json";

    // If output is same dir and we'd overwrite, save as
    // nntr_config_quantized.json
    if (output_dir == model_path) {
      output_config_path = output_dir + "/nntr_config_quantized.json";
    }

    std::ofstream config_out(output_config_path);
    if (!config_out.is_open()) {
      throw std::runtime_error("Failed to open output config: " +
                               output_config_path);
    }
    config_out << new_nntr_cfg.dump(4) << std::endl;
    config_out.close();

    std::cout << "  Config saved to: " << output_config_path << "\n";

    // When writing to a separate directory, copy the auxiliary files the
    // runtime needs (config.json / generation_config.json are required by
    // CausalLM; tokenizer files are needed for generation) so the output
    // directory is self-contained and runnable on its own.
    if (output_dir != model_path) {
      const char *aux_files[] = {"config.json",
                                 "generation_config.json",
                                 "tokenizer.json",
                                 "tokenizer_config.json",
                                 "special_tokens_map.json",
                                 "vocab.json",
                                 "merges.txt",
                                 "modules.json",
                                 "sample.png"};
      for (const char *fname : aux_files) {
        std::filesystem::path src = std::filesystem::path(model_path) / fname;
        if (!std::filesystem::exists(src))
          continue;
        std::filesystem::copy_file(
          src, std::filesystem::path(output_dir) / fname,
          std::filesystem::copy_options::overwrite_existing);
        std::cout << "  Copied " << fname << " to output directory\n";
      }
    }

    // =========================================================================
    // Done
    // =========================================================================
    std::cout << "\n";
    std::cout << "==========================================================\n";
    std::cout << "  Quantization complete!\n";
    std::cout << "==========================================================\n";
    std::cout << "\n";
    std::cout << "To run the quantized model:\n";
    if (output_dir == model_path) {
      std::cout << "  1. Rename nntr_config_quantized.json to "
                   "nntr_config.json\n";
      std::cout << "  2. nntr_causallm " << model_path << "\n";
    } else {
      std::cout << "  nntr_causallm " << output_dir << "\n\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
