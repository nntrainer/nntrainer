// SPDX-License-Identifier: Apache-2.0
/**
 * @file   quick_dot_ai_qnn.cpp
 * @brief  QNN model implementation for Quick.AI template
 * @note   This file implements a layer that executes QNN binary file within
 * transformer.h architecture.
 */

#include "quick_dot_ai_qnn.h"
#include "android_memory_allocator.h"
#include "engine.h"
#include "generate_qnn_utils.h"
#include "graph_parser.h"
#include <climits>
#include <unistd.h>
#include <xgrammar/xgrammar_wrapper.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <codecvt>
#include <locale>
#endif

using namespace ml::train;
using namespace nntrainer;
using namespace causallm;

namespace {

bool is_absolute_path(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

std::string dirname(const std::string &path) {
  auto pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return "";
  }
  return path.substr(0, pos);
}

std::string rebase_relative_to_model_file(const std::string &path,
                                          const std::string &model_file) {
  if (path.empty() || is_absolute_path(path)) {
    return path;
  }

  auto base_dir = dirname(model_file);
  if (base_dir.empty()) {
    return path;
  }
  return base_dir + "/" + path;
}

int read_token_id_or_default(const json &cfg, const char *key,
                             int default_value) {
  if (!cfg.contains(key) || cfg[key].is_null()) {
    return default_value;
  }

  const auto &value = cfg[key];
  if (value.is_number_integer() || value.is_number_unsigned()) {
    return value.get<int>();
  }
  if (value.is_array()) {
    if (value.empty()) {
      return default_value;
    }
    return value.front().get<int>();
  }

  throw std::invalid_argument(std::string(key) +
                              " must be an integer or array");
}

// Format a scale value with enough precision. std::to_string
// silently caps at 6 fractional digits and mangles tiny QNN scales
// (e.g. 0.0004169851... → "0.000417"). Across 70+ tensors × 35 layers
// × every token the dequant drift compounds into representation
// collapse after a few dozen tokens.
inline std::string format_float_precise(float v) {
  std::ostringstream os;
  os << std::setprecision(std::numeric_limits<float>::max_digits10) << v;
  return os.str();
}

} // namespace

/**
 * @brief Dequantize 4-bit packed values to UINT16
 * @param packed_data Pointer to packed 4-bit data (2 values per byte)
 * @param size_in_bytes Size of packed data in bytes
 * @param scale Scale factor for dequantization
 * @param offset Offset for dequantization
 * @return Vector of dequantized UINT16 values
 */
std::vector<uint16_t> dequantize_4bit_packed(const uint8_t *packed_data,
                                             size_t size_in_bytes, float scale,
                                             int offset) {
  std::vector<uint16_t> result;
  result.reserve(size_in_bytes * 2); // 2 values per byte

  for (size_t i = 0; i < size_in_bytes; ++i) {
    uint8_t byte = packed_data[i];

    // Extract lower 4 bits (first value)
    uint8_t lower_nibble = byte & 0x0F;

    // Extract upper 4 bits (second value)
    uint8_t upper_nibble = (byte >> 4) & 0x0F;

    // Dequantize to float and convert to UINT16
    float lower_float = (static_cast<float>(lower_nibble) + offset) * scale;
    float upper_float = (static_cast<float>(upper_nibble) + offset) * scale;

    // Clamp to UINT16 range and convert
    uint16_t lower_uint16 =
      static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, lower_float)));
    uint16_t upper_uint16 =
      static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, upper_float)));

    result.push_back(lower_uint16);
    result.push_back(upper_uint16);
  }

  return result;
}

/**
 * @brief Parse scale and offset from json file
 */
std::pair<float, int>
parse_scale_offset_from_json(const std::string &json_file_path) {
  std::ifstream json_file(json_file_path);
  if (!json_file.is_open()) {
    throw std::runtime_error("Failed to open json file: " + json_file_path);
  }

  std::stringstream buffer;
  buffer << json_file.rdbuf();
  std::string json_str = buffer.str();

  // Simple parsing for "scale" and "offset"
  // This is a very basic parser, assuming the format is fixed
  float scale = 1.0f;
  int offset = 0;

  size_t scale_pos = json_str.find("\"scale\"");
  if (scale_pos != std::string::npos) {
    size_t colon_pos = json_str.find(":", scale_pos);
    if (colon_pos != std::string::npos) {
      size_t value_start = json_str.find_first_not_of(" \t\n\r", colon_pos + 1);
      if (value_start != std::string::npos) {
        size_t value_end = json_str.find_first_of(",}", value_start);
        if (value_end != std::string::npos) {
          std::string scale_str =
            json_str.substr(value_start, value_end - value_start);
          scale = std::stof(scale_str);
        }
      }
    }
  }

  size_t offset_pos = json_str.find("\"offset\"");
  if (offset_pos != std::string::npos) {
    size_t colon_pos = json_str.find(":", offset_pos);
    if (colon_pos != std::string::npos) {
      size_t value_start = json_str.find_first_not_of(" \t\n\r", colon_pos + 1);
      if (value_start != std::string::npos) {
        size_t value_end = json_str.find_first_of(",}", value_start);
        if (value_end != std::string::npos) {
          std::string offset_str =
            json_str.substr(value_start, value_end - value_start);
          offset = std::stoi(offset_str);
        }
      }
    }
  }

  return std::make_pair(scale, offset);
}

std::string qnn_to_nntrainer_datatype(std::string qnn_dtype) {
  if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_16") {
    return "UINT16";
  } else if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_8") {
    return "UINT8";
  } else if (qnn_dtype == "QNN_DATATYPE_FLOAT_16") {
    return "FP16";
  } else {
    LOGE("[MM-DIAG] qnn_to_nntrainer_datatype: UNSUPPORTED qnn_dtype '%s'",
         qnn_dtype.c_str());
    throw std::invalid_argument("qnn_dtype is " + qnn_dtype);
  }
}

causallm::IO_TensorType get_qnn_input_data(TensorInfo tensor_object,
                                           std::set<void *> &allocated_ptrs) {
  int size = GraphParser::get_tensor_size(tensor_object);
  std::string qnn_dtype = tensor_object.data_type;

  if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_16" ||
      qnn_dtype == "QNN_DATATYPE_FLOAT_16") {
    auto *ptr = (uint16_t *)allocate(size);
    allocated_ptrs.insert(ptr);
    return ptr;
  } else if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_8") {
    auto *ptr = (uint8_t *)allocate(size);
    allocated_ptrs.insert(ptr);
    return ptr;
  } else {
    LOGE("[MM-DIAG] get_qnn_input_data: UNSUPPORTED qnn_dtype '%s'",
         qnn_dtype.c_str());
    throw std::invalid_argument("qnn_dtype is " + qnn_dtype);
  }
}

void *causallm::Quick_Dot_AI_QNN::tracked_allocate(size_t size) {
  void *ptr = allocate(size);
  allocated_ptrs_.insert(ptr);
  return ptr;
}

void causallm::Quick_Dot_AI_QNN::deallocate_all() {
  LOGD("Quick_Dot_AI_QNN::deallocate_all: freeing %zu tracked pointers",
       allocated_ptrs_.size());
  for (auto *ptr : allocated_ptrs_) {
    LOGD("Quick_Dot_AI_QNN::deallocate_all: deallocating ptr=%p", ptr);
    deallocate(ptr);
  }
  allocated_ptrs_.clear();
}

std::string causallm::Quick_Dot_AI_QNN::promptToUtf8(const WSTR &prompt) {
#if defined(_WIN32)
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(prompt);
#else
  return prompt;
#endif
}

causallm::Quick_Dot_AI_QNN::~Quick_Dot_AI_QNN() {
  // Tear down each graph's NeuralNetwork (and the QNNGraph layer inside it)
  // FIRST, so ~QNNGraph releases its zero-copy references to our input
  // buffers before we free them. Without this, the deallocate loop below
  // would free memory that QNNGraph still tracks, and ~QNNGraph — invoked
  // later as part of `models` member destruction — would touch freed
  // memory.
  for (auto &[model_name, model] : models) {
    model.model_handle.reset();
  }
  deallocate_all();
}

void causallm::Quick_Dot_AI_QNN::initialize(const std::string &native_lib_dir) {
  native_lib_dir_ = native_lib_dir;
  initialize();
}

void causallm::Quick_Dot_AI_QNN::initialize() {
  int status;

  auto &ct_engine = nntrainer::Engine::Global();
  LOGD("qnn_engine registering .... ");

  NNTR_THROW_IF(ct_engine.registerContext("libqnn_context.so", ""),
                std::runtime_error)
    << "Fail to register QNN Context";

  LOGD("qnn_engine registering done ");
  std::cout << "--------------- " << binary_config_path << std::endl;

  Transformer::registerCustomLayers();

  GraphParser graph_parser = GraphParser();
  auto graphs_info = graph_parser.parseJsonFile(binary_config_path);
  for (const auto &graph_name : graphs_to_use) {
    auto current_model = createModel(ml::train::ModelType::NEURAL_NET);
    std::string out_dim;
    std::string out_data_format;
    std::string out_tensor_format;
    std::string input_names;
    std::string in_quant;
    std::string out_quant;

    NNTR_THROW_IF(graphs_info.find(graph_name) == graphs_info.end(),
                  std::runtime_error)
      << graph_name << " does not exist in model binary config"
      << binary_config_path << "!";

    auto &current_graphs_info = graphs_info[graph_name];
    std::vector<causallm::IO_TensorType> model_inputs;

    for (const auto &tensor_object : current_graphs_info.raw_inputs) {
      auto &tensor_name = tensor_object.name;
      // if (uses_embedding &&
      //     (tensor_name == "inputs_embeds" || tensor_name == "input_embeds"))
      //     {
      // auto input_shape = tensor_object.dimensions;
      // int input_size = input_shape[0];
      // std::string input_shape_string = std::to_string(input_shape[0]);
      // for (int i = 1; i < input_shape.size() - 1; i++) {
      //   input_shape_string += ":";
      //   input_shape_string += std::to_string(input_shape[i]);
      //   input_size *= input_shape[i];
      // }
      // current_model->addLayer(createLayer(
      //     "embedding",
      //     {withKey("name", tensor_name), withKey("in_dim", vocab_size),
      //      withKey("input_shape", input_shape_string),
      //      withKey("out_dim", input_shape.back())}));

      // model_inputs.push_back((float *)tracked_allocate(sizeof(float) *
      // input_size));
      if (uses_embedding &&
          (tensor_name == "inputs_embeds" || tensor_name == "input_embeds")) {
        auto input_shape = tensor_object.dimensions;
        int input_size = input_shape[0];
        std::string input_shape_string = std::to_string(input_shape[0]);
        for (int i = 1; i < (int)input_shape.size() - 1; i++) {
          input_shape_string += ":";
          input_shape_string += std::to_string(input_shape[i]);
          input_size *= input_shape[i];
        }
        // ── Debug: dump inputs_embeds tensor quant params ──
        std::cout << "[EMB-IN-DBG] graph=" << graph_name
                  << " name=" << tensor_name
                  << " dtype=" << tensor_object.data_type
                  << " scale=" << format_float_precise(tensor_object.scale)
                  << " offset=" << tensor_object.offset
                  << " shape=" << input_shape_string << ":"
                  << input_shape.back() << std::endl;

        // Use the CausalLM tensorwise-4bit-aware embedding layer.
        // When `embedding_file_name` points at a JSON manifest with a
        // 4-bit packed LUT, the layer loads the table once via a
        // path-keyed shared cache so peer graphs share one in-memory copy.
        std::vector<std::string> emb_props = {
          withKey("name", tensor_name),
          withKey("in_dim", vocab_size),
          withKey("input_shape", input_shape_string),
          withKey("out_dim", input_shape.back()),
        };
        if (uses_embedding && !embedding_file_name.empty()) {
          emb_props.push_back(
            withKey("quantized_lut_path", embedding_file_name));
          // Round-trip-precise float string so the layer's requant uses
          // the exact same scale QNN sees on the input_embeds tensor.
          emb_props.push_back(withKey(
            "output_quant_scale", format_float_precise(tensor_object.scale)));
          emb_props.push_back(withKey("output_quant_offset",
                                      std::to_string(tensor_object.offset)));
        }

        current_model->addLayer(createLayer("embedding_layer", emb_props));
        model_inputs.push_back(
          (float *)tracked_allocate(sizeof(float) * input_size));
      } else {
        auto input_shape = tensor_object.dimensions;
        std::string input_shape_string = std::to_string(input_shape[0]);

        for (int i = 1; i < input_shape.size(); i++) {
          input_shape_string += ":";
          input_shape_string += std::to_string(input_shape[i]);
        }
        std::cout << tensor_name << " : " << input_shape_string << std::endl;
        current_model->addLayer(createLayer(
          "input",
          {withKey("name", tensor_name),
           // Give each input layer the QNN tensor's real dtype. Without it the
           // layer defaults to the model tensor type (UINT16), so e.g. the
           // UINT8 KV-cache inputs get a tensor that claims 2x the bytes the
           // fed buffer actually holds — any copy into the QNN-read output
           // then over-reads the source. Correct dtypes also keep the
           // input/output tensors the same size so the InputLayer copy is safe.
           withKey("input_dtype",
                   qnn_to_nntrainer_datatype(tensor_object.data_type)),
           withKey("input_shape", input_shape_string)}));
        model_inputs.push_back(
          get_qnn_input_data(tensor_object, allocated_ptrs_));
      }

      if (!input_names.empty()) {
        input_names += ", ";
      }
      input_names += tensor_name;

      if (!in_quant.empty()) {
        in_quant += ",";
      }
      in_quant += tensor_name;
      in_quant += ":";
      in_quant += format_float_precise(tensor_object.scale);
      in_quant += ":";
      in_quant += std::to_string(tensor_object.offset);
    }

    for (const auto &tensor_object : current_graphs_info.raw_outputs) {
      if (!out_dim.empty()) {
        out_dim += ",";
      }
      out_dim += std::to_string(tensor_object.dimensions[0]);
      for (int i = 1; i < tensor_object.dimensions.size(); i++) {
        out_dim += ":";
        out_dim += std::to_string(tensor_object.dimensions[i]);
      }

      if (!out_data_format.empty()) {
        out_data_format += ",";
      }
      out_data_format += qnn_to_nntrainer_datatype(tensor_object.data_type);

      if (!out_tensor_format.empty()) {
        out_tensor_format += ",";
      }
      out_tensor_format += "OUT_TENSOR";

      if (!out_quant.empty()) {
        out_quant += ",";
      }
      out_quant += tensor_object.name;
      out_quant += ":";
      out_quant += format_float_precise(tensor_object.scale);
      out_quant += ":";
      out_quant += std::to_string(tensor_object.offset);
    }

    std::cout << "graph_name -------------   : " << graph_name << std::endl;

    LayerHandle qnn_layer = createLayer(
      "qnn_graph",
      {withKey("name", graph_name), withKey("path", model_file_name),
       withKey("dim", out_dim), withKey("tensor_dtype", out_data_format),
       withKey("tensor_type", out_tensor_format),
       withKey("input_layers", input_names),
       withKey("input_quant_param", in_quant),
       withKey("output_quant_param", out_quant), withKey("engine", "qnn")});
    current_model->addLayer(qnn_layer);
    std::cout << "end qnn graph" << std::endl;
    current_model->setProperty({withKey("batch_size", 1), withKey("epochs", 1),
                                withKey("model_tensor_type", "UINT16-UINT16")});

    auto optimizer = createOptimizer("sgd", {withKey("learning_rate", 0.001)});
    current_model->setOptimizer(std::move(optimizer));

    status = current_model->compile(ExecutionMode::INFERENCE);
    if (status) {
      LOGE("[MM-DIAG] graph '%s': Model COMPILE failed status=%d",
           graph_name.c_str(), status);
      throw std::invalid_argument("Model compilation failed!");
    }
    std::cout << "end compile" << std::endl;
    status = current_model->initialize(ExecutionMode::INFERENCE);
    if (status) {
      LOGE("[MM-DIAG] graph '%s': Model INITIALIZE failed status=%d",
           graph_name.c_str(), status);
      throw std::invalid_argument("Model initialization failed!");
    }
    std::cout << "end inititlize" << std::endl;
    current_model->summarize(std::cout,
                             ml_train_summary_type_e::ML_TRAIN_SUMMARY_MODEL);

    // std::shared_ptr<ml::train::Layer>emb;
    // current_model->getLayer("input_embeds", &emb);
    // auto node = std::static_pointer_cast<nntrainer::LayerNode>(emb);
    // auto in_dt = node->getInputDimensions()[0].getDataType();
    // std::cout << "embedding input dtype = %d",(int)in_dt);

    models[graph_name] = {current_graphs_info, std::move(current_model),
                          model_inputs};
    std::cout << "----------------------- end graph" << std::endl;
  }

  // QNN builds precompiled graphs here instead of a symbolic nntrainer graph,
  // so it never calls Transformer::initialize() (which would compile an empty
  // constructModel() graph). Mark the model initialized ourselves to satisfy
  // the Transformer base contract: load_weight()/repack_weight()/save_weight()
  // all guard on is_initialized and throw if it is still false.
  is_initialized = true;
}

void causallm::Quick_Dot_AI_QNN::load_weight(const std::string &weight_path) {
  // QNN context is loaded lazily in QNNGraph::forwarding() on first
  // inference, so no model_handle->load() call is needed here.
  // Allocate tensors for inference - required for input/output buffers
  for (auto &[key, value] : models) {
    value.model_handle->allocate(ExecutionMode::INFERENCE);
  }
}

void causallm::Quick_Dot_AI_QNN::repack_weight() {
  // No-op. QNN runs precompiled graphs (stored in `models`); the base
  // Transformer symbolic `model` is never constructed for QNN, so the base
  // repack_weight()'s `model->forEachLayer(...)` would dereference a null
  // pointer and segfault. QNN weights are baked into the serialized graph
  // binary — there is nothing to repack here.
}

void causallm::Quick_Dot_AI_QNN::save_weight(const std::string &weight_path) {
  // Unimplemented.
}

void causallm::Quick_Dot_AI_QNN::setupParameters(json &cfg,
                                                 json &generation_cfg,
                                                 json &nntr_cfg) {
  // Read nntr_config parameters
  LOGD("----------------in Quick_Dot_AI_QNN : setupParameters");
  model_file_name = nntr_cfg["model_file_name"].get<std::string>();
  LOGD("----------------binary_config_path : %s", model_file_name.c_str());
  binary_config_path = nntr_cfg["binary_config_path"].get<std::string>();
  binary_config_path =
    rebase_relative_to_model_file(binary_config_path, model_file_name);
  LOGD("----------------binary_config_path : %s", binary_config_path.c_str());
  graphs_to_use = nntr_cfg["graphs_to_use"].get<std::vector<std::string>>();
  for (auto s : graphs_to_use) {
    LOGD("----------------graphs_to_use : %s", s.c_str());
  }
  vocab_size = cfg["vocab_size"].get<int>();
  LOGD("----------------vocab size : %d", vocab_size);

  // Multimodal opt-in: when uses_embedding=false, the LLM graph's
  // input(s)_embeds tensor is fed with pre-computed uint16 embeddings
  // rather than token IDs via an embedding layer. Derived classes
  // also mmap embedding_file_name for per-token lookup during
  // generation (see e.g. Gauss3_8_QNN::lookupEmbedding).
  if (nntr_cfg.contains("uses_embedding")) {
    uses_embedding = nntr_cfg["uses_embedding"].get<bool>();
  }
  LOGD("---------------- uses_embedding : %d", uses_embedding);

  if (nntr_cfg.contains("embedding_file_name")) {
    embedding_file_name = nntr_cfg["embedding_file_name"].get<std::string>();
    embedding_file_name =
      rebase_relative_to_model_file(embedding_file_name, model_file_name);
    LOGD("---------------- embedding_file_name : %s",
         embedding_file_name.c_str());
  }

  // Read generation_config parameters
  padding_token =
    generation_cfg.contains("padding_token")
      ? read_token_id_or_default(generation_cfg, "padding_token", 0)
      : read_token_id_or_default(generation_cfg, "pad_token_id", 0);
  eos_token = read_token_id_or_default(generation_cfg, "eos_token_id", 0);
  temperature = generation_cfg.value("temperature", 1.0f);
  top_k = generation_cfg.value("top_k", 50);
  top_p = generation_cfg.value("top_p", 1.0f);
  repetition_penalty = generation_cfg.value("repetition_penalty", 1.0f);
  logit_scale = generation_cfg.value("logit_scale", 1.0f);
  logit_offset = generation_cfg.value("logit_offset", 0);

  // Read optional lora_path
  lora_path = nntr_cfg.value("lora_path", "");
}

int causallm::Quick_Dot_AI_QNN::sample(uint16_t *pointer, int length,
                                       int *tokens, int number_of_tokens,
                                       float logit_scale, int logit_offset,
                                       float repetition_penalty,
                                       float temperature, float top_p,
                                       int top_k) {
  // Apply grammar mask if xgrammar_ is provided (from base class)
  if (xgrammar_ != nullptr && xgrammar_->isGrammarEnabled()) {
    xgrammar_->applyGrammarMask(pointer, vocab_size, logit_scale, logit_offset);
  }

  // Call the free function sample from generate_qnn_utils.cpp
  int token =
    ::sample(pointer, length, tokens, number_of_tokens, logit_scale,
             logit_offset, repetition_penalty, temperature, top_p, top_k);

  if (token == eos_token || token == padding_token) {
    return token;
  }

  // Accept token in grammar matcher if xgrammar_ is provided (from base class)
  if (xgrammar_ != nullptr && xgrammar_->isGrammarEnabled()) {
    xgrammar_->getGrammarMatcher()->AcceptToken(token);
    // Update bitmask for next token
    xgrammar_->getGrammarMatcher()->FillNextTokenBitmask(
      &xgrammar_->getBitmaskTensor());
  }
  return token;
}

void causallm::Quick_Dot_AI_QNN::resetXGrammar() {
  if (xgrammar_ != nullptr) {
    xgrammar_->resetGrammar();
  }
}

// Note: this TU has both `using namespace ml::train;` and `using namespace
// nntrainer;`, so unqualified `Tensor` is ambiguous. The base virtual uses
// ml::train::Tensor, so qualify explicitly here.
std::pair<ml::train::Tensor, ml::train::Tensor>
causallm::Quick_Dot_AI_QNN::constructModel() {
  // Unimplemented: QNN executes a precompiled binary graph rather than an
  // nntrainer symbolic graph, so the graph-builder overrides are inert. Return
  // empty tensors to satisfy the symbolic-graph base interface.
  return {ml::train::Tensor(), ml::train::Tensor()};
}

ml::train::Tensor causallm::Quick_Dot_AI_QNN::createTransformerDecoderBlock(
  const int layer_id, ml::train::Tensor input) {
  // Unimplemented (see constructModel).
  return ml::train::Tensor();
}

ml::train::Tensor causallm::Quick_Dot_AI_QNN::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  ml::train::Tensor query, ml::train::Tensor key, ml::train::Tensor value) {
  // Unimplemented (see constructModel).
  return ml::train::Tensor();
}

ml::train::Tensor
causallm::Quick_Dot_AI_QNN::createMlp(const int layer_id, int dim,
                                      int hidden_dim, ml::train::Tensor input) {
  // Unimplemented (see constructModel).
  return ml::train::Tensor();
}

void causallm::Quick_Dot_AI_QNN::registerCustomLayers() {
  // Unimplemented.
}

void causallm::Quick_Dot_AI_QNN::quantize_uint16_memcpy(float *src,
                                                        uint16_t *dest,
                                                        int count, float scale,
                                                        int offset) {
  for (int i = 0; i < count; i++) {
    if (std::isfinite(src[i])) {
      float quantized_value = src[i] / scale - offset;
      dest[i] = static_cast<uint16_t>(
        std::max(0.0f, std::min(65535.0f, quantized_value)));
    } else {
      // Warning message?
      dest[i] = 0;
    }
  }
}
