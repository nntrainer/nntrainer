// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   screenai_caption.cpp
 * @date   19 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  ScreenAI caption orchestrator implementation: SigLIP2 encoder +
 *         BERT decoder with greedy autoregressive caption generation.
 */

#include "screenai_caption.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <llm_util.hpp>
#include <performance_metrics.h>
#include <tokenizers_cpp.h>

namespace causallm {

namespace {
/// Load a file as a binary string (tokenizer blob).
std::string loadBytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("ScreenAICaption: cannot open file: " + path);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string buffer(static_cast<size_t>(size), ' ');
  if (!file.read(&buffer[0], size))
    throw std::runtime_error("ScreenAICaption: cannot read file: " + path);
  return buffer;
}

/// Resolve a possibly-relative or stale path against the model directory.
/// Prefers the path as-given if it exists; otherwise falls back to
/// <dir>/<basename> (handles stale absolute paths from nntr_config).
std::string resolvePath(const std::string &dir, const std::string &name) {
  if (name.empty())
    return name;
  if (std::filesystem::exists(name))
    return name;
  if (dir.empty())
    return name;
  std::filesystem::path p(name);
  if (p.is_absolute()) {
    // Stale absolute path: retry with just the filename under the model dir.
    auto fallback = std::filesystem::path(dir) / p.filename();
    if (std::filesystem::exists(fallback))
      return fallback.string();
    return name;
  }
  return (std::filesystem::path(dir) / name).string();
}
} // anonymous namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ScreenAICaption::ScreenAICaption(json &cfg, json &generation_cfg,
                                 json &nntr_cfg) :
  Transformer(),
  cfg_(cfg),
  generation_cfg_(generation_cfg),
  nntr_cfg_(nntr_cfg) {
  // Orchestrator-level runtime parameters.
  BATCH_SIZE = nntr_cfg_.value("batch_size", 1u);
  MODEL_TENSOR_TYPE = nntr_cfg_.value("model_tensor_type", "FP32-FP32");
  NUM_TO_GENERATE = nntr_cfg_.value("num_to_generate", 32);

  // If the caller injected the model directory via nntr_cfg["model_dir"],
  // use it as the base path for resolving weight/tokenizer files.
  if (model_dir_.empty() && nntr_cfg_.contains("model_dir"))
    model_dir_ = nntr_cfg_["model_dir"].get<std::string>();

  // Tokenizer is loaded in initialize(), after model_dir_ is known, so the
  // tokenizer_file path can be resolved relative to the model directory (the
  // configured absolute path may be stale).
}

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------

void ScreenAICaption::initialize() {
  // ----- Tokenizer (WordPiece): resolve tokenizer_file against the model dir
  //       (the configured absolute path may be stale).
  if (!tokenizer && nntr_cfg_.contains("tokenizer_file")) {
    const std::string tok_file =
      resolvePath(model_dir_, nntr_cfg_["tokenizer_file"].get<std::string>());
    tokenizer = tokenizers::Tokenizer::FromBlobJSON(loadBytes(tok_file));
  }

  // ----- Encoder: build a patched nntr_cfg that skips the (text) tokenizer
  //       and declares the MODEL type so the base Transformer ctor is happy.
  // The encoder ALWAYS runs FP32: its patch_embed_conv is a 4D kernel and the
  // conv2d layer forces the kernel dtype to the model's global weight dtype
  // (no per-layer override), so a Q4_0 global type fails graph construction
  // ("Q4_0_Tensor must be 2 dimensional ..."). nntr_quantize correspondingly
  // leaves the encoder weights FP32 and quantizes only the decoder FCs.
  json enc_nntr = nntr_cfg_;
  enc_nntr["model_type"] = "Model";
  enc_nntr["skip_tokenizer"] = true;
  enc_nntr["model_tensor_type"] = "FP32-FP32";
  enc_nntr["fc_layer_dtype"] = "FP32";
  enc_nntr["embedding_dtype"] = "FP32";
  json enc_gen = generation_cfg_;

  encoder_ = std::make_unique<Siglip2VisionEncoder>(cfg_, enc_gen, enc_nntr);
  encoder_->initialize();

  // ----- Decoder: BertDecoder hardcodes its own params (no JSON needed), but
  //       its FC weight tensor type must follow nntr_config so a quantized
  //       (e.g. Q4_0-FP32) decoder graph is built to load quantized weights.
  //       The encoder already reads model_tensor_type from nntr_cfg in its
  //       setupParameters(); mirror that for the decoder here. Embeddings stay
  //       FP32 (only FC layers are quantized by nntr_quantize for caption).
  decoder_ = std::make_unique<BertDecoder>();
  const std::string mtt = nntr_cfg_.value("model_tensor_type", "FP32-FP32");
  const std::string fcdt = nntr_cfg_.value("fc_layer_dtype", "FP32");
  decoder_->setTensorTypes(mtt, fcdt);
  decoder_->initialize();
  // Allocate + bind the self-attention KV cache (cross-cache buffers are
  // allocated + filled per-image in prefillCrossCache during run()).
  decoder_->allocateAndBindKVCache();

  is_initialized = true;
}

// ---------------------------------------------------------------------------
// load_weight
// ---------------------------------------------------------------------------

void ScreenAICaption::load_weight(const std::string &weight_path) {
  (void)weight_path; // encoder/decoder paths come from nntr_config
  if (!is_initialized)
    throw std::runtime_error(
      "ScreenAICaption::load_weight: call initialize() first");

  const std::string enc_name =
    nntr_cfg_.value("encoder_model_file_name", std::string());
  const std::string dec_name =
    nntr_cfg_.value("decoder_model_file_name", std::string());
  if (enc_name.empty() || dec_name.empty())
    throw std::runtime_error(
      "ScreenAICaption::load_weight: encoder_model_file_name / "
      "decoder_model_file_name missing from nntr_config");

  encoder_weight_file_ = resolvePath(model_dir_, enc_name);
  decoder_weight_file_ = resolvePath(model_dir_, dec_name);

  encoder_->load_weight(encoder_weight_file_);
  decoder_->load_weight(decoder_weight_file_);
}

// ---------------------------------------------------------------------------
// save_weight  (delegate to encoder_ / decoder_ sub-models)
// ---------------------------------------------------------------------------
//
// ScreenAICaption owns two Transformer sub-models (encoder_/decoder_), each
// with its own `model` and save_weight; the base ScreenAICaption itself never
// populates the inherited `model` member. The quantize tool drives the
// lifecycle factory.create -> initialize() -> load_weight(src) -> save_weight()
// once on the orchestrator, so this override fans the single save out to both
// sub-models, deriving each output path from the encoder/decoder input file
// names plus the dtype suffix the caller embedded in @p weight_path.

void ScreenAICaption::save_weight(
  const std::string &weight_path, ml::train::TensorDim::DataType dtype,
  const std::map<std::string, ml::train::TensorDim::DataType> &layer_dtype_map,
  ml::train::ISA target_isa) {
  if (!is_initialized)
    throw std::runtime_error(
      "ScreenAICaption::save_weight: call initialize() first");
  if (!encoder_ || !decoder_)
    throw std::runtime_error(
      "ScreenAICaption::save_weight: sub-models not constructed");

  // The caller (nntr_quantize) builds weight_path from model_file_name, which
  // for the caption config is the ENCODER bin (the dtype-suffixed, ISA-tagged
  // output name). So the encoder output is exactly weight_path. The decoder
  // output reuses that same dtype/ISA-tagged basename, but with the encoder
  // input stem swapped for the decoder input stem (e.g.
  // nntr_siglip2_encoder_q40_..._DEFAULT.bin ->
  // nntr_caption_decoder_q40_..._DEFAULT.bin), keeping both files in the same
  // output directory.
  const std::filesystem::path out_path(weight_path);
  const std::filesystem::path out_dir = out_path.has_parent_path()
                                          ? out_path.parent_path()
                                          : std::filesystem::path(".");
  const std::string out_name = out_path.filename().string();

  const std::string enc_in =
    nntr_cfg_.value("encoder_model_file_name", std::string());
  const std::string dec_in =
    nntr_cfg_.value("decoder_model_file_name", std::string());
  if (enc_in.empty() || dec_in.empty())
    throw std::runtime_error(
      "ScreenAICaption::save_weight: encoder_model_file_name / "
      "decoder_model_file_name missing from nntr_config");

  // Strip a trailing source-dtype tag (e.g. "_fp32") so the encoder/decoder
  // base names match how nntr_quantize's generateOutputBinName() rewrites them
  // (it removes the old dtype suffix before appending the new one).
  auto base_of = [](const std::string &name) {
    std::string s = std::filesystem::path(name).stem().string();
    for (const char *tag : {"_fp32", "_fp16", "_q40", "_q4_0", "_q6k", "_q6_k",
                            "_q4k", "_q4_k"}) {
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

  // Encoder file == caller's weight_path. Derive the decoder file name by
  // replacing the encoder base name with the decoder base name inside the
  // generated output basename (e.g. nntr_siglip2_encoder_q40_..._DEFAULT.bin ->
  // nntr_caption_decoder_q40_..._DEFAULT.bin). If the encoder base is not found
  // (e.g. an explicit --output_bin unrelated to the encoder name), prefix the
  // decoder base instead so the two files stay distinct.
  const std::string enc_out = weight_path;
  std::string dec_name;
  auto pos = out_name.find(enc_base);
  if (!enc_base.empty() && pos != std::string::npos) {
    dec_name = out_name.substr(0, pos) + dec_base +
               out_name.substr(pos + enc_base.size());
  } else {
    dec_name = dec_base + "_" + out_name;
  }
  const std::string dec_out = (out_dir / dec_name).string();

  std::cout << "[ScreenAICaption] save_weight -> encoder: " << enc_out << "\n";
  std::cout << "[ScreenAICaption] save_weight -> decoder: " << dec_out << "\n";

  encoder_->save_weight(enc_out, dtype, layer_dtype_map, target_isa);
  decoder_->save_weight(dec_out, dtype, layer_dtype_map, target_isa);
}

// ---------------------------------------------------------------------------
// generateFromEncoderHidden  (single shared decode path)
//
// Token-counting convention:
//   prefill phase  = encoder forward pass + cross-cache fill
//                    (BD_ENC_LEN image patch tokens, measured in prefill_us_)
//   generation phase = the FULL autoregressive token loop
//                    (all decoded tokens from pos=0 to EOS, measured in
//                     generation_us_)
//   prefill_tokens + generation_tokens accounts for all produced tokens.
// ---------------------------------------------------------------------------

void ScreenAICaption::generateFromEncoderHidden(
  const std::vector<float> &enc_hidden, bool log_output, long long prefill_us) {
  const size_t expected = static_cast<size_t>(BertDecoder::BD_ENC_LEN) *
                          static_cast<size_t>(BertDecoder::BD_DIM);
  if (enc_hidden.size() != expected)
    throw std::runtime_error(
      "ScreenAICaption: encoder_hidden size mismatch (got " +
      std::to_string(enc_hidden.size()) + ", expected " +
      std::to_string(expected) + ")");

  // Fill the per-layer cross-attention K/V cache ONCE from the encoder hidden;
  // it stays fixed for the whole generation loop.
  decoder_->prefillCrossCache(enc_hidden.data(), DECODER_START_TOKEN_ID);
  decoder_->setKVCachePosition(0);

  token_ids_.clear();
  token_ids_.push_back(DECODER_START_TOKEN_ID); // golden includes the [CLS]

  int cur = DECODER_START_TOKEN_ID;
  const int max_steps = NUM_TO_GENERATE > 0 ? NUM_TO_GENERATE : 32;

  // ------------------------------------------------------------------
  // Generation phase: FULL autoregressive loop (pos 0 .. EOS or max).
  // All decoded tokens (including position 0) are counted in generation_tokens.
  // ------------------------------------------------------------------
  auto start_generation = std::chrono::high_resolution_clock::now();
  unsigned int generation_cnt = 0;

  for (unsigned int pos = 0; pos < static_cast<unsigned int>(max_steps);
       ++pos) {
    std::vector<float> logits =
      decoder_->decodeToken(enc_hidden.data(), cur, pos);

    // Greedy argmax (do_sample == false).
    int next = 0;
    float best = logits[0];
    for (int v = 1; v < static_cast<int>(BertDecoder::BD_NUM_VOCAB); ++v) {
      if (logits[static_cast<size_t>(v)] > best) {
        best = logits[static_cast<size_t>(v)];
        next = v;
      }
    }

    token_ids_.push_back(next);
    ++generation_cnt;
    if (next == EOS_TOKEN_ID)
      break;
    cur = next;
  }

  auto finish_generation = std::chrono::high_resolution_clock::now();
  long long generation_us =
    std::chrono::duration_cast<std::chrono::microseconds>(finish_generation -
                                                          start_generation)
      .count();

  // ------------------------------------------------------------------
  // Total wall-clock: prefill_us was measured by the caller (encoder encode
  // + cross-cache fill); generation_us is the token loop above.
  // ------------------------------------------------------------------
  long long total_us = prefill_us + generation_us;

  // Decode caption text (skip special tokens are not stripped by the C++
  // tokenizer Decode; we drop the leading [CLS] and trailing [SEP] explicitly
  // so the text matches the golden skip_special_tokens decode).
  caption_.clear();
  if (tokenizer) {
    std::vector<int32_t> body;
    body.reserve(token_ids_.size());
    for (int t : token_ids_) {
      if (t == DECODER_START_TOKEN_ID || t == EOS_TOKEN_ID)
        continue;
      body.push_back(static_cast<int32_t>(t));
    }
    caption_ = tokenizer->Decode(body);
  }

  // Guard every TPS division: use µs duration converted to seconds.
  // prefill_tps: BD_ENC_LEN patch tokens processed per second.
  // generation_tps: autoregressive tokens per second.
  double prefill_ms = static_cast<double>(prefill_us) / 1000.0;
  double generation_ms = static_cast<double>(generation_us) / 1000.0;
  double total_ms = static_cast<double>(total_us) / 1000.0;

  double prefill_tps =
    (prefill_us > 0)
      ? (static_cast<double>(BertDecoder::BD_ENC_LEN) / (prefill_us / 1e6))
      : 0.0;
  double generation_tps =
    (generation_us > 0 && generation_cnt > 0)
      ? (static_cast<double>(generation_cnt) / (generation_us / 1e6))
      : 0.0;

  size_t peak_memory = getPeakMemoryKb();

  if (log_output) {
    std::cout << "[ScreenAICaption] token_ids:";
    for (int t : token_ids_)
      std::cout << " " << t;
    std::cout << "\n[ScreenAICaption] caption: " << caption_ << "\n";

    std::cout << "\n";
    std::cout << "=================[ ScreenAICaption with NNTrainer "
                 "]===================\n";
    std::cout << "encoder+prefill: " << BertDecoder::BD_ENC_LEN << " tokens, "
              << prefill_ms << " ms, " << prefill_tps << " TPS\n";
    std::cout << "generation: " << generation_cnt << " tokens, "
              << generation_ms << " ms, " << generation_tps << " TPS\n";
    std::cout << "total: " << total_ms << " ms\n";
    std::cout << "peak memory: " << peak_memory << " KB\n";
    std::cout << "============================================================="
                 "=========\n";
  }

  // Populate performance_metrics (inherited from Transformer).
  // prefill_tokens: BD_ENC_LEN image patch tokens processed by the encoder.
  // generation_tokens: all decoded tokens in the autoregressive loop (pos
  // 0..N).
  performance_metrics.prefill_tokens =
    static_cast<unsigned int>(BertDecoder::BD_ENC_LEN);
  performance_metrics.prefill_duration_ms = prefill_ms;
  performance_metrics.generation_tokens = generation_cnt;
  performance_metrics.generation_duration_ms = generation_ms;
  performance_metrics.total_duration_ms = total_ms;
  performance_metrics.peak_memory_kb = peak_memory;

  // Mark this instance as having completed a successful run so that
  // getPerformanceMetrics() (gated on hasRun()) is accessible.
  has_run_ = true;
}

// ---------------------------------------------------------------------------
// run  (image path -> caption, normal C++ resize path)
// ---------------------------------------------------------------------------

void ScreenAICaption::run(const WSTR prompt, bool do_sample,
                          const WSTR system_prompt, const WSTR tail_prompt,
                          bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  if (!is_initialized)
    throw std::runtime_error("ScreenAICaption::run: call initialize() first");

  const std::string image_path(prompt);

  // ------------------------------------------------------------------
  // Prefill phase: encoder forward pass (BD_ENC_LEN image patches -> hidden)
  // Timing covers: image decode/resize, encoder forward, plus the cross-cache
  // prefill inside generateFromEncoderHidden.
  // ------------------------------------------------------------------
  auto start_prefill = std::chrono::high_resolution_clock::now();
  std::vector<float> enc_hidden = encoder_->encode(image_path);
  auto finish_prefill = std::chrono::high_resolution_clock::now();
  long long prefill_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           finish_prefill - start_prefill)
                           .count();

  // ------------------------------------------------------------------
  // generateFromEncoderHidden handles cross-cache prefill, the full
  // autoregressive generation loop, caption decoding, banner printing, and
  // metrics population in a single shared path used by both run() and
  // runFromPixels().
  // ------------------------------------------------------------------
  generateFromEncoderHidden(enc_hidden, log_output, prefill_us);
}

// ---------------------------------------------------------------------------
// runFromPixels  (parity Stage A: PyTorch-preprocessed pixels)
// ---------------------------------------------------------------------------

std::string ScreenAICaption::runFromPixels(const float *pixel_data,
                                           size_t pixel_count,
                                           bool log_output) {
  if (!is_initialized)
    throw std::runtime_error(
      "ScreenAICaption::runFromPixels: call initialize() first");

  // Prefill phase: encodePixels (no image file I/O, just encoder forward pass).
  auto start_prefill = std::chrono::high_resolution_clock::now();
  std::vector<float> enc_hidden =
    encoder_->encodePixels(pixel_data, pixel_count);
  auto finish_prefill = std::chrono::high_resolution_clock::now();
  long long prefill_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           finish_prefill - start_prefill)
                           .count();

  // Shared decode path: populates metrics and sets has_run_ = true.
  generateFromEncoderHidden(enc_hidden, log_output, prefill_us);
  return caption_;
}

} // namespace causallm
