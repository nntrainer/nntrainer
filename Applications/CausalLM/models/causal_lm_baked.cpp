// SPDX-License-Identifier: Apache-2.0
/** @file causal_lm_baked.cpp
 * @brief Offline system-prefix baking and lazy persistent radix import.
 */
#include <causal_lm.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <htp_backend/htp_attention_state.h>
#include <htp_backend/htp_runtime_config.h>
#ifdef ENABLE_HEXKL
#include <gemma4_htp_causallm.h>
#endif
#include <optional>
#include <streamer.h>
namespace causallm {
namespace {
#ifdef ENABLE_HEXKL
template <typename T> json setting(const T &v) { return v; }
template <typename T> json setting(const std::optional<T> &v) {
  return v ? json(*v) : json();
}
void fingerprintSetting(json &j, const char *key) {
  if (j.contains(key) && j[key].is_string() &&
      !j[key].get<std::string>().empty())
    j[key] = BakedPrefix::fingerprintFile(j[key].get<std::string>());
}
#endif
} // namespace
json CausalLM::deviceSystemPrefixIdentity() {
#ifndef ENABLE_HEXKL
  throw std::logic_error("Baked system prefixes require HTP");
#else
  if (!system_prefix_identity_.is_null())
    return system_prefix_identity_;
  auto nntr = system_prefix_nntr_;
  for (const char *key :
       {"sample_input", "chat_input", "num_to_generate", "system_prompt",
        "kv_prefix_flash", "kv_prebaked_prefixes", "model_file_name", "htp",
        "moe"})
    nntr.erase(key);
  nntr["weights"] = BakedPrefix::fingerprintFile(weight_path_);
  for (const char *key :
       {"tokenizer_file", "embedding_file_name", "ple_file_name",
        "act_scale_json", "moe_hadamard_signs", "kv_quant_file"})
    fingerprintSetting(nntr, key);
  json runtime;
  const auto &r = nntrainer::htpRuntimeConfig();
#define NNTR_BAKED_HTP(type, name, default_value)                              \
  runtime["htp"][#name] = setting(r.htp.name);
  NNTR_HTP_CONFIG_FIELDS(NNTR_BAKED_HTP)
#undef NNTR_BAKED_HTP
#define NNTR_BAKED_MOE(type, name, default_value)                              \
  runtime["moe"][#name] = setting(r.moe.name);
  NNTR_MOE_CONFIG_FIELDS(NNTR_BAKED_MOE)
#undef NNTR_BAKED_MOE
  fingerprintSetting(runtime["htp"], "static_linear");
  fingerprintSetting(runtime["moe"], "tl2_prepared_file");
  json geometry = json::array();
  for (int l = 0; l < NUM_LAYERS; ++l)
    geometry.push_back(
      {{"width", getKVCacheWidth(l)}, {"rows", getKVCacheRows(l)}});
  const char *threads = std::getenv("NNTR_NUM_THREADS");
  system_prefix_identity_ = {
    {"contract", "original-q8-q2-u16-native-prefix-v1"},
    {"model", system_prefix_config_},
    {"nntr", nntr},
    {"runtime", runtime},
    {"max_seq", MAX_SEQ_LEN},
    {"plane", INIT_SEQ_LEN},
    {"chunk", prefillDriveChunk()},
    {"geometry", geometry},
    {"threads", threads ? threads : "auto"}};
  return system_prefix_identity_;
#endif
}
json CausalLM::bakeDeviceSystemPrefix(const std::string &system_prompt,
                                      const std::string &directory) {
#ifndef ENABLE_HEXKL
  (void)system_prompt;
  (void)directory;
  throw std::logic_error("System-prefix baking requires HTP");
#else
  if (!device_kv_manager_ || !device_prefill_capture_ || USE_KVCACHE ||
      system_prefix_writer_ || getSessionTokenPosition() != 0 ||
      REPETITION_PENALTY != 1.0f)
    throw std::logic_error(
      "Bake requires original q8 radix and an empty active session");
  const auto chunk = prefillDriveChunk();
  if (!chunk || chunk != INIT_SEQ_LEN ||
      prefillPlaneFor(MAX_SEQ_LEN - 1) != INIT_SEQ_LEN)
    throw std::logic_error(
      "Bake requires a stable native prefill plane (use init_seq_len=4096)");
  ensureTokenizer();
  const auto input = tokenizer->Encode(system_prompt, true);
  if (input.size() >= MAX_SEQ_LEN)
    throw std::length_error("System prefix exceeds context capacity");
  const unsigned safe = static_cast<unsigned>(input.size() / chunk * chunk);
  if (!safe)
    throw std::invalid_argument(
      "System prompt has no complete native prefill boundary; dynamic scales "
      "forbid splitting/padding it");
  if (safe + 4u > MAX_SEQ_LEN || EOS_TOKEN_ID.empty())
    throw std::length_error("No room to bake a complete native prefix");
  std::vector<unsigned> tokens(input.begin(), input.begin() + safe);
  // Only the first safe rows are baked. Three probe input tokens trigger their
  // ordinary complete native calls; they are not stored or added to requests.
  std::vector<int> probe(input.begin(), input.begin() + safe);
  for (unsigned i = 0; i < 3; ++i)
    probe.push_back(EOS_TOKEN_ID.front());
  const auto text = tokenizer->Decode(probe);
  if (tokenizer->Encode(text, true) != probe)
    throw std::invalid_argument(
      "Bake probe tokenization changed the system prefix");
  const auto identity_start = std::chrono::steady_clock::now();
  const auto identity = deviceSystemPrefixIdentity();
  const double identity_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                              identity_start)
      .count();
  system_prefix_writer_ = std::make_unique<BakedPrefix::Writer>(
    directory, identity, std::move(tokens), input.size());
  const int old_count = NUM_TO_GENERATE;
  auto *old_streamer = streamer_;
  NUM_TO_GENERATE = 1;
  streamer_ = nullptr;
  const auto started = std::chrono::steady_clock::now();
  try {
    run(text, false, "", "", false);
    auto result = system_prefix_writer_->finish();
    result["identity_ms"] = identity_ms;
    result["bake_ms"] = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - started)
                          .count();
    system_prefix_writer_.reset();
    last_generated_ids_.clear();
    output_list.clear();
    NUM_TO_GENERATE = old_count;
    streamer_ = old_streamer;
    setSessionTokenPosition(0);
    return result;
  } catch (...) {
    system_prefix_writer_.reset();
    last_generated_ids_.clear();
    output_list.clear();
    NUM_TO_GENERATE = old_count;
    streamer_ = old_streamer;
    setSessionTokenPosition(0);
    throw;
  }
#endif
}
json CausalLM::loadDeviceSystemPrefix(const std::string &directory) {
#ifndef ENABLE_HEXKL
  (void)directory;
  throw std::logic_error("Baked system prefixes require HTP");
#else
  if (!device_kv_manager_ || !device_prefill_capture_ ||
      system_prefix_writer_ || REPETITION_PENALTY != 1.0f)
    throw std::logic_error("Baked import requires original q8 radix");
  const auto *gemma = dynamic_cast<const Gemma4HtpTransformer *>(this);
  if (!gemma)
    throw std::logic_error("Baked prefix import currently supports Gemma4 HTP");
  const auto started = std::chrono::steady_clock::now();
  auto asset = BakedPrefix::load(directory, deviceSystemPrefixIdentity(),
                                 MAX_SEQ_LEN, NUM_LAYERS);
  for (unsigned t : asset.tokens)
    if (t >= NUM_VOCAB)
      throw std::invalid_argument("Baked token outside vocabulary");
  auto history = std::make_shared<DevicePrefillHistory>();
  for (const auto &[position, saved] : asset.frames) {
    if (saved->plane != INIT_SEQ_LEN || saved->chunk != prefillDriveChunk())
      throw std::invalid_argument("Baked native call plan differs");
    for (unsigned l = 0; l < NUM_LAYERS; ++l) {
      const auto width = getKVCacheWidth(l), heads = gemma->getKVHeadCount(l);
      const auto &layer = saved->layout->layers.at(l);
      const auto &c = *saved->calibration.at(l);
      if (layer.width != width || position - layer.base > getKVCacheRows(l) ||
          !c.calibrated || c.scale_k.size() != heads ||
          c.kmax.size() != heads || c.scale_v.size() != width ||
          c.vmax.size() != width || c.q_enc.size() != 2u * NUM_HEADS ||
          c.out_enc.size() != 2u * NUM_HEADS || c.floor_k != layer.dropped_k ||
          c.floor_v != layer.dropped_v ||
          (!c.floor_k.empty() && c.floor_k.size() != width) ||
          c.floor_v.size() != c.floor_k.size())
        throw std::invalid_argument("Baked layer/calibration geometry differs");
      for (float v : c.scale_k)
        if (v <= 0)
          throw std::invalid_argument("Invalid baked K scale");
      for (float v : c.scale_v)
        if (v <= 0)
          throw std::invalid_argument("Invalid baked V scale");
      for (size_t i = 0; i < c.q_enc.size(); i += 2)
        if (c.q_enc[i] <= 0 || c.out_enc[i] <= 0)
          throw std::invalid_argument("Invalid baked Q/output scale");
      for (const auto *v : {&c.kmax, &c.vmax, &c.floor_k, &c.floor_v})
        for (float f : *v)
          if (f < 0)
            throw std::invalid_argument("Invalid baked running maximum");
    }
    DevicePrefillFrame f;
    f.from = saved->from;
    f.plane = saved->plane;
    f.chunk = saved->chunk;
    f.baked = saved;
    f.calibration = saved->calibration;
    history->emplace(position, std::move(f));
  }
  auto previous = device_prefill_history_;
  device_prefill_history_ = history;
  try {
    device_kv_manager_->publish(asset.tokens, asset.tokens.size(), 0);
  } catch (...) {
    device_prefill_history_ = std::move(previous);
    throw;
  }
  device_prefill_history_ = std::move(previous);
  ++system_prefix_stats_.loaded_assets;
  return {{"directory", directory},
          {"baked_tokens", asset.tokens.size()},
          {"system_tokens", asset.system_tokens},
          {"boundaries", asset.frames.size()},
          {"load_ms", std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started)
                        .count()}};
#endif
}
} // namespace causallm
