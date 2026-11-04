// SPDX-License-Identifier: Apache-2.0
/**
 * @file qwen_cpu_benchmark.cpp
 * @brief Fixed-work CPU timings, separate from the correctness logits probe.
 * @author Codex (OpenAI)
 * @bug Timings include the run API and token selection, not attention alone.
 */
#include <causal_lm.h>
#include <factory.h>
#include <model_registry.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
using json = causallm::json;
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::duration elapsed) {
  return std::chrono::duration<double, std::milli>(elapsed).count();
}

json readJson(const std::filesystem::path &path) {
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("Cannot read " + path.string());
  json value;
  stream >> value;
  return value;
}

/** One scalar write forces the same non-EOS work on every profile.
 * No logits scan, finite check, dump, or correctness comparison is timed here.
 */
class FixedTokens final : public causallm::LogitsProcessor {
public:
  explicit FixedTokens(unsigned int expected) : expected_(expected) {
    ready.reserve(expected);
    ids.reserve(expected);
  }

  void process(float *logits, unsigned int count, unsigned int batch) override {
    if (batch || ready.size() >= expected_ || count <= 35)
      throw std::runtime_error("Invalid fixed-work benchmark callback");
    ready.push_back(Clock::now());
    logits[32 + ids.size() % 4] = std::numeric_limits<float>::max();
  }

  void acceptToken(unsigned int id, unsigned int batch) override {
    if (batch || id != 32 + ids.size() % 4)
      throw std::runtime_error("Fixed non-EOS token was not selected");
    ids.push_back(id);
  }

  std::vector<Clock::time_point> ready;
  std::vector<unsigned int> ids;

private:
  unsigned int expected_;
};
} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr
      << "Usage: qwen_cpu_benchmark <model_dir> <request.json> <report.json>\n";
    return 2;
  }
  try {
    const auto directory = std::filesystem::absolute(argv[1]);
    auto config = readJson(directory / "config.json");
    auto generation = readJson(directory / "generation_config.json");
    auto nntr = readJson(directory / "nntr_config.json");
    const auto request = readJson(argv[2]);
    const unsigned int context = request.at("context_tokens");
    const unsigned int decode = request.at("decode_tokens");
    const unsigned int warmup = request.at("warmup");
    const unsigned int repetitions = request.at("repetitions");
    if (!context || !decode || !repetitions || nntr.at("batch_size") != 1 ||
        nntr.at("skip_prefill") != false ||
        nntr.at("max_seq_len").get<unsigned int>() <= context + decode)
      throw std::runtime_error(
        "Invalid benchmark geometry or generation budget");
    for (const char *key : {"tokenizer_file", "model_file_name"}) {
      const auto path = std::filesystem::path(nntr.at(key).get<std::string>());
      if (path.is_relative())
        nntr[key] = (directory / path).string();
    }
    // IDs 32..35 are ordinary Qwen vocabulary entries, never EOS. Keep EOS
    // enabled in the model and force these IDs rather than editing stop rules.
    auto eos = generation.at("eos_token_id");
    if (!eos.is_array())
      eos = json::array({eos});
    for (const auto &id : eos)
      if (id.get<unsigned int>() >= 32 && id.get<unsigned int>() <= 35)
        throw std::runtime_error("Benchmark token conflicts with EOS");

    causallm::registerAllModels();
    const auto factory_begin = Clock::now();
    auto model = causallm::Factory::Instance().create("Qwen3ForCausalLM",
                                                      config, generation, nntr);
    const auto initialize_begin = Clock::now();
    if (!model)
      throw std::runtime_error("Qwen3ForCausalLM is not registered");
    model->initialize();
    const auto load_begin = Clock::now();
    model->load_weight(nntr.at("model_file_name").get<std::string>());
    const auto prepare_begin = Clock::now();
    auto *causal = dynamic_cast<causallm::CausalLM *>(model.get());
    if (!causal)
      throw std::runtime_error("Benchmark needs the CausalLM session API");
    std::string prompt;
    prompt.reserve(context * 2);
    for (unsigned int i = 0; i < context; ++i)
      prompt += " a";
    const auto input_ids = model->getTokenizer()->Encode(prompt, true);
    if (input_ids.size() != context)
      throw std::runtime_error(
        "Repeated prompt did not encode to the requested length");
    model->setNumToGenerate(decode);
    const auto actual_plane = model->prefillPlaneFor(0);
    const auto actual_chunk = model->prefillDriveChunk();
    if (request.value("expected_plane", actual_plane) != actual_plane ||
        request.value("expected_prefill_chunk", actual_chunk) != actual_chunk ||
        (request.value("single_block", false) &&
         (actual_chunk != 0 || std::getenv("NNTR_PREFILL_CHUNK") != nullptr)))
      throw std::runtime_error("Requested prefill strategy was not applied");
    const auto prepare_end = Clock::now();
    json report = {
      {"request", request},
      {"input_ids", input_ids},
      {"activation_plane", actual_plane},
      {"effective_prefill_chunk", actual_chunk},
      {"prefill_chunk_environment_present",
       std::getenv("NNTR_PREFILL_CHUNK") != nullptr},
      {"expected_prefill_graph_calls",
       actual_chunk ? (context + actual_chunk - 1) / actual_chunk : 1},
      {"factory_ms", milliseconds(initialize_begin - factory_begin)},
      {"initialize_ms", milliseconds(load_begin - initialize_begin)},
      {"weight_load_and_pack_ms", milliseconds(prepare_begin - load_begin)},
      {"tokenizer_prompt_prepare_ms",
       milliseconds(prepare_end - prepare_begin)},
      {"timing_scope",
       "run API; fixed token processor; no correctness observer"},
      {"trials", json::array()}};
    for (unsigned int iteration = 0; iteration < warmup + repetitions;
         ++iteration) {
      causal->setSessionTokenPosition(0);
      FixedTokens tokens(decode + 1);
      model->setLogitsProcessor(&tokens);
      const auto begin = Clock::now();
      model->run(prompt, false, "", "", false);
      const auto end = Clock::now();
      model->setLogitsProcessor(nullptr);
      const auto metrics = model->getPerformanceMetrics();
      if (tokens.ids.size() != decode + 1 ||
          tokens.ready.size() != decode + 1 ||
          metrics.prefill_tokens != context ||
          metrics.generation_tokens != decode ||
          metrics.prompt_truncated_tokens ||
          model->prefillPlaneFor(0) != actual_plane ||
          model->prefillDriveChunk() != actual_chunk ||
          model->getKvLen() != int(context + decode))
        throw std::runtime_error(
          "Truncation, early EOS, or cache-position mismatch");
      std::vector<double> intervals;
      intervals.reserve(decode);
      for (size_t i = 1; i < tokens.ready.size(); ++i)
        intervals.push_back(
          milliseconds(tokens.ready[i] - tokens.ready[i - 1]));
      report["trials"].push_back(
        {{"iteration", iteration},
         {"warmup", iteration < warmup},
         {"run_wall_ms", milliseconds(end - begin)},
         {"run_to_first_logits_ms", milliseconds(tokens.ready.front() - begin)},
         {"decode_logits_intervals_ms", intervals},
         {"prefill_compute_ms", metrics.prefill_duration_ms},
         {"decode_api_ms", metrics.generation_duration_ms},
         {"decode_first_step_ms", metrics.first_token_ms},
         {"decode_steady_median_ms", metrics.decode_steady_ms},
         {"peak_memory_kb", metrics.peak_memory_kb},
         {"position_after", model->getKvLen()},
         {"forced_ids", tokens.ids}});
    }
    std::ofstream output(argv[3]);
    output << report.dump(2) << '\n';
    if (!output)
      throw std::runtime_error("Cannot save benchmark report");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
