// SPDX-License-Identifier: Apache-2.0
/** @file qwen_cpu_probe.cpp
 * @brief Record unmodified CPU logits and token IDs for model parity checks.
 * @author Codex (OpenAI)
 * @bug Supports batch-1 Qwen3 inference with raw FP32 logits only.
 */
#include <causal_lm.h>
#include <factory.h>
#include <model_registry.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
using json = causallm::json;

json readJson(const std::filesystem::path &path) {
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("Cannot read " + path.string());
  json value;
  stream >> value;
  return value;
}

/** Observes logits without changing the selected distribution. */
class Capture final : public causallm::LogitsProcessor {
public:
  explicit Capture(std::filesystem::path output) : output_(std::move(output)) {}

  void process(float *logits, unsigned int count, unsigned int batch) override {
    if (batch != 0)
      throw std::runtime_error("The CPU validation probe requires batch=1");
    const auto path =
      output_.string() + ".step" + std::to_string(rows.size()) + ".f32";
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(logits), count * sizeof(float));
    if (!stream)
      throw std::runtime_error("Cannot write logits to " + path);
    std::vector<unsigned int> order(count);
    std::iota(order.begin(), order.end(), 0);
    const size_t n = std::min<size_t>(10, count);
    const bool finite = std::all_of(logits, logits + count,
                                    [](float v) { return std::isfinite(v); });
    json top = json::array();
    if (finite) {
      std::partial_sort(
        order.begin(), order.begin() + n, order.end(),
        [logits](auto a, auto b) { return logits[a] > logits[b]; });
      for (size_t i = 0; i < n; ++i)
        top.push_back({{"token_id", order[i]}, {"logit", logits[order[i]]}});
    }
    rows.push_back({{"logits_file", path},
                    {"vocab_size", count},
                    {"all_finite", finite},
                    {"top10", top}});
  }

  void acceptToken(unsigned int id, unsigned int batch) override {
    if (batch != 0)
      throw std::runtime_error("The CPU validation probe requires batch=1");
    ids.push_back(id);
  }

  std::vector<unsigned int> ids;
  json rows = json::array();

private:
  std::filesystem::path output_;
};
} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr
      << "Usage: qwen_cpu_probe <model_dir> <request.json> <report.json>\n";
    return 2;
  }
  try {
    const auto path = std::filesystem::absolute(argv[1]);
    auto config = readJson(path / "config.json");
    auto generation = readJson(path / "generation_config.json");
    auto nntr = readJson(path / "nntr_config.json");
    const auto request = readJson(argv[2]);
    for (const char *key : {"tokenizer_file", "model_file_name"}) {
      auto file = std::filesystem::path(nntr.at(key).get<std::string>());
      if (file.is_relative())
        nntr[key] = (path / file).string();
    }
    if (nntr.at("batch_size").get<unsigned int>() != 1)
      throw std::runtime_error("The CPU validation probe requires batch=1");
    causallm::registerAllModels();
    auto model = causallm::Factory::Instance().create("Qwen3ForCausalLM",
                                                      config, generation, nntr);
    if (!model)
      throw std::runtime_error("Qwen3ForCausalLM is not registered");
    model->initialize();
    model->load_weight(nntr.at("model_file_name").get<std::string>());
    auto run = [&](causallm::Transformer &running, const json &turn,
                   const std::string &output) {
      const auto prompt = turn.at("rendered_prompt").get<std::string>();
      const auto input_ids = running.getTokenizer()->Encode(prompt);
      if (turn.contains("input_ids") && json(input_ids) != turn.at("input_ids"))
        throw std::runtime_error("HF and nntrainer input token IDs disagree");
      const int before = running.getKvLen();
      Capture capture(output);
      running.setLogitsProcessor(&capture);
      try {
        running.run(prompt, false, "", "", false);
      } catch (...) {
        running.setLogitsProcessor(nullptr);
        throw;
      }
      running.setLogitsProcessor(nullptr);
      return json{
        {"request", turn},           {"input_ids", input_ids},
        {"position_before", before}, {"position_after", running.getKvLen()},
        {"token_ids", capture.ids},  {"steps", capture.rows}};
    };
    json report = run(*model, request, argv[3]);
    if (request.contains("continuations")) {
      auto *causal = dynamic_cast<causallm::CausalLM *>(model.get());
      if (!causal)
        throw std::runtime_error("Continuation requires a CausalLM session");
      std::unique_ptr<causallm::Transformer> resumed;
      causallm::CausalLM *resumed_causal = nullptr;
      if (request.value("snapshot_resume", false)) {
        const std::string snapshot = std::string(argv[3]) + ".kv";
        const int position = model->getKvLen();
        causal->save_kvcache(snapshot, position);
        resumed = causallm::Factory::Instance().create(
          "Qwen3ForCausalLM", config, generation, nntr);
        if (!resumed)
          throw std::runtime_error("Qwen3 resume model is not registered");
        resumed->initialize();
        resumed->load_weight(nntr.at("model_file_name").get<std::string>());
        resumed_causal = dynamic_cast<causallm::CausalLM *>(resumed.get());
        if (!resumed_causal)
          throw std::runtime_error("Snapshot resume requires CausalLM");
        const auto restored =
          resumed_causal->resumeFromKVCacheFile(snapshot, 0);
        report["snapshot"] = {{"path", snapshot},
                              {"bytes", std::filesystem::file_size(snapshot)},
                              {"saved_position", position},
                              {"restored_position", restored}};
        report["resumed_continuations"] = json::array();
      }
      report["continuations"] = json::array();
      for (const auto &turn : request.at("continuations")) {
        causal->setSessionTokenPosition(model->getKvLen());
        const std::string prefix =
          std::string(argv[3]) + ".turn" +
          std::to_string(report["continuations"].size() + 1);
        report["continuations"].push_back(run(*model, turn, prefix));
        if (resumed) {
          resumed_causal->setSessionTokenPosition(resumed->getKvLen());
          report["resumed_continuations"].push_back(
            run(*resumed, turn, prefix + ".resumed"));
        }
      }
    }
    std::ofstream stream(argv[3]);
    stream << report.dump(2) << '\n';
    if (!stream)
      throw std::runtime_error("Cannot write probe report");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
