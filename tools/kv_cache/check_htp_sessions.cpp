// SPDX-License-Identifier: Apache-2.0
/** @file check_htp_sessions.cpp
 * @brief Android Gemma4 device sessions/prefix/Flash correctness and timings.
 * Build against the same CausalLM library and ABI as nntr_causallm.
 * Usage: check_htp_sessions MODEL_DIR FIXTURE_JSON OUTPUT_JSON
 * Fixture: {"seeds":[{"prompt":"..."}, ...], "query":"..."}.
 */
#include <causal_lm.h>
#include <chrono>
#include <factory.h>
#include <filesystem>
#include <fstream>
#include <htp_model_config.h>
#include <htp_runtime_config.h>
#include <iostream>
#include <model_registry.h>

using json = causallm::json;
using Clock = std::chrono::steady_clock;
static json read(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Cannot read " + path.string());
  json value;
  file >> value;
  return value;
}
static void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}
int main(int argc, char **argv) {
  if (argc != 4 && argc != 5)
    return 2;
  try {
    std::cerr << "CHECK_PHASE config" << std::endl;
    const std::filesystem::path dir = argv[1];
    auto cfg = read(dir / "config.json");
    auto gen = read(dir / "generation_config.json");
    auto nc = read(dir / "nntr_config.json");
    const auto fixture = read(argv[2]);
    std::cerr << "CHECK_PHASE runtime" << std::endl;
    nntrainer::configureHtpRuntime(causallm::normalizeHtpRuntimeConfig(nc));
    causallm::registerAllModels();
    auto model =
      causallm::Factory::Instance().create("Gemma4ForCausalLM", cfg, gen, nc);
    std::cerr << "CHECK_PHASE initialize" << std::endl;
    model->initialize();
    std::cerr << "CHECK_PHASE load" << std::endl;
    model->load_weight(nc["model_file_name"].get<std::string>());
    std::cerr << "CHECK_PHASE repack" << std::endl;
    model->repack_weight();
    std::cerr << "CHECK_PHASE ready" << std::endl;
    auto *c = dynamic_cast<causallm::CausalLM *>(model.get());
    require(c != nullptr, "CausalLM unavailable");
    json report = {{"config", nc},
                   {"reference", json::array()},
                   {"switched", json::array()},
                   {"prefix", json::array()}};
    auto write = [&] {
      std::ofstream file(argv[3]);
      file << report.dump(2) << '\n';
    };
    auto run = [&](const std::string &text, bool reset) {
      if (reset)
        c->setSessionTokenPosition(0);
      const auto start = Clock::now();
      model->run(text, false, "", "", false);
      const auto &m = c->getPerformanceMetrics();
      require(!m.prompt_truncated_tokens, "Prompt truncated");
      return json{{"ids", c->getLastGeneratedTokenIds()},
                  {"position", c->getKvLen()},
                  {"prefill_tokens", m.prefill_tokens},
                  {"ttft_ms", m.ttft_ms},
                  {"decode_steady_ms", m.decode_steady_ms},
                  {"decode_steady_tokens", m.decode_steady_tokens},
                  {"kv_bytes", c->getDeviceKVCacheBytes()},
                  {"wall_ms", std::chrono::duration<double, std::milli>(
                                Clock::now() - start)
                                .count()}};
    };
    const std::string query = fixture["query"];
    std::vector<std::vector<json>> refs;
    std::vector<std::string> prompts;
    std::vector<std::vector<int>> seed_histories;
    run("<bos><|turn>user\nSay READY.<turn|>\n<|turn>model\n", true);
    for (const auto &seed : fixture["seeds"]) {
      const std::string prompt = seed["prompt"];
      prompts.push_back(prompt);
      std::vector<json> turns;
      turns.push_back(run(prompt, true));
      auto history = model->getTokenizer()->Encode(prompt, true);
      const auto ids = c->getLastGeneratedTokenIds();
      history.insert(history.end(), ids.begin(), ids.end());
      seed_histories.push_back(history);
      for (unsigned turn = 1; turn < 3; ++turn)
        turns.push_back(run(query, false));
      refs.push_back(turns);
      report["reference"].push_back(turns);
      write();
    }
    if (!nc.at("htp").value("kv_paged", false)) {
      report["passed"] = true;
      write();
      return 0;
    }
    if (argc == 5) {
      const auto baseline = read(argv[4]);
      for (size_t i = 0; i < refs.size(); ++i)
        for (size_t t = 0; t < refs[i].size(); ++t)
          require(refs[i][t]["ids"] == baseline["reference"][i][t]["ids"] &&
                    refs[i][t]["position"] ==
                      baseline["reference"][i][t]["position"],
                  "Paged/linear model output or position differs");
    }
    // A complete-history prefill is a different reference from a resident
    // M1 continuation. Prefix hits must match the former exactly.
    std::vector<json> prefix_refs;
    json flash_ref;
    for (size_t i = 0; i < prompts.size(); ++i) {
      auto full = seed_histories[i];
      const auto delta = model->getTokenizer()->Encode(query, false);
      full.insert(full.end(), delta.begin(), delta.end());
      const auto text = model->getTokenizer()->Decode(full);
      prefix_refs.push_back(run(text, true));
      if (i + 1 == prompts.size())
        flash_ref = run(query, false);
    }
    report["prefix_reference"] = prefix_refs;
    report["flash_reference"] = flash_ref;
    c->setSessionTokenPosition(0);
    c->enableDeviceKVManager(8, 8);
    std::vector<uint32_t> sessions{0};
    for (size_t i = 1; i < prompts.size(); ++i)
      sessions.push_back(c->createDeviceKVSession());
    std::vector<uint32_t> forks;
    for (unsigned turn = 0; turn < 3; ++turn) {
      for (size_t i = 0; i < prompts.size(); ++i) {
        const auto start = Clock::now();
        c->selectDeviceKVSession(sessions[i]);
        const double switch_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - start)
            .count();
        auto result = run(turn ? query : prompts[i], false);
        require(result["ids"] == refs[i][turn]["ids"] &&
                  result["position"] == refs[i][turn]["position"],
                "Session output/position changed");
        result["session"] = i;
        result["turn"] = turn;
        result["switch_ms"] = switch_ms;
        report["switched"].push_back(result);
        write();
        if (turn == 0)
          forks.push_back(c->forkDeviceKVSession());
      }
    }
    // A fork retained seed state while its parent advanced two turns.
    for (size_t i = 0; i < forks.size(); ++i) {
      c->selectDeviceKVSession(forks[i]);
      const auto result = run(query, false);
      require(result["ids"] == refs[i][1]["ids"], "Fork was overwritten");
    }
    const auto scratch = c->createDeviceKVSession();
    c->selectDeviceKVSession(scratch);
    for (size_t i = 0; i < prompts.size(); ++i) {
      auto full = seed_histories[i];
      const auto delta = model->getTokenizer()->Encode(query, false);
      full.insert(full.end(), delta.begin(), delta.end());
      const auto text = model->getTokenizer()->Decode(full);
      require(model->getTokenizer()->Encode(text, true) == full,
              "History retokenization changed");
      const auto before = c->deviceKVManagerStats();
      auto result = run(text, true);
      const auto after = c->deviceKVManagerStats();
      const bool reusable =
        model->getTokenizer()->Encode(prompts[i], true).size() - 1 >=
        nc.at("init_seq_len").get<unsigned>();
      require(after.hits == before.hits + (reusable ? 1u : 0u) &&
                (reusable ? after.reused_tokens > before.reused_tokens
                          : after.reused_tokens == before.reused_tokens),
              "Unexpected original-call prefix hit/miss");
      require(result["ids"] == prefix_refs[i]["ids"] &&
                result["position"] == prefix_refs[i]["position"],
              "Prefix output or position differs from fresh full history");
      result["reused_tokens"] = after.reused_tokens - before.reused_tokens;
      report["prefix"].push_back(result);
      write();
    }
    const std::string snapshot = std::string(argv[3]) + ".kv";
    c->saveDeviceKVCache(snapshot, true);
    require(!c->getDeviceKVCacheBytes(), "Park did not release active pages");
    c->loadDeviceKVCache(snapshot);
    const auto restored = run(query, false);
    require(restored["ids"] == flash_ref["ids"] &&
              restored["position"] == flash_ref["position"],
            "Paged Flash continuation changed");
    std::filesystem::remove(snapshot);
    report["flash"] = restored;
    const auto stats = c->deviceKVManagerStats();
    report["stats"] = {{"sessions", stats.sessions},
                       {"prefixes", stats.prefixes},
                       {"radix_nodes", stats.radix_nodes},
                       {"hits", stats.hits},
                       {"reused_tokens", stats.reused_tokens}};
    report["passed"] = true;
    write();
    std::cout << "DEVICE_KV_RESULT " << report.dump() << '\n';
  } catch (const std::exception &e) {
    std::cerr << "DEVICE_KV_ERROR " << e.what() << '\n';
    return 1;
  }
}
