// SPDX-License-Identifier: Apache-2.0
/** @file check_htp_page_prefix.cpp
 * @brief Real-model partial page-prefix reuse and source-session preservation.
 * Usage: check_htp_page_prefix MODEL_DIR FIXTURE_JSON OUTPUT_JSON
 */
#include <causal_lm.h>
#include <chrono>
#include <cstdint>
#include <factory.h>
#include <filesystem>
#include <fstream>
#include <htp_model_config.h>
#include <htp_runtime_config.h>
#include <iostream>
#include <model_registry.h>
#include <sstream>
#include <streamer.h>

using json = causallm::json;
using Clock = std::chrono::steady_clock;
static json read(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Cannot read " + path.string());
  json result;
  file >> result;
  return result;
}
static void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
static uint64_t availableMemoryKb() {
#ifdef __ANDROID__
  std::ifstream file("/proc/meminfo");
  std::string line;
  while (std::getline(file, line))
    if (line.rfind("MemAvailable:", 0) == 0) {
      uint64_t value = 0;
      std::istringstream(line.substr(13)) >> value;
      return value;
    }
#endif
  return 0;
}
struct Observer {
  BaseStreamer base;
  Clock::time_point first{};
  static int put(BaseStreamer *b, const char *s) {
    auto *self = reinterpret_cast<Observer *>(b);
    if (s && *s && self->first == Clock::time_point{})
      self->first = Clock::now();
    return 0;
  }
  static void end(BaseStreamer *) {}
};
int main(int argc, char **argv) {
  if (argc != 4)
    return 2;
  try {
    const std::filesystem::path dir = argv[1];
    auto cfg = read(dir / "config.json"),
         gen = read(dir / "generation_config.json"),
         nc = read(dir / "nntr_config.json");
    const auto fixture = read(argv[2]);
    nntrainer::configureHtpRuntime(causallm::normalizeHtpRuntimeConfig(nc));
    causallm::registerAllModels();
    auto model =
      causallm::Factory::Instance().create("Gemma4ForCausalLM", cfg, gen, nc);
    model->initialize();
    model->load_weight(nc["model_file_name"].get<std::string>());
    model->repack_weight();
    auto *c = dynamic_cast<causallm::CausalLM *>(model.get());
    require(c, "CausalLM unavailable");
    const unsigned common = fixture.value("common_tokens", 768u);
    const std::string source = fixture["source"];
    auto source_ids = model->getTokenizer()->Encode(source, true);
    require(source_ids.size() == fixture.value("source_tokens", 1024u),
            "Source token length differs from fixture");
    require(common < source_ids.size(), "Invalid common prefix length");
    std::vector<std::string> branches;
    for (const auto &item : fixture["suffixes"]) {
      const auto suffix =
        model->getTokenizer()->Encode(item.get<std::string>(), false);
      std::vector<int> ids(source_ids.begin(), source_ids.begin() + common);
      ids.insert(ids.end(), suffix.begin(), suffix.end());
      const auto text = model->getTokenizer()->Decode(ids);
      require(model->getTokenizer()->Encode(text, true) == ids,
              "Branch retokenization differs");
      unsigned matched = 0;
      while (matched < ids.size() && matched < source_ids.size() &&
             ids[matched] == source_ids[matched])
        ++matched;
      require(matched == common,
              "Branch does not diverge exactly at requested token");
      branches.push_back(text);
    }
    const std::string query = fixture["follow_up"];
    json report = {{"config", nc},
                   {"fixture", fixture},
                   {"source_ids", source_ids},
                   {"branches", branches},
                   {"baseline", json::array()},
                   {"reused", json::array()}};
    auto write = [&] {
      std::ofstream out(argv[3]);
      out << report.dump(2) << '\n';
    };
    static const BaseStreamerVTable vtable = {Observer::put, Observer::end};
    auto run = [&](const std::string &text, bool reset) {
      if (reset)
        c->setSessionTokenPosition(0);
      Observer observer;
      observer.base.vtable = &vtable;
      c->setStreamer(&observer.base);
      const auto start = Clock::now();
      model->run(text, false, "", "", false);
      const auto end = Clock::now();
      c->setStreamer(nullptr);
      require(observer.first != Clock::time_point{}, "No first delta");
      const auto m = c->getPerformanceMetrics();
      require(!m.prompt_truncated_tokens, "Prompt truncated");
      return json{
        {"input_ids", model->getTokenizer()->Encode(text, reset)},
        {"ids", c->getLastGeneratedTokenIds()},
        {"output", c->getOutput()},
        {"position", c->getKvLen()},
        {"prefill_tokens", m.prefill_tokens},
        {"prefill_ms", m.prefill_duration_ms},
        {"request_to_first_delta_ms",
         std::chrono::duration<double, std::milli>(observer.first - start)
           .count()},
        {"metrics_ttft_ms", m.ttft_ms},
        {"decode_ms_per_token", m.decode_steady_ms},
        {"steady_steps", m.decode_steady_tokens},
        {"kv_bytes", c->getDeviceKVCacheBytes()},
        {"prefix_host_bytes", c->getDeviceKVPrefixBytes()},
        {"peak_memory_kb", m.peak_memory_kb},
        {"wall_ms",
         std::chrono::duration<double, std::milli>(end - start).count()}};
    };
    run("<bos><|turn>user\nSay READY.<turn|>\n<|turn>model\n", true);
    report["source_reference"] = run(source, true);
    report["source_continuation_reference"] = run(query, false);
    for (const auto &text : branches) {
      report["baseline"].push_back(run(text, true));
      write();
    }
    if (!nc.at("htp").value("kv_prefix_cache", false)) {
      report["baseline_only"] = true;
      report["passed"] = true;
      write();
      return 0;
    }
    c->setSessionTokenPosition(0);
    c->enableDeviceKVManager(8, 8);
    const bool flash = fixture.contains("flash");
    if (flash) {
      const auto &f = fixture.at("flash");
      c->enableDeviceKVPrefixFlash(f.at("directory"), f.at("budget_bytes"),
                                   f.at("staging_bytes"));
    }
    const auto source_result = run(source, false);
    report["source_cache_miss"] = source_result;
    require(source_result["ids"] == report["source_reference"]["ids"],
            "Source reference changed");
    if (flash) {
      c->drainDeviceKVPrefixFlash();
      const auto stats = c->deviceKVPrefixFlashStats();
      require(stats.written > 0 && stats.pending_bytes == 0 && !stats.failed,
              "Source boundary was not saved to Flash");
    }
    const auto source_fork = c->forkDeviceKVSession();
    const auto repeat_session = c->createDeviceKVSession();
    c->selectDeviceKVSession(repeat_session);
    const auto repeat_before = c->deviceKVManagerStats();
    report["source_repeat"] = run(source, false);
    report["source_repeat"]["reused_tokens"] =
      c->deviceKVManagerStats().reused_tokens - repeat_before.reused_tokens;
    require(report["source_repeat"]["ids"] ==
                report["source_reference"]["ids"] &&
              report["source_repeat"]["position"] ==
                report["source_reference"]["position"],
            "Repeated source differs from original whole-call inference");
    const unsigned source_end = static_cast<unsigned>(source_ids.size() - 1);
    const unsigned plane = nc.at("init_seq_len");
    const unsigned repeated_hit =
      source_end % plane == 1 ? source_end - 1 : source_end;
    require(report["source_repeat"]["reused_tokens"].get<unsigned>() ==
              repeated_hit,
            "Original prefill endpoint was not reused");
    bool fresh_ids_equal = true;
    bool quality_checks_passed = true;
    for (size_t i = 0; i < branches.size(); ++i) {
      const auto session = c->createDeviceKVSession();
      c->selectDeviceKVSession(session);
      const auto before = c->deviceKVManagerStats();
      const auto input_tokens =
        model->getTokenizer()->Encode(branches[i], true).size();
      // Fixtures use the unchanged 4096-row original prefill plane.
      unsigned expected_hit = 0;
      const unsigned plane = nc.at("init_seq_len");
      const unsigned target_end = static_cast<unsigned>(input_tokens - 1);
      const unsigned source_end = static_cast<unsigned>(source_ids.size() - 1);
      const unsigned last_full = source_end / plane * plane;
      std::vector<unsigned> boundaries{last_full, source_end};
      if (flash)
        for (unsigned to = plane; to < last_full; to += plane)
          boundaries.push_back(to);
      for (unsigned to : boundaries) {
        if (to < 64 || to > common || to > target_end)
          continue;
        const unsigned call_from = (to - 1) / plane * plane;
        if (to == std::min(target_end, call_from + plane))
          expected_hit = std::max(expected_hit, to);
      }
      const auto flash_before = c->deviceKVPrefixFlashStats();
      auto result = run(branches[i], false);
      const auto flash_after = c->deviceKVPrefixFlashStats();
      result["flash_hits"] = flash_after.flash_hits - flash_before.flash_hits;
      result["flash_read_ms"] = flash_after.read_ms - flash_before.read_ms;
      if (flash && i == 0 && expected_hit && expected_hit < last_full)
        require(result["flash_hits"].get<unsigned>() == 1,
                "Earlier sliding boundary did not come from Flash");
      const auto after = c->deviceKVManagerStats();
      result["reused_tokens"] = after.reused_tokens - before.reused_tokens;
      require(after.hits == before.hits + (expected_hit ? 1u : 0u) &&
                result["reused_tokens"] == expected_hit,
              "Partial page prefix was not reused");
      const auto suffix_tokens =
        model->getTokenizer()->Encode(branches[i], true).size() - expected_hit;
      const unsigned last_prompt_decode =
        nc.value("skip_prefill", false) ? 1u : 0u;
      result["processed_suffix_tokens"] =
        result["prefill_tokens"].get<unsigned>() + last_prompt_decode;
      require(result["processed_suffix_tokens"].get<size_t>() == suffix_tokens,
              "Wrong suffix length");
      result["matches_fresh_ids"] =
        result["ids"] == report["baseline"][i]["ids"];
      result["matches_fresh_position"] =
        result["position"] == report["baseline"][i]["position"];
      require(result["matches_fresh_position"].get<bool>(),
              "Prefix final position differs from fresh");
      report["reused"].push_back(result);
      write();
      fresh_ids_equal =
        fresh_ids_equal && result["matches_fresh_ids"].get<bool>();
      if (fixture.contains("expected_codes")) {
        result["contains_expected_code"] =
          result["output"].get<std::string>().find(
            fixture["expected_codes"][i].get<std::string>()) !=
          std::string::npos;
        result["fresh_contains_expected_code"] =
          report["baseline"][i]["output"].get<std::string>().find(
            fixture["expected_codes"][i].get<std::string>()) !=
          std::string::npos;
        quality_checks_passed =
          quality_checks_passed &&
          result["fresh_contains_expected_code"].get<bool>() &&
          result["contains_expected_code"].get<bool>();
        require(!result["fresh_contains_expected_code"].get<bool>() ||
                  result["contains_expected_code"].get<bool>(),
                "Branch reused a stale answer");
      }
      report["reused"].back() = result;
      write();
    }
    c->selectDeviceKVSession(source_fork);
    report["source_after_branches"] = run(query, false);
    require(report["source_after_branches"]["ids"] ==
              report["source_continuation_reference"]["ids"],
            "Original 1024-token source was overwritten");
    report["prefix_host_before_clear"] = c->getDeviceKVPrefixBytes();
    report["available_kb_before_prefix_clear"] = availableMemoryKb();
    c->clearDeviceKVPrefixes();
    report["available_kb_after_prefix_clear"] = availableMemoryKb();
    if (flash) {
      c->drainDeviceKVPrefixFlash();
      const auto stats = c->deviceKVPrefixFlashStats();
      report["flash_after_clear_bytes"] = stats.flash_bytes;
      require(stats.flash_bytes == 0 && stats.pending_bytes == 0,
              "Prefix clear retained Flash files");
    }
    report["prefix_host_after_clear"] = c->getDeviceKVPrefixBytes();
    require(c->getDeviceKVPrefixBytes() == 0 && c->getDeviceKVCacheBytes() > 0,
            "Clearing prefixes must free host pins and preserve active DSP KV");
    report["prefix_checks_passed"] = true;
    report["fresh_output_ids_match"] = fresh_ids_equal;
    report["quality_checks_passed"] = quality_checks_passed;
    report["passed"] = fresh_ids_equal;
    write();
    std::cout << "PAGE_PREFIX_RESULT " << report.dump() << '\n';
    if (!fresh_ids_equal) {
      std::cerr << "PAGE_PREFIX_FRESH_OUTPUT_MISMATCH: structural prefix and "
                   "source checks passed; fresh greedy IDs differ\n";
      return 3;
    }
  } catch (const std::exception &e) {
    std::cerr << "PAGE_PREFIX_ERROR " << e.what() << '\n';
    return 1;
  }
}
