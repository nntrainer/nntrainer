// SPDX-License-Identifier: Apache-2.0
/** @file benchmark_htp_prefix_cold.cpp
 * @brief Same-model cold ON/OFF storage benchmark; numerical policy stays
 * fixed. Usage: htp_prefix_cold_bench MODEL_DIR FIXTURE_JSON OUTPUT_JSON
 * Fixture: {warmup: string, prompts: [string, ...]}. Eight balanced rounds.
 */
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gemma4_htp_moe_causallm.h>
#include <htp_model_config.h>
#include <htp_runtime_config.h>
#include <iostream>
#include <memory>
#include <model_registry.h>
#include <stdexcept>
#include <streamer.h>
using json = causallm::json;
using Clock = std::chrono::steady_clock;
namespace causallm {
// Test-only control of storage retention. No attention runtime flag changes.
class PrefixColdBenchmark final : public Gemma4HtpMoECausalLM {
public:
  PrefixColdBenchmark(json &c, json &g, json &n) :
    Transformer(sanitizeConfig(c), sanitizeGenerationConfig(g, c), n,
                ModelType::CAUSALLM),
    Gemma4HtpMoECausalLM(c, g, n) {}
  void retainPrefixes(bool enabled, bool flash) {
    device_prefill_capture_ = flash || enabled;
    device_prefix_flash_capture_ = enabled;
  }
};
} // namespace causallm
static json read(const std::filesystem::path &p) {
  std::ifstream f(p);
  json j;
  f >> j;
  return j;
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
         nc = read(dir / "nntr_config.json"), fixture = read(argv[2]);
    const auto runtime = causallm::normalizeHtpRuntimeConfig(nc);
    if (runtime.htp.kv_seed_q)
      throw std::invalid_argument(
        "Cold prefix benchmark requires FP16 KV seeding");
    nntrainer::configureHtpRuntime(runtime);
    causallm::registerAllModels();
    auto model = std::make_unique<causallm::PrefixColdBenchmark>(cfg, gen, nc);
    model->initialize();
    model->load_weight(nc.at("model_file_name"));
    model->repack_weight();
    model->enableDeviceKVManager(2, 1);
    const bool flash = fixture.contains("flash");
    if (flash) {
      const auto &f = fixture.at("flash");
      model->enableDeviceKVPrefixFlash(f.at("directory"), f.at("budget_bytes"),
                                       f.at("staging_bytes"));
    }
    static const BaseStreamerVTable vt = {Observer::put, Observer::end};
    json report = {{"config", nc},
                   {"fixture", fixture},
                   {"references", json::array()},
                   {"samples", json::array()}};
    auto write = [&] {
      std::ofstream f(argv[3]);
      f << report.dump(2) << '\n';
    };
    auto run = [&](const std::string &text, bool retain) {
      Observer observer;
      observer.base.vtable = &vt;
      model->setStreamer(&observer.base);
      model->drainDeviceKVPrefixFlash();
      const auto flash_before = model->deviceKVPrefixFlashStats();
      const auto started = Clock::now();
      model->clearDeviceKVPrefixes();
      model->setSessionTokenPosition(0);
      model->retainPrefixes(retain, flash);
      const auto before = model->deviceKVManagerStats();
      model->run(text, false, "", "", false);
      const auto ended = Clock::now();
      model->setStreamer(nullptr);
      const auto m = model->getPerformanceMetrics();
      if (observer.first == Clock::time_point{} || m.prompt_truncated_tokens ||
          model->deviceKVManagerStats().hits != before.hits)
        throw std::runtime_error("Invalid cold request");
      const auto after_run = model->deviceKVPrefixFlashStats();
      const auto drain_started = Clock::now();
      model->drainDeviceKVPrefixFlash();
      const auto drain_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - drain_started)
          .count();
      const auto after_drain = model->deviceKVPrefixFlashStats();
      if (flash && retain && fixture.value("require_backup", true) &&
          after_drain.submitted == flash_before.submitted)
        throw std::runtime_error("No earlier boundary was backed up");
      return json{
        {"flash_submitted", after_drain.submitted - flash_before.submitted},
        {"flash_written", after_drain.written - flash_before.written},
        {"flash_skipped", after_drain.skipped - flash_before.skipped},
        {"flash_failed", after_drain.failed - flash_before.failed},
        {"flash_staging_ms", after_drain.staging_ms - flash_before.staging_ms},
        {"flash_write_ms", after_drain.write_ms - flash_before.write_ms},
        {"flash_pending_at_run_end", after_run.pending_bytes},
        {"post_request_drain_ms", drain_ms},
        {"capture", retain},
        {"ids", model->getLastGeneratedTokenIds()},
        {"position", model->getKvLen()},
        {"output", model->getOutput()},
        {"prefill_tokens", m.prefill_tokens},
        {"prefill_ms", m.prefill_duration_ms},
        {"ttft_ms", m.ttft_ms},
        {"request_to_first_delta_ms",
         std::chrono::duration<double, std::milli>(observer.first - started)
           .count()},
        {"wall_ms",
         std::chrono::duration<double, std::milli>(ended - started).count()},
        {"prefix_host_bytes", model->getDeviceKVPrefixBytes()},
        {"decode_ms_per_token", m.decode_steady_ms}};
    };
    run(fixture.at("warmup"), false);
    for (const auto &p : fixture.at("prompts"))
      report["references"].push_back(run(p, false));
    const bool on[] = {false, true, true, false, true, false, false, true};
    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned p = 0; p < fixture.at("prompts").size(); ++p) {
        auto v = run(fixture["prompts"][p], on[i]);
        v["round"] = i;
        v["prompt"] = p;
        const auto &ref = report["references"][p];
        if (v["ids"] != ref["ids"] || v["position"] != ref["position"])
          throw std::runtime_error("Native output ID or position changed");
        report["samples"].push_back(v);
        write();
      }
    }
    report["passed"] = true;
    write();
    model.reset();
    causallm::CausalLM::releaseDeviceCaches();
    std::cout << "PAIRED_RESULT PASS\n";
  } catch (const std::exception &e) {
    std::cerr << "PAIRED_ERROR " << e.what() << '\n';
    return 1;
  }
}
