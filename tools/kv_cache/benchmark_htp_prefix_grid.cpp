// SPDX-License-Identifier: Apache-2.0
/** @file benchmark_htp_prefix_grid.cpp
 * @brief Same-model 4K..20K recompute/DRAM/Flash comparison with exact IDs.
 * Usage: htp_prefix_grid_bench MODEL_DIR FLASH_DIRECTORY OUTPUT_JSON
 * Does not change numerical runtime settings. K means 1024 tokens.
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
#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif
using json = causallm::json;
using Clock = std::chrono::steady_clock;
namespace causallm {
/** Keep the producer immutable across samples; do not publish measured
 * branches back into the tree (which would turn later Flash tests into DRAM
 * hits). Suspend only the token lookup for forced-recompute controls.
 */
class PrefixGridBenchmark final : public Gemma4HtpMoECausalLM {
public:
  PrefixGridBenchmark(json &c, json &g, json &n) :
    Transformer(sanitizeConfig(c), sanitizeGenerationConfig(g, c), n,
                ModelType::CAUSALLM),
    Gemma4HtpMoECausalLM(c, g, n) {}
  void capture(bool enabled, bool archive) {
    device_prefill_capture_ = enabled;
    device_prefix_flash_capture_ = archive;
    suppress_publication_ = !enabled;
  }
  std::unique_ptr<DeviceKVManager> suspendLookup() {
    return std::move(device_kv_manager_);
  }
  void resumeLookup(std::unique_ptr<DeviceKVManager> m) {
    device_kv_manager_ = std::move(m);
  }

protected:
  void beforePrefillBlock(unsigned from, unsigned to) override {
    Gemma4HtpMoECausalLM::beforePrefillBlock(from, to);
    // restorePrefix and calibration restoration have finished before this
    // hook. Only request-local metadata is dropped; producer handles survive.
    if (suppress_publication_)
      device_prefill_history_.reset();
  }

private:
  bool suppress_publication_ = true;
};
} // namespace causallm
static json read(const std::filesystem::path &p) {
  std::ifstream f(p);
  json j;
  f >> j;
  return j;
}
static void require(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
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
static unsigned adviseDropFiles(const std::filesystem::path &root) {
  unsigned count = 0;
#if defined(__linux__)
  for (const auto &item : std::filesystem::recursive_directory_iterator(root))
    if (item.is_regular_file() && item.path().extension() == ".kv") {
      const int fd = ::open(item.path().c_str(), O_RDONLY | O_CLOEXEC);
      require(fd >= 0, "Cannot open backing for cache-eviction advice");
      const int rc = ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
      ::close(fd);
      require(rc == 0, "Cannot advise dropping clean file pages");
      ++count;
    }
#else
  (void)root;
#endif
  return count;
}
int main(int argc, char **argv) {
  if (argc != 4)
    return 2;
  json report;
  auto write = [&] {
    std::ofstream out(argv[3]);
    out << report.dump(2) << '\n';
  };
  try {
    const std::filesystem::path dir = argv[1], flash_dir = argv[2];
    auto cfg = read(dir / "config.json"),
         gen = read(dir / "generation_config.json"),
         nc = read(dir / "nntr_config.json");
    const auto runtime = causallm::normalizeHtpRuntimeConfig(nc);
    require(!runtime.htp.kv_seed_q,
            "Grid requires the original FP16 seed route");
    nntrainer::configureHtpRuntime(runtime);
    causallm::registerAllModels();
    auto model = std::make_unique<causallm::PrefixGridBenchmark>(cfg, gen, nc);
    model->initialize();
    model->load_weight(nc.at("model_file_name"));
    model->repack_weight();
    model->enableDeviceKVManager(1, 1);
    model->enableDeviceKVPrefixFlash(flash_dir.string(), size_t(3) << 30,
                                     size_t(768) << 20);
    const std::vector<unsigned> lengths{4096, 8192, 12288, 16384, 20480};
    std::string filler = "<bos><|turn>user\n";
    for (unsigned i = 0; i < 2400; ++i)
      filler += "The mountain village has orchards, a bridge and a library. ";
    auto *tokenizer = model->getTokenizer();
    const auto body = tokenizer->Encode(filler, true);
    require(body.size() > 26000, "Filler too short");
    const auto branch_tail =
      tokenizer->Encode("\n\nRemember the code RUBY-456. What is the code? "
                        "Reply only with the code.<turn|>\n<|turn>model\n",
                        false);
    const auto source_tail =
      tokenizer->Encode("\n\nRemember the code ORCHID-842. Reply only with "
                        "READY.<turn|>\n<|turn>model\n",
                        false);
    auto textOf = [&](const std::vector<int> &ids) {
      const auto text = tokenizer->Decode(ids);
      require(tokenizer->Encode(text, true) == ids, "Token roundtrip differs");
      return text;
    };
    auto sourceOf = [&](unsigned total) {
      require(total > source_tail.size(), "Invalid source length");
      std::vector<int> ids(body.begin(),
                           body.begin() + total - source_tail.size());
      ids.insert(ids.end(), source_tail.begin(), source_tail.end());
      return textOf(ids);
    };
    std::vector<std::string> branches;
    for (unsigned length : lengths) {
      std::vector<int> ids(body.begin(), body.begin() + length);
      ids.insert(ids.end(), branch_tail.begin(), branch_tail.end());
      branches.push_back(textOf(ids));
    }
    const auto flash_source = sourceOf(25088);
    const auto flash_source_ids = tokenizer->Encode(flash_source, true);
    for (size_t i = 0; i < lengths.size(); ++i) {
      const auto ids = tokenizer->Encode(branches[i], true);
      unsigned common = 0;
      while (common < ids.size() && ids[common] == flash_source_ids[common])
        ++common;
      require(common == lengths[i], "Branch boundary differs");
    }
    report = {{"config", nc},
              {"flash_directory", flash_dir.string()},
              {"lengths", lengths},
              {"branches", branches},
              {"flash_source", flash_source},
              {"flash_source_tokens", flash_source_ids},
              {"samples", json::array()},
              {"dram_producers", json::array()},
              {"passed", false}};
    write();
    static const BaseStreamerVTable vt = {Observer::put, Observer::end};
    auto run = [&](const std::string &text, bool lookup, bool capture,
                   bool archive, unsigned expect_reused, bool expect_flash) {
      model->setSessionTokenPosition(0);
      model->capture(capture, archive);
      std::unique_ptr<causallm::DeviceKVManager> held;
      if (!lookup)
        held = model->suspendLookup();
      Observer observer;
      observer.base.vtable = &vt;
      model->setStreamer(&observer.base);
      const auto before = model->deviceKVManagerStats();
      const auto fb = model->deviceKVPrefixFlashStats();
      const auto start = Clock::now();
      try {
        model->run(text, false, "", "", false);
      } catch (...) {
        if (held)
          model->resumeLookup(std::move(held));
        throw;
      }
      const auto end = Clock::now();
      const auto after = model->deviceKVManagerStats();
      if (held)
        model->resumeLookup(std::move(held));
      model->setStreamer(nullptr);
      const auto fa = model->deviceKVPrefixFlashStats();
      const auto m = model->getPerformanceMetrics();
      require(observer.first != Clock::time_point{} &&
                !m.prompt_truncated_tokens,
              "No first delta or input was truncated");
      require(after.reused_tokens - before.reused_tokens == expect_reused,
              "Wrong reused token count");
      require(fa.flash_hits - fb.flash_hits == unsigned(expect_flash),
              "Expected tier was not used");
      require(fa.staging_hits == fb.staging_hits,
              "Pending staging used instead of Flash");
      return json{
        {"input_ids", tokenizer->Encode(text, true)},
        {"ids", model->getLastGeneratedTokenIds()},
        {"output", model->getOutput()},
        {"position", model->getKvLen()},
        {"ttft_ms", m.ttft_ms},
        {"external_ttft_ms",
         std::chrono::duration<double, std::milli>(observer.first - start)
           .count()},
        {"wall_ms",
         std::chrono::duration<double, std::milli>(end - start).count()},
        {"prefill_ms", m.prefill_duration_ms},
        {"prefill_tokens", m.prefill_tokens},
        {"reused_tokens", expect_reused},
        {"flash_hits", fa.flash_hits - fb.flash_hits},
        {"flash_read_ms", fa.read_ms - fb.read_ms},
        {"backup_submitted", fa.submitted - fb.submitted},
        {"backup_skipped", fa.skipped - fb.skipped},
        {"backup_failed", fa.failed - fb.failed},
        {"backup_staging_ms", fa.staging_ms - fb.staging_ms},
        {"backup_write_ms_at_run_end", fa.write_ms - fb.write_ms},
        {"backup_pending_at_run_end", fa.pending_bytes},
        {"reserved_flash_bytes", fa.flash_bytes},
        {"kv_payload_bytes", model->getDeviceKVCacheBytes()},
        {"prefix_host_bytes", model->getDeviceKVPrefixBytes()},
        {"peak_memory_kb", m.peak_memory_kb},
        {"decode_ms_per_token", m.decode_steady_ms},
        {"decode_steps", m.decode_steady_tokens}};
    };
    report["warmup"] =
      run("<bos><|turn>user\nSay READY.<turn|>\n<|turn>model\n", false, false,
          false, 0, false);
    report["flash_producer"] = run(flash_source, true, true, true, 0, false);
    model->drainDeviceKVPrefixFlash();
    const auto fs = model->deviceKVPrefixFlashStats();
    require(fs.written == 5 && fs.skipped == 0 && fs.failed == 0,
            "Not all five earlier boundaries were archived");
    report["flash_backing"] = {{"written", fs.written},
                               {"file_bytes", fs.flash_bytes},
                               {"write_ms", fs.write_ms},
                               {"staging_ms", fs.staging_ms},
                               {"peak_staging_bytes", fs.peak_pending_bytes}};
    write();
    // Alternate recompute/hit order; controls retain the same producer memory.
    for (size_t i = 0; i < lengths.size(); ++i) {
      for (unsigned round = 0; round < 2; ++round) {
        json reference, hit;
        auto cold = [&] {
          reference = run(branches[i], false, false, false, 0, false);
        };
        auto reuse = [&] {
          const auto advised = adviseDropFiles(flash_dir);
          hit = run(branches[i], true, false, false, lengths[i], true);
          hit["file_cache_advice_count"] = advised;
        };
        if (round % 2) {
          reuse();
          cold();
        } else {
          cold();
          reuse();
        }
        require(reference["ids"] == hit["ids"] &&
                  reference["position"] == hit["position"],
                "Flash output IDs/position differ from recomputation");
        report["samples"].push_back({{"tier", "flash"},
                                     {"common_tokens", lengths[i]},
                                     {"round", round},
                                     {"reference", reference},
                                     {"hit", hit}});
        write();
        std::cout << "GRID flash " << lengths[i] << " round " << round
                  << " PASS\n"
                  << std::flush;
      }
    }
    model->clearDeviceKVPrefixes();
    model->drainDeviceKVPrefixFlash();
    require(model->deviceKVPrefixFlashStats().flash_bytes == 0,
            "Flash clear leaked backing");
    for (size_t i = 0; i < lengths.size(); ++i) {
      const auto source = sourceOf(lengths[i] + 512);
      report["dram_producers"].push_back(
        run(source, true, true, false, 0, false));
      write();
      for (unsigned round = 0; round < 2; ++round) {
        json reference, hit;
        auto cold = [&] {
          reference = run(branches[i], false, false, false, 0, false);
        };
        auto reuse = [&] {
          hit = run(branches[i], true, false, false, lengths[i], false);
        };
        if (round % 2) {
          reuse();
          cold();
        } else {
          cold();
          reuse();
        }
        require(reference["ids"] == hit["ids"] &&
                  reference["position"] == hit["position"],
                "DRAM output IDs/position differ from recomputation");
        report["samples"].push_back({{"tier", "dram"},
                                     {"common_tokens", lengths[i]},
                                     {"round", round},
                                     {"reference", reference},
                                     {"hit", hit}});
        write();
        std::cout << "GRID dram " << lengths[i] << " round " << round
                  << " PASS\n"
                  << std::flush;
      }
      model->clearDeviceKVPrefixes();
    }
    report["cold_backup_20K"] = json::array();
    const auto long_source = sourceOf(20992);
    const bool backup_modes[] = {false, true, true, false};
    for (bool backup : backup_modes) {
      model->drainDeviceKVPrefixFlash();
      model->clearDeviceKVPrefixes();
      auto result = run(long_source, true, true, backup, 0, false);
      model->drainDeviceKVPrefixFlash();
      result["backup_enabled"] = backup;
      const auto &reference = report["dram_producers"].back();
      require(result["ids"] == reference["ids"] &&
                result["position"] == reference["position"],
              "20K cold backup output IDs/position changed");
      require(result["backup_submitted"].get<unsigned>() ==
                  (backup ? 4u : 0u) &&
                result["backup_skipped"].get<unsigned>() == 0 &&
                result["backup_failed"].get<unsigned>() == 0,
              "20K cold backup was incomplete");
      report["cold_backup_20K"].push_back(result);
      write();
      std::cout << "GRID cold backup=" << backup << " PASS\n" << std::flush;
    }
    model->clearDeviceKVPrefixes();
    model->drainDeviceKVPrefixFlash();
    report["passed"] = true;
    write();
    std::cout << "GRID_RESULT PASS\n";
  } catch (const std::exception &e) {
    report["error"] = e.what();
    write();
    std::cerr << "GRID_ERROR " << e.what() << '\n';
    return 1;
  }
}
