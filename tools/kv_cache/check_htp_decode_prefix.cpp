// SPDX-License-Identifier: Apache-2.0
/** Android end-to-end check for a decode checkpoint at the 4096 boundary. */
#include <causal_lm.h>
#include <cstdlib>
#include <factory.h>
#include <filesystem>
#include <fstream>
#include <htp_model_config.h>
#include <htp_runtime_config.h>
#include <iostream>
#include <model_registry.h>

using json = causallm::json;
static json readJson(const std::filesystem::path &path) {
  std::ifstream f(path);
  json value;
  f >> value;
  return value;
}
static void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
int main(int argc, char **argv) {
  if (argc != 6 && argc != 7)
    return 2;
  try {
    const std::filesystem::path dir(argv[1]);
    auto cfg = readJson(dir / "config.json");
    auto gen = readJson(dir / "generation_config.json");
    auto nc = readJson(dir / "nntr_config.json");
    nntrainer::configureHtpRuntime(causallm::normalizeHtpRuntimeConfig(nc));
    causallm::registerAllModels();
    auto model =
      causallm::Factory::Instance().create("Gemma4ForCausalLM", cfg, gen, nc);
    std::cerr << "DECODE_PREFIX_PHASE initialize" << std::endl;
    model->initialize();
    std::cerr << "DECODE_PREFIX_PHASE load " << nc["model_file_name"]
              << std::endl;
    model->load_weight(nc["model_file_name"].get<std::string>());
    model->repack_weight();
    auto *c = dynamic_cast<causallm::CausalLM *>(model.get());
    require(c, "CausalLM unavailable");
    std::ifstream fixture(argv[2]);
    const std::string text((std::istreambuf_iterator<char>(fixture)), {});
    auto tokens = model->getTokenizer()->Encode(text, true);
    const unsigned boundary = std::stoul(argv[5]);
    const unsigned seed_length = argc == 7 ? std::stoul(argv[6]) : boundary - 1;
    require(boundary && boundary % 4096u == 0 && seed_length <= boundary &&
              tokens.size() >= seed_length,
            "Fixture is too short for the requested boundary");
    tokens.resize(seed_length);
    const auto seed = model->getTokenizer()->Decode(tokens);
    require(model->getTokenizer()->Encode(seed, true) == tokens,
            "Seed token round-trip failed");
    model->run(seed, false, "", "", false);
    const auto baseline_ids = c->getLastGeneratedTokenIds();
    const auto baseline_metrics = c->getPerformanceMetrics();
    c->setSessionTokenPosition(0);
    c->enableDeviceKVManager(8, 8);
    c->enableDeviceKVPrefixFlash(argv[3], size_t(2) << 30, size_t(1) << 30);
    c->enableDeviceKVDecodePrefixFlash();
    model->run(seed, false, "", "", false);
    const auto seed_ids = c->getLastGeneratedTokenIds();
    const auto seed_metrics = c->getPerformanceMetrics();
    require(seed_ids == baseline_ids, "Checkpoint capture changed seed IDs");
    require(seed_ids.size() >= 3, "Generation did not cross 4096");
    auto full = tokens;
    full.insert(full.end(), seed_ids.begin(), seed_ids.end());
    std::string suffix = " Continue with one more sentence.";
    if (const char *path = std::getenv("NNTR_DECODE_PREFIX_SUFFIX_FILE")) {
      std::ifstream f(path);
      require(bool(f), "Cannot read suffix fixture");
      suffix.assign(std::istreambuf_iterator<char>(f), {});
    }
    const auto delta = model->getTokenizer()->Encode(suffix, false);
    full.insert(full.end(), delta.begin(), delta.end());
    const auto complete = model->getTokenizer()->Decode(full);
    require(model->getTokenizer()->Encode(complete, true) == full,
            "Full history token round-trip failed");
    c->drainDeviceKVPrefixFlash();
    require(c->deviceKVPrefixFlashStats().written > 0,
            "Decode checkpoint was not written");
    model->run(suffix, false, "", "", false);
    const auto reference_ids = c->getLastGeneratedTokenIds();
    const auto reference_position = c->getKvLen();
    const auto reference_ttft = c->getPerformanceMetrics().ttft_ms;
    const auto before = c->deviceKVManagerStats();
    const auto other_session = c->createDeviceKVSession();
    c->selectDeviceKVSession(other_session);
    model->run(complete, false, "", "", false);
    const auto hit_ids = c->getLastGeneratedTokenIds();
    const auto hit_position = c->getKvLen();
    const auto hit_ttft = c->getPerformanceMetrics().ttft_ms;
    const auto restore = c->decodePrefixRestoreStats();
    const auto after = c->deviceKVManagerStats();
    const auto flash = c->deviceKVPrefixFlashStats();
    c->selectDeviceKVSession(0);
    require(c->getKvLen() == reference_position,
            "Original session position changed after checkpoint hit");
    c->selectDeviceKVSession(other_session);
    require(c->getKvLen() == hit_position,
            "Restored session position changed after switching");
    c->setSessionTokenPosition(0);
    model->run(complete, false, "", "", false);
    const auto repeated_ids = c->getLastGeneratedTokenIds();
    const auto repeated_position = c->getKvLen();
    const auto repeated_ttft = c->getPerformanceMetrics().ttft_ms;
    const auto repeated_restore = c->decodePrefixRestoreStats();
    bool corrupted = false;
    for (const auto &file :
         std::filesystem::recursive_directory_iterator(argv[3])) {
      if (file.path().extension() != ".kv")
        continue;
      std::fstream stream(file.path(),
                          std::ios::in | std::ios::out | std::ios::binary);
      stream.seekg(48);
      char byte = 0;
      stream.get(byte);
      stream.seekp(48);
      stream.put(byte ^ 0x55);
      require(bool(stream), "Cannot corrupt checkpoint fixture");
      corrupted = true;
      break;
    }
    require(corrupted, "No checkpoint file to corrupt");
    c->setSessionTokenPosition(0);
    model->run(complete, false, "", "", false);
    const auto corrupt_ids = c->getLastGeneratedTokenIds();
    const auto corruption_failures = c->deviceKVPrefixFlashStats().failed;
    c->clearDeviceKVPrefixes();
    c->setSessionTokenPosition(0);
    model->run(complete, false, "", "", false);
    const auto recompute_ids = c->getLastGeneratedTokenIds();
    const auto recompute_ttft = c->getPerformanceMetrics().ttft_ms;
    auto textOf = [&](const std::vector<unsigned> &ids) {
      return model->getTokenizer()->Decode(
        std::vector<int32_t>(ids.begin(), ids.end()));
    };
    json result = {
      {"input_tokens", full.size()},
      {"seed_tokens", tokens.size()},
      {"seed_ids", seed_ids},
      {"reference_ids", reference_ids},
      {"hit_ids", hit_ids},
      {"hit_text", textOf(hit_ids)},
      {"reference_text", textOf(reference_ids)},
      {"recompute_text", textOf(recompute_ids)},
      {"repeated_ids", repeated_ids},
      {"repeated_hit_ttft_ms", repeated_ttft},
      {"repeated_restore_read_ms", repeated_restore.read_ms},
      {"repeated_restore_conversion_ms", repeated_restore.conversion_ms},
      {"repeated_suffix_prefill_calls", repeated_restore.prefill_calls},
      {"repeated_prefix_tokens", repeated_restore.prefix_tokens},
      {"restore_read_ms", restore.read_ms},
      {"restore_conversion_ms", restore.conversion_ms},
      {"restore_bind_ms", restore.bind_ms},
      {"suffix_prefill_ms", restore.prefill_ms},
      {"suffix_prefill_tokens", restore.suffix_tokens},
      {"suffix_prefill_calls", restore.prefill_calls},
      {"recompute_ids", recompute_ids},
      {"corrupt_fallback_ids", corrupt_ids},
      {"corruption_failures", corruption_failures},
      {"reference_position", reference_position},
      {"hit_position", hit_position},
      {"reused_tokens", after.reused_tokens - before.reused_tokens},
      {"baseline_seed_decode_ms", baseline_metrics.decode_steady_ms},
      {"capture_seed_decode_ms", seed_metrics.decode_steady_ms},
      {"baseline_seed_generation_ms", baseline_metrics.generation_duration_ms},
      {"capture_seed_generation_ms", seed_metrics.generation_duration_ms},
      {"seed_decode_tokens", seed_metrics.decode_steady_tokens},
      {"reference_ttft_ms", reference_ttft},
      {"hit_ttft_ms", hit_ttft},
      {"recompute_ttft_ms", recompute_ttft},
      {"capture_ms", flash.staging_ms},
      {"write_ms", flash.write_ms},
      {"flash_hits", flash.flash_hits}};
    std::ofstream(argv[4]) << result.dump(2) << '\n';
    std::cout << result.dump() << std::endl;
    require(after.reused_tokens - before.reused_tokens == boundary,
            "Decode checkpoint was not selected");
    require(hit_ids == repeated_ids && hit_position == repeated_position &&
              hit_position == full.size() + hit_ids.size() - 1u,
            "Batched decode prefix restoration is not deterministic");
    require(restore.prefix_tokens == boundary && restore.prefill_calls == 1 &&
              restore.suffix_tokens > 1,
            "Restored suffix did not use one batched prefill call");
    require(repeated_restore.prefix_tokens == boundary &&
              repeated_restore.prefill_calls == 1,
            "Repeated run did not restore the same decode boundary");
    if (const char *expected = std::getenv("NNTR_DECODE_PREFIX_EXPECT"))
      require(
        textOf(hit_ids).find(expected) != std::string::npos &&
          textOf(recompute_ids).find(expected) != std::string::npos,
        "Restored or recomputed answer did not contain the expected text");
    require(corruption_failures && corrupt_ids == recompute_ids,
            "Corrupt checkpoint did not fall back to recomputation");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "DECODE_PREFIX_CHECK: " << e.what() << std::endl;
    return 1;
  }
}
