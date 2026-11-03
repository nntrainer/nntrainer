// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_causallm_gemma4_htp.cpp
 * @brief Validate the fused Gemma4 HTP model and its input contract.
 * @author NNTrainer contributors
 * @bug See the project issue tracker.
 */
#include <causallm_test_utils.h>
#include <factory.h>
#include <fstream>
#include <gemma4_htp_moe_causallm.h>
#include <gemma4_moe_causallm.h>
#include <gtest/gtest.h>
#include <llm_util.hpp>
#include <model_registry.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
namespace {
class ReferenceGemma4Htp final
  : public causallm_test::CausalLMTestAdapter<causallm::Gemma4HtpMoECausalLM> {
public:
  ReferenceGemma4Htp(causallm::json &cfg, causallm::json &gen,
                     causallm::json &nntr) :
    causallm::Transformer(
      causallm::Gemma4HtpTransformer::sanitizeConfig(cfg),
      causallm::Gemma4HtpTransformer::sanitizeGenerationConfig(gen, cfg), nntr,
      causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Gemma4HtpMoECausalLM>(cfg, gen,
                                                                       nntr) {}
};
TEST(Gemma4HtpDifferential, FusedFP32MatchesPinnedReference) {
  causallm_test::runFp32DifferentialChecks(
    {"gemma4_htp_moe_tiny",
     [](causallm::json &cfg, causallm::json &gen, causallm::json &nntr) {
       return std::make_unique<ReferenceGemma4Htp>(cfg, gen, nntr);
     }});
}
struct FixtureConfig {
  std::filesystem::path dir =
    causallm_test::findFixtureDir("gemma4_htp_moe_tiny");
  causallm::json model = causallm::LoadJsonFile((dir / "config.json").string());
  causallm::json generation =
    causallm::LoadJsonFile((dir / "generation_config.json").string());
  causallm::json nntr =
    causallm::LoadJsonFile((dir / "nntr_config.json").string());
  FixtureConfig() {
    nntr["tokenizer_file"] = (dir / "tokenizer.json").string();
  }
};

TEST(Gemma4HtpContract, FactoryRequiresExplicitProfileAndFormat) {
  causallm::registerAllModels();
  auto create = [](FixtureConfig &f) {
    return causallm::Factory::Instance().create("Gemma4ForCausalLM", f.model,
                                                f.generation, f.nntr);
  };
  FixtureConfig f;
  auto htp = create(f);
  ASSERT_NE(dynamic_cast<causallm::Gemma4HtpMoECausalLM *>(htp.get()), nullptr);
  f.nntr.erase("gemma4_weight_format");
  EXPECT_THROW(create(f), std::invalid_argument);
  f.nntr.erase("gemma4_execution_profile");
  auto main = create(f);
  EXPECT_NE(dynamic_cast<causallm::Gemma4MoECausalLM *>(main.get()), nullptr);
  f.nntr["moe_engine"] = "htp";
  EXPECT_THROW(create(f), std::invalid_argument);
}

TEST(Gemma4HtpContract, RejectsUnsupportedCacheEngineAndDtypes) {
  for (const auto &setting :
       std::vector<causallm::json>{{{"kv_window_ring", true}},
                                   {{"batch_size", 2}},
                                   {{"moe_engine", "gpu"}},
                                   {{"moe_layer_dtype", "QS2CX_WH"}},
                                   {{"model_tensor_type", "FP32-FP16"}},
                                   {{"embedding_dtype", "UINT8"}}}) {
    FixtureConfig f;
    f.nntr.update(setting);
    EXPECT_THROW(causallm::Gemma4HtpMoECausalLM(f.model, f.generation, f.nntr),
                 std::invalid_argument)
      << setting.dump();
  }
}

TEST(Gemma4HtpContract, RejectsTruncatedWeightBeforeInference) {
  FixtureConfig f;
  ReferenceGemma4Htp model(f.model, f.generation, f.nntr);
  model.initializeModel();
  auto path =
    std::filesystem::temp_directory_path() / "nntr-gemma4-htp-truncated.bin";
  struct RemoveFile {
    std::filesystem::path path;
    ~RemoveFile() {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  } cleanup{path};
  {
    std::ofstream file(path, std::ios::binary);
    file.put('x');
  }
  EXPECT_THROW(model.loadWeight(path.string()), std::runtime_error);
}

/** @brief The "[PPL] decode tokens=" summary line of one run() with
 *         NNTR_PPL_DECODE=@a ids, or "" when the run printed none. */
std::string pplDecodeRun(FixtureConfig &f, const std::string &ids) {
  ReferenceGemma4Htp model(f.model, f.generation, f.nntr);
  model.initializeModel();
  model.loadWeight(
    (f.dir / f.nntr["model_file_name"].get<std::string>()).string());
  setenv("NNTR_PPL_DECODE", ids.c_str(), 1);
  testing::internal::CaptureStderr();
  model.runPrompt("hello tok4");
  const std::string err = testing::internal::GetCapturedStderr();
  unsetenv("NNTR_PPL_DECODE");
  const size_t at = err.find("[PPL] decode tokens=");
  return at == std::string::npos ? "" : err.substr(at, err.find('\n', at) - at);
}

/** @brief Field @a key ("tokens=", "top1=", ...) of a summary line. */
std::string pplField(const std::string &line, const std::string &key) {
  const size_t at = line.find(" " + key);
  if (at == std::string::npos)
    return "";
  const size_t from = at + 1 + key.size();
  return line.substr(from, line.find(' ', from) - from);
}

/**
 * NNTR_PPL_DECODE (ported from a21cae045 / fead3371b): a self run scores
 * its own greedy path and writes it; a forced run on that file scores the
 * same logits (nll_sum equal, every top1 the target); a changed id is read
 * (the forced run's top1 count drops). With the prefill's pick and with
 * skip_prefill, where decode step 0 produces the first id.
 */
TEST(Gemma4HtpPplDecode, ForcedRunScoresTheSelfRunsPath) {
  for (const bool skip : {false, true}) {
    FixtureConfig f;
    f.nntr["max_seq_len"] = 16;
    f.nntr["num_to_generate"] = 4;
    f.nntr["skip_prefill"] = skip;
    const std::string ids = (std::filesystem::temp_directory_path() /
                             ("nntr-ppl-decode-" + std::to_string(skip)))
                              .string();
    std::remove(ids.c_str());
    const std::string self = pplDecodeRun(f, ids);
    ASSERT_FALSE(self.empty()) << "skip_prefill=" << skip;
    EXPECT_EQ(pplField(self, "source="), "self") << self;
    const std::string n = pplField(self, "tokens=");
    ASSERT_FALSE(n.empty()) << self;
    EXPECT_EQ(pplField(self, "top1="), n + "/" + n) << self;

    std::vector<unsigned int> path;
    {
      std::ifstream in(ids);
      for (unsigned int id; in >> id;)
        path.push_back(id);
    }
    // the prefill's pick (when there is one) plus one id a scored step
    ASSERT_EQ(path.size(), std::stoul(n) + (skip ? 0u : 1u)) << self;

    const std::string forced = pplDecodeRun(f, ids);
    EXPECT_EQ(pplField(forced, "source="), "file") << forced;
    EXPECT_EQ(pplField(forced, "tokens="), n) << forced;
    EXPECT_EQ(pplField(forced, "top1="), n + "/" + n) << forced;
    EXPECT_EQ(pplField(forced, "nll_sum="), pplField(self, "nll_sum="))
      << self << "\n"
      << forced;

    // a different target at the last scored step: read, not the greedy one
    path.back() = path.back() == 0u ? 1u : path.back() - 1u;
    {
      std::ofstream out(ids);
      for (unsigned int id : path)
        out << id << '\n';
    }
    const std::string changed = pplDecodeRun(f, ids);
    EXPECT_EQ(pplField(changed, "top1="),
              std::to_string(std::stoul(n) - 1u) + "/" + n)
      << changed;
    EXPECT_NE(pplField(changed, "nll_sum="), pplField(self, "nll_sum="))
      << changed;
    std::remove(ids.c_str());
  }
}
} // namespace
