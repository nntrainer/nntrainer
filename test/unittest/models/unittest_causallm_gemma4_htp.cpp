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
} // namespace
