// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_ext_model.cpp
 * @brief   Host side of the model-extension contract: the plugin ABI check and
 *          load diagnostics (absent vs failed), and the speculative-decoding
 *          model search that must reach an LLM behind a vision encoder.
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "ext_model_loader.h"
#include "model_callbacks.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

using namespace quick_dot_ai;

// ---------------------------------------------------------------------------
// The host's registry, as the plugin fixture sees it. The test executable is
// linked with -rdynamic so the fixture's undefined reference binds here.
// ---------------------------------------------------------------------------
static std::vector<std::string> g_registered_ids;

namespace quick_dot_ai {
void register_model_descriptor(const ModelDescriptor *desc) {
  if (desc != nullptr && desc->id != nullptr)
    g_registered_ids.push_back(desc->id);
}
} // namespace quick_dot_ai

namespace {
class ExtModelLoad : public ::testing::Test {
protected:
  void SetUp() override { g_registered_ids.clear(); }
};
} // namespace

TEST(ExtModelAbi, AcceptsMatchingInfo) {
  const QdaExtAbiInfo host = currentExtAbiInfo();
  std::string why;
  EXPECT_TRUE(checkExtAbi(&host, host, &why));
  EXPECT_TRUE(why.empty());
}

TEST(ExtModelAbi, HostSizesAreTheCurrentLayout) {
  const QdaExtAbiInfo host = currentExtAbiInfo();
  EXPECT_EQ(host.abi_version, QDA_EXT_ABI_VERSION);
  EXPECT_EQ(host.sizeof_model_descriptor, sizeof(ModelDescriptor));
  EXPECT_EQ(host.sizeof_model_callbacks, sizeof(ModelCallbacks));
  // nine-field descriptor: the field a pre-SD plugin lacks is the last one
  EXPECT_EQ(offsetof(ModelDescriptor, sd_variant_id) + sizeof(void *),
            sizeof(ModelDescriptor));
}

TEST(ExtModelAbi, RefusesEveryKindOfMismatch) {
  const QdaExtAbiInfo host = currentExtAbiInfo();
  std::string why;

  QdaExtAbiInfo old_version = host;
  old_version.abi_version = host.abi_version - 1;
  EXPECT_FALSE(checkExtAbi(&old_version, host, &why));
  EXPECT_NE(why.find("ABI mismatch"), std::string::npos);

  QdaExtAbiInfo old_descriptor = host; // 8-field descriptor, same version
  old_descriptor.sizeof_model_descriptor = host.sizeof_model_descriptor - 8;
  EXPECT_FALSE(checkExtAbi(&old_descriptor, host, &why));
  EXPECT_NE(why.find("sizeof(ModelDescriptor)"), std::string::npos);

  QdaExtAbiInfo old_callbacks = host;
  old_callbacks.sizeof_model_callbacks = host.sizeof_model_callbacks - 32;
  EXPECT_FALSE(checkExtAbi(&old_callbacks, host, &why));

  EXPECT_FALSE(checkExtAbi(nullptr, host, &why));
}

TEST(ExtModelLoadError, AbsentOnlyWhenThePluginItselfIsMissing) {
  const std::string lib = "libqai_ext_model.so";
  // bionic
  EXPECT_TRUE(extDlopenErrorMeansAbsent(
    "dlopen failed: library \"libqai_ext_model.so\" not found", lib));
  // glibc
  EXPECT_TRUE(extDlopenErrorMeansAbsent(
    "libqai_ext_model.so: cannot open shared object file: No such file or "
    "directory",
    lib));
  // the plugin is there but a dependency is not
  EXPECT_FALSE(extDlopenErrorMeansAbsent(
    "dlopen failed: library \"libquick_dot_ai_api.so\" needed by "
    "\"/data/app/x/lib/arm64/libqai_ext_model.so\" not found",
    lib));
  EXPECT_FALSE(extDlopenErrorMeansAbsent(
    "dlopen failed: cannot locate symbol \"_ZN6foo\" referenced by "
    "\"libqai_ext_model.so\"",
    lib));
  EXPECT_FALSE(extDlopenErrorMeansAbsent("dlopen failed: bad ELF magic", lib));
}

#if defined(EXT_GOOD_PLUGIN) && defined(EXT_STALE_PLUGIN) &&                   \
  defined(EXT_NO_ABI_PLUGIN)
TEST_F(ExtModelLoad, MatchingPluginIsLoadedAndRegisters) {
  const auto s = loadExtensionModelsFrom(EXT_GOOD_PLUGIN, true);
  EXPECT_EQ(s.state, ExtModelState::LOADED) << s.message;
  EXPECT_EQ(s.registrars, 1);
  EXPECT_EQ(s.message.rfind("loaded", 0), 0u);
  ASSERT_EQ(g_registered_ids.size(), 1u);
  EXPECT_EQ(g_registered_ids[0], "fixture-model");
}

TEST_F(ExtModelLoad, StalePluginIsRefusedBeforeItRegistersAnything) {
  const auto s = loadExtensionModelsFrom(EXT_STALE_PLUGIN, true);
  EXPECT_EQ(s.state, ExtModelState::FAILED);
  EXPECT_NE(s.message.find("ABI mismatch"), std::string::npos) << s.message;
  EXPECT_NE(s.message.find("plugin was built for (abi 0"), std::string::npos)
    << s.message;
  EXPECT_TRUE(g_registered_ids.empty());
}

TEST_F(ExtModelLoad, PluginWithoutAbiQueryIsRefused) {
  const auto s = loadExtensionModelsFrom(EXT_NO_ABI_PLUGIN, true);
  EXPECT_EQ(s.state, ExtModelState::FAILED);
  EXPECT_NE(s.message.find("qai_ext_model_abi_info"), std::string::npos)
    << s.message;
  EXPECT_TRUE(g_registered_ids.empty());
}
#endif

TEST_F(ExtModelLoad, NoPluginOnThePathIsAbsentNotFailed) {
  const auto s = loadExtensionModelsFrom("libqai_ext_model_nonexistent.so",
                                         /*explicit_path=*/false);
  // On bionic/glibc the loader reports the name it could not find.
  EXPECT_EQ(s.state, ExtModelState::ABSENT) << s.message;
  EXPECT_EQ(s.message.rfind("absent", 0), 0u);
}

TEST_F(ExtModelLoad, NamedFileThatIsMissingIsAFailure) {
  const auto s = loadExtensionModelsFrom("/nonexistent/dir/libqai_ext_model.so",
                                         /*explicit_path=*/true);
  EXPECT_EQ(s.state, ExtModelState::FAILED);
  EXPECT_EQ(s.message.rfind("failed", 0), 0u);
}

// ---------------------------------------------------------------------------
// configureSpeculativeDecoding(): which sub-model is configured
// ---------------------------------------------------------------------------
namespace {
ErrorCode noop_sd(causallm::Transformer *, bool) {
  return CAUSAL_LM_ERROR_NONE;
}
} // namespace

TEST(SpeculativeDecodingSearch, ReachesTheLlmBehindAVisionEncoder) {
  auto &reg = ModelCallbackRegistry::instance();
  ModelCallbacks vision; // a producer: no SD
  ModelCallbacks llm;
  llm.configure_speculative_decoding = noop_sd;
  reg.register_for("TestVisionEncoder", vision);
  reg.register_for("TestSdLlm", llm);

  EXPECT_EQ(
    findSpeculativeDecodingModel({"TestVisionEncoder", "TestSdLlm"}, reg), 1);
  // single-model handle
  EXPECT_EQ(findSpeculativeDecodingModel({"TestSdLlm"}, reg), 0);
  // the first model that has it wins
  reg.register_for("TestSdLlm2", llm);
  EXPECT_EQ(findSpeculativeDecodingModel(
              {"TestVisionEncoder", "TestSdLlm", "TestSdLlm2"}, reg),
            1);
  EXPECT_EQ(findSpeculativeDecodingModel(
              {"TestVisionEncoder", "TestSdLlm2", "TestSdLlm"}, reg),
            1);
}

TEST(SpeculativeDecodingSearch, NoCallbackAnywhereIsNotFound) {
  auto &reg = ModelCallbackRegistry::instance();
  EXPECT_EQ(findSpeculativeDecodingModel({}, reg), -1);
  EXPECT_EQ(findSpeculativeDecodingModel({"NeverRegisteredArch"}, reg), -1);
  reg.register_for("TestPlainLlm", ModelCallbacks{});
  EXPECT_EQ(findSpeculativeDecodingModel({"TestPlainLlm", "Other"}, reg), -1);
}
