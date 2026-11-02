// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_causallm_runtime.cpp
 * @date 29 October 2026
 * @brief Cross-platform CPU attention policy and runtime snapshot contracts
 * @author Codex (OpenAI)
 * @bug Device kernels are validated by the separate Android model tests.
 */
#include <env_compat.h>
#include <gtest/gtest.h>
#include <runtime_config.h>

#include <cstdlib>
#include <string>

namespace {
class ScopedEnv {
public:
  ScopedEnv(const char *key, const char *value) : name(key) {
    const auto *old = std::getenv(key);
    present = old != nullptr;
    if (present)
      previous = old;
    assign(value);
  }
  ~ScopedEnv() { assign(present ? previous.c_str() : nullptr); }
  ScopedEnv(const ScopedEnv &) = delete;
  ScopedEnv &operator=(const ScopedEnv &) = delete;

private:
  void assign(const char *value) const {
    if (value) {
      ::setenv(name.c_str(), value, 1);
    } else {
#ifdef _WIN32
      ::_putenv_s(name.c_str(), "");
#else
      ::unsetenv(name.c_str());
#endif
    }
  }
  std::string name, previous;
  bool present;
};

using causallm::CpuAttentionProfile;
using causallm::resolveRuntimeEngine;
using causallm::RuntimeConfig;
using causallm::supportsCpuAttentionFp32;

TEST(CausalRuntimeEngine, UnsetUsesLiveContexts) {
  EXPECT_EQ(resolveRuntimeEngine("", false, false, false), "cpu");
  EXPECT_EQ(resolveRuntimeEngine("", false, true, false), "gpu");
  EXPECT_EQ(resolveRuntimeEngine("", false, false, true), "cuda");
  EXPECT_EQ(resolveRuntimeEngine("", false, true, true), "gpu");
}

TEST(CausalRuntimeEngine, ExplicitRegisteredSelectionWins) {
  EXPECT_EQ(resolveRuntimeEngine("cpu", true, true, true), "cpu");
  EXPECT_EQ(resolveRuntimeEngine("cuda", true, true, true), "cuda");
  EXPECT_EQ(resolveRuntimeEngine("gpu", true, true, true), "gpu");
  EXPECT_EQ(resolveRuntimeEngine("vendor", true, true, true), "vendor");
}

TEST(CausalRuntimeEngine, UnavailableSelectionFollowsCoreCpuFallback) {
  EXPECT_EQ(resolveRuntimeEngine("gpu", false, false, false), "cpu");
  EXPECT_EQ(resolveRuntimeEngine("cuda", false, true, false), "cpu");
  EXPECT_EQ(resolveRuntimeEngine("unknown", false, true, true), "cpu");
}

class CausalRuntimeSnapshot : public ::testing::Test {
protected:
  CausalRuntimeSnapshot() :
    engine("NNTR_ENGINE", "cpu"),
    fp32("NNTR_CPU_ATTENTION_FP32", nullptr),
    tiled("NNTR_CPU_KV_TILED", nullptr),
    trace("NNTR_KV_CPU_TRACE", nullptr),
    legacy("NNTR_CPU_KV_LEGACY", nullptr) {}

private:
  ScopedEnv engine, fp32, tiled, trace, legacy;
};

TEST_F(CausalRuntimeSnapshot, LegacyOptionsAreDefault) {
  const auto config = RuntimeConfig::captureForEngine("cpu");
  EXPECT_TRUE(config.cpu);
  EXPECT_EQ(config.engine, "cpu");
  EXPECT_FALSE(config.cpu_attention_fp32);
  EXPECT_FALSE(config.cpu_kv_tiled);
  EXPECT_FALSE(config.cpu_trace);
  EXPECT_FALSE(config.cpu_kv_legacy);
  EXPECT_FALSE(config.device_attention);
}

TEST_F(CausalRuntimeSnapshot, TiledAndFp32AreIndependent) {
  {
    ScopedEnv tiled("NNTR_CPU_KV_TILED", "1");
    const auto config = RuntimeConfig::captureForEngine("cpu");
    EXPECT_TRUE(config.cpu_kv_tiled);
    EXPECT_FALSE(config.cpu_attention_fp32);
  }
  {
    ScopedEnv fp32("NNTR_CPU_ATTENTION_FP32", "1");
    const auto config = RuntimeConfig::captureForEngine("cpu");
    EXPECT_FALSE(config.cpu_kv_tiled);
    EXPECT_TRUE(config.cpu_attention_fp32);
  }
}

TEST_F(CausalRuntimeSnapshot, ExplicitZeroDisablesEachOption) {
  ScopedEnv tiled("NNTR_CPU_KV_TILED", "0");
  ScopedEnv fp32("NNTR_CPU_ATTENTION_FP32", "0");
  ScopedEnv trace("NNTR_KV_CPU_TRACE", "0");
  const auto config = RuntimeConfig::captureForEngine("cpu");
  EXPECT_FALSE(config.cpu_kv_tiled);
  EXPECT_FALSE(config.cpu_attention_fp32);
  EXPECT_FALSE(config.cpu_trace);
}

TEST_F(CausalRuntimeSnapshot, FreshObjectsSeeEnvChangesOldObjectsStayStable) {
  const auto old_config = RuntimeConfig::captureForEngine("cpu");
  ScopedEnv tiled("NNTR_CPU_KV_TILED", "1");
  ScopedEnv fp32("NNTR_CPU_ATTENTION_FP32", "1");
  ScopedEnv trace("NNTR_KV_CPU_TRACE", "1");
  const auto new_config = RuntimeConfig::captureForEngine("cpu");
  EXPECT_FALSE(old_config.cpu_kv_tiled);
  EXPECT_FALSE(old_config.cpu_attention_fp32);
  EXPECT_FALSE(old_config.cpu_trace);
  EXPECT_TRUE(new_config.cpu_kv_tiled);
  EXPECT_TRUE(new_config.cpu_attention_fp32);
  EXPECT_TRUE(new_config.cpu_trace);
}

TEST_F(CausalRuntimeSnapshot, UppercaseCpuAndUnavailableBackendResolveToCpu) {
  ScopedEnv uppercase("NNTR_ENGINE", "CPU");
  EXPECT_EQ(nntr_engine_env(), "cpu");
  EXPECT_EQ(resolveRuntimeEngine(nntr_engine_env(), true, false, false), "cpu");
  ScopedEnv unavailable("NNTR_ENGINE", "unregistered-profile-backend");
  EXPECT_EQ(resolveRuntimeEngine(nntr_engine_env(), false, false, false),
            "cpu");
}

TEST_F(CausalRuntimeSnapshot, UnsetAndExplicitCpuShareCpuOnlyFallback) {
  const auto explicit_cpu = RuntimeConfig::captureForEngine(
    resolveRuntimeEngine("cpu", true, false, false));
  ScopedEnv unset("NNTR_ENGINE", nullptr);
  const auto implicit_cpu = RuntimeConfig::captureForEngine(
    resolveRuntimeEngine(nntr_engine_env(), false, false, false));
  EXPECT_EQ(implicit_cpu.engine, explicit_cpu.engine);
  EXPECT_EQ(implicit_cpu.cpu, explicit_cpu.cpu);
  EXPECT_EQ(implicit_cpu.cpu_trace, explicit_cpu.cpu_trace);
}

TEST_F(CausalRuntimeSnapshot, DeviceRuntimeIgnoresCpuOnlyFlags) {
  ScopedEnv tiled("NNTR_CPU_KV_TILED", "1");
  ScopedEnv trace("NNTR_KV_CPU_TRACE", "1");
  ScopedEnv legacy("NNTR_CPU_KV_LEGACY", "1");
  const auto config = RuntimeConfig::captureForEngine("gpu");
  EXPECT_FALSE(config.cpu);
  EXPECT_FALSE(config.cpu_kv_tiled);
  EXPECT_FALSE(config.cpu_kv_legacy);
  EXPECT_FALSE(config.cpu_trace);
}

TEST_F(CausalRuntimeSnapshot, LegacyCpuFacadeIsAnIndependentOptOut) {
  ScopedEnv legacy("NNTR_CPU_KV_LEGACY", "1");
  const auto config = RuntimeConfig::captureForEngine("cpu");
  EXPECT_TRUE(config.cpu_kv_legacy);
  EXPECT_FALSE(config.cpu_attention_fp32);
  EXPECT_FALSE(config.cpu_kv_tiled);
  ScopedEnv disabled("NNTR_CPU_KV_LEGACY", "0");
  EXPECT_FALSE(RuntimeConfig::captureForEngine("cpu").cpu_kv_legacy);
}

TEST(CpuAttentionAdmission, SupportsExternalFp32RopeHalfCacheInference) {
  EXPECT_TRUE(supportsCpuAttentionFp32(CpuAttentionProfile{}));
}

TEST(CpuAttentionAdmission, RequiresCpuRuntime) {
  CpuAttentionProfile profile;
  profile.cpu = false;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}

TEST(CpuAttentionAdmission, RequiresInference) {
  CpuAttentionProfile profile;
  profile.inference = false;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}

TEST(CpuAttentionAdmission, RequiresBatchOne) {
  CpuAttentionProfile profile;
  profile.batch = 0;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
  profile.batch = 2;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}

TEST(CpuAttentionAdmission, RequiresExternalCache) {
  CpuAttentionProfile profile;
  profile.external_cache = false;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}

TEST(CpuAttentionAdmission, RejectsSink) {
  CpuAttentionProfile profile;
  profile.sink = true;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}

TEST(CpuAttentionAdmission, RejectsInt8) {
  CpuAttentionProfile profile;
  profile.int8_cache = true;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}

TEST(CpuAttentionAdmission, RejectsRing) {
  CpuAttentionProfile profile;
  profile.ring = true;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}

TEST(CpuAttentionAdmission, RejectsOhwi) {
  CpuAttentionProfile profile;
  profile.ohwi = true;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}

TEST(CpuAttentionAdmission, RequiresRope) {
  CpuAttentionProfile profile;
  profile.rope = false;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}

TEST(CpuAttentionAdmission, RequiresFp32Io) {
  CpuAttentionProfile profile;
  profile.fp32_io = false;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}

TEST(CpuAttentionAdmission, RequiresHalfCache) {
  CpuAttentionProfile profile;
  profile.half_cache = false;
  EXPECT_FALSE(supportsCpuAttentionFp32(profile));
}
} // namespace
