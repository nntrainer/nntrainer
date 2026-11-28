// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_mha_cpu_profile.cpp
 * @date   28 October 2026
 * @brief  Android CPU FP32 external-cache attention profile admission tests
 * @author Codex (OpenAI)
 * @bug    Checks Android native-FP16 admission without executing attention.
 */

#include <env_compat.h>
#include <gtest/gtest.h>
#include <layer_context.h>
#include <mha_core.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/** Restore environment state even when a profile check throws. */
class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : key(name) {
    const char *old = std::getenv(name);
    had_old = old != nullptr;
    if (had_old)
      old_value = old;
    assign(value);
  }

  ~ScopedEnv() { assign(had_old ? old_value.c_str() : nullptr); }

  ScopedEnv(const ScopedEnv &) = delete;
  ScopedEnv &operator=(const ScopedEnv &) = delete;

private:
  void assign(const char *value) const {
    if (value) {
      ::setenv(key.c_str(), value, 1);
    } else {
#ifdef _WIN32
      ::_putenv_s(key.c_str(), "");
#else
      ::unsetenv(key.c_str());
#endif
    }
  }

  std::string key, old_value;
  bool had_old;
};

#if ENABLE_FP16 && defined(__ANDROID__)
using DataType = ml::train::TensorDim::DataType;
using ExecutionMode = ml::train::ExecutionMode;
using ComputeEngine = ml::train::LayerComputeEngine;

struct Profile {
  unsigned int batch = 1;
  DataType query = DataType::FP32;
  DataType key = DataType::FP32;
  DataType value = DataType::FP32;
  DataType cache = DataType::FP16;
  ExecutionMode mode = ExecutionMode::INFERENCE;
  ComputeEngine engine = ComputeEngine::CPU;
  bool external = true;
  bool rope = true;
  bool sink = false;
  std::string activation = "FP32";
};

class MHACpuProfile : public ::testing::Test {
protected:
  MHACpuProfile() :
    option("NNTR_CPU_ATTENTION_FP32", "1"),
    engine("NNTR_ENGINE", "cpu"),
    int8("NNTR_KV_INT8", nullptr),
    ohwi("NNTR_KV_OHWI", nullptr),
    ring("NNTR_KV_WINDOW_RING", "0"),
    image("NNTR_KV_IMG_ATTN", nullptr),
    mha_gpu("NNTR_MHA_GPU", "0") {}

  nntrainer::InitLayerContext context(const Profile &p) {
    const auto dim = [&p](unsigned int rows, unsigned int width,
                          DataType type) {
      return ml::train::TensorDim({p.batch, 1, rows, width},
                                  {ml::train::TensorDim::Format::NCHW, type});
    };
    std::vector<ml::train::TensorDim> inputs = {
      dim(4, 32, p.query), dim(4, 16, p.key), dim(4, 16, p.value)};
    if (p.external) {
      inputs.push_back(dim(16, 16, p.cache));
      inputs.push_back(dim(16, 16, p.cache));
    }
    return nntrainer::InitLayerContext(inputs, {true}, false, "mha_cpu_profile",
                                       "", 0.0f, {"NCHW", "FP32", p.activation},
                                       1.0f, p.mode, p.engine);
  }

  void finalize(const Profile &p) {
    causallm::MHACoreLayer layer;
    layer.setProperty({"num_heads=4", "num_heads_KV=2", "max_timestep=16",
                       "max_position_embeddings=16", "init_seq_len=4",
                       "use_gemm_attention=true",
                       std::string("rope_theta=") + (p.rope ? "10000" : "0"),
                       std::string("use_sink=") + (p.sink ? "true" : "false")});
    auto init = context(p);
    layer.finalize(init);
    ASSERT_EQ(init.getOutSpecs().size(), 1u);
    EXPECT_EQ(init.getOutSpecs()[0].variable_spec.dim.batch(), p.batch);
  }

  void rejected(const Profile &p) {
    try {
      finalize(p);
      FAIL() << "Unsupported profile was accepted";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find("NNTR_CPU_ATTENTION_FP32"),
                std::string::npos)
        << error.what();
    }
  }

private:
  ScopedEnv option, engine, int8, ohwi, ring, image, mha_gpu;
};

TEST_F(MHACpuProfile, SupportedInference) {
  EXPECT_NO_THROW(finalize(Profile{}));
}

TEST_F(MHACpuProfile, ExplicitZeroPreservesLegacy) {
  ScopedEnv disabled("NNTR_CPU_ATTENTION_FP32", "0");
  Profile p;
  p.batch = 2;
  p.query = p.key = p.value = DataType::FP16;
  p.activation = "FP16";
  EXPECT_NO_THROW(finalize(p));
}

TEST_F(MHACpuProfile, RejectsTraining) {
  Profile p;
  p.mode = ExecutionMode::TRAIN;
  rejected(p);
}

TEST_F(MHACpuProfile, RejectsValidation) {
  Profile p;
  p.mode = ExecutionMode::VALIDATE;
  rejected(p);
}

TEST_F(MHACpuProfile, RejectsBatchTwo) {
  Profile p;
  p.batch = 2;
  rejected(p);
}

TEST_F(MHACpuProfile, AcceptsEngineNeutralResidencyContext) {
  // CausalLM declares mha_core engine-neutral. Its init context can carry a
  // GPU residency stamp even though NNTR_ENGINE=cpu runs attention on CPU.
  Profile p;
  p.engine = ComputeEngine::GPU;
  EXPECT_NO_THROW(finalize(p));
}

TEST_F(MHACpuProfile, UnavailableBackendUsesCpuFallback) {
  ScopedEnv unavailable("NNTR_ENGINE", "unregistered-profile-backend");
  EXPECT_NO_THROW(finalize(Profile{}));
}

TEST_F(MHACpuProfile, UnsetEngineUsesRegisteredCpuRuntime) {
  ScopedEnv unset("NNTR_ENGINE", nullptr);
  if (causallm::runtimeEngine() == "cpu")
    EXPECT_NO_THROW(finalize(Profile{}));
  else
    rejected(Profile{});
}

TEST_F(MHACpuProfile, RejectsHalfQuery) {
  Profile p;
  p.query = DataType::FP16;
  rejected(p);
}

TEST_F(MHACpuProfile, RejectsHalfKey) {
  Profile p;
  p.key = DataType::FP16;
  rejected(p);
}

TEST_F(MHACpuProfile, RejectsHalfValue) {
  Profile p;
  p.value = DataType::FP16;
  rejected(p);
}

TEST_F(MHACpuProfile, RejectsFloatCache) {
  Profile p;
  p.cache = DataType::FP32;
  rejected(p);
}

TEST_F(MHACpuProfile, RejectsInternalCache) {
  Profile p;
  p.external = false;
  rejected(p);
}

TEST_F(MHACpuProfile, RejectsNoRope) {
  Profile p;
  p.rope = false;
  rejected(p);
}

TEST_F(MHACpuProfile, RejectsInt8) {
  ScopedEnv int8("NNTR_KV_INT8", "1");
  rejected(Profile{});
}

TEST_F(MHACpuProfile, RejectsSink) {
  Profile p;
  p.sink = true;
  rejected(p);
}
#else
TEST(MHACpuProfile, RequiresAndroidFp16Build) {
  GTEST_SKIP() << "The FP32 preservation option belongs to Android FP16 builds";
}
#endif
} // namespace
