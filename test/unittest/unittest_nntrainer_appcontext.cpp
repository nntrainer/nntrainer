// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   unittest_app_context.h
 * @date   9 November 2020
 * @brief  This file contains app context related functions and classes that
 * manages the global configuration of the current environment
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <typeinfo>

#include <optimizer_devel.h>
#include <weight.h>

#include <app_context.h>
#include <engine.h>
#include <nntrainer_error.h>
#include <nntrainer_test_util.h>

/**
 * @brief   Directory for appcontext unittests
 *
 */
class nntrainerAppContextDirectory : public ::testing::Test {

protected:
  void SetUp() override {
    auto status = std::filesystem::create_directory("testdir");
    ASSERT_EQ(status, true);

    std::ofstream file(
      std::filesystem::path("testdir").append("testfile.txt").string());
    ASSERT_EQ(file.fail(), false);

    file << "testdata";
    ASSERT_EQ(file.fail(), false);

    file.close();

    char buf[2048];
    char *ret = getcwd(buf, 2048);
    ASSERT_NE(ret, nullptr);
    current_directory = std::string(buf);
  }

  void TearDown() override {
    int status = remove(
      std::filesystem::path("testdir").append("testfile.txt").string().c_str());
    ASSERT_EQ(status, 0);

    status = rmdir("testdir");
    ASSERT_EQ(status, 0);
  }

  std::string current_directory;
};

TEST_F(nntrainerAppContextDirectory, readFromGetPath_p) {
  auto &eg = nntrainer::Engine::Global();

  std::string path = eg.getWorkingPath("testfile.txt");
  EXPECT_EQ(path, "testfile.txt");

  eg.setWorkingDirectory("testdir");

  path = eg.getWorkingPath("testfile.txt");
  EXPECT_EQ(path, std::filesystem::path(current_directory)
                    .append("testdir")
                    .append("testfile.txt"));

  std::ifstream file(path);
  std::string s;
  file >> s;
  EXPECT_EQ(s, "testdata");

  file.close();

  const auto current_path_absolute = std::filesystem::current_path().string();
  path = eg.getWorkingPath(current_path_absolute);
  EXPECT_EQ(path, current_path_absolute);

  path = eg.getWorkingPath("");
  EXPECT_EQ(path, std::filesystem::path(current_directory).append("testdir"));
}

TEST_F(nntrainerAppContextDirectory, notExisitingSetDirectory_n) {
  auto &eg = nntrainer::Engine::Global();

  EXPECT_THROW(eg.setWorkingDirectory("testdir_does_not_exist"),
               std::invalid_argument);
}

/**
 * @brief   Custom Optimizer for unittests
 *
 */
class CustomOptimizer : public nntrainer::Optimizer {
public:
  /** Full custom optimizer example which overrides all functions */
  const std::string getType() const override { return "identity_optimizer"; }

  double getDefaultLearningRate() const override { return 1.0; }

  void setProperty(const std::vector<std::string> &values) override {}

  std::vector<nntrainer::TensorDim>
  getOptimizerVariableDim(const nntrainer::TensorDim &dim) override {
    return std::vector<nntrainer::TensorDim>();
  }

  void applyGradient(nntrainer::RunOptimizerContext &context) override {}
};

/**
 * @brief   Custom Optimizer for unittests
 *
 */
class CustomOptimizer2 : public nntrainer::Optimizer {
public:
  /** Minimal custom optimizer example which define only necessary functions */
  const std::string getType() const override { return "identity_optimizer"; }

  double getDefaultLearningRate() const override { return 1.0; }

  std::vector<nntrainer::TensorDim>
  getOptimizerVariableDim(const nntrainer::TensorDim &dim) override {
    return std::vector<nntrainer::TensorDim>();
  }

  void applyGradient(nntrainer::RunOptimizerContext &context) override {}
};

/**
 * @brief   Custom Layer for unittests
 *
 * @todo solidify the api signature
 */
class CustomLayer : public nntrainer::Layer {
public:
  static constexpr const char *type = "identity_layer";

  void setProperty(const std::vector<std::string> &values) override {}

  const std::string getType() const override { return CustomLayer::type; }
};

using AC = nntrainer::AppContext;

AC::PtrType<nntrainer::Optimizer>
createCustomOptimizer(const AC::PropsType &v) {
  auto p = std::make_unique<CustomOptimizer>();
  p->setProperty(v);
  return p;
}

/**
 * @brief AppContextTest for parametrized test
 *
 * @param std::string key of the registerFactory
 * @param int int_key of the registerFactory
 */
class AppContextTest
  : public ::testing::TestWithParam<std::tuple<std::string, int>> {};

TEST_P(AppContextTest, RegisterCreateCustomOptimizer_p) {
  std::tuple<std::string, int> param = GetParam();
  std::string key = std::get<0>(param);
  int int_key = std::get<1>(param);

  auto ac = nntrainer::AppContext();
  int num_id = ac.registerFactory(createCustomOptimizer, key, int_key);
  EXPECT_EQ(num_id, ((int_key == -1) ? (-1) * int_key : int_key));
  auto opt = ac.createObject<nntrainer::Optimizer>(
    ((key == "") ? "identity_optimizer" : key), {});
  auto &optimizer = *opt.get();
  EXPECT_EQ(typeid(optimizer).hash_code(), typeid(CustomOptimizer).hash_code());
  opt = ac.createObject<nntrainer::Optimizer>(num_id, {});
  auto &new_optimizer = *opt.get();
  EXPECT_EQ(typeid(new_optimizer).hash_code(),
            typeid(CustomOptimizer).hash_code());
}

GTEST_PARAMETER_TEST(RegisterCreateCustomOptimizerTests, AppContextTest,
                     ::testing::Values(std::make_tuple("", -1),
                                       std::make_tuple("custom_key", -1),
                                       std::make_tuple("custom_key", 5)));

TEST(AppContextTest, RegisterFactoryWithClashingKey_p) {
  auto ac = nntrainer::AppContext();

  int first_key = ac.registerFactory(createCustomOptimizer, "custom_key");
  int second_key = ac.registerFactory(createCustomOptimizer, "custom_key");

  EXPECT_EQ(first_key, second_key);
}

TEST(AppContextTest, RegisterFactoryWithClashingIntKey_p) {
  auto ac = nntrainer::AppContext();

  int first_key = ac.registerFactory(createCustomOptimizer, "custom_key", 3);
  int second_key =
    ac.registerFactory(createCustomOptimizer, "custom_other_key", 3);

  EXPECT_EQ(first_key, second_key);
}

TEST(AppContextTest, RegisterFactoryWithClashingAutoKey_p) {
  auto ac = nntrainer::AppContext();

  int first_key = ac.registerFactory(createCustomOptimizer);
  int second_key = ac.registerFactory(createCustomOptimizer);

  EXPECT_EQ(first_key, second_key);
}

TEST(AppContextTest, createObjectNotExistingKey_n) {
  auto ac = nntrainer::AppContext();

  ac.registerFactory(createCustomOptimizer);
  EXPECT_THROW(ac.createObject<nntrainer::Optimizer>("not_exisiting_key"),
               nntrainer::exception::not_supported);
}

TEST(AppContextTest, createObjectNotExistingIntKey_n) {
  auto ac = nntrainer::AppContext();

  int num = ac.registerFactory(createCustomOptimizer);
  EXPECT_THROW(ac.createObject<nntrainer::Optimizer>(num + 3),
               nntrainer::exception::not_supported);
}

TEST(AppContextTest, callingUnknownFactoryOptimizerWithKey_n) {
  auto ac = nntrainer::AppContext();

  int num = ac.registerFactory(
    nntrainer::AppContext::unknownFactory<nntrainer::Optimizer>, "unknown",
    999);

  EXPECT_EQ(num, 999);
  EXPECT_THROW(ac.createObject<nntrainer::Optimizer>("unknown"),
               std::invalid_argument);
}

TEST(AppContextTest, callingUnknownFactoryOptimizerWithIntKey_n) {
  auto ac = nntrainer::AppContext();

  int num = ac.registerFactory(
    nntrainer::AppContext::unknownFactory<nntrainer::Optimizer>, "unknown",
    999);

  EXPECT_EQ(num, 999);
  EXPECT_THROW(ac.createObject<nntrainer::Optimizer>(num),
               std::invalid_argument);
}

/**
 * @brief   ExecPlan resolver tests (caps-only overload)
 *
 */
TEST(ExecPlanResolverTest, capsOnly_cpuBackend_p) {
  nntrainer::DeviceCaps caps;
  caps.backend = "cpu";

  auto plan = nntrainer::resolveExecPlan(caps);
  EXPECT_EQ(plan.gemm_path, nntrainer::GemmPath::CPU);
}

TEST(ExecPlanResolverTest, capsOnly_gpuWithDpas_p) {
  nntrainer::DeviceCaps caps;
  caps.backend = "gpu";
  caps.dpas = true;

  auto plan = nntrainer::resolveExecPlan(caps);
  EXPECT_EQ(plan.gemm_path, nntrainer::GemmPath::XMX);
}

TEST(ExecPlanResolverTest, capsOnly_gpuWithoutDpas_p) {
  nntrainer::DeviceCaps caps;
  caps.backend = "gpu";
  caps.dpas = false;

  auto plan = nntrainer::resolveExecPlan(caps);
  EXPECT_EQ(plan.gemm_path, nntrainer::GemmPath::DP4A);
}

TEST(ExecPlanResolverTest, capsOnly_cudaBackend_p) {
  nntrainer::DeviceCaps caps;
  caps.backend = "cuda";

  auto plan = nntrainer::resolveExecPlan(caps);
  EXPECT_EQ(plan.gemm_path, nntrainer::GemmPath::CUBLAS);
}

TEST(ExecPlanResolverTest, capsOnly_hostCoherentMirrorsIntegrated_p) {
  nntrainer::DeviceCaps integrated_caps;
  integrated_caps.backend = "gpu";
  integrated_caps.integrated = true;
  EXPECT_EQ(nntrainer::resolveExecPlan(integrated_caps).host_coherent, true);

  nntrainer::DeviceCaps discrete_caps;
  discrete_caps.backend = "gpu";
  discrete_caps.integrated = false;
  EXPECT_EQ(nntrainer::resolveExecPlan(discrete_caps).host_coherent, false);
}

TEST(ExecPlanResolverTest, capsOnly_decodeGpuDefaultsFalse_p) {
  nntrainer::DeviceCaps caps;
  caps.backend = "gpu";
  caps.dpas = true;

  auto plan = nntrainer::resolveExecPlan(caps);
  EXPECT_EQ(plan.decode_gpu, false);
}

/**
 * @brief   ExecPlan resolver tests (caps + ModelFeatures matcher overload)
 *
 */
TEST(ExecPlanResolverTest, matcher_decodeGpuOnGpuBackend_p) {
  nntrainer::DeviceCaps caps;
  caps.backend = "gpu";
  caps.dpas = true;

  nntrainer::ModelFeatures features;
  features.decode_gpu = true;

  auto plan = nntrainer::resolveExecPlan(caps, features);
  EXPECT_EQ(plan.decode_gpu, true);
}

TEST(ExecPlanResolverTest, matcher_decodeGpuOnCudaBackend_p) {
  nntrainer::DeviceCaps caps;
  caps.backend = "cuda";

  nntrainer::ModelFeatures features;
  features.decode_gpu = true;

  auto plan = nntrainer::resolveExecPlan(caps, features);
  EXPECT_EQ(plan.decode_gpu, true);
}

TEST(ExecPlanResolverTest, matcher_decodeGpuGatedOffOnCpuBackend_n) {
  nntrainer::DeviceCaps caps;
  caps.backend = "cpu";

  nntrainer::ModelFeatures features;
  features.decode_gpu = true;

  auto plan = nntrainer::resolveExecPlan(caps, features);
  EXPECT_EQ(plan.decode_gpu, false);
}

TEST(ExecPlanResolverTest, matcher_decodeGpuFalseStaysFalseOnGpuBackend_n) {
  nntrainer::DeviceCaps caps;
  caps.backend = "gpu";
  caps.dpas = true;

  nntrainer::ModelFeatures features;
  features.decode_gpu = false;

  auto plan = nntrainer::resolveExecPlan(caps, features);
  EXPECT_EQ(plan.decode_gpu, false);
}

TEST(ExecPlanResolverTest, matcher_capsDerivedCellsMatchCapsOnlyOverload_p) {
  nntrainer::DeviceCaps caps;
  caps.backend = "gpu";
  caps.dpas = true;
  caps.integrated = false;

  nntrainer::ModelFeatures features;
  features.decode_gpu = true;

  auto caps_only_plan = nntrainer::resolveExecPlan(caps);
  auto matcher_plan = nntrainer::resolveExecPlan(caps, features);

  EXPECT_EQ(matcher_plan.gemm_path, caps_only_plan.gemm_path);
  EXPECT_EQ(matcher_plan.host_coherent, caps_only_plan.host_coherent);
}

/**
 * @brief   toString() smoke tests for the ExecPlan/ModelFeatures resolver
 *          types and their enums
 *
 */
TEST(ExecPlanResolverTest, toString_execPlanContainsGemmPathKey_p) {
  nntrainer::ExecPlan plan;
  std::string dump = plan.toString();

  EXPECT_FALSE(dump.empty());
  EXPECT_NE(dump.find("gemm_path="), std::string::npos);
}

TEST(ExecPlanResolverTest, toString_modelFeaturesContainsQkNormKey_p) {
  nntrainer::ModelFeatures features;
  std::string dump = features.toString();

  EXPECT_FALSE(dump.empty());
  EXPECT_NE(dump.find("qk_norm="), std::string::npos);
}

TEST(ExecPlanResolverTest, toString_gemmPathEnumLiterals_p) {
  EXPECT_STREQ(nntrainer::toString(nntrainer::GemmPath::CPU), "CPU");
  EXPECT_STREQ(nntrainer::toString(nntrainer::GemmPath::DP4A), "DP4A");
  EXPECT_STREQ(nntrainer::toString(nntrainer::GemmPath::XMX), "XMX");
  EXPECT_STREQ(nntrainer::toString(nntrainer::GemmPath::CUBLAS), "CUBLAS");
}

/**
 * @brief   Minimal concrete layer, so the registration facade can be exercised
 *          end to end (register through Engine, then create through Context).
 */
class FacadeLayer : public nntrainer::Layer {
public:
  static constexpr const char *type = "facade_test_layer";

  void finalize(nntrainer::InitLayerContext &context) override {}
  void forwarding(nntrainer::RunLayerContext &context, bool training) override {
  }
  void calcDerivative(nntrainer::RunLayerContext &context) override {}
  void setProperty(const std::vector<std::string> &values) override {}
  bool supportBackwarding() const override { return false; }
  const std::string getType() const override { return FacadeLayer::type; }
};

/**
 * @brief   The open backend registry: an engine name resolves against the LIVE
 *          registered-context set, not a closed enum, and an unregistered name
 *          falls back to "cpu" rather than resolving to something
 *          getRegisteredContext() would later reject.
 */
TEST(EngineRegistryTest, parseComputeEngineResolvesRegisteredName_p) {
  auto &engine = nntrainer::Engine::Global();

  EXPECT_EQ(engine.parseComputeEngine({"engine=cpu"}), "cpu");
  /** case-insensitive: the property is normalised before the lookup */
  EXPECT_EQ(engine.parseComputeEngine({"engine=CPU"}), "cpu");
  /** no engine property at all keeps the default */
  EXPECT_EQ(engine.parseComputeEngine({}), "cpu");
}

TEST(EngineRegistryTest, parseComputeEngineFallsBackForUnknownName_n) {
  auto &engine = nntrainer::Engine::Global();

  /** a name no Context registered under must not resolve to itself: the graph
   *  would then ask getRegisteredContext() for a name that throws */
  EXPECT_EQ(engine.parseComputeEngine({"engine=no_such_backend"}), "cpu");
  EXPECT_NO_THROW(
    engine.getRegisteredContext(engine.parseComputeEngine({"engine=vulkan"})));
}

/**
 * @brief   The registration facade dispatches through Context, so a caller
 *          never needs the concrete context type.
 */
TEST(EngineRegistryTest, registerLayerFactoryThroughFacade_p) {
  auto &engine = nntrainer::Engine::Global();

  int key = engine.registerLayerFactory(
    "cpu", nntrainer::createLayer<FacadeLayer>, "facade_test_layer");
  EXPECT_NE(key, -1);

  auto layer = engine.getRegisteredContext("cpu")->createLayerObject(
    "facade_test_layer", {});
  EXPECT_NE(layer, nullptr);
}

TEST(EngineRegistryTest, registerLayerFactoryOnUnknownEngine_n) {
  auto &engine = nntrainer::Engine::Global();

  /** an unregistered engine name is an error at resolution, not a silent -1 */
  EXPECT_THROW(engine.registerLayerFactory("no_such_backend",
                                           nntrainer::createLayer<FacadeLayer>,
                                           "unreachable_layer"),
               std::invalid_argument);
}

/**
 * @brief   Each Context declares its own residency plane, so the layer graph
 *          needs no central name-to-enum table.
 */
TEST(EngineRegistryTest, contextDeclaresItsResidencyPlane_p) {
  auto &engine = nntrainer::Engine::Global();

  EXPECT_EQ(engine.getRegisteredContext("cpu")->residencyEngine(),
            ml::train::LayerComputeEngine::CPU);
}

/**
 * @brief   The CPU context reports a host-coherent capability snapshot, and the
 *          resolver turns it into a CPU execution plan.
 */
TEST(EngineRegistryTest, cpuContextReportsHostCoherentCaps_p) {
  auto &engine = nntrainer::Engine::Global();
  const nntrainer::DeviceCaps &caps =
    engine.getRegisteredContext("cpu")->caps();

  EXPECT_EQ(caps.backend, "cpu");
  EXPECT_TRUE(caps.integrated);

  auto plan = nntrainer::resolveExecPlan(caps);
  EXPECT_EQ(plan.gemm_path, nntrainer::GemmPath::CPU);
  EXPECT_TRUE(plan.host_coherent);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
