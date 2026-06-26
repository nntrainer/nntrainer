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
