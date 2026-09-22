// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file        unittest_dynamic_library_loader.cpp
 * @date        22 September 2026
 * @brief       Unit test for DynamicLibraryLoader error reporting
 * @see         https://github.com/nntrainer/nntrainer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <type_traits>

#include <dynamic_library_loader.h>

namespace {

#if defined(_WIN32)
constexpr const char *valid_library = "kernel32.dll";
constexpr const char *valid_symbol = "Sleep";
constexpr const char *missing_library = "nntrainer_no_such_library_42.dll";
#else
constexpr const char *valid_library = nullptr; /**< the running program */
constexpr const char *valid_symbol = "printf";
constexpr const char *missing_library = "nntrainer_no_such_library_42.so";
#endif

constexpr const char *missing_symbol = "nntrainer_no_such_symbol_42";

/**
 * @brief Fixture draining the error left behind by unrelated code
 */
class DynamicLibraryLoaderTest : public ::testing::Test {
protected:
  void SetUp() override { nntrainer::DynamicLibraryLoader::getLastError(); }
};

} // namespace

/**
 * @brief getLastError() must hand out an owning string, not a pointer into a
 * destroyed temporary
 */
TEST_F(DynamicLibraryLoaderTest, getLastErrorReturnsOwningString) {
  static_assert(
    std::is_same_v<decltype(nntrainer::DynamicLibraryLoader::getLastError()),
                   std::string>,
    "getLastError() must return std::string by value");
}

/**
 * @brief a successful load must not report an error, otherwise every caller
 * checking the error after loadSymbol() fails spuriously
 */
TEST_F(DynamicLibraryLoaderTest, successfulLoadReportsNoError) {
  void *handle = nntrainer::DynamicLibraryLoader::loadLibrary(
    valid_library, RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr) << nntrainer::DynamicLibraryLoader::getLastError();
  EXPECT_TRUE(nntrainer::DynamicLibraryLoader::getLastError().empty());

  void *symbol =
    nntrainer::DynamicLibraryLoader::loadSymbol(handle, valid_symbol);
  EXPECT_NE(symbol, nullptr);
  EXPECT_TRUE(nntrainer::DynamicLibraryLoader::getLastError().empty());

  nntrainer::DynamicLibraryLoader::freeLibrary(handle);
}

/**
 * @brief an error raised by unrelated code must not be reported as a loader
 * error of the following successful load
 */
TEST_F(DynamicLibraryLoaderTest, unrelatedErrorIsNotReportedAsLoaderError) {
  ASSERT_NE(std::remove("nntrainer_no_such_file_42"), 0);

  void *handle = nntrainer::DynamicLibraryLoader::loadLibrary(
    valid_library, RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr);
  EXPECT_TRUE(nntrainer::DynamicLibraryLoader::getLastError().empty());

  nntrainer::DynamicLibraryLoader::freeLibrary(handle);
}

/**
 * @brief a failed load must report a readable, non-empty description which is
 * consumed once read
 */
TEST_F(DynamicLibraryLoaderTest, loadLibraryMissingReportsError_n) {
  void *handle = nntrainer::DynamicLibraryLoader::loadLibrary(
    missing_library, RTLD_LAZY | RTLD_LOCAL);
  EXPECT_EQ(handle, nullptr);

  EXPECT_FALSE(nntrainer::DynamicLibraryLoader::getLastError().empty());
  EXPECT_TRUE(nntrainer::DynamicLibraryLoader::getLastError().empty());
}

/**
 * @brief a failed symbol lookup must report a non-empty description
 */
TEST_F(DynamicLibraryLoaderTest, loadSymbolMissingReportsError_n) {
  void *handle = nntrainer::DynamicLibraryLoader::loadLibrary(
    valid_library, RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr);
  nntrainer::DynamicLibraryLoader::getLastError();

  EXPECT_EQ(nntrainer::DynamicLibraryLoader::loadSymbol(handle, missing_symbol),
            nullptr);
  EXPECT_FALSE(nntrainer::DynamicLibraryLoader::getLastError().empty());

  nntrainer::DynamicLibraryLoader::freeLibrary(handle);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
