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

#if defined(_WIN32)
/**
 * @brief path of the dl_error_fixture module, from --dl-error-fixture=<path>
 * or, failing that, dl_error_fixture.dll next to the test executable
 */
std::string dl_error_fixture;

/**
 * @brief whether this system has a message text for @a code in a language
 * describeError() tries (language 0 also searches US English)
 *
 * @note May leave a thread error behind; call it before the loader call under
 * test, not between that call and getLastError().
 */
bool systemHasMessage(DWORD code) {
  LPSTR buffer = nullptr;
  const DWORD length = FormatMessageA(
    FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
      FORMAT_MESSAGE_IGNORE_INSERTS,
    nullptr, code, 0, reinterpret_cast<LPSTR>(&buffer), 0, nullptr);
  if (buffer != nullptr) {
    LocalFree(buffer);
  }
  return length != 0;
}
#endif

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

#if defined(_WIN32)
/**
 * @brief an error left behind right before a successful symbol lookup must not
 * be reported as a loader error of that lookup
 *
 * @note GetProcAddress() is not documented to reset the thread error on success
 * and does not in practice, so this pins the reset in loadSymbol(). Windows
 * only: on POSIX clearing the error on success is up to the dl implementation
 * and the wrapper adds nothing to pin.
 */
TEST_F(DynamicLibraryLoaderTest, staleErrorIsNotReportedAfterSymbolLookup) {
  void *handle = nntrainer::DynamicLibraryLoader::loadLibrary(
    valid_library, RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr);

  SetLastError(ERROR_ACCESS_DENIED);
  void *symbol =
    nntrainer::DynamicLibraryLoader::loadSymbol(handle, valid_symbol);
  EXPECT_NE(symbol, nullptr);
  EXPECT_TRUE(nntrainer::DynamicLibraryLoader::getLastError().empty());

  nntrainer::DynamicLibraryLoader::freeLibrary(handle);
}

/**
 * @brief an error raised inside a successful LoadLibraryA() call, here by the
 * loaded module's DllMain, must not be reported as a loader error
 *
 * @note This pins that loadLibrary() clears the error after the call rather
 * than before it. The fixture module is built and passed in by meson, or
 * found next to the test executable.
 */
TEST_F(DynamicLibraryLoaderTest, errorRaisedDuringSuccessfulLoadIsNotReported) {
  if (dl_error_fixture.empty()) {
    GTEST_SKIP() << "dl_error_fixture.dll is not next to the test; run with "
                    "--dl-error-fixture=<path>";
  }

  void *handle = nntrainer::DynamicLibraryLoader::loadLibrary(
    dl_error_fixture.c_str(), RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr) << nntrainer::DynamicLibraryLoader::getLastError();
  EXPECT_TRUE(nntrainer::DynamicLibraryLoader::getLastError().empty());

  nntrainer::DynamicLibraryLoader::freeLibrary(handle);
}

/**
 * @brief an error the system has no message text for is reported as the bare
 * code, and is still consumed once read
 *
 * @note Bit 29 marks a customer-defined code, which never has a system message.
 */
TEST_F(DynamicLibraryLoaderTest, errorWithoutSystemMessageIsReportedAsCode) {
  const DWORD code = 0x20000001;
  ASSERT_FALSE(systemHasMessage(code));
  const std::string expected = std::to_string(code);

  SetLastError(code);
  const std::string error = nntrainer::DynamicLibraryLoader::getLastError();
  EXPECT_EQ(error, expected);
  EXPECT_TRUE(nntrainer::DynamicLibraryLoader::getLastError().empty());
}
#endif

/**
 * @brief a failed load must report a readable, non-empty description which is
 * consumed once read
 */
TEST_F(DynamicLibraryLoaderTest, loadLibraryMissingReportsError_n) {
#if defined(_WIN32)
  const bool has_message = systemHasMessage(ERROR_MOD_NOT_FOUND);
#endif
  void *handle = nntrainer::DynamicLibraryLoader::loadLibrary(
    missing_library, RTLD_LAZY | RTLD_LOCAL);
  EXPECT_EQ(handle, nullptr);

  const std::string error = nntrainer::DynamicLibraryLoader::getLastError();
  EXPECT_FALSE(error.empty());
#if defined(_WIN32)
  const std::string code = std::to_string(ERROR_MOD_NOT_FOUND);
  if (has_message) {
    /** the system message, followed by the numeric code for reference */
    const std::string suffix = " (" + code + ")";
    ASSERT_GT(error.size(), suffix.size()) << error;
    EXPECT_EQ(
      error.compare(error.size() - suffix.size(), suffix.size(), suffix), 0)
      << error;
    const std::string text = error.substr(0, error.size() - suffix.size());
    EXPECT_NE(text.find_first_not_of(' '), std::string::npos) << error;
  } else {
    /** no message resources on this system: the bare code */
    EXPECT_EQ(error, code);
  }
#endif
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
#if defined(_WIN32)
  const std::string fixture_arg = "--dl-error-fixture=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.compare(0, fixture_arg.size(), fixture_arg) == 0) {
      dl_error_fixture = arg.substr(fixture_arg.size());
    }
  }
  if (dl_error_fixture.empty()) {
    /** fall back to the module built next to this executable, if present */
    char exe_path[MAX_PATH];
    const DWORD length = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    if (length != 0 && length < MAX_PATH) {
      std::string candidate(exe_path, length);
      candidate.erase(candidate.find_last_of("\\/") + 1);
      candidate += "dl_error_fixture.dll";
      if (GetFileAttributesA(candidate.c_str()) != INVALID_FILE_ATTRIBUTES) {
        dl_error_fixture = candidate;
      }
    }
  }
#endif
  return RUN_ALL_TESTS();
}
