/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file        unittest_util_func.cpp
 * @date        10 April 2020
 * @brief       Unit test for util_func.cpp.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <nntrainer_logger.h>
#include <nntrainer_test_util.h>
#include <util_func.h>

TEST(nntrainer_util_func, sqrtFloat_01_p) {
  float x = 9871.0;
  float sx = nntrainer::sqrtFloat(x);

  EXPECT_NEAR(sx * sx, x, tolerance * 10);
}

TEST(nntrainer_util_func, logFloat_01_p) {
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (width) + k + 1);

  nntrainer::Tensor Results = input.apply<float>(nntrainer::logFloat<float>);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], (float)log(indata[i]), tolerance);
  }
}

// TEST(nntrainer_util_func, rotate_180_p) {
//   nntrainer::TensorDim dim(1, 1, 2, 3);

//   float data[6] = {1, 2, 3, 4, 5, 6};
//   float rotated_data[6] = {6, 5, 4, 3, 2, 1};

//   nntrainer::Tensor tensor1(dim, data);
//   nntrainer::Tensor tensor2 = nntrainer::rotate_180(tensor1);

//   for (unsigned int i = 0, cnt = 0; i < dim.batch(); ++i)
//     for (unsigned int j = 0; j < dim.channel(); ++j)
//       for (unsigned int k = 0; k < dim.height(); ++k)
//         for (unsigned int l = 0; l < dim.width(); ++l)
//           EXPECT_EQ(tensor2.getValue(i, j, k, l), rotated_data[cnt++]);
// }

TEST(nntrainer_util_func, checkedRead_n) {
  std::ifstream file("not existing file");
  char array[5];

  EXPECT_THROW(nntrainer::checkedRead(file, array, 5), std::runtime_error);
}

TEST(nntrainer_util_func, checkedWrite_n) {
  std::ofstream file("!@/not good file");
  char array[5] = "abcd";

  EXPECT_THROW(nntrainer::checkedWrite(file, array, 5), std::runtime_error);
}

TEST(nntrainer_util_func, readString_n) {
  std::ifstream file("not existing file");

  EXPECT_THROW(nntrainer::readString(file), std::runtime_error);
}

TEST(nntrainer_util_func, writeString_n) {
  std::ofstream file("!@/not good file");
  std::string str = "abcd";

  EXPECT_THROW(nntrainer::writeString(file, str), std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_no_error_p) {
  EXPECT_NO_THROW(nntrainer::throw_status(ML_ERROR_NONE));
}

TEST(nntrainer_util_func, throw_status_invalid_argument_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_INVALID_PARAMETER),
               std::invalid_argument);
}

TEST(nntrainer_util_func, throw_status_try_again_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_TRY_AGAIN), std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_not_supported_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_NOT_SUPPORTED),
               std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_out_of_memory_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_OUT_OF_MEMORY), std::bad_alloc);
}

TEST(nntrainer_util_func, throw_status_timed_out_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_TIMED_OUT), std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_permission_denied_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_PERMISSION_DENIED),
               std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_unknown_error_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_UNKNOWN), std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_default_n) {
  EXPECT_THROW(nntrainer::throw_status(-12345), std::runtime_error);
}

namespace {

/**
 * @brief Set or unset one environment variable for the lifetime of a scope,
 *        restoring the previous value afterwards.
 */
class ScopedEnv {
public:
  /**
   * @brief Set @a name to @a value, or unset it when @a value is nullptr.
   */
  ScopedEnv(const char *name, const char *value) : name_(name) {
    const char *old = std::getenv(name);
    had_ = (old != nullptr);
    if (had_)
      old_ = old;
    apply(value);
  }

  /**
   * @brief Restore the previous value.
   */
  ~ScopedEnv() { apply(had_ ? old_.c_str() : nullptr); }

private:
  void apply(const char *value) {
#if defined(_WIN32)
    _putenv_s(name_.c_str(), value != nullptr ? value : "");
#else
    if (value != nullptr)
      setenv(name_.c_str(), value, 1);
    else
      unsetenv(name_.c_str());
#endif
  }

  std::string name_;
  std::string old_;
  bool had_ = false;
};

/**
 * @brief A fresh temporary directory, removed with everything under it.
 */
class ScopedTempDir {
public:
  ScopedTempDir() {
    static int seq = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("nntr_userdir_test_" + std::to_string(::time(nullptr)) + "_" +
             std::to_string(seq++));
    std::filesystem::create_directories(path_);
  }
  ~ScopedTempDir() {
    std::error_code ec;
    // make anything a test made read-only removable again
    for (auto it = std::filesystem::recursive_directory_iterator(path_, ec);
         !ec && it != std::filesystem::recursive_directory_iterator();
         it.increment(ec))
      std::filesystem::permissions(it->path(),
                                   std::filesystem::perms::owner_all,
                                   std::filesystem::perm_options::add, ec);
    std::filesystem::remove_all(path_, ec);
  }
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

/**
 * @brief Make @a p a regular file, so that nothing can be created under it --
 *        an "unwritable directory" that does not depend on who runs the test.
 */
void makeRegularFile(const std::filesystem::path &p) {
  std::ofstream f(p);
  f << "not a directory";
}

} // namespace

/**
 * @brief The logger must not throw, nor create anything, when its directory
 *        cannot be created (the Windows System32 crash).
 */
TEST(nntrainer_user_dir, logger_open_under_regular_file_n) {
  ScopedTempDir tmp;
  const auto blocker = tmp.path() / "blocker";
  makeRegularFile(blocker);
  const auto dir = blocker / "logs";

  std::ofstream out;
  bool ok = true;
  EXPECT_NO_THROW(
    ok = nntrainer::Logger::openLogFile(dir.string(), "log_test.out", out));
  EXPECT_FALSE(ok);
  EXPECT_FALSE(out.is_open());
  EXPECT_FALSE(std::filesystem::exists(dir));
}

/**
 * @brief Same, with a read-only parent directory. Root ignores permission
 *        bits, so this one can only run unprivileged.
 */
TEST(nntrainer_user_dir, logger_open_under_readonly_dir_n) {
#if defined(_WIN32)
  GTEST_SKIP() << "POSIX permission bits only";
#else
  if (::geteuid() == 0)
    GTEST_SKIP() << "root bypasses directory permissions";
  ScopedTempDir tmp;
  const auto ro = tmp.path() / "ro";
  std::filesystem::create_directories(ro);
  std::filesystem::permissions(
    ro, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec,
    std::filesystem::perm_options::replace);
  const auto dir = ro / "logs";

  std::ofstream out;
  bool ok = true;
  EXPECT_NO_THROW(
    ok = nntrainer::Logger::openLogFile(dir.string(), "log_test.out", out));
  EXPECT_FALSE(ok);
  EXPECT_FALSE(out.is_open());
  EXPECT_FALSE(std::filesystem::exists(dir));
#endif
}

/**
 * @brief An empty directory means "disabled": nothing is touched.
 */
TEST(nntrainer_user_dir, logger_open_empty_dir_n) {
  std::ofstream out;
  bool ok = true;
  EXPECT_NO_THROW(ok = nntrainer::Logger::openLogFile("", "log_test.out", out));
  EXPECT_FALSE(ok);
  EXPECT_FALSE(out.is_open());
}

/**
 * @brief A writable directory is created and the log file opened in it.
 */
TEST(nntrainer_user_dir, logger_open_writable_p) {
  ScopedTempDir tmp;
  const auto dir = tmp.path() / "a" / "logs";
  std::ofstream out;
  EXPECT_TRUE(
    nntrainer::Logger::openLogFile(dir.string(), "log_test.out", out));
  EXPECT_TRUE(out.is_open());
  out.close();
  EXPECT_TRUE(std::filesystem::is_regular_file(dir / "log_test.out"));
}

/**
 * @brief NNTR_LOG_DIR set to "", "off" or "OFF" disables file logging; any
 *        other value is taken exactly.
 */
TEST(nntrainer_user_dir, log_dir_override_p) {
  {
    ScopedEnv e("NNTR_LOG_DIR", "off");
    EXPECT_EQ(nntrainer::Logger::resolveLogDir(), "");
  }
  {
    ScopedEnv e("NNTR_LOG_DIR", "OFF");
    EXPECT_EQ(nntrainer::Logger::resolveLogDir(), "");
  }
#if !defined(_WIN32) // the Windows CRT cannot hold an empty variable
  {
    ScopedEnv e("NNTR_LOG_DIR", "");
    EXPECT_EQ(nntrainer::Logger::resolveLogDir(), "");
  }
#endif
  ScopedTempDir tmp;
  const std::string explicit_dir = (tmp.path() / "mylogs").string();
  {
    ScopedEnv e("NNTR_LOG_DIR", explicit_dir.c_str());
    EXPECT_EQ(nntrainer::Logger::resolveLogDir(), explicit_dir);
  }
}

/**
 * @brief NNTR_KERNEL_CACHE_DIR set to "", "off" or "Off" disables the kernel
 *        cache; any other value is taken exactly.
 */
TEST(nntrainer_user_dir, kernel_cache_override_p) {
  {
    ScopedEnv e("NNTR_KERNEL_CACHE_DIR", "off");
    EXPECT_EQ(nntrainer::resolveUserDataDir("NNTR_KERNEL_CACHE_DIR",
                                            "nntrainer_opencl_kernels"),
              "");
  }
  {
    ScopedEnv e("NNTR_KERNEL_CACHE_DIR", "Off");
    EXPECT_EQ(nntrainer::resolveUserDataDir("NNTR_KERNEL_CACHE_DIR",
                                            "nntrainer_opencl_kernels"),
              "");
  }
#if !defined(_WIN32)
  {
    ScopedEnv e("NNTR_KERNEL_CACHE_DIR", "");
    EXPECT_EQ(nntrainer::resolveUserDataDir("NNTR_KERNEL_CACHE_DIR",
                                            "nntrainer_opencl_kernels"),
              "");
  }
#endif
  ScopedTempDir tmp;
  const std::string explicit_dir = (tmp.path() / "kc").string();
  {
    ScopedEnv e("NNTR_KERNEL_CACHE_DIR", explicit_dir.c_str());
    EXPECT_EQ(nntrainer::resolveUserDataDir("NNTR_KERNEL_CACHE_DIR",
                                            "nntrainer_opencl_kernels"),
              explicit_dir);
  }
}

/**
 * @brief An absolute configured path is honoured exactly; an empty one
 *        disables.
 */
TEST(nntrainer_user_dir, configured_absolute_or_empty_p) {
  ScopedTempDir tmp;
  const std::string abs_dir = (tmp.path() / "abs_kernels").string();
  EXPECT_EQ(nntrainer::resolveUserDataDir(nullptr, abs_dir), abs_dir);
  EXPECT_EQ(nntrainer::resolveUserDataDir(nullptr, ""), "");
}

#if !defined(__ANDROID__)
/**
 * @brief A relative configured path resolves to an absolute path under the
 *        per-user directory -- not the working directory, even when the
 *        working directory cannot be written.
 */
TEST(nntrainer_user_dir, relative_resolves_under_user_dir_unwritable_cwd_p) {
  ScopedTempDir tmp;
  const auto user = tmp.path() / "user";
  std::filesystem::create_directories(user);

  // stand in a directory nothing can be created under
  const auto ro = tmp.path() / "ro_cwd";
  std::filesystem::create_directories(ro);
  std::filesystem::permissions(
    ro, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec,
    std::filesystem::perm_options::replace);
  const auto old_cwd = std::filesystem::current_path();
  std::filesystem::current_path(ro);

  std::string kc, log;
  {
    ScopedEnv k("NNTR_KERNEL_CACHE_DIR", nullptr);
    ScopedEnv l("NNTR_LOG_DIR", nullptr);
#if defined(_WIN32)
    ScopedEnv a("LOCALAPPDATA", user.string().c_str());
#else
    ScopedEnv x("XDG_CACHE_HOME", nullptr);
    ScopedEnv h("HOME", user.string().c_str());
#endif
    kc = nntrainer::resolveUserDataDir("NNTR_KERNEL_CACHE_DIR",
                                       "nntrainer_opencl_kernels");
    log = nntrainer::Logger::resolveLogDir();
  }
  std::filesystem::current_path(old_cwd);

#if defined(_WIN32)
  const auto base = user;
#else
  const auto base = user / ".cache";
#endif
  EXPECT_TRUE(std::filesystem::path(kc).is_absolute());
  EXPECT_EQ(kc, (base / "nntrainer" / "nntrainer_opencl_kernels").string());
  EXPECT_EQ(log, (base / "nntrainer" / "logs").string());
  EXPECT_FALSE(std::filesystem::exists(ro / "nntrainer_opencl_kernels"));
  EXPECT_FALSE(std::filesystem::exists(ro / "logs"));
}

/**
 * @brief The first per-user variable that is set wins.
 */
TEST(nntrainer_user_dir, user_base_precedence_p) {
#if defined(_WIN32)
  {
    ScopedEnv a("LOCALAPPDATA", "C:\\Users\\u\\AppData\\Local");
    ScopedEnv b("APPDATA", "C:\\Users\\u\\AppData\\Roaming");
    EXPECT_EQ(nntrainer::getUserCacheBaseDir(), "C:\\Users\\u\\AppData\\Local");
  }
  {
    ScopedEnv a("LOCALAPPDATA", nullptr);
    ScopedEnv b("APPDATA", "C:\\Users\\u\\AppData\\Roaming");
    EXPECT_EQ(nntrainer::getUserCacheBaseDir(),
              "C:\\Users\\u\\AppData\\Roaming");
  }
  {
    ScopedEnv a("LOCALAPPDATA", nullptr);
    ScopedEnv b("APPDATA", nullptr);
    ScopedEnv c("TEMP", "C:\\Users\\u\\AppData\\Local\\Temp");
    EXPECT_EQ(nntrainer::getUserCacheBaseDir(),
              "C:\\Users\\u\\AppData\\Local\\Temp");
  }
#else
  {
    ScopedEnv x("XDG_CACHE_HOME", "/x/cache");
    ScopedEnv h("HOME", "/h");
    EXPECT_EQ(nntrainer::getUserCacheBaseDir(), "/x/cache");
  }
  {
    ScopedEnv x("XDG_CACHE_HOME", nullptr);
    ScopedEnv h("HOME", "/h");
    EXPECT_EQ(nntrainer::getUserCacheBaseDir(), "/h/.cache");
  }
#endif
}

/**
 * @brief With no per-user directory at all, a relative configured path
 *        resolves to "" (disabled) -- never to the working directory.
 */
TEST(nntrainer_user_dir, no_user_dir_disables_n) {
#if defined(_WIN32)
  ScopedEnv a("LOCALAPPDATA", nullptr);
  ScopedEnv b("APPDATA", nullptr);
  ScopedEnv c("TEMP", nullptr);
#else
  ScopedEnv x("XDG_CACHE_HOME", nullptr);
  ScopedEnv h("HOME", nullptr);
#endif
  ScopedEnv k("NNTR_KERNEL_CACHE_DIR", nullptr);
  ScopedEnv l("NNTR_LOG_DIR", nullptr);
  EXPECT_EQ(nntrainer::getUserCacheBaseDir(), "");
  EXPECT_EQ(nntrainer::resolveUserDataDir("NNTR_KERNEL_CACHE_DIR",
                                          "nntrainer_opencl_kernels"),
            "");
  EXPECT_EQ(nntrainer::Logger::resolveLogDir(), "");
}
#endif // !__ANDROID__

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    ml_loge("Failed to init gtest\n");
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    ml_loge("Failed to run test.\n");
  }

  return result;
}
