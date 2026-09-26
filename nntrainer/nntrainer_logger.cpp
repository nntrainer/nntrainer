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
 * @file nntrainer_logger.cpp
 * @date 02 April 2020
 * @brief NNTrainer Logger
 *        This allows to logging nntrainer logs.
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <nntrainer_logger.h>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <system_error>
#include <util_func.h>

namespace nntrainer {

/**
 * @brief     logfile name
 */
const char *const Logger::logfile_name = "log_nntrainer_";
const char *const Logger::logfile_subdir = "logs";
/**
 * @brief     instance for single logger
 */
Logger *Logger::ainstance = nullptr;

/**
 * @brief     mutex for lock
 */
std::mutex Logger::smutex;

Logger &Logger::instance() {
  static Cleanup cleanup;

  std::lock_guard<std::mutex> guard(smutex);
  if (ainstance == nullptr)
    ainstance = new Logger();
  return *ainstance;
}

Logger::Cleanup::~Cleanup() {
  std::lock_guard<std::mutex> guard(Logger::smutex);
  delete Logger::ainstance;
  Logger::ainstance = nullptr;
}

Logger::~Logger() {
  try {
    outputstream.close();
  } catch (...) {
    std::cerr << "Error closing the log file\n";
  }
}

std::string Logger::resolveLogDir() {
  return resolveUserDataDir("NNTR_LOG_DIR", logfile_subdir);
}

bool Logger::openLogFile(const std::string &dir, const std::string &file_name,
                         std::ofstream &out) noexcept {
  try {
    if (dir.empty())
      return false;
    // The error_code overload: the throwing one raises
    // std::filesystem::filesystem_error for a directory that cannot be
    // created, and the logger is first constructed inside whatever call
    // logged first -- model load, typically -- so that exception used to
    // take the whole load down.
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
      return false;
    out.open((std::filesystem::path(dir) / file_name).string(),
             std::ios_base::app);
    if (!out.is_open() || !out.good()) {
      out.close();
      return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

Logger::Logger() : ts_type(NNTRAINER_LOG_TIMESTAMP_SEC) {
  struct tm now;
  getLocaltime(&now);
  std::stringstream ss;
  ss << logfile_name << std::dec << (now.tm_year + 1900) << std::setfill('0')
     << std::setw(2) << (now.tm_mon + 1) << std::setfill('0') << std::setw(2)
     << now.tm_mday << std::setfill('0') << std::setw(2) << now.tm_hour
     << std::setfill('0') << std::setw(2) << now.tm_min << std::setfill('0')
     << std::setw(2) << now.tm_sec << ".out";

  // Fail-soft: logging is diagnostics, and a directory that cannot be written
  // must never cost the caller its model load. Without a log file, warnings
  // and errors go to stderr (see log()) and the rest is dropped.
  const std::string dir = resolveLogDir();
  if (!dir.empty() && !openLogFile(dir, ss.str(), outputstream))
    std::cerr << "nntrainer: cannot write log files under " << dir
              << "; file logging disabled, warnings and errors go to stderr"
              << std::endl;
}

void Logger::log(const std::string &message,
                 const nntrainer_loglevel loglevel) {
  std::lock_guard<std::mutex> guard(smutex);
  std::stringstream ss;

  switch (loglevel) {
  case NNTRAINER_LOG_INFO:
    ss << "[NNTRAINER INFO  ";
    break;
  case NNTRAINER_LOG_WARN:
    ss << "[NNTRAINER WARN  ";
    break;
  case NNTRAINER_LOG_ERROR:
    ss << "[NNTRAINER ERROR ";
    break;
  case NNTRAINER_LOG_DEBUG:
    ss << "[NNTRAINER DEBUG ";
    break;
  default:
    break;
  }

  if (ts_type == NNTRAINER_LOG_TIMESTAMP_MS) {
    static auto start = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch() - start)
                .count();

    ss << "[ " << ms << " ]";
  } else if (ts_type == NNTRAINER_LOG_TIMESTAMP_SEC) {
    struct tm now;
    getLocaltime(&now);

    ss << std::dec << (now.tm_year + 1900) << '-' << std::setfill('0')
       << std::setw(2) << (now.tm_mon + 1) << '-' << std::setfill('0')
       << std::setw(2) << now.tm_mday << ' ' << std::setfill('0')
       << std::setw(2) << now.tm_hour << ':' << std::setfill('0')
       << std::setw(2) << now.tm_min << ':' << std::setfill('0') << std::setw(2)
       << now.tm_sec << ']';
  }

  if (outputstream.is_open())
    outputstream << ss.str() << " " << message << std::endl;
  else if (loglevel >= NNTRAINER_LOG_WARN)
    std::cerr << ss.str() << " " << message << std::endl;
}

} /* namespace nntrainer */

#ifdef __cplusplus
extern "C" {
#endif
void __nntrainer_log_print(nntrainer_loglevel loglevel,
                           const std::string format_str, ...) {
  int final_n, n = ((int)format_str.size()) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(new char[n]);
    std::strncpy(&formatted[0], format_str.c_str(), format_str.size());
    va_start(ap, format_str);
    final_n = vsnprintf(&formatted[0], n, format_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }

  std::string ss = std::string(formatted.get());

#if defined(__LOGGING__)
  nntrainer::Logger::instance().log(ss, loglevel);
#else

#if defined(DEBUG)
  switch (loglevel) {
  case NNTRAINER_LOG_ERROR:
    std::cerr << ss << std::endl;
    break;
  case NNTRAINER_LOG_INFO:
  case NNTRAINER_LOG_WARN:
  case NNTRAINER_LOG_DEBUG:
    std::cout << ss << std::endl;
  default:
    break;
  }
#endif
#endif
}

#ifdef __cplusplus
}
#endif
