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
 * @file nntrainer_logger.h
 * @date 02 April 2020
 * @brief NNTrainer Logger
 *        This allows to logging nntrainer logs.
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNTRAINER_LOGGER_H___
#define __NNTRAINER_LOGGER_H___
#ifdef __cplusplus

#include <fstream>
#include <mutex>
#include <string>

/**
 * @brief     Log Level of NNtrainer
 *            0. informations
 *            1. warnings
 *            2. errors
 *            3. debugging informations
 */
typedef enum {
  NNTRAINER_LOG_DEBUG = 0,
  NNTRAINER_LOG_INFO,
  NNTRAINER_LOG_WARN,
  NNTRAINER_LOG_ERROR
} nntrainer_loglevel;

/**
 * @brief     Log TimeStamp type
 */
typedef enum {
  NNTRAINER_LOG_TIMESTAMP_SEC = 0,
  NNTRAINER_LOG_TIMESTAMP_MS,
} nntrainer_log_timestamp;

namespace nntrainer {

/**
 * @class   NNTrainer Logger Class
 * @brief   Class for Logging. This is alternatives when there is no logging
 * system. For the tizen, we are going to use dlog and it is android_log for
 * android.
 */
class Logger {
public:
  /**
   * @brief     Logging Instance Function. Get a lock and create Logger if it
   * is null;
   */
  static Logger &instance();

  /**
   * @brief     Logging member function for logging messages.
   */
  void log(const std::string &message,
           const nntrainer_loglevel loglevel = NNTRAINER_LOG_INFO);

  /**
   * @brief     Set timestamp type
   */
  void setTimeStampType(nntrainer_log_timestamp type) { ts_type = type; }

  /**
   * @brief     Directory the log file is written to.
   *
   * NNTR_LOG_DIR names it outright; set to the empty string or "off" it
   * disables file logging. Otherwise it is "logs" under the per-user base
   * directory (see resolveUserDataDir()), never the working directory.
   *
   * @return    the directory, or "" when file logging is disabled
   */
  static std::string resolveLogDir();

  /**
   * @brief     Create @a dir if needed and open @a file_name in it for
   *            appending. Never throws.
   *
   * @param     dir directory to write into; "" fails without touching disk
   * @param     file_name log file name inside @a dir
   * @param     out stream to open; left closed on failure
   * @return    true when @a out is open for writing
   */
  static bool openLogFile(const std::string &dir, const std::string &file_name,
                          std::ofstream &out) noexcept;

protected:
  /**
   * @brief     Logging instance
   */
  static Logger *ainstance;
  /**
   * @brief     Log file name
   */
  static const char *const logfile_name;
  /**
   * @brief     Log directory name under the per-user base directory.
   */
  static const char *const logfile_subdir;
  /**
   * @brief     output stream; closed when file logging is disabled or the
   *            log directory could not be written
   */
  std::ofstream outputstream;

  /**
   * @class     Class to make sure the single logger
   * @brief     sure the single logger
   */
  friend class Cleanup;

  /**
   * @brief     Class to make sure the single logger
   */
  class Cleanup {
  public:
    ~Cleanup();
  };

private:
  /**
   * @brief     Constructor
   */
  Logger();

  /**
   * @brief     Destructor
   */
  virtual ~Logger();

  /**
   * @brief     Copy Constructor
   */
  Logger(const Logger &);

  /**
   * @brief     Assignment operator
   */
  Logger &operator=(const Logger &);

  /**
   * @brief     mutex for logging
   */
  static std::mutex smutex;

  /**
   * @brief     log timestamp type
   */
  nntrainer_log_timestamp ts_type;
};
} /* namespace nntrainer */

extern "C" {

#endif /* __cplusplus */

/**
 * @brief     Interface function for C
 */
void __nntrainer_log_print(nntrainer_loglevel loglevel,
                           const std::string format, ...);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __NNTRAINER_LOGGER_H___ */
