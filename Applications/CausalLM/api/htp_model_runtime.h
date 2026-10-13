// SPDX-License-Identifier: Apache-2.0
/**
 * @author NNTrainer contributors
 * @bug See the project issue tracker.
 */
/** @file htp_model_runtime.h
 *  @brief Exclusive native HTP runtime ownership for a QuickDotAI handle.
 */
#ifndef QUICK_DOT_AI_HTP_MODEL_RUNTIME_H
#define QUICK_DOT_AI_HTP_MODEL_RUNTIME_H

#ifdef ENABLE_HEXKL
#include <cstdlib>
#include <filesystem>
#include <htp_backend.h>
#include <lfm2_moe_layer.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quick_dot_ai {

class HtpModelRuntime {
public:
  explicit HtpModelRuntime(const char *native_lib_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (owned_)
      throw std::runtime_error(
        "Only one active QuickDotAI HTP model is supported");
    if (!native_lib_dir || !*native_lib_dir)
      throw std::invalid_argument(
        "HTP requires native_lib_dir with the v81 DSP skel");
    const auto dir = std::filesystem::canonical(native_lib_dir);
    if (!std::filesystem::is_regular_file(dir / "libnntr_hvx_skel.so"))
      throw std::invalid_argument(
        "HTP native_lib_dir is missing libnntr_hvx_skel.so");

    // The HTP runtime configuration is built into the library
    // (nntrainer/tensor/htp_backend/htp_fixed_opts.h); only the DSP library
    // path is set here, and the weight page-cache policy
    // (NNTR_HTP_PAGE_CACHE) is left to the environment.
    const std::vector<std::pair<std::string, std::string>> options = {
      {"ADSP_LIBRARY_PATH", dir.string()},
    };
    if (!options_.empty() && options_ != options)
      throw std::invalid_argument(
        "HTP runtime settings are fixed for this process");
    // Validate all values before changing any. Several PR options are read
    // once by function-local statics and cannot be changed between requests.
    for (const auto &option : options) {
      const char *old = std::getenv(option.first.c_str());
      if (old && option.second != old)
        throw std::invalid_argument("Conflicting HTP runtime option: " +
                                    option.first);
    }
    for (const auto &option : options)
      if (setenv(option.first.c_str(), option.second.c_str(), 1) != 0)
        throw std::runtime_error("Could not initialize HTP option: " +
                                 option.first);
    options_ = options;
    try {
      nntrainer::htp_prepare_runtime();
    } catch (...) {
      nntrainer::htp_release_runtime();
      throw;
    }
    owned_ = true;
  }

  ~HtpModelRuntime() {
    std::lock_guard<std::mutex> lock(mutex_);
    nntrainer::htp_release_runtime();
    causallm::resetLfm2HtpExpertPool();
    owned_ = false;
  }

  HtpModelRuntime(const HtpModelRuntime &) = delete;
  HtpModelRuntime &operator=(const HtpModelRuntime &) = delete;

private:
  inline static std::mutex mutex_;
  inline static bool owned_ = false;
  inline static std::vector<std::pair<std::string, std::string>> options_;
};
} // namespace quick_dot_ai
#endif
#endif
