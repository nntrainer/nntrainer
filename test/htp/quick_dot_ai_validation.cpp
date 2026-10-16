// SPDX-License-Identifier: Apache-2.0
/**
 * @author NNTrainer contributors
 * @bug See the project issue tracker.
 */
/** @file quick_dot_ai_validation.cpp
 *  @brief Device checks of HTP load failures and CPU handle coexistence.
 *
 * Standalone public C API test; build and invocation are recorded in
 * docs/measurements/pr4416-quickdotai-48a6bbb/README.md.
 */
#include <quick_dot_ai_api.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
struct Handle {
  CausalLmHandle value = nullptr;
  ~Handle() { destroyModelHandle(value); }
};

void check(bool ok, const char *name) {
  if (!ok)
    throw std::runtime_error(name);
  std::cout << "[HTP-VALIDATION] PASS " << name << std::endl;
}

ErrorCode loadHtp(Handle &h, const char *base, const char *lib) {
  return loadModelHandleByName(CAUSAL_LM_BACKEND_NPU, "gemma4-26b-a4b-htp",
                               CAUSAL_LM_QUANTIZATION_W4A32, lib, base,
                               &h.value);
}

std::string runCpu(Handle &h) {
  std::string text;
  const auto callback = [](const char *delta, void *data) {
    *static_cast<std::string *>(data) += delta ? delta : "";
    return 0;
  };
  check(runModelHandleStreaming(h.value, "hello world", callback, &text) ==
          CAUSAL_LM_ERROR_NONE,
        "CPU streaming");
  check(!text.empty(), "CPU output is nonempty");
  return text;
}
} // namespace

int main(int argc, char **argv) {
  // failure <bad-base> <lib-dir>
  // recovery <bad-base> <lib-dir> <good-base>
  // cpu <cpu-base> <lib-dir> <htp-base> <htp-prompt-file>
  if (argc < 4)
    return 2;
  Config options{};
  options.use_chat_template = false;
  setOptions(options);
  try {
    const std::string mode = argv[1];
    if (mode == "failure" || mode == "recovery") {
      Handle failed;
      check(loadHtp(failed, argv[2], argv[3]) ==
                CAUSAL_LM_ERROR_MODEL_LOAD_FAILED &&
              !failed.value,
            "failed HTP load returns no handle");
      if (mode == "recovery") {
        if (argc != 5)
          return 2;
        Handle valid;
        check(loadHtp(valid, argv[4], argv[3]) == CAUSAL_LM_ERROR_NONE,
              "HTP load recovers after failure");
        check(unloadModelHandle(valid.value) == CAUSAL_LM_ERROR_NONE,
              "unload recovered HTP model");
      }
    } else if (mode == "cpu" && argc == 6) {
      const auto loadCpu = [&](Handle &h) {
        check(loadModelHandleByName(CAUSAL_LM_BACKEND_CPU, "gemma4-cpu",
                                    CAUSAL_LM_QUANTIZATION_W32A32, nullptr,
                                    argv[2], &h.value) == CAUSAL_LM_ERROR_NONE,
              "CPU load");
      };
      Handle reference;
      loadCpu(reference);
      const auto expected = runCpu(reference);
      Handle cpu;
      loadCpu(cpu);
      Handle htp;
      check(loadHtp(htp, argv[4], argv[3]) == CAUSAL_LM_ERROR_NONE,
            "HTP load with a live CPU handle");
      std::ifstream file(argv[5]);
      const std::string prompt((std::istreambuf_iterator<char>(file)), {});
      check(!prompt.empty(), "HTP prompt fixture");
      std::string htp_output;
      check(runModelHandleStreaming(
              htp.value, prompt.c_str(),
              [](const char *delta, void *data) {
                *static_cast<std::string *>(data) += delta ? delta : "";
                return 0;
              },
              &htp_output) == CAUSAL_LM_ERROR_NONE &&
              !htp_output.empty(),
            "HTP generation after CPU initialization");
      check(runCpu(cpu) == expected, "CPU output while HTP is loaded");
      check(unloadModelHandle(htp.value) == CAUSAL_LM_ERROR_NONE, "HTP unload");
      Handle after;
      loadCpu(after);
      check(runCpu(after) == expected, "CPU output after HTP release");
      check(unloadModelHandle(cpu.value) == CAUSAL_LM_ERROR_NONE, "CPU unload");
    } else {
      return 2;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[HTP-VALIDATION] FAIL " << e.what() << std::endl;
    return 1;
  }
}
