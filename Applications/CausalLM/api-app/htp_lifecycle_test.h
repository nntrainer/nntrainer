// SPDX-License-Identifier: Apache-2.0
/**
 * @author NNTrainer contributors
 * @bug See the project issue tracker.
 */
/** @file htp_lifecycle_test.h
 *  @brief Device integration checks through the public QuickDotAI C API.
 */
#ifndef QUICK_DOT_AI_HTP_LIFECYCLE_TEST_H
#define QUICK_DOT_AI_HTP_LIFECYCLE_TEST_H

#include "quick_dot_ai_api.h"
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

static int run_htp_lifecycle_test(const char *prompt, const char *base,
                                  const char *lib_dir) {
  struct Handle {
    CausalLmHandle value = nullptr;
    ~Handle() { destroyModelHandle(value); }
  } handle;
  const auto check = [](bool ok, const std::string &what) {
    if (!ok)
      throw std::runtime_error(what);
    std::cout << "[HTP-API-TEST] PASS " << what << std::endl;
  };
  const auto load = [&](CausalLmHandle *out) {
    return loadModelHandleByName(CAUSAL_LM_BACKEND_NPU, "gemma4-26b-a4b-htp",
                                 CAUSAL_LM_QUANTIZATION_W4A32, lib_dir, base,
                                 out);
  };
  struct Stream {
    std::mutex mu;
    std::condition_variable cv;
    unsigned callbacks = 0;
    unsigned stop_after = 0;
    bool done = false;
    std::string text;
    static int callback(const char *delta, void *data) {
      auto &s = *static_cast<Stream *>(data);
      std::lock_guard<std::mutex> lock(s.mu);
      s.text += delta ? delta : "";
      ++s.callbacks;
      s.cv.notify_all();
      return s.stop_after && s.callbacks >= s.stop_after;
    }
  };
  try {
    Handle invalid;
    check(loadModelHandleByName(CAUSAL_LM_BACKEND_CPU, "gemma4-26b-a4b-htp",
                                CAUSAL_LM_QUANTIZATION_W4A32, lib_dir, base,
                                &invalid.value) ==
              CAUSAL_LM_ERROR_INVALID_PARAMETER &&
            !invalid.value,
          "reject CPU backend");
    check(load(&handle.value) == CAUSAL_LM_ERROR_NONE, "load");
    check(load(&invalid.value) == CAUSAL_LM_ERROR_MODEL_LOAD_FAILED &&
            !invalid.value,
          "reject second active HTP model");

    const char *trace_env = std::getenv("QDA_HTP_TRACE_PREFIX");
    const std::string trace_prefix = trace_env ? trace_env : "";
    const auto generate = [&](const std::string &name, unsigned callback_stop,
                              bool thread_cancel, bool unload) {
      if (!trace_prefix.empty()) {
        const auto path = trace_prefix + "-" + name + ".json";
#ifdef _WIN32
        _putenv_s("NNTR_HTP_TOKEN_TRACE", path.c_str());
#else
        setenv("NNTR_HTP_TOKEN_TRACE", path.c_str(), 1);
#endif
      }
      Stream stream;
      stream.stop_after = callback_stop;
      ErrorCode cancel_result = CAUSAL_LM_ERROR_UNKNOWN;
      ErrorCode unload_result = CAUSAL_LM_ERROR_UNKNOWN;
      std::thread canceller;
      if (thread_cancel) {
        canceller = std::thread([&] {
          std::unique_lock<std::mutex> lock(stream.mu);
          stream.cv.wait(lock,
                         [&] { return stream.callbacks >= 8 || stream.done; });
          if (stream.done)
            return;
          lock.unlock();
          cancel_result = cancelModelHandle(handle.value);
          if (unload)
            unload_result = unloadModelHandle(handle.value);
        });
      }
      const auto result = runModelHandleStreaming(handle.value, prompt,
                                                  Stream::callback, &stream);
      {
        std::lock_guard<std::mutex> lock(stream.mu);
        stream.done = true;
      }
      stream.cv.notify_all();
      if (canceller.joinable())
        canceller.join();
      check(result == CAUSAL_LM_ERROR_NONE && stream.callbacks != 0,
            name + " generation");
      if (callback_stop)
        check(stream.callbacks == callback_stop, name + " callback stop");
      if (thread_cancel)
        check(cancel_result == CAUSAL_LM_ERROR_NONE, name + " cancel");
      if (unload)
        check(unload_result == CAUSAL_LM_ERROR_NONE, name + " unload");
      std::cout << "[HTP-API-TEST] " << name
                << " callbacks=" << stream.callbacks
                << " bytes=" << stream.text.size() << std::endl;
      return stream.text;
    };
    const auto expected = generate("first", 0, false, false);
    check(generate("repeat", 0, false, false) == expected,
          "repeat output parity");
    generate("callback-cancel", 8, false, false);
    check(generate("after-cancel", 0, false, false) == expected,
          "output parity after callback cancellation");
    generate("thread-cancel", 0, true, false);
    check(generate("after-thread-cancel", 0, false, false) == expected,
          "output parity after thread cancellation");
    generate("unload-active", 0, true, true);
    check(cancelModelHandle(handle.value) == CAUSAL_LM_ERROR_NOT_INITIALIZED,
          "unloaded handle state");
    check(destroyModelHandle(handle.value) == CAUSAL_LM_ERROR_NONE, "destroy");
    handle.value = nullptr;
    for (unsigned i = 0; i < 2; ++i) {
      check(load(&handle.value) == CAUSAL_LM_ERROR_NONE, "reload");
      check(generate("reload-" + std::to_string(i), 0, false, false) ==
              expected,
            "reload output parity");
      check(unloadModelHandle(handle.value) == CAUSAL_LM_ERROR_NONE, "unload");
      check(unloadModelHandle(handle.value) == CAUSAL_LM_ERROR_NONE,
            "unload twice");
      destroyModelHandle(handle.value);
      handle.value = nullptr;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[HTP-API-TEST] FAIL " << e.what() << std::endl;
    return 1;
  }
}
#endif
