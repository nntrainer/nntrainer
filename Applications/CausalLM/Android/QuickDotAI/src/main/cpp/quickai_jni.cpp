// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quickai_jni.cpp
 * @brief   JNI shim that forwards calls from Kotlin's
 *          com.example.quickdotai.NativeCausalLm object to the
 *          handle-based C entry points declared in
 *          Applications/CausalLM/api/quick_dot_ai_api.h.
 * @author  junbong.yu <junbong.yu@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This file contains no business logic — only JNI marshalling:
 *   jstring   <-> const char*
 *   jlong     <-> CausalLmHandle
 *   ErrorCode +  struct PerformanceMetrics -> Kotlin data classes.
 *
 * Higher-level concerns (per-model threading, FIFO queue, Gemma4
 * routing) live one level up in NativeQuickDotAI.kt and in the host
 * app's worker (for QuickAIService that is ModelWorker / ModelRegistry;
 * SampleTestAPP drives NativeQuickDotAI directly from its own
 * background dispatcher).
 */

#include <algorithm>
#include <android/log.h>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <jni.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "quick_dot_ai_api.h"

#define LOG_TAG "quickai_jni"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// ---------------------------------------------------------------------------
// Cached global references to the Kotlin result data-classes so we can
// construct them back in JNI. Looked up once in JNI_OnLoad.
// ---------------------------------------------------------------------------

struct JniCache {
  jclass loadResultCls = nullptr; // NativeCausalLm$LoadResult
  jmethodID loadResultCtor = nullptr;

  jclass runResultCls = nullptr; // NativeCausalLm$RunResult
  jmethodID runResultCtor = nullptr;

  jclass metricsResultCls = nullptr; // NativeCausalLm$MetricsResult
  jmethodID metricsResultCtor = nullptr;
};

JniCache g_cache;

jclass find_global(JNIEnv *env, const char *name) {
  jclass local = env->FindClass(name);
  if (local == nullptr) {
    LOGE("FindClass failed: %s", name);
    return nullptr;
  }
  auto *global = reinterpret_cast<jclass>(env->NewGlobalRef(local));
  env->DeleteLocalRef(local);
  return global;
}

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void * /*reserved*/) {
  JNIEnv *env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK ||
      env == nullptr) {
    return JNI_ERR;
  }

  g_cache.loadResultCls =
    find_global(env, "com/example/quickdotai/NativeCausalLm$LoadResult");
  if (g_cache.loadResultCls != nullptr) {
    g_cache.loadResultCtor =
      env->GetMethodID(g_cache.loadResultCls, "<init>", "(IJ)V");
  }

  g_cache.runResultCls =
    find_global(env, "com/example/quickdotai/NativeCausalLm$RunResult");
  if (g_cache.runResultCls != nullptr) {
    g_cache.runResultCtor = env->GetMethodID(g_cache.runResultCls, "<init>",
                                             "(ILjava/lang/String;)V");
  }

  g_cache.metricsResultCls =
    find_global(env, "com/example/quickdotai/NativeCausalLm$MetricsResult");
  if (g_cache.metricsResultCls != nullptr) {
    g_cache.metricsResultCtor =
      env->GetMethodID(g_cache.metricsResultCls, "<init>", "(IIDIDDDJ)V");
  }

  return JNI_VERSION_1_6;
}

// ---------------------------------------------------------------------------
// setOptions
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_setOptionsNative(
  JNIEnv * /*env*/, jobject /*thiz*/, jboolean useChatTemplate,
  jboolean debugMode, jboolean verbose) {
  Config cfg;
  cfg.use_chat_template = (useChatTemplate == JNI_TRUE);
  cfg.debug_mode = (debugMode == JNI_TRUE);
  cfg.verbose = (verbose == JNI_TRUE);
  return static_cast<jint>(setOptions(cfg));
}

// ---------------------------------------------------------------------------
// chdir
//
// The C API in quick_dot_ai_api.cpp hardcodes the model discovery prefix to
// the relative path "./models/<model>-<quant>" (see resolve_model_path()),
// which ties model lookup to the process's current working directory.
// Android apps start with cwd="/" so the only way to point the loader at
// an app-owned directory is to chdir(2) the process before calling
// loadModelHandle. This helper exposes that chdir to Kotlin as a thin
// wrapper — returning 0 on success or the POSIX errno on failure, which
// NativeQuickDotAI surfaces as CAUSAL_LM_ERROR_MODEL_LOAD_FAILED.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_chdirNative(JNIEnv *env,
                                                       jobject /*thiz*/,
                                                       jstring pathJ) {
  if (pathJ == nullptr) {
    return EINVAL;
  }
  const char *path = env->GetStringUTFChars(pathJ, nullptr);
  if (path == nullptr) {
    return ENOMEM;
  }
  int rc = chdir(path);
  int err = (rc == 0) ? 0 : errno;
  env->ReleaseStringUTFChars(pathJ, path);
  return static_cast<jint>(err);
}

// ---------------------------------------------------------------------------
// loadModelHandle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jobject JNICALL
Java_com_example_quickdotai_NativeCausalLm_loadModelHandleNative(
  JNIEnv *env, jobject /*thiz*/, jint backendOrdinal, jint modelOrdinal,
  jint quantOrdinal, jstring nativeLibDirJ, jstring modelBasePathJ,
  jstring htpBackendConfigPathJ) {
  const char *native_lib_dir = nullptr;
  if (nativeLibDirJ != nullptr) {
    native_lib_dir = env->GetStringUTFChars(nativeLibDirJ, nullptr);
  }

  const char *model_base_path = nullptr;
  if (modelBasePathJ != nullptr) {
    model_base_path = env->GetStringUTFChars(modelBasePathJ, nullptr);
  }

  const char *htp_backend_config_path = nullptr;
  if (htpBackendConfigPathJ != nullptr) {
    htp_backend_config_path =
      env->GetStringUTFChars(htpBackendConfigPathJ, nullptr);
  }

  const char *previous_htp_backend_config_path =
    getenv("QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH");
  const bool had_previous_htp_backend_config_path =
    previous_htp_backend_config_path != nullptr;
  std::string previous_htp_backend_config_path_value;
  if (had_previous_htp_backend_config_path) {
    previous_htp_backend_config_path_value = previous_htp_backend_config_path;
  }
  const bool has_htp_backend_config_path =
    htp_backend_config_path != nullptr && htp_backend_config_path[0] != '\0';
  if (has_htp_backend_config_path) {
    setenv("QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH", htp_backend_config_path,
           1);
  }

  CausalLmHandle handle = nullptr;
  ErrorCode ec =
    loadModelHandle(static_cast<BackendType>(backendOrdinal),
                    static_cast<ModelType>(modelOrdinal),
                    static_cast<ModelQuantizationType>(quantOrdinal),
                    native_lib_dir, model_base_path, &handle);

  if (has_htp_backend_config_path) {
    if (had_previous_htp_backend_config_path) {
      setenv("QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH",
             previous_htp_backend_config_path_value.c_str(), 1);
    } else {
      unsetenv("QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH");
    }
  }

  if (native_lib_dir != nullptr && nativeLibDirJ != nullptr) {
    env->ReleaseStringUTFChars(nativeLibDirJ, native_lib_dir);
  }
  if (model_base_path != nullptr && modelBasePathJ != nullptr) {
    env->ReleaseStringUTFChars(modelBasePathJ, model_base_path);
  }
  if (htp_backend_config_path != nullptr && htpBackendConfigPathJ != nullptr) {
    env->ReleaseStringUTFChars(htpBackendConfigPathJ, htp_backend_config_path);
  }

  if (g_cache.loadResultCls == nullptr || g_cache.loadResultCtor == nullptr) {
    return nullptr;
  }
  return env->NewObject(g_cache.loadResultCls, g_cache.loadResultCtor,
                        static_cast<jint>(ec), reinterpret_cast<jlong>(handle));
}

// ---- loadModelHandleByName (T4 string-id path) ----------------------------
extern "C" JNIEXPORT jlong JNICALL
Java_com_example_quickdotai_NativeCausalLm_loadModelHandleByNameNative(
  JNIEnv *env, jobject /*thiz*/, jint backend, jstring modelIdJ, jint quant,
  jstring nativeLibDirJ, jstring modelBasePathJ) {
  const char *id = env->GetStringUTFChars(modelIdJ, nullptr);
  const char *nld =
    nativeLibDirJ ? env->GetStringUTFChars(nativeLibDirJ, nullptr) : nullptr;
  const char *mbp =
    modelBasePathJ ? env->GetStringUTFChars(modelBasePathJ, nullptr) : nullptr;

  CausalLmHandle h = nullptr;
  ErrorCode ec = loadModelHandleByName(
    static_cast<BackendType>(backend), id,
    static_cast<ModelQuantizationType>(quant), nld, mbp, &h);

  env->ReleaseStringUTFChars(modelIdJ, id);
  if (nld)
    env->ReleaseStringUTFChars(nativeLibDirJ, nld);
  if (mbp)
    env->ReleaseStringUTFChars(modelBasePathJ, mbp);

  return (ec == CAUSAL_LM_ERROR_NONE)
           ? static_cast<jlong>(reinterpret_cast<uintptr_t>(h))
           : 0L;
}

// ---- nativeQueryCatalog ---------------------------------------------------
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_quickdotai_NativeCausalLm_nativeQueryCatalog(
  JNIEnv *env, jobject /*thiz*/) {
  return env->NewStringUTF(getModelCatalogJson());
}

// ---- encodeModelHandle (embedding vector) ---------------------------------
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_quickdotai_NativeCausalLm_encodeModelHandleNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jstring textJ) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  if (handle == nullptr || textJ == nullptr) {
    return nullptr;
  }
  const char *text = env->GetStringUTFChars(textJ, nullptr);

  float *vec = nullptr;
  int dim = 0;
  ErrorCode ec = encodeModelHandle(handle, text, &vec, &dim);

  env->ReleaseStringUTFChars(textJ, text);

  if (ec != CAUSAL_LM_ERROR_NONE || vec == nullptr || dim <= 0) {
    if (vec != nullptr) {
      freeEmbedding(vec);
    }
    return nullptr; // null signals failure to the Kotlin layer
  }

  jfloatArray arr = env->NewFloatArray(dim);
  if (arr != nullptr) {
    env->SetFloatArrayRegion(arr, 0, dim, vec);
  }
  freeEmbedding(vec);
  return arr;
}

// ---------------------------------------------------------------------------
// runModelHandleStreaming
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jobject JNICALL
Java_com_example_quickdotai_NativeCausalLm_getPerformanceMetricsHandleNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);

  PerformanceMetrics m{};
  ErrorCode ec = getPerformanceMetricsHandle(handle, &m);

  if (g_cache.metricsResultCls == nullptr ||
      g_cache.metricsResultCtor == nullptr) {
    return nullptr;
  }
  return env->NewObject(g_cache.metricsResultCls, g_cache.metricsResultCtor,
                        static_cast<jint>(ec),
                        static_cast<jint>(m.prefill_tokens),
                        static_cast<jdouble>(m.prefill_duration_ms),
                        static_cast<jint>(m.generation_tokens),
                        static_cast<jdouble>(m.generation_duration_ms),
                        static_cast<jdouble>(m.total_duration_ms),
                        static_cast<jdouble>(m.initialization_duration_ms),
                        static_cast<jlong>(m.peak_memory_kb));
}

// ---------------------------------------------------------------------------
// unloadModelHandle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_unloadModelHandleNative(
  JNIEnv * /*env*/, jobject /*thiz*/, jlong handleJlong) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  return static_cast<jint>(unloadModelHandle(handle));
}

// ---------------------------------------------------------------------------
// destroyModelHandle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_destroyModelHandleNative(
  JNIEnv * /*env*/, jobject /*thiz*/, jlong handleJlong) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  return static_cast<jint>(destroyModelHandle(handle));
}

// ---------------------------------------------------------------------------
// configureSpeculativeDecoding
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_configureSpeculativeDecodingNative(
  JNIEnv * /*env*/, jobject /*thiz*/, jlong handleJlong, jboolean use_sd) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  return static_cast<jint>(configureSpeculativeDecoding(handle, (bool)use_sd));
}

// ---------------------------------------------------------------------------
// cancelModelHandle
//
// Requests cancellation of an in-progress streaming run. Thread-safe:
// can be called from any thread (e.g., UI cancel button handler).
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_cancelModelHandleNative(
  JNIEnv * /*env*/, jobject /*thiz*/, jlong handleJlong) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                      "cancelModelHandleNative: handle=%p", (void *)handle);
  auto result = cancelModelHandle(handle);
  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                      "cancelModelHandleNative: returned %d", result);
  return static_cast<jint>(result);
}

// ---------------------------------------------------------------------------
// runModelHandleStreaming
//
// Forwards deltas from the native quick_dot_ai_api streaming callback to a
// Kotlin NativeStreamListener.onDelta(String). See AsyncAndStreaming.md §4
// at the repo root for the design rationale — in particular, the
// callback fires on the SAME thread that invoked this JNI entry point
// (the ModelWorker thread), which means we do NOT need AttachCurrentThread:
// the JNIEnv* captured here is still valid throughout every callback.
// ---------------------------------------------------------------------------
namespace {
struct StreamCtx {
  JNIEnv *env;
  jobject listener;  // local ref owned by the JNI entry frame
  jmethodID onDelta; // Ljava/lang/String;)V
};

inline bool is_valid_utf8(const char *s) {
  if (s == nullptr)
    return true;
  while (*s) {
    unsigned char c = static_cast<unsigned char>(*s);
    int follow = 0;
    if (c < 0x80) {
      ++s;
      continue;
    } else if ((c & 0xE0) == 0xC0)
      follow = 1;
    else if ((c & 0xF0) == 0xE0)
      follow = 2;
    else if ((c & 0xF8) == 0xF0)
      follow = 3;
    else
      return false; // invalid lead byte

    for (int i = 0; i < follow; ++i) {
      ++s;
      if (*s == '\0' || (static_cast<unsigned char>(*s) & 0xC0) != 0x80)
        return false; // missing or invalid continuation byte
    }
    ++s;
  }
  return true;
}

int stream_trampoline(const char *delta, void *user_data) {
  auto *ctx = static_cast<StreamCtx *>(user_data);
  if (ctx == nullptr || ctx->env == nullptr || ctx->listener == nullptr ||
      ctx->onDelta == nullptr) {
    return 1; // cancel
  }
  const char *deltaStr = delta != nullptr ? delta : "";
  if (!is_valid_utf8(deltaStr)) {
    LOGE("Invalid UTF-8 delta skipped (lead=0x%02X)",
         static_cast<unsigned char>(deltaStr[0]));
    return 0; // skip this delta, keep streaming
  }
  jstring js = ctx->env->NewStringUTF(deltaStr);
  if (js == nullptr) {
    // OOM or pending exception; clear and ask the native runner to stop.
    if (ctx->env->ExceptionCheck()) {
      ctx->env->ExceptionClear();
    }
    return 1;
  }
  ctx->env->CallVoidMethod(ctx->listener, ctx->onDelta, js);
  ctx->env->DeleteLocalRef(js);
  if (ctx->env->ExceptionCheck()) {
    // Surface Kotlin-side errors as cancellation; the Kotlin override
    // in NativeQuickDotAI.runStreaming will catch the exception on the
    // JNI call's return and report it through StreamSink.onError.
    ctx->env->ExceptionDescribe();
    ctx->env->ExceptionClear();
    return 1;
  }
  return 0;
}
} // namespace

extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runModelHandleStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jstring promptJ,
  jobject listenerObj) {
  if (promptJ == nullptr || listenerObj == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Resolve the onDelta(String)V method id per-call. We can't cache
  // this globally because NativeStreamListener is a `fun interface`
  // and the concrete class of `listenerObj` varies call-to-call.
  jclass listenerCls = env->GetObjectClass(listenerObj);
  if (listenerCls == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }
  jmethodID onDelta =
    env->GetMethodID(listenerCls, "onDelta", "(Ljava/lang/String;)V");
  env->DeleteLocalRef(listenerCls);
  if (onDelta == nullptr) {
    if (env->ExceptionCheck()) {
      env->ExceptionClear();
    }
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  const char *prompt = env->GetStringUTFChars(promptJ, nullptr);
  if (prompt == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  StreamCtx ctx{env, listenerObj, onDelta};
  ErrorCode ec =
    runModelHandleStreaming(handle, prompt, &stream_trampoline, &ctx);

  env->ReleaseStringUTFChars(promptJ, prompt);
  return static_cast<jint>(ec);
}

// ---------------------------------------------------------------------------
// runMultimodalHandleStreaming
//
// Multimodal streaming inference that accepts preprocessed image patches
// (as FloatArray) and a text prompt. The pixel values are converted from
// jfloatArray to native float* and passed to the C API.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runMultimodalHandleStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jstring promptJ,
  jfloatArray pixelValuesJ, jint numPatches, jint originalHeight,
  jint originalWidth, jobject listenerObj) {
  if (promptJ == nullptr || pixelValuesJ == nullptr || listenerObj == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Resolve the onDelta(String)V method id
  jclass listenerCls = env->GetObjectClass(listenerObj);
  if (listenerCls == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }
  jmethodID onDelta =
    env->GetMethodID(listenerCls, "onDelta", "(Ljava/lang/String;)V");
  env->DeleteLocalRef(listenerCls);
  if (onDelta == nullptr) {
    if (env->ExceptionCheck()) {
      env->ExceptionClear();
    }
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  const char *prompt = env->GetStringUTFChars(promptJ, nullptr);
  if (prompt == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Get float* from FloatArray
  float *pixels = env->GetFloatArrayElements(pixelValuesJ, nullptr);
  if (pixels == nullptr) {
    env->ReleaseStringUTFChars(promptJ, prompt);
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  StreamCtx ctx{env, listenerObj, onDelta};

  ErrorCode ec = runMultimodalHandleStreaming(
    handle, prompt, pixels, numPatches, originalHeight, originalWidth,
    &stream_trampoline, &ctx);

  // Release resources
  env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);
  env->ReleaseStringUTFChars(promptJ, prompt);

  return static_cast<jint>(ec);
}

namespace {

/**
 * @brief Convert a Kotlin QuickAiChatMessage to C CausalLMChatMessage.
 *
 * QuickAiChatMessage structure:
 *   role: QuickAiChatRole (enum) — call name() to get String
 *   parts: List<PromptPart> — extract Text parts and concatenate
 *
 * Returns false if conversion fails (sets JNI exception).
 */
bool convertQuickAiChatMessage(JNIEnv *env, jobject msgObj,
                               std::string &outRole, std::string &outContent) {
  if (msgObj == nullptr)
    return false;

  jclass msgCls = env->GetObjectClass(msgObj);
  if (msgCls == nullptr)
    return false;

  // --- role: QuickAiChatRole enum ---
  jfieldID roleFid =
    env->GetFieldID(msgCls, "role", "Lcom/example/quickdotai/QuickAiChatRole;");
  if (roleFid == nullptr) {
    env->DeleteLocalRef(msgCls);
    return false;
  }
  jobject roleEnum = env->GetObjectField(msgObj, roleFid);
  if (roleEnum == nullptr) {
    env->DeleteLocalRef(msgCls);
    return false;
  }

  // Call enum.name() and convert to lowercase for OpenAI API compatibility
  jclass enumCls = env->GetObjectClass(roleEnum);
  jmethodID nameMid = env->GetMethodID(enumCls, "name", "()Ljava/lang/String;");
  jstring roleNameJ = (jstring)env->CallObjectMethod(roleEnum, nameMid);
  if (roleNameJ != nullptr) {
    const char *roleName = env->GetStringUTFChars(roleNameJ, nullptr);
    outRole = roleName ? roleName : "";
    // Convert to lowercase: "SYSTEM" -> "system", "USER" -> "user", "ASSISTANT"
    // -> "assistant"
    std::transform(outRole.begin(), outRole.end(), outRole.begin(), ::tolower);
    env->ReleaseStringUTFChars(roleNameJ, roleName);
    env->DeleteLocalRef(roleNameJ);
  }
  env->DeleteLocalRef(enumCls);
  env->DeleteLocalRef(roleEnum);

  // --- parts: List<PromptPart> ---
  jfieldID partsFid = env->GetFieldID(msgCls, "parts", "Ljava/util/List;");
  if (partsFid == nullptr) {
    env->DeleteLocalRef(msgCls);
    return false;
  }
  jobject partsList = env->GetObjectField(msgObj, partsFid);
  if (partsList == nullptr) {
    env->DeleteLocalRef(msgCls);
    return false;
  }

  // Get List.size() and List.get()
  jclass listCls = env->GetObjectClass(partsList);
  jmethodID sizeMid = env->GetMethodID(listCls, "size", "()I");
  jmethodID getMid = env->GetMethodID(listCls, "get", "(I)Ljava/lang/Object;");

  jint partsSize = env->CallIntMethod(partsList, sizeMid);
  std::string content;

  // PromptPart.Text class
  jclass textPartCls = env->FindClass("com/example/quickdotai/PromptPart$Text");
  if (textPartCls == nullptr) {
    if (env->ExceptionCheck())
      env->ExceptionClear();
  } else {
    jfieldID textFid =
      env->GetFieldID(textPartCls, "text", "Ljava/lang/String;");
    if (textFid != nullptr) {
      for (jint p = 0; p < partsSize; ++p) {
        jobject partObj = env->CallObjectMethod(partsList, getMid, p);
        if (partObj == nullptr)
          continue;

        if (env->IsInstanceOf(partObj, textPartCls)) {
          jstring textJ = (jstring)env->GetObjectField(partObj, textFid);
          if (textJ != nullptr) {
            const char *text = env->GetStringUTFChars(textJ, nullptr);
            if (text != nullptr) {
              if (!content.empty())
                content += " ";
              content += text;
              env->ReleaseStringUTFChars(textJ, text);
            }
            env->DeleteLocalRef(textJ);
          }
        }
        env->DeleteLocalRef(partObj);
      }
    }
    env->DeleteLocalRef(textPartCls);
  }

  outContent = std::move(content);

  env->DeleteLocalRef(listCls);
  env->DeleteLocalRef(partsList);
  env->DeleteLocalRef(msgCls);
  return true;
}

} // namespace

// ---------------------------------------------------------------------------
// runModelHandleWithMessagesStreaming
//
// Streaming inference with OpenAI message format on a specific handle.
// Converts jobjectArray of QuickAiChatMessage to C array, applies chat
// template, then drives streaming generation token-by-token.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runModelHandleWithMessagesStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jobjectArray messagesJ,
  jboolean addGenerationPrompt, jobject listenerObj) {

  if (messagesJ == nullptr || listenerObj == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Resolve listener method
  jclass listenerCls = env->GetObjectClass(listenerObj);
  if (listenerCls == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }
  jmethodID onDelta =
    env->GetMethodID(listenerCls, "onDelta", "(Ljava/lang/String;)V");
  env->DeleteLocalRef(listenerCls);
  if (onDelta == nullptr) {
    if (env->ExceptionCheck())
      env->ExceptionClear();
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Convert messages
  jsize len = env->GetArrayLength(messagesJ);
  std::vector<CausalLMChatMessage> msgs;
  msgs.reserve(len);
  std::vector<std::string> roleStorage;
  std::vector<std::string> contentStorage;
  roleStorage.reserve(len);
  contentStorage.reserve(len);

  for (jsize i = 0; i < len; ++i) {
    jobject msgObj = env->GetObjectArrayElement(messagesJ, i);
    std::string roleStr, contentStr;
    if (msgObj != nullptr &&
        convertQuickAiChatMessage(env, msgObj, roleStr, contentStr)) {
      roleStorage.push_back(std::move(roleStr));
      contentStorage.push_back(std::move(contentStr));
      msgs.push_back(
        {roleStorage.back().c_str(), contentStorage.back().c_str()});
    }
    if (msgObj != nullptr)
      env->DeleteLocalRef(msgObj);
  }

  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  StreamCtx ctx{env, listenerObj, onDelta};

  ErrorCode ec = runModelHandleWithMessagesStreaming(
    handle, msgs.data(), msgs.size(), addGenerationPrompt == JNI_TRUE,
    &stream_trampoline, &ctx);

  return static_cast<jint>(ec);
}

// ---------------------------------------------------------------------------
// runMultimodalHandleWithMessagesStreaming
//
// Streaming multimodal inference with OpenAI message format on a specific
// handle.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runMultimodalHandleWithMessagesStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jobjectArray messagesJ,
  jboolean addGenerationPrompt, jfloatArray pixelValuesJ, jint numPatches,
  jint originalHeight, jint originalWidth, jobject listenerObj) {

  if (messagesJ == nullptr || pixelValuesJ == nullptr ||
      listenerObj == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Resolve listener method
  jclass listenerCls = env->GetObjectClass(listenerObj);
  if (listenerCls == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }
  jmethodID onDelta =
    env->GetMethodID(listenerCls, "onDelta", "(Ljava/lang/String;)V");
  env->DeleteLocalRef(listenerCls);
  if (onDelta == nullptr) {
    if (env->ExceptionCheck())
      env->ExceptionClear();
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Convert messages
  jsize len = env->GetArrayLength(messagesJ);
  std::vector<CausalLMChatMessage> msgs;
  msgs.reserve(len);
  std::vector<std::string> roleStorage;
  std::vector<std::string> contentStorage;
  roleStorage.reserve(len);
  contentStorage.reserve(len);

  for (jsize i = 0; i < len; ++i) {
    jobject msgObj = env->GetObjectArrayElement(messagesJ, i);
    std::string roleStr, contentStr;
    if (msgObj != nullptr &&
        convertQuickAiChatMessage(env, msgObj, roleStr, contentStr)) {
      roleStorage.push_back(std::move(roleStr));
      contentStorage.push_back(std::move(contentStr));
      msgs.push_back(
        {roleStorage.back().c_str(), contentStorage.back().c_str()});
    }
    if (msgObj != nullptr)
      env->DeleteLocalRef(msgObj);
  }

  float *pixels = env->GetFloatArrayElements(pixelValuesJ, nullptr);
  if (pixels == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  StreamCtx ctx{env, listenerObj, onDelta};

  ErrorCode ec = runMultimodalHandleWithMessagesStreaming(
    handle, msgs.data(), msgs.size(), addGenerationPrompt == JNI_TRUE, pixels,
    numPatches, originalHeight, originalWidth, &stream_trampoline, &ctx);

  env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);

  return static_cast<jint>(ec);
}

// ---------------------------------------------------------------------------
// runModelHandleWithJsonStreaming
//
// Streaming inference with OpenAI JSON format on a specific handle.
// Accepts a JSON string containing messages, tools, functions, etc.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runModelHandleWithJsonStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jstring jsonRequestJ,
  jobject listenerObj) {

  if (jsonRequestJ == nullptr || listenerObj == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Resolve listener method
  jclass listenerCls = env->GetObjectClass(listenerObj);
  if (listenerCls == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }
  jmethodID onDelta =
    env->GetMethodID(listenerCls, "onDelta", "(Ljava/lang/String;)V");
  env->DeleteLocalRef(listenerCls);
  if (onDelta == nullptr) {
    if (env->ExceptionCheck())
      env->ExceptionClear();
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  const char *jsonRequest = env->GetStringUTFChars(jsonRequestJ, nullptr);
  if (jsonRequest == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  StreamCtx ctx{env, listenerObj, onDelta};

  ErrorCode ec = runModelHandleWithJsonStreaming(handle, jsonRequest,
                                                 &stream_trampoline, &ctx);

  env->ReleaseStringUTFChars(jsonRequestJ, jsonRequest);

  return static_cast<jint>(ec);
}

// ---------------------------------------------------------------------------
// runMultimodalMultiImageStreamingNative
//
// Multimodal streaming inference with multi-image support (V-JEPA).
// Accepts preprocessed pixel values for multiple images along with
// per-image metadata (patches per image, heights, widths).
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runMultimodalMultiImageStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jstring promptJ,
  jfloatArray pixelValuesJ, jint numPatches, jint numImages,
  jintArray patchesPerImageJ, jintArray originalHeightsJ,
  jintArray originalWidthsJ, jobject listenerObj) {

  if (promptJ == nullptr || pixelValuesJ == nullptr ||
      patchesPerImageJ == nullptr || originalHeightsJ == nullptr ||
      originalWidthsJ == nullptr || listenerObj == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Resolve listener method
  jclass listenerCls = env->GetObjectClass(listenerObj);
  if (listenerCls == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }
  jmethodID onDelta =
    env->GetMethodID(listenerCls, "onDelta", "(Ljava/lang/String;)V");
  env->DeleteLocalRef(listenerCls);
  if (onDelta == nullptr) {
    if (env->ExceptionCheck())
      env->ExceptionClear();
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  const char *prompt = env->GetStringUTFChars(promptJ, nullptr);
  if (prompt == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Get float* from FloatArray
  float *pixels = env->GetFloatArrayElements(pixelValuesJ, nullptr);
  if (pixels == nullptr) {
    env->ReleaseStringUTFChars(promptJ, prompt);
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Get int* from IntArrays
  jint *patchesPerImage = env->GetIntArrayElements(patchesPerImageJ, nullptr);
  if (patchesPerImage == nullptr) {
    env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);
    env->ReleaseStringUTFChars(promptJ, prompt);
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  jint *originalHeights = env->GetIntArrayElements(originalHeightsJ, nullptr);
  if (originalHeights == nullptr) {
    env->ReleaseIntArrayElements(patchesPerImageJ, patchesPerImage, JNI_ABORT);
    env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);
    env->ReleaseStringUTFChars(promptJ, prompt);
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  jint *originalWidths = env->GetIntArrayElements(originalWidthsJ, nullptr);
  if (originalWidths == nullptr) {
    env->ReleaseIntArrayElements(originalHeightsJ, originalHeights, JNI_ABORT);
    env->ReleaseIntArrayElements(patchesPerImageJ, patchesPerImage, JNI_ABORT);
    env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);
    env->ReleaseStringUTFChars(promptJ, prompt);
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  StreamCtx ctx{env, listenerObj, onDelta};

  // Debug: log multi-image metadata
  __android_log_print(
    ANDROID_LOG_DEBUG, LOG_TAG,
    "runMultimodalMultiImageStreamingNative: handle=%p, numPatches=%d, "
    "numImages=%d, pixelValues[0..4]=%f,%f,%f,%f,%f",
    (void *)handle, numPatches, numImages, pixels[0],
    (numPatches > 1 ? pixels[1] : 0.0f), (numPatches > 2 ? pixels[2] : 0.0f),
    (numPatches > 3 ? pixels[3] : 0.0f), (numPatches > 4 ? pixels[4] : 0.0f));
  for (int i = 0; i < numImages; ++i) {
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                        "  image[%d]: patches=%d, height=%d, width=%d", i,
                        patchesPerImage[i], originalHeights[i],
                        originalWidths[i]);
  }

  ErrorCode ec = runMultimodalMultiImageHandleStreaming(
    handle, prompt, pixels, numPatches, numImages,
    reinterpret_cast<int *>(patchesPerImage),
    reinterpret_cast<int *>(originalHeights),
    reinterpret_cast<int *>(originalWidths), &stream_trampoline, &ctx);

  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                      "runMultimodalMultiImageStreamingNative: returned ec=%d",
                      (int)ec);

  // Release resources
  env->ReleaseIntArrayElements(originalWidthsJ, originalWidths, JNI_ABORT);
  env->ReleaseIntArrayElements(originalHeightsJ, originalHeights, JNI_ABORT);
  env->ReleaseIntArrayElements(patchesPerImageJ, patchesPerImage, JNI_ABORT);
  env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);
  env->ReleaseStringUTFChars(promptJ, prompt);

  return static_cast<jint>(ec);
}

// ---------------------------------------------------------------------------
// runMultimodalMultiImageWithMessagesStreamingNative
//
// Streaming multimodal inference with multi-image + messages (V-JEPA).
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runMultimodalMultiImageWithMessagesStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jobjectArray messagesJ,
  jboolean addGenerationPrompt, jfloatArray pixelValuesJ, jint numPatches,
  jint numImages, jintArray patchesPerImageJ, jintArray originalHeightsJ,
  jintArray originalWidthsJ, jobject listenerObj) {

  if (messagesJ == nullptr || pixelValuesJ == nullptr ||
      patchesPerImageJ == nullptr || originalHeightsJ == nullptr ||
      originalWidthsJ == nullptr || listenerObj == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Resolve listener method
  jclass listenerCls = env->GetObjectClass(listenerObj);
  if (listenerCls == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }
  jmethodID onDelta =
    env->GetMethodID(listenerCls, "onDelta", "(Ljava/lang/String;)V");
  env->DeleteLocalRef(listenerCls);
  if (onDelta == nullptr) {
    if (env->ExceptionCheck())
      env->ExceptionClear();
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Convert messages
  jsize len = env->GetArrayLength(messagesJ);
  std::vector<CausalLMChatMessage> msgs;
  msgs.reserve(len);
  std::vector<std::string> roleStorage;
  std::vector<std::string> contentStorage;
  roleStorage.reserve(len);
  contentStorage.reserve(len);

  for (jsize i = 0; i < len; ++i) {
    jobject msgObj = env->GetObjectArrayElement(messagesJ, i);
    std::string roleStr, contentStr;
    if (msgObj != nullptr &&
        convertQuickAiChatMessage(env, msgObj, roleStr, contentStr)) {
      roleStorage.push_back(std::move(roleStr));
      contentStorage.push_back(std::move(contentStr));
      msgs.push_back(
        {roleStorage.back().c_str(), contentStorage.back().c_str()});
    }
    if (msgObj != nullptr)
      env->DeleteLocalRef(msgObj);
  }

  // Get float* from FloatArray
  float *pixels = env->GetFloatArrayElements(pixelValuesJ, nullptr);
  if (pixels == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Get int* from IntArrays
  jint *patchesPerImage = env->GetIntArrayElements(patchesPerImageJ, nullptr);
  if (patchesPerImage == nullptr) {
    env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  jint *originalHeights = env->GetIntArrayElements(originalHeightsJ, nullptr);
  if (originalHeights == nullptr) {
    env->ReleaseIntArrayElements(patchesPerImageJ, patchesPerImage, JNI_ABORT);
    env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  jint *originalWidths = env->GetIntArrayElements(originalWidthsJ, nullptr);
  if (originalWidths == nullptr) {
    env->ReleaseIntArrayElements(originalHeightsJ, originalHeights, JNI_ABORT);
    env->ReleaseIntArrayElements(patchesPerImageJ, patchesPerImage, JNI_ABORT);
    env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  StreamCtx ctx{env, listenerObj, onDelta};

  // Debug: log multi-image + messages metadata
  __android_log_print(
    ANDROID_LOG_DEBUG, LOG_TAG,
    "runMultimodalMultiImageWithMessagesStreamingNative: handle=%p, "
    "numMessages=%zu, numPatches=%d, numImages=%d, "
    "pixelValues[0..4]=%f,%f,%f,%f,%f",
    (void *)handle, msgs.size(), numPatches, numImages, pixels[0],
    (numPatches > 1 ? pixels[1] : 0.0f), (numPatches > 2 ? pixels[2] : 0.0f),
    (numPatches > 3 ? pixels[3] : 0.0f), (numPatches > 4 ? pixels[4] : 0.0f));
  for (int i = 0; i < numImages; ++i) {
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                        "  image[%d]: patches=%d, height=%d, width=%d", i,
                        patchesPerImage[i], originalHeights[i],
                        originalWidths[i]);
  }
  for (size_t i = 0; i < msgs.size(); ++i) {
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                        "  msg[%zu]: role='%s' contentLen=%zu", i, msgs[i].role,
                        strlen(msgs[i].content));
  }

  ErrorCode ec = runMultimodalMultiImageHandleWithMessagesStreaming(
    handle, msgs.data(), msgs.size(), addGenerationPrompt == JNI_TRUE, pixels,
    numPatches, numImages, reinterpret_cast<int *>(patchesPerImage),
    reinterpret_cast<int *>(originalHeights),
    reinterpret_cast<int *>(originalWidths), &stream_trampoline, &ctx);

  __android_log_print(
    ANDROID_LOG_DEBUG, LOG_TAG,
    "runMultimodalMultiImageWithMessagesStreamingNative: returned ec=%d",
    (int)ec);

  // Release resources
  env->ReleaseIntArrayElements(originalWidthsJ, originalWidths, JNI_ABORT);
  env->ReleaseIntArrayElements(originalHeightsJ, originalHeights, JNI_ABORT);
  env->ReleaseIntArrayElements(patchesPerImageJ, patchesPerImage, JNI_ABORT);
  env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);

  return static_cast<jint>(ec);
}
