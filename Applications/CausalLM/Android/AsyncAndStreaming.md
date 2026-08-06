# Android Native Async & Streaming 🔄

Quick.AI streaming is synchronous at the native C boundary and asynchronous at
the host-app boundary. The native call blocks the worker thread while invoking a
callback for each token delta; the app decides how to dispatch those deltas to
UI or transport layers.

## 🧭 Scope

This document covers the current `QuickDotAI` AAR path:

```text
QuickDotAI.kt
  └── NativeQuickDotAI.kt
      └── NativeCausalLm.kt
          └── quickai_jni.cpp
              └── quick_dot_ai_api.h / libquick_dot_ai_api.so
```

It does not describe the planned REST/foreground-service layer.

## 🧵 Streaming Contract

Native streaming functions are synchronous:

```c
ErrorCode runModelHandleStreaming(
    CausalLmHandle handle,
    const char *inputTextPrompt,
    CausalLmTokenCallback callback,
    void *user_data);
```

While the function runs, it calls:

```c
typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);
```

Returning `0` continues generation. Returning non-zero requests cooperative
cancellation at the next token boundary.

## 🔌 JNI Bridge

`quickai_jni.cpp` converts native token callbacks into Kotlin listener calls.
Because callbacks run on the same thread that entered JNI, the bridge can use
the current `JNIEnv *` without attaching a new thread.

Kotlin then forwards deltas to `StreamSink`:

```kotlin
interface StreamSink {
    fun onDelta(text: String)
    fun onReasoningDelta(text: String) {}
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}
```

## 🧱 QuickDotAI Methods

Current streaming methods include:

| Method | Input shape |
|---|---|
| `runModelHandleWithMessagesStreaming()` | `List<QuickAiChatMessage>` |
| `runModelHandleWithJsonStreaming()` | OpenAI-style JSON string |
| `runMultimodalHandleWithMessagesStreaming()` | OpenAI-style messages with image parts |
| `runMultimodalHandleStreaming()` | `List<PromptPart>` |
| `runChatModelHandleStreaming()` | Active chat session + text |
| `runChatMultimodalHandleStreaming()` | Active chat session + image/text parts |

The removed flat methods (`runStreaming`, `runWithMessagesStreaming`, and
`chatRunStreaming`) should not be used in new docs or app code.

## 🚦 Cancellation

- `QuickDotAI.cancel()` forwards to the native handle cancel path when the
  engine supports it.
- `QuickDotAI.chatCancel()` cancels the active chat session.
- Native cancellation is cooperative and may stop at the next generated token.
- `LiteRTLm` uses its Kotlin-side cancellation flag/session logic.

## ✅ Failure Semantics

Each streaming call should emit exactly one terminal sink event:

- `onDone()` on success
- `onError(error, message)` on failure

Native non-zero `ErrorCode` values are mapped through `QuickAiError.fromNativeCode`.

## 📎 Related Docs

- [QuickDotAI AAR API](QuickDotAI/README.md)
- [Android Architecture](Architecture.md)
- [Chat and OpenAI Usage Examples](../docs/ChatAndOpenAIUsage.md)
- [C API Reference](../api/README.md)
