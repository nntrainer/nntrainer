# Android Architecture 📱

This document describes the current Android state of Quick.AI and separates it
from the planned REST/foreground-service layer that older documents described
as if it already existed.

## ✅ Current Gradle Modules

The Android build currently includes:

```text
Android/
├── QuickDotAI/       # AAR module
└── SampleTestAPP/    # Direct sample app using the AAR
```

`Android/settings.gradle.kts` includes only `:QuickDotAI` and
`:SampleTestAPP`.

## 🧱 QuickDotAI AAR

`QuickDotAI` exposes the public Kotlin API in
`com.example.quickdotai`.

Key files:

| File | Role |
|---|---|
| `QuickDotAI.kt` | Public interface and `BackendResult` / `StreamSink` contracts |
| `Types.kt` | Serializable request/response DTOs, model enums, errors, metrics |
| `NativeQuickDotAI.kt` | Kotlin wrapper around one native `CausalLmHandle` |
| `NativeCausalLm.kt` | Low-level JNI declarations |
| `LiteRTLm.kt` | LiteRT-LM engine wrapper for the `gemma4` (`ModelIds.GEMMA4`) model |
| `NativeChatSession.kt` | Native chat-session helper |
| `LiteRTLmChatSession.kt` | LiteRT-LM chat-session helper |
| `ImageStore.kt` | Per-session image cache (SHA-256 dedup) |
| `LlavaNextImageProcessor.kt` | Native multimodal preprocessing helper (any-resolution patching) |
| `PilloBilinearResizer.kt` | Pillow-compatible bilinear resize used by the image processors |
| `src/main/cpp/quickai_jni.cpp` | JNI bridge to `quick_dot_ai_api.h` |
| `src/main/cpp/CMakeLists.txt` | Builds `libquickai_jni.so` and links `libquick_dot_ai_api.so` |

## 🔌 Native Path

`NativeQuickDotAI` owns one native handle:

```text
NativeQuickDotAI
  └── NativeCausalLm.ensureLoaded()
      ├── optionally loads libqnn_context.so when it is packaged
      └── loads libquickai_jni.so
            └── links/calls libquick_dot_ai_api.so
                  └── links/calls libcausallm.so
```

The native API surface is declared in `api/quick_dot_ai_api.h`.
The preferred calls are handle-based:

- `loadModelHandleByName` (dispatched from Kotlin via the
  `loadModelHandleByNameNative` JNI declaration in `NativeCausalLm.kt`)
- `runModelHandleWithMessagesStreaming`
- `runModelHandleWithJsonStreaming`
- `runMultimodalHandleStreaming`
- `cancelModelHandle`
- `destroyModelHandle`

## ModelCatalog

Model selection in the AAR is driven by the `ModelCatalog` singleton. Models
are identified by string ids rather than an enum.

### Seeding

`ModelCatalog` is seeded on first access by calling `nativeQueryCatalog()`
through JNI, which delegates to `getModelCatalogJson()` in
`libquick_dot_ai_api.so`. Hardcoded LiteRT descriptors (e.g., `gemma4`) are
merged in at the Kotlin layer.

### Key types

| Type | Role |
|---|---|
| `enum class RuntimeKind { NATIVE, LITERT }` | Selects the engine path |
| `enum class Capability { STREAMING, MESSAGES_API, MULTIMODAL, TOOL_USE, EMBEDDING, MULTI_IMAGE }` | Per-model feature flags |
| `data class ModelDescriptor(id, family, displayName, runtime, backends, capabilities)` | Descriptor from the catalog |
| `object ModelIds` | String constants for well-known model ids |
| `object ModelCatalog` | Singleton: `all()`, `families()`, `selectable()`, `selectableFamilies()`, `runtimesFor(family)`, `backendsFor(family, rt)`, `resolve(family, rt, backend)`, `byId(id)` |

### 3-axis cascading UI

`SampleTestAPP` presents a 3-axis cascading UI:

1. **Family** — populated from `ModelCatalog.selectableFamilies()`
2. **Runtime chip row** — populated from `ModelCatalog.runtimesFor(selectedFamily)`
3. **Backend chip row** — populated from `ModelCatalog.backendsFor(selectedFamily, selectedRuntime)`

The app lists only **selectable** (generative) models. Embedding-only models
such as `tiny-bert` — which expose only the `EMBEDDING` capability and have no
public output path — are filtered out by `selectableFamilies()`. They remain in
the AAR catalog and are still reachable through `ModelCatalog.all()` /
`ModelCatalog.byId(...)`.

The resolved descriptor is obtained via `ModelCatalog.resolve(family, runtime, backend)`
and passed directly to `createEngine()`.

### Engine factory

```kotlin
QuickDotAI.createEngine(context, descriptor: ModelDescriptor): QuickDotAI
```

`createEngine` dispatches to `NativeQuickDotAI` (for `RuntimeKind.NATIVE`) or
`LiteRTLm` (for `RuntimeKind.LITERT`) based on `descriptor.runtime`.

### LoadModelRequest

`LoadModelRequest.modelId` is a `String` catalog id. The cache key is
`"$modelId:${quantization.name}"`. The JNI call dispatched on load is
`loadModelHandleByNameNative`.

## 🌗 LiteRT Runtime Path

`LiteRTLm` is selected for the `gemma4` (`ModelIds.GEMMA4`) model and takes a `.litertlm` file path
through `LoadModelRequest.modelPath`. `visionBackend != null` enables
multimodal calls for engines/models that support image input.

## 🧵 Threading Model

A `QuickDotAI` instance is not internally thread-safe. Host apps should drive a
loaded engine from one worker thread. `SampleTestAPP` follows this pattern with
a background dispatcher.

Streaming callbacks are delivered to the caller-provided `StreamSink`.
Apps that update UI must marshal callbacks to the main thread.

## 🧪 SampleTestAPP

`SampleTestAPP` is the current runnable Android sample. It links the
`:QuickDotAI` module directly; it does not start a REST service and does not
communicate over sockets.

## 🗺️ Planned Service Layer

The following pieces are design targets, not current Gradle modules:

| Planned component | Status |
|---|---|
| `LauncherApp` foreground-service bootstrap UI | Planned |
| `QuickAIService` remote foreground service | Planned |
| NanoHTTPD loopback REST server | Planned |
| `RequestDispatcher`, `ModelRegistry`, `ModelWorker` | Planned |
| Standalone REST client app | Planned |

When implemented, the service layer should wrap the same `QuickDotAI` AAR
contract rather than inventing a separate model API.

## 📦 Packaging

`../build_android.sh` owns the current Android packaging workflow. Its
no-option mode builds the canonical CPU native artifacts with Meson. `--app`
adds the standalone app, stages `libcausallm.so`,
`libquick_dot_ai_api.so`, and their nntrainer dependencies into
`QuickDotAI/prebuilt_libs/`, and assembles the QuickDotAI AAR and
SampleTestAPP without touching a device.

Use `--qnn` with `QNN_SDK_ROOT` to select the QNN variant; CPU is the default.
`--install` pushes the canonical native libraries and
tools, and installs SampleTestAPP only when combined with `--app`. Use
`ANDROID_SERIAL` to select a device when necessary. `--cache` reuses a
compatible engine build when available and rebuilds it on a cache miss.

## 📎 Related Docs

- [QuickDotAI AAR API](QuickDotAI/README.md)
- [Android Native Async & Streaming](AsyncAndStreaming.md)
- [Main README](../README.md)
