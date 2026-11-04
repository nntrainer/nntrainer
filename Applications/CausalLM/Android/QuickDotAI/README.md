# QuickDotAI Android library

`QuickDotAI` is the Kotlin library (an AAR) that Android apps use to run
language models with nntrainer. It offers one interface, `QuickDotAI`, over
two engines:

- `NativeQuickDotAI`: the nntrainer engine through JNI (`libquickai_jni.so`
  over the [C API](../../api/README.md)). CPU models, QNN models and the
  Hexagon NPU (HTP) models Gemma4 26B-A4B and LFM2 24B-A2B.
- `LiteRTLm`: Google's LiteRT-LM runtime for `.litertlm` files (Gemma4 E2B).

The sample app that exercises all of it is
[QuickDotAIAPP](../QuickDotAIAPP/README.md); its structure is described in
[Architecture.md](../Architecture.md). This page is the library's contract.

## Add the dependency

```kotlin
dependencies {
    implementation(project(":QuickDotAI"))   // or the published QuickDotAI-debug.aar
}
```

The AAR carries every native library the engine needs for `arm64-v8a` (the
only supported ABI): `libquickai_jni.so`, `libquick_dot_ai_api.so`,
`libcausallm.so`, `libnntrainer.so`, `libccapi-nntrainer.so`,
`libc++_shared.so`, and for the HTP build `libsdkl.so` and the DSP image
`libnntr_hvx_skel.so`. `libcdsprpc.so` is not packaged; the phone's vendor
copy is used. Minimum SDK 33, compile SDK 36, Java 17.

Build it with `Applications/CausalLM/build_android.sh --htp --app` (or
without `--htp` for a CPU-only package); see
[Build from source](../../../../docs/quickai/build.md). A direct Gradle build
must pass `-PnntrainerNdkPath=<absolute NDK path>`.

### Packaging rules for the consuming app

The HTP build needs two settings in the app's `build.gradle.kts`, which the
sample app shows:

```kotlin
android {
    packaging {
        jniLibs.useLegacyPackaging = true
        jniLibs.keepDebugSymbols += "**/libnntr_hvx_skel.so"
    }
}
```

The DSP image must be extracted to the native library directory as a file
(legacy packaging) and must not be stripped: it is a Hexagon ELF, not an ARM
library, and the FastRPC loader reads it byte for byte. Pass
`applicationInfo.nativeLibraryDir` as `LoadModelRequest.nativeLibDir`; the
engine refuses to open the NPU without it.

Reading model packs from `/sdcard/Download` needs the all-files permission
(`MANAGE_EXTERNAL_STORAGE`), which the library's manifest declares and the
app must request at run time (`ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION`).

## Load and run a model

```kotlin
val descriptor = ModelCatalog.byId(ModelIds.GEMMA4_26B_A4B_HTP)!!
val engine = createEngine(context, descriptor)          // NativeQuickDotAI for a NATIVE descriptor
val result = engine.load(
    LoadModelRequest(
        modelId = ModelIds.GEMMA4_26B_A4B_HTP,
        backend = BackendType.NPU,
        quantization = QuantizationType.W4A32,
        nativeLibDir = context.applicationInfo.nativeLibraryDir,
        modelBasePath = "/sdcard/Download/aistudio-mobile/models",
    )
)
when (result) {
    is BackendResult.Ok -> engine.runModelHandleWithMessagesStreaming(
        listOf(QuickAiChatMessage(QuickAiChatRole.USER, listOf(PromptPart.Text("Summarize …")))),
        object : StreamSink {
            override fun onDelta(text: String) { /* append */ }
            override fun onDone() { /* finished */ }
            override fun onError(error: QuickAiError, message: String?) { /* show */ }
        })
    is BackendResult.Err -> log(result.error, result.message)   // message = the engine's load diagnostic
}
```

The model directory is `<modelBasePath>/gemma4-26b-a4b-htp-w4a32` (or
`lfm2-24ba2b-htp-w4a32`); `modelBasePath` null means
`/sdcard/Download/aistudio-mobile/models/`. The HTP models accept only
`BackendType.NPU` with `QuantizationType.W4A32`; anything else fails with
`MODEL_LOAD_FAILED`. See [Model packs](../../../../docs/quickai/model-packs.md)
for the pack contents.

## The `QuickDotAI` interface

Every method returns a `BackendResult<T>` (`Ok(value)` or
`Err(error: QuickAiError, message: String?)`); nothing throws across the
library boundary.

| Member | Purpose |
|---|---|
| `kind`, `architecture`, `chatSessionId` | `"native"` or `"litert-lm"`; the engine-reported architecture; the open chat session id or null |
| `load(req: LoadModelRequest)` | Load once before inference. Loading an already loaded engine returns `Ok` without reloading. |
| `runModelHandleWithMessagesStreaming(messages, sink)` | OpenAI-style messages (`QuickAiChatMessage(role, parts)`), rendered with the pack's chat template, streamed. The normal text path. |
| `runMultimodalHandleWithMessagesStreaming(messages, sink)` | Messages whose last user turn carries `ImageBytes` parts; vision models only. |
| `runModelHandleWithJsonStreaming(jsonRequest, sink)` | A raw OpenAI JSON request (`messages`, `tools`, …) passed to the template as is. The only path that forwards `tools`. |
| `runMultimodalHandle(parts)`, `runMultimodalHandleStreaming(parts, sink)` | Image plus text without the messages wrapper; vision models only. |
| `encode(text)` | Sentence embedding for embedding models. |
| `openChatSession(config)`, `runChatModelHandleStreaming(text, sink)`, `runChatMultimodalHandleStreaming(parts, sink)`, `chatRebuild(messages)`, `chatCancel()`, `closeChatSession()` | Chat sessions. One session at a time. |
| `cancel()` | Stop the running generation at the next token. Safe from any thread; a no-op when idle. |
| `metrics()` | `PerformanceMetrics` of the last run: prefill and generation token counts and durations, total and initialisation time, peak memory, time to first token, steady decode time per token. |
| `unload()` | Release the weights and the DSP session; the instance stays usable for another `load`. |
| `close()` | Release everything. Idempotent. |
| `createEngine(context, descriptor, modelBasePath = null)` | Factory: `NativeQuickDotAI` for a `NATIVE` descriptor, `LiteRTLm` for `LITERT`. |

`LoadModelRequest` fields: `modelId` (required), `backend` (default GPU),
`quantization` (default W4A32), `nativeLibDir`, `modelBasePath`,
`useSpeculativeDecoding`, and for LiteRT `modelPath`, `visionBackend`,
`cacheDir`, `maxNumTokens`. `htpBackendConfigPath` is accepted but not
forwarded by the native engine.

### Streaming, threads and cancellation

- A sink receives zero or more `onDelta(text)` calls, then exactly one
  `onDone()` or `onError(error, message)`. `onReasoningDelta` carries the
  reasoning channel of LiteRT models that have one.
- The native engine calls the sink on the thread that called the run; the
  callback must not block. LiteRT may call it from an internal thread.
- A `QuickDotAI` instance is **not** thread-safe. Keep one instance per
  loaded model and drive it from one worker thread (the sample app uses a
  single-thread executor); never run two generations on one instance at
  once.
- `cancel()` and `chatCancel()` may be called from any thread, including
  the UI thread. The native engine sets a stop flag and the run returns at
  the next token boundary with `onDone`.
- `unload()` waits for an active run to return; cancel first when you need
  an early stop.

### Chat sessions

Chat sessions give a model conversation memory. The LiteRT engine keeps a
real session with a system prompt, sampling settings and a thinking switch
(`QuickAiChatSessionConfig`), and `chatRebuild` reseeds it. The native
engine implements the session calls, but each `runChatModelHandleStreaming`
sends only that one user message through the pack's template, and
`chatRebuild` does nothing: the native engine keeps no conversation state
between calls. To give a native model history, pass the whole transcript
through `runModelHandleWithMessagesStreaming`.

For the HTP models, do **not** pass the transcript. Their packs hold 1,280
positions (Gemma4 chat preset) and a prompt that does not fit is cut at the
end without an error (nntrainer issue #4474). The sample app sends only the
newest user message to `ModelIds.GEMMA4_26B_A4B_HTP` and
`ModelIds.LFM2_24BA2B_HTP`.

## The catalog

`ModelCatalog` lists what the installed native libraries can run plus the
LiteRT entry. It is read from the C API (`getModelCatalogJson`) when
`libquickai_jni.so` loads, so a CPU-only build does not list the HTP models.

| Query | Returns |
|---|---|
| `ModelCatalog.all()`, `byId(id)` | all descriptors; one by id |
| `families()`, `runtimesFor(family)`, `backendsFor(family, runtime)`, `resolve(family, runtime, backend)` | the three-axis navigation the sample app's model sheet uses |
| `selectable()` | descriptors with a usable capability (`STREAMING`, `MESSAGES_API`, `MULTIMODAL` or `TOOL_USE`) |

A `ModelDescriptor` has `id`, `family`, `displayName`, `runtime`
(`NATIVE` or `LITERT`), `backends` and `capabilities` (`STREAMING`,
`MESSAGES_API`, `MULTIMODAL`, `TOOL_USE`, `EMBEDDING`, `MULTI_IMAGE`,
`SPECULATIVE`). The HTP descriptors are `gemma4-26b-a4b-htp` (family
`gemma4-26b-a4b`) and `lfm2-24ba2b-htp` (family `lfm2`), both `NATIVE`,
backend `NPU`, capabilities `STREAMING` and `MESSAGES_API`. `ModelIds`
holds the id constants.

## Errors

`QuickAiError` codes: `NONE`, `INVALID_PARAMETER`, `MODEL_LOAD_FAILED`,
`INFERENCE_FAILED`, `NOT_INITIALIZED`, `INFERENCE_NOT_RUN`, `UNKNOWN`,
`QUEUE_FULL`, `MODEL_NOT_FOUND`, `UNSUPPORTED`, `BAD_REQUEST`. A load
failure's `message` is the engine's diagnostic (the C API's
`getLastModelLoadError()`), for example
`HTP native_lib_dir is missing libnntr_hvx_skel.so` or
`Unknown runtime configuration: htp.pagecache`. Operations without a loaded
model return `NOT_INITIALIZED`. The C API's `UNSUPPORTED` code (6) maps to
`QuickAiError.UNKNOWN`, not to `QuickAiError.UNSUPPORTED`.

## The LiteRT engine

`LiteRTLm` loads a `.litertlm` file; with a blank `modelPath` it looks for
`<modelBasePath>/gemma-4-E2B-it/gemma-4-E2B-it.litertlm`. It supports chat
sessions with a system prompt, sampling settings and thinking on or off,
multimodal input, and reports a reasoning channel through
`onReasoningDelta`. Its `cancel()` lets the current answer finish.

## Verifying a package

`Applications/CausalLM/Android/verify_native_package.py --variant htp` checks
an AAR and an APK: every staged library is present, the DSP image is
byte-identical to the built one and is a v81 image, no `libQnn*` or
`libcdsprpc.so` is packaged, and every `NEEDED` entry resolves. The build
script runs it after Gradle.
