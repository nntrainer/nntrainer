# QuickDotAI C API

This directory contains the public, handle-based C API used by the CausalLM
command-line test and the QuickDotAI Android AAR. The canonical header is
`quick_dot_ai_api.h`; it is self-contained for both C and C++ consumers.

The API supports model-catalog discovery, model load and destruction,
streaming text and multimodal inference, cancellation, chat sessions, tool
constraints through xgrammar, and performance metrics. Model selection uses
stable string catalog identifiers rather than the removed global `ModelType`
API.

## Build and link

The standalone Android build produces:

- `libcausallm.so`: models, layers, tokenizer integration, and xgrammar.
- `libquick_dot_ai_api.so`: the public C ABI in this directory.
- `libnntrainer.so` and `libccapi-nntrainer.so`: engine dependencies.

From `Applications/CausalLM`, run `./build_android.sh --app`. The AAR
mode is CPU-only unless QNN is explicitly enabled and never installs to a
device without `--install`. See the parent
[README](../README.md#4-android-build--test) for cache, QNN, and
device-install options. The no-option command builds the same canonical
CPU-native libraries and tools without Gradle.

Native consumers should include `quick_dot_ai_api.h` and link
`libquick_dot_ai_api.so`. Its transitive shared libraries must be available to
the platform loader. Android packages built by `build_android.sh` stage those
dependencies automatically.

## Lifecycle

The normal handle-based lifecycle is:

1. Inspect `getModelCatalogJson()` or choose a known catalog id.
2. Load with `loadModelHandleByName()`.
3. Run a handle-based streaming or chat-session entry point.
4. Use `cancelModelHandle()` when cancellation is required.
5. Release the model with `destroyModelHandle()`.

Refer to declarations and ownership comments in
[`quick_dot_ai_api.h`](quick_dot_ai_api.h) for the complete API contract.

## Gemma4 MoE on native HTP

The Android `--htp` native build registers `gemma4-26b-a4b-htp`, a
`NATIVE` model with the `NPU` backend and streaming/messages capabilities. Load it with
`CAUSAL_LM_QUANTIZATION_W4A32`; its directory is
`<model_base_path>/gemma4-26b-a4b-htp-w4a32`. This descriptor uses the validated
Gemma4 26B A4B pack: `gemma4_execution_profile=htp_fused`,
`gemma4_weight_format=htp_fused_v1`, QS2CX_WH experts, QS4CX FCs and Q4_0
embedding/head. The metadata does not convert another pack into this format.

Pass an absolute `native_lib_dir` containing the matching v81
`libnntr_hvx_skel.so`. The API sets the DSP search path and required runtime
options before opening HTP, and returns a load error if initialization fails.
Settings and the library directory remain fixed for the process. Conflicting
environment values and a second active HTP handle are rejected. Destroying or
unloading the active handle releases its workers, DSP session and weight
registries so another handle can load the model.

Each streaming run starts an independent prompt and rebuilds its KV caches;
the loaded weights stay resident. KV snapshot/resume is not supported by this
preset. Returning nonzero from the streaming callback or calling
`cancelModelHandle()` stops at a token boundary. Unload waits for any active
run to return; cancel first when an early stop is needed. The CPU prefill head
is retained for repeated runs (about 396 MiB for the validated untied pack).

Device checks and build commands are recorded in the
[QuickDotAI HTP guide](../../../docs/gemma4-htp-quickdotai.md).
Build the AAR/APK with `build_android.sh --htp --app`. The native SDK environment
must specify `HEX_ARCH=v81`, `HEXAGON_SDK_ROOT`, `DEFAULT_HEXAGON_TOOLS_ROOT`,
`HEXKL_ROOT` and the matching `HEXKL_SDK_VER`. Android packaging requires a JDK
and Android SDK. HTP and QNN are separate variants; do not combine their flags.
The package verifier checks ARM dependencies and preserves the DSP ELF bytes.

SampleTestAPP exposes the `gemma4-26b-a4b` family with `NATIVE / NPU / W4A32`.
Use an app-readable model base directory, then Load and Run in the OpenAI tab.
Selecting this model seeds both OpenAI and Chat with the validated summarization
prompt. Load/Run preserves edits to that prompt.

For interactive use, copy `chat_template.jinja`, `generation_config.json` and
`nntr_config.json` from [the HTP chat preset](../res/gemma4/htp_chat/) into the
validated model directory alongside its weights and tokenizer. The template is
Google's Gemma4 26B template at `35b4173`, with adjacent error-message string
literals joined for the current minja parser. It adds the empty thought channel
used when thinking is off. The preset restores EOS IDs `[1, 106, 50]`, retains
greedy sampling, and provides 1,280 context positions so the 1,028-token default
prompt and its generation budget fit without truncating the template suffix.
The earlier 1,024/128 benchmark pack deliberately disabled EOS; retain that pack
only for reproducing those measurements. Its fallback template is not the
recommended interactive configuration.

Chat input is plain message text; the API applies the template. The sample's raw
output view includes the final `<turn|>` marker. Each native HTP request remains
independent; the Chat tab does not retain conversation history for this preset.
See [the Chat verification results](../../../docs/gemma4-htp-quickdotai.md#validation)
for the exact template source, model settings and device results.

Load failures include the native diagnostic returned by
`getLastModelLoadError()` on the worker thread. The app compares the complete
load request before reusing a handle, and Reload always replaces the handle.

See the [SampleTestAPP validation results](../../../docs/gemma4-htp-quickdotai.md#validation)
for device checks and the current 16 KB compatibility limitation of the SDK
runtime/DSP packaging. Keep `jniLibs.useLegacyPackaging = true` and exclude
`libnntr_hvx_skel.so` from stripping in downstream applications.

## LFM2 MoE on native HTP

The `--htp` build also registers `lfm2-24ba2b-htp` with the NPU backend.
Load it with `CAUSAL_LM_BACKEND_NPU` and `CAUSAL_LM_QUANTIZATION_W4A32`
from `<model_base_path>/lfm2-24ba2b-htp-w4a32`.

The only supported LFM2 HTP format is `lfm2_weight_format=sr15_v2`, with
`lfm2_execution_profile=htp_fused`: TL2 experts, affine WH projections and
router, Q4_0 embedding and DW3_INT4 short convolutions. Keep the generated
`conversion_report.json` alongside the weights; the loader requires both
checkpoint rotation records. The legacy unrotated LFM2 pack is rejected.

Pass the absolute directory containing the matching v81 DSP library as
`native_lib_dir`. HTP runtime ownership follows the lifecycle above: only
one active HTP handle is supported, and each streaming call starts an
independent prompt. Public KV snapshot/resume remains unsupported. The
SR15 performance defaults in `lfm2_runtime_defaults.h` are selected by the
native CLI only; the API does not install those process-wide settings.
See the [LFM2 guide](../../../docs/lfm2-htp-user-guide.md) for conversion and
runtime requirements.

The standalone `test/htp/quick_dot_ai_validation.cpp` device check accepts
`lfm2 <legacy-base> <lib-dir> <sr15-base> <prompt-file>`. It checks legacy
format rejection, SR15 load, two identical greedy streaming requests and
unload. Each base contains the `lfm2-24ba2b-htp-w4a32` directory; the prompt
file contains an already formatted prompt.

For SampleTestAPP, select family `lfm2`, runtime `NATIVE` and backend `NPU`.
The model base path must contain the same `lfm2-24ba2b-htp-w4a32` directory.
The APK does not include model weights. Normal launches use
`/sdcard/Download/aistudio-mobile/models/`; stage the model there to use
Chat **Open** or OpenAI **Load model** without editing the path. A path
supplied to an instrumentation test is temporary and is not saved for the
next app launch.
The opt-in app lifecycle test accepts `htpModelId=lfm2-24ba2b-htp`; its
default remains Gemma4. After installing the HTP app and instrumentation
APKs, granting model-file access and acknowledging any first-launch system
dialogs, run:

```bash
adb -s "$DEVICE" shell am instrument -w -r \
  -e class 'com.example.sampletestapp.HtpAppLifecycleTest#loadGenerateStopRepeatReloadAndReportInvalidPath' \
  -e htpModelId lfm2-24ba2b-htp \
  -e htpModelBase /sdcard/Download/aistudio-mobile/models \
  -e htpMessages /sdcard/Download/lfm2/messages.json \
  com.example.sampletestapp.test/androidx.test.runner.AndroidJUnitRunner
```

`messages.json` is an OpenAI messages array. Use greedy generation and a
prompt/output limit long enough to exercise cancellation (for example,
96 generated tokens). The test checks repeatability, stop, unload during
generation, reload and recovery from an invalid model path. Reports and
native token traces are written under the app's `files/htp-app-validation`.

Also verify the normal-launch path with
`-e class 'com.example.sampletestapp.HtpAppLifecycleTest#lfm2ChatOpensFromDefaultPathAfterActivityRestart' -e htpDefaultPathCheck true`.
This test uses the default model path and LFM2 prompt. It opens Chat and
generates an answer, compares the OpenAI output and repeats after restarting the
Activity. Reports are stored in `files/htp-default-path-validation`.

## Compatibility

Consumers must use `quick_dot_ai_api.h` and `libquick_dot_ai_api.so`; the
former `causal_lm_api.h` ABI is not provided. Android builds use the same
canonical `libcausallm.so` core for native tools and app packaging.
