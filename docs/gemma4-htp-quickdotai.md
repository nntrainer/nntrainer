# Gemma4 MoE on HTP through QuickDotAI

New here? Start with the [user guide](gemma4-htp-user-guide.md) (build, model
pack, app, command line, measured results) and the
[code guide](gemma4-htp-code-guide.md) (how the HTP path is put together).
This page is the setup and validation record of the integration. For the
current native TL2/affine and rotated pack contracts, see the
[TL2 guide](gemma4-htp-tl2-guide.md). The
[LFM2 guide](lfm2-htp-user-guide.md) covers LFM2 and SR15 CLI support.

Status as of 2026-10-18: the native runtime also carries compact TL2
expert storage, prefill-memory reclaim and the decode changes described
in the [TL2 validation record](gemma4-htp-tl2-guide.md#current-validation-coverage).
Its latest S26 measurements are CLI checks; the dated app/API lifecycle
results below were not rerun as part of that series. After compact-slot
RPC changes, stage matching ARM libraries, generated stub and DSP skeleton.

The native HTP preset runs Gemma4 26B A4B through the same QuickDotAI handle,
JNI and Android sample interfaces as other native models. Select
`gemma4-26b-a4b` / `NATIVE` / `NPU` in SampleTestAPP. Quantization is `W4A32`.

The HTP implementation is ported from PR #4416 at
`48a6bbb22f96087f9b993cf87e33c70f9840fe26`, onto main at
`6ca5d2b7722a9676b763f35b31257f21125b3aca`. The source PR revision is fixed.
The default Gemma4 factory and converter remain the main implementations;
the fused runtime requires explicit profile and weight-format metadata.

## Model and runtime contract

The public descriptor is `gemma4-26b-a4b-htp`. Its directory is
`<model_base_path>/gemma4-26b-a4b-htp-w4a32/` and contains the tokenizer,
model configuration, generation configuration and converted weights.

The validated pack uses `gemma4_execution_profile=htp_fused`,
`gemma4_weight_format=htp_fused_v1`, QS2CX_WH experts, QS4CX fully connected
weights, Q4_0 embedding/head, FP32 activations and q8 attention KV. Merely
changing metadata does not convert weights. The separate
[HTP converter](../Applications/CausalLM/res/gemma4/htp_weight_converter.py)
produces the pinned fused layout; its tiny FP32 reference and conversion
checks are included with the tests.

This preset supports text, batch one and one active HTP handle per process.
Each generation starts an independent prompt while retaining the loaded
weights. Native Chat does not accumulate conversation history for this
preset. The public API does not expose KV snapshot/resume; the native CLI
has a separate [snapshot workflow](gemma4-htp-kv-cache-s25u.md).
Concurrent HTP handles are not supported.
Cancellation stops at a token boundary. Unload waits for active work and
releases the DSP session, worker pools and registered weights.

## Android build

Provide an Android NDK, JDK 17, Android SDK, Hexagon SDK/toolchain and matching
HexKL addon. The tested versions were NDK r30, Hexagon SDK 6.4.0.1,
Hexagon tools 19.0.04, and HexKL 1.0 beta 2 with its 6.4.0.2 library directory.
Set these variables to their installation paths:

```bash
export ANDROID_NDK=/path/to/android-ndk
export JAVA_HOME=/path/to/jdk-17
export ANDROID_HOME=/path/to/android-sdk
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.1
export DEFAULT_HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04"
export HEXKL_ROOT=/path/to/hexkl_addon
export HEXKL_SDK_VER=6.4.0.2
export HEX_ARCH=v81
bash Applications/CausalLM/build_android.sh --htp --app
```

HTP and QNN are separate build variants; combining their flags is rejected.
The build checks the runtime dependencies and preserves the v81 DSP ELF
when creating `QuickDotAI-debug.aar` and `SampleTestAPP-debug.apk`.
The device's vendor `libcdsprpc.so` is used at runtime; the SDK link stub is
not packaged. Downstream apps must retain `jniLibs.useLegacyPackaging=true`
and exclude `libnntr_hvx_skel.so` from stripping.

Install the APK, grant its requested model-file access, and set MODEL BASE
PATH to the directory containing the model folder. Load/Run, Stop, Unload
and Reload operate through the existing single engine worker. A changed
base path or backend cannot silently reuse the old handle. Native load
diagnostics are returned to the UI.

## Chat and the default prompt

Copy the three JSON/Jinja files from the
[HTP chat preset](../Applications/CausalLM/res/gemma4/htp_chat/) into the
validated pack. This preset is for the weight layout described above.
Model selection fills OpenAI with a messages array and Chat with its user
message: a three-sentence English summarization request about Ardley.
Both use the same bundled resource; Load/Run preserves user edits.

Chat accepts plain text. The model's template supplies the turn delimiters
and the empty thought channel recommended for the 26B thinking-off mode:

```text
<bos><|turn>user
I bought a red car yesterday<turn|>
<|turn>model
<|channel>thought
<channel|>
```

The template is Google's `gemma-4-26B-A4B-it` template at `35b4173`.
Adjacent error-message string literals are joined for the existing minja
parser; rendering logic is unchanged. Sources:
[template](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/35b4173/chat_template.jinja),
[thinking mode](https://ai.google.dev/gemma/docs/capabilities/thinking), and
[generation configuration](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/47b6801/generation_config.json).

The interactive preset uses EOS IDs `[1, 106, 50]` and greedy sampling.
Its 1,280 context positions accommodate the 1,028-token summary prompt plus
the generation budget. The earlier 1,024/128 comparison deliberately used
an older template and disabled EOS. Those benchmark settings must not be
used for interactive responses: they continue generating after the turn
ends. SampleTestAPP displays the raw final `<turn|>` marker.

## Validation

Validation used an SM-S948N running Android 16 with Hexagon v81. The source
PR and integration used the same 7,540,724,856-byte weight file, SHA-256
`9a4c16e33a9eede48bd745d392fc8b868bf3b08b1362c1a2390745ca8d8c4a06`.

| Check | Result |
|---|---|
| Host ASan regression | Nine suites; 245 passed, eight ARM/Bionic-only tests skipped, no failures. Leak detection disabled. |
| Tiny fused model/converter | Required FP32 fixture passed; production conversion matched the independent 600,488-byte reference. |
| Pinned CLI comparison | Three alternating runs each; all 128 output IDs matched with identical 1,024 input IDs. |
| QuickDotAI lifecycle | Seven complete runs matched the CLI; callback/thread cancellation, active unload, invalid pack and reload checked. |
| SampleTestAPP lifecycle | Seven complete runs matched the CLI; Stop and active Unload returned matching partial prefixes; path failure and recovery checked. |
| Chat short prompt | 19 input / 117 output tokens; repeated run identical; stopped at the first end-of-turn ID. |
| Chat/OpenAI summary | Both 1,028 input / 109 output tokens, identical IDs, complete thought-off suffix and no generated thought channel. |
| Android builds | Native CPU/HTP/QNN, HTP and QNN AAR/APK builds passed. QNN SDK 2.47.1.260610; QNN inference not tested. |
| Packaging | ARM dependencies, v81 DSP image/hash and installed APK hash checked. |

On the alternating CLI runs, prefill median was 552.916 tokens/s for the
fixed PR and 558.647 for the integration; decode was 13.5409 and 14.0736.
Temperature/frequency were not fixed, so these are regression observations,
not a claimed performance improvement.

Opt-in device tests use the actual Activity controls. Stage the bundled
`gemma4_moe_messages.json` resource at an app-readable path and supply:

```bash
adb shell am instrument -w -r \
  -e class 'com.example.sampletestapp.HtpAppLifecycleTest#chatUsesGemma4PromptFormat' \
  -e htpModelBase /sdcard/Download/aistudio-mobile/models \
  -e htpMessages /sdcard/Download/aistudio-mobile/messages.json \
  com.example.sampletestapp.test/androidx.test.runner.AndroidJUnitRunner
```

The separate `loadGenerateStopRepeatReloadAndReportInvalidPath` method
checks the model lifecycle. Tests without model-path arguments skip the
large device workload. The native API lifecycle checks can also be run
by setting `QDA_HTP_LIFECYCLE=1` when invoking `quick_dot_ai_test` for
`gemma4-26b-a4b-htp`, and with the fixture validation harness
in `test/htp/quick_dot_ai_validation.cpp`.

## Known limits

The test device uses 4KB pages. HexKL's ARM `libsdkl.so` has 4KB LOAD
alignment, and the DSP ELF is packaged in the native library directory.
Android displays a debug 16KB compatibility warning. This build is not
validated for 16KB-page devices. Support requires a compatible HexKL
runtime and review of DSP asset extraction; the warning is not suppressed.

Windows, Tizen, Yocto and the complete upstream CI matrix were not run as
part of the device integration validation. The PR's CI results remain the
source of truth for those platforms.
