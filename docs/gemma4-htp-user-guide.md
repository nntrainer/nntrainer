# Gemma4 on the Hexagon HTP: user guide

This guide takes you from a clean Ubuntu machine to Gemma4 26B A4B generating
text on the NPU (Hexagon HTP) of a Galaxy S26 Ultra, through the command-line
tool and through the SampleTestAPP Android app. It assumes nothing about the
HTP code; the companion [code guide](gemma4-htp-code-guide.md) explains how
the pieces fit together, and the shorter
[setup and validation record](gemma4-htp-quickdotai.md) lists what was
validated and how.

The setup and measurements below describe the original 2026-10-13
QS2/Q4 integration. For the current native TL2/affine and rotated weight
contracts, use the [TL2 guide](gemma4-htp-tl2-guide.md). The
[LFM2 guide](lfm2-htp-user-guide.md) covers LFM2/SR15 and the latest matched
LFM2/Gemma4 measurements; those packs and workloads differ from this record.

Updated 2026-10-18: the current native TL2 pack uses compact expert slots,
reclaims prefill staging and supports an affine untied head on the NPU.
The latest 1024-input, C16/preload-128 series recorded medians of 621.36
prefill and 12.32 decode tok/s over three runs with matching reference IDs.
For the earlier 2048-input decode comparison, separate memory savings and
their limits, see the [TL2 validation record](gemma4-htp-tl2-guide.md#current-validation-coverage).

## 1. What you get

- Gemma4 26B A4B (a mixture-of-experts model, 128 experts, 4 B active) runs
  its decoder blocks on the Hexagon HTP. The CPU keeps the tokenizer, the
  embedding lookup, half of the language-model head and the sampling loop.
- Weights are stored in the `htp_fused_v1` layout: 2-bit palette experts
  (`QS2CX_WH`), 4-bit dense projections (`QS4CX`), 4-bit embedding and head
  (`Q4_0`), an int8 attention KV cache. The whole pack is 7.5 GB.
- Prefill (reading the prompt) runs layer by layer through the HTP; decoding
  (producing tokens) runs inside one resident DSP graph that returns only the
  next token id, so the CPU is idle between tokens.
- The same binary still runs every other model on the CPU or the GPU exactly
  as before; the HTP path is selected only by two keys in the model's
  `nntr_config.json`.
- The branch carries the HTP work of PR #4416 as its own history (553
  commits) plus its 18 later commits: the DSP-side q2 calibration, the
  selectable page-cache reader, resident-weight sharing, the windowed KV cache
  of the sliding layers, growing int8 scales and int8 attention at any
  context length.

## 2. What you need

| Item | Tested with |
|---|---|
| Phone | Samsung SM-S948N (Galaxy S26 Ultra), Snapdragon SM8850, Hexagon v81, Android 16, 4 KB pages, USB debugging on |
| Host | Ubuntu 24.04, 8 cores, 32 GB RAM, about 60 GB free disk |
| Android NDK | r30 |
| JDK | 17 (Temurin 17.0.20) |
| Android SDK | platform 36, build-tools 36.0.0 (35.0.0 also used by Gradle), cmake 3.22.1, platform-tools |
| Hexagon SDK | 6.4.0.1 with Hexagon Tools 19.0.04 |
| HexKL | 1.0 beta 2, its `lib/6.4.0.2` directory (`hexagon_toolv19_v81` and `armv8_android26`) |
| Python | 3.10+, with `torch`, `transformers` (5.15 or newer with `Gemma4Config`), `numpy`, `safetensors` for the converter |
| Model | the Hugging Face checkpoint `google/gemma-4-26B-A4B-it` |

The Hexagon SDK and HexKL come from Qualcomm; they are not part of this
repository. Only a v81 device is supported by this build (`HEX_ARCH=v81` is
enforced).

## 3. Get the source

```bash
git clone https://github.com/nntrainer/nntrainer.git
cd nntrainer
git submodule sync && git submodule update --init --recursive --depth 1
```

Where the HTP code lives:

| Path | What |
|---|---|
| `nntrainer/tensor/htp_backend/` | ARM side of the backend (FastRPC session, op table, graph and token driver) and the DSP kernels under `hvx/` and `hmx/` |
| `test/htp/` | the DSP skel sources (`nntr_hvx_*.c`), the IDL and `build.sh` that builds `libnntr_hvx_skel.so` |
| `test/htp/host/` | host checks that run on x86 without a phone |
| `Applications/CausalLM/layers/htp_*` | attention, QKV and fused blocks of the HTP graph |
| `Applications/CausalLM/models/gemma4/gemma4_htp_*` | the fused Gemma4 model and its weight loader |
| `Applications/CausalLM/res/gemma4/` | `htp_weight_converter.py` and the `htp_chat/` preset |
| `Applications/CausalLM/api/` | the QuickDotAI C API the app uses (`htp_model_runtime.h` fixes the HTP runtime options) |
| `Applications/CausalLM/Android/` | the QuickDotAI library and SampleTestAPP |

## 4. Build

### 4.1 Host build (no phone needed)

This builds the library, the CPU-side tests and the Gemma4 HTP model tests
with a tiny FP32 reference. It does not need the Hexagon SDK.

```bash
meson setup build -Denable-transformer=true
ninja -C build
# model tests on the CPU engine (meson sets the library paths)
NNTR_ENGINE=cpu meson test -C build --print-errorlogs unittest_causallm_models
# the HTP registries and the int8 KV attention
meson test -C build unittest_hexkl_attn_f16_plan unittest_hexkl_kv_tiles_f16 unittest_hexkl_kv_q \
  unittest_hexkl_attn_q_plan unittest_hvx_softmax_q unittest_cpu_kv_q_attention
# kernel host checks (scalar stand-ins for the DSP; one to two hours at -O2)
bash test/htp/host/run_host_checks.sh
```

`NNTR_ENGINE=cpu` matters when the tree was configured with
`-Denable-opencl=true`: that build defaults to the GPU engine, which has no
layers on a machine without an OpenCL device. The model tests create
scratch directories under `$TMPDIR`; point it somewhere writable if `/tmp`
holds stale `nntrainer_causallm_tiny_*` directories owned by another user.

### 4.2 Android native libraries and the app

Export the tool locations once per shell (adjust the paths):

```bash
export ANDROID_NDK=$HOME/android-ndk-r30
export JAVA_HOME=$HOME/jdk/jdk-17.0.20.1+1
export ANDROID_HOME=$HOME/android-sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export PATH=$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH

export HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1
source "$HEXAGON_SDK_ROOT/setup_sdk_env.source"   # sets DEFAULT_HEXAGON_TOOLS_ROOT
export HEXKL_ROOT=$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon
export HEXKL_SDK_VER=6.4.0.2
export HEX_ARCH=v81
```

Then build:

```bash
# native libraries, CLI tools and the AAR/APK
bash Applications/CausalLM/build_android.sh --htp --app
# the instrumentation test APK (optional, for the device tests in section 8)
( cd Applications/CausalLM/Android && ./gradlew -PnntrainerNdkPath=$ANDROID_NDK :SampleTestAPP:assembleDebugAndroidTest )
```

What the script does, in order: generates the FastRPC stub from
`test/htp/nntr_hvx.idl`, builds the DSP skel `test/htp/build/libnntr_hvx_skel.so`
with the Hexagon toolchain, cross-builds libnntrainer for arm64 with
`-Denable-htp=true`, builds the CausalLM app graph (`nntr_causallm`,
`quick_dot_ai_test`, `libcausallm.so`, `libquick_dot_ai_api.so`), stages the
runtime libraries, assembles the AAR and the APK with Gradle, and finally runs
`Applications/CausalLM/Android/verify_native_package.py`, which checks that
the APK carries the v81 skel and `libsdkl.so` with the expected hashes.

Outputs:

| File | Where |
|---|---|
| `nntr_causallm`, `quick_dot_ai_test`, `libcausallm.so`, `libquick_dot_ai_api.so` | `builddir_app/htp/` |
| `libnntrainer.so`, `libccapi-nntrainer.so` | `builddir/android_build_result/lib/arm64-v8a/` |
| `libnntr_hvx_skel.so` | `test/htp/build/` |
| `libsdkl.so` | `$HEXKL_ROOT/lib/6.4.0.2/armv8_android26/` |
| `QuickDotAI-debug.aar` | `Applications/CausalLM/Android/QuickDotAI/build/outputs/aar/` |
| `SampleTestAPP-debug.apk` | `Applications/CausalLM/Android/SampleTestAPP/build/outputs/apk/debug/` |
| `SampleTestAPP-debug-androidTest.apk` | `Applications/CausalLM/Android/SampleTestAPP/build/outputs/apk/androidTest/debug/` |

A full build takes about 15 minutes on 8 cores. `--htp` and `--qnn` are
separate variants; the HTP APK cannot run QNN models.

## 5. Prepare the model pack

Convert the Hugging Face checkpoint to the fused layout:

```bash
python3 Applications/CausalLM/res/gemma4/htp_weight_converter.py \
  --model_path /path/to/gemma-4-26B-A4B-it \
  --output_name /path/to/pack/nntr_gemma4_26b_a4b_qs2cx_fcqs4cx.bin
```

The converter writes the weight file and a manifest beside it with the SHA-256
and the two keys the loader requires. The validated file is 7,540,724,856
bytes with SHA-256
`9a4c16e33a9eede48bd745d392fc8b868bf3b08b1362c1a2390745ca8d8c4a06`.

Assemble the pack directory. Its name is fixed, because the app and the C API
look for it under the model base path:

```text
gemma4-26b-a4b-htp-w4a32/
  nntr_gemma4_26b_a4b_qs2cx_fcqs4cx.bin   # from the converter
  config.json                             # from the checkpoint
  tokenizer.json, tokenizer_config.json   # from the checkpoint
  nntr_config.json                        # from Applications/CausalLM/res/gemma4/htp_chat/
  generation_config.json                  # from Applications/CausalLM/res/gemma4/htp_chat/
  chat_template.jinja                     # from Applications/CausalLM/res/gemma4/htp_chat/
```

The three preset files are the interactive configuration: the official chat
template with the empty thought channel, EOS ids `1, 106, 50`, a 1,280-token
context and the default summarization prompt. The keys in `nntr_config.json`
that matter:

| Key | Value | Meaning |
|---|---|---|
| `gemma4_execution_profile` | `htp_fused` | selects the fused HTP model instead of the default Gemma4 model |
| `gemma4_weight_format` | `htp_fused_v1` | the only layout the loader accepts |
| `moe_layer_dtype` / `fc_layer_dtype` | `QS2CX_WH` / `QS4CX` | expert and dense weight formats |
| `moe_layer_dtype_per_layer` | `{"28": "QS4CX_WH_AFFINE", "29": "QS4CX_WH_AFFINE"}` | optional; overrides `moe_layer_dtype` for the listed MoE layers (the rotated ternary checkpoint keeps layers 28 and 29 in the 4-bit affine container, the rest `TL2CX_WH`) |
| `moe_hadamard_signs` | `rotation_signs.json` | optional, with `moe_hadamard`; a JSON beside the model (`{"format": "nntr_hadamard_signs_v1", "in": [±1 × hidden], "down": [±1 × moe_intermediate]}`): the checkpoint's per-dimension signs applied after the Hadamard, the same for every MoE layer. Without it a seeded (sign-flipped) rotation gives wrong text; with it the rotation runs on the HVX path (no HMX fp16 variant) |
| `moe_hadamard` / `moe_hadamard_scope` | `both_hmx` / `all` | optional; the online Hadamard rotation of the expert inputs a pre-rotated checkpoint needs (`rotation_meta.json`), with the words of `NNTR_MOE_HADAMARD` (off, up, down, both, both_scalar, up_hmx, down_hmx, both_hmx) and `NNTR_MOE_HADAMARD_SCOPE` (2bit, all); the environment wins when set. `all` rotates every expert layer and never the dense FFN. Unrotated weights with the rotation on, or the reverse, give wrong text |
| `embedding_dtype` / `lmhead_dtype` | `Q4_0` | embedding table and head; an untied head (`tie_word_embeddings: false` in `config.json`) may be `QS4CX`, and then runs on the DSP as WH slices for both the prefill's last row and decode; the slices are cached beside the model as `<model file>.lmhead_wh` after the first load (delete it after re-quantizing the head) |
| `attention_kv_dtype` | `q8` | int8 KV cache (required) |
| `moe_engine`, `attention_engine`, `attn_proj_engine`, `dense_ffn_engine` | `htp` | which blocks run on the DSP |
| `lmhead_engine` | `cpu` | the tied Q4_0 head stays split: the DSP runs decode, the CPU the prefill's last row; a `QS4CX` untied head ignores this key and runs on the DSP throughout |
| `moe_cache_experts` | `16` | experts kept resident per layer |
| `init_seq_len` / `max_seq_len` | `1280` | context |
| `num_to_generate` | `127` | generation budget per call |

Push the pack to the phone (the app and the tools read it from external
storage):

```bash
adb shell mkdir -p /sdcard/Download/aistudio-mobile/models
adb push gemma4-26b-a4b-htp-w4a32 /sdcard/Download/aistudio-mobile/models/
```

## 6. Run the app (SampleTestAPP)

1. Install the APK. If a SampleTestAPP signed by someone else is already on
   the phone, uninstall it first; otherwise `adb install` fails with
   `INSTALL_FAILED_UPDATE_INCOMPATIBLE`.

   ```bash
   adb install -r -t Applications/CausalLM/Android/SampleTestAPP/build/outputs/apk/debug/SampleTestAPP-debug.apk
   adb shell appops set com.example.sampletestapp MANAGE_EXTERNAL_STORAGE allow
   ```

   The second line grants the "all files" access the app needs to read the
   pack; you can also grant it under Settings > Apps > SampleTestAPP. Android
   shows a one-time warning that the app is not built for 16 KB pages; the
   tested phone uses 4 KB pages and runs it.

2. Open the app. In the model tab pick **FAMILY** `gemma4-26b-a4b`,
   **RUNTIME** `NATIVE`, **BACKEND** `NPU` (the only backend this family
   offers; the entry is labelled `[HTP]`). Quantization is `W4A32`.
3. Set **MODEL BASE PATH** to `/sdcard/Download/aistudio-mobile/models` (the
   directory that contains the pack, not the pack itself).
4. Tap **Load model**. The status line reads `Loaded gemma4-26b-a4b-htp ...`
   after about ten seconds. The OpenAI and Chat inputs are pre-filled with the
   summarization prompt from the preset.
5. Tap **Run (streaming)** in the OpenAI tab, or **+ Open** and then **▶** in
   the Chat tab. The answer streams in and stops at the first end-of-turn
   marker, which the app shows as `<turn|>`. The default prompt produces a
   three-sentence summary of the Ardley text (109 tokens).
6. **Stop** cancels at the next token boundary; **Unload** releases the DSP
   session; **Reload** or a changed base path loads a fresh handle. A bad base
   path shows `Load failed: [MODEL_LOAD_FAILED] ...` with the native reason.

Only one HTP model can be loaded per process, each generation starts a fresh
prompt (no conversation history), and the public API does not expose KV
snapshot or resume. The native CLI has a separate
[KV snapshot workflow](gemma4-htp-kv-cache-s25u.md); its historical S25
measurements are distinct from current S26 throughput results.

## 7. Run from the command line

The same binaries run without the app. Stage them in a directory on the
phone and run `nntr_causallm` with the pack:

```bash
D=/data/local/tmp/htp
adb shell mkdir -p $D
for f in builddir_app/htp/nntr_causallm builddir_app/htp/libcausallm.so builddir_app/htp/libquick_dot_ai_api.so \
         builddir/android_build_result/lib/arm64-v8a/libnntrainer.so builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so \
         $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
         test/htp/build/libnntr_hvx_skel.so $HEXKL_ROOT/lib/6.4.0.2/armv8_android26/libsdkl.so; do adb push $f $D/; done
adb shell chmod 755 $D/nntr_causallm

adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D \
  ./nntr_causallm /sdcard/Download/aistudio-mobile/models/gemma4-26b-a4b-htp-w4a32"
```

`ADSP_LIBRARY_PATH` must point at the directory holding
`libnntr_hvx_skel.so`. Nothing else has to be set: the runtime configuration
the model was validated with (the resident decode graph, 16 resident experts
per layer, the int8 decode attention, the dense projections in the DSP arena,
the host copies dropped once the DSP holds them, four DMA queues for the
expert feed, the ARM's next-layer expert prediction with its read-ahead, and
the prefill scratch released after the first token, and the expert pool grown
to 34 experts a layer after the first token where the DSP's address space
allows) is built into the library
(`nntrainer/tensor/htp_fixed_opts.h`). Environment controls remain available:
`NNTR_HTP_PAGE_CACHE` unset keeps the mmap loader; `0` selects direct I/O.
The fixed-capacity measurement preset sets `moe_cache_experts=16` in the
pack, `NNTR_MOE_CACHE_GROW=0` and `NNTR_MOE_PRELOAD_EXPERTS=128`.
`NNTR_MOE_SERVER_DONE=1` enables the optional completion hint; it is off
by default. See the [code guide](gemma4-htp-code-guide.md#7-runtime-configuration-and-options)
for the controls and the TL2 guide for the conditions of each measurement.

With no prompt argument the CLI uses `sample_input` from `nntr_config.json`.
Pass your own prompt as the second argument; the chat template from the pack
is applied to it. Add `NNTR_HTP_TOKEN_TRACE=/data/local/tmp/htp/trace.json`
to get the input and output token ids as JSON, which is how the numbers in
section 9 were compared.

The public C API has its own device check that loads, generates, cancels,
unloads and reloads through the same calls the app makes:

```bash
adb push builddir_app/htp/quick_dot_ai_test $D/
adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D QUICKAI_NATIVE_LIB_DIR=$D QDA_HTP_LIFECYCLE=1 \
  ./quick_dot_ai_test gemma4-26b-a4b-htp 'I bought a red car yesterday' true W4A32 1 /sdcard/Download/aistudio-mobile/models"
```

It prints one `[HTP-API-TEST] PASS ...` line per check (29 in total).

## 8. Tests

| Level | Command | What it proves |
|---|---|---|
| Host, kernels | `bash test/htp/host/run_host_checks.sh` | the DSP kernels' scheduling, layouts and arithmetic on scalar stand-ins; ends with `ALL CHECKS PASS` |
| Host, model | `unittest_causallm_models` (section 4.1) | the fused FP32 tiny model matches its pinned reference; bad packs are rejected before inference |
| Host, registries | `meson test -C build unittest_hexkl_attn_f16_plan unittest_hexkl_kv_tiles_f16 unittest_hexkl_kv_q unittest_hexkl_attn_q_plan unittest_hvx_softmax_q unittest_cpu_kv_q_attention` | attention plans, KV tile registries, the int8 KV attention and the integer softmax |
| Device, C API | `quick_dot_ai_test ... QDA_HTP_LIFECYCLE=1` (section 7) | load, repeat parity, cancellation, unload, reload, invalid path |
| Device, app | the two instrumentation tests below | the app's controls, prompt format and token traces |

The app tests need the model pack on the phone, the messages file staged
somewhere the app can read, and both APKs installed:

```bash
adb push Applications/CausalLM/Android/SampleTestAPP/src/main/res/raw/gemma4_moe_messages.json /sdcard/Download/aistudio-mobile/messages.json
adb install -r -t Applications/CausalLM/Android/SampleTestAPP/build/outputs/apk/androidTest/debug/SampleTestAPP-debug-androidTest.apk
R=com.example.sampletestapp.test/androidx.test.runner.AndroidJUnitRunner
adb shell am instrument -w -r -e class 'com.example.sampletestapp.HtpAppLifecycleTest#chatUsesGemma4PromptFormat' \
  -e htpModelBase /sdcard/Download/aistudio-mobile/models -e htpMessages /sdcard/Download/aistudio-mobile/messages.json $R
adb shell am instrument -w -r -e class 'com.example.sampletestapp.HtpAppLifecycleTest#loadGenerateStopRepeatReloadAndReportInvalidPath' \
  -e htpModelBase /sdcard/Download/aistudio-mobile/models -e htpMessages /sdcard/Download/aistudio-mobile/messages.json $R
```

Each run leaves its token traces and outputs in the app's private files
directory (`htp-chat-validation/`, `htp-app-validation/`); read them with
`adb exec-out run-as com.example.sampletestapp tar c files | tar x`.

## 9. Results

All numbers below are from the Galaxy S26 Ultra described in section 2 with
the validated 7.5 GB pack. They are regression observations: the phone was
not thermally controlled and single runs vary by 10-20 %.

### 9.1 This branch, measured 2026-10-13

Summary prompt (1,028 input tokens, 109 output tokens, greedy, stops at the
end of turn), `nntr_causallm` with the runtime options of section 7:

| Run | Prefill | Time to first token | Decode (108 tokens) | Steady decode | Peak memory (host anon + DSP) |
|---|---|---|---|---|---|
| this branch | 1,812 ms (567 tok/s) | 1.86 s | 8,813 ms (12.3 tok/s) | 73.7 ms/token | 4.2 GB |
| pre-rebase head of this PR, same phone, same hour | 1,779 ms (578 tok/s) | 1.81 s | 9,983 ms (10.8 tok/s) | 82.2 ms/token | 5.1 GB |

The two runs produced the same 109 token ids, so the difference is run-to-run
noise, not the code. The PR's own validation (three alternating runs each,
1,024 input and 128 output tokens with EOS disabled) recorded prefill medians
of 552.9 and 558.6 tok/s and decode of 13.5 and 14.1 tok/s.

Short prompt through the C API (19 input tokens, 117 output tokens): first
call 1.6 s to the first token (this includes building the resident graph),
decode 16.1-16.6 tok/s; the repeat call reproduces the same 441 bytes.

In the app, the default summary (Chat or OpenAI tab) takes 11-18 s from tap
to the end-of-turn marker depending on how warm the phone is; the short
prompt takes about 7 s.

Correctness, same day: the token ids are identical across the CLI, the C API
test and the app, identical to the pinned source PR's build, and the app's
seven full generations across cancel, unload, reload and a failed load all
match. On the host, the fused FP32 tiny model matches its reference logits and
the CPU and GPU paths of the other models produce the same text as before the
HTP series was merged.

### 9.2 The later PR #4416 commits, measured on their source branch

The 18 commits after the pinned revision are part of this branch. Their
authors measured them on the same phone model on the source branch
(`docs/measurements/s26u-next-round-handoff.md` and
`docs/measurements/pr4416-memory-87f8580/` in PR #4416); the figures
describe the code this branch now carries:

| Prompt | Prefill | Decode, last 64 tokens |
|---|---|---|
| 1,024 tokens | 1,650-2,281 ms | 12.1-14.9 tok/s |
| 2,048 tokens | 3,141-3,436 ms | 11.2-15.3 tok/s |
| 4,096 tokens | 7,068-7,518 ms | 11.9-13.3 tok/s |

The newer head also bounds the model's page-cache use, measured with a
2,048-token prompt: peak memory 8.4 GB (up to 3.5 GB of it the model's page
cache) at 645 tok/s prefill and 9.9 tok/s decode, or 3.6 GB with the page
cache bounded at 682 tok/s prefill and 7.0 tok/s decode
(`NNTR_HTP_PAGE_CACHE=0`). On this branch, built from the same head state,
the summary prompt of section 9.1 gives 1,633 ms (630 tok/s) prefill and
13.8 tok/s (99 tokens in 7.2 s) decode, and the 100 generated token ids are identical to a
build of the PR #4416 head run with the same options; they differ from the
pinned revision's 109 tokens at the third token, as the growing int8 scales
change the arithmetic.

## 10. When something goes wrong

| Symptom | Cause and fix |
|---|---|
| `HTP native_lib_dir is missing libnntr_hvx_skel.so` | `ADSP_LIBRARY_PATH` / `nativeLibDir` does not contain the skel |
| `Gemma4 HTP requires NPU / W4A32` | the app was asked for another backend or quantization |
| `Load failed: [MODEL_LOAD_FAILED] Failed to open file ...` | the base path does not contain `gemma4-26b-a4b-htp-w4a32/` |
| `Only one active QuickDotAI HTP model is supported` | unload the first handle before loading another |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | an APK with another signature is installed; uninstall it |
| host model tests fail with `Key is not found ... fully_connected` | an OpenCL build defaults to the GPU engine; set `NNTR_ENGINE=cpu` |
| `verify_native_package.py` fails | the APK was assembled without `jniLibs.useLegacyPackaging=true` or the skel was stripped |
| 16 KB page warning on install | expected on this build; it is not validated for 16 KB-page devices |
