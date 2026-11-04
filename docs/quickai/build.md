# Build from source

How to build the Android native libraries, the command-line tools, the
QuickDotAI library (AAR) and the Quick.AI app (APK) for the Hexagon NPU, and
how to build and test the host side on a Linux PC. One script,
`Applications/CausalLM/build_android.sh`, drives the whole Android build.

## What you need

| Item | Notes |
|---|---|
| Linux host | Ubuntu 24.04 with about 60 GB of free disk is known to work. Meson, Ninja, CMake, Python 3, a C/C++ toolchain. |
| Android NDK | r30 is the tested version. Set `ANDROID_NDK` (or `NDK_ROOT`). |
| Rust toolchain | `cargo` and `rustc`, used once to build the tokenizer library for Android from the in-tree crate. |
| JDK 17 and an Android SDK | Only for `--app` (Gradle): platform 36, build-tools 36.0.0, cmake 3.22.1, platform-tools. Gradle finds them through its own configuration; the repository's Gradle files take only the NDK path. |
| Hexagon SDK | 6.4.0.1 with Hexagon Tools 19.0.04 is the tested version. From Qualcomm, not part of this repository. |
| HexKL | 1.0 beta 2, whose `lib/6.4.0.2` directory holds `hexagon_toolv19_v81` and `armv8_android26`. From Qualcomm. |
| Phone | A Snapdragon device with a Hexagon **v81** DSP (Galaxy S26 Ultra, SM8850) running Android 16 with 4 KB pages, USB debugging on. Other DSP generations are refused by the build. |
| Model pack | See [Model packs](model-packs.md); the pack is not part of the build. |

Get the source:

```bash
git clone https://github.com/nntrainer/nntrainer.git
cd nntrainer
git submodule sync && git submodule update --init --recursive --depth 1
```

The Android script initialises the submodules it needs (`xgrammar` and its
`dlpack`, the engine's `subprojects/`) if you skip this step.

## Android: libraries, tools, AAR and APK

Set the environment once per shell. The Hexagon SDK ships a script that sets
`HEXAGON_SDK_ROOT` and `DEFAULT_HEXAGON_TOOLS_ROOT`:

```bash
source /path/to/Hexagon_SDK/6.4.0.1/setup_sdk_env.source
export HEXKL_ROOT=/path/to/hexkl_addon      # contains lib/6.4.0.2/...
export HEXKL_SDK_VER=6.4.0.2                # default; the lib/<ver> subtree to use
export HEX_ARCH=v81                         # required; only v81 is accepted
export ANDROID_NDK=/path/to/android-ndk-r30

cd Applications/CausalLM
./build_android.sh --htp --app
```

`--htp` selects the native Hexagon variant; `--app` adds the Gradle build of
the AAR and the APK. The script stops with a clear message if a variable is
missing (`--htp requires HEXKL_ROOT`), if the DSP target is not v81
(`Error: the Gemma4 HTP runtime requires HEX_ARCH=v81.`) or if `--htp` is
combined with `--qnn` (`Error: --htp and --qnn select separate native
variants.`).

What the script does, in order:

1. Generates the FastRPC stub from `nntrainer/tensor/htp_backend/dsp/nntr_hvx.idl`
   (`generate_stub.sh`) and builds the DSP skel `libnntr_hvx_skel.so` with
   `hexagon-clang` and HexKL (`dsp/build.sh`). The skel build refuses
   undefined symbols the DSP image will not provide and checks that the ELF
   flags say v81.
2. Restores `json.hpp`, builds the tokenizer static library for arm64 if it
   is missing (`build_tokenizer_android.sh`, Rust), and builds the nntrainer
   engine for Android (`tools/package_android.sh`) into
   `builddir/android_build_result/`.
3. Configures and builds the CausalLM application graph for the variant into
   `builddir_app/htp/`.
4. With `--app`: stages the runtime libraries into
   `Android/QuickDotAI/prebuilt_libs/`, runs Gradle
   (`:QuickDotAI:assembleDebug :QuickDotAIAPP:assembleDebug`) and verifies
   the packages with `Android/verify_native_package.py`, which checks that
   every staged library is in the AAR and the APK, that the packaged skel is
   byte-identical to the staged one and is a v81 image, and that no `libQnn*`
   or `libcdsprpc.so` was packaged. It writes
   `<APK>.native-manifest.json` with the SHA-256 of each library.

Outputs:

| Artifact | Path |
|---|---|
| `nntr_causallm`, `quick_dot_ai_test`, `nntr_quantize`, `nntr_safetensors_info` | `builddir_app/htp/` (repository root) |
| `libcausallm.so`, `libquick_dot_ai_api.so` | `builddir_app/htp/` (repository root) |
| `libnntrainer.so`, `libccapi-nntrainer.so` | `builddir/android_build_result/lib/arm64-v8a/` |
| `libnntr_hvx_skel.so` (DSP skel) | `nntrainer/tensor/htp_backend/dsp/build/` |
| `libsdkl.so` (HexKL ARM runtime) | `$HEXKL_ROOT/lib/$HEXKL_SDK_VER/armv8_android26/` |
| `libc++_shared.so` | the NDK's `sysroot/usr/lib/aarch64-linux-android/` |
| AAR | `Applications/CausalLM/Android/QuickDotAI/build/outputs/aar/QuickDotAI-debug.aar` |
| APK | `Applications/CausalLM/Android/QuickDotAIAPP/build/outputs/apk/debug/QuickDotAIAPP-debug.apk` |

The seven runtime libraries staged into the AAR for the HTP variant are
`libnntrainer.so`, `libccapi-nntrainer.so`, `libcausallm.so`,
`libquick_dot_ai_api.so`, `libc++_shared.so`, `libsdkl.so` and
`libnntr_hvx_skel.so`. `libcdsprpc.so` is never packaged; the phone's vendor
copy is used.

The ARM libraries, the generated stub and the DSP skel form one wire
contract. Always rebuild and deploy them together; an old skel with a new
library refuses to load the decode graph rather than misbehaving.

Other options:

| Option | Effect |
|---|---|
| `--install` | Pushes the four executables and the runtime libraries to `/data/local/tmp/Quick.AI` on the phone and writes `run_causallm.sh` / `run_test.sh` wrappers there that export `LD_LIBRARY_PATH`, `ADSP_LIBRARY_PATH` and `NNTR_NUM_THREADS`. With `--app`, also installs the APK. Needs `adb`; set `ANDROID_SERIAL` when more than one device is attached. |
| `--nntr-threads=N` | CPU compute threads baked into the engine and the wrappers. Default 8 for the HTP variant. |
| `--cache` | Reuses a compatible engine build in `builddir/android_build_result`. |
| `--clean` | Recreates `builddir_app/htp` before configuring. |
| `NNTR_HTP_DIAGNOSTICS=1` | Compiles the diagnostic DSP probes into the skel. Off by default; the production skel keeps their entry points as stubs. |

The instrumentation test APK is not built by the script:

```bash
cd Applications/CausalLM/Android
./gradlew -PnntrainerNdkPath=$ANDROID_NDK :QuickDotAIAPP:assembleDebugAndroidTest
# -> QuickDotAIAPP/build/outputs/apk/androidTest/debug/QuickDotAIAPP-debug-androidTest.apk
```

CPU-only and QNN packages are built by the same script without `--htp`
(`./build_android.sh --app`, `./build_android.sh --qnn --app`); they are
described in the [CausalLM README](../../Applications/CausalLM/README.md#4-android-build--test).
HTP and QNN are separate builds and cannot be combined in one package.

## Install the app

```bash
adb install -r -t Applications/CausalLM/Android/QuickDotAIAPP/build/outputs/apk/debug/QuickDotAIAPP-debug.apk
adb shell appops set com.example.quickdotaiapp MANAGE_EXTERNAL_STORAGE allow
```

The second line grants the "all files" access the app needs to read a model
pack from `/sdcard`; you can also grant it under Settings > Apps > Quick.AI.
If a Quick.AI build signed with another key is already installed,
`adb install` fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`; uninstall it
first. Android shows a one-time warning that the app is not built for 16 KB
pages; see [Troubleshooting](troubleshooting.md#known-limits).

## Host build and tests (no phone)

The host build compiles the same `libnntrainer` and the CausalLM models for
x86 and runs the tests that need no device:

```bash
meson setup build -Denable-transformer=true
ninja -C build
NNTR_ENGINE=cpu meson test -C build --print-errorlogs unittest_causallm_models
meson test -C build unittest_hexkl_attn_f16_plan unittest_hexkl_kv_tiles_f16 unittest_hexkl_kv_q \
  unittest_hexkl_attn_q_plan unittest_hvx_softmax_q unittest_cpu_kv_q_attention
bash test/htp/host/run_host_checks.sh
```

| Command | What it proves |
|---|---|
| `unittest_causallm_models` | The fused FP32 tiny Gemma4 HTP model matches its pinned reference on the CPU engine; packs without the HTP profile, with unsupported dtypes or with truncated weights are rejected before inference; LFM2 model contracts. `NNTR_ENGINE=cpu` matters on a build with OpenCL enabled, whose default engine is the GPU. |
| the six `unittest_hexkl_*` / `unittest_hvx_*` / `unittest_cpu_*` targets | Attention plans, KV tile registries, the integer softmax and the CPU mirror of the int8 KV attention. They build in every host configuration; no Hexagon SDK needed. |
| `run_host_checks.sh` | The DSP kernel sources compiled on scalar stand-ins: expert routing and prefetch ownership, DMA ring reuse, worker pool lanes, mailbox and token-driver protocol, the resident decode graph, TL2 packing and conversion, affine correction, static encodings, and more, each against a scalar reference. It stops at the first failing check (`set -e`); reaching the end without an error is the pass. It does **not** verify the hardware arithmetic; the device does that. Optional extras: with `DEFAULT_HEXAGON_TOOLS_ROOT` set it also runs the real HVX GEMV and quantizer kernels under the Hexagon tools' emulation. |

Two meson options compile the HTP sources on the host without a phone:

| Option | Meaning |
|---|---|
| `-Dhtp-inproc=true -Dhexagon-sdk-root=<path>` | The DSP skel sources are compiled into `libnntrainer.so` against stand-ins (`test/htp/host/{inproc,standin,hvx_emu,stub}`). A compile gate, not a benchmark. |
| `-Denable-htp=true -Dhexkl-sdk-root=<path to hexkl_addon>` | The real FastRPC backend, as the Android build uses it. Linking needs the HexKL libraries; running needs the DSP. |

## Device tests

| Test | Command | What it proves |
|---|---|---|
| C API lifecycle | `quick_dot_ai_test` with `QDA_HTP_LIFECYCLE=1`, see [Command line](cli.md#the-c-api-lifecycle-test) | load, repeat parity, cancellation from the callback and from another thread, unload during generation, reload, invalid packs |
| App lifecycle | `HtpAppLifecycleTest` instrumentation, see [Using the app](app.md#device-tests) | the app's controls and prompt format, with token traces per step |
| Continuation | `-Denable-htp-device-check=true` in the application graph builds `htp_continuation_check PACK gemma\|lfm rows\|chunks OUTPUT_PREFIX` | resident continuation against single-row feeding |

The kernel-level HVX unit tests under `test/unittest/unittest_hvx_*.cpp` are
sources for ndk-build; only `unittest_hvx_add` has a module in
`test/jni/Android.mk` today.
