# Building nntrainer with the MediaTek NeuroPilot (Neuron Runtime) backend

This is the counterpart to [QNN_BUILD.md](QNN_BUILD.md) for MediaTek's
NeuroPilot NPU stack. The two backends are structurally similar (a
pluggable `nntrainer::Context` built as its own shared library, dlopen'd by
`Engine` at startup), but Neuron's build is simpler: there is no
configure-time SDK vendoring step and no per-SoC library selection.

## 1. Obtain the NeuroPilot SDK

Get a NeuroPilot SDK from MediaTek (e.g.
`neuropilot-sdk-premium-<version>-build<date>`). It ships:

- `neuron_sdk/<soc>/lib/libneuron_runtime.so` — the runtime, one copy per
  supported SoC (e.g. `mt6989`, `mt6991`, `mt6993`, `mt6899`, `mt6881`).
- `neuron_sdk/dummy/lib/libneuron_runtime.so` — a no-hardware stub, useful
  for exercising the plugin/registration path on a host with no device.
- `neuron_sdk/host/bin/ncc-tflite` — the offline compiler that turns a
  TFLite model into a `.dla` (Deep Learning Archive) binary.
- `neuron_sdk/<soc>/include/neuron/api/{RuntimeAPI.h,Types.h}` — the C API
  this backend targets. These are already vendored into this repo (see
  `nntrainer/neuron/vendor/README.md`) — you do **not** need the SDK just to
  build nntrainer's neuron plugin.

You do need the SDK to:
- Compile a model into a `.dla` with `ncc-tflite --platform-config <soc>`.
- Push the SoC-matching `libneuron_runtime.so` to the device.

### Tested versions

- NeuroPilot Premium 9.0.9 (neuron 9.3.1), target SoC `mt6991`.
- Confirmed working end-to-end on real mt6991 hardware with a quantized
  UINT8 MobileNetV2 `.dla`; output matched MediaTek's `neuronrt` reference
  tool byte-for-byte. See §4 below for a compatibility caveat this build
  uncovered.

## 2. Configure the build

```
export ANDROID_NDK=/path/to/android-ndk-r26d
export PATH=$ANDROID_NDK:$PATH

meson setup builddir \
  -Dplatform=android \
  -Denable-neuron=true \
  -Darm-arch=armv8.2-a \
  -Denable-tflite-backbone=false \
  -Denable-tflite-interpreter=false

ninja -C builddir
```

No SDK root option is needed (contrast with `-Dqnn-sdk-root=`) — the plugin
only needs the two vendored headers, which are already in the tree.

The extra flags are not optional in practice:

- **`-Darm-arch=armv8.2-a`** — this option defaults to `none`, which emits no
  `-march` flag, and the bundled KleidiAI micro-kernels then fail to compile
  with `#error "Dotprod extension required to compile this micro-kernel"`.
  `armv8.2-a` expands to `-march=armv8.2-a+fp16+dotprod+i8mm`. The mt6991 is
  armv9, so `-Darm-arch=armv9.2-a` also works; it is unrelated to Neuron but
  the build does not succeed without one of them.
- **`-Denable-tflite-*=false`** — otherwise meson requires a vendored
  TensorFlow-Lite include tree that a Neuron-only build has no use for.

Verify the two artifacts landed:

```
ls builddir/jni/arm64-v8a/libneuron_context.so
ls builddir/jni/arm64-v8a/nntrainer_neuron_smoke
```

> Incremental note: the ndk-build step is wrapped in a meson `custom_target`
> whose declared output is the `arm64-v8a` *directory*, so ninja treats it as
> up to date once that directory exists. After editing `jni/Android.mk.in` or
> the plugin sources, either build from a fresh builddir or re-run ndk-build
> directly (`cd builddir/jni && ndk-build NDK_LIBS_OUT=$PWD -j$(nproc)`).

## 3. What gets built

With `enable-neuron=true`:

- `nntrainer/neuron/meson.build` adds the plugin's sources/includes to the
  ndk-build module list (`MESON_NEURON_SRCS` / `MESON_NEURON_INCS` in
  `jni/meson.build`).
- `jni/Android.mk.in`'s `neuron_context` module builds
  `libneuron_context.so`, linked against `libnntrainer.so`, compiled with
  `-DENABLE_NEURON=1 -DPLUGGABLE=1`.
- `jni/Android.mk.in`'s `nntrainer_neuron_smoke` module builds the on-device
  smoke-test executable. This lives in the Android.mk rather than in
  `Applications/CausalLM/neuron/meson.build` because the android build skips
  the meson `Applications/` tree entirely ("android app is not supported for
  now, building app skipped"); the meson target serves desktop builds only.
- `Engine::add_default_object()` (`engine.cpp`) tries
  `registerContext("libneuron_context.so", "")` under `#if ENABLE_NEURON`,
  non-fatally (a warning, not a crash, if the plugin can't be loaded — same
  behavior as the QNN path).
- The plugin registers exactly one layer type, `neuron_graph`, which wraps
  execution of one precompiled `.dla` network via `NeuronRuntime_*`.
- A layer routes to this backend via `engine=neuron` (see
  `ml::train::LayerComputeEngine::NEURON` /
  `props::ComputeEngineTypeInfo::EnumStr[]` — `"neuron"`).

## 4. Runtime: locating libneuron_runtime.so

Like QNN's `libQnnHtp.so`, `libneuron_context.so` does not link
`libneuron_runtime.so` at build time — it `dlopen`s it by name at
`NeuronContext::init()`. Make sure the **SoC-matching** copy (e.g.
`neuron_sdk/mt6991/lib/libneuron_runtime.so` for a Dimensity 9400 device) is
on `LD_LIBRARY_PATH` or alongside the executable in the app's native lib
directory. Mixing SoC copies will fail to load or produce wrong results.

For host-only testing without a device, two environment variables switch to
the SDK's no-hardware stub:

```
export QUICK_DOT_AI_NEURON_LIB=/path/to/neuron_sdk/dummy/lib/libneuron_runtime.so
export QUICK_DOT_AI_NEURON_NULL_DEVICE=1
```

> **A second, separate compatibility axis: DLA schema version.** Even with the
> correct SoC, `NeuronRuntime_loadNetworkFromFile` can reject a valid `.dla`
> with a generic `Cannot load network`, logged to logcat as
> `The DLA file is V<A>, mismatching runtime V<B>`. This is a version stamped
> by the `ncc-tflite` release that compiled the `.dla`, checked against what
> the deployed runtime's DLA parser understands — unrelated to the SoC or the
> Neuron Runtime API version (e.g. "9.3.x"). This is not a hypothetical: it
> is exactly what happened compiling a model with one SDK release and
> deploying against an older bundled runtime. If you hit it, either deploy a
> newer `libneuron_runtime.so` (matching the SDK that produced the `.dla`) via
> `QUICK_DOT_AI_NEURON_LIB`, or recompile the `.dla` with an `ncc-tflite`
> release the deployed runtime supports. `neuronrt -m hw -a model.dla -i
> input.bin` (shipped in the same SDK as `ncc-tflite`) is the fastest way to
> check a given `.dla`/runtime pairing independent of nntrainer entirely — see
> `NEURON_BACKEND_GUIDE.md` §8 and §11 for the full diagnosis path.

## 5. Compiling a model to .dla

Confirmed working invocation (the exact command used to produce the UINT8
MobileNetV2 `.dla` this backend was validated against, from an
already-quantized `.tflite`):

```
./ncc-tflite --platform-config mt6991 --relax-fp32 \
    ./mobilenet_v2.tflite -o ./model.dla --verbose
```

`--platform-config <soc>` replaced the deprecated `--arch` flag as of SDK
9.x. The headers are SoC-invariant, but the `.dla` itself is not — compile
once per target SoC. `--relax-fp32` and `--verbose` are confirmed applicable;
consult `ncc-tflite --help` for flags relevant to your own model (e.g.
quantizing a float `.tflite` at compile time, if it isn't already quantized).
Also keep in mind the DLA-schema-version caveat in §4: the SDK release used
here, not just the `--platform-config` value, matters for compatibility with
whatever runtime you deploy against.

## 6. Verifying the build

`nntrainer/Applications/CausalLM/neuron/nntrainer_neuron_smoke` (built as
part of `-Denable-neuron=true`, see §2) is the end-to-end check: it builds a
2-layer model (`input -> neuron_graph`), loads a `.dla`, runs inference, and
optionally compares against a golden file. Confirmed working example, for a
quantized UINT8 model:

```
export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH
./nntrainer_neuron_smoke model.dla \
    --in-shape=1:224:224:3 --in-dtype=UINT8 \
    --out-shape=1:1001:1:1 --out-dtype=UINT8 \
    --input=input.bin --dump=out.bin
```

For a genuine correctness check (not just "it ran"), compare `out.bin` against
`neuronrt -m hw -a model.dla -i input.bin`'s `output0.bin` byte-for-byte — see
`NEURON_BACKEND_GUIDE.md` §10.4 for details; this is exactly how the backend
was first confirmed correct on real hardware.

## 7. Why the SDK is not committed to this repository

Same rationale as QNN_BUILD.md §8: the NeuroPilot SDK is MediaTek's
proprietary software and cannot be redistributed here. Only the two small,
stable API headers are vendored, not the runtime libraries or the offline
toolchain.
