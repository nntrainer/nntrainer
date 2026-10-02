---
title: Test JNI
...

# Test

These unit tests are built with `ndk-build` for `arm64-v8a` (see `Application.mk`).
`Android.mk` here defines one module per test executable.

## Prerequisites

Build the library for Android first, so that the prebuilt shared libraries exist:

``` bash
./tools/package_android.sh -Denable-opencl=false
```

Then, from the repository root, three prerequisites apply.

**1. Prebuilt libraries.** `Android.mk` reads `libnntrainer.so` and
`libccapi-nntrainer.so` from `builddir/jni/<abi>/`. The command above already
builds them there, so there is nothing to do by hand -- but note that it *is* a
hard requirement: they are consumed through `PREBUILT_SHARED_LIBRARY`, which
aborts at parse time if the files are missing, so `Android.mk` cannot even be
parsed without a prior Android build of the library. (Do not confuse that path
with `builddir/android_build_result/lib/<abi>/`, which is the installed copy
taken *from* it.)

**2. googletest.** It is not vendored in this repository, but the NDK ships it.
Take it from the same NDK you build with:

``` bash
cp -R "$(dirname "$(readlink -f "$(command -v ndk-build)")")/sources/third_party/googletest" .
```

**3. The `-march` flag.** `package_android.sh` forwards the architecture flag to
meson as `-Darm-march`, where `ndk-build` never sees it, and `ARM_MARCH_FLAGS`
has no default of its own. Left unset, these modules build for the NDK default
(`armv8-a` on `arm64-v8a`), which still compiles: none of the sources
`Android.mk` builds uses FP16 NEON intrinsics, only the `_FP16` scalar type,
which AArch64 provides without `+fp16`. Pass it anyway as `MESON_ARM_MARCH`, so
the test modules target the same architecture as the library they link against.
The value below is the one `package_android.sh` uses by default: it reads
`tools/cross/android_armv8-2-a.json` unless `--arm-arch=` selects a different
config.

## Building

Build a single test module by naming it, which is usually what you want:

``` bash
cd test/jni
ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk \
  NDK_APPLICATION_MK=./Application.mk \
  MESON_ARM_MARCH="-march=armv8.2-a+fp16+dotprod+i8mm" \
  unittest_nntrainer_cpu_backend_fp16
```

Omit the module name to build every test module. Add
`MESON_ENABLE_OPENCL=1` if the library was built with `-Denable-opencl=true`, so
that the OpenCL prebuilt is linked in.

The resulting executables are under `obj/local/arm64-v8a/`. They are arm64
binaries, so running them requires an aarch64 device or emulator; `adb push` them
along with `libnntrainer.so`, `libccapi-nntrainer.so` and `libc++_shared.so`.

## In CI

`.github/workflows/android.yml` performs the above for
`unittest_nntrainer_cpu_backend_fp16` as a **compile-only** check, since the
runner is x86_64 and cannot execute an arm64 binary. Nothing in CI runs these
tests; see issue #4370.
