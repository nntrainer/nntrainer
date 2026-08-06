# CausalLM Model Unit Test Guide

`unittest_causallm_models` is the CausalLM reference test suite — 55 tests covering:

- **Tiny-model deterministic tests**: argmax, logit, and weight round-trip checks with random weights
- **FP32 differential tests**: compare nntrainer outputs against HuggingFace reference logits
- **Q4_0 differential tests**: on-the-fly quantization via `nntr_quantize`, then error-bound check

Reference fixtures are shipped as tarballs under `packaging/`:
- `causallm_reference_generation.tar.gz` — CausalLM model fixtures
- `causallm_reference_embedding.tar.gz` — embedding model fixtures

---

## x86 (Ubuntu host)

### Prerequisites

```bash
sudo apt-get install -y libgtest-dev meson ninja-build
```

### Build

```bash
# enable-fp16=false is required — the model unittest target is excluded from FP16 builds
meson setup builddir \
  -Denable-test=true \
  -Denable-app=true \
  -Denable-fp16=false \
  -Dopenblas-num-threads=1

cd builddir && ninja
```

Meson automatically extracts the fixture tarballs during build into
`test/unittest/models/causallm_reference/`.

### Run

```bash
# full suite
./Applications/CausalLM/run_unittest_x86.sh

# rebuild from scratch
./Applications/CausalLM/run_unittest_x86.sh --rebuild

# filter by model or test type
./Applications/CausalLM/run_unittest_x86.sh --filter '*Gemma4*'
./Applications/CausalLM/run_unittest_x86.sh --filter '*FP32MatchesHF*'
./Applications/CausalLM/run_unittest_x86.sh --filter '*Q40*'
```

The script handles meson setup, ninja build, fixture extraction, and sets
`NNTR_QUANTIZE_BIN` automatically.

### Expected result

```
[==========] 55 tests from 22 test suites ran.
[  PASSED  ] 55 tests.
```

### Common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `unittest_causallm_models` not built | `enable-fp16=true` or `enable-transformer=false` excludes the target | Script sets both flags automatically; run without `--rebuild` to reuse an existing builddir |
| Build fails with "opening dependency file: No such file or directory" | Parallel compiler race on fresh builddir | Run once without `--rebuild`; the incremental build will succeed |
| All differential tests SKIP | Fixtures not extracted | Script extracts automatically; verify `test/unittest/models/causallm_reference/` exists |
| All Q4_0 tests SKIP | `NNTR_QUANTIZE_BIN` not set | Script sets it automatically |

---

## Android (arm64-v8a)

### Prerequisites

| Item | Notes |
|------|-------|
| Android NDK r26d | `export ANDROID_NDK=/path/to/android-ndk-r26d && export PATH="$ANDROID_NDK:$PATH"` |
| Connected device | Verified via `adb devices` |
| `Applications/CausalLM/lib/libtokenizers_android_c.a` | arm64 build — copy from an existing workspace or build with `./Applications/CausalLM/build_tokenizer_android.sh` (requires Rust + network) |
| `Applications/CausalLM/json.hpp` | Run `./jni/prepare_encoder.sh builddir 0.2` if absent |

### Step 1: Build the canonical Android native artifacts

```bash
./Applications/CausalLM/build_android.sh
# core:  builddir_app/cpu/libcausallm.so
# tools: builddir_app/cpu/nntr_quantize
```

### Step 2: Vendor googletest (one-time)

```bash
cp -r $ANDROID_NDK/sources/third_party/googletest \
      Applications/CausalLM/jni/googletest
```

Skip if `jni/googletest/` already exists.

### Step 3: ndk-build

```bash
ndk-build \
  -C Applications/CausalLM/jni \
  NDK_PROJECT_PATH=Applications/CausalLM/jni \
  NDK_LIBS_OUT=Applications/CausalLM/jni/libs \
  NDK_OUT=Applications/CausalLM/jni/obj \
  APP_BUILD_SCRIPT=Applications/CausalLM/jni/Android.mk \
  NDK_APPLICATION_MK=Applications/CausalLM/jni/Application.mk \
  CAUSALLM_PREBUILT_LIB=../../../builddir_app/cpu/libcausallm.so \
  unittest_causallm_models \
  -j$(nproc)
# output: Applications/CausalLM/jni/obj/local/arm64-v8a/
```

### Step 4: Extract fixtures (if not already done)

```bash
tar xzf packaging/causallm_reference_generation.tar.gz -C test/unittest/models/
tar xzf packaging/causallm_reference_embedding.tar.gz  -C test/unittest/models/
# produces: test/unittest/models/causallm_reference/
```

### Step 5: Push to device

```bash
INSTALL=/data/local/tmp/nntr_causallm_test
TEST=Applications/CausalLM/jni/obj/local/arm64-v8a
NATIVE=builddir_app/cpu
ENGINE=builddir/android_build_result/lib/arm64-v8a

adb shell "mkdir -p $INSTALL/causallm_reference $INSTALL/tmp"

adb push $TEST/unittest_causallm_models $INSTALL/
adb push $NATIVE/nntr_quantize          $INSTALL/
adb push $NATIVE/libcausallm.so         $INSTALL/
adb push $ENGINE/libnntrainer.so        $INSTALL/
adb push $ENGINE/libccapi-nntrainer.so  $INSTALL/
adb push $ENGINE/libc++_shared.so       $INSTALL/
adb push test/unittest/models/causallm_reference $INSTALL/

adb shell "chmod 755 $INSTALL/unittest_causallm_models $INSTALL/nntr_quantize"
```

### Run (steps 1–6 automated)

```bash
# full suite
./Applications/CausalLM/run_unittest_android.sh

# reuse existing nntrainer build and ndk-build artifacts
./Applications/CausalLM/run_unittest_android.sh --cache

# filter by model or test type
./Applications/CausalLM/run_unittest_android.sh --filter '*Gemma4*'
./Applications/CausalLM/run_unittest_android.sh --cache --filter '*Q40*'
```

The script runs all six steps above end-to-end. Use `--cache` to reuse a
compatible canonical native build and the test-only ndk-build artifact.

### Expected result

```
[==========] 55 tests from 22 test suites ran.
[  PASSED  ] 55 tests.
```

### Required environment variables

| Variable | Purpose | If unset |
|----------|---------|----------|
| `LD_LIBRARY_PATH` | Locate `.so` files on device | Library load failure at startup |
| `TMPDIR` | Override `std::filesystem::temp_directory_path()` | Q4_0 tests fail to create temp dir |
| `NNTRAINER_CAUSALLM_FIXTURE_DIR` | Override fixture root path | Fixtures not found; all differential tests SKIP |
| `NNTR_QUANTIZE_BIN` | Path to on-device `nntr_quantize` binary | All Q4_0 tests SKIP |

### Common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| All differential tests SKIP | `NNTRAINER_CAUSALLM_FIXTURE_DIR` not set or fixtures not pushed | Verify `adb shell ls $INSTALL/causallm_reference/` |
| All Q4_0 tests SKIP | `NNTR_QUANTIZE_BIN` not set or binary not pushed | Verify `adb shell ls $INSTALL/nntr_quantize` |
| Linker error: `libcausallm.so` not found | Libraries not in `LD_LIBRARY_PATH` | Ensure all `.so` files are in `$INSTALL` |

