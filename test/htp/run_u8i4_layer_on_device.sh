#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Builds the DSP skel + the ARM gtest for the u8i4 AND u8i8 layer endpoints
# and runs them on a connected device. Both live in one gtest binary
# (unittest_hvx_mm_u8i4 despite the name -- it predates the u8i8 fixture),
# so one run covers both.
#
# What this checks, in order:
#   1. DSP skel compiles clean against HexKL (-Wall -Werror)
#   2. libnntrainer.so + the ARM gtest binary build for arm64-v8a
#   3. unittest_hvx_mm_u8i4 passes on-device -- the original u8i4 accuracy
#      harness (Shape1-4), the u8i4 layer-endpoint tests, and the u8i8
#      layer-endpoint tests
#   4. the reported layer_x4/harness speedup (u8i4) lands near the 1.7-2x
#      the cross-matmul prefetch was measured at, and the u8i8-vs-u8i4 ratio
#      is sane (both printed, not asserted -- see the ReportPerCallCost*
#      tests' comments)
#
# Required, with no default -- these are per-machine install paths, and a
# guessed one fails later and less clearly than an unset one does here:
#   HEXAGON_SDK_ROOT  Hexagon SDK install (its directory name is the version
#                     test/htp/build.sh looks for under HEXKL_ROOT/lib)
#   HEXKL_ROOT        hexkl_addon install. NOT the copy under
#                     Hexagon_SDK/*/addons/ -- that one has no v79 micro lib
#   ANDROID_NDK       NDK install used for the arm64-v8a build
#
# Optional:
#   DEVICE_TMP        on-device scratch dir (default below)
#   HEXAGON_TOOLS_VER Hexagon toolchain dir name under tools/HEXAGON_Tools;
#                     auto-detected when exactly one is installed

set -eu

: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT to your Hexagon SDK install}"
: "${HEXKL_ROOT:?set HEXKL_ROOT to your hexkl_addon path}"
: "${ANDROID_NDK:?set ANDROID_NDK to your NDK install}"
: "${DEVICE_TMP:=/data/local/tmp/htp_u8i4_layer_test}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_LOG="${RUN_LOG:-$(mktemp -t hvx_mm_u8i4_device_run.XXXXXX.log)}"

log()  { echo -e "\n\033[1;36m==> $*\033[0m"; }
fail() { echo -e "\033[1;31mFAILED: $*\033[0m" >&2; exit 1; }

for v in HEXAGON_SDK_ROOT HEXKL_ROOT ANDROID_NDK; do
  [ -d "${!v}" ] || fail "$v does not exist: ${!v}"
done

export HEXAGON_SDK_ROOT
# The toolchain directory carries a version in its name that moves with the
# SDK, so pin it only when there is a choice to make.
TOOLS_DIR="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools"
if [ -n "${HEXAGON_TOOLS_VER:-}" ]; then
  export DEFAULT_HEXAGON_TOOLS_ROOT="$TOOLS_DIR/$HEXAGON_TOOLS_VER"
else
  n_tools=$(find "$TOOLS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  [ "$n_tools" -eq 1 ] ||
    fail "found $n_tools toolchains under $TOOLS_DIR -- set HEXAGON_TOOLS_VER to the one to use"
  export DEFAULT_HEXAGON_TOOLS_ROOT="$(find "$TOOLS_DIR" -mindepth 1 -maxdepth 1 -type d)"
fi
[ -d "$DEFAULT_HEXAGON_TOOLS_ROOT" ] ||
  fail "Hexagon tools not at $DEFAULT_HEXAGON_TOOLS_ROOT"

if ! adb get-state >/dev/null 2>&1; then
  fail "no device visible to adb -- check 'adb devices'"
fi
DEVICE="$(adb get-serialno)"
echo "device: $DEVICE"

# --- 1. DSP skel -----------------------------------------------------------
log "1/4  Building the DSP skel (test/htp/build.sh)"
(
  cd "$REPO_ROOT/test/htp"
  HEXKL_ROOT="$HEXKL_ROOT" bash build.sh
)
SKEL="$REPO_ROOT/test/htp/build/libnntr_hvx_skel.so"
[ -f "$SKEL" ] || fail "skel did not build: $SKEL"

# --- 2. libnntrainer.so for arm64-v8a --------------------------------------
# unittest_hvx_mm_u8i4 does not itself call into nntrainer, but Android.mk's
# PREBUILT_SHARED_LIBRARY entries for it are parsed unconditionally, so the
# .so must exist on disk before ndk-build will process ANY target in this
# file. -Denable-htp is not required for this test (it never goes through
# nntrainer's HtpComputeOps seam), but matches what the CausalLM-side work
# will eventually need from the same builddir, so building it now saves a
# second full rebuild.
LIBNNTRAINER="$REPO_ROOT/builddir/jni/arm64-v8a/libnntrainer.so"
if [ -f "$LIBNNTRAINER" ]; then
  log "2/4  libnntrainer.so already built, skipping (rm -rf builddir to force)"
else
  log "2/4  Building libnntrainer.so for arm64-v8a (tools/package_android.sh)"
  # subprojects/iniparser (and occasionally others) can be an unfetched
  # wrap-git placeholder in a fresh worktree -- same class of gotcha as the
  # googletest one below, just tripped by the android build instead of the
  # host one.
  if [ ! -f "$REPO_ROOT/subprojects/iniparser/src/iniparser.c" ]; then
    echo "  subprojects/iniparser looks unfetched -- running git submodule update"
    git -C "$REPO_ROOT" submodule update --init subprojects/iniparser
  fi
  (
    cd "$REPO_ROOT"
    ANDROID_NDK="$ANDROID_NDK" PATH="$ANDROID_NDK:$PATH" \
      ./tools/package_android.sh --arm-arch=armv8.2-a -Dwerror=false
  )
fi
[ -f "$LIBNNTRAINER" ] || fail "libnntrainer.so did not build: $LIBNNTRAINER"

# --- 3. ARM gtest -----------------------------------------------------------
log "3/4  Building unittest_hvx_mm_u8i4 (ndk-build, arm64-v8a)"
GTEST_SUBMODULE="$REPO_ROOT/subprojects/googletest"
if [ ! -f "$GTEST_SUBMODULE/googletest/include/gtest/gtest.h" ]; then
  echo "  googletest submodule looks unfetched -- running git submodule update"
  git -C "$REPO_ROOT" submodule update --init subprojects/googletest
fi
# test/jni/Android.mk's googletest_main module expects ./googletest/{src,include}
# relative to test/jni/ -- there is no setup script that creates this, so do
# it here rather than committing a real copy into the tree.
if [ ! -e "$REPO_ROOT/test/jni/googletest" ]; then
  ln -sfn "$GTEST_SUBMODULE/googletest" "$REPO_ROOT/test/jni/googletest"
fi
(
  cd "$REPO_ROOT/test/jni"
  export ANDROID_NDK
  # Override, do not rely on the default: this shell's profile may already
  # export NNTRAINER_ROOT pointing at a different checkout (it does on at
  # least one dev machine this was verified on), which Android.mk's
  # ifndef-guarded default silently defers to.
  "$ANDROID_NDK/ndk-build" \
    NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
    APP_BUILD_SCRIPT=./Android.mk \
    NNTRAINER_ROOT="$REPO_ROOT" \
    HEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    unittest_hvx_mm_u8i4
)
TEST_BIN="$REPO_ROOT/test/jni/obj/local/arm64-v8a/unittest_hvx_mm_u8i4"
[ -f "$TEST_BIN" ] || fail "test binary did not build: $TEST_BIN"

# --- 4. push + run -----------------------------------------------------------
log "4/4  Pushing to $DEVICE:$DEVICE_TMP and running"
adb shell "mkdir -p $DEVICE_TMP"
adb push "$SKEL" "$DEVICE_TMP/" >/dev/null
adb push "$TEST_BIN" "$DEVICE_TMP/" >/dev/null
# c++_shared runtime the test binary links against (APP_STL in Application.mk).
# The prebuilt directory is named after the build host, so glob it rather
# than assuming linux-x86_64.
CXX_SHARED=$(find "$ANDROID_NDK/toolchains/llvm/prebuilt" \
  -path "*/aarch64-linux-android/libc++_shared.so" -print -quit 2>/dev/null || true)
if [ -n "$CXX_SHARED" ]; then
  adb push "$CXX_SHARED" "$DEVICE_TMP/" >/dev/null
else
  echo "  warning: libc++_shared.so not found under $ANDROID_NDK -- the test"
  echo "  will fail to load unless the device already has a copy"
fi

echo "  (unsigned-PD enable happens inside the test itself via remote_session_control)"
adb shell "cd $DEVICE_TMP && \
  chmod +x unittest_hvx_mm_u8i4 && \
  LD_LIBRARY_PATH=$DEVICE_TMP ADSP_LIBRARY_PATH=$DEVICE_TMP \
  ./unittest_hvx_mm_u8i4" 2>&1 | tee "$RUN_LOG"

echo
log "Summary"
grep -E "^\[  (PASSED|FAILED)|U8I[48]_FIELD" "$RUN_LOG" || true
echo
echo "Full log: $RUN_LOG"
echo "Gate to clear before building on top of this: all PASSED, and the"
echo "printed U8I4_FIELD path=layer_x4 field=speedup_vs_harness value=..."
echo "in the neighbourhood of 1.7-2. If it is not, stop and find out why."
