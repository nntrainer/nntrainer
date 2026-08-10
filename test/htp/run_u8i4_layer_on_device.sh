#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Builds the DSP skel + the ARM gtest for the u8i4 AND u8i8 layer endpoints
# (HTP_PR_PLAN.md PR①/PR② / doc15 §8 item 3) and runs it on a connected
# device. Both live in one gtest binary (unittest_hvx_mm_u8i4 despite the
# name -- it predates the u8i8 fixture), so one run covers both.
#
# What this checks, in order:
#   1. DSP skel compiles clean against HexKL beta2 (-Wall -Werror)
#   2. libnntrainer.so + the ARM gtest binary build for arm64-v8a
#   3. unittest_hvx_mm_u8i4 passes on-device -- the original u8i4 accuracy
#      harness (Shape1-4), the u8i4 layer-endpoint tests, and the u8i8
#      layer-endpoint tests
#   4. the reported layer_x4/harness speedup (u8i4) lands near doc13 §3a's
#      1.7-2x, and the u8i8-vs-u8i4 ratio is sane (both printed, not
#      asserted -- see the ReportPerCallCost* tests' comments)
#
# Override any of these if your paths differ:
: "${HEXAGON_SDK_ROOT:=$HOME/workspace/Hexagon_SDK/6.4.0.2}"
: "${HEXKL_ROOT:=$HOME/workspace/hxkl-beta2/hexkl_addon}"   # beta2, NOT the
                                                            # one under
                                                            # Hexagon_SDK/*/addons/
: "${HEXKL_SDK_VER:=6.4.0.2}"
: "${ANDROID_NDK:=$HOME/workspace/android-ndk-r26d}"
: "${DEVICE_TMP:=/data/local/tmp/htp_u8i4_layer_test}"

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

log()  { echo -e "\n\033[1;36m==> $*\033[0m"; }
fail() { echo -e "\033[1;31mFAILED: $*\033[0m" >&2; exit 1; }

for v in HEXAGON_SDK_ROOT HEXKL_ROOT ANDROID_NDK; do
  [ -d "${!v}" ] || fail "$v does not exist: ${!v}"
done

export HEXAGON_SDK_ROOT
export DEFAULT_HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04"
[ -d "$DEFAULT_HEXAGON_TOOLS_ROOT" ] ||
  fail "Hexagon tools not at $DEFAULT_HEXAGON_TOOLS_ROOT -- check the toolchain version dir name"

if ! adb get-state >/dev/null 2>&1; then
  fail "no device visible to adb -- check 'adb devices'"
fi
DEVICE="$(adb get-serialno)"
echo "device: $DEVICE"

# --- 1. DSP skel -----------------------------------------------------------
log "1/4  Building the DSP skel (test/htp/build.sh, HexKL $HEXKL_SDK_VER)"
(
  cd "$REPO_ROOT/test/htp"
  HEXKL_ROOT="$HEXKL_ROOT" HEXKL_SDK_VER="$HEXKL_SDK_VER" bash build.sh
)
SKEL="$REPO_ROOT/test/htp/build/libnntr_hvx_skel.so"
[ -f "$SKEL" ] || fail "skel did not build: $SKEL"

# --- 2. libnntrainer.so for arm64-v8a --------------------------------------
# unittest_hvx_mm_u8i4 does not itself call into nntrainer, but Android.mk's
# PREBUILT_SHARED_LIBRARY entries for it are parsed unconditionally, so the
# .so must exist on disk before ndk-build will process ANY target in this
# file. -Denable-htp is not required for this test (it never goes through
# nntrainer's HtpComputeOps seam), but matches what PR③ will eventually need
# from the same builddir, so building it now saves a second full rebuild.
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
log "3/4  Building unittest_hvx_mm_u8i4 + _softmax + _attn (ndk-build, arm64-v8a)"
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
    unittest_hvx_mm_u8i4 unittest_hvx_softmax unittest_hvx_attn \
    unittest_hvx_fc
)
TEST_BIN="$REPO_ROOT/test/jni/obj/local/arm64-v8a/unittest_hvx_mm_u8i4"
[ -f "$TEST_BIN" ] || fail "test binary did not build: $TEST_BIN"
# The softmax fixture carries the blocked-softmax gate: PHASE B of the fused
# attention loop against mha_softmax_blocked_ref, the same host model the whole
# flash loop is specified against.
SOFTMAX_BIN="$REPO_ROOT/test/jni/obj/local/arm64-v8a/unittest_hvx_softmax"
[ -f "$SOFTMAX_BIN" ] || fail "test binary did not build: $SOFTMAX_BIN"
# PHASE A (S = Q.Kt) against mha_htp_host_scores, the same stage the fused
# forward will run.
ATTN_BIN="$REPO_ROOT/test/jni/obj/local/arm64-v8a/unittest_hvx_attn"
[ -f "$ATTN_BIN" ] || fail "test binary did not build: $ATTN_BIN"
# The other half of the QNN comparison table: one fully connected layer at
# Qwen3-0.6B's q_proj shape, u8 x i8, reported the way QNN net-run reports.
FC_BIN="$REPO_ROOT/test/jni/obj/local/arm64-v8a/unittest_hvx_fc"
[ -f "$FC_BIN" ] || fail "test binary did not build: $FC_BIN"

# --- 4. push + run -----------------------------------------------------------
log "4/4  Pushing to $DEVICE:$DEVICE_TMP and running"
adb shell "mkdir -p $DEVICE_TMP"
adb push "$SKEL" "$DEVICE_TMP/" >/dev/null
adb push "$TEST_BIN" "$DEVICE_TMP/" >/dev/null
adb push "$SOFTMAX_BIN" "$DEVICE_TMP/" >/dev/null
adb push "$ATTN_BIN" "$DEVICE_TMP/" >/dev/null
adb push "$FC_BIN" "$DEVICE_TMP/" >/dev/null
# c++_shared runtime the test binary links against (APP_STL in Application.mk)
CXX_SHARED="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
[ -f "$CXX_SHARED" ] && adb push "$CXX_SHARED" "$DEVICE_TMP/" >/dev/null

echo "  (unsigned-PD enable happens inside the test itself via remote_session_control)"
adb shell "cd $DEVICE_TMP && \
  chmod +x unittest_hvx_mm_u8i4 && \
  LD_LIBRARY_PATH=$DEVICE_TMP ADSP_LIBRARY_PATH=$DEVICE_TMP \
  ./unittest_hvx_mm_u8i4" 2>&1 | tee /tmp/hvx_mm_u8i4_device_run.log

echo
log "4b/4  HVX softmax, including the blocked (flash attention PHASE B) gate"
adb shell "cd $DEVICE_TMP && \
  chmod +x unittest_hvx_softmax && \
  LD_LIBRARY_PATH=$DEVICE_TMP ADSP_LIBRARY_PATH=$DEVICE_TMP \
  ./unittest_hvx_softmax" 2>&1 | tee /tmp/hvx_softmax_device_run.log

echo
log "4c/4  Attention PHASE A (S = Q.Kt) against the host model"
adb shell "cd $DEVICE_TMP && \
  chmod +x unittest_hvx_attn && \
  LD_LIBRARY_PATH=$DEVICE_TMP ADSP_LIBRARY_PATH=$DEVICE_TMP \
  ./unittest_hvx_attn" 2>&1 | tee /tmp/hvx_attn_device_run.log

echo
log "4d/4  Fully connected layer cost (Qwen3-0.6B q_proj, u8 x i8)"
adb shell "cd $DEVICE_TMP && \
  chmod +x unittest_hvx_fc && \
  LD_LIBRARY_PATH=$DEVICE_TMP ADSP_LIBRARY_PATH=$DEVICE_TMP \
  ./unittest_hvx_fc" 2>&1 | tee /tmp/hvx_fc_device_run.log

echo
log "Summary"
grep -E "^\[  (PASSED|FAILED)|U8I[48]_FIELD" /tmp/hvx_mm_u8i4_device_run.log || true
grep -E "^\[  (PASSED|FAILED)|BLOCKED_FIELD" /tmp/hvx_softmax_device_run.log || true
grep -E "^\[  (PASSED|FAILED)|ATTN_FIELD" /tmp/hvx_attn_device_run.log || true
grep -E "^\[  (PASSED|FAILED)|FC_FIELD" /tmp/hvx_fc_device_run.log || true
echo
echo "Full logs: /tmp/hvx_mm_u8i4_device_run.log"
echo "           /tmp/hvx_softmax_device_run.log"
echo "           /tmp/hvx_attn_device_run.log"
echo "           /tmp/hvx_fc_device_run.log"
echo
echo "Tables and a HTML report from those logs:"
echo "  tools/htp_attn_report.py"
echo "  tools/htp_attn_report.py --html /tmp/htp_attn_report.html"
echo "  tools/htp_fc_report.py            # the fully connected layer"
echo "  tools/htp_fc_report.py --qnn 512=<ms>,1024=<ms>   # overlay QNN, per M"
echo "Gate to clear before starting PR③: all PASSED, and the printed"
echo "U8I4_FIELD path=layer_x4 field=speedup_vs_harness value=... should be"
echo "in the neighbourhood of 1.7-2 (doc13 §3a). If it is not, stop and find"
echo "out why before building anything on top of this."
echo
echo "Gate for the attention work: HvxSoftmaxBlocked all PASSED, and both"
echo "BLOCKED_FIELD max_abs_err_p / max_abs_err_l <= 1e-6 against the host"
echo "model. That is what makes the later bit-exactness gates meaningful."
