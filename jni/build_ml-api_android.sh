#! /bin/bash
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics
#
# @file build_ml-api_android.sh
# @date 18 August 2026
# @brief Build ml-api (nnstreamer native) for Android locally.
# @author Jaemin Shin <jaemin2.shin@ax.samsung.com>
#
# This mirrors nnstreamer/api .github/workflows/daily-build-android.yml
# (the job that publishes nnstreamer-lite-native.zip to the nnstreamer
# release S3 bucket). It is used as a fallback by prepare_ml-api.sh when the
# prebuilt artifact is unavailable on S3.
#
# It downloads the GStreamer Android prebuilts, checks out the nnstreamer
# source repositories, and runs the upstream build-nnstreamer-android.sh with
# the exact flags used by the CI 'android-build' composite action. The result
# is the '*-native-*.zip' produced under <api>/android_lib, whose absolute path
# is printed on the LAST line of stdout. All build noise goes to stderr so the
# caller can capture the path with $( ).
#
# usage: ./build_ml-api_android.sh <workdir>
#
# Overridable via environment:
#   ML_API_REF            git ref of nnstreamer/api           (default: pinned)
#   ML_API_BUILD_TYPE     all|lite|single|internal            (default: lite)
#   ML_API_TARGET_ABI     target ABI                          (default: arm64-v8a)
#   GST_ANDROID_VERSION   GStreamer android universal version (default: 1.24.13)

set -e

# --- configuration (matches the upstream CI at the pinned ref) --------------
# nnstreamer/api commit that the CI 'android-build' action was taken from.
API_REF=${ML_API_REF:-e7fedae8956bcd2b3c627ecaa01984707296bbdb}
GST_VER=${GST_ANDROID_VERSION:-1.24.13}
# 'lite' matches the nnstreamer-lite-native.zip that prepare_ml-api.sh consumes.
BUILD_TYPE=${ML_API_BUILD_TYPE:-lite}
TARGET_ABI=${ML_API_TARGET_ABI:-arm64-v8a}

WORKDIR=$1
[ -z "${WORKDIR}" ] && WORKDIR="$(pwd)/ml-api-build"
mkdir -p "${WORKDIR}"
WORKDIR="$(cd "${WORKDIR}" && pwd)"

# Because meson runs this via run_command (which captures all output), the only
# thing a failing CI job shows is "failed with status 1". Persist a full log to
# a fixed path so it can be dumped on failure, and mirror progress lines there.
# Only the final artifact path ever goes to stdout; everything else is stderr.
BUILD_LOG="${WORKDIR}/build_ml-api.log"
: > "${BUILD_LOG}"
log() { echo "[ml_api-build] $*" | tee -a "${BUILD_LOG}" >&2; }
log "workdir: ${WORKDIR}"
log "full build log: ${BUILD_LOG}"

# --- toolchain sanity -------------------------------------------------------
# The upstream build reads ANDROID_NDK_ROOT; accept the common aliases too.
if [ -z "${ANDROID_NDK_ROOT}" ]; then
  ANDROID_NDK_ROOT="${ANDROID_NDK:-${ANDROID_NDK_HOME}}"
fi
if [ -z "${ANDROID_NDK_ROOT}" ] || [ ! -d "${ANDROID_NDK_ROOT}" ]; then
  log "ERROR: Android NDK not found. Set ANDROID_NDK_ROOT (CI uses r25c)."
  exit 1
fi
export ANDROID_NDK_ROOT
export ANDROID_NDK="${ANDROID_NDK:-${ANDROID_NDK_ROOT}}"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT}}"
log "using ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT}"
case "${ANDROID_NDK_ROOT}" in
  *r25c*) : ;;
  *) log "WARNING: upstream CI builds with NDK r25c; a different NDK may fail." ;;
esac

# The nnstreamer gradle build requires JDK 17. GitHub-hosted runners preinstall
# several JDKs and expose them as JAVA_HOME_<ver>_<arch>; prefer the 17 one so
# the build does not depend on whatever `java` the job happens to default to.
if [ -z "${ML_API_JAVA_HOME}" ]; then
  for jh in "${JAVA_HOME_17_X64}" "${JAVA_HOME_17_ARM64}" "${JAVA_HOME_17_arm64}"; do
    [ -n "${jh}" ] && [ -x "${jh}/bin/java" ] && ML_API_JAVA_HOME="${jh}" && break
  done
fi
if [ -n "${ML_API_JAVA_HOME}" ] && [ -x "${ML_API_JAVA_HOME}/bin/java" ]; then
  export JAVA_HOME="${ML_API_JAVA_HOME}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
  log "using JAVA_HOME=${JAVA_HOME}"
fi
if command -v java >/dev/null 2>&1; then
  java_major=$(java -version 2>&1 | sed -n '1s/.*version "\([0-9]*\).*/\1/p')
  [ "${java_major}" != "17" ] && \
    log "WARNING: upstream CI builds with Java 17 (found Java ${java_major:-unknown}); set ML_API_JAVA_HOME to a JDK 17."
else
  log "WARNING: 'java' not found; the gradle build requires a JDK (CI uses 17)."
fi

for tool in git wget tar unzip zip; do
  command -v "${tool}" >/dev/null 2>&1 || { log "ERROR: '${tool}' is required."; exit 1; }
done

# --- 1) GStreamer Android prebuilts -----------------------------------------
export GSTREAMER_ROOT_ANDROID="${WORKDIR}/gst_root_android"
if [ ! -d "${GSTREAMER_ROOT_ANDROID}/arm64" ]; then
  log "downloading GStreamer android universal ${GST_VER}"
  mkdir -p "${GSTREAMER_ROOT_ANDROID}"
  gst_pkg="gstreamer-1.0-android-universal-${GST_VER}.tar.xz"
  gst_url="https://gstreamer.freedesktop.org/data/pkg/android/${GST_VER}/${gst_pkg}"
  if ! wget -q "${gst_url}" -O "${WORKDIR}/${gst_pkg}"; then
    log "ERROR: failed to download ${gst_url}"
    exit 1
  fi
  tar -xf "${WORKDIR}/${gst_pkg}" -C "${GSTREAMER_ROOT_ANDROID}"
  rm -f "${WORKDIR}/${gst_pkg}"
else
  log "GStreamer android prebuilts already present, skip download"
fi

# --- 2) source repositories -------------------------------------------------
# Clone (shallow) if missing; leave existing checkouts untouched so repeated
# meson reconfigures do not re-clone. Sub-repos track their default branch, as
# the upstream CI does; only nnstreamer/api is pinned to the CI ref.
_clone() {
  local repo=$1 dir=$2 ref=$3
  if [ ! -d "${WORKDIR}/${dir}/.git" ]; then
    log "cloning ${repo}"
    git clone --depth 1 "https://github.com/${repo}.git" "${WORKDIR}/${dir}"
  else
    log "${dir} already checked out, skip clone"
  fi
  if [ -n "${ref}" ]; then
    log "checking out ${repo}@${ref}"
    git -C "${WORKDIR}/${dir}" fetch --depth 1 origin "${ref}"
    git -C "${WORKDIR}/${dir}" checkout -q FETCH_HEAD
  fi
}

_clone nnstreamer/api                          api                          "${API_REF}"
_clone nnstreamer/nnstreamer                   nnstreamer                   ""
_clone nnstreamer/nnstreamer-edge              nnstreamer-edge              ""
_clone nnstreamer/nnstreamer-android-resource  nnstreamer-android-resource  ""
_clone nnstreamer/deviceMLOps.MLAgent          deviceMLOps.MLAgent          ""

# --- 3) environment expected by build-nnstreamer-android.sh -----------------
export NNSTREAMER_ROOT="${WORKDIR}/nnstreamer"
export NNSTREAMER_EDGE_ROOT="${WORKDIR}/nnstreamer-edge"
export NNSTREAMER_ANDROID_RESOURCE="${WORKDIR}/nnstreamer-android-resource"
export MLOPS_AGENT_ROOT="${WORKDIR}/deviceMLOps.MLAgent"
export ML_API_ROOT="${WORKDIR}/api"

# --- 4) build ---------------------------------------------------------------
# Flags copied verbatim from the CI android-build composite action (arm64-v8a).
log "starting nnstreamer android build (build_type=${BUILD_TYPE}, abi=${TARGET_ABI}); this may take a long time"
build_flags=( "--target_abi=${TARGET_ABI}" "--enable_nnfw=yes" "--build_type=${BUILD_TYPE}" )
if [ "${TARGET_ABI}" = "arm64-v8a" ]; then
  build_flags+=( "--enable_mqtt=yes" "--enable_ml_offloading=yes" \
                 "--enable_ml_service=yes" "--enable_llamacpp=yes" )
fi
# Capture the (very verbose) build output to the log file so it survives meson's
# output capture; keep stdout clean for the artifact path. On failure, echo the
# tail to stderr and propagate the non-zero status.
log "running: build-nnstreamer-android.sh ${build_flags[*]}"
build_status=0
bash "${ML_API_ROOT}/java/build-nnstreamer-android.sh" "${build_flags[@]}" \
  >> "${BUILD_LOG}" 2>&1 || build_status=$?
if [ "${build_status}" -ne 0 ]; then
  log "ERROR: nnstreamer android build failed (exit ${build_status}); tail of ${BUILD_LOG}:"
  tail -n 100 "${BUILD_LOG}" >&2
  exit "${build_status}"
fi

# --- 5) locate the native zip (main/jni/nnstreamer/{include,lib}) -----------
result_dir="${ML_API_ROOT}/android_lib"
native_zip=$(ls -t "${result_dir}"/*-native-*.zip 2>/dev/null | head -1)
if [ -z "${native_zip}" ] || [ ! -f "${native_zip}" ]; then
  log "ERROR: build finished but no *-native-*.zip found under ${result_dir}"
  exit 1
fi
log "built native artifact: ${native_zip}"

# Contract: last stdout line is the absolute artifact path.
echo "${native_zip}"
