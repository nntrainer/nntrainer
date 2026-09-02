#! /bin/bash
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
#
# @file prepare_ml-api-common.sh
# @date 10 June 2021
# @brief This file is a helper tool to ml-api-common dependency for android build
# @author Parichay Kapoor <pk.kapoor@samsung.com>
#
# usage: ./prepare_ml-api-common.sh target

set -e
TARGET=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Note: zip name can be nnstreamer-native-*.zip but this file is heavier to download
FILE_PREFIX=nnstreamer-lite-native
ZIP_NAME=${FILE_PREFIX}.zip
URL="https://nnstreamer-release.s3-ap-northeast-2.amazonaws.com/nnstreamer/latest/android/"

echo "PREPARING ml_api at ${TARGET}"

[ ! -d ${TARGET} ] && mkdir -p ${TARGET}

pushd ${TARGET}

function _download_ml_api {
  [ -f $ZIP_NAME ] && echo "${ZIP_NAME} exists, skip downloading" && return 0
  echo "[ml_api] downloading ${ZIP_NAME}\n"
  if ! wget -r -l1 -nH --cut-dirs=3 ${URL}${ZIP_NAME} -O ${ZIP_NAME} --no-check-certificate ; then
    echo "[ml_api] Download failed, please check url\n"
    # wget leaves an empty file behind on failure; drop it so a retry/fallback
    # is not fooled into thinking the archive already exists.
    rm -f ${ZIP_NAME}
    return 1
  fi
  echo "[ml_api] Finish downloading ml_api\n"
  return 0
}

function _build_ml_api {
  # Fallback for when the prebuilt archive is not published on S3: build it
  # locally by reproducing nnstreamer/api's daily-build-android workflow.
  echo "[ml_api] prebuilt ${ZIP_NAME} unavailable; building ml-api locally"
  local built_zip
  if ! built_zip=$("${SCRIPT_DIR}/build_ml-api_android.sh" "${TARGET}/local-build"); then
    echo "[ml_api] local ml-api build failed; see ${TARGET}/local-build/build_ml-api.log"
    return 1
  fi
  echo "[ml_api] using locally built archive: ${built_zip}"
  cp -f "${built_zip}" "${ZIP_NAME}"
  return 0
}

function _obtain_ml_api {
  # Prefer the prebuilt archive from S3; fall back to a local build.
  _download_ml_api && return 0
  _build_ml_api
}

function _extract_ml_api {
  echo "[ml_api] unzip ml_api\n"
  unzip -q ${ZIP_NAME} -d ${FILE_PREFIX}
  rm -f ${ZIP_NAME}
}

function _cleanup_ml_api {
  echo "[ml_api] cleanup ml_api \n"
  # move include to the target location
  mv -f ${FILE_PREFIX}/main/jni/nnstreamer/include .
  mv -f ${FILE_PREFIX}/main/jni/nnstreamer/lib .
  # remove all untarred directories/files
  rm -rf ${FILE_PREFIX}
  # cleanup all files other than ml_api and tizen_error
  find include ! \( -name '*.h' \) -type f -exec rm -f {} +
  find lib ! \( -name 'libnnstreamer-native.so' -or -name 'libgstreamer_android.so' \) -type f -exec rm -f {} +
}

# Under `set -e`, keep these as body statements of an `if` (not an `&&` chain)
# so a failure in _obtain_ml_api aborts the script and meson's `check: true`
# reports it, instead of being silently swallowed by the && short-circuit.
if [ ! -d "${TARGET}/include" ]; then
  _obtain_ml_api
  _extract_ml_api
  _cleanup_ml_api
fi

popd
