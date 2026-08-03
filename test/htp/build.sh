#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Generates the FastRPC stub/skel from nntr_hvx.idl and builds the DSP skel.
#
# Prerequisite: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
# Override the target with: HEX_ARCH=v75 ./build.sh

set -eu

: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"

HEX_ARCH="${HEX_ARCH:-v79}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p generated build

"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" \
    -I "$HEXAGON_SDK_ROOT/incs" \
    -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -mdll -o generated nntr_hvx.idl

"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" \
    -m"$HEX_ARCH" -mhvx -mhvx-length=128B -G0 -O3 -fPIC -shared \
    -Wall -Werror \
    -I generated \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/posix" \
    -isystem "$HEXAGON_SDK_ROOT/incs" \
    -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
    -isystem "$HEXAGON_SDK_ROOT/ipc/fastrpc/incs" \
    hvx_add_f32.c generated/nntr_hvx_skel.c \
    -o build/libnntr_hvx_skel.so

echo "built: $SCRIPT_DIR/build/libnntr_hvx_skel.so ($HEX_ARCH)"
