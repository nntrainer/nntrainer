#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Generates the FastRPC stub/skel from nntr_hvx.idl and builds the DSP skel.
#
# Prerequisite: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
#   The SDK must be 6.1.1.0 or newer: HexKL ships libhexkl_micro.a for v79
#   only from lib/6.1.1.0 up. 6.4.0.1 is the verified combination.
# Override the target with: HEX_ARCH=v75 ./build.sh
# Override HexKL with:      HEXKL_ROOT=/path/to/hexkl_addon ./build.sh

set -eu

: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"

HEX_ARCH="${HEX_ARCH:-v79}"
HEXKL_ROOT="${HEXKL_ROOT:-$HOME/Downloads/hexkl_addon}"
HEXKL_SDK_VER="${HEXKL_SDK_VER:-6.4.0.1}"
HEXKL_TOOLS_VARIANT="${HEXKL_TOOLS_VARIANT:-toolv19}"
HEXKL_LIB="$HEXKL_ROOT/lib/$HEXKL_SDK_VER/hexagon_${HEXKL_TOOLS_VARIANT}_${HEX_ARCH}/libhexkl_micro.a"

if [ ! -f "$HEXKL_LIB" ]; then
    echo "Error: HexKL static library not found:" >&2
    echo "  $HEXKL_LIB" >&2
    echo "HexKL provides v79 only for lib/6.1.1.0 and newer." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND="$REPO_ROOT/nntrainer/tensor/htp_backend"
cd "$SCRIPT_DIR"

mkdir -p generated build

"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" \
    -I "$HEXAGON_SDK_ROOT/incs" \
    -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -mdll -o generated nntr_hvx.idl

SRCS="hvx_add_f32.c nntr_hvx_mm_u8i4.c generated/nntr_hvx_skel.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_mm_u8i4.c"
SRCS="$SRCS $BACKEND/hvx/hvx_quant_u8.c $BACKEND/hvx/hvx_dequant_i32.c"

"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" \
    -m"$HEX_ARCH" -mhvx -mhvx-length=128B -G0 -O3 -fPIC -shared \
    -Wall -Werror \
    -I generated \
    -I "$HEXKL_ROOT/include" \
    -I "$BACKEND/hvx" \
    -I "$BACKEND/hmx" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/posix" \
    -isystem "$HEXAGON_SDK_ROOT/incs" \
    -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
    -isystem "$HEXAGON_SDK_ROOT/ipc/fastrpc/incs" \
    $SRCS \
    "$HEXKL_LIB" \
    -o build/libnntr_hvx_skel.so

echo "built: $SCRIPT_DIR/build/libnntr_hvx_skel.so ($HEX_ARCH, hexkl $HEXKL_SDK_VER)"
