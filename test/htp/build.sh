#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Generates the FastRPC stub/skel from nntr_hvx.idl and builds the DSP skel.
#
# Prerequisite: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
#   The SDK must be 6.1.1.0 or newer: HexKL ships libhexkl_micro.a for v79
#   only from lib/6.1.1.0 up. 6.4.0.1 is the verified combination.
# Override the target with: HEX_ARCH=v75 ./build.sh
# Set HexKL path:           HEXKL_ROOT=/path/to/hexkl_addon ./build.sh

set -eu

: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"

HEX_ARCH="${HEX_ARCH:-v79}"
HEXKL_ROOT="${HEXKL_ROOT:?set HEXKL_ROOT to your hexkl_addon path}"
HEXAGON_SDK_VER="${HEXAGON_SDK_VER:-$(basename "$HEXAGON_SDK_ROOT")}"
HEXKL_TOOLS_VARIANT="${HEXKL_TOOLS_VARIANT:-toolv19}"

# HexKL addon layout (newer versions):
#   lib/<SDK_VERSION>/hexagon_toolv<NN>_<arch>/libhexkl_micro.a
# Try to find libhexkl_micro.a:
#   1. First try: lib/$HEXAGON_SDK_VER/hexagon_${HEXKL_TOOLS_VARIANT}_${HEX_ARCH}/
#   2. If not found: try lib/hexagon_${HEXKL_TOOLS_VARIANT}_${HEX_ARCH}/ (fallback for older layout)
HEXKL_LIB_VERSIONED="$HEXKL_ROOT/lib/$HEXAGON_SDK_VER/hexagon_${HEXKL_TOOLS_VARIANT}_${HEX_ARCH}/libhexkl_micro.a"
HEXKL_LIB_FLAT="$HEXKL_ROOT/lib/hexagon_${HEXKL_TOOLS_VARIANT}_${HEX_ARCH}/libhexkl_micro.a"

if [ -f "$HEXKL_LIB_VERSIONED" ]; then
    HEXKL_LIB="$HEXKL_LIB_VERSIONED"
elif [ -f "$HEXKL_LIB_FLAT" ]; then
    HEXKL_LIB="$HEXKL_LIB_FLAT"
else
    HEXKL_LIB="$HEXKL_LIB_VERSIONED"  # default for error message
fi

if [ ! -f "$HEXKL_LIB" ]; then
    echo "Error: HexKL static library not found:" >&2
    echo "  Tried: $HEXKL_LIB_VERSIONED" >&2
    [ -z "$HEXKL_LIB_FLAT" ] || echo "  Tried: $HEXKL_LIB_FLAT" >&2
    echo "" >&2
    echo "Diagnosis:" >&2
    echo "  HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT (SDK version: $HEXAGON_SDK_VER)" >&2
    echo "  HEXKL_ROOT=$HEXKL_ROOT" >&2
    echo "  HEX_ARCH=$HEX_ARCH" >&2
    echo "  HEXKL_TOOLS_VARIANT=$HEXKL_TOOLS_VARIANT" >&2
    echo "" >&2
    echo "Available SDK versions in HexKL:" >&2
    ls -d "$HEXKL_ROOT/lib"/* 2>/dev/null | xargs -I {} basename {} | sort -rV | sed 's/^/  /' >&2 || echo "  (none)" >&2
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

SRCS="hvx_add_f32.c nntr_hvx_mm_u8i4.c nntr_hvx_mm_u8i8.c nntr_hvx_attn_f16.c"
SRCS="$SRCS generated/nntr_hvx_skel.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_mm_u8i4.c $BACKEND/hmx/hexkl_mm_u8i4_dma.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_mm_u8i8_dma.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_dma_ring.c $BACKEND/hmx/hexkl_attn_f16.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_kv_tiles_f16.c"
SRCS="$SRCS $BACKEND/hvx/hvx_quant_u8.c $BACKEND/hvx/hvx_dequant_i32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_worker_pool.c $BACKEND/hvx/hvx_attn_decode_f16.c"

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

echo "built: $SCRIPT_DIR/build/libnntr_hvx_skel.so ($HEX_ARCH, SDK=$HEXAGON_SDK_VER)"
echo "  HexKL library: $HEXKL_LIB"
