#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Cross-builds the arm64 Android host-side binaries: the RPC round-trip test
# (hexagon_rpc_test) and the qwen3 e2e harness (hexagon_e2e_test).
# Requires build_skel.sh to have run first (generates the stub).
#
# Prerequisites: ANDROID_NDK set; HEXAGON_SDK_ROOT set.

set -eu

: "${ANDROID_NDK:?set ANDROID_NDK to your NDK root}"
: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"

API="${ANDROID_API:-31}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/build_hexagon"
GEN="$OUT/generated"
HOST_DIR="$REPO/nntrainer/tensor/hexagon/host"
HTP_DIR="$REPO/nntrainer/tensor/hexagon/htp"
APP_DIR="$REPO/Applications/CausalLM/hexagon"
TC="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
CDSPRPC_DIR="$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64"

[ -f "$GEN/nntr_htp_stub.c" ] || { echo "run build_skel.sh first"; exit 1; }

mkdir -p "$OUT/host"

INCS=(-I "$GEN" -I "$HOST_DIR" -I "$HTP_DIR" -I "$APP_DIR"
      -isystem "$HEXAGON_SDK_ROOT/incs"
      -isystem "$HEXAGON_SDK_ROOT/incs/stddef"
      -isystem "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc")

"$TC/aarch64-linux-android${API}-clang" -c -O2 -fPIC -Wall \
    "${INCS[@]}" "$GEN/nntr_htp_stub.c" -o "$OUT/host/nntr_htp_stub.o"

# -static-libstdc++: the binaries run from /data/local/tmp where the NDK's
# libc++_shared.so is not available.
build() { # <output name> <sources...>
  local out="$1"; shift
  "$TC/aarch64-linux-android${API}-clang++" -std=c++17 -O2 -Wall -Werror \
      -static-libstdc++ "${INCS[@]}" \
      "$HOST_DIR/rpcmem_allocator.cpp" "$HOST_DIR/hexagon_runner.cpp" "$@" \
      "$OUT/host/nntr_htp_stub.o" -L "$CDSPRPC_DIR" -lcdsprpc \
      -o "$OUT/host/$out"
  echo "built: $OUT/host/$out"
}

build hexagon_rpc_test "$REPO/test/hexagon/hexagon_rpc_test.cpp"
build hexagon_e2e_test "$REPO/test/hexagon/hexagon_e2e_test.cpp" \
    "$HOST_DIR/graph_lowering.cpp" "$APP_DIR/qwen3_lowering.cpp" \
    "$APP_DIR/hex_image.cpp"
