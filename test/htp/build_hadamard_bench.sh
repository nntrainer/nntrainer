#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run build.sh first: client and skel must be generated from the same IDL.
set -eu
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"
: "${ANDROID_NDK:?set ANDROID_NDK}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$HERE/../../nntrainer/tensor/htp_backend"
CLANG="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang"
"$CLANG" -std=gnu11 -O2 -Wall -Wextra -Werror -DNNTR_HADAMARD_DEVICE \
  -I "$HERE/generated" -I "$BACKEND/hvx" \
  -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "$HERE/host/hadamard_check.c" "$BACKEND/hvx/hvx_hadamard_f32.c" \
  "$HERE/generated/nntr_hvx_stub.c" \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" \
  -lcdsprpc -ldl -llog -lm -Wl,-z,max-page-size=16384 \
  -o "$HERE/build/hadamard_check_device"
"$CLANG" -O2 -c "$HERE/generated/nntr_hvx_stub.c" \
  -I "$HERE/generated" -I "$HEXAGON_SDK_ROOT/incs" \
  -I "$HEXAGON_SDK_ROOT/incs/stddef" -o "$HERE/build/hadamard_stub.o"
"${CLANG}++" -std=c++17 -O2 -Wall -Wextra -Werror -static-libstdc++ \
  -I "$HERE/generated" -I "$HEXAGON_SDK_ROOT/incs" \
  -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "$HERE/hadamard_moe_bench.cpp" "$HERE/build/hadamard_stub.o" \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" \
  -lcdsprpc -ldl -llog -lm -Wl,-z,max-page-size=16384 \
  -o "$HERE/build/hadamard_moe_bench"

"$CLANG" -std=gnu11 -O2 -Wall -Wextra -Werror \
  -I "$HERE/generated" -I "$HEXAGON_SDK_ROOT/incs" \
  -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "$HERE/hadamard_hmx_check.c" "$HERE/build/hadamard_stub.o" \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" \
  -lcdsprpc -ldl -llog -lm -Wl,-z,max-page-size=16384 \
  -o "$HERE/build/hadamard_hmx_check"
