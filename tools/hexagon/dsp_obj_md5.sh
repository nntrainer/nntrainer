#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# tools/hexagon/dsp_obj_md5.sh [tree-root] [out-dir]
# Compiles every DSP source of build_skel.sh with -c (same flags, same
# toolchain, HEX_ARCH / HEX_EXTRA_CFLAGS honoured) and prints one md5 per
# object, sorted by file name. Two trees whose lists agree produce the same
# skel code; the linked .so itself is not comparable because hexagon-link
# embeds its /tmp/<src>-xxxxxx.o command line (hexagon-gates, "no DSP bytes
# changed"). Default tree-root is this repo, default out-dir is
# build_hexagon/objproof/<basename of tree-root>.
# Prerequisite: source $HEXAGON_SDK_ROOT/setup_sdk_env.source

set -eu

: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"

HEX_ARCH="${HEX_ARCH:-v79}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TREE="$(cd "${1:-$HERE}" && pwd)"
OUT="${2:-$HERE/build_hexagon/objproof/$(basename "$TREE")}"
HTP_DIR="$TREE/nntrainer/tensor/hexagon/htp"

mkdir -p "$OUT/generated" "$OUT/obj"

HEX_CLANG="$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang"
if "$HEX_CLANG" -m"$HEX_ARCH" -mhvx -mhvx-length=128B -mhvx-ieee-fp -c -x c /dev/null -o /dev/null 2>/dev/null; then
  HEX_IEEE_FLAG=-mhvx-ieee-fp
else
  HEX_IEEE_FLAG=
fi

"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" \
    -I "$HEXAGON_SDK_ROOT/incs" \
    -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -mdll -o "$OUT/generated" "$HTP_DIR/nntr_htp.idl"

SRCS=("$HTP_DIR/executor.c" "$OUT/generated/nntr_htp_skel.c")
for f in "$HTP_DIR"/worker_pool.c "$HTP_DIR"/htp_graph.c \
         "$HTP_DIR"/ops/*.c "$HTP_DIR"/hvx/*.c "$HTP_DIR"/dma/*.c; do
  [ -e "$f" ] && SRCS+=("$f")
done

# Same HexKL rule as build_skel.sh: the hmx objects and -DHTP_HMX=1 exist
# only when the addon is mounted and HEX_EXTRA_CFLAGS does not carry
# -DHTP_HMX=0.
HEXKL_ADDON_ROOT="${HEXKL_ADDON_ROOT:-/opt/qcom/hexkl_addon}"
HEXKL_LIB="$HEXKL_ADDON_ROOT/lib/hexagon_${DEFAULT_TOOLS_VARIANT:-toolv19}_${HEX_ARCH}/libhexkl_micro.a"
HTP_HMX=0
case " ${HEX_EXTRA_CFLAGS:-} " in
  *" -DHTP_HMX=0 "*) ;;
  *) [ -f "$HEXKL_LIB" ] && [ -f "$HEXKL_ADDON_ROOT/include/hexkl_micro.h" ] && HTP_HMX=1 ;;
esac
HMX_CFLAGS=()
if [ "$HTP_HMX" = 1 ] && [ -d "$HTP_DIR/hmx" ]; then
  HMX_CFLAGS=(-DHTP_HMX=1 -I "$HTP_DIR/hmx" -I "$HEXKL_ADDON_ROOT/include")
  for f in "$HTP_DIR"/hmx/*.c; do
    [ -e "$f" ] && SRCS+=("$f")
  done
fi

for src in "${SRCS[@]}"; do
  obj="$OUT/obj/$(basename "$src" .c).o"
  "$HEX_CLANG" \
      -m"$HEX_ARCH" -mhvx -mhvx-length=128B $HEX_IEEE_FLAG -G0 -O3 -fPIC \
      -Wall -Werror -Wno-unused-function \
      -I "$OUT/generated" \
      -I "$HTP_DIR" -I "$HTP_DIR/ops" -I "$HTP_DIR/hvx" -I "$HTP_DIR/hex" \
      -I "$HTP_DIR/dma" \
      -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/qurt" \
      -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/posix" \
      -isystem "$HEXAGON_SDK_ROOT/incs" \
      -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
      -isystem "$HEXAGON_SDK_ROOT/ipc/fastrpc/incs" \
      ${HMX_CFLAGS[@]+"${HMX_CFLAGS[@]}"} \
      ${HEX_EXTRA_CFLAGS:-} \
      -c "$src" -o "$obj"
done

echo "dsp_obj_md5: tree=$TREE arch=$HEX_ARCH HTP_HMX=$HTP_HMX extra='${HEX_EXTRA_CFLAGS:-}'"
(cd "$OUT/obj" && md5sum -- *.o | sort -k2)
