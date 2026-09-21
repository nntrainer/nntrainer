#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Generates FastRPC stub/skel from nntr_htp.idl and cross-builds the DSP skel.
# Output: build_hexagon/generated/ (qaic output), build_hexagon/skel/libnntr_htp_skel.so
#
# Prerequisite: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
# Target arch defaults to v79 (the shipping skel since issue #35); the v75
# fallback, which runs unchanged on v79 silicon: HEX_ARCH=v75 ./tools/hexagon/build_skel.sh
# Extra compiler flags (appended, so they override): HEX_EXTRA_CFLAGS=-DFOO

set -eu

: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"

HEX_ARCH="${HEX_ARCH:-v79}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HTP_DIR="$REPO/nntrainer/tensor/hexagon/htp"
OUT="$REPO/build_hexagon"

mkdir -p "$OUT/generated" "$OUT/skel"

# HEXAGON.md §7 rule 4: the kernels need -mhvx-ieee-fp for the fp16 intrinsics
# (accepted by toolchain 8.8 and 19.0.04 on v75 and v79). Probe it instead of
# assuming it so an SDK that drops or implies the flag still builds; the
# outcome is printed so every build log records which way it went.
HEX_CLANG="$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang"
if "$HEX_CLANG" -m"$HEX_ARCH" -mhvx -mhvx-length=128B -mhvx-ieee-fp -c -x c /dev/null -o /dev/null 2>/dev/null; then
  HEX_IEEE_FLAG=-mhvx-ieee-fp
else
  HEX_IEEE_FLAG=
fi
echo "hexagon-clang $HEX_ARCH: -mhvx-ieee-fp probe -> ${HEX_IEEE_FLAG:-<not accepted, omitted>}"

"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" \
    -I "$HEXAGON_SDK_ROOT/incs" \
    -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -mdll -o "$OUT/generated" "$HTP_DIR/nntr_htp.idl"

SRCS=("$HTP_DIR/executor.c" "$OUT/generated/nntr_htp_skel.c")
# htp graph executor sources (same auto-include pattern as build_sim_test.sh)
for f in "$HTP_DIR"/worker_pool.c "$HTP_DIR"/htp_graph.c \
         "$HTP_DIR"/ops/*.c "$HTP_DIR"/hvx/*.c "$HTP_DIR"/dma/*.c; do
  [ -e "$f" ] && SRCS+=("$f")
done

# HexKL (issue #65 S0): when the addon mount carries libhexkl_micro.a for
# this toolchain/arch the skel is built with -DHTP_HMX=1, compiles hmx/*.c
# and links the library; the addon is never copied into the tree
# (tools/docker/run.sh mounts ~/Qualcomm/hexkl_addon at /opt/qcom/hexkl_addon).
# Without it, or with -DHTP_HMX=0 in HEX_EXTRA_CFLAGS, the build is the
# HVX-only tree (no hmx object, no HexKL symbol; HEXAGON.md section 5.3).
HEXKL_ADDON_ROOT="${HEXKL_ADDON_ROOT:-/opt/qcom/hexkl_addon}"
HEXKL_LIB="$HEXKL_ADDON_ROOT/lib/hexagon_${DEFAULT_TOOLS_VARIANT:-toolv19}_${HEX_ARCH}/libhexkl_micro.a"
HTP_HMX=0
case " ${HEX_EXTRA_CFLAGS:-} " in
  *" -DHTP_HMX=0 "*) ;;
  *) [ -f "$HEXKL_LIB" ] && [ -f "$HEXKL_ADDON_ROOT/include/hexkl_micro.h" ] && HTP_HMX=1 ;;
esac
HMX_CFLAGS=()
HMX_LIBS=()
if [ "$HTP_HMX" = 1 ]; then
  HMX_CFLAGS=(-DHTP_HMX=1 -I "$HTP_DIR/hmx" -I "$HEXKL_ADDON_ROOT/include")
  for f in "$HTP_DIR"/hmx/*.c; do
    [ -e "$f" ] && SRCS+=("$f")
  done
  HMX_LIBS=("$HEXKL_LIB")
  echo "HexKL: $HEXKL_LIB (HTP_HMX=1)"
else
  echo "HexKL: not linked (HTP_HMX=0; ${HEXKL_LIB} $([ -f "$HEXKL_LIB" ] && echo present || echo absent))"
fi

"$HEX_CLANG" \
    -m"$HEX_ARCH" -mhvx -mhvx-length=128B $HEX_IEEE_FLAG -G0 -O3 -fPIC -shared \
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
    "${SRCS[@]}" \
    ${HMX_LIBS[@]+"${HMX_LIBS[@]}"} \
    -o "$OUT/skel/libnntr_htp_skel.so"

echo "built: $OUT/skel/libnntr_htp_skel.so ($HEX_ARCH, HTP_HMX=$HTP_HMX)"
