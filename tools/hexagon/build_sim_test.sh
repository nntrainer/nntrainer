#!/bin/bash
# tools/hexagon/build_sim_test.sh
# Cross-builds the hexagon-sim test lib for HEX_ARCH (default v79, the
# shipping arch since issue #35; v75 is the fallback, built and simulated
# only when a change touches an __HVX_ARCH__ branch or on request, not per
# PR). Needs $HEXAGON_SDK_ROOT/rtos/qurt/compute<arch>/sdksim_bin/runelf.pbn;
# SDK 6.4 ships both. Prereq: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
# Extra compiler flags (appended, so they override): HEX_EXTRA_CFLAGS=-DFOO
set -eu
: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"
HEX_ARCH="${HEX_ARCH:-v79}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HTP_DIR="$REPO/nntrainer/tensor/hexagon/htp"
SIM_DIR="$REPO/test/hexagon/sim"
OUT="$REPO/build_hexagon/sim"
mkdir -p "$OUT"

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

SRCS=("$SIM_DIR"/*.c)
# htp sources are picked up as they land (ops/, hvx/, worker_pool, htp_graph)
for f in "$HTP_DIR"/worker_pool.c "$HTP_DIR"/htp_graph.c \
         "$HTP_DIR"/ops/*.c "$HTP_DIR"/hvx/*.c "$HTP_DIR"/dma/*.c; do
  [ -e "$f" ] && SRCS+=("$f")
done

# HexKL (issue #65 S0): same rule as build_skel.sh. With libhexkl_micro.a on
# the addon mount the library is built with -DHTP_HMX=1, hmx/*.c and the
# HexKL link, and the 14th test (hmx) runs; without it, or with -DHTP_HMX=0
# in HEX_EXTRA_CFLAGS, the 13 HVX tests build as before and `hmx` reports
# itself skipped. The plain -mv79 simulator core (v79na_1) executes HMX.
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
    -m"$HEX_ARCH" -mhvx -mhvx-length=128B $HEX_IEEE_FLAG -G0 -O2 -g -fPIC -shared \
    -Wall -Werror -Wno-unused-function \
    -I "$HTP_DIR" -I "$HTP_DIR/ops" -I "$HTP_DIR/hvx" -I "$HTP_DIR/hex" -I "$HTP_DIR/dma" -I "$SIM_DIR" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/posix" \
    -isystem "$HEXAGON_SDK_ROOT/incs" \
    -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
    ${HMX_CFLAGS[@]+"${HMX_CFLAGS[@]}"} \
    ${HEX_EXTRA_CFLAGS:-} \
    "${SRCS[@]}" \
    ${HMX_LIBS[@]+"${HMX_LIBS[@]}"} \
    -o "$OUT/libnntr_sim_test.so"
# Stamp the arch and the extra flags next to the library: run_sim_test.sh
# refuses to boot a different simulator against it, or to run a variant build
# (HEX_EXTRA_CFLAGS) as if it were the plain one (both arches and every
# variant share this output directory).
echo "$HEX_ARCH ${HEX_EXTRA_CFLAGS:-}" > "$OUT/libnntr_sim_test.arch"
echo "built: $OUT/libnntr_sim_test.so ($HEX_ARCH${HEX_EXTRA_CFLAGS:+ $HEX_EXTRA_CFLAGS}, HTP_HMX=$HTP_HMX)"
