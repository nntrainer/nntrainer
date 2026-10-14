#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Generates the FastRPC stub/skel from nntr_hvx.idl and builds the DSP skel.
#
# Prerequisite: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
#   The SDK must be 6.1.1.0 or newer: HexKL ships libhexkl_micro.a for v79
#   only from lib/6.1.1.0 up. 6.4.0.1 is the verified combination.
# Override the target with: HEX_ARCH=v81 ./build.sh (the S26 Ultra; v79 = S25)
# Override HexKL with:      HEXKL_ROOT=/path/to/hexkl_addon ./build.sh

set -eu

: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"

HEX_ARCH="${HEX_ARCH:-v79}"
# Unset HEXKL_ROOT picks the first install that is actually there. The two
# candidates are the beta2 drop the device script (run_u8i4_layer_on_device.sh)
# defaults to and the older unpack-in-Downloads location; a stale default cost a
# bisect step, because this script failing is easy to miss when the caller does
# not stop on it.
if [ -z "${HEXKL_ROOT:-}" ]; then
    HEXKL_ROOT="$HOME/workspace/hxkl-beta2/hexkl_addon"
    if [ ! -d "$HEXKL_ROOT" ] && [ -d "$HOME/Downloads/hexkl_addon" ]; then
        HEXKL_ROOT="$HOME/Downloads/hexkl_addon"
    fi
fi
# Likewise the SDK version: fall back to the newest one this install ships.
if [ -z "${HEXKL_SDK_VER:-}" ]; then
    HEXKL_SDK_VER="$(ls -1 "$HEXKL_ROOT/lib" 2>/dev/null | sort -V | tail -1 || true)"
    HEXKL_SDK_VER="${HEXKL_SDK_VER:-6.4.0.2}"
fi
HEXKL_TOOLS_VARIANT="${HEXKL_TOOLS_VARIANT:-toolv19}"
HEXKL_LIB="$HEXKL_ROOT/lib/$HEXKL_SDK_VER/hexagon_${HEXKL_TOOLS_VARIANT}_${HEX_ARCH}/libhexkl_micro.a"

if [ ! -f "$HEXKL_LIB" ]; then
    echo "Error: HexKL static library not found:" >&2
    echo "  $HEXKL_LIB" >&2
    # Say which of the three parts of that path is actually wrong. The old
    # message here named the version policy no matter what was missing,
    # which points at the wrong thing when HEXKL_ROOT is simply not where
    # this script guesses -- the common case, since the default below is one
    # developer's download directory and run_u8i4_layer_on_device.sh
    # overrides it while a direct ./build.sh does not.
    if [ ! -d "$HEXKL_ROOT" ]; then
        echo "HEXKL_ROOT does not exist: $HEXKL_ROOT" >&2
        echo "Pass the real one, e.g." >&2
        echo "  HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon HEXKL_SDK_VER=6.4.0.2 ./build.sh" >&2
    elif [ ! -d "$HEXKL_ROOT/lib/$HEXKL_SDK_VER" ]; then
        echo "HEXKL_SDK_VER=$HEXKL_SDK_VER is not under $HEXKL_ROOT/lib. Available:" >&2
        ls -1 "$HEXKL_ROOT/lib" 2>/dev/null | sed 's/^/  /' >&2
    else
        echo "That version exists but has no ${HEXKL_TOOLS_VARIANT}_${HEX_ARCH} build. Available:" >&2
        ls -1 "$HEXKL_ROOT/lib/$HEXKL_SDK_VER" 2>/dev/null | sed 's/^/  /' >&2
        echo "Pick an HEX_ARCH (or HEXKL_SDK_VER) from the list above." >&2
    fi
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND="$REPO_ROOT/nntrainer/tensor/htp_backend"
# $BACKEND/.. is nntrainer/tensor, where the headers shared with the host live
# -- swiglu_det.h is the DSP's SwiGLU specification AND what the ARM side and
# the offline quantizer compile against, so it cannot sit behind the HTP-only
# include path.
cd "$SCRIPT_DIR"

mkdir -p generated build

"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" \
    -I "$HEXAGON_SDK_ROOT/incs" \
    -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -mdll -o generated nntr_hvx.idl

# [#132 PR 2] hvx_q4_gemv_f32.c, the sf_probe entry and online Hadamard
# are compiled with -mhvx-ieee-fp, the flag the first sitting's
# silicon-checked kernel was built with; the rest of the skel is built
# without it (its Q6_Vsf_* intrinsics keep their qf32 lowering, A's bits),
# so these files are compiled on their own and linked in as objects.
for f in "$BACKEND/hvx/hvx_q4_gemv_f32.c" "$BACKEND/hvx/hvx_hadamard_f32.c" nntr_hvx_sf_probe.c; do
"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" \
    -m"$HEX_ARCH" -mhvx -mhvx-length=128B -mhvx-ieee-fp -G0 -O3 -fPIC \
    -Wall -Werror ${HEX_EXTRA_CFLAGS:-} \
    -I generated -I "$BACKEND/.." -I "$BACKEND" -I "$BACKEND/hvx" -I "$BACKEND/hmx" \
    -I "$HEXKL_ROOT/include" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/posix" \
    -isystem "$HEXAGON_SDK_ROOT/incs" \
    -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
    -isystem "$HEXAGON_SDK_ROOT/ipc/fastrpc/incs" \
    -c "$f" -o "build/$(basename "${f%.c}").o"
done

SRCS="hvx_add_f32.c nntr_hvx_mm_u8i4.c nntr_hvx_mm_u8i8.c nntr_hvx_softmax.c nntr_hvx_attn.c nntr_hvx_dma_probe.c nntr_hvx_graph.c nntr_hvx_small_ops.c nntr_hvx_attn_m1.c nntr_hvx_dspq_bench.c nntr_hvx_dspq.c nntr_hvx_token.c generated/nntr_hvx_skel.c"
SRCS="$SRCS nntr_hvx_attn_m1_probe.c nntr_hvx_hadamard_probe.c build/hvx_hadamard_f32.o"
SRCS="$SRCS nntr_hvx_vlut_probe.c nntr_hvx_tl2_probe.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_mm_u8i4.c $BACKEND/hmx/hexkl_mm_u8i4_dma.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_mm_u8i4_moe.c $BACKEND/hmx/hexkl_graph.c $BACKEND/hmx/hexkl_token.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_conv_block.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_mm_u8i8_dma.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_dma_ring.c $BACKEND/hmx/hexkl_dma_trace.c $BACKEND/hmx/hexkl_kv_quant.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_probe.c $BACKEND/hmx/hexkl_acc_tile.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_attn_dtype.c $BACKEND/hmx/hexkl_attn_u8.c"
SRCS="$SRCS $BACKEND/hvx/hvx_quant_u8.c $BACKEND/hvx/hvx_dequant_i32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_swiglu_f32.c $BACKEND/hvx/hvx_conv_gate_f32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_m1_ops_f32.c $BACKEND/hvx/hvx_attn_m1_f32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_scale_add_f32.c $BACKEND/hvx/hvx_rmsnorm_rows_f32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_router_rows_f32.c $BACKEND/hvx/hvx_rope_rows_f32.c $BACKEND/hvx/hvx_softcap_f32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_gather_ah_u8.c"
SRCS="$SRCS $BACKEND/hvx/hvx_softmax_f32.c $BACKEND/hvx/hvx_softmax_blocked_f32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_worker_pool.c $BACKEND/hvx/hvx_gemm_u8i4_wh.c"
SRCS="$SRCS $BACKEND/hvx/hvx_expand_i2i4.c $BACKEND/hvx/hvx_gemv_u8tl2.c"
SRCS="$SRCS nntr_hvx_fc_q4.c build/hvx_q4_gemv_f32.o build/nntr_hvx_sf_probe.o"
SRCS="$SRCS nntr_hvx_mailbox.c"
# nntrainer/nntrainer#4343: fp16 and quantized (int8 / int4) KV-cache attention
SRCS="$SRCS nntr_hvx_attn_f16.c nntr_hvx_attn_q.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_attn_f16.c $BACKEND/hmx/hexkl_kv_tiles_f16.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_attn_q.c $BACKEND/hmx/hexkl_kv_q.c"
SRCS="$SRCS $BACKEND/hvx/hvx_attn_decode_f16.c $BACKEND/hvx/hvx_attn_decode_q.c"
SRCS="$SRCS $BACKEND/hvx/hvx_kv_quant.c"
SRCS="$SRCS $BACKEND/hvx/hvx_softmax_q.c $BACKEND/hmx/hexkl_attn_q2.c"

"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" \
    -m"$HEX_ARCH" -mhvx -mhvx-length=128B -mhmx -G0 -O3 -fPIC -shared \
    -Wall -Werror \
    ${HEX_EXTRA_CFLAGS:-} \
    -I generated \
    -I "$HEXKL_ROOT/include" \
    -I "$BACKEND/.." \
    -I "$BACKEND" \
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

READELF="$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-readelf"
# Runtime imports the DSP image provides: FastRPC HAP_*, compute_resource_*,
# QuRT, compiler-rt/CRT (__hexagon_*, __extendhfsf2, __cxa_finalize,
# __register_frame_info_bases) and libc. Anything else -- in particular a
# hexkl_*/hvx_*/nntr_* function -- is a project file missing from SRCS; the
# linker accepts it, the on-device loader does not (#97: 0x80000406).
# ceil, ldexp, ldexpf, lround and _Log (#260: #4415's int8 attention,
# hvx_softmax_q.c / hexkl_attn_q2.c) sit in the toolchain's libc.a beside
# the sqrtf / rintf / lroundf already listed; libm.a is an empty archive.
# rintf, sqrtf, memcmp and __truncsfhf2 (the sibling of __extendhfsf2) came in
# with the fp16 / quantized attention kernels of nntrainer/nntrainer#4343.
UND=$("$READELF" --dyn-syms build/libnntr_hvx_skel.so | awk '$7=="UND" && $8!="" {print $8}')
# A real skel always imports HAP_*/qurt_*: an empty list means readelf is
# missing or its column layout changed, and the check below would pass
# vacuously.
if [ -z "$UND" ]; then
    echo "Error: $READELF returned no undefined symbols; guard cannot run" >&2
    exit 1
fi
BAD=$(echo "$UND" | grep -Ev '^(HAP_|compute_resource_|qurt_|dspqueue_|__hexagon_|__extendhfsf2$|__cxa_finalize$|__register_frame_info_bases$|malloc$|free$|calloc$|memalign$|memcpy$|memset$|lroundf$|nearbyintf$|snprintf$|vsnprintf$|strlcpy$|rintf$|sqrtf$|memcmp$|__truncsfhf2$|ceil$|ldexp$|ldexpf$|lround$|_Log$)' || true)
if [ -n "$BAD" ]; then
    echo "Error: skel has undefined symbols the DSP image will not provide:" >&2
    echo "$BAD" | sed 's/^/  /' >&2
    echo "Add the defining .c to SRCS in $0 (see #97)." >&2
    exit 1
fi
# [#141] dspqueue_* is optional in the DSP image. A non-weak import would stop
# the whole skel from loading (0x80000406) where dspqueue is absent, taking
# the production app down with the bench; weak, the bench's start returns
# AEE_EUNSUPPORTED instead.
STRONG_DSPQ=$("$READELF" --dyn-syms build/libnntr_hvx_skel.so |
    awk '$7=="UND" && $8 ~ /^dspqueue_/ && $5!="WEAK" {print $8}')
if [ -n "$STRONG_DSPQ" ]; then
    echo "Error: dspqueue_* must be imported WEAK (#pragma weak), found:" >&2
    echo "$STRONG_DSPQ" | sed 's/^/  /' >&2
    exit 1
fi
echo "UNDEFINED SYMBOLS OK ($(echo "$UND" | wc -l) runtime imports)"
# Both arches write the same file name (the phone loads it by name), so the
# ELF e_flags are the only thing that tells a v79 skel from a v81 one.
if ! "$READELF" -h build/libnntr_hvx_skel.so | grep -q "Flags:.*0x${HEX_ARCH#v}\b"; then
    echo "Error: skel ELF flags do not say $HEX_ARCH:" >&2
    "$READELF" -h build/libnntr_hvx_skel.so | grep Flags: >&2
    exit 1
fi
echo "ARCH OK ($(echo "$HEX_ARCH" | tr v V))"

echo "built: $SCRIPT_DIR/build/libnntr_hvx_skel.so ($HEX_ARCH, hexkl $HEXKL_SDK_VER)"
echo "NOTE: this is the DSP skel only. If nntr_hvx.idl changed, the ARM client"
echo "      stub needs regenerating too, or the build fails on the new symbols"
echo "      (or worse, an old client meets this skel and the call returns"
echo "      EBADPARM):"
echo "        bash $REPO_ROOT/nntrainer/tensor/htp_backend/generate_stub.sh"
