#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One IDL and qaic invocation for the ARM stub and DSP skel.
set -eu
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT to a Hexagon SDK checkout}"
if [[ $# != 2 || ( "$1" != host && "$1" != dsp ) ]]; then
    echo "Usage: $0 host|dsp output-directory" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QAIC="$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic"
if [[ ! -x "$QAIC" ]]; then
    echo "Error: qaic not found or not executable: $QAIC" >&2
    exit 1
fi
mkdir -p "$2"
OUT_DIR="$(cd "$2" && pwd)"
"$QAIC" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -mdll -o "$OUT_DIR" "$SCRIPT_DIR/nntr_hvx.idl"
if [[ "$1" == host ]]; then
    rm -f "$OUT_DIR/nntr_hvx_skel.c"
fi
