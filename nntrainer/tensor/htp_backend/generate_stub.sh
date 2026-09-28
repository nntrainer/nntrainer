#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Generates the FastRPC client stub HtpBackend links against, from the same
# IDL the DSP skel (test/htp/build.sh) is built from -- one source of truth
# for the interface, not a copy kept in step by hand.
#
# Prerequisite: a Hexagon SDK checkout (for qaic and <remote.h>/<AEEStdErr.h>).
#   HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.2 ./generate_stub.sh
#
# Run this before configuring a build with -Denable-htp=true: meson.build
# errors out with this same instruction if generated/nntr_hvx_stub.c is
# missing, rather than silently building without HTP acceleration. Not a
# meson custom_target, deliberately: qaic is a proprietary SDK tool, not
# something a normal build host has, and test/htp/build.sh already
# established the convention of a manual generation step for it.

set -eu

: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT to a Hexagon SDK checkout}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/generated"

QAIC="$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic"
if [ ! -x "$QAIC" ]; then
  echo "Error: qaic not found or not executable: $QAIC" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

"$QAIC" \
  -I "$HEXAGON_SDK_ROOT/incs" \
  -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  -mdll -o "$OUT_DIR" \
  "$REPO_ROOT/test/htp/nntr_hvx.idl"

# The DSP-side skel source is not needed by this (host) stub.
rm -f "$OUT_DIR/nntr_hvx_skel.c"

echo "generated: $OUT_DIR/nntr_hvx.h, $OUT_DIR/nntr_hvx_stub.c"
