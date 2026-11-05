#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Generate the ARM client from the backend's canonical DSP interface.
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/generated"
bash "$SCRIPT_DIR/dsp/generate_rpc.sh" host "$OUT_DIR"
echo "generated: $OUT_DIR/nntr_hvx.h, $OUT_DIR/nntr_hvx_stub.c"
echo "Rebuild the DSP skel with $SCRIPT_DIR/dsp/build.sh after an IDL change."
