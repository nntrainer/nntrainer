#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Diagnostic entry point; the production DSP build is owned by the backend.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../../nntrainer/tensor/htp_backend/dsp/build.sh" "$@"
