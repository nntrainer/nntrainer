#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../../nntrainer/tensor/htp_backend/dsp/generate_rpc.sh" "$@"
