#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run one command inside the nntrainer Hexagon dev container with the repo,
# the Hexagon SDK, HexKL and the model directory mounted.
#
#   tools/docker/run.sh ./tools/hexagon/build_host_x86.sh
#   HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/run_sim_test.sh profile acc
#   tools/docker/run.sh --build-image            # (re)build the image only
#   tools/docker/run.sh                          # interactive shell
#
# Host-side knobs (environment):
#   NNTR_DOCKER_IMAGE   image tag            (default nntrainer-hexagon-dev:ubuntu24.04)
#   HEXAGON_SDK_DIR     host dir holding <version>/ SDK trees (default ~/Qualcomm/Hexagon_SDK)
#   HEXAGON_SDK_VERSION pick one version dir (default: newest)
#   HEXKL_ADDON_DIR     host dir with libhexkl_micro.a etc. (default ~/Qualcomm/hexkl_addon, optional)
#   MODEL_DIR           mounted read-only at /model (default Applications/CausalLM/res/qwen3/qwen3-0.6b)
#   NNTR_DOCKER_ROOT=1  run as root (needed only by setup_wizard.sh)
# Passed through into the container: HEX_ARCH, HEX_EXTRA_CFLAGS, SIM_TIMING,
# NNTR_NUM_THREADS, NNTR_REQUIRE_SDK, HEXAGON_SDK_VERSION.

set -euo pipefail

# OrbStack installs its docker CLI under ~/.orbstack/bin; make it visible to non-login shells.
[ -d "$HOME/.orbstack/bin" ] && export PATH="$HOME/.orbstack/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${NNTR_DOCKER_IMAGE:-nntrainer-hexagon-dev:ubuntu24.04}"
HEXAGON_SDK_DIR="${HEXAGON_SDK_DIR:-$HOME/Qualcomm/Hexagon_SDK}"
HEXKL_ADDON_DIR="${HEXKL_ADDON_DIR:-$HOME/Qualcomm/hexkl_addon}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/Applications/CausalLM/res/qwen3/qwen3-0.6b}"
PLATFORM=linux/amd64

if ! command -v docker >/dev/null 2>&1; then
  echo "run.sh: docker not found. Install OrbStack (https://orbstack.dev) or Docker Desktop, then retry." >&2
  exit 1
fi

build_image() {
  docker build --platform "$PLATFORM" -t "$IMAGE" "$SCRIPT_DIR"
}

if [ "${1:-}" = "--build-image" ]; then
  build_image
  exit 0
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "run.sh: image $IMAGE not found, building it (takes a while: NDK download ~1 GB)" >&2
  build_image
fi

args=(run --rm --platform "$PLATFORM" -v "$REPO_ROOT:/work" -w /work)
if [ -t 0 ] && [ -t 1 ]; then args+=(-it); fi
if [ "${NNTR_DOCKER_ROOT:-0}" != "1" ]; then args+=(--user "$(id -u):$(id -g)"); fi

mkdir -p "$HEXAGON_SDK_DIR"
args+=(-v "$HEXAGON_SDK_DIR:/opt/qcom/Hexagon_SDK")
[ -d "$HEXKL_ADDON_DIR" ] && args+=(-v "$HEXKL_ADDON_DIR:/opt/qcom/hexkl_addon")
[ -d "$MODEL_DIR" ] && args+=(-v "$MODEL_DIR:/model:ro")

for v in HEX_ARCH HEX_EXTRA_CFLAGS SIM_TIMING NNTR_NUM_THREADS NNTR_REQUIRE_SDK HEXAGON_SDK_VERSION; do
  if [ -n "${!v:-}" ]; then args+=(-e "$v=${!v}"); fi
done
# HOME must be writable for a non-root uid that has no passwd entry (pip/transformers caches)
args+=(-e HOME=/tmp/nntr-home)

if [ $# -eq 0 ]; then
  exec docker "${args[@]}" "$IMAGE" bash
fi
exec docker "${args[@]}" "$IMAGE" "$@"
