#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One-time, human-driven setup for the Mac client. Walks through the steps
# that only a person can do (install the container runtime, log in to
# Qualcomm, copy model files) and verifies each one. Re-run it any time;
# completed steps are skipped.
#
#   tools/docker/setup_wizard.sh            # all steps
#   tools/docker/setup_wizard.sh sdk        # only the SDK step
#
# Steps: runtime | image | sdk | model | check

set -euo pipefail

# OrbStack installs its docker CLI under ~/.orbstack/bin; make it visible to non-login shells.
[ -d "$HOME/.orbstack/bin" ] && export PATH="$HOME/.orbstack/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HEXAGON_SDK_DIR="${HEXAGON_SDK_DIR:-$HOME/Qualcomm/Hexagon_SDK}"
QPM_DEB_DIR="${QPM_DEB_DIR:-$HOME/Qualcomm/downloads}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/Applications/CausalLM/res/qwen3/qwen3-0.6b}"
IMAGE="${NNTR_DOCKER_IMAGE:-nntrainer-hexagon-dev:ubuntu24.04}"

say()  { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m   ok: %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m   WARN: %s\033[0m\n' "$*"; }
todo() { printf '\033[1;33m   TODO: %s\033[0m\n' "$*"; }
ask()  { read -r -p "   $1 [Enter to continue] " _; }

step_runtime() {
  say "1/5 container runtime"
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    ok "docker is available: $(docker version --format '{{.Server.Platform.Name}} {{.Server.Version}}' 2>/dev/null || echo unknown)"
  else
    todo "install OrbStack (recommended, fastest amd64 emulation on Apple silicon):"
    echo  "      brew install --cask orbstack        # or download from https://orbstack.dev"
    echo  "      open -a OrbStack                    # first launch creates the docker socket"
    echo  "   (Docker Desktop also works: enable 'Use Rosetta for x86_64/amd64 emulation' in Settings > General.)"
    ask "Install it, then press"
    docker info >/dev/null 2>&1 || { echo "   docker still not reachable; rerun the wizard later."; exit 1; }
    ok "docker is available"
  fi
  local arch
  arch="$(docker run --rm --platform linux/amd64 ubuntu:24.04 uname -m 2>/dev/null || true)"
  if [ "$arch" = "x86_64" ]; then ok "linux/amd64 emulation works"; else
    echo "   linux/amd64 containers do not run (got '$arch'). Enable Rosetta/amd64 emulation in the runtime settings."; exit 1; fi
}

step_image() {
  say "2/5 dev image"
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then ok "$IMAGE exists (rebuild with: tools/docker/run.sh --build-image)"; else
    echo "   building $IMAGE (downloads NDK r26d ~1 GB; 10-20 min)"; "$SCRIPT_DIR/run.sh" --build-image; ok "built"; fi
}

step_sdk() {
  say "3/5 Hexagon SDK (6.4 or newer) under $HEXAGON_SDK_DIR"
  mkdir -p "$HEXAGON_SDK_DIR" "$QPM_DEB_DIR"
  local found
  found="$(ls -d "$HEXAGON_SDK_DIR"/6.*/ 2>/dev/null | sort -V | tail -1 || true)"
  if [ -n "$found" ] && [ -f "$found/setup_sdk_env.source" ]; then ok "SDK present: $found"; return; fi
  local deb
  deb="$(ls "$QPM_DEB_DIR"/QualcommPackageManager3*Linux*x86*.deb 2>/dev/null | head -1 || true)"
  if [ -z "$deb" ]; then
    todo "download the LINUX x86 .deb of Qualcomm Package Manager 3 (qpm-cli) with your Qualcomm account:"
    echo  "      https://qpm.qualcomm.com  ->  Download  ->  Linux (x86_64) .deb"
    echo  "      save it into: $QPM_DEB_DIR"
    echo  "   (The Linux build is used because the SDK is installed inside the amd64 container; a macOS qpm is not needed.)"
    ask "Put the .deb there, then press"
    deb="$(ls "$QPM_DEB_DIR"/QualcommPackageManager3*Linux*x86*.deb 2>/dev/null | head -1 || true)"
    [ -n "$deb" ] || { echo "   no .deb found in $QPM_DEB_DIR"; exit 1; }
  fi
  echo "   installing qpm-cli in a throwaway root container and running the interactive login + install."
  echo "   You will be asked for your Qualcomm credentials. Product name: hexagonsdk6.x; the newest version is installed"
  echo "   unless QPM_SDK_VERSION=<x.y.z.w> is set (the --info line printed before the install lists the versions)."
  ask "Press to start"
  docker run --rm -it --platform linux/amd64 \
    -e QPM_SDK_VERSION="${QPM_SDK_VERSION:-}" \
    -v "$QPM_DEB_DIR:/deb:ro" -v "$HEXAGON_SDK_DIR:/opt/qcom/Hexagon_SDK" \
    ubuntu:24.04 bash -lc '
      set -e
      apt-get update >/dev/null && apt-get install -y --no-install-recommends ca-certificates libglib2.0-0 librtmp1 libsasl2-2 >/dev/null
      dpkg -i /deb/QualcommPackageManager3*Linux*x86*.deb || apt-get install -y -f
      qpm-cli --version
      qpm-cli --login
      qpm-cli --license-activate hexagonsdk6.x
      qpm-cli --info hexagonsdk6.x || true
      if [ -n "${QPM_SDK_VERSION:-}" ]; then
        qpm-cli --install hexagonsdk6.x --version "$QPM_SDK_VERSION" --path /opt/qcom/Hexagon_SDK
      else
        qpm-cli --install hexagonsdk6.x --path /opt/qcom/Hexagon_SDK
      fi
      ls /opt/qcom/Hexagon_SDK
    '
  found="$(ls -d "$HEXAGON_SDK_DIR"/6.*/ 2>/dev/null | sort -V | tail -1 || true)"
  [ -n "$found" ] && [ -f "$found/setup_sdk_env.source" ] && ok "SDK installed: $found" || { echo "   SDK not found after install; check the qpm-cli output."; exit 1; }
}

step_model() {
  say "4/5 model files in $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  local bin="$MODEL_DIR/nntr_qwen3_0.6b_w8cx_DEFAULT.bin" hf="$MODEL_DIR/hf"
  if [ ! -f "$hf/model.safetensors" ] || [ ! -f "$hf/tokenizer.json" ]; then
    echo "   downloading Qwen/Qwen3-0.6B (public, ~1.5 GB) into $hf"
    "$SCRIPT_DIR/run.sh" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='/work/${MODEL_DIR#$REPO_ROOT/}/hf',
                  allow_patterns=['*.json', '*.txt', 'model.safetensors'])"
  fi
  ok "HF checkpoint + tokenizer present ($hf)"
  if [ ! -f "$bin" ]; then
    echo "   building the W8_CX checkpoint from the HF weights (tools/hexagon/make_w8cx_bin.py, ~2 min)"
    "$SCRIPT_DIR/run.sh" python3 tools/hexagon/make_w8cx_bin.py \
      "/work/${MODEL_DIR#$REPO_ROOT/}/hf" "/work/${MODEL_DIR#$REPO_ROOT/}/nntr_qwen3_0.6b_w8cx_DEFAULT.bin" | tail -1
  fi
  ok "W8_CX checkpoint present ($(stat -f %z "$bin" 2>/dev/null || stat -c %s "$bin") bytes; expected 598230528)"
  # The md5 is informational: another HuggingFace snapshot legitimately yields a
  # different .bin (HEXAGON.md section 5.1). The size line above is the hard gate.
  local expected_md5=7562313bb4cc70410450d0ab3a4fa563 md5
  md5="$(md5 -q "$bin" 2>/dev/null || md5sum "$bin" | cut -d' ' -f1)"
  echo "   md5 $md5"
  if [ "$md5" != "$expected_md5" ]; then
    warn "md5 differs from the #23 baseline ($expected_md5): a different HF snapshot, or a stale/partial .bin"
  fi
}

step_check() {
  say "5/5 smoke test inside the container"
  "$SCRIPT_DIR/run.sh" bash -c 'echo "sdk: ${HEXAGON_SDK_ROOT:-<none>}"; echo "ndk: $ANDROID_NDK"; which hexagon-clang qaic clang-format-14 meson ninja g++ 2>/dev/null || true; python3 -c "import transformers, numpy; print(\"python ok\")"'
  "$SCRIPT_DIR/run.sh" ./tools/hexagon/build_host_x86.sh >/dev/null && "$SCRIPT_DIR/run.sh" ./build_x86_hexagon/test_lowering
  "$SCRIPT_DIR/run.sh" ./build_x86_hexagon/test_w8cx_bin /model/nntr_qwen3_0.6b_w8cx_DEFAULT.bin
  ok "container builds the x86 tools; lowering and W8_CX reader tests pass"
  echo
  echo "Done. Next: in Claude Code run /hexagon-cycle (issue #23 rebuilds hvx_impl with the new SDK)."
}

case "${1:-all}" in
  runtime) step_runtime ;;
  image)   step_runtime; step_image ;;
  sdk)     step_runtime; step_sdk ;;
  model)   step_model ;;
  check)   step_check ;;
  all)     step_runtime; step_image; step_sdk; step_model; step_check ;;
  *) echo "usage: $0 [runtime|image|sdk|model|check|all]"; exit 1 ;;
esac
