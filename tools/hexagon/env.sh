#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
##
# @file    env.sh
# @brief   Environment for native Hexagon builds on the Ubuntu workstation
# @author  dlwlzzero <dlwlzzero@gmail.com>
#
# Since 2026-09-21 every gate (x86 reference, simulator, skel) runs natively
# on the workstation instead of through tools/docker/run.sh. Source this
# before any tools/hexagon/*.sh:
#
#     source tools/hexagon/env.sh
#
# Overrides: HEXAGON_SDK_ROOT (default 6.4.0.1 under /local/mnt/workspace),
# HEXKL_ADDON_ROOT (default ~/Qualcomm/hexkl_addon: include/ + lib/ pointing
# at the HexKL package's lib/<sdk version>/), NNTR_MODEL_DIR, HEX_ARCH.
# hexagon-sim links libncurses.so.5, which Ubuntu 24.04 no longer ships;
# the jammy copies live in ~/.local/lib/hexagon-compat (LD_LIBRARY_PATH).

_nntr_sdk="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1}"
if [ ! -f "$_nntr_sdk/setup_sdk_env.source" ]; then
  echo "env.sh: no SDK at $_nntr_sdk" >&2
  return 1 2>/dev/null || exit 1
fi
# setup_sdk_env.source returns at once when HEXAGON_SDK_ROOT is already set
# (a re-source would then leave DEFAULT_TOOLS_VARIANT and PATH untouched).
unset HEXAGON_SDK_ROOT
# shellcheck disable=SC1091
source "$_nntr_sdk/setup_sdk_env.source" >/dev/null
unset _nntr_sdk

export HEXKL_ADDON_ROOT="${HEXKL_ADDON_ROOT:-$HOME/Qualcomm/hexkl_addon}"
export NNTR_MODEL_DIR="${NNTR_MODEL_DIR:-/local/mnt/workspace/models/qwen3-0.6b}"
export HEX_ARCH="${HEX_ARCH:-v79}"
case ":${LD_LIBRARY_PATH:-}:" in
  *":$HOME/.local/lib/hexagon-compat:"*) ;;
  *) export LD_LIBRARY_PATH="$HOME/.local/lib/hexagon-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
# Tools/bin (hexagon-clang, hexagon-sim) and ~/.local/bin (clang-format-14)
for _d in "$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin" "$HOME/.local/bin"; do
  case ":$PATH:" in *":$_d:"*) ;; *) export PATH="$_d:$PATH" ;; esac
done
unset _d

echo "hexagon env: SDK $HEXAGON_SDK_ROOT ($DEFAULT_TOOLS_VARIANT), HexKL $HEXKL_ADDON_ROOT, model $NNTR_MODEL_DIR, arch $HEX_ARCH"
