#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint: locate the mounted Hexagon SDK, source its
# environment (setup_sdk_env.source exports HEXAGON_SDK_ROOT,
# DEFAULT_HEXAGON_TOOLS_ROOT and puts hexagon-clang on PATH), then exec the
# requested command. Works without an SDK too (x86-only builds).

set -eu

BASE="${HEXAGON_SDK_BASE:-/opt/qcom/Hexagon_SDK}"

if [ -z "${HEXAGON_SDK_ROOT:-}" ]; then
  if [ -n "${HEXAGON_SDK_VERSION:-}" ] && [ -d "$BASE/$HEXAGON_SDK_VERSION" ]; then
    HEXAGON_SDK_ROOT="$BASE/$HEXAGON_SDK_VERSION"
  else
    # newest version directory, e.g. 6.4.0.2 sorts after 6.3.0.0
    HEXAGON_SDK_ROOT="$(ls -d "$BASE"/*/ 2>/dev/null | sort -V | tail -1 || true)"
    HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT%/}"
  fi
fi

if [ -n "${HEXAGON_SDK_ROOT:-}" ] && [ -f "$HEXAGON_SDK_ROOT/setup_sdk_env.source" ]; then
  # setup_sdk_env.source returns early when HEXAGON_SDK_ROOT is already set,
  # so hand it the path via BASH_SOURCE (it derives the root from its own
  # location) and let it export everything itself. It needs `python`, `make`
  # and `awk`, and it (re)links the qaic binary inside the SDK tree.
  sdk_setup="$HEXAGON_SDK_ROOT/setup_sdk_env.source"
  unset HEXAGON_SDK_ROOT
  set +u   # the SDK script tests unset variables
  # shellcheck disable=SC1090
  source "$sdk_setup" >/dev/null
  set -u
  if [ -z "${DEFAULT_HEXAGON_TOOLS_ROOT:-}" ] || [ ! -d "$DEFAULT_HEXAGON_TOOLS_ROOT" ]; then
    echo "nntr-entrypoint: setup_sdk_env.source did not set DEFAULT_HEXAGON_TOOLS_ROOT (got '${DEFAULT_HEXAGON_TOOLS_ROOT:-}')" >&2
    [ "${NNTR_REQUIRE_SDK:-0}" = "1" ] && exit 2
  fi
elif [ "${NNTR_REQUIRE_SDK:-0}" = "1" ]; then
  echo "nntr-entrypoint: no Hexagon SDK under $BASE (run tools/docker/setup_wizard.sh)" >&2
  exit 2
fi

if [ -d /opt/qcom/hexkl_addon ] && [ -n "$(ls -A /opt/qcom/hexkl_addon 2>/dev/null)" ]; then
  export HEXKL_ADDON=/opt/qcom/hexkl_addon
fi

exec "$@"
