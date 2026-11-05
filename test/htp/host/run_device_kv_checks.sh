#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Native DSP kernel checks using the repository's HVX emulation.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$repo_dir"
check_dir=$(mktemp -d "${TMPDIR:-/tmp}/nntr-device-kv.XXXXXX")
trap 'rm -rf "$check_dir"' EXIT HUP INT TERM
sanitizers=""
if [ "${SANITIZE:-0}" = 1 ]; then
  sanitizers="-fsanitize=address,undefined -fno-omit-frame-pointer"
fi
${CXX:-c++} -std=c++17 -O2 -Wall -Wextra -Werror $sanitizers \
  -I Applications/CausalLM test/htp/host/device_kv_manager_check.cpp \
  -o "$check_dir/manager"
"$check_dir/manager"
${CXX:-c++} -std=c++17 -O2 -Wall -Wextra -Werror $sanitizers \
  -I nntrainer/utils -I nntrainer/tensor/htp_backend \
  test/htp/host/decode_kv_convert_check.cpp nntrainer/utils/fp16.cpp \
  -o "$check_dir/convert"
"$check_dir/convert"
for name in attn_snapshot_check attn_paged_check; do
  ${CC:-cc} -std=gnu11 -O2 -ffp-contract=off -pthread -include malloc.h \
    $sanitizers -I test/htp/host/hvx_emu -I test/htp/host/stub \
    -I nntrainer/tensor -I nntrainer/tensor/htp_backend/hvx \
    "test/htp/host/$name.c" \
    nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_f32.c \
    nntrainer/tensor/htp_backend/hvx/hvx_worker_pool.c -lm \
    -o "$check_dir/$name"
  "$check_dir/$name"
done
