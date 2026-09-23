#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Runs the HTP kernels' host checks: no device, no Hexagon SDK, no HMX.
#
# What these are and are not. The HMX and HVX primitives are replaced by
# scalar stand-ins in stub/ and in the check itself, so this does NOT
# verify the hardware's arithmetic -- the device tests do that, and this
# would give false confidence if it were read as doing so. What it does
# verify is everything around the arithmetic: which rows each expert gets,
# which weights it uses, that the buffer reuse in the weight DMA pipeline
# does not clobber a live buffer, that the scatter lands on the right
# output row with the right routing weight, that empty and
# multi-block experts work, and that the VTCM layout matches the hand
# arithmetic in docs/htp_attention/46_moe_resident_kernel_design.md.
#
# The DMA stand-in completes immediately, which is the point: a transfer
# issued into a buffer that is still being read shows up as a wrong result
# here rather than as a rare corruption on device.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$(cd "$HERE/../../../nntrainer/tensor/htp_backend" && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

cc=${CC:-gcc}
# -O2: the M=1 cases run the HMX stand-in at the real shape (64 rows a
# tile, scalar); about half a minute at -O2 and several at -O1, since the
# HMX reference is computed once per (shape, M) and reused. One build since
# #113: the l2fetch lead and the GEMV row loop are a per-call field of the
# moe_set_opts word, so the check itself sweeps the whole
# {0, 192, 384, 768, 1536} KB x {rows4, rows1} matrix plus the build's own
# defaults, instead of this loop compiling one configuration at a time;
# since #117 that matrix twice, with the arena read and with the VTCM
# feed, whose push/wait schedule the check's scoreboard holds
# (M1 GEMV VTCM FEED SCHEDULE OK).
"$cc" -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter \
  -DMOE_TAIL_MAX_ROWS=16u \
  -I "$HERE/stub" -I "$HERE/.." -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -o "$OUT/moe_layer_host_check" \
  "$HERE/moe_layer_host_check.c" "$BACKEND/hmx/hexkl_mm_u8i4_moe.c" \
  "$BACKEND/hmx/hexkl_dma_trace.c" -lm

"$OUT/moe_layer_host_check"

# The worker pool's two lanes on pthreads (stub/qurt.h). Concurrency is
# exercised for real here -- 3 workers, a caller that helps -- but a
# desktop scheduler is not QuRT's; the device is still where the timing
# and the HVX-context sharing are checked.
"$cc" -std=c11 -O1 -Wall -Wextra -Wno-unused-parameter -pthread \
  -I "$HERE/stub" -I "$BACKEND/hvx" \
  -o "$OUT/worker_pool_host_check" \
  "$HERE/worker_pool_host_check.c" "$BACKEND/hvx/hvx_worker_pool.c"

"$OUT/worker_pool_host_check"

# The DMA probe's descriptor plan (test/htp/nntr_dma_probe_plan.h): the
# skel runs the same header-only function, so the geometry checked here --
# disjoint rows, in-bounds destinations, worker balance -- is what the
# device DMAs. Bandwidth is measured on the device, never here.
"$cc" -std=c99 -O1 -Wall -Wextra -Wno-unused-parameter \
  -I "$HERE/.." \
  -o "$OUT/dma_probe_host_check" \
  "$HERE/dma_probe_host_check.c"

"$OUT/dma_probe_host_check"

# The M=1 GEMV default and the MoE profile row's rest (htp_moe_opts.h,
# #101/#102): pure arithmetic on #94 sitting 2's printed rows.
"$cc" -std=c99 -O1 -Wall -Wextra -Wno-unused-parameter \
  -I "$BACKEND" \
  -o "$OUT/moe_opts_host_check" \
  "$HERE/moe_opts_host_check.c" -lm

"$OUT/moe_opts_host_check"

# The MoE call's DMA ring trace (hexkl_dma_trace.c, #87): completion
# brackets, the union-of-intervals busy time, ring depth and the blocked
# bit on a scripted timeline whose expected numbers are hand arithmetic.
"$cc" -std=c99 -O1 -Wall -Wextra -Wno-unused-parameter \
  -I "$HERE/stub" -I "$BACKEND/hmx" \
  -o "$OUT/dma_trace_host_check" \
  "$HERE/dma_trace_host_check.c" "$BACKEND/hmx/hexkl_dma_trace.c"

"$OUT/dma_trace_host_check"

# The #100 replay cells (nntr_moe_dma_plan.h): the 16 lists the gtest
# replays, inside the skel's parse rules and the VTCM bound, and the tag
# sum the gtest expects, simulated and held against a byte-by-byte copy.
"$cc" -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter \
  -I "$HERE/.." \
  -o "$OUT/replay_cells_host_check" \
  "$HERE/replay_cells_host_check.c"

"$OUT/replay_cells_host_check"

# The skel's dma_replay entry itself (nntr_hvx_dma_probe.c, compiled as-is
# against replay_stub/: a descriptor lands whole when started or linked):
# every #100 cell's res[12] equals the tag simulator the gtest uses. The
# device's DMA timing and interleaving are not modelled here.
# (memalign is in the Hexagon libc's stdlib.h, in glibc's malloc.h.)
"$cc" -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -pthread \
  -include malloc.h -I "$HERE/replay_stub" -I "$HERE/stub" -I "$HERE/.." \
  -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -o "$OUT/dma_replay_host_check" \
  "$HERE/dma_replay_host_check.c" "$HERE/../nntr_hvx_dma_probe.c" \
  "$BACKEND/hmx/hexkl_dma_ring.c" "$BACKEND/hmx/hexkl_dma_trace.c" \
  "$BACKEND/hvx/hvx_worker_pool.c"

"$OUT/dma_replay_host_check"

# The real HVX GEMV (hvx_gemm_u8i4_wh.c, the skel's own source) on x86
# against the Hexagon tools' HVX emulation, libnative: every stand-in above
# replaces the kernel, this runs it. g++ links because libnative.a is C++.
# Then a mutation self-test: the same check against a copy of the kernel
# with one token changed must fail, or the check is not looking.
LIBNATIVE="${DEFAULT_HEXAGON_TOOLS_ROOT:-}/Tools/libnative"
if [ -f "$LIBNATIVE/lib/libnative.a" ]; then
  gemv_native() { # gemv_native <kernel.c> <exe>
    "$cc" -std=gnu99 -O1 -fno-strict-aliasing -DHVX_UVector=HEXAGON_Vect1024 \
      -I "$LIBNATIVE/include" -I "$BACKEND/hvx" -c "$1" -o "$2.k.o"
    "$cc" -std=gnu99 -O1 -Wall -Wextra -I "$BACKEND/hvx" \
      -c "$HERE/gemv_native_check.c" -o "$2.c.o"
    g++ -o "$2" "$2.c.o" "$2.k.o" "$LIBNATIVE/lib/libnative.a"
  }
  gemv_native "$BACKEND/hvx/hvx_gemm_u8i4_wh.c" "$OUT/gemv_native_check"
  "$OUT/gemv_native_check"
  # One mutant per loop: the one-row loop's final shift (caught only on
  # the lone rows m = 1, 5, 9, 13, which is how this shows that loop runs)
  # and the four-row loop's. Sending m = 1 to the four-row loop is not a
  # mutant: its row 0 is the same int32 by construction.
  for mut in 's/vasr_VwR(acc, 4)/vasr_VwR(acc, 3)/' \
    's/vasr_VwR(acc0, 4)/vasr_VwR(acc0, 3)/'; do
    sed "$mut" "$BACKEND/hvx/hvx_gemm_u8i4_wh.c" > "$OUT/mutant.c"
    if cmp -s "$OUT/mutant.c" "$BACKEND/hvx/hvx_gemm_u8i4_wh.c"; then
      echo "HVX GEMV MUTATION DID NOT APPLY: $mut"; exit 1
    fi
    gemv_native "$OUT/mutant.c" "$OUT/gemv_mutant"
    if "$OUT/gemv_mutant" > "$OUT/mutant.log"; then
      echo "HVX GEMV MUTANT PASSED (the check is blind): $mut"; exit 1
    fi
    echo "HVX GEMV MUTANT CAUGHT: $mut ($(grep -o 'bad=[0-9]*' "$OUT/mutant.log" | tail -1))"
  done
else
  echo "HVX GEMV NATIVE CHECK SKIPPED (no $LIBNATIVE/lib/libnative.a;" \
    "source tools/htp/env.sh)"
fi
