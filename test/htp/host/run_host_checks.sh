#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Runs the HTP kernels' host checks: no device, no Hexagon SDK, no HMX.
#
# What these are and are not. The HMX and HVX primitives are replaced by
# scalar stand-ins in stub/, standin/ and the check itself, so this does NOT
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
  -I "$HERE/stub" -I "$HERE/standin" -I "$HERE/.." -I "$BACKEND/.." \
  -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -o "$OUT/moe_layer_host_check" \
  "$HERE/moe_layer_host_check.c" "$HERE/standin/hvx_scalar.c" \
  "$BACKEND/hmx/hexkl_mm_u8i4_moe.c" "$BACKEND/hmx/hexkl_dma_trace.c" -lm

"$OUT/moe_layer_host_check"

# The conv block kernel (doc 51 section 2) on the same stand-ins. It is
# built on the MoE kernel's exported helpers, so that file links in too.
# -ffp-contract=off: the conv gate's reference is a separate multiply and
# add per tap, as the HVX computes it, and the host compiler must not fuse
# them into an FMA the device does not have.
"$cc" -std=c99 -O1 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
  -I "$HERE/stub" -I "$HERE/standin" -I "$BACKEND/.." -I "$BACKEND/hmx" \
  -I "$BACKEND/hvx" \
  -o "$OUT/conv_block_host_check" \
  "$HERE/conv_block_host_check.c" "$HERE/hvx_scalar_stubs.c" \
  "$HERE/standin/hvx_scalar.c" \
  "$BACKEND/hmx/hexkl_conv_block.c" "$BACKEND/hmx/hexkl_mm_u8i4_moe.c" \
  "$BACKEND/hmx/hexkl_dma_trace.c" -lm

"$OUT/conv_block_host_check"

# The FC / projection call (hexkl_mm_u8i4_layer_run) on the same stand-ins:
# several handles against one activation, the pooled epilogue's batching.
"$cc" -std=c99 -O1 -Wall -Wextra -Wno-unused-parameter \
  -I "$HERE/stub" -I "$HERE/standin" -I "$BACKEND/.." -I "$BACKEND/hmx" \
  -I "$BACKEND/hvx" \
  -o "$OUT/fc_layer_host_check" \
  "$HERE/fc_layer_host_check.c" "$HERE/hvx_scalar_stubs.c" \
  "$HERE/standin/hvx_scalar.c" \
  "$BACKEND/hmx/hexkl_mm_u8i4_dma.c" -lm

"$OUT/fc_layer_host_check"

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

# The per-token entry (#85): htp_graph_desc.h's validator and LFM2
# builder (the LFM2.5 list validates, each mutation fails with its own
# code, never AEE_EBADPARM) and hexkl_graph.c's forward loop on the tiny
# fixture's shapes -- the identity with nothing resident, and with MOE
# resident byte-equal to a direct layer_run call on a recording stand-in
# of the kernel (the kernel's own loops are the first check's). Since
# #130 the same binary links the REAL small-op and m=1 attention sources
# on hvx_emu/ (and the worker pool on pthreads), and memcmp's each
# resident stretch's output against the scalar specs -- so the same
# -ffp-contract=off / -include malloc.h flags as the two checks below.
"$cc" -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
  -pthread -include malloc.h \
  -I "$HERE/hvx_emu" -I "$HERE/stub" -I "$BACKEND/.." -I "$BACKEND" \
  -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -o "$OUT/graph_host_check" \
  "$HERE/graph_host_check.c" "$BACKEND/hmx/hexkl_graph.c" \
  "$BACKEND/hvx/hvx_m1_ops_f32.c" "$BACKEND/hvx/hvx_conv_gate_f32.c" \
  "$BACKEND/hvx/hvx_attn_m1_f32.c" "$BACKEND/hvx/hvx_worker_pool.c" \
  "$BACKEND/hvx/hvx_scale_add_f32.c" -lm

"$OUT/graph_host_check"

# The M=1 small ops (#82): the REAL HVX sources hvx_m1_ops_f32.c and
# hvx_conv_gate_f32.c compiled against hvx_emu/ (one IEEE f32 op per lane,
# no stand-in) and memcmp'd with the scalar spec nntrainer/tensor/
# m1_ops_det.h, plus the spec's tolerance against a double reference.
# -ffp-contract=off and no -ffast-math: the spec rounds every op on its
# own and keeps subnormals, and the host compiler must too.
"$cc" -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
  -I "$HERE/hvx_emu" -I "$BACKEND/.." -I "$BACKEND/hvx" \
  -o "$OUT/m1_ops_host_check" \
  "$HERE/m1_ops_host_check.c" "$BACKEND/hvx/hvx_m1_ops_f32.c" \
  "$BACKEND/hvx/hvx_conv_gate_f32.c" -lm

"$OUT/m1_ops_host_check"

# Decode attention at m=1 (#81): the REAL HVX source hvx_attn_m1_f32.c on
# hvx_emu/ with the REAL worker pool on pthreads (stub/qurt.h) at 0, 3 and
# 7 workers, memcmp'd against nntrainer/tensor/attn_m1_det.h for L = 1,
# 63, 64, 65, 512, 1024, plus append-chain == bulk, the L = 1 case and the
# error codes; the spec against a double reference within 2^-13 max|V|.
# -include malloc.h: the cache is memalign(128), which the Hexagon libc
# declares in stdlib.h and glibc in malloc.h.
"$cc" -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
  -pthread -include malloc.h \
  -I "$HERE/hvx_emu" -I "$HERE/stub" -I "$BACKEND/.." -I "$BACKEND/hvx" \
  -o "$OUT/attn_m1_host_check" \
  "$HERE/attn_m1_host_check.c" "$BACKEND/hvx/hvx_attn_m1_f32.c" \
  "$BACKEND/hvx/hvx_worker_pool.c" -lm

"$OUT/attn_m1_host_check"
