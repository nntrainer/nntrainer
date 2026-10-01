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

# [#90] The two-reader probe's ring cell (f2, fresh = 1) on the same
# skel source and stand-in, which here also counts the bytes it landed
# and the source pages it read: bytes = res[2] x res[1], one rotation's
# distinct footprint, the tag for 1 / 20 / 500 calls; then the DDR bounds
# (85.3 GB/s, #77's 3265.6 refused) and the net-per-token arithmetic.
"$cc" -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -pthread \
  -include malloc.h -I "$HERE/replay_stub" -I "$HERE/stub" -I "$HERE/.." \
  -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -o "$OUT/two_reader_host_check" \
  "$HERE/two_reader_host_check.c" "$HERE/../nntr_hvx_dma_probe.c" \
  "$BACKEND/hmx/hexkl_dma_ring.c" "$BACKEND/hmx/hexkl_dma_trace.c" \
  "$BACKEND/hvx/hvx_worker_pool.c" -lm

"$OUT/two_reader_host_check"

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
# Since #132 Part B also the Q4M1 kinds (FC, DENSE_FFN, LM_HEAD) on the
# REAL hvx_q4_gemv_f32.c against the CPU-order specs, then four mutants
# of hexkl_graph.c's Q4M1 kernels, each of which must fail the check.
graph_check() { # graph_check <hexkl_graph.c> <exe>
  "$cc" -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
    -Wno-format-truncation -pthread -include malloc.h \
    -I "$HERE/hvx_emu" -I "$HERE/stub" -I "$BACKEND/.." -I "$BACKEND" \
    -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
    -o "$2" \
    "$HERE/graph_host_check.c" "$1" \
    "$BACKEND/hvx/hvx_m1_ops_f32.c" "$BACKEND/hvx/hvx_conv_gate_f32.c" \
    "$BACKEND/hvx/hvx_attn_m1_f32.c" "$BACKEND/hvx/hvx_worker_pool.c" \
    "$BACKEND/hvx/hvx_scale_add_f32.c" "$BACKEND/hvx/hvx_q4_gemv_f32.c" -lm
}
graph_check "$BACKEND/hmx/hexkl_graph.c" "$OUT/graph_host_check"
"$OUT/graph_host_check"
# gate and up swapped; the part offset fixed at one group; down fed the
# FFN input's quantization; the argmax over the first slice only; [plan 201
# S1] the miss round's experts before the first miss dropped, and its later
# rows added in reverse order; [plan 201 S4] the softmax router run as the
# sigmoid one, and an RMSNORM's N1 bit ignored
for mut in 's/hvx_swiglu_cpu_f32(gate, up, act, op->N,/hvx_swiglu_cpu_f32(up, gate, act, op->N,/' \
  's/y += g->q4m1\[h\[p\]\].N;/y += Q4M1_GROUP;/' \
  's/graph_prep(op, act, op->N, &g->act);/(void)act;/' \
  's/hvx_argmax_first_f32(g->logits, op->N)/hvx_argmax_first_f32(g->logits, op->N \/ 2u)/' \
  's/  if (first != 0u) {/  if (0) {/' \
  's/    hvx_scale_add_rows_f32(out, g->moe_rows + (size_t)i \* op->N_out, 1.0f,/    hvx_scale_add_rows_f32(out, g->moe_rows + (size_t)(n - 1u - i + first) * op->N_out, 1.0f,/' \
  's/  if (op->eps_bits != 0u) {/  if (0) {/' \
  's/((op->feed \& HTP_GRAPH_NORM_N1) != 0u ? hvx_rmsnorm_n1_f32/(0 ? hvx_rmsnorm_n1_f32/'; do
  sed "$mut" "$BACKEND/hmx/hexkl_graph.c" > "$OUT/hexkl_graph_mutant.c"
  if cmp -s "$OUT/hexkl_graph_mutant.c" "$BACKEND/hmx/hexkl_graph.c"; then
    echo "GRAPH Q4M1 MUTATION DID NOT APPLY: $mut"; exit 1
  fi
  graph_check "$OUT/hexkl_graph_mutant.c" "$OUT/graph_mutant"
  if "$OUT/graph_mutant" > "$OUT/graph_mutant.log"; then
    echo "GRAPH Q4M1 MUTANT PASSED (the check is blind): $mut"; exit 1
  fi
  echo "GRAPH Q4M1 MUTANT CAUGHT: $mut ($(grep -c '^FAIL' "$OUT/graph_mutant.log") failed checks)"
done

# [#132 Part B E2] The two-session token driver (hmx/hexkl_token.c): S1 and
# S2 on two pthreads over the hd64 list split by mask, every MoE row across
# a malloc'd mailbox page, bit-identical to the one-session run for 10 000
# tokens (TOKEN DRIVER BIT-IDENTICAL), then the lost-post, stale-read and
# failed-side paths (TOKEN DRIVER FAILURE PATHS OK; ~3 s of timeouts).
# Same kernels and flags as the graph check. Then two mutants of
# hexkl_token.c, each of which must fail it: S2 reading its own row back
# as the MoE output, and the trailer check gone.
token_check() { # token_check <hexkl_token.c> <exe> [hexkl_graph.c]
  "$cc" -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
    -Wno-format-truncation -pthread -include malloc.h \
    -I "$HERE/hvx_emu" -I "$HERE/stub" -I "$BACKEND/.." -I "$BACKEND" \
    -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
    -o "$2" \
    "$HERE/token_host_check.c" "$1" "${3:-$BACKEND/hmx/hexkl_graph.c}" \
    "$BACKEND/hvx/hvx_m1_ops_f32.c" "$BACKEND/hvx/hvx_conv_gate_f32.c" \
    "$BACKEND/hvx/hvx_attn_m1_f32.c" "$BACKEND/hvx/hvx_worker_pool.c" \
    "$BACKEND/hvx/hvx_scale_add_f32.c" "$BACKEND/hvx/hvx_q4_gemv_f32.c" -lm
}
token_check "$BACKEND/hmx/hexkl_token.c" "$OUT/token_host_check"
"$OUT/token_host_check"
# [plan 201 S1] the pool's third mutant: the answer's evictions not
# cleared, so an evicted expert's table entry names the bytes its pair now
# holds (TOKEN POOL BIT-IDENTICAL must fail)
for mut in 's/    in = tk_row(theirs);/    in = tk_row(mine);/' \
  's/  if (\*(const uint32_t \*)(slot + HEXKL_MBOX_LINE + row) != seq) {/  if (0) {/' \
  's/  for (i = 0; rc == AEE_SUCCESS \&\& i < a->n_evict; ++i) {/  for (i = 0; 0 \&\& i < a->n_evict; ++i) {/'; do
  sed "$mut" "$BACKEND/hmx/hexkl_token.c" > "$OUT/hexkl_token_mutant.c"
  if cmp -s "$OUT/hexkl_token_mutant.c" "$BACKEND/hmx/hexkl_token.c"; then
    echo "TOKEN MUTATION DID NOT APPLY: $mut"; exit 1
  fi
  token_check "$OUT/hexkl_token_mutant.c" "$OUT/token_mutant"
  if "$OUT/token_mutant" > "$OUT/token_mutant.log"; then
    echo "TOKEN MUTANT PASSED (the check is blind): $mut"; exit 1
  fi
  echo "TOKEN MUTANT CAUGHT: $mut ($(grep -c '^FAIL' "$OUT/token_mutant.log") failed checks)"
done

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
# [plan 201 S4] The softmax router's renormalisation skipped in the spec
# (weight = p * per-expert scale): the kernel calls the same pick, so only
# the check against #4296's formula in double can see it -- it must.
mkdir -p "$OUT/mut_m1"
mut='s/  inv = swiglu_det_recip(t);/  inv = 1.0f;/'
sed "$mut" "$BACKEND/../m1_ops_det.h" > "$OUT/mut_m1/m1_ops_det.h"
if cmp -s "$OUT/mut_m1/m1_ops_det.h" "$BACKEND/../m1_ops_det.h"; then
  echo "ROUTER SOFTMAX MUTATION DID NOT APPLY: $mut"; exit 1
fi
"$cc" -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
  -I "$OUT/mut_m1" -I "$HERE/hvx_emu" -I "$BACKEND/.." -I "$BACKEND/hvx" \
  -o "$OUT/m1_ops_mutant" \
  "$HERE/m1_ops_host_check.c" "$BACKEND/hvx/hvx_m1_ops_f32.c" \
  "$BACKEND/hvx/hvx_conv_gate_f32.c" -lm
if "$OUT/m1_ops_mutant" > "$OUT/m1_ops_mutant.log"; then
  echo "ROUTER SOFTMAX MUTANT PASSED (the check is blind): $mut"; exit 1
fi
echo "ROUTER SOFTMAX MUTANT CAUGHT: renormalise skipped ($(grep -c '^FAIL: router softmax' "$OUT/m1_ops_mutant.log") failed checks)"

# [plan 201 S4] RoPE at Gemma 4's head_dim 256 / 512: the kernel pairing
# (i, i + 32) inside each 64-lane chunk -- LFM2's head_dim 64 RoPE reused
# per chunk, the same bytes at 64 -- must fail the hd256 / hd512 lines.
mut='s/\*vb = (HVX_UVector \*)(x + h);/*vb = (HVX_UVector *)(x + LANES);/'
sed "$mut" "$BACKEND/hvx/hvx_m1_ops_f32.c" > "$OUT/hvx_m1_ops_rope_mut.c"
if cmp -s "$OUT/hvx_m1_ops_rope_mut.c" "$BACKEND/hvx/hvx_m1_ops_f32.c"; then
  echo "ROPE MUTATION DID NOT APPLY: $mut"; exit 1
fi
"$cc" -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
  -I "$HERE/hvx_emu" -I "$BACKEND/.." -I "$BACKEND/hvx" \
  -o "$OUT/m1_ops_rope_mutant" \
  "$HERE/m1_ops_host_check.c" "$OUT/hvx_m1_ops_rope_mut.c" \
  "$BACKEND/hvx/hvx_conv_gate_f32.c" -lm
if "$OUT/m1_ops_rope_mutant" > "$OUT/m1_ops_rope_mutant.log"; then
  echo "ROPE MUTANT PASSED (the check is blind): $mut"; exit 1
fi
echo "ROPE MUTANT CAUGHT: hd64 pairing per chunk ($(grep -c '^FAIL: rope' "$OUT/m1_ops_rope_mutant.log") failed checks, rope64 lines $(grep -c 'rope64 pos=.* bad=0' "$OUT/m1_ops_rope_mutant.log")/5 still bad=0)"

# [plan 201 S4] The MoE epilogue's GeGLU-tanh: geglu_det_one (swiglu_det.h)
# against f64 incl. the tanh saturation ends and subnormals, then the REAL
# hvx_dequant_i32.c epilogue on hvx_emu/ (geglu and swiglu) bit for bit
# against the spec (GEGLU OK). Then gelu swapped for silu in the kernel,
# which must fail it.
geglu_check() { # geglu_check <hvx_dequant_i32.c> <exe>
  "$cc" -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
    -Wno-format-truncation -pthread -include malloc.h \
    -I "$HERE/hvx_emu" -I "$HERE/stub" -I "$BACKEND/.." -I "$BACKEND" \
    -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
    -o "$2" "$HERE/geglu_host_check.c" "$1" \
    "$BACKEND/hvx/hvx_worker_pool.c" -lm
}
geglu_check "$BACKEND/hvx/hvx_dequant_i32.c" "$OUT/geglu_host_check"
"$OUT/geglu_host_check"
mut='s/c->geglu ? hvx_geglu_det_sf(g, u)/c->geglu ? hvx_swiglu_det_sf(g, u)/'
sed "$mut" "$BACKEND/hvx/hvx_dequant_i32.c" > "$OUT/hvx_dequant_i32.c"
if cmp -s "$OUT/hvx_dequant_i32.c" "$BACKEND/hvx/hvx_dequant_i32.c"; then
  echo "GEGLU MUTATION DID NOT APPLY: $mut"; exit 1
fi
geglu_check "$OUT/hvx_dequant_i32.c" "$OUT/geglu_mutant"
if "$OUT/geglu_mutant" > "$OUT/geglu_mutant.log"; then
  echo "GEGLU MUTANT PASSED (the check is blind): $mut"; exit 1
fi
echo "GEGLU MUTANT CAUGHT: gelu replaced by silu ($(grep -o 'geglu bit-exact [0-9/]*' "$OUT/geglu_mutant.log"))"

# The CPU-order Q4_0 FC (#132 PR 2): the spec q4_gemv_cpu_det.h against an
# independent model of the Android CPU's quantizer and fused chain, six
# mutants (Q4 GEMV CPU-ORDER OK), and the REAL DSP source hvx_q4_gemv_f32.c
# on hvx_emu/ in all its variants against the spec (Q4 GEMV BIT-IDENTICAL).
# gnu11 for _Float16, the host's own f16 conversions.
"$cc" -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
  -I "$HERE/hvx_emu" -I "$BACKEND/.." -I "$BACKEND/hvx" \
  -o "$OUT/q4_gemv_host_check" \
  "$HERE/q4_gemv_host_check.c" "$BACKEND/hvx/hvx_q4_gemv_f32.c" -lm

"$OUT/q4_gemv_host_check"

# [#178] The mailbox hop (nntr_hvx_mailbox.c, included as-is): both roles on
# two pthreads, 10 000 exchanges at 0 and 8 KiB, no timeout, no stale word,
# checksums equal; a lone role times out instead of hanging (MAILBOX OK).
# The DSP cache maintenance and the hop's cost are the device's (probe Q2).
"$cc" -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -pthread \
  -o "$OUT/mailbox_host_check" "$HERE/mailbox_host_check.c"

"$OUT/mailbox_host_check"

# Decode attention at m=1 (#81, fp16 CPU order since #152): the spec
# nntrainer/tensor/attn_m1_det.h against an independent _Float16 model of
# the Android CPU attention with five mutants (ATTN M1 F16 CPU-ORDER OK);
# hvx_convert.h's fp16 primitives against the spec (ATTN M1 PRIM) and,
# since #170, hvx_attn_m1_hf.h's fp16-lane ones (ATTN M1 HF PRIM OK); then
# the REAL HVX source hvx_attn_m1_f32.c on hvx_emu/ with the REAL worker
# pool on pthreads (stub/qurt.h) at 0, 3 and 7 workers, memcmp'd against
# the spec for L = 1, 63, 64, 65, 512, 513, 1024, 1536 (max_seq 2048) at
# (n_kv, gqa) = (8, 4),
# (1, 2), (2, 3), (1, 8), head_dim 64, plus append-chain == bulk, the L = 1 case,
# a division tie and the error codes; and the phase words (#146), which
# must leave the output bytes alone (ATTN M1 PHASES OK).
# -include malloc.h: the cache is memalign(128), which the Hexagon libc
# declares in stdlib.h and glibc in malloc.h.
"$cc" -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
  -pthread -include malloc.h \
  -I "$HERE/hvx_emu" -I "$HERE/stub" -I "$BACKEND/.." -I "$BACKEND/hvx" \
  -o "$OUT/attn_m1_host_check" \
  "$HERE/attn_m1_host_check.c" "$BACKEND/hvx/hvx_attn_m1_f32.c" \
  "$BACKEND/hvx/hvx_worker_pool.c" -lm

"$OUT/attn_m1_host_check"

# weight_swap_u8i4_arena (doc 52 section 10.12): the real skel entry point
# over the real weight registry, the arena plain aligned memory. The header
# qaic would generate is written from the IDL, so the entry points'
# definitions are checked against the IDL too -- a mismatch fails here.
python3 "$HERE/gen_nntr_hvx_h.py" "$HERE/../nntr_hvx.idl" "$OUT/nntr_hvx.h"
"$cc" -std=c99 -O1 -Wall -Wextra -Wno-unused-parameter \
  -I "$OUT" -I "$HERE/stub" -I "$HERE/standin" -I "$HERE/.." -I "$BACKEND/.." \
  -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -o "$OUT/swap_host_check" \
  "$HERE/swap_host_check.c" "$HERE/../nntr_hvx_mm_u8i4.c" \
  "$BACKEND/hmx/hexkl_mm_u8i4_dma.c" "$BACKEND/hmx/hexkl_dma_trace.c" \
  "$HERE/hvx_scalar_stubs.c" "$HERE/standin/hvx_scalar.c" -lm

"$OUT/swap_host_check"
