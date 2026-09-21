# 25 — Cross-op weight prefetch for decode (ledger ①) and the `MM_TB` / `MM16_R` / chunk device sweep (⑨ ⑩)

Issue: dlwlzzero/nntrainer#25 (`prio:p1`). Contract: `docs/plans/0000-agent-system-and-env.md`.
Branch: `hvx/25-decode-prefetch` from `hvx_impl` (`f9d1a062`). Planned 2026-09-18 after #24 closed
ledger ⑫ (the logits return is < 0.1 ms inside a 34 ms step; the whole stage-1 W8A8 decode gap is
the weight stream plus #41's idle-gap clock).

**Status (2026-09-18): H1 measured, H2 folded into #57.** H1 (`docs/measurements/25-decode-prefetch.md`,
unit `R3CY10WM83Y`, v79): G1 PASS 0.884 → prefetch on by default; D FAIL (+9.9 %) → max-fit chunk stays;
G3 PASS; G1' ceiling 54 tok/s @512 (32.3 GB/s, `mm8`+`mm16`+`lg`). H2 is not run: `MM_TB` / `MM16_TB` are
inert at m=1 and decode `mm8` already sits under the pure DMA wait, so the sweep is a prefill question
and lives in #57's device session together with the `MM16_R` epilogue generalisation (ledger ⑩). Step 7's
"then build H2" and step 8's "after H2" are superseded; the PR follows H1. Next decode lever: ATTN (#58);
`down` on the DMA ring (#59, ledger ⑬).

Two deliverables, two handoffs: **H1** decides the prefetch and produces the W8 weight-stream
ceiling that #51 waits on; **H2** is the kernel-constant sweep on top of whatever H1 adopts. No ABI
change, no host change, no image change.

## 1. Goal and gate

Every device number in this plan is **DSP Mcycles per decode step** (median `pcycles` of the 63
`n=1` steps of `--chunk 128 --steps 64`, ÷ 1e6), on a **named unit**, read from the **second pass**
of a reversed-order double run (HEXAGON.md §7 rule 9 (a)–(d)). tok/s and host ms are recorded but
only compared within one unit and one session.

| # | Acceptance (issue) | Pass means |
|---|---|---|
| G1 | decode @512 improves by about the non-matmul share of the P1 profile | H1 variant **B** (prefetch) vs **A** (same skel, prefetch compiled out) on the same unit, pass 2: `Mcyc(B) ≤ 0.90 × Mcyc(A)` at 512 (the sim P3 decode512 profile puts non-matmul ops at ≈ 26 % of op cycles and the barrier at 0.93 ms; §3 explains why the exposed first-chunk DMA is the same order). 0.90–0.97 = partial, adopt but record; > 0.97 = no lever, keep the graph-lifetime queue and go to D / H2 for the decision |
| G1' | (supervisor, 2026-09-18) the W8 weight-stream ceiling | H1 variant **C** (`HTP_MM_STREAM_ONLY`): `GB/s = 595,984,384 B ÷ (Mcyc(C) ÷ pcycles_per_us)` and `ceiling tok/s = 1000 ÷ step_ms(C)`, both at 512 / 1024 / 4096. The 512 value replaces the provisional `≥ 60` in HEXAGON_BENCHMARK.md Goals (stage 1, W8A8) and is the input #51 waits on. The skel prints the exact byte count (`nntr_htp: stream bytes/step=...`) so the number is not hand-derived |
| G2 | the sweep fixes defaults with a device number behind each value | H2: for each of `MM_TB` (prefill, m=128) and `MM16_R` (decode m=1 and prefill), the value with the lowest Mcyc on pass 2 becomes the `#define` default; a difference < 2 % keeps the current value (4 / 4 / 2). `MM16_TB` (prefill only) is swept in an optional H3 only if `MM16_R` moved prefill by > 3 % |
| G3 | outputs bit-identical to the current path (`--eval` PPL, generated ids) | A, B, D and every H2 skel: `E2E ppl` and `top1` on `t512_23.i32` equal to the digit (`33.0884 / 189` on the v75 skel, the v79 figure of #23 B on that arch), and the `top1=` sequence of the 63 generated steps byte-identical to the A log of the same run (`diff <(grep -o 'top1=[0-9]*' a.log) <(... b.log)`). C is measurement-only: its outputs are garbage by construction and its accuracy cells read `n/a (stream only)` |
| G4 | 13 sim tests pass; `matmul_dma` covers the graph-lifetime queue | 13 × `SIM_TEST <name> PASS` on v75 (rung 3, once, before the PR); `test_matmul_dma` gains the four cases of §4 step 3 and keeps the `memcmp` against the DDR path (that check is arch-independent, so it also holds on the v79 simulator where `matmul_dma` fails the *reference* bound by ±1 LSB, a pre-existing #35 item); `profile acc` STAT bit-identical to `max_abs=0.077216 max_rel=98.2637` (HEXAGON.md §8.3) — the DMA moves the same bytes into the same slots, so nothing may move |

Environment check inside H1: A must land within ±5 % of the unit's last recorded Mcyc at 512
(`R3CY10WM83Y`: 65.1 v75 / — v79; `R3CY205ZMND`: 64.3 v75 / 60.1 v79). Outside that band the run is
void (stale artifact or wrong unit), not a result.

## 2. Where it lives

Verified against `hvx_impl` @ `f9d1a062`.

| File:line | Today | Change |
|---|---|---|
| `nntrainer/tensor/hexagon/htp/ops/hvx-matmul.c:30-37` | `MM_DMA_QUEUE_CAP 4`, `#define MM_TB 4u` unguarded | `#ifndef MM_TB` guards on `MM_TB` (`:37`), `MM16_R` / `MM16_TB` (`:241-242`) so `HEX_EXTRA_CFLAGS=-DMM_TB=8u` builds the sweep variants without a source edit; new `HTP_MM_CHUNK_ROWS` (0 = max-fit, the current behaviour), `HTP_MM_STREAM_ONLY`, `HTP_MM_NO_PREFETCH` knobs next to `HTP_MM_NO_VTCM` (`:125`) |
| `hvx-matmul.c:121-181` (`mm_worker_vtcm`) | per-op `memalign` + `dma_queue_init` (`:141-146`), kick chunk 0 (`:151-154`), pipeline `kick c+1 / pop c / compute c` (`:156-176`), `dma_queue_free` + `free` (`:178-179`) | uses `c->dmaq[wid]` (graph lifetime); on entry checks the worker's prefetch record (`c->pf[wid]`: `desc`, `buf`, `rows`) — hit → chunk 0 is already in flight in `pf.buf`, skip the kick; miss with something pending → `dma_queue_flush` then kick as today; after popping the last chunk's DMA and before computing it, kicks chunk 0 of `c->next_mm` (if any, and if its `rows_per_buf ≥ 32`) into the buffer the last chunk does not occupy, and writes `c->pf[wid]`. The `dma_queue_push_ddr_to_vtcm` return value (`:153`, `:167`, unchecked today) is checked: `false` → flush and retry once, never silently skip |
| `hvx-matmul.c:98-115` (`mm_tiles`), `:67-93` (`mm_tile`), `:244-288` (`mm16_block`) | kernels | **untouched** except the `#ifndef` guards; `HTP_MM_STREAM_ONLY` skips the `mm_tiles` call at `:174` and, in `mm16_worker` (`:295-317`), replaces the row loop by a DMA read of the worker's `[n0, n1)` rows into its slab in slab-sized pieces (measurement only; §3) |
| `hvx-matmul.c:185-202` (`mm_worker`) | DDR fallback when `mm_worker_vtcm` returns false | fallback path also flushes a pending prefetch for this worker (otherwise the next matmul finds a stale descriptor) |
| `nntrainer/tensor/hexagon/htp/ops/htp_ops.h:21-42` (`struct htp_exec_ctx`) | `vtcm`, `vtcm_size`, profile counters | adds `dma_queue_t *dmaq` (`[n_workers]`, owned by the graph), `void *dmaq_mem`, `struct htp_mm_prefetch *pf` (`[n_workers]`), `const struct nntr_htp_op_desc *next_mm` (set per op by the forward loop) |
| `nntrainer/tensor/hexagon/htp/htp_graph.c:27-109` (`htp_graph_init_ex`) | VTCM acquire at `:91-107` | after the pool exists (`:61`) allocates `wp_size(pool)` queues with `dma_queue_sizeof(MM_DMA_QUEUE_CAP)` / `dma_queue_alignof()` in one `memalign` block and the `pf` array; builds `g->next_mm[i]` = index of the next `MATMUL_W8A8` / `MATMUL_LOGITS` op after `i` (or `UINT32_MAX`) once, from the validated op-list |
| `htp_graph.c:118-160` (`htp_graph_forward_upto`) | plain dispatch loop | before `htp_op_table[kind](...)` at `:151`: `g->ctx.next_mm = next_mm[i] < n_ops_limit ? &g->ops[next_mm[i]] : NULL` (a partial run never prefetches past its limit); under `HTP_PROF_FARF` one `FARF(ALWAYS, "nntr_htp: prof n=%u pc=%llu mm8=%llu mm16=%llu lg=%llu attn=%llu rest=%llu")` line per call from the per-kind deltas of this call (§3, device profile) |
| `htp_graph.c:206-218` (`htp_graph_destroy`) | releases VTCM at `:209-210` before the pool at `:215-216` | runs a `wp_run` flush job (every worker `dma_queue_flush(c->dmaq[wid])`) **before** `HAP_compute_res_release`, so no DMA is in flight into released VTCM; frees `dmaq_mem` / `pf` |
| `nntrainer/tensor/hexagon/htp/htp_graph.h:20-25` | `struct htp_graph` | adds `uint32_t *next_mm` |
| `nntrainer/tensor/hexagon/htp/dma/dma-queue.h:184-296` | push returns `false` when the ring is full; `dma_queue_pop` spins on `desc->done` with `dmpoll()` | untouched (imported MIT code); the plan relies on `done` being a memory flag, §5 risk 3 |
| `nntrainer/tensor/hexagon/htp/executor.c:68-75` (`nntr_htp_close`) | `FARF(ALWAYS, "nntr_htp: close")` | under `HTP_MM_STREAM_ONLY` also prints `stream bytes/step` (Σ `n*k` over the matmul ops, computed at init); no IDL change |
| `test/hexagon/sim/test_matmul_dma.c:51-57`, `:101-127` | hand-built ctx, one op, three VTCM sizes | ctx gets its queues through a new `htp_graph_dma_init(ctx, nw)` / `htp_graph_dma_destroy` pair exported from `htp_graph.c` (so the test and the graph allocate the same way); new cases in §4 step 3 |
| `test/hexagon/sim/test_graph.c:265-270`, `:285-300`, `:398-412` | decode forward, `forward_upto` partial runs, 2-worker pool | unchanged code, now exercising the real q/k/v → o → gate/up → LOGITS prefetch chain, the past-the-limit rule and the mismatch flush; STATs must stay `graph_prefill 0.0245416/9.3795`, `graph_decode 0.0202219/23.975` |
| `tools/hexagon/build_skel.sh:9,58`, `build_sim_test.sh:43` | `HEX_EXTRA_CFLAGS` appended | untouched; the variants are pure `-D` flags |
| `nntrainer/tensor/hexagon/htp/nntr_htp_common.h:18-19` | `NNTR_HTP_ABI_VERSION 4` | **unchanged**. No wire, image or layout change; `lower_qwen3`, `ref_graph_forward`, `find_divergence.py`, `nntr_hexpack`, HEXAGON.md §1.4 / §2.1 do not move |

Scope boundary: `MATMUL_W8A16` (`down_proj`, 84 MB of the 596 MB per step) keeps its direct DDR
read (ledger ⑬ is a separate A/B); the prefetch chain skips it (v → o across ROPE/ATTN, o → gate
across ADD/RMSNORM, up → next layer's q across SILU_MUL/down/ADD/RMSNORM, up → LOGITS in the last
layer). Only variant C streams `down` rows, and only to count them in the ceiling.

## 3. Design

**Chosen: free-buffer handover with a graph-lifetime per-worker queue.** Each worker keeps its
two half-slab buffers and its own `dma_queue` for the session. Inside an op the pipeline is the
current one (kick c+1, pop c, compute c). The change is at the tail: once the last chunk's DMA has
been popped, the *other* buffer is idle for the rest of the op, so the worker kicks chunk 0 of the
next tiled matmul (`c->next_mm`, same `(wid, nw)` split, same `rows_per_buf` formula for that op's
`k`) into it and records `{desc, buf, rows}`. The next matmul's `mm_worker_vtcm` finds the record,
starts with chunk 0 already in flight and continues alternating from that buffer. Every op boundary
in between (ROPE, ATTN, ADD, RMSNORM, SILU_MUL, `down`) runs on the same persistent QuRT worker
threads, so the descriptor keeps draining while they execute and `dma_queue_pop` on entry is a
no-wait pop if the window was long enough.

Why this is the decode lever: on the device the pool has 6 workers (v79 silicon; `wp_create` reads
`qurt_hvx_get_units`), so a slab is 699,008 B and `rows_per_buf` is 320 rows at k=1024, 160 at
k=2048. Per worker and per layer the decode W8A8 ops are then 1 chunk (k, v: 160–192 rows), 2
chunks (q: 341 rows; o: 192 rows at k=2048; gate/up: 512 rows) — i.e. **the first chunk's DMA,
half or all of the op's bytes, is fully exposed** today (this is ledger ⑨'s observation) and the
DDR sits idle during every non-matmul op. The handover keeps the DMA engine busy across the
q → k → v and gate → up pairs as well as across the non-matmul windows, which is why the gate is
set at 10 % rather than at the barrier-sized 3 %. Nothing numerical changes: the same bytes land in
the same VTCM bytes and `mm_tiles` is untouched, so G3 is bit-identity by construction.

Invariants the implementer must keep (each has a `matmul_dma` case in §4 step 3):

1. A prefetch is only kicked when the next op's `rows_per_buf ≥ 32` for this worker (the DDR
   fallback of `mm_worker_vtcm:135-136` must never find a pending descriptor).
2. `mm_worker_vtcm` compares `pf.desc` with its own `d` (pointer identity: the graph's `ops` array
   is stable for the session): mismatch → `dma_queue_flush` first. This covers `forward_upto`
   partial runs (`find_divergence.py`, `--dump-op`), the DDR fallback and any re-init.
3. `next_mm` is clipped to `n_ops_limit`; the last matmul of a full run prefetches nothing (a
   wrap-around prefetch of layer-0 `q` across the RPC boundary is possible — the weights are
   static — but is ≈ 0.3 % of the stream and is deferred, not planned).
4. `htp_graph_destroy` flushes on the workers before releasing VTCM.
5. The queue ring (`MM_DMA_QUEUE_CAP 4`, 3 usable slots) holds at most the in-op `c+1` kick plus
   one cross-op prefetch; every push return value is checked.

**Rejected: the ledger ① layout (three-way slab split, `buf[0]` reserved for the next op).** It
guarantees no buffer collision by construction, but it shrinks the in-op double buffer from ½ to ⅓
of the slab (`rows_per_buf` 320 → 213 at k=1024), turning today's 1-chunk decode ops into 2-chunk
ones and adding a DMA descriptor per chunk on prefill; and it leaves a third of VTCM idle inside
every op that has no successor to prefetch (LOGITS, 155 MB of the stream). The handover reaches the
same overlap with the current chunking and one extra pending descriptor per worker. The kernel
rules of HEXAGON.md §7 are not touched by either variant (no arithmetic changes).

**The device profile (`HTP_PROF_FARF`).** The gate G1 is stated against "the non-matmul share",
which exists only as a simulator number today. The per-kind `prof_cycles` already accumulate on the
device (`htp_graph.c:150-155`); one `FARF(ALWAYS)` line per forward call with this call's deltas
gives the device split (W8A8 / W8A16 / LOGITS / ATTN / rest) at ≈ µs cost, is captured by
`run_e2e_test.sh` into `logs/hexagon/device_farf_<stamp>.log`, and needs no IDL change. Every H1/H2
skel carries the flag so the overhead cancels in the ratios; the shipping default leaves it off.

**The stream-only ceiling (`HTP_MM_STREAM_ONLY`, variant C).** The whole graph runs, but the tiled
matmuls skip `mm_tiles` after popping each chunk (the DMA pattern, chunking and prefetch of B are
kept), and `mm16_worker` DMA-reads its `[n0, n1)` rows into its slab in slab-sized pieces instead
of multiplying them. The step time of C is therefore "this graph with free matmul compute": the
weight stream plus the non-matmul ops, which is the ceiling the stage-1 W8A8 goal must be stated
against (supervisor comment on #25). The FARF profile line additionally gives the matmul-only share
of C, from which a pure DMA GB/s follows; both are reported.

**The sweep (H2)** is a build matrix on top of the adopted H1 defaults, nothing more: `MM_TB` is
inert at m=1 (`mm_tiles:111-113` takes the `tb=1` tail for every token when `m < MM_TB`), so it is
a prefill-only knob and `MM_TB 1` is dropped from the four-skel budget (it is the P3 "no token
blocking" case, dominated by 2 unless 2 already beats 4 — then H3 adds it); `MM16_R` matters at both
m=1 and m=128; `MM16_TB` only at m=128 (H3, conditional).

## 4. Steps

Rungs are from `.claude/skills/hexagon-gates`. Simulator budget (contract §7): per-step
`run_sim_test.sh matmul_dma` only; `profile acc` and the 13 tests once, before the PR. `HEX_ARCH=v75`
for the simulator (the v79 simulator's four ±1 LSB failures are #35's, not this issue's); device
skels `HEX_ARCH=v79` (the primary since 2026-09-17; `build_skel.sh:16` already defaults to it) — the
H1/H2 pass rules are ratios within one handoff, so the arch choice does not enter them.

1. **Knob guards and the device profile line.** `#ifndef` guards on `MM_TB`, `MM16_R`, `MM16_TB`;
   `HTP_MM_CHUNK_ROWS` (caps `rows_per_buf`); `HTP_PROF_FARF` line in `htp_graph_forward_upto`.
   Gate: rung 0 + rung 4 for the default build, and `HEX_EXTRA_CFLAGS="-DMM_TB=8u -DMM16_R=8u
   -DHTP_MM_CHUNK_ROWS=64u -DHTP_PROF_FARF"` compiles; the default skel's md5 equals the one from
   an untouched tree (the guards change no byte).
2. **Graph-lifetime queues.** `htp_graph_dma_init/destroy`, `ctx.dmaq/pf`, `mm_worker_vtcm` uses
   `c->dmaq[wid]` (no prefetch yet), destroy-time flush. Gate: `run_sim_test.sh matmul_dma` PASS
   with the existing three sizes (the 64 KB case must still take the DDR fallback with the queue
   allocated but unused).
3. **Prefetch.** `next_mm` table, per-op `ctx.next_mm`, tail kick + entry hit/miss/flush, DDR
   fallback flush, `HTP_MM_NO_PREFETCH` compile-out. `test_matmul_dma` gains: (a) two consecutive
   W8A8 ops (k=1024, n=3072 then k=2048, n=1024) run with `next_mm` set, both bit-identical to the
   DDR path and `pf` state consumed (a counter in the ctx under `#ifdef NNTR_HTP_SIM_TEST` or a
   return code from a test-only accessor; the test asserts ≥ 1 prefetch hit per worker); (b)
   prefetch for op X, then op Y runs — mismatch flush, still bit-identical; (c) a prefetch pending
   when the next op falls back to DDR (256 KB VTCM, k=2048) — flush, bit-identical; (d) the 4 MB /
   256 KB / 64 KB sweep as today with prefetch on. Gate: `matmul_dma` PASS; then `run_sim_test.sh
   graph` PASS with unchanged STATs (the real chain, partial runs, 2-worker pool).
4. **Stream-only variant.** `HTP_MM_STREAM_ONLY` in `mm_worker_vtcm` and `mm16_worker`, the
   `stream bytes/step` FARF. Gate: rung 4 with the flag compiles; no simulator test (measurement-
   only path, outputs are garbage by definition; the sim cannot say anything about it).
5. **Simulator, once.** Rung 2 (`profile acc`, STAT bit-identical) and rung 3 (13 tests) on v75 on
   the final tree. Record the `SIM_PROF` per-kind pcycles as a relative signal only (expect
   `MATMUL_W8A8` per_call unchanged within noise — the simulator does not charge DDR).
6. **Build H1 (device measurement unavoidable — handoff 1, `docs/measurements/25-decode-prefetch.md`).**
   Four `HEX_ARCH=v79` skels, copied to `build_hexagon/skel/libnntr_htp_skel.<variant>.so`, md5 +
   commit in the artifact table; harness `hexagon_e2e_test` from the same tree (unchanged source,
   md5 recorded); images `qwen3_full` / `qwen3_full4k` and tokens `t512_23.i32`, `t1024.i32`,
   `t4096.i32` as in #23/#24 (md5s repeated).

   | variant | `HEX_EXTRA_CFLAGS` | answers |
   |---|---|---|
   | A control | `-DHTP_PROF_FARF -DHTP_MM_NO_PREFETCH` | environment check (±5 % of the unit's last Mcyc), the A/B baseline, the device non-matmul share |
   | B prefetch | `-DHTP_PROF_FARF` | G1 |
   | C ceiling | `-DHTP_PROF_FARF -DHTP_MM_STREAM_ONLY` | G1' (GB/s, ceiling tok/s at 512 / 1024 / 4096) |
   | D chunk | `-DHTP_PROF_FARF -DHTP_MM_CHUNK_ROWS=64u` | ledger ⑨ on top of B: adopt if ≥ 2 % below B at 512 decode and not above B on prefill |

   Run sheet (unit serial passed explicitly, `adb devices` output pasted; `run_device_test.sh`
   RPC_TEST PASS first): pass 1 = A, B, C, D; pass 2 = D, C, B, A; each = `--tokens t512_23.i32
   --chunk 128 --steps 64` then `--eval` (C skips `--eval`); then B alone at 1024 and 4096
   (`--steps 64` + `--eval`) and C alone at 1024 and 4096 (`--steps 64`) for the goal rows; finally
   `adb shell md5sum` of the skel that ran. Estimated device time **≈ 30 min** (8 × ≈ 1 min at
   512, 4096 image push ≈ 3 min if absent, four 1024/4096 runs ≈ 8 min, md5 checks). Result table
   columns: variant, skel md5 (from the log), unit, ctx, pass, prefill tok/s, decode median host ms,
   decode median DSP Mcyc, `pcycles_per_us`, decode tok/s, FARF `mm8/mm16/lg/attn/rest` Mcyc,
   `--eval` PPL / top-1, generated-ids identical to A (y/n); for C: stream bytes/step, GB/s, ceiling
   tok/s. Reference cells: #24 `R3CY10WM83Y` 65.1 / 85.0 / 225.9 (v75) and #23 `R3CY205ZMND` 64.3 /
   84.2 / 213.0 (v75), 60.1 / 76.5 / 177.6 (v79); PPL 33.0884 / 189 (v75) or the #23 B v79 value.
   Label `state:needs-measurement`.
7. **Read H1 back** (`state:measured` → implementer): fix the prefetch default (on if G1 passes,
   compiled in but off otherwise), the chunk default (D rule), write the HEXAGON.md §8.2 block and
   the benchmark goal row from C (§6). Then **build H2 (handoff 2, `docs/measurements/25-kernel-sweep.md`)**:
   four v79 skels on the adopted defaults, all `-DHTP_PROF_FARF`: `-DMM_TB=2u`, `-DMM_TB=8u`,
   `-DMM16_R=2u`, `-DMM16_R=8u` (the H1 B or D skel of the same tree is the `MM_TB 4 / MM16_R 4 /
   MM16_TB 2` reference and is re-run first and last as the control, rule 9 (d)). 512 tokens only,
   `--steps 64` + `--eval` per skel, two passes reversed; prefill tok/s (m=128) and decode Mcyc
   (m=1) come from the same run. Estimated **≈ 15 min**. Pass rule G2. Optional H3 (`MM16_TB 1`,
   `MM16_TB 4`, `MM_TB 1`, best combination) only if `MM16_R` moved prefill by > 3 % or `MM_TB 2`
   beat 4.
8. **PR** into `hvx_impl` after H2 (or after H1 if H2 changes nothing): defaults fixed, both
   handoff files filled and committed on the branch, `HTP_PROF_FARF` / `HTP_MM_STREAM_ONLY` /
   `HTP_MM_NO_PREFETCH` kept as measurement flags next to `HTP_MM_NO_VTCM`, docs of §6. Rung 0–4
   pass lines in the PR, skel md5 of the final default build.

## 5. Risks

1. **Bandwidth is invisible on the simulator** (gap rule 1). The prefetch's whole value is DDR
   overlap; the sim will show `MATMUL_W8A8` per_call unchanged and cannot rank A/B/D. The handoff
   makes it visible through the A-vs-B Mcyc ratio on one unit in one session, with A also serving
   as the environment check against the unit's recorded baseline.
2. **Unit and clock** (§7 rule 9). Two units at 1.91 / 2.05 GHz; the tables carry the serial and
   `pcycles_per_us` per row, the pass rules are Mcyc ratios, and tok/s is only compared to the same
   unit's earlier rows. The first run after a push is 3–5 % high: pass 1 is discarded by design.
3. **A descriptor pending across `wp_run` barriers** is the one new runtime pattern. The user-DMA
   engine is per hardware thread and `dma_queue_pop` spins on the descriptor's `done` flag in
   memory (`dma-queue.h:273-296`), so a worker that is rescheduled still sees completion; but if
   QuRT did not preserve DMA context across the barrier on silicon, symptoms would be a hang in the
   first prefetched op or garbage output. The simulator runs the same QuRT and `matmul_dma` (a)–(d)
   plus `graph` catch a functional break; on the device G3's byte-identical ids and the per-step
   `pcycles` line (a stall shows as a > 2× step) catch the rest. Fallback if it bites: kick the
   prefetch from a `wp_run` job at the *start* of the following non-matmul op instead of at the
   tail of the matmul (same threads, no cross-barrier descriptor is then older than one op).
4. **qf32 numerics**: none introduced (no arithmetic change); the `--eval` PPL column is still
   mandatory and must be equal to the digit, which is stronger than the usual band.
5. **Stale artifacts** (gap rule 3): four skels of identical size differ only by flags; each row
   repeats the md5 that `run_e2e_test.sh` pushed and the final `adb shell md5sum`, and A's ±5 %
   environment check voids a run that used the wrong skel. The C skel's garbage outputs must not
   be mistaken for a failure: its accuracy cells are pre-filled `n/a`.
6. **VTCM release with DMA in flight** (destroy-time) and **ring overflow** (unchecked push today)
   are correctness risks of the change itself, closed by invariants 4–5 and the `matmul_dma` cases.
7. **Sim time**: one `matmul_dma` run per step (seconds), `profile acc` + 13 tests once (≈ 15 min
   under Rosetta); no `SIM_TIMING`, no v79 simulator run (the v79 surface is untouched).

## 6. Docs to update

* `docs/backend_guide/HEXAGON.md`: §2.2 VTCM paragraph (`:285-296`: graph-lifetime queue,
  handover prefetch, the chunk knob, `MATMUL_W8A16` still direct); §5.2 `test_matmul_dma`
  description (`:556`: the four new cases); §5.3 measurement flags (`HTP_MM_NO_VTCM` family);
  §8.2 new block "Cross-op prefetch and the W8 weight-stream ceiling (#25)" with the H1/H2 tables
  (unit, skel md5, pass 2), the device per-kind split from the FARF line, and the GB/s of C;
  §8.3 one line (STAT bit-identical, per_call unchanged, as expected without a DDR model); §9:
  drop the ① ⑨ ⑩ sentence, update the "device re-measurement" bullet (`MM_TB`'s real effect is
  now measured), leave ⑬ and #41; §7: a new rule only if the device disagrees with the simulator
  (handoff skill, reading step 4).
* `docs/backend_guide/HEXAGON_BENCHMARK.md`: new `nntrainer` rows for the adopted skel at 512 /
  1024 / 4096 with the unit serial; Goals stage-1 W8A8 "Goal" cell replaced by C's measured
  ceiling (tok/s and GB/s, with the unit and date) and the "Now" cell moved to B; the w4a8 row's
  "after #25 measures the W8 ceiling" resolved with the number; Log row.
* `docs/superpowers/specs/hexagon-hvx-optimization/07-follow-ups.md`: ① ⑨ ⑩ get a completion
  note in the style of ⑪ / ⑭ (chosen design differs from ①'s sketch: handover, not a reserved
  buffer).
* Issue #51 gets a comment with the C row (its gate input); #41 is unaffected.
