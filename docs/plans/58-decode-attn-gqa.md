# 58 — Decode attention (m=1): position-split jobs over the 6 workers, K^T / V shared across the GQA pair

Issue: dlwlzzero/nntrainer#58 (`prio:p1`, filed by the supervisor 2026-09-18 from the #25 FARF split).
Contract: `docs/plans/0000-agent-system-and-env.md`. Branch: `hvx/58-decode-attn-gqa`, cut from
`hvx_impl` **after the #25 PR merges** (the `HTP_PROF_FARF` line in `htp_graph_forward_upto` and
`tools/hexagon/summ_farf_prof.py` are the instrument this issue's gate reads; nothing else from #25 is
needed — the plan uses `l2fetch`, not the graph-lifetime DMA queue, see §3). If the implementer starts
before the merge, branch from `origin/hvx/25-decode-prefetch @ 53416100` and rebase once; the file
overlap is two hunks (`htp_ops.h` struct, `htp_graph.c` scratch sizing). Line numbers below are
`hvx_impl @ 7575b9f5`; `htp_graph.c` shifts by ≈ +64 after #25.

## 1. Goal and gate

| # | Acceptance (issue) | Pass means (measurable) |
|---|---|---|
| G1 | FARF `attn` ≤ 0.5 × control at 4096 and ≤ 9 Mcyc at 512 | Handoff §4 step 6, same unit, same session, reversed double pass, **pass 2**, `summ_farf_prof.py` decode median of the `attn=` field: `attn(B) ≤ 0.5 × attn(A)` at 4096 (A ≈ 146.6 → B ≤ 73 Mcyc) **and** `attn(B) ≤ 9.0 Mcyc` at 512 (A ≈ 17.3); the decode `DSP Mcyc/step` column drops by the same amount ± 2 Mcyc (no cost moved into another kind) |
| G2 | PPL in the v79 band | B `--eval` on `t512_23.i32`: 41.4947 / 162 ± 0.3 % / ± 4 (the merge changes the fp32 summation order, so "to the digit" is not required); B on `t4096.i32`: 1.5623 / 3759 ± 0.3 % / ± 4; the same for whichever variant becomes the default (C) |
| G3 | a new m=1 long-context sim case (pos ≥ 1024) | `HEX_ARCH=v79 run_sim_test.sh attn` → `SIM_TEST attn PASS` with the three new decode cases of §2 (`attn_decode_p1024`, `attn_decode_p1087`, `attn_decode_p100`, each on the default pool and on a 5-worker pool) inside the existing 2e-2 / 5e-3 bound; the existing `attn_prefill` STAT `0.000488281/0.208165` **bit-identical** (m > 1 keeps one job per kv head and the same per-lane op order) |
| G4 | `profile acc` and the 13 tests | `SIM_TEST profile PASS`, `profile_prefill_acc STAT 0.0659682/92.7791` bit-identical (prefill path unchanged), `graph_prefill 0.0218946/6.88818` and `graph_decode 0.0197323/23.2374` bit-identical (the graph test's decode has `L = 9` → one 64-lane block → no split, §3), 13/13 |
| G5 | the 36 Mcyc @4096 figure is the follow-on gate | recorded, not gated: variant C's `attn` at 4096 and the implied KV GB/s go to HEXAGON.md §8.2 / §9 as the input of the next issue |

Environment check inside the handoff: A at 512, pass 2, within ± 5 % of **54.6 Mcyc** (#25 B on
`R3CY10WM83Y`; A *is* the post-#25 `hvx_impl`, prefetch on); outside that band the session is void.
No ABI change: `NNTR_HTP_ABI_VERSION` stays 4 (`nntr_htp_common.h:18-19`); the KV layout (§2.2 of
HEXAGON.md: K^T `[head_dim][max_seq]`, V `[max_seq][head_dim]`) is untouched; `attn_scratch` is
DSP-private, so `lower_qwen3`, `nntr_hexpack`, `.hexcfg`, `ref_graph_forward`, `find_divergence.py`
and the sim `graph` test do not move.

## 2. Where it lives

| File:line (hvx_impl @ 7575b9f5) | Today | Change |
|---|---|---|
| `nntrainer/tensor/hexagon/htp/ops/hvx-attn.c:53-54` | `h0/h1 = n_kv·wid/nw`: 8 heads over 6 workers → workers 0, 2, 5 take two heads | jobs `(h, b)` = (kv head, position block), `nb` blocks per head (§3), job `j` runs on worker `j % nw`; interleaved so every worker gets `n_kv·nb/nw` jobs |
| `hvx-attn.c:61-68` (KV append) | every worker appends its heads' rows first | at m=1 only the job whose block holds position `pos` appends (it is the only reader of that position); at m > 1 `nb = 1` and the single job appends as today |
| `hvx-attn.c:70-94` (score loop, per q head) | `for g in group: for t: for p0: for i: mpyacc(acc, K^T[i][p0], splat q[t,hq][i])` — K^T streamed once **per q head** | `for t: for p0: for i: k = vmem(K^T[i][p0]); for g: acc[g] = mpyacc(acc[g], k, splat q[t,h·group+g][i])` — one K^T load per `(t, p0, i)` serves the `group` (= 2) q heads; `group` accumulator pairs (4 vectors at group 2). Same op order per accumulator → bit-identical scores |
| `hvx-attn.c:96-111` (scalar softmax) | over `[0, L)` | over the job's range `[p_lo, p_hi)` with row-local indexing `scores[p − p_lo]`; per (t, g) records the block max `m_b` and sum `l_b` instead of dividing; the `s16` in-place narrowing keeps its invariant (§3) |
| `hvx-attn.c:113-125` (PV) | per position, per q head: 2 V loads + 2 mpyacc | per 64-position lane block, per position: 2 V loads, then for `g`: splat `s16[g][p]`, 2 mpyacc — one V load per position for both q heads (8 accumulator vectors); output kept in fp32 as the block partial `o_b` when `nb > 1`, else scaled and narrowed exactly as today (`:127-140`) |
| `hvx-attn.c:146-152` (`hvx_op_attn`, `Follow-up:` note) | one `wp_run` | `wp_run(attn_sdpa_worker)` then, only when `nb > 1`, `wp_run(attn_merge_worker)` (16 q heads over the workers; §3); note rewritten: the row-block reuse for prefill is #26 |
| `hvx-attn.c` (new, top of file) | — | `#ifndef HTP_ATTN_NB` `#define HTP_ATTN_NB 0u` (0 = auto `nw / gcd(n_kv, nw)`, `1` = no split: handoff variant D); `#ifndef HTP_ATTN_L2FETCH` `#define HTP_ATTN_L2FETCH 0` (variant C, `hex_l2fetch` from `hex/hex-utils.h:41-46`, one lane block ahead for K^T and V) |
| `nntrainer/tensor/hexagon/htp/ops/htp_ops.h:31` | `attn_scratch` `[n_workers][max_seq]` fp32 | layout `[n_workers][group][max_seq]` fp32 scores **+** partials `o` `[n_heads][nb_max][head_dim]` fp32 (128-B aligned; `nb_max = n_workers`) **+** `ml` `[n_heads][nb_max][2]` fp32 **+** 64 floats for the merge weights; new `static inline size_t htp_attn_scratch_bytes(int nw, const struct nntr_htp_oplist_header *cfg)` so the graph and the test size it the same way |
| `nntrainer/tensor/hexagon/htp/htp_graph.c:81-82` (≈ `:145` after #25) | `wp_size · max_seq · 4 + 128` | `htp_attn_scratch_bytes(wp_size(pool), &g->cfg)` (qwen3, 6 workers, 4224: 203 KB + 48 KB + 0.8 KB) |
| `test/hexagon/sim/test_attn.c:20-25, 109-158` | `N_HEADS 4 / N_KV 2 / MAX_SEQ 64`; m=8 @0, m=1 @8; scratch `wp_size · MAX_SEQ` (`:140-141`) | keep both cases and the STAT; add a second shape block (`n_heads 16 / n_kv 8 / head_dim 128 / max_seq 1088 / n_layers 1`, the run_step helper takes the shape as a struct) with the KV cache pre-filled with the same random fp16 in both layouts (transposed for the kernel, row-major for `ref_attn`); cases m=1 @ pos 1024 (`L = 1025`: 17 lane blocks, `nb = 3` on 6 workers → 6/6/5, tail lane block of 1), m=1 @ 1087 (`L = 1088`, no tail), m=1 @ 100 (`L = 101`, `nb = 2`, one full block + a 37-lane block); each repeated on a `wp_create(5)` pool (`nb = 5`, 40 jobs, remainder rotation); `HAP_perf_get_pcycles` around `hvx_op_attn` of the p1024 case printed as `attn_decode_p1024 STAT pcycles=<n>` (relative signal); scratch via `htp_attn_scratch_bytes` |
| `test/hexagon/sim/test_graph.c:30-34, 263-270`, `test_profile.c` | `MAX_SEQ 64`, decode at pos 8 | unchanged code; both decode paths have `L < 64` → `nb = 1` → STATs bit-identical (G4) |
| `nntrainer/tensor/hexagon/htp/hvx/hvx-base.h:226-245`, `hvx-exp.h:168` | `hvx_vec_mpyacc_f32_f16`, `hvx_exp_f32` | unchanged; the merge reuses `hvx_exp_f32` (weights) and `Vqf32_vmpy_VsfVsf` / `Vqf32_vadd_VsfVsf` / `Vsf_equals_Vqf32` (no chained qf32 adds, §7 rule 1) |
| `tools/hexagon/build_skel.sh:43-45`, `build_sim_test.sh` | glob `ops/*.c`, `HEX_EXTRA_CFLAGS` | untouched; the variants are `-D` flags |

Scope boundary: prefill (m > 1) keeps one job per kv head and only gains the GQA-pair fusion (a
free, bit-identical side effect); the row-block reuse across query rows, the vector softmax and the
⑮ bisection stay in #26.

## 3. Design

**What the #25 numbers say the kernel is.** Per kv head and layer at L=4096 the kernel issues
≈ 32 K `mpyacc` and ≈ 32 K 128-B vector loads (K^T twice, V twice, once per q head) and takes
2.6 Mcyc on the critical worker (146.6 / 28 / 2 heads) — **≈ 80 cycles per 128-B load**, i.e. one
worker pulls ≈ 3 GB/s. The DDR traffic is 2 × 470 MB = 940 MB per step at 4096 (the K^T of one head is
1 MB, the second pass does not hit L2), ≈ 13 GB/s aggregate with four of six workers busy. So the loop
is **latency-bound per worker** (strided K^T lines, no hardware prefetch, one accumulator chain), not
DDR-bound (the weight DMA reaches 37 GB/s on the same unit). Three things follow: halving the loads
(fusion) halves the time as long as latency dominates; balancing the six workers cuts the critical
path 2 → 1.33 head-equivalents; and issuing the lines ahead (`l2fetch`) is what converts latency into
bandwidth — the lever that decides whether 36 Mcyc is reachable (G5).

**Chosen: GQA-fused, position-split SDPA with a flash-decoding merge (ledger 05 ⑤), `l2fetch` as a
measured knob.**

*Jobs.* `L = pos + m`, `nlb = ceil(L / 64)` lane blocks. At m = 1: `nb = min(nw / gcd(n_kv, nw), nlb)`
(6 workers, 8 heads → 3; 4 workers → 1; 2 workers → 1; 5 → 5), overridable by `HTP_ATTN_NB`. At m > 1:
`nb = 1`. Jobs `j ∈ [0, n_kv · nb)`, `h = j / nb`, `b = j % nb`; block `b` of head `h` owns
`nlb / nb` lane blocks, the `nlb % nb` remainder lane blocks going to the blocks with
`(b + h) % nb < nlb % nb` (rotated by head so worker `w`, which always draws the same `b` when
`nw % nb == 0`, does not always draw the long block). Worker `w` runs jobs `w, w + nw, …` — 24 jobs,
4 per worker on qwen3 / v79; at 4096 `nlb = 65` → blocks of 21–22 lane blocks, ≤ 5 % imbalance;
at 512 `nlb = 9` → 3/3/3.

*Per job.* (1) If m = 1 and `pos` lies in this block: the KV append for head `h` (as `:61-68`).
(2) Scores for both q heads of `h` from one K^T stream: per lane block `p0`, per `i < 128` one
`hvx_vmem(kh + i·max_seq + p0)`, two `Q6_Vh_vsplat_R(q[g][i])`, two `hvx_vec_mpyacc_f32_f16`; vshuff
and store to the per-worker rows `scores[g][p − p_lo]` — one `[max_seq]` fp32 row per q head of
the pair, `[n_workers][group][max_seq]`, sized by `htp_attn_scratch_bytes` (both heads' scores are
produced by the same K^T pass, so both rows are live at once). (3) Scalar
softmax per q head over `[p_lo, p_hi)` exactly as `:96-111` (scale, max skipping NaN as the
reference, subtract, `hvx_exp_f32`, sum, in-place fp16 narrowing — the invariant at `:46-48` holds
for any prefix of lane blocks processed in order), recording `m_b`, `l_b` and **not** dividing when
`nb > 1`. (4) PV per lane block, per position: two V loads, per q head one splat of `s16[g][p]` and
two `mpyacc` — 8 accumulator vectors. (5) `nb == 1`: scale by `1 / l` and narrow as `:127-140`
(today's numerics, so every existing STAT is bit-identical). `nb > 1`: store `o_b` (fp32, 2 vector
pairs → 4 vectors, 512 B) and `(m_b, l_b)` into the partials area.

*Merge pass* (`wp_run`, one q head per job, `n_heads` jobs over `nw` workers; HVX only on workers,
§7 rule 7): `M = max_b m_b` (scalar, NaN-skipping like the reference), `w_b = exp(m_b − M)` through
`hvx_exp_f32` on a 64-float padded scratch row (the same exp the block softmax used; no libm
dependency), `l = Σ w_b · l_b` (scalar fp32), `o = Σ_b w_b · o_b` as
`Vsf_equals_Vqf32(Vqf32_vadd_VsfVsf(o, Vsf_equals_Vqf32(Vqf32_vmpy_VsfVsf(o_b, splat w_b))))` — one
qf32 op per add, never chained (rule 1), then `× 1/l` and `hvx_vec_f32_to_f16_shuff` as today. Cost:
one extra barrier per ATTN op (≈ 4.3 Kcyc × 28 ≈ 0.12 Mcyc/step) plus 16 × 3 vector FMAs — noise.
Numerics: the block-local softmax narrows probabilities relative to the block max instead of the row
max and the fp32 sums are re-associated; that is the "merge changes the order" clause of the issue
(± 0.3 % PPL). The `nb == 1` path is untouched so the sim records only move where the split is on.

*`HTP_ATTN_L2FETCH` (variant C).* Before the score pass of lane block `p0`, `hex_l2fetch(kh + p0 + 64,
128 B, max_seq · 2 B, 128 rows)` for the next block's K^T (one 2-D descriptor: exactly the 128 lines
the loop will touch) and `hex_l2fetch_block(vh + (p0 + 64) · hd, 16 KB)` for the next V block; 32 KB
in flight per worker, 192 KB across six, well inside L2 (the weight DMA lands in VTCM and does not
compete for L2). Values are unchanged (a prefetch), so the sim `attn` STATs are the proof it does not
corrupt anything and the device decides whether it is on by default (§4 step 7).

**Expected on the device.** Fusion halves the loads and the DDR bytes (470 MB per step at 4096);
balance lowers the critical path from 2 to 1.33 head-equivalents; together ≈ 3× on the attention
time if it stays latency-bound: 146.6 → ≈ 50 Mcyc at 4096 (gate 73), 17.3 → ≈ 6 + fixed (≈ 0.5
Mcyc of barriers and appends) at 512 (gate 9). With `l2fetch` the bound becomes bandwidth:
470 MB at 24–37 GB/s = 24–37 Mcyc at 4096, which is the G5 figure the handoff records. B vs C in
the same session separates the two regimes (C ≪ B ⇒ latency was the wall and the prefetch is the
follow-on lever; C ≈ B ⇒ the fused loop is already bandwidth-bound and the next lever is fewer bytes,
i.e. the int8/fp8 KV the issue rules out of scope).

**Rejected: split by q head (16 jobs, 3/3/3/3/2/2), no fusion** (the issue's lever 1, second form;
#26's variant C). It balances the workers (critical path 4 → 3 q-head passes, 1.33×) but keeps every
K^T / V vector loaded twice, so it cannot reach 0.5 × at 4096 in a latency-bound loop; and it is
incompatible with fusion, which needs the pair on one worker. Variant D (`HTP_ATTN_NB=1`: fusion
without the split) measures the balance share instead, at 512 only.

**Rejected: K/V through VTCM by DMA** (the issue's lever 3 as written). The free half-slab during
ATTN belongs to the #25 prefetch record (`c->pf[wid]`: which half is free depends on the previous
matmul's last chunk), so a KV DMA would have to negotiate slab ownership with `hvx-matmul.c`, and it
needs a per-block descriptor per worker per lane block. `l2fetch` needs no buffer, no ownership and
no ABI, and the same handoff row tells whether it already closes the latency gap; DMA stays the
follow-on if C lands above 36 Mcyc and its GB/s reading is well below the 37 the weight ring reaches.

**Overlap and ordering with #26** (`docs/plans/26-prefill-attn.md`, `origin/hvx/26-prefill-attn`,
`state:planned`). Both plans rewrite `attn_worker`. #58 lands first (it is the decode gate in front
of #51; the 4096 goal is the furthest). #26 then rebases onto the #58 kernel and keeps its own scope:
its `(h, row block)` jobs become the m > 1 arm of the same SDPA worker (`nb = 1` there), its vector
softmax replaces the scalar block softmax in both arms (and removes the ~3.7 Mcyc/step of scalar
passes that survive #58 at 4096), and its `HTP_ATTN_RQ` scratch supersedes the `[n_workers][2][max_seq]`
rows (`[n_workers][group][max_seq]` in #58 is its RQ = group special case). The GQA fusion is the same code in both plans, so #58 delivers the "what it buys at decode"
paragraph of #26 §3 early and #26's variant C (no-GQA control) is answered by #58's D at m=1; #26 §9
listed ledger ⑤ (decode column split) as skipped — #58 closes ⑤. The ⑮ bisection is untouched.

## 4. Steps

All commands through `tools/docker/run.sh` from `/Users/dlwlzzero/Projects/nntrainer-58` (or the
implementer's worktree), `HEX_ARCH=v79`. Simulator budget (gates skill): `attn` per step (~1 min),
`graph` once when `htp_graph.c` changes, `profile acc` + the 13 tests once before the PR.

**Step 0 — branch.** `hvx/58-decode-attn-gqa` from `hvx_impl` after the #25 PR merges (or from
`origin/hvx/25-decode-prefetch @ 53416100`, rebased once). Verify `git log --oneline -1 --
tools/hexagon/summ_farf_prof.py` is non-empty.

**Step 1 — tests first (the kernel untouched).** `htp_attn_scratch_bytes` in `htp_ops.h` and its two
call sites; the second shape block, the KV pre-fill helper, the three long-context cases on the
default and the 5-worker pool, the `pcycles` STAT. Gate: `run_sim_test.sh attn` → `SIM_TEST attn
PASS` on the old kernel (the cases prove themselves), `attn_prefill 0.000488281/0.208165` unchanged;
record `attn_decode_p1024 STAT pcycles=<B0>` in the PR body.

**Step 2 — GQA fusion only.** Restructure `attn_worker` into the job form of §3 with `nb` forced to 1
(no merge yet): one K^T stream and one V stream per (t, kv head) serve both q heads. Gate: `attn`
PASS with **every** STAT bit-identical to step 1 (same per-lane op order); `pcycles` recorded (expect
a modest drop: the sim charges no DDR latency, only the halved load count).

**Step 3 — position split and merge.** `nb` per §3, partials, `attn_merge_worker`, second `wp_run`,
`HTP_ATTN_NB`. Gate: `attn` PASS, the six long-context cases inside 2e-2 / 5e-3, `attn_prefill` and
the small `attn_decode` unchanged, `attn_decode_p1024 pcycles ≤ 0.6 × B0` (the simulator does model
six threads, so the balance must show; below that floor the implementer splits the timer per stage,
pastes the split on the issue and stops). Then `run_sim_test.sh graph` → PASS, `graph_prefill` /
`graph_decode` / `graph_prefill_2workers` bit-identical (`htp_graph.c` changed).

**Step 4 — `HTP_ATTN_L2FETCH`.** The two prefetches of §3 behind the knob. Gate:
`HEX_EXTRA_CFLAGS=-DHTP_ATTN_L2FETCH=1 build_sim_test.sh` + `attn` → PASS with STATs identical to
step 3 (values cannot move); same for `-DHTP_ATTN_NB=1` (variant D). Rebuild plain afterwards.
`clang-format-14` on the changed files; review against the §7 list (qf-format ops only, 128-B
alignment of the partials, `wp_size` from the pool, no HVX outside `wp_run`).

**Step 5 — pre-PR gates and skels.** Plain v79 build: `profile acc` → PASS, STAT bit-identical
(G4); the 13 tests → 13/13. Skels (all `-DHTP_PROF_FARF` so the FARF cost cancels): **A** from a
clean checkout of the `hvx_impl` HEAD the branch is based on, **B** = this branch (`HTP_ATTN_NB` auto,
`HTP_ATTN_L2FETCH 0`), **C** = `-DHTP_ATTN_L2FETCH=1`, **D** = `-DHTP_ATTN_NB=1`. `build_host_test.sh`.
md5 of every artifact; images and token files as #25 (`qwen3_full`, `qwen3_full4k`, `t512_23.i32`,
`t1024.i32`, `t4096.i32`, all already on the device).

**Step 6 — handoff (the device measurement is unavoidable: G1, G2, G5).**
`docs/measurements/58-decode-attn-gqa.md` (`hexagon-handoff` template), one session **≤ 20 min**,
unit named, `HTP_PROF_FARF` on every skel, `summ_farf_prof.py` over the logs:

| run | variants | what it reads | est. |
|---|---|---|---|
| 512 speed, pass 1 then pass 2 reversed | A B C D / D C B A | G1 at 512 (`attn` column, pass 2), D vs B = balance share, C vs B = latency share, env check on A | 8 × ~0.5 min |
| 4096 speed, pass 1 then pass 2 reversed | A B / B A, then C once | G1 at 4096, G5 (C's `attn` Mcyc → KV GB/s = 470 MB ÷ time) | 5 × ~2.5 min |
| 1024 speed | B once | goal row | 1 min |
| `--eval` | B at 512 (`t512_23`) and 4096; C at 512 | G2 | 3 × ~1 min |

Result table columns: variant, pass, skel md5 from the log, ctx, prefill tok/s, decode host ms,
decode tok/s, **DSP Mcyc/step**, `pcycles_per_us`, **FARF `mm8 / mm16 / lg / attn / rest`**, log
stamp; accuracy table with PPL / top-1 and the reference cells (41.4947 / 162, 5.8595 / 692,
1.5623 / 3759). Reference cells for the unit: #25 B (54.575 / 73.744 / 184.603 Mcyc, `attn` 17.26 /
34.2 / 146.6). Pass rules written into the doc: env (A 512 pass 2 within ± 5 % of 54.6), G1 (B), G2
(B, and C if C becomes the default), "C default" rule: `attn(C) ≤ 0.9 × attn(B)` at 4096 and not worse
at 512 → `HTP_ATTN_L2FETCH 1` ships; D is informational. Set `state:needs-measurement`, stop.

**Step 7 — after `state:measured`.** Set the `HTP_ATTN_L2FETCH` default per the rule; docs per §6;
commits `[htp]` tests, `[htp]` kernel, `[htp]` knobs, `[docs]`; `git commit -s`; PR into `hvx_impl`;
`state:review`. If G1 fails at 4096 but G3/G4 hold, the kernel still merges if B is not slower
anywhere (it is a correctness-neutral rewrite) and the issue returns to `state:needs-plan` with the
A/B/C/D `attn` line quoted — the C row then says whether the next plan is DMA-to-VTCM (C ≪ B but
> 36) or fewer KV bytes (C ≈ B).

## 5. Risks

* **Everything this issue buys is invisible on the simulator** (gap rule 1: no DDR latency or
  bandwidth model). The sim gates prove correctness and the thread-level balance (`pcycles ≤ 0.6 ×
  B0`), never the 0.5 ×. The handoff's A/B/C/D in one session, pass 2, `attn` column, is the only
  verdict; C vs B is what makes "latency- or bandwidth-bound" visible instead of argued.
* **Merge numerics on silicon** (gap rule 2). New on the device: block-relative fp16 narrowing of
  probabilities and the qf32 weighted sum of ≤ 6 partials — the same instruction classes the P3/P4
  epilogues and `hvx_vec_mpyacc_f32_f16`'s qf path already proved (§7 rules 1, 5, 8); `hvx_exp_f32`
  on `m_b − M ≤ 0` is the argument range the block softmax already feeds it. The `--eval` columns at
  512 and 4096 (where `nb = 3` is on at every layer) are the three-way record; a deviation beyond
  ± 0.3 % is a kernel bug, not a rounding note, and follows the ⑮ procedure with `--dump-op` on the
  1-layer image.
* **Tail lanes and empty blocks.** Lanes `≥ L` inside the last lane block were harmless garbage
  before and still are (the scalar passes stop at `L`); a block never has zero valid positions
  (`nb ≤ nlb`). The p1024 (tail of 1), p100 (tail of 37) and 5-worker (remainder rotation) cases cover
  the index arithmetic at model shape (`n_kv 8`, `head_dim 128`).
* **Scratch growth** ≈ 250 KB at 6 workers / 4224 (DDR `memalign`); the partials are 128-B aligned by
  construction (`head_dim · 4 = 512 B` per `o_b`).
* **`l2fetch` side effects.** A wrong descriptor prefetches the wrong lines (slower, never wrong
  values); L2 pressure from six workers × 32 KB is small, and the weight stream bypasses L2 (VTCM).
  The sim runs with the flag on prove it changes no value.
* **The #26 collision.** Same function; #26 is planned but not started. The ordering in §3 is
  explicit; if #26 starts first anyway, its rebase cost is the m > 1 arm only.
* **4096 cross-session offset (#53)** — every 4096 verdict is B vs A in the same session and pass
  (rule 9(e)); `pcycles_per_us` wander is why the rules read Mcyc, not ms.
* **Stale artifacts** (gap rule 3): four skels of two size classes; every result row repeats the md5
  the harness printed; the FARF line proves the profiling build ran.

## 6. Docs to update

* HEXAGON.md §1 (`:30-84`, the ATTN paragraph): jobs `(kv head, position block)` at m=1, GQA-fused
  K^T / V streams, the merge pass, `HTP_ATTN_NB` / `HTP_ATTN_L2FETCH`; §2.2 (`:289-323`):
  `attn_scratch` layout (rows + partials, `htp_attn_scratch_bytes`), KV layout unchanged (say so);
  §3 (`:375-437`): the `hvx-attn.c` line; §5.2: the new `attn` cases and STATs; §7: a rule only if
  the device deviates (the merge's block-relative narrowing and the ± 0.3 % band otherwise go to
  §8.2); §8.2: a "#58" block with the A/B/C/D table (date, unit, md5s), the KV GB/s reading and the
  `l2fetch` decision; §8.3: `attn_decode_p1024` before/after pcycles (v79, workers 6) and the
  unchanged `profile acc` / `graph` STATs; §9: rewrite the first bullet (what #58 measured, what is
  left toward 36 Mcyc: #26's vector softmax, DMA-to-VTCM or KV bytes), note ledger ⑤ closed.
* `HEXAGON_BENCHMARK.md`: three `nntrainer` rows (B or C, v79, 512 / 1024 / 4096, unit, handoff
  cited); goals "Now" for decode @512 and @4096 and the w4a8 row's ATTN arithmetic (#51's budget);
  log line.
* `docs/superpowers/specs/hexagon-hvx-optimization/05-attention-eltwise.md`: ⑤ done with the numbers;
  `07-follow-ups.md`: the decode-attention entry.
* `hvx-attn.c` `Follow-up:` note (source, implementer's PR): prefill row-block reuse → #26.
* `docs/plans/26-prefill-attn.md` (when #26 rebases): §2/§3 reference the #58 kernel as the base.
