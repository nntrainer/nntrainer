# 26 — Prefill attention: K^T reuse across query rows (P5) and the layer-0 qf32 drift bisection (⑮)

Issue: dlwlzzero/nntrainer#26 (`prio:p1`; HEXAGON.md §9 first two bullets, ledger 05 ①③④ and 07 ⑮).
Contract: `docs/plans/0000-agent-system-and-env.md`. Branch: `hvx/26-prefill-attn` from `hvx_impl`
(`355a60de`). **Rebase onto `hvx_impl` as soon as #35 (`hvx/35-v79-skel`) merges**: v79 is the primary
arch (user decision 2026-09-17), #35 makes it the sim/skel default and moves the graph / `profile acc`
STATs to a new baseline, and every gate below is `HEX_ARCH=v79` (v75 = skel compile + `attn`/`graph`
once at PR time, per the post-#35 gates skill). Until #35 merges, set `HEX_ARCH=v79` explicitly.
**One device handoff (§6)**, carrying both the P5 speed variants and the ⑮ per-op drift table.

## 1. Goal and gate

| # | Acceptance (issue) | Pass means |
|---|---|---|
| G1 | prefill @4096 Mcycles/tok drops measurably on the device | handoff §6, variant B vs A (same session, same image `qwen3_full4k`, `--chunk 128 --steps 64`): prefill DSP Mcyc/tok at 4096 **≤ 0.80 ×** A's (A ≈ v79 #23: 34.4 tok/s; the 4096 row is where ATTN is the largest single cost), and not worse than A at 512 / 1024 (±3 %). Decode DSP Mcyc at 4096 is **recorded, not gated** (§3 says what to expect); if B beats A there by > 10 % the benchmark "decode @4096" row moves too. |
| G2 | 8-token sim accuracy gate holds | `HEX_ARCH=v79 run_sim_test.sh profile acc` → `SIM_TEST profile PASS`; `profile_prefill_acc STAT` inside 0.1/0.1 (the value **moves** — vector softmax changes the exp/sum order — and is recorded in §8.3 as the P5 line next to #35's). |
| G3 | `attn` and `graph` sim tests pass | `SIM_TEST attn PASS` with the existing 2e-2/5e-3 bound on every case incl. the new ones of §5 step 2 (m=5, m=128 @ pos 0 and pos 129 with `MAX_SEQ 256`, m=1 @ pos 129), and the new relative signal `attn_prefill128_p512 STAT pcycles=<n>` **≤ 0.60 ×** the step-1 baseline on the same simulator (ledger 05 predicate was 2×; 0.60 is the floor below which the implementer stops and reports the per-stage split rather than opening the PR). `SIM_TEST graph PASS` (`graph_prefill`/`graph_decode` inside 3e-2/5e-2, `graph_partial_attn` inside 1e-2/1e-2); 13/13. |
| G4 | the drift source is named in HEXAGON.md §8 with the op index and a proposed fix or a §7 rule | the filled drift table of §6 (1-layer image, 128-token chunk, ops 0…18) has a `local_rel_rms` for every op; §8.2 names the first op whose local error exceeds **1e-3** (op index, kind, value) with a costed fix, **or** states that no op exceeds it and §7 gets the amplification rule of §4 (b). Either outcome closes ⑮ for this issue. |
| G5 | docs | §8 of this plan. |

No ABI change: `NNTR_HTP_ABI_VERSION` stays 4 (`nntr_htp_common.h:18-19`); the KV layout (§2.2, K^T
`[head_dim][max_seq]`, V `[max_seq][head_dim]`) is unchanged; `attn_scratch` is DSP-private. `lower_qwen3`,
`nntr_hexpack`, `.hexcfg`, `find_divergence.py` op indices: untouched. `ref_ops.c` gains one function
(`ref_graph_forward_range`) that `ref_graph_forward_upto` calls with `first = 0`, so the x86 reference PPL
(33.0195 / 184 on the P4 512-token prompt) is unchanged by construction and is re-checked in step 5's gate.

## 2. Where it lives

| File:line | Today | Change |
|---|---|---|
| `nntrainer/tensor/hexagon/htp/ops/hvx-attn.c:30-144` (`attn_worker`) | one job per kv head (`:53-54`), append and SDPA in the same worker (`:61-68`, `:70-142`), one query row at a time re-streams K^T (`:84-94`) and V (`:120-125`), scalar 3-pass softmax (`:96-107`) | two `wp_run` passes: `attn_append_worker` (KV append, split by (kv head, token)) then `attn_sdpa_worker` over jobs `(h, row block)`; a row block = the `group` q heads of kv head `h` × `HTP_ATTN_RQ/group` query rows sharing every K^T and V vector load; vector softmax (§4) |
| `hvx-attn.c:146-152` (`hvx_op_attn`, `Follow-up:` note) | one `wp_run` | two `wp_run`s (+1 barrier per ATTN ≈ 4.3 K cyc, 28/tok: < 0.06 ms/tok); note removed |
| `nntrainer/tensor/hexagon/htp/ops/htp_ops.h:31` (`attn_scratch`) | `[n_workers][max_seq]` fp32 | `[n_workers][HTP_ATTN_RQ][max_seq]` fp32; `#define HTP_ATTN_RQ 4` here (overridable by `-DHTP_ATTN_RQ=8` for the device sweep), comment states the layout |
| `nntrainer/tensor/hexagon/htp/htp_graph.c:81-82` | `wp_size * max_seq * 4 + 128` | `wp_size * HTP_ATTN_RQ * max_seq * 4 + 128` (4224 × 4 × 4 B = 66 KB/worker) |
| `test/hexagon/sim/test_attn.c:20-26,109-158` | `N_HEADS 4 / N_KV 2 / MAX_SEQ 64`; m=8 @0 then m=1 @8; scratch `wp_size * MAX_SEQ` (`:140-141`) | `MAX_SEQ 256`; cases m=8 @0, m=5 @8 (tail block of 1 with RQ=4), m=1 @13, then m=128 @0 (`attn_prefill128`, rel-RMS printed), m=128 @128 (L crosses 64-lane blocks, tail lanes 1…63), m=1 @256-1; plus the timed qwen-shape case `attn_prefill128_p512` (`n_heads 16 / n_kv 8 / MAX_SEQ 1024`, m=128 @ pos 512, `HAP_perf_get_pcycles` around `hvx_op_attn`, printed as `STAT pcycles=`); scratch sized with `HTP_ATTN_RQ` |
| `test/hexagon/sim/test_graph.c:291` (`graph_partial_attn`) | 1e-2/1e-2 | unchanged; it is the forward_upto-through-ATTN check |
| `test/hexagon/sim/ref_ops.c:226-316` | `ref_graph_forward_upto(…, n_ops_limit)` | add `ref_graph_forward_range(…, first, limit)`; `_upto` = `range(0, limit)` |
| `test/hexagon/hexagon_ref_run.cpp:10-18,48-52,139-141,205-223` | `--dump-op i` runs `[0, i)` and dumps op `i-1`'s output | add `--act-in <file>`: run `[0, i-1)`, overwrite ACT with the file (must be exactly `g.act_size` bytes, else error), run `[i-1, i)`, dump — i.e. the reference op applied to the DSP's own input (the local-error probe P4 did by hand for `down_proj`). Host-only; x86 build (`build_host_x86.sh:38-43`) |
| `tools/hexagon/op_drift.py` (new) | — | drives the ⑮ table (§4 (b)): for every op `i` of a 1-layer image, DSP output (`run_e2e_test.sh … --dump-op i+1`), DSP ACT state before op `i` (`--dump-op i --dump-buf 2 --dump-off 0 --dump-bytes <act_size>`; `htp_graph_buf_ref` allows the whole buffer, `htp_graph.c:196-203`), ref output (`hexagon_ref_run --dump-op i+1`) and ref-on-DSP-input (`--act-in`); prints one `DRIFT op=<i> kind=<k> prop_rel_rms=<p> local_rel_rms=<l> max_abs=<a>` per op; for the ATTN row additionally the numpy decomposition of §4 (b) |
| `test/hexagon/hexagon_e2e_test.cpp:176-190` | `--dump-op` over `forward_debug` | unchanged (the whole-ACT dump is already expressible; 3.9 MB rout buffer) |
| `nntrainer/tensor/hexagon/htp/hvx/hvx-base.h:221-240` (`hvx_vec_mpyacc_f32_f16`), `hvx-exp.h:168` (`hvx_exp_f32`) | — | unchanged; the kernel keeps both helpers (v79 native `Wsf_vmpyacc`, `HTP_FORCE_QF_HELPERS` opt-in from #35) |
| `Applications/CausalLM/hexagon/qwen3_lowering.cpp:219-243` | ATTN op | untouched |
| `docs/backend_guide/HEXAGON.md` §1 (`:30-84` ATTN description), §2.2 (`:257-275`), §7 (new rule 9), §8.2, §8.3, §9; `HEXAGON_BENCHMARK.md` rows + goals; ledger `05-attention-eltwise.md` task list, `07-follow-ups.md` ⑮ | — | §8 |

## 3. Design

**Chosen: row-blocked SDPA with GQA fusion, two-pass append/SDPA, vector softmax (ledger 05 ①③④ in one
kernel; ② and ⑤ skipped, see §9).**

*Jobs.* Pass 1 (`attn_append_worker`): the `m × n_kv` (token, kv head) appends split flat over workers —
the K^T strided scalar stores stay (28 K stores/tok/layer at m=128 against 2L×16 vector mpyacc per token;
ledger ② is "measured unnecessary" until the per-op profile says otherwise). Pass 2 (`attn_sdpa_worker`):
jobs `(h, b)` with `h < n_kv` and `b < ceil(m / TQ)`, `TQ = HTP_ATTN_RQ / group` (qwen3: group 2, RQ 4 →
TQ 2; decode m=1 → one block of 2 rows = the 2 q heads of the kv head), assigned interleaved
(`job = wid + k·nw`) so the causal length spread balances. 8 × 64 = 512 jobs at m=128, 8 at m=1.
Rows in a block: `r = g·TQ + tq`, query `q[t0+tq][h·group+g]`, causal `L_r = pos + t0 + tq + 1`; the
block runs to `L_max = max L_r` and rows with shorter `L_r` mask the extra positions (they only exist
inside one 64-lane block for TQ ≤ 4… in general the mask is per row: lanes `≥ L_r`).

*Scores.* For each 64-position block `p0`, `for i in 0..hd`: one `hvx_vmem(kh + i·max_seq + p0)` load,
then `RQ` × (`Q6_Vh_vsplat_R(q_r[i])`, `hvx_vec_mpyacc_f32_f16(acc_r, k, splat)`). `RQ` accumulator pairs
(RQ 4 → 8 vectors, RQ 8 → 16). Every K^T vector is loaded once per block instead of once per (q head, row):
`2·group·TQ` = 4× fewer K^T loads at RQ 4 (8× at RQ 8). Stores to `scores[r][p0…]` after the vshuff, as today.

*Softmax (per row, vector, 64 lanes per step).* (1) `Vqf32_vmpy_VsfVsf(s, scale)` → `Vsf_equals_Vqf32`
(reference order `s * scale` first, then max — `ref_ops.c:192`), tail lanes `≥ L_r` replaced by the
`-inf` bit pattern with `Q6_V_vmux_QVV(Q6_Q_vsetq_R(4·(L_r − p0)), …)`; (2) running max with the
**integer** order-preserving trick (`x ^ ((x >> 31) & 0x7fffffff)` then `Q6_Vw_vmax_VwVw`; one horizontal
fold of 32 lanes at the end) — no IEEE `Vsf_vmax` (§7 rule 1); (3) `Vqf32_vsub_VsfVsf(s, mx)` →
`Vsf_equals_Vqf32`, `hvx_exp_f32` over the 64-multiple length, then tail lanes **zeroed** with the same mask
(the borrowed exp is not specified at `-inf`, so the mask is applied after exp, not before); (4) sum as
`Vqf32_vadd_VsfVsf` → `Vsf_equals_Vqf32` per vector (the pattern of `hvx_vec_mpyacc_f32_f16`'s v75 path, no
chained qf32 adds), one horizontal fold; (5) narrowing to `s16` as today (`hvx_vec_f32_to_f16`, in place).
Scalar `1/sum` and the epilogue (`:127-140`) stay. NaN: the integer max treats a NaN score as a large value
and the row becomes NaN, where `ref_attn` skips it (`if (scores[p] > mx)`); same class as rule 6's NaN note,
documented, not handled.

*PV.* Rows in groups of 4 (RQ 4: one group; RQ 8: V is streamed twice): per position `p`, 2 V-row loads,
`4 × (splat s16[r][p], 2 mpyacc)`: 8 pairs = 16 accumulator vectors. Each V vector is loaded once per 4
rows instead of once per row.

*What it buys at decode (supervisor note).* m=1, so "reuse across query rows" is the GQA fusion only: the
two q heads of a kv head share each K^T and V load, halving the per-token KV traffic. At 4096 that
traffic is 28 layers × 8 kv heads × (1 MB K^T + 1 MB V) = 448 MB unique, read **twice** today = 896 MB per
token — more than the 598 MB of weights — and the context-dependent part of decode is 177.6 − 60.1 =
117.5 Mcyc/tok (v79, #23). If decode ATTN is KV-bandwidth-bound on silicon (the simulator cannot say;
gap rule 1), B should take a sizeable part of that 117 Mcyc; if it is issue-bound (splat + mpyacc per
K^T vector do not shrink with fusion) it takes little. Variant C of §6 (per-q-head jobs, no fusion, RQ 4)
is the control that separates the two: B faster than C at 4096 decode ⇒ bandwidth; equal ⇒ issue-bound,
and the next decode lever is a position split with a softmax merge (flash-decoding), not this kernel.
Neither the job count (8 at decode, as today) nor the barrier count moves the needle at m=1.

**Rejected: per-q-head jobs `(h, g, row block)` as the default** (ledger 05 ① as written). Better job
granularity at decode (16 vs 8), but every K^T/V vector is then loaded `group` times — the opposite of the
issue's title at the shape that matters. Kept as device variant C so the trade is measured, not argued.

**Rejected: fp32 probabilities in PV** (drop the `s16` narrowing; `Vqf32_vmpy_VsfVsf` on widened V). It
removes the one systematic fp16 rounding inside ATTN (candidate ⑮ source, ~2.8e-4 rel-RMS) but doubles
PV's multiply count (two `Wqf32_vmpy` + two adds per pair instead of one `mpyacc`). Only worth it if the
⑮ table (G4) shows ATTN's local error above 1e-3 — then it is the "proposed fix", costed by the same
handoff's speed columns, not adopted blind.

## 4. The ⑮ bisection (own step; stop condition)

(a) **What the 2.65 % is not.** P4 already showed the int16 kernel exact to its definition
(local rel-RMS 1.6e-4). The chain before `down_proj` in layer 0 is `ATTN → W8A8(o) → ADD → RMSNORM →
W8A8(gate/up) → SILU_MUL`; §6 of HEXAGON.md measured 5–8× amplification per int8 matmul on this image
(ATTN 1e-4 → next W8A8 3e-3). Two matmuls and a SILU take 1e-4 to ~1e-2, i.e. the observed 2.65 %
**can** be pure amplification of a within-spec ATTN difference (the fp16 probability narrowing gives
exactly ~1e-4). The bisection must therefore separate *propagated* from *local* error per op, or it
names the wrong op.

(b) **Procedure** (`tools/hexagon/op_drift.py`, one command on the workstation; device via
`run_e2e_test.sh`, reference via `hexagon_ref_run`; 1-layer image `nntr_hexpack --layers 1`, the 512-token
P4 prompt truncated to the first 128-token chunk; ops 0…18 = EMBED, RMSNORM, q/k/v, RMSNORM×2, ROPE,
**ATTN = op 8**, o, ADD, RMSNORM, gate/up, **SILU_MUL = op 14**, down, ADD, RMSNORM, LOGITS):

1. per op `i`: `ref_i` = reference output; `dsp_i` = DSP output; `dsp_state_{i}` = DSP ACT before op `i`;
   `loc_i` = reference op applied to `dsp_state_i` (`--act-in`). `prop_rel_rms = rms(dsp_i − ref_i) / rms(ref_i)`,
   `local_rel_rms = rms(dsp_i − loc_i) / rms(loc_i)`. (LOGITS: `prop` from `--eval` PPL only, as §6.)
2. the ATTN row is decomposed offline in numpy from `dsp_state_8` (q, kb, vb slots; offsets from
   `hexagon_ref_run --list-ops`) against `dsp_8`: (i) fp32 exact, (ii) fp16-rounded probabilities before PV,
   (iii) (ii) + fp16-rounded `s·scale`… whichever variant matches `dsp_8` to ≤ 3e-5 names the stage.
3. **Stop condition.** The first op with `local_rel_rms > 1e-3` is *the* drift source (G4): if it is ATTN,
   the numpy stage says which of "fp16 probabilities" / "qf32 accumulation order" it is and the fix is the
   rejected-alternative of §3 (fp32 PV) or a summation-order change, filed as a follow-up with the device
   cost from this handoff; if it is a W8A8 op, it is the epilogue/quantizer and goes to a new issue (out of
   scope here, the kernel is #35's). If **no** op exceeds 1e-3, the finding is "amplification, not error"
   and the deliverable is §7 rule 9: *DSP-vs-reference rel-RMS grows ~5–8× per int8 re-quantization;
   per-op correctness is judged on the local error (reference op on the DSP's own input) ≤ 1e-3, and on
   the `--eval` PPL band, never on the propagated rel-RMS.* No second device round for ⑮ inside #26.

## 5. Steps

`tools/docker/run.sh` from the repo root; `HEX_ARCH=v79` everywhere (sim needs `workers=6` on v79, so
pcycles are compared v79-to-v79 only). Simulator budget: `attn` per step (~1 min), `graph` when
`htp_graph.c` or `ref_ops.c` changes, `profile acc` + 13 tests once before the PR.

**Step 0 — branch** `hvx/26-prefill-attn` (done by the orchestrator). Do not add `build_*`.

**Step 1 — baseline probe (the one planner-deferred sim run, ~2 min).** Add only the timed
`attn_prefill128_p512` case and the `attn_prefill128` rel-RMS print to `test_attn.c` (kernel untouched),
`HEX_ARCH=v79 build_sim_test.sh`, `run_sim_test.sh attn`. Record `attn_prefill128_p512 STAT pcycles=<B0>`
and `attn_prefill128 STAT rel_rms=<r0>` in the PR body: `B0` is G3's denominator, `r0` (expected ~1e-4)
is the sim value of ATTN's local error before the rewrite. Gate: `SIM_TEST attn PASS`.

**Step 2 — tests first.** The remaining `test_attn.c` cases of §2 (`MAX_SEQ 256`, m=5/1/128 at the listed
positions), scratch sized by `HTP_ATTN_RQ`. Gate: `run_sim_test.sh attn` → PASS on the old kernel (the
old kernel ignores RQ; this proves the cases themselves).

**Step 3 — the kernel.** `hvx-attn.c` per §3 (append pass, SDPA pass, vector softmax, PV blocks),
`htp_ops.h` `HTP_ATTN_RQ`, `htp_graph.c` scratch. Review against the §7 list: qf-format ops only
(`Vqf32_vmpy/vsub/vadd_VsfVsf`, `Vsf_equals_Vqf32`, integer vmax, `vmux`; no `Vsf_vmax/vadd`), 128-B
alignment (`max_seq % 64 == 0` keeps K^T rows and score blocks aligned; `scores` per row = `max_seq` floats,
128-B multiple), worker count from `wp_size`. `clang-format-14`.
Gate: `run_sim_test.sh attn` → PASS, all STATs inside 2e-2/5e-3, `attn_prefill128_p512 pcycles ≤ 0.60·B0`,
`attn_prefill128 rel_rms` recorded (expected ≈ r0: the narrowing is unchanged, only the fp32 order moved).
Then `run_sim_test.sh graph` → PASS (`htp_graph.c` changed). If the pcycle gate fails, split the timer
(scores / softmax / PV) in a scratch build, paste the split on the issue, and stop.

**Step 4 — `HTP_ATTN_RQ=8` and variant C.** Build-time knobs only: `-DHTP_ATTN_RQ=8` (PV in two groups
of 4) and `-DHTP_ATTN_NO_GQA` (jobs `(h, g, block)`, `TQ = RQ`). Gate: `HEX_EXTRA_CFLAGS=-DHTP_ATTN_RQ=8
build_sim_test.sh` + `attn` PASS; same for `-DHTP_ATTN_NO_GQA`; STATs pasted into the handoff so a device
oddity is attributable. Rebuild plain afterwards.

**Step 5 — ⑮ tooling (host only, no DSP bytes).** `ref_ops.c` `ref_graph_forward_range`,
`hexagon_ref_run --act-in`, `tools/hexagon/op_drift.py`. Gate 1: `build_host_x86.sh`; `test_lowering`
→ `LOWER_TEST PASS`; regenerate `qwen3_full` and `hexagon_ref_run --eval` on the P4 512-token prompt →
**33.0195 / 184 unchanged**; `hexagon_ref_run --dump-op 15 --act-in <its own --dump-op 14 ACT dump>`
must reproduce its plain `--dump-op 15` output byte for byte (self-test of `--act-in`). No skel change
in this step: skel md5 from step 6 is the proof.

**Step 6 — pre-PR sim gates + skels.** Plain build: 13 tests (`HEX_ARCH=v79`) → 13/13; `profile acc` →
PASS, STAT recorded (G2). v75: `build_sim_test.sh` + `attn graph` → PASS (the kernel takes the
`< 79` qf helpers there). Skels: `HEX_ARCH=v79 build_skel.sh` (B), `HEX_EXTRA_CFLAGS=-DHTP_ATTN_RQ=8` (D),
`-DHTP_ATTN_NO_GQA` (C), and A = the skel of `hvx_impl` HEAD at the rebase point (build it from a clean
checkout of that commit; its md5 must match #35's B1 if #35 merged unchanged). `build_host_test.sh`.
Record every md5. Also pack the 1-layer image (`nntr_hexpack … --layers 1`) and its md5.

**Step 7 — handoff (device measurement unavoidable: G1 and G4).** Write
`docs/measurements/26-prefill-attn.md` per §6, commit it, `state:needs-measurement`, stop.

**Step 8 — after `state:measured`.** Read the table (§6 pass rules). Pick the default `HTP_ATTN_RQ`
(B or D) and fuse/no-fuse from the 512 sweep + the 4096 goal rows; the ⑮ verdict per §4 (b) 3. Docs per
§8. Commits: `[htp]` tests+kernel, `[htp]` knobs, `[tools]` drift tooling, `[docs]`; `git commit -s`;
PR into `hvx_impl`; `state:review`. If G1 fails (B not ≤ 0.80·A at 4096) but G3 held, the kernel still
merges only if it is not slower anywhere (it is a correctness-neutral rewrite); the issue returns to
`state:needs-plan` with the per-variant 4096 line quoted.

## 6. Handoff (`docs/measurements/26-prefill-attn.md`, `hexagon-handoff` template)

Four skels, all SDK 6.4 container builds at the step-6 commit, `HEX_ARCH=v79`:

| variant | skel | what it answers | runs |
|---|---|---|---|
| A | `libnntr_htp_skel.base.so` (pre-P5 kernel, same commit base) | control, same session | 512 / 1024 / 4096 speed; `--eval` at 512 |
| B | `.rq4.so` (default: GQA fused, RQ 4) | G1 | 512 / 1024 / 4096 speed + `--eval` at each; **drift table** on the 1-layer image |
| C | `.nogqa.so` (`-DHTP_ATTN_NO_GQA`) | decode-4096 bandwidth vs issue (§3) | 512 sweep + 4096 (decode column is the point) |
| D | `.rq8.so` (`-DHTP_ATTN_RQ=8`) | RQ knob | 512 sweep; 4096 only if B passes G1 and time allows |

Images: `qwen3_full` (512/1024), `qwen3_full4k` (`--max-seq 4224`), `qwen3_1l` (1-layer, for the
drift script); token files `t512.i32` (P4 prompt), `t1024.i32`, `t4096.i32`, `t128.i32` (first chunk).
Steps as `docs/measurements/23-sdk64-baseline.md`, plus one line:
`python3 tools/hexagon/op_drift.py /tmp/qwen3_1l /tmp/t128.i32 --serial R3CY10WM83Y --chunk 128`
(≈ 40 harness invocations, ~10 min; prints `DRIFT op=…` lines and `DRIFT attn_stage=…`).
Result table columns (contract §2): prefill tok/s, **prefill DSP Mcyc/tok**, decode median host ms, decode
tok/s, decode DSP Mcyc/tok, `--eval` PPL, top-1, x86 ref PPL (33.0195 / 184 at 512), skel md5 from the
log; reference cells from #23 v79 (207.4 / 131.2 / 34.4 prefill, 60.1 / 76.5 / 177.6 decode Mcyc; PPL band
33.07–33.45 at 512). Second table: the 19 `DRIFT` rows. Estimated device time ~45 min.

Pass: G1 on B vs A; PPL of B/C/D inside the #23 band at every context (a variant outside the band is a
kernel bug, not a rounding note — the sim STATs bound the rewrite to fp32-order changes). Reading C: §3.

## 7. Risks

* **Sim pcycles vs device time** (gap rule 1). The simulator charges neither the K^T/V re-streaming
  from L2/DDR at L=4096 nor DDR contention with the weight path; the 0.60 gate is a compute-only signal,
  and the device 4096 rows are the verdict. A/B/C/D in one session make bandwidth (B vs C) and register
  pressure (B vs D) visible separately.
* **qf32 / IEEE numerics on silicon** (gap rule 2). The new softmax is the first vector max/exp/sum in
  ATTN on this part; the integer-max trick and `vmux` are exact by construction, the qf32 add/sub/mpy are
  the same instructions the P3/P4 epilogues already proved on silicon (§7 rules 5, 8). The `--eval`
  columns per variant plus the drift table (ATTN row, local error) show any silicon-only deviation directly,
  which is what the sim `attn_prefill128 rel_rms` is there to be compared with (three-way record).
* **Tail-lane handling.** Lanes `≥ L_r` were harmless garbage before (ignored by scalar loops); now
  they feed vector max and sum. `MAX_SEQ 256` cases at pos 128 / 255 exercise tails of 1…63 lanes;
  `graph_partial_attn` and the 1-layer drift table catch a mask bug at model shape.
* **Scratch growth** `wp_size × RQ × max_seq × 4`: 6 × 8 × 4224 × 4 = 811 KB at RQ 8, DDR (memalign),
  fine; the validator's `max_seq % 64` keeps every row block aligned.
* **#35 rebase.** Disjoint files (`hvx-quant.h`, `hvx-base.h`, `build_*.sh`, `test_quant.c`) except
  `HEXAGON.md` §8.3/§9 and the gates skill — doc hunks only. #35 moves `graph`/`profile acc` STATs; this
  plan moves them again (softmax order), so the P5 record must be taken **after** the rebase, once.
* **Stale artifacts** (gap rule 3). Four skels of one size class plus a third image (`qwen3_1l`);
  every result row repeats the md5 the harness printed; `push_if_changed` is md5-based (⑪).
* **Drift procedure yields "no op > 1e-3".** Then G4 is met by rule 9 and no kernel fix is owed; the
  issue text's "proposed fix or a §7 rule" allows it. The risk is only that the reviewer expects a fix.
* **Container busy** (#36). Step 1 is the only sim run before the kernel exists; everything else queues
  behind the per-step `attn` runs, each ~1 min.

## 8. Docs to update (G5)

* HEXAGON.md §1 (`:30-84`, the ATTN paragraph): two-pass structure, row blocks, GQA fusion, `HTP_ATTN_RQ`.
  §2.2 (`:257-275`): `attn_scratch` layout `[n_workers][RQ][max_seq]`; KV layout unchanged (say so).
  §7: new rule 9 — softmax tail-lane rule (mask at max, zero after exp), integer max instead of
  `Vsf_vmax`, NaN behaviour; plus, per §4 (b) 3, either the amplification rule or the named-op paragraph.
  §8.2: A/B/C/D rows from the handoff with the date and md5s; the ⑮ verdict paragraph (op index, values).
  §8.3: the P5 `profile acc` STAT and `attn_prefill128_p512` before/after pcycles (v79, workers 6).
  §9: drop the first two bullets; add follow-ups that the table produced (position split for decode if C ≈ B;
  fp32 PV if ATTN local error > 1e-3; ledger ② if the append pass shows in the per-op profile).
* `HEXAGON_BENCHMARK.md`: three new `nntrainer` rows (P5, v79, 512/1024/4096) with the handoff cited;
  goals "Now" for prefill @512 and decode @4096 if they moved; log line.
* Ledger `05-attention-eltwise.md`: task list marked (①③④ done, ② ⑤ "measured unnecessary" with the
  numbers); `07-follow-ups.md` ⑮: result + pointer to §8.2.
* `.claude/skills/hexagon-gates`: no change (the `attn` per-step rule already covers this kind).

## 9. Out of scope

* Ledger ② (vector K^T transpose on append) and ⑤ (decode column split): skipped with the numbers of §3;
  re-opened only by the per-op profile.
* Position-split / flash-decoding softmax merge for decode; op fusion (ledger ②/ABI v5); any ABI bump.
* Fixing a W8A8 quantizer/epilogue drift if the ⑮ table names one: new issue (that code is #35's).
* Adopting fp32 PV: only as a follow-up costed by this handoff.
* HMX / HexKL (stage 2); the host/RPC logits path (#24).
