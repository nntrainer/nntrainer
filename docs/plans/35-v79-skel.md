# 35 — v79-native skel: make the quantiser decode jam-independent, verify on device, flip the defaults to v79

Issue: dlwlzzero/nntrainer#35 (`prio:p1`, rewritten 2026-09-17; ledger ④ steps 3–4).
Contract: `docs/plans/0000-agent-system-and-env.md`. Branch: `hvx/35-v79-skel` from `539c3604`
(= #23's SDK 6.4 baseline, which carries HEXAGON.md §5.2 / §7 / §8.3 / §9 and
`docs/measurements/23-sdk64-baseline.md`; rebase onto `hvx_impl` once #23's PR merges).
Order: after #24 and #33, before #26. **One device handoff (G3, §6).**

## 1. Goal and gate

Gates are the issue's, made measurable. The build is judged on **both** simulators and the
device (§7 rule 1).

| # | Acceptance (issue) | Pass means |
|---|---|---|
| G1 | v79 sim: 13/13 + `profile acc` PASS by way (i) | `HEX_ARCH=v79 run_sim_test.sh` prints 13 × `SIM_TEST <name> PASS` (`smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed attn logits graph`) and `SIM_TEST profile PASS`; `quant` tie rows (existing + the extended one of §5 step 2) byte-identical; `quant_generic pm1 <= 13/65536`, `quant16_generic pm1 <= 512/25600` (the existing bounds, `test_quant.c:124,148`); `matmul_w8a8_m1`, `matmul_dma ref_*`, `logits` inside their existing 2e-3/1e-3 and 5e-3/1e-2 bounds (`test_matmul.c:91`, `test_matmul_dma.c:120`, `test_logits.c:80`). No test bound is widened. |
| G2 | v75 sim unchanged | 13/13 + `profile acc` PASS with every STAT **bit-identical** to §8.3's SDK 6.4 record — `quant_generic pm1=0/65536`, `matmul_w8a8_m8 0.0078125/0.000788644`, `logits 7.62939e-06/2.36832e-07` — **except the int16 quantiser rows and the graph / `profile acc` STATs, which move to the values recorded in HEXAGON.md §8.3 (new baseline)**: `quant16_generic pm1=162/25600` (was 167), `graph` prefill `0.0273907/8.00437` (was `0.0245416/9.3795`) / decode `0.0174583/24.125` (was `0.0202219/23.975`), `profile_prefill_acc 0.0757427/66.5909` (was `0.077216/98.2637`). Reason: sign-symmetric decode — the old magic add on v75 rounded positive int16 near-ties (`n + 0.51`) up and negative ones toward zero; no jam-free decode reproduces that, and an arch branch would enshrine it (supervisor decision, issue #35 comment 2026-09-17: option 1). |
| G3 | device, one handoff (§6) | B1 (v79, G1 build): `--eval /tmp/t512.i32` (P4 prompt) inside the P4 band against x86 33.0195 / 184, i.e. PPL 33.07–33.45; prefill tok/s at 512/1024/4096 within ±5 % of #23 variant B (207.4 / 131.2 / 34.4) and decode DSP Mcyc/tok within ±5 % of 60.1 / 76.5 / 177.6; no hang, SSR or FARF fatal. B2 (`-DHTP_FORCE_QF_HELPERS`) and A (v75 control) recorded, not gated. |
| G4 | flip | `build_sim_test.sh` and `run_sim_test.sh` default `v79` (`build_skel.sh` already does, §3); `.claude/skills/hexagon-gates` runs rungs 2–3 on v79 and builds the v75 skel in rung 4 before every PR; HEXAGON.md §5.2/§5.3 command lines say `HEX_ARCH=v79`. |
| G5 | docs | §8 of this plan. |

No ABI change: `NNTR_HTP_ABI_VERSION` stays (`nntr_htp_common.h`), no wire, image or `.hexcfg`
field moves; `lower_qwen3`, `ref_graph_forward`, `nntr_hexpack`, `find_divergence.py` untouched.

## 2. Evidence (planner probe, 2026-09-17, one v79 `quant` run)

Probe patch: `/private/tmp/claude-501/-Users-dlwlzzero-Projects-nntrainer/fe085278-1edc-4422-99a1-28120d10626d/scratchpad/35-qf32-probe.patch`
(43 lines added to `test/hexagon/sim/test_quant.c`, not committed; step 1 replays it on v75).
Built with `HEX_ARCH=v79 build_sim_test.sh` at `539c3604` (`-mhvx-ieee-fp probe -> -mhvx-ieee-fp`),
run `HEX_ARCH=v79 run_sim_test.sh quant` (`runmain=hexagon_toolv19_v79`). Findings, all on v79:

1. **The issue's hypothesis is confirmed: the v79 qf32 chain has no Von Neumann jam.**
   For 13 samples (`±1.5 ±2.5 ±0.5 1 0.25 1/3 ±100.7 3.3 126.6`):
   `Vsf_equals_Vqf32(Vqf32_vmpy_VsfVsf(p, 0.7))` **equals the IEEE RNE product bit for bit** in
   every case; `Vqf32_vmpy(p, 1.0)` and `Vqf32_vadd_VsfVsf(p, 0)`→sf are exact;
   `hvx_vec_f16_to_f32` (the `>= 79` `Wsf_vmpy_VhfVhf` widening) is exact. The magic add
   `Vsf_equals_Vqf32(Vqf32_vadd_VsfVsf(p, 1.5*2^8))` minus the magic bits gives
   `d = RNE(p * 2^15)`: **even** for every exact tie (`1.5 -> 49152`, `-1.5 -> -49152`,
   `-100.7 -> -3299738` from `-3299737.5`), odd only where RNE happens to land odd
   (`1/3 -> 10923`). Rule 6's v75 form `d = 2*floor(p*2^14)+1` (always odd) does not hold on v79.
2. **The failing decode is `hvx-quant.h:69-70`, and it is a sign bug, not noise.** With `d` exact,
   `Q6_Vw_vasr_VwR(d, 1) + Q6_Vuw_vlsr_VuwR(d, 31)` (meant to strip the jam's implicit half and
   shift negative values toward zero) moves every **negative** value one 2^-14 unit toward zero.
   Kernel on a hand row (`x[0]=127`, so `inv = 1.0`): `1.5->2 2.5->2 3.5->4 4.5->4 7.5->8`
   (right) but `-1.5->-1 -3.5->-3 -7.5->-7` (ref `-2 -4 -8`); `-2.5->-2 -4.5->-4 -126.5->-126`
   are right only because RNE-to-even and toward-zero coincide there. The `rand0/rand15/k3072`
   failures are the same case: `x/scale = -63.5` exactly in fp32, kernel `-63`, ref `-64`.
3. **All four failing tests are this one quantiser.** `test_matmul.c:76` calls
   `hvx_op_matmul_w8a8`, which quantises `x` through `quant_worker` (`hvx-matmul.c:216-226`,
   `wp_run` at `:324`, `:338`, `:358`) — the scalar `ref_quant_row` is only the *reference*
   side. A single ±1 int8 element changes one dot by `|w| <= 127` and gives the observed
   `max_rel` 0.046 (`matmul_w8a8_m1`) and 0.27 (`matmul_dma ref_4096k`, `logits`): 40–250 fp16
   ulps, far beyond anything the epilogue can do. The issue's "w8a8_m1 fails in the epilogue
   alone" is therefore wrong. The `mm_tile` epilogue (`hvx-matmul.c:81-83`) is two
   `Vqf32_vmpy`→sf products (= IEEE on v79 by finding 1) and one `Vhf_equals_Wqf32` narrowing:
   **no change needed**; the existing 2e-3/1e-3 bound covers v75's one-ulp qf32 difference (§7 rule 5).
4. Unchanged from #23: `quant_generic 13/65536`, `quant16_generic 229/25600`, the four FAIL lines,
   `Fatal Error: Cause: 0x5 Cause2: 0x20` on the abort path.

Also found: the existing tie row (`test_quant.c:88-93`) only contains `+(even).5` and
`-(odd).5`, so it cannot distinguish RNE from round-half-toward-minus-infinity and never
exercises `+(odd).5` / `-(even).5`. Step 2 adds those (must pass on v75 as well; see §7).

## 3. Where it lives

| File:line | Today | Change |
|---|---|---|
| `nntrainer/tensor/hexagon/htp/hvx/hvx-quant.h:57-59` | `magic`, `one`, `half1` splats | `magic` goes; add `sign/exp/mant` masks and `150 - sh` splat |
| `hvx-quant.h:67` | `f = Vsf_equals_Vqf32(Vqf32_vmpy_VsfVsf(w, vinv))` | **unchanged** (the product; per-arch rounding stays, bounded by the generic-rate STATs) |
| `hvx-quant.h:68-70` | magic add + `vasr(d,1) + lsr(d,31)` decode | replaced by the integer decode of §4 producing the same signed `d'` (= `sign * floor(|f| * 2^sh)`) |
| `hvx-quant.h:71-75` | ties-to-even step + shift | **unchanged** |
| `hvx-quant.h:19-43` | comment describes the jam-based decode | rewritten for the integer decode; the 2^-sh resolution, 3.6e-5 / 1 % modelled rates and the NaN note stay |
| `nntrainer/tensor/hexagon/htp/ops/hvx-matmul.c:68-95` (`mm_tile`), `:216-226` (`quant_worker`) | — | **untouched** |
| `nntrainer/tensor/hexagon/htp/hvx/hvx-base.h:168`, `:221`, `:242` | `#if __HVX_ARCH__ >= 79` / `< 79` | `#if __HVX_ARCH__ >= 79 && !defined(HTP_FORCE_QF_HELPERS)` / `#if __HVX_ARCH__ < 79 \|\| defined(HTP_FORCE_QF_HELPERS)` (B2 opt-in; `:130` `hvx_vec_neg_f32` is a sign flip either way, left alone) |
| `test/hexagon/sim/test_quant.c:88-93` | tie row `+(even).5`, `-(odd).5` | second tie row with sign independent of parity (§5 step 2) |
| `test/hexagon/sim/test_quant.c:124`, `:148` | rate bounds 13 / 2 % | unchanged |
| `tools/hexagon/build_skel.sh:16` | `HEX_ARCH="${HEX_ARCH:-v79}"` **already** (line 8 comment says "override: v75") | keep; comment states v79 default, v75 fallback |
| `tools/hexagon/build_sim_test.sh:9`, `tools/hexagon/run_sim_test.sh:6` | `v75` | `v79` (G4) |
| `.claude/skills/hexagon-gates/SKILL.md:22-23`, `:54-55`, `:65`, `:69-71`, `:76` | v75 commands, v79 "only at PR time" | v79 commands; v75 = skel compile in rung 4 plus the 13 tests when `hvx-quant.h`, `hvx-base.h` or a qf32 kernel changes |
| `docs/backend_guide/HEXAGON.md` §5.2 `:462-497`, §5.3 `:548-553`, §7 rule 1 `:682-706`, rule 6 `:740-760`, §8.2 `:985-996`, §8.3 `:1224-1229`, §9 `:1279-1303`; `HEXAGON_BENCHMARK.md:33-35`, `:46-48`, `:61` | — | §8 |

## 4. Design

**Chosen: way (i), an integer-only decode that reproduces v75's 2^-sh grid rounding on any
qf32 rounding mode.** The product `f` stays as it is (line 67). Everything after it becomes
integer arithmetic on the sf bits, so no qf32 add and no `Vsf_equals_Vqf32` sits between the
product and the integer result, and the jam (v75) / no-jam (v79) difference can no longer
reach the rounding step:

```
sg = Vw_vasr_VwR(f, 31)                                   /* 0 or -1 */
af = V_vand_VV(f, 0x7fffffff)
e  = Vuw_vlsr_VuwR(af, 23)
m  = V_vor_VV(V_vand_VV(af, 0x007fffff), 0x00800000)      /* 24-bit significand */
s  = Vw_vmin_VwVw(Vw_vmax_VwVw(Vw_vsub_VwVw(splat(150 - sh), e), 0), 31)
t  = Vw_vlsr_VwVw(m, s)                                    /* floor(|f| * 2^sh) */
d' = Vw_vsub_VwVw(V_vxor_VV(t, sg), sg)                    /* sign * t */
a  = d' + half1 + ((d' >> sh) & 1);  q = a >> sh           /* lines 71-75, unchanged */
```

`sh` = 14 (int8) / 6 (int16); every intrinsic (`vasr/vlsr/vasl_VwVw`, `vmin/vmax_VwVw`) is
unguarded in `hvx_hexagon_protos.h` (checked, SDK 6.4.0.2), so the same bytes compile on v75
and v79. `s` is clamped to `[0, 31]`: `|f| < 2^-22` and `f = 0` give `t = 0` (correct), and
NaN/inf (`e = 255`) give `t = m` which the existing `vpack_sat` saturates — deterministic,
documented in the header comment (today a NaN element is whatever the qf32 add makes of it).
`t` fits 32 bits because `|p| <= qmax` (`127 * 2^14`, `32767 * 2^6 < 2^21`). About nine extra
integer ops per 32 elements; `quant_worker` is not a profile line of its own and the G3 ±5 %
speed band is the check.

Why this reproduces v75 bit for bit: rule 6 documents today's v75 path as "floor at 2^-sh,
then ties-to-even on that grid", and the tie/rand/generic rows pass on v75 with exactly that
behaviour; the new decode computes that floor directly. It is also correct on v79: with `f`
= the IEEE product (finding 1), `floor(|f| * 2^sh)` is exact on every tie, so the tie rows
pass, and the generic rate becomes the modelled 3.6e-5 / 1 % (bounds 2e-4 / 2 %). Step 1's v75
probe run checks the documented form before any kernel edit (§7, first risk).

**Rejected (ii), a documented per-arch bound.** Finding 2 shows the v79 failure is not a
rounding-mode nuance but a decode that rounds every negative tie toward zero — a systematic
bias the device hides only because exact `.5` products are rare on real activations. A bound
would enshrine a known-wrong rounding on the arch about to become the default and would need
the `matmul`/`logits` tests to accept ±1 int8 on one element (`max_rel` 0.27), which is not
"one fp16 ulp".

**Rejected (iii), full-precision `lrintf(f)` on both arches** (same integer decode without the
2^-sh floor, i.e. RNE at bit 0 of the product). Strictly better numerics — on v79 the
quantiser would become bit-exact with `ref_quant_row` for every input — but it moves the v75
STATs (`quant16_generic 167/25600`, `profile acc`, `graph`) and so fails G2 as the issue wrote
it. Filed as a follow-up for after the flip (§9), when v79 is the reference arch and a
re-baseline is one record, not two.

**Rejected: `Q6_Vw_equals_Vsf`** (`>= 73`) — an sf→int conversion whose rounding mode is
exactly the kind of per-arch float semantics this change removes; it would need its own probe
on both arches and on silicon.

**Rejected: an `__HVX_ARCH__ >= 79` branch in `quant_row`.** Keeps v75 bytes identical for
free, but leaves two decodes to maintain and leaves rule 6's jam assumption in the tree.

**Epilogue:** no change (finding 3). **`HTP_FORCE_QF_HELPERS`:** a compile-time opt-in only,
never set by default; it exists so G3's B2 answers ledger ④ (3) (IEEE vs qf helpers on the
same silicon, same toolchain) without a second branch.

## 5. Steps

Commands run from the repo root through `tools/docker/run.sh`; `HEX_ARCH` is explicit
everywhere until step 6. Simulator budget (contract §7): per step only the affected test kind;
the 13 tests + `profile acc` once per arch before the PR.

**Step 0 — branch.** `hvx/35-v79-skel` from `539c3604` (done by the orchestrator). Do not
`git add build_hexagon/`, `build_x86_hexagon/`.

**Step 1 — v75 probe (one `quant` run, ~1 min).** Apply the probe patch (§2) to
`test_quant.c` *without committing*, `HEX_ARCH=v75 build_sim_test.sh`,
`HEX_ARCH=v75 run_sim_test.sh quant`, paste the `SIM_PROBE` lines into the PR body, revert.
Expected (rule 6): `d` **odd** for every sample; `mpy0.7` differs from IEEE by one ulp on some
samples; the kernel hand row gives `lrintf` on every entry **including** `+1.5 -> 2`,
`-3.5 -> -4`, `-7.5 -> -8`. If any hand-row entry is wrong on v75, stop: comment the line on
the issue — the existing v75 record is then itself off on one tie class and the supervisor
decides between G2's bit-identity and correctness before step 3 (§7).
Gate: the `SIM_PROBE kernel` line on v75 matches `lrintf` on all 16 entries.

**Step 2 — tests first.** `test_quant.c`: after the existing tie row (`:88-93`) add
`tie2`: `x[0] = 127`, `x[i] = ((i >> 1) & 1 ? -1 : 1) * ((i % 127) + 0.5)` for `i >= 1`
(sign independent of parity, so all four classes appear), `check_row("tie2", 1024, 1u, NULL)`;
same pattern for int16 as `i16_tie2` (`x[0] = 32767`, `i % 2047`). Nothing else changes; the
rate bounds stay.
Gate: `HEX_ARCH=v75 run_sim_test.sh quant` → `SIM_TEST quant PASS`, `quant_generic 0/65536`,
`quant16_generic 167/25600` (bit-identical; the new rows must not change the STAT lines).
v79 is expected to still FAIL here (`tie` at `i=1`).

**Step 3 — the decode.** `hvx-quant.h` as in §3/§4, header comment rewritten (keep the
modelled rates, the dyadic-data remark and the NaN behaviour; replace the jam explanation with:
the product's per-arch rounding is the only qf32 dependence left, and it is what the generic
STATs bound). `clang-format-14`.
Gate (v79 first, it is the one that changes): `HEX_ARCH=v79 build_sim_test.sh`;
`run_sim_test.sh quant` → PASS, tie/tie2 exact, `quant_generic` and `quant16_generic` STATs
recorded (expected `<= 13` and near 167); then `matmul`, `matmul_dma`, `logits` → PASS inside
their existing bounds. Then `HEX_ARCH=v75 build_sim_test.sh` + `quant`, `matmul` → PASS with
the G2 STATs bit-identical. A v75 STAT that moves means the decode is not v75's floor; compare
against step 1's `SIM_PROBE` lines and fix the decode, never the bound.

**Step 4 — `HTP_FORCE_QF_HELPERS` and the skels.** `hvx-base.h` conditionals as in §3.
Gate 4: `HEX_ARCH=v79 build_skel.sh` → `libnntr_htp_skel.v79.so` (B1);
`HEX_ARCH=v79 HEX_EXTRA_CFLAGS=-DHTP_FORCE_QF_HELPERS build_skel.sh` → `.v79qf.so` (B2);
`HEX_ARCH=v75 build_skel.sh` → `.v75.so` (A); `build_host_test.sh`. Record every md5.
One extra sim sweep for B2's identity only: `HEX_ARCH=v79 HEX_EXTRA_CFLAGS=-DHTP_FORCE_QF_HELPERS
build_sim_test.sh` + `run_sim_test.sh quant matmul attn rmsnorm eltwise` (the kinds whose helpers
switch) → PASS lines pasted into the handoff, so a B2 device oddity is attributable.

**Step 5 — pre-PR simulator gates (once per arch).** Rebuild the plain v79 and v75 libraries
(no extra flags), then for each: the 13 tests (~5 min) and `profile acc` (v75 ~10 min, v79
~17 min with `workers=6`). Pass: G1 and G2 as in §1. Record the v79 STATs (they are the new
§5.2/§8.3 v79 reference; `profile_prefill_acc` on v79 will move from `0.0899355/49.2742`
because negative ties in W8A8/int16 rows now round to even).

**Step 6 — handoff (G3, device measurement unavoidable).** Write
`docs/measurements/35-v79-skel.md` per §6, commit it with the md5s, set
`state:needs-measurement`, stop. The flip (step 7) waits for the filled table.

**Step 7 — flip + docs (after `state:measured`, G3 pass).** `build_sim_test.sh:9`,
`run_sim_test.sh:6` → `v79`; `build_skel.sh:8` comment; `hexagon-gates` skill; HEXAGON.md and
the benchmark per §8. Gate: `tools/docker/run.sh ./tools/hexagon/build_sim_test.sh` with no
`HEX_ARCH` prints `built: ... (v79)`; skel md5s unchanged from step 4 (no DSP bytes change in
this step). One `[htp]` commit (decode + tests), one `[htp]` (`HTP_FORCE_QF_HELPERS`), one
`[tools]` (defaults + skill), one `[docs]`; `git commit -s`; PR into `hvx_impl`; issue →
`state:review`. If G3 fails: no flip, B1's numbers go to §8.2 as data, and the decode stays
(it is right on both simulators); the issue returns to `state:needs-plan` with the device
line quoted.

## 6. Handoff G3 (`docs/measurements/35-v79-skel.md`, `hexagon-handoff` template)

Three skels (slot 4 is free; #25's device matrix may ride along **only if the supervisor says
so** when the handoff is written), all built in the container at the step-4 commit, SDK 6.4.0.2:

| variant | skel | purpose | runs |
|---|---|---|---|
| B1 | `libnntr_htp_skel.v79.so` | the gate | 512 / 1024 / 4096 × (`--chunk 128 --steps 64`, `--eval`) on `qwen3_full` / `qwen3_full4k`; plus `--eval` on `/tmp/t512.i32` (P4 prompt) |
| B2 | `libnntr_htp_skel.v79qf.so` (`-DHTP_FORCE_QF_HELPERS`) | ledger ④ (3): IEEE vs qf helpers on the same silicon; recorded | 512 only: speed + `--eval` (`t512_23.i32`) |
| A | `libnntr_htp_skel.v75.so` | same-session control | 512 only: speed + `--eval` |

Steps as in `docs/measurements/23-sdk64-baseline.md` (same images, token files and log
lines; `md5sum` of the skel before each variant, `adb shell md5sum` after the last). Result
table columns: prefill tok/s, decode median host ms, decode tok/s, DSP Mcyc/tok, PPL, top-1,
each with the #23 reference in brackets (B1 vs #23 B; A vs #23 A). Estimated device time
~25 min. The tie row itself cannot run on the device (no unit-test harness there, §7 rule 6
"still to do"); the `--eval` band is the silicon evidence, as it was for #23.

Pass (B1 only): §1 G3. Reading B2: if B2 matches B1's PPL/top-1 and speed within noise, the
2026-08 v79 failures were a toolchain-8.x artefact (rule 1 closes); if B2 is slower but
correct, the IEEE helpers are a genuine v79 speed source; if B2 is *wrong*, the qf helpers
have a v79-specific problem that the v75 skel does not see — record, file, do not block.

## 7. Risks

* **Rule 6's description of v75 is inexact** (the probe could not run on v75 within budget).
  Exposed by step 1 before any kernel edit and by G2's bit-identity after; both are cheap
  (`quant` runs in about a minute). If v75 fails the new `tie2` row, the choice "bit-identical
  v75" vs "correct v75" is not the implementer's: comment and wait for the supervisor.
* **A near-tie in the fixed test data.** `matmul_w8a8_m1` / `ref_4096k` / `logits` compare
  one row each; if an element's IEEE product (v79) and jam product (v75) fall on different
  sides of a `.5`, one arch gets ±1 int8 and the test fails by `max_rel` ~0.2 on that arch only.
  Rate ~1e-5 per element, a few thousand elements: a few percent. If it happens, change the
  test's seed/data, not the bound (the bound stays "one fp16 ulp").
* **Simulator ≠ device for qf32 numerics** (gap rule 2). The v79 simulator says "IEEE-exact
  products"; silicon may not. G3's `--eval` band and the B1/B2/A three-way on the same session
  make a silicon-only deviation visible as a PPL gap between B1 and A. A device-only divergence
  becomes the next issue with the 1-layer `find_divergence.py` procedure (HEXAGON.md §6).
* **Bandwidth / speed** (gap rule 1). The decode adds integer ops; the simulator cannot price
  them against DDR-bound phases, so the ±5 % band against #23 B is the only speed verdict.
  A miss outside the band with matching PPL blocks the flip, not the decode.
* **Stale artifacts** (gap rule 3). Three skels of the same size class; every row repeats the
  md5 read from the device. The workstation rebuild path of #23 (SDK 6.4.0.1) is acceptable
  again if the Mac's binaries cannot be copied, provided the handoff table records which SDK
  built what.
* **v79 `profile acc` at ~17 min** (6 workers) becomes the default gate cost after the flip.
  Acceptable under contract §7 (once per PR); the skill text says so.
* **#23 PR pending**: this branch starts on #23's commit; `git rebase hvx_impl` after #23
  merges is conflict-free by construction (same tree). #24/#33 touch no file in §3.

## 8. Docs to update (G5)

* HEXAGON.md §5.2 (`:483-497`): replace the 9/4 paragraph with the v79 record after the fix
  (13/13, `profile acc` STAT, `quant_generic`/`quant16_generic` STATs on both arches, `tie2`
  row described next to `tie`); command lines `:465-466` → `HEX_ARCH=v79`, with a line that
  v75 is still built and run before every PR.
* §5.3 (`:551`): `HEX_ARCH=v79 ... build_skel.sh` (default), v75 fallback line.
* §7 rule 1 (`:682-706`): resolution paragraph from B2 — either "toolchain-8.x artefact,
  closed" or what B2 showed; the shipping skel is `HEX_ARCH=v79`; v75 remains the documented
  fallback that runs unchanged on v79 silicon.
* §7 rule 6 (`:740-760`): the decode no longer depends on the jam; state the integer decode,
  that the product is the only qf32 op left, the per-arch product rounding (jam on v75,
  IEEE-exact on the v79 simulator per this probe), the `tie2` coverage, and that the tie row
  on silicon is still covered only by `--eval`.
* §8.2 (`:985-996`): B1/B2/A rows from the handoff; §8.3 (`:1224-1229`): the new v79 STAT
  line and wall times.
* §9 (`:1279-1303`): remove the v79 bullet; add a follow-up bullet for (iii) (full-precision
  quantiser on the v79 reference arch) if the supervisor wants it in the ledger.
* Ledger ④ marked complete where it is cited (`HEXAGON.md:704`, `HEXAGON_BENCHMARK.md:33`).
* `HEXAGON_BENCHMARK.md`: rows `:33-35` become the shipping rows (note updated with the #35
  handoff); goals table `:46-48` "Now" moves to the v79 numbers; log line `:61` gets a
  successor entry.
* `.claude/skills/hexagon-gates/SKILL.md`: v79 commands in rungs 2–4, v75 policy in the
  budget bullet (`:22-23`) and rung 3 note (`:69-71`).

## 9. Out of scope

* No `#error` on IEEE helpers, no change to `hvx_vec_mpyacc_f32_f16` / `add/sub/mul` beyond
  the `HTP_FORCE_QF_HELPERS` gate; ATTN, RMSNORM, SILU_MUL, W8A16 fold untouched.
* No epilogue change (`mm_tile`), no test bound widened, no ABI bump.
* (iii) full-precision quantiser and the v75 STAT re-baseline it implies: follow-up after the flip.
* Ledger ⑮ (upstream qf32 drift) stays with #26; the tie row on silicon stays indirect.
* #25's device matrix rides in slot 4 only on the supervisor's say-so; otherwise not in this handoff.
