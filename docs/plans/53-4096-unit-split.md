# 53 — 4096-context offset on `R3CY10WM83Y`: run #23 B's exact v79 binary on this unit to split unit/session from code

Issue: dlwlzzero/nntrainer#53 (`prio:p1`, filed 2026-09-18 from the #24 and #35 measurements).
Contract: `docs/plans/0000-agent-system-and-env.md`. Branch: `hvx/53-4096-unit-split` from
`hvx_impl` @ `202278a4`. **No source change. One device handoff, two skels, four 4096 runs.**
Independent of #25 (its own `hvx/25-*` branch; nothing here touches source, so no conflict).

## 1. Goal and gate

The issue's acceptance, made measurable. Every number is the median decode DSP Mcyc/step
(`E2E decode ... median_pcycles`) and the prefill tok/s (4096 ÷ Σ `us` of the 32 `n=128` steps /
1e6) of a `run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64`
run on `R3CY10WM83Y`, v79 skel, current harness. Skel **S1** = #23 B's binary
(`aba0126e848f9d993ef72ecfbf6fcab7`, `HEX_ARCH=v79` @ `e086248a`, or its rebuild — §4 step 1);
skel **S2** = `HEX_ARCH=v79` @ `202278a4` (`hvx_impl` HEAD, post-#35).

Two orders, each after ≥ 10 min of phone idle: **O1** = S1 then S2, **O2** = S2 then S1. HEXAGON.md
§7 rule 9(d) says the first run after an idle/push is cold, so the reading is *slot-matched*:
first-slot pair `S1(O1)` vs `S2(O2)`, second-slot pair `S2(O1)` vs `S1(O2)`. Let
`Δ = mean over the two slots of (S2 − S1) / S1` in Mcyc.

| outcome | gate (Mcyc, both slots) | consequence (issue) |
|---|---|---|
| **Unit / session** | `|Δ| ≤ 2 %` in both slots, i.e. S1 reads ≈ 186–188 like S2 | #23's 4096 cells (177.6 Mcyc / 34.4 tok/s, unit `R3CY205ZMND`) are re-baselined to this unit from the four readings; the cross-unit 4096 band is set from them; the benchmark's 4096 goal row cites the within-unit number only; rule 9(e) closes |
| **Code / build** | S1 within 2 % of #23 B (≤ 181.2 Mcyc) **and** S2 ≥ 4 % above S1 in both slots | a real regression between `e086248a` and `hvx_impl`; follow-up issue: v79 skel at `1bbd8d53` (splits #43–#46 from #35 + #50), then per-commit; fix or revert |
| **Inconclusive** | anything else (e.g. S1 ≈ 178 and S2 within 2–4 % of it, or the two slots disagree in sign) | one more O1 + O2 pair on a second day before any re-baseline; the 4096 cell stays "within one unit and session only" (rule 9(e)) |

Prefill tok/s is recorded and read with the same slot rule (band ±3 %) but does not decide: it is
host-wall based and #24 showed this unit's host clock moves independently of DSP cycles.
Accuracy: one `--eval` on `t4096.i32` per skel (S1 expected 1.5687 / 3758 = #23 B; S2 1.5623 / 3759
= #35 B1), required by the numerical-gap rule, run *after* the four speed runs so it cannot heat
them.

No ABI change (`NNTR_HTP_ABI_VERSION` is `4u` at `e086248a` and at HEAD,
`nntrainer/tensor/hexagon/htp/nntr_htp_common.h:18-19`), so the HEAD harness and the P4
`qwen3_full4k` image (`e5c92fad…` / `b0d8aca9…`) load S1 without modification; `lower_qwen3`,
`ref_graph_forward`, sim `graph`, `find_divergence.py`, `nntr_hexpack` are untouched.

## 2. Where it lives

Nothing under `nntrainer/`, `test/` or `tools/` changes. The deliverables are documents:

| path | role |
|---|---|
| `docs/measurements/53-4096-unit-split.md` (new) | the handoff the user runs (§4 step 2) |
| `docs/backend_guide/HEXAGON.md:956-972` (§7 rule 9(e)), `:1258-1300` (§8.2 #35 block), `:1617-1621` (§9 open item (a)) | resolved per the outcome (§6) |
| `docs/backend_guide/HEXAGON_BENCHMARK.md:38`, `:41`, `:44` (4096 rows), `:57` (decode @4096 goal row), history table `:71-72` | re-baseline or regression note (§6) |

What the handoff relies on, verified at HEAD:

* `tools/hexagon/run_e2e_test.sh:36-46` pushes the skel only when its md5 differs
  (`push_if_changed`), so alternating S1/S2 costs one 50 KB push each; the 600 MB image is
  compared by md5 and not re-pushed.
* `test/hexagon/hexagon_e2e_test.cpp:117-120` prints
  `E2E decode steps=63 median_us=… median_pcycles=… pcycles_per_us=…` (since #24) — the medians
  and the clock the table reads. The first-ten vs last-ten decode medians the issue asks for are
  **not** printed; the handoff computes them from the 63 `E2E step … n=1 pcycles=<c> …` lines
  (`hexagon_e2e_test.cpp:305`) with one awk line (§4 step 2), no harness change.
* `tools/hexagon/build_skel.sh:16` defaults to v79; the S1 rebuild passes `HEX_ARCH=v79`
  explicitly anyway. The link is not byte-reproducible (relocation order, #35 §2 / gates rung
  note), so a rebuilt S1 has a **different md5 at the same size (50,920 B on SDK 6.4.0.1)**; the
  handoff records whichever md5 ran and, for a rebuild, the `git rev-parse HEAD` of the worktree
  it came from.
* DSP source diff `e086248a → 202278a4` (`git diff -w`, non-comment lines): one
  `nntr_htp_token_ids_ok` call in `htp_graph.c` (O(n_tokens), init-time), #35's integer quantiser
  decode in `hvx/hvx-quant.h` (per-element), #50's op-extent table in `nntr_htp_common.h`
  (validator, init-time). None scales with context; that is why the issue expects outcome 1.

## 3. Design

**Chosen: two skels from one toolchain, four 4096 runs in mirrored order, slot-matched reading.**
Both skels come from the workstation's SDK 6.4.0.1 / hexagon-clang 19.0.04 — S1 is the stored
#23 B binary if `build_hexagon/skel/` still holds `aba0126e…` (it was built there), else a
rebuild of `e086248a` in a throw-away `git worktree` with the same SDK; S2 is built from
`202278a4` in the main worktree. Same harness (`202278a4`, static logits map), same image, same
token file, same unit. The only variable between S1 and S2 is the DSP source tree. Mirroring the
order and comparing like slots cancels the rule 9(d) cold-first-run effect instead of trying to
warm it away (a warm-up run would add heat to a 168 s sustained-load measurement whose thermal
state is one of the hypotheses).

**Rejected: three skels (add `1bbd8d53`) in the same sitting.** It would pre-empt the bisect in
outcome 2, but six 4096 runs plus idles is ~50 min wall, and if outcome 1 holds (the DSP diff
gives no reason to expect otherwise) the third skel is wasted. The bisect is the follow-up issue's
first step, filed only if outcome 2 is read.

**Also rejected: reading S2 against #35 B1's 186.8 / 188.0 instead of re-running it.** #35 B1 is
a different binary (`8d8cdb26…`, `305c51f1`, pre-#50) from a different session; the issue's
whole point is that cross-session 4096 cells on this unit are not comparable yet.

No kernel is written, so HEXAGON.md §7 rules 1–8 (qf-format ops, tie rows, NaN) are not engaged.

## 4. Steps

1. **Branch + handoff (implementer, container, ≈ 20 min, no device).** `git checkout -b
   hvx/53-4096-unit-split 202278a4`. Write `docs/measurements/53-4096-unit-split.md` from the
   `hexagon-handoff` template with:
   * Artifact table: S1 = `aba0126e848f9d993ef72ecfbf6fcab7` (50,920 B, `HEX_ARCH=v79` @
     `e086248a`, workstation SDK 6.4.0.1 — from `docs/measurements/23-sdk64-baseline.md`) with the
     fallback rule "if `md5sum build_hexagon/skel/libnntr_htp_skel.v79.so` (and any other copy
     under `build_hexagon/`) is not `aba0126e…`, rebuild: `git worktree add /tmp/nntr-23b e086248a
     && (cd /tmp/nntr-23b && HEX_ARCH=v79 ./tools/hexagon/build_skel.sh) && cp
     /tmp/nntr-23b/build_hexagon/skel/libnntr_htp_skel.so build_hexagon/skel/libnntr_htp_skel.v79-23b.so`,
     expect 50,920 B, write the md5 you get"; S2 = `HEX_ARCH=v79 ./tools/hexagon/build_skel.sh` @
     `202278a4` copied to `libnntr_htp_skel.v79-head.so` (expect ≈ 50,984 B = the container's 50,792 B at
     `4b39062f` + the 192 B SDK 6.4.0.1 delta seen in #35; write the md5); `hexagon_e2e_test` / `hexagon_rpc_test` from `./tools/hexagon/build_host_test.sh`
     @ `202278a4`; the P4 image and `t4096.i32` (`29684fbd372495912951cb3cc0d25e20`, #24 table).
     Record the container's own v79 skel md5 at `202278a4` (gates rung 4) as the Mac-side reference
     row, marked "not used on the device".
   * Steps block (one screen):
     ```
     git fetch && git checkout hvx/53-4096-unit-split
     source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/setup_sdk_env.source; export ANDROID_NDK=/opt/android-ndk-r26d
     md5sum build_hexagon/skel/*.so                       # S1 present? else the rebuild line above
     HEX_ARCH=v79 ./tools/hexagon/build_skel.sh && cp build_hexagon/skel/libnntr_htp_skel.so build_hexagon/skel/libnntr_htp_skel.v79-head.so
     ./tools/hexagon/build_host_test.sh && md5sum build_hexagon/skel/libnntr_htp_skel.v79-23b.so build_hexagon/skel/libnntr_htp_skel.v79-head.so build_hexagon/host/hexagon_e2e_test
     ./tools/hexagon/run_device_test.sh                   # RPC_TEST PASS with S2 (loads the new harness)
     # -- context snapshot before each 4096 run (optional, 1 line): adb shell dumpsys battery | grep -E 'level|status|temperature'
     # O1: phone idle >= 10 min (screen off, USB attached, nothing running), then
     cp build_hexagon/skel/libnntr_htp_skel.v79-23b.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so
     ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64     # S1(O1)
     cp build_hexagon/skel/libnntr_htp_skel.v79-head.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so
     ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64     # S2(O1)
     # O2: idle >= 10 min again, then S2 first, S1 second (same four lines, swapped)
     # accuracy, after the speed runs: --eval on t4096.i32 with S2, then S1
     adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so                                            # last skel that ran
     # first-ten vs last-ten decode medians, per log:
     awk '/^E2E step/ && / n=1 /{for(i=1;i<=NF;i++) if($i ~ /^pcycles=/){split($i,a,"=");c[++n]=a[2]}} END{asort(c);m=int(n/2);print "all",n,"median",c[m+1]; }' logs/hexagon/e2e_<stamp>.log
     awk '/^E2E step/ && / n=1 /{for(i=1;i<=NF;i++) if($i ~ /^pcycles=/){split($i,a,"=");v[++n]=a[2]}} END{for(k=1;k<=10;k++){f[k]=v[k];l[k]=v[n-10+k]} asort(f);asort(l);print "first10",(f[5]+f[6])/2,"last10",(l[5]+l[6])/2}' logs/hexagon/e2e_<stamp>.log
     ```
     (gawk is on the workstation; if `asort` is missing, `sort -n` the extracted column instead.)
   * Expected lines: `E2E init ok weights=599180800 …`, 32 × `n=128` steps, 63 × `n=1` steps,
     `E2E decode steps=63 median_us=… median_pcycles=… pcycles_per_us=…`, `E2E wall_ms …`;
     `--eval`: `E2E ppl … steps 4095 top1 …`.
   * Result table, one row per run in execution order: order, slot, skel md5 (from the `md5sum`
     before the run), wall-clock start, idle minutes before, prefill tok/s, `median_us`,
     `median_pcycles`, `pcycles_per_us`, first-ten / last-ten median Mcyc, battery temp (if taken),
     with the references in brackets: #23 B 4096 = 34.4 tok/s / 177.6 Mcyc (`R3CY205ZMND`,
     2026-09-17, malloc harness); #35 B1 = 32.1 / 32.7 tok/s, 188.0 / 186.8 Mcyc (this unit,
     2026-09-18). Accuracy rows: S1 vs 1.5687 / 3758, S2 vs 1.5623 / 3759.
   * The §1 verdict table copied in, so the user sees the outcome at the desk.
   * Estimated time: the four 4096 speed runs are ≈ 12 min device-active (the issue's "≤ 15 min"
     budget); the two 4096 `--eval` runs add ≈ 8 min; with the 2 × 10 min idles the sitting is
     ≈ 40 min wall. State both numbers at the top of the handoff.
   Gate: `tools/docker/run.sh ./tools/hexagon/build_skel.sh` at `202278a4` succeeds and its md5 is
   in the handoff (rung 4, v79); **no sim run** — no DSP bytes change on this branch (`git diff
   202278a4 --stat -- nntrainer/tensor/hexagon/htp` empty). Commit the handoff, push, label
   `state:needs-measurement`.
2. **Device (user, unavoidable).** Runs the handoff exactly as written; fills the table; commits on
   the branch; `state:measured`. Variants: 2 skels (S1, S2), 4096 only — the issue's decidable
   change is a goal-row check at one context, not a 512 sweep; no 512/1024 rows are taken because
   both sessions already showed them in band and they would add heat before the 4096 runs.
3. **Read-back (supervisor, next cycle).** Apply the §1 table. Outcome 1 → §6 edits, close #53.
   Outcome 2 → file the bisect issue (v79 skel @ `1bbd8d53`, then per commit between `1bbd8d53`
   and `202278a4`; candidates in order: #35 `813a6b7a`, #50 `e77a2275`, #33 `7c8efb61`), keep
   rule 9(e) as is, close #53 as "code". Outcome 3 → re-issue step 2 once (second day), then decide
   on eight readings.

## 5. Risks

* **The stored S1 is gone and the rebuild differs from `aba0126e…`.** #35's run rebuilt "all three
  skels" on the workstation on 2026-09-18, so `libnntr_htp_skel.v79.so` is most likely
  `8d8cdb26…` now. A rebuild at `e086248a` with the same SDK 6.4.0.1 gives the same code bytes
  (link order only — gates rung note), so the comparison is still "the #23 tree"; the handoff
  says so and records the md5 and the size. If the size is not 50,920 B the SDK on the
  workstation moved — record `hexagon-clang --version` and treat S1 as "the #23 tree on today's
  SDK", which still splits code from unit but not from toolchain (the issue's hypothesis (b)).
* **Cold-first-run confound (rule 9(d)).** Handled by mirrored order + slot matching (§3). If the
  two slots disagree by > 2 % for the *same* skel, that spread is itself the session effect and
  the reading is outcome 3, not 1.
* **The 10 min idle is not a controlled thermal state.** Battery temperature before each run is
  the one cheap observable; the handoff asks for it. A ~7 min idle recovered 0.7 % in #35, so the
  idle is expected to matter little; it is there so both orders start from the same state.
* **Harness differs from #23 B's** (static logits map since #24, `E2E decode` summary line).
  DSP Mcyc are unaffected by host memory kind (#24: three kinds within 70 µs); prefill tok/s is
  host-wall and may differ by the static map's ≤ 0.1 ms/step — irrelevant at 3 s per 128-token
  chunk. Not a risk for the Mcyc gate.
* **Stale artifacts.** `push_if_changed` compares md5 per run; the handoff has the user copy the
  `md5sum` printed before each run and the final `adb shell md5sum` read-back, as in #35.
* **Simulator says nothing here.** Bandwidth/DCVS effects are the hypothesis; the simulator does
  not model them (gap rule 1). The plan contains no simulator step by design.

## 6. Docs to update (after `state:measured`)

Outcome 1 (expected):
* `HEXAGON.md` §7 rule 9(e) (`:956-972`): replace "until #53 runs…" with the result — the 4096
  offset is the unit/session; the 4096 reference for `R3CY10WM83Y` is the mean of the four
  readings; cross-unit 4096 band = (S1 here vs #23 B there). Keep "compare 4096 within one unit".
* `HEXAGON.md` §8.2 #35 block (`:1284-1292`): the 4096 FAIL rows re-read as PASS against the
  within-unit reference; add the #53 four-run table with date, skel md5s, `pcycles_per_us`.
* `HEXAGON.md` §9 (`:1617-1621`): open item (a) closed.
* `HEXAGON_BENCHMARK.md:41`, `:44` (4096 rows): drop the "issue #53" caveats, cite the within-unit
  reference; `:38` (#23 v79 4096 row): annotate with the cross-unit band; `:57` (decode @4096
  goal row): "Now" cites the `R3CY10WM83Y` number only; history table: one 2026-09-xx row.
* Guide `06-performance.html` gets the four readings (guide-writer, next `--guide` cycle).

Outcome 2: rule 9(e) stays; §9 gains the bisect issue; benchmark 4096 rows keep the caveat with
the new issue number.
