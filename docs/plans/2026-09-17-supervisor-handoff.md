# Supervisor handoff — 2026-09-17 remote cycle → workstation

For the `hexagon-supervisor` (and the orchestrator running `/hexagon-cycle`) on the
Linux workstation. Contract: `docs/plans/0000-agent-system-and-env.md`. User-facing
summary of the same state: `docs/plans/2026-09-17-handover.md`; tracking issue #42.
Written by the remote session `session_01AETWMAyrqeDgFwAHv5es83`, which had no SDK,
container, model or device and could push only one branch.

## 0. Environment facts for the workstation session

* Linux PC, Hexagon SDK **6.4.0.1** native at `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1`
  (`hexagon-clang 19.0.04`, `DEFAULT_TOOLS_VARIANT=toolv19`), NDK `/opt/android-ndk-r26d`,
  API 31, phone on USB (single device, no serial needed). **No `tools/docker/run.sh`**:
  `source $SDK/setup_sdk_env.source; export ANDROID_NDK=/opt/android-ndk-r26d` and call
  `tools/hexagon/*.sh` directly — every gate in `.claude/skills/hexagon-gates` works that
  way, drop the `tools/docker/run.sh` prefix.
* The x86 reference build works with the system gcc; the workstation `hexagon_ref_run` gives
  **41.5466 / 164** on `t512_23.i32` (container clang: 41.2365 / 161) and exactly
  **33.0195 / 184** on the P4 prompt `t512.i32` (HEXAGON.md §5.1) — quote which build a
  reference number comes from.
* GitHub: the claude.ai GitHub MCP **can write** (issue comments, labels, issue create, PR
  close were all done from the remote session). Contract §4's "writes need gh" note is
  stale; either tool is fine. `git push` from an agent session may be limited to its own
  branch (403 on others) — check before planning pushes.
* Agent worktrees (`isolation: worktree`) start on `main`, not on the current branch: the
  first instruction to every subagent must be `git checkout -B <branch> <base>`.

## 1. The integration branch and where to cut it

Everything from this cycle is on **`claude/issue-23-device-measurement-crvngx`**, 40
commits on `main` @ `a7ea056`, in this order (`git log --reverse origin/main..`):

| # | range | segment | goes to |
|---|---|---|---|
| 1–18 | `e4072273..f66bf3b1` | #34 history replacement: 15 squashed commits (all `[component]`, signed; PRs #29–#31 folded in; `[todo]`→`[docs]`; m12/n2 wording fixes folded in) + `[htp]` clang-format of the 11 ggml imports + `sim_model.c`/`test_attn.c` + `[hexagon]` `/**` openers and python `@brief` tags + `[CI]` static.check/clang build on PRs into `hvx_impl` (#28) | **replaces `hvx_impl`** |
| 19–26 | `9a3bebdf..ee136691` | the 8 commits of `hvx/23-sdk64-baseline` rebased (cherry-picked cleanly) | PR #23 |
| 27–28 | `57017920`, `895b6684` | #23 read-back (benchmark rows, §7 rule 1, §9, ledger ④/⑭) and the §5.2/§9/ledger correction on where the v79 sim failures live | PR #23 |
| 29 | `b5d66aa7` | #24 plan | PR #24 |
| 30–34 | `e025994a..5cfb4c6d` | #33 plan + implementation + the `static_assert` fix | PR #33 (depends on 19–28 for the HEXAGON.md §5.2 paragraph) |
| 35–39 | `10f1186f..268fbb73` | #24 implementation + handoff + doxygen fixup | PR #24 after `state:measured` (depends on 30–34) |
| 40 | `84230102` | user handover doc | any |

Tree facts: segment 1 alone has tree == old `hvx_impl` @ `c33b2c9` + hygiene only (verified
by `git diff` before the hygiene commits). The branches `hvx/history-clean`,
`hvx/23-sdk64-baseline`, `hvx/33-dsp-bounds` are superseded by this branch;
`hvx/fix-container-sdk-env`, `hvx/w8cx-from-hf` are merged (PRs #30, #31).

## 2. Decisions taken this cycle (do not re-litigate)

* #34: replace `hvx_impl` history — **yes** (user). Imported ggml files are reformatted and
  doxygen-tagged in place, not excluded. `ponytail:` markers → `Follow-up:`; `v2 op-list` →
  `ABI v4`; `Task N` wording removed; the two Korean comments translated.
* #28: workflow edit approved (user); implemented as commit 18.
* #35: rewritten in place (v79 skel is correct and 10–27 % faster on silicon; the four v79
  sim failures are in qf-only code, `quant_row` and the W8A8 `mm_tile` epilogue — neither
  file has an `__HVX_ARCH__` branch). Standing rule in §7 rule 1: a v79 change needs both
  gates. `#error` on IEEE helpers is dropped.
* #32: (b) — `make_w8cx_bin.py` is the producer; hvx_m3 commits are not cherry-picked;
  md5 question answered (`7562313b…`, `.hexw` byte-identical). `state:planned` p2.
* #24: premise corrected (host-wall − DSP gap 0.5 / 2.9 / 2.2 ms, not 40 ms); p0 → p1;
  ABI stays v4; rpcmem logits buffer + `FASTRPC_MAP_STATIC` probe; top-k / DSP argmax are
  contingencies with a trigger (transport ≥ 2 ms after rpcmem+static).
* #33: two holes beyond the plan were fixed in the same commits (ATTN `in1`/`in2` bounds;
  `d.m != 0` refused on every kind but `MATMUL_LOGITS`). Gates 2–4 PASS on the workstation.
* Error encoding: FastRPC returns DSP-originated AEE codes offset by `0x80000400`
  (`0x8000040E` = `AEE_EBADPARM`); `hexagon_runner.cpp` asserts exactly that.
* PR #39 was accidental (whole backend against `main`); closed.

## 3. Board (open `hexagon` issues) and what the supervisor does with each

| issue | label | supervisor action next cycle |
|---|---|---|
| #24 host logits | `state:needs-measurement` p1 | when `docs/measurements/24-host-logits.md` is filled: read back per `hexagon-handoff`; apply the plan's §3 decision rule (ship rpcmem, or static if ≥ 0.1 ms better; top-k only if static ≥ 2 ms over the 8-float call; gap > 5 ms with transport < 1 ms → it is #41, not #24); copy G2/G4 rows to HEXAGON_BENCHMARK.md; close ledger ⑫; `state:in-progress` → implementer finishes §6 docs (`<fill from handoff>` placeholders) and opens the PR |
| #33 DSP bounds | `state:review` p0 | PR not yet open (see §4). STAT bit-identity read-off deferred: check `profile_prefill_acc 0.077216/98.2637`, `graph_prefill 0.0245416/9.3795`, `graph_decode 0.0202219/23.975` from the next `logs/hexagon/` push before merge |
| #23 SDK 6.4 baseline | `state:measured` p0 | done except the PR (segments 19–28). Close on merge; ledger ⑭ already closed |
| #41 idle-gap DSP clock | `state:needs-plan` p1 | the #24 handoff's `--eval` `pcycles_per_us` line is its opening number: record it in the issue body when #24 is measured |
| #35 v79 skel | `state:needs-plan` p1 | next for the planner after #24/#33 (order: #35 → #26 → #25). Planner must first probe the hypothesis (v79 qf32→sf rounding ≠ v75's Von Neumann jam) with a two-line sim test |
| #32 W8_CX producer | `state:planned` p2 | next small implementer job (2 docs + `qwen3_w8cx_bin.{h,cpp}` @brief + wizard md5 warning); plan text is the issue comment, copy it to `docs/plans/32-w8cx-producer.md` |
| #40 causal_lm hex error text | `state:needs-plan` p2 | outside the subtree; one-line fix, needs user OK (core app) |
| #36 single op-shape source | `state:needs-plan` p1 | sequence after #33 merges (same files); #33's plan §2 shape list is its spec |
| #37 lifecycle polish | `state:needs-plan` p2 | body needs restating: `Qwen3W8cxBin` already munmaps; "cursor overflow" has no such identifier — ask for file:line |
| #38 hexcfg IO into core | `state:needs-plan` p2 | registry half touches core (`nntrainer/engine.h`) → user OK at plan time |
| #25, #26 | `state:needs-plan` p1 | unchanged; #25's 4-variant device matrix could ride in #35's handoff |
| #28, #34 | `needs-user` | close both once `hvx_impl` is replaced |
| #42 handover | `needs-user` p0 | close when A–D below are done |

## 4. Merge plan to execute (user + supervisor, in order)

1. **User**: run the #24 handoff (`docs/measurements/24-host-logits.md`, ≈ 25 min; harnesses
   rebuilt with `build_host_test.sh`, skel reused from #33 gate 4). Push the filled file and
   `logs/hexagon/`. Label `state:measured`.
2. **User**: replace `hvx_impl` with segment 1:
   `git push --force-with-lease=hvx_impl:c33b2c9 origin f66bf3b1:hvx_impl`; delete
   `hvx/fix-container-sdk-env hvx/w8cx-from-hf hvx/history-clean`; close #34 and #28.
   Sanity: the next PR into `hvx_impl` must pass `Static checks` and `cpp-linter` (PR #39
   failed both on the old tree).
3. **Supervisor/orchestrator**: cut PRs from the integration branch, base `hvx_impl`:
   * `hvx/23-sdk64-baseline-v2` = commits 19–28 (`git checkout -b … 895b6684`), PR #23.
   * after #23 merges: `hvx/33-dsp-bounds-v2` = commits 30–34 rebased onto `hvx_impl`
     (drop 29 into the #24 branch), PR #33 with one `<details>` per commit
     (`.github/PULL_REQUEST_TEMPLATE.md`). Merge only after the deferred sim STAT read-off.
   * after #33 merges and #24 is measured: `hvx/24-host-logits-v2` = commits 29, 35–39 +
     the docs commit that fills the `<fill from handoff>` placeholders, PR #24.
   `git rebase --onto hvx_impl <prev segment tip> <branch>` is expected to be conflict-free
   except HEXAGON.md §5.2, which the remote session merged once by hand (keep the #23
   toolchain paragraph and insert #33's "graph also carries the negatives" sentence after
   "bit-exact by construction").
4. **Deferred simulator gate** on the integration branch (or on each PR branch), before
   any of the three PRs merges: `HEX_ARCH=v75 ./tools/hexagon/build_sim_test.sh`;
   `run_sim_test.sh profile acc` → STAT `0.077216/98.2637`; the 13 tests → 13 PASS,
   `quant_generic 0/65536`, `quant16_generic 167/25600`, graph STATs above. #24 touches no
   DSP file, so any STAT movement means a mistake in the host-only claim.
5. **Guide writer**: after the first merge into the new `hvx_impl`, refresh
   `docs/backend_guide/hexagon-guide/` (contract §6 step 4).

## 5. Verified / not verified by the remote session

* Verified: segment 1's reformatted tree compiles and passes sim 13/13 + `profile acc` +
  skel/harness build (gates 2–4 of #33 were run by the user on this exact branch at
  `5cfb4c6d`); x86 gate (`LOWER_TEST PASS`, `oplist header check: PASS`) after every
  integration step; CI scripts (`doxygen-tag.sh` advanced, `newline.sh`, `executable.sh`,
  `signed-off-by.sh`, `nobody.sh`, `timestamp.sh`), `clang-format-14 --dry-run` and
  `git diff --check` clean on every changed file; every commit `[component]` + signed-off.
* Not verified: #24's `hexagon_runner.cpp` / harness compile with the real SDK headers
  (stubbed; `AEE_EALREADY` and `fastrpc_mmap` are already used by the pre-existing `init()`,
  so the only new symbol is the `FASTRPC_MAP_STATIC` enumerator, probed by the build
  scripts); `hexagon_ref_run --eval` unchanged-PPL check for #33/#24 (no model here); the
  STAT bit-identity read-off for #33 (logs were not transferable).
* Contract deviations to record: #23's device binaries were a workstation SDK 6.4.0.1 build,
  not the container's (ledger ⑭ notes it); the remote session acted as supervisor,
  orchestrator and integrator in one, and its branch is an integration branch rather than
  per-issue `hvx/*` branches — §4 above restores the per-issue layout.

## 6. Rules learned this cycle (candidates for the contract / skills)

* Handoff results must carry the skel md5 *from a log*: add a `md5sum` echo of the local
  skel to `run_e2e_test.sh` before the next handoff (variant A of #23 has only an inferred
  identity).
* `run_sim_test.sh` must pick the toolchain image by `DEFAULT_TOOLS_VARIANT`, never
  lexically (done, commit 20).
* Reference PPLs depend on the `hexagon_ref_run` build (clang vs gcc, 0.75 % on the new
  prompt); the P4 prompt is the accuracy anchor until one build is canonical.
* A v79 change is judged by both gates (§7 rule 1); simulator ±1 LSB failures did not
  predict device failure, and a device pass does not clear a failing sim test.
