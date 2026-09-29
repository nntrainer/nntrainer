# 41 — Hold the cDSP clock through host idle gaps: a session-lifetime `HAP_power` DCVS vote

Issue: dlwlzzero/nntrainer#41 (`prio:p1`). Contract: `docs/plans/0000-agent-system-and-env.md`.
Branch: `hvx/41-dsp-clock-vote` from `hvx_impl` @ `05d99341`. Planned 2026-09-18 from the issue body
and the three supervisor comments of 2026-09-18 (gate rephrased to the unit's own generation-mode
clock; #25 fill + #24 table (iii) as the "before" pair; unsigned-PD availability verified, not assumed;
power cost recorded with and without the vote; vote released at `nntr_htp_close`).

**Branch base, stated explicitly (supervisor bullet 4).** #25 is `state:needs-measurement` today
(`hvx/25-decode-prefetch` @ `d7c30bbc`, not merged), so this branch starts from `hvx_impl` HEAD.
Nothing here overlaps #25: #25 changes `hvx-matmul.c`, `dma/`, `htp_graph.c`; this plan changes
`executor.c`, the host harness and docs. The line the handoff reads — `E2E decode steps=… pcycles_per_us=…`
— is on `hvx_impl` since #24 (`test/hexagon/hexagon_e2e_test.cpp:106-121`); #25's `HTP_PROF_FARF`
per-call line is not needed. If #25 merges before the implementer starts, rebase onto it and use its
skel B as the "no vote" reference; otherwise the "no vote" reference is #35 B1 (the `hvx_impl` skel,
same unit) and #24 table (iii).

## 1. Goal and gate

Every number is read on **one named unit in one session** (HEXAGON.md §7 rule 9 (a)–(d)); the
serial goes next to every reading. `pcycles_per_us` is the harness's own ratio
(`median_pcycles / median_us` over the `n=1` steps), i.e. the DSP clock the host saw during that loop.

**The "before" pair on `R3CY10WM83Y` (not re-measured here, supervisor bullet 1):** `--eval` 1167.2 /
1177.6 (malloc / rpcmem, #24 table (iii)) against 1905–1915 in `--chunk 128 --steps 64` the same
minute — 0.61×. #25's H1 fill adds the same pair for the v79 prefetch skel (every generation run and
every `--eval` print the ratio). HEXAGON.md §8.2 "Host path" and §7 rule 9 record it.

| # | Acceptance (issue, as rephrased 2026-09-18) | Pass means |
|---|---|---|
| G1 | `--eval` clock within 5 % of the same session's generation-mode clock at 512, vote in place | V1 skel, `t512_23.i32`: `pcycles_per_us(--eval) ≥ 0.95 × pcycles_per_us(--chunk 128 --steps 64)` of the same session (today 0.61×). Both readings and the serial in one table row. |
| G1' | the vote holds independent of gap length (planner addition; makes G1 controlled rather than incidental to `log_softmax_at`'s ~10 ms) | V1, `--chunk 128 --steps 64 --gap-ms 10` and `--gap-ms 100` (new harness flag, §3): `pcycles_per_us ≥ 0.95 ×` the `--gap-ms 0` run of the same session. The DSP-side `nntr_htp: clk=<MHz>` FARF at each forward entry (V1/V2 carry `-DHTP_POWER_FARF`) reads ≥ 0.95 × the init-time value on every decode step — the clock *at the moment the RPC lands*, independent of the pcycles arithmetic. |
| G2 | the vote does not lower the busy clock (supervisor bullet 1) | V1 generation @512: `pcycles_per_us ≥ 0.98 ×` the no-vote reference of this unit (1905–1915 v75 #24; #35 B1 / #25 B v79 fill) and DSP Mcyc/step within ±2 % of #35 B1 61.2 (or #25 B pass 2 if rebased). 1024 / 4096 generation rows recorded for HEXAGON_BENCHMARK.md (the 4096 cell only within-session, rule 9 (e)). |
| G3 | PPL / top-1 unchanged | V1 and V2 `--eval t512_23.i32` = **41.4947 / 162** to the digit (v79 record on this unit, #35 B1) and the `top1=` sequence of the 63 generated steps byte-identical to #35 B1's / #25 B's log (`diff <(grep -o 'top1=[0-9]*' a.log) <(… b.log)`). The vote touches no kernel; anything else is a stale artifact. |
| G4 | app-path decode @512 within 10 % of the harness generation-mode number | `nntrainer_causallm` with `engine="htp"`, V1 skel, prompt = the detokenised 512-token prefix of `eval_23.txt` (`make_tokens.py --limit 512` writes it), `num_to_generate` 64: the app's `generation: 64 tokens, <ms>, <TPS>` line gives `TPS ≥ 0.90 × (1000 / median_us)` of the same session's V1 harness generation run. Same unit, same sitting. |
| G5 | power cost checked, not assumed (issue + supervisor bullet 3) | For N (no vote), V1 and V2: `dumpsys battery` temperature before / after the app run, and 10 samples of `/sys/class/power_supply/battery/current_now` during a `--gap-ms 1000 --steps 30` harness run (30 s with the session open and the DSP idle 97 % of the time — the multi-turn chat case). Recorded, not gated; the ship rule below reads it. |
| G6 | vote released at close; what happens without close | `sysMonApp getstate --getVotes 1 --q6 cdsp` (SDK `tools/utils/sysmon/sysMonApp`) shows the DCVS_V3 vote during a `--gap-ms 1000` run, no vote after the harness exits normally, and no vote after `kill -9` of a running harness (the PD dies with its client process; this is the documented FastRPC PD lifetime, verified here rather than assumed). |
| G7 | sim + compile gates | `profile acc` STAT bit-identical to the #35 v79 record (`max_abs=0.0659682 max_rel=92.7791`), 13/13 PASS with every STAT equal to that record; rung 4 default skel + variants + both harnesses; the app built with `-Denable-hexagon=true`. |

**Ship rule.** V1 (corner floor) is the default if G1–G4 pass. If V2 (DCVS performance mode, no
floor) also passes G1/G1' and its idle `current_now` is lower than V1's, V2 ships instead. If V1's
idle current is more than 2× N's, the doc says so in §1.2 and the supervisor decides whether the
default stays on (the knob is compile-time; a runtime toggle would be an IDL change and is not
planned until a number asks for it).

No ABI change: `NNTR_HTP_ABI_VERSION` (`nntrainer/tensor/hexagon/htp/nntr_htp_common.h:18`) stays;
no wire, image or `.hexcfg` field moves; `lower_qwen3`, `ref_graph_forward`, the sim `graph` test,
`find_divergence.py`, `nntr_hexpack` and HEXAGON.md §1.4 / §2 are untouched.

## 2. Where it lives

Verified against `hvx_impl` @ `05d99341` and the container-mounted SDK 6.4.0.2
(`~/Qualcomm/Hexagon_SDK/6.4.0.2`, mounted at `/opt/qcom/Hexagon_SDK`).

| File:line | Today | Change |
|---|---|---|
| `nntrainer/tensor/hexagon/htp/executor.c:23-33` (`struct session`) | mappings, op-list copy, graph | + `void *power_ctx; int power_voted;` |
| `executor.c:77-133` (`nntr_htp_init`) | validate, map, `htp_graph_init` | after `init ok` (`:130`): `session_power_vote(s)` once per session (`power_voted` guards re-init); its `rc`, the request that was made and `HAP_power_get(HAP_power_get_clk_Freq)` go to one `FARF(ALWAYS, "nntr_htp: power vote rc=%d corner=%u clk=%u MHz")`. A non-zero `rc` is logged, **not** an init failure (the session is still usable at DCVS's clock). |
| `executor.c:68-75` (`nntr_htp_close`) | destroy graph, unmap, free | before `free(s)`: `session_power_release(s)` — `HAP_power_destroy(ctx)` (weak `HAP_power_destroy_client`, `HAP_power.h:806-812`); on `AEE_EUNSUPPORTEDAPI` the SDK's own reset (`examples/profiling/src/rpcperf_imp.c:123-127`: a DCVS_v3 request with `set_core_params = TRUE` and zeroed corners, plus `set_bus_params` the same way); then `HAP_utils_destroy_context`. |
| `executor.c:135-163` (`nntr_htp_forward`) | op loop | `#ifdef HTP_POWER_FARF`: one `HAP_power_get(clk_Freq)` + `FARF(ALWAYS, "nntr_htp: clk=%u MHz pos=%u")` at entry (before the op loop, so the reading is the clock the RPC landed on). Off in the shipping build. |
| `executor.c` (new, ~40 lines) | — | `session_power_vote` / `session_power_release` behind `#ifndef HTP_POWER_VOTE` / `#define HTP_POWER_VOTE 1` and `#ifndef HTP_POWER_CORNER` / `#define HTP_POWER_CORNER HAP_DCVS_VCORNER_TURBO_PLUS`; `HTP_POWER_VOTE 0` compiles the calls out (the N control build for later A/Bs). |
| `tools/hexagon/build_skel.sh:44-49` | `SRCS` include `executor.c` | no change; `HAP_power.h` is under `$HEXAGON_SDK_ROOT/incs` (already `-isystem`), the symbols resolve from the DSP image at load like `HAP_compute_res_*` / `HAP_mmap` do. |
| `tools/hexagon/build_sim_test.sh:31-36` | `SRCS` = sim tests + `worker_pool.c htp_graph.c ops/ hvx/ dma/` | no change: `executor.c` is **not** in the simulator library, so no sim stub for `HAP_power_set` is needed and the sim runs are pure regression (G7). |
| `test/hexagon/hexagon_e2e_test.cpp:51-56` (`Opts`), `:58-66` (usage), `:140-160` (parse), `:311-334` (generation loop) | timed RPC, `argmax` outside the timed region | `--gap-ms N` (default 0): `std::this_thread::sleep_for` after each decode `fwd()` in the generation loop, outside the timed region like `argmax`; `print_decode_summary` gains ` gap_ms=<N>` at the end of the `E2E decode` line (parsers grep by key, `docs/measurements/25-decode-prefetch.md` reads `pcycles_per_us=`). Host-only. |
| `nntrainer/tensor/hexagon/host/hexagon_runner.cpp:35-55` | unsigned-PD request + `nntr_htp_open` | no change (the vote is DSP-side; §3 says why). |
| `Applications/CausalLM/hexagon/hexagon_backend.cpp:42-54` | runner create / init / static logits map | no change; `~HexagonRunner` (`hexagon_runner.cpp:67-74`) calls `nntr_htp_close`, which now releases the vote. |
| `docs/backend_guide/HEXAGON.md` §1.2 (`:132-138`), §5.4 (`:683-749`), §7 rule 9 (`:934-970`), §8.2 M5 bullet (`:1029-1033`) and "Host path" (`:1183-1250`), §9 (`:1577-1592`) | see §6 | |
| `docs/backend_guide/HEXAGON_BENCHMARK.md` | method paragraph, stage-1 W8A8 "Now" cell, log | see §6 |

**The unsigned-PD fact, verified in the container's SDK 6.4.0.2 docs (supervisor bullet 2).**
`docs/software/ipc/rpc.html`, "Unsigned PD available services": *Thread creation and thread
services; HVX contexts; **Clock frequency controls**; VTCM; Cache operations; Map HLOS memory…*;
"Unsigned PD limitations" lists UBWCDMA / camera streamer, QuRT timers and L2 cache locking — not
power. `docs/software/system_integration.html`, "Unsigned PD services and limitations": *…Cache
management operations; **Clock and power management**; VTCM allocation and usage.* The SDK's own
`examples/profiling` (default `-U 1`, unsigned PD) votes DCVS_v3 core + bus corners exactly this way
(`src/rpcperf_imp.c:100-121`). And HEXAGON.md §8.2 (M5, `:1029-1033`) records that an earlier
`HAP_power` vote from this skel in this unsigned PD was **accepted (rc 0)** — it was dropped because
the harness had no idle gaps, which is the case #41 now creates on purpose. So `HAP_power_set` is
documented as available and was accepted on this device; whether it is *honoured* during a gap is
what G1/G1' measure, and the fallback if it is not (supervisor bullet 2) is a host-side vote through
the session context — recorded as the rejected alternative in §3, to be reopened only on a failed G1.

## 3. Design

**Chosen: one DSP-side DCVS_v3 vote per session, in `nntr_htp_init`, released in `nntr_htp_close`.**

```c
/* executor.c — sketch; HAP_power.h:353-378 (payload), :630-643 (init helper), :196-227 (corners) */
static void session_power_vote(struct session *s) {
#if HTP_POWER_VOTE
  HAP_power_request_t req;
  int rc;
  s->power_ctx = HAP_utils_create_context();           /* HAP_power.h:824 */
  memset(&req, 0, sizeof req);
  req.type = HAP_power_set_apptype;
  req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;        /* :162 */
  (void)HAP_power_set(s->power_ctx, &req);
  memset(&req, 0, sizeof req);
  req.type = HAP_power_set_DCVS_v3;
  req.dcvs_v3.set_dcvs_enable = TRUE; req.dcvs_v3.dcvs_enable = TRUE;
#if HTP_POWER_VOTE == 1   /* floor: DCVS may go up, never below the corner */
  req.dcvs_v3.dcvs_option = HAP_DCVS_V2_ADJUST_ONLY_UP;               /* :335 */
  req.dcvs_v3.set_core_params = TRUE;
  req.dcvs_v3.core_params.target_corner = HTP_POWER_CORNER;
  req.dcvs_v3.core_params.min_corner = HTP_POWER_CORNER;
  req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.set_bus_params = TRUE;  /* same three corners */
#else                     /* 2: DCVS decides, with performance thresholds */
  req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;             /* :338 */
#endif
  req.dcvs_v3.set_latency = TRUE; req.dcvs_v3.latency = 40;           /* µs; SDK system_optimizations: "40 µs sleep latency" */
  rc = HAP_power_set(s->power_ctx, &req);
  s->power_voted = 1;
  /* FARF rc + HAP_power_get(HAP_power_get_clk_Freq) */
#endif
}
```

Why this shape:

* *Session lifetime, DSP side.* The idle gap is between RPCs, so the vote must outlive a call; the
  only DSP object that outlives calls is the session (`struct session`, the `remote_handle64`). A
  vote made inside `forward()` would be too late by construction (the clock is already down when
  the RPC lands and DCVS ramps in 1–2 ms, `HAP_power.h:339-350`).
* *Floor, not pin.* `ADJUST_ONLY_UP` with `min = target = TURBO_PLUS`, `max = MAX` keeps DCVS free
  to go higher (the unit's top corner is what the generation loop already reaches: 1.91 GHz on
  `R3CY10WM83Y`, 2.05 on `R3CY205ZMND`, rule 9) and forbids the drop that the gap triggers. Pinning
  `min = max = MAX` (the SDK example) would also work but hides G2's question (does the vote change
  the busy clock) behind a fixed value. `TURBO_PLUS` (= `TURBO_L1`, `HAP_power.h:287`) is the SDK
  optimisation guide's recommendation ("vote for TURBO or TURBO_L1, and a 40 µs DSP sleep latency");
  `HTP_POWER_CORNER` lets a later handoff try `TURBO_L3` if the FARF clock reads below the
  generation-loop value.
* *Sleep latency 40 µs, LPM untouched.* The latency vote is what removes the wake-up cost of the
  next RPC; `set_sleep_disable` is left FALSE (the header, `:28-48`, says regular users should not
  force LPM levels — the latency vote already selects the mode).
* *V2 exists to price the floor.* `PERFORMANCE_MODE` alone lowers DCVS's ramp-down thresholds
  without a corner floor; if it holds through a 10 ms gap it is the cheaper vote, and G5 says by how
  much. If it does not hold through 100 ms (`--gap-ms 100`) that is a fact for §1.2, not a failure.
* *Release.* `HAP_power_destroy` (weak) at close; if the image lacks it, the SDK's own reset request.
  Without a close — the app killed, the host process crashing — the unsigned PD is torn down with
  its client process and every HAP client in it (this is what makes unsigned PDs "extensions to
  their CPU client processes", system_integration.html); G6 verifies it with `sysMonApp getstate
  --getVotes 1` after a `kill -9` instead of asserting it. What the plan **does** commit to in
  writing (§1.2): while an `engine="htp"` process holds its session open — a chat loop waiting for
  the user — the vote is held too; that is the cost G5 measures, and the doc states it.
* *No kernel change.* HEXAGON.md §7 rules (qf-format ops, tie rows, NaN) are not touched; the sim
  STATs must be bit-identical (G7) because no DSP arithmetic moved.

**Rejected: a host-side vote through a new IDL method (`set_power(in uint32 mode)` or an `init`
flag).** It costs an ABI bump (`NNTR_HTP_ABI_VERSION` v5, `.hexcfg` rejection of old images, §1.4
edits, `lower_qwen3` / `ref_graph_forward` / sim `graph` / `find_divergence.py` / `nntr_hexpack` all
move with it) to carry one bit whose only sensible value today is "on for the session"; the host
has nothing to add that the DSP does not already know at `init`. It becomes the fallback only if G1
fails with rc 0 (the vote accepted but not honoured in this PD), in which case the host would have
to vote through a different mechanism anyway (FastRPC QoS / a privileged client), which is a
`needs-user` question, not a planning one. **Also rejected (issue):** a low-priority spin worker to
keep the DSP "busy" — burns a HVX-unit's worth of power to fake load, and the SDK documents the
proper lever as available in unsigned PDs. **Deferred, filed as a follow-up candidate:** host-side
FastRPC QoS (`remote_handle64_control(DSPRPC_CONTROL_QOS, PM_QOS, 100 µs)`, system_optimizations.html)
— it shortens the CPU-side wake latency of the RPC return, not the DSP clock; host-only, orthogonal,
and not mixed into this handoff so the table answers one question.

## 4. Steps

Each step ends in a gate from `.claude/skills/hexagon-gates`. Sim budget (contract §7): the change
is under `htp/`, so rungs 2 and 3 run **once**, right before the PR (step 6), never per commit.

1. **Harness `--gap-ms` (host-only).** `hexagon_e2e_test.cpp`: `Opts.gap_ms`, usage, parse, sleep
   after each decode `fwd()` in the generation loop (`:326-331`; not in `--eval`, whose gap is the
   real `log_softmax_at`), ` gap_ms=%u` appended to the `E2E decode` line. Gate: rung 0 (format)
   and rung 4 host build; `--gap-ms 0` output identical to today's except the suffix.
2. **The vote.** `executor.c` as in §2/§3: `session_power_vote` / `session_power_release`, the
   `power vote rc=… clk=…` FARF at init, the `HTP_POWER_FARF` per-forward clock line, `HTP_POWER_VOTE`
   0/1/2 and `HTP_POWER_CORNER` knobs (`#ifndef` guards, same pattern as `HTP_FORCE_QF_HELPERS`,
   `hvx/hvx-base.h:168-173`). `rc != 0` logs and continues. Gate: rung 0; rung 4
   `build_skel.sh` (default, `-Wall -Werror`: `HAP_power.h` pulls `<stdlib.h>`/`<string.h>`, fine)
   plus `HEX_EXTRA_CFLAGS=-DHTP_POWER_VOTE=0` and `=2` to prove all three branches compile; the
   `HTP_POWER_VOTE=0` skel's 11 DSP **object** md5s equal the base tree's except `executor.o`
   (gates skill, rung 2 bullet) — documents that nothing but the session glue changed.
3. **rung 1** (x86 reference): `build_host_x86.sh`, `test_lowering` → `LOWER_TEST PASS`,
   `test_w8cx_bin` → `W8CX_BIN_TEST PASS`, oplist header test. No packer/lowering change, so no
   image regeneration; the reference PPL is not expected to move and is not re-run.
4. **App build in the container** (rung 4b, first time an app binary is part of a handoff):
   `tools/package_android.sh . -Denable-hexagon=true -Dhexagon-sdk-root=/opt/qcom/Hexagon_SDK/<ver>`
   then `Applications/CausalLM/build_android.sh --cache` (HEXAGON.md §5.4 — without `--cache` the
   app script rebuilds nntrainer without the backend). Artifacts: `jni/libs/arm64-v8a/nntrainer_causallm`,
   `libcausallm_core.so`, `libnntrainer.so`, `libccapi-nntrainer.so`; md5 each. If the container
   cannot build the app (NDK r26d is baked in, but the app script has not been run there before),
   the handoff says so and the user builds it on the workstation as #24 did for the harnesses —
   state which in the artifact table.
5. **Handoff doc** `docs/measurements/41-dsp-clock-vote.md` (hexagon-handoff template), ≤ 20 min,
   **unit serial next to every reading**, with-vote rows only for the harness (supervisor bullet 1).
   Skels (3, all v79, `build_skel.sh` @ the handoff commit):
   * **V1** `-DHTP_POWER_FARF` (floor vote, default corner) — G1, G1', G2, G3, G4, G5, G6;
   * **V2** `-DHTP_POWER_FARF -DHTP_POWER_VOTE=2` (performance mode, no floor) — G1, G1', G3, G5;
   * **N** the base tree's default skel already on the device (#35 B1 `8d8cdb26…` / 50,920 B, or
     #25 B if rebased) — app-path no-vote row and the G5 no-vote current only; its harness numbers
     are the #24 / #25 references. (A fourth, `-DHTP_POWER_CORNER=HAP_DCVS_VCORNER_TURBO_L3`, is
     built and listed but run only if V1's init FARF clock reads < 0.95 × the generation-loop clock.)
   Steps, in this order (warm-up rule 9 (d): the first e2e run after a push is cold, so V1's
   gen-512 is run first and again last; the second is the record):
   ```
   adb devices                                             # serial into the header
   ./tools/hexagon/run_device_test.sh                      # RPC_TEST PASS with V1 (also warms the DSP)
   run V1 --tokens /tmp/t512_23.i32 --chunk 128 --steps 64                 # cold, discarded
   run V1 --tokens /tmp/t512_23.i32 --eval                                 # G1 numerator
   run V1 --tokens /tmp/t512_23.i32 --chunk 128 --steps 64 --gap-ms 10     # G1'
   run V1 --tokens /tmp/t512_23.i32 --chunk 128 --steps 64 --gap-ms 100    # G1'
   run V2 --tokens /tmp/t512_23.i32 --chunk 128 --steps 64 ; run V2 … --eval ; run V2 … --gap-ms 100
   # G5/G6: idle window with the session open, sysMonApp votes + current_now sampled from a 2nd shell
   for v in N V1 V2: run $v … --chunk 128 --steps 30 --gap-ms 1000   (≈ 35 s each; sample every 3 s)
   kill -9 a V1 --gap-ms 1000 run mid-way → sysMonApp getstate --getVotes 1 --q6 cdsp   # G6
   # app path (G4/G5): dumpsys battery before/after each; prompt = t512_23.txt contents
   nntrainer_causallm <model_dir> "$(cat t512_23.txt)"  with N, then with V1   (ADSP_LIBRARY_PATH per §5.4)
   run V1 --tokens /tmp/t1024.i32 --chunk 128 --steps 64 ; run V1 /tmp/qwen3_full4k --tokens /tmp/t4096.i32 …   # G2 rows
   run V1 --tokens /tmp/t512_23.i32 --chunk 128 --steps 64                 # G1 denominator / G2, the record
   adb shell md5sum …libnntr_htp_skel.so …hexagon_e2e_test                 # stale-artifact check
   ```
   Expected lines: `nntr_htp: power vote rc=0 corner=<n> clk=<MHz>` in `device_farf_*.log` right
   after `init ok` (rc ≠ 0 → note it; G1 then answers whether the vote was honoured anyway);
   63 × `nntr_htp: clk=<MHz> pos=<p>` per V1/V2 run; `E2E decode … pcycles_per_us=<r> gap_ms=<g>`.
   Result table columns: variant, mode (gen / eval / gap-ms), skel md5 from `run()`, `pcycles_per_us`,
   FARF clk min / median over the decode steps, DSP Mcyc/step, host ms, tok/s, PPL / top-1, serial,
   log stamp; a second table for G5 (temperature before/after, `current_now` median during the idle
   window, N / V1 / V2) and G6 (votes shown: during / after exit / after kill). Set
   `state:needs-measurement`. **This is the unavoidable device step**: the simulator has no DCVS.
6. **Pre-PR simulator gates** (rungs 2–3, once, v79, ≈ 25 min under Rosetta): `build_sim_test.sh`,
   `run_sim_test.sh profile acc` → `SIM_TEST profile PASS`, STAT bit-identical to the #35 v79 record;
   the 13 tests → 13 × PASS with STATs equal to that record (`quant_generic 0/65536`,
   `quant16_generic 184/25600`, `graph_decode 0.0197323/23.2374`, …). Anything that moved is a
   stale build (the sim library does not even contain `executor.c`). Record the log path in the
   handoff's "Simulator record" section as #25 did.
7. **After `state:measured`:** apply the ship rule (§1), set the default `HTP_POWER_VOTE` (1 or 2)
   and `HTP_POWER_CORNER` accordingly, write the docs of §6, open the PR into `hvx_impl` with the
   rung-4 md5s and "DSP bytes changed: `executor.c` only (session glue), sim STATs bit-identical".

## 5. Risks

1. **Accepted but not honoured (rc 0, clock still drops).** The M5 vote returned rc 0 with no
   visible effect because there was no gap to see; G1/G1' with `--gap-ms` are the first runs that
   can tell "accepted" from "honoured". Visible as: `power vote rc=0` yet `--eval pcycles_per_us`
   ≈ 1170 and the per-forward `clk=` line ≈ 1.1 GHz. Then the plan's fallback is the host-side
   route (§3), which is `needs-user` (privileged client / QoS) — the issue is re-planned, not forced.
2. **Honoured for the core clock but the gap is DDR, not clock.** If `clk=` reads 1.9 GHz on every
   step yet `pcycles_per_us` still lags, the pcycles are being spent waiting on a downclocked bus
   (the vote covers `bus_params` too, but an unhonoured bus vote would look exactly like this).
   Visible as: `clk=` high, Mcyc/step *up* vs G2's reference. Next step would be `bus_params` at
   `MAX` (V1 variant) — one more skel, still inside the 4-skel budget.
3. **The busy clock moves.** A corner floor could push the busy clock *up* on a unit whose DCVS
   otherwise settled at TURBO (the 1.91 vs 2.05 GHz split of rule 9 may be binning, or may be a
   DCVS choice). Visible as G2's `pcycles_per_us` outside 0.98–1.05× of the reference; either way
   the number is recorded per unit and tok/s stays a within-unit comparison (rule 9 (b)).
4. **Thermal.** A held corner during the 4096 run's 168 s of sustained load may throttle earlier
   than the no-vote run. The 4096 cell is already within-session only (rule 9 (e), #53); the
   handoff records battery temperature before it and treats a > 5 % Mcyc miss at 4096 as
   inconclusive, as #25/#53 do.
5. **Stale artifacts.** Three skels plus an app binary from two possible build hosts: every row
   repeats the md5 `run()` printed; the app row repeats `md5sum` of `nntrainer_causallm` and of the
   skel under `ADSP_LIBRARY_PATH` at that moment; the `power vote` FARF line's presence is itself
   the proof that the V1/V2 skel ran (N has no such line).
6. **Simulator gap.** None for the numerics (no kernel change), and none for the vote (the
   simulator cannot represent DCVS); the sim gate here is purely "nothing else moved".

## 6. Docs to update

* `HEXAGON.md` §1.2 "Session setup": the vote (what `init` requests — apptype compute, DCVS_v3
  `ADJUST_ONLY_UP` / corner floor / 40 µs latency —, that it is per session, released at `close`,
  and held for as long as an `engine="htp"` process keeps its session open, with G5's cost in
  mA / °C), the `HTP_POWER_VOTE` / `HTP_POWER_CORNER` / `HTP_POWER_FARF` knobs, and the SDK-docs
  citation that clock/power management is an unsigned-PD service.
* `HEXAGON.md` §5.3: `--gap-ms` in the harness command list; §5.4: the app-path decode number and
  RSS refreshed (the table there is M5-era, 13.0 tok/s).
* `HEXAGON.md` §7 rule 9: append (f) — the idle-gap clock before / after the vote on the named unit,
  and the standing rule that harness numbers from this PR on are with-vote numbers.
* `HEXAGON.md` §8.2: a "Clock during host idle gaps (#41)" paragraph under "Host path" with the
  before pair (#24 (iii) / #25) and the after rows (G1, G1', G2, G4, G5), and a one-line cross-
  reference from the M5 bullet (`:1029-1033`) that the 2026-08 vote is now the shipped one.
* `HEXAGON.md` §9: close the #41 bullet; add the FastRPC QoS follow-up candidate.
* `HEXAGON_BENCHMARK.md`: method paragraph ("since #41 the cDSP holds its corner through host
  gaps; app-path decode is within 10 % of the harness"), the stage-1 W8A8 "Now" cell's #41 clause,
  the 512 / 1024 / 4096 V1 rows (unit named), and a log row.
