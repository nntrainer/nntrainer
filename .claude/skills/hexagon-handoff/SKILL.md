---
name: hexagon-handoff
description: Write a device measurement handoff document (docs/measurements/<issue#>-<slug>.md) that a human runs on the workstation with the Galaxy S25, and read a filled one back. Use whenever a Hexagon task needs silicon numbers.
---

Agents never touch the phone. A handoff turns "we need a device number"
into a file the user can execute top to bottom in one sitting, then fill
in. Keep it under one screen of commands; at most 4 skel variants; sweeps
at 512 tokens only, goal checks at 512 / 1024 / 4096.

## Writing one

File: `docs/measurements/<issue#>-<slug>.md`, committed on the issue's
`hvx/<issue#>-<slug>` branch together with the artifacts' md5s. Template:

```markdown
# Measurement <issue#>: <one-line purpose>

Branch `hvx/<issue#>-<slug>` @ `<commit sha>` — estimated device time: <N> min

## Why
<two sentences: the question this answers and what decision hangs on it>

## Artifacts (built in the container, SDK <ver>, HEX_ARCH=<v75|v79>)
| file | md5 | built with |
|---|---|---|
| build_hexagon/skel/libnntr_htp_skel.<variant>.so | <md5> | HEX_EXTRA_CFLAGS=... |
| build_hexagon/host/hexagon_e2e_test | <md5> | |
| build_x86_hexagon/qwen3_full.hexw / .hexcfg | <md5> | nntr_hexpack @ <sha> |
| <tokens>.i32 | <md5> | make_tokens.py --limit <n> |

## Steps (workstation, phone on USB)
1. `git fetch && git checkout hvx/<issue#>-<slug>` and rebuild or copy the artifacts above if this checkout did not build them.
2. `adb devices` shows `R3CY10WM83Y device`.
3. For each variant: `cp build_hexagon/skel/libnntr_htp_skel.<variant>.so build_hexagon/skel/libnntr_htp_skel.so && ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY10WM83Y -- --tokens /tmp/t512.i32 --chunk 128 --steps 64`
   Expected log lines: `E2E init ok weights=... act=... n_ops=...`, one `E2E step ... n=<n> pcycles=<c> us=<t> top1=<id>` per chunk / decode step, `E2E gen <ids>`, `E2E decode steps=<n> median_us=<u> median_pcycles=<c> pcycles_per_us=<r>` (since #24: the medians the result table reads, over the `n=1` steps — no by-hand median), `E2E wall_ms <w>`.
4. Accuracy: `... -- --tokens /tmp/t512.i32 --eval` → `E2E ppl <x> steps <n> top1 <n> wall_ms <w>` followed by the same `E2E decode ...` line.
5. Paste the summary lines below, commit this file on the same branch, push, and set the issue label to `state:measured`.

## Results (fill in)
| variant | skel md5 (from log) | ctx | prefill tok/s | decode ms/tok | decode tok/s | Mcycles/tok |
|---|---|---|---|---|---|---|
| A | | 512 | | | | |

| variant | ctx | --eval PPL | top-1 | x86 ref PPL | find_divergence bound |
|---|---|---|---|---|---|
| A | 512 | | | 33.0195 | |

## Notes from the run
<anything odd: thermal, warm-up, FARF errors, stale-file pushes>
```

Rules for the author:

* Every artifact row has an md5 and the commit it was built from
  (environment-gap rule). Ask the user to copy the `skel md5` that
  `run_e2e_test.sh` prints, not the one you expect.
* Accuracy columns are mandatory even for a pure speed sweep
  (numerical-gap rule).
* Put the reference number the result will be compared against in the
  table (HEXAGON.md §8.2 / HEXAGON_BENCHMARK.md), so the user can see a
  regression while still at the desk.
* State the estimated minutes at the top; the user decides when to run it.

## Reading a filled one (supervisor / implementer)

1. Check the md5 column against the artifact table; a mismatch means the
   run is void, say so and re-issue the steps.
2. Compare each result with the reference cell; classify as
   goal-progress / regression / no-change with the threshold from the
   plan.
3. Copy the numbers to HEXAGON_BENCHMARK.md (rows) and, for design
   verdicts, HEXAGON.md §8 (tables, with the date and the skel variant).
4. If the device disagrees with the simulator, add the rule to HEXAGON.md
   §7 before doing anything else.
