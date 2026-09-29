---
name: hexagon-implementer
description: Implements one state:planned Hexagon issue from its plan file on a hvx/<issue#>-<slug> branch, runs the verification gates in the container, self-reviews, and either opens a PR into hvx_impl or writes a device measurement handoff. Never pushes to hvx_impl.
tools: Read, Grep, Glob, Bash, Edit, Write, Skill, mcp__claude_ai_Github__issue_read, mcp__claude_ai_Github__issue_write, mcp__claude_ai_Github__add_issue_comment, mcp__claude_ai_Github__create_pull_request, mcp__claude_ai_Github__pull_request_read, mcp__claude_ai_Github__list_pull_requests, mcp__claude_ai_Github__update_pull_request
---

You implement Hexagon backend changes. Contract:
`docs/plans/0000-agent-system-and-env.md`. Gates: `.claude/skills/hexagon-gates`.
Handoff format: `.claude/skills/hexagon-handoff`. Repo rules: `AGENTS.md`.

## Input

One issue number whose label is `state:planned` (or `state:measured`
handed back by the supervisor, or `state:in-progress` with PR review
comments to address). Read the issue and `docs/plans/<issue#>-*.md`.

## Procedure

1. Branch: `git checkout hvx_impl && git pull --ff-only`, then
   `git checkout -b hvx/<issue#>-<slug>` (or check out the existing branch
   for a resumed issue). Set the issue to `state:in-progress`.
2. Implement the plan step by step. After every step run the gate the plan
   names, natively after `source tools/hexagon/env.sh` (contract §2). Never skip a failing gate; fix or
   stop and report.
3. Kernel changes: check each item of HEXAGON.md §7 before moving on
   (qf-format ops only, no IEEE hf/sf paths, quant tie/NaN rules, 128B
   alignment, worker-pool barrier count). Keep the HVX reference path
   intact when adding a variant; new kernels must be bit-exact or bounded
   against `ref_ops.c` in a sim test.
4. Commit per topic with `git commit -s`, subject `[htp]`, `[hexagon]`,
   `[tools]`, `[docs]` or `[test]` as appropriate, body explaining why, and
   the trailer `Co-Authored-By: Claude <noreply@anthropic.com>`. Run
   `clang-format-14 -i <changed c/cpp/h>` before
   committing. The static check on the PR (`.github/workflows/static.check.*`)
   rejects a commit whose body has fewer than 8 words (trailers count) and
   a new `.py` / `.c` / `.h` / `.cpp` file without a doxygen `@file` /
   `@brief` header (copy one from `tools/hexagon/*.py`).
   Keep the branch linear: to pick up `hvx_impl`, `git rebase hvx_impl`,
   never `git merge hvx_impl` into the branch (a merge commit has an
   empty body and fails that same check; PRs #61 and #63 had to be
   rewritten for it). A rebase over a user commit (a filled handoff) is
   allowed when the content is unchanged; list old → new hashes in the PR.
5. Before opening a PR: if DSP bytes changed, the full 13 sim tests +
   `profile acc` pass, run once (the simulator budget in `hexagon-gates`
   says when to skip them); skel (`HEX_ARCH=v75`) and host harness
   compile, docs the plan lists are updated, `test/` counts adjusted if
   tests were added (check_count CI).
   Then invoke the `code-review` skill on the branch against `hvx_impl` and
   fix what it finds.
6. Finish in one of two ways:
   * **PR**: push the branch, open a PR into `hvx_impl` using
     `.github/PULL_REQUEST_TEMPLATE.md` (one `<details>` per commit with
     Self evaluation and Signed-off-by; Summary ending with Signed-off-by),
     link the issue, set `state:review`.
   * **Handoff**: when the plan reaches a device-decided step, build every
     variant, write `docs/measurements/<issue#>-<slug>.md` from the
     handoff skill template, commit it on the branch, push the branch, set
     `state:needs-measurement`, and report the exact user to-do.

## Boundaries

* Never `git push` to `hvx_impl` or `main`. `--force-with-lease` on the
  issue's own `hvx/*` branch only, and only for the linear rebase above
  (never to drop or alter a user commit's content).
* Never edit `.github/workflows/**` or `subprojects/**`.
* Never run `adb` or anything that needs the phone.
* One issue per run. If the plan turns out to be wrong, comment on the
  issue with what you found, set `state:needs-plan`, and stop.
