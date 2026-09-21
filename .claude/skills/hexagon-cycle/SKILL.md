---
name: hexagon-cycle
description: Run one supervision → planning → implementation cycle of the Hexagon NPU backend work, driven by issue labels on dlwlzzero/nntrainer. Args: "--architecture" adds a clean-architecture sweep, "--guide" refreshes the beginner guide, "--dry-run" only reports what would run.
disable-model-invocation: true
---

You are the orchestrator for one cycle of the Hexagon work. The contract is
`docs/plans/0000-agent-system-and-env.md`; do not restate it, follow it.

Preconditions (check, do not fix silently): current branch is `hvx_impl`
or a `hvx/*` branch with a clean tree; `source tools/hexagon/env.sh` works
(SDK 6.4.0.1 + HexKL found); `gh auth status` is logged in; the
GitHub MCP is connected as the repo owner. If the tree is dirty, stop and
say what is uncommitted.

## Steps

1. Fetch the open `hexagon` issues (`list_issues`, labels `hexagon`).
   Print a one-line board: counts per `state:*` and the `needs-user` list.
   With `--dry-run`, stop here and say which subagents would run.
2. Spawn `hexagon-supervisor` (subagent) with the board and the args
   (`--architecture` if given). Wait. If it edited docs, commit them on
   `hvx_impl` as `[docs] ...` with `-s` (docs only; refuse if the diff
   touches anything else).
3. If any issue is `state:needs-plan` and not `needs-user`: spawn
   `hexagon-planner` with the highest-priority one. Wait.
4. If no issue is `state:in-progress` (a `state:needs-measurement` issue
   does not block, as long as the new issue needs no device step):
   pick the highest-priority `state:planned` (or a `state:measured` the
   supervisor handed back as `in-progress`) and spawn `hexagon-implementer`
   with it. Wait. If an issue is `state:in-progress` with unanswered PR
   review comments, spawn the implementer on it instead.
5. If `--guide` was given, or a PR into `hvx_impl` merged since the last
   guide commit (compare `git log -1 -- docs/backend_guide/hexagon-guide`
   with merges on `hvx_impl`), **or a measurement handoff was filled since
   that commit** (an issue moved to `state:measured` this cycle, or a
   `docs/measurements/*.md` newer than the guide commit, on `hvx_impl` or
   on the issue's `hvx/*` branch): spawn `hexagon-guide-writer`. Name the
   measurement files (with `<branch>:<path>` for ones not yet on
   `hvx_impl`) so `06-performance.html` records them. Wait. Commit its
   output on a `hvx/guide-<date>` branch and open a PR (docs only).
   Also publish the guide: GitHub Pages serves the `htp_html` branch's
   `/docs`, so whenever `hvx_impl`'s `docs/backend_guide/hexagon-guide/`
   differs from `htp_html`'s, copy it onto `htp_html` (in a separate
   `git worktree`, `git checkout hvx_impl -- docs/backend_guide/hexagon-guide`)
   and push one `[html]` commit. Then it is live at
   `https://dlwlzzero.github.io/nntrainer/backend_guide/hexagon-guide/`.
6. Final report to the user, in this shape and nothing more:
   * what each subagent did (one line each, issue numbers and PR links),
   * **Your to-do**: every `needs-measurement` handoff (path + estimated
     minutes), every PR waiting for merge, every `needs-user` question,
   * whether another cycle would do anything right now (yes/no and why).

Never run `adb`, never push to `hvx_impl` except the docs-only commits in
steps 2 and 5, never push to `htp_html` except the guide copy in step 5,
never edit `.github/workflows`.
