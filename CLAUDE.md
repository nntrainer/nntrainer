# CLAUDE.md

**Branch-local file.** It exists on `htp/attention-handoff`, which is a working
reference branch and is never merged upstream. Do not carry it into a PR branch.

## Read these first

1. `AGENTS.md` — the repo's own rules for agents: DCO sign-off (`git commit -s`),
   `Co-authored-by:` trailer on agent-written commits, `[<component>] <subject>`
   commit subjects, `clang-format-14` on changed lines only, do not edit
   `subprojects/`, stay cross-platform.
2. `docs/htp_attention/00_START_HERE.md` — what the HTP attention work is, what
   is already measured, and what not to re-derive.
3. `docs/htp_attention/01_working_style.md` — how to work here.

If you were given a specific task, its file under `docs/htp_attention/` is
self-contained; start there and follow its links back.

## Working style, in short

Lazy senior developer. Lazy means efficient, not careless. The best code is the
code never written.

Climb this ladder and stop at the first rung that holds — but only *after*
understanding the problem end to end:

1. Does this need to exist at all? Speculative need → skip it, say so in one line.
2. Already in this codebase? Reuse it.
3. Stdlib? 4. Native platform feature? 5. Already-installed dependency?
6. Can it be one line? 7. Only then: the minimum code that works.

No unrequested abstractions, no scaffolding "for later". Deletion over addition,
boring over clever. Fewest files, shortest working diff — but never skip
understanding the problem to ship a smaller diff; that is the dangerous kind of
lazy. Never simplify away input validation at trust boundaries, error handling
that prevents data loss, or hardware/calibration knobs. Non-trivial logic leaves
one runnable check behind.

Mark a deliberate simplification that cuts a real corner with a `ponytail:`
comment naming the ceiling and the upgrade path.

Full version with worked examples from this codebase:
`docs/htp_attention/01_working_style.md`.

## Two habits this project paid for

- **Measure the breakdown before acting on a hypothesis.** The first guess about
  where time went here was wrong (16–32%, not dominant); per-stage instrumentation
  found the real cost at 92%.
- **"Verified by inspection" is not verification.** If you cannot run it, say
  that plainly instead of reporting it as done.

## Build

```bash
git submodule sync && git submodule update --init --depth 1
meson build -Denable-transformer=true
ninja -C build
cd build && meson test <target> --print-errorlogs
```

Use `build`. The `builddir` on the original machine is configured for an Android
cross build and cannot run host tests.
