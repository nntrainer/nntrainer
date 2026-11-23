# AGENTS.md — Guide for AI coding agents working on nntrainer

This file is the entry point for AI coding agents (Claude Code, etc.) working in
this repository. Read it before building, testing, committing, or opening a PR.

Detailed rules live in `docs/` — this file links to them rather than copying, so
the source of truth stays in one place.

## ⚡ Non-negotiable rules (do these before every commit / PR)

- **Sign off every commit:** `git commit -s` (DCO). A `Signed-off-by:` line is
  required; commits without it fail the review check.
- **Mark agent-authored commits:** if an AI agent wrote the commit, add a
  `Co-authored-by:` trailer identifying the agent, e.g.
  `Co-authored-by: Claude <noreply@anthropic.com>`.
- **Format C/C++ with clang-format 14:** run `clang-format-14 -i <files>` on any
  `.c`/`.cpp`/`.h` you changed, using the repo's [`.clang-format`](https://github.com/nntrainer/nntrainer/blob/main/.clang-format).
  - CI (`cpp_linter`) only checks the **lines you changed**, version **14**. To
    reproduce CI exactly on changed lines only:
    `git clang-format-14 <base-sha>`.
- **Commit message format:** `[<component>] <short description>`.
  Examples: `[CausalLM] ...`, `[test] ...`, `[kv_cache_manager] ...`.
- **One commit = one topic.** Keep each commit focused; provide enough background.
- **Do not edit `subprojects/`** — these are vendored dependencies (minja,
  benchmark, ...). Changes there do not belong in nntrainer PRs.
- **Stay cross-platform.** This builds on Android, Tizen, Windows, and Yocto in
  addition to Ubuntu. Avoid Linux-only headers/APIs unless properly guarded.

## 🔧 Quick commands

```bash
# First time only: pull submodules (required to build)
git submodule sync && git submodule update --init --depth 1

# Configure + build (from repo root)
meson build -Denable-transformer=true
ninja -C build

# Run unit tests (gtest based)
ninja -C build test

# Format changed files (clang-format 14)
clang-format-14 -i <changed files>
# ...or match CI exactly (changed lines only, against the PR base):
git clang-format-14 <base-sha>
```

See [docs/getting-started.md](https://github.com/nntrainer/nntrainer/blob/main/docs/getting-started.md) for full build
dependencies and platform-specific builds (Tizen/GBS, Android, Windows).

## 📝 PR description rules (agents get this wrong most often)

Write the PR body using [`.github/PULL_REQUEST_TEMPLATE.md`](https://github.com/nntrainer/nntrainer/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
exactly:

- **One `<details>` block per commit**, each containing the commit message, a
  **Self evaluation** (Build test / Run test), and a `Signed-off-by:` line.
- A **Summary** section, also ending with `Signed-off-by:`.

## ✅ CI gates — the definition of "done"

A PR is mergeable only when these pass. They run on every pull request:

| Check | What it verifies |
|-------|------------------|
| `C++ Format Checker` (cpp_linter) | clang-format 14 on changed lines |
| `Static checkers and verifiers` (static.check) | static analysis / verifiers |
| `Check Review` (check_count) | review/count checks |
| `CodeQL Advanced` | security/code scanning |
| Build matrix | Ubuntu Meson (gcc & clang), Android NDK, CausalLM Android, Tizen/GBS, Ubuntu Pdebuild, Windows (Meson / on-ARM / cross-compile) |

`Yocto build` (`yocto_build.yml`) is not a PR gate: it builds nntrainer with
Yocto (qemux86-64) nightly on `main` and on manual dispatch.

If you add or remove tests, expect `check_count` to react — update counts
accordingly.

## 🗂️ Repo map

- `api/` — public API definitions (`capi` = C-API, `ccapi` = C++-API)
- `nntrainer/` — all core NNTrainer code
- `Applications/` — examples and apps (e.g. CausalLM)
- `test/` — gtest unit tests, grouped into subdirectories
- `nnstreamer/` — NNStreamer sub-filter code for NNTrainer
- `jni/` — Android/Java build scripts
- `tools/` — developmental tools and scripts
- `packaging/` — Tizen RPM build scripts
- `debian/` — Debian/Ubuntu packaging
- `subprojects/` — vendored dependencies (**do not edit**)

## 📚 Deep dive (read only when needed)

- Coding convention → [docs/coding-convention.md](https://github.com/nntrainer/nntrainer/blob/main/docs/coding-convention.md)
- Contributing & merge process, sign-off details → [docs/contributing.md](https://github.com/nntrainer/nntrainer/blob/main/docs/contributing.md)
- Build & dependencies → [docs/getting-started.md](https://github.com/nntrainer/nntrainer/blob/main/docs/getting-started.md)
- Running test cases → [docs/how-to-use-testcases.md](https://github.com/nntrainer/nntrainer/blob/main/docs/how-to-use-testcases.md)
