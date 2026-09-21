# 32 — W8_CX producer: name `make_w8cx_bin.py` in the docs, print the checkpoint md5 in the wizard

Issue: dlwlzzero/nntrainer#32 (`prio:p2`, `state:planned`). Decision (b), supervisor 2026-09-17:
`tools/hexagon/make_w8cx_bin.py` stays the only producer; the hvx_m3 `nntr_quantize --fc_dtype W8_CX`
commits are not cherry-picked. Contract: `docs/plans/0000-agent-system-and-env.md`.
Branch: `hvx/32-w8cx-producer` from `hvx_impl` (after the #34 replacement, tip `8686d520`).
**No kernel, ABI or DSP change; no simulator run; no device handoff** (contract §7 host-only budget).

## 1. Goal and gate

| # | Acceptance (issue) | Pass means |
|---|---|---|
| G1 | HEXAGON.md §2.4 and §5.1 name the producer that is in the tree | §2.4 no longer says `nntr_quantize` writes the `.bin`; §5.1 states the expected size and md5 next to the generator line |
| G2 | The two `@brief` lines of `qwen3_w8cx_bin.{h,cpp}` name the producer | `grep -n nntr_quantize Applications/CausalLM/hexagon/qwen3_w8cx_bin.*` is empty |
| G3 | The wizard prints the `.bin` md5 and warns on mismatch | a fresh clone + `tools/docker/setup_wizard.sh` prints `md5 7562313bb4cc70410450d0ab3a4fa563`; a different snapshot prints a warning, not a failure |
| G4 | Nothing else moves | `./build_x86_hexagon/test_w8cx_bin` still PASS; `git diff --stat` shows only the five files in §2 |

Evidence for (b): on `hvx_impl` nothing needs a CPU-side W8_CX DataType — `grep -rn 'W8_CX\|w8cx' nntrainer/`
is empty, `Applications/CausalLM/quantize.cpp` lists FP32/FP16/Q4_0/Q4_K/Q6_K/QS4CX only, and every consumer
(`Qwen3W8cxBin`, `nntr_hexpack`, `HexagonBackend::create`, `test_w8cx_bin`) reads the `.bin` file. The open
question from the issue body is answered by the #23 handoff: the workstation's `.bin` has md5
`7562313bb4cc70410450d0ab3a4fa563` and its packed `qwen3_full.hexw` matches the container's byte for byte,
so the generator reproduces the nntr_quantize bytes.

## 2. Where it lives

| File:line | Today | Change |
|---|---|---|
| `docs/backend_guide/HEXAGON.md:294` (§2.4) | "`nntr_quantize` writes the W8_CX `.bin` (header-less, 598,230,528 B for …)" | "`tools/hexagon/make_w8cx_bin.py` writes the W8_CX `.bin` from the HuggingFace directory (same primitive and tensor order as `nntr_quantize --fc_dtype W8_CX` on the hvx_m3 branch, which is not on this branch)"; keep the size |
| `docs/backend_guide/HEXAGON.md:402-409` (§5.1) | generator line, size only | add the expected md5 `7562313bb4cc70410450d0ab3a4fa563` for Qwen3-0.6B next to 598,230,528 B, and note that other HF snapshots are legitimate and only change the md5 |
| `Applications/CausalLM/hexagon/qwen3_w8cx_bin.h:5, :30` and `qwen3_w8cx_bin.cpp:5` | "Read-only mmap view over an nntr_quantize W8_CX qwen3 checkpoint" | "… over a W8_CX qwen3 checkpoint (`tools/hexagon/make_w8cx_bin.py`)" |
| `tools/docker/setup_wizard.sh:116` | `ok "W8_CX checkpoint present (<bytes> bytes; expected 598230528)"` | after the size line: compute `md5sum`/`md5 -q` of `$bin`, print it, and print a warning (a `warn()` helper next to `ok()` at `:27`; not a failure) when it differs from `7562313bb4cc70410450d0ab3a4fa563` |

## 3. Design

Docs and one wizard line only. The md5 is a warning because a different HuggingFace snapshot legitimately
produces a different `.bin`; the size check stays the hard gate. No reference PPL or benchmark row changes.

## 4. Steps

1. `git checkout -b hvx/32-w8cx-producer hvx_impl`.
2. `[docs]` commit: HEXAGON.md §2.4 and §5.1 (G1) + the three `@brief` lines (G2; a comment-only edit in
   `Applications/CausalLM/hexagon/`, `clang-format-14 --dry-run` clean, `tools/scripts/doxygen-tag.sh` clean).
3. `[tools]` commit: `setup_wizard.sh` md5 line (G3); `bash -n tools/docker/setup_wizard.sh`.
4. Gate: `./tools/hexagon/build_host_x86.sh && ./build_x86_hexagon/test_w8cx_bin /model/nntr_qwen3_0.6b_w8cx_DEFAULT.bin`
   → PASS (G4); run the wizard once and paste the md5 line into the PR.
5. PR into `hvx_impl` (`.github/PULL_REQUEST_TEMPLATE.md`, one `<details>` per commit); issue → `state:review`.

## 5. Risks

* `md5sum` is absent on macOS (`md5 -q` instead): the wizard already branches on `stat -f`/`stat -c` at `:116`;
  use the same pattern.
* HEXAGON.md §2.4/§5.1 hunks: #23's branch edits §5.2/§8/§9 and #33's edits §1.4/§5.2; disjoint, rebase is
  conflict-free.

## 6. Docs to update

* HEXAGON.md §2.4, §5.1 (above). `HEXAGON_BENCHMARK.md`: no change. Ledger: no entry (nothing performance-related).
