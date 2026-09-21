# 36 — One op-shape source for the validator, `hexagon_ref_run`, `ref_ops.c` and `sim_model.c`

Issue: dlwlzzero/nntrainer#36 (`prio:p1`, from the 2026-09-17 review, finding A1).
Contract: `docs/plans/0000-agent-system-and-env.md`. Branch: `hvx/36-op-shape` from `hvx_impl` (`1bbd8d53`, which includes the merged #33 validator, PR #44).
A pure refactor: **no byte of any op-list, weight image, validator verdict, reference logit or kernel output may move.** Every gate below is a byte or STAT identity; no device measurement and no handoff.

## 1. Goal and gate

| # | Acceptance (issue) | Pass means |
|---|---|---|
| G1 | One `nntr_htp_op_shape(header, desc, n_tokens)` (or equivalent) in the shared header, consumed by the validator, `hexagon_ref_run`, `ref_ops.c` and `sim_model.c` | `nntr_htp_op_extent()` + `nntr_htp_op_rows()` + `nntr_htp_kv_bytes()` + `nntr_htp_oplist_bytes()` live in `nntrainer/tensor/hexagon/htp/nntr_htp_common.h`; `grep -n 'max_chunk \* d.k\|m \* d\.n \* 2\|\* 128u \* 2u' nntr_htp_common.h hexagon_ref_run.cpp ref_ops.c sim_model.c` finds no per-kind byte formula outside `nntr_htp_op_extent` (§2 lists the exact sites that go away); `test_oplist_header.c` asserts the per-kind extent table by literal numbers (§4 step 1) |
| G2 | One RoPE table generator | `nntr_htp_rope_row_f32()` in `nntrainer/tensor/hexagon/htp/nntr_htp_rope.h`; `graph_lowering.cpp:37-49` and `ref_ops.c:119-128` both reduce to a loop over it; `grep -rn 'powf(theta' nntrainer test Applications` finds exactly one site |
| G3 | Bit-identical `--eval` PPL on x86 | `hexagon_ref_run /work/build_x86_hexagon/qwen3_full --tokens /work/build_x86_hexagon/t512.i32 --eval` → `PPL 41.2365 ... top1 161` (the container clang figure, HEXAGON.md §5.1, #33 plan step 2), and `md5sum qwen3_full.hexw` identical before/after re-packing (the RoPE table is inside the image) |
| G4 | 13 sim tests and `profile acc` STATs unchanged | 13 × `SIM_TEST <name> PASS` on v75; `graph_prefill STAT 0.0245416/9.3795`, `graph_decode 0.0202219/23.975`, `profile_prefill_acc STAT max_abs=0.077216 max_rel=98.2637` bit-identical to HEXAGON.md §8.3; the 12 rc-5 negatives of `test_graph.c:133-151` still return 5 |
| G5 | No ABI change | `NNTR_HTP_ABI_VERSION` stays 4 (`nntr_htp_common.h:18-19`, `test_oplist_header.c:77`); `.hexcfg`, `nntr_hexpack` output, `lower_qwen3` bytes, `find_divergence.py` untouched; `hexagon_ref_run --list-ops` output byte-identical before/after (§4 step 2) |

`check_count`: the new checks are in the standalone x86 program and the simulator library; **no gtest is added or removed.**

## 2. Where it lives

Where the per-op shape knowledge is today, and where each copy goes:

| File:line | Today | Change |
|---|---|---|
| `nntrainer/tensor/hexagon/htp/nntr_htp_common.h:246-358` | per-kind `switch` computing the byte extents inline with the `check_ref` calls (EMBED 247-257, RMSNORM 258-275, W8A8/W8A16 276-287, ROPE 288-298, ATTN 299-320, SILU_MUL 321-331, ADD 332-342, LOGITS 343-355) | the extents move into `nntr_htp_op_extent()` (§3); the `switch` keeps only the **rules** (`n % 64`, PER_HEAD `n % head_dim`, `layer < n_layers`, `max_seq % 64`, KV size) and one generic loop over the four refs; the KV size at `:315-316` uses `nntr_htp_kv_bytes()` |
| `nntr_htp_common.h:171-179` | rc-5 comment | unchanged list; add one line saying extents come from `nntr_htp_op_extent` |
| `nntrainer/tensor/hexagon/htp/ops/htp_ops.h:36-40` (`htp_m`) | `d->m ? d->m : c->n_tokens` | `return nntr_htp_op_rows(d, c->n_tokens);` (same expression, one home; the kernels are otherwise untouched) |
| `test/hexagon/hexagon_ref_run.cpp:68-93` (`out_ref`) | own `switch` with the same formulas, called at `:163` (`--list-ops`, rows = `cfg.max_chunk`) and `:211` (`--dump-op`, rows = `n`) | wrapper over `nntr_htp_op_extent`: rows = `n_tokens`; if `e.out_alias_in0` (ROPE) report `d.in0` with `e.in0` bytes, else `d.out` with `e.out`. `find_divergence.py:93-97` parses the `DUMP ... bytes=` line, so it follows automatically |
| `test/hexagon/sim/ref_ops.c:245` | `m = d->m ? d->m : n_tokens` | `nntr_htp_op_rows(d, n_tokens)` |
| `test/hexagon/sim/ref_ops.c:119-128` (`ref_rope_table_fill`) | second copy of the angle formula, `(__fp16)` cast | loop over `nntr_htp_rope_row_f32`, same `(__fp16)` cast on the sim (§3); `ref_ops.h:61-64` comment points at the shared row |
| `test/hexagon/sim/ref_ops.c:159-170` (`ref_attn` KV append) | ref's row-major K layout | **untouched** (§7) |
| `nntrainer/tensor/hexagon/host/graph_lowering.cpp:33-49` (`write_rope_table`) | first copy of the angle formula, `f32_to_f16_bits` | loop over `nntr_htp_rope_row_f32`, same `f32_to_f16_bits` |
| `nntrainer/tensor/hexagon/htp/nntr_htp_rope.h` | — | **new** plain-C header: `<stdint.h>`, `<math.h>` only; add to `hexagon_host_headers` in `nntrainer/tensor/hexagon/meson.build:7-12` next to `htp/nntr_htp_common.h` |
| `Applications/CausalLM/hexagon/qwen3_lowering.cpp:110-111` | `g.kv_size = 2ull * ...` | `nntr_htp_kv_bytes(cfg.n_layers, cfg.n_kv_heads, cfg.max_seq, cfg.head_dim)`; `:365` oplist size → `nntr_htp_oplist_bytes(n_ops)`. **Op emission (`:121-347`) is untouched:** the op-list bytes are the ABI and G5 compares them |
| `test/hexagon/sim/sim_model.c:77-82` | `kv_bytes`, `oplist_len` formulas | the two shared helpers; `sim_model_build_oplist` (`:186-255`) stays a hand lowering (§3, rejected alternative) but ends with `nntr_htp_oplist_validate(buf, p->oplist_len, sizes)` on the plan's own sizes and returns its rc (signature `void` → `int`; callers `test_graph.c:95`, `test_profile.c:151` check it) so the sim's lowering is bound to the same shape source as the real one |
| `test/hexagon/sim/sim_model.c:123-124` | `ref_rope_table_fill` | unchanged call, now reaching the shared row |
| `test/hexagon/test_oplist_header.c:72-` | rc cases 1-5 from #33 | new literal extent table (§4 step 1) after `build_valid` |
| `test/hexagon/sim/test_graph.c:41-44` (`OPLIST_LEN`, `KV_BYTES`), `test/hexagon/test_lowering.cpp:105-107,735` | independent recomputations | **untouched**: a test that recomputes the number by hand is the check on the helper |
| `nntrainer/tensor/hexagon/htp/ops/hvx-*.c`, `htp_graph.c`, `executor.c`, `hexagon_runner.cpp`, `hexagon_backend.cpp` | — | untouched (kernels: §7 review list trivially satisfied) |
| `docs/backend_guide/HEXAGON.md` §1.4 (`:121-192`), §2.1 (`:215-217`), §3 (`:343-400`) | — | §6 |

ABI: no header or descriptor field changes; `NNTR_HTP_ABI_VERSION` stays 4. Consumers that must move together: none — the point of the change is that `lower_qwen3`, `ref_graph_forward`, the sim `graph` test, `find_divergence.py` and `nntr_hexpack` all read the same numbers they read today.

Lowering shapes the extent table must reproduce byte-for-byte (verified against `qwen3_lowering.cpp` and the validator): EMBED `k = hidden` (:125), `in0` TOKENS; RMSNORM `n = hidden` (:142, :257, :328) and PER_HEAD `n = n_heads*128` / `n_kv_heads*128` with a `head_dim`-long gamma (:190, :203); ROPE in place (:213-216); ATTN `in1`/`in2` are the chunk's fresh K/V rows (:225-227); W8A16 `k = ffn` (:304); LOGITS `m = 1, k = hidden, n = vocab, in0 = act_t` (:339-342) whose `in0` the validator sizes for `max_chunk` rows (#33, `nntr_htp_common.h:344-347`).

## 3. Design

**One extent function, `rows` supplied by the caller, `d.m` ignored.**

```c
struct nntr_htp_op_extent {
  uint64_t in0, in1, in2, out; /* bytes each ref must cover for `rows` token rows */
  uint32_t used;               /* bit i set: ref i (in0,in1,in2,out) is read/written */
  uint32_t out_alias_in0;      /* ROPE: out is in place on in0, not checked separately */
};
/* rows: the validator passes h->max_chunk (the widest count forward() can pass),
 * executors pass n_tokens. d->m is ignored: the validator forces it to 0 on every
 * kind but LOGITS, whose kernel ignores it. Returns 0, or 1 for an unknown kind. */
static inline int nntr_htp_op_extent(const struct nntr_htp_oplist_header *h,
                                     const struct nntr_htp_op_desc *d, uint32_t rows,
                                     struct nntr_htp_op_extent *e);
static inline uint32_t nntr_htp_op_rows(const struct nntr_htp_op_desc *d, uint32_t n_tokens); /* d->m ? d->m : n_tokens */
static inline uint64_t nntr_htp_kv_bytes(uint32_t n_layers, uint32_t n_kv_heads, uint32_t max_seq, uint32_t head_dim); /* 2 * L * H * S * D * 2 */
static inline uint64_t nntr_htp_oplist_bytes(uint32_t n_ops); /* 64 + 64 * n_ops */
```

The per-kind table `nntr_htp_op_extent` encodes (r = rows, `k`/`n` from the descriptor, header fields from `h`; `128u` literal where the code has it today, `head_dim == 128` being a header rule at `:196`):

| kind | in0 | in1 | in2 | out | used | note |
|---|---|---|---|---|---|---|
| EMBED | `r*4` (TOKENS ids) | `vocab*k` (tiled int8) | `vocab*4` (scales) | `r*k*2` | 1111 | |
| RMSNORM | `r*n*2` | `(PER_HEAD ? head_dim : n)*2` | 0 | `r*n*2` | 1101 | gamma length is the only flag-dependent extent |
| MATMUL_W8A8, W8A16 | `r*k*2` | `n*k` | `n*4` | `r*n*2` | 1111 | same operand layout (`:277`) |
| ROPE | `r*n_heads*128*2` | `r*n_kv_heads*128*2` | `max_seq*128*2` (table) | `= in0` | 0111 | `out_alias_in0 = 1` |
| ATTN | `r*n_heads*128*2` | `r*n_kv_heads*128*2` | `r*n_kv_heads*128*2` | `r*n_heads*128*2` | 1111 | the KV cache is not a ref: `nntr_htp_kv_bytes` |
| SILU_MUL, ADD | `r*n*2` | `r*n*2` | 0 | `r*n*2` | 1011 | |
| MATMUL_LOGITS | `r*k*2` | `n*k` | `n*4` | `n*4` | 1111 | validator passes `r = max_chunk` → exactly #33's `max_chunk*k*2`; the executors' `r = n_tokens` is the extent they read (row `n_tokens-1`) |

`used` exists so the validator's accept/reject set is reproduced exactly: today it checks precisely the refs each kind reads, including a 0-byte extent when `n == 0` (an RMSNORM with `n = 0` and an `in0.offset` past the buffer is rejected today and must stay rejected); a "skip if bytes == 0" loop would silently widen the accepted set. The validator becomes: rules as today → `nntr_htp_op_extent(&h, &d, h.max_chunk, &e)` → for each ref with its `used` bit, `nntr_htp_check_ref(ref.buf, ref.offset, bytes, buf_size)`; ROPE's `out` bit is clear, so `:289` keeps its meaning.

The header stays include-free beyond `<stdint.h>`/`<string.h>` (`:14-15`): the extent function is integer arithmetic, so the DSP skel, the QuRT sim library, the NDK host build, the x86 `g++` tools and the `gcc -Wall -Werror` self-check all compile it unchanged. That is what "both the DSP validator and the x86/sim C code include it without pulling QuRT headers" means here: nothing in the shared header may include a `HAP_*`/`qurt*` header, and nothing does.

**RoPE: share the fp32 row, keep each side's fp16 conversion.** `nntr_htp_rope_row_f32(float row[128], uint32_t p, float theta)` computes `inv_freq = powf(theta, -2.0f * (float)i / 128.0f)`, `row[i] = cosf((float)p * inv_freq)`, `row[64 + i] = sinf(...)` — the identical expression both copies use today (`graph_lowering.cpp:43-46`, `ref_ops.c:123-125`; no add, so `-ffp-contract` cannot fuse anything). The host converts each row with `f32_to_f16_bits` as before; the sim casts `(__fp16)` as before. Bytes therefore cannot move on either side by construction: same fp32 operations in the same order against the same libm each side already uses, then the same conversion path. It lives in its own header `nntr_htp_rope.h` because it needs `<math.h>`, which the wire-format header must not drag into every DSP translation unit. Rejected: a shared fp16-emitting generator using an integer RNE converter on both sides. It would be one function fewer, but the sim's table would then depend on `f32_to_f16_bits` agreeing with hexagon-clang's `__fp16` cast on every subnormal `sin` value (`p = 1, i = 63` gives `1.2e-6`, well inside fp16 subnormals); that is probably true and cannot be proven on x86, and a 1-ulp table change would move the `graph`/`profile` STATs that G4 pins.

**Rejected alternative for `sim_model.c`: calling `lower_qwen3` from the sim.** The sim library is C compiled by hexagon-clang (`build_sim_test.sh:28-45`); `lower_qwen3` is C++ with `std::vector` in the app directory (`Applications/CausalLM/hexagon`), and moving the qwen3 recipe into a shared C header would put a model-specific lowering into the model-agnostic runtime header (HEXAGON.md §2 keeps recipes with the app on purpose). So `sim_model_build_oplist` stays a hand lowering; what it gains is the two size helpers and the self-validation, which binds its output to the same extent table the real lowering is checked against (`test_lowering.cpp:643-646,743-745`). Its 16-op-per-layer sequence is already cross-checked by `test_graph.c:133-151`'s index/kind guards.

## 4. Steps

Commands run from the repo root through `tools/docker/run.sh`; `HEX_ARCH=v75` always explicit. **No simulator run before step 4** (contract §7: `ref_ops.c`, `sim_model.c` and `htp/` change, so rungs 2 and 3 run once, right before the PR). The container is shared with #35's implementer: do not start a second container; if `run.sh` reports the container busy, wait rather than build natively.

**Step 0 — branch and baselines.** `git checkout -b hvx/36-op-shape hvx_impl`. Before touching anything, record the "before" artifacts in `logs/hexagon/36-before/`: `./tools/hexagon/build_host_x86.sh`; `./build_x86_hexagon/nntr_hexpack /model/<w8cx>.bin /work/build_x86_hexagon/qwen3_full` then `md5sum qwen3_full.hexw` (expected 598,623,744 B); `hexagon_ref_run qwen3_full --list-ops > listops.before`; `hexagon_ref_run qwen3_full --tokens t512.i32 --chunk 8 --dump-op N --dump-out dump_N.before` for N ∈ {1 (EMBED), 8 (ROPE, in place), 9 (ATTN), 450 (final RMSNORM)}; the `--eval` line if not already in the log (`PPL 41.2365 ... top1 161`, ~160 s). `git diff --stat` must be empty after this step.

**Step 1 — shared helpers + x86 table.** Add `nntr_htp_op_rows`, `nntr_htp_kv_bytes`, `nntr_htp_oplist_bytes`, `struct nntr_htp_op_extent`, `nntr_htp_op_extent` to `nntr_htp_common.h` above `nntr_htp_oplist_validate`; rewrite the validator's `switch` per §3 (rules stay, extents come from the function, one ref loop). Add `nntr_htp_rope.h`. In `test_oplist_header.c`, after `build_valid` (header: `max_chunk 4`, `hidden 128`, `vocab 32`, `n_heads 2`, `n_kv_heads 2`, `max_seq 8`, `head_dim 128`), assert the table by literal numbers with `rows = 4`:
EMBED `k=128` → `16 / 4096 / 128 / 1024`, used `0xF`; RMSNORM `n=128` → `1024 / 256 / – / 1024`, used `0xB`; RMSNORM PER_HEAD `n=256` → `2048 / 256 / – / 2048` (gamma stays `head_dim`); W8A8 and W8A16 `k=128, n=256` → `1024 / 32768 / 1024 / 2048`, used `0xF`; ROPE → `2048 / 2048 / 2048`, `out_alias_in0 == 1`, used `0x7`; ATTN → `2048 / 2048 / 2048 / 2048`, used `0xF`; SILU_MUL and ADD `n=128` → `1024 / 1024 / – / 1024`, used `0xB`; LOGITS `m=1, k=128, n=32` → `1024 / 4096 / 128 / 128` at `rows = 4` **and** `in0 == 256` at `rows = 1`; `kind = 9` → returns 1; `nntr_htp_op_rows(m=0, 3) == 3`, `(m=1, 3) == 1`; `nntr_htp_kv_bytes(1, 2, 8, 128) == 8192`; `nntr_htp_oplist_bytes(2) == 192`; `nntr_htp_rope_row_f32(row, 0, 1e6f)` gives `row[i] == 1.0f`, `row[64+i] == 0.0f` for all i (p = 0 is libm-independent). The #33 rc-5 cases stay as they are and must still pass — they are the proof the validator consumes the table without moving a verdict.
Gate 1: `gcc -Wall -Werror -o /tmp/t test/hexagon/test_oplist_header.c -lm && /tmp/t` → `oplist header check: PASS` (`-lm` is new: the rope row calls `powf`/`cosf`/`sinf`; update the command in the `hexagon-gates` skill and HEXAGON.md §5.1 in the same PR); `./tools/hexagon/build_host_x86.sh` and `./build_x86_hexagon/test_lowering` → `LOWER_TEST PASS` (the validator still accepts the tiny and the qwen3-0.6b lowering).

**Step 2 — consumers.** `htp_ops.h` `htp_m`; `hexagon_ref_run.cpp` `out_ref`; `ref_ops.c:245` rows and `:119-128` rope; `graph_lowering.cpp` `write_rope_table`; `qwen3_lowering.cpp:110-111,365`; `sim_model.c:77-82,186-255` (+ `sim_model.h:52-53` comment and `int` return; `test_graph.c:95` and `test_profile.c:151` print `SIM_TEST <t> FAIL sim_model validate rc=%d` on nonzero).
Gate 2 (x86, all byte identities): rebuild; `md5sum qwen3_full.hexw` after re-packing equals step 0 (proves the host RoPE table did not move); `--list-ops` → `diff listops.before -` empty (proves `out_ref` for all 451 ops at `rows = max_chunk`); the four `--dump-op` files `cmp` equal (proves `out_ref` at `rows = n`, incl. the ROPE alias, and that `ref_graph_forward_upto` is unchanged); `--eval` → `PPL 41.2365 ... top1 161` (G3); `LOWER_TEST PASS`; `oplist header check: PASS`. A `-fsyntax-only` pass with the NDK `clang++` from `build_host_test.sh` over `graph_lowering.cpp` is not needed: rung 4 compiles it.

**Step 3 — dead code and docs.** Remove the now-unused `kKindName`-adjacent leftovers only if the compiler flags them (`-Wall -Werror` is on); keep `kKindName`. §6 docs. Gate: `git diff --stat` shows only the files in §2.

**Step 4 — simulator, once (rungs 2 + 3).** `HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/build_sim_test.sh`; `run_sim_test.sh graph` → `SIM_TEST graph PASS` with `graph_prefill STAT 0.0245416/9.3795`, `graph_decode 0.0202219/23.975` (bit-identical: the sim RoPE table, the rows rule and the 12 rc-5 negatives all sit under this one test); `run_sim_test.sh profile acc` → `profile_prefill_acc STAT max_abs=0.077216 max_rel=98.2637` (~10 min); then the 13 tests `smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed attn logits graph` all PASS (~5 min), `quant_generic 0/65536`, `quant16_generic 167/25600` as in §8.3. Any STAT that moves by one digit fails the step; bisect by reverting `ref_ops.c:119-128` first (the only sim-side numeric site), then `htp_ops.h`.

**Step 5 — skel and harness (rung 4).** `HEX_ARCH=v75 ./tools/hexagon/build_skel.sh` and `./tools/hexagon/build_host_test.sh` build; record both md5s in the PR. The skel md5 **will** change (the validator's compiled form moved), which is expected and is why step 4 runs; no v79 build (no `__HVX_ARCH__` surface touched). No handoff: nothing here is a performance or silicon question; the DSP-side change is confined to the init-time validator and an inline row-count wrapper.

**Step 6 — format and PR.** `tools/docker/run.sh clang-format-14 -i` on the changed `.c/.cpp/.h`; three commits with `git commit -s`: `[htp] Share the op extent table, row rule and size helpers in nntr_htp_common.h` (header, validator, `htp_ops.h`, `test_oplist_header.c`), `[hexagon] Consume the shared op extents and RoPE row on host, x86 ref and sim` (`graph_lowering.cpp`, `nntr_htp_rope.h`, meson, `qwen3_lowering.cpp`, `hexagon_ref_run.cpp`, `ref_ops.*`, `sim_model.*`, `test_graph.c`, `test_profile.c`), `[docs]`. PR into `hvx_impl` with every gate line and the before/after md5s pasted; issue → `state:review`.

## 5. Risks

* **A byte count differs from #33's and a bounds negative flips.** The extent table is asserted by literal numbers on x86 before the sim runs (step 1), and the 12 `test_graph.c` rc-5 cases plus `logits_in0_short`/`attn_in2_short` are the sim-side proof. `LOGITS in0` at `rows = max_chunk` is the one entry that is not `d.m`-shaped; it is asserted at both `rows = 4` and `rows = 1`.
* **The sim RoPE table moves by an ulp.** Ruled out by construction (§3: same fp32 expression, same `(__fp16)` cast), and detected by the `graph`/`profile` STAT identity if it were wrong. The `rope` test alone would not catch it (kernel and reference read the same table), which is why the STAT lines, not `SIM_TEST rope PASS`, are the gate.
* **The host RoPE table moves.** Detected by the `.hexw` md5 in gate 2 before any sim time is spent.
* **`used`-mask semantics.** If the implementer loops "check when bytes > 0" instead, degenerate `n == 0` lists are accepted where they were rejected. No existing test covers it; the plan's `used` bits make the reproduction exact and `test_oplist_header.c` asserts the mask per kind.
* **Container contention.** #35's implementer holds the container; steps 0-3 are x86-only and short, step 4 is the one long block (~20 min) and must not overlap a `profile acc` of another issue (two Rosetta simulators halve each other).
* **Pending history-clean replacement of `hvx_impl` (#34)** as in the #33 plan: if `hvx_impl` is force-replaced first, `git rebase --onto hvx_impl 1bbd8d53 hvx/36-op-shape` is conflict-free; keep the PR base `hvx_impl`.
* **Stale x86 artifacts.** `build_x86_hexagon/` is a plain output directory; step 0 rebuilds before recording baselines so "before" is `1bbd8d53`, not an older binary.
* Simulator-vs-device gaps (bandwidth, qf32 numerics): none exposed — no kernel arithmetic changes; the skel md5 changes only through the init-time validator, and `AEE_EBADPARM` mapping (`executor.c:149-152`) is untouched.

## 6. Docs to update

* `docs/backend_guide/HEXAGON.md` §1.4 (`:150-170`): after the per-op rule list, one sentence: the byte extents behind every "sized for `max_chunk` rows" clause come from `nntr_htp_op_extent()` in `nntr_htp_common.h`, which `hexagon_ref_run --list-ops`/`--dump-op`, `ref_graph_forward` and `sim_model` share, so the four cannot disagree on an operand size.
* HEXAGON.md §2.1 item 3 (`:215-217`): "precomputed on the host by `nntr_htp_rope_row_f32` (`nntr_htp_rope.h`), the same row the simulator reference fills".
* HEXAGON.md §3 (`:347`): add `nntr_htp_rope.h` under `htp/` ("RoPE cos/sin row, shared by the packer and the sim reference"); extend the `nntr_htp_common.h` line with "op extents / row rule / size helpers".
* HEXAGON.md §5.2 (`:503-513`): `graph` now also fails if `sim_model_build_oplist`'s self-validation rejects the hand lowering (one line).
* `HEXAGON_BENCHMARK.md`: no row changes (no performance claim).

## 7. Out of scope

* **K^T KV layout** (`hvx-attn.c:57-59` transposed K vs `ref_ops.c:159-170` row-major K, `test_attn.c:31-40` translating between them, §2.2). The two layouts are intentionally different (the ref keeps rows, the kernel streams 64 positions per vector); sharing an index helper means editing a kernel and its reference together, which is a kernel change with its own `attn` gate, not a shape refactor. Only the KV *size* is shared here (`nntr_htp_kv_bytes`). File as a follow-up if the supervisor wants one home for the index.
* **Replacing `sim_model.c` with `lower_qwen3`** (§3, rejected).
* **A per-op `n_ops`-indexed extent cache on the DSP** for runtime dumps (`htp_graph_buf_ref`): the executor bounds-checks dump refs against buffer sizes, not op extents, and that is enough.
