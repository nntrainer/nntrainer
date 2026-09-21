# 33 — DSP runtime/validator bounds: token ids, ATTN.layer, k > 0, n % 64, LOGITS row

Issue: dlwlzzero/nntrainer#33 (`prio:p0`, from the 2026-09-17 review, findings M1/M2/M5).
Contract: `docs/plans/0000-agent-system-and-env.md`. Branch: `hvx/33-dsp-bounds` from `hvx_impl` (`25067f02`).
Every gate below is decidable on x86 and the v75 simulator; **no device measurement and no handoff.**
Independent of #23 (`state:needs-measurement`): shares no source file, only two `[tools]` commits (§5).

## 1. Goal and gate

| # | Acceptance (issue) | Pass means |
|---|---|---|
| G1 | `nntr_htp_oplist_validate` rejects `ATTN.layer >= n_layers`, `k == 0` on the four `k % 128` kinds, `n % 64 != 0` on RMSNORM/ADD/SILU_MUL, PER_HEAD `n % head_dim != 0`, and a `MATMUL_LOGITS` `in0` that cannot hold `max_chunk` rows | `test/hexagon/test_oplist_header.c` asserts rc 5 for each case and rc 0 for its control; `oplist header check: PASS`; `LOWER_TEST PASS` (the validator still accepts the tiny and the qwen3-0.6b lowering, `test_lowering.cpp:643-646,743-745`) |
| G2 | `htp_graph_forward_upto` rejects any token id `>= vocab`; `nntr_htp_forward` returns `AEE_EBADPARM`; KV/ACT untouched | `test_graph` forward negatives (§4 step 3) inside `SIM_TEST graph PASS`; the AEE mapping is the unchanged `executor.c:149-152` (any nonzero rc → `AEE_EBADPARM`) |
| G3 | `HexagonBackend::forward` pre-checks ids on the host | ids checked over the whole request before the first RPC; returns the `AEE_EBADPARM` value (§3) |
| G4 | 13 sim tests + `profile acc` still pass on v75 | 13 × `SIM_TEST <name> PASS`; `profile_prefill_acc STAT max_abs=0.077216 max_rel=98.2637` bit-identical to HEXAGON.md §8.3 (SDK 6.4 record) |
| G5 | Docs | HEXAGON.md §1.4 lists the new init checks and the forward-time id check; §1.3 unchanged (rc mapping does not move) |

No ABI change: no header/descriptor field is added or re-interpreted, only lists that the kernels could never execute are now refused. `NNTR_HTP_ABI_VERSION` stays 4 (`nntr_htp_common.h:18-19`; `test_oplist_header.c:77` keeps `== 4u`); `.hexcfg`, `nntr_hexpack`, `lower_qwen3`, `ref_graph_forward`, `find_divergence.py` do not move. `check_count`: all new tests are the standalone x86 program and the simulator library, outside `meson test`; **no gtest is added or removed, the gtest count does not change** (review m11 stays as it is).

## 2. Where it lives

| File:line | Today | Change |
|---|---|---|
| `nntrainer/tensor/hexagon/htp/nntr_htp_common.h:195-200` | `k % 128` for W8A8/W8A16/LOGITS/EMBED; `k == 0` passes | add `d.k == 0` → 5 to the same condition |
| `nntr_htp_common.h:228-239` (RMSNORM), `:276-293` (SILU_MUL, ADD) | bounds only | `d.n % 64u` → 5 (the kernels load whole 64-half vectors: `hvx-rmsnorm.c:39-43`, `hvx-eltwise.c:47-49,106-108`); RMSNORM with `FLAG_PER_HEAD` additionally `d.n % h.head_dim` → 5 (`hvx-rmsnorm.c:49-50` chunks by `head_dim`) |
| `nntr_htp_common.h:263-275` (ATTN) | KV size and `max_seq % 64` only | `d.layer >= h.n_layers` → 5 (`hvx-attn.c:58-59` indexes KV by `d->layer` unconditionally) |
| `nntr_htp_common.h:294-304` (LOGITS) | `in0` checked for `m = d.m` (= 1) rows | check `in0` for `h.max_chunk * d.k * 2` bytes regardless of `d.m` (`hvx-matmul.c:350-356` reads row `c->n_tokens - 1`, `n_tokens <= max_chunk` by `htp_graph.c:129`) |
| `nntr_htp_common.h:155-157` | rc comment `5 bad op` | comment lists the new op rules; numbering unchanged |
| `nntrainer/tensor/hexagon/htp/htp_graph.c:127-135` | runtime gate: count, pos, logits length | add: every `(uint32_t)tokens[t] >= g->cfg.vocab` → return 1, before `g->ctx.*` is written (`:137-140`) and before the op loop (`:143-151`); a negative `int32` becomes `>= 2^31 > vocab` by the cast, the same cast `hvx-embed.c:33` uses |
| `nntrainer/tensor/hexagon/htp/executor.c:149-152,189-194` | nonzero → `AEE_EBADPARM` | unchanged; both `nntr_htp_forward` and `nntr_htp_forward_debug` go through `forward_upto`, so both reject |
| `nntrainer/tensor/hexagon/htp/ops/hvx-embed.c`, `hvx-attn.c`, `hvx-matmul.c`, `hvx-rmsnorm.c`, `hvx-eltwise.c` | — | **untouched** (no kernel change; §7 review list trivially satisfied; STAT bit-identity is the proof) |
| `nntrainer/tensor/hexagon/host/hexagon_runner.h` (+ `hexagon_runner.cpp:14`) | — | `constexpr int kHexagonBadParm = 0x8000040E;` in the header (SDK-free by design, `hexagon_backend.h:42-43`), `static_assert(kHexagonBadParm == AEE_EBADPARM)` in the `.cpp`, which already includes `AEEStdErr.h`. `Applications/CausalLM/jni/Android.mk:47-48` adds only `host`/`htp` include dirs, no SDK `incs/stddef`, so the app cannot include `AEEStdErr.h` itself |
| `Applications/CausalLM/hexagon/hexagon_backend.cpp:51-63` | chunk loop, no id check | before the loop: any `tokens[i] < 0 || (uint32_t)tokens[i] >= cfg_.vocab` → return `kHexagonBadParm`; the caller `causal_lm.cpp:381-395` then throws the same `0x8000040e` text a DSP rejection produces |
| `test/hexagon/test_oplist_header.c:130-185` | rc cases 1–5 | new assert cases (§4 step 1); `build_valid` (`:21-70`) stays the base |
| `test/hexagon/sim/test_graph.c:88-113,125-135` | init/forward negatives | new init negatives on mutated copies of the sim-model op-list; forward negatives for bad ids with a KV/ACT byte compare |
| `test/hexagon/sim/test_embed.c:86` | kernel-level harness, calls `hvx_op_embed` directly | unchanged: the check site is `forward_upto`, not the kernel, so `test_graph` carries the negative |
| `docs/backend_guide/HEXAGON.md:143-147` (§1.4), `:115-119` (§1.3) | promises "every tensor ref bounds-checked" | §1.4 rewritten (§6); §1.3 untouched |

Lowering shapes the rules must keep accepting (`Applications/CausalLM/hexagon/qwen3_lowering.cpp`): EMBED `k = hidden` (:125); RMSNORM `n = hidden` (:142, :257, :328), PER_HEAD `n = n_heads*128` / `n_kv_heads*128` (:190, :203); ATTN `layer = l < n_layers` (:223); ADD `n = hidden` (:247, :316); SILU_MUL `n = ffn` (:292); W8A16 `k = ffn` (:304); LOGITS `m = 1, k = hidden, in0 = act_t` sized `max_chunk*hidden*2` (:100, :339-342). `hidden % 64` and `ffn % 64` are already header rules (`:174`), `head_dim == 128` too, so nothing the lowering emits can trip a new rule. The sim model (`test/hexagon/sim/sim_model.c:213-256`, ACT slots sized for `max_chunk` rows per `sim_model.h:29-32`) and the profile shape (`test_profile.c:32-33`: 1024/3072/4096/2048/128) satisfy them the same way.

## 3. Design

**Static properties go to the validator, per-call values to `forward_upto`.** The validator sees the op-list and the buffer sizes once at init (`htp_graph.c:49-51`); everything M2/M5 name is a function of those alone, so init is where the contract (§1.4) says they are refused, and the kernels keep their "validated input" assumption (`hvx-matmul.c:46-48,331-334`). Token ids exist only per call, so they are checked in `htp_graph_forward_upto` next to the other runtime arguments (`:127-135`): one compare per id (≤ 128 per call, negligible against the ~451-op loop), before any buffer pointer or KV row is touched, on the RPC thread, so a failure is an rc and not a worker trap. Rejected: checking inside `embed_worker` (`hvx-embed.c:32-42`) — kernels return `void` and run on pool threads, so the only options there are a silent clamp (wrong logits, invisible) or a trap (the M5 hang). Rejected: checking only in `nntr_htp_forward` (`executor.c:148`) — leaves `forward_debug` and the simulator's direct `htp_graph_forward` callers unguarded.

**LOGITS row: validator fix, not a contract note.** The kernel's read of row `n_tokens - 1` is bounded by `max_chunk`, which the validator already uses as the row count for every `m == 0` op (`:214`). Bounding `in0` by `max_chunk` rows makes the §1.4 promise true for the one op whose `m` (1) is not its read extent; a comment would leave a hand-built list able to read past ACT. The kernel (`hvx-matmul.c:356`) and the reference (`ref_ops.c:299-304`) stay as they are; `d.m` stays unconstrained (the kernel ignores it).

**No new rc number.** rc 5 already means "bad op" (`:155-157`), the host maps every nonzero rc except 3 to `AEE_EBADPARM` (`executor.c:126`), and the FARF at `executor.c:123` prints the rc; a finer number would not reach the host. The x86 table therefore gains cases, not numbers. Rejected: returning the failing op index — changes the validator's return contract for all callers; out of scope.

**Host pre-check on the whole request, not per chunk.** `HexagonBackend::forward` checks all `n_tokens` before the first `runner_->forward`, so a bad id in chunk 3 cannot leave chunks 1–2 appended to KV. The DSP check stays as the last line of defence for other callers (`hexagon_e2e_test`, `hexagon_ref_run` already check at `hexagon_e2e_test.cpp:151`, `hexagon_ref_run.cpp:184`).

## 4. Steps

Commands run from the repo root through `tools/docker/run.sh`; `HEX_ARCH=v75` always explicit.

**Step 0 — branch.** `git checkout -b hvx/33-dsp-bounds hvx_impl`, then `git cherry-pick 0ea50b75 a2213134` (#23's `[tools]` commits: SDK 6.4.0.2 ships `hexagon_toolv88_v75` next to `hexagon_toolv19_v75`, and `hvx_impl`'s lexical glob in `run_sim_test.sh` picks the 8.8 image for a 19.0.04-built library). State the cherry-picks in the PR body; they vanish on rebase once #23 merges. Do not `git add` `build_x86_hexagon/` (ignored only on #23's branch).

**Step 1 — validator + x86 table.** Edit `nntr_htp_common.h` as in §2. Add to `test_oplist_header.c` (each on a fresh `build_valid`; expected rc, then the control):
`ops[1].k = 0` (W8A8) → 5; the same with `kind = W8A16`, `LOGITS` (`n = 32`), `EMBED` → 5; `ops[0].n = 65` (RMSNORM) → 5; `ops[0].flags = PER_HEAD, n = 192` → 5 and `n = 256` → 0 (192 is a multiple of 64, so only the `head_dim` rule can reject it); `ops[0].kind = ADD, n = 65` → 5; `SILU_MUL, n = 65` → 5; `ops[1].kind = ATTN, layer = 1, h.max_seq = 64, buf_size[KV] = 65536` → 5 and `layer = 0` → 0; `ops[1].kind = LOGITS, m = 1, k = 128, n = 32, in0.offset = 7936` (one 256 B row fits in the 8192 B ACT, four do not) → 5 and `in0.offset = 1024` → 0.
Gate 1: `./tools/hexagon/build_host_x86.sh`; `./build_x86_hexagon/test_lowering` → `LOWER_TEST PASS`; `gcc -Wall -Werror -o /tmp/t test/hexagon/test_oplist_header.c && /tmp/t` → `oplist header check: PASS`. A `LOWER_TEST` failure here means a rule is wrong, not the lowering (§2 shape list): fix the rule.

**Step 2 — forward-time id check + host pre-check.** `htp_graph.c`, `hexagon_runner.{h,cpp}`, `hexagon_backend.cpp` as in §2.
Gate 1 again, plus `./build_x86_hexagon/hexagon_ref_run /work/build_x86_hexagon/qwen3_full --tokens /work/build_x86_hexagon/t512.i32 --eval` → `PPL 41.2365 ... top1 161` unchanged (already generated at `4b25ac3c`; `hexagon_ref_run` shares only the header and the lowering, so this proves the reference path is untouched, 160 s). `hexagon_backend.cpp` is compiled only by the Android app build, so add a syntax check with the NDK `clang++` that `tools/hexagon/build_host_test.sh` uses: `-std=c++17 -fsyntax-only -DENABLE_HEXAGON=1 -I nntrainer/tensor/hexagon/host -I nntrainer/tensor/hexagon/htp -I Applications/CausalLM/hexagon Applications/CausalLM/hexagon/hexagon_backend.cpp` (its includes are SDK-free).

**Step 3 — sim negatives.** `test_graph.c`, before the good `htp_graph_init` (`:115-123`), a table-driven loop over mutated copies of `ol` (op indices from `sim_model.c:213-256`: 0 EMBED, layer-0 ops 1–16, 33 final RMSNORM, 34 LOGITS), each expecting `htp_graph_init != 0` and printing `SIM_TEST graph FAIL <case> accepted` otherwise:
`attn_layer`: `ops[8].layer = N_LAYERS`; `k0_w8a8`: `ops[2].k = 0`; `k0_w8a16`: `ops[15].k = 0`; `k0_logits`: `ops[34].k = 0`; `k0_embed`: `ops[0].k = 0`; `rmsnorm_n65`: `ops[1].n = 65`; `perhead_n192`: `ops[5].n = HEAD_DIM + 64`; `add_n65`: `ops[10].n = 65`; `silu_n65`: `ops[14].n = 65`; `logits_in0_short`: `ops[34].in0.offset = P.atotal - HIDDEN * 2` (one row fits, `MAX_CHUNK` rows do not; 128-aligned).
In the forward-negative block (`:125-135`, KV/ACT and their reference copies are still all-zero, `:84-86`): `{VOCAB}` n=1, `{-1}` n=1 (= 0xFFFFFFFF), and the 8-token prefill with `VOCAB` in the last slot must each return nonzero (`SIM_TEST graph FAIL bad token id accepted`), and afterwards `memcmp(kv, rkv, KV_BYTES) == 0 && memcmp(act, ract, P.atotal) == 0` (`SIM_TEST graph FAIL rejected forward touched KV/ACT`). Id `VOCAB - 1` (511) already runs in `prefill_tok` as the accepted boundary. `test_embed.c` unchanged.
Gate 2: `HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/build_sim_test.sh`; `run_sim_test.sh graph` → `SIM_TEST graph PASS` with `graph_prefill STAT 0.0245416/9.3795`, `graph_decode 0.0202219/23.975` unchanged; `run_sim_test.sh embed` → `SIM_TEST embed PASS`; `run_sim_test.sh profile acc` → `SIM_TEST profile PASS`, `profile_prefill_acc STAT max_abs=0.077216 max_rel=98.2637` (bit-identical; ~10 min).
Gate 3: the 13 tests `smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed attn logits graph` all PASS (~4 min), `quant_generic 0/65536`, `quant16_generic 167/25600` as in §8.3.
Gate 4: `HEX_ARCH=v75 ./tools/hexagon/build_skel.sh` and `./tools/hexagon/build_host_test.sh` build (both touch changed files: `htp_graph.c`, `hexagon_runner.cpp`); no md5 record needed, no handoff.

**Step 4 — docs** (§6). Gate: none beyond review.

**Step 5 — format and PR.** Gate 0: `tools/docker/run.sh clang-format-14 -i` on the changed `.c/.cpp/.h`; `git diff --stat` shows only intended files (do not touch `test_oplist_header.c:81`'s Korean comment beyond the line if the formatter leaves it; #34 owns it). One `[htp]` commit for validator + graph check + tests, one `[hexagon]` for the host pre-check, one `[docs]`; `git commit -s`. PR into `hvx_impl` with the gate lines pasted; issue → `state:review`. **No measurement handoff: nothing here is a performance or silicon question.**

## 5. Risks

* **A rule rejects the real lowering.** Mitigated by construction (§2 shape list) and by order: gate 1's `test_lowering` validates both the tiny and the qwen3-0.6b dims on x86 before any sim run; `profile acc` then validates the qwen3-2L sim model. If either fails, the rule is wrong.
* **Kernel-level sim harnesses build descriptors by hand** (`test_rmsnorm.c:57,91`, `test_eltwise.c:21`, `test_logits.c:60-73`) and never call the validator; they are unaffected, and the forward-time check lives only in `htp_graph.c`, so only `graph`/`profile` see it.
* **#23 (SDK 6.4, needs-measurement).** No source overlap; the two cherry-picked `[tools]` commits are identical patches and drop out on `git rebase hvx_impl` after #23 merges. Docs: #23 edits HEXAGON.md §5.2/§8.3/§9, this plan edits §1.4 — disjoint hunks. Do not wait for #23's measurement.
* **#35 (v75 skel default, IEEE guard).** Disjoint files (`build_skel.sh`, `hvx-base.h`); the v79 sim failures (`quant matmul matmul_dma logits`, §5.2) are pre-existing and every gate here is `HEX_ARCH=v75`. No interaction.
* **Pending history-clean replacement of `hvx_impl` (#34).** `git diff hvx_impl hvx/history-clean` is empty (trees identical, verified 2026-09-17). If the user force-replaces `hvx_impl` before this PR merges: `git rebase --onto hvx_impl 25067f02 hvx/33-dsp-bounds` is conflict-free; keep the PR base `hvx_impl`, never target `hvx/history-clean`.
* **AEE mapping not observable on the simulator.** `executor.c` is not in the sim build (`build_sim_test.sh:28-33`); the `AEE_EBADPARM` value is guaranteed by the unchanged `executor.c:149-152` and the `static_assert` in `hexagon_runner.cpp`. Acceptable: no rc convention changes.
* **Host pre-check compiles only in the Android app build.** Kept to a loop and one constant, no new includes; the `-fsyntax-only` check in step 2 is the compile gate.
* Simulator-vs-device gaps: none exposed — no kernel bytes change, no timing claim is made.

## 6. Docs to update

* `docs/backend_guide/HEXAGON.md` §1.4 (`:143-147`): replace "`init()` validates everything up front ... `forward()` only checks runtime arguments (token count ≤ `max_chunk`, position < `max_seq`, logits length)" with the full list: header rules (`:174-177`), per-op rules incl. `k > 0` and `k % 128` for the four int8 kinds, `n % 32` tiled kinds, `n % 64` RMSNORM/ADD/SILU_MUL, PER_HEAD `n % head_dim`, `ATTN.layer < n_layers`, LOGITS `in0` sized for `max_chunk` rows, every ref bounds-checked; `forward()` additionally rejects any token id `>= vocab` with `AEE_EBADPARM` before touching KV/ACT, and the host pre-checks in `HexagonBackend::forward`. Add one line to the "v4 additions" paragraph (`:148-159`) that these rules were added after the 2026-09-17 review without an ABI bump.
* HEXAGON.md §1.3 (`:115-119`): unchanged (state so in the PR).
* HEXAGON.md §5.2 (`:440-`): in the `graph` description, mention the init negatives (shape rules) and the token-id forward negatives; the test count stays 13.
* `HEXAGON_BENCHMARK.md`: no row changes (no performance claim).
