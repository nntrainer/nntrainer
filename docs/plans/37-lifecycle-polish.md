# 37 — Host/DSP lifecycle polish (m1, m2, m4–m8, m13; m3 already closed)

Issue: dlwlzzero/nntrainer#37 (`prio:p2`, from the 2026-09-17 review).
Contract: `docs/plans/0000-agent-system-and-env.md`. Base: `hvx_impl` @ `5bc88578`
(includes #24 host logits, #25 prefetch, #72 / #65 S0 HexKL). Planned 2026-09-21.

Re-grounded against the current tree, not the issue's `8686d520` snapshot:

| Item | Issue's site | Today | Status |
|---|---|---|---|
| m1 delayed-fd `fastrpc_munmap` | `hexagon_runner.cpp:52-56` | `map_on_dsp` `:79-83`, `init` loop `:89-96`, `~HexagonRunner` `:67-74` unmaps only `static_maps_` | **open** |
| m2 alloc failure → `AEE_EBADPARM` | `htp_graph.c:61-63, 74-87` | `wp_create` `:127-129`, scratch `:164-168`, DMA queues `:244-247` all `return 1`; `executor.c:122-127` maps every rc but 3 to `AEE_EBADPARM` | **open** |
| m3 per-call DMA queue | `hvx-matmul.c:141-146, 178-179` | queues are graph-lifetime since `11a42bb1` (#25): `htp_graph_dma_init` `htp_graph.c:38-68`, `c->dmaq[wid]` in `mm_worker_vtcm` `hvx-matmul.c:236` | **closed by #25**, nothing to do |
| m4 scale over-read | `hvx-matmul.c:267` | `mm16_block` `hvx-matmul.c:424` `hvx_vmemu(sw)` reads 128 B for a 4-lane slice; callers `:476-480` pass `sw + jn` | **open** |
| m5 `Cursor::alloc` truncation | `qwen3_lowering.cpp:28-33` | unchanged, `:25-38` | **open** |
| m6 `.hexcfg` parsing | `hex_image.cpp:57-59, 81-82, 44-49, 24-28` | unchanged (same lines) | **open** |
| m7 silent `return` on `n % 32` | `hvx-matmul.c:333-334, 347-348` | `:498-499` (`hvx_op_matmul_w8a8`), `:512-513` (`hvx_op_matmul_logits`) | **open** |
| m8 `qurt_hvx_lock` unchecked | `worker_pool.c:44` | unchanged | **open** |
| m13 no session lock | `executor.c:135-158, 165-` | `nntr_htp_forward` `:143-171`, `nntr_htp_forward_debug` `:173-205`, `nntr_htp_init` `:77-141` share `s->graph.ctx` | **open** |

## 1. Goal and gate

| # | Acceptance (issue) | Pass means |
|---|---|---|
| G1 | m1, m2, m7, m8, m13 fixed | code review against §3 + `hexagon_rpc_test` / `hexagon_e2e_test` PASS on device (G5); the new `RPC_TEST reopen x8 ok` line (§4 step 2) |
| G2 | m4 fixed without moving numerics | `run_sim_test.sh matmul` (v79) STATs bit-identical: `matmul_w8a16_m1 0/0`, `m2 0/0`, `m8 0.0078125/0.000974659` (HEXAGON.md §5.2 v79 record); cases `(7,3072,100)` and `(2,128,6)` of `test_matmul.c:137,141` exercise the tail path |
| G3 | m5, m6 with x86 negatives | `LOWER_TEST PASS` with the new checks of §4 step 1; `qwen3_full.hexw` md5 unchanged after re-packing, `.hexcfg` gains exactly one `abi=4` line, `hexagon_ref_run --eval` PPL/top-1 identical to the HEXAGON.md §5.1 figure for the prompt used |
| G4 | 13 sim tests + `profile acc` unchanged | v79: 13 × `SIM_TEST <name> PASS` (+ `hmx` with HexKL), `profile_prefill_acc STAT max_abs=0.0659682 max_rel=92.7791`, `graph_prefill 0.0218946/6.88818`, `graph_decode 0.0197323/23.2374` (§8.3 #25/#68 v79 record); a moved per-op pcycle count is recorded in §8.3 as a change, not a bug |
| G5 | device regression, one handoff, no performance claim | `hexagon_rpc_test`: `bad-version rejected ok`, `init ok`, `reopen x8 ok`, all rows PASS; `hexagon_e2e_test --eval` on `t512.i32`: PPL / top-1 inside the §8.1 device band; the table repeats the skel md5 |
| G6 | no ABI change | `NNTR_HTP_ABI_VERSION` stays 4; no op-list, image byte or IDL change |

`check_count`: no gtest added or removed (all new checks live in `test_lowering.cpp`, `hexagon_rpc_test.cpp` and the sim library).

## 2. Where it lives

Two branches, stacked, so the host-only half needs no simulator run (contract §7):

**PR A `hvx/37-host-lifecycle` (no DSP bytes)**

| File:line | Item | Change |
|---|---|---|
| `nntrainer/tensor/hexagon/host/hexagon_runner.h:99-105` | m1 | second `std::vector<StaticMap> delayed_maps_` (same struct) |
| `nntrainer/tensor/hexagon/host/hexagon_runner.cpp:79-96` | m1 | `map_on_dsp` returns whether *this* call mapped it; `init` records `{fd,data,size}` only on `AEE_SUCCESS` (an `AEE_EALREADY` for an fd already in `delayed_maps_` is this runner's re-init; for an unknown fd it is foreign and is neither recorded nor unmapped) |
| `hexagon_runner.cpp:67-74` | m1 | destructor order: static munmaps → `nntr_htp_close` (DSP `HAP_munmap`, `executor.c:70-71`) → delayed munmaps. The failed-init path (`:101-112`) needs nothing: both callers destroy the runner (`hexagon_runner.h:34`, `hexagon_backend.cpp:43-46`) and `HexagonBackend` declares the buffers before `runner_` (`hexagon_backend.h:48-49`), so they outlive the unmap |
| `Applications/CausalLM/hexagon/qwen3_lowering.cpp:25-38` | m5 | `Cursor::alloc` throws `std::overflow_error` when `align128(cur_) + bytes > UINT32_MAX` |
| `Applications/CausalLM/hexagon/hex_image.cpp:20-36` | m6 | `write_hexcfg` adds `abi=<NNTR_HTP_ABI_VERSION>` (include `nntr_htp_common.h`; `build_host_x86.sh` already passes `-I htp`) |
| `hex_image.cpp:38-84` | m6 | strip a trailing `\r` at `:44-49`; `parse_u32` / `parse_f32` helpers (`errno`, `endptr`, empty and trailing text rejected, key name and value in the message) replace `:57-59` and `:81-82`; `abi` check next to `weight_layout` (`:63-69`) |
| `Applications/CausalLM/hexagon/hex_image.h:23-26` | m6 | comment: 13 lines, `abi=` rule |
| `test/hexagon/test_lowering.cpp:677-719` (after), `:756` | m5, m6 | negatives (§4 step 1) |
| `test/hexagon/hexagon_rpc_test.cpp:57-63` | m1, m13 | `reopen x8` loop (create → init → one forward → destroy) after `init ok` |

**PR B `hvx/37-dsp-lifecycle` (DSP bytes; branched from A)**

| File:line | Item | Change |
|---|---|---|
| `nntrainer/tensor/hexagon/htp/htp_graph.h:33-39` | m2 | `#define HTP_GRAPH_RC_NOMEM 6` (validator uses 1–5, `nntr_htp_common.h:291-407`); doc the rc set |
| `htp_graph.c:127-129, 164-168, 244-247` | m2 | `return HTP_GRAPH_RC_NOMEM` |
| `executor.c:122-127` | m2 | `rc == 3 → AEE_EUNSUPPORTED; rc == HTP_GRAPH_RC_NOMEM → AEE_ENOMEMORY; else AEE_EBADPARM` |
| `executor.c:23-33, 58-75, 77, 143, 173` | m13 | `qurt_mutex_t lock` in `struct session`; init in `nntr_htp_open`, destroy in `nntr_htp_close`; lock/unlock around the bodies of `init`, `forward`, `forward_debug` (every return path — one early-out helper or a single-exit shape) |
| `worker_pool.c:20-55, 57-118` | m8 | `ready_sem` + per-worker `hvx_ok`; `worker_main` reports its lock result before the job loop; `wp_create` waits `n` readies and runs the existing rollback (`:98-115`) when any worker failed → `NULL` → `htp_graph_init` `HTP_GRAPH_RC_NOMEM` → host `AEE_ENOMEMORY` with `FARF(ERROR, "wp: qurt_hvx_lock failed (worker %d, rc %d)")` |
| `ops/hvx-matmul.c:401-445, 476-480` | m4 | `mm16_block` gains `bool tail` (or `n - jn < 32`) and loads the scale slice from a 128 B-aligned zeroed stack copy of `rows` floats when the 128 B window would cross `sw + n`; the in-range case keeps `hvx_vmemu` |
| `ops/hvx-matmul.c:498-499, 512-513` | m7 | `FARF(ERROR, "matmul: n=%u not a multiple of %u, op skipped", ...)` before the `return` (`HAP_farf.h` already reaches this file through `dma-queue.h:18`) |
| `test/hexagon/sim/test_pool.c` | m8 | nothing new is expressible (the lock cannot be made to fail on the simulator); the existing pool/graph tests are the regression |

ABI: **no change** (`nntr_htp_common.h:18-19` stays `4u`). The `abi=` line is a `.hexcfg` addition
read by `read_hexcfg` only; `find_divergence.py:50-53` parses `n_layers=` by prefix and ignores
other lines; `hexagon_e2e_test.cpp:174` and `hexagon_ref_run.cpp:139` go through `read_hexcfg`.
A missing `abi` key is read as 4 (every keyless image is v4: `weight_layout` exists only since v4,
HEXAGON.md §2.4), so no device or x86 image has to be repacked; an `abi` that differs from
`NNTR_HTP_ABI_VERSION` is rejected. When #65 S1 bumps to v5 this becomes the "v4 image on a v5
host" rejection S1's plan asks for (`65-w4-htp-port.md` §2.2, "read_hexcfg rejects v4 images")
with no further code.

**Conflicts to note.** #65 S1 (`hvx/69-*`, next in this cycle) owns `qwen3_lowering.cpp:74-88`
and the emits, `hex_image.cpp:28,63-68`, `test_lowering.cpp`. PR A touches `qwen3_lowering.cpp:25-38`
(the `Cursor` class only), `hex_image.cpp` inside the same two functions, and appends to
`test_lowering.cpp` `main`. Whichever merges second rebases; the hunks do not overlap in lines
but do in files. #38 later *moves* `hex_image.{h,cpp}` into core; m6 changes content only, so
#38 rebases as a rename. #40 is untouched (`causal_lm.cpp`, outside the subtree).

## 3. Design

**m1** — record and pair. Alternative rejected: unmap inside `init()`'s failure path; it would
double-unmap when the caller destroys the runner anyway, and it does not cover the success path.
Order matters: the DSP still holds `HAP_mmap` views until `nntr_htp_close`, so the delayed
unmaps come after close; the static logits map has no DSP view and stays before close.

**m2 / m8** — one new rc, resource failures become `AEE_ENOMEMORY`; a worker that cannot lock
an HVX unit makes `wp_create` fail (the pool never runs a job on a context-less worker, and
`wp_run`'s barrier is never entered with a short pool). Alternative rejected: `abort()` in the
worker — kills the PD, the host sees a connection reset and loses the AEE code and the FARF.

**m4** — bound the load in the kernel, lanes 0..3 unchanged, so every STAT stays bit-identical
(lanes 4..31 are never stored: `hvx_vec_store_u(y, 2*rows, ...)`, `hvx-matmul.c:443`). The
branch runs only on the last block of a worker's range (`jn + 32 > n`), so the per-call cost is
a few scalar stores per op. Alternative rejected: pad `*_s` allocations in `lower_qwen3` — moves
the image layout (ABI bump, `.hexw` md5, S1 conflict) to fix a kernel-side read bound; the
executor already keeps its own read pads in its own scratch (`attn_scratch + 128`,
`htp_graph.c:160-163`), and this keeps that rule (HEXAGON.md §7 rules: qf-format ops only,
tie-row and NaN rules untouched — the epilogue arithmetic does not change).

**m13** — serialize with a `qurt_mutex` rather than document only: an uncontended lock costs
nanoseconds against a 34 ms step, and the failure it prevents (two callers sharing `c->xq`,
`c->pf`, the KV append) is silent corruption. `close` is not locked: destroying a session with a
call in flight is a host bug documented in §1.2 ("one runner, one owner").

**m5 / m6 / m7** — as the issue states; `Cursor` throws on the *end* of the allocation
(`start` alone can be < 2³² while the tensor crosses it). The host `init` also casts
`weights.size()` to `int` (`hexagon_runner.cpp:98`; IDL `sequence<uint8>`), so the practical
WEIGHTS ceiling is 2 GiB — a doc line in §2.1, not a code change here.

## 4. Steps

1. **PR A, m5 + m6 + tests** (rung 0, 1). `test_lowering.cpp` negatives: `lower_qwen3` with
   `vocab = 1u << 20, hidden = 4096` throws; `.hexcfg` variants through the existing `rejects`
   lambda: CRLF body **accepted** and round-trips, `n_layers=abc` / `n_layers=` / `n_layers=12x`
   rejected with `what()` containing `n_layers`, `rms_eps=1e` rejected, `abi=5` rejected, body
   without `abi=` accepted, written file contains `abi=4\n`. Gate: `build_host_x86.sh`,
   `test_lowering` → `LOWER_TEST PASS`; `nntr_hexpack` → `.hexw` md5 identical, `--eval` PPL
   identical (G3).
2. **PR A, m1 + reopen loop** (rung 0, 4). Gate: `build_host_test.sh` produces both harnesses;
   `git diff hvx_impl --stat -- nntrainer/tensor/hexagon/htp` empty ("no DSP bytes changed").
   Open PR A (`state:review` stays on the issue until PR B).
3. **PR B, m2 + m8** (rung 0; per-step `run_sim_test.sh pool` and `graph`). Gate: both PASS,
   `test_graph.c`'s 12 rc-5 negatives still return 5.
4. **PR B, m4 + m7** (per-step `run_sim_test.sh matmul`). Gate: G2 STATs.
5. **PR B, m13** (rung 4 only; `executor.c` is not in the sim library, `build_sim_test.sh:31-35`).
   Gate: `build_skel.sh` links.
6. **Once before PR B:** rung 2 `profile acc` + rung 3 the 13 tests (+ `hmx`), v79 → G4. Then
   rung 4 skel + harness, record md5s (`dsp_obj_md5.sh` for the objects).
7. **Device (unavoidable, one handoff, one skel):** `docs/measurements/37-lifecycle.md` per the
   `hexagon-handoff` skill: `run_device_test.sh` (`hexagon_rpc_test`: bad-version, init, reopen
   x8, transport rows) and `run_e2e_test.sh ... --tokens /tmp/t512.i32 --eval` with the PR B
   skel; result table = PASS/FAIL per row + `E2E ppl / top1` + `E2E decode median_us` (accuracy
   and runtime paired, no goal comparison) + skel md5. No variants, no sweeps. Label
   `state:needs-measurement`; on `state:measured` with all rows PASS, the supervisor closes on
   PR B's merge.

## 5. Risks

* **Device-only behaviour** (m1 order of `fastrpc_munmap` vs `nntr_htp_close`, m13 mutex, m8 lock
  failure): the simulator cannot exercise any of them. The handoff's `reopen x8` row and a
  `logcat -s adsprpc` grep for `fastrpc_munmap`/`HAP_munmap` errors make a wrong order visible;
  m8 stays code-review only (it cannot be provoked on either target).
* **qf32 numerics:** m4 changes a load, not arithmetic; the G2/G4 STAT identity is the proof.
  If any STAT moves, the change is wrong, not "tuning".
* **Bandwidth:** no path changes; the §8.3 per-op pcycles may drift by code layout — recorded as
  such (G4), never as a performance claim.
* **Stale artifacts:** two `.hexcfg` shapes now coexist (with / without `abi=`); the grandfather
  rule keeps old images valid, and the handoff repeats the `.hexcfg` md5 and the skel md5.
* **Branch conflicts** with #65 S1 (`hex_image.cpp`, `qwen3_lowering.cpp`, `test_lowering.cpp`):
  small, line-disjoint hunks; the second merger rebases (§2).

## 6. Docs to update

* HEXAGON.md §1.1 (`:99-136`): the delayed maps are unmapped by the runner after close;
  `AEE_EALREADY` rule restated with ownership. §1.2 (`:137-143`): calls on one handle are
  serialized on the DSP by a session mutex; close must not race a call. §1.3 (`:145-149`): the AEE
  code table — `EUNSUPPORTED` version, `EBADPARM` op-list/runtime args, `ENOMEMORY`
  mmap/alloc/HVX-lock, `EBADSTATE` forward before init. §2.1 (`:243-`): no scale padding; the
  W8A16 kernel bounds its last block itself; layout limit 2³² (Cursor) and the 2 GiB IDL
  ceiling. §2.4 (`:400-422`): 13 lines, `abi=`, missing-key rule, strict parsing. §3 (`:458`):
  `test_lowering` line mentions the negatives. §8.3: one line if `profile acc` pcycles moved.
* Suggested §7 rule (supervisor's section): "a kernel never reads past a tensor's validator
  extent; read pads live in executor scratch, not in the image".
* HEXAGON_BENCHMARK.md: no rows (no performance claim).
