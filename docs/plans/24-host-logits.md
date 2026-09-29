# 24 — Host/RPC decode path: stop copying 151,936 logits per step (ledger ⑫)

Issue: dlwlzzero/nntrainer#24 (`prio:p0`). Ledger: ⑫. Contract: `docs/plans/0000-agent-system-and-env.md`.
Branch: `hvx/24-host-logits` from `hvx_impl` (= `history-clean-candidate`, `f66bf3b`).
Written without SDK/container/device; every `path:line` below was read on that tree, and gate 1 was run on it (`LOWER_TEST PASS`, `oplist header check: PASS`).

## 1. Goal and gate

Issue AC, verbatim: *decode host wall per step @512 within 5 ms of the DSP pcycle time; generated ids identical to the current path for greedy decoding; `--eval` PPL unchanged; `hexagon_rpc_test` passes; ABI bump documented in HEXAGON.md §1.4 and `.hexcfg` rejects old images if the ACT layout changes.*

**What the lever is worth today (re-read of HEXAGON.md §8.2 as amended by #23, 2026-09-17, v75 skel, `--chunk 128 --steps 64`).** The harness times only the RPC (`hexagon_e2e_test.cpp:163-165`: `t0 … runner->forward … us`); argmax and `log_softmax_at` are outside the timed region (`:232,238`, `:210-212`). Host wall vs the op loop at the 2.09 GHz top clock the cDSP holds while hammered (§8.2 M5 note, `HEXAGON.md:802`):

| ctx | host median ms | DSP Mcyc | DSP ms @2.09 GHz | gap ms | pcycles÷us (clock lower bound) |
|---|---|---|---|---|---|
| 512 | 31.3 | 64.3 | 30.8 | **0.5** | 2.054 GHz |
| 1024 | 43.2 | 84.2 | 40.3 | **2.9** | 1.949 GHz |
| 4096 | 104.1 | 213.0 | 101.9 | **2.2** | 2.046 GHz |

P4 generation mode said the same (59.5 M = 28.5 ms vs 31.2 ms host, gap 2.7 ms, `HEXAGON.md:899-902`). The "~40 ms" in the issue is P3 generation (35 vs 76.4 ms), which §8.2 already marks as not comparable, plus the teacher-forced `--eval` rows (P3 19 ms, P4 17.3 ms gap) — there `pcycles÷us` is **1.15 GHz**: the DSP runs at about half clock because the host spends ~10 ms between RPCs in 151,936 double `exp()` calls, an idle-gap/DCVS effect, not a copy cost. So **the AC's 5 ms bound is already met by the current code in the benchmark's own method**, the lever's ceiling is 0.5–3 ms of a 31–104 ms step (≤ 2–7 %), and the stage-1 goal (70.3 tok/s = 14.2 ms/step) is below the DSP op loop alone (30.8 ms): #24 cannot deliver the goal; ① ⑨ ⑩ (the 598 MB weight path) must. What forward() actually moves: 4 B of token ids in, 607,744 B of fp32 logits out as `rout sequence<float>` (`nntr_htp.idl:30-31`) into **malloc memory** (`hexagon_e2e_test.cpp:157`, `causal_lm.cpp:385`), which the FastRPC library/driver serve through the staging-copy path; rpcmem allocations take the ION zero-copy path instead (§1.1's "shared zero-copy" is only used for WEIGHTS/KV/ACT today).

The plan therefore (i) removes the copy in the cheapest legal form — candidate (c) reduced to *a host-side rpcmem logits buffer, no ABI change* — and (ii) turns the "40 ms" into a measured, component-resolved number that closes ⑫ and corrects the docs. (a)/(b) are contingencies with a stated trigger.

| # | Gate | Pass means |
|---|---|---|
| G1 transport | `hexagon_rpc_test` (n_ops == 0 dummy path) with `logitsLen` 8 vs 151,936, host memory malloc / rpcmem / rpcmem+STATIC | `RPC_TEST forward_full_us mem=<m> <us>` medians recorded; `full(rpcmem) − forward_us(8)` = cost of the 607,744 B return. `RPC_TEST PASS`, `check_rpc_log.py` VERDICT PASS |
| G2 gap | e2e @512 `--chunk 128 --steps 64`, shipped variant | new line `E2E decode steps=63 median_us=<u> median_pcycles=<c> pcycles_per_us=<r>`; `u − c/2090` ≤ 5 ms (expected ≈ 0.5 ms); `r` ≥ 2.05 (→ gap < 1 ms whatever the true top clock) |
| G3 ids / PPL | same 512 run, three memory variants; `--eval` 512 malloc vs rpcmem | `E2E gen …` byte-identical across A/B/C and to #23's A log; `E2E ppl 33.0884 steps 511 top1 189` (t512.i32, #23 A row) identical in both — the DSP writes the same bytes, only the return path changes |
| G4 goal rows | 512 / 1024 / 4096 with the shipped variant | tok/s within #23's noise (31.9 / 23.2 / 9.6; 512 carries ±5 %); rows go to HEXAGON_BENCHMARK.md |
| G5 x86 + sim | rungs 1–4 | `LOWER_TEST PASS`, oplist `PASS`; 13 × `SIM_TEST … PASS`, `profile_prefill_acc STAT max_abs=0.077216 max_rel=98.2637` bit-identical (no DSP byte changes); skel + harness build, md5 recorded |
| G6 docs | §6 | HEXAGON.md intro/§1.1/§8.2/§9, benchmark lever text, ledger ⑫ closed with the G1/G2 numbers |

**No ABI change**: `NNTR_HTP_ABI_VERSION` stays 4 (`nntr_htp_common.h:18-19`), IDL (`nntr_htp.idl:30-31`), validator, `.hexcfg`, `lower_qwen3`, `ref_graph_forward`, sim `graph` test, `find_divergence.py`, `nntr_hexpack` all untouched; `E2E init ok … act=3932160 n_ops=451` must print unchanged as the proof.

## 2. Where it lives

| File:line | Today | Change |
|---|---|---|
| `test/hexagon/hexagon_e2e_test.cpp:157` | `std::vector<float> logits(cfg.vocab)` (malloc) | `--logits-mem malloc\|rpcmem\|static` (default `rpcmem`): `RpcmemBuffer` of `vocab*4` B; `static` additionally registers it via the new runner helper below |
| `hexagon_e2e_test.cpp:160-173, 200-219, 224-243` | per-step `E2E step … us=… top1=…` | unchanged format; collect `us`/`pcycles` of the `n=1` steps after the prompt (skip the first) and print `E2E decode steps=<n> median_us=<u> median_pcycles=<c> pcycles_per_us=<r>` before `E2E wall_ms` / after `E2E ppl` (also in `--eval`, where `r` exposes the idle-gap clock for free) |
| `test/hexagon/hexagon_rpc_test.cpp:56-69, 73-78` | 32 × `forward` with 8 logits, pattern check | keep as is (parser markers); add 3 × 32 iterations with `n_logits = 151936` over `{malloc, rpcmem, static}` printing `RPC_TEST forward_full_us mem=<m> <us>`, pattern-check elements 0 and 151935 (`executor.c:156-162` handles any `logitsLen`), then `RPC_TEST PASS` |
| `tools/hexagon/check_rpc_log.py:55-62, 80-85` | required markers, `forward_us` stats | markers unchanged; add per-`mem` min/median for `forward_full_us` lines (informational) |
| `nntrainer/tensor/hexagon/host/hexagon_runner.h:43-45`, `hexagon_runner.cpp:117-121, 154-163` | `map_on_dsp` = `FASTRPC_MAP_FD_DELAYED` for init buffers; `forward` passes the caller's pointer | add `int HexagonRunner::register_static(const RpcmemBuffer &)` = `fastrpc_mmap(CDSP_DOMAIN_ID, fd, data, 0, size, FASTRPC_MAP_STATIC)` (flag name checked against `$HEXAGON_SDK_ROOT/incs/remote.h` in step 0; `AEE_EALREADY` → 0 as at `:120`); `forward` unchanged |
| `host/rpcmem_allocator.cpp:21` | `RPCMEM_DEFAULT_FLAGS` (CPU-cached) | unchanged: the driver does the cache ops for RPC arguments, which is exactly why the buffer stays an argument (§3) |
| `Applications/CausalLM/hexagon/hexagon_backend.h:102-110`, `hexagon_backend.cpp:31-33, 51-63` | three rpcmem buffers; `forward` writes into the caller's `float *` | own `logits_` (`RpcmemBuffer(vocab*4)`), RPC into it, `memcpy` to the caller's buffer (≈ 0.1 ms host, keeps `causal_lm.cpp:385-388,654` ownership untouched); signature unchanged |
| `nntrainer/tensor/hexagon/htp/{nntr_htp.idl,executor.c,htp_graph.c:132-133,ops/hvx-matmul.c:84-90,343-361}` | rout logits, `n_logits == vocab`, unaligned 128 B fp32 stores | **untouched** (rpcmem is page-aligned; the kernel already stores unaligned) |
| `tools/hexagon/build_host_test.sh:49-52` | source lists | unchanged |
| Docs | `HEXAGON.md:22-26, 89-90, 853-857, 899-902, 1145-1149`; `HEXAGON_BENCHMARK.md:27-29, 40`; `07-follow-ups.md:66-68` | §6 |

## 3. Design

**(c0): keep the `rout`, make its memory rpcmem.** The FastRPC user library recognises a pointer inside an `rpcmem_alloc` buffer and passes its fd instead of staging a copy; the kernel driver then only maps (once, with `FASTRPC_MAP_STATIC`; per call otherwise) and does cache maintenance around the call. That gives the issue's "rpcmem slot the host reads zero-copy" with zero DSP, IDL, validator or image change, keeps `--eval`, sampling (`causal_lm.cpp:331-337`) and `forward_debug` intact, and keeps greedy ids bit-identical by construction (same kernel, same bytes, same host `argmax`). Two knobs are measured, not assumed: rpcmem vs malloc (the copy) and STATIC vs per-call mapping.

**Measure the transport in isolation before the graph.** `hexagon_rpc_test` runs the n_ops == 0 dummy path (`executor.c:111,156-162`): the DSP-side fill of 151,936 floats is a scalar loop (~0.3 ms, identical across variants), so `forward_full_us(mem) − forward_us(8 floats)` is the cost of returning 607,744 B through each path — the number the issue never had. The e2e `E2E decode …` summary then shows what survives in the real step.

**Decision rule after `state:measured`.** (1) Ship `rpcmem` (STATIC too if it beats plain rpcmem by ≥ 0.1 ms) — it is free and correct even if the saving is tens of µs. (2) If `forward_full_us(static)` still costs ≥ 2 ms over the 8-float call, the transport is real → follow-on plan for (a) below. (3) If the e2e gap exceeds 5 ms while G1 shows the transport < 1 ms, the residual is not the logits: file "DSP clock during host idle gaps" (see §5) instead of touching the ABI. (4) If rpcmem == malloc in G1, libadsprpc did not recognise the buffer: add `remote_register_buf_attr(data, size, fd, 0)` in `RpcmemBuffer` and re-issue the rpc_test rows only.

**Rejected: (c) as written — LOGITS into an ACT slot, IDL `forward()` without the rout (ABI v5).** It removes only the per-call handling of one buffer, but moves coherency from the driver to us: a DSP `qurt_mem_cache_clean` after `MATMUL_LOGITS` plus a CPU-side invalidate that rpcmem does not expose for a buffer that is not an RPC argument; the workaround (`RPCMEM_FLAG_UNCACHED` CPU mapping) makes the host's 600 KB read slower than the copy it replaces. It would also bump the ABI, grow `act_size` (+607,744 B), and touch `lower_qwen3`, the validator's LOGITS case (`nntr_htp_common.h:294-303`), `ref_graph_forward`, `test_graph`, `hexagon_ref_run --dump-op`, HEXAGON.md §1.4/§2.2 — for a gain G1 is expected to show is sub-ms.

**Contingency (a), only on trigger (2):** an *additional* IDL method `forward_topk(in token_ids, in pos, in k, rout sequence<float> v, rout sequence<int32> i, rout pcycles)`, ABI v5, DSP top-k on the same fp32 logits with lowest-index tie-break (matches `argmax()` `>` in both harnesses and `std::max_element`), `k == 1` = (b). `forward()` stays for `--eval`/sampling; `.hexcfg` unchanged (no image layout change). Not planned here.

## 4. Steps

Commands from the repo root through `tools/docker/run.sh`; `HEX_ARCH=v75` explicit. Branch `hvx/24-host-logits` from `hvx_impl`.

**Step 0 — probe the SDK headers (no source change).** `grep -n "FASTRPC_MAP_STATIC\|FASTRPC_MAP_FD_DELAYED\|remote_register_buf" $HEXAGON_SDK_ROOT/incs/remote.h; grep -n "RPCMEM_DEFAULT_FLAGS\|RPCMEM_FLAG" $HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc/rpcmem.h`. Gate: both flag names printed; paste into the plan's step-0 record. If `FASTRPC_MAP_STATIC` is absent, drop the `static` variant (two variants remain).

**Step 1 — harness + runner + backend (host only).** Edit per §2. Gate 1: `./tools/hexagon/build_host_x86.sh`, `./build_x86_hexagon/test_lowering` → `LOWER_TEST PASS`, `gcc -Wall -Werror … test_oplist_header.c` → `oplist header check: PASS` (proves nothing below the host moved; the x86 reference PPL 33.0195/184 is untouched by construction — no packer/lowering/`ref_ops` edit). `hexagon_backend.cpp` is app-build-only: NDK `clang++ -std=c++17 -fsyntax-only -DENABLE_HEXAGON=1 -I nntrainer/tensor/hexagon/host -I nntrainer/tensor/hexagon/htp -I Applications/CausalLM/hexagon -isystem $HEXAGON_SDK_ROOT/incs -isystem $HEXAGON_SDK_ROOT/incs/stddef -isystem $HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc Applications/CausalLM/hexagon/hexagon_backend.cpp`.

**Step 2 — simulator (unchanged bytes, ladder rungs 2–3 once).** `build_sim_test.sh`; `run_sim_test.sh profile acc` → `SIM_TEST profile PASS`, STAT `0.077216/98.2637`; the 13 tests → 13 PASS, `graph_prefill 0.0245416/9.3795`, `graph_decode 0.0202219/23.975`, `quant_generic 0/65536`, `quant16_generic 167/25600`. Any STAT movement means a DSP file was touched by mistake.

**Step 3 — skel + harness (gate 4).** `HEX_ARCH=v75 build_skel.sh` → `libnntr_htp_skel.v75.so`; `build_host_test.sh` → `hexagon_rpc_test`, `hexagon_e2e_test`; `md5sum` all three. The harness md5 must differ from #23's `abb4fb72…`; the skel is a rebuild of unchanged sources (record its md5; it need not equal #23's `81dcaab4…`, different SDK patch). Images/tokens: reuse the workstation's `/tmp/qwen3_full`, `/tmp/qwen3_full4k`, `t512.i32` (P4 prompt, the accuracy anchor), `t1024.i32`, `t4096.i32` with the md5s of `docs/measurements/23-sdk64-baseline.md`.

**Step 4 — device handoff (unavoidable: every number here is transport or wall time).** `docs/measurements/24-host-logits.md` from the `hexagon-handoff` template, estimated **25 min**, **one skel**, one harness, variants = harness flags: **A `--logits-mem malloc`** (today's path), **B `rpcmem`**, **C `static`**.
1. `run_device_test.sh` → `RPC_TEST open ok`, `rpcmem ok`, `bad-version rejected ok`, `init ok`, 32 × `RPC_TEST forward_us <us>`, 3 × 32 × `RPC_TEST forward_full_us mem={malloc,rpcmem,static} <us>`, `RPC_TEST pattern ok`, `RPC_TEST PASS`; `python3 tools/hexagon/check_rpc_log.py logs/hexagon/device_test_<stamp>.log` → `VERDICT: PASS` (G1 table: per-mem median).
2. For A, B, C: `run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512.i32 --chunk 128 --steps 64 --logits-mem <v>` → `E2E init ok weights=598623744 kv=234881024 act=3932160 n_ops=451`, 4 × `E2E step … n=128 …`, 63 × `E2E step … n=1 pcycles=<c> us=<t> top1=<id> …`, `E2E gen <64 ids>`, `E2E decode steps=63 median_us=<u> median_pcycles=<c> pcycles_per_us=<r>`, `E2E wall_ms <w>` (G2 on the shipped variant; G3 `E2E gen` identical across the three, reference: #23 A 512 log).
3. `--eval` at 512 for A and B: `E2E ppl <x> steps 511 top1 <n> wall_ms <w>` + the `E2E decode …` line — PPL/top-1 identical to each other and to #23's `33.0884 / 189` (G3); `pcycles_per_us` here is the idle-gap clock (data for §5's follow-up, not a gate).
4. Goal rows with B (or C if it won G1): 1024 (`qwen3_full`, `t1024.i32`) and 4096 (`qwen3_full4k`, `t4096.i32`), `--chunk 128 --steps 64` (G4; the 512 row is step 2).
5. Read back `adb shell md5sum /data/local/tmp/nntr_htp/{libnntr_htp_skel.so,hexagon_e2e_test}` into the table. Result tables: (i) rpc_test `mem | median forward_full_us | − forward_us(8)`; (ii) `variant | ctx | prefill tok/s | median_us | median_pcycles | pcycles_per_us | gap ms @2.09 | decode tok/s | gen ids identical?` with #23's A rows as reference; (iii) `variant | --eval PPL | top-1 | reference 33.0884 / 189`. Set `state:needs-measurement`.

**Step 5 — after `state:measured`.** Apply the §3 decision rule; default `--logits-mem` becomes the winner; write §6 docs with the measured numbers; gate 0 (`clang-format-14` on changed `.cpp/.h/.py` untouched); commits: `[hexagon] host: rpcmem logits buffer …`, `[test] hexagon: transport rows in rpc_test, decode summary in e2e`, `[docs] …`; PR into `hvx_impl`, issue → `state:review`. If trigger (2) or (3) fires, the PR still ships (c0) and the trigger's follow-up is filed by the supervisor with the handoff numbers.

## 5. Risks

* **Zero-copy assumption.** If libadsprpc needs an explicit registration, B == A in G1 — visible as two equal medians; rule (4) of §3 is the fix, re-measured by the rpc_test rows alone (2 min).
* **Clock assumption (2.09 GHz).** `pcycles_per_us` is a lower bound printed per run; at 512 it read 2.054 GHz in #23, so the gap is < 1 ms for any top clock ≤ 2.09. If `r` drops in B/C at equal pcycles, that *is* transport moving into the timed region and the table shows it.
* **512 decode spread (±5 %, #23 caveat).** A/B/C run back to back in one session on one image; compare medians and `r`, not tok/s alone; the 1024/4096 rows are flat (#23) and carry the goal check.
* **Idle-gap DCVS (not this issue, but this plan's most useful side finding).** `--eval` `pcycles_per_us` ≈ 1.15 GHz means the app (`engine="htp"`, tokenizer/print work between RPCs) pays what the P3/P4 `--eval` rows paid. The M5 `HAP_power` vote was judged in generation mode only (`HEXAGON.md:799-805`); the supervisor should file "HAP_power / DCVS vote for gapped decode" with step 4.3's `r` as the opening number.
* **#33 overlap.** #33 edits `hexagon_backend.cpp:51-63` (host id pre-check) and `hexagon_runner.h`; rebase onto `hvx_impl` after it merges and keep its loop ahead of the RPC. #23's §8.2 table is the reference text for §6 — base the doc edits on the post-#23 `hvx_impl`.
* **Stale artifacts.** Harness md5 differs from #23 by construction; skel md5 and `act=3932160` on the `E2E init ok` line are the layout/identity checks; `push_if_changed` compares md5.
* **Simulator-vs-device gaps exposed: none** — no DSP byte changes, no numerics claim; every performance number comes from the handoff.

## 6. Docs to update

* `HEXAGON.md:22-26` (intro): replace "~40 ms per generation step" with the measured gap and transport (G1/G2), and point at ① ⑨ ⑩ as the decode levers.
* `HEXAGON.md:89-90` (§1.1): "`forward()` carries only `token_ids` in and `logits` out" → add that the host logits buffer is rpcmem (ION zero-copy, `FASTRPC_MAP_STATIC` if shipped) and the measured cost of the 607,744 B return; note malloc memory takes the copy path.
* `HEXAGON.md:853-857, 899-902` (§8.2): keep the P3/P4 history, append a "Host path (#24, date)" paragraph with table (i)/(ii) and the `--eval` clock observation; `:1145-1149` (§9): rewrite the host/RPC bullet to the outcome (closed, or the (a) trigger if it fired).
* `HEXAGON_BENCHMARK.md:40`: lever text "⑫ host logits path (~40 ms/step)" → measured value; `:27-29`: supervisor adds the G4 rows (date, v75, `--logits-mem`); log row.
* `docs/superpowers/specs/hexagon-hvx-optimization/07-follow-ups.md:66-68` ⑫: done with plan/handoff paths and the numbers; (a)/(b) "not needed unless transport ≥ 2 ms".
* `.claude/skills/hexagon-handoff/SKILL.md` template: mention the `E2E decode …` summary line once it exists (removes the by-hand median step of the #23 handoff).
