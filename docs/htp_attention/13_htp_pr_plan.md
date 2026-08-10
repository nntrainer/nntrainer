# HTP u8i4/u8i8 micro+DMA+cross — implementation plan

Working plan, not repo documentation. Written 2026-08-06.
Executor: next session. Read this top to bottom before touching code.

---

## 0. Established facts — do not re-derive

Each of these was verified this session by reading the tree, not inferred.

| fact | evidence |
| :-- | :-- |
| `#4236` is fetchable without merge | `git fetch upstream 'refs/pull/4236/head:pr4236'` → `0d1df7428` |
| `upstream/main` has **no** SDKL macro FC dispatch | `git grep sdkl_npu_mm upstream/main -- nntrainer/` → empty |
| `#4236` is **pure micro API** | uses only `hexkl_micro_{hw_init,hmx_lock,hmx_unlock,hmx_setup_acc_read_int32,hmx_config_size}` |
| the macro/micro one-way door is **our branch's problem only** | `sdkl_npu_mm` exists only in `claude/…-6ycsx0-v2:nntrainer/tensor/htp_backend/hmx_ops/hexkl_mm.cpp` |
| `nntr_hvx_open` / `nntr_hvx_close` already exist | `test/htp/hvx_add_f32.c:29,40` — the session hook point is already there |
| upstream `ComputeOps` has **no** quantized-matmul virtual | `compute_ops.h` has `sgemm_fp32`/`sgemv_fp32`/… only; the `#4162` merge dropped `shgemm`/u8i8 virtuals |
| the bench is already dtype-parametrised over u8i4 **and** u8i8 | `hexkl_micro_fc_bench.c` passes a `const dtype_ops *dt` through every entry point |
| the cross-matmul win needs **multiple matmuls per call** | doc13 §3a: 1.7–2× over SDKL with cross-prefetch; *single-matmul is parity only* |

**Consequence of the last row — the single most important design constraint.**
A lean endpoint that does one matmul per FastRPC call cannot show the win: there
is nothing to prefetch across, and the ~170 µs per-call toll (doc15 §3) is paid
per matmul. The endpoint must take **a layer's worth of matmuls in one call.**

**Consequence of the macro/micro row.** Do not port
`hmx_ops/hexkl_mm.cpp` (macro API) upstream. Upstream is clean today; landing it
would plant the irreversible conflict. PR① replaces it rather than joining it.

---

## 1. Branches

```
upstream/main
  └─ pr4236                       ← base for everything below
      ├─ htp/u8i4-dma-cross       PR①
      │   └─ htp/u8i8-dma-cross   PR②   (branches off PR①)
      │       └─ htp/arm-seam     PR③   (branches off PR②)
      │
htp/bench-evidence                push only, never merged — reference link
```

Every PR body says `Depends on #4236` plus the preceding PR number.
When `#4236` lands: `git rebase --onto upstream/main pr4236 <branch>`.

---

## 2. How to run it on device — this already works today

`#4236` ships a complete device harness. Our PRs extend it rather than inventing one.

```bash
# 1. DSP skel
source ~/workspace/Hexagon_SDK/6.4.0.2/setup_sdk_env.source
HEXKL_ROOT=~/workspace/Hexagon_SDK/6.4.0.2/addons/hexkl_addon \
HEXKL_SDK_VER=6.4.0.2 ./test/htp/build.sh
#   → test/htp/build/libnntr_hvx_skel.so

# 2. ARM gtest
cd test/jni && ndk-build   # see test/jni/Android.mk

# 3. push skel + binary, set ADSP_LIBRARY_PATH to the skel dir, run
```

Existing tests in `test/unittest/unittest_hvx_mm_u8i4.cpp` (fixture `HmxMmU8I4`):
`Shape1_Minimal`, `Shape2_DecodeSingleToken`, `Shape3_PrefillQwen3Scale`,
`Shape4_MultipleRowBlocks`. It opens a session with `nntr_hvx_open(uri, &handle_)`
and checks against an ARM-side scalar reference.

**Reuse that reference for every new test.** Do not write a second one.

Device: Galaxy S25 Ultra, `R3CY10WM83Y`, V79.

---

## 3. PR① — `htp/u8i4-dma-cross` — IMPLEMENTATION DONE, compiled, not yet run on device

**Correction from earlier in this session: `#4236` was 6 commits behind
`upstream/main` (merge-base `2b9140bae`, `#4236` tip `0d1df7428`,
`upstream/main` tip `f97c2e26b`).** The branch was built by mistake directly
off the stale `pr4236` ref first; fixed with
`git rebase --onto upstream/main $(git merge-base upstream/main pr4236) htp/u8i4-dma-cross`,
which dropped 4 of #4236's 13 commits as already-upstream and rebased the
rest cleanly. **Always rebase onto current `upstream/main` before starting
work on a branch stacked on an open PR — do not assume the PR ref is current.**

Branch `htp/u8i4-dma-cross`, 13 commits on `upstream/main` (9 from #4236,
rebased; 4 new — dma ring, weight-registry+layer-run module, session
scoping + IDL, tests). T1–T6 below are implemented. **Compiled clean on
both sides** (§3a), not yet run on a device.

### 3a. How this was verified (compile only) — do this again after any edit

Hexagon SDK 6.4.0.2 is at `~/workspace/Hexagon_SDK/6.4.0.2`; its own
`setup_sdk_env.source` fails here ("missed components") but the pieces it
would have set are all present, so set them by hand instead of fixing that
script:

```bash
export HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2
export DEFAULT_HEXAGON_TOOLS_ROOT=~/workspace/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04
```

**Use the beta2 HexKL addon, not the one under `Hexagon_SDK/6.4.0.2/addons/`.**
The latter is beta1 (`hexkl_micro_hw_init` takes 2 args); `#4236`'s code
(and this branch's) is written against beta2's 3-arg signature
(`vtcm_base, vtcm_size, hmx_fp16_rate`). The working one:

```bash
export HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon
export HEXKL_SDK_VER=6.4.0.2
cd test/htp && bash build.sh
# → test/htp/build/libnntr_hvx_skel.so (v79, hexkl 6.4.0.2), 0 warnings under -Werror
```

If `qaic` rejects the IDL with `unexpected "o" / expecting "in", "rout" or
"inrout"` — this qaic version has no plain `out` for a scalar; use `rout`
even for a single value (already fixed in the committed IDL, noting the
gotcha for the next scalar out-param anyone adds).

ARM-side syntax check (full `ndk-build` additionally wants
`builddir/jni/arm64-v8a/libnntrainer.so`, i.e. the whole library
cross-built first — not done this session, not needed to check the test
file itself):

```bash
export GTEST_ROOT=<repo>/subprojects/googletest/googletest   # `git submodule update --init` first if empty
g++ -std=c++17 -fsyntax-only -I "$GTEST_ROOT/include" -I test/htp/generated \
  -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/incs" \
  test/unittest/unittest_hvx_mm_u8i4.cpp
```

Confirmed the qaic-generated `test/htp/generated/nntr_hvx.h` prototypes for
`nntr_hvx_weight_register_u8i4`/`weight_release_u8i4`/`mm_u8i4_layer` match
the hand-written implementations in `nntr_hvx_mm_u8i4.c` and the calls in
the test file exactly — this was a real cross-check via the actual `qaic`
compiler, not an assumption.

### 3b. Device gate — CLEARED, on `R3CY10WM83Y`, this session

`test/htp/run_u8i4_layer_on_device.sh` (committed, `0b1ee7fc9`) builds and
runs the whole thing end to end. Ran it for real:

```
[HmxMmU8I4]      4 accuracy-harness shapes — all PASS (unchanged by T1)
[HmxMmU8I4Layer] 5 new layer-endpoint tests — all PASS
U8I4_FIELD path=harness  field=us_per_matmul value=7431.59
U8I4_FIELD path=layer_x1 field=us_per_matmul value=3073.85
U8I4_FIELD path=layer_x4 field=us_per_matmul value=2243.17
U8I4_FIELD path=layer_x4 field=speedup_vs_harness value=3.31
```

**3.31× — above doc13 §3a's 1.7–2× gate.** That number was DSP-only, no
FastRPC in the timed region; this one is end-to-end ARM↔DSP, so the
session-scoped HMX (no per-call hw_init/lock) and the dropped debug
buffers stack on top of the DMA/prefetch win doc13 isolated. **Cleared —
PR②/③ can start.**

Three more environment gotchas found running this for the first time
(now baked into the script and into memory
`htp-fastrpc-seam-next-steps`, so this list will not need to grow again
for the same reasons):
- `subprojects/iniparser` unfetched (empty wrap-git placeholder) — same
  class as the -v2 work's issue, different subproject.
- This shell's profile exports `NNTRAINER_ROOT=/home/leeseunghui/nntrainer`
  — an unrelated checkout — which silently wins over Android.mk's
  `ifndef`-guarded default. Must override explicitly on every `ndk-build`
  invocation, the same shape of bug as the -v2 work's `LD_LIBRARY_PATH`
  issue.
- `test/jni/Android.mk`'s `googletest_main` module expects
  `test/jni/googletest/{src,include}` and nothing creates it — symlink to
  `subprojects/googletest/googletest` by hand (script does this now).

`ndk-build`'s executable output lands at
`test/jni/obj/local/arm64-v8a/<module>`, not `libs/arm64-v8a/` — no
install step ran, this was a direct module build.

---

## 3c. PR② — `hexkl_mm_u8i8_dma.h/.c`, u8i8 layer endpoint — DONE, device-verified

Built as 4 more commits on the same `htp/u8i4-dma-cross` branch (pushed;
did not split into a separate branch — see below for why that is fine).
`hexkl_mm_u8i8_dma.h/.c` mirror `hexkl_mm_u8i4_dma.h/.c` exactly (tile
bytes 1024 vs 512, HexKL's `_i8` primitives instead of `_i4`) rather than
sharing one dtype-parametrised module — deliberate: the u8i4 path was
already device-verified, and folding both behind one abstraction would
mean re-verifying it to trust, for a saving of ~150 duplicated lines.
Same call for the IDL (`weight_register_u8i8`/`weight_release_u8i8`/
`mm_u8i8_layer`, separate ops, not a `dtype` enum on the u8i4 ones) and
for the test fixture (`HmxMmU8I8Layer` mirrors `HmxMmU8I4Layer`).

No u8i8 accuracy-harness endpoint exists to mirror `mm_u8i4_from_f32` --
`#4236` never added one -- so `HmxMmU8I8Layer`'s tests check directly
against the same per-weight integer reference the u8i4 tests use, with a
new `quantize_weights_symmetric_i8` (range [-128, 127] instead of qs4cx's
[-8, 7]).

`test/htp/run_u8i4_layer_on_device.sh` needed no build/push/run changes --
both fixtures live in one gtest binary (`unittest_hvx_mm_u8i4`, despite
the name) -- just a summary/comment update to cover both.

**Verified on `R3CY10WM83Y`: 14/14 tests pass** (the original 9 plus 5 new
u8i8 tests):
```
U8I8_FIELD path=layer_x4              field=us_per_matmul value=2255.69
U8I8_FIELD path=layer_x4_vs_u8i4_x1   field=ratio         value=1.12
```
u8i8's `layer_x4` (~2256 µs/matmul) lands close to u8i4's (~2200 µs/matmul)
despite double the weight bytes per tile -- the DMA/prefetch mechanics
mask the width difference at this shape. 1.12× over u8i4's un-prefetched
single-matmul call, printed not asserted (same reasoning as PR①'s
`ReportPerCallCost`).

Two branches, as originally planned (§1) -- caught and fixed a mistake
mid-session where PR②'s 4 commits had landed on the same local branch as
PR①. `origin/htp/u8i4-dma-cross` was already pushed at that point (14
commits) and untouched by the mistake, so the fix was just local:
`htp/u8i4-dma-cross` reset back to that 14-commit tip (`0b1ee7fc9`,
matching origin exactly), and a new `htp/u8i8-dma-cross` created at the
18-commit tip so it carries PR①'s 14 plus its own 4. Pushed: `origin/htp/u8i8-dma-cross` = `983cf0e3d`. When opening the PR,
set base = `htp/u8i4-dma-cross` (PR①'s branch), not `main`, so the diff
shown is only these 4 commits.

---

Turns `#4236`'s accuracy harness into a performance path and lands the verified
staging optimisations in one coherent change. (An earlier draft split this into
two PRs; that split is artificial because the endpoint is new code, not a
refactor of shipped code.)

### What is wrong with the harness today

All four are visible in the 115 lines of `test/htp/nntr_hvx_mm_u8i4.c`:

| # | problem | site | measured cost |
| --: | :-- | :-- | :-- |
| 1 | `hexkl_micro_hw_init` + `hmx_lock`/`unlock` per invoke | `hmx_bringup()` inside the entry point | unmeasured, nonzero |
| 2 | weight re-baked per invoke | `hexkl_mm_u8i4_bake_weights()` | **~72 µs/call** (doc15 §3) |
| 3 | four `rout` debug buffers | `act_u8_ah` (m_pad×K), `act_scale`, `act_zp`, `acc_i32` (m_pad×N) | part of ~170 µs transport |
| 4 | no user-DMA, no cross-matmul prefetch | single `hexkl_mm_u8i4_run` | the 1.7–2× left on the table |

Problem 2 is the same defect as doc11's 39.3 ms/token staging, at a different layer.

### Tasks

**T1 — session-scoped HMX.**
Move `hmx_bringup()` and `hexkl_micro_hmx_lock()` into `nntr_hvx_open()`
(`test/htp/hvx_add_f32.c`, or a new `nntr_hvx_session.c`); `hmx_unlock` into
`nntr_hvx_close()`. Store `{vtcm_base, vtcm_size, config_off}` in a session
struct reachable from the `remote_handle64`.
*Watch:* `hvx_add_f32.c` currently returns a trivial handle — check what it
stores before widening it.

**T2 — weight registration, baked once.**
New IDL ops:
```
AEEResult weight_register(in uint32 K, in uint32 N, in sequence<int8> w_i4_rm,
                          in sequence<float> w_scale, in sequence<int32> colsum_w,
                          rout uint64 w_handle);
AEEResult weight_release(in uint64 w_handle);
```
Bake to WH once at registration. Keep the WH bytes in DDR (not VTCM) so total
resident weight is not bounded by the VTCM arena — T5 streams them in per tile.

*Do not* copy `-v2`'s NPU-resident weight cache design. Device A/B showed it
crashes above 256 MiB resident and regresses 4.8× at the default
(`e1f5bb480`, memory `htp-decode-wh-cache-fix-broken`, doc11 §3 update).

**T3 — lean layer endpoint.**
```
AEEResult mm_u8i4_layer(in uint32 M, in uint32 K,
                        in sequence<uint64> w_handles,   // one per matmul
                        in sequence<float> act_f32,      // shared activation
                        in sequence<float> bias_cat,
                        rout sequence<float> out_cat);   // concatenated
```
Returns only the output. Quantise the activation once and reuse it across every
matmul in the list — q/k/v share an activation, and so do gate/up.

Keep `mm_u8i4_from_f32` untouched: `unittest_hvx_mm_u8i4.cpp` is the accuracy
harness and must keep passing.

The exact signature is the first thing to validate — sequence-of-handles
marshalling through `qaic` may need a fixed-size array instead. Settle it before
writing the kernel side.

**T4 — DMA staging module.**
Port `matmul_resident_dma` (`hexkl_micro_fc_bench.c:522`) into
`nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.c/.h`.
Bench signature to work from:
```c
static int matmul_resident_dma(uint8_t *vtcm_base, uint32_t vtcm_size,
                               const vtcm_plan *p, uint32_t A_rows,
                               uint32_t n_inner, uint32_t W_cols, int32_t *out,
                               const uint8_t *act, const uint8_t *wt_wh_ddr,
                               uint64_t *t_act, uint64_t *t_hmx,
                               uint64_t *t_accread, uint64_t *t_dma, ...);
```
Drop the `t_*` timing outs from the production signature; keep them behind a
build flag for the bench.
`vtcm_plan` is the bench's own layout struct — reconcile it with `#4236`'s
`hexkl_mm_u8i4_layout` rather than carrying both.

**T5 — cross-matmul prefetch.**
Port `run_pipelined_layer` (`:1135`) into the layer loop from T3: while matmul
*i* computes, DMA in matmul *i+1*'s first weight tiles.

**T6 — tests.**
Add to `unittest_hvx_mm_u8i4.cpp` (or a sibling `unittest_hvx_mm_layer.cpp`):
- correctness: layer endpoint vs the existing scalar reference, per matmul
- session reuse: two `mm_u8i4_layer` calls on one handle give identical results
- weight lifecycle: `weight_register` → use → `weight_release` → re-register
- perf: report per-call µs for (a) `mm_u8i4_from_f32` baseline (b) lean, no DMA
  (c) lean + DMA + prefetch. Print, do not assert a threshold — thermal state
  makes a hard bound flaky.

Report through `field=… value=…` marker lines, as the existing benches do; a
report script silently dropping shapes bit us twice.

### Verification gate before PR② starts

On `R3CY10WM83Y`: correctness tests green, and the (c)/(a) ratio reproduces the
1.7–2× of doc13 §3a within noise. If it does not, stop and find out why —
PR② and PR③ both build on this being true.

---

## 4. PR② — `htp/u8i8-dma-cross`

The bench already carries a `dtype_ops` vtable covering both widths, so this is
parametrisation, not a second implementation.

**T1** — `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i8.c/.h`, mirroring
`hexkl_mm_u8i4.c/.h` (plan / bake_weights / run).

**T2** — port the bench's `dtype_ops` struct so `hexkl_mm_*_dma.c` becomes
dtype-generic; u8i4 keeps working through the same path.

**T3** — extend the IDL: prefer a `dtype` enum parameter on `weight_register` +
`mm_layer` over duplicating both ops. Decide once, early.

**T4** — extend the tests with u8i8 shapes; keep both widths in the same fixture.

**Accuracy note for the PR body.** u8i8 fixes the activation at `uint8` and
widens only the weight slot, so its risk profile is not u8i4's. QNN's shipping
Gauss deployment quantises its KV cache to **int8, not int4** — verified by
reading `~/workspace/Quick.AI`'s `gauss*_qnn.cpp` runtime dtype assertions.

---

## 5. PR③ — `htp/arm-seam`

The point where `nntrainer_causallm` actually runs on the NPU. Everything before
this is callable only from gtest.

**T1 — a quantised-matmul virtual on `ComputeOps`.**
Upstream dropped the `shgemm`/u8i8 virtuals in the `#4162` merge, so add one,
plus a `supports_*()` predicate so the CPU path stays the default and the NPU
path is opt-in per layer.

**T2 — `HtpComputeOps` implementation** calling the FastRPC stub from PR①/②.
Open the session once per process, not per call — it now holds the HMX lock.

**T3 — weight lifecycle on the ARM side.**
`weight_register` at model load, `weight_release` at teardown. This is where the
WH bake belongs.

**Reusable from `-v2` (format/dtype work, independent of macro vs micro):**
- `quantizer.cpp`'s QINT8 contract — per-channel weight scale + `zp_corr`
  (`zp_corr = 128 · Σ_k W_i8[n,k]`), returns `[N, K]` for the WH bake
- `char_tensor.cpp` carrying `zp_corr` alongside per-channel scales
- the WH-trailer codec (`wh_trailer.cpp/.h`) and its `neuralnet.cpp` /
  `layer_devel.h` load hooks

**Not reusable — deliberately dropped:**
- `hmx_ops/hexkl_mm.cpp` and its `sdkl_npu_mm_*` dispatch (macro API)
- the `e1f5bb480` NPU-resident weight cache

**T4 — e2e test.** Extend the CausalLM test harness so one real FC runs through
the NPU path and matches CPU within tolerance. This is the artefact that lets a
model be run by hand afterwards.

---

## 6. Commit and PR conventions

```
[HTP] <imperative subject, ≤60 chars>

<2–4 lines: what changed and why. Numbers where they exist.>

Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>
Co-authored-by: Claude <noreply@anthropic.com>
```

Subjects follow the existing history: `[HTP]`, `[Tensor]`, `[CausalLM]`, `[docs]`.
Bodies stay short — the current branch's 20-line bodies are the thing being fixed.

PR bodies carry the measurement table and `Depends on #4236` (+ prior PR).
The load-bearing findings move into the PR that they justify:

| finding | goes to |
| :-- | :-- |
| macro↔micro one-way door (doc15 §2) | PR① body — it is why the endpoint is micro-only |
| FastRPC ~170 µs/call (doc15 §3) | PR① body — why the endpoint is per-layer |
| u8i4 FC 1.7–2× cross-prefetch (doc13 §3a) | PR① body |
| fp16 MHA 1.9–2× (doc14) | nowhere — out of scope, stays on `htp/bench-evidence` |

---

## 7. Push through the corporate proxy

The proxy rejects large packs (HTTP 400) and rate-limits (503). Small pushes work.

```bash
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
# one commit at a time; 503 is transient, re-running resumes
for c in $(git rev-list --reverse <base>..<branch>); do
  until git push origin "$c:refs/heads/<branch>"; do sleep 15; done
done
```

Leave `http.proxy` / `https.proxy` set to `http://10.112.1.184:8080` — unsetting
them does not help and breaks other access.

State as of writing: `claude/hexkl-mha-hmx-optimization-6ycsx0-v2` is pushed up
to its 2nd commit; a stray `tmp-chunk-test` branch on `origin` should be deleted.

---

## 8. Order of work

1. `htp/bench-evidence` — push only, no PR to merge. Gives PR①–③ something to link.
2. **PR①** — the substantial one. Gate on §3's verification before continuing.
3. **PR②** — cheap if PR①'s module came out dtype-generic.
4. **PR③** — the one that makes a model run.

Do not start PR③ before PR① passes its device gate; its whole value depends on
the seam being fast.
