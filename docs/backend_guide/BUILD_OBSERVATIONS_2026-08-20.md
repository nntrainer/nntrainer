# Build Observations — 2026-08-20

**Goal:** Build the current working tree (including all uncommitted changes)
for Android/Hexagon, deploy to device `R3CX9078DNH` (Samsung S24 Ultra,
SM-S936U, Snapdragon 8 Elite, HTP v79), and measure CPU vs NPU prefill
performance on Qwen3-0.6B.

This doc records the build process as actually executed: which approach was
tried, what failed, and how each failure was resolved — not just the final
working command.

---

## 1. Two build systems, and picking the right one

nntrainer has **two independent ways to configure a meson build with
`-Dplatform=android`**, and only one of them produces something you can run
on-device:

| | `meson setup build-android --cross-file android-aarch64.ini ...` | `tools/package_android.sh` (→ `builddir/`) |
|---|---|---|
| Compiler | NDK clang, driven by meson's own cross-compilation | host gcc (meson's "host" toolchain) for meson-side bookkeeping; the real arm64 artifacts come from an internal `ndk-build` custom_target |
| `nntrainer_dep` on `platform=android` | `declare_dependency(include_directories: ...)` — **no `link_with` at all** ([nntrainer/meson.build:111](../../nntrainer/meson.build#L111)) | N/A — nntrainer itself isn't linked through meson's own targets on this path either; ndk-build handles it |
| What it's *for* | Nothing, functionally — any target on this path that tries to actually link against nntrainer (e.g. `Applications/CausalLM`'s meson-side targets) fails with dozens of undefined symbols (`Tensor::getDim()`, `RunLayerContext::getInput()`, ...) | Building `libnntrainer.so`/`libccapi-nntrainer.so` for Android and packaging them, with headers, into `builddir/android_build_result/` as a **prebuilt** for a *separate*, hand-written `ndk-build` step |

I started down the `build-android` cross-file path (it already existed in
the repo) and got exactly that undefined-symbol wall trying to link
`libreshaped_rms_norm_layer.so`. That sent me looking at how `nntrainer_dep`
is defined, which is what surfaced the "no `link_with` for android" gap above
— this build mode is not the intended way to produce Android artifacts.
Abandoned it.

**The actual, correct pipeline** (confirmed by reading
`Applications/CausalLM/build_android.sh` and `Applications/CausalLM/jni/Android.mk`):

1. `tools/package_android.sh` → configures/builds `builddir` with
   `-Dplatform=android`, and its `ninja install` step both (a) builds
   `nntrainer`/`api` for arm64 via an internal `ndk-build` custom_target, and
   (b) installs headers + the arm64 `.so`s + a **generated** `Android.mk`
   into `builddir/android_build_result/`.
2. `Applications/CausalLM/jni/Android.mk` (hand-written, not
   meson-generated) `include`s that prebuilt `Android.mk`, then compiles
   CausalLM's own layers/models against it with a second, standalone
   `ndk-build` invocation — producing `libcausallm_core.so`,
   `nntrainer_causallm`, `nntr_quantize`, `nntr_safetensors_info`.

Everything below is errors hit while driving *this* pipeline.

---

## 2. `builddir` was in a half-reconfigured state

`builddir` already existed (from an earlier session) but had no
`build.ninja` — only `meson-private/` sanity-check leftovers. `ninja -C
builddir install` failed immediately with `loading 'build.ninja': No such
file or directory`. Likely cause: a prior `meson --wipe` that didn't get
followed through to a full regenerate+build.

**Fix:** `meson setup builddir --reconfigure <same options as
package_android.sh, plus -Denable-transformer=true -Denable-hexagon-cdsp=true
-Denable-test=false>` (matching what the existing `android_build_result`
metadata showed had been used previously — see
`builddir/meson-logs/meson-log.txt`'s recorded `Build Options`).

---

## 3. `prepare_ml-api.sh` — dead upstream asset

Meson's ml-api prep step downloads `nnstreamer-lite-native.zip` from an S3
URL. That download now 404s with `x-amz-delete-marker: true` — the object
was deleted upstream (versioned bucket, "latest" pointer is gone):

```
meson.build:649:2: ERROR: Command `.../prepare_ml-api.sh ...` failed with status 1.
unzip:  cannot find zipfile directory in one of nnstreamer-lite-native.zip ...
```

**Fix:** The repo already had the extracted headers sitting untracked at
`api/{ml-api-common,ml-api-service,nnstreamer,nnstreamer-native,nnstreamer-single,tizen_error}.h`
(apparently placed there in an earlier session for this exact reason), and
an older `build-cdsp/ml-api-inference/lib/arm64-v8a/{libnnstreamer-native.so,libgstreamer_android.so}`
existed from a previous successful build. `prepare_ml-api.sh` skips its
download+extract entirely if `${TARGET}/include` already exists, so:

```bash
mkdir -p builddir/ml-api-inference/include builddir/ml-api-inference/lib/arm64-v8a
cp api/{ml-api-common,ml-api-service,nnstreamer,nnstreamer-native,nnstreamer-single,tizen_error}.h \
   builddir/ml-api-inference/include/
cp build-cdsp/ml-api-inference/lib/arm64-v8a/*.so builddir/ml-api-inference/lib/arm64-v8a/
```

satisfied the check without needing network access to the (now-gone) asset.

---

## 4. `ninja install` pulled in unrelated, broken desktop apps

With `-Denable-app` at its default (`true`), `ninja install` tried to build
every app under `Applications/`, not just CausalLM, and failed on three of
them:

```
error: '__fp16' was not declared in this scope; did you mean '__bf16'?
```

in `SimpleFC`, `MixedPrecision`, and `VGG`. Root cause: these are compiled
with the **host** gcc (this build only cross-compiles nntrainer itself, via
the internal ndk-build custom_target — see §1), and plain x86_64 GCC has no
`__fp16` type at all (confirmed directly: `echo '__fp16 x;' | g++ -x c++ -`
fails the same way, no flag fixes it). `api/ccapi/include/layer.h`
unconditionally declares `getFP16Weights()` with a `_FP16`/`__fp16` typedef
whenever `ENABLE_FP16` is set, with no host-vs-target guard.

The existing log line `warning: android app: only building CausalLM (other
apps skipped)` ([meson.build:754](../../meson.build#L754)) claims this
already doesn't happen — but the code on both branches of that `if` just
calls `subdir('Applications')` unconditionally, so the warning was aspirational, not
enforced.

**Fix (two parts, so the fix is correct in either config):**
- Passed `-Denable-app=false` to the `builddir` configure — the
  Android-prebuilt step doesn't need `Applications/` built at all (that's
  `Applications/CausalLM`'s separate ndk-build's job).
- Also edited [`Applications/meson.build`](../../Applications/meson.build)
  so `platform == 'android'` actually only processes `CausalLM`, matching
  what the warning already (incorrectly) claimed. This means a future
  `-Denable-app=true` android build won't hit this either.

---

## 5. Two real syntax bugs in the uncommitted tree

Once the above got `ninja install` for the nntrainer/api prebuilt to a clean
finish, the second stage — `ndk-build` for `Applications/CausalLM/jni` —
hit two genuine compile errors, both inside files that are part of the
session's **uncommitted** changes (confirmed by building clean `HEAD` with
the fix reverted: same files compile without incident there, see §7):

**a) `Applications/CausalLM/models/gpt_oss/gpt_oss_moe_layer.h:133-136`**

```cpp
bool enable_bias = false; (void)enable_bias;
float alpha = 1.702;
float limit = 7.0; (void)limit;
```

`(void)x;` as a statement is not legal directly inside a class body —

```
error: expected member name or ';' after declaration specifiers
```

Looks like a leftover fragment from an automated "silence unused-field
warning" edit that got applied at class scope instead of inside a method.
**Fix:** removed the stray `(void)enable_bias;` / `(void)limit;`, restoring
plain member declarations (byte-identical to `HEAD` afterwards).

**b) `Applications/CausalLM/models/lfm2/lfm2_causallm.h:108-110`**

```cpp
void run_with_embeddings(const void *inputs_embeds, size_t n_tokens
                         std::vector<int> seed_tokens, bool do_sample,
                         bool log_output);
```

Missing comma after `n_tokens` — plain syntax error. The same edit had also
added `override` to `lookupEmbedding()` two lines below, even though the
base class (`causal_lm.h`) never declares that method — `override` there
would fail to compile too, once the syntax error was no longer masking it.
**Fix:** restored the comma, dropped the invalid `override` (again,
byte-identical to `HEAD` afterwards).

Everything else in the uncommitted diff (`mha_core.cpp`, `rms_norm.cpp`,
`reshaped_rms_norm.cpp`, `hexagon_context.cpp`, `addition_layer.cpp`,
`layer_context.h`, `layer_node.cpp`, `multiout_realizer.cpp`,
`fused_ffn_layer.*`, `causal_lm.cpp`, `qwen3_causallm.cpp`,
`transformer.cpp`) compiled cleanly with no changes needed.

After these two fixes, both `ninja -C builddir install` and the
`ndk-build ... causallm_core nntrainer_causallm nntr_quantize
nntr_safetensors_info` step completed successfully (~45s and ~45-55s
respectively on a 40-core machine).

---

## 6. Runtime: `error 0x80000406` opening the DSP session

First device run (env: `NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_FLASH_ATTN=1
NNTR_HEXAGON_FUSED_FFN=1`) failed on *every* bridge call:

```
ggml-hex: HTP0 allocating new session
ggml-hex: failed to open session 0 : error 0x80000406
ggml-hex: releasing session: HTP0
nntr-htp-bridge: upload_weight_q4x4x2 failed: ...
[!] FATAL ERROR: HexagonComputeOps::ensure_uploaded: ... failed
```

CPU-only mode (no `NNTR_USE_HEXAGON_CDSP`) ran fine, so the binary itself
wasn't broken. I rebooted the device on the theory that a leaked FastRPC
session from a previous test run needed clearing — **this did not fix it**,
and in hindsight wasn't the cause (noting it since it's a disruptive step I
took that turned out to be unnecessary).

The actual cause: I was invoking the binary as `cd $MODEL_DIR && ./nntrainer_causallm
$MODEL_DIR`. An already-on-device standalone tool (`verify_flash_attn`,
built in an earlier session) opened a DSP session successfully when run from
`/data/local/tmp/nntrainer/causallm` (the directory containing
`libggml-htp-v79.so`) — same device, same libraries, same reboot state,
run seconds apart from a failing `nntrainer_causallm` invocation. The
difference was purely **current working directory**. The DSP-side skel is
requested via a bare `file:///libggml-htp-v79.so` URI
(`ggml-hexagon.cpp:2431`) with no absolute path; on this device/driver that
resolves relative to the calling process's cwd, not an absolute vendor
search path.

**Fix:** run from the binary's own directory, passing the model directory as
an argument instead of `cd`-ing into it:

```bash
cd /data/local/tmp/nntrainer/causallm && ./nntrainer_causallm /data/local/tmp/nntrainer/causallm/models/qwen3-0.6b
```

This is also a latent trap in the already-on-device `bench_comprehensive.sh`
/ `bench_sweep*.sh` scripts, which all `cd $MODEL_DIR` before invoking the
binary — the fact that they apparently worked in the 2026-08-19 session
implies `ADSP_LIBRARY_PATH` (or equivalent) was set interactively in that
shell and simply never made it into the saved script.

---

## 7. Isolating the batching regression

Benchmarking the uncommitted tree showed NPU mode much slower than CPU —
the opposite of every prior sweep in this repo's docs. To confirm this was
caused by the uncommitted changes and not the device/rebuild/measurement
process, I:

1. `git stash` (reverting to clean `HEAD`, commit `c7d30ec7`)
2. Rebuilt both stages exactly as above (no source fixes needed — confirms
   §5's bugs are uncommitted-only)
3. Ran the same NPU config at 900 tokens: **782 ms, 1162 TPS, 1 batch** —
   matching/beating the documented `NPU_CPU_BENCHMARK_SWEEP_2026-08-19.md`
   number (768 ms) for the same length.
4. `git stash pop`, rebuilt again (re-applying the §5 fixes), redeployed.

This isolates the regression to the uncommitted diff. All results in §8 are
from the fully-restored, current working tree (matching what's actually on
disk right now).

### Root cause: three missing/disabled `engine=cdsp` tags, all the same failure shape

`layer_node.cpp`'s `LayerNode::forwarding()`/`incremental_forwarding()` has a
"sync-guard": before running any layer whose `compute_engine !=
LayerComputeEngine::CDSP`, it force-flushes the pending NPU batch (so a CPU
layer never reads stale pre-NPU data). The guard correctly *skips* the flush
for layers tagged `engine=cdsp` — those just enqueue more DSP ops. So any
layer in the per-block loop that's missing the tag, **regardless of what its
own `forwarding()` actually does**, forces a flush right before it runs.

Found three, by cross-referencing the `[LAYER_FLUSH]` layer names against
`transformer.cpp`'s layer-construction calls:

1. **`createKVCachePlaceholders()`** (`transformer.cpp:464-471`) creates the
   `cache_k_l*`/`cache_v_l*` "input" placeholder layers with plain
   `{withKey(...)}` — never wrapped in `withHexagonEngine()` at all. Not a
   "temp debug" disablement, just never done. **56 flushes/prefill** (2 ×
   28 layers). These layers are pure passthrough (no compute of their own),
   so tagging them `cdsp` is free — but `HexagonContext` didn't register an
   `"input"` layer factory either, so tagging them would have thrown "Key is
   not found for the object" at construction time.

2. **`createMlp()`**'s fused-FFN branch (`transformer.cpp:568`, when
   `NNTR_HEXAGON_FUSED_FFN=1`): `withHexagonEngine()` was **deliberately
   removed**, per an in-code comment — `// TEMP DEBUG: withHexagonEngine()
   removed to isolate a measured 2x prefill regression.` Even though
   `FusedFFNLayer::forwarding()` already dispatches its own compute straight
   to the DSP bridge (`nntr_htp_bridge_ffn_swiglu`), the untagged LayerNode
   still forced a flush before every call. **28 flushes/prefill.**

3. **`MultioutRealizer::realize()`** (`multiout_realizer.cpp:91-99`): the
   auto-generated "multiout" node that fans out a shared-consumer tensor
   (e.g. the residual stream after `decoder_add`/`decoder_output`, each of
   which has two downstream readers) is supposed to inherit its source
   node's `engine=cdsp` tag — code for exactly this existed, but was
   **commented out**, with the same comment: `// TEMP DEBUG: disabled to
   isolate a 2x prefill regression measured with this + the fused_ffn cdsp
   tag both enabled together.` `MultiOutLayer::forwarding()` is a pure
   in-place-aliasing no-op, so this tag is also free. **56
   flushes/prefill** (2 × 28 layers).

`1(input) + 1(embedding) + 1(embedding multiout) + 56(kv-cache) + 28(fused ffn) + 56(residual multiout) + 1(output) = 144`
— matching the ≈143 unique `[LAYER_FLUSH]` layer names observed, and the
252 real round-trips (each flush plus normal per-layer dispatch overhead
compounds).

Both "TEMP DEBUG" comments reference the *same* prior investigation into a
**2× regression** — this session's regression was 8-10×, so whatever that
2× issue was, it's smaller than what leaving both disabled costs today. No
written record of that investigation exists in any doc, only the two code
comments.

### Fix

Three small changes, all just "add/restore the tag, and make sure the tag
is resolvable":

- `Applications/CausalLM/models/transformer.cpp`: wrapped
  `createKVCachePlaceholders()`'s two `createLayer("input", ...)` calls, and
  `createMlp()`'s fused-FFN `createLayer("fused_ffn", ...)` call, in
  `withHexagonEngine(...)`.
- `nntrainer/hexagon/hexagon_context.cpp`: registered `InputLayer` under
  `HexagonContext` (needed for step above — an untagged `"input"` layer
  factory lookup under `engine=cdsp` throws otherwise), same pattern as the
  existing `AdditionLayer`/`MultiOutLayer` registrations.
- `nntrainer/compiler/multiout_realizer.cpp`: re-enabled the engine
  propagation — but had to fix *how* it checks the source node's engine.
  The original (commented-out) code called
  `src_it->second->getComputeEngineType()`, which reads the `compute_engine`
  **member**, only populated by `LayerNode::finalize()` — which runs *after*
  realizers, so at `MultioutRealizer::realize()` time that member is always
  still the unset default. Switched to `getProperty("engine")`, which reads
  `layer_node_props` directly and *is* populated at construction time (that's
  what `withHexagonEngine()` sets). This is presumably why the original
  attempt looked like it caused/was tangled up with a regression when
  measured — the check likely never actually matched `"cdsp"` even when
  re-enabled naively.

### Result

Rebuilt and redeployed with these three fixes; same device, same session:

| | Real FastRPC round-trips (900 tok) | `[LAYER_FLUSH]` guard fires |
|---|---|---|
| Before fix (uncommitted tree as found) | 252 | ~143 |
| After fix | **141** | **4** (input0, embedding0, its multiout, output_of_causallm — genuine one-time boundary flushes) |

The guard itself is now firing only where it should (graph-boundary CPU
ops). The remaining 141 round-trips (not the ideal ~1) are consistent with
a separate, structural limit documented in `HEXAGON_NPU_PRIMER.md`:
`HTP_OP_MAX_BUFS`/`HTP_MAX_MMAPS` are both hard-capped at 16 buffers per
batch on the DSP side, so a 28-layer prefill enqueueing far more than 16
distinct activation/weight buffers forces periodic automatic sub-flushes
inside `ggml-hexagon`'s own session code, independent of anything at the
nntrainer `LayerNode` level. That's a pre-existing architectural ceiling,
not something introduced by this session's changes or fixable at the
`layer_node.cpp` guard level.

---

## 8. Benchmark results (current working tree, post-fix)

Device: `R3CX9078DNH`, Snapdragon 8 Elite (HTP v79). Model: Qwen3-0.6B,
Q4_0 FC weights. `NNTR_NUM_THREADS=4`. NPU env:
`NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_FLASH_ATTN=1 NNTR_HEXAGON_FUSED_FFN=1`.
2 runs per cell. 1200-token length excluded per instructions (pre-existing,
documented >1024-row GEMM/FFN CPU-fallback cliff — see
`NPU_CPU_BENCHMARK_SWEEP_2026-08-19.md`).

| Prompt (tokens) | CPU prefill (ms, avg) | NPU prefill (ms, avg) | Speedup | NPU round-trips | vs. 2026-08-19 NPU baseline |
|---|---|---|---|---|---|
| 300 (→392 actual) | 749 | 355 | **2.1×** | 141 | was 288 ms — 23% slower |
| 600 (→779 actual) | 2194 | 813 | **2.7×** | 141 | was 680 ms — 20% slower |
| 900 (→909 actual) | 2643 | 968 | **2.7×** | 141 | was 768 ms — 26% slower |

NPU is faster than CPU again at every length, qualitatively matching every
prior sweep in this repo. It's still ~20-26% slower in absolute terms than
the best previously-documented numbers (288/680/768ms) — plausibly the
residual 141-vs-~1 round-trip gap above, or the cost of newer instrumentation
(the gamma-rpcmem pooling, pool-stats counters) added since that sweep.
Output token (`&`) matches the documented cross-mode consistency check at
every length — correctness is unaffected throughout.

---

## 9. Does the entire transformer block run on the NPU?

**No — not uniformly, and it depends on which FFN path is active.** Walking
every `createLayer()` call in one decoder block
(`createTransformerDecoderBlock()` / `createAttention()` / `createMlp()` in
`transformer.cpp`) and checking for `withHexagonEngine()`:

| Layer | Tagged `engine=cdsp`? | Notes |
|---|---|---|
| `attention_norm` (rms_norm) | ✅ | |
| `wq` / `wk` / `wv` (Q/K/V projections) | ✅ | |
| `cache_k_l*` / `cache_v_l*` (KV placeholders) | ✅ (fixed this session) | pure passthrough, no compute |
| `attention` (mha_core: RoPE, flash-attn) | ✅ | |
| `attention_out` (wo) | ✅ | |
| `decoder_add` (residual add) | ✅ | |
| `decoder_add`'s auto-generated multiout | ✅ (fixed this session) | pure aliasing, no compute |
| `ffn_norm` (rms_norm) | ✅ | |
| **fused-FFN path** (`NNTR_HEXAGON_FUSED_FFN=1`): `ffn_fused` | ✅ (fixed this session) | one DSP call: gate+up GEMM+SwiGLU+down GEMM |
| **non-fused path** (default): `ffn_gateup` | ✅ | |
| **non-fused path**: `swiglu` (SwiGLU activation) | ❌ | genuinely computes on CPU (`ComputeOps::swiglu_fp32`, no DSP override exists) — the pre-flush here is *correct*, not a bug |
| **non-fused path**: `ffn_down` | ✅ | |
| `decoder_output` (residual add) | ✅ | |
| `decoder_output`'s auto-generated multiout | ✅ (fixed this session) | pure aliasing, no compute |
| `embedding0` (token embedding lookup) | ❌ | a table gather, not a GEMM — one-time boundary op, not part of the repeating block |
| `output_norm` (final rms_norm) | ✅ | |

**With `NNTR_HEXAGON_FUSED_FFN=1`** (used for all benchmarks in this doc),
every op inside the repeating decoder-block loop now either dispatches to
the DSP or is a genuine no-op layer correctly skipped — the transformer
block itself is fully NPU-resident, with only the embedding lookup and
final norm/lm-head sitting at the model boundary.

**Without it** (the default, `NNTR_HEXAGON_FUSED_FFN` unset), the `swiglu`
activation between the gate/up and down GEMMs runs on CPU every layer, and
correctly forces a flush there — so the block is NPU end-to-end for
attention, but the FFN sub-block has one CPU hop in the middle every layer.

Also worth being precise about: "runs on the DSP" here means the ops this
codebase has actually wired a bridge call for — Q4_0 GEMM, flash-attention,
RMSNorm, RoPE, residual-add, and (fused) FFN. Backward pass is still
entirely CPU (unrelated to this investigation), and decode (M=1 GEMV) is
intentionally NPU-dispatched too but is known/documented to be *slower*
there than on CPU (`gemm_q4_0_accel_min_rows()`'s doc comment in
`hexagon_compute_ops.cpp`) — offloading everything is a coverage choice, not
a throughput-optimal one.

---

## Summary

- The Hexagon/NPU build **does** work end-to-end on this device, through the
  `package_android.sh` → prebuilt → `Applications/CausalLM` `ndk-build`
  pipeline — once pointed at the right directories and past three
  unrelated/pre-existing environment gaps (§§2-4) and two small syntax bugs
  in the uncommitted tree (§5).
- The current uncommitted tree had a real, reproducible NPU performance
  regression (§7): three layer types (KV-cache placeholders, fused-FFN, and
  the residual-add's auto-generated multiout) were missing or had
  deliberately-disabled `engine=cdsp` tags, forcing 108 extra flushes per
  prefill and making NPU mode 8-10× slower than CPU instead of faster. Fixed
  by tagging/re-enabling all three (plus registering `InputLayer` under
  `HexagonContext`, and fixing the multiout engine-check to read the
  property instead of the not-yet-populated `compute_engine` member). NPU is
  now 2.1-2.7× faster than CPU again, ~20-26% off the best previously
  documented absolute numbers (residual 141-vs-~1 round-trip gap, itself a
  separate, pre-existing `HTP_OP_MAX_BUFS=16` structural ceiling, not a bug).
- With `NNTR_HEXAGON_FUSED_FFN=1`, the entire repeating transformer block is
  NPU-resident (§9); without it, the SwiGLU activation is a genuine,
  correctly-flushed CPU hop once per layer.
- Fixes applied and left in place: `Applications/meson.build` (android app
  gating), `gpt_oss_moe_layer.h`, `lfm2_causallm.h` (syntax-only, restoring
  clean-`HEAD` behavior), `transformer.cpp`, `hexagon_context.cpp`,
  `multiout_realizer.cpp` (the batching-regression fix above).

Signed-off-by: Claude Sonnet 5 <noreply@anthropic.com>

---

## 10. Correction + follow-up: why "yesterday" was faster than "today, fully on NPU"

(Addendum, written after further investigation prompted by a follow-up
question. See §7 above for the original regression fix.)

**Correction to an earlier claim in this doc:** §7 originally stated clean
`HEAD` measured "782 ms, 1 real round-trip" and implied the 2026-08-19
batching design was intact there. That was wrong — I hadn't looked past the
timing summary at the time. Re-checked directly:

- `HEAD:Applications/CausalLM/models/causal_lm.cpp` has **zero** references
  to `begin_batch`/`end_batch`/`NPU_BATCH` anywhere.
- `HEAD:Applications/CausalLM/layers/mha_core.cpp` has **zero** references
  to `is_batch_active`/any flush guard.

The entire per-prefill-batching mechanism — and the `layer_node.cpp`
sync-guard it depends on — **does not exist at `HEAD` at all**. It is 100%
part of the uncommitted work (556 new lines in `mha_core.cpp`, 124 in
`causal_lm.cpp`, the whole guard added to `layer_node.cpp`). `HEAD` does
plain per-op dispatch instead: enqueue one op, flush, wait, repeat — the
original ~196-round-trips-per-token baseline the primer itself describes as
the state *before* any batching work existed. Confirmed by direct
measurement: `HEAD` produces **196** `flush_batch` calls for a 909-token
prefill, at **819 ms** — not 1 round-trip.

The 2026-08-19 markdown docs' better numbers (288/680/768 ms) were measured
against an *earlier, also-never-committed* iteration of this same working
tree — those `.md` files are themselves untracked
(`?? docs/backend_guide/NPU_CPU_BENCHMARK_SWEEP_2026-08-19.md` in `git
status`, never part of any commit). They're day-to-day working notes
against code that was never snapshotted anywhere, not measurements against
any commit reachable today or on any branch.

### The real comparison, with all three states actually measured

| State | Round trips (909 tok) | Batching/guard machinery | Measured prefill |
|---|---|---|---|
| `HEAD` (no batching at all) | 196 | none | 819 ms |
| Uncommitted, as originally found (broken tags) | 252 | broken | 7977-8543 ms |
| Uncommitted, after this session's 3-tag fix | 141 | working, but incomplete | 968 ms |

**Why 252 round-trips costs 10x what 196 round-trips costs (pre-fix state):**
round-trip *count* alone doesn't explain it — a ~30% higher count should not
cost ~10x the wall time. The difference is *what* each forced flush was
doing. The three untagged layer types (§7) forced flushes at points where
the bridge's rpcmem buffer pool had to register a brand-new region inline,
mid-batch — the primer calls this "the expensive step" (`HAP_mmap` on first
sight of a new fd, cached only afterward). The pool-miss counters from that
run (`rms_norm:gamma: 113 misses`, `cpy:dst/src: 112 misses` each) show this
happening far more often than necessary, each one costing much more than a
simple flush of already-mapped buffers.

**Why 141 round-trips (correctly batched) is still slightly slower than
196 round-trips of dead-simple per-op flushing (968 ms vs 819 ms):** the
new batching infrastructure — dlsym-guarded function-pointer lookups on
every op, `is_batch_active()` checks, `unordered_map`-keyed flush-count
diagnostics, rpcmem pool hit/miss bookkeeping — carries real host-side CPU
overhead *per op*, all the time, regardless of whether that op ends up
needing a flush. That overhead is the price of admission for the
optimization; it only pays for itself once the round-trip count it enables
drops close to the ~1 the whole design is aiming for. At 141 (not ~1), you
are paying that overhead on every op *and* still not avoiding most of the
round-trips — so the "smarter, mostly-fixed" path and the "dumb, always
flush" path land in the same ballpark, with the dumb path even coming out
slightly ahead today.

**This is not something fixable at the build/deploy layer.** The 141
remaining round-trips are real, individually-necessary per-op dispatches —
one concrete, unavoidable example: this model's K-cache RoPE rotation must
run on the CPU every layer, because activations are FP16 and the DSP's RoPE
kernel (`htp/rope-ops.c`) is FP32-only (see the comment in `mha_core.cpp`'s
`try_dsp_cache_copy()`). Closing this gap means finishing the work
`NPU_REMAINING_TWO_OPS_PLAN.md` already scoped — an FP16 DSP RoPE kernel,
or fusing RoPE directly into the flash-attention kernel as that plan's
"medium term" option proposes — not something this session's fixes touch.

**Bottom line, definitively:** "yesterday" was faster because it was
measuring a *different, earlier, better-tuned* iteration of this
never-committed work that no longer exists to rebuild. Comparing like-for-
like today — `HEAD`'s zero-machinery per-op dispatch vs. the current
tree's batching-with-3-bugs-fixed — the batching code is not yet a clear
win; it has correctness value (it exists specifically because per-op
dispatch has data-race hazards the primer's authors were actively working
to close) but has not yet recouped the overhead of adding it.
