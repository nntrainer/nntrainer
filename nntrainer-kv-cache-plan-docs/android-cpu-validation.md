# Android Arm CPU validation

**Historical scope:** the main body records the preserved 2026-10-28 builds.
The 2026-10-29 review-followup results are a separate section at the end and
in [cpu-performance.md](cpu-performance.md); their current runtime and counts
do not replace the older manifests.

This continues the Linux CPU evidence in [linux-cpu-execution.md](linux-cpu-execution.md)
and [qwen3-cpu-validation.md](qwen3-cpu-validation.md). Qwen3-0.6B remains the
validation model. Generated checkpoints, binaries, trace logs and logits remain
under ignored build directories; model files are not committed.

2026-10-28 result: **64 successful model probe runs and 80 native unit tests** on
`bacd420f`, including eight model
runs on the final corrected runtime with token and full-logit bitwise parity
against the earlier successful FP32-attention profile. Ten initialization
rejections from two intermediate guard builds are preserved separately below;
both integration issues were corrected before the final pass.

This records the preserved 2026-10-28 correctness builds. The later default
packing policy and controlled timing work are documented in
[cpu-performance.md](cpu-performance.md). The current validator requires
`--serial` explicitly; bounded ARM tile packing is selected explicitly with
`--cpu-kv-tiled` when replaying the historical tiled profile.

## Environment review (2026-10-28)

The selected device is **bacd420f**. Every command uses `adb -s bacd420f`.
The device reports model `23116PN5BC`, Snapdragon SM8650, Android 15 / API 35,
and `arm64-v8a`. Initial available RAM was approximately 11.2 GiB and available
storage approximately 227 GB. The host NDK is `/opt/android-ndk-r28c`, revision
`28.2.13676358`.

The device test directory is `/data/local/tmp/nntr_kv_cpu_bacd420f`. The harness
does not clean unrelated device paths or stop other device processes. It uses
an explicit timeout around each probe process.

The Linux plan requires several adjustments for this target:

- FP32 model bytes can be reused. Quantized FC packing is ISA dependent;
  the Linux X86-packed Q4 checkpoint must be regenerated with `--isa ARM`.
- Android CPU execution must be selected explicitly (`NNTR_ENGINE=cpu`).
  Android deployment alone provides no evidence of GPU/NPU memory sharing.
- ARM flash attention currently has a full-prefix K/V packing path. The
  harness records its copy bytes rather than assuming the Linux x86 direct
  consumer optimization applies to Arm.
- Half-bit KV storage differs numerically from the Hugging Face FP32 KV
  reference. Cross-architecture logits need a numerical comparison. A same
  device live-versus-restored session can require byte-identical logits.
- The Android runtime and the probe must come from the same build. Host-built
  shared libraries and the Linux probe cannot run on the phone.

## Checkpoint preparation

The official checkpoint is `Qwen/Qwen3-0.6B`, pinned revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Existing verified local source
and FP32 conversion are reused without another download.

```bash
mkdir -p build-kv-android/models
NNTR_ENGINE=cpu NNTR_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  build-kv-cpu/Applications/CausalLM/nntr_quantize \
  build-kv-cpu/models/qwen3-0.6b-fp32 \
  --output build-kv-android/models/qwen3-0.6b-q40 \
  --fc_dtype Q4_0 --embd_dtype Q6_K --lmhead_dtype Q6_K \
  --isa ARM --no_sidecar \
  --output_bin nntr_qwen3_0.6b_q40_embdq6k_arm.bin
```

The quantizer completed successfully. Activations remain FP32; FC weights use
Q4_0 and tied embedding/LM-head weights use Q6_K.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Original HF `model.safetensors` | See Linux manifest | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| `nntr_qwen3_0.6b_fp32.bin` | 2,384,199,680 | `d60649a75e7faf3a4d50f046dbea6607d53e645fe4999ab181e4a651b84da7e9` |
| `nntr_qwen3_0.6b_q40_embdq6k_arm.bin` | 375,614,464 | `6964b206ce3b813418400b9728ca2b92f1874e1227389237f549095e4d1f40d5` |

The generated manifest is `build-kv-android/models/qwen3-android-manifest.json`.
The Android harness checks SHA256 again on the device before inference. An
existing remote model can be reused with `--skip-model-push`; its hash is still
checked. The two model weight files consume approximately 2.6 GiB together.

## Validation harness

`tools/kv_cache/validate_qwen3_android.py` reuses the read-only
`qwen_cpu_probe.cpp` logits observer. It prepares non-thinking greedy HF
references locally, stages model configurations for Android, deploys the
runtime and model assets to the dedicated directory, then pulls raw float
logits and JSON reports. Device stderr retains KV trace witnesses. It records
runtime and probe hashes, source revision/diff hash, explicit environment,
model hashes, positions, finite logits and first-step numerical errors.

Prepare references without using the phone:

```bash
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
  /home/jayden/miniforge3/envs/rkv-hf/bin/python \
  tools/kv_cache/validate_qwen3_android.py --serial bacd420f \
  --prepare-only --output build-kv-android/validation/default \
  --tokens 32 --threads 4 --include-session --snapshot-resume
```

The prepared HF outputs are:

| Case | Reference output, excluding EOS |
| --- | --- |
| Arithmetic: `Compute 2 + 2.` | `2 + 2 = 4.` |
| Korean capital question | `대한민국의 수도는 서울입니다.` |
| Repeated context, 164 input tokens | `France.` |
| Two-turn session | `hello` → `hello` |

Runtime execution additionally needs `--probe <Android-probe>` and
`--runtime-dir <directory-containing-Android-shared-libraries>`.
Use `--reuse-reference` to reuse the prepared requests. `--prefill-chunk 17`
tests several transactional append steps; `--min-prefill 1` selects the flash
consumer for small prefill lengths. `--linux-reference <CPU-harness-output>`
adds first-step numerical comparisons for matching model/case report names.
`--cpu-attention-fp32` sets `NNTR_CPU_ATTENTION_FP32=1`; the default explicitly
sets it to `0` for legacy numerical behavior. Baseline runs use
`--allow-compatibility-trace` to keep functional parity separate from direct
cache access evidence. This mode still checks manager backing geometry.

FP32 token mismatches against HF are reported as a failed parity check.
Quantized token mismatches are recorded for review because quantization can
change greedy choices. Nonfinite logits fail both profiles. Live versus
fresh-model snapshot restore requires identical token IDs, every float-logit
byte, and saved/restored/continued positions. Traces check owner geometry,
producer allocation addresses and aliases; compatibility history packing is
reported as a remaining copy rather than hidden behind an alias claim.
For `flash-tiled`, full-prefix history copying and bounded kernel tile packing
are separate counters. The strict Qwen trace auditor requires zero history
copying and a positive tile-pack count with at most 65,536 scratch bytes per
worker (two half-bit planes, key tile ≤128, head width 128). A tile path with
nonzero tile packing is not described as copy-free.
This bound is verified for Qwen in a fresh probe process with this fixed tile
and head geometry. Thread-local vectors retain capacity from earlier calls;
a process that previously ran a larger head width or key tile may retain a
higher scratch high-water mark. The measurement is not a universal bound for
arbitrary model changes inside one long-lived process.

## Native cache unit tests

The NDK test build uses three explicit targets added to the Android test
makefile. The facade snapshot fixture uses `TMPDIR` instead of assuming `/tmp`
exists on Android.

```bash
ANDROID_NDK=/opt/android-ndk-r28c /opt/android-ndk-r28c/ndk-build \
  -C Applications/CausalLM/jni NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs \
  NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk \
  NDK_APPLICATION_MK=./Application.mk \
  CAUSALLM_PREBUILT_LIB=../../../builddir_app/cpu/libcausallm.so \
  unittest_kv_cache_core unittest_kv_cache_manager unittest_kv_ring -j4
```

All three binaries ran on `bacd420f` in the dedicated `checks` directory with
`NNTR_ENGINE=cpu`, `TMPDIR=checks` and `LD_LIBRARY_PATH=checks` (absolute paths
in the actual invocation). Results: **core 18 + facade 25 + ring 22 = 65 PASS,
0 failures, 0 skips**. XML, stdout and the binary/hash manifest are retained in
`build-kv-android/validation/checks/validation.json` and adjacent files. The
corresponding host facade 25-test regression also passes.

The Android probe is built from the same standalone app compile/link metadata:

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py --build builddir_app/cpu \
  --output builddir_app/cpu/validation/probe/qwen_cpu_probe
```

The probe translation unit disables fast-math so finite-logit observations
remain meaningful. Baseline probe and runtime binaries are preserved under
`build-kv-android/validation/baseline/`.

## Execution status

Host ARM checkpoint conversion, all four HF reference cases, native cache
unit tests and **16 baseline model runs have completed successfully** (eight
default runs plus eight chunk-17/forced-flash runs).
The baseline runtime is from before the Arm attention changes, with
`NNTR_CPU_ATTENTION_FP32=0`. The full command was:

```bash
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
  /home/jayden/miniforge3/envs/rkv-hf/bin/python \
  tools/kv_cache/validate_qwen3_android.py --serial bacd420f \
  --reuse-reference --output build-kv-android/validation/baseline \
  --tokens 32 --threads 4 --include-session --snapshot-resume \
  --allow-compatibility-trace \
  --probe build-kv-android/validation/baseline/qwen_cpu_probe \
  --runtime-dir build-kv-android/validation/baseline/runtime \
  --linux-reference build-kv-cpu/validation/qwen3-post/default
```

| Case | Android FP32 | Android ARM Q4 | Compared with HF |
| --- | --- | --- | --- |
| Arithmetic | `2 + 2 = 4.` | `2 + 2 = 4.` | Both token sequences equal |
| Korean | `대한민국의 수도는 서울입니다.` | Same | Both token sequences equal |
| 164-token context | `France.` | `France` | Q4 omits the period, as on Linux |
| Same session, two turns | `hello` → `hello` | Same | Both token sequences equal |

All recorded logits are finite. Android FP32 versus HF first-step maximum
absolute differences are approximately 0.038–0.077; legacy Android narrows
query/key/value intermediates to FP16 even for FP32 models. Cross-architecture
logit bytes are therefore different. ARM Q4 first-step differences from HF are
approximately 6.68–9.53. These differences are numerical observations, not an
accuracy benchmark.

Both model profiles preserve session positions `0 → 21 → 21 → 49`.
Saving at position 21, loading weights into a fresh model and restoring its KV
snapshot produces exactly the same second-turn token IDs and every logit byte
as the live model. Each Android snapshot is 2,408,576 bytes. Each core cache
has 28 owner layers and 58,720,256 backing bytes at capacity 512.

The 164-token baseline prefill uses `flash-packed` and records **18,808,832
history-copy bytes per model run**. Short prompts use the per-row consumer.
Legacy producer traces report `profile=compatibility`, so producer alias and
append-copy counts are **unmeasured**, represented by absent/false witnesses
and `null` copy counts. Baseline functional success must not be interpreted
as direct producer write evidence.

The extra baseline uses the same command with output
`build-kv-android/validation/baseline-chunk17` and the additional flags
`--prefill-chunk 17 --min-prefill 1 --skip-model-push`. All eight cases pass,
including exact same-device live/restored session parity. Requests and HF
reference logits are identical to the default baseline. Flash packing is
observed for chunked prefill; its accumulated history-copy bytes remain in
each report. Chunked arithmetic and Korean answers match HF, and Q4 retains
the same known long-answer punctuation difference.

Reports and pulled logits are retained under
`build-kv-android/validation/baseline/` and `baseline-chunk17/`.

## New Arm consumer and explicit FP32 attention profile

The rebuilt app is preserved at `build-kv-android/validation/post/`, sharing
the baseline engine build. The new ARM consumer reads cache tiles from the
manager allocation and packs only bounded K/V kernel tiles. The default
attention arithmetic remains compatible with the old Android behavior;
preserving FP32 attention intermediates requires `--cpu-attention-fp32`.

**Both legacy-behavior profiles, 16 runs total, pass exact baseline parity**.
Default and chunk-17/forced-flash runs retain the same token IDs and every
recorded logit byte, including session continuations and fresh-model snapshot
restoration. Outputs are retained in `post-legacy/` and
`post-legacy-chunk17/`. Their commands use the baseline settings plus:

```bash
--skip-model-push \
--probe build-kv-android/validation/post/qwen_cpu_probe \
--runtime-dir build-kv-android/validation/post/runtime \
--android-baseline build-kv-android/validation/baseline \
--require-baseline-bitwise
```

The chunk-17 command instead compares against `baseline-chunk17/` and adds
`--prefill-chunk 17 --min-prefill 1`.

For the 164-token default long case, the old full-prefix materialization
counter falls from 18,808,832 bytes to zero. **Tile packing remains nonzero:**
96,337,920 bytes accumulated over the run, with 65,536 K/V scratch bytes per
worker. Query blocks and grouped-query heads reuse history through separate
tile packs. This removes the full-prefix temporary allocation; it does not
establish a reduction in total copy traffic. The tiled path is not described
as all-copy-free or direct kernel consumption of the original cache layout.

**The explicit FP32 attention default profile also passes all eight runs**.
All FP32 tokens match HF; Q4 retains only the known long-answer punctuation
difference. Every instrumented producer and consumer is associated with its
actual owning cache allocation, all 28 owner layers are witnessed per step,
and strict trace audits pass. The direct producer has zero append-copy bytes;
RoPE and conversion into final half-bit cache values remain required writes.
Per-row attention reads directly from that cache. Flash uses bounded tile
packing with zero full-prefix materialization. Session snapshot restoration
remains token- and logit-byte-identical on this profile.

These runs are retained in `post-fp32/` and use the new runtime plus
`--cpu-attention-fp32`, without `--allow-compatibility-trace`. Comparison with
the old Android numerical baseline is recorded without requiring bitwise
equality, because preserving FP32 intermediates intentionally changes logits.

FP32 default first-step maximum absolute differences from HF are 0.01894
(arithmetic), 0.00976 (Korean), 0.00992 (long context) and 0.02373 (session).
The corresponding available Linux comparisons are approximately
0.00822 / 0.00489 / 0.00844 for arithmetic / Korean / session. Neither backend
uses HF's native FP32 cache, so bitwise cross-backend equality is not required.

The explicit FP32 chunk-17/forced-flash profile passes all eight runs and
strict trace audits, including live/restored session equality. Its command
adds `--prefill-chunk 17 --min-prefill 1` with output
`build-kv-android/validation/post-fp32-chunk17`.

## Direct same-allocation CPU evidence

The final profile forces the per-row consumer for **all eight cases**,
including the 164-token context and two-turn snapshot restoration:

```bash
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
  /home/jayden/miniforge3/envs/rkv-hf/bin/python \
  tools/kv_cache/validate_qwen3_android.py --serial bacd420f \
  --reuse-reference --output build-kv-android/validation/post-fp32-per-row \
  --tokens 32 --threads 4 --include-session --snapshot-resume \
  --cpu-attention-fp32 --min-prefill 999999 --skip-model-push \
  --probe build-kv-android/validation/post/qwen_cpu_probe \
  --runtime-dir build-kv-android/validation/post/runtime \
  --linux-reference build-kv-cpu/validation/qwen3-post/per-row
```

Every run passes. For every recorded append step, all 28 layer K/V allocation
pairs match the manager-owned backing, the producer and consumer report the
same read/write aliases, and **append-copy, full-prefix-copy and tile-pack
counts are all zero**. Necessary RoPE/conversion writes into final cache
storage remain counted separately. This is direct, instrumented evidence of
the CPU producer and per-row consumer using the same manager allocation on
this Android device. It is not evidence of cross-device CPU/GPU/NPU physical
memory sharing.

All 24 opt-in FP32-profile trace files pass the independent strict auditor
after execution. The auditor's original Linux default traces (10 files) still
pass. Counter validation also accepts a bounded tile witness and rejects
missing counters, zero claimed packing and scratch exceeding 65,536 bytes.
Strict harness mode also requires a successful strict-audit result and
explicit zero append/history-copy counts. Fault injection into an actual
device trace rejects a full-prefix `flash-packed` regression, compatibility
profile, nonzero append copy, oversized tile scratch and missing consumer
witnesses. Evidence is retained in
`build-kv-android/validation/strict-gate-negative-checks.json`. All 56 saved
device traces were rechecked with the strengthened gate; they still pass
under their respective strict or explicit baseline compatibility modes.

| Profile | Probe runs | Result |
| --- | ---: | --- |
| Original Android default | 8 | PASS; functional baseline |
| Original Android chunk17 + forced flash | 8 | PASS; functional baseline |
| New runtime, legacy arithmetic, default | 8 | PASS; full-logit bitwise baseline parity |
| New runtime, legacy arithmetic, chunk17 + flash | 8 | PASS; full-logit bitwise baseline parity |
| Explicit FP32 attention, default | 8 | PASS; strict cache trace + snapshot parity |
| Explicit FP32 attention, chunk17 + flash | 8 | PASS; strict trace with bounded tile packing |
| Explicit FP32 attention, forced per-row | 8 | PASS; direct cache aliases, all three copy counters zero |

This pre-final matrix contains **56 passing model probe runs**, alongside
the initial **65 native cache unit tests**. The final section adds the corrected
runtime verification and current 80-test native result. All FP32 outputs match
HF token IDs; the only observed Q4 token
difference in these cases is the previously known long-answer punctuation.
All logits are finite. Fourteen live-versus-fresh-model snapshot comparisons
(two model profiles × seven execution profiles) retain identical token IDs,
every recorded continuation logit byte and saved/restored positions.

The consolidated artifact is
`build-kv-android/validation/android-summary.json`; each profile retains its
raw JSON reports, float logits, stderr trace, runtime/probe hashes and request
hashes. `post-fp32*/strict-trace-audit.json` records the independent audits.
These artifacts and model files remain uncommitted. Timing values are retained
for diagnosis and were not collected as a controlled performance benchmark.

The post-change runtime used for these 40 new-runtime runs is explicitly
preserved in `post/execution-metadata.json`, including runtime/probe hashes,
probe ABI/compile/link metadata, Meson options and the source revision/diff
hash recorded when validation began. Subsequent initialization-only CPU,
inference and batch guards are a separate build; their final smoke results
must be attributed to that build rather than replacing this execution record.

## Final initialization-guard integration check

The final opt-in guard restricts the profile to an explicitly selected CPU
process, inference mode, batch 1, external linear FP16 cache,
positive RoPE theta and FP32 Q/K/V/activation tensors. Sink, INT8, ring and
OHWI cache paths remain outside this profile. The default flag value `0`
preserves legacy behavior.

The final Android profile unit tests add **15 PASS**, covering the supported
configuration, neutral GPU residency metadata on a CPU process, explicit-zero
legacy behavior and rejected configurations. Core/facade/ring 65 and profile
15 were all rerun against the final corrected CPU runtime: **80 PASS,
0 failures, 0 skips**. Device runtime and executable hashes match their frozen
host files. The authoritative current manifest is
`build-kv-android/validation/checks/final-cpu.validation.json`, with adjacent
`final-cpu.*` XML and logs; see
[android-build-checks.md](android-build-checks.md). Earlier
`mha_cpu_profile.validation.json` and `final-guarded.validation.json` remain
historical records for their earlier source/runtime revisions.

The first guarded full-model smoke, preserved in `final-guarded/`, rejected
both FP32 and Q4 during initialization before inference. Its app SHA256 is
`18518528ed856b0a3e3732a0a2883d62e4a92f81385e18ac75c75d2aa5e14321`;
`build.json`, runtime, probe and per-model stderr remain preserved. The common
message is the new batch-1 CPU inference profile guard. This failure exposed
a real graph-initialization integration difference that the direct layer
unit fixtures did not exercise. Investigation identified an additional
existing initialization bug:
`LayerNode::refinalize()` rebuilt `InitLayerContext` without the model's
execution mode, selecting the constructor's default `TRAIN` even for an
inference graph. The first finalize path already passed the correct mode.
The new guard would reject this stale context when refinalization is reached.

A core fix now forwards the graph execution mode through refinalization.
The next rebuilt runtime, preserved in `final-verified/`, reaches
`execution_mode=INFERENCE` and `batch=1` but still rejects all eight model
cases because `layer_engine=GPU` while `process_engine=cpu`. CausalLM marks
`mha_core` as engine-neutral for tensor residency, and `LayerNode::finalize()`
stamps that context GPU for residency bookkeeping even on the CPU process.
The new guard confused that residency marker with actual CPU execution.

This second diagnostic build has engine SHA256
`0b7204882344ffe917cd1fd4e991e84e7e0930f0ad5611fbf31a1727d7479ad8`
and app SHA256
`0c0cb896aab075c7b299e95e7780eb90207841e053d4513146ea2efece029976`.
Its `build.json`, probe/runtime, report and detailed stderr remain preserved.
The mode propagation fix is valid. The redundant layer-context engine check
was removed because it reads residency metadata. Actual process CPU,
inference and batch guards remain required; a GPU residency marker alone
does not mean the CPU profile executes on a GPU. The profile unit case now
explicitly allows that neutral context while continuing to reject a non-CPU
process. These are initialization-check implementation issues, rather than
Qwen checkpoint or conversion errors.

The 56-run pre-guard matrix remains valid evidence for its own preserved
runtime. These two diagnostic builds produced ten initialization rejections
(2 + 8), with no inference outputs; they remain separate from successful
model-validation counts.

### Final corrected CPU runtime: eight successful runs

The corrected core mode propagation and neutral-context interpretation are
both present in `build-kv-android/validation/final-cpu/`. This directory keeps
the final `build.json`, runtime, freshly compiled probe and ABI metadata.

| Final binary | SHA256 |
| --- | --- |
| `libnntrainer.so` | `0b7204882344ffe917cd1fd4e991e84e7e0930f0ad5611fbf31a1727d7479ad8` |
| `libcausallm.so` | `78502cc34a2ee950e3d177186a0bb3c43950376e4df3f6daa1aea21ca24f0cd7` |

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py --build builddir_app/cpu \
  --output build-kv-android/validation/final-cpu/qwen_cpu_probe

PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
  /home/jayden/miniforge3/envs/rkv-hf/bin/python \
  tools/kv_cache/validate_qwen3_android.py --serial bacd420f \
  --reuse-reference --output build-kv-android/validation/final-cpu \
  --tokens 32 --threads 4 --include-session --snapshot-resume \
  --cpu-attention-fp32 --cpu-kv-tiled --skip-model-push \
  --probe build-kv-android/validation/final-cpu/qwen_cpu_probe \
  --runtime-dir build-kv-android/validation/final-cpu/runtime \
  --linux-reference build-kv-cpu/validation/qwen3-post/default \
  --android-baseline build-kv-android/validation/post-fp32 \
  --require-baseline-bitwise
```

All eight FP32/Q4 arithmetic, Korean, long-context and two-turn/snapshot cases
pass with exit code 0. Every token ID and every recorded logit byte, including
live continuations, matches `post-fp32/` exactly. All logits are finite. Every
strict cache trace passes independently; snapshot restoration remains
byte-identical with positions `0 → 21 → 21 → 49`. The long case retains bounded
tile packing; short cases and session decode retain direct per-row reads.
The known Q4 long-answer missing period remains unchanged.

`final-cpu/strict-trace-audit.json` records the additional eight independent
trace audits. The successful total is now **64 probe runs** (56 earlier + 8
final), with **32 strict opt-in trace files** and **16 same-device snapshot
comparisons**. The ten earlier initialization rejections are historical
diagnostic attempts, not successful inference runs. Runtime/source hashes
keep the earlier matrix and final corrected build distinct.

## 2026-10-29 review followup: current candidate

The current candidate adds checked scope/read/write/readiness enforcement to
the actual model execution path and restores **packed as the default**.
Default packing can materialize the full K/V history. Safety enforcement of
checked views must not be confused with a zero-history-copy claim. The
candidate's source/library hashes are frozen in
`build-kv-perf/runtime/android-candidate/build.json`.

Natural-output revalidation uses the same existing reference requests and
logits without an HF forward pass:

| Android current profile | Cases | Result and trace interpretation |
| --- | ---: | --- |
| Legacy-half packed, flags `0/0` | 8 | Full-bitwise baseline parity; compatibility functional validation |
| FP32-attention packed, flags `1/0` | 8 | Full-bitwise FP32 reference parity; history packing remains allowed |
| FP32-attention tiled, flags `1/1` | 8 | Full-bitwise FP32 reference parity and strict alias/copy trace audit |

Flags above are `NNTR_CPU_ATTENTION_FP32` / `NNTR_CPU_KV_TILED`. All 24 cases
retain identical tokens and every logit byte, including live continuations
and snapshot restoration, against their same-arithmetic references. All
logits are finite. The strict direct-cache profile is FP32 attention with
per-row consumption, or FP32 attention with tiled opt-in and separately
accounted bounded tile copies. Packed profiles are functional compatibility
validation rather than a claim of zero history materialization.

Results are retained under `build-kv-perf/correctness/` in
`android-candidate-packed/`, `android-candidate-fp32-packed/` and
`android-candidate-tiled/`; the last directory also has
`strict-trace-audit.json`. A separate Linux candidate suite passes all eight
existing full-logit comparisons in `correctness/linux-final/`. Linux and
Android case counts are kept separate.

The same current Android runtime passes native **85** plus actual Qwen/LFM2
graph **7**, for **92 PASS**, recorded in
`build-kv-android/validation/checks/review-final.validation.json` and its native
and actual-graph manifests. Details and exact commands are in
[android-build-checks.md](android-build-checks.md). These 92 tests are distinct
from the 24 full-model natural-output cases above.

Performance measurements and the initial order-sensitive 4K gate failure are
reported in [cpu-performance.md](cpu-performance.md). All 1K/4K measurements
use a 128-row activation plane and 128-token prefill chunks; they do not
measure an unchunked single 4K forward. Packed stays the default, and tiled
latency/memory observations are not used as a general speedup claim.

Current replay commands now require explicit `--serial` and can select
`--cpu-kv-tiled`. These CLI clarifications are for replay with the current
tool; the original execution environment remains exactly as recorded in each
preserved historical manifest, which was not rewritten.
