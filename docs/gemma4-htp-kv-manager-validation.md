# Gemma4 26B A4B: HTP and KV manager validation

## State on 2026-11-03

**The rebuilt v81 DSP runs the supplied Gemma4 26B A4B model.** The first
full-model run completed 128 generated tokens at 20.898 tok/s (10 prompt
tokens, 1,734 ms prefill, 6,125 ms generation). DSP queue errors, timeouts,
stale responses and detach/unmap failures were zero.

The work starts at PR [#4428](https://github.com/nntrainer/nntrainer/pull/4428),
revision `e91adaf25a828ea9180edfa384687dc9be2fe8fc`, in branch
`kv/pr4428-htp-cache`, worktree `/tmp/nntr-pr4428-validation`.
The KV core is ported from `ee0e83d3b` (`kv-dram-flash-cache`). The HTP
integration uses its contiguous shared-slab ownership and position/reset
management. DSP int8 cache handles/calibration and resident decode remain
owned by the existing HTP backend. Device session switching, radix prefix
reuse and Flash restoration are **not integrated or validated** by this work.

The later resident-conversation contribution is recorded below under
[Resident KV sessions and request reservations](#resident-kv-sessions-and-request-reservations).
It adds continuation of one live conversation; it does not add device
session switching, radix reuse or Flash restoration.

## Environment and model

| Item | Observed value |
|---|---|
| Phone | `R5KL20NFRCK`, SM-S948N, SM8850, Android 16 |
| NDK | 27.2.12479018, arm64-v8a, Android API 29 |
| Hexagon SDK | 6.4.0.2, official Linux archive |
| Hexagon compiler | 19.0.04, confirmed by `hexagon-clang --version` |
| Host tools | GCC/G++ 12.3 and QEMU 6.2, unpacked Ubuntu packages |
| Installed tools | `/home/seungbaek/toolchain/nntrainer-htp-20261103` |
| Host model | `/home/seungbaek/Downloads/gemma4` |
| Phone model | `/data/local/tmp/nntr_pr4428_kv_validation/models/gemma4` |
| Phone ARM runtime | `/data/local/tmp/nntr_pr4428_kv_validation/pr4428` |
| Evidence directory | `/tmp/nntr-pr4428-validation-results` |

Source `toolchain/nntrainer-htp-20261103/env.sh` under the user's home directory
to reuse the installed SDK, NDK and host tools. System compiler selection was
not changed. The SDK and package archives have recorded SHA256 hashes in
`dependencies.sha256` under the evidence directory.

The supplied pack uses native `TL2CX_WH` experts, `QS4CX_WH_AFFINE` dense/head
weights, Q4_0 embedding, q8 attention and signed Hadamard rotation (`both`,
scope `all`). There are 30 layers and 128 experts, with eight selected per
token. The model file is 6,604,561,848 bytes. Its final 2,368 bytes are the
activation-scale trailer declared in `nntr_config.json`; the conversion
report's older 6,604,559,480-byte payload/hash should not be compared with
the entire updated file as if they covered the same bytes. This revision
does not consume the `act_scale_*` keys; activation-scale optimization is
not established by this smoke.

All ten transferred files match the host SHA256 manifest. The source files
were not edited. With user authorization, only `/data/local/tmp/ddtree` and
`/data/local/tmp/ddtree_260903` were removed to make room. After model/runtime
staging the phone had approximately 4.7 GiB free. The existing Gemma4 GPU
models were retained.

## Validation results

| Check | Result and scope |
|---|---|
| PR host build, HTP disabled | Passed. CausalLM models, quantizer and eight selected test executables built. |
| Host model tests | 125 registered: 124 passed, one deprecated QINT4 fixture skipped. Initial seven failures were missing `nntr_quantize`; building it resolved them. |
| Host legacy KV facade | 33/33 passed. Includes the PR's shared allocation/window management tests. |
| Host KV/attention helper tests | 43/43 passed across six executables: FP16 plan 10, FP16 registry 3, quantized registry 6, quantized plan 5, integer softmax 13, CPU quantized attention 6. |
| Android PR helper tests | 37/37 passed, zero skips. Compiled directly from the pinned PR: FP16/quantized registries, attention plans and integer softmax. These execute on ARM, not HMX/HVX hardware. |
| Android existing manager | Core 63/63 and facade 26/26 passed, zero skips, from the existing CPU manager revision. Covers sessions, radix branches and Flash restore/error paths. Separate runtime from the PR build. |
| HTP host-check script | Kernel/routing/arena/graph/token checks reached the router handoff check. The first run used GCC 11 and stopped at unsupported `_Float16`; the GCC 12 run reached an LSan sandbox restriction. Neither original invocation is reported as a complete pass. |
| Router handoff continuation | Reran the affected ASan/UBSan check outside the sandbox restriction: 194 cases passed. Synthetic DSP hits in this host fixture are not phone DSP execution. |
| Remaining host checks | TL2 refresh pair passed; aarch64 ABS MAX under QEMU passed, 1,800 cases with zero bad. |
| Optional libnative checks | Passed after SDK installation: 5,760 HVX GEMV cases, 3,456 u8i2 cases, 72 aligned FC cases, 640 TL2 quantization cases. All three injected mutants were rejected. This validates host emulation, not device throughput. |
| HTP in-process build | Passed after the fixes described below, GCC/G++ 12 and `werror=false`. |
| In-process targeted checks | Gemma4 HTP tiny reference/input contract 4/4 and KV facade 33/33 passed; targeted process exit reported zero mapped memory. |
| In-process full model suite | 125 registered: 124 passed, one skip, successful process exit when rerun outside the sandbox. The original sandbox-only teardown timeout did not reproduce. |
| Android HTP ARM engine/app | Built with the SDK and beta 1 ARM SDKL link dependency. CLI and continuation checker loaded on the phone; subsequent production v81 DSP rebuild also passed. |
| Full 26B Android smoke | Passed after rebuilding the DSP and correcting its search path: 128 generated tokens, 20.898 tok/s. Initial failed attempt is retained in the evidence directory. |
| Integrated Android KV core/facade | 63/63 core and 36/36 facade tests passed using the integrated Android shared libraries, zero skips. These are ARM ownership/algorithm tests; the separate full-model comparison exercises DSP execution. |
| Integrated host KV core/facade | 63/63 core and 36/36 facade tests passed with successful process exit. Initial sandbox-only core teardown timeout did not reproduce outside the sandbox. |

Logs and XML files are under the evidence directory, including `android/`,
`host-unit-tests-complete.log`, `host-checks-gcc12.log`, `libnative-checks.log`,
`router-handoff.log`, `inproc-model-tests-complete.log` and
`android/htp-model-smoke.log`. The host-check script stopped only near its
end, but the separately rerun tail and optional checks are not a new single
uninterrupted successful invocation.

## Build fixes on top of the PR

1. Format the staging warning into a bounded buffer before `ml_logw`.
   The Linux logging macro concatenates a string literal with its format;
   passing a template parameter directly fails compilation. Android's
   logging macro did not expose the problem.
2. Keep `nntr_hvx_kv_release_q` with the shared session lifecycle code.
   Its implementation is unchanged. The in-process backend excludes the
   quantized prefill RPC source and previously could not link this symbol.
3. Exclude the scalar RMSNorm/RoPE row stand-ins from the in-process build,
   which already compiles their production kernel definitions. Keep them
   for the standalone host-check fixtures.

Changed lines were checked with clang-format 14 and `git diff --check`.
The production v81 DSP was subsequently rebuilt using the explicit micro API provider described below.

## Existing device DSP audit (subsequent correction)

The phone already has v81 skeletons. The earlier explanation that a missing
local SDK alone prevented running was incomplete: none of these device files
had been staged in the smoke's DSP search path. The existence of a skeleton,
its RPC coverage and the loader search path are separate checks.

All 44 discovered skeleton paths were grouped by SHA256 into 30 distinct
binaries. Every representative was pulled and checked against its device
hash. A read-only ELF audit decoded each exported `nntr_hvx_slim` QAIC
interface: method count, ordered names, argument types and directions,
scalar/buffer counts and alignments. The reference metadata was generated
from the pinned PR IDL with the SDK's qaic and compiled as a metadata-only
Hexagon ELF. That reference ELF has unresolved kernel implementations and
is **never deployed or executed**.

| Skeleton | Actual QAIC methods, including open/close | Comparison with PR's 134 methods |
|---|---:|---|
| `g4ops_1016/ship/libnntr_hvx_skel.so` | 125 | All existing ordinals/signatures match; nine appended methods absent. ELF flags explicitly report V81. |
| `nntr-w4a16-max/max-e/libnntr_hvx_skel.so` | 120 | Fourteen PR methods absent; some shared methods have different ordinals/signatures. ELF flags explicitly report V81. |
| All 30 unique device binaries | One with 125, 17 with 120, 12 with 60 | No complete match to the current interface. |

For the 125-method skeleton, `arena_attach` **is present**, at ordinal 6,
with the same signature as the PR. Consequently, the original arena error
44 cannot be attributed to an absent `arena_attach` RPC.

A small Android probe built with the PR's real ARM stub loaded the
125-method skeleton from a separate `rpc-probe` directory. Actual device
results were:

```text
open_rc=0
session_info_rc=0
release_prefill_scratch_rc=-2147482604 hex=0x80000414
moe_set_hadamard_signs_rc=-2147482604 hex=0x80000414
close_rc=0
```

The SDK defines `0x80000414` as `AEE_EUNSUPPORTED` (API not implemented).
These are actual RPC responses, not conclusions from string searches.
The current PR uses ordinal 129 for the supplied rotated pack's signs and
131 for prefill scratch release; neither is in the 125-method interface.

Using the same working directory and probe, the original empty DSP search
path instead failed `open` with `0x80000406` (`AEE_EUNABLETOLOAD`). Pointing
only `ADSP_LIBRARY_PATH` at the staged skeleton made open/query/close pass.
The earlier full-model run therefore had an unverified DSP load and must
not be used to diagnose a specific missing RPC from its later error 44.

Audit evidence: `rpc-audit/report.json`, `current-methods.json`, per-binary
`device/*.methods.json`, `probe-absent-methods.log` and
`probe-empty-cwd-{empty-path,dsp-path}.log` under the evidence directory.

## Rebuilt v81 DSP and successful execution

The public HexKL beta 1 package lacks v81 support; compiling its v79 archive
into a v81 ELF did not initialize HMX. Instead, the already working v81
`g4ops_1016/ship/libnntr_hvx_skel.so` supplies only the HexKL micro API as an
explicit dynamic dependency. Its SHA256 is
`a167afcf746d9c7f1aa2f74f7bb4f454d78b79d8000e664dcc5c2067a2dba01e`.
It is copied unchanged as `libnntr_hexkl_provider.so`. The complete current
nntrainer DSP source and current QAIC interface are freshly compiled; this
is not the older 125-method skeleton reused as the RPC endpoint.

`dsp/build.sh` accepts an optional `HEXKL_MICRO_DSO` path, validates its
architecture, links its basename as a dependency, and binds the new
skeleton's own symbols locally with `-Bsymbolic`. The undefined-symbol
guard allows only micro symbols actually exported by that provider, in
addition to the existing DSP runtime imports. The default static-library
build is unchanged. Provider headers must match its ABI: this provider uses
the three-argument `hexkl_micro_hw_init`, matching the PR's beta 2 declarations.
A matching official v81 static library can replace this local validation
arrangement when available; no vendor binaries or SDK headers are committed.

Reproduce this host's DSP build:

```bash
source ~/toolchain/nntrainer-htp-20261103/env.sh
export HEXKL_ROOT=/tmp/nntr-pr4428-validation-results/hexkl-device
export HEXKL_SDK_VER=device-v81
export HEXKL_MICRO_DSO=$HEXKL_ROOT/lib/libnntr_hexkl_provider.so
NNTR_HTP_DIAGNOSTICS=0 bash nntrainer/tensor/htp_backend/dsp/build.sh
```

The new ELF reports V81 and all 134 QAIC methods match the current IDL's
names, ordinals and wire signatures. A real phone probe successfully opened,
queried, released prefill scratch, set/reset Hadamard signs and closed. Session
info reports HMX enabled, 8 MiB VTCM, eight HVX contexts and FP16 rate 8.

Both DSP files must be in `ADSP_LIBRARY_PATH`, and the skeleton must also be
available from the working directory used by the CLI. ARM libraries use
`LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64`; placing vendor libraries
before system libraries caused an unrelated Android framework symbol mismatch.
The SDK `libcdsprpc.so` link stub is never deployed; execution uses the phone's
real vendor FastRPC library.

The supplied example's EOS configuration contains token 1 only, so its
128-token smoke repeats turns after answering the greeting. Performance
comparisons below use deterministic fixed-length generation and compare IDs,
without modifying the original weight pack.

## KV integration and comparison

`NNTR_HTP_KV_MANAGER=1` selects the imported core manager for HTP shared
FP32/FP16 planes. Its allocator wraps the same HTP shared allocation/free
callbacks. Existing Tensor views pin the core allocation, and aliases refer
to the same storage. There are no per-token cache copies or extra RPCs.
`NNTR_HTP_KV_MANAGER=0` selects the original PR slab owner in the same binary.
Allocation failure on the core path fails explicitly instead of falling back
to ordinary heap memory. DSP int8 KV and backend completion/registration stay
on their original path. Window compaction, dropped-row maxima and bases are
preserved by the facade; unit tests cover compaction, ownership, reset and moves.

The imported session/radix/Flash core remains covered by its CPU tests. Its
host-plane snapshot alone cannot restore an HTP session: a future backend
snapshot must also restore int8 KV/calibration, window bases and resident
decode state. Those features are not enabled by this switch.

The controlled phone comparison uses three conditions: the PR baseline plus
build fixes, the integrated binary with ownership OFF, and that same binary
with ownership ON. All use the same newly rebuilt DSP/provider and weights,
greedy sampling, eight CPU threads, 16 cached experts, disabled cache growth
and fixed lengths. Cases are 1,024 input + 64 decode tokens and 2,048 input +
128 decode tokens. After one warmup per condition, five measurements per
condition alternate forward/reverse condition order. Warm runs do not establish
cold-storage performance. The CLI's generated-ID capture does not force a CPU
logit readback. Logs, IDs, thermal snapshots and `runs.json` are under
`android/compare/` in the evidence directory.

All 36 runs completed: six warmups and 30 measurements. Every captured ID
sequence matched across all three conditions and repetitions within each case:
65 IDs for p1024 and 129 for p2048 (prefill prediction plus decode steps).
The first smoke's 20.898 tok/s is a different short greeting/configuration;
it is not the throughput of these fixed-budget long-input cases.

Median results, five measurements per row; decode excludes the first decode
step and reports the CLI's steady per-token latency:

| Input / decode steps | Condition | Prefill ms | TTFT ms | Steady decode ms/token | E2E ms | Peak MiB |
|---|---|---:|---:|---:|---:|---:|
| 1024 / 64 | PR baseline | 1866 | 1943.09 | 85.8203 | 7864 | 3285.4 |
| 1024 / 64 | Core OFF | 1854 | 1920.88 | 84.3639 | 7707 | 3343.9 |
| 1024 / 64 | Core ON | 1805 | 1871.04 | 85.0536 | 7756 | 3321.5 |
| 2048 / 128 | PR baseline | 3649 | 3743.52 | 84.1437 | 15553 | 3380.7 |
| 2048 / 128 | Core OFF | 3633 | 3734.82 | 85.1473 | 15560 | 3379.3 |
| 2048 / 128 | Core ON | 3622 | 3725.58 | 84.1358 | 15525 | 3382.0 |

Core ON versus OFF: prefill -2.64% / -0.30%, steady decode +0.82% / -1.19%,
and E2E +0.64% / -0.22% for p1024 / p2048. Observed median regressions are
below the provisional 3% threshold, but this is not a general no-regression
guarantee. The phone's skin temperature rose from 30.1 to 45.2 C, with
reported skin thermal status 0 through 3. Alternating condition order helps
with drift but does not remove thermal effects or supply statistical confidence.
For example, p1024 steady decode ranges were 80.16–86.58 ms for the PR,
83.67–86.52 ms OFF, and 82.05–86.08 ms ON. The small differences must not
be advertised as a speedup.

Peak memory is the CLI's RssAnon + accelerator accounting, not just ru_maxrss.
The ON owner reports 30 layers and 519,045,120 bytes (495 MiB) of shared KV
planes. Both paths retain the original DSP quantized cache implementation.
A machine-readable record with every run, ranges, token hashes, settings and
DSP hashes is committed in
[`validation/gemma4-htp-kv-manager-20261103.json`](validation/gemma4-htp-kv-manager-20261103.json).
These runs validate single-context contiguous ownership only; they do not
validate session switching, a real-context window rollover, or prefix/Flash
restore on the NPU.

## Running the staged manager build

```bash
adb -s R5KL20NFRCK shell 'cd /data/local/tmp/nntr_pr4428_kv_validation/core-htp &&
  LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 \
  ADSP_LIBRARY_PATH=/data/local/tmp/nntr_pr4428_kv_validation/core-htp \
  NNTR_NUM_THREADS=8 NNTR_HTP_KV_MANAGER=1 \
  ./nntr_causallm /data/local/tmp/nntr_pr4428_kv_validation/models/gemma4'
```

Use `models/p1024` or `models/p2048` for the fixed comparison configurations.
Evidence includes `android/htp-model-new-dsp-smoke.log`,
`rpc-audit/probe-new-v81-provider.log`, `android/v81-dsp-provider-build.log`,
`android/working-v81-dsp.sha256`, `android/integrated-{core,facade}.xml`,
and `core-integration-unit-tests.log`.

## Resident KV sessions and request reservations

Contribution recorded on 2026-11-03, implemented in `729cabcaa` on top of
PR #4428. The change preserves one conversation's DSP INT8 KV, quantization
scales, graph and absolute positions across successive requests. The final
emitted token remains pending attention and is automatically consumed before
the next request's new input. This avoids recalculating the previous history.

Request reservations round the required decode capacity to 64 rows. Global
caches grow only when needed; sliding caches discard complete tiles while
retaining their attention window. Both operations preserve the retained
INT8 bytes and scales without requantization and run between requests. Reset
also clears and synchronizes sliding bases in the host and the reused DSP
graph. The C++ CLI supports request sequences and reports committed positions,
pending-token state, output IDs and allocated device KV payload.

### Observed contribution

Validated on SM-S948N, Android 16, Hexagon v81 with the supplied Gemma4 26B
A4B TL2 rotated model. The model context limit was 16,384 tokens, the initial
prompt had 8,194 tokens, and both compared configurations kept the expert
pool fixed and used the existing attention arithmetic.

| Measurement | Reference | With the contribution | Observed reduction |
|---|---:|---:|---:|
| Follow-up TTFT | 23,251.06 ms, full history recomputed | 1,969.97 ms, resident KV reused | 91.53% |
| Allocated DSP INT8 K+V payload | 266.25 MiB, full context reservation | 186.875 MiB, request reservation | 79.375 MiB (29.81%) |

The follow-up comparison used the same token history. Full recomputation
processed 8,226 tokens after resetting an already loaded model; continuation
processed 21 new user tokens and one pending feedback token. These are single
scenario observations, not averages over a broader workload. The payload
comparison measures allocated DSP K+V bytes, excluding scales, attention
scratch and the host FP16 prefill allocation, whose policy is unchanged.

### Validation and scope

- Four consecutive resident requests returned the expected parcel code.
  Every output token and committed KV position matched the full-capacity
  configuration. The full-recomputation follow-up returned the same output
  token IDs as the reused request.
- Global growth and a sliding drop of 64 rows executed on the device.
  Long-context reset and independent requests with both options disabled
  passed regression checks.
- Host attention, reservation and configuration checks passed, including
  exact output preservation after resizing/compaction and targeted
  ASan/UBSan checks. Leak detection was excluded because the execution
  environment prevents LeakSanitizer under ptrace.
- The v81 DSP skeleton, Android engine, C++ application and CLI built
  successfully. Changed C/C++ files pass clang-format 14.

The contribution covers one synchronous resident conversation per model.
Additional input uses the existing resident single-token path. Resizing
temporarily retains old and replacement buffers. Device snapshots, arbitrary
rewind, concurrent session switching and device radix reuse are outside this
contribution.

Enable `htp.kv_session=true` and `htp.kv_request_reserve=true`; pass only new
input on subsequent `run()` calls and use `setSessionTokenPosition(0)` to
start a fresh conversation. The ARM RPC client and DSP skeleton must both be
rebuilt. See the [configuration and C++ request-sequence guide](gemma4-htp-code-guide.md#resident-gemma4-kv-sessions).

The committed [evidence record](validation/gemma4-htp-kv-session-20261103.json)
contains the exact measurements, configuration, device binary hashes,
implementation revision and correctness checks. Local raw evidence remains
under `/tmp/nntr-kv-session-20261103`.

## DSP INT8 Flash checkpoints (2026-11-04)

The new C++ `saveDeviceKVCache(path, park)` / `loadDeviceKVCache(path)`
path was validated on the Gemma4 26B A4B HTP fused model on SM-S948N,
Android 16, Hexagon v81. These measurements use the actual NPU/DSP model,
not the Qwen CPU Flash path. Expert capacity was fixed at 16 per layer,
expert growth disabled, existing attention arithmetic retained, and resident
sessions and request reservations enabled.

Each case has three measured trials after model/graph warmup. The original
prompt has N-2 tokens and generates two tokens. A checkpoint contains N-1
committed INT8 KV rows plus one pending feedback token. The recompute request
prefills exactly the same N input IDs. Resident and restored requests consume
the pending token, and their output IDs and committed positions match exactly.
At 4096 an assistant-turn delimiter prevents decoded generated text from
merging with the original prompt during re-tokenization. Token identity is
checked before any sample is accepted. Rebuilding INT8 scales during full
recomputation is allowed; its output is recorded independently.

| History tokens | Snapshot MiB | Recompute TTFT ms | Cold restore ms | Cold restore + request TTFT ms |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 15.04 | 1,696.55 | 44.14 | 115.60 |
| 512 | 56.29 | 1,890.64 | 175.03 | 326.89 |
| 1024 | 111.29 | 2,250.09 | 361.43 | 583.52 |
| 4096 | 147.54 | 6,749.82 | 509.38 | 672.36 |

Values are medians. Cold preparation uses file `fsync` followed by
`POSIX_FADV_DONTNEED`; `/proc/self/io` records at least the full snapshot's
bytes as storage reads on every cold restore. Restore time includes reads,
checksum, RPC transfer, DSP allocation/import and atomic commit. Model loading
and prior export are excluded from hit TTFT. Including measured export/park,
file synchronization/advice, restore and resumed TTFT gives 965.47 ms at
1024 and 1,230.46 ms at 4096. This is observed latency, not a NAND durability
or energy measurement. Clocks are not locked; three trials do not establish
production latency percentiles. Expert-cache residency can change between
paths. Recompute and warm/cold order alternate by repetition.

The size increase from 1024 to 4096 is limited by retaining sliding-layer
windows while global layers retain the longer history. Relative savings
are consequently not monotonic across lengths. Unaligned N+2-token cases
were also measured separately; they require an additional prefill chunk and
are not used in the table above.

Validation includes:

- All restored output IDs and committed positions match resident continuation.
- Park releases device K/V allocations; the reported K+V payload becomes zero.
  Quantizer metadata and decode scratch remain in the parked context.
- A corrupted KV payload fails checksum verification after candidate import;
  full before/after active checkpoint files are byte-identical. A subsequent
  healthy restore and continuation succeed.
- Four host FP16 snapshot entry points reject parked resident DSP sessions.
- Host round trips preserve bytes and exact attention outputs in ordinary,
  integer and Q8 query modes, including a 64-row sliding compaction.
- Targeted ASan/UBSan checks pass (`detect_leaks=0`, matching the sandbox's
  LeakSanitizer limitation). ARM engine/application and v81 DSP builds pass.

This provides explicit, synchronous checkpoints in the same loaded model
process. Automatic DSP radix/LRU/Flash budget integration and restoration
across model processes are not implemented. The checksum-protected file
contains a loaded-graph cookie; unrelated models cannot adopt it.

[Measured data, configuration and checks](validation/gemma4-dsp-flash-20261104.json)
include raw per-trial values and binary hashes. Native benchmark sources,
commands, logs and CSV are in `/tmp/nntr-dsp-flash-20261104/`.

## Alternating independent DSP conversations (2026-11-04)

Gemma4 26B A4B HTP fused was tested on SM-S948N / Android 16 / Hexagon v81
using the committed DSP checkpoint implementation (`93aef5b15`). The same
loaded model alternated two independent conversations. A remembered
`ZEBRA-731`; B remembered `ORCHID-842`.

For each scenario, the reference process ran A0/A1/A2/A3, reset, then
B0/B1/B2/B3 without checkpoints. The switched process ran
A0/B0/A1/B1/A2/B2/A3/B3. Each switched request saved and parked its updated
snapshot; each continuation restored its own latest snapshot first.

| Scenario | Initial A/B tokens | Matching switched requests | Flash restores | Final A/B positions | Final sliding A/B bases |
| --- | --- | --- | --- | --- | --- |
| Short | 25 / 25 | 8 / 8 | 6 | 132 / 135 | 0 / 0 |
| Different history lengths | 2040 / 506 | 8 / 8 | 6 | 2158 / 617 | 1080 / 0 |

Every switched response's token IDs, committed before/after positions,
pending-token state and prefill token count matched its independent
reference. All continuation replies contained their own code and none
contained the other conversation's code. Every park reported zero device
K/V payload. Snapshot headers confirmed that A and B used the same loaded
graph cookie and retained distinct positions and sliding bases; the long A
conversation also executed a 64-row sliding compaction.

Across both scenarios, 16 reference requests and 16 switched requests ran,
including 12 Flash restores and 16 parks. This verifies explicit sequential
snapshot switching within one loaded model process. OS file caches were
not evicted; this correctness check is separate from the cold-storage
latency measurements above. Concurrent inference, automatic session-ID
management and automatic DSP LRU policies are outside this check.

[Per-request results, request fixtures and binary hashes](validation/gemma4-dsp-session-switch-20261104.json)
are recorded with the validation. Runner, commands and raw logs are in
`/tmp/nntr-dsp-multisession-20261104/`.

## A/B switching latency versus full recomputation (2026-11-04)

A native C++ trace replay compared A/B Flash switching with rebuilding the
same complete token history before every request. The inputs and expected
resident outputs come from the independent-conversation test above. Native
retokenization must match every recorded input ID. Model/graph loading is
excluded; a BaseStreamer timestamps the first decoded delta without enabling
a logits processor or changing the normal DSP token-ID path. Its external
clock includes independent-prompt reset/cleanup before `CausalLM::run()`.

Each configuration replays A0/B0/A1/B1/A2/B2/A3/B3 three times in both modes,
alternating mode order by repetition. Initial A0/B0 seeds are excluded from
the latency summary. Each direction has nine samples (three different follow-up
turns, each repeated three times), rather than nine copies of one prompt.

Cold switching saves/parks and fsyncs the outgoing conversation. Before
restoring an incoming file it requests `FADV_DONTNEED`; process I/O counters
verify at least the complete snapshot size as storage reads on all 54 restores.
"Full switch TTFT" sums the previous other conversation's save/park and fsync,
incoming cache advice, restore, and run-to-first-delta. It assumes serialized
outgoing I/O and excludes generation of the previous response and benchmark
logging. Durability synchronization is stronger than the save API's default.

For the long scenario, a 4096-row prefill plane lets the recompute history fit
one prefill chunk. This is the primary comparison; the original 2048-row
configuration is retained as a sensitivity result. Medians in milliseconds:

| Scenario / direction | Incoming history tokens | Recompute TTFT | Restore + new-input TTFT | Full switch TTFT | Full-switch faster samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short B to A, plane 128 | 54–122 | 1,667.90 | 1,737.25 | 1,765.84 | 3/9 |
| Short A to B, plane 128 | 54–124 | 1,674.39 | 1,637.66 | 1,670.17 | 3/9 |
| Short B to long A, plane 4096 | 2080–2148 | 4,520.13 | 2,688.10 | 3,125.72 | 9/9 |
| Long A to short B, plane 4096 | 536–606 | 2,260.16 | 2,174.94 | 2,979.99 | 0/9 |

Across each six-continuation trace, full switch costs are 2.6–4.4% higher
than recomputation in the short scenario. With a long A and short B and the
4096-row plane, they are 4.7–11.0% lower (paired reduction median 8.75%).
These are sums of first-token latencies over the six continuations, not
throughput or complete-response wall times. At plane 2048, long-A recompute
crosses a chunk boundary and the paired reduction increases to 24.2–29.3%;
that configuration would overstate the advantage over the larger prefill
block and is not the primary result.

Restoring a long conversation benefits from avoiding its full prefill, but
saving the outgoing long conversation can make the next short conversation
slower. In the 4096-plane case the outgoing save/park plus fsync median is
382.58 ms for B-to-A and 760.27 ms for A-to-B. In addition, resident continuation
currently processes the 20-plus new input tokens through the single-row DSP
path. Short full-history prompts use batched prefill, so reuse need not save
time even before outgoing I/O is charged. These observations motivate
comparing reuse and recompute costs rather than treating every Flash hit as
a guaranteed latency improvement.

All 72 switched requests (including seeds) match resident reference output
IDs and positions. Of 72 recomputed requests, 69 match the reference IDs;
short B turn 3 adds a repeated thought-channel marker in all three repeats
while retaining the correct code. Inputs are identical, but full-history
INT8 calibration can differ from retained-cache continuation. This comparison
therefore measures TTFT and does not claim equivalent complete-generation
cost. No session-code leakage was observed in the previously validated
switched traces. Clocks are unlocked and samples are exploratory; no
production percentile or universal benefit is established.

[Raw trials, per-turn and per-trace comparisons, settings and hashes](validation/gemma4-dsp-switch-performance-20261104.json)
are recorded. Native runner, build commands, logs and CSV are in
`/tmp/nntr-dsp-switch-perf-20261104/`. No inference implementation changes were
needed for this measurement.

## Direct request-triggered dump and restore (2026-11-04)

The preceding A/B performance test saved/parked after each response and summed
that outgoing cost with the next request's restore and TTFT. It did not start
one timer with the outgoing KV still resident. This follow-up implements the
literal arrival-triggered sequence:

1. Leave A's K/V allocated after A's response; do not dump or park it yet.
2. At B request arrival, verify the outgoing K/V payload is nonzero and start
   one monotonic request timer.
3. Dump/park A and fsync its file, then restore B's snapshot.
4. Process B's new input and stop that same timer at the first streamed delta.
5. Leave B's K/V resident until the next A request arrives.

Target file-cache eviction is prepared before arrival without changing the
outgoing resident KV. All 18 target restores read at least the complete file
size from storage. This test enforces an outgoing resident conversation; it
does not saturate all physical DRAM or simulate an OOM budget.

The same frozen input histories and model configuration were compared across
three profiles, rotating profile order over three repetitions. Each direction
has nine continuation samples. Gemma4 HTP fused on SM-S948N / Android 16 / v81,
4096-row prefill plane, fixed 16 experts/layer, no expert growth. Medians are
request-arrival-to-first-delta milliseconds:

| Direction | Discard outgoing, recompute incoming | Dump outgoing, recompute incoming | Dump outgoing, restore incoming |
| --- | ---: | ---: | ---: |
| Long A to short B | 2,120.72 | 2,712.59 | 2,836.52 |
| Short B to long A | 4,398.18 | 4,470.61 | 2,954.86 |

Incoming A histories contain 2080–2148 tokens; B histories contain 536–606.
The first alternative permits losing the outgoing KV. The latter two preserve
it on Flash, so they isolate the choice of recomputing or restoring the target
under that preservation requirement. For A-to-B, restore is about 4.6% slower
than preserving A and recomputing B in this measurement. For B-to-A, restore
is about 33.9% faster than preserving B and recomputing A. Clocks are unlocked
and these small samples do not establish universal latency guarantees.

In total, 72 inference requests (excluding warmup) ran, including 54 summarized
continuations, 42 outgoing dump/park operations and 18 storage-backed restores.
All restored outputs and positions match the independent resident reference.
Every successful request leaves target K/V resident for the next arrival, and
every dump verifies that outgoing K/V allocations were released. No inference
implementation change was needed; only the native benchmark's timing and
checkpoint scheduling changed.

[Direct intervals, stage timings, protocol and settings](validation/gemma4-dsp-direct-switch-20261104.json)
are recorded. Native runner, compilation command, logs and CSV are in
`/tmp/nntr-dsp-direct-switch-20261104/`.

## Crossed outgoing/incoming history lengths (2026-11-04)

A 3x3 experiment varied A and B initial user-history prompts independently
across exactly 128, 1024 and 3072 tokens. The resulting resident KV lengths
were 129, 1026 and 3074 tokens; B's complete recompute inputs, including its
initial generated reply and the fixed new question, were 153, 1050 and 3098
tokens. Every input fits the same 4096-row prefill plane in one chunk.

Both timed policies preserve A: dump/park/fsync A, then either fully
recompute B or restore B's snapshot and process the new question. Before
each policy A actually runs and finishes a response, leaving its KV resident.
A single request timer starts before dumping A and stops at B's first decoded
delta. B's file cache is dropped before arrival, and all 27 restores verify
storage reads at least the full snapshot size. B continuation output IDs and
positions must equal its independent resident reference. These are not sums
of separately measured intervals and do not use a discard-A baseline.

Gemma4 HTP fused, SM-S948N / Android 16 / Hexagon v81, 8 threads, fixed 16
experts/layer, growth disabled, default attention arithmetic. Three repetitions
per cell, rotating source/target order and alternating policy order. Values
below are medians in seconds, shown as save+recompute to save+restore:

| A initial tokens / B initial tokens | 128 | 1024 | 3072 |
| --- | --- | --- | --- |
| 128 | 1.964 → 1.971 | 2.709 → 2.508 | 5.675 → 2.610 |
| 1024 | 2.498 → 2.466 | 3.192 → 2.904 | 6.180 → 3.096 |
| 3072 | 2.589 → 2.561 | 3.360 → 3.165 | 6.533 → 3.282 |

Median-based time reduction from restoring B instead of recomputing it:

| A initial tokens / B initial tokens | 128 | 1024 | 3072 |
| --- | ---: | ---: | ---: |
| 128 | -0.3% | 7.4% | 54.0% |
| 1024 | 1.2% | 9.0% | 49.9% |
| 3072 | 1.1% | 5.8% | 49.8% |

Small B histories show essentially no advantage in these samples. Around
1024 initial tokens the advantage is modest; at 3072 it is substantial.
Longer A adds a common dump cost to both policies and raises both absolute
latencies. The fixed new question, initial reply format and one-resume
protocol differ from the preceding progressive multi-turn trace; its precise
crossover must not be generalized to other prompts or generation policies.
Clocks are unlocked, with only three samples per cell, so small differences
near zero are not production guarantees.

All 54 timed requests passed their input/state checks, all 27 cold restores
matched resident output IDs and positions, and each pair started from matching
A committed lengths and snapshot sizes. Initial target seeding, independent
reference continuation and preparing A are outside the B request timer.

[All samples, actual lengths, stage times, inputs and binary hashes](validation/gemma4-dsp-length-grid-20261104.json)
are archived. Native runner, build commands, raw logs and CSV are in
`/tmp/nntr-dsp-length-grid-20261104/`.
