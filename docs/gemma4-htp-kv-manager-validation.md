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
