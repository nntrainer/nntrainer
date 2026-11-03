# Gemma4 26B A4B: HTP and KV manager validation

## State on 2026-11-03

**Full-model generation and the manager performance comparison have not
passed.** The supplied model is on the phone and its files match the host.
The Android HTP engine, CLI and continuation checker build and load, but the
model smoke stops while registering an expert arena on the DSP.

The work starts at PR [#4428](https://github.com/nntrainer/nntrainer/pull/4428),
revision `e91adaf25a828ea9180edfa384687dc9be2fe8fc`, in branch
`kv/pr4428-htp-cache`, worktree `/tmp/nntr-pr4428-validation`.
The separately tested CPU KV manager is `ee0e83d3b` from
`kv-dram-flash-cache`. It has **not** been connected to the HTP path in this
worktree. Passing its independent tests does not establish HTP compatibility.

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
| In-process full model suite | All 125 test bodies completed (124 passed, one skip), but the process did not exit within the 60-second Meson timeout. This is an unresolved teardown issue; the overall run failed by timeout. |
| Android HTP ARM engine/app | Built with the SDK and beta 1 ARM SDKL link dependency. CLI and continuation checker loaded on the phone. This does not validate a v81 DSP build. |
| Full 26B Android smoke | Failed during expert arena registration; no generated tokens or throughput result. |

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
The production v81 DSP rebuild still needs the package below.

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

## Current full-model blocker

The public download used for HexKL supplied beta 1, with v73/v75/v79
libraries and no v81 target. PR #4428's production DSP build needs the
matching **HexKL beta 2**, including
`lib/6.4.0.2/hexagon_toolv19_v81/libhexkl_micro.a`. No PR-matched DSP
skeleton has been built or staged here. An old device skeleton is not
treated as a compatible substitute. Beta 1's ARM SDKL library was used
only to finish the ARM build and loader check.

The real-model smoke reached a 542-op graph and then reported:

```text
Failed to repack weights: HTP arena: cannot place an expert slot (1 MiB).
nntr_hvx_arena_attach(16 MiB) failed: err=44 (0x0000002c)
mapped=0 MiB in 0 chunks, RSS=2018 MB
```

The full-model error 44 is not itself proof of an absent RPC. The audit
above separately confirms both the original DSP loading failure and the
older skeleton's unsupported appended methods. A matching ARM client/DSP
skeleton must be tested before diagnosing a model, memory-budget or
manager problem. This run never used the new KV manager.
There was no active inference process left after the smoke.

Reproduce that diagnostic with the staged runtime:

```sh
cd /data/local/tmp/nntr_pr4428_kv_validation/pr4428
LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 \
ADSP_LIBRARY_PATH=/data/local/tmp/nntr_pr4428_kv_validation/pr4428 \
NNTR_NUM_THREADS=8 ./nntr_causallm \
  /data/local/tmp/nntr_pr4428_kv_validation/models/gemma4
```

System libraries must precede vendor libraries in this loader path; a
vendor-first path failed on a `libandroidfw`/`libbase` symbol mismatch.
Do not deploy the SDK's link-only `libcdsprpc.so` to the phone.

## Manager integration and comparison still required

The existing core manager owns plain host storage and supports FP32/FP16.
The HTP path uses accelerator-visible slabs, per-layer window bases and
dropped-row maxima, plus separate DSP int8 KV handles and calibration.
Resident decode bypasses the normal host attention layer calls. A manager
integration must preserve these paths, including their lifetimes and
completion, rather than copying host cache bytes every token.

Once a matching DSP runtime works, compare the fixed PR baseline, the
integrated build with the manager disabled, and the same build with
contiguous manager ownership enabled. Require identical generated token
IDs and verify the actual ownership/HTP dispatch path. Use identical
weights, prompts, sampler, expert budget, thread count and context.
Alternate at least five measured runs per condition after warmup; record
thermal state, TTFT, prefill throughput, decode ms/token and memory.
Separate loading, cold I/O and warm execution. A provisional goal is
median prefill/decode regression within 3%, with uncertainty reported
when run-to-run variation is larger.

Only after that compare two sessions A/B/A/B, then radix reuse and Flash
restore. Their backend snapshot must also restore int8 KV/calibration,
window base and resident decode state. Neither single-session NPU
compatibility nor performance parity is claimed by the results above.
