# PaliGemma / π0.5 implementation audit — 2026-11-13

The implementations are **not yet optimal**. Numerical validation of a model
and optimal execution of that model are different acceptance criteria.
The tested topology and equations can be retained, but model execution,
backend integration, operator dispatch and memory ownership still need work.
It would be premature to freeze the model code and focus only on arithmetic
kernels. All experiments below used isolated libraries/binaries; production
model, runtime, kernel and configuration sources were unchanged during this audit.

The preceding benchmark is in [the validation report](paligemma-pi05-20261113.md).
Those were first requests after loading in separate processes, not steady-state
throughput or a proof that the selected implementation was optimal.

## What is already sound

The tested layer counts, dimensions, prefix/causal masks, RoPE, cache mapping,
normalization and flow equations agree with the PyTorch references within the
recorded tolerances. In particular:

- π0.5 runs the backbone once and reuses its prefix K/V for all ten expert
  steps. The expert computes only the 50 action rows; it does not rerun the
  complete backbone at every step. Its unused vocabulary projection is omitted.
- PaliGemma projects only the final active row into vocabulary logits and
  reuses its K/V cache during decode.
- Complete outputs, not just a matching token prefix, are checked. This is
  evidence for the tested observations, not a proof for all possible inputs.

No architecture redesign or checkpoint change is needed to address the
execution defects identified here.

## Controlled counterexamples to optimal execution

| Experiment | Baseline | Isolated change | Result and interpretation |
|---|---:|---:|---|
| PaliGemma CUDA, first request | 3.718 s | 3.061 s | Only set `NNTR_CUDA_ELTWISE=1`; existing residual/scalar device paths become available. Full output checks pass. |
| π0.5 CUDA, first request | 3.761 s | 3.146 s | Same setting; every velocity and action passes. This is a configuration/backend-connection improvement, with no new kernel. |
| π0.5 CPU expert step, unpinned | 413.7 ms | 235.4 ms | Replace the model's scalar attention loops with existing `nntrainer::sgemm`; no common kernel changes. Full velocity/action checks pass over three requests. |
| π0.5 CPU expert step, fixed 8 P cores | 486.9 ms | 307.7 ms | Repeat the BLAS mapping test with identical affinity; full checks pass. |
| π0.5 CPU full request, fixed 8 P cores | 11.923 s | 10.980 s | Median of three loaded-model requests with only the model attention mapping changed. |
| π0.5 CPU, BLAS attention plus AVX dispatch | 11.923 s | 9.922 s | Combined experiment; full outputs pass. Attribution must use the separate experiments above. |

The CPU BLAS experiment changes only the attention implementation in the
model-specific host expert. The scalar code allocates a scores vector inside
each query/head loop: 50×8×18×10 = 72,000 allocations per action request.
The experiment uses 180 score matrices and strided GEMMs for QK and PV. Model
weights, precision, cache content, masks, number of layers and Euler steps
stay the same. This directly falsifies the claim that model implementation
work is exhausted and only new kernels remain.

The GPU setting connects a fast path already present in `AdditionLayer` and
scalar multiplication. The model's CUDA profile currently sets readable
activations, synchronization and K/V allocation policy, but does not enable
this elementwise path. Tests here cover the base observations; making the
setting a deployment default still needs the ordinary regression workflow.

## GPU phase attribution

These are diagnostic third-request times from the same loaded model.
NNTrainer uses its existing profile before the elementwise setting change;
PyTorch uses the benchmark's matched language/vision/expert compute precision.
Both perform the same complete request and validate outputs.

| GPU component | NNTrainer | PyTorch |
|---|---:|---:|
| PaliGemma vision | approximately 1.78 s | 0.056 s |
| PaliGemma language, prefill plus decode | approximately 1.68 s | 0.100 s prefill + 0.213 s decode, plus generation orchestration |
| π0.5 vision | approximately 1.77 s | 0.058 s |
| π0.5 backbone prefix | approximately 1.11–1.16 s | 0.099 s |
| π0.5 ten expert steps | approximately 0.35 s | 0.221 s |

This points to the shared vision path and prefix execution before the GPU
expert attention kernel. In the traced π0.5 expert, the custom attention
kernel accounts for roughly 0.20 s across ten steps: significant within the
expert, but only about 6% of the original approximately 3.2 s warm request.
A perfect replacement of that kernel alone cannot close the whole gap.

CUPTI records approximately 0.34–0.37 s of GPU kernel activity during the
approximately 1.77–1.78 s NNTrainer vision phase. This is not a claim that the
remaining interval is entirely CPU computation. It includes execution outside
recorded kernels, host work, memory/coherency handling and instrumentation.
Host CUDA API durations overlap GPU activity and must not be added to it.
GPU kernel intervals can themselves include memory stalls.

Code and runtime evidence agree on several causes:

- Vision FCs, LayerNorm and GELU do reach CUDA: FC pointer diagnostics show
  managed input/weight/output and CUPTI records cuBLAS/normalization kernels.
  The remaining problem is not simply that all dense computation is on CPU.
- Vision FC bias addition and FP32 residual addition still use host tensor
  operations. GPU-produced managed activations repeatedly become host inputs,
  followed by another GPU operation. These common FP32 paths need device
  dispatch and coherent lifetime/residency planning.
- Custom vision attention invokes per-head cuBLAS QK/PV operations. With 16
  heads this means 32 GEMMs and a softmax launch per block, instead of using
  a batched/fused attention primitive. Choosing an existing batched primitive
  is an operator-mapping/integration task as well as backend work; its exact
  speedup has not been measured here.
- π0.5 prefix attention reads Q/K/V into host vectors, computes FP32 RoPE on
  CPU, transposes, uploads, performs attention, and downloads the result.
  Across a request, the instrumented host Q/K/V reads take about 107–114 ms
  and RoPE about 45–47 ms. Explicit attention copies total 82.58 MiB per
  prefix, excluding managed-memory migration. FP32 prefix K/V retention is
  intentional for accuracy; a device implementation should retain that math.
- Copying the exported prefix into the expert cache and preparing its device
  cache take only a few milliseconds in warm requests. The CPU time-condition
  work is approximately 2–3 ms per ten-step request, and the host Euler
  arithmetic is negligible. The small per-step action/velocity transfers
  are not the first performance priority.
- Lazy expert weight upload contributes roughly 145 ms of device copy time
  to the first expert step. This is a cold-request/lifecycle issue; it should
  not be confused with steady expert arithmetic throughput.

The FP16 elementwise setting cuts the warm π0.5 prefix from approximately
1.16 s to 0.46 s and PaliGemma language execution from approximately 1.68 s
to 0.73 s in the instrumented runs. It leaves the FP32 vision problem largely
untouched. These traced numbers describe attribution, not new deployment
benchmark guarantees.

## CPU dispatch and scheduling

The generic FP32 `CpuComputeOps::geglu` evaluates scalar `std::tanh`, even
though this tree already contains an AVX implementation of `tanh_gelu_mul`.
Connecting that existing function in an isolated common-library build passes
the full PaliGemma outputs and π0.5 velocity/action checks.

A separate layer-profile comparison measured PaliGemma prefill GeGLU at
1.501 s through the scalar path and 0.054 s through AVX. However, total request
times in that pair were 10.57 s and 10.91 s: other phases were slower in the
second run. The isolated operator improvement is established; a universal
end-to-end speedup from this dispatch change alone is not. Thread placement,
BLAS/OpenMP interaction and phase scheduling need controlled follow-up.

The fixed-affinity experiments bind to logical CPUs 0,2,4,6,8,10,12,14, one
thread per P core, with eight NNTrainer/BLAS/OpenMP threads. Their whole-request
times differ from the original unpinned benchmark and must not be substituted
into that table. PyTorch uses MKL/oneDNN while this build uses OpenBLAS, so the
remaining matrix-multiply gap also requires backend-level comparison.

## Memory finding: loader peak is not steady model memory

The earlier 27.74 GiB PyTorch π0.5 figure was a process high-water mark during
construction and checkpoint loading. After loaded-model requests, host RSS
was much lower. The observed warm values are:

| Model/device | NNTrainer host RSS | PyTorch host RSS |
|---|---:|---:|
| PaliGemma CPU | 13.25 GiB | 11.69 GiB |
| π0.5 CPU | 12.95 GiB | 13.45 GiB |
| PaliGemma CUDA | 2.71 GiB | 1.61 GiB |
| π0.5 CUDA | 4.37 GiB | 3.64 GiB |

These are host RSS snapshots after repeated requests, not a sum with VRAM.
Managed allocations overlap accounting domains. GPU VRAM in the earlier
matched runs was close, approximately 7 GiB for π0.5. The peak-loading result
therefore does not establish a large steady-state model-memory advantage.

There are concrete model memory improvements left:

- PaliGemma's embedding-bypass package keeps a separate approximately 1.96
  GiB FP32 host lookup table in addition to its vocabulary-head record.
  Hugging Face shares embedding/head storage. Weight sharing/layout handling
  or sparse mapped lookup is still a model/loader integration task.
- π0.5 retains approximately 1.60 GiB of host expert/flow weights after its
  CUDA expert has copied device weights into its cache. This supports the
  current host implementation, but it is not minimal for a CUDA-only policy.
- The model loader, mapped-file policy and inference-resident footprint should
  be measured separately before treating memory as an architectural win.

## Recommended focus

1. **Connect existing optimized paths correctly.** Make the model CUDA profile
   explicit and regression-qualify FP16 elementwise execution. Apply the
   model-local CPU expert BLAS mapping. These have passing controlled evidence.
2. **Fix shared GPU residency/dispatch around vision.** FP32 FC bias and
   residual operations, tensor copies, synchronization and attention batching
   have larger end-to-end opportunity than the expert attention arithmetic
   alone. The same SigLIP path serves both models.
3. **Improve model execution and memory ownership.** Keep π0.5 prefix/RoPE/KV
   on device while preserving FP32 cache precision; expose component backend
   selection explicitly; share/map PaliGemma embedding/head weights and avoid
   unnecessary retained host expert payload in CUDA-only use.
4. **Then tune individual GEMM/attention/SIMD kernels using the new profile.**
   CPU vector dispatch is already worth integrating/testing, but standalone
   operator gains must be checked at request level. Kernel tuning remains
   necessary; it is not the only remaining task.

## Artifacts and limits

All sources, command scripts, raw logs, CUPTI TSV traces, phase summaries,
strict numerical checks and reviewable `cpu_attention_blas.patch` /
`vector_glu.patch` are under
`.validation/paligemma-pi05-20261113/architecture-audit/`.
`findings.json` and `audit-summary.json` contain machine-readable results.
The accompanying repository JSON contains the selected findings.

CUPTI GPU timestamps are collected through the
[NVIDIA Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html).
Unified Memory byte-counter enabling returned `CUPTI_ERROR_NOT_READY`, even
after context creation, so no total managed-transfer-byte or page-fault counts
are asserted. CUPTI and phase markers can perturb wall time; untraced setting
tests, existing layer profiles and strict output comparisons corroborate the
conclusions.

Two prototype results are intentionally excluded: an initial AVX adapter with
incorrect argument order failed output validation and was corrected before
the passing experiments; an ENV-switch hybrid attempt was invalid because
`causallm_engine()` caches its first backend. The latter was stopped rather
than misreported as CPU-vision/CUDA-language performance. Per-component routing
needs a real model/backend interface, not environment mutation after startup.

No experimental patch was applied to production code, no checkpoint was
changed and no commit was created. These experiments establish actionable
counterexamples to optimal execution; they do not claim a globally optimal
implementation or portability/performance results on other platforms.
