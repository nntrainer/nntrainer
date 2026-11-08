# CPU performance measurement

The performance runner is separate from correctness validation. It does not
load an HF reference, dump or scan logits, enable KV tracing, inspect saved
float dtypes, or write files inside the timed request. Existing baseline,
post and final correctness snapshots remain unchanged.

The Q4 default packed path stayed within the observed 5% budget on Linux and
in Android's heated reverse-order comparison. Android's first 4K comparison
exceeded the budget and is preserved; its result was not reproduced after
reversing the order. Packed remains the default. The tiled experiment has
observational latency/memory results, without a general speedup claim.

`qwen_cpu_benchmark.cpp` loads one model, prepares a deterministic synthetic
prompt, then executes independent requests with the cache position reset to
zero. The logits hook writes one scalar to force vocabulary IDs 32–35 in a
fixed sequence. These ordinary Qwen IDs are checked against configured EOS.
The model retains its EOS rules, but each trial must produce exactly one
prefill-selected token plus the requested number of decode steps. A count,
prompt truncation or final cache-position mismatch fails the run. This is a
fixed-work throughput experiment, not a model-quality assessment.
All profiles pin `skip_prefill=false` for that common phase boundary. These
numbers are not a claim of peak production throughput with every optional
LM-head/prefill optimization enabled. Decode count zero is rejected because
the model API interprets it as unlimited generation, not prefill-only work.

## Timed phases and limitations

- Factory creation, graph initialization, and weight loading are separate
  timers. `load_weight()` includes its built-in eager packing, so the reported
  field is `weight_load_and_pack_ms` rather than a claim of disk-only time.
- Tokenizer initialization, prompt construction and initial token encoding
  are prepared before warmup and recorded separately.
- Initial cache allocation and any lazy kernel work occur inside the first
  warmup request. Public CausalLM APIs do not expose a separate cache-prepare
  call. No allocation-only duration is inferred from a residual subtraction.
- Each request records wall time and high-resolution time to its first
  logits callback. That latter row includes `run()` input preparation, repeated
  encoding and prefill; it is not pure attention latency.
- The existing model API reports compute-prefill milliseconds and first/steady
  decode timing. Compute-prefill is currently rounded to integer milliseconds.
  The new callback also records every decode-ready interval, including normal
  token selection and output-tokenizer work between callbacks.
- The callback performs a timestamp and one scalar write, with no logits
  scan/dump or finite check. Its small overhead remains part of API latency.
  Kernel flags and the model's normal forward path are unchanged.
- Warmup is excluded from summaries. Default policy is one warmup and five
  measured repetitions in the same loaded process. Median, range, mean and
  sample deviation are retained; repetitions=1 has no sample deviation.

This measures a warm loaded model and reused KV allocation, rather than cold
storage, model load plus generation, or standalone attention. Different model
loads and independent launches still have separately recorded preparation
costs. Correctness must be checked independently for the same runtime/profile.

## Build and configuration

The generic builder accepts an alternate runner source without changing the
correctness probe:

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py --build build-kv-cpu \
  --source tools/kv_cache/qwen_cpu_benchmark.cpp \
  --output build-kv-perf/bin/linux/qwen_cpu_benchmark

python3 tools/kv_cache/build_qwen_cpu_probe.py --build builddir_app/cpu \
  --source tools/kv_cache/qwen_cpu_benchmark.cpp \
  --output build-kv-perf/bin/android/qwen_cpu_benchmark
```

The builder records compile/link commands and runner source hashes. For a
preserved older CausalLM ABI, build its runner using the matching headers and
an explicit `--header-tree`. A runtime comparison uses each runner with
its matching shared libraries; binaries are not rebuilt during measurement.

The candidate adds a virtual `Transformer::hasKVCacheProducer()` hook. A runner
compiled against those new headers must not be paired with an older runtime's
vtable. The 69b9c96a5 project header tree is preserved separately for baseline
compilation (also matching the older correctness baseline's base-class layout):

```bash
mkdir -p /tmp/nntrainer-kv-benchmark-69b9c96a5
git archive 69b9c96a5 Applications/CausalLM nntrainer api | \
  tar -x -C /tmp/nntrainer-kv-benchmark-69b9c96a5
python3 tools/kv_cache/build_qwen_cpu_probe.py --build builddir_app/cpu \
  --source tools/kv_cache/qwen_cpu_benchmark.cpp \
  --header-tree /tmp/nntrainer-kv-benchmark-69b9c96a5 \
  --header-tree-revision 69b9c96a5 \
  --output build-kv-perf/bin/android-baseline/qwen_cpu_benchmark
```

Project include paths map to that tree. Vendored dependencies and generated
build headers retain their existing dependency locations; the exact generated
`json.hpp` is supplied separately without adding current model headers back
into the baseline search path. Header hashes and the source revision label
accompany the baseline runner metadata. The candidate runner is separately
rebuilt after the candidate runtime is ready.

Qwen3's verified `max_position_embeddings` is 40,960. Contexts 1,024 and 4,096
fit with decode room. A 128-row activation plane and explicit 128-token
prefill chunk avoid growing the activation plane with context length. The
default capacity rounds `(context + decode_tokens + 64)` up to a 128-row grid:
1,152 and 4,224 slots for 32 decode steps. Qwen's half-bit KV backing uses
114,688 bytes per slot, approximately 126 MiB and 462 MiB respectively. FP32
weights occupy 2.38 GB; the selected device has enough RAM for these cases.
The runner rejects capacities that exceed the model position bound and
rejects any actual prompt truncation.

**Every 1K/4K result below uses plane 128 and prefill chunk 128.** A 4,096-token
context is fed through 32 forwards of up to 128 tokens, rather than one
unchunked 4,096-row prefill. These timings and memory observations do not
evaluate a single 4K prefill, a BQ64 sweep, or reviewer traffic estimates for
that different workload. Hardware copy traffic was not measured here.

The four current profiles set both flags explicitly:

| Profile | `NNTR_CPU_ATTENTION_FP32` | `NNTR_CPU_KV_TILED` |
| --- | ---: | ---: |
| `legacy-packed` | 0 | 0 |
| `legacy-tiled` | 0 | 1 |
| `fp32-packed` | 1 | 0 |
| `fp32-tiled` | 1 | 1 |

`legacy` names the older attention arithmetic, not the cache-manager opt-out
`NNTR_CPU_KV_LEGACY=1`. Inherited NNTR variables are removed, that opt-out is
not enabled, and the candidate uses its default checked cache execution path.

An older baseline binary may ignore flags it does not implement. Its label,
runtime hashes and matching runner ABI distinguish it from the current build.
On Linux, use the same activation/KV profile when comparing old and current
runtimes; the Android FP32-intermediate opt-in has Android-specific meaning.

## Commands and artifacts

```bash
python3 tools/kv_cache/benchmark_qwen3_cpu.py \
  --target linux --probe build-kv-perf/bin/linux/qwen_cpu_benchmark \
  --runtime-dir <matching-linux-libraries> \
  --model build-kv-cpu/models/qwen3-0.6b-fp32 \
  --model build-kv-cpu/models/qwen3-0.6b-q40 \
  --profile legacy-packed --label current \
  --context 1024 --context 4096 --decode-tokens 32 \
  --warmup 1 --repetitions 5 --threads 4 \
  --output build-kv-perf/linux-current

python3 tools/kv_cache/benchmark_qwen3_cpu.py \
  --target android --serial bacd420f \
  --probe build-kv-perf/bin/android/qwen_cpu_benchmark \
  --runtime-dir <matching-android-libraries> \
  --model build-kv-cpu/models/qwen3-0.6b-fp32 \
  --model build-kv-android/models/qwen3-0.6b-q40 \
  --reuse-device-models --profile fp32-tiled --label current \
  --context 1024 --context 4096 --decode-tokens 32 \
  --warmup 1 --repetitions 5 --threads 4 \
  --output build-kv-perf/android-current-fp32-tiled
```

Android requires an explicit `--serial`; no device is selected implicitly.
All mutations go under the selected device's dedicated `benchmarks/` directory.
`--reuse-device-models` reads and hashes the existing dedicated `models/`
assets without replacing them. ARM-packed Q4 is required on Android; the x86
Q4 model is used on Linux. Host weight hashes and device weight hashes must
match before inference.

Each output directory must be fresh. Source, runner, library, weight, request
and stage-configuration hashes accompany every result. Device thermal service,
battery temperature, accessible thermal zones and CPU frequency/governor rows
are recorded before and after each model/context process, outside timing.
Linux records accessible sysfs temperature/frequency data similarly. Missing
or permission-restricted sensors remain explicit in the logs.

Threads are fixed to four with OpenBLAS threads one. Frequency, scheduling,
temperature, charging and thermal throttling are observations, not controlled
variables. `--cooldown-seconds` can insert up to 60 seconds between processes;
its actual setting is recorded. Profile order and run time must be considered
before attributing timing differences to the cache implementation. Screening
can use fewer repetitions before expensive 1K/4K runs, with the actual sample
count retained. No speedup conclusion follows from a single trial.

`--baseline-report <completed-output>` verifies identical weight hashes,
prompt IDs, forced decode IDs, capacity, target/device and thread/plane/chunk
settings before reporting relative median changes. The default regression
budget is 5% for request wall time, compute-prefill and steady decode latency.
`regression_gate_passed` is separate from successful benchmark execution. The
gate applies only to the same arithmetic family (`legacy` or `fp32`) and
warmup/repetition policy. Packed/tiled changes are recorded while remaining
eligible for a same-arithmetic comparison. Changes of arithmetic family remain
explicitly excluded from the gate. Thermal/order uncertainty
still requires interpretation even when the observed medians fit the budget.

## Results (2026-10-29)

All rows use Qwen3-0.6B Q4, batch 1, threads 4, OpenBLAS threads 1, 32 fixed
decode steps, one warmup and five measured repetitions. Values are medians;
all raw samples and before/after sensor observations are retained.

| Backend / runtime / packing | Context | Prefill API ms | Request wall ms | Steady decode ms/token |
| --- | ---: | ---: | ---: | ---: |
| Linux post-P2 baseline | 1,024 | 3,090 | 3,592.05 | 15.52 |
| Linux candidate, default | 1,024 | 3,108 | 3,605.15 | 15.56 |
| Linux post-P2 baseline | 4,096 | 28,460 | 29,687.46 | 37.82 |
| Linux candidate, default | 4,096 | 28,113 | 29,304.01 | 36.88 |
| Android earlier baseline, packed | 1,024 | 3,138 | 3,720.33 | 18.13 |
| Android candidate, packed | 1,024 | 3,172 | 3,763.73 | 18.46 |
| Android earlier baseline, packed | 4,096 | 29,932 | 31,169.62 | 38.96 |
| Android candidate, packed | 4,096 | 32,985 | 34,300.25 | 40.83 |
| Android reverse-order baseline, packed | 4,096 | 33,467 | 34,816.93 | 41.70 |
| Android candidate, tiled opt-in | 4,096 | 27,863 | 29,131.14 | 39.23 |

Linux baseline uses the untouched 22-library `qwen3-post/post-libs` snapshot.
That runtime already includes the earlier P2 x86 prefix-copy optimization.
This comparison measures the new checked-view/readiness enforcement, rather
than attributing the original prefix optimization's benefit to this change.
Linux request/prefill/decode changes are +0.36/+0.58/+0.26% at 1K and
−1.29/−1.22/−2.48% at 4K: none exceeds the 5% observed budget. Host compilation
and correctness inference were excluded from its timing windows.

Android default packed changes are +1.17/+1.08/+1.82% at 1K. The initial 4K
comparison **fails** the budget: +10.04/+10.20/+4.78%. Its earlier baseline
spans 28.17–34.36 seconds, while the candidate spans 34.20–34.47 seconds.
Re-running the untouched baseline after the candidate gives 34.82 seconds;
the candidate compared with that heated reverse baseline is
−1.48/−1.44/−2.09%. The initial failure remains in its original report.
An over-budget regression was not reproduced in the heated reverse order;
this does not prove a general improvement or eliminate all frequency/order
uncertainty. All measurements were warm loaded requests; earlier/cooler here
describes the device thermal order rather than a cold model load.

The tiled 4K experiment is the **same legacy-half arithmetic and Q4 weights**
as the candidate packed row. Its observed medians are lower by
15.07/15.53/3.91%, but its battery temperature starts at 26.9°C versus 30.4°C
for the packed 4K row, and its request range is 26.22–29.31 seconds. Thermal
status is 0 at these recorded endpoints; that does not fix clocks or remove
thermal/order effects. These are observations, not a tiled speedup claim or
a reason to change the default.

Maximum measured `peak_memory_kb` is 980,340 for candidate packed and 931,472
for tiled (about 47.7 MiB lower in this experiment). This existing API field
tracks RssAnon plus accelerator accounting, not total process RSS or hardware
copy traffic. Allocator history and sampling contribute to the observation;
do not attribute the entire delta to a single K/V buffer. Separately,
correctness traces confirm bounded tile scratch and zero full-prefix
materialization for the opt-in path. Tile packing remains a real copy.

There are ten model/context benchmark processes: 50 measured requests plus
ten warmups. FP32-weight performance, FP32-attention performance and Android
tiled 1K performance were not measured in this run. Their CLI profiles are
available for follow-up measurements.

## Separate correctness and artifact integrity

The candidate passes 32 natural-output cases without HF recomputation:

- Linux: eight FP32/Q4 arithmetic, Korean, long-context and session/snapshot
  cases, using the existing post-P2 requests and full-logit references.
- Android: eight cases each for default legacy-half packed, FP32-attention
  packed and FP32-attention tiled. Every token and every logit byte, including
  continuations and fresh-model snapshot restoration, matches its earlier
  same-arithmetic reference. The tiled profile also passes all eight strict
  allocation/copy trace audits.

Packed functional validation explicitly permits its history materialization;
it is not reported as a zero-copy consumer. The current validator records
`--cpu-kv-tiled` explicitly. Reused references require only the Rust tokenizer
decoder, without importing a Torch/HF model. Performance and correctness use
different executables and output directories.

An initial Linux staging error mixed an old validation `libcausallm.so` with
the new core, causing eight initialization failures. Those files/reports are
preserved in `runtime/linux-candidate` and `correctness/linux-candidate`.
The corrected freeze selects unique Meson shared-library targets and records
their origins; `runtime/linux-candidate-final` passes all eight cases without
a model conversion or encoding-design change. This was an integration ABI
pairing error, not evidence of a bad checkpoint.

Artifacts live under ignored `build-kv-perf/`:

| Artifact | Contents |
| --- | --- |
| `android-baseline-q4-long/` | Earlier Android packed 1K/4K |
| `android-candidate-halfpacked-q4-long/` | Candidate packed, including initial failed 4K gate |
| `android-baseline-q4-reverse/` | Heated reverse-order 4K baseline |
| `android-candidate-halftiled-q4-4k/` | Same-arithmetic tiled experiment |
| `linux-baseline-q4-long/`, `linux-candidate-q4-long/` | Linux post-P2 baseline/candidate |
| `correctness/linux-final/` | Eight corrected-freeze Linux comparisons |
| `correctness/android-candidate-packed/` | Eight default legacy-half comparisons |
| `correctness/android-candidate-fp32-packed/` | Eight FP32 packed comparisons |
| `correctness/android-candidate-tiled/` | Eight FP32 tiled comparisons and strict audit |
| `performance-summary.json` | Consolidated measurements, comparisons and source versions |
| `performance-checkpoint.json` | Completed handoff state and exact locations |

Per-run reports preserve the harness hash used during measurement. The final
source hash is recorded separately in the summary; subsequent explicit UTF-8
and `check=False` lint clarifications do not rewrite measured provenance.
Models, libraries and raw reports are not committed. Existing correctness
snapshots remain unchanged. Full GitHub platform builds and CodeQL were not
run as part of this local measurement task.
