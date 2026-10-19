# LFM2 MoE on the Hexagon HTP

LFM2 24B A2B runs its hybrid convolution/attention decoder on Hexagon v81
using the shared Gemma4 HTP backend. This guide covers the current weight
format, conversion, execution and validated performance in PR #4428 as of 2026-10-19.
For SDK installation, Android packaging and device staging, follow the
[Gemma4 user guide](gemma4-htp-user-guide.md); the
[code guide](gemma4-htp-code-guide.md) explains the shared backend.

The automatic SR15 preset reached **1157.72 prefill tok/s and 17.3877 decode
tok/s** as medians of three runs on the measured S26 Ultra with 2048 input
tokens and preload 128. The native CLI now selects this preset automatically
for `lfm2_moe` + `sr15_v2` + `htp_fused`. Explicit environment variables take
precedence. These are short-run results for one fixed prompt, not sustained
throughput or a per-run guarantee.

## Supported model pack

LFM2 HTP supports only `lfm2_weight_format=sr15_v2`. The pack uses
`lfm2_execution_profile=htp_fused`, batch one, FP32 graph
tensors and HTP engines for attention, projections, dense FFN, MoE and
the LM head. LFM2 uses sigmoid routing with expert bias, SwiGLU, embedding
scale 1 and `attention_kv_dtype=q8_row` for unnormalized value projections.
Gemma4's `q8` cache setting is a different model contract.

| Component | `sr15_v2` |
|---|---|
| Routed experts | `TL2CX_WH` |
| Attention, dense FFN, conv projections and LM head | `QS4CX_WH_AFFINE` |
| Embedding | `Q4_0` |
| Router | `QS4CX_WH_AFFINE` |
| Short-convolution weights | `DW3_INT4` on disk, decoded to FP32 at load |
| Norms and expert bias | FP32 |
| Online expert rotation | Both input and down rotations, with checkpoint basis and signs |
| Activation quantization | Dynamic |
| Public QuickDotAI descriptor | `lfm2-24ba2b-htp`, NPU/W4A32 |

The model factory and public QuickDotAI loader both require SR15. The
legacy unrotated LFM2 pack is no longer supported; convert the matching
SR15 checkpoint instead of changing an old pack's metadata. Place the
public API pack in `<model_base_path>/lfm2-24ba2b-htp-w4a32/`.

The fused profile requires 64-wide attention heads, three-tap bias-free
short convolutions, untied embeddings, normalized top-k probabilities,
routed scaling factor 1, and 32 or 64 experts. It rejects CPU/GPU engine
mixing, memory swap and the KV window ring. See
[setupParameters](../Applications/CausalLM/models/lfm2_moe/lfm2_htp_moe_causallm.cpp)
for the full geometry checks.

## Convert the weights

Run the converter from the repository root with NumPy and a host C
compiler installed. It reads local checkpoints, verifies emitted records,
and refuses to overwrite an existing output directory. SDK binaries and
production weights are supplied separately.

For the rotated SR15 checkpoint:

```bash
python3 tools/convert_lfm2_sr15.py /path/to/sr15-checkpoint \
  --encodings /path/to/encodings \
  --output /path/to/lfm2-sr15 \
  --tokenizer-source /path/to/tokenizer \
  --rotation both
```

The encoding directory contains `lfm2_24b_w4a8kv8_generation.encodings`
and `lfm2_24b_w4a8kv8_prefill.encodings`. The source supplies the model and
generation configuration, chat template and rotation tensors; the tokenizer
directory supplies `tokenizer.json` and `tokenizer_config.json`.

Keep the generated pack together:

- The model binary named by `nntr_config.json` (`LFM2-24B-A2B-Ternary-INT4.bin`).
- `conversion_report.json`, including tensor offsets, sizes, hashes and
  rotation records.
- `config.json`, `generation_config.json`, `nntr_config.json`, tokenizer
  files and the chat template when supplied.
- SR15's `activation_scales.bin`, retained as conversion output. The
  runtime does not consume it: `activation_quantization` must be `dynamic`.

Native TL2 preserves the ternary expert values. Affine WH recovers the
source INT4 grid and channel offsets; an exact BF16 round trip does not
mean the recovered FP32 grid is identical to the BF16-rounded values.
SR15's Q4_0 embedding is requantized, and its conversion error is reported
separately. The converter validates serialization, not full-model quality.

SR15 requires `moe_hadamard=both` and `moe_hadamard_scope=all`. Disabling
the down rotation changes the model and is not a valid speed comparison.
Its 2048-wide input and oriented H12 x H128 down transforms use the
checkpoint's basis, normalization and signs.

## Build and run

Use the SDK/toolchain settings in the
[Android setup](gemma4-htp-quickdotai.md#android-build), then build:

```bash
bash Applications/CausalLM/build_android.sh --htp --app
```

Stage the ARM executable and libraries with the DSP skeleton from the same
build. Rebuild both sides after IDL changes. On the Android device:

```sh
runtime_dir=/data/local/tmp/nntrainer/runtime
model_dir=/data/local/tmp/nntrainer/lfm2-sr15
export LD_LIBRARY_PATH="$runtime_dir" ADSP_LIBRARY_PATH="$runtime_dir"
"$runtime_dir/nntr_causallm" "$model_dir"
```

The converter's SR15 defaults are `init_seq_len=256`, `max_seq_len=2048`
and `num_to_generate=32`. They are a smoke preset, not the measured
2048-token preset below. Keep dtype, engine and rotation metadata from
the converter when changing the prompt or generation budget.

Each run begins an independent prompt with the loaded weights. Inputs
longer than `init_seq_len` are processed in chunks up to `max_seq_len`.
The native model's KV snapshot includes attention state and a `<path>.conv`
companion containing the convolution state. A missing companion or a
different geometry/position is rejected. This native model capability
does not imply snapshot support through the public QuickDotAI loader.
Weight reload requires a new model handle.

For QuickDotAI, build with `--htp` and load `lfm2-24ba2b-htp` using
`CAUSAL_LM_BACKEND_NPU` and `CAUSAL_LM_QUANTIZATION_W4A32`. Pass an absolute
`native_lib_dir` containing the matching v81 DSP library. The public API
accepts SR15 packs and rejects other LFM2 formats and KV snapshot options.
Each streaming call starts an independent prompt. See the
[C API guide](../Applications/CausalLM/api/README.md#lfm2-moe-on-native-htp)
for the handle lifecycle.

## Default SR15 CLI preset

`nntr_causallm` applies the preset after reading the model configuration and
before creating the model. It sets only environment variables that are
absent. An explicit zero is preserved, so the caller can disable individual
options. `NNTR_LFM2_DEFAULT_PRESET=0` disables automatic preset selection.
The CLI does not change the checkpoint, quantization, rotation, prompt,
context length, generation budget or configured expert-cache size.

The preset uses eight CPU threads, direct model-file I/O
(`NNTR_HTP_PAGE_CACHE=0`), no load prefetch and startup preload 128.
It enables the selected prefill whole-call, convolution, mapped-view,
parallel norm/add/staging and attention handoff paths; convolution stripes
have eight rows. Prefill pruning stays disabled. Decode uses stable TL2
slots, paired expert reads, demand handoff, gamma-aware prediction with
residual blend 0.5 and top-three prefetch, and six direct-read slots.

After the first decode step, idle prefill readers and unused heap pages are
released and `NNTR_MOE_CACHE_GROW=1` adds one expert slot per layer. With
`moe_cache_experts=16`, the 38-layer LFM2 cache starts at 608 experts and
grows to 646. The arena allocation grain is 4 MiB. This phase reuse is not
a fixed-cache comparison and does not establish a total-memory bound.
The exact defaults are in
[`lfm2_runtime_defaults.h`](../Applications/CausalLM/lfm2_runtime_defaults.h).

These process-level defaults belong to the single-model CLI. They are not
installed by the library or QuickDotAI API, and are not selected for
Gemma4 or other execution profiles. Diagnostic tracing and
profiling are not enabled by the preset.

For example, use the default combined preset, or retain the initial cache:

```sh
"$runtime_dir/nntr_causallm" "$model_dir"
NNTR_MOE_CACHE_GROW=0 "$runtime_dir/nntr_causallm" "$model_dir"
```

Remove environment overrides left by previous experiments before comparing
with the default run. In particular, old `NNTR_MOE_CACHE_GROW=0` or
`NNTR_MOE_PRELOAD_EXPERTS=384` exports still take precedence.

## Validated 2048-token workload

The 2026-10-19 comparison used SM-S948U / Hexagon v81, batch one,
`init_seq_len=2048`, `max_seq_len=4096`, C16, greedy generation and 127
measured decode steps: 128 output IDs including the token selected by
prefill. EOS ID 7 was suppressed only for this fixed-length benchmark.

To reproduce its prompt in an existing converted SR15 pack, update a copy
of its configuration on the host:

```bash
python3 - /path/to/lfm2-sr15/nntr_config.json <<'PYCONFIG'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
notes = (
    "A community garden provides vegetables and a place for neighbors to work "
    "together. Volunteers share tools, water plants, make compost, and keep "
    "a planting calendar. "
)
config.update(
    init_seq_len=2048,
    max_seq_len=4096,
    num_to_generate=127,
    moe_cache_experts=16,
    bad_word_ids=[7],
    skip_prefill=False,
    sample_input=(
        "<|im_start|>user\n" + notes * 65
        + "..\nUsing the notes above, write a detailed practical guide "
        "of at least 200 words for starting a community garden.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
)
config.pop("chat_input", None)
path.write_text(json.dumps(config, indent=2) + "\n")
PYCONFIG
```

The measured binary is 5,433,850,513 bytes, SHA256
`8a424db49615749dc28496bd007f53d9961e700c0c2cc022bc638056648228bc`.
The measured configuration SHA256 is
`e9e3b1678443255e028f1f3336be1c6aecc1311e379cff2838a9851e392a528b`.
All six paired runs reproduced the same 128 IDs, whose newline-separated
file SHA256 is
`6706ee8cd9a2117a77e25a9ad1f99fa91906e9e0a87e2567cf3235b134015882`.
Different checkpoints, tokenizers or prompts are different workloads.

| Runtime / preset | Runs | Prefill median (range), tok/s | Decode median (range), tok/s |
|---|---:|---:|---:|
| Prefill optimization only | 3 | 1147.98 (1146.70–1153.80) | 16.303 (16.2967–16.3176) |
| Combined prefill and decode | 3 | 1153.15 (1145.41–1157.72) | 17.402 (17.3996–17.4187) |

The runs alternated baseline/candidate and candidate/baseline. Launches
required five seconds without another inference process, current HAL AP
at most 34 °C and skin at most 36.5 °C. No layer or HTP profiler was active.
Combined decode improved 6.74% relative to the matched prefill-only
runtime. The app's prefill and generation timers define these rates;
they do not measure process startup or complete first-token latency.

The final CLI was then tested three times with no performance environment
overrides. Only the runtime library paths and output-ID capture were set;
the automatic preset was active in every run. Prefill measured 1159.03,
1151.21 and 1157.72 tok/s (median **1157.72**), and decode measured 17.4259,
17.2813 and 17.3877 tok/s (median **17.3877**). All three runs reproduced
the same 128 IDs above. These runs verify automatic selection separately
from the earlier paired optimization comparison.

The same final runtime also passed three paired Gemma4 native TL2/affine
regression runs with 1024 input tokens, C16/preload 128 and 64 decode steps.
All 65 output IDs matched the reference, and the LFM2 preset remained
inactive. Baseline and current medians were respectively 621.736/12.9110
and 632.099/12.9923 prefill/decode tok/s. See the
[Gemma4 guide](gemma4-htp-user-guide.md) for the workload and comparison.

The earlier 2026-10-18 SR15 result (903.00 prefill / 17.85 decode tok/s)
used a different 1024-token prompt and preload 384. Its preload-128
comparison was 798.13 / 17.74 tok/s. Those historical results must not be
mixed with the 2048-token comparison or interpreted as its baseline.

## Validation and limits

The 2026-10-19 runtime passed Android ARM64 application/backend and v81
DSP builds. The default-selection check covers model/profile isolation,
explicit overrides and opt-out. Additional host checks passed sanitized
prefetch cancellation and signed-Hadamard parity/canary cases, including
portable and SDK libnative paths. The optional native HVX GEMV/quantizer
checks were run separately with the SDK tools path set.
The host suite completed through the repaired router-handoff fixture and
remaining TL2/abs-max checks. The optional abs-max NEON emulation was not
run because the AArch64 cross compiler or QEMU was unavailable.

Earlier SR15 validation passed Android ARM64 and v81 DSP builds and 13 device
regression entries covering Hadamard transforms, convolution continuation,
KV operations, embedding row caching, RMSNorm, repeated prompts and
snapshot cross-loading. Host checks cover routing assignment order,
attention preparation/overlap, staging ownership, dynamic KV tiles and
startup registration failure/retry. Useful maintained entry points are:

```bash
bash test/htp/host/run_host_checks.sh
python3 test/htp/host/lfm2_conversion_check.py
```

Device parity here means parity with the recorded nntrainer runtime and
pack. It is not a perplexity result or proof of equivalence to an external
ONNX model. A diagnostic run with `NNTR_HTP_PROFILE=2` changed baseline
IDs and was excluded from both timing and correctness claims. Use ordinary
runs for reported throughput; these local validations do not establish
the status of the full cross-platform CI matrix.

## Evidence and code references

Raw validation artifacts are kept outside the commits in this PR. This
guide retains the measured workload, configuration, medians, memory
definition and validation limits. Per-run logs, output-ID files, memory
samples and binary manifests remain local measurement artifacts.
Earlier bring-up numbers and intermediate trials are not current results.

| Source | Responsibility |
|---|---|
| [LFM2 HTP model](../Applications/CausalLM/models/lfm2_moe/lfm2_htp_moe_causallm.cpp) | Format checks, hybrid graph, rotations, conv/KV snapshots |
| [Shared MoE layer](../Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp) | Routing, expert residency and startup reads; also used by Gemma4 |
| [Convolution layer](../Applications/CausalLM/layers/conv_block_layer.cpp) | Three-tap convolution, gates and continuation state |
| [SR15 converter](../tools/convert_lfm2_sr15.py) | Packed weights, rotation records and activation sidecar |
| [HTP compute operations](../nntrainer/tensor/htp_backend/htp_compute_ops.cpp) | ARM/DSP transport, expert reads and execution |
