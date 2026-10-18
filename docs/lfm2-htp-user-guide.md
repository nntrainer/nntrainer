# LFM2 MoE on the Hexagon HTP

LFM2 24B A2B runs its hybrid convolution/attention decoder on Hexagon v81
using the shared Gemma4 HTP backend. This guide covers the current weight
formats, conversion, execution and validated performance in PR #4428 as of 2026-10-18.
For SDK installation, Android packaging and device staging, follow the
[Gemma4 user guide](gemma4-htp-user-guide.md); the
[code guide](gemma4-htp-code-guide.md) explains the shared backend.

The SR15 CLI preset reaches a median 903.00 prefill tok/s and 17.85 decode
tok/s on the measured S26 Ultra with startup preload 384. With the default
preload 128, the corresponding medians are 798.13 and 17.74 tok/s. These are
different presets with different startup costs; neither result is a
per-run throughput guarantee.

## Supported model packs

Both packs use `lfm2_execution_profile=htp_fused`, batch one, FP32 graph
tensors and HTP engines for attention, projections, dense FFN, MoE and
the LM head. LFM2 uses sigmoid routing with expert bias, SwiGLU, embedding
scale 1 and `attention_kv_dtype=q8_row` for unnormalized value projections.
Gemma4's `q8` cache setting is a different model contract.

| Component | Unrotated `htp_fused_v1` | Rotated `sr15_v2` |
|---|---|---|
| Routed experts | `TL2CX_WH` | `TL2CX_WH` |
| Attention, dense FFN, conv projections and LM head | `QS4CX_WH_AFFINE` | `QS4CX_WH_AFFINE` |
| Embedding | FP32 | `Q4_0` |
| Router | FP32 | `QS4CX_WH_AFFINE` |
| Short-convolution weights | FP32 | `DW3_INT4` on disk, decoded to FP32 at load |
| Norms and expert bias | FP32 | FP32 |
| Online expert rotation | Off | Both input and down rotations, with checkpoint basis and signs |
| Activation quantization | Dynamic | Dynamic |
| Public QuickDotAI descriptor | `lfm2-24ba2b-htp`, NPU/W4A32 | Not accepted by the current public API guard; use the native CLI |

The model factory accepts both formats. The public QuickDotAI loader
currently requires `lfm2_weight_format=htp_fused_v1`; changing an SR15
pack's metadata to bypass that check would misdescribe its binary layout.
The public unrotated pack directory is
`<model_base_path>/lfm2-24ba2b-htp-w4a32/`.

The fused profile requires 64-wide attention heads, three-tap bias-free
short convolutions, untied embeddings, normalized top-k probabilities,
routed scaling factor 1, and 32 or 64 experts. It rejects CPU/GPU engine
mixing, memory swap and the KV window ring. See
[setupParameters](../Applications/CausalLM/models/lfm2_moe/lfm2_htp_moe_causallm.cpp)
for the full geometry checks.

## Convert the weights

Run the converters from the repository root with NumPy and a host C
compiler installed. They read local checkpoints, verify emitted records,
and refuse to overwrite an existing output directory. SDK binaries and
production weights are supplied separately.

For an unrotated BF16-dequantized ternary checkpoint:

```bash
python3 tools/convert_lfm2_htp.py /path/to/checkpoint \
  --output /path/to/lfm2-24ba2b-htp-w4a32 \
  --tokenizer-source /path/to/tokenizer
python3 test/htp/host/lfm2_conversion_check.py
```

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

- The model binary named by `nntr_config.json` (`LFM2-24B-A2B-Ternary-INT4.bin`
  for SR15, `nntr_lfm2_tl2_affine.bin` for the unrotated pack).
- `conversion_report.json`, including tensor offsets, sizes, hashes and,
  for SR15, rotation records.
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
1024-token preset below. Keep dtype, engine and rotation metadata from
the converter when changing the prompt or generation budget.

Each run begins an independent prompt with the loaded weights. Inputs
longer than `init_seq_len` are processed in chunks up to `max_seq_len`.
The native model's KV snapshot includes attention state and a `<path>.conv`
companion containing the convolution state. A missing companion or a
different geometry/position is rejected. This native model capability
does not imply snapshot support through the public QuickDotAI loader.
Weight reload requires a new model handle.

## Validated performance preset

The latest matched measurements used SM-S948U / Hexagon v81 on 2026-10-18,
eight threads, 16 cached experts per MoE layer, no cache growth, direct
model-file I/O and no layer/HTP profiling. SR15 had 1024 input tokens,
`max_seq_len=2048`, greedy generation and 127 decode steps: 128 output IDs
including the first token selected by prefill. EOS ID 7 was suppressed
only for this fixed-length benchmark.

For the same workload, generate the benchmark `nntr_config.json` on the
host and place it in the matching converted pack. This preserves the
original prompt and settings without depending on a removed measurement
commit. The EOS suppression is only for this fixed-length benchmark.

```bash
python3 - /path/to/lfm2-sr15/nntr_config.json <<'PYCONFIG'
import json
import sys
from pathlib import Path

config = json.loads(r'''
{
  "model_type": "CausalLM",
  "model_tensor_type": "FP32-FP32",
  "model_file_name": "LFM2-24B-A2B-Ternary-INT4.bin",
  "lfm2_weight_format": "sr15_v2",
  "lfm2_execution_profile": "htp_fused",
  "embedding_dtype": "Q4_0",
  "fc_layer_dtype": "QS4CX_WH_AFFINE",
  "lmhead_dtype": "QS4CX_WH_AFFINE",
  "router_dtype": "QS4CX_WH_AFFINE",
  "moe_layer_dtype": "TL2CX_WH",
  "short_conv_dtype": "DW3_INT4",
  "moe_engine": "htp",
  "attn_proj_engine": "htp",
  "dense_ffn_engine": "htp",
  "attention_engine": "htp",
  "attention_kv_dtype": "q8_row",
  "lmhead_engine": "htp",
  "moe_hadamard": "both",
  "moe_hadamard_scope": "all",
  "batch_size": 1,
  "init_seq_len": 1024,
  "max_seq_len": 2048,
  "num_to_generate": 127,
  "tokenizer_file": "tokenizer.json",
  "bad_word_ids": [
    7
  ],
  "moe_cache_experts": 16,
  "sample_input": "",
  "activation_quantization": "dynamic",
  "skip_prefill": false
}
''')
notes = (
    "A community garden provides vegetables and a place for neighbors to work "
    "together. Volunteers share tools, water plants, make compost, and keep "
    "a planting calendar. "
)
config["sample_input"] = (
    "<|im_start|>user\n" + notes * 31
    + "A community garden provides vegetables and a place for neighbors to work "
    "together. Volunteers share tools, water plants, make compost, and keep "
    "a planting Note. \nUsing the notes above, write a detailed practical guide "
    "of at least 200 words for starting a community garden.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + "\n")
PYCONFIG
```

The measured binary is 5,433,850,513 bytes, SHA256
`8a424db49615749dc28496bd007f53d9961e700c0c2cc022bc638056648228bc`.
The benchmark configuration SHA256 is
`fcbea14cb03aaee7bbceb6fa43ca8f3bd1f25e7995ec01c073bcddf9ac172964`.
A different checkpoint, tokenizer or prompt is a different workload.

On the device, use the runtime/model paths set above:

```sh
unset NNTR_HTP_PROFILE NNTR_LAYER_PROFILE NNTR_M0_PROFILE
export NNTR_NUM_THREADS=8 NNTR_HTP_PAGE_CACHE=0 NNTR_LOAD_PREFETCH=0
export NNTR_MOE_CACHE_GROW=0 NNTR_MOE_PRELOAD_EXPERTS=384
export NNTR_HTP_TL2_STABLE_SLOTS=1 NNTR_LFM2_PREFILL_EARLY=1
export NNTR_LFM2_DECODE_PAIR_READ=1 NNTR_LFM2_DECODE_DEMAND_HANDOFF=1
export NNTR_HTP_PREDICT=2 NNTR_MOE_PREFETCH_RESIDUAL=0.5 NNTR_INIT_TRACE=1
"$runtime_dir/nntr_causallm" "$model_dir"
```

Use `NNTR_MOE_PRELOAD_EXPERTS=128` for the default-preload comparison.
Preload 384 changes startup residency, not expert capacity: C16 remains
608 resident slots across 38 MoE layers. It moves work ahead of prefill,
so evaluate startup and first-token latency as well as prefill throughput.
These options are an explicit benchmark preset, not a change to defaults.

The final benchmark compared the runtime after the six SR15 optimization
commits (ending with `[CausalLM] Batch native LFM startup expert loads`)
against the earlier runtime ending with `[HTP] Accelerate LFM2 INT4 router
and 1536-wide Hadamard rotation`. The table reports the final candidate
medians from that series. The history cleanup removed measurement-only
commits and changed descendant commit IDs without changing runtime sources.
Later documentation changes are not separate device measurements.

| Model and preset | Input tokens | Decode steps | Runs | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| SR15, preload 384 | 1024 | 127 | 5 | 903.00 | 17.85 |
| SR15, preload 128 | 1024 | 127 | 3 | 798.13 | 17.74 |
| Unrotated LFM2, preload 128 | 1024 | 64 | 3 | 841.41 | 17.08 |
| Unrotated LFM2, preload 128 | 2048 | 64 | 3 | 965.58 | 16.10 |
| Unrotated Gemma4, preload 128 | 1024 | 64 | 3 | 621.36 | 12.32 |

All 34 normal baseline/candidate runs preserved their reference output
IDs. The unrotated LFM2 1024-token decode median was 1.31% lower than its
paired baseline; the series does not establish zero performance regression.
The unrotated LFM2 runs additionally enabled `NNTR_LFM2_ROUTER_INPUT_HANDOFF=1`
and `NNTR_LFM2_ROUTER_ARM=1`; Gemma4 used the common environment without
the LFM2 optimization switches. The cohorts have different decode lengths
and must not be treated as a controlled cross-model speed comparison.

| SR15 preset | Startup to load-ready ms | Run TTFT ms | Estimated main-to-first-token ms | Sampled peak MiB |
|---|---:|---:|---:|---:|
| Preload 128 | 728.60 | 1287.97 | 2018.47 | 2110.85 |
| Preload 384 | 869.30 | 1138.12 | 2005.12 | 2110.28 |

Startup includes configuration, initialization, loading and repacking.
Run TTFT includes run setup and prefill. Estimated main-to-first-token is
their per-run sum, excluding process-loader startup and the small gap
between scopes; table entries are medians, so they need not add exactly.
Memory is sampled every 20 ms over process lifetime as RssAnon plus
deduplicated size-bearing FD allocations. It excludes unreported DSP/driver
memory. At preload 384 the peak increased by 47.86 MiB versus its matched
baseline despite unchanged expert capacity. Prefill ranged from 824.48 to
914.29 tok/s across the five candidate runs.

## Validation and limits

The final runtime passed Android ARM64 and v81 DSP builds and 13 device
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

The measurement archives and their archive-only commits have been removed
from this PR. This guide retains the measured workload, configuration,
medians, memory definition and validation limits; the raw per-run logs,
output-ID files, memory samples and binary manifests are no longer included.
Earlier bring-up numbers and intermediate trials are not current results.

| Source | Responsibility |
|---|---|
| [LFM2 HTP model](../Applications/CausalLM/models/lfm2_moe/lfm2_htp_moe_causallm.cpp) | Format checks, hybrid graph, rotations, conv/KV snapshots |
| [Shared MoE layer](../Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp) | Routing, expert residency and startup reads; also used by Gemma4 |
| [Convolution layer](../Applications/CausalLM/layers/conv_block_layer.cpp) | Three-tap convolution, gates and continuation state |
| [SR15 converter](../tools/convert_lfm2_sr15.py) | Packed weights, rotation records and activation sidecar |
| [HTP compute operations](../nntrainer/tensor/htp_backend/htp_compute_ops.cpp) | ARM/DSP transport, expert reads and execution |
