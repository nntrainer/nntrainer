# Gemma4 HTP native TL2 and rotated weights

Gemma4 HTP supports native ternary experts, affine INT4 projections and
checkpoint-matched online Hadamard rotation. This guide describes the
implemented formats and their validation limits in PR #4428 as of
2026-10-18. The dated measurements below describe their own workloads;
documentation updates are not new device measurements.
Use the [user guide](gemma4-htp-user-guide.md) for the shared build and
staging procedure and the [QuickDotAI guide](gemma4-htp-quickdotai.md)
for the API lifecycle. The QS2/Q4 pack described there and the native TL2
pack described here have different weight layouts.

## Model contract

Select `gemma4_execution_profile=htp_fused` and
`gemma4_weight_format=htp_fused_v1`. A native TL2 pack has:

| Component | Format |
|---|---|
| Routed experts | `TL2CX_WH` |
| Attention projections, dense FFN and untied LM head | `QS4CX_WH_AFFINE` |
| Embedding | `Q4_0` |
| Router, norms and block scalars | FP32 |
| Attention KV cache | `q8` |

All WH expert layers run on HTP. The fused model is batch one; dtype
metadata must match the serialized records. For mixed expert packs,
`moe_layer_dtype_per_layer` overrides `moe_layer_dtype` by zero-based layer
index, for example:

```json
{
  "moe_layer_dtype": "TL2CX_WH",
  "moe_layer_dtype_per_layer": {
    "28": "QS4CX_WH_AFFINE",
    "29": "QS4CX_WH_AFFINE"
  }
}
```

Use those overrides only when layers 28 and 29 actually have affine WH
records. A four-bit container can hold ternary values; its runtime slot
size still follows the four-bit format. Mixed slot classes and read-ahead
can therefore increase memory even with unchanged `moe_cache_experts`.
The model configuration must include its global attention geometry
(`global_head_dim` and `num_global_key_value_heads`) as well as local
attention settings.

## Weight layouts and execution

A native TL2 record contains
`ceil(N/128) * ceil(K/24) * 640` packed bytes, followed by N FP32 scales
and N FP32 signed-weight column sums. Its implicit levels are -1, 0 and 1;
there is no palette or scale division by seven. The file stores one
expert payload. Prefill expands panels to WH int4 in DSP scratch for HMX;
decode uses the native TL2 LUT/GEMV path. File payload offsets and arena
capacity are distinct layouts and must not be interchanged.

Native TL2 expert slots now retain the packed payload and metadata without
the unused legacy QS2 padding. For the measured Gemma4 expert shape, one
gate/up-plus-down slot shrinks from 1,523,712 to 1,290,240 bytes. Registration
and rebinding use the compact TL2 RPC contract; rebuild the ARM library,
generated IDL stub and DSP skeleton together. The on-disk format is unchanged.
Legacy QS2 and prepared-convertible TL2 keep their padded layouts. Compact
storage must never be retagged as QS2.

An affine WH record contains `K*N/2` WH int4 bytes, N FP32 scales,
N FP32 plain signed-code column sums and N INT32 offsets. Its values are
`w[k,n] = scale[n] * (q[k,n] + offset[n])`. The kernel corrects the same
quantized activation's accumulator with
`offset[n] * sum_k(u8[k] - zero_point)` before dequantization. It does not
discard the offset or replace the source grid with symmetric min/max
quantization. Exact BF16 round-trip verification and FP32-grid error are
separate converter metrics.

Affine dense FFN decode supports the existing M=1..4 GEMV path with
gate/up/down row-sum correction. The earlier first-decode HMX failure is
fixed; the intermediate bring-up result is not a current limitation.
For the affine 262144-vocabulary head, prefill evaluates only the final
input row on HTP, in 64 slices of 4096 outputs; it does not evaluate a
full vocabulary projection for every prompt row.

## Conversion

The [conversion wrapper](../Applications/CausalLM/res/gemma4/convert_gemma4_tl2.py)
takes a verified QS2-expert/Q4_0 Gemma4 pack with a complete
`conversion_report.json`, plus the corresponding BF16-dequantized HF
checkpoint for affine grid recovery. It is not a general converter for
arbitrary mixed or rotated checkpoints.

Build its native helper from the repository root:

```bash
mkdir -p /tmp/nntr-tl2-tools
cc -O3 -c nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c \
  -I nntrainer/tensor/htp_backend/hvx \
  -o /tmp/nntr-tl2-tools/hvx_gemv_u8tl2.o
c++ -O3 -std=c++17 -pthread tools/convert_tl2_model.cpp \
  /tmp/nntr-tl2-tools/hvx_gemv_u8tl2.o \
  -I nntrainer/tensor -I nntrainer/tensor/htp_backend \
  -I nntrainer/tensor/htp_backend/hvx \
  -o /tmp/nntr-tl2-tools/convert_tl2_model
python3 Applications/CausalLM/res/gemma4/convert_gemma4_tl2.py \
  /path/to/verified-qs2-q40-pack \
  --output /path/to/gemma4-tl2-affine \
  --converter /tmp/nntr-tl2-tools/convert_tl2_model \
  --hf-source /path/to/bf16-checkpoint
```

Use Python 3.11+ for this wrapper (`hashlib.file_digest`) and NumPy; the
affine stage reads the BF16 rows itself (no PyTorch, no safetensors, no
whole-file mmap). Existing output directories are refused. Keep the generated binary, report, model/generation configuration,
tokenizer and chat template together. Changing JSON dtype names does not
convert the binary. Rebuild and stage the matching ARM libraries and v81
DSP skeleton using the [Android build instructions](gemma4-htp-quickdotai.md#android-build).

### The rotated checkpoint in one run

[weight_converter.py](../Applications/CausalLM/res/gemma4-26b-a4b/weight_converter.py)
takes the pre-rotated, fake-quantized BF16 HF checkpoint directly (experts
exact ternary per output channel, FC/head on an int4 grid, `rotation_meta.json`)
and chains the stages above: it writes the QS2 pack itself (expert codes with
the BF16 scale and column sum, embedding Q4_0, norms and router FP32 with
`router.scale` folded with `1/sqrt(hidden)` and the router gate transposed),
builds `convert_tl2_model`, runs the wrapper, then adds `rotation_signs.json`
(regenerated from the seed: `RandomState(seed).randint(0, 2)`, hidden then
moe_intermediate, `s = 1 - 2 * bit`), the rotation keys and config.json's
`global_head_dim`/`num_global_key_value_heads`. The experts of
`--affine-layers` (default `28,29`, the recipe's 4-bit expert layers) are
written as `QS4CX_WH_AFFINE` with the int4 grid recovered like FC/head and
listed in `moe_layer_dtype_per_layer`; every other MoE layer becomes native
TL2CX_WH. A layer whose values are not ternary is written affine whatever
the option.

```bash
python3 Applications/CausalLM/res/gemma4-26b-a4b/weight_converter.py \
  --source /path/to/gemma4-26b-a4b-crispy --output /path/to/gemma4-26b-a4b-rotated-2bit-4bit
```

2026-10-19: with the defaults the output is byte-identical to the hand-made
`nntr_gemma4_26b_a4b_2bit_4bit.bin` (SHA256 `ce3b6e6e...`), so the
2026-10-18 demo results apply unchanged. With `--affine-layers ''` (all 30
layers TL2; the 28/29 values of this checkpoint are exactly ternary, so the
weights are the same) the S26 Ultra p2048/g128 run used 8 arena chunks
(1792 MiB) instead of 12 (2624 MiB), peak 3669 MB instead of 4723 MB, with
prefill 613 vs 577 TPS and decode 15.5 vs 11.8 TPS back to back; the sub_agent
demo gave the same token ids for one-shot think, one-shot no-think and
KV-split no-think, and diverged after 34 tokens for KV-split think.

## Rotated checkpoints

The runtime can rotate the expert input and the post-activation input to
the down projection. The mode, scope and signs must match the checkpoint:

```json
{
  "moe_hadamard": "both",
  "moe_hadamard_scope": "all",
  "moe_hadamard_signs": "rotation_signs.json"
}
```

`rotation_signs.json` is resolved relative to the weight directory. Its
format is `nntr_hadamard_signs_v1`, with an `in` array of 2816 signs and a
`down` array of 704 signs for Gemma4 26B A4B, each value -1 or +1. The
supported convention applies the normalized H44 x Sylvester transform
then output-coordinate signs: `(x H) * signs`. Orientation, factor order
and signs are part of the weight contract. A seed alone is not a substitute
for the matching vectors unless the generator convention is known; the
converter above regenerates them and they match the vectors recovered from
the first rotated pack. [The sign recovery tool](../tools/htp/recover_hadamard_signs.py)
can compare reference and rotated packs for diagnosis.

Scope `all` covers routed experts of both two- and four-bit formats;
dense FFNs remain unrotated. The default scope `2bit` does not rotate
four-bit expert containers. Environment variables `NNTR_MOE_HADAMARD`
and `NNTR_MOE_HADAMARD_SCOPE` take precedence over the configuration,
so clear stale overrides when switching packs. Rotation stays off for
an unrotated checkpoint. A rotation-on smoke with unrotated weights does
not validate model output quality.

The strict 2026-10-20 checkpoint uses `moe_hadamard: "both_paley"`: its
Paley-I basis differs from the earlier `both` basis. Use the checkpoint's
rotation signs and its asymmetric U4 embedding sidecar, including the
manifest's BF16 rounding flag.

Paley input signs are applied with HVX before the existing FP32 transform;
the 704-wide down input then uses the same two-row kernel as `both`.
Applying the signs in a scalar loop and leaving mode `4` outside that batch
dispatch made the down rotation expensive. The vector sign pass and batch
dispatch preserve the checkpoint basis and the output-coordinate signs.
Selecting `both` or `both_hmx` instead would change the checkpoint contract.

## Static linear activation quantization

For the strict checkpoint, add its supplied input encodings to
`nntr_config.json`:

```json
{
  "static_activation_file": "static_linear.json"
}
```

The path is relative to the model directory, or absolute.
`NNTR_HTP_STATIC_LINEAR` overrides it for existing experiment scripts.
Without either setting, linear activation quantization remains dynamic.
Both supplied formats are accepted:

- `static_linear.json`: physical weight names mapped to `[scale, zero_point]`.
- `encoding_provenance.json`: its `sites` object of logical linear inputs,
  using `scale` and `-offset` as the unsigned eight-bit grid. The expert
  sites expand to every expert in the layer.

The 26B A4B strict sidecars describe 296 logical sites, mapped to 7,916
weight inputs. Q/K/V, output projections, dense gate/up/down, routed
experts, the FP32 router input and the output head use their supplied
grids in both prefill and decode. Expert grids are restored after cache
eviction and handle reuse; gate and up can use different grids. Quantization
rounds to nearest even and saturates to the unsigned eight-bit range.
Expert quantization follows the checkpoint's rotation.

This path requires HTP linear engines and WH weights (TL2, QS2 or QS4
experts, QS4 FC/head), with an FP32 router. Missing, extra, unsupported or
invalid encodings fail loading. The old output-projection suffix `:proj`
is accepted as an alias for `:weight`. The supplied expert input grid is
shared across a layer; per-expert down grids are supported. HMX FP16
Hadamard modes are unsupported with these static encodings.

These sidecars cover U8 **linear inputs**, not all simulator QDQ boundaries.
The HTP attention implementation does not consume the external
`kv_quant_file` calibration as a replacement for its own cache scales.
Setting `attention_kv_dtype: "q8"` still selects HTP's existing cache
quantization. The source checkpoint's other activation boundaries require
separate encodings and implementation before full simulator parity can
be claimed.

The host checks include static grids, saturation, ties, distinct gate/up
grids and prefill/decode kernel comparisons. Dense checks also run with
aliased input/output and multiple chunks, as used by Gemma's decode graph.
`run_host_checks.sh` uses the application's `json.hpp`; set `NNTR_JSON_INCLUDE`
if it is installed elsewhere.

The 2026-10-20 S26 Ultra check loads all 7,916 encodings and completes
generation, including expert-cache misses. Both sidecar formats produce
the same smoke-test token IDs and answer `1 + 1 = 2`. An initial static
implementation erased the dense input when clearing its output: Gemma's
decode graph uses the same slot for both. The static dense path now
preserves that input through all chunks. The regression fails before this
fix and matches independent fixed-grid arithmetic afterwards.

**The original agent quality gate still fails.** The PetCare sub-agent TC
(1,625 input tokens, 2,048-token prefill capacity, existing HTP q8 KV) now
produces coherent text and ends at a tool response after 424 tokens. Its call
is `launchPetMonitoring(petName="초코")`, while the original strict fixture
expects `turnOnPetMonitoring(petName="Choco")`. The dynamic control also
chooses `launchPetMonitoring`. Fixing the corrupted dense input does not
establish parity with the original HF simulation or a passing agent TC.

### Korean 2K summary check (2026-10-20)

The S26 Ultra (v81) check uses the same strict weights, q8 KV cache, greedy
sampling, and 2,048 input IDs including the chat template. Prefill processes
2,047 IDs; the last input ID is consumed at the decode handoff. The dynamic
rows below are three-run medians, with no profiling and a cool, idle start
(AP at most 34 C). The static row is one regression run.

| Runtime | Prefill tok/s | Decode tok/s | Output IDs |
| --- | ---: | ---: | ---: |
| Static-capable runtime, dynamic, before Paley optimization | 355.81 | 18.98 | 323 |
| Model compatibility only, dynamic, before optimization | 353.36 | 19.10 | 323 |
| Final static-capable runtime, dynamic | 677.82 | 19.42 | 323 |
| Final static-capable runtime, static | 82.16 | 16.86 | 268 |

Every dynamic run emits the same 323 IDs; the final static run also matches
all 268 IDs of its earlier static result. Both terminate normally. This
checks output stability for one summary, not broad accuracy or the agent
quality gate above. The dynamic summary still omits the satisfaction figure
and misstates the seat-release timing.

The profile located 4.33 of 5.74 seconds of the slow prefill in sparse MoE.
A separate synthetic 2K MoE benchmark held input, routing and weights fixed:
the old HVX/HMX rotations took 35.17/32.13 ms per layer, while the original
Paley implementation took 128.43 ms. These are diagnostic DSP times, not
end-to-end model throughput. The optimization retained the synthetic
output hashes and the real summary IDs; it did not disable rotation.

Gemma4's rotated pack and [LFM2 SR15](lfm2-htp-user-guide.md) use different
rotation metadata; they are not interchangeable.

## Current validation coverage

The latest common-runtime device series on 2026-10-18 reports the
unrotated Gemma4 pack at 1024 input tokens and 64 decode steps (65 output
IDs), with C16, startup preload 128, eight threads, page cache off and no
HTP/layer profiling. Three candidate runs have medians of **621.36 prefill
tok/s** and **12.32 decode tok/s**, with reference IDs preserved. The
[LFM2 guide](lfm2-htp-user-guide.md#validated-performance-preset) gives the
shared series' source revisions and measurement definitions. That series
does not establish a new result for the rotated Gemma4 checkpoint.

Affine kernel validation covers M=1/2/3/4/64/65/129 at small and full Gemma
shapes against an independent int4 reference on S26 Ultra. Whole-model
affine generation completed with matching IDs across profiled and ordinary
runs in the dedicated [decode validation](validation/gemma4-tl2-affine-decode-s26-20261015.json).
Those 2048-input measurements are a separate workload, not samples for
the current 1024-input median.

Rotated mixed-format Gemma4 has device generation and unrotated regression
coverage from the earlier checkpoint-specific trials.
It does not establish perplexity or general model quality. The later
snapshot/demo investigation recorded one intermittent expert-load rebind
failure that did not recur in eleven repeats; it remains unexplained.
Snapshot/chunked Gemma4 generation can differ from a single prefill because
int8 cache scale calibration depends on the rows present at calibration.
Neither observation should be replaced by the superseded first-decode
failure diagnosis.

### Decode optimization, 2026-10-15

The matched S26 Ultra / Hexagon v81 comparison used native TL2/affine WH
weights, 2048 input tokens, C16 (480 expert slots across 30 MoE layers),
startup preload 128, no cache growth, zero model page-cache residency and
no HTP profiling. Each run recorded 101 decode steps and 102 output IDs,
including the first token selected by prefill.

| Runtime | Three decode rates, tok/s | Median tok/s | Median decode ms |
|---|---|---:|---:|
| Before decode optimization | 12.5031, 12.4661, 12.4507 | 12.4661 | 8102 |
| Decode candidate, `NNTR_MOE_SERVER_DONE=1` | 13.0931, 13.2633, 13.2563 | 13.2563 | 7619 |

The median improvement is 6.34%, or 4.78 ms per decode step. The candidate
includes the opt-in completion hint; this is not a measurement of the
default settings alone. All six output ID files matched exactly (SHA256
`6e6ac3677d2f11bfc6b40d7c271d031e1e831427fe6ab29da00a574801e24d60`).
The 16 tok/s target was not reached. These runs preceded the separate
memory changes and cannot be combined with their savings as one measured
result, or pooled with the 2026-10-18 1024-token series above.

The accepted decode changes remove native TL2 activation packing/copies,
reuse FC/attention inputs on v81, keep exactly representable FP32 router
weights in lossless BF16 storage, avoid unused greedy-logit copies and
unrequested per-op timing, and use a dedicated demand-read pool. Coalesced
expert reads and stale speculative-read cancellation were already in the
baseline. Scheduler-yield and alternate-server-affinity experiments were
not retained. Native TL2 decode remains HVX LUT/GEMV; concurrent HMX/HVX
execution is not a validated improvement in this series.

### Memory changes and their measurement scope

The current implementation releases backend prefill staging before decode,
allocates QKV/dense-FFN CPU fallback scratch only when fallback runs, and
tries a gap-reusing layout for eligible contiguous CPU/HTP inference pools,
retaining the old layout if it is smaller. Compact TL2 slots additionally
remove unused arena capacity. These changes preserve the fallback/cache
semantics; they do not remove every FP16 KV allocation.

| Separate experiment | Observation |
|---|---|
| C16 before/after staging-related changes | Active-decode sampled median 3173.25 -> 3124.42 MiB, a 48.83 MiB reduction; DMA accounted for 48.29 MiB |
| C16 sampled external peak in the same records | 3752.35 -> 3490.64 MiB; sparse samples do not establish a complete peak budget |
| Fixed C24 compact-slot comparison | 128 MiB less allocated DMA; logical payload saving was 160.31 MiB |
| Compact C24 versus the old C16 | Active-decode memory was 158.97 MiB higher despite compact storage |

The active-decode statistic is the median of samples 6–9 seconds into the
run, not the final teardown sample. Its external total is ARM smaps RSS
plus unique process-owned DMA-buffer FD sizes minus DMA-mapped RSS overlap.
DSP heap was logged separately and was not joined/deduplicated into this
total. This is not a complete ARM+DMA+DSP memory budget.

Logical slot savings at C16 are 106.875 MiB; 64 MiB arena chunking, spare
prefetch slots and metadata mean this is not measured process-memory saving.
The separate prefill audit avoided 71.5 MiB of dense-FFN reservations and
94 MiB of global-QKV reservations at 2048 rows. Their phase lifetimes
overlap, so these figures must not be added together. Compact C20 was not
measured in that audit, and a larger expert cache was not adopted as proof
of reaching 16 tok/s within the old C16 memory budget.

### Combined-runtime validation, 2026-10-16

After combining the memory and decode changes, one S26 Ultra run with
the completion hint off recorded 669.938 prefill / 13.1905 decode tok/s;
one with it on recorded 666.233 / 13.6634. Both used 2048 input tokens,
preload 128, 480 slots, no growth and zero model page-cache residency.
Both 102-ID outputs matched the six earlier outputs and their hash above.
These are integration checks, not a controlled throughput comparison or a
combined memory measurement. Completion hints remain off by default: an
earlier hint-enabled run ended without a preserved application exit code,
and the cause was not established by later successful repeats.

Android ARM64 engine/app and matching v79/v81 kernels built successfully;
the changed kernels passed `-Wall -Werror`. The complete HTP host suite
passed with GCC 13 and Qualcomm libnative, including 5760 native GEMV
cases, 640 dense/AH quantization cases, compact register/rebind bounds and
legacy transitions, exact router storage and token-completion checks.
ASan/UBSan checked compact layouts and 10,000 randomized planner layouts
without overlap, overflow or allocation growth. Demand-pool checks covered
6000 reuses, exception joining and serialization; completion checks covered
stale/torn records, old-skeleton fallback and races.

TSan could not start because of a host runtime mapping error. Optional
AArch64/NEON emulation lacked its cross-compiler/QEMU. No corpus PPL,
complete platform CI result or new combined memory budget was established.
The measurement files and their archive-only commits have been removed
from this PR; the results and their limitations are retained here.

## Code and checks

| Source | Responsibility |
|---|---|
| [Gemma4 HTP MoE model](../Applications/CausalLM/models/gemma4/gemma4_htp_moe_causallm.cpp) | Per-layer formats and rotation sign loading |
| [Shared MoE layer](../Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp) | Routed experts and residency |
| [TL2 kernels](../nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c) | Native packing, decode and prefill expansion |
| [Affine correction](../nntrainer/tensor/htp_backend/hmx/hexkl_wh_affine.h) | Offset arithmetic |
| [MoE kernels](../nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c) | Rotation, quantization and fused execution |
| [Hadamard kernels](../nntrainer/tensor/htp_backend/hvx/hvx_hadamard_f32.c) | FP32 transforms and signs |

Run `bash test/htp/host/run_host_checks.sh` for host arithmetic and
ownership checks. Device tools include `tools/tl2_device_check.cpp`,
`tools/wh_affine_device_check.cpp`, `tools/tl2_swap_device_check.cpp` and
`tools/gemma4_prefill_device_check.cpp`. Host checks and serialization
checks complement device generation; they do not replace it.
