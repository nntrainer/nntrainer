# Batched prefill after a decode radix hit

## Implemented behavior

The opt-in 4096-token decode checkpoint now restores enough host state to run
the unmatched prompt with the original HMX prefill path. Previously a decode
radix hit forced every remaining input token through sequential M1 decode.

1. Capture the existing DSP INT8 snapshot and its prefill calibration at the
   boundary. The snapshot still contains INT8 K/V, per-dimension scales,
   global/sliding bases and processed token identity. Flash writes remain on
   the bounded background worker.
2. Load and verify the immutable Flash image. Reconstruct row-major FP16 host
   history from the tiled INT8 values and saved scales. Preserve sliding
   dropped maxima and restore the original prefill calibration.
3. Run the suffix as native HMX prefill, subject to the existing activation
   plane size. Under `skip_prefill=true`, the final input row uses the original
   first-decode-row path. Resume ordinary resident decode afterwards.

DSP KV and the Flash checkpoint remain **INT8**. FP16 is temporary host input
to the existing prefill implementation, which quantizes its attention KV.
This bridge therefore includes INT8 → FP16 → prefill INT8 conversion. It is
not a direct INT8-to-HMX KV handoff and cannot recover original FP16 precision.
No projection, FFN, MoE, attention kernel or calibration policy was changed.

ARM NEON conversion has an independent scalar oracle, including multiple
heads, partial tiles, dimension tails and sliding offsets. The enclosing Flash
checksum already covers the complete immutable DSP image, so this caller skips
a second full embedded-checksum pass. Other callers default to checking that
checksum. Blob loading also copies directly into the result instead of
concatenating a temporary buffer and taking a second copy with `substr()`.

Reconstructed decode histories are not published as native FP16 prefill
boundaries. DSP session save/load/select/fork continue using their original
INT8 state path. All settings remain opt-in as documented in
[decode checkpoints](gemma4-htp-decode-prefix-checkpoints.md).

## Android measurements

SM-S948N / SM8850 v81, Gemma4 26B A4B, existing q8/Q2 U16 path, 4096-row
prefill plane. Each measured hit follows a Flash-worker drain; the worker
synchronizes writes and advises eviction of clean file pages. Frequencies are
not locked and filesystem advice does not guarantee a cold device.

The source is the mountain-village filler. Prefill ends one token before the
boundary, then generates 32 tokens. A new session supplies the complete token
history plus ` Continue with one more sentence.`. Exactly 4096 or 8192 tokens
are reused; the remaining 37 input tokens use one 36-row HMX prefill followed
by the original final-input decode row. The second hit restores the same
decode checkpoint, rather than a newly captured prefill endpoint.

Times are seconds. Historical M1 results are retained for context, not a
frequency-controlled paired comparison.

| Boundary | Previous sequential hit | Batched hit | Repeated batched hit | Full recomputation |
|---|---:|---:|---:|---:|
| 4096 | 3.710 | 2.641 | 2.708 | 9.226 |
| 8192 | 5.507 | 3.317 | 3.299 | 18.311 |

| Boundary | Flash load and blob extraction | Host conversion | Bind | Suffix HMX prefill |
|---|---:|---:|---:|---:|
| 4096 | 0.493 | 0.149 | 0.018 | 1.447 |
| 8192 | 0.860 | 0.259 | 0.025 | 1.470 |

These components are not total TTFT: tokenization, allocation, native handoff
and first decode also contribute. `read_ms` includes storage verification and
blob extraction, not just physical Flash reads. Values are from the first hit;
the JSON files contain both hits and complete generated IDs.

The older 2.94-second FP16 Flash result used another suffix and earlier device
conditions. A current control using the unchanged FP16 prefill Flash path
reused 8192 tokens, ran 26 HMX rows and took **4.179 s**, including **1.735 s**
of Flash loading/checksum. A decode-checkpoint code-answer request with 27 HMX
rows took **3.725 / 3.701 s**, including 1.021 s load and 0.310 s conversion
for its first hit. These are adjacent observations with slightly different
prompts and producer histories, not a claim of universal timing parity.

## Output and state validation

* The first and repeated decode-boundary hits return identical IDs and final
  positions. Source session selection and restoration remain valid.
* The 8K `RUBY-456` fixture returns the expected answer with identical IDs for
  live continuation, both Flash hits and full recomputation. Both 41-row and
  27-row HMX suffix fixtures pass.
* A 3967-token prefill followed by 144 generated tokens crosses 4096 with
  sliding rows older than the live window. Restoration trims those rows,
  carries their maxima and passes repeated restore and position checks.
  Its code-answer variant also returns `RUBY-456`; output IDs match live
  continuation, repeated restoration and full recomputation.
* Deliberate Flash corruption is rejected and falls back to the same IDs as
  full recomputation. Failed/capacity-limited storage, LRU, cleanup and immutable
  backing checks also pass.
* Android session switching, forks and direct DSP Flash snapshot restoration
  pass the existing session checker.
* Ubuntu: five KV/cache/dispatch test binaries and four Gemma4 HTP model
  contract/differential cases pass. Scalar/emulated DSP checks pass normally
  and under ASan/UBSan; Flash blob checks pass normally and under ASan/UBSan.

**Output equivalence is limited.** The 8K free-continuation filler and the
longer 4K sliding fixture have different IDs from M1 continuation/full prefill.
The longer filler also emits structural tokens in its free continuation.
Repeatability and a correct code-answer fixture do not establish general
quality equivalence. Dequantized history plus batched HMX queries has different
rounding/quantization from original M1 continuation. Direct DSP snapshot resume
remains available for preserving that trajectory. An early multi-turn quality
fixture returned EOS even under full recomputation and is excluded as an
invalid quality control.

Checkpoint export/staging is still synchronous and has measurable boundary
cost (about 0.6–1.2 s in these runs). Only checksum/file writing is asynchronous.
The change addresses suffix execution and restore overhead; it does not make
capture free or implement direct INT8 prefill KV consumption.

## Reproduction and evidence

Build the Android engine, install its headers, then rebuild the CausalLM
device checks. Rebuild consumers after changing `ComputeOps`; mixing installed
headers and old objects produces an ABI mismatch.

```sh
export LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH="$PWD"
export NNTR_NUM_THREADS=8 NNTR_HTP_KV_MANAGER=1
./htp_decode_kv_convert_check
./htp_prefix_flash_check
./htp_decode_prefix_check pack input.txt cache 4k.json 4096
./htp_decode_prefix_check pack input.txt cache 8k.json 8192

# Query fixture: remember RUBY-456 and reply with the code.
NNTR_DECODE_PREFIX_SUFFIX_FILE=quality-suffix.txt \
NNTR_DECODE_PREFIX_EXPECT=RUBY-456 \
  ./htp_decode_prefix_check pack input.txt cache code-8k.json 8192 8161
```

`input.txt` starts with `<bos><|turn>user\n` followed by enough repetitions of
`The mountain village has orchards, a bridge and a library. ` to reach the
requested seed length. The checker verifies token round trips, truncates at
that seed length and constructs all subsequent requests from actual generated
IDs. The default is `boundary - 1`; the last argument overrides it. The
quality suffix is `\n\nRemember the code RUBY-456. What is the code? Reply only
with the code.<turn|>\n<|turn>model\n`.

Raw IDs, counters, configuration and control summaries are in
[validation/decode-prefix-batch-20261105](validation/decode-prefix-batch-20261105).
