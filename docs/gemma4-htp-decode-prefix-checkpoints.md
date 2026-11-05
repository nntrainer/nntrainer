# Gemma4 HTP decode prefix checkpoints (experimental)

Decode can cross a 4096-token boundary after its original prefill. With the
option below, the radix manager captures the completed DSP INT8 state at each
such absolute position, including per-layer scales, global and sliding KV,
sliding bases and processed token IDs. It exports an immutable checkpoint on
the inference thread and submits it to the bounded Flash worker. The worker
writes and checksums in parallel with subsequent inference.

A radix hit reconstructs native FP16 host rows from the saved INT8 values and
scales, restores the saved prefill calibration, and runs unmatched input through
native batched HMX prefill. The following decode uses the existing FP16 seed
handoff. See [batched restoration validation](gemma4-htp-decode-prefix-batch.md)
for current measurements and output semantics. Direct DSP session snapshot
save/load remains separate.

```json
"kv_prefix_flash": {
  "directory": "/data/local/tmp/nntr-prefix-cache",
  "budget_bytes": 2147483648,
  "staging_bytes": 1073741824,
  "decode_checkpoints": true
}
```

This also requires the existing `htp.kv_session`, `htp.kv_paged` and
`htp.kv_prefix_cache` settings. `decode_checkpoints` defaults to false. The
radix prefix count, Flash capacity and staging budget are shared with existing
prefill prefixes. As with existing Flash backing, a full budget, failed write,
eviction or corrupted file can leave a boundary unavailable; lookup falls back
to another available boundary or recomputation. Files are process-local.

## Semantics and limitations

A checkpoint stores the original INT8 history. Radix reuse now evaluates new
queries with HMX prefill, so it does **not** promise bit-identical results to
sequential M1 continuation or full recomputation. Reconstructed FP16 rows retain
the saved INT8 precision; they cannot recover values discarded by quantization.
Token-only matching also does not identify the original calculation history.
Native FP16 prefill prefixes are kept separate: a reconstructed decode prefix
is not republished as an original FP16 prefill boundary.

Exporting DSP data and copying it into immutable host staging are synchronous.
Only the subsequent checksum and Flash write are asynchronous. Thus a
boundary can cause a visible decode latency spike; the feature does not meet a
zero-regression or fully hidden-capture requirement. Staging is bounded for
queued bytes, but export and encoding also allocate temporary host buffers.

## Historical Android measurements before batched restoration

SM-S948N, SM8850 v81, Gemma4 26B A4B, original q8/Q2 HTP path. One run per
boundary; a prompt ending one token before the boundary generated across it.
The reference continues that same live session. The hit starts a new request
with the same tokens, restores the Flash checkpoint, then processes the suffix.
The full-prefill comparison processes the same complete token sequence from
zero. Times are seconds; the generation interval includes multiple tokens.

| Decode boundary | Live continuation TTFT | Flash hit TTFT | Full prefill TTFT | Capture and staging | Background write |
|---|---:|---:|---:|---:|---:|
| 4096 | 0.716 | 3.710 | 8.640 | 0.499 | 0.457 |
| 8192 | 0.794 | 5.507 | 16.847 | 0.856 | 1.020 |

At both boundaries, all output IDs and final positions matched the live
continuation. The full-prefill output IDs differed, so the TTFT numbers do not
establish an output-equivalent speedup over full prefill. The seed generation
interval was 2.605 s without capture and 2.986 s with capture at 4K; at 8K it
was 4.273 s and 4.625 s respectively. These are single samples without locked
frequencies. See [4K evidence](validation/decode-prefix-4k-20261105/4k.json)
and [8K evidence](validation/decode-prefix-4k-20261105/8k.json) for token IDs.

The Android session regression covers two sessions, repeated switching, forks
and the existing DSP snapshot save/load. Host checks cover radix publication,
Flash capacity, asynchronous storage, corruption and DSP snapshot operations.
The decode checkpoint test also switches between two sessions and verifies
that a deliberately corrupted file falls back to the same output as full
recomputation. An additional test begins decode at exactly 4096 tokens, where
prefill and decode entries share a token boundary.

Remaining work: reduce synchronous export/copy latency and temporary memory;
exercise repeated boundary crossing in one long decode, contention and
saturation; and establish how callers identify the intended decode trajectory.
Full-prefill equivalence cannot be assumed from a token match alone.
