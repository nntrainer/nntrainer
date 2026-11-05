# Decode KV prefix checkpoints: implementation plan

Starting point: `f1bbd0995`, branch `kv/decode-4k-prefix-flash`.

Current batched suffix restoration: [implementation and validation](gemma4-htp-decode-prefix-batch.md).

Initial status and measurements: [decode checkpoint report](gemma4-htp-decode-prefix-checkpoints.md).
4K and 8K device checkpoints and same-trajectory continuation pass on Android.
Full-prefill output differs, and synchronous capture is measurable, so this is
an opt-in implementation rather than an output- and latency-equivalent feature.

1. Done: capture the DSP INT8 KV state, scales, per-layer sliding bases, absolute
   position and the actual processed token IDs at each 4096-token boundary.
   Account for the pending sampled token and forced replay inputs.
2. Done, with temporary-memory and capture-latency work remaining: export a
   stable copy without letting Flash I/O read mutable DSP buffers.
   Enforce bounded staging and Flash capacity, handle I/O failures, and expose
   counters and timing. Keep the inference path valid when storage is full.
3. Done: register each complete decode checkpoint in the existing token radix index.
   Give prefill and decode checkpoints distinct types, preserve the longest
   compatible match and manage their lifetime with the existing prefix LRU.
4. Done: reconstruct the selected INT8 checkpoint into native host FP16
   history, restore calibration, and process the unmatched suffix with batched
   HMX prefill. Direct session snapshot restoration retains the M1 path. Ensure token identity,
   session switches, forks, reset and snapshot cleanup remain correct.
5. Validated in the initial implementation: validate boundary cases around 4096 and 8192 positions, multiple branches,
   sliding-window eviction, corrupt/evicted checkpoints and session switching.
   Compare exact output IDs and position to an uninterrupted decode reference,
   and report any differences against a full prefill recomputation separately.
6. Initial single-sample Android measurements done; repeat and broaden: measure decode overhead, checkpoint capture/write,
   restore and end-to-end latency with hits and misses. Run the relevant host
   checks, format changed C++ with clang-format 14, and commit/push the branch.

## Constraints identified

* The existing prefill radix stores FP16 prefill frames and calibration; the
  existing DSP snapshot stores INT8 decode state. Their restore paths differ.
* `save_decode_kv()` currently performs DSP export and file writing while
  holding the decode graph mutex. Moving that call to a worker would race live
  inference; immutable staging must first be exported on the inference thread.
* A radix hit must not run a batched host prefill against host KV placeholders
  that do not contain the restored DSP INT8 history.
* Token IDs alone do not guarantee bit-identical state when the same tokens
  were originally evaluated by different prefill/decode paths. Validation must
  distinguish continuing the captured state from recomputing it as prefill.
