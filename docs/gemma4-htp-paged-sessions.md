# Gemma4 HTP device KV sessions and prefix reuse

This opt-in path manages the authoritative DSP INT8 KV of the fully resident
Gemma4 MoE graph. The existing contiguous DSP path remains the default.

Exact fresh/reused output-ID checks now pass in the opt-in `q8_row` radix
mode. See [current implementation and validation](gemma4-htp-radix-prefix.md)
for its configuration, memory cost and measured TTFT. The historical model
measurements below used the earlier INT8-state slicing path and must not be
interpreted as validation of the new prefill policy.

| Feature | Implementation |
| --- | --- |
| Sessions | Create, select, delete and fork independent conversations in one loaded model. One conversation executes at a time. |
| Paged KV | Native DSP page tables. Each physical page stores one layer/head's 64-token K tile and V rows. Attention reads the pages directly, without gathering history. |
| Sharing | Retained states share immutable physical pages. Append detaches the pages it modifies before dispatching workers. Quantizer state and runtime positions belong to each state. |
| Radix prefix index | Compressed token edges select causally reproducible FP16 prefill pages. A hit restores the matched pages and processes the unmatched suffix in HMX before seeding DSP INT8 decode. |
| Global / Sliding | Global grows by pages. Sliding reserve releases whole old tiles and remaps retained pages without copying their KV payload. |
| Flash | Existing transactional INT8 snapshots also export/import paged caches. Parking releases the active page table's references. Pages pinned by other sessions/prefixes remain resident. |

## Enable

Build both the ARM engine/application and the DSP skeleton from this revision.
Use the fully resident `htp_fused` Gemma4 MoE profile and configure:

```json
{
  "attention_kv_dtype": "q8_row",
  "skip_prefill": true,
  "init_seq_len": 4096,
  "htp": {
    "kv_prefix_cache": true,
    "kv_session": true,
    "kv_paged": true,
    "kv_request_reserve": true
  }
}
```

All attention, attention projection, dense FFN, MoE and LM head engines must
be HTP. This does not enable the CPU manager's page table on the DSP.
No new RPC wire method was added; the checkpoint control RPC now also supports
retaining, selecting and releasing resident handles. An older DSP skeleton
rejects those actions, so deploy the matching skeleton with the ARM build.

## C++ sessions

After initializing/loading/repacking the model:

```cpp
causal->enableDeviceKVManager(/*max_sessions=*/8, /*max_prefixes=*/8);
const auto a = causal->activeDeviceKVSession(); // session 0
const auto b = causal->createDeviceKVSession();
model->run(prompt_a, false);
causal->selectDeviceKVSession(b);
model->run(prompt_b, false);
causal->selectDeviceKVSession(a);
model->run(new_user_turn_a, false); // only the new turn
const auto branch = causal->forkDeviceKVSession();
```

`selectDeviceKVSession()` captures the outgoing state and restores the incoming
state using shared device pages. No checkpoint file or full-history host copy
is involved. A newly created session starts empty. Forking captures the current
completed state; later parent/child writes are isolated. Select a different
session before deleting one with `destroyDeviceKVSession(id)`.

`setSessionTokenPosition(0)` resets the active conversation's token identity.
Flash snapshots have no token ID history: loading one permits continuation but
cannot publish its unknown history into the radix index. A failed inference
must be reset before switching. These APIs are serialized, request-boundary
operations; they do not schedule concurrent DSP inference.

## Native request sequences

The existing `nntr_causallm MODEL_DIR --requests FILE` interface accepts names:

```json
[
  {"device_session":"A", "prompt":"<complete first A prompt>"},
  {"device_session":"B", "prompt":"<complete first B prompt>"},
  {"device_session":"A", "prompt":"<new A turn only>"},
  {"device_session":"B", "prompt":"<new B turn only>"}
]
```

The first use of each name creates an empty session. Requests without a name
use `default`. Results include switch latency, active KV payload bytes, session
count, retained-prefix state count, radix node count, prefix hits and reused token count.
This is the native request sequence interface; HTTP serving is a separate layer.

## Prefix semantics and bounds

The radix index publishes the available 64-token boundaries of the computed
prefill. A retained 1024-token prompt can supply its first 768 tokens without
executing a separate 768-token prompt. Host FP16 pages are immutable and shared;
boundary metadata contains their layout and attention state. Unlike a decode
snapshot's INT8 pages, these preserve the input precision needed to reproduce
fresh prefill and its final decode quantization. Prefix eviction releases these
pins; decode sessions retain their own INT8 state.

A new complete input searches for the longest available prefix strictly before
its final token. A divergence after 750 tokens reuses 704 tokens. A continuation
of an existing conversation uses its resident state instead of radix lookup.
The unmatched suffix runs as HMX prefill, then the last prompt token seeds and
starts DSP decode. The source session remains unchanged.

This requires `htp.kv_prefix_cache=true` and `attention_kv_dtype=q8_row`.
Per-row quantization and fixed attention reduction tiles remove dependence on
future suffix rows. The original `q8` whole-block quantization remains available
for sessions with prefix caching disabled. Configuration or attention failures
in radix mode raise an error instead of changing the prefill policy silently.

Limits are counts of sessions and retained prefixes, totalling at most 120.
Prefix retention uses LRU. These limits do **not** constitute a DRAM byte
budget. Explicit DSP Flash snapshots remain supported; the host prefill radix
index is not persisted in those snapshots. `getDeviceKVCacheBytes()` reports
active DSP K/V payload, while `getDeviceKVPrefixBytes()` counts unique retained
host prefill K/V pages. See the current validation for their separate costs.

## Checks

Host kernel/session checks:

```sh
CC=gcc-12 test/htp/host/run_device_kv_checks.sh
# Optional sanitizer build (compiler must support _Float16):
CC=gcc-12 SANITIZE=1 ASAN_OPTIONS=detect_leaks=0 \
  test/htp/host/run_device_kv_checks.sh
```

The kernel check compares contiguous and paged attention byte for byte across
page boundaries for sliding/global shapes and FP16, Q16 and Q8 query arithmetic.
It also checks shared tails, parent preservation, whole-page sliding release,
snapshot import and parking. The C++ manager check covers radix splits, proper
prefix terminals, LRU, session branching and failed restore isolation.

`tools/kv_cache/check_htp_sessions.cpp` is a native Android integration checker.
It uses the model's tokenizer and unmodified greedy generation, checks repeated
A/B switches and fork continuation against independent references, checks radix
hits against independent full-history prefill, and checks paged Flash restore.
The current native checker targets build through `app_build` with
`-Denable-htp-device-check=true`. Runtime metrics report steady decode
as median ms/token; convert to TPS with `1000 / decode_steady_ms`.

## Historical Android result — before exact prefill reuse

SM-S948N, Android 16, SM8850/v81; Gemma4 26B A4B HTP fused, 8 ARM threads,
8 requested decode steps, no forced tokens/logits instrumentation. Seed lengths
were 128 and 3072 tokens. The long conversation exercised a nonzero sliding
base. Whole-tile release is covered separately by the host kernel check.

| Check | Result |
| --- | --- |
| Paged vs contiguous references | All 6 requests: output IDs and final positions equal. |
| A/B alternation | All 6 requests: equal to independent resident references. |
| Forks | Both seed forks resumed correctly after parents advanced twice. |
| Radix reuse | 2 hits, reusing 129 and 3074 processed tokens; each computed only the 24-token suffix; outputs equal to resident continuation. |
| Paged Flash | Save/park reports zero active payload; restore and continuation preserve output IDs. Other saved states remain pinned. |
| CLI session names | All 4 A/B/A/B requests preserve IDs and final positions. |
| Resident switch | Five actual switches: 4.99–13.05 ms, median 11.49 ms. No Flash I/O. |

These are one-pass integration timings, with no frequency lock. The reference
runs' steady decode samples ranged from one to seven steps, and the 128-token
seed ended before there were steady steps. In matched nonzero samples, paged
median ms/token differed by about -0.7% to +2.9%. This is insufficient to certify
no performance regression. First-prompt TTFT was 1802 vs 1798 ms at 128 tokens
and 5351 vs 5391 ms at 3072 tokens (contiguous vs paged). Radix suffix TTFT was
1768 and 1958 ms; these runs are compared to resident continuation for correctness,
not presented as a controlled fresh-recompute speedup.

Full fixture, configuration, per-request IDs/positions/metrics, CLI output and
binary hashes: [validation JSON](validation/gemma4-dsp-paged-sessions-20261104.json).


## Historical speed comparison with session management enabled

The earlier contiguous/paged comparison isolates the page layout, with the
session manager disabled in both reference phases. The table below instead
compares the contiguous independent resident references to paged KV **with the
session manager and radix publication enabled** during A/B alternation. The
manager TTFT column adds the externally measured session selection time to
`run()`'s TTFT; decode TPS is `1000 / median steady ms/token` and excludes the
first decode step. Initial means the seed request; continuation 1/2 each adds
24 actual input tokens, including pending feedback. Seed lengths identify the
conversation, not its current total history length.

| Seed tokens | Request | Baseline TTFT (ms) | Manager switch + TTFT (ms) | Baseline decode TPS | Manager decode TPS | TPS change | Steady steps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | Initial | 1802 | 1715 | — | — | — | 0 |
| 3072 | Initial | 5351 | 5352 | 13.38 | 14.05 | +5.04% | 1 |
| 128 | Continuation 1 | 1774 | 1708 | 11.70 | 12.10 | +3.39% | 7 |
| 3072 | Continuation 1 | 1759 | 1776 | 12.50 | 12.49 | -0.11% | 7 |
| 128 | Continuation 2 | 1717 | 1717 | 12.41 | 12.02 | -3.14% | 7 |
| 3072 | Continuation 2 | 1954 | 1819 | 16.89 | 16.08 | -4.80% | 7 |

All outputs and positions match. This was one pass in separate phases without
frequency locking or randomized interleaving; timings include expert-cache/I/O
and thermal variability. The +5.04% sample contains only one steady step.
The observed -4.80% to +5.04% TPS range does not establish an improvement or
prove zero overhead. It also cannot attribute a difference specifically to
page lookup, COW or the manager. Prefix-hit timings (1768/1958 ms) were not
paired with full-history recomputation of the same complete inputs, so they
must not be quoted as a measured recomputation speedup.


## Historical INT8 page-prefix validation

`tools/kv_cache/check_htp_page_prefix.cpp` checks the case that a complete
1024-token source is retained once and a new input diverges after token 768.
The host manager check requires 768 tokens to be reused without capturing an
independent 768-token state. The DSP kernel check verifies byte-exact attention
against the same encoded prefix, release of active suffix references and
preservation of the original 1024-token source. ASan/UBSan covers this truncation
and branching path. Sliding coverage tests keep an older available source and
exclude only boundaries whose required rows were actually released.


### Earlier Android partial-prefix output mismatch

The page-prefix fix was checked separately from the earlier checkpoint-only
measurements. One 1024-token prompt was retained. Each of two new inputs shared
only its first 768 tokens, then supplied a distinct 32-token suffix. Neither had
an independently captured 768-token checkpoint. Both reused 768 tokens and
computed 32 suffix tokens (31 in the prefill metric plus one final input token
in decode with `skip_prefill=true`). The original source conversation resumed
with exactly the same IDs as its independent resident continuation reference.
Host encoded-attention checks and ASan/UBSan also pass, including an unaligned
sliding-local tail and zeroing uncommitted tail rows in exported snapshots.

**Fresh greedy output equality failed.** The native checker records this and
exits 3 after completing the structural/source checks. It does not label the
whole model validation as passed. Fresh contiguous and paged runs had identical
IDs for both inputs; the difference appears between fresh execution and reuse.
The cause within the differing execution/quantizer histories is not isolated.

| Branch | Fresh contiguous TTFT (ms) | Fresh paged TTFT (ms) | Reuse TTFT (ms) | Fresh output | Reuse output |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 2287 | 2291 | 2662 | DIAMOND-123<turn\|> | <\|channel>thought <channel\|>DIAMOND-123<turn\|> |
| 2 | 2387 | 2374 | 2630 | <eos> | <\|channel>thought <channel\|>RUBY-456<turn\|> |

These samples show no latency benefit for this short-suffix case, and their
output differences prevent a controlled same-output speed claim. The normal
fresh path can process multiple input tokens in HMX, whereas resident suffix
continuation currently runs through the token decode path. This route difference
is relevant to the timing, but its separate cost was not isolated. An earlier
`skip_prefill=false` trial also differed and is retained in the evidence.
Full data and the failed equality outcome are in
[partial-prefix validation](validation/gemma4-dsp-page-prefix-20261104.json).
