# Gemma4 HTP device KV sessions and prefix reuse

This opt-in path manages the authoritative DSP INT8 KV of the fully resident
Gemma4 MoE graph. The existing contiguous DSP path remains the default.

Exact fresh/reused output-ID checks now pass in the opt-in `q8` radix
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
  "attention_kv_dtype": "q8",
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

The radix index publishes the available 256-token calibration boundaries of
the computed prefill; physical DSP pages contain 64 tokens. A retained
1024-token prompt can supply its first 768 tokens without executing a separate
768-token prompt. Host FP16 pages are immutable and shared;
boundary metadata contains their layout and attention state. Unlike a decode
snapshot's INT8 pages, these preserve the input precision needed to reproduce
fresh prefill and its final decode quantization. Prefix eviction releases these
pins; decode sessions retain their own INT8 state.

A new complete input searches for the longest available prefix strictly before
its final token. A divergence after 750 tokens reuses 512 tokens in the current
Q2 policy.
At least three suffix tokens are left for the block-prefill graph. A
continuation of an existing conversation uses its resident state instead of
radix lookup.
The unmatched suffix runs as HMX prefill, then the last prompt token seeds and
starts DSP decode. The source session remains unchanged.

This requires `htp.kv_prefix_cache=true` and `attention_kv_dtype=q8`.
The existing Q2 kernels use causal 256-query calibration domains in prefix
mode. This changes calibration grouping relative to whole-call q8: exact
fresh/reused IDs are verified under the new grouping, while equivalence with
legacy whole-call outputs is not claimed. The original q8 grouping remains
available for sessions with prefix caching disabled. Configuration or
attention failures in radix mode raise an error instead of changing the prefill policy silently.

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
