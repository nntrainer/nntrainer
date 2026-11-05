# Gemma4 HTP device KV sessions and prefix reuse

This opt-in path manages the authoritative DSP INT8 KV of the fully resident
Gemma4 MoE graph. The existing contiguous DSP path remains the default.

Exact fresh/reused output-ID checks now pass in the opt-in `q8` radix
mode. See [current implementation and validation](gemma4-htp-radix-prefix.md)
for its configuration, memory cost, optional asynchronous Flash backing and
measured TTFT. Earlier INT8-state slicing measurements are historical and do
not describe the current original-call prefill policy.

| Feature | Implementation |
| --- | --- |
| Sessions | Create, select, delete and fork independent conversations in one loaded model. One conversation executes at a time. |
| Paged KV | Native DSP page tables. Each physical page stores one layer/head's 64-token K tile and V rows. Attention reads the pages directly, without gathering history. |
| Sharing | DSP sessions/forks share immutable physical pages. Append detaches modified pages. Radix entries independently pin host FP16 slabs and exact calibration. |
| Radix prefix index | Compressed token edges select completed original prefill calls with a compatible call plan. A hit restores the matched pages and processes the unmatched suffix in HMX before seeding DSP INT8 decode. |
| Global / Sliding | Global grows by pages. Sliding reserve releases whole old tiles and remaps retained pages without copying their KV payload. |
| Flash | DSP INT8 session snapshots support explicit save/park/restore. Optional asynchronous host-prefix backing retains earlier native prefill boundaries for radix hits under byte budgets and file LRU. |

## Enable

Build both the ARM engine/application and the DSP skeleton from this revision.
Use the fully resident `htp_fused` Gemma4 MoE profile and configure:

```json
{
  "attention_kv_dtype": "q8",
  "skip_prefill": true,
  "init_seq_len": 4096,
  "htp": {
    "attn_q2": true,
    "oproj_u16": true,
    "kv_seed_q": false,
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

The compressed radix index publishes completed original prefill calls, not
arbitrary physical-page boundaries. It retains the last completed full native
block (4096 tokens for the 4608-token source, 8192 for the 8704-token source)
plus the final multi-row prefill endpoint in DRAM. Optional asynchronous prefix
Flash backing preserves earlier completed native boundaries; it does not change
calibration or split native calls. A new request must match both token
IDs and the original call plan. Repeated complete inputs can reuse their full
prefill; a longer request can reuse completed full native chunks. A prefix inside
a whole-call calibration group safely misses. For example, a single 1024-token
source cannot supply an exact 768-token crop under the unchanged original q8
policy. Physical DSP pages remain 64 tokens. Optional direct KVQ seed mode
(`htp.kv_seed_q=true`) preserves its original computation and does not retain
prefixes; the validated host snapshot path uses `false`.

Host FP16 snapshots retain immutable shared slabs, exact calibration and sliding
layout. A hit resumes the remaining original calls and the final prompt token's
DSP decode. No 256-token calibration policy or per-domain O-projection encoding
is used. Non-default repetition penalties and debug token-by-token resumed prefill
fall back to the original computation. One-row native prefill tails initialize
their KV window before transferring ownership to DSP. Resident conversations
continue using their existing native DSP state.

Limits are counts of sessions and retained prefixes, totalling at most 120.
Prefix retention uses LRU; there is no DRAM byte budget. Shared views pin entire
RPC slabs, including unused reserved capacity. `getDeviceKVCacheBytes()` reports
active DSP payload; `getDeviceKVPrefixBytes()` reports unique pinned host slabs
and materialized page copies. Flash snapshots preserve DSP session state, not
the host radix index. Optional prefix Flash backing has its own staging/file
byte budgets and process-local radix metadata. See [the current contract and evidence](gemma4-htp-radix-prefix.md).

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
