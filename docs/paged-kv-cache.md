# Paged CPU KV cache and attention

The internal `nntrainer::kv::KvCacheManager` supports a paged CPU backend.
Set `CacheSpec::page_tokens` to a positive token count (for example 16).
Zero keeps the existing contiguous backend. Each owner layer and batch has
its own logical block table; layer aliases resolve to the owner's table.
Only blocks needed for an admitted step are allocated. Physical blocks hold
token-major KV heads, without expanding grouped query heads.

`KvPagePool` can be shared between managers, allowing independent sessions to
reuse released physical storage. Each acquired block is a handle; when the
session table and every exported view drop it, the block returns to a per-size
idle list, so acquiring and releasing blocks is O(1). Its optional byte budget
includes 64-byte allocation rounding and idle blocks; when a new allocation
would exceed it, idle blocks are freed first. `trim()` frees all idle blocks.
Pool and session operations are synchronous and thread confined; they must all
run on the same thread or be externally serialized.

```cpp
using namespace nntrainer::kv;
auto pool = std::make_shared<KvPagePool>(256 * 1024 * 1024);
CacheSpec spec;
LayerSpec layer;
layer.num_kv_heads = 2;
layer.head_dim = 64;
layer.capacity = layer.max_tokens = 4096;
layer.encoding = Encoding::Float32;
layer.storage_type = StorageType::Float32;
spec.layers = {layer};
spec.page_tokens = 16;
KvCacheManager cache;
cache.initialize(spec, std::make_shared<nntrainer::MemAllocator>(), pool);

// k/v: [new_tokens, kv_heads, head_dim], in the cache encoding.
// q: FP32 [new_tokens, query_heads, head_dim], with RoPE already applied.
auto step = cache.beginStep(new_tokens);
cache.append(step, 0, k.data(), v.data(), k.size() * sizeof(float));
cache.markReady(step, 0); // After producing all batches of this owner layer.
AttentionOptions options;
options.query_heads = 8;
auto output = cache.attention(step, 0, q, options);
cache.commit(std::move(step));
```

`attention()` reads the checked KV segments directly. It does not gather KV
history or allocate the full attention score matrix. It supports causal
prefill, decode, GQA/MQA/MHA, noncausal attention, a trailing window and
optional logit softcapping. Queries and results are FP32; cache data can be
FP32 or IEEE binary16 (including UINT16 backing without native half support).
Scale defaults to `1/sqrt(head_dim)`. The window ends at each query's causal
endpoint, or at the step endpoint for noncausal attention; pages entirely
before the window are skipped.

One task handles a query token and the query heads sharing one KV head, so
each K/V row is loaded once per group. The group is split only when needed to
occupy every `ThreadManager` thread, so decode is parallel as well. Keys are
scored in chunks of 64 with an FP32 online softmax (one rescale per chunk).
Dot products and value accumulation use AVX2+FMA (F16C for binary16) or
AArch64 NEON, with a scalar fallback elsewhere, including ARMv7. Small calls
stay on the calling thread. Exceptions raised by workers, such as a nonfinite
score, are rethrown on the calling thread.

Producers can also write through `writeKey()` / `writeValue()` segments instead
of `append()`. A page-crossing write returns multiple segments in logical order.
Call `markReady()` before reading or attending, and commit only after every
owner producer is ready. Layer aliases cannot produce their own KV.

Failed step admission leaves the session unchanged; successfully allocated
blocks may remain idle in the pool. Aborting a step returns newly allocated
whole blocks beyond the committed position. Reset returns all session blocks;
backward `setPosition()` returns whole blocks beyond the new position. A
partial tail remains private to that session. Forward seek is rejected because
it would publish unwritten KV. Sharing full blocks between sessions is
described under "Sessions and prefix caching" below.

Every exported view pins its physical block. Invalidating a view prevents API
access but does not recycle its allocation until the view itself is released.
A raw pointer must not escape its synchronous operation. `CacheStats` reports
session-owned bytes; `KvPagePool::reservedBytes()` also includes idle and
view-pinned blocks. A bounded pool can reject a step while stale views pin
blocks; release the views before retrying.

CausalLM and SentenceTransformer inference can select this backend at model
construction time:

```sh
NNTR_ENGINE=cpu NNTR_KV_PAGED=1 <application> <arguments>
```

`NNTR_KV_PAGED=0` (the default) retains the contiguous backend. Page size defaults
to the host flash key block (512 tokens on x86, 128 elsewhere);
`NNTR_KV_PAGE_TOKENS` overrides it with a positive integer. The current graph adapter admits
batch-1 NCHW CPU inference with external caches. INT8, OHWI, training, the legacy
KV manager and device backends are rejected rather than falling back to a dense
cache. Sliding windows use attention masks over paged history, with no ring
layout. Prefix caching is opt-in (see below); there is no asynchronous
completion.

In paged mode graph cache inputs have shape `[1, 1, 1, kv_width]`. These one-row
allocations are stable per-layer identifiers, including distinct identifiers for
aliased layers; they never contain cache history. Their pointers resolve through
the checked `KvStepScope` bound around each inference call. MHA dispatches to the
paged path before constructing any dense cache slices or entering a device arm.
It applies the layer's RoPE to the new Q/K token block, converts K/V to the cache
encoding, and appends only owner KV. Aliased layers consume the source's pages
without another append. Skipped prefill consumers preserve the
producer/readiness contract. FP32 activations are rotated in FP32 and then
narrowed with IEEE binary16 rounding, like the contiguous store.
`Tensor::copyData` is not used for that conversion: from FP32 to UINT16 it is
an integer cast on ARM and the fallback backend.

With a binary16 cache and no sink or softcap, attention runs through the host
flash kernel (`gemm_attention`) over a page table. If `page_tokens` is a
multiple of its key block, the x86 arm and the tiled ARM arm
(`NNTR_CPU_KV_TILED`) read pages in place. Other configurations, including
the default ARM arm whose GEMMs require contiguous rows, gather the history
into the kernel's pack buffers once per call. FP32 caches, attention sinks
(a denominator-only logit per query head), softcapping, and
`NNTR_KV_NO_FLASH=1` use the manager's online-softmax reader instead.

The facade's `getKeyCache()` / `getValueCache()` return identity tensors in paged
mode. They must not be treated as dense history; its dense read/write view APIs
reject that use. The core's `keyMemory()` / `valueMemory()` also reject paged
storage. Graph consumers use `graphKeyMemory()` / `graphValueMemory()` and the
checked step access API instead. Pool byte counts exclude the small graph
identity allocations and temporary new-token transformation buffers.

Snapshots stream logical K/V planes directly to/from the pages, preserving the
existing geometry, alias and model-tag checks. Loading stages a replacement
page table and publishes it only after the complete payload succeeds. Graph
identifiers remain stable across reset and restore. A bounded shared pool may
need space for both the old and replacement tables during restore.

The snapshot writer now emits version 2 with raw cache-encoding bytes on both
the paged and contiguous paths, allowing interchange between them. Version 1
FP32/FP16 snapshots remain readable. Version 1 and headerless UINT16 snapshots
are rejected: Tensor serialization adds quantization metadata to UINT16, which
is not a raw half-bit KV stream. New UINT16 snapshots have an explicit raw-byte
format and do not use quantized Tensor serialization.

Numerical tests in `unittest_kv_cache_core.cpp` compare paged attention against
an independent dense softmax reference across page-crossing prefill and decode,
multiple batches, GQA, aliases, windows and softcapping. Additional cases cover
binary16 decoding, pinned pages, shared pool admission, reuse and allocation
failure. A real Qwen graph regression compares contiguous and paged logits across
page boundaries and snapshot restore; Gemma graph cases exercise paged inference.
These tests were added without building or running them on macOS.

## Sessions and prefix caching

A `KvCacheManager` holds one or more sessions. A session is one independent
sequence: its own page table, committed position, step state and token ids.
`initialize()` creates and selects session `KvCacheManager::kDefaultSession`
(0). With paged storage, `createSession()` adds empty sessions that share the
spec, page pool, prefix cache and the one-row graph identity tensors, so a
graph bound once can run any of them. `selectSession()` routes every later
step, view, seek and snapshot call to that session; it is rejected while a
step is live. `destroySession()` releases a session that is not selected;
its exported views become stale. Contiguous storage keeps exactly one
session.

`beginStep(count, tokens)` can carry the ids of the rows the step writes.
On commit they extend the session's known ids when the known prefix already
reaches the step; a step without ids ends the known prefix until the session
is truncated back into it. Aborted steps discard their ids. Snapshot restore
forgets all ids, because snapshots do not store them.

`KvPrefixCache` shares full pages between sessions, including sessions of
other managers that use the same pool and page layout:

```cpp
auto pool = std::make_shared<KvPagePool>(256 * 1024 * 1024);
auto prefix = std::make_shared<KvPrefixCache>(pool, /*max_blocks=*/0);
KvCacheManager cache;
cache.initialize(spec, std::make_shared<nntrainer::MemAllocator>(), pool,
                 prefix);

// Prompt ids; keep the last one for the step that produces logits.
size_t from = cache.reusePrefix(ids.data(), ids.size() - 1);
auto step = cache.beginStep(ids.size() - from, ids.data() + from);
```

A block covers one page of positions and holds that page of every owner layer
for K and V. Its identity is the parent block plus its own token ids, compared
exactly, so a block matches only after the identical prefix and a hash
collision cannot produce a false match. On commit, every newly completed full
page whose ids are known is published; a partial tail page stays private.
`reusePrefix()` keeps the session's own rows that match the leading tokens,
or, if the cache holds a longer chain, adopts those shared pages without
copying or allocating; all later rows are released. It returns the position
to continue from, which is always a page multiple when cached pages were
adopted.

Shared pages are never written in place. Appends always start in a page that
the session owns, except after a truncation into the middle of a shared page;
the next step then copies that page's committed rows into a private page
before writing (copy-on-write). The original stays intact for the cache and
for other sessions.

The cache holds page handles. `CacheStats::backing_bytes` counts the pages a
session references, including shared ones, so summing it over sessions
counts shared pages more than once; `KvPagePool::reservedBytes()` counts each
page once, including pages only the cache holds. When the pool budget would
be exceeded, the pool asks registered reclaimers to free pages before
failing. The prefix cache then drops its least recently used leaf block
whose pages no session references. `max_blocks` bounds the number of cached
blocks the same way; blocks still referenced by a session are never dropped,
since that would free nothing. `evictUnused()` and `clear()` drop blocks
explicitly.

The first manager attached to a prefix cache binds its page layout (page
size, layer count, per-layer heads, head size, encoding, aliases and
producers). Other managers must match, must use the cache's pool, and must
use batch 1. The check cannot see weights: share one prefix cache only
between sessions of the same model. Reused rows are only correct when the KV
cache is the whole decoder state for those positions. Hybrid models with
convolution or recurrent blocks therefore must not use it.

CausalLM enables prefix caching with:

```sh
NNTR_ENGINE=cpu NNTR_KV_PAGED=1 NNTR_KV_PREFIX_CACHE=1 <application> <arguments>
```

`NNTR_KV_PREFIX_CACHE` without paged CPU storage is rejected. The default
bound is two contexts of blocks (`2 * ceil(max_seq_len / page_tokens)`), which
leaves room for about one context of unreferenced blocks next to a full live
session; `NNTR_KV_PREFIX_CACHE_BLOCKS` overrides it (0 removes the block
bound; the pool has no budget in CausalLM). Models with a layer that has no
KV producer, such as LFM2's convolution blocks, keep prefix caching disabled
and report that on stderr.

While `run()` executes, each step records the token ids it feeds. Before the
prompt is prefilled, `run()` calls `reusePrefix()` with the session's known
history up to the continuation position followed by the prompt, minus the
last prompt token (minus the last two with `skip_prefill`, whose final prompt
token is fed by the first decode step). It then prefills only the remaining
tokens. A new conversation that starts with an already seen system prompt
reuses its published pages; regenerating from an earlier position reuses the
matching rows of the same session. The precomputed system-prompt file path
does not reuse, since a restored snapshot has no ids.
`getLastPrefixReusedTokens()` reports the reused prompt tokens of the last
`run()`, and `NNTR_KV_CPU_TRACE=1` prints a `[kv-prefix]` line per request.

`CausalLM::createKVSession()`, `selectKVSession()`, `destroyKVSession()` and
`getActiveKVSession()` expose sessions to applications: each session is a
separate conversation over the same model and graph. Sessions can be created
once the cache is allocated, which happens on the first `run()` or resume,
because the first prompt settles the prefill plane the cache is sized from.
Selecting a session moves the session token position to its committed
length, so the next `run()` continues that conversation;
`setSessionTokenPosition(0)` afterwards starts it over. A selected session
uses the same save, load and resume calls as before.

Unit tests in `unittest_kv_cache_core.cpp` cover independent sessions with
shared graph identities, id tracking through abort, truncation and restore,
publication and adoption across sessions, exact chain matching,
copy-on-write after truncation, eviction under a pool budget and a block
bound, and layout binding. Like the rest of this backend, they were added
without building or running them on macOS.
