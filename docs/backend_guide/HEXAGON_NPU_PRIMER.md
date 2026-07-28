# Hexagon NPU primer — how the cDSP backend works

Written for someone who already understands nntrainer's CPU path (`ComputeOps` →
`float_tensor.cpp` → the `__ggml_*` kernels, a pre-packed Q4_0 `.bin`, no memory
management to speak of) and knows nothing about the DSP, FastRPC, or
ggml-hexagon. Everything is built up from that starting point.

See [`HEXAGON_NPU_OBSERVATION_LOG.md`](HEXAGON_NPU_OBSERVATION_LOG.md) for the
chronological record of what was tried, what broke, and what the numbers did.

---

## 1. The thing you're actually talking to

A Snapdragon 8 Elite is not one processor. It's several, sharing one pool of
DRAM:

```
┌──────────────────────────────────────────────────────────┐
│                        DRAM (LPDDR)                       │
└───▲──────────────────▲──────────────────▲────────────────┘
    │                  │                  │
┌───┴─────┐      ┌─────┴──────┐     ┌─────┴──────┐
│ ARM CPU │      │    GPU     │     │ Hexagon    │
│ Android │      │  (Adreno)  │     │ DSP  "NPU" │
└─────────┘      └────────────┘     └────────────┘
```

The "NPU" is a **Hexagon DSP** — a genuinely separate CPU with its own
instruction set, its own OS (QuRT, a small RTOS), and **its own MMU and virtual
address space**. It is not a coprocessor you poke registers on. It's closer to a
second computer inside the phone that happens to share the RAM chips.

Three parts of it matter:

- **HVX** — a 1024-bit SIMD unit. Think "NEON but 8× wider." Good at elementwise
  work, quantize/dequantize, softmax.
- **HMX** — a systolic matrix-multiply array. This is the actual "tensor core."
  It's what makes matmul fast, and it's the only reason to bother with the NPU.
- **VTCM** — ~8 MB of fast SRAM next to HVX/HMX. This is a *scratchpad*, not a
  cache: software explicitly DMAs data into it. Nothing is fast unless it's
  staged into VTCM first.

**The single most important consequence**, and the root of every memcpy in this
backend:

> A pointer from `malloc()` on the ARM side is **meaningless** to the DSP.
> Different MMU, different page tables, different address space. The DSP cannot
> dereference it. Not "slowly" — *at all*.

On CPU, `weight.getData<uint8_t>()` is a pointer and the kernel just reads it.
That's the entire memory story, which is why it's fair to say nntrainer's CPU
path has "no memory management." On the DSP that sentence has no meaning.
Getting bytes to where the DSP can see them **is** the work.

---

## 2. FastRPC — how you call a function on the DSP

FastRPC is Qualcomm's remote-procedure-call mechanism. The mental model is a
network RPC where the "network" is on-die.

The code is split into two halves **compiled by different compilers into
different binaries**:

| | runs on | compiler | artifact |
|---|---|---|---|
| **stub** | ARM | Android NDK clang | inside `libggml-hexagon.so` |
| **skel** | Hexagon DSP | `hexagon-clang` (Hexagon SDK) | `libggml-htp-v79.so` |

That's why building this needs **two** toolchains, and why `libggml-htp-v79.so`
is a separate file pushed to the device. It contains Hexagon machine code; the
ARM linker can't touch it.

The interface between them is declared in an IDL file (`htp/htp_iface.idl`) —
`open`, `close`, `start`, `stop`, `mmap`, `munmap`. A generator turns that into a
stub (ARM) and a skel (DSP). Calling `htp_iface_open(...)` on ARM marshals
arguments, traps into the FastRPC kernel driver, which wakes the DSP, which
dispatches into the skel.

Session bring-up (`ggml-hexagon.cpp:2318`, `allocate()`) does roughly:

1. `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)` — by default the DSP
   only runs Qualcomm-signed code. This asks for an **unsigned protection
   domain** so it will load *our* skel. On locked-down retail devices this is
   exactly what fails.
2. `htp_iface_open("file:///libggml-htp-v79.so?...")` — spawn the DSP-side
   process and load the skel. **This is the expensive step.**
3. `dspqueue_create(...)` — set up the fast path (section 4).
4. `htp_iface_start(...)` — DSP allocates VTCM, creates its worker-thread pool
   (one per HVX context), sets up DMA queues.

`v79` is the HTP architecture version: 8 Gen 3 = v75, 8 Elite = v79. Wrong
version = wrong instruction set = won't load. That's why `opt_arch` must be
resolved before any session exists.

Note also **`domain-id 3`** in the logs — the DSP has multiple domains (ADSP for
audio, CDSP for compute). 3 is CDSP.

---

## 3. rpcmem — the shared memory, and why it's three steps

FastRPC arguments are fine for small scalars. You cannot pass a 400 MB weight
tensor through an RPC argument. So there's a separate mechanism: **shared memory
both processors can address.**

`ggml_hexagon_shared_buffer` (`ggml-hexagon.cpp:210`) does it in three distinct
steps, and the distinction matters:

```c
base = rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);  // :254
fd   = rpcmem_to_fd(base);                                                // :260
fastrpc_mmap(domain_id, fd, base, 0, size, flags);                         // :221
```

1. **`rpcmem_alloc2`** — allocate from a **dma-buf** heap rather than the normal
   heap. You get an ARM-usable virtual address, but the allocation is
   *shareable*: it has a kernel-level identity, not just a process-local address.
2. **`rpcmem_to_fd`** — get that identity as a **file descriptor**. An fd is the
   kernel's currency for "this specific chunk of physical memory," and it's what
   can be handed to another processor.
3. **`fastrpc_mmap`** — tell the FastRPC driver to map that fd into the **DSP's**
   address space, so the DSP gets its own virtual address for the same physical
   pages.

`RPCMEM_HEAP_ID_SYSTEM` matters: it's the ordinary system heap, not the small
contiguous/CMA heap, which is why hundreds of MB can be mapped. (The CMA
exhaustion warning in `neuralnet.cpp:242-250` is about QNN's path, which uses a
different heap — it does not apply here.)

There is a fourth, implicit step **on the DSP**: `htp/main.c:678-708` calls
`HAP_mmap` the first time it sees a given fd and caches the result in
`ctx->mmap[]` (`reuse_buf`, `main.c:649`). **Keyed on fd.** This is why the
original per-call design was so slow.

### Common misconception

rpcmem is **not** a third place "between" CPU and NPU memory. It is ordinary
DRAM. What's special is that it is *mapped into both* address spaces — same
physical bytes, two different virtual addresses.

It is also **not faster memory.** There is no performance benefit to putting
something in rpcmem. The only reason is that the DSP would otherwise be unable
to touch it at all.

### Pinned vs delayed — what actually gets released

`fastrpc_mmap` takes a flag (`:219`):

- `FASTRPC_MAP_FD` (**pinned**) — establish the DSP mapping now, keep it forever.
- `FASTRPC_MAP_FD_DELAYED` (not pinned) — map lazily on first DSP touch;
  `unmap()` will tell the DSP to drop it.

Pinning concerns **only the DSP's page-table entry**. Three separate resources
are in play:

| resource | created by | released by |
|---|---|---|
| the memory itself | `rpcmem_alloc2` | `rpcmem_free` |
| host-side FastRPC registration | `fastrpc_mmap` | `fastrpc_munmap` |
| **DSP-side page mapping** | `HAP_mmap` on DSP, cached by fd | `htp_iface_munmap` |

`shared_buffer::unmap()` does the last two — **except it skips
`htp_iface_munmap` when pinned** (`:237`). So a pinned buffer's DSP mapping is
never torn down. Free it and allocate again, the kernel hands back the same fd
number, the DSP finds that fd in its cache and reuses a translation pointing at
memory that no longer exists.

**So: pinned means "I will never free this."** Weight arenas qualify. A growable
staging buffer does not. Getting this wrong produces
`buffer mapping failed ... error 0x0000001a` followed by a segfault.

Cache flushing is a **completely separate** mechanism (section 4) and has nothing
to do with pinning.

---

## 4. dspqueue — why there isn't an RPC per matmul

A FastRPC call is a kernel trap plus an inter-processor interrupt — tens of
microseconds. Paying that per matmul would be hopeless.

So the hot path doesn't use FastRPC at all. `dspqueue` is a **lock-free ring
buffer in shared memory**. The host writes a request packet; the DSP is already
spinning on the other end and picks it up.

```
enqueue_op(node)   :2276  → append to a host-side batch. NO DSP interaction.
flush_batch()      :2254  → serialize the whole batch into a shm block,
                             dspqueue_write() once.               :2270
flush_pending()    :2216  → dspqueue_read(), block until response. :2227
flush(all)         :2284  → flush_batch() then flush_pending()
```

`opt_opbatch` defaults to **1024** — up to a thousand ops can be stacked into one
submission. llama.cpp does exactly that: it walks a whole `ggml_cgraph`, enqueues
every offloadable node, then flushes once.

**Our bridge enqueues one op and immediately flushes.** That is the remaining
~196-round-trips-per-token cost, and what "op batching" would fix.

Note the ring buffer is not an alternative to rpcmem — both are always used.
rpcmem carries the **data** (weights, activations); the ring buffer carries the
**commands** ("do MUL_MAT on buffer 0 offset 4096"). Every matmul uses both.

### The wire format

This is where memory and dispatch tie together. From `htp/htp-ops.h`:

```c
struct htp_tensor {
    uint32_t data;   // ← OFFSET from a buffer base, not a pointer!
    uint32_t size;
    uint32_t flags;
    uint16_t type;   // matches ggml_type
    uint16_t bi;     // index into the batch's buffer list
    uint32_t ne[4];  // dimensions
    uint32_t nb[4];  // strides in bytes
};
struct htp_buf_desc { uint64_t base; uint64_t size; uint32_t flags; uint32_t fd; };
struct htp_op_desc  { uint32_t opcode; uint32_t flags; int32_t params[16];
                      uint16_t src[6]; uint16_t dst; };
```

A tensor is **(buffer index, byte offset, shape, strides)** — not an address,
because the host's address and the DSP's address for the same bytes are
*different numbers*. The host sends `t->data - sbuf->base`
(`ggml-hexagon.cpp:1926`) and the DSP does
`t->data = bufs[t->bi].base + t->data` (`main.c:748-757`), rewriting the offset
into its own pointer.

Two hard limits fall out: **`HTP_OP_MAX_BUFS` and `HTP_MAX_MMAPS` are both 16.**
At most 16 distinct buffers per batch, and 16 mappings resident on the DSP. A
28-block Qwen3 has 196 weight tensors, so "one rpcmem buffer per weight" is
structurally impossible — weights must be pooled into a few big arenas.

### Cache coherency

The two processors have separate caches over shared DRAM, so somebody must
flush. It happens **on the DSP, per op** (`main.c:779`):

```c
if (!(src->flags & HTP_TENSOR_FLUSHED) && (src->flags & HTP_TENSOR_COMPUTE))
    hex_l2flush(src->data, src->size);
```

`hex_l2flush` (`htp/hex-utils.h:91`) walks cache lines — *unless* the region
exceeds 128 KB, in which case it escalates to flush-and-invalidate the DSP's
**entire** data cache.

The `HTP_TENSOR_COMPUTE` flag comes from the buffer's `usage` field. Weights
marked `USAGE_COMPUTE` get flushed every op, and any real FC weight is >128 KB,
so that means a **full D-cache nuke, 196× per token**. `USAGE_WEIGHTS` means
"write-once, never flush" and is what weight arenas must use.

---

## 5. What ggml-hexagon is

llama.cpp with one extra backend. Two halves:

**Host side** — `ggml/src/ggml-hexagon/ggml-hexagon.cpp` (~4,100 lines). A normal
ggml backend: registers a device, buffer types, and a `graph_compute`. ggml's
scheduler asks "can you do this op?" (`ggml_hexagon_supported_mul_mat`, `:2656`),
and for claimed ops `graph_compute` enqueues them and flushes.

**DSP side** — `ggml/src/ggml-hexagon/htp/*.c`. The actual kernels:
`matmul-ops.c` (~5,000 lines), `hmx-matmul-ops.c`, `softmax-ops.c`,
`rope-ops.c`, etc. `main.c` is the DSP-side dispatch loop: read a batch, resolve
buffers, `switch` on opcode, run it on the worker pool.

**Two buffer types**, and this is the part that matters for us:

- normal `buffer_type` — for activations. Plain memcpy on `tensor_set`.
- `repack_buffer_type` — for weights. `tensor_set` **transforms the bytes**
  (`:1606-1652`): Q4_0 in → `repack_q4_0_q4x4x2()` → q4x4x2 out.

So in llama.cpp you hand it standard GGUF Q4_0 and the repack happens
automatically at model load. You never see it.

---

## 6. The layout problem — two orthogonal things

There are **two separate problems**, and both need solving:

| problem | question | solution |
|---|---|---|
| **Address space** | can the DSP *see* these bytes? | rpcmem — a **copy** |
| **Data layout** | are the bytes *arranged* how this kernel wants? | repack — a **rearrangement** |

nntrainer already lives with the second one on CPU. The `.bin` is not plain GGUF
Q4_0 — `repack_q4_0(..., ISA::ARM)` at quantize time interleaves 4 rows into
`q4_0x4`, because the NEON kernel `__ggml_q4_0_4x8_q8_0_GEMM` processes 4 rows at
once and wants their nibbles adjacent. x86 uses `q4_0x8` for the same reason with
8 rows. That's why the README warns a model quantized on x86 won't run on ARM.

The DSP wants a **third** arrangement, `q4x4x2`. Per row of K elements:

```
[ all quants: K/2 bytes ][ all scales: (K/256)*16 bytes ]
   ↑ element j paired in one byte with element j+128
```

Total = 9K/16 bytes per row — **exactly the same size** as plain Q4_0. That's
deliberate; it makes the repack size-preserving.

So there are three mutually unreadable layouts of the same weights: `q4_0x4`
(ARM), `q4_0x8` (x86), `q4x4x2` (HTP). Feed one kernel another's bytes and you
get garbage **with no error**.

### Activation quantization

The CPU path quantizes activations to `q8_0` on the fly inside the GEMM. The DSP
does the same thing, just to `q8x4x2`, into VTCM (`matmul-ops.c:4697`, the
`vec_dot_q4x4x2_q8x4x2_*` family). Activations therefore go over as **plain
F32** and the DSP quantizes them itself — no host-side work.

This is also why DSP and CPU results differ by ~0.03: different activation
quantization granularity, not a bug.

---

## 7. The two flows side by side

```
CPU (the familiar path)
───────────────────────
FullyConnectedLayer::forwarding
  input_.dot(weight, hidden_)
    FloatTensor::dot → dotQnK                     float_tensor.cpp:956
      getOps()->gemm_q4_0_fp32(M,N,K, act, K, weight, N, out, N)
        nntrainer::gemm_q4_0
          __ggml_q4_0_4x8_q8_0_GEMM   ← reads weight pointer directly.
                                        quantizes act to q8_0 in a stack buffer.
                                        writes out. Done. One address space.


NPU (cDSP path)
───────────────
FullyConnectedLayer::forwarding          ← IDENTICAL. Layer unchanged.
  input_.dot(weight, hidden_)
    FloatTensor::dot → dotQnK             ← IDENTICAL code path
      if (supports_accel && M >= min_rows)          ← the only branch added
        HexagonComputeOps::gemm_q4_0_accel_fp32(weight, act, out, M,N,K)
          │
          ├─ FIRST TIME THIS WEIGHT IS SEEN:
          │    unpack_q4_0(weight → scratch1)        q4_0x4  → plain q4_0
          │    repack_q4_0_to_htp_q4x4x2(scratch1 → scratch2)  → q4x4x2
          │    nntr_htp_bridge_upload_weight_q4x4x2(key=weight, scratch2)
          │      └─ memcpy into a pinned rpcmem arena   ← COPY #1, once per weight
          │
          └─ EVERY CALL:
               nntr_htp_bridge_gemm_q4_0(key, act, out, M,N,K)
                 memcpy(act → rpcmem staging)          ← COPY #2, per call
                 build 3 htp_tensor descriptors (offsets, not pointers)
                 enqueue_op(MUL_MAT) ; flush()         ← FastRPC round trip
                   └─ DSP: resolve offsets → its own addresses
                           DMA weight tiles DDR → VTCM
                           quantize act → q8x4x2 in VTCM
                           HMX or HVX matmul
                           write result to staging, flush its cache
                 memcpy(staging → out)                 ← COPY #3, per call
```

### The three memcpys

| copy | direction | why it exists | frequency |
|---|---|---|---|
| #1 weight | nntrainer weight pool → rpcmem arena | pool is ordinary heap; DSP can't see it | **once per weight**, at first use |
| #2 activation | nntrainer activation tensor → rpcmem staging | same reason | per GEMM (~1.9 MB prefill, 4 KB decode) |
| #3 output | rpcmem staging → nntrainer output tensor | same reason, reversed | per GEMM |

**Every one exists for exactly one reason: nntrainer's memory is not rpcmem.**
There is no other cause. The zero-copy endgame is to route nntrainer's
`MemoryPool` through an rpcmem `MemAllocator` — then all three vanish.

---

## 8. Why nntrainer is at a structural disadvantage here

Not a criticism of nntrainer's design in general — it's specifically about
offloading to an accelerator that rewards batching.

**llama.cpp has a graph object; nntrainer doesn't.** In llama.cpp a
`ggml_cgraph` is an explicit list of nodes. The backend can inspect *all* of it,
decide which nodes it wants, reorder them (`ggml_hexagon_graph_optimize_reorder`,
`:3351`), enqueue everything, and flush **once** per forward pass.

In nntrainer, *the layers are the graph*. Each `forwarding()` call is one node,
and there is no data structure describing the whole forward pass. `ComputeOps` is
handed one matmul at a time with no knowledge of what comes next. Consequences:

| | llama.cpp | nntrainer |
|---|---|---|
| ops per submission | whole graph (up to 1024) | **1** |
| round trips per token | a handful | **196** |
| can reorder to exploit shared activations | yes | no |
| DSP's `src1_spad` activation cache hits | yes (same batch) | never (one op per batch) |
| sees op fusion opportunities | yes | no |

That last row matters more than it looks: the DSP caches the dynamically
quantized activation and skips re-quantizing when consecutive ops in the *same
batch* share `src1` (`matmul-ops.c:4687-4702`). Q/K/V share an activation, and so
do gate/up. llama.cpp gets that for free. We re-quantize every time.

The partial escape hatch is `gemm_q4_0_batch_fp32` — nntrainer's existing "N
weights, one shared activation" call (`float_tensor.cpp:771`). It's not a graph,
but it's enough to collapse Q/K/V into one submission and gate/up into another,
taking 196 dispatches to ~112.

**The flip side, and the reason this project exists:** QNN takes the whole
architecture and compiles it into an opaque graph blob, so you lose control of
the KV cache, sampling, and layer structure. Here nntrainer keeps all of it — we
swapped exactly one kernel. `FullyConnectedLayer` is byte-identical between CPU
and NPU runs. That was the whole point, and the batching deficit is the price.

---

## 9. Why decode belongs on the CPU

This is the most important practical conclusion, so it's worth stating plainly.

Measured with llama.cpp's own `llama-bench`, Qwen3-0.6B Q4_0, 8 Elite (v79),
4 threads — i.e. the mature reference implementation, not ours:

| test | CPU | NPU (HTP0) | NPU vs CPU |
|---|---|---|---|
| prefill, 90 tok | 721.8 | 1014.6 | **1.41× faster** |
| prefill, 512 tok | 571.1 | **2045.9** | **3.58× faster** |
| decode, 128 tok | **158.9** | 34.6 | **4.6× slower** |

Decode on the DSP is 4.6× slower than CPU *in llama.cpp*, with full graph
batching and weights resident from load. That is structural, not an
implementation gap:

- Decode is GEMV — you stream every weight to produce one output row. It's
  **bandwidth-bound**, and the DSP has no bandwidth advantage over the CPU.
- HMX cannot engage below M=32 (`matmul-ops.c:4762`, `m_hmx = M & ~31`), so
  decode only ever gets HVX — wide SIMD competing with 4 ARM cores.
- On top of that you pay a FastRPC round trip per op.

Prefill is the opposite: HMX scales with M while CPU attention is O(n²), which is
why the reference's advantage *grows* from 1.41× to 3.58× with prompt length.

**Hence the hybrid design.** `gemm_q4_0_accel_min_rows()` routes prefill to the
DSP and leaves decode on the CPU. Note this is not "NPU decode was made as fast
as CPU decode" — the DSP is simply idle during generation.

---

## 10. Where to read the code, in order

1. `ggml-hexagon.cpp:210-298` — `shared_buffer`. 90 lines, and it's the whole
   memory model.
2. `ggml-hexagon.cpp:2216-2290` — `enqueue_op` / `flush_batch` /
   `flush_pending`. The dispatch model.
3. `htp/htp-ops.h:118-145` — the wire structs. Everything clicks once you see
   that `data` is a `uint32_t`.
4. `htp/main.c:710-804` — the DSP receiving side: resolve buffers, flush caches,
   dispatch.
5. `ggml/src/ggml-hexagon/nntr-htp-bridge.cpp` — ours. Read the header comment
   first; it explains the memory model and both traps.
6. `htp/matmul-ops.c:4710-4780` — `op_matmul`'s HMX gate. Where `M >= 32` comes
   from.

nntrainer side:
- `nntrainer/hexagon/hexagon_compute_ops.cpp` — the `ComputeOps` subclass, the
  layout conversion, and `gemm_q4_0_accel_min_rows()` with the measured
  threshold.
- `nntrainer/hexagon/hexagon_context.cpp` — registers the `"cdsp"` context.
- `nntrainer/tensor/float_tensor.cpp:996-1010` — the one branch in `dotQnK`.

---

## 11. Two things to just remember

- **rpcmem is not "faster memory."** It's *reachable* memory. The only reason to
  use it is that the DSP would otherwise be unable to touch the bytes at all.
- **The DSP is not magic silicon that makes matmul fast.** It's fast only when
  HMX engages (M ≥ 32) and data is staged into VTCM. Below that it's a wide-SIMD
  unit competing with 4 ARM cores while paying an RPC round trip — which is
  exactly why decode loses, prefill wins, and the measured crossover was 215
  rows.
