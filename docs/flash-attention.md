# Flash Attention in `mha_core`

This document covers the GEMM-based flash attention path that lives inside
`mha_core.cpp` / `mha_core.h`. It started as a V-JEPA encoder-only
optimization and now also covers the **causal LLM prefill** path with **GQA**
and **sliding window** support, and dispatches to a **fused AVX2 + F16C
kernel** on x86.

The flash path runs only for **prefill** (multi-token attention). Single-token
decode uses the existing per-row dot kernels — flash there is pure overhead.

---

## Configuration

`nntr_config.json`:

```json
{
  ...
  "use_flash_attention": true
}
```

- Default: `true` if the field is absent. Flash is on by default for prefill.
- `false` forces the reference per-row dot path for both prefill and decode.
- Wired through `Transformer::createAttention` → `mha_core` property
  `use_gemm_attention`. Per-model overrides (e.g. `Qwen3Transformer`) honor
  the same config field.
- Decode (`step_size == 1`) never enters flash regardless of this setting,
  because of the internal gate `step_size >= FLASH_MIN_PREFILL (=32)`.

---

## Algorithm

Two phases per `MHACoreLayer::gemm_attention` call. All heads share the input
buffers; work is parallelized via `ThreadManager::parallel_for`.

### Phase 1 — de-interleave heads

Q/K/V cache rows in nntrainer are stored with heads interleaved per row
(stride `num_heads_* × head_dim`). Per head we copy out a contiguous
`[N, head_dim]` view so the GEMM kernel can stream contiguous memory.

| Tensor | Heads      | Phase-1 buffer dtype                |
|--------|------------|-------------------------------------|
| Q      | num_heads_Q  | FP32                                  |
| K, V   | num_heads_KV | uint16 (FP16 bits) on x86 / FP32 elsewhere |

On **x86** we skip the FP32 conversion of K/V — the fused hsgemm kernel
reads FP16 bits directly using F16C. That saves ~14 MB of staging memory
and one full memory pass over K/V (per call).

### Phase 2 — flash attention with online softmax

Balanced parallel_for over `(h_q, qb)` work units, where `qb` is a query
block of `Bq` (default 256) rows and `h_q` ranges over all query heads.

For each work unit, iterate key blocks `kb` in steps of `Bk` (default 512):

1. **GQA pointer math**: `h_kv = h_q / gqa_size`. Q reads from `Qa[h_q]`,
   K/V read from `Ka/Va[h_kv]`.

2. **Causal upper-bound block-skip** (if `is_causal`): if
   `kb > cache_from + qb + bq - 1`, the smallest key index in this block is
   already past the largest query position in the current query-block.
   `break` the key loop — this and every later key block contribute 0 FLOPs.

3. **Sliding window lower-bound block-skip** (if windowed): if
   `kb + bk + W <= cache_from + qb + 1`, the largest key index in this
   block is still below the window. `continue`.

4. **QK GEMM**: compute `S = α · Q[qb..qb+bq] × K[kb..kb+bk]^T` as FP32.
   On x86: `mha_hsgemm_avx2(...)`. Other platforms: pre-converted Phase-1
   buffer + `nntrainer::sgemm`.

5. **In-place mask** for the boundary cases:
   - **Causal boundary**: if `kb + bk > cache_from + qb + 1`, some keys in
     this block are above the diagonal for some rows. Per row `i`, set
     `S[i, k] = -INFINITY` for `k > cache_from + qb + i - kb`.
   - **Window boundary**: similar, lower-bound masking
     `S[i, k] = -INFINITY` for `k <= cache_from + qb + i - W - kb`.

6. **Online softmax** (one pass): block max → running max update
   `nm = max(mrow[i], bm)`; rescale `Ol *= exp(mrow[i] - nm)`; exp into S
   in place (`s[k] := exp(s[k] - nm)`); accumulate `lrow[i]`.
   x86 uses scalar `std::exp` in this row loop; ARM uses a 4-wide NEON
   Cephes exp (`exp_ps` from `neon_mathfun.h`).

7. **AV GEMM**: `Pacc = S × V[kb..kb+bk]`. x86 → `mha_hsgemm_avx2`. Other
   platforms → `nntrainer::sgemm`. Accumulate `Ol += Pacc`.

After all key blocks, divide `Ol` by `lrow` and scatter to the output
tensor at the interleaved head stride.

### What flash skips vs computes

| Block position vs causal diagonal | Cost |
|---|---|
| Above diagonal (`kb > q_abs_hi`)            | **0 FLOPs** — block-skip + `break` |
| Boundary (diagonal cuts through the block)  | Full block QK + AV computed, in-place `-INFINITY` mask on the masked entries |
| Below diagonal                              | Full compute (all entries valid)  |

Boundary blocks have bounded waste — at most one per query-block per head.
Removing that waste would require per-row GEMM dispatch, which loses more
than the boundary-mask waste costs.

---

## x86 fused hsgemm (`mha_hsgemm_avx2`)

Defined in `mha_core.cpp`, used only by the flash path. Two access patterns:

- **`TransB = true`** (QK): `C[m, n] = α · Σ_k A[m,k] · fp16(B[n,k])`. The
  inner loop loads 8 floats from A and 8 fp16 bits from B and FMAs into 4
  parallel accumulators (4-row block over `m`) to amortize the F16-to-F32
  conversion across 4 rows.

- **`TransB = false`** (AV): `C[m, n] = α · Σ_k A[m,k] · fp16(B[k,n])`. Inner
  loop broadcasts a single A scalar and loads 8 contiguous fp16 bits from
  B per `k` step, FMAing into an 8-lane accumulator. 8-wide N blocking.

Both paths use `_mm256_cvtph_ps` (F16C) — available on every x86_64 CPU
since Ivy Bridge (2011). `-march=native -mavx2 -mfma` are already passed
project-wide.

On **ARM** the QK GEMM uses `hgemm_f16xf16_f32_fmlal`
(`arm/hgemm/hgemm.cpp`) when the query is FP16 — the default `q_fp16` path
(native `Q4_0-FP16`, or a `Q4_0-FP32` model wrap-converted to FP16 under
`ENABLE_FP16=1`). It widens FP16×FP16 products into FP32 via FMLAL
(`vfmlalq_low/high_f16`) and is 4×2 register-blocked (see Tile-size tuning).
When the query is FP32 (`ENABLE_FP16=0`), QK falls back to `nntrainer::shgemm`
(internally scopy(FP16→FP32)+sgemm). The AV GEMM always uses
`nntrainer::neon::hgemm_f16xf16_f16` (the packed 8×16 FP16 micro-kernel inside
`hgemm()`).

---

## Build

The standard build picks up the changes automatically:

```bash
# x86
meson setup build -Denable-app=true -Denable-test=false \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false \
  -Denable-transformer=true
ninja -C build Applications/CausalLM/nntr_causallm

# Android (ARM)
cd Applications/CausalLM
export ANDROID_NDK=/path/to/ndk
./build_android.sh
```

Notes:
- `Applications/meson.build` may need apps without local deps (DeepQ
  jsoncpp, YOLO opencv, LLaMA/PicoGPT) commented out in dev environments.
- Android.mk currently builds with `ENABLE_FP16=0`; flash still works
  because the flash path reads cache as raw `uint16_t` bits and does not
  depend on `_FP16` typedefs.

---

## Tile-size tuning

`Bq` and `Bk` are compile-time constants in `gemm_attention()`, currently
**32 / 128**, tuned for the Cortex-A76 L1 (64 KB): the FP32 score buffer
(`Bq·Bk·4` = 16 KB) plus the FP16 `Sp16`/`Ol`/`Pacc` tiles stay L1-resident
(~36 KB total). The earlier 256 / 512 (tuned for V-JEPA on S26 Ultra) blew
out L1+L2 for the shorter causal-LLM head_dim shapes.

The QK micro-kernel `hgemm_f16xf16_f32_fmlal` is **4×2 register-blocked**:
each k-step loads 4 A-rows and 2 B-rows once and reuses them across the
tile's 8 outputs (cutting B reloads 4× and A reloads 2× vs the prior naive
per-(m,n) triple-loop). Per-output K-accumulation order is kept identical to
the single-output path, so the result is **bit-identical** to the unblocked
kernel (verified by byte-identical generated tokens). Measured on S25-class
Qwen3-0.6B `Q4_0-FP16` (8 threads, ~686-tok prefill): per-call QK
**55.2 µs → 30.8 µs (~1.8×)**, attention wall (summed/`ATTN_TOTAL`)
**470 ms → 356 ms (−24 %)**.

---

## Benchmarks

### V-JEPA 2.1 ViT-B 16-frame (encoder, non-causal, N = 4608)

| Build | Path | e2e | RAM | cos vs FP32 reference |
|---|---|---|---|---|
| x86 FP32 | reference (no flash)       | 27.0 s | ~1.7 GB | 1.000 (baseline) |
| x86 FP32 | naive flash (pre-convert + sgemm) | 72–77 s | 1.06 GB | 1.000 |
| **x86 FP32** | **fused hsgemm flash**        | **13.9 s** | 1.06 GB | **1.000** |
| **x86 Q4_0** | fused hsgemm flash            | **13.1 s** | 528 MB  | 0.991 (matches historical Q4 result) |

The fused hsgemm flash is **~2× faster than the reference path** at this
sequence length (and ~5× faster than the earlier non-fused flash).

### Qwen3-0.6B (causal, GQA = 2, head_dim = 128, layers = 28)

#### Short prefill (36 tokens) — flash overhead is tiny

| Build | Path | Prefill | Generation | RAM |
|---|---|---|---|---|
| x86 Q4 | flash off | 1551 ms (23 TPS) | 32 tok / 10620 ms | 0.92 GB |
| x86 Q4 | flash on  | 128 ms (281 TPS) | 32 tok / 441 ms   | 0.92 GB |

#### 1 K prefill (1003 tokens) — production-relevant case

| Device | Path | Prefill | TPS | Generation | RAM | Speedup |
|---|---|---|---|---|---|---|
| **S25 Ultra (SM-S938N)** | flash off | 2967 ms | 338 | 32 tok / 586 ms (54.6 TPS) | 965 MB | – |
| **S25 Ultra (SM-S938N)** | **flash on** | **1491 ms** | **672** | 32 tok / 583 ms (54.9 TPS) | 964 MB | **2.0×** |
| **S26 Ultra (SM-S948U)** | flash off | 3047 ms | 329 | 32 tok / 622 ms (51 TPS)   | 965 MB | – |
| **S26 Ultra (SM-S948U)** | **flash on** | **1768 ms** | **567** | 32 tok / 627 ms (51 TPS)   | 961 MB | **1.7×** |
| x86 (Ryzen + AVX2)       | flash off | 5994 ms | 167 | 32 tok / 777 ms (41 TPS)   | 924 MB | – |
| x86 (Ryzen + AVX2)       | flash on  | 6242 ms | 160 | 32 tok / 761 ms (42 TPS)   | 924 MB | 0.96× |

**Notes on the 1 K results**

- ARM device sees the expected ~2× prefill speedup. Decode is identical
  (decode does not enter flash).
- x86 1 K causal sees no win because the reference path's
  `compute_kcaches<uint16_t>` is already tight (AVX2 inline dot + only
  triangular work for causal), while our flash still computes full
  boundary blocks and then masks. Production gain on x86 is largest at
  very short prefill (5×) and at long non-causal sequences (V-JEPA 2× faster).
- RAM is essentially unchanged on/off — the model weights (~375 MB) and
  KV cache (~235 MB) dominate; flash's transient ~22 MB matches the
  reference's ~33 MB triangular attention buffer in magnitude.

#### Output equivalence (1 K prefill, greedy decode)

Same 32 generated tokens flash-on vs flash-off on each platform:
> *"Okay, so the user wants me to explain the differences between
>   supervised, unsupervised, and reinforcement learning, and list five
>   practical applications. Let me…"*

Bit-level diff is in the noise of FP32 reordering — first ~30 tokens
identical, then small numerical drift can flip a low-probability tail
token on some runs (expected behavior).

---

## Files touched

- `Applications/CausalLM/layers/mha_core.cpp` — flash core + fused hsgemm
- `Applications/CausalLM/layers/mha_core.h`   — `use_gemm_attention` property,
  `gemm_attention(query, K, V, out, N_kv, N_q, cache_from)` signature
- `Applications/CausalLM/models/transformer.h`   — `USE_FLASH_ATTENTION` field
- `Applications/CausalLM/models/transformer.cpp` — read
  `nntr_cfg["use_flash_attention"]`, pass to mha_core property
- `Applications/CausalLM/models/qwen3/qwen3_causallm.cpp` — same wiring in
  Qwen3-specific `createAttention`

---

## Known limitations

1. **x86 1 K causal is ~equal to reference** (see benchmarks above). The
   boundary block does full QK + AV before masking; a future custom GEMM
   that strips the upper triangle from the kernel itself would close
   this gap. Not blocking — production target is ARM device, and short
   prefill on x86 already wins big.

2. **No AVX-512 path.** `mha_hsgemm_avx2` is AVX2 + F16C only. Adding
   AVX-512 (16-wide, F16C → wider conversion) would help ML-tuned x86.

3. **PR 3933 (Gemma4) is not yet compatible** with the current upstream
   API. The PR was authored before commit `9159ec1c CausalLM: migrate to
   ml::train::Tensor symbolic graph API` and still uses the legacy
   `vector<LayerHandle> createAttention(... std::string)` + `void
   constructModel()` pattern. Cherry-picking the PR cleanly puts the
   gemma4 model in a half-ported state where `load_weight()` fails with
   `getRunContext layer needs to be configured first!`. Gemma4 needs to
   be rebased onto the symbolic Tensor graph API by its author before
   the flash path can be benchmarked there. Our flash code is
   model-agnostic and will work as-is once gemma4 builds.

4. **Decode is unchanged.** Single-token decode keeps the existing
   per-row dot kernels. The flash path is structurally a poor fit for
   decode (one query row vs. tile size 256), and the reference path is
   already efficient there.

---

## Precision matrix (ARM device)

Three precision configurations are supported, controlled by the
`ENABLE_FP16` build flag (Android.mk) and (implicitly) the model's
`model_tensor_type`:

All paths now share one structure: **QK → FP32 score buffer `S` → online
softmax (writes FP16 `Sp16`) → AV**. Only the QK kernel differs by build:

| Build / model | Q at gemm_attention | QK kernel | Score buffer | Softmax | AV kernel |
|---|---|---|---|---|---|
| `ENABLE_FP16=0`, `Q4_0-FP32` | FP32 | `shgemm` (FP32 Q × FP16 K) | FP32 | NEON exp → `Sp16` (FP16) | `hgemm_f16xf16_f16` |
| `ENABLE_FP16=1` (`Q4_0-FP16` native, or `Q4_0-FP32` wrap-converted) | FP16 | `hgemm_f16xf16_f32_fmlal` (FP16×FP16→FP32, 4×2 blocked) | FP32 | NEON exp → `Sp16` (FP16) | `hgemm_f16xf16_f16` |

K/V are FP16 storage in both cases; QK accumulates in FP32 (no FP16-product
overflow), and AV (`hgemm_f16xf16_f16`) accumulates FP16-chunked → FP32. The
default ARM build is `ENABLE_FP16=1`, matching the FP16 storage + FP32
partial-accumulation precision the NPU target uses. (Earlier docs described
an all-FP16 `hgemm_f16xf16_f16` QK with an FP16 score buffer and a Phase-1 Q
FP16→FP32 + `shgemm` variant — both superseded by the unified fmlal QK path
above.)

## Attention region profile (S25-class, Qwen3-0.6B `Q4_0-FP16`, ~686-tok prefill)

Per-region wall-clock summed across 8 worker threads (`PROFILE` build,
`std::chrono::steady_clock`; `ATTN_TOTAL` is the single-thread wall of the
whole `gemm_attention` call). At `Bq=32, Bk=128` the QK shape is
M=32, K=128, N=128; AV is M=32, K=128, N=128.

| Region | naive QK | **4×2 blocked QK** |
|---|---|---|
| ATTN_QK (`hgemm_f16xf16_f32_fmlal`) | 1781 ms (55.2 µs/call) | **995 ms (30.8 µs/call)** |
| ATTN_AV (`hgemm_f16xf16_f16`) | 846 ms | 846 ms (unchanged) |
| ATTN_SOFTMAX | 619 ms | 577 ms |
| **ATTN_TOTAL (attention wall)** | 470 ms | **356 ms** |

Register-blocking the QK kernel cut it **~1.8×** and the attention wall
**−24 %** (≈8 % off prefill, since attention is ~30 % of prefill wall while
the FC Q4_0 GEMM — already SMMLA/i8mm — dominates the rest). Output stays
bit-identical (per-output K-accumulation order preserved).

> Historical note: an earlier profile here compared `shgemm` (613 µs/call)
> vs `hgemm_f16xf16_f16` (854 µs/call) for QK at the old `Bq=256/Bk=512` tiles
> and concluded the in-tree FP16 GEMM had register-blocking headroom. That
> headroom is what the 4×2 blocking above realized; the old `shgemm`-vs-
> `hgemm_f16xf16_f16` QK comparison no longer reflects the code (QK is now the
> blocked fmlal kernel). Attempts to also speed up AV — a thread_local C32
> scratch (no measured gain) and a dedicated FP32-accumulating AV kernel
> (~2.3× *slower*: the packed 8×16 FP16 `hgemm_f16xf16_f16` already uses
> `vfmaq_laneq_f16` at full FP16 throughput) — were both reverted.

## Qwen3-0.6B benchmarks — S25 Ultra (historical, pre-register-blocking)

> These e2e numbers predate the 4×2 QK register-blocking and the unified
> fmlal QK path. The "Phase-1 Q FP16→FP32 + `shgemm`" and "all-FP16
> `hgemm_f16xf16_f16`" QK variant rows no longer exist in the code (QK is now
> `hgemm_f16xf16_f32_fmlal`, blocked). Kept for historical context; the
> current per-region QK/attention numbers are in "Attention region profile"
> above.

(1003-token prefill + 32-token greedy decode, `NNTR_NUM_THREADS=8`,
`Q4_0-FP32` activation; outputs match the reference path token-for-token.)

| Build      | Path                                  | Prefill (TPS)   | Decode (TPS)    | e2e      |
|---|---|---|---|---|
| `ENABLE_FP16=0` | reference (no flash)              | 2 967 ms (338)  | 586 ms (54.6)   | 3 580 ms |
| `ENABLE_FP16=0` | flash, V-JEPA `shgemm` path       | **1 458 ms (688)** | 583 ms (54.9) | **2 070 ms** |
| `ENABLE_FP16=1` | reference (no flash)              | 1 752 ms (572)  | 484 ms (66.1)   | 2 259 ms |
| `ENABLE_FP16=1` | flash, Phase-1 Q FP16→FP32 + `shgemm` | 1 535 ms (653) | 586 ms (54.6)\* | 2 149 ms |
| `ENABLE_FP16=1` | flash, all-FP16 (`hgemm_f16xf16_f16`)  | 1 656 ms (606)  | 486 ms (65.8)   | 2 166 ms |

The fastest configuration overall is **`ENABLE_FP16=0` + flash**
(2 070 ms e2e, 688 TPS prefill): `shgemm` keeps QK in FP32 scores,
and the FP32 reference decode path on this build still beats the
forwarding() wrap-conversion overhead the `ENABLE_FP16=1` build pays
per layer. That setting is preserved for benchmarking but the
production target is `ENABLE_FP16=1` because it matches the FP16
storage + FP32 partial-accumulation precision the NPU runs at.

\* The Q FP16→FP32 cvt variant happens to take a non-fused decode path
that runs slower than the all-FP16 decode; per-call decode kernel choice
matters more than prefill kernel choice for total e2e.

The all-FP16 path is currently slower than the FP16→FP32+shgemm variant
purely because of the `hgemm_f16xf16_f16` vs `shgemm` kernel gap measured
above; matching the reference precision (FP16 storage throughout) costs
~120 ms / 1 K prefill but is what NPU deployment wants.

## Failed experiment — model-wide FP16 activation (`Q4_0-FP16`)

We tried switching `model_tensor_type` from `Q4_0-FP32` to `Q4_0-FP16`
to eliminate the per-layer FP32↔FP16 wrap-conversion inside
`mha_core::forwarding()`. This requires:

1. Lifting the embedding layer's hard FP32 input check (done in
   commit `0ff355c2`).
2. Keeping the token-ID input itself FP32 — vocab ≈ 150 k cannot fit in
   FP16's effective integer range (2048), so any ID > 2048 rounds into
   garbage and trips the embedding bounds check.
3. Every downstream layer's FP16 codepath being correct end-to-end.

We made (1) and (2) work locally, but the model still segfaulted at
runtime in some FP16 codepath we could not pinpoint without root-level
symbol resolution on the device. Searching the repository, **no
CausalLM model in `res/` declares a `-FP16` activation
`model_tensor_type`**, so this path appears to be unmaintained for
the CausalLM stack. Tracked as future work; the embedding-side check
removal is preserved upstream-friendly even though the rest of the
chain still needs work.
