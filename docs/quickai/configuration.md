# Configuration reference

Every runtime setting of the HTP path lives in the model pack's
`nntr_config.json`, in two JSON objects: `htp` (transport, memory, dense
kernels, attention, diagnostics) and `moe` (expert caching, reads,
prediction, expert kernels, rotation). The command line and the QuickDotAI
API read the same file, so a setting made once in the pack applies to both.
No environment variable tunes the HTP runtime.

The schema is `nntrainer/tensor/htp_backend/htp_runtime_config.h`, its
validation `htp_runtime_config.cpp`, and the parser
`Applications/CausalLM/models/htp_model_config.h`. The pack-level keys that
*select* a model and describe its weights (`gemma4_execution_profile`,
`moe_layer_dtype`, `attention_kv_dtype`, …) are not runtime settings; they
are listed in [Model packs](model-packs.md).

## How settings are read

- **Strict parsing.** Unknown keys, wrong JSON types, out-of-range values
  and incompatible combinations are rejected before the DSP session opens,
  with the key named in the error:

  ```text
  Unknown runtime configuration: htp.pagecache
  Invalid runtime configuration: htp.dsp_queue      (for "dsp_queue": 1)
  htp.page_cache must be -1, 0 or 1
  moe.prediction must be auto, off, dsp-page, dsp-direct or arm-direct
  ```

  Booleans are JSON `true` / `false`, never `0` / `1` or strings. Keys
  marked *optional* below accept `null`, which means "let the runtime
  choose". The command line prints the message after `[!] FATAL ERROR:` and
  exits with status 1; the API returns `CAUSAL_LM_ERROR_MODEL_LOAD_FAILED`
  and the message is available from `getLastModelLoadError()`.
- **Fixed while loaded.** The configuration is resolved when the model
  loads and cannot change while the DSP session is open (reconfiguring with
  different values fails with `HTP configuration cannot change while a
  session is open`). Edit the pack and reload the model to change a value.
- **One runtime per process.** There is one HTP configuration per process.
  A CPU model loaded in the same process while an HTP model is active
  inherits the `htp.page_cache` and `htp.load_prefetch` file-reading
  policies, because those feed the shared weight loader.
- **Pack metadata only.** `htp` or `moe` objects in a pack that is not an
  HTP model are an error (`htp/moe runtime configuration requires an HTP
  model`).
- **Relative paths.** `htp.static_linear` and `moe.tl2_prepared_file` are
  resolved relative to the pack directory. Diagnostic output paths
  (`htp.token_trace`, `htp.dump_dir`, `moe.trace`, `moe.route_log`,
  `moe.predict_log`) are used as written.

A minimal pack needs neither object: all defaults are built in. This is the
measurement policy used for the numbers in [Performance](performance.md):

```json
{
  "htp": {"page_cache": 0, "token_trace": "/data/local/tmp/htp/trace.json"},
  "moe": {"cache_experts": 16, "cache_grow": 0, "preload_experts": 128}
}
```

## The settings most people touch

| Key | Default | Why you would set it |
|---|---|---|
| `htp.page_cache` | `-1` | `0` reads weights with direct I/O and bounds the process footprint at some decode speed; `-1` keeps the loader's default, which lets the kernel page cache hold the file; `1` uses cached reads with a bounded per-file cache. See [Performance, Memory](performance.md#memory). |
| `moe.cache_experts` | `16` | Experts kept resident per MoE layer. More experts, fewer flash reads per token, more memory. |
| `moe.preload_experts` | `128` | How many experts per layer are read at start, bounded by `cache_experts` growth room. |
| `moe.cache_grow` | `null` | `null` lets the pool grow after the first token towards 34 experts per layer as memory allows; `0` fixes the pool at `cache_experts` (use this for repeatable measurements); a positive number adds that many slots. |
| `moe.prediction` | `"auto"` | Expert prefetch: `auto` picks `arm-direct` for softmax routers (Gemma4) and `off` otherwise; `off` disables prediction; `dsp-page` needs `htp.page_cache` `-1`. |
| `moe.hadamard`, `moe.hadamard_scope` | `"off"`, `"2bit"` | Online rotation for pre-rotated checkpoints. Must match the weights; see [Model packs](model-packs.md#rotated-checkpoints). |
| `htp.token_trace` | `""` | Path of a JSON file with the input and output token IDs of each generation. The way runs are compared for correctness. |
| `htp.release_prefill_scratch` | `true` | Releases the CPU side's prefill scratch when decode starts. Leave on. |
| `htp.attn_f16`, `htp.attn_q2` | `true`, `true` | `false` moves attention back to the CPU reference paths. Diagnosis only; much slower. |
| `htp.static_linear` | `""` | Overrides the pack's `static_activation_file` with another static activation encoding file. |
| `htp.profile` | `0` | `1` prints host-side stage timings, `2` also times DSP entries (changes the execution path and may change output IDs), `3` repeats timed calls five times. Not for throughput numbers. |

Everything else is a tuning or diagnostic knob with a working default. The
complete lists follow.

## `htp` keys

Category: **policy** = user-relevant behaviour; **tuning** = performance
knob most users leave alone; **diagnostic** = tracing, dumps and reference
checks; **LFM2** = only affects the LFM2 model or its layer shapes.

| Key | Type | Default | Allowed | Effect | Category |
|---|---|---|---|---|---|
| `poll_us` | unsigned | `5000` | any | Microseconds a DSP call polls before blocking on an interrupt (FastRPC QoS). | tuning |
| `profile` | unsigned | `0` | 0..3 | Per-stage timing: 1 host stages, 2 also DSP-timed entries (outputs forced to the host), 3 repeats timed calls five times. | diagnostic |
| `dma_trace_calls` | unsigned | `3` | ≤ INT_MAX | Under profiling, timed calls per bucket whose DMA trace is dumped. | diagnostic |
| `dsp_queue` | bool | `true` | | `false` keeps the single-token MoE and decode calls on FastRPC instead of the dspqueue. | tuning |
| `dsp_queue_spin_us` | unsigned | `1000` | any | Spin time of the per-layer dspqueue before it sleeps. | tuning |
| `e2e_spin_us` | unsigned | `0` | any | DSP spin time at miss waits and between tokens on the resident decode queue. | tuning |
| `page_cache` | int | `-1` | -1, 0, 1 | Weight-file I/O: -1 loader default, 0 direct I/O (`O_DIRECT`), 1 cached reads with a bounded per-file cache. Linux/Android only. | policy |
| `load_prefetch` | optional bool | `null` | | Load-time read-ahead thread on or off; `null` falls back to the environment variable `NNTR_LOAD_PREFETCH` (on unless it is `0`). | tuning |
| `arena_grain_mib` | unsigned | `64` | 4, 16, 64 | Allocation grain of the DSP arena chunks. | tuning |
| `prefill_rows` | unsigned | `512` | any | Row chunk of the conv-block prefill call; 0 sends the whole prefill at once. | LFM2 |
| `release_prefill_scratch` | bool | `true` | | Discards the CPU model's inference scratch pages after prefill. | policy |
| `release_prefill_readers` | bool | `false` | | Joins idle prefill reader threads after prefill; a later prefill recreates them. | tuning |
| `purge_prefill` | bool | `false` | | Android: returns the allocator's free pages to the system at the prefill-to-decode boundary. | tuning |
| `keep_out` | bool | `true` | | `false` copies every HTP output whole to the host instead of keeping it staged (reference path; needed by `moe.host_mask` and `moe.debug`). | diagnostic |
| `keep_arm_weights` | bool | `false` | | Keeps the host copy of WH weights after DSP registration. | diagnostic |
| `stage_debug` | bool | `false` | | Checks that host reads of staged tensors are not stale window copies. | diagnostic |
| `prefill_io_trace` | bool | `false` | | Prefill readers print per-read timings after each batch. | diagnostic |
| `init_trace` | bool | `false` | | Prints `[HTP-MEM]` arena capacity and usage once the readers stop. | diagnostic |
| `token_trace` | string | `""` | path | JSON `{input_ids, output_ids}` written at the end of each generation; empty disables. An unopenable path fails the run. | diagnostic |
| `static_linear` | string | `""` | path | Static activation encoding file for the linear inputs; overrides the pack's `static_activation_file`. Relative to the pack. | policy |
| `dump_dir` | string | `""` | path | Directory for FC and MoE call dumps; also disables retained prefill views. | diagnostic |
| `dump_all` | bool | `false` | needs `dump_dir` | Also dumps every resident-decode forward call. | diagnostic |
| `fc_feed` | string | `"vtcm"` | vtcm, l2 | Feed mode of the Q4 FC ops in the decode graph. | tuning |
| `fc_lanes_small` | optional unsigned | `null` | 1..8 | HVX lanes for small FC ops; `null` keeps the kernel default. Setting only one of the two lane keys sends 6 (small) or 3 (large) for the other. | tuning |
| `fc_lanes_large` | optional unsigned | `null` | 1..8 | HVX lanes for large FC ops. | tuning |
| `native_fc` | bool | `false` | | Native Q4 kernels for every Q4 op; requires Q4_0 weights, fails on QS4CX. | tuning |
| `fc_whole_prefill` | bool | `false` | | Whole prefill activation in one strided FC call instead of row chunks. | tuning |
| `fc_wh_heap` | bool | `false` | | FC WH images on the DSP heap instead of their own arena chunk. | tuning |
| `dense_early` | bool | `true` | | During a miss round the DSP runs the dense branch while the ARM reads experts. | tuning |
| `qkv_to_host` | bool | `false` | | Copies QKV outputs whole to the host (reference path). | diagnostic |
| `oproj_u16` | bool | `true` | | `false` dequantizes the attention context on the host instead of the DSP's u16 input path (reference). | diagnostic |
| `kv_seed_q` | bool | `false` | | Registers decode attention lazily and seeds each layer's decode KV from its quantized prefill cache on the DSP. | tuning |
| `attn_f16` | bool | `true` | | `false` disables the accelerated attention path (CPU attention). | policy |
| `attn_q2` | bool | `true` | | Allows the int8 KV-cache attention kernel. | policy |
| `attn_dsp_calib` | bool | `true` | | Runs the int8 scale calibration on the DSP when possible. | tuning |
| `attn_f32` | bool | `true` | | `false` turns off the DSP FP32 attention run and its calibration. | diagnostic |
| `attn_trace` | bool | `false` | | Logs every attention call with host timings. | diagnostic |
| `attn_f16_check` | bool | `false` | | Compares DSP fp16 attention against the CPU on the same rows. | diagnostic |
| `attn_f32_check` | optional int | `null` | ≥ 0 | FP32 reference check: 1 compares with NEON, 2 reruns the DSP; disables the DSP calibration. | diagnostic |
| `q2_margin` | float | `1.0` | > 0 | Multiplier on the observed K/V maxima when deriving int8 KV scales. | tuning |
| `attn_m1_fast` | bool | `false` | | Integer decode attention over the int8 cache; faster, not bit-identical to the CPU reference. | tuning |
| `l2_check` | bool | `false` | | Checks DSP outputs and row scales for non-finite values. | diagnostic |
| `l2_diff` | bool | `false` | | Runs both MoE FFN paths and reports their SNR. | diagnostic |
| `l2_shadow` | bool | `false` | | Runs the accelerated path, then substitutes the reference output. | diagnostic |
| `lfm2_default_preset` | bool | `true` | | `false` stops the command line from merging the LFM2 SR15 preset (below). | LFM2 |
| `lfm2_router_input_handoff` | bool | `false` | | Router input hand-off ticket for 64-expert routers. | LFM2 |
| `lfm2_prefill_views` | bool | `false` | | CPU consumers read prefill outputs directly from staging. | LFM2 |
| `lfm2_parallel_stage` | bool | `false` | | Parallel staging copy of MoE prefill inputs. | LFM2 |
| `lfm2_attn_ion_handoff` | bool | `false` | | Attention Q input taken directly from an ION staging buffer. | LFM2 |
| `lfm2_conv_retain_out` | bool | `false` | | With `lfm2_prefill_views`, conv layers keep their output on the DSP side. | LFM2 |
| `lfm2_conv_channel_tile` | bool | `false` | | Channel-tiled NEON loop in the CPU conv gate. | LFM2 |
| `lfm2_conv_state_only` | bool | `false` | | State-only variant of the CPU conv gate. | LFM2 |
| `lfm2_conv_stripe_rows` | unsigned | `64` | 8, 16, 32, 64, 128, 256 | Row stripe per job in the CPU conv gate. | LFM2 |
| `lfm2_conv_whole_prefill` | bool | `false` | | Conv in-projection in one call over all rows. | LFM2 |
| `lfm2_parallel_add` | bool | `false` | | Parallel residual add for prefill rows. | LFM2 |
| `lfm2_parallel_norm` | bool | `false` | | Parallel RMSNorm rows for width-2048 prefill. | LFM2 |

## `moe` keys

| Key | Type | Default | Allowed | Effect | Category |
|---|---|---|---|---|---|
| `cache_experts` | unsigned | `16` | ≥ 1 | Resident expert pool capacity per MoE layer. | policy |
| `preload_experts` | unsigned | `128` | any | Cap on experts preloaded per layer at start. | tuning |
| `cache_grow` | optional unsigned | `null` | any | Extra experts per layer the pool grows by after the first token; `null` grows towards 34 as memory allows, `0` disables growth. | policy |
| `lrfu_half_life` | double | `32.0` | ≥ 0 | Expert eviction: > 0 is LRFU with this half-life in tokens, 0 is plain LRU. | tuning |
| `prefill_lookahead` | unsigned | `1048576` | ≤ INT_MAX | How many MoE layers ahead prefill reads experts, bounded by free slots. | tuning |
| `prefill_readers` | unsigned | `6` | ≥ 1 | Prefill expert reader threads. | tuning |
| `worker_cpus` | array of int | `[]` | each 0..31 | CPUs the readers and background workers pin to; empty means every core but the caller's. | tuning |
| `prediction` | string | `"auto"` | auto, off, dsp-page, dsp-direct, arm-direct | Expert prediction and prefetch mode. `dsp-page` requires `htp.page_cache` `-1`. | policy |
| `decode_prefetch_readers` | optional unsigned | `null` | 1..8 | Decode prefetch reader threads; `null` picks 0 with prediction off, 2 for direct TL2 prefetch with `page_cache` 0, else 4. | tuning |
| `prefetch_slots` | unsigned | `32` | 1..32 | Staged expert slots for direct prefetch. | tuning |
| `prefetch_top` | optional unsigned | `null` | 0..16 | How many predicted experts are prefetched; `null` is 16 on the DSP-hint path and 8 on the ARM direct path. | tuning |
| `prefetch_residual` | float | `0.5` | 0..1 | Residual blend of the ARM router predictor. | tuning |
| `prefetch_confidence` | float | `0.0` | 0..1 | > 0 gates predictions by confidence. | tuning |
| `prefetch_join` | bool | `true` | | A miss round joins pending prefetches instead of taking their buffers. | tuning |
| `miss_readers` | unsigned | `4` | ≥ 1 | Readers of a decode miss round. | tuning |
| `demand_pool` | bool | `true` | | Dedicated demand-read pool instead of the compute pool. | tuning |
| `coalesce_read` | bool | `true` | | Coalesces expert reads (effective with `page_cache` 0). | tuning |
| `cancel_stale_read` | bool | `true` | | Cancels stale prefetch reads (effective with `page_cache` 0). | tuning |
| `server_done` | bool | `false` | | Lets the DSP's completion hint park the pool server before dspqueue reception. | tuning |
| `fadvise` | unsigned | `0` | 0..2 | With `page_cache` -1 only: 1 drops an expert's pages once in the arena and asks them back on eviction, 2 drops only. | tuning |
| `pin` | unsigned | `0` | 0..2 | With `page_cache` -1 only: 1 populates and locks each layer's expert range, 2 reads it into anonymous memory. | tuning |
| `pin_floor_mib` | unsigned | `2048` | any | `MemAvailable` floor kept while pinning. | tuning |
| `m1_gemv` | bool | `true` | | Single-token expert GEMV kernel instead of the HMX loop. | tuning |
| `gemv_lead_kb` | unsigned | `192` | 0..8128 | GEMV DMA lead in KiB. | tuning |
| `gemv_rows1` | bool | `true` | | GEMV tuning bit. | tuning |
| `gemv_feed` | bool | `true` | | GEMV tuning bit. | tuning |
| `dma_queues` | unsigned | `4` | 1..4 | Expert kernel DMA queues. | tuning |
| `dma_bypass` | bool | `true` | | Expert kernel DMA bypass. | tuning |
| `hadamard` | string | `"off"` | off, up, down, both, both_scalar, up_hmx, down_hmx, both_hmx, both_paley | Online rotation of expert activations. Must match the checkpoint. | policy |
| `hadamard_scope` | string | `"2bit"` | 2bit, all | `2bit` rotates only 2-bit expert containers, `all` rotates every routed expert layer. | policy |
| `host_mask` | unsigned | `0` | 0..7 | Moves the MoE pre-norm (1), post-norm (2) and epilogue (4) to the host (reference; needs `htp.keep_out` false). | diagnostic |
| `debug` | bool | `false` | | Compares staged activations against the host bytes on every MoE call. | diagnostic |
| `profile` | bool | `false` | | Host-side prefill MoE layer profile. | diagnostic |
| `descriptor_trace` | bool | `false` | | Prints descriptor lines for the first two experts of each layer. | diagnostic |
| `trace` | string | `""` | path | Routing trace file. | diagnostic |
| `route_log` | string | `""` | path | Per-token routed expert sets of the resident decode, one line per MoE op. | diagnostic |
| `predict_log` | string | `""` | path | ARM predictor trace file. | diagnostic |
| `router_u8i8` | bool | `false` | | Router logits on the HMX (u8 × i8) instead of FP32. | tuning |
| `tl2_prepared_file` | string | `""` | path; needs `page_cache` -1 | Prepared TL2 artifact for 2-bit experts (resident decode only). Relative to the pack. | policy |
| `tl2_decode` | bool | `true` | | `false` keeps the prefill representation for decode instead of native TL2. | tuning |
| `tl2_direct_read` | bool | `true` | | Prepared TL2 reads scatter directly into their destinations. | tuning |
| `tl2_stable_slots` | bool | `false` | | Stable TL2 slots for the LFM2 expert shape with `page_cache` 0. | LFM2 |
| `lfm2_router_arm` | bool | `false` | | Router logits on the ARM for TL2 experts; off when `router_u8i8`. | LFM2 |
| `lfm2_prefill_early` | bool | `false` | | Queues expert prefill reads before the first prefill block. | LFM2 |
| `lfm2_prefill_prune` | bool | `false` | | Prunes prefill expert batches to the routed keys. | LFM2 |
| `lfm2_compact_recency` | bool | `false` | | Compact LRU recency refresh after prefill. | LFM2 |
| `lfm2_decode_pair_read` | bool | `false` | | Reads both weights of an expert pair natively (LFM2 shape, `page_cache` 0). | LFM2 |
| `lfm2_decode_demand_handoff` | bool | `false` | | A miss round claims the prefetched pair with hand-off (LFM2 shape, `page_cache` 0). | LFM2 |
| `lfm2_prefetch_gamma` | bool | `false` | | Gamma-normalized ARM prediction for sigmoid routers. | LFM2 |

## Rules between keys

Rejected at load:

- `htp.dump_all` requires `htp.dump_dir`.
- `moe.tl2_prepared_file` and `moe.prediction` `"dsp-page"` require
  `htp.page_cache` `-1`.

Silently gated at run time:

- `moe.fadvise` and `moe.pin` do nothing unless `htp.page_cache` is `-1`.
- `moe.coalesce_read`, `moe.cancel_stale_read`, `moe.tl2_stable_slots`,
  `moe.lfm2_decode_pair_read` and `moe.lfm2_decode_demand_handoff` act only
  with `htp.page_cache` `0`.
- `htp.profile` 2 or 3 moves the MoE norms and epilogue to the host
  (`moe.host_mask` 7) and copies outputs to the host regardless of
  `htp.keep_out`. Its token IDs can differ from an ordinary run.
- `htp.attn_f32_check` disables `htp.attn_dsp_calib`.
- `htp.dump_dir`, `htp.l2_diff`, `htp.l2_shadow` and `htp.keep_out` `false`
  disable retained prefill views.
- `moe.router_u8i8` disables `moe.lfm2_router_arm`.

## The LFM2 CLI preset

The command-line tool, and only it, merges a measured performance preset
into a pack whose `model_type` is `lfm2_moe`, `lfm2_weight_format` is
`sr15_v2` and `lfm2_execution_profile` is `htp_fused`. It prints
`[LFM2] SR15 HTP CLI defaults active; explicit configuration values take
precedence`. A key that the pack sets explicitly, including `false` or `0`,
wins over the preset. `"htp": {"lfm2_default_preset": false}` turns the
preset off. The QuickDotAI API never applies it, so an app that wants the
same behaviour must write these values into the pack:

| Object | Keys the preset sets |
|---|---|
| `htp` | `page_cache` 0, `load_prefetch` false, `fc_whole_prefill` true, `lfm2_parallel_norm` true, `lfm2_parallel_add` true, `lfm2_conv_whole_prefill` true, `lfm2_conv_channel_tile` true, `lfm2_prefill_views` true, `lfm2_conv_state_only` true, `lfm2_parallel_stage` true, `lfm2_conv_retain_out` true, `lfm2_conv_stripe_rows` 8, `lfm2_attn_ion_handoff` true, `arena_grain_mib` 4, `purge_prefill` true, `release_prefill_readers` true |
| `moe` | `cache_grow` 1, `preload_experts` 128, `tl2_stable_slots` true, `lfm2_prefill_early` true, `lfm2_decode_pair_read` true, `lfm2_decode_demand_handoff` true, `prediction` "arm-direct", `prefetch_residual` 0.5, `lfm2_prefill_prune` false, `lfm2_compact_recency` true, `prefetch_slots` 6, `lfm2_prefetch_gamma` true, `prefetch_top` 3 |
| top level | `num_threads` 8 when absent |

The preset does not activate for Gemma4 packs.

## Threads

The top-level `num_threads` key of `nntr_config.json` sets the CPU compute
thread count for the command-line tool. The environment variable
`NNTR_NUM_THREADS`, when set, takes precedence over it. The API does not
read `num_threads`; its thread count comes from `NNTR_NUM_THREADS` or the
build default.

## Environment variables that still matter

| Variable | Role |
|---|---|
| `ADSP_LIBRARY_PATH` | Directory that holds `libnntr_hvx_skel.so`; the command line must set it, the API sets it from `native_lib_dir`. |
| `NNTR_LOAD_PREFETCH` | Fallback for `htp.load_prefetch` when that key is `null`; `0` turns the load read-ahead off. |
| `NNTR_NUM_THREADS` | CPU compute threads, overrides `num_threads`. |
| `NNTR_GENERATED_IDS` | Command line: path to write the generated token IDs. |
| `NNTR_INIT_TRACE` | Command line: load-phase timing trace. |
| `NNTR_HTP_DIAGNOSTICS` | A **build** option of the DSP skel that compiles in the diagnostic probes. Not a runtime setting. |

The old `NNTR_HTP_*`, `NNTR_MOE_*` and `NNTR_LFM2_*` runtime variables are
not read. Their replacements:

| Former variable | Key |
|---|---|
| `NNTR_HTP_PAGE_CACHE` | `htp.page_cache` |
| `NNTR_HTP_PROFILE` | `htp.profile` |
| `NNTR_HTP_PREDICT` (with or without decode prefetch) | `moe.prediction` |
| `NNTR_MOE_CACHE_GROW` | `moe.cache_grow` |
| `NNTR_MOE_SERVER_DONE` | `moe.server_done` |
| `NNTR_MOE_PREFETCH` | `moe.prefill_lookahead` (prefill batches) and `moe.decode_prefetch_readers` (decode readers) |
| `NNTR_MOE_PREFETCH_READERS` | `moe.prefill_readers` |

## Legacy flat keys

Older packs carry `moe_cache_experts`, `moe_hadamard` and
`moe_hadamard_scope` at the top level of `nntr_config.json`. They are still
accepted:

- `moe_cache_experts` `0` means 16. If `moe.cache_experts` is also present
  with a different value, loading fails with `Conflicting moe.cache_experts
  settings`.
- `moe_hadamard` / `moe_hadamard_scope` win over a nested `null`; a nested
  value that differs fails with `Conflicting moe.hadamard settings` or
  `Conflicting moe.hadamard_scope settings`.
- After parsing, the resolved values are written back to the flat keys,
  which the model loaders read. Prefer the nested form for new packs and
  remove the flat key when you set a different nested value.
