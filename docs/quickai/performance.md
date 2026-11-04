# Performance

Measured throughput of the two HTP model packs on the Galaxy S26 Ultra
(Snapdragon SM8850, Hexagon v81). Every row names its pack, prompt, settings
and date, because the numbers depend on all of them. Use them to recognise
a healthy setup and to compare against your own runs, not as a guarantee:
the phones were not thermally controlled and single runs vary by 10 to 20 %.

## How the numbers are defined

- **Prefill tok/s** is input tokens divided by the prefill time that
  `nntr_causallm` prints and the app's Metrics screen shows. It excludes
  process start, model load and the first-token latency of building the
  resident decode graph.
- **Decode tok/s** is generated tokens divided by the generation time of the
  same timers. "N decode steps" means N tokens after the one that prefill
  selects.
- **Peak memory** is given with its definition in each row; the sources
  differ (process RSS plus DSP heap, or ARM smaps RSS plus unique DMA-buffer
  sizes).
- **Correctness** in every row below means that the generated token IDs
  were identical across the repeated runs of that row. It does not mean a
  perplexity or benchmark score; none has been measured on this path.

## Gemma4 26B-A4B

All rows: greedy decoding, batch one, `attention_kv_dtype` `q8`, 16 resident
experts per layer (`moe.cache_experts` 16), eight CPU threads.

| Pack | Prompt | Prefill tok/s | Decode tok/s | Memory | Settings and date |
|---|---|---|---|---|---|
| QS2CX_WH experts, QS4CX dense, Q4_0 tied head (7,540,724,856 bytes) | summary prompt, 1,028 in, 109 out, stops at end of turn | 567 (1,812 ms) | 12.3 | 4.2 GB (host anonymous + DSP heap) | CLI, first token after 1.86 s; 2026-10-13 |
| same pack | 1,024 in, 128 out, EOS disabled | 552.9 and 558.6 (medians of 3 alternating runs each) | 13.5 and 14.1 | - | CLI, two builds compared, same IDs; 2026-10-13 |
| same pack, later runtime | summary prompt, 1,028 in, 99 out | 630 (1,633 ms) | 13.8 | - | CLI; 2026-10-18 |
| same pack, later runtime | 2,048 in | 645 | 9.9 | 8.4 GB peak, up to 3.5 GB of it the model's page cache (`htp.page_cache` -1) | CLI, default loader; 2026-10-18 |
| same pack, later runtime | 2,048 in | 682 | 7.0 | 3.6 GB peak | CLI, direct file I/O (`htp.page_cache` 0); 2026-10-18 |
| native TL2CX_WH experts, QS4CX_WH_AFFINE dense and untied head, unrotated | 1,024 in, 64 decode steps | 621.7 to 632.1 (medians of 3) | 12.91 to 12.99 | - | CLI, `htp.page_cache` 0, `moe.preload_experts` 128, `moe.cache_grow` 0; 65 IDs identical across six runs; 2026-10-19 |
| same native pack | 2,048 in, 101 decode steps | 666 to 670 | 13.19 (completion hint off), 13.66 (`moe.server_done` true) | active decode about 3.1 GiB (ARM smaps RSS + unique DMA buffers, median of samples 6 to 9 s into the run) | CLI, `htp.page_cache` 0, preload 128, no cache growth; 102 IDs identical; 2026-10-16 |
| rotated ternary pack, all 30 expert layers TL2CX_WH | 2,048 in, 128 out | 613 | 15.5 | 3,669 MB peak, 8 arena chunks (1,792 MiB) | CLI, back-to-back with the row below; 2026-10-19 |
| rotated ternary pack, layers 28 and 29 as QS4CX_WH_AFFINE | 2,048 in, 128 out | 577 | 11.8 | 4,723 MB peak, 12 arena chunks (2,624 MiB) | CLI; 2026-10-19 |
| rotated ternary pack | summary prompt, 1,028 in, 88 out | - | 18 to 19 | - | QuickDotAIAPP, seven identical runs of the lifecycle test; 2026-11-03 |
| QS2/Q4 pack through the C API | 19 in, 117 out | first token after 1.6 s on the first call (includes building the resident graph) | 16.1 to 16.6 | - | `quick_dot_ai_test`, repeat call reproduces the same output; 2026-10-13 |

Prompt length matters more than anything else for prefill time. The same
QS2/Q4 pack on the same phone model measured:

| Prompt | Prefill time | Decode tok/s over the last 64 tokens |
|---|---|---|
| 1,024 tokens | 1,650 to 2,281 ms | 12.1 to 14.9 |
| 2,048 tokens | 3,141 to 3,436 ms | 11.2 to 15.3 |
| 4,096 tokens | 7,068 to 7,518 ms | 11.9 to 13.3 |

In the app, the bundled summary prompt takes 11 to 18 s from tap to the
end-of-turn marker depending on how warm the phone is; a short question
takes about 7 s.

## LFM2 24B-A2B (SR15 pack)

All rows: SM-S948U, greedy decoding, batch one, 16 resident experts per
layer, the automatic SR15 CLI preset (see
[Configuration](configuration.md#the-lfm2-cli-preset)), `init_seq_len`
2048, `max_seq_len` 4096.

| Prompt | Prefill tok/s | Decode tok/s | Settings and date |
|---|---|---|---|
| 2,048 in, 127 decode steps, EOS suppressed for the fixed length | 1,157.72 (median of 3: 1,159.03 / 1,151.21 / 1,157.72) | 17.39 (17.43 / 17.28 / 17.39) | CLI with the automatic preset, no other settings; all three runs produced the same 128 IDs; 2026-10-19 |
| same prompt, paired comparison | 1,147.98 (prefill optimisation only) and 1,153.15 (combined runtime) | 16.30 and 17.40 | three alternating runs each, identical IDs; 2026-10-19 |
| 1,024 in, different prompt | 903.0 (`moe.preload_experts` 384) and 798.1 (preload 128) | 17.85 and 17.74 | earlier runtime; not comparable with the 2,048-token rows; 2026-10-18 |

The LFM2 preset was confirmed not to activate for Gemma4 packs in the same
runs.

## Memory

- Direct file I/O (`htp.page_cache` 0) trades decode speed for a bounded
  footprint: on the QS2/Q4 pack at 2,048 tokens, 3.6 GB instead of 8.4 GB
  peak and 7.0 instead of 9.9 decode tok/s.
- All-TL2 expert storage uses less arena than a mixed pack: 8 arena chunks
  instead of 12 at 2,048 tokens for the rotated checkpoint, and was faster
  in the same back-to-back run.
- Releasing prefill staging before decode saved about 49 MiB of
  active-decode memory on the native pack at 16 experts per layer; the
  compact TL2 slot layout saved 128 MiB of DMA allocation at 24 experts per
  layer. These are separate experiments and must not be added together.
- Peak memory grows with the prompt: the 2,048-token rows above are 2 to 4
  GB higher than the 1,028-token rows with the default loader.

## How to reproduce a row

1. Build and stage the binaries as described in
   [Build from source](build.md) and run the pack from the command line as in
   [Command line](cli.md). The CLI prints the prefill and generation timings.
2. Put the measurement settings into the pack's `nntr_config.json`
   (`htp` and `moe` objects), for example:

   ```json
   {
     "htp": {"page_cache": 0, "token_trace": "/data/local/tmp/htp/trace.json"},
     "moe": {"cache_experts": 16, "cache_grow": 0, "preload_experts": 128}
   }
   ```

   `htp.token_trace` writes the input and output token IDs of each run, which
   is how the "identical IDs" statements above were checked.
3. For a fixed-length benchmark, disable the end-of-turn stop in the pack's
   configuration (`bad_word_ids` with the EOS id, `num_to_generate` for the
   length). Keep the interactive preset for anything a person reads.
4. Alternate the two configurations you compare (A/B, B/A, A/B), wait five
   seconds between launches, and start each run with the phone below about
   34 °C at the application processor and 36.5 °C at the skin sensor.
   Report medians and ranges.
5. In the app, the Metrics screen shows the same prefill and generation
   timers; the Power tab repeats a preset prompt and records energy per
   token from the battery gauge together with the timings.

Raw records of some of the rows (per-run JSON, logs, the long-context KV
snapshot run on a Galaxy S25 Ultra) are kept under
[`docs/validation/`](../validation/).

## What these numbers do not show

- No perplexity or task-accuracy score has been measured on the HTP path.
  "Identical IDs" compares runs of this runtime with each other and with its
  CPU reference, not with an external implementation.
- Sustained throughput under thermal throttling was not measured; all rows
  are short runs on a phone that had cooled down.
- Model load time is not in the tables. Loading the 7.5 GB pack and
  building the resident graph takes on the order of ten seconds in the app
  before the first prompt.
- Only the Galaxy S26 Ultra (Hexagon v81) was measured with the current
  build. Other v81 devices should behave similarly; v79 devices are not
  supported by this build.
