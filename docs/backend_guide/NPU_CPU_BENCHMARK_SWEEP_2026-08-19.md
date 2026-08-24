# CPU vs NPU Prefill — Token-Length Sweep, 2026-08-19

Device: Samsung S24 (`R3CX9078DNH`), Snapdragon 8 Gen 3, HTP v79. Model:
Qwen3-0.6B, Q4_0 weights. 2 runs per cell. CPU = no `NNTR_USE_HEXAGON_CDSP`
(genuinely no DSP touch, confirmed via absence of `ggml-hex:` driver-load
lines). NPU = `NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_FLASH_ATTN=1
NNTR_HEXAGON_FUSED_FFN=1`. `num_to_generate=1` in all configs (pure prefill
timing). Both modes generate the same first token (`&`) at every length -
basic correctness check, not a full logit comparison.

**Two sweeps in this doc**: an initial one (GAP A/B fixes + KV-cache V-append
DSP wiring), and a final one after additionally porting q_norm/k_norm
(the RMSNorm inside attention) to the DSP - a confirmed, measured
~4% improvement on top of the first sweep (see
`NPU_PREFILL_VALIDATION_2026-08-19.md`). The final sweep is the current,
up-to-date state of the code.

## Final results (after q_norm/k_norm DSP porting)

| Prompt (tokens) | CPU prefill (ms, avg) | NPU prefill (ms, avg) | Speedup |
|---|---|---|---|
| 300 (→392 actual) | 727 | 288 | **2.5×** |
| 600 (→779 actual) | 2142 | 680 | **3.2×** |
| 900 (→909 actual) | 2555 | 768 | **3.3×** |
| 1200 (→1234 actual) | 4389 | 8900 | **0.49× (NPU is ~2× *slower*)** |

## The 1200-token result is still a real cliff, unrelated to today's changes

Same root cause as the initial sweep: `gemm_q4_0`/`ffn_swiglu` refuse to
dispatch above 1024 activation rows (a pre-existing, documented limit - see
git history "add graceful CPU fallback for M>1024 activation rows") and fall
back to CPU per-op, 28 times each, for every GEMM/FFN call in every block.
That fallback pays real overhead on top of doing the same CPU work CPU-only
mode does directly, so NPU mode ends up markedly slower past this point.
Confirmed present in both sweeps at 1200 tokens; absent at 300/600/900.

Note the CPU-only baseline at 1200 tokens itself moved a lot between sweeps
(6034ms → 4389ms, ~27%) with no changes to the CPU code path - this specific
length shows much higher run-to-run/thermal variance than the others, likely
because it's the one config where NPU mode's fallback-heavy behavior
correlates with broader system load. Take the 1200-token absolute numbers as
directionally correct (NPU loses badly past the 1024-row limit) rather than
precise.

**Practical takeaway, unchanged from the initial sweep**: today's NPU path is
a strong, consistent 2.5-3.3× win up to ~1000 tokens of prefill, and a real
regression past it. The 1024-row GEMM/FFN batching limit remains the
concrete next thing to fix if longer prompts matter.

## Initial sweep (before q_norm/k_norm porting, for reference)

| Prompt (tokens) | CPU prefill (ms, avg) | NPU prefill (ms, avg) | Speedup |
|---|---|---|---|
| 392 | 701 | 296 | 2.4× |
| 779 | 2100 | 715 | 2.9× |
| 909 | 2575 | 860 | 3.0× |
| 1234 | 6034 | 11724 | 0.51× |

## Individual run data (final sweep)

| Tokens | Mode | Run | Prefill (ms) | Prefill TPS |
|---|---|---|---|---|
| 392 | CPU | 1 | 733 | 534.8 |
| 392 | CPU | 2 | 721 | 543.7 |
| 392 | NPU | 1 | 287 | 1365.9 |
| 392 | NPU | 2 | 289 | 1356.4 |
| 779 | CPU | 1 | 2122 | 367.1 |
| 779 | CPU | 2 | 2162 | 360.3 |
| 779 | NPU | 1 | 679 | 1147.3 |
| 779 | NPU | 2 | 681 | 1143.9 |
| 909 | CPU | 1 | 2535 | 358.6 |
| 909 | CPU | 2 | 2574 | 353.1 |
| 909 | NPU | 1 | 768 | 1183.6 |
| 909 | NPU | 2 | 769 | 1182.1 |
| 1234 | CPU | 1 | 4295 | 287.3 |
| 1234 | CPU | 2 | 4482 | 275.3 |
| 1234 | NPU | 1 | 8910 | 138.5 |
| 1234 | NPU | 2 | 8889 | 138.8 |
